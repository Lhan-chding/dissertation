"""CPU credit and transactional pilot regressions; no model-quality claim."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sr_f3_ce.training import fresh_optimizer
from sr_f4 import c_pilot_training as pilot
from sr_f4.common import read, write
from sr_f4.credit_design_replay import proposed_residual
from srf4_ref.sampling import seed_for


def credit_inputs(*, j=None, p=None, s=None, **normalization):
    j = j or [0] * 8
    p = p or ["PASS"] * 4 + ["FAIL"] * 4
    s = s or ["PASS", "FAIL"] * 4
    records, labels = [], []
    for i in range(8):
        records.append(dict(row_id=f"row-{i}", group_id="g", qid="q", score={"J": j[i]}))
        labels.append(dict(
            row_id=f"row-{i}", verifier_version=pilot.VERIFIER_VERSION,
            normalization=dict(j=j[i], protocol_valid=True, finished=True, defined=True,
                               sufficient=False, **normalization),
            dimensions=dict(P=dict(status=p[i]), S=dict(status=s[i])),
        ))
    return records, labels


def credit(records, labels, arm="SELECT", **kwargs):
    return pilot.group_credit(
        dict(qid="q", world_id="w", cell="I", family="INTERVAL"), records, labels,
        arm=arm, block=kwargs.get("block", 0), step=kwargs.get("step", 1), group_index=0,
    )


def test_fprime_exact_math_preserves_zero_j_groups_and_ignores_sufficiency():
    records, labels = credit_inputs()
    result = credit(records, labels)
    assert result["active"] and result["signal"] and result["native_J_all_zero"]
    assert result["selection_signal"] and result["zero_native_J_selection_signal"]
    assert result["native_l1"] == 0
    assert result["select_minus_evid_l1"] == .375
    assert result["native_advantages"] == [0] * 8
    # Scores [1,.5,1,.5,0,0,0,0], known mean .375, beta=.25.
    expected = [.15625, .03125, .15625, .03125, -.09375, -.09375, -.09375, -.09375]
    assert result["residual"] == expected == result["final_advantages"]
    assert sum(result["residual"]) == 0
    assert credit(records, labels, "J")["final_advantages"] == [0] * 8
    assert credit(records, labels, "EVID")["residual"] == [.0625] * 4 + [-.0625] * 4


def test_known_subset_mean_is_over_six_known_and_unknown_receives_zero():
    records, labels = credit_inputs(p=["PASS", "PASS", "FAIL", "FAIL", "FAIL", "FAIL",
                                      "UNKNOWN", "UNKNOWN"])
    result = credit(records, labels)
    assert result["candidate_indices"] == list(range(8))
    assert result["known_indices"] == list(range(6))
    assert result["active"]
    assert result["residual"] == [.1875, .0625, -.0625, -.0625, -.0625, -.0625, 0, 0]
    labels[5]["dimensions"]["S"]["status"] = "UNKNOWN"
    result = credit(records, labels)
    assert result["eligibility_reason"] == "KNOWN_COVERAGE_BELOW_75_PERCENT"
    assert result["residual"] == [0] * 8


def test_successes_and_invalid_failures_are_untouched_and_native_j_is_exact():
    records, labels = credit_inputs(j=[1, 1, 0, 0, 0, 0, 0, 0])
    labels[2]["normalization"]["finished"] = False
    labels[3]["normalization"]["protocol_valid"] = False
    labels[4]["normalization"]["defined"] = False
    result = credit(records, labels)
    assert result["candidate_indices"] == [5, 6, 7]
    assert result["residual"][:5] == [0] * 5
    assert result["final_advantages"][:5] == result["native_advantages"][:5]
    assert result["native_advantages"] == pilot.native_advantages(
        [r["score"]["J"] for r in records])
    original = copy.deepcopy((records, labels))
    credit(records, labels, "SELECT_SHUF")
    assert (records, labels) == original
    labels[0]["normalization"]["j"] = 0
    with pytest.raises(ValueError, match="native binary J"):
        credit(records, labels)


def test_single_shuffle_receipt_reconstructs_exact_proposed_residual_and_shared_seed():
    records, labels = credit_inputs()
    shuf = credit(records, labels, "SELECT_SHUF")
    select = credit(records, labels, "SELECT")
    assert select["shuffle"]["seed"] == shuf["shuffle"]["seed"]
    observed = pilot._observations(records, labels)
    assert shuf["residual"] == proposed_residual(
        observed, "SELECT_SHUF", pilot.POLICY, shuffle_seed=shuf["shuffle"]["seed"])
    perm = shuf["shuffle"]["permutation_source_indices"]
    assert sorted(perm) == list(range(8))
    for destination, source in enumerate(perm):
        assert observed[destination].perception == observed[source].perception
    scores = [.5 * (r.perception == "PASS")
              * (1 + (shuf["shuffle"]["selection_after"][i] == "PASS"))
              for i, r in enumerate(observed)]
    assert shuf["residual"] == [.25 * (v - sum(scores) / 8) for v in scores]
    # A homogeneous S stratum legitimately produces unchanged labels, once.
    records, labels = credit_inputs(s=["PASS"] * 8)
    unchanged = credit(records, labels, "SELECT_SHUF")
    assert unchanged["shuffle"]["applied"] and unchanged["shuffle"]["changed_count"] == 0
    assert unchanged["signal"] and not unchanged["selection_signal"]
    assert unchanged["select_minus_shuffle_l1"] == 0


class PilotToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([.25]))

    def save_pretrained(self, path, *, safe_serialization):
        assert safe_serialization
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_text(str(float(self.theta.detach())))


def test_run_resume_commits_once_and_reuses_same_checkpoint_rollouts(tmp_path, monkeypatch):
    from sr_f4 import c_pilot, run, runtime

    generated, updates, runtimes = [], [], []
    task = dict(qid="q", world_id="w", cell="I", family="INTERVAL", split=pilot.data.TRAIN_SPLIT,
                world=dict(categories=["January", "February"], domain=[0, 100],
                           series=dict(Alpha=[10, 20], Beta=[1, 2])),
                query=dict(family="INTERVAL", start=0, end=1, include_start=True,
                           include_end=True, aggregate="sum", value="diff"))
    frozen = dict(code_revision="frozen-code", plan=dict(steps=16, beta=.25))
    monkeypatch.setattr(c_pilot, "require_frozen", lambda root: frozen)
    monkeypatch.setattr(pilot, "task_index", lambda root: {"q": task})
    monkeypatch.setattr(pilot, "image_index", lambda root: {"q": {}})
    monkeypatch.setattr(run, "input_for", lambda *args: {})

    def assignments(tasks, block, step):
        return [dict(qid="q", group_index=i, G=8, protocol="structured",
                     seeds=[seed_for(pilot.data.MASTER, block, step, i, d) for d in range(8)])
                for i in range(16)]

    monkeypatch.setattr(pilot.data, "training_assignments", assignments)
    write(tmp_path / "DATA/TRAIN_PLAN_b0.json",
          {str(step): assignments([task], 0, step) for step in range(1, 17)})

    def load(root, block, task_id, *, training):
        assert training
        model = PilotToyModel()
        current = SimpleNamespace(model=model, identity=dict(run_id=task_id, step=0),
                                  torch=SimpleNamespace(cuda=SimpleNamespace(
                                      reset_peak_memory_stats=lambda: None,
                                      max_memory_allocated=lambda: 0,
                                      max_memory_reserved=lambda: 0)))

        def generate(model_input, root, seeds, *, protocol, on_completion):
            generated.append((current.identity["step"], tuple(seeds)))
            result = []
            for seed in seeds:
                row = dict(seed=seed, protocol=protocol, finish_reason="stop", tokens=[1],
                           sampler_logprobs=[-.5], model_identity=dict(current.identity),
                           full_text=json.dumps(dict(evidence=[], work="Selected: January",
                                                     answer=999)))
                on_completion(row)
                result.append(row)
            return result

        current.generate_group = generate
        runtimes.append(current)
        return current, *fresh_optimizer(model)

    monkeypatch.setattr(runtime, "load", load)

    def update(current, optimizer, scheduler, records, sizes, reference, *, final_advantages):
        assert len(records) == len(final_advantages) == 128 and sizes == [8] * 16
        assert reference["theta"].item() == .25  # parent, including after checkpoint recovery
        updates.append(current.identity["step"] + 1)
        optimizer.zero_grad()
        current.model.theta.sum().backward()
        optimizer.step()
        scheduler.step()
        return dict(final_advantages=final_advantages, optimizer_updates=1)

    monkeypatch.setattr(pilot, "update", update)
    real_save, failures = pilot.save_checkpoint, []

    def save(path, *args, **kwargs):
        if kwargs["step"] == 2 and not failures:
            failures.append(True)
            raise RuntimeError("crash after update before commit")
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(pilot, "save_checkpoint", save)
    spec = dict(kind="c_pilot_train", task_id="pilot-b0-SELECT", block=0, arm="SELECT")
    with pytest.raises(RuntimeError, match="before commit"):
        pilot.run_train(tmp_path, spec)
    directory = tmp_path / "RUNS/C/b0/SELECT"
    assert read(directory / "PROGRESS.json")["successful_updates"] == 1
    assert len(generated) == 32  # step2 cached despite the missing update commit
    result = pilot.run_train(tmp_path, spec)
    assert result["successful_updates"] == 16 and result["answers"] == 2048
    assert len(generated) == 16 * 16  # resumed step2 did not regenerate
    assert updates == [1, 2, *range(2, 17)]
    assert read(directory / "PROGRESS.json")["successful_updates"] == 16
    assert (directory / "checkpoints/step-016.pt").exists()
    assert read(directory / "adapters/step-016/PUBLISHED.json")["status"] == "COMPLETE"
    assert len(list((directory / "credit").glob("*.json"))) == 16
    before = len(generated), len(updates)
    pilot.run_train(tmp_path, spec)
    assert (len(generated), len(updates)) == before
    frozen["code_revision"] = "different"
    with pytest.raises(ValueError, match="Registered pilot record differs"):
        pilot.run_train(tmp_path, spec)


def test_publication_repairs_export_after_committed_checkpoint(tmp_path):
    class Exporter:
        calls = 0

        def save_pretrained(self, path, *, safe_serialization):
            self.calls += 1
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_model.safetensors").write_text(str(self.calls))
            if self.calls == 1:
                raise RuntimeError("interrupted export")

    latest = tmp_path / "checkpoints/latest.pt"
    latest.parent.mkdir()
    latest.write_bytes(b"committed checkpoint")
    runtime = SimpleNamespace(model=Exporter())
    with pytest.raises(RuntimeError, match="interrupted export"):
        pilot._publish(runtime, tmp_path, latest, 16, {"task_id": "test"})
    pilot._publish(runtime, tmp_path, latest, 16, {"task_id": "test"})
    pilot._publish(runtime, tmp_path, latest, 16, {"task_id": "test"})
    assert runtime.model.calls == 2
    assert read(tmp_path / "PROGRESS.json")["target_updates"] == 16
    assert (tmp_path / "checkpoints/step-016.pt").read_bytes() == latest.read_bytes()


def test_registered_sidecar_identity_cannot_be_silently_rewritten(tmp_path):
    destination = tmp_path / "sidecar.json"
    write(destination, {"qid": "first"})
    with pytest.raises(ValueError, match="differs"):
        pilot._registered_write(destination, {"qid": "second"})
