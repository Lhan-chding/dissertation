"""CPU-only local-credit update and durable-recovery regressions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sr_f3_ce.training import fresh_optimizer
from sr_f4 import c_local_training as trainer
from sr_f4.common import read, write
from srf4_ref.sampling import seed_for


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([-.5, -.5, -.5, -.5]))

    def save_pretrained(self, path, *, safe_serialization):
        assert safe_serialization
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_text(str(self.theta.detach().tolist()))


def update_inputs(*, auxiliary=True):
    records, credits = [], []
    for group_index in range(16):
        group_id = f"g{group_index}"
        rows = [dict(row_id=f"{group_id}/r{i}", group_id=group_id, qid=f"q{group_index}",
                     score={"J": 1 if i % 2 else 0}, tokens=[1, 2, 3, 4],
                     sampler_logprobs=[-.5] * 4, model_input={"qid": f"q{group_index}"},
                     run_root="/test", protocol="structured") for i in range(8)]
        masks = [dict(row_id=row["row_id"], evidence_mask=[True, False, False, False],
                      selection_mask=[False, True, False, False]) for row in rows]
        credits.append(dict(
            group_id=group_id, qid=f"q{group_index}",
            native_advantages=trainer.native_advantages([r["score"]["J"] for r in rows]),
            evidence_advantages=[.125 if auxiliary else 0] * 8,
            selection_advantages=[-.125 if auxiliary else 0] * 8, record_masks=masks,
        ))
        records.extend(rows)
    return records, credits


def make_runtime():
    model = ToyModel()
    prepared = []
    runtime = SimpleNamespace(model=model)

    def prepare_input(model_input, root, *, protocol):
        prepared.append(model_input)
        return dict(model_input)

    runtime.prepare_input = prepare_input
    runtime.batch_reference_forward = lambda records, ref: [ref["theta"].clone()]
    runtime.batch_sequence_forward = lambda records, **kwargs: [model.theta * 1]
    return runtime, prepared


def test_update_uses_separate_local_channels_and_native_probability_gate():
    from mm_core.training import trainable_state

    runtime, prepared = make_runtime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    records, credits = update_inputs()
    before = copy.deepcopy((records, credits))
    receipt = trainer.update(runtime, optimizer, scheduler, records, [8] * 16,
                             trainable_state(runtime.model), credits)
    assert (records, credits) == before
    assert len(prepared) == 16
    assert receipt["optimizer_updates"] == 1 and receipt["sequence_count"] == 128
    assert receipt["token_count"] == 512
    assert receipt["sampler_train_mean_absolute_difference"] == 0
    assert receipt["evidence_mask_tokens"] == receipt["selection_mask_tokens"] == 128
    assert receipt["evidence_applied_rows"] == receipt["selection_applied_rows"] == 128
    assert runtime.model.theta[0] > -.5  # positive evidence credit
    assert runtime.model.theta[1] < -.5  # negative selection credit
    assert torch.equal(runtime.model.theta[2:], torch.tensor([-.5, -.5]))


def test_mapping_abstention_keeps_native_rows_and_suppresses_only_auxiliary():
    from mm_core.training import trainable_state

    records, credits = update_inputs()
    for credit in credits:
        for audit in credit["record_masks"]:
            audit.update(evidence_mask=[False] * 4, selection_mask=[False] * 4,
                         reason="EXACT_GENERATED_TOKEN_MAPPING_UNAVAILABLE")
    runtime, _ = make_runtime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    receipt = trainer.update(runtime, optimizer, scheduler, records, [8] * 16,
                             trainable_state(runtime.model), credits)
    assert receipt["sequence_count"] == 128
    assert receipt["evidence_applied_rows"] == receipt["selection_applied_rows"] == 0
    assert receipt["evidence_policy"] == receipt["selection_policy"] == 0
    assert torch.equal(runtime.model.theta, torch.full((4,), -.5))


def test_native_arm_matches_unchanged_native_update():
    from mm_core.training import trainable_state
    from sr_f4.training import update as native_update

    records, credits = update_inputs(auxiliary=False)
    local, _ = make_runtime()
    native, _ = make_runtime()
    local_optim, local_scheduler = fresh_optimizer(local.model)
    native_optim, native_scheduler = fresh_optimizer(native.model)
    local_receipt = trainer.update(local, local_optim, local_scheduler, records, [8] * 16,
                                   trainable_state(local.model), credits)
    native_receipt = native_update(native, native_optim, native_scheduler, records, [8] * 16,
                                   trainable_state(native.model))
    assert torch.equal(local.model.theta, native.model.theta)
    for field in ("policy", "kl", "loss", "native_advantages", "token_count",
                  "gradient_norm_before_clip", "tis_truncated_fraction"):
        assert local_receipt[field] == native_receipt[field]


def test_probability_mismatch_never_commits_optimizer_or_scheduler():
    from mm_core.training import trainable_state
    from sr_f12.training import ProbabilityMismatch

    records, credits = update_inputs()
    records[0]["sampler_logprobs"] = [-3] * 4
    runtime, _ = make_runtime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    # Make every row mismatch so the mean discrepancy, not chance, triggers.
    for record in records:
        record["sampler_logprobs"] = [-3] * 4
    before = runtime.model.theta.detach().clone(), scheduler.state_dict()
    with pytest.raises(ProbabilityMismatch):
        trainer.update(runtime, optimizer, scheduler, records, [8] * 16,
                       trainable_state(runtime.model), credits)
    assert torch.equal(runtime.model.theta, before[0])
    assert scheduler.state_dict() == before[1]
    assert not optimizer.state


@pytest.mark.parametrize("corruption", ["row", "native", "mask", "overlap", "coefficient"])
def test_credit_identity_and_masks_fail_before_gradient(corruption):
    records, credits = update_inputs()
    if corruption == "row":
        credits[0]["record_masks"][0]["row_id"] = "another-row"
    elif corruption == "native":
        credits[0]["native_advantages"][0] = 9
    elif corruption == "mask":
        credits[0]["record_masks"][0]["selection_mask"] = [True]
    elif corruption == "overlap":
        credits[0]["record_masks"][0]["selection_mask"] = [True, False, False, False]
    else:
        credits[0]["selection_advantages"][0] = float("nan")
    with pytest.raises(ValueError):
        trainer._validate_credits(records, [8] * 16, credits)


def test_resume_reuses_committed_policy_rollouts_and_masks(tmp_path, monkeypatch):
    from sr_f4 import c_local, run, runtime

    generated, updates, failed = [], [], []
    task = dict(qid="q", world_id="w", root_id="root", cell="I", family="INTERVAL",
                split=trainer.data.TRAIN_SPLIT,
                world=dict(categories=["January", "February"], domain=[0, 100],
                           series=dict(Alpha=[10, 20], Beta=[1, 2])),
                query=dict(family="INTERVAL", start=0, end=1, include_start=True,
                           include_end=True, aggregate="sum", value="diff"))
    spec = dict(kind="c_local_train", task_id="CLC-T-b0-SELECT", block=0, arm="SELECT")
    frozen = dict(code_revision="frozen-code", steps=12)
    monkeypatch.setattr(c_local, "require_frozen", lambda root: frozen)
    monkeypatch.setattr(c_local, "task_matrix", lambda: [spec])
    monkeypatch.setattr(trainer, "task_index", lambda root: {"q": task})
    monkeypatch.setattr(trainer, "image_index", lambda root: {"q": {}})
    monkeypatch.setattr(run, "input_for", lambda *args: {})

    def assignments(tasks, block, step):
        return [dict(qid="q", group_index=i, G=8, protocol="structured",
                     seeds=[seed_for(trainer.data.MASTER, block, step, i, d) for d in range(8)])
                for i in range(16)]

    monkeypatch.setattr(trainer.data, "training_assignments", assignments)
    write(tmp_path / "DATA/TRAIN_PLAN_b0.json",
          {str(step): assignments([task], 0, step) for step in range(1, 13)})

    def load(root, block, task_id, *, training):
        assert training
        model = ToyModel()
        current = SimpleNamespace(model=model, identity=dict(run_id=task_id, step=0),
                                  processor=SimpleNamespace(tokenizer=None),
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
                           full_text=json.dumps(dict(evidence=[], work="SELECT: January; CALC: 1",
                                                     answer=999)))
                on_completion(row)
                result.append(row)
            return result

        current.generate_group = generate
        return current, *fresh_optimizer(model)

    monkeypatch.setattr(runtime, "load", load)
    monkeypatch.setattr(trainer.credit_method, "measure",
                        lambda task, row: {"row_id": row["row_id"]})
    monkeypatch.setattr(trainer.credit_method, "token_masks", lambda row, label, tokenizer:
                        dict(evidence_mask=[False], selection_mask=[False], reason="UNKNOWN"))

    def group_credit(task, rows, labels, *, arm, seed):
        assert seed == seed_for(trainer.data.MASTER, "c-local-selection-shuffle", 0,
                               rows[0]["step"], int(rows[0]["group_id"].rsplit("g", 1)[1]), "q")
        return dict(group_id=rows[0]["group_id"], qid=task["qid"], native_advantages=[0] * 8,
                    evidence_advantages=[0] * 8, selection_advantages=[0] * 8,
                    eligibility_reason="UNKNOWN")

    monkeypatch.setattr(trainer.credit_method, "group_credit", group_credit)

    def update(current, optimizer, scheduler, rows, sizes, reference, credits):
        assert len(rows) == 128 and len(credits) == 16 and sizes == [8] * 16
        assert torch.equal(reference["theta"], torch.full((4,), -.5))
        step = current.identity["step"] + 1
        persisted = read(tmp_path / "RUNS/C_LOCAL/b0/SELECT/credit" / f"{step:03d}.json")
        assert persisted["groups"] == credits  # audit durable before any gradient
        updates.append(step)
        optimizer.zero_grad()
        current.model.theta.sum().backward()
        optimizer.step()
        scheduler.step()
        return dict(optimizer_updates=1)

    monkeypatch.setattr(trainer, "update", update)
    real_save = trainer.save_checkpoint

    def save(path, *args, **kwargs):
        if kwargs["step"] == 2 and not failed:
            failed.append(True)
            raise RuntimeError("crash after update before atomic commit")
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    with pytest.raises(RuntimeError, match="before atomic commit"):
        trainer.run_train(tmp_path, spec)
    directory = tmp_path / "RUNS/C_LOCAL/b0/SELECT"
    assert read(directory / "PROGRESS.json")["successful_updates"] == 1
    assert len(generated) == 32
    result = trainer.run_train(tmp_path, spec)
    assert result["successful_updates"] == 12 and result["answers"] == 1536
    assert len(generated) == 12 * 16
    assert updates == [1, 2, *range(2, 13)]
    assert (directory / "checkpoints/step-012.pt").exists()
    assert read(directory / "adapters/step-012/PUBLISHED.json")["status"] == "COMPLETE"
    assert len(list((directory / "credit").glob("*.json"))) == 12
    before = len(generated), len(updates)
    trainer.run_train(tmp_path, spec)
    assert (len(generated), len(updates)) == before
    frozen["code_revision"] = "changed"
    with pytest.raises(ValueError, match="Registered local-credit record differs"):
        trainer.run_train(tmp_path, spec)


def test_export_recovery_publishes_only_final_committed_adapter(tmp_path):
    class Exporter:
        calls = 0

        def save_pretrained(self, path, *, safe_serialization):
            self.calls += 1
            Path(path).mkdir(parents=True, exist_ok=True)
            if self.calls == 1:
                raise RuntimeError("interrupted export")

    latest = tmp_path / "checkpoints/latest.pt"
    latest.parent.mkdir()
    latest.write_bytes(b"committed checkpoint")
    runtime = SimpleNamespace(model=Exporter())
    with pytest.raises(RuntimeError, match="interrupted export"):
        trainer._publish(runtime, tmp_path, latest, 12, {"task_id": "test"})
    trainer._publish(runtime, tmp_path, latest, 12, {"task_id": "test"})
    trainer._publish(runtime, tmp_path, latest, 12, {"task_id": "test"})
    assert runtime.model.calls == 2
    assert read(tmp_path / "PROGRESS.json")["target_updates"] == 12
    assert (tmp_path / "checkpoints/step-012.pt").read_bytes() == latest.read_bytes()
