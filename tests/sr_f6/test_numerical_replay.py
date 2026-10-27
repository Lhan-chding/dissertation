"""Tiny CPU numerical fixtures only; none is SERVER_ACTUAL or GPU certification."""

from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mm_core.training import capture_rng, state_hash, trainable_state
from sr_f6 import numerical_replay as numerical
from sr_f6.common import file_hash


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
        self.unused = torch.nn.Parameter(torch.ones(2))
        self.frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.register_buffer("constant", torch.tensor([7.0]))
        self.branch = torch.nn.Dropout(0.3)


class TinyReplayFixture:
    def __init__(self, root):
        self.model = TinyAdapter()
        self.model.train()
        self.model.branch.eval()  # Preserve mixed per-module modes, not just root.training.
        self.model.adapter.grad = torch.tensor([0.25, 0.5])
        self.model.frozen.grad = torch.tensor([-0.5])
        self.reference = trainable_state(self.model)
        self.identity = dict(
            parent_id="b0",
            reference_parent="b0",
            step=0,
            reference_hash=state_hash(self.reference),
            run_root=str(root),
        )
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        self.eos_ids = [99]
        self.calls, self.reference_calls = [], []
        self.fixture_accounting = []
        self.defect = None
        self.zero_graph = False
        self.prepared = dict(
            inputs={
                "input_ids": torch.tensor([[7, 8, 9]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "mm_token_type_ids": torch.tensor([[0, 0, 0]]),
                "pixel_values": torch.zeros(1, 3),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )
        self.prepared["routing"] = dict(input_tensor_hash=state_hash(self.prepared["inputs"]))

    def prepare_input(self, safe, root, *, protocol):
        assert set(safe) == {"prompt", "image_file", "image_sha256"}
        assert protocol == "structured_v4"
        random.random(), np.random.rand(), torch.rand(())
        if self.defect == "prompt":
            self.prepared["inputs"]["input_ids"][0, 0] = 99
        return self.prepared

    def value(self, row, adapter):
        x = torch.tensor(row["tokens"], dtype=torch.float32) / 100
        if self.zero_graph:
            return torch.full_like(x, -2) + adapter.sum() * 0
        return -2 + x * adapter[0] + x.square() * adapter[1]

    def batch_reference_forward(self, records, reference):
        assert len(records) == 1
        self.reference_calls.append(records[0]["row_id"])
        self.account("t03_reference", records[0])
        if self.defect == "reference" and len(self.reference_calls) == 1:
            reference["adapter"].add_(0.01)
        return [self.value(records[0], reference["adapter"])]

    def batch_sequence_forward(self, records, *, purpose, grad):
        assert len(records) == 1 and purpose == "t03_native" and grad is True
        row = records[0]
        self.calls.append(row["row_id"])
        self.account(purpose, row)
        random.random(), np.random.rand(), torch.rand(())
        if self.defect == "forward" and len(self.calls) == 3:
            raise RuntimeError("fixture forward failure")
        if self.defect == "interrupt" and len(self.calls) == 3:
            raise KeyboardInterrupt("fixture interruption")
        if self.defect == "replace_adapter" and len(self.calls) == 1:
            self.model.adapter = torch.nn.Parameter(self.model.adapter.detach().clone())
        if self.defect == "flags" and len(self.calls) == 1:
            self.model.unused.requires_grad_(False)
            self.model.frozen.requires_grad_(True)
        if self.defect in {"adapter", "base", "buffer"} and len(self.calls) == 1:
            target = {
                "adapter": self.model.adapter,
                "base": self.model.frozen,
                "buffer": self.model.constant,
            }[self.defect]
            with torch.no_grad():
                target.add_(0.01)
        result = self.value(row, self.model.adapter)
        if self.defect == "nonfinite" and len(self.calls) == 2:
            result = result * float("nan")
        if self.defect == "shape" and len(self.calls) == 2:
            result = result[:-1]
        return [result]

    def account(self, purpose, row):
        self.fixture_accounting.append(
            dict(
                task_id="SYNTHETIC_ONLY",
                parent="b0",
                kind="extra_forward_sequences",
                count=1,
                metadata=dict(purpose=purpose, completion_tokens=len(row["tokens"])),
            )
        )


def prepare(tmp_path, *, all_zero_j=False, zero_graph=False, lengths=None, deltas=None):
    runtime = TinyReplayFixture(tmp_path)
    runtime.zero_graph = zero_graph
    records, raw = [], []
    for draw in range(8):
        length = lengths[draw] if lengths else 2 + draw % 3
        tokens = [10 + index % 10 for index in range(length)]
        finish = "length" if length == 1536 else "balanced"
        if draw == 7 and length < 1536:
            tokens[-1], finish = 99, "eos"
        model_input = dict(
            prompt="CPU numerical fixture", image_file="fixture.png", image_sha256="a" * 64
        )
        row = dict(
            row_id=f"fixture/{draw}",
            qid="DEV_CAL_FIXTURE_P0_v0",
            draw=draw,
            model_input=model_input,
            protocol="structured_v4",
            source_sha256="0" * 64,
            policy_hash="1"
            * 64,  # Original annotated census file-inventory identity remains untouched.
            tokens=tokens,
            raw_tokens=tokens[:],
            backend_sequence_tokens=tokens + [0] * (draw % 2),
            backend_padding_count=draw % 2,
            completion_token_count=length,
            prompt_token_count=3,
            native_prompt_inputs={
                k: v.tolist() for k, v in runtime.prepared["inputs"].items() if k != "pixel_values"
            },
            prompt_tensor_hash=runtime.prepared["routing"]["input_tensor_hash"],
            finish_reason=finish,
            generation_status="COMPLETE",
            technical_validation_errors=[],
        )
        values = runtime.value(row, runtime.reference["adapter"]).tolist()
        row["sampler_logprobs"] = [v - (deltas[draw] if deltas else 0) for v in values]
        raw.append(copy.deepcopy(row))
        label = int(not all_zero_j and draw in (1, 3, 5) and finish == "balanced")
        row.update(J=label, V=1, finished=int(finish == "balanced"), S=0, B=None)
        # The raw original reason remains available independently of old scorer normalization.
        if finish == "eos":
            row["finish_reason"] = "length"
        records.append(row)
    return runtime, records, raw, runtime.reference


def snapshot(runtime):
    return dict(
        parameters=state_hash({n: p.detach() for n, p in runtime.model.named_parameters()}),
        grads=state_hash({n: p.grad for n, p in runtime.model.named_parameters()}),
        flags={n: p.requires_grad for n, p in runtime.model.named_parameters()},
        modes=[m.training for m in runtime.model.modules()],
        rng=state_hash(capture_rng()),
        reference=state_hash(runtime.reference),
    )


def run(items, path):
    return numerical.run_synthetic(*items, output_directory=path)


def test_actual_adapter_gradients_native_objective_and_exact_state_restoration(tmp_path):
    items = prepare(tmp_path)
    runtime, records, raw, _ = items
    before = snapshot(runtime)
    saved_rows = copy.deepcopy((records, raw))
    report = run(items, tmp_path / "output")
    assert report["scope"] == report["status"] == "SYNTHETIC_ONLY"
    assert report["numerical_result"] == "PASS"
    assert snapshot(runtime) == before
    assert (records, raw) == saved_rows
    assert len(runtime.calls) == len(runtime.reference_calls) == 8
    assert len(report["probability_vectors"]) == 8
    assert len(report["comparisons"]) == 16
    for comparison in report["comparisons"]:
        assert set(comparison["parameters"]) == {"adapter", "unused"}
        assert comparison["parameters"]["unused"]["native_unused"] is True
        assert comparison["parameters"]["unused"]["f6_unused"] is True
        assert comparison["parameters"]["adapter"]["violations"] == 0
        assert comparison["loss"]["f6"] == pytest.approx(comparison["loss"]["native"], abs=1e-6)
    assert report["assessment"]["probe_nonzero_parameter_rows"] == 8
    assert report["boundaries"][-1]["finish_reason"] == "eos"
    assert report["boundaries"][-1]["eos_positions"] == [len(raw[-1]["tokens"]) - 1]
    assert report["observed_coverage"]["tokens_1536"] == "NOT_OBSERVED"
    assert report["observed_coverage"]["backend_padding"] == "OBSERVED"
    assert report["training_authorized"] is False and report["gate_status"] == "NOT_SIGNED"
    assert report["generated_responses"] == report["optimizer_updates"] == 0
    assert json.loads((tmp_path / "output/RESULT.json").read_bytes()) == report


def test_all_zero_actual_advantages_still_test_positive_probe_without_reward_claim(tmp_path):
    report = run(prepare(tmp_path, all_zero_j=True), tmp_path / "output")
    assert report["numerical_result"] == "PASS"
    assert report["native_advantages"] == [0.0] * 8
    assert report["assessment"]["native_nonzero_parameter_rows"] == 0
    assert report["assessment"]["probe_nonzero_parameter_rows"] == 8
    assert report["observed_coverage"]["actual_native_advantage"] == "ALL_ZERO_OR_NOT_AVAILABLE"
    assert "not reward" in report["observed_coverage"]["positive_probe"]


def test_positive_probe_zero_adapter_gradient_is_inconclusive(tmp_path):
    report = run(prepare(tmp_path, zero_graph=True), tmp_path / "output")
    assert report["status"] == "SYNTHETIC_ONLY"
    assert report["numerical_result"] == "INCONCLUSIVE"
    assert report["assessment"]["probe_nonzero_parameter_rows"] == 0


@pytest.mark.parametrize(
    "defect", ["forward", "nonfinite", "shape", "prompt", "adapter", "reference", "flags"]
)
def test_failure_keeps_actual_trace_and_restores_adapter_rng_modes_flags_grads(tmp_path, defect):
    items = prepare(tmp_path)
    runtime = items[0]
    runtime.defect = defect
    before = snapshot(runtime)
    report = run(items, tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    assert report["status"] == "SYNTHETIC_ONLY"
    assert snapshot(runtime) == before
    assert (tmp_path / "output/STARTED.json").exists()
    assert (tmp_path / "output/RESULT.json").exists()
    assert report["restoration"]["adapter_restored"] is True
    assert report["restoration"]["gradients_restored"] is True


@pytest.mark.parametrize("defect", ["base", "buffer"])
def test_frozen_base_mutation_is_detected_and_poisoned_not_claimed_restored(tmp_path, defect):
    items = prepare(tmp_path)
    runtime = items[0]
    runtime.defect = defect
    before = snapshot(runtime)
    report = run(items, tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    assert runtime._srf6_replay_poisoned is True
    state = report["restoration"]
    assert state["frozen_base_buffer_versions_unchanged"] is False
    assert state["runtime_requires_reload"] is True
    assert state["adapter_restored"] is state["rng_restored"] is state["gradients_restored"] is True
    assert snapshot(runtime)["grads"] == before["grads"]
    assert snapshot(runtime)["modes"] == before["modes"]


def test_keyboard_interrupt_still_restores_and_persists_failure(tmp_path):
    items = prepare(tmp_path)
    items[0].defect = "interrupt"
    before = snapshot(items[0])
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        run(items, tmp_path / "output")
    assert snapshot(items[0]) == before
    report = json.loads((tmp_path / "output/RESULT.json").read_bytes())
    assert report["numerical_result"] == "FAIL"
    assert len(report["probability_vectors"]) == 2


def test_replaced_parameter_cannot_skip_other_finally_restoration(tmp_path):
    items = prepare(tmp_path)
    runtime = items[0]
    original = runtime.model.adapter
    original_grad = original.grad.clone()
    before = snapshot(runtime)
    runtime.defect = "replace_adapter"
    report = run(items, tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    assert runtime._srf6_replay_poisoned is True
    assert report["restoration"]["adapter_restored"] is False
    assert report["restoration"]["parameter_objects_unchanged"] is False
    assert report["restoration"]["rng_restored"] is True
    assert report["restoration"]["module_modes_restored"] is True
    assert snapshot(runtime)["rng"] == before["rng"]
    assert torch.equal(original.grad, original_grad)


def test_backward_failure_preserves_preexisting_gradients(tmp_path):
    items = prepare(tmp_path)
    before = snapshot(items[0])

    def fail(_):
        raise RuntimeError("fixture backward failure")

    hook = items[0].model.adapter.register_hook(fail)
    try:
        report = run(items, tmp_path / "output")
    finally:
        hook.remove()
    assert report["numerical_result"] == "FAIL"
    assert snapshot(items[0]) == before
    assert "backward failure" in report["errors"][0]["error"]


def test_gradient_difference_is_detected_even_when_losses_equal(tmp_path, monkeypatch):
    original = numerical.sequence_loss

    def wrong_gradient(current, *args, **kwargs):
        result = original(current, *args, **kwargs)
        result["loss"] += 0.1 * (current - current.detach()).mean()
        return result

    monkeypatch.setattr(numerical, "sequence_loss", wrong_gradient)
    report = run(prepare(tmp_path), tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    first = report["comparisons"][0]
    assert first["loss"]["f6"] == first["loss"]["native"]
    assert first["parameters"]["adapter"]["violations"] > 0


def test_scalar_loss_difference_is_detected(tmp_path, monkeypatch):
    original = numerical.sequence_loss

    def wrong_loss(*args, **kwargs):
        result = original(*args, **kwargs)
        result["loss"] += 0.01
        return result

    monkeypatch.setattr(numerical, "sequence_loss", wrong_loss)
    report = run(prepare(tmp_path), tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    assert report["comparisons"][0]["parameters"]["adapter"]["violations"] == 0


def test_mean_gate_uses_whole_g8_tokens_not_each_sequence(tmp_path):
    items = prepare(tmp_path, lengths=[1] + [100] * 7, deltas=[0.02] + [0] * 7)
    report = run(items, tmp_path / "output")
    assert report["numerical_result"] == "PASS"
    audit = report["assessment"]["probability_audit"]
    assert audit["sampler_train_max_absolute_difference"] > 0.01
    assert audit["sampler_train_mean_absolute_difference"] < 0.01
    assert audit["token_count"] == 701


@pytest.mark.parametrize("gate", ["mean", "max", "tis"])
def test_all_three_actual_probability_gates_block(tmp_path, gate):
    if gate == "mean":
        lengths, deltas = [100] * 8, [0.02] * 8
    elif gate == "max":
        lengths, deltas = [1] + [1000] * 7, [5.1] + [0] * 7
    else:
        lengths, deltas = [122] + [1536] * 7, [0.694] + [0] * 7
    report = run(prepare(tmp_path, lengths=lengths, deltas=deltas), tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    audit = numerical._numerical_audit(report["probability_vectors"])
    if gate != "mean":
        assert audit["sampler_train_mean_absolute_difference"] < 0.01
    if gate == "tis":
        assert audit["tis_truncated_fraction"] > 0.01
        assert audit["sampler_train_max_absolute_difference"] < 5


@pytest.mark.parametrize("defect", ["tokens", "padding", "eos", "technical", "1537"])
def test_original_token_boundary_failures_are_not_repaired_or_reencoded(tmp_path, defect):
    items = prepare(tmp_path)
    row, raw = items[1][0], items[2][0]
    if defect == "tokens":
        row["tokens"][0] += 1
    elif defect == "padding":
        raw["backend_sequence_tokens"] += [7]
        raw["backend_padding_count"] += 1
    elif defect == "eos":
        raw["finish_reason"] = "eos"
    elif defect == "technical":
        raw["technical_validation_errors"] = ["FIXTURE_INVALID"]
    else:
        raw["tokens"] = raw["raw_tokens"] = row["tokens"] = [1] * 1537
    report = run(items, tmp_path / "output")
    assert report["numerical_result"] == "FAIL"
    assert not items[0].calls


def test_1536_fixture_coverage_remains_synthetic_only(tmp_path):
    report = run(prepare(tmp_path, lengths=[1536] * 8), tmp_path / "output")
    assert report["numerical_result"] == "PASS"
    assert report["observed_coverage"]["tokens_1536"] == "OBSERVED"
    assert report["scope"] == "SYNTHETIC_ONLY"
    assert report["observed_coverage"]["eos"] == "NOT_OBSERVED"


@pytest.mark.parametrize(
    "mutation", ["vector", "gradient", "loss", "missing_parameter", "missing_row", "restoration"]
)
def test_assessment_recomputes_facts_and_ignores_status_flags(tmp_path, mutation):
    report = run(prepare(tmp_path), tmp_path / "output")
    report["status"] = report["numerical_result"] = "PASS"
    if mutation == "vector":
        report["probability_vectors"][0]["train_logprobs"][0] += 8
    elif mutation == "gradient":
        report["comparisons"][0]["parameters"]["adapter"]["violations"] = 1
    elif mutation == "loss":
        report["comparisons"][0]["loss"]["f6"] += 1
    elif mutation == "missing_parameter":
        del report["comparisons"][0]["parameters"]["unused"]
    elif mutation == "missing_row":
        report["comparisons"].pop()
    else:
        report["restoration"]["rng_after_sha256"] = "9" * 64
    assert numerical.assess_facts(report)["status"] == "FAIL"


def test_production_rejects_arbitrary_sessions_and_synthetic_receipts(tmp_path, monkeypatch):
    from sr_f6 import replay

    report = run(prepare(tmp_path), tmp_path / "T03")
    with pytest.raises(PermissionError, match="Authenticated"):
        numerical.run_session(SimpleNamespace(scope="SERVER_ACTUAL"))
    # Only isolate the provenance dependency to exercise the scope rejection;
    # no fake GPU report is produced and no numerical gate is mocked.
    monkeypatch.setattr(replay, "verify_spec", lambda *args, **kwargs: {"groups": {"b0": {}}})
    path = tmp_path / "T03/RESULT.json"
    with pytest.raises(PermissionError, match="Synthetic"):
        numerical.summary(
            tmp_path,
            phase_identity_sha256="a" * 64,
            replay_spec_sha256="b" * 64,
            receipts=[dict(path="T03/RESULT.json", sha256=file_hash(path))],
        )
    assert report["scope"] == "SYNTHETIC_ONLY"


@pytest.mark.parametrize(
    "kind", ["completion_attempts", "generated_tokens", "extra_forward", "wrong_length", "missing"]
)
def test_accounting_uses_actual_terminal_events_and_refuses_extra_calls(tmp_path, kind):
    report = run(prepare(tmp_path), tmp_path / "output")
    events = report["accounting"]["events"]
    assert len(events) == 16
    assert report["assessment"]["accounting"]["reference_calls"] == 8
    assert report["assessment"]["accounting"]["current_calls"] == 8
    if kind in ("completion_attempts", "generated_tokens"):
        events.append(dict(task_id="SYNTHETIC_ONLY", parent="b0", kind=kind, count=1, metadata={}))
    elif kind == "extra_forward":
        events.append(copy.deepcopy(events[-1]))
    elif kind == "wrong_length":
        events[0]["metadata"]["completion_tokens"] += 1
    else:
        events.pop()
    assert numerical.assess_facts(report)["status"] == "FAIL"


def test_run_load_failure_is_retained_and_cli_fail_is_nonzero(tmp_path, monkeypatch):
    from sr_f6 import replay, runtime

    monkeypatch.setattr(replay, "verify_spec", lambda *a, **kw: {})

    def fail(*args, **kwargs):
        assert kwargs["mode"] == "REPLAY"
        raise RuntimeError("CPU fixture loader failure")

    monkeypatch.setattr(runtime, "load", fail)
    with pytest.raises(RuntimeError, match="loader failure"):
        numerical.run(
            tmp_path,
            "b0",
            "fixture-failed-load",
            phase_identity_sha256="a" * 64,
            replay_spec_sha256="b" * 64,
        )
    audit = tmp_path / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS/fixture-failed-load"
    assert json.loads((audit / "FAILURE.json").read_bytes())["status"] == "FAIL"
    assert not (audit / "T03/RESULT.json").exists()
    monkeypatch.setattr(numerical, "run", lambda *a, **kw: dict(status="FAIL"))
    assert (
        numerical.main(
            [
                "run",
                "--root",
                str(tmp_path),
                "--parent",
                "b0",
                "--task-id",
                "fixture",
                "--phase-identity-sha256",
                "a" * 64,
                "--spec-sha256",
                "b" * 64,
            ]
        )
        == 1
    )


def test_prior_output_is_never_overwritten(tmp_path):
    items = prepare(tmp_path)
    result = run(items, tmp_path / "output")
    content = (tmp_path / "output/RESULT.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(items, tmp_path / "output")
    assert content == (tmp_path / "output/RESULT.json").read_bytes()
    assert json.loads(content) == result


def test_cli_has_only_implemented_entries():
    with pytest.raises(SystemExit) as result:
        numerical.main(["--help"])
    assert result.value.code == 0


@pytest.mark.parametrize(
    "payload",
    [
        '{"x":1,"x":2}',
        '{"v":{"x":1,"x":2}}',
        '{"v":NaN}',
        '{"v":Infinity}',
        '{"v":-Infinity}',
        '{"v":1e999}',
        '{"v":-1e999}',
    ],
)
def test_evidence_json_rejects_duplicate_keys_and_nonfinite_literals(payload):
    with pytest.raises(PermissionError):
        numerical._json(payload)


@pytest.mark.parametrize("state", ["unknown", "failed", "unreported_result"])
def test_summary_retains_incomplete_failed_and_unreported_attempts(tmp_path, monkeypatch, state):
    from sr_f6 import replay

    spec_ref = dict(path=replay.SPEC_PATH, sha256="b" * 64)
    # No real model receipt is fabricated; only attempt-discovery behavior is isolated.
    monkeypatch.setattr(
        replay, "verify_spec", lambda *a, **kw: dict(groups={p: {} for p in ("b0", "b1", "b2")})
    )
    task = tmp_path / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS/fixture-attempt"
    task.mkdir(parents=True)
    (task / "ATTEMPT.json").write_text(
        json.dumps(
            dict(spec_ref=spec_ref, phase_identity_sha256="a" * 64, task_id=task.name, parent="b0")
        )
    )
    if state == "failed":
        (task / "FAILURE.json").write_text(
            json.dumps(dict(status="FAIL", error="synthetic load failure"))
        )
    elif state == "unreported_result":
        (task / "T03").mkdir()
        (task / "T03/RESULT.json").write_text(json.dumps(dict(scope="SYNTHETIC_ONLY")))
    result = numerical.summary(
        tmp_path, phase_identity_sha256="a" * 64, replay_spec_sha256="b" * 64, receipts=[]
    )
    assert result["status"] == ("FAIL" if state == "failed" else "INCOMPLETE")
    assert result["training_authorized"] is False
    assert len(result["attempts"]) == 1
    if state == "unknown":
        assert result["unknown_attempts"] == [task.name]
    elif state == "unreported_result":
        assert len(result["unreported_results"]) == 1


def test_a_lookalike_exact_session_has_no_private_active_authorization(tmp_path):
    from sr_f6.replay import ReplaySession

    session = ReplaySession()
    session.runtime = prepare(tmp_path)[0]
    session.scope = "SERVER_ACTUAL"
    session._lock = SimpleNamespace(closed=False)
    with pytest.raises(PermissionError, match="Inactive"):
        numerical.run_session(session)


@pytest.mark.parametrize("artifact", ["T03/RESULT.json", "FAILURE.json", "T03/STARTED.json"])
def test_orphan_attempt_products_are_retained_as_unknown(tmp_path, monkeypatch, artifact):
    from sr_f6 import replay

    monkeypatch.setattr(
        replay, "verify_spec", lambda *a, **kw: dict(groups={p: {} for p in ("b0", "b1", "b2")})
    )
    task = tmp_path / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS/orphan-fixture"
    path = task / artifact
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(scope="SYNTHETIC_ONLY", status="FAIL")))
    result = numerical.summary(
        tmp_path, phase_identity_sha256="a" * 64, replay_spec_sha256="b" * 64, receipts=[]
    )
    assert result["status"] == "INCOMPLETE"
    assert result["unknown_attempts"] == [task.name]
    assert result["attempts"] == [
        dict(
            task_id=task.name,
            parent=None,
            status="ORPHAN_WITHOUT_ATTEMPT",
            entries=[artifact.split("/")[0]],
        )
    ]


@pytest.mark.parametrize(
    "field",
    [
        None,
        "parent",
        "task_id",
        "spec_ref",
        "phase_identity_sha256",
        "optimizer_updates",
        "generated_responses",
        "missing",
        "other_directory",
    ],
)
def test_actual_attempt_identity_and_exact_output_path_are_required(tmp_path, field):
    spec = dict(path="PHASES/PILOT_V1/T03_SPEC.json", sha256="b" * 64)
    audit = tmp_path / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS/fixture-task"
    if field == "other_directory":
        audit = tmp_path / "PHASE_RUNS/PILOT_V1/elsewhere/fixture-task"
    audit.mkdir(parents=True)
    attempt = dict(
        parent="b0",
        task_id="fixture-task",
        spec_ref=spec,
        phase_identity_sha256="a" * 64,
        optimizer_updates=0,
        generated_responses=0,
    )
    if field in ("parent", "task_id", "phase_identity_sha256"):
        attempt[field] = "wrong"
    elif field == "spec_ref":
        attempt[field] = {**spec, "sha256": "c" * 64}
    elif field in ("optimizer_updates", "generated_responses"):
        attempt[field] = 1
    if field != "missing":
        (audit / "ATTEMPT.json").write_text(json.dumps(attempt))
    kwargs = dict(parent="b0", task_id="fixture-task", phase_pin="a" * 64, spec_ref=spec)
    if field is None:
        bound = numerical._attempt_reference(tmp_path, audit, **kwargs)
        assert bound == dict(
            path="PHASE_RUNS/PILOT_V1/REPLAY_RUNS/fixture-task/ATTEMPT.json",
            sha256=file_hash(audit / "ATTEMPT.json"),
        )
    else:
        with pytest.raises(PermissionError):
            numerical._attempt_reference(tmp_path, audit, **kwargs)
