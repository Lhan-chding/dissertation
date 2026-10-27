"""Pilot loop tests with actual torch Adam/checkpoints and explicitly synthetic data."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import state_hash, trainable_state
from sr_f3_ce.training import fresh_optimizer
from sr_f6 import pilot
from sr_f6.registration import DIAGNOSTIC_ROWS
from sr_f6.transactions import RunTransactions
from srf6_ref.design import make_root

_spec = importlib.util.spec_from_file_location(
    "srf6_pilot_rollout_fixture", Path(__file__).with_name("test_rollouts.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)


class CPUTrainerRuntime(_fixtures.FakeRuntime):
    def __init__(self, model, identity, text):
        super().__init__(model, identity, text)
        self.parent = trainable_state(model)

    def prepare_input(self, safe, root, *, protocol):
        inputs = dict(
            input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones((1, 2), dtype=torch.long)
        )
        return dict(inputs=inputs, routing=dict(input_tensor_hash=state_hash(inputs)))

    def _probability(self, row, state):
        change = sum(value.sum() for value in state.values()) - sum(
            value.sum() for value in self.parent.values()
        )
        slope = (row["seed"] % 4 + 1) / 16
        return torch.tensor([-1.0, -2.0, -3.0]) + change * slope

    def generate_group(self, safe, root, seeds, *, protocol, on_completion):
        self.reserve("completion_attempts", len(seeds), group_seed=seeds[0], protocol="structured")

        def capture(row):
            prepared = self.prepare_input(safe, root, protocol=protocol)
            row.update(
                prompt=safe["prompt"],
                prompt_token_count=2,
                prompt_tensor_hash=prepared["routing"]["input_tensor_hash"],
                native_prompt_inputs={k: v.tolist() for k, v in prepared["inputs"].items()},
                sampler_logprobs=self._probability(row, dict(self.model.named_parameters()))
                .detach()
                .tolist(),
                task_id=self.identity["task_id"],
                execution_origin="SYNTHETIC_TEST",
            )
            on_completion(row)
            self.reserve(
                "generated_tokens",
                len(row["tokens"]),
                group_seed=seeds[0],
                row_index=row["group_row_index"],
            )

        return super().generate_group(safe, root, seeds, protocol=protocol, on_completion=capture)

    def batch_reference_forward(self, records, reference):
        return [self._probability(row, reference) for row in records]

    def batch_sequence_forward(self, records, *, purpose, grad):
        return [self._probability(row, dict(self.model.named_parameters())) for row in records]

    def reserve(self, kind, count, **metadata):
        path = (
            Path(self.identity["audit_root"]) / "accounting" / (self.identity["task_id"] + ".jsonl")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _append(
            path,
            dict(
                kind=kind,
                count=count,
                metadata=metadata,
                task_id=self.identity["task_id"],
                parent=self.identity["parent_id"],
            ),
        )


@pytest.fixture(scope="module")
def prototype():
    return make_root("DEV_PILOT", "BINDING", 0, 2026102712)["tasks"]["P1"][0]


def _case(path, prototype):
    def edit(reg):
        reg["identity"]["diagnostic_row_indices"] = list(DIAGNOSTIC_ROWS)

    case = _fixtures.prepare(path, prototype, edit_registration=edit)
    new = case["experiment"] / "PHASE_RUNS/PILOT_V1/PILOT_FROZEN/runs/b0/r0/U11"
    new.parent.mkdir(parents=True)
    case["run"].rename(new)
    case["run"] = new
    case["identity"]["diagnostic_row_indices"] = list(DIAGNOSTIC_ROWS)
    case["parent_model"] = copy.deepcopy(case["runtime"].model)
    case["runtime_identity"] = copy.deepcopy(case["runtime"].identity)
    case["text"] = case["runtime"].text
    return case


def _fresh(case, task):
    runtime = CPUTrainerRuntime(
        copy.deepcopy(case["parent_model"]),
        {
            **copy.deepcopy(case["runtime_identity"]),
            "task_id": task,
            "audit_root": str(case["run"]),
        },
        case["text"],
    )
    optimizer, scheduler = fresh_optimizer(runtime.model)
    target = case["run"] / "runtime_identity" / (task + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump({**runtime.identity, "execution_origin": "SYNTHETIC_TEST"}, stream)
    return runtime, optimizer, scheduler


def _run(case, task, failpoint=None):
    runtime, optimizer, scheduler = _fresh(case, task)
    with RunTransactions(case["run"], case["identity"]) as tx:
        result = pilot.run_synthetic(
            tx,
            runtime,
            optimizer,
            scheduler,
            experiment_root=case["experiment"],
            registration_ref=case["ref"],
            failpoint=failpoint,
        )
    return result, runtime, optimizer, scheduler


def test_twelve_actual_cpu_updates_commit_original_rollouts_without_claiming_gpu(
    tmp_path, prototype
):
    case = _case(tmp_path, prototype)
    result, runtime, _optimizer, _scheduler = _run(case, "complete")
    assert result["status"] == "SYNTHETIC_ONLY"
    assert result["segment_updates"] == 12 and result["segment_new_model_calls"] == 1536
    assert runtime.calls == 1536 and runtime.identity["step"] == 12
    assert result["training_authorized"] is False
    with RunTransactions(case["run"], case["identity"]) as tx:
        markers = tx.published()
        assert [m["step"] for m in markers] == list(range(13))
        assert markers[-1]["sampler_state"]["logical_calls"] == 1536
        assert len(markers[-1]["sampler_state"]["consumed_qids"]) == 12
        for marker in markers[1:]:
            event = tx._reference(marker["references"]["evidence"])
            assert event["status"] == "PREUPDATE_PASS"
            assert event["source_audit_event"]["byte_count"] > 0
            channels = event["channel_gradients"]
            assert channels["status"] == (
                "MEASURED" if marker["step"] in (1, 4, 8, 12) else "NOT_SCHEDULED"
            )
    assert state_hash(trainable_state(runtime.model)) != case["identity"]["reference_sha256"]


@pytest.mark.parametrize("failure", ["after_ready", "after_update", "after_rename"])
def test_interruption_restores_parent_reference_and_uses_published_commit_or_exact_cache(
    tmp_path, prototype, failure
):
    baseline = _case(tmp_path / "baseline", prototype)
    _run(baseline, "continuous")
    expected = torch.load(baseline["run"] / "checkpoints/step-012/state.pt", weights_only=True)
    case = _case(tmp_path / "interrupted", prototype)

    def crash(point, step):
        if point == failure and step == 1:
            raise RuntimeError("synthetic physical interruption")

    with pytest.raises(RuntimeError, match="physical interruption"):
        _run(case, "broken", crash)
    with RunTransactions(case["run"], case["identity"]) as tx:
        assert tx.published()[-1]["step"] == (1 if failure == "after_rename" else 0)
    raw_before = {
        p.relative_to(case["run"]).as_posix(): p.read_bytes()
        for p in (case["run"] / "rollouts/step-001").rglob("*.jsonl")
    }
    assert len(raw_before) == 16
    result, runtime, _optimizer, _scheduler = _run(case, "resumed")
    assert result["status"] == "SYNTHETIC_ONLY" and runtime.calls == 1408
    assert result["committed_step"] == 12
    assert raw_before == {
        p.relative_to(case["run"]).as_posix(): p.read_bytes()
        for p in (case["run"] / "rollouts/step-001").rglob("*.jsonl")
    }
    assert runtime.identity["reference_hash"] == case["identity"]["reference_sha256"]
    with RunTransactions(case["run"], case["identity"]) as tx:
        assert len(tx.published()) == 13
    actual = torch.load(case["run"] / "checkpoints/step-012/state.pt", weights_only=True)
    for key in (
        "policy",
        "optimizer",
        "scheduler",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    ):
        assert state_hash(actual[key]) == state_hash(expected[key]), key
    for key in (
        "next_step",
        "committed_log_step",
        "consumed_qids",
        "logical_calls",
        "actual_calls",
        "rollout_cursor",
    ):
        assert actual["sampler_state"][key] == expected["sampler_state"][key]


def _event_fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    runtime = SimpleNamespace(
        identity=dict(
            run_root=str(tmp_path),
            audit_root=str(run),
            execution_mode="TRAINING",
            phase="PILOT_V1",
            task_id="t1",
        )
    )
    tx = SimpleNamespace(directory=run, identity=dict(run_id="logical", arm="U11"))
    event = dict(
        status="PREUPDATE_PASS",
        task_id="t1",
        at="fixture",
        run_id="logical",
        arm="U11",
        committed_source_step=0,
        policy_before_sha256="a" * 64,
        audit_root=str(run),
        diagnostics=dict(status="FINITE", sampler_train_mean_absolute_difference=0.0),
        channel_gradients=dict(status="NOT_SCHEDULED"),
    )
    update = {
        **event["diagnostics"],
        "status": "UPDATED_UNCOMMITTED",
        "channel_gradients": event["channel_gradients"],
        "policy_before_sha256": "a" * 64,
    }
    return tx, runtime, event, update


def _append(path, value):
    with path.open("ab") as stream:
        stream.write(json.dumps(value).encode() + b"\n")


def test_preupdate_bridge_only_freezes_current_append_slice(tmp_path):
    tx, runtime, event, update = _event_fixture(tmp_path)
    path = tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl"
    _append(path, {**event, "run_id": "unrelated_old_failure", "status": "FAIL"})
    before = path.read_bytes()
    position = pilot._audit_position(runtime)
    _append(path, event)
    ref = pilot._freeze_preupdate(tx, runtime, position, update, 1)
    saved = json.loads((tx.directory / ref["path"]).read_bytes())
    assert saved["source_audit_event"]["byte_offset"] == len(before)
    assert saved["run_id"] == "logical"
    assert path.read_bytes().startswith(before)


@pytest.mark.parametrize(
    "defect",
    [
        "no_new_event",
        "extra_event",
        "wrong_step",
        "wrong_task",
        "wrong_policy",
        "wrong_diagnostics",
        "missing_newline",
        "replaced_log",
    ],
)
def test_preupdate_bridge_rejects_old_pass_partial_or_mismatched_actual_event(tmp_path, defect):
    tx, runtime, event, update = _event_fixture(tmp_path)
    path = tx.directory / "NUMERICAL_TOKEN_AUDIT.jsonl"
    _append(path, event)
    position = pilot._audit_position(runtime)
    if defect == "replaced_log":
        path.rename(path.with_suffix(".original"))
    if defect == "wrong_step":
        event["committed_source_step"] = 1
    if defect == "wrong_task":
        event["task_id"] = "different-physical-attempt"
    if defect == "wrong_policy":
        event["policy_before_sha256"] = "b" * 64
    if defect == "wrong_diagnostics":
        event["diagnostics"] = dict(status="FINITE", sampler_train_mean_absolute_difference=0.2)
    if defect != "no_new_event":
        _append(path, event)
    if defect == "extra_event":
        _append(path, event)
    if defect == "missing_newline":
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(PermissionError):
        pilot._freeze_preupdate(tx, runtime, position, update, 1)
    assert not (tx.directory / "PREUPDATE_EVIDENCE").exists()
