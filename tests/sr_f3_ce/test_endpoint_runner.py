import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import endpoint_runner
from sr_f3_ce.evaluation import EXPERIMENT_ID, group_slots, object_hash, sha_file


def test_endpoint_and_release_require_all_terminal_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(endpoint_runner, "_completed", lambda root: set())
    with pytest.raises(PermissionError, match="terminal receipts"):
        endpoint_runner.execute_endpoint(tmp_path, "SRF2_J6_s71001")
    with pytest.raises(PermissionError, match="terminal receipts"):
        endpoint_runner.execute_release(tmp_path)
    with pytest.raises(PermissionError, match="Unregistered"):
        endpoint_runner.execute_endpoint(tmp_path, "BEST_SEED")


def test_all_training_receipts_do_not_release_missing_endpoint(monkeypatch, tmp_path):
    from sr_f3_ce.evaluation import endpoint_models

    training = {"train--" + m["model_id"] for m in endpoint_models()[3:]}
    monkeypatch.setattr(endpoint_runner, "_completed", lambda root: training)
    endpoint_runner._require_terminal(tmp_path)
    with pytest.raises(PermissionError, match="eval--"):
        endpoint_runner._require_terminal(tmp_path, evaluations=True)


def test_final_checkpoint_load_exactly_and_does_not_select_intermediate(monkeypatch, tmp_path):
    import torch

    from mm_core.training import state_hash, trainable_state
    from sr_f1.contract import digest
    from sr_f3_ce import checkpoint, runner

    model = dict(
        model_id="SRF3CE_EXEC_b1", parent_id="SRF2_J6_s71002", block=1, arm="EXEC", step=64
    )
    layer = torch.nn.Linear(1, 1)
    runtime = SimpleNamespace(model=layer, identity={}, adapter_path="parent")
    runtime.stable_model_identity = lambda: dict(runtime.identity)
    runtime.current_adapter_identity = lambda: dict(policy=state_hash(trainable_state(layer)))
    parameters = {k: torch.ones_like(v) for k, v in trainable_state(layer).items()}
    (tmp_path / "RUN_IDENTITY.json").write_text('{"frozen":"identity"}')
    state = dict(
        run_identity=dict(
            run_id=model["model_id"],
            arm="EXEC",
            block=1,
            parent_id=model["parent_id"],
            stage="SCIENCE",
            experiment_id="SR-F3-CE-20261016",
            updates=64,
            future_seed=73102,
            run_identity_sha256=sha_file(tmp_path / "RUN_IDENTITY.json"),
        ),
        reference_hash=state_hash(trainable_state(layer)),
        committed_logical_step=64,
        parameters=parameters,
    )
    asked = []

    def read(directory, *, step):
        asked.append(step)
        return state, dict(step=step, sha256="checkpoint")

    monkeypatch.setattr(runner, "load_runtime", lambda *args: (runtime, None, None))
    monkeypatch.setattr(checkpoint, "read_checkpoint", read)
    export = tmp_path / "runs" / model["model_id"] / "adapters/step-64"
    export.mkdir(parents=True)
    (export / "adapter.bin").write_bytes(b"adapter")
    marker = dict(
        step=64,
        run_identity_sha256=digest(state["run_identity"]),
        policy_hash=state_hash(parameters),
        files={"adapter.bin": sha_file(export / "adapter.bin")},
    )
    (export / "IDENTITY.json").write_text(json.dumps(marker))
    loaded, provenance = endpoint_runner._load_endpoint_runtime(
        tmp_path, model, "eval--" + model["model_id"]
    )
    assert asked == [64]
    assert provenance["policy_hash"] == state_hash(parameters)
    assert loaded.identity["model_id"] == model["model_id"]
    state["committed_logical_step"] = 48
    with pytest.raises(PermissionError, match="checkpoint identity"):
        endpoint_runner._load_endpoint_runtime(tmp_path, model, "eval--" + model["model_id"])


def test_release_delivered_raw_is_reopened_and_rescored(monkeypatch, tmp_path):
    from sr_f3_ce import analysis, orchestration, runner

    plan = Path(__file__).resolve().parents[2] / "artifacts/sr_f3_ce_20261016/plan"
    task = json.loads((plan / "manifests/TASKS_AUDIT_ONLY.jsonl").read_text().splitlines()[0])
    # Small execution-layer fixture; fixed full-matrix counts are separately
    # tested by evaluation.test_exact_fixed_slot_budgets, without fake GPU runs.
    slot = dict(
        model_id="SRF2_J6_s71001",
        step=0,
        pool="TEST_ID",
        family=task["family"],
        root_id=task["root_id"],
        qid=task["qid"],
        protocol="structured",
        view=None,
        draw=0,
        samples_per_prompt=1,
        generation={},
        slot_id="slot",
        sealed=True,
    )
    group = group_slots([slot])[0]
    raw = dict(
        slot,
        group_id=group["group_id"],
        raw_text="invalid model answer",
        finish_reason="eos",
        generation_status="COMPLETE",
        tokens=[1],
        decoded_prefix_views={str(n): "invalid model answer" for n in (128, 256, 512, 1024)},
        technical_validation_errors=[],
    )
    monkeypatch.setattr(endpoint_runner, "_require_terminal", lambda root, evaluations=False: None)
    monkeypatch.setattr(
        runner,
        "dataset_and_store",
        lambda root: (SimpleNamespace(tasks={task["qid"]: task}, runs=[]), None),
    )
    panels = {"CHARTQAPRO": dict(status="UNAVAILABLE", image_count=0, reason="test fixture")}
    monkeypatch.setattr(endpoint_runner, "_external", lambda root: ({}, {}, panels, None))
    monkeypatch.setattr(endpoint_runner, "endpoint_models", lambda: [])
    monkeypatch.setattr(endpoint_runner, "endpoint_slots", lambda *args: [slot])
    monkeypatch.setattr(
        endpoint_runner, "_raw_groups", lambda *args: iter([(group, [raw], {"sha256": "rawfile"})])
    )
    monkeypatch.setattr(
        endpoint_runner,
        "release_evaluation",
        lambda *args, **kwargs: dict(
            status="RELEASED",
            experiment_id=EXPERIMENT_ID,
            raw_groups={group["group_id"]: object_hash([raw])},
        ),
    )
    checks = []

    def primary(rows, tasks, receipt):
        analysis._assert_release(rows, receipt)
        checks.append(rows[0]["score"]["value_correct"])
        return dict(status="NUMERICAL_ANALYSIS_COMPLETE")

    monkeypatch.setattr(analysis, "primary_analysis", primary)
    monkeypatch.setattr(orchestration, "write_task_receipt", lambda root, task_id, **kwargs: kwargs)
    receipt = endpoint_runner.execute_release(tmp_path)
    assert checks == [0]
    assert receipt["evidence"]["raw_rescoring"] == "PASS"
    rescoring = json.loads((tmp_path / "release/RAW_RESCORING.json").read_text())
    assert rescoring["records"] == 1 and rescoring["gzip_read_to_eof"]
    assert rescoring["mismatches"] == []
    assert not rescoring["independently_implemented_scorer_claimed"]
    assert (tmp_path / "release/ENDPOINT_RAW.jsonl.gz").is_file()


def test_immutable_release_file_never_overwrites(tmp_path):
    temporary, final = tmp_path / "temp", tmp_path / "final"
    temporary.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(PermissionError):
        endpoint_runner._publish_file(temporary, final)
    assert final.read_bytes() == b"old" and temporary.read_bytes() == b"new"
