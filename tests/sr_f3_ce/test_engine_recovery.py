"""ENGINE orchestration checks; CPU fixtures cannot release the real GPU gate."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from mm_core.training import state_hash, trainable_state
from sr_f3_ce import engine_recovery as engine
from sr_f3_ce.checkpoint import commit_checkpoint, read_checkpoint
from sr_f3_ce.protocol import file_hash
from sr_f3_ce.protocol_revision import revision_config
from sr_f3_ce.training import fresh_optimizer
from sr_f12.evaluation import durable_json

torch = pytest.importorskip("torch")


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(
        "engine_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_fixture(root, support):
    artifacts = []
    for record in support:
        record["serialized_teacher_target"] = {"text": "CPU mask fixture only"}
        for arm in ("PREFIX", "EXEC"):
            path = root / f"native/support_masks/{record['qid']}-{arm}.json"
            durable_json(
                path,
                {
                    key: record[key]
                    for key in ("tokens", "answer_mask", "work_mask", "serialized_teacher_target")
                },
            )
            artifacts.append(dict(path=str(path.relative_to(root)), sha256=file_hash(path)))
    durable_json(root / "native/SUPPORT_MASK_RECEIPT.json", dict(status="PASS", files=artifacts))


def test_probe_is_cpu_only_reuses_cache_and_never_performs_adam(tmp_path):
    fixture = fixture_module("test_training")
    runtime = fixture.ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    records, support = fixture.task_records(runtime), fixture.supports()
    native_fixture(tmp_path, support)
    before = fixture.model_state(runtime, optimizer, scheduler)
    receipt = engine.verify_engine_probes(
        tmp_path, runtime, optimizer, scheduler, records, reference, support
    )
    assert receipt["status"] == "CPU_EVIDENCE_ONLY"
    assert receipt["actual_cuda"] is False
    assert receipt["physical_updates"] == receipt["new_rollouts"] == 0
    assert receipt["ce_zero_equals_pure_rl_exact"] and receipt["nonzero_ce_gradient"]
    assert receipt["selected_microbatch_size"] == 4
    assert fixture.model_state(runtime, optimizer, scheduler) == before
    calls = runtime.task_calls
    assert (
        engine.verify_engine_probes(
            tmp_path, runtime, optimizer, scheduler, records, reference, support
        )
        == receipt
    )
    assert runtime.task_calls == calls
    path = tmp_path / receipt["artifacts"][0]["path"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(PermissionError, match="artifact changed"):
        engine.verify_engine_probes(
            tmp_path, runtime, optimizer, scheduler, records, reference, support
        )


def test_probe_cuda_oom_fallback_is_whole_state_and_reports_single_feasible(tmp_path):
    fixture = fixture_module("test_training")
    runtime = fixture.ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    records, support = fixture.task_records(runtime), fixture.supports()
    native_fixture(tmp_path, support)
    original = runtime.batch_sequence_forward

    def only_one(rows, **kwargs):
        if len(rows) > 1:
            raise torch.cuda.OutOfMemoryError("CPU-injected resource failure")
        return original(rows, **kwargs)

    runtime.batch_sequence_forward = only_one
    before = fixture.model_state(runtime, optimizer, scheduler)
    receipt = engine.verify_engine_probes(
        tmp_path, runtime, optimizer, scheduler, records, reference, support
    )
    assert receipt["selected_microbatch_size"] == 1
    assert receipt["feasible_microbatches"] == [1]
    assert receipt["microbatch_comparison_scope"] == "ONLY_ONE_FEASIBLE"
    assert receipt["microbatch_comparisons"] == []
    assert receipt["gpu_cross_microbatch_equivalence_established"] is False
    assert len(receipt["attempts"]) == 2
    assert all(row["live_state_restored"] for row in receipt["attempts"])
    assert fixture.model_state(runtime, optimizer, scheduler) == before


def test_intents_are_durable_ordered_and_bound_to_six_updates(tmp_path, monkeypatch):
    from sr_f3_ce import orchestration

    monkeypatch.setattr(
        orchestration, "worker_binding", lambda *_: dict(job_id="CPU_TEST", claim_id="CPU_TEST")
    )
    for index, (segment, step) in enumerate(engine.PHYSICAL_SEQUENCE, 1):
        receipt = engine.reserve_engine_step(
            tmp_path,
            dict(
                segment=segment,
                step=step,
                arm="EXEC",
                microbatch_size=4,
                policy_hash=state_hash("CPU fixture"),
            ),
        )
        assert receipt["index"] == index
        assert receipt["unknown_outcome_counts_against_budget"] is True
    with pytest.raises(PermissionError, match="Six physical"):
        engine.reserve_engine_step(
            tmp_path, dict(segment="split", step=4, arm="EXEC", microbatch_size=4)
        )
    assert len(engine._intent_rows(tmp_path)) == 6


def test_unknown_adam_outcome_cannot_be_repeated(tmp_path, monkeypatch):
    from sr_f3_ce import orchestration

    monkeypatch.setattr(
        orchestration, "worker_binding", lambda *_: dict(job_id="CPU_TEST", claim_id="CPU_TEST")
    )
    metadata = dict(
        segment="continuous",
        step=1,
        arm="EXEC",
        microbatch_size=4,
        policy_hash=state_hash("CPU fixture"),
    )
    engine.reserve_engine_step(tmp_path, metadata)
    with pytest.raises(PermissionError, match="Unresolved/repeated"):
        engine.reserve_engine_step(tmp_path, metadata)
    assert len(engine._intent_rows(tmp_path)) == 1


def test_step_two_copy_preserves_exact_committed_bytes_and_state(tmp_path):
    fixture = fixture_module("test_checkpoint")
    runtime = fixture.ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    for step in (1, 2):
        fixture.one_step(runtime, optimizer, scheduler, reference, step)
    state = fixture.make_state(runtime, optimizer, scheduler, reference, 2)
    continuous = tmp_path / "engine/resume/continuous/checkpoints"
    receipt = commit_checkpoint(continuous, state)
    engine._copy_step_two(tmp_path)
    split = tmp_path / "engine/resume/split/checkpoints"
    assert file_hash(split / receipt["path"]) == receipt["sha256"]
    assert state_hash(read_checkpoint(split)[0]) == state_hash(state)
    assert read_checkpoint(split)[1] == receipt
    engine._copy_step_two(tmp_path)
    assert not (split / "commit-04.json").exists()


def test_segment_launch_uses_real_new_process_and_preserves_failure(tmp_path, monkeypatch):
    code = tmp_path / "code"
    script = code / "scripts/sr_f3_ce/run_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("import os\nprint(os.getpid())\n")
    monkeypatch.setattr(engine, "__file__", str(code / "src/sr_f3_ce/engine_recovery.py"))
    base = tmp_path / "engine/resume"
    engine._run_segment(tmp_path, "continuous")
    process = json.loads((base / "continuous.PROCESS.json").read_text())
    assert process["pid"] != os.getpid()
    assert int((base / "continuous.stdout.log").read_text()) == process["pid"]
    engine._run_segment(tmp_path, "continuous")  # receipt-only reuse; no second process
    assert json.loads((base / "continuous.PROCESS.json").read_text()) == process
    script.write_text("raise SystemExit(2)\n")
    with pytest.raises(RuntimeError, match="exit code 2"):
        engine._run_segment(tmp_path, "split")
    with pytest.raises(RuntimeError, match="Prior ENGINE subprocess failed"):
        engine._run_segment(tmp_path, "split")


def test_unknown_child_outcome_blocks_blind_rerun(tmp_path):
    durable_json(tmp_path / "engine/resume/continuous.LAUNCH.json", dict(command=[]))
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        engine._run_segment(tmp_path, "continuous")


def test_engine_state_requires_new_root_and_actual_c2_prefill(tmp_path):
    durable_json(tmp_path / "RUN_IDENTITY.json", {"stage": "CPU_REVISION_FIXTURE"})
    durable_json(
        tmp_path / "plan/config/SR_F3_CE.json",
        revision_config({"generation": {"prefill": "{"}}, "c2"),
    )
    identity = engine._protocol_identity(tmp_path)
    state = dict(
        run_identity={
            key: identity[key]
            for key in ("run_identity_sha256", "protocol_revision", "assistant_prefill")
        }
    )
    assert engine._verify_state_protocol(tmp_path, state) == identity
    for key, old in (
        ("run_identity_sha256", "0" * 64),
        ("protocol_revision", "v1"),
        ("assistant_prefill", "{"),
    ):
        with pytest.raises(PermissionError, match="another run or protocol"):
            engine._verify_state_protocol(
                tmp_path, {"run_identity": dict(state["run_identity"], **{key: old})}
            )


def test_existing_engine_process_cannot_be_reused_under_changed_protocol(tmp_path, monkeypatch):
    code = tmp_path / "code"
    script = code / "scripts/sr_f3_ce/run_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("import os\nprint(os.getpid())\n")
    monkeypatch.setattr(engine, "__file__", str(code / "src/sr_f3_ce/engine_recovery.py"))
    durable_json(tmp_path / "RUN_IDENTITY.json", {"stage": "CPU_REVISION_FIXTURE"})
    engine._run_segment(tmp_path, "continuous")
    before = (tmp_path / "engine/resume/continuous.PROCESS.json").read_bytes()
    durable_json(
        tmp_path / "plan/config/SR_F3_CE.json",
        revision_config({"generation": {"prefill": "{"}}, "c2"),
    )
    with pytest.raises(PermissionError, match="protocol identity changed"):
        engine._run_segment(tmp_path, "continuous")
    assert (tmp_path / "engine/resume/continuous.PROCESS.json").read_bytes() == before


def test_existing_engine_adam_intent_cannot_move_into_selected_revision(tmp_path, monkeypatch):
    from sr_f3_ce import orchestration

    monkeypatch.setattr(
        orchestration, "worker_binding", lambda *_: dict(job_id="CPU_TEST", claim_id="CPU_TEST")
    )
    engine.reserve_engine_step(
        tmp_path, dict(segment="continuous", step=1, arm="EXEC", microbatch_size=1)
    )
    original = (tmp_path / "engine/resume/optimizer_intents/01.json").read_bytes()
    durable_json(
        tmp_path / "plan/config/SR_F3_CE.json",
        revision_config({"generation": {"prefill": "{"}}, "c2"),
    )
    with pytest.raises(PermissionError, match="another protocol run"):
        engine._intent_rows(tmp_path)
    assert (tmp_path / "engine/resume/optimizer_intents/01.json").read_bytes() == original
