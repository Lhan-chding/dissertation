import json
import os
import sys
import types
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

from sr_f1 import baseline_parallel as parallel
from sr_f1.evaluation import BASELINE, evaluate_slots, read_jsonl

PACKAGE = Path(__file__).resolve().parents[2] / "docs/sr_f1/package"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FakeRuntime:
    identity: ClassVar = {"model_hash": "frozen", "model_id": BASELINE, "step": 0}
    adapter_identity = "frozen-adapter"

    def __init__(self, stop=None, fail=False):
        self.calls = []
        self.stop = stop
        self.fail = fail

    def generate(self, **kwargs):
        self.calls.append(kwargs["seed"])
        row = dict(
            raw_text="{}", tokens=[kwargs["seed"] % 997 + 1], old_logprobs=[-1.0], truncated=False
        )
        kwargs["on_completion"](row)
        if self.stop is not None:
            self.stop["requested"] = True
        if self.fail:
            raise RuntimeError("synthetic child failure after durable raw")
        return row


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    slots = parallel.baseline_slots(PACKAGE)[:12]
    monkeypatch.setattr(parallel, "baseline_slots", lambda _root: list(slots))
    inputs = {
        slot["qid"]: {"qid": slot["qid"], "text": "q", "plain_text": "q", "image_file": None}
        for slot in slots
    }
    write_rows(tmp_path / parallel.INPUT_FILES[0], [])
    write_rows(tmp_path / parallel.INPUT_FILES[1], inputs.values())
    write_rows(tmp_path / parallel.INPUT_FILES[2], [])
    write(tmp_path / parallel.INPUT_FILES[3], [])
    write(tmp_path / "COMMON_START.json", {"model_id": BASELINE, "step": 0})
    write(tmp_path / "EXECUTION_FREEZE.json", {"frozen": True})
    monkeypatch.setenv("SR_F1_TASK_ID", "BASELINE")
    monkeypatch.setenv("SR_F1_ATTEMPT_ID", "BASELINE_attempt0000")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    return tmp_path, slots, inputs


@pytest.mark.parametrize("count", range(1, 6))
def test_all_original_slots_partition_once_with_unchanged_seeds_and_no_test(count):
    slots = parallel.baseline_slots(PACKAGE)
    shards = [parallel.shard_slots(slots, rank, count) for rank in range(count)]
    union = {row["slot_id"]: row for shard in shards for row in shard}
    assert len(slots) == sum(map(len, shards)) == len(union) == 6784
    assert union == {row["slot_id"]: row for row in slots}
    assert not any(row["pool"].startswith("TEST") for row in union.values())
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("rank,count", [(0, 0), (0, 6), (-1, 3), (3, 3), (True, 3), (0, True)])
def test_invalid_partition_is_rejected(rank, count):
    with pytest.raises(ValueError):
        parallel.shard_slots([], rank, count)


def test_serial_and_reordered_shards_are_identical_and_resume_does_not_generate(fixture):
    root, slots, inputs = fixture
    serial = FakeRuntime()
    serial_path = root / "serial.jsonl"
    evaluate_slots(serial, slots, inputs, {}, root, serial_path)
    outputs, calls = [], []
    for rank in (2, 0, 1):
        runtime = FakeRuntime()
        shard = parallel.shard_slots(slots, rank, 3)
        path = parallel.raw_path(root, rank, 3)
        evaluate_slots(runtime, shard, inputs, {}, root, path)
        before = path.read_bytes()
        evaluate_slots(runtime, shard, inputs, {}, root, path)
        assert len(runtime.calls) == len(shard)
        assert path.read_bytes() == before
        calls.extend(runtime.calls)
        outputs.extend(read_jsonl(path))
    assert sorted(calls) == sorted(serial.calls)
    assert sorted(outputs, key=lambda row: row["slot_id"]) == sorted(
        read_jsonl(serial_path), key=lambda row: row["slot_id"]
    )
    audit = parallel.validate_baseline_outputs(root, slots, 3)
    assert audit["generated_slots"] == len(slots)


def fill(root, slots, inputs, count=3):
    for rank in range(count):
        evaluate_slots(
            FakeRuntime(),
            parallel.shard_slots(slots, rank, count),
            inputs,
            {},
            root,
            parallel.raw_path(root, rank, count),
        )


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "wrong_shard", "input", "model", "extra"]
)
def test_parent_coverage_rejects_missing_corrupt_and_duplicate_records(fixture, failure):
    root, slots, inputs = fixture
    fill(root, slots, inputs)
    path = parallel.raw_path(root, 0, 3)
    rows = list(read_jsonl(path))
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[1] = rows[0]
    elif failure == "wrong_shard":
        rows[0] = next(read_jsonl(parallel.raw_path(root, 1, 3)))
    elif failure == "input":
        rows[0]["input_hash"] = "0" * 64
    elif failure == "model":
        rows[0]["model_identity"]["model_hash"] = "changed"
    elif failure == "extra":
        write_rows(root / "raw/evaluation" / f"{BASELINE}_step0_baseline.jsonl", rows)
    write_rows(path, rows)
    with pytest.raises(ValueError):
        parallel.validate_baseline_outputs(root, slots, 3)


def test_manifest_prevents_topology_or_plan_or_input_change_and_serial_resampling(fixture):
    root, _slots, _inputs = fixture
    before, _ = parallel.prepare_manifest({"plan": "frozen"}, root, 3)
    assert parallel.prepare_manifest({"plan": "frozen"}, root, 3)[0] == before
    with pytest.raises(PermissionError, match="identity changed"):
        parallel.prepare_manifest({"plan": "frozen"}, root, 4)
    with pytest.raises(PermissionError, match="identity changed"):
        parallel.prepare_manifest({"plan": "changed"}, root, 3)
    path = root / "raw/evaluation" / f"{BASELINE}_step0_baseline.jsonl"
    write_rows(path, [])
    with pytest.raises(PermissionError, match="serial baseline"):
        parallel.prepare_manifest({"plan": "frozen"}, root, 3)


def install_fake_children(
    monkeypatch, root, *, fail_rank=None, stop_after_one=False, tamper_receipt=False
):
    children, runtimes = [], []

    def load(*_args, **_kwargs):
        runtime = FakeRuntime(fail=int(os.environ["SRF1_BASELINE_RANK"]) == fail_rank)
        runtime.identity = {
            **runtime.identity,
            "hardware": {
                "cuda_visible_device_count": 1,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "cuda_uuid": "unique-" + os.environ["CUDA_VISIBLE_DEVICES"],
            },
        }
        if stop_after_one:
            original = runtime.generate

            def generate(**kwargs):
                result = original(**kwargs)
                _attempt, _job, directory = parallel._attempt(root)
                parallel._request_stop(directory, "PREEMPTION")
                return result

            runtime.generate = generate
        runtimes.append(runtime)
        return runtime

    monkeypatch.setitem(
        sys.modules, "sr_f1.runtime", types.SimpleNamespace(load_for_evaluation=load)
    )

    class Child:
        def __init__(self, cmd, *, env, stdout, stderr):
            self.pid = os.getpid()
            self.rank = int(cmd[-1])
            self.code = 0
            children.append(self)
            with (
                patch.dict(os.environ, env, clear=True),
                patch.object(parallel.os, "getppid", return_value=os.getpid()),
            ):
                try:
                    parallel.run_child(root, self.rank)
                except RuntimeError:
                    self.code = 1
                if tamper_receipt and self.rank == 0:
                    _attempt, _job, directory = parallel._attempt(root)
                    path = directory / "rank00_RESULT.json"
                    receipt = json.loads(path.read_text())
                    receipt["raw_sha256"] = "0" * 64
                    write(path, receipt)

        def poll(self):
            return self.code

    monkeypatch.setattr(parallel.subprocess, "Popen", Child)
    return children, runtimes


def test_parent_and_children_complete_exact_coverage_with_independent_gpu_masks(
    fixture, monkeypatch
):
    root, slots, _inputs = fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    children, runtimes = install_fake_children(monkeypatch, root)
    result = parallel.run_parallel_baseline({"frozen": True}, root, 3)
    assert result["status"] == "COMPLETE"
    assert result["metadata"]["generated_slots"] == len(slots)
    assert sum(len(runtime.calls) for runtime in runtimes) == len(slots)
    assert len(children) == 3
    assert not (root / "raw/evaluation" / f"{BASELINE}_step0_baseline.jsonl").exists()
    for rank, device in enumerate(("GPU-a", "GPU-b", "GPU-c")):
        path = (
            root
            / parallel.DIRECTORY
            / "attempts/BASELINE_attempt0000"
            / f"rank{rank:02d}_PROCESS.json"
        )
        assert json.loads(path.read_text())["cuda_visible_devices"] == device


def test_parent_boundary_checkpoint_then_new_attempt_resume_retains_raw(fixture, monkeypatch):
    root, slots, _inputs = fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    _children, runtimes = install_fake_children(monkeypatch, root, stop_after_one=True)
    result = parallel.run_parallel_baseline({"frozen": True}, root, 3)
    assert result["status"] == "CHECKPOINTED"
    assert result["metadata"]["full_state"] and result["metadata"]["identity_verified"]
    assert result["metadata"]["generated_slots"] == 1
    prefix = parallel.raw_path(root, 0, 3).read_bytes()
    assert sum(len(runtime.calls) for runtime in runtimes) == 1
    monkeypatch.setenv("SR_F1_ATTEMPT_ID", "BASELINE_attempt0001")
    _children, runtimes = install_fake_children(monkeypatch, root)
    result = parallel.run_parallel_baseline({"frozen": True}, root, 3)
    assert result["status"] == "COMPLETE"
    assert parallel.raw_path(root, 0, 3).read_bytes().startswith(prefix)
    assert sum(len(runtime.calls) for runtime in runtimes) == len(slots) - 1


def test_child_failure_is_preserved_and_never_falsely_checkpointed(fixture, monkeypatch):
    root, _slots, _inputs = fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    children, _runtimes = install_fake_children(monkeypatch, root, fail_rank=1)
    with pytest.raises(RuntimeError, match="child failed"):
        parallel.run_parallel_baseline({}, root, 3)
    assert all(child.poll() is not None for child in children)
    assert parallel.raw_path(root, 1, 3).stat().st_size > 0
    directory = root / parallel.DIRECTORY / "attempts/BASELINE_attempt0000"
    assert json.loads((directory / "rank01_RESULT.json").read_text())["status"] == "FAILED"
    assert not (directory / "COVERAGE.json").exists()


def test_child_receipt_corruption_blocks_parent_completion(fixture, monkeypatch):
    root, _slots, _inputs = fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    install_fake_children(monkeypatch, root, tamper_receipt=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        parallel.run_parallel_baseline({}, root, 3)


def test_poll_failure_requests_only_this_attempt_children_stop(tmp_path, monkeypatch):
    stop = tmp_path / "STOP_REQUEST.json"

    class Child:
        def __init__(self, status):
            self.status = status

        def poll(self):
            return self.status if self.status is not None else (0 if stop.exists() else None)

    monkeypatch.setattr(parallel.time, "sleep", lambda _value: None)
    assert parallel._wait_children([Child(1), Child(None)], tmp_path, {"requested": False}) == [
        1,
        0,
    ]
    assert json.loads(stop.read_text())["reason"] == "CHILD_FAILED"


def test_stop_boundary_is_truthy_and_observes_parent_marker_and_signal(tmp_path):
    import signal

    stop = tmp_path / "stop.json"
    with parallel._stop_boundary(stop) as boundary:
        assert bool(boundary) and not boundary["requested"]
        write(stop, {"stop": True})
        assert boundary["requested"]
    with parallel._stop_boundary() as boundary:
        assert not boundary["requested"]
        signal.raise_signal(signal.SIGUSR1)
        assert boundary["requested"]


def test_failed_parent_waits_then_terminates_only_owned_unresponsive_children(
    tmp_path, monkeypatch
):
    clock = iter([0, 301, 301, 361, 361, 362])
    monkeypatch.setattr(parallel.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(parallel.time, "sleep", lambda _value: None)

    class Child:
        pid = 9101
        status = None
        terminated = False
        waited = False

        def poll(self):
            return self.status

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.status = -9

        def wait(self):
            self.waited = True
            return self.status

    child = Child()
    assert parallel._wait_children(
        [child], tmp_path, {"requested": True}, parent_exception=True
    ) == [-9]
    assert child.terminated and child.waited
    receipt = json.loads((tmp_path / "FAILED_CHILD_CLEANUP.json").read_text())
    assert receipt["child_pids"] == [child.pid]
    assert receipt["reason"] == "PARENT_EXCEPTION"
