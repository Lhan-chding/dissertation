"""Complete-state persistence and a real subprocess CPU 4 versus 2+2 audit."""

import copy
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mm_core.training import state_hash, trainable_state
from sr_f3_ce.checkpoint import (
    capture_named_rng,
    checkpoint_state,
    commit_checkpoint,
    compare_recovery_states,
    load_checkpoint,
    read_checkpoint,
    restore_named_rng,
)
from sr_f3_ce.training import fresh_optimizer, update

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "sr_f3_ce_training_fixtures", Path(__file__).with_name("test_training.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
ToyRuntime = _fixture.ToyRuntime
task_records = _fixture.task_records
supports = _fixture.supports
model_state = _fixture.model_state

IDENTITY = {"run_id": "CPU_ENGINE_NOT_GPU_GATE", "arm": "EXEC", "stage": "ENGINE"}
STREAM = state_hash("frozen ENGINE input stream")
SAMPLING = state_hash("T1 top_p1 top_k0")
SOURCES = {"base": state_hash("CPU toy only"), "code": state_hash("test source")}


def make_state(runtime, optimizer, scheduler, reference, step, *, statistics=None, streams=None):
    return checkpoint_state(
        runtime,
        optimizer,
        scheduler,
        run_identity=IDENTITY,
        step=step,
        reference=reference,
        stream_hash=STREAM,
        sampling_hash=SAMPLING,
        source_hashes=SOURCES,
        used_task_qids=[f"task-{index}" for index in range(step * 16)],
        used_support_qids=[f"support-{index}" for index in range(step * 4)],
        committed_records_hash=state_hash([f"cache-{index}" for index in range(1, step + 1)]),
        microbatch_size=4,
        step_statistics=statistics or {},
        rng_streams=streams,
    )


def one_step(runtime, optimizer, scheduler, reference, step):
    return update(
        runtime,
        optimizer,
        scheduler,
        task_records(runtime, offset=(step - 1) * 16),
        arm="EXEC",
        reference=reference,
        support_records=supports(offset=(step - 1) * 4),
        step=step,
    )


def restore(directory, runtime, optimizer, scheduler, reference, **kwargs):
    return load_checkpoint(
        directory,
        runtime,
        optimizer,
        scheduler,
        run_identity=IDENTITY,
        reference=reference,
        stream_hash=STREAM,
        sampling_hash=SAMPLING,
        source_hashes=SOURCES,
        **kwargs,
    )


def test_checkpoint_is_atomic_idempotent_complete_and_immutable(tmp_path):
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    state = make_state(runtime, optimizer, scheduler, reference, 0)
    receipt = commit_checkpoint(tmp_path, state)
    assert commit_checkpoint(tmp_path, state) == receipt
    actual, read_receipt = read_checkpoint(tmp_path)
    assert receipt == read_receipt
    assert state_hash(actual) == state_hash(state)
    assert set(receipt["field_hashes"]) == set(state)
    changed = copy.deepcopy(state)
    changed["step_statistics"] = {"different": True}
    with pytest.raises(RuntimeError, match="conflicting"):
        commit_checkpoint(tmp_path, changed)
    assert read_checkpoint(tmp_path)[1] == receipt
    assert len(list(tmp_path.glob("step-00-*.pt"))) == 2  # preserve uncommitted incident bytes


def test_stale_latest_pointer_recovers_from_immutable_marker(tmp_path):
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    old = commit_checkpoint(tmp_path, make_state(runtime, optimizer, scheduler, reference, 0))
    metrics = one_step(runtime, optimizer, scheduler, reference, 1)
    latest = commit_checkpoint(
        tmp_path, make_state(runtime, optimizer, scheduler, reference, 1, statistics=metrics)
    )
    (tmp_path / "LATEST.json").write_text(json.dumps(old))
    new_runtime = ToyRuntime()
    new_optimizer, new_scheduler = fresh_optimizer(new_runtime.model)
    restored = restore(tmp_path, new_runtime, new_optimizer, new_scheduler, reference)
    assert restored["committed_logical_step"] == 1
    assert json.loads((tmp_path / "LATEST.json").read_text()) == latest
    assert state_hash(new_optimizer.state_dict()) == state_hash(optimizer.state_dict())


def test_recovery_statistics_keep_losses_and_cost_counts_strict_but_exclude_wall_times():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    metrics = one_step(runtime, optimizer, scheduler, reference, 1)
    state = make_state(runtime, optimizer, scheduler, reference, 1, statistics=metrics)
    replay = copy.deepcopy(state)
    replay["step_statistics"]["training_seconds"] += 10
    replay["step_statistics"]["attempts"][0]["wall_seconds"] += 10
    replay["step_statistics"]["total_cost"]["diagnostic_backward_host_wall_seconds"] += 10
    comparison = compare_recovery_states(state, replay)
    assert comparison["all_logical_fields_exact"]
    assert comparison["raw_step_statistics_exact"] is False
    assert comparison["comparisons"]["step_statistics"] is True
    for key in ("loss", "sampler_train_mean_absolute_difference"):
        changed = copy.deepcopy(replay)
        changed["step_statistics"][key] += 0.001
        comparison = compare_recovery_states(state, changed)
        assert comparison["comparisons"]["step_statistics"] is False
        assert comparison["all_logical_fields_exact"] is False
    changed = copy.deepcopy(replay)
    changed["step_statistics"]["total_cost"]["optimizer_step_calls_attempted"] += 1
    assert compare_recovery_states(state, changed)["all_logical_fields_exact"] is False


@pytest.mark.parametrize(
    "tamper", ["bytes", "path", "field_hash", "reference", "source", "parameter_inventory"]
)
def test_corruption_or_identity_mismatch_does_not_mutate_live_state(tmp_path, tamper):
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    state = make_state(runtime, optimizer, scheduler, reference, 0)
    if tamper == "parameter_inventory":
        state["parameters"]["extra"] = torch.tensor(1.0)
    receipt = commit_checkpoint(tmp_path, state)
    if tamper == "bytes":
        with (tmp_path / receipt["path"]).open("ab") as handle:
            handle.write(b"tamper")
    elif tamper in ("path", "field_hash"):
        receipt = copy.deepcopy(receipt)
        if tamper == "path":
            receipt["path"] = "../escape.pt"
        else:
            receipt["field_hashes"]["parameters"] = "0" * 64
        (tmp_path / "commit-00.json").write_text(json.dumps(receipt))
    target_reference = copy.deepcopy(reference)
    if tamper == "reference":
        target_reference["theta"].add_(1.0)
    target_sources = SOURCES if tamper != "source" else {**SOURCES, "base": "0" * 64}
    before = model_state(runtime, optimizer, scheduler)
    with pytest.raises(PermissionError):
        load_checkpoint(
            tmp_path,
            runtime,
            optimizer,
            scheduler,
            run_identity=IDENTITY,
            reference=target_reference,
            stream_hash=STREAM,
            sampling_hash=SAMPLING,
            source_hashes=target_sources,
        )
    assert model_state(runtime, optimizer, scheduler) == before


def test_cursors_used_ids_optimizer_steps_and_science_pool_disjointness_are_checked():
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    state = make_state(runtime, optimizer, scheduler, reference, 0)
    with pytest.raises(PermissionError, match=r"Adam|scheduler"):
        make_state(runtime, optimizer, scheduler, reference, 1)
    from sr_f3_ce.checkpoint import _validate_state

    changed = copy.deepcopy(state)
    changed["cursors"]["task"] = 16
    with pytest.raises(PermissionError, match="cursors"):
        _validate_state(changed)
    one_step(runtime, optimizer, scheduler, reference, 1)
    changed = make_state(runtime, optimizer, scheduler, reference, 1)
    changed["run_identity"]["stage"] = "SCIENCE"
    changed["used_support_qids"][0] = "task-0"
    with pytest.raises(PermissionError, match="overlap"):
        _validate_state(changed)
    changed = make_state(runtime, optimizer, scheduler, reference, 1)
    changed["run_identity"]["stage"] = "SCIENCE"
    changed["step_statistics"] = {"engineering_zero_aux": True, "scientific_recipe_eligible": False}
    with pytest.raises(PermissionError, match="Engineering-only"):
        _validate_state(changed)


def test_named_rng_stream_roundtrip_and_wrong_stream_inventory_rollback(tmp_path):
    streams = {
        "task": random.Random(1),
        "support": np.random.default_rng(2),
        "monitor": torch.Generator().manual_seed(3),
        "test": random.Random(4),
    }
    before = capture_named_rng(streams)
    expected = [
        streams["task"].random(),
        float(streams["support"].random()),
        float(torch.rand((), generator=streams["monitor"])),
    ]
    restore_named_rng(before, streams)
    actual = [
        streams["task"].random(),
        float(streams["support"].random()),
        float(torch.rand((), generator=streams["monitor"])),
    ]
    assert expected == actual
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    commit_checkpoint(
        tmp_path, make_state(runtime, optimizer, scheduler, reference, 0, streams=streams)
    )
    live_before = model_state(runtime, optimizer, scheduler)
    with pytest.raises(PermissionError, match="inventory"):
        restore(tmp_path, runtime, optimizer, scheduler, reference, rng_streams={})
    assert model_state(runtime, optimizer, scheduler) == live_before


def test_new_process_two_plus_two_equals_continuous_four_with_six_updates(tmp_path):
    """512 generated-cache rows plus 256 replayed rows; no resampling in child."""
    runtime = ToyRuntime()
    optimizer, scheduler = fresh_optimizer(runtime.model)
    reference = trainable_state(runtime.model)
    cache = {}
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    split.mkdir()
    for step in range(1, 5):
        records = task_records(runtime, offset=(step - 1) * 16)
        support = supports(offset=(step - 1) * 4)
        cache[step] = {"records": records, "support": support}
        metrics = update(
            runtime,
            optimizer,
            scheduler,
            records,
            arm="EXEC",
            reference=reference,
            support_records=support,
            step=step,
        )
        state = make_state(runtime, optimizer, scheduler, reference, step, statistics=metrics)
        if step in (2, 4):
            receipt = commit_checkpoint(continuous, state)
        if step == 2:
            for name in (receipt["path"], "commit-02.json", "LATEST.json"):
                shutil.copyfile(continuous / name, split / name)
    torch.save(cache, tmp_path / "cache.pt")
    cache_hash = state_hash(cache)
    script = tmp_path / "resume_cpu_engine.py"
    script.write_text("""
import runpy
import sys
from pathlib import Path
import torch
torch.set_num_threads(1)
fixture = runpy.run_path(sys.argv[1])
root = Path(sys.argv[2])
runtime = fixture["ToyRuntime"]()
optimizer, scheduler = fixture["fresh_optimizer"](runtime.model)
reference = fixture["trainable_state"](runtime.model)
fixture["restore"](root / "split", runtime, optimizer, scheduler, reference)
cache = torch.load(root / "cache.pt", weights_only=False)
for step in (3, 4):
    metrics = fixture["update"](runtime, optimizer, scheduler, cache[step]["records"],
        arm="EXEC", reference=reference, support_records=cache[step]["support"], step=step)
state = fixture["make_state"](runtime, optimizer, scheduler, reference, 4, statistics=metrics)
fixture["commit_checkpoint"](root / "split", state)
(root / "child_receipt.json").write_text(__import__("json").dumps({
    "pid": __import__("os").getpid(), "physical_updates": 2,
    "new_rollouts": 0, "replayed_rollouts": 256, "cache_hash": fixture["state_hash"](cache)}))
""")
    env = dict(os.environ)
    src = Path(__file__).resolve().parents[2] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, str(script), str(Path(__file__).resolve()), str(tmp_path)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    child = json.loads((tmp_path / "child_receipt.json").read_text())
    assert child["pid"] != os.getpid()
    assert child["cache_hash"] == cache_hash
    assert 4 + child["physical_updates"] == 6
    assert sum(len(row["records"]) for row in cache.values()) + child["new_rollouts"] == 512
    assert child["replayed_rollouts"] == 256
    comparison = compare_recovery_states(read_checkpoint(continuous)[0], read_checkpoint(split)[0])
    assert comparison["all_logical_fields_exact"], comparison["comparisons"]
    assert comparison["actual_gpu_execution_confirmed"] is False
    assert comparison["sensitivity_is_automatic_pass"] is False
