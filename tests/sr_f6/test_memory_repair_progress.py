"""Durable progress is bounded observation, never evidence of success or restart state."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_core.training import capture_rng, state_hash
from sr_f6_memory_repair import validation as v

_spec = importlib.util.spec_from_file_location(
    "validation_progress_fixture", Path(__file__).with_name("test_memory_repair_validation.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def writer(tmp_path):
    return v.ProgressWriter(tmp_path, contract_sha256="1" * 64, repair_revision="2" * 40)


def counts():
    return dict.fromkeys(v.ProgressWriter.COUNT_FIELDS, 0)


def test_latest_progress_bound_identity_and_never_complete(tmp_path):
    progress = writer(tmp_path)
    manager = SimpleNamespace(pack_count=3, arena_physical_bytes=32, private_answer="secret")
    manager.summary = lambda: pytest.fail("Progress must not inventory the model")
    for i, stage in enumerate(("baseline_forward", "row_forward", "measurement_finished")):
        progress(stage, None, 128 if i == 2 else 0, counts(), manager)
        path = tmp_path / "PROGRESS.json"
        record = json.loads(path.read_text())
        assert record["sequence"] == i
        assert record["status"] == "NOT_COMPLETE_NOT_RESUMABLE"
        assert record["contract_sha256"] == "1" * 64
        assert record["repair_revision"] == "2" * 40
        assert record["storage"] == {"pack_count": 3, "arena_physical_bytes": 32}
        assert record["elapsed_seconds"] >= 0
        assert record["previous_stage_seconds"] >= 0
        assert path.stat().st_size <= 32768
        assert path.stat().st_mode & 0o777 == 0o600
    assert record["stage_seconds"].keys() == {"baseline_forward", "row_forward"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["PROGRESS.json"]


@pytest.mark.parametrize("replacement", ["existing", "symlink", "later_symlink", "hardlink"])
def test_no_unowned_progress_overwrite(tmp_path, replacement):
    progress = writer(tmp_path)
    target = tmp_path / "PROGRESS.json"
    protected = tmp_path / "protected"
    protected.write_text("preserve")
    if replacement == "existing":
        target.write_text("preserve")
    elif replacement == "symlink":
        target.symlink_to(protected)
    else:
        progress("baseline_forward", 0, 0, counts())
        if replacement == "later_symlink":
            target.unlink()
            target.symlink_to(protected)
        else:
            (tmp_path / "alias").hardlink_to(target)
    before = target.read_bytes()
    with pytest.raises((PermissionError, FileExistsError)):
        progress("row_forward", 0, 0, counts())
    assert target.read_bytes() == before
    assert protected.read_text() == "preserve"


def test_directory_replacement_is_rejected(tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    progress = writer(directory)
    directory.rename(tmp_path / "original")
    directory.mkdir()
    with pytest.raises(PermissionError, match="directory changed"):
        progress("baseline_forward", 0, 0, counts())
    assert not list(directory.iterdir())


@pytest.mark.parametrize("bad", ["stage", "row", "count", "generated", "updated"])
def test_progress_rejects_unregistered_content(tmp_path, bad):
    progress = writer(tmp_path)
    stage, row, call_counts = "row_forward", 0, counts()
    if bad == "stage":
        stage = "COMPLETE"
    elif bad == "row":
        row = 128
    elif bad == "count":
        call_counts["private_answer"] = 1
    elif bad == "generated":
        call_counts["generated_responses"] = 1
    else:
        call_counts["optimizer_updates"] = 1
    with pytest.raises(PermissionError):
        progress(stage, row, 0, call_counts)
    assert not list(tmp_path.iterdir())


def test_progress_observation_preserves_original_calls_and_data(tmp_path):
    runtime, optimizer, _, rows, credits, originals = _fixture.setup(tmp_path)
    originals_before = copy.deepcopy((rows, credits, originals))
    observations = []
    output = tmp_path / "progress"
    output.mkdir()
    progress = writer(output)

    def observe(stage, row, completed, call_counts, manager):
        progress(stage, row, completed, call_counts, manager)
        observations.append(json.loads((output / "PROGRESS.json").read_text()))

    evidence, tensors = v.measure_fixture(runtime, rows, credits, originals, progress=observe)
    assert (rows, credits, originals) == originals_before
    assert runtime.calls == [rows[0]["row_id"]] + [r["row_id"] for r in rows]
    assert len([o for o in observations if o["stage"] == "row_finished"]) == 128
    assert len([o for o in observations if o["stage"].startswith("diagnostic_")]) == 192
    assert observations[-1]["counts"] == evidence["counts"]
    assert observations[-1]["completed_rows"] == 128
    assert observations[-1]["status"] == "NOT_COMPLETE_NOT_RESUMABLE"
    assert not optimizer.state
    assert v.recompute(evidence, tensors, originals) == evidence["comparisons"]


@pytest.mark.parametrize("failed_stage", ["row_backward", "restoration", "measurement_finished"])
def test_failed_progress_write_preserves_partial_and_restores(tmp_path, failed_stage):
    runtime, optimizer, _, rows, credits, originals = _fixture.setup(tmp_path)
    before, rng = state_hash(runtime.model.state_dict()), state_hash(capture_rng())

    def failed_write(stage, *_):
        if stage == failed_stage:
            raise OSError("progress storage unavailable")

    with pytest.raises(OSError, match="storage unavailable") as failure:
        v.measure_fixture(runtime, rows, credits, originals, progress=failed_write)
    evidence, tensors = failure.value.validation_partial
    assert evidence["restoration"]["adapter_restored"]
    assert evidence["restoration"]["rng_restored"]
    assert "baseline" in tensors
    assert state_hash(runtime.model.state_dict()) == before
    assert state_hash(capture_rng()) == rng
    assert runtime.model.weight.grad is None
    assert not optimizer.state
    assert "scope" not in evidence
