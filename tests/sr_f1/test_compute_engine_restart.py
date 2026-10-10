"""A previous complete ENGINE is never reused to qualify changed compute code."""

import json

import pytest

from sr_f1 import engine
from sr_f1.runtime import file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def trace(root, revision, *, complete=True, comparison=True):
    mode = root / "engineering/engine/natural"
    for track in ("continuous", "split"):
        identity = {} if revision is None else {"compute_parallel": {"repair_sha256": revision}}
        write(mode / track / "RUN_MANIFEST.json", {"run_identity": {"training_identity": identity}})
        write(mode / track / "raw.json", {"original": track})
    if complete:
        write(mode / "continuous/CONTINUOUS_COMPLETE.json", {"old": True})
        write(mode / "split/FIRST_COMPLETE.json", {"old": True})
        write(mode / "split/RESUME_COMPLETE.json", {"old": True})
    if comparison:
        write(mode / "COMPARISON.json", {"artifact_hashes": {}, "stress_allowed": False})
    return mode


@pytest.mark.parametrize("revision", [None, "0" * 64])
def test_complete_old_trials_comparison_and_zero_reuse_markers_are_preserved(tmp_path, revision):
    write(tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", {"repair": "new"})
    mode = trace(tmp_path, revision)
    markers = [
        tmp_path / "engineering/engine/ZERO_UPDATE_REUSE_ACTIVATED.json",
        tmp_path / "engineering/engine/ZERO_UPDATE_REUSE_PROCESS.json",
    ]
    for marker in markers:
        write(marker, {"consumed": True})
    before = {str(p.relative_to(mode)): p.read_bytes() for p in mode.rglob("*") if p.is_file()}
    archived = engine._preserve_interrupted_mode(tmp_path, "natural")
    assert not mode.exists()
    assert archived.parent == tmp_path / "engineering/engine_interrupted"
    for name, content in before.items():
        assert (archived / name).read_bytes() == content
    assert all(json.loads(p.read_text()) == {"consumed": True} for p in markers)
    receipt = json.loads((archived / "COMPUTE_REPAIR_RESTART_PRESERVED.json").read_text())
    assert receipt["repeat_scope"] == "entire_engine_mode"
    assert receipt["current_compute_repair_sha256"] == file_hash(
        tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"
    )
    assert engine._preserve_previous_compute_mode(tmp_path, "natural") is None


def test_current_complete_comparison_is_not_archived(tmp_path):
    repair = tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"
    write(repair, {"repair": "new"})
    mode = trace(tmp_path, file_hash(repair))
    assert engine._preserve_interrupted_mode(tmp_path, "natural") is None
    assert (mode / "COMPARISON.json").exists()


def test_current_partial_trial_retains_existing_full_restart_semantics(tmp_path):
    repair = tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"
    write(repair, {"repair": "new"})
    trace(tmp_path, file_hash(repair), complete=False, comparison=False)
    archive = engine._preserve_interrupted_mode(tmp_path, "natural")
    assert (archive / "INTERRUPTION_PRESERVED.json").exists()
    assert not (archive / "COMPUTE_REPAIR_RESTART_PRESERVED.json").exists()


def test_run_archives_old_comparison_before_considering_reuse(tmp_path, monkeypatch):
    write(tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", {"repair": "new"})
    mode = trace(tmp_path, None)
    monkeypatch.setattr(engine, "verified_adapter", lambda *a: {})

    def schedule(root):
        assert not mode.exists()
        assert len(list((tmp_path / "engineering/engine_interrupted").glob("natural-*"))) == 1
        raise RuntimeError("stop before model execution")

    monkeypatch.setattr(engine, "engine_schedule", schedule)
    with pytest.raises(RuntimeError, match="stop before model"):
        engine._run_engine({}, tmp_path, None)


def test_old_global_gate_fails_closed_without_archiving_or_overwriting(tmp_path, monkeypatch):
    write(tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json", {"repair": "new"})
    mode = trace(tmp_path, None)
    gate = tmp_path / "ENGINE_PROBABILITY_GRADIENT_RESUME.json"
    write(gate, {"artifact_hashes": {}, "stress": None})
    old_gate = gate.read_bytes()
    monkeypatch.setattr(engine, "verified_adapter", lambda *a: {})
    with pytest.raises(PermissionError, match="previous compute revision"):
        engine._run_engine({}, tmp_path, None)
    assert gate.read_bytes() == old_gate
    assert mode.exists()
    assert not (tmp_path / "engineering/engine_interrupted").exists()


def test_current_global_gate_requires_both_current_revision_manifests(tmp_path, monkeypatch):
    repair = tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"
    write(repair, {"repair": "new"})
    mode = trace(tmp_path, file_hash(repair))
    write(
        tmp_path / "ENGINE_PROBABILITY_GRADIENT_RESUME.json",
        {"artifact_hashes": {}, "stress": None},
    )
    monkeypatch.setattr(engine, "verified_adapter", lambda *a: {})
    assert engine._run_engine({}, tmp_path, None)["status"] == "COMPLETE"
    (mode / "split/RUN_MANIFEST.json").unlink()
    with pytest.raises(PermissionError, match="previous compute revision"):
        engine._run_engine({}, tmp_path, None)
