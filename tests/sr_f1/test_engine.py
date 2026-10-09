from collections import Counter

import pytest

from sr_f1 import engine


def test_engine_uses_balanced_engine_only_pool_and_one_16_by_8_batch_per_step(
    tmp_path, monkeypatch
):
    tasks = {
        f"{family}-{chart}-{i}": dict(pool="ENGINE", family=family, chart=chart)
        for family in ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")
        for chart in ("line", "grouped_bar")
        for i in range(16)
    }
    tasks["forbidden_test"] = {"pool": "TEST_ID", "family": "CROSS", "chart": "line"}
    monkeypatch.setattr(engine, "load_tasks", lambda root: tasks)
    first = engine.engine_schedule(tmp_path)
    assert first == engine.engine_schedule(tmp_path)
    assert len(first) == 64 and len({r["qid"] for r in first}) == 64
    for step in range(1, 5):
        selected = [r for r in first if r["step"] == step]
        assert len(selected) == 16 and sum(len(r["rollout_seeds"]) for r in selected) == 128
        quotas = Counter((tasks[r["qid"]]["family"], tasks[r["qid"]]["chart"]) for r in selected)
        assert set(quotas.values()) == {2}
    assert engine.engine_run("natural")["arms"] == ["DEC", "GATE", "DEC", "GATE"]


def test_missing_actual_gpu_evidence_cannot_publish_engine_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.compare_engine(tmp_path, "natural")
    assert not (tmp_path / "ENGINE_PROBABILITY_GRADIENT_RESUME.json").exists()


def test_interrupted_engine_trace_is_preserved_before_repeating_all_registered_boundaries(tmp_path):
    directory = tmp_path / "engineering/engine/natural/continuous"
    directory.mkdir(parents=True)
    (directory / "raw.json").write_text('{"actual_failed_attempt":true}')
    archive = engine._preserve_interrupted_mode(tmp_path, "natural")
    assert not directory.exists()
    assert (archive / "continuous/raw.json").read_text() == '{"actual_failed_attempt":true}'
    assert (archive / "INTERRUPTION_PRESERVED.json").exists()


def test_engine_checkpoint_marker_retains_valid_immutable_paths_after_track_archive(tmp_path):
    from sr_f1.runtime import atomic_json, file_hash, read_json

    original = tmp_path / "engineering/engine/natural/continuous/checkpoints"
    original.mkdir(parents=True)
    (original / "real-state.pt").write_bytes(b"fixture-checkpoint-bytes")
    sha = file_hash(original / "real-state.pt")
    atomic_json(original / "LATEST.json", {"path": "real-state.pt", "sha256": sha, "step": 1})
    result = engine._engine_recovery_snapshot(
        tmp_path,
        original / "LATEST.json",
        {
            "reason": "PREEMPTION",
            "metadata": {"full_state": True, "identity_verified": True, "next_update": 2},
        },
    )
    engine._preserve_interrupted_mode(tmp_path, "natural")
    assert all((tmp_path / name).is_file() for name in result["artifacts"])
    assert file_hash(tmp_path / result["artifacts"][1]) == sha
    assert read_json(tmp_path / result["artifacts"][0])["path"] == "state.pt"
