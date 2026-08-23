from __future__ import annotations

import json
from pathlib import Path

import pytest

from compensability.study_c3 import config_runtime
from compensability.study_c3.io import sha256_file


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_c2_evaluation_provenance_rejects_fiber_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw.jsonl"
    fibers = tmp_path / "fibers.jsonl"
    training = tmp_path / "training.json"
    summary = tmp_path / "summary.json"
    manifest = tmp_path / "manifest.json"
    raw.write_text("{}\n", encoding="utf-8")
    fibers.write_text("{}\n", encoding="utf-8")
    _write_json(training, {"status": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE"})
    _write_json(
        summary,
        {
            "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
            "raw_row_count": 5632,
            "rollout_seed_base": 1,
            "rollout_seed_algorithm": "phase5_rollout_seed_v1",
        },
    )
    _write_json(
        manifest,
        {
            "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
            "raw_row_count": 5632,
            "evaluation_scene_count": 176,
            "evaluation_pair_count": 88,
            "sampled_rollouts": 16,
            "raw_rows_sha256": sha256_file(raw),
            "summary_sha256": sha256_file(summary),
            "fiber_rows_sha256": "0" * 64,
            "training_pair_manifest_sha256": sha256_file(training),
            "model_snapshot_sha256": config_runtime.MODEL_SNAPSHOT_SHA256,
        },
    )
    monkeypatch.setattr(config_runtime, "C2_EVALUATION_RAW", raw)
    monkeypatch.setattr(config_runtime, "C2_EVALUATION_SUMMARY", summary)
    monkeypatch.setattr(config_runtime, "C2_EVALUATION_MANIFEST", manifest)
    monkeypatch.setattr(config_runtime, "C2_FIBER_ROWS", fibers, raising=False)
    monkeypatch.setattr(config_runtime, "C2_TRAINING_MANIFEST", training, raising=False)

    with pytest.raises(ValueError, match="drifted"):
        config_runtime.validate_c2_evaluation_sources()
