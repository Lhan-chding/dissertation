"""Pre-GPU run integrity and command boundary acceptance tests."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_hash_and_phase_artifacts(tmp_path):
    from src.core import canonical_hash, file_hash, phase_artifacts, write_json

    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    item = tmp_path / "table.json"
    write_json(item, {"count": 0})
    phase_artifacts(tmp_path, "P1", "PENDING_GPU", {"measured": False}, [item])
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "PENDING_GPU"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["files"][0]["sha256"] == file_hash(item)


def test_resume_checks_all_identity_fields_and_detects_tampering(tmp_path):
    from src.core import RunStore

    identity = {"model_hash": "m", "data_hash": "d", "config_hash": "c"}
    run = RunStore(tmp_path, identity)
    assert run.append({"sample_key": "s", "category": "I"}) is True
    assert run.append({"sample_key": "s", "category": "I"}) is False
    with pytest.raises(ValueError, match="conflict"):
        run.append({"sample_key": "s", "category": "X"})
    resumed = RunStore(tmp_path, identity, resume=True)
    assert resumed.keys == {"s"}
    for field in identity:
        with pytest.raises(ValueError, match="identity"):
            RunStore(tmp_path, {**identity, field: "different"}, resume=True)
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, identity)
    with (tmp_path / "samples.jsonl").open("a") as stream:
        stream.write('{"unfinished":')
    with pytest.raises(ValueError, match="incomplete"):
        RunStore(tmp_path, identity, resume=True)


def test_config_boundary_and_budget():
    from src.core import load_config, validate_config
    from src.rollout import estimate_budget

    config = load_config(Path(__file__).parents[1] / "configs/locked.json")
    validate_config(config, phase="smoke")
    with pytest.raises(ValueError, match="revision"):
        validate_config(config, phase="frozen")
    budget = estimate_budget(config, "smoke")
    assert budget["prompt_count"] == 36
    assert budget["rollout_count"] >= 36 * 8
    assert budget["max_completion_tokens"] == 64
    assert budget["backward_count"] > 0
    assert budget["gpu_hours"] is None
    assert budget["estimate_only"] is True


def test_sample_seed_independent_of_training_seed():
    from src.core import sample_identity

    a = sample_identity("run", "p", 0, 1, 0)
    assert a == sample_identity("run", "p", 0, 1, 0)
    assert a != sample_identity("run", "p", 0, 1, 1)
    assert a != sample_identity("run", "q", 0, 1, 0)


def test_confirm_access_is_guarded_and_logged(tmp_path):
    from src.core import load_split

    (tmp_path / "confirm.jsonl").write_text('{"base_scene_id":"s"}\n')
    with pytest.raises(PermissionError, match="confirm"):
        load_split(tmp_path, "confirm")
    assert not (tmp_path / "access.jsonl").exists()
    rows = load_split(tmp_path, "confirm", allow_confirm=True, purpose="frozen endpoint")
    assert rows[0]["base_scene_id"] == "s"
    assert json.loads((tmp_path / "access.jsonl").read_text())["purpose"] == "frozen endpoint"


def test_cli_dry_run_is_model_free(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.rollout", "--phase", "smoke", "--dry-run",
         "--out", str(tmp_path / "P1")],
        cwd=Path(__file__).parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "P1/status.json").read_text())["status"] == "DRY_RUN"
    assert not (tmp_path / "P1/samples.jsonl").exists()


def test_report_keeps_pending_distinct_from_observations(tmp_path):
    from src.core import phase_artifacts
    from src.report import build_report

    phase_artifacts(tmp_path / "runs/P2", "P2", "PASS", {"evidence_kind": "toy_math"})
    phase_artifacts(tmp_path / "runs/P1", "P1", "PENDING_GPU", {"measured": False})
    build_report(tmp_path / "runs", tmp_path / "reports")
    report = (tmp_path / "reports/results_report_zh.md").read_text()
    assert "PENDING_GPU" in report
    assert "toy_math" in report
    table_status = json.loads((tmp_path / "reports/table_status.json").read_text())
    assert table_status["training_steps.parquet"]["status"] == "NOT_RUN"
    assert not (tmp_path / "reports/training_steps.parquet").exists()


def test_statistics_use_prompt_weights_and_distinct_ratios():
    from src.statistics import summarize

    rows = [
        {"prompt_id": "p", "base_scene_id": "s", "category": "X"},
        {"prompt_id": "p", "base_scene_id": "s", "category": "I"},
        {"prompt_id": "q", "base_scene_id": "t", "category": "S"},
    ]
    result = summarize(rows)
    assert result["pX"] == .25
    assert result["v"] == .75
    assert result["qX"] == pytest.approx(1 / 3)
    assert result["macro_mean_per_prompt_qX"] == .5
    assert result["independent_scene_count"] == 2
    assert result["category_counts"]["W"] == 0
    assert result["missing_categories"] == ["W"]
