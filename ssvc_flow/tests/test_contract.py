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
        [
            sys.executable,
            "-m",
            "src.rollout",
            "--phase",
            "smoke",
            "--dry-run",
            "--out",
            str(tmp_path / "P1"),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
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
    assert result["pX"] == 0.25
    assert result["v"] == 0.75
    assert result["qX"] == pytest.approx(1 / 3)
    assert result["macro_mean_per_prompt_qX"] == 0.5
    assert result["independent_scene_count"] == 2
    assert result["category_counts"]["W"] == 0
    assert result["missing_categories"] == ["W"]


def test_statistics_undefined_and_invalid_cases():
    from src.statistics import summarize

    rows = [{"prompt_id": "p", "base_scene_id": "s", "category": "I"}]
    result = summarize(rows)
    assert result["qX"] is None and result["truth_given_answer"] is None
    assert result["macro_qX_NA_fraction"] == 1
    for invalid in [[], [{**rows[0], "category": "unknown"}]]:
        with pytest.raises(ValueError):
            summarize(invalid)
    with pytest.raises(ValueError, match="weights"):
        summarize(rows, {"p": 0.5})
    with pytest.raises(ValueError, match="nonnegative"):
        summarize([*rows, {**rows[0], "prompt_id": "q"}], {"p": -1, "q": 2})


def test_cluster_uncertainty_preserves_scenes_and_marks_undefined():
    from src.statistics import cluster_bootstrap

    rows = [{"prompt_id": "p", "base_scene_id": "s", "category": "X"}]
    assert cluster_bootstrap(rows)["status"] == "UNKNOWN"
    many = [*rows, {"prompt_id": "q", "base_scene_id": "t", "category": "X"}]
    result = cluster_bootstrap(many, repeats=100)
    assert result["low"] == result["high"] == 1
    result = cluster_bootstrap([{**row, "category": "I"} for row in many], "qX", repeats=100)
    assert result["status"] == "UNKNOWN"
    with pytest.raises(ValueError):
        cluster_bootstrap(many, repeats=1)
    with pytest.raises(ValueError, match="outer seed"):
        cluster_bootstrap([{**row, "train_seed": i} for i, row in enumerate(many)])


def test_config_validation_invalid_inputs_and_resolved_revision():
    from src.core import load_config, validate_config

    original = load_config()
    cases = [
        {**original, "models": {**original["models"], "qwen35_9b": {"id": "invented"}}},
        {**original, "generation": {**original["generation"], "temperature": 0.7}},
        {**original, "generation": {**original["generation"], "max_new_tokens": 2}},
        {**original, "training": {**original["training"], "K": 4}},
        {**original, "training": {**original["training"], "smoke_updates": 10}},
    ]
    for config in cases:
        with pytest.raises(ValueError):
            validate_config(config)
    resolved = {
        **original,
        "models": {
            **original["models"],
            "qwen35_9b": {**original["models"]["qwen35_9b"], "revision": "a" * 40},
        },
    }
    validate_config(resolved, "frozen")


def test_store_rejects_missing_manifest_malformed_rows_and_keys(tmp_path):
    from src.core import RunStore, load_config, load_split

    identity = {"model_hash": "m", "data_hash": "d", "config_hash": "c"}
    with pytest.raises(ValueError, match="identity"):
        RunStore(tmp_path, {})
    with pytest.raises(FileNotFoundError):
        RunStore(tmp_path, identity, resume=True)
    run = RunStore(tmp_path, identity)
    with pytest.raises(ValueError, match="sample_key"):
        run.append({})
    for body in ["{}\n", "not json\n", '{"sample_key":"a"}\n{"sample_key":"a"}\n']:
        (tmp_path / "samples.jsonl").write_text(body)
        with pytest.raises(ValueError):
            RunStore(tmp_path, identity, resume=True)
    (tmp_path / "bad.json").write_text("[]")
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.json")
    with pytest.raises(ValueError):
        load_split(tmp_path, "../train")


def test_main_gpu_status_uses_backend_gate(monkeypatch, tmp_path):
    import types

    from src import rollout

    panel = [
        {
            "base_scene_id": str(i),
            "constraint_family": f"f{i % 3}",
            "chart_type": f"c{(i // 3) % 2}",
            "operation": f"o{i // 6}",
        }
        for i in range(18)
    ]
    assert len(rollout.calibration_panel(panel + panel)) == 18
    with pytest.raises(ValueError):
        rollout.calibration_panel(panel[:2])
    monkeypatch.setattr(rollout, "load_split", lambda *args, **kwargs: panel)
    for passed in (False, True):
        fake = types.SimpleNamespace(run_smoke=lambda *a, passed=passed, **kw: {"passed": passed})
        monkeypatch.setitem(sys.modules, "src.smoke_runtime", fake)
        assert rollout.main(["--phase", "smoke", "--out", str(tmp_path / str(passed))]) == (
            0 if passed else 1
        )

    def fail(*args, **kwargs):
        raise RuntimeError("fixture backend unavailable")

    monkeypatch.setitem(sys.modules, "src.smoke_runtime", types.SimpleNamespace(run_smoke=fail))
    assert rollout.main(["--phase", "smoke", "--out", str(tmp_path / "failed")]) == 1
    assert (tmp_path / "failed/failures.jsonl").exists()
    assert rollout.main(["--phase", "frozen", "--out", str(tmp_path / "P3")]) == 2
    assert rollout.main(["--phase", "smoke", "--dry-run", "--out", str(tmp_path / "dry")]) == 0


def test_report_cli(tmp_path):
    from src.report import main

    main(["--run-root", str(tmp_path / "runs"), "--out", str(tmp_path / "report")])
    assert (tmp_path / "report/results_report_zh.md").exists()


def test_existing_phase_evidence_is_never_replaced_by_rejected_attempt(tmp_path):
    from src.core import phase_artifacts
    from src.rollout import main

    phase_artifacts(tmp_path, "P1", "PASS", {"measured": True})
    before = (tmp_path / "status.json").read_bytes()
    assert main(["--phase", "smoke", "--dry-run", "--out", str(tmp_path)]) == 2
    assert main(["--phase", "smoke", "--out", str(tmp_path)]) == 2
    assert (tmp_path / "status.json").read_bytes() == before


def test_orphan_records_cannot_be_assigned_a_new_identity(tmp_path):
    from src.core import RunStore

    (tmp_path / "samples.jsonl").write_text('{"sample_key":"old"}\n')
    with pytest.raises(ValueError, match="orphan"):
        RunStore(tmp_path, {"model_hash": "m", "data_hash": "d", "config_hash": "c"})


def test_frozen_budget_includes_greedy_and_respects_split():
    from src.core import load_config
    from src.rollout import estimate_budget

    budget = estimate_budget(load_config(), "frozen", split="dev")
    assert budget["rollout_count"] == 4896
    assert budget["sampled_rollout_count"] == 4608
    assert budget["greedy_rollout_count"] == 288
    assert estimate_budget(load_config(), "frozen", split="confirm")["prompt_count"] == 576
    with pytest.raises(ValueError):
        estimate_budget(load_config(), "frozen", split="made_up")


def test_report_acknowledges_real_smoke_without_claiming_scientific_results(tmp_path):
    from src.core import phase_artifacts, write_json
    from src.report import build_report

    phase_artifacts(tmp_path / "runs/P1", "P1", "PASS", {"passed": True})
    write_json(tmp_path / "runs/P1/model_audit.json", {"passed": True})
    build_report(tmp_path / "runs", tmp_path / "report")
    text = (tmp_path / "report/results_report_zh.md").read_text()
    assert "P1 兼容性测量已记录" in text
    assert "P3" in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("lora_rank", 16),
        ("lora_alpha", 32),
        ("lora_dropout", 0.1),
        ("beta_kl", 0.01),
        ("weight_decay", 0.1),
        ("epsilon", 0.01),
        ("alpha", 1),
        ("microbatch", 2),
        ("base_dtype", "float32"),
    ],
)
def test_locked_config_cannot_claim_ignored_algorithm_settings(field, value):
    from src.core import load_config, validate_config

    config = load_config()
    with pytest.raises(ValueError, match="locked"):
        validate_config({**config, "training": {**config["training"], field: value}})


def test_smoke_budget_counts_policy_reference_and_backward_forwards():
    from src.core import load_config
    from src.rollout import estimate_budget

    budget = estimate_budget(load_config())
    assert budget["teacher_forcing_forward_count"] == 960
    assert budget["policy_reference_scoring_forward_count"] == 704
    assert budget["update_forward_count"] == 128
    assert budget["postupdate_likelihood_forward_count"] == 128
