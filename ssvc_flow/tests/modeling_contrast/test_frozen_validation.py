"""Frozen N4 orchestration is tested without collecting or training trajectories."""

import json
from pathlib import Path

import pytest

from src.modeling_contrast.frozen_validation import (
    calibration_envelopes,
    frozen_configurations,
    project_frozen_validation_budget,
    run_frozen_validation,
    score_empirical_coverage,
)

CONFIG = json.loads(
    (Path(__file__).parents[2] / "configs/modeling_contrast/protocol.json").read_text()
)


def lock():
    return {
        "selected": [
            {
                "model": "C6",
                "method": "O_LR_MIX",
                "n": 256,
                "rank": 2,
                "alpha": 0.01,
                "eta": 0.1,
                "fit_banks": 4,
            }
        ]
    }


def test_configurations_keep_rank_and_coefficients_fixed_at_all_three_n():
    configurations = frozen_configurations(CONFIG, lock())
    selected = [row for row in configurations if row["method"] == "C6"]
    assert {row["n"] for row in selected} == {16, 64, 256}
    assert {(row["rank"], row["alpha"], row["eta"], row["fit_banks"]) for row in selected} == {
        (2, 0.01, 0.1, 4),
    }
    assert {"C0", "C1_LEGACY_EXACT_RULE", "C1_LEGACY_FROZEN", "C3"} <= {
        row["method"] for row in configurations
    }


def metric(seed, error):
    return {
        "configuration_id": "frozen",
        "target": "joint_1_minus_joint_0",
        "population": "all",
        "seed": seed,
        "arm": "X_BASE",
        "anchor": 8,
        "pX_abs_errors": [error] * 24,
        "v_abs_errors": [error] * 24,
    }


def test_six_seed_calibration_uses_seed_maxima_and_has_no_95_guarantee():
    result = calibration_envelopes(
        [metric(seed, (seed - 500) / 1000) for seed in range(501, 507)], CONFIG
    )
    assert result["distribution_free_95_claim"] is False
    assert result["rows"][0]["radius"] == 0.006
    assert len(result["rows"][0]["per_seed_maxima"]) == 6
    covered = score_empirical_coverage([metric(601, 0.005), metric(602, 0.007)], result)
    assert covered[0]["empirical_seed_coverage"] == 0.5
    assert covered[0]["test_seed_count"] == 2
    assert covered[0]["simultaneous_safety_certification"] is False


def test_calibration_refuses_test_seed_and_missing_six_seed_support():
    with pytest.raises(ValueError, match="calibration seed"):
        calibration_envelopes([metric(601, 0)], CONFIG)
    result = calibration_envelopes([metric(501, 0)], CONFIG)
    assert result["rows"][0]["radius"] is None
    assert result["rows"][0]["status"] == "INSUFFICIENT_CALIBRATION_SEEDS"


def test_missing_measured_resource_profiles_fail_closed_before_any_training(tmp_path):
    (tmp_path / "N2").mkdir()
    result = project_frozen_validation_budget(CONFIG, lock(), tmp_path, fresh_root=tmp_path)
    assert result["resource_gate"]["passed"] is False
    assert result["status"] == "RESOURCE_REVIEW_REQUIRED"
    assert result["fresh_anchor_count"] == 96
    assert "NO_MATCHING_FROZEN_FIT_RESOURCE_EXAMPLE" in result["missing_resource_evidence"]


def test_orchestrator_freezes_six_seed_calibration_before_any_test_unit(tmp_path, monkeypatch):
    from src.modeling_contrast import collect_fresh
    from src.modeling_contrast import frozen_validation as module
    from src.modeling_contrast.io import RunWriter
    from src.modeling_contrast.study import configuration_id

    run = tmp_path / "runs/modeling_contrast_v2"
    raw = run / "N4/raw"
    raw.mkdir(parents=True)
    selected = lock()
    selected["source_hashes"] = {}
    binding = dict.fromkeys(("source", "config", "data", "selector", "packet"), "a" * 64)
    entries = [
        {"id": f"seed{seed}_{arm}", "seed": seed, "arm": arm, "split": role}
        for role in ("fresh_calibration", "fresh_locked_test")
        for seed in CONFIG["data_roles"][role + "_seeds"]
        for arm in CONFIG["fresh_cpu"]["main_arms"]
    ]
    selection_path = run / "selection.json"
    selection_path.write_text(json.dumps(selected))
    forecast = {
        "wall_seconds": 100,
        "peak_ram_gib": 1,
        "added_output_bytes": 10000,
        "temporary_bytes": 0,
    }
    monkeypatch.setattr(
        module,
        "project_frozen_validation_budget",
        lambda *_a, **_k: {
            "forecast": forecast,
            "resource_gate": {"passed": True},
        },
    )
    monkeypatch.setattr(
        module,
        "validate_validation_inputs",
        lambda *_a, **_k: {
            "binding": binding,
            "lock": selected,
            "manifest": {"trajectories": entries},
            "selection_sha256": module.digest(selection_path),
            "raw_manifest_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(collect_fresh, "_verify_files", lambda *_a: None)
    calls = []

    def fake_unit(_config, _raw, entry, ai, configurations, out):
        if entry["split"] == "fresh_locked_test":
            calibration = json.loads((run / "N4_validation/CALIBRATION_LOCK.json").read_text())
            assert len(calibration["calibration_seed_ids"]) == 6
            assert calibration["frozen_before_locked_test_scoring"] is True
        calls.append((entry["seed"], ai, entry["split"]))
        out.mkdir(parents=True)
        (out / "predictions.npz").write_bytes(b"test seam: no trajectory generation")
        result = []
        for c in configurations:
            result.append(
                {
                    **c,
                    "configuration_id": configuration_id(c),
                    "target": "joint_1_minus_joint_0",
                    "population": "all",
                    "seed": entry["seed"],
                    "arm": entry["arm"],
                    "anchor": [8, 24, 40][ai],
                    "noise_replica": 0,
                    "case_count": 4,
                    "actual_rank": 1,
                    "k": 1,
                    "predicted_difference_norm": 0.01,
                    "event_error_ss": 0.01,
                    "event_truth_ss": 1.0,
                    "pX_error_ss": 0.01,
                    "pX_truth_ss": 1.0,
                    "v_error_ss": 0.01,
                    "v_truth_ss": 1.0,
                    "pX_abs_errors": [0.001] * 24,
                    "v_abs_errors": [0.001] * 24,
                    "v2_role": entry["split"],
                }
            )
        return result, []

    monkeypatch.setattr(module, "_unit", fake_unit)
    with RunWriter(run / "N4_validation", binding, project_root=tmp_path) as writer:
        result = run_frozen_validation(
            CONFIG, selection_path, raw, run, writer, existing_run_roots=[tmp_path]
        )
    assert len(calls) == 96
    assert all(role == "fresh_calibration" for _, _, role in calls[:36])
    assert all(role == "fresh_locked_test" for _, _, role in calls[36:])
    assert result["new_optimizer_updates"] == 0
    assert (run / "N4_validation/INDEPENDENT_VALIDATION.json").is_file()


def test_real_complete_collection_can_be_prepared_before_resume(tmp_path, monkeypatch):
    from src.modeling_contrast import collect_fresh
    from src.modeling_contrast import frozen_validation as module
    from src.modeling_contrast.io import RunWriter

    run = tmp_path / "runs/modeling_contrast_v2"
    binding = dict.fromkeys(("source", "config", "data", "selector", "packet"), "a" * 64)
    entries = [
        {"seed": seed, "arm": arm, "split": role}
        for role in ("fresh_calibration", "fresh_locked_test")
        for seed in CONFIG["data_roles"][role + "_seeds"]
        for arm in CONFIG["fresh_cpu"]["main_arms"]
    ]
    with RunWriter(run / "N4", binding, project_root=tmp_path) as writer:
        writer.write_json(
            "raw/manifest.json", {"profile": "contrast_fresh", "trajectories": entries}
        )
        writer.write_json(
            "FRESH_COLLECTION_SUMMARY.json", {"status": "PASS", "trajectory_count": 32}
        )
    selection = run / "selection.json"
    selection.write_text(json.dumps(lock()))
    monkeypatch.setattr(
        collect_fresh,
        "validate_fresh_gate",
        lambda *_a, **_k: {
            "binding": binding,
            "lock": {**lock(), "data_gate": {"fresh_seed_inventory_roots": [str(tmp_path)]}},
            "selection_sha256": module.digest(selection),
        },
    )
    forecast = {
        "wall_seconds": 1,
        "peak_ram_gib": 1,
        "added_output_bytes": 1000,
        "temporary_bytes": 0,
    }
    monkeypatch.setattr(
        module,
        "project_frozen_validation_budget",
        lambda *_a, **_k: {
            "resource_gate": {"passed": True},
            "forecast": forecast,
        },
    )
    receipt = module.prepare_frozen_validation(
        CONFIG, selection, run / "N4/raw", run, existing_run_roots=[tmp_path]
    )
    assert receipt["collection_status_before_validation"] == "COMPLETE"
    with RunWriter(run / "N4", binding, project_root=tmp_path, resume=True) as writer:
        module._check_prepared_collection(receipt, selection, run / "N4/raw")
        writer.write_json("validation_fixture.json", {"validated": True})
    (run / "N4/raw/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="raw collection hash changed"):
        module._check_prepared_collection(receipt, selection, run / "N4/raw")
