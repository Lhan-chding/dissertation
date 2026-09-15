"""Tiny sealed fixtures exercise posthoc aggregation without production imports."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/summarize_modeling_n4_results.py"
spec = importlib.util.spec_from_file_location("n4_fact_summary", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sealed(tmp_path):
    run = tmp_path / "run"
    stage = run / "N4_validation"
    config = dict(
        method="C3", observation="O_IND", n=64, rank="FULL", alpha=0.01, eta=0.1, fit_banks=8
    )
    ident = "C3|O_IND|64|FULL|0.01|0.1|8"
    write_json(stage / "FROZEN_VALIDATION_PROTOCOL.json", {"configurations": [config]})
    write_json(
        stage / "stage_result.json",
        {"status": "FROZEN_FRESH_CPU_VALIDATION_COMPLETE", "wall_seconds": 9},
    )
    for role, seed in (("fresh_calibration", 501), ("fresh_locked_test", 601)):
        metrics, model_meta = [], []
        for noise, errors, energy in ((0, [1, 2, 3, 4, 5, 6], 100), (1, [2] * 6, 4)):
            base = dict(
                config,
                configuration_id=ident,
                seed=seed,
                arm="X_BASE",
                anchor=8,
                noise_replica=noise,
                actual_rank=2 - noise,
                k=3 + noise,
                v2_role=role,
                original_role=role,
            )
            model_meta.append(base)
            row = dict(base, target="joint_1_minus_joint_0", population="all", case_count=1)
            for quantity in ("pX", "v"):
                row.update(
                    {
                        f"{quantity}_error_ss": sum(x * x for x in errors),
                        f"{quantity}_truth_ss": energy,
                        f"{quantity}_abs_errors": errors,
                    }
                )
            metrics.append(row)
            metrics.append(
                dict(
                    row,
                    method="D_DIRECT",
                    configuration_id="D_DIRECT|O_IND|64|0|0.0|0.0|0",
                    rank=0,
                    alpha=0.0,
                    eta=0.0,
                    fit_banks=0,
                    actual_rank=0,
                    k=0,
                )
            )
        known = dict(
            metrics[0],
            configuration_id="D_KNOWN_EVENT_LOGP",
            method="D_KNOWN_EVENT_LOGP",
            observation="EXACT_KNOWN_EVENTS",
            n=0,
        )
        for quantity in ("pX", "v"):
            known.update(
                {
                    f"{quantity}_error_ss": 0,
                    f"{quantity}_truth_ss": 0,
                    f"{quantity}_abs_errors": [0] * 6,
                }
            )
        for name, values in (("metrics", metrics), ("known_event_metrics", [known])):
            (stage / f"{role}_{name}.jsonl.gz").write_bytes(
                gzip.compress("".join(json.dumps(row) + "\n" for row in values).encode())
            )
        unit = stage / role / f"seed{seed}_X_BASE/a0"
        freeze = {"frozen_before_scoring": True, "models": model_meta, "fit_seconds": 2.5}
        write_json(unit / "models/freeze_metadata.json.gz", freeze)
        write_json(
            unit / "models/freeze.json",
            {
                "metadata_file": "freeze_metadata.json.gz",
                "metadata_sha256": sha(unit / "models/freeze_metadata.json.gz"),
                "fit_seconds": 2.5,
            },
        )
        write_json(
            unit / "cost_ledger.json.gz",
            {
                "v2_role": role,
                "known_event": {"score_requests": 3},
                "packets": {
                    "O_IND:n64": {
                        "method": "O_IND",
                        "n": 64,
                        "wall_seconds": 4,
                        "packets": [{"cost": {"generated_actions": 5, "sampling_seconds": 0.1}}],
                        "C1_supplementary_costs": [{"cost": {"generated_actions": 2}}],
                    }
                },
            },
        )
    outputs = {
        str(path.relative_to(stage)): sha(path) for path in stage.rglob("*") if path.is_file()
    }
    write_json(stage / "RUN_MANIFEST.json", {"status": "COMPLETE", "outputs": outputs})
    return run


def test_pooled_sufficient_statistics_groups_costs_and_all_baselines(sealed, tmp_path):
    result = module.summarize(sealed, tmp_path / "posthoc")
    row = next(
        row
        for row in result["pooled_metrics"]
        if row["role"] == "fresh_calibration" and row["method"] == "C3"
    )
    assert row["pX_nrmse"] == pytest.approx(np.sqrt(115 / 104))
    assert row["pX_mae"] == pytest.approx(2.75)
    assert row["pX_error_q95"] == pytest.approx(5.45)
    assert row["pX_worst_group_by_mae"] == 5
    assert row["pX_worst_group_mae"] == 4
    assert (row["actual_rank_min"], row["actual_rank_max"], row["k_min"], row["k_max"]) == (
        1,
        2,
        3,
        4,
    )
    assert {row["method"] for row in result["pooled_metrics"]} == {
        "C3",
        "D_DIRECT",
        "D_KNOWN_EVENT_LOGP",
    }
    known = next(row for row in result["pooled_metrics"] if row["method"] == "D_KNOWN_EVENT_LOGP")
    assert known["pX_nrmse"] is None and known["pX_max_abs_error"] == 0
    cost = result["costs_by_role"]["fresh_calibration"]
    assert cost["total_recorded_cost"]["generated_actions"] == 7
    assert cost["total_recorded_cost"]["score_requests"] == 3
    assert cost["packet_wall_seconds"] == 4 and cost["model_fit_wall_seconds"] == 2.5
    assert result["coverage"]["observations_present"] == ["O_IND"]
    assert "O_CRN" in result["coverage"]["observations_not_in_frozen_panel"]
    assert len(list((tmp_path / "posthoc").iterdir())) == 2
    assert result["csv_sha256"] == sha(tmp_path / "posthoc/N4_FACTUAL_METRICS.csv")


def test_no_clobber_stage_path_and_corrupt_source_fail_closed(sealed, tmp_path):
    with pytest.raises(ValueError, match="sealed"):
        module.summarize(sealed, sealed / "N4_validation/posthoc")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        module.summarize(sealed, existing)
    metrics = sealed / "N4_validation/fresh_calibration_metrics.jsonl.gz"
    metrics.write_bytes(metrics.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        module.summarize(sealed, tmp_path / "corrupt_summary")
    assert not (tmp_path / "corrupt_summary").exists()


def test_optional_raw_group_energy_is_bound_and_never_invented(sealed, tmp_path):
    groups = []
    for index, energy in enumerate((10, 12, 14, 16, 22, 30)):
        errors = np.array([index + 1, 2])
        row = {
            "role": "fresh_calibration",
            "configuration_id": "C3|O_IND|64|FULL|0.01|0.1|8",
            "method": "C3",
            "n": 64,
            "target": "joint_1_minus_joint_0",
            "population": "all",
            "group": f"group{index}",
        }
        for quantity in ("pX", "v"):
            row.update(
                {
                    quantity + "_error_ss": float(errors @ errors),
                    quantity + "_truth_ss": energy,
                    quantity + "_nrmse": float(np.sqrt((errors @ errors) / energy)),
                    quantity + "_mae": float(errors.mean()),
                    quantity + "_error_q95": float(np.quantile(errors, 0.95)),
                }
            )
        groups.append(row)
    audit_path = tmp_path / "independent_raw_audit.json"
    audit = {
        "status": "PASS",
        "run_root": str(sealed),
        "group_pooled": groups,
        "stage_manifest_hashes": {"N4_validation": sha(sealed / "N4_validation/RUN_MANIFEST.json")},
    }
    write_json(audit_path, audit)
    result = module.summarize(sealed, tmp_path / "with_audit", raw_audit_json=audit_path)
    row = next(
        row
        for row in result["pooled_metrics"]
        if row["role"] == "fresh_calibration" and row["method"] == "C3"
    )
    assert row["pX_worst_defined_group_nrmse"] == pytest.approx(np.sqrt(40 / 30))
    assert row["pX_worst_defined_group_by_nrmse"] == "group5"
    assert result["raw_group_audit"]["group_rows_applied"] == 6
    audit["stage_manifest_hashes"]["N4_validation"] = "unrelated-stage"
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match="same complete N4 stage"):
        module.summarize(sealed, tmp_path / "wrong_audit", raw_audit_json=audit_path)
