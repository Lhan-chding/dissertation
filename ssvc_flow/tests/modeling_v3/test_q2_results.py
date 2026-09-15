"""Small complete Q2 originals retain the full two-seed/arm/anchor matrix."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v3 import q2_results as q
from src.modeling_v3.cpu_campaign import _digest, _finish, _json, _npz, _sources, trajectory_specs


def protocol():
    config = json.loads(Path("configs/modeling_v3/protocol.json").read_text())
    base = {
        "study": "DIMENSION",
        "m": 8,
        "n": 4,
        "selector": "STRATIFIED_RANDOM",
        "design_seed": 2026091500,
        "estimator": "PRESERVE_XI",
        "model": "FULL_RIDGE",
        "rank_cap": "FULL",
        "alpha": 1e-5,
    }
    designs = [
        base,
        {**base, "model": "PCA", "rank_cap": 1},
        {**base, "model": "PCA", "rank_cap": "FULL"},
        {**base, "model": "ZERO"},
    ]
    config["q2_analysis_fixture"] = {
        "designs": designs,
        "probe_groups": [0, 0, 1, 1],
        "heldout_banks": 1,
        "parameter_dimension": 2,
    }
    return config, designs


def originals(
    root,
    config,
    designs,
    *,
    omit_origin=None,
    omit_design=None,
    unknown=False,
    omit_direct=False,
    role="development",
    source_drift=False,
):
    root.mkdir()
    bound = {
        "config_sha256": _digest(config),
        "source_hashes": _sources(),
        "stage": "Q2",
        "role": role,
        "pilot": False,
        "designs": designs,
        "collection_sha256": "fixture-collection",
        "calibration_receipt": None,
        "fixture": True,
    }
    if source_drift:
        bound["source_hashes"] = {
            **bound["source_hashes"],
            "src/modeling_v3/coverage.py": "changed",
        }
    all_rows = []
    _json(root / "DESIGNS.json", designs)
    for trajectory in trajectory_specs(config, "development"):
        for anchor in config["cpu"]["anchors"]:
            origin_id = f"s{trajectory['seed']}_{trajectory['arm']}_a{anchor}_repeat00"
            if omit_origin == (trajectory["seed"], trajectory["arm"], anchor):
                continue
            unit = root / "units" / origin_id
            unit.mkdir(parents=True)
            scale = 1.0 if trajectory["seed"] == 101 else 10.0
            truth = np.broadcast_to(np.array([1.0, 2.0, 3.0, -6.0]) * scale, (3, 4, 4)).copy()
            truth[:, 2:] *= 2  # Group averages must use the actual two-probe membership.
            rows, predictions = [], {}
            for design in designs:
                design_id = _digest(design)[:20]
                if omit_design == design_id and trajectory["seed"] == 101 and anchor == 8:
                    continue
                fit = unit / "fits" / design_id
                fit.mkdir(parents=True)
                residual = np.zeros_like(truth)
                if trajectory["seed"] == 101:
                    residual[:] = 1.0
                if design["model"] == "PCA" and design["rank_cap"] == 1:
                    residual *= 2
                prediction = truth + residual
                if design["model"] == "ZERO":
                    prediction = np.zeros_like(truth)
                if unknown and trajectory["seed"] == 101 and anchor == 8:
                    prediction[0] = np.nan
                labels = np.array(
                    [
                        "MEASUREMENT_UNRESOLVED",
                        "OUT_OF_CALIBRATION_SPAN",
                        "NO_CALIBRATION_EXCITATION",
                    ]
                )
                if design["model"] == "ZERO":
                    labels[:] = "STATISTICAL_BASELINE_ONLY"
                rank = 0 if design["model"] == "ZERO" else 1 if design["rank_cap"] == 1 else 2
                path = fit / "PREDICTIONS.npz"
                _npz(
                    path,
                    prediction=prediction,
                    classifications=labels,
                    rho=np.array([0.01, 0.2, 0.8]),
                    leverage=np.array([1.0, 2.0, 20.0]),
                    e_norm=np.array([1.0, 2.0, 3.0]),
                    e_perp_norm=np.array([0.01, 0.4, 2.4]),
                    singular_values=np.array([4.0, 2.0]),
                    Q=np.eye(2),
                    directions=np.eye(2)[:, :rank],
                    query_e=np.ones((3, 2)),
                )
                from src.modeling_v3.io import sha256_file

                row = {
                    "design_id": design_id,
                    "design": design,
                    "seed": trajectory["seed"],
                    "arm": trajectory["arm"],
                    "initialization_seed": trajectory["initialization_seed"],
                    "anchor": anchor,
                    "repeat": 0,
                    "origin_id": origin_id,
                    "k": 2,
                    "r": rank,
                    "rank_comparison_eligible": 0 < rank < 2,
                    "condition_number": 2.0,
                    "target_excitation_from_actual_parameter_contrasts": [True, True, False],
                    "fit_metadata": {"regression": "RIDGE", "output_policy": "RAW4"},
                    "predictions_relative": str(path.relative_to(unit)),
                    "prediction_sha256": sha256_file(path),
                    "status": "FITTED",
                }
                _json(
                    fit / "QUERY_LOG.json",
                    {
                        "hidden_query_labels_read": False,
                        "role": role,
                        "training_seed": trajectory["seed"],
                        "prediction_sha256": row["prediction_sha256"],
                    },
                )
                _finish(fit, {"design": design, "origin": origin_id}, row)
                predictions[design_id] = row["prediction_sha256"]
                rows.append(
                    {**row, "metrics": {"mse": 987654321}}
                )  # Deliberately false cached metric.
            _json(unit / "PREDICTION_LOCK.json", predictions)
            _npz(unit / "QUERY_REFERENCE.npz", truth=truth)
            _json(
                unit / "REFERENCE_ACCESS_RECEIPT.json",
                {
                    "prediction_lock_sha256": sha256_file(unit / "PREDICTION_LOCK.json"),
                    "reference_status": "KNOWN_EVENT_SCORE_EXACT_FINITE_ACTION_ENUMERATION",
                    "heldout_read_after_prediction_freeze": True,
                    "training_seed": trajectory["seed"],
                },
            )
            direct = []
            if not omit_direct:
                direct_root = unit / "direct_n4"
                direct_root.mkdir()
                base = np.stack((truth[0] + 0.5, truth[1] - 0.5), axis=0)[None]
                _npz(direct_root / "estimates.npz", PRESERVE_XI=base)
                _finish(direct_root, {"n": 4}, {"n": 4})
                direct.append({"model": "DIRECT_MEASURE", "estimator": "PRESERVE_XI", "n": 4})
            direct.append({"model": "KNOWN_EVENT_SCORE"})
            summary = {
                "origin": origin_id,
                "results": rows,
                "direct_baselines": direct,
                "prediction_lock_sha256": sha256_file(unit / "PREDICTION_LOCK.json"),
            }
            _json(unit / "RESULTS.json", summary)
            _finish(unit, {**bound, "origin": origin_id}, summary)
            all_rows.extend(rows)
    summary = {
        "stage": "Q2",
        "role": role,
        "pilot": False,
        "scientific_status": "DEVELOPMENT_ONLY",
        "results": all_rows,
        "fit_count": len(all_rows),
        "probe_groups": config["q2_analysis_fixture"]["probe_groups"],
    }
    _json(root / "COVERAGE_SUMMARY.json", summary)
    _finish(root, bound, summary)
    return root


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def test_actual_complete_development_recomputes_metrics_and_keeps_baselines(tmp_path):
    config, designs = protocol()
    root = originals(tmp_path / "original", config, designs)
    result = q.analyze_coverage(config, [root], tmp_path / "analysis", fixture=True)
    assert result["status"] == "DEVELOPMENT_ANALYZED"
    assert result["scientific_status"] == "DEVELOPMENT_ONLY"
    assert result["independent_development_seeds"] == [101, 201]
    assert result["unit_matrix"]["origin_count"] == 12
    assert result["completed_fits"] == 48
    assert result["winner_selected"] is False
    assert result["confirmatory_inference"] == "NOT_PERFORMED"
    design_id = _digest(designs[0])[:20]
    rows = read_csv(tmp_path / "analysis" / "SUMMARY_METRICS.csv")
    value = next(
        row
        for row in rows
        if row["design_id"] == design_id
        and row["target"] == "ALL_TARGETS"
        and row["channel"] == "RAW4"
    )
    assert float(value["all_case_mse"]) == 0.5
    assert float(value["q95_absolute_residual"]) == 1.0
    assert float(value["error_ss"]) == 288
    assert float(value["nrmse"]) == pytest.approx(
        np.sqrt(float(value["error_ss"]) / float(value["signal_ss"]))
    )
    grouped = next(
        row for row in rows if row["design_id"] == design_id and row["channel"] == "group_v"
    )
    assert float(grouped["bias"]) == -0.5  # v=-I, not X+S+W.
    assert {row["kind"] for row in rows} >= {"MODEL", "ZERO", "DIRECT_MEASURE", "KNOWN_EVENT_SCORE"}
    exact = [row for row in rows if row["kind"] == "KNOWN_EVENT_SCORE"]
    assert all(float(row["error_ss"]) == 0 for row in exact)
    dimensions = read_csv(tmp_path / "analysis" / "RANK_COMPARISONS.csv")
    assert {row["rank_relation"] for row in dimensions} == {"R_LESS_K", "R_EQUALS_K"}


def test_unknown_is_not_finite_and_geometry_classes_are_retained(tmp_path):
    config, designs = protocol()
    root = originals(tmp_path / "original", config, designs, unknown=True)
    q.analyze_coverage(config, [root], tmp_path / "analysis", fixture=True)
    rows = read_csv(tmp_path / "analysis" / "SUMMARY_METRICS.csv")
    value = next(
        row
        for row in rows
        if row["design_id"] == _digest(designs[0])[:20]
        and row["target"] == "ALL_TARGETS"
        and row["channel"] == "RAW4"
    )
    assert value["all_case_mse"] == ""
    assert value["nrmse"] == ""
    assert int(value["unknown_cases"]) == int(value["all_cases"])
    assert int(value["finite_cases"]) < int(value["all_cases"])
    summaries = read_csv(tmp_path / "analysis" / "DESIGN_ORIGIN_SUMMARY.csv")
    labels = json.loads(summaries[0]["classification_counts"])
    assert labels["OUT_OF_CALIBRATION_SPAN"] == 1
    assert labels["NO_CALIBRATION_EXCITATION"] == 1
    assert summaries[0]["condition_number"] == "2.0"


@pytest.mark.parametrize("failure", ["origin", "design", "duplicate", "direct", "source", "role"])
def test_fixture_still_requires_all_independent_axes_and_original_identity(tmp_path, failure):
    config, designs = protocol()
    kwargs = {
        "origin": {"omit_origin": (101, "X_BASE", 8)},
        "design": {"omit_design": _digest(designs[1])[:20]},
        "direct": {"omit_direct": True},
        "source": {"source_drift": True},
        "role": {"role": "locked_test"},
        "duplicate": {},
    }[failure]
    root = originals(tmp_path / "original", config, designs, **kwargs)
    inputs = [root, root] if failure == "duplicate" else [root]
    with pytest.raises(ValueError):
        q.analyze_coverage(config, inputs, tmp_path / "analysis", fixture=True)


def test_changed_raw_original_and_fixture_as_production_are_rejected(tmp_path):
    config, designs = protocol()
    root = originals(tmp_path / "original", config, designs)
    with pytest.raises((ValueError, PermissionError), match=r"server|Slurm|fixture"):
        q.analyze_coverage(config, [root], tmp_path / "production")
    original = next(root.glob("units/*/fits/*/PREDICTIONS.npz"))
    original.write_bytes(original.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        q.analyze_coverage(config, [root], tmp_path / "analysis", fixture=True)
