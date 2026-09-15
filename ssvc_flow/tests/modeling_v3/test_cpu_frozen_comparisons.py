"""Actual tiny original arrays prove paired scope and seed denominators."""

import copy

import numpy as np
import pytest
from test_frozen_comparisons import frozen_fixture

from src.modeling_v3.cpu_campaign import _digest, _npz, development_designs
from src.modeling_v3.cpu_results import _frozen_primary_comparisons
from src.modeling_v3.io import canonical_hash


def original_pairs(tmp_path):
    config, selected = frozen_fixture()
    lock = {"selected": selected, "selection_hash": canonical_hash(selected)}
    designs = {_digest(d)[:20]: d for d in development_designs(config, selected=selected)}
    needed = {
        e["design_id"]
        for s in selected["primary_comparison_family"]["hypotheses"][1:3]
        for e in [s["left"], s["right"]]
    }
    records = {did: [] for did in needed}
    units = {}
    for seed in range(6):
        for anchor in [8, 24]:
            key = (seed, "X_BASE", anchor, 0, 7001)
            unit = tmp_path / f"s{seed}_a{anchor}"
            unit.mkdir()
            truth = np.zeros((3, 2, 4))
            _npz(unit / "QUERY_REFERENCE.npz", truth=truth)
            direct = unit / "direct_n1024"
            direct.mkdir()
            _npz(
                direct / "estimates.npz",
                PRESERVE_XI=np.zeros((1, 2, 2, 4)),
                PILOT_SHRINK_ZERO_SUM=np.ones((1, 2, 2, 4)),
            )
            for did in needed:
                d = designs[did]
                k = 3 if anchor == 8 else 2
                r = min(d["rank_cap"], k) if type(d["rank_cap"]) is int else k
                # The enormous non-compression error must not enter the r<k comparison.
                err = (2 if anchor == 8 else 1000) if d["model"] == "RESPONSE_SVD" else 1
                if d["selector"] == "STRATIFIED_RANDOM":
                    err = 3
                fit = unit / did
                fit.mkdir()
                pred = np.full_like(truth, err)
                _npz(
                    fit / "PREDICTIONS.npz",
                    prediction=pred,
                    Q=np.eye(3)[:, :k],
                    directions=np.eye(3)[:, :r],
                    query_e=np.ones((3, 3)),
                    selected_bank_indices=np.array([0, 1]),
                    classifications=np.array(["OUT_OF_CALIBRATION_SPAN"] * 3),
                )
                row = {
                    "design_id": did,
                    "design": d,
                    "seed": seed,
                    "arm": "X_BASE",
                    "anchor": anchor,
                    "repeat": 0,
                    "initialization_seed": 7001,
                    "origin_id": unit.name,
                    "k": k,
                    "r": r,
                }
                rec = {
                    "row": row,
                    "unit": unit,
                    "prediction_path": fit / "PREDICTIONS.npz",
                    "truth_path": unit / "QUERY_REFERENCE.npz",
                }
                records[did].append(rec)
                units[key] = rec
    data = {
        "records": records,
        "units": units,
        "designs": designs,
        "seeds": list(range(6)),
        "groups": [0, 1],
        "originals": {},
    }
    return config, lock, data


def test_actual_r_less_k_intersection_excludes_full_equivalence_before_errors(tmp_path):
    config, lock, data = original_pairs(tmp_path)
    report = _frozen_primary_comparisons(config, lock, data, fixture=True)
    assert report["summary"]["status"] == "PENDING"
    rq3 = next(r for r in report["results"] if r["hypothesis_id"] == "RQ3")
    assert rq3["status"] == "AVAILABLE" and rq3["estimate"] == 3
    assert all(
        r["eligible_origin_count"] == 1 and r["total_origin_count"] == 2
        for r in rq3["paired_seed_totals"]
    )
    assert all(r["channel_totals"]["group_pX"]["count"] == 2 for r in rq3["paired_seed_totals"])
    assert rq3["accepted_mask_used"] is False
    rq1 = report["results"][0]
    assert rq1["estimate"] == 1


def test_unresolved_eligible_original_keeps_seed_and_forbids_pvalue(tmp_path):
    config, lock, data = original_pairs(tmp_path)
    low = next(
        r
        for did, rr in data["records"].items()
        for r in rr
        if r["row"]["design"]["model"] == "RESPONSE_SVD"
        and r["row"]["seed"] == 0
        and r["row"]["anchor"] == 8
    )
    path = low["prediction_path"]
    arrays = dict(np.load(path))
    arrays["prediction"][0, 0, 0] = np.nan
    path.unlink()
    _npz(path, **arrays)
    report = _frozen_primary_comparisons(config, lock, data, fixture=True)
    rq3 = report["results"][2]
    assert rq3["status"] == "UNKNOWN" and rq3["raw_pvalue"] is None
    assert len(rq3["paired_seed_totals"]) == 6


def test_geometry_mismatch_is_not_a_matched_dimension_comparison(tmp_path):
    config, lock, data = original_pairs(tmp_path)
    low = next(
        r
        for rr in data["records"].values()
        for r in rr
        if r["row"]["design"]["model"] == "RESPONSE_SVD"
    )
    path = low["prediction_path"]
    arrays = dict(np.load(path))
    arrays["selected_bank_indices"] = np.array([1, 2])
    path.unlink()
    _npz(path, **arrays)
    with pytest.raises(ValueError, match=r"geometry|bank"):
        _frozen_primary_comparisons(config, lock, data, fixture=True)


def test_complete_analysis_verifier_recomputes_pvalues_from_original_arrays(tmp_path):
    from src.modeling_v3.cpu_campaign import _finish
    from src.modeling_v3.cpu_results import _collect, verify_cpu_primary_comparisons
    from src.modeling_v3.io import atomic_json, finalize_run, sha256_file, source_identity

    root = tmp_path / "responses"
    root.mkdir()
    config, lock, data = original_pairs(root)
    config["observation"]["total_draws_grid"] = [1024]
    lock["selected"]["models"] = ["FULL_RIDGE", "RESPONSE_SVD"]
    lock.update(
        selection_hash=canonical_hash(lock["selected"]),
        config_sha256=canonical_hash(config),
        source_sha256=source_identity()["sha256"],
        development_evidence={},
    )
    designs = development_designs(config, selected=lock["selected"])
    # Complete the full (small fixture) frozen matrix, preserving actual pair originals.
    (root / "units").mkdir()
    rows = []
    for key, old in list(data["units"].items()):
        unit = old["unit"]
        destination = root / "units" / unit.name
        unit.rename(destination)
        unit = destination
        entries = []
        for d in designs:
            did = _digest(d)[:20]
            fit = unit / did
            path = fit / "PREDICTIONS.npz"
            k = 3 if key[2] == 8 else 2
            r = min(d["rank_cap"], k) if type(d["rank_cap"]) is int else k
            if not path.exists():
                fit.mkdir()
                _npz(
                    path,
                    prediction=np.ones((3, 2, 4)),
                    Q=np.eye(3)[:, :k],
                    directions=np.eye(3)[:, :r],
                    query_e=np.ones((3, 3)),
                    selected_bank_indices=np.array([0, 1]),
                    classifications=np.array(["OUT_OF_CALIBRATION_SPAN"] * 3),
                )
            row = {
                "design_id": did,
                "design": d,
                "origin_id": unit.name,
                "seed": key[0],
                "arm": key[1],
                "anchor": key[2],
                "repeat": key[3],
                "initialization_seed": key[4],
                "k": k,
                "r": r,
                "predictions_relative": str(path.relative_to(unit)),
                "prediction_sha256": sha256_file(path),
            }
            _finish(fit, {}, row)
            entries.append(row)
        _finish(unit / "direct_n1024", {}, {})
        body = {"results": entries}
        _finish(unit, {}, body)
        rows.extend(entries)
    _finish(
        root,
        {"config_sha256": canonical_hash(config), "source_hashes": source_identity()["files"]},
        {
            "stage": "Q3",
            "role": "locked_test",
            "pilot": False,
            "probe_groups": [0, 1],
            "results": rows,
        },
    )
    actual = _collect(config, lock, [root], "locked_test", True)
    primary = _frozen_primary_comparisons(config, lock, actual, fixture=True)
    report = {
        "schema": "ssvc-v3-cpu-test-analysis-1",
        "stage": "Q3_LOCKED_TEST_ANALYSIS",
        "status": "TEST_FIXTURE_NOT_AUTHORIZATION",
        "fixture": True,
        "selection_hash": lock["selection_hash"],
        "config_sha256": canonical_hash(config),
        "source_sha256": source_identity()["sha256"],
        "test_seeds": actual["seeds"],
        "unit_matrix": actual["matrix"],
        "source_manifest_hashes": actual["originals"],
        "primary_comparison_family": primary,
    }

    def save(name, value):
        out = tmp_path / name
        out.mkdir()
        atomic_json(out / "TEST_ANALYSIS.json", value)
        finalize_run(out, {})
        return {
            "path": str(out / "TEST_ANALYSIS.json"),
            "sha256": sha256_file(out / "TEST_ANALYSIS.json"),
        }

    bound = save("analysis", report)
    assert verify_cpu_primary_comparisons(config, lock, bound, fixture=True) == primary
    bad = copy.deepcopy(report)
    bad["primary_comparison_family"]["results"][0]["raw_pvalue"] = 0.99
    with pytest.raises(ValueError, match="statistics differ"):
        verify_cpu_primary_comparisons(config, lock, save("forged", bad), fixture=True)
