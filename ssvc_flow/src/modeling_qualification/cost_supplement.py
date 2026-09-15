"""Bounded, read-only replay and allocated full-cost audit of frozen M2/M4 data.

No fitting rule, stored count, trajectory or original result is modified. The
allocation estimates are accounting conventions, not per-configuration timings.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .collection import load_observed
from .evaluation import _dp, _dy, file_hash, write_csv
from .io import write_json
from .math_contracts import to_helmert
from .models import array_hash, fit_local_model
from .tracking import cost_amortization, fixed_window_prediction


def allocate_panel_costs(summary, manifest, file_bytes, samples_grid, replicas):
    """Allocate measured totals exactly across their declared accounting units."""
    exact_panels = summary["exact_panel_calls"]
    finite_panels = exact_panels * len(samples_grid) * replicas
    forks = (
        len(manifest["trajectories"])
        * len(manifest["anchors"])
        * sum(len(v) for v in manifest["bank_roles"].values())
        * 3
    )
    total_bytes = sum(file_bytes.values())
    io_per_byte = summary["output_io_seconds"] / total_bytes
    shared_bytes = sum(v for k, v in file_bytes.items() if k != "counts_file")
    result = {
        "status": "ALLOCATED_ESTIMATE",
        "exact_panels_measured": exact_panels,
        "finite_count_panels_allocated": finite_panels,
        "fork_operations_allocated": forks,
        "measured_totals_seconds": {
            k: summary[k]
            for k in ("exact_seconds", "branch_seconds", "measurement_seconds", "output_io_seconds")
        },
        "timed_output_bytes_by_category": file_bytes,
        "exact_panel_compute_seconds": summary["exact_seconds"] / exact_panels,
        "branch_operation_seconds": summary["branch_seconds"] / forks,
        "finite_count_panel_seconds": summary["measurement_seconds"] / finite_panels,
        "exact_panel_output_io_seconds": io_per_byte * shared_bytes / exact_panels,
        "finite_count_panel_output_io_seconds": io_per_byte
        * file_bytes["counts_file"]
        / finite_panels,
        "allocation_rule": (
            "Counts time divided equally across n x noise x unique exact panels; "
            "no measured n-specific "
            "generation timings exist. M2 output IO is allocated by physical bytes to counts vs "
            "observations/oracle/RNG, then divided across corresponding panels. Shared optimizer "
            "snapshot/RNG output is charged in the non-count allocation."
        ),
        "count_alias_policy": (
            "M2 count generation reuses duplicate branch outputs; denominator "
            "uses measured unique exact panel calls"
        ),
        "not_a_per_configuration_measurement": True,
    }
    result["measurement_total_reconstructed_seconds"] = (
        result["finite_count_panel_seconds"] * finite_panels
    )
    result["output_io_total_reconstructed_seconds"] = (
        result["exact_panel_output_io_seconds"] * exact_panels
        + result["finite_count_panel_output_io_seconds"] * finite_panels
    )
    return result


def _stored_panels(root, entry, n, anchor, horizon, *, anchor_only=False):
    path = root / entry["oracle_file" if n == "exact" else "counts_file"]
    with np.load(path, allow_pickle=False) as data:
        values = data["p"] if n == "exact" else data[f"trajectory_n{n}"][0] / int(n)
        value = values[anchor] if anchor_only else values[anchor + 1 : anchor + horizon + 1]
        return value.copy()


def _serialize_measure(out, arrays):
    begin = time.perf_counter()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    serialization = time.perf_counter() - begin
    path = out / "_bounded_replay_temporary.npz"
    begin = time.perf_counter()
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    disk = time.perf_counter() - begin
    # Scratch output is produced by this supplement only; no original artifacts
    # are overwritten. Hashes below retain reproducible array identities.
    path.unlink()
    return serialization, disk, len(payload)


def _replay_benchmark(root, manifest, selection, out, repeats=3):
    entry = next(e for e in manifest["trajectories"] if e["seed"] == 301 and e["arm"] == "X_BASE")
    anchor, horizon = 8, 16
    ai = manifest["anchors"].index(anchor)
    specs = [
        selection["frozen_grid"]["B0_PERSISTENCE|0"],
        selection["frozen_grid"]["B6_FULL_Q_RIDGE|FULL"],
        selection["selected"],
    ]
    exact = selection.get("exact_selection", {})
    if exact.get("rank_cap") is not None:
        specs.append(
            {
                "method": "B5_EXACT_SELECTED_DIAGNOSTIC",
                "rank_cap": exact["rank_cap"],
                "alpha": exact["alpha"],
            }
        )
    measured = []
    for n in ("exact", 16, 64, 256):
        for repeat in range(repeats):
            start = time.perf_counter()
            dense = _stored_panels(root, entry, n, anchor, horizon)
            read = time.perf_counter() - start
            serialize, write, size = _serialize_measure(out, {"probability": dense})
            measured.append(
                {
                    "kind": "DENSE_STORED_PANEL_REPLAY",
                    "method": "DENSE",
                    "rank_cap": None,
                    "n": n,
                    "repeat": repeat,
                    "H": horizon,
                    "read_seconds": read,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "serialization_seconds": serialize,
                    "disk_write_fsync_seconds": write,
                    "output_bytes": size,
                    "array_hash": array_hash(dense),
                    "elapsed_total_seconds": time.perf_counter() - start,
                }
            )
            for spec in specs:
                start = time.perf_counter()
                if spec["method"] == "B0_PERSISTENCE":
                    anchor_p = _stored_panels(root, entry, n, anchor, horizon, anchor_only=True)
                    read = time.perf_counter() - start
                    fit_seconds = 0.0
                    before = time.perf_counter()
                    raw = np.broadcast_to(anchor_p, (horizon, *anchor_p.shape)).copy()
                    prediction = time.perf_counter() - before
                else:
                    view = load_observed(root, entry["id"], n=None if n == "exact" else n, noise=0)
                    fit = view.anchor(ai)
                    read = time.perf_counter() - start
                    before = time.perf_counter()
                    model = fit_local_model(
                        fit["d"],
                        _dy(fit["p0"], fit["p1"]),
                        spec["method"],
                        spec["rank_cap"],
                        spec["alpha"],
                    )
                    fit_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    predicted, _ = fixed_window_prediction(
                        to_helmert(fit["p0"]),
                        view.updates[anchor : anchor + horizon],
                        model["U"],
                        model["C"],
                    )
                    raw = fit["p0"] + _dp((predicted - to_helmert(fit["p0"])).reshape(horizon, -1))
                    prediction = time.perf_counter() - before
                serialize, write, size = _serialize_measure(out, {"probability": raw})
                measured.append(
                    {
                        "kind": "FROZEN_RULE_STORED_DATA_REPLAY",
                        "method": spec["method"],
                        "rank_cap": spec["rank_cap"],
                        "n": n,
                        "repeat": repeat,
                        "H": horizon,
                        "read_seconds": read,
                        "fit_seconds": fit_seconds,
                        "predict_seconds": prediction,
                        "serialization_seconds": serialize,
                        "disk_write_fsync_seconds": write,
                        "output_bytes": size,
                        "array_hash": array_hash(raw),
                        "elapsed_total_seconds": time.perf_counter() - start,
                    }
                )
    return measured


def step_status_impact(path):
    counts, affected = Counter(), 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            counts[row["tolerance_status"]] += 1
            if row["tolerance_status"] == "UNKNOWN":
                # If no UNKNOWN rows exist, every possible priority-conflict row
                # is ruled out, including invalid-simplex failures.
                absolute_fail = (
                    float(row["raw_delta_worst_group_error"]) > 0.01
                    or float(row["raw_delta_mae"]) > 0.02
                )
                affected += int(absolute_fail)
    return {
        "original_status_counts": dict(counts),
        "absolute_failure_hidden_by_unknown_rows": affected,
        "any_possible_priority_affected_rows": 0 if not counts["UNKNOWN"] else None,
        "status": "NO_ORIGINAL_RESULT_ROWS_AFFECTED"
        if not counts["UNKNOWN"]
        else "REQUIRES_DETAILED_UNKNOWN_ROW_REVIEW",
    }


def run_cost_supplement(run_root: Path, out: Path):
    started = time.perf_counter()
    run_root, out = Path(run_root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m2, m4, m3 = run_root / "M2", run_root / "M4", run_root / "M3_verified"
    manifest = json.loads((m2 / "manifest.json").read_text())
    summary = json.loads((m2 / "summary.json").read_text())
    config = json.loads((m2 / "resolved_config.json").read_text())
    selection = json.loads((m3 / "selection.json").read_text())
    tracking = json.loads((m4 / "summary.json").read_text())
    sources = [
        m2 / "manifest.json",
        m2 / "summary.json",
        m3 / "selection.json",
        m4 / "summary.json",
        m4 / "cost_audit.json",
        m4 / "tracking_cpu_cost.csv",
        m4 / "prediction_freeze.json",
        m4 / "tracking_windows.csv",
    ]
    hashes_before = {str(path.relative_to(run_root)): file_hash(path) for path in sources}
    sizes = {
        key: sum((m2 / e[key]).stat().st_size for e in manifest["trajectories"])
        for key in ("observations_file", "oracle_file", "counts_file", "rng_file")
    }
    allocation = allocate_panel_costs(
        summary,
        manifest,
        sizes,
        config["measurement"]["samples_per_prompt_grid"],
        config["measurement"]["noise_replicas"],
    )
    write_json(out / "allocation_basis.json", allocation)
    benchmark = _replay_benchmark(m2, manifest, selection, out)
    write_csv(out / "fixed_anchor_replay_benchmark.csv", benchmark)
    benchmarks = defaultdict(list)
    for row in benchmark:
        benchmarks[(row["method"], str(row["rank_cap"]), str(row["n"]))].append(row)
    benchmark_means = []
    for key, values in benchmarks.items():
        benchmark_means.append(
            {
                "method": key[0],
                "rank_cap": key[1],
                "n": key[2],
                "measurement_label": "MEASURED_EXISTING_ARTIFACT_REPLAY",
                "seed": 301,
                "arm": "X_BASE",
                "anchor": 8,
                "H": 16,
                "repeat_count": len(values),
                **{
                    field: float(np.mean([r[field] for r in values]))
                    for field in (
                        "read_seconds",
                        "fit_seconds",
                        "predict_seconds",
                        "serialization_seconds",
                        "disk_write_fsync_seconds",
                        "elapsed_total_seconds",
                    )
                },
                "all_repeat_array_hashes_equal": len({r["array_hash"] for r in values}) == 1,
            }
        )
    write_csv(out / "replay_benchmark_means.csv", benchmark_means)
    tracked = tracking["aggregate_metrics"]
    cost_rows, horizons = [], config["tracking"]["horizons"]
    for bench in benchmark_means:
        if bench["method"] == "DENSE":
            continue
        dense = next(b for b in benchmark_means if b["method"] == "DENSE" and b["n"] == bench["n"])
        panels = 0 if bench["method"] == "B0_PERSISTENCE" else 24
        finite = bench["n"] != "exact"
        measure = allocation["finite_count_panel_seconds"] if finite else 0.0
        io_cost = allocation["exact_panel_output_io_seconds"] + (
            allocation["finite_count_panel_output_io_seconds"] if finite else 0.0
        )
        paid_panel = allocation["exact_panel_compute_seconds"] + measure + io_cost
        branch = allocation["branch_operation_seconds"]
        for h in horizons:
            matching = [
                r
                for r in tracked
                if r["method"] == bench["method"]
                and str(r["rank_cap"]) == bench["rank_cap"]
                and str(r["n"]) == bench["n"]
                and r["telemetry"] == "PROJECTION_AND_PATH"
            ]
            qualified_h = max((r["H"] for r in matching if r["meets_tolerance"]), default=0)
            calibration = panels * (paid_panel + branch) + bench["fit_seconds"]
            # Only H16 replay is measured. Smaller-H replay components are
            # explicitly proportional allocation estimates, not measured runs.
            model_replay = bench["read_seconds"] + (h / 16) * (
                bench["predict_seconds"]
                + bench["serialization_seconds"]
                + bench["disk_write_fsync_seconds"]
            )
            dense_replay = (h / 16) * (
                dense["read_seconds"]
                + dense["serialization_seconds"]
                + dense["disk_write_fsync_seconds"]
            )
            model_cost = calibration + paid_panel + model_replay
            dense_cost = h * paid_panel + dense_replay
            cost_rows.append(
                {
                    "method": bench["method"],
                    "rank_cap": bench["rank_cap"],
                    "n": bench["n"],
                    "H": h,
                    "fit_budget_required": 0 if panels == 0 else 8,
                    "calibration_panels_required": panels,
                    "cost_label": "ALLOCATED_ESTIMATE_WITH_MEASURED_REPLAY",
                    "calibration_exact_compute_seconds": panels
                    * allocation["exact_panel_compute_seconds"],
                    "calibration_branch_seconds": panels * branch,
                    "calibration_counts_seconds": panels * measure,
                    "calibration_M2_output_io_seconds": panels * io_cost,
                    "frozen_fit_replay_seconds": bench["fit_seconds"],
                    "refresh_full_panel_seconds": paid_panel,
                    "tracking_read_predict_output_seconds": model_replay,
                    "dense_read_output_seconds": dense_replay,
                    "model_full_accounted_cpu_seconds": model_cost,
                    "dense_full_accounted_cpu_seconds": dense_cost,
                    "model_minus_dense_seconds": model_cost - dense_cost,
                    "all_included_cost_components": (
                        "exact panel + branch + counts + M2 write allocation + read + fit "
                        "+ predict + output serialization/fsync"
                    ),
                    "largest_tested_qualified_H": qualified_h or None,
                    "cpu_cost_evidence_status": (
                        "ACCOUNTED_ALLOCATION_ESTIMATE_NOT_PER_CONFIGURATION_MEASUREMENT"
                    ),
                    "real_Qwen_seconds": None,
                }
            )
    write_csv(out / "full_cpu_cost_allocated.csv", cost_rows)
    amortization = []
    low_budget_evidence = "No matching M4 windows; M3 bank ablations use n64 main rule"
    for banks in (1, 2, 4, 8):
        abstract = cost_amortization(banks, 16)
        for n in ("exact", 16, 64, 256):
            method = "B5_EXACT_SELECTED_DIAGNOSTIC"
            available = [
                r
                for r in tracked
                if r["method"] == method
                and str(r["n"]) == str(n)
                and r["telemetry"] == "PROJECTION_AND_PATH"
            ]
            verified = (
                sorted(r["H"] for r in available if r["meets_tolerance"]) if banks == 8 else []
            )
            status = (
                "UNVERIFIED_ACCURACY_AT_BREAK_EVEN"
                if banks < 8
                else "NOT_COST_FEASIBLE_AT_TESTED_HORIZON"
            )
            amortization.append(
                {
                    "method": method,
                    "n": n,
                    "fit_banks": banks,
                    "calibration_panels": banks * 3,
                    "abstract_break_even_H": abstract["break_even_horizon"],
                    "maximum_tested_tracking_H": 16 if banks == 8 else None,
                    "qualified_tracking_H": verified,
                    "break_even_within_verified_accurate_range": False,
                    "accuracy_at_break_even_status": status,
                    "low_budget_available_evidence": low_budget_evidence
                    if banks < 8
                    else "M4 frozen windows",
                    "abstract_model_units_at_H16": abstract["model_cost_at_horizon"],
                    "abstract_dense_units_at_H16": abstract["dense_cost_at_horizon"],
                }
            )
    write_csv(out / "bank_budget_break_even_qualification.csv", amortization)
    impact = step_status_impact(m4 / "tracking_windows.csv")
    write_json(out / "step_status_source_fix_impact.json", impact)
    unchanged = {
        str(path.relative_to(run_root)): file_hash(path) for path in sources
    } == hashes_before
    if not unchanged:
        raise RuntimeError("original evidence changed during read-only cost supplement")
    evidence = {
        "status": "COMPLETED",
        "source_hashes": hashes_before,
        "original_evidence_unchanged": unchanged,
        "wall_seconds": time.perf_counter() - started,
        "new_observations": 0,
        "new_training_steps": 0,
        "replay_benchmark_rows": len(benchmark),
        "allocated_cpu_cost_rows": len(cost_rows),
        "bank_budget_rows": len(amortization),
        "step_status_fix_impact": impact,
        "measured_scope": (
            "Fixed existing seed301 X_BASE anchor8 H16, frozen rules, n "
            "exact/16/64/256 noise0, three repeat "
            "reads/fits/predictions/writes"
        ),
        "unmeasured_terms_are_not_zero": (
            "M2 n-specific count generation and per-file write timings "
            "unavailable; measured aggregate totals are explicitly allocated"
        ),
        "all_replay_hashes_repeat": all(
            b["all_repeat_array_hashes_equal"] for b in benchmark_means
        ),
        "frozen_model_selection_unchanged": True,
        "source_script_sha256": file_hash(Path(__file__)),
        "low_budget_evidence_scope": low_budget_evidence,
        "no_cost_savings_claim_for_low_budget": True,
        "main_abstract_result": (
            "24 calibration panels + 1 refresh = 25 units > H16 dense 16; "
            "break-even H25 exceeds tested H16"
        ),
        "output_limit_bytes": 5 * 1024**2,
    }
    write_json(out / "summary.json", evidence)
    output_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if output_bytes > 5 * 1024**2:
        raise RuntimeError("cost supplement exceeded output limit")
    evidence["output_bytes"] = output_bytes
    write_json(out / "summary.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_cost_supplement(args.run_root, args.out), ensure_ascii=False))
