"""Post-seal CPU timing audit; never collects samples or reads oracle outcomes.

Every configuration gets a new preparation/coefficient cache. Input files are
loaded before the timed fit, so "cold" means cold algorithmic calibration state,
not a cold operating-system page cache or fresh BLAS process. Sampling/scoring
timings come from the existing cold-baseline ledgers and remain distinguished
from historical acquisition timings that were never recorded.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np

from src.modeling_qualification.models import array_hash

from .cost import break_even
from .cost_study import priced
from .study import configuration_id, contrast_inputs, digest, fwrite, jwrite

FINITE_MODELS = ("C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6")
SELECTION_SEEDS = (201, 202, 203, 204)
CONFIG_FIELDS = ("method", "observation", "n", "rank", "alpha", "eta", "fit_banks")


def instrument_cpu_components(ledger, method):
    """Recorded CPU operation elapsed times, with method-specific nesting."""
    sampling = float(ledger.get("sampling_seconds", 0.0))
    scoring = float(ledger.get("score_seconds", 0.0))
    forward = float(ledger.get("forward_seconds", 0.0))
    if not all(np.isfinite(v) and v >= 0 for v in (sampling, scoring, forward)):
        raise ValueError("Finite nonnegative instrument timings required")
    method = method.replace("-", "_")
    # ORIGIN performs the support check (and lazy policy forward) before entering
    # timed sample/score calls. All other instruments load inside those calls.
    outside_forward = forward if method == "O_LR_ORIGIN" else 0.0
    elapsed = sampling + scoring + outside_forward
    historical = method == "O_IND" and (
        ledger.get("reused_actions", 0) > 0 or ledger.get("historical_unique_policy_count", 0) > 0
    )
    return {
        "sampling_seconds": sampling,
        "score_seconds": scoring,
        "forward_seconds": forward,
        "additional_nonoverlapping_forward_seconds": outside_forward,
        "timed_cpu_seconds": elapsed,
        "cold_acquisition_cpu_seconds": None if historical else elapsed,
        "status": "HISTORICAL_ACQUISITION_CPU_UNMEASURED"
        if historical
        else "RECORDED_PRIMITIVE_CPU_TIMES",
        "coverage": "primitive call elapsed time; packet construction and orchestration excluded",
        "known_label_lookup_outside_score_timer": method == "D_KNOWN_EVENT_LOGP",
    }


def operation_slack(calibration, direct, ratio, *, fit_query_cpu_seconds):
    """Remaining operation-cost budget; never assume seconds equal generation units."""
    lower = priced(calibration, ratio)
    upper = priced(calibration, ratio, historical_label_bound="upper")
    direct_price = priced(direct, ratio)
    slack = direct_price - lower
    return {
        "score_to_generation_ratio": ratio,
        "label_to_generation_ratio": 0.05,
        "calibration_operation_lower_bound": lower,
        "calibration_operation_upper_bound": upper,
        "direct_operation_cost_four_queries": direct_price,
        "remaining_fit_plus_four_queries_generation_units": slack,
        "remaining_units_at_label_upper_bound": direct_price - upper,
        "max_cpu_second_price_generation_units": slack / fit_query_cpu_seconds
        if slack >= 0 and fit_query_cpu_seconds > 0
        else None,
        "cpu_price_conversion_is_assumed": False,
    }


def audit_fit_query(
    theta, packet, config, group_map, *, query_repeats=16, expected_prediction_hash=None
):
    """Fresh calibration setup + fit and separately timed four-bank predictions."""
    from .estimators import fit_contrast_model, prepare_contrast
    from .joint_packets import fit_packet_covariance, retained_fit_rows

    if config["method"] not in FINITE_MODELS or config["n"] <= 0:
        raise ValueError("Only sealed finite C2-C6 configurations are eligible")
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (14, 3, 737) or not np.isfinite(theta).all():
        raise ValueError("Frozen parent 14-bank/3-candidate/737-parameter inputs required")
    if type(query_repeats) is not int or query_repeats < 1:
        raise ValueError("Positive warm query repeat count required")
    banks = int(config["fit_banks"])
    if not 1 <= banks <= 8:
        raise ValueError("Only fit banks 0..7 may calibrate a model")
    method, n = config["method"], config["n"]
    started = time.perf_counter_ns()
    process_started = time.process_time_ns()
    e = contrast_inputs(theta[:banks]).reshape(2 * banks, 737)
    response = packet["helmert"][0, :banks].reshape(2 * banks, -1, 3)
    covariance = fit_packet_covariance(packet, 0, range(banks))
    active_rows = retained_fit_rows(packet, 0, range(banks))
    prepared = prepare_contrast(
        e,
        response,
        covariance=covariance if method in ("C3", "C5", "C6") else None,
        eta=config["eta"],
        observation_n=n,
        active_rows=active_rows,
    )
    setup_process = (time.process_time_ns() - process_started) / 1e9
    setup = (time.perf_counter_ns() - started) / 1e9
    started = time.perf_counter_ns()
    process_started = time.process_time_ns()
    model = fit_contrast_model(
        e,
        response,
        method,
        config["rank"],
        config["alpha"],
        prepared=prepared,
        eta=config["eta"],
        group_map=group_map,
        observation_n=n,
    )
    fitting_process = (time.process_time_ns() - process_started) / 1e9
    fitting = (time.perf_counter_ns() - started) / 1e9

    def query():
        update = contrast_inputs(theta[10:14]).reshape(8, 737)
        return model["predict"](update).reshape(4, 2, -1, 3)

    started = time.perf_counter_ns()
    process_started = time.process_time_ns()
    prediction = query()
    query_process = (time.process_time_ns() - process_started) / 1e9
    query_first = (time.perf_counter_ns() - started) / 1e9
    prediction_hash = array_hash(prediction)
    warm = []
    for _ in range(query_repeats):
        started = time.perf_counter_ns()
        repeated = query()
        warm.append((time.perf_counter_ns() - started) / 1e9)
        if array_hash(repeated) != prediction_hash:
            raise ValueError("Repeated pure prediction changed its output")
    return {
        "parameter_count": 737,
        "actual_rank": int(model["r"]),
        "k": int(model["k"]),
        "model_status": model["status"],
        "setup_cpu_seconds": setup,
        "fit_cpu_seconds": fitting,
        "calibration_fit_cpu_seconds": setup + fitting,
        "query_first_cpu_seconds": query_first,
        "query_warm_cpu_seconds": warm,
        "query_warm_median_cpu_seconds": float(np.median(warm)),
        "setup_process_cpu_seconds": setup_process,
        "fit_process_cpu_seconds": fitting_process,
        "query_first_process_cpu_seconds": query_process,
        "prediction_hash": prediction_hash,
        "prediction_matches_sealed": None
        if expected_prediction_hash is None
        else prediction_hash == expected_prediction_hash,
        "cross_configuration_preparation_cache_reused": False,
        "cache_boundary": (
            "loaded input arrays; fresh Q/covariance/coefficient preparation per config"
        ),
        "setup_contains_joint_covariance_and_prepare": True,
        "timing_units": "elapsed seconds on CPU; process CPU seconds separately reported",
        "four_query_banks": [10, 11, 12, 13],
        "noise_replica": 0,
    }


def discover_sealed_configs(n2_root):
    """Read prediction seals only, excluding metrics, truth and oracle diagnostics."""
    root = Path(n2_root)
    units, inputs = {}, {}
    for stage in ("N2B", "N2C", "N2E"):
        for seed in SELECTION_SEEDS:
            for arm in ("X_BASE", "X_VALID"):
                trajectory = f"seed{seed}_{arm}"
                for ai in range(3):
                    path = root / stage / trajectory / f"a{ai}" / "freeze.json"
                    if not path.exists():
                        continue
                    seal = json.loads(path.read_text())
                    if seal.get("frozen_before_scoring") is not True:
                        raise ValueError("CPU audit requires sealed predictions")
                    meta_path = path.parent / seal["metadata_file"]
                    if (
                        meta_path.parent != path.parent
                        or digest(meta_path) != seal["metadata_sha256"]
                    ):
                        raise ValueError("Sealed metadata path/hash mismatch")
                    with gzip.open(meta_path, "rt") as stream:
                        meta = json.load(stream)
                    if meta.get("frozen_before_scoring") is not True:
                        raise ValueError("CPU audit requires sealed metadata")
                    inputs[str(path)] = digest(path)
                    inputs[str(meta_path)] = digest(meta_path)
                    key = (trajectory, ai)
                    unit = units.setdefault(
                        key,
                        {
                            "trajectory_id": trajectory,
                            "seed": seed,
                            "arm": arm,
                            "anchor_index": ai,
                            "configs": {},
                        },
                    )
                    for row in meta["models"]:
                        if (
                            row.get("noise_replica") != 0
                            or row.get("method") not in FINITE_MODELS
                            or row.get("n", 0) <= 0
                        ):
                            continue
                        if (
                            row["seed"] != seed
                            or row["arm"] != arm
                            or row["anchor"] != (8, 24, 40)[ai]
                        ):
                            raise ValueError("Sealed model identity does not match its unit")
                        config = {name: row[name] for name in CONFIG_FIELDS}
                        cid = configuration_id(config)
                        if cid in unit["configs"]:
                            if (
                                unit["configs"][cid]["expected_prediction_hash"]
                                != row["prediction_hash"]
                            ):
                                raise ValueError(
                                    "Same sealed configuration has inconsistent predictions"
                                )
                            unit["configs"][cid]["sealed_stages"].append(stage)
                        else:
                            unit["configs"][cid] = dict(
                                config,
                                configuration_id=cid,
                                expected_prediction_hash=row["prediction_hash"],
                                sealed_stages=[stage],
                                expected_packet_hash=meta.get("packet_hashes", {}).get(
                                    str((config["observation"], config["n"]))
                                ),
                            )
    result = []
    for unit in units.values():
        unit["configs"] = list(unit["configs"].values())
        if unit["configs"]:
            result.append(unit)
    return {
        "units": result,
        "input_files": inputs,
        "source": "sealed N2B/N2C/N2E noise0 only",
        "oracle_read": False,
    }


def _packet_path(run_root, unit, config):
    relative = (
        Path(unit["trajectory_id"])
        / f"a{unit['anchor_index']}"
        / f"{config['observation']}_n{config['n']}"
    )
    candidates = [run_root / "N1/packets" / relative, run_root / "N2/packets" / relative]
    expected = config.get("expected_packet_hash")
    for candidate in candidates:
        path = candidate / "packet_arrays.npz"
        if path.exists() and (expected is None or digest(path) == expected):
            return candidate
    raise ValueError("Missing/hash-mismatched sealed paid packet")


def run_cpu_cost_audit(parent_root, run_root, out, *, query_repeats=16):
    """Run only after the completed N2 seal exists; produces no new policy calls."""
    from .observation_study import load_packet
    from .targets import group_xv_matrix

    parent_root, run_root, out = Path(parent_root), Path(run_root), Path(out)
    n2 = run_root / "N2"
    manifest_path = n2 / "RUN_MANIFEST.json"
    if (
        not manifest_path.exists()
        or json.loads(manifest_path.read_text()).get("status") != "COMPLETE"
    ):
        raise ValueError("N2_NOT_COMPLETE; CPU audit must wait for completed sealed configurations")
    plan = discover_sealed_configs(n2)
    if len(plan["units"]) != 24:
        raise ValueError("All 24 selection seed/arm/anchor units are required")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    inputs = dict(plan["input_files"])
    source_root = Path(__file__).parent
    for name in (
        "cpu_cost_audit.py",
        "estimators.py",
        "covariance.py",
        "joint_packets.py",
        "targets.py",
        "cost_study.py",
        "packet_codec.py",
        "packet_reconstruction.py",
        "observation_study.py",
        "study.py",
    ):
        source = source_root / name
        inputs[str(source)] = digest(source)
    inputs[str(manifest_path)] = digest(manifest_path)
    parent_manifest_path = parent_root / "manifest.json"
    manifest = json.loads(parent_manifest_path.read_text())
    inputs[str(parent_manifest_path)] = digest(parent_manifest_path)
    entries = {entry["id"]: entry for entry in manifest["trajectories"]}
    metadata_path = parent_root / "probe_metadata.json"
    group_map = group_xv_matrix(json.loads(metadata_path.read_text()))
    inputs[str(metadata_path)] = digest(metadata_path)
    fit_rows, comparison_rows = [], []
    for unit in plan["units"]:
        entry = entries[unit["trajectory_id"]]
        parameter_path = parent_root / entry["observations_file"]
        actual = digest(parameter_path)
        if actual != entry["sha256"]["observations_file"]:
            raise ValueError("Frozen parameter hash mismatch")
        inputs[str(parameter_path)] = actual
        with np.load(parameter_path, allow_pickle=False) as saved:
            theta = saved["branch_theta"][unit["anchor_index"]].copy()
        packets, costs = {}, {}
        for config in unit["configs"]:
            key = (config["observation"], config["n"])
            if key not in packets:
                path = _packet_path(run_root, unit, config)
                packets[key] = load_packet(path)
                for name in ("packet_arrays.npz", "packet_metadata.json.gz", "packets.json"):
                    source = path / name
                    if source.exists():
                        inputs[str(source)] = digest(source)
                cost_path = (
                    n2
                    / "cost"
                    / unit["trajectory_id"]
                    / f"a{unit['anchor_index']}"
                    / f"{key[0]}_n{key[1]}.json"
                )
                cost = json.loads(cost_path.read_text())
                inputs[str(cost_path)] = digest(cost_path)
                if (
                    cost.get("cold_cache_each_baseline") is not True
                    or cost.get("direct_replay_matches_paid_packet") is not True
                ):
                    raise ValueError("Cold direct timing lacks its paid-packet replay check")
                costs[key] = cost
            timing = audit_fit_query(
                theta,
                packets[key],
                config,
                group_map,
                query_repeats=query_repeats,
                expected_prediction_hash=config["expected_prediction_hash"],
            )
            identity = {
                name: unit[name] for name in ("trajectory_id", "seed", "arm", "anchor_index")
            }
            identity.update({name: config[name] for name in CONFIG_FIELDS})
            identity["configuration_id"] = config["configuration_id"]
            fit_rows.append(dict(identity, **timing))
            calibration = costs[key]["fit_by_bank_budget"][str(config["fit_banks"])]
            instrument = instrument_cpu_components(calibration, key[0])
            fit_query = timing["calibration_fit_cpu_seconds"] + timing["query_first_cpu_seconds"]
            baselines = [("D_DIRECT", costs[key]["direct_four_evaluation_banks"], key[0])]
            if "LR" in key[0]:
                baselines.append(
                    (
                        "D_KNOWN_EVENT_LOGP",
                        costs[key]["known_event_four_evaluation_banks"],
                        "D_KNOWN_EVENT_LOGP",
                    )
                )
            for baseline, direct, method in baselines:
                direct_cpu = instrument_cpu_components(direct, method)
                acquisition = instrument["cold_acquisition_cpu_seconds"]
                total = None if acquisition is None else acquisition + fit_query
                crossing = (
                    None
                    if total is None
                    else break_even(
                        acquisition + timing["calibration_fit_cpu_seconds"],
                        timing["query_first_cpu_seconds"] / 4,
                        direct_cpu["timed_cpu_seconds"] / 4,
                    )["queries"]
                )
                for ratio in (0.05, 0.25, 1.0):
                    comparison_rows.append(
                        dict(
                            identity,
                            comparison_baseline=baseline,
                            access_regime="SAMPLE_AND_LOGP" if "LR" in key[0] else "SAMPLE_ONLY",
                            calibration_instrument_cpu=instrument,
                            direct_instrument_cpu=direct_cpu,
                            calibration_instrument_cpu_seconds=acquisition,
                            calibration_fit_cpu_seconds=timing["calibration_fit_cpu_seconds"],
                            query_four_banks_cpu_seconds=timing["query_first_cpu_seconds"],
                            surrogate_total_recorded_cpu_seconds=total,
                            surrogate_cpu_seconds_lower_bound=instrument["timed_cpu_seconds"]
                            + fit_query,
                            direct_recorded_cpu_seconds=direct_cpu["timed_cpu_seconds"],
                            cpu_slack_timed_components_seconds=None
                            if total is None
                            else direct_cpu["timed_cpu_seconds"] - total,
                            cpu_break_even_queries_timed_components=crossing,
                            cpu_comparison_scope=(
                                "recorded primitives plus cold fit/query; "
                                "unrecorded orchestration excluded"
                            ),
                            **operation_slack(
                                calibration, direct, ratio, fit_query_cpu_seconds=fit_query
                            ),
                        )
                    )
        print(
            json.dumps(
                {
                    "phase": "CPU_COST_AUDIT",
                    "trajectory": unit["trajectory_id"],
                    "anchor_index": unit["anchor_index"],
                    "configs": len(unit["configs"]),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    fwrite(out / "CPU_COST_BY_UNIT.csv", comparison_rows)
    with gzip.open(out / "CPU_FIT_QUERY_RAW.jsonl.gz", "wt") as stream:
        for row in fit_rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    grouped = {}
    for row in fit_rows:
        grouped.setdefault(row["configuration_id"], []).append(row)
    by_configuration = []
    for cid, rows in sorted(grouped.items()):
        by_configuration.append(
            {
                "configuration_id": cid,
                "unit_count": len(rows),
                "cold_fit_cpu_seconds_mean": float(
                    np.mean([row["calibration_fit_cpu_seconds"] for row in rows])
                ),
                "cold_fit_cpu_seconds_median": float(
                    np.median([row["calibration_fit_cpu_seconds"] for row in rows])
                ),
                "four_query_first_cpu_seconds_mean": float(
                    np.mean([row["query_first_cpu_seconds"] for row in rows])
                ),
                "four_query_warm_cpu_seconds_median": float(
                    np.median([row["query_warm_median_cpu_seconds"] for row in rows])
                ),
                "all_predictions_match_sealed": all(
                    row["prediction_matches_sealed"] for row in rows
                ),
            }
        )
    summary = {
        "status": "PASS"
        if all(row["prediction_matches_sealed"] for row in fit_rows)
        else "SEALED_PREDICTION_REPRODUCTION_MISMATCH",
        "selection_seeds": list(SELECTION_SEEDS),
        "unit_count": 24,
        "unique_fit_evaluations": len(fit_rows),
        "comparison_rows": len(comparison_rows),
        "cold_fit_cpu_seconds_total": sum(row["calibration_fit_cpu_seconds"] for row in fit_rows),
        "first_query_cpu_seconds_total": sum(row["query_first_cpu_seconds"] for row in fit_rows),
        "warm_query_repeats": query_repeats,
        "all_predictions_match_sealed": all(row["prediction_matches_sealed"] for row in fit_rows),
        "wall_seconds": time.perf_counter() - started,
        "cpu_is_737_parameter_toy_only": True,
        "no_new_policy_samples": True,
        "new_optimizer_updates": 0,
        "oracle_outcomes_read": False,
        "primary_science_ranking_modified": False,
        "by_configuration": by_configuration,
        "numpy_version": np.__version__,
        "historical_cpu_acquisition_is_unmeasured_not_zero": True,
        "recorded_instrument_cpu_omits_packet_construction_and_general_orchestration": True,
        "operation_units_and_cpu_seconds_not_equated": True,
        "input_file_count": len(inputs),
        "outputs": {
            "CPU_COST_BY_UNIT.csv": digest(out / "CPU_COST_BY_UNIT.csv"),
            "CPU_FIT_QUERY_RAW.jsonl.gz": digest(out / "CPU_FIT_QUERY_RAW.jsonl.gz"),
        },
    }
    jwrite(out / "CPU_COST_INPUT_HASHES.json", inputs)
    jwrite(out / "CPU_COST_SUMMARY.json", summary)
    return summary
