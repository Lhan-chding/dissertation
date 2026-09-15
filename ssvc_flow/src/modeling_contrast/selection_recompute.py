"""Independent N3 arithmetic, paired seed bootstrap, gates and byte bindings.

No production selection/evaluation helper is imported. This verifier consumes
the sealed N2 sufficient-statistic records; verification of predictions against
raw oracle arrays is a separate explicitly reported upstream evidence layer.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

KEY = ("seed", "arm", "anchor", "noise_replica")
MODELS = {"C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"}


def canonical(value):
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return json.load(stream)


def bootstrap_nrmse(method, baseline, seeds, replicates, random_seed):
    """Independent pooled-energy bootstrap, one draw index per training seed."""
    method, baseline = np.asarray(method, float), np.asarray(baseline, float)
    seeds = np.asarray(seeds)
    if method.shape != baseline.shape or method.shape != (len(seeds), 2):
        raise ValueError("paired [error_ss,truth_ss] and seed identities required")
    if not np.isfinite(method).all() or not np.isfinite(baseline).all():
        raise ValueError("nonfinite paired energies")
    if (method < 0).any() or (baseline < 0).any():
        raise ValueError("negative energies")
    if not np.allclose(method[:, 1], baseline[:, 1], rtol=1e-12, atol=0):
        return {"status": "PAIRED_TRUTH_ENERGY_MISMATCH", "cluster_count": 0}
    unique = sorted(set(seeds.tolist()))
    left = np.array([method[seeds == s].sum(0) for s in unique])
    right = np.array([baseline[seeds == s].sum(0) for s in unique])
    if not unique:
        return {"status": "NO_PAIRED_CLUSTERS", "cluster_count": 0}
    if np.any(left[:, 1] <= 0) or np.any(right[:, 1] <= 0):
        return {"status": "NO_SIGNAL_IN_SEED_CLUSTER", "cluster_count": len(unique)}
    chosen = np.random.default_rng(random_seed).integers(0, len(unique), (replicates, len(unique)))
    a, b = left[chosen].sum(1), right[chosen].sum(1)
    values = np.sqrt(a[:, 0] / a[:, 1]) - np.sqrt(b[:, 0] / b[:, 1])
    point = np.sqrt(left[:, 0].sum() / left[:, 1].sum()) - np.sqrt(
        right[:, 0].sum() / right[:, 1].sum()
    )
    per = np.sqrt(left[:, 0] / left[:, 1]) - np.sqrt(right[:, 0] / right[:, 1])
    return {
        "status": "POSTHOC_DEVELOPMENT_SEED_CLUSTER_BOOTSTRAP"
        if len(unique) > 1
        else "INSUFFICIENT_SEED_CLUSTERS",
        "cluster_count": len(unique),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": random_seed,
        "paired_mean": float(point),
        "ci025": float(np.quantile(values, 0.025)),
        "ci975": float(np.quantile(values, 0.975)),
        "per_seed": [
            {"seed": s, "paired_difference": float(v), "row_count": int((seeds == s).sum())}
            for s, v in zip(unique, per, strict=True)
        ],
    }


def recompute_candidates(rows, config):
    seeds = set(config["data_roles"]["legacy_selection_seeds"])
    primary = config["primary_target"]["name"]
    unique = {}
    duplicate_count = 0
    for row in rows:
        if row["seed"] not in seeds or row["target"] != primary or row["population"] != "all":
            continue
        key = (row["configuration_id"], *(row[k] for k in KEY))
        if key in unique:

            def strip(r):
                return {k: v for k, v in r.items() if k != "stage"}

            if canonical(strip(row)) != canonical(strip(unique[key])):
                raise ValueError(f"conflicting duplicate N2 record: {key}")
            duplicate_count += 1
        else:
            unique[key] = row
    rows = list(unique.values())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration_id"]].append(row)
    expected = {
        (s, a, h, n)
        for s in seeds
        for a in config["fresh_cpu"]["main_arms"]
        for h in config["branches"]["anchors"]
        for n in range(config["observation"]["noise_replicas"])
    }
    candidates = []
    bootstraps = []
    for cid, block in grouped.items():
        first = block[0]
        if first["method"] not in MODELS or first["n"] not in (64, 256):
            continue
        if first["observation"] not in config["observation"]["methods"]:
            continue
        got = {
            "configuration_id": cid,
            "model": first["method"],
            "method": first["observation"],
            "n": first["n"],
            "actual_rank": max(r["actual_rank"] for r in block),
            "predicted_difference_norm": math.sqrt(
                sum(r["predicted_difference_norm"] ** 2 for r in block)
            ),
            "selection_coverage_complete": {tuple(r[k] for k in KEY) for r in block} == expected,
        }
        for q in ("pX", "v"):
            error = sum(r[q + "_error_ss"] for r in block)
            truth = sum(r[q + "_truth_ss"] for r in block)
            got[q + "_nrmse"] = math.sqrt(error / truth) if truth > 0 else None
        for name, model in [("C0", "C0"), ("C1", "C1_LEGACY_EXACT_RULE")]:
            base = [r for r in rows if r["method"] == model and r["n"] == first["n"]]
            fallback = not base
            if fallback:
                base = [r for r in rows if r["method"] == model and r["n"] == 0]
            base_ids = {r["configuration_id"] for r in base}
            keys = KEY[:-1] if fallback else KEY
            mapped = {tuple(r[k] for k in keys): r for r in base}
            wanted = [tuple(r[k] for k in keys) for r in block]
            stable = []
            for q in ("pX", "v"):
                if len(base_ids) != 1:
                    result = {
                        "status": "MISSING_BASELINE"
                        if not base_ids
                        else "AMBIGUOUS_BASELINE_CONFIGURATION",
                        "cluster_count": 0,
                    }
                elif set(wanted) != set(mapped) or len(mapped) != len(base):
                    result = {"status": "INCOMPLETE_PAIRED_BLOCKS", "cluster_count": 0}
                else:
                    result = bootstrap_nrmse(
                        [[r[q + "_error_ss"], r[q + "_truth_ss"]] for r in block],
                        [[mapped[k][q + "_error_ss"], mapped[k][q + "_truth_ss"]] for k in wanted],
                        [r["seed"] for r in block],
                        config["statistics"]["bootstrap_replicates"],
                        config["statistics"]["bootstrap_seed"],
                    )
                bootstraps.append(
                    {"configuration_id": cid, "baseline": name, "quantity": q, **result}
                )
                stable.append(
                    result.get("cluster_count") == len(seeds) and result.get("ci975", math.inf) < 0
                )
            got["stable_improvement_" + name] = got["selection_coverage_complete"] and all(stable)
        got["science_without_nonalias"] = (
            got["actual_rank"] >= 1
            and got["predicted_difference_norm"] > 0
            and all(
                got[q + "_nrmse"] is not None
                and 0 <= got[q + "_nrmse"] < config["statistics"]["nrmse_" + q + "_max"]
                for q in ("pX", "v")
            )
            and got["stable_improvement_C0"]
            and got["stable_improvement_C1"]
        )
        candidates.append(got)
    return {
        "candidates": candidates,
        "bootstrap": bootstraps,
        "deduplicated_primary_rows": duplicate_count,
    }


def _cost_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("score_to_generation_ratio", "break_even_queries"):
            try:
                row[key] = float(row[key]) if row.get(key) else None
            except ValueError:
                row[key] = None
        for key in ("cost_feasible", "within_domain", "includes_fit_score_label"):
            row[key] = str(row.get(key)).lower() == "true"
    return rows


def _verify_summary_csv(path, records, seeds, by_seed, equal):
    keys = ["configuration_id", "target", "population"] + (["seed"] if by_seed else [])
    unique = {}
    for row in records:
        if row["seed"] not in seeds:
            continue
        key = tuple(row[k] for k in ("configuration_id", "target", "population", *KEY))
        if key in unique:
            if canonical({k: v for k, v in row.items() if k != "stage"}) != canonical(
                {k: v for k, v in unique[key].items() if k != "stage"}
            ):
                raise ValueError("conflicting full-panel metric duplicate")
        else:
            unique[key] = row
    groups = defaultdict(list)
    for row in unique.values():
        groups[tuple(str(row[k]) for k in keys)].append(row)
    with path.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    mapped = {tuple(row[k] for k in keys): row for row in table}
    equal(path.name + ".keys", sorted(groups), sorted(mapped))
    equal(path.name + ".unique_rows", len(table), len(mapped))
    for key, block in groups.items():
        actual = mapped.get(key, {})
        for q in ("pX", "v", "event"):
            error = sum(r[q + "_error_ss"] for r in block)
            truth = sum(r[q + "_truth_ss"] for r in block)
            values = {
                q + "_error_ss": error,
                q + "_truth_ss": truth,
                q + "_nrmse": math.sqrt(error / truth) if truth > 0 else None,
            }
            if q != "event":
                deviations = np.concatenate([r[q + "_abs_errors"] for r in block])
                values.update(
                    {
                        q + "_mae": float(deviations.mean()),
                        q + "_error_q95": float(np.quantile(deviations, 0.95)),
                    }
                )
            for field, value in values.items():
                text = actual.get(field)
                observed = float(text) if text else None
                equal(path.name + "." + str(key) + "." + field, value, observed)
    return len(table)


def run_selection_recompute(run_root, out):
    """Write N3_INDEPENDENT_RECOMPUTE.json and return its PASS/FAIL receipt."""
    run_root, out = Path(run_root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    n3 = run_root / "N3"
    lock_path = n3 / "MODEL_SELECTION_LOCK.json"
    lock = read_json(lock_path)
    errors = []
    checked = 0
    maximum = 0.0

    def equal(name, a, b):
        nonlocal checked, maximum
        checked += 1
        if (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and not isinstance(b, bool)
        ):
            difference = abs(a - b)
            maximum = max(maximum, difference)
            okay = (
                math.isfinite(a) and math.isfinite(b) and np.isclose(a, b, rtol=1e-11, atol=1e-12)
            )
        else:
            okay = a == b
        if not okay:
            errors.append({"field": name, "expected": a, "observed": b})

    equal(
        "lock_sha256",
        canonical({k: v for k, v in lock.items() if k != "lock_sha256"}),
        lock["lock_sha256"],
    )
    config_path = Path(__file__).resolve().parents[2] / "configs/modeling_contrast/protocol.json"
    config = read_json(config_path)
    equal("protocol_sha256", file_hash(config_path), lock["protocol_sha256"])
    binding = lock["binding"]
    for field, key in [
        ("source_hashes", "source"),
        ("input_hashes", "data"),
        ("packet_hashes", "packet"),
    ]:
        equal("binding." + key, canonical(lock[field]), binding[key])
    equal("binding.config", lock["protocol_sha256"], binding["config"])
    equal("binding.selector", canonical(lock["selected"]), binding["selector"])
    bound_files = {}
    for field in ("source_hashes", "input_hashes", "packet_hashes", "selector_source_hashes"):
        for filename, wanted in lock[field].items():
            path = Path(filename)
            if not path.is_absolute():
                path = config_path.parents[2] / path
            if str(path) not in bound_files:
                bound_files[str(path)] = file_hash(path) if path.is_file() else None
            equal("file:" + str(path), wanted, bound_files[str(path)])
    metrics_path = run_root / "N2/METRICS.jsonl.gz"
    selection_seeds = set(config["data_roles"]["legacy_selection_seeds"])
    with gzip.open(metrics_path, "rt") as stream:
        rows = [row for line in stream if (row := json.loads(line))["seed"] in selection_seeds]
    evidence = recompute_candidates(rows, config)
    table = read_json(n3 / "PAIRED_BOOTSTRAP.json")

    def bk(r):
        return (r["configuration_id"], r["baseline"], r["quantity"])

    actual_boot = {bk(r): r for r in table}
    equal("bootstrap_table_keys", sorted(bk(r) for r in evidence["bootstrap"]), sorted(actual_boot))
    for row in evidence["bootstrap"]:
        actual = actual_boot.get(bk(row), {})
        for field in (
            "status",
            "cluster_count",
            "bootstrap_replicates",
            "bootstrap_seed",
            "paired_mean",
            "ci025",
            "ci975",
        ):
            if field in row:
                equal(str(bk(row)) + "." + field, row[field], actual.get(field))
        for seed_row in row.get("per_seed", []):
            found = next(
                (v for v in actual.get("per_seed", []) if v["seed"] == seed_row["seed"]), {}
            )
            for field, value in seed_row.items():
                equal(
                    str(bk(row)) + ".per_seed." + str(seed_row["seed"]) + "." + field,
                    value,
                    found.get(field),
                )
    actual_candidates = {r["configuration_id"]: r for r in lock["candidate_audit"]}
    equal(
        "candidate_ids",
        sorted(r["configuration_id"] for r in evidence["candidates"]),
        sorted(actual_candidates),
    )
    # Actual nonalias coverage is independently derived from parent parameter arrays.
    manifests = []
    for filename in lock["input_hashes"]:
        path = Path(filename)
        if path.name == "manifest.json" and path.is_file():
            value = read_json(path)
            if isinstance(value, dict) and value.get("trajectories"):
                manifests.append((path, value))
    nonalias = {}
    if len(manifests) != 1:
        errors.append({"field": "parent_manifest_unique", "count": len(manifests)})
    else:
        path, manifest = manifests[0]
        for entry in manifest["trajectories"]:
            if entry["seed"] not in config["data_roles"]["legacy_selection_seeds"]:
                continue
            with np.load(path.parent / entry["observations_file"], allow_pickle=False) as z:
                theta = z["branch_theta"]
                anchors = z["anchors"]
            for ai, anchor in enumerate(anchors):
                nonalias[(entry["seed"], entry["arm"], int(anchor))] = int(
                    np.any(theta[ai, 10:14, 1] != theta[ai, 10:14, 0], axis=-1).sum()
                )
    costs = _cost_rows(run_root / "N2/COST_FRONTIER.csv")
    equal("cost_frontier_source", costs, lock["cost_frontier"])
    accepted = []
    science_count = 0
    for row in evidence["candidates"]:
        cid = row["configuration_id"]
        actual = actual_candidates.get(cid, {})
        for field in (
            "model",
            "method",
            "n",
            "actual_rank",
            "predicted_difference_norm",
            "pX_nrmse",
            "v_nrmse",
            "stable_improvement_C0",
            "stable_improvement_C1",
            "selection_coverage_complete",
        ):
            equal(cid + "." + field, row[field], actual.get(field))
        units = sum(nonalias.values())
        equal(cid + ".nonalias_eval_units", units, actual.get("nonalias_eval_units"))
        science = row["science_without_nonalias"] and units > 0
        equal(cid + ".science_pass", science, actual.get("science_pass"))
        science_count += science
        access = "SAMPLE_AND_LOGP" if row["method"].startswith("O_LR") else "SAMPLE_ONLY"
        favorable = []
        for cost in costs:
            q = cost.get("break_even_queries")
            if (
                cost["configuration_id"] == cid
                and cost["access_regime"] == access
                and cost["includes_fit_score_label"]
                and cost["within_domain"]
                and cost["score_to_generation_ratio"]
                in config["cost"]["score_to_generation_ratios"]
                and (
                    cost["cost_feasible"]
                    or (isinstance(q, (int, float)) and math.isfinite(q) and q >= 0)
                )
                and cost["comparison_baseline"]
                in ("D_DIRECT", "D_KNOWN_EVENT_LOGP", "D_FULL_FINITE_ENUMERATION")
                and (access != "SAMPLE_ONLY" or cost["comparison_baseline"] == "D_DIRECT")
            ):
                favorable.append(cost)
        equal(cid + ".cost_pass", bool(favorable), actual.get("cost_pass"))
        if science and favorable:
            accepted.append(row)
    accepted.sort(
        key=lambda r: (
            max(r["pX_nrmse"], r["v_nrmse"]),
            r["actual_rank"],
            r["n"],
            r["configuration_id"],
        )
    )
    equal(
        "selected_ids",
        [r["configuration_id"] for r in accepted[:2]],
        [r["configuration_id"] for r in lock["selected"]],
    )
    for row in lock["selected"]:
        equal(
            "selected_matches_audit." + row["configuration_id"],
            row,
            actual_candidates[row["configuration_id"]],
        )
    equal("science_gate", "PASS" if science_count else "FAIL", lock["science_gate"]["status"])
    equal("cost_gate", "PASS" if accepted else "FAIL", lock["cost_gate"]["status"])
    forecast = lock["resource_gate"]["forecast"]
    resource = True
    for field, limit in [
        ("total_walltime_hours", "max_total_walltime_hours"),
        ("peak_ram_gib", "max_ram_gib"),
        ("added_output_gib", "max_added_output_gib"),
        ("temporary_gib", "max_temporary_gib"),
    ]:
        value = forecast.get(field)
        resource &= (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= config["resources"][limit]
        )
    equal("resource_gate", "PASS" if resource else "FAIL", lock["resource_gate"]["status"])
    resource_path = n3 / "FRESH_RESOURCE_PROJECTION.json"
    projection = read_json(resource_path) if resource_path.is_file() else None
    if projection:
        values = projection["forecast"]
        expected_resource = {
            "total_walltime_hours": values["wall_seconds"] / 3600
            if values["wall_seconds"] is not None
            else None,
            "peak_ram_gib": values["peak_ram_gib"],
            "added_output_gib": values["added_output_bytes"] / 2**30,
            "temporary_gib": values["temporary_bytes"] / 2**30,
        }
    else:
        values = read_json(run_root / "smoke/smoke_result.json")["forecast"]
        expected_resource = {
            "total_walltime_hours": values["wall_seconds"] / 3600,
            "peak_ram_gib": values["peak_ram_gib"],
            "added_output_gib": max(values["added_output_bytes"], forecast["current_result_bytes"])
            / 2**30,
            "temporary_gib": values["temporary_bytes"] / 2**30,
        }
    for field, value in expected_resource.items():
        equal("resource_source." + field, value, forecast.get(field))
    allowed = all(
        lock[g]["status"] == "PASS"
        for g in ("science_gate", "cost_gate", "resource_gate", "input_identity_gate", "data_gate")
    )
    equal("allow_fresh_cpu", allowed, lock["allow_fresh_cpu"])
    gates = lock
    expected_status = (
        "BLOCKED_INPUT_IDENTITY"
        if gates["input_identity_gate"]["status"] != "PASS"
        else "STOP_FOR_REVIEW"
        if gates["data_gate"].get("fresh_seed_collision") is True
        else "BLOCKED_MISSING_LEGACY_INPUTS_OR_SEED_AUDIT"
        if gates["data_gate"]["status"] != "PASS"
        else "RESOURCE_REVIEW_REQUIRED"
        if not resource
        else "NO_ACCEPTABLE_MODEL"
        if not science_count
        else "NO_COST_FEASIBLE_SURROGATE"
        if not accepted
        else "FROZEN_FOR_CONDITIONAL_FRESH_CPU"
    )
    equal("decision_status", expected_status, lock["status"])
    equal("statistics", config["statistics"], lock["statistics"])
    for field in ("fresh_calibration_seeds", "fresh_locked_test_seeds"):
        equal(field, config["data_roles"][field], lock[field])
    summary_counts = {}
    for by_seed, name in [
        (False, "CONTRAST_SELECTION_POOLED.csv"),
        (True, "CONTRAST_SELECTION_BY_SEED.csv"),
    ]:
        summary_counts[name] = _verify_summary_csv(
            n3 / name, rows, set(config["data_roles"]["legacy_selection_seeds"]), by_seed, equal
        )
    status = "FAIL" if errors else "PASS"
    evidence_paths = [
        lock_path,
        metrics_path,
        n3 / "PAIRED_BOOTSTRAP.json",
        n3 / "CONTRAST_SELECTION_POOLED.csv",
        n3 / "CONTRAST_SELECTION_BY_SEED.csv",
        run_root / "N2/COST_FRONTIER.csv",
        config_path,
        run_root / "smoke/smoke_result.json",
    ]
    if resource_path.is_file():
        evidence_paths.append(resource_path)
    result = {
        "status": status,
        "comparisons": checked,
        "checks": {
            name: {"status": status, "joint_verification_receipt": True}
            for name in ("N3_BOOTSTRAP", "N3_POOLED", "N3_BINDINGS", "N3_GATES")
        },
        "errors": errors[:100],
        "error_count": len(errors),
        "max_absolute_numeric_difference": maximum,
        "candidate_count": len(evidence["candidates"]),
        "bootstrap_rows": len(evidence["bootstrap"]),
        "summary_rows": summary_counts,
        "evidence_files": {str(path.resolve()): file_hash(path) for path in evidence_paths},
        "bootstrap_replicates": config["statistics"]["bootstrap_replicates"],
        "lock_sha256": file_hash(lock_path),
        "metrics_sha256": file_hash(metrics_path),
        "bound_files_verified": len(bound_files),
        "independent_selection_helpers_called": False,
        "production_evaluation_helpers_called": False,
        "training_calls": 0,
        "sampling_calls": 0,
        "scope": [
            "N2 sufficient-statistic energy pooling",
            "paired whole-training-seed bootstrap",
            "nonalias parameter coverage",
            "N3 scientific/nonzero and cost eligibility",
            "selected ranking",
            "resource threshold arithmetic",
            "canonical and live byte bindings",
        ],
        "limitations": [
            "Raw predictions and oracle arrays are verified separately by recompute.py; "
            "this receipt does not repeat that verification.",
            "Cost-frontier source identity and eligibility are checked; physical cost "
            "measurement and extrapolation assumptions are not independently rerun.",
            "Resource thresholds and bound inputs are checked; future runtime and peak-memory "
            "forecasts are estimates, not measurements of an unrun experiment.",
            "Data-gate collection completeness and seed-collision audit are inherited "
            "from bound N0 artifacts.",
        ],
    }
    out.mkdir(parents=True)
    (out / "N3_INDEPENDENT_RECOMPUTE.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (out / "RECOMPUTATION_SUMMARY.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (out / "RECOMPUTED_BOOTSTRAP.json").write_text(
        json.dumps(evidence["bootstrap"], indent=2, allow_nan=False) + "\n"
    )
    return result
