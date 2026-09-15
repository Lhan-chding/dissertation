"""Build N3 decisions from sealed legacy metrics and complete paired evidence.

This module only reads N0-N2 artifacts. It neither fits models nor observes fresh
labels; every stability verdict comes from paired training-seed bootstrap data.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import (
    calibration_resolution,
    direction_grid,
    paired_seed_bootstrap,
    resolution_grid,
)
from .selection import canonical_hash, freeze_selection
from .study import TARGETS, aggregate_rows, configuration_id, digest, load_metadata

ROOT = Path(__file__).resolve().parents[2]
_KEY = ("seed", "arm", "anchor", "noise_replica")
_MODELS = {"C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"}


def _json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _deduplicate(rows):
    unique = {}
    duplicates = 0
    for row in rows:
        key = tuple(row[name] for name in ("configuration_id", "target", "population", *_KEY))
        if key in unique:
            # N2B and N2C can legitimately contain an identical C2 configuration.
            old = {k: v for k, v in unique[key].items() if k != "stage"}
            new = {k: v for k, v in row.items() if k != "stage"}
            if canonical_hash(old) != canonical_hash(new):
                raise ValueError(f"conflicting frozen metrics for repeated block: {key}")
            duplicates += 1
        else:
            unique[key] = row
    return list(unique.values()), duplicates


def _paired(method_rows, all_rows, baseline, quantity, config):
    name = "C0" if baseline == "C0" else "C1_LEGACY_EXACT_RULE"
    n = method_rows[0]["n"]
    baseline_rows = [row for row in all_rows if row["method"] == name and row["n"] == n]
    exact_fallback = False
    if not baseline_rows:
        baseline_rows = [row for row in all_rows if row["method"] == name and row["n"] == 0]
        exact_fallback = bool(baseline_rows)
    configurations = {row["configuration_id"] for row in baseline_rows}
    if len(configurations) != 1:
        return {
            "status": "MISSING_BASELINE"
            if not configurations
            else "AMBIGUOUS_BASELINE_CONFIGURATION",
            "cluster_count": 0,
        }
    key_fields = _KEY[:-1] if exact_fallback else _KEY
    baseline_map = {tuple(row[k] for k in key_fields): row for row in baseline_rows}
    keys = [tuple(row[k] for k in key_fields) for row in method_rows]
    if set(keys) != set(baseline_map) or len(baseline_map) != len(baseline_rows):
        return {
            "status": "INCOMPLETE_PAIRED_BLOCKS",
            "cluster_count": 0,
            "method_blocks": len(set(keys)),
            "baseline_blocks": len(baseline_map),
            "unmatched_method_blocks": len(set(keys) - set(baseline_map)),
            "unmatched_baseline_blocks": len(set(baseline_map) - set(keys)),
        }
    paired = [baseline_map[key] for key in keys]
    method_stats = [
        [row[f"{quantity}_error_ss"], row[f"{quantity}_truth_ss"]] for row in method_rows
    ]
    baseline_stats = [[row[f"{quantity}_error_ss"], row[f"{quantity}_truth_ss"]] for row in paired]
    if not np.allclose(
        np.asarray(method_stats)[:, 1], np.asarray(baseline_stats)[:, 1], rtol=1e-12, atol=0
    ):
        return {"status": "PAIRED_TRUTH_ENERGY_MISMATCH", "cluster_count": 0}
    result = paired_seed_bootstrap(
        method_stats,
        baseline_stats,
        [row["seed"] for row in method_rows],
        replicates=config["statistics"]["bootstrap_replicates"],
        seed=config["statistics"]["bootstrap_seed"],
        statistic="nrmse",
    )
    result.update(
        baseline_configuration_id=next(iter(configurations)),
        baseline_alignment="EXACT_BASELINE_REUSED_WITHIN_SEED"
        if exact_fallback
        else "SAME_N_AND_REPLICA",
        comparison_scope="POSTHOC_LEGACY_SELECTION_SEEDS_ONLY",
    )
    return result


def _aggregate(rows, keys=("configuration_id", "target", "population")):
    """Augment sufficient-statistic scores with exact pooled MAE/quantiles."""
    summaries = aggregate_rows(rows, keys=keys)
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    for summary in summaries:
        block = buckets[tuple(summary[key] for key in keys)]
        cases = sum(row["case_count"] for row in block)
        summary["case_count"] = cases
        summary["event_element_count"] = cases * 72 * 4
        summary["event_mae"] = sum(row["event_mae"] * row["case_count"] for row in block) / cases
        for quantity in ("pX", "v"):
            group_errors = np.concatenate(
                [np.asarray(row[f"{quantity}_abs_errors"]).reshape(-1, 6) for row in block]
            )
            q95 = np.quantile(group_errors, 0.95, axis=0)
            worst = int(np.argmax(q95))
            summary[f"{quantity}_worst_group_index_by_error_q95"] = worst
            summary[f"{quantity}_worst_group_error_q95"] = float(q95[worst])
            summary[f"{quantity}_worst_group_mae"] = float(group_errors[:, worst].mean())
    return summaries


def evaluate_rows(rows, config, nonalias_units):
    """Produce pooled scores, paired evidence, and honest nonzero candidates."""
    selection_seeds = set(config["data_roles"]["legacy_selection_seeds"])
    rows, duplicate_count = _deduplicate(row for row in rows if row["seed"] in selection_seeds)
    by_seed = _aggregate(rows, keys=("configuration_id", "target", "population", "seed"))
    pooled = _aggregate(rows)
    primary = config["primary_target"]["name"]
    primary_rows = [row for row in rows if row["target"] == primary and row["population"] == "all"]
    by_config = defaultdict(list)
    for row in primary_rows:
        by_config[row["configuration_id"]].append(row)
    candidates, bootstrap, resolution = [], [], []
    for aggregate in pooled:
        panel = {
            quantity: {
                "pooled_nrmse": aggregate[f"{quantity}_nrmse"],
                "error_q95": aggregate[f"{quantity}_error_q95"],
            }
            for quantity in ("pX", "v")
        }
        for result in resolution_grid(panel, delta_grid=config["statistics"]["delta_grid"]):
            quantity = result.pop("target")
            resolution.append(
                {
                    "configuration_id": aggregate["configuration_id"],
                    "target": aggregate["target"],
                    "population": aggregate["population"],
                    "quantity": quantity,
                    **result,
                    "interval_type": "NO_INDEPENDENT_FRESH_CALIBRATION_YET",
                }
            )
        if (
            aggregate["target"] != primary
            or aggregate["population"] != "all"
            or aggregate["method"] not in _MODELS
            or aggregate["n"] not in (64, 256)
            or aggregate["observation"] not in config["observation"]["methods"]
        ):
            continue
        rr = by_config[aggregate["configuration_id"]]
        expected_blocks = {
            (seed, arm, anchor, noise)
            for seed in selection_seeds
            for arm in config["fresh_cpu"]["main_arms"]
            for anchor in config["branches"]["anchors"]
            for noise in range(config["observation"]["noise_replicas"])
        }
        actual_blocks = {tuple(row[key] for key in _KEY) for row in rr}
        complete_coverage = actual_blocks == expected_blocks
        stable = {}
        for baseline in ("C0", "C1"):
            results = []
            for quantity in ("pX", "v"):
                result = _paired(rr, primary_rows, baseline, quantity, config)
                bootstrap.append(
                    {
                        "configuration_id": aggregate["configuration_id"],
                        "baseline": baseline,
                        "quantity": quantity,
                        "target": primary,
                        **result,
                    }
                )
                results.append(result)
            stable[baseline] = complete_coverage and all(
                result.get("cluster_count") == len(selection_seeds)
                and result.get("ci975") is not None
                and result["ci975"] < 0
                for result in results
            )
        distinct_anchors = {(row["seed"], row["arm"], row["anchor"]) for row in rr}
        candidates.append(
            {
                **aggregate,
                "model": aggregate["method"],
                "method": aggregate["observation"],
                "access_regime": "SAMPLE_AND_LOGP"
                if aggregate["observation"].startswith("O_LR")
                else "SAMPLE_ONLY",
                "actual_rank": aggregate["actual_rank_max"],
                "nonalias_eval_units": sum(nonalias_units.get(key, 0) for key in distinct_anchors),
                "stable_improvement_C0": stable["C0"],
                "stable_improvement_C1": stable["C1"],
                "baseline_stability_evidence": (
                    "5000 paired training-seed cluster NRMSE bootstrap; "
                    "pX and v CI upper bounds both < 0"
                ),
                "selection_seed_count": len({row["seed"] for row in rr}),
                "selection_coverage_complete": complete_coverage,
                "expected_selection_blocks": len(expected_blocks),
                "actual_selection_blocks": len(actual_blocks),
                "distinct_anchor_count": len(distinct_anchors),
                "rank_zero_anchor_count": len(
                    {
                        (row["seed"], row["arm"], row["anchor"])
                        for row in rr
                        if row["actual_rank"] == 0
                    }
                ),
            }
        )
    return {
        "candidates": candidates,
        "by_seed": by_seed,
        "pooled": pooled,
        "bootstrap": bootstrap,
        "resolution": resolution,
        "deduplicated_rows": duplicate_count,
        "unique_metric_rows": len(rows),
    }


def _nonalias(parent_root, manifest, seeds):
    result = {}
    for entry in manifest["trajectories"]:
        if entry["seed"] not in seeds:
            continue
        path = (parent_root / entry["observations_file"]).resolve()
        if parent_root.resolve() not in path.parents:
            raise ValueError("parent observation escapes parent root")
        with np.load(path, allow_pickle=False) as values:
            theta, anchors = values["branch_theta"], values["anchors"]
        for index, anchor in enumerate(anchors):
            active = np.any(theta[index, 10:14, 1] != theta[index, 10:14, 0], axis=-1)
            result[(entry["seed"], entry["arm"], int(anchor))] = int(active.sum())
    return result


def _cost_frontier(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("score_to_generation_ratio", "break_even_queries"):
            if key in row:
                try:
                    row[key] = float(row[key]) if row[key] else None
                except ValueError:
                    row[key] = None
        for key in ("cost_feasible", "within_domain", "includes_fit_score_label"):
            if key in row:
                row[key] = str(row[key]).lower() == "true"
    return rows


def frozen_direction_curves(config, parent_root, run_root):
    """Score fixed diagnostic rules directly from sealed N2C predictions.

    This descriptive view combines old observation-development/selection seeds;
    the separate four-selection-seed view must fail the five-seed direction gate.
    It never influences configuration ranking or the N3 scientific gate.
    """
    from src.modeling_qualification.math_contracts import helmert

    from .parent import load_parent_oracle

    parent_root, run_root = Path(parent_root), Path(run_root)
    rule_path = run_root / "N2/frozen_nonzero_rules.json"
    if not rule_path.is_file():
        return {"status": "NO_FROZEN_NONZERO_RULES", "rows": [], "source_hashes": {}}
    rules = _json(rule_path)
    if isinstance(rules, dict):
        rules = rules.get("rules", rules.get("selected", []))
    ids = {rule.get("configuration_id") or configuration_id(rule) for rule in rules}
    if not ids:
        return {"status": "NO_FROZEN_NONZERO_RULES", "rows": [], "source_hashes": {}}
    seeds_allowed = set(config["data_roles"]["observation_development_seeds"])
    selection_seeds = set(config["data_roles"]["legacy_selection_seeds"])
    seeds_allowed |= selection_seeds
    _, groups, _ = load_metadata(parent_root)
    h = helmert()
    gathered = {key: {"pred": [], "truth": [], "seeds": [], "units": []} for key in ids}
    sources = {str(rule_path.resolve()): digest(rule_path)}
    records = sorted((run_root / "N2/N2C").rglob("freeze.json"))
    records += sorted((run_root / "N2/N2C").rglob("freeze.json.gz"))
    observed = set()
    for freeze_path in records:
        freeze = _json(freeze_path)
        metadata_sources = {}
        if freeze.get("metadata_file"):
            metadata_path = (freeze_path.parent / freeze["metadata_file"]).resolve()
            if freeze_path.parent.resolve() not in metadata_path.parents:
                raise ValueError("frozen metadata path escapes prediction directory")
            metadata_hash = digest(metadata_path)
            if metadata_hash != freeze.get("metadata_sha256"):
                raise ValueError("changed frozen model metadata")
            full = _json(metadata_path)
            for key in ("prediction_sha256", "frozen_before_scoring"):
                if key in full and full[key] != freeze.get(key):
                    raise ValueError("inconsistent prediction and model metadata binding")
            freeze = {**full, **freeze}
            metadata_sources[str(metadata_path)] = metadata_hash
        relevant = [
            (index, model)
            for index, model in enumerate(freeze.get("models", []))
            if model["configuration_id"] in ids and model["seed"] in seeds_allowed
        ]
        if not relevant:
            continue
        prediction_path = freeze_path.parent / "predictions.npz"
        actual_hash = digest(prediction_path)
        if actual_hash != freeze.get("prediction_sha256") or not freeze.get(
            "frozen_before_scoring"
        ):
            raise ValueError(f"unsealed or changed diagnostic predictions: {prediction_path}")
        sources[str(freeze_path.resolve())] = digest(freeze_path)
        sources[str(prediction_path.resolve())] = actual_hash
        sources.update(metadata_sources)
        with np.load(prediction_path, allow_pickle=False) as values:
            predictions = values["prediction"]
        if predictions.ndim != 5 or predictions.shape[1:] != (4, 2, 72, 3):
            raise ValueError("frozen predictions require [models,4,2,72,3] Helmert shape")
        trajectory_id = freeze_path.parent.parent.name
        anchor_index = int(freeze_path.parent.name.removeprefix("a"))
        # Opening semantic truth follows byte verification of already sealed predictions.
        oracle = load_parent_oracle(parent_root, trajectory_id)
        p = oracle["branch_p"][anchor_index, 10:14]
        truth = p[:, 1:] - p[:, :1]
        for index, model in relevant:
            identity = tuple(model[name] for name in ("configuration_id", *_KEY))
            if identity in observed:
                raise ValueError("duplicate frozen diagnostic rule/seed/arm/anchor/replica")
            observed.add(identity)
            collected = gathered[model["configuration_id"]]
            collected["pred"].append(predictions[index] @ h.T)
            collected["truth"].append(truth)
            collected["seeds"].extend([model["seed"]] * 4)
            collected["units"].extend(
                (model["arm"], model["anchor"], bank) for bank in range(10, 14)
            )
    rows = []
    for ident, collected in gathered.items():
        if not collected["pred"]:
            continue
        pp, tt = np.concatenate(collected["pred"]), np.concatenate(collected["truth"])
        pred = np.concatenate((pp, (pp[:, 0] - pp[:, 1])[:, None]), axis=1)
        truth = np.concatenate((tt, (tt[:, 0] - tt[:, 1])[:, None]), axis=1)
        seeds = np.asarray(collected["seeds"])
        units = np.asarray(collected["units"], dtype=object)
        scopes = {
            "legacy_diagnostic_development_and_selection": np.ones(len(seeds), dtype=bool),
            "legacy_selection_only": np.isin(seeds, list(selection_seeds)),
        }
        for scope, mask in scopes.items():
            for index, target in enumerate(TARGETS):
                curves = direction_grid(
                    pred[mask, index],
                    truth[mask, index],
                    groups,
                    seeds[mask],
                    units[mask],
                    delta_grid=config["statistics"]["delta_grid"],
                    min_units=config["statistics"]["min_direction_anchor_bank_units"],
                    min_seeds=config["statistics"]["min_direction_training_seeds"],
                )
                for quantity in ("pX", "v"):
                    rows.extend(
                        {
                            "configuration_id": ident,
                            "target": target,
                            "quantity": quantity,
                            "scope": scope,
                            "used_for_N3_selection": False,
                            "noise_replicas_are_independent_training_seeds": False,
                            **row,
                        }
                        for row in curves[quantity]
                    )
    return {
        "status": "DESCRIPTIVE_LEGACY_ONLY" if rows else "NO_MATCHING_FROZEN_PREDICTIONS",
        "rows": rows,
        "source_hashes": sources,
    }


def build_selection(config, parent_root, run_root):
    """Read evidence and prepare a lock before the N3 RunWriter is initialized."""
    parent_root, run_root = Path(parent_root).resolve(), Path(run_root).resolve()
    audit_path = run_root / "N0/parent_integrity_audit.json"
    audit = _json(audit_path)
    manifest = _json(parent_root / "manifest.json")
    input_hashes, mismatches = {}, []
    for row in [*audit.get("files", []), *audit.get("archives", [])]:
        path = Path(row["path"])
        if not path.is_file():
            mismatches.append(f"MISSING_N0_INPUT:{path}")
            continue
        actual = digest(path)
        input_hashes[str(path.resolve())] = actual
        if actual != row.get("sha256"):
            mismatches.append(f"CHANGED_N0_INPUT:{path}")
    for name in ("manifest.json", "probe_metadata.json", "train_metadata.json", "dataset.npz"):
        path = parent_root / name
        if path.is_file():
            input_hashes[str(path)] = digest(path)
        else:
            mismatches.append(f"MISSING_PARENT_IDENTITY:{path}")
    if (
        audit.get("manifest_sha256")
        and digest(parent_root / "manifest.json") != audit["manifest_sha256"]
    ):
        mismatches.append("CHANGED_PARENT_MANIFEST")
    input_hashes[str(audit_path)] = digest(audit_path)
    metric_path = run_root / "N2/METRICS.jsonl.gz"
    with gzip.open(metric_path, "rt") as stream:
        seeds = set(config["data_roles"]["legacy_selection_seeds"])
        metrics = [row for line in stream if (row := json.loads(line))["seed"] in seeds]
    input_hashes[str(metric_path)] = digest(metric_path)
    for name in (
        "AGGREGATE.csv",
        "selected_observations.json",
        "frozen_nonzero_rules.json",
        "COST_FRONTIER.csv",
    ):
        path = run_root / "N2" / name
        if path.is_file():
            input_hashes[str(path)] = digest(path)
    source_paths = list((ROOT / "src/modeling_contrast").glob("*.py"))
    source_paths += [
        ROOT / "src/modeling_qualification" / name
        for name in ("toy.py", "math_contracts.py", "collection.py")
    ]
    for row in audit.get("source_inventory", []):
        relative = Path(row["path"])
        path = ROOT.parent / relative if relative.parts[0] == "ssvc_flow" else ROOT / relative
        if not path.is_file() or digest(path) != row.get("sha256"):
            mismatches.append(f"CHANGED_PARENT_SOURCE:{path}")
        elif path not in source_paths:
            source_paths.append(path)
    source_hashes = {str(path.resolve()): digest(path) for path in sorted(source_paths)}
    packet_hashes = {}
    for stage in ("N1", "N2"):
        metadata = sorted((run_root / stage).rglob("packets.json"))
        metadata += sorted((run_root / stage).rglob("packets.json.gz"))
        for path in metadata:
            packet = path.parent / "packet_arrays.npz"
            summary = _json(path)
            recorded = summary.get("packet_arrays_sha256")
            if summary.get("metadata_sha256"):
                full_metadata = path.parent / "packet_metadata.json.gz"
                if (
                    not full_metadata.is_file()
                    or digest(full_metadata) != summary["metadata_sha256"]
                ):
                    mismatches.append(f"CHANGED_FULL_PACKET_METADATA:{full_metadata}")
                else:
                    input_hashes[str(full_metadata.resolve())] = digest(full_metadata)
            if not packet.is_file():
                mismatches.append(f"MISSING_FROZEN_PACKET:{packet}")
                continue
            actual = digest(packet)
            packet_hashes[str(packet)] = actual
            input_hashes[str(path)] = digest(path)
            if actual != recorded:
                mismatches.append(f"CHANGED_FROZEN_PACKET:{packet}")
    for receipt_path in sorted((run_root / "N2/C1_auxiliary").rglob("receipt.json")):
        receipt = _json(receipt_path)
        auxiliary = receipt_path.parent / "C1_auxiliary.npz"
        if not auxiliary.is_file() or digest(auxiliary) != receipt.get("auxiliary_sha256"):
            mismatches.append(f"CHANGED_C1_AUXILIARY:{auxiliary}")
            continue
        packet_hashes[str(auxiliary.resolve())] = digest(auxiliary)
        packet_hashes[str(receipt_path.resolve())] = digest(receipt_path)
        if receipt.get("source_packet_sha256") not in packet_hashes.values():
            mismatches.append(f"UNBOUND_C1_SOURCE_PACKET:{receipt_path}")
    for freeze_path in sorted((run_root / "N2").rglob("freeze.json")):
        for auxiliary in _json(freeze_path).get("c1_auxiliary_hashes", {}).values():
            if not auxiliary.get("path"):
                continue
            path = Path(auxiliary["path"]).resolve()
            receipt_path = path.parent / "receipt.json"
            if packet_hashes.get(str(path)) != auxiliary.get("arrays") or packet_hashes.get(
                str(receipt_path)
            ) != auxiliary.get("receipt"):
                mismatches.append(f"C1_FIT_SEAL_HASH_MISMATCH:{freeze_path}")
    evidence = evaluate_rows(metrics, config, _nonalias(parent_root, manifest, seeds))
    directions = frozen_direction_curves(config, parent_root, run_root)
    input_hashes.update(directions["source_hashes"])
    evidence["direction"] = directions["rows"]
    smoke_path = run_root / "smoke/smoke_result.json"
    smoke = _json(smoke_path) if smoke_path.is_file() else {}
    if smoke_path.is_file():
        input_hashes[str(smoke_path)] = digest(smoke_path)
    forecast = smoke.get("forecast", {})
    current_bytes = sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())
    projected = {
        "total_walltime_hours": forecast.get("wall_seconds", float("inf")) / 3600,
        "peak_ram_gib": forecast.get("peak_ram_gib"),
        "added_output_gib": max(forecast.get("added_output_bytes", float("inf")), current_bytes)
        / 1024**3,
        "temporary_gib": forecast.get("temporary_bytes", float("inf")) / 1024**3,
        "current_result_bytes": current_bytes,
        "basis": "maximum of smoke whole-run total projection and measured existing result bytes",
    }
    # JSON evidence represents missing estimates explicitly, never NaN/Infinity.
    projected = {
        key: None if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in projected.items()
    }
    seed_audit = audit.get("seed_collision_audit", {})
    global_seed_audit_path = (
        ROOT / "docs/modeling_contrast/results/n0_parent/parent_integrity_audit.json"
    )
    if run_root.is_relative_to(ROOT / "runs") and global_seed_audit_path.is_file():
        seed_audit = _json(global_seed_audit_path).get("seed_collision_audit", {})
        input_hashes[str(global_seed_audit_path)] = digest(global_seed_audit_path)
    expected = config["parent_input"]["expected_legacy_trajectories"]
    data = {
        "status": "PASS"
        if (
            audit.get("status") == "PASS"
            and not mismatches
            and audit.get("parent_archive_verified") is True
        )
        else "FAIL",
        "legacy_complete": audit.get("complete_trajectories") == expected
        and len(manifest["trajectories"]) == expected,
        "fresh_seed_collision": True
        if seed_audit.get("colliding_seeds")
        else (False if seed_audit.get("status") == "PASS" else None),
        "evidence_mismatches": mismatches,
        "fresh_seed_inventory_roots": seed_audit.get("roots", []),
        "fresh_seed_manifests_scanned": seed_audit.get("manifests_scanned"),
        "fresh_seed_fixture_manifests_excluded": seed_audit.get(
            "excluded_test_fixture_manifest_count"
        ),
        "parent_integrity_audit_sha256": digest(audit_path),
    }
    lock = freeze_selection(
        evidence["candidates"],
        protocol=config,
        input_hashes=input_hashes,
        source_hashes=source_hashes,
        packet_hashes=packet_hashes,
        selector_hashes={
            str(path): digest(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("selection.py").resolve(),
            )
        },
        resource_forecast=projected,
        data_gate=data,
        cost_frontier=_cost_frontier(run_root / "N2/COST_FRONTIER.csv"),
    )
    fresh_budget = None
    if lock["selected"]:
        from .frozen_validation import project_frozen_validation_budget

        fresh_budget = project_frozen_validation_budget(config, lock, run_root)
        input_hashes.update(fresh_budget["evidence_hashes"])
        values = fresh_budget["forecast"]
        projected = {
            "total_walltime_hours": values["wall_seconds"] / 3600
            if values["wall_seconds"] is not None
            else None,
            "peak_ram_gib": values["peak_ram_gib"],
            "added_output_gib": values["added_output_bytes"] / 2**30,
            "temporary_gib": values["temporary_bytes"] / 2**30,
            "basis": "measured existing core outputs plus complete projected fresh CPU increment",
        }
        lock = freeze_selection(
            evidence["candidates"],
            protocol=config,
            input_hashes=input_hashes,
            source_hashes=source_hashes,
            packet_hashes=packet_hashes,
            selector_hashes={
                str(path): digest(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(__file__).with_name("selection.py").resolve(),
                    Path(__file__).with_name("frozen_validation.py").resolve(),
                )
            },
            resource_forecast=projected,
            data_gate=data,
            cost_frontier=_cost_frontier(run_root / "N2/COST_FRONTIER.csv"),
        )
    evidence["fresh_budget"] = fresh_budget
    evidence["lock"] = lock
    evidence["indicators"] = {
        "status": lock["status"],
        "allow_fresh_cpu": lock["allow_fresh_cpu"],
        "candidate_count": len(evidence["candidates"]),
        "science_pass_count": sum(row["science_pass"] for row in lock["candidate_audit"]),
        "cost_pass_count": sum(row["cost_pass"] for row in lock["candidate_audit"]),
        "selection_seed_count": len(seeds),
        "scope": "POSTHOC_LEGACY_DEVELOPMENT_ONLY",
        "calibration_resolution": calibration_resolution(
            len(config["data_roles"]["fresh_calibration_seeds"])
        ),
        "fresh_cpu_executed": False,
        "new_qwen_calls": 0,
        "new_gpu_calls": 0,
        "online_ssvc": False,
        "evidence_mismatches": mismatches,
        "deduplicated_rows": evidence["deduplicated_rows"],
        "unique_metric_rows": evidence["unique_metric_rows"],
        "independent_interval_status": "NOT_RUN_FRESH_CALIBRATION",
        "direction_status": directions["status"],
    }
    return evidence


def _csv(rows):
    stream = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def write_selection(writer, payload):
    """Persist an already prepared lock using its exact RunWriter binding."""
    if writer.binding != payload["lock"]["binding"]:
        raise ValueError("N3 writer binding differs from prepared selection lock")
    writer.write_json("MODEL_SELECTION_LOCK.json", payload["lock"])
    writer.write_text("CONTRAST_SELECTION_BY_SEED.csv", _csv(payload["by_seed"]))
    writer.write_text("CONTRAST_SELECTION_POOLED.csv", _csv(payload["pooled"]))
    writer.write_json("PAIRED_BOOTSTRAP.json", payload["bootstrap"])
    writer.write_text("RESOLUTION.csv", _csv(payload["resolution"]))
    writer.write_text("DIRECTION_CURVES.csv", _csv(payload.get("direction", [])))
    writer.write_json("DECISION_INDICATORS.json", payload["indicators"])
    writer.write_json("FRESH_RESOURCE_PROJECTION.json", payload.get("fresh_budget"))
    return payload["indicators"]


def run_selection(config, parent_root, run_root, writer):
    """Convenience adapter; callers normally build first to initialize binding."""
    return write_selection(writer, build_selection(config, parent_root, run_root))
