"""Frozen N4 measurement and validation of an already collected CPU corpus.

Importing this module never collects trajectories. All deployment choices come
from N3; calibration precedes locked-test scoring and cannot change any model.
"""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import calibration_resolution, paired_seed_bootstrap
from .io import canonical_hash, verify_run_manifest
from .protocol import ROOT, cpu_environment, resource_gate, validate_config
from .study import TARGETS, aggregate_rows, configuration_id, digest, load_metadata, metric_rows


def _json(path):
    path = Path(path)
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return json.load(stream)


def _freeze_metadata(path):
    value = _json(path)
    if value.get("metadata_file"):
        extra = (path.parent / value["metadata_file"]).resolve()
        if path.parent.resolve() not in extra.parents or digest(extra) != value["metadata_sha256"]:
            raise ValueError("frozen metadata identity mismatch")
        value = {**_json(extra), **value}
    return value


def frozen_configurations(config, lock):
    """Expand only prespecified n robustness and required fixed baselines."""
    if not 1 <= len(lock.get("selected", [])) <= 2:
        raise ValueError("one or two frozen selected configurations required")
    result = {}
    ns = [config["observation"]["primary_n"], *config["observation"]["secondary_n"]]
    for chosen in lock["selected"]:
        for n in ns:
            row = {
                "stage": "N4_FROZEN",
                "method": chosen["model"],
                "observation": chosen["method"],
                "n": n,
                "rank": chosen["rank"],
                "alpha": chosen["alpha"],
                "eta": chosen["eta"],
                "fit_banks": chosen["fit_banks"],
                "selection_primary_n": chosen["n"],
                "retuned_on_fresh_data": False,
            }
            result[configuration_id(row)] = row
            full = {**row, "method": "C3", "rank": "FULL"}
            result.setdefault(configuration_id(full), full)
    from .model_study import baseline_configs

    for n in ns:
        for row in baseline_configs(n):
            result[configuration_id(row)] = {
                **row,
                "stage": "N4_FROZEN_BASELINE",
                "retuned_on_fresh_data": False,
            }
    return list(result.values())


def _current_resources(run_root):
    measured = {}
    stage = run_root / "N2/stage_result.json"
    if stage.is_file():
        measured = _json(stage).get("resources", {}).get("measured", {})
    checks = run_root / "N2/resource_checkpoints.jsonl"
    if checks.is_file():
        lines = checks.read_text().splitlines()
        if lines:
            measured = {**measured, **json.loads(lines[-1]).get("measured", {})}
    if "wall_seconds" not in measured:
        measured["wall_seconds"] = sum(
            _json(path).get("wall_seconds", 0) for path in run_root.glob("*/stage_result.json")
        )
    return measured


def project_frozen_validation_budget(config, lock, run_root, *, fresh_root=None):
    """Project all 96 fresh anchors from actual packet and frozen-fit artifacts.

    Uses the maximum observed bytes/time for each method/n and the maximum
    per-model prediction footprint among matching frozen configurations. Source
    examples, prior measured cost, raw collection and reserves remain explicit.
    Missing examples fail the resource gate instead of assuming zero cost.
    """
    validate_config(config)
    run_root = Path(run_root)
    configurations = frozen_configurations(config, lock)
    needed = {(row["observation"], row["n"]) for row in configurations}
    profiles = defaultdict(list)
    evidence = {}
    for stage in ("N1", "N2"):
        for path in (run_root / stage).rglob("packets.json"):
            record = _json(path)
            key = (record.get("method"), record.get("n"))
            if key not in needed:
                continue
            files = [p for p in path.parent.iterdir() if p.is_file()]
            profiles[key].append(
                {
                    "bytes": sum(p.stat().st_size for p in files),
                    "wall_seconds": record["wall_seconds"],
                    "source": str(path.resolve()),
                }
            )
            evidence[str(path.resolve())] = digest(path)
    fit_rates = []
    signatures = {
        (row["method"], row["observation"], row["rank"], row["alpha"], row["eta"], row["fit_banks"])
        for row in configurations
    }
    for path in (run_root / "N2").rglob("freeze.json"):
        mini = _json(path)
        if mini.get("oracle_diagnostic"):
            continue
        full = _freeze_metadata(path)
        models = full.get("models", [])
        if not models or not any(
            tuple(
                row.get(key)
                for key in ("method", "observation", "rank", "alpha", "eta", "fit_banks")
            )
            in signatures
            for row in models
        ):
            continue
        prediction = path.parent / "predictions.npz"
        files = [prediction, path]
        if mini.get("metadata_file"):
            files.append(path.parent / mini["metadata_file"])
        fit_rates.append(
            {
                "bytes_per_prediction": sum(p.stat().st_size for p in files) / len(models),
                "wall_per_prediction": full["fit_seconds"] / len(models),
                "source": str(path.resolve()),
            }
        )
        evidence[str(path.resolve())] = digest(path)
    missing = [
        f"NO_PACKET_RESOURCE_EXAMPLE:{method}:n{n}"
        for method, n in needed
        if not profiles[(method, n)]
    ]
    if not fit_rates:
        missing.append("NO_MATCHING_FROZEN_FIT_RESOURCE_EXAMPLE")
    anchors = config["fresh_cpu"]["expected_trajectories"] * len(config["branches"]["anchors"])
    packet_bytes = anchors * sum(
        max(row["bytes"] for row in profiles[key]) for key in needed if profiles[key]
    )
    packet_wall = anchors * sum(
        max(row["wall_seconds"] for row in profiles[key]) for key in needed if profiles[key]
    )
    predictions = anchors * len(configurations) * config["observation"]["noise_replicas"]
    fit_bytes = predictions * max((row["bytes_per_prediction"] for row in fit_rates), default=0)
    # Small frozen panels share less preparation than large N2 sweeps.
    fit_wall = 2 * predictions * max((row["wall_per_prediction"] for row in fit_rates), default=0)
    raw_bytes = raw_wall = 0.0
    if fresh_root is None:
        audit_path = run_root / "N0/parent_integrity_audit.json"
        if audit_path.is_file():
            audit = _json(audit_path)
            summary_path = Path(audit["root"]) / "summary.json"
            if summary_path.is_file():
                parent = _json(summary_path)
                ratio = config["fresh_cpu"]["expected_trajectories"] / parent["trajectory_count"]
                # Parent counts had five replicas; the fresh collector has eight.
                replica_margin = max(1, config["observation"]["noise_replicas"] / 5)
                raw_bytes = parent["output_bytes"] * ratio * replica_margin
                raw_wall = parent["wall_seconds"] * ratio * replica_margin
                evidence[str(summary_path.resolve())] = digest(summary_path)
            else:
                missing.append("NO_RAW_COLLECTION_RESOURCE_EXAMPLE")
        else:
            missing.append("NO_RAW_COLLECTION_RESOURCE_EXAMPLE")
    current = _current_resources(run_root)
    if fresh_root is not None and (Path(fresh_root) / "summary.json").is_file():
        current["wall_seconds"] += _json(Path(fresh_root) / "summary.json").get("wall_seconds", 0)
    current_bytes = sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file())
    docs = ROOT / "docs/modeling_contrast/results"
    current_bytes += (
        sum(p.stat().st_size for p in docs.rglob("*") if p.is_file()) if docs.exists() else 0
    )
    # Budget includes model/direction tables, direct scoring receipts and raw metrics.
    reserve_bytes, reserve_wall = 64 * 2**20, 300
    increment = raw_bytes + packet_bytes + fit_bytes + reserve_bytes
    forecast = {
        "wall_seconds": current["wall_seconds"] + raw_wall + packet_wall + fit_wall + reserve_wall,
        "peak_ram_gib": current.get("peak_ram_gib", 0) + 0.5,
        "added_output_bytes": current_bytes + increment,
        "temporary_bytes": current.get("temporary_bytes", 0),
    }
    if missing:
        forecast["wall_seconds"] = None
    gate = resource_gate(forecast, config)
    return {
        "status": gate["status"],
        "forecast": forecast,
        "resource_gate": gate,
        "missing_resource_evidence": missing,
        "fresh_anchor_count": anchors,
        "existing_result_bytes": current_bytes,
        "fresh_increment_bytes": increment,
        "fresh_raw_bytes": raw_bytes,
        "fresh_packet_bytes": packet_bytes,
        "fresh_prediction_bytes": fit_bytes,
        "reserve_bytes": reserve_bytes,
        "packet_profiles": {f"{key[0]}:n{key[1]}": value for key, value in profiles.items()},
        "fit_profiles": fit_rates,
        "evidence_hashes": evidence,
        "basis": "actual method/n maxima; matching rank fit footprints; 96 anchors; no retuning",
        "live_resource_checks_required": True,
    }


def calibration_envelopes(rows, config):
    expected = set(config["data_roles"]["fresh_calibration_seeds"])
    maxima = defaultdict(dict)
    for row in rows:
        if row["seed"] not in expected:
            raise ValueError("only independent calibration seed rows may set an envelope")
        for quantity in ("pX", "v"):
            key = (row["configuration_id"], row["target"], row["population"], quantity)
            error = max(row[f"{quantity}_abs_errors"], default=0)
            maxima[key][row["seed"]] = max(error, maxima[key].get(row["seed"], 0))
    result = []
    for key, values in maxima.items():
        supported = set(values) == expected
        result.append(
            {
                **dict(
                    zip(("configuration_id", "target", "population", "quantity"), key, strict=True)
                ),
                "radius": max(values.values()) if supported else None,
                "status": "EMPIRICAL_SEED_MAX_ENVELOPE"
                if supported
                else "INSUFFICIENT_CALIBRATION_SEEDS",
                "per_seed_maxima": {str(seed): value for seed, value in sorted(values.items())},
            }
        )
    payload = {
        **calibration_resolution(len(expected)),
        "rows": result,
        "calibration_seed_ids": sorted(expected),
        "frozen_before_locked_test_scoring": True,
        "interval_type": "EMPIRICAL_SEED_MAX_ENVELOPE_NOT_DISTRIBUTION_FREE",
        "simultaneous_safety_certification": False,
    }
    payload["calibration_sha256"] = canonical_hash(payload)
    return payload


def score_empirical_coverage(rows, calibration):
    envelopes = {
        tuple(row[key] for key in ("configuration_id", "target", "population", "quantity")): row
        for row in calibration["rows"]
    }
    maxima = defaultdict(dict)
    for row in rows:
        if row["seed"] in calibration["calibration_seed_ids"]:
            raise ValueError("calibration seeds cannot be independent test units")
        for quantity in ("pX", "v"):
            key = (row["configuration_id"], row["target"], row["population"], quantity)
            maxima[key][row["seed"]] = max(
                max(row[f"{quantity}_abs_errors"], default=0), maxima[key].get(row["seed"], 0)
            )
    result = []
    for key, values in maxima.items():
        envelope = envelopes.get(key, {})
        radius = envelope.get("radius")
        result.append(
            {
                **dict(
                    zip(("configuration_id", "target", "population", "quantity"), key, strict=True)
                ),
                "radius": radius,
                "interval_width": None if radius is None else 2 * radius,
                "test_seed_count": len(values),
                "empirical_seed_coverage": None
                if radius is None
                else float(np.mean([v <= radius for v in values.values()])),
                "per_seed_max_error": values,
                "resolution_status": "RESOLUTION_PASS"
                if radius is not None and radius <= 0.00025 / 2
                else "RESOLUTION_FAIL",
                "distribution_free_95_claim": False,
                "simultaneous_safety_certification": False,
                "interval_type": calibration["interval_type"],
            }
        )
    return result


def validate_validation_inputs(
    config, selection_path, fresh_root, *, existing_run_roots, resource_forecast
):
    from .collect_fresh import validate_fresh_gate
    from .parent import audit_seed_collisions

    gate = validate_fresh_gate(
        config,
        selection_path,
        existing_run_roots=existing_run_roots,
        resource_forecast=resource_forecast,
    )
    raw = Path(fresh_root).resolve()
    collection = verify_run_manifest(raw.parent)
    if collection["status"] != "COMPLETE" or collection["binding"] != gate["binding"]:
        raise ValueError("fresh raw collection is incomplete or uses another frozen selection")
    manifest = _json(raw / "manifest.json")
    expected = {
        (seed, arm, role)
        for role in ("fresh_calibration", "fresh_locked_test")
        for seed in config["data_roles"][role + "_seeds"]
        for arm in config["fresh_cpu"]["main_arms"]
    }
    actual = [(row["seed"], row["arm"], row["split"]) for row in manifest["trajectories"]]
    if len(actual) != 32 or set(actual) != expected or manifest.get("profile") != "contrast_fresh":
        raise ValueError("fresh trajectory roles/seeds differ from the frozen 32-run protocol")
    roots = gate["lock"]["data_gate"].get("fresh_seed_inventory_roots", existing_run_roots)
    inventory = audit_seed_collisions(roots)
    unexpected = [
        row
        for row in inventory["collisions"]
        if Path(row["manifest"]).resolve() != raw / "manifest.json"
    ]
    if inventory["scan_errors"] or unexpected:
        raise ValueError(
            "STOP_FOR_REVIEW: unexpected fresh seed collision or incomplete workspace scan"
        )
    return {
        **gate,
        "manifest": manifest,
        "raw_manifest_sha256": digest(raw / "manifest.json"),
        "collection_status_before_validation": collection["status"],
        "collection_root": str(raw.parent),
        "collection_output_hashes": dict(collection["outputs"]),
    }


def prepare_frozen_validation(
    config, selection_path, fresh_root, run_root, *, existing_run_roots, resource_forecast=None
):
    """Validate a COMPLETE collection before opening its optional resumed writer."""
    lock = _json(selection_path)
    budget = project_frozen_validation_budget(config, lock, run_root, fresh_root=fresh_root)
    if not budget["resource_gate"]["passed"]:
        raise ValueError("RESOURCE_REVIEW_REQUIRED: frozen validation projection")
    forecast = dict(budget["forecast"])
    if resource_forecast is not None:
        forecast = {
            key: max(value, resource_forecast.get(key, value)) for key, value in forecast.items()
        }
    receipt = validate_validation_inputs(
        config,
        selection_path,
        fresh_root,
        existing_run_roots=existing_run_roots,
        resource_forecast=forecast,
    )
    receipt["validation_budget"] = budget
    receipt["preparation_sha256"] = canonical_hash(receipt)
    return receipt


def _check_prepared_collection(receipt, selection_path, fresh_root):
    if receipt.get("preparation_sha256") != canonical_hash(
        {k: v for k, v in receipt.items() if k != "preparation_sha256"}
    ):
        raise ValueError("prepared validation receipt changed")
    if receipt.get("collection_status_before_validation") != "COMPLETE":
        raise ValueError("validation was not prepared from a complete collection")
    if Path(receipt["collection_root"]).resolve() != Path(fresh_root).resolve().parent:
        raise ValueError("prepared collection root changed")
    if digest(selection_path) != receipt["selection_sha256"]:
        raise ValueError("N3 selection changed after preparation")
    for name, expected in receipt["collection_output_hashes"].items():
        path = (Path(receipt["collection_root"]) / name).resolve()
        if (
            Path(receipt["collection_root"]).resolve() not in path.parents
            or digest(path) != expected
        ):
            raise ValueError("prepared raw collection hash changed")


def _unit(config, fresh_root, entry, ai, configurations, out):
    from src.modeling_qualification.math_contracts import helmert

    from .model_study import fit_unit
    from .observation_study import measure_unit, service_arguments
    from .observations import ToyWorldService, known_event_logp
    from .parent import load_parent_oracle

    role = entry["split"]
    out.mkdir(parents=True, exist_ok=False)
    pairs = sorted({(row["observation"], row["n"]) for row in configurations})
    packets = {
        pair: measure_unit(
            fresh_root, entry, ai, *pair, out / "packets" / f"{pair[0]}_n{pair[1]}", data_role=role
        )
        for pair in pairs
    }
    rows = fit_unit(fresh_root, entry, ai, configurations, packets, out / "models", data_role=role)
    # Direct measurements were sealed in packets before the model's first oracle read.
    _, groups, _ = load_metadata(fresh_root)
    truth = load_parent_oracle(fresh_root, entry["id"])["branch_p"][ai, 10:14]
    contrast = truth[:, 1:] - truth[:, :1]
    for (observation, n), packet in packets.items():
        c = dict(
            method="D_DIRECT", observation=observation, n=n, rank=0, alpha=0.0, eta=0.0, fit_banks=0
        )
        for noise in range(8):
            identity = dict(
                c,
                configuration_id=configuration_id(c),
                seed=entry["seed"],
                arm=entry["arm"],
                anchor=[8, 24, 40][ai],
                noise_replica=noise,
                actual_rank=0,
                k=0,
                original_role=role,
                v2_role=role,
                stage="N4_FROZEN_DIRECT",
                prediction_hash=packet["artifact_sha256"],
            )
            rows.extend(
                metric_rows(
                    packet["helmert"][noise, 10:14] @ helmert().T, contrast, groups, identity
                )
            )
    # Cold known-event scoring receives its own paid cache; only pX/v are claimed.
    theta, features, categories, prompt_ids, _ = service_arguments(fresh_root, entry, ai)
    service = ToyWorldService.from_parameters(theta, features, categories, prompt_ids=prompt_ids)
    policies = [f"b{bank}o{op}" for bank in range(10, 14) for op in range(3)]
    known = known_event_logp(service, policies)
    np.savez_compressed(out / "known_event_scores.npz", pX=known["pX"], v=known["v"])
    known_rows = []
    for index, target in enumerate(TARGETS):
        endpoints = ((1, 0), (2, 0), (1, 2))[index]
        row = dict(
            configuration_id="D_KNOWN_EVENT_LOGP",
            method="D_KNOWN_EVENT_LOGP",
            observation="EXACT_KNOWN_EVENTS",
            n=0,
            target=target,
            population="all",
            seed=entry["seed"],
            arm=entry["arm"],
            anchor=[8, 24, 40][ai],
            noise_replica=0,
            case_count=4,
            original_role=role,
            v2_role=role,
            event_metrics_status="NOT_MEASURED_BY_THIS_BASELINE",
        )
        for quantity in ("pX", "v"):
            values = known[quantity].reshape(4, 3, 72)
            exact = truth[..., 0] if quantity == "pX" else 1 - truth[..., 3]
            predicted = values[:, endpoints[0]] - values[:, endpoints[1]]
            actual = exact[:, endpoints[0]] - exact[:, endpoints[1]]
            error = np.stack(
                [(predicted - actual)[:, groups == g].mean(1) for g in sorted(set(groups))], 1
            )
            signal = np.stack([actual[:, groups == g].mean(1) for g in sorted(set(groups))], 1)
            row.update(
                {
                    f"{quantity}_error_ss": float(np.sum(error**2)),
                    f"{quantity}_truth_ss": float(np.sum(signal**2)),
                    f"{quantity}_abs_errors": np.abs(error).ravel().tolist(),
                }
            )
        known_rows.append(row)
    costs = {f"{key[0]}:n{key[1]}": packet["metadata"] for key, packet in packets.items()}
    with gzip.open(out / "cost_ledger.json.gz", "wt") as stream:
        json.dump({"packets": costs, "known_event": known["cost"], "v2_role": role}, stream)
    return rows, known_rows


def _bootstrap_test(rows, config):
    result = []
    primary = [r for r in rows if r["target"] == TARGETS[0] and r["population"] == "all"]
    configurations = {
        row["configuration_id"]
        for row in primary
        if row["method"] in {"C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"}
    }
    for ident in sorted(configurations):
        method = [r for r in primary if r["configuration_id"] == ident]
        for baseline_name in ("C0", "C1_LEGACY_EXACT_RULE"):
            baseline = [
                r for r in primary if r["method"] == baseline_name and r["n"] == method[0]["n"]
            ]
            keys = ("seed", "arm", "anchor", "noise_replica")
            mapping = {tuple(r[k] for k in keys): r for r in baseline}
            if {tuple(r[k] for k in keys) for r in method} != set(mapping):
                result.append(
                    {
                        "configuration_id": ident,
                        "baseline": baseline_name,
                        "status": "INCOMPLETE_PAIRED_BLOCKS",
                    }
                )
                continue
            for quantity in ("pX", "v"):
                paired = [mapping[tuple(r[k] for k in keys)] for r in method]
                columns = (quantity + "_error_ss", quantity + "_truth_ss")
                comparison = paired_seed_bootstrap(
                    [[r[k] for k in columns] for r in method],
                    [[r[k] for k in columns] for r in paired],
                    [r["seed"] for r in method],
                    replicates=5000,
                    seed=config["statistics"]["bootstrap_seed"],
                    statistic="nrmse",
                )
                if comparison["status"] == "POSTHOC_DEVELOPMENT_SEED_CLUSTER_BOOTSTRAP":
                    comparison["status"] = "FROZEN_FRESH_TEST_DESCRIPTIVE_COMPARISON"
                result.append(
                    {
                        "configuration_id": ident,
                        "baseline": baseline_name,
                        "quantity": quantity,
                        **comparison,
                    }
                )
    return result


def run_frozen_validation(
    config,
    selection_path,
    fresh_root,
    run_root,
    writer,
    *,
    existing_run_roots,
    resource_forecast=None,
    validated_collection_receipt=None,
):
    """Measure, predict, calibrate and score fixed fresh CPU runs; never train."""
    from .resources import StageResources
    from .selection_driver import _csv

    cpu_environment()
    lock = _json(selection_path)
    budget = project_frozen_validation_budget(config, lock, run_root, fresh_root=fresh_root)
    if not budget["resource_gate"]["passed"]:
        raise ValueError("RESOURCE_REVIEW_REQUIRED: frozen measurement/prediction projection")
    forecast = dict(budget["forecast"])
    if resource_forecast is not None:
        for key in forecast:
            forecast[key] = max(forecast[key], resource_forecast.get(key, forecast[key]))
    if validated_collection_receipt is None:
        gate = validate_validation_inputs(
            config,
            selection_path,
            fresh_root,
            existing_run_roots=existing_run_roots,
            resource_forecast=forecast,
        )
    else:
        _check_prepared_collection(validated_collection_receipt, selection_path, fresh_root)
        gate = validated_collection_receipt
    if writer.binding != gate["binding"]:
        raise ValueError("validation writer differs from frozen N3 binding")
    configurations = frozen_configurations(config, gate["lock"])
    writer.write_json(
        "FROZEN_VALIDATION_PROTOCOL.json",
        {
            "configurations": configurations,
            "selection_sha256": gate["selection_sha256"],
            "raw_manifest_sha256": gate["raw_manifest_sha256"],
            "retuning_allowed": False,
            "budget": budget,
            "training_updates_performed_by_validation": 0,
        },
    )
    resources = StageResources(config, run_root, writer.out)
    if (Path(fresh_root) / "summary.json").is_file():
        resources.prior_wall += _json(Path(fresh_root) / "summary.json").get("wall_seconds", 0)
    started = time.perf_counter()
    calibration = None
    fold_rows = {}
    for role in ("fresh_calibration", "fresh_locked_test"):
        if role == "fresh_locked_test" and calibration is None:
            raise AssertionError("independent calibration must be frozen before test scoring")
        rows, known_rows = [], []
        entries = [r for r in gate["manifest"]["trajectories"] if r["split"] == role]
        for entry in entries:
            for ai in range(3):
                resources.check(f"before:{role}:{entry['id']}:a{ai}")
                out = writer.out / role / entry["id"] / f"a{ai}"
                model, known = _unit(config, Path(fresh_root), entry, ai, configurations, out)
                rows.extend(model)
                known_rows.extend(known)
                for path in out.rglob("*"):
                    if path.is_file():
                        writer.register_existing(path.relative_to(writer.out))
                resources.check(f"after:{role}:{entry['id']}:a{ai}")
        fold_rows[role] = rows + known_rows
        for name, values in (("metrics", rows), ("known_event_metrics", known_rows)):
            content = "".join(
                json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in values
            )
            writer.write_bytes(f"{role}_{name}.jsonl.gz", gzip.compress(content.encode(), mtime=0))
        writer.write_text(
            f"{role}_BY_SEED.csv",
            _csv(aggregate_rows(rows, keys=("configuration_id", "target", "population", "seed"))),
        )
        if role == "fresh_calibration":
            calibration = calibration_envelopes(rows + known_rows, config)
            writer.write_json("CALIBRATION_LOCK.json", calibration)
        else:
            writer.write_json(
                "EMPIRICAL_COVERAGE.json", score_empirical_coverage(rows + known_rows, calibration)
            )
            writer.write_json("PAIRED_FROZEN_TEST_BOOTSTRAP.json", _bootstrap_test(rows, config))
    if digest(selection_path) != gate["selection_sha256"]:
        raise ValueError("selection changed during frozen validation")
    from .collect_fresh import _verify_files

    _verify_files(gate["lock"]["source_hashes"], ROOT, "source after validation")
    test_models = [
        row for row in fold_rows["fresh_locked_test"] if row["method"] != "D_KNOWN_EVENT_LOGP"
    ]
    aggregate = aggregate_rows(test_models)
    selected_ids = {
        configuration_id(
            {
                "method": row["model"],
                "observation": row["method"],
                "n": row["n"],
                "rank": row["rank"],
                "alpha": row["alpha"],
                "eta": row["eta"],
                "fit_banks": row["fit_banks"],
            }
        )
        for row in gate["lock"]["selected"]
    }
    selected_results = [
        row
        for row in aggregate
        if row["configuration_id"] in selected_ids
        and row["target"] == TARGETS[0]
        and row["population"] == "all"
    ]
    for row in selected_results:
        row["prediction_status"] = (
            "PREDICTION_PASS"
            if all(
                row.get(quantity + "_nrmse") is not None and row[quantity + "_nrmse"] <= 0.75
                for quantity in ("pX", "v")
            )
            else "PREDICTION_FAIL"
        )
        row["resolution_status"] = (
            "RESOLUTION_PASS"
            if all(row[quantity + "_error_q95"] <= 0.000125 for quantity in ("pX", "v"))
            else "RESOLUTION_FAIL"
        )
    summary = {
        "status": "FROZEN_FRESH_CPU_VALIDATION_COMPLETE",
        "calibration_seed_count": 6,
        "test_seed_count": 10,
        "trajectory_count": 32,
        "anchor_count": 96,
        "configurations": configurations,
        "wall_seconds": time.perf_counter() - started,
        "calibration_sha256": calibration["calibration_sha256"],
        "distribution_free_95_claim": False,
        "simultaneous_safety_certification": False,
        "new_optimizer_updates": 0,
        "new_qwen_calls": 0,
        "new_gpu_calls": 0,
        "online_ssvc": False,
        "generalization_scope": (
            "new training RNG only; same dataset, initialization and finite-action CPU model"
        ),
        "resources": resources.check("complete"),
    }
    writer.register_existing("resource_checkpoints.jsonl")
    independent = {
        "status": "INDEPENDENT_VALIDATION_COMPLETE",
        "scope": summary["generalization_scope"],
        "selected_results": selected_results,
        "science_status": "PREDICTION_PASS"
        if selected_results
        and all(row["prediction_status"] == "PREDICTION_PASS" for row in selected_results)
        else "PREDICTION_FAIL",
        "input_hashes": {
            str((Path(fresh_root) / "manifest.json").resolve()): gate["raw_manifest_sha256"],
            str(Path(selection_path).resolve()): gate["selection_sha256"],
        },
        "selection_sha256": gate["selection_sha256"],
        "prediction_hashes": {
            str((writer.out / name).resolve()): value
            for name, value in writer.output_hashes.items()
            if name.endswith("predictions.npz")
        },
        "scoring_hashes": {
            str((writer.out / name).resolve()): value
            for name, value in writer.output_hashes.items()
            if "metrics" in name or "COVERAGE" in name
        },
        "distribution_free_95_claim": False,
        "simultaneous_safety_certification": False,
        "calibration_seed_count": 6,
        "test_seed_count": 10,
    }
    writer.write_json("INDEPENDENT_VALIDATION.json", independent)
    writer.write_json("stage_result.json", summary)
    return summary
