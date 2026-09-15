#!/usr/bin/env python3
"""Summarize sealed N4 metrics and recorded costs without running any model.

Usage: python scripts/summarize_modeling_n4_results.py --run-root RUN_ROOT \
    --out-dir RUN_ROOT/posthoc/n4_facts

Only stdlib and NumPy are imported. Every consumed artifact is checked against
N4_validation/RUN_MANIFEST.json. This is a factual metrics summary, not a raw
prediction audit, statistical decision, or deployment recommendation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

TARGET = "joint_1_minus_joint_0"
ROLES = ("fresh_calibration", "fresh_locked_test")
QUANTITIES = ("pX", "v")
OBSERVATIONS = ("O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX")
IDENTITY = ("configuration_id", "seed", "arm", "anchor", "noise_replica")
CONFIG_FIELDS = ("method", "observation", "n", "rank", "alpha", "eta", "fit_banks")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"nonnumeric {name}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"nonfinite or negative {name}")
    return value


def configuration_id(row):
    return "|".join(str(row[key]) for key in CONFIG_FIELDS)


def safe_member(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative sealed artifact: {name}")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"artifact escapes sealed stage: {name}")
    return path


class Sources:
    def __init__(self, stage):
        self.stage = stage
        self.files = {}
        path = stage / "RUN_MANIFEST.json"
        self.manifest = json.loads(path.read_text())
        self.files[str(path)] = sha256(path)
        if self.manifest.get("status") != "COMPLETE" or (stage / ".writer.lock").exists():
            raise ValueError("N4_validation is not a complete, inactive sealed stage")

    def check(self, path, expected=None):
        path = path.resolve()
        if self.stage not in path.parents:
            raise ValueError("input outside sealed N4_validation")
        name = path.relative_to(self.stage).as_posix()
        bound = self.manifest.get("outputs", {}).get(name)
        actual = sha256(path)
        if bound is None or actual != bound or (expected is not None and actual != expected):
            raise ValueError(f"sealed artifact hash mismatch: {path}")
        if str(path) in self.files and self.files[str(path)] != actual:
            raise ValueError(f"input changed while summarizing: {path}")
        self.files[str(path)] = actual
        return path

    def read(self, path, expected=None):
        path = self.check(path, expected)
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
            return json.load(stream)

    def rows(self, path):
        path = self.check(path)
        with gzip.open(path, "rt") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"invalid metric row at {path}:{line_number}")
                    yield value

    def unchanged(self):
        for name, expected in self.files.items():
            if sha256(name) != expected:
                raise ValueError(f"input changed while summarizing: {name}")


def add_cost(total, cost):
    """Add each recorded cost once; peaks are maxima, not additive costs."""
    if not isinstance(cost, dict):
        raise ValueError("cost ledger must be an object")
    for name, value in cost.items():
        number(value, name)
        total[name] = (
            max(total.get(name, 0), value)
            if name.startswith("peak_")
            else total.get(name, 0) + value
        )


def load_units(sources, role):
    stage = sources.stage
    costs = {
        "unit_count": 0,
        "packet_block_count": 0,
        "frozen_model_records": 0,
        "packet_wall_seconds": 0.0,
        "model_fit_wall_seconds": 0.0,
        "packet_cost": {},
        "C1_supplementary_cost": {},
        "known_event_cost": {},
        "total_recorded_cost": {},
        "by_observation_n": {},
    }
    model_rows = {}
    cost_units = set()
    for path in sorted((stage / role).glob("*/a*/cost_ledger.json.gz")):
        unit = path.parent
        value = sources.read(path)
        if value.get("v2_role") != role:
            raise ValueError("cost ledger role mismatch")
        cost_units.add(unit)
        costs["unit_count"] += 1
        add_cost(costs["known_event_cost"], value["known_event"])
        for key, packet in value["packets"].items():
            destination = costs["by_observation_n"].setdefault(
                key,
                {
                    "method": packet["method"],
                    "n": packet["n"],
                    "packet_block_count": 0,
                    "wall_seconds": 0.0,
                    "packet_cost": {},
                    "C1_supplementary_cost": {},
                },
            )
            if (destination["method"], destination["n"]) != (packet["method"], packet["n"]):
                raise ValueError("packet ledger key has inconsistent method/n")
            wall = number(packet["wall_seconds"], "packet wall_seconds")
            destination["packet_block_count"] += 1
            destination["wall_seconds"] += wall
            costs["packet_block_count"] += 1
            costs["packet_wall_seconds"] += wall
            for item in packet["packets"]:
                add_cost(destination["packet_cost"], item["cost"])
                add_cost(costs["packet_cost"], item["cost"])
            for item in packet.get("C1_supplementary_costs", []):
                add_cost(destination["C1_supplementary_cost"], item["cost"])
                add_cost(costs["C1_supplementary_cost"], item["cost"])
    freeze_units = set()
    for path in sorted((stage / role).glob("*/a*/models/freeze.json")):
        freeze_units.add(path.parent.parent)
        value = sources.read(path)
        if value.get("metadata_file"):
            extra = safe_member(path.parent, value["metadata_file"])
            value = {**sources.read(extra, value["metadata_sha256"]), **value}
        if value.get("frozen_before_scoring") is not True:
            raise ValueError("model freeze does not attest prediction before scoring")
        costs["model_fit_wall_seconds"] += number(value["fit_seconds"], "fit_seconds")
        costs["frozen_model_records"] += len(value["models"])
        for row in value["models"]:
            key = tuple(row[field] for field in IDENTITY)
            if key in model_rows or row.get("v2_role") != role:
                raise ValueError("duplicate frozen model identity or role mismatch")
            model_rows[key] = row
    if not cost_units or cost_units != freeze_units:
        raise ValueError(f"cost/freeze unit coverage mismatch for {role}")
    for field in ("packet_cost", "C1_supplementary_cost", "known_event_cost"):
        add_cost(costs["total_recorded_cost"], costs[field])
    return costs, model_rows


def metric_accumulator(first):
    return {
        "metadata": {key: first.get(key) for key in CONFIG_FIELDS},
        "seeds": set(),
        "units": set(),
        "noise_replicas": set(),
        "rows": 0,
        "actual_ranks": [],
        "ks": [],
        "group_count": None,
        **{name: {"error_ss": 0.0, "truth_ss": 0.0, "errors": []} for name in QUANTITIES},
    }


def accumulate(acc, row):
    if {key: row.get(key) for key in CONFIG_FIELDS} != acc["metadata"]:
        raise ValueError("configuration metadata changes within one identity")
    acc["seeds"].add(row["seed"])
    acc["units"].add((row["seed"], row["arm"], row["anchor"]))
    acc["noise_replicas"].add(row["noise_replica"])
    acc["rows"] += 1
    for field, destination in (("actual_rank", "actual_ranks"), ("k", "ks")):
        if row.get(field) is not None:
            acc[destination].append(number(row[field], field))
    cases = row.get("case_count")
    if not isinstance(cases, int) or cases <= 0:
        raise ValueError("positive case_count required for group-major recovery")
    for quantity in QUANTITIES:
        errors = np.asarray(row[quantity + "_abs_errors"], dtype=float)
        if (
            errors.ndim != 1
            or len(errors) != cases * 6
            or not np.isfinite(errors).all()
            or (errors < 0).any()
        ):
            raise ValueError("expected finite bank-major absolute errors for six groups")
        error_ss = number(row[quantity + "_error_ss"], quantity + " error_ss")
        energy = number(row[quantity + "_truth_ss"], quantity + " truth_ss")
        if not math.isclose(float(errors @ errors), error_ss, rel_tol=1e-10, abs_tol=1e-18):
            raise ValueError("absolute error vector and stored error_ss disagree")
        acc[quantity]["error_ss"] += error_ss
        acc[quantity]["truth_ss"] += energy
        acc[quantity]["errors"].append(errors.reshape(cases, 6))


def finalize(role, ident, acc):
    base = {
        "role": role,
        "configuration_id": ident,
        **acc["metadata"],
        "target": TARGET,
        "population": "all",
        "seed_count": len(acc["seeds"]),
        "seed_ids": sorted(acc["seeds"]),
        "unit_count": len(acc["units"]),
        "noise_replica_count": len(acc["noise_replicas"]),
        "metric_row_count": acc["rows"],
    }
    for source, prefix in (("actual_ranks", "actual_rank"), ("ks", "k")):
        values = acc[source]
        base.update(
            {
                prefix + "_min": min(values) if values else None,
                prefix + "_max": max(values) if values else None,
            }
        )
    pooled = {**base, "metric_scope": "pooled", "group_index": None}
    groups = [{**base, "metric_scope": "group", "group_index": index} for index in range(6)]
    for quantity in QUANTITIES:
        stats = acc[quantity]
        errors = np.concatenate(stats["errors"])
        pooled.update(
            {
                quantity + "_error_ss": stats["error_ss"],
                quantity + "_truth_ss": stats["truth_ss"],
                quantity + "_nrmse": math.sqrt(stats["error_ss"] / stats["truth_ss"])
                if stats["truth_ss"] > 0
                else None,
                quantity + "_nrmse_status": "DEFINED"
                if stats["truth_ss"] > 0
                else "UNKNOWN_ZERO_SIGNAL",
                quantity + "_mae": float(errors.mean()),
                quantity + "_error_q95": float(np.quantile(errors, 0.95)),
                quantity + "_max_abs_error": float(errors.max()),
                quantity + "_scalar_error_count": int(errors.size),
            }
        )
        group_mae = errors.mean(axis=0)
        group_q95 = np.quantile(errors, 0.95, axis=0)
        pooled.update(
            {
                quantity + "_worst_group_by_mae": int(np.argmax(group_mae)),
                quantity + "_worst_group_mae": float(group_mae.max()),
                quantity + "_worst_group_by_q95": int(np.argmax(group_q95)),
                quantity + "_worst_group_q95": float(group_q95.max()),
            }
        )
        for index, row in enumerate(groups):
            row.update(
                {
                    quantity + "_error_ss": float(np.sum(errors[:, index] ** 2)),
                    quantity + "_truth_ss": None,
                    quantity + "_nrmse": None,
                    quantity + "_nrmse_status": "UNKNOWN_GROUP_TRUTH_ENERGY_NOT_STORED",
                    quantity + "_mae": float(group_mae[index]),
                    quantity + "_error_q95": float(group_q95[index]),
                    quantity + "_max_abs_error": float(errors[:, index].max()),
                    quantity + "_scalar_error_count": len(errors),
                }
            )
    return pooled, groups


def apply_raw_group_audit(path, run, sources, pooled, csv_rows):
    """Join an optional independent raw audit only when the sealed stage matches."""
    if path is None:
        return {"status": "NOT_PROVIDED", "group_rows_applied": 0}
    path = Path(path).resolve()
    audit_hash = sha256(path)
    audit = json.loads(path.read_text())
    if (
        audit.get("status") != "PASS"
        or Path(audit.get("run_root", "")).resolve() != run
        or audit.get("stage_manifest_hashes", {}).get("N4_validation")
        != sources.files[str(sources.stage / "RUN_MANIFEST.json")]
    ):
        raise ValueError("raw group audit does not bind the same complete N4 stage")
    sources.files[str(path)] = audit_hash
    keyed = {}
    for row in audit.get("group_pooled", []):
        if row.get("target") != TARGET or row.get("population") != "all":
            raise ValueError("raw group audit has a different target/population")
        keyed.setdefault((row["role"], row["configuration_id"]), []).append(row)
    applied = 0
    for summary in pooled:
        key = (summary["role"], summary["configuration_id"])
        if key not in keyed:
            continue
        rows = sorted(keyed.pop(key), key=lambda row: row["group"])
        if len(rows) != 6 or len({row["group"] for row in rows}) != 6:
            raise ValueError("raw audit must retain all six groups for each configuration")
        destination = sorted(
            [
                row
                for row in csv_rows
                if (row["role"], row["configuration_id"]) == key and row["metric_scope"] == "group"
            ],
            key=lambda row: row["group_index"],
        )
        for original, target in zip(rows, destination, strict=True):
            if original.get("n") != summary["n"]:
                raise ValueError("raw group audit n differs from frozen metric configuration")
            target["group_name"] = original["group"]
            for quantity in QUANTITIES:
                for suffix in ("error_ss", "mae", "error_q95"):
                    field = quantity + "_" + suffix
                    if not math.isclose(
                        original[field], target[field], rel_tol=1e-10, abs_tol=1e-15
                    ):
                        raise ValueError("raw audited group errors differ from sealed metrics")
                energy = number(original[quantity + "_truth_ss"], "group truth_ss")
                nrmse = math.sqrt(target[quantity + "_error_ss"] / energy) if energy > 0 else None
                supplied = original[quantity + "_nrmse"]
                if (supplied is None) != (nrmse is None) or (
                    nrmse is not None
                    and not math.isclose(supplied, nrmse, rel_tol=1e-10, abs_tol=1e-15)
                ):
                    raise ValueError("raw audited group NRMSE disagrees with sufficient statistics")
                target.update(
                    {
                        quantity + "_truth_ss": energy,
                        quantity + "_nrmse": nrmse,
                        quantity + "_nrmse_status": "DEFINED_FROM_RAW_AUDIT"
                        if energy > 0
                        else "UNKNOWN_ZERO_SIGNAL",
                    }
                )
            applied += 1
        for quantity in QUANTITIES:
            if not math.isclose(
                sum(row[quantity + "_truth_ss"] for row in destination),
                summary[quantity + "_truth_ss"],
                rel_tol=1e-10,
                abs_tol=1e-15,
            ):
                raise ValueError("raw audited group energies do not sum to pooled energy")
            defined = [row for row in destination if row[quantity + "_nrmse"] is not None]
            worst = max(defined, key=lambda row: row[quantity + "_nrmse"]) if defined else None
            summary[quantity + "_group_nrmse_unknown_count"] = 6 - len(defined)
            summary[quantity + "_worst_defined_group_nrmse"] = (
                worst[quantity + "_nrmse"] if worst else None
            )
            summary[quantity + "_worst_defined_group_by_nrmse"] = (
                worst["group_name"] if worst else None
            )
    if keyed:
        raise ValueError("raw audit contains configurations absent from summarized metrics")
    return {
        "status": "APPLIED_TO_AVAILABLE_GROUP_ROWS",
        "group_rows_applied": applied,
        "source": str(path),
        "sha256": audit_hash,
        "scope": "only configurations explicitly present in the independent raw group audit",
    }


def summarize(run_root, out_dir, *, raw_audit_json=None):
    run, out = Path(run_root).resolve(), Path(out_dir).resolve()
    for name in ("N3_server", "N4", "N4_validation"):
        sealed = run / name
        if out == sealed or sealed in out.parents:
            raise ValueError("output cannot be inside a sealed stage")
    if out.exists():
        raise FileExistsError(f"posthoc output already exists: {out}")
    sources = Sources(run / "N4_validation")
    protocol = sources.read(sources.stage / "FROZEN_VALIDATION_PROTOCOL.json")
    stage_result = sources.read(sources.stage / "stage_result.json")
    configurations = protocol["configurations"]
    pairs = {(row["observation"], row["n"]) for row in configurations}
    expected = {configuration_id(row) for row in configurations} | {"D_KNOWN_EVENT_LOGP"}
    expected |= {
        configuration_id(
            dict(
                method="D_DIRECT",
                observation=observation,
                n=n,
                rank=0,
                alpha=0.0,
                eta=0.0,
                fit_banks=0,
            )
        )
        for observation, n in pairs
    }
    pooled, csv_rows, costs_by_role, missing = [], [], {}, {}
    for role in ROLES:
        costs_by_role[role], frozen = load_units(sources, role)
        buckets, identities, observed_model_keys = {}, set(), set()
        for suffix in ("metrics", "known_event_metrics"):
            path = sources.stage / f"{role}_{suffix}.jsonl.gz"
            for row in sources.rows(path):
                if row.get("v2_role") != role:
                    raise ValueError("metric row role mismatch")
                if row.get("target") != TARGET or row.get("population") != "all":
                    continue
                key = tuple(row[field] for field in IDENTITY)
                if key in identities:
                    raise ValueError("duplicate primary/all metric identity")
                identities.add(key)
                if row["method"] not in ("D_DIRECT", "D_KNOWN_EVENT_LOGP"):
                    if key not in frozen or any(
                        row.get(field) != frozen[key].get(field)
                        for field in (*CONFIG_FIELDS, "actual_rank", "k")
                    ):
                        raise ValueError(
                            "metric dimensions/configuration differ from frozen metadata"
                        )
                    observed_model_keys.add(key)
                ident = row["configuration_id"]
                if ident != "D_KNOWN_EVENT_LOGP" and ident != configuration_id(row):
                    raise ValueError("metric configuration ID is inconsistent")
                accumulate(buckets.setdefault(ident, metric_accumulator(row)), row)
        if observed_model_keys != set(frozen):
            raise ValueError("model metric/freeze identity coverage mismatch")
        missing[role] = sorted(expected - set(buckets))
        for ident, acc in sorted(buckets.items()):
            summary, groups = finalize(role, ident, acc)
            pooled.append(summary)
            csv_rows.extend([summary, *groups])
    group_recovery = apply_raw_group_audit(raw_audit_json, run, sources, pooled, csv_rows)
    buffer = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in csv_rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
        )
    csv_bytes = buffer.getvalue().encode()
    observations = sorted({observation for observation, _ in pairs})
    result = {
        "schema_version": "n4-factual-summary-v1",
        "status": "FACTUAL_SUMMARY_COMPLETE",
        "run_root": str(run),
        "primary_target": TARGET,
        "population": "all",
        "stage_execution_status": stage_result.get("status"),
        "stage_wall_seconds": stage_result.get("wall_seconds"),
        "coverage": {
            "status": "COMPLETE"
            if not any(missing.values())
            else "MISSING_EXPECTED_CONFIGURATIONS",
            "expected_configuration_ids": sorted(expected),
            "missing_by_role": missing,
            "observations_present": observations,
            "observations_not_in_frozen_panel": sorted(set(OBSERVATIONS) - set(observations)),
        },
        "pooled_metrics": pooled,
        "costs_by_role": costs_by_role,
        "known_event_accuracy_scope": "pX/v primary/all group-mean absolute errors; "
        "not per-action classification accuracy",
        "group_order": "indices 0..5 follow the sorted group order "
        "of the sealed bank-major metrics",
        "group_nrmse_status": "UNKNOWN unless group energy was provided by the bound raw audit",
        "raw_group_audit": group_recovery,
        "cost_scope": "each packet, C1 supplement, known-event ledger "
        "and unit freeze counted once; "
        "shared measurement is not charged per model",
        "timing_scope": "packet wall_seconds and freeze fit_seconds are recorded wallclock "
        "components, not isolated CPU timings; ledger timers may overlap and are not summed "
        "into stage walltime",
        "integrity_scope": "hashes of consumed files verified against COMPLETE stage manifest; "
        "raw predictions/oracles are not recomputed by this script",
        "source_files": dict(sorted(sources.files.items())),
        "script_sha256": sha256(Path(__file__)),
        "csv_file": "N4_FACTUAL_METRICS.csv",
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "new_model_calls": 0,
        "new_training_updates": 0,
        "scientific_decision_made": False,
    }
    sources.unchanged()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "N4_FACTUAL_METRICS.csv").open("xb") as stream:
        stream.write(csv_bytes)
    with (out / "N4_FACTUAL_SUMMARY.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-audit-json",
        type=Path,
        help="optional audit_modeling_n4_results.py PASS receipt with group energies",
    )
    args = parser.parse_args()
    result = summarize(args.run_root, args.out_dir, raw_audit_json=args.raw_audit_json)
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"]["status"],
                "pooled_rows": len(result["pooled_metrics"]),
                "out_dir": str(args.out_dir.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
