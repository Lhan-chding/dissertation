"""Summarize recorded physical work, preserving the scope of every ledger."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

from .cost import COUNTERS
from .io import sha256_file


def audit_recorded_costs(run_root):
    root = Path(run_root).resolve()
    evidence = {}
    groups = defaultdict(lambda: defaultdict(float))
    wall = {}
    seen = set()
    missing = []

    def read(path):
        evidence[str(path)] = sha256_file(path)
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
            return json.load(stream)

    def add(group, cost):
        cost = dict(cost)
        # Parent Jacobian diagnostics predate the contrast ledger schema.
        if "batched_vjp_calls" in cost:
            cost["Jacobian_VJP_calls"] = cost["batched_vjp_calls"]
            cost["physical_policy_forwards"] = cost["scored_prompt_policies"]
        for key in set(COUNTERS) | set(cost):
            value = cost.get(key, 0)
            if isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError("negative physical operation cost")
                groups[group][key] += value

    # Metadata is the physical observation ledger. Fit-bank acquisition views in
    # COST_LEDGER are scenarios that overlap these measurements and are not added.
    for stage in [
        "smoke_failed_resource_02",
        "smoke_failed_resource_03",
        "smoke",
        "N1",
        "N2",
        "N4",
    ]:
        base = root / stage
        for path in sorted(base.rglob("packets.json")):
            meta = read(path)
            full = path.parent / "packet_metadata.json.gz"
            if full.exists():
                meta = read(full)
            packet = path.parent / "packet_arrays.npz"
            identity = str(packet.resolve())
            if identity in seen:
                raise ValueError("same physical packet included twice")
            seen.add(identity)
            if not meta.get("packets"):
                missing.append(str(path))
                continue
            for row in meta["packets"]:
                add(stage + "_finite_measurement", row["cost"])
            for row in meta.get("C1_supplementary_costs", []):
                add(stage + "_C1_additional_measurement", row["cost"])
        for path in sorted(base.rglob("C1_auxiliary/*/*/*/receipt.json")):
            add(stage + "_C1_repair", read(path).get("additional_cost", {}))
        for path in sorted(base.rglob("freeze.json")):
            meta = read(path)
            if "fit_seconds" in meta:
                groups[stage + "_fit_and_seal"]["fit_seconds"] += meta["fit_seconds"]
        for path in sorted(base.rglob("covariance_audit.json")):
            add(stage + "_oracle_covariance", read(path).get("oracle_cost", {}))
        for path in sorted(base.rglob("direct_known_event.json")):
            item = read(path)
            add(stage + "_known_event_audit", item.get("cost", item))
        for path in sorted(base.rglob("diagnostics.json")):
            item = read(path)
            if "oracle_covariance_cost" in item:
                add(stage + "_oracle_isolation", item["oracle_covariance_cost"])
        for path in sorted((base / "cost").rglob("*.json")):
            item = read(path)
            for name in [
                "direct_four_evaluation_banks",
                "known_event_four_evaluation_banks",
                "full_enumeration_four_evaluation_banks",
            ]:
                if name in item:
                    add(stage + "_cold_baseline_replay", item[name])
        stage_result = base / "stage_result.json"
        if stage_result.exists():
            wall[stage] = read(stage_result).get("wall_seconds", 0.0)
        smoke = base / "smoke_result.json"
        if smoke.exists():
            s = read(smoke)
            wall[stage] = s.get("observation_wall_seconds", 0.0) + s.get("fit_wall_seconds", 0.0)
    for stage in ["oracle_diagnostics_pre_active_fix", "oracle_diagnostics"]:
        path = root / stage / "SUMMARY.json"
        if path.exists():
            s = read(path)
            add(stage, s.get("cost", {}))
            wall[stage] = s.get("elapsed_seconds", 0.0)
    for path in sorted((root / "implementation/cost_smoke").glob("*.json")):
        item = read(path)
        for name in [
            "direct_four_evaluation_banks",
            "known_event_four_evaluation_banks",
            "full_enumeration_four_evaluation_banks",
        ]:
            if name in item:
                add("cold_cost_smoke", item[name])
    repair = root / "implementation/C1_auxiliary_regression/receipt.json"
    if repair.exists():
        add("C1_regression_frozen_measurement", read(repair).get("additional_cost", {}))
    totals = defaultdict(float)
    for counters in groups.values():
        for key, value in counters.items():
            totals[key] += value
    return {
        "schema_version": "contrast-recorded-cost-v1",
        "status": "PASS" if not missing else "PARTIAL",
        "operation_counts": dict(totals),
        "categories": dict(groups),
        "wall_seconds_by_stage": wall,
        "evidence_files": evidence,
        "packet_ledger_count": len(seen),
        "missing_packet_ledgers": missing,
        "reused_actions_are_references_not_new_draws": True,
        "scenario_acquisition_views_not_double_counted": True,
        "fit_and_seal_wall_includes_prediction_and_compression": True,
        "scope": (
            "Recorded smoke/N1/N2/N4 measurement and baseline ledgers, "
            "plus explicit Jacobian and repair diagnostics"
        ),
        "remaining_work_ledgers": (
            "CPU timing replay, independent reconstruction, "
            "N0 historical ablations and regression have separate receipts"
        ),
        "early_failed_smoke_cost_status": (
            "UNRECORDED_PARTIAL_ATTEMPT_RETAINED; no successful stage substituted"
        ),
    }
