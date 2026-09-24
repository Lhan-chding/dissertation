"""Read-only P0 evidence audit. Historical P outcomes are never renamed E/test."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

HISTORICAL_ENDPOINTS = tuple(
    [("O1", a) for a in ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "GDPO_R4", "SAW_R4")]
    + [("O2", a) for a in ("R0", "R2", "R3", "R4", "GDPO_R4", "SAW_R4")]
)
ROW_FIELDS = (
    "sample_id",
    "prompt_id",
    "base_scene_id",
    "family",
    "interface",
    "raw_completion",
    "event",
    "parsed_world",
    "relation_numerator",
    "relation_denominator",
    "relation_score",
    "answer_correct",
    "valid",
)


def _read(path, receipts):
    payload = Path(path).read_bytes()
    receipts.append(
        {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    )
    return json.loads(payload)


def collect_history(root, panels_path, access_path=None):
    """Read saved endpoint draws and hash each small imported original once.

    No checkpoint tensor, model weight, sealed confirm, or generated image is
    read. The compact import retains exact completion strings and identifiers;
    full token/probability outputs remain at the recorded original paths.
    """
    root, receipts = Path(root), []
    panels = _read(panels_path, receipts)
    endpoints, rows = [], []
    for origin, action in HISTORICAL_ENDPOINTS:
        branch = root / origin / action
        state = branch / "H32.json"
        endpoint = {
            "origin": origin,
            "action": action,
            "state_path": str(state),
            "state_present": state.is_file(),
            "observations": [],
        }
        if state.is_file():
            # Inspect state metadata, not the multi-GB optimizer payload.
            metadata = _read(state, receipts)
            endpoint["state_metadata"] = {
                k: metadata[k]
                for k in (
                    "path",
                    "sha256",
                    "state_hash",
                    "horizon",
                    "step",
                    "status",
                    "execution_kind",
                )
                if k in metadata
            }
        for manifest in sorted(branch.glob("observations/**/LOOK_32.json")):
            look = _read(manifest, receipts)
            if look.get("horizon") != 32:
                continue
            observation = {**look, "manifest_path": str(manifest)}
            imported = []
            for path in sorted((manifest.parent / "samples").glob("*/*.json")):
                block = _read(path, receipts)
                for row in block["rows"]:
                    imported.append({k: row[k] for k in ROW_FIELDS})
            observation["imported_rows"] = len(imported)
            endpoint["observations"].append(observation)
            rows.extend(
                {**row, "origin": origin, "action": action, "panel": look["panel"]}
                for row in imported
            )
        endpoints.append(endpoint)
    access = []
    if access_path is not None and Path(access_path).is_file():
        data = Path(access_path).read_bytes()
        receipts.append(
            {
                "path": str(access_path),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        access = [json.loads(line) for line in data.splitlines() if line.strip()]
    return {
        "schema": "prospective-history-import-v1",
        "root": str(root),
        "panels": panels,
        "endpoints": endpoints,
        "rows": rows,
        "access_records": access,
        "source_receipts": receipts,
        "confirm_payload_opened_by_audit": False,
    }


def audit_rows(rows, prompts):
    """Reparse old completions; exact rational arithmetic verifies identities."""
    from .semantics import reward_bin, semantic_features

    prompts = {p["prompt_id"]: p for p in prompts}
    counts, residuals, buckets = Counter(), Counter(), defaultdict(dict)
    endpoint = defaultdict(Counter)
    sample_ids = set()
    for row in rows:
        identity = (row.get("origin"), row.get("action"), row.get("panel"), row["sample_id"])
        if identity in sample_ids:
            raise ValueError("Duplicate historical sample identity")
        sample_ids.add(identity)
        prompt = prompts[row["prompt_id"]]
        phi = semantic_features(row["raw_completion"], prompt)
        for key in (
            "event",
            "parsed_world",
            "relation_numerator",
            "relation_denominator",
            "relation_score",
            "answer_correct",
            "valid",
        ):
            if key in row and row[key] != phi[key]:
                raise ValueError(f"Historical semantic mismatch: {key}, {row['sample_id']}")
        x, a, v = int(phi["event"] == "X"), int(phi["answer_correct"]), int(phi["valid"])
        c = Fraction(phi["relation_numerator"], phi["relation_denominator"])
        residuals["S=A-X"] += abs(int(phi["event"] == "S") - (a - x))
        residuals["W=V-A"] += abs(int(phi["event"] == "W") - (v - a))
        residuals["I=1-V"] += abs(int(phi["event"] == "I") - (1 - v))
        counts["rows"] += 1
        counts[phi["event"]] += 1
        if prompt["family"] == "trend":
            residuals["trend_C1_nonX=2C2-C-X"] += abs(int(c == 1 and not x) - (2 * c * c - c - x))
            counts["trend_rows"] += 1
        if v:
            residuals["X=F_and_B0"] += abs(x - int(phi["F"] == 1 and phi["B"] == 0))
            residuals["coord_accuracy=(F+3-B)/4"] += abs(
                Fraction(sum(phi["coordinate_correct"]), 4) - Fraction(phi["F"] + 3 - phi["B"], 4)
            )
        elif any(phi[k] is not None for k in ("F", "B", "M")):
            raise ValueError("Invalid completion has nonmissing repair labels")
        key = (
            row.get("origin", "unknown"),
            row.get("action", "unknown"),
            row.get("panel", "unknown"),
        )
        agg = endpoint[key]
        agg["rows"] += 1
        agg["X"] += x
        agg["C1_nonX"] += int(c == 1 and not x)
        agg["valid"] += v
        agg["damage_any"] += int(v and phi["B"] > 0)
        agg["single_edit"] += int(v and phi["M"] == 1)
        agg["coordinate_correct"] += sum(phi["coordinate_correct"])
        if v:
            reward = reward_bin(
                phi["event"], phi["relation_numerator"], phi["relation_denominator"]
            )
            bucket = (row["prompt_id"], reward)
            repair = (phi["F"], phi["B"], phi["M"])
            if repair not in buckets[bucket]:
                buckets[bucket][repair] = {
                    "sample_id": row["sample_id"],
                    "origin": row.get("origin"),
                    "action": row.get("action"),
                    "panel": row.get("panel"),
                    "raw_completion": row["raw_completion"],
                    "F": phi["F"],
                    "B": phi["B"],
                    "M": phi["M"],
                }
    examples = []
    for (pid, reward), repairs in sorted(buckets.items()):
        if len(repairs) > 1:
            scene = prompts[pid]["scene"]
            examples.append(
                {
                    "prompt_id": pid,
                    "reward_bin": reward,
                    "observed_world": scene["observed_world"],
                    "truth_world": scene["truth_world"],
                    "distinct_repair_atoms": len(repairs),
                    "examples": list(repairs.values())[:4],
                }
            )
    metrics = []
    for (origin, action, panel), agg in sorted(endpoint.items()):
        n, valid = agg["rows"], agg["valid"]
        metrics.append(
            {
                "origin": origin,
                "action": action,
                "panel": panel,
                **dict(agg),
                "pX": agg["X"] / n,
                "p_C1_nonX": agg["C1_nonX"] / n,
                "coordinate_accuracy_invalid_zero": agg["coordinate_correct"] / (4 * n),
                "damage_any_given_valid": agg["damage_any"] / valid if valid else None,
                "single_edit_given_valid": agg["single_edit"] / valid if valid else None,
            }
        )
    return {
        "counts": dict(counts),
        "identity_absolute_residual_sums": {k: str(v) for k, v in residuals.items()},
        "identities_pass": all(v == 0 for v in residuals.values()),
        "same_reward_different_repair_buckets": len(examples),
        "same_reward_different_repair_examples": examples[:20],
        "endpoint_metrics": metrics,
        "interpretation": (
            "Descriptive information nonredundancy only; "
            "no prospective selection benefit established."
        ),
    }


def audit_history(config, out):
    """CLI entry: config gives history_import or root/panels_path/access_path."""
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    if "history_import" in config:
        path = Path(config["history_import"])
        payload = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        source = json.loads(payload)
    else:
        source = collect_history(
            config["history_root"], config["history_panels_path"], config.get("history_access_path")
        )
    panels = source["panels"]["panels"]
    observed_counts = defaultdict(Counter)
    for row in source["rows"]:
        observed_counts[row["origin"], row["action"], row["panel"]][row["prompt_id"]] += 1
    for endpoint in source["endpoints"]:
        for obs in endpoint["observations"]:
            panel = obs["panel"]
            counts = observed_counts[endpoint["origin"], endpoint["action"], panel]
            if (
                obs["look"] != 32
                or counts != Counter({p["prompt_id"]: 32 for p in panels[panel]})
                or sum(counts.values()) != obs["imported_rows"]
            ):
                raise ValueError(
                    "Historical panel requires exactly 32 draws for every expected prompt"
                )
    result = audit_rows(source["rows"], [p for rows in panels.values() for p in rows])
    e_ready = []
    for endpoint in source["endpoints"]:
        for obs in endpoint["observations"]:
            if (
                obs["panel"] == "E"
                and obs["look"] == 32
                and obs["imported_rows"] == len(panels["E"]) * 32
            ):
                e_ready.append([endpoint["origin"], endpoint["action"]])
    result.update(
        schema="prospective-history-audit-v1",
        expected_endpoints=16,
        present_endpoints=sum(e["state_present"] for e in source["endpoints"]),
        E_reusable_endpoints=e_ready,
        E_missing_endpoints=[list(e) for e in HISTORICAL_ENDPOINTS if list(e) not in e_ready],
        E_status="REUSABLE_SAVED_DRAWS" if len(e_ready) == 16 else "E_MEASUREMENTS_REQUIRED",
        E_exposure_status="NO_E_ENDPOINT_MANIFEST_FOUND_IN_SCOPED_ROOT"
        if not e_ready
        else "EXTERNAL_PANEL_RECHECK",
        endpoint_inventory=source["endpoints"],
        source_receipts=source["source_receipts"],
        exposure={
            "scope": source["root"],
            "access_records": source["access_records"],
            "historical_panels_sealed_confirm_read": source["panels"].get("sealed_confirm_read"),
            "confirm_payload_opened_by_audit": source["confirm_payload_opened_by_audit"],
            "absence_of_access_record_proves_never_read": False,
            "excluded": source["panels"].get("excluded", {}),
            "panel_scene_ids": {
                k: sorted({p["base_scene_id"] for p in rows}) for k, rows in panels.items()
            },
        },
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "HISTORY_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return result
