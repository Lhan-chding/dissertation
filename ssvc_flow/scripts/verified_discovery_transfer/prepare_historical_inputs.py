"""Read-only historical metadata and proven training-replay collector.

Run on the historical server with --root and redirect stdout locally. This
script never writes to its input tree and never calls a model or solver.
Completeness refers to the declared extant metadata inventory, not deleted
evidence or arbitrary external storage. All generated scenes are excluded,
whether or not they can be shown to have received a model call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

PRUNED = {
    "envs",
    "cache",
    "images",
    ".git",
    "__pycache__",
    "node_modules",
    "checkpoints",
    "checkpoint",
    "raw",
    "segments",
    "startup",
    "logs",
    "tmp",
    "archives",
    "transfer",
    "exports",
    ".pytest_cache",
    "fixtures",
    "r1_fixtures",
    "verified_discovery_transfer_20261006",
}
DATA_NAMES = {
    "train.jsonl",
    "dev.jsonl",
    "confirm.jsonl",
    "natural_pool.jsonl",
    "calibration.jsonl",
    "control.jsonl",
    "ood.jsonl",
    "scenes.jsonl",
}
QUOTAS = {"duplicate_encoding": 32, "cross_series": 16, "trend": 16}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def load_rows(path):
    data = path.read_bytes()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    else:
        rows = [json.loads(data)]
    return data, rows


def inventory(root):
    paths, errors, symlinks = [], [], []
    for base, dirs, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
        for name in dirs:
            path = Path(base) / name
            if path.is_symlink():
                symlinks.append({"path": str(path), "target": str(path.resolve())})
        dirs[:] = [d for d in dirs if d not in PRUNED]
        for name in files:
            if name.startswith("._") or not name.endswith((".json", ".jsonl")):
                continue
            candidate = name in DATA_NAMES or any(
                key in name.lower()
                for key in ("scene", "prompt", "prepared", "cases", "audit_labels")
            )
            if not candidate:
                continue
            path = Path(base) / name
            try:
                if path.stat().st_size > 20_000_000:
                    errors.append({"path": str(path), "error": "metadata exceeds 20 MB bound"})
                else:
                    paths.append(path)
            except OSError as exc:
                errors.append({"path": str(path), "error": str(exc)})
    return sorted(paths), errors, symlinks


def collect(root):
    paths, errors, symlinks = inventory(root)
    sources, excluded, scenes, no_truth_files = [], set(), set(), []
    for path in paths:
        try:
            data, rows = load_rows(path)
            count = 0
            for row in rows:
                for obj in objects(row):
                    world = obj.get("truth_world", obj.get("true_world"))
                    if not (
                        isinstance(world, list)
                        and len(world) == 4
                        and all(type(v) is int for v in world)
                    ):
                        continue
                    excluded.add(tuple(sorted(world)))
                    scenes.add(
                        obj.get(
                            "base_scene_id", obj.get("task_id", digest(json.dumps(world).encode()))
                        )
                    )
                    count += 1
            receipt = {
                "path": str(path),
                "sha256": digest(data),
                "bytes": len(data),
                "world_occurrences": count,
            }
            if count:
                sources.append(
                    {
                        **receipt,
                        "scope": (
                            "all generated scenes including train, evaluation and uncalled metadata"
                        ),
                    }
                )
            else:
                no_truth_files.append(receipt)
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})

    historical = root / "prospective_selection_v2_20260924"
    prepared_path = historical / "prepared_development.json"
    prepared_data = prepared_path.read_bytes()
    prepared = json.loads(prepared_data)
    records = {
        p["prompt_id"]: p for p in prepared["source_prompts"] + prepared["continuation_prompts"]
    }
    banned = {p["base_scene_id"] for panel in prepared["panels"].values() for p in panel}
    cases_path = (
        root
        / "protocol_state_probes_v1_20261005/code_ad09688"
        / "ssvc_flow/docs/protocol_state_probes/design/manifests/cases.jsonl"
    )
    cases_data, cases = load_rows(cases_path)
    banned.update(case["base_scene_id"] for case in cases)
    banned_sources = [
        {
            "path": str(prepared_path),
            "sha256": digest(prepared_data),
            "scope": "all historical D/P panel scene IDs",
        },
        {
            "path": str(cases_path),
            "sha256": digest(cases_data),
            "scope": "all probe D/P/U22 scene IDs, including all transformations",
        },
    ]

    # Scan the entire frozen lineage's committed outputs before scene-ID ranking.
    # Failed/uncommitted attempts never enter the candidate population.
    eligible, selected_ids, scanned = [], set(), []
    segments = historical / "campaign/sources/61001/segments"
    for commit_path in sorted(segments.glob("*/COMMIT.json")):
        commit_data = commit_path.read_bytes()
        commit = json.loads(commit_data)
        path = Path(commit["samples"]["path"])
        if not path.resolve().is_relative_to(commit_path.parent.resolve()):
            raise ValueError(f"Committed samples escaped their segment: {path}")
        data = path.read_bytes()
        artifact_hash = digest(data)
        if artifact_hash != commit["samples"]["sha256"]:
            raise ValueError(f"Committed samples hash mismatch: {path}")
        scanned.append(
            {
                "path": str(path),
                "sha256": artifact_hash,
                "bytes": len(data),
                "commit_path": str(commit_path),
                "commit_sha256": digest(commit_data),
                "start": commit["start"],
                "stop": commit["stop"],
            }
        )
        for line_number, line in enumerate(data.splitlines(), 1):
            row = json.loads(line)
            record = records.get(row.get("prompt_id"))
            if not record or row.get("base_scene_id") in banned | selected_ids:
                continue
            family = record["family"]
            if family not in QUOTAS:
                continue
            if not (record.get("split") == row.get("split") == row.get("role") == "train"):
                continue
            if (
                record.get("interface") != "SYMBOLIC_FRESH"
                or row.get("interface") != "SYMBOLIC_FRESH"
            ):
                continue
            if row.get("event") != "X" or row.get("category") != "X":
                continue
            raw = row.get("raw_completion")
            try:
                vector = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(vector, list)
                or len(vector) != 4
                or not all(type(v) is int and 0 <= v <= 99 for v in vector)
            ):
                continue
            if vector != record["scene"]["truth_world"]:
                continue
            prompt = row.get("input_audit", {}).get("final_prompt", "")
            if record["prompt"]["system"] not in prompt or record["prompt"]["user"] not in prompt:
                continue
            source = {
                "split": "train",
                "interface": "SYMBOLIC_FRESH",
                "protocol": "O0",
                "request_id": row["sample_key"],
                "base_scene_id": row["base_scene_id"],
                "artifact_path": str(path),
                "artifact_sha256": artifact_hash,
                "commit_path": str(commit_path),
                "commit_sha256": digest(commit_data),
                "line": line_number,
                "row_sha256": digest(line + b"\n"),
                "prompt_registry_path": str(prepared_path),
                "prompt_registry_sha256": digest(prepared_data),
                "lineage_id": row["lineage_id"],
                "step": row["step"],
                "historical_event": row["event"],
                "historical_role": row["role"],
                "prompt_identity_verified": True,
                "banned_panel_overlap": False,
            }
            eligible.append(
                {
                    "record": record,
                    "raw_completion": raw,
                    "canonical_vector": vector,
                    "source": source,
                    "verified": True,
                }
            )
            selected_ids.add(row["base_scene_id"])
    selected, counts = [], Counter()
    for candidate in sorted(eligible, key=lambda c: c["source"]["base_scene_id"]):
        family = candidate["record"]["family"]
        if counts[family] < QUOTAS[family]:
            selected.append(candidate)
            counts[family] += 1

    exclusion = {
        "schema": "vdt-historical-exclusion-v1",
        "complete": not errors and bool(sources),
        "completeness_scope": (
            "extant generated scene and scene/prompt/case/prepared metadata "
            "in the inventoried project tree"
        ),
        "sources": sources,
        "truth_orbits": [list(v) for v in sorted(excluded)],
        "exposed_scene_count": len(scenes),
        "exposed_scene_count_semantics": (
            "conservative generated-scene superset, not demonstrated model-call count"
        ),
        "inventory_root": str(root),
        "inventory_paths": [str(p) for p in paths],
        "pruned_directory_names": sorted(PRUNED),
        "symlink_directories": symlinks,
        "metadata_without_truth": no_truth_files,
        "errors": errors,
        "limitations": [
            "Deleted, inaccessible and external historical metadata cannot be certified.",
            "Archived copies and raw model output trees are not reparsed; "
            "all extant generated metadata is conservatively excluded.",
            "Completeness is metadata-inventory scoped, "
            "not a universal claim about every historical exposure.",
        ],
    }
    return {
        "exclusion": exclusion,
        "replay_candidates": selected,
        "replay_receipt": {
            "schema": "vdt-historical-replay-receipt-v1",
            "complete": dict(counts) == QUOTAS,
            "counts": dict(counts),
            "required_counts": QUOTAS,
            "banned_scene_count": len(banned),
            "eligible_counts_before_scene_id_selection": dict(
                Counter(c["record"]["family"] for c in eligible)
            ),
            "banned_sources": banned_sources,
            "scanned_sources": scanned,
            "selection_rule": (
                "source lineage 61001; all COMMIT.json-bound and SHA256-verified samples; "
                "first accepted raw per eligible scene by segment/row order; "
                "then global ascending base_scene_id and fixed family quotas"
            ),
            "verification": (
                "actual historical train output; exact registered O0 system/user in actual input; "
                "strict integer vector equals scene truth; no new model call or solver target"
            ),
        },
    }


def audit_dataset_bindings(root):
    """Trace extant small runtime/config/manifest data roots to scene files."""
    bindings, errors, roots = [], [], set()
    skip = PRUNED | {"legacy_fixtures"}
    for base, dirs, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
        dirs[:] = [name for name in dirs if name not in skip]
        for name in files:
            if name.startswith("._") or not name.endswith(".json"):
                continue
            if not any(
                term in name.lower()
                for term in ("runtime", "manifest", "machine", "config", "binding")
            ):
                continue
            path = Path(base) / name
            if path.stat().st_size > 2_000_000:
                continue
            try:
                data = path.read_bytes()
                value = json.loads(data)
            except (OSError, ValueError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
                continue
            refs = []
            for obj in objects(value):
                for key, item in obj.items():
                    if key == "data_root" and isinstance(item, str) and item.startswith("/"):
                        resolved = str(Path(item).resolve())
                        refs.append({"key": key, "path": item, "resolved": resolved})
                        roots.add(resolved)
            if refs:
                bindings.append({"path": str(path), "sha256": digest(data), "references": refs})
    datasets = []
    for root_name in sorted(roots):
        dataset = Path(root_name)
        receipt = {"root": root_name, "exists": dataset.is_dir(), "files": [], "truth_orbits": []}
        orbits = set()
        if not dataset.is_dir():
            errors.append({"path": root_name, "error": "referenced data root missing"})
        for path in sorted(dataset.glob("*.jsonl")):
            if path.name.startswith("._"):
                continue
            data, rows = load_rows(path)
            count = 0
            for row in rows:
                world = row.get("truth_world")
                if (
                    isinstance(world, list)
                    and len(world) == 4
                    and all(type(v) is int for v in world)
                ):
                    orbits.add(tuple(sorted(world)))
                    count += 1
            receipt["files"].append(
                {"path": str(path), "sha256": digest(data), "world_count": count}
            )
        receipt["truth_orbits"] = [list(v) for v in sorted(orbits)]
        datasets.append(receipt)
    return {
        "schema": "vdt-historical-dataset-bindings-v1",
        "bindings": bindings,
        "datasets": datasets,
        "errors": errors,
        "scope": (
            "extant <=2 MB JSON runtime/config/manifest/machine/binding files; "
            "explicit data_root keys"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-bindings", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            audit_dataset_bindings(args.root) if args.audit_bindings else collect(args.root),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
