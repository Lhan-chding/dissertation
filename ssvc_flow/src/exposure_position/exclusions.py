"""Audited, global historical truth-orbit exclusions for SER-J23.

Only declared sources and bounded, explicitly recorded roots are inspected. A
PASS certifies that scope, never all private files or unknown hosts. No model
answers, observed vectors, or inferred solutions are promoted to truth.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from ..exposure_substitution.schema import digest, valid_vector, write_json
from .schema import EXPERIMENT_ID

REQUIRED_CATEGORIES = ("historical_index", "legacy_j2", "later_local", "manual_fixtures")
KINDS = frozenset({"history_index", "audit", "training_targets", "manual_fixture"})
DEFAULT_GLOBS = (
    "historical_exclusion.json",
    "*audit*.jsonl",
    "*AUDIT*.jsonl",
    "verified_targets.jsonl",
    "common_targets.jsonl",
    "replay_targets.jsonl",
    "donor_targets.jsonl",
    "train.jsonl",
    "control.jsonl",
    "dev.jsonl",
    "calibration.jsonl",
    "ood.jsonl",
    "MANUAL_FIXTURE_ORBIT_EXCLUSIONS.json",
)
PRUNED_DIRECTORIES = (
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "checkpoints",
    "weights",
    "adapters",
    "raw",
    "raw_outputs",
    "evaluations",
    "eval_outputs",
    "logs",
    "tensorboard",
    "wandb",
)
ANSWER_FIELDS = frozenset(
    {
        "raw",
        "raw_text",
        "raw_answer",
        "raw_response",
        "answer",
        "answers",
        "output",
        "outputs",
        "response",
        "responses",
        "model_outputs",
        "completion",
        "completions",
        "parsed_vector",
        "observed",
        "observed_world",
        "observed_world_public",
    }
)


def truth_orbit(values):
    if not valid_vector(values):
        raise ValueError("Truth requires four exact integers in [0,99]")
    return tuple(sorted(values))


def _safe_path(value):
    if not isinstance(value, (str, Path)) or not str(value) or "\x00" in str(value):
        raise ValueError("Explicit nonempty filesystem path required")
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden in exclusion source paths")
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlinked exclusion paths are forbidden")
    return path


def _source(spec):
    if not isinstance(spec, dict) or not {"path", "category", "kind"} <= spec.keys():
        raise ValueError("Source requires path, category and kind")
    if spec["kind"] not in KINDS or not isinstance(spec["category"], str) or not spec["category"]:
        raise ValueError("Invalid exclusion source category/kind")
    if spec["category"] == "historical_index" and spec["kind"] != "history_index":
        raise ValueError("Historical index category requires an explicit complete index")
    return {
        "path": str(_safe_path(spec["path"])),
        "category": spec["category"],
        "kind": spec["kind"],
    }


def _check_historical_index(value):
    if (
        not isinstance(value, dict)
        or value.get("schema") != "vdt-historical-exclusion-v1"
        or value.get("complete") is not True
        or value.get("errors")
    ):
        raise ValueError("Complete historical exclusion index with no errors is required")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Historical index must retain source provenance")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not all(
                isinstance(source.get(k), str) and source[k] for k in ("path", "scope", "sha256")
            )
            or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
            or ".." in Path(source["path"]).parts
        ):
            raise ValueError("Malformed historical source provenance")
    if not isinstance(value.get("truth_orbits"), list) or not value["truth_orbits"]:
        raise ValueError("Historical index cannot supply an empty exclusion list")
    count = value.get("exposed_scene_count")
    if type(count) is not int or count < len(set(map(truth_orbit, value["truth_orbits"]))):
        raise ValueError("Historical exposed scene count cannot be less than orbit count")


def _extract(value, kind):
    found, counts = set(), Counter()

    def add(vector, field, *, sorted_required=False):
        orbit = truth_orbit(vector)
        if sorted_required and list(orbit) != vector:
            raise ValueError("Explicit truth orbit keys must already be sorted")
        found.add(orbit)
        counts[field] += 1
        return orbit

    def walk(node):
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        row_orbits = []
        for key in ("true_world", "canonical_vector", "truth_orbit_key"):
            if key in node:
                row_orbits.append(add(node[key], key, sorted_required=key == "truth_orbit_key"))
        if kind == "training_targets" and "target" in node:
            if not isinstance(node["target"], str) or not any(
                isinstance(node.get(key), str) and node[key]
                for key in ("source", "source_task_id", "label_source")
            ):
                raise ValueError("Explicit training target requires a string and source identity")
            row_orbits.append(add(json.loads(node["target"]), "target"))
        if len(set(row_orbits)) > 1:
            raise ValueError("Explicit truth, canonical target and orbit fields disagree")
        for key, child in node.items():
            if key in ("truth_orbits", "truth_orbit_keys"):
                if not isinstance(child, list):
                    raise ValueError("Truth orbit collection must be a list")
                for vector in child:
                    add(vector, key, sorted_required=True)
            elif key not in ANSWER_FIELDS | {
                "target",
                "true_world",
                "canonical_vector",
                "truth_orbit_key",
            }:
                walk(child)

    if kind == "history_index":
        _check_historical_index(value)
    walk(value)
    return found, dict(sorted(counts.items()))


def extract_truth_orbits(value, *, kind="audit"):
    """Parse explicit audit truth; only declared training sources may use target."""
    if kind not in KINDS:
        raise ValueError("Unknown exclusion source kind")
    return _extract(value, kind)[0]


def _read_source(spec, max_file_bytes):
    path = _safe_path(spec["path"])
    if path.suffix not in (".json", ".jsonl") or not path.is_file():
        raise ValueError("Existing regular JSON/JSONL exclusion file required")
    if path.stat().st_size > max_file_bytes:
        raise ValueError("Exclusion file exceeds declared byte bound")
    # Hash the exact bytes parsed; reject an in-place change during the read.
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("Exclusion source changed during read")
    text = raw.decode("utf-8")
    value = (
        [json.loads(line) for line in text.splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else json.loads(text)
    )
    orbits, counts = _extract(value, spec["kind"])
    record = dict(
        spec,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        rows=len(value) if isinstance(value, list) else 1,
        truth_orbits=len(orbits),
        truth_field_counts=counts,
        status="VERIFIED" if orbits else "NO_EXPLICIT_TRUTH",
    )
    if spec["kind"] == "history_index":
        record["inherited_provenance"] = {
            "source_count": len(value["sources"]),
            "sources_hash": digest(value["sources"]),
            "completeness_scope": value.get("completeness_scope"),
            "limitations": value.get("limitations", []),
            "underlying_archived_paths_rechecked": False,
        }
    return orbits, record


def _kind_for(path):
    if path.name == "historical_exclusion.json":
        return "history_index"
    if "target" in path.name.lower():
        return "training_targets"
    if path.name == "MANUAL_FIXTURE_ORBIT_EXCLUSIONS.json":
        return "manual_fixture"
    return "audit"


def _failure_record(spec, error, max_file_bytes):
    """Retain byte identity even when a readable source has invalid JSON/truth."""
    record = dict(
        spec,
        status="UNREADABLE_OR_INVALID",
        error=str(error),
        sha256=None,
        bytes=None,
        truth_orbits=None,
    )
    try:
        path = _safe_path(spec["path"])
        if path.is_file():
            record["bytes"] = path.stat().st_size
            if record["bytes"] <= max_file_bytes:
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        pass  # An unreadable source is a recorded blocking error, never empty truth.
    return record


def _scan(spec):
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("category"), str)
        or not spec["category"]
    ):
        raise ValueError("Scan root requires an explicit category")
    root = _safe_path(spec.get("path", ""))
    globs = tuple(spec.get("globs", DEFAULT_GLOBS))
    if not globs or any(not isinstance(g, str) or not g or "/" in g or "\\" in g for g in globs):
        raise ValueError("Scan globs must be nonempty filename patterns")
    pruned = tuple(sorted(set(PRUNED_DIRECTORIES) | set(spec.get("excluded_directories", ()))))
    if any(not isinstance(d, str) or not d or "/" in d or d in (".", "..") for d in pruned):
        raise ValueError("Excluded directory names must be simple components")
    record = {
        "path": str(root),
        "category": spec["category"],
        "globs": list(globs),
        "excluded_directories": list(pruned),
        "matched_files": [],
        "errors": [],
        "pruned_paths": [],
        "visited_directories": 0,
    }
    if not root.is_dir():
        raise ValueError("Existing exclusion scan directory required")
    sources = []

    def onerror(error):
        record["errors"].append({"path": str(error.filename), "error": str(error)})

    for current, directories, files in os.walk(root, followlinks=False, onerror=onerror):
        record["visited_directories"] += 1
        kept = []
        for name in sorted(directories):
            path = Path(current) / name
            if name in pruned:
                record["pruned_paths"].append(str(path))
            elif path.is_symlink():
                record["errors"].append(
                    {"path": str(path), "error": "symlink directory not followed"}
                )
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in globs):
                path = Path(current) / name
                record["matched_files"].append(str(path))
                sources.append(
                    {"path": str(path), "category": spec["category"], "kind": _kind_for(path)}
                )
    return sources, record


def audit_exclusions(
    *,
    exact_sources,
    scan_roots=(),
    required_categories=REQUIRED_CATEGORIES,
    output_dir=None,
    max_file_bytes=64 * 1024 * 1024,
):
    """Return PASS or BLOCKED with every failure retained; optionally write receipts.

    Exact sources require path/category/kind; scan roots require path/category and
    may supply filename globs or extra excluded directory names. The default
    categories enforce an extant historical index plus J2, later-local, and manual
    evidence. Extra categories can be required for a declared server inventory.
    """
    required = list(required_categories)
    if (
        not required
        or any(not isinstance(c, str) or not c for c in required)
        or len(set(required)) != len(required)
        or type(max_file_bytes) is not int
        or max_file_bytes <= 0
    ):
        raise ValueError("Nonempty unique required categories and positive byte bound required")
    exact_sources, scan_roots = list(exact_sources), list(scan_roots)
    errors, sources, scans, declared = [], [], [], []
    for raw in exact_sources:
        try:
            spec = _source(raw)
            sources.append(spec)
            declared.append(spec)
        except (OSError, ValueError, TypeError) as error:
            errors.append({"source": str(raw), "error": str(error)})
    for raw in scan_roots:
        try:
            discovered, record = _scan(raw)
            scans.append(record)
            sources.extend(discovered)
            errors.extend(record["errors"])
        except (OSError, ValueError, TypeError) as error:
            errors.append({"scan_root": str(raw), "error": str(error)})
    unique = {(s["path"], s["category"], s["kind"]): s for s in sources}
    orbits, records, verified = set(), [], set()
    exact_keys = {(s["path"], s["category"], s["kind"]) for s in declared}
    for key, raw in sorted(unique.items()):
        try:
            spec = _source(raw)
            found, record = _read_source(spec, max_file_bytes)
            records.append(record)
            orbits.update(found)
            if found:
                verified.add(spec["category"])
            elif key in exact_keys:
                errors.append(
                    {"path": spec["path"], "error": "Explicit source has no recognized truth"}
                )
        except (OSError, ValueError, TypeError, RecursionError) as error:
            records.append(_failure_record(raw, error, max_file_bytes))
            errors.append({"path": raw["path"], "error": str(error)})
    missing = sorted(set(required) - verified)
    if missing:
        errors.append(
            {"error": "Required exclusion categories lack verified truth", "categories": missing}
        )
    if not orbits:
        errors.append({"error": "Empty history cannot certify absence of overlap"})
    audit = {
        "schema": "ser-j23-exclusion-audit-v1",
        "phase_id": EXPERIMENT_ID,
        "status": "BLOCKED_EXCLUSION_AUDIT" if errors else "PASS",
        "model_calls": 0,
        "truth_orbit_keys": [list(orbit) for orbit in sorted(orbits)],
        "truth_orbit_count": len(orbits),
        "files": records,
        "errors": errors,
        "max_file_bytes": max_file_bytes,
        "coverage": {
            "scope": "explicit_declared_sources_and_roots_only",
            "exact_sources": declared,
            "scan_roots": scans,
            "required_categories": required,
            "verified_categories": sorted(verified),
            "all_required_categories_verified": not missing,
            "not_claimed": ["all_private_files", "unlisted_hosts_or_roots"],
        },
    }
    audit["audit_hash"] = digest(audit)
    if output_dir is not None:
        destination = _safe_path(output_dir)
        for filename in ("EXCLUSION_AUDIT.json", "TRUTH_ORBITS.json"):
            _safe_path(destination / filename)
            _safe_path(destination / (filename + ".partial"))
        write_json(destination / "EXCLUSION_AUDIT.json", audit)
        write_json(
            destination / "TRUTH_ORBITS.json",
            {
                "schema": "ser-j23-truth-orbits-v1",
                "phase_id": EXPERIMENT_ID,
                "status": audit["status"],
                "audit_hash": audit["audit_hash"],
                "truth_orbit_keys": audit["truth_orbit_keys"],
            },
        )
    return audit


def verify_exclusion_audit(audit, *, verify_sources=True):
    """Verify receipt integrity and, by default, recheck its exact bounded scope."""
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != "ser-j23-exclusion-audit-v1"
        or audit.get("phase_id") != EXPERIMENT_ID
        or audit.get("status") != "PASS"
        or audit.get("model_calls") != 0
        or audit.get("errors")
        or audit.get("audit_hash") != digest({k: v for k, v in audit.items() if k != "audit_hash"})
    ):
        raise ValueError("Verified PASS exclusion audit required")
    coverage = audit.get("coverage", {})
    required, verified = (
        coverage.get("required_categories", []),
        coverage.get("verified_categories", []),
    )
    if (
        not set(REQUIRED_CATEGORIES) <= set(required)
        or not set(required) <= set(verified)
        or coverage.get("all_required_categories_verified") is not True
        or coverage.get("scope") != "explicit_declared_sources_and_roots_only"
        or not audit.get("files")
        or not audit.get("truth_orbit_keys")
    ):
        raise ValueError("Incomplete exclusion coverage or empty historical index")
    keys = extract_truth_orbits({"truth_orbit_keys": audit["truth_orbit_keys"]})
    if (
        len(keys) != audit.get("truth_orbit_count")
        or [list(k) for k in sorted(keys)] != audit["truth_orbit_keys"]
    ):
        raise ValueError("Exclusion orbit count/order mismatch")
    if verify_sources:
        fresh = audit_exclusions(
            exact_sources=coverage["exact_sources"],
            scan_roots=coverage["scan_roots"],
            required_categories=required,
            max_file_bytes=audit["max_file_bytes"],
        )
        if fresh["audit_hash"] != audit["audit_hash"]:
            raise ValueError("Exclusion source inventory or content changed after audit")
    return keys


def load_verified_exclusions(audit_path, *, verify_sources=True):
    path = _safe_path(audit_path)
    audit = json.loads(path.read_text(encoding="utf-8"))
    return verify_exclusion_audit(audit, verify_sources=verify_sources)
