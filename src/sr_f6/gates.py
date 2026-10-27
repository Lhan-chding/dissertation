"""Fail-closed phase gates; hashes establish identity, never scientific truth."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .common import digest, file_hash, read, safe_evidence
from .contract import AMENDMENTS, validate_config

REQUIRED = {
    "SOURCE_BINDING": {
        "actual_server_source",
        "model_content_identity",
        "three_parent_identities",
        "native_scorer_identity",
        "environment_identity",
        "retention_ids_and_files",
    },
    "DATA_AND_MEASUREMENT": {
        "dual_oracle",
        "global_holdout",
        "split_isolation",
        "factor_matching",
        "gold_blind_inputs",
        "native_j_parity_300",
        "processor_visual_48",
        "token_alignment",
        "blind_response_review_360",
    },
    "PARENT_CENSUS": {
        "complete_manifest_matrix",
        "observability",
        "conditional_exposure",
        "shuffle_exposure",
        "training_coverage_manipulation",
    },
    "RUNTIME_PILOT": {
        "nine_complete_pilots",
        "probability_gates",
        "gradient_audit",
        "single_update_transactions",
        "recovery_parity",
        "repeated_intervention",
        "development_endpoints",
    },
    "FORMAL_FREEZE": {
        "source_config_data_identity",
        "registered_54_path_order",
        "all_external_ids",
        "test_access_lock",
        "statistics_identity",
    },
}
PREFLIGHT_REQUIRED = {
    "actual_server_source",
    "data_math_audit",
    "native_j_parity_300",
    "processor_visual_48",
    "actual_tokenizer_fixtures",
    "dev_cal_manifest",
}


def _manifest_members(root, base, manifest, manifest_digest):
    """Hash at freeze once; unchanged files reuse an explicit verification cache.

    Metadata is only a cache invalidation key, never the scientific identity.
    A content change triggers rehash against the original immutable manifest.
    """
    cache_dir = Path(root) / ".verification-cache"
    cache_dir.mkdir(exist_ok=True)
    if cache_dir.is_symlink():
        raise PermissionError("Verification cache cannot be a symlink")
    cache_path = cache_dir / f"{manifest_digest}.json"
    if cache_path.is_symlink():
        raise PermissionError("Verification cache file cannot be a symlink")
    cache = read(cache_path) if cache_path.exists() else {}
    current = {}
    seen = set()
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise PermissionError("Empty content manifest")
    for member in manifest["files"]:
        actual = safe_evidence(base, member["path"])
        if actual in seen:
            raise PermissionError("Duplicate manifest member")
        stat = actual.stat()
        metadata = [stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        expected = {"stat": metadata, "sha256": member.get("sha256")}
        if stat.st_size != member.get("bytes"):
            raise PermissionError("Manifest member size differs: " + str(actual))
        if cache.get(str(actual)) != expected:
            if file_hash(actual) != member.get("sha256"):
                raise PermissionError("Manifest member changed: " + str(actual))
            after = actual.stat()
            if metadata != [
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ]:
                raise PermissionError("Manifest member changed during verification")
        current[str(actual)] = expected
        seen.add(actual)
    if current != cache:
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("x") as stream:
            json.dump(current, stream)
        os.replace(temporary, cache_path)
    return seen


def _evidence(root, entries, required, context):
    if not isinstance(entries, list) or not entries:
        raise PermissionError("No actual evidence")
    checked = set()
    paths = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {"path", "sha256", "checks"} <= entry.keys():
            raise PermissionError("Incomplete evidence binding")
        path = safe_evidence(root, entry["path"])
        if path in paths or file_hash(path) != entry["sha256"]:
            raise PermissionError("Duplicate or changed evidence")
        paths.add(path)
        report = read(path)
        if report.get("status") != "PASS" or report.get("checks") is None:
            raise PermissionError("Evidence must itself contain completed checks")
        if report.get("context") != context:
            raise PermissionError("Evidence belongs to a different configuration/source/data")
        for dependency in report.get("inputs", []):
            leaf = safe_evidence(root, dependency["path"])
            if file_hash(leaf) != dependency.get("sha256"):
                raise PermissionError("Evidence input changed: " + str(leaf))
        for check in entry["checks"]:
            if check not in required or report["checks"].get(check) is not True:
                raise PermissionError("Evidence check is absent or not successful")
            checked.add(check)
    if checked != required:
        raise PermissionError("Missing evidence checks: " + ", ".join(sorted(required - checked)))


def _context(root, config_path):
    config = read(config_path)
    validate_config(config)
    identity = read(Path(root) / "IDENTITY.json")
    expected = {
        "config_sha256": file_hash(config_path),
        "implementation_digest": digest(AMENDMENTS),
    }
    if any(identity.get(k) != v for k, v in expected.items()):
        raise PermissionError("Identity does not bind the current contract")
    for name in ("source_manifest", "data_manifest"):
        item = identity.get(name, {})
        path = safe_evidence(root, item.get("path", ""))
        if file_hash(path) != item.get("sha256"):
            raise PermissionError("Identity artifact changed: " + name)
        manifest = read(path)
        base = Path(__file__).resolve().parents[2] if name == "source_manifest" else root
        seen = _manifest_members(root, base, manifest, item["sha256"])
        if name == "source_manifest":
            expected = set((base / "src").rglob("*.py")) | set((base / "tests/sr_f6").rglob("*.py"))
            if not expected <= seen:
                raise PermissionError(
                    "Current executable sources and inherited dependencies are not fully bound"
                )
    return identity


def require_census_ready(root, *, config_path=None):
    root = Path(root)
    config_path = config_path or root / "CONFIG.json"
    context = _context(root, config_path)
    receipt = read(root / "PREFLIGHT/CENSUS_READY.json")
    if (
        receipt.get("status") != "READY_FOR_CENSUS"
        or receipt.get("phase") != "DEV_CAL"
        or receipt.get("execution_origin") != "SERVER_ACTUAL"
        or receipt.get("context") != context
        or receipt.get("optimizer_updates") != 0
    ):
        raise PermissionError("No bound, actual census preflight")
    _evidence(root, receipt.get("evidence"), PREFLIGHT_REQUIRED, context)
    return context


def launch_check(root, *, config_path=None, through="FORMAL_FREEZE"):
    root = Path(root)
    if through not in REQUIRED:
        raise ValueError("Unknown stage")
    try:
        context = _context(root, config_path or root / "CONFIG.json")
        for stage, checks in REQUIRED.items():
            receipt = read(root / "GATES" / f"{stage}.json")
            if (
                receipt.get("stage") != stage
                or receipt.get("status") != "PASS"
                or receipt.get("execution_origin") != "SERVER_ACTUAL"
                or receipt.get("context") != context
                or receipt.get("exit_code") != 0
                or not receipt.get("executed_command")
                or not receipt.get("started_at")
                or not receipt.get("completed_at")
            ):
                raise PermissionError("Incomplete or mismatched stage: " + stage)
            _evidence(root, receipt.get("evidence"), checks, context)
            if stage == through:
                break
    except (OSError, ValueError, TypeError, KeyError, PermissionError) as exc:
        return {"allowed": False, "through": through, "errors": [str(exc)]}
    return {
        "allowed": True,
        "through": through,
        "errors": [],
        "limitation": (
            "content identity and required checks; not independent proof of evidence truth"
        ),
    }
