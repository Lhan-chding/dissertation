"""Verified relocation and resource-only amendment of a frozen N3 decision.

This module never fits, selects, bootstraps, or runs a model. Old source bytes
must remain available in a separate snapshot when execution source has changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from .io import sha256_file, verify_run_manifest
from .protocol import CONFIG_SHA256, config_sha256, load_config, resource_gate, validate_config
from .selection import canonical_hash, verify_selection_lock

_SHA = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = ("source_hashes", "input_hashes", "packet_hashes", "selector_source_hashes")
_CHANGEABLE = {
    "status",
    "allow_fresh_cpu",
    "resource_gate",
    "protocol_sha256",
    "source_hashes",
    "input_hashes",
    "packet_hashes",
    "selector_source_hashes",
    "binding",
    "lock_sha256",
    "data_gate",
    "server_transfer",
}


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"absolute path without traversal required: {path}")
    return path


def _mappings(mapping):
    items = list(mapping.items()) if isinstance(mapping, dict) else list(mapping)
    normalized = []
    seen = set()
    for original, destination in items:
        original = _absolute(original)
        if original in seen:
            raise ValueError("duplicate source prefix mapping")
        seen.add(original)
        normalized.append((original, _absolute(destination).resolve()))
    return sorted(normalized, key=lambda pair: len(pair[0].parts), reverse=True)


def _relocate(path, mappings, original_project_root):
    path = Path(path)
    if not path.is_absolute():
        path = original_project_root / path
    path = _absolute(path)
    for original, destination in mappings:
        if path == original or original in path.parents:
            result = (destination / path.relative_to(original)).resolve()
            if result != destination and destination not in result.parents:
                raise ValueError(f"relocated path escapes destination: {path}")
            return result
    raise ValueError(f"unmapped original evidence path: {path}")


def _verified_mapping(mapping, mappings, original_project_root, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"missing {label} hashes")
    result = {}
    for name, expected in mapping.items():
        path = _relocate(name, mappings, original_project_root)
        if not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise ValueError(f"invalid {label} hash: {name}")
        if str(path) in result:
            raise ValueError(f"multiple original paths collapse onto {path}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} hash changed or missing: {path}")
        result[str(path)] = expected
    return result


def _current_sources(mapping, label):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"missing {label} hashes")
    result = {}
    for name, expected in mapping.items():
        path = _absolute(name).resolve()
        if not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise ValueError(f"invalid {label} hash")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} hash changed or missing: {path}")
        if str(path) in result:
            raise ValueError(f"duplicate current source path: {path}")
        result[str(path)] = expected
    return result


def _merge_hashes(target, other):
    for path, expected in other.items():
        if path in target and target[path] != expected:
            raise ValueError(f"conflicting lineage hashes: {path}")
        target[path] = expected


def _science_payload(lock):
    return {key: value for key, value in lock.items() if key not in _CHANGEABLE}


def _original_decision(lock, manifest):
    if not verify_selection_lock(lock) or manifest["binding"] != lock.get("binding"):
        raise ValueError("original selection lock hash or binding changed")
    if manifest["status"] != "COMPLETE":
        raise ValueError("original N3 is not COMPLETE")
    for gate in ("science_gate", "cost_gate", "input_identity_gate", "data_gate"):
        if lock.get(gate, {}).get("status") != "PASS":
            raise ValueError(f"original {gate} did not pass; relocation cannot repair science")
    if lock.get("status") != "RESOURCE_REVIEW_REQUIRED" or lock.get("allow_fresh_cpu") is not False:
        raise ValueError("resource-only transfer requires the original resource-blocked lock")
    if lock.get("resource_gate", {}).get("status") != "FAIL":
        raise ValueError("original resource gate must retain its failed verdict")
    if lock.get("protocol_sha256") != CONFIG_SHA256 or "server_transfer" in lock:
        raise ValueError("requires the original unamended protocol decision")
    if not 1 <= len(lock.get("selected", [])) <= 2:
        raise ValueError("one or two original frozen selections required")
    for name, field in (
        ("source", "source_hashes"),
        ("data", "input_hashes"),
        ("packet", "packet_hashes"),
    ):
        if lock["binding"].get(name) != canonical_hash(lock.get(field)):
            raise ValueError(f"original {name} binding changed")
    if lock["binding"].get("selector") != canonical_hash(lock["selected"]):
        raise ValueError("original selector binding changed")
    if lock["binding"].get("config") != CONFIG_SHA256:
        raise ValueError("original config binding changed")


def _amendment(config, path):
    validate_config(config)
    original = load_config()
    stripped = {key: value for key, value in config.items() if key != "execution_amendment"}
    permitted = copy.deepcopy(original)
    permitted["resources"]["max_added_output_gib"] = config["resources"]["max_added_output_gib"]
    if stripped != permitted or not config.get("execution_amendment"):
        raise ValueError("amendment must be resource-only with explicit execution metadata")
    path = _absolute(path).resolve()
    expected = config_sha256(config)
    if (
        expected == CONFIG_SHA256
        or sha256_file(path) != expected
        or json.loads(path.read_bytes()) != config
    ):
        raise ValueError("amendment byte hash or pinned server config mismatch")
    return path, expected


def build_server_transfer(
    original_selection_path,
    *,
    prefix_mappings,
    original_project_root,
    amended_config,
    amendment_path,
    resource_forecast,
    source_hashes,
    server_seed_inventory_roots,
    selector_source_hashes=None,
    original_source_prefix_mappings=None,
):
    """Prepare a new lock after verifying all relocated historical evidence.

    ``resource_forecast`` uses execution units: wall_seconds, peak_ram_gib,
    added_output_bytes, temporary_bytes. Prefixes and current-source keys are
    absolute paths. The optional source map locates immutable old source bytes;
    ordinary evidence always uses ``prefix_mappings``. No files are written.
    """
    original_path = _absolute(original_selection_path).resolve()
    if original_path.name != "MODEL_SELECTION_LOCK.json" or original_path.parent.name != "N3":
        raise ValueError("original selection must be N3/MODEL_SELECTION_LOCK.json")
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    manifest = verify_run_manifest(original_path.parent)
    _original_decision(original, manifest)
    amendment_path, protocol_hash = _amendment(amended_config, amendment_path)
    authorized_original = amended_config["execution_amendment"].get(
        "original_selection_lock_sha256"
    )
    if hashlib.sha256(original_bytes).hexdigest() != authorized_original:
        raise ValueError("original selection file hash differs from the authorized amendment")
    old_root = _absolute(original_project_root)
    mappings = _mappings(prefix_mappings)
    source_mappings = (
        _mappings(original_source_prefix_mappings)
        if original_source_prefix_mappings is not None
        else mappings
    )
    relocated = {
        field: _verified_mapping(
            original.get(field),
            source_mappings if field in ("source_hashes", "selector_source_hashes") else mappings,
            old_root,
            field,
        )
        for field in _HASH_FIELDS
    }
    current_sources = _current_sources(source_hashes, "server source")
    selectors = _current_sources(selector_source_hashes or source_hashes, "server selector source")
    for path, expected in selectors.items():
        if current_sources.get(path) != expected:
            raise ValueError("server selector source must be included in execution source binding")
    roots = sorted({str(_absolute(path).resolve()) for path in server_seed_inventory_roots})
    if not roots or any(not Path(path).is_dir() for path in roots):
        raise ValueError("existing explicit server seed inventory roots required")
    files = {
        str(original_path.parent / name): expected for name, expected in manifest["outputs"].items()
    }
    files[str(original_path.parent / "RUN_MANIFEST.json")] = sha256_file(
        original_path.parent / "RUN_MANIFEST.json"
    )
    preserved = {name: (original_path.parent / name).read_bytes() for name in manifest["outputs"]}
    # Retain every historical source, statistic, packet, and amendment in lineage.
    inputs = dict(relocated["input_hashes"])
    for mapping in (
        relocated["source_hashes"],
        relocated["selector_source_hashes"],
        files,
        {str(amendment_path): protocol_hash},
    ):
        _merge_hashes(inputs, mapping)
    gate = resource_gate(resource_forecast, amended_config)
    lock = copy.deepcopy(original)
    lock.update(
        source_hashes=current_sources,
        selector_source_hashes=selectors,
        input_hashes=inputs,
        packet_hashes=relocated["packet_hashes"],
        protocol_sha256=protocol_hash,
        resource_gate={
            "status": "PASS" if gate["passed"] else "FAIL",
            "reasons": gate["reasons"],
            "forecast": {
                "total_walltime_hours": resource_forecast.get("wall_seconds", 0) / 3600
                if resource_forecast.get("wall_seconds") is not None
                else None,
                "peak_ram_gib": resource_forecast.get("peak_ram_gib"),
                "added_output_gib": resource_forecast.get("added_output_bytes", 0) / 2**30
                if resource_forecast.get("added_output_bytes") is not None
                else None,
                "temporary_gib": resource_forecast.get("temporary_bytes", 0) / 2**30
                if resource_forecast.get("temporary_bytes") is not None
                else None,
                "basis": "authorized server resource amendment; frozen scientific selection",
            },
        },
        allow_fresh_cpu=bool(gate["passed"]),
        status="FROZEN_FOR_CONDITIONAL_FRESH_CPU" if gate["passed"] else "RESOURCE_REVIEW_REQUIRED",
    )
    lock["data_gate"]["fresh_seed_inventory_roots"] = roots
    provenance = {
        "schema_version": "verified-server-transfer-v1",
        "original_selection_file_sha256": sha256_file(original_path),
        "original_lock_sha256": original["lock_sha256"],
        "original_binding": original["binding"],
        "original_protocol_sha256": original["protocol_sha256"],
        "amendment_sha256": protocol_hash,
        "scientific_payload_sha256": canonical_hash(_science_payload(original)),
        "original_seed_inventory_roots": original["data_gate"].get(
            "fresh_seed_inventory_roots", []
        ),
        "server_seed_inventory_roots": roots,
        "prefix_mappings": [[str(a), str(b)] for a, b in mappings],
        "original_source_prefix_mappings": [[str(a), str(b)] for a, b in source_mappings],
        "relocated_original_hashes": relocated,
        "original_N3_hashes": files,
        "resource_gate": gate,
        "science_recomputed": False,
        "configuration_reselected": False,
        "fresh_scale_changed": False,
        "data_gate_change": "seed-inventory locator roots only; original scientific audit retained",
        "original_project_root": str(old_root),
        "original_selection_path": str(original_path),
    }
    lock["server_transfer"] = {
        key: provenance[key]
        for key in (
            "schema_version",
            "original_selection_file_sha256",
            "original_lock_sha256",
            "original_protocol_sha256",
            "amendment_sha256",
            "scientific_payload_sha256",
        )
    }
    lock["server_transfer"]["receipt_sha256"] = canonical_hash(provenance)
    lock["binding"] = {
        "source": canonical_hash(current_sources),
        "config": protocol_hash,
        "data": canonical_hash(inputs),
        "packet": canonical_hash(lock["packet_hashes"]),
        "selector": canonical_hash(lock["selected"]),
    }
    del lock["lock_sha256"]
    lock["lock_sha256"] = canonical_hash(lock)
    if _science_payload(lock) != _science_payload(original):
        raise AssertionError("server transfer changed frozen scientific decisions")
    return {
        "lock": lock,
        "receipt": provenance,
        "original_lock_bytes": original_bytes,
        "original_N3_bytes": preserved,
        "original_manifest_bytes": (original_path.parent / "RUN_MANIFEST.json").read_bytes(),
    }


def write_server_transfer(writer, payload):
    """Publish the prepared transfer only in a new, hash-bound N3_server stage."""
    lock, receipt = payload["lock"], payload["receipt"]
    if writer.out.name != "N3_server" or writer.resumed:
        raise ValueError("server transfer requires a distinct new N3_server writer")
    if not verify_selection_lock(lock) or writer.binding != lock["binding"]:
        raise ValueError("prepared lock or writer binding changed")
    if lock["server_transfer"]["receipt_sha256"] != canonical_hash(receipt):
        raise ValueError("prepared transfer receipt changed")
    original = json.loads(payload["original_lock_bytes"])
    if (
        hashlib.sha256(payload["original_lock_bytes"]).hexdigest()
        != receipt["original_selection_file_sha256"]
    ):
        raise ValueError("prepared original lock bytes changed")
    if canonical_hash(_science_payload(lock)) != receipt[
        "scientific_payload_sha256"
    ] or _science_payload(lock) != _science_payload(original):
        raise ValueError("prepared payload changed frozen science")
    if {k: v for k, v in original["data_gate"].items() if k != "fresh_seed_inventory_roots"} != {
        k: v for k, v in lock["data_gate"].items() if k != "fresh_seed_inventory_roots"
    }:
        raise ValueError("prepared payload changed scientific data audit")
    original_manifest = json.loads(payload["original_manifest_bytes"])
    manifest_path = str(Path(receipt["original_selection_path"]).parent / "RUN_MANIFEST.json")
    if (
        hashlib.sha256(payload["original_manifest_bytes"]).hexdigest()
        != receipt["original_N3_hashes"][manifest_path]
    ):
        raise ValueError("prepared original manifest bytes changed")
    if set(payload["original_N3_bytes"]) != set(original_manifest["outputs"]):
        raise ValueError("prepared original N3 evidence coverage changed")
    # Recheck source/evidence after preparation and before publishing any bytes.
    for field in _HASH_FIELDS:
        _current_sources(lock[field], field)
    for name, data in payload["original_N3_bytes"].items():
        source = str(Path(receipt["original_selection_path"]).parent / name)
        if hashlib.sha256(data).hexdigest() != receipt["original_N3_hashes"].get(source):
            raise ValueError("prepared original N3 bytes changed")
    writer.write_bytes("ORIGINAL_MODEL_SELECTION_LOCK.json", payload["original_lock_bytes"])
    writer.write_bytes("ORIGINAL_N3_RUN_MANIFEST.json", payload["original_manifest_bytes"])
    for name, data in payload["original_N3_bytes"].items():
        writer.write_bytes(Path("original_N3") / name, data)
    writer.write_json("SERVER_TRANSFER_RECEIPT.json", receipt)
    writer.write_json("MODEL_SELECTION_LOCK.json", lock)
