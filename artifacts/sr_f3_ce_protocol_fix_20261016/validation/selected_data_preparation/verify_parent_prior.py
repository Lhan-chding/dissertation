#!/usr/bin/env python3
"""Check fresh parent/prior receipts after the existing CPU audit/export CLIs.

This helper never calls audit_parents/export_prior, loads model weights, submits
work, edits a parent or rewrites an existing receipt. Print its JSON into a new
Slurm log or capture it as a new auxiliary receipt; preserve failures and partials.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from prepare_selected_data import authenticated_context, parse_args, require_cpu_allocation


def verify_parent_prior(args):
    context = authenticated_context(args)
    old, new, ablation = (context[key] for key in ("old", "new", "ablation"))
    from sr_f3_ce.ablation import verify_selection
    from sr_f3_ce.preflight import bounded_file, file_hash, read_json

    expected_parent = copy.deepcopy(read_json(old / "parents/PARENT_AUDIT.json"))
    expected_parent["package_manifest_sha256"] = file_hash(new / "plan/PACKAGE_SHA256.json")
    actual_parent = read_json(new / "parents/PARENT_AUDIT.json")
    if actual_parent != expected_parent:
        raise PermissionError("Unexpected fresh parent audit difference")
    parent_files = []
    for parent in actual_parent["parents"]:
        if parent.get("optimizer_inherited") is not False:
            raise PermissionError("Fresh parents cannot inherit an optimizer")
        for item in parent["copied_files"]:
            path = bounded_file(new, item["path"], allow_internal_symlink=False)
            if file_hash(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise PermissionError("Fresh parent copied file differs: " + item["path"])
            parent_files.append(dict(item))
    expected_prior = copy.deepcopy(read_json(old / "prior/RAW_EXPORT.json"))
    expected_prior["parent_audit_sha256"] = file_hash(new / "parents/PARENT_AUDIT.json")
    actual_prior = read_json(new / "prior/RAW_EXPORT.json")
    if actual_prior != expected_prior:
        raise PermissionError("Unexpected fresh historical export difference")
    identical = {}
    for relative in (
        "prior/HISTORICAL_IMAGES.json",
        "prior/LEGACY_IMAGE_COPY.json",
        "prior/MODEL_ENVIRONMENT_IDENTITY.json",
        "prior/SOURCE_AUDIT.json",
        "prior/BASE_ENVIRONMENT_AUDIT.json",
        "prior/raw/FILE_MANIFEST.json",
    ):
        current = file_hash(bounded_file(new, relative, allow_internal_symlink=False))
        if current != file_hash(bounded_file(old, relative, allow_internal_symlink=False)):
            raise PermissionError("Historical identity changed: " + relative)
        identical[relative] = current
    index = actual_prior["record_index"]
    if index["path"] != "prior/raw/RECORD_INDEX.jsonl":
        raise PermissionError("Historical raw index path differs")
    path = bounded_file(new, index["path"], allow_internal_symlink=False)
    if file_hash(path) != index["sha256"] or path.stat().st_size != index["bytes"]:
        raise PermissionError("Historical raw record index changed")
    final_cpu = require_cpu_allocation()
    if verify_selection(ablation) != context["selection"]:
        raise PermissionError("Protocol selection changed during verification")
    return {
        "status": "PARENT_AND_PRIOR_REBINDING_PASS",
        "protocol_revision": context["revision"],
        "root": str(new),
        "protocol_selection_sha256": file_hash(ablation / "PROTOCOL_SELECTION.json"),
        "parent_audit_sha256": file_hash(new / "parents/PARENT_AUDIT.json"),
        "prior_raw_export_sha256": file_hash(new / "prior/RAW_EXPORT.json"),
        "parent_audit_only_registered_difference": "package_manifest_sha256",
        "prior_export_only_registered_difference": "parent_audit_sha256",
        "parent_copied_files_rehashed": parent_files,
        "historical_identity_files_identical": identical,
        "record_index_verified": index,
        "historical_status_preserved": actual_prior["status"],
        "historical_counts_preserved": actual_prior["counts"],
        "historical_raw_payload_rehashed_by_this_verifier": False,
        "cpu_allocation_before": context["cpu_allocation"],
        "cpu_allocation_after": final_cpu,
        "helper_sha256": file_hash(Path(__file__).resolve()),
        "audit_parents_called": False,
        "export_prior_called": False,
        "model_calls": 0,
        "gpu_calls": 0,
        "scientific_training_authorized_by_this_receipt": False,
    }


def main(argv=None):
    result = verify_parent_prior(parse_args(argv, description=__doc__))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
