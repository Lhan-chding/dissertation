"""Immutable 360-answer packet derived only after verified confirmation support.

The frozen producer remains 6e5d363/SCHEMA_CONFIRM_V3. This separately bound
consumer creates no model outputs and grants no training permission. Only
blind/EXTRACTION.jsonl is given to extraction; world/query is released to the
judgment stage after extraction has been sealed. Private labels stay private.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from sr_f6.common import digest
from sr_f6.review_package import (
    _compare,
    _directory,
    _fsync_directory,
    _linked_stage,
    _payload,
    _publish,
    _require,
    _require_commit,
)

from . import sampling, source

SCHEMA = "srf6-followup-review-package-v1"
DESTINATION = "REVIEW/PACKAGE"
SELECTION = "RECOMPUTED_FROM_VERIFIED_FULL_CONFIRMATION"


def _ref(relative, payload):
    return dict(path=relative, sha256=hashlib.sha256(payload).hexdigest())


def _derive(root):
    context = source.require_support(root, purpose="review")
    root = Path(context["root"])
    evidence = context["evidence"]
    producer = evidence["input_producer"]
    records, raw = context["records"], context["raw_by_row"]
    _require(
        evidence.get("status") == "VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED"
        and producer.get("phase") == sampling.PHASE
        and producer.get("revision") == source.PRODUCER_REVISION
        and producer.get("deployment") == str(source.PRODUCER_DEPLOYMENT)
        and len(records) == len(raw) == 9216
        and evidence.get("source_rows") == evidence.get("actual_recorded_calls") == 9216,
        "Complete original 6e5 confirmation is required",
    )
    support = context["support_binding"]
    _require(
        support.get("status") == "VERIFIED_SUPPORT_PASS_REVIEW_PENDING"
        and support["consumer"] == context["consumer"]
        and support["records_sha256"] == digest(records)
        and support["raw_records_sha256"] == digest(raw),
        "Reconstructed confirmation/support identities differ",
    )
    consumer = context["consumer"]
    source_identity = dict(
        consumer_registration_sha256=consumer["registration_sha256"],
        consumer_source_manifest_sha256=consumer["source_manifest_sha256"],
        producer_registration_sha256=evidence["registration_sha256"],
        support_binding_sha256=context["support_binding_ref"]["sha256"],
        records_sha256=support["records_sha256"],
        raw_records_sha256=support["raw_records_sha256"],
    )
    ledger, blind = sampling.build_review(records, raw_by_row=raw, source_identity=source_identity)
    _require(
        ledger["sampling_status"] == "COMPLETE_360" and len(blind) == 360,
        "Insufficient unique answers; retain the shortfall, never redraw or relax the cap",
    )
    tasks = context["producer_context"]["tasks"]
    _require(len(tasks) == 384, "All 384 original confirmation tasks required")
    expected = {
        f"{parent}/{qid}/{draw}"
        for parent in ("b0", "b1", "b2")
        for qid in tasks
        for draw in range(8)
    }
    _require(set(raw) == expected, "Exact original three-parent G8 matrix required")
    judgment = []
    for row in ledger["selected"]:
        task = tasks[row["qid"]]
        _require(
            task["split"] == sampling.SPLIT
            and all(task[key] == row[key] for key in ("root_id", "family", "factor")),
            "Original task lineage differs from selected answer",
        )
        judgment.append(
            dict(anonymous_id=row["anonymous_id"], world=task["world"], query=task["query"])
        )
    # Support already verified all external leaves. Adopt those identities, never
    # rehash model weights or replay the full raw matrix a second time here.
    external = {}
    for ref in support["dependencies"]:
        pin = {key: ref[key] for key in ("path", "sha256")}
        _require(external.setdefault(pin["path"], pin) == pin, "Conflicting upstream pin")
    registration = context["registration"]
    inputs = [
        dict(path="REGISTRATION.json", sha256=consumer["registration_sha256"]),
        dict(path="SOURCE_MANIFEST.json", sha256=consumer["source_manifest_sha256"]),
        dict(path="SPEC.json", sha256=registration["spec"]["sha256"]),
        dict(path="SUPPORT_BINDING.json", sha256=context["support_binding_ref"]["sha256"]),
    ]
    validation = {k: v for k, v in evidence.items() if k not in ("records", "raw_by_row")}
    synthetic = any(row.get("execution_origin") == "SYNTHETIC_TEST" for row in raw.values())
    files = {
        "private/selection.json": _payload(ledger),
        "blind/EXTRACTION.jsonl": _payload(blind, lines=True),
        "judgment/TASKS.jsonl": _payload(judgment, lines=True),
        "private/SUPPORT.json": _payload(support),
        "private/CONFIRMATION_VALIDATION.json": _payload(validation),
    }
    refs = {name: _ref(f"{DESTINATION}/{name}", value) for name, value in files.items()}
    package = dict(
        schema=SCHEMA,
        consumer=consumer,
        input_producer=producer,
        phase=sampling.PHASE,
        split=sampling.SPLIT,
        execution_origin="SYNTHETIC_TEST" if synthetic else "ACTUAL_CONFIRMATION",
        status="BLIND_PACKAGE_ONLY",
        review_status="REVIEW_NOT_RUN",
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        model_calls=0,
        optimizer_updates=0,
        ledger=refs["private/selection.json"],
        blind=refs["blind/EXTRACTION.jsonl"],
        judgment_tasks=refs["judgment/TASKS.jsonl"],
        support=refs["private/SUPPORT.json"],
        confirmation_validation=refs["private/CONFIRMATION_VALIDATION.json"],
        ledger_source_identity=source_identity,
        selection_provenance=SELECTION,
        selection_inputs_sha256=ledger["selection_inputs_sha256"],
        blind_items_sha256=ledger["blind_items_sha256"],
        inputs=inputs + [refs[name] for name in sorted(refs)],
        external_inputs=[external[path] for path in sorted(external)],
    )
    files["BINDING.json"] = _payload(package)
    return dict(
        context=context, package=package, files=files, ledger=ledger, blind=blind, tasks=tasks
    )


def _store(root, destination, files):
    """Exclusive exact publication, including the existing Ceph-safe fallback."""
    target = _directory(root, destination)
    parent = target.parent
    stage = _directory(
        root, str(target.relative_to(root).parent / ("." + target.name + ".staging"))
    )
    if target.exists():
        _compare(target, files)
        if stage.exists():
            _linked_stage(stage, target, files)
        return
    if stage.exists():
        _compare(stage, files)
    else:
        stage.mkdir(mode=0o700)
        for relative, payload in files.items():
            path = stage / relative
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for path in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
            _fsync_directory(path)
        _fsync_directory(stage)
        _compare(stage, files)
    _publish(stage, target)
    _fsync_directory(parent)


def _verify_publication(root, destination, files):
    target = _directory(root, destination)
    _compare(target, files)
    stage = _directory(
        root, str(target.relative_to(root).parent / ("." + target.name + ".staging"))
    )
    if stage.exists():
        _linked_stage(stage, target, files)


def write_package(root):
    """Publish only the fixed package. Retries must match every immutable byte."""
    bundle = _derive(root)  # No directory is created before actual support verifies.
    root = bundle["context"]["root"]
    parent = _directory(root, "REVIEW")
    parent.mkdir(mode=0o700, exist_ok=True)
    _fsync_directory(Path(root))
    fd = os.open(parent / ".package.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _store(root, DESTINATION, bundle["files"])
        return dict(
            status="BLIND_PACKAGE_ONLY",
            review_status="REVIEW_NOT_RUN",
            package=_ref(f"{DESTINATION}/BINDING.json", bundle["files"]["BINDING.json"]),
            training_allowed=False,
            gate_status="NOT_SIGNED",
            consumer=bundle["package"]["consumer"],
            input_producer=bundle["package"]["input_producer"],
        )
    finally:
        os.close(fd)


def verify_package(root):
    """Recompute selection from support-validated original rows once per call."""
    root = source._root(root)
    _require_commit(_directory(root, DESTINATION))
    bundle = _derive(root)
    _verify_publication(root, DESTINATION, bundle["files"])
    reference = _ref(f"{DESTINATION}/BINDING.json", bundle["files"]["BINDING.json"])
    return dict(
        status="VERIFIED_REVIEW_PACKAGE_NOT_GATE_SIGNED",
        training_allowed=False,
        gate_status="NOT_SIGNED",
        consumer=bundle["package"]["consumer"],
        input_producer=bundle["package"]["input_producer"],
        package_ref=reference,
        inputs=[*bundle["package"]["inputs"], reference],
        external_inputs=bundle["package"]["external_inputs"],
        artifacts={key: bundle[key] for key in ("package", "ledger", "blind", "tasks")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("write", "verify"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = write_package(args.root) if args.operation == "write" else verify_package(args.root)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
