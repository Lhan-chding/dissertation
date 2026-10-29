"""Prepare the separately registered review in one bounded, zero-GPU process.

The original registration, support and package APIs remain authoritative. Their
private process caches avoid reconstructing unchanged confirmation evidence in
separate processes. Preparation receipts grant no review or model permission.
Every stage is write-once; interrupted or failed preparation is never retried.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from . import review_package, source

OPERATIONS = source.PROJECT / "sr_f6_operations_20261027" / "CONFIRM_FOLLOWUP_V2_EXECUTION_20261029"
DIRECTORY = "REPAIR_PREPARE"
SCHEMA = "srf6-followup-repair-prepare-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sync(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record(directory, name, value):
    source._write(directory / name, value)
    _sync(directory)


def _permissions():
    return dict(
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
    )


def prepare(root, revision):
    """Register, bind complete support, and publish the fixed package exactly once."""
    root = source._root(root, exists=False)
    source._require(
        not root.exists() and not root.is_symlink(),
        "Preserve existing or partial preparation root; no automatic retry",
    )
    deployment = source._deployment(revision)
    operations = source._path(OPERATIONS)
    source._require(operations.is_dir(), "Registered preparation operations directory required")
    directory = source._path(operations / DIRECTORY, exists=False)
    source._require(
        not directory.exists() and not directory.is_symlink(),
        "Prior or unknown preparation preserved; no automatic retry",
    )
    directory.mkdir(mode=0o700, exist_ok=False)
    _sync(operations)
    identity = dict(
        schema=SCHEMA,
        root=str(root),
        deployment=str(deployment),
        revision=revision,
        **_permissions(),
    )
    _record(directory, "ATTEMPT.json", dict(identity, status="STARTED", started_at=_now()))
    stage = "REGISTER"
    try:
        _record(directory, stage + ".ATTEMPT.json", dict(identity, stage=stage, started_at=_now()))
        registered = source.register(root, revision)
        consumer = registered["consumer"]
        _record(
            directory,
            stage + ".RESULT.json",
            dict(
                identity,
                stage=stage,
                completed_at=_now(),
                status=registered["registration"]["status"],
                consumer=consumer,
                registration=dict(
                    path=str(root / "REGISTRATION.json"),
                    sha256=consumer["registration_sha256"],
                ),
                source_manifest=registered["registration"]["source_manifest"],
                spec=registered["registration"]["spec"],
            ),
        )
        stage = "BIND_SUPPORT"
        _record(directory, stage + ".ATTEMPT.json", dict(identity, stage=stage, started_at=_now()))
        supported = source.bind_support(root)
        source._require(supported["consumer"] == consumer, "Consumer changed during preparation")
        _record(
            directory,
            stage + ".RESULT.json",
            dict(
                identity,
                stage=stage,
                completed_at=_now(),
                status=supported["support_binding"]["status"],
                consumer=consumer,
                support_binding=supported["support_binding_ref"],
            ),
        )
        stage = "REVIEW_PACKAGE"
        _record(directory, stage + ".ATTEMPT.json", dict(identity, stage=stage, started_at=_now()))
        package = review_package.write_package(root)
        source._require(
            package["consumer"] == consumer
            and package["status"] == "BLIND_PACKAGE_ONLY"
            and package["review_status"] == "REVIEW_NOT_RUN"
            and package["training_allowed"] is False
            and package["gate_status"] == "NOT_SIGNED",
            "Preparation may only produce the unreviewed fixed package",
        )
        _record(
            directory,
            stage + ".RESULT.json",
            dict(identity, stage=stage, completed_at=_now(), result=package),
        )
        result = dict(
            identity,
            status="PREPARED_NOT_AUTHORIZED",
            review_status="REVIEW_NOT_RUN",
            completed_at=_now(),
            consumer=consumer,
            input_producer=package["input_producer"],
            package=package["package"],
            support_binding=supported["support_binding_ref"],
            preparation_directory=str(directory),
        )
        _record(directory, "RESULT.json", result)
        return result
    except Exception as exc:
        _record(
            directory,
            "FAILURE.json",
            dict(
                identity,
                status="STOP_PRESERVE",
                stage=stage,
                failed_at=_now(),
                error_type=type(exc).__name__,
                error=str(exc),
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.revision), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
