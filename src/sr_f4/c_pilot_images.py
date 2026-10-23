"""Local image routing and bounded recovery of the F-prime preparation defect."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from .common import now, read, rows, write, write_rows

RECOVERY = "RECOVERY_IMAGE_ROUTING"


def local_image(root, source_root, image):
    """Copy the registered pixels inside DATA, retaining every non-path field."""
    source_data = (Path(source_root) / "DATA").resolve()
    path = Path(image["image_file"])
    source = (source_data / path).resolve()
    if not source.is_relative_to(source_data) or not source.is_file():
        raise PermissionError("Image source is outside registered source DATA or missing")
    relative = Path("retained_images") / source.relative_to(source_data)
    data_root = (Path(root) / "DATA").resolve()
    destination = data_root / relative
    if not destination.resolve().is_relative_to(data_root) or destination.is_symlink():
        raise PermissionError("Local image destination escapes pilot DATA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".copy-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise OSError("Image copy size mismatch")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(image, image_file=relative.as_posix())


def repair_routes(root, revision):
    """Route-only amendment; never change FROZEN/checkpoint/scientific identity."""
    from .c_pilot import require_frozen
    from .pool import _locked

    root = Path(root)
    directory = root / RECOVERY
    receipt_path = directory / "RECEIPT.json"
    manifest = root / "DATA/IMAGE_MANIFEST.jsonl"
    frozen_inputs = root / "FROZEN_INPUTS.json"
    with _locked(root) as state:
        if receipt_path.exists() and read(receipt_path)["status"] == "APPLIED":
            require_frozen(root)
            return read(receipt_path)
        # A claim can precede this lock. Wait for every already claimed child
        # to finish reading its manifests before replacing the two input files.
        for key, row in state["tasks"].items():
            if row["status"] == "RUNNING":
                identity = root / "runtime_identity" / f"{key}.json"
                started = datetime.fromisoformat(row["owner"]["started_at"]).timestamp()
                if not identity.exists() or identity.stat().st_mtime < started:
                    raise RuntimeError(f"Child still initializing; retry routing repair: {key}")
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt["status"] != "PREPARED" or receipt["revision"] != revision:
                raise ValueError("Unrecognized routing recovery")
            original = read(directory / "FROZEN_INPUTS.before.json")
            amended = read(directory / "FROZEN_INPUTS.after.json")
        else:
            require_frozen(root)
            original = read(frozen_inputs)
            source_root = Path(read(root / "CONFIG.json")["source_root"])
            tasks = {t["qid"]: t for t in original["tasks"]}
            images, changes = [], []
            for image in original["images"]:
                updated = image
                if Path(image["image_file"]).is_absolute():
                    if tasks[image["qid"]].get("pool") not in ("TEST_ID", "CHARTQA"):
                        raise ValueError("Unexpected external image in training/new test")
                    updated = local_image(root, source_root, image)
                    changes.append(dict(qid=image["qid"], before=image, after=updated))
                images.append(updated)
            if not changes:
                raise ValueError("No legacy absolute image routes to repair")
            amended = dict(original, images=images)
            directory.mkdir(parents=True, exist_ok=True)
            write(directory / "FROZEN_INPUTS.before.json", original)
            write(directory / "FROZEN_INPUTS.after.json", amended)
            write_rows(directory / "IMAGE_MANIFEST.before.jsonl", original["images"])
            receipt = dict(
                status="PREPARED",
                at=now(),
                revision=revision,
                changes=changes,
                eligible_attempts={
                    key: len(row["attempts"])
                    for key, row in state["tasks"].items()
                    if row["spec"]["kind"] == "c_pilot_eval"
                    and row["status"] in ("FAILED", "RUNNING")
                },
                unchanged="tasks, pixels, image identities, prompts, seeds, budgets, "
                "FROZEN.json, CONFIG.json, checkpoint identity, all generated outputs",
            )
            write(receipt_path, receipt)
        # Resume a partially written pair only if both still match this amendment.
        if read(frozen_inputs) not in (original, amended) or list(rows(manifest)) not in (
            original["images"],
            amended["images"],
        ):
            raise ValueError("Image inputs changed outside recorded routing recovery")
        write_rows(manifest, amended["images"])
        write(frozen_inputs, amended)
        require_frozen(root)
        receipt.update(status="APPLIED", applied_at=now())
        write(receipt_path, receipt)
        return receipt


def retry_route_failures(root):
    """Retry only proven pre-amendment route failures; retain all prior attempts."""
    from .c_pilot import require_frozen
    from .pool import _locked, _save, _task_lease

    root = Path(root)
    receipt = read(root / RECOVERY / "RECEIPT.json")
    if receipt["status"] != "APPLIED":
        raise ValueError("Image routing repair must be applied first")
    require_frozen(root)
    events = {}
    for event in rows(root / "technical_events.jsonl"):
        if "error_type" in event:
            events[event["task_id"]] = event
    retried = []
    with _locked(root) as state:
        for key, row in state["tasks"].items():
            if row["status"] != "FAILED" or row["spec"]["kind"] != "c_pilot_eval":
                continue
            if not row["attempts"] or (root / "tasks" / key / "COMPLETE.json").exists():
                continue
            attempt, event = row["attempts"][-1], events.get(key, {})
            started = attempt["owner"]["started_at"]
            if not (
                attempt["status"] == "FAILED"
                and len(row["attempts"]) == receipt["eligible_attempts"].get(key)
                and started < receipt["applied_at"]
                and started <= event.get("at", "") <= attempt["finished_at"]
                and event.get("error_type") == "PermissionError"
                and event.get("error") == "Image path escapes run root"
            ):
                continue
            lease = _task_lease(root, key)
            if lease is None:
                continue
            try:
                row.setdefault("recoveries", []).append(
                    dict(
                        at=now(),
                        recovery=RECOVERY,
                        failed_attempt=len(row["attempts"]),
                        evidence=event,
                        reason=row.get("reason"),
                    )
                )
                row.update(status="PENDING", reason="Resume after image route repair")
                row.pop("owner", None)
                retried.append(key)
            finally:
                lease.close()
        if retried:
            _save(root, state)
    return retried


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    receipt = repair_routes(args.root, args.revision)
    print(
        dict(
            status=receipt["status"],
            routes=len(receipt["changes"]),
            retried=retry_route_failures(args.root),
        )
    )


if __name__ == "__main__":
    main()
