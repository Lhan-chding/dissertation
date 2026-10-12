#!/usr/bin/env python3
"""Copy and verify immutable F1.2 inputs into a fresh F2 run on a CPU allocation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f1.contract import file_hash
from sr_f1.data import load_inputs, load_schedule, load_tasks
from sr_f2.protocol import PARENT_ROOT, SEEDS, build_config
from sr_f12.protocol import immutable_json, object_hash


def contained(root, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise PermissionError("Unsafe relative input path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError("Input path escapes run root")
    return path


def model_file(model, name):
    """Permit checksum-verified Hugging Face snapshot links only within its cache."""
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise PermissionError("Unsafe model file name")
    path = model / name
    canonical = path.resolve(strict=True)
    inside_snapshot = canonical.is_relative_to(model.resolve(strict=True))
    inside_blob_cache = model.parent.name == "snapshots" and canonical.is_relative_to(
        (model.parent.parent / "blobs").resolve(strict=True)
    )
    if not canonical.is_file() or not (inside_snapshot or inside_blob_cache):
        raise PermissionError("Model link escapes its frozen snapshot/cache")
    return path


def prepare(root, prior_root):
    from sr_f2.diagnostics import build_diagnostic_manifest

    root, prior = Path(root).resolve(), Path(prior_root).resolve(strict=True)
    if root == prior or root.is_relative_to(prior) or prior.is_relative_to(root):
        raise PermissionError("Separate sibling run roots required")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "REGISTRATION.json").exists():
        raise PermissionError("Prepared inputs are frozen; do not prepare again")
    source = json.loads((prior / "INPUT_IDENTITY.json").read_text())
    if source.get("status") != "PASS":
        raise PermissionError("Parent input identity did not pass")
    copied = []
    for item in source["files"]:
        original, target = contained(prior, item["path"]), contained(root, item["path"])
        if file_hash(original) != item["sha256"] or original.stat().st_size != item["bytes"]:
            raise PermissionError("Parent input changed: " + item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target) != item["sha256"]:
                raise PermissionError("Preexisting F2 input differs: " + item["path"])
        else:
            # New root only. Never hard-link mutable files to the prior experiment.
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        if file_hash(target) != item["sha256"]:
            raise PermissionError("Copied input checksum differs")
        copied.append(item)
    # External verification receipts are not all listed by INPUT_IDENTITY.
    for original in sorted((prior / "external/chartqa").glob("*.json*")):
        relative = str(original.relative_to(prior))
        target = contained(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if file_hash(target) != file_hash(original):
            raise PermissionError("ChartQA receipt changed")
    model = Path(source["model_path"]).resolve(strict=True)
    for item in source["model_files"]:
        path = model_file(model, item["name"])
        if file_hash(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise PermissionError("Shared base model identity changed")
    inputs, tasks = load_inputs(root), load_tasks(root)
    schedule_hashes = {str(seed): object_hash(load_schedule(root, seed)) for seed in SEEDS}
    manifest = build_diagnostic_manifest(tasks, inputs)
    immutable_json(root / "DIAGNOSTIC_MANIFEST.json", manifest)
    receipt = dict(
        status="PASS",
        prior_root=str(prior),
        root=str(root),
        files=copied,
        prior_identity_sha256=file_hash(prior / "INPUT_IDENTITY.json"),
        model_path=str(model),
        model_files=source["model_files"],
        model_weights_hash=source["model_weights_hash"],
        schedule_hashes=schedule_hashes,
        diagnostic_manifest_sha256=object_hash(manifest),
        copied_files=len(copied),
        model_calls=0,
    )
    immutable_json(root / "INPUT_IDENTITY.json", receipt)
    plan = build_config()
    immutable_json(root / "config/SR_F2.json", plan)
    return dict(
        status="PASS",
        copied_files=len(copied),
        model_calls=0,
        diagnostic_manifest_sha256=object_hash(manifest),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, default=Path(PARENT_ROOT))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise PermissionError("Input/model hashing must use an allocated CPU job")
    print(json.dumps(prepare(args.root, args.prior_root), sort_keys=True))


if __name__ == "__main__":
    main()
