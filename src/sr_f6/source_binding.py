"""Read-only content binding of the inherited F5 production sources.

This emits PARTIAL until loaded model/processor evidence is supplied separately.
It never upgrades packages, downloads a model, signs a gate, or edits F5 inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import stat
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
F5_CODE = PROJECT / "sr_f5_recipe_v2_47d5e85"
F5_RUN = PROJECT / "sr_f5_semantic_j_recipe_v2_20261026"
SOURCE_FILES = (
    "src/sr_f5/runtime.py",
    "src/sr_f5/measurement.py",
    "src/sr_f5/training.py",
    "src/sr_f5/pilot.py",
    "src/sr_f4/runtime.py",
    "src/sr_f4/task_contract.py",
    "src/sr_f3_ce/runtime.py",
    "src/sr_f3_ce/training.py",
    "src/sr_f3_ce/contract.py",
    "src/sr_f12/runtime.py",
    "src/sr_f12/training.py",
    "src/mm_dev/runtime.py",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def stat_cache_key(value):
    """Metadata is a revalidation hint, not a scientific content identity.

    Device IDs are excluded because shared mounts expose different device IDs
    on different nodes. Inode/size/mtime/ctime changes trigger revalidation.
    """
    return {
        name: getattr(value, "st_" + name)
        for name in ("ino", "size", "mtime_ns", "ctime_ns", "mode")
    }


def content_inventory(files):
    return [
        {key: value for key, value in item.items() if key != "stat_cache_key"} for item in files
    ]


_VERIFIED_STATS = {}


def verify_bound_files(files, boundary=PROJECT):
    """Reuse frozen hashes while paths/stats match; rehash only changed files.

    A verified changed-stat/same-content observation is cached in this process.
    The immutable original binding is never overwritten by the cache.
    """
    boundary = Path(boundary).resolve(strict=True)
    if not isinstance(files, list) or not files:
        raise ValueError("Nonempty bound source file inventory required")
    seen = set()
    results = []
    for bound in files:
        path = Path(bound["path"])
        resolved = path.resolve(strict=True)
        if (
            not path.is_absolute()
            or resolved != Path(bound["resolved_path"])
            or not resolved.is_relative_to(boundary)
            or path in seen
        ):
            raise PermissionError("Bound path changed, duplicated, or escaped the project")
        seen.add(path)
        before = resolved.stat()
        current = stat_cache_key(before)
        if not stat.S_ISREG(before.st_mode) or before.st_size != bound["bytes"]:
            raise PermissionError("Bound file type or size changed: " + str(path))
        digest = bound["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid bound content SHA256")
        cache_id = (str(resolved), digest)
        cached = _VERIFIED_STATS.get(cache_id, bound.get("stat_cache_key"))
        rehashed = current != cached
        if rehashed and file_sha256(resolved) != digest:
            raise PermissionError("Bound file content changed: " + str(path))
        after = stat_cache_key(resolved.stat())
        if after != current or path.resolve(strict=True) != resolved:
            raise PermissionError("Bound file changed during verification: " + str(path))
        _VERIFIED_STATS[cache_id] = current
        results.append(
            dict(
                path=str(path),
                sha256=digest,
                verification="CONTENT_REHASHED" if rehashed else "UNCHANGED_STAT_CACHE",
                stat_cache_key=current,
            )
        )
    return dict(
        status="PASS",
        files=results,
        content_rehashed=sum(r["verification"] == "CONTENT_REHASHED" for r in results),
        unchanged_stat_reused=sum(r["verification"] == "UNCHANGED_STAT_CACHE" for r in results),
        cache_scope="metadata unchanged since content binding or this process verification",
    )


def inventory_file(path, boundary=PROJECT):
    path, boundary = Path(path), Path(boundary).resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(boundary) or not resolved.is_file():
        raise PermissionError("Source target is outside the canonical project or not a file")
    before = resolved.stat()
    result = dict(
        path=str(path),
        resolved_path=str(resolved),
        bytes=before.st_size,
        sha256=file_sha256(resolved),
        symlink=path.is_symlink(),
        stat_cache_key=stat_cache_key(before),
    )
    after = resolved.stat()
    if stat_cache_key(before) != stat_cache_key(after) or path.resolve(strict=True) != resolved:
        raise RuntimeError("Source changed while being hashed: " + str(path))
    return result


def bind(
    config, *, project=PROJECT, source=F5_CODE, previous=F5_RUN, config_path=None, deployment=None
):
    project, source, previous = Path(project), Path(source), Path(previous)
    if project.resolve(strict=True) != project:
        raise PermissionError("Project root must be canonical")
    for location in (source, previous):
        if location.resolve(strict=True) != location or not location.is_relative_to(project):
            raise PermissionError("F5 source must be a canonical owned-project descendant")
    old_config = json.loads((previous / "CONFIG.json").read_text())
    for key in ("model_path", "snapshot", "parents"):
        if old_config["model"][key] != config["model"][key]:
            raise PermissionError("Bound F5 model identity differs: " + key)
    source_files = [inventory_file(source / name, project) for name in SOURCE_FILES]
    deployment = Path(deployment) if deployment is not None else Path(__file__).resolve().parents[2]
    inherited = []
    for name, bound in zip(SOURCE_FILES, source_files, strict=True):
        local = deployment / name
        digest = file_sha256(local)
        if digest != bound["sha256"]:
            raise PermissionError("Inherited production module drifted from bound F5: " + name)
        inherited.append(dict(path=str(local), relative=name, sha256=digest))
    parents = {}
    for parent, directory in config["model"]["parents"].items():
        path = Path(directory)
        files = [
            inventory_file(path / name, project)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ]
        metadata = json.loads((path / "adapter_config.json").read_text())
        parents[parent] = dict(
            path=directory,
            files=files,
            lora_config=metadata,
            content_sha256=object_sha256(content_inventory(files)),
        )
    model = Path(config["model"]["model_path"])
    if model.name != config["model"]["snapshot"]:
        raise PermissionError("Snapshot directory differs from registration")
    # The HF snapshot may link to blobs; each target is independently bounded
    # to this project before reading. Weights receive content hashes too.
    model_files = [inventory_file(p, project) for p in sorted(model.rglob("*")) if p.is_file()]
    names = {Path(x["path"]).name for x in model_files}
    if not {"config.json", "tokenizer_config.json"} <= names or not any(
        name.endswith(".safetensors") for name in names
    ):
        raise FileNotFoundError("Incomplete registered model/tokenizer snapshot")
    retention = {}
    for name in (
        "LEGACY_TASKS.jsonl",
        "CHARTQA_TASKS.jsonl",
        "LEGACY_SOURCE_ROUTES.jsonl",
        "CHARTQA_SOURCE_ROUTES.jsonl",
        "FROZEN.json",
    ):
        retention[name] = inventory_file(previous / "RETENTION" / name, project)
    for name, count in (("LEGACY", 128), ("CHARTQA", 250)):
        rows = [
            json.loads(line)
            for line in (previous / "RETENTION" / f"{name}_TASKS.jsonl").read_text().splitlines()
            if line
        ]
        if len(rows) != count or len({row["qid"] for row in rows}) != count:
            raise ValueError("Frozen retention ID matrix differs: " + name)
    source_root = Path(json.loads((previous / "RETENTION/FROZEN.json").read_text())["source_root"])
    retention_images = {}
    for name, expected_count in (("LEGACY", 128), ("CHARTQA", 250)):
        tasks = [
            json.loads(line)
            for line in (previous / "RETENTION" / f"{name}_TASKS.jsonl").read_text().splitlines()
            if line
        ]
        routes = [
            json.loads(line)
            for line in (previous / "RETENTION" / f"{name}_SOURCE_ROUTES.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        if len(routes) != expected_count or {r["qid"] for r in routes} != {t["qid"] for t in tasks}:
            raise ValueError("Retention image/qid correspondence differs: " + name)
        image_records = []
        for route in routes:
            relative = Path(route["image_file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise PermissionError("Retention image route must be a safe source-relative path")
            image = inventory_file(source_root / relative, project)
            if image["sha256"] != route["image_sha256"]:
                raise PermissionError("Frozen retention image content changed: " + route["qid"])
            image_records.append(dict(image, qid=route["qid"]))
        retention_images[name] = image_records
    versions = {}
    for name in ("torch", "transformers", "peft", "Pillow", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        status="PARTIAL",
        phase="SOURCE_BINDING",
        execution_origin="SERVER_ACTUAL",
        at=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        python=platform.python_version(),
        versions=versions,
        config_canonical_sha256=object_sha256(config),
        config_file_sha256=file_sha256(config_path) if config_path is not None else None,
        previous_source=str(source),
        previous_run=str(previous),
        previous_config=inventory_file(previous / "CONFIG.json", project),
        source_files=source_files,
        inherited_deployment_files=inherited,
        source_content_sha256=object_sha256(inherited),
        parents=parents,
        model_files=model_files,
        model_content_sha256=object_sha256(content_inventory(model_files)),
        retention=retention,
        retention_images=retention_images,
        optimizer_inheritance=False,
        model_calls=0,
        optimizer_updates=0,
        remaining=[
            "actual loaded trainable-parameter identity",
            "actual tokenizer and processor regression",
        ],
        claim="content identity only; not a SOURCE_BINDING gate PASS",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError("Preserve the existing binding; use a fresh receipt path")
    result = bind(config, config_path=args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "model_calls": 0}))


if __name__ == "__main__":
    main()
