"""Actual server CPU preparation, without model sampling or automatic gate promotion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import digest, file_hash, now, read, write_once
from .contract import AMENDMENTS, ROOT, validate_config


def source_manifest(deployment):
    deployment = Path(deployment).resolve(strict=True)
    paths = sorted(
        set((deployment / "src").rglob("*.py"))
        | set((deployment / "tests/sr_f6").rglob("*.py"))
        | set((deployment / "scripts/sr_f6").glob("*.py"))
        | set((deployment / "scripts/sr_f6").glob("*.sbatch"))
        | set((deployment / "scripts/sr_f6").glob("*.sh"))
    )
    return {
        "files": [
            {
                "path": str(p.relative_to(deployment)),
                "bytes": p.stat().st_size,
                "sha256": file_hash(p),
            }
            for p in paths
        ],
        "scope": "all executable source dependencies and SR-F6 entrypoints",
    }


def historical_audit(root, source_binding):
    """Check explicit historical manifests; no filesystem-wide discovery."""
    from srf6_ref.core import fingerprint

    from .census import rows

    previous = Path(source_binding["previous_run"])
    project = previous.parent
    sources = [
        previous / "DATA/TASKS_PRIVATE.jsonl",
        project / "sr_f4_c_local_pilot_20261024/DATA/TASKS.jsonl",
        project / "sr_f4_consolidated_20261022/DATA/TASKS.jsonl",
    ]
    historical = {}
    source_receipts = []
    for path in sources:
        if not path.resolve(strict=True).is_relative_to(project):
            raise PermissionError("Historical source outside project")
        records = rows(path)
        count = 0
        for record in records:
            world = record.get("world")
            if world is None:
                if record.get("pool") != "CHARTQA":
                    raise ValueError("Unclassified historical row without world")
                continue
            historical.setdefault(fingerprint(world), {"path": str(path), "qid": record.get("qid")})
            count += 1
        if not count:
            raise ValueError("Empty historical world source")
        source_receipts.append({"path": str(path), "sha256": file_hash(path), "world_rows": count})
    checked = 0
    collisions = []
    for path in sorted((root / "DATA").glob("*/TASKS_PRIVATE.jsonl")):
        for record in rows(path):
            checked += 1
            prior = historical.get(fingerprint(record["world"]))
            if prior:
                collisions.append({"qid": record["qid"], "source": prior})
    report = {
        "status": "FAIL" if collisions else "PASS",
        "sources": source_receipts,
        "checked_question_rows": checked,
        "collisions": collisions,
        "model_calls": 0,
        "comparison": "rename/order invariant Alpha/Beta fact multiset",
    }
    write_once(root / "PREFLIGHT/HISTORICAL_WORLD_AUDIT.json", report)
    if collisions:
        raise RuntimeError("Historical world overlap; retain artifacts and stop")
    return report


def initialize(root, config_file, revision, *, resume_source_failure=False):
    """Only a named pre-data source failure can be explicitly resumed."""
    root, config_file = Path(root), Path(config_file)
    if resume_source_failure:
        allowed = {
            "slurm",
            "CONFIG.json",
            "IMPLEMENTATION_REGISTRATION.json",
            "PREPARATION_STARTED.json",
            "SOURCE_BINDING_CACHE.json",
            "SOURCE_BINDING_CACHE.json.lock",
        }
        if any(
            p.name not in allowed and not p.name.startswith("PREPARATION_RECOVERY_")
            for p in root.iterdir()
        ):
            raise PermissionError(
                "Recovery only supports failed source binding before data/model work"
            )
        if (root / "CONFIG.json").read_bytes() != config_file.read_bytes():
            raise PermissionError("Recovery configuration changed")
        if read(root / "IMPLEMENTATION_REGISTRATION.json") != AMENDMENTS:
            raise PermissionError("Recovery implementation contract changed")
        previous = read(root / "PREPARATION_STARTED.json")
        if previous.get("model_calls") != 0 or previous.get("optimizer_updates") != 0:
            raise PermissionError("Recovery cannot claim pre-model status")
        write_once(
            root / f"PREPARATION_RECOVERY_{revision[:12]}.json",
            {
                "at": now(),
                "previous_started_record": previous,
                "new_revision": revision,
                "reason": (
                    "explicit recovery of failed source binding; no data or model outputs exist"
                ),
            },
        )
    else:
        if root.exists() and any(p.name != "slurm" for p in root.iterdir()):
            raise FileExistsError("Existing preparation must be diagnosed; no silent rebuild")
        root.mkdir(parents=True, exist_ok=True)
        with (root / "CONFIG.json").open("xb") as stream:
            stream.write(config_file.read_bytes())
        write_once(root / "IMPLEMENTATION_REGISTRATION.json", AMENDMENTS)
        write_once(
            root / "PREPARATION_STARTED.json",
            {
                "at": now(),
                "revision": revision,
                "status": "STARTED",
                "model_calls": 0,
                "optimizer_updates": 0,
            },
        )


def prepare(root, config_file, revision, *, resume_source_failure=False):
    from .census import freeze_spec
    from .data import build
    from .source_binding import bind

    root, config_file = Path(root), Path(config_file)
    config = read(config_file)
    validate_config(config)
    if root != ROOT or root.resolve() != ROOT:
        raise PermissionError("Only the canonical registered SR-F6 root is writable")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Immutable full Git revision required")
    initialize(root, config_file, revision, resume_source_failure=resume_source_failure)
    binding = bind(config, config_path=config_file, cache_path=root / "SOURCE_BINDING_CACHE.json")
    write_once(root / "SOURCE_BINDING.json", binding)
    build(root, config)
    historical_audit(root, binding)
    deployment = Path(__file__).resolve().parents[2]
    write_once(root / "SOURCE_MANIFEST.json", source_manifest(deployment))
    identity = {
        "config_sha256": file_hash(root / "CONFIG.json"),
        "implementation_digest": digest(AMENDMENTS),
        "revision": revision,
        "source_manifest": {
            "path": "SOURCE_MANIFEST.json",
            "sha256": file_hash(root / "SOURCE_MANIFEST.json"),
        },
        "data_manifest": {
            "path": "DATA/MANIFEST.json",
            "sha256": file_hash(root / "DATA/MANIFEST.json"),
        },
    }
    write_once(root / "IDENTITY.json", identity)
    freeze_spec(
        root, {parent: item["content_sha256"] for parent, item in binding["parents"].items()}
    )
    report = {
        "status": "CPU_PREPARED_NOT_RELEASED",
        "at": now(),
        "identity": identity,
        "model_calls": 0,
        "optimizer_updates": 0,
        "remaining": [
            "actual processor visual review",
            "native J regression on server",
            "registered census and real-response audit",
            "pilot and recovery",
            "formal freeze, training and evaluation",
        ],
    }
    write_once(root / "CPU_PREPARED.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resume-source-failure", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.root,
                args.config,
                args.revision,
                resume_source_failure=args.resume_source_failure,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
