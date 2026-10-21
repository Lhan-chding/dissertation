"""Package the completed registered run once, without extra hash/readback passes."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from .common import now, read, write


def package(root):
    root = Path(root)
    report = read(root / "ANALYSIS/A_B_ENDPOINT_ANALYSIS.json")
    if report["status"] != "A_B_ENDPOINTS_COMPLETE_C_BLOCKED":
        raise RuntimeError("A/B coverage and the documented C boundary must be resolved first")
    destination = root / "DELIVERY"
    receipt = destination / "PACKAGE.json"
    if receipt.exists():
        previous = read(receipt)
        existing = Path(previous["archive"])
        if existing.is_file() and existing.stat().st_size == previous["archive_bytes"]:
            return previous
        raise RuntimeError("A completed package receipt has no matching archive")
    destination.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_relative_to(destination)
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
        and path.name != "RESULT_MANIFEST.json"
        # Mutable pool coordination records remain on the server. Task receipts,
        # model accounting and technical events are included in the package.
        and not path.is_relative_to(root / "pool")
    )
    files = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size) for path in paths]
    manifest = root / "RESULT_MANIFEST.json"
    write(
        manifest,
        dict(
            at=now(),
            status=report["status"],
            files=files,
            total_bytes=sum(row["bytes"] for row in files),
            exclusions=["mutable pool coordination/logs", "Python bytecode", "DELIVERY itself"],
            inherited_model_files=(
                "REFERENCED_ONLY in INPUT_MANIFEST; not a self-contained retraining bundle"
            ),
            verification=(
                "Single archive write with file metadata; no additional hash or archive "
                "readback sweep, per user request"
            ),
        ),
    )
    archive = destination / "SRF4_A_B_COMPLETE_C_BLOCKED.tar.gz"
    temporary = archive.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as stream:
        for path in [manifest, *paths]:
            stream.add(
                path,
                arcname="SRF4_CONSOLIDATED_20261022/" + str(path.relative_to(root)),
                recursive=False,
            )
    os.replace(temporary, archive)
    result = dict(
        at=now(),
        status="PACKAGE_WRITTEN",
        scientific_status=report["status"],
        archive=str(archive),
        archive_bytes=archive.stat().st_size,
        files=len(files) + 1,
        manifest=str(manifest),
        local_delivery="NOT_YET_TRANSFERRED",
    )
    write(receipt, result)
    return result
