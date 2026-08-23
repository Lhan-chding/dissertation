"""Deterministic, symlink-free Study C3 evidence archive construction."""

from __future__ import annotations

import gzip
import io
import tarfile
from collections.abc import Iterable
from pathlib import Path


def _safe_relative(path: Path) -> Path:
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"Study C3 archive member is unsafe: {path}")
    return path


def create_deterministic_evidence_archive(
    *, root: Path, sources: Iterable[Path], output: Path
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Study C3 evidence root is missing or unsafe: {root}")
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"Study C3 archive overwrite is forbidden: {output}")
    members = tuple(sorted({_safe_relative(Path(path)) for path in sources}, key=str))
    if not members:
        raise ValueError("Study C3 evidence archive cannot be empty")
    cached: list[tuple[Path, bytes]] = []
    for relative in members:
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Study C3 evidence source is missing or unsafe: {relative}")
        cached.append((relative, source.read_bytes()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative, data in cached:
                    info = tarfile.TarInfo(relative.as_posix())
                    info.size = len(data)
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))


__all__ = ["create_deterministic_evidence_archive"]
