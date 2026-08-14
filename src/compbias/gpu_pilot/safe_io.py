"""Symlink-resistant publication helpers for private GPU-pilot artifacts."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path


def prepare_output_path(root: Path, target: Path, *, allow_existing: bool) -> Path:
    root = Path(os.path.abspath(os.fspath(root)))
    target = Path(os.path.abspath(os.fspath(target)))
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"output target escapes its approved root: {target}") from error
    if relative == Path("."):
        raise RuntimeError("output target must be below its approved root")
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise RuntimeError(f"approved output root does not exist: {root}") from error
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError(f"approved output root is not a regular directory: {root}")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        with suppress(FileExistsError):
            current.mkdir()
        metadata = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"output parent is not a regular directory: {current}")
    if target.exists() or target.is_symlink():
        metadata = target.lstat()
        if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"output target is not a regular file: {target}")
        if not allow_existing:
            raise FileExistsError(f"output target already exists: {target}")
    return target


def atomic_write_bytes(root: Path, target: Path, payload: bytes) -> None:
    target = prepare_output_path(root, target, allow_existing=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json_text(root: Path, target: Path, payload: str) -> None:
    atomic_write_bytes(root, target, payload.encode("utf-8"))


def prepare_new_output_directory(root: Path, target: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(root)))
    target = Path(os.path.abspath(os.fspath(target)))
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise RuntimeError("output directory escapes its approved root") from error
    if relative == Path("."):
        raise RuntimeError("output directory must be below its approved root")
    metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"approved output root is not a regular directory: {root}")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        with suppress(FileExistsError):
            current.mkdir()
        current_metadata = current.lstat()
        if current.is_symlink() or not stat.S_ISDIR(current_metadata.st_mode):
            raise RuntimeError(f"output parent is not a regular directory: {current}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    return target
