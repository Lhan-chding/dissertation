"""Bounded paths, atomic no-clobber artifacts and hash-bound resumable writers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .protocol import ROOT

BINDING_KEYS = ("source", "config", "data", "selector", "packet")
MANIFEST_NAME = "RUN_MANIFEST.json"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def checked_output_path(path: Path | str, *, project_root: Path | str = ROOT) -> Path:
    root = Path(project_root).resolve()
    path = Path(path)
    path = (root / path if not path.is_absolute() else path).resolve()
    allowed = [root / "runs/modeling_contrast_v2", root / "docs/modeling_contrast/results"]
    if not any(path == item or item in path.parents for item in allowed):
        raise ValueError(f"output path escapes the new contrast result roots: {path}")
    return path


def _atomic_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard-link publication is atomic and refuses an existing target.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bytes(path, value: bytes, *, project_root=ROOT) -> None:
    _atomic_bytes(checked_output_path(path, project_root=project_root), value)


def write_text(path, value: str, *, project_root=ROOT) -> None:
    write_bytes(path, value.encode("utf-8"), project_root=project_root)


def write_json(path, value: Any, *, project_root=ROOT) -> None:
    write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        project_root=project_root,
    )


def write_npz(path, arrays: dict, *, project_root=ROOT) -> None:
    import numpy as np

    checked = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object array forbidden: {key}")
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite array forbidden: {key}")
        checked[key] = array
    stream = io.BytesIO()
    np.savez_compressed(stream, **checked)
    write_bytes(path, stream.getvalue(), project_root=project_root)


def source_hashes(root: Path | str) -> dict[str, str]:
    root = Path(root).resolve()
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    }


def source_snapshot(source_root, out, *, project_root=ROOT) -> dict:
    source_root = Path(source_root).resolve()
    out = checked_output_path(out, project_root=project_root)
    files = source_hashes(source_root)
    for relative, digest in files.items():
        data = (source_root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source changed during snapshot")
        write_bytes(out / "files" / relative, data, project_root=project_root)
    result = {"source_root": str(source_root), "files": files, "source_hash": canonical_hash(files)}
    write_json(out / "SOURCE_MANIFEST.json", result, project_root=project_root)
    return result


def validate_binding(binding: dict) -> None:
    if set(binding) != set(BINDING_KEYS) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in binding.values()
    ):
        raise ValueError("binding requires source/config/data/selector/packet SHA256 hashes")


def verify_run_manifest(out: Path | str, binding: dict | None = None) -> dict:
    out = Path(out).resolve()
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    validate_binding(manifest["binding"])
    if binding is not None and manifest["binding"] != binding:
        raise ValueError("resume binding mismatch")
    for relative, digest in manifest["outputs"].items():
        path = (out / relative).resolve()
        if out not in path.parents or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"output hash/path mismatch: {relative}")
    actual = {
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file() and path.name not in (MANIFEST_NAME, ".writer.lock")
    }
    if actual != set(manifest["outputs"]):
        raise ValueError("unregistered output files prevent verified resume")
    return manifest


class RunWriter(AbstractContextManager):
    """One stage writer; every completed artifact is journaled by byte hash.

    Use ``with RunWriter(out, binding) as writer`` and its write methods.
    Existing completed files are immutable, even on resume. A surviving lock
    after process death requires explicit operator review; no automatic takeover.
    """

    def __init__(self, out, binding, *, resume=False, project_root=ROOT):
        validate_binding(binding)
        self.project_root = project_root
        self.out = checked_output_path(out, project_root=project_root)
        self.binding = dict(binding)
        self.resume = resume
        self.resumed = False
        self.output_hashes: dict[str, str] = {}
        self._active = False
        self.manifest = {}
        self.previous_status = None

    def __enter__(self):
        if self.out.exists() and not self.resume:
            raise FileExistsError(f"no-clobber stage already exists: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = self.out / ".writer.lock"
        try:
            lock_fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise RuntimeError("stage has an active or unreconciled writer lock") from error
        with os.fdopen(lock_fd, "w") as stream:
            json.dump({"pid": os.getpid(), "created_ns": time.time_ns()}, stream)
        try:
            existing = self.out / MANIFEST_NAME
            if existing.exists():
                self.manifest = verify_run_manifest(self.out, self.binding)
                self.output_hashes = dict(self.manifest["outputs"])
                self.resumed = True
                self.previous_status = self.manifest["status"]
            else:
                if any(path != self.lock for path in self.out.iterdir()):
                    raise ValueError("existing unbound files prevent resume")
                self.manifest = {
                    "schema_version": "contrast-run-v1",
                    "binding": self.binding,
                    "created_ns": time.time_ns(),
                    "outputs": {},
                    "status": "RUNNING",
                }
            self._active = True
            self._journal("RUNNING")
            return self
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def _journal(self, status):
        self.manifest.update(outputs=self.output_hashes, status=status)
        _atomic_bytes(
            self.out / MANIFEST_NAME,
            (json.dumps(self.manifest, indent=2, allow_nan=False) + "\n").encode(),
            overwrite=True,
        )

    def path(self, relative) -> Path:
        if not self._active:
            raise RuntimeError("writer context is not active")
        path = checked_output_path(self.out / relative, project_root=self.project_root)
        if self.out not in path.parents or path.name in (MANIFEST_NAME, ".writer.lock"):
            raise ValueError("output must be a nonreserved file inside this stage")
        return path

    def register_existing(self, relative) -> str:
        path = self.path(relative)
        name = str(path.relative_to(self.out))
        digest = sha256_file(path)
        if name in self.output_hashes and self.output_hashes[name] != digest:
            raise ValueError("registered output hash changed")
        self.output_hashes[name] = digest
        self._journal("RUNNING")
        return digest

    def write_bytes(self, relative, value):
        write_bytes(self.path(relative), value, project_root=self.project_root)
        return self.register_existing(relative)

    def write_text(self, relative, value):
        return self.write_bytes(relative, value.encode("utf-8"))

    def write_json(self, relative, value):
        return self.write_text(
            relative, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    def write_npz(self, relative, arrays):
        write_npz(self.path(relative), arrays, project_root=self.project_root)
        return self.register_existing(relative)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._journal("COMPLETE" if exc_type is None else "INTERRUPTED")
        finally:
            self._active = False
            self.lock.unlink(missing_ok=True)
        return False
