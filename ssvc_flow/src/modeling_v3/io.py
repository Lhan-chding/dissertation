"""Immutable evidence, explicit identity, and process-exclusive resume.

No legacy directory restrictions or budget caps are inherited. Callers choose a
new output root. Each request is atomically published once and is recoverable
after process death without treating a partial experiment as complete.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)  # atomic no-clobber publication on shared storage
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(
        path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    )


def atomic_npz(path, arrays):
    import numpy as np

    values = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError("object arrays are forbidden")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def source_identity(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    files = {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted((root / "src").rglob("*.py"))
        if not p.is_symlink()
    }
    return {"sha256": canonical_hash(files), "files": files}


def verify_manifest(root, binding=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if binding is not None and manifest["binding"] != binding:
        raise ValueError("manifest binding mismatch")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"manifest path/hash mismatch: {relative}")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and str(p.relative_to(root)) not in {"RUN_MANIFEST.json", "COMPLETE.json", ".writer.lock"}
    }
    if actual != set(manifest["files"]):
        raise ValueError("unregistered files in manifest")
    if (root / "COMPLETE.json").exists():
        receipt = json.loads((root / "COMPLETE.json").read_text())
        if receipt["manifest_sha256"] != sha256_file(root / "RUN_MANIFEST.json"):
            raise ValueError("completion manifest hash mismatch")
    return manifest


def finalize_run(root, binding, *, status="COMPLETE", metadata=None):
    root = Path(root)
    if (root / "RUN_MANIFEST.json").exists():
        manifest = verify_manifest(root, binding)
        if manifest["status"] != status:
            raise ValueError("published manifest status mismatch")
        if status == "COMPLETE" and not (root / "COMPLETE.json").exists():
            atomic_json(
                root / "COMPLETE.json", {"manifest_sha256": sha256_file(root / "RUN_MANIFEST.json")}
            )
        return manifest
    files = {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and str(p.relative_to(root)) not in {".writer.lock", "RUN_MANIFEST.json", "COMPLETE.json"}
    }
    manifest = {
        "schema": "ssvc-v3-manifest-1",
        "status": status,
        "binding": binding,
        "files": files,
        "metadata": metadata or {},
    }
    atomic_json(root / "RUN_MANIFEST.json", manifest)
    if status == "COMPLETE":
        atomic_json(
            root / "COMPLETE.json", {"manifest_sha256": sha256_file(root / "RUN_MANIFEST.json")}
        )
    return manifest


class AttemptStore:
    def __init__(self, out, binding, resume=False):
        self.out, self.binding, self.resume = Path(out), dict(binding), resume
        self.records = {}
        self._lock = None
        self.complete = False
        self.published = False

    def __enter__(self):
        exists = self.out.exists()
        if exists and not self.resume:
            raise FileExistsError(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._lock = (self.out / ".writer.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise RuntimeError("another writer owns this unit") from error
        try:
            binding_path = self.out / "BINDING.json"
            if binding_path.exists():
                if json.loads(binding_path.read_text()) != self.binding:
                    raise ValueError("resume binding mismatch")
            else:
                atomic_json(binding_path, self.binding)
            (self.out / "records").mkdir(exist_ok=True)
            for path in (self.out / "records").glob("*.json"):
                record = json.loads(path.read_text())
                if record.get("payload_hash") != canonical_hash(record.get("payload")):
                    raise ValueError("record hash mismatch")
                if path.stem != canonical_hash(record["key"]) or record["key"] in self.records:
                    raise ValueError("record key mismatch")
                self.records[record["key"]] = record
            if (self.out / "RUN_MANIFEST.json").exists():
                verify_manifest(self.out, self.binding)
                self.published = True
                self.complete = (self.out / "COMPLETE.json").exists()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def record(self, key, value):
        if not isinstance(key, str) or not key:
            raise ValueError("nonempty request key required")
        record = {"key": key, "payload_hash": canonical_hash(value), "payload": value}
        if key in self.records:
            if self.records[key] != record:
                raise ValueError("request key payload conflict")
            return False
        if self.complete or self.published:
            raise ValueError("complete unit cannot accept new requests")
        atomic_json(self.out / "records" / (canonical_hash(key) + ".json"), record)
        self.records[key] = record
        return True

    def finish(self, expected_keys, metadata=None):
        expected = list(expected_keys)
        if len(expected) != len(set(expected)) or set(expected) != set(self.records):
            raise ValueError("expected request keys differ from observed keys")
        if not self.complete:
            finalize_run(self.out, self.binding, metadata=metadata)
            self.complete = True
        return verify_manifest(self.out, self.binding)

    def __exit__(self, *_):
        if self._lock is not None and not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
