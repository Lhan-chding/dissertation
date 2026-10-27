"""Immutable phase provenance; binding never authorizes scientific execution.

S0 is verified against its explicit historical deployment, never this module's
location. A later phase binds a complete new source inventory and a declared
delta, while retaining the original root identity and evidence producer context.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

from .common import digest, file_hash, read
from .contract import AMENDMENTS, PACKAGE_CONFIG_SHA256, PROJECT, ROOT

ORIGIN = "CENSUS_S0"
PHASES = {
    "CENSUS_REVIEW_V1": ({ORIGIN}, {ORIGIN}),
    "PILOT_V1": ({ORIGIN}, {ORIGIN, "CENSUS_REVIEW_V1"}),
    "FORMAL_V1": ({ORIGIN, "PILOT_V1"}, {ORIGIN, "CENSUS_REVIEW_V1", "PILOT_V1"}),
}
MANIFEST_SCHEMA = "srf6-phase-source-manifest-v1"
IDENTITY_SCHEMA = "srf6-phase-identity-v1"
DELTA_SCHEMA = "srf6-phase-source-delta-v1"
CACHE_SCHEMA = "srf6-phase-content-cache-v1"


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PermissionError("Expected a content SHA256")
    return value


def _revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise PermissionError("A full immutable Git revision is required")
    return value


def _relative(value):
    if not isinstance(value, str) or not value:
        raise PermissionError("A nonempty relative path is required")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise PermissionError("Unsafe relative provenance path")
    return path


def _path(path, *, exists=True):
    """Reject symlinks at every level, including the project boundary."""
    if not isinstance(path, (str, Path)):
        raise PermissionError("An explicit provenance path is required")
    path, project = Path(path), Path(PROJECT)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(project):
        raise PermissionError("Provenance path is outside the authorized project")
    if project.resolve(strict=True) != project or not project.is_dir():
        raise PermissionError("Project boundary is not a canonical directory")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise PermissionError("Symlinked provenance paths are forbidden")
        if item == project:
            break
    try:
        resolved = path.resolve(strict=exists)
    except OSError as exc:
        raise PermissionError("Missing or inaccessible provenance path") from exc
    if resolved != path:
        raise PermissionError("Provenance path is not canonical")
    return path


def _root(root):
    root = _path(root)
    if root != Path(PROJECT) / ROOT.name or not root.is_dir():
        raise PermissionError("Only the registered SR-F6 run root is allowed")
    return root


def _deployment(path, revision):
    revision = _revision(revision)
    path = _path(path)
    match = re.fullmatch(r"sr_f6_code_([0-9a-f]{7,40})", path.name)
    if (
        path.parent != Path(PROJECT)
        or not path.is_dir()
        or match is None
        or not revision.startswith(match.group(1))
    ):
        raise PermissionError("Deployment must be the declared revision under this project")
    return path


def _file(base, relative):
    path = _path(Path(base) / _relative(relative))
    if not path.is_file():
        raise PermissionError("Provenance member is not a regular file")
    return path


def _payload(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _stat(path):
    value = path.stat()
    # Device numbers differ between cluster mounts; they are not content identity.
    return [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode]


class _ContentCache:
    """Persist successful file checks, including when a later check fails."""

    def __init__(self, root):
        self.directory = _path(root / ".verification-cache", exists=False)
        self.directory.mkdir(exist_ok=True)
        self.path = _path(self.directory / "phases.json", exists=False)
        self.lock_path = _path(self.directory / "phases.lock", exists=False)

    def __enter__(self):
        self.lock = self.lock_path.open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        try:
            value = read(self.path) if self.path.exists() else {"schema": CACHE_SCHEMA, "files": {}}
            if value.get("schema") != CACHE_SCHEMA or not isinstance(value.get("files"), dict):
                raise PermissionError("Invalid phase content cache")
            self.files = value["files"]
            self.dirty = False
            return self
        except BaseException:
            self.lock.close()
            raise

    def hash(self, path):
        path = _path(path)
        before = _stat(path)
        cached = self.files.get(str(path), {})
        if cached.get("stat") == before:
            value = _sha(cached.get("sha256"))
        else:
            value = file_hash(path)
            if before != _stat(path):
                raise PermissionError("File changed while hashing")
            self.files[str(path)] = {"stat": before, "sha256": value}
            self.dirty = True
        return value

    def adopt_verified(self, path, expected_sha256, previous):
        """Reuse gates' prior manifest-bound check, never an unbound cache SHA."""
        current = _stat(path)
        if (
            isinstance(previous, dict)
            and previous.get("sha256") == expected_sha256
            and previous.get("stat") == current[:4]
        ):
            entry = {"stat": current, "sha256": expected_sha256}
            if self.files.get(str(path)) != entry:
                self.files[str(path)] = entry
                self.dirty = True

    def __exit__(self, *_):
        try:
            if self.dirty:
                fd, temporary = tempfile.mkstemp(
                    prefix=".phases-", suffix=".tmp", dir=self.directory
                )
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(_payload({"schema": CACHE_SCHEMA, "files": self.files}))
                        stream.flush()
                        os.fsync(stream.fileno())
                    _path(self.path, exists=False)
                    os.replace(temporary, self.path)
                finally:
                    if Path(temporary).exists():
                        Path(temporary).unlink()
        finally:
            self.lock.close()


def _source_paths(deployment, *, legacy=False):
    paths = set()
    for relative in ("src", "tests/sr_f6", "scripts/sr_f6"):
        base = _path(deployment / relative, exists=False)
        if not base.exists():
            if relative == "src":
                raise PermissionError("Deployment has no source tree")
            continue
        if not base.is_dir():
            raise PermissionError("Source scope is not a directory")
        for directory, children, files in os.walk(base, followlinks=False):
            for name in (*children, *files):
                _path(Path(directory) / name)
            for name in files:
                path = Path(directory) / name
                script = relative == "scripts/sr_f6"
                if script and legacy and path.parent != base:
                    continue
                extensions = {".py", ".sbatch"} if script else {".py"}
                if script and not legacy:
                    extensions.add(".sh")
                if path.suffix in extensions:
                    paths.add(str(path.relative_to(deployment)))
    if not paths:
        raise PermissionError("Empty executable source scope")
    return paths


def _members(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise PermissionError("Missing manifest members")
    output = {}
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise PermissionError("Unexpected manifest member schema")
        _relative(item["path"])
        _sha(item["sha256"])
        if type(item["bytes"]) is not int or item["bytes"] < 0 or item["path"] in output:
            raise PermissionError("Invalid or duplicate manifest member")
        output[item["path"]] = item
    if not output:
        raise PermissionError("Empty manifest")
    return output


def _verify_manifest(
    base, manifest, cache, *, source=False, legacy=False, verified_manifest_sha256=None
):
    members = _members(manifest)
    if source and set(members) != _source_paths(base, legacy=legacy):
        raise PermissionError("Executable source inventory is not complete and exact")
    previous = {}
    if verified_manifest_sha256 is not None:
        legacy_path = _path(
            cache.directory / f"{_sha(verified_manifest_sha256)}.json", exists=False
        )
        if legacy_path.exists():
            previous = read(legacy_path)
            if not isinstance(previous, dict):
                raise PermissionError("Invalid historical manifest verification cache")
    for item in members.values():
        path = _file(base, item["path"])
        cache.adopt_verified(path, item["sha256"], previous.get(str(path)))
        if path.stat().st_size != item["bytes"] or cache.hash(path) != item["sha256"]:
            raise PermissionError("Bound manifest content changed: " + str(path))


def _build_manifest(deployment, cache):
    files = []
    for relative in sorted(_source_paths(deployment)):
        path = _file(deployment, relative)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": cache.hash(path)})
    return {
        "schema": MANIFEST_SCHEMA,
        "scope": "all src Python, SR-F6 tests and recursive Python/shell/sbatch entrypoints",
        "files": files,
    }


def build_manifest(deployment, *, revision):
    """Inventory an explicit authorized deployment; no phase is registered."""
    root = _root(Path(PROJECT) / ROOT.name)
    deployment = _deployment(deployment, revision)
    with _ContentCache(root) as cache:
        return _build_manifest(deployment, cache)


def source_delta(before, after, *, reasons):
    """Declare every added/modified/deleted source; unexplained changes fail."""
    old, new = _members(before), _members(after)
    changed = sorted(path for path in old.keys() | new.keys() if old.get(path) != new.get(path))
    if not isinstance(reasons, dict) or set(reasons) != set(changed):
        raise PermissionError("Every source difference requires exactly one declared reason")
    changes = []
    for path in changed:
        reason = reasons[path]
        if not isinstance(reason, str) or not reason.strip():
            raise PermissionError("Source change reason cannot be empty")
        changes.append(
            {
                "path": path,
                "kind": "added"
                if path not in old
                else "deleted"
                if path not in new
                else "modified",
                "before": old.get(path),
                "after": new.get(path),
                "reason": reason,
            }
        )
    return {
        "schema": DELTA_SCHEMA,
        "changes": changes,
        "unchanged_count": len(old.keys() & new.keys())
        - sum(path in old and path in new for path in changed),
    }


def _bound_json(root, reference, cache):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise PermissionError("Invalid immutable artifact reference")
    path = _file(root, reference["path"])
    if cache.hash(path) != _sha(reference["sha256"]):
        raise PermissionError("Bound provenance artifact changed: " + str(path))
    return read(path)


def _verify_origin(root, deployment, cache):
    identity = read(_file(root, "IDENTITY.json"))
    deployment = _deployment(deployment, identity.get("revision"))
    config = read(_file(root, "CONFIG.json"))
    if (
        cache.hash(_file(root, "CONFIG.json")) != PACKAGE_CONFIG_SHA256
        or identity.get("config_sha256") != PACKAGE_CONFIG_SHA256
        or config.get("deployment", {}).get("root") != str(ROOT)
        or identity.get("implementation_digest") != digest(AMENDMENTS)
        or digest(read(_file(root, "IMPLEMENTATION_REGISTRATION.json"))) != digest(AMENDMENTS)
        or identity.get("source_manifest", {}).get("path") != "SOURCE_MANIFEST.json"
        or identity.get("data_manifest", {}).get("path") != "DATA/MANIFEST.json"
    ):
        raise PermissionError("Original run identity/configuration differs")
    source = _bound_json(root, identity["source_manifest"], cache)
    data = _bound_json(root, identity["data_manifest"], cache)
    _verify_manifest(
        deployment,
        source,
        cache,
        source=True,
        legacy=True,
        verified_manifest_sha256=identity["source_manifest"]["sha256"],
    )
    _verify_manifest(
        root, data, cache, verified_manifest_sha256=identity["data_manifest"]["sha256"]
    )
    return {
        "phase": ORIGIN,
        "status": "BOUND_NOT_AUTHORIZED",
        "training_allowed": False,
        "deployment": str(deployment),
        "identity": identity,
        "identity_sha256": cache.hash(_file(root, "IDENTITY.json")),
        "config_canonical_sha256": digest(config),
    }


def verify_origin(root, *, deployment):
    """Verify historical S0 from its declared deployment without rewriting it."""
    root = _root(root)
    with _ContentCache(root) as cache:
        return _verify_origin(root, deployment, cache)


def _phase(phase):
    if phase not in PHASES:
        raise PermissionError("Unregistered phase; S0 cannot be registered again")
    return phase


def _dependencies(root, report, relative, cache, active, visited):
    """Follow only explicit receipt inputs/evidence, never arbitrary inventories."""
    if relative in active:
        raise PermissionError("Cyclic evidence dependency")
    if relative in visited:
        return
    active = active | {relative}
    if isinstance(report, dict):
        for name in ("inputs", "evidence"):
            references = report.get(name, [])
            if not isinstance(references, list):
                raise PermissionError("Receipt dependencies must be an explicit list")
            for reference in references:
                if not isinstance(reference, dict) or not {"path", "sha256"} <= reference.keys():
                    raise PermissionError("Malformed receipt dependency")
                if reference["path"] in active:
                    raise PermissionError("Cyclic evidence dependency")
                path = _file(root, reference["path"])
                if cache.hash(path) != _sha(reference["sha256"]):
                    raise PermissionError("Evidence dependency changed")
                # JSONL/raw trajectories, images and other payloads are hash-only.
                # SOURCE_BINDING.model_files is deliberately not traversed.
                child = read(path) if path.suffix == ".json" else None
                _dependencies(root, child, reference["path"], cache, active, visited)
    visited.add(relative)


def _upstreams(root, phase, entries, origin, cache, visiting):
    if not isinstance(entries, list) or not entries:
        raise PermissionError("Explicit upstream producer/evidence references are required")
    found, normalized, visited = set(), [], set()
    required, allowed = PHASES[phase]
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "producer_phase",
            "identity_sha256",
            "evidence",
        }:
            raise PermissionError("Invalid upstream reference schema")
        producer = entry["producer_phase"]
        if producer in found or producer not in allowed:
            raise PermissionError("Duplicate, future or mixed upstream phase")
        found.add(producer)
        if producer == ORIGIN:
            context = origin
        else:
            record = read(_file(root, f"PHASES/{producer}/IDENTITY.json"))
            context = _verify_phase(root, producer, record.get("deployment"), cache, visiting)
        if context["identity_sha256"] != _sha(entry["identity_sha256"]):
            raise PermissionError("Upstream producer identity changed")
        evidence, seen = entry["evidence"], set()
        if not isinstance(evidence, list) or not evidence:
            raise PermissionError("Each upstream producer requires actual bound evidence")
        for reference in evidence:
            report = _bound_json(root, reference, cache)
            if reference["path"] in seen:
                raise PermissionError("Duplicate upstream evidence")
            seen.add(reference["path"])
            if not isinstance(report, dict) or report.get("context") != context["identity"]:
                raise PermissionError("Evidence context differs from its explicit producer")
            # PASS/FAIL/NOT_RUN are deliberately not interpreted here. Consumers
            # separately enforce their scientific gates and response denominators.
            _dependencies(root, report, reference["path"], cache, set(), visited)
        normalized.append({**entry, "evidence": sorted(evidence, key=lambda item: item["path"])})
    if not required <= found:
        raise PermissionError("Required upstream producer is missing")
    return sorted(normalized, key=lambda item: item["producer_phase"])


def _phase_identity(root, phase, deployment, revision, origin, manifest, delta, upstream):
    import hashlib

    prefix = f"PHASES/{phase}"
    return {
        "schema": IDENTITY_SCHEMA,
        "phase": phase,
        "status": "BOUND_NOT_AUTHORIZED",
        "training_allowed": False,
        "root": str(root),
        "deployment": str(deployment),
        "revision": revision,
        "config_sha256": origin["identity"]["config_sha256"],
        "config_canonical_sha256": origin["config_canonical_sha256"],
        "implementation_digest": origin["identity"]["implementation_digest"],
        "data_manifest": origin["identity"]["data_manifest"],
        "origin": {
            "phase": ORIGIN,
            "deployment": origin["deployment"],
            "identity": {"path": "IDENTITY.json", "sha256": origin["identity_sha256"]},
            "source_manifest": origin["identity"]["source_manifest"],
        },
        "source_manifest": {
            "path": f"{prefix}/SOURCE_MANIFEST.json",
            "sha256": hashlib.sha256(_payload(manifest)).hexdigest(),
        },
        "delta": {
            "path": f"{prefix}/DELTA.json",
            "sha256": hashlib.sha256(_payload(delta)).hexdigest(),
        },
        "upstream": upstream,
    }


def _validate_delta(before, after, expected):
    try:
        changes = expected["changes"]
        reasons = {item["path"]: item["reason"] for item in changes}
        if len(reasons) != len(changes):
            raise PermissionError("Duplicate source delta")
        actual = source_delta(before, after, reasons=reasons)
    except (KeyError, TypeError) as exc:
        raise PermissionError("Malformed source delta") from exc
    if actual != expected:
        raise PermissionError("Actual source delta differs from its explicit registration")
    return actual


def _verify_phase(root, phase, deployment, cache, visiting):
    _phase(phase)
    if phase in visiting:
        raise PermissionError("Cyclic phase provenance")
    visiting = visiting | {phase}
    relative = f"PHASES/{phase}/IDENTITY.json"
    identity_path = _file(root, relative)
    identity = read(identity_path)
    if identity.get("schema") != IDENTITY_SCHEMA or identity.get("phase") != phase:
        raise PermissionError("Phase identity schema/name differs")
    deployment = _deployment(deployment, identity.get("revision"))
    if identity.get("deployment") != str(deployment):
        raise PermissionError("Caller deployment differs from registered phase")
    origin = _verify_origin(root, identity.get("origin", {}).get("deployment"), cache)
    if identity.get("origin", {}).get("identity", {}).get("sha256") != origin["identity_sha256"]:
        raise PermissionError("Original root identity changed after phase registration")
    manifest = _bound_json(root, identity.get("source_manifest"), cache)
    delta = _bound_json(root, identity.get("delta"), cache)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PermissionError("Unexpected phase source manifest")
    _verify_manifest(deployment, manifest, cache, source=True)
    before = _bound_json(root, origin["identity"]["source_manifest"], cache)
    _validate_delta(before, manifest, delta)
    upstream = _upstreams(root, phase, identity.get("upstream"), origin, cache, visiting)
    expected = _phase_identity(
        root, phase, deployment, identity["revision"], origin, manifest, delta, upstream
    )
    if identity != expected:
        raise PermissionError("Phase context differs from original configuration/data/provenance")
    return {
        "phase": phase,
        "status": "BOUND_NOT_AUTHORIZED",
        "training_allowed": False,
        "deployment": str(deployment),
        "identity": identity,
        "identity_sha256": cache.hash(identity_path),
    }


def verify_phase(root, phase, *, deployment, expected_identity_sha256=None):
    """Verify only provenance; a scientific gate must separately allow actions."""
    root = _root(root)
    with _ContentCache(root) as cache:
        result = _verify_phase(root, phase, deployment, cache, set())
        if expected_identity_sha256 is not None and result["identity_sha256"] != _sha(
            expected_identity_sha256
        ):
            raise PermissionError("Phase differs from the caller's pinned identity")
        return result


def register_phase(
    root, *, phase, deployment, revision, origin_deployment, upstream, expected_delta
):
    """Atomically register a complete prospective bundle, or verify exact retry.

    No timestamp is injected: retrying identical declared content is idempotent.
    Existing phase files are never overwritten, even after a failed validation.
    """
    root, phase = _root(root), _phase(phase)
    deployment = _deployment(deployment, revision)
    directory = _path(root / "PHASES", exists=False)
    directory.mkdir(exist_ok=True)
    lock_path = _path(directory / ".registration.lock", exists=False)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with _ContentCache(root) as cache:
            origin = _verify_origin(root, origin_deployment, cache)
            if str(deployment) == origin["deployment"]:
                raise PermissionError("A later phase cannot relabel the frozen S0 deployment")
            manifest = _build_manifest(deployment, cache)
            before = _bound_json(root, origin["identity"]["source_manifest"], cache)
            delta = _validate_delta(before, manifest, expected_delta)
            references = _upstreams(root, phase, upstream, origin, cache, {phase})
            identity = _phase_identity(
                root, phase, deployment, revision, origin, manifest, delta, references
            )
            bundle = {
                "SOURCE_MANIFEST.json": manifest,
                "DELTA.json": delta,
                "IDENTITY.json": identity,
            }
            target = _path(directory / phase, exists=False)
            if target.exists():
                for name, value in bundle.items():
                    if _file(target, name).read_bytes() != _payload(value):
                        raise FileExistsError("Preserve existing phase; replacement is forbidden")
            else:
                staging = Path(tempfile.mkdtemp(prefix=f".{phase}-", dir=directory))
                try:
                    for name, value in bundle.items():
                        with (staging / name).open("xb") as stream:
                            stream.write(_payload(value))
                            stream.flush()
                            os.fsync(stream.fileno())
                    for path in (staging,):
                        fd = os.open(path, os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    os.rename(staging, target)
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                finally:
                    if staging.exists():
                        for name in bundle:
                            (staging / name).unlink(missing_ok=True)
                        staging.rmdir()
            return _verify_phase(root, phase, deployment, cache, set())
