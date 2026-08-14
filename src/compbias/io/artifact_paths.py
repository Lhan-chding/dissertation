"""Closed output-path validation for local experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_EXPERIMENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SECRET_LIKE_PATTERN = re.compile(
    r"(?:api[-_]?key|secret|pass(?:word|wd)|token|credential|private[-_]?key)",
    re.IGNORECASE,
)
_OWNERSHIP_SCHEMA_VERSION = 1
_MAX_OWNERSHIP_JSON_BYTES = 16 * 1024 * 1024
_MAX_OWNERSHIP_JSON_DEPTH = 64
_MAX_OWNERSHIP_JSON_NODES = 100_000


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _reject_symlink_traversal(path: Path, boundary: Path, label: str) -> None:
    if not _is_within(path, boundary):
        raise ValueError(f"{label} must stay within its approved artifact root")
    current = path
    while current != boundary:
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
        current = current.parent
    if boundary.is_symlink():
        raise ValueError(f"{label} artifact root must not be a symlink")


def validated_artifact_path(
    value: Path | str,
    *,
    repository_root: Path | str,
    label: str,
    suffix: str | tuple[str, ...] | None = None,
) -> Path:
    """Validate an output under ``artifacts/`` or the system temporary root.

    Repository-local paths are deliberately checked before the temporary-root
    exception, so a checkout located under ``/tmp`` cannot use that exception
    to overwrite its own source or configuration files.
    """

    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{label} must be path-like")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must not be empty")
    root = _lexical_absolute(Path(repository_root))
    raw = Path(value)
    candidate = _lexical_absolute(root / raw if not raw.is_absolute() else raw)
    repository_artifacts = root / "artifacts"
    temporary_root = _lexical_absolute(Path(tempfile.gettempdir()).resolve())

    if _is_within(candidate, root):
        approved_root = repository_artifacts
        if not _is_within(candidate, approved_root) or candidate == approved_root:
            raise ValueError(f"{label} must stay under the repository artifacts directory")
    elif _is_within(candidate, temporary_root) and candidate != temporary_root:
        approved_root = temporary_root
    else:
        raise ValueError(
            f"{label} must stay under the repository artifacts directory or temporary root"
        )

    _reject_symlink_traversal(candidate, approved_root, label)
    if candidate.exists() and candidate.is_dir() and suffix is not None:
        raise ValueError(f"{label} must be a file, not a directory")
    if candidate.exists() and not candidate.is_dir() and suffix is None:
        raise ValueError(f"{label} must be a directory, not a file")
    if suffix is not None:
        allowed = (suffix,) if isinstance(suffix, str) else suffix
        if not allowed or any(not isinstance(item, str) or not item for item in allowed):
            raise ValueError("suffix must contain one or more non-empty strings")
        if candidate.suffix not in allowed:
            rendered = ", ".join(allowed)
            raise ValueError(f"{label} must use an approved suffix: {rendered}")
    return candidate


def ensure_distinct_nonoverlapping(paths: Mapping[str, Path]) -> None:
    """Reject equal or parent/child output targets before any write occurs."""

    items = tuple((name, _lexical_absolute(path)) for name, path in paths.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"artifact output paths overlap: {left_name} and {right_name}")


@dataclass(frozen=True, slots=True)
class ArtifactOwnership:
    """Immutable specification for one runner-owned artifact transaction."""

    marker_path: Path
    marker_root: Path
    tool: str
    experiment: str
    config_sha256: str
    targets: tuple[tuple[str, Path], ...]
    primary_json: str
    primary_schema_version: int
    primary_experiment: str
    overwrite: bool
    had_existing_bundle: bool


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_experiment_name(value: object) -> str:
    """Return a path-safe, non-secret experiment identifier."""

    if not isinstance(value, str) or _SAFE_EXPERIMENT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "experiment must be 1-128 ASCII letters, digits, dots, hyphens, or "
            "underscores and begin with a letter or digit"
        )
    if _SECRET_LIKE_PATTERN.search(value) is not None:
        raise ValueError("experiment must not contain secret-like credential labels")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_mapping(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate JSON key: {key!r}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{label} contains non-finite JSON number: {value}")
        return parsed

    try:
        if path.stat().st_size > _MAX_OWNERSHIP_JSON_BYTES:
            raise ValueError(f"{label} exceeds the 16 MiB safety limit")
        raw = path.read_bytes()
        if len(raw) > _MAX_OWNERSHIP_JSON_BYTES:
            raise ValueError(f"{label} exceeds the 16 MiB safety limit")
        loaded = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except ValueError as error:
        if "16 MiB" in str(error) or "non-finite" in str(error) or "duplicate" in str(error):
            raise
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    except (json.JSONDecodeError, OSError, UnicodeError, RecursionError) as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    pending: list[tuple[object, int]] = [(loaded, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if depth > _MAX_OWNERSHIP_JSON_DEPTH or visited > _MAX_OWNERSHIP_JSON_NODES:
            raise ValueError(f"{label} exceeds the permitted depth or complexity")
        if isinstance(current, dict):
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
    return loaded


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"{label} has invalid fields ({'; '.join(details)})")


def _validated_targets(
    paths: Mapping[str, Path], repository_root: Path | str
) -> tuple[tuple[str, Path], ...]:
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError("artifact targets must be a non-empty mapping")
    normalized: dict[str, Path] = {}
    for raw_name, raw_path in paths.items():
        name = _nonempty_string(raw_name, "artifact target name")
        if name in normalized:
            raise ValueError(f"duplicate artifact target name: {name}")
        lexical = _lexical_absolute(Path(raw_path))
        if not lexical.suffix:
            raise ValueError(f"artifact target {name!r} must have a file suffix")
        normalized[name] = validated_artifact_path(
            lexical,
            repository_root=repository_root,
            label=f"artifact target {name}",
            suffix=lexical.suffix,
        )
    ensure_distinct_nonoverlapping(normalized)
    return tuple(sorted(normalized.items()))


def _ownership_marker(
    *,
    targets: tuple[tuple[str, Path], ...],
    repository_root: Path | str,
    tool: str,
) -> tuple[Path, Path]:
    parent_strings = [os.fspath(path.parent) for _name, path in targets]
    marker_root = Path(os.path.commonpath(parent_strings))
    target_identity = json.dumps(
        {name: path.relative_to(marker_root).as_posix() for name, path in targets},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{tool}\0{target_identity}".encode()).hexdigest()[:16]
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(tool).stem).strip("-") or "runner"
    marker = marker_root / f".compbias-{stem}-{digest}.ownership.json"
    return (
        validated_artifact_path(
            marker,
            repository_root=repository_root,
            label="artifact ownership marker",
            suffix=".json",
        ),
        marker_root,
    )


def _relative_targets(
    ownership: ArtifactOwnership,
    source_paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, str], ...]:
    sources = dict(ownership.targets) if source_paths is None else source_paths
    return tuple(
        {
            "name": name,
            "path": path.relative_to(ownership.marker_root).as_posix(),
            "sha256": _sha256(sources[name]),
        }
        for name, path in ownership.targets
    )


def _validate_primary_json(
    ownership: ArtifactOwnership,
    target_map: Mapping[str, Path] | None = None,
) -> None:
    targets = dict(ownership.targets) if target_map is None else target_map
    payload = _strict_json_mapping(targets[ownership.primary_json], "primary artifact JSON")
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != ownership.primary_schema_version
    ):
        raise FileExistsError("primary artifact JSON schema does not match this runner")
    if payload.get("experiment") != ownership.primary_experiment:
        raise FileExistsError("primary artifact JSON experiment does not match this runner")


def _validate_existing_ownership(ownership: ArtifactOwnership) -> None:
    marker = ownership.marker_path
    if marker.is_symlink():
        raise ValueError("artifact ownership marker must not be a symlink")
    if not marker.is_file():
        raise FileExistsError(
            "existing artifacts have no valid ownership marker; move them aside before rerunning"
        )
    payload = _strict_json_mapping(marker, "artifact ownership marker")
    expected_keys = {
        "schema_version",
        "kind",
        "tool",
        "experiment",
        "config_sha256",
        "primary_json",
        "targets",
    }
    _exact_keys(payload, expected_keys, "artifact ownership marker")
    expected_scalars = {
        "schema_version": _OWNERSHIP_SCHEMA_VERSION,
        "kind": "compbias_artifact_ownership",
        "tool": ownership.tool,
        "experiment": ownership.experiment,
        "config_sha256": ownership.config_sha256,
        "primary_json": ownership.primary_json,
    }
    for field, expected in expected_scalars.items():
        if type(payload.get(field)) is not type(expected) or payload[field] != expected:
            raise FileExistsError(f"artifact ownership {field} does not match the requested run")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("artifact ownership marker targets must be a list")
    expected_targets = {
        name: path.relative_to(ownership.marker_root).as_posix() for name, path in ownership.targets
    }
    observed_targets: dict[str, tuple[str, str]] = {}
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(f"artifact ownership target {index} must be an object")
        _exact_keys(raw_target, {"name", "path", "sha256"}, f"ownership target {index}")
        name = _nonempty_string(raw_target.get("name"), f"ownership target {index}.name")
        relative_path = _nonempty_string(raw_target.get("path"), f"ownership target {index}.path")
        digest = _nonempty_string(raw_target.get("sha256"), f"ownership target {index}.sha256")
        if name in observed_targets:
            raise ValueError(f"artifact ownership marker repeats target {name!r}")
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"artifact ownership target {name!r} has invalid SHA-256")
        observed_targets[name] = (relative_path, digest)
    if set(observed_targets) != set(expected_targets):
        raise FileExistsError("artifact ownership target set is not complete or does not match")
    for name, path in ownership.targets:
        relative_path, digest = observed_targets[name]
        if relative_path != expected_targets[name]:
            raise FileExistsError(f"artifact ownership path does not match target {name!r}")
        if _sha256(path) != digest:
            raise FileExistsError(f"artifact hash does not match ownership for target {name!r}")
    _validate_primary_json(ownership)


def prepare_artifact_ownership(
    paths: Mapping[str, Path],
    *,
    repository_root: Path | str,
    tool: str,
    experiment: str,
    config_sha256: str,
    primary_json: str,
    primary_schema_version: int,
    primary_experiment: str,
    overwrite: bool,
) -> ArtifactOwnership:
    """Validate a complete output set before a first run or trusted overwrite."""

    normalized_tool = _nonempty_string(tool, "tool")
    normalized_experiment = validate_experiment_name(experiment)
    normalized_primary = _nonempty_string(primary_json, "primary_json")
    normalized_primary_experiment = validate_experiment_name(primary_experiment)
    if _SHA256_PATTERN.fullmatch(config_sha256) is None:
        raise ValueError("config_sha256 must be a lowercase SHA-256 hex digest")
    if type(primary_schema_version) is not int or primary_schema_version < 1:
        raise ValueError("primary_schema_version must be a positive integer")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean")
    targets = _validated_targets(paths, repository_root)
    if normalized_primary not in dict(targets):
        raise ValueError("primary_json must name one registered artifact target")
    marker, marker_root = _ownership_marker(
        targets=targets,
        repository_root=repository_root,
        tool=normalized_tool,
    )
    ownership = ArtifactOwnership(
        marker_path=marker,
        marker_root=marker_root,
        tool=normalized_tool,
        experiment=normalized_experiment,
        config_sha256=config_sha256,
        targets=targets,
        primary_json=normalized_primary,
        primary_schema_version=primary_schema_version,
        primary_experiment=normalized_primary_experiment,
        overwrite=overwrite,
        had_existing_bundle=False,
    )
    invalid_targets = tuple(
        name
        for name, path in targets
        if (path.exists() or path.is_symlink()) and (path.is_symlink() or not path.is_file())
    )
    if invalid_targets:
        raise ValueError(
            "artifact targets must be regular, non-symlink files: " + ", ".join(invalid_targets)
        )
    present = tuple(name for name, path in targets if path.exists())
    if present and len(present) != len(targets):
        raise FileExistsError(
            "artifact target set is not complete; move partial outputs aside before rerunning"
        )
    if present and not overwrite:
        rendered = ", ".join(str(dict(targets)[name]) for name in present)
        raise FileExistsError(
            f"output already exists; pass --overwrite to replace tool artifacts: {rendered}"
        )
    if present:
        _validate_existing_ownership(ownership)
    elif marker.exists() or marker.is_symlink():
        raise FileExistsError(
            "artifact ownership marker exists without its complete target set; move it aside"
        )
    if present:
        ownership = ArtifactOwnership(
            marker_path=ownership.marker_path,
            marker_root=ownership.marker_root,
            tool=ownership.tool,
            experiment=ownership.experiment,
            config_sha256=ownership.config_sha256,
            targets=ownership.targets,
            primary_json=ownership.primary_json,
            primary_schema_version=ownership.primary_schema_version,
            primary_experiment=ownership.primary_experiment,
            overwrite=ownership.overwrite,
            had_existing_bundle=True,
        )
    return ownership


def _ownership_payload(
    ownership: ArtifactOwnership,
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": _OWNERSHIP_SCHEMA_VERSION,
        "kind": "compbias_artifact_ownership",
        "tool": ownership.tool,
        "experiment": ownership.experiment,
        "config_sha256": ownership.config_sha256,
        "primary_json": ownership.primary_json,
        "targets": list(_relative_targets(ownership, source_paths)),
    }


def _write_staged_marker(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _promote_staged_file(source: Path, destination: Path) -> None:
    """Promote one staged file; kept narrow so failure injection is deterministic."""

    os.replace(source, destination)


def _backup_accepted_file(source: Path, backup: Path) -> None:
    """Move one accepted file into the transaction-local rollback area."""

    os.replace(source, backup)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_regular_destination(path: Path, ownership: ArtifactOwnership, label: str) -> None:
    _reject_symlink_traversal(path, ownership.marker_root, label)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RuntimeError(f"promoted artifact is unavailable: {label}") from error
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"promoted artifact is not a regular file: {label}")


def _remove_promoted_file(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"cannot roll back non-file artifact target: {path}")


def _validate_staged_targets(
    ownership: ArtifactOwnership,
    staged_paths: Mapping[str, Path],
) -> None:
    if set(staged_paths) != {name for name, _path in ownership.targets}:
        raise ValueError("staged artifact target set differs from ownership")
    invalid = tuple(
        name for name, path in staged_paths.items() if path.is_symlink() or not path.is_file()
    )
    if invalid:
        raise FileNotFoundError(
            "staged artifact targets must be complete regular files: " + ", ".join(invalid)
        )
    for path in staged_paths.values():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(next(iter(staged_paths.values())).parent)
    _validate_primary_json(ownership, staged_paths)


def _revalidate_commit_destination(ownership: ArtifactOwnership) -> None:
    if ownership.had_existing_bundle:
        _validate_existing_ownership(ownership)
        return
    unexpected = tuple(
        path for _name, path in ownership.targets if path.exists() or path.is_symlink()
    )
    if unexpected or ownership.marker_path.exists() or ownership.marker_path.is_symlink():
        raise FileExistsError("artifact destination changed after ownership preparation")


def _commit_artifact_transaction(
    ownership: ArtifactOwnership,
    staged_paths: Mapping[str, Path],
    transaction_root: Path,
    after_promote: Callable[[], None] | None,
) -> None:
    _validate_staged_targets(ownership, staged_paths)
    _revalidate_commit_destination(ownership)
    marker_stage = transaction_root / "ownership-marker.json"
    _write_staged_marker(marker_stage, _ownership_payload(ownership, staged_paths))
    _fsync_directory(transaction_root)
    backup_root = transaction_root / "accepted-backup"
    backup_root.mkdir()
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    # Revoke the commit record before moving accepted data, then restore it last
    # on rollback.  New data follows the inverse order: data first, marker last.
    destinations = (("ownership_marker", ownership.marker_path), *ownership.targets)
    try:
        if ownership.had_existing_bundle:
            for index, (_name, destination) in enumerate(destinations):
                _reject_symlink_traversal(
                    destination,
                    ownership.marker_root,
                    "accepted artifact backup",
                )
                if destination.is_symlink() or not destination.is_file():
                    raise RuntimeError("accepted artifact changed before backup")
                backup = backup_root / f"{index:04d}-{destination.name}"
                _backup_accepted_file(destination, backup)
                backups.append((destination, backup))
            _fsync_directory(backup_root)
            for parent in {destination.parent for destination, _backup in backups}:
                _fsync_directory(parent)
        for name, destination in ownership.targets:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _reject_symlink_traversal(destination, ownership.marker_root, name)
            _promote_staged_file(staged_paths[name], destination)
            promoted.append(destination)
            _validate_regular_destination(destination, ownership, name)
        ownership.marker_path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_traversal(
            ownership.marker_path,
            ownership.marker_root,
            "artifact ownership marker",
        )
        _promote_staged_file(marker_stage, ownership.marker_path)
        promoted.append(ownership.marker_path)
        _validate_regular_destination(
            ownership.marker_path,
            ownership,
            "artifact ownership marker",
        )
        for parent in {path.parent for path in promoted}:
            _fsync_directory(parent)
        if after_promote is not None:
            after_promote()
    except BaseException as error:
        rollback_errors: list[Exception] = []
        for destination in reversed(promoted):
            try:
                _remove_promoted_file(destination)
            except (OSError, RuntimeError) as rollback_error:
                rollback_errors.append(rollback_error)
        for destination, backup in reversed(backups):
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "artifact transaction failed and accepted-bundle rollback was incomplete"
            ) from error
        for parent in {path.parent for path in (*promoted, *(item[0] for item in backups))}:
            _fsync_directory(parent)
        raise


@contextmanager
def artifact_ownership_transaction(
    ownership: ArtifactOwnership,
    *,
    after_promote: Callable[[], None] | None = None,
) -> Iterator[Mapping[str, Path]]:
    """Stage and promote a complete owned bundle, rolling back every failure.

    ``after_promote`` is part of the same commit boundary.  A failure while
    finalizing required external evidence (for example, a complete RunLogger
    bundle) therefore restores the previously accepted artifact bytes.
    """

    if not isinstance(ownership, ArtifactOwnership):
        raise TypeError("ownership must be an ArtifactOwnership")
    if after_promote is not None and not callable(after_promote):
        raise TypeError("after_promote must be callable or None")
    ownership.marker_root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_traversal(
        ownership.marker_root,
        ownership.marker_root,
        "artifact transaction root",
    )
    transaction_root = Path(
        tempfile.mkdtemp(
            dir=ownership.marker_root,
            prefix=f".{ownership.marker_path.stem}.transaction-",
        )
    )
    staged = {
        name: transaction_root / f"{index:04d}-{path.name}"
        for index, (name, path) in enumerate(ownership.targets)
    }
    try:
        yield MappingProxyType(staged)
        _commit_artifact_transaction(
            ownership,
            staged,
            transaction_root,
            after_promote,
        )
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)


def finalize_artifact_ownership(ownership: ArtifactOwnership) -> None:
    """Atomically bind a successfully written complete artifact set to its owner."""

    if not isinstance(ownership, ArtifactOwnership):
        raise TypeError("ownership must be an ArtifactOwnership")
    missing = tuple(name for name, path in ownership.targets if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"cannot finalize incomplete artifact target set: {', '.join(missing)}"
        )
    symlinks = tuple(name for name, path in ownership.targets if path.is_symlink())
    if symlinks:
        raise ValueError(f"artifact targets must not be symlinks: {', '.join(symlinks)}")
    _validate_primary_json(ownership)
    payload = _ownership_payload(ownership, dict(ownership.targets))
    ownership.marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ownership.marker_path.parent,
            prefix=f".{ownership.marker_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, ownership.marker_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


__all__ = [
    "ArtifactOwnership",
    "artifact_ownership_transaction",
    "ensure_distinct_nonoverlapping",
    "finalize_artifact_ownership",
    "prepare_artifact_ownership",
    "validate_experiment_name",
    "validated_artifact_path",
]
