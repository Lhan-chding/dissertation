#!/usr/bin/env python3
"""Build source-hashed Markdown evidence tables from the experiment registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

_REGISTERED_HEADING = "## Registered experiments"
_EXPECTED_COLUMNS = (
    "ID",
    "Phase",
    "Question",
    "Primary protocol",
    "Current status",
    "Evidence or blocker",
)
_ALLOWED_STATUSES = frozenset(
    {
        "VERIFIED_CPU",
        "IMPLEMENTED_NOT_RECORDED",
        "PARTIAL_GATE",
        "PREREGISTERED_NOT_RUN",
        "BLOCKED_BY_GATE",
        "NOT_APPLICABLE",
    }
)
_JSON_REFERENCE = re.compile(r"`([^`]+\.json)`")
_JSON_BINDING = re.compile(r"`([^`]+\.json)`\s+SHA-256\s+`([0-9a-f]{64})`")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_REGISTRY_BYTES = 16 * 1024 * 1024
_MAX_FLATTENED_METRICS = 10_000
_SENSITIVE_PATH_COMPONENT = re.compile(
    r"(?:^|[.\[])"
    r"(?:api[_-]?(?:key|token)|access[_-]?token|auth[_-]?token|password|secret|private[_-]?key|email)"
    r"(?:$|[.\[])",
    re.IGNORECASE,
)
_EMAIL_VALUE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LOCAL_PATH_VALUE = re.compile(
    r"(?:/Users/[^/\s]+(?:/|$)|/home/[^/\s]+(?:/|$)|/private/tmp(?:/|$)|"
    r"/tmp(?:/|$)|/var/folders(?:/|$)|[A-Za-z]:\\Users\\[^\\\s]+(?:\\|$))"
)
_SECRET_VALUE = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|rk|ghp)[_-][A-Za-z0-9_-]{12,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_-]{12,}\b|"
    r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b|"
    r"\bAIza[0-9A-Za-z_-]{16,}\b|"
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b)"
)


class RegistryRow(NamedTuple):
    experiment_id: str
    phase: str
    question: str
    protocol: str
    status: str
    evidence: str


def _read_regular_text(path: Path, *, label: str, maximum_bytes: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if metadata.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds {maximum_bytes} byte limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(maximum_bytes + 1)
    except OSError as error:
        raise ValueError(f"{label} must be a readable regular file: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(raw) > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} byte limit")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 text") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_markdown_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        raise ValueError("registered experiment table rows must start and end with '|'")
    cells: list[str] = []
    buffer: list[str] = []
    escaped = False
    in_code = False
    for character in stripped[1:-1]:
        if escaped:
            buffer.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
            buffer.append(character)
        elif character == "`":
            in_code = not in_code
            buffer.append(character)
        elif character == "|" and not in_code:
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(character)
    if escaped:
        raise ValueError("registered experiment table contains a dangling escape")
    if in_code:
        raise ValueError("registered experiment table contains an unclosed code span")
    cells.append("".join(buffer).strip())
    return tuple(cells)


def _parse_registry(text: str) -> tuple[RegistryRow, ...]:
    lines = text.splitlines()
    try:
        heading_index = next(
            index for index, line in enumerate(lines) if line.strip() == _REGISTERED_HEADING
        )
    except StopIteration as error:
        raise ValueError("registry is missing the 'Registered experiments' section") from error

    table_lines: list[str] = []
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines and stripped:
            raise ValueError("registered experiment table is interrupted by non-table text")
    if len(table_lines) < 3:
        raise ValueError("Registered experiments must contain a header and at least one row")
    if _split_markdown_row(table_lines[0]) != _EXPECTED_COLUMNS:
        raise ValueError("Registered experiments table header does not match the required schema")
    separator = _split_markdown_row(table_lines[1])
    if len(separator) != len(_EXPECTED_COLUMNS) or any(
        not re.fullmatch(r":?-{3,}:?", cell) for cell in separator
    ):
        raise ValueError("Registered experiments table has an invalid separator row")

    rows: list[RegistryRow] = []
    identifiers: set[str] = set()
    for line in table_lines[2:]:
        cells = _split_markdown_row(line)
        if len(cells) != len(_EXPECTED_COLUMNS):
            raise ValueError("registered experiment row has the wrong number of columns")
        if any(not cell for cell in cells):
            raise ValueError("registered experiment row contains an empty required cell")
        experiment_id, phase, question, protocol, status, evidence = cells
        if experiment_id in identifiers:
            raise ValueError(f"duplicate registry experiment ID {experiment_id!r}")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"unknown registry status {status!r} for {experiment_id}")
        _reject_private_text(
            tuple((column, cell) for column, cell in zip(_EXPECTED_COLUMNS, cells, strict=True)),
            source=f"registry row {experiment_id!r}",
        )
        identifiers.add(experiment_id)
        rows.append(
            RegistryRow(
                experiment_id=experiment_id,
                phase=phase,
                question=question,
                protocol=protocol,
                status=status,
                evidence=evidence,
            )
        )
    return tuple(rows)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise ValueError(f"accepted registry metric source is missing: {path}")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"accepted registry metric source exceeds {_MAX_JSON_BYTES} bytes: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"cannot parse accepted metric source {path}: {error}") from error


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_reference(reference: str, registry: Path) -> tuple[Path, str]:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError("registry JSON references must be relative, not absolute")
    bases = (registry.parent.resolve(), Path.cwd().resolve())
    candidates: list[tuple[Path, str]] = []
    for base in bases:
        candidate = (base / relative).resolve()
        if not _is_within(candidate, base):
            continue
        if candidate.is_file():
            label = relative.as_posix() if base == Path.cwd().resolve() else candidate.name
            candidates.append((candidate, label))
    unique = {candidate for candidate, _ in candidates}
    if not unique:
        raise ValueError(f"accepted registry metric source is missing or unsafe: {reference}")
    if len(unique) > 1:
        raise ValueError(f"accepted registry metric source is ambiguous: {reference}")
    return candidates[0]


def _flatten(value: object, prefix: str = "", depth: int = 0) -> tuple[tuple[str, object], ...]:
    if depth > 20:
        raise ValueError("accepted metric JSON nesting exceeds 20 levels")
    if isinstance(value, Mapping):
        rows: list[tuple[str, object]] = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("accepted metric JSON objects must have string keys")
            child = f"{prefix}.{key}" if prefix else key
            rows.extend(_flatten(value[key], child, depth + 1))
        return tuple(rows)
    if isinstance(value, list):
        rows = []
        for index, child_value in enumerate(value):
            rows.extend(_flatten(child_value, f"{prefix}[{index}]", depth + 1))
        return tuple(rows)
    if not prefix:
        raise ValueError("accepted metric JSON root must be an object or array")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"accepted metric {prefix} must be finite")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(f"accepted metric {prefix} has unsupported type")
    return ((prefix, value),)


def _escape_markdown(value: object) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return format(value, ".12g")
    return _escape_markdown(value)


def _reject_private_metrics(
    flattened: tuple[tuple[str, object], ...],
    *,
    source: Path,
) -> None:
    for metric_path, value in flattened:
        if _SENSITIVE_PATH_COMPONENT.search(metric_path):
            raise ValueError(
                f"privacy check rejected sensitive metric path {metric_path!r} in {source}"
            )
        if isinstance(value, str):
            _reject_private_text(
                ((metric_path, value),),
                source=f"accepted metric source {source}",
            )


def _reject_private_text(
    values: tuple[tuple[str, str], ...],
    *,
    source: str,
) -> None:
    for label, value in values:
        if (
            _EMAIL_VALUE.search(value) is not None
            or _LOCAL_PATH_VALUE.search(value) is not None
            or _SECRET_VALUE.search(value) is not None
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"privacy check rejected sensitive value at {label!r} in {source}")


def _source_label(path: Path, registry: Path) -> str:
    worktree = Path.cwd().resolve()
    resolved = path.resolve()
    if _is_within(resolved, worktree):
        return resolved.relative_to(worktree).as_posix()
    if _is_within(resolved, registry.parent.resolve()):
        return resolved.relative_to(registry.parent.resolve()).as_posix()
    return resolved.name


def _command_provenance(registry: Path, output: Path, argv: Sequence[str] | None) -> str:
    del argv
    return (
        "python scripts/build_paper_tables.py --registry "
        f"{_source_label(registry, registry)} --output {_source_label(output, registry)}"
    )


def _build_report(registry: Path, output: Path, argv: Sequence[str] | None) -> str:
    registry_text = _read_regular_text(
        registry,
        label="registry",
        maximum_bytes=_MAX_REGISTRY_BYTES,
    )
    rows = _parse_registry(registry_text)
    nested_rows: list[tuple[str, str, str, object, str]] = []
    source_hashes: dict[str, str] = {_source_label(registry, registry): _sha256(registry)}
    loaded: dict[Path, tuple[tuple[str, object], ...]] = {}
    for row in rows:
        if row.status != "VERIFIED_CPU":
            continue
        references = tuple(_JSON_REFERENCE.findall(row.evidence))
        bindings = tuple(_JSON_BINDING.findall(row.evidence))
        if tuple(reference for reference, _digest in bindings) != references:
            raise ValueError(
                f"VERIFIED_CPU row {row.experiment_id!r} must bind every JSON source "
                "as `path.json` SHA-256 `<64 lowercase hex characters>`"
            )
        for reference, expected_digest in bindings:
            source_path, _ = _resolve_reference(reference, registry)
            digest = _sha256(source_path)
            if digest != expected_digest:
                raise ValueError(
                    f"accepted registry metric source SHA-256 mismatch for {reference}: "
                    f"expected {expected_digest}, observed {digest}"
                )
            flattened = loaded.get(source_path)
            if flattened is None:
                flattened = _flatten(_load_json(source_path))
                if len(flattened) > _MAX_FLATTENED_METRICS:
                    raise ValueError(
                        f"accepted metric source has more than {_MAX_FLATTENED_METRICS} leaves"
                    )
                _reject_private_metrics(flattened, source=source_path)
                loaded[source_path] = flattened
            label = _source_label(source_path, registry)
            source_hashes[label] = digest
            nested_rows.extend(
                (row.experiment_id, label, metric, value, digest) for metric, value in flattened
            )

    lines = [
        "# Registry-derived evidence tables",
        "",
        "This report is a deterministic derivation. `VERIFIED_CPU` JSON bundles are expanded; ",
        "partial, blocked, and preregistered rows are shown in status tables but are not promoted ",
        "to accepted metrics.",
        "",
        f"Command: `{_command_provenance(registry, output, argv)}`",
        "",
        "## Experiment status",
        "",
        "| ID | Phase | Status | Question | Evidence / blocker |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _escape_markdown(value)
                for value in (
                    row.experiment_id,
                    row.phase,
                    row.status,
                    row.question,
                    row.evidence,
                )
            )
            + " |"
        )
    lines.extend(["", "## Status counts", "", "| Status | Experiments |", "|---|---:|"])
    for status in sorted({row.status for row in rows}):
        lines.append(f"| `{status}` | {sum(row.status == status for row in rows)} |")
    lines.extend(["", "## Accepted nested metrics", ""])
    if nested_rows:
        lines.extend(
            [
                "| Registry ID | Source | Metric path | Value | Source SHA-256 |",
                "|---|---|---|---:|---|",
            ]
        )
        for experiment_id, source, metric, value, digest in nested_rows:
            lines.append(
                f"| {_escape_markdown(experiment_id)} | `{_escape_markdown(source)}` | "
                f"`{_escape_markdown(metric)}` | `{_render_value(value)}` | `{digest}` |"
            )
    else:
        lines.append("No `VERIFIED_CPU` row references a readable JSON metric bundle.")
    lines.extend(["", "## Source hashes", "", "| Source | SHA-256 |", "|---|---|"])
    for label, digest in sorted(source_hashes.items()):
        lines.append(f"| `{_escape_markdown(label)}` | `{digest}` |")
    return "\n".join(lines) + "\n"


def _write_new_text(path: Path, text: str) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True, help="experiment registry Markdown")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/paper_tables.md"),
        help="new Markdown report path (default: artifacts/reports/paper_tables.md)",
    )
    args = parser.parse_args(argv)
    try:
        from compbias.io.artifact_paths import validated_artifact_path

        args.output = validated_artifact_path(
            args.output,
            repository_root=Path.cwd(),
            label="paper-table output",
            suffix=".md",
        )
        if args.registry.resolve() == args.output.resolve():
            raise ValueError("output must not replace the registry")
        report = _build_report(args.registry, args.output, argv)
        _write_new_text(args.output, report)
    except (FileExistsError, OSError, TypeError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"COMPLETE: wrote registry-derived table to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
