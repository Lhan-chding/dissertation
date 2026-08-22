"""Observable first-line world actions and read-only legacy parser audit."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .rewards import classify_completion

_WORLD_LINE = re.compile(r"(-?\d+),(-?\d+),(-?\d+),(-?\d+)")


def _parse_line(line: str) -> tuple[int, int, int, int] | None:
    match = _WORLD_LINE.fullmatch(line.strip())
    if match is None:
        return None
    world = tuple(int(value) for value in match.groups())
    if any(value < 2 or value > 18 for value in world):
        return None
    return world  # type: ignore[return-value]


def parse_first_world_tuple(completion: str) -> tuple[int, int, int, int] | None:
    """Audit-only syntactic first-line tuple, before the action-domain gate."""

    if not isinstance(completion, str) or not completion:
        return None
    first_line = completion.splitlines()[0] if completion.splitlines() else completion
    match = _WORLD_LINE.fullmatch(first_line.strip())
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def parse_first_world_action(completion: str) -> tuple[int, int, int, int] | None:
    """Parse only the anchored first line; never search later text."""

    if not isinstance(completion, str) or not completion:
        return None
    parsed = parse_first_world_tuple(completion)
    if parsed is None or any(value < 2 or value > 18 for value in parsed):
        return None
    return parsed


def parse_legacy_exact_world(completion: str) -> tuple[int, int, int, int] | None:
    """Reproduce the strict full-completion parser for post-hoc audit only."""

    if not isinstance(completion, str):
        return None
    return _parse_line(completion.strip())


def audit_action_censoring(
    completions: Sequence[str],
    *,
    truth: tuple[int, int, int, int],
    operation: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Compare legacy observability with the frozen first-line action protocol."""

    rows: list[dict[str, object]] = []
    for index, completion in enumerate(completions):
        classified = classify_completion(completion, truth=truth, operation=operation)
        legacy = parse_legacy_exact_world(completion)
        first = parse_first_world_action(completion)
        rows.append(
            {
                "index": index,
                "completion": completion,
                "legacy_parse_success": legacy is not None,
                "first_line_parse_success": first is not None,
                "first_line_kind": classified.kind.value,
                "informatively_censored": legacy is None and classified.kind.value in {"X", "S"},
            }
        )
    return tuple(rows)


__all__ = [
    "audit_action_censoring",
    "parse_first_world_action",
    "parse_first_world_tuple",
    "parse_legacy_exact_world",
]
