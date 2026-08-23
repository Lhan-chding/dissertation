"""Pre-registered P0/P1/P2 parsers for Study C3 action-channel audits."""

from __future__ import annotations

import re
from collections.abc import Callable

World = tuple[int, int, int, int]

_INTEGER = r"-?\d+"
_P0 = re.compile(rf"({_INTEGER}),({_INTEGER}),({_INTEGER}),({_INTEGER})")
_CSV = re.compile(
    rf"\s*({_INTEGER})\s*,\s*({_INTEGER})\s*,\s*({_INTEGER})\s*,\s*({_INTEGER})\s*"
)
_BRACKETED = re.compile(
    rf"\s*[\[(]\s*({_INTEGER})\s*,\s*({_INTEGER})\s*,\s*({_INTEGER})\s*,\s*"
    rf"({_INTEGER})\s*[\])]\s*"
)
_LABEL_EQUALS = re.compile(
    rf"\s*v1\s*=\s*({_INTEGER})\s*,\s*v2\s*=\s*({_INTEGER})\s*,\s*"
    rf"v3\s*=\s*({_INTEGER})\s*,\s*v4\s*=\s*({_INTEGER})\s*",
    re.IGNORECASE,
)
_LABEL_COLON = re.compile(
    rf"\s*v1\s*:\s*({_INTEGER})\s+v2\s*:\s*({_INTEGER})\s+"
    rf"v3\s*:\s*({_INTEGER})\s+v4\s*:\s*({_INTEGER})\s*",
    re.IGNORECASE,
)
_FULL_PATTERNS = (_CSV, _BRACKETED, _LABEL_EQUALS, _LABEL_COLON)
_EXPLICIT_SEGMENTS = (
    re.compile(rf"[\[(]\s*{_INTEGER}(?:\s*,\s*{_INTEGER}){{3}}\s*[\])]"),
    re.compile(
        rf"v1\s*=\s*{_INTEGER}\s*,\s*v2\s*=\s*{_INTEGER}\s*,\s*"
        rf"v3\s*=\s*{_INTEGER}\s*,\s*v4\s*=\s*{_INTEGER}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"v1\s*:\s*{_INTEGER}\s+v2\s*:\s*{_INTEGER}\s+"
        rf"v3\s*:\s*{_INTEGER}\s+v4\s*:\s*{_INTEGER}",
        re.IGNORECASE,
    ),
)


def _world(match: re.Match[str]) -> World:
    values = tuple(int(value) for value in match.groups())
    if len(values) != 4:  # pragma: no cover - all registered regexes have four groups
        raise RuntimeError("registered Study C3 parser has the wrong arity")
    return values  # type: ignore[return-value]


def _in_domain(world: World, *, minimum: int = 2, maximum: int = 18) -> bool:
    return all(minimum <= value <= maximum for value in world)


def _fullmatch(text: str) -> World | None:
    for pattern in _FULL_PATTERNS:
        match = pattern.fullmatch(text)
        if match is not None:
            return _world(match)
    return None


def parse_p0(completion: str, *, minimum: int = 2, maximum: int = 18) -> World | None:
    """Reproduce Study C2's canonical, anchored first-line parser exactly."""

    if not isinstance(completion, str) or not completion:
        return None
    first_line = completion.splitlines()[0] if completion.splitlines() else completion
    match = _P0.fullmatch(first_line.strip())
    if match is None:
        return None
    world = _world(match)
    return world if _in_domain(world, minimum=minimum, maximum=maximum) else None


def parse_complete_action_any_domain(completion: str) -> World | None:
    """Parse only a complete registered action form without applying the domain gate."""

    if not isinstance(completion, str):
        return None
    return _fullmatch(completion.strip())


def parse_p1(completion: str, *, minimum: int = 2, maximum: int = 18) -> World | None:
    """Parse a full canonical-semantic action; never mine arbitrary prose."""

    world = parse_complete_action_any_domain(completion)
    return (
        world
        if world is not None and _in_domain(world, minimum=minimum, maximum=maximum)
        else None
    )


def parse_p2(completion: str, *, minimum: int = 2, maximum: int = 18) -> World | None:
    """Parse the first line or first explicit, self-contained action segment only."""

    if not isinstance(completion, str) or not completion:
        return None
    first_line = completion.splitlines()[0]
    direct = parse_p1(first_line, minimum=minimum, maximum=maximum)
    if direct is not None:
        return direct
    for pattern in _EXPLICIT_SEGMENTS:
        match = pattern.search(first_line)
        if match is None:
            continue
        candidate = parse_complete_action_any_domain(match.group(0))
        if candidate is not None and _in_domain(candidate, minimum=minimum, maximum=maximum):
            return candidate
    return None


PARSERS: dict[str, Callable[[str], World | None]] = {
    "P0": parse_p0,
    "P1": parse_p1,
    "P2": parse_p2,
}

__all__ = [
    "PARSERS",
    "World",
    "parse_complete_action_any_domain",
    "parse_p0",
    "parse_p1",
    "parse_p2",
]
