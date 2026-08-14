"""Strict parser that preserves every structured-rollout failure as data."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class ParseStatus(str, Enum):
    """Closed result states for rollout parsing."""

    OK = "ok"
    INVALID_JSON = "invalid_json"
    MISSING_FIELD = "missing_field"
    MALFORMED = "malformed"
    INVALID_TYPE = "invalid_type"


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _detach(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _detach(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach(item) for item in value]
    return value


def _normalize_json_evidence(value: object) -> object:
    """Freeze JSON-compatible evidence or retain only safe type metadata."""

    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
        normalized = json.loads(encoded)
    except (RecursionError, TypeError, ValueError):
        normalized = {"non_string_type": type(value).__name__}
    return _freeze(normalized)


@dataclass(frozen=True)
class ParseResult:
    """One immutable result for one input rollout, successful or otherwise."""

    status: ParseStatus
    sample_id: str
    raw_text: object
    perceived_scene: Mapping[str, object] | None = None
    reasoning_action: Mapping[str, object] | None = None
    answer: object = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        object.__setattr__(self, "status", ParseStatus(self.status))
        object.__setattr__(
            self,
            "raw_text",
            self.raw_text
            if isinstance(self.raw_text, str)
            else _normalize_json_evidence(self.raw_text),
        )
        if self.perceived_scene is not None:
            if not isinstance(self.perceived_scene, Mapping):
                raise TypeError("perceived_scene must be a mapping")
            object.__setattr__(self, "perceived_scene", _freeze(self.perceived_scene))
        if self.reasoning_action is not None:
            if not isinstance(self.reasoning_action, Mapping):
                raise TypeError("reasoning_action must be a mapping")
            object.__setattr__(self, "reasoning_action", _freeze(self.reasoning_action))
        object.__setattr__(self, "answer", _freeze(self.answer))

    def to_mapping(self) -> dict[str, object]:
        """Return a detached JSON-compatible representation."""

        return {
            "status": self.status.value,
            "sample_id": self.sample_id,
            "raw_text": _detach(self.raw_text),
            "perceived_scene": _detach(self.perceived_scene),
            "reasoning_action": _detach(self.reasoning_action),
            "answer": _detach(self.answer),
            "error_code": self.error_code,
        }


def _failure(status: ParseStatus, sample_id: str, raw_text: object, error_code: str) -> ParseResult:
    return ParseResult(
        status=status,
        sample_id=sample_id,
        raw_text=raw_text,
        error_code=error_code,
    )


_TAGS = ("perception", "reasoning", "answer")
_MAX_RAW_BYTES = 65_536
_MAX_JSON_DEPTH = 32


def _reject_constant(token: str) -> object:
    raise ValueError(f"non-standard JSON constant: {token}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return 0


def _strict_json(text: str) -> object:
    decoded = json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if _json_depth(decoded) > _MAX_JSON_DEPTH:
        raise ValueError("JSON nesting depth exceeds the parser limit")
    return decoded


def parse_trajectory(raw_text: str, *, sample_id: str) -> ParseResult:
    """Parse exactly three ordered JSON sections without raising on model output."""

    if not isinstance(raw_text, str):
        return _failure(ParseStatus.INVALID_TYPE, sample_id, raw_text, "raw_text_not_string")
    try:
        raw_size = len(raw_text.encode("utf-8"))
    except UnicodeEncodeError:
        return _failure(ParseStatus.MALFORMED, sample_id, raw_text, "raw_text_invalid_unicode")
    if raw_size > _MAX_RAW_BYTES:
        return _failure(ParseStatus.MALFORMED, sample_id, raw_text, "raw_text_too_large")
    sections: dict[str, tuple[str, tuple[int, int]]] = {}
    for tag in _TAGS:
        matches = tuple(re.finditer(rf"<{tag}>(.*?)</{tag}>", raw_text, flags=re.DOTALL))
        if len(matches) > 1:
            return _failure(ParseStatus.MALFORMED, sample_id, raw_text, f"duplicate_{tag}")
        if len(matches) == 1:
            match = matches[0]
            sections[tag] = (match.group(1), match.span())

    if not sections:
        return _failure(ParseStatus.MALFORMED, sample_id, raw_text, "missing_structured_tags")
    for tag in _TAGS:
        if tag not in sections:
            return _failure(ParseStatus.MISSING_FIELD, sample_id, raw_text, f"missing_{tag}")

    spans = tuple(sections[tag][1] for tag in _TAGS)
    if not (spans[0][0] < spans[1][0] < spans[2][0]):
        return _failure(ParseStatus.MALFORMED, sample_id, raw_text, "invalid_tag_order")
    outside = raw_text[: spans[0][0]]
    outside += raw_text[spans[0][1] : spans[1][0]]
    outside += raw_text[spans[1][1] : spans[2][0]]
    outside += raw_text[spans[2][1] :]
    if outside.strip():
        return _failure(ParseStatus.MALFORMED, sample_id, raw_text, "unexpected_text")

    decoded: dict[str, object] = {}
    for tag in _TAGS:
        try:
            decoded[tag] = _strict_json(sections[tag][0])
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            return _failure(ParseStatus.INVALID_JSON, sample_id, raw_text, f"invalid_{tag}_json")
    for tag in ("perception", "reasoning"):
        if not isinstance(decoded[tag], Mapping):
            return _failure(ParseStatus.INVALID_TYPE, sample_id, raw_text, f"{tag}_not_object")
    if decoded["answer"] is None:
        return _failure(ParseStatus.INVALID_TYPE, sample_id, raw_text, "answer_is_null")
    return ParseResult(
        status=ParseStatus.OK,
        sample_id=sample_id,
        raw_text=raw_text,
        perceived_scene=decoded["perception"],  # type: ignore[arg-type]
        reasoning_action=decoded["reasoning"],  # type: ignore[arg-type]
        answer=decoded["answer"],
    )


def parse_many(records: Iterable[Mapping[str, object]]) -> tuple[ParseResult, ...]:
    """Return exactly one ordered parse result for each input record."""

    results: list[ParseResult] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each parse record must be a mapping")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("each parse record requires a non-empty sample_id")
        results.append(parse_trajectory(record.get("raw_text"), sample_id=sample_id))  # type: ignore[arg-type]
    return tuple(results)
