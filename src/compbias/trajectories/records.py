"""Closed immutable schemas for v2 natural-mediator evidence."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal


def _identifier(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty string of at most {maximum} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return number


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _freeze_json(value: object, *, depth: int = 0) -> object:
    if depth > 64:
        raise ValueError("record payload exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("record payload numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("record payload mapping keys must be strings")
            frozen[key] = _freeze_json(item, depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, depth=depth + 1) for item in value)
    raise TypeError("record payloads must be JSON-compatible")


def _detach_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _detach_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detach_json(item) for item in value]
    return copy.deepcopy(value)


def _relative_image_path(value: object) -> str:
    path = _identifier(value, "image_path", maximum=1024)
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise ValueError("image_path must be a safe repository-relative POSIX path")
    return path


@dataclass(frozen=True, slots=True)
class NaturalMediatorRecord:
    mediator_record_id: str
    sample_id: str
    checkpoint_id: str
    interface_id: str
    rollout_id: str
    image_path: str
    question: str
    gold_scene: Mapping[str, object]
    natural_mediator_raw: str
    natural_mediator_parsed: Mapping[str, object]
    parser_confidence: float
    parse_failed: bool
    error_type: str
    task_severity: float
    original_answer: str
    original_reward: int
    rng_seed: int
    activation_ref: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "mediator_record_id",
            "sample_id",
            "checkpoint_id",
            "interface_id",
            "rollout_id",
            "question",
            "natural_mediator_raw",
            "error_type",
            "original_answer",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, maximum=4096))
        object.__setattr__(self, "image_path", _relative_image_path(self.image_path))
        object.__setattr__(self, "gold_scene", _freeze_json(self.gold_scene))
        object.__setattr__(
            self, "natural_mediator_parsed", _freeze_json(self.natural_mediator_parsed)
        )
        object.__setattr__(
            self, "parser_confidence", _unit(self.parser_confidence, "parser_confidence")
        )
        if not isinstance(self.parse_failed, bool):
            raise TypeError("parse_failed must be boolean")
        object.__setattr__(self, "task_severity", _nonnegative(self.task_severity, "task_severity"))
        if self.original_reward not in {0, 1} or isinstance(self.original_reward, bool):
            raise ValueError("original_reward must be binary")
        _integer(self.rng_seed, "rng_seed")
        if self.activation_ref is not None:
            _identifier(self.activation_ref, "activation_ref", maximum=1024)

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_kind": "natural",
            "mediator_record_id": self.mediator_record_id,
            "sample_id": self.sample_id,
            "checkpoint_id": self.checkpoint_id,
            "interface_id": self.interface_id,
            "rollout_id": self.rollout_id,
            "image_path": self.image_path,
            "question": self.question,
            "gold_scene": _detach_json(self.gold_scene),
            "natural_mediator_raw": self.natural_mediator_raw,
            "natural_mediator_parsed": _detach_json(self.natural_mediator_parsed),
            "parser_confidence": self.parser_confidence,
            "parse_failed": self.parse_failed,
            "error_type": self.error_type,
            "task_severity": self.task_severity,
            "original_answer": self.original_answer,
            "original_reward": self.original_reward,
            "rng_seed": self.rng_seed,
            "activation_ref": self.activation_ref,
        }


@dataclass(frozen=True, slots=True)
class ForkedContinuationRecord:
    mediator_record_id: str
    fork_id: str
    image_cut_mode: str
    continuation_seed: int
    answer: str
    reward: int
    replay_fidelity_metadata: Mapping[str, object]
    source_kind: Literal["natural", "synthetic"]

    def __post_init__(self) -> None:
        for name in ("mediator_record_id", "fork_id", "image_cut_mode", "answer"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, maximum=4096))
        _integer(self.continuation_seed, "continuation_seed")
        if self.reward not in {0, 1} or isinstance(self.reward, bool):
            raise ValueError("reward must be binary")
        if self.source_kind not in {"natural", "synthetic"}:
            raise ValueError("source_kind must be natural or synthetic")
        object.__setattr__(
            self, "replay_fidelity_metadata", _freeze_json(self.replay_fidelity_metadata)
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "mediator_record_id": self.mediator_record_id,
            "fork_id": self.fork_id,
            "image_cut_mode": self.image_cut_mode,
            "continuation_seed": self.continuation_seed,
            "answer": self.answer,
            "reward": self.reward,
            "replay_fidelity_metadata": _detach_json(self.replay_fidelity_metadata),
            "source_kind": self.source_kind,
        }


@dataclass(frozen=True, slots=True)
class SyntheticMediatorRecord:
    mediator_record_id: str
    sample_id: str
    checkpoint_id: str
    interface_id: str
    target_error_type: str
    construction_method: str
    synthetic_mediator: Mapping[str, object]
    nearest_natural_state_ids: tuple[str, ...]
    transport_signature: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "mediator_record_id",
            "sample_id",
            "checkpoint_id",
            "interface_id",
            "target_error_type",
            "construction_method",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "synthetic_mediator", _freeze_json(self.synthetic_mediator))
        neighbours = tuple(
            _identifier(value, "nearest_natural_state_id")
            for value in self.nearest_natural_state_ids
        )
        if not neighbours:
            raise ValueError("nearest_natural_state_ids must not be empty")
        object.__setattr__(self, "nearest_natural_state_ids", neighbours)
        signature = tuple(
            _finite(value, "transport_signature") for value in self.transport_signature
        )
        if not signature:
            raise ValueError("transport_signature must not be empty")
        object.__setattr__(self, "transport_signature", signature)

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_kind": "synthetic",
            "mediator_record_id": self.mediator_record_id,
            "sample_id": self.sample_id,
            "checkpoint_id": self.checkpoint_id,
            "interface_id": self.interface_id,
            "target_error_type": self.target_error_type,
            "construction_method": self.construction_method,
            "synthetic_mediator": _detach_json(self.synthetic_mediator),
            "nearest_natural_state_ids": list(self.nearest_natural_state_ids),
            "transport_signature": list(self.transport_signature),
        }


@dataclass(frozen=True, slots=True)
class CrossedRiskRecord:
    sample_id: str
    interface_id: str
    perception_source: Literal["model", "oracle"]
    reasoner_source: Literal["model", "oracle"]
    loss: float
    reward: float
    seed: int

    def __post_init__(self) -> None:
        _identifier(self.sample_id, "sample_id")
        _identifier(self.interface_id, "interface_id")
        if self.perception_source not in {"model", "oracle"}:
            raise ValueError("perception_source must be model or oracle")
        if self.reasoner_source not in {"model", "oracle"}:
            raise ValueError("reasoner_source must be model or oracle")
        object.__setattr__(self, "loss", _unit(self.loss, "loss"))
        object.__setattr__(self, "reward", _unit(self.reward, "reward"))
        _integer(self.seed, "seed")


@dataclass(frozen=True, slots=True)
class CheckpointDistributionRecord:
    sample_id: str
    checkpoint_from: str
    checkpoint_to: str
    interface_id: str
    error_type: str
    prob_from: float
    prob_to: float
    selection_ratio: float | None
    new_support: bool

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "checkpoint_from",
            "checkpoint_to",
            "interface_id",
            "error_type",
        ):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "prob_from", _unit(self.prob_from, "prob_from"))
        object.__setattr__(self, "prob_to", _unit(self.prob_to, "prob_to"))
        if not isinstance(self.new_support, bool):
            raise TypeError("new_support must be boolean")
        if self.new_support:
            if self.prob_from != 0.0 or self.prob_to <= 0.0 or self.selection_ratio is not None:
                raise ValueError("new support requires prob_from=0, prob_to>0, and no ratio")
        else:
            if self.prob_from <= 0.0 or self.selection_ratio is None:
                raise ValueError("common support requires prob_from>0 and a ratio")
            ratio = _nonnegative(self.selection_ratio, "selection_ratio")
            if not math.isclose(ratio, self.prob_to / self.prob_from, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("selection_ratio must equal prob_to / prob_from")
