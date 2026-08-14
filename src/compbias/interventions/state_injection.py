"""Image-free do-state interventions with rollout-level provenance."""

from __future__ import annotations

import copy
import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from compbias.envs.cva_world.corruptions import apply_error
from compbias.models.structured_parser import ParseStatus


class StateReasoner(Protocol):
    """Minimal reasoner boundary used by state-injection experiments."""

    def generate(
        self,
        perceived_state: Mapping[str, object],
        question: Mapping[str, object],
        *,
        seed: int,
    ) -> str: ...


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _detach(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _detach(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach(item) for item in value]
    if isinstance(value, (set, frozenset)):
        detached = [_detach(item) for item in value]
        try:
            return sorted(
                detached,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("set-valued intervention evidence must be JSON compatible") from error
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """One retained rollout under one explicit perceived-state intervention."""

    sample_id: str
    error_id: str
    error_family: str
    severity: float
    error_parameters: Mapping[str, object]
    model_id: str
    checkpoint: str
    rollout_seed: int
    perceived_state: Mapping[str, object]
    question: Mapping[str, object]
    canonical_answer: object
    raw_output: str
    parsed: object
    reward: float
    view: str = "interventional"

    def __post_init__(self) -> None:
        for name in ("sample_id", "error_id", "error_family", "model_id", "checkpoint"):
            _validate_identifier(getattr(self, name), name)
        if isinstance(self.rollout_seed, bool) or not isinstance(self.rollout_seed, int):
            raise TypeError("rollout_seed must be an integer")
        if (
            isinstance(self.severity, bool)
            or not isinstance(self.severity, (int, float))
            or not math.isfinite(float(self.severity))
            or float(self.severity) < 0.0
        ):
            raise ValueError("severity must be a non-negative finite number")
        object.__setattr__(self, "severity", float(self.severity))
        if not isinstance(self.raw_output, str):
            raise TypeError("raw_output must be a string")
        if self.reward not in (0.0, 1.0) or isinstance(self.reward, bool):
            raise ValueError("reward must be a binary outcome in {0.0, 1.0}")
        object.__setattr__(self, "reward", float(self.reward))
        if self.view != "interventional":
            raise ValueError("view must be exactly 'interventional'")
        object.__setattr__(self, "error_parameters", _freeze(self.error_parameters))
        object.__setattr__(self, "perceived_state", _freeze(self.perceived_state))
        object.__setattr__(self, "question", _freeze(self.question))
        object.__setattr__(self, "canonical_answer", _freeze(self.canonical_answer))
        parsed = self.parsed
        to_mapping = getattr(parsed, "to_mapping", None)
        if callable(to_mapping):
            parsed = to_mapping()
        object.__setattr__(self, "parsed", _freeze(parsed))

    def to_mapping(self) -> dict[str, object]:
        """Return a detached serialization-friendly record."""

        parsed = _detach(self.parsed)
        return {
            "sample_id": self.sample_id,
            "error_id": self.error_id,
            "error_family": self.error_family,
            "severity": self.severity,
            "error_parameters": _detach(self.error_parameters),
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "rollout_seed": self.rollout_seed,
            "perceived_state": _detach(self.perceived_state),
            "question": _detach(self.question),
            "canonical_answer": _detach(self.canonical_answer),
            "raw_output": self.raw_output,
            "parsed": parsed,
            "reward": self.reward,
            "view": self.view,
        }


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_seeds(rollout_seeds: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(rollout_seeds)
    if not seeds:
        raise ValueError("rollout_seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("each rollout_seed must be an integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("rollout_seeds must be unique")
    return seeds


def _parse(parser: Callable[..., object], raw_output: str, sample_id: str) -> object:
    try:
        parameters = inspect.signature(parser).parameters
    except (TypeError, ValueError):
        # Some extension callables do not expose a signature. Call once using
        # the full parser contract and preserve any internal exception.
        return parser(raw_output, sample_id=sample_id)

    sample_parameter = parameters.get("sample_id")
    if sample_parameter is not None:
        if sample_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            return parser(raw_output, sample_id)
        return parser(raw_output, sample_id=sample_id)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return parser(raw_output, sample_id=sample_id)
    return parser(raw_output)


def _parse_succeeded(parsed: object) -> bool:
    status = getattr(parsed, "status", None)
    if status is ParseStatus.OK:
        return True
    return getattr(status, "value", status) == getattr(ParseStatus.OK, "value", "ok")


def run_state_injection(
    samples: Iterable[object],
    *,
    reasoner: StateReasoner,
    parser: Callable[..., object],
    verifier: Callable[[object, object], float],
    model_id: str,
    checkpoint: str,
    rollout_seeds: Iterable[int],
    image: None = None,
) -> tuple[InterventionRecord, ...]:
    """Run every sample/error/seed combination using only the injected state.

    ``image`` is deliberately present as a guardrail: any non-``None`` value is
    rejected before samples are consumed or the reasoner is called.
    """

    if image is not None:
        raise ValueError("image must be None for do-state injection")
    if not callable(getattr(reasoner, "generate", None)):
        raise TypeError("reasoner must define a callable generate method")
    if not callable(parser) or not callable(verifier):
        raise TypeError("parser and verifier must be callable")

    checked_model_id = _validate_identifier(model_id, "model_id")
    checked_checkpoint = _validate_identifier(checkpoint, "checkpoint")
    seeds = _validate_seeds(rollout_seeds)
    records: list[InterventionRecord] = []

    for sample in tuple(samples):
        sample_id = _validate_identifier(getattr(sample, "sample_id", None), "sample_id")
        scene = getattr(sample, "scene", None)
        question = getattr(sample, "question", None)
        errors = getattr(sample, "error_catalog", None)
        if not isinstance(scene, Mapping) or not isinstance(question, Mapping):
            raise TypeError("each sample must expose mapping-valued scene and question")
        if errors is None:
            raise TypeError("each sample must expose an error_catalog")

        for error in tuple(errors):
            perceived_state = apply_error(scene, error)
            if not isinstance(perceived_state, Mapping):
                raise TypeError("apply_error must return a perceived-state mapping")
            frozen_state = _freeze(perceived_state)
            frozen_question = _freeze(question)

            for seed in seeds:
                raw_output = reasoner.generate(
                    _detach(frozen_state),
                    _detach(frozen_question),
                    seed=seed,
                )
                if not isinstance(raw_output, str):
                    raise TypeError("reasoner raw output must be a string")
                parsed = _parse(parser, raw_output, sample_id)
                reward = 0.0
                if _parse_succeeded(parsed):
                    reward = float(verifier(parsed, sample.canonical_answer))
                    if not math.isfinite(reward) or reward not in (0.0, 1.0):
                        raise ValueError("verifier reward must be a binary outcome in {0.0, 1.0}")

                records.append(
                    InterventionRecord(
                        sample_id=sample_id,
                        error_id=_validate_identifier(getattr(error, "error_id", None), "error_id"),
                        error_family=_validate_identifier(
                            getattr(error, "family", None), "error family"
                        ),
                        severity=float(error.severity),
                        error_parameters=error.parameters,
                        model_id=checked_model_id,
                        checkpoint=checked_checkpoint,
                        rollout_seed=seed,
                        perceived_state=frozen_state,
                        question=frozen_question,
                        canonical_answer=sample.canonical_answer,
                        raw_output=raw_output,
                        parsed=parsed,
                        reward=reward,
                    )
                )

    return tuple(records)
