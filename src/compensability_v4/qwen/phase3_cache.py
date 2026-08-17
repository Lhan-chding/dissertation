"""Real S5 exact-cache/full-history parity orchestration.

The artifact produced here is a measurement-validity proof for I4.  It does
not score recovery quality and never applies an empirical success threshold.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compbias.recoverability.phase_c_screen import build_family_constraints

from .cache_continuation import CachedGenerationState, build_suffix_token_ids
from .manual_generation import (
    _mapping,
    _one_dimensional_ids,
    _prepare_visual_chat_batch,
    manual_greedy_generate,
)
from .phase2_candidate import CueCondition


def _closed_messages(messages: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(MappingProxyType(dict(message)) for message in messages)


@dataclass(frozen=True, slots=True)
class CacheParityCall:
    call_id: str
    scene_id: str
    family: str
    condition: CueCondition
    cached_state: CachedGenerationState
    new_user_text: str
    suffix_token_ids: tuple[int, ...]
    full_history_messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CacheParityRecord:
    call_id: str
    scene_id: str
    family: str
    condition: CueCondition
    interface: str
    claim_family: str
    prefix_token_count: int
    suffix_token_ids: tuple[int, ...]
    image_token_positions: tuple[int, ...]
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int
    cached_generated_token_ids: tuple[int, ...]
    full_generated_token_ids: tuple[int, ...]
    cached_generated_logit_sha256: tuple[str, ...]
    full_generated_logit_sha256: tuple[str, ...]
    cached_suffix_position_ids: tuple[tuple[int, ...], ...]
    full_suffix_position_ids: tuple[tuple[int, ...], ...]
    cached_cache_position: tuple[int, ...]
    full_cache_position: tuple[int, ...]
    mrope_axes: int
    suffix_parity_verified: bool
    token_parity_verified: bool
    logit_parity_verified: bool
    mrope_parity_verified: bool
    cache_position_parity_verified: bool
    rng_seed: int
    generation_config: Mapping[str, Any]

    def to_mapping(self) -> dict[str, object]:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"condition", "generation_config"}
        }
        payload["cue_condition"] = self.condition.value
        payload["generation_config"] = dict(self.generation_config)
        return payload


def _token_ids(value: object) -> tuple[int, ...]:
    converted = _one_dimensional_ids(value)
    if not converted:
        raise RuntimeError("chat template produced no token ids")
    return converted


def build_cache_parity_plan(
    cached_states: Iterable[CachedGenerationState],
    *,
    condition_turns: Mapping[str, Mapping[CueCondition, str]],
    expected_scenes: int,
) -> tuple[CacheParityCall, ...]:
    """Build all four exact suffixes for each immutable natural-state cache."""

    states = tuple(sorted(cached_states, key=lambda item: item.sample_id))
    if (
        isinstance(expected_scenes, bool)
        or not isinstance(expected_scenes, int)
        or expected_scenes <= 0
    ):
        raise ValueError("expected_scenes must be a positive integer")
    if len(states) != expected_scenes or len({state.sample_id for state in states}) != len(states):
        raise RuntimeError("S5 cached-state scene count or identifiers drifted")
    if set(condition_turns) != {state.sample_id for state in states}:
        raise RuntimeError("S5 condition turns do not align with cached states")
    calls: list[CacheParityCall] = []
    for state in states:
        processor = state.processor
        if processor is None or not callable(getattr(processor, "apply_chat_template", None)):
            raise RuntimeError("S5 cached state lacks its original chat-template processor")
        turns = condition_turns[state.sample_id]
        if set(turns) != set(CueCondition):
            raise RuntimeError("S5 requires all four cue conditions per scene")
        family = str(getattr(state, "family", "unknown"))
        for condition in CueCondition:
            text = turns[condition]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("S5 continuation turns must be non-empty")
            full_messages = (
                *state.chat_messages,
                MappingProxyType({"role": "user", "content": text}),
            )
            full_ids = _token_ids(
                processor.apply_chat_template(
                    [dict(message) for message in full_messages],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            suffix = build_suffix_token_ids(state.token_ids, full_ids)
            calls.append(
                CacheParityCall(
                    call_id=f"{state.sample_id}.{condition.value}",
                    scene_id=state.sample_id,
                    family=family,
                    condition=condition,
                    cached_state=state,
                    new_user_text=text,
                    suffix_token_ids=suffix,
                    full_history_messages=_closed_messages(full_messages),
                )
            )
    if len(calls) != expected_scenes * len(CueCondition):
        raise RuntimeError("S5 parity-call count drifted")
    return tuple(calls)


def _clone_state(state: CachedGenerationState) -> CachedGenerationState:
    try:
        cache = copy.deepcopy(state.past_key_values)
    except Exception as error:  # pragma: no cover - depends on server cache type
        raise RuntimeError("S5 could not isolate the per-condition KV cache") from error
    return replace(state, past_key_values=cache)


def _prepare_full_history_batch(
    processor: object,
    messages: Sequence[Mapping[str, Any]],
    model: object,
) -> object:
    return _prepare_visual_chat_batch(processor, messages, model)


def _first_positions(
    result: object, *, suffix_length: int, full_history: bool
) -> tuple[tuple[int, ...], ...]:
    histories = getattr(result, "forward_position_ids", ())
    if not histories:
        raise RuntimeError("S5 runtime exposed no MRoPE position trace")
    positions = tuple(tuple(int(item) for item in axis) for axis in histories[0])
    if len(positions) != 3 or any(len(axis) < suffix_length for axis in positions):
        raise RuntimeError("S5 runtime MRoPE trace is malformed")
    if full_history:
        return tuple(axis[-suffix_length:] for axis in positions)
    if any(len(axis) != suffix_length for axis in positions):
        raise RuntimeError("S5 cached path did not process the exact suffix")
    return positions


def _first_cache_position(
    result: object, *, suffix_length: int, full_history: bool
) -> tuple[int, ...]:
    histories = getattr(result, "forward_cache_positions", ())
    if not histories:
        raise RuntimeError("S5 runtime exposed no cache_position trace")
    positions = tuple(int(item) for item in histories[0])
    if len(positions) < suffix_length:
        raise RuntimeError("S5 runtime cache_position trace is malformed")
    selected = positions[-suffix_length:] if full_history else positions
    if len(selected) != suffix_length:
        raise RuntimeError("S5 cached path did not expose one cache position per suffix token")
    return selected


def _real_cache_full_trace(
    model: object,
    processor: object,
    call: CacheParityCall,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    import torch

    state = _clone_state(call.cached_state)
    device = getattr(model, "device", None)
    suffix_ids = torch.tensor([call.suffix_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [state.attention_mask + (1,) * len(call.suffix_token_ids)],
        dtype=torch.long,
        device=device,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    cached = manual_greedy_generate(
        model,
        {"input_ids": suffix_ids, "attention_mask": attention_mask},
        max_new_tokens=max_new_tokens,
        eos_token_ids=getattr(tokenizer, "eos_token_id", None),
        rng_seed=state.rng_seed,
        generation_config=state.generation_config,
        past_key_values=state.past_key_values,
        prior_token_ids=state.token_ids,
        prior_position_ids=state.position_ids,
    )
    full_batch = _prepare_full_history_batch(processor, call.full_history_messages, model)
    full_input_ids = _token_ids(_mapping(full_batch)["input_ids"])
    if full_input_ids != state.token_ids + call.suffix_token_ids:
        raise RuntimeError("S5 full-history processor tokens differ from the exact suffix contract")
    full = manual_greedy_generate(
        model,
        full_batch,
        max_new_tokens=max_new_tokens,
        eos_token_ids=getattr(tokenizer, "eos_token_id", None),
        rng_seed=state.rng_seed,
        generation_config=state.generation_config,
    )
    suffix_length = len(call.suffix_token_ids)
    return {
        "suffix_token_ids": call.suffix_token_ids,
        "cached_generated_token_ids": cached.generated_token_ids,
        "full_generated_token_ids": full.generated_token_ids,
        "cached_generated_logits": cached.generated_logits,
        "full_generated_logits": full.generated_logits,
        "cached_suffix_position_ids": _first_positions(
            cached, suffix_length=suffix_length, full_history=False
        ),
        "full_suffix_position_ids": _first_positions(
            full, suffix_length=suffix_length, full_history=True
        ),
        "cached_cache_position": _first_cache_position(
            cached, suffix_length=suffix_length, full_history=False
        ),
        "full_cache_position": _first_cache_position(
            full, suffix_length=suffix_length, full_history=True
        ),
        "mrope_axes": 3,
    }


def _nested_floats(value: object) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        flattened = value.detach().to("cpu").reshape(-1).tolist()
        return tuple(float(item) for item in flattened)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[float] = []
        for item in value:
            result.extend(_nested_floats(item))
        return tuple(result)
    return (float(value),)


def _logit_hash(value: object) -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is required on the server
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest = hashlib.sha256()
        digest.update(json.dumps(tuple(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0float32\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        return digest.hexdigest()
    payload = json.dumps(_nested_floats(value), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _assert_logit_parity(
    cached: Sequence[object],
    full: Sequence[object],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if len(cached) != len(full):
        raise RuntimeError("S5 logit parity failed because generated lengths differ")
    for step, (left, right) in enumerate(zip(cached, full, strict=True)):
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is required on the server
            torch = None  # type: ignore[assignment]
        if torch is not None and torch.is_tensor(left) and torch.is_tensor(right):
            left_tensor = left.detach().to(device="cpu", dtype=torch.float32)
            right_tensor = right.detach().to(device="cpu", dtype=torch.float32)
            if left_tensor.shape != right_tensor.shape:
                raise RuntimeError(f"S5 generated-logit parity failed at step {step}")
            equal = (
                torch.equal(left_tensor, right_tensor)
                if absolute_tolerance == 0.0 and relative_tolerance == 0.0
                else torch.allclose(
                    left_tensor,
                    right_tensor,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    equal_nan=False,
                )
            )
            if not equal:
                raise RuntimeError(f"S5 generated-logit parity failed at step {step}")
            continue
        left_values, right_values = _nested_floats(left), _nested_floats(right)
        if len(left_values) != len(right_values) or any(
            not math.isclose(a, b, abs_tol=absolute_tolerance, rel_tol=relative_tolerance)
            for a, b in zip(left_values, right_values, strict=True)
        ):
            raise RuntimeError(f"S5 generated-logit parity failed at step {step}")


def execute_cache_parity_plan(
    model: object,
    processor: object,
    calls: Iterable[CacheParityCall],
    *,
    max_new_tokens: int,
    logit_absolute_tolerance: float,
    logit_relative_tolerance: float,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[CacheParityRecord, ...]:
    """Execute cache/full-history paths and require every objective parity invariant."""

    frozen = tuple(calls)
    if not frozen or len({call.call_id for call in frozen}) != len(frozen):
        raise ValueError("S5 parity calls must be non-empty with unique identifiers")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("S5 max_new_tokens must be a positive integer")
    tolerances = (logit_absolute_tolerance, logit_relative_tolerance)
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("S5 logit tolerances must be finite and non-negative")
    shortcut = getattr(model, "compare_cache_and_full_history", None)
    records: list[CacheParityRecord] = []
    for completed, call in enumerate(frozen, start=1):
        trace = (
            shortcut(call, processor, max_new_tokens=max_new_tokens)
            if callable(shortcut)
            else _real_cache_full_trace(
                model,
                processor,
                call,
                max_new_tokens=max_new_tokens,
            )
        )
        if tuple(trace.get("suffix_token_ids", ())) != call.suffix_token_ids:
            raise RuntimeError("S5 exact suffix parity failed")
        cached_tokens = tuple(int(item) for item in trace["cached_generated_token_ids"])
        full_tokens = tuple(int(item) for item in trace["full_generated_token_ids"])
        if cached_tokens != full_tokens:
            raise RuntimeError("S5 generated-token parity failed")
        cached_logits = tuple(trace["cached_generated_logits"])
        full_logits = tuple(trace["full_generated_logits"])
        _assert_logit_parity(
            cached_logits,
            full_logits,
            absolute_tolerance=logit_absolute_tolerance,
            relative_tolerance=logit_relative_tolerance,
        )
        cached_positions = tuple(
            tuple(int(item) for item in axis) for axis in trace["cached_suffix_position_ids"]
        )
        full_positions = tuple(
            tuple(int(item) for item in axis) for axis in trace["full_suffix_position_ids"]
        )
        if len(cached_positions) != 3 or cached_positions != full_positions:
            raise RuntimeError("S5 MRoPE parity failed")
        cached_cache = tuple(int(item) for item in trace["cached_cache_position"])
        full_cache = tuple(int(item) for item in trace["full_cache_position"])
        if cached_cache != full_cache:
            raise RuntimeError("S5 cache_position parity failed")
        records.append(
            CacheParityRecord(
                call_id=call.call_id,
                scene_id=call.scene_id,
                family=call.family,
                condition=call.condition,
                interface="I4_exact_cached_natural_continuation",
                claim_family="natural_visual_revision",
                prefix_token_count=len(call.cached_state.token_ids),
                suffix_token_ids=call.suffix_token_ids,
                image_token_positions=call.cached_state.image_token_positions,
                image_grid_thw=call.cached_state.image_grid_thw,
                visual_token_count=call.cached_state.visual_token_count,
                cached_generated_token_ids=cached_tokens,
                full_generated_token_ids=full_tokens,
                cached_generated_logit_sha256=tuple(_logit_hash(item) for item in cached_logits),
                full_generated_logit_sha256=tuple(_logit_hash(item) for item in full_logits),
                cached_suffix_position_ids=cached_positions,
                full_suffix_position_ids=full_positions,
                cached_cache_position=cached_cache,
                full_cache_position=full_cache,
                mrope_axes=int(trace["mrope_axes"]),
                suffix_parity_verified=True,
                token_parity_verified=True,
                logit_parity_verified=True,
                mrope_parity_verified=True,
                cache_position_parity_verified=True,
                rng_seed=call.cached_state.rng_seed,
                generation_config=call.cached_state.generation_config,
            )
        )
        if progress is not None:
            progress(completed, len(frozen))
    return tuple(records)


def summarize_cache_parity(records: Iterable[CacheParityRecord]) -> dict[str, object]:
    frozen = tuple(records)
    if not frozen or len({record.call_id for record in frozen}) != len(frozen):
        raise ValueError("S5 records must be non-empty with unique identifiers")
    by_scene: dict[str, set[CueCondition]] = {}
    for record in frozen:
        by_scene.setdefault(record.scene_id, set()).add(record.condition)
    if any(conditions != set(CueCondition) for conditions in by_scene.values()):
        raise RuntimeError("S5 records do not contain all four conditions per scene")
    checks = {
        "all_token_parity_verified": all(record.token_parity_verified for record in frozen),
        "all_logit_parity_verified": all(record.logit_parity_verified for record in frozen),
        "all_mrope_parity_verified": all(record.mrope_parity_verified for record in frozen),
        "all_cache_position_parity_verified": all(
            record.cache_position_parity_verified for record in frozen
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("S5 objective parity proof is incomplete")
    return {
        "schema_version": 1,
        "status": "PHASE_3_EXACT_CACHE_PARITY_VERIFIED",
        "number_of_scenes": len(by_scene),
        "number_of_parity_calls": len(frozen),
        "condition_counts": dict(
            sorted(Counter(record.condition.value for record in frozen).items())
        ),
        **checks,
        "i4_primary_eligible": True,
        "claim_family": "natural_visual_revision",
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }


def validate_cache_output_path(output: Path) -> None:
    """Reject overwrite and symlink redirection before expensive model execution."""

    if not output.is_absolute():
        raise RuntimeError("S5 output path must be absolute")
    for candidate in (output, *output.parents):
        if candidate.is_symlink():
            raise RuntimeError("S5 output path must not contain a symlink")
    if output.exists():
        raise FileExistsError("refusing to overwrite an S5 cache-parity artifact")


def write_cache_parity_outputs(
    output: Path,
    *,
    records: Iterable[CacheParityRecord],
    summary: Mapping[str, object],
) -> None:
    frozen = tuple(records)
    if not frozen:
        raise ValueError("S5 records must not be empty")
    validate_cache_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [record.to_mapping() for record in frozen],
        "summary": dict(summary),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def facts_for_condition(
    *,
    family: str,
    truth: tuple[int, int, int, int],
    observed: tuple[int, int, int, int],
    counterfactual: tuple[int, int, int, int],
    condition: CueCondition,
) -> tuple[Mapping[str, object], ...]:
    """Rebuild the S3 cue condition from frozen worlds without model output."""

    if condition is CueCondition.NO_CUE:
        return ()
    if condition is CueCondition.VALID_CUE:
        facts = build_family_constraints(family, truth)
        rows: list[Mapping[str, object]] = []
        for fact in facts:
            kind = type(fact).__name__
            if kind == "PairSumConstraint":
                row = {
                    "type": "pair_sum",
                    "left_index": fact.left_index,
                    "right_index": fact.right_index,
                    "total": fact.total,
                    "fact_id": fact.constraint_id,
                }
            elif kind == "KnownValueConstraint":
                row = {
                    "type": "known_value",
                    "index": fact.index,
                    "value": fact.value,
                    "fact_id": fact.constraint_id,
                }
            else:
                row = {
                    "type": "arithmetic_progression",
                    "indices": list(fact.indices),
                    "fact_id": fact.constraint_id,
                }
            rows.append(MappingProxyType(row))
        return tuple(rows)
    if condition is CueCondition.SHAM_CUE:
        unchanged = next(
            index
            for index, pair in enumerate(zip(truth, observed, strict=True))
            if pair[0] == pair[1]
        )
        count = len(build_family_constraints(family, truth))
        return tuple(
            MappingProxyType(
                {
                    "type": "known_value",
                    "index": unchanged,
                    "value": observed[unchanged],
                    "fact_id": f"sham-{index:02d}",
                }
            )
            for index in range(count)
        )
    return tuple(
        MappingProxyType(
            {
                "type": "known_value",
                "index": index,
                "value": value,
                "fact_id": f"counterfactual-{index}",
            }
        )
        for index, value in enumerate(counterfactual)
    )


def build_condition_turns(
    *,
    correction_prompt: str,
    family: str,
    truth: tuple[int, int, int, int],
    observed: tuple[int, int, int, int],
    counterfactual: tuple[int, int, int, int],
) -> dict[CueCondition, str]:
    if not isinstance(correction_prompt, str) or not correction_prompt.strip():
        raise ValueError("S5 correction prompt must be non-empty")
    return {
        condition: correction_prompt
        + "\n"
        + json.dumps(
            {
                "facts": [
                    dict(item)
                    for item in facts_for_condition(
                        family=family,
                        truth=truth,
                        observed=observed,
                        counterfactual=counterfactual,
                        condition=condition,
                    )
                ]
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        for condition in CueCondition
    }


__all__ = [
    "CacheParityCall",
    "CacheParityRecord",
    "build_cache_parity_plan",
    "build_condition_turns",
    "execute_cache_parity_plan",
    "facts_for_condition",
    "summarize_cache_parity",
    "validate_cache_output_path",
    "write_cache_parity_outputs",
]
