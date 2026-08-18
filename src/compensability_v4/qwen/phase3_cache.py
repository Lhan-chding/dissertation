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
class LogitStepEvidence:
    step_index: int
    cached_shape: tuple[int, ...]
    full_shape: tuple[int, ...]
    cached_dtype: str
    full_dtype: str
    realized_token_id: int
    cached_argmax_token_id: int
    full_argmax_token_id: int
    cached_realized_token_logit: float
    full_realized_token_logit: float
    realized_token_logit_delta: float
    max_abs_diff: float
    max_rel_diff: float
    nonzero_count: int
    l2_diff: float
    argmax_abs_diff_token_id: int
    exact_identity: bool
    decision_parity_verified: bool

    def to_mapping(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


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
    decision_parity_verified: bool
    logit_parity_verified: bool
    logit_step_evidence: tuple[LogitStepEvidence, ...]
    mrope_parity_verified: bool
    cache_position_parity_verified: bool
    rng_seed: int
    generation_config: Mapping[str, Any]

    def to_mapping(self) -> dict[str, object]:
        payload = {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field not in {"condition", "generation_config", "logit_step_evidence"}
        }
        payload["cue_condition"] = self.condition.value
        payload["generation_config"] = dict(self.generation_config)
        payload["logit_step_evidence"] = [
            evidence.to_mapping() for evidence in self.logit_step_evidence
        ]
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


def _explain_logit_parity(
    cached: Sequence[object],
    full: Sequence[object],
    generated_tokens: Sequence[int],
    *,
    atol: float,
    rtol: float,
) -> tuple[LogitStepEvidence, ...]:
    """Validate exact greedy decisions and disclose every full-logit difference."""

    import torch

    tolerances = (atol, rtol)
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("S5 logit tolerances must be finite and non-negative")
    if len(cached) != len(full) or len(cached) != len(generated_tokens):
        raise RuntimeError("S5 logit evidence lengths do not match generated tokens")
    evidence: list[LogitStepEvidence] = []
    for step, (cached_value, full_value, realized_token) in enumerate(
        zip(cached, full, generated_tokens, strict=True)
    ):
        if not torch.is_tensor(cached_value) or not torch.is_tensor(full_value):
            raise RuntimeError(f"S5 logit evidence at step={step} must contain tensors")
        cached_tensor = cached_value.detach().to(device="cpu", dtype=torch.float32)
        full_tensor = full_value.detach().to(device="cpu", dtype=torch.float32)
        cached_shape = tuple(cached_tensor.shape)
        full_shape = tuple(full_tensor.shape)
        if cached_shape != full_shape or len(cached_shape) != 2 or cached_shape[0] != 1:
            raise RuntimeError(
                f"S5 logit shape parity failed at step={step}: "
                f"cached_shape={cached_shape}, full_shape={full_shape}"
            )
        if not torch.isfinite(cached_tensor).all() or not torch.isfinite(full_tensor).all():
            raise RuntimeError(f"S5 logit tensors must be finite at step={step}")
        token_id = int(realized_token)
        vocabulary_size = cached_shape[1]
        if token_id < 0 or token_id >= vocabulary_size:
            raise RuntimeError(f"S5 realized token is outside the vocabulary at step={step}")
        cached_argmax = int(torch.argmax(cached_tensor, dim=-1).item())
        full_argmax = int(torch.argmax(full_tensor, dim=-1).item())
        if cached_argmax != full_argmax:
            raise RuntimeError(
                f"S5 stepwise argmax parity failed at step={step}: "
                f"cached_argmax={cached_argmax}, full_argmax={full_argmax}"
            )
        if cached_argmax != token_id:
            raise RuntimeError(
                f"S5 realized-token top-1 parity failed at step={step}: "
                f"realized_token={token_id}, argmax={cached_argmax}"
            )
        absolute = torch.abs(cached_tensor - full_tensor)
        scale = torch.maximum(torch.abs(cached_tensor), torch.abs(full_tensor))
        relative = torch.where(scale > 0, absolute / scale, torch.zeros_like(absolute))
        flat_max_index = int(torch.argmax(absolute).item()) if absolute.numel() else 0
        exact_identity = (
            torch.equal(cached_tensor, full_tensor)
            if atol == 0.0 and rtol == 0.0
            else torch.allclose(
                cached_tensor,
                full_tensor,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
        )
        cached_realized = float(cached_tensor[0, token_id].item())
        full_realized = float(full_tensor[0, token_id].item())
        evidence.append(
            LogitStepEvidence(
                step_index=step,
                cached_shape=cached_shape,
                full_shape=full_shape,
                cached_dtype=str(cached_value.dtype),
                full_dtype=str(full_value.dtype),
                realized_token_id=token_id,
                cached_argmax_token_id=cached_argmax,
                full_argmax_token_id=full_argmax,
                cached_realized_token_logit=cached_realized,
                full_realized_token_logit=full_realized,
                realized_token_logit_delta=full_realized - cached_realized,
                max_abs_diff=float(torch.max(absolute).item()),
                max_rel_diff=float(torch.max(relative).item()),
                nonzero_count=int(torch.count_nonzero(absolute).item()),
                l2_diff=float(torch.linalg.vector_norm(absolute).item()),
                argmax_abs_diff_token_id=flat_max_index % vocabulary_size,
                exact_identity=bool(exact_identity),
                decision_parity_verified=True,
            )
        )
    return tuple(evidence)


def _token_divergence_message(
    call: CacheParityCall,
    cached_tokens: tuple[int, ...],
    full_tokens: tuple[int, ...],
    cached_logits: Sequence[object],
    full_logits: Sequence[object],
) -> str:
    """Describe the first greedy-token divergence without relaxing the gate."""

    import torch

    common_prefix_length = 0
    for cached_token, full_token in zip(cached_tokens, full_tokens, strict=False):
        if cached_token != full_token:
            break
        common_prefix_length += 1
    step = common_prefix_length
    cached_token = cached_tokens[step] if step < len(cached_tokens) else None
    full_token = full_tokens[step] if step < len(full_tokens) else None
    cached_value = cached_logits[step] if step < len(cached_logits) else None
    full_value = full_logits[step] if step < len(full_logits) else None

    def tensor_evidence(value: object, token_id: int | None) -> tuple[object, ...]:
        if not torch.is_tensor(value):
            return (None, type(value).__name__, None, None, None)
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        shape = tuple(tensor.shape)
        finite = bool(torch.isfinite(tensor).all())
        if tensor.numel() == 0:
            return (shape, str(value.dtype), finite, None, None)
        argmax = int(torch.argmax(tensor).item())
        realized = None
        if token_id is not None and len(shape) == 2 and shape[0] == 1 and 0 <= token_id < shape[1]:
            realized = float(tensor[0, token_id].item())
        return (shape, str(value.dtype), finite, argmax, realized)

    cached_shape, cached_dtype, cached_finite, cached_argmax, cached_realized = tensor_evidence(
        cached_value, cached_token
    )
    full_shape, full_dtype, full_finite, full_argmax, full_realized = tensor_evidence(
        full_value, full_token
    )
    max_abs_diff = max_rel_diff = nonzero_count = l2_diff = max_diff_token = None
    if (
        torch.is_tensor(cached_value)
        and torch.is_tensor(full_value)
        and cached_shape == full_shape
        and cached_finite
        and full_finite
    ):
        cached_tensor = cached_value.detach().to(device="cpu", dtype=torch.float32)
        full_tensor = full_value.detach().to(device="cpu", dtype=torch.float32)
        absolute = torch.abs(cached_tensor - full_tensor)
        scale = torch.maximum(torch.abs(cached_tensor), torch.abs(full_tensor))
        relative = torch.where(scale > 0, absolute / scale, torch.zeros_like(absolute))
        if absolute.numel():
            flat_index = int(torch.argmax(absolute).item())
            max_abs_diff = float(torch.max(absolute).item())
            max_rel_diff = float(torch.max(relative).item())
            nonzero_count = int(torch.count_nonzero(absolute).item())
            l2_diff = float(torch.linalg.vector_norm(absolute).item())
            max_diff_token = flat_index % int(absolute.shape[-1])
    return (
        "S5 generated-token parity failed: "
        f"call_id={call.call_id}, scene_id={call.scene_id}, "
        f"cue_condition={call.condition.value}, family={call.family}, "
        f"first_mismatch_step={step}, common_prefix_length={common_prefix_length}, "
        f"cached_token={cached_token}, full_token={full_token}, "
        f"cached_generated_token_ids={cached_tokens}, "
        f"full_generated_token_ids={full_tokens}, "
        f"cached_logit_shape={cached_shape}, full_logit_shape={full_shape}, "
        f"cached_logit_dtype={cached_dtype}, full_logit_dtype={full_dtype}, "
        f"cached_logit_finite={cached_finite}, full_logit_finite={full_finite}, "
        f"cached_argmax={cached_argmax}, full_argmax={full_argmax}, "
        f"cached_realized_token_logit={cached_realized}, "
        f"full_realized_token_logit={full_realized}, "
        f"max_abs_diff={max_abs_diff}, max_rel_diff={max_rel_diff}, "
        f"nonzero_count={nonzero_count}, l2_diff={l2_diff}, "
        f"argmax_abs_diff_token_id={max_diff_token}"
    )


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
        cached_logits = tuple(trace["cached_generated_logits"])
        full_logits = tuple(trace["full_generated_logits"])
        if cached_tokens != full_tokens:
            raise RuntimeError(
                _token_divergence_message(
                    call,
                    cached_tokens,
                    full_tokens,
                    cached_logits,
                    full_logits,
                )
            )
        logit_step_evidence = _explain_logit_parity(
            cached_logits,
            full_logits,
            cached_tokens,
            atol=logit_absolute_tolerance,
            rtol=logit_relative_tolerance,
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
                decision_parity_verified=True,
                logit_parity_verified=all(
                    evidence.exact_identity for evidence in logit_step_evidence
                ),
                logit_step_evidence=logit_step_evidence,
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
        "all_decision_parity_verified": all(record.decision_parity_verified for record in frozen),
        "all_logit_parity_verified": all(record.logit_parity_verified for record in frozen),
        "all_mrope_parity_verified": all(record.mrope_parity_verified for record in frozen),
        "all_cache_position_parity_verified": all(
            record.cache_position_parity_verified for record in frozen
        ),
    }
    required_checks = (
        checks["all_token_parity_verified"],
        checks["all_decision_parity_verified"],
        checks["all_mrope_parity_verified"],
        checks["all_cache_position_parity_verified"],
    )
    if not all(required_checks):
        raise RuntimeError("S5 objective parity proof is incomplete")
    return {
        "schema_version": 2,
        "status": "PHASE_3_EXACT_CACHE_DECISION_PARITY_VERIFIED",
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
    "LogitStepEvidence",
    "build_cache_parity_plan",
    "build_condition_turns",
    "execute_cache_parity_plan",
    "facts_for_condition",
    "summarize_cache_parity",
    "validate_cache_output_path",
    "write_cache_parity_outputs",
]
