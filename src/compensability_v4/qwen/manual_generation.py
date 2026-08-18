"""Auditable greedy generation for Qwen natural-state capture.

This intentionally does not call ``model.generate``.  Every selected token and
the cache that consumed it are visible to the continuation/parity machinery.
Heavy runtime dependencies are imported only inside execution functions.
"""

from __future__ import annotations

import inspect
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ManualGenerationResult:
    """Immutable token and cache evidence from one deterministic decode."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    all_token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[tuple[int, ...], ...]
    past_key_values: object
    generation_config: Mapping[str, Any]
    rng_seed: int
    generated_logits: tuple[object, ...] = ()
    forward_position_ids: tuple[tuple[tuple[int, ...], ...], ...] = ()
    forward_cache_positions: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.all_token_ids != self.prompt_token_ids + self.generated_token_ids:
            raise ValueError("manual-generation token provenance is inconsistent")
        if len(self.attention_mask) != len(self.all_token_ids):
            raise ValueError("manual-generation attention mask is misaligned")
        if any(len(axis) != len(self.all_token_ids) for axis in self.position_ids):
            raise ValueError("manual-generation position ids are misaligned")
        object.__setattr__(
            self, "generation_config", MappingProxyType(dict(self.generation_config))
        )


def _mapping(batch: object) -> dict[str, Any]:
    if isinstance(batch, Mapping):
        return dict(batch)
    keys = getattr(batch, "keys", None)
    if callable(keys):
        return {key: batch[key] for key in keys()}  # type: ignore[index]
    raise TypeError("prepared Qwen batch must be mapping-like")


def _one_dimensional_ids(tensor: object) -> tuple[int, ...]:
    to_list = getattr(tensor, "tolist", None)
    value = to_list() if callable(to_list) else tensor
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise RuntimeError("manual generation supports exactly one sample")
        value = value[0]
    if not isinstance(value, Sequence):
        raise TypeError("token ids must be sequence-like")
    return tuple(int(token_id) for token_id in value)


def _move_batch_to_model(batch: object, model: object) -> object:
    device = getattr(model, "device", None)
    move = getattr(batch, "to", None)
    if device is not None and callable(move):
        return move(device)
    if isinstance(batch, Mapping) and device is not None:
        return {
            key: value.to(device) if callable(getattr(value, "to", None)) else value
            for key, value in batch.items()
        }
    return batch


def _require_visual_batch_metadata(batch: object) -> object:
    prepared = _mapping(batch)
    missing = tuple(key for key in ("input_ids", "image_grid_thw") if key not in prepared)
    if missing:
        raise RuntimeError(
            "prepared Qwen visual batch is missing required runtime metadata: " + ", ".join(missing)
        )
    return batch


def _supports_structured_chat_output(processor: object) -> bool:
    apply_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_template):
        raise TypeError("Qwen processor exposes no callable chat template")
    try:
        parameters = inspect.signature(apply_template).parameters.values()
    except (TypeError, ValueError):
        return True
    names = {parameter.name for parameter in parameters}
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or {
        "return_dict",
        "return_tensors",
    }.issubset(names)


def _prepare_visual_chat_batch(
    processor: object,
    messages: Sequence[Mapping[str, Any]],
    model: object,
) -> object:
    """Prepare one multimodal chat without discarding processor grid metadata."""

    closed_messages = [dict(message) for message in messages]
    if _supports_structured_chat_output(processor):
        batch = processor.apply_chat_template(
            closed_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        _require_visual_batch_metadata(batch)
    else:
        # Compatibility path for older processors without structured chat output.
        template = processor.apply_chat_template(
            closed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(template, str) or not template:
            raise RuntimeError("Qwen visual chat template is invalid")
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:  # pragma: no cover - server-only dependency
            raise RuntimeError("qwen-vl-utils is required for visual state capture") from error
        image_inputs, video_inputs = process_vision_info(closed_messages)
        batch = processor(
            text=[template],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        _require_visual_batch_metadata(batch)
    return _move_batch_to_model(batch, model)


def _position_tuple(position_ids: object | None, length: int) -> tuple[tuple[int, ...], ...]:
    if position_ids is None:
        return (tuple(range(length)),)
    to_list = getattr(position_ids, "tolist", None)
    value = to_list() if callable(to_list) else position_ids
    if not isinstance(value, Sequence) or not value:
        raise RuntimeError("position_ids are empty")
    # Qwen MRoPE is commonly [3, batch, sequence]. Generation may prepend a
    # text-only packing axis, yielding [4, batch, sequence]; Qwen's model
    # forward strips that first axis before applying the rotary embeddings.
    packed_layout = len(value) in (3, 4) and all(
        isinstance(axis, Sequence) and len(axis) == 1 and isinstance(axis[0], Sequence)
        for axis in value
    )
    if packed_layout:
        axes = tuple(tuple(int(item) for item in axis[0]) for axis in value)
    elif len(value) == 1 and isinstance(value[0], Sequence):
        axes = (tuple(int(item) for item in value[0]),)
    elif len(value) == 4:
        raise RuntimeError("unsupported rank-two four-axis position_ids")
    else:
        axes = tuple(tuple(int(item) for item in axis) for axis in value)  # type: ignore[arg-type]
    if packed_layout and len(axes) == 4:
        axes = axes[1:]
    if any(len(axis) < length for axis in axes):
        raise RuntimeError("runtime position_ids do not align with the token sequence")
    return tuple(axis[-length:] for axis in axes)


def _eos_set(eos_token_ids: int | Sequence[int] | None) -> frozenset[int]:
    if eos_token_ids is None:
        return frozenset()
    if isinstance(eos_token_ids, int):
        return frozenset({eos_token_ids})
    return frozenset(int(token_id) for token_id in eos_token_ids)


def _extend_position_ids(position_ids: object, added: int) -> object:
    """Append monotonic text positions while preserving Qwen MRoPE axes."""

    import torch

    if added <= 0:
        return position_ids
    if position_ids.ndim == 3:
        last = position_ids[:, :, -1:]
        offsets = torch.arange(1, added + 1, device=position_ids.device).view(1, 1, -1)
    elif position_ids.ndim == 2:
        last = position_ids[:, -1:]
        offsets = torch.arange(1, added + 1, device=position_ids.device).view(1, -1)
    else:
        raise RuntimeError("unsupported runtime position-id rank")
    return torch.cat((position_ids, last + offsets), dim=-1)


def _prepared_forward(
    model: object,
    full_input_ids: object,
    model_kwargs: dict[str, Any],
    *,
    next_sequence_length: int,
    is_first_iteration: bool,
) -> tuple[object, dict[str, Any], tuple[int, ...] | None]:
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if callable(prepare):
        prepared = prepare(
            full_input_ids,
            **{
                **model_kwargs,
                "next_sequence_length": next_sequence_length,
                "is_first_iteration": is_first_iteration,
            },
        )
        arguments = _mapping(prepared)
    else:
        arguments = dict(model_kwargs)
        arguments["input_ids"] = full_input_ids
    arguments["use_cache"] = True
    arguments["return_dict"] = True
    cache_positions = _cache_position_trace(arguments)
    output = model(**arguments)
    return output, arguments, cache_positions


def _cache_position_trace(arguments: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Capture explicit positions or derive them from the pre-forward KV length."""

    explicit = arguments.get("cache_position")
    if explicit is not None:
        return _one_dimensional_ids(explicit)

    input_ids = arguments.get("input_ids")
    inputs_embeds = arguments.get("inputs_embeds")
    active_input = input_ids if input_ids is not None else inputs_embeds
    if active_input is None:
        return None
    shape = getattr(active_input, "shape", None)
    if shape is None:
        return None
    sequence_axis = -1 if input_ids is not None else -2
    try:
        sequence_length = operator.index(shape[sequence_axis])
    except (IndexError, TypeError) as error:
        raise RuntimeError("S5 runtime input exposes no valid sequence length") from error
    if sequence_length <= 0:
        raise RuntimeError("S5 runtime input sequence length must be positive")

    cache = arguments.get("past_key_values")
    if cache is None:
        start = 0
    else:
        get_seq_length = getattr(cache, "get_seq_length", None)
        if not callable(get_seq_length):
            return None
        try:
            start = operator.index(get_seq_length())
        except TypeError as error:
            raise RuntimeError("S5 runtime cache exposes no valid sequence length") from error
        if start < 0:
            raise RuntimeError("S5 runtime cache sequence length must be non-negative")
    return tuple(range(start, start + sequence_length))


def manual_greedy_generate(
    model: object,
    batch: object,
    *,
    max_new_tokens: int,
    eos_token_ids: int | Sequence[int] | None = None,
    rng_seed: int = 0,
    generation_config: Mapping[str, Any] | None = None,
    past_key_values: object | None = None,
    prior_token_ids: Sequence[int] = (),
    prior_position_ids: Sequence[Sequence[int]] = (),
) -> ManualGenerationResult:
    """Greedily decode one sample and return a cache containing every token."""

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    config = dict(generation_config or {})
    if config.get("do_sample", False) is not False:
        raise RuntimeError("natural-state capture requires deterministic greedy decoding")
    temperature = float(config.get("temperature", 0.0))
    if not math.isfinite(temperature) or temperature < 0.0:
        raise RuntimeError("temperature must be finite and non-negative")
    config.update({"do_sample": False, "max_new_tokens": max_new_tokens})

    import torch

    torch.manual_seed(rng_seed)
    initial = _mapping(batch)
    if "input_ids" not in initial:
        raise RuntimeError("prepared Qwen batch has no input_ids")
    full_input_ids = initial.pop("input_ids")
    if int(full_input_ids.shape[0]) != 1:
        raise RuntimeError("manual generation supports exactly one sample")
    suffix_prompt_ids = _one_dimensional_ids(full_input_ids)
    prior_ids = tuple(int(token_id) for token_id in prior_token_ids)
    if (past_key_values is None) != (not prior_ids):
        raise ValueError("prior token ids and past_key_values must be provided together")
    if past_key_values is not None:
        prior_positions = tuple(tuple(int(item) for item in axis) for axis in prior_position_ids)
        if not prior_positions or any(len(axis) != len(prior_ids) for axis in prior_positions):
            raise ValueError("prior position ids must align with cached token ids")
        prior_tensor = torch.tensor(
            [prior_ids], dtype=full_input_ids.dtype, device=full_input_ids.device
        )
        full_input_ids = torch.cat((prior_tensor, full_input_ids), dim=-1)
        initial["past_key_values"] = past_key_values
        suffix_length = len(suffix_prompt_ids)
        suffix_positions = tuple(
            tuple(axis[-1] + offset for offset in range(1, suffix_length + 1))
            for axis in prior_positions
        )
        full_positions = tuple(
            prior_positions[index] + suffix_positions[index]
            for index in range(len(prior_positions))
        )
        position_tensor = torch.tensor(
            full_positions,
            dtype=torch.long,
            device=full_input_ids.device,
        )
        initial["position_ids"] = (
            position_tensor.unsqueeze(1) if len(full_positions) == 3 else position_tensor
        )
    prompt_ids = prior_ids + suffix_prompt_ids
    attention_mask = initial.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(full_input_ids)
        initial["attention_mask"] = attention_mask
    prepare_position_ids = getattr(model, "_prepare_position_ids_for_generation", None)
    if initial.get("position_ids") is None and callable(prepare_position_ids):
        position_ids = prepare_position_ids(full_input_ids, initial)
        if position_ids is not None:
            initial["position_ids"] = position_ids
    initialize_cache_position = getattr(model, "_get_initial_cache_position", None)
    if callable(initialize_cache_position):
        initial = initialize_cache_position(full_input_ids, initial)
    eos = _eos_set(eos_token_ids)
    generated: list[int] = []
    active_cache: object | None = past_key_values
    last_prepared: dict[str, Any] = {}
    position_history: list[tuple[tuple[int, ...], ...]] = []
    cache_position_history: list[tuple[int, ...]] = []
    generated_logits: list[object] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            processed_length = len(suffix_prompt_ids) if not generated else 1
            output, last_prepared, prepared_cache_positions = _prepared_forward(
                model,
                full_input_ids,
                initial,
                next_sequence_length=processed_length,
                is_first_iteration=not generated,
            )
            prepared_positions = last_prepared.get("position_ids")
            if prepared_positions is not None:
                position_history.append(_position_tuple(prepared_positions, processed_length))
            if prepared_cache_positions is not None:
                cache_position_history.append(prepared_cache_positions)
            generated_logits.append(output.logits[:, -1, :].detach().to("cpu"))
            next_token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            next_tensor = torch.tensor(
                [[next_token]], dtype=full_input_ids.dtype, device=full_input_ids.device
            )
            full_input_ids = torch.cat((full_input_ids, next_tensor), dim=-1)
            generated.append(next_token)
            active_cache = output.past_key_values
            update_kwargs = getattr(model, "_update_model_kwargs_for_generation", None)
            if callable(update_kwargs):
                initial = update_kwargs(output, initial, is_encoder_decoder=False)
            else:
                initial["past_key_values"] = active_cache
                initial["attention_mask"] = torch.cat(
                    (
                        initial["attention_mask"],
                        torch.ones(
                            (1, 1),
                            dtype=initial["attention_mask"].dtype,
                            device=initial["attention_mask"].device,
                        ),
                    ),
                    dim=-1,
                )
                if initial.get("position_ids") is not None:
                    initial["position_ids"] = _extend_position_ids(initial["position_ids"], 1)
            current_positions = initial.get("position_ids")
            if current_positions is not None:
                current_length = int(current_positions.shape[-1])
                missing = int(full_input_ids.shape[-1]) - current_length
                if missing > 0:
                    initial["position_ids"] = _extend_position_ids(current_positions, missing)
            if next_token in eos:
                break

        if not generated or active_cache is None:
            raise RuntimeError("manual generation produced no token/cache state")
        # Each forward cache excludes the token just selected from its logits.
        # Consume that last token once, so suffix continuation starts exactly
        # after all token_ids recorded below.
        final_output, last_prepared, prepared_cache_positions = _prepared_forward(
            model,
            full_input_ids,
            initial,
            next_sequence_length=1,
            is_first_iteration=False,
        )
        active_cache = final_output.past_key_values
        prepared_positions = last_prepared.get("position_ids")
        if prepared_positions is not None:
            position_history.append(_position_tuple(prepared_positions, 1))
        if prepared_cache_positions is not None:
            cache_position_history.append(prepared_cache_positions)

    all_ids = _one_dimensional_ids(full_input_ids)
    final_attention = _one_dimensional_ids(initial["attention_mask"])
    if position_history:
        axis_count = len(position_history[0])
        if any(len(chunk) != axis_count for chunk in position_history):
            raise RuntimeError("runtime position-id rank changed during generation")
        generated_positions = tuple(
            tuple(item for chunk in position_history for item in chunk[axis])
            for axis in range(axis_count)
        )
        prior_positions = tuple(tuple(int(item) for item in axis) for axis in prior_position_ids)
        if prior_ids:
            if len(prior_positions) != axis_count or any(
                len(axis) != len(prior_ids) for axis in prior_positions
            ):
                raise ValueError("prior position ids must align with cached token ids")
            normalized_positions = tuple(
                prior_positions[axis] + generated_positions[axis] for axis in range(axis_count)
            )
        else:
            normalized_positions = generated_positions
        if any(len(axis) != len(all_ids) for axis in normalized_positions):
            raise RuntimeError("captured runtime positions do not cover every cached token")
    else:
        normalized_positions = (tuple(range(len(all_ids))),)
    return ManualGenerationResult(
        prompt_token_ids=prompt_ids,
        generated_token_ids=tuple(generated),
        all_token_ids=all_ids,
        attention_mask=final_attention,
        position_ids=normalized_positions,
        past_key_values=active_cache,
        generation_config=config,
        rng_seed=rng_seed,
        generated_logits=tuple(generated_logits),
        forward_position_ids=tuple(position_history),
        forward_cache_positions=tuple(cache_position_history),
    )


def generate_observation_with_cache(
    model: object,
    processor: object,
    image: object,
    prompt: str,
    *,
    sample_id: str,
    resized_height: int,
    resized_width: int,
    max_new_tokens: int = 64,
    rng_seed: int = 0,
) -> dict[str, object]:
    """Run Qwen's normal visual chat path and capture exact natural state."""

    if resized_height <= 0 or resized_width <= 0:
        raise ValueError("fixed visual dimensions must be positive")
    if resized_height % 28 or resized_width % 28:
        raise ValueError("fixed visual dimensions must be integer multiples of 28")
    messages = (
        {
            "role": "user",
            "content": (
                {
                    "type": "image",
                    "image": image,
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                },
                {"type": "text", "text": prompt},
            ),
        },
    )
    batch = _prepare_visual_chat_batch(processor, messages, model)
    eos_ids = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    result = manual_greedy_generate(
        model,
        batch,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
        rng_seed=rng_seed,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded = tokenizer.decode(result.generated_token_ids, skip_special_tokens=True)
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    if not isinstance(image_token_id, int):
        raise RuntimeError("runtime config exposes no image_token_id")
    image_positions = tuple(
        index for index, token_id in enumerate(result.all_token_ids) if token_id == image_token_id
    )
    if not image_positions:
        raise RuntimeError("prepared visual prompt contains no runtime image tokens")
    grid = _one_dimensional_ids(_mapping(batch)["image_grid_thw"])
    if len(grid) != 3:
        raise RuntimeError("runtime image_grid_thw must contain exactly three values")
    from .cache_continuation import CachedGenerationState

    assistant_messages = (*messages, {"role": "assistant", "content": decoded})
    state = CachedGenerationState(
        sample_id=sample_id,
        token_ids=result.all_token_ids,
        attention_mask=result.attention_mask,
        position_ids=result.position_ids,
        image_token_positions=image_positions,
        image_grid_thw=(grid[0], grid[1], grid[2]),
        visual_token_count=len(image_positions),
        generation_config=result.generation_config,
        rng_seed=result.rng_seed,
        past_key_values=result.past_key_values,
        chat_messages=assistant_messages,
        generated_token_ids=result.generated_token_ids,
        processor=processor,
    )
    return {"text": decoded, "generated_token_ids": result.generated_token_ids, "state": state}


__all__ = [
    "ManualGenerationResult",
    "generate_observation_with_cache",
    "manual_greedy_generate",
]
