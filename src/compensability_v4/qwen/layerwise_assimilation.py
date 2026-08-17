"""Layerwise candidate-logit projection and constraint-assimilation gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .introspect_model import language_layer_count


def _as_float(value: object) -> float:
    item = getattr(value, "item", None)
    result = float(item() if callable(item) else value)  # type: ignore[arg-type]
    if not math.isfinite(result):
        raise RuntimeError("layerwise candidate logit is not finite")
    return result


def _candidate_ids(
    label_token_ids: Mapping[str, int] | Sequence[int],
) -> tuple[tuple[str, int], ...]:
    if isinstance(label_token_ids, Mapping):
        pairs = tuple((str(label), int(token_id)) for label, token_id in label_token_ids.items())
    else:
        pairs = tuple((str(token_id), int(token_id)) for token_id in label_token_ids)
    if not pairs or len({token_id for _, token_id in pairs}) != len(pairs):
        raise ValueError("candidate token ids must be non-empty and unique")
    if any(token_id < 0 for _, token_id in pairs):
        raise ValueError("candidate token ids must be non-negative")
    return pairs


def _runtime_projection_modules(model: object) -> tuple[object, object]:
    get_head = getattr(model, "get_output_embeddings", None)
    head = get_head() if callable(get_head) else getattr(model, "lm_head", None)
    backbone = getattr(model, "model", None)
    norm = getattr(backbone, "norm", None)
    if head is None or not callable(head):
        raise RuntimeError("runtime model exposes no output embedding/lm head")
    if norm is None or not callable(norm):
        raise RuntimeError("runtime model exposes no final language norm")
    return norm, head


def _forward(model: object, batch: object) -> tuple[object, Mapping[str, object]]:
    arguments = dict(batch) if isinstance(batch, Mapping) else vars(batch)
    try:
        import torch
    except ImportError as error:  # pragma: no cover - only real runtime needs torch
        raise RuntimeError("layerwise Qwen projection requires torch") from error
    with torch.inference_mode():
        output = model(
            **arguments,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    return output, arguments


def layerwise_candidate_logits(
    model: object,
    batch: object,
    label_token_ids: Mapping[str, int] | Sequence[int],
) -> list[dict[str, float]]:
    """Project every language-layer state through the runtime norm and head.

    Qwen/Hugging Face hidden-state output contains the embedding state followed
    by one state per decoder layer.  The final state is already normalized;
    earlier decoder states receive the same runtime final norm before the head.
    A mandatory final-layer parity comparison guards this interpretation.
    """

    pairs = _candidate_ids(label_token_ids)
    output, arguments = _forward(model, batch)
    hidden_states = tuple(getattr(output, "hidden_states", ()) or ())
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("runtime model has no config")
    try:
        layer_count = language_layer_count(config)
    except RuntimeError as error:
        raise RuntimeError(f"runtime config has no valid language-layer count: {error}") from error
    if len(hidden_states) != layer_count + 1:
        raise RuntimeError(
            "runtime hidden-state count does not match num_hidden_layers: "
            f"{len(hidden_states)} versus {layer_count + 1}"
        )
    attention_mask = arguments.get("attention_mask")
    final_index = -1
    if attention_mask is not None:
        final_index = int(attention_mask[0].sum().item()) - 1  # type: ignore[index,union-attr]
        if final_index < 0:
            raise RuntimeError("layerwise prompt contains no attended token")
    norm, head = _runtime_projection_modules(model)
    result: list[dict[str, float]] = []
    for index, hidden in enumerate(hidden_states[1:]):
        token_hidden = hidden[0, final_index]
        # The last returned hidden state is post-norm in Qwen2.5-VL.
        projected_hidden = token_hidden if index == layer_count - 1 else norm(token_hidden)
        vocabulary_logits = head(projected_hidden)
        result.append({label: _as_float(vocabulary_logits[token_id]) for label, token_id in pairs})
    standard_logits = output.logits[0, final_index]
    standard = {label: _as_float(standard_logits[token_id]) for label, token_id in pairs}
    validate_final_layer_logits(result, standard)
    return result


def validate_final_layer_logits(
    layerwise_logits: Sequence[Mapping[str, float]],
    forward_logits: Mapping[str, float],
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> None:
    """Fail closed unless the final projected candidates match normal forward."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("logit tolerances must be non-negative")
    if not layerwise_logits or set(layerwise_logits[-1]) != set(forward_logits):
        raise RuntimeError("final-layer candidate logits do not match the standard forward pass")
    for label, expected in forward_logits.items():
        observed = layerwise_logits[-1][label]
        if not math.isclose(
            float(observed),
            float(expected),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise RuntimeError(
                "final-layer candidate logits do not match the standard forward pass "
                f"for {label!r}: {observed} versus {expected}"
            )


def layerwise_margins(
    layerwise_logits: Sequence[Mapping[str, float]],
    *,
    true_label: str,
    observed_label: str,
) -> tuple[float, ...]:
    """Compute ``M_F^(l)`` from task logits, never latent distances."""

    margins: list[float] = []
    for logits in layerwise_logits:
        if true_label not in logits or observed_label not in logits:
            raise ValueError("true and observed labels must exist at every layer")
        margins.append(float(logits[true_label]) - float(logits[observed_label]))
    return tuple(margins)


def classify_assimilation_profile(
    no_cue_margins: Sequence[float], valid_cue_margins: Sequence[float]
) -> str:
    """Classify the four preregistered qualitative layerwise profiles."""

    if len(no_cue_margins) != len(valid_cue_margins) or not valid_cue_margins:
        raise ValueError("no-cue and valid-cue margins must be non-empty and aligned")
    deltas = tuple(
        valid - base for base, valid in zip(no_cue_margins, valid_cue_margins, strict=True)
    )
    positive = tuple(delta > 0.0 for delta in deltas)
    if not any(positive):
        return "no_assimilation"
    if valid_cue_margins[-1] > 0.0:
        return "successful_revision"
    if positive[-1]:
        return "persistent_but_insufficient_assimilation"
    return "transient_assimilation"


__all__ = [
    "classify_assimilation_profile",
    "layerwise_candidate_logits",
    "layerwise_margins",
    "validate_final_layer_logits",
]
