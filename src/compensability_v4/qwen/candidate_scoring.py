"""Teacher-forced candidate-label scoring for Phase 1 and Phase 2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _encoded_label(tokenizer: object, label: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose encode()")
    encoded = encode(label, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise RuntimeError(f"tokenizer returned invalid ids for label {label!r}")
    token_ids = tuple(int(item) for item in encoded)
    if any(item < 0 for item in token_ids):
        raise RuntimeError(f"tokenizer returned a negative id for label {label!r}")
    return token_ids


def find_single_token_labels(
    tokenizer: object, candidates: Sequence[str], minimum: int
) -> tuple[str, ...]:
    """Select distinct, stable one-token labels while preserving search order."""

    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise ValueError("minimum must be a positive integer")
    labels: list[str] = []
    used_labels: set[str] = set()
    used_token_ids: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate or candidate in used_labels:
            continue
        token_ids = _encoded_label(tokenizer, candidate)
        if len(token_ids) == 1 and token_ids[0] not in used_token_ids:
            labels.append(candidate)
            used_labels.add(candidate)
            used_token_ids.add(token_ids[0])
        if len(labels) == minimum:
            break
    if len(labels) < minimum:
        raise RuntimeError("insufficient single-token labels")
    return tuple(labels)


def _tokenizer(processor: object) -> object:
    tokenizer = getattr(processor, "tokenizer", None)
    return processor if tokenizer is None else tokenizer


def _prepare_prompt(processor: object, prompt: str) -> object:
    if not callable(processor):
        raise TypeError("real-runtime scoring requires a callable processor")
    try:
        return processor(text=[prompt], padding=True, return_tensors="pt")
    except TypeError:
        return processor(prompt, return_tensors="pt")


def _move_to_model_device(batch: object, model: object) -> object:
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


def next_token_logits(model: object, processor: object, prompt: str) -> object:
    """Return the standard-forward logits at the final non-padding token."""

    fake_runtime = getattr(model, "next_token_logits", None)
    if callable(fake_runtime):
        return fake_runtime(prompt)

    import torch

    batch = _move_to_model_device(_prepare_prompt(processor, prompt), model)
    arguments = dict(batch) if isinstance(batch, Mapping) else vars(batch)
    with torch.inference_mode():
        output = model(**arguments, use_cache=False, return_dict=True)
    logits = output.logits
    attention_mask = arguments.get("attention_mask")
    if attention_mask is None:
        return logits[0, -1]
    final_index = int(attention_mask[0].sum().item()) - 1
    if final_index < 0:
        raise RuntimeError("candidate prompt contains no attended token")
    return logits[0, final_index]


def _logit_at(logits: object, token_id: int) -> float:
    value = logits[token_id]  # type: ignore[index]
    item = getattr(value, "item", None)
    result = float(item() if callable(item) else value)
    if not math.isfinite(result):
        raise RuntimeError("candidate logit is not finite")
    return result


def score_candidate_labels(
    model: object,
    processor: object,
    prompt: str,
    candidate_labels: Sequence[str],
    worlds: Sequence[Sequence[int]],
) -> dict[tuple[int, int, int, int], float]:
    """Map candidate worlds to raw next-token logits, independent of order."""

    if len(candidate_labels) != len(worlds) or not candidate_labels:
        raise ValueError("candidate labels and worlds must be non-empty and aligned")
    tokenizer = _tokenizer(processor)
    label_token_ids = tuple(_encoded_label(tokenizer, label) for label in candidate_labels)
    if any(len(ids) != 1 for ids in label_token_ids):
        raise RuntimeError("all candidate labels must be single token")
    if len({ids[0] for ids in label_token_ids}) != len(label_token_ids):
        raise RuntimeError("candidate label token ids must be unique")
    normalized_worlds = tuple(tuple(int(value) for value in world) for world in worlds)
    if any(len(world) != 4 for world in normalized_worlds):
        raise ValueError("each candidate world must contain exactly four values")
    if len(set(normalized_worlds)) != len(normalized_worlds):
        raise RuntimeError("candidate worlds must be unique")
    logits = next_token_logits(model, processor, prompt)
    scored: dict[tuple[int, int, int, int], float] = {}
    for token_ids, world in zip(label_token_ids, normalized_worlds, strict=True):
        scored[world] = _logit_at(logits, token_ids[0])
    return scored


def candidate_log_probabilities(scores: Mapping[object, float]) -> dict[object, float]:
    """Normalize a candidate logit set with a stable log-sum-exp."""

    if not scores:
        raise ValueError("candidate score mapping must be non-empty")
    maximum = max(float(value) for value in scores.values())
    denominator = maximum + math.log(
        sum(math.exp(float(value) - maximum) for value in scores.values())
    )
    return {key: float(value) - denominator for key, value in scores.items()}


def candidate_margin(
    scores: Mapping[tuple[int, int, int, int], float],
    true_world: tuple[int, int, int, int],
    observed_world: tuple[int, int, int, int],
) -> float:
    """Return the task-defined true-versus-observed logit margin ``M_F``."""

    try:
        return float(scores[true_world]) - float(scores[observed_world])
    except KeyError as error:
        raise ValueError("true and observed worlds must both be scored candidates") from error


__all__ = [
    "candidate_log_probabilities",
    "candidate_margin",
    "find_single_token_labels",
    "next_token_logits",
    "score_candidate_labels",
]
