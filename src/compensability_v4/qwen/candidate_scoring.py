"""Candidate-label utilities for fake-runtime and real-runtime scoring."""

from __future__ import annotations


def find_single_token_labels(tokenizer, candidates: list[str], minimum: int) -> tuple[str, ...]:
    labels = []
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1:
            labels.append(candidate)
        if len(labels) == minimum:
            break
    if len(labels) < minimum:
        raise RuntimeError("insufficient single-token labels")
    return tuple(labels)


def score_candidate_labels(model, processor, prompt: str, candidate_labels, worlds) -> dict[tuple[int, int, int, int], float]:
    logits = model.next_token_logits(prompt)
    scored = {}
    for label, world in zip(candidate_labels, worlds, strict=True):
        token_id = processor.tokenizer.encode(label, add_special_tokens=False)[0]
        scored[tuple(world)] = logits[token_id]
    return scored
