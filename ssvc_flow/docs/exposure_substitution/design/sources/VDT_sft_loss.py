"""Explicit causal, response-only, sequence-normalized SFT objective."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodedExample:
    task_id: str
    prompt_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None = None

    @property
    def input_ids(self):
        return self.prompt_ids + self.target_ids


def encode_example(adapter, prompt, target, *, task_id="", data_root=None, max_total_tokens=1024):
    """Tokenize once; keep actual chat prefix and append exactly one primary EOS."""
    if set(prompt) != {"system", "user"}:
        raise ValueError("SFT requires the registered symbolic public O0 prompt")
    world = json.loads(target) if isinstance(target, str) else target
    if (
        not isinstance(world, list)
        or len(world) != 4
        or any(type(x) is not int or not 0 <= x <= 99 for x in world)
    ):
        raise ValueError("Canonical target must be four legal integers")
    target = json.dumps(world, separators=(",", ":"))
    prepared = adapter.prepare(prompt, data_root)
    inputs = prepared["inputs"]
    unexpected = set(inputs) - {"input_ids", "attention_mask", "mm_token_type_ids"}
    if unexpected:
        raise ValueError(f"Unregistered symbolic processor inputs: {sorted(unexpected)}")
    prompt_ids = tuple(inputs["input_ids"][0].tolist())
    if not prompt_ids or not inputs["attention_mask"].all():
        raise ValueError("One unpadded nonempty prompt required")
    tokenizer = adapter.processor.tokenizer
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos not in adapter.eos_ids:
        raise ValueError("Audited tokenizer primary EOS required")
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    if not target_ids or any(t in adapter.eos_ids for t in target_ids):
        raise ValueError("Canonical JSON contains unexpected EOS")
    target_ids.append(eos)
    if len(prompt_ids) + len(target_ids) > max_total_tokens:
        raise ValueError("SFT input exceeds max_total_tokens; truncation is forbidden")
    types = inputs.get("mm_token_type_ids")
    return EncodedExample(
        str(task_id),
        prompt_ids,
        tuple(target_ids),
        tuple(types[0].tolist()) if types is not None else None,
    )


def collate_examples(examples, *, pad_id, device="cpu"):
    import torch

    if not examples:
        raise ValueError("Empty microbatch")
    length = max(len(x.input_ids) for x in examples)
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    use_types = any(x.token_type_ids is not None for x in examples)
    if use_types and any(x.token_type_ids is None for x in examples):
        raise ValueError("Mixed processor token-type conventions")
    types = torch.zeros_like(ids) if use_types else None
    for i, example in enumerate(examples):
        p, n = len(example.prompt_ids), len(example.input_ids)
        if not p or not example.target_ids:
            raise ValueError("Nonempty prompt and target required")
        ids[i, :n] = torch.tensor(example.input_ids, device=device)
        attention[i, :n] = 1
        labels[i, p:n] = ids[i, p:n]
        if use_types:
            types[i, :p] = torch.tensor(example.token_type_ids, device=device)
    result = {"input_ids": ids, "attention_mask": attention}
    if use_types:
        result["mm_token_type_ids"] = types
    return result, labels


def per_sequence_nll(logits, labels):
    """Logit p-1 predicts target p; PAD=EOS remains loss-bearing only inside length."""
    import torch
    import torch.nn.functional as F

    if logits.shape[:2] != labels.shape:
        raise ValueError("Full-sequence logits and labels must have equal B,T axes")
    shifted = labels[:, 1:]
    mask = shifted.ne(-100)
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("Every sequence needs at least one supervised target")
    token_nll = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted)
    means = token_nll.sum(-1) / counts
    if not torch.isfinite(means).all():
        raise FloatingPointError("Nonfinite completion loss")
    return means, token_nll, mask


def forward_batch(adapter, examples):
    inputs, labels = collate_examples(examples, pad_id=adapter.pad_id, device=adapter.device)
    adapter._reset_positions()
    output = adapter.model(**inputs, use_cache=False)
    return output.logits, labels


def accumulated_backward(adapter, examples, *, microbatch_size=4, denominator=16):
    """Every actual sequence contributes 1/16, including replay-only's four."""
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= 4:
        raise ValueError("Registered microbatch range is 1..4")
    if denominator != 16 or len(examples) not in (4, 16):
        raise ValueError("Registered update contains 16 slots or four replay-only slots")
    total, tokens, losses = 0.0, 0, []
    for start in range(0, len(examples), microbatch_size):
        logits, labels = forward_batch(adapter, examples[start : start + microbatch_size])
        per_sequence, _, mask = per_sequence_nll(logits, labels)
        loss = per_sequence.sum() / denominator
        loss.backward()
        total += float(loss.detach())
        tokens += int(mask.sum())
        losses.extend(per_sequence.detach().cpu().tolist())
    return {
        "loss": total,
        "sequence_nll": losses,
        "target_tokens": tokens,
        "sequence_count": len(examples),
        "sequence_coefficient": 1 / denominator,
    }
