"""Autoregressive likelihood, generated-token mask, and measured cache audits."""

from __future__ import annotations

import math


def completion_mask(token_ids, prompt_length, *, eos_ids, pad_id, attention_mask=None):
    """Include all actually sampled tokens through EOS; padding needs position data.

    A sampled PAD vocabulary ID is an action token under pure softmax. Its ID alone
    never proves that a position was added by batch padding.
    """
    if not 0 <= prompt_length <= len(token_ids):
        raise ValueError("prompt_length is outside the sequence")
    ended = False
    result = []
    for index, token in enumerate(token_ids):
        active = (
            index >= prompt_length
            and not ended
            and (attention_mask is None or bool(attention_mask[index]))
        )
        is_eos = active and token in eos_ids
        result.append(active)
        ended = ended or is_eos
    return result


def selected_token_logprobs(logits, token_ids, *, prompt_length, eos_ids, pad_id):
    """Single-sequence shifted softmax; FP32 logits, gradients preserved."""
    import torch

    if logits.ndim != 3 or logits.shape[0] != 1 or token_ids.shape[0] != 1:
        raise ValueError("Likelihood uses unpadded microbatch=1 inputs")
    if logits.shape[1] != token_ids.shape[1]:
        raise ValueError("Logits must cover every input position")
    mask = completion_mask(token_ids[0].tolist(), prompt_length, eos_ids=eos_ids, pad_id=pad_id)
    positions = torch.tensor(
        [i - 1 for i, active in enumerate(mask) if active], device=logits.device
    )
    if not len(positions) or prompt_length < 1:
        raise ValueError("At least one prompt and one generated token required")
    selected = logits[0, positions].float().log_softmax(-1)
    return selected.gather(-1, token_ids[0, positions + 1].unsqueeze(-1)).squeeze(-1)


def cache_comparison(
    full_greedy,
    cached_greedy,
    full_logprobs,
    cached_logprobs,
    *,
    top1_threshold=0.999,
    logprob_tolerance=0.02,
):
    """Return measured alarms; this does not certify exact hidden-state equivalence."""
    count = len(full_greedy)
    if not count or any(len(v) != count for v in (cached_greedy, full_logprobs, cached_logprobs)):
        raise ValueError("Cache audit needs equal, nonempty observations")
    differences = [abs(a - b) for a, b in zip(full_logprobs, cached_logprobs, strict=False)]
    if not all(math.isfinite(v) for v in differences):
        raise ValueError("Cache audit contains nonfinite values")
    top1 = sum(a == b for a, b in zip(full_greedy, cached_greedy, strict=False)) / count
    mean_error = sum(differences) / count
    return {
        "tokens": count,
        "top1_agreement": top1,
        "mean_selected_logprob_error": mean_error,
        "max_selected_logprob_error": max(differences),
        "top1_threshold": top1_threshold,
        "logprob_tolerance": logprob_tolerance,
        "passed": top1 >= top1_threshold and mean_error <= logprob_tolerance,
    }


def audit_model_cache(adapter, prepared, completion):
    """Compare full recompute, chunk and token continuation; copy the complete cache.

    Adapter owns model-specific multimodal metadata. Unsupported copy is an explicit
    I4 outcome; no ordinary KV-only approximation is substituted.
    """
    import copy

    import torch

    with torch.no_grad():
        reference = adapter.continuation_scores(prepared, completion, mode="full")
        recompute = adapter.continuation_scores(prepared, completion, mode="full")
        result = {
            "recompute_baseline": cache_comparison(
                reference[0], recompute[0], reference[1], recompute[1]
            )
        }
        for mode in ("chunk", "token"):
            observed = adapter.continuation_scores(prepared, completion, mode=mode)
            result[mode] = cache_comparison(reference[0], observed[0], reference[1], observed[1])
        try:
            prefix = adapter.make_prefix_state(prepared)
            a = adapter.continue_from_state(copy.deepcopy(prefix), completion)
            b = adapter.continue_from_state(copy.deepcopy(prefix), completion)
            same = cache_comparison(a[0], b[0], a[1], b[1])
            reference_checks = [
                cache_comparison(reference[0], candidate[0], reference[1], candidate[1])
                for candidate in (a, b)
            ]
            supported = same["passed"] and all(check["passed"] for check in reference_checks)
            result["I4"] = {
                "status": "supported" if supported else "unsupported",
                "complete_framework_cache_copied": True,
                **same,
                "branches_vs_full_reference": reference_checks,
                "passed": supported,
            }
        except (TypeError, RuntimeError, AttributeError, NotImplementedError) as exc:
            result["I4"] = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
    result["passed"] = all(
        result[key]["passed"] for key in ("recompute_baseline", "chunk", "token")
    )
    return result
