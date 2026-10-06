"""Train-only real-model bridge. Diagnostics are evidence, not scientific success."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .sft_loss import (
    accumulated_backward,
    collate_examples,
    encode_example,
    forward_batch,
    per_sequence_nll,
)
from .sft_runtime import SFTRuntime


def _gradient_snapshot(runtime):
    return {
        n: p.grad.detach().cpu().clone()
        for n, p in runtime.adapter.model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def _gradient_comparison(left, right):
    import torch

    if set(left) != set(right) or not left:
        raise ValueError("Gradient names differ or are empty")
    dot = norm_left = norm_right = delta = 0.0
    maximum = 0.0
    for name in left:
        a, b = left[name].double(), right[name].double()
        dot += float((a * b).sum())
        norm_left += float(a.square().sum())
        norm_right += float(b.square().sum())
        delta += float((a - b).square().sum())
        maximum = max(maximum, float((a - b).abs().max()))
    return {
        "cosine": dot / max((norm_left * norm_right) ** 0.5, 1e-300),
        "relative_l2": (delta / max(norm_left, 1e-300)) ** 0.5,
        "left_norm": norm_left**0.5,
        "right_norm": norm_right**0.5,
        "max_abs": maximum,
        "finite": all(torch.isfinite(v).all().item() for v in [*left.values(), *right.values()]),
    }


def run_bridge_from_backend(backend, *, rows, output_dir):
    if len(rows) != 8 or len({r["task_id"] for r in rows}) != 8:
        raise ValueError("Bridge requires exactly eight fixed distinct train-only cases")
    if any(r.get("split") not in ("T_train", "R_replay") for r in rows):
        raise ValueError("Explicit train-only bridge role is required")
    runtime = SFTRuntime.from_frozen_backend(
        backend,
        seed=73101,
        view_identity={
            "purpose": "DISPOSABLE_TECHNICAL_BRIDGE",
            "tasks": [r["task_id"] for r in rows],
        },
    )
    examples = [
        encode_example(
            backend.adapter,
            r["prompt"],
            r["target"],
            task_id=r["task_id"],
            data_root=backend.data_root,
        )
        for r in rows
    ]
    return run_bridge(
        runtime, examples, output_dir=output_dir, rows=rows, data_root=backend.data_root
    )


def run_bridge(runtime, examples, *, output_dir, rows=None, data_root=None):
    """Measure full/prefix, padding, causal logits, gradients and two-update resume.

    Prefix/full or batch/micro numerical differences are reported for review.
    No arbitrary global BF16 tolerance converts differences into a certificate.
    This modifies a disposable student only; reload the parent for formal arms.
    """
    import torch

    from ..optimizer_fork import state_hash

    if len(examples) != 8 or len({e.task_id for e in examples}) != 8:
        raise ValueError("Exactly eight fixed bridge examples required")
    device = torch.device(runtime.adapter.device)
    if device.type != "cuda":
        raise ValueError("Real bridge requires CUDA; use CPU unit tests for fixtures")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = runtime.adapter
    runtime.bind_data(examples, examples)
    initial = runtime.capture()
    diagnostics, padding_diagnostics, exact_causal = [], [], True
    adapter.model.train()
    with torch.no_grad():
        for offset in (0, 4):
            group = examples[offset : offset + 4]
            batch_logits, batch_labels = forward_batch(adapter, group)
            batch_mean, batch_nll, batch_mask = per_sequence_nll(batch_logits, batch_labels)
            # Alter PAD token values only, preserving attention and tensor shape.
            padded_inputs, _ = collate_examples(group, pad_id=adapter.pad_id, device=adapter.device)
            repeated_batch, _ = forward_batch(adapter, group)
            padded_inputs["input_ids"][padded_inputs["attention_mask"] == 0] = group[0].prompt_ids[
                0
            ]
            adapter._reset_positions()
            changed_padding = adapter.model(**padded_inputs, use_cache=False).logits
            valid = padded_inputs["attention_mask"].bool()
            padding_delta = float(
                (batch_logits[valid].float() - changed_padding[valid].float()).abs().max()
            )
            padding_baseline = float(
                (batch_logits[valid].float() - repeated_batch[valid].float()).abs().max()
            )
            exact_causal = exact_causal and padding_delta <= padding_baseline
            padding_diagnostics.append(
                {
                    "tasks": [x.task_id for x in group],
                    "padding_logits_max_abs": padding_delta,
                    "same_batch_logits_max_abs": padding_baseline,
                }
            )
            # Keep only target values before releasing the large vocabulary tensors.
            batch_values = [batch_nll[k][batch_mask[k]].cpu() for k in range(4)]
            del batch_logits, repeated_batch, changed_padding
            for k, example in enumerate(group):
                logits, labels = forward_batch(adapter, [example])
                means, nll, mask = per_sequence_nll(logits, labels)
                values = nll[0][mask[0]].cpu()
                repeat, _ = forward_batch(adapter, [example])
                repeat_nll = per_sequence_nll(repeat, labels)[1][0][mask[0]].cpu()
                cut = len(example.prompt_ids) + max(1, len(example.target_ids) // 2)
                inputs, _ = collate_examples(
                    [example], pad_id=adapter.pad_id, device=adapter.device
                )
                # Mutate only strictly future IDs with the same shape and attention.
                inputs["input_ids"][:, cut:] = example.prompt_ids[0]
                adapter._reset_positions()
                mutated = adapter.model(**inputs, use_cache=False).logits
                delta = (logits[:, :cut].float() - mutated[:, :cut].float()).abs().max().item()
                baseline = (logits[:, :cut].float() - repeat[:, :cut].float()).abs().max().item()
                exact_causal = exact_causal and delta <= baseline
                prefix = None
                if rows is not None:
                    prepared = adapter.prepare(rows[offset + k]["prompt"], data_root)
                    prefix = (
                        (-adapter.logprobs(prepared, list(example.target_ids), require_grad=False))
                        .cpu()
                        .tolist()
                    )
                    adapter.model.train()
                diagnostics.append(
                    {
                        "task_id": example.task_id,
                        "prompt_tokens": len(example.prompt_ids),
                        "target_ids": list(example.target_ids),
                        "first_target_logit_index": len(example.prompt_ids) - 1,
                        "full_token_nll": values.tolist(),
                        "batch_token_nll": batch_values[k].tolist(),
                        "repeat_token_nll": repeat_nll.tolist(),
                        "prefix_token_nll": prefix,
                        "sequence_nll": float(means[0]),
                        "batch_sequence_nll": float(batch_mean[k]),
                        "future_logits_max_abs": delta,
                        "same_path_logits_max_abs": baseline,
                    }
                )
                (output / "bridge_token_diagnostics.json").write_text(
                    json.dumps({"examples": diagnostics, "padding": padding_diagnostics}, indent=2)
                    + "\n"
                )
                del logits, repeat, mutated
    runtime.restore(initial)
    gradient_runs = []
    for micro in (4, 4, 1):
        runtime.optimizer.zero_grad(set_to_none=True)
        accumulated_backward(adapter, examples[:4], microbatch_size=micro)
        gradient_runs.append(_gradient_snapshot(runtime))
    baseline_grad = _gradient_comparison(gradient_runs[0], gradient_runs[1])
    micro_grad = _gradient_comparison(gradient_runs[0], gradient_runs[2])
    if not micro_grad["finite"] or micro_grad["left_norm"] == 0 or micro_grad["right_norm"] == 0:
        raise RuntimeError("Missing/nonfinite bridge gradients")
    prefix_gradient = None
    if rows is not None:
        # One bounded example compares the new objective to the inherited prefix path.
        runtime.optimizer.zero_grad(set_to_none=True)
        full_logits, full_labels = forward_batch(adapter, examples[:1])
        per_sequence_nll(full_logits, full_labels)[0].mean().backward()
        full_gradient = _gradient_snapshot(runtime)
        del full_logits, full_labels
        runtime.optimizer.zero_grad(set_to_none=True)
        prepared = adapter.prepare(rows[0]["prompt"], data_root)
        prefix_loss = -adapter.logprobs(
            prepared, list(examples[0].target_ids), require_grad=True
        ).mean()
        prefix_loss.backward()
        prefix_gradient = _gradient_comparison(full_gradient, _gradient_snapshot(runtime))
        del prefix_loss, full_gradient
    (output / "bridge_gradient_diagnostics.json").write_text(
        json.dumps(
            {
                "same_path": baseline_grad,
                "microbatch": micro_grad,
                "full_vs_prefix": prefix_gradient,
            },
            indent=2,
        )
        + "\n"
    )
    runtime.restore(initial)
    first = runtime.update()
    after_first = runtime.capture()
    first_checkpoint = runtime.save(output / "bridge_step001.pt")
    generated = []
    # Two actual updated-policy generations; 2 is below the total bridge cap128.
    if rows is not None:
        for index in (0, 1):
            prepared = adapter.prepare(rows[index]["prompt"], data_root)
            with torch.no_grad():
                value = adapter.generate(
                    prepared, seed=83101 + index, max_new_tokens=64, do_sample=True
                )
            generated.append({"task_id": examples[index].task_id, "student_update": 1, **value})
    # Generation must not perturb the compared training RNG / state.
    runtime.restore(after_first)
    second = runtime.update()
    expected = runtime.capture()
    runtime.resume(first_checkpoint["path"])
    repeated_second = runtime.update()
    resumed = runtime.capture()
    resume_exact = state_hash(expected) == state_hash(resumed)
    runtime.restore(initial)
    result = {
        "schema": "verified-discovery-sft-bridge-v1",
        "execution_kind": "REAL_CUDA_TECHNICAL_BRIDGE",
        "examples": diagnostics,
        "padding_diagnostics": padding_diagnostics,
        "causal_within_same_path_baseline": exact_causal,
        "gradient_same_path": baseline_grad,
        "gradient_microbatch": micro_grad,
        "gradient_full_vs_prefix": prefix_gradient,
        "update1": first,
        "update2": second,
        "resumed_update2": repeated_second,
        "resume_exact": resume_exact,
        "updated_generations": generated,
        "generation_draws": len(generated),
        "generation_cap": 128,
        "formal_training_updates": 0,
        "bridge_weights_restored_to_parent": True,
        "next_action": (
            "Review BF16 deltas and causality before formal SFT; reload parent for every arm"
        ),
    }
    result.update(assess_bridge_diagnostics(result))
    (output / "SFT_BRIDGE.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def assess_bridge_diagnostics(result):
    """Fail closed on incomplete evidence; compare prefix errors to observed noise.

    This is a pure readout: no forward calls, revised budgets, or BF16 tolerance.
    Gradient comparisons use dimensionless errors because the one-example prefix
    gradient and four-example microbatch diagnostic have different loss scales.
    """
    invalid, causal, review = [], [], []

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    examples = result.get("examples")
    if not isinstance(examples, list) or len(examples) != 8:
        invalid.append("Eight complete train-only token diagnostics are required")
        examples = []
    task_ids = []
    for index, example in enumerate(examples):
        label = f"example[{index}]"
        if not isinstance(example, dict):
            invalid.append(f"{label}: missing diagnostic object")
            continue
        task_id = example.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            invalid.append(f"{label}: missing task identity")
        else:
            task_ids.append(task_id)
        target_ids = example.get("target_ids")
        if (
            not isinstance(target_ids, list)
            or not target_ids
            or any(type(t) is not int or t < 0 for t in target_ids)
        ):
            invalid.append(f"{label}: missing target token identities")
            target_ids = []
        prompt_tokens = example.get("prompt_tokens")
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or example.get("first_target_logit_index") != prompt_tokens - 1
        ):
            invalid.append(f"{label}: incorrect or missing first-target shift")
        vectors = {}
        for key in ("full_token_nll", "repeat_token_nll", "batch_token_nll", "prefix_token_nll"):
            values = example.get(key)
            if (
                not isinstance(values, list)
                or not values
                or len(values) != len(target_ids)
                or not all(finite(value) for value in values)
            ):
                invalid.append(f"{label}: missing, nonfinite or misaligned {key}")
            else:
                vectors[key] = values
        if len(vectors) == 4:
            full, repeat, prefix = (
                vectors[key] for key in ("full_token_nll", "repeat_token_nll", "prefix_token_nll")
            )
            if any(abs(p - f) > abs(r - f) for f, r, p in zip(full, repeat, prefix, strict=True)):
                review.append(f"{label}: prefix/full token difference exceeds same-path noise")
            # Preserve the existing conservative batch gate.
            if full != vectors["batch_token_nll"]:
                review.append(f"{label}: batch/full token values differ")
        for key in (
            "sequence_nll",
            "batch_sequence_nll",
            "future_logits_max_abs",
            "same_path_logits_max_abs",
        ):
            if not finite(example.get(key)):
                invalid.append(f"{label}: missing or nonfinite {key}")
        delta, baseline = (
            example.get("future_logits_max_abs"),
            example.get("same_path_logits_max_abs"),
        )
        if finite(delta) and finite(baseline):
            if delta < 0 or baseline < 0:
                invalid.append(f"{label}: negative absolute logit error")
            elif delta > baseline:
                causal.append(f"{label}: future suffix changes earlier logits beyond baseline")
    if examples and len(set(task_ids)) != 8:
        invalid.append("Eight distinct task identities are required")

    padding = result.get("padding_diagnostics")
    if not isinstance(padding, list) or len(padding) != 2:
        invalid.append("Two padding diagnostic batches are required")
        padding = []
    padded_tasks = []
    for index, item in enumerate(padding):
        if not isinstance(item, dict):
            invalid.append(f"padding[{index}]: missing diagnostic object")
            continue
        tasks = item.get("tasks")
        if (
            not isinstance(tasks, list)
            or len(tasks) != 4
            or not all(isinstance(t, str) for t in tasks)
        ):
            invalid.append(f"padding[{index}]: four task identities required")
        else:
            padded_tasks.extend(tasks)
        delta, baseline = item.get("padding_logits_max_abs"), item.get("same_batch_logits_max_abs")
        if not finite(delta) or not finite(baseline) or delta < 0 or baseline < 0:
            invalid.append(f"padding[{index}]: missing or invalid numerical evidence")
        elif delta > baseline:
            causal.append(f"padding[{index}]: masked padding changes valid logits beyond baseline")
    if padding and sorted(padded_tasks) != sorted(task_ids):
        invalid.append("Padding batches do not cover the eight token diagnostic tasks exactly")

    gradients = {}
    for key in ("gradient_same_path", "gradient_microbatch", "gradient_full_vs_prefix"):
        value = result.get(key)
        fields = ("cosine", "relative_l2", "left_norm", "right_norm", "max_abs")
        if (
            not isinstance(value, dict)
            or value.get("finite") is not True
            or any(not finite(value.get(field)) for field in fields)
        ):
            invalid.append(f"Missing or nonfinite {key}")
        elif (
            value["left_norm"] <= 0
            or value["right_norm"] <= 0
            or value["relative_l2"] < 0
            or value["max_abs"] < 0
        ):
            invalid.append(f"Zero gradient or invalid error in {key}")
        else:
            gradients[key] = value
    if len(gradients) == 3:
        baseline = gradients["gradient_same_path"]
        for key in ("gradient_microbatch", "gradient_full_vs_prefix"):
            value = gradients[key]
            if value["relative_l2"] > baseline["relative_l2"] or abs(1 - value["cosine"]) > abs(
                1 - baseline["cosine"]
            ):
                review.append(f"{key}: gradient difference exceeds same-path noise")

    for key in ("causal_within_same_path_baseline", "resume_exact"):
        if type(result.get(key)) is not bool:
            invalid.append(f"Missing boolean evidence: {key}")
        elif not result[key]:
            causal.append(f"Failed required evidence: {key}")
    status = (
        "BLOCKED_INVALID_DIAGNOSTICS"
        if invalid
        else "BLOCKED_CAUSALITY_OR_RESUME"
        if causal
        else "NUMERICAL_REVIEW_REQUIRED"
        if review
        else "PASS"
    )
    return {"status": status, "diagnostic_gate_reasons": invalid + causal + review}
