"""One registered SR-F5 update; this module never samples or selects data.

The caller commits a checkpoint only after this function succeeds. A failure
before optimizer.step preserves numerical evidence and consumes no Adam step.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from sr_f3_ce.training import validate_optimizer
from sr_f4.common import append, now
from sr_f12.training import ProbabilityMismatch, enforce_probability_gate
from srf5_ref.credit import ARMS
from srf5_ref.credit import credits as calculate_credits
from srf5_ref.objective import sequence_loss

BATCH_SIZE = 128
DIAGNOSTIC_STEPS = frozenset((1, 12, 36, 72))


def _mask(record):
    mask = record.get("selected_mask", record.get("mask"))
    if not isinstance(mask, list) or len(mask) != len(record["tokens"]):
        raise ValueError("Selected mask must match the original sampled token vector")
    if any(type(value) is not bool for value in mask):
        raise ValueError("Selected token masks must be boolean")
    if type(record.get("mask_ok")) is not bool or record["mask_ok"] != any(mask):
        raise ValueError("mask_ok must agree with the mapped token mask")
    return list(mask)


def _credit_rows(records):
    rows = []
    for record in records:
        score = record["score"]
        for name in ("J", "V", "finish", "value_correct"):
            if type(score[name]) not in (int, bool) or score[name] not in (0, 1):
                raise ValueError(f"Nonbinary {name} label")
        if "J_M3" in score and score["J_M3"] != score["J"]:
            raise ValueError("J must denote the common protocol J_M3, not HARD reward")
        _mask(record)
        rows.append({**score, "qid": record["qid"], "mask_ok": record["mask_ok"]})
    return rows


def build_credit_group(records, arm, seed):
    """Bind the exact applied and counterfactual coefficients to saved rows."""
    if len(records) != 8 or arm not in ARMS or type(seed) is not int:
        raise ValueError("One registered arm, integer seed and complete G8 required")
    if len({r["group_id"] for r in records}) != 1 or len({r["qid"] for r in records}) != 1:
        raise ValueError("Credits cannot mix questions or independent groups")
    ids = [r["row_id"] for r in records]
    if len(set(ids)) != 8 or not all(isinstance(v, str) and v for v in ids):
        raise ValueError("Every generated row needs a unique nonempty row_id")
    result = calculate_credits(_credit_rows(records), arm, seed)
    return {
        **result,
        "seed": seed,
        "group_id": records[0]["group_id"],
        "qid": records[0]["qid"],
        "row_ids": ids,
        "selected_masks": [_mask(r) for r in records],
    }


def _validate(records, group_sizes, credits):
    if group_sizes != [8] * 16 or len(records) != BATCH_SIZE or len(credits) != 16:
        raise ValueError("Exactly sixteen G8 groups and all 128 answers are required")
    if len({r["row_id"] for r in records}) != BATCH_SIZE:
        raise ValueError("Duplicate generated row identity")
    if len({c["group_id"] for c in credits}) != 16:
        raise ValueError("Duplicate independent group identity")
    if len({c["arm"] for c in credits}) != 1:
        raise ValueError("An update must belong to exactly one training arm")
    roots = {str(r["run_root"]) for r in records}
    if len(roots) != 1:
        raise ValueError("An update cannot mix run roots")
    output = []
    for offset, receipt in zip(range(0, BATCH_SIZE, 8), credits, strict=True):
        group = records[offset : offset + 8]
        for record in group:
            tokens = record["tokens"]
            if not isinstance(tokens, list) or not 1 <= len(tokens) <= 1024:
                raise ValueError("Original completion token count outside frozen bounds")
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("Original token IDs must be unsigned integers")
            if len(record["sampler_logprobs"]) != len(tokens):
                raise ValueError("Sampler probabilities must align with sampled tokens")
            if record.get("generation_status", "COMPLETE") != "COMPLETE":
                raise ValueError("Technically invalid generation cannot enter an update")
            if record.get("technical_validation_errors"):
                raise ValueError("Unresolved generation errors cannot enter an update")
        expected = build_credit_group(group, receipt["arm"], receipt["seed"])
        if receipt != expected:
            raise ValueError("Saved credit receipt differs from recomputed actual coefficients")
        for index, record in enumerate(group):
            output.append(
                {
                    "row_id": record["row_id"],
                    "native": receipt["native"][index],
                    "aux": receipt["aux"][index],
                    "scope": receipt["scope"],
                    "mask": receipt["selected_masks"][index],
                }
            )
    return output


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _quantiles(values):
    import torch

    if not bool(torch.isfinite(values).all()):
        return {name: None for name in ("min", "p50", "p95", "p99", "p99_9", "max")}
    fractions = {"min": 0.0, "p50": 0.5, "p95": 0.95, "p99": 0.99, "p99_9": 0.999, "max": 1.0}
    return {
        key: float(torch.quantile(values, probability)) for key, probability in fractions.items()
    }


def numerical_audit(traces):
    """Retain distinct sampler/train and reference/current tails, with row IDs."""
    import torch

    if not traces:
        return {"token_count": 0, "status": "NO_FORWARDS"}
    sampler = torch.tensor([p for row in traces for p in row["sampler_logprobs"]])
    current = torch.tensor([p for row in traces for p in row["train_logprobs"]])
    reference = torch.tensor([p for row in traces for p in row["reference_logprobs"]])
    delta = current - sampler
    absolute = delta.abs()
    displacement = reference - current
    kl = torch.expm1(displacement) - displacement
    flat = []
    for row in traces:
        for i, token in enumerate(row["tokens"]):
            flat.append(
                {
                    "row_id": row["row_id"],
                    "qid": row["qid"],
                    "token_position": i,
                    "token_id": token,
                    "sampler_logprob": _finite_number(row["sampler_logprobs"][i]),
                    "train_logprob": _finite_number(row["train_logprobs"][i]),
                    "reference_logprob": _finite_number(row["reference_logprobs"][i]),
                }
            )
    for i, point in enumerate(flat):
        point.update(
            sampler_train_delta=_finite_number(delta[i]),
            reference_current_delta=_finite_number(displacement[i]),
            kl_token=_finite_number(kl[i]),
        )
    finite = bool(torch.isfinite(absolute).all()) and bool(torch.isfinite(kl).all())
    quantiles = _quantiles(absolute)
    largest_kl = torch.topk(kl, min(10, len(flat))).indices.tolist()
    largest_mismatch = torch.topk(absolute, min(10, len(flat))).indices.tolist()
    total = float(kl.sum())
    top_total = float(kl[largest_kl].sum())
    return {
        "status": "FINITE" if finite else "NONFINITE",
        "token_count": len(flat),
        "sampler_train_mean_absolute_difference": _finite_number(absolute.mean()),
        "sampler_train_p99_absolute_difference": quantiles["p99"],
        "sampler_train_max_absolute_difference": quantiles["max"],
        "sampler_train_absolute_quantiles": quantiles,
        "reference_current_signed_quantiles": _quantiles(displacement),
        "tis_truncated_fraction": float((delta > math.log(2)).float().mean()),
        "tis_cap": 2.0,
        "kl_token_max": _finite_number(kl.max()),
        "kl_top10_fraction_of_total": top_total / total if finite and total > 0 else None,
        "kl_nonfinite_tokens": int((~torch.isfinite(kl)).sum()),
        "sampler_train_top10": [flat[i] for i in largest_mismatch],
        "kl_top10": [flat[i] for i in largest_kl],
    }


def _norm(parameters):
    import torch

    vectors = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(vectors).sum().sqrt()) if vectors else 0.0


def _channel_summary(native, auxiliary):

    n2 = sum(float(v.double().square().sum()) for v in native)
    a2 = sum(float(v.double().square().sum()) for v in auxiliary)
    dot = sum(
        float((n.double() * a.double()).sum()) for n, a in zip(native, auxiliary, strict=True)
    )
    return {
        "native_gradient_norm": math.sqrt(n2),
        "auxiliary_gradient_norm": math.sqrt(a2),
        "auxiliary_to_native_norm": math.sqrt(a2 / n2) if n2 else None,
        "native_auxiliary_cosine": dot / math.sqrt(n2 * a2) if n2 and a2 else None,
    }


def update(
    runtime,
    optimizer,
    scheduler,
    records,
    group_sizes,
    reference,
    credits,
    *,
    diagnostic_row_indices=(),
):
    """One /128 update, with every probability gate before optimizer.step.

    Optional channel gradients reuse actual forward graphs at a caller-frozen
    subset. They are permitted only at steps 1/12/36/72 and never enter .grad.
    The caller records the subset in its frozen plan; empty means NOT_RUN.
    """
    import torch

    coefficients = _validate(records, group_sizes, credits)
    for record in records:
        identity = record.get("adapter_identity")
        # Native Runtime calls the parent field "parent" and stores parent_id.
        if not isinstance(identity, dict) or any(
            source not in identity or identity[source] != runtime.identity.get(target)
            for source, target in (("step", "step"), ("run_id", "run_id"), ("parent", "parent_id"))
        ):
            raise ValueError("Rollout adapter identity differs from current committed policy")
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    indices = tuple(diagnostic_row_indices)
    step = int(runtime.identity.get("step", 0)) + 1
    if len(set(indices)) != len(indices) or any(
        type(i) is not int or not 0 <= i < 128 for i in indices
    ):
        raise ValueError("Diagnostic row subset must contain unique registered indices")
    if indices and step not in DIAGNOSTIC_STEPS:
        raise ValueError("Channel gradient diagnostic outside registered steps")
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    before = [p.detach().cpu().clone() for p in parameters]
    channels = (
        [[torch.zeros_like(p, device="cpu") for p in parameters] for _ in range(2)]
        if indices
        else None
    )
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    runtime.model.eval()
    traces, totals = [], {key: 0.0 for key in ("loss", "native", "aux", "kl")}
    previous_input, prepared = None, None
    audit_path = Path(records[0]["run_root"]) / "NUMERICAL_TOKEN_AUDIT.jsonl"
    audit_identity = {
        "at": now(),
        "run_id": runtime.identity.get("run_id"),
        "step": step,
        "arm": credits[0]["arm"],
        "row_ids": [r["row_id"] for r in records],
    }
    for field in ("pilot_lane", "pilot_execution_id"):
        if runtime.identity.get(field) is not None:
            audit_identity[field] = runtime.identity[field]
    try:
        for index, (record, coefficient) in enumerate(zip(records, coefficients, strict=True)):
            key = (record["qid"], record["model_input"], record["protocol"])
            if key != previous_input:
                prepared = runtime.prepare_input(
                    record["model_input"], record["run_root"], protocol=record["protocol"]
                )
                previous_input = key
            item = dict(record, prepared=prepared)
            with torch.no_grad():
                references = runtime.batch_reference_forward([item], reference)
            currents = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)
            if len(currents) != 1 or len(references) != 1:
                raise ValueError("Single-sequence runtime returned wrong row count")
            current, ref = currents[0], references[0]
            sampler = torch.as_tensor(
                record["sampler_logprobs"], device=current.device, dtype=torch.float32
            )
            if (
                current.ndim != 1
                or current.numel() != len(record["tokens"])
                or ref.shape != current.shape
            ):
                raise ValueError("Forward probabilities changed the sampled token path")
            traces.append(
                {
                    "row_id": record["row_id"],
                    "qid": record["qid"],
                    "tokens": list(record["tokens"]),
                    "sampler_logprobs": sampler.detach().cpu().tolist(),
                    "train_logprobs": current.detach().cpu().tolist(),
                    "reference_logprobs": ref.detach().cpu().tolist(),
                }
            )
            result = sequence_loss(
                current,
                sampler,
                ref,
                coefficient["native"],
                coefficient["aux"],
                coefficient["mask"],
                coefficient["scope"],
            )
            if index in indices:
                for target, name in zip(channels, ("native", "aux"), strict=True):
                    gradients = torch.autograd.grad(
                        result[name] / 128, parameters, retain_graph=True, allow_unused=True
                    )
                    for buffer, gradient in zip(target, gradients, strict=True):
                        if gradient is not None:
                            buffer.add_(gradient.detach().cpu())
                if hasattr(runtime, "reserve"):
                    runtime.reserve(
                        "diagnostic_gradient_calls", 2, row_id=record["row_id"], step=step
                    )
            (result["loss"] / 128).backward()
            for name in totals:
                totals[name] += float(result[name].detach()) / 128
        audit = numerical_audit(traces)
        if audit["status"] != "FINITE":
            raise ProbabilityMismatch("Nonfinite probability/KL diagnostics", audit)
        enforce_probability_gate(audit)
        if audit["sampler_train_max_absolute_difference"] > 5:
            raise ProbabilityMismatch(
                "Token sampler/train discrepancy exceeds 5 nat; diagnose before update", audit
            )
        norm_before = float(
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        )
        norm_after = _norm(parameters)
        append(audit_path, {**audit_identity, "status": "PREUPDATE_PASS", "diagnostics": audit})
    except Exception as exc:
        audit = numerical_audit(traces)
        failure = {
            **audit_identity,
            "status": "UNCOMMITTED_FAILURE",
            "error": str(exc),
            "diagnostics": audit,
            "probability_vectors": traces,
        }

        # Nonfinite floats must still be preserved without invalid JSON literals.
        def safe(value):
            if isinstance(value, float):
                return _finite_number(value)
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [safe(v) for v in value]
            return value

        append(audit_path, safe(failure))
        optimizer.zero_grad(set_to_none=True)
        if isinstance(exc, ProbabilityMismatch):
            exc.diagnostics.update(probability_vectors=traces)
        raise
    try:
        optimizer.step()
        scheduler.step()
    except Exception as exc:
        append(
            audit_path,
            {
                **audit_identity,
                "status": "UPDATE_ATTEMPT_FAILED_RESTORE_CHECKPOINT",
                "error": str(exc),
            },
        )
        raise
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        append(
            audit_path, {**audit_identity, "status": "NONFINITE_AFTER_OPTIMIZER_RESTORE_CHECKPOINT"}
        )
        raise FloatingPointError(
            "Nonfinite adapter after update; restore last committed checkpoint"
        )
    update_norm = math.sqrt(
        sum(
            float((p.detach().cpu().double() - old.double()).square().sum())
            for p, old in zip(parameters, before, strict=True)
        )
    )
    return {
        **totals,
        **audit,
        "gradient_norm_before_clip": norm_before,
        "gradient_norm_after_clip": norm_after,
        "parameter_update_norm": update_norm,
        "channel_gradients": {
            "status": "MEASURED",
            "row_indices": list(indices),
            **_channel_summary(*channels),
        }
        if indices
        else {"status": "NOT_RUN"},
        "probability_vectors": traces,
        "train_logprobs": [r["train_logprobs"] for r in traces],
        "reference_logprobs": [r["reference_logprobs"] for r in traces],
        "actual_credit_groups": credits,
        "sequence_count": 128,
        "optimizer_updates": 1,
        "microbatch_size": 1,
        "group_sizes": list(group_sizes),
        "kl_coefficient": 0.02,
        "loss_reduction": "mean_of_128_sequence_means; auxiliary normalized independently",
        "training_seconds": time.monotonic() - started,
    }
