"""Fixed-sequence R1 measurements; no cache forks or production engine changes."""

from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict

from .core import canonical_hash


def reference_panel(scenes):
    cells = defaultdict(list)
    for row in scenes:
        if row["split"] != "calibration":
            raise ValueError("R1 may only use calibration")
        cells[(row["constraint_family"], row["chart_type"])].append(row)
    if len(cells) != 6:
        raise ValueError("Expected three families and two chart types")
    selected = []
    for _key, rows in sorted(cells.items()):
        ranked = sorted(rows, key=lambda r: canonical_hash(["R1-reference-v1", r["base_scene_id"]]))
        if len(ranked) < 2:
            raise ValueError("Insufficient calibration cell")
        selected.extend(ranked[:2])
    if len({r["base_scene_id"] for r in selected}) != 12:
        raise ValueError("Reference panel must contain 12 distinct scenes")
    return selected


def parity_comparison(reference, candidate, config):
    import numpy as np

    rt, rp = reference
    ct, cp = candidate
    if not rp or len({len(rt), len(rp), len(cp), len(ct) if ct is not None else len(rp)}) != 1:
        raise ValueError("Equal nonempty fixed-sequence scores required")
    delta = np.asarray(cp, dtype=float) - np.asarray(rp, dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite parity scores")
    mean, p99 = float(np.abs(delta).mean()), float(np.quantile(np.abs(delta), 0.99))
    agreement = None if ct is None else sum(a == b for a, b in zip(rt, ct, strict=True)) / len(rt)
    return {
        "tokens": len(rp),
        "mean_abs_token_logp": mean,
        "p99_abs_token_logp": p99,
        "max_abs_token_logp": float(np.abs(delta).max()),
        "top1_agreement": agreement,
        "top1_measured": ct is not None,
        "sequence_log_ratio_candidate_minus_reference": float(delta.sum()),
        "passed": mean <= config["parity_alarm_mean_abs_token_logp"]
        and p99 <= config["parity_alarm_p99_abs_token_logp"]
        and (agreement is None or agreement >= config["parity_alarm_min_top1_agreement"]),
    }


def prefix_scores(adapter, prepared, completion):
    """The exact no-cache, last-row head path used by reference generation."""
    import torch

    top, scores = [], []
    adapter.model.eval()
    with torch.no_grad():
        for i, token in enumerate(completion):
            adapter._reset_positions()
            logits = (
                adapter.model(
                    **adapter.full_inputs(prepared, completion[:i]),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, -1]
                .float()
            )
            top.append(int(logits.argmax()))
            scores.append(float(logits.log_softmax(-1)[token]))
    return top, scores


def fixed_sequence_audit(adapter, prepared, completion, config):
    """Replay only: this never represents a synthetic action as generated evidence."""
    reference = prefix_scores(adapter, prepared, completion)
    candidates = {
        "reference_repeat": prefix_scores(adapter, prepared, completion),
    }
    for name, require_grad in (("evaluation_likelihood", False), ("training_likelihood", True)):
        scores = adapter.logprobs(prepared, completion, require_grad=require_grad)
        candidates[name] = (None, scores.detach().cpu().tolist())
        del scores
    return {
        name: {
            "path": name,
            "reference_top1": reference[0],
            "reference_logp": reference[1],
            "candidate_top1": candidate[0],
            "candidate_logp": candidate[1],
            "comparison": parity_comparison(reference, candidate, config),
        }
        for name, candidate in candidates.items()
    }


def reference_execution_audit(adapter, prepared, generated, config):
    """Measure the actual production path, independently of rejected accelerations."""
    checks = fixed_sequence_audit(adapter, prepared, generated["token_ids"], config)
    reference = checks["reference_repeat"]
    candidate = (generated["behavior_top1_token_ids"], generated["behavior_token_logprobs"])
    checks["behavior"] = {
        "path": "behavior",
        "reference_top1": reference["reference_top1"],
        "reference_logp": reference["reference_logp"],
        "candidate_top1": candidate[0],
        "candidate_logp": candidate[1],
        "comparison": parity_comparison(
            (reference["reference_top1"], reference["reference_logp"]), candidate, config
        ),
    }
    return checks


def production_gate(checks, optimizations, expected_sequences=120):
    required = ("reference_repeat", "behavior", "evaluation_likelihood", "training_likelihood")
    passed = all(
        checks.get(name, {}).get("passed") is True
        and checks[name].get("sequences") == expected_sequences
        and checks[name].get("failed_sequence_checks", 0) == 0
        and (
            name not in ("reference_repeat", "behavior")
            or checks[name].get("top1_measured") is True
        )
        for name in required
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "selected_path": "uncached_prefix_recompute",
        "checks": checks,
        "adopted_optimization_paths": [],
        "rejected_optimization_paths": sorted(
            k for k, v in optimizations.items() if not v["passed"]
        ),
        "ordinary_cache_adopted": False,
        "I4": "NOT_RUN",
    }


def collate_full_sequences(adapter, items, completions, side):
    """Pad only at the batch boundary; image tensors retain item ordering."""
    import torch
    import torch.nn.functional as F

    if side not in ("left", "right") or not items or len(items) != len(completions):
        raise ValueError("Invalid batch or padding side")
    full = [
        adapter.full_inputs(item, tokens) for item, tokens in zip(items, completions, strict=True)
    ]
    lengths = [x["input_ids"].shape[1] for x in full]
    width = max(lengths)
    allowed = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
    if any(set(x) - allowed for x in full):
        raise ValueError("Unreviewed processor batch field")
    result = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key == "mm_token_type_ids" and not any(key in x for x in full):
            continue
        rows = []
        for x, length in zip(full, lengths, strict=True):
            value = x.get(key, torch.zeros_like(x["input_ids"]))
            pads = (width - length, 0) if side == "left" else (0, width - length)
            rows.append(F.pad(value, pads, value=adapter.pad_id if key == "input_ids" else 0))
        result[key] = torch.cat(rows)
    for key in ("pixel_values", "image_grid_thw"):
        values = [x[key] for x in full if key in x]
        if values:
            result[key] = torch.cat(values)
    starts = [
        (width - length if side == "left" else 0) + item["audit"]["prompt_token_count"] - 1
        for item, length in zip(items, lengths, strict=True)
    ]
    return result, starts


def batch_teacher_scores(adapter, items, completions, side="left"):
    import torch

    adapter.model.eval()
    adapter._reset_positions()
    inputs, starts = collate_full_sequences(adapter, items, completions, side)
    inputs = {
        key: value.to(
            adapter.device,
            dtype=(
                next(adapter.model.parameters()).dtype if value.is_floating_point() else value.dtype
            ),
        )
        for key, value in inputs.items()
    }
    with torch.no_grad():
        output = adapter.model(**inputs, use_cache=False)
        return [
            adapter._summarize_scores(output.logits[i, start : start + len(tokens)], tokens)
            for i, (start, tokens) in enumerate(zip(starts, completions, strict=True))
        ]


class ForwardMeter:
    """Count a single shared backbone boundary, without duplicate wrapper hooks."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.reason = "audit"
        self.calls = defaultdict(int)
        self.token_slots = defaultdict(int)
        self.padding = defaultdict(int)
        self.seconds = defaultdict(float)
        self.sync_seconds = 0.0
        self.generation_first = False
        base = (
            adapter.model.get_base_model()
            if hasattr(adapter.model, "get_base_model")
            else adapter.model
        )
        self.handle = base.model.register_forward_pre_hook(self._before, with_kwargs=True)

    def _before(self, module, args, kwargs):
        reason = self.reason
        if reason == "generation":
            reason = "generation_prefix_recompute"
            self.generation_first = False
        self.calls[reason] += 1
        ids = kwargs.get("input_ids")
        if ids is not None:
            self.token_slots[reason] += ids.numel()
            mask = kwargs.get("attention_mask")
            if mask is not None and mask.shape == ids.shape:
                self.padding[reason] += int((mask == 0).sum())

    def sync(self):
        import torch

        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.sync_seconds += time.perf_counter() - start

    @contextlib.contextmanager
    def scope(self, reason):
        previous = self.reason
        self.reason = reason
        self.generation_first = reason == "generation"
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[reason] += time.perf_counter() - start
            self.reason = previous

    def report(self):
        return {
            "forward_by_reason": dict(self.calls),
            "input_token_slots_by_reason": dict(self.token_slots),
            "padding_token_slots_by_reason": dict(self.padding),
            "seconds_by_reason": dict(self.seconds),
            "device_sync_seconds": self.sync_seconds,
            "vision_encoder_calls": self.adapter.vision_forward_calls,
            "count_boundary": "shared multimodal backbone, once per top-level forward",
            "backward_recompute": "not a new top-level forward; backward calls measured separately",
        }


def aggregate_parity(records, config):
    """Token-weighted statistics plus every sequence ratio, never just categories."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["path"]].append(record)
    result = {}
    for path, rows in grouped.items():
        ref_top1, ref_logp, cand_top1, cand_logp = [], [], [], []
        for row in rows:
            ref_top1.extend(row["reference_top1"])
            ref_logp.extend(row["reference_logp"])
            if row["candidate_top1"] is not None:
                cand_top1.extend(row["candidate_top1"])
            cand_logp.extend(row["candidate_logp"])
        ref = (ref_top1, ref_logp)
        cand = (cand_top1 or None, cand_logp)
        result[path] = {
            **parity_comparison(ref, cand, config),
            "sequences": len(rows),
            "sequence_log_ratios": [
                math.fsum(r["candidate_logp"]) - math.fsum(r["reference_logp"]) for r in rows
            ],
            "failed_sequence_checks": sum(not r["comparison"]["passed"] for r in rows),
        }
    return result
