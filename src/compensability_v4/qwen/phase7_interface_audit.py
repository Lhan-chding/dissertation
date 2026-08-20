"""Objective interface diagnostics derived from frozen Phase 7 trace evidence."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

from compensability_v4.eval.statistics import holm_adjust

_CHECKPOINTS = frozenset(
    {
        "Base",
        "C0",
        "C1",
        "T",
        "Base_AnswerOnly_RL",
        "Recovery_LoRA_RecoveryOutcome_RL",
        "Recovery_LoRA_AnswerOnly_RL",
    }
)


def _integer_or_none(value: object, label: str) -> int | None:
    if value is not None and type(value) is not int:
        raise TypeError(f"{label} must be an integer or null")
    return value


def _validated(rows: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    frozen: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema_version") != 1:
            raise ValueError("Phase 7 interface evidence row is malformed")
        chain = row.get("chain_row")
        if not isinstance(chain, Mapping):
            raise ValueError("Phase 7 interface evidence chain_row is malformed")
        scene_id, checkpoint = chain.get("scene_id"), chain.get("checkpoint")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        if checkpoint not in _CHECKPOINTS:
            raise ValueError("checkpoint is outside the frozen Phase 7 checkpoint set")
        raw, parse_success = row.get("final_answer_raw"), row.get("final_answer_parse_success")
        if not isinstance(raw, str):
            raise TypeError("final_answer_raw must be a string")
        if type(parse_success) is not bool:
            raise TypeError("final_answer_parse_success must be boolean")
        parsed = _integer_or_none(row.get("final_answer"), "final_answer")
        executed = _integer_or_none(
            row.get("chosen_operation_execution"), "chosen_operation_execution"
        )
        truth = _integer_or_none(row.get("ground_truth_answer"), "ground_truth_answer")
        exact = chain.get("final_answer_exact")
        if type(exact) is not bool or truth is None:
            raise TypeError("final_answer_exact must be boolean and ground_truth_answer integer")
        if parse_success is not (parsed is not None):
            raise ValueError("final answer parse label contradicts the frozen trace")
        if exact is not (parsed == truth):
            raise ValueError("final answer exact label contradicts the frozen trace")
        identity = (scene_id, str(checkpoint))
        if identity in identities:
            raise ValueError("Phase 7 interface evidence identity must be unique")
        identities.add(identity)
        frozen.append(
            {
                "scene_id": scene_id,
                "checkpoint": checkpoint,
                "parse_success": parse_success,
                "free_exact": exact,
                "executor_exact": executed == truth,
                "trace_consistent": parsed is not None
                and executed is not None
                and parsed == executed,
            }
        )
    if not frozen:
        raise ValueError("Phase 7 interface evidence must not be empty")
    return tuple(sorted(frozen, key=lambda row: (row["checkpoint"], row["scene_id"])))


def _interval(
    values: Mapping[str, float], *, resamples: int, seed: int, confidence: float = 0.95
) -> dict[str, float | int]:
    observed = tuple(values.values())
    estimate = sum(observed) / len(observed)
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(observed) for _ in observed) / len(observed) for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    low = max(0, min(resamples - 1, int(alpha * resamples)))
    high = max(0, min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1))
    return {
        "estimate": estimate,
        "ci_low": boot[low],
        "ci_high": boot[high],
        "confidence": confidence,
        "number_of_scenes": len(observed),
    }


def _sign_flip(values: tuple[float, ...], *, seed: int) -> float:
    observed = abs(sum(values) / len(values))
    if observed == 0.0:
        return 1.0
    rng = random.Random(seed)
    total = 10_000
    count = sum(
        abs(sum(value if rng.getrandbits(1) else -value for value in values) / len(values))
        >= observed - 1e-15
        for _ in range(total)
    )
    return (count + 1) / (total + 1)


def _paired(
    rows: tuple[dict[str, object], ...],
    *,
    before: str,
    after: str,
    resamples: int,
    seed: int,
    margin: float,
) -> dict[str, object]:
    def values(checkpoint: str, metric: str) -> dict[str, float]:
        return {
            str(row["scene_id"]): float(row[metric])
            for row in rows
            if row["checkpoint"] == checkpoint
        }

    before_executor, after_executor = (
        values(before, "executor_exact"),
        values(after, "executor_exact"),
    )
    before_free, after_free = values(before, "free_exact"), values(after, "free_exact")
    shared = sorted(
        before_executor.keys() & after_executor.keys() & before_free.keys() & after_free.keys()
    )
    if not shared:
        return {"estimate": None, "paired_scene_count": 0}
    executor = {scene: after_executor[scene] - before_executor[scene] for scene in shared}
    free = {scene: after_free[scene] - before_free[scene] for scene in shared}
    interval = _interval(executor, resamples=resamples, seed=seed)
    tost = _interval(executor, resamples=resamples, seed=seed + 10_000, confidence=0.90)
    free_estimate = sum(free.values()) / len(free)
    differences = tuple(executor[scene] for scene in shared)
    return {
        **interval,
        "paired_scene_count": len(shared),
        "two_sided_sign_flip_p_value": _sign_flip(differences, seed=seed),
        "holm_adjusted_p_value": None,
        "free_generation_registered_estimate": free_estimate,
        "interface_contribution_estimate": free_estimate - float(interval["estimate"]),
        "tost": {
            "method": "scene_clustered_percentile_bootstrap_ci",
            "margin": margin,
            "confidence": 0.90,
            "ci_low": tost["ci_low"],
            "ci_high": tost["ci_high"],
            "equivalent": bool(float(tost["ci_low"]) > -margin and float(tost["ci_high"]) < margin),
        },
    }


def summarize_phase7_interface_evidence(
    rows: Iterable[Mapping[str, object]],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 2026082101,
    tost_margin: float = 0.02,
) -> dict[str, object]:
    """Separate strict free generation from deterministic chain execution evidence."""
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise TypeError("bootstrap_seed must be an integer")
    if isinstance(tost_margin, bool) or not isinstance(tost_margin, (int, float)):
        raise TypeError("tost_margin must be numeric")
    if not math.isfinite(float(tost_margin)) or tost_margin <= 0.0:
        raise ValueError("tost_margin must be positive and finite")
    frozen = _validated(rows)
    checkpoints = sorted({str(row["checkpoint"]) for row in frozen})
    by_checkpoint: dict[str, dict[str, int | float]] = {}
    for checkpoint in checkpoints:
        selected = tuple(row for row in frozen if row["checkpoint"] == checkpoint)
        count = len(selected)
        parse_count = sum(bool(row["parse_success"]) for row in selected)
        free_count = sum(bool(row["free_exact"]) for row in selected)
        executor_count = sum(bool(row["executor_exact"]) for row in selected)
        consistent_count = sum(bool(row["trace_consistent"]) for row in selected)
        by_checkpoint[checkpoint] = {
            "number_of_rows": count,
            "number_of_scenes": len({str(row["scene_id"]) for row in selected}),
            "final_answer_parse_count": parse_count,
            "final_answer_parse_rate": parse_count / count,
            "free_generation_answer_exact_count": free_count,
            "free_generation_answer_exact_rate": free_count / count,
            "deterministic_chain_answer_exact_count": executor_count,
            "deterministic_chain_answer_exact_rate": executor_count / count,
            "parsed_trace_consistent_count": consistent_count,
            "parsed_trace_consistent_rate": consistent_count / count,
        }
    effects = {
        name: _paired(
            frozen,
            before=before,
            after="T",
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + index,
            margin=float(tost_margin),
        )
        for index, (name, before) in enumerate((("T_minus_C0", "C0"), ("T_minus_C1", "C1")))
    }
    available = {
        name: float(effect["two_sided_sign_flip_p_value"])
        for name, effect in effects.items()
        if effect.get("two_sided_sign_flip_p_value") is not None
    }
    for name, adjusted in holm_adjust(available).items():
        effects[name]["holm_adjusted_p_value"] = adjusted
    return {
        "schema_version": 1,
        "status": "PHASE_7_INTERFACE_DIAGNOSTIC_AUDITED",
        "number_of_rows": len(frozen),
        "number_of_scenes": len({str(row["scene_id"]) for row in frozen}),
        "free_generation_evidence_preserved": True,
        "post_hoc_parser_relaxation_applied": False,
        "by_checkpoint": by_checkpoint,
        "deterministic_chain_effects": effects,
    }


__all__ = ["summarize_phase7_interface_evidence"]
