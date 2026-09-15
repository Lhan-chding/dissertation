"""Physical operation ledger; reused information is never counted as new samples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

COUNTERS = (
    "generated_actions",
    "generated_sequences",
    "sample_policy_evaluations",
    "cross_policy_action_scores",
    "verified_labels",
    "exact_probability_calls",
    "Jacobian_VJP_calls",
    "optimizer_updates",
    "physical_policy_forwards",
    "score_requests",
    "score_cache_hits",
    "label_cache_hits",
    "reused_actions",
    "sampling_seconds",
    "score_seconds",
    "forward_seconds",
    "fit_seconds",
    "io_seconds",
    "added_disk_bytes",
    "peak_ram_bytes",
)


@dataclass
class CostLedger:
    values: dict = field(default_factory=lambda: {key: 0 for key in COUNTERS})

    def add(self, key, value=1):
        if key not in self.values or not math.isfinite(value) or value < 0:
            raise ValueError("Known cost field and finite nonnegative value required")
        self.values[key] += value

    def snapshot(self):
        return dict(self.values)

    def delta(self, before):
        return {key: value - before.get(key, 0) for key, value in self.values.items()}


def scenario_cost(cost, score_to_generation=0.25, label_to_generation=0.0):
    if any(not math.isfinite(v) or v < 0 for v in (score_to_generation, label_to_generation)):
        raise ValueError("Finite nonnegative scenario ratios required")
    return (
        cost.get("generated_actions", 0)
        + score_to_generation * cost.get("cross_policy_action_scores", 0)
        + label_to_generation * cost.get("verified_labels", 0)
    )


def break_even(calibration_cost, prediction_cost, direct_cost):
    if any(not math.isfinite(v) or v < 0 for v in (calibration_cost, prediction_cost, direct_cost)):
        raise ValueError("Finite nonnegative costs required")
    saving = direct_cost - prediction_cost
    if saving <= 0:
        return {"status": "NO_COST_FEASIBLE_SURROGATE", "queries": None}
    return {
        "status": "CONDITIONAL_COST_CROSSING",
        "queries": max(1, math.ceil(calibration_cost / saving)),
    }
