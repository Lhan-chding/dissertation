#!/usr/bin/env python3
"""Create the frozen Phase C power artifact without reading experiment outcomes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from compbias.recoverability.power import (
    PowerCurve,
    PowerCurvePoint,
    PowerSimulationConfig,
    build_fixed_sample_plan,
    simulate_paired_power,
    simulate_paired_tost_power,
)

SCENE_GRID = (400, 600, 800, 1066)
REPETITIONS = 2_000
SEED = 2026081605


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    test: str
    discordance: float
    scene_icc: float
    true_effect: float
    alpha: float


SCENARIOS = (
    Scenario("recoverable_effect", "one_sided_positive", 0.30, 0.40, 0.05, 0.025),
    Scenario("counterfactual_target_shift", "one_sided_positive", 0.30, 0.40, 0.05, 0.025),
    Scenario("counterfactual_original_suppression", "one_sided_positive", 0.25, 0.35, 0.05, 0.025),
    Scenario("sham_equivalence", "paired_tost", 0.10, 0.10, 0.00, 0.05),
    Scenario("nonrecoverable_equivalence", "paired_tost", 0.10, 0.10, 0.00, 0.05),
)


def _curve(scenario: Scenario, scenario_index: int) -> tuple[PowerCurve, dict[str, object]]:
    points: list[PowerCurvePoint] = []
    serialized: list[dict[str, object]] = []
    for grid_index, scenes in enumerate(SCENE_GRID):
        config = PowerSimulationConfig(
            scenes=scenes,
            forks_per_arm=8,
            baseline_rate=0.20,
            target_effect=scenario.true_effect,
            discordance=scenario.discordance,
            scene_icc=scenario.scene_icc,
            alpha=scenario.alpha,
            repetitions=REPETITIONS,
            seed=SEED + scenario_index * 100 + grid_index,
        )
        if scenario.test == "paired_tost":
            result = simulate_paired_tost_power(config, margin=0.02)
        else:
            result = simulate_paired_power(config)
        power = round(result.estimated_power, 6)
        points.append(PowerCurvePoint(scenes, power))
        serialized.append({"estimated_power": power, "scenes": scenes})
    return PowerCurve(scenario.name, tuple(points)), {
        "name": scenario.name,
        "test": scenario.test,
        "discordance": scenario.discordance,
        "scene_icc": scenario.scene_icc,
        "true_effect": scenario.true_effect,
        "alpha": scenario.alpha,
        "baseline_rate": 0.20,
        "curve": serialized,
    }


def build_payload() -> dict[str, object]:
    built = tuple(_curve(scenario, index) for index, scenario in enumerate(SCENARIOS))
    plan = build_fixed_sample_plan(
        tuple(curve for curve, _item in built),
        target_power=0.90,
        eligibility_rate_lower=0.15,
        intake_scenes=8000,
        family_quotas={"cross_series": 400, "trend": 400, "duplicate_encoding": 266},
    )
    if not plan.feasible or plan.required_eligible_scenes != 1066:
        raise RuntimeError("frozen Phase C design is not adequately powered")
    return {
        "schema_version": 1,
        "artifact_type": "recoverability_v1_phase_c_power",
        "status": "PREREGISTERED_NOT_RUN",
        "independent_unit": "semantic_scene",
        "forks_per_arm": 8,
        "target_power": 0.90,
        "alpha": 0.05,
        "equivalence_margin": 0.02,
        "repetitions": REPETITIONS,
        "seed": SEED,
        "registered_intake_scenes": 8000,
        "required_eligible_scenes": plan.required_eligible_scenes,
        "eligibility_rate_lower": plan.eligibility_rate_lower,
        "required_intake_scenes": plan.required_intake_scenes,
        "feasible": plan.feasible,
        "family_quotas": dict(plan.family_quotas),
        "scenarios": [item for _curve_value, item in built],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite power artifact: {args.output}")
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
