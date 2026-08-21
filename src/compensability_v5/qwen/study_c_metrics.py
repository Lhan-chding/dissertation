"""Pure raw-trace diagnostics for the v5 Study-C reward contrast."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

STUDY_C_SEED = 2026082301
PRIMARY_INITIALIZATION = "B3"
SECONDARY_INITIALIZATION = "B2"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026082302


class StudyCError(RuntimeError):
    """Raised when Study C cannot preserve its registered causal contrast."""


def read_study_c_trace(path: Path) -> tuple[dict[str, object], ...]:
    """Read an immutable JSONL trace and reject unsafe or malformed inputs."""

    if path.is_symlink() or not path.is_file():
        raise StudyCError(f"Study C reward trace is missing or unsafe: {path}")
    try:
        rows = tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudyCError(f"Study C reward trace is invalid: {error}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise StudyCError("Study C reward trace must contain JSON objects")
    return rows


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _arm_summary(rows: Sequence[Mapping[str, object]], group_size: int) -> dict[str, object]:
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        call_index, scene_id = row.get("reward_call_index"), row.get("scene_id")
        if type(call_index) is not int or not isinstance(scene_id, str):
            raise StudyCError("Study C reward trace group keys are malformed")
        grouped[(call_index, scene_id)].append(row)
    if not grouped or any(len(group) != group_size for group in grouped.values()):
        raise StudyCError("Study C recorded reward group size drifted")
    variances: list[float] = []
    informative = correction_bearing = state_bearing = 0
    for group in grouped.values():
        rewards = tuple(float(row["reward"]) for row in group)
        if any(reward not in (0.0, 1.0) for reward in rewards):
            raise StudyCError("Study C rewards must be binary")
        mean = sum(rewards) / group_size
        variances.append(mean * (1.0 - mean))
        informative += 0.0 < mean < 1.0
        exact_present = any(row.get("exact_world_recovery") is True for row in group)
        answer_failure_present = any(row.get("answer_correct") is False for row in group)
        non_exact_present = any(row.get("exact_world_recovery") is False for row in group)
        correction_bearing += exact_present and answer_failure_present
        state_bearing += exact_present and non_exact_present
    total = len(rows)
    answer_successes = sum(row.get("answer_correct") is True for row in rows)
    exact_successes = sum(row.get("exact_world_recovery") is True for row in rows)
    parse_successes = sum(row.get("parse_success") is True for row in rows)
    shortcut_successes = sum(row.get("shortcut_answer_success") is True for row in rows)
    group_count = len(grouped)
    return {
        "rollout_count": total,
        "group_count": group_count,
        "group_size": group_size,
        "mean_group_reward_variance": sum(variances) / group_count,
        "informative_group_rate": informative / group_count,
        "correction_bearing_group_rate": correction_bearing / group_count,
        "state_correction_bearing_group_rate": state_bearing / group_count,
        "world_recovery_rate": exact_successes / total,
        "answer_accuracy_from_world": answer_successes / total,
        "correction_purity": _safe_rate(exact_successes, answer_successes),
        "answer_fiber_shortcut_rate": shortcut_successes / total,
        "parse_rate": parse_successes / total,
        "subjective_success_threshold_applied": False,
    }


def _filter_rows(
    rows: Sequence[Mapping[str, object]], *, key: str, value: object
) -> tuple[Mapping[str, object], ...]:
    return tuple(row for row in rows if row.get(key) == value)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise StudyCError("cannot average an empty Study C scene stratum")
    return sum(values) / len(values)


def _interval(values: Sequence[float]) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        raise StudyCError("cannot form an interval from empty bootstrap values")
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return [lower, upper]


def _reward_by_fiber_interaction(
    by_arm_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    results: dict[str, object] = {}
    for initialization in (PRIMARY_INITIALIZATION, SECONDARY_INITIALIZATION):
        answer_name = f"{initialization}_answer"
        state_name = f"{initialization}_exact_state"
        if answer_name not in by_arm_rows or state_name not in by_arm_rows:
            continue
        scene_rates: dict[str, dict[str, tuple[float, int]]] = {}
        for arm_name in (answer_name, state_name):
            rows = by_arm_rows[arm_name]
            for scene_id in sorted({str(row["scene_id"]) for row in rows}):
                selected = _filter_rows(rows, key="scene_id", value=scene_id)
                fiber_sizes = {row.get("fiber_size") for row in selected}
                if len(fiber_sizes) != 1 or type(next(iter(fiber_sizes))) is not int:
                    raise StudyCError("Study C scene fiber_size drifted within an arm")
                exact = sum(row.get("exact_world_recovery") is True for row in selected)
                scene_rates.setdefault(scene_id, {})[arm_name] = (
                    exact / len(selected),
                    int(next(iter(fiber_sizes))),
                )
        strata: dict[str, list[float]] = {"singleton": [], "multi_state": []}
        for scene_id, rates in sorted(scene_rates.items()):
            if set(rates) != {answer_name, state_name}:
                raise StudyCError(f"Study C reward arms do not share scene {scene_id}")
            answer_rate, answer_fiber = rates[answer_name]
            state_rate, state_fiber = rates[state_name]
            if answer_fiber != state_fiber:
                raise StudyCError(f"Study C reward arms disagree on fiber_size for {scene_id}")
            stratum = "singleton" if answer_fiber == 1 else "multi_state"
            strata[stratum].append(state_rate - answer_rate)
        result: dict[str, object] = {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "by_multiplicity": {
                stratum: {
                    "scene_count": len(values),
                    "state_minus_answer_world_recovery": None if not values else _mean(values),
                }
                for stratum, values in strata.items()
            },
        }
        if all(strata.values()):
            rng = random.Random(BOOTSTRAP_SEED)
            draws: list[float] = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                singleton = _mean([rng.choice(strata["singleton"]) for _ in strata["singleton"]])
                multi = _mean([rng.choice(strata["multi_state"]) for _ in strata["multi_state"]])
                draws.append(multi - singleton)
            result.update(
                {
                    "status": "ESTIMATED",
                    "interaction_definition": (
                        "(state-answer world recovery in multi-state fibers) - "
                        "(state-answer world recovery in singleton fibers)"
                    ),
                    "estimate": _mean(strata["multi_state"]) - _mean(strata["singleton"]),
                    "scene_bootstrap_95_ci": _interval(draws),
                }
            )
        else:
            result.update(
                {
                    "status": "INSUFFICIENT_FIBER_STRATA",
                    "estimate": None,
                    "scene_bootstrap_95_ci": None,
                }
            )
        results[initialization] = result
    return results


def _registered_stop_signals(
    interaction: Mapping[str, object],
    by_arm_rows: Mapping[str, Sequence[Mapping[str, object]]],
    baseline_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    primary = interaction.get("B3")
    primary_mapping = primary if isinstance(primary, Mapping) else {}
    estimate = primary_mapping.get("estimate")
    interval = primary_mapping.get("scene_bootstrap_95_ci")
    interaction_triggered = (
        isinstance(estimate, (int, float))
        and not isinstance(estimate, bool)
        and float(estimate) > 0.0
        and isinstance(interval, list)
        and len(interval) == 2
        and float(interval[0]) > 0.0
    )
    interaction_rule = {
        "triggered": interaction_triggered,
        "rule": "B3 reward-by-fiber interaction > 0 and scene-bootstrap CI excludes 0",
        "evidence": {"estimate": estimate, "scene_bootstrap_95_ci": interval},
    }

    final = by_arm_rows.get("B3_answer")
    baseline = baseline_rows.get("B3_answer")
    trajectory_rule: dict[str, object]
    if final is None or baseline is None:
        trajectory_rule = {
            "triggered": False,
            "rule": "large-fiber answer accuracy rises while world recovery falls vs B3 init",
            "status": "INSUFFICIENT_PRE_RL_BASELINE",
            "evidence": None,
        }
    else:

        def large(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
            return tuple(
                row
                for row in rows
                if isinstance(row.get("fiber_bin"), str)
                and "large" in str(row["fiber_bin"]).lower()
            )

        final_large, baseline_large = large(final), large(baseline)
        final_keys = {(row.get("scene_id"), row.get("rollout_seed")) for row in final_large}
        baseline_keys = {(row.get("scene_id"), row.get("rollout_seed")) for row in baseline_large}
        if not final_keys and not baseline_keys:
            trajectory_rule = {
                "triggered": False,
                "rule": ("large-fiber answer accuracy rises while world recovery falls vs B3 init"),
                "status": "INSUFFICIENT_LARGE_FIBER_ROWS",
                "evidence": {
                    "final_row_count": len(final_large),
                    "baseline_row_count": len(baseline_large),
                },
            }
            return {
                "reward_by_fiber_interaction": interaction_rule,
                "answer_up_world_down_large_fibers": trajectory_rule,
                "any_registered_signal_triggered": bool(interaction_rule["triggered"]),
                "subjective_threshold_used": False,
            }
        if final_keys != baseline_keys:
            raise StudyCError("B3 large-fiber pre/post evaluation rows are not seed-paired")

        def scene_rates(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
            result: dict[str, float] = {}
            for scene_id in sorted({str(row["scene_id"]) for row in rows}):
                selected = _filter_rows(rows, key="scene_id", value=scene_id)
                result[scene_id] = sum(row.get(field) is True for row in selected) / len(selected)
            return result

        final_answer = scene_rates(final_large, "answer_correct")
        baseline_answer = scene_rates(baseline_large, "answer_correct")
        final_world = scene_rates(final_large, "exact_world_recovery")
        baseline_world = scene_rates(baseline_large, "exact_world_recovery")
        scene_ids = sorted(final_answer)
        answer_delta = _mean([final_answer[item] - baseline_answer[item] for item in scene_ids])
        world_delta = _mean([final_world[item] - baseline_world[item] for item in scene_ids])
        trajectory_rule = {
            "triggered": answer_delta > 0.0 and world_delta < 0.0,
            "rule": "large-fiber answer accuracy rises while world recovery falls vs B3 init",
            "status": "PAIRED_PRE_POST_EVALUATED",
            "evidence": {
                "paired_scene_count": len(scene_ids),
                "rollouts_per_scene": len(final_large) // len(scene_ids),
                "answer_accuracy_delta": answer_delta,
                "world_recovery_delta": world_delta,
            },
        }
    return {
        "reward_by_fiber_interaction": interaction_rule,
        "answer_up_world_down_large_fibers": trajectory_rule,
        "any_registered_signal_triggered": bool(
            interaction_rule["triggered"] or trajectory_rule["triggered"]
        ),
        "subjective_threshold_used": False,
    }


def build_study_c_summary(
    trace_paths: Mapping[str, Path],
    *,
    group_size: int,
    baseline_trace_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Build group, scene, fiber, answer, world, and interaction diagnostics."""

    if type(group_size) is not int or group_size < 2:
        raise StudyCError("Study C summary group_size must be at least two")
    by_arm_rows = {arm: read_study_c_trace(path) for arm, path in trace_paths.items()}
    if not by_arm_rows:
        raise StudyCError("Study C summary has no arm traces")
    by_arm = {arm: _arm_summary(rows, group_size) for arm, rows in sorted(by_arm_rows.items())}
    baseline_rows = {
        arm: read_study_c_trace(path)
        for arm, path in ({} if baseline_trace_paths is None else baseline_trace_paths).items()
    }
    per_scene: list[dict[str, object]] = []
    for arm, rows in sorted(by_arm_rows.items()):
        for scene_id in sorted({str(row["scene_id"]) for row in rows}):
            selected = _filter_rows(rows, key="scene_id", value=scene_id)
            exemplar = selected[0]
            per_scene.append(
                {
                    "arm": arm,
                    "scene_id": scene_id,
                    "family": exemplar.get("family"),
                    "fiber_size": exemplar.get("fiber_size"),
                    "fiber_bin": exemplar.get("fiber_bin"),
                    "support_bin": exemplar.get("support_bin"),
                    **_arm_summary(selected, group_size),
                }
            )
    fiber_bins = sorted({str(row["fiber_bin"]) for rows in by_arm_rows.values() for row in rows})
    by_fiber_bin: dict[str, dict[str, object]] = {}
    for fiber_bin in fiber_bins:
        arm_metrics = {
            arm: _arm_summary(selected, group_size)
            for arm, rows in sorted(by_arm_rows.items())
            if (selected := _filter_rows(rows, key="fiber_bin", value=fiber_bin))
        }
        entry: dict[str, object] = {"arms": arm_metrics}
        for initialization in (PRIMARY_INITIALIZATION, SECONDARY_INITIALIZATION):
            answer_name = f"{initialization}_answer"
            state_name = f"{initialization}_exact_state"
            if answer_name in arm_metrics and state_name in arm_metrics:
                contrast = float(arm_metrics[state_name]["world_recovery_rate"]) - float(
                    arm_metrics[answer_name]["world_recovery_rate"]
                )
                entry[f"{initialization}_state_minus_answer_world_recovery"] = contrast
                if initialization == PRIMARY_INITIALIZATION:
                    entry["state_minus_answer_world_recovery"] = contrast
        by_fiber_bin[fiber_bin] = entry
    interaction = _reward_by_fiber_interaction(by_arm_rows)
    return {
        "schema_version": 1,
        "status": "STUDY_C_DIAGNOSTICS_COMPLETE",
        "seed": STUDY_C_SEED,
        "group_size": group_size,
        "by_arm": by_arm,
        "per_scene": per_scene,
        "by_fiber_bin": by_fiber_bin,
        "reward_by_fiber_interaction": interaction,
        "reward_by_fiber_interaction_reported": True,
        "registered_stop_signals": _registered_stop_signals(
            interaction, by_arm_rows, baseline_rows
        ),
        "subjective_success_threshold_applied": False,
    }


__all__ = ["StudyCError", "build_study_c_summary", "read_study_c_trace"]
