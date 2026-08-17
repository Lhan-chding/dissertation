"""Scene-level policy-support, pass@K, and informative-group metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from compensability_v4.theory.policy_support import (
    informative_group_probability,
    mean_informative_group_rate,
)


@dataclass(frozen=True, slots=True)
class ScenePolicySupport:
    scene_id: str
    rollout_count: int
    success_count: int
    success_probability: float
    pass_at_k: float
    informative_group_probability: float


@dataclass(frozen=True, slots=True)
class PolicySupportSummary:
    group_size: int
    number_of_scenes: int
    mean_success_probability: float
    mean_pass_at_k: float
    informative_group_rate: float
    by_scene: tuple[ScenePolicySupport, ...]


@dataclass(frozen=True, slots=True)
class RewardGroupDiagnostic:
    scene_id: str
    group_id: str
    rollout_count: int
    mean_reward: float
    reward_variance: float


@dataclass(frozen=True, slots=True)
class RewardVarianceSummary:
    number_of_scenes: int
    number_of_groups: int
    mean_scene_reward_variance: float
    all_zero_group_rate: float
    all_one_group_rate: float
    non_degenerate_group_rate: float
    groups: tuple[RewardGroupDiagnostic, ...]


def summarize_policy_support(
    rows: Iterable[Mapping[str, object]], *, group_size: int
) -> PolicySupportSummary:
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("support rows must be mappings")
        scene_id = row.get("scene_id")
        success = row.get("success")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if not isinstance(success, bool):
            raise TypeError("success must be boolean")
        grouped.setdefault(scene_id, []).append(success)
    if not grouped:
        raise ValueError("support rows must not be empty")
    per_scene: list[ScenePolicySupport] = []
    for scene_id, outcomes in sorted(grouped.items()):
        successes = sum(outcomes)
        probability = successes / len(outcomes)
        per_scene.append(
            ScenePolicySupport(
                scene_id=scene_id,
                rollout_count=len(outcomes),
                success_count=successes,
                success_probability=probability,
                pass_at_k=1.0 - (1.0 - probability) ** group_size,
                informative_group_probability=informative_group_probability(
                    probability, group_size
                ),
            )
        )
    probabilities = tuple(item.success_probability for item in per_scene)
    return PolicySupportSummary(
        group_size=group_size,
        number_of_scenes=len(per_scene),
        mean_success_probability=sum(probabilities) / len(probabilities),
        mean_pass_at_k=sum(item.pass_at_k for item in per_scene) / len(per_scene),
        informative_group_rate=mean_informative_group_rate(probabilities, group_size),
        by_scene=tuple(per_scene),
    )


def summarize_group_reward_variance(
    rows: Iterable[Mapping[str, object]],
) -> RewardVarianceSummary:
    """Report actual group variance without imposing an empirical success threshold."""

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("reward rows must be mappings")
        scene_id = row.get("scene_id")
        group_id = row.get("group_id")
        reward = row.get("reward")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError("reward must be numeric")
        numeric_reward = float(reward)
        if not 0.0 <= numeric_reward <= 1.0:
            raise ValueError("reward must lie in [0, 1]")
        grouped.setdefault((scene_id, group_id), []).append(numeric_reward)
    if not grouped:
        raise ValueError("reward rows must not be empty")
    diagnostics: list[RewardGroupDiagnostic] = []
    for (scene_id, group_id), rewards in sorted(grouped.items()):
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        diagnostics.append(
            RewardGroupDiagnostic(
                scene_id=scene_id,
                group_id=group_id,
                rollout_count=len(rewards),
                mean_reward=mean,
                reward_variance=variance,
            )
        )
    scene_variances: dict[str, list[float]] = {}
    for group in diagnostics:
        scene_variances.setdefault(group.scene_id, []).append(group.reward_variance)
    per_scene = tuple(sum(variances) / len(variances) for variances in scene_variances.values())
    count = len(diagnostics)
    all_zero = sum(group.mean_reward == 0.0 for group in diagnostics) / count
    all_one = sum(group.mean_reward == 1.0 for group in diagnostics) / count
    non_degenerate = sum(group.reward_variance > 0.0 for group in diagnostics) / count
    return RewardVarianceSummary(
        number_of_scenes=len(scene_variances),
        number_of_groups=count,
        mean_scene_reward_variance=sum(per_scene) / len(per_scene),
        all_zero_group_rate=all_zero,
        all_one_group_rate=all_one,
        non_degenerate_group_rate=non_degenerate,
        groups=tuple(diagnostics),
    )
