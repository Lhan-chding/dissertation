"""Build the local CPU-side support packages for budget-matched B0--B3 LoRA."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy

from compensability_v5.audit.budget_audit import assert_budget_matched

_ARM_ORDER = ("B0", "B1", "B2", "B3")
_REQUIRED_SCENE_FIELDS = frozenset(
    {
        "scene_id",
        "semantic_scene_id",
        "prompt",
        "truth",
        "natural_observation",
        "constraint_matrix",
        "constraint_targets",
        "answer_operation",
        "transformation",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "steps",
        "optimizer",
        "lora_rank",
        "lora_targets",
        "gradient_accumulation",
        "approximate_flops",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {"parent_manifest_sha256", "child_manifest_sha256", "frozen_scenes_sha256"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SupportBuildError(ValueError):
    """The support-package freeze does not satisfy the preregistered local contract."""


def _require_scene(scene: object, *, index: int) -> Mapping[str, object]:
    if not isinstance(scene, Mapping):
        raise SupportBuildError(f"scene {index} must be a mapping")
    missing = sorted(_REQUIRED_SCENE_FIELDS - set(scene))
    if missing:
        raise SupportBuildError(f"scene {index} is missing required fields: {missing}")
    prompt = scene["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise SupportBuildError("prompt must be a non-empty string")
    return scene


def _require_training_budget(training_budget: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(training_budget, Mapping):
        raise SupportBuildError("training_budget must be a mapping")
    actual = frozenset(training_budget)
    missing = sorted(_TRAINING_FIELDS - actual)
    unknown = sorted(actual - _TRAINING_FIELDS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise SupportBuildError("training_budget must use the closed schema: " + "; ".join(details))
    return {
        "steps": training_budget["steps"],
        "optimizer": deepcopy(training_budget["optimizer"]),
        "lora_rank": training_budget["lora_rank"],
        "lora_targets": deepcopy(training_budget["lora_targets"]),
        "gradient_accumulation": training_budget["gradient_accumulation"],
        "approximate_flops": training_budget["approximate_flops"],
    }


def _row_variants(
    scene: Mapping[str, object],
) -> dict[str, tuple[tuple[str, str, str], ...]]:
    truth = ",".join(str(value) for value in scene["truth"])
    observation = ",".join(str(value) for value in scene["natural_observation"])
    prompt = str(scene["prompt"])
    matrix_rows = "; ".join(
        ",".join(str(value) for value in row) for row in scene["constraint_matrix"]
    )
    targets = ",".join(str(value) for value in scene["constraint_targets"])
    transformation = scene["transformation"]
    return {
        "B0": (
            ("format_csv", f"Clean world: {truth}. Serialize it as canonical CSV.", truth),
            (
                "format_json",
                f"Clean JSON input [{truth}]. Return the same world as canonical CSV.",
                truth,
            ),
            (
                "format_yaml",
                f"Clean YAML values are {truth}. Return them as canonical CSV.",
                truth,
            ),
            (
                "format_pipe",
                f"Clean pipe-form values are {truth.replace(',', '|')}. Return canonical CSV.",
                truth,
            ),
            (
                "format_key_value",
                f"Clean keyed values correspond to world {truth}. Return canonical CSV.",
                truth,
            ),
            (
                "format_quoted",
                f"Clean quoted input corresponds to world {truth}. Return canonical CSV.",
                truth,
            ),
        ),
        "B1": (
            (
                "forward_sum_ab",
                f"Given clean world {truth}, verify a+b and return the world.",
                truth,
            ),
            (
                "forward_sum_cd",
                f"Given clean world {truth}, verify c+d and return the world.",
                truth,
            ),
            (
                "forward_difference_ab",
                f"Given clean world {truth}, verify a-b; return the world.",
                truth,
            ),
            (
                "forward_difference_cd",
                f"Given clean world {truth}, verify c-d; return the world.",
                truth,
            ),
            (
                "forward_validate_constraints",
                f"For clean values {truth}, validate A={matrix_rows} and b={targets}.",
                truth,
            ),
            (
                "forward_serialize_facts",
                f"Validate and serialize the supplied clean values {truth}; no recovery.",
                truth,
            ),
        ),
        "B2": (
            (
                "template_observe",
                f"{prompt}\nStage 1: parse observation {observation}; recover world.",
                truth,
            ),
            (
                "template_constraints",
                f"{prompt}\nStage 2: use constraints {matrix_rows}; recover world.",
                truth,
            ),
            (
                "template_targets",
                f"{prompt}\nStage 3: use targets {targets}; recover world.",
                truth,
            ),
            (
                "template_locate_error",
                f"{prompt}\nStage 4: locate the corrupted coordinate; recover world.",
                truth,
            ),
            (
                "template_repair",
                f"{prompt}\nStage 5: repair the corrupted value; recover world.",
                truth,
            ),
            ("template_serialize", f"{prompt}\nStage 6: return the recovered world as CSV.", truth),
        ),
        "B3": (
            ("orbit_variable_permuted", f"{prompt}\nVariable-permuted recovery task.", truth),
            ("orbit_fact_order", f"{prompt}\nFact-order recovery task: {matrix_rows}.", truth),
            ("orbit_fact_id", f"{prompt}\nFact-ID-renamed recovery task: {targets}.", truth),
            (
                "orbit_equivalent_basis",
                f"{prompt}\nEquivalent-basis task: {transformation}.",
                truth,
            ),
            ("orbit_mixed_system", f"{prompt}\nMixed unary/relational task: {observation}.", truth),
            ("orbit_balanced_error", f"{prompt}\nBalanced coordinate/sign recovery task.", truth),
        ),
    }


def build_budget_matched_support(
    scenes: Iterable[Mapping[str, object]],
    *,
    token_counter: Callable[[str], int],
    training_budget: Mapping[str, object],
    source_provenance: Mapping[str, str],
    target_token_relative_tolerance: float = 0.01,
) -> dict[str, object]:
    """Build the local support freeze and fail closed on any budget drift."""

    scene_tuple = tuple(
        _require_scene(scene, index=index) for index, scene in enumerate(tuple(scenes))
    )
    if not scene_tuple:
        raise SupportBuildError("at least one source scene is required")
    if len({str(scene["scene_id"]) for scene in scene_tuple}) != len(scene_tuple):
        raise SupportBuildError("source scene_id values must be unique")
    if len({str(scene["semantic_scene_id"]) for scene in scene_tuple}) != len(scene_tuple):
        raise SupportBuildError("source semantic_scene_id values must be unique")
    if not callable(token_counter):
        raise SupportBuildError("token_counter must be callable")
    budget_template = _require_training_budget(training_budget)
    if not isinstance(source_provenance, Mapping) or set(source_provenance) != _PROVENANCE_FIELDS:
        raise SupportBuildError("source_provenance must use the closed three-hash schema")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in source_provenance.values()
    ):
        raise SupportBuildError("source_provenance values must be lowercase SHA-256 digests")

    rows_by_arm: dict[str, list[dict[str, object]]] = {arm: [] for arm in _ARM_ORDER}
    for scene in scene_tuple:
        variants = _row_variants(scene)
        for arm in _ARM_ORDER:
            if len(variants[arm]) != 6:
                raise AssertionError(f"{arm} must define exactly six rows per source")
            for variant_index, (task_name, prompt, completion) in enumerate(variants[arm], start=1):
                target_tokens = token_counter(completion)
                if (
                    not isinstance(target_tokens, int)
                    or isinstance(target_tokens, bool)
                    or target_tokens <= 0
                ):
                    raise SupportBuildError("token_counter must return a positive integer")
                rows_by_arm[arm].append(
                    {
                        "schema_version": 1,
                        "arm": arm,
                        "variant_index": variant_index,
                        "scene_id": scene["scene_id"],
                        "semantic_scene_id": scene["semantic_scene_id"],
                        "task_name": task_name,
                        "prompt": prompt,
                        "completion": completion,
                        "target_tokens": target_tokens,
                    }
                )

    budgets: dict[str, dict[str, object]] = {}
    for arm, rows in rows_by_arm.items():
        budgets[arm] = {
            **deepcopy(budget_template),
            "unique_source_scenes": len(scene_tuple),
            "rows": len(rows),
            "target_tokens": sum(int(row["target_tokens"]) for row in rows),
        }
    assert_budget_matched(budgets, target_token_relative_tolerance=target_token_relative_tolerance)
    return {
        "schema_version": 1,
        "status": "V5_BUDGET_MATCHED_SUPPORT_FROZEN",
        "source_scene_count": len(scene_tuple),
        "arms": rows_by_arm,
        "budgets": budgets,
        "target_token_relative_tolerance": target_token_relative_tolerance,
        "source_provenance": dict(source_provenance),
        "pilot_schedule": {
            "hardware": "single_RTX_4090",
            "batch_size": 1,
            "gradient_accumulation": budget_template["gradient_accumulation"],
            "epochs": 1,
            "optimizer_steps": budget_template["steps"],
        },
    }


__all__ = ["SupportBuildError", "build_budget_matched_support"]
