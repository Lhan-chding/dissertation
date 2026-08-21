"""Closed-schema input gates for the Study B pilot."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from compensability_v5.audit.budget_audit import assert_budget_matched

ARMS = ("B0", "B1", "B2", "B3")
PILOT_SEED = 2026082201
CANONICAL_SOURCE_SCENES = 96
CANONICAL_ROWS_PER_ARM = 576
CANONICAL_EVALUATION_SEMANTIC_SCENES = 32
CANONICAL_EVALUATION_ROWS = 160
INDEPENDENT_EVALUATION_SPLIT = "independent_v4_support_dev"
INDEPENDENT_RAW_ARCHIVE_SHA256 = "f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84"
CANONICAL_STEPS = 72
CANONICAL_BATCH_SIZE = 1
CANONICAL_GRADIENT_ACCUMULATION = 8
CANONICAL_LORA_RANK = 16
CANONICAL_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
TARGET_TOKEN_RELATIVE_TOLERANCE = 0.01
MODEL_SNAPSHOT_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
REGISTERED_AXES = frozenset(
    {"iid", "variable_permutation", "error_position", "fact_order", "constraint_graph"}
)
RELATIONAL_FAMILIES = frozenset({"pair_sum", "cross_series", "trend"})
_WORLD_PATTERN = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")


class StudyBError(ValueError):
    """A Study-B input, execution, or evidence invariant was violated."""


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise StudyBError(f"{label} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StudyBError(f"{label} must be a positive integer")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyBError(f"{label} must be a non-empty string")
    return value


def _rows_for_arm(value: object, arm: str) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise StudyBError(f"{arm} support rows must be a non-empty sequence")
    required = {
        "schema_version",
        "arm",
        "variant_index",
        "scene_id",
        "semantic_scene_id",
        "task_name",
        "prompt",
        "completion",
        "target_tokens",
    }
    rows: list[dict[str, object]] = []
    for index, source in enumerate(value):
        if not isinstance(source, Mapping) or set(source) != required:
            raise StudyBError(f"{arm} support row {index} has a malformed closed schema")
        if source.get("schema_version") != 1 or source.get("arm") != arm:
            raise StudyBError(f"{arm} support row {index} has incorrect identity fields")
        for field in ("scene_id", "semantic_scene_id", "task_name", "prompt", "completion"):
            _text(source[field], f"{arm} row {index}.{field}")
        variant_index = source["variant_index"]
        if (
            not isinstance(variant_index, int)
            or isinstance(variant_index, bool)
            or variant_index not in range(1, 7)
        ):
            raise StudyBError(f"{arm} row {index}.variant_index must be in 1..6")
        _positive_int(source["target_tokens"], f"{arm} row {index}.target_tokens")
        rows.append(dict(source))
    return tuple(rows)


def validate_support_package(package: object) -> dict[str, object]:
    """Validate matched budgets against the actual frozen row payloads."""

    if not isinstance(package, Mapping):
        raise StudyBError("support package must be a mapping")
    expected = {
        "schema_version",
        "status",
        "source_scene_count",
        "arms",
        "budgets",
        "target_token_relative_tolerance",
        "source_provenance",
        "pilot_schedule",
    }
    if set(package) != expected:
        raise StudyBError("support package has a malformed closed schema")
    if package.get("schema_version") != 1 or package.get("status") != (
        "V5_BUDGET_MATCHED_SUPPORT_FROZEN"
    ):
        raise StudyBError("support package is not the registered frozen package")
    provenance = package["source_provenance"]
    expected_provenance = {
        "parent_manifest_sha256",
        "child_manifest_sha256",
        "frozen_scenes_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance:
        raise StudyBError("support source_provenance has a malformed closed schema")
    if any(
        not isinstance(provenance[name], str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance[name]) is None
        for name in expected_provenance
    ):
        raise StudyBError("support source_provenance contains an invalid SHA-256")
    pilot_schedule = package["pilot_schedule"]
    if pilot_schedule != {
        "hardware": "single_RTX_4090",
        "batch_size": CANONICAL_BATCH_SIZE,
        "gradient_accumulation": CANONICAL_GRADIENT_ACCUMULATION,
        "epochs": 1,
        "optimizer_steps": CANONICAL_STEPS,
    }:
        raise StudyBError("support pilot_schedule differs from the canonical one-4090 pilot")
    arms_value, budgets_value = package["arms"], package["budgets"]
    if not isinstance(arms_value, Mapping) or set(arms_value) != set(ARMS):
        raise StudyBError("support package arms must be exactly B0, B1, B2, B3")
    if not isinstance(budgets_value, Mapping):
        raise StudyBError("support package budgets must be a mapping")
    tolerance_value = package["target_token_relative_tolerance"]
    if (
        not isinstance(tolerance_value, (int, float))
        or isinstance(tolerance_value, bool)
        or float(tolerance_value) != TARGET_TOKEN_RELATIVE_TOLERANCE
    ):
        raise StudyBError("support package must register the 0.01 target-token tolerance")
    try:
        assert_budget_matched(
            budgets_value,  # type: ignore[arg-type]
            target_token_relative_tolerance=TARGET_TOKEN_RELATIVE_TOLERANCE,
        )
    except (TypeError, ValueError) as error:
        raise StudyBError(f"support budget mismatch: {error}") from error

    arms = {arm: _rows_for_arm(arms_value[arm], arm) for arm in ARMS}
    source_scene_count = _positive_int(package["source_scene_count"], "source_scene_count")
    if source_scene_count != CANONICAL_SOURCE_SCENES:
        raise StudyBError(
            f"Study B requires exactly {CANONICAL_SOURCE_SCENES} unique source scenes"
        )
    reference_sources = {str(row["scene_id"]) for row in arms["B0"]}
    if len(reference_sources) != source_scene_count:
        raise StudyBError("source_scene_count differs from actual support rows")
    for arm, rows in arms.items():
        budget = budgets_value[arm]
        if not isinstance(budget, Mapping):
            raise StudyBError(f"{arm} budget must be a mapping")
        sources = {str(row["scene_id"]) for row in rows}
        if sources != reference_sources:
            raise StudyBError("all arms must use the same source scenes")
        if len(rows) != budget["rows"] or len(rows) != CANONICAL_ROWS_PER_ARM:
            raise StudyBError(f"{arm} actual row count differs from the canonical budget")
        if len(sources) != budget["unique_source_scenes"] or len(rows) != len(sources) * 6:
            raise StudyBError(f"{arm} source count or rows per source differs from its budget")
        variants_by_scene: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            variants_by_scene[str(row["scene_id"])].add(int(row["variant_index"]))
        if any(indices != set(range(1, 7)) for indices in variants_by_scene.values()):
            raise StudyBError(f"{arm} must contain variant_index 1..6 for every source scene")
        actual_tokens = sum(int(row["target_tokens"]) for row in rows)
        if actual_tokens != budget["target_tokens"]:
            raise StudyBError(f"{arm} actual target tokens differ from its budget")
        if budget["steps"] != CANONICAL_STEPS:
            raise StudyBError(f"{arm} must register exactly {CANONICAL_STEPS} optimizer steps")
        if budget["gradient_accumulation"] != CANONICAL_GRADIENT_ACCUMULATION:
            raise StudyBError(f"{arm} has incorrect gradient accumulation")
        if budget["lora_rank"] != CANONICAL_LORA_RANK:
            raise StudyBError(f"{arm} must register LoRA rank {CANONICAL_LORA_RANK}")
        if tuple(budget["lora_targets"]) != CANONICAL_LORA_TARGETS:
            raise StudyBError(f"{arm} LoRA targets differ from the seven language targets")
        if budget["optimizer"] != {
            "name": "adamw",
            "learning_rate": 2e-5,
            "weight_decay": 0.0,
        }:
            raise StudyBError(f"{arm} optimizer differs from the canonical Study B budget")
        if int(budget["steps"]) * CANONICAL_BATCH_SIZE * int(
            budget["gradient_accumulation"]
        ) != len(rows):
            raise StudyBError(f"{arm} steps, batch, accumulation, and one epoch do not agree")
    return {
        "arms": arms,
        "budgets": {arm: dict(budgets_value[arm]) for arm in ARMS},
        "target_token_relative_tolerance": TARGET_TOKEN_RELATIVE_TOLERANCE,
        "rows_per_source_per_arm": 6,
        "source_provenance": dict(provenance),
        "pilot_schedule": dict(pilot_schedule),
    }


def _constraint_system(
    row: Mapping[str, object],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    matrix_value, targets_value = row.get("constraint_matrix"), row.get("constraint_targets")
    if (
        not isinstance(matrix_value, Sequence)
        or isinstance(matrix_value, (str, bytes))
        or not matrix_value
        or not isinstance(targets_value, Sequence)
        or isinstance(targets_value, (str, bytes))
        or len(matrix_value) != len(targets_value)
    ):
        raise StudyBError("constraint matrix and targets must be equally sized sequences")
    matrix: list[tuple[int, ...]] = []
    for index, raw in enumerate(matrix_value):
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 4
            or any(not isinstance(value, int) or isinstance(value, bool) for value in raw)
        ):
            raise StudyBError(f"constraint row {index} must contain four integers")
        matrix.append(tuple(raw))
    if any(not isinstance(value, int) or isinstance(value, bool) for value in targets_value):
        raise StudyBError("constraint targets must contain integers")
    return tuple(matrix), tuple(targets_value)  # type: ignore[return-value]


def _axes(row: Mapping[str, object]) -> tuple[str, ...]:
    explicit = row.get("evaluation_axes")
    if explicit is not None:
        if (
            not isinstance(explicit, Sequence)
            or isinstance(explicit, (str, bytes))
            or not explicit
            or any(not isinstance(axis, str) or axis not in REGISTERED_AXES for axis in explicit)
            or len(set(explicit)) != len(explicit)
        ):
            raise StudyBError("evaluation_axes contains an unregistered or duplicate axis")
        axes = tuple(explicit)
    else:
        derived = {
            "familiar": "iid",
            "variable_permuted": "variable_permutation",
            "fact_order_permuted": "fact_order",
            "error_position_permuted": "error_position",
            "equivalent_basis": "constraint_graph",
            "sparse_mixed_ood": "constraint_graph",
        }.get(row.get("graph_axis"))
        if derived is None:
            raise StudyBError("evaluation row requires registered evaluation_axes or graph_axis")
        axes = (derived,)
    if len(axes) != 1:
        raise StudyBError("each Study B factorial row must represent exactly one evaluation axis")
    return axes


def validate_evaluation_rows(
    values: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    try:
        sources = tuple(values)
    except TypeError as error:
        raise StudyBError("evaluation rows must be iterable") from error
    if not sources:
        raise StudyBError("evaluation rows must be non-empty")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    represented_axes: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise StudyBError(f"evaluation row {index} must be a mapping")
        scene_id = _text(source.get("scene_id"), f"evaluation row {index}.scene_id")
        if scene_id in seen:
            raise StudyBError("evaluation scene_id values must be unique")
        seen.add(scene_id)
        _text(source.get("semantic_scene_id"), f"evaluation row {index}.semantic_scene_id")
        truth = _world(source.get("truth"), f"evaluation row {index}.truth")
        observed = _world(
            source.get("natural_observation"), f"evaluation row {index}.natural_observation"
        )
        if observed == truth:
            raise StudyBError("Study B evaluation requires an actual observed error")
        family = _text(source.get("family"), f"evaluation row {index}.family")
        if source.get("split") != INDEPENDENT_EVALUATION_SPLIT:
            raise StudyBError("Study B evaluation must use the independent legacy-v4 split")
        source_sha256 = source.get("source_sha256")
        if (
            not isinstance(source_sha256, Mapping)
            or source_sha256.get("raw_archive") != INDEPENDENT_RAW_ARCHIVE_SHA256
        ):
            raise StudyBError("Study B evaluation lacks the frozen legacy-v4 archive provenance")
        matrix, targets = _constraint_system(source)
        axes = _axes(source)
        represented_axes.update(axes)
        row = dict(source)
        row.update(
            {
                "scene_id": scene_id,
                "truth": list(truth),
                "natural_observation": list(observed),
                "family": family,
                "constraint_matrix": [list(item) for item in matrix],
                "constraint_targets": list(targets),
                "evaluation_axes": list(axes),
            }
        )
        rows.append(row)
    missing = REGISTERED_AXES - represented_axes
    if missing:
        raise StudyBError(f"evaluation rows are missing registered axes: {sorted(missing)}")
    by_parent: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["semantic_scene_id"])].append(str(row["evaluation_axes"][0]))
    if (
        len(by_parent) != CANONICAL_EVALUATION_SEMANTIC_SCENES
        or len(rows) != CANONICAL_EVALUATION_ROWS
    ):
        raise StudyBError(
            "Study B evaluation requires exactly 32 independent parents and 160 axis rows"
        )
    if any(len(axes) != 5 or set(axes) != REGISTERED_AXES for axes in by_parent.values()):
        raise StudyBError("every evaluation parent must contain each of the five axes exactly once")
    return tuple(rows)


def evaluation_rows_from_study_a(
    values: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Convert hash-bound Study A Base scenario definitions; never use model outputs as labels."""

    try:
        sources = tuple(values)
    except TypeError as error:
        raise StudyBError("Study A evaluation rows must be iterable") from error
    selected = [
        row
        for row in sources
        if isinstance(row, Mapping)
        and row.get("checkpoint") == "Base"
        and row.get("split") == INDEPENDENT_EVALUATION_SPLIT
    ]
    if not selected:
        raise StudyBError("evaluation input contains no Study A Base scenario rows")
    graph_to_axis = {
        "canonical": "iid",
        "variable_permuted": "variable_permutation",
        "error_location_permuted": "error_position",
        "fact_order_permuted": "fact_order",
        "equivalent_basis_graph_ood": "constraint_graph",
    }
    converted: list[dict[str, object]] = []
    axes_by_parent: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(selected):
        if row.get("checkpoint_sha256") != MODEL_SNAPSHOT_SHA256:
            raise StudyBError("Study A Base scenario row has the wrong model snapshot SHA")
        axis = graph_to_axis.get(row.get("graph_axis"))
        if axis is None:
            raise StudyBError(f"Study A row {index} has an unregistered graph axis")
        scenario_id = _text(row.get("scenario_id"), f"Study A row {index}.scenario_id")
        parent = _text(row.get("orbit_parent"), f"Study A row {index}.orbit_parent")
        source_sha256 = row.get("source_sha256")
        if (
            not isinstance(source_sha256, Mapping)
            or source_sha256.get("raw_archive") != INDEPENDENT_RAW_ARCHIVE_SHA256
        ):
            raise StudyBError("Study A row lacks the frozen legacy-v4 archive provenance")
        axes_by_parent[parent].add(axis)
        converted.append(
            {
                "scene_id": scenario_id,
                "semantic_scene_id": parent,
                "truth": list(_world(row.get("truth"), f"Study A row {index}.truth")),
                "natural_observation": list(
                    _world(row.get("observed"), f"Study A row {index}.observed")
                ),
                "family": _text(row.get("family"), f"Study A row {index}.family"),
                "split": INDEPENDENT_EVALUATION_SPLIT,
                "source_sha256": {"raw_archive": INDEPENDENT_RAW_ARCHIVE_SHA256},
                "graph_axis": str(row.get("graph_axis")),
                "evaluation_axes": [axis],
                "constraint_matrix": row.get("constraint_matrix"),
                "constraint_targets": row.get("constraint_targets"),
                "study_a_prompt_sha256": row.get("prompt_sha256"),
                "study_a_source_scene_id": row.get("source_scene_id"),
            }
        )
    if any(axes != REGISTERED_AXES for axes in axes_by_parent.values()):
        raise StudyBError("every Study A orbit parent must contain all five registered axes")
    if (
        len(axes_by_parent) != CANONICAL_EVALUATION_SEMANTIC_SCENES
        or len(converted) != CANONICAL_EVALUATION_ROWS
    ):
        raise StudyBError("Study A evaluation must contain 32 parents and 160 Base axis rows")
    return validate_evaluation_rows(converted)


def unified_world_prompt(row: Mapping[str, object]) -> str:
    """Build the one registered text-only world-output prompt for every arm."""

    observed = _world(row.get("natural_observation"), "natural_observation")
    matrix, targets = _constraint_system(row)
    equations = "\n".join(
        f"{','.join(map(str, coefficients))} = {target}"
        for coefficients, target in zip(matrix, targets, strict=True)
    )
    return (
        f"Observed values: {','.join(map(str, observed))}\n"
        f"Constraint rows (A | b):\n{equations}\n"
        "Exactly one observed coordinate may be wrong. Recover the true world. "
        "Return exactly four comma-separated integers only."
    )


def parse_world(text: object) -> tuple[int, int, int, int] | None:
    if not isinstance(text, str):
        return None
    matched = _WORLD_PATTERN.fullmatch(text)
    if matched is None:
        return None
    return tuple(int(value) for value in matched.groups())  # type: ignore[return-value]


__all__ = [
    "ARMS",
    "CANONICAL_BATCH_SIZE",
    "CANONICAL_EVALUATION_ROWS",
    "CANONICAL_EVALUATION_SEMANTIC_SCENES",
    "CANONICAL_GRADIENT_ACCUMULATION",
    "CANONICAL_LORA_RANK",
    "CANONICAL_LORA_TARGETS",
    "CANONICAL_ROWS_PER_ARM",
    "CANONICAL_SOURCE_SCENES",
    "CANONICAL_STEPS",
    "INDEPENDENT_EVALUATION_SPLIT",
    "INDEPENDENT_RAW_ARCHIVE_SHA256",
    "MODEL_SNAPSHOT_SHA256",
    "PILOT_SEED",
    "REGISTERED_AXES",
    "RELATIONAL_FAMILIES",
    "TARGET_TOKEN_RELATIVE_TOLERANCE",
    "StudyBError",
    "evaluation_rows_from_study_a",
    "parse_world",
    "unified_world_prompt",
    "validate_evaluation_rows",
    "validate_support_package",
]
