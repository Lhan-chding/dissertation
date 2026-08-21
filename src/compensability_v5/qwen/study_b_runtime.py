"""Single-GPU Study-B training and text-world evaluation runtime.

The orchestration layer is dependency-light and callback-testable.  Torch,
Transformers, PEFT, and Datasets are imported only by :class:`QwenStudyBBackend`
after the CLI has applied its explicit acknowledgement and offline gates.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
WORLD_PATTERN = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")
REGISTERED_AXES = frozenset(
    {"iid", "variable_permutation", "error_position", "fact_order", "constraint_graph"}
)
RELATIONAL_FAMILIES = frozenset({"pair_sum", "cross_series", "trend"})


class StudyBError(ValueError):
    """A Study-B input, execution, or evidence invariant was violated."""


class StudyBBackend(Protocol):
    """Minimal backend boundary used by both the real Qwen runner and CPU tests."""

    def load_base(self, *, arm: str, expected_model_sha256: str) -> Mapping[str, object]: ...

    def train(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        budget: dict[str, object],
        seed: int,
        output: Path,
    ) -> Mapping[str, object]: ...

    def evaluate(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        prompts: tuple[str, ...],
        seed: int,
        output: Path,
    ) -> Iterable[Mapping[str, object]]: ...

    def release(self, session: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class _TrainingRow:
    example_id: str
    prompt: str
    completion: str


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StudyBError(f"unsafe or missing regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash a complete adapter tree including paths, sizes, and file hashes."""

    if path.is_symlink() or not path.is_dir():
        raise StudyBError(f"adapter tree is missing or unsafe: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise StudyBError("adapter tree must contain at least one regular file")
    digest = hashlib.sha256()
    for candidate in files:
        if candidate.is_symlink():
            raise StudyBError(f"adapter tree contains a symlink: {candidate}")
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    rows: list[dict[str, object]] = []
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
    tolerance = TARGET_TOKEN_RELATIVE_TOLERANCE
    try:
        assert_budget_matched(
            budgets_value,  # type: ignore[arg-type]
            target_token_relative_tolerance=tolerance,
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
        if len(rows) != budget["rows"]:
            raise StudyBError(f"{arm} actual row count differs from its budget")
        if len(rows) != CANONICAL_ROWS_PER_ARM:
            raise StudyBError(f"{arm} must contain exactly {CANONICAL_ROWS_PER_ARM} rows")
        if len(sources) != budget["unique_source_scenes"]:
            raise StudyBError(f"{arm} actual source count differs from its budget")
        if len(rows) != len(sources) * 6:
            raise StudyBError(f"{arm} must contain exactly six rows per source scene")
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
            raise StudyBError(
                f"{arm} must register gradient accumulation {CANONICAL_GRADIENT_ACCUMULATION}"
            )
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
        "target_token_relative_tolerance": tolerance,
        "rows_per_source_per_arm": 6,
        "source_provenance": dict(provenance),
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
        graph_axis = row.get("graph_axis")
        derived = {
            "familiar": "iid",
            "variable_permuted": "variable_permutation",
            "fact_order_permuted": "fact_order",
            "error_position_permuted": "error_position",
            "equivalent_basis": "constraint_graph",
            "sparse_mixed_ood": "constraint_graph",
        }.get(graph_axis)
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
    if len(by_parent) != CANONICAL_EVALUATION_SEMANTIC_SCENES or len(rows) != (
        CANONICAL_EVALUATION_ROWS
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
    """Convert Study A's hash-bound Base scenario rows into Study B evaluation rows.

    Study A publishes both Base and T rows for the same 32 natural-error parents.
    Study B consumes the frozen scenario definitions only once, selecting Base
    solely as the provenance anchor; no Study A model output is used as a label.
    """

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
        graph_axis = row.get("graph_axis")
        axis = graph_to_axis.get(graph_axis)
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
                "graph_axis": str(graph_axis),
                "evaluation_axes": [axis],
                "constraint_matrix": row.get("constraint_matrix"),
                "constraint_targets": row.get("constraint_targets"),
                "study_a_prompt_sha256": row.get("prompt_sha256"),
                "study_a_source_scene_id": row.get("source_scene_id"),
            }
        )
    if any(axes != REGISTERED_AXES for axes in axes_by_parent.values()):
        raise StudyBError("every Study A orbit parent must contain all five registered axes")
    if len(axes_by_parent) != CANONICAL_EVALUATION_SEMANTIC_SCENES or len(converted) != (
        CANONICAL_EVALUATION_ROWS
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
    matched = WORLD_PATTERN.fullmatch(text)
    if matched is None:
        return None
    return tuple(int(value) for value in matched.groups())  # type: ignore[return-value]


def _metric_block(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(rows)
    relational = [row for row in rows if row["family"] in RELATIONAL_FAMILIES]
    return {
        "count": count,
        "parsed_rate": sum(row["parsed_world"] is not None for row in rows) / count,
        "exact_world_rate": sum(bool(row["exact_world"]) for row in rows) / count,
        "genuine_recovery_rate": sum(bool(row["genuine_recovery"]) for row in rows) / count,
        "observation_copy_rate": sum(bool(row["observation_copy"]) for row in rows) / count,
        "relational_count": len(relational),
        "relational_genuine_recovery_rate": (
            sum(bool(row["genuine_recovery"]) for row in relational) / len(relational)
            if relational
            else None
        ),
        "mean_candidate_margin": (
            sum(
                float(row["candidate_margin"])
                for row in rows
                if row.get("candidate_margin") is not None
            )
            / sum(row.get("candidate_margin") is not None for row in rows)
            if any(row.get("candidate_margin") is not None for row in rows)
            else None
        ),
    }


def summarize_evaluations(
    evaluation_rows: Iterable[Mapping[str, object]],
    outputs: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    rows = validate_evaluation_rows(evaluation_rows)
    output_rows = tuple(outputs)
    if len(output_rows) != len(rows):
        raise StudyBError("backend must return exactly one evaluation output per scene")
    by_scene: dict[str, Mapping[str, object]] = {}
    for output in output_rows:
        if not isinstance(output, Mapping):
            raise StudyBError("evaluation output must be a mapping")
        scene_id = _text(output.get("scene_id"), "evaluation output scene_id")
        if scene_id in by_scene:
            raise StudyBError("backend returned duplicate evaluation scene_id")
        if set(output) - {"scene_id", "completion", "candidate_margin"}:
            raise StudyBError("evaluation output contains unregistered fields")
        completion = _text(output.get("completion"), "evaluation completion")
        margin = output.get("candidate_margin")
        if margin is not None and (
            not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
        ):
            raise StudyBError("candidate_margin must be finite when supplied")
        by_scene[scene_id] = {"completion": completion, "candidate_margin": margin}
    if set(by_scene) != {str(row["scene_id"]) for row in rows}:
        raise StudyBError("backend evaluation scene set differs from the frozen evaluation set")

    enriched: list[dict[str, object]] = []
    by_axis: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        output = by_scene[str(row["scene_id"])]
        parsed = parse_world(output["completion"])
        truth = tuple(row["truth"])
        observed = tuple(row["natural_observation"])
        exact = parsed == truth
        enriched_row = {
            "scene_id": row["scene_id"],
            "semantic_scene_id": row["semantic_scene_id"],
            "family": row["family"],
            "graph_axis": row.get("graph_axis"),
            "evaluation_axes": row["evaluation_axes"],
            "truth": list(truth),
            "natural_observation": list(observed),
            "prompt_sha256": canonical_sha256(unified_world_prompt(row)),
            "completion": output["completion"],
            "parsed_world": list(parsed) if parsed is not None else None,
            "exact_world": exact,
            "genuine_recovery": exact and observed != truth,
            "observation_copy": parsed == observed,
            "candidate_margin": output.get("candidate_margin"),
        }
        enriched.append(enriched_row)
        by_family[str(row["family"])].append(enriched_row)
        for axis in row["evaluation_axes"]:
            by_axis[str(axis)].append(enriched_row)
            if axis != "iid":
                by_axis["structural_ood"].append(enriched_row)
    summary = {
        "schema_version": 1,
        "statistical_unit": "semantic_scene",
        "prompt_protocol": "unified_text_world_v1",
        "overall": _metric_block(enriched),
        "by_axis": {name: _metric_block(group) for name, group in sorted(by_axis.items())},
        "by_family": {name: _metric_block(group) for name, group in sorted(by_family.items())},
    }
    return summary, tuple(enriched)


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise StudyBError(f"missing immutable Study-B artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StudyBError(f"Study-B JSON artifact is not a mapping: {path}")
    return value


def _completed_arm_result(path: Path, *, arm: str, run_signature: str) -> dict[str, object]:
    result = _read_json(path / "result.json")
    if result.get("arm") != arm or result.get("run_signature") != run_signature:
        raise StudyBError(f"completed {arm} artifact does not match this run")
    expected = result.get("adapter_tree_sha256")
    if expected != tree_sha256(path / "final_adapter"):
        raise StudyBError(f"completed {arm} adapter tree hash changed")
    evaluation_path = path / "evaluation_rows.jsonl"
    if result.get("evaluation_rows_sha256") != sha256_file(evaluation_path):
        raise StudyBError(f"completed {arm} evaluation row log changed")
    training_log = path / "training_log.json"
    if training_log.is_symlink() or not training_log.is_file():
        raise StudyBError(f"completed {arm} training log is missing or unsafe")
    _validate_freeze_evidence(result.get("trainable_manifest"), result.get("frozen_hashes"))
    return result


def _axis_rate(result: Mapping[str, object], axis: str) -> float:
    evaluation = result.get("evaluation_metrics")
    if not isinstance(evaluation, Mapping):
        raise StudyBError("arm result lacks evaluation_metrics")
    by_axis = evaluation.get("by_axis")
    if not isinstance(by_axis, Mapping) or not isinstance(by_axis.get(axis), Mapping):
        raise StudyBError(f"arm result lacks registered {axis} evaluation")
    rate = by_axis[axis].get("exact_world_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        raise StudyBError(f"arm result has invalid {axis} exact-world rate")
    return float(rate)


def _primary_contrasts(
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    b2, b3 = results["B2"], results["B3"]
    return {
        "B3_minus_B2": {
            "iid_exact_world_rate": _axis_rate(b3, "iid") - _axis_rate(b2, "iid"),
            "variable_permutation_exact_world_rate": _axis_rate(b3, "variable_permutation")
            - _axis_rate(b2, "variable_permutation"),
            "error_position_exact_world_rate": _axis_rate(b3, "error_position")
            - _axis_rate(b2, "error_position"),
            "fact_order_exact_world_rate": _axis_rate(b3, "fact_order")
            - _axis_rate(b2, "fact_order"),
            "constraint_graph_exact_world_rate": _axis_rate(b3, "constraint_graph")
            - _axis_rate(b2, "constraint_graph"),
            "structural_ood_exact_world_rate": _axis_rate(b3, "structural_ood")
            - _axis_rate(b2, "structural_ood"),
        }
    }


def _validate_freeze_evidence(trainable: object, frozen: object) -> None:
    if not isinstance(trainable, Mapping) or any(
        trainable.get(field) is not True
        for field in ("vision_frozen", "merger_frozen", "base_language_frozen")
    ):
        raise StudyBError("Study B requires vision and merger frozen with language Base")
    target_modules = trainable.get("target_modules")
    parameter_names = trainable.get("trainable_parameter_names")
    if (
        not isinstance(target_modules, Sequence)
        or isinstance(target_modules, (str, bytes))
        or not target_modules
        or not isinstance(parameter_names, Sequence)
        or isinstance(parameter_names, (str, bytes))
        or not parameter_names
        or any(
            not isinstance(name, str)
            or "lora_" not in name
            or ".language_model." not in name
            or ".visual." in name
            or ".merger" in name
            for name in parameter_names
        )
    ):
        raise StudyBError("Study B trainable manifest is not language-LoRA-only")
    if not isinstance(frozen, Mapping):
        raise StudyBError("backend must return frozen component hashes")
    hashes = frozen.get("sha256_by_component")
    required = ("vision", "merger", "language_base")
    if not isinstance(hashes, Mapping) or not set(required).issubset(hashes):
        raise StudyBError("frozen evidence lacks vision, merger, or language Base hashes")
    if any(
        not isinstance(hashes[name], str) or re.fullmatch(r"[0-9a-f]{64}", hashes[name]) is None
        for name in required
    ):
        raise StudyBError("frozen component evidence contains an invalid SHA-256")


def run_study_b(
    *,
    support_package: Mapping[str, object],
    evaluation_rows: Iterable[Mapping[str, object]],
    output: Path,
    backend: StudyBBackend,
    expected_model_sha256: str,
    seed: int = PILOT_SEED,
    resume: bool = False,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute the immutable one-seed B0--B3 training/evaluation loop."""

    if seed != PILOT_SEED:
        raise StudyBError(f"Study B pilot seed must be exactly {PILOT_SEED}")
    if expected_model_sha256 != MODEL_SNAPSHOT_SHA256:
        raise StudyBError("Study B model SHA differs from the frozen Qwen v4 snapshot")
    validated_support = validate_support_package(support_package)
    eval_rows = validate_evaluation_rows(evaluation_rows)
    support_semantic_ids = {
        str(row["semantic_scene_id"]) for row in validated_support["arms"]["B0"]
    }
    evaluation_semantic_ids = {str(row["semantic_scene_id"]) for row in eval_rows}
    overlap = support_semantic_ids & evaluation_semantic_ids
    if overlap:
        raise StudyBError(
            f"Study B training/evaluation semantic scenes overlap: {sorted(overlap)[:3]}"
        )
    prompts = tuple(unified_world_prompt(row) for row in eval_rows)
    output = Path(output)
    if output.is_symlink():
        raise StudyBError("Study B output must not be a symlink")

    run_signature_payload = {
        "schema_version": 1,
        "study": "B_budget_matched_structural_support_lora",
        "seed": seed,
        "model_snapshot_sha256": expected_model_sha256,
        "support_package_sha256": canonical_sha256(support_package),
        "evaluation_rows_sha256": canonical_sha256(eval_rows),
        "arms": list(ARMS),
        "prompt_protocol": "unified_text_world_v1",
        "training_device": "single_4090",
        "per_device_train_batch_size": CANONICAL_BATCH_SIZE,
        "fixed_sequence_padding": True,
        "provenance": dict(provenance or {}),
    }
    run_signature = canonical_sha256(run_signature_payload)
    manifest = {**run_signature_payload, "run_signature": run_signature}
    manifest_path = output / "run_manifest.json"
    completed_path = output / "completed.json"
    if output.exists():
        if not resume:
            raise FileExistsError("Study B output already exists; use --resume only for this run")
        if not output.is_dir() or _read_json(manifest_path) != manifest:
            raise StudyBError("resume manifest differs from the requested Study B run")
        if completed_path.is_file():
            completed = _read_json(completed_path)
            completed_results: dict[str, dict[str, object]] = {}
            for arm in ARMS:
                completed_results[arm] = _completed_arm_result(
                    output / "arms" / arm,
                    arm=arm,
                    run_signature=run_signature,
                )
            expected_tokens = {
                arm: result["base_load_token"] for arm, result in completed_results.items()
            }
            if (
                completed.get("status") != "STUDY_B_SINGLE_SEED_COMPLETE"
                or completed.get("seed") != seed
                or completed.get("model_snapshot_sha256") != expected_model_sha256
                or completed.get("run_signature") != run_signature
                or completed.get("run_manifest_sha256") != canonical_sha256(manifest)
                or completed.get("arm_results") != completed_results
                or completed.get("base_load_tokens") != expected_tokens
                or completed.get("primary_contrasts") != _primary_contrasts(completed_results)
            ):
                raise StudyBError("completed Study B payload or referenced arm evidence drifted")
            return completed
    else:
        if resume:
            raise StudyBError("cannot resume a Study B output that does not exist")
        output.mkdir(parents=True)
        _write_json_new(manifest_path, manifest)

    arms_root = output / "arms"
    attempts_root = output / "attempts"
    arms_root.mkdir(exist_ok=True)
    attempts_root.mkdir(exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    load_tokens: dict[str, str] = {}
    observed_tokens: set[str] = set()
    observed_flops_reference: float | None = None
    for arm in ARMS:
        final_arm = arms_root / arm
        if final_arm.exists():
            result = _completed_arm_result(final_arm, arm=arm, run_signature=run_signature)
            results[arm] = result
            token = _text(result.get("base_load_token"), f"{arm}.base_load_token")
            load_tokens[arm] = token
            observed_tokens.add(token)
            metrics = result.get("training_metrics")
            if not isinstance(metrics, Mapping):
                raise StudyBError(f"completed {arm} result lacks training metrics")
            observed_flops = metrics.get("observed_total_flos")
            if (
                not isinstance(observed_flops, (int, float))
                or isinstance(observed_flops, bool)
                or not math.isfinite(float(observed_flops))
                or float(observed_flops) <= 0
            ):
                raise StudyBError(f"completed {arm} lacks positive observed FLOPs")
            if observed_flops_reference is None:
                observed_flops_reference = float(observed_flops)
            elif float(observed_flops) != observed_flops_reference:
                raise StudyBError("observed training FLOPs differ across Study B arms")
            continue
        attempt = Path(tempfile.mkdtemp(prefix=f"{arm}-", dir=attempts_root))
        session: Mapping[str, object] | None = None
        try:
            session = backend.load_base(arm=arm, expected_model_sha256=expected_model_sha256)
            if not isinstance(session, Mapping):
                raise StudyBError("backend load_base must return a mapping")
            if session.get("model_sha256") != expected_model_sha256:
                raise StudyBError(f"{arm} backend loaded a different Base snapshot")
            load_token = _text(session.get("load_token"), f"{arm}.load_token")
            if load_token in observed_tokens:
                raise StudyBError("each arm must reload a distinct fresh Base session")
            observed_tokens.add(load_token)
            load_tokens[arm] = load_token
            rows = validated_support["arms"][arm]
            budget = validated_support["budgets"][arm]
            training = backend.train(
                session=session,
                arm=arm,
                rows=rows,
                budget=budget,
                seed=seed,
                output=attempt,
            )
            if not isinstance(training, Mapping):
                raise StudyBError("backend train must return a mapping")
            if training.get("observed_target_tokens") != budget["target_tokens"]:
                raise StudyBError(
                    f"{arm} actual tokenizer completion count differs from frozen budget"
                )
            training_metrics = training.get("training_metrics")
            if not isinstance(training_metrics, Mapping):
                raise StudyBError(f"{arm} backend must return training metrics")
            if training_metrics.get("train_steps") != budget["steps"]:
                raise StudyBError(f"{arm} did not execute the exact optimizer-step budget")
            observed_flops = training_metrics.get("observed_total_flos")
            if (
                not isinstance(observed_flops, (int, float))
                or isinstance(observed_flops, bool)
                or not math.isfinite(float(observed_flops))
                or float(observed_flops) <= 0
            ):
                raise StudyBError(f"{arm} backend must report positive observed FLOPs")
            if observed_flops_reference is None:
                observed_flops_reference = float(observed_flops)
            elif float(observed_flops) != observed_flops_reference:
                raise StudyBError("observed training FLOPs differ across Study B arms")
            manifest_value = training.get("trainable_manifest")
            frozen_hashes = training.get("frozen_hashes")
            _validate_freeze_evidence(manifest_value, frozen_hashes)
            assert isinstance(manifest_value, Mapping)
            assert isinstance(frozen_hashes, Mapping)
            adapter = Path(_text(training.get("adapter_path"), "adapter_path"))
            if adapter.resolve() != (attempt / "final_adapter").resolve():
                raise StudyBError(
                    "backend adapter must be written to the assigned attempt directory"
                )
            adapter_hash = tree_sha256(adapter)
            if not (attempt / "training_log.json").is_file():
                raise StudyBError(f"{arm} backend did not write the required training log")
            output_rows = tuple(
                backend.evaluate(
                    session=session,
                    arm=arm,
                    rows=eval_rows,
                    prompts=prompts,
                    seed=seed,
                    output=attempt,
                )
            )
            evaluation_metrics, enriched = summarize_evaluations(eval_rows, output_rows)
            evaluation_path = attempt / "evaluation_rows.jsonl"
            if evaluation_path.exists() or evaluation_path.is_symlink():
                raise StudyBError("backend must not pre-create the registered evaluation row log")
            _write_jsonl_new(evaluation_path, enriched)
            result = {
                "schema_version": 1,
                "status": "STUDY_B_ARM_COMPLETE",
                "arm": arm,
                "seed": seed,
                "run_signature": run_signature,
                "base_model_sha256": expected_model_sha256,
                "base_load_token": load_token,
                "budget": budget,
                "adapter_tree_sha256": adapter_hash,
                "training_metrics": dict(training_metrics),
                "observed_target_tokens": training["observed_target_tokens"],
                "trainable_manifest": dict(manifest_value),
                "frozen_hashes": dict(frozen_hashes),
                "evaluation_metrics": evaluation_metrics,
                "evaluation_rows_sha256": sha256_file(evaluation_path),
            }
            _write_json_new(attempt / "result.json", result)
            attempt.rename(final_arm)
            results[arm] = result
        finally:
            if session is not None:
                backend.release(session)

    contrasts = _primary_contrasts(results)
    completed = {
        "schema_version": 1,
        "status": "STUDY_B_SINGLE_SEED_COMPLETE",
        "seed": seed,
        "model_snapshot_sha256": expected_model_sha256,
        "run_signature": run_signature,
        "run_manifest_sha256": canonical_sha256(manifest),
        "base_load_tokens": load_tokens,
        "arm_results": results,
        "primary_contrasts": contrasts,
        "evidence_class": "single_seed_pilot",
    }
    _write_json_new(completed_path, completed)
    return completed


class QwenStudyBBackend:  # pragma: no cover - requires the pinned CUDA/Qwen server
    """Real offline Qwen2.5-VL backend reusing the verified v4 LoRA helpers."""

    def __init__(self, *, model_path: Path, max_sequence_length: int = 512) -> None:
        self.model_path = Path(model_path)
        self.max_sequence_length = _positive_int(max_sequence_length, "max_sequence_length")

    def load_base(self, *, arm: str, expected_model_sha256: str) -> Mapping[str, object]:
        _require_single_4090()
        from compensability_v4.qwen.model_loader import load_pinned_qwen, require_server_model

        verified = require_server_model(self.model_path, expected_model_sha256)
        model, processor = load_pinned_qwen(model_path=verified, device_map="cuda:0")
        return {
            "model": model,
            "processor": processor,
            "model_sha256": expected_model_sha256,
            "load_token": f"{arm}-{uuid.uuid4().hex}",
        }

    def train(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        budget: dict[str, object],
        seed: int,
        output: Path,
    ) -> Mapping[str, object]:
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments, set_seed

        from compensability_v4.training.phase4 import (
            Phase4TrainingConfig,
            _chat_training_features,
            _import_torch,
            attach_language_lora,
            discover_language_lora_targets,
            freeze_base_parameters,
            trainable_parameter_manifest,
        )

        optimizer = budget["optimizer"]
        if not isinstance(optimizer, Mapping) or optimizer.get("name") != "adamw":
            raise StudyBError("real Study B backend supports only the registered adamw optimizer")
        model, processor = session["model"], session["processor"]
        config = Phase4TrainingConfig(
            precision="bf16",
            lora_rank=int(budget["lora_rank"]),
            lora_alpha=2 * int(budget["lora_rank"]),
            lora_dropout=0.0,
            gradient_checkpointing=True,
            vision_frozen=True,
            merger_frozen=True,
            learning_rate=float(optimizer["learning_rate"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(budget["gradient_accumulation"]),
            num_train_epochs=1,
            max_sequence_length=self.max_sequence_length,
            seed=seed,
            selection_split="support_dev",
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        training_rows = tuple(
            _TrainingRow(
                example_id=f"{arm}:{index}:{row['scene_id']}",
                prompt=str(row["prompt"]),
                completion=str(row["completion"]),
            )
            for index, row in enumerate(rows)
        )
        features = _chat_training_features(
            tokenizer=tokenizer,
            rows=training_rows,
            max_sequence_length=self.max_sequence_length,
        )
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise StudyBError("Qwen tokenizer has no exact encode method")
        observed_token_counts = tuple(
            len(encode(row.completion, add_special_tokens=False)) for row in training_rows
        )
        frozen_token_counts = tuple(int(row["target_tokens"]) for row in rows)
        if observed_token_counts != frozen_token_counts:
            raise StudyBError(
                f"{arm} completion token counts differ from the frozen real-tokenizer audit"
            )
        observed_target_tokens = sum(observed_token_counts)
        if observed_target_tokens != int(budget["target_tokens"]):
            raise StudyBError(f"{arm} actual completion token total differs from its budget")
        discovered = discover_language_lora_targets(model)
        requested = tuple(budget["lora_targets"])
        targets = tuple(name for name in discovered if name.rsplit(".", 1)[-1] in requested)
        if {name.rsplit(".", 1)[-1] for name in targets} != set(requested):
            raise StudyBError("registered LoRA targets differ from actual Qwen language modules")
        frozen = freeze_base_parameters(model)
        adapter_model = attach_language_lora(model, config=config, targets=targets)
        if isinstance(session, dict):
            session["model"] = adapter_model
        trainable = trainable_parameter_manifest(adapter_model, targets)
        torch = _import_torch()
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_id, int):
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(pad_id, int):
            raise StudyBError("Qwen tokenizer has no padding token")

        def fixed_collator(batch: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
            result: dict[str, list[list[int]]] = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
            for item in batch:
                padding = self.max_sequence_length - len(item["input_ids"])
                if padding < 0:
                    raise StudyBError("training example exceeds fixed Study B sequence length")
                result["input_ids"].append(list(item["input_ids"]) + [pad_id] * padding)
                result["attention_mask"].append(list(item["attention_mask"]) + [0] * padding)
                result["labels"].append(list(item["labels"]) + [-100] * padding)
            return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}

        set_seed(seed)
        arguments = TrainingArguments(
            output_dir=str(output / "trainer_state"),
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(budget["gradient_accumulation"]),
            max_steps=int(budget["steps"]),
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            seed=seed,
            data_seed=seed,
            optim="adamw_torch",
        )
        trainer = Trainer(
            model=adapter_model,
            args=arguments,
            train_dataset=Dataset.from_list(features),
            data_collator=fixed_collator,
        )
        train_result = trainer.train()
        observed_steps = int(trainer.state.global_step)
        if observed_steps != int(budget["steps"]):
            raise StudyBError("Trainer did not execute the exact registered optimizer steps")
        adapter = output / "final_adapter"
        adapter_model.save_pretrained(str(adapter))
        log_history = list(trainer.state.log_history)
        _write_json_new(output / "training_log.json", log_history)
        return {
            "adapter_path": str(adapter),
            "observed_target_tokens": observed_target_tokens,
            "training_metrics": {
                "train_steps": observed_steps,
                "train_loss": float(train_result.metrics.get("train_loss", math.nan)),
                "observed_total_flos": float(getattr(trainer.state, "total_flos", 0.0)),
                "registered_approximate_flops": float(budget["approximate_flops"]),
                "fixed_padded_sequence_length": self.max_sequence_length,
                "per_device_train_batch_size": 1,
            },
            "trainable_manifest": trainable,
            "frozen_hashes": frozen,
        }

    def evaluate(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        prompts: tuple[str, ...],
        seed: int,
        output: Path,
    ) -> Iterable[Mapping[str, object]]:
        import torch

        model, processor = session["model"], session["processor"]
        tokenizer = getattr(processor, "tokenizer", processor)
        model.eval()
        output_rows: list[dict[str, object]] = []
        for row, prompt in zip(rows, prompts, strict=True):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            device = getattr(model, "device", "cuda:0")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            completion = tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ).strip()
            output_rows.append({"scene_id": row["scene_id"], "completion": completion})
        return tuple(output_rows)

    def release(self, session: Mapping[str, object]) -> None:
        if isinstance(session, dict):
            session.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def require_offline_environment(environment: Mapping[str, str] | None = None) -> None:
    current = os.environ if environment is None else environment
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [name for name in required if current.get(name) != "1"]
    if missing:
        raise StudyBError("offline environment is incomplete: " + ", ".join(missing))


def _require_single_4090() -> None:  # pragma: no cover - requires CUDA
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise StudyBError("Study B requires exactly one visible CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if "4090" not in device_name:
        raise StudyBError(f"Study B pilot requires a 4090, observed {device_name}")
    if not torch.cuda.is_bf16_supported():
        raise StudyBError("Study B requires bf16 support")


def verify_runtime_package_lock(path: Path) -> dict[str, object]:
    """Verify the exact Python/GPU dependency lock and single-4090 boundary."""

    import yaml

    if path.is_symlink() or not path.is_file():
        raise StudyBError("Study B package lock must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "python",
        "cuda",
        "packages",
    }:
        raise StudyBError("Study B package lock has a malformed closed schema")
    if payload.get("schema_version") != 1:
        raise StudyBError("Study B package lock schema version differs")
    python_version = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
    if payload.get("python") != python_version:
        raise StudyBError(f"Python version differs from lock: observed {python_version}")
    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or not packages:
        raise StudyBError("Study B package lock has no exact package versions")
    for name, expected in packages.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise StudyBError("Study B package lock package entries must be strings")
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise StudyBError(f"locked GPU package is missing: {name}") from error
        if observed != expected:
            raise StudyBError(
                f"locked GPU package version mismatch for {name}: {observed} != {expected}"
            )
    import torch

    observed_cuda = getattr(torch.version, "cuda", None)
    if observed_cuda != payload.get("cuda"):
        raise StudyBError(f"CUDA version differs from lock: observed {observed_cuda}")
    _require_single_4090()
    return payload


__all__ = [
    "ARMS",
    "MODEL_SNAPSHOT_SHA256",
    "PILOT_SEED",
    "REGISTERED_AXES",
    "QwenStudyBBackend",
    "StudyBBackend",
    "StudyBError",
    "canonical_sha256",
    "evaluation_rows_from_study_a",
    "parse_world",
    "require_offline_environment",
    "run_study_b",
    "sha256_file",
    "summarize_evaluations",
    "tree_sha256",
    "unified_world_prompt",
    "validate_evaluation_rows",
    "validate_support_package",
    "verify_runtime_package_lock",
]
