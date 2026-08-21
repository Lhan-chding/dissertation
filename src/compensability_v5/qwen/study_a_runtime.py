"""Executable, inference-only Study A audit for the frozen Base and T policies.

The module deliberately keeps model loading behind explicit callables so its
data construction, resume logic, metrics, and publication path are testable on
CPU without suggesting that the registered GPU audit has run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from compensability_v4.qwen.phase5_runtime import (
    completion_log_probability,
    freeze_inference_model,
    generate_completion,
    phase5_rollout_seed,
    tree_sha256,
)
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, parse_world
from compensability_v5.audit.audit_v4_raw import SafeTarArchive
from compensability_v5.audit.fiber_multiplicity import enumerate_one_edit_worlds

BASE_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
T_ADAPTER_SHA256 = "807a61c2e3f7b532b162554dee6e7df83d654fb1f10cc464e9dcb5f6f8efd5c7"
RAW_ARCHIVE_SHA256 = "f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84"
RAW_ARCHIVE_MEMBER = "artifacts/v4/support_dev/held_out_natural_errors.jsonl"
STUDY_A_ACK = "I_UNDERSTAND_THIS_RUNS_V5_STUDY_A_INFERENCE"
CHECKPOINTS = ("Base", "T")
GRAPH_AXES = (
    "canonical",
    "variable_permuted",
    "error_location_permuted",
    "fact_order_permuted",
    "equivalent_basis_graph_ood",
)
VARIABLE_PERMUTATION = (1, 2, 3, 0)
PROMPT_VERSION = "v5-study-a-common-world-text-1"
OBSERVATION_PROMPT_VERSION = "v5-neutral-observation-1"
OBSERVATION_PROMPT = (
    "Read the chart and report the four values in fixed position order. "
    "Reply with exactly four comma-separated integers a,b,c,d and no other text."
)
_INTEGER = re.compile(r"[+-]?\d+")

World = tuple[int, int, int, int]
Matrix = tuple[tuple[int, int, int, int], ...]
CheckpointLoader = Callable[[str], tuple[object, object]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"input is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _world_text(world: World) -> str:
    return ",".join(str(value) for value in world)


def _world(value: object, label: str) -> World:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{label} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _fact_equation(fact: Mapping[str, object]) -> tuple[tuple[int, int, int, int], int]:
    kind = fact.get("type")
    row = [0, 0, 0, 0]
    if kind == "known_value":
        index, target = fact.get("index"), fact.get("value")
        if type(index) is not int or not 0 <= index < 4 or type(target) is not int:
            raise ValueError("known-value fact is malformed")
        row[index] = 1
        return tuple(row), target  # type: ignore[return-value]
    if kind == "pair_sum":
        left, right, target = (
            fact.get("left_index"),
            fact.get("right_index"),
            fact.get("total"),
        )
        if (
            type(left) is not int
            or type(right) is not int
            or left == right
            or not 0 <= left < 4
            or not 0 <= right < 4
            or type(target) is not int
        ):
            raise ValueError("pair-sum fact is malformed")
        row[left] = 1
        row[right] = 1
        return tuple(row), target  # type: ignore[return-value]
    if kind == "arithmetic_progression":
        indices = fact.get("indices")
        if (
            not isinstance(indices, Sequence)
            or isinstance(indices, (str, bytes))
            or len(indices) != 3
            or any(type(index) is not int or not 0 <= index < 4 for index in indices)
            or len(set(indices)) != 3
        ):
            raise ValueError("arithmetic-progression fact is malformed")
        left, middle, right = indices
        row[left], row[middle], row[right] = 1, -2, 1
        return tuple(row), 0  # type: ignore[return-value]
    raise ValueError(f"unregistered natural-error fact type: {kind!r}")


def _linear_system(error: HeldOutNaturalError) -> tuple[Matrix, tuple[int, ...]]:
    equations = tuple(_fact_equation(dict(fact)) for fact in error.facts)
    if len(equations) < 2:
        raise ValueError("Study A equivalent-basis audit requires at least two facts")
    matrix = tuple(row for row, _target in equations)
    targets = tuple(target for _row, target in equations)
    if any(
        sum(coefficient * value for coefficient, value in zip(row, error.truth, strict=True))
        != target
        for row, target in zip(matrix, targets, strict=True)
    ):
        raise ValueError("natural-error facts do not accept the recorded truth")
    return matrix, targets


def _operation(scene_id: str) -> tuple[str, tuple[int, ...]]:
    name = ("sum", "difference", "max_minus_min")[
        int(hashlib.sha256(f"study-a-operation:{scene_id}".encode()).hexdigest(), 16) % 3
    ]
    return name, (0, 1) if name != "max_minus_min" else (0, 1, 2, 3)


def _apply_operation(world: World, name: str, indices: tuple[int, ...]) -> int:
    if name == "sum" and len(indices) == 2:
        return world[indices[0]] + world[indices[1]]
    if name == "difference" and len(indices) == 2:
        return world[indices[0]] - world[indices[1]]
    if name == "max_minus_min" and tuple(sorted(indices)) == (0, 1, 2, 3):
        values = tuple(world[index] for index in indices)
        return max(values) - min(values)
    raise ValueError("Study A answer operation is malformed")


def _fiber_size(
    observed: World, *, operation: str, indices: tuple[int, ...], answer: int
) -> int:
    return sum(
        _apply_operation(candidate, operation, indices) == answer
        for candidate in enumerate_one_edit_worlds(observed, range(2, 19))
    )


def _fiber_bin(size: int) -> str:
    if size == 1:
        return "singleton"
    if size <= 4:
        return "multi_2_4"
    return "multi_5_plus"


def _prompt(observed: World, matrix: Matrix, targets: tuple[int, ...]) -> str:
    equations = "\n".join(
        f"{','.join(str(value) for value in row)} = {target}"
        for row, target in zip(matrix, targets, strict=True)
    )
    return (
        f"Observed values: {_world_text(observed)}\n"
        f"Constraint rows (A | b):\n{equations}\n"
        "Recover the true world. Return exactly four comma-separated integers only.\n"
    )


def _permuted_world(world: World, permutation: tuple[int, int, int, int]) -> World:
    return tuple(world[index] for index in permutation)  # type: ignore[return-value]


def _relocate_error(error: HeldOutNaturalError) -> tuple[World, dict[str, object]]:
    source = error.error_indices[0]
    delta = error.observed[source] - error.truth[source]
    for offset in range(1, 4):
        target = (source + offset) % 4
        for candidate_delta in (delta, -delta):
            replacement = error.truth[target] + candidate_delta
            if 2 <= replacement <= 18:
                observed = list(error.truth)
                observed[target] = replacement
                return tuple(observed), {  # type: ignore[return-value]
                    "kind": "error_location_permutation",
                    "source_index": source,
                    "target_index": target,
                    "source_delta": delta,
                    "target_delta": candidate_delta,
                }
    raise ValueError("natural error cannot be relocated inside the frozen value domain")


@dataclass(frozen=True, slots=True)
class StudyAScenario:
    scenario_id: str
    source_scene_id: str
    orbit_parent: str
    family: str
    graph_axis: str
    truth: World
    observed: World
    constraint_matrix: Matrix
    constraint_targets: tuple[int, ...]
    answer_operation: str
    answer_indices: tuple[int, ...]
    correct_answer: int
    fiber_size: int
    transformation: Mapping[str, object]
    pushforward_permutation: tuple[int, int, int, int]
    prompt: str
    prompt_sha256: str
    capture_label: str = "legacy_single_in_domain"
    error_count: int = 1
    observation_in_domain: bool = True
    observation_strict_parse_success: bool = True
    split: str = "independent_v4_support_dev"

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["truth"] = list(self.truth)
        payload["observed"] = list(self.observed)
        payload["constraint_matrix"] = [list(row) for row in self.constraint_matrix]
        payload["constraint_targets"] = list(self.constraint_targets)
        payload["answer_indices"] = list(self.answer_indices)
        payload["pushforward_permutation"] = list(self.pushforward_permutation)
        payload["transformation"] = dict(self.transformation)
        return payload


def build_study_a_scenarios(error: HeldOutNaturalError) -> tuple[StudyAScenario, ...]:
    """Build the five preregistered text-only views of one natural v4 error."""

    if not isinstance(error, HeldOutNaturalError) or len(error.error_indices) != 1:
        raise ValueError("Study A requires a v4 single natural-error record")
    matrix, targets = _linear_system(error)
    operation, canonical_indices = _operation(error.scene_id)
    relocated, relocation = _relocate_error(error)
    equivalent_matrix = (
        tuple(left + right for left, right in zip(matrix[0], matrix[1], strict=True)),
        *matrix[1:],
    )
    equivalent_targets = (targets[0] + targets[1], *targets[1:])
    variants: tuple[
        tuple[
            str,
            World,
            World,
            Matrix,
            tuple[int, ...],
            tuple[int, ...],
            Mapping[str, object],
        ],
        ...,
    ] = (
        (
            "canonical",
            error.truth,
            error.observed,
            matrix,
            targets,
            canonical_indices,
            {"kind": "identity"},
        ),
        (
            "variable_permuted",
            _permuted_world(error.truth, VARIABLE_PERMUTATION),
            _permuted_world(error.observed, VARIABLE_PERMUTATION),
            tuple(tuple(row[index] for index in VARIABLE_PERMUTATION) for row in matrix),
            targets,
            tuple(VARIABLE_PERMUTATION.index(index) for index in canonical_indices),
            {"kind": "variable_permutation", "permutation": list(VARIABLE_PERMUTATION)},
        ),
        (
            "error_location_permuted",
            error.truth,
            relocated,
            matrix,
            targets,
            canonical_indices,
            relocation,
        ),
        (
            "fact_order_permuted",
            error.truth,
            error.observed,
            tuple(reversed(matrix)),
            tuple(reversed(targets)),
            canonical_indices,
            {"kind": "fact_order_permutation", "order": list(reversed(range(len(matrix))))},
        ),
        (
            "equivalent_basis_graph_ood",
            error.truth,
            error.observed,
            equivalent_matrix,
            equivalent_targets,
            canonical_indices,
            {"kind": "equivalent_basis", "row_operation": "row0_plus_row1"},
        ),
    )
    scenarios: list[StudyAScenario] = []
    for axis, truth, observed, variant_matrix, variant_targets, indices, transformation in variants:
        answer = _apply_operation(truth, operation, indices)
        prompt = _prompt(observed, variant_matrix, variant_targets)
        pushforward = VARIABLE_PERMUTATION if axis == "variable_permuted" else (0, 1, 2, 3)
        scenarios.append(
            StudyAScenario(
                scenario_id=f"{error.scene_id}::{axis}",
                source_scene_id=error.scene_id,
                orbit_parent=error.scene_id,
                family=error.family,
                graph_axis=axis,
                truth=truth,
                observed=observed,
                constraint_matrix=variant_matrix,
                constraint_targets=variant_targets,
                answer_operation=operation,
                answer_indices=indices,
                correct_answer=answer,
                fiber_size=_fiber_size(
                    observed,
                    operation=operation,
                    indices=indices,
                    answer=answer,
                ),
                transformation=transformation,
                pushforward_permutation=pushforward,
                prompt=prompt,
                prompt_sha256=_sha256_bytes(prompt.encode()),
            )
        )
    return tuple(scenarios)


def _phase2_error_location_variant(
    truth: World,
    observed: World,
    matrix: Matrix,
    indices: tuple[int, ...],
) -> tuple[World, World, Matrix, tuple[int, ...], Mapping[str, object], tuple[int, ...]]:
    errors = tuple(index for index in range(4) if truth[index] != observed[index])
    if len(errors) == 1:
        source = errors[0]
        delta = observed[source] - truth[source]
        for offset in range(1, 4):
            target = (source + offset) % 4
            for candidate_delta in (delta, -delta):
                replacement = truth[target] + candidate_delta
                if 2 <= replacement <= 18:
                    relocated = list(truth)
                    relocated[target] = replacement
                    return (
                        truth,
                        tuple(relocated),  # type: ignore[arg-type]
                        matrix,
                        indices,
                        {
                            "kind": "error_location_permutation",
                            "source_index": source,
                            "target_index": target,
                            "source_delta": delta,
                            "target_delta": candidate_delta,
                        },
                        (0, 1, 2, 3),
                    )
    source = errors[0] if errors else 0
    target = (source + 1) % 4
    permutation_list = list(range(4))
    permutation_list[source], permutation_list[target] = target, source
    permutation = tuple(permutation_list)
    return (
        _permuted_world(truth, permutation),  # type: ignore[arg-type]
        _permuted_world(observed, permutation),  # type: ignore[arg-type]
        tuple(tuple(row[index] for index in permutation) for row in matrix),
        tuple(permutation.index(index) for index in indices),
        {
            "kind": "error_location_permutation",
            "permutation": list(permutation),
            "source_error_indices": list(errors),
        },
        permutation,  # type: ignore[return-value]
    )


def build_phase2_study_a_scenarios(scene: Mapping[str, object]) -> tuple[StudyAScenario, ...]:
    """Build Study A views from one frozen Phase-2a captured parent."""

    required = {
        "scene_id",
        "semantic_scene_id",
        "family",
        "truth",
        "natural_observation",
        "constraint_matrix",
        "constraint_targets",
        "answer_operation",
    }
    if required - set(scene):
        raise ValueError("captured Phase-2a scene is missing Study A fields")
    truth = _world(scene["truth"], "Phase-2a truth")
    observed = _world(scene["natural_observation"], "Phase-2a natural observation")
    raw_matrix = scene["constraint_matrix"]
    raw_targets = scene["constraint_targets"]
    if (
        not isinstance(raw_matrix, Sequence)
        or isinstance(raw_matrix, (str, bytes))
        or not isinstance(raw_targets, Sequence)
        or isinstance(raw_targets, (str, bytes))
    ):
        raise ValueError("Phase-2a constraint system is malformed")
    matrix = tuple(_world(row, "Phase-2a constraint row") for row in raw_matrix)
    targets = tuple(raw_targets)
    if len(matrix) < 2 or len(matrix) != len(targets) or any(type(item) is not int for item in targets):
        raise ValueError("Phase-2a constraint rows and targets are malformed")
    operation_payload = scene["answer_operation"]
    if not isinstance(operation_payload, Mapping):
        raise ValueError("Phase-2a answer operation is malformed")
    operation = operation_payload.get("operator")
    raw_indices = operation_payload.get("indices")
    if not isinstance(operation, str) or not isinstance(raw_indices, Sequence):
        raise ValueError("Phase-2a answer operation is malformed")
    indices = tuple(raw_indices)
    if any(type(index) is not int or not 0 <= index < 4 for index in indices):
        raise ValueError("Phase-2a answer indices are malformed")
    error_variant = _phase2_error_location_variant(truth, observed, matrix, indices)
    equivalent_matrix = (
        tuple(left + right for left, right in zip(matrix[0], matrix[1], strict=True)),
        *matrix[1:],
    )
    equivalent_targets = (targets[0] + targets[1], *targets[1:])
    variants = (
        ("canonical", truth, observed, matrix, targets, indices, {"kind": "identity"}, (0, 1, 2, 3)),
        (
            "variable_permuted",
            _permuted_world(truth, VARIABLE_PERMUTATION),
            _permuted_world(observed, VARIABLE_PERMUTATION),
            tuple(tuple(row[index] for index in VARIABLE_PERMUTATION) for row in matrix),
            targets,
            tuple(VARIABLE_PERMUTATION.index(index) for index in indices),
            {"kind": "variable_permutation", "permutation": list(VARIABLE_PERMUTATION)},
            VARIABLE_PERMUTATION,
        ),
        (
            "error_location_permuted",
            error_variant[0],
            error_variant[1],
            error_variant[2],
            targets,
            error_variant[3],
            error_variant[4],
            error_variant[5],
        ),
        (
            "fact_order_permuted",
            truth,
            observed,
            tuple(reversed(matrix)),
            tuple(reversed(targets)),
            indices,
            {"kind": "fact_order_permutation", "order": list(reversed(range(len(matrix))))},
            (0, 1, 2, 3),
        ),
        (
            "equivalent_basis_graph_ood",
            truth,
            observed,
            equivalent_matrix,
            equivalent_targets,
            indices,
            {"kind": "equivalent_basis", "row_operation": "row0_plus_row1"},
            (0, 1, 2, 3),
        ),
    )
    result: list[StudyAScenario] = []
    for axis, variant_truth, variant_observed, variant_matrix, variant_targets, variant_indices, transformation, pushforward in variants:
        answer = _apply_operation(variant_truth, operation, variant_indices)
        prompt = _prompt(variant_observed, variant_matrix, variant_targets)
        result.append(
            StudyAScenario(
                scenario_id=f"{scene['semantic_scene_id']}::{axis}",
                source_scene_id=str(scene["semantic_scene_id"]),
                orbit_parent=str(scene["semantic_scene_id"]),
                family=str(scene["family"]),
                graph_axis=axis,
                truth=variant_truth,
                observed=variant_observed,
                constraint_matrix=variant_matrix,
                constraint_targets=variant_targets,
                answer_operation=operation,
                answer_indices=variant_indices,
                correct_answer=answer,
                fiber_size=_fiber_size(
                    variant_observed,
                    operation=operation,
                    indices=variant_indices,
                    answer=answer,
                ),
                transformation=transformation,
                pushforward_permutation=pushforward,
                prompt=prompt,
                prompt_sha256=_sha256_bytes(prompt.encode()),
                capture_label=str(scene.get("capture_label", "unclassified")),
                error_count=int(scene.get("error_count", 0)),
                observation_in_domain=all(2 <= value <= 18 for value in variant_observed),
                observation_strict_parse_success=bool(
                    scene.get("observation_strict_parse_success", False)
                ),
                split="phase2a_training_source",
            )
        )
    return tuple(result)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return payload


def _read_jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{label} is empty or malformed")
    return rows  # type: ignore[return-value]


def load_phase2a_parents(
    phase2a_root: Path, *, expected_parent_count: int = 96
) -> tuple[tuple[dict[str, object], ...], str]:
    """Verify the immutable Phase-2a parent and return one familiar row per semantic scene."""

    if phase2a_root.is_symlink() or not phase2a_root.is_dir():
        raise RuntimeError("Phase-2a root is missing or unsafe")
    manifest_path = phase2a_root / "parent_manifest.json"
    rows_path = phase2a_root / "pre_model_rows.jsonl"
    manifest = _read_json(manifest_path, "Phase-2a parent manifest")
    rows = _read_jsonl(rows_path, "Phase-2a parent rows")
    if (
        manifest.get("status") != "PHASE_2A_PRE_MODEL_FROZEN"
        or manifest.get("row_count") != len(rows)
        or manifest.get("rows_sha256") != sha256_file(rows_path)
        or manifest.get("model_calls") != 0
        or manifest.get("observation_capture_required") is not True
    ):
        raise RuntimeError("Phase-2a parent manifest provenance drifted")
    familiar = tuple(row for row in rows if row.get("graph_axis") == "familiar")
    if (
        len(familiar) != expected_parent_count
        or len({row.get("semantic_scene_id") for row in familiar}) != len(familiar)
        or any(row.get("observation_status") != "pending_server_capture" for row in rows)
    ):
        raise RuntimeError("Phase-2a familiar parent closure drifted")
    for row in familiar:
        relative = row.get("image_path")
        digest = row.get("image_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("Phase-2a familiar image binding is malformed")
        image = phase2a_root / relative
        if image.resolve(strict=False).is_relative_to(phase2a_root.resolve()) is False:
            raise RuntimeError("Phase-2a familiar image path escapes its root")
        if sha256_file(image) != digest:
            raise RuntimeError("Phase-2a familiar image SHA-256 mismatch")
    return tuple(sorted(familiar, key=lambda row: str(row["semantic_scene_id"]))), sha256_file(
        manifest_path
    )


def _parse_natural_observation(raw: str) -> tuple[World | None, bool]:
    strict = parse_world(raw)
    if strict is not None:
        return strict, True
    tokens = _INTEGER.findall(raw)
    if len(tokens) != 4:
        return None, False
    return tuple(int(token) for token in tokens), False  # type: ignore[return-value]


def _capture_observation(
    model: object,
    processor: object,
    *,
    image_path: Path,
    sample_id: str,
    seed: int,
) -> tuple[str, tuple[int, ...]]:
    shortcut = getattr(model, "study_a_observe", None)
    if callable(shortcut):
        raw, token_ids = shortcut(
            image_path=image_path,
            prompt=OBSERVATION_PROMPT,
            sample_id=sample_id,
            seed=seed,
        )
        return str(raw), tuple(int(item) for item in token_ids)
    from PIL import Image

    from compensability_v4.qwen.manual_generation import generate_observation_with_cache

    with Image.open(image_path) as image:
        result = generate_observation_with_cache(
            model,
            processor,
            image.convert("RGB"),
            OBSERVATION_PROMPT,
            sample_id=sample_id,
            resized_height=280,
            resized_width=280,
            max_new_tokens=32,
            rng_seed=seed,
        )
    return str(result["text"]), tuple(result["generated_token_ids"])  # type: ignore[arg-type]


def _observation_label(truth: World, observed: World, *, strict_parse: bool) -> str:
    errors = sum(left != right for left, right in zip(truth, observed, strict=True))
    in_domain = all(2 <= value <= 18 for value in observed)
    if errors == 1 and in_domain:
        return "primary_single_in_domain"
    if errors == 0 and in_domain:
        return "no_error_control"
    if errors > 1 and in_domain:
        return "stress_multiple_error"
    if errors > 1:
        return "stress_multiple_error_out_of_domain"
    if not in_domain:
        return "stress_out_of_domain"
    return "strict" if strict_parse else "relaxed_parse"


def capture_phase2a_natural_observations(
    *,
    phase2a_root: Path,
    output_root: Path,
    work_root: Path,
    model: object,
    processor: object,
    expected_parent_count: int = 96,
    seed: int = 2026082101,
) -> tuple[tuple[dict[str, object], ...], str]:
    """Capture Base observations once and publish an append-only child manifest."""

    parents, parent_manifest_sha256 = load_phase2a_parents(
        phase2a_root, expected_parent_count=expected_parent_count
    )
    parent_rows_sha256 = sha256_file(phase2a_root / "pre_model_rows.jsonl")
    image_bundle_sha256 = _sha256_bytes(
        "".join(
            f"{row['semantic_scene_id']}\0{row['image_path']}\0{row['image_sha256']}\n"
            for row in parents
        ).encode()
    )
    observation_prompt_sha256 = _sha256_bytes(OBSERVATION_PROMPT.encode())
    metadata = {
        "schema_version": 1,
        "status": "V5_PHASE2A_OBSERVATION_TRACE_IN_PROGRESS",
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_rows_sha256": parent_rows_sha256,
        "image_bundle_sha256": image_bundle_sha256,
        "base_sha256": BASE_SHA256,
        "processor_source_sha256": BASE_SHA256,
        "observation_prompt": OBSERVATION_PROMPT,
        "observation_prompt_sha256": observation_prompt_sha256,
        "observation_prompt_version": OBSERVATION_PROMPT_VERSION,
        "seed": seed,
        "semantic_scene_ids": [row["semantic_scene_id"] for row in parents],
    }
    trace_path, completed = _load_or_create_trace(work_root, metadata)
    # The generic trace loader keys by checkpoint/scenario; capture rows use the
    # same two explicit fields to reuse its duplicate and truncation defenses.
    expected_capture_keys = {
        ("BaseObservation", str(row["semantic_scene_id"])) for row in parents
    }
    if set(completed) - expected_capture_keys:
        raise RuntimeError("Phase-2a observation trace contains unregistered rows")
    for key, row in completed.items():
        if (
            row.get("semantic_scene_id") != key[1]
            or row.get("natural_observation") is None
            or not isinstance(row.get("generated_token_ids"), list)
        ):
            raise RuntimeError("Phase-2a resumed observation row provenance drifted")
    for parent in parents:
        semantic_id = str(parent["semantic_scene_id"])
        key = ("BaseObservation", semantic_id)
        if key in completed:
            continue
        raw, token_ids = _capture_observation(
            model,
            processor,
            image_path=phase2a_root / str(parent["image_path"]),
            sample_id=semantic_id,
            seed=phase5_rollout_seed(seed, semantic_id, 0),
        )
        observed, strict_parse = _parse_natural_observation(raw)
        row = {
            "schema_version": 1,
            "checkpoint": "BaseObservation",
            "scenario_id": semantic_id,
            "semantic_scene_id": semantic_id,
            "raw_output": raw,
            "generated_token_ids": list(token_ids),
            "strict_parse_success": strict_parse,
            "natural_observation": list(observed) if observed is not None else None,
        }
        _append_trace(trace_path, row)
        completed[key] = row
        if observed is None:
            raise RuntimeError(
                f"Phase-2a neutral observation is not deterministically parseable: {semantic_id}"
            )
    captures = {key[1]: row for key, row in completed.items()}
    if set(captures) != {str(row["semantic_scene_id"]) for row in parents}:
        raise RuntimeError("Phase-2a observation capture closure is incomplete")
    frozen: list[dict[str, object]] = []
    for parent in parents:
        semantic_id = str(parent["semantic_scene_id"])
        capture = captures[semantic_id]
        observed = _world(capture["natural_observation"], "captured natural observation")
        truth = _world(parent["truth"], "Phase-2a truth")
        operation = parent["answer_operation"]
        if not isinstance(operation, Mapping):
            raise RuntimeError("Phase-2a answer operation drifted")
        indices = tuple(operation["indices"])  # type: ignore[arg-type]
        answer = _apply_operation(truth, str(operation["operator"]), indices)
        fiber_size = _fiber_size(
            observed,
            operation=str(operation["operator"]),
            indices=indices,
            answer=answer,
        )
        frozen.append(
            {
                "schema_version": 1,
                "scene_id": str(parent["scene_id"]),
                "semantic_scene_id": semantic_id,
                "family": parent["family"],
                "prompt": _prompt(
                    observed,
                    tuple(tuple(row) for row in parent["constraint_matrix"]),  # type: ignore[arg-type]
                    tuple(parent["constraint_targets"]),  # type: ignore[arg-type]
                ),
                "truth": list(truth),
                "natural_observation": list(observed),
                "constraint_matrix": parent["constraint_matrix"],
                "constraint_targets": parent["constraint_targets"],
                "answer_operation": dict(operation),
                "correct_answer": answer,
                "transformation": dict(parent["transformation"]),  # type: ignore[arg-type]
                "graph_axis": "canonical",
                "capture_label": _observation_label(
                    truth, observed, strict_parse=bool(capture["strict_parse_success"])
                ),
                "observation_in_domain": all(2 <= value <= 18 for value in observed),
                "error_count": sum(
                    left != right for left, right in zip(truth, observed, strict=True)
                ),
                "fiber_size": fiber_size,
                "fiber_bin": _fiber_bin(fiber_size),
                "fiber_definition": {
                    "distance": "hamming_at_most_one",
                    "value_domain": [2, 18],
                },
                "observation_prompt_version": OBSERVATION_PROMPT_VERSION,
                "observation_raw_output": capture["raw_output"],
                "observation_strict_parse_success": capture["strict_parse_success"],
                "image_path": parent["image_path"],
                "image_sha256": parent["image_sha256"],
            }
        )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Phase-2a child observation publication already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        scenes_path = temporary / "frozen_scenes.jsonl"
        with scenes_path.open("x", encoding="utf-8") as stream:
            for row in frozen:
                stream.write(_canonical_json(row) + "\n")
        trace_copy = temporary / "observation_trace.jsonl"
        trace_copy.write_bytes(trace_path.read_bytes())
        child = {
            "schema_version": 1,
            "status": "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN",
            "parent_manifest_sha256": parent_manifest_sha256,
            "parent_rows_sha256": parent_rows_sha256,
            "image_bundle_sha256": image_bundle_sha256,
            "parent_manifest_modified": False,
            "base_sha256": BASE_SHA256,
            "processor_source_sha256": BASE_SHA256,
            "observation_prompt_sha256": observation_prompt_sha256,
            "semantic_scene_count": len(frozen),
            "capture_label_counts": dict(
                sorted(
                    (label, sum(row["capture_label"] == label for row in frozen))
                    for label in {str(row["capture_label"]) for row in frozen}
                )
            ),
            "frozen_scenes_sha256": sha256_file(scenes_path),
            "observation_trace_sha256": sha256_file(trace_copy),
            "prompt_search_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        }
        child_path = temporary / "child_manifest.json"
        child_path.write_text(_canonical_json(child) + "\n", encoding="utf-8")
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(frozen), sha256_file(output_root / "child_manifest.json")


def load_phase2a_child(output_root: Path) -> tuple[tuple[dict[str, object], ...], str]:
    manifest_path = output_root / "child_manifest.json"
    scenes_path = output_root / "frozen_scenes.jsonl"
    manifest = _read_json(manifest_path, "Phase-2a child manifest")
    scenes = _read_jsonl(scenes_path, "Phase-2a frozen scenes")
    if (
        manifest.get("status") != "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN"
        or manifest.get("semantic_scene_count") != len(scenes)
        or manifest.get("frozen_scenes_sha256") != sha256_file(scenes_path)
        or manifest.get("parent_manifest_modified") is not False
        or manifest.get("base_sha256") != BASE_SHA256
        or manifest.get("prompt_search_invoked") is not False
        or manifest.get("training_invoked") is not False
        or manifest.get("rl_invoked") is not False
    ):
        raise RuntimeError("Phase-2a child observation provenance drifted")
    return scenes, sha256_file(manifest_path)


def run_phase2a_study_a(
    *,
    phase2a_root: Path,
    child_root: Path,
    capture_work_root: Path,
    output_root: Path,
    audit_work_root: Path,
    checkpoint_loader: CheckpointLoader,
    legacy_errors: Iterable[HeldOutNaturalError] | None = None,
    legacy_raw_archive_sha256: str | None = None,
    expected_parent_count: int = 96,
    k: int = 8,
    sampling_seed: int = 2026082101,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Capture all frozen parents, then execute their five-axis Base/T audit."""

    if child_root.exists() or child_root.is_symlink():
        frozen, child_sha256 = load_phase2a_child(child_root)
    else:
        base, processor = checkpoint_loader("Base")
        freeze_inference_model(base)
        try:
            frozen, child_sha256 = capture_phase2a_natural_observations(
                phase2a_root=phase2a_root,
                output_root=child_root,
                work_root=capture_work_root,
                model=base,
                processor=processor,
                expected_parent_count=expected_parent_count,
                seed=sampling_seed,
            )
        finally:
            del base
    phase2_scenarios = tuple(
        scenario for scene in frozen for scenario in build_phase2_study_a_scenarios(scene)
    )
    legacy_scenarios = tuple(
        scenario
        for error in (legacy_errors or ())
        for scenario in build_study_a_scenarios(error)
    )
    if bool(legacy_scenarios) != bool(legacy_raw_archive_sha256):
        raise ValueError("legacy diagnostic errors and raw archive hash must be supplied together")
    summary = run_study_a(
        scenarios=(*phase2_scenarios, *legacy_scenarios),
        source_name="phase2a_child_manifest",
        source_sha256=child_sha256,
        additional_source_sha256=(
            {"raw_archive": legacy_raw_archive_sha256}
            if legacy_raw_archive_sha256 is not None
            else None
        ),
        output_root=output_root,
        work_root=audit_work_root,
        checkpoint_loader=checkpoint_loader,
        k=k,
        sampling_seed=sampling_seed,
        progress=progress,
    )
    return summary


def load_natural_errors(
    raw_archive: Path, *, expected_sha256: str = RAW_ARCHIVE_SHA256
) -> tuple[tuple[HeldOutNaturalError, ...], str]:
    """Read the one registered member without extracting the supplied archive."""

    with SafeTarArchive(raw_archive, expected_sha256=expected_sha256) as archive:
        rows = archive.read_jsonl(RAW_ARCHIVE_MEMBER)
        observed_sha256 = archive.archive_sha256
    errors = tuple(HeldOutNaturalError.from_mapping(row) for row in rows)
    if not errors or len({error.scene_id for error in errors}) != len(errors):
        raise RuntimeError("Study A natural-error source is empty or duplicated")
    if any(
        len(error.error_indices) != 1 or error.stage1_model_sha256 != BASE_SHA256
        for error in errors
    ):
        raise RuntimeError("Study A source is not the frozen single-error Base capture")
    return tuple(sorted(errors, key=lambda item: item.scene_id)), observed_sha256


def require_study_a_authorization(
    *, execute: bool, acknowledgement: str | None, environment: Mapping[str, str] | None = None
) -> None:
    if not execute:
        raise PermissionError("Study A requires explicit --execute")
    if acknowledgement != STUDY_A_ACK:
        raise PermissionError("Study A requires the exact execution acknowledgement")
    current = os.environ if environment is None else environment
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    if any(current.get(name) != "1" for name in required):
        raise RuntimeError("Study A requires a complete offline environment")


def require_t_adapter(path: Path) -> Path:
    if tree_sha256(path) != T_ADAPTER_SHA256:
        raise RuntimeError("Study A T adapter tree SHA-256 mismatch")
    return path


def load_gpu_checkpoint(checkpoint: str, *, t_adapter: Path) -> tuple[object, object]:
    """Load one verified local checkpoint, with PEFT used only for T."""

    from compensability_v4.qwen.model_loader import load_pinned_qwen

    if checkpoint not in CHECKPOINTS:
        raise ValueError("Study A checkpoint is not registered")
    base, processor = load_pinned_qwen()
    model = base
    if checkpoint == "T":
        require_t_adapter(t_adapter)
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(t_adapter), is_trainable=False)
    freeze_inference_model(model)
    return model, processor


def _scenario_seed(base_seed: int, scenario: StudyAScenario, rollout_index: int) -> int:
    # Excludes checkpoint and transform identity so the full orbit receives
    # common random numbers for paired equivariance comparisons.
    return phase5_rollout_seed(base_seed, scenario.orbit_parent, rollout_index)


def _pushforward(output: World | None, scenario: StudyAScenario) -> World | None:
    return None if output is None else _permuted_world(output, scenario.pushforward_permutation)


def _measure_scenario(
    *,
    model: object,
    processor: object,
    checkpoint: str,
    scenario: StudyAScenario,
    canonical_row: Mapping[str, object] | None,
    k: int,
    sampling_seed: int,
) -> dict[str, object]:
    raw, greedy_ids = generate_completion(
        model,
        processor,
        scenario.prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=32,
        seed=sampling_seed,
    )
    greedy = parse_world(raw)
    truth_text, observed_text = _world_text(scenario.truth), _world_text(scenario.observed)
    logp_truth = completion_log_probability(model, processor, scenario.prompt, truth_text)
    logp_observed = completion_log_probability(model, processor, scenario.prompt, observed_text)
    sample_raw: list[str] = []
    sample_ids: list[list[int]] = []
    sample_seeds: list[int] = []
    sample_outputs: list[World | None] = []
    for index in range(k):
        seed = _scenario_seed(sampling_seed, scenario, index)
        sampled_raw, sampled_ids = generate_completion(
            model,
            processor,
            scenario.prompt,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            max_new_tokens=32,
            seed=seed,
        )
        sample_raw.append(sampled_raw)
        sample_ids.append(list(sampled_ids))
        sample_seeds.append(seed)
        sample_outputs.append(parse_world(sampled_raw))
    successes = [output == scenario.truth for output in sample_outputs]
    pushed_greedy: World | None = None
    paired_consistency: list[bool] = []
    if canonical_row is not None:
        canonical_greedy_raw = canonical_row.get("greedy_output")
        canonical_greedy = (
            _world(canonical_greedy_raw, "cached canonical greedy output")
            if canonical_greedy_raw is not None
            else None
        )
        pushed_greedy = _pushforward(canonical_greedy, scenario)
        paired_consistency.append(greedy is not None and greedy == pushed_greedy)
        canonical_samples = canonical_row.get("sample_outputs")
        if not isinstance(canonical_samples, list) or len(canonical_samples) != k:
            raise RuntimeError("Study A canonical sample trace is malformed")
        for canonical_output, transformed_output in zip(
            canonical_samples, sample_outputs, strict=True
        ):
            parsed_canonical = (
                _world(canonical_output, "cached canonical sampled output")
                if canonical_output is not None
                else None
            )
            expected = _pushforward(parsed_canonical, scenario)
            paired_consistency.append(
                transformed_output is not None and transformed_output == expected
            )
    defect = (
        0.0
        if canonical_row is None
        else 1.0 - sum(paired_consistency) / len(paired_consistency)
    )
    return {
        "schema_version": 1,
        **scenario.to_mapping(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": BASE_SHA256 if checkpoint == "Base" else T_ADAPTER_SHA256,
        "greedy_raw_output": raw,
        "greedy_token_ids": list(greedy_ids),
        "greedy_output": list(greedy) if greedy is not None else None,
        "greedy_parse_success": greedy is not None,
        "greedy_exact_recovery": greedy == scenario.truth,
        "candidate_logp_truth": logp_truth,
        "candidate_logp_observed": logp_observed,
        "candidate_margin_true_observed": logp_truth - logp_observed,
        "sample_raw_outputs": sample_raw,
        "sample_token_ids": sample_ids,
        "sample_seeds": sample_seeds,
        "sample_outputs": [
            list(output) if output is not None else None for output in sample_outputs
        ],
        "sample_exact_recovery": successes,
        "exact_recovery_probability": sum(successes) / k,
        "pass_at_k": any(successes),
        "pushed_forward_canonical_greedy": (
            list(pushed_greedy) if pushed_greedy is not None else None
        ),
        "equivariance_consistent": defect == 0.0,
        "equivariance_defect": defect,
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
    }


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [row[field] for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise RuntimeError(f"Study A metric {field} is malformed")
    return sum(float(value) for value in values) / len(values)


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    noncanonical = [row for row in rows if row["graph_axis"] != "canonical"]
    return {
        "checkpoint_scene_count": len(rows),
        "semantic_scene_count": len({str(row["source_scene_id"]) for row in rows}),
        "mean_fiber_size": _mean(rows, "fiber_size"),
        "greedy_exact_recovery_rate": sum(bool(row["greedy_exact_recovery"]) for row in rows)
        / len(rows),
        "sample_exact_recovery_rate": _mean(rows, "exact_recovery_probability"),
        "pass_at_k_rate": sum(bool(row["pass_at_k"]) for row in rows) / len(rows),
        "candidate_margin_true_observed_mean": _mean(
            rows, "candidate_margin_true_observed"
        ),
        "equivariance_defect_mean": (
            _mean(noncanonical, "equivariance_defect") if noncanonical else 0.0
        ),
    }


def summarize_study_a(
    rows: Sequence[Mapping[str, object]],
    *,
    source_name: str,
    source_sha256: str,
    additional_source_sha256: Mapping[str, str] | None = None,
    k: int,
    sampling_seed: int,
) -> dict[str, object]:
    expected_checkpoints = set(CHECKPOINTS)
    if not rows or {str(row.get("checkpoint")) for row in rows} != expected_checkpoints:
        raise RuntimeError("Study A summary requires complete Base/T evidence")
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = {
        "checkpoint": defaultdict(list),
        "family": defaultdict(list),
        "graph_axis": defaultdict(list),
        "capture_label": defaultdict(list),
        "split": defaultdict(list),
    }
    for row in rows:
        for dimension in grouped:
            grouped[dimension][str(row[dimension])].append(row)
    return {
        "schema_version": 1,
        "status": "V5_STUDY_A_EXECUTED",
        "source_sha256": {
            source_name: source_sha256,
            **dict(additional_source_sha256 or {}),
            "Base": BASE_SHA256,
            "T": T_ADAPTER_SHA256,
        },
        "prompt_template_version": PROMPT_VERSION,
        "semantic_scene_count": len({str(row["source_scene_id"]) for row in rows}),
        "scenario_count": len({str(row["scenario_id"]) for row in rows}),
        "scenario_checkpoint_count": len(rows),
        "k": k,
        "sampling_seed": sampling_seed,
        "by_checkpoint": {
            key: _group_summary(value) for key, value in sorted(grouped["checkpoint"].items())
        },
        "by_family": {
            key: _group_summary(value) for key, value in sorted(grouped["family"].items())
        },
        "by_graph_axis": {
            key: _group_summary(value) for key, value in sorted(grouped["graph_axis"].items())
        },
        "by_capture_label": {
            key: _group_summary(value)
            for key, value in sorted(grouped["capture_label"].items())
        },
        "by_split": {
            key: _group_summary(value) for key, value in sorted(grouped["split"].items())
        },
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
        "confirmatory_data_used": False,
    }


def _trace_metadata(
    *,
    scenarios: Sequence[StudyAScenario],
    source_name: str,
    source_sha256: str,
    additional_source_sha256: Mapping[str, str] | None,
    k: int,
    sampling_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "V5_STUDY_A_RAW_TRACE_IN_PROGRESS",
        "source_sha256": {
            source_name: source_sha256,
            **dict(additional_source_sha256 or {}),
            "Base": BASE_SHA256,
            "T": T_ADAPTER_SHA256,
        },
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "k": k,
        "sampling_seed": sampling_seed,
        "prompt_template_version": PROMPT_VERSION,
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
    }


def _load_or_create_trace(
    work_root: Path, metadata: Mapping[str, object]
) -> tuple[Path, dict[tuple[str, str], dict[str, object]]]:
    if work_root.is_symlink():
        raise RuntimeError("Study A work root must not be a symlink")
    metadata_path, trace_path = work_root / "trace_meta.json", work_root / "raw_trace.jsonl"
    if work_root.exists():
        if not work_root.is_dir() or not metadata_path.is_file() or metadata_path.is_symlink():
            raise RuntimeError("Study A resume root is incomplete or unsafe")
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed != dict(metadata):
            raise RuntimeError("Study A resume metadata drifted")
    else:
        work_root.mkdir(parents=True)
        with metadata_path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json(dict(metadata)) + "\n")
    rows: dict[tuple[str, str], dict[str, object]] = {}
    if trace_path.exists():
        if trace_path.is_symlink() or not trace_path.is_file():
            raise RuntimeError("Study A raw resume trace is unsafe")
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Study A raw trace is malformed at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError("Study A raw trace rows must be objects")
            key = (str(row.get("checkpoint")), str(row.get("scenario_id")))
            if key in rows:
                raise RuntimeError("Study A raw trace contains duplicate rows")
            rows[key] = row
    return trace_path, rows


def _append_trace(trace_path: Path, row: Mapping[str, object]) -> None:
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(dict(row)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_resumed_rows(
    rows: Mapping[tuple[str, str], Mapping[str, object]],
    scenarios: Mapping[str, StudyAScenario],
    *,
    k: int,
    sampling_seed: int,
) -> None:
    for (checkpoint, scenario_id), row in rows.items():
        scenario = scenarios.get(scenario_id)
        expected_checkpoint_hash = BASE_SHA256 if checkpoint == "Base" else T_ADAPTER_SHA256
        if (
            checkpoint not in CHECKPOINTS
            or scenario is None
            or row.get("schema_version") != 1
            or row.get("checkpoint_sha256") != expected_checkpoint_hash
            or row.get("source_scene_id") != scenario.source_scene_id
            or row.get("family") != scenario.family
            or row.get("graph_axis") != scenario.graph_axis
            or row.get("split") != scenario.split
            or row.get("capture_label") != scenario.capture_label
            or row.get("error_count") != scenario.error_count
            or row.get("observation_in_domain") is not scenario.observation_in_domain
            or row.get("observation_strict_parse_success")
            is not scenario.observation_strict_parse_success
            or row.get("prompt_sha256") != scenario.prompt_sha256
            or row.get("truth") != list(scenario.truth)
            or row.get("observed") != list(scenario.observed)
            or row.get("training_invoked") is not False
            or row.get("rl_invoked") is not False
            or row.get("prompt_search_invoked") is not False
        ):
            raise RuntimeError("Study A resumed raw row provenance drifted")
        seeds = row.get("sample_seeds")
        expected_seeds = [_scenario_seed(sampling_seed, scenario, index) for index in range(k)]
        samples = row.get("sample_outputs")
        if seeds != expected_seeds or not isinstance(samples, list) or len(samples) != k:
            raise RuntimeError("Study A resumed raw row sampling contract drifted")


def _atomic_publish(
    *,
    output_root: Path,
    trace_path: Path,
    ordered_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Study A output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        per_scenario = temporary / "per_scenario.jsonl"
        with per_scenario.open("x", encoding="utf-8") as stream:
            for row in ordered_rows:
                stream.write(_canonical_json(dict(row)) + "\n")
        raw_trace = temporary / "raw_trace.jsonl"
        raw_trace.write_bytes(trace_path.read_bytes())
        summary_path = temporary / "summary.json"
        summary_path.write_text(_canonical_json(dict(summary)) + "\n", encoding="utf-8")
        published_files = [per_scenario, raw_trace, summary_path]
        split_names = {
            "phase2a_training_source": "phase2a",
            "independent_v4_support_dev": "legacy_independent",
        }
        by_split = summary.get("by_split")
        for split, prefix in split_names.items():
            split_rows = [row for row in ordered_rows if row.get("split") == split]
            if not split_rows:
                continue
            rows_path = temporary / f"{prefix}_per_scenario.jsonl"
            with rows_path.open("x", encoding="utf-8") as stream:
                for row in split_rows:
                    stream.write(_canonical_json(dict(row)) + "\n")
            split_summary = {
                "schema_version": 1,
                "status": "V5_STUDY_A_SPLIT_EVALUATED",
                "split": split,
                "source_sha256": summary["source_sha256"],
                "metrics": by_split.get(split) if isinstance(by_split, Mapping) else None,
                "checkpoint_scenario_count": len(split_rows),
                "training_invoked": False,
                "rl_invoked": False,
                "prompt_search_invoked": False,
            }
            split_summary_path = temporary / f"{prefix}_summary.json"
            split_summary_path.write_text(
                _canonical_json(split_summary) + "\n", encoding="utf-8"
            )
            published_files.extend((rows_path, split_summary_path))
        base_phase2 = sorted(
            (
                row
                for row in ordered_rows
                if row.get("split") == "phase2a_training_source"
                and row.get("checkpoint") == "Base"
                and row.get("graph_axis") == "canonical"
            ),
            key=lambda row: (
                float(row["exact_recovery_probability"]),
                str(row["source_scene_id"]),
            ),
        )
        if base_phase2:
            support_names = ("low", "medium", "high")
            enriched_path = temporary / "phase2a_enriched_frozen_scenes.jsonl"
            with enriched_path.open("x", encoding="utf-8") as stream:
                for index, row in enumerate(base_phase2):
                    support_bin = support_names[min(2, index * 3 // len(base_phase2))]
                    enriched = {
                        "schema_version": 1,
                        "scene_id": row["scenario_id"],
                        "semantic_scene_id": row["source_scene_id"],
                        "split": row["split"],
                        "family": row["family"],
                        "prompt": row["prompt"],
                        "truth": row["truth"],
                        "natural_observation": row["observed"],
                        "constraint_matrix": row["constraint_matrix"],
                        "constraint_targets": row["constraint_targets"],
                        "answer_operation": {
                            "operator": row["answer_operation"],
                            "indices": row["answer_indices"],
                        },
                        "transformation": row["transformation"],
                        "capture_label": row["capture_label"],
                        "error_count": row["error_count"],
                        "observation_in_domain": row["observation_in_domain"],
                        "observation_strict_parse_success": row[
                            "observation_strict_parse_success"
                        ],
                        "fiber_size": row["fiber_size"],
                        "fiber_bin": _fiber_bin(int(row["fiber_size"])),
                        "base_exact_recovery_probability": row[
                            "exact_recovery_probability"
                        ],
                        "support_bin": support_bin,
                    }
                    stream.write(_canonical_json(enriched) + "\n")
            published_files.append(enriched_path)
        manifest = {
            "schema_version": 1,
            "status": "V5_STUDY_A_ATOMICALLY_PUBLISHED",
            "source_sha256": dict(summary["source_sha256"]),  # type: ignore[arg-type]
            "files": {
                path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in published_files
            },
            "training_invoked": False,
            "rl_invoked": False,
            "prompt_search_invoked": False,
        }
        (temporary / "manifest.json").write_text(
            _canonical_json(manifest) + "\n", encoding="utf-8"
        )
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_study_a(
    *,
    errors: Iterable[HeldOutNaturalError] | None = None,
    scenarios: Iterable[StudyAScenario] | None = None,
    raw_archive_sha256: str | None = None,
    source_name: str = "raw_archive",
    source_sha256: str | None = None,
    additional_source_sha256: Mapping[str, str] | None = None,
    output_root: Path,
    work_root: Path,
    checkpoint_loader: CheckpointLoader,
    k: int = 8,
    sampling_seed: int = 2026082101,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Run or resume every Base/T scenario, then publish one immutable directory."""

    if type(k) is not int or k <= 0 or type(sampling_seed) is not int or sampling_seed <= 0:
        raise ValueError("Study A K and sampling seed must be positive integers")
    effective_sha256 = source_sha256 or raw_archive_sha256
    if not isinstance(effective_sha256, str) or len(effective_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in effective_sha256
    ):
        raise ValueError("Study A source SHA-256 is malformed")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("Study A source name must be non-empty")
    if (errors is None) == (scenarios is None):
        raise ValueError("Study A requires exactly one of natural errors or frozen scenarios")
    if errors is not None:
        error_rows = tuple(sorted(errors, key=lambda item: item.scene_id))
        if not error_rows or len({error.scene_id for error in error_rows}) != len(error_rows):
            raise ValueError("Study A natural errors must be non-empty and unique")
        scenario_rows = tuple(
            scenario for error in error_rows for scenario in build_study_a_scenarios(error)
        )
    else:
        scenario_rows = tuple(scenarios or ())
        if not scenario_rows or len({row.scenario_id for row in scenario_rows}) != len(
            scenario_rows
        ):
            raise ValueError("Study A frozen scenarios must be non-empty and unique")
    metadata = _trace_metadata(
        scenarios=scenario_rows,
        source_name=source_name,
        source_sha256=effective_sha256,
        additional_source_sha256=additional_source_sha256,
        k=k,
        sampling_seed=sampling_seed,
    )
    trace_path, completed = _load_or_create_trace(work_root, metadata)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenario_rows}
    _validate_resumed_rows(
        completed,
        scenario_by_id,
        k=k,
        sampling_seed=sampling_seed,
    )
    expected_keys = {
        (checkpoint, scenario.scenario_id)
        for checkpoint in CHECKPOINTS
        for scenario in scenario_rows
    }
    if set(completed) - expected_keys:
        raise RuntimeError("Study A resume trace contains unregistered checkpoint/scenario rows")
    total = len(expected_keys)
    for checkpoint in CHECKPOINTS:
        missing = [
            scenario
            for scenario in scenario_rows
            if (checkpoint, scenario.scenario_id) not in completed
        ]
        if not missing:
            continue
        model, processor = checkpoint_loader(checkpoint)
        freeze_inference_model(model)
        try:
            for scenario in missing:
                canonical_key = (checkpoint, f"{scenario.source_scene_id}::canonical")
                canonical_row = None if scenario.graph_axis == "canonical" else completed.get(
                    canonical_key
                )
                if scenario.graph_axis != "canonical" and canonical_row is None:
                    raise RuntimeError("Study A canonical row must precede its orbit transforms")
                row = _measure_scenario(
                    model=model,
                    processor=processor,
                    checkpoint=checkpoint,
                    scenario=scenario,
                    canonical_row=canonical_row,
                    k=k,
                    sampling_seed=sampling_seed,
                )
                key = (checkpoint, scenario.scenario_id)
                _append_trace(trace_path, row)
                completed[key] = row
                if progress is not None:
                    progress(checkpoint, len(completed), total)
        finally:
            del model
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
    if set(completed) != expected_keys:
        raise RuntimeError("Study A execution did not close all checkpoint/scenario rows")
    ordered = tuple(
        completed[(checkpoint, scenario.scenario_id)]
        for checkpoint in CHECKPOINTS
        for scenario in scenario_rows
    )
    summary = summarize_study_a(
        ordered,
        source_name=source_name,
        source_sha256=effective_sha256,
        additional_source_sha256=additional_source_sha256,
        k=k,
        sampling_seed=sampling_seed,
    )
    _atomic_publish(
        output_root=output_root,
        trace_path=trace_path,
        ordered_rows=ordered,
        summary=summary,
    )
    return summary


__all__ = [
    "BASE_SHA256",
    "CHECKPOINTS",
    "GRAPH_AXES",
    "PROMPT_VERSION",
    "RAW_ARCHIVE_MEMBER",
    "RAW_ARCHIVE_SHA256",
    "STUDY_A_ACK",
    "T_ADAPTER_SHA256",
    "StudyAScenario",
    "build_study_a_scenarios",
    "load_gpu_checkpoint",
    "load_natural_errors",
    "require_study_a_authorization",
    "require_t_adapter",
    "run_study_a",
    "sha256_file",
    "summarize_study_a",
]
