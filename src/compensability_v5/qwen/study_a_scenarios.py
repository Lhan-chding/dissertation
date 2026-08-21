"""Executable, inference-only Study A audit for the frozen Base and T policies.

The module deliberately keeps model loading behind explicit callables so its
data construction, resume logic, metrics, and publication path are testable on
CPU without suggesting that the registered GPU audit has run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from compensability_v4.qwen.phase5_support import HeldOutNaturalError
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
PHASE2A_PARENT_MANIFEST_SHA256 = "7ca604ec780176d23a890bc7aa0f7d8d73a5bef8fc4be1f524fc0191ec781cbf"
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


def _fiber_size(observed: World, *, operation: str, indices: tuple[int, ...], answer: int) -> int:
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
    if (
        len(matrix) < 2
        or len(matrix) != len(targets)
        or any(type(item) is not int for item in targets)
    ):
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
        (
            "canonical",
            truth,
            observed,
            matrix,
            targets,
            indices,
            {"kind": "identity"},
            (0, 1, 2, 3),
        ),
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
    for (
        axis,
        variant_truth,
        variant_observed,
        variant_matrix,
        variant_targets,
        variant_indices,
        transformation,
        pushforward,
    ) in variants:
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
