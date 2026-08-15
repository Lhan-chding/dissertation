"""Deterministic one-shot screen for the amended confirmatory Phase C."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from compbias.gpu_pilot.chart_data import _draw_chart

from .bridge import Stage1Evidence, parse_stage1_evidence
from .compatibility import (
    ArithmeticProgressionConstraint,
    KnownValueConstraint,
    PairSumConstraint,
    VisibleConstraint,
)
from .operators import apply_operation
from .phase_c_amendment import PhaseCAmendment
from .selection import SceneCandidate, select_fixed_family_quotas
from .stage1_v2 import build_stage1_v2_messages

_CHART_TYPES = ("grouped_bar", "line")
_OPERATIONS = ("difference", "sum", "max_minus_min")
_FAMILIES = ("cross_series", "duplicate_encoding", "trend")
_VALUE_DOMAIN = tuple(range(2, 19))


def _question(operation: str) -> str:
    return {
        "difference": "What is the first value minus the second value?",
        "sum": "What is the sum of the first two values?",
        "max_minus_min": "What is the maximum value minus the minimum value?",
    }[operation]


@dataclass(frozen=True, slots=True)
class PhaseCScreenDatasetRecord:
    schema_version: int
    dataset_id: str
    scene_id: str
    split: str
    family: str
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]
    question: str
    answer: int
    image: str
    format_retries: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Phase C screen schema_version must equal one")
        if self.family not in _FAMILIES:
            raise ValueError("Phase C screen family is not registered")
        if self.chart_type not in _CHART_TYPES or self.operation not in _OPERATIONS:
            raise ValueError("Phase C screen stratum is not registered")
        if self.split != "phase_c_screen":
            raise ValueError("Phase C screen split is invalid")
        if not isinstance(self.values, tuple) or len(self.values) != 4:
            raise ValueError("Phase C values must be an exact four-integer tuple")
        if any(type(value) is not int or value not in _VALUE_DOMAIN for value in self.values):
            raise ValueError("Phase C values must lie in the registered domain")
        if type(self.answer) is not int or self.answer != apply_operation(
            self.values, self.operation
        ):
            raise ValueError("Phase C answer must be derived from the registered operation")
        if self.format_retries != 0 or type(self.format_retries) is not int:
            raise ValueError("Phase C screen format retries must remain zero")


@dataclass(frozen=True, slots=True)
class PhaseCScreenScene:
    record: PhaseCScreenDatasetRecord
    image_path: Path


@dataclass(frozen=True, slots=True)
class PhaseCScreenRecord:
    scene_id: str
    family: str
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]
    raw_text: str
    parse_success: bool
    parse_error: str | None
    perceived_values: tuple[int, int, int, int] | None
    natural_perception_error: bool
    one_position_error: bool
    operator_sensitive: bool
    design_recoverability_validated: bool
    eligible: bool
    selected: bool


@dataclass(frozen=True, slots=True)
class PhaseCScreenReport:
    scenes: int
    model_calls: int
    parse_successes: int
    parse_rate: float
    natural_perception_errors: int
    one_position_errors: int
    operator_sensitive_errors: int
    design_recoverable_errors: int
    eligible_by_family: tuple[tuple[str, int], ...]
    selected_by_family: tuple[tuple[str, int], ...]
    selected_scene_ids: tuple[str, ...]
    failure_codes: tuple[str, ...]
    screen_passed: bool
    confirmatory_arm_execution_authorized: bool
    format_retries: int
    allow_sample_extension: bool
    allow_quota_redistribution: bool
    training_authorized: bool
    rl_authorized: bool
    training_invoked: bool


def _trend_constraints(
    values: tuple[int, int, int, int],
) -> tuple[ArithmeticProgressionConstraint, ...]:
    constraints: list[ArithmeticProgressionConstraint] = []
    for middle in range(4):
        others = [index for index in range(4) if index != middle]
        for left, right in combinations(others, 2):
            if 2 * values[middle] == values[left] + values[right]:
                constraints.append(
                    ArithmeticProgressionConstraint(
                        f"trend-{left}-{middle}-{right}",
                        (left, middle, right),
                    )
                )
    return tuple(constraints)


def build_family_constraints(
    family: str,
    values: tuple[int, int, int, int],
) -> tuple[VisibleConstraint, ...]:
    """Create the registered redundant cue without using an observed error position."""

    if family == "cross_series":
        return tuple(
            PairSumConstraint(f"pair-{left}-{right}", left, right, values[left] + values[right])
            for left, right in combinations(range(4), 2)
        )
    if family == "duplicate_encoding":
        return tuple(
            KnownValueConstraint(f"duplicate-{index}", index, value)
            for index, value in enumerate(values)
        )
    if family == "trend":
        constraints = _trend_constraints(values)
        if len(constraints) < 2:
            raise ValueError("trend scene must entail at least two registered progressions")
        return constraints
    raise ValueError("Phase C family is not registered")


def _valid_reserved(values: object) -> bool:
    return (
        isinstance(values, tuple)
        and len(values) == 4
        and all(type(value) is int for value in values)
    )


def build_phase_c_screen_records(
    amendment: PhaseCAmendment,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
) -> tuple[PhaseCScreenDatasetRecord, ...]:
    """Build 8,000 fresh scenes balanced over family, chart, and operation."""

    if not isinstance(amendment, PhaseCAmendment):
        raise TypeError("amendment must be a PhaseCAmendment")
    if amendment.intake_scenes != 8000 or amendment.format_retries != 0:
        raise ValueError("Phase C screen contract drifted")
    if not isinstance(reserved_numeric_tables, Set):
        raise TypeError("reserved_numeric_tables must be set-like")
    if any(not _valid_reserved(values) for values in reserved_numeric_tables):
        raise ValueError("reserved numeric tables are invalid")
    reserved = frozenset(reserved_numeric_tables)
    strata = tuple(
        (family, chart, operation)
        for family in _FAMILIES
        for chart in _CHART_TYPES
        for operation in _OPERATIONS
    )
    rng = random.Random(amendment.seed)
    chosen: set[tuple[int, int, int, int]] = set()
    records: list[PhaseCScreenDatasetRecord] = []
    for index in range(amendment.intake_scenes):
        family, chart_type, operation = strata[index % len(strata)]
        while True:
            candidate = tuple(rng.randint(_VALUE_DOMAIN[0], _VALUE_DOMAIN[-1]) for _ in range(4))
            if candidate in reserved or candidate in chosen:
                continue
            if family == "trend" and len(_trend_constraints(candidate)) < 2:
                continue
            break
        values = candidate  # type: ignore[assignment]
        chosen.add(values)
        scene_id = f"phase-c-screen-{index:06d}"
        records.append(
            PhaseCScreenDatasetRecord(
                schema_version=1,
                dataset_id=amendment.dataset_id,
                scene_id=scene_id,
                split="phase_c_screen",
                family=family,
                chart_type=chart_type,
                operation=operation,
                values=values,
                question=_question(operation),
                answer=apply_operation(values, operation),
                image=f"images/{scene_id}.png",
                format_retries=0,
            )
        )
    return tuple(records)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_bundle_sha256(root: Path, records: Sequence[PhaseCScreenDatasetRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.image):
        digest.update(record.image.encode())
        digest.update(b"\0")
        digest.update(_sha256(root / record.image).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _payload(record: PhaseCScreenDatasetRecord) -> dict[str, object]:
    return {**asdict(record), "values": list(record.values)}


def write_phase_c_screen_dataset(
    amendment: PhaseCAmendment,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
    output_dir: Path,
) -> dict[str, object]:
    """Materialize the fixed screen once before loading the model."""

    if not output_dir.is_absolute():
        raise ValueError("Phase C screen dataset path must be absolute")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite Phase C screen dataset")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("Phase C screen dataset parent must be a regular directory")
    records = build_phase_c_screen_records(
        amendment, reserved_numeric_tables=reserved_numeric_tables
    )
    output_dir.mkdir()
    try:
        (output_dir / "images").mkdir()
        for record in records:
            _draw_chart(
                output_dir / record.image,
                chart_type=record.chart_type,
                values=record.values,
                size=(512, 384),
                ood=False,
                render_mode="axis_scale_v0_3",
            )
        records_path = output_dir / "records.jsonl"
        with records_path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(_payload(record), sort_keys=True) + "\n")
        strata = Counter(
            f"{record.family}|{record.chart_type}|{record.operation}" for record in records
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "recoverability_phase_c_v2_screen_dataset",
            "status": "FROZEN_PHASE_C_SCREEN_DATASET_NOT_EVALUATED",
            "dataset_id": amendment.dataset_id,
            "seed": amendment.seed,
            "record_count": len(records),
            "strata_counts": dict(sorted(strata.items())),
            "records_sha256": _sha256(records_path),
            "images_sha256": _image_bundle_sha256(output_dir, records),
            "render_mode": "axis_scale_v0_3",
            "format_retries": 0,
            "model_calls": 0,
            "training_invoked": False,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def verify_phase_c_screen_dataset(
    amendment: PhaseCAmendment,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
    dataset_root: Path,
) -> tuple[dict[str, object], tuple[PhaseCScreenScene, ...]]:
    """Replay records and image hashes before the first screen call."""

    if not dataset_root.is_absolute() or dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("Phase C screen dataset root must be a regular absolute directory")
    expected = build_phase_c_screen_records(
        amendment, reserved_numeric_tables=reserved_numeric_tables
    )
    records_path = dataset_root / "records.jsonl"
    manifest_path = dataset_root / "manifest.json"
    if any(path.is_symlink() or not path.is_file() for path in (records_path, manifest_path)):
        raise ValueError("Phase C screen dataset artifacts must be regular files")
    rows = tuple(
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line
    )
    if rows != tuple(_payload(record) for record in expected):
        raise ValueError("Phase C screen records differ from deterministic replay")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("records_sha256") != _sha256(records_path):
        raise ValueError("Phase C screen manifest or records digest is invalid")
    root = dataset_root.resolve()
    scenes: list[PhaseCScreenScene] = []
    for record in expected:
        image = (root / record.image).resolve()
        if root not in image.parents or image.is_symlink() or not image.is_file():
            raise ValueError("Phase C screen image escaped the dataset root")
        scenes.append(PhaseCScreenScene(record, image))
    if manifest.get("images_sha256") != _image_bundle_sha256(root, expected):
        raise ValueError("Phase C screen image bundle digest differs")
    if manifest.get("record_count") != 8000 or manifest.get("model_calls") != 0:
        raise ValueError("Phase C screen manifest contract differs")
    return manifest, tuple(scenes)


def _compatible_answers(
    record: PhaseCScreenDatasetRecord,
    perceived: tuple[int, int, int, int],
) -> tuple[int, ...]:
    constraints = build_family_constraints(record.family, record.values)
    candidates = {perceived}
    for index in range(4):
        for value in _VALUE_DOMAIN:
            values = list(perceived)
            values[index] = value
            candidates.add(tuple(values))  # type: ignore[arg-type]
    compatible = [
        values
        for values in candidates
        if all(constraint.accepts(values) for constraint in constraints)
    ]
    return tuple(sorted({apply_operation(values, record.operation) for values in compatible}))


def evaluate_phase_c_screen(
    records: tuple[PhaseCScreenDatasetRecord, ...],
    *,
    amendment: PhaseCAmendment,
    generate: Callable[[PhaseCScreenDatasetRecord, tuple[dict[str, object], ...]], str],
    quotas: Mapping[str, int] | None = None,
) -> tuple[PhaseCScreenReport, tuple[PhaseCScreenRecord, ...]]:
    """Call Stage 1 once per scene and select fixed family quotas without reruns."""

    if not isinstance(records, tuple) or not records:
        raise ValueError("Phase C screen records must be a non-empty tuple")
    if any(not isinstance(record, PhaseCScreenDatasetRecord) for record in records):
        raise TypeError("Phase C screen records contain an invalid item")
    identifiers = tuple(record.scene_id for record in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Phase C screen scene identifiers must be unique")
    if not callable(generate):
        raise TypeError("generate must be callable")
    registered_quotas = dict(amendment.selected_family_quotas) if quotas is None else dict(quotas)
    if set(record.family for record in records) - set(registered_quotas):
        raise ValueError("screen record family is absent from quotas")
    messages = build_stage1_v2_messages()
    provisional: list[PhaseCScreenRecord] = []
    candidates: list[SceneCandidate] = []
    for record in records:
        raw = generate(record, messages)
        if not isinstance(raw, str):
            raise TypeError("Phase C screen decoder must return text")
        evidence: Stage1Evidence | None = None
        parse_error: str | None = None
        try:
            evidence = parse_stage1_evidence(raw)
        except ValueError as error:
            parse_error = str(error)
        perceived = evidence.target_facts if evidence is not None else None
        perception_error = perceived is not None and perceived != record.values
        mismatch_count = (
            sum(left != right for left, right in zip(perceived, record.values, strict=True))
            if perceived is not None
            else 0
        )
        one_position = perception_error and mismatch_count == 1
        operator_sensitive = bool(
            one_position
            and perceived is not None
            and apply_operation(perceived, record.operation) != record.answer
        )
        answers = _compatible_answers(record, perceived) if operator_sensitive and perceived else ()
        design_valid = answers == (record.answer,)
        eligible = bool(evidence is not None and operator_sensitive and design_valid)
        provisional.append(
            PhaseCScreenRecord(
                scene_id=record.scene_id,
                family=record.family,
                chart_type=record.chart_type,
                operation=record.operation,
                values=record.values,
                raw_text=raw,
                parse_success=evidence is not None,
                parse_error=parse_error,
                perceived_values=perceived,
                natural_perception_error=bool(perception_error),
                one_position_error=bool(one_position),
                operator_sensitive=operator_sensitive,
                design_recoverability_validated=design_valid,
                eligible=eligible,
                selected=False,
            )
        )
        candidates.append(
            SceneCandidate(
                scene_id=record.scene_id,
                family=record.family,
                stage1_parse_success=evidence is not None,
                natural_perception_error=bool(perception_error),
                operator_sensitive=operator_sensitive,
                design_recoverability_validated=design_valid,
            )
        )
    failures: tuple[str, ...] = ()
    selected: tuple[SceneCandidate, ...] = ()
    try:
        selected = select_fixed_family_quotas(
            candidates, quotas=registered_quotas, seed=amendment.seed
        )
    except ValueError as error:
        if not str(error).startswith("quota unmet for "):
            raise
        failures = ("phase_c_screen_family_quota_unmet",)
    selected_ids = frozenset(item.scene_id for item in selected)
    final_rows = tuple(
        PhaseCScreenRecord(**{**asdict(row), "selected": row.scene_id in selected_ids})
        for row in provisional
    )
    eligible_counts = Counter(row.family for row in final_rows if row.eligible)
    selected_counts = Counter(row.family for row in final_rows if row.selected)
    passed = not failures
    report = PhaseCScreenReport(
        scenes=len(records),
        model_calls=len(records),
        parse_successes=sum(row.parse_success for row in final_rows),
        parse_rate=sum(row.parse_success for row in final_rows) / len(final_rows),
        natural_perception_errors=sum(row.natural_perception_error for row in final_rows),
        one_position_errors=sum(row.one_position_error for row in final_rows),
        operator_sensitive_errors=sum(row.operator_sensitive for row in final_rows),
        design_recoverable_errors=sum(row.design_recoverability_validated for row in final_rows),
        eligible_by_family=tuple(sorted(eligible_counts.items())),
        selected_by_family=tuple(sorted(selected_counts.items())),
        selected_scene_ids=tuple(item.scene_id for item in selected),
        failure_codes=failures,
        screen_passed=passed,
        confirmatory_arm_execution_authorized=passed,
        format_retries=0,
        allow_sample_extension=False,
        allow_quota_redistribution=False,
        training_authorized=False,
        rl_authorized=False,
        training_invoked=False,
    )
    return report, final_rows
