"""Fixed Phase-N scene plan and original-protocol prevalence collection."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Callable, Sequence, Set
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.gpu_pilot.chart_data import _draw_chart
from compbias.gpu_pilot.structured_generation import (
    build_structured_messages,
    numeric_answer_matches,
    validate_pilot_trajectory,
)
from compbias.gpu_pilot.taxonomy import natural_error_type, pilot_operation_result
from compbias.models.structured_parser import ParseResult, ParseStatus, parse_trajectory

from .config import AnalysisConfig, PhaseNConfig
from .natural_inference import NaturalPrevalenceObservation, analyze_natural_prevalence

_CHART_TYPES = ("grouped_bar", "line")
_OPERATIONS = ("difference", "sum", "max_minus_min")
_VALUE_MIN = 2
_VALUE_MAX = 18


def _answer(values: tuple[int, int, int, int], operation: str) -> int:
    return pilot_operation_result(values, operation)


def _question(operation: str) -> str:
    return {
        "difference": "What is the first value minus the second value?",
        "sum": "What is the sum of the first two values?",
        "max_minus_min": "What is the maximum value minus the minimum value?",
    }[operation]


@dataclass(frozen=True, slots=True)
class PhaseNDatasetRecord:
    schema_version: int
    dataset_id: str
    sample_id: str
    split: str
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]
    question: str
    answer: int
    image: str
    mechanism: str


@dataclass(frozen=True, slots=True)
class PhaseNScene:
    record: PhaseNDatasetRecord
    image_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_bundle_sha256(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(root / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _record_payload(record: PhaseNDatasetRecord) -> dict[str, object]:
    return {**asdict(record), "values": list(record.values)}


def build_phase_n_records(
    config: PhaseNConfig,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
) -> tuple[PhaseNDatasetRecord, ...]:
    """Create the fixed 4,000-scene plan without outcome-adaptive sampling."""

    if not isinstance(config, PhaseNConfig):
        raise TypeError("config must be PhaseNConfig")
    if config.scenes != 4000 or config.max_format_retries != 0:
        raise ValueError("Phase N must remain 4,000 scenes with zero retries")
    if config.allow_sample_extension:
        raise ValueError("Phase N sample extension is forbidden")
    if not isinstance(reserved_numeric_tables, Set):
        raise TypeError("reserved_numeric_tables must be set-like")
    reserved = frozenset(reserved_numeric_tables)
    if any(
        not isinstance(values, tuple)
        or len(values) != 4
        or any(type(value) is not int for value in values)
        for values in reserved
    ):
        raise ValueError("reserved numeric tables are invalid")
    rng = random.Random(config.seed)
    chosen: set[tuple[int, int, int, int]] = set()
    strata = tuple((chart, operation) for chart in _CHART_TYPES for operation in _OPERATIONS)
    records: list[PhaseNDatasetRecord] = []
    for index in range(config.scenes):
        while True:
            candidate = tuple(rng.randint(_VALUE_MIN, _VALUE_MAX) for _ in range(4))
            if candidate not in reserved and candidate not in chosen:
                break
        values = candidate  # type: ignore[assignment]
        chosen.add(values)
        chart_type, operation = strata[index % len(strata)]
        sample_id = f"natural-{index:06d}"
        records.append(
            PhaseNDatasetRecord(
                schema_version=1,
                dataset_id=config.dataset_id,
                sample_id=sample_id,
                split="phase_n",
                chart_type=chart_type,
                operation=operation,
                values=values,
                question=_question(operation),
                answer=_answer(values, operation),
                image=f"images/{sample_id}.png",
                mechanism="iid",
            )
        )
    return tuple(records)


def write_phase_n_dataset(
    config: PhaseNConfig,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
    output_dir: Path,
) -> dict[str, object]:
    """Materialize the deterministic Phase-N plan once before model loading."""

    if not output_dir.is_absolute():
        raise ValueError("Phase N dataset path must be absolute")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite Phase N dataset")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("Phase N dataset parent must be a regular directory")
    records = build_phase_n_records(config, reserved_numeric_tables=reserved_numeric_tables)
    output_dir.mkdir()
    try:
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
                stream.write(json.dumps(_record_payload(record), sort_keys=True) + "\n")
        relative_images = tuple(record.image for record in records)
        strata = Counter(f"{record.chart_type}|{record.operation}" for record in records)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "recoverability_phase_n_dataset",
            "status": "FROZEN_PHASE_N_DATASET_NOT_EVALUATED",
            "dataset_id": config.dataset_id,
            "source_protocol": config.source_protocol,
            "seed": config.seed,
            "record_count": len(records),
            "render_mode": "axis_scale_v0_3",
            "image_size": [512, 384],
            "strata_counts": dict(sorted(strata.items())),
            "records_path": "records.jsonl",
            "records_sha256": _sha256(records_path),
            "images_generated": len(records),
            "images_sha256": _image_bundle_sha256(output_dir, relative_images),
            "format_retries": 0,
            "allow_sample_extension": False,
            "model_calls": 0,
            "training_invoked": False,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def verify_phase_n_dataset(
    config: PhaseNConfig,
    *,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
    dataset_root: Path,
) -> tuple[dict[str, object], tuple[PhaseNScene, ...]]:
    """Replay records and every image digest before the first model call."""

    if not dataset_root.is_absolute() or dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError("Phase N dataset root must be a regular absolute directory")
    expected = build_phase_n_records(config, reserved_numeric_tables=reserved_numeric_tables)
    records_path = dataset_root / "records.jsonl"
    manifest_path = dataset_root / "manifest.json"
    if any(path.is_symlink() or not path.is_file() for path in (records_path, manifest_path)):
        raise ValueError("Phase N dataset artifacts must be regular files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Phase N manifest must be an object")
    rows: list[dict[str, object]] = []
    with records_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Phase N record must be an object")
                rows.append(row)
    if tuple(rows) != tuple(_record_payload(record) for record in expected):
        raise ValueError("Phase N records differ from deterministic replay")
    if manifest.get("records_sha256") != _sha256(records_path):
        raise ValueError("Phase N records digest mismatch")
    root = dataset_root.resolve()
    scenes: list[PhaseNScene] = []
    for record in expected:
        image = (root / record.image).resolve()
        if root not in image.parents or image.is_symlink() or not image.is_file():
            raise ValueError("Phase N image escaped its dataset root")
        scenes.append(PhaseNScene(record, image))
    image_digest = _image_bundle_sha256(root, tuple(record.image for record in expected))
    if manifest.get("images_sha256") != image_digest:
        raise ValueError("Phase N image bundle digest mismatch")
    if manifest.get("record_count") != config.scenes or manifest.get("model_calls") != 0:
        raise ValueError("Phase N manifest contract mismatch")
    return manifest, tuple(scenes)


def build_phase_n_observation(
    record: PhaseNDatasetRecord,
    parsed: ParseResult,
) -> NaturalPrevalenceObservation:
    """Derive the preregistered labels from strict parsed evidence and ground truth."""

    if not isinstance(record, PhaseNDatasetRecord):
        raise TypeError("record must be PhaseNDatasetRecord")
    if not isinstance(parsed, ParseResult):
        raise TypeError("parsed must be ParseResult")
    if parsed.status is not ParseStatus.OK:
        return NaturalPrevalenceObservation(record.sample_id, False, None, None)
    assert parsed.perceived_scene is not None
    perceived = parsed.perceived_scene["values"]
    if not isinstance(perceived, tuple):
        raise ValueError("validated perceived values must be a tuple")
    perception_error = perceived != record.values
    sensitive = bool(
        perception_error and pilot_operation_result(perceived, record.operation) != record.answer
    )
    repair = sensitive and numeric_answer_matches(parsed.answer, record.answer)
    return NaturalPrevalenceObservation(record.sample_id, True, sensitive, repair)


@dataclass(frozen=True, slots=True)
class PhaseNRecord:
    sample_id: str
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]
    expected_answer: int
    raw_text: str
    parse_status: str
    parse_error_code: str | None
    perceived_values: tuple[int, ...] | None
    reported_answer: int | float | None
    error_type: str
    operator_sensitive_error: bool | None
    strict_natural_repair_candidate: bool | None
    format_attempts: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PhaseNReport:
    scenes: int
    model_calls: int
    parsed_scenes: int
    operator_sensitive_errors: int
    strict_natural_repair_candidates: int
    parse_rate: float
    primary_rate: float
    parsed_prevalence: float
    all_attempt_prevalence: float
    one_sided_cp_upper: float
    parse_failure_sensitivity_lower: float
    parse_failure_sensitivity_upper: float
    primary_denominator: str
    h1_supported: bool
    inconclusive: bool
    reason_code: str | None
    error_counts: tuple[tuple[str, int], ...]
    strata_counts: tuple[tuple[str, int], ...]
    format_retries: int
    allow_sample_extension: bool
    training_invoked: bool


def run_phase_n(
    records: tuple[PhaseNDatasetRecord, ...],
    *,
    phase_config: PhaseNConfig,
    analysis_config: AnalysisConfig,
    generate: Callable[[PhaseNDatasetRecord, tuple[dict[str, object], ...]], str],
) -> tuple[PhaseNReport, tuple[PhaseNRecord, ...]]:
    """Collect exactly one legacy trajectory per fixed scene and stop."""

    if not isinstance(records, tuple) or len(records) != phase_config.scenes:
        raise ValueError("Phase N requires exactly the preregistered 4,000 scenes")
    if phase_config.scenes != 4000 or phase_config.max_format_retries != 0:
        raise ValueError("Phase N execution contract drifted")
    if phase_config.allow_sample_extension:
        raise ValueError("Phase N sample extension is forbidden")
    identifiers = tuple(record.sample_id for record in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Phase N scene identifiers must be unique")
    observations: list[NaturalPrevalenceObservation] = []
    rows: list[PhaseNRecord] = []
    error_counts: Counter[str] = Counter()
    strata_counts: Counter[str] = Counter()
    for record in records:
        messages = build_structured_messages(
            question=record.question,
            operation=record.operation,
            retry_index=0,
            expected_value_count=4,
        )
        raw = generate(record, messages)
        if not isinstance(raw, str):
            raise TypeError("Phase N decoder must return text")
        parsed = validate_pilot_trajectory(
            parse_trajectory(raw, sample_id=record.sample_id),
            operation=record.operation,
            expected_value_count=4,
        )
        observation = build_phase_n_observation(record, parsed)
        record_mapping = asdict(record)
        error_type = natural_error_type(record_mapping, parsed)
        perceived = None
        if parsed.perceived_scene is not None:
            candidate = parsed.perceived_scene.get("values")
            if isinstance(candidate, tuple):
                perceived = candidate
        reported = parsed.answer if isinstance(parsed.answer, (int, float)) else None
        attempt = {
            "attempt_index": 0,
            "raw_text": raw,
            "status": parsed.status.value,
            "error_code": parsed.error_code,
        }
        rows.append(
            PhaseNRecord(
                sample_id=record.sample_id,
                chart_type=record.chart_type,
                operation=record.operation,
                values=record.values,
                expected_answer=record.answer,
                raw_text=raw,
                parse_status=parsed.status.value,
                parse_error_code=parsed.error_code,
                perceived_values=perceived,
                reported_answer=reported,
                error_type=error_type,
                operator_sensitive_error=observation.operator_sensitive_error,
                strict_natural_repair_candidate=observation.strict_repair_candidate,
                format_attempts=(attempt,),
            )
        )
        observations.append(observation)
        error_counts[error_type] += 1
        strata_counts[f"{record.chart_type}|{record.operation}"] += 1
    summary = analyze_natural_prevalence(
        tuple(observations),
        null_rate=analysis_config.phase_n_null_rate,
        alpha=analysis_config.alpha,
        minimum_eligible=analysis_config.phase_n_minimum_eligible,
    )
    parsed_scenes = sum(item.parse_success for item in observations)
    eligible = sum(item.operator_sensitive_error is True for item in observations)
    repairs = sum(item.strict_repair_candidate is True for item in observations)
    report = PhaseNReport(
        scenes=len(records),
        model_calls=len(records),
        parsed_scenes=parsed_scenes,
        operator_sensitive_errors=eligible,
        strict_natural_repair_candidates=repairs,
        parse_rate=summary.parse_rate,
        primary_rate=summary.primary_rate,
        parsed_prevalence=summary.parsed_prevalence,
        all_attempt_prevalence=summary.all_attempt_prevalence,
        one_sided_cp_upper=summary.one_sided_cp_upper,
        parse_failure_sensitivity_lower=summary.parse_failure_sensitivity_lower,
        parse_failure_sensitivity_upper=summary.parse_failure_sensitivity_upper,
        primary_denominator=summary.primary_denominator,
        h1_supported=summary.h1_supported,
        inconclusive=summary.inconclusive,
        reason_code=summary.reason_code,
        error_counts=tuple(sorted(error_counts.items())),
        strata_counts=tuple(sorted(strata_counts.items())),
        format_retries=0,
        allow_sample_extension=False,
        training_invoked=False,
    )
    return report, tuple(rows)
