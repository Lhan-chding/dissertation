"""Development-only Stage-1 v2 interface probe after the failed v1 bridge."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .bridge import Stage1Evidence, parse_stage1_evidence
from .evidence import SERVER_PACKAGE_PATHS, ProtocolLockResult, verify_protocol_lock

_CHART_TYPES = frozenset({"grouped_bar", "line"})
_OPERATIONS = frozenset({"difference", "max_minus_min", "sum"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PROBE_PER_STRATUM = 4
_PROBE_SCENES = len(_CHART_TYPES) * len(_OPERATIONS) * _PROBE_PER_STRATUM
STAGE1_V2_SERVER_PACKAGE_LOCK_PATH = "configs/recoverability/server_package_lock_stage1_v2.yaml"
STAGE1_V2_SERVER_PACKAGE_PATHS = SERVER_PACKAGE_PATHS | frozenset(
    {
        "configs/paths.example.yaml",
        "configs/recoverability/bridge_v1_failure.yaml",
        "configs/recoverability/stage1_v2_probe.yaml",
        "experiments/recoverability_v1/00_stage1_v2_preflight.py",
        "experiments/recoverability_v1/04_stage1_v2_probe.py",
        "src/compbias/recoverability/bridge_v1_failure.py",
        "src/compbias/recoverability/stage1_v2.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage1_v2_runtime_paths(
    runtime_path: Path,
    *,
    registered_example: Path,
) -> None:
    """Require the server-local paths file to match the locked template exactly."""

    for candidate, label in (
        (runtime_path, "runtime paths"),
        (registered_example, "registered paths example"),
    ):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"{label} must be a regular file")
    if _sha256(runtime_path) != _sha256(registered_example):
        raise ValueError("runtime paths must byte-match the locked registered example")


def verify_stage1_v2_server_package_lock(
    path: Path,
    *,
    repository_root: Path,
) -> ProtocolLockResult:
    """Verify the one registered v2 probe closure without changing the v1 verifier."""

    root = repository_root.resolve()
    expected_path = root / STAGE1_V2_SERVER_PACKAGE_LOCK_PATH
    if path.resolve() != expected_path or path.is_symlink():
        raise ValueError("Stage-1 v2 server package lock path is not canonical")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != STAGE1_V2_SERVER_PACKAGE_PATHS:
        missing = sorted(STAGE1_V2_SERVER_PACKAGE_PATHS - observed)
        extra = sorted(observed - STAGE1_V2_SERVER_PACKAGE_PATHS)
        raise ValueError(
            f"Stage-1 v2 server package lock closure mismatch; missing={missing}, extra={extra}"
        )
    return result


@dataclass(frozen=True, slots=True)
class Stage1V2ProbeConfig:
    schema_version: int
    status: str
    dataset_id: str
    output_subdirectory: str
    source_dataset_id: str
    source_split: str
    scenes: int
    per_stratum: int
    format_retries: int
    required_parse_rate: float
    allow_rerun: bool
    hypothesis_test: bool


def load_stage1_v2_probe_config(path: Path) -> Stage1V2ProbeConfig:
    """Load the closed development-only one-shot probe contract."""

    mapping = load_yaml_mapping(path, label="Stage-1 v2 probe config")
    fields = {
        "schema_version",
        "status",
        "dataset_id",
        "output_subdirectory",
        "source_dataset_id",
        "source_split",
        "scenes",
        "per_stratum",
        "format_retries",
        "required_parse_rate",
        "allow_rerun",
        "hypothesis_test",
    }
    reject_unknown_fields(mapping, fields, label="Stage-1 v2 probe config")
    if set(mapping) != fields:
        raise ValueError("Stage-1 v2 probe config must contain every registered field")
    expected: dict[str, object] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_PROBE_NOT_RUN",
        "dataset_id": "CVA-Recoverability-Stage1-V2-Dev-Probe",
        "output_subdirectory": "stage1_v2_dev_probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "scenes": _PROBE_SCENES,
        "per_stratum": _PROBE_PER_STRATUM,
        "format_retries": 0,
        "required_parse_rate": 1.0,
        "allow_rerun": False,
        "hypothesis_test": False,
    }
    if dict(mapping) != expected:
        raise ValueError("Stage-1 v2 probe config differs from the registered contract")
    return Stage1V2ProbeConfig(**expected)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Stage1V2Scene:
    scene_id: str
    image_path: Path
    chart_type: str
    operation: str
    values: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or _IDENTIFIER.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a bounded safe identifier")
        if not isinstance(self.image_path, Path) or not self.image_path.is_absolute():
            raise ValueError("image_path must be absolute")
        if self.chart_type not in _CHART_TYPES:
            raise ValueError("chart_type is not registered")
        if self.operation not in _OPERATIONS:
            raise ValueError("operation is not registered")
        if not isinstance(self.values, tuple) or len(self.values) != 4:
            raise ValueError("values must contain exactly four integers")
        if any(type(value) is not int for value in self.values):
            raise TypeError("values must contain exact integers")


@dataclass(frozen=True, slots=True)
class Stage1V2ProbeRecord:
    scene_id: str
    chart_type: str
    operation: str
    raw_text: str
    parse_success: bool
    exact_transcription: bool
    error_code: str | None


@dataclass(frozen=True, slots=True)
class Stage1V2ProbeReport:
    scenes: int
    model_calls: int
    parse_rate: float
    exact_transcription_rate: float
    error_counts: tuple[tuple[str, int], ...]
    format_retries: int
    training_invoked: bool
    probe_passed: bool


def build_stage1_v2_messages() -> tuple[dict[str, object], ...]:
    """Return the frozen question-free four-position transcription prompt."""

    grammar = (
        '{"target_facts":[INTEGER,INTEGER,INTEGER,INTEGER],'
        '"redundant_facts":[],"axis_facts":["integer_ticks"]}'
    )
    system = (
        "You are a strict chart transcription interface. Read every plotted mark at positions "
        "A, B, C, and D from left to right. Return exactly one JSON object by replacing each "
        f"INTEGER in this literal grammar: {grammar} All four integers are required. The output "
        "must begin with { and end with }. Do not compute an answer or operation. Do not emit "
        "Markdown fences, prose, reasoning, extra keys, or trailing text."
    )
    user = (
        "Transcribe all four plotted values at positions A, B, C, and D. Do not compute anything."
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )


def _exact_int_values(value: object) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("probe values must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _relative_image(value: object, *, dataset_root: Path) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("probe image must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("probe image must remain inside the dataset root")
    root = dataset_root.resolve()
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError("probe image must remain inside the dataset root")
    return resolved


def select_stage1_v2_probe_scenes(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset_root: Path,
) -> tuple[Stage1V2Scene, ...]:
    """Select the first four canonical dev rows in every chart/operator stratum."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mappings")
    if not isinstance(dataset_root, Path) or not dataset_root.is_absolute():
        raise ValueError("dataset_root must be absolute")
    candidates: dict[tuple[str, str], list[Stage1V2Scene]] = {
        (chart, operation): []
        for chart in sorted(_CHART_TYPES)
        for operation in sorted(_OPERATIONS)
    }
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every probe row must be a mapping")
        if row.get("split") != "dev":
            continue
        scene_id = row.get("sample_id")
        if not isinstance(scene_id, str) or _IDENTIFIER.fullmatch(scene_id) is None:
            raise ValueError("probe sample_id must be a bounded safe identifier")
        if scene_id in identifiers:
            raise ValueError("probe dev scene identifiers must be unique")
        identifiers.add(scene_id)
        chart_type = row.get("chart_type")
        operation = row.get("operation")
        if chart_type not in _CHART_TYPES or operation not in _OPERATIONS:
            raise ValueError("probe dev row uses an unregistered stratum")
        scene = Stage1V2Scene(
            scene_id=scene_id,
            image_path=_relative_image(row.get("image"), dataset_root=dataset_root),
            chart_type=str(chart_type),
            operation=str(operation),
            values=_exact_int_values(row.get("values")),
        )
        candidates[(scene.chart_type, scene.operation)].append(scene)
    selected: list[Stage1V2Scene] = []
    for key in sorted(candidates):
        available = sorted(candidates[key], key=lambda scene: scene.scene_id)
        if len(available) < _PROBE_PER_STRATUM:
            raise ValueError(
                "Stage-1 v2 probe stratum underfilled: "
                f"chart_type={key[0]}, operation={key[1]}, "
                f"available={len(available)}, required={_PROBE_PER_STRATUM}"
            )
        selected.extend(available[:_PROBE_PER_STRATUM])
    if len(selected) != _PROBE_SCENES:
        raise AssertionError("Stage-1 v2 probe selection size drifted")
    return tuple(selected)


def _error_code(error: ValueError) -> str:
    message = str(error)
    return {
        "Stage-1 output must be one exact JSON object": "not_exact_json_object",
        "Stage-1 evidence schema is invalid": "schema_invalid",
        "target_facts must contain exactly four integers": "target_facts_not_four_integers",
        "bridge redundant_facts must be the empty list": "redundant_facts_not_empty",
        "axis_facts must contain bounded identifiers": "axis_facts_invalid",
        "Stage-1 output must be bounded text": "output_not_bounded_text",
    }.get(message, "other_strict_parse_failure")


def run_stage1_v2_probe(
    scenes: tuple[Stage1V2Scene, ...],
    *,
    generate: Callable[[Stage1V2Scene, tuple[dict[str, object], ...]], str],
) -> tuple[Stage1V2ProbeReport, tuple[Stage1V2ProbeRecord, ...]]:
    """Run one deterministic image call per scene with no retries or Stage 2."""

    if not isinstance(scenes, tuple) or not scenes:
        raise ValueError("probe scenes must be a non-empty tuple")
    if any(not isinstance(scene, Stage1V2Scene) for scene in scenes):
        raise TypeError("probe scenes must contain Stage1V2Scene instances")
    identifiers = tuple(scene.scene_id for scene in scenes)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("probe scene identifiers must be unique")
    if not callable(generate):
        raise TypeError("generate must be callable")
    messages = build_stage1_v2_messages()
    records: list[Stage1V2ProbeRecord] = []
    counts: Counter[str] = Counter()
    parsed_count = 0
    exact_count = 0
    for scene in scenes:
        raw = generate(scene, messages)
        evidence: Stage1Evidence | None = None
        error_code: str | None = None
        try:
            evidence = parse_stage1_evidence(raw)
        except ValueError as error:
            error_code = _error_code(error)
            counts[error_code] += 1
        parse_success = evidence is not None
        exact = parse_success and evidence.target_facts == scene.values
        parsed_count += int(parse_success)
        exact_count += int(exact)
        records.append(
            Stage1V2ProbeRecord(
                scene_id=scene.scene_id,
                chart_type=scene.chart_type,
                operation=scene.operation,
                raw_text=raw,
                parse_success=parse_success,
                exact_transcription=exact,
                error_code=error_code,
            )
        )
    total = len(records)
    parse_rate = parsed_count / total
    return (
        Stage1V2ProbeReport(
            scenes=total,
            model_calls=total,
            parse_rate=parse_rate,
            exact_transcription_rate=exact_count / total,
            error_counts=tuple(sorted(counts.items())),
            format_retries=0,
            training_invoked=False,
            probe_passed=total == _PROBE_SCENES and parse_rate == 1.0,
        ),
        tuple(records),
    )
