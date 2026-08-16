"""Frozen Phase-C v2 screen result and exact server-artifact replay."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PhaseCScreenFrozenResult:
    schema_version: int
    status: str
    server_revision_observed: str
    amendment_id: str
    dataset_id: str
    phase_c_screen_exit: int
    scenes: int
    model_calls: int
    parse_successes: int
    parse_rate: float
    natural_perception_errors: int
    one_position_errors: int
    operator_sensitive_errors: int
    design_recoverable_errors: int
    eligible_scenes: int
    eligible_by_family: tuple[tuple[str, int], ...]
    selected_scenes: int
    selected_by_family: tuple[tuple[str, int], ...]
    screen_passed: bool
    confirmatory_arm_execution_authorized: bool
    failure_codes: tuple[str, ...]
    format_retries: int
    allow_sample_extension: bool
    allow_quota_redistribution: bool
    training_authorized: bool
    rl_authorized: bool
    training_invoked: bool
    model_snapshot_sha256: str
    server_package_lock_sha256: str
    source_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FrozenEligibleScene:
    scene_id: str
    family: str
    chart_type: str
    operation: str
    true_values: tuple[int, int, int, int]
    perceived_values: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not self.scene_id or not isinstance(self.scene_id, str):
            raise ValueError("scene_id must be non-empty text")
        if self.family not in {"cross_series", "trend", "duplicate_encoding"}:
            raise ValueError("eligible scene family is not registered")
        if self.chart_type not in {"grouped_bar", "line"}:
            raise ValueError("eligible scene chart type is not registered")
        if self.operation not in {"sum", "difference", "max_minus_min"}:
            raise ValueError("eligible scene operation is not registered")
        for values, label in (
            (self.true_values, "true_values"),
            (self.perceived_values, "perceived_values"),
        ):
            if not isinstance(values, tuple) or len(values) != 4:
                raise ValueError(f"{label} must contain four integers")
            if any(type(value) is not int for value in values):
                raise TypeError(f"{label} must contain exact integers")
        if any(not 2 <= value <= 18 for value in self.true_values):
            raise ValueError("true_values must lie in the registered render domain")
        if sum(a != b for a, b in zip(self.true_values, self.perceived_values, strict=True)) != 1:
            raise ValueError("eligible scene must preserve the frozen one-position error")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "FINAL_PHASE_C_V2_SCREEN_QUOTA_UNDERFILL_DO_NOT_RERUN",
        "server_revision_observed": "0320792",
        "amendment_id": "recoverability-phase-c-v2-20260815",
        "dataset_id": "CVA-Recoverability-Causal-v2",
        "phase_c_screen_exit": 3,
        "scenes": 8000,
        "model_calls": 8000,
        "parse_successes": 7905,
        "parse_rate": 0.988125,
        "natural_perception_errors": 957,
        "one_position_errors": 907,
        "operator_sensitive_errors": 606,
        "design_recoverable_errors": 580,
        "eligible_scenes": 580,
        "eligible_by_family": {"cross_series": 208, "duplicate_encoding": 182, "trend": 190},
        "selected_scenes": 0,
        "selected_by_family": {},
        "screen_passed": False,
        "confirmatory_arm_execution_authorized": False,
        "failure_codes": ["phase_c_screen_family_quota_unmet"],
        "format_retries": 0,
        "allow_sample_extension": False,
        "allow_quota_redistribution": False,
        "training_authorized": False,
        "rl_authorized": False,
        "training_invoked": False,
        "model_snapshot_sha256": (
            "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
        ),
        "server_package_lock_sha256": (
            "5ec95a6110323de556784981fd8c43160de2af27d63a6501479b213867ba2fec"
        ),
        "source_sha256": {
            "preflight": "83c08c1c6afee9e525cdff591cacb8dec6976d47e6a869c4ea902e8a9c81b8fd",
            "attempt_marker": "2aea0218ff056c2bd0386667a8440e7b9b0d073da2b9984b67f3bea780b93eaf",
            "dataset_manifest": "bc57389dc3164b6aeba8d4565aecfaea3fa7ba171b4df4843c8ec86cbee8a19f",
            "dataset_records": "36e09f7e15107057fd1b942875d12259b1f281e0354b87c82ed17f420693c766",
            "screen_report": "4921aa91844844e24626d4ad03517a84099569b390867c4f1cd12503f0ef10a1",
            "screen_records": "f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a",
            "console": "f87d675a2890d9594c5d406819d3f23ed051d740c5528e8062668a61612ea3c8",
        },
    }


def load_phase_c_screen_frozen_result(path: Path) -> PhaseCScreenFrozenResult:
    mapping = load_yaml_mapping(path, label="Phase C screen frozen result")
    expected = _expected_mapping()
    reject_unknown_fields(mapping, set(expected), label="Phase C screen frozen result")
    if dict(mapping) != expected:
        raise ValueError("Phase C screen frozen result differs from server evidence")
    return PhaseCScreenFrozenResult(
        **{
            **mapping,
            "eligible_by_family": tuple(sorted(mapping["eligible_by_family"].items())),
            "selected_by_family": tuple(sorted(mapping["selected_by_family"].items())),
            "failure_codes": tuple(mapping["failure_codes"]),
            "source_sha256": tuple(sorted(mapping["source_sha256"].items())),
        }
    )


def _json_lines(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Phase C screen record must be a JSON object")
            rows.append(row)
    return tuple(rows)


def verify_phase_c_screen_artifacts(
    frozen: PhaseCScreenFrozenResult,
    *,
    preflight: Path,
    attempt_marker: Path,
    dataset_manifest: Path,
    dataset_records: Path,
    screen_report: Path,
    screen_records: Path,
    console_log: Path,
) -> tuple[FrozenEligibleScene, ...]:
    """Hash-bind all artifacts and recover exactly the 580 eligible mediators."""

    if not isinstance(frozen, PhaseCScreenFrozenResult):
        raise TypeError("frozen must be a PhaseCScreenFrozenResult")
    paths = {
        "preflight": preflight,
        "attempt_marker": attempt_marker,
        "dataset_manifest": dataset_manifest,
        "dataset_records": dataset_records,
        "screen_report": screen_report,
        "screen_records": screen_records,
        "console": console_log,
    }
    expected_hashes = dict(frozen.source_sha256)
    if set(paths) != set(expected_hashes):
        raise ValueError("Phase C screen artifact closure differs")
    for label, path in paths.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or _SHA256.fullmatch(expected_hashes[label]) is None
        ):
            raise ValueError(f"Phase C screen {label} artifact is invalid")
        if _sha256(path) != expected_hashes[label]:
            raise ValueError(f"Phase C screen {label} differs from frozen evidence")
    report = json.loads(screen_report.read_text(encoding="utf-8"))
    required = {
        "scenes": frozen.scenes,
        "model_calls": frozen.model_calls,
        "parse_successes": frozen.parse_successes,
        "screen_passed": False,
        "confirmatory_arm_execution_authorized": False,
        "training_invoked": False,
    }
    if not isinstance(report, dict) or any(
        report.get(key) != value for key, value in required.items()
    ):
        raise ValueError("Phase C screen report semantics differ from frozen evidence")
    if not math.isclose(float(report.get("parse_rate", -1.0)), frozen.parse_rate):
        raise ValueError("Phase C screen parse rate differs from frozen evidence")
    dataset = _json_lines(dataset_records)
    screened = _json_lines(screen_records)
    if len(dataset) != frozen.scenes or len(screened) != frozen.scenes:
        raise ValueError("Phase C screen row count differs from frozen evidence")
    dataset_ids = {row.get("scene_id") for row in dataset}
    if len(dataset_ids) != frozen.scenes:
        raise ValueError("Phase C screen dataset identifiers are not unique")
    eligible: list[FrozenEligibleScene] = []
    for row in screened:
        if row.get("scene_id") not in dataset_ids:
            raise ValueError("Phase C screen result references an unknown scene")
        if row.get("eligible") is not True:
            continue
        if not all(
            row.get(field) is True
            for field in (
                "parse_success",
                "natural_perception_error",
                "one_position_error",
                "operator_sensitive",
                "design_recoverability_validated",
            )
        ):
            raise ValueError("Phase C eligible row violates the frozen predicate")
        truth = row.get("values")
        perceived = row.get("perceived_values")
        if not isinstance(truth, list) or not isinstance(perceived, list):
            raise ValueError("Phase C eligible values are invalid")
        eligible.append(
            FrozenEligibleScene(
                scene_id=str(row.get("scene_id")),
                family=str(row.get("family")),
                chart_type=str(row.get("chart_type")),
                operation=str(row.get("operation")),
                true_values=tuple(truth),  # type: ignore[arg-type]
                perceived_values=tuple(perceived),  # type: ignore[arg-type]
            )
        )
    counts = Counter(scene.family for scene in eligible)
    if len(eligible) != frozen.eligible_scenes or counts != Counter(
        dict(frozen.eligible_by_family)
    ):
        raise ValueError("Phase C eligible set differs from frozen evidence")
    return tuple(sorted(eligible, key=lambda item: item.scene_id))
