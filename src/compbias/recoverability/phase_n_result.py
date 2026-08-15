"""Immutable anchor for the completed, originally inconclusive Phase N."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping
from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PhaseNFrozenResult:
    schema_version: int
    status: str
    server_revision_observed: str
    phase_n_exit: int
    dataset_id: str
    model_snapshot_sha256: str
    server_package_lock_sha256: str
    measurement_qualification_report_sha256: str
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
    reason_code: str
    error_counts: tuple[tuple[str, int], ...]
    format_retries: int
    allow_sample_extension: bool
    training_invoked: bool
    source_sha256: tuple[tuple[str, str], ...]


def _expected() -> PhaseNFrozenResult:
    return PhaseNFrozenResult(
        schema_version=1,
        status="FINAL_INCONCLUSIVE_PHASE_N_DO_NOT_RERUN",
        server_revision_observed="2c0ad9f609820b9ee34cb6ca7bff45a1528ce17f",
        phase_n_exit=3,
        dataset_id="CVA-Natural-Prevalence-v1",
        model_snapshot_sha256="e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
        server_package_lock_sha256="c884f38a2a62654c29818bfbcf669aa9c57cd9291135bcc44b91dca8a60e5692",
        measurement_qualification_report_sha256="11a2d7f44d7fdf115954baacbac1b6c53269a8e00368c2b78f27ba53c24c640e",
        scenes=4000,
        model_calls=4000,
        parsed_scenes=3784,
        operator_sensitive_errors=836,
        strict_natural_repair_candidates=33,
        parse_rate=0.946,
        primary_rate=0.039473684210526314,
        parsed_prevalence=0.00872093023255814,
        all_attempt_prevalence=0.00825,
        one_sided_cp_upper=0.05242826275410656,
        parse_failure_sensitivity_lower=0.03136882129277566,
        parse_failure_sensitivity_upper=0.23669201520912547,
        primary_denominator="parsed_operator_sensitive_errors",
        h1_supported=False,
        inconclusive=True,
        reason_code="phase_n_h1_upper_not_below_threshold",
        error_counts=(
            ("compensated_visual_error", 33),
            ("none", 1433),
            ("operator_invariant_visual_error", 77),
            ("parse_failure", 216),
            ("reasoning_error", 1344),
            ("visual_error", 897),
        ),
        format_retries=0,
        allow_sample_extension=False,
        training_invoked=False,
        source_sha256=(
            ("attempt_marker", "73b12419ada8538987b87f73b2dc4604f0d84ee1b58cd409f38b59bcdf2198dd"),
            ("console", "d426f02055c43658615835c350b8b4c2ddc873cb92e60ba5922f1f695535fd2f"),
            (
                "dataset_manifest",
                "a0a032eb87cc2be2875e9095e667bce6993f24cdf413b62234ea713e19f537ce",
            ),
            ("dataset_records", "429d1da23995bf14a4779fd8bac243a6423ed3abb91c91c160eb2c7b06247a43"),
            ("phase_n_records", "431f95ce5336eb6f72413ab108854b9d693b5f3d95e501e84dad0d5139f43134"),
            ("phase_n_report", "733da0124d9027c0ac6363836a85be1bc74b8e9bf0c6a0476e0219f80fdf1399"),
            ("preflight", "6f3ca8bbfb11e2c84821da7fe0f8399fb281b91d9c3070f76158b597f929afe0"),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase_n_frozen_result(path: Path) -> PhaseNFrozenResult:
    """Load the exact Phase-N outcome without changing its failed original gate."""

    mapping = load_yaml_mapping(path, label="Phase N frozen result")
    fields = set(PhaseNFrozenResult.__dataclass_fields__)
    reject_unknown_fields(mapping, fields, label="Phase N frozen result")
    if set(mapping) != fields:
        raise ValueError("Phase N frozen result is incomplete")
    source = mapping["source_sha256"]
    counts = mapping["error_counts"]
    if not isinstance(source, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for key, value in source.items()
    ):
        raise ValueError("Phase N source digests are invalid")
    if not isinstance(counts, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in counts.items()
    ):
        raise ValueError("Phase N error counts are invalid")
    candidate = PhaseNFrozenResult(
        **{
            **mapping,
            "error_counts": tuple(sorted(counts.items())),
            "source_sha256": tuple(sorted(source.items())),
        }
    )
    if candidate != _expected():
        raise ValueError("Phase N frozen result differs from server evidence")
    return candidate


def verify_phase_n_result_artifacts(
    frozen: PhaseNFrozenResult,
    *,
    preflight: Path,
    attempt_marker: Path,
    dataset_manifest: Path,
    dataset_records: Path,
    report: Path,
    records: Path,
    console_log: Path,
) -> PhaseNFrozenResult:
    """Bind all seven server artifacts before Phase-C model loading."""

    if frozen != _expected():
        raise ValueError("Phase N frozen result is not canonical")
    paths = {
        "preflight": preflight,
        "attempt_marker": attempt_marker,
        "dataset_manifest": dataset_manifest,
        "dataset_records": dataset_records,
        "phase_n_report": report,
        "phase_n_records": records,
        "console": console_log,
    }
    digests = dict(frozen.source_sha256)
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != digests[label]:
            raise ValueError(f"Phase N {label} is missing or differs from frozen evidence")
    payload = load_strict_json_mapping(report, label="Phase N report", max_bytes=128 * 1024)
    expected_report = asdict(frozen)
    for key in ("status", "server_revision_observed", "phase_n_exit", "source_sha256"):
        expected_report.pop(key)
    expected_report.pop("error_counts")
    for key, value in expected_report.items():
        if payload.get(key) != value:
            raise ValueError(f"Phase N report mismatch for {key}")
    if payload.get("error_counts") != dict(frozen.error_counts):
        raise ValueError("Phase N report error counts differ")
    marker = load_strict_json_mapping(attempt_marker, label="Phase N attempt", max_bytes=32 * 1024)
    if marker.get("status") != "PHASE_N_STARTED_DO_NOT_RERUN":
        raise ValueError("Phase N attempt marker status is invalid")
    try:
        console = console_log.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Phase N console must be UTF-8") from error
    if console.count("phase_n_exit=3") < 1:
        raise ValueError("Phase N console lacks exit=3 evidence")
    count = 0
    with records.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                if not isinstance(json.loads(line), dict):
                    raise ValueError("Phase N record must be an object")
                count += 1
    if count != frozen.scenes:
        raise ValueError("Phase N record count differs")
    return frozen
