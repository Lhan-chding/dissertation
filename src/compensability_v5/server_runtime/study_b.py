"""Generic-gate callbacks for the concrete Study-B Qwen runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.model_loader import MODEL_PATH
from compensability_v5.qwen.study_b_metrics import paired_bootstrap_contrasts
from compensability_v5.qwen.study_b_runtime import (
    MODEL_SNAPSHOT_SHA256,
    PILOT_SEED,
    QwenStudyBBackend,
    StudyBError,
    evaluation_rows_from_study_a,
    require_offline_environment,
    run_study_b,
    sha256_file,
    tree_sha256,
    validate_evaluation_rows,
    validate_support_package,
    verify_runtime_package_lock,
)

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_LOCK = ROOT / "configs/v5/server_package_lock.yaml"


def _input_paths(validation: Mapping[str, object]) -> tuple[Path, ...]:
    raw = validation.get("input_sha256")
    if not isinstance(raw, Mapping) or not raw:
        raise StudyBError("validated callback payload has no hash-bound inputs")
    paths: list[Path] = []
    for name, digest in raw.items():
        path = Path(str(name))
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or path.is_symlink()
            or not path.is_file()
        ):
            raise StudyBError("validated callback input mapping is malformed")
        if sha256_file(path) != digest:
            raise StudyBError(f"validated callback input SHA-256 drifted: {path}")
        paths.append(path)
    return tuple(paths)


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_callback_package_lock(expected: object) -> dict[str, object]:
    if not isinstance(expected, str) or sha256_file(PACKAGE_LOCK) != expected:
        raise StudyBError("generic callback package-lock SHA-256 drifted")
    return verify_runtime_package_lock(PACKAGE_LOCK)


def _axis_exact_rate(result: Mapping[str, object], axis: str) -> float:
    metrics = result.get("evaluation_metrics")
    by_axis = metrics.get("by_axis") if isinstance(metrics, Mapping) else None
    block = by_axis.get(axis) if isinstance(by_axis, Mapping) else None
    rate = block.get("exact_world_rate") if isinstance(block, Mapping) else None
    if (
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
    ):
        raise StudyBError(f"completed Study-B result has invalid {axis} rate")
    return float(rate)


def _evaluation_evidence(path: Path) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise StudyBError("completed Study-B evaluation evidence is malformed")
    return rows


def _rebuild_primary_contrasts(
    completed_path: Path, arm_results: Mapping[str, object]
) -> dict[str, object]:
    b2, b3 = arm_results["B2"], arm_results["B3"]
    assert isinstance(b2, Mapping) and isinstance(b3, Mapping)
    axes = (
        "iid",
        "variable_permutation",
        "error_position",
        "fact_order",
        "constraint_graph",
        "structural_ood",
    )
    rates = {
        f"{axis}_exact_world_rate": _axis_exact_rate(b3, axis) - _axis_exact_rate(b2, axis)
        for axis in axes
    }
    arms_root = completed_path.parent / "arms"
    paired = paired_bootstrap_contrasts(
        _evaluation_evidence(arms_root / "B2" / "evaluation_rows.jsonl"),
        _evaluation_evidence(arms_root / "B3" / "evaluation_rows.jsonl"),
    )
    return {"B3_minus_B2": rates, "paired_inference": paired}


def _discover_packages(
    validation: Mapping[str, object],
) -> tuple[Mapping[str, object], tuple[dict[str, object], ...]]:
    support: Mapping[str, object] | None = None
    evaluation: tuple[dict[str, object], ...] | None = None
    errors: list[str] = []
    for path in _input_paths(validation):
        candidates: object
        try:
            if path.suffix == ".jsonl":
                candidates = tuple(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            else:
                candidates = _json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: {error}")
            continue
        if isinstance(candidates, Mapping):
            try:
                validate_support_package(candidates)
            except (TypeError, ValueError):
                row_candidate = candidates.get("rows")
                if row_candidate is not None:
                    candidates = row_candidate
            else:
                if support is not None:
                    raise StudyBError("multiple frozen support packages were supplied")
                support = candidates
                continue
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            try:
                validated_rows = validate_evaluation_rows(candidates)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                try:
                    validated_rows = evaluation_rows_from_study_a(  # type: ignore[arg-type]
                        candidates
                    )
                except (TypeError, ValueError):
                    continue
            if evaluation is not None:
                raise StudyBError("multiple frozen Study-B evaluation packages were supplied")
            evaluation = validated_rows
    if support is None or evaluation is None:
        detail = f"; unreadable={errors}" if errors else ""
        raise StudyBError(
            "Study-B callback requires one support package and one five-axis evaluation package"
            + detail
        )
    return support, evaluation


def run_budget_matched_lora(validation: Mapping[str, object]) -> dict[str, object]:
    """Run B0--B3 through the common 06 gate after all generic checks pass."""

    if validation.get("phase") != "phase4_budget_matched_lora":
        raise StudyBError("run_budget_matched_lora received the wrong validated phase")
    output_value = validation.get("output")
    if not isinstance(output_value, str) or not output_value:
        raise StudyBError("validated callback output is missing")
    support, evaluation = _discover_packages(validation)
    require_offline_environment()
    _verify_callback_package_lock(validation.get("package_lock_sha256"))
    backend = QwenStudyBBackend(model_path=Path(MODEL_PATH), max_sequence_length=512)
    return run_study_b(
        support_package=support,
        evaluation_rows=evaluation,
        output=Path(output_value),
        backend=backend,
        expected_model_sha256=MODEL_SNAPSHOT_SHA256,
        seed=PILOT_SEED,
        provenance={
            "generic_gate": "scripts/v5/06_train_budget_matched_lora.py",
            "config_sha256": validation.get("config_sha256"),
            "package_lock_sha256": validation.get("package_lock_sha256"),
            "input_sha256": dict(validation["input_sha256"]),  # type: ignore[arg-type]
        },
    )


def run_orbit_support(
    validation: Mapping[str, object], request: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Immutably export the already-unified Study-B structural evaluation."""

    if validation.get("phase") != "phase5_structural_support_evaluation":
        raise StudyBError("run_orbit_support received the wrong validated phase")
    completed: dict[str, object] | None = None
    completed_path: Path | None = None
    for path in _input_paths(validation):
        try:
            value = _json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("status") == "STUDY_B_SINGLE_SEED_COMPLETE":
            if completed is not None:
                raise StudyBError("multiple Study-B completed results were supplied")
            completed = value
            completed_path = path
    if completed is None or completed_path is None:
        raise StudyBError("orbit-support callback requires hash-bound Study-B completed.json")
    arm_results = completed.get("arm_results")
    contrasts = completed.get("primary_contrasts")
    if not isinstance(arm_results, Mapping) or set(arm_results) != {"B0", "B1", "B2", "B3"}:
        raise StudyBError("completed Study-B result lacks all four arm evaluations")
    contrast = contrasts.get("B3_minus_B2") if isinstance(contrasts, Mapping) else None
    expected_contrast_fields = {
        "iid_exact_world_rate",
        "variable_permutation_exact_world_rate",
        "error_position_exact_world_rate",
        "fact_order_exact_world_rate",
        "constraint_graph_exact_world_rate",
        "structural_ood_exact_world_rate",
    }
    if not isinstance(contrast, Mapping) or set(contrast) != expected_contrast_fields:
        raise StudyBError("completed Study-B result lacks the B3-minus-B2 contrast")
    paired = contrasts.get("paired_inference") if isinstance(contrasts, Mapping) else None
    if not isinstance(paired, Mapping) or set(paired) != {
        "bootstrap",
        "relational_constraint_graph",
        "structural_ood",
        "stop_signal",
    }:
        raise StudyBError("completed Study-B result lacks paired inference")
    if paired.get("bootstrap") != {
        "method": "paired_scene_cluster_percentile",
        "seed": 2026082202,
        "resamples": 10_000,
        "confidence_level": 0.95,
    }:
        raise StudyBError("completed Study-B paired-bootstrap registration drifted")
    ci_lowers: list[float] = []
    for section_name in ("relational_constraint_graph", "structural_ood"):
        section = paired.get(section_name)
        if not isinstance(section, Mapping) or set(section) != {
            "exact_world",
            "genuine_recovery",
        }:
            raise StudyBError(f"completed Study-B {section_name} inference is malformed")
        for metric in section.values():
            if not isinstance(metric, Mapping) or set(metric) != {
                "semantic_scene_count",
                "delta",
                "ci95",
            }:
                raise StudyBError("completed Study-B paired metric is malformed")
            count, delta, interval = (
                metric["semantic_scene_count"],
                metric["delta"],
                metric["ci95"],
            )
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                or not isinstance(delta, (int, float))
                or isinstance(delta, bool)
                or not math.isfinite(float(delta))
                or not isinstance(interval, list)
                or len(interval) != 2
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in interval
                )
                or float(interval[0]) > float(interval[1])
            ):
                raise StudyBError("completed Study-B paired metric values are malformed")
            ci_lowers.append(float(interval[0]))
    stop_signal = paired.get("stop_signal")
    if (
        not isinstance(stop_signal, Mapping)
        or stop_signal.get("rule") != "B3_minus_B2_paired_CI95_lower_gt_zero"
        or not isinstance(stop_signal.get("triggered"), bool)
        or stop_signal.get("triggered") != (max(ci_lowers) > 0.0)
        or completed.get("stop_signal") != stop_signal
    ):
        raise StudyBError("completed Study-B stop signal drifted")
    for arm in ("B0", "B1", "B2", "B3"):
        arm_result = arm_results[arm]
        if not isinstance(arm_result, Mapping):
            raise StudyBError(f"completed Study-B {arm} result is malformed")
        metrics = arm_result.get("evaluation_metrics")
        by_axis = metrics.get("by_axis") if isinstance(metrics, Mapping) else None
        if not isinstance(by_axis, Mapping) or not {
            "iid",
            "variable_permutation",
            "error_position",
            "fact_order",
            "constraint_graph",
            "structural_ood",
        }.issubset(by_axis):
            raise StudyBError(f"completed Study-B {arm} lacks registered axis metrics")
        arm_root = completed_path.parent / "arms" / arm
        if tree_sha256(arm_root / "final_adapter") != arm_result.get("adapter_tree_sha256"):
            raise StudyBError(f"completed Study-B {arm} adapter tree drifted")
        if sha256_file(arm_root / "evaluation_rows.jsonl") != arm_result.get(
            "evaluation_rows_sha256"
        ):
            raise StudyBError(f"completed Study-B {arm} evaluation rows drifted")
    rebuilt_contrasts = _rebuild_primary_contrasts(completed_path, arm_results)
    if contrasts != rebuilt_contrasts:
        raise StudyBError("completed Study-B primary contrasts drifted")
    output_value = validation.get("output")
    if not isinstance(output_value, str) or not output_value:
        raise StudyBError("validated callback output is missing")
    output = Path(output_value)
    if output.exists() or output.is_symlink():
        raise FileExistsError("orbit-support output already exists")
    result = {
        "schema_version": 1,
        "status": "STUDY_B_ORBIT_SUPPORT_EXPORTED",
        "evidence_class": completed.get("evidence_class"),
        "seed": completed.get("seed"),
        "model_snapshot_sha256": completed.get("model_snapshot_sha256"),
        "run_signature": completed.get("run_signature"),
        "k": (request or {}).get("k"),
        "prompt_protocol": "unified_text_world_v1",
        "arm_evaluation_metrics": {
            arm: arm_results[arm]["evaluation_metrics"] for arm in ("B0", "B1", "B2", "B3")
        },
        "primary_contrasts": rebuilt_contrasts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


__all__ = ["run_budget_matched_lora", "run_orbit_support"]
