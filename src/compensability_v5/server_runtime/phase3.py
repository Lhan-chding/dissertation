"""Server callbacks for the calibrated no-training v5 decisive pilot.

The orbit callback executes the unified Study A runtime. Gradient alignment is
outside the calibrated 4090 pilot and therefore fails closed instead of
publishing a placeholder that could be mistaken for experimental evidence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from compensability_v5.qwen.study_a_runtime import (
    BASE_SHA256,
    RAW_ARCHIVE_SHA256,
    STUDY_A_ACK,
    T_ADAPTER_SHA256,
    load_gpu_checkpoint,
    load_natural_errors,
    require_study_a_authorization,
    require_t_adapter,
    run_study_a,
    sha256_file,
)

DEFERRED_STATUS = "DEFERRED_NOT_REQUIRED_BY_4090_DECISIVE_PILOT"
T_ADAPTER_ENV = "COMPBIAS_V5_T_ADAPTER"


def _validated_callback(
    validation: Mapping[str, object], options: Mapping[str, object], *, phase: str, task: str
) -> tuple[Path, Mapping[str, object]]:
    if (
        not isinstance(validation, Mapping)
        or validation.get("schema_version") != 1
        or validation.get("phase") != phase
        or not isinstance(options, Mapping)
        or options.get("task") != task
    ):
        raise RuntimeError(f"{task} callback received incompatible validated execution")
    output_value = validation.get("output")
    inputs = validation.get("input_sha256")
    if not isinstance(output_value, str) or not output_value or not isinstance(inputs, Mapping):
        raise RuntimeError(f"{task} callback validation mapping is malformed")
    output = Path(output_value)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{task} callback refuses to overwrite {output}")
    return output, inputs


def _raw_archive(inputs: Mapping[str, object]) -> Path:
    matches = [
        Path(path)
        for path, digest in inputs.items()
        if isinstance(path, str) and digest == RAW_ARCHIVE_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError("orbit callback requires exactly one frozen v4 raw archive input")
    return matches[0]


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"orbit callback temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(path)
    except Exception:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _completed_summary(result_root: Path, raw_digest: str) -> dict[str, object] | None:
    if not result_root.exists():
        return None
    summary_path, manifest_path = result_root / "summary.json", result_root / "manifest.json"
    if (
        result_root.is_symlink()
        or not result_root.is_dir()
        or summary_path.is_symlink()
        or manifest_path.is_symlink()
        or not summary_path.is_file()
        or not manifest_path.is_file()
    ):
        raise RuntimeError("orbit callback found an incomplete Study A publication")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sources = {
        "raw_archive": raw_digest,
        "Base": BASE_SHA256,
        "T": T_ADAPTER_SHA256,
    }
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "V5_STUDY_A_EXECUTED"
        or summary.get("source_sha256") != expected_sources
        or not isinstance(manifest, dict)
        or manifest.get("status") != "V5_STUDY_A_ATOMICALLY_PUBLISHED"
        or manifest.get("source_sha256") != expected_sources
    ):
        raise RuntimeError("orbit callback found incompatible completed Study A evidence")
    return summary


def run_orbit_audit(
    validation: Mapping[str, object], options: Mapping[str, object]
) -> dict[str, object]:
    """Execute Base/T Study A and publish the generic gate's JSON pointer."""

    output, inputs = _validated_callback(
        validation,
        options,
        phase="phase3_orbit_audit",
        task="orbit_audit",
    )
    if options.get("k") != 8:
        raise RuntimeError("orbit callback requires the frozen K=8 sampling contract")
    adapter_value = os.environ.get(T_ADAPTER_ENV)
    if not adapter_value:
        raise RuntimeError(f"orbit callback requires {T_ADAPTER_ENV}")
    adapter = require_t_adapter(Path(adapter_value))
    raw_archive = _raw_archive(inputs)
    errors, raw_digest = load_natural_errors(raw_archive)
    require_study_a_authorization(execute=True, acknowledgement=STUDY_A_ACK)
    result_root = output.parent / f"{output.stem}_study_a"
    work_root = output.parent / f".{output.stem}_study_a_work"
    summary = _completed_summary(result_root, raw_digest)
    if summary is None:
        summary = run_study_a(
            errors=errors,
            raw_archive_sha256=raw_digest,
            output_root=result_root,
            work_root=work_root,
            checkpoint_loader=lambda checkpoint: load_gpu_checkpoint(
                checkpoint,
                t_adapter=adapter,
            ),
            k=8,
            sampling_seed=2026082101,
            progress=lambda checkpoint, complete, total: print(
                f"PROGRESS: {checkpoint} {complete}/{total} checkpoint-scenarios complete",
                flush=True,
            ),
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "V5_STUDY_A_ORBIT_CALLBACK_COMPLETE",
        "study_a_status": summary.get("status"),
        "study_a_result_root": str(result_root),
        "study_a_summary_sha256": sha256_file(result_root / "summary.json"),
        "source_sha256": summary.get("source_sha256"),
        "semantic_scene_count": summary.get("semantic_scene_count"),
        "scenario_checkpoint_count": summary.get("scenario_checkpoint_count"),
        "by_checkpoint": summary.get("by_checkpoint"),
        "by_family": summary.get("by_family"),
        "by_graph_axis": summary.get("by_graph_axis"),
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
    }
    _atomic_json(output, payload)
    return payload


def run_gradient_alignment(
    validation: Mapping[str, object], options: Mapping[str, object]
) -> dict[str, object]:
    """Fail closed: gradient alignment was removed from this decisive pilot."""

    _validated_callback(
        validation,
        options,
        phase="phase3_gradient_alignment",
        task="gradient_alignment",
    )
    raise RuntimeError(DEFERRED_STATUS)


__all__ = [
    "DEFERRED_STATUS",
    "T_ADAPTER_ENV",
    "run_gradient_alignment",
    "run_orbit_audit",
]
