"""Measure Base/C0/C1/T policy support on frozen held-out natural errors."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _guards import ROOT, sha256  # noqa: E402

from compensability_v4.qwen.model_loader import (  # noqa: E402
    MODEL_PATH,
    MODEL_SNAPSHOT_SHA256,
    load_pinned_qwen,
)
from compensability_v4.qwen.phase5_runtime import (  # noqa: E402
    checkpoint_adapter_path,
    checkpoint_tree_hashes,
    freeze_inference_model,
    load_phase5_config,
    measure_checkpoint,
    verify_phase5_package_lock,
)
from compensability_v4.qwen.phase5_support import (  # noqa: E402
    CheckpointSceneMeasurement,
    HeldOutNaturalError,
    PolicyCheckpoint,
    summarize_phase5_policy_support,
    write_phase5_outputs,
)

CONFIG = ROOT / "configs/recoverability/v4_phase_5.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_5.yaml"
SUPPORT_DEV_ROOT = ROOT / "artifacts/v4/support_dev"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
CACHE_ROOT = ROOT / "artifacts/v4/support_work/phase5-r1"
_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_5.yaml",
    "configs/recoverability/v4/phase_1_3_prompts.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/09_prepare_phase5_support_dev.py",
    "scripts/v4/10_measure_policy_support.py",
    "src/compensability_v4/qwen/phase5_runtime.py",
    "src/compensability_v4/qwen/phase5_support.py",
)


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 5 {label} must be a regular JSON file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Phase 5 {label} must contain one JSON object")
    return payload


def _jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 5 {label} must be a regular JSONL file")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 5 {label} is empty or malformed")
    return rows  # type: ignore[return-value]


def _load_support_dev(root: Path) -> tuple[tuple[HeldOutNaturalError, ...], str, str]:
    errors_path, summary_path = root / "held_out_natural_errors.jsonl", root / "summary.json"
    summary = _json(summary_path, "support-dev summary")
    error_hash = sha256(errors_path)
    rows = _jsonl(errors_path, "held-out natural errors")
    if (
        summary.get("status") != "PHASE_5_SUPPORT_DEV_FROZEN"
        or summary.get("held_out_natural_error_count") != len(rows)
        or summary.get("held_out_natural_errors_sha256") != error_hash
        or summary.get("selection_uses_model_outcome_threshold") is not False
        or summary.get("confirmatory_data_used") is not False
        or summary.get("training_invoked") is not False
        or summary.get("rl_invoked") is not False
    ):
        raise RuntimeError("Phase 5 support-dev summary/provenance is malformed")
    errors = tuple(HeldOutNaturalError.from_mapping(row) for row in rows)
    if not errors or len({error.scene_id for error in errors}) != len(errors):
        raise RuntimeError("Phase 5 held-out natural-error pool is empty or duplicated")
    return errors, error_hash, sha256(summary_path)


def _cache_paths(root: Path, checkpoint: PolicyCheckpoint) -> tuple[Path, Path]:
    return root / f"{checkpoint.value}.jsonl", root / f"{checkpoint.value}.meta.json"


def _load_cache(
    root: Path,
    checkpoint: PolicyCheckpoint,
    *,
    checkpoint_sha256: str,
    support_dev_sha256: str,
    config_sha256: str,
) -> tuple[CheckpointSceneMeasurement, ...] | None:
    rows_path, meta_path = _cache_paths(root, checkpoint)
    if not rows_path.exists() and not meta_path.exists():
        return None
    if (
        rows_path.is_symlink()
        or meta_path.is_symlink()
        or not rows_path.is_file()
        or not meta_path.is_file()
    ):
        raise RuntimeError(f"Phase 5 {checkpoint.value} resume cache is incomplete or unsafe")
    meta = _json(meta_path, f"{checkpoint.value} resume metadata")
    rows = _jsonl(rows_path, f"{checkpoint.value} resume rows")
    if meta != {
        "schema_version": 1,
        "status": "PHASE_5_CHECKPOINT_MEASUREMENT_COMPLETE",
        "checkpoint": checkpoint.value,
        "checkpoint_sha256": checkpoint_sha256,
        "support_dev_sha256": support_dev_sha256,
        "config_sha256": config_sha256,
        "row_count": len(rows),
        "rows_sha256": sha256(rows_path),
        "training_invoked": False,
        "rl_invoked": False,
    }:
        raise RuntimeError(f"Phase 5 {checkpoint.value} resume cache provenance drifted")
    return tuple(CheckpointSceneMeasurement.from_mapping(row) for row in rows)


def _write_cache(
    root: Path,
    checkpoint: PolicyCheckpoint,
    rows: tuple[CheckpointSceneMeasurement, ...],
    *,
    checkpoint_sha256: str,
    support_dev_sha256: str,
    config_sha256: str,
) -> None:
    rows_path, meta_path = _cache_paths(root, checkpoint)
    if rows_path.exists() or rows_path.is_symlink() or meta_path.exists() or meta_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase 5 {checkpoint.value} resume cache")
    root.mkdir(parents=True, exist_ok=True)
    temporary = rows_path.with_suffix(".jsonl.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("Phase 5 temporary resume cache already exists")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_mapping(), sort_keys=True, allow_nan=False) + "\n")
    temporary.rename(rows_path)
    meta = {
        "schema_version": 1,
        "status": "PHASE_5_CHECKPOINT_MEASUREMENT_COMPLETE",
        "checkpoint": checkpoint.value,
        "checkpoint_sha256": checkpoint_sha256,
        "support_dev_sha256": support_dev_sha256,
        "config_sha256": config_sha256,
        "row_count": len(rows),
        "rows_sha256": sha256(rows_path),
        "training_invoked": False,
        "rl_invoked": False,
    }
    with meta_path.open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _load_checkpoint(checkpoint: PolicyCheckpoint, *, run_root: Path) -> tuple[object, object]:
    base, processor = load_pinned_qwen(model_path=Path(MODEL_PATH))
    adapter_path = checkpoint_adapter_path(run_root, checkpoint)
    model = base
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    freeze_inference_model(model)
    return model, processor


def _release(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _preflight(output_paths: tuple[Path, ...]) -> None:
    missing = [
        package
        for package in ("torch", "transformers", "peft", "pyarrow")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise RuntimeError("Phase 5 missing required packages: " + ", ".join(missing))
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 5 requires CUDA with bf16 support")
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise FileExistsError("refusing to overwrite Phase 5 policy-support artifacts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--support-dev-root", type=Path, default=SUPPORT_DEV_ROOT)
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 5 policy-support measurement requires explicit --execute.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        payload, config = load_phase5_config(arguments.config)
        lock_hash = verify_phase5_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=_LOCKED_PATHS,
        )
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, dict)
        output_paths = (
            ROOT / str(artifacts["policy_support_by_scene"]),
            ROOT / str(artifacts["informative_group_rate"]),
            ROOT / str(artifacts["pass_at_k"]),
        )
        _preflight(output_paths)
        errors, support_dev_hash, support_summary_hash = _load_support_dev(
            arguments.support_dev_root
        )
        adapter_hashes = checkpoint_tree_hashes(arguments.phase4_run_root)
        checkpoint_hashes = {"Base": MODEL_SNAPSHOT_SHA256, **adapter_hashes}
        config_hash = sha256(arguments.config)
        measurements: list[CheckpointSceneMeasurement] = []
        for checkpoint in PolicyCheckpoint:
            checkpoint_hash = checkpoint_hashes[checkpoint.value]
            cached = _load_cache(
                arguments.cache_root,
                checkpoint,
                checkpoint_sha256=checkpoint_hash,
                support_dev_sha256=support_dev_hash,
                config_sha256=config_hash,
            )
            if cached is not None:
                if {row.scene_id for row in cached} != {error.scene_id for error in errors}:
                    raise RuntimeError(f"Phase 5 {checkpoint.value} resume scene closure drifted")
                rows = cached
                print(f"RESUMED: Phase 5 {checkpoint.value} checkpoint evidence", flush=True)
            else:
                model, processor = _load_checkpoint(checkpoint, run_root=arguments.phase4_run_root)
                rows = measure_checkpoint(
                    model=model,
                    processor=processor,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_hash,
                    errors=errors,
                    config=config,
                    progress=lambda complete, total, name=checkpoint.value: (
                        print(
                            f"PROGRESS: {name} {complete}/{total} scenes complete",
                            flush=True,
                        )
                        if complete % 10 == 0 or complete == total
                        else None
                    ),
                )
                _release(model)
                _write_cache(
                    arguments.cache_root,
                    checkpoint,
                    rows,
                    checkpoint_sha256=checkpoint_hash,
                    support_dev_sha256=support_dev_hash,
                    config_sha256=config_hash,
                )
            measurements.extend(rows)
        summary = summarize_phase5_policy_support(
            errors=errors,
            measurements=measurements,
            pass_at_k=config.pass_at_k,
            informative_group_size=config.informative_group_size,
            sampling_temperature=config.temperature,
            sampling_seed=config.sampling_seed,
        )
        summary.update(
            config_sha256=config_hash,
            package_lock_sha256=lock_hash,
            model_snapshot_sha256=MODEL_SNAPSHOT_SHA256,
            support_dev_summary_sha256=support_summary_hash,
            adapter_tree_sha256=adapter_hashes,
        )
        write_phase5_outputs(
            parquet_path=output_paths[0],
            informative_path=output_paths[1],
            pass_at_k_path=output_paths[2],
            measurements=measurements,
            summary=summary,
            source_sha256={"support_dev": support_dev_hash, **checkpoint_hashes},
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 5 policy-support outputs written under {output_paths[0].parent}")
    for path in output_paths:
        print(f"SHA256 {sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
