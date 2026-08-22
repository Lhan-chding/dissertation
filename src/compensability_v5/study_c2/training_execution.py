"""Artifact-producing half of the Study C2 Stage 25 training runtime."""

from __future__ import annotations

import gc
import math
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from compensability_v5.qwen.study_b_runtime import tree_sha256

from .io import read_json, read_jsonl, sha256_file, write_json_new, write_jsonl_new
from .paths import (
    FIBER_ROWS,
    STAGE25_EXECUTION_CONTRACT,
    TRAINING_PAIR_MANIFEST,
    TRAINING_ROOT,
)
from .training_runtime import (
    _CHECKPOINT_STEPS,
    _EXPECTED_PROMPTS,
    _KINDS,
    TRAINING_ACK,
    TrainingProgressCallback,
    build_traced_reward,
    build_training_group_diagnostics,
    preflight_training_arm,
    select_training_rows,
)


class TrainerLike(Protocol):
    state: object

    def train(self, *, resume_from_checkpoint: str | None = None) -> object: ...

    def save_model(self, output_dir: str) -> object: ...


TrainerFactory = Callable[..., TrainerLike]


def _restore_trace(trace_path: Path, checkpoint: Path, output_dir: Path) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Study C2 resume checkpoint is missing or unsafe")
    try:
        checkpoint.resolve().relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("Study C2 resume checkpoint must be inside its arm output") from error
    snapshot = checkpoint / "raw_reward_trace.jsonl"
    if snapshot.is_symlink() or not snapshot.is_file():
        raise ValueError("Study C2 checkpoint lacks its reward-trace snapshot")
    temporary = trace_path.with_suffix(".restore.tmp")
    shutil.copyfile(snapshot, temporary)
    temporary.replace(trace_path)


def _write_or_validate_arm_config(
    path: Path, arm_config: Mapping[str, object], *, resume: bool
) -> None:
    if not resume:
        write_json_new(path, arm_config)
        return
    if read_json(path) != dict(arm_config):
        raise ValueError("Study C2 resume arm config differs from the frozen config")


def _pair_manifest_if_complete() -> dict[str, object] | None:
    arm_manifests = {
        name: TRAINING_ROOT / name / "manifest.json"
        for name in ("C2_answer_reward", "C2_exact_state_reward")
    }
    if not all(path.is_file() and not path.is_symlink() for path in arm_manifests.values()):
        return None
    payloads = {name: read_json(path) for name, path in arm_manifests.items()}
    if any(value.get("status") != "STUDY_C2_ARM_TRAINING_COMPLETE" for value in payloads.values()):
        raise ValueError("Study C2 pair contains an incomplete arm manifest")
    result: dict[str, object] = {
        "schema_version": 2,
        "status": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE",
        "arms": {
            name: {
                "manifest_sha256": sha256_file(arm_manifests[name]),
                "final_adapter_sha256": payloads[name]["final_adapter_sha256"],
                "raw_reward_trace_sha256": payloads[name]["raw_reward_trace_sha256"],
            }
            for name in sorted(arm_manifests)
        },
        "reward_only_pair_verified": True,
        "training_prompt_count_per_arm": _EXPECTED_PROMPTS,
        "optimizer_steps_per_arm": _EXPECTED_PROMPTS,
        "training_invoked": True,
        "rl_invoked": True,
        "gpu_invoked": True,
    }
    if TRAINING_PAIR_MANIFEST.exists():
        if read_json(TRAINING_PAIR_MANIFEST) != result:
            raise ValueError("existing Study C2 pair manifest drifted")
    else:
        write_json_new(TRAINING_PAIR_MANIFEST, result)
    return result


def run_training_arm(
    *,
    arm: str,
    config_path: Path,
    execution_contract_path: Path = STAGE25_EXECUTION_CONTRACT,
    b3_adapter: Path,
    b3_sha256: str,
    acknowledgement: str,
    resume_from_checkpoint: Path | None = None,
    trainer_factory: TrainerFactory | None = None,
    backend_validator: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if acknowledgement != TRAINING_ACK:
        raise PermissionError("exact Study C2 Stage 25 training acknowledgement is required")
    preflight = preflight_training_arm(
        arm=arm,
        config_path=config_path,
        execution_contract_path=execution_contract_path,
        b3_adapter=b3_adapter,
        b3_sha256=b3_sha256,
        backend_validator=backend_validator,
    )
    arm_config = preflight["arm_config"]
    if not isinstance(arm_config, Mapping):
        raise RuntimeError("Study C2 preflight returned no arm config")
    output_dir = Path(str(arm_config["output_directory"]))
    manifest_path = output_dir / "manifest.json"
    trace_path = output_dir / "raw_reward_trace.jsonl"
    resume = resume_from_checkpoint is not None
    if manifest_path.exists():
        raise RuntimeError(f"completed Study C2 arm exists; overwrite forbidden: {output_dir}")
    if resume:
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError("Study C2 resume output is missing or unsafe")
        assert resume_from_checkpoint is not None
        _restore_trace(trace_path, resume_from_checkpoint, output_dir)
    elif output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"fresh Study C2 arm output exists; overwrite forbidden: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    _write_or_validate_arm_config(output_dir / "arm_config.json", arm_config, resume=resume)
    training_rows = select_training_rows(read_jsonl(FIBER_ROWS))
    group_size = int(preflight["group_size"])
    total_steps = int(preflight["expected_optimizer_steps"])
    reward = build_traced_reward(
        arm_config=arm_config,
        training_rows=training_rows,
        trace_path=trace_path,
        group_size=group_size,
    )
    callback = TrainingProgressCallback(
        str(arm_config["name"]),
        total_steps=total_steps,
        trace_path=trace_path,
        output_dir=output_dir,
    )
    if trainer_factory is None:
        from .training_backend import create_training_trainer

        trainer_factory = create_training_trainer
    dataset = tuple(
        {
            "prompt": [{"role": "user", "content": str(row["prompt"])}],
            "scene_id": row["scene_id"],
        }
        for row in training_rows
    )
    try:
        print(
            f"PROGRESS: loading B3 policy for {arm_config['name']} with frozen B3 KL reference",
            flush=True,
        )
        trainer = trainer_factory(
            arm_config=arm_config,
            b3_adapter=b3_adapter,
            dataset=dataset,
            reward_function=reward,
            output_dir=output_dir,
            group_size=group_size,
            callbacks=(callback,),
        )
        print(
            f"PROGRESS: {arm_config['name']} training 192 prompts for one epoch",
            flush=True,
        )
        trainer.train(
            resume_from_checkpoint=(
                None if resume_from_checkpoint is None else str(resume_from_checkpoint)
            )
        )
        observed_steps = getattr(getattr(trainer, "state", None), "global_step", None)
        if observed_steps != total_steps:
            raise RuntimeError(
                "Study C2 trainer executed "
                f"{observed_steps} optimizer steps, expected {total_steps}"
            )
        for step in _CHECKPOINT_STEPS:
            checkpoint = output_dir / f"checkpoint-{step}"
            if not checkpoint.is_dir() or not (checkpoint / "raw_reward_trace.jsonl").is_file():
                raise RuntimeError(f"Study C2 trainer did not preserve checkpoint-{step}")
        logs = getattr(getattr(trainer, "state", None), "log_history", None)
        if not isinstance(logs, list) or any(
            any(
                isinstance(value, float) and not math.isfinite(value)
                for value in entry.values()
            )
            for entry in logs
            if isinstance(entry, Mapping)
        ):
            raise RuntimeError("Study C2 trainer log history is missing or non-finite")
        write_json_new(output_dir / "trainer_log_history.json", {"rows": logs})
        final_adapter = output_dir / "final_adapter"
        if final_adapter.exists() or final_adapter.is_symlink():
            raise RuntimeError("Study C2 final adapter exists before the final save")
        trainer.save_model(str(final_adapter))
        final_adapter_sha256 = tree_sha256(final_adapter)
    finally:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    diagnostics = build_training_group_diagnostics(
        trace_path=trace_path,
        group_size=group_size,
        expected_group_count=_EXPECTED_PROMPTS,
    )
    diagnostics_path = output_dir / "group_diagnostics.jsonl"
    write_jsonl_new(diagnostics_path, diagnostics)
    counts = Counter(str(row["kind"]) for row in read_jsonl(trace_path))
    summary: dict[str, object] = {
        "schema_version": 2,
        "status": "STUDY_C2_ARM_TRAINING_SUMMARIZED",
        "arm": arm_config["name"],
        "reward_function_id": arm_config["reward_function_id"],
        "training_prompt_count": _EXPECTED_PROMPTS,
        "group_size": group_size,
        "rollout_count": _EXPECTED_PROMPTS * group_size,
        "optimizer_steps": total_steps,
        "epochs": 1,
        "counts": {kind: counts.get(kind, 0) for kind in _KINDS},
        "reward_hamming_distance": sum(
            int(row["reward_hamming_distance"]) for row in diagnostics
        ),
        "checkpoint_steps": list(_CHECKPOINT_STEPS),
    }
    summary_path = output_dir / "summary.json"
    write_json_new(summary_path, summary)
    manifest: dict[str, object] = {
        **{key: value for key, value in preflight.items() if key != "arm_config"},
        "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
        "arm": arm_config["name"],
        "reward_function_id": arm_config["reward_function_id"],
        "arm_config_sha256": sha256_file(output_dir / "arm_config.json"),
        "raw_reward_trace_sha256": sha256_file(trace_path),
        "group_diagnostics_sha256": sha256_file(diagnostics_path),
        "summary_sha256": sha256_file(summary_path),
        "trainer_log_sha256": sha256_file(output_dir / "trainer_log_history.json"),
        "final_adapter_sha256": final_adapter_sha256,
        "resumed_from_checkpoint": (
            None if resume_from_checkpoint is None else str(resume_from_checkpoint.resolve())
        ),
        "reference_initialization": "frozen_copy_of_B3_adapter",
        "checkpoint_steps": list(_CHECKPOINT_STEPS),
        "training_invoked": True,
        "optimizer_step_invoked": True,
        "rl_invoked": True,
        "gpu_invoked": True,
    }
    write_json_new(manifest_path, manifest)
    pair_manifest = _pair_manifest_if_complete()
    return {
        **manifest,
        "pair_complete": pair_manifest is not None,
        "pair_manifest_sha256": (
            None if pair_manifest is None else sha256_file(TRAINING_PAIR_MANIFEST)
        ),
    }


__all__ = ["run_training_arm"]
