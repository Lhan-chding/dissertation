"""Preflight or execute the three registered Phase 6 GRPO variants."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
from compensability_v4.qwen.phase5_runtime import tree_sha256  # noqa: E402
from compensability_v4.qwen.phase6_runtime import (  # noqa: E402
    PHASE6_LOCKED_PATHS,
    load_phase6_execution_manifest,
    verify_phase6_package_lock,
)
from compensability_v4.training.phase4 import (  # noqa: E402
    attach_language_lora,
    discover_language_lora_targets,
    freeze_base_parameters,
    load_phase4_config,
    trainable_parameter_manifest,
)
from compensability_v4.training.phase6 import (  # noqa: E402
    Phase6Example,
    Phase6Variant,
    RewardGroupTrace,
    RewardKind,
    load_phase6_config,
    score_phase6_completion,
    summarize_reward_groups,
)

CONFIG = ROOT / "configs/recoverability/v4_phase_6.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml"
PHASE4_CONFIG = ROOT / "configs/recoverability/v4_phase_4.yaml"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
DATA_ROOT = ROOT / "artifacts/v4/rl/data"
RUN_ROOT = ROOT / "artifacts/v4/rl/runs/phase6-r1"
EXECUTION_MANIFEST = ROOT / "artifacts/v4/phase6/execution_manifest.json"
_ACK = "I_UNDERSTAND_THIS_STARTS_PHASE_6_GRPO_TRAINING"


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 6 {label} is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Phase 6 {label} must contain one object")
    return value


def _rows(path: Path, label: str) -> tuple[Phase6Example, ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 6 {label} is missing or unsafe")
    values = tuple(
        Phase6Example.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values:
        raise RuntimeError(f"Phase 6 {label} is empty")
    return values


def _validate_data(
    root: Path,
    *,
    execution_manifest_sha256: str,
    config_sha256: str,
    package_lock_sha256: str,
) -> tuple[dict[RewardKind, tuple[Phase6Example, ...]], str]:
    recovery_path, answer_path, summary_path = (
        root / "recovery_outcome.jsonl",
        root / "answer_only.jsonl",
        root / "summary.json",
    )
    summary = _json(summary_path, "RL data summary")
    recovery, answer = _rows(recovery_path, "recovery rows"), _rows(answer_path, "answer rows")
    if (
        summary.get("status") != "PHASE_6_RL_DATA_FROZEN"
        or summary.get("recovery_outcome_count") != len(recovery)
        or summary.get("answer_only_count") != len(answer)
        or summary.get("scene_count") != len(recovery)
        or summary.get("recovery_outcome_sha256") != sha256(recovery_path)
        or summary.get("answer_only_sha256") != sha256(answer_path)
        or summary.get("config_sha256") != config_sha256
        or summary.get("package_lock_sha256") != package_lock_sha256
        or summary.get("confirmatory_data_used") is not False
        or summary.get("subjective_success_threshold_applied") is not False
        or summary.get("execution_manifest_sha256") != execution_manifest_sha256
    ):
        raise RuntimeError("Phase 6 RL data summary/provenance is malformed")
    if {row.scene_id for row in recovery} != {row.scene_id for row in answer}:
        raise RuntimeError("Phase 6 reward-view scene closure drifted")
    return {
        RewardKind.RECOVERY_OUTCOME: recovery,
        RewardKind.ANSWER_ONLY: answer,
    }, sha256(summary_path)


def _verify_frozen_components(
    manifest: Mapping[str, object],
    *,
    phase4_run_root: Path,
) -> None:
    source_hashes = manifest.get("source_sha256")
    phase4_hashes = manifest.get("phase4_adapter_sha256")
    if not isinstance(source_hashes, Mapping) or not isinstance(phase4_hashes, Mapping):
        raise RuntimeError("Phase 6 execution manifest is missing frozen-component hashes")
    if source_hashes.get("Base") != MODEL_SNAPSHOT_SHA256:
        raise RuntimeError("Phase 6 Base model hash drifted from the execution manifest")
    if manifest.get("model_snapshot_sha256") != MODEL_SNAPSHOT_SHA256:
        raise RuntimeError("Phase 6 model snapshot drifted from the execution manifest")
    for checkpoint, relative in {
        "C0": "C0_format_only/final_adapter",
        "C1": "C1_forward_arithmetic/final_adapter",
        "T": "T_constraint_recovery/final_adapter",
    }.items():
        observed = tree_sha256(phase4_run_root / relative)
        if observed != phase4_hashes.get(checkpoint) or observed != source_hashes.get(checkpoint):
            raise RuntimeError(
                f"Phase 6 cannot verify all required frozen model components ({checkpoint})"
            )


def _trl_api() -> tuple[object, object]:
    from trl import GRPOConfig, GRPOTrainer

    config_parameters = inspect.signature(GRPOConfig).parameters
    trainer_parameters = inspect.signature(GRPOTrainer).parameters
    required_config = {
        "output_dir",
        "learning_rate",
        "max_steps",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_prompt_length",
        "max_completion_length",
        "bf16",
        "temperature",
        "top_p",
        "top_k",
        "beta",
        "use_vllm",
    }
    if not required_config.issubset(config_parameters) or not {
        "model",
        "reward_funcs",
        "args",
        "train_dataset",
        "processing_class",
        "callbacks",
    }.issubset(trainer_parameters):
        raise RuntimeError("Phase 6 installed TRL GRPO API differs from the frozen executor")
    return GRPOConfig, GRPOTrainer


def _release(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _prepare_model(variant: Phase6Variant) -> tuple[object, object, dict[str, object]]:
    model, processor = load_pinned_qwen(model_path=Path(MODEL_PATH), device_map="cuda:0")
    targets = discover_language_lora_targets(model)
    frozen = freeze_base_parameters(model)
    if variant.initial_checkpoint == "Base":
        model = attach_language_lora(
            model,
            config=load_phase4_config(PHASE4_CONFIG),
            targets=targets,
        )
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(PHASE4_RUN_ROOT / "T_constraint_recovery/final_adapter"),
            is_trainable=True,
        )
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable_checkpointing):
            enable_checkpointing()
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_grads):
            enable_input_grads()
    manifest = trainable_parameter_manifest(model, targets)
    return model, processor, {"frozen": frozen, "trainable": manifest}


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = tuple(value)
        if parts and all(isinstance(part, Mapping) for part in parts):
            content = parts[-1].get("content")  # type: ignore[union-attr]
            if isinstance(content, str):
                return content
    raise RuntimeError("Phase 6 TRL completion has an unsupported structure")


def _expanded(values: object, size: int, label: str) -> tuple[object, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise RuntimeError(f"Phase 6 reward metadata {label} is malformed")
    items = tuple(values)
    if len(items) == size:
        return items
    if size % len(items):
        raise RuntimeError(f"Phase 6 reward metadata {label} cannot align to completions")
    repeat = size // len(items)
    return tuple(item for item in items for _ in range(repeat))


def _reward_function(
    *,
    examples: tuple[Phase6Example, ...],
    variant: Phase6Variant,
    trace_path: Path,
):
    index = {row.example_id: row for row in examples}
    call_index = 0
    if trace_path.is_file():
        existing = tuple(
            json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
        )
        if existing:
            indices = tuple(row.get("reward_call_index") for row in existing)
            if any(type(value) is not int or int(value) < 0 for value in indices):
                raise RuntimeError("Phase 6 reward trace call indices are malformed")
            call_index = max(int(value) for value in indices) + 1

    def reward(completions: Sequence[object], **kwargs: object) -> list[float]:
        nonlocal call_index
        ids = _expanded(kwargs.get("example_id"), len(completions), "example_id")
        trainer_state = kwargs.get("trainer_state")
        step = int(getattr(trainer_state, "global_step", -1))
        records: list[dict[str, object]] = []
        rewards: list[float] = []
        for position, (completion, example_id) in enumerate(zip(completions, ids, strict=True)):
            if not isinstance(example_id, str) or example_id not in index:
                raise RuntimeError("Phase 6 reward received an unknown example_id")
            example = index[example_id]
            text = _completion_text(completion)
            score = score_phase6_completion(example, text)
            rewards.append(score.reward)
            records.append(
                {
                    "schema_version": 1,
                    "trainer_step": step,
                    "reward_call_index": call_index,
                    "position": position,
                    "variant": variant.value,
                    "example_id": example.example_id,
                    "scene_id": example.scene_id,
                    "reward_kind": example.reward_kind.value,
                    "completion": text,
                    "reward": score.reward,
                    "exact_world_recovery": score.exact_world_recovery,
                    "observation_copy": score.observation_copy,
                    "answer_exact": score.answer_exact,
                    "parse_success": score.parse_success,
                }
            )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        call_index += 1
        return rewards

    reward.__name__ = f"{variant.value}_reward"
    return reward


def _restore_reward_trace(checkpoint: str | None, trace_path: Path) -> None:
    if checkpoint is None:
        if trace_path.exists():
            raise RuntimeError("Phase 6 fresh run has a stale reward trace")
        return
    snapshot = Path(checkpoint) / "reward_trace.jsonl"
    if snapshot.is_symlink() or not snapshot.is_file():
        raise RuntimeError("Phase 6 checkpoint is missing its reward-trace snapshot")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trace_path.with_suffix(".restore.tmp")
    shutil.copyfile(snapshot, temporary)
    temporary.replace(trace_path)


def _reward_trace_callback(trace_path: Path, output_dir: Path) -> object:
    from transformers import TrainerCallback

    class RewardTraceCheckpointCallback(TrainerCallback):
        def on_save(self, args: object, state: object, control: object, **kwargs: object) -> object:
            step = getattr(state, "global_step", None)
            if type(step) is not int or step < 0 or not trace_path.is_file():
                raise RuntimeError("Phase 6 cannot checkpoint an incomplete reward trace")
            checkpoint_root = output_dir / f"checkpoint-{step}"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trace_path, checkpoint_root / "reward_trace.jsonl")
            return control

    return RewardTraceCheckpointCallback()


def _metric(log: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        value = log.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _diagnostics(
    trace_path: Path,
    *,
    variant: Phase6Variant,
    group_size: int,
    logs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows = tuple(
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    )
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or row.get("variant") != variant.value:
            raise RuntimeError("Phase 6 reward trace is malformed")
        grouped[(int(row["reward_call_index"]), str(row["example_id"]))].append(row)
    metrics = {
        int(log["step"]): log
        for log in logs
        if isinstance(log, Mapping) and type(log.get("step")) is int
    }
    groups: list[RewardGroupTrace] = []
    for (call_index, example_id), group in sorted(grouped.items()):
        if len(group) != group_size:
            raise RuntimeError("Phase 6 recorded reward group size drifted")
        step = int(group[0]["trainer_step"])
        log = metrics.get(step, {})
        groups.append(
            RewardGroupTrace(
                group_id=f"{call_index}:{example_id}",
                scene_id=str(group[0]["scene_id"]),
                variant=variant,
                rewards=tuple(float(row["reward"]) for row in group),
                kl=_metric(log, "kl", "objective/kl"),
                entropy=_metric(log, "entropy", "objective/entropy"),
                exact_world_recovery_count=sum(
                    row.get("exact_world_recovery") is True for row in group
                ),
                observation_copy_count=sum(row.get("observation_copy") is True for row in group),
            )
        )
    return summarize_reward_groups(groups, group_size=group_size)


def _completed(
    root: Path,
    variant: Phase6Variant,
    *,
    data_hash: str,
    config_hash: str,
    package_lock_sha256: str,
    execution_manifest_sha256: str,
) -> bool:
    evidence_path = root / variant.value / "execution_evidence.json"
    if not evidence_path.exists():
        return False
    evidence = _json(evidence_path, f"{variant.value} execution evidence")
    adapter = root / variant.value / "final_adapter"
    if (
        evidence.get("status") != "PHASE_6_VARIANT_TRAINED"
        or evidence.get("variant") != variant.value
        or evidence.get("data_summary_sha256") != data_hash
        or evidence.get("config_sha256") != config_hash
        or evidence.get("package_lock_sha256") != package_lock_sha256
        or evidence.get("execution_manifest_sha256") != execution_manifest_sha256
        or evidence.get("final_adapter_tree_sha256") != tree_sha256(adapter)
    ):
        raise RuntimeError(f"Phase 6 completed evidence drifted for {variant.value}")
    return True


def _last_checkpoint(root: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(root)) if root.is_dir() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--execution-manifest", type=Path, default=EXECUTION_MANIFEST)
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 6 GRPO requires explicit --execute.")
        return 2
    if not arguments.execution_manifest_sha256:
        print("BLOCKED: Phase 6 GRPO requires --execution-manifest-sha256.")
        return 2
    if not arguments.preflight_only and os.environ.get("COMPBIAS_V4_PHASE6_RL_ACK") != _ACK:
        print("BLOCKED: COMPBIAS_V4_PHASE6_RL_ACK is required before any optimizer step.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        _payload, config = load_phase6_config(arguments.config)
        lock_hash = verify_phase6_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE6_LOCKED_PATHS,
        )
        config_hash = sha256(arguments.config)
        manifest = load_phase6_execution_manifest(
            arguments.execution_manifest,
            expected_sha256=arguments.execution_manifest_sha256,
            expected_config_sha256=config_hash,
            expected_package_lock_sha256=lock_hash,
        )
        _verify_frozen_components(manifest, phase4_run_root=PHASE4_RUN_ROOT)
        examples, data_hash = _validate_data(
            arguments.data_root,
            execution_manifest_sha256=arguments.execution_manifest_sha256,
            config_sha256=config_hash,
            package_lock_sha256=lock_hash,
        )
        GRPOConfig, GRPOTrainer = _trl_api()
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Phase 6 requires CUDA with bf16 support")
        t_adapter = PHASE4_RUN_ROOT / "T_constraint_recovery/final_adapter"
        if not t_adapter.is_dir() or t_adapter.is_symlink():
            raise RuntimeError("Phase 6 recovery adapter is missing or unsafe")
        if arguments.preflight_only:
            for initial in (Phase6Variant.BASE_ANSWER_ONLY, Phase6Variant.RECOVERY_OUTCOME):
                model, _processor, _manifests = _prepare_model(initial)
                _release(model)
            print("READY: Phase 6 data, TRL API, CUDA, models, and trainability preflight passed")
            return 0
        from datasets import Dataset

        arguments.run_root.mkdir(parents=True, exist_ok=True)
        for variant in Phase6Variant:
            if _completed(
                arguments.run_root,
                variant,
                data_hash=data_hash,
                config_hash=config_hash,
                package_lock_sha256=lock_hash,
                execution_manifest_sha256=arguments.execution_manifest_sha256,
            ):
                print(f"RESUMED: Phase 6 {variant.value} already complete", flush=True)
                continue
            variant_root = arguments.run_root / variant.value
            checkpoint = _last_checkpoint(variant_root)
            if variant_root.exists() and checkpoint is None:
                raise RuntimeError(
                    f"Phase 6 {variant.value} output exists without a resumable checkpoint"
                )
            rows = examples[variant.reward_kind]
            model, processor, manifests = _prepare_model(variant)
            trace_path = variant_root / "reward_trace.jsonl"
            _restore_reward_trace(checkpoint, trace_path)
            training_args = GRPOConfig(
                output_dir=str(variant_root),
                learning_rate=config.learning_rate,
                max_steps=config.max_steps,
                num_generations=config.group_size,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_prompt_length=config.max_prompt_length,
                max_completion_length=config.max_completion_length,
                bf16=True,
                fp16=False,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                beta=config.kl_beta,
                use_vllm=config.use_vllm,
                gradient_checkpointing=True,
                logging_steps=1,
                save_strategy="steps",
                save_steps=config.checkpoint_steps,
                save_total_limit=2,
                report_to="none",
                remove_unused_columns=False,
                seed=config.seed,
            )
            dataset = Dataset.from_list([row.to_mapping() for row in rows])
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=_reward_function(
                    examples=rows,
                    variant=variant,
                    trace_path=trace_path,
                ),
                args=training_args,
                train_dataset=dataset,
                processing_class=processor,
                callbacks=[_reward_trace_callback(trace_path, variant_root)],
            )
            trainer.train(resume_from_checkpoint=checkpoint)
            final_adapter = variant_root / "final_adapter"
            trainer.save_model(str(final_adapter))
            metrics_path = variant_root / "metrics.json"
            with metrics_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    trainer.state.log_history, stream, sort_keys=True, indent=2, allow_nan=False
                )
                stream.write("\n")
            diagnostics = _diagnostics(
                trace_path,
                variant=variant,
                group_size=config.group_size,
                logs=trainer.state.log_history,
            )
            diagnostics_path = variant_root / "grpo_signal_diagnostics.json"
            with diagnostics_path.open("x", encoding="utf-8") as stream:
                json.dump(diagnostics, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            evidence = {
                "schema_version": 1,
                "status": "PHASE_6_VARIANT_TRAINED",
                "variant": variant.value,
                "initial_checkpoint": variant.initial_checkpoint,
                "reward_kind": variant.reward_kind.value,
                "config_sha256": config_hash,
                "package_lock_sha256": lock_hash,
                "execution_manifest_sha256": arguments.execution_manifest_sha256,
                "data_summary_sha256": data_hash,
                "initial_T_adapter_tree_sha256": tree_sha256(t_adapter),
                "final_adapter_tree_sha256": tree_sha256(final_adapter),
                "metrics_sha256": sha256(metrics_path),
                "reward_trace_sha256": sha256(trace_path),
                "diagnostics_sha256": sha256(diagnostics_path),
                "manifests": manifests,
                "training_invoked": True,
                "rl_invoked": True,
                "confirmatory_data_used": False,
                "subjective_success_threshold_applied": False,
            }
            with (variant_root / "execution_evidence.json").open("x", encoding="utf-8") as stream:
                json.dump(evidence, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            _release(model)
            print(f"READY: Phase 6 {variant.value} GRPO complete", flush=True)
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 6 GRPO variants written below {arguments.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
