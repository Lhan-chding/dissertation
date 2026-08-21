"""Adapters from the generic v5 gates to the executable Study-C runtime."""

from __future__ import annotations

import gc
import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from compensability_v4.qwen.model_loader import MODEL_PATH, load_pinned_qwen
from compensability_v4.qwen.phase5_runtime import tree_sha256
from compensability_v4.training.phase4 import freeze_base_parameters
from compensability_v5.data.common_action_freeze import assert_common_action_preflight
from compensability_v5.qwen.study_c_runtime import (
    StudyCArm,
    StudyCError,
    StudyCScene,
    build_grpo_config_kwargs,
    build_study_c_summary,
    load_study_c_scenes,
    qwen_text_evaluation_sampler,
    registered_study_c_arms,
    run_study_c_arm,
    split_study_c_scenes,
    validate_study_c_config_payload,
    validate_study_c_prompt_lengths,
)
from compensability_v5.training.train_common_space_grpo import (
    assert_common_space_reward_isolation,
)
from compensability_v5.training.train_support_lora import (
    require_offline_environment,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/v5/common_space_grpo.yaml"


def _validation(value: Mapping[str, object], phase: str) -> tuple[dict[str, str], Path]:
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise StudyCError("server validation must be a schema_version 1 mapping")
    if value.get("phase") != phase:
        raise StudyCError(f"server validation phase must be {phase}")
    inputs = value.get("input_sha256")
    output = value.get("output")
    if not isinstance(inputs, Mapping) or not inputs or not isinstance(output, str):
        raise StudyCError("server validation inputs/output are malformed")
    detached: dict[str, str] = {}
    for raw_path, digest in inputs.items():
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise StudyCError("server validation input hashes are malformed")
        path = Path(raw_path)
        if sha256_file(path) != digest:
            raise StudyCError(f"hash-bound input drifted: {path}")
        detached[str(path.resolve())] = digest
    return detached, Path(output)


def _common_action_package(inputs: Mapping[str, str]) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for name in inputs:
        path = Path(name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("status") == "V5_COMMON_ACTION_SPACE_FROZEN":
            matches.append((path, payload))
    if len(matches) != 1:
        raise StudyCError("exactly one hash-bound common-action freeze is required")
    path, package = matches[0]
    assert_common_action_preflight(package)
    return path, package


def _release(objects: list[object]) -> None:
    objects.clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _trainer_factory(
    arm: StudyCArm,
    adapter: Path,
    scenes: tuple[StudyCScene, ...],
    holder: list[object],
):
    def factory(**kwargs: object) -> object:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from trl import GRPOConfig, GRPOTrainer

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise StudyCError("Study C requires CUDA with bf16 support")
        required_config = {
            "output_dir",
            "learning_rate",
            "max_steps",
            "num_generations",
            "max_completion_length",
            "beta",
        }
        required_trainer = {"model", "reward_funcs", "args", "train_dataset"}
        missing_config = sorted(required_config - inspect.signature(GRPOConfig).parameters.keys())
        missing_trainer = sorted(
            required_trainer - inspect.signature(GRPOTrainer).parameters.keys()
        )
        if missing_config or missing_trainer:
            raise StudyCError(
                "installed TRL GRPO API differs from Study C: "
                f"missing_config={missing_config}, missing_trainer={missing_trainer}"
            )
        model_path = Path(os.environ.get("COMPBIAS_V5_MODEL_PATH", MODEL_PATH))
        model, processor = load_pinned_qwen(model_path=model_path, device_map="cuda:0")
        max_observed = validate_study_c_prompt_lengths(
            scenes, processor, max_prompt_length=arm.max_prompt_length
        )
        freeze_base_parameters(model)
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        for method_name in ("gradient_checkpointing_enable", "enable_input_require_grads"):
            method = getattr(model, method_name, None)
            if callable(method):
                method()
        args = GRPOConfig(
            **build_grpo_config_kwargs(
                arm,
                Path(str(kwargs["output_dir"])),
                tuple(inspect.signature(GRPOConfig).parameters),
            )
        )
        dataset = kwargs.get("dataset")
        if not isinstance(dataset, tuple):
            raise StudyCError("Study C trainer received malformed frozen dataset")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=kwargs["reward_function"],
            args=args,
            train_dataset=Dataset.from_list(list(dataset)),
            processing_class=processor,
            callbacks=list(kwargs["callbacks"]),  # type: ignore[arg-type]
        )
        trainer._study_c_prompt_length_audit = {
            "mode": "external_preflight",
            "limit": arm.max_prompt_length,
            "max_observed": max_observed,
            "passed_to_grpo_config": "max_prompt_length"
            in inspect.signature(GRPOConfig).parameters,
        }
        holder.extend((model, processor, trainer))
        return trainer

    return factory


def run_common_space_grpo(
    validation: Mapping[str, object], arms: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    """Run the B3 answer/state pair after the generic 09 gate has validated inputs."""

    require_offline_environment()
    input_hashes, output = _validation(validation, "phase7_common_space_grpo")
    if sha256_file(CONFIG) != validation.get("config_sha256"):
        raise StudyCError("canonical Study C config hash differs from generic validation")
    raw_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise StudyCError("canonical Study C config must contain one mapping")
    registered_contract = validate_study_c_config_payload(raw_config)
    assert_common_space_reward_isolation(arms)
    manifest_path, package = _common_action_package(input_hashes)
    if package.get("arms") != arms:
        raise StudyCError("generic-gate arms differ from the hash-bound common-action freeze")
    scenes = load_study_c_scenes(package)
    split_contract = registered_contract["data_split"]
    if not isinstance(split_contract, dict):
        raise StudyCError("Study C split contract is malformed")
    split_study_c_scenes(
        scenes,
        expected_train_count=int(split_contract["train_scene_count"]),
        expected_eval_count=int(split_contract["eval_scene_count"]),
    )
    initialization_hashes = package.get("initialization_hashes")
    if not isinstance(initialization_hashes, Mapping):
        raise StudyCError("common-action freeze has no initialization hashes")
    b3 = os.environ.get("COMPBIAS_V5_B3_ADAPTER")
    if not b3:
        raise StudyCError("COMPBIAS_V5_B3_ADAPTER is required")
    b3_adapter = Path(b3)
    b3_hash = tree_sha256(b3_adapter)
    if initialization_hashes.get("B3") != b3_hash:
        raise StudyCError("B3 adapter hash differs from the common-action freeze")
    include_b2 = os.environ.get("COMPBIAS_V5_INCLUDE_B2") == "1"
    b2_adapter: Path | None = None
    b2_hash: str | None = None
    if include_b2:
        b2 = os.environ.get("COMPBIAS_V5_B2_ADAPTER")
        if not b2:
            raise StudyCError("COMPBIAS_V5_INCLUDE_B2=1 requires COMPBIAS_V5_B2_ADAPTER")
        b2_adapter = Path(b2)
        b2_hash = tree_sha256(b2_adapter)
        if initialization_hashes.get("B2") != b2_hash:
            raise StudyCError("B2 adapter hash differs from the common-action freeze")
    study_arms = registered_study_c_arms(
        initialization="B3",
        initialization_hash=b3_hash,
        include_b2=include_b2,
        b2_initialization_hash=b2_hash,
    )
    if output.exists() or output.is_symlink():
        raise StudyCError(f"Study C output exists; overwrite forbidden: {output}")
    provenance = {
        "common_action_manifest": input_hashes[str(manifest_path.resolve())],
        "config": str(validation["config_sha256"]),
        "package_lock": str(validation["package_lock_sha256"]),
    }
    traces: dict[str, Path] = {}
    baseline_traces: dict[str, Path] = {}
    for arm in study_arms:
        adapter = b3_adapter if arm.initialization == "B3" else b2_adapter
        if adapter is None:
            raise StudyCError(f"missing adapter for {arm.initialization}")
        arm_output = output / arm.name
        holder: list[object] = []

        def evaluation_factory(
            _trainer: object,
            selected: StudyCArm = arm,
            selected_holder: list[object] = holder,
        ):
            return qwen_text_evaluation_sampler(
                arm=selected,
                model=selected_holder[0],
                processor=selected_holder[1],
            )

        try:
            run_study_c_arm(
                arm=arm,
                scenes=scenes,
                output_dir=arm_output,
                trainer_factory=_trainer_factory(arm, adapter, scenes, holder),
                provenance_sha256=provenance,
                pre_training_evaluation_sampler_factory=(
                    evaluation_factory if arm.reward_function == "answer" else None
                ),
                evaluation_sampler_factory=evaluation_factory,
            )
        finally:
            _release(holder)
        traces[arm.name] = arm_output / "eval_raw_rows.jsonl"
        if arm.reward_function == "answer":
            baseline_traces[arm.name] = arm_output / "pre_training_eval_raw_rows.jsonl"
    summary = build_study_c_summary(
        traces,
        group_size=study_arms[0].group_size,
        baseline_trace_paths=baseline_traces,
    )
    summary["measurement_scope"] = "post_training_frozen_eval"
    summary["rollouts_per_scene"] = 16
    summary["source_trace_sha256"] = {
        arm: sha256_file(path) for arm, path in sorted(traces.items())
    }
    summary["baseline_trace_sha256"] = {
        arm: sha256_file(path) for arm, path in sorted(baseline_traces.items())
    }
    summary["provenance_sha256"] = provenance
    summary["registered_contract"] = registered_contract
    summary_path = output / "study_c_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def run_v5_evaluation(
    validation: Mapping[str, object], request: Mapping[str, object]
) -> dict[str, object]:
    """Summarize explicitly hash-bound Study-C raw traces for generic entrypoint 10."""

    input_hashes, output = _validation(validation, "phase8_v5_evaluation")
    if request.get("task") != "evaluate_v5" or request.get("k") != 8:
        raise StudyCError("v5 evaluation request differs from Study C group size 8")
    traces: dict[str, Path] = {}
    baseline_traces: dict[str, Path] = {}
    for name in input_hashes:
        path = Path(name)
        if path.suffix != ".jsonl":
            continue
        try:
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        except (OSError, UnicodeError, json.JSONDecodeError, IndexError):
            continue
        arm = first.get("arm") if isinstance(first, Mapping) else None
        if isinstance(arm, str) and arm in {
            "B3_answer",
            "B3_exact_state",
            "B2_answer",
            "B2_exact_state",
        }:
            trace_kind = first.get("trace_kind")
            target = baseline_traces if trace_kind == "pre_training_frozen_eval" else traces
            if arm in target:
                raise StudyCError(f"multiple hash-bound traces claim arm {arm}")
            target[arm] = path
    if not {"B3_answer", "B3_exact_state"}.issubset(traces):
        raise StudyCError("v5 evaluation requires both B3 Study C raw traces")
    for arm, path in {**traces, **{f"baseline:{k}": v for k, v in baseline_traces.items()}}.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        per_scene: dict[str, int] = {}
        for row in rows:
            scene_id = row.get("scene_id") if isinstance(row, Mapping) else None
            if not isinstance(scene_id, str):
                raise StudyCError(f"post-training evaluation trace is malformed: {arm}")
            per_scene[scene_id] = per_scene.get(scene_id, 0) + 1
        if not per_scene or set(per_scene.values()) != {16}:
            raise StudyCError(
                f"post-training evaluation trace must contain 16 rollouts per scene: {arm}"
            )
    if output.exists() or output.is_symlink():
        raise StudyCError(f"v5 evaluation output exists; overwrite forbidden: {output}")
    summary = build_study_c_summary(traces, group_size=8, baseline_trace_paths=baseline_traces)
    summary["measurement_scope"] = "post_training_evaluation_16_rollouts_per_scene"
    summary["source_trace_sha256"] = dict(sorted(input_hashes.items()))
    summary["evaluation_config_sha256"] = validation.get("config_sha256")
    summary["evaluation_package_lock_sha256"] = validation.get("package_lock_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


__all__ = ["run_common_space_grpo", "run_v5_evaluation"]
