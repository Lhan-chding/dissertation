#!/usr/bin/env python3
"""Run the one-seed v5 Study-C B3 reward-only GRPO contrast on a server."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v4.qwen.model_loader import MODEL_PATH, load_pinned_qwen  # noqa: E402
from compensability_v4.qwen.phase5_runtime import (  # noqa: E402
    tree_sha256 as study_c_output_tree_sha256,
)
from compensability_v4.training.phase4 import freeze_base_parameters  # noqa: E402
from compensability_v5.data.common_action_freeze import (  # noqa: E402
    assert_common_action_preflight,
)
from compensability_v5.qwen.study_b_runtime import (  # noqa: E402
    tree_sha256 as study_b_adapter_tree_sha256,
)
from compensability_v5.qwen.study_c_runtime import (  # noqa: E402
    ACTION_PARSER_ID,
    STUDY_C_ACK,
    STUDY_C_SEED,
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
from compensability_v5.training.train_support_lora import (  # noqa: E402
    require_offline_environment,
    sha256_file,
)

CONFIG = ROOT / "configs/v5/common_space_grpo.yaml"
PACKAGE_LOCK = ROOT / "configs/v5/server_package_lock.yaml"
COMMON_ACTION_MANIFEST = ROOT / "artifacts/v5/data/common_space_rl.json"
OUTPUT_ROOT = ROOT / "artifacts/v5/rl/study-c-pilot"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-b2", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--config-sha256")
    parser.add_argument("--package-lock", type=Path, default=PACKAGE_LOCK)
    parser.add_argument("--package-lock-sha256")
    parser.add_argument("--common-action-manifest", type=Path, default=COMMON_ACTION_MANIFEST)
    parser.add_argument("--common-action-manifest-sha256")
    parser.add_argument("--b3-adapter", type=Path)
    parser.add_argument("--b3-adapter-sha256")
    parser.add_argument("--b2-adapter", type=Path)
    parser.add_argument("--b2-adapter-sha256")
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def _require_digest(value: str | None, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StudyCError(f"{label} requires a lowercase SHA-256")
    return value


def _verify_file(
    path: Path,
    expected: str | None,
    label: str,
    *,
    canonical: Path | None = None,
) -> str:
    if canonical is not None and path.resolve() != canonical.resolve():
        raise StudyCError(f"{label} must be canonical repository file: {canonical}")
    wanted = _require_digest(expected, label)
    actual = sha256_file(path)
    if actual != wanted:
        raise StudyCError(f"{label} SHA-256 mismatch")
    return actual


def _verify_tree(path: Path, expected: str | None, label: str) -> str:
    wanted = _require_digest(expected, label)
    actual = study_b_adapter_tree_sha256(path)
    if actual != wanted:
        raise StudyCError(f"{label} tree SHA-256 mismatch")
    return actual


def _load_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise StudyCError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudyCError(f"cannot load {label}: {error}") from error
    if not isinstance(payload, dict):
        raise StudyCError(f"{label} must contain one mapping")
    return payload


def _validate_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StudyCError("Study C config must contain one mapping")
    return validate_study_c_config_payload(payload)


def _trl_api() -> tuple[type[Any], type[Any]]:
    from trl import GRPOConfig, GRPOTrainer

    required_config = {
        "output_dir",
        "learning_rate",
        "max_steps",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_completion_length",
        "bf16",
        "temperature",
        "top_p",
        "top_k",
        "beta",
        "use_vllm",
    }
    required_trainer = {
        "model",
        "reward_funcs",
        "args",
        "train_dataset",
        "processing_class",
        "callbacks",
    }
    missing_config = sorted(required_config - inspect.signature(GRPOConfig).parameters.keys())
    missing_trainer = sorted(required_trainer - inspect.signature(GRPOTrainer).parameters.keys())
    if missing_config or missing_trainer:
        raise StudyCError(
            "installed TRL GRPO API differs from the frozen Study C executor: "
            f"missing_config={missing_config}, missing_trainer={missing_trainer}"
        )
    return GRPOConfig, GRPOTrainer


def _latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if not path.is_symlink() and path.is_dir() and suffix.isdigit():
            checkpoints.append((int(suffix), path))
    return None if not checkpoints else max(checkpoints)[1]


def _prepare_arm_output(output_dir: Path, *, resume: bool) -> Path | None:
    """Authorize a checkpoint resume or an initialization-only empty restart."""

    if output_dir.is_symlink():
        raise StudyCError(f"Study C arm output is an unsafe symlink: {output_dir}")
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise StudyCError(f"Study C arm output is not a directory: {output_dir}")
    checkpoint = _latest_checkpoint(output_dir)
    if checkpoint is not None:
        if not resume:
            raise StudyCError(
                f"{output_dir.name} has partial output without an authorized resumable checkpoint"
            )
        return checkpoint
    if resume and next(output_dir.iterdir(), None) is None:
        output_dir.rmdir()
        print(
            f"RESUMED: removed verified empty initialization-only output {output_dir.name}",
            flush=True,
        )
        return None
    raise StudyCError(
        f"{output_dir.name} has partial output without an authorized resumable checkpoint"
    )


def _release(value: object) -> None:
    del value
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _factory_for_adapter(
    *,
    arm: StudyCArm,
    adapter: Path,
    model_path: Path,
    scenes: tuple[StudyCScene, ...],
    holder: list[object],
):
    def factory(**kwargs: object) -> object:
        import torch
        from datasets import Dataset
        from peft import PeftModel

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise StudyCError("Study C requires CUDA with bf16 support")
        GRPOConfig, GRPOTrainer = _trl_api()
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
        training_args = GRPOConfig(
            **build_grpo_config_kwargs(
                arm,
                Path(str(kwargs["output_dir"])),
                tuple(inspect.signature(GRPOConfig).parameters),
            )
        )
        dataset_rows = kwargs.get("dataset")
        if not isinstance(dataset_rows, tuple):
            raise StudyCError("Study C trainer received malformed frozen dataset")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=kwargs["reward_function"],
            args=training_args,
            train_dataset=Dataset.from_list(list(dataset_rows)),
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


def _verified_complete(arm: StudyCArm, output_dir: Path, provenance: dict[str, str]) -> bool:
    evidence_path = output_dir / "execution_evidence.json"
    if not evidence_path.exists():
        return False
    evidence = _load_json(evidence_path, f"{arm.name} execution evidence")
    trace = output_dir / "raw_reward_trace.jsonl"
    diagnostics = output_dir / "group_diagnostics.json"
    metrics = output_dir / "trainer_log_history.json"
    eval_rows = output_dir / "eval_raw_rows.jsonl"
    eval_summary = output_dir / "eval_summary.json"
    pre_rows = output_dir / "pre_training_eval_raw_rows.jsonl"
    pre_summary = output_dir / "pre_training_eval_summary.json"
    final_adapter = output_dir / "final_adapter"
    if (
        evidence.get("status") != "STUDY_C_ARM_COMPLETE"
        or evidence.get("arm") != arm.to_mapping()
        or evidence.get("provenance_sha256") != provenance
        or evidence.get("reward_trace_sha256") != sha256_file(trace)
        or evidence.get("diagnostics_sha256") != sha256_file(diagnostics)
        or evidence.get("trainer_log_sha256") != sha256_file(metrics)
        or evidence.get("post_training_evaluation_invoked") is not True
        or evidence.get("eval_raw_rows_sha256") != sha256_file(eval_rows)
        or evidence.get("eval_summary_sha256") != sha256_file(eval_summary)
        or evidence.get("final_adapter_tree_sha256") != study_c_output_tree_sha256(final_adapter)
        or (
            arm.reward_function == "answer"
            and (
                evidence.get("pre_training_evaluation_invoked") is not True
                or evidence.get("pre_training_eval_raw_rows_sha256") != sha256_file(pre_rows)
                or evidence.get("pre_training_eval_summary_sha256") != sha256_file(pre_summary)
            )
        )
    ):
        raise StudyCError(f"completed Study C evidence drifted for {arm.name}")
    return True


def _build_summary_payload(
    *,
    traces: dict[str, Path],
    baseline_traces: dict[str, Path],
    group_size: int,
    provenance: dict[str, str],
    registered_contract: dict[str, object],
) -> dict[str, object]:
    summary = build_study_c_summary(
        traces,
        group_size=group_size,
        baseline_trace_paths=baseline_traces,
    )
    return {
        **summary,
        "measurement_scope": "post_training_frozen_eval",
        "rollouts_per_scene": 16,
        "source_trace_sha256": {arm: sha256_file(path) for arm, path in sorted(traces.items())},
        "baseline_trace_sha256": {
            arm: sha256_file(path) for arm, path in sorted(baseline_traces.items())
        },
        "provenance_sha256": provenance,
        "registered_contract": registered_contract,
    }


def _verify_existing_summary(summary_path: Path, expected: dict[str, object]) -> None:
    existing = _load_json(summary_path, "Study C summary")
    if existing != expected:
        raise StudyCError("existing Study C summary content drifted from validated traces")


def _fixture() -> dict[str, object]:
    arms = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)
    return {
        "status": "FIXTURE_DRY_RUN_OK",
        "seed": STUDY_C_SEED,
        "action_parser_id": ACTION_PARSER_ID,
        "arms": [arm.name for arm in arms],
    }


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        print(json.dumps(_fixture(), sort_keys=True))
        return 0
    if not arguments.execute:
        print("BLOCKED: Study C GRPO requires explicit --execute.")
        return 2
    try:
        if arguments.ack != STUDY_C_ACK:
            raise StudyCError(f"exact --ack {STUDY_C_ACK} is required")
        require_offline_environment()
        config_hash = _verify_file(
            arguments.config, arguments.config_sha256, "Study C config", canonical=CONFIG
        )
        lock_hash = _verify_file(
            arguments.package_lock,
            arguments.package_lock_sha256,
            "Study C package lock",
            canonical=PACKAGE_LOCK,
        )
        manifest_hash = _verify_file(
            arguments.common_action_manifest,
            arguments.common_action_manifest_sha256,
            "Study C common-action manifest",
        )
        registered_contract = _validate_config(arguments.config)
        package = _load_json(arguments.common_action_manifest, "Study C common-action manifest")
        assert_common_action_preflight(package)
        scenes = load_study_c_scenes(package)
        split_contract = registered_contract["data_split"]
        if not isinstance(split_contract, dict):
            raise StudyCError("Study C split contract is malformed")
        split_study_c_scenes(
            scenes,
            expected_train_count=int(split_contract["train_scene_count"]),
            expected_eval_count=int(split_contract["eval_scene_count"]),
        )
        if arguments.b3_adapter is None:
            raise StudyCError("--b3-adapter is required")
        b3_hash = _verify_tree(
            arguments.b3_adapter, arguments.b3_adapter_sha256, "Study C B3 adapter"
        )
        initialization_hashes = package.get("initialization_hashes")
        if (
            not isinstance(initialization_hashes, dict)
            or initialization_hashes.get("B3") != b3_hash
        ):
            raise StudyCError("B3 adapter hash differs from common-action freeze")
        b2_hash: str | None = None
        if arguments.include_b2:
            if arguments.b2_adapter is None:
                raise StudyCError("--include-b2 requires --b2-adapter")
            b2_hash = _verify_tree(
                arguments.b2_adapter, arguments.b2_adapter_sha256, "Study C B2 adapter"
            )
            if initialization_hashes.get("B2") != b2_hash:
                raise StudyCError("B2 adapter hash differs from common-action freeze")
        arms = registered_study_c_arms(
            initialization="B3",
            initialization_hash=b3_hash,
            include_b2=arguments.include_b2,
            b2_initialization_hash=b2_hash,
        )
        provenance = {
            "common_action_manifest": manifest_hash,
            "config": config_hash,
            "package_lock": lock_hash,
        }
        if arguments.preflight_only:
            _trl_api()
            print(
                json.dumps(
                    {
                        "status": "STUDY_C_SERVER_PREFLIGHT_OK",
                        "seed": STUDY_C_SEED,
                        "scene_count": len(scenes),
                        "arms": [arm.name for arm in arms],
                        "training_invoked": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        output_root = arguments.output_root.resolve(strict=False)
        canonical_root = OUTPUT_ROOT.resolve(strict=False)
        try:
            output_root.relative_to(canonical_root)
        except ValueError as error:
            raise StudyCError(f"Study C outputs must remain below {OUTPUT_ROOT}") from error
        traces: dict[str, Path] = {}
        baseline_traces: dict[str, Path] = {}
        for arm in arms:
            arm_output = output_root / arm.name
            traces[arm.name] = arm_output / "eval_raw_rows.jsonl"
            if arm.reward_function == "answer":
                baseline_traces[arm.name] = arm_output / "pre_training_eval_raw_rows.jsonl"
            if _verified_complete(arm, arm_output, provenance):
                if not arguments.resume:
                    raise StudyCError(
                        f"{arm.name} is already complete; pass --resume to verify and continue"
                    )
                print(f"RESUMED: verified complete {arm.name}", flush=True)
                continue
            checkpoint = _prepare_arm_output(arm_output, resume=arguments.resume)
            adapter = arguments.b3_adapter if arm.initialization == "B3" else arguments.b2_adapter
            if adapter is None:
                raise StudyCError(f"missing adapter for {arm.initialization}")
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
                    trainer_factory=_factory_for_adapter(
                        arm=arm,
                        adapter=adapter,
                        model_path=arguments.model_path,
                        scenes=scenes,
                        holder=holder,
                    ),
                    provenance_sha256=provenance,
                    resume_from_checkpoint=checkpoint,
                    pre_training_evaluation_sampler_factory=(
                        evaluation_factory if arm.reward_function == "answer" else None
                    ),
                    evaluation_sampler_factory=evaluation_factory,
                )
            finally:
                for value in reversed(holder):
                    _release(value)
            print(f"READY: Study C {arm.name} complete", flush=True)
        summary_path = output_root / "study_c_summary.json"
        expected_summary = _build_summary_payload(
            traces=traces,
            baseline_traces=baseline_traces,
            group_size=arms[0].group_size,
            provenance=provenance,
            registered_contract=registered_contract,
        )
        if summary_path.exists():
            if not arguments.resume:
                raise StudyCError("Study C summary exists; overwrite forbidden")
            _verify_existing_summary(summary_path, expected_summary)
        else:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("x", encoding="utf-8") as stream:
                json.dump(expected_summary, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        print(f"STUDY_C_COMPLETE: output={output_root} seed={STUDY_C_SEED}")
        return 0
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
