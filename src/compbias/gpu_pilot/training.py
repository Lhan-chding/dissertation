"""TRL/PEFT executor imported only after explicit GPU acknowledgement."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from compbias.io.logging import capture_environment
from compbias.io.strict_json import write_new_json
from compbias.models.structured_parser import ParseStatus, parse_trajectory

from .config import load_pilot_paths
from .qwen_smoke import load_local_qwen
from .safe_io import atomic_write_json_text, prepare_new_output_directory
from .structured_generation import (
    build_structured_instruction,
    numeric_answer_matches,
    validate_pilot_trajectory,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def _tree_sha256(root: Path) -> str:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise RuntimeError(f"cannot inspect final adapter tree: {root}") from error
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise RuntimeError("final adapter must be a regular non-symlink directory")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if path.is_symlink():
            raise RuntimeError(f"adapter tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"adapter tree contains a non-regular file: {path}")
        files += 1
        total_bytes += metadata.st_size
        if files > 10_000 or total_bytes > 20 * 1024**3:
            raise RuntimeError("final adapter tree exceeds the registered artifact budget")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    if files == 0:
        raise RuntimeError("final adapter tree is empty")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"audit artifact must be a regular non-symlink file: {path}")
    if metadata.st_size > 1024**3:
        raise RuntimeError(f"audit artifact exceeds the 1 GiB limit: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def _canonical_execution_command(stage: str) -> tuple[str, ...]:
    if stage == "pilot_a":
        return (
            "experiments/gpu_pilot/05_pilot_a.py",
            "--config",
            "configs/train/pilot_a.yaml",
            "--execute",
        )
    return (
        "experiments/gpu_pilot/06_pilot_b.py",
        "--config",
        "configs/train/pilot_b_lm_only.yaml",
        "--execute",
    )


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        content = completion[-1].get("content")
        if isinstance(content, str):
            return content
    return ""


def outcome_reward(
    completions: list[object],
    answer: list[object],
    operation: list[object],
    expected_value_count: list[object],
    **_: object,
) -> list[float]:
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        operation_value = operation[index]
        value_count = expected_value_count[index]
        if not isinstance(operation_value, str):
            raise ValueError("reward operation must be a string")
        if isinstance(value_count, bool) or not isinstance(value_count, int):
            raise ValueError("reward expected_value_count must be an integer")
        parsed = validate_pilot_trajectory(
            parse_trajectory(_completion_text(completion), sample_id=f"reward-{index}"),
            operation=operation_value,
            expected_value_count=value_count,
        )
        expected = answer[index]
        rewards.append(
            1.0
            if parsed.status is ParseStatus.OK and numeric_answer_matches(parsed.answer, expected)
            else 0.0
        )
    return rewards


def _audited_reward(path: Path) -> Callable[..., list[float]]:
    if path.exists():
        raise FileExistsError(f"rollout audit already exists: {path}")
    call_index = 0

    def reward(
        completions: list[object],
        answer: list[object],
        operation: list[object],
        expected_value_count: list[object],
        sample_id: list[object],
        **kwargs: object,
    ) -> list[float]:
        nonlocal call_index
        rewards = outcome_reward(
            completions,
            answer,
            operation,
            expected_value_count,
            **kwargs,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "x" if call_index == 0 else "a"
        with path.open(mode, encoding="utf-8") as stream:
            for row_index, value in enumerate(rewards):
                record = {
                    "schema_version": 1,
                    "call_index": call_index,
                    "row_index": row_index,
                    "sample_id": sample_id[row_index],
                    "raw_completion": _completion_text(completions[row_index]),
                    "expected_answer": answer[row_index],
                    "operation": operation[row_index],
                    "expected_value_count": expected_value_count[row_index],
                    "reward": value,
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        call_index += 1
        return rewards

    reward.__name__ = "audited_outcome_reward"
    return reward


def _write_jsonl_new(path: Path, records: list[Mapping[str, object]]) -> None:
    if not records:
        raise RuntimeError(f"cannot publish an empty JSONL audit: {path.name}")
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")


def _language_lora_pattern(model: object) -> str:
    endings = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = [
        name
        for name, _module in model.named_modules()  # type: ignore[attr-defined]
        if "language_model" in name and name.endswith(endings)
    ]
    if not names:
        raise RuntimeError(
            "no language-only LoRA targets were discovered; stop for architecture audit"
        )
    return "^(?:" + "|".join(re.escape(name) for name in names) + ")$"


def _prompt_for_a(record: dict[str, Any]) -> list[dict[str, str]]:
    parsed = record.get("parsed")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("perceived_scene"), dict):
        raise ValueError("Pilot A record lacks validated perceived_scene evidence")
    mediator = json.dumps(
        parsed["perceived_scene"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    question = record.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("Pilot A record lacks a validated question")
    operation = record.get("operation")
    values = record.get("values")
    if not isinstance(operation, str) or not isinstance(values, list):
        raise ValueError("Pilot A record lacks validated operation or values")
    instruction = build_structured_instruction(
        operation=operation,
        expected_value_count=len(values),
    )
    text = (
        "The image is unavailable. Treat the following naturally produced visual evidence "
        "as fixed. "
        f"Evidence: {mediator}\nQuestion: {question}\n"
        "Copy the fixed evidence values exactly into the perception field. "
        f"{instruction}"
    )
    return [{"role": "user", "content": text}]


def _pilot_a_rows(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    eligible = {"visual_error", "compensated_visual_error"}
    return [
        {
            "prompt": _prompt_for_a(record),
            "sample_id": record["sample_id"],
            "answer": record["answer"],
            "operation": record["operation"],
            "expected_value_count": len(record["values"]),
        }
        for record in records
        if record.get("error_type") in eligible
    ]


def _prompt_for_b(record: dict[str, Any]) -> list[dict[str, object]]:
    question = record.get("question")
    operation = record.get("operation")
    values = record.get("values")
    if (
        not isinstance(question, str)
        or not question
        or not isinstance(operation, str)
        or not isinstance(values, list)
    ):
        raise ValueError("Pilot B record lacks validated question, operation, or values")
    instruction = build_structured_instruction(
        operation=operation,
        expected_value_count=len(values),
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"{question} {instruction}",
                },
            ],
        }
    ]


def run_grpo_stage(config: dict[str, object]) -> None:
    if os.environ.get("COMPBIAS_GPU_EXECUTION_ACK") != "I_UNDERSTAND_THIS_STARTS_GPU_TRAINING":
        raise RuntimeError("COMPBIAS_GPU_EXECUTION_ACK is missing at the training executor")
    project_root = Path(__file__).resolve().parents[3]
    stage = str(config["stage"])
    stage_config_path = config.get("validated_stage_config_path")
    paths_config_path_value = config.get("validated_paths_config_path")
    if not isinstance(stage_config_path, str) or not isinstance(paths_config_path_value, str):
        raise RuntimeError("validated stage and paths config locations are missing")
    from .stages import load_stage_config

    canonical_config = load_stage_config(Path(stage_config_path), stage)
    supplied_base = {key: config.get(key) for key in canonical_config}
    if supplied_base != canonical_config:
        raise RuntimeError("in-memory training config differs from the validated stage config")
    paths_config = Path(paths_config_path_value).resolve()
    expected_paths_config = (project_root / str(canonical_config["paths_config"])).resolve()
    if paths_config != expected_paths_config:
        raise RuntimeError("validated paths config does not match the canonical stage config")
    paths = load_pilot_paths(paths_config)
    if paths.project_root != project_root:
        raise RuntimeError("paths.project_root must equal the active clean project checkout")
    training = canonical_config["training"]
    assert isinstance(training, dict)
    initial_environment = capture_environment(
        worktree=project_root,
        dataset_manifest_hash=None,
        seed=int(training["seed"]),
        model_revision=None,
        verl_revision=None,
        command=_canonical_execution_command(stage),
    )
    if initial_environment["git_commit"] is None or initial_environment["git_dirty"] is not False:
        raise RuntimeError("training executor requires the current clean Git commit")
    from .preflight import audit_server
    from .qwen_smoke import run_smoke

    preflight_path = paths.outputs / "preflight" / "report.json"
    live_preflight = audit_server(paths)
    atomic_write_json_text(
        paths.outputs,
        preflight_path,
        json.dumps(live_preflight, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    live_smoke = run_smoke(paths.model_path, paths.outputs / "smoke")
    if live_smoke.get("smoke_passed") is not True:
        raise RuntimeError("live known-answer smoke failed; GPU training remains blocked")
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    config = {
        **config,
        "validated_preflight_path": str(preflight_path),
        "validated_smoke_path": str(paths.outputs / "smoke" / "smoke_report.json"),
    }
    from .execution_gate import validate_execution_evidence

    current_hashes = validate_execution_evidence(
        config,
        paths,
        stage_config_path=Path(stage_config_path),
        paths_config_path=paths_config,
    )
    evidence = current_hashes
    current_environment = capture_environment(
        worktree=project_root,
        dataset_manifest_hash=current_hashes["dataset_manifest"],
        seed=int(training["seed"]),
        model_revision=current_hashes["model_snapshot"],
        verl_revision=None,
        command=_canonical_execution_command(stage),
    )
    for key in ("git_commit", "git_dirty", "package_versions", "cuda_available", "gpu_devices"):
        if current_environment[key] != initial_environment.get(key):
            raise RuntimeError(f"execution environment changed after authorization: {key}")
    if current_environment["git_commit"] is None or current_environment["git_dirty"] is not False:
        raise RuntimeError("training executor requires the current clean Git commit")
    environment = current_environment
    output_root = paths.outputs.absolute()
    output_dir = prepare_new_output_directory(
        output_root,
        output_root / str(config["output_subdir"]) / "runs" / f"run-{uuid4().hex}",
    )
    preflight_source = Path(str(config["validated_preflight_path"]))
    smoke_source = Path(str(config["validated_smoke_path"]))
    preflight_snapshot = output_dir / "authorization" / "preflight.json"
    smoke_snapshot = output_dir / "authorization" / "smoke.json"
    _copy_new(preflight_source, preflight_snapshot)
    _copy_new(smoke_source, smoke_snapshot)
    if (
        _sha256(preflight_snapshot) != evidence["preflight"]
        or _sha256(smoke_snapshot) != evidence["smoke"]
    ):
        raise RuntimeError("volatile authorization reports changed before the training snapshot")
    import torch
    from datasets import Dataset, Image
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    model, processor = load_local_qwen(paths.model_path)
    target_pattern = _language_lora_pattern(model)
    lora = LoraConfig(
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_pattern,
    )
    if stage == "pilot_a":
        records = _read_jsonl(paths.project_root / str(config["natural_records"]))
        rows = _pilot_a_rows(records)
    elif stage == "pilot_b_lm_only":
        manifest = paths.project_root / str(config["dataset_manifest"])
        records = _read_jsonl(manifest.parent / "records.jsonl")
        rows = [
            {
                "prompt": _prompt_for_b(record),
                "sample_id": record["sample_id"],
                "image": str(manifest.parent / str(record["image"])),
                "answer": record["answer"],
                "operation": record["operation"],
                "expected_value_count": len(record["values"]),
            }
            for record in records
            if record.get("split") == "pilot_train"
        ]
    else:
        raise ValueError(f"unsupported stage: {stage}")
    if not rows:
        raise RuntimeError("training dataset is empty")
    dataset = Dataset.from_list(rows)
    if stage == "pilot_b_lm_only":
        dataset = dataset.cast_column("image", Image())
    args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(training["learning_rate"]),
        max_steps=int(training["max_steps"]),
        num_generations=int(training["num_generations"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_prompt_length=int(training["max_prompt_length"]),
        max_completion_length=int(training["max_completion_length"]),
        bf16=True,
        report_to="none",
        save_strategy="steps",
        save_steps=max(1, int(training["max_steps"]) // 4),
        seed=int(training["seed"]),
        use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=_audited_reward(output_dir / "rollouts.jsonl"),
        args=args,
        train_dataset=dataset,
        peft_config=lora,
    )
    torch.cuda.reset_peak_memory_stats(0)
    trainer.train()
    _write_jsonl_new(output_dir / "metrics.jsonl", list(trainer.state.log_history))
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    final_adapter_hash = _tree_sha256(final_adapter)
    metrics_hash = _sha256(output_dir / "metrics.jsonl")
    rollouts_hash = _sha256(output_dir / "rollouts.jsonl")
    write_new_json(
        output_dir / "execution_evidence.json",
        {
            "schema_version": 1,
            "artifact_type": "compbias_gpu_pilot_execution_evidence",
            "stage": stage,
            "training_completed": True,
            "end_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "final_adapter_sha256": final_adapter_hash,
            "metrics_sha256": metrics_hash,
            "rollouts_sha256": rollouts_hash,
            "evidence_sha256": dict(evidence),
            "authorization_files": {
                "preflight": "authorization/preflight.json",
                "smoke": "authorization/smoke.json",
            },
            "environment": dict(environment),
        },
    )
