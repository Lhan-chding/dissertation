"""Pinned offline Qwen backend and server environment gates for Study B."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compensability_v5.qwen.study_b_inputs import StudyBError


@dataclass(frozen=True, slots=True)
class _TrainingRow:
    example_id: str
    prompt: str
    completion: str


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StudyBError(f"{label} must be a positive integer")
    return value


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class QwenStudyBBackend:  # pragma: no cover - requires the pinned CUDA/Qwen server
    """Real offline Qwen2.5-VL backend reusing the verified v4 LoRA helpers."""

    def __init__(self, *, model_path: Path, max_sequence_length: int = 512) -> None:
        self.model_path = Path(model_path)
        self.max_sequence_length = _positive_int(max_sequence_length, "max_sequence_length")

    def load_base(self, *, arm: str, expected_model_sha256: str) -> Mapping[str, object]:
        _require_single_4090()
        from compensability_v4.qwen.model_loader import load_pinned_qwen, require_server_model

        verified = require_server_model(self.model_path, expected_model_sha256)
        model, processor = load_pinned_qwen(model_path=verified, device_map="cuda:0")
        return {
            "model": model,
            "processor": processor,
            "model_sha256": expected_model_sha256,
            "load_token": f"{arm}-{uuid.uuid4().hex}",
        }

    def train(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        budget: dict[str, object],
        seed: int,
        output: Path,
    ) -> Mapping[str, object]:
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments, set_seed

        from compensability_v4.training.phase4 import (
            Phase4TrainingConfig,
            _chat_training_features,
            _import_torch,
            attach_language_lora,
            discover_language_lora_targets,
            freeze_base_parameters,
            trainable_parameter_manifest,
        )

        optimizer = budget["optimizer"]
        if not isinstance(optimizer, Mapping) or optimizer.get("name") != "adamw":
            raise StudyBError("real Study B backend supports only the registered adamw optimizer")
        model, processor = session["model"], session["processor"]
        config = Phase4TrainingConfig(
            precision="bf16",
            lora_rank=int(budget["lora_rank"]),
            lora_alpha=2 * int(budget["lora_rank"]),
            lora_dropout=0.0,
            gradient_checkpointing=True,
            vision_frozen=True,
            merger_frozen=True,
            learning_rate=float(optimizer["learning_rate"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(budget["gradient_accumulation"]),
            num_train_epochs=1,
            max_sequence_length=self.max_sequence_length,
            seed=seed,
            selection_split="support_dev",
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        training_rows = tuple(
            _TrainingRow(
                example_id=f"{arm}:{index}:{row['scene_id']}",
                prompt=str(row["prompt"]),
                completion=str(row["completion"]),
            )
            for index, row in enumerate(rows)
        )
        features = _chat_training_features(
            tokenizer=tokenizer,
            rows=training_rows,
            max_sequence_length=self.max_sequence_length,
        )
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise StudyBError("Qwen tokenizer has no exact encode method")
        observed_token_counts = tuple(
            len(encode(row.completion, add_special_tokens=False)) for row in training_rows
        )
        frozen_token_counts = tuple(int(row["target_tokens"]) for row in rows)
        if observed_token_counts != frozen_token_counts:
            raise StudyBError(
                f"{arm} completion token counts differ from the frozen real-tokenizer audit"
            )
        observed_target_tokens = sum(observed_token_counts)
        if observed_target_tokens != int(budget["target_tokens"]):
            raise StudyBError(f"{arm} actual completion token total differs from its budget")
        discovered = discover_language_lora_targets(model)
        requested = tuple(budget["lora_targets"])
        targets = tuple(name for name in discovered if name.rsplit(".", 1)[-1] in requested)
        if {name.rsplit(".", 1)[-1] for name in targets} != set(requested):
            raise StudyBError("registered LoRA targets differ from actual Qwen language modules")
        frozen = freeze_base_parameters(model)
        adapter_model = attach_language_lora(model, config=config, targets=targets)
        if isinstance(session, dict):
            session["model"] = adapter_model
        trainable = trainable_parameter_manifest(adapter_model, targets)
        torch = _import_torch()
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_id, int):
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(pad_id, int):
            raise StudyBError("Qwen tokenizer has no padding token")

        def fixed_collator(batch: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
            result: dict[str, list[list[int]]] = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
            for item in batch:
                padding = self.max_sequence_length - len(item["input_ids"])
                if padding < 0:
                    raise StudyBError("training example exceeds fixed Study B sequence length")
                result["input_ids"].append(list(item["input_ids"]) + [pad_id] * padding)
                result["attention_mask"].append(list(item["attention_mask"]) + [0] * padding)
                result["labels"].append(list(item["labels"]) + [-100] * padding)
            return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}

        set_seed(seed)
        arguments = TrainingArguments(
            output_dir=str(output / "trainer_state"),
            learning_rate=float(optimizer["learning_rate"]),
            weight_decay=float(optimizer["weight_decay"]),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=int(budget["gradient_accumulation"]),
            max_steps=int(budget["steps"]),
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            seed=seed,
            data_seed=seed,
            optim="adamw_torch",
        )
        trainer = Trainer(
            model=adapter_model,
            args=arguments,
            train_dataset=Dataset.from_list(features),
            data_collator=fixed_collator,
        )
        train_result = trainer.train()
        observed_steps = int(trainer.state.global_step)
        if observed_steps != int(budget["steps"]):
            raise StudyBError("Trainer did not execute the exact registered optimizer steps")
        adapter = output / "final_adapter"
        adapter_model.save_pretrained(str(adapter))
        log_history = list(trainer.state.log_history)
        _write_json_new(output / "training_log.json", log_history)
        return {
            "adapter_path": str(adapter),
            "observed_target_tokens": observed_target_tokens,
            "training_metrics": {
                "train_steps": observed_steps,
                "train_loss": float(train_result.metrics.get("train_loss", math.nan)),
                "observed_total_flos": float(getattr(trainer.state, "total_flos", 0.0)),
                "registered_approximate_flops": float(budget["approximate_flops"]),
                "fixed_padded_sequence_length": self.max_sequence_length,
                "per_device_train_batch_size": 1,
            },
            "trainable_manifest": trainable,
            "frozen_hashes": frozen,
        }

    def evaluate(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        prompts: tuple[str, ...],
        seed: int,
        output: Path,
    ) -> Iterable[Mapping[str, object]]:
        import torch

        model, processor = session["model"], session["processor"]
        tokenizer = getattr(processor, "tokenizer", processor)
        model.eval()
        for row, prompt in zip(rows, prompts, strict=True):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            device = getattr(model, "device", "cuda:0")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            completion = tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ).strip()
            yield {"scene_id": row["scene_id"], "completion": completion}

    def release(self, session: Mapping[str, object]) -> None:
        if isinstance(session, dict):
            session.clear()
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def require_offline_environment(environment: Mapping[str, str] | None = None) -> None:
    current = os.environ if environment is None else environment
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [name for name in required if current.get(name) != "1"]
    if missing:
        raise StudyBError("offline environment is incomplete: " + ", ".join(missing))


def _require_single_4090() -> None:  # pragma: no cover - requires CUDA
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise StudyBError("Study B requires exactly one visible CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if "4090" not in device_name:
        raise StudyBError(f"Study B pilot requires a 4090, observed {device_name}")
    if not torch.cuda.is_bf16_supported():
        raise StudyBError("Study B requires bf16 support")


def verify_runtime_package_lock(path: Path) -> dict[str, object]:
    """Verify the exact Python/GPU dependency lock and single-4090 boundary."""

    import yaml

    if path.is_symlink() or not path.is_file():
        raise StudyBError("Study B package lock must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "python",
        "cuda",
        "packages",
    }:
        raise StudyBError("Study B package lock has a malformed closed schema")
    if payload.get("schema_version") != 1:
        raise StudyBError("Study B package lock schema version differs")
    python_version = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
    if payload.get("python") != python_version:
        raise StudyBError(f"Python version differs from lock: observed {python_version}")
    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or not packages:
        raise StudyBError("Study B package lock has no exact package versions")
    for name, expected in packages.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise StudyBError("Study B package lock package entries must be strings")
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise StudyBError(f"locked GPU package is missing: {name}") from error
        if observed != expected:
            raise StudyBError(
                f"locked GPU package version mismatch for {name}: {observed} != {expected}"
            )
    import torch

    observed_cuda = getattr(torch.version, "cuda", None)
    if observed_cuda != payload.get("cuda"):
        raise StudyBError(f"CUDA version differs from lock: observed {observed_cuda}")
    _require_single_4090()
    return payload


__all__ = ["QwenStudyBBackend", "require_offline_environment", "verify_runtime_package_lock"]
