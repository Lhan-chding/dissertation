"""TRL/PEFT executor imported only after explicit GPU acknowledgement."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from compbias.models.structured_parser import ParseStatus, parse_trajectory

from .config import load_pilot_paths
from .qwen_smoke import load_local_qwen


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        content = completion[-1].get("content")
        if isinstance(content, str):
            return content
    return ""


def outcome_reward(completions: list[object], answer: list[object], **_: object) -> list[float]:
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        parsed = parse_trajectory(_completion_text(completion), sample_id=f"reward-{index}")
        expected = answer[index]
        rewards.append(
            1.0
            if parsed.status is ParseStatus.OK
            and str(parsed.answer).strip() == str(expected).strip()
            else 0.0
        )
    return rewards


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


def _prompt_for_a(record: dict[str, Any]) -> str:
    mediator = record.get("natural_mediator_raw")
    question = record.get("question")
    return (
        "The image is unavailable. Treat the following naturally produced visual evidence "
        "as fixed. "
        f"Evidence: {mediator}\nQuestion: {question}\n"
        "Return <perception>{...}</perception><reasoning>{...}</reasoning><answer>...</answer>."
    )


def _prompt_for_b(record: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{record['question']} Return exactly tagged JSON perception, reasoning, "
                        "and answer fields."
                    ),
                },
            ],
        }
    ]


def run_grpo_stage(config: dict[str, object]) -> None:
    import torch
    from datasets import Dataset, Image
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    project_root = Path(__file__).resolve().parents[3]
    paths_config = project_root / str(config["paths_config"])
    paths = load_pilot_paths(paths_config)
    stage = str(config["stage"])
    training = config["training"]
    assert isinstance(training, dict)
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
        rows = [
            {"prompt": _prompt_for_a(record), "answer": record["answer"]}
            for record in records
            if record.get("error_type") not in {None, "none"}
        ]
    elif stage == "pilot_b_lm_only":
        manifest = paths.project_root / str(config["dataset_manifest"])
        records = _read_jsonl(manifest.parent / "records.jsonl")
        rows = [
            {
                "prompt": _prompt_for_b(record),
                "image": str(manifest.parent / str(record["image"])),
                "answer": record["answer"],
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
    output_dir = paths.outputs / str(config["output_subdir"])
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
        reward_funcs=outcome_reward,
        args=args,
        train_dataset=dataset,
        peft_config=lora,
    )
    torch.cuda.reset_peak_memory_stats(0)
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
