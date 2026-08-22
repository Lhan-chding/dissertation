"""Pinned TRL/Qwen backend for Study C2 Stage 25."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.model_loader import load_pinned_qwen
from compensability_v4.training.phase4 import freeze_base_parameters

from .training_runtime import build_grpo_config_kwargs, truncate_first_line_token_ids


def validate_training_backend_api() -> dict[str, object]:  # pragma: no cover - server lock
    """Reject TRL versions that cannot preserve the registered scientific contract."""

    from trl import GRPOConfig, GRPOTrainer

    config_parameters = set(inspect.signature(GRPOConfig).parameters)
    trainer_parameters = set(inspect.signature(GRPOTrainer).parameters)
    required_config = {
        "output_dir",
        "learning_rate",
        "num_train_epochs",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_completion_length",
        "bf16",
        "temperature",
        "top_p",
        "beta",
        "generation_kwargs",
        "shuffle_dataset",
    }
    required_trainer = {
        "model",
        "reward_funcs",
        "args",
        "train_dataset",
        "processing_class",
        "callbacks",
    }
    missing_config = sorted(required_config - config_parameters)
    missing_trainer = sorted(required_trainer - trainer_parameters)
    if missing_config or missing_trainer:
        raise RuntimeError(
            "installed TRL GRPO API differs from Study C2: "
            f"missing_config={missing_config}, missing_trainer={missing_trainer}"
        )
    source = inspect.getsource(GRPOTrainer.__init__)
    reference_copy = 'add_adapter("ref"' in source and "ref_param.data.copy_(param.data)" in source
    if not reference_copy:
        raise RuntimeError("installed TRL does not freeze a copied pretrained PEFT reference")
    if not hasattr(GRPOTrainer, "_generate_single_turn"):
        raise RuntimeError("installed TRL lacks the registered generation override seam")
    return {
        "grpo_config_required_fields": True,
        "grpo_trainer_required_fields": True,
        "reference_adapter_copy": True,
        "first_line_generation_override": True,
    }


def _tokenizer(processor: object) -> object:
    return getattr(processor, "tokenizer", processor)


def _newline_and_eos(processor: object) -> tuple[int, int]:
    tokenizer = _tokenizer(processor)
    encode = getattr(tokenizer, "encode", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if not callable(encode) or type(eos) is not int:
        raise RuntimeError("Study C2 tokenizer lacks encode/EOS support")
    newline = encode("\n", add_special_tokens=False)
    if not isinstance(newline, list) or len(newline) != 1 or type(newline[0]) is not int:
        raise RuntimeError("Study C2 newline is not one tokenizer token")
    return int(newline[0]), eos


def _validate_prompt_lengths(
    processor: object,
    dataset: Sequence[Mapping[str, object]],
    *,
    maximum: int,
) -> int:
    render = getattr(processor, "apply_chat_template", None)
    if not callable(render):
        render = getattr(_tokenizer(processor), "apply_chat_template", None)
    if not callable(render):
        raise RuntimeError("Study C2 processor lacks chat-template tokenization")
    observed: list[int] = []
    for row in dataset:
        prompt = row.get("prompt")
        ids = render(prompt, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise RuntimeError("Study C2 chat template returned malformed token IDs")
        if len(ids) > maximum:
            raise RuntimeError(f"Study C2 prompt exceeds {maximum} tokens: {row.get('scene_id')}")
        observed.append(len(ids))
    if not observed:
        raise RuntimeError("Study C2 training dataset is empty")
    return max(observed)


def _verify_reference_copy(model: object) -> None:
    import torch

    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or not {"default", "ref"}.issubset(configs):
        raise RuntimeError("TRL did not install the frozen B3 reference adapter")
    named = dict(model.named_parameters())  # type: ignore[attr-defined]
    default_names = [name for name in named if ".default." in name]
    if not default_names:
        raise RuntimeError("Study C2 trainable B3 adapter parameters are missing")
    for name in default_names:
        reference_name = name.replace(".default.", ".ref.")
        if reference_name not in named or not torch.equal(
            named[name].detach(), named[reference_name].detach()
        ):
            raise RuntimeError("TRL B3 reference adapter differs before the first optimizer step")
        if named[reference_name].requires_grad:
            raise RuntimeError("TRL B3 reference adapter is unexpectedly trainable")


def create_training_trainer(
    *,
    arm_config: Mapping[str, object],
    b3_adapter: Path,
    dataset: Sequence[Mapping[str, object]],
    reward_function: object,
    output_dir: Path,
    group_size: int,
    callbacks: Sequence[object],
) -> object:  # pragma: no cover - pinned server GPU path
    validate_training_backend_api()
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    base, processor = load_pinned_qwen(device_map="cuda:0")
    training = arm_config.get("training")
    if not isinstance(training, Mapping):
        raise RuntimeError("Study C2 arm lacks its training contract")
    max_observed = _validate_prompt_lengths(
        processor,
        dataset,
        maximum=int(training["max_prompt_length"]),
    )
    freeze_base_parameters(base)
    model = PeftModel.from_pretrained(base, str(b3_adapter), is_trainable=True)
    newline_token_id, eos_token_id = _newline_and_eos(processor)
    supported = tuple(inspect.signature(GRPOConfig).parameters)
    arguments = GRPOConfig(
        **build_grpo_config_kwargs(
            arm_config=arm_config,
            output_dir=output_dir,
            group_size=group_size,
            eos_token_id=eos_token_id,
            newline_token_id=newline_token_id,
            supported_parameters=supported,
        )
    )

    class FirstLineGRPOTrainer(GRPOTrainer):
        def _generate_single_turn(self, prompt_ids, images, multimodal_fields):  # type: ignore[no-untyped-def]
            completion_ids, logprobs = super()._generate_single_turn(
                prompt_ids, images, multimodal_fields
            )
            return truncate_first_line_token_ids(
                completion_ids=completion_ids,
                logprobs=logprobs,
                newline_token_id=newline_token_id,
                eos_token_id=eos_token_id,
            )

    trainer = FirstLineGRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=arguments,
        train_dataset=Dataset.from_list([dict(row) for row in dataset]),
        processing_class=processor,
        callbacks=list(callbacks),
    )
    _verify_reference_copy(model)
    trainer._study_c2_runtime_audit = {
        "max_prompt_length": int(training["max_prompt_length"]),
        "max_prompt_tokens_observed": max_observed,
        "newline_token_id": newline_token_id,
        "eos_token_id": eos_token_id,
        "reference_adapter_copy": True,
    }
    return trainer


__all__ = ["create_training_trainer", "validate_training_backend_api"]
