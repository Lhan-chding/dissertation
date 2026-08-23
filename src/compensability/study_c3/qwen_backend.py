"""Pinned offline Qwen tokenizer, free decoder, and valid-world FSA decoder."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.model_loader import MODEL_PATH, load_pinned_qwen, require_server_model
from compensability_v4.training.phase4 import freeze_base_parameters
from compensability_v5.qwen.study_b_backend import (
    require_offline_environment,
    verify_runtime_package_lock,
)

from .paths import PACKAGE_LOCK
from .valid_world_trie import ValidWorldTrie


def _tokenizer(processor: object) -> object:
    return getattr(processor, "tokenizer", processor)


def load_pinned_processor() -> object:  # pragma: no cover - pinned server snapshot
    require_offline_environment()
    verified = require_server_model()
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(verified), local_files_only=True, trust_remote_code=False
    )


def load_token_counter() -> object:  # pragma: no cover - pinned server snapshot
    processor = load_pinned_processor()
    encode = getattr(_tokenizer(processor), "encode", None)
    if not callable(encode):
        raise RuntimeError("Study C3 Qwen tokenizer has no encode method")

    def count(text: str) -> int:
        token_ids = encode(text, add_special_tokens=False)
        if not isinstance(token_ids, list) or any(type(token) is not int for token in token_ids):
            raise RuntimeError("Study C3 Qwen tokenizer returned malformed token IDs")
        return len(token_ids)

    return count


def build_pinned_valid_world_trie() -> ValidWorldTrie:  # pragma: no cover - server snapshot
    processor = load_pinned_processor()
    return ValidWorldTrie.build(_tokenizer(processor), minimum=2, maximum=18)


class _TrieLogitsProcessor:
    def __init__(self, trie: ValidWorldTrie, *, prompt_length: int) -> None:
        self.trie = trie
        self.prompt_length = prompt_length
        self.records: list[dict[str, object]] = []

    def __call__(self, input_ids: object, scores: object) -> object:
        import torch

        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or not isinstance(scores, torch.Tensor)
            or scores.ndim != 2
            or len(input_ids) != len(scores)
        ):
            raise RuntimeError("Study C3 logits processor received malformed tensors")
        masked = torch.full_like(scores, float("-inf"))
        for batch_index in range(len(input_ids)):
            prefix = [int(token) for token in input_ids[batch_index, self.prompt_length :].tolist()]
            allowed = sorted(self.trie.allowed_next(prefix))
            allowed_tensor = torch.as_tensor(allowed, dtype=torch.long, device=scores.device)
            masked[batch_index, allowed_tensor] = scores[batch_index, allowed_tensor]
            pre_token = int(torch.argmax(scores[batch_index]).item())
            post_token = int(torch.argmax(masked[batch_index]).item())
            self.records.append(
                {
                    "prefix_token_ids": prefix,
                    "allowed_token_ids": allowed,
                    "mask_before": {
                        "argmax_token_id": pre_token,
                        "argmax_logit": float(scores[batch_index, pre_token].item()),
                        "finite_count": int(torch.isfinite(scores[batch_index]).sum().item()),
                    },
                    "mask_after": {
                        "argmax_token_id": post_token,
                        "argmax_logit": float(masked[batch_index, post_token].item()),
                        "finite_count": int(torch.isfinite(masked[batch_index]).sum().item()),
                    },
                    "masked_token_count": int(scores.shape[-1]) - len(allowed),
                }
            )
        return masked


def _render_prompt(processor: object, prompt: str) -> str:
    render = getattr(processor, "apply_chat_template", None)
    if not callable(render):
        render = getattr(_tokenizer(processor), "apply_chat_template", None)
    if not callable(render):
        raise RuntimeError("Study C3 Qwen processor lacks a chat template")
    rendered = render(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Study C3 Qwen chat template returned invalid text")
    return rendered


def _stop_ids(processor: object) -> tuple[int, int]:
    tokenizer = _tokenizer(processor)
    encode = getattr(tokenizer, "encode", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if not callable(encode) or type(eos) is not int:
        raise RuntimeError("Study C3 Qwen tokenizer lacks EOS/encode support")
    newline = encode("\n", add_special_tokens=False)
    if not isinstance(newline, list) or len(newline) != 1 or type(newline[0]) is not int:
        raise RuntimeError("Study C3 newline must be one Qwen token")
    return int(newline[0]), eos


def _truncate(ids: list[int], *, stop_ids: set[int]) -> list[int]:
    for index, token in enumerate(ids):
        if token in stop_ids:
            return ids[: index + 1]
    return ids


class QwenCheckpointSampler:  # pragma: no cover - pinned CUDA/server path
    def __init__(self, *, adapter_path: Path) -> None:
        verify_runtime_package_lock(PACKAGE_LOCK)
        from peft import PeftModel

        base, self.processor = load_pinned_qwen(device_map="cuda:0")
        freeze_base_parameters(base)
        self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
        self.model.eval()
        self.tokenizer = _tokenizer(self.processor)
        self.newline_token_id, self.eos_token_id = _stop_ids(self.processor)
        self.trie = ValidWorldTrie.build(self.tokenizer, minimum=2, maximum=18)
        self.prepare = self.processor if callable(self.processor) else None
        self.decode = getattr(self.processor, "batch_decode", None)
        if not callable(self.decode):
            self.decode = getattr(self.tokenizer, "batch_decode", None)
        if self.prepare is None or not callable(self.decode):
            raise RuntimeError("Study C3 Qwen processor lacks encode/decode support")

    def __call__(
        self,
        row: Mapping[str, object],
        seeds: Sequence[int],
        decoder: str,
    ) -> tuple[dict[str, object], ...]:
        import torch
        from transformers import LogitsProcessorList

        if decoder not in {"free_16", "free_48", "valid_world_fsa"}:
            raise ValueError(f"unregistered Study C3 decoder: {decoder}")
        rendered = _render_prompt(self.processor, str(row["prompt"]))
        maximum = 16 if decoder == "free_16" else 48
        outputs: list[dict[str, object]] = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            batch = self.prepare(
                text=[rendered],
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            move = getattr(batch, "to", None)
            if callable(move):
                batch = move(self.model.device)
            if not isinstance(batch, Mapping):
                keys = getattr(batch, "keys", None)
                if not callable(keys):
                    raise RuntimeError("Study C3 Qwen batch is not mapping-like")
                batch = {key: batch[key] for key in keys()}
            input_ids = batch.get("input_ids")
            shape = getattr(input_ids, "shape", None)
            if shape is None or len(shape) != 2:
                raise RuntimeError("Study C3 Qwen input IDs are malformed")
            prompt_length = int(shape[1])
            if prompt_length > 512:
                raise RuntimeError("Study C3 Qwen prompt exceeds 512 tokens")
            trie_processor = (
                _TrieLogitsProcessor(self.trie, prompt_length=prompt_length)
                if decoder == "valid_world_fsa"
                else None
            )
            generation_kwargs: dict[str, object] = {
                **dict(batch),
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 1.0,
                "max_new_tokens": maximum,
                "eos_token_id": (
                    self.eos_token_id
                    if decoder == "valid_world_fsa"
                    else [self.eos_token_id, self.newline_token_id]
                ),
                "use_cache": True,
            }
            if trie_processor is not None:
                generation_kwargs["logits_processor"] = LogitsProcessorList([trie_processor])
            with torch.inference_mode():
                generated = self.model.generate(**generation_kwargs)
            token_ids = [int(token) for token in generated[0, prompt_length:].tolist()]
            stop_ids = (
                {self.eos_token_id}
                if decoder == "valid_world_fsa"
                else {self.eos_token_id, self.newline_token_id}
            )
            token_ids = _truncate(token_ids, stop_ids=stop_ids)
            decoded = self.decode(
                [token_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, Sequence) or len(decoded) != 1:
                raise RuntimeError("Study C3 Qwen decoder returned malformed text")
            if trie_processor is not None and any(
                not math.isfinite(float(record["mask_after"]["argmax_logit"]))  # type: ignore[index]
                for record in trie_processor.records
            ):
                raise RuntimeError("Study C3 constrained logit mask produced non-finite choice")
            outputs.append(
                {
                    "completion": str(decoded[0]),
                    "token_ids": token_ids,
                    "mask_records": [] if trie_processor is None else trie_processor.records,
                }
            )
        return tuple(outputs)

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass


def checkpoint_sampler_factory(checkpoint: Mapping[str, object]) -> QwenCheckpointSampler:
    path = checkpoint.get("adapter_path")
    if not isinstance(path, str):
        raise ValueError("Study C3 checkpoint lacks adapter_path")
    return QwenCheckpointSampler(adapter_path=Path(path))


__all__ = [
    "QwenCheckpointSampler",
    "build_pinned_valid_world_trie",
    "checkpoint_sampler_factory",
    "load_pinned_processor",
    "load_token_counter",
]
