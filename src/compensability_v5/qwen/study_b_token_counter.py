"""Exact Qwen completion token counting for the frozen Study-B support package."""

from __future__ import annotations

import os
from functools import lru_cache

from compensability_v4.qwen.model_loader import MODEL_PATH, require_server_model


def _load_tokenizer() -> object:
    verified = require_server_model(MODEL_PATH)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(verified),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Qwen tokenizer exposes no exact encode() method")
    return tokenizer


@lru_cache(maxsize=1)
def _get_tokenizer() -> object:
    return _load_tokenizer()


def count_completion_tokens(text: str) -> int:
    if not isinstance(text, str) or not text:
        raise ValueError("completion text must be a non-empty string")
    tokenizer = _get_tokenizer()
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("cached tokenizer exposes no exact encode() method")
    token_ids = encode(text, add_special_tokens=False)
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(not isinstance(value, int) or isinstance(value, bool) for value in token_ids)
    ):
        raise RuntimeError("Qwen tokenizer returned malformed token ids")
    return len(token_ids)


__all__ = ["count_completion_tokens"]
