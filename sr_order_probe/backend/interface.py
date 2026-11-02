"""Contract only. Codex must bind the existing read-only native F6 generation path.
This file deliberately does not import or construct an optimizer.
"""
from __future__ import annotations
from typing import Protocol, Any

class BoundReadOnlyBackend(Protocol):
    def identity(self) -> dict[str, Any]:
        """Actual loaded snapshot, adapter digest, processor/tokenizer and mode."""
        ...
    def generate_one(self, request: dict[str, Any]) -> dict[str, Any]:
        """Only public prompt/image/prefill are supplied to the model.
        Return suffix_text, suffix_token_ids, finish_reason, prompt_input_ids,
        prefill_boundary metadata, sampling token/probability trace, timing,
        and consumed image digest. Preserve all invalid model responses.
        """
        ...
    def close(self) -> None: ...


def make_backend(*args, **kwargs):
    raise NotImplementedError(
        'SERVER_BINDING_REQUIRED: adapt the already-validated F6 FP32 inference runtime; '
        'do not substitute a mock, new hosted model, or fresh training checkpoint.')
