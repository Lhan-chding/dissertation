"""Qwen3.5 uses the explicit official multimodal class, never a guessed VL ID."""

from .base import HuggingFaceAdapter


class Qwen35Adapter(HuggingFaceAdapter):
    model_class = "Qwen3_5ForConditionalGeneration"
    requires_thinking_switch = True
