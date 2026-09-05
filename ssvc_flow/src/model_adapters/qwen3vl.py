"""Technical fallback; selection requires documented main-model incompatibility."""

from .base import HuggingFaceAdapter


class Qwen3VLAdapter(HuggingFaceAdapter):
    model_class = "Qwen3VLForConditionalGeneration"
