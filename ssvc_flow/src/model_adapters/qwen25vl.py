"""Historical anchor and same-generation bridge use official input adaptation."""

from .base import HuggingFaceAdapter


class Qwen25VLAdapter(HuggingFaceAdapter):
    model_class = "Qwen2_5_VLForConditionalGeneration"
