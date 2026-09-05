"""Lazy model registry; CPU audit commands never import GPU dependencies."""

from .qwen3vl import Qwen3VLAdapter
from .qwen25vl import Qwen25VLAdapter
from .qwen35 import Qwen35Adapter

ADAPTERS = {
    "qwen35_9b": Qwen35Adapter,
    "qwen25vl_3b": Qwen25VLAdapter,
    "qwen25vl_7b": Qwen25VLAdapter,
    "qwen3vl_8b": Qwen3VLAdapter,
}


def load_adapter(model_key, model_spec, **kwargs):
    if model_key not in ADAPTERS:
        raise ValueError(f"Unsupported model key: {model_key}")
    return ADAPTERS[model_key].load(model_spec, **kwargs)
