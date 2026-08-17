"""Layerwise logit validation helpers."""

from __future__ import annotations


def validate_final_layer_logits(layerwise_logits, forward_logits) -> None:
    if not layerwise_logits or layerwise_logits[-1] != forward_logits:
        raise RuntimeError("final-layer candidate logits do not match the standard forward pass")
