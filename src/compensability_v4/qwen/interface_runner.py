"""Interface naming discipline for v4."""

from __future__ import annotations

from enum import Enum


class InterfaceName(str, Enum):
    I0_HARD_TEXT = "I0_hard_text"
    I3_SAME_CONVERSATION = "I3_same_conversation"
    I4_EXACT_CACHE = "I4_exact_cache"


def interface_family(interface: InterfaceName) -> str:
    if interface is InterfaceName.I0_HARD_TEXT:
        return "symbolic_downstream_recovery"
    return "natural_visual_revision"
