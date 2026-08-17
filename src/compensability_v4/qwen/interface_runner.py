"""Qwen interface-ladder dispatch with strict scientific claim boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class InterfaceName(str, Enum):
    I0_HARD_TEXT = "I0_hard_text"
    I1_SOFT_REPORT = "I1_soft_report"
    I2_CANDIDATE_WORLD = "I2_candidate_world"
    I3_SAME_CONVERSATION = "I3_same_conversation"
    I4_EXACT_CACHE = "I4_exact_cache"


def interface_family(interface: InterfaceName) -> str:
    if interface is InterfaceName.I0_HARD_TEXT:
        return "symbolic_downstream_recovery"
    if interface in {InterfaceName.I3_SAME_CONVERSATION, InterfaceName.I4_EXACT_CACHE}:
        return "natural_visual_revision"
    return "intervention_diagnostic"


@dataclass(frozen=True, slots=True)
class InterfaceResult:
    """One interface output with an explicit main-result eligibility bit."""

    sample_id: str
    interface: InterfaceName
    family: str
    output_text: str
    parity_verified: bool
    primary_eligible: bool
    metadata: Mapping[str, Any]


class InterfaceRunner:
    """Dispatch explicit I0--I4 implementations without conflating claims."""

    def __init__(self, handlers: Mapping[InterfaceName, Callable[..., object]]) -> None:
        self._handlers = dict(handlers)

    def run(
        self,
        interface: InterfaceName,
        *,
        sample_id: str,
        primary_result: bool = False,
        **kwargs: object,
    ) -> InterfaceResult:
        if not isinstance(interface, InterfaceName):
            raise TypeError("interface must be an InterfaceName")
        handler = self._handlers.get(interface)
        if handler is None:
            raise RuntimeError(f"no runtime handler configured for {interface.value}")
        raw = handler(sample_id=sample_id, **kwargs)
        if isinstance(raw, Mapping):
            output_text = raw.get("text")
            parity_verified = raw.get("parity_verified", False) is True
            metadata = dict(raw)
        else:
            output_text = raw
            parity_verified = False
            metadata = {}
        if not isinstance(output_text, str):
            raise RuntimeError("interface handler did not return text")
        if interface is InterfaceName.I4_EXACT_CACHE and primary_result and not parity_verified:
            raise RuntimeError("I4 primary execution is blocked until cache parity is verified")
        eligible = interface not in {
            InterfaceName.I1_SOFT_REPORT,
            InterfaceName.I2_CANDIDATE_WORLD,
        }
        return InterfaceResult(
            sample_id=sample_id,
            interface=interface,
            family=interface_family(interface),
            output_text=output_text,
            parity_verified=parity_verified,
            primary_eligible=eligible
            and (interface is not InterfaceName.I4_EXACT_CACHE or parity_verified),
            metadata=metadata,
        )


__all__ = ["InterfaceName", "InterfaceResult", "InterfaceRunner", "interface_family"]
