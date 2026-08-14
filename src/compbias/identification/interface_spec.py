"""Explicit operational mediator interface declarations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _component(value: object, name: str) -> str:
    if not isinstance(value, str) or _COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded safe component")
    return value


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """An operational replay boundary, not a unique anatomical decomposition."""

    interface_id: str
    mode: Literal["behavioral", "activation"]
    boundary: str
    image_cut_mode: str
    parser_id: str
    oracle_serializer_id: str

    def __post_init__(self) -> None:
        for name in (
            "interface_id",
            "boundary",
            "image_cut_mode",
            "parser_id",
            "oracle_serializer_id",
        ):
            object.__setattr__(self, name, _component(getattr(self, name), name))
        if self.mode not in {"behavioral", "activation"}:
            raise ValueError("mode must be behavioral or activation")

    def to_mapping(self) -> dict[str, str]:
        return asdict(self)
