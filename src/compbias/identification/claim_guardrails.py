"""Language-level guardrails for partially identified compensation claims."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACQUISITION = re.compile(
    r"(?:improv\w*|learn\w*|enhanc\w*).{0,32}acquisition|acquisition.{0,32}(?:improv\w*|learn\w*|enhanc\w*)",
    re.IGNORECASE,
)
_UNIQUE_INTERNAL = re.compile(
    r"(?:true|unique|actual)\s+(?:internal\s+)?(?:perception|reasoning)|"
    r"(?:perception|reasoning)\s+(?:module|boundary)\s+(?:is|was)\s+(?:true|unique)",
    re.IGNORECASE,
)
_OPERATIONAL = re.compile(r"operational (?:compensation )?certificate", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    allowed: bool
    reason: str
    replacement: str


def assess_claim(
    claim: str,
    *,
    acquisition_frozen: bool,
    black_box: bool,
) -> ClaimAssessment:
    """Reject claims that exceed the chosen update and observation regime."""

    if not isinstance(claim, str) or not claim.strip() or len(claim) > 4_096:
        raise ValueError("claim must be non-empty text no longer than 4096 characters")
    if not isinstance(acquisition_frozen, bool) or not isinstance(black_box, bool):
        raise TypeError("regime flags must be booleans")
    if acquisition_frozen and _ACQUISITION.search(claim):
        return ClaimAssessment(
            allowed=False,
            reason="visual acquisition is frozen and cannot be claimed to improve",
            replacement="Report readout, reasoning, or interaction change with acquisition fixed.",
        )
    if black_box and _UNIQUE_INTERNAL.search(claim):
        return ClaimAssessment(
            allowed=False,
            reason="black-box behavior does not identify a unique internal factorization",
            replacement=(
                "Report an operational compensation certificate under a pre-registered "
                "interface family."
            ),
        )
    if black_box and "compensation" in claim.lower() and _OPERATIONAL.search(claim) is None:
        return ClaimAssessment(
            allowed=False,
            reason="black-box compensation claims must be explicitly operational",
            replacement=(
                "Report an operational compensation certificate under a pre-registered "
                "interface family."
            ),
        )
    return ClaimAssessment(
        allowed=True, reason="claim matches the evidence boundary", replacement=""
    )
