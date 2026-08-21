"""Fail-closed construction of the local v5 advisor packet."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

_REQUIRED_RESULTS = ("support_results", "confirmation_results", "reward_results")


def _complete(payload: object) -> bool:
    return isinstance(payload, Mapping) and payload.get("complete") is True


def _registered_stop(payload: object) -> bool:
    if not isinstance(payload, Mapping) or payload.get("complete") is not True:
        return False
    stop_signal = payload.get("stop_signal")
    return (
        isinstance(stop_signal, Mapping)
        and stop_signal.get("triggered") is True
        and isinstance(stop_signal.get("rule"), str)
        and bool(stop_signal["rule"])
    )


def build_advisor_packet(results: Mapping[str, object]) -> dict[str, object]:
    """Build a compact advisor packet or return a status-only blocker."""

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping")
    unknown = set(results) - set(_REQUIRED_RESULTS)
    if unknown:
        raise ValueError(f"unregistered result payloads: {sorted(map(str, unknown))}")
    missing = sorted(key for key in _REQUIRED_RESULTS if not _complete(results.get(key)))
    if (
        missing == ["reward_results"]
        and _registered_stop(results.get("support_results"))
        and _complete(results.get("confirmation_results"))
    ):
        return {
            "schema_version": 1,
            "status": "PARTIAL_DECISIVE_PILOT",
            "study_c_status": "NOT_RUN_DUE_TO_REGISTERED_STOP",
            "results": deepcopy(
                {
                    "support_results": results["support_results"],
                    "confirmation_results": results["confirmation_results"],
                }
            ),
        }
    if missing:
        return {
            "schema_version": 1,
            "status": "BLOCKED_MISSING_RESULTS",
            "missing_results": missing,
        }
    return {
        "schema_version": 1,
        "status": "ADVISOR_PACKET_READY",
        "results": deepcopy({key: results[key] for key in _REQUIRED_RESULTS}),
    }
