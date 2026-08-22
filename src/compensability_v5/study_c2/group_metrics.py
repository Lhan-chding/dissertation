"""Corrective excitation and per-rollout K efficiency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


def exact_shortcut_group_probability(p_exact: float, p_shortcut: float, k: int) -> float:
    if not 0.0 <= p_exact <= 1.0 or not 0.0 <= p_shortcut <= 1.0:
        raise ValueError("support masses must be probabilities")
    if p_exact + p_shortcut > 1.0 or k <= 0:
        raise ValueError("support masses/group size are invalid")
    return 1.0 - (1.0 - p_exact) ** k - (1.0 - p_shortcut) ** k + (1.0 - p_exact - p_shortcut) ** k


def summarize_group(kinds: Sequence[str]) -> dict[str, object]:
    counts = Counter(kinds)
    if not kinds or set(counts) - {"X", "S", "F", "U"}:
        raise ValueError("group must contain X/S/F/U labels")
    answer = [kind in {"X", "S"} for kind in kinds]
    state = [kind == "X" for kind in kinds]
    return {
        "AIGR": len(set(answer)) > 1,
        "SIGR": len(set(state)) > 1,
        "RDGR": counts["S"] > 0,
        "ESGR": counts["X"] > 0 and counts["S"] > 0,
        "counts": {kind: counts[kind] for kind in ("X", "S", "F", "U")},
    }


def choose_group_size(
    support: Sequence[Mapping[str, object]], *, candidates: Sequence[int] = (8, 16, 32)
) -> dict[str, object]:
    if not support:
        raise ValueError("support audit cannot be empty")
    efficiencies: dict[int, float] = {}
    for k in candidates:
        values = [
            exact_shortcut_group_probability(float(row["p_X"]), float(row["p_S"]), k) / k
            for row in support
        ]
        efficiencies[int(k)] = sum(values) / len(values)
    best = max(efficiencies.values())
    selected = min(k for k, value in efficiencies.items() if value == best)
    return {"selected_k": selected, "efficiency_by_k": efficiencies}


__all__ = ["choose_group_size", "exact_shortcut_group_probability", "summarize_group"]
