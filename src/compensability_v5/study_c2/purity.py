"""Policy-weighted answer-fiber truth purity."""

from __future__ import annotations

from collections.abc import Mapping


def fiber_purity(*, p_exact: float, p_shortcut: float) -> float | None:
    if p_exact < 0 or p_shortcut < 0 or p_exact + p_shortcut > 1:
        raise ValueError("fiber masses must be valid probabilities")
    denominator = p_exact + p_shortcut
    return None if denominator == 0 else p_exact / denominator


def purity_update(
    *, before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float | None]:
    before_x, before_s = float(before["X"]), float(before["S"])
    after_x, after_s = float(after["X"]), float(after["S"])
    before_purity = fiber_purity(p_exact=before_x, p_shortcut=before_s)
    after_purity = fiber_purity(p_exact=after_x, p_shortcut=after_s)
    return {
        "delta_exact_mass": after_x - before_x,
        "delta_shortcut_mass": after_s - before_s,
        "delta_answer_success_mass": (after_x + after_s) - (before_x + before_s),
        "before_fiber_purity": before_purity,
        "after_fiber_purity": after_purity,
        "delta_fiber_purity": None
        if before_purity is None or after_purity is None
        else after_purity - before_purity,
    }


__all__ = ["fiber_purity", "purity_update"]
