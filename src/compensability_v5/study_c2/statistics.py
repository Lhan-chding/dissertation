"""Scene-paired bootstrap estimates for the registered Study C2 contrast."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def paired_collision_difference_in_differences(
    rows: Sequence[Mapping[str, object]], *, resamples: int, seed: int
) -> dict[str, object]:
    cells: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        cells[str(row["pair_id"])][(str(row["condition"]), str(row["arm"]))] = float(row["exact"])
    required = {
        ("collision", "state"),
        ("collision", "answer"),
        ("separating", "state"),
        ("separating", "answer"),
    }
    effects: list[float] = []
    for pair_id, values in sorted(cells.items()):
        if set(values) != required:
            raise ValueError(f"pair {pair_id} lacks a complete reward/condition grid")
        effects.append(
            values[("collision", "state")]
            - values[("collision", "answer")]
            - values[("separating", "state")]
            + values[("separating", "answer")]
        )
    if not effects or resamples <= 0:
        raise ValueError("paired bootstrap requires pairs and positive resamples")
    estimate = sum(effects) / len(effects)
    rng = random.Random(seed)
    draws = [sum(rng.choice(effects) for _ in effects) / len(effects) for _ in range(resamples)]
    return {
        "estimate": estimate,
        "pair_count": len(effects),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_95_ci": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


__all__ = ["paired_collision_difference_in_differences"]
