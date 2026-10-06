"""Task-cluster statistics conditional on the registered trained models."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist, mean


def pass_at_k(successes, draws, k):
    if type(successes) is not int or not 0 <= successes <= draws or not 1 <= k <= draws:
        raise ValueError("Invalid finite-stream pass@k counts")
    return 1.0 - (
        math.comb(draws - successes, k) / math.comb(draws, k) if draws - successes >= k else 0.0
    )


def wilson(successes, draws, confidence=0.95):
    if draws < 1 or not 0 <= successes <= draws:
        raise ValueError("Invalid Wilson counts")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / draws
    center = (p + z * z / (2 * draws)) / (1 + z * z / draws)
    half = z * math.sqrt(p * (1 - p) / draws + z * z / (4 * draws * draws)) / (1 + z * z / draws)
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_cluster_bootstrap(rows, *, replicates=5000, seed=106069):
    """Rows contain family, base_instance_id, unit, left, right (probabilities).

    Every scene's parents/repeats and shared E/G interfaces move together.
    Missing units are an error; they must not silently alter the estimand.
    """
    if not rows or replicates < 1:
        raise ValueError("Nonempty paired observations required")
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        unit = row["unit"]
        scene = row["base_instance_id"]
        family = row["family"]
        if unit in groups[family][scene]:
            raise ValueError("Duplicate scene/unit")
        left, right = float(row["left"]), float(row["right"])
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (left, right)):
            raise ValueError("Nonfinite or out-of-range estimate")
        groups[family][scene][unit] = left - right
    units = {row["unit"] for row in rows}
    if any(set(values) != units for scenes in groups.values() for values in scenes.values()):
        raise ValueError("Incomplete paired parent/repeat panel")
    clusters = [[mean(values.values()) for values in scenes.values()] for scenes in groups.values()]
    estimate = mean(mean(values) for values in clusters)
    rng = random.Random(seed)
    draws = sorted(
        mean(mean(rng.choices(values, k=len(values))) for values in clusters)
        for _ in range(replicates)
    )
    return {
        "difference": estimate,
        "interval_95": [draws[int(0.025 * (replicates - 1))], draws[int(0.975 * (replicates - 1))]],
        "replicates": replicates,
        "scene_count": sum(map(len, clusters)),
        "unit_count": len(units),
        "scope": (
            "conditional on executed checkpoints; sce"
            "ne-cluster uncertainty, not a seed-popul"
            "ation interval"
        ),
    }


def finite_panel_mcse(rows):
    """Aggregate Bernoulli sampling variance with shared-stream covariance.

    Each row: stream_id, successes, draws, coefficient. Repeated stream IDs
    denote the *same* raw samples; their signed weights combine before variance.
    """
    streams = {}
    for row in rows:
        key = row["stream_id"]
        count = (row["successes"], row["draws"])
        if count[1] < 2 or not 0 <= count[0] <= count[1]:
            raise ValueError("At least two draws required")
        if key in streams and streams[key]["count"] != count:
            raise ValueError("Shared raw stream has inconsistent counts")
        item = streams.setdefault(key, {"count": count, "weight": 0.0})
        item["weight"] += row["coefficient"]
    variance = 0.0
    boundary = []
    for key, item in streams.items():
        c, n = item["count"]
        p = c / n
        variance += item["weight"] ** 2 * p * (1 - p) / (n - 1)
        if c in (0, n):
            boundary.append({"stream_id": key, "unseen_success": c == 0, "wilson_95": wilson(c, n)})
    return {
        "mcse": math.sqrt(variance),
        "boundary_streams": boundary,
        "scope": "fixed-panel sampling uncertainty; zero plug-in variance is not certainty",
    }
