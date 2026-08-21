"""Independent Phase-8 observation-error cardinality and magnitude audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .fiber_multiplicity import validate_domain, validate_world

RawRow = Mapping[str, Any]


def _unique_rows(rows: Iterable[RawRow], *, label: str) -> dict[str, RawRow]:
    keyed: dict[str, RawRow] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id in keyed:
            raise ValueError(f"duplicate {label} scene_id: {scene_id}")
        keyed[scene_id] = row
    if not keyed:
        raise ValueError(f"{label} rows must not be empty")
    return keyed


def phase8_error_summary(
    scene_rows: Iterable[RawRow],
    observation_rows: Iterable[RawRow],
    *,
    value_domain: Iterable[int] = range(2, 19),
) -> dict[str, Any]:
    """Join frozen truth and observation rows and recompute every error statistic."""

    scenes = _unique_rows(scene_rows, label="scene")
    observations = _unique_rows(observation_rows, label="observation")
    if scenes.keys() != observations.keys():
        missing_observations = sorted(scenes.keys() - observations.keys())
        missing_scenes = sorted(observations.keys() - scenes.keys())
        raise ValueError(
            "Phase-8 scene/observation IDs differ: "
            f"missing_observations={missing_observations[:5]}, missing_scenes={missing_scenes[:5]}"
        )
    domain = frozenset(validate_domain(value_domain))
    histogram: Counter[str] = Counter()
    error_counts: list[int] = []
    l1_errors: list[int] = []
    out_of_domain = 0
    zero_error_count = 0
    per_scene: list[dict[str, Any]] = []
    for scene_id in sorted(scenes):
        truth = validate_world(scenes[scene_id]["truth"], label="truth")
        observation_row = observations[scene_id]
        observed_value = observation_row.get("observed_values", observation_row.get("observed"))
        observed = validate_world(observed_value, label="observed_values")
        error_indices = tuple(
            index
            for index, pair in enumerate(zip(truth, observed, strict=True))
            if pair[0] != pair[1]
        )
        l1_error = sum(
            abs(expected - actual) for expected, actual in zip(truth, observed, strict=True)
        )
        is_ood = any(value not in domain for value in observed)
        error_count = len(error_indices)
        if error_count == 0:
            zero_error_count += 1
            continue
        histogram[str(error_count)] += 1
        error_counts.append(error_count)
        l1_errors.append(l1_error)
        out_of_domain += is_ood
        per_scene.append(
            {
                "scene_id": scene_id,
                "error_count": error_count,
                "error_indices": list(error_indices),
                "l1_error": l1_error,
                "observation_out_of_domain": is_ood,
            }
        )
    if not error_counts:
        raise ValueError("Phase-8 rows contain no natural-error scenes")
    return {
        "input_scene_count": len(scenes),
        "scene_count": len(error_counts),
        "zero_error_scene_count_excluded": zero_error_count,
        "error_count_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "mean_error_count": sum(error_counts) / len(error_counts),
        "max_l1_error": max(l1_errors),
        "mean_l1_error": sum(l1_errors) / len(l1_errors),
        "out_of_domain_scene_count": out_of_domain,
        "out_of_domain_scene_rate": out_of_domain / len(error_counts),
        "value_domain": sorted(domain),
        "per_scene": per_scene,
    }


def confirm_error_cardinality_summary(
    rows: Iterable[RawRow], *, in_domain: Iterable[int] = range(2, 19)
) -> dict[str, Any]:
    """Compatibility helper for already joined truth/observation fixtures."""

    frozen_rows = tuple(rows)
    if not frozen_rows:
        raise ValueError("rows must not be empty")
    if all("truth" in row for row in frozen_rows):
        scenes = (
            {"scene_id": str(index), "truth": row["truth"]} for index, row in enumerate(frozen_rows)
        )
        observations = (
            {
                "scene_id": str(index),
                "observed_values": row.get("observed_values", row.get("observed")),
            }
            for index, row in enumerate(frozen_rows)
        )
        return phase8_error_summary(scenes, observations, value_domain=in_domain)
    allowed = frozenset(validate_domain(in_domain))
    histogram: Counter[str] = Counter()
    out_of_domain = 0
    for row in frozen_rows:
        indices = row["error_indices"]
        if isinstance(indices, (str, bytes)) or not isinstance(indices, (list, tuple)):
            raise TypeError("error_indices must be a sequence")
        observed = validate_world(row.get("observed_values", row.get("observed")), label="observed")
        histogram[str(len(indices))] += 1
        out_of_domain += any(value not in allowed for value in observed)
    return {
        "error_count_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "out_of_domain_scene_count": out_of_domain,
    }


__all__ = ["confirm_error_cardinality_summary", "phase8_error_summary"]
