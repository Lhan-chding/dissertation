"""Separate scene uncertainty, fixed-panel MC and equal-information replay."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np


def paired_scene_bootstrap(
    rows: Sequence[Mapping],
    *,
    replicates: int = 5000,
    seed: int = 20260919,
    alpha: float = 0.05,
    family_weights: Mapping[str, float] | None = None,
    interfaces: Sequence[str] | None = None,
) -> dict:
    """Bootstrap paired scene means stratified by family, never completions.

    Each row contains family, base_scene, interface and left/right endpoint
    means from all completions for that prompt. Both endpoints and all paired
    interfaces stay together. Default family weights are equal, interfaces
    are equally weighted. No inference across training seeds is implied.
    """
    if type(replicates) is not int or replicates <= 0 or not 0 < alpha < 1 or not rows:
        raise ValueError("Nonempty scenes, positive replicates and valid alpha required")
    if interfaces is None:
        interfaces = sorted({row["interface"] for row in rows})
    if len(interfaces) != 2 or len(set(interfaces)) != 2:
        raise ValueError("The paired two-interface panel is required")
    scenes = defaultdict(dict)
    scene_family = {}
    for row in rows:
        family, scene, interface = row["family"], row["base_scene"], row["interface"]
        if scene in scene_family and scene_family[scene] != family:
            raise ValueError("One base_scene cannot belong to multiple families")
        scene_family[scene] = family
        if interface not in interfaces or interface in scenes[(family, scene)]:
            raise ValueError("Unexpected or duplicate scene/interface")
        left, right = float(row["left"]), float(row["right"])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Endpoint means must be finite")
        scenes[(family, scene)][interface] = left - right
    by_family = defaultdict(list)
    for (family, _), values in sorted(scenes.items()):
        if set(values) != set(interfaces):
            raise ValueError("Missing paired interface")
        by_family[family].append(math.fsum(values.values()) / len(interfaces))
    if family_weights is None:
        family_weights = {key: 1 / len(by_family) for key in by_family}
    if (
        set(family_weights) != set(by_family)
        or any(not math.isfinite(w) or w < 0 for w in family_weights.values())
        or not math.isclose(math.fsum(family_weights.values()), 1, abs_tol=1e-12, rel_tol=0)
    ):
        raise ValueError("Prespecified family weights must match and sum to one")
    rng = np.random.default_rng(seed)
    draws = np.zeros(replicates)
    estimate = 0.0
    for family, scene_values in sorted(by_family.items()):
        values = np.asarray(scene_values)
        estimate += family_weights[family] * float(values.mean())
        # One index samples the entire scene; no second completion resample.
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        draws += family_weights[family] * values[indices].mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": estimate,
        "interval": [float(low), float(high)],
        "replicates": replicates,
        "seed": seed,
        "evidence_kind": "APPROXIMATE_PAIRED_SCENE_BOOTSTRAP",
        "scope": "SCENE_SAMPLING_STRATIFIED_BY_FAMILY",
        "scene_counts": {key: len(values) for key, values in by_family.items()},
        "family_weights": dict(family_weights),
        "paired_interfaces": list(interfaces),
        "nested_completion_resampling": False,
        "training_seed_inference": False,
        "fixed_panel_mc_included": False,
        "finite_sample_certificate": False,
    }


def fixed_panel_mc_error(samples: Sequence[Sequence[float]], *, prompt_weights=None) -> dict:
    """Descriptive MC standard error; zero empirical SE is not certainty.

    Rows may contain different numbers of completions. Use per-draw paired
    differences to preserve a coupled endpoint design, and independent streams
    across prompts. This descriptive SE does not authorize sequential decisions.
    """
    values = [np.asarray(row, dtype=float) for row in samples]
    if not values or any(
        row.ndim != 1 or len(row) < 2 or not np.isfinite(row).all() for row in values
    ):
        raise ValueError("At least two finite independent draws per prompt required")
    weights = (
        np.ones(len(values)) / len(values)
        if prompt_weights is None
        else np.asarray(prompt_weights, dtype=float)
    )
    if (
        weights.shape != (len(values),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Prompt weights must be nonnegative and sum to one")
    variance = math.fsum(
        float(weight**2 * row.var(ddof=1) / len(row))
        for weight, row in zip(weights, values, strict=True)
    )
    return {
        "estimate": math.fsum(
            float(weight * row.mean()) for weight, row in zip(weights, values, strict=True)
        ),
        "empirical_standard_error": math.sqrt(variance),
        "evidence_kind": "DESCRIPTIVE_FIXED_PANEL_MONTE_CARLO",
        "scope": "GENERATION_RANDOMNESS_FIXED_PANEL",
        "zero_empirical_se_is_certainty": False,
        "finite_sample_certificate": False,
        "scene_sampling_included": False,
        "training_seed_inference": False,
    }


def replay_representations(
    streams: Mapping[str, Sequence[Mapping]],
    *,
    representation_builder: Callable,
    rules: Mapping[str, Callable],
    looks: Sequence[int] = (32, 128, 512),
) -> list[dict]:
    """Replay all Z0-Z4 rules on exactly the same per-stream acquired prefix.

    A builder receives only ``(prefix_streams, level)`` and must project to the
    registered representation. A rule receives only that projection. Reference
    or future rows are never passed to either callback. Callbacks must be pure
    and not close over external test/reference data; Python is not a sandbox.
    Deep copies prevent one rule's mutation from altering another's evidence.
    """
    levels = ("Z0", "Z1", "Z2", "Z3", "Z4")
    if not streams or set(rules) != set(levels):
        raise ValueError("Nonempty streams and all five registered rules required")
    if (
        not looks
        or tuple(sorted(set(looks))) != tuple(looks)
        or any(type(n) is not int or n <= 0 for n in looks)
    ):
        raise ValueError("Increasing distinct positive fixed looks required")
    if any(len(rows) < max(looks) for rows in streams.values()):
        raise ValueError("Every stream must cover the requested look; no unequal budgets")
    results = []
    for look in looks:
        prefix = {key: copy.deepcopy(list(rows[:look])) for key, rows in streams.items()}
        decisions = {}
        for level in levels:
            representation = representation_builder(copy.deepcopy(prefix), level)
            decisions[level] = copy.deepcopy(rules[level](copy.deepcopy(representation)))
        results.append(
            {
                "look": look,
                "samples_per_stream": {key: look for key in streams},
                "total_observations": look * len(streams),
                "decisions": decisions,
                "same_prefix": True,
                "reference_access": False,
                "training_seed_inference": False,
            }
        )
    return results


def independent_reference_comparison(selected_interval, reference_intervals: Mapping) -> dict:
    """Reference intervals remain uncertain; never treat the best mean as truth.

    Inputs must already have simultaneous coverage for this comparison. This
    function describes the maximal reference advantage over the selected item.
    """
    from .decision_readout import _interval

    selected = _interval(selected_interval)
    references = {key: _interval(value) for key, value in reference_intervals.items()}
    if selected is None or not references:
        raise ValueError("Selected interval and registered references required")
    unresolved = [
        key
        for key, value in references.items()
        if value is None or (value[0] <= selected[1] and value[1] >= selected[0])
    ]
    regret = (
        None
        if any(value is None for value in references.values())
        else [
            max(0.0, max(value[0] for value in references.values()) - selected[1]),
            max(0.0, max(value[1] for value in references.values()) - selected[0]),
        ]
    )
    return {
        "regret_interval": regret,
        "reference_unresolved_fraction": len(unresolved) / len(references),
        "reference_unresolved_candidates": unresolved,
        "reference_is_exact_truth": False,
    }
