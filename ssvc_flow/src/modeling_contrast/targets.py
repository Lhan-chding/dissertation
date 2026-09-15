"""Target identities and fixed linear maps; parent event/action semantics unchanged."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from src.modeling_qualification.math_contracts import helmert, to_nested


def inference_fingerprint(theta, schema_id="parent_737_fp64"):
    values = np.ascontiguousarray(theta, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Finite inference parameters required")
    return hashlib.sha256(
        schema_id.encode() + str(values.shape).encode() + values.tobytes()
    ).hexdigest()


def contrast_decomposition(origin, baseline, candidates):
    origin, baseline, candidates = (
        np.asarray(v, dtype=np.float64) for v in (origin, baseline, candidates)
    )
    if origin.shape != baseline.shape or candidates.shape[1:] != origin.shape:
        raise ValueError("Expected origin/baseline and candidate-by-origin-shaped arrays")
    if not all(np.isfinite(v).all() for v in (origin, baseline, candidates)):
        raise ValueError("Finite aligned responses required")
    # C deliberately never subtracts an estimated origin, avoiding residual anchor noise.
    return {"T": candidates - origin, "B": baseline - origin, "C": candidates - baseline}


def contrast_incidence(pairs, aliases=None):
    aliases = aliases or {}

    def canonical(value):
        seen = set()
        while value in aliases and aliases[value] != value:
            if value in seen:
                raise ValueError("Alias cycle")
            seen.add(value)
            value = aliases[value]
        return value

    pairs = [(canonical(left), canonical(right)) for left, right in pairs]
    policies = tuple(dict.fromkeys(policy for pair in pairs for policy in pair))
    matrix = np.zeros((len(pairs), len(policies)), dtype=np.float64)
    for index, (baseline, candidate) in enumerate(pairs):
        matrix[index, policies.index(baseline)] -= 1
        matrix[index, policies.index(candidate)] += 1
    return policies, matrix


def group_xv_matrix(metadata):
    """Rows are group0 pX,v; group1 pX,v; ...; columns are prompt-major Helmert."""
    if not metadata or len({row["prompt_id"] for row in metadata}) != len(metadata):
        raise ValueError("Unique nonempty prompt identities required")
    groups = sorted({row["group"] for row in metadata})
    weights = np.asarray([row["weight"] for row in metadata], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Finite nonnegative fixed weights required")
    result = np.zeros((2 * len(groups), 3 * len(metadata)))
    event_map = np.stack([helmert()[0], helmert()[:3].sum(axis=0)])
    for j, group in enumerate(groups):
        ids = [i for i, row in enumerate(metadata) if row["group"] == group]
        total = weights[ids].sum()
        if total <= 0:
            raise ValueError("Positive weight in each group required")
        for i in ids:
            result[2 * j : 2 * j + 2, 3 * i : 3 * i + 3] = event_map * weights[i] / total
    return result


def contrast_taylor_bound(lipschitz, main_update, auxiliary_increment, *, at_baseline=False):
    d, e = (np.asarray(v, dtype=np.float64) for v in (main_update, auxiliary_increment))
    if d.shape != e.shape or not np.isfinite(d).all() or not np.isfinite(e).all():
        raise ValueError("Finite aligned updates required")
    if not math.isfinite(lipschitz) or lipschitz < 0:
        raise ValueError("Nonnegative finite Lipschitz constant required")
    return float(
        0.5 * lipschitz * np.linalg.norm(e) ** 2
        + (0 if at_baseline else lipschitz * np.linalg.norm(d) * np.linalg.norm(e))
    )


conditional_coordinates = to_nested
