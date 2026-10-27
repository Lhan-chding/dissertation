"""Bound G8 semantic credit, with all rows and counterfactual mappings retained.

Records are flat scored rows. Results are ordered by draw, and coefficients are
also keyed by row_id; callers must not infer alignment from an input list order.
This module does not score answers or certify that a policy hash is genuine.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict

ARMS = {
    "U00": ("P0", "J"),
    "U01": ("P0", "COND"),
    "U10": ("P1", "J"),
    "U11": ("P1", "COND"),
    "C_GLOBAL": ("P1", "GLOBAL"),
    "C_SHUF": ("P1", "SHUF"),
}
COMPONENTS = ("S", "B")
IDENTITY_FIELDS = ("group_id", "qid", "policy_hash", "parent", "factor", "root_id", "family")
OPTIONAL_IDENTITY_FIELDS = ("experiment_id", "phase", "run_id", "repeat", "step", "variant")
AUX_COEFFICIENT = 0.5
COMPONENT_WEIGHTS = {"S": 0.5, "B": 0.5}


def _binary(value, name):
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError(f"{name} must be a binary integer or boolean")


def _validate(records, arm, seed, aux_coefficient, component_weights):
    if not isinstance(arm, str) or arm not in ARMS or type(seed) is not int:
        raise ValueError("Registered arm and integer shuffle seed required")
    if aux_coefficient != AUX_COEFFICIENT or component_weights != COMPONENT_WEIGHTS:
        raise ValueError("Registered lambda=0.5 and S/B weights=0.5 cannot change")
    if not isinstance(records, (list, tuple)) or len(records) != 8:
        raise ValueError("One complete G8 is required")
    required = {*IDENTITY_FIELDS, "row_id", "draw", "J", "V", "finished", *COMPONENTS}
    for row in records:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("Missing scored G8 fields")
        for name in (*IDENTITY_FIELDS, "row_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError(f"Nonempty {name} identity required")
        if not re.fullmatch(r"[0-9a-f]{64}", row["policy_hash"]):
            raise ValueError("policy_hash must be a SHA256 content identity")
        if type(row["draw"]) is not int or row["draw"] not in range(8):
            raise ValueError("Draw indices must be integers 0 through 7")
        for name in ("J", "V", "finished"):
            _binary(row[name], name)
        for name in COMPONENTS:
            if row[name] is not None:
                _binary(row[name], name)
        if row["J"] and not (row["V"] and row["finished"]):
            raise ValueError("J=1 requires common protocol validity and completion")
    if len({row["row_id"] for row in records}) != 8:
        raise ValueError("Duplicate G8 row identity")
    if {row["draw"] for row in records} != set(range(8)):
        raise ValueError("Duplicate or missing G8 draw")
    for name in IDENTITY_FIELDS:
        if len({row[name] for row in records}) != 1:
            raise ValueError(f"Mixed {name} in one G8")
    for name in OPTIONAL_IDENTITY_FIELDS:
        if any(name in row for row in records):
            if not all(name in row for row in records):
                raise ValueError(f"Incomplete {name} identity")
            # Canonical JSON avoids accidentally accepting 0 and False as one identity.
            if (
                len({json.dumps(row[name], allow_nan=False, sort_keys=True) for row in records})
                != 1
            ):
                raise ValueError(f"Mixed {name} in one G8")
    if records[0]["factor"] != ARMS[arm][0]:
        raise ValueError("Data factor does not match the registered arm")
    return sorted(records, key=lambda row: row["draw"])


def _mode(rows, mode, seed):
    groups = defaultdict(list)
    eligibility = []
    for index, row in enumerate(rows):
        mask = tuple(row[name] is not None for name in COMPONENTS)
        eligible = bool(row["V"] and row["finished"] and any(mask))
        eligibility.append(
            {"row_id": row["row_id"], "eligible": eligible, "known_mask": list(mask)}
        )
        if eligible:
            groups[(None if mode == "GLOBAL" else row["J"], mask)].append(index)
    components = {name: [0.0] * 8 for name in COMPONENTS}
    strata = []
    for (joint, mask), indices in groups.items():
        means = {
            name: sum(rows[index][name] for index in indices) / len(indices) if mask[pos] else None
            for pos, name in enumerate(COMPONENTS)
        }
        if len(indices) >= 2:
            for name in COMPONENTS:
                if means[name] is not None:
                    for index in indices:
                        components[name][index] = COMPONENT_WEIGHTS[name] * (
                            rows[index][name] - means[name]
                        )
        strata.append(
            {
                "J": joint,
                "known_mask": list(mask),
                "indices": list(indices),
                "row_ids": [rows[index]["row_id"] for index in indices],
                "size": len(indices),
                "component_means": means,
            }
        )
    candidate = [
        AUX_COEFFICIENT * sum(components[name][index] for name in COMPONENTS) for index in range(8)
    ]
    applied = candidate.copy()
    permutation = list(range(8))
    rng = random.Random(seed)
    for stratum in strata:
        indices = stratum["indices"]
        sources = indices.copy()
        if mode == "SHUF":
            rng.shuffle(sources)  # Exactly one draw, including identity permutations.
        for destination, source in zip(indices, sources, strict=True):
            permutation[destination] = source
            applied[destination] = candidate[source]
        stratum["source_indices"] = sources
        stratum["source_row_ids"] = [rows[index]["row_id"] for index in sources]
    if mode == "J":
        applied = [0.0] * 8
    for stratum in strata:
        for name, values in (("candidate", candidate), ("applied", applied)):
            selected = [values[index] for index in stratum["indices"]]
            stratum[f"{name}_sum"] = sum(selected)
            stratum[f"{name}_L1"] = sum(abs(value) for value in selected)
            stratum[f"{name}_L2"] = math.sqrt(sum(value * value for value in selected))
    exchangeable = any(
        len({candidate[index] for index in stratum["indices"]}) > 1 for stratum in strata
    )
    return {
        "mode": mode,
        "seed": seed if mode == "SHUF" else None,
        "candidate": candidate,
        "raw_unscaled": [value / AUX_COEFFICIENT for value in candidate],
        "applied": applied,
        "component_unscaled": components,
        "strata": strata,
        "eligibility": eligibility,
        "eligible_count": sum(row["eligible"] for row in eligibility),
        "denominator": 8,
        "destination_to_source_index": permutation,
        "destination_to_source_row_id": [rows[index]["row_id"] for index in permutation],
        "nonzero": any(abs(value) > 1e-15 for value in applied),
        "exchangeable": exchangeable,
        "permutation_changed": mode == "SHUF"
        and any(abs(left - right) > 1e-15 for left, right in zip(candidate, applied, strict=True)),
    }


def build_credit_group(records, arm, seed, *, aux_coefficient=0.5, component_weights=None):
    """Bind all four credit modes to one immutable, validated G8 input group."""
    weights = COMPONENT_WEIGHTS if component_weights is None else component_weights
    rows = _validate(records, arm, seed, aux_coefficient, weights)
    mean = sum(row["J"] for row in rows) / 8
    deviation = math.sqrt(sum((row["J"] - mean) ** 2 for row in rows) / 8)
    native = [(row["J"] - mean) / deviation if deviation else 0.0 for row in rows]
    counterfactuals = {mode: _mode(rows, mode, seed) for mode in ("J", "COND", "GLOBAL", "SHUF")}
    selected = counterfactuals[ARMS[arm][1]]
    optional_fields = tuple(name for name in OPTIONAL_IDENTITY_FIELDS if name in rows[0])
    bound_fields = (
        *IDENTITY_FIELDS,
        *optional_fields,
        "row_id",
        "draw",
        "J",
        "V",
        "finished",
        "S",
        "B",
    )
    bound_rows = [{key: row[key] for key in bound_fields} for row in rows]
    identity = {name: rows[0][name] for name in (*IDENTITY_FIELDS, *optional_fields)}
    return {
        **identity,
        "arm": arm,
        "mode": selected["mode"],
        "seed": seed,
        "lambda": AUX_COEFFICIENT,
        "component_weights": dict(COMPONENT_WEIGHTS),
        "row_ids": [row["row_id"] for row in rows],
        "draws": list(range(8)),
        "scored_rows": bound_rows,
        "scored_group_sha256": hashlib.sha256(
            json.dumps(bound_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "native": native,
        "candidate": list(selected["candidate"]),
        "applied": list(selected["applied"]),
        "coefficients_by_row_id": {
            row["row_id"]: {"native": native[index], "aux": selected["applied"][index]}
            for index, row in enumerate(rows)
        },
        "counterfactuals": counterfactuals,
    }
