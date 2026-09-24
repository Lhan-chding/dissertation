"""Strict repair labels; reward bins use exact relation counts, never rounding."""

from __future__ import annotations

import math
from fractions import Fraction

from ..decision_modeling.semantic_schema import semantic_features as _semantic_features

EVENTS = ("X", "S", "W", "I")


def _world(value):
    return (
        isinstance(value, (list, tuple)) and len(value) == 4 and all(type(x) is int for x in value)
    )


def repair_features(observed, truth, prediction):
    if not _world(observed) or not _world(truth):
        raise ValueError("Four integer coordinates required")
    changed = [i for i in range(4) if observed[i] != truth[i]]
    if len(changed) != 1:
        raise ValueError("Exactly one corrupted coordinate required")
    if not _world(prediction):
        return dict(
            valid=False,
            F=None,
            B=None,
            M=None,
            copy=None,
            single_edit=None,
            coord_accuracy=0.0,
            damage_any=None,
            damage_penalty=1.0,
            Bdamage=1.0,
        )
    j = changed[0]
    f = int(prediction[j] == truth[j])
    b = sum(prediction[k] != truth[k] for k in range(4) if k != j)
    m = sum(prediction[k] != observed[k] for k in range(4))
    return dict(
        valid=True,
        F=f,
        B=b,
        M=m,
        copy=int(m == 0),
        single_edit=int(m == 1),
        coord_accuracy=(f + 3 - b) / 4,
        damage_any=int(b > 0),
        damage_penalty=b / 3,
        Bdamage=b / 3,
    )


def reward_bin(event, numerator, denominator):
    if event not in EVENTS or type(numerator) is not int or type(denominator) is not int:
        raise ValueError("Event and exact integer relation counts required")
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise ValueError("Invalid relation counts")
    if (event == "I" and numerator != 0) or (event == "X" and numerator != denominator):
        raise ValueError("Reward consistency violated")
    c = Fraction(numerator, denominator)
    return f"{event}:{c.numerator}/{c.denominator}"


def reward_vector(row):
    reward_bin(row["event"], row["relation_numerator"], row["relation_denominator"])
    event = row["event"]
    return [
        float(event == "X"),
        float(event in ("X", "S")),
        float(event != "I"),
        row["relation_numerator"] / row["relation_denominator"],
    ]


def validate_repair(row):
    f, b, m = (row[k] for k in ("F", "B", "M"))
    if row["event"] == "I":
        if (f, b, m) != (None, None, None):
            raise ValueError("Invalid output must have missing repair coordinates")
        return
    if type(f) is not int or f not in (0, 1) or type(b) is not int or not 0 <= b <= 3:
        raise ValueError("Invalid F/B repair labels")
    if type(m) is not int or not 0 <= m <= 4 or m not in (b, b + 1) or (f and m != b + 1):
        raise ValueError("Impossible edit count")
    if (row["event"] == "X") != (f == 1 and b == 0):
        raise ValueError("X iff F=1 and B=0")


def semantic_features(raw, prompt):
    result = _semantic_features(raw, prompt)
    scene = prompt.get("scene", prompt)
    if scene.get("track", prompt.get("track", "N")) != "N":
        raise ValueError("Prospective repair reward is registered only for N")
    repair = repair_features(scene["observed_world"], scene["truth_world"], result["parsed_world"])
    result.update(repair)
    validate_repair(result)
    reward_vector(result)
    return result


def event_probabilities(x, a, v):
    if not all(math.isfinite(t) for t in (x, a, v)) or not 0 <= x <= a <= v <= 1:
        raise ValueError("Expected 0 <= X <= A <= V <= 1")
    return dict(X=x, S=a - x, W=v - a, I=1 - v)


repair = repair_features
