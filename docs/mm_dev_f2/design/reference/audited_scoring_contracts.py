"""Production CPU contracts for MM-CORE-F1, derived from its frozen reference.

No model calls, rendering, training, scheduler operations, or production estimates.
All exact algebra uses Fraction. Production must additionally audit tokenizer,
image routing, per-request provenance, state persistence and hardware behavior.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import isfinite
from typing import Any

Number = int | float | str | Decimal | Fraction


def rational(value: Number) -> Fraction:
    """Parse a finite decimal without silently rounding to the expected answer."""
    if isinstance(value, bool):
        raise ValueError("BOOLEAN_IS_NOT_NUMBER")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("NONFINITE_NUMBER")
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    if not d.is_finite():
        raise ValueError("NONFINITE_NUMBER")
    # Resource guards, not bounds of the chart's numeric domain.
    if len(d.as_tuple().digits) > 64 or abs(d.as_tuple().exponent) > 128:
        raise ValueError("NUMERIC_RESOURCE_LIMIT")
    return Fraction(d)


def operate(values: Sequence[Fraction], op: str) -> Fraction:
    if len(values) < 2:
        raise ValueError("AT_LEAST_TWO_VALUES_REQUIRED")
    if op == "sum":
        return sum(values, Fraction(0))
    if op == "difference":
        if len(values) != 2:
            raise ValueError("ORDERED_DIFFERENCE_REQUIRES_TWO_VALUES")
        return values[0] - values[1]
    if op == "range":
        return max(values) - min(values)
    raise ValueError(f"UNKNOWN_OPERATION:{op}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"DUPLICATE_KEY:{k}")
        result[k] = v
    return result


def _reject_constant(text: str) -> None:
    raise ValueError(f"JSON_NONFINITE_CONSTANT:{text}")


@dataclass(frozen=True)
class Parsed:
    readings: tuple[Fraction, ...] | None
    answer: Fraction | None
    schema_ok: bool
    field_order_ok: bool
    errors: tuple[str, ...]

    @property
    def L_P(self) -> bool:
        return self.readings is not None

    @property
    def L_A(self) -> bool:
        return self.answer is not None

    @property
    def L(self) -> bool:
        return self.L_P and self.L_A


def parse_response(raw: str, n: int) -> Parsed:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("INVALID_EXPECTED_ARITY")
    if not isinstance(raw, str):
        raise ValueError("RAW_RESPONSE_MUST_BE_TEXT")
    if len(raw.encode("utf-8")) > 65536:
        return Parsed(None, None, False, False, ("TEXT_RESOURCE_LIMIT",))
    try:
        obj = json.loads(
            raw.strip(),
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs_no_duplicates,
        )
    except (ValueError, json.JSONDecodeError, ArithmeticError, RecursionError) as exc:
        return Parsed(None, None, False, False, (f"JSON_ERROR:{exc}",))
    if not isinstance(obj, dict):
        return Parsed(None, None, False, False, ("TOP_LEVEL_NOT_OBJECT",))
    errors: list[str] = []
    rv = obj.get("readings")
    av = obj.get("answer")
    readings: tuple[Fraction, ...] | None = None
    answer: Fraction | None = None
    if not isinstance(rv, list) or len(rv) != n:
        errors.append("READINGS_MISSING_OR_WRONG_ARITY")
    elif any(not isinstance(x, Decimal) for x in rv):
        errors.append("READINGS_NONNUMERIC")
    else:
        try:
            readings = tuple(rational(x) for x in rv)
        except (ValueError, ArithmeticError) as exc:
            errors.append(f"READINGS_ERROR:{exc}")
    if not isinstance(av, Decimal):
        errors.append("ANSWER_MISSING_OR_NONNUMERIC")
    else:
        try:
            answer = rational(av)
        except (ValueError, ArithmeticError) as exc:
            errors.append(f"ANSWER_ERROR:{exc}")
    correct_keys = set(obj) == {"readings", "answer"}
    if not correct_keys:
        errors.append("SCHEMA_KEYS_MISMATCH")
    order = (
        "readings" in obj
        and "answer" in obj
        and list(obj).index("readings") < list(obj).index("answer")
    )
    # Extra fields and reversed order do not erase unambiguous scorable numbers.
    return Parsed(
        readings,
        answer,
        correct_keys and readings is not None and answer is not None,
        bool(order),
        tuple(errors),
    )


def score(
    raw: str,
    truth: Sequence[Number],
    op: str,
    delta: Number = 1,
    axis: tuple[Number, Number] | None = None,
) -> dict[str, Any]:
    v = tuple(rational(x) for x in truth)
    d = rational(delta)
    if d <= 0:
        raise ValueError("DELTA_MUST_BE_POSITIVE")
    true_answer = operate(v, op)
    if axis is not None:
        lo, hi = map(rational, axis)
        if lo >= hi:
            raise ValueError("INVALID_AXIS")
    p = parse_response(raw, len(v))
    out: dict[str, Any] = {
        "parsed": p,
        "L_P": p.L_P,
        "L_A": p.L_A,
        "L": p.L,
        "P": None,
        "C": None,
        "A": None,
        "cell_exact": None,
        "cell_tolerance": None,
        "e": None,
        "near_miss": None,
        "S": None,
        "eps_P": None,
        "eps_C": None,
        "eps_O": None,
        "outside_axis": None,
    }
    if p.L_A:
        out["A"] = p.answer == true_answer
    if p.L_P:
        assert p.readings is not None
        r = p.readings
        out["P"] = r == v
        e = tuple((x - y) / d for x, y in zip(r, v, strict=True))
        out["e"] = e
        out["near_miss"] = 0 < max(abs(x) for x in e) <= 1
        denom = sum((x * x for x in e), Fraction(0)) / len(e)
        if denom:
            mean = sum(e, Fraction(0)) / len(e)
            out["S"] = mean * mean / denom
        out["eps_P"] = (operate(r, op) - true_answer) / d
        if axis is not None:
            out["outside_axis"] = any(x < lo or x > hi for x in r)
    if p.L:
        assert p.readings is not None and p.answer is not None
        r, y = p.readings, p.answer
        out["C"] = y == operate(r, op)
        out["cell_exact"] = "".join(str(int(out[k])) for k in ("P", "C", "A"))
        if out["cell_exact"] in ("101", "110"):
            raise AssertionError("EXACT_CONTRACT_VIOLATION")
        out["eps_C"] = (y - operate(r, op)) / d
        out["eps_O"] = (y - true_answer) / d
        assert out["eps_O"] == out["eps_P"] + out["eps_C"]
        tol = d / 2
        pt = all(abs(x - z) <= tol for x, z in zip(r, v, strict=True))
        ct = abs(y - operate(r, op)) <= tol
        at = abs(y - true_answer) <= tol
        out["cell_tolerance"] = "".join(str(int(x)) for x in (pt, ct, at))
    out["P_full"] = bool(p.L_P and out["P"])
    out["A_full"] = bool(p.L_A and out["A"])
    out["J_full"] = bool(p.L and out["P"] and out["A"])
    return out


def reconstruct_six(
    p: Fraction, a: Fraction, c: Fraction, j: Fraction, z: Fraction
) -> dict[str, Fraction]:
    cells = {
        "111": j,
        "100": p - j,
        "011": a - j - z,
        "010": c - a + z,
        "001": z,
        "000": 1 - p - c + j - z,
    }
    if any(x < 0 or x > 1 for x in cells.values()):
        raise ValueError("INCOMPATIBLE_MARGINALS")
    assert sum(cells.values()) == 1
    return cells


def collision_audit(
    xs: Sequence[tuple[Fraction, ...]], ys: Sequence[tuple[Fraction, ...]]
) -> dict[str, Fraction | int | None]:
    """Valid vectors only; caller must separately provide total coverage.

    All KxK comparisons are dependent observations, not new independent samples.
    """
    if xs and ys and len(xs[0]) != len(ys[0]):
        raise ValueError("MISMATCHED_READSET_ARITY")
    arities = {len(x) for x in list(xs) + list(ys)}
    if len(arities) > 1:
        raise ValueError("MISMATCHED_READSET_ARITY")

    def within(vs: Sequence[tuple[Fraction, ...]]) -> Fraction | None:
        m = len(vs)
        if m < 2:
            return None
        count = sum(v * (v - 1) for v in Counter(vs).values())
        return Fraction(count, m * (m - 1))

    w1, w2 = within(xs), within(ys)
    cross = Fraction(sum(x == y for x in xs for y in ys), len(xs) * len(ys)) if xs and ys else None
    contrast = (
        (w1 + w2) / 2 - cross if w1 is not None and w2 is not None and cross is not None else None
    )
    return {
        "m1": len(xs),
        "m2": len(ys),
        "C11": w1,
        "C22": w2,
        "C12": cross,
        "collision_contrast": contrast,
    }


def mixed_probability(rho: Fraction, group: int) -> Fraction:
    if not 0 <= rho <= 1 or group < 1:
        raise ValueError("INVALID_BERNOULLI_GROUP")
    return 1 - rho**group - (1 - rho) ** group


def beta_mixed_predictive(
    success: int, samples: int, group: int, prior: Fraction = Fraction(1, 2)
) -> Fraction:
    if samples < 0 or not 0 <= success <= samples or group < 1 or prior <= 0:
        raise ValueError("INVALID_BETA_INPUT")
    a, b = success + prior, samples - success + prior

    def moment(x: Fraction) -> Fraction:
        ans = Fraction(1)
        for k in range(group):
            ans *= (x + k) / (a + b + k)
        return ans

    return 1 - moment(a) - moment(b)


QUOTAS = {
    "a0": (Fraction(1, 4),) * 4,
    "aP": (Fraction(1, 8), Fraction(3, 8), Fraction(1, 8), Fraction(3, 8)),
    "aC": (Fraction(1, 8), Fraction(1, 8), Fraction(3, 8), Fraction(3, 8)),
}


def absolute_utility(
    d_a: Sequence[Number],
    d_p: Sequence[Number],
    w: Sequence[Number],
    v: Sequence[Number],
    lam: Number,
    eta: Number,
    cost: Number,
) -> Fraction:
    if not (len(d_a) == len(d_p) == len(w) == len(v)):
        raise ValueError("LAYER_DIMENSION_MISMATCH")
    weights, guards = tuple(map(rational, w)), tuple(map(rational, v))
    ll, ee, cc = map(rational, (lam, eta, cost))
    if any(x < 0 for x in weights + guards + (ll, ee, cc)):
        raise ValueError("NEGATIVE_WEIGHT_OR_COST")
    gain = sum((wi * rational(di) for wi, di in zip(weights, d_a, strict=True)), Fraction(0))
    harm = sum(
        (vi * max(-rational(di), Fraction(0)) for vi, di in zip(guards, d_p, strict=True)),
        Fraction(0),
    )
    return gain - ll * harm - ee * cc


def assert_split_integrity(rows: Iterable[Mapping[str, str]]) -> None:
    """Do not use equality of scalar answers as evidence of split leakage."""
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        split = row["split"]
        for key in ("root_family_id", "source_graph_hash", "image_hash", "processed_image_hash"):
            value = row.get(key, "")
            if not value:
                continue
            signature = (key, value)
            previous = seen.setdefault(signature, split)
            if previous != split:
                raise ValueError(f"CROSS_SPLIT_LEAK:{key}:{value}")


def check_permission(stage: str, *, trigger: bool = False, prefreeze: bool = False) -> None:
    stages = {"INVENTORY", "CPU_CONTRACT", "MM-AUDIT", "MM-ENGINE", "BRIDGE"}
    if stage not in stages:
        raise PermissionError(f"STAGE_NOT_AUTHORIZED:{stage}")
    if stage in {"MM-AUDIT", "MM-ENGINE", "BRIDGE"} and not prefreeze:
        raise PermissionError("MISSING_PRE_INFERENCE_FREEZE")
    if stage in {"MM-ENGINE", "BRIDGE"} and not trigger:
        raise PermissionError("MISSING_CONDITIONAL_TRIGGER")


def check_budget(
    spent: Mapping[str, Number], increment: Mapping[str, Number], limits: Mapping[str, Number]
) -> None:
    unknown = set(increment) - set(limits)
    if unknown:
        raise ValueError(f"UNREGISTERED_RESOURCE:{sorted(unknown)}")
    for key, limit in limits.items():
        used, add, cap = map(rational, (spent.get(key, 0), increment.get(key, 0), limit))
        if min(used, add, cap) < 0:
            raise ValueError("NEGATIVE_RESOURCE_COUNT")
        if used + add > cap:
            raise PermissionError(f"BUDGET_EXCEEDED:{key}")


def planned_counts(bridge: bool, engine: bool = True) -> dict[str, int]:
    format_completions = (3 if bridge else 2) * 96 * 4
    audit_completions = 384 * 4
    engine_updates = 8 if engine else 0
    bridge_updates = 16 if bridge else 0
    engine_rollouts = engine_updates * 2 * 8
    return {
        "format_completions": format_completions,
        "audit_completions": audit_completions,
        "engine_rollouts": engine_rollouts,
        "technical_generation_allowance": 48,
        "normal_total_with_technical_allowance": format_completions
        + audit_completions
        + engine_rollouts
        + 48,
        "optimizer_physical_updates": bridge_updates + engine_updates,
        "bridge_sequence_exposures": 128 if bridge else 0,
        "future_dev_updates_not_authorized": 4 * 32 + 30 * 32,
    }
