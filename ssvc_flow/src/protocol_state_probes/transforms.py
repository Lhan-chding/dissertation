"""Public-only rendering and fixed emission/canonical coordinate transformations."""

import json
from fractions import Fraction

from .ptlc import external, permutation, run, transform_star

NAMES = "abcd"
FWD = (0, 1, 2, 3)
REV = (3, 2, 1, 0)
PROTOCOLS = ("O0", "A1", "B0", "B1", "BR", "L00", "L01", "L10", "L11")
SYSTEM = (
    "You are repairing a four-integer chart record. Return only one JSON array\n"
    "containing four integers from 0 to 99. Do not include an explanation."
)
OPERATIONS = {
    "sum4": "a+b+c+d",
    "difference_pairs": "(a+b)-(c+d)",
    "range4": "max(a,b,c,d)-min(a,b,c,d)",
}
PUBLIC_KEYS = {"observed", "H", "b", "family", "operation", "original_system", "original_user"}


def emit(canonical, order):
    order = permutation(order)
    if len(canonical) != 4:
        raise ValueError("Four coordinates required")
    return [canonical[k] for k in order]


def restore(emitted, order):
    order = permutation(order)
    if len(emitted) != 4:
        raise ValueError("Four coordinates required")
    y = [None] * 4
    for value, coordinate in zip(emitted, order, strict=True):
        y[coordinate] = value
    return y


def valid_domain(y):
    return (
        isinstance(y, (list, tuple))
        and len(y) == 4
        and all(type(v) is int and 0 <= v <= 99 for v in y)
    )


def matrix_from_scene(scene):
    cue = scene["cue"]
    if cue["family"] == "trend":
        return [[1, -2, 1, 0], [0, 1, -2, 1]], [0, 0]
    if cue["family"] == "cross_series":
        H, b = [], []
        for i, j, rhs in sorted(cue["edges"]):
            if type(i) is not int or type(j) is not int or i == j or i not in FWD or j not in FWD:
                raise ValueError("Invalid public relation edge")
            row = [0] * 4
            row[i] = row[j] = 1
            H.append(row)
            b.append(rhs)
        return H, b
    if cue["family"] == "duplicate_encoding":
        k = cue["known_index"]
        if type(k) is not int or k not in FWD:
            raise ValueError("Invalid public known coordinate")
        row = [0] * 4
        row[k] = 1
        return [row], [cue["known_value"]]
    raise ValueError("Unsupported public relation family")


def public_from_record(record):
    """The only fields read here are public observations, cues and original text."""
    H, b = matrix_from_scene(record["scene"])
    return {
        "observed": list(record["scene"]["observed_world"]),
        "H": H,
        "b": b,
        "family": record["family"],
        "operation": record["operation"],
        "original_system": record["prompt"]["system"],
        "original_user": record["prompt"]["user"],
    }


def strict_public(public):
    if set(public) != PUBLIC_KEYS:
        raise ValueError("Renderer accepts PUBLIC_KEYS only; audit information forbidden")
    if not valid_domain(public["observed"]):
        raise ValueError("Observed record must be four integers in 0..99")
    if public["operation"] not in OPERATIONS:
        raise ValueError("Unknown operation")
    if (
        not public["H"]
        or len(public["H"]) != len(public["b"])
        or any(len(row) != 4 for row in public["H"])
    ):
        raise ValueError("Invalid public equation shape")
    if any(type(v) is not int for row in public["H"] for v in row) or any(
        type(v) is not int for v in public["b"]
    ):
        raise ValueError("Public equation coefficients must be exact integers")
    if public["original_system"] != SYSTEM or not isinstance(public["original_user"], str):
        raise ValueError("Unknown parent messages; review rather than rewrite")


def format_linear(row, rhs):
    parts = []
    # Preserve the registered leaf-minus-reference wording, including negative rhs.
    ordered = sorted(zip(NAMES, row, strict=True), key=lambda pair: Fraction(pair[1]) < 0)
    for name, coefficient in ordered:
        coefficient = Fraction(coefficient)
        if not coefficient:
            continue
        magnitude = abs(coefficient)
        term = name if magnitude == 1 else f"{magnitude}*{name}"
        parts.append(
            (
                ("-" if coefficient < 0 else "")
                if not parts
                else (" - " if coefficient < 0 else " + ")
            )
            + term
        )
    return "".join(parts) + " = " + str(Fraction(rhs))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Template span missing or ambiguous: " + old)
    return text.replace(old, new, 1)


def b0_order(H, b):
    _, _, meta = transform_star(H, b, FWD)
    center, first = meta["center"], meta["first_leaf"]
    by_leaf = {
        next(k for k, v in enumerate(row) if v and k != center): (row, rhs)
        for row, rhs in zip(H, b, strict=True)
    }
    leaves = [first] + [k for k in FWD if k not in (center, first)]
    return [list(by_leaf[k][0]) for k in leaves], [by_leaf[k][1] for k in leaves]


def render(public, protocol):
    strict_public(public)
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown protocol")
    H, b = [list(row) for row in public["H"]], list(public["b"])
    user = public["original_user"]
    order = REV if protocol in ("A1", "L01", "L11") else FWD
    if order == REV:
        user = replace_once(
            user,
            "Return [a,b,c,d], not the downstream answer.",
            "Return [d,c,b,a], not the downstream answer.",
        )
    if protocol.startswith("L"):
        input_order = REV if protocol in ("L10", "L11") else FWD
        old = (
            "The observed record is " + json.dumps(public["observed"], separators=(",", ":")) + "."
        )
        new = (
            "The observed record is "
            + ", ".join(f"{NAMES[k]}={public['observed'][k]}" for k in input_order)
            + "."
        )
        user = replace_once(user, old, new)
    if protocol in ("B0", "B1", "BR"):
        if public["family"] != "cross_series":
            raise ValueError("B protocols require sum-star cross tasks")
        if protocol == "B0":
            H, b = b0_order(H, b)
        elif protocol == "BR":
            H, b = list(reversed(H)), list(reversed(b))
        else:
            H, b, _ = transform_star(H, b, FWD)
        prefix, suffix = (
            "The following relationships are reliable:\n",
            "\nThe downstream calculation is:",
        )
        if user.count(prefix) != 1 or user.count(suffix) != 1:
            raise ValueError("Equation template span ambiguous")
        before, remaining = user.split(prefix)
        _, after = remaining.split(suffix)
        user = (
            before
            + prefix
            + "\n".join(format_linear(row, rhs) for row, rhs in zip(H, b, strict=True))
            + suffix
            + after
        )
    return {
        "system": public["original_system"],
        "user": user,
        "output_order": list(order),
        "H_display": H,
        "b_display": b,
    }


def predict(public, protocol):
    rendered = render(public, protocol)
    return predict_equations(
        rendered["H_display"], rendered["b_display"], public["observed"], rendered["output_order"]
    )


def predict_equations(H, b, observed, order):
    predictions = {}
    for variant in ("V1", "V2"):
        y, trace = run(H, b, observed, order, variant=variant)
        predictions[variant] = {
            "canonical": external(y),
            "emitted": external(emit(y, order)),
            "integer_prediction": all(x.denominator == 1 for x in y),
            "in_domain": all(x.denominator == 1 and 0 <= x <= 99 for x in y),
            "trace": trace,
        }
    return predictions
