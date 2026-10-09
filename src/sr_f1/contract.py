"""Authenticated uploaded contract and the final 16-by-8 advantage boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from decimal import Decimal, DecimalException
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "docs/sr_f1/package"
PLAN_ID = "SR-F1-20261009"
ARMS = ("A", "J", "PART", "DEC", "GATE")
SEEDS = (71001, 71002, 71003)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def reference_module(name):
    path = PACKAGE / "reference" / (name + ".py")
    manifest = json.loads((PACKAGE / "PACKAGE_SHA256.json").read_text())
    relative = str(path.relative_to(PACKAGE))
    if file_hash(path) != manifest[relative]:
        raise PermissionError("Uploaded reference changed: " + relative)
    spec = importlib.util.spec_from_file_location("sr_f1.reference_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_reference = reference_module("semantic_contract")
stable_seed = _reference.stable_seed
_original_fraction = _reference.fraction_of
_original_json = _reference.load_json_strict


def fraction_of(value, *, text=False):
    """Reject explosive Decimal exponents before constructing a Fraction.

    A nonzero permitted fraction has numerator <= 10**12 and denominator
    <= 10**9. A finite decimal in that set needs at most 29 decimal places
    after removing trailing zeroes (2**30 already exceeds the denominator cap).
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Nonfinite numeric payload")
        if not value:
            return _original_fraction(0, text=text)
        _sign, digits, exponent = value.as_tuple()
        digits = list(digits)
        while digits[-1] == 0:
            digits.pop()
            exponent += 1
        if exponent < -29 or len(digits) + exponent - 1 > 12:
            raise ValueError("Numeric payload exceeds exact protocol bounds")
    return _original_fraction(value, text=text)


def load_json_strict(text):
    try:
        return _original_json(text)
    except DecimalException as exc:
        raise ValueError("Invalid Decimal payload") from exc


# Scoped to this authenticated, separately loaded module; the archived source
# remains byte-identical. These are parser robustness repairs, not score changes.
_reference.fraction_of = fraction_of
_reference.load_json_strict = load_json_strict
score = _reference.score
execute = _reference.execute
gold_evidence = _reference.gold_evidence
gold_output = _reference.gold_output
FAMILIES = _reference.FAMILIES


def load_config():
    return json.loads((PACKAGE / "config/SR_F1.json").read_text())


def reward_advantages(arm, scores):
    """The entire update is mandatory, including zero-contrast groups."""
    if len(scores) != 16 or any(len(group) != 8 for group in scores):
        raise ValueError("SR-F1 final advantages require the entire 16 x 8 update")
    return _reference.reward_advantages(arm, scores, beta=0.5, kappa=0.25)
