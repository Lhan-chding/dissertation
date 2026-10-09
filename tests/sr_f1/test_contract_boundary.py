"""Regressions at the model-output and final-advantage boundaries."""

import json
from decimal import Decimal
from fractions import Fraction

import pytest

from sr_f1.contract import PACKAGE, fraction_of, reward_advantages, score
from sr_f1.data import model_input


def task():
    return json.loads(
        (PACKAGE / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl").read_text().splitlines()[0]
    )


@pytest.mark.parametrize(
    "number",
    [
        "1e999999999",
        "1e-999999999",
        "1e99999999999999999999999999999999999999",
        "1e-99999999999999999999999999999999999999",
    ],
)
def test_short_extreme_exponents_remain_scored_failures(number):
    row = task()
    result = score('{"answer":' + number + "}", row["world"], row["query"])
    assert result["A"] == result["J"] == 0


def test_decimal_bounds_preserve_exact_valid_values():
    assert fraction_of(Decimal("0e999999999")) == 0
    assert fraction_of(Decimal("1000000000000.0000000000000")) == 10**12
    assert fraction_of(Decimal("0.00000000186264514923095703125")) == Fraction(1, 2**29)
    with pytest.raises(ValueError):
        fraction_of(Decimal("0.000000000931322574615478515625"))


def test_gold_sidecar_cannot_change_model_visible_record():
    raw = json.loads((PACKAGE / "manifests/MODEL_INPUTS.jsonl").read_text().splitlines()[0])
    for protocol in ("evidence_answer", "answer_only"):
        expected = model_input(raw, protocol)
        contaminated = dict(
            raw,
            answer="secret",
            required_evidence=["secret"],
            root_id="secret",
            query={"family": "FAKE"},
            world={"Alpha": 999},
            qid="changed",
        )
        assert model_input(contaminated, protocol) == expected
        assert set(expected) == {"image_file", "text"}


@pytest.mark.parametrize("arm", ["A", "J", "PART", "DEC", "GATE"])
def test_full_batch_boundary_rejects_microbatch_normalization(arm):
    zero = {"A": 0, "J": 0, "E": 0, "p_read": 0}
    with pytest.raises(ValueError):
        reward_advantages(arm, [[zero] * 8])
    advantage, info = reward_advantages(arm, [[zero] * 8 for _ in range(16)])
    assert advantage.shape == (16, 8)
    assert not advantage.any()
    assert info["prompt_count"] == 16
