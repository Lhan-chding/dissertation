"""Production contracts must reproduce dispatch identities and exact constant groups."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from mm_dev.contract import advantages, canonical, run_matrix, seed_usage, slots

DESIGN = Path(__file__).resolve().parents[2] / "docs/mm_dev_f2/design/config"


@pytest.mark.parametrize(
    "value", [Fraction(x) for x in ("0", "1/6", "1/4", "1/3", "1/2", "2/3", "3/4", "5/6", "1")]
)
def test_exact_constant_reward_group(value):
    assert advantages([value] * 8) == [0.0] * 8


def test_all_production_streams_match_dispatched_bytes():
    streams = {"ENGINE_e0": slots("ENGINE_F2", "a0", 0, 4, steps=4)}
    for repeat in (0, 1):
        streams[f"PREP_p{repeat}"] = slots("PREP", "a0", repeat, 32)
        for action in ("a0", "aP", "aC"):
            streams[f"CONT_f{repeat}_{action}"] = slots("CONTINUE", action, repeat, 64)
    for name, rows in streams.items():
        assert (
            b"".join(canonical(row) + b"\n" for row in rows)
            == (DESIGN / "schedules" / (name + ".jsonl")).read_bytes()
        )
    assert run_matrix() == json.loads((DESIGN / "run_matrix.json").read_text())
    assert seed_usage() == json.loads((DESIGN / "seed_registry.json").read_text())
