import json
from collections import Counter
from pathlib import Path

import pytest

from sr_f6.planning import draw_seed, formal_paths, root_schedule
from srf6_ref.core import FAMILIES
from srf6_ref.planning import draw_seed as reference_draw_seed


def test_complete_paired_order_and_independent_parents():
    orders = [root_schedule(48, 2026102701, parent, 0) for parent in ("b0", "b1", "b2")]
    assert orders[0] != orders[1] != orders[2]
    assert orders[0] == root_schedule(48, 2026102701, "b0", 0)
    for order in orders:
        assert len(order) == 72
        flat = [root for group in order for root in group]
        assert len(flat) == len(set(flat)) == 288
        for window in range(6):
            counts = Counter(
                root.removeprefix("TRAIN_").rsplit("_", 1)[0]
                for group in order[window * 12 : (window + 1) * 12]
                for root in group
            )
            assert counts == {family: 8 for family in FAMILIES}


def test_pilot_has_correct_lineage():
    schedule = root_schedule(8, 2026102701, "b1", 0, split="DEV_PILOT")
    assert len(schedule) == 12
    assert all(root.startswith("DEV_PILOT_") for group in schedule for root in group)
    with pytest.raises(ValueError):
        root_schedule(8, 2026102701, "b1", 1, split="DEV_PILOT")
    with pytest.raises(ValueError):
        root_schedule(8, 2026102701, "b1", 0, split="TEST_ID")


def test_draw_seed_remains_registered_and_has_no_arm_input():
    args = (2026102701, "b0", 1, 5, 3, 2, 7)
    assert draw_seed(*args) == reference_draw_seed(*args)
    with pytest.raises(ValueError):
        draw_seed(2026102701, "b0", 1, 5, 3, 2, 8)


def test_formal_order_is_frozen_random_order_with_all_controls():
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    paths = formal_paths(config)
    assert len(paths) == len({path["run_id"] for path in paths}) == 54
    assert paths == formal_paths(config)
    for start in range(0, 54, 6):
        block = paths[start : start + 6]
        assert {path["arm"] for path in block} == set(config["arms"])
        assert len({(path["parent"], path["repeat"]) for path in block}) == 1
        assert [path["block_order"] for path in block] == list(range(6))
    assert any(
        [path["arm"] for path in paths[i : i + 6]] != config["arms"] for i in range(0, 54, 6)
    )
