from collections import Counter

import pytest

from sr_f1.contract import PACKAGE
from sr_f1.data import load_tasks
from sr_f2.workers import endpoint_slots


def test_endpoint_reuses_exact_registered_seed_and_reduced_panels():
    from sr_f12.protocol import iter_model_slots

    tasks = load_tasks(PACKAGE)
    rows = endpoint_slots(tasks, "SRF2_EARLY_s71001")
    prior = [
        r
        for r in iter_model_slots(tasks, [], "SRF1_2_J_s71001", stage="final", test_released=True)
        if r["pool"] in ("TEST_ID", "TEST_COMPOSITION")
    ]
    assert len(rows) == 12800
    assert Counter((r["pool"], r["protocol"]) for r in rows) == {
        ("TEST_ID", "evidence_answer"): 8192,
        ("TEST_ID", "answer_only"): 4096,
        ("TEST_COMPOSITION", "evidence_answer"): 512,
    }
    for current, old in zip(rows, prior, strict=True):
        for key in ("sampling_seed", "group_seed", "generation", "qid", "draw"):
            assert current[key] == old[key]
    assert len({row["slot_id"] for row in rows}) == 12800


def test_unregistered_endpoint_rejected():
    with pytest.raises(PermissionError):
        endpoint_slots({}, "SRF2_PART_s71001")
