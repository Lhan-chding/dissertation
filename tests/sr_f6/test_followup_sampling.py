from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from sr_f6_followup import sampling

spec = importlib.util.spec_from_file_location(
    "followup_legacy_sampling_fixtures", Path(__file__).with_name("test_review.py")
)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def pool(n=16):
    rows = legacy.full_pool(n)
    for row in rows:
        row.update(phase=sampling.PHASE, split=sampling.SPLIT)
        for key in ("root_id", "qid", "row_id"):
            row[key] = row[key].replace("DEV_CAL", sampling.SPLIT)
    return rows


def build(rows):
    return sampling.build_review(
        rows, raw_by_row=legacy.raw_fixture_mapping(rows), source_identity=legacy.SOURCE
    )


def test_determinism_original_text_cap_and_seed():
    rows = pool()
    first = build(rows)
    assert build(list(reversed(rows))) == first
    with pytest.raises(ValueError, match="frozen"):
        sampling.build_review(
            rows, raw_by_row=legacy.raw_fixture_mapping(rows), source_identity=legacy.SOURCE, seed=0
        )


def test_same_qid_exact_text_dedup_never_global_text_dedup():
    rows = pool(64)
    for row in rows:
        row["full_text"] = row["raw_text"] = "identical exact text"
    ledger, _ = build(rows)
    assert ledger["unique_units"] == 1152
    assert ledger["selected_count"] == 360
    assert ledger["alias_policy"] == "explicit_source_row_provenance_only"


def test_rare_priority_and_unknown_original_raw_stop():
    rows = pool()
    legacy.update_labels(rows[-1], J=1, S=0, B=1)
    legacy.update_labels(rows[-2], J=0, S=None, B=None, finish=0)
    raw = legacy.raw_fixture_mapping(rows)
    raw[rows[-2]["row_id"]]["finish_reason"] = "eos"
    ledger, blind = sampling.build_review(rows, raw_by_row=raw, source_identity=legacy.SOURCE)
    selected = {r["row_id"]: r for r in ledger["selected"]}
    assert rows[-1]["row_id"] in selected and rows[-2]["row_id"] in selected
    item = selected[rows[-2]["row_id"]]
    assert item["generation_finish_reason"] == "eos"
    assert (
        next(r for r in blind if r["anonymous_id"] == item["anonymous_id"])[
            "generation_finish_reason"
        ]
        == "eos"
    )


@pytest.mark.parametrize("mutation", ["old_phase", "old_root", "old_alias", "duplicate", "raw"])
def test_original_identity_cannot_be_replaced(mutation):
    rows = pool()
    raw = legacy.raw_fixture_mapping(rows)
    if mutation == "old_phase":
        rows[0]["phase"] = "PARENT_CENSUS"
    elif mutation == "old_root":
        rows[0]["root_id"] = "DEV_CAL_BINDING_0000"
    elif mutation == "old_alias":
        rows[0]["source_row_id"] = "b0/DEV_CAL_BINDING_0000_P0_v0/0"
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        raw[rows[0]["row_id"]]["tokens"] = [100]
    with pytest.raises((PermissionError, ValueError)):
        sampling.build_review(rows, raw_by_row=raw, source_identity=legacy.SOURCE)
