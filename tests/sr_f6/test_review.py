from __future__ import annotations

import copy
import json
import random
from collections import Counter

import pytest

from sr_f6 import review

SEED = 2026102701
SOURCE = {"census_spec_sha256": "a" * 64, "independent_review_source_sha256": "b" * 64}


def response(parent="b0", factor="P0", family="BINDING", index=0, *, text=None, labels=None):
    root = f"DEV_CAL_{family}_{index % 8:04d}"
    variant, draw = (index // 8) % 4, (index // 32) % 8
    qid = f"{root}_{factor}_v{variant}"
    values = dict(A=1, J=1, S=1, B=1, V=1, finish=1)
    if labels:
        values.update(labels)
    text = f'  {{"answer": "{index}/3", "work": "原始\\n文本"}}\n' if text is None else text
    return dict(
        phase="PARENT_CENSUS",
        parent=parent,
        factor=factor,
        family=family,
        root_id=root,
        qid=qid,
        draw=draw,
        row_id=f"{parent}/{qid}/{draw}",
        full_text=text,
        raw_text=text,
        finish_reason="balanced" if values["finish"] else "length",
        seed=42,
        tokens=[10, 20],
        model_input={"prompt": "Same input does not prove a cached alias"},
        score=dict(values),
        **values,
    )


def full_pool(n=16):
    return [response(*block, index=index) for block in review.BLOCKS for index in range(n)]


def raw_fixture_mapping(rows):
    """Synthetic sampler fixtures only; production callers must read CENSUS/raw."""
    fields = ("qid", "row_id", "seed", "tokens", "full_text", "model_input", "finish_reason")
    return {row["row_id"]: {key: copy.deepcopy(row[key]) for key in fields} for row in rows}


def build(rows):
    return review.build_review(
        rows, raw_by_row=raw_fixture_mapping(rows), seed=SEED, source_identity=SOURCE
    )


def update_labels(row, **labels):
    row.update(labels)
    row["score"].update(labels)
    if "finish" in labels:
        row["finish_reason"] = "balanced" if labels["finish"] else "length"


def test_complete_360_unique_rows_exact_cell_family_quotas_and_caps():
    records = full_pool()
    before = copy.deepcopy(records)
    ledger, blind = build(records)
    assert records == before
    assert ledger["status"] == "BLIND_PACKAGE_ONLY"
    assert ledger["review_status"] == "REVIEW_NOT_RUN"
    assert ledger["sampling_status"] == "COMPLETE_360"
    assert ledger["selected_count"] == len(blind) == 360
    assert ledger["reviewer_type"] is None
    selected = ledger["selected"]
    assert len({r["row_id"] for r in selected}) == len({r["origin_id"] for r in selected}) == 360
    assert Counter((r["parent"], r["factor"]) for r in selected) == {
        (parent, factor): 60 for parent in review.PARENTS for factor in review.FACTORS
    }
    assert Counter((r["parent"], r["factor"], r["family"]) for r in selected) == {
        block: 10 for block in review.BLOCKS
    }
    assert max(Counter((r["parent"], r["factor"], r["root_id"]) for r in selected).values()) <= 4
    # The cap is intentionally not global: otherwise only 192 would be possible.
    assert max(Counter(r["root_id"] for r in selected).values()) > 4


def test_deterministic_under_input_reordering_and_fixed_domains():
    records = full_pool()
    first = build(records)
    random.Random(782).shuffle(records)
    assert build(records) == first
    assert first[0]["seed_domains"] == review.DOMAINS
    with pytest.raises(ValueError, match="frozen"):
        review.build_review(
            records, raw_by_row=raw_fixture_mapping(records), seed=SEED + 1, source_identity=SOURCE
        )


def test_blind_allowlist_exact_text_no_gold_or_program_fields_and_shuffled_order():
    records = full_pool()
    for record in records:
        record["oracle"] = {"answer": "SECRET_ORACLE"}
        record["witness"] = {"gold_selected": "SECRET_WITNESS"}
    ledger, blind = build(records)
    originals = {row["row_id"]: row for row in records}
    private = {row["anonymous_id"]: row for row in ledger["selected"]}
    for item in blind:
        assert set(item) == {
            "anonymous_id",
            "raw_text",
            "protocol_version",
            "generation_finish_reason",
        }
        assert item["anonymous_id"].startswith("R") and item["anonymous_id"][1:].isdigit()
        assert item["raw_text"] == originals[private[item["anonymous_id"]]["row_id"]]["full_text"]
        assert "SECRET" not in json.dumps(item)
    groups = [(row["parent"], row["factor"], row["family"]) for row in ledger["selected"]]
    assert groups != sorted(groups)
    assert len(set(groups[:20])) > 5


def test_label_changes_affect_private_sampling_but_never_leak_into_blind_schema():
    records = full_pool(10)
    first_ledger, first_blind = build(records)
    changed = copy.deepcopy(records)
    for row in changed:
        update_labels(row, J=0, S=None, B=0, V=0)
    ledger, blind = build(changed)
    # All candidates are needed, so label changes cannot reorder the same blind set.
    assert blind == first_blind
    assert ledger["selected"] != first_ledger["selected"]
    assert all("program_labels" not in item for item in blind)
    larger = full_pool(16)
    original, _ = build(larger)
    chosen = {row["row_id"] for row in original["selected"]}
    adverse = next(row for row in larger if row["row_id"] not in chosen)
    update_labels(adverse, S=0)
    updated, _ = build(larger)
    assert adverse["row_id"] in {row["row_id"] for row in updated["selected"]}


def test_rare_unknown_invalid_and_incomplete_coverage_with_all_rare_cases_listed():
    records = full_pool(16)
    changes = [
        dict(J=1, S=0, B=1),
        dict(J=1, S=1, B=0),
        dict(J=0, S=1, B=0),
        dict(J=0, S=0, B=1),
        dict(J=0, S=None),
        dict(J=0, B=None),
        dict(J=0, V=0),
        dict(J=0, finish=0, S=None, B=None),
    ]
    for block in review.BLOCKS:
        subset = [r for r in records if (r["parent"], r["factor"], r["family"]) == block]
        for row, labels in zip(subset, changes, strict=False):
            update_labels(row, **labels)
    ledger, _ = build(records)
    assert len(ledger["rare_cases"]) == 36 * 4
    for block in ledger["blocks"]:
        assert all(entry["selected_count"] >= 1 for entry in block["targets"].values())
        assert all(entry["coverage_shortfall"] == 0 for entry in block["targets"].values())


def test_missing_and_overquota_rare_states_report_real_counts_not_copies():
    records = full_pool(16)
    for row in records:
        update_labels(row, S=0)
    ledger, blind = build(records)
    assert len(blind) == 360
    assert len(ledger["rare_cases"]) == len(records)
    assert sum(case["selected"] for case in ledger["rare_cases"]) == 360
    for block in ledger["blocks"]:
        rare = block["targets"]["J1_S0"]
        assert (rare["raw_count"], rare["unique_count"], rare["selected_count"]) == (16, 16, 10)
        assert rare["unreviewed_unique_count"] == 6
        absent = block["targets"]["B_UNKNOWN"]
        assert absent["coverage_shortfall"] == 1
        assert absent["shortfall_reason"] == "NOT_OBSERVED"


def test_deduplicate_exact_qid_text_within_cell_without_collapsing_other_parents_or_qids():
    records = full_pool(10)
    for row in records:
        row["full_text"] = row["raw_text"] = '{"answer":1}'
    duplicate = response(index=32, text='{"answer":1}')
    records.append(duplicate)  # Same b0/P0/qid as draw zero; only its draw differs.
    ledger, blind = build(records)
    assert len(blind) == 360
    assert ledger["raw_count"] == 361
    assert ledger["unique_units"] == 360
    assert len({item["raw_text"] for item in blind}) == 1
    assert any(len(item["source_row_ids"]) == 2 for item in ledger["selected"])


def test_low_unique_supply_and_root_cap_shortfalls_are_explicit():
    records = [response(index=32 * draw, text='{"answer":1}') for draw in range(8)]
    ledger, blind = build(records)
    assert len(blind) == 1
    assert ledger["sampling_status"] == "INSUFFICIENT_UNIQUE_RESPONSES"
    assert ledger["shortfall"] == 359
    # Eight distinct answers at one root still provide at most four slots.
    records = [response(index=32 * draw) for draw in range(8)]
    ledger, blind = build(records)
    assert len(blind) == 4
    block = next(
        b
        for b in ledger["blocks"]
        if b["family"] == "BINDING" and b["parent"] == "b0" and b["factor"] == "P0"
    )
    assert block["unique_count"] == 8 and block["shortfall"] == 6


def test_explicit_alias_is_counted_once_but_same_input_seed_is_not_an_alias():
    rows = [response(factor="P0", text="same"), response(factor="P1", text="same")]
    assert len(build(rows)[1]) == 2
    rows[1]["provenance"] = {"cached_from_row_id": rows[0]["row_id"]}
    ledger, blind = build(rows)
    assert len(blind) == 1
    assert ledger["unique_units"] == 2
    assert len({r["origin_id"] for r in ledger["selected"]}) == 1


def test_alias_capacity_lookahead_preserves_feasible_360_and_uses_alternate_draw():
    rows = full_pool(10)
    b0 = next(
        r for r in rows if r["parent"] == "b0" and r["factor"] == "P0" and r["family"] == "BINDING"
    )
    b1 = next(
        r for r in rows if r["parent"] == "b0" and r["factor"] == "P1" and r["family"] == "BINDING"
    )
    b1["full_text"] = b1["raw_text"] = b0["full_text"]
    b1["source_row_id"] = b0["row_id"]
    assert len(build(rows)[1]) == 359
    alternative = response(factor="P0", index=32, text=b0["full_text"])
    rows.append(alternative)  # Same content unit, independently generated second draw.
    ledger, blind = build(rows)
    assert len(blind) == 360
    assert alternative["row_id"] in {r["row_id"] for r in ledger["selected"]}
    assert len({r["origin_id"] for r in ledger["selected"]}) == 360


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows[0].update(qid="FORMAL_TEST_BINDING_0000_P0_v0"),
        lambda rows: rows[0].update(A=True),
        lambda rows: rows[0]["score"].update(A=True),
        lambda rows: rows[0].update(S=0),
        lambda rows: rows[0].update(raw_text="rewritten"),
        lambda rows: rows[0].update(finish_reason="unknown"),
        lambda rows: rows[0].update(provenance="bad"),
    ],
)
def test_reject_changed_identity_malformed_labels_or_edited_text(mutation):
    rows = [response()]
    mutation(rows)
    with pytest.raises((ValueError, PermissionError)):
        build(rows)


def test_reject_inconsistent_duplicate_labels_alias_cycles_and_changed_alias_text():
    rows = [response(text="same"), response(index=32, text="same", labels={"S": 0})]
    with pytest.raises(ValueError, match="inconsistent labels"):
        build(rows)
    rows = [response(text="same"), response(factor="P1", text="same")]
    rows[0]["source_row_id"] = rows[1]["row_id"]
    rows[1]["source_row_id"] = rows[0]["row_id"]
    with pytest.raises(ValueError, match="Cyclic"):
        build(rows)
    rows[0].pop("source_row_id")
    rows[1]["full_text"] = rows[1]["raw_text"] = "changed"
    with pytest.raises(ValueError, match="disagree"):
        build(rows)


def test_no_scoring_or_gold_guided_extraction_is_invoked(monkeypatch):
    from sr_f6 import measurement

    def forbidden(*args, **kwargs):
        raise AssertionError("The review sampler must never score or extract against gold")

    monkeypatch.setattr(measurement, "score_response", forbidden)
    monkeypatch.setattr(measurement, "parse", forbidden)
    monkeypatch.setattr(measurement, "oracle_a", forbidden)
    assert len(build([response()])[1]) == 1


def test_writer_separates_private_ledger_and_blind_package_and_never_overwrites(tmp_path):
    destination = tmp_path / "review"
    rows = [response()]
    raw = raw_fixture_mapping(rows)
    result = review.write_review(
        rows, destination, raw_by_row=raw, seed=SEED, source_identity=SOURCE
    )
    assert result["status"] == "BLIND_PACKAGE_ONLY"
    assert result["review_status"] == "REVIEW_NOT_RUN"
    assert (destination / "private/selection.json").stat().st_mode & 0o777 == 0o600
    assert (destination / "private").stat().st_mode & 0o777 == 0o700
    assert sorted(p.name for p in (destination / "blind").iterdir()) == ["EXTRACTION.jsonl"]
    payload = json.loads((destination / "blind/EXTRACTION.jsonl").read_text())
    assert payload["raw_text"] == response()["full_text"]
    before = (destination / "private/selection.json").read_bytes()
    with pytest.raises(FileExistsError):
        review.write_review(rows, destination, raw_by_row=raw, seed=SEED, source_identity=SOURCE)
    assert (destination / "private/selection.json").read_bytes() == before


@pytest.mark.parametrize("seed,identity", [(True, SOURCE), (SEED, {}), (SEED, {"x": "unbound"})])
def test_explicit_registered_seed_and_source_identity_required(seed, identity):
    with pytest.raises(ValueError):
        review.build_review([], raw_by_row={}, seed=seed, source_identity=identity)


def test_raw_eos_survives_group_length_normalization_without_changing_unknown_labels():
    raw_row = response(text='{"selected":[]', labels=dict(A=0, J=0, S=None, B=None, V=0, finish=0))
    raw_row.update(
        finish_reason="eos", truncated=False, balanced_token_count=None, boundary_token_count=None
    )
    group_row = copy.deepcopy(raw_row)
    # The frozen census annotator overwrites this field with the measurement reason.
    group_row["finish_reason"] = "length"
    group_row["score"]["finish_reason"] = "length"
    before = copy.deepcopy(group_row)
    ledger, blind = review.build_review(
        [group_row], raw_by_row={raw_row["row_id"]: raw_row}, source_identity=SOURCE
    )
    assert group_row == before
    assert blind[0]["generation_finish_reason"] == "eos"
    selected = ledger["selected"][0]
    assert selected["generation_finish_reason_source"] == "CENSUS_RAW_BY_ROW"
    assert selected["program_finish_reason"] == "length"
    assert selected["program_labels"]["finish"] == 0
    assert selected["program_labels"]["S"] is None
    assert selected["program_labels"]["B"] is None
    assert selected["stratum"] == [0, 0, "UNKNOWN", "UNKNOWN"]
    # Accidentally passing the annotated group as raw evidence cannot silently
    # launder its normalized reason while the native stop flags still say EOS.
    with pytest.raises(PermissionError, match="native stop flags"):
        review.build_review(
            [group_row], raw_by_row={group_row["row_id"]: group_row}, source_identity=SOURCE
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("qid", "DEV_CAL_BINDING_0001_P0_v0"),
        ("row_id", "b1/DEV_CAL_BINDING_0000_P0_v0/0"),
        ("seed", 43),
        ("seed", True),
        ("tokens", [10, 21]),
        ("tokens", [True]),
        ("full_text", "changed"),
        ("model_input", {"prompt": "different task"}),
        ("finish_reason", "unknown"),
        ("truncated", True),
        ("balanced_token_count", None),
    ],
)
def test_original_sampling_identity_and_stop_metadata_are_checked(field, value):
    rows = [response()]
    raw = raw_fixture_mapping(rows)
    raw[rows[0]["row_id"]][field] = value
    with pytest.raises((ValueError, PermissionError)):
        review.build_review(rows, raw_by_row=raw, source_identity=SOURCE)


def test_missing_or_extra_raw_rows_are_rejected_without_falling_back_to_group_reason():
    rows = [response()]
    with pytest.raises(ValueError, match="raw sampling record required"):
        review.build_review(rows, raw_by_row={}, source_identity=SOURCE)
    with pytest.raises(ValueError, match="raw sampling row mapping"):
        review.build_review(rows, raw_by_row=None, source_identity=SOURCE)
    raw = raw_fixture_mapping(rows)
    raw["unexpected"] = dict(raw[rows[0]["row_id"]])
    with pytest.raises(PermissionError, match="exact annotated census row set"):
        review.build_review(rows, raw_by_row=raw, source_identity=SOURCE)


def test_writer_rejects_changed_seed_before_creating_any_output(tmp_path):
    destination = tmp_path / "changed-seed"
    rows = [response()]
    with pytest.raises(ValueError, match="frozen at 2026102701"):
        review.write_review(
            rows,
            destination,
            raw_by_row=raw_fixture_mapping(rows),
            source_identity=SOURCE,
            seed=SEED + 1,
        )
    assert not destination.exists()
