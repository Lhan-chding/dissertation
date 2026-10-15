import hashlib

import pytest

from sr_f3_ce.external import (
    blind_item_identity,
    build_type_registry,
    freeze_chartqapro,
    select_chartqapro,
    typed_audit,
)


def candidates():
    return [
        dict(
            qid=f"q{i}",
            source_row_index=i,
            image_id=f"image{i // 2}",
            image_sha256=f"sha{i // 2}",
            image_pixel_sha256=f"pix{i // 2}",
            question_type="Factoid",
            question_count=1,
            question="question",
            paragraph="context",
            gold="never used",
        )
        for i in range(10)
    ]


def test_subset_selects_only_identity_not_gold_or_iteration_order():
    rows = candidates()
    result = select_chartqapro(rows, prior_image_sha256=["sha0"], limit=3)
    changed = [
        dict(r, gold="changed", answer="wrong", historical_error=True) for r in reversed(rows)
    ]
    assert select_chartqapro(changed, prior_image_sha256=["sha0"], limit=3) == result
    assert len(result) == 3
    assert len({r["image_id"] for r in result}) == 3
    assert all(r["image_id"] != "image0" for r in result)
    assert all("gold" not in r and "answer" not in r for r in result)
    expected_images = sorted(
        [f"image{i}" for i in range(1, 5)],
        key=lambda x: hashlib.sha256(("SRF3CE-external|" + x).encode()).hexdigest(),
    )[:3]
    assert [r["image_id"] for r in result] == expected_images
    assert len(select_chartqapro(rows, prior_pixel_sha256=["pix0"])) == 4


def test_freeze_rejects_missing_historical_image_audit(tmp_path):
    with pytest.raises(RuntimeError, match="prior-used-image"):
        freeze_chartqapro(
            "missing",
            tmp_path,
            history_receipt={},
            scorer_source="missing",
            prompt_source="missing",
            anls_wheel="missing",
        )


def review(item, reviewer, kind="COUNT_INT", comparison="EXACT"):
    return dict(
        qid=item["qid"],
        reviewer_id=reviewer,
        kind=kind,
        comparison=comparison,
        input_identity=blind_item_identity(item),
        method="independent_visual_annotation",
        evidence="Visible axis and question ask an integer duration, not a calendar year",
        used_gold=False,
        used_model_output=False,
    )


def test_keyword_alone_never_certifies_type_and_two_reviews_are_required():
    item = dict(qid="q", question="How many years?", image_sha256="visible-image")
    default = build_type_registry([item], [], outputs_unseen=True)["rows"][0]
    assert default["kind"] == default["status"] == "UNRESOLVED"
    one = build_type_registry([item], [review(item, "a")], outputs_unseen=True)["rows"][0]
    assert one["status"] == "UNRESOLVED"
    two = build_type_registry([item], [review(item, "a"), review(item, "b")], outputs_unseen=True)[
        "rows"
    ][0]
    assert two["kind"] == "COUNT_INT"
    disagreement = build_type_registry(
        [item], [review(item, "a"), review(item, "b", "YEAR")], outputs_unseen=True
    )
    assert disagreement["rows"][0]["status"] == "UNRESOLVED"
    incomplete = review(item, "b")
    del incomplete["used_gold"]
    unknown_exposure = build_type_registry(
        [item], [review(item, "a"), incomplete], outputs_unseen=True
    )
    assert unknown_exposure["rows"][0]["status"] == "UNRESOLVED"
    with pytest.raises(PermissionError):
        build_type_registry([item], [], outputs_unseen=False)


def test_typed_units_require_frozen_type_and_do_not_select_gold_matching_number():
    base = dict(status="RESOLVED", kind="PERCENT_POINT", comparison="EXACT")
    assert typed_audit("2%", "2", base, original_score=0)["typed_score"] == 1
    ratio = dict(base, kind="RATIO_FRACTION")
    assert typed_audit("2%", "0.02", ratio, original_score=0)["typed_score"] == 1
    assert typed_audit("0.02%", "0.0002", ratio, original_score=0)["typed_score"] == 1
    assert typed_audit("10 then 20", "20", base, original_score=0)["typed_score"] is None
    unresolved = typed_audit(
        "2%", "2", dict(base, status="UNRESOLVED", kind="UNRESOLVED"), original_score=0
    )
    assert unresolved["original_score"] == 0 and unresolved["typed_score"] is None
    assert (
        typed_audit("2021", "2020", dict(base, kind="YEAR"), original_score=1)["typed_score"] == 0
    )
