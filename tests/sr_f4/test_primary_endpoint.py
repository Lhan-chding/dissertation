"""Known-case reporting preserves denominators and avoids formal pass claims."""

import copy

import pytest

from sr_f4.reporting import known_v7_endpoint, known_v7_report_lines


def summaries():
    # Observed numerator pattern; fixtures exercise both value and native J.
    counts = {
        "PARENT": ((15, 15), (7, 7), (1, 0)),
        "J_UNIFORM_NEW_REPEAT": ((235, 232), (193, 192), (3, 1)),
        "LZ": ((244, 242), (239, 237), (243, 239)),
    }
    return [
        dict(
            block=block,
            arm=arm,
            panel="TEST_ID",
            family="INTERVAL",
            variant=7,
            answers=256,
            worlds=32,
            **{
                metric: dict(mean=total / 256, sum=total, n=256)
                for metric, total in zip(("value_correct", "J"), totals, strict=True)
            },
        )
        for arm, blocks in counts.items()
        for block, totals in enumerate(blocks)
    ]


def test_known_case_exact_filter_counts_deltas_and_no_mutation():
    values = summaries()
    before = copy.deepcopy(values)
    unrelated = [
        dict(values[0], panel="B_CONFIRM"),
        dict(values[0], family="TOPK"),
        dict(values[0], variant=6),
    ]
    result = known_v7_endpoint(list(reversed(values + unrelated)))
    assert result["status"] == "COMPLETE"
    assert result["formal_acceptance"] == "NOT_ASSESSED_BY_POINT_THRESHOLDS"
    assert len(result["rows"]) == 6
    assert values == before
    b2 = next(r for r in result["rows"] if r["block"] == 2 and r["metric"] == "value_correct")
    assert b2["arms"]["LZ"] == dict(mean=243 / 256, sum=243, n=256)
    assert b2["lz_minus_uniform"] == 240 / 256
    assert b2["lz_minus_parent"] == 242 / 256
    rendered = "\n".join(known_v7_report_lines(result))
    assert "94.92 (243/256)" in rendered
    assert "+93.75" in rendered
    assert "不作正式E1通过判定" in rendered
    assert "SR-F3" not in rendered


def test_missing_arm_does_not_become_zero_or_complete_comparison():
    values = [r for r in summaries() if not (r["block"] == 2 and r["arm"] == "LZ")]
    result = known_v7_endpoint(values)
    assert result["status"] == "INCOMPLETE"
    assert result["missing"] == [dict(block=2, arm="LZ")]
    assert all(r["block"] != 2 for r in result["rows"])
    assert "INCOMPLETE" in "\n".join(known_v7_report_lines(result))


@pytest.mark.parametrize(
    "change",
    [
        dict(answers=128),
        dict(worlds=16),
        dict(value_correct=dict(mean=0.9, sum=15, n=256)),
        dict(J=dict(mean=0.1, sum=1, n=10)),
        dict(J=dict(mean=float("nan"), sum=15, n=256)),
    ],
)
def test_incomplete_or_inconsistent_summary_is_explicit(change):
    values = summaries()
    values[0].update(change)
    result = known_v7_endpoint(values)
    assert result["status"] == "INCOMPLETE"
    assert result["invalid"]
    assert all(r["block"] != 0 for r in result["rows"])


def test_duplicate_cell_is_rejected():
    values = summaries()
    with pytest.raises(ValueError, match="Duplicate known-v7"):
        known_v7_endpoint([*values, values[0]])
