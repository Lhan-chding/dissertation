import copy

import pytest

from sr_f6.interface_v3_audit import j_gap_breakdown


def row(index, *, valid=1, answer=1, joint=0, known=True, truth=1, sufficient=0):
    return dict(
        row_id=str(index),
        parent="b0",
        factor="P0",
        family="TOPK",
        finish_reason="balanced",
        wire_score=dict(
            A=answer,
            V=valid,
            J=joint,
            certificate_truth=0,
            certificate_sufficient=0,
            presentation=0,
            finish=1,
        ),
        diagnostic=dict(
            certificate=dict(known=known, truth=truth, sufficient=sufficient),
            core_score=dict(reason="diagnostic"),
        ),
    )


def test_parser_default_zeros_do_not_turn_unknown_into_false():
    rows = [row(0, known=False, truth=None, sufficient=None), row(1), row(2, valid=0)]
    before = copy.deepcopy(rows)
    result = j_gap_breakdown(rows)["all"]
    assert result["exclusive_gap_partition"] == dict(
        INVALID_INTERFACE=1,
        UNREADABLE_CERTIFICATE=1,
        TRUE_BUT_INSUFFICIENT=1,
        FALSE_CERTIFICATE=0,
        OTHER_J_REQUIREMENT=0,
    )
    assert result["content_certificate_labels"]["truth"] == dict(true=2, false=0, unknown=1)
    assert result["native_flag_counts_not_semantic_labels"]["certificate_truth"] == {"0": 3}
    assert result["A_correct_V_valid"] == 2
    assert rows == before


def test_wrong_answers_do_not_enter_answer_correct_gap_and_empty_denominator_is_unknown():
    report = j_gap_breakdown([row(0, valid=0, answer=0)])
    assert report["all"]["A_correct_J_zero"] == 0
    assert report["all"]["true_insufficient_fraction_of_A_correct_V_valid"] is None
    assert report["additional_admission_gate"] is False


def test_exclusive_partition_preserves_false_certificate_and_other_reason():
    result = j_gap_breakdown(
        [row(0, truth=0, sufficient=0), row(1, sufficient=1), row(2, sufficient=1, joint=1)]
    )
    assert result["all"]["exclusive_gap_partition"]["FALSE_CERTIFICATE"] == 1
    assert result["all"]["exclusive_gap_partition"]["OTHER_J_REQUIREMENT"] == 1
    assert result["all"]["J_correct"] == 1
    assert len(result["gap_rows"]) == 2


def test_duplicate_rows_and_inconsistent_original_labels_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        j_gap_breakdown([row(0), row(0)])
    with pytest.raises(ValueError, match="requires"):
        j_gap_breakdown([row(0, answer=0, joint=1)])
    with pytest.raises(ValueError, match="binary"):
        j_gap_breakdown([row(0, valid=True)])
