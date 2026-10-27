import copy

import pytest

from sr_f6.credit import build_credit_group
from srf6_ref.credit import credit as reference_credit


def rows():
    return [
        {
            "row_id": f"row-{index}",
            "group_id": "group-0",
            "qid": "qid-0",
            "policy_hash": "a" * 64,
            "parent": "b0",
            "factor": "P1",
            "root_id": "root-0",
            "family": "BINDING",
            "draw": index,
            "J": 0,
            "V": 1,
            "finished": 1,
            "S": int(index == 7),
            "B": int(index == 7),
        }
        for index in range(8)
    ]


def test_legal_extreme_and_all_counterfactuals_match_reference():
    data = rows()
    original = copy.deepcopy(data)
    receipt = build_credit_group(data, "U11", 17)
    assert receipt["applied"] == [-0.0625] * 7 + [0.4375]
    for mode, actual in receipt["counterfactuals"].items():
        expected = reference_credit(data, mode, seed=17)
        assert actual["applied"] == expected["applied"]
        assert actual["candidate"] == expected["candidate"]
    assert data == original


def test_unknown_uses_separate_strata_without_reweighting():
    data = rows()
    for row in data:
        row["B"] = None
    receipt = build_credit_group(data, "U11", 19)
    assert receipt["applied"] == [-0.03125] * 7 + [0.21875]
    data[7]["B"] = 1
    assert build_credit_group(data, "U11", 19)["applied"] == [0.0] * 8


def test_invalid_or_incomplete_rows_remain_in_native_and_batch_denominator():
    data = rows()
    data[7].update(V=0, finished=0)
    receipt = build_credit_group(data, "U11", 17)
    assert receipt["applied"] == [0.0] * 8
    assert receipt["counterfactuals"]["COND"]["denominator"] == 8
    assert receipt["counterfactuals"]["COND"]["eligible_count"] == 7
    assert len(receipt["coefficients_by_row_id"]) == 8


def test_single_shuffle_preserves_multiset_and_full_mapping():
    data = rows()
    receipt = build_credit_group(data, "C_SHUF", 17)
    shuffled = receipt["counterfactuals"]["SHUF"]
    assert sorted(shuffled["candidate"]) == sorted(shuffled["applied"])
    assert sum(shuffled["applied"]) == 0
    assert shuffled["seed"] == 17
    for destination, source in enumerate(shuffled["destination_to_source_index"]):
        assert shuffled["applied"][destination] == shuffled["candidate"][source]
        assert shuffled["destination_to_source_row_id"][destination] == f"row-{source}"


def test_identity_permutation_is_kept_without_retry():
    # With this fixed seed the unique nonzero row retains its coefficient.
    data = rows()
    receipt = build_credit_group(data, "C_SHUF", 9)
    expected = reference_credit(data, "SHUF", seed=9)
    assert expected["permutation_changed"] is False
    assert receipt["applied"] == expected["applied"]
    assert (
        receipt["counterfactuals"]["SHUF"]["permutation_changed"] == expected["permutation_changed"]
    )


def test_canonical_draw_order_and_pure_j_control():
    data = rows()
    data[0]["J"] = 1
    result = build_credit_group(data, "U10", 7)
    assert result == build_credit_group(list(reversed(data)), "U10", 7)
    assert result["applied"] == [0.0] * 8
    assert (
        result["counterfactuals"]["GLOBAL"]["applied"]
        != result["counterfactuals"]["COND"]["applied"]
    )


@pytest.mark.parametrize(
    "field,value", (("draw", 0), ("row_id", "row-0"), ("qid", "other"), ("policy_hash", "b" * 64))
)
def test_duplicate_or_mixed_identity_rejected(field, value):
    data = rows()
    data[7][field] = value
    with pytest.raises(ValueError):
        build_credit_group(data, "U11", 7)


@pytest.mark.parametrize(
    "field,value", (("J", float("nan")), ("S", float("inf")), ("B", 0.5), ("V", "1"))
)
def test_invalid_or_nonfinite_labels_rejected(field, value):
    data = rows()
    data[0][field] = value
    with pytest.raises(ValueError):
        build_credit_group(data, "U11", 7)


def test_registered_dose_factor_and_completion_cannot_drift():
    with pytest.raises(ValueError, match="lambda"):
        build_credit_group(rows(), "U11", 7, aux_coefficient=0.25)
    with pytest.raises(ValueError, match="weights"):
        build_credit_group(rows(), "U11", 7, component_weights={"S": 1.0, "B": 0.0})
    with pytest.raises(ValueError, match="factor"):
        build_credit_group(rows(), "U01", 7)
    data = rows()
    data[0].update(J=1, V=0)
    with pytest.raises(ValueError, match="J=1"):
        build_credit_group(data, "U11", 7)
