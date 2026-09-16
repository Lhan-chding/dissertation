import numpy as np
import pytest

from src.modeling_v4.calibration_selection import nested_bank_order


def test_block_qr_prefers_new_directions_and_keeps_whole_nested_banks():
    # B has a larger initial trace, but contributes no direction after A.
    blocks = np.array([[[3.0, 0], [2.0, 0]], [[2.0, 0], [2.0, 0]], [[0, 1], [0, 1]]])
    e = blocks.reshape(6, 2)
    gram = e @ e.T
    frozen = gram.copy()
    order = nested_bank_order(["A", "B", "C"], gram, columns_per_bank=2)
    np.testing.assert_array_equal(order, [0, 2, 1])
    np.testing.assert_array_equal(gram, frozen)
    rotated = e @ np.array([[0.0, 1], [-1, 0]])
    np.testing.assert_array_equal(
        nested_bank_order(["A", "B", "C"], rotated @ rotated.T, columns_per_bank=2), order
    )
    np.testing.assert_array_equal(
        nested_bank_order(["A", "B", "C"], gram * 1e-24, columns_per_bank=2), order
    )


def test_stratified_random_prefix_uses_metadata_without_changing_full_input():
    ids = [f"b{i}" for i in range(8)]
    strata = ["a"] * 4 + ["b"] * 4
    gram = np.eye(24)
    first = nested_bank_order(ids, gram, strata=strata, method="STRATIFIED_RANDOM", seed=41001)
    second = nested_bank_order(ids, gram, strata=strata, method="STRATIFIED_RANDOM", seed=41001)
    np.testing.assert_array_equal(first, second)
    assert sorted(first) == list(range(8))
    for m in [2, 4, 6, 8]:
        assert sum(strata[i] == "a" for i in first[:m]) == m // 2


def test_duplicate_and_non_psd_inputs_rejected_without_mutation():
    with pytest.raises(ValueError, match="Unique"):
        nested_bank_order(["same", "same"], np.eye(6))
    with pytest.raises(ValueError, match="semidefinite"):
        nested_bank_order(["one"], np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="two registered"):
        nested_bank_order(["one"], np.eye(3), method="BLOCK_LOGDET")
