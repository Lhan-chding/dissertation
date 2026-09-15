"""M1 constructive counterexamples and explicit positive controls."""

import json

import numpy as np
import pytest

from src.modeling_qualification.witnesses import (
    _adam_history,
    run_witnesses,
    witness_w1,
    witness_w2,
    witness_w3,
    witness_w4,
    witness_w5,
    witness_w6,
)


def test_w1_keeps_prompt_identity():
    w = witness_w1()
    assert w["pooled_change_norm"] == 0
    assert w["per_prompt_change_norm"] > 0
    assert w["prompt_ids"] == ["a", "b"]
    np.testing.assert_array_equal(np.array(w["after"]), np.array(w["before"])[::-1])


def test_w2_equal_present_probabilities_do_not_identify_response():
    w = witness_w2()
    np.testing.assert_array_equal(w["p_start_plus"], w["p_start_minus"])
    assert w["delta_pX_plus"] > 0 and w["delta_pX_minus"] < 0
    assert w["same_update"]


def test_w3_legal_adam_histories_same_parameter_gradient():
    w = witness_w3()
    assert w["same_theta"] and w["same_current_gradient"]
    assert w["updates"][0] * w["updates"][1] < 0
    assert w["same_history_replay_equal"]
    assert w["explicit_zero_gradient_update"] != 0
    assert w["none_gradient_update"] == 0
    assert max(w["actual_d_first_order_errors"]) < max(w["p_only_errors"])
    assert w["global_rng_unchanged"]


@pytest.mark.parametrize("history", [[-0.8] * 3, [0.8] * 3])
def test_w3_history_keeps_exact_zero_and_preserves_legal_moments(history):
    """A floating-point inverse replay must not manufacture equal starting theta."""
    parameter, optimizer, initial = _adam_history(history)
    assert initial == 0.0
    assert float(parameter.detach()[0]) == 0.0
    assert optimizer.param_groups[0]["lr"] == 0.01
    state = optimizer.state[parameter]
    expected_m, expected_v = 0.0, 0.0
    for gradient in history:
        expected_m = 0.9 * expected_m + (1 - 0.9) * gradient
        expected_v = 0.999 * expected_v + (1 - 0.999) * gradient**2
    assert int(state["step"]) == len(history)
    np.testing.assert_allclose(state["exp_avg"].numpy(), [expected_m], rtol=0, atol=1e-16)
    np.testing.assert_allclose(state["exp_avg_sq"].numpy(), [expected_v], rtol=0, atol=1e-18)


def test_w4_same_summary_opposite_semantics_and_minimax_bound():
    w = witness_w4()
    assert [r["h"] for r in w["rows"]] == [0.01, 0.1, 0.5]
    for r in w["rows"]:
        assert r["projection_plus"] == r["projection_minus"]
        assert r["residual_norm_plus"] == r["residual_norm_minus"]
        assert r["delta_pX_plus"] > 0 > r["delta_pX_minus"]
        assert r["minimax_absolute_error_lower_bound"] == pytest.approx(
            (r["pX_plus"] - r["pX_minus"]) / 2
        )


def test_w5_rank_and_predefined_low_energy_positive_negative_controls():
    w = witness_w5()
    assert w["exact_rank4"]["response_rank"] == 4
    assert (
        w["exact_rank4"]["update_rank_before"] == w["exact_rank4"]["update_rank_after_duplication"]
    )
    assert len(w["low_energy_rows"]) == 2
    assert {r["energy_rule_passes_group_threshold"] for r in w["low_energy_rows"]} == {False, True}
    for r in w["low_energy_rows"]:
        assert r["energy_rank"] == 4
        assert r["group_selected_worst_error"] <= r["group_threshold"]


def test_w6_intermediate_net_path_and_wrong_omission_model():
    w = witness_w6()
    excursion = w["paths"]["out_and_return"]
    assert excursion[0]["absolute_error"] > 0
    assert excursion[-1]["absolute_error"] == 0
    assert excursion[-1]["net_bound"] == 0
    assert excursion[-1]["path_bound"] > 0
    assert w["endpoint_only_misses_excursion"]
    assert w["wrong_zero_omission_model_broken"]
    for path in w["paths"].values():
        for row in path:
            assert row["absolute_error"] <= row["net_bound"] + 1e-14
            assert row["net_bound"] <= row["path_bound"] + 1e-14


def test_witness_runner_no_clobber_and_recomputable_arrays(tmp_path):
    result = run_witnesses({}, tmp_path / "M1")
    assert result["status"] == "PASS"
    assert set(result["witnesses"]) == {"W1", "W2", "W3", "W4", "W5", "W6"}
    saved = json.loads((tmp_path / "M1" / "identifiability_witnesses.json").read_text())
    assert saved["execution_kind"] == "ANALYTIC_FIXTURE"
    with pytest.raises(FileExistsError):
        run_witnesses({}, tmp_path / "M1")
