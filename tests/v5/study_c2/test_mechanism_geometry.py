from __future__ import annotations

from compensability_v5.study_c2.correction_geometry import correction_vector_metrics
from compensability_v5.study_c2.counterfactual_relabeling import counterfactual_reward_matrix
from compensability_v5.study_c2.purity import fiber_purity, purity_update
from compensability_v5.study_c2.quotient_geometry import reward_visibility


def test_policy_weighted_fiber_purity_and_update_decomposition() -> None:
    assert fiber_purity(p_exact=0.2, p_shortcut=0.3) == 0.4
    update = purity_update(before={"X": 0.2, "S": 0.3}, after={"X": 0.25, "S": 0.5})
    assert update["delta_exact_mass"] == 0.05
    assert update["delta_answer_success_mass"] == 0.25
    assert update["delta_fiber_purity"] < 0


def test_reward_visibility_and_counterfactual_operation_matrix() -> None:
    truth = (4, 5, 6, 7)
    observation = (3, 5, 6, 7)
    operations = (
        {"operation_id": "visible", "operator": "sum", "indices": [0, 1]},
        {"operation_id": "null", "operator": "sum", "indices": [1, 2]},
    )

    spectrum = reward_visibility(truth, observation, operations)
    matrix = counterfactual_reward_matrix(
        actions=(truth, observation, (3, 6, 6, 7)),
        truth=truth,
        operations=operations,
    )

    assert spectrum == {"visible": 1, "null": 0}
    assert matrix[1]["answer_rewards"] == {"visible": 0, "null": 1}
    assert matrix[1]["state_reward"] == 0
    assert matrix[2]["shortcut_operation_count"] == 1


def test_correction_vector_geometry_separates_correct_edit_copy_and_overedit() -> None:
    operation = {"operator": "sum", "indices": [0, 1]}
    exact = correction_vector_metrics(
        truth=(4, 5, 6, 7), observation=(3, 5, 6, 7), candidate=(4, 5, 6, 7), operation=operation
    )
    copy = correction_vector_metrics(
        truth=(4, 5, 6, 7), observation=(3, 5, 6, 7), candidate=(3, 5, 6, 7), operation=operation
    )
    overedit = correction_vector_metrics(
        truth=(4, 5, 6, 7), observation=(3, 5, 6, 7), candidate=(4, 6, 6, 7), operation=operation
    )

    assert exact["support_overlap"] == 1.0
    assert exact["extra_edit_count"] == 0
    assert copy["edit_count"] == 0
    assert overedit["extra_edit_count"] == 1
    assert overedit["answer_null_component_l1"] > 0
