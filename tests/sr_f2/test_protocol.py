import copy

import pytest

from sr_f2.protocol import ARMS, build_config, scientific_matrix, technical_matrix, validate_config


def test_exact_matrix_and_user_clarifications():
    plan = build_config()
    assert validate_config(plan) == plan
    assert len(scientific_matrix()) == 12
    assert {row["arm"] for row in technical_matrix()} == set(ARMS)
    assert all(row["updates"] == 96 for row in scientific_matrix())
    assert plan["diagnostic_views"]["protocol_questions"] == 512
    assert plan["statistics"]["c4_seed_rule"] == "all_three_strictly_gt_minus_2pp"
    assert plan["statistics"]["c4_secondary_plus_2pp_is_gate"] is False
    assert plan["resources"]["max_teacher_qos_gpus"] == 8
    assert plan["resources"]["max_teacher_qos_submitted_jobs_per_user"] == 5


@pytest.mark.parametrize(
    "field,value", [("lr", 1e-4), ("updates", 104), ("gradient_microbatch_sequences", 2)]
)
def test_frozen_training_recipe_cannot_drift(field, value):
    plan = build_config()
    plan["training"][field] = value
    with pytest.raises(PermissionError):
        validate_config(plan)


def test_independent_copy_and_parent_history_not_inherited():
    plan = build_config()
    other = copy.deepcopy(plan)
    validated = validate_config(plan)
    validated["training"]["arms"].clear()
    assert plan == other
    assert plan["history_policy"]["preserve_all_prior_results"]
    assert "delete" not in str(plan["history_policy"])
    assert plan["technical_check"]["maximum_lr_adjustments"] == 0
