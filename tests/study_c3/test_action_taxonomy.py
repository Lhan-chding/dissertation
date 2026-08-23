from __future__ import annotations

from compensability.study_c3.action_taxonomy import (
    ActionClass,
    FailureCategory,
    classify_action,
    classify_failure,
)

OPERATION = {"operator": "sum", "indices": [0, 1]}
TRUTH = (2, 3, 4, 5)


def test_x_s_w_i_are_mutually_exclusive() -> None:
    assert classify_action(TRUTH, truth=TRUTH, operation=OPERATION).action_class is ActionClass.X
    assert (
        classify_action((3, 2, 4, 6), truth=TRUTH, operation=OPERATION).action_class
        is ActionClass.S
    )
    assert (
        classify_action((2, 4, 4, 5), truth=TRUTH, operation=OPERATION).action_class
        is ActionClass.W
    )
    assert classify_action(None, truth=TRUTH, operation=OPERATION).action_class is ActionClass.I


def test_failure_taxonomy_has_registered_precedence() -> None:
    assert classify_failure("2,3,4,5") is FailureCategory.CANONICAL_VALID_IN_DOMAIN
    assert classify_failure("[2,3,4,5]") is FailureCategory.SEMANTIC_VALID_NONCANONICAL
    assert classify_failure("9,24,12,14") is FailureCategory.COMPLETE_OUT_OF_DOMAIN
    assert classify_failure("v1=2,v2=3,v3=4") is FailureCategory.LABELED_INCOMPLETE_OR_TRUNCATED
    assert (
        classify_failure("Let's solve the problem carefully.")
        is FailureCategory.FREE_PROSE_WITHOUT_COMPLETE_ACTION
    )
    assert classify_failure("???") is FailureCategory.OTHER_INVALID
