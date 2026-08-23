from __future__ import annotations

from compensability.study_c3.action_audit_runtime import (
    audit_existing_rows,
    summarize_action_audit,
)


def _fiber(scene_id: str) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "pair_id": scene_id.rsplit("-", 1)[0],
        "split": "dev",
        "condition": "collision",
        "family": "cross_series",
        "truth": [2, 3, 4, 5],
        "operation": {"operator": "sum", "indices": [0, 1]},
    }


def _raw(scene_id: str, arm: str, completion: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "arm": arm,
        "scene_id": scene_id,
        "pair_id": scene_id.rsplit("-", 1)[0],
        "split": "dev",
        "condition": "collision",
        "family": "cross_series",
        "rollout_index": 0,
        "seed": 11,
        "completion": completion,
        "parse_success": completion == "2,3,4,5",
        "state_reward": int(completion == "2,3,4,5"),
        "answer_reward": int(completion == "2,3,4,5"),
    }


def test_existing_rows_are_reclassified_without_overwriting_original_metrics() -> None:
    raw = (
        _raw("pair-a-collision", "C2_answer_reward", "2,3,4,5"),
        _raw("pair-b-collision", "C2_exact_state_reward", "[2,3,4,5]"),
        _raw("pair-c-collision", "C2_exact_state_reward", "v1=2,v2=3,v3=4"),
    )
    fibers = tuple(_fiber(str(row["scene_id"])) for row in raw)

    audited = audit_existing_rows(raw, fibers, token_counter=lambda text: len(text))

    assert len(audited) == 3
    assert audited[0]["P0"]["valid"] == 1
    assert audited[1]["P0"]["valid"] == 0
    assert audited[1]["P1"] == {"valid": 1, "exact": 1, "answer_correct": 1}
    assert audited[1]["original_parse_success"] is False
    assert audited[2]["failure_category"] == "labeled_incomplete_or_truncated"
    assert audited[2]["fourth_value_completed_before_token_budget"] is False


def test_summary_is_stratified_and_reports_conditional_purity() -> None:
    raw = (
        _raw("pair-a-collision", "C2_answer_reward", "2,3,4,5"),
        _raw("pair-b-collision", "C2_answer_reward", "[2,3,4,5]"),
    )
    audited = audit_existing_rows(
        raw,
        tuple(_fiber(str(row["scene_id"])) for row in raw),
        token_counter=lambda text: 4,
    )

    summary = summarize_action_audit(audited)

    arm = summary["by_arm"]["C2_answer_reward"]
    assert arm["row_count"] == 2
    assert arm["parsers"]["P0"]["parse_rate"] == 0.5
    assert arm["parsers"]["P1"]["parse_rate"] == 1.0
    assert arm["parsers"]["P1"]["truth_purity_given_valid"] == 1.0
    assert summary["by_family"]["cross_series"]["row_count"] == 2
    assert summary["by_condition"]["collision"]["row_count"] == 2
