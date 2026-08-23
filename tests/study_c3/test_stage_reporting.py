from __future__ import annotations

from compensability.study_c3.report_runtime import fact_report_markdown
from compensability.study_c3.stage_a_b_runtime import build_failure_examples


def test_failure_examples_are_grouped_facts_without_reclassification() -> None:
    rows = (
        {
            "arm": "C2_answer_reward",
            "failure_category": "free_prose_without_complete_action",
            "scene_id": "scene-a",
            "completion": "Let's solve...",
        },
        {
            "arm": "C2_exact_state_reward",
            "failure_category": "labeled_incomplete_or_truncated",
            "scene_id": "scene-b",
            "completion": "v1=2,v2=3,v3=4",
        },
    )

    rendered = build_failure_examples(rows, limit_per_cell=1)

    assert "C2_answer_reward" in rendered
    assert "free_prose_without_complete_action" in rendered
    assert "Let's solve..." in rendered
    assert "v1=2,v2=3,v3=4" in rendered


def test_result_report_is_fact_only_and_marks_single_seed_scope() -> None:
    rendered = fact_report_markdown(
        {
            "git_commit_sha": "a" * 40,
            "model_snapshot_sha256": "b" * 64,
            "single_seed_mechanism_pilot": True,
            "action_audit": {"row_count": 5632},
            "analysis": {"primary_outcomes": ["action_validity"]},
        }
    )

    assert "single_seed_mechanism_pilot: true" in rendered
    assert "git_commit_sha" in rendered
    assert "机制结论" not in rendered
    assert "建议" not in rendered
