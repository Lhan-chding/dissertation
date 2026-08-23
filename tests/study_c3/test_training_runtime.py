from __future__ import annotations

from pathlib import Path

from compensability.study_c3.factorial_design import (
    build_factorial_arms,
    validate_study_c3_config,
)
from compensability.study_c3.io import read_jsonl
from compensability.study_c3.training_runtime import (
    build_group_diagnostics,
    build_traced_reward,
)

from .test_factorial_pairing import _config


def _row() -> dict[str, object]:
    return {
        "scene_id": "scene-a",
        "pair_id": "pair-a",
        "condition": "collision",
        "family": "cross_series",
        "prompt_sha256": "a" * 64,
        "truth": [2, 3, 4, 5],
        "operation": {"operator": "sum", "indices": [0, 1]},
    }


def test_traced_reward_preserves_semantic_validity_and_combined_channels(tmp_path: Path) -> None:
    arm = build_factorial_arms(validate_study_c3_config(_config()), initialization_hash="b" * 64)[3]
    trace = tmp_path / "raw_reward_trace.jsonl"
    reward = build_traced_reward(
        arm_config=arm,
        training_rows=(_row(),),
        trace_path=trace,
        group_size=8,
        token_counter=lambda text: len(text.split(",")),
    )
    completions = [
        "2,3,4,5",
        "v1=3,v2=2,v3=4,v4=6",
        "2,4,4,5",
        "not an action",
    ] * 2

    selected = reward(completions, scene_id=["scene-a"], trainer_state=None)

    assert selected == [3.0, 1.0, 1.0, 0.0] * 2
    rows = read_jsonl(trace)
    assert [row["action_class"] for row in rows[:4]] == ["X", "S", "W", "I"]
    assert [row["semantic_reward"] for row in rows[:4]] == [1, 0, 0, 0]
    assert [row["validity_reward"] for row in rows[:4]] == [1, 1, 1, 0]
    assert all("combined_reward" in row for row in rows)
    assert all("failure_category" in row for row in rows)

    diagnostics = build_group_diagnostics(trace, expected_group_count=1, group_size=8)
    assert diagnostics[0]["counts"] == {"X": 2, "S": 2, "W": 2, "I": 2}
    assert diagnostics[0]["validity_bearing_group"] is True
    assert diagnostics[0]["truth_bearing_group"] is True
    assert diagnostics[0]["answer_disagreement_group"] is True
