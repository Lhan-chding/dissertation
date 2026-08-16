from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from compbias.recoverability.phase_c_arms import (
    PHASE_C_ARMS,
    FrozenEligibleScene,
    build_phase_c_arm_calls,
    evaluate_phase_c_arm_call,
)
from compbias.recoverability.phase_c_postscreen_amendment import (
    load_phase_c_postscreen_amendment,
)
from compbias.recoverability.phase_c_screen_result import (
    load_phase_c_screen_frozen_result,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN_RESULT = ROOT / "configs/recoverability/phase_c_screen_v2_frozen_result.yaml"
AMENDMENT = ROOT / "configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml"


def _scene(
    scene_id: str,
    family: str,
    truth: tuple[int, int, int, int],
    perceived: tuple[int, int, int, int],
    operation: str = "sum",
) -> FrozenEligibleScene:
    return FrozenEligibleScene(
        scene_id=scene_id,
        family=family,
        chart_type="line" if family == "trend" else "grouped_bar",
        operation=operation,
        true_values=truth,
        perceived_values=perceived,
    )


def test_postscreen_amendment_preserves_failure_and_freezes_all_eligible() -> None:
    screen = load_phase_c_screen_frozen_result(SCREEN_RESULT)
    amendment = load_phase_c_postscreen_amendment(AMENDMENT, screen=screen)

    assert screen.screen_passed is False
    assert screen.phase_c_screen_exit == 3
    assert screen.eligible_scenes == 580
    assert dict(screen.eligible_by_family) == {
        "cross_series": 208,
        "duplicate_encoding": 182,
        "trend": 190,
    }
    assert amendment.original_screen_passed is False
    assert amendment.fixed_family_quota_gate_withdrawn is True
    assert amendment.arm_outcomes_observed is False
    assert amendment.frozen_eligible_scenes == 580
    assert amendment.model_call_cap == 27_840
    assert amendment.original_target_power_met is False
    assert amendment.confirmatory_arm_execution_authorized is True
    assert amendment.training_authorized is False
    assert amendment.rl_authorized is False


def test_arm_plan_is_complete_deterministic_and_gold_free() -> None:
    amendment = load_phase_c_postscreen_amendment(
        AMENDMENT,
        screen=load_phase_c_screen_frozen_result(SCREEN_RESULT),
    )
    scenes = (
        _scene("cross", "cross_series", (8, 4, 5, 9), (7, 4, 5, 9)),
        _scene("duplicate", "duplicate_encoding", (7, 4, 5, 9), (6, 4, 5, 9)),
        _scene("trend", "trend", (4, 6, 8, 10), (3, 6, 8, 10)),
    )

    first = build_phase_c_arm_calls(scenes, amendment=amendment)
    second = build_phase_c_arm_calls(scenes, amendment=amendment)

    assert first == second
    assert len(first) == 3 * 6 * 8
    assert len({call.call_id for call in first}) == len(first)
    assert Counter(call.arm for call in first) == Counter({arm: 24 for arm in PHASE_C_ARMS})
    assert set(Counter((call.scene_id, call.arm) for call in first).values()) == {8}
    for call in first:
        serialized = json.dumps(call.messages, sort_keys=True)
        assert "true_values" not in serialized
        assert "gold_answer" not in serialized
        assert "cue_condition" not in serialized
        assert call.image_available is False
        assert call.format_retries == 0


def test_arm_scoring_requires_parse_execution_answer_and_dataflow() -> None:
    amendment = load_phase_c_postscreen_amendment(
        AMENDMENT,
        screen=load_phase_c_screen_frozen_result(SCREEN_RESULT),
    )
    scene = _scene("cross", "cross_series", (8, 4, 5, 9), (7, 4, 5, 9))
    valid_call = next(
        call
        for call in build_phase_c_arm_calls((scene,), amendment=amendment)
        if call.arm == "valid" and call.fork_index == 0
    )
    raw = json.dumps(
        {
            "variables": {"a": 8, "b": 4, "c": 5, "d": 9},
            "steps": [{"op": "add", "inputs": ["a", "b"], "output": "result"}],
            "return": "result",
        },
        separators=(",", ":"),
    )

    passed = evaluate_phase_c_arm_call(valid_call, raw)
    failed = evaluate_phase_c_arm_call(valid_call, "not-json")

    assert passed.program_parse_success is True
    assert passed.program_execution_success is True
    assert passed.answer_correct is True
    assert passed.required_cue_on_dataflow is True
    assert passed.faithful_success is True
    assert failed.faithful_success is False
    assert failed.error_code == "program_parse_failure"
