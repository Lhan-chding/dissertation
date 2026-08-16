from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
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
    verify_phase_c_screen_artifacts,
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


def test_trend_counterfactual_preserves_one_position_error_when_local_world_is_absent() -> None:
    amendment = load_phase_c_postscreen_amendment(
        AMENDMENT,
        screen=load_phase_c_screen_frozen_result(SCREEN_RESULT),
    )
    scene = _scene(
        "trend-fallback",
        "trend",
        (5, 8, 11, 2),
        (5, 18, 11, 2),
        operation="sum",
    )

    calls = build_phase_c_arm_calls((scene,), amendment=amendment)
    counterfactual = [call for call in calls if call.arm == "counterfactual"]

    assert len(counterfactual) == 8
    public = json.loads(counterfactual[0].messages[1]["content"])
    assert sum(
        left != right
        for left, right in zip(
            public["observed_values"], counterfactual[0].expected_values, strict=True
        )
    ) == 1


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_screen_artifact_replay_recovers_all_580_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    frozen = load_phase_c_screen_frozen_result(SCREEN_RESULT)
    paths = {
        "preflight": tmp_path / "preflight.json",
        "attempt_marker": tmp_path / "attempt.json",
        "dataset_manifest": tmp_path / "manifest.json",
        "dataset_records": tmp_path / "dataset.jsonl",
        "screen_report": tmp_path / "report.json",
        "screen_records": tmp_path / "screen.jsonl",
        "console": tmp_path / "console.log",
    }
    for label in ("preflight", "attempt_marker", "dataset_manifest"):
        paths[label].write_text("{}\n", encoding="utf-8")
    paths["console"].write_text("phase_c_screen_exit=3\n", encoding="utf-8")
    report = {
        "scenes": 8000,
        "model_calls": 8000,
        "parse_successes": 7905,
        "parse_rate": 0.988125,
        "screen_passed": False,
        "confirmatory_arm_execution_authorized": False,
        "training_invoked": False,
    }
    paths["screen_report"].write_text(json.dumps(report) + "\n", encoding="utf-8")
    family_counts = {"cross_series": 208, "duplicate_encoding": 182, "trend": 190}
    family_sequence = [
        family for family, count in family_counts.items() for _index in range(count)
    ]
    dataset_lines: list[str] = []
    screen_lines: list[str] = []
    for index in range(8000):
        scene_id = f"phase-c-screen-{index:06d}"
        dataset_lines.append(json.dumps({"scene_id": scene_id}))
        if index < 580:
            family = family_sequence[index]
            truth = [4, 6, 8, 10] if family == "trend" else [8, 4, 5, 9]
            perceived = [truth[0] - 1, *truth[1:]]
            row = {
                "scene_id": scene_id,
                "family": family,
                "chart_type": "line" if family == "trend" else "grouped_bar",
                "operation": "sum",
                "values": truth,
                "perceived_values": perceived,
                "parse_success": True,
                "natural_perception_error": True,
                "one_position_error": True,
                "operator_sensitive": True,
                "design_recoverability_validated": True,
                "eligible": True,
            }
        else:
            row = {"scene_id": scene_id, "eligible": False}
        screen_lines.append(json.dumps(row))
    paths["dataset_records"].write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
    paths["screen_records"].write_text("\n".join(screen_lines) + "\n", encoding="utf-8")
    replay = replace(
        frozen,
        source_sha256=tuple(sorted((label, _digest(path)) for label, path in paths.items())),
    )

    scenes = verify_phase_c_screen_artifacts(
        replay,
        preflight=paths["preflight"],
        attempt_marker=paths["attempt_marker"],
        dataset_manifest=paths["dataset_manifest"],
        dataset_records=paths["dataset_records"],
        screen_report=paths["screen_report"],
        screen_records=paths["screen_records"],
        console_log=paths["console"],
    )
    assert len(scenes) == 580
    assert Counter(scene.family for scene in scenes) == Counter(family_counts)

    paths["console"].write_text("phase_c_screen_exit=0\n", encoding="utf-8")
    try:
        verify_phase_c_screen_artifacts(
            replay,
            preflight=paths["preflight"],
            attempt_marker=paths["attempt_marker"],
            dataset_manifest=paths["dataset_manifest"],
            dataset_records=paths["dataset_records"],
            screen_report=paths["screen_report"],
            screen_records=paths["screen_records"],
            console_log=paths["console"],
        )
    except ValueError as error:
        assert "differs from frozen evidence" in str(error)
    else:
        raise AssertionError("tampered Phase C screen evidence was accepted")
