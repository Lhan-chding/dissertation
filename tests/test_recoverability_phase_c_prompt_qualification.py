from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from compbias.recoverability.phase_c_prompt_qualification import (
    PROMPT_QUALIFICATION_CONDITIONS,
    PhaseCPromptQualificationConfig,
    build_phase_c_prompt_qualification_calls,
    evaluate_phase_c_prompt_qualification_call,
    load_phase_c_prompt_qualification_config,
    summarize_phase_c_prompt_qualification,
)
from compbias.recoverability.phase_c_screen_result import FrozenEligibleScene

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/recoverability/phase_c_prompt_qualification_v1.yaml"


def _config() -> PhaseCPromptQualificationConfig:
    return load_phase_c_prompt_qualification_config(CONFIG)


def _scenes() -> tuple[FrozenEligibleScene, ...]:
    scenes: list[FrozenEligibleScene] = []
    index = 0
    for family in ("cross_series", "duplicate_encoding", "trend"):
        for operation in ("difference", "sum", "max_minus_min"):
            truth = (9, 4, 6, 2) if family != "trend" else (2, 4, 6, 8)
            perceived = (8, 4, 6, 2) if family != "trend" else (3, 4, 6, 8)
            scenes.append(
                FrozenEligibleScene(
                    scene_id=f"synthetic-{index:02d}",
                    family=family,
                    chart_type="line",
                    operation=operation,
                    true_values=truth,
                    perceived_values=perceived,
                )
            )
            index += 1
    return tuple(scenes)


def _strict_program(values: tuple[int, int, int, int], operation: str) -> str:
    variables = dict(zip(("a", "b", "c", "d"), values, strict=True))
    if operation == "sum":
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
    elif operation == "difference":
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
    else:
        steps = [
            {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
            {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
    return json.dumps({"variables": variables, "steps": steps, "return": "result"})


def test_prompt_qualification_config_freezes_only_36_diagnostic_calls() -> None:
    config = _config()
    assert config.schema_version == 1
    assert config.conditions == PROMPT_QUALIFICATION_CONDITIONS
    assert config.families == ("cross_series", "duplicate_encoding", "trend")
    assert config.operations == ("difference", "sum", "max_minus_min")
    assert config.scenes_per_cell == 1
    assert config.forks_per_condition == 2
    assert config.model_call_cap == 36
    assert config.format_retries == 0
    assert config.hypothesis_tested is False
    assert config.scale_authorized is False
    assert config.training_authorized is False
    assert config.rl_authorized is False


def test_call_plan_is_balanced_and_prompt_explains_the_constraint_language() -> None:
    calls = build_phase_c_prompt_qualification_calls(_scenes(), config=_config())
    assert len(calls) == 36
    assert len({call.call_id for call in calls}) == 36
    assert len({call.scene_id for call in calls}) == 9
    assert Counter((call.family, call.operation, call.condition) for call in calls) == {
        (family, operation, condition): 2
        for family in ("cross_series", "duplicate_encoding", "trend")
        for operation in ("difference", "sum", "max_minus_min")
        for condition in PROMPT_QUALIFICATION_CONDITIONS
    }
    for call in calls:
        system = str(call.messages[0]["content"])
        assert "0 means a, 1 means b, 2 means c, and 3 means d" in system
        assert "known_value" in system
        assert "pair_sum" in system
        assert "arithmetic_progression" in system
        assert '"op", "inputs", and "output"' in system
        assert "No Markdown" in system
        evidence = json.loads(str(call.messages[1]["content"]))
        if call.condition == "valid_cue":
            assert evidence["redundant_facts"]
        else:
            assert evidence["redundant_facts"] == []


def test_selection_is_deterministic_when_more_than_one_scene_is_available() -> None:
    extras = tuple(
        FrozenEligibleScene(
            scene_id=f"extra-{scene.scene_id}",
            family=scene.family,
            chart_type=scene.chart_type,
            operation=scene.operation,
            true_values=scene.true_values,
            perceived_values=scene.perceived_values,
        )
        for scene in _scenes()
    )
    forward = build_phase_c_prompt_qualification_calls(_scenes() + extras, config=_config())
    reverse = build_phase_c_prompt_qualification_calls(
        tuple(reversed(_scenes() + extras)), config=_config()
    )
    assert tuple(call.call_id for call in forward) == tuple(call.call_id for call in reverse)


def test_scoring_keeps_format_and_semantic_content_as_separate_metrics() -> None:
    call = next(
        item
        for item in build_phase_c_prompt_qualification_calls(_scenes(), config=_config())
        if item.condition == "valid_cue" and item.operation == "sum"
    )
    strict = evaluate_phase_c_prompt_qualification_call(
        call, _strict_program(call.expected_values, call.operation)
    )
    assert strict.strict_parse_success is True
    assert strict.strict_execution_success is True
    assert strict.semantic_world_extracted is True
    assert strict.semantic_world_exact is True
    assert strict.semantic_answer_correct is True

    fenced = evaluate_phase_c_prompt_qualification_call(
        call,
        "```json\n"
        + json.dumps({"variables": dict(zip("abcd", call.expected_values, strict=True))})
        + "\n```",
    )
    assert fenced.strict_parse_success is False
    assert fenced.semantic_world_extracted is True
    assert fenced.semantic_world_exact is True
    assert fenced.semantic_answer_correct is True

    malformed_outer_json = (
        '{"variables":'
        + json.dumps(dict(zip("abcd", call.expected_values, strict=True)))
        + ',"steps":[{"op":"add","inputs":[a,b],"output":"result"}],'
        '"return":"result"}'
    )
    semantic_only = evaluate_phase_c_prompt_qualification_call(call, malformed_outer_json)
    assert semantic_only.strict_parse_success is False
    assert semantic_only.semantic_values == call.expected_values
    assert semantic_only.semantic_answer_correct is True

    array_world = evaluate_phase_c_prompt_qualification_call(
        call, json.dumps({"variables": list(call.expected_values)})
    )
    assert array_world.strict_parse_success is False
    assert array_world.semantic_values == call.expected_values


def test_summary_never_turns_the_diagnostic_into_scale_authorization() -> None:
    calls = build_phase_c_prompt_qualification_calls(_scenes(), config=_config())
    records = tuple(
        evaluate_phase_c_prompt_qualification_call(
            call, _strict_program(call.expected_values, call.operation)
        )
        for call in calls
    )
    report = summarize_phase_c_prompt_qualification(records, config=_config())
    assert report["model_calls"] == 36
    assert report["strict_schema_parse_rate"] == 1.0
    assert report["semantic_world_extraction_rate"] == 1.0
    assert report["semantic_world_exact_rate_over_all"] == 1.0
    assert report["semantic_answer_accuracy_over_all"] == 1.0
    assert report["scale_authorized"] is False
    assert report["hypothesis_tested"] is False
    assert report["training_invoked"] is False
    assert set(report["by_condition"]) == set(PROMPT_QUALIFICATION_CONDITIONS)

    failures = tuple(
        evaluate_phase_c_prompt_qualification_call(call, "not JSON") for call in calls
    )
    failed_report = summarize_phase_c_prompt_qualification(failures, config=_config())
    assert failed_report["semantic_world_extraction_rate"] == 0.0
    assert failed_report["semantic_world_exact_rate_among_extracted"] == 0.0
    assert failed_report["semantic_answer_accuracy_among_extracted"] == 0.0


def test_config_rejects_any_attempt_to_expand_the_call_cap(tmp_path: Path) -> None:
    mutated = CONFIG.read_text(encoding="utf-8").replace("model_call_cap: 36", "model_call_cap: 37")
    path = tmp_path / "expanded.yaml"
    path.write_text(mutated, encoding="utf-8")
    with pytest.raises(ValueError, match="frozen 36-call diagnostic"):
        load_phase_c_prompt_qualification_config(path)


def test_call_plan_and_summary_reject_incomplete_or_ambiguous_inputs() -> None:
    config = _config()
    with pytest.raises(ValueError, match="non-empty tuple"):
        build_phase_c_prompt_qualification_calls((), config=config)
    with pytest.raises(TypeError, match="invalid item"):
        build_phase_c_prompt_qualification_calls((object(),), config=config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identifiers must be unique"):
        build_phase_c_prompt_qualification_calls((*_scenes(), _scenes()[0]), config=config)
    with pytest.raises(ValueError, match=r"no eligible scene for trend\|max_minus_min"):
        build_phase_c_prompt_qualification_calls(_scenes()[:-1], config=config)
    with pytest.raises(TypeError, match="config must be"):
        build_phase_c_prompt_qualification_calls(_scenes(), config=object())  # type: ignore[arg-type]

    calls = build_phase_c_prompt_qualification_calls(_scenes(), config=config)
    with pytest.raises(TypeError, match="call must be"):
        evaluate_phase_c_prompt_qualification_call(object(), "{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="raw_text must be text"):
        evaluate_phase_c_prompt_qualification_call(calls[0], object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact frozen 36-call diagnostic"):
        summarize_phase_c_prompt_qualification((), config=config)

    records = tuple(
        evaluate_phase_c_prompt_qualification_call(call, "not JSON") for call in calls
    )
    with pytest.raises(ValueError, match="identifiers must be unique"):
        summarize_phase_c_prompt_qualification((*records[:-1], records[0]), config=config)
