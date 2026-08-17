from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from compbias.recoverability.phase_c_screen_result import FrozenEligibleScene
from compbias.recoverability.phase_c_world_recovery import (
    WORLD_RECOVERY_CONDITIONS,
    PhaseCWorldRecoveryConfig,
    build_phase_c_world_recovery_calls,
    evaluate_phase_c_world_recovery_call,
    load_phase_c_world_recovery_config,
    parse_world_recovery_output,
    summarize_phase_c_world_recovery,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/recoverability/phase_c_world_recovery_v1.yaml"
PROMPT = ROOT / "prompts/world_recovery_v1_main.system.txt"


def _config() -> PhaseCWorldRecoveryConfig:
    return load_phase_c_world_recovery_config(CONFIG)


def _scenes() -> tuple[FrozenEligibleScene, ...]:
    rows = (
        ("cross-a", "cross_series", (9, 4, 6, 2), (8, 4, 6, 2)),
        ("cross-b", "cross_series", (7, 11, 5, 14), (7, 10, 5, 14)),
        ("cross-extra", "cross_series", (10, 3, 8, 12), (10, 3, 9, 12)),
        ("duplicate-a", "duplicate_encoding", (12, 5, 18, 9), (12, 5, 18, 7)),
        ("duplicate-b", "duplicate_encoding", (10, 14, 12, 3), (9, 14, 12, 3)),
        ("duplicate-extra", "duplicate_encoding", (8, 6, 13, 4), (8, 6, 13, 5)),
        ("trend-a", "trend", (2, 4, 6, 8), (3, 4, 6, 8)),
        ("trend-b", "trend", (3, 5, 7, 9), (3, 6, 7, 9)),
        ("trend-extra", "trend", (4, 7, 10, 13), (4, 7, 11, 13)),
    )
    return tuple(
        FrozenEligibleScene(
            scene_id=scene_id,
            family=family,
            chart_type="line",
            operation="difference",
            true_values=truth,
            perceived_values=observed,
        )
        for scene_id, family, truth, observed in rows
    )


def test_world_recovery_config_freezes_exactly_twelve_calls() -> None:
    config = _config()
    assert config.schema_version == 1
    assert config.qualification_id == "recoverability-phase-c-world-recovery-v1r1"
    assert config.output_subdirectory.endswith("phase_c_world_recovery_v1r1")
    assert config.conditions == WORLD_RECOVERY_CONDITIONS
    assert config.families == ("cross_series", "duplicate_encoding", "trend")
    assert config.cases_per_family == 2
    assert config.model_call_cap == 12
    assert config.max_new_tokens == 32
    assert config.do_sample is False
    assert config.format_retries == 0
    assert config.hypothesis_tested is False
    assert config.scale_authorized is False
    assert config.training_authorized is False
    assert config.rl_authorized is False


def test_call_plan_is_deterministic_balanced_and_world_only() -> None:
    system_prompt = PROMPT.read_text(encoding="utf-8")
    calls = build_phase_c_world_recovery_calls(
        _scenes(), config=_config(), system_prompt=system_prompt
    )
    reverse = build_phase_c_world_recovery_calls(
        tuple(reversed(_scenes())), config=_config(), system_prompt=system_prompt
    )
    assert len(calls) == 12
    assert tuple(call.call_id for call in calls) == tuple(call.call_id for call in reverse)
    assert len({call.scene_id for call in calls}) == 6
    assert Counter((call.family, call.condition) for call in calls) == {
        (family, condition): 2
        for family in ("cross_series", "duplicate_encoding", "trend")
        for condition in WORLD_RECOVERY_CONDITIONS
    }

    for call in calls:
        assert call.messages[0] == {"role": "system", "content": system_prompt}
        user = str(call.messages[1]["content"])
        assert user.index("redundant_facts:") < user.index("observed_values:")
        for forbidden in (
            "true_values",
            "error_index",
            "family_label",
            "valid_cue",
            "no_cue",
            "operation",
            "steps",
            "return",
        ):
            assert forbidden not in user

    duplicate_valid = next(
        call
        for call in calls
        if call.family == "duplicate_encoding" and call.condition == "valid_cue"
    )
    evidence = json.loads(duplicate_valid.facts_json)
    assert len(evidence) == 4
    assert {item["kind"] for item in evidence} == {"known_value"}


def test_parser_separates_exact_format_from_semantic_recovery() -> None:
    exact = parse_world_recovery_output("6,14,4,14")
    assert exact.exact_format_compliance is True
    assert exact.values == (6, 14, 4, 14)

    fenced = parse_world_recovery_output("```text\n[6, 14, 4, 14]\n```")
    assert fenced.exact_format_compliance is False
    assert fenced.values == (6, 14, 4, 14)

    prose = parse_world_recovery_output("Recovered world:\n6,14,4,14")
    assert prose.exact_format_compliance is False
    assert prose.values == (6, 14, 4, 14)

    ambiguous = parse_world_recovery_output("6,14,4,14\n6,14,4,15")
    assert ambiguous.values is None
    assert ambiguous.parse_failure is True

    inline = parse_world_recovery_output("The answer is 6,14,4,14.")
    assert inline.values is None
    assert inline.parse_failure is True


def test_scoring_uses_hidden_truth_for_both_conditions_and_condition_specific_facts() -> None:
    calls = build_phase_c_world_recovery_calls(
        _scenes(), config=_config(), system_prompt=PROMPT.read_text(encoding="utf-8")
    )
    no_cue = next(call for call in calls if call.condition == "no_cue")
    valid = next(
        call for call in calls if call.scene_id == no_cue.scene_id and call.condition == "valid_cue"
    )
    copied = evaluate_phase_c_world_recovery_call(
        no_cue, ",".join(str(value) for value in no_cue.observed_values)
    )
    assert copied.true_world_recovery is False
    assert copied.observation_copy is True
    assert copied.all_facts_satisfied is None
    assert copied.minimal_valid_repair is None

    corrected = evaluate_phase_c_world_recovery_call(
        valid, ",".join(str(value) for value in valid.true_values)
    )
    assert corrected.true_world_recovery is True
    assert corrected.observation_copy is False
    assert corrected.all_facts_satisfied is True
    assert corrected.edit_count == 1
    assert corrected.minimal_valid_repair is True
    assert corrected.correct_error_localization is True


def test_summary_uses_mutually_exclusive_pair_categories_and_keeps_duplicate_separate() -> None:
    calls = build_phase_c_world_recovery_calls(
        _scenes(), config=_config(), system_prompt=PROMPT.read_text(encoding="utf-8")
    )
    records = []
    for call in calls:
        if call.condition == "no_cue":
            values = call.observed_values
        elif call.family == "cross_series":
            values = call.true_values
        elif call.family == "duplicate_encoding":
            values = call.observed_values
        else:
            values = tuple(value + 1 for value in call.observed_values)
        records.append(
            evaluate_phase_c_world_recovery_call(call, ",".join(str(value) for value in values))
        )
    report = summarize_phase_c_world_recovery(tuple(records), config=_config())
    assert report["model_calls"] == 12
    assert report["pair_category_counts"] == {
        "cue_corrected": 2,
        "cue_ignored": 2,
        "cue_overedited": 2,
    }
    assert report["nontrivial_families"] == ["cross_series", "trend"]
    assert report["duplicate_encoding_role"] == (
        "full_trusted_state_restatement_instruction_following_control"
    )
    assert "pooled_primary_recovery_rate" not in report
    assert report["hypothesis_tested"] is False
    assert report["scale_authorized"] is False
    assert report["training_invoked"] is False


def test_case_validation_rejects_ambiguous_or_nonviolating_inputs() -> None:
    config = _config()
    system_prompt = PROMPT.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty tuple"):
        build_phase_c_world_recovery_calls((), config=config, system_prompt=system_prompt)
    with pytest.raises(ValueError, match="at least two eligible cases"):
        build_phase_c_world_recovery_calls(
            tuple(scene for scene in _scenes() if scene.scene_id not in {"trend-b", "trend-extra"}),
            config=config,
            system_prompt=system_prompt,
        )


def test_ambiguous_candidates_are_skipped_before_frozen_selection() -> None:
    ambiguous = FrozenEligibleScene(
        scene_id="trend-ambiguous",
        family="trend",
        chart_type="line",
        operation="difference",
        true_values=(2, 2, 3, 4),
        perceived_values=(2, 2, 2, 4),
    )
    calls = build_phase_c_world_recovery_calls(
        (ambiguous, *_scenes()),
        config=_config(),
        system_prompt=PROMPT.read_text(encoding="utf-8"),
    )
    assert len(calls) == 12
    assert "trend-ambiguous" not in {call.scene_id for call in calls}
