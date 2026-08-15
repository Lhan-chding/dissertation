from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from compbias.recoverability.phase_c_amendment import load_phase_c_amendment
from compbias.recoverability.phase_c_screen import (
    PhaseCScreenDatasetRecord,
    build_phase_c_screen_records,
    evaluate_phase_c_screen,
)
from compbias.recoverability.phase_n_result import load_phase_n_frozen_result


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "configs/recoverability/recoverability_phase_c_v2_amendment.yaml"
PHASE_N_RESULT = ROOT / "configs/recoverability/phase_n_frozen_result.yaml"


def test_phase_n_result_is_preserved_and_amendment_is_prospective() -> None:
    phase_n = load_phase_n_frozen_result(PHASE_N_RESULT)
    amendment = load_phase_c_amendment(AMENDMENT, phase_n=phase_n)

    assert phase_n.h1_supported is False
    assert phase_n.inconclusive is True
    assert phase_n.reason_code == "phase_n_h1_upper_not_below_threshold"
    assert phase_n.primary_rate == pytest.approx(33 / 836)
    assert phase_n.one_sided_cp_upper == 0.05242826275410656
    assert amendment.original_continuation_threshold == 0.05
    assert amendment.amended_continuation_threshold == 0.10
    assert amendment.original_phase_n_gate_passed is False
    assert amendment.phase_c_outcomes_observed is False
    assert amendment.confirmatory_phase_c_authorized is True
    assert amendment.training_authorized is False


def test_amendment_rejects_tampering_with_the_original_result(tmp_path: Path) -> None:
    text = AMENDMENT.read_text(encoding="utf-8")
    tampered = tmp_path / "amendment.yaml"
    tampered.write_text(
        text.replace("original_phase_n_gate_passed: false", "original_phase_n_gate_passed: true"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical|original"):
        load_phase_c_amendment(tampered, phase_n=load_phase_n_frozen_result(PHASE_N_RESULT))


def test_phase_c_screen_plan_is_fixed_balanced_unique_and_disjoint() -> None:
    amendment = load_phase_c_amendment(
        AMENDMENT,
        phase_n=load_phase_n_frozen_result(PHASE_N_RESULT),
    )
    reserved = {(2, 3, 4, 5), (10, 11, 12, 13)}
    first = build_phase_c_screen_records(amendment, reserved_numeric_tables=reserved)
    second = build_phase_c_screen_records(amendment, reserved_numeric_tables=reserved)

    assert first == second
    assert len(first) == 8000
    assert len({row.scene_id for row in first}) == 8000
    assert len({row.values for row in first}) == 8000
    assert not ({row.values for row in first} & reserved)
    counts = Counter((row.family, row.chart_type, row.operation) for row in first)
    assert len(counts) == 18
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(row.split == "phase_c_screen" for row in first)
    assert all(row.format_retries == 0 for row in first)


def _record(scene_id: str, family: str, values: tuple[int, int, int, int]) -> PhaseCScreenDatasetRecord:
    return PhaseCScreenDatasetRecord(
        schema_version=1,
        dataset_id="CVA-Recoverability-Causal-v2",
        scene_id=scene_id,
        split="phase_c_screen",
        family=family,
        chart_type="line",
        operation="sum",
        values=values,
        question="What is the sum of the first two values?",
        answer=values[0] + values[1],
        image=f"images/{scene_id}.png",
        format_retries=0,
    )


def test_screen_eligibility_is_strict_and_quota_selection_is_fail_closed() -> None:
    amendment = load_phase_c_amendment(
        AMENDMENT,
        phase_n=load_phase_n_frozen_result(PHASE_N_RESULT),
    )
    records = (
        _record("cross-ok", "cross_series", (8, 4, 5, 9)),
        _record("trend-ok", "trend", (4, 6, 8, 10)),
        _record("duplicate-ok", "duplicate_encoding", (7, 4, 5, 9)),
        _record("parse-fail", "cross_series", (9, 4, 5, 8)),
        _record("operator-null", "trend", (4, 6, 8, 11)),
    )
    raw = {
        "cross-ok": '{"target_facts":[7,4,5,9],"redundant_facts":[],"axis_facts":["integer_ticks"]}',
        "trend-ok": '{"target_facts":[3,6,8,10],"redundant_facts":[],"axis_facts":["integer_ticks"]}',
        "duplicate-ok": '{"target_facts":[6,4,5,9],"redundant_facts":[],"axis_facts":["integer_ticks"]}',
        "parse-fail": "not-json",
        # C/D differ but SUM(A,B) is unchanged: operator-invariant, hence ineligible.
        "operator-null": '{"target_facts":[4,6,7,11],"redundant_facts":[],"axis_facts":["integer_ticks"]}',
    }

    report, rows = evaluate_phase_c_screen(
        records,
        amendment=amendment,
        generate=lambda record, _messages: raw[record.scene_id],
        quotas={"cross_series": 1, "trend": 1, "duplicate_encoding": 1},
    )

    assert report.screen_passed is True
    assert report.confirmatory_arm_execution_authorized is True
    assert report.training_authorized is False
    assert report.training_invoked is False
    assert set(report.selected_scene_ids) == {"cross-ok", "trend-ok", "duplicate-ok"}
    by_id = {row.scene_id: row for row in rows}
    assert by_id["parse-fail"].eligible is False
    assert by_id["operator-null"].operator_sensitive is False

    failed, _ = evaluate_phase_c_screen(
        records,
        amendment=amendment,
        generate=lambda record, _messages: raw[record.scene_id],
        quotas={"cross_series": 2, "trend": 1, "duplicate_encoding": 1},
    )
    assert failed.screen_passed is False
    assert failed.confirmatory_arm_execution_authorized is False
    assert failed.failure_codes == ("phase_c_screen_family_quota_unmet",)


def test_screen_never_authorizes_training_even_when_all_quotas_are_met() -> None:
    amendment = load_phase_c_amendment(
        AMENDMENT,
        phase_n=load_phase_n_frozen_result(PHASE_N_RESULT),
    )
    records = (_record("cross-only", "cross_series", (8, 4, 5, 9)),)
    report, _ = evaluate_phase_c_screen(
        records,
        amendment=amendment,
        generate=lambda *_: '{"target_facts":[7,4,5,9],"redundant_facts":[],"axis_facts":["integer_ticks"]}',
        quotas={"cross_series": 1},
    )
    assert report.screen_passed is True
    assert report.confirmatory_arm_execution_authorized is True
    assert report.training_authorized is False
    assert report.rl_authorized is False
