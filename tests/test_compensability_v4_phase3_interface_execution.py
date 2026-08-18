"""RED contracts for real S6 I0--I4 interface-ladder execution."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from compensability_v4.diagnostics.interface_ladder import CueCondition, Interface

INTERFACES = tuple(Interface)
CONDITIONS = tuple(CueCondition)
SOURCE_SHA256 = {
    "S3_candidate": "b" * 64,
    "S5_cache": "d" * 64,
    "S6_runtime": "e" * 64,
}
TRUTH = (6, 14, 10, 10)
OBSERVED = (5, 14, 10, 10)
COUNTERFACTUAL = (5, 15, 10, 10)


def _subject():
    return importlib.import_module("compensability_v4.qwen.phase3_interface")


def _source_stage(interface: Interface) -> str:
    return {
        Interface.I0_HARD_TEXT: "S6_runtime",
        Interface.I1_SOFT_REPORT: "S6_runtime",
        Interface.I2_CANDIDATE_WORLD: "S3_candidate",
        Interface.I3_SAME_CONVERSATION: "S5_cache",
        Interface.I4_EXACT_CACHE: "S5_cache",
    }[interface]


def _source_branch(interface: Interface) -> str:
    return {
        Interface.I0_HARD_TEXT: "fresh_text_runtime",
        Interface.I1_SOFT_REPORT: "stage1_soft_report_runtime",
        Interface.I2_CANDIDATE_WORLD: "teacher_forced_candidate",
        Interface.I3_SAME_CONVERSATION: "full_history",
        Interface.I4_EXACT_CACHE: "cached_continuation",
    }[interface]


def _soft_report_payload() -> dict[str, object]:
    return {
        "top_k": 2,
        "positions": [
            {
                "index": index,
                "candidates": [
                    {"value": OBSERVED[index], "relative_logit": 0.0},
                    {
                        "value": (
                            TRUTH[index] if TRUTH[index] != OBSERVED[index] else OBSERVED[index] + 1
                        ),
                        "relative_logit": -0.25,
                    },
                ],
            }
            for index in range(4)
        ],
    }


def _records(
    subject,
    *,
    scenes: int,
    i4_diagnostic_scene_ids: frozenset[str] = frozenset(),
):
    rows = []
    for scene_index in range(scenes):
        scene_id = f"scene-{scene_index:03d}"
        if scenes == 579:
            family = (
                "cross_series"
                if scene_index < 208
                else "duplicate_encoding"
                if scene_index < 390
                else "trend"
            )
        else:
            family = ("cross_series", "duplicate_encoding", "trend")[scene_index % 3]
        output = TRUTH if scene_index % 2 == 0 else OBSERVED
        for condition in CONDITIONS:
            for interface in INTERFACES:
                if interface is Interface.I1_SOFT_REPORT and condition is not CueCondition.NO_CUE:
                    continue
                i4_token_diagnostic = (
                    interface is Interface.I4_EXACT_CACHE
                    and condition is CueCondition.NO_CUE
                    and scene_id in i4_diagnostic_scene_ids
                )
                interface_diagnostic = interface in {
                    Interface.I1_SOFT_REPORT,
                    Interface.I2_CANDIDATE_WORLD,
                }
                diagnostic_only = interface_diagnostic or i4_token_diagnostic
                rows.append(
                    subject.InterfaceLadderRecord(
                        call_id=f"{scene_id}.{condition.value}.{interface.value}",
                        scene_id=scene_id,
                        family=family,
                        interface=interface,
                        condition=condition,
                        true_world=TRUTH,
                        observed_world=OBSERVED,
                        counterfactual_world=COUNTERFACTUAL,
                        output_world=(
                            None
                            if interface is Interface.I1_SOFT_REPORT
                            else COUNTERFACTUAL
                            if condition is CueCondition.COUNTERFACTUAL_CUE
                            else output
                        ),
                        parse_success=(None if interface is Interface.I1_SOFT_REPORT else True),
                        diagnostic_payload=(
                            _soft_report_payload()
                            if interface is Interface.I1_SOFT_REPORT
                            else None
                        ),
                        source_stage=_source_stage(interface),
                        source_branch=_source_branch(interface),
                        source_call_id=(
                            f"{scene_id}.{condition.value}.{_source_branch(interface)}"
                        ),
                        source_artifact_sha256=SOURCE_SHA256[_source_stage(interface)],
                        structural_validity_verified=True,
                        primary_eligible=not diagnostic_only,
                        diagnostic_only=diagnostic_only,
                        diagnostic_reason=(
                            "token_divergence"
                            if i4_token_diagnostic
                            else "interface_diagnostic"
                            if interface_diagnostic
                            else None
                        ),
                    )
                )
    return tuple(rows)


def test_s6_requires_579_scenes_four_cues_five_interfaces_and_separates_i4_diagnostics():
    subject = _subject()
    diagnostic_scenes = frozenset(f"scene-{index:03d}" for index in range(33))
    rows = _records(
        subject,
        scenes=579,
        i4_diagnostic_scene_ids=diagnostic_scenes,
    )

    frozen = subject.validate_interface_ladder_records(
        rows,
        expected_scenes=579,
        expected_conditions=4,
        expected_interfaces=5,
        expected_source_sha256=SOURCE_SHA256,
    )

    assert len(frozen) == 579 * 17
    assert len({row.call_id for row in frozen}) == len(frozen)
    assert len({row.scene_id for row in frozen}) == 579
    assert Counter(row.interface for row in frozen) == Counter(
        {
            Interface.I0_HARD_TEXT: 579 * 4,
            Interface.I1_SOFT_REPORT: 579,
            Interface.I2_CANDIDATE_WORLD: 579 * 4,
            Interface.I3_SAME_CONVERSATION: 579 * 4,
            Interface.I4_EXACT_CACHE: 579 * 4,
        }
    )
    assert Counter(row.condition for row in frozen) == Counter(
        {
            CueCondition.NO_CUE: 579 * 5,
            CueCondition.VALID_CUE: 579 * 4,
            CueCondition.SHAM_CUE: 579 * 4,
            CueCondition.COUNTERFACTUAL_CUE: 579 * 4,
        }
    )
    assert Counter(row.family for row in frozen) == Counter(
        {"cross_series": 208 * 17, "duplicate_encoding": 182 * 17, "trend": 189 * 17}
    )
    by_scene = {
        scene_id: tuple(row for row in frozen if row.scene_id == scene_id)
        for scene_id in {row.scene_id for row in frozen}
    }
    assert all(len(rows) == 17 for rows in by_scene.values())
    assert all({row.interface for row in rows} == set(INTERFACES) for rows in by_scene.values())
    assert all({row.condition for row in rows} == set(CONDITIONS) for rows in by_scene.values())
    i4 = tuple(row for row in frozen if row.interface is Interface.I4_EXACT_CACHE)
    assert sum(row.primary_eligible for row in i4) == 579 * 4 - 33
    assert sum(row.diagnostic_reason == "token_divergence" for row in i4) == 33
    assert all(
        not row.primary_eligible and row.diagnostic_only
        for row in i4
        if row.diagnostic_reason == "token_divergence"
    )
    with pytest.raises(FrozenInstanceError):
        frozen[0].primary_eligible = False
    by_interface = {row.interface: row for row in frozen if row.scene_id == "scene-000"}
    assert by_interface[Interface.I0_HARD_TEXT].source_stage == "S6_runtime"
    i1 = by_interface[Interface.I1_SOFT_REPORT]
    assert i1.source_stage == "S6_runtime"
    assert i1.source_branch == "stage1_soft_report_runtime"
    assert i1.output_world is None
    assert i1.parse_success is None
    assert i1.diagnostic_payload["top_k"] == 2
    assert isinstance(i1.diagnostic_payload["positions"], tuple)
    assert len(i1.diagnostic_payload["positions"]) == 4
    with pytest.raises(TypeError):
        i1.diagnostic_payload["top_k"] = 3
    with pytest.raises(TypeError):
        i1.diagnostic_payload["positions"][0]["index"] = 7
    assert by_interface[Interface.I2_CANDIDATE_WORLD].source_stage == "S3_candidate"
    assert by_interface[Interface.I3_SAME_CONVERSATION].source_stage == "S5_cache"
    assert by_interface[Interface.I3_SAME_CONVERSATION].source_branch == "full_history"
    assert by_interface[Interface.I4_EXACT_CACHE].source_stage == "S5_cache"
    assert by_interface[Interface.I4_EXACT_CACHE].source_branch == "cached_continuation"
    assert all(
        row.output_world is not None and row.parse_success is True
        for interface, row in by_interface.items()
        if interface is not Interface.I1_SOFT_REPORT
    )


def test_s6_summary_uses_only_complete_scene_pairs_and_reports_objective_strata():
    subject = _subject()
    rows = _records(
        subject,
        scenes=2,
        i4_diagnostic_scene_ids=frozenset({"scene-001"}),
    )
    frozen = subject.validate_interface_ladder_records(
        rows,
        expected_scenes=2,
        expected_conditions=4,
        expected_interfaces=5,
        expected_source_sha256=SOURCE_SHA256,
    )

    summary = subject.summarize_interface_ladder(frozen, bootstrap_resamples=200, seed=17)

    assert summary["schema_version"] == 1
    assert summary["status"] == "PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS"
    assert summary["number_of_source_scenes"] == 2
    assert summary["number_of_cells"] == 34
    assert summary["primary_paired_scene_count"] == 1
    assert summary["excluded_primary_scene_ids"] == ["scene-001"]
    assert summary["i4_exact_eligible_call_count"] == 7
    assert summary["i4_token_diagnostic_call_count"] == 1
    assert summary["intervention_diagnostic_cell_count"] == 10
    assert summary["primary_analysis_cell_count"] == 12
    assert summary["diagnostic_call_ids"] == [
        "scene-001.no_cue.I4_exact_cached_natural_continuation"
    ]
    assert set(summary["by_interface"]) == {interface.value for interface in INTERFACES}
    assert set(summary["by_cue_condition"]) == {condition.value for condition in CONDITIONS}
    assert set(summary["by_family"]) == {"cross_series", "duplicate_encoding"}
    i0_no_cue = summary["by_interface"][Interface.I0_HARD_TEXT.value]["by_cue_condition"][
        CueCondition.NO_CUE.value
    ]["exact_world_recovery"]
    assert i0_no_cue == {
        "estimate": 1.0,
        "ci_low": 1.0,
        "ci_high": 1.0,
        "confidence": 0.95,
        "number_of_scenes": 1,
    }
    assert summary["effects"]["spontaneous_visual_revision"]["number_of_scenes"] == 1
    assert summary["effects"]["fact_conditioned_revision"]["number_of_scenes"] == 1
    assert summary["scene_is_statistical_unit"] is True
    assert summary["subjective_success_threshold_applied"] is False
    assert "minimum_recovery_accuracy" not in summary
    assert "maximum_diagnostic_rate" not in summary


def test_s6_validation_fails_closed_on_missing_cells_hash_or_structural_drift():
    subject = _subject()
    rows = _records(subject, scenes=1)

    with pytest.raises(RuntimeError, match="duplicate"):
        subject.validate_interface_ladder_records(
            (*rows, rows[0]),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    with pytest.raises(RuntimeError, match=r"complete|missing"):
        subject.validate_interface_ladder_records(
            rows[:-1],
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    i1 = next(row for row in rows if row.interface is Interface.I1_SOFT_REPORT)
    without_i1 = tuple(row for row in rows if row.call_id != i1.call_id)
    with pytest.raises(RuntimeError, match=r"I1|soft.report|no.cue|complete|missing"):
        subject.validate_interface_ladder_records(
            without_i1,
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    with pytest.raises(RuntimeError, match=r"I1|soft.report|no.cue|cell"):
        subject.validate_interface_ladder_records(
            (
                *rows,
                replace(
                    i1,
                    call_id=i1.call_id.replace("no_cue", "valid_cue"),
                    condition=CueCondition.VALID_CUE,
                ),
            ),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    with pytest.raises(RuntimeError, match=r"SHA|hash"):
        subject.validate_interface_ladder_records(
            (replace(rows[0], source_artifact_sha256="drift"), *rows[1:]),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    with pytest.raises(RuntimeError, match="structural"):
        subject.validate_interface_ladder_records(
            (replace(rows[0], structural_validity_verified=False), *rows[1:]),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    i4 = next(row for row in rows if row.interface is Interface.I4_EXACT_CACHE)
    with pytest.raises(RuntimeError, match=r"I4|eligible|diagnostic"):
        subject.validate_interface_ladder_records(
            (
                replace(
                    i4,
                    diagnostic_only=True,
                    diagnostic_reason="token_divergence",
                    primary_eligible=True,
                ),
                *(row for row in rows if row.call_id != i4.call_id),
            ),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    i0 = next(row for row in rows if row.interface is Interface.I0_HARD_TEXT)
    with pytest.raises(RuntimeError, match=r"source|runtime"):
        subject.validate_interface_ladder_records(
            (replace(i0, source_stage="S1_capability"), *(row for row in rows if row != i0)),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    with pytest.raises(RuntimeError, match=r"diagnostic|top-k|payload"):
        subject.validate_interface_ladder_records(
            (
                replace(i1, diagnostic_payload=None),
                *(row for row in rows if row != i1),
            ),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )
    i3 = next(row for row in rows if row.interface is Interface.I3_SAME_CONVERSATION)
    with pytest.raises(RuntimeError, match=r"branch|full.history"):
        subject.validate_interface_ladder_records(
            (
                replace(i3, source_branch="cached_continuation"),
                *(row for row in rows if row != i3),
            ),
            expected_scenes=1,
            expected_conditions=4,
            expected_interfaces=5,
            expected_source_sha256=SOURCE_SHA256,
        )


def test_s6_writes_per_scene_and_summary_without_overwrite(tmp_path: Path):
    subject = _subject()
    rows = subject.validate_interface_ladder_records(
        _records(subject, scenes=1),
        expected_scenes=1,
        expected_conditions=4,
        expected_interfaces=5,
        expected_source_sha256=SOURCE_SHA256,
    )
    summary = subject.summarize_interface_ladder(rows, bootstrap_resamples=50, seed=3)
    per_scene = tmp_path / "interface_ladder/per_scene.jsonl"
    summary_path = tmp_path / "interface_ladder/summary.json"

    subject.write_interface_ladder_outputs(
        per_scene,
        summary_path,
        records=rows,
        summary=summary,
    )

    payloads = [json.loads(line) for line in per_scene.read_text().splitlines()]
    assert len(payloads) == 17
    persisted_i1 = next(
        row
        for row in payloads
        if row["interface"] == Interface.I1_SOFT_REPORT.value
        and row["cue_condition"] == CueCondition.NO_CUE.value
    )
    assert persisted_i1["output_world"] is None
    assert persisted_i1["diagnostic_payload"]["top_k"] == 2
    assert len(persisted_i1["diagnostic_payload"]["positions"]) == 4
    assert json.loads(summary_path.read_text())["number_of_source_scenes"] == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.write_interface_ladder_outputs(
            per_scene,
            summary_path,
            records=rows,
            summary=summary,
        )


def _load_script(path: Path):
    script_directory = str(path.parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    spec = importlib.util.spec_from_file_location("phase3_interface_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_06_entrypoint_delegates_to_real_hash_bound_execution(monkeypatch):
    module = _load_script(
        Path(__file__).resolve().parents[1] / "scripts/v4/06_run_interface_ladder.py"
    )
    captured = {}
    monkeypatch.setattr(
        module,
        "run_phase_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("S6 must execute the real interface ladder, not preflight")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "run_interface_ladder_cli",
        lambda **kwargs: captured.update(kwargs) or 17,
        raising=False,
    )

    assert module.main() == 17
    assert captured["phase"] == "phase_3_interface_ladder"
    assert captured["expected_scenes"] == 579
    assert captured["expected_conditions"] == 4
    assert captured["expected_interfaces"] == 5
    assert captured["expected_cells_per_scene"] == 17
    assert captured["required_sources"] == (
        "screen",
        "capability_per_scene",
        "capability_summary",
        "capability_gaps",
        "candidate_labels",
        "candidate_scores",
        "candidate_summary",
        "layerwise_per_scene",
        "layerwise_summary",
        "cache_parity",
    )
    assert captured["measurement_sources"] == {
        "I0": "S6_runtime",
        "I1": "S6_runtime",
        "I2": "S3_candidate",
        "I3": "S5_cache.full_history",
        "I4": "S5_cache.cached_continuation",
    }
    assert captured["provenance_only_sources"] == (
        "capability_per_scene",
        "capability_summary",
        "capability_gaps",
        "layerwise_per_scene",
        "layerwise_summary",
    )
    assert captured["input_roles"] == {
        "screen": "runtime_scene_source",
        "capability_per_scene": "provenance_only",
        "capability_summary": "provenance_only",
        "capability_gaps": "provenance_only",
        "candidate_labels": "I2_provenance",
        "candidate_scores": "I2_measurement",
        "candidate_summary": "I2_provenance",
        "layerwise_per_scene": "provenance_only",
        "layerwise_summary": "provenance_only",
        "cache_parity": "I3_I4_measurement",
    }
    assert captured["output_paths"] == {
        "per_scene": "artifacts/v4/interface_ladder/per_scene.jsonl",
        "summary": "artifacts/v4/interface_ladder/summary.json",
    }
