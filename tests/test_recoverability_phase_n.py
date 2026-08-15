from __future__ import annotations

from collections import Counter
from pathlib import Path

from compbias.models.structured_parser import parse_trajectory
from compbias.recoverability import phase_n as phase_n_module
from compbias.recoverability.config import load_recoverability_protocol
from compbias.recoverability.phase_n import (
    PhaseNDatasetRecord,
    build_phase_n_observation,
    build_phase_n_records,
    run_phase_n,
    verify_phase_n_dataset,
    write_phase_n_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/recoverability/recoverability_v1.yaml"


def _raw(values: tuple[int, int, int, int], operation: str, answer: int) -> str:
    encoded = ",".join(str(value) for value in values)
    return (
        f'<perception>{{"values":[{encoded}]}}</perception>'
        f'<reasoning>{{"operation":"{operation}"}}</reasoning>'
        f"<answer>{answer}</answer>"
    )


def _record(
    *,
    sample_id: str = "natural-000000",
    operation: str = "sum",
    values: tuple[int, int, int, int] = (6, 6, 11, 5),
    answer: int = 12,
) -> PhaseNDatasetRecord:
    return PhaseNDatasetRecord(
        schema_version=1,
        dataset_id="CVA-Natural-Prevalence-v1",
        sample_id=sample_id,
        split="phase_n",
        chart_type="line",
        operation=operation,
        values=values,
        question="What is the sum of the first two values?",
        answer=answer,
        image=f"images/{sample_id}.png",
        mechanism="iid",
    )


def test_phase_n_plan_is_fixed_unique_balanced_and_disjoint() -> None:
    protocol = load_recoverability_protocol(PROTOCOL)
    reserved = {(2, 2, 2, 2), (3, 3, 3, 3)}

    first = build_phase_n_records(protocol.phase_n, reserved_numeric_tables=reserved)
    second = build_phase_n_records(protocol.phase_n, reserved_numeric_tables=reserved)

    assert first == second
    assert len(first) == 4000
    assert len({record.sample_id for record in first}) == 4000
    assert len({record.values for record in first}) == 4000
    assert not {record.values for record in first}.intersection(reserved)
    assert all(record.dataset_id == "CVA-Natural-Prevalence-v1" for record in first)
    assert all(record.split == "phase_n" for record in first)
    counts = Counter((record.chart_type, record.operation) for record in first)
    assert set(counts) == {
        (chart, operation)
        for chart in ("grouped_bar", "line")
        for operation in ("difference", "sum", "max_minus_min")
    }
    assert max(counts.values()) - min(counts.values()) <= 1


def test_phase_n_observation_separates_sensitive_repair_from_invariance() -> None:
    record = _record()

    strict = build_phase_n_observation(
        record,
        parse_trajectory(_raw((5, 6, 11, 5), "sum", 12), sample_id=record.sample_id),
    )
    invariant = build_phase_n_observation(
        record,
        parse_trajectory(_raw((5, 7, 11, 5), "sum", 12), sample_id=record.sample_id),
    )
    visual_error = build_phase_n_observation(
        record,
        parse_trajectory(_raw((5, 6, 11, 5), "sum", 11), sample_id=record.sample_id),
    )
    grounded_reasoning_error = build_phase_n_observation(
        record,
        parse_trajectory(_raw(record.values, "sum", 11), sample_id=record.sample_id),
    )
    parse_failure = build_phase_n_observation(
        record,
        parse_trajectory("not structured", sample_id=record.sample_id),
    )

    assert (strict.operator_sensitive_error, strict.strict_repair_candidate) == (True, True)
    assert (invariant.operator_sensitive_error, invariant.strict_repair_candidate) == (
        False,
        False,
    )
    assert (visual_error.operator_sensitive_error, visual_error.strict_repair_candidate) == (
        True,
        False,
    )
    assert (
        grounded_reasoning_error.operator_sensitive_error,
        grounded_reasoning_error.strict_repair_candidate,
    ) == (False, False)
    assert parse_failure.parse_success is False
    assert parse_failure.operator_sensitive_error is None
    assert parse_failure.strict_repair_candidate is None


def test_phase_n_runs_exactly_one_legacy_call_per_scene_without_extension() -> None:
    protocol = load_recoverability_protocol(PROTOCOL)
    records = tuple(
        _record(
            sample_id=f"natural-{index:06d}",
            operation="difference",
            values=(10, 4, 2, 3),
            answer=6,
        )
        for index in range(4000)
    )
    calls = 0

    def generate(record: PhaseNDatasetRecord, _messages: tuple[dict[str, object], ...]) -> str:
        nonlocal calls
        calls += 1
        if int(record.sample_id.rsplit("-", 1)[1]) < 800:
            return _raw((9, 4, 2, 3), "difference", 5)
        return _raw(record.values, "difference", record.answer)

    report, rows = run_phase_n(
        records,
        phase_config=protocol.phase_n,
        analysis_config=protocol.analysis,
        generate=generate,
    )

    assert calls == 4000
    assert len(rows) == 4000
    assert report.scenes == report.model_calls == 4000
    assert report.parsed_scenes == 4000
    assert report.operator_sensitive_errors == 800
    assert report.strict_natural_repair_candidates == 0
    assert report.h1_supported is True
    assert report.inconclusive is False
    assert report.format_retries == 0
    assert report.allow_sample_extension is False
    assert report.training_invoked is False


def test_phase_n_dataset_is_written_once_and_replayed_before_model_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol = load_recoverability_protocol(PROTOCOL)
    target = tmp_path / "phase-n"

    def fake_draw(path: Path, **_kwargs: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixed-image")

    monkeypatch.setattr(phase_n_module, "_draw_chart", fake_draw)
    manifest = write_phase_n_dataset(
        protocol.phase_n,
        reserved_numeric_tables={(2, 2, 2, 2)},
        output_dir=target,
    )
    replayed, scenes = verify_phase_n_dataset(
        protocol.phase_n,
        reserved_numeric_tables={(2, 2, 2, 2)},
        dataset_root=target,
    )

    assert replayed == manifest
    assert len(scenes) == 4000
    assert manifest["model_calls"] == 0
    assert manifest["training_invoked"] is False
