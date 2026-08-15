from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from compbias.recoverability.measurement_qualification import (
    MeasurementQualificationConfig,
    QualificationScene,
    build_qualification_records,
    load_measurement_qualification_config,
    load_reserved_numeric_tables,
    one_sided_binomial_lower,
    run_measurement_qualification,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recoverability" / "measurement_qualification_v1.yaml"


def _source_records(path: Path, *, count: int = 12) -> str:
    rows = []
    for index in range(count):
        rows.append(
            {
                "schema_version": 1,
                "dataset_id": "CVA-Chart-Pilot-v0.3",
                "sample_id": f"source-{index:06d}",
                "split": "pilot_train",
                "chart_type": "line" if index % 2 else "grouped_bar",
                "operation": ("difference", "sum", "max_minus_min")[index % 3],
                "values": [2 + index, 3 + index, 4 + index, 5 + index],
                "question": "frozen source question",
                "answer": 0,
                "image": f"images/source-{index:06d}.png",
                "mechanism": "iid",
            }
        )
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config() -> MeasurementQualificationConfig:
    return load_measurement_qualification_config(CONFIG)


def _scene(index: int) -> QualificationScene:
    operations = ("difference", "max_minus_min", "sum")
    return QualificationScene(
        scene_id=f"qualification-{index:06d}",
        image_path=Path(f"/dataset/images/qualification-{index:06d}.png"),
        chart_type="line" if index % 2 else "grouped_bar",
        operation=operations[index % len(operations)],
        values=(8, 4, 5, 9),
    )


def _stage1_raw(scene: QualificationScene, *, first: int | None = None) -> str:
    values = list(scene.values)
    if first is not None:
        values[0] = first
    return json.dumps(
        {
            "target_facts": values,
            "redundant_facts": [],
            "axis_facts": ["integer_ticks"],
        },
        separators=(",", ":"),
    )


def _stage2_raw(operation: str, values: tuple[int, int, int, int]) -> str:
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
    return json.dumps(
        {
            "variables": dict(zip(("a", "b", "c", "d"), values, strict=True)),
            "steps": steps,
            "return": "result",
        },
        separators=(",", ":"),
    )


def test_measurement_qualification_config_is_closed_and_not_a_hypothesis_test() -> None:
    config = _config()

    assert config.status == "PREREGISTERED_NOT_RUN"
    assert config.dataset_id == "CVA-Recoverability-Measurement-Qualification-v1"
    assert config.output_subdirectory == "measurement_qualification_v1"
    assert config.source_dataset_id == "CVA-Chart-Pilot-v0.3"
    assert config.source_dataset_records_sha256 == (
        "92ccdf54b11e2a6c12e12ef5273137824c6f3b94f38224abeb32d8319b83a62b"
    )
    assert config.scenes == 300
    assert config.per_stratum == 50
    assert config.format_retries == 0
    assert config.confidence == 0.95
    assert config.minimum_lower_bound == 0.98
    assert config.allow_rerun is False
    assert config.hypothesis_test is False
    assert config.confirmatory_execution_authorized is False


def test_measurement_qualification_config_rejects_any_contract_drift(tmp_path: Path) -> None:
    original = CONFIG.read_text(encoding="utf-8")
    changed = tmp_path / "changed.yaml"
    changed.write_text(original.replace("scenes: 300", "scenes: 301"), encoding="utf-8")

    with pytest.raises(ValueError, match="registered contract"):
        load_measurement_qualification_config(changed)


def test_reserved_tables_are_hash_bound_and_strictly_validated(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    digest = _source_records(source)

    tables = load_reserved_numeric_tables(source, expected_sha256=digest)
    assert len(tables) == 12
    assert (2, 3, 4, 5) in tables

    source.write_text(source.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_reserved_numeric_tables(source, expected_sha256=digest)


def test_qualification_records_are_balanced_deterministic_and_disjoint() -> None:
    reserved = frozenset((index, index + 1, index + 2, index + 3) for index in range(100))

    first = build_qualification_records(_config(), reserved_numeric_tables=reserved)
    second = build_qualification_records(
        _config(),
        reserved_numeric_tables=frozenset(reversed(tuple(reserved))),
    )

    assert first == second
    assert len(first) == 300
    assert len({row.sample_id for row in first}) == 300
    assert len({row.values for row in first}) == 300
    assert not {row.values for row in first}.intersection(reserved)
    assert Counter((row.chart_type, row.operation) for row in first) == {
        (chart, operation): 50
        for chart in ("grouped_bar", "line")
        for operation in ("difference", "max_minus_min", "sum")
    }
    assert all(row.split == "qualification" for row in first)
    assert all(row.dataset_id == _config().dataset_id for row in first)


def test_one_sided_binomial_lower_freezes_qualification_support_boundary() -> None:
    assert one_sided_binomial_lower(300, 300, confidence=0.95) == pytest.approx(0.9900639180555423)
    assert one_sided_binomial_lower(299, 300, confidence=0.95) == pytest.approx(0.9842854451084161)
    assert one_sided_binomial_lower(298, 300, confidence=0.95) == pytest.approx(0.9791637694729753)
    with pytest.raises(ValueError):
        one_sided_binomial_lower(True, 300, confidence=0.95)


def test_measurement_qualification_passes_only_the_interface_not_the_hypothesis() -> None:
    scenes = tuple(_scene(index) for index in range(300))
    stage1_calls: list[str] = []
    stage2_calls: list[str] = []

    def stage1(scene: QualificationScene, _messages: tuple[dict[str, object], ...]) -> str:
        stage1_calls.append(scene.scene_id)
        return _stage1_raw(scene)

    def stage2(
        scene: QualificationScene,
        perceived: tuple[int, int, int, int],
        _messages: tuple[dict[str, object], ...],
    ) -> str:
        stage2_calls.append(scene.scene_id)
        return _stage2_raw(scene.operation, perceived)

    report, records = run_measurement_qualification(
        scenes,
        config=_config(),
        stage1_generate=stage1,
        stage2_generate=stage2,
    )

    assert stage1_calls == [scene.scene_id for scene in scenes]
    assert stage2_calls == stage1_calls
    assert report.scenes == 300
    assert report.model_calls == 600
    assert report.stage1_parse_rate == 1.0
    assert report.stage2_program_parse_rate == 1.0
    assert report.stage2_execution_rate == 1.0
    assert report.executor_answer_accuracy == 1.0
    assert report.exact_transcription_rate == 1.0
    assert report.qualification_passed is True
    assert report.gate_failures == ()
    assert report.hypothesis_tested is False
    assert report.confirmatory_execution_authorized is False
    assert report.training_invoked is False
    assert report.format_retries == 0
    assert all(record.final_answer == record.executed_result for record in records)


def test_two_stage_qualification_counts_stage1_failures_without_retry_or_survivor_repair() -> None:
    scenes = tuple(_scene(index) for index in range(300))
    stage2_calls = 0

    def stage1(scene: QualificationScene, _messages: tuple[dict[str, object], ...]) -> str:
        if scene.scene_id in {"qualification-000000", "qualification-000001"}:
            return "not-json"
        return _stage1_raw(scene)

    def stage2(
        scene: QualificationScene,
        perceived: tuple[int, int, int, int],
        _messages: tuple[dict[str, object], ...],
    ) -> str:
        nonlocal stage2_calls
        stage2_calls += 1
        return _stage2_raw(scene.operation, perceived)

    report, records = run_measurement_qualification(
        scenes,
        config=_config(),
        stage1_generate=stage1,
        stage2_generate=stage2,
    )

    assert stage2_calls == 298
    assert report.model_calls == 598
    assert report.stage1_parse_successes == 298
    assert report.stage1_parse_lower < 0.98
    assert report.qualification_passed is False
    assert report.gate_failures == ("stage1_parse_lower_below_0_98",)
    assert sum(record.stage1_error_code is not None for record in records) == 2


def test_stage2_failures_use_the_full_preregistered_denominator_and_fail_closed() -> None:
    scenes = tuple(_scene(index) for index in range(300))

    def stage1(scene: QualificationScene, _messages: tuple[dict[str, object], ...]) -> str:
        return _stage1_raw(scene, first=7 if scene.scene_id == "qualification-000010" else None)

    def stage2(
        scene: QualificationScene,
        perceived: tuple[int, int, int, int],
        _messages: tuple[dict[str, object], ...],
    ) -> str:
        if scene.scene_id in {"qualification-000000", "qualification-000001"}:
            return '{"answer":4}'
        return _stage2_raw(scene.operation, perceived)

    report, records = run_measurement_qualification(
        scenes,
        config=_config(),
        stage1_generate=stage1,
        stage2_generate=stage2,
    )

    assert report.exact_transcription_rate == pytest.approx(299 / 300)
    assert report.stage2_program_parse_successes == 298
    assert report.stage2_program_parse_lower < 0.98
    assert report.qualification_passed is False
    assert report.gate_failures == (
        "stage2_program_parse_lower_below_0_98",
        "stage2_execution_lower_below_0_98",
        "executor_answer_lower_below_0_98",
    )
    assert sum(record.stage2_error_code == "program_parse_failure" for record in records) == 2


def test_config_dataclass_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _config().scenes = 301  # type: ignore[misc]
