from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compbias.recoverability.dsl.executor import TrustedBinding
from compbias.recoverability.dsl.result_program import (
    ResultProgramParseError,
    evaluate_result_program,
    execute_result_program,
    parse_result_program,
)
from compbias.recoverability.stage2_v2 import (
    Stage2V1DiagnosticAnchor,
    Stage2V2Scene,
    build_stage2_v2_messages,
    load_stage2_v1_diagnostic_anchor,
    load_stage2_v2_probe_config,
    load_stage2_v2_scenes_from_stage1_records,
    run_stage2_v2_probe,
    verify_stage2_v1_diagnostic,
    verify_stage2_v2_server_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = ROOT / "configs" / "recoverability" / "stage2_v2_probe.yaml"
DIAGNOSTIC_ANCHOR = ROOT / "configs" / "recoverability" / "stage2_v1_diagnostic_result.yaml"
SERVER_LOCK = ROOT / "configs" / "recoverability" / "server_package_lock_stage2_v2.yaml"


def _raw(
    *,
    variables: dict[str, int] | None = None,
    steps: list[dict[str, object]] | None = None,
    returned: str = "result",
) -> str:
    return json.dumps(
        {
            "variables": variables or {"a": 8, "b": 4, "c": 5, "d": 9},
            "steps": steps or [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}],
            "return": returned,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _scene(index: int, operation: str = "difference") -> Stage2V2Scene:
    return Stage2V2Scene(
        scene_id=f"dev-{index:06d}",
        operation=operation,
        evidence=(8, 4, 5, 9),
    )


def _registered_raw(scene: Stage2V2Scene) -> str:
    if scene.operation == "sum":
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
    elif scene.operation == "difference":
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
    else:
        steps = [
            {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
            {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
    return _raw(
        variables=dict(zip(("a", "b", "c", "d"), scene.evidence, strict=True)),
        steps=steps,
    )


def test_result_program_executor_is_the_only_numeric_answer_source() -> None:
    raw = _raw()
    program = parse_result_program(raw)

    result = execute_result_program(
        program,
        constraint_bindings={
            "a": TrustedBinding("stage1_target_a", 8),
            "b": TrustedBinding("stage1_target_b", 4),
        },
    )

    assert "answer" not in inspect.signature(type(program)).parameters
    assert result.executed_result == 4
    assert result.final_answer == 4
    assert result.return_variable == "result"
    assert result.consumed_constraint_ids == ("stage1_target_a", "stage1_target_b")


@pytest.mark.parametrize(
    "raw",
    [
        '{"variables":{"a":8},"steps":[],"return":"a","answer":8}',
        '{"variables":{"a":8},"steps":[],"return":"a"}',
        _raw(returned="missing"),
        _raw(returned="a"),
        _raw().replace('"return":"result"', '"return":1'),
        _raw().replace('"a":8', '"a":true'),
        _raw() + " trailing",
    ],
)
def test_result_program_parser_rejects_answer_keys_and_noncanonical_returns(raw: str) -> None:
    with pytest.raises(ResultProgramParseError):
        parse_result_program(raw)


def test_result_program_evaluation_fails_closed_without_repair() -> None:
    malformed = evaluate_result_program('{"answer":4}', constraint_bindings={})
    assert malformed.program_parse_success is False
    assert malformed.program_execution_success is False
    assert malformed.final_answer is None
    assert malformed.error_code == "program_parse_failure"

    untrusted = evaluate_result_program(
        _raw(),
        constraint_bindings={"a": TrustedBinding("stage1_target_a", 99)},
    )
    assert untrusted.program_parse_success is True
    assert untrusted.program_execution_success is False
    assert untrusted.final_answer is None
    assert untrusted.error_code == "program_execution_failure"


@pytest.mark.parametrize("operation", ["difference", "max_minus_min", "sum"])
def test_stage2_v2_prompt_is_exact_gold_free_and_has_no_numeric_answer_slot(
    operation: str,
) -> None:
    scene = Stage2V2Scene(
        scene_id="dev-000001",
        operation=operation,
        evidence=(8, 4, 5, 9),
    )

    messages = build_stage2_v2_messages(scene)
    serialized = json.dumps(messages, sort_keys=True)
    system = str(messages[0]["content"])

    assert tuple(inspect.signature(build_stage2_v2_messages).parameters) == ("scene",)
    assert len(messages) == 2
    assert '"return":"result"' in system
    assert '"answer"' not in system
    assert "gold" not in serialized.lower()
    assert "image" not in serialized.lower()
    assert "question" not in serialized.lower()
    assert all(str(value) in serialized for value in scene.evidence)


def test_stage2_v2_probe_uses_executor_answer_and_never_retries() -> None:
    operations = ("difference", "max_minus_min", "sum")
    scenes = tuple(_scene(index, operations[index % 3]) for index in range(24))
    calls: list[str] = []

    def generate(scene: Stage2V2Scene, messages: tuple[dict[str, object], ...]) -> str:
        calls.append(scene.scene_id)
        assert messages == build_stage2_v2_messages(scene)
        return _registered_raw(scene)

    report, records = run_stage2_v2_probe(scenes, generate=generate)

    assert calls == [scene.scene_id for scene in scenes]
    assert report.scenes == report.model_calls == 24
    assert report.program_parse_rate == 1.0
    assert report.execution_rate == 1.0
    assert report.executor_answer_accuracy == 1.0
    assert report.format_retries == 0
    assert report.training_invoked is False
    assert report.probe_passed is True
    assert "program_answer_consistency" not in inspect.signature(type(report)).parameters
    assert all(record.final_answer == record.executed_result for record in records)


def test_stage2_v2_probe_records_strict_failures_once() -> None:
    scenes = tuple(_scene(index) for index in range(24))
    calls = 0

    def generate(
        _scene_value: Stage2V2Scene,
        _messages: tuple[dict[str, object], ...],
    ) -> str:
        nonlocal calls
        calls += 1
        return '{"answer":4}'

    report, records = run_stage2_v2_probe(scenes, generate=generate)

    assert calls == 24
    assert report.program_parse_rate == 0.0
    assert report.execution_rate == 0.0
    assert report.executor_answer_accuracy == 0.0
    assert dict(report.error_counts) == {"program_parse_failure": 24}
    assert all(record.error_code == "program_parse_failure" for record in records)


def test_stage2_v2_configs_freeze_the_failed_v1_dependency_and_one_shot_boundary() -> None:
    probe = load_stage2_v2_probe_config(PROBE_CONFIG)
    anchor = load_stage2_v1_diagnostic_anchor(DIAGNOSTIC_ANCHOR)

    assert probe.status == "DEVELOPMENT_PROBE_NOT_RUN"
    assert probe.dataset_id == "CVA-Recoverability-Stage2-V2-Dev-Probe"
    assert probe.scenes == 24
    assert probe.format_retries == 0
    assert probe.required_program_parse_rate == 1.0
    assert probe.required_execution_rate == 1.0
    assert probe.required_executor_answer_accuracy == 1.0
    assert probe.allow_rerun is False
    assert probe.hypothesis_test is False
    assert anchor.status == "FINAL_VERIFIED_STAGE2_V1_FAILURE_DO_NOT_RERUN"
    assert anchor.diagnostic_sha256 == (
        "d85510ea829a000bc31002f874e5a0ec795421aadec9f9042438d78337d9e7b4"
    )
    assert anchor.source_stage2_records_sha256 == (
        "b8ce766b9ba0555fe780e88195a0e4d4d3294cc8813357e7a33a5cfeea19793c"
    )


def test_stage2_v1_diagnostic_anchor_rejects_tampering(tmp_path: Path) -> None:
    payload = {
        "verified": True,
        "records": 24,
        "replayed_program_parse_successes": 19,
        "replayed_program_execution_successes": 19,
        "replayed_program_answer_matches": 13,
        "replayed_operation_result_correct": 13,
        "model_calls": 0,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
        "training_invoked": False,
        "source_stage2_records_sha256": "b" * 64,
    }
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    anchor = Stage2V1DiagnosticAnchor(
        schema_version=1,
        status="FINAL_VERIFIED_STAGE2_V1_FAILURE_DO_NOT_RERUN",
        diagnostic_sha256=digest,
        source_stage2_records_sha256="b" * 64,
        records=24,
        program_parse_successes=19,
        program_execution_successes=19,
        executor_answer_matches=13,
        verified=True,
        model_calls=0,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
    )

    assert verify_stage2_v1_diagnostic(anchor, path).verified is True

    path.write_text(json.dumps({**payload, "records": 23}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_stage2_v1_diagnostic(anchor, path)

    restored = json.dumps({**payload, "records": 23})
    path.write_text(restored, encoding="utf-8")
    changed = replace(anchor, diagnostic_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="semantics"):
        verify_stage2_v1_diagnostic(changed, path)


def test_stage2_v2_stage1_loader_is_hash_bound_and_strict(tmp_path: Path) -> None:
    records = tmp_path / "probe_records.jsonl"
    rows = []
    for index in range(24):
        rows.append(
            {
                "scene_id": f"dev-{index:06d}",
                "chart_type": "line",
                "operation": ("difference", "max_minus_min", "sum")[index % 3],
                "raw_text": (
                    '{"target_facts":[8,4,5,9],"redundant_facts":[],"axis_facts":["integer_ticks"]}'
                ),
                "parse_success": True,
                "exact_transcription": True,
                "error_code": None,
            }
        )
    records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    digest = hashlib.sha256(records.read_bytes()).hexdigest()

    scenes = load_stage2_v2_scenes_from_stage1_records(records, expected_sha256=digest)

    assert tuple(scene.scene_id for scene in scenes) == tuple(
        f"dev-{index:06d}" for index in range(24)
    )
    assert all(scene.evidence == (8, 4, 5, 9) for scene in scenes)

    records.write_text(records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_stage2_v2_scenes_from_stage1_records(records, expected_sha256=digest)


def test_stage2_v2_server_package_lock_matches_the_exact_current_closure() -> None:
    result = verify_stage2_v2_server_package_lock(SERVER_LOCK, repository_root=ROOT)

    assert result.verified is True
    assert {item.relative_path for item in result.files} >= {
        "configs/recoverability/stage2_v1_diagnostic_result.yaml",
        "configs/recoverability/stage2_v2_probe.yaml",
        "experiments/recoverability_v1/00_stage2_v2_preflight.py",
        "experiments/recoverability_v1/07_stage2_v2_probe.py",
        "src/compbias/recoverability/dsl/result_program.py",
        "src/compbias/recoverability/stage2_v2.py",
    }


def test_stage2_v2_server_entrypoints_are_one_shot_and_package_locked() -> None:
    preflight = ROOT / "experiments" / "recoverability_v1" / "00_stage2_v2_preflight.py"
    probe = ROOT / "experiments" / "recoverability_v1" / "07_stage2_v2_probe.py"
    lock = ROOT / "configs" / "recoverability" / "server_package_lock_stage2_v2.yaml"

    preflight_source = preflight.read_text(encoding="utf-8")
    probe_source = probe.read_text(encoding="utf-8")
    lock_text = lock.read_text(encoding="utf-8")

    assert "stage2_v1_diagnostic_result.yaml" in lock_text
    assert "stage2_v1_failure.py" in lock_text
    assert "result_program.py" in lock_text
    assert "stage2_v2.py" in lock_text
    assert "07_stage2_v2_probe.py" in lock_text
    assert "_BOOTSTRAP_SERVER_PATHS" in preflight_source
    assert "_BOOTSTRAP_SERVER_PATHS" in probe_source
    assert "frozenset(paths) != _BOOTSTRAP_SERVER_PATHS" in preflight_source
    assert "frozenset(paths) != _BOOTSTRAP_SERVER_PATHS" in probe_source
    assert "--stage2-v1-diagnostic" in probe_source
    assert "format_retries" in probe_source
    assert "attempt_marker" in probe_source
    assert 'open("x"' in probe_source
    assert "load_local_qwen" in probe_source
    assert "load_local_qwen" not in preflight_source
    assert "run_stage2_v1_probe" not in probe_source
    assert "confirmatory_execution_authorized" in probe_source
    assert "hypothesis_tested" in probe_source
