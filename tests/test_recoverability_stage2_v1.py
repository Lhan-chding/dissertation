from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from compbias.recoverability.stage1_v2 import Stage1V2Scene
from compbias.recoverability.stage2_v1 import (
    Stage1V2FrozenResult,
    Stage2V1Scene,
    build_stage2_v1_messages,
    load_stage1_v2_frozen_result,
    load_stage2_v1_probe_config,
    run_stage2_v1_probe,
    verify_stage1_v2_frozen_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
PROBE_CONFIG = ROOT / "configs" / "recoverability" / "stage2_v1_probe.yaml"
FROZEN_STAGE1 = ROOT / "configs" / "recoverability" / "stage1_v2_frozen_result.yaml"


@pytest.mark.parametrize(
    ("operation", "step_fragment"),
    [
        (
            "difference",
            '{"op":"subtract","inputs":["a","b"],"output":"result"}',
        ),
        ("sum", '{"op":"add","inputs":["a","b"],"output":"result"}'),
        (
            "max_minus_min",
            '"steps":[{"op":"max","inputs":["a","b","c","d"],'
            '"output":"high"},{"op":"min","inputs":["a","b","c","d"],'
            '"output":"low"},{"op":"subtract","inputs":["high","low"],'
            '"output":"result"}]',
        ),
    ],
)
def test_stage2_v1_prompt_is_exact_evidence_bound_and_gold_free(
    operation: str,
    step_fragment: str,
) -> None:
    scene = Stage2V1Scene(
        scene_id="dev-000001",
        operation=operation,
        evidence=(8, 4, 5, 9),
    )

    messages = build_stage2_v1_messages(scene)
    serialized = json.dumps(messages, sort_keys=True)
    system = str(messages[0]["content"])

    assert tuple(inspect.signature(build_stage2_v1_messages).parameters) == ("scene",)
    assert len(messages) == 2
    assert '"variables":{"a":8,"b":4,"c":5,"d":9}' in system
    assert step_fragment in system
    assert '"answer":INTEGER' in system
    assert all(
        forbidden not in serialized.lower()
        for forbidden in (
            "gold",
            "true_values",
            "correct_answer",
            "cue_condition",
            "image",
            "question",
            "markdown",
        )
    )


def test_stage2_v1_probe_config_is_closed_development_only_and_one_shot() -> None:
    config = load_stage2_v1_probe_config(PROBE_CONFIG)

    assert config.status == "DEVELOPMENT_PROBE_NOT_RUN"
    assert config.dataset_id == "CVA-Recoverability-Stage2-V1-Dev-Probe"
    assert config.source_dataset_id == "CVA-Chart-Pilot-v0.3"
    assert config.source_split == "dev"
    assert config.scenes == 24
    assert config.format_retries == 0
    assert config.required_program_parse_rate == 1.0
    assert config.required_execution_rate == 1.0
    assert config.required_program_answer_consistency == 1.0
    assert config.allow_rerun is False
    assert config.hypothesis_test is False


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("format_retries: 0", "format_retries: false"),
        ("required_program_parse_rate: 1.0", "required_program_parse_rate: true"),
    ],
)
def test_stage2_v1_probe_config_rejects_boolean_numeric_aliases(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    tampered = tmp_path / "stage2_v1_probe.yaml"
    tampered.write_text(
        PROBE_CONFIG.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        load_stage2_v1_probe_config(tampered)


def test_stage1_v2_server_result_is_frozen_with_external_hashes() -> None:
    frozen = load_stage1_v2_frozen_result(FROZEN_STAGE1)

    assert frozen.status == "FINAL_PASSED_DEVELOPMENT_PROBE"
    assert frozen.scenes == 24
    assert frozen.model_calls == 24
    assert frozen.parse_rate == 1.0
    assert frozen.exact_transcriptions == 22
    assert frozen.exact_transcription_rate == pytest.approx(22 / 24)
    assert frozen.probe_passed is True
    assert frozen.hypothesis_tested is False
    assert frozen.confirmatory_execution_authorized is False
    assert frozen.training_invoked is False
    assert dict(frozen.source_sha256) == {
        "console": "95028b7bcc29f9533e5570068673c3258661a9c18167055df4ff8cfd956e80e8",
        "preflight": "e37e980468f445baf9568df7e40a364c5174c00034eb97fcce889057345eee5f",
        "probe_records": "0b6f0518c84c83bb4b4d78c6d08db526e9449a4b022fcfc10ba607e79cfae7fa",
        "probe_report": "a838645b55117c63114e529cdf38b5f124ed7f404a077823e214093d21c42f3a",
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("parse_rate: 1.0", "parse_rate: true"),
        ("probe_passed: true", "probe_passed: 1"),
    ],
)
def test_stage1_v2_frozen_result_rejects_boolean_numeric_aliases(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    tampered = tmp_path / "stage1_v2_frozen_result.yaml"
    tampered.write_text(
        FROZEN_STAGE1.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        load_stage1_v2_frozen_result(tampered)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage1_v2_artifacts_are_hash_bound_and_semantically_replayed(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "artifact_type": "recoverability_stage1_v2_metadata_preflight",
                "ready": True,
                "large_gpu_started": False,
                "model_loaded": False,
                "training_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    console = tmp_path / "console.log"
    console.write_text("frozen successful probe\n", encoding="utf-8")
    report = tmp_path / "probe_report.json"
    report.write_text(
        json.dumps(
            {
                "artifact_type": "recoverability_stage1_v2_development_probe",
                "dataset_id": "CVA-Recoverability-Stage1-V2-Dev-Probe",
                "source_dataset_id": "CVA-Chart-Pilot-v0.3",
                "source_split": "dev",
                "model_snapshot_sha256": "a" * 64,
                "scenes": 24,
                "model_calls": 24,
                "parse_rate": 1.0,
                "exact_transcription_rate": 22 / 24,
                "probe_passed": True,
                "hypothesis_tested": False,
                "confirmatory_execution_authorized": False,
                "training_invoked": False,
            }
        ),
        encoding="utf-8",
    )
    records = tmp_path / "probe_records.jsonl"
    canonical: list[Stage1V2Scene] = []
    with records.open("x", encoding="utf-8") as stream:
        for index in range(24):
            scene_id = f"dev-{index:06d}"
            values = (8, 4, 5, 9)
            perceived = (7, 4, 5, 9) if index in {3, 19} else values
            operation = ("difference", "max_minus_min", "sum")[index % 3]
            chart_type = ("grouped_bar", "line")[index % 2]
            canonical.append(
                Stage1V2Scene(
                    scene_id=scene_id,
                    image_path=tmp_path / f"{scene_id}.png",
                    chart_type=chart_type,
                    operation=operation,
                    values=values,
                )
            )
            raw = json.dumps(
                {
                    "target_facts": perceived,
                    "redundant_facts": [],
                    "axis_facts": ["integer_ticks"],
                },
                separators=(",", ":"),
            )
            stream.write(
                json.dumps(
                    {
                        "scene_id": scene_id,
                        "chart_type": chart_type,
                        "operation": operation,
                        "raw_text": raw,
                        "parse_success": True,
                        "exact_transcription": perceived == values,
                        "error_code": None,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    frozen = Stage1V2FrozenResult(
        schema_version=1,
        status="FINAL_PASSED_DEVELOPMENT_PROBE",
        dataset_id="CVA-Recoverability-Stage1-V2-Dev-Probe",
        source_dataset_id="CVA-Chart-Pilot-v0.3",
        source_split="dev",
        model_snapshot_sha256="a" * 64,
        scenes=24,
        model_calls=24,
        parse_rate=1.0,
        exact_transcriptions=22,
        exact_transcription_rate=22 / 24,
        mismatch_scene_ids=("dev-000003", "dev-000019"),
        probe_passed=True,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        source_sha256=tuple(
            sorted(
                {
                    "preflight": _sha256(preflight),
                    "console": _sha256(console),
                    "probe_report": _sha256(report),
                    "probe_records": _sha256(records),
                }.items()
            )
        ),
    )

    verification = verify_stage1_v2_frozen_artifacts(
        frozen,
        preflight_path=preflight,
        console_path=console,
        report_path=report,
        records_path=records,
        canonical_scenes=tuple(canonical),
    )

    assert verification.verified is True
    assert verification.exact_transcriptions == 22
    assert verification.mismatch_scene_ids == ("dev-000003", "dev-000019")
    assert len(verification.scenes) == 24

    report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_stage1_v2_frozen_artifacts(
            frozen,
            preflight_path=preflight,
            console_path=console,
            report_path=report,
            records_path=records,
            canonical_scenes=tuple(canonical),
        )


def _scene(index: int, operation: str = "difference") -> Stage2V1Scene:
    return Stage2V1Scene(
        scene_id=f"dev-{index:06d}",
        operation=operation,
        evidence=(8, 4, 5, 9),
    )


def _valid_program(operation: str, *, a: int = 8) -> str:
    variables = {"a": a, "b": 4, "c": 5, "d": 9}
    if operation == "difference":
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
        answer = a - 4
    elif operation == "sum":
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
        answer = a + 4
    else:
        steps = [
            {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
            {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
        answer = max(a, 4, 5, 9) - min(a, 4, 5, 9)
    return json.dumps(
        {"variables": variables, "steps": steps, "answer": answer},
        separators=(",", ":"),
    )


def test_stage2_v1_probe_makes_one_text_call_per_scene_and_passes_only_full_contract() -> None:
    operations = ("difference", "sum", "max_minus_min")
    scenes = tuple(_scene(index, operations[index % 3]) for index in range(24))
    calls: list[str] = []

    def generate(scene: Stage2V1Scene, messages: tuple[dict[str, object], ...]) -> str:
        calls.append(scene.scene_id)
        assert messages == build_stage2_v1_messages(scene)
        return _valid_program(scene.operation)

    report, records = run_stage2_v1_probe(scenes, generate=generate)

    assert calls == [scene.scene_id for scene in scenes]
    assert report.scenes == 24
    assert report.model_calls == 24
    assert report.program_parse_rate == 1.0
    assert report.execution_rate == 1.0
    assert report.program_answer_consistency == 1.0
    assert report.operation_result_accuracy == 1.0
    assert report.format_retries == 0
    assert report.training_invoked is False
    assert report.probe_passed is True
    assert all(record.operation_result_correct for record in records)


@pytest.mark.parametrize(
    ("raw", "expected_error"),
    [
        ("not-json", "program_parse_failure"),
        (_valid_program("difference", a=99), "program_execution_failure"),
    ],
)
def test_stage2_v1_probe_never_retries_or_repairs_failures(
    raw: str,
    expected_error: str,
) -> None:
    scenes = tuple(_scene(index) for index in range(24))
    calls = 0

    def generate(
        _scene_value: Stage2V1Scene,
        _messages: tuple[dict[str, object], ...],
    ) -> str:
        nonlocal calls
        calls += 1
        return raw

    report, records = run_stage2_v1_probe(scenes, generate=generate)

    assert calls == 24
    assert report.model_calls == 24
    assert report.probe_passed is False
    assert {record.error_code for record in records} == {expected_error}


def test_stage2_v1_scene_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        Stage2V1Scene(scene_id="../bad", operation="sum", evidence=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        Stage2V1Scene(scene_id="dev-000001", operation="divide", evidence=(1, 2, 3, 4))
    with pytest.raises(TypeError):
        Stage2V1Scene(scene_id="dev-000001", operation="sum", evidence=(True, 2, 3, 4))
