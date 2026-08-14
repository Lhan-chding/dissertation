from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.bridge import (
    BridgeScene,
    build_stage1_messages,
    build_stage2_messages,
    parse_stage1_evidence,
    run_bridge_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


def _scene(index: int) -> BridgeScene:
    values = (3 + index % 4, 7, 5, 2)
    return BridgeScene(
        scene_id=f"bridge_{index:03d}",
        image_path=Path(f"/dataset/images/bridge_{index:03d}.png"),
        question="What is the maximum value minus the minimum value?",
        operation="max_minus_min",
        values=values,
        answer=max(values) - min(values),
    )


def test_stage1_schema_contains_evidence_only_and_rejects_answer_leakage() -> None:
    raw = json.dumps(
        {
            "target_facts": [3, 7, 5, 2],
            "redundant_facts": [],
            "axis_facts": ["integer_ticks"],
        },
        separators=(",", ":"),
    )
    evidence = parse_stage1_evidence(raw)

    assert evidence.target_facts == (3, 7, 5, 2)
    assert evidence.redundant_facts == ()
    assert evidence.axis_facts == ("integer_ticks",)
    for invalid in (
        raw + " trailing",
        raw.replace('"axis_facts"', '"answer":4,"axis_facts"'),
        raw.replace('"target_facts"', '"reasoning":[],"target_facts"'),
        raw.replace("[3,7,5,2]", "[3,7,5]"),
    ):
        with pytest.raises(ValueError):
            parse_stage1_evidence(invalid)


def test_two_stage_messages_never_include_image_or_hidden_gold_in_stage2() -> None:
    scene = _scene(0)
    stage1 = build_stage1_messages(question=scene.question)
    evidence = parse_stage1_evidence(
        '{"target_facts":[3,7,5,2],"redundant_facts":[],"axis_facts":["integer_ticks"]}'
    )
    stage2 = build_stage2_messages(
        evidence=evidence,
        question=scene.question,
        operation=scene.operation,
    )

    assert all('"type": "image"' not in json.dumps(message) for message in stage2)
    assert "gold" not in json.dumps(stage1).lower()
    assert "gold" not in json.dumps(stage2).lower()
    assert '"gold_answer"' not in json.dumps(stage2)
    assert "cue_condition" not in json.dumps(stage2)
    assert "ablated" not in json.dumps(stage2)
    assert len(stage1) == len(stage2) == 2


def test_bridge_runs_exactly_one_legacy_and_one_two_stage_trajectory_per_scene() -> None:
    scenes = tuple(_scene(index) for index in range(6))
    calls = {"legacy": 0, "stage1": 0, "stage2": 0}

    def legacy(scene: BridgeScene) -> str:
        calls["legacy"] += 1
        values = ",".join(str(value) for value in scene.values)
        return (
            f'<perception>{{"values":[{values}]}}</perception>'
            '<reasoning>{"operation":"max_minus_min"}</reasoning>'
            f"<answer>{scene.answer}</answer>"
        )

    def stage1(scene: BridgeScene, _messages: tuple[dict[str, object], ...]) -> str:
        calls["stage1"] += 1
        return json.dumps(
            {
                "target_facts": list(scene.values),
                "redundant_facts": [],
                "axis_facts": ["integer_ticks"],
            },
            separators=(",", ":"),
        )

    def stage2(
        scene: BridgeScene,
        _messages: tuple[dict[str, object], ...],
    ) -> str:
        calls["stage2"] += 1
        return json.dumps(
            {
                "variables": {
                    "a": scene.values[0],
                    "b": scene.values[1],
                    "c": scene.values[2],
                    "d": scene.values[3],
                },
                "steps": [
                    {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
                    {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
                    {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
                ],
                "answer": scene.answer,
            },
            separators=(",", ":"),
        )

    report, records = run_bridge_protocol(
        scenes,
        legacy_generate=legacy,
        stage1_generate=stage1,
        stage2_generate=stage2,
        equivalence_margin=0.03,
    )

    assert calls == {"legacy": 6, "stage1": 6, "stage2": 6}
    assert len(records) == 6
    assert report.scenes == 6
    assert report.legacy_answer_accuracy == 1.0
    assert report.two_stage_answer_accuracy == 1.0
    assert report.stage1_parse_rate == 1.0
    assert report.program_answer_consistency == 1.0
    assert report.protocols_mergeable is True
    assert report.accuracy_difference_ci90 == (0.0, 0.0)
    assert report.perception_difference_ci90 == (0.0, 0.0)
    assert report.model_calls == 18
    assert report.independent_unit == "semantic_scene"


def test_bridge_parse_failure_is_a_failure_and_never_retried_or_sent_to_stage2() -> None:
    scenes = (_scene(0), _scene(1))
    stage2_calls = 0

    def stage2(_scene_value: BridgeScene, _messages: tuple[dict[str, object], ...]) -> str:
        nonlocal stage2_calls
        stage2_calls += 1
        raise AssertionError("stage2 must not run after Stage-1 parse failure")

    report, records = run_bridge_protocol(
        scenes,
        legacy_generate=lambda scene: "malformed",
        stage1_generate=lambda _scene_value, _messages: "malformed",
        stage2_generate=stage2,
        equivalence_margin=0.03,
    )

    assert stage2_calls == 0
    assert report.stage1_parse_rate == 0.0
    assert report.two_stage_answer_accuracy == 0.0
    assert all(record.stage1_parse_success is False for record in records)
    assert report.protocols_mergeable is False


def test_bridge_requires_paired_equivalence_intervals_not_point_differences() -> None:
    scenes = tuple(_scene(index) for index in range(20))

    def legacy(scene: BridgeScene) -> str:
        values = ",".join(str(value) for value in scene.values)
        answer = scene.answer if int(scene.scene_id[-3:]) else scene.answer + 1
        return (
            f'<perception>{{"values":[{values}]}}</perception>'
            '<reasoning>{"operation":"max_minus_min"}</reasoning>'
            f"<answer>{answer}</answer>"
        )

    def stage1(scene: BridgeScene, _messages: tuple[dict[str, object], ...]) -> str:
        return json.dumps(
            {
                "target_facts": list(scene.values),
                "redundant_facts": [],
                "axis_facts": ["integer_ticks"],
            },
            separators=(",", ":"),
        )

    def stage2(scene: BridgeScene, _messages: tuple[dict[str, object], ...]) -> str:
        return json.dumps(
            {
                "variables": dict(zip(("a", "b", "c", "d"), scene.values, strict=True)),
                "steps": [
                    {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "hi"},
                    {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "lo"},
                    {"op": "subtract", "inputs": ["hi", "lo"], "output": "result"},
                ],
                "answer": scene.answer,
            },
            separators=(",", ":"),
        )

    report, _ = run_bridge_protocol(
        scenes,
        legacy_generate=legacy,
        stage1_generate=stage1,
        stage2_generate=stage2,
        equivalence_margin=0.03,
    )

    assert abs(report.legacy_answer_accuracy - report.two_stage_answer_accuracy) == pytest.approx(
        0.05
    )
    assert report.accuracy_difference_ci90[1] > 0.03
    assert report.protocols_mergeable is False


def test_bridge_cli_requires_explicit_server_execution_flag() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "03_bridge.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--execute" in completed.stdout
    assert "--server-package-lock" in completed.stdout
    assert "--preflight-report" in completed.stdout
    assert "--external-evidence" in completed.stdout
    assert "--v03-records" in completed.stdout
    blocked = subprocess.run(
        [
            sys.executable,
            str(script),
            "--paths",
            "configs/paths.yaml",
            "--protocol",
            "configs/recoverability/recoverability_v1.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout
