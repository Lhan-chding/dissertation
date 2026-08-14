"""Fixed two-protocol bridge between the frozen v0.3 and two-stage interface."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from compbias.gpu_pilot.structured_generation import validate_pilot_trajectory
from compbias.models.structured_parser import ParseStatus, parse_trajectory

from .dsl.executor import evaluate_program

_OPERATION = frozenset({"difference", "sum", "max_minus_min"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_OUTPUT_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class Stage1Evidence:
    target_facts: tuple[int, int, int, int]
    redundant_facts: tuple[object, ...]
    axis_facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BridgeScene:
    scene_id: str
    image_path: Path
    question: str
    operation: str
    values: tuple[int, int, int, int]
    answer: int

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or _IDENTIFIER.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a bounded safe identifier")
        if not isinstance(self.image_path, Path) or not self.image_path.is_absolute():
            raise ValueError("image_path must be absolute")
        if not isinstance(self.question, str) or not self.question or "\x00" in self.question:
            raise ValueError("question must be non-empty and contain no NUL")
        if self.operation not in _OPERATION:
            raise ValueError("operation is not registered")
        if not isinstance(self.values, tuple) or len(self.values) != 4:
            raise ValueError("values must contain exactly four integers")
        if any(type(value) is not int for value in self.values) or type(self.answer) is not int:
            raise TypeError("values and answer must be exact integers")


@dataclass(frozen=True, slots=True)
class BridgeRecord:
    scene_id: str
    legacy_raw: str
    legacy_parse_success: bool
    legacy_answer_correct: bool
    legacy_perception_error: bool
    stage1_raw: str
    stage1_parse_success: bool
    stage1_perception_error: bool
    stage2_raw: str | None
    program_parse_success: bool
    program_execution_success: bool
    program_answer_match: bool
    two_stage_answer_correct: bool


@dataclass(frozen=True, slots=True)
class BridgeReport:
    scenes: int
    model_calls: int
    legacy_parse_rate: float
    legacy_answer_accuracy: float
    legacy_perception_error_rate: float
    stage1_parse_rate: float
    stage1_perception_error_rate: float
    stage2_program_parse_rate: float
    program_answer_consistency: float
    two_stage_answer_accuracy: float
    mean_legacy_response_bytes: float
    mean_stage1_response_bytes: float
    mean_stage2_response_bytes: float
    equivalence_margin: float
    protocols_mergeable: bool
    independent_unit: str


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Stage-1 JSON key: {key}")
        result[key] = value
    return result


def parse_stage1_evidence(raw: str) -> Stage1Evidence:
    """Parse one exact evidence-only JSON object; no answer or reasoning is accepted."""

    if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_OUTPUT_BYTES:
        raise ValueError("Stage-1 output must be bounded text")
    try:
        payload = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Stage-1 output must be one exact JSON object") from error
    if not isinstance(payload, dict) or set(payload) != {
        "target_facts",
        "redundant_facts",
        "axis_facts",
    }:
        raise ValueError("Stage-1 evidence schema is invalid")
    target = payload["target_facts"]
    redundant = payload["redundant_facts"]
    axis = payload["axis_facts"]
    if (
        not isinstance(target, list)
        or len(target) != 4
        or any(type(item) is not int for item in target)
    ):
        raise ValueError("target_facts must contain exactly four integers")
    if redundant != []:
        raise ValueError("bridge redundant_facts must be the empty list")
    if not isinstance(axis, list) or any(
        not isinstance(item, str) or _IDENTIFIER.fullmatch(item) is None for item in axis
    ):
        raise ValueError("axis_facts must contain bounded identifiers")
    return Stage1Evidence(tuple(target), (), tuple(axis))  # type: ignore[arg-type]


def build_stage1_messages(*, question: str) -> tuple[dict[str, object], ...]:
    if not isinstance(question, str) or not question or "\x00" in question:
        raise ValueError("question must be non-empty and contain no NUL")
    system = (
        "You are a strict perception interface. Read the image once. Return one JSON object "
        "with exactly target_facts, redundant_facts, and axis_facts. Do not reason or compute "
        "the requested result. Use four integers in A,B,C,D order, [] for redundant_facts, "
        'and ["integer_ticks"] for axis_facts. No Markdown or trailing text.'
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    )


def build_stage2_messages(
    *,
    evidence: Stage1Evidence,
    question: str,
    operation: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(evidence, Stage1Evidence):
        raise TypeError("evidence must be Stage1Evidence")
    if operation not in _OPERATION:
        raise ValueError("operation is not registered")
    if not isinstance(question, str) or not question or "\x00" in question:
        raise ValueError("question must be non-empty and contain no NUL")
    public = {
        "target_facts": list(evidence.target_facts),
        "redundant_facts": [],
        "axis_facts": list(evidence.axis_facts),
    }
    grammar = (
        '{"variables":{"a":INTEGER,"b":INTEGER,"c":INTEGER,"d":INTEGER},'
        '"steps":[{"op":"WHITELISTED_OP","inputs":["VARIABLE"],'
        '"output":"NEW_VARIABLE"}],"answer":INTEGER}'
    )
    system = (
        "You are a strict text-only reasoning interface. The image is unavailable. Return one "
        "exact JSON DSL program with variables, steps, and answer. Allowed ops: read, add, "
        "subtract, max, min, argmax, argmin, solve_sum_constraint, "
        "solve_difference_constraint, interpolate_arithmetic_progression, lookup_duplicate, "
        "compare. Every input must reference a prior variable and the final step must compute "
        "the reported answer. No Markdown or trailing text. Grammar: "
        f"{grammar}"
    )
    user = json.dumps(
        {
            "evidence": public,
            "question": question,
            "operation": operation,
            "cue_condition": "ablated",
            "image_available": False,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )


def _legacy_parse(scene: BridgeScene, raw: str) -> tuple[bool, bool, bool]:
    parsed = validate_pilot_trajectory(
        parse_trajectory(raw, sample_id=scene.scene_id),
        operation=scene.operation,
        expected_value_count=4,
    )
    parse_success = parsed.status is ParseStatus.OK
    perceived = parsed.perceived_scene.get("values") if parsed.perceived_scene else None
    perception_error = parse_success and perceived != scene.values
    answer_correct = parse_success and parsed.answer == scene.answer
    return parse_success, bool(answer_correct), bool(perception_error)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_bridge_protocol(
    scenes: tuple[BridgeScene, ...],
    *,
    legacy_generate: Callable[[BridgeScene], str],
    stage1_generate: Callable[[BridgeScene, tuple[dict[str, object], ...]], str],
    stage2_generate: Callable[[BridgeScene, tuple[dict[str, object], ...]], str],
    equivalence_margin: float,
) -> tuple[BridgeReport, tuple[BridgeRecord, ...]]:
    """Run fixed calls once per scene; failures remain failures and are never retried."""

    if not isinstance(scenes, tuple) or not scenes:
        raise ValueError("scenes must be a non-empty tuple")
    if any(not isinstance(scene, BridgeScene) for scene in scenes):
        raise TypeError("scenes must contain BridgeScene instances")
    identifiers = [scene.scene_id for scene in scenes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("bridge scene identifiers must be unique")
    if (
        isinstance(equivalence_margin, bool)
        or not isinstance(equivalence_margin, (int, float))
        or not math.isfinite(float(equivalence_margin))
        or not 0 < float(equivalence_margin) < 1
    ):
        raise ValueError("equivalence_margin must lie in (0, 1)")
    records: list[BridgeRecord] = []
    legacy_lengths: list[int] = []
    stage1_lengths: list[int] = []
    stage2_lengths: list[int] = []
    model_calls = 0
    for scene in scenes:
        legacy_raw = legacy_generate(scene)
        model_calls += 1
        legacy_parse, legacy_correct, legacy_perception = _legacy_parse(scene, legacy_raw)
        stage1_raw = stage1_generate(scene, build_stage1_messages(question=scene.question))
        model_calls += 1
        stage1_parse = False
        stage1_perception = False
        stage2_raw: str | None = None
        evaluation = evaluate_program("", constraint_bindings={})
        try:
            evidence = parse_stage1_evidence(stage1_raw)
            stage1_parse = True
            stage1_perception = evidence.target_facts != scene.values
        except ValueError:
            evidence = None
        if evidence is not None:
            stage2_raw = stage2_generate(
                scene,
                build_stage2_messages(
                    evidence=evidence,
                    question=scene.question,
                    operation=scene.operation,
                ),
            )
            model_calls += 1
            evaluation = evaluate_program(stage2_raw, constraint_bindings={})
            stage2_lengths.append(len(stage2_raw.encode("utf-8")))
        two_stage_correct = bool(
            evaluation.program_parse_success
            and evaluation.program_execution_success
            and evaluation.program_answer_match
            and evaluation.final_answer == scene.answer
        )
        records.append(
            BridgeRecord(
                scene_id=scene.scene_id,
                legacy_raw=legacy_raw,
                legacy_parse_success=legacy_parse,
                legacy_answer_correct=legacy_correct,
                legacy_perception_error=legacy_perception,
                stage1_raw=stage1_raw,
                stage1_parse_success=stage1_parse,
                stage1_perception_error=stage1_perception,
                stage2_raw=stage2_raw,
                program_parse_success=evaluation.program_parse_success,
                program_execution_success=evaluation.program_execution_success,
                program_answer_match=evaluation.program_answer_match,
                two_stage_answer_correct=two_stage_correct,
            )
        )
        legacy_lengths.append(len(legacy_raw.encode("utf-8")))
        stage1_lengths.append(len(stage1_raw.encode("utf-8")))
    total = len(records)
    legacy_parse_rate = sum(item.legacy_parse_success for item in records) / total
    legacy_accuracy = sum(item.legacy_answer_correct for item in records) / total
    legacy_perception = sum(item.legacy_perception_error for item in records) / total
    stage1_parse_rate = sum(item.stage1_parse_success for item in records) / total
    stage1_perception = sum(item.stage1_perception_error for item in records) / total
    program_parse = sum(item.program_parse_success for item in records) / total
    program_consistency = sum(item.program_answer_match for item in records) / total
    two_stage_accuracy = sum(item.two_stage_answer_correct for item in records) / total
    margin = float(equivalence_margin)
    protocols_mergeable = bool(
        legacy_parse_rate >= 0.95
        and stage1_parse_rate >= 0.98
        and program_consistency >= 0.95
        and abs(legacy_accuracy - two_stage_accuracy) <= margin
        and abs(legacy_perception - stage1_perception) <= margin
    )
    return (
        BridgeReport(
            scenes=total,
            model_calls=model_calls,
            legacy_parse_rate=legacy_parse_rate,
            legacy_answer_accuracy=legacy_accuracy,
            legacy_perception_error_rate=legacy_perception,
            stage1_parse_rate=stage1_parse_rate,
            stage1_perception_error_rate=stage1_perception,
            stage2_program_parse_rate=program_parse,
            program_answer_consistency=program_consistency,
            two_stage_answer_accuracy=two_stage_accuracy,
            mean_legacy_response_bytes=_mean(legacy_lengths),
            mean_stage1_response_bytes=_mean(stage1_lengths),
            mean_stage2_response_bytes=_mean(stage2_lengths),
            equivalence_margin=margin,
            protocols_mergeable=protocols_mergeable,
            independent_unit="semantic_scene",
        ),
        tuple(records),
    )


def decode_text_qwen_once(
    model: object,
    processor: object,
    messages: tuple[dict[str, object], ...],
) -> str:
    """Decode a text-only second-stage call with no image input."""

    import torch

    text = processor.apply_chat_template(  # type: ignore[attr-defined]
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(  # type: ignore[operator]
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(  # type: ignore[attr-defined]
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]  # type: ignore[attr-defined]
