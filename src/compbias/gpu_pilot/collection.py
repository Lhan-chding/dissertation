"""Natural structured-evidence collection from the local Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from compbias.models.structured_parser import ParseStatus, parse_trajectory

from .config import load_pilot_paths
from .qwen_smoke import load_local_qwen


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} in {path} is not a JSON object")
            records.append(value)
    return tuple(records)


def _prompt(record: Mapping[str, object]) -> str:
    return (
        "Read the chart. Return exactly "
        '<perception>{"values":[...]}</perception>'
        f'<reasoning>{{"operation":"{record["operation"]}"}}</reasoning>'
        "<answer>NUMBER</answer>. Do not add any other text.\n"
        f"Question: {record['question']}"
    )


def _model_answer(model: object, processor: object, image: Path, prompt: str) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(  # type: ignore[operator]
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)  # type: ignore[attr-defined]
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]  # type: ignore[attr-defined]


def _error_type(record: Mapping[str, object], parsed: object) -> str:
    if parsed.status is not ParseStatus.OK:  # type: ignore[attr-defined]
        return "parse_failure"
    perceived = parsed.perceived_scene  # type: ignore[attr-defined]
    expected_values = record.get("values")
    perceived_values = perceived.get("values") if isinstance(perceived, Mapping) else None
    perception_correct = perceived_values == expected_values
    answer_correct = str(parsed.answer).strip() == str(record.get("answer")).strip()  # type: ignore[attr-defined]
    if perception_correct and answer_correct:
        return "none"
    if not perception_correct and answer_correct:
        return "compensated_visual_error"
    if not perception_correct:
        return "visual_error"
    return "reasoning_error"


def collect_split(
    dataset_dir: Path,
    model_path: Path,
    output_path: Path,
    *,
    split: str,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"collection output already exists: {output_path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = tuple(
        record
        for record in _read_jsonl(dataset_dir / "records.jsonl")
        if record.get("split") == split
    )
    if not source:
        raise ValueError(f"no records found for split {split}")
    model, processor = load_local_qwen(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    correct = 0
    parsed_count = 0
    with output_path.open("x", encoding="utf-8") as stream:
        for index, record in enumerate(source):
            raw = _model_answer(
                model,
                processor,
                dataset_dir / str(record["image"]),
                _prompt(record),
            )
            parsed = parse_trajectory(raw, sample_id=str(record["sample_id"]))
            error_type = _error_type(record, parsed)
            counts[error_type] = counts.get(error_type, 0) + 1
            answer_correct = (
                parsed.status is ParseStatus.OK
                and str(parsed.answer).strip() == str(record["answer"]).strip()
            )
            correct += int(answer_correct)
            parsed_count += int(parsed.status is ParseStatus.OK)
            result = {
                **record,
                "rollout_id": f"{split}-rollout-{index:06d}",
                "raw_text": raw,
                "parsed": parsed.to_mapping(),
                "reward": int(answer_correct),
                "error_type": error_type,
            }
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    total = len(source)
    visual_errors = counts.get("visual_error", 0) + counts.get("compensated_visual_error", 0)
    return {
        "schema_version": 1,
        "split": split,
        "records": total,
        "answer_accuracy": correct / total,
        "parse_rate": parsed_count / total,
        "natural_perception_error_rate": visual_errors / total,
        "error_counts": counts,
        "output": str(output_path),
    }


def calibration_gate(report: Mapping[str, object]) -> tuple[str, ...]:
    failures: list[str] = []
    accuracy = float(report["answer_accuracy"])
    parse_rate = float(report["parse_rate"])
    error_rate = float(report["natural_perception_error_rate"])
    counts = report["error_counts"]
    assert isinstance(counts, Mapping)
    supported = sum(int(count) >= 10 for name, count in counts.items() if name != "none")
    if not 0.30 <= accuracy <= 0.75:
        failures.append("base_answer_accuracy_outside_30_75_percent")
    if not 0.15 <= error_rate <= 0.50:
        failures.append("natural_perception_error_outside_15_50_percent")
    if parse_rate < 0.95:
        failures.append("evidence_parse_rate_below_95_percent")
    if supported < 3:
        failures.append("fewer_than_three_supported_natural_error_families")
    return tuple(failures)


def main(argv: Sequence[str] | None = None, *, calibration: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--split", default="calibration" if calibration else "pilot_train")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("BLOCKED: natural collection requires explicit --execute on the reviewed GPU server")
        return 2
    try:
        paths = load_pilot_paths(args.paths)
        dataset = paths.data / "generated" / "cva_chart_pilot_v0_1"
        target = paths.trajectories / "natural" / f"{args.split}_records.jsonl"
        report = collect_split(dataset, paths.model_path, target, split=args.split)
        failures = calibration_gate(report) if calibration else ()
        report = {**report, "gate_failures": list(failures), "gate_passed": not failures}
        report_path = target.with_suffix(".summary.json")
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if not failures else 3
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3


def calibration_main(argv: Sequence[str] | None = None) -> int:
    return main(argv, calibration=True)
