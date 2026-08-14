"""Offline Qwen2.5-VL smoke inference with structured evidence parsing."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from compbias.models.structured_parser import ParseStatus

from .config import load_pilot_paths
from .preflight import model_snapshot_sha256
from .safe_io import atomic_write_bytes, atomic_write_json_text
from .structured_generation import generate_with_format_retries, numeric_answer_matches


def load_local_qwen(
    model_path: Path,
    *,
    model_class: Any | None = None,
    processor_class: Any | None = None,
    torch_dtype: object | None = None,
) -> tuple[object, object]:
    """Load only local audited bytes; never execute repository custom code."""

    if not model_path.is_absolute() or not model_path.is_dir():
        raise ValueError("model_path must be an existing absolute directory")
    if model_class is None or processor_class is None or torch_dtype is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_class = Qwen2_5_VLForConditionalGeneration if model_class is None else model_class
        processor_class = AutoProcessor if processor_class is None else processor_class
        torch_dtype = torch.bfloat16 if torch_dtype is None else torch_dtype
    common = {"local_files_only": True, "trust_remote_code": False}
    model = model_class.from_pretrained(
        str(model_path.resolve()),
        torch_dtype=torch_dtype,
        device_map="cuda:0",
        **common,
    )
    processor = processor_class.from_pretrained(str(model_path.resolve()), **common)
    return model, processor


def render_smoke_chart(path: Path, *, output_root: Path | None = None) -> int:
    image = Image.new("RGB", (512, 384), "white")
    draw = ImageDraw.Draw(image)
    values = (3, 7, 5)
    colors = ("#3b82f6", "#f97316", "#10b981")
    for index, (value, color) in enumerate(zip(values, colors, strict=True)):
        left = 90 + index * 120
        draw.rectangle((left, 320 - value * 30, left + 70, 320), fill=color)
        draw.text((left + 25, 330), chr(ord("A") + index), fill="black")
        draw.text((left + 25, 300 - value * 30), str(value), fill="black")
    draw.text((90, 30), "Smoke chart: values A, B, C", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    atomic_write_bytes(path.parent if output_root is None else output_root, path, buffer.getvalue())
    return max(values) - min(values)


def _with_image(
    messages: tuple[dict[str, object], ...],
    image_path: Path,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        if message["role"] == "user":
            result.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": message["content"]},
                    ],
                }
            )
        else:
            result.append(dict(message))
    return result


def decode_qwen_once(
    model: object,
    processor: object,
    image_path: Path,
    messages: tuple[dict[str, object], ...],
) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    multimodal_messages = _with_image(messages, image_path)
    text = processor.apply_chat_template(  # type: ignore[attr-defined]
        multimodal_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(multimodal_messages)
    inputs = processor(  # type: ignore[operator]
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(  # type: ignore[attr-defined]
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]  # type: ignore[attr-defined]


def run_smoke(model_path: Path, output_dir: Path) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    snapshot_before = model_snapshot_sha256(model_path)
    image_path = output_dir / "smoke_chart.png"
    expected = render_smoke_chart(image_path, output_root=output_dir.parent)
    model, processor = load_local_qwen(model_path)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    generation = generate_with_format_retries(
        lambda messages: decode_qwen_once(model, processor, image_path, messages),
        question="Read the chart and compute max minus min.",
        operation="max_minus_min",
        sample_id="smoke-000001",
        expected_value_count=3,
    )
    torch.cuda.synchronize(0)
    latency = time.perf_counter() - started
    parsed = generation.parsed
    format_passed = parsed.status is ParseStatus.OK
    answer_correct = format_passed and numeric_answer_matches(parsed.answer, expected)
    smoke_passed = format_passed and answer_correct
    snapshot_after = model_snapshot_sha256(model_path)
    if snapshot_after != snapshot_before:
        raise RuntimeError("model snapshot changed during known-answer smoke")
    report = {
        "schema_version": 1,
        "artifact_type": "qwen25vl3b_offline_smoke",
        "training_invoked": False,
        "model_path": str(model_path.resolve()),
        "model_snapshot_sha256": snapshot_before,
        "expected_answer": expected,
        "raw_response": generation.raw_text,
        "parsed": parsed.to_mapping(),
        "format_attempts": list(generation.attempts),
        "format_retries": len(generation.attempts) - 1,
        "format_passed": format_passed,
        "smoke_passed": smoke_passed,
        "answer_correct": answer_correct,
        "latency_seconds": latency,
        "peak_memory_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
    }
    target = output_dir / "smoke_report.json"
    atomic_write_json_text(
        output_dir.parent,
        target,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = load_pilot_paths(args.paths)
        report = run_smoke(paths.model_path, paths.outputs / "smoke")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("smoke_passed") is True and report.get("answer_correct") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
