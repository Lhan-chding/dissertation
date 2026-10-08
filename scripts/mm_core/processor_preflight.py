#!/usr/bin/env python3
"""Actual CPU Qwen processor QA; never load model weights or call generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_core.generator import CANVAS, PLOT, audit_split_rows, validate_dataset
from mm_core.vl_runtime import GENERATION, QwenRuntime, hash_json


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reconstruct_rgb(pixel_values: Any, grid: list[list[int]], image_processor: Any) -> Any:
    """Invert the installed Qwen merged spatial/temporal patch layout exactly.

    Verified against Qwen2VLImageProcessor._preprocess: flattened axes are
    [grid_h/merge, grid_w/merge, merge_h, merge_w, C, temporal, patch_h, patch_w].
    A still image duplicates its temporal plane. No interpolation is performed.
    """
    import numpy as np
    from PIL import Image

    if len(grid) != 1 or grid[0][0] != 1:
        raise ValueError("Only one still-image grid is registered")
    _, grid_h, grid_w = grid[0]
    patch = int(image_processor.patch_size)
    merge = int(image_processor.merge_size)
    temporal = int(image_processor.temporal_patch_size)
    array = (
        pixel_values.detach().cpu().numpy()
        if hasattr(pixel_values, "detach")
        else np.asarray(pixel_values)
    )
    if array.shape != (grid_h * grid_w, 3 * temporal * patch * patch):
        raise ValueError("Unexpected pixel tensor shape")
    patches = array.reshape(
        grid_h // merge, grid_w // merge, merge, merge, 3, temporal, patch, patch
    )
    for plane in range(1, temporal):
        if not np.array_equal(patches[:, :, :, :, :, 0], patches[:, :, :, :, :, plane]):
            raise ValueError("Still-image temporal planes differ")
    chw = (
        patches[:, :, :, :, :, 0]
        .transpose(4, 0, 2, 5, 1, 3, 6)
        .reshape(3, grid_h * patch, grid_w * patch)
    )
    rgb = chw.transpose(1, 2, 0).astype(np.float64)
    if image_processor.do_normalize:
        rgb = rgb * np.asarray(image_processor.image_std) + np.asarray(image_processor.image_mean)
    if image_processor.do_rescale:
        rgb = rgb / float(image_processor.rescale_factor)
    if not np.isfinite(rgb).all() or float(rgb.min()) < -0.01 or float(rgb.max()) > 255.01:
        raise ValueError("Inverse processor values fall outside the image domain")
    return Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))


def _expanded_generation(model_path: Path, processor: Any) -> dict[str, Any]:
    from transformers import GenerationConfig

    original = json.loads((model_path / "generation_config.json").read_text())
    eos = original.get("eos_token_id")
    eos_ids = [int(x) for x in (eos if isinstance(eos, list) else [eos]) if x is not None]
    if not eos_ids:
        raise ValueError("Model generation EOS set is unavailable")
    config = GenerationConfig(
        **GENERATION,
        eos_token_id=eos_ids,
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
    )
    return {
        "generation_config_expanded": config.to_dict(),
        "original_generation_config": original,
        "original_generation_config_sha256": file_hash(model_path / "generation_config.json"),
        "runtime_generation_overrides": GENERATION,
    }


def _environment(torch: Any, processor: Any) -> dict[str, Any]:
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "Pillow",
        "numpy",
        "safetensors",
        "tokenizers",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    processor_class = type(processor.image_processor)
    source = inspect.getsource(processor_class._preprocess)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "processor_class": f"{processor_class.__module__}.{processor_class.__qualname__}",
        "processor_preprocess_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "preflight_code_sha256": file_hash(Path(__file__)),
        "vl_runtime_code_sha256": file_hash(Path(inspect.getfile(QwenRuntime))),
        "cuda_context_initialized": torch.cuda.is_initialized(),
        "model_weights_loaded": False,
        "new_model_calls": 0,
    }


def _contact_sheet(root: Path, processed: list[dict[str, Any]]) -> tuple[str, list[str]]:
    from PIL import Image, ImageDraw, ImageFont

    chosen = {}
    for row in processed:
        chosen.setdefault((row["chart_type"], row["V"], row["D"]), row)
    if len(chosen) != 8:
        raise ValueError("The eight chart/V/D processed sample cells are incomplete")
    sheet = Image.new("RGB", (1792, 736), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    samples = []
    for index, (cell, row) in enumerate(sorted(chosen.items())):
        x, y = index % 4 * 448, index // 4 * 368
        draw.text((x + 8, y + 6), " / ".join(cell), font=font, fill="black")
        with Image.open(root / row["processed_image_path"]) as image:
            sheet.paste(image.resize((448, 336)), (x, y + 30))
        samples.append(row["processed_image_path"])
    path = "data/processed_contact_sheet.png"
    sheet.save(root / path)
    return path, samples


def preflight(run_root: Path, model_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    root = run_root.resolve()
    if (root / "manifests/PRE_INFERENCE_FREEZE.json").exists():
        raise PermissionError("Processor annotation is forbidden after inference freeze")
    if (root / "manifests/PROCESSOR_PREFLIGHT.json").exists():
        raise FileExistsError("Completed processor preflight exists; preserve it")
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preflight cannot start in an initialized CUDA context")
    runtime = QwenRuntime.processor_only(model_path)
    processor = runtime.processor
    environment = _environment(torch, processor)
    write_json(root / "manifests/PROCESSOR_ENVIRONMENT_LOCK.json", environment)
    input_paths = [
        "data/questions.jsonl",
        "data/sidecars.jsonl",
        "data/model_inputs.jsonl",
        "data/images.jsonl",
        "data/sources.jsonl",
    ]
    original_hashes = {path: file_hash(root / path) for path in input_paths}
    backup_path = root / "manifests/PRE_PROCESSOR_DATA_HASHES.json"
    if not backup_path.exists():
        write_json(backup_path, original_hashes)
    questions = read_jsonl(root / "data/questions.jsonl")
    inputs = read_jsonl(root / "data/model_inputs.jsonl")
    if len(questions) != 864 or len(inputs) != 864:
        raise ValueError("Full authorized 864-question panel is required")
    if [r["question_id"] for r in questions] != [r["question_id"] for r in inputs]:
        raise ValueError("Input and private question identities disagree")
    registered_images = {row["image_id"] for row in questions}
    if len(registered_images) != 288:
        raise ValueError("Full authorized 288-image panel is required")
    processed: dict[str, dict[str, Any]] = {}
    routing_rows = []
    pixel_splits: dict[str, set[str]] = defaultdict(set)
    processed_dir = root / "data/processed_images"
    processed_dir.mkdir(parents=True, exist_ok=True)
    maximum_error = 0
    for index, (private, model_input) in enumerate(zip(questions, inputs, strict=True)):
        prepared = runtime.prepare(model_input, root)
        routing = prepared["routing"]
        image_id = private["image_id"]
        if image_id not in processed:
            rgb = reconstruct_rgb(
                prepared["inputs"]["pixel_values"],
                routing["image_grid_thw"],
                processor.image_processor,
            )
            if rgb.size != CANVAS or list(rgb.size) != routing["processed_size"]:
                raise ValueError("Reconstructed dimensions disagree with actual processor grid")
            with Image.open(root / private["image_path"]) as original:
                error = np.abs(
                    np.asarray(rgb, dtype=np.int16)
                    - np.asarray(original.convert("RGB"), dtype=np.int16)
                )
            maximum_error = max(maximum_error, int(error.max()))
            if int(error.max()) > 1:
                raise ValueError("Processor inverse differs by more than one RGB quantization unit")
            path = f"data/processed_images/{image_id}.png"
            rgb.save(root / path)
            spacing = (PLOT[3] - PLOT[1]) / 100 * rgb.height / CANVAS[1]
            grid_y = [
                round((PLOT[3] - value * (PLOT[3] - PLOT[1]) / 100) * rgb.height / CANVAS[1])
                for value in range(101)
            ]
            minimum_gap = min(abs(a - b) for a, b in pairwise(grid_y))
            if spacing < 4 or minimum_gap < 4:
                raise ValueError("Actual processed image violates four pixels per delta")
            processed[image_id] = {
                "image_id": image_id,
                "split": private["split"],
                "root_family_id": private["root_family_id"],
                "chart_type": private["chart_type"],
                "V": private["V"],
                "D": private["D"],
                "source_image_path": private["image_path"],
                "source_image_sha256": private["image_hash"],
                "processed_image_path": path,
                "processed_image_hash": file_hash(root / path),
                "processed_pixel_sha256": routing["processed_pixel_sha256"],
                "processed_size": list(rgb.size),
                "pixels_per_delta": spacing,
                "minimum_raster_grid_gap_pixels": minimum_gap,
                "max_inverse_rgb_error": int(error.max()),
                "processor_hash": runtime.identity["processor_hash"],
                "image_grid_thw": routing["image_grid_thw"],
                "pixel_values_shape": routing["pixel_values_shape"],
                "image_token_count": routing["image_token_count"],
            }
        image_record = processed[image_id]
        if routing["processed_pixel_sha256"] != image_record["processed_pixel_sha256"]:
            raise ValueError("Identical image processing differs across operation prompts")
        pixel_splits[routing["processed_pixel_sha256"]].add(private["split"])
        private.update(
            {
                "processed_image_hash": image_record["processed_image_hash"],
                "processed_pixel_sha256": routing["processed_pixel_sha256"],
                "processed_image_path": image_record["processed_image_path"],
                "processed_size": routing["processed_size"],
                "processed_pixels_per_delta": image_record["pixels_per_delta"],
                "processor_hash": runtime.identity["processor_hash"],
                "chat_template_hash": runtime.identity["chat_template_hash"],
                "processor_status": "ACTUAL_CPU_PROCESSOR_VERIFIED",
                "readability_status": "CPU_PASS_PENDING_VISUAL",
                "actual_model_tokenized_prompt_hash": routing["input_ids_sha256"],
                "actual_chat_text_sha256": routing["chat_text_sha256"],
                "prompt_token_count": routing["input_ids_shape"][1],
            }
        )
        model_input["processed_pixel_sha256"] = routing["processed_pixel_sha256"]
        routing_rows.append(
            {"question_id": private["question_id"], "image_id": image_id, **routing}
        )
        if (index + 1) % 96 == 0:
            print(
                json.dumps({"questions_processed": index + 1, "images_processed": len(processed)}),
                flush=True,
            )
    if len(processed) != 288 or len(routing_rows) != 864:
        raise ValueError("Actual processor coverage is incomplete")
    split_audit = audit_split_rows(questions)
    for fingerprint, splits in pixel_splits.items():
        if len(splits) > 1:
            split_audit["findings"].append(
                {
                    "field": "processed_pixel_sha256",
                    "identity": fingerprint,
                    "splits": sorted(splits),
                }
            )
    if split_audit["findings"]:
        raise ValueError(f"Cross-split processed collisions: {split_audit['findings']}")
    processed_rows = list(processed.values())
    contact_sheet, samples = _contact_sheet(root, processed_rows)
    write_jsonl(root / "data/processed_images.jsonl", processed_rows)
    write_jsonl(root / "data/processor_routing.jsonl", routing_rows)
    write_jsonl(root / "data/questions.jsonl", questions)
    write_jsonl(root / "data/sidecars.jsonl", questions)
    write_jsonl(root / "data/model_inputs.jsonl", inputs)
    images = read_jsonl(root / "data/images.jsonl")
    for row in images:
        image_record = processed[row["image_id"]]
        row.update(
            {
                key: image_record[key]
                for key in (
                    "processed_image_path",
                    "processed_image_hash",
                    "processed_pixel_sha256",
                    "processed_size",
                    "processor_hash",
                )
            }
        )
        row["processed_pixels_per_delta"] = image_record["pixels_per_delta"]
        row["processor_status"] = "ACTUAL_CPU_PROCESSOR_VERIFIED"
        row["readability_status"] = "CPU_PASS_PENDING_VISUAL"
    write_jsonl(root / "data/images.jsonl", images)
    processed_manifest_hash = file_hash(root / "data/processed_images.jsonl")
    split_audit.update(
        {
            "status": "PASS",
            "processed_image_audit": "PASS_ALL_REGISTERED_IMAGES",
            "processed_image_count": len(processed),
            "processed_pixel_count": len(pixel_splits),
            "processed_manifest_sha256": processed_manifest_hash,
            "processor_hash": runtime.identity["processor_hash"],
        }
    )
    write_json(root / "manifests/ROOT_SPLIT_AUDIT.json", split_audit)
    report = {
        "status": "CPU_PASS_PENDING_VISUAL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **runtime.identity,
        "tokenizer_hash": hash_json(processor.tokenizer.get_vocab()),
        **_expanded_generation(model_path, processor),
        "environment_lock_hash": file_hash(root / "manifests/PROCESSOR_ENVIRONMENT_LOCK.json"),
        "environment_lock_path": "manifests/PROCESSOR_ENVIRONMENT_LOCK.json",
        "source_image_count": len(registered_images),
        "processed_image_count": len(processed),
        "question_count": len(routing_rows),
        "processed_size": list(CANVAS),
        "min_pixels_per_delta": min(r["pixels_per_delta"] for r in processed_rows),
        "min_raster_grid_gap_pixels": min(
            r["minimum_raster_grid_gap_pixels"] for r in processed_rows
        ),
        "maximum_inverse_rgb_error": maximum_error,
        "reconstruction": "inverse_actual_normalized_merged_temporal_patch_tensor",
        "processed_manifest_path": "data/processed_images.jsonl",
        "processed_manifest_sha256": processed_manifest_hash,
        "question_routing_path": "data/processor_routing.jsonl",
        "question_routing_sha256": file_hash(root / "data/processor_routing.jsonl"),
        "full_resolution_sample_paths": samples,
        "contact_sheet_path": contact_sheet,
        "artifact_sha256": {
            str(path.relative_to(root)): file_hash(path)
            for path in sorted((root / "data").glob("*.jsonl"))
        },
        "root_split_audit_sha256": file_hash(root / "manifests/ROOT_SPLIT_AUDIT.json"),
        "model_weights_loaded": False,
        "new_model_calls": 0,
        "new_training_runs": 0,
        "cuda_context_initialized": torch.cuda.is_initialized(),
    }
    if report["cuda_context_initialized"]:
        raise RuntimeError("Unexpected CUDA initialization during CPU-only processor preflight")
    generator_path = root / "manifests/GENERATOR_REPORT.json"
    generator = json.loads(generator_path.read_text())
    generator.update(
        {
            "status": "CPU_PASS_PENDING_VISUAL",
            "processor_status": "ACTUAL_CPU_PROCESSOR_VERIFIED",
            "processed_size": list(CANVAS),
            "processed_resolution_qa_status": "PASS",
            "visual_review_status": "CPU_PASS_PENDING_VISUAL",
            "root_split_audit_status": "PASS",
            "prompt_token_count_status": "ACTUAL_PROCESSOR_TOKENIZED",
            "artifact_sha256": report["artifact_sha256"],
            "processor_preflight_path": "manifests/PROCESSOR_PREFLIGHT.json",
        }
    )
    generator["validation"] = validate_dataset(root)
    if generator["validation"]["status"] != "PASS":
        raise ValueError("Annotated dataset validation failed")
    write_json(generator_path, generator)
    report["generator_report_sha256"] = file_hash(generator_path)
    write_json(root / "manifests/PROCESSOR_PREFLIGHT.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        report = preflight(args.run_root, args.model_path)
    except Exception as error:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(
            args.run_root / "engineering" / f"PROCESSOR_PREFLIGHT_FAILURE_{stamp}.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "new_model_calls": 0,
            },
        )
        raise
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "processed_image_count",
                    "question_count",
                    "processed_size",
                    "min_pixels_per_delta",
                    "maximum_inverse_rgb_error",
                    "cuda_context_initialized",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
