"""Deterministic, bounded MM-CORE-F1 chart generator (CPU only).

Truth-bearing rows are private sidecars. Only ``model_inputs.jsonl`` is an input
manifest, and that file never contains chart values, answers or D labels.
Original PNG geometry cannot certify a model processor's later resampling.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import mean
from typing import Any

VERSION = "mm-core-f1-chart-v1"
PANELS = {
    "FORMAT_TUNE": 4,
    "FORMAT_CHECK": 4,
    "AUDIT_MEASURE": 16,
    "BRIDGE_TRAIN": 8,
    "ENGINE_TEST": 4,
}
CHART_TYPES = ("grouped_bar", "line")
OPERATIONS = ("sum", "difference", "range")
LEVELS = ("low", "high")
CANVAS = (896, 672)
PLOT = (82, 112, 874, 600)
CATEGORIES = ("Cedar", "Birch", "Maple", "Oak", "Pine", "Willow")
SERIES = ("Amber", "Teal", "Violet", "Coral")
COLORS = ("#1261a0", "#d46b08", "#793bb4", "#198274")
TARGET_COLUMNS = (0, 2, 4)
PROMPT_TEMPLATE = (
    "Read the values for the listed chart items, in exactly the listed order.\n"
    "Required items: {ordered_item_descriptions}\n"
    "Operation: {operation_description}\n"
    "Use the chart's numeric unit. Report values to the nearest integer.\n"
    'Return one JSON object with "readings" first and "answer" second:\n'
    '{{"readings": [number, ...], "answer": number}}\n'
    "Both fields must be your own response to the image. Do not include explanations,\n"
    "code blocks, extra fields, or tools."
)
OPERATION_TEXT = {
    "sum": "sum",
    "difference": "first value minus second value",
    "range": "maximum minus minimum",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))


def decimal_difficulty(operation: str, values: list[int]) -> str:
    """Classify the registered units carry/borrow; no negative differences."""
    if len(values) != (3 if operation == "range" else 2):
        raise ValueError("incorrect operand count")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 10 <= v <= 89 for v in values):
        raise ValueError("operands must be two-digit integers in 10..89")
    if operation == "sum":
        high = values[0] % 10 + values[1] % 10 >= 10
    elif operation == "difference":
        if values[0] <= values[1]:
            raise ValueError("ordered difference must be positive")
        high = values[0] % 10 < values[1] % 10
    elif operation == "range":
        high = max(values) % 10 < min(values) % 10
    else:
        raise ValueError(f"unregistered operation: {operation}")
    return "high" if high else "low"


def make_prompt(operation: str) -> str:
    count = 3 if operation == "range" else 2
    items = [f"series Amber at {CATEGORIES[column]}" for column in TARGET_COLUMNS[:count]]
    return PROMPT_TEMPLATE.format(
        ordered_item_descriptions="; ".join(items),
        operation_description=OPERATION_TEXT[operation],
    )


def _numeric_source(
    seed: int, family: str, difficulty: str, candidate_limit: int
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    seed_key = f"{VERSION}|{seed}|{family}|{difficulty}"
    rng = random.Random(int(_sha_bytes(seed_key.encode("utf-8")), 16))
    rejected: Counter[str] = Counter()
    for attempt in range(1, candidate_limit + 1):
        values = [rng.randint(10, 89) for _ in range(3)]
        if values[0] <= values[1]:
            rejected["ordered_difference_not_positive"] += 1
            continue
        if min(abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]) < 5:
            rejected["target_spacing_below_five_units"] += 1
            continue
        labels = {
            op: decimal_difficulty(op, values[: 3 if op == "range" else 2]) for op in OPERATIONS
        }
        if any(label != difficulty for label in labels.values()):
            rejected["carry_borrow_labels_not_jointly_satisfied"] += 1
            continue
        # Both visual variants derive from this complete source, not from a small
        # answer vector. V=low displays the first two series, V=high all four.
        graph = [[rng.randint(10, 89) for _ in CATEGORIES] for _ in SERIES]
        for column, value in zip(TARGET_COLUMNS, values, strict=True):
            graph[0][column] = value
        source = {
            "categories": list(CATEGORIES),
            "series": list(SERIES),
            "values": graph,
            "axis": [0, 100],
            "units": "count",
            "delta": 1,
            "target_series_index": 0,
            "target_columns": list(TARGET_COLUMNS),
        }
        return source, {
            "root_family_id": family,
            "numeric_level": difficulty,
            "attempts": attempt,
            "candidate_limit": candidate_limit,
            "accepted": 1,
            "rejected": dict(sorted(rejected.items())),
            "status": "CONSTRUCTED",
        }
    return None, {
        "root_family_id": family,
        "numeric_level": difficulty,
        "attempts": candidate_limit,
        "candidate_limit": candidate_limit,
        "accepted": 0,
        "rejected": dict(sorted(rejected.items())),
        "status": "GENERATOR_CELL_UNAVAILABLE",
    }


def _font(size: int) -> Any:
    from PIL import ImageFont

    # Record the selected file's hash in the report; never silently change it on
    # resume. A portable Pillow fallback remains legible at an explicit size.
    paths = (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _font_identity() -> dict[str, Any]:
    font = _font(17)
    path = getattr(font, "path", None)
    if isinstance(path, str) and Path(path).is_file():
        return {"name": Path(path).name, "sha256": _sha_bytes(Path(path).read_bytes())}
    return {"name": "Pillow embedded font", "sha256": None}


def _render(
    source: dict[str, Any], chart_type: str, visual_level: str, path: Path
) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    font, small, title = _font(17), _font(14), _font(21)
    left, top, right, bottom = PLOT
    series_count = 2 if visual_level == "low" else 4
    pixels_per_unit = (bottom - top) / 100
    text_bounds: list[tuple[int, int, int, int]] = []

    def text(x: float, y: float, message: str, selected_font: Any = font) -> None:
        bounds = draw.textbbox((x, y), message, font=selected_font)
        text_bounds.append(bounds)
        draw.text((x, y), message, font=selected_font, fill="#182735")

    def y_pos(value: int) -> int:
        return round(bottom - value * pixels_per_unit)

    text(28, 13, "Chart values", title)
    text(595, 19, "Unit: count | Precision: 1 unit", small)
    # V adds distinguishable series and a two-row legend without moving the axis.
    for index in range(series_count):
        legend_col = index if series_count == 2 else index % 2
        legend_row = 0 if series_count == 2 else index // 2
        x, y = 310 + legend_col * 190, 50 + legend_row * 26
        draw.rectangle((x, y + 4, x + 23, y + 14), fill=COLORS[index])
        text(x + 33, y, SERIES[index])
    y_coordinates = [y_pos(value) for value in range(101)]
    for value in range(101):
        y = y_pos(value)
        color = "#c0c9d0" if value % 10 == 0 else "#ecf0f3"
        draw.line((left, y, right, y), fill=color, width=1)
        if value % 10 == 0:
            text(left - 39, y - 8, str(value), small)
    draw.line((left, top, left, bottom, right, bottom), fill="#354655", width=2)
    category_width = (right - left) / len(CATEGORIES)
    positions = [left + (index + 0.5) * category_width for index in range(len(CATEGORIES))]
    for x, category in zip(positions, CATEGORIES, strict=True):
        bounds = draw.textbbox((0, 0), category, font=font)
        text(x - (bounds[2] - bounds[0]) / 2, bottom + 17, category)
    text(306, 643, "Read against the integer grid.", small)
    target_geometry = []
    if chart_type == "grouped_bar":
        group_width = category_width * 0.78
        bar_width = group_width / series_count
        for col, center in enumerate(positions):
            for series_index in range(series_count):
                x0 = center - group_width / 2 + series_index * bar_width + 2
                x1 = x0 + bar_width - 4
                value = source["values"][series_index][col]
                y = y_pos(value)
                draw.rectangle((round(x0), y, round(x1), bottom - 1), fill=COLORS[series_index])
                if series_index == 0 and col in TARGET_COLUMNS:
                    target_geometry.append(
                        {
                            "item_id": f"Amber:{CATEGORIES[col]}",
                            "bbox": [round(x0), y, round(x1), bottom - 1],
                            "value_y": y,
                        }
                    )
    else:
        # Draw the target series last, with a white halo around its markers.
        # Distinct symbols make series identity available without relying on hue.
        for series_index in reversed(range(series_count)):
            points = [
                (x, y_pos(value))
                for x, value in zip(positions, source["values"][series_index], strict=True)
            ]
            draw.line(points, fill=COLORS[series_index], width=2)
            for col, (x, y) in enumerate(points):
                radius = 3
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="white")
                if series_index == 0:
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius), fill=COLORS[series_index]
                    )
                elif series_index == 1:
                    draw.rectangle(
                        (x - radius, y - radius, x + radius, y + radius), fill=COLORS[series_index]
                    )
                elif series_index == 2:
                    draw.polygon(
                        ((x, y - 4), (x - 4, y + 3), (x + 4, y + 3)), fill=COLORS[series_index]
                    )
                else:
                    draw.polygon(
                        ((x, y - 4), (x + 4, y), (x, y + 4), (x - 4, y)), fill=COLORS[series_index]
                    )
                if series_index == 0 and col in TARGET_COLUMNS:
                    target_geometry.append(
                        {
                            "item_id": f"Amber:{CATEGORIES[col]}",
                            "bbox": [round(x - 5), y - 5, round(x + 5), y + 5],
                            "value_y": y,
                        }
                    )
    text_clipped = any(
        x0 < 0 or y0 < 0 or x1 >= CANVAS[0] or y1 >= CANVAS[1] for x0, y0, x1, y1 in text_bounds
    )
    targets_clipped = any(
        g["bbox"][0] < left or g["bbox"][1] < top or g["bbox"][2] > right or g["bbox"][3] > bottom
        for g in target_geometry
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    with Image.open(path) as saved:
        actual_size = list(saved.size)
    minimum_grid_gap = min(abs(a - b) for a, b in pairwise(y_coordinates))
    return {
        "original_size": actual_size,
        "processed_size": None,
        "plot_bounds": list(PLOT),
        "original_pixels_per_delta": pixels_per_unit,
        "minimum_raster_grid_gap_pixels": minimum_grid_gap,
        "processed_pixels_per_delta": None,
        "target_geometry": target_geometry,
        "text_clipped": text_clipped,
        "targets_clipped": targets_clipped,
        "exact_value_data_labels": False,
        "original_geometry_status": "PASS"
        if minimum_grid_gap >= 4 and not text_clipped and not targets_clipped
        else "FAIL",
        "processor_status": "NOT_EXECUTED",
        "readability_status": "READABILITY_REVIEW_REQUIRED",
    }


def audit_split_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect full-source, family, original and processed image split collisions."""
    identifiers = (
        "root_family_id",
        "source_graph_hash",
        "visible_source_hash",
        "image_hash",
        "processed_image_hash",
        "processed_pixel_sha256",
    )
    findings: list[dict[str, Any]] = []
    observed: dict[str, int] = {}
    for key in identifiers:
        by_identity: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.get(key)
            if value:
                by_identity[value].add(row["split"])
        observed[key] = len(by_identity)
        for identity, splits in sorted(by_identity.items()):
            if len(splits) > 1:
                findings.append({"field": key, "identity": identity, "splits": sorted(splits)})
    return {
        "status": "FAIL" if findings else "PASS_FOR_AVAILABLE_ORIGINAL_ASSETS",
        "findings": findings,
        "distinct_counts": observed,
        "processed_image_audit": "PENDING_PROCESSOR"
        if not observed["processed_image_hash"]
        else "CHECKED_PRESENT_HASHES",
        "small_target_vectors_and_answers_are_not_leakage_keys": True,
    }


def _contact_sheet(run_root: Path, image_rows: list[dict[str, Any]]) -> list[str]:
    from PIL import Image, ImageDraw

    # One original from every chart/V/D combination; small previews are explicitly
    # not substituted for actual processor outputs or full-resolution inspection.
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in image_rows:
        chosen.setdefault((row["chart_type"], row["visual_level"], row["numeric_level"]), row)
    sheet = Image.new("RGB", (1792, 736), "white")
    draw = ImageDraw.Draw(sheet)
    samples = []
    for index, (key, row) in enumerate(sorted(chosen.items())):
        x, y = index % 4 * 448, index // 4 * 368
        draw.text((x + 8, y + 6), " / ".join(key), font=_font(16), fill="black")
        with Image.open(run_root / row["image_path"]) as original:
            sheet.paste(original.resize((448, 336)), (x, y + 30))
        samples.append(row["image_path"])
    output = run_root / "data" / "readability_contact_sheet.png"
    sheet.save(output)
    return samples


def generate_dataset(
    run_root: Path, seed: int = 20261008, *, candidate_limit: int = 10000
) -> dict[str, Any]:
    """Generate only the five authorized panels, refusing existing data output.

    The finite candidate limit is an engineering bound, never a search against
    model results. Call ``validate_dataset`` for a read-only audit of a prior run.
    """
    import PIL

    run_root = Path(run_root)
    if candidate_limit < 0:
        raise ValueError("candidate_limit must be nonnegative")
    data = run_root / "data"
    if data.exists() and any(data.iterdir()):
        raise FileExistsError(
            "data already exists; preserve it and validate instead of overwriting"
        )
    data.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    attempts = []
    protocol_hash = _fingerprint({"template": PROMPT_TEMPLATE, "operation_text": OPERATION_TEXT})
    for split, family_count in PANELS.items():
        for index in range(family_count):
            family = f"{split.lower()}-r{index:03d}"
            for difficulty in LEVELS:
                source, receipt = _numeric_source(seed, family, difficulty, candidate_limit)
                attempts.append(receipt)
                if source is None:
                    continue
                numeric_id = f"{family}-d{difficulty}"
                source_hash = _fingerprint(source)
                sources.append(
                    {
                        "split": split,
                        "root_family_id": family,
                        "numeric_root_id": numeric_id,
                        "source_graph_hash": source_hash,
                        "source": source,
                    }
                )
                target_values = [source["values"][0][column] for column in TARGET_COLUMNS]
                for visual in LEVELS:
                    series_count = 2 if visual == "low" else 4
                    visible_source = {
                        **source,
                        "series": source["series"][:series_count],
                        "values": source["values"][:series_count],
                    }
                    for chart in CHART_TYPES:
                        render_variant = f"v{visual}-{chart}"
                        image_id = f"{numeric_id}-{render_variant}"
                        image_path = f"data/images/{image_id}.png"
                        qa = _render(source, chart, visual, run_root / image_path)
                        image_hash = _sha_bytes((run_root / image_path).read_bytes())
                        image_meta = {
                            "split": split,
                            "root_family_id": family,
                            "numeric_root_id": numeric_id,
                            "image_id": image_id,
                            "render_variant_id": render_variant,
                            "chart_type": chart,
                            "visual_level": visual,
                            "numeric_level": difficulty,
                            "source_graph_hash": source_hash,
                            "visible_source_hash": _fingerprint(visible_source),
                            "image_hash": image_hash,
                            "image_sha256": image_hash,
                            "image_path": image_path,
                            **qa,
                        }
                        image_rows.append(image_meta)
                        for operation in OPERATIONS:
                            values = target_values[: 3 if operation == "range" else 2]
                            prompt = make_prompt(operation)
                            prompt_hash = _sha_bytes(prompt.encode("utf-8"))
                            question_id = f"{image_id}-{operation}"
                            ordered_items = [
                                f"Amber:{CATEGORIES[c]}" for c in TARGET_COLUMNS[: len(values)]
                            ]
                            gold = (
                                sum(values)
                                if operation == "sum"
                                else values[0] - values[1]
                                if operation == "difference"
                                else max(values) - min(values)
                            )
                            row = {
                                **image_meta,
                                "question_id": question_id,
                                "readset_id": f"{image_id}-readset-{len(values)}",
                                "operation": operation,
                                "V": visual,
                                "D": difficulty,
                                "ordered_item_ids": ordered_items,
                                "true_values_decimal": [str(v) for v in values],
                                "true_values": values,
                                "gold_answer_decimal": str(gold),
                                "delta_decimal": "1",
                                "delta": 1,
                                "delta_source": "integer_grid_min_raster_gap_4_pixels",
                                "axis_min": "0",
                                "axis_max": "100",
                                "quantity_units": "count",
                                "protocol_hash": protocol_hash,
                                "prompt": prompt,
                                "prompt_sha256": prompt_hash,
                                "actual_model_prompt_hash": prompt_hash,
                                "actual_model_tokenized_prompt_hash": None,
                                "prompt_characters": len(prompt),
                                "prompt_token_count": None,
                                "target_digit_count": [len(str(v)) for v in values],
                                "target_magnitude": max(values),
                                "minimum_target_value_gap": min(
                                    abs(a - b)
                                    for i, a in enumerate(values)
                                    for b in values[i + 1 :]
                                ),
                                "extrema_positions": {
                                    "min": values.index(min(values)),
                                    "max": values.index(max(values)),
                                },
                                "gold_response_characters": len(
                                    json.dumps({"readings": values, "answer": gold})
                                ),
                            }
                            rows.append(row)
                            inputs.append(
                                {
                                    "split": split,
                                    "question_id": question_id,
                                    "image_id": image_id,
                                    "image_path": image_path,
                                    "image_sha256": image_hash,
                                    "prompt": prompt,
                                    "prompt_sha256": prompt_hash,
                                }
                            )
    _write_jsonl(data / "questions.jsonl", rows)
    _write_jsonl(data / "sidecars.jsonl", rows)
    _write_jsonl(data / "model_inputs.jsonl", inputs)
    _write_jsonl(data / "sources.jsonl", sources)
    _write_jsonl(data / "images.jsonl", image_rows)
    samples = _contact_sheet(run_root, image_rows) if image_rows else []
    split_audit = audit_split_rows(rows)
    _write_json(run_root / "manifests" / "ROOT_SPLIT_AUDIT.json", split_audit)
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells["/".join(row[k] for k in ("split", "chart_type", "operation", "V", "D"))].append(row)
    statistics = {}
    for split in PANELS:
        for chart in CHART_TYPES:
            for operation in OPERATIONS:
                for visual in LEVELS:
                    for difficulty in LEVELS:
                        key = "/".join((split, chart, operation, visual, difficulty))
                        selected = cells[key]
                        statistics[key] = {
                            "questions": len(selected),
                            "expected": PANELS[split],
                            "constructible_fraction": len(selected) / PANELS[split],
                            "prompt_characters": sorted({r["prompt_characters"] for r in selected}),
                            "gold_response_characters": [
                                r["gold_response_characters"] for r in selected
                            ],
                            "target_magnitudes": [r["target_magnitude"] for r in selected],
                            "target_value_gaps": [r["minimum_target_value_gap"] for r in selected],
                        }
    unavailable = [a for a in attempts if not a["accepted"]]
    original_qa_pass = bool(image_rows) and all(
        r["original_geometry_status"] == "PASS" for r in image_rows
    )
    report = {
        "generator_version": VERSION,
        "seed": seed,
        "generator_code_sha256": _sha_bytes(Path(__file__).read_bytes()),
        "renderer": {"library": "Pillow", "version": PIL.__version__, "font": _font_identity()},
        "status": "GENERATOR_CELL_UNAVAILABLE" if unavailable else "READABILITY_REVIEW_REQUIRED",
        "authorized_panel_roots": PANELS,
        "root_families": len({r["root_family_id"] for r in rows}),
        "numeric_roots": len(sources),
        "images": len(image_rows),
        "questions": len(rows),
        "expected_questions": sum(PANELS.values()) * 24,
        "panel_question_counts": dict(Counter(r["split"] for r in rows)),
        "candidate_receipts": attempts,
        "unavailable_cells": unavailable,
        "candidate_acceptance_fraction": len(sources) / sum(a["attempts"] for a in attempts)
        if any(a["attempts"] for a in attempts)
        else 0,
        "cell_statistics": statistics,
        "original_size": list(CANVAS),
        "processed_size": None,
        "original_geometry_status": "PASS" if original_qa_pass else "FAIL",
        "original_pixels_per_delta": (PLOT[3] - PLOT[1]) / 100,
        "minimum_raster_grid_gap_pixels": 4,
        "processor_status": "NOT_EXECUTED",
        "processed_resolution_qa_status": "READABILITY_REVIEW_REQUIRED",
        "visual_review_status": "READABILITY_REVIEW_REQUIRED",
        "full_resolution_sample_paths": samples,
        "contact_sheet_path": "data/readability_contact_sheet.png" if samples else None,
        "contact_sheet_is_thumbnail_not_processor_output": True,
        "protocol_sha256": protocol_hash,
        "prompt_token_count_status": "PENDING_TOKENIZER",
        "mean_prompt_characters": mean(r["prompt_characters"] for r in rows) if rows else None,
        "root_split_audit_status": split_audit["status"],
        "input_only_manifest": "data/model_inputs.jsonl",
        "private_truth_files": [
            "data/questions.jsonl",
            "data/sidecars.jsonl",
            "data/sources.jsonl",
        ],
        "artifact_sha256": {
            str(path.relative_to(run_root)): _sha_bytes(path.read_bytes())
            for path in sorted(data.glob("*.jsonl"))
        },
        "new_model_calls": 0,
        "new_training_runs": 0,
        "scientific_measurement_ready": False,
    }
    _write_json(run_root / "manifests" / "GENERATOR_REPORT.json", report)
    report["validation"] = validate_dataset(run_root)
    _write_json(run_root / "manifests" / "GENERATOR_REPORT.json", report)
    return report


def validate_dataset(run_root: Path) -> dict[str, Any]:
    """Read-only checks of actual saved source data, pixels, prompts and splits."""
    from PIL import Image

    run_root = Path(run_root)
    data = run_root / "data"
    rows = [json.loads(line) for line in (data / "questions.jsonl").read_text().splitlines()]
    sources = [json.loads(line) for line in (data / "sources.jsonl").read_text().splitlines()]
    inputs = [json.loads(line) for line in (data / "model_inputs.jsonl").read_text().splitlines()]
    issues = []
    by_source = {r["numeric_root_id"]: r for r in sources}
    if len({r["question_id"] for r in rows}) != len(rows):
        issues.append("duplicate_question_id")
    if Counter(r["split"] for r in rows) != Counter(
        {split: count * 24 for split, count in PANELS.items()}
    ):
        issues.append("registered_panel_counts_incomplete")
    for source in sources:
        if _fingerprint(source["source"]) != source["source_graph_hash"]:
            issues.append(f"source_hash:{source['numeric_root_id']}")
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    images_checked = set()
    processed_checked = set()
    for row in rows:
        qid = row["question_id"]
        if row["split"] not in PANELS:
            issues.append(f"forbidden_split:{qid}")
        if decimal_difficulty(row["operation"], row["true_values"]) != row["D"]:
            issues.append(f"difficulty:{qid}")
        expected_prompt = make_prompt(row["operation"])
        if (
            row["prompt"] != expected_prompt
            or _sha_bytes(row["prompt"].encode("utf-8")) != row["prompt_sha256"]
        ):
            issues.append(f"prompt:{qid}")
        source = by_source[row["numeric_root_id"]]
        expected_values = [
            source["source"]["values"][0][column]
            for column in TARGET_COLUMNS[: len(row["true_values"])]
        ]
        if (
            expected_values != row["true_values"]
            or [str(v) for v in expected_values] != row["true_values_decimal"]
        ):
            issues.append(f"truth_source:{qid}")
        if source["source_graph_hash"] != row["source_graph_hash"]:
            issues.append(f"row_source_hash:{qid}")
        if row["image_id"] not in images_checked:
            path = (run_root / row["image_path"]).resolve()
            if not path.is_relative_to(run_root.resolve()):
                issues.append(f"image_path_escape:{qid}")
                continue
            if _sha_bytes(path.read_bytes()) != row["image_hash"]:
                issues.append(f"image_hash:{qid}")
            with Image.open(path) as image:
                image.load()
                if list(image.size) != row["original_size"] or image.size != CANVAS:
                    issues.append(f"actual_image_dimensions:{qid}")
            if (
                row["minimum_raster_grid_gap_pixels"] < 4
                or row["original_geometry_status"] != "PASS"
            ):
                issues.append(f"original_geometry:{qid}")
            images_checked.add(row["image_id"])
        if row.get("processed_image_hash") and row["image_id"] not in processed_checked:
            processed_path = (run_root / row["processed_image_path"]).resolve()
            if not processed_path.is_relative_to(run_root.resolve()):
                issues.append(f"processed_image_path_escape:{qid}")
                continue
            if _sha_bytes(processed_path.read_bytes()) != row["processed_image_hash"]:
                issues.append(f"processed_image_hash:{qid}")
            with Image.open(processed_path) as processed_image:
                processed_image.load()
                if list(processed_image.size) != row["processed_size"]:
                    issues.append(f"processed_image_dimensions:{qid}")
            if row.get("processed_pixels_per_delta", 0) < 4:
                issues.append(f"processed_image_geometry:{qid}")
            processed_checked.add(row["image_id"])
        if row["operation"] in ("sum", "difference"):
            pairs[row["image_id"]][row["operation"]] = row
    for image_id, pair in pairs.items():
        if set(pair) != {"sum", "difference"}:
            issues.append(f"missing_matched_operation:{image_id}")
            continue
        for key in ("readset_id", "true_values", "ordered_item_ids", "image_hash"):
            if pair["sum"][key] != pair["difference"][key]:
                issues.append(f"matched_readset:{image_id}:{key}")
    allowed_input_keys = {
        "split",
        "question_id",
        "image_id",
        "image_path",
        "image_sha256",
        "prompt",
        "prompt_sha256",
    }
    if len(inputs) != len(rows):
        issues.append("model_input_count")
    for model_input, private in zip(inputs, rows, strict=False):
        expected_input_keys = allowed_input_keys | (
            {"processed_pixel_sha256"} if private.get("processed_pixel_sha256") else set()
        )
        if set(model_input) != expected_input_keys or any(
            model_input[k] != private[k] for k in expected_input_keys
        ):
            issues.append(f"model_input_contract:{model_input.get('question_id')}")
    split_audit = audit_split_rows(rows)
    if split_audit["findings"]:
        issues.append("cross_split_leakage")
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "questions_checked": len(rows),
        "original_images_decoded_and_hashed": len(images_checked),
        "matched_sum_difference_pairs": len(pairs),
        "processed_images_checked": len(processed_checked),
        "processor_and_visual_review": (
            "CPU_PASS_PENDING_VISUAL" if processed_checked else "READABILITY_REVIEW_REQUIRED"
        ),
    }
