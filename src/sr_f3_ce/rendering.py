"""Deterministic readable charts; numeric labels are axes ticks only.

Rendering receipts describe source-image geometry. They explicitly do not certify
native multimodal processor behavior; that requires the separate GPU/runtime gate.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
from collections import defaultdict
from pathlib import Path

from .data import (
    FAMILIES,
    FrozenDataset,
    file_sha256,
    independently_answer,
    object_sha256,
)

CANVAS = (1024, 768)
PLOT = (106, 128, 978, 648)
PALETTES = {
    "main": {"Alpha": "#2869ad", "Beta": "#cf7028", "Gamma": "#6b8c36"},
    "novel": {"Alpha": "#7b4ba5", "Beta": "#008b80", "Gamma": "#c44f73"},
}
MARKERS = {
    "main": ("circle", "square", "triangle"),
    "novel": ("diamond", "triangle", "circle"),
}


def _font_path(font_path=None) -> Path:
    if font_path is not None:
        return Path(font_path).resolve(strict=True)
    # Use a legal installed font, record its bytes, never bundle the font itself.
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    try:
        import matplotlib

        candidate = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if candidate.is_file():
            return candidate.resolve()
    except ImportError:
        pass
    raise ValueError("No readable installed font; supply --font-path explicitly")


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(
                "Existing output differs; preserve it and use a new run root: "
                + str(path)
            )
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A different concurrent preparation must not silently replace an artifact.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(
                    "Concurrent output identity conflict: " + str(path)
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, obj):
    _atomic_bytes(
        path,
        (
            json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode(),
    )


def _marker(draw, x, y, shape, color, radius=6, *, hollow=False):
    box = (x - radius, y - radius, x + radius, y + radius)
    fill = None if hollow else color
    if shape == "circle":
        draw.ellipse(box, fill=fill, outline=color, width=2)
    elif shape == "square":
        draw.rectangle(box, fill=fill, outline=color, width=2)
    elif shape == "diamond":
        draw.polygon(
            [
                (x, y - radius - 1),
                (x + radius + 1, y),
                (x, y + radius + 1),
                (x - radius - 1, y),
            ],
            fill=fill,
            outline=color,
            width=2,
        )
    else:
        draw.polygon(
            [
                (x, y - radius - 1),
                (x + radius + 1, y + radius),
                (x - radius - 1, y + radius),
            ],
            fill=fill,
            outline=color,
            width=2,
        )


def render_chart(task: dict, font_path=None) -> tuple[bytes, dict]:
    """Only world/chart/style influence pixels; query/gold/IDs never enter PNGs."""
    import PIL
    from PIL import Image, ImageDraw, ImageFont, features

    path = _font_path(font_path)
    fonts = {size: ImageFont.truetype(str(path), size) for size in (20, 22, 28)}
    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)
    labels = []

    def text(xy, value, *, size=22, anchor="la"):
        font = fonts[size]
        box = font.getmask(value).getbbox()
        if box is None or box[3] - box[1] < 9:
            raise ValueError("Major character ink height below 9 pixels")
        layout = draw.textbbox(xy, value, font=font, anchor=anchor)
        if min(layout[:2]) < 0 or layout[2] > CANVAS[0] or layout[3] > CANVAS[1]:
            raise ValueError("Visible text would be clipped")
        labels.append(
            {
                "text": value,
                "font_size": size,
                "ink_height_px": box[3] - box[1],
                "bounding_box": list(layout),
            }
        )
        draw.text(xy, value, font=font, fill="#202833", anchor=anchor)

    world, chart, style = task["world"], task["chart"], task["render_style"]
    if chart not in ("grouped_bar", "line") or style not in PALETTES:
        raise ValueError("Unregistered chart style")
    categories = world["categories"]
    series = world["series"]
    if tuple(series) not in (("Alpha", "Beta"), ("Alpha", "Beta", "Gamma")):
        raise ValueError("Unexpected series order")
    left, top, right, bottom = PLOT
    units = (bottom - top) / 100
    if units < 3:
        raise ValueError("One value unit is smaller than 3 source pixels")
    title = {
        4: "Quarterly quantities",
        8: "Eight-category quantities",
    }.get(len(categories), "Six-category quantities")
    text((CANVAS[0] // 2, 24), title, size=28, anchor="ma")
    text((left - 2, 92), "Count", size=20)
    for value in range(0, 101, 10):
        y = bottom - value * units
        draw.line([(left, y), (right, y)], fill="#dce1e7", width=1)
        draw.line([(left - 5, y), (left, y)], fill="#485461", width=2)
        text((left - 12, y), str(value), size=20, anchor="rm")
    draw.line([(left, top), (left, bottom), (right, bottom)], fill="#485461", width=2)
    spacing = (right - left) / len(categories)
    xs = [left + (i + 0.5) * spacing for i in range(len(categories))]
    for x, category in zip(xs, categories, strict=True):
        text((x, bottom + 18), category, size=22, anchor="ma")
    point_series = []
    for index, (name, values) in enumerate(series.items()):
        color = PALETTES[style][name]
        marker = MARKERS[style][index]
        legend_x = 290 + index * 210
        if chart == "line":
            draw.line([(legend_x - 28, 82), (legend_x + 3, 82)], fill=color, width=3)
            _marker(
                draw,
                legend_x - 13,
                82,
                marker,
                color,
                radius=4 + 3 * index,
                hollow=index > 0,
            )
        else:
            draw.rectangle((legend_x - 28, 74, legend_x - 8, 90), fill=color)
        text((legend_x + 11, 82), name, size=22, anchor="lm")
        if chart == "grouped_bar":
            width = min(37, spacing * 0.74 / len(series))
            group_width = width * len(series)
            for x, value in zip(xs, values, strict=True):
                x0 = x - group_width / 2 + index * width
                draw.rectangle(
                    (
                        round(x0 + 1),
                        round(bottom - value * units),
                        round(x0 + width - 1),
                        bottom - 1,
                    ),
                    fill=color,
                    outline=color,
                    width=1,
                )
        else:
            points = [
                (round(x), round(bottom - value * units))
                for x, value in zip(xs, values, strict=True)
            ]
            draw.line(points, fill=color, width=3)
            point_series.append((points, marker, color, index))
    # Draw every line before any data marker. Nested hollow later-series markers
    # retain each series' identity when the registered values coincide exactly.
    for points, marker, color, index in point_series:
        for x, y in points:
            _marker(draw, x, y, marker, color, radius=4 + 3 * index, hollow=index > 0)
    text(
        (CANVAS[0] // 2, 718),
        "Unit: count. Read categories from left to right.",
        size=20,
        anchor="ma",
    )
    target = io.BytesIO()
    image.save(target, format="PNG", optimize=False, compress_level=6)
    data = target.getvalue()
    with Image.open(io.BytesIO(data)) as check:
        if check.size != CANVAS or check.mode != "RGB" or check.info:
            raise ValueError("Unexpected PNG geometry, mode or hidden metadata")
    metrics = {
        "width": CANVAS[0],
        "height": CANVAS[1],
        "pixel_budget": CANVAS[0] * CANVAS[1],
        "pixels_per_value_unit": units,
        "minimum_major_label_ink_height_px": min(
            row["ink_height_px"] for row in labels
        ),
        "numeric_value_labels": False,
        "coincident_line_markers": "nested_solid_and_hollow_after_all_lines",
        "png_metadata": {},
        "labels": labels,
        "chart": chart,
        "render_style": style,
        "category_count": len(categories),
        "series_names": list(series),
        "font_path": str(path),
        "font_sha256": file_sha256(path),
        "pillow_version": PIL.__version__,
        "freetype_version": features.version_module("freetype2"),
    }
    return data, metrics


def fixed_spotchecks(dataset: FrozenDataset) -> list[dict]:
    """Deterministic rule coverage selected from frozen IDs, never model outputs."""
    tasks = dataset.tasks
    chosen = {}

    def choose(rule, predicate):
        options = sorted(
            (t for t in tasks.values() if predicate(t)), key=lambda t: t["qid"]
        )
        if not options:
            raise ValueError("No frozen semantic coverage for: " + rule)
        qid = options[0]["qid"]
        chosen.setdefault(qid, []).append(rule)

    for family in FAMILIES:
        for variant in range(8):
            choose(
                f"{family}: frozen TRAIN_RL variant {variant}",
                lambda t, f=family, v=variant: (
                    t["pool"] == "TRAIN_RL"
                    and t["family"] == f
                    and t["variant"] == v
                    and t["chart"] == "grouped_bar"
                ),
            )
        for pool in ("TEST_COMPOSITION", "TEST_LANGUAGE", "TEST_RENDER", "TEST_ARITY"):
            choose(
                f"{family}: {pool}",
                lambda t, f=family, p=pool: (
                    t["family"] == f and t["pool"] == p and t["chart"] == "line"
                ),
            )
    choose(
        "CROSS explicitly retained empty strict selection",
        lambda t: (
            t["pool"] == "TRAIN_RL"
            and t["family"] == "CROSS"
            and t["variant"] == 0
            and all(
                a <= b
                for a, b in zip(
                    t["world"]["series"]["Alpha"],
                    t["world"]["series"]["Beta"],
                    strict=True,
                )
            )
        ),
    )
    checks = []
    for qid, rules in sorted(chosen.items()):
        task = tasks[qid]
        answer = independently_answer(task["world"], task["query"])
        checks.append(
            {
                "qid": qid,
                "rules": rules,
                "image_file": f"images/{qid}.png",
                "question": task["question"],
                "query": task["query"],
                "expected_answer": None if answer is None else str(answer),
                "independent_answer_checked": True,
                "visual_review_status": "PENDING_HUMAN_OR_AGENT_IMAGE_REVIEW",
            }
        )
    if len(checks) < 24:
        raise ValueError("Fewer than 24 fixed semantic/render checks")
    return checks


def render_dataset(
    dataset: FrozenDataset, output_dir: str | Path, *, font_path=None
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    font = _font_path(font_path)
    cached = {}
    rows = []
    by_hash = defaultdict(set)
    minimum_font = float("inf")
    for qid, task in dataset.tasks.items():
        render_key = object_sha256(
            {key: task[key] for key in ("world", "chart", "render_style")}
        )
        if render_key not in cached:
            cached[render_key] = render_chart(task, font)
        data, metrics = cached[render_key]
        minimum_font = min(minimum_font, metrics["minimum_major_label_ink_height_px"])
        relative = f"images/{qid}.png"
        _atomic_bytes(root / relative, data)
        sha256 = hashlib.sha256(data).hexdigest()
        by_hash[sha256].add(task["root_id"])
        rows.append(
            {
                "qid": qid,
                "image_file": relative,
                "image_sha256": sha256,
                "render_key": render_key,
                "width": CANVAS[0],
                "height": CANVAS[1],
                "chart": task["chart"],
                "render_style": task["render_style"],
            }
        )
    collisions = {
        key: sorted(value) for key, value in by_hash.items() if len(value) > 1
    }
    if collisions:
        raise ValueError(
            "Cross-root image SHA collision; preserve outputs and audit: "
            + str(collisions)
        )
    _atomic_bytes(
        root / "IMAGE_MANIFEST.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ).encode(),
    )
    metrics_example = next(iter(cached.values()))[1]
    audit = {
        "status": "SOURCE_IMAGES_VERIFIED",
        "task_image_routes": len(rows),
        "unique_render_inputs": len(cached),
        "unique_png_sha256": len(by_hash),
        "cross_root_png_sha_collisions": collisions,
        "canvas": list(CANVAS),
        "pixel_budget": CANVAS[0] * CANVAS[1],
        "pixels_per_value_unit": (PLOT[3] - PLOT[1]) / 100,
        "minimum_major_label_ink_height_px": minimum_font,
        "numeric_value_labels": False,
        "font": {
            "path": str(font),
            "sha256": file_sha256(font),
            "bytes_distributed": False,
        },
        "pillow_version": metrics_example["pillow_version"],
        "freetype_version": metrics_example["freetype_version"],
        "python_version": platform.python_version(),
        "renderer_sha256": file_sha256(__file__),
        "image_manifest_sha256": file_sha256(root / "IMAGE_MANIFEST.jsonl"),
        "native_processor_check": "NOT_RUN",
        "model_calls": 0,
    }
    write_json(root / "RENDER_AUDIT.json", audit)
    checks = fixed_spotchecks(dataset)
    write_json(
        root / "SEMANTIC_RENDER_SPOTCHECK.json",
        {
            "selection": "First lexicographic frozen qid per registered coverage rule",
            "count": len(checks),
            "items": checks,
            "selection_used_model_outputs": False,
            "visual_review_status": "PENDING",
        },
    )
    return audit
