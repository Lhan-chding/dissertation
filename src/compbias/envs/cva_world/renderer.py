"""Pure image renderers that expose scene evidence and never task answers."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Final

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .schema import CVASample, TaskFamily

SUPPORTED_VISUAL_STYLES: Final[tuple[str, ...]] = (
    "baseline",
    "font_weight_bold",
    "size_compact",
    "rotation_tilted",
    "contrast_low",
    "background_grid",
    "occlusion_local",
    "blur_mild",
    "distractor_marks",
    "layout_shifted",
)
"""Closed visual-factor catalog used by the frozen CVA v2 protocol."""

_LEGACY_VISUAL_STYLE_ALIASES: Final[dict[str, str]] = {
    "font_a": "baseline",
    "font_b": "font_weight_bold",
    "rotated": "rotation_tilted",
}

VISUAL_STYLE_APPLICABILITY: Final[Mapping[str, tuple[TaskFamily, ...]]] = MappingProxyType(
    {
        style: (
            (
                TaskFamily.DIGIT_OFFSET,
                TaskFamily.GAUGE_CALIBRATION,
                TaskFamily.BAR_CHART_AGGREGATE,
            )
            if style == "font_weight_bold"
            else tuple(TaskFamily)
        )
        for style in SUPPORTED_VISUAL_STYLES
    }
)
"""Task families where each visual factor has a literal rendering effect."""


def validate_visual_style(style: str) -> str:
    """Return the canonical style or reject an unregistered visual factor."""

    if not isinstance(style, str):
        raise TypeError("visual style must be a string")
    canonical = _LEGACY_VISUAL_STYLE_ALIASES.get(style, style)
    if canonical not in SUPPORTED_VISUAL_STYLES:
        supported = ", ".join(SUPPORTED_VISUAL_STYLES)
        raise ValueError(f"unsupported visual style {style!r}; expected one of: {supported}")
    return canonical


def is_visual_style_applicable(style: str, task_family: TaskFamily | str) -> bool:
    """Whether a named factor is meaningful for the task family's evidence."""

    canonical = validate_visual_style(style)
    try:
        family = TaskFamily(task_family)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported task_family: {task_family!r}") from error
    return family in VISUAL_STYLE_APPLICABILITY[canonical]


def sample_render_coordinates(sample_id: str, *, base_seed: int) -> tuple[int, int]:
    """Return stable semantic seed and realization index encoded by a sample ID."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    match = re.fullmatch(r"(.+)_r(\d{2})", sample_id)
    if match is None:
        raise ValueError("sample_id must end in a two-digit realization suffix")
    semantic_id, realization = match.groups()
    semantic_offset = int.from_bytes(
        hashlib.sha256(semantic_id.encode("ascii")).digest()[:8], "big"
    )
    return base_seed + semantic_offset, int(realization)


def build_contact_sheet(
    rendered: Sequence[tuple[str, Image.Image]], *, columns: int = 5
) -> Image.Image:
    """Build the canonical contact sheet used by generation and replay audit."""

    if not rendered:
        raise ValueError("rendered must contain at least one image")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    tile_width, tile_height = 272, 292
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (sample_id, image) in enumerate(rendered):
        if not isinstance(sample_id, str) or not isinstance(image, Image.Image):
            raise TypeError("rendered entries must be (sample_id, PIL image) pairs")
        row, column = divmod(position, columns)
        x, y = column * tile_width, row * tile_height
        preview = image.copy()
        preview.thumbnail((256, 256), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + 8, y + 8))
        draw.text((x + 8, y + 268), sample_id[:40], fill="black")
    return sheet


@dataclass(frozen=True)
class RenderConfig:
    """Deterministic raster dimensions, visual style, and local seed."""

    width: int = 256
    height: int = 192
    style: str = "baseline"
    seed: int = 0
    realization_index: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or not 1 <= self.width <= 4096
        ):
            raise ValueError("width must be an integer from 1 to 4096")
        if (
            isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or not 1 <= self.height <= 4096
        ):
            raise ValueError("height must be an integer from 1 to 4096")
        validate_visual_style(self.style)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.realization_index, bool)
            or not isinstance(self.realization_index, int)
            or self.realization_index < 0
        ):
            raise ValueError("realization_index must be a non-negative integer")


def _palette(style: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if style == "contrast_low":
        return ((220, 222, 218), (133, 136, 132), (162, 169, 175))
    if style == "background_grid":
        return ((244, 247, 250), (27, 34, 43), (49, 104, 159))
    return ((249, 249, 246), (28, 32, 38), (58, 113, 165))


def _stable_seed(config: RenderConfig, *, salt: int) -> int:
    """Mix the two public seed coordinates without using global RNG state."""

    return (
        (config.seed & 0xFFFFFFFF) * 1_000_003 + config.realization_index * 97_409 + salt * 65_537
    ) & 0xFFFFFFFFFFFFFFFF


def _font(style: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load Pillow's bundled font at a deterministic size."""

    size = max(8, size)
    return ImageFont.load_default(size=size)


def _draw_background(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    style: str,
    config: RenderConfig,
) -> None:
    if style != "background_grid":
        return
    rng = random.Random(_stable_seed(config, salt=11))
    spacing = 18 + 4 * rng.randrange(3)
    color = (219, 228, 236)
    offset_x = rng.randrange(spacing)
    offset_y = rng.randrange(spacing)
    for x in range(offset_x, image.width, spacing):
        draw.line((x, 0, x, image.height - 1), fill=color, width=1)
    for y in range(offset_y, image.height, spacing):
        draw.line((0, y, image.width - 1, y), fill=color, width=1)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TypeError(f"{field} must be a finite number")
    return float(value)


def _number_label(value: float) -> str:
    """Format a scene measurement without adding representation-only noise."""

    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _centered_text(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    text: str,
    fill: tuple[int, int, int],
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
    stroke_width: int = 0,
) -> None:
    font = font or ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    x = (canvas.width - (box[2] - box[0])) // 2
    y = (canvas.height - (box[3] - box[1])) // 2
    draw.text(
        (x, y),
        text,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=fill,
    )


def _style_width(width: int, *, bold: bool) -> int:
    return max(1, width + (1 if bold else 0))


def _draw_digit(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: Mapping[str, object],
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
    text_stroke_width: int = 0,
) -> None:
    value = scene.get("value")
    _number(value, "value")
    margin = max(5, min(image.size) // 12)
    draw.rounded_rectangle(
        (margin, margin, image.width - margin - 1, image.height - margin - 1),
        radius=max(2, margin // 2),
        outline=accent,
        width=max(1, margin // 3),
    )
    _centered_text(
        draw,
        image,
        str(value),
        ink,
        font=font,
        stroke_width=text_stroke_width,
    )


def _draw_count(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: Mapping[str, object],
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    bold: bool = False,
) -> None:
    count_value = scene.get("count")
    if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 0:
        raise ValueError("count must be a non-negative integer")
    if count_value == 0:
        draw.line(
            (10, image.height // 2, image.width - 10, image.height // 2),
            fill=ink,
            width=_style_width(2, bold=bold),
        )
        return
    columns = max(1, math.ceil(math.sqrt(count_value * image.width / image.height)))
    rows = math.ceil(count_value / columns)
    cell_width = image.width / (columns + 1)
    cell_height = image.height / (rows + 1)
    radius = max(1, int(min(cell_width, cell_height) * 0.22))
    shape = scene.get("shape", "circle")
    for index in range(count_value):
        row, column = divmod(index, columns)
        x = int((column + 1) * cell_width)
        y = int((row + 1) * cell_height)
        bounds = (x - radius, y - radius, x + radius, y + radius)
        if shape == "square":
            draw.rectangle(bounds, fill=accent, outline=ink, width=_style_width(1, bold=bold))
        else:
            draw.ellipse(bounds, fill=accent, outline=ink, width=_style_width(1, bold=bold))


def _draw_gauge(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: Mapping[str, object],
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    bold: bool = False,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
) -> None:
    reading = _number(scene.get("reading"), "reading")
    minimum = _number(scene.get("minimum", 0), "minimum")
    maximum = _number(scene.get("maximum", 10), "maximum")
    if maximum <= minimum or not minimum <= reading <= maximum:
        raise ValueError("gauge requires minimum <= reading <= maximum")
    cx, cy = image.width // 2, int(image.height * 0.72)
    radius = max(8, int(min(image.width * 0.38, image.height * 0.58)))
    bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.arc(
        bounds,
        200,
        340,
        fill=ink,
        width=_style_width(max(2, radius // 18), bold=bold),
    )
    fraction = (reading - minimum) / (maximum - minimum)
    angle = math.radians(200 + 140 * fraction)
    endpoint = (
        cx + int(radius * 0.82 * math.cos(angle)),
        cy + int(radius * 0.82 * math.sin(angle)),
    )
    draw.line(
        (cx, cy, *endpoint),
        fill=accent,
        width=_style_width(max(2, radius // 15), bold=bold),
    )
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=ink)
    font = font or ImageFont.load_default()
    stroke_width = 1 if bold else 0
    label_y = min(image.height - 12, cy + 4)
    draw.text(
        (max(2, cx - radius), label_y),
        _number_label(minimum),
        fill=ink,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=ink,
    )
    maximum_label = _number_label(maximum)
    maximum_box = draw.textbbox((0, 0), maximum_label, font=font)
    draw.text(
        (min(image.width - (maximum_box[2] - maximum_box[0]) - 2, cx + radius - 8), label_y),
        maximum_label,
        fill=ink,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=ink,
    )
    reading_label = _number_label(reading)
    reading_box = draw.textbbox((0, 0), reading_label, font=font)
    draw.text(
        (cx - (reading_box[2] - reading_box[0]) // 2, max(2, cy - radius - 13)),
        reading_label,
        fill=accent,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=accent,
    )


def _draw_bars(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: Mapping[str, object],
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    bold: bool = False,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
) -> None:
    bars = scene.get("bars")
    if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)) or not bars:
        raise ValueError("bars must be a non-empty sequence")
    values = tuple(_number(value, "bar") for value in bars)
    minimum = _number(scene.get("minimum", 0), "minimum")
    observed_maximum = max(values)
    default_maximum = 10 ** max(1, math.ceil(math.log10(observed_maximum)))
    maximum = _number(scene.get("maximum", default_maximum), "maximum")
    if minimum < 0 or min(values) < minimum or observed_maximum <= minimum:
        raise ValueError("bar heights must be within a non-negative absolute scale")
    if maximum <= minimum or observed_maximum > maximum:
        raise ValueError("maximum must exceed minimum and contain every bar height")
    font = font or ImageFont.load_default()
    stroke_width = 1 if bold else 0
    margin = max(18, image.width // 12)
    baseline = image.height - max(20, image.height // 8)
    chart_top = max(10, image.height // 12)
    line_width = _style_width(2, bold=bold)
    draw.line((margin, baseline, image.width - margin, baseline), fill=ink, width=line_width)
    draw.line((margin, chart_top, margin, baseline), fill=ink, width=line_width)
    draw.text(
        (2, baseline - 5),
        _number_label(minimum),
        fill=ink,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=ink,
    )
    draw.text(
        (2, chart_top - 5),
        _number_label(maximum),
        fill=ink,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=ink,
    )
    slot = (image.width - 2 * margin) / len(values)
    for index, value in enumerate(values):
        bar_width = max(2, int(slot * 0.58))
        height = max(
            1,
            round((baseline - chart_top) * (value - minimum) / (maximum - minimum)),
        )
        center = margin + (index + 0.5) * slot
        draw.rectangle(
            (int(center - bar_width / 2), baseline - height, int(center + bar_width / 2), baseline),
            fill=accent,
            outline=ink,
            width=_style_width(1, bold=bold),
        )
        value_label = _number_label(value)
        value_box = draw.textbbox((0, 0), value_label, font=font)
        draw.text(
            (
                int(center - (value_box[2] - value_box[0]) / 2),
                min(baseline + 4, image.height - 12),
            ),
            value_label,
            fill=ink,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=ink,
        )


def _draw_relation(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    scene: Mapping[str, object],
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    bold: bool = False,
) -> None:
    relation = scene.get("relation")
    if relation not in {"left_of", "right_of", "above", "below", "parallel", "intersect"}:
        raise ValueError(f"unsupported relation: {relation!r}")
    cx, cy = image.width // 2, image.height // 2
    dx, dy = max(12, image.width // 5), max(12, image.height // 5)
    radius = max(3, min(image.size) // 18)
    if relation in {"left_of", "right_of"}:
        first, second = ((cx - dx, cy), (cx + dx, cy))
        if relation == "right_of":
            first, second = second, first
        first_bounds = (
            first[0] - radius,
            first[1] - radius,
            first[0] + radius,
            first[1] + radius,
        )
        second_bounds = (
            second[0] - radius,
            second[1] - radius,
            second[0] + radius,
            second[1] + radius,
        )
        draw.ellipse(first_bounds, fill=accent, outline=ink, width=_style_width(1, bold=bold))
        draw.rectangle(
            second_bounds,
            fill=ink,
            outline=accent,
            width=_style_width(1, bold=bold),
        )
    elif relation in {"above", "below"}:
        first, second = ((cx, cy - dy), (cx, cy + dy))
        if relation == "below":
            first, second = second, first
        first_bounds = (
            first[0] - radius,
            first[1] - radius,
            first[0] + radius,
            first[1] + radius,
        )
        second_bounds = (
            second[0] - radius,
            second[1] - radius,
            second[0] + radius,
            second[1] + radius,
        )
        draw.ellipse(first_bounds, fill=accent, outline=ink, width=_style_width(1, bold=bold))
        draw.rectangle(
            second_bounds,
            fill=ink,
            outline=accent,
            width=_style_width(1, bold=bold),
        )
    elif relation == "parallel":
        line_width = _style_width(3, bold=bold)
        draw.line((cx - dx, cy - dy, cx - dx // 2, cy + dy), fill=ink, width=line_width)
        draw.line((cx + dx // 2, cy - dy, cx + dx, cy + dy), fill=accent, width=line_width)
    else:
        line_width = _style_width(3, bold=bold)
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=ink, width=line_width)
        draw.line((cx + dx, cy - dy, cx - dx, cy + dy), fill=accent, width=line_width)


def _render_evidence(
    scene: Mapping[str, object],
    family: TaskFamily,
    size: tuple[int, int],
    style: str,
    config: RenderConfig,
    background: tuple[int, int, int],
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
) -> Image.Image:
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    _draw_background(draw, image, style, config)
    if family is TaskFamily.DIGIT_OFFSET:
        font_size = max(16, min(image.size) // 3)
        _draw_digit(
            image,
            draw,
            scene,
            ink,
            accent,
            font=_font(style, font_size),
            text_stroke_width=1 if style == "font_weight_bold" else 0,
        )
    else:
        renderers = {
            TaskFamily.COUNT_TRANSFORM: _draw_count,
            TaskFamily.GAUGE_CALIBRATION: _draw_gauge,
            TaskFamily.BAR_CHART_AGGREGATE: _draw_bars,
            TaskFamily.RELATION_RULE: _draw_relation,
        }
        bold = style == "font_weight_bold" and is_visual_style_applicable(style, family)
        if family in {TaskFamily.GAUGE_CALIBRATION, TaskFamily.BAR_CHART_AGGREGATE}:
            renderers[family](
                image,
                draw,
                scene,
                ink,
                accent,
                bold=bold,
                font=_font(style, max(10, min(image.size) // 16)),
            )
        else:
            renderers[family](image, draw, scene, ink, accent, bold=bold)
    return image


def _paste_position(
    outer_size: tuple[int, int],
    content_size: tuple[int, int],
    config: RenderConfig,
) -> tuple[int, int]:
    outer_width, outer_height = outer_size
    content_width, content_height = content_size
    center = ((outer_width - content_width) // 2, (outer_height - content_height) // 2)
    if validate_visual_style(config.style) != "layout_shifted":
        return center
    positions = (
        (max(4, outer_width // 20), max(4, outer_height // 20)),
        (outer_width - content_width - max(4, outer_width // 20), max(4, outer_height // 20)),
        (max(4, outer_width // 20), outer_height - content_height - max(4, outer_height // 20)),
        (
            outer_width - content_width - max(4, outer_width // 20),
            outer_height - content_height - max(4, outer_height // 20),
        ),
    )
    return positions[_stable_seed(config, salt=17) % len(positions)]


def _apply_local_occlusion(
    image: Image.Image,
    background: tuple[int, int, int],
    config: RenderConfig,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    rng = random.Random(_stable_seed(config, salt=23))
    width = max(8, image.width // 8)
    height = max(6, image.height // 14)
    quadrants = (
        (image.width // 5, image.height // 4),
        (image.width * 3 // 5, image.height // 4),
        (image.width // 5, image.height * 2 // 3),
        (image.width * 3 // 5, image.height * 2 // 3),
    )
    x, y = quadrants[rng.randrange(len(quadrants))]
    draw.rounded_rectangle(
        (x, y, min(image.width - 2, x + width), min(image.height - 2, y + height)),
        radius=max(1, height // 4),
        fill=background,
        outline=(175, 178, 181),
    )
    return result


def _apply_distractors(image: Image.Image, config: RenderConfig) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    rng = random.Random(_stable_seed(config, salt=29))
    color = (183, 119, 70)
    radius = max(2, min(image.size) // 50)
    corners = (
        (image.width // 12, image.height // 12),
        (image.width * 11 // 12, image.height // 12),
        (image.width // 12, image.height * 11 // 12),
        (image.width * 11 // 12, image.height * 11 // 12),
    )
    first = rng.randrange(len(corners))
    for index in (first, (first + 1) % len(corners)):
        x, y = corners[index]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.line((x - radius, y, x + radius, y), fill=color, width=1)
        draw.line((x, y - radius, x, y + radius), fill=color, width=1)
    return result


def render_scene(
    scene: Mapping[str, object],
    task_family: TaskFamily | str,
    config: RenderConfig,
) -> Image.Image:
    """Render only scene evidence, deterministically and without global RNG use."""

    if not isinstance(scene, Mapping):
        raise TypeError("scene must be a mapping")
    if not isinstance(config, RenderConfig):
        raise TypeError("config must be a RenderConfig")
    try:
        family = TaskFamily(task_family)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported task_family: {task_family!r}") from error
    style = validate_visual_style(config.style)
    background, ink, accent = _palette(style)
    content_scale = 0.68 if style == "size_compact" else 0.78 if style == "layout_shifted" else 1.0
    content_size = (
        max(32, round(config.width * content_scale)),
        max(32, round(config.height * content_scale)),
    )
    content = _render_evidence(
        scene,
        family,
        content_size,
        style,
        config,
        background,
        ink,
        accent,
    )
    image = Image.new("RGB", (config.width, config.height), background)
    image.paste(content, _paste_position(image.size, content.size, config))
    if style == "rotation_tilted":
        angles = (-7, -5, 5, 7)
        angle = angles[(config.seed + config.realization_index) % len(angles)]
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=background)
    elif style == "occlusion_local":
        image = _apply_local_occlusion(image, background, config)
    elif style == "blur_mild":
        radii = (0.65, 0.85, 1.05, 1.25)
        radius = radii[_stable_seed(config, salt=31) % len(radii)]
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif style == "distractor_marks":
        image = _apply_distractors(image, config)
    return image.convert("RGB")


def render_sample(sample: CVASample, config: RenderConfig) -> Image.Image:
    """Render a sample from scene, family, and its registered visual style only."""

    if not isinstance(sample, CVASample):
        raise TypeError("sample must be a CVASample")
    registered_config = replace(config, style=sample.split_keys.visual_style)
    return render_scene(sample.scene, sample.task_family, registered_config)
