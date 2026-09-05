"""Deterministic Pillow charts with explicit a,b / c,d record mapping."""

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .verifiers import executor


def _canvas(variant):
    image = Image.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    colors = ("#2166ac", "#b2182b") if variant == "standard" else ("#007f5f", "#8e44ad")
    for value in range(0, 101, 20):
        y = 420 - value * 3.2
        draw.line((76, y, 716, y), fill="#dedede", width=1)
        draw.text((62, y), str(value), fill="#222222", font=small, anchor="rm")
    draw.line((76, 100, 76, 420, 716, 420), fill="#222222", width=2)
    draw.text((76, 76), "Value", fill="#222222", font=font)
    for x, label in ((256, "Position 1: a, b"), (556, "Position 2: c, d")):
        draw.text((x, 442), label, fill="#222222", font=font, anchor="mt")
    for x, color, label in (
        (90, colors[0], "Series 1 (a, c)"),
        (390, colors[1], "Series 2 (b, d)"),
    ):
        draw.rectangle((x, 48, x + 18, 66), fill=color)
        draw.text((x + 26, 48), label, fill="#222222", font=font)
    return image, draw, font, colors


def render_chart(world, chart_type, path, variant="standard"):
    executor(world, "sum4")
    if chart_type not in {"grouped_bar", "line"}:
        raise ValueError("chart_type must be grouped_bar or line")
    if variant not in {"standard", "alternate_palette"}:
        raise ValueError("unsupported render variant")
    image, draw, font, colors = _canvas(variant)
    draw.text((384, 16), "Four-value chart", fill="#111111", font=font, anchor="mt")
    series_values = [[world[0], world[2]], [world[1], world[3]]]
    labels = ()
    for series, values in enumerate(series_values):
        points = [(x, 420 - value * 3.2) for x, value in zip((256, 556), values, strict=True)]
        if chart_type == "line":
            draw.line(points, fill=colors[series], width=4 if series == 0 else 2)
        for index, ((x, y), value) in enumerate(zip(points, values, strict=True)):
            if chart_type == "grouped_bar":
                x += -42 if series == 0 else 42
                draw.rectangle((x - 32, y, x + 32, 420), fill=colors[series])
            else:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[series])
                x += -12 if series == 0 else 12
            variable = "abcd"[2 * index + series]
            anchor = ("rb" if series == 0 else "lb") if chart_type == "line" else "mb"
            labels = (
                *labels,
                (
                    (x, y - 8),
                    f"{variable}={value}",
                    anchor,
                    colors[series] if chart_type == "line" else "#111111",
                ),
            )
    # Draw labels last with opposite anchors, so close/crossing series stay readable.
    for position, text, anchor, color in labels:
        draw.rectangle(draw.textbbox(position, text, font=font, anchor=anchor), fill="white")
        draw.text(position, text, font=font, fill=color, anchor=anchor)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return {
        "width": 768,
        "height": 512,
        "series_values": series_values,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "render_variant": variant,
    }
