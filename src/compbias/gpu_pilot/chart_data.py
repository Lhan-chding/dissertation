"""Versioned deterministic CVA-Chart-Pilot renderer and manifest writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from .config import PilotDataConfig, load_pilot_data_config, load_pilot_paths


def _answer(values: tuple[int, ...], operation: str) -> int:
    if operation == "sum":
        return values[0] + values[1]
    if operation == "difference":
        return values[0] - values[1]
    if operation == "max_minus_min":
        return max(values) - min(values)
    raise ValueError(f"unsupported operation: {operation}")


def _question(operation: str) -> str:
    return {
        "sum": "What is the sum of the first two values?",
        "difference": "What is the first value minus the second value?",
        "max_minus_min": "What is the maximum value minus the minimum value?",
    }[operation]


def _draw_chart(
    path: Path,
    *,
    chart_type: str,
    values: tuple[int, ...],
    size: tuple[int, int],
    ood: bool,
    render_mode: str,
) -> None:
    width, height = size
    image = Image.new("RGB", size, "#f4f1e8" if ood else "white")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = 70, 50, width - 40, height - 55
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
    if render_mode == "direct_labels_v0_1":
        max_value = max(values) + 2
        ticks: tuple[int, ...] = ()
    elif render_mode == "axis_scale_v0_2":
        max_value = max(20, max(values) + max(values) % 2)
        ticks = tuple(range(0, max_value + 1, 2))
    else:
        raise ValueError(f"unsupported render mode: {render_mode}")
    x_step = (plot_right - plot_left) / len(values)
    for tick in ticks:
        y = plot_bottom - tick / max_value * (plot_bottom - plot_top)
        if tick:
            draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.line((plot_left - 4, y, plot_left, y), fill="black", width=1)
        draw.text((plot_left - 28, y - 6), str(tick), fill="#374151")
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        x = plot_left + (index + 0.5) * x_step
        y = plot_bottom - value / max_value * (plot_bottom - plot_top)
        points.append((x, y))
        draw.text((x - 4, plot_bottom + 12), chr(ord("A") + index), fill="black")
        if render_mode == "direct_labels_v0_1":
            draw.text((x - 4, y - 18), str(value), fill="black")
    if chart_type == "grouped_bar":
        colors = ("#1d4ed8", "#ea580c", "#059669", "#7c3aed")
        for (x, y), color in zip(points, colors, strict=True):
            half = x_step * 0.22
            draw.rectangle((x - half, y, x + half, plot_bottom), fill=color)
            if ood:
                draw.line((x - half, y, x + half, plot_bottom), fill="#111827", width=1)
    elif chart_type == "line":
        draw.line(points, fill="#b91c1c" if ood else "#1d4ed8", width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#111827")
    else:
        raise ValueError(f"unsupported chart type: {chart_type}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_bundle_sha256(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(root / relative).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _records(config: PilotDataConfig) -> Iterator[dict[str, object]]:
    rng = random.Random(config.seed)
    global_index = 0
    for split, count in config.split_counts.items():
        for index in range(count):
            chart_type = config.chart_types[(global_index + index) % len(config.chart_types)]
            operation = config.operations[(global_index + index) % len(config.operations)]
            values = tuple(rng.randint(2, 18) for _ in range(4))
            sample_id = f"{split}-{index:06d}"
            yield {
                "schema_version": 1,
                "dataset_id": config.dataset_id,
                "sample_id": sample_id,
                "split": split,
                "chart_type": chart_type,
                "operation": operation,
                "values": list(values),
                "question": _question(operation),
                "answer": _answer(values, operation),
                "image": f"images/{sample_id}.png",
                "mechanism": "shifted_style" if split == "mechanism_ood" else "iid",
            }
        global_index += count


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def generate_dataset(config: PilotDataConfig, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"pilot dataset output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        records = tuple(_records(config))
        for record in records:
            relative = record["image"]
            if not isinstance(relative, str):
                raise TypeError("image must be a relative path")
            _draw_chart(
                output_dir / relative,
                chart_type=str(record["chart_type"]),
                values=tuple(int(value) for value in record["values"]),  # type: ignore[arg-type]
                size=config.image_size,
                ood=record["split"] == "mechanism_ood",
                render_mode=config.render_mode,
            )
        records_path = output_dir / "records.jsonl"
        _write_jsonl(records_path, records)
        iid = tuple(record for record in records if record["split"] == "iid_test")
        pairs: list[dict[str, object]] = []
        for pair_index, source in enumerate(iid[: config.counterfactual_pairs]):
            values = tuple(int(value) for value in source["values"])  # type: ignore[arg-type]
            changed = (values[0] + 3, *values[1:])
            counterfactual_id = f"counterfactual-{pair_index:06d}"
            operation = str(source["operation"])
            relative = f"counterfactual/{counterfactual_id}.png"
            _draw_chart(
                output_dir / relative,
                chart_type=str(source["chart_type"]),
                values=changed,
                size=config.image_size,
                ood=True,
                render_mode=config.render_mode,
            )
            pairs.append(
                {
                    "pair_id": f"pair-{pair_index:06d}",
                    "source_sample_id": source["sample_id"],
                    "counterfactual_sample_id": counterfactual_id,
                    "image": relative,
                    "values": list(changed),
                    "question": source["question"],
                    "answer": _answer(changed, operation),
                    "operation": operation,
                }
            )
        pairs_path = output_dir / "counterfactual_pairs.jsonl"
        _write_jsonl(pairs_path, pairs)
        calibration = tuple(
            record["sample_id"] for record in records if record["split"] == "calibration"
        )
        audit_ids = calibration[: config.natural_audit]
        manifest = {
            "schema_version": 1,
            "dataset_id": config.dataset_id,
            "record_count": len(records),
            "split_counts": dict(config.split_counts),
            "counterfactual_pairs": len(pairs),
            "natural_audit_ids": list(audit_ids),
            "records_path": "records.jsonl",
            "records_sha256": _sha256(records_path),
            "counterfactual_path": "counterfactual_pairs.jsonl",
            "counterfactual_sha256": _sha256(pairs_path),
            "images_generated": len(records) + len(pairs),
            "images_sha256": _image_bundle_sha256(
                output_dir,
                [str(record["image"]) for record in records]
                + [str(pair["image"]) for pair in pairs],
            ),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
    except BaseException:
        shutil = __import__("shutil")
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = load_pilot_paths(args.paths)
        config = load_pilot_data_config(args.config)
        target = paths.data / "generated" / config.output_slug
        manifest = generate_dataset(config, target)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
