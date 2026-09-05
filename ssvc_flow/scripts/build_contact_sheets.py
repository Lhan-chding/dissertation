"""Prepare 36 original-resolution calibration charts for independent human review."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_KEYS = ("constraint_family", "chart_type", "operation")
CELLS = tuple(
    itertools.product(
        ("duplicate_encoding", "cross_series", "trend"),
        ("grouped_bar", "line"),
        ("sum4", "difference_pairs", "range4"),
    )
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_scenes(scenes):
    """Pick two scenes per cell by opaque ID; no model results enter selection."""
    if any(scene["split"] != "calibration" for scene in scenes):
        raise ValueError("contact sheets may read only calibration scenes")
    grouped = {
        cell: sorted(
            (scene for scene in scenes if tuple(scene[key] for key in CELL_KEYS) == cell),
            key=lambda scene: scene["base_scene_id"],
        )
        for cell in CELLS
    }
    if any(len(rows) < 2 for rows in grouped.values()):
        raise ValueError(
            "at least two calibration images are required for every one of the 18 cells"
        )
    selected = [[grouped[cell][rank] for cell in CELLS] for rank in (0, 1)]
    if len({scene["base_scene_id"] for sheet in selected for scene in sheet}) != 36:
        raise ValueError("selected calibration scene IDs must be unique")
    return selected


def _image_path(dataset_root, scene):
    path = (dataset_root / scene["image_path"]).resolve()
    if not path.is_relative_to(dataset_root.resolve()):
        raise ValueError("image path escapes the dataset directory")
    if _sha256(path) != scene["image_hash"]:
        raise ValueError("image hash does not match the generated scene")
    return path


def _sheet(dataset_root, out, scenes, number):
    sheet = Image.new("RGB", (2304, 3408), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for index, scene in enumerate(scenes):
        x, y = (index % 3) * 768, (index // 3) * 568
        with Image.open(_image_path(dataset_root, scene)) as source:
            if source.size != (768, 512):
                raise ValueError("contact sheets require original 768x512 source charts")
            sheet.paste(source.convert("RGB"), (x, y + 56))
        draw.text((x + 12, y + 4), scene["base_scene_id"], font=font, fill="#111111")
        draw.text(
            (x + 12, y + 29), " / ".join(scene[key] for key in CELL_KEYS), font=font, fill="#111111"
        )
        draw.rectangle((x, y, x + 767, y + 567), outline="#bbbbbb", width=1)
    filename = f"calibration_contact_sheet_{number:02d}.png"
    sheet.save(out / filename, format="PNG", optimize=False)
    return {
        "path": filename,
        "sha256": _sha256(out / filename),
        "width": 2304,
        "height": 3408,
        "scenes": [
            {key: scene[key] for key in ("base_scene_id", *CELL_KEYS, "image_path", "image_hash")}
            for scene in scenes
        ],
    }


def build_contact_sheets(dataset_root, out):
    dataset_root, out = Path(dataset_root), Path(out)
    source_manifest = json.loads((dataset_root / "manifest.json").read_text())
    if not source_manifest.get("images_rendered"):
        raise ValueError("source dataset images must be rendered")
    calibration_path = dataset_root / "calibration.jsonl"
    source_hash = _sha256(calibration_path)
    if source_hash != source_manifest["files"]["calibration.jsonl"]["sha256"]:
        raise ValueError("calibration source hash does not match the dataset manifest")
    scenes = [json.loads(line) for line in calibration_path.read_text().splitlines()]
    selected = select_scenes(scenes)
    out.mkdir(parents=True, exist_ok=True)
    sheets = [
        _sheet(dataset_root, out, rows, number) for number, rows in enumerate(selected, start=1)
    ]
    manifest = {
        "source_split": "calibration",
        "source_calibration_sha256": source_hash,
        "source_dataset_hash": source_manifest["dataset_hash"],
        "total_images": 36,
        "selection_rule": "Two lowest opaque base_scene_id values per family/chart/operation cell",
        "image_path_base": "dataset_directory",
        "human_review_status": "PENDING",
        "agent_review_status": "NOT_REVIEWED",
        "sheets": sheets,
    }
    (out / "contact_sheets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_contact_sheets(args.dataset, args.out)
    print(
        json.dumps(
            {
                "total_images": manifest["total_images"],
                "sheets": [sheet["path"] for sheet in manifest["sheets"]],
                "human_review_status": manifest["human_review_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
