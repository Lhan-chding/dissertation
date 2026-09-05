import hashlib
import json
from collections import Counter

import pytest
from PIL import Image

from scripts.build_contact_sheets import build_contact_sheets, select_scenes
from src.generate_worlds import generate_dataset


def test_contact_sheets_preserve_all_cells_and_review_status(tmp_path):
    dataset, out, repeated = tmp_path / "data", tmp_path / "review", tmp_path / "repeat"
    generate_dataset(dataset, sizes={"calibration": 36}, render=True)
    manifest = build_contact_sheets(dataset, out)
    second_manifest = build_contact_sheets(dataset, repeated)
    assert manifest == second_manifest
    assert manifest["human_review_status"] == "PENDING"
    assert manifest["agent_review_status"] == "NOT_REVIEWED"
    assert manifest["source_split"] == "calibration"
    assert manifest["total_images"] == 36
    for sheet in manifest["sheets"]:
        image_path = out / sheet["path"]
        assert Image.open(image_path).size == (2304, 3408)
        assert image_path.read_bytes() == (repeated / sheet["path"]).read_bytes()
        assert sheet["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
        cells = Counter(
            (row["constraint_family"], row["chart_type"], row["operation"])
            for row in sheet["scenes"]
        )
        assert len(cells) == 18 and set(cells.values()) == {1}
        assert all(
            "truth_world" not in row and "observed_world" not in row for row in sheet["scenes"]
        )
    assert (
        len({row["base_scene_id"] for sheet in manifest["sheets"] for row in sheet["scenes"]}) == 36
    )
    assert json.loads((out / "contact_sheets_manifest.json").read_text()) == manifest


def test_selection_is_independent_of_input_order(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sizes={"calibration": 36}, render=False)
    scenes = [json.loads(line) for line in (dataset / "calibration.jsonl").read_text().splitlines()]
    assert select_scenes(scenes) == select_scenes(list(reversed(scenes)))
    with pytest.raises(ValueError, match="two"):
        select_scenes(scenes[:18])
    with pytest.raises(ValueError, match="calibration"):
        select_scenes([{**row, "split": "confirm"} for row in scenes])


def test_contact_sheet_refuses_tampered_source(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sizes={"calibration": 36}, render=True)
    path = dataset / "calibration.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        build_contact_sheets(dataset, tmp_path / "review")


def test_contact_sheet_refuses_unrendered_dataset(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(dataset, sizes={"calibration": 36}, render=False)
    with pytest.raises(ValueError, match="render"):
        build_contact_sheets(dataset, tmp_path / "review")
