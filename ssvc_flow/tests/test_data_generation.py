import hashlib
import json
from collections import Counter

import pytest
from PIL import Image

from src.constraint_solver import solve
from src.generate_worlds import DEFAULT_SIZES, generate_dataset
from src.prompts import SYSTEM_PROMPT, build_prompt, cue_text
from src.render_charts import render_chart


def read_scenes(out, split):
    return [json.loads(line) for line in (out / f"{split}.jsonl").read_text().splitlines()]


def test_default_split_contract():
    assert DEFAULT_SIZES == {
        "calibration": 144,
        "train": 576,
        "control": 144,
        "dev": 144,
        "confirm": 288,
        "ood": 288,
        "natural_pool": 2048,
    }


def test_balanced_unique_deterministic_and_unleaked_generation(tmp_path):
    sizes = {
        "calibration": 18,
        "train": 36,
        "control": 18,
        "dev": 18,
        "confirm": 18,
        "ood": 18,
        "natural_pool": 18,
    }
    out = tmp_path / "first"
    manifest = generate_dataset(out, seed=17, sizes=sizes, render=False)
    other = tmp_path / "second"
    generate_dataset(other, seed=17, sizes=sizes, render=False)
    hashes = set()
    for split, count in sizes.items():
        scenes = read_scenes(out, split)
        assert len(scenes) == count
        assert (out / f"{split}.jsonl").read_bytes() == (other / f"{split}.jsonl").read_bytes()
        for scene in scenes:
            assert scene["truth_structure_hash"] not in hashes
            hashes.add(scene["truth_structure_hash"])
            assert solve(scene["observed_world"], scene["cue"]) == [scene["truth_world"]]
            assert (
                sum(
                    a != b
                    for a, b in zip(scene["truth_world"], scene["observed_world"], strict=True)
                )
                == 1
            )
            assert scene["solution_count"] == 1
            assert len(scene["base_scene_id"]) > 10
            assert all(
                str(value) not in scene["image_path"].split("/")[-1].split("_")
                for value in scene["truth_world"]
            )
        if split in {"calibration", "train", "control", "dev", "confirm"}:
            cells = Counter(
                (r["constraint_family"], r["chart_type"], r["operation"]) for r in scenes
            )
            assert len(cells) == 18 and len(set(cells.values())) == 1
            assert all(max(r["truth_world"]) <= 49 for r in scenes)
        if split == "train":
            assert Counter(r["interface"] for r in scenes) == {
                "SYMBOLIC_FRESH": 18,
                "IMAGE_CUE_FRESH": 18,
            }
            assert (
                len(
                    set(
                        Counter(
                            (
                                r["constraint_family"],
                                r["chart_type"],
                                r["operation"],
                                r["interface"],
                            )
                            for r in scenes
                        ).values()
                    )
                )
                == 1
            )
    ood = read_scenes(out, "ood")
    assert Counter(r["ood_subtype"] for r in ood) == {
        "graph_structure": 6,
        "numeric_shift": 6,
        "render_expression": 6,
    }
    assert all(
        r["constraint_family"] == "cross_series" and r["graph_topology"] in {"path", "cycle"}
        for r in ood
        if r["ood_subtype"] == "graph_structure"
    )
    assert all(min(r["truth_world"]) >= 50 for r in ood if r["ood_subtype"] == "numeric_shift")
    assert manifest["confirm_evaluated"] is False
    assert (out / "schema.json").is_file()
    assert (
        manifest["files"]["train.jsonl"]["sha256"]
        == hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "sizes", [{"train": 18}, {"calibration": 19}, {"natural_pool": 2049}, {"ood": 17}, {"bad": 1}]
)
def test_invalid_split_sizes_rejected(tmp_path, sizes):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, sizes=sizes, render=False)


def test_fixed_prompt_templates_ignore_truth_fields():
    scene = {
        "observed_world": [12, 2, 3, 4],
        "cue": {"family": "duplicate_encoding", "known_index": 0, "known_value": 1},
        "operation": "sum4",
        "image_path": "images/opaque.png",
        "truth_world": [1, 2, 3, 4],
        "executor_answer": 10,
    }
    symbolic = build_prompt(scene, "SYMBOLIC_FRESH")
    visual = build_prompt(scene, "IMAGE_CUE_FRESH")
    assert symbolic["system"] == SYSTEM_PROMPT
    assert "image_path" not in symbolic
    assert visual["image_path"] == "images/opaque.png"
    assert visual["user"] == "The attached chart also shows the true record.\n" + symbolic["user"]
    assert "[1,2,3,4]" not in symbolic["user"]
    changed = dict(scene, truth_world=[90, 91, 92, 93], executor_answer=366)
    assert build_prompt(changed, "SYMBOLIC_FRESH") == symbolic
    assert (
        cue_text({"family": "cross_series", "edges": [[2, 3, 7], [0, 1, 3]]})
        == "a + b = 3\nc + d = 7"
    )
    assert cue_text({"family": "trend"}) == "b - a = c - b = d - c"
    with pytest.raises(ValueError):
        build_prompt(scene, "I4")


@pytest.mark.parametrize("chart_type", ["grouped_bar", "line"])
def test_render_is_deterministic_and_preserves_mapping(tmp_path, chart_type):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    metadata = render_chart([11, 22, 33, 44], chart_type, first)
    render_chart([11, 22, 33, 44], chart_type, second)
    assert first.read_bytes() == second.read_bytes()
    assert Image.open(first).size == (768, 512)
    assert metadata["series_values"] == [[11, 33], [22, 44]]
    assert metadata["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()


def test_renderer_rejects_invalid_chart(tmp_path):
    with pytest.raises(ValueError):
        render_chart([1, 2, 3, 4], "pie", tmp_path / "bad.png")
    with pytest.raises(ValueError):
        render_chart([1, 2, 3, 4], "line", tmp_path / "bad.png", "low_resolution")
    with pytest.raises(ValueError):
        cue_text({"family": "unknown"})


def test_rendered_dataset_records_hashes_and_diagnostics(tmp_path):
    manifest = generate_dataset(tmp_path, sizes={"calibration": 18}, render=True)
    first = read_scenes(tmp_path, "calibration")[0]
    assert (
        first["image_hash"]
        == hashlib.sha256((tmp_path / first["image_path"]).read_bytes()).hexdigest()
    )
    assert first["image_width"] == 768 and first["image_height"] == 512
    assert first["image_token_count"] is None
    assert manifest["human_spot_check"]["required_images"] == 36
    assert manifest["human_spot_check"]["status"] == "PENDING"
    diagnostics = read_scenes(tmp_path, "diagnostic")
    assert {r["identifiability"] for r in diagnostics} == {
        "no_solution",
        "multiple_solutions",
        "no_cue",
    }


def test_default_generator_capacity_and_complete_ood_composition(tmp_path):
    manifest = generate_dataset(tmp_path, render=False)
    assert manifest["split_sizes"] == DEFAULT_SIZES
    assert manifest["global_truth_uniqueness"] is True
    assert manifest["unique_main_solutions"] == 3632
    natural = read_scenes(tmp_path, "natural_pool")
    assert Counter(r["constraint_family"] for r in natural) == {
        "duplicate_encoding": 880,
        "cross_series": 880,
        "trend": 288,
    }
    ood = read_scenes(tmp_path, "ood")
    for subtype in ("numeric_shift", "render_expression"):
        assert (
            len(
                {
                    (r["constraint_family"], r["chart_type"], r["operation"])
                    for r in ood
                    if r["ood_subtype"] == subtype
                }
            )
            == 18
        )
    assert Counter(r["graph_topology"] for r in ood if r["ood_subtype"] == "graph_structure") == {
        "path": 48,
        "cycle": 48,
    }
    with pytest.raises(FileExistsError):
        generate_dataset(tmp_path, render=False)
