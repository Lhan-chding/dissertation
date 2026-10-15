"""Frozen manifests, independent execution, visible images and sidecar isolation."""

import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from sr_f3_ce.data import (
    ModelInputStore,
    bounded_path,
    compare_historical_worlds,
    independently_answer,
    load_frozen_dataset,
    validate_manifest_rows,
    world_key,
)
from sr_f3_ce.prompts import CONTRACT
from sr_f3_ce.rendering import PALETTES, PLOT, fixed_spotchecks, render_chart

PLAN = Path(__file__).resolve().parents[2] / "artifacts/sr_f3_ce_20261016/plan"


@pytest.fixture(scope="module")
def dataset():
    if not PLAN.is_dir():
        pytest.skip("Private frozen user package is not installed in this checkout")
    return load_frozen_dataset(PLAN)


def test_complete_frozen_package_and_independent_answer_audit(dataset):
    assert len(dataset.roots) == 624
    assert len(dataset.tasks) == 4096
    assert len(dataset.runs) == 12
    assert set(dataset.schedules) == {0, 1, 2}
    assert dataset.audit["independent_answer_mismatches"] == []
    assert dataset.audit["cross_root_world_collisions"] == []
    assert dataset.audit["package"]["read_only"]


def test_bad_schedule_is_not_silently_repaired(dataset):
    schedules = copy.deepcopy(dataset.schedules)
    schedules[0][0]["rl_qids"][0] = schedules[0][1]["rl_qids"][0]
    with pytest.raises(ValueError, match="repeated, omitted or cross-pool"):
        validate_manifest_rows(
            dataset.roots, dataset.tasks, dataset.inputs, dataset.runs, schedules
        )


def test_historical_comparison_retains_complete_world_identity(dataset, tmp_path):
    task = next(iter(dataset.tasks.values()))
    source = tmp_path / "old.json"
    old_world = copy.deepcopy(task["world"])
    old_world.pop("domain")
    old_world["unit"] = "count"
    source.write_text(json.dumps([{"world": old_world}]))
    audit = compare_historical_worlds(dataset.tasks, [source])
    assert audit["status"] == "FAIL" and task["qid"] in audit["overlapping_qids"]
    assert compare_historical_worlds(dataset.tasks, [])["status"] == "NOT_CHECKED"
    gamma = copy.deepcopy(task["world"])
    gamma["series"]["Gamma"] = [90] * len(gamma["categories"])
    assert world_key(gamma) == world_key(task["world"])


def test_independent_executor_boundary_ties_and_empty_set():
    world = {
        "categories": ["a", "b"],
        "domain": [10, 90],
        "series": {"Alpha": [50, 50], "Beta": [20, 30]},
    }
    assert (
        independently_answer(
            world,
            {
                "family": "THRESHOLD",
                "threshold": 50,
                "comparator": "gt",
                "aggregate": "mean",
            },
        )
        is None
    )
    assert (
        independently_answer(
            world,
            {
                "family": "THRESHOLD",
                "threshold": 50,
                "comparator": "ge",
                "aggregate": "count",
            },
        )
        == 2
    )
    assert (
        independently_answer(
            world,
            {"family": "TOPK", "k": 1, "direction": "largest", "aggregate": "sum"},
        )
        == 20
    )
    assert (
        independently_answer(
            world,
            {
                "family": "INTERVAL",
                "start": 0,
                "end": 1,
                "include_start": False,
                "aggregate": "mean",
            },
        )
        == 20
    )


@pytest.mark.parametrize("chart", ["grouped_bar", "line"])
@pytest.mark.parametrize("style", ["main", "novel"])
@pytest.mark.parametrize("arity,gamma", [(4, False), (4, True), (6, False)])
def test_render_is_readable_deterministic_and_contains_no_sidecar_metadata(
    chart, style, arity, gamma
):
    world = {
        "categories": ["January", "February", "March", "April", "May", "June"][:arity],
        "domain": [10, 90],
        "series": {
            "Alpha": [10, 90, 31, 64, 39, 72][:arity],
            "Beta": [90, 10, 67, 22, 18, 59][:arity],
        },
    }
    if gamma:
        world["series"]["Gamma"] = [15, 33, 45, 88]
    row = {"world": world, "chart": chart, "render_style": style}
    image_bytes, metrics = render_chart(row)
    polluted = {
        **row,
        "answer": 999999,
        "qid": "SECRET",
        "query": {"aggregate": "mean"},
    }
    assert render_chart(polluted)[0] == image_bytes
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert image.size == (1024, 768)
        assert image.info == {}
    assert metrics["pixels_per_value_unit"] >= 3
    assert metrics["minimum_major_label_ink_height_px"] >= 9
    assert not metrics["numeric_value_labels"]


def test_fixed_spotchecks_cover_queries_templates_boundary_styles_and_arity(dataset):
    checks = fixed_spotchecks(dataset)
    assert len(checks) >= 48
    assert checks == fixed_spotchecks(dataset)
    assert all(check["independent_answer_checked"] for check in checks)
    assert any("empty strict" in rule for check in checks for rule in check["rules"])
    for family in ("CROSS", "THRESHOLD", "TOPK", "INTERVAL"):
        for variant in range(8):
            assert any(
                f"{family}: frozen TRAIN_RL variant {variant}" in item["rules"]
                for item in checks
            )


@pytest.mark.parametrize("style", ["main", "novel"])
def test_coincident_line_values_keep_both_series_visible(style):
    from PIL import ImageColor

    world = {
        "categories": ["January", "February", "March", "April"],
        "domain": [10, 90],
        "series": {"Alpha": [50] * 4, "Beta": [50] * 4},
    }
    data, _ = render_chart({"world": world, "chart": "line", "render_style": style})
    left, top, right, bottom = PLOT
    x, y = round(left + (right - left) / 8), round(bottom - (bottom - top) / 2)
    with Image.open(io.BytesIO(data)) as image:
        patch = image.crop((x - 12, y - 12, x + 13, y + 13))
        pixels = patch.load()
        colors = {pixels[x, y] for x in range(patch.width) for y in range(patch.height)}
    assert ImageColor.getrgb(PALETTES[style]["Alpha"]) in colors
    assert ImageColor.getrgb(PALETTES[style]["Beta"]) in colors


def test_gold_sidecar_mutation_cannot_affect_model_inputs(tmp_path):
    # This store has no handle to sidecars. Changing every answer/world/teacher field,
    # including an audit question, leaves all three model input fields byte-identical.
    qid = "ENGINE_CROSS_0000_line_v0"
    prompt = (
        "For categories where Alpha is strictly greater than Beta, "
        "compute the sum of the Alpha values.\n"
        + CONTRACT
    )
    frozen = {
        qid: {
            "qid": qid,
            "pool": "ENGINE",
            "chart": "line",
            "prompt": prompt,
            "render_style": "main",
        }
    }
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / f"{qid}.png"
    Image.new("RGB", (1024, 768), "white").save(image)
    images = {
        qid: {
            "qid": qid,
            "image_file": f"images/{qid}.png",
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
    }
    store = ModelInputStore(frozen, images, tmp_path)
    before = {
        p: store.model_input(qid, p)
        for p in ("structured", "answer_only", "natural_work")
    }
    sidecar = tmp_path / "TASKS_AUDIT_ONLY.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "qid": qid,
                "answer": "SECRET",
                "world": "SECRET",
                "question": "SECRET",
                "teacher": "SECRET",
            }
        )
        + "\n"
    )
    frozen[qid]["prompt"] = "Caller mutation must not alter store"
    after = {
        p: store.model_input(qid, p)
        for p in ("structured", "answer_only", "natural_work")
    }
    assert before == after
    assert all(
        set(value) == {"prompt", "image_file", "image_sha256"}
        for value in after.values()
    )
    assert all(
        "SECRET" not in value["prompt"] and qid not in value["prompt"]
        for value in after.values()
    )
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="image changed"):
        store.model_input(qid)


def test_model_image_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        bounded_path(tmp_path, "../outside.png")
    with pytest.raises(ValueError):
        bounded_path(tmp_path, "/tmp/outside.png")
