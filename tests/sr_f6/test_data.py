import copy
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from sr_f6.data import (
    all_tasks,
    audit_banks,
    composition_correction_report,
    construct_banks,
    make_composition_root,
    natural_interface_tasks,
    public_prompt,
    render_source,
)
from srf6_ref import design


@pytest.fixture(scope="module")
def config():
    return json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())


@pytest.fixture(scope="module")
def banks(config):
    return construct_banks(config)


def test_full_registered_banks_independent_oracles_and_manipulation(banks, config):
    report = audit_banks(banks, config)
    assert report["status"] == "PASS"
    assert report["question_oracle_comparisons"] == 4704
    assert report["cross_split_world_duplicates"] == 0
    assert report["engineered_train_anchors"] == 96
    assert report["training_manipulation"]["status"] == "PASS"
    assert {split: row["questions"] for split, row in report["splits"].items()} == {
        "DEV_CAL": 384,
        "DEV_PILOT": 384,
        "DEV_EVAL": 96,
        "TRAIN": 2304,
        "TEST_ID": 384,
        "TEST_COMPOSITION": 768,
        "TEST_MECHANISM": 192,
        "TEST_CLASSOUT": 192,
    }
    assert report["processor_readability"] == "NOT_RUN"


def test_composition_does_not_call_paired_optimizer(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("COMPOSITION used P1")

    monkeypatch.setattr(design, "paired_views", prohibited)
    root = make_composition_root("BINDING", 0, 2026102716)
    assert set(root["tasks"]) == {"P0"}


def test_original_composition_acceptance_bug_is_reproduced(banks, config):
    report = composition_correction_report(
        banks["TEST_COMPOSITION"], config["data"]["seeds"]["TEST_COMPOSITION"]
    )
    assert report["roots_compared"] == 192
    assert report["changed_root_count"] == 7
    assert all(
        row["corrected_attempt"] < row["original_attempt"] for row in report["changed_roots"]
    )
    assert report["model_outputs_used"] is False


def test_private_labels_cannot_change_prompt(banks):
    task = banks["DEV_CAL"][0]["tasks"]["P0"][0]
    changed = copy.deepcopy(task)
    changed.update(
        gold_answer="DO NOT LEAK",
        gold_selected=["secret"],
        root_id="private",
        factor="private",
        engineered_collision_anchor="private",
    )
    changed["world"]["series"] = {"Alpha": [999] * 6, "Beta": [-999] * 6}
    assert public_prompt(task) == public_prompt(changed)
    assert "DO NOT LEAK" not in public_prompt(changed)


def test_natural_interface_is_paired_and_has_no_json_instruction(banks, config):
    tasks = natural_interface_tasks(banks["TEST_ID"], config["master_seed"])
    assert len(tasks) == len({task["paired_id_qid"] for task in tasks}) == 128
    id_qids = {task["qid"] for task in all_tasks(banks["TEST_ID"])}
    assert all(task["paired_id_qid"] in id_qids for task in tasks)
    for task in tasks:
        prompt = public_prompt(task, natural=True)
        assert "Return exactly one JSON" not in prompt
        assert "bindings" not in prompt
        assert prompt.endswith("Do not output JSON.")


def test_render_has_exact_geometry_and_correct_bars(banks):
    task = banks["DEV_CAL"][0]["tasks"]["P0"][0]
    payload, metrics = render_source(task["world"])
    image = Image.open(io.BytesIO(payload))
    assert image.size == (1280, 896)
    assert image.mode == "RGB"
    assert not image.info
    assert metrics["pixels_per_six_units"] >= 30
    assert metrics["bar_width_px"] == 37
    assert metrics["processor_readability"] == "NOT_RUN"
    colors = {"Alpha": (63, 110, 156), "Beta": (184, 112, 48)}
    for bar in metrics["bars"]:
        x0, y0, x1, y1 = bar["bounding_box"]
        assert image.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == colors[bar["series"]]
        assert abs((755 - y0) - 6.3 * bar["value"]) <= 0.5
    anchor = banks["DEV_CAL"][0]["tasks"]["P1"][0]
    assert render_source(anchor["world"])[0] == payload


def test_audit_rejects_unmatched_or_missing_inputs(banks, config):
    changed = copy.deepcopy(banks)
    changed["TRAIN"][0]["tasks"]["P1"][1]["edit_descriptor"] = ["wrong"]
    with pytest.raises(ValueError, match="edit-cost"):
        audit_banks(changed, config)
    changed = {**banks, "DEV_CAL": banks["DEV_CAL"][:-1]}
    with pytest.raises(ValueError, match="roots"):
        audit_banks(changed, config)


def test_classout_rejects_duplicate_actual_views(banks, config):
    changed = copy.deepcopy(banks)
    views = changed["TEST_CLASSOUT"][0]["tasks"]["P0"]
    qid, variant = views[2]["qid"], views[2]["variant"]
    views[2] = dict(views[1], qid=qid, variant=variant)
    with pytest.raises(ValueError, match="distinct actual worlds"):
        audit_banks(changed, config)


def test_classout_rejects_forged_edit_descriptor(banks, config):
    changed = copy.deepcopy(banks)
    changed["TEST_CLASSOUT"][0]["tasks"]["P0"][1]["edit_descriptor"][0][2] *= -1
    with pytest.raises(ValueError, match="descriptor differs"):
        audit_banks(changed, config)
