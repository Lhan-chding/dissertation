"""CPU contract tests independent of any model or processor."""

import json
import re
from pathlib import Path

import pytest

from mm_core.generator import (
    OPERATIONS,
    PANELS,
    _fingerprint,
    _numeric_source,
    audit_split_rows,
    decimal_difficulty,
    generate_dataset,
    make_prompt,
    validate_dataset,
)


@pytest.mark.parametrize("difficulty", ["low", "high"])
def test_joint_carry_borrow_and_determinism(difficulty):
    first, receipt = _numeric_source(20261008, "test-family", difficulty, 10000)
    second, repeated = _numeric_source(20261008, "test-family", difficulty, 10000)
    assert first == second and receipt == repeated
    assert first is not None
    assert 1 <= receipt["attempts"] <= receipt["candidate_limit"]
    values = [first["values"][0][i] for i in (0, 2, 4)]
    assert values[0] > values[1]
    for operation in OPERATIONS:
        assert (
            decimal_difficulty(operation, values[: 3 if operation == "range" else 2]) == difficulty
        )


def test_finite_candidate_exhaustion_is_explicit():
    source, receipt = _numeric_source(20261008, "test", "high", 0)
    assert source is None
    assert receipt["status"] == "GENERATOR_CELL_UNAVAILABLE"
    assert receipt["attempts"] == 0


@pytest.mark.parametrize("operation", OPERATIONS)
def test_prompt_contains_no_numeric_sidecar_or_difficulty(operation):
    prompt = make_prompt(operation)
    assert not re.search(r"\d", prompt)
    assert "true_values" not in prompt and "difficulty" not in prompt
    assert prompt.index('"readings"') < prompt.index('"answer"')
    assert "series Amber at Cedar; series Amber at Maple" in prompt
    assert ("series Amber at Pine" in prompt) == (operation == "range")


@pytest.mark.parametrize(
    "field",
    [
        "root_family_id",
        "source_graph_hash",
        "visible_source_hash",
        "image_hash",
        "processed_image_hash",
    ],
)
def test_cross_split_leakage_rejected(field):
    rows = [{"split": "FORMAT_TUNE", field: "same"}, {"split": "FORMAT_CHECK", field: "same"}]
    result = audit_split_rows(rows)
    assert result["status"] == "FAIL"
    assert result["findings"][0]["field"] == field


def test_small_vector_coincidence_is_not_source_leakage():
    rows = [
        {"split": "FORMAT_TUNE", "true_values": [45, 22]},
        {"split": "FORMAT_CHECK", "true_values": [45, 22]},
    ]
    assert audit_split_rows(rows)["findings"] == []


def test_full_source_hash_ignores_family_labels_but_detects_nontarget_change():
    source, _ = _numeric_source(20261008, "test", "low", 10000)
    modified = json.loads(json.dumps(source))
    modified["values"][3][5] += 1
    assert _fingerprint(source) != _fingerprint(modified)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("mm_core_generator")
    return root, generate_dataset(root)


def test_registered_panel_sizes_and_processor_boundary(generated):
    _, report = generated
    assert report["questions"] == 864
    assert report["images"] == 288
    assert report["root_families"] == 36
    assert report["panel_question_counts"] == {split: count * 24 for split, count in PANELS.items()}
    assert report["original_geometry_status"] == "PASS"
    assert report["original_size"] == [896, 672]
    assert report["processed_size"] is None
    assert report["status"] == "READABILITY_REVIEW_REQUIRED"
    assert report["scientific_measurement_ready"] is False
    assert report["validation"]["status"] == "PASS"
    assert len(report["full_resolution_sample_paths"]) == 8


def test_saved_rows_source_matching_and_input_truth_separation(generated):
    root, _ = generated
    assert validate_dataset(root)["status"] == "PASS"
    rows = [json.loads(line) for line in (root / "data/questions.jsonl").read_text().splitlines()]
    inputs = [
        json.loads(line) for line in (root / "data/model_inputs.jsonl").read_text().splitlines()
    ]
    assert set(inputs[0]) == {
        "split",
        "question_id",
        "image_id",
        "image_path",
        "image_sha256",
        "prompt",
        "prompt_sha256",
    }
    assert all(not Path(row["image_path"]).is_absolute() for row in rows)
    assert all(row["minimum_raster_grid_gap_pixels"] >= 4 for row in rows)
    assert all(row["exact_value_data_labels"] is False for row in rows)
    assert len({row["root_family_id"] for row in rows}) == 36


def test_existing_dataset_is_never_overwritten(generated):
    root, _ = generated
    before = (root / "data/questions.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        generate_dataset(root)
    assert (root / "data/questions.jsonl").read_bytes() == before


def test_negative_and_boolean_operands_rejected():
    with pytest.raises(ValueError):
        decimal_difficulty("difference", [22, 45])
    with pytest.raises(ValueError):
        decimal_difficulty("sum", [True, 22])
