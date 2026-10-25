"""CPU-only guards for a bounded saved-token diagnostic; no model load."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "diagnose_probability", Path(__file__).parents[2] / "scripts/sr_f5/diagnose_probability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group():
    records = [
        dict(
            row_id=f"G3-b1-SHUF/007/06/{i}",
            adapter_identity=dict(step=6, run_id="G3-b1-SHUF"),
            tokens=[1, 2],
            raw_tokens=[1, 2],
            sampler_logprobs=[-1.0, -2.0],
            generation_status="COMPLETE",
            technical_validation_errors=[],
        )
        for i in range(8)
    ]
    return dict(
        spec=dict(identity=dict(run_id="G3-b1-SHUF"), step=7, assignment=dict(group_index=6)),
        records=records,
    )


def test_select_complete_group_preserves_tokens(diagnostic):
    saved = group()
    before = copy.deepcopy(saved)
    assert diagnostic.select_records(saved) is saved["records"]
    assert saved == before


@pytest.mark.parametrize("change", ["partial", "policy", "token", "order"])
def test_changed_or_incomplete_identity_rejected(diagnostic, change):
    saved = group()
    if change == "partial":
        saved["records"].pop()
    elif change == "policy":
        saved["records"][0]["adapter_identity"]["step"] = 5
    elif change == "token":
        saved["records"][0]["raw_tokens"][0] = 8
    else:
        saved["records"].reverse()
    with pytest.raises(PermissionError):
        diagnostic.select_records(saved)


def test_prompt_identity_includes_mask_grid_and_processor(diagnostic):
    prepared = {
        "inputs": {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "image_grid_thw": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones(1, 3),
        },
        "routing": {"input_tensor_hash": "pixel-and-token-hash"},
    }
    row = {
        "native_prompt_inputs": {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "image_grid_thw": [[1, 2, 3]],
        },
        "prompt_tensor_hash": "pixel-and-token-hash",
    }
    assert diagnostic.verify_prompt(prepared, row)["processor_tensor_hash_equal"]
    row["native_prompt_inputs"]["attention_mask"][0][1] = 0
    with pytest.raises(PermissionError):
        diagnostic.verify_prompt(prepared, row)


def test_difference_preserves_tail(diagnostic):
    summary = diagnostic.differences([0.0, -6.0, 0.0], [0.0, 0.0, 0.0])
    assert summary == dict(
        mean_absolute_difference=2.0, max_absolute_difference=6.0, max_position=1
    )
    with pytest.raises(ValueError):
        diagnostic.differences([1.0], [1.0, 2.0])


def test_output_cannot_rewrite_science_or_existing_attempt(diagnostic, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "OWNER", tmp_path)
    root = tmp_path / "run"
    (root / "CONTINUATION/diagnostic").mkdir(parents=True)
    (root / "G3").mkdir()
    output = root / "CONTINUATION/diagnostic/bf16.json"
    assert diagnostic.output_path(root, output) == root
    with pytest.raises(PermissionError):
        diagnostic.output_path(root, root / "G3/result.json")
    output.with_suffix(".started.json").write_text("{}")
    with pytest.raises(FileExistsError):
        diagnostic.output_path(root, output)
