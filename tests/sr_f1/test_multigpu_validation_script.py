"""Receipt safety and synthetic model checks, separate from real CUDA acceptance."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "srf1_multigpu_validation_script", REPO / "scripts/sr_f1/validate_multigpu.py"
)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_unavailable_cuda_records_failure_exclusively_without_success(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "validation.json"
    arguments = ["--output", str(output), "--spill-directory", str(tmp_path / "synthetic-scratch")]
    assert validation.main(arguments) == 1
    before = output.read_bytes()
    report = json.loads(before)
    assert report["status"] == "FAILED"
    assert report["real_9b_engine_qualified"] is False
    assert report["scientific_data_read"] is False
    assert report["error_type"] == "AssertionError"
    assert "CUDA is unavailable" in report["error"]
    assert report["started_utc"] and report["finished_utc"]
    with pytest.raises(FileExistsError, match="preserve"):
        validation.main(arguments)
    assert output.read_bytes() == before


def test_existing_scratch_is_not_adopted_or_cleaned(tmp_path):
    scratch = tmp_path / "existing"
    scratch.mkdir()
    sentinel = scratch / "keep.bin"
    sentinel.write_bytes(b"pre-existing experiment evidence")
    with pytest.raises(FileExistsError):
        validation.main(
            ["--output", str(tmp_path / "receipt.json"), "--spill-directory", str(scratch)]
        )
    assert sentinel.read_bytes() == b"pre-existing experiment evidence"
    assert not (tmp_path / "receipt.json").exists()


def test_symlink_scratch_parent_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(PermissionError, match="canonical"):
        validation.main(
            [
                "--output",
                str(tmp_path / "receipt.json"),
                "--spill-directory",
                str(alias / "scratch"),
            ]
        )
    assert not (actual / "scratch").exists()


def test_receipt_retains_completed_cases_if_later_check_fails(tmp_path, monkeypatch):
    def interrupted(directory, expected_gpus, report):
        report["cases"] = [{"case": "earlier_case", "logprobs_exact": True}]
        raise RuntimeError("later CUDA fixture failed")

    monkeypatch.setattr(validation, "run_validation", interrupted)
    output = tmp_path / "failed.json"
    assert (
        validation.main(["--output", str(output), "--spill-directory", str(tmp_path / "new")]) == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "FAILED"
    assert report["cases"] == [{"case": "earlier_case", "logprobs_exact": True}]
    assert report["error"] == "later CUDA fixture failed"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_script_fixture_and_dual_backward_helpers_on_cpu(tmp_path, dtype):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with validation.tiny_runtime(tmp_path, dtype, device="cpu") as (runtime, prepared):
            runtime.activation_cpu_budget_bytes = 1024
            baseline = validation.measure(runtime, prepared, offloaded=False)
            copied = validation.measure(runtime, prepared, offloaded=True)
            validation.compare_measurements(baseline, copied)
            assert copied["offload"]["spill_file_cleaned"]
            assert copied["offload"]["disk_read_bytes"] > copied["offload"]["disk_storage_bytes"]
            norms = validation.verify_history(runtime, prepared)
            assert all(value > 0 for value in norms.values())
    finally:
        torch.set_num_threads(previous_threads)
    assert not list(tmp_path.rglob("*.bin"))


def test_exact_comparison_checks_signed_zero_bits():
    assert torch.equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not validation.bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert validation.bitwise_equal(torch.tensor([0.0]), torch.tensor([0.0]))
