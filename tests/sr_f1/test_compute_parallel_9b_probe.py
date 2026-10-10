"""CPU boundary checks for the real-model technical probe; no model execution."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "srf1_real_compute_probe", REPO / "scripts/sr_f1/validate_compute_parallel_9b.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def saved_inputs(tmp_path, monkeypatch):
    root = tmp_path / "run"
    raw = root / "engineering/engine_interrupted/preserved/rollouts"
    raw.mkdir(parents=True)
    image = root / "image.png"
    image.write_bytes(b"technical fixture image")
    row = dict(qid="engine-0001", image_file="image.png", text="frozen prompt")
    manifest = root / "manifests/MODEL_INPUTS.jsonl"
    manifest.parent.mkdir()
    # Stop reading immediately once the required ENGINE metadata is found.
    manifest.write_text(json.dumps(row) + "\nINVALID_SEALED_TEST_SENTINEL\n")
    package = tmp_path / "package"
    (package / "manifests").mkdir(parents=True)
    (package / "manifests/MODEL_INPUTS.jsonl").write_bytes(manifest.read_bytes())
    monkeypatch.setattr(probe, "PACKAGE", package)
    for index in range(4):
        record = dict(
            logical_step=1,
            qid=row["qid"],
            generation_status="COMPLETE",
            tokens=[6, 7],
            old_logprobs=[-1.0, -2.0],
            policy_hash="0" * 64,
            image_sha256=probe.file_hash(image),
            input_hash=probe.digest(row),
        )
        record["record_hash"] = probe.digest(record)
        (raw / f"01-00-{index}.json").write_text(json.dumps(record))
    (raw / "01-99-9.json").write_text("SEALED_TEST_MUST_NOT_READ")
    return root, raw


def test_exact_four_engine_inputs_no_fifth_raw_or_test_read(tmp_path, monkeypatch):
    root, raw = saved_inputs(tmp_path, monkeypatch)
    examples, identity = probe.load_examples(root, raw)
    assert len(examples) == len(identity) == 4
    assert all(x["record"]["qid"].startswith("engine-") for x in examples)
    assert all(x["advantage"] == 1.0 for x in examples)
    assert all(len(x["sha256"]) == 64 for x in identity)


def test_changed_raw_or_short_probability_path_fails(tmp_path, monkeypatch):
    root, raw = saved_inputs(tmp_path, monkeypatch)
    path = raw / "01-00-0.json"
    record = json.loads(path.read_bytes())
    record["old_logprobs"] = [-1.0]
    path.write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="hash differs"):
        probe.load_examples(root, raw)
    record["record_hash"] = probe.digest({k: v for k, v in record.items() if k != "record_hash"})
    path.write_text(json.dumps(record))
    with pytest.raises(PermissionError, match="probability path shortened"):
        probe.load_examples(root, raw)


def test_symlink_raw_or_outside_root_rejected(tmp_path, monkeypatch):
    root, raw = saved_inputs(tmp_path, monkeypatch)
    alias = root / "raw_alias"
    alias.symlink_to(raw, target_is_directory=True)
    with pytest.raises(PermissionError, match="canonical"):
        probe.load_examples(root, alias)
    with pytest.raises(PermissionError, match="inside run root"):
        probe.load_examples(root, tmp_path)


def test_private_spill_exclusive_parent_and_verified_child(tmp_path):
    quota, output = tmp_path / "quota", tmp_path / "PROBE_NEW"
    directory = probe.private_spill_directory(quota, output, 0)
    assert directory.stat().st_mode & 0o777 == 0o700
    assert probe.private_spill_directory(quota, output, 1) == directory
    with pytest.raises(FileExistsError):
        probe.private_spill_directory(quota, output, 0)
    directory.chmod(0o755)
    with pytest.raises(PermissionError, match="private"):
        probe.private_spill_directory(quota, output, 1)


def test_long_probe_requires_nonzero_finite_both_gradients():
    valid = dict(token_count=768, policy_gradients=[torch.ones(2)], total_gradients=[torch.ones(2)])
    assert len(probe.verify_long_gradients([valid] * 4)) == 4
    bad = {**valid, "policy_gradients": [torch.zeros(2)]}
    with pytest.raises(PermissionError, match="zero gradients"):
        probe.verify_long_gradients([bad] * 4)
    bad = {**valid, "total_gradients": [torch.full((2,), float("nan"))]}
    with pytest.raises(PermissionError, match="nonfinite"):
        probe.verify_long_gradients([bad] * 4)


def test_missing_slurm_writes_failure_never_accepts_engine(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    output = root / "technical_incidents/PROBE_FAIL"
    original = probe.file_hash
    monkeypatch.setattr(
        probe,
        "file_hash",
        lambda path: (
            probe.FREEZE_SHA256 if Path(path).name == "EXECUTION_FREEZE.json" else original(path)
        ),
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--plan",
            str(root / "plan.json"),
            "--run-root",
            str(root),
            "--raw-directory",
            str(root / "raw"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(PermissionError, match="Slurm"):
        probe.main()
    receipt = json.loads((output / "PROBE_9B.json").read_bytes())
    assert receipt["status"] == "FAIL"
    assert receipt["full_engine_accepted"] is False
    assert receipt["scientific_training_started"] is False
    assert len(receipt["source_hashes"]) == 5
