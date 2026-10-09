"""Actual CPU pickle/tensor comparisons, independent of the training hash helper."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mm_dev.checkpoint_recount import FIELDS, publish_recount, verify_engine_checkpoints
from mm_dev.common import PLAN_SHA256
from mm_dev.contract import PLAN_ID

REPOSITORY = Path(__file__).resolve().parents[2]
PLAN = REPOSITORY / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"
CLI = REPOSITORY / "scripts/mm_dev/verify_engine_checkpoints.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


class CheckpointRun:
    def __init__(self, root, mode="natural"):
        self.root, self.mode = root, mode
        plan = json.loads(PLAN.read_text())
        questions = root / "data/questions.jsonl"
        questions.parent.mkdir()
        questions.write_text("")
        self.freeze = {
            "plan_id": PLAN_ID,
            "status": "FROZEN",
            "authorized": True,
            "run_root": str(root),
            "plan_sha256": PLAN_SHA256,
            "model_identity": {**plan["model"], "model_files": [{"fixture": "CPU only"}]},
            "resource_policy": plan["resource"],
            "input_hashes": {"data/questions.jsonl": sha(questions)},
            "source_hashes": {
                str(path.relative_to(REPOSITORY)): sha(path)
                for directory in ("src/mm_core", "src/mm_dev", "scripts/mm_dev")
                for path in (REPOSITORY / directory).rglob("*.py")
            },
        }
        put(root / "manifests/F2_FREEZE.json", self.freeze)
        self.run_identity = {
            "run_id": "ENGINE_F2_" + mode.upper(),
            "phase": "ENGINE_F2",
            "stress": mode == "stress",
        }
        self.states = {}
        # Real Adam state with tensor step/exp_avg/exp_avg_sq, without a model or GPU.
        parameter = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        optimizer = torch.optim.AdamW([parameter], lr=1e-5)
        parameter.sum().backward()
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for step in (2, 4):
            state = {
                "plan_id": PLAN_ID,
                "run_identity": self.run_identity,
                "committed_logical_step": step,
                "parameters": {"lora.weight": parameter.detach().clone()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": {
                    "python": random.Random(123).getstate(),
                    "numpy": np.random.RandomState(123).get_state(),
                    "cpu": torch.get_rng_state(),
                    "cuda": [torch.tensor([1, 2, 3], dtype=torch.uint8)],
                    "scalar": np.uint32(12),
                },
                "reference": {"lora.weight": torch.ones(2, 2)},
                "reference_hash": "f" * 64,
                "input_stream_hash": "schedule-fixture",
                "sampling_hash": "sampling-fixture",
                "cursor": {"next_logical_step": step + 1, "next_slot": 0, "next_sample_index": 0},
                "diagnostics_hash": "d" * 64,
                "token_path_hash": "t" * 64,
            }
            for track in ("continuous", "split"):
                self.states[track, step] = copy.deepcopy(state)
                self.save(track, step)
        for track in ("continuous", "split"):
            put(
                self.directory(track) / "RUN_MANIFEST.json",
                {
                    "plan_id": PLAN_ID,
                    "run_identity": self.run_identity,
                    "stream_hash": "schedule-fixture",
                    "sampling_hash": "sampling-fixture",
                },
            )
        self.markers()

    def directory(self, track):
        return self.root / "engineering" / self.mode / track

    def commit_path(self, track, step):
        return self.directory(track) / "checkpoints" / f"commit-{step:02d}.json"

    def checkpoint_path(self, track, step):
        return self.directory(track) / "checkpoints" / f"step-{step:02d}-fixture.pt"

    def save(self, track, step):
        path = self.checkpoint_path(track, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.states[track, step], path)
        # Deliberately identical opaque hashes even for divergent states. The independent
        # comparison must inspect actual values, not accept these hash strings as proof.
        put(
            self.commit_path(track, step),
            {
                "step": step,
                "path": path.name,
                "sha256": sha(path),
                "state_hash": "0" * 64,
                "field_hashes": dict.fromkeys(FIELDS, "0" * 64),
            },
        )

    def markers(self):
        for track, segment, step in (
            ("continuous", "continuous", 4),
            ("split", "first", 2),
            ("split", "resume", 4),
        ):
            put(
                self.directory(track) / (segment.upper() + "_COMPLETE.json"),
                {
                    "mode": self.mode,
                    "segment": segment,
                    "freeze_sha256": sha(self.root / "manifests/F2_FREEZE.json"),
                    "run_identity": self.run_identity,
                    "final_step": step,
                    "checkpoint": json.loads(self.commit_path(track, step).read_text()),
                },
            )


@pytest.fixture
def run(tmp_path):
    return CheckpointRun(tmp_path.resolve())


@pytest.mark.parametrize("mode", ["natural", "stress"])
def test_real_cpu_states_equal_without_any_state_hash_comparison(tmp_path, mode):
    run = CheckpointRun(tmp_path.resolve(), mode)
    with (
        patch("mm_core.training.state_hash", side_effect=AssertionError("Hash equality forbidden")),
        patch.object(torch, "load", wraps=torch.load) as load,
    ):
        result = verify_engine_checkpoints(run.root, mode)
    assert result["status"] == "VERIFIED_ELEMENTWISE_EQUAL"
    assert result["cuda_initialized"] is False
    assert result["mode"] == mode
    assert result["freeze_sha256"] == sha(run.root / "manifests/F2_FREEZE.json")
    assert [row["step"] for row in result["comparisons"]] == [2, 4]
    assert result["total_counts"]["tensors"] > 10
    assert result["total_counts"]["numpy_arrays"] == 2
    assert result["total_counts"]["numpy_scalars"] == 2
    assert result["total_counts"]["scalars"] > 1000
    assert all(set(row["fields"]) == FIELDS for row in result["comparisons"])
    assert len(result["artifact_hashes"]) == 14
    assert load.call_count == 4
    for call in load.call_args_list:
        assert call.kwargs == {"map_location": "cpu", "weights_only": False}
    assert verify_engine_checkpoints(run.root, mode) == result


@pytest.mark.parametrize("step", [2, 4])
@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: s["parameters"]["lora.weight"].add_(1), "tensor mismatch.*parameters"),
        (
            lambda s: s["parameters"].update(
                {"lora.weight": torch.ones(2, 2, dtype=torch.float64)}
            ),
            "tensor mismatch",
        ),
        (lambda s: s["parameters"].update({"lora.weight": torch.ones(4)}), "tensor mismatch"),
        (lambda s: s["optimizer"]["state"][0]["exp_avg"].add_(1), "tensor mismatch.*optimizer"),
        (lambda s: s["optimizer"]["state"][0]["step"].add_(1), "tensor mismatch.*optimizer"),
        (lambda s: s["scheduler"].update(last_epoch=123), "scalar mismatch.*scheduler"),
        (lambda s: s["rng"]["numpy"][1].__setitem__(0, 999), "NumPy array mismatch"),
        (lambda s: s["rng"].update(python=(3, (123,), None)), "container length mismatch.*rng"),
        (lambda s: s["rng"]["cpu"].__setitem__(0, 123), "tensor mismatch.*rng"),
        (lambda s: s["rng"].update(scalar=np.uint32(13)), "NumPy scalar mismatch"),
        (lambda s: s["cursor"].update(next_slot=False), "type mismatch.*cursor"),
        (
            lambda s: s["reference"].update(extra=torch.zeros(1)),
            "container keys mismatch.*reference",
        ),
        (
            lambda s: s["optimizer"]["param_groups"][0].update(betas=[0.9, 0.999]),
            "type mismatch.*optimizer",
        ),
    ],
)
def test_value_dtype_shape_adam_rng_scalar_and_container_tamper(run, step, change, match):
    change(run.states["split", step])
    run.save("split", step)
    run.markers()
    with pytest.raises(ValueError, match=match):
        verify_engine_checkpoints(run.root, "natural")


def test_bad_bytes_are_rejected_before_pickle_load(run):
    with run.checkpoint_path("continuous", 2).open("ab") as handle:
        handle.write(b"changed")
    with (
        patch.object(torch, "load", side_effect=AssertionError("Must not deserialize")),
        pytest.raises(PermissionError, match="committed SHA"),
    ):
        verify_engine_checkpoints(run.root, "natural")


@pytest.mark.parametrize("name", ["../outside.pt", "/tmp/outside.pt", "sub/nested.pt"])
def test_checkpoint_paths_cannot_escape_or_redirect(run, name):
    path = run.commit_path("continuous", 2)
    commit = json.loads(path.read_text())
    commit["path"] = name
    put(path, commit)
    with pytest.raises(PermissionError, match=r"local \.pt"):
        verify_engine_checkpoints(run.root, "natural")


def test_symlink_checkpoint_is_rejected(run):
    path = run.checkpoint_path("continuous", 2)
    alternate = path.with_name("other.pt")
    path.rename(alternate)
    path.symlink_to(alternate.name)
    with pytest.raises(PermissionError, match="Symlink"):
        verify_engine_checkpoints(run.root, "natural")


def test_changed_freeze_and_wrong_segment_binding_are_rejected(run):
    marker = run.directory("split") / "FIRST_COMPLETE.json"
    data = json.loads(marker.read_text())
    data["freeze_sha256"] = "1" * 64
    put(marker, data)
    with pytest.raises(PermissionError, match="frozen run"):
        verify_engine_checkpoints(run.root, "natural")
    run.markers()
    data = json.loads(marker.read_text())
    data["checkpoint"]["sha256"] = "1" * 64
    put(marker, data)
    with pytest.raises(PermissionError, match="checkpoint binding"):
        verify_engine_checkpoints(run.root, "natural")


def test_cuda_initialized_process_cannot_claim_cpu_independence(run):
    with (
        patch.object(torch.cuda, "is_initialized", return_value=True),
        pytest.raises(RuntimeError, match="CUDA uninitialized"),
    ):
        verify_engine_checkpoints(run.root, "natural")


def test_atomic_receipt_is_reusable_only_when_exactly_identical(run):
    result = verify_engine_checkpoints(run.root, "natural")
    path = publish_recount(run.root, result)
    before = path.read_bytes()
    assert publish_recount(run.root, result) == path
    assert path.read_bytes() == before
    changed = copy.deepcopy(result)
    changed["total_counts"]["tensors"] += 1
    with pytest.raises(PermissionError, match="already differs"):
        publish_recount(run.root, changed)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".pending-recount-*"))


def test_standalone_cli_real_frozen_source_precheck_and_repeatable_receipt(run):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [
        sys.executable,
        "-I",
        str(CLI),
        "--plan",
        str(PLAN),
        "--run-root",
        str(run.root),
        "--mode",
        "natural",
    ]
    first = subprocess.run(command, cwd=run.root, env=environment, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    output = run.root / "engineering/natural/INDEPENDENT_CHECKPOINT_RECOUNT.json"
    before = output.read_bytes()
    second = subprocess.run(command, cwd=run.root, env=environment, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == before
    assert json.loads(before)["cuda_initialized"] is False


def test_cli_requires_plan_and_bad_freeze_does_not_publish(run):
    missing = subprocess.run(
        [sys.executable, str(CLI), "--run-root", str(run.root), "--mode", "natural"],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 2 and "--plan" in missing.stderr
    run.freeze["authorized"] = False
    put(run.root / "manifests/F2_FREEZE.json", run.freeze)
    bad = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--plan",
            str(PLAN),
            "--run-root",
            str(run.root),
            "--mode",
            "natural",
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0 and "authorized F2 freeze" in bad.stderr
    assert not (run.root / "engineering/natural/INDEPENDENT_CHECKPOINT_RECOUNT.json").exists()
