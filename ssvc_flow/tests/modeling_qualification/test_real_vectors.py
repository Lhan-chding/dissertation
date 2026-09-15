"""Actual checkpoint vectors are distinguished from response rank and metadata."""

import hashlib
import json

import numpy as np
import pytest
import torch

from src.core import canonical_hash
from src.modeling_qualification.real_vectors import (
    BANKS,
    extract_real_vectors,
    parameter_geometry,
    prepare_vector_extraction,
    require_cpu_allocation,
)
from src.optimizer_fork import state_hash


def test_streaming_geometry_uses_actual_deltas_and_does_not_raise_rank_for_aliases():
    origin = {"a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), "b": torch.zeros(2)}
    first = {"a": origin["a"] + torch.tensor([1.0, 0.0, 0.0]), "b": torch.zeros(2)}
    second = {"a": origin["a"] + torch.tensor([0.0, 2.0, 0.0]), "b": torch.zeros(2)}
    got = parameter_geometry(origin, [first, second, first], ["joint_0", "joint_1", "alias"])
    assert got["numerical_update_rank"] == 2
    np.testing.assert_allclose(got["gram"], [[1, 0, 1], [0, 4, 0], [1, 0, 1]], atol=1e-14)
    assert got["update_norms"] == [1.0, 2.0, 1.0]
    assert got["cosines"][0][1] == 0
    assert got["cosines"][0][2] == 1
    assert got["delta_hashes"][0] == got["delta_hashes"][2]
    assert got["semantic_response_rank"] is None
    assert got["parameter_count"] == 5


def test_zero_norm_cosine_null_and_near_collinearity_has_stable_qr_rank():
    origin = {"p": torch.zeros(4, dtype=torch.float64)}
    got = parameter_geometry(
        origin,
        [origin, {"p": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)}],
        ["zero", "unit"],
    )
    assert got["cosines"][0] == [None, None]
    assert got["cosine_null_reason"] == "ONE_OR_BOTH_VECTORS_HAVE_ZERO_NORM"
    a = {"p": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)}
    b = {"p": torch.tensor([1.0, 1e-8, 0.0, 0.0], dtype=torch.float64)}
    got = parameter_geometry(origin, [a, b], ["a", "b"])
    assert got["numerical_update_rank"] == 2  # Gram eigenvalues alone lose this direction.


@pytest.mark.parametrize(
    "bad",
    [
        {"q": torch.zeros(2)},
        {"p": torch.zeros(3)},
        {"p": torch.zeros(2, dtype=torch.float64)},
        {"p": torch.tensor([float("nan"), 0.0])},
    ],
)
def test_geometry_refuses_unbound_layout_dtype_and_nonfinite(bad):
    with pytest.raises(ValueError):
        parameter_geometry({"p": torch.zeros(2)}, [bad], ["bad"])


def test_production_cpu_allocation_check(monkeypatch):
    for key in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        require_cpu_allocation()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="GPU"):
        require_cpu_allocation()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    require_cpu_allocation()


def checkpoint_fixture(root):
    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def checkpoint(path, identity, state):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": identity, "state": state, "state_hash": state_hash(state)}, path)
        return sha(path)

    origin = {
        "parameters": {"p": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"step": torch.tensor(64.0)}}},
        "metadata": {"arm": "X_BASE", "checkpoint_step": 64},
    }
    origin_path = root / "legacy/checkpoint.pt"
    origin_sha = checkpoint(origin_path, {"fixture": True}, origin)
    plan = {
        "paths": {"new_run_root": str(root / "followup")},
        "source_files": {"fixture.py": "hash"},
        "r4_binding": {
            "warm_checkpoint": {
                "path": str(origin_path),
                "file_sha256": origin_sha,
                "state_hash": state_hash(origin),
                "identity": {"fixture": True},
            }
        },
    }
    write(root / "plan.json", plan)
    for bank_name in BANKS:
        bank = root / "followup/S1" / bank_name
        identity = {
            "bank_id": bank_name,
            "origin_state_hash": "followup_recaptured_origin",
            "validated_plan_hash": canonical_hash(plan),
            "source_hash": canonical_hash(plan["source_files"]),
        }
        write(bank / "identity.json", identity)
        write(bank / "status.json", {"status": "MEASURED"})
        write(bank / "policy_aliases.json", {})
        entries = []
        for cid, delta in [("joint_0", [1.0, 0.0]), ("joint_1", [0.0, 1.0])]:
            spec = {"id": cid, "policy": "joint"}
            ci = {**identity, "candidate_spec": spec}
            state = {
                "parameters": {"p": origin["parameters"]["p"] + torch.tensor(delta)},
                "optimizer": {"state": {0: {"step": torch.tensor(65.0)}}},
                "optimizer_parameter_names": [["p"]],
            }
            cp = bank / "candidates" / cid / "checkpoint.pt"
            cp_sha = checkpoint(cp, ci, state)
            record = {
                "candidate_id": cid,
                "candidate_spec": spec,
                "checkpoint_identity": ci,
                "checkpoint_sha256": cp_sha,
                "candidate_state_hash": state_hash(state),
                "candidate_parameter_hash": state_hash(state["parameters"]),
                "candidate_optimizer_state_hash": state_hash(state["optimizer"]),
                "audit": {
                    "origin_state_hash": identity["origin_state_hash"],
                    "optimizer_hash_before": state_hash(origin["optimizer"]),
                    "parameter_hash_before": state_hash(
                        {key: state_hash(value) for key, value in origin["parameters"].items()}
                    ),
                    "actual_step_norm": 1.0,
                },
            }
            rp = cp.with_name("candidate.json")
            write(rp, record)
            entries.extend(
                {
                    "path": path.relative_to(bank).as_posix(),
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                }
                for path in [cp, rp]
            )
        write(bank / "manifest.json", {"files": entries})
        write(
            bank / "completed.json",
            {
                "status": "MEASURED",
                "identity_hash": canonical_hash(identity),
                "manifest_sha256": sha(bank / "manifest.json"),
            },
        )
    return root / "plan.json"


def test_full_cpu_checkpoint_byte_state_optimizer_binding_and_readonly(tmp_path):
    plan = checkpoint_fixture(tmp_path / "source")
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in plan.parent.rglob("*") if p.is_file()
    }
    public, _, _, _ = prepare_vector_extraction(plan)
    assert public["checkpoint_file_count"] == 11
    result = extract_real_vectors(plan, tmp_path / "out", require_slurm=False)
    assert result["status"] == "ACTUAL_UPDATE_GEOMETRY_CPU_VERIFIED"
    assert len(result["banks"]) == 5
    for bank in result["banks"]:
        assert bank["actual_origin_relative_updates"]["numerical_update_rank"] == 2
        assert all(
            row["byte_sha256_verified"] and row["full_state_hash_recomputed"]
            for row in bank["checkpoint_verification"]
        )
        assert bank["origin_followup_full_state_hash_recomputed"] is False
    assert result["semantic_response_rank"] is None
    assert str(tmp_path) not in (tmp_path / "out/real_update_geometry.json").read_text()
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before
    with pytest.raises(FileExistsError):
        extract_real_vectors(plan, tmp_path / "out", require_slurm=False)


def test_checkpoint_byte_tampering_rejected_before_restricted_load(tmp_path):
    plan = checkpoint_fixture(tmp_path / "source")
    path = plan.parent / "followup/S1/bank00/candidates/joint_0/checkpoint.pt"
    content = bytearray(path.read_bytes())
    content[100] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        extract_real_vectors(plan, tmp_path / "out", require_slurm=False)
    assert (
        json.loads((tmp_path / "out/failure.json").read_text())["status"]
        == "FAILED_OR_RESOURCE_REVIEW_REQUIRED"
    )
