"""Byte identity and full-state regressions for the model-free J23 legacy audit."""

from __future__ import annotations

import copy
import json
import random
import sqlite3
from collections import Counter

import numpy as np
import pytest
import torch

from src.exposure_position import legacy
from src.optimizer_fork import state_hash


def _student(step=64):
    arm = "A_LOCAL_C1"
    schedule = [
        {
            "arm": arm,
            "update": update,
            "slot": slot,
            "task_id": f"task-{slot}-{update % 32}",
            "role": "common" if slot < 11 else "donor" if slot == 11 else "replay",
        }
        for update in range(1, 257)
        for slot in range(16)
    ]
    parameters = {
        f"model.language_model.layers.{layer}.mlp.{leaf}.lora_{side}.default.weight": torch.ones(
            (8, 1) if side == "A" else (1, 8), dtype=torch.float32
        )
        for layer in range(32)
        for leaf in ("gate_proj", "up_proj", "down_proj")
        for side in ("A", "B")
    }
    names = sorted(parameters)
    identity = {
        "experiment_id": "SER_J2_20261007",
        "parent": "S96",
        "block": 0,
        "arm": arm,
        "seed": 108701,
    }
    state = {
        "schema": "ser-j2-student-state-v1",
        "identity": identity,
        "parameters": parameters,
        "parameter_names": names,
        "optimizer": {
            "state": {
                i: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros_like(parameters[name]),
                    "exp_avg_sq": torch.zeros_like(parameters[name]),
                }
                for i, name in enumerate(names)
            }
            if step
            else {},
            "param_groups": [
                {
                    "params": list(range(192)),
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0,
                    "lr": 1e-5 * (min(step / 8, 1) if step else 1),
                }
            ],
        },
        "rng": {
            "python": random.Random(17).getstate(),
            "numpy": ("MT19937", np.random.RandomState(17).get_state()[1].tolist(), 624, 0, 0.0),
            "torch": torch.Generator(device="cpu").manual_seed(17).get_state(),
            "cuda": [torch.tensor([2], dtype=torch.uint8)],
        },
        "sampler": {
            "schema": "SER-J2-explicit-cursor-v1",
            "schedule_id": legacy.digest(schedule),
            "arm": arm,
            "committed_step": step,
        },
        "scheduler": {"kind": "linear8_constant", "completed_updates": step},
        "step": step,
        "seed": 108701,
        "exposures": dict(Counter(row["task_id"] for row in schedule if row["update"] <= step)),
        "role_exposures": {"common": 11 * step, "donor": step, "replay": 4 * step},
        "buffers": {"buffer": torch.zeros(1)},
        "module_modes": {"": True},
        "position_state": {},
        "adapter_position_state": {},
        "metadata": {},
    }
    return state, identity, schedule


def _save(path, state):
    checksum = state_hash(state)
    torch.save({"state": state, "state_hash": checksum}, path)
    return {
        "path": str(path),
        "step": state["step"],
        "sha256": legacy.file_digest(path),
        "state_hash": checksum,
    }


@pytest.mark.parametrize("step", (0, 64, 128, 192, 256))
def test_complete_student_bytes_and_exact_prefix_verified(tmp_path, step):
    state, identity, schedule = _student(step)
    path = tmp_path / f"step{step:03d}.pt"
    receipt = _save(path, state)
    before = path.read_bytes()
    result = legacy._verify_student_checkpoint(
        path, receipt, identity, schedule, legacy._tensor_inventory(state)
    )
    assert result["status"] == "VERIFIED_FULL_CPU_STATE"
    assert result["state_hash"] == receipt["state_hash"]
    assert result["source_identity"] == identity
    assert path.read_bytes() == before


def test_rewritten_checkpoint_cannot_replace_registered_file_bytes(tmp_path):
    state, identity, schedule = _student()
    path = tmp_path / "step064.pt"
    receipt = _save(path, state)
    state["parameters"][state["parameter_names"][0]].add_(1)
    _save(path, state)
    with pytest.raises(ValueError, match="file SHA256"):
        legacy._verify_student_checkpoint(
            path, receipt, identity, schedule, legacy._tensor_inventory(state)
        )


def test_recomputed_file_hash_does_not_bypass_retained_full_state_hash(tmp_path):
    state, _, _ = _student()
    path = tmp_path / "state.pt"
    receipt = _save(path, state)
    state["buffers"]["buffer"].add_(1)
    replacement = _save(path, state)
    receipt["sha256"] = replacement["sha256"]
    with pytest.raises(ValueError, match="Full CPU checkpoint state hash"):
        legacy._checkpoint_payload(path, receipt)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda s: s["identity"].update(experiment_id="SER_J23_20261008"), "identity"),
        (lambda s: s["sampler"].update(committed_step=63), "sampler"),
        (lambda s: s["optimizer"].update(state={}), "AdamW state"),
        (lambda s: s["optimizer"]["state"][0].update(step=torch.tensor(63.0)), "Adam step"),
        (lambda s: s["scheduler"].update(completed_updates=63), "scheduler"),
        (lambda s: s["role_exposures"].update(donor=63), "role exposures"),
        (lambda s: s["rng"].pop("cuda"), "RNG"),
        (lambda s: s["module_modes"].update({"": "train"}), "module modes"),
        (lambda s: s["parameters"].pop(s["parameter_names"][0]), "192"),
        (
            lambda s: s["parameters"].update(
                {s["parameter_names"][0]: s["parameters"][s["parameter_names"][0]].half()}
            ),
            "dtype",
        ),
    ],
)
def test_corrupt_state_is_rejected_even_if_attacker_rehashes_it(tmp_path, mutation, match):
    state, identity, schedule = _student()
    identity = copy.deepcopy(identity)
    inventory = legacy._tensor_inventory(state)
    mutation(state)
    path = tmp_path / "step064.pt"
    receipt = _save(path, state)
    with pytest.raises(ValueError, match=match):
        legacy._verify_student_checkpoint(path, receipt, identity, schedule, inventory)


def test_equal_total_but_wrong_per_task_exposures_are_rejected(tmp_path):
    state, identity, schedule = _student()
    inventory = legacy._tensor_inventory(state)
    first, second = list(state["exposures"])[:2]
    state["exposures"][first] += 1
    state["exposures"][second] -= 1
    assert sum(state["exposures"].values()) == 16 * 64
    path = tmp_path / "step064.pt"
    with pytest.raises(ValueError, match="per-task exposures"):
        legacy._verify_student_checkpoint(path, _save(path, state), identity, schedule, inventory)


@pytest.mark.parametrize(
    "errors,verify_weights,expected",
    [
        ([], True, "REUSE_CANDIDATE"),
        ([], False, "WEIGHTS_NOT_VERIFIED"),
        ([{"category": "reuse_checkpoint"}], True, "RERUN_CANDIDATE"),
        ([{"category": "scientific_contract"}], True, "BLOCKED"),
        ([{"category": "scientific_contract"}], False, "BLOCKED"),
        ([{"category": "diagnostic_checkpoint"}], True, "REUSE_CANDIDATE"),
    ],
)
def test_no_missing_parent_or_unread_weights_can_be_promoted(errors, verify_weights, expected):
    assert legacy._status(errors, verify_weights) == expected


def test_absent_legacy_run_returns_all_errors_and_durable_blocked_receipts(tmp_path, monkeypatch):
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"legacy_input_sha256": legacy.REGISTERED_INPUT_SHA256}))
    run = tmp_path / "legacy"
    run.mkdir()
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU audit initialized CUDA"))
    result = legacy.audit_legacy(design, run, tmp_path / "audit")
    assert result["status"] == "BLOCKED"
    assert result["execution_mode_frozen"] is False
    assert result["model_calls"] == result["optimizer_updates"] == 0
    assert result["full_A2_B2_checkpoints_verified"] == 0
    assert all(key in result["checks"] for key in legacy.REGISTERED_INPUT_SHA256)
    assert len(result["errors"]) >= 8
    assert list(run.iterdir()) == []
    assert json.loads((tmp_path / "audit/REUSE_AUDIT.json").read_text()) == result
    bindings = json.loads((tmp_path / "audit/PARENT_AND_ENDPOINT_BINDINGS.json").read_text())
    assert bindings["parents"] == {}
    assert bindings["checkpoint_lookup"] == {}


def test_audit_output_cannot_modify_source_run(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        legacy.audit_legacy(
            tmp_path / "design.json", tmp_path / "legacy", tmp_path / "legacy/audit"
        )
    assert not (tmp_path / "legacy").exists()


def _json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": legacy.file_digest(path)}


def _parent(tmp_path):
    from src.model_adapters.base import pure_generation_options

    model = {"id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
    generation = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "max_new_tokens": 64}
    inherited = {
        "qwen": {
            **model,
            "generation": {**generation, "thinking": False},
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0,
                "target_leaf_modules": ["gate_proj", "up_proj", "down_proj"],
                "expected_modules": 96,
                "freeze_vision": True,
                "freeze_base": True,
            },
        }
    }
    prompt_protocol = {
        "execution": {"frozen_inference_only": True, "new_training_allowed": False},
        "model": {
            **model,
            "base_dtype": "bfloat16",
            "primary_engine": "historical uncached-prefix-recompute",
        },
        "generation": {
            **generation,
            "generation_config": pure_generation_options(64),
            "enable_thinking": False,
            "new_likelihood_scoring": False,
        },
    }
    state, _, _ = _student(96)
    state.update(
        schema="ssvc-v4-complete-trainable-state-1",
        scheduler=None,
        sampler={"kind": "mapping", "state": {}},
        optimizer_parameter_names=[state["parameter_names"]],
        frozen_parameter_hash="f" * 64,
        metadata={
            "lineage_id": 61001,
            "source_recipe": "R0",
            "checkpoint_step": 96,
            "recipe": "R0",
        },
    )
    model_audit = {
        "lora_modules": [f"module{i}" for i in range(96)],
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "frozen_parameter_hash": "f" * 64,
    }
    parent = {
        "snapshot": {
            "path": str(tmp_path / "snapshot"),
            "revision": model["revision"],
            "files": {},
        },
        "config": {"model": model, "generation_proposed_N": generation},
        "gate": {
            "certificate": {
                "status": "PASS",
                "selected_path": "uncached_prefix_recompute",
                "model_audit": model_audit,
            }
        },
    }
    parent_binding = _json_file(tmp_path / "parent.json", parent)
    private = {
        "bindings": {"parent_validated_plan": parent_binding},
        "inherited_config": inherited,
        "config_hash": "a" * 64,
    }
    runtime = {
        "execution_kind": "REAL_CUDA_MODEL",
        "model_hash": legacy.digest(parent["snapshot"]),
        "config_hash": legacy.digest(inherited),
    }
    manifest = {
        "kind": "PROSPECTIVE_TRAINING",
        "fixture": False,
        "steps": 96,
        "recipe": "R0",
        "identity": {"lineage_id": 61001, "role": "source", "source_recipe": "R0"},
        "runtime_identity": runtime,
        "config_hash": private["config_hash"],
    }
    manifest_binding = _json_file(tmp_path / "MANIFEST.json", manifest)
    identity = {"manifest_hash": legacy.digest(manifest), "step": 96}
    path = tmp_path / "state.pt"
    torch.save({"identity": identity, "state": state, "state_hash": state_hash(state)}, path)
    checkpoint = {
        "path": str(path),
        "sha256": legacy.file_digest(path),
        "identity": identity,
        "state_hash": state_hash(state),
    }
    commit = {"stop": 96, "manifest_hash": legacy.digest(manifest), "checkpoint": checkpoint}
    record = {
        "checkpoint_id": "S96",
        "status": "AVAILABLE",
        "source": {"id": "S96", "kind": "source", "lineage": 61001, "source_step": 96},
        "checkpoint": checkpoint,
        "checkpoint_original_binding": copy.deepcopy(checkpoint),
        "manifest": manifest_binding,
        "commit": _json_file(tmp_path / "COMMIT.json", commit),
        "runtime_identity": runtime,
    }
    return record, private, prompt_protocol


def test_original_parent_commit_manifest_and_full_cpu_state_are_verified(tmp_path, monkeypatch):
    record, private, protocol = _parent(tmp_path)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU audit initialized CUDA"))
    binding, inventory, _ = legacy._audit_parent(record, private, protocol, verify_weights=True)
    assert binding["checkpoint_full_cpu_state_verified"]
    assert binding["checkpoint"]["state_hash"] == record["checkpoint"]["state_hash"]
    assert len(inventory) == 192
    assert binding["model_loaded"] is False
    assert len(binding["inference_fingerprint"]) == 64


@pytest.mark.parametrize("verify_weights", [False, True])
def test_parent_checkpoint_keeps_exact_historical_backend_schema(tmp_path, verify_weights):
    record, private, protocol = _parent(tmp_path)
    original = copy.deepcopy(record["checkpoint"])
    assert set(original) == {"path", "sha256", "state_hash", "identity"}
    binding, _, _ = legacy._audit_parent(record, private, protocol, verify_weights=verify_weights)
    # Both FrozenBackend.receipt.checkpoint and historical MODEL_IDENTITY's
    # parent_checkpoint retain this exact four-field object. Audit-only byte
    # metadata must not make strict historical inference identity checks fail.
    assert binding["checkpoint"] == original
    assert record["checkpoint"] == original
    if verify_weights:
        assert binding["checkpoint_verified_bytes"] == (tmp_path / "state.pt").stat().st_size
    else:
        assert "checkpoint_verified_bytes" not in binding


def test_parent_exists_but_wrong_commit_is_not_available(tmp_path):
    record, private, protocol = _parent(tmp_path)
    commit = json.loads((tmp_path / "COMMIT.json").read_text())
    commit["checkpoint"]["state_hash"] = "0" * 64
    record["commit"] = _json_file(tmp_path / "COMMIT.json", commit)
    with pytest.raises(ValueError, match="COMMIT changed"):
        legacy._audit_parent(record, private, protocol, verify_weights=True)


def test_parent_metadata_mode_never_implies_tensor_verification(tmp_path):
    record, private, protocol = _parent(tmp_path)
    (tmp_path / "state.pt").unlink()
    binding, inventory, _ = legacy._audit_parent(record, private, protocol, verify_weights=False)
    assert binding["checkpoint_full_cpu_state_verified"] is False
    assert inventory is None
    with pytest.raises(FileNotFoundError):
        legacy._audit_parent(record, private, protocol, verify_weights=True)


def test_released_queue_is_read_without_mutation_and_bound_to_matrix(tmp_path):
    from src.verified_discovery_transfer.queue import digest as queue_digest

    frozen = {"plan_hash": "plan"}
    jobs = [
        {"id": f"train.{p}.{b}.{a}", "kind": "train"}
        for p in legacy.PARENTS
        for b in range(3)
        for a in legacy.LEGACY_ARMS
    ]
    jobs += [{"id": f"eval.{i}", "kind": "eval"} for i in range(95)]
    rows = [{"id": job["id"], "payload": job, "status": "COMPLETE", "result": {}} for job in jobs]
    registry = {"plan_hash": queue_digest(frozen), "jobs": jobs}
    registry["identity"] = queue_digest(registry)
    _json_file(tmp_path / "REGISTERED_MATRIX.json", registry)
    receipts = []
    for job in jobs[18:]:
        receipt = {"job_id": job["id"], "status": "COMPLETE"}
        _json_file(tmp_path / "evaluations" / job["id"] / "completion.json", receipt)
        receipts.append(receipt)
    matrix_digest = queue_digest(rows)
    release = {
        "status": "RELEASED",
        "scientific_completion": True,
        "technical_missing": [],
        "all_registered_models_terminal": True,
        "all_registered_terminal": True,
        "registered_training_jobs": 18,
        "registered_evaluation_jobs": 95,
        "plan_hash": frozen["plan_hash"],
        "matrix_digest": matrix_digest,
        "completed_evaluation_receipts": receipts,
    }
    _json_file(tmp_path / "FINAL_RELEASE.json", release)
    database = tmp_path / "queue.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs(id TEXT,payload TEXT,status TEXT,result TEXT)")
        connection.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        connection.executemany(
            "INSERT INTO jobs VALUES(?,?,?,?)",
            [(row["id"], json.dumps(row["payload"]), row["status"], "{}") for row in rows],
        )
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            [("identity", registry["identity"]), ("released", matrix_digest)],
        )
    before = database.read_bytes()
    _, _, training = legacy._release_and_queue(tmp_path, frozen)
    assert len(training) == 18
    assert database.read_bytes() == before
    release["matrix_digest"] = "changed"
    _json_file(tmp_path / "FINAL_RELEASE.json", release)
    with pytest.raises(ValueError, match="queue/release/matrix"):
        legacy._release_and_queue(tmp_path, frozen)
