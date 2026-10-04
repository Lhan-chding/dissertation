"""Checkpoint identity and raw-only inference contracts; no real model downloads."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling_v3.io import canonical_hash, sha256_file
from src.protocol_state_probes.checkpoint_catalog import check_checkpoints
from src.protocol_state_probes.inference import FrozenBackend, restore_frozen_state


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sha256_file(path)}


def committed_checkpoint(tmp_path, *, branch=False):
    root = tmp_path / "campaign"
    checkpoint = root / "segments/025_032/attempt_123/state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not loaded by CPU catalog")
    identity = {"lineage_id": 61001, "source_recipe": "R0", "role": "source"}
    record = {"id": "S32", "lineage": 61001, "source_step": 32, "kind": "source"}
    recipe = "R0"
    if branch:
        identity = {
            "lineage_id": 61001,
            "origin_id": "61001_t96",
            "repeat": 1,
            "recipe_id": "R4",
            "role": "development",
        }
        record.update(
            id="R4_128", source_step=96, kind="branch", recipe="R4", repeat=1, continuation_steps=32
        )
        recipe = "R4"
    manifest = {
        "kind": "PROSPECTIVE_TRAINING",
        "fixture": False,
        "identity": identity,
        "recipe": recipe,
        "steps": 96 if not branch else 32,
        "runtime_identity": {"execution_kind": "REAL_CUDA_MODEL"},
    }
    write_json(root / "MANIFEST.json", manifest)
    binding = {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "identity": {"manifest_hash": canonical_hash(manifest), "step": 32},
        "state_hash": "1" * 64,
    }
    commit = {
        "start": 25,
        "stop": 32,
        "manifest_hash": canonical_hash(manifest),
        "checkpoint": binding,
    }
    write_json(root / "segments/025_032/COMMIT.json", commit)
    record["export_time_candidates"] = [
        {"path": str(checkpoint), "bytes": checkpoint.stat().st_size}
    ]
    return record, root, manifest, commit


def test_catalog_reads_commit_without_importing_torch_or_loading_tensors(tmp_path, monkeypatch):
    record, root, _, _ = committed_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    result = check_checkpoints([record])
    assert result["checkpoints"][0]["status"] == "AVAILABLE"
    assert result["model_loaded"] is False
    assert result["checkpoint_tensors_loaded"] is False
    assert result["checkpoints"][0]["commit"]["path"] == str(root / "segments/025_032/COMMIT.json")


def test_catalog_never_promotes_uncommitted_attempt_and_retains_missing(tmp_path):
    record, root, _, _ = committed_checkpoint(tmp_path)
    (root / "segments/025_032/COMMIT.json").unlink()
    result = check_checkpoints([record])["checkpoints"][0]
    assert result["status"] == "MISSING"
    assert "checkpoint" not in result


@pytest.mark.parametrize("branch", [False, True])
def test_catalog_rejects_wrong_semantic_lineage_and_recipe(tmp_path, branch):
    record, root, manifest, commit = committed_checkpoint(tmp_path, branch=branch)
    manifest["identity"]["lineage_id"] = 61002
    write_json(root / "MANIFEST.json", manifest)
    commit["manifest_hash"] = canonical_hash(manifest)
    commit["checkpoint"]["identity"]["manifest_hash"] = canonical_hash(manifest)
    write_json(root / "segments/025_032/COMMIT.json", commit)
    result = check_checkpoints([record])["checkpoints"][0]
    assert result["status"] == "INVALID"
    assert "lineage" in result["errors"][0]["error"].lower()


def test_catalog_explicit_mount_mapping_preserves_export_path(tmp_path):
    record, _, _, _ = committed_checkpoint(tmp_path)
    original = record["export_time_candidates"][0]["path"]
    record["export_time_candidates"][0]["path"] = original.replace(str(tmp_path), "/old/mount")
    result = check_checkpoints([record], {"/old/mount": str(tmp_path)})["checkpoints"][0]
    assert result["status"] == "AVAILABLE"
    assert result["source"] == record
    assert result["checkpoint"]["path"] == original


def tiny_adapter_and_state():
    import torch

    from src.followup_updates import _forward_state

    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Dropout(0))
    model.register_buffer("position_buffer", torch.tensor([2.0]))
    adapter = SimpleNamespace(model=model, _reset_positions=lambda: None)
    state = {
        "parameters": {n: p.detach().clone() + 5 for n, p in model.named_parameters()},
        **_forward_state(model, adapter),
    }
    return adapter, state


def test_restore_is_strict_freezes_everything_and_preserves_saved_values():
    import torch

    adapter, state = tiny_adapter_and_state()
    restore_frozen_state(adapter, state)
    for name, parameter in adapter.model.named_parameters():
        assert torch.equal(parameter.detach(), state["parameters"][name])
        assert not parameter.requires_grad
    assert all(not module.training for module in adapter.model.modules())


@pytest.mark.parametrize(
    "damage", ["name", "shape", "dtype", "buffer_missing", "buffer_shape", "topology"]
)
def test_restore_rejects_drift_before_writing(damage):
    import torch

    adapter, state = tiny_adapter_and_state()
    before = {n: p.detach().clone() for n, p in adapter.model.named_parameters()}
    if damage == "name":
        state["parameters"]["alien"] = state["parameters"].pop("0.weight")
    elif damage == "shape":
        state["parameters"]["0.weight"] = torch.zeros(3)
    elif damage == "dtype":
        state["parameters"]["0.weight"] = state["parameters"]["0.weight"].double()
    elif damage == "buffer_missing":
        state["buffers"] = {}
    elif damage == "buffer_shape":
        state["buffers"]["position_buffer"] = torch.zeros(3)
    elif damage == "topology":
        state["module_modes"]["alien"] = False
    with pytest.raises(ValueError):
        restore_frozen_state(adapter, state)
    assert all(torch.equal(p, before[n]) for n, p in adapter.model.named_parameters())


def fake_backend(monkeypatch):
    import torch

    from src.protocol_state_probes import inference

    backend = FrozenBackend.__new__(FrozenBackend)
    backend.protocol = {"generation": {"max_new_tokens": 64}}
    backend.counters = {
        "generated_sequences": 0,
        "generated_tokens": 0,
        "backward_calls": 0,
        "optimizer_updates": 0,
        "scored_sequences": 0,
        "forward_calls": 0,
        "elapsed_seconds": 0.0,
    }
    backend.receipt = {"checkpoint_id": "TEST", "inference_fingerprint": "frozen"}
    backend._prepared_cache = {}
    model = torch.nn.Linear(1, 1)
    model.requires_grad_(False)
    model.eval()
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "not json")

    def generate(prepared, **kwargs):
        assert not torch.is_grad_enabled()
        return {
            "raw_completion": "not json",
            "token_ids": [3, 9],
            "completion_length": 2,
            "stop_reason": "eos",
            "behavior_token_logprobs": [-1.0, -0.2],
            "elapsed_seconds": 0.1,
        }

    adapter = SimpleNamespace(
        model=model,
        eos_ids={9},
        pad_id=9,
        device="cpu",
        forward_calls=0,
        processor=SimpleNamespace(tokenizer=tokenizer),
        generate=generate,
        prepare=lambda prompt, root: {"inputs": {}, "audit": {"prompt": prompt}},
    )
    backend.adapter = adapter
    backend.data_root = None
    backend._guard_baseline = inference._model_guard(model)
    return backend


def test_public_boundary_rejects_audit_keys_before_prepare(monkeypatch):
    backend = fake_backend(monkeypatch)
    with pytest.raises(ValueError, match="system/user"):
        backend.generate_public({"system": "x", "user": "y", "truth": [1, 2, 3, 4]}, seed=1)


def test_raw_invalid_answer_is_retained_without_old_semantics_or_retry(monkeypatch):
    backend = fake_backend(monkeypatch)
    raw = backend.generate_public({"system": "x", "user": "y"}, seed=1)
    assert raw["raw_text"] == "not json"
    assert raw["generated_length"] == 2
    assert "semantic" not in raw and "event" not in raw
    assert backend.counters["generated_sequences"] == 1
    assert backend.counters["optimizer_updates"] == backend.counters["backward_calls"] == 0


def test_allow_gpu_gate_precedes_runtime_or_framework_reads(tmp_path):
    with pytest.raises(PermissionError):
        FrozenBackend(tmp_path / "missing.json", {}, {}, allow_gpu=False)


def test_full_loader_restores_checkpoint_without_optimizer_or_lora_reset(tmp_path, monkeypatch):
    import torch

    from src import followup_backend, next_stage_runtime
    from src.followup_updates import _forward_state
    from src.modeling_v3 import vlm_campaign
    from src.optimizer_fork import state_hash

    record, root, manifest, commit = committed_checkpoint(tmp_path)
    project = Path(__file__).resolve().parents[1]
    protocol = json.loads((project / "docs/protocol_state_probes/design/protocol.json").read_text())
    inherited = json.loads((project / "configs/modeling_v4/protocol.json").read_text())
    model = torch.nn.Module()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2, 2)))
    model.config = SimpleNamespace(to_dict=lambda: {"fixture": True})
    audit = {
        "probability_execution": "uncached_prefix_recompute",
        "eos_token_ids": [9],
        "lora_modules": [f"layer_{i}" for i in range(96)],
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        **{
            key: "fixed"
            for key in (
                "processor_hash",
                "tokenizer_hash",
                "chat_template_hash",
                "frozen_parameter_hash",
            )
        },
    }
    adapter = SimpleNamespace(
        model=model,
        audit=audit,
        eos_ids={9},
        pad_id=9,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=1)),
        _reset_positions=lambda: None,
    )
    parent = {
        "snapshot": {
            "path": str(tmp_path / protocol["model"]["revision"]),
            "revision": protocol["model"]["revision"],
        },
        "config": {
            "model": {key: protocol["model"][key] for key in ("id", "revision")},
            "generation_proposed_N": protocol["generation"],
        },
        "gate": {
            "certificate": {
                "status": "PASS",
                "selected_path": "uncached_prefix_recompute",
                "model_audit": audit,
            }
        },
        "r4_binding": {"environment": {"fixture": "measured-env"}},
    }
    parent_binding = write_json(tmp_path / "parent.json", parent)
    manifest.update(
        runtime_identity={
            "execution_kind": "REAL_CUDA_MODEL",
            "model_hash": canonical_hash(parent["snapshot"]),
            "config_hash": canonical_hash(inherited),
            "environment": parent["r4_binding"]["environment"],
        },
        config_hash="prospective-config",
    )
    write_json(root / "MANIFEST.json", manifest)
    checkpoint_identity = {"manifest_hash": canonical_hash(manifest), "step": 32}
    state = {
        "schema": "ssvc-v4-complete-trainable-state-1",
        "parameters": {"lora_A": torch.full((2, 2), 7.0)},
        "metadata": {
            "lineage_id": 61001,
            "source_recipe": "R0",
            "recipe": "R0",
            "checkpoint_step": 32,
        },
        "optimizer": {"state": {"must_remain_cpu": torch.ones(1)}, "param_groups": []},
        "rng": {},
        "sampler": {"kind": "mapping", "state": {}},
        "scheduler": None,
        **_forward_state(model, adapter),
    }
    checkpoint_path = Path(record["export_time_candidates"][0]["path"])
    torch.save(
        {"identity": checkpoint_identity, "state_hash": state_hash(state), "state": state},
        checkpoint_path,
    )
    record["export_time_candidates"][0]["bytes"] = checkpoint_path.stat().st_size
    commit.update(
        manifest_hash=canonical_hash(manifest),
        checkpoint={
            "identity": checkpoint_identity,
            "state_hash": state_hash(state),
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
    )
    write_json(root / "segments/025_032/COMMIT.json", commit)
    runtime_path = tmp_path / "runtime.json"
    write_json(
        runtime_path,
        {
            "schema": "prospective-private-runtime-v1",
            "config_hash": "prospective-config",
            "bindings": {"parent_validated_plan": parent_binding},
            "inherited_config": inherited,
        },
    )
    available = check_checkpoints([record])["checkpoints"][0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(vlm_campaign, "_allocated_gpu_info", lambda: {"name": "NVIDIA RTX PRO6000"})
    monkeypatch.setattr(
        next_stage_runtime,
        "validate_runtime_environment",
        lambda gate: parent["r4_binding"]["environment"],
    )
    monkeypatch.setattr(followup_backend, "_load_local_adapter", lambda *args: adapter)

    def forbid_optimizer(*args, **kwargs):
        raise AssertionError("Frozen inference must never construct an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", forbid_optimizer)
    backend = FrozenBackend(runtime_path, available, protocol, allow_gpu=True)
    assert torch.equal(backend.adapter.model.lora_A, torch.full((2, 2), 7.0))
    assert not backend.adapter.model.lora_A.requires_grad
    assert backend.receipt["checkpoint_metadata"]["checkpoint_step"] == 32
    assert backend.receipt["optimizer_constructed"] is False
    assert backend.receipt["lora_reset_after_restore"] is False


def test_sample_rng_is_restored_and_mutation_fault_preserves_raw(monkeypatch):
    import torch

    from src.modeling_v3.vlm_observation import ObservationFault

    backend = fake_backend(monkeypatch)
    original = backend.adapter.generate

    def sampled(*args, **kwargs):
        torch.manual_seed(kwargs["seed"])
        torch.rand(3)
        return original(*args, **kwargs)

    backend.adapter.generate = sampled
    before = torch.get_rng_state().clone()
    backend.generate_public({"system": "x", "user": "y"}, seed=42)
    assert torch.equal(before, torch.get_rng_state())

    def mutating(*args, **kwargs):
        raw = original(*args, **kwargs)
        next(backend.adapter.model.parameters()).add_(1)
        return raw

    backend.adapter.generate = mutating
    with pytest.raises(ObservationFault) as fault:
        backend.generate_public({"system": "x", "user": "y"}, seed=42)
    assert fault.value.raw_result["raw_completion"] == "not json"
