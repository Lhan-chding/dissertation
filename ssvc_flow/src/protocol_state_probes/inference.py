"""Frozen raw-only access to the historical Qwen3.5 uncached generation path.

The checkpoint is deserialized on CPU once. Only its parameters and forward
state are restored; no optimizer is constructed and no saved RNG is resumed.
Public messages are the complete model input boundary. Scoring lives elsewhere.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from ..modeling_v3.io import canonical_hash, sha256_file
from .checkpoint_catalog import map_path, read_bound_json, validate_manifest_identity


def _model_guard(model):
    """Cheap mutation guard; never copy or hash all model weights per answer."""
    return (
        tuple((name, id(p), p._version, p.requires_grad) for name, p in model.named_parameters()),
        tuple((name, id(b), b._version) for name, b in model.named_buffers()),
        tuple((name, module.training) for name, module in model.named_modules()),
    )


def _memory_peaks(device):
    """Read process-local allocator counters without synchronizing or scanning."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return {"peak_memory_allocated_bytes": None, "peak_memory_reserved_bytes": None}
    return {
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def restore_frozen_state(adapter, state):
    """Validate all forward tensor names/shapes/dtypes before modifying the model."""
    import torch

    from ..followup_updates import POSITION_ATTRIBUTES, _restore_forward, _safe

    model = adapter.model
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not parameters or set(parameters) != set(state["parameters"]):
        raise ValueError("Checkpoint parameter names differ from the inherited adapter")
    modules = dict(model.named_modules())
    if set(modules) != set(state["module_modes"]):
        raise ValueError("Checkpoint module topology mismatch")
    buffers = {
        f"{name}.{key}" if name else key: value
        for name, module in modules.items()
        for key, value in module._buffers.items()
    }
    if set(buffers) != set(state["buffers"]):
        raise ValueError("Checkpoint buffer names differ from the inherited adapter")

    def check_tensor(name, live, saved):
        if live is None or saved is None:
            if live is not saved:
                raise ValueError("Checkpoint buffer presence differs: " + name)
        elif (
            not isinstance(saved, torch.Tensor)
            or live.shape != saved.shape
            or live.dtype != saved.dtype
        ):
            raise ValueError("Checkpoint tensor shape/dtype mismatch: " + name)

    for name, live in parameters.items():
        check_tensor(name, live, state["parameters"][name])
    for name, live in buffers.items():
        check_tensor(name, live, state["buffers"][name])
    for key in state["position_state"]:
        module_name, attribute = key.rsplit(":", 1)
        if module_name not in modules:
            raise ValueError("Checkpoint position-state module mismatch")
        if attribute not in POSITION_ATTRIBUTES:
            raise ValueError("Unknown saved position/cache attribute")
    if set(state["adapter_position_state"]) - set(POSITION_ATTRIBUTES):
        raise ValueError("Unknown saved adapter position/cache attribute")
    if any(type(mode) is not bool for mode in state["module_modes"].values()):
        raise ValueError("Saved module modes must be boolean")
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError("Historical adapter must have dropout disabled")
    _safe(
        {
            key: state[key]
            for key in ("parameters", "buffers", "position_state", "adapter_position_state")
        }
    )
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state["parameters"][name].to(device=parameter.device))
        _restore_forward(model, state, adapter)
        for name, parameter in parameters.items():
            if not torch.equal(parameter.detach(), state["parameters"][name].to(parameter.device)):
                raise RuntimeError("Checkpoint parameter restoration differs: " + name)
    # Restoration is complete before freezing. Never initialize/reset LoRA here.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    adapter._reset_positions()


def _validate_protocol(protocol, inherited, parent):
    from ..model_adapters.base import pure_generation_options

    if (
        protocol["execution"].get("frozen_inference_only") is not True
        or protocol["execution"].get("new_training_allowed") is not False
    ):
        raise ValueError("Frozen inference protocol required")
    model = protocol["model"]
    if (
        model["id"] != "Qwen/Qwen3.5-9B"
        or model["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    ):
        raise ValueError("Only the registered immutable Qwen3.5-9B base is allowed")
    qwen = inherited["qwen"]
    for key in ("id", "revision"):
        if qwen[key] != model[key] or parent["config"]["model"][key] != model[key]:
            raise ValueError("Inherited model identity differs: " + key)
    if parent["snapshot"]["revision"] != model["revision"] or model["base_dtype"] != "bfloat16":
        raise ValueError("Base snapshot revision/dtype differs")
    expected_lora = {
        "r": 8,
        "alpha": 16,
        "dropout": 0,
        "target_leaf_modules": ["gate_proj", "up_proj", "down_proj"],
        "expected_modules": 96,
        "freeze_vision": True,
        "freeze_base": True,
    }
    if qwen["lora"] != expected_lora:
        raise ValueError("Inherited LoRA topology differs")
    generation = protocol["generation"]
    if generation["generation_config"] != pure_generation_options(64):
        raise ValueError("Generation options differ from the historical implementation")
    if (
        generation.get("enable_thinking") is not False
        or generation.get("new_likelihood_scoring") is not False
    ):
        raise ValueError("Thinking and new likelihood scoring are forbidden")
    for key in ("temperature", "top_p", "top_k", "max_new_tokens"):
        if (
            generation[key] != qwen["generation"][key]
            or generation[key] != parent["config"]["generation_proposed_N"][key]
        ):
            raise ValueError("Inherited generation contract differs: " + key)
    if qwen["generation"]["thinking"] is not False:
        raise ValueError("Historical thinking switch differs")
    if model["primary_engine"] != "historical uncached-prefix-recompute":
        raise ValueError("Unbridged inference engine change")


def _validate_checkpoint_metadata(record, metadata):
    source = record["source"]
    if metadata.get("lineage_id") != source["lineage"] or metadata.get("source_recipe") != "R0":
        raise ValueError("Checkpoint tensor metadata lineage/R0 ancestry differs")
    step = source["source_step"] if source["kind"] == "source" else source["continuation_steps"]
    if metadata.get("checkpoint_step") != step:
        raise ValueError("Checkpoint tensor metadata step differs")
    if source["kind"] == "branch":
        expected = {
            "origin_id": f"{source['lineage']}_t{source['source_step']}",
            "origin_checkpoint_step": source["source_step"],
            "repeat": source["repeat"],
            "recipe": source["recipe"],
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint tensor metadata branch identity differs")
    elif metadata.get("recipe") != "R0":
        raise ValueError("Checkpoint tensor metadata source reward differs")


class FrozenBackend:
    """One restored checkpoint, one allocated PRO6000, and no scoring/training API."""

    def __init__(self, runtime_path, checkpoint_record, protocol, allow_gpu=False):
        if allow_gpu is not True:
            raise PermissionError("Real frozen model execution requires --allow-gpu")
        if checkpoint_record.get("status") != "AVAILABLE":
            raise ValueError("A CPU-verified available checkpoint record is required")
        if checkpoint_record.get("checkpoint_id") != checkpoint_record.get("source", {}).get("id"):
            raise ValueError("Checkpoint record ID differs from its registered source")
        began = time.perf_counter()
        self.protocol = copy.deepcopy(protocol)
        record = copy.deepcopy(checkpoint_record)
        mappings = record.get("path_mappings", {})
        private = json.loads(Path(runtime_path).read_text())
        if private.get("schema") != "prospective-private-runtime-v1":
            raise ValueError("The historical private prospective runtime is required")
        parent_binding = private["bindings"]["parent_validated_plan"]
        parent = read_bound_json(parent_binding, mappings)
        _validate_protocol(protocol, private["inherited_config"], parent)
        # The already-mapped catalog bindings are read without applying mappings twice.
        manifest = read_bound_json(record["manifest"])
        commit = read_bound_json(record["commit"])
        validate_manifest_identity(record["source"], manifest, commit)
        if commit["checkpoint"] != record["checkpoint_original_binding"]:
            raise ValueError("Historical COMMIT changed since availability verification")
        expected_checkpoint = {
            **commit["checkpoint"],
            "path": str(map_path(commit["checkpoint"]["path"], mappings)),
        }
        if record["checkpoint"] != expected_checkpoint:
            raise ValueError("Mapped checkpoint binding differs from the authoritative COMMIT")
        historical_identity = manifest["runtime_identity"]
        if historical_identity["model_hash"] != canonical_hash(parent["snapshot"]):
            raise ValueError("Checkpoint was trained against a different base snapshot")
        if historical_identity["config_hash"] != canonical_hash(private["inherited_config"]):
            raise ValueError("Checkpoint inherited runtime configuration differs")
        if manifest["config_hash"] != private["config_hash"]:
            raise ValueError("Checkpoint prospective protocol identity differs")

        import torch

        from ..followup_backend import _load_local_adapter
        from ..modeling_v3.vlm_campaign import _allocated_gpu_info
        from ..next_stage_runtime import validate_runtime_environment
        from ..optimizer_fork import state_hash

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("One allocated visible CUDA GPU is required")
        allocated_gpu = _allocated_gpu_info()
        if "PRO6000" not in allocated_gpu["name"].upper().replace(" ", ""):
            raise ValueError("The allocated device is not a PRO6000")
        environment = validate_runtime_environment(parent["gate"])
        if (
            environment != parent["r4_binding"]["environment"]
            or environment != historical_identity["environment"]
        ):
            raise ValueError("Measured historical runtime dependencies have changed")
        spec = record["checkpoint"]
        checkpoint_began = time.perf_counter()
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError("Checkpoint file digest mismatch")
        payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
        if payload["identity"] != spec["identity"] or payload["state_hash"] != spec["state_hash"]:
            raise ValueError("Checkpoint payload binding differs from COMMIT")
        state = payload["state"]
        if state_hash(state) != spec["state_hash"]:
            raise ValueError("Checkpoint full CPU state digest mismatch")
        if state.get("schema") != "ssvc-v4-complete-trainable-state-1":
            raise ValueError("Complete historical source state is required")
        required = {
            "parameters",
            "optimizer",
            "rng",
            "sampler",
            "buffers",
            "module_modes",
            "position_state",
            "adapter_position_state",
            "metadata",
            "scheduler",
        }
        if not required <= state.keys() or state["scheduler"] is not None:
            raise ValueError("A full historical LoRA/Adam/RNG/sampler checkpoint is required")
        if not isinstance(state["optimizer"], dict) or state["sampler"].get("kind") != "mapping":
            raise ValueError("Complete historical optimizer/sampler identity is required")
        _validate_checkpoint_metadata(record, state["metadata"])
        checkpoint_seconds = time.perf_counter() - checkpoint_began
        snapshot = copy.deepcopy(parent["snapshot"])
        snapshot["path"] = str(map_path(snapshot["path"], mappings).resolve())
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        self.adapter = _load_local_adapter(snapshot, parent["config"]["model"])
        audit = self.adapter.audit
        certificate = parent["gate"]["certificate"]
        if (
            certificate.get("status") != "PASS"
            or certificate.get("selected_path") != "uncached_prefix_recompute"
        ):
            raise ValueError("Historical uncached inference certificate is unavailable")
        for key in (
            "processor_hash",
            "tokenizer_hash",
            "chat_template_hash",
            "frozen_parameter_hash",
            "eos_token_ids",
            "lora_modules",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
        ):
            if audit.get(key) != certificate["model_audit"].get(key) or audit.get(key) is None:
                raise ValueError("Historical forward identity changed: " + key)
        if (
            audit["probability_execution"] != "uncached_prefix_recompute"
            or len(audit["lora_modules"]) != 96
        ):
            raise ValueError("Historical adapter topology/execution path differs")
        if not self.adapter.eos_ids or sorted(self.adapter.eos_ids) != audit["eos_token_ids"]:
            raise ValueError("Actual EOS boundary differs from the historical model")
        for name, parameter in self.adapter.model.named_parameters():
            if parameter.requires_grad and (
                "lora_" not in name or str(parameter.dtype) != "torch.float32"
            ):
                raise ValueError("Unexpected LoRA parameter or dtype")
        forward_hash = state_hash(
            {
                key: state[key]
                for key in ("parameters", "buffers", "position_state", "adapter_position_state")
            }
        )
        metadata = copy.deepcopy(state["metadata"])
        restore_frozen_state(self.adapter, state)
        self.data_root = private.get("data_root")
        self._prepared_cache = {}
        self._guard_baseline = _model_guard(self.adapter.model)
        load_memory = _memory_peaks(self.adapter.device)
        self.counters = {
            "generated_sequences": 0,
            "generated_tokens": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_updates": 0,
            "adam_updates": 0,
            "scored_sequences": 0,
            "scored_tokens": 0,
            "elapsed_seconds": 0.0,
            **load_memory,
        }
        self.receipt = {
            "schema": "protocol-probe-frozen-backend-v1",
            "execution_kind": "REAL_CUDA_FROZEN_INFERENCE",
            "checkpoint_id": record["checkpoint_id"],
            "checkpoint": spec,
            "commit": record["commit"],
            "manifest": record["manifest"],
            "checkpoint_metadata": metadata,
            "checkpoint_full_cpu_state_verified": True,
            "runtime": {"path": str(runtime_path), "sha256": sha256_file(runtime_path)},
            "parent_plan": parent_binding,
            "path_mappings": mappings,
            "historical_runtime_identity": historical_identity,
            "model_audit": audit,
            "inference_fingerprint": canonical_hash(
                {
                    "checkpoint_sha256": spec["sha256"],
                    "base_snapshot_hash": historical_identity["model_hash"],
                    "forward_state_hash": forward_hash,
                    "mode": "eval_positions_reset",
                    "generation": protocol["generation"],
                }
            ),
            "generation_options": protocol["generation"]["generation_config"],
            "eos_token_ids": sorted(self.adapter.eos_ids),
            "pad_token_id": self.adapter.pad_id,
            "bos_token_id": self.adapter.processor.tokenizer.bos_token_id,
            "forward_config": self.adapter.model.config.to_dict(),
            "allocated_gpu": allocated_gpu,
            "environment": environment,
            "checkpoint_load_and_verify_seconds": checkpoint_seconds,
            "total_load_seconds": time.perf_counter() - began,
            "load_peak_memory_allocated_bytes": load_memory["peak_memory_allocated_bytes"],
            "load_peak_memory_reserved_bytes": load_memory["peak_memory_reserved_bytes"],
            "memory_peak_scope": "process_since_historical_loader_reset_before_model_load",
            "optimizer_constructed": False,
            "optimizer_tensors_on_gpu": False,
            "backward_calls": 0,
            "optimizer_updates": 0,
            "likelihood_rescoring_calls": 0,
            "all_parameters_frozen": True,
            "dropout_disabled": True,
            "eval_mode": True,
            "checkpoint_restored_before_freeze": True,
            "lora_reset_after_restore": False,
        }
        # Adam, sampler and old RNG remain committed historical identity only.
        del state, payload

    def generate_public(self, prompt, seed, max_new_tokens=64):
        import torch

        from ..modeling_v3.vlm_observation import ObservationFault, validate_action

        if (
            not isinstance(prompt, dict)
            or set(prompt) != {"system", "user"}
            or any(not isinstance(v, str) for v in prompt.values())
        ):
            raise ValueError("Model input accepts only public string system/user fields")
        if type(seed) is not int or seed < 0 or seed >= 2**63:
            raise ValueError("Seed must be a nonnegative signed 64-bit integer")
        if type(max_new_tokens) is not int or max_new_tokens != 64:
            raise ValueError("The registered generation horizon is exactly 64 tokens")
        if _model_guard(self.adapter.model) != self._guard_baseline:
            raise RuntimeError("Frozen model state changed before generation")
        key = canonical_hash(prompt)
        if key not in self._prepared_cache:
            self._prepared_cache[key] = self.adapter.prepare(copy.deepcopy(prompt), self.data_root)
        prepared = self._prepared_cache[key]
        began, before = time.perf_counter(), self.adapter.forward_calls
        device = torch.device(self.adapter.device)
        devices = [device.index or 0] if device.type == "cuda" else []
        raw = None
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                raw = self.adapter.generate(prepared, seed=seed, max_new_tokens=64, do_sample=True)
            self.counters["generated_sequences"] += 1
            self.counters["generated_tokens"] += len(raw.get("token_ids", []))
            try:
                flags = validate_action(
                    raw,
                    eos_ids=self.adapter.eos_ids,
                    max_new_tokens=64,
                    tokenizer=self.adapter.processor.tokenizer,
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ObservationFault(str(exc), raw) from exc
        finally:
            self.counters["elapsed_seconds"] += time.perf_counter() - began
            self.counters["forward_calls"] += self.adapter.forward_calls - before
            memory = _memory_peaks(device)
            for name, value in memory.items():
                self.counters[name] = (
                    max(self.counters.get(name) or 0, value) if value is not None else None
                )
            if _model_guard(self.adapter.model) != self._guard_baseline:
                raise ObservationFault("Frozen model changed during generation", raw)
        return {
            "raw_text": raw["raw_completion"],
            "token_ids": raw["token_ids"],
            "stop_reason": raw["stop_reason"],
            "generated_length": raw["completion_length"],
            "elapsed_seconds": raw["elapsed_seconds"],
            "chosen_token_logprobs": raw["behavior_token_logprobs"],
            "input_audit": copy.deepcopy(prepared["audit"]),
            "public_prompt_hash": key,
            "inference_fingerprint": self.receipt["inference_fingerprint"],
            "eos_token_ids": sorted(self.adapter.eos_ids),
            "pad_token_id": self.adapter.pad_id,
            "seed": seed,
            "extra_rescoring": False,
            **memory,
            **flags,
        }
