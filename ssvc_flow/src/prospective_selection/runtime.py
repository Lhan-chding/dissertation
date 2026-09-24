"""Small private runtime bindings; preserve the measured D2 model contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..modeling_v3.io import canonical_hash, sha256_file
from ..modeling_v4 import gpu_collect as gpu
from .protocol import validate_protocol


def read_json(path):
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def check_inherited_contract(config, inherited):
    validate_protocol(config)
    q, t = inherited["qwen"], config["training"]
    if any(q[k] != config["model"][k] for k in ("id", "revision")):
        raise ValueError("Inherited checkpoint identity differs")
    for key in ("B", "K", "Lnorm", "reward_epsilon"):
        if q[key] != t[key]:
            raise ValueError("Inherited training setting differs: " + key)
    for key in ("max_new_tokens", "temperature", "top_p", "top_k", "thinking"):
        if q["generation"][key] != t[key]:
            raise ValueError("Sampling mismatch requires an explicit measured bridge: " + key)
    for key, value in t["adamw"].items():
        if q["optimizer"][key] != value:
            raise ValueError("Inherited Adam setting differs: " + key)
    if q["optimizer"]["grad_clip"] != t["gradient_clip"]:
        raise ValueError("Gradient clip mismatch")
    return {
        "model": q,
        "initialization": "literal LoRA seed17, common to every new lineage",
        "old_initialization": "historical V3 hash-derived seed retained only in historical runs",
    }


def prepare_runtime(config, old_runtime_path, prepared_path, out):
    """Verify original small bindings once. Never load model/checkpoint tensors here."""
    old = read_json(old_runtime_path)
    for name in ("tasks", "panels"):
        gpu._verify_bound_file(old[name])
    tasks = read_json(old["tasks"]["path"])
    bridge = check_inherited_contract(config, tasks["config"])
    prepared = read_json(prepared_path)
    result = {
        "schema": "prospective-private-runtime-v1",
        "config_hash": canonical_hash(config),
        "inherited_config": tasks["config"],
        "bindings": tasks["bindings"],
        "data_root": old["data_root"],
        "historical_runtime": str(old_runtime_path),
        "prepared": {
            "path": str(Path(prepared_path).resolve()),
            "sha256": sha256_file(prepared_path),
        },
        "contract_bridge": bridge,
        "prepared_schema": prepared.get("schema"),
        "checkpoint_tensors_loaded": False,
    }
    gpu._publish(Path(out), result)
    return result


def load_runtime(config, runtime_path, *, allow_gpu=False):
    if allow_gpu is not True:
        raise PermissionError("Real execution requires --allow-gpu")
    private = read_json(runtime_path)
    if private["config_hash"] != canonical_hash(config):
        raise ValueError("Private runtime belongs to a different protocol")
    check_inherited_contract(config, private["inherited_config"])
    gpu._verify_bound_file(private["prepared"])
    prepared = read_json(private["prepared"]["path"])
    run = gpu.load_runtime(private["inherited_config"], private["bindings"], execute_gpu=True)
    # The inherited loader reinitializes by a historical source seed. This new
    # experiment explicitly uses one literal seed17 for all new source lineages.
    import torch

    from ..smoke_runtime import _seed_everything

    if run["optimizer"].state:
        raise ValueError("Refuse to replace a warm optimizer initialization")
    _seed_everything(config["model"]["lora_initialization_seed"])
    with torch.no_grad():
        for name, parameter in sorted(run["adapter"].model.named_parameters()):
            if not parameter.requires_grad:
                continue
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
            elif "lora_B" in name:
                parameter.zero_()
            else:
                raise ValueError("Unexpected trainable parameter: " + name)
    run.update(
        data_root=private["data_root"],
        prospective_config=config,
        prospective_config_hash=canonical_hash(config),
        prepared_data=prepared,
    )
    run["initial_state"] = run["state"].capture({"checkpoint_step": 0, "lora_seed": 17})
    return run
