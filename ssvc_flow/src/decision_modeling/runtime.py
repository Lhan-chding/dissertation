"""Private runtime bindings for importing original full checkpoints once."""

from __future__ import annotations

import copy
from pathlib import Path

from ..modeling_v3.io import canonical_hash, sha256_file
from ..modeling_v4 import gpu_collect as gpu
from ..modeling_v4.measurement_preview import _check_runtime


def read_config(path):
    config = gpu._read(path)
    inherited = Path(__file__).parents[2] / "configs/modeling_v4/protocol.json"
    if config.get("version") != "decision-modeling-v1":
        raise ValueError("Expected the decision-modeling-v1 protocol")
    if (
        config["source_config_sha256"] != sha256_file(inherited)
        or config["model"] != gpu._read(inherited)["qwen"]
    ):
        raise ValueError("Source model/Adam/generation settings changed")
    blocks = config["blocks"]
    if [blocks[k] for k in ("B", "K", "steps", "Lnorm", "reward_epsilon")] != [4, 8, 32, 64, 1e-4]:
        raise ValueError("Changed the registered training contract")
    if config["observations"]["looks"] != [32, 128, 512]:
        raise ValueError("Changed the registered fixed looks")
    return config


def bound_json(binding):
    gpu._verify_bound_file(binding)
    return gpu._read(binding["path"])


def prepare_runtime(config, *, tasks_path, preview_path, panels_path, out, data_root=None):
    """Read metadata only. Checkpoint .pt files remain untouched until GPU use."""
    tasks = gpu._read(tasks_path)
    if (
        canonical_hash({k: v for k, v in tasks.items() if k != "task_list_hash"})
        != tasks["task_list_hash"]
    ):
        raise ValueError("Source task list hash mismatch")
    if tasks["config"]["qwen"] != config["model"]:
        raise ValueError("Source task model settings differ")
    preview = gpu._read(preview_path)
    if preview["tasks"]["sha256"] != sha256_file(tasks_path):
        raise ValueError("Preview and source task list mismatch")
    banks = [bound_json(b) for b in preview["banks"]]
    active = [
        b
        for b in banks
        if b["policies"]["joint_0"]["inference_fingerprint"]
        != b["policies"]["joint_1"]["inference_fingerprint"]
    ]
    if len(active) != 1:
        raise ValueError("Expected the one original active preview bank")
    origins = {}
    for name, spec in config["blocks"]["origins"].items():
        complete = gpu._read(
            Path(tasks["root"]) / "tasks" / f"source_{spec['seed']}_{spec['arm']}" / "COMPLETE.json"
        )
        origin = complete["origins"][str(spec["step"])]
        if any(origin["identity"][k] != spec[k] for k in ("seed", "arm", "step")):
            raise ValueError("Original checkpoint seed/arm/step mismatch")
        origins[name] = origin
    result = {
        "schema": "decision-modeling-private-runtime-v1",
        "config_hash": canonical_hash(config),
        "tasks": gpu._binding(tasks_path),
        "preview": gpu._binding(preview_path),
        "panels": gpu._binding(panels_path),
        "data_root": str(data_root or tasks["inputs"]["data_root"]),
        "source_root": tasks["root"],
        "origins": origins,
        "preview_origin": bound_json(preview["origin_policy"]),
        "preview_policies": [active[0]["policies"][name] for name in ("joint_0", "joint_1")],
        "checkpoint_files_read": False,
        "private_paths": True,
    }
    gpu._publish(Path(out), result)
    return result


def runtime_metadata(config, runtime_path, *, out):
    private = gpu._read(runtime_path)
    if private["config_hash"] != canonical_hash(config):
        raise ValueError("Protocol changed since runtime binding")
    source, target = Path(private["source_root"]).resolve(), Path(out).resolve()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("New outputs must be separate from the original campaign")
    tasks = bound_json(private["tasks"])
    panels = bound_json(private["panels"])
    if panels["config_hash"] != canonical_hash(config) or panels["status"] != "READY":
        raise ValueError("P/E panels not frozen against this protocol")
    if panels["inputs_hash"] != canonical_hash(
        {k: v for k, v in panels.items() if k != "inputs_hash"}
    ):
        raise ValueError("P/E panel contents changed")
    if tasks["config"]["qwen"] != config["model"]:
        raise ValueError("Bound source model settings differ")
    data_root = Path(private["data_root"]).resolve()
    expected = {value["path"]: value["sha256"] for value in panels["input_bindings"].values()}
    expected.update(panels["image_verification"]["sha256_by_relative_path"])
    for relative, sha in expected.items():
        path = (data_root / relative).resolve()
        if not path.is_relative_to(data_root) or not path.is_file() or sha256_file(path) != sha:
            raise ValueError("Server input changed from frozen panel: " + relative)
    return private, tasks, panels


def load_runtime(config, runtime_path, *, out, execute_gpu=False):
    if not execute_gpu:
        raise PermissionError("GPU execution requires --execute-gpu on the server")
    private, tasks, panels = runtime_metadata(config, runtime_path, out=out)
    runtime = gpu.load_runtime(tasks["config"], tasks["bindings"], execute_gpu=True)
    _check_runtime(runtime, private["preview_origin"], [])
    runtime["data_root"] = private["data_root"]
    runtime["decision_config_hash"] = canonical_hash(config)
    runtime["decision_private"] = private
    runtime["decision_panels"] = panels
    runtime["decision_train_prompts"] = copy.deepcopy(tasks["inputs"]["train_prompts"])
    for prompt in runtime["decision_train_prompts"]:
        prompt["data_root"] = private["data_root"]
    return runtime


def panel_prompts(runtime, name):
    if name not in ("P", "E"):
        raise ValueError("Expected P or E")
    prompts = copy.deepcopy(runtime["decision_panels"]["panels"][name])
    for prompt in prompts:
        prompt["data_root"] = runtime["data_root"]
    return prompts


def policy_from_checkpoint(runtime, spec, *, candidate_id):
    state = runtime["checkpoint_cache"].load(spec)
    required = {"parameters", "optimizer", "rng", "sampler", "buffers", "module_modes"}
    if not required <= state.keys():
        raise ValueError("Full original LoRA/Adam/RNG/sampler/forward checkpoint required")
    return {
        "candidate_id": candidate_id,
        "checkpoint": spec,
        "inference_fingerprint": gpu._fingerprint(state, runtime["identity"]),
    }
