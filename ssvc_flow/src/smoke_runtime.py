"""P1 only: measured compatibility, probability checks and real reversible updates.

This runtime never discovers or connects to a server. Run it on the authorized NTU
CUDA device after upload. Later pilot/confirm execution remains gated by P1.
"""

from __future__ import annotations

import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

from .core import (
    PROJECT_ROOT,
    RunStore,
    canonical_hash,
    file_hash,
    sample_identity,
    source_commit,
    write_json,
)
from .grpo_update import grouped_advantages, perform_update, reward_channels
from .likelihood import audit_model_cache
from .optimizer_fork import (
    capture_state,
    compare_parameters,
    load_checkpoint,
    parameter_hash,
    restore_state,
    save_checkpoint,
    state_hash,
)
from .prompts import build_prompt
from .verifiers import annotate

INTERFACES = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")


class Telemetry:
    def reset(self):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self):
        import torch

        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if torch.cuda.is_available()
            else None,
            "peak_cpu_rss_bytes": ram if sys.platform == "darwin" else ram * 1024,
        }


def _seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    # Unsupported nondeterministic kernels must raise instead of claiming equality.
    torch.use_deterministic_algorithms(True)


def _runtime_lock(config, adapter, out, identity):
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    source_files = {
        str(path.relative_to(PROJECT_ROOT)): file_hash(path)
        for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
    }
    environment = {
        "pip_freeze_hash": canonical_hash(freeze.stdout),
        "source_files": source_files,
        "source_commit": source_commit(),
    }
    prior_path = out / "runtime_lock.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior.get("environment_identity") != environment:
            raise ValueError("Resume dependency/source drift; preserving the original runtime lock")
        return prior
    (out / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    resolved = {
        **config,
        "models": {
            **config["models"],
            config["_model_key"]: {
                **config["models"][config["_model_key"]],
                "revision": adapter.revision,
            },
        },
    }
    lock = {
        "identity": identity,
        "model_revision": adapter.revision,
        "resolved_config": resolved,
        "source_commit": source_commit(),
        "environment_identity": environment,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pip_freeze_sha256": file_hash(out / "pip-freeze.txt"),
        "deterministic_algorithms": True,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "runtime_status": "CANDIDATE_UNTIL_P1_PASS",
    }
    write_json(out / "runtime_lock.json", lock)
    return lock


def _verify_scene_images(scenes, data_root):
    root = Path(data_root).resolve()
    for scene in scenes:
        path = (root / scene["image_path"]).resolve()
        if not path.is_relative_to(root) or file_hash(path) != scene["image_hash"]:
            raise ValueError(
                "Image path or image content hash differs from the locked data manifest"
            )


def _new_sample(adapter, prepared, scene, interface, seed, step, index, run_id, config, telemetry):
    telemetry.reset()
    generation = adapter.generate(
        prepared, seed=seed, max_new_tokens=config["generation"]["max_new_tokens"]
    )
    memory = telemetry.snapshot()
    tokens = generation["token_ids"]
    if not tokens:
        raise RuntimeError("Generation returned no action or EOS tokens")
    old = adapter.logprobs(prepared, tokens).detach().cpu().tolist()
    reference = adapter.reference_logprobs(prepared, tokens).detach().cpu().tolist()
    behavior = generation["behavior_token_logprobs"]
    if len(old) != len(behavior):
        raise RuntimeError("Generation and teacher-forcing masks differ")
    error = sum(abs(a - b) for a, b in zip(old, behavior, strict=False)) / len(old)
    prompt_id = canonical_hash({"scene": scene["base_scene_id"], "interface": interface})
    annotation = annotate(
        generation["raw_completion"], scene["truth_world"], scene["operation"], scene["cue"]
    )
    channels = reward_channels(annotation["category"], "X_BASE")
    return {
        **scene,
        **annotation,
        **generation,
        **prepared["audit"],
        **memory,
        "run_id": run_id,
        "phase": "P1",
        "model_id": adapter.model_id,
        "model_revision": adapter.revision,
        "dtype": "bfloat16",
        "adapter_hash": parameter_hash(adapter.model, trainable=True),
        "optimizer_step": step,
        "train_seed": config["training"]["seed"],
        "sample_seed": seed,
        "prompt_id": prompt_id,
        "interface": interface,
        "prompt_hash": prepared["audit"]["final_prompt_hash"],
        "generation_config_hash": canonical_hash(config["generation"]),
        "sample_key": sample_identity(run_id, prompt_id, step, seed, index),
        "raw_rewards_by_channel": channels,
        "reward_sum": channels["sum"],
        "old_logprobs": old,
        "base_token_logprobs": reference,
        "logprob_behavior_sum": sum(behavior),
        "logprob_base_sum": sum(reference),
        "behavior_teacher_forcing_mean_error": error,
        "on_policy_likelihood_passed": error <= 0.02,
    }


def _collect_group(adapter, store, scene, interface, prepared, step, group_id, config, telemetry):
    before = parameter_hash(adapter.model, trainable=True)
    group = []
    for index in range(config["training"]["K"]):
        seed = int(
            canonical_hash(
                {
                    "sample_seed": config.get("sample_seed", 104729),
                    "group_id": group_id,
                    "index": index,
                }
            )[:8],
            16,
        ) % (2**31)
        prompt_id = canonical_hash({"scene": scene["base_scene_id"], "interface": interface})
        key = sample_identity(store.out.name, prompt_id, step, seed, index)
        row = store.records.get(key)
        if row is None:
            row = _new_sample(
                adapter,
                prepared,
                scene,
                interface,
                seed,
                step,
                index,
                store.out.name,
                config,
                telemetry,
            )
            row = {**row, "group_id": group_id, "K": config["training"]["K"]}
            store.append(row)
        if row["adapter_hash"] != before:
            raise ValueError("Rollout bank policy differs from restored optimizer checkpoint")
        for field in ("input_tensor_hash", "tokenized_prompt_hash"):
            if row.get(field) != prepared["audit"].get(field):
                raise ValueError(f"Rollout bank {field} differs from current processor input")
        group.append({**row, "prepared": prepared})
    after = parameter_hash(adapter.model, trainable=True)
    if before != after:
        raise RuntimeError("Policy changed during staged K generation")
    return group


def _group_audit(group):
    stats = grouped_advantages([row["reward_sum"] for row in group])
    return {
        **stats,
        "group_id": group[0]["group_id"],
        "K": len(group),
        "contains_X": any(row["category"] == "X" for row in group),
        "contains_S": any(row["category"] == "S" for row in group),
        "contains_I": any(row["category"] == "I" for row in group),
        "sample_keys": [row["sample_key"] for row in group],
    }


def _real_update(adapter, optimizer, groups, config, telemetry):
    telemetry.reset()
    start = time.perf_counter()
    train = config["training"]
    result = perform_update(
        adapter,
        optimizer,
        groups,
        lnorm=train["Lnorm"],
        clip_epsilon=train["clip_epsilon"],
        grad_clip=train["grad_clip"],
    )
    return {
        **result,
        **telemetry.snapshot(),
        "update_seconds": time.perf_counter() - start,
        "group_statistics": [_group_audit(group) for group in groups],
        "arm": "X_BASE",
        "epsilon_convention": "sqrt(population_variance + epsilon^2)",
        "loss_reduction": "sum_generated_tokens/(B*K*64)",
        "beta_kl": 0.0,
    }


def _replay_audit(adapter, optimizer, origin, expected, groups, config, telemetry, out, identity):
    path = out / "resume_probe.pt"
    save_checkpoint(path, origin, identity)
    candidates = []
    starts = []
    updates = []
    for _repeat in range(2):
        checkpoint = load_checkpoint(path, identity)
        restore_state(adapter.model, optimizer, checkpoint)
        starts.append(
            {
                "parameters": parameter_hash(adapter.model, trainable=True),
                "optimizer": state_hash(optimizer.state_dict()),
                "rng": state_hash(capture_state(adapter.model, optimizer)["rng"]),
            }
        )
        updates.append(_real_update(adapter, optimizer, groups, config, telemetry))
        candidates.append(capture_state(adapter.model, optimizer, expected["metadata"]))
    comparisons = [compare_parameters(expected, candidate) for candidate in candidates]
    same_starts = starts[0] == starts[1]
    optimizer_equal = all(
        state_hash(candidate["optimizer"]) == state_hash(expected["optimizer"])
        for candidate in candidates
    )
    rng_equal = all(
        state_hash(candidate["rng"]) == state_hash(expected["rng"]) for candidate in candidates
    )
    result = {
        "same_lambda": 0.0,
        "identical_start_hashes": same_starts,
        "starts": starts,
        "comparisons_to_uninterrupted": comparisons,
        "optimizer_state_equal": optimizer_equal,
        "rng_state_equal": rng_equal,
        "replay_updates": updates,
        "lambda_zero_auxiliary_parameter_difference": max(
            item["parameters_max_abs_difference"] for item in comparisons
        ),
        "lambda_zero_basis": (
            "same reward formula and same start; candidates are direct replays of X_BASE"
        ),
        "passed": same_starts
        and optimizer_equal
        and rng_equal
        and all(item["parameters_allclose"] for item in comparisons),
    }
    restore_state(adapter.model, optimizer, expected)
    write_json(out / "fork_resume_audit.json", result)
    return result


def _image_audit(adapter, prepared, panel, config):
    # Hold all text/cue/observation fixed; change only image bytes.
    scene = panel[0]
    replacement = next(item for item in panel if item["image_hash"] != scene["image_hash"])
    prompt = build_prompt(scene, "IMAGE_CUE_FRESH")
    changed = adapter.prepare(
        {**prompt, "image_path": replacement["image_path"]}, config["data_root"]
    )
    original = prepared[(scene["base_scene_id"], "IMAGE_CUE_FRESH")]
    left = (
        adapter.logprobs(original, [adapter.processor.tokenizer.eos_token_id])
        .detach()
        .cpu()
        .tolist()
    )
    left_repr = adapter.last_vision_hash
    right = (
        adapter.logprobs(changed, [adapter.processor.tokenizer.eos_token_id])
        .detach()
        .cpu()
        .tolist()
    )
    right_repr = adapter.last_vision_hash
    different_inputs = (
        original["audit"]["pixel_values_hash"] != changed["audit"]["pixel_values_hash"]
    )
    different_repr = left_repr is not None and right_repr is not None and left_repr != right_repr
    return {
        "text_held_fixed": original["audit"]["final_prompt"] == changed["audit"]["final_prompt"],
        "original_pixel_hash": original["audit"]["pixel_values_hash"],
        "replacement_pixel_hash": changed["audit"]["pixel_values_hash"],
        "original_visual_representation_hash": left_repr,
        "replacement_visual_representation_hash": right_repr,
        "original_eos_logprob": left,
        "replacement_eos_logprob": right,
        "representation_changed": different_repr,
        "passed": different_inputs and different_repr,
    }


def _check_kl_alarms(steps, config):
    threshold = config.get("diagnostics", {}).get("max_empirical_sequence_kl", 1.0)
    if any(step["empirical_same_training_bank_kl"] > threshold for step in steps):
        raise RuntimeError(
            "P1 empirical same-training-bank sequence KL exceeded the preregistered alarm; "
            "checkpoint and abnormal batch preserved; resume cannot bypass this terminal alarm"
        )


def run_smoke(
    config, scenes, out, model_key, resume=False, *, _adapter_factory=None, _telemetry=None
):
    """Execute real P1 by default; private injection supports CPU fixture tests only."""
    import torch

    from .model_adapters import load_adapter

    if _adapter_factory is None and not torch.cuda.is_available():
        raise RuntimeError(
            "P1 awaits an authorized NTU CUDA GPU; no model download or remote access attempted"
        )
    if len(scenes) != 18 or any(scene["split"] != "calibration" for scene in scenes):
        raise ValueError("P1 must use exactly 18 calibration scenes, never train/dev/confirm")
    _verify_scene_images(scenes, config["data_root"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    telemetry = _telemetry or Telemetry()
    config = {**config, "_model_key": model_key}
    config_hash = config.get("locked_config_hash") or canonical_hash(
        {k: v for k, v in config.items() if k != "_model_key"}
    )
    data_hash = canonical_hash(scenes)
    spec = config["models"][model_key]
    if resume:
        previous_lock = json.loads((out / "runtime_lock.json").read_text())
        if (
            previous_lock["identity"]["config_hash"] != config_hash
            or previous_lock["identity"]["data_hash"] != data_hash
        ):
            raise ValueError("Resume data/config changed; refusing model load")
        spec = {**spec, "revision": previous_lock["model_revision"]}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _seed_everything(config["training"]["seed"])
    adapter = (_adapter_factory or load_adapter)(model_key, spec, image_token_limit=768)
    identity = {
        "model_hash": canonical_hash(
            {
                "id": adapter.model_id,
                "revision": adapter.revision,
                "processor_hash": adapter.audit.get("processor_hash"),
                "tokenizer_hash": adapter.audit.get("tokenizer_hash"),
            }
        ),
        "data_hash": data_hash,
        "config_hash": config_hash,
    }
    store = RunStore(out, identity, resume=resume)
    lock = _runtime_lock(config, adapter, out, identity)
    if hasattr(adapter.processor, "save_pretrained"):
        adapter.processor.save_pretrained(out / "processor_snapshot")
    train = config["training"]
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=train["lr"],
        betas=tuple(train["betas"]),
        eps=train["adam_eps"],
        weight_decay=0.0,
    )
    checkpoint_path = out / "checkpoint.pt"
    completed_step = 0
    checkpoint = None
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, identity)
        completed_step = checkpoint["metadata"]["step"]
    # Initial compatibility always uses the identical seeded step-zero adapter.
    prepared = {
        (scene["base_scene_id"], interface): adapter.prepare(
            build_prompt(scene, interface), config["data_root"]
        )
        for scene in scenes
        for interface in INTERFACES
    }
    initial_groups = [
        _collect_group(
            adapter,
            store,
            scene,
            interface,
            prepared[(scene["base_scene_id"], interface)],
            0,
            f"initial:{scene['base_scene_id']}:{interface}",
            config,
            telemetry,
        )
        for scene in scenes
        for interface in INTERFACES
    ]
    cache = {}
    for interface in INTERFACES:
        group = next(group for group in initial_groups if group[0]["interface"] == interface)
        cache[interface] = audit_model_cache(adapter, group[0]["prepared"], group[0]["token_ids"])
    write_json(out / "cache_audit.json", cache)
    image = _image_audit(adapter, prepared, scenes, config)
    write_json(out / "image_influence_audit.json", image)
    if not all(row["on_policy_likelihood_passed"] for group in initial_groups for row in group):
        raise RuntimeError(
            "P1 sampling/teacher-forcing consistency alarm; no optimizer update allowed"
        )
    if not all(item["passed"] for item in cache.values()) or not image["passed"]:
        raise RuntimeError("P1 cache/image compatibility failed; no optimizer update allowed")
    base_before = parameter_hash(adapter.model, trainable=False)
    if checkpoint is not None:
        restore_state(adapter.model, optimizer, checkpoint)
    else:
        checkpoint = capture_state(
            adapter.model,
            optimizer,
            {
                "step": 0,
                "sample_keys": sorted(store.keys),
                "sampler_state": {"next_update": 0},
                "training_steps": [],
            },
        )
        save_checkpoint(checkpoint_path, checkpoint, identity)
    steps = list(checkpoint["metadata"].get("training_steps", []))
    _check_kl_alarms(steps, config)
    fork = (
        json.loads((out / "fork_resume_audit.json").read_text())
        if (out / "fork_resume_audit.json").exists()
        else None
    )
    for step in range(completed_step, train["smoke_updates"]):
        groups = []
        for position in range(train["B"]):
            scene = scenes[(step * train["B"] + position) % len(scenes)]
            interface = INTERFACES[position % 2]
            groups.append(
                _collect_group(
                    adapter,
                    store,
                    scene,
                    interface,
                    prepared[(scene["base_scene_id"], interface)],
                    step,
                    f"update:{step}:{position}",
                    config,
                    telemetry,
                )
            )
        if not all(row["on_policy_likelihood_passed"] for group in groups for row in group):
            raise RuntimeError(
                "P1 current update bank failed sampling/teacher-forcing consistency; "
                "failed samples and previous checkpoint preserved; no optimizer update allowed"
            )
        origin = capture_state(
            adapter.model, optimizer, {"step": step, "sample_keys": sorted(store.keys)}
        )
        update = _real_update(adapter, optimizer, groups, config, telemetry)
        steps.append({**update, "optimizer_step": step + 1})
        expected = capture_state(
            adapter.model,
            optimizer,
            {
                "step": step + 1,
                "sample_keys": sorted(store.keys),
                "sampler_state": {"next_update": step + 1},
                "training_steps": steps,
            },
        )
        if step == 1:
            fork = _replay_audit(
                adapter, optimizer, origin, expected, groups, config, telemetry, out, identity
            )
        save_checkpoint(checkpoint_path, expected, identity)
        write_json(out / "training_steps.json", steps)
        _check_kl_alarms(steps, config)
    base_after = parameter_hash(adapter.model, trainable=False)
    adapter.model.eval()
    all_rows = list(store.records.values())
    checks = {
        "sampling_likelihood": all(row["on_policy_likelihood_passed"] for row in all_rows),
        "cache": all(item["passed"] for item in cache.values()),
        "image": image["passed"],
        "frozen_base": base_before == base_after,
        "nonempty_gradients_and_measured_step": any(
            step["actual_step_norm"] > 0 and step["grad_norm_preclip"] > 0 for step in steps
        ),
        "fork_resume": fork is not None and fork["passed"],
        "real_optimizer_steps": len(steps) == train["smoke_updates"],
    }
    elapsed = sum(row["elapsed_seconds"] for row in all_rows)
    token_count = sum(row["completion_length"] for row in all_rows)
    result = {
        **adapter.audit,
        "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE" if _adapter_factory else "REAL_CUDA_MODEL",
        "checks": checks,
        "passed": all(checks.values()),
        "runtime_lock": lock,
        "raw_sample_count": len(all_rows),
        "initial_prompt_count": len(initial_groups),
        "initial_rollout_count": sum(map(len, initial_groups)),
        "training_steps": steps,
        "cache_audit": cache,
        "image_audit": image,
        "fork_resume_audit": fork,
        "frozen_base_hash_before": base_before,
        "frozen_base_hash_after": base_after,
        "generated_tokens": token_count,
        "generation_seconds": elapsed,
        "generation_tokens_per_second": token_count / elapsed if elapsed else None,
        "generation_calls_this_invocation": adapter.generation_calls,
        "forward_calls_this_invocation": adapter.forward_calls,
        "backward_calls_including_replays": sum(step["backward_calls"] for step in steps)
        + sum(step["backward_calls"] for step in (fork or {}).get("replay_updates", [])),
        "peak_single_generation_cuda_bytes": max(
            (row["peak_cuda_bytes"] or 0 for row in all_rows), default=0
        ),
        "peak_staged_K_generation_cuda_bytes": max(
            (row["peak_cuda_bytes"] or 0 for row in all_rows), default=0
        ),
        "peak_backward_cuda_bytes": max(
            (step["peak_cuda_bytes"] or 0 for step in steps), default=0
        ),
        **telemetry.snapshot(),
        "later_phases": "NOT_RUN; P1 evidence must be reviewed before pilot",
        "budget_estimation": {
            "seconds_per_prompt_K8": elapsed / (len(all_rows) / 8) if all_rows else None,
            "mean_update_seconds": sum(step["update_seconds"] for step in steps) / len(steps)
            if steps
            else None,
            "extrapolation_is_not_a_runtime_guarantee": True,
        },
    }
    if hasattr(adapter.model, "save_pretrained"):
        adapter.model.save_pretrained(out / "adapter_final")
    if result["passed"]:
        write_json(
            out / "validated_runtime_lock.json",
            {**lock, "runtime_status": "P1_PASSED", "execution_kind": result["execution_kind"]},
        )
    write_json(out / "model_audit.json", result)
    return result
