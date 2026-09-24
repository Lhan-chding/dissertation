"""Prospective on-policy branches with eight-update durable transactions.

Only committed segment attempts are authoritative. All failed attempts retain raw
outputs, but are excluded from sample counts; resume repeats at most eight updates.
The historical PPO/Adam path is shared through an explicit reward callback.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from ..decision_modeling.block_runner import _restore_checked, _validate_complete, update_batch
from ..decision_modeling.reward_recipes import BASELINES, RECIPES, compute_advantages, reward_vector
from ..grpo_update import grouped_advantages
from ..modeling_v3.io import canonical_hash
from ..modeling_v3.vlm_observation import (
    ObservationFault,
    PrefixObservationBackend,
    _logps,
    validate_action,
)
from ..modeling_v4 import gpu_collect as gpu
from ..optimizer_fork import state_hash

ALL_RECIPES = (*RECIPES, *BASELINES, "DIRECT_REPAIR_R4")


class GenerationBackend(PrefixObservationBackend):
    """Validated inherited prefix generation without redundant full rescore.

    Behavior scores remain the old-policy PPO denominator. Smoke independently
    checks them against the inherited scorer; evaluation needs no policy scores.
    """

    def generate(self, prompt, *, seed, max_new_tokens=64):
        from .semantics import semantic_features

        if self.current_fingerprint is None:
            raise ValueError("A bound live policy identity is required")
        prepared = self._prepared(prompt)
        with self._measurement():
            raw = self.adapter.generate(
                prepared, seed=seed, max_new_tokens=max_new_tokens, do_sample=True
            )
            try:
                flags = validate_action(
                    raw,
                    eos_ids=self.adapter.eos_ids,
                    max_new_tokens=max_new_tokens,
                    tokenizer=self.adapter.processor.tokenizer,
                )
                behavior = _logps(raw["behavior_token_logprobs"], len(raw["token_ids"]))
            except (KeyError, ValueError) as exc:
                raise ObservationFault(str(exc), raw) from exc
        semantic = semantic_features(raw["raw_completion"], prompt)
        self.counters["generated_sequences"] += 1
        self.counters["generated_tokens"] += len(raw["token_ids"])
        self.counters["accepted_samples"] += 1
        return {
            **raw,
            **flags,
            "semantic": semantic,
            "event": semantic["event"],
            "category": semantic["event"],
            "reward_vector": reward_vector(semantic),
            "prompt_id": prompt["prompt_id"],
            "base_scene_id": prompt["base_scene_id"],
            "family": prompt["family"],
            "interface": prompt["interface"],
            "prompt_record_hash": canonical_hash(prompt),
            "input_audit": prepared["audit"],
            "old_logprobs": behavior,
            "generation_sequence_logp": sum(behavior),
            "inference_fingerprint": self.current_fingerprint,
            "probability_execution": self.adapter.audit["probability_execution"],
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": sorted(self.adapter.eos_ids),
            "pad_token_id": self.adapter.pad_id,
            "runtime_identity": self.runtime_identity,
            "extra_rescoring": False,
        }


def make_backend(runtime):
    return GenerationBackend(
        runtime["adapter"],
        runtime_identity=runtime["identity"],
        policy_loader=runtime["policy_loader"],
        data_root=runtime.get("data_root"),
        parity_tolerances=runtime["parity_tolerances"],
        state_guard=runtime["state_guard"],
    )


def recipe_advantages(rewards, groups, *, recipe, epsilon=1e-4):
    """Four-channel methods remain byte-for-byte inherited; direct repair is named."""
    if recipe != "DIRECT_REPAIR_R4":
        return compute_advantages(rewards, recipe, epsilon=epsilon)
    baseline = compute_advantages(rewards, "R4", epsilon=epsilon)
    damage = np.asarray(
        [[row["semantic"]["Bdamage"] for row in group] for group in groups], dtype=float
    )
    if (
        damage.shape != np.asarray(rewards).shape[:2]
        or not np.isfinite(damage).all()
        or ((damage < 0) | (damage > 1)).any()
    ):
        raise ValueError("Invalid direct repair damage channel")
    total = np.asarray(baseline["total_rewards"]) - 0.5 * damage
    statistics = [grouped_advantages(group, epsilon) for group in total]
    five = np.concatenate([np.asarray(rewards), damage[..., None]], axis=-1).tolist()
    return {
        **baseline,
        "recipe": recipe,
        "channels": ["X", "A", "V", "C", "Bdamage"],
        "original_reward_vectors": baseline["reward_vectors"],
        "reward_vectors": five,
        "five_channel_rewards": five,
        "priorities": [2.0, 0.0, 0.5, 0.5, -0.5],
        "actual_weights": [2.0, 0.0, 0.5, 0.5, -0.5],
        "Bdamage": damage.tolist(),
        "damage_weight": -0.5,
        "total_rewards": total.tolist(),
        "group_statistics": statistics,
        "advantages": [s["advantages"] for s in statistics],
    }


class CommonScheduleSampler:
    def __init__(
        self,
        runtime,
        prompts,
        schedule,
        *,
        lineage_id,
        repeat,
        experiment_seed,
        origin_id,
        recipe,
        role="continuation",
    ):
        self.runtime, self.by_id = runtime, {p["prompt_id"]: p for p in prompts}
        if len(self.by_id) != len(prompts) or any(p.get("split") != "train" for p in prompts):
            raise ValueError("Unique train-only prompts required")
        self.schedule = copy.deepcopy(schedule)
        flattened = [pid for step in schedule["train_steps"] for pid in step]
        if (
            any(len(step) != 4 for step in schedule["train_steps"])
            or len(set(flattened)) != len(flattened)
            or not set(flattened) <= self.by_id.keys()
        ):
            raise ValueError("Common schedule must be B4 without replacement")
        self.lineage_id, self.repeat, self.experiment_seed = lineage_id, repeat, experiment_seed
        self.origin_id, self.recipe, self.role = origin_id, recipe, role
        self.backend, self.state_identity = make_backend(runtime), None
        self.last_cost = {}

    def __call__(self, step, ledger):
        from .sampler import training_seed

        sampler = self.runtime["sampler"]
        if sampler.get("position") != 4 * (step - 1) or (
            sampler.get("schedule_hash") != self.schedule["schedule_hash"]
        ):
            raise ValueError("Common schedule cursor/identity mismatch")
        if not self.state_identity:
            raise ValueError("Missing live policy identity")
        self.backend.current_fingerprint = self.state_identity
        before = copy.deepcopy(self.backend.counters)
        groups = []
        for pid in self.schedule["train_steps"][step - 1]:
            prompt, group = self.by_id[pid], []
            for draw in range(8):
                seed = training_seed(
                    experiment_seed=self.experiment_seed,
                    lineage=self.lineage_id,
                    repeat=self.repeat,
                    step=step,
                    prompt_id=pid,
                    draw=draw,
                )
                raw = self.backend.generate(prompt, seed=seed)
                row = {
                    **raw,
                    "sample_key": canonical_hash(
                        [self.origin_id, self.recipe, self.repeat, self.role, step, pid, draw]
                    ),
                    "sample_seed": seed,
                    "draw_index": draw,
                    "step": step,
                    "origin_id": self.origin_id,
                    "lineage_id": self.lineage_id,
                    "recipe": self.recipe,
                    "repeat": self.repeat,
                    "split": "train",
                    "role": "train",
                    "schedule_id": self.schedule["schedule_id"],
                }
                ledger(row)
                group.append({**row, "prepared": self.backend._prepared(prompt)})
            groups.append(group)
        sampler.update(position=4 * step, step=step)
        self.last_cost = {k: self.backend.counters[k] - before[k] for k in before}
        return groups


def _read(path):
    return json.loads(Path(path).read_text())


def run_training(
    runtime,
    initial_state,
    sample_groups,
    *,
    recipe,
    steps,
    out,
    identity,
    checkpoints=(0, 8, 32),
    resume=False,
    evaluate=None,
    fixture=False,
):
    """Execute source or branch, publishing complete Adam state every eight steps."""
    import torch

    if recipe not in ALL_RECIPES or type(steps) is not int or steps <= 0:
        raise ValueError("Positive steps and registered recipe required")
    if not fixture and not isinstance(sample_groups, CommonScheduleSampler):
        raise ValueError("Production requires the live-policy common sampler")
    if not fixture and runtime["identity"].get("execution_kind") != "REAL_CUDA_MODEL":
        raise ValueError("Production training requires REAL_CUDA_MODEL")
    _validate_complete(initial_state)
    root = Path(out)
    manifest = {
        "kind": "PROSPECTIVE_TRAINING",
        "identity": identity,
        "recipe": recipe,
        "steps": steps,
        "initial_state_hash": state_hash(initial_state),
        "runtime_identity": runtime["identity"],
        "config_hash": runtime.get("prospective_config_hash"),
        "B": 4,
        "K": 8,
        "Lnorm": 64,
        "checkpoint_interval": 8,
        "fixture": fixture,
    }
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("Training output exists; explicit resume required")
    root.mkdir(parents=True, exist_ok=True)
    gpu._publish(root / "MANIFEST.json", manifest)
    checkpoints = sorted(set(checkpoints) | {0, steps})
    commit_paths = sorted((root / "segments").glob("*/COMMIT.json"))
    commits = [_read(path) for path in commit_paths]
    expected_start = 1
    for commit in commits:
        if commit["start"] != expected_start or commit["manifest_hash"] != canonical_hash(manifest):
            raise ValueError("Noncontiguous/foreign segment commit")
        gpu.verify_artifact_bindings(commit)
        expected_start = commit["stop"] + 1
    if commits and commits[-1]["stop"] > steps:
        raise ValueError("Cannot truncate completed training")
    start_binding = gpu._save_state(
        root / "checkpoints" / "H00.pt",
        initial_state,
        {"manifest_hash": canonical_hash(manifest), "step": 0},
    )
    checkpoint = commits[-1]["checkpoint"] if commits else start_binding
    state = runtime["checkpoint_cache"].load(checkpoint)
    _restore_checked(runtime, state)
    endpoints = {0: start_binding}
    for commit in commits:
        if commit["stop"] in checkpoints:
            endpoints[commit["stop"]] = commit["checkpoint"]

    def publish_endpoint(step, binding):
        gpu._publish(root / f"H{step:02d}.json", {"step": step, "checkpoint": binding})
        if evaluate is not None and step > 0 and not (root / f"EVAL_H{step:02d}.json").exists():
            from .evaluation import isolated_evaluation

            def at_endpoint():
                _restore_checked(runtime, runtime["checkpoint_cache"].load(binding))
                return evaluate(runtime, binding, step)

            result = isolated_evaluation(runtime, at_endpoint)
            gpu._publish(root / f"EVAL_H{step:02d}.json", result)

    for step, binding in endpoints.items():
        publish_endpoint(step, binding)
    cursor = expected_start
    while cursor <= steps:
        # Also stop at named source anchors (24,32,88,96).
        stop = min(cursor + 7, steps, min((c for c in checkpoints if c >= cursor), default=steps))
        segment = root / "segments" / f"{cursor:03d}_{stop:03d}"
        attempt_root = segment / f"attempt_{time.time_ns()}"
        attempt_root.mkdir(parents=True)
        rows_count, step_audits = 0, []
        began = time.perf_counter()
        device = next(runtime["adapter"].model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            with (attempt_root / "samples.jsonl").open("x") as stream:

                def ledger(row):
                    nonlocal rows_count
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    rows_count += 1

                for step in range(cursor, stop + 1):
                    sample_groups.state_identity = canonical_hash(
                        [canonical_hash(manifest), checkpoint["state_hash"], step]
                    )
                    began_sampling = time.perf_counter()
                    groups = sample_groups(step, ledger)
                    if len(groups) != 4 or any(len(group) != 8 for group in groups):
                        raise ValueError("Exactly B4/K8 required")
                    sampling_seconds = time.perf_counter() - began_sampling
                    began_update = time.perf_counter()
                    audit = update_batch(
                        runtime, groups, recipe, advantage_callback=recipe_advantages
                    )
                    audit.update(
                        step=step,
                        sampling_seconds=sampling_seconds,
                        update_seconds=time.perf_counter() - began_update,
                        sample_cost=copy.deepcopy(sample_groups.last_cost),
                    )
                    gpu._publish(attempt_root / f"UPDATE_{step:03d}.json", audit)
                    step_audits.append(audit)
                os.fsync(stream.fileno())
            post = runtime["state"].capture(
                {
                    **initial_state["metadata"],
                    "checkpoint_step": stop,
                    "prospective_step": stop,
                    "recipe": recipe,
                }
            )
            checkpoint = gpu._save_state(
                attempt_root / "state.pt",
                post,
                {"manifest_hash": canonical_hash(manifest), "step": stop},
            )
            commit = {
                "manifest_hash": canonical_hash(manifest),
                "start": cursor,
                "stop": stop,
                "checkpoint": checkpoint,
                "samples": gpu._binding(attempt_root / "samples.jsonl"),
                "updates": [
                    gpu._binding(attempt_root / f"UPDATE_{s:03d}.json")
                    for s in range(cursor, stop + 1)
                ],
                "training_outputs": rows_count,
                "elapsed_seconds": time.perf_counter() - began,
                "sampling_seconds": sum(a["sampling_seconds"] for a in step_audits),
                "update_seconds": sum(a["update_seconds"] for a in step_audits),
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            }
            gpu._publish(segment / "COMMIT.json", commit)
            commits.append(commit)
        except BaseException as exc:
            from ..modeling_v3.vlm_observation import _fault_json

            gpu._publish(
                attempt_root / "FAILURE.json",
                _fault_json(
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "last_committed_step": cursor - 1,
                        "authoritative": False,
                        "raw_result": getattr(exc, "raw_result", None),
                    }
                ),
            )
            raise
        if stop in checkpoints:
            endpoints[stop] = checkpoint
            publish_endpoint(stop, checkpoint)
        cursor = stop + 1
    result = {
        "status": "CPU_FIXTURE_COMPLETE" if fixture else "TRAINING_COMPLETE",
        "identity": identity,
        "steps": steps,
        "training_outputs": sum(c["training_outputs"] for c in commits),
        "checkpoints": {str(k): v for k, v in endpoints.items()},
        "checkpoint": checkpoint,
        "elapsed_training_seconds": sum(c["elapsed_seconds"] for c in commits),
        "sampling_seconds": sum(c["sampling_seconds"] for c in commits),
        "update_seconds": sum(c["update_seconds"] for c in commits),
        "authoritative_segments": [
            str(p) for p in sorted((root / "segments").glob("*/COMMIT.json"))
        ],
    }
    gpu._publish(root / "COMPLETE.json", result)
    return result


def run_branch(
    runtime,
    origin,
    *,
    prompts,
    schedule,
    lineage_id,
    origin_id,
    recipe,
    repeat,
    out,
    experiment_seed=20260924,
    steps=32,
    resume=False,
    evaluate=None,
    fixture=False,
    role="development",
    authorization_receipt=None,
):
    if role == "locked_test" and authorization_receipt is None:
        raise PermissionError("Final test branches require registry authorization after decision")
    if not fixture and (len(prompts) != 192 or len(schedule["train_steps"]) != 32 or steps != 32):
        raise ValueError("Production branch requires 192-prompt pool and common 32-step schedule")
    state = runtime["checkpoint_cache"].load(origin)
    _restore_checked(runtime, state)
    parent_sampler = copy.deepcopy(state["sampler"])
    runtime["sampler"].clear()
    runtime["sampler"].update(
        role="continuation",
        position=0,
        step=0,
        schedule_id=schedule["schedule_id"],
        schedule_hash=schedule["schedule_hash"],
        parent_source_sampler=parent_sampler,
    )
    branch_state = runtime["state"].capture(
        {
            **state["metadata"],
            "origin_id": origin_id,
            "origin_checkpoint_step": state["metadata"].get("checkpoint_step"),
            "checkpoint_step": 0,
            "lineage_id": lineage_id,
            "repeat": repeat,
        }
    )
    sampler = CommonScheduleSampler(
        runtime,
        prompts,
        schedule,
        lineage_id=lineage_id,
        repeat=repeat,
        experiment_seed=experiment_seed,
        origin_id=origin_id,
        recipe=recipe,
    )
    return run_training(
        runtime,
        branch_state,
        sampler,
        recipe=recipe,
        steps=steps,
        out=out,
        identity={
            "origin_id": origin_id,
            "lineage_id": lineage_id,
            "recipe_id": recipe,
            "repeat": repeat,
            "schedule_id": schedule["schedule_id"],
            "role": role,
            "authorization_receipt": authorization_receipt,
        },
        checkpoints=(0, 8, steps),
        resume=resume,
        evaluate=evaluate,
        fixture=fixture,
    )
