"""Real adapter-integrated on-policy branches with atomic, complete-state recovery.

Historical Qwen prefix generation and PPO objective are reused without changing
historical modules. Only LoRA/Adam/RNG/sampler and forward state are checkpointed;
the base is guarded by tensor versions, not hashed on every training step.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path

from .. import followup_updates as complete
from .. import fork_gradients as legacy
from ..grpo_update import torch_ppo_loss
from ..modeling_v3.io import canonical_hash
from ..modeling_v4 import gpu_collect as gpu
from ..optimizer_fork import state_hash
from .reward_recipes import BASELINES, RECIPES, compute_advantages, reward_vector


def _publish(path, value):
    gpu._publish(Path(path), value)


def _read(path):
    return json.loads(Path(path).read_text())


def _validate_complete(state):
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
    if not required <= state.keys() or state.get("schema") not in {
        "ssvc-v4-complete-trainable-state-1",
        complete.SCHEMA,
    }:
        raise ValueError("A real complete LoRA/Adam/RNG/sampler checkpoint is required")
    if state["sampler"] is None or state["scheduler"] is not None:
        raise ValueError("Inherited sampler and no-scheduler contract required")
    complete._safe(state)


def _restore_checked(runtime, state):
    _validate_complete(state)
    runtime["state"].restore(state)
    now = runtime["state"].capture(state["metadata"])
    # Schema-specific derived fields differ; compare every persisted operational field.
    keys = (
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
    )
    if state_hash({k: now[k] for k in keys}) != state_hash({k: state[k] for k in keys}):
        raise RuntimeError("Branch start complete state restoration differs")


class SourceGroupSampler:
    """Continue original source prompt schedule; sample each arm's live policy."""

    def __init__(self, runtime, train_prompts, origin_state, *, origin_id, recipe):
        from ..modeling_v3.vlm_observation import PrefixObservationBackend
        from ..modeling_v4.data_adapter import build_training_schedule

        self.runtime, self.by_id = runtime, {p["prompt_id"]: p for p in train_prompts}
        if len(self.by_id) != len(train_prompts) or any(
            p.get("split", "train") != "train" for p in train_prompts
        ):
            raise ValueError("Unique train-only prompt records required")
        meta = origin_state["metadata"]
        self.origin_step = int(meta["checkpoint_step"])
        self.schedule = build_training_schedule(train_prompts, int(meta["seed"]))
        sampler = origin_state["sampler"]["state"]
        if (
            sampler.get("position") != 4 * self.origin_step
            or sampler.get("schedule_hash") != self.schedule["schedule_hash"]
        ):
            raise ValueError("Origin sampler position/hash differs from source training schedule")
        self.origin_id, self.recipe = origin_id, recipe
        self.backend = PrefixObservationBackend(
            runtime["adapter"],
            runtime_identity=runtime["identity"],
            policy_loader=lambda _: (_ for _ in ()).throw(
                RuntimeError("Training must not switch policy")
            ),
            data_root=runtime.get("data_root"),
            parity_tolerances=runtime["parity_tolerances"],
            state_guard=runtime["state_guard"],
        )
        self.state_identity = None

    def __call__(self, step, ledger):
        from .semantic_schema import semantic_features

        runtime = self.runtime
        absolute_step = self.origin_step + step
        sampler = runtime["sampler"]
        if sampler.get("position") != 4 * (absolute_step - 1):
            raise ValueError("Training sampler cursor mismatch")
        if absolute_step > len(self.schedule["train_steps"]):
            raise ValueError("Inherited source schedule is exhausted")
        if not self.state_identity:
            raise ValueError("Missing current live training-state identity")
        # Exact complete-state identity is deliberately more restrictive than an
        # inference alias. No cross-step/cross-arm generated output is reused.
        self.backend.current_fingerprint = self.state_identity
        groups = []
        before = copy.deepcopy(self.backend.counters)
        for pid in self.schedule["train_steps"][absolute_step - 1]:
            prompt, group = self.by_id[pid], []
            for draw in range(8):
                seed = random.getrandbits(63)
                raw = self.backend.generate(prompt, seed=seed, max_new_tokens=64)
                semantic = semantic_features(raw["raw_completion"], prompt)
                if semantic["event"] != raw["category"]:
                    raise ValueError("Inherited parser disagrees with frozen semantic schema")
                row = {
                    **raw,
                    "semantic": semantic,
                    "reward_vector": reward_vector(semantic),
                    "sample_key": canonical_hash([self.origin_id, self.recipe, step, pid, draw]),
                    "sample_seed": seed,
                    "draw_index": draw,
                    "step": step,
                    "origin_id": self.origin_id,
                    "recipe": self.recipe,
                    "split": "train",
                    "role": "train",
                    "old_logprobs": raw["behavior_token_logprobs"],
                }
                ledger(row)
                group.append({**row, "prepared": self.backend._prepared(prompt)})
            groups.append(group)
        sampler.update(
            step=absolute_step,
            position=4 * absolute_step,
            schedule_hash=self.schedule["schedule_hash"],
        )
        self.last_cost = {k: self.backend.counters[k] - before[k] for k in before}
        return groups


def update_batch(
    runtime, groups, recipe, *, epsilon=1e-4, lnorm=64, clip_epsilon=0.2, grad_clip=1.0
):
    """One true Adam update; inherited token PPO denominator, per-sample rewards."""
    import torch

    adapter, optimizer = runtime["adapter"], runtime["optimizer"]
    if complete._unusable(adapter):
        raise RuntimeError("Adapter was poisoned by failed state restoration")
    if not (lnorm > 0 and 0 < clip_epsilon < 1 and grad_clip > 0):
        raise ValueError("Invalid PPO/gradient clipping constants")
    runtime["state"].check_frozen()
    complete._ownership(adapter.model, optimizer)
    binding = legacy._bank_binding(groups)
    rewards = []
    for group in groups:
        vectors = []
        for row in group:
            vector = row.get("reward_vector") or reward_vector(row["semantic"])
            expected = reward_vector({"event": row["category"], "relation_score": vector[3]})
            if list(vector) != expected:
                raise ValueError("Stored reward vector disagrees with sampled semantic event")
            vectors.append(vector)
        rewards.append(vectors)
    stats = compute_advantages(rewards, recipe, epsilon=epsilon)
    params = legacy._parameters(adapter.model)
    modes = complete._forward_state(adapter.model, adapter)
    versions = {n: p._version for n, p in params.items()}
    records, loss_sum = [], 0.0
    forwards_before = adapter.forward_calls
    optimizer.zero_grad(set_to_none=True)
    try:
        for group, advantages in zip(groups, stats["advantages"], strict=True):
            for row, advantage in zip(group, advantages, strict=True):
                new = adapter.logprobs(row["prepared"], row["token_ids"], require_grad=True)
                old = legacy._old_tensor(row, new.device)
                records.append(legacy._ratio_record(new, old, row, clip_epsilon))
                loss, _ = torch_ppo_loss(
                    new[None],
                    old[None],
                    torch.tensor([advantage], device=new.device),
                    torch.ones_like(new[None], dtype=torch.bool),
                    lnorm=lnorm,
                    clip_epsilon=clip_epsilon,
                    total_sequences=binding["sequences"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite multi-reward PPO loss")
                loss.backward()
                loss_sum += float(loss.detach())
        if versions != {n: p._version for n, p in params.items()}:
            raise RuntimeError("Parameters changed during gradient computation")
        if any(p.grad is not None for p in adapter.model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameter received a gradient")
        for p in params.values():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if not torch.isfinite(p.grad).all():
                raise FloatingPointError("Nonfinite gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            list(params.values()), grad_clip, error_if_nonfinite=True
        )
        legacy._finite_adam_state(optimizer)
        optimizer.step()
        legacy._finite_adam_state(optimizer)
        if any(not torch.isfinite(p).all() for p in params.values()):
            raise FloatingPointError("Nonfinite trainable parameter after Adam")
        runtime["state"].check_frozen()
        return {
            "reward_statistics": stats,
            "bank": binding,
            "loss": loss_sum,
            "grad_norm_preclip": float(norm),
            "optimizer_updates": 1,
            "backward_calls": binding["sequences"],
            "ratio_records": records,
            "physical_forward_calls": adapter.forward_calls - forwards_before,
            "Lnorm": lnorm,
            "clip_epsilon": clip_epsilon,
            "grad_clip": grad_clip,
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
        try:
            complete._restore_forward(adapter.model, modes, adapter)
        except BaseException:
            complete._poison(adapter)
            raise


def _evaluate_isolated(runtime, evaluate, checkpoint, step):
    saved = runtime["state"].capture({"evaluation_isolation": step})
    try:
        if checkpoint is not None:
            _restore_checked(runtime, runtime["checkpoint_cache"].load(checkpoint))
        return evaluate(runtime, checkpoint, step)
    finally:
        # Restores all RNG, Adam, modes, position state and sampler as well as LoRA.
        _restore_checked(runtime, saved)


def run_block(
    runtime,
    origin_state,
    sample_groups,
    *,
    origin_id,
    recipe,
    steps,
    out,
    resume=False,
    evaluate=None,
    epsilon=1e-4,
    lnorm=64,
    clip_epsilon=0.2,
    grad_clip=1.0,
    fixture=False,
):
    """Run to H8/H32. ``sample_groups(step, ledger)`` must sample the live adapter.

    Every attempt retains raw rows and failures. A committed step is immutable;
    resumes restore the last durable checkpoint instead of replaying its update.
    """
    if recipe not in (*RECIPES, *BASELINES) or type(steps) is not int or steps not in (8, 32):
        raise ValueError("Known recipe and planned H8/H32 endpoint required")
    if not fixture and not isinstance(sample_groups, SourceGroupSampler):
        raise ValueError("Real blocks require the adapter-integrated SourceGroupSampler")
    if not fixture and not runtime.get("decision_config_hash"):
        raise ValueError("Real blocks require a frozen decision protocol hash")
    _validate_complete(origin_state)
    root = Path(out)
    manifest = {
        "kind": "DECISION_MODELING_BLOCK",
        "origin_id": origin_id,
        "recipe": recipe,
        "origin_state_hash": state_hash(origin_state),
        "runtime_identity": runtime["identity"],
        "decision_config_hash": runtime.get("decision_config_hash"),
        "epsilon": epsilon,
        "Lnorm": lnorm,
        "clip_epsilon": clip_epsilon,
        "grad_clip": grad_clip,
        "fixture": fixture,
        "normalization_version": "dm-rewards-v1",
    }
    if (
        root.exists()
        and any(p.name not in {".writer.lock", "DRY_RUN_block.json"} for p in root.iterdir())
        and not resume
    ):
        raise FileExistsError("Block output is nonempty; use resume with unchanged identity")
    root.mkdir(parents=True, exist_ok=True)
    _publish(root / "MANIFEST.json", manifest)
    commits = []
    for step in range(1, 33):
        path = root / "steps" / f"H{step:02d}" / "COMMIT.json"
        if path.exists():
            if step != len(commits) + 1:
                raise ValueError("Noncontiguous block commits")
            item = _read(path)
            if item["manifest_hash"] != canonical_hash(manifest):
                raise ValueError("Step belongs to another block identity")
            gpu.verify_artifact_bindings(item)
            commits.append(item)
    if len(commits) > steps:
        raise ValueError("Cannot shorten an already completed block")
    state = runtime["checkpoint_cache"].load(commits[-1]["checkpoint"]) if commits else origin_state
    _restore_checked(runtime, state)
    if commits:
        checkpoint = commits[-1]["checkpoint"]
    else:
        checkpoint = gpu._save_state(
            root / "checkpoints" / "H00.pt", state, {**manifest, "step": 0}
        )
    initial_checkpoint = gpu._save_state(
        root / "checkpoints" / "H00.pt", origin_state, {**manifest, "step": 0}
    )
    _publish(
        root / "H00.json",
        {"origin_state_hash": manifest["origin_state_hash"], "checkpoint": initial_checkpoint},
    )

    def finalize_endpoint(horizon, endpoint_checkpoint):
        _publish(
            root / f"H{horizon:02d}.json", {"step": horizon, "checkpoint": endpoint_checkpoint}
        )
        if evaluate is not None and not (root / f"EVAL_H{horizon:02d}.json").exists():
            try:
                result = _evaluate_isolated(runtime, evaluate, endpoint_checkpoint, horizon)
                _publish(root / f"EVAL_H{horizon:02d}.json", result)
            except BaseException as error:
                failed = root / f"EVAL_H{horizon:02d}_FAILURE_{time.time_ns()}.json"
                _publish(failed, {"type": type(error).__name__, "message": str(error)})
                raise

    # Recover a crash after durable update commit but before endpoint receipt/eval.
    for horizon in (8, 32):
        if horizon <= len(commits):
            finalize_endpoint(horizon, commits[horizon - 1]["checkpoint"])
    for step in range(len(commits) + 1, steps + 1):
        step_root = root / "steps" / f"H{step:02d}"
        step_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (step_root / f"attempt_{attempt:04d}").exists():
            attempt += 1
        attempt_root = step_root / f"attempt_{attempt:04d}"
        attempt_root.mkdir()
        started = time.perf_counter()
        import torch

        cuda_device = next(runtime["adapter"].model.parameters()).device
        if cuda_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cuda_device)
        try:
            sample_groups.state_identity = canonical_hash(
                [runtime["identity"], checkpoint["state_hash"]]
            )
            with (attempt_root / "samples.jsonl").open("x") as stream:

                def ledger(row):
                    stream.write(
                        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())

                groups = sample_groups(step, ledger)
            if not fixture and (len(groups) != 4 or any(len(g) != 8 for g in groups)):
                raise ValueError("D2 requires exactly B4/K8")
            sampling_seconds = time.perf_counter() - started
            update_start = time.perf_counter()
            audit = update_batch(
                runtime,
                groups,
                recipe,
                epsilon=epsilon,
                lnorm=lnorm,
                clip_epsilon=clip_epsilon,
                grad_clip=grad_clip,
            )
            update_seconds = time.perf_counter() - update_start
            _publish(attempt_root / "UPDATE.json", audit)
            post = runtime["state"].capture(
                {
                    **origin_state["metadata"],
                    "decision_block_step": step,
                    "decision_origin_id": origin_id,
                    "decision_recipe": recipe,
                }
            )
            checkpoint = gpu._save_state(
                attempt_root / "state.pt", post, {**manifest, "step": step}
            )
            commit = {
                "step": step,
                "manifest_hash": canonical_hash(manifest),
                "checkpoint": checkpoint,
                "samples": gpu._binding(attempt_root / "samples.jsonl"),
                "update": gpu._binding(attempt_root / "UPDATE.json"),
                "sampling_seconds": sampling_seconds,
                "update_seconds": update_seconds,
                "sample_cost": getattr(sample_groups, "last_cost", None),
                "training_outputs": sum(map(len, groups)),
                "peak_memory_allocated_bytes": (
                    torch.cuda.max_memory_allocated(cuda_device)
                    if cuda_device.type == "cuda"
                    else None
                ),
            }
            _publish(step_root / "COMMIT.json", commit)
            commits.append(commit)
            state = post
        except BaseException as error:
            fault = {
                "step": step,
                "type": type(error).__name__,
                "message": str(error),
                "raw_result": getattr(error, "raw_result", None),
                "last_committed_step": len(commits),
            }
            from ..modeling_v3.vlm_observation import _fault_json

            _publish(attempt_root / "FAILURE.json", _fault_json(fault))
            raise
        if step in (8, 32):
            finalize_endpoint(step, checkpoint)
    result = {
        "status": "CPU_FIXTURE_COMPLETE" if fixture else "BLOCK_COMPLETE",
        "origin_id": origin_id,
        "recipe": recipe,
        "steps": len(commits),
        "checkpoint": checkpoint,
        "training_outputs": sum(c["training_outputs"] for c in commits),
        "completed_horizons": [0] + [h for h in (8, 32) if h <= len(commits)],
        "evaluation": "SEPARATE_OBSERVE_COMMAND" if evaluate is None else "ISOLATED_CALLBACK",
    }
    _publish(root / f"RESULT_H{steps:02d}.json", result)
    return result


def run_from_runtime(
    runtime,
    origin,
    recipe,
    steps,
    out,
    *,
    train_prompts,
    origin_id,
    resume=False,
    evaluate=None,
    **kwargs,
):
    """Executable Qwen path: existing verified checkpoint binding + source prompts."""
    origin_state = runtime["checkpoint_cache"].load(origin)
    sampler = SourceGroupSampler(
        runtime, train_prompts, origin_state, origin_id=origin_id, recipe=recipe
    )
    return run_block(
        runtime,
        origin_state,
        sampler,
        origin_id=origin_id,
        recipe=recipe,
        steps=steps,
        out=out,
        resume=resume,
        evaluate=evaluate,
        **kwargs,
    )
