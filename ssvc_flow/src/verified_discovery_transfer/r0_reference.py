"""Fresh-Adam R0_RESET32; live O0 samples and the inherited on-policy PPO loss."""

from __future__ import annotations

import time
from pathlib import Path

from .queue import digest, read_json, write_json


def zero_gradient_adam_step(optimizer, parameters, lr):
    """Zero advantages still advance Adam moments and its update counter."""
    import torch

    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()


def run_r0(run, job, backend):
    import torch

    from ..followup_updates import _forward_state, _restore_forward
    from ..grpo_update import grouped_advantages, torch_ppo_loss
    from ..modeling_v3.vlm_observation import validate_action
    from .public_tasks import RoleDataset, compile_case, verify_raw
    from .sft_runtime import SFTRuntime

    run = Path(run)
    output = run / "evidence" / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    tasks = RoleDataset(run / "cohort", "teacher_train").public("T_train")
    schedule = read_json(run / "R0_PROMPT_SCHEDULES.json")
    if digest({k: v for k, v in schedule.items() if k != "digest"}) != schedule["digest"]:
        raise ValueError("Frozen R0 schedule modified")
    if (
        schedule["cohort_manifest_digest"]
        != RoleDataset(run / "cohort", "teacher_train").manifest["manifest_digest"]
    ):
        raise ValueError("R0 schedule cohort differs")
    by_id = {task["task_id"]: task for task in tasks}
    ids = schedule["repeats"][str(job["repeat"])]
    if len(ids) != 128 or len(set(ids)) != 128 or not set(ids) <= by_id.keys():
        raise ValueError("Frozen R0 train-only schedule is incomplete")
    tasks = [by_id[tid] for tid in ids]
    identity = {
        "kind": "R0_RESET32",
        "parent": job["parent"],
        "repeat": job["repeat"],
        "seed": job["seed"],
        "parent_checkpoint": backend.receipt["checkpoint"],
        "tasks_hash": digest(tasks),
        "settings": {
            "steps": 32,
            "B": 4,
            "K": 8,
            "reward": "2*X",
            "epsilon": 1e-4,
            "Lnorm": 64,
            "ppo_clip": 0.2,
            "fresh_AdamW": True,
            "warmup": 8,
        },
    }
    runtime = SFTRuntime.from_frozen_backend(backend, seed=job["seed"], view_identity=identity)
    # Only the focus sampler drives R0. The saved unused replay sampler is inert.
    runtime.bind_data(tasks, tasks)
    checkpoints = sorted(output.glob("step*.pt"))
    if checkpoints:
        runtime.resume(checkpoints[-1])
    else:
        runtime.save(output / "step000.pt")
    adapter = runtime.adapter
    parameters = [p for p in adapter.model.parameters() if p.requires_grad]
    receipts = []
    generated_sequences = generated_tokens = 0
    began = time.perf_counter()
    while runtime.step < 32:
        step = runtime.step + 1
        rows = []
        attempt = output / "attempts" / f"update{step:03d}_{time.time_ns()}.json"
        try:
            saved_forward = _forward_state(adapter.model, adapter)
            indices = runtime.focus_sampler.take(4)
            selected = [tasks[i] for i in indices]
            groups = []
            for task in selected:
                case = compile_case(task, "O0")
                prepared = adapter.prepare(case["prompt"], backend.data_root)
                group = []
                for draw in range(8):
                    seed = int(
                        digest([job["seed"], step, task["task_id"], draw, "R0_ON_POLICY"])[:16], 16
                    ) % (2**63 - 1)
                    atomic = (
                        output
                        / "rollouts"
                        / f"update{step:03d}"
                        / (digest([task["task_id"], draw]) + ".json")
                    )
                    request_identity = digest([identity, step, task["task_id"], draw, seed])
                    if atomic.exists():
                        receipt = read_json(atomic)
                        if (
                            receipt["identity"] != request_identity
                            or digest(receipt["raw"]) != receipt["raw_hash"]
                        ):
                            raise ValueError("Committed R0 atomic rollout changed")
                        raw = receipt["raw"]
                    else:
                        device = torch.device(adapter.device)
                        devices = [device.index or 0] if device.type == "cuda" else []
                        with torch.random.fork_rng(devices=devices), torch.no_grad():
                            raw = adapter.generate(
                                prepared, seed=seed, max_new_tokens=64, do_sample=True
                            )
                        write_json(
                            atomic,
                            {"identity": request_identity, "raw": raw, "raw_hash": digest(raw)},
                        )
                        generated_sequences += 1
                        generated_tokens += len(raw["token_ids"])
                    flags = validate_action(
                        raw,
                        eos_ids=adapter.eos_ids,
                        max_new_tokens=64,
                        tokenizer=adapter.processor.tokenizer,
                    )
                    verified = verify_raw(task, "O0", raw["raw_completion"])
                    row = {
                        "task_id": task["task_id"],
                        "update": step,
                        "draw": draw,
                        "seed": seed,
                        **raw,
                        **flags,
                        **verified,
                        "reward": 2.0 * verified["public_verifier_pass"],
                    }
                    rows.append(row)
                    group.append((prepared, row))
                groups.append(group)
            write_json(
                attempt,
                {"status": "ROLLOUT_COMPLETE_UNCOMMITTED", "rows": rows, "identity": identity},
            )
            _restore_forward(adapter.model, saved_forward, adapter)
            adapter._reset_positions()
            runtime.optimizer.zero_grad(set_to_none=True)
            advantages = [
                grouped_advantages([row["reward"] for _, row in group], 1e-4)["advantages"]
                for group in groups
            ]
            lr = 1e-5 * min(step / 8, 1)
            losses = []
            max_log_ratio = 0.0
            if all(a == 0 for group in advantages for a in group):
                zero_gradient_adam_step(runtime.optimizer, parameters, lr)
                grad_norm = 0.0
            else:
                for group, values in zip(groups, advantages, strict=True):
                    for (prepared, row), advantage in zip(group, values, strict=True):
                        new = adapter.logprobs(prepared, row["token_ids"], require_grad=True)
                        old = torch.tensor(
                            row["behavior_token_logprobs"], device=new.device, dtype=torch.float32
                        )
                        if new.shape != old.shape:
                            raise ValueError("Behavior/target token length mismatch")
                        max_log_ratio = max(max_log_ratio, float((new.detach() - old).abs().max()))
                        loss, _ = torch_ppo_loss(
                            new[None],
                            old[None],
                            torch.tensor([advantage], device=new.device),
                            torch.ones_like(new[None], dtype=torch.bool),
                            lnorm=64,
                            clip_epsilon=0.2,
                            total_sequences=32,
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite on-policy loss")
                        loss.backward()
                        losses.append(float(loss.detach()))
                for p in parameters:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError("Nonfinite R0 gradient")
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                )
                for group in runtime.optimizer.param_groups:
                    group["lr"] = lr
                runtime.optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("Nonfinite R0 weights")
            _restore_forward(adapter.model, saved_forward, adapter)
            runtime.optimizer.zero_grad(set_to_none=True)
            runtime.step = step
            for task in selected:
                runtime.exposures[task["task_id"]] = runtime.exposures.get(task["task_id"], 0) + 1
            stats = {
                "status": "UPDATE_APPLIED",
                "update": step,
                "lr": lr,
                "loss": sum(losses),
                "gradient_norm": grad_norm,
                "advantages": advantages,
                "zero_advantages": not losses,
                "behavior_log_ratio_max_abs": max_log_ratio,
                "rollout_path": str(attempt),
                "optimizer_updates": 1,
            }
            write_json(output / f"update{step:03d}.json", stats)
            checkpoint = runtime.save(output / f"step{step:03d}.pt")
            write_json(
                output / "LAST_COMMIT.json",
                {
                    "step": step,
                    "checkpoint": checkpoint,
                    "rollout_path": str(attempt),
                    "identity": identity,
                },
            )
            if step in (8, 32):
                receipts.append(checkpoint)
            # Retain registered endpoints plus the latest exact-resume state.
            previous = output / f"step{step - 1:03d}.pt"
            if step - 1 not in (0, 8, 32) and previous.exists():
                previous.unlink()
        except BaseException as exc:
            write_json(
                attempt,
                {
                    "status": "FAILED_ATTEMPT",
                    "rows": rows,
                    "identity": identity,
                    "error": repr(exc),
                },
            )
            raise
    result = {
        "status": "COMPLETE",
        "execution_kind": "REAL_CUDA_ON_POLICY_R0_RESET32",
        "identity": identity,
        "checkpoints": receipts,
        "optimizer_updates": 32,
        "generated_sequences_this_process": generated_sequences,
        "generated_tokens_this_process": generated_tokens,
        "gpu_seconds_this_process": time.perf_counter() - began,
        "not_equal_compute_to_SFT": True,
    }
    write_json(output / "result.json", result)
    return result


def freeze_prompt_schedules(run):
    """Freeze train-only stratification before model calls; labels stay outside runtime."""
    import random
    from collections import defaultdict

    from .public_tasks import RoleDataset

    run = Path(run)
    data = RoleDataset(run / "cohort", "gold")
    tasks = data.public("T_train")
    audit = {a["task_id"]: a for a in data.audit("T_train")}
    groups = defaultdict(list)
    for task in tasks:
        a = audit[task["task_id"]]
        groups[(task["family"], a["corrupted_index"], a["center"])].append(task["task_id"])
    repeats = {}
    for repeat, seed in enumerate((73101, 73102)):
        rng = random.Random(seed)
        chosen = []
        for (family, _j, _center), ids in sorted(groups.items(), key=lambda item: str(item[0])):
            chosen.extend(rng.sample(sorted(ids), 4 if family == "cross_series" else 16))
        rng.shuffle(chosen)
        if len(chosen) != 128:
            raise ValueError("R0 stratified schedule must contain 128 tasks")
        repeats[str(repeat)] = chosen
    receipt = {
        "repeats": repeats,
        "cohort_manifest_digest": data.manifest["manifest_digest"],
        "source": "T_train metadata only; no E/G",
    }
    receipt["digest"] = digest(receipt)
    write_json(run / "R0_PROMPT_SCHEDULES.json", receipt)
    return receipt
