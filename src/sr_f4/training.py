"""Variable-G adapter around the unchanged native sequence objective."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path


def native_advantages(rewards):
    """Native float64 population std plus epsilon, extended to arbitrary G.

    Nonbinary inputs are accepted for preregistered offline scalar replay only.
    The production update separately requires the original binary J.
    """
    import numpy as np

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A finite nonempty reward group is required")
    std = values.std(ddof=0)
    return ((values - values.mean()) / (std + 1e-8) if std > 0 else values * 0).tolist()


def validate_groups(records, group_sizes):
    if len(records) != 128 or sum(group_sizes) != 128 or any(g not in (8, 32) for g in group_sizes):
        raise ValueError("Exactly 128 answers in independent G8/G32 groups required")
    offset, seen = 0, set()
    for size in group_sizes:
        group = records[offset : offset + size]
        key, qid = group[0]["group_id"], group[0]["qid"]
        if key in seen or any(r["group_id"] != key or r["qid"] != qid for r in group):
            raise ValueError("Group identity mismatch; repeated qids need distinct group IDs")
        seen.add(key)
        for row in group:
            if row["score"]["J"] not in (0, 1):
                raise ValueError("Training J is binary")
            if not 1 <= len(row["tokens"]) <= 1024 or len(row["tokens"]) != len(
                row["sampler_logprobs"]
            ):
                raise ValueError("Completion tokens and sampler probabilities do not align")
        offset += size


def update(
    runtime, optimizer, scheduler, records, group_sizes, reference, *, final_advantages=None
):
    """One fresh on-policy update, microbatch one, native /128 reduction.

    Failures never commit a step. The caller resumes its last atomic checkpoint
    and may reuse only that checkpoint's cached rollouts.
    """
    import torch

    from sr_f12.training import (
        enforce_probability_gate,
        probability_diagnostics,
        sequence_objective,
    )

    validate_groups(records, group_sizes)
    native, offset = [], 0
    for size in group_sizes:
        native.extend(native_advantages([r["score"]["J"] for r in records[offset : offset + size]]))
        offset += size
    final = native if final_advantages is None else list(final_advantages)
    if len(final) != 128 or not torch.isfinite(torch.tensor(final)).all():
        raise ValueError("128 finite final advantages required")
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    diffs, truncations, logps = [], [], []
    totals = dict(policy=0.0, kl=0.0, loss=0.0)
    previous_qid, prepared = None, None
    for record, advantage in zip(records, final, strict=True):
        # Reuse the same immutable native input tensors across that question's
        # microbatches. No model state, output, or decoder cache is reused.
        if record["qid"] != previous_qid:
            prepared = runtime.prepare_input(
                record["model_input"], record["run_root"], protocol=record["protocol"]
            )
            previous_qid = record["qid"]
        item = dict(record, prepared=prepared)
        with torch.no_grad():
            ref = runtime.batch_reference_forward([item], reference)[0]
        current = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)[0]
        sampler = torch.as_tensor(
            record["sampler_logprobs"], device=current.device, dtype=torch.float32
        )
        result = sequence_objective(current, sampler, ref, advantage)
        (result["loss"] / 128).backward()
        for name in totals:
            totals[name] += float(result[name].detach()) / 128
        diffs.append(result["absolute_difference"].cpu())
        truncations.append(result["tis_truncated"].cpu())
        logps.append(current.detach().cpu().tolist())
        del current, ref, result, item
    probabilities = probability_diagnostics(diffs, truncations)
    enforce_probability_gate(probabilities)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError("Nonfinite adapter after optimizer update")
    return dict(
        totals,
        **probabilities,
        gradient_norm_before_clip=float(norm),
        training_seconds=time.monotonic() - started,
        group_sizes=list(group_sizes),
        native_advantages=native,
        final_advantages=final,
        train_logprobs=logps,
        sequence_count=128,
        optimizer_updates=1,
        microbatch_size=1,
        loss_reduction="mean_of_128_sequence_token_means",
        kl_coefficient=0.02,
    )


def save_checkpoint(path, runtime, optimizer, scheduler, *, identity, step, sampler_state):
    """Tensor/primitive-only checkpoint; load always uses weights_only=True."""
    import numpy as np
    import torch

    from mm_core.training import _cpu_tree, trainable_state

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    state = dict(
        identity=identity,
        step=step,
        policy=trainable_state(runtime.model),
        optimizer=_cpu_tree(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),
        sampler_state=sampler_state,
        python_rng=random.getstate(),
        numpy_rng=[np_state[0], np_state[1].tolist(), *np_state[2:]],
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
    )
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, runtime, optimizer, scheduler, *, identity):
    import numpy as np
    import torch

    from sr_f1.runtime import copy_parameters

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("Checkpoint belongs to another registered path")
    copy_parameters(runtime.model, state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    from sr_f3_ce.training import validate_optimizer

    validate_optimizer(optimizer, scheduler, model=runtime.model)
    random.setstate(state["python_rng"])
    rng = state["numpy_rng"]
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"], state["sampler_state"]
