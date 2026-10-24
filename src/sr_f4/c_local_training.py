"""Bounded C local-credit training with native transactional checkpoint recovery.

Each update preserves all 128 native outcome observations. Separate detached
semantic advantages act only on their auditable generated-token spans.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

from srf4_ref.sampling import seed_for

from . import c_local_credit as credit_method
from . import c_local_data as data
from .common import now, read, write
from .run import group, image_index, task_index
from .training import load_checkpoint, native_advantages, save_checkpoint, validate_groups

CREDIT_SCHEMA = "srf4-c-local-credit-v1"


def _registered_write(path, value):
    """A replay may reuse a record, but cannot replace its registered identity."""
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Registered local-credit record differs: {path}")
    else:
        write(path, value)


def _validate_credits(records, group_sizes, credits):
    """Bind detached local coefficients and masks to the unchanged native rows."""
    import math

    validate_groups(records, group_sizes)
    if group_sizes != [data.G] * data.GROUPS or len(credits) != data.GROUPS:
        raise ValueError("Local-credit training requires sixteen registered G8 groups")
    flattened, offset = [], 0
    for size, credit in zip(group_sizes, credits, strict=True):
        current = records[offset:offset + size]
        if credit["group_id"] != current[0]["group_id"] or credit["qid"] != current[0]["qid"]:
            raise ValueError("Credit group identity differs from rollout")
        native = native_advantages([row["score"]["J"] for row in current])
        if credit["native_advantages"] != native:
            raise ValueError("Local credit must preserve the original native J advantages")
        names = ("native_advantages", "evidence_advantages", "selection_advantages")
        if any(len(credit[name]) != size for name in names) or len(credit["record_masks"]) != size:
            raise ValueError("Credit rows and token-mask records must match G8")
        for i, record in enumerate(current):
            audit = credit["record_masks"][i]
            if audit["row_id"] != record["row_id"]:
                raise ValueError("Credit token-mask row identity differs")
            coefficients = [credit[name][i] for name in names]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in coefficients):
                raise ValueError("Local advantages must be finite detached numeric scalars")
            for name in ("evidence_mask", "selection_mask"):
                mask = audit[name]
                if len(mask) != len(record["tokens"]) or any(type(v) is not bool for v in mask):
                    raise ValueError("Generated-token boolean masks must match completion tokens")
            if any(a and b for a, b in zip(audit["evidence_mask"], audit["selection_mask"],
                                          strict=True)):
                raise ValueError("Evidence and selection spans must not overlap")
            flattened.append(dict(row_id=record["row_id"], native_advantage=coefficients[0],
                                  evidence_advantage=coefficients[1],
                                  selection_advantage=coefficients[2],
                                  evidence_mask=audit["evidence_mask"],
                                  selection_mask=audit["selection_mask"]))
        offset += size
    return flattened


def update(runtime, optimizer, scheduler, records, group_sizes, reference, credits):
    """One 128-sequence update; outcome/KL once, auxiliaries per valid span.

    No parameter update or scheduler advance occurs before the existing sampler
    probability gate passes. Failures resume from the last atomic checkpoint.
    """
    import torch

    from sr_f12.training import enforce_probability_gate, probability_diagnostics

    coefficients = _validate_credits(records, group_sizes, credits)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    diffs, truncations, logps = [], [], []
    totals = dict(policy=0.0, kl=0.0, loss=0.0, native_policy=0.0,
                  evidence_policy=0.0, selection_policy=0.0)
    previous_qid, prepared = None, None
    for record, coefficient in zip(records, coefficients, strict=True):
        if record["qid"] != previous_qid:
            prepared = runtime.prepare_input(
                record["model_input"], record["run_root"], protocol=record["protocol"]
            )
            previous_qid = record["qid"]
        item = dict(record, prepared=prepared)
        with torch.no_grad():
            ref = runtime.batch_reference_forward([item], reference)[0]
        current = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)[0]
        sampler = torch.as_tensor(record["sampler_logprobs"], device=current.device,
                                  dtype=torch.float32)
        result = credit_method.sequence_objective(
            current, sampler, ref,
            native_advantage=coefficient["native_advantage"],
            evidence_advantage=coefficient["evidence_advantage"],
            selection_advantage=coefficient["selection_advantage"],
            evidence_mask=coefficient["evidence_mask"],
            selection_mask=coefficient["selection_mask"],
        )
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
        totals, **probabilities, gradient_norm_before_clip=float(norm),
        training_seconds=time.monotonic() - started, group_sizes=list(group_sizes),
        native_advantages=[c["native_advantage"] for c in coefficients],
        evidence_advantages=[c["evidence_advantage"] for c in coefficients],
        selection_advantages=[c["selection_advantage"] for c in coefficients],
        evidence_mask_tokens=sum(sum(c["evidence_mask"]) for c in coefficients),
        selection_mask_tokens=sum(sum(c["selection_mask"]) for c in coefficients),
        evidence_applied_rows=sum(c["evidence_advantage"] != 0 and any(c["evidence_mask"])
                                  for c in coefficients),
        selection_applied_rows=sum(c["selection_advantage"] != 0 and any(c["selection_mask"])
                                   for c in coefficients),
        train_logprobs=logps, sequence_count=128, optimizer_updates=1, microbatch_size=1,
        loss_reduction="mean_of_128_native_sequence_means_plus_separate_span_means",
        kl_coefficient=0.02,
    )


def _counts(groups):
    return dict(
        groups=len(groups), responses=sum(len(g["native_advantages"]) for g in groups),
        eligibility_reasons=dict(Counter(g["eligibility_reason"] for g in groups)),
        native_l1=sum(sum(map(abs, g["native_advantages"])) for g in groups),
        evidence_l1=sum(sum(map(abs, g["evidence_advantages"])) for g in groups),
        selection_l1=sum(sum(map(abs, g["selection_advantages"])) for g in groups),
        evidence_mask_tokens=sum(sum(m["evidence_mask"]) for g in groups
                                 for m in g["record_masks"]),
        selection_mask_tokens=sum(sum(m["selection_mask"]) for g in groups
                                  for m in g["record_masks"]),
    )


def _publish(runtime, directory, latest, step, identity):
    """Only committed final parameters are exported; interrupted exports retry."""
    if step == data.STEPS:
        endpoint = directory / "checkpoints" / f"step-{step:03d}.pt"
        if not endpoint.exists():
            os.link(latest, endpoint)
        adapter = directory / "adapters" / f"step-{step:03d}"
        marker = adapter / "PUBLISHED.json"
        publication = dict(step=step, identity=identity, status="COMPLETE")
        if marker.exists():
            if read(marker) != publication:
                raise ValueError("Published local-credit adapter identity differs")
        else:
            runtime.model.save_pretrained(adapter, safe_serialization=True)
            write(marker, publication)
    write(directory / "PROGRESS.json", dict(
        at=now(), successful_updates=step, target_updates=data.STEPS, answers=step * 128,
    ))


def run_train(root, spec):
    from mm_core.training import trainable_state

    from .c_local import require_frozen, task_matrix
    from .runtime import load

    root = Path(root)
    registered = {s["task_id"]: s for s in task_matrix()}
    expected = registered.get(spec.get("task_id"), {})
    if (spec.get("kind") != "c_local_train" or type(spec.get("block")) is not int
            or not expected or any(
                spec.get(key) != expected.get(key)
                for key in ("task_id", "kind", "block", "arm"))):
        raise ValueError("Unregistered C local-credit training task")
    frozen = require_frozen(root)
    task_id, block, arm = spec["task_id"], spec["block"], spec["arm"]
    tasks, images = task_index(root), image_index(root)
    training_plan = read(root / "DATA" / f"TRAIN_PLAN_b{block}.json")
    directory = root / "RUNS/C_LOCAL" / f"b{block}" / arm
    identity = dict(
        experiment="SRF4-C-LOCAL-PILOT", task_id=task_id, block=block, arm=arm,
        master_seed=data.MASTER, steps=data.STEPS, responses_per_step=128, group_size=data.G,
        credit_schema=CREDIT_SCHEMA, verifier_version=credit_method.VERIFIER_VERSION, freeze=frozen,
    )
    _registered_write(directory / "run_identity.json", identity)
    runtime, optimizer, scheduler = load(root, block, task_id, training=True)
    # Capture the original registered parent BEFORE restoring the trained policy.
    reference = trainable_state(runtime.model)
    latest = directory / "checkpoints/latest.pt"
    step = 0
    if latest.exists():
        step, restored = load_checkpoint(latest, runtime, optimizer, scheduler, identity=identity)
        if (type(step) is not int or not 0 <= step <= data.STEPS
                or restored != {"next_step": step + 1}):
            raise ValueError("Invalid local-credit checkpoint cursor")
    else:
        save_checkpoint(latest, runtime, optimizer, scheduler, identity=identity, step=0,
                        sampler_state={"next_step": 1})
    runtime.identity["step"] = step
    _publish(runtime, directory, latest, step, identity)
    for update_step in range(step + 1, data.STEPS + 1):
        started = time.monotonic()
        runtime.torch.cuda.reset_peak_memory_stats()
        assignments = training_plan[str(update_step)]
        expected_assignments = data.training_assignments(list(tasks.values()), block, update_step)
        if assignments != expected_assignments:
            raise ValueError("Local-credit assignments differ from the frozen plan")
        if len(assignments) != data.GROUPS or any(
            a.get("G") != data.G or len(a["seeds"]) != data.G for a in assignments
        ):
            raise ValueError("Local-credit requires sixteen explicit G8 assignments per update")
        _registered_write(directory / "assignments" / f"{update_step:03d}.json",
                          dict(step=update_step, groups=assignments))
        records, credits = [], []
        for assignment in assignments:
            task = tasks[assignment["qid"]]
            current = group(
                root, runtime, task, images[task["qid"]], assignment,
                directory / "rollouts" / f"{update_step:03d}", task_id=task_id, block=block,
                master=data.MASTER, step=update_step, arm=arm, source=data.TRAIN_SPLIT,
            )
            if any(r.get("model_identity", {}).get("step") != update_step - 1
                   or r.get("model_identity", {}).get("run_id") != task_id for r in current):
                raise ValueError("Cached local rollout is not from the committed pre-update policy")
            labels = [credit_method.measure(task, r) for r in current]
            masks = [dict(credit_method.token_masks(r, label, runtime.processor.tokenizer),
                          row_id=r["row_id"])
                     for r, label in zip(current, labels, strict=True)]
            # Persist the actual generated-token masks before any gradient work.
            sidecar = (directory / "semantic" / f"{update_step:03d}"
                       / f"{assignment['group_index']:05d}.json")
            _registered_write(sidecar, dict(group_id=current[0]["group_id"],
                                           qid=task["qid"], world_id=task["world_id"],
                                           root_id=task.get("root_id", task["world_id"]),
                                           cell=task["cell"], labels=labels, record_masks=masks))
            seed = seed_for(data.MASTER, "c-local-selection-shuffle", block, update_step,
                            assignment["group_index"], task["qid"])
            credit = credit_method.group_credit(task, current, labels, arm=arm, seed=seed)
            credit.update(group_index=assignment["group_index"], block=block, step=update_step,
                          arm=arm, cell=task["cell"], family=task.get("family"),
                          record_masks=masks, labels=labels)
            credits.append(credit)
            records.extend(current)
        credit = dict(schema=CREDIT_SCHEMA, step=update_step, block=block, arm=arm,
                      verifier_version=credit_method.VERIFIER_VERSION,
                      groups=credits, counts=_counts(credits))
        _registered_write(directory / "credit" / f"{update_step:03d}.json", credit)
        receipt = update(runtime, optimizer, scheduler, records, [data.G] * data.GROUPS,
                         reference, credits)
        elapsed = time.monotonic() - started
        receipt.update(
            step=update_step, block=block, arm=arm, at=now(), credit_counts=credit["counts"],
            checkpoint_before=dict(task_id=task_id, successful_update=update_step - 1),
            checkpoint_after=dict(task_id=task_id, successful_update=update_step),
            wall_seconds=elapsed, gpu_hours=elapsed / 3600,
            peak_gpu_allocated_bytes=runtime.torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=runtime.torch.cuda.max_memory_reserved(),
            generated_tokens=sum(len(r["tokens"]) for r in records),
        )
        write(directory / "updates" / f"{update_step:03d}.json", receipt)
        save_checkpoint(latest, runtime, optimizer, scheduler, identity=identity, step=update_step,
                        sampler_state={"next_step": update_step + 1})
        runtime.identity["step"] = update_step
        _publish(runtime, directory, latest, update_step, identity)
    return dict(block=block, arm=arm, successful_updates=data.STEPS, answers=data.STEPS * 128,
                checkpoint=str(directory / "checkpoints" / f"step-{data.STEPS:03d}.pt"),
                adapter=str(directory / "adapters" / f"step-{data.STEPS:03d}"))
