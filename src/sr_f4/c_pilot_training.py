"""Frozen F-prime pilot: native training with an audited additive residual."""

from __future__ import annotations

import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from srf4_ref.credit import Observation
from srf4_ref.sampling import seed_for

from . import c_pilot_data as data
from .common import now, read, write
from .credit_design_replay import ARMS, BETA, POLICIES, eligibility, proposed_residual
from .run import group, image_index, task_index
from .training import load_checkpoint, native_advantages, save_checkpoint, update
from .verification_v3 import VERIFIER_VERSION, verify_semantics

POLICY = next(p for p in POLICIES if p.name == "F_PRIME_KNOWN_75")
CREDIT_SCHEMA = "srf4-c-fprime-credit-v1"


def _registered_write(path, value):
    """A replay may reuse a record, but cannot silently replace its identity."""
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Registered pilot record differs: {path}")
    else:
        write(path, value)


def _observations(records, labels):
    if len(records) != data.G or len(labels) != len(records):
        raise ValueError("A pilot credit group requires exactly G8 records and labels")
    observed = []
    for record, label in zip(records, labels, strict=True):
        if label.get("row_id") != record["row_id"] or label.get("verifier_version") != (
            VERIFIER_VERSION
        ):
            raise ValueError("Pilot semantic row or verifier identity differs")
        n = label["normalization"]
        if n["j"] != record["score"]["J"]:
            raise ValueError("Semantic normalization differs from native binary J")
        if any(type(n[k]) is not bool for k in ("protocol_valid", "finished", "defined")):
            raise ValueError("Pilot eligibility flags must be booleans")
        if n["sufficient"] is not None and type(n["sufficient"]) is not bool:
            raise ValueError("Pilot sufficiency must be boolean or unknown")
        observed.append(Observation(
            record["row_id"], n["j"], n["protocol_valid"], n["finished"], n["sufficient"],
            n["defined"], label["dimensions"]["P"]["status"], label["dimensions"]["S"]["status"],
        ))
    return observed


def _shuffle_receipt(observations, known, seed, *, applied):
    """Log the one actual Fisher-Yates permutation, including unchanged draws.

    Shuffling source indices uses exactly the same random calls and stratum
    order as proposed_residual's value shuffle; there is no retry or rejection.
    """
    permutation = list(range(len(observations)))
    strata, rng = [], random.Random(seed)
    if applied:
        for p in ("FAIL", "PASS"):
            destinations = [i for i in known if observations[i].perception == p]
            sources = list(destinations)
            rng.shuffle(sources)
            for destination, source in zip(destinations, sources, strict=True):
                permutation[destination] = source
            strata.append(dict(perception=p, indices=destinations, source_indices=sources))
    before = [r.selection for r in observations]
    after = [before[i] for i in permutation]
    return dict(seed=seed, applied=applied, permutation_source_indices=permutation,
                selection_before=before, selection_after=after, within_p_strata=strata,
                changed_count=sum(a != b for a, b in zip(before, after, strict=True)))


def group_credit(task, records, labels, *, arm, block, step, group_index):
    """Compute exact native A_J + beta*d; preserve all rows and zero-J groups."""
    if arm not in ARMS:
        raise ValueError("Unregistered pilot arm")
    if len({r["group_id"] for r in records}) != 1 or any(r["qid"] != task["qid"] for r in records):
        raise ValueError("Pilot group identities differ")
    observed = _observations(records, labels)
    candidates, known, reason = eligibility(observed, POLICY)
    seed = seed_for(data.MASTER, "c-pilot-selection-shuffle", block, step, group_index, task["qid"])
    native = native_advantages([r.j for r in observed])
    residuals = {a: proposed_residual(observed, a, POLICY, shuffle_seed=seed) for a in ARMS}
    residual = residuals[arm]
    final = [a + d for a, d in zip(native, residual, strict=True)]
    selection_signal = reason == "ACTIVE" and {
        observed[i].selection for i in known if observed[i].perception == "PASS"
    } == {"PASS", "FAIL"}
    zero_native = all(r.j == 0 for r in observed)
    diagnostics = {}
    for name, values in residuals.items():
        advantages = [a + d for a, d in zip(native, values, strict=True)]
        diagnostics[name] = dict(
            signal=any(v != 0 for v in values), residual_l1=sum(abs(v) for v in values),
            residual_linf=max(map(abs, values)), nonzero_count=sum(v != 0 for v in values),
            positive_failure_count=sum(r.j == 0 and a > 0
                                       for r, a in zip(observed, advantages, strict=True)),
        )
    return dict(
        group_id=records[0]["group_id"], qid=task["qid"], world_id=task["world_id"],
        cell=task["cell"],
        family=task.get("family", task.get("query", {}).get("family")), group_index=group_index,
        observations=[asdict(r) for r in observed], candidate_indices=candidates,
        known_indices=known,
        candidate_count=len(candidates), known_count=len(known), eligibility_reason=reason,
        active=reason == "ACTIVE", signal=diagnostics[arm]["signal"],
        selection_signal=selection_signal, native_J_all_zero=zero_native,
        zero_native_J_selection_signal=zero_native and selection_signal,
        native_l1=sum(map(abs, native)),
        select_minus_evid_l1=sum(abs(a - b) for a, b in
                                 zip(residuals["SELECT"], residuals["EVID"], strict=True)),
        select_minus_shuffle_l1=sum(
            abs(a - b) for a, b in zip(residuals["SELECT"], residuals["SELECT_SHUF"], strict=True)),
        counterfactual_note="same_own_rollouts_diagnostic_not_causal",
        native_advantages=native,
        residual=residual, final_advantages=final, arm_diagnostics=diagnostics,
        shuffle=_shuffle_receipt(observed, known, seed,
                                 applied=reason == "ACTIVE" and arm == "SELECT_SHUF"),
    )


def _counts(groups):
    return dict(
        groups=len(groups), responses=sum(len(g["observations"]) for g in groups),
        active_groups=sum(g["active"] for g in groups),
        signal_groups=sum(g["signal"] for g in groups),
        selection_signal_groups=sum(g["selection_signal"] for g in groups),
        zero_native_J_groups=sum(g["native_J_all_zero"] for g in groups),
        zero_native_J_selection_signal_groups=sum(g["zero_native_J_selection_signal"]
                                                 for g in groups),
        native_l1=sum(g["native_l1"] for g in groups),
        select_minus_evid_l1=sum(g["select_minus_evid_l1"] for g in groups),
        select_minus_shuffle_l1=sum(g["select_minus_shuffle_l1"] for g in groups),
        candidates=sum(g["candidate_count"] for g in groups),
        known=sum(g["known_count"] for g in groups),
        eligibility_reasons=dict(Counter(g["eligibility_reason"] for g in groups)),
        shuffle_changed_groups=sum(g["shuffle"]["changed_count"] > 0 for g in groups),
        arm_diagnostics={a: dict(
            signal_groups=sum(g["arm_diagnostics"][a]["signal"] for g in groups),
            residual_l1=sum(g["arm_diagnostics"][a]["residual_l1"] for g in groups),
            nonzero_count=sum(g["arm_diagnostics"][a]["nonzero_count"] for g in groups),
            positive_failure_count=sum(g["arm_diagnostics"][a]["positive_failure_count"]
                                       for g in groups),
        ) for a in ARMS},
    )


def _publish(runtime, directory, latest, step, identity):
    """Publish only committed parameters; incomplete adapter exports are retried."""
    if step == data.STEPS:
        endpoint = directory / "checkpoints" / f"step-{step:03d}.pt"
        if not endpoint.exists():
            os.link(latest, endpoint)
        adapter = directory / "adapters" / f"step-{step:03d}"
        marker = adapter / "PUBLISHED.json"
        publication = dict(step=step, identity=identity, status="COMPLETE")
        if marker.exists():
            if read(marker) != publication:
                raise ValueError("Published pilot adapter identity differs")
        else:
            runtime.model.save_pretrained(adapter, safe_serialization=True)
            write(marker, publication)
    write(directory / "PROGRESS.json", dict(
        at=now(), successful_updates=step, target_updates=data.STEPS, answers=step * 128,
    ))


def run_train(root, spec):
    from mm_core.training import trainable_state

    from .c_pilot import require_frozen
    from .runtime import load

    root = Path(root)
    if spec.get("kind") != "c_pilot_train" or spec.get("arm") not in ARMS or (
        type(spec.get("block")) is not int or spec["block"] not in (0, 1, 2)
    ):
        raise ValueError("Unregistered F-prime pilot training task")
    frozen = require_frozen(root)
    task_id, block, arm = spec["task_id"], spec["block"], spec["arm"]
    tasks, images = task_index(root), image_index(root)
    training_plan = read(root / "DATA" / f"TRAIN_PLAN_b{block}.json")
    directory = root / "RUNS/C" / f"b{block}" / arm
    identity = dict(
        experiment="SRF4-C-FPRIME-PILOT", task_id=task_id, block=block, arm=arm,
        master_seed=data.MASTER, steps=data.STEPS, responses_per_step=128, group_size=data.G,
        verifier_version=VERIFIER_VERSION, policy=POLICY.name, beta=BETA, freeze=frozen,
    )
    _registered_write(directory / "run_identity.json", identity)
    runtime, optimizer, scheduler = load(root, block, task_id, training=True)
    # Capture the original registered parent before restoring a trained policy.
    reference = trainable_state(runtime.model)
    latest = directory / "checkpoints/latest.pt"
    step = 0
    if latest.exists():
        step, restored = load_checkpoint(latest, runtime, optimizer, scheduler, identity=identity)
        if (type(step) is not int or not 0 <= step <= data.STEPS
                or restored != {"next_step": step + 1}):
            raise ValueError("Invalid pilot checkpoint cursor")
    else:
        save_checkpoint(latest, runtime, optimizer, scheduler, identity=identity, step=0,
                        sampler_state={"next_step": 1})
    runtime.identity["step"] = step
    _publish(runtime, directory, latest, step, identity)
    for update_step in range(step + 1, data.STEPS + 1):
        started = time.monotonic()
        runtime.torch.cuda.reset_peak_memory_stats()
        assignments = training_plan[str(update_step)]
        expected = data.training_assignments(list(tasks.values()), block, update_step)
        if assignments != expected:
            raise ValueError("Pilot assignments differ from the pre-frozen training plan")
        if len(assignments) != 16 or any(a.get("G") != data.G or len(a["seeds"]) != data.G
                                         for a in assignments):
            raise ValueError("Pilot requires sixteen explicit G8 assignments per update")
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
                raise ValueError("Cached pilot rollout is not from the committed pre-update policy")
            labels = [verify_semantics(task, r, r["score"]) for r in current]
            sidecar = (directory / "semantic" / f"{update_step:03d}"
                       / f"{assignment['group_index']:05d}.json")
            _registered_write(
                sidecar, dict(group_id=current[0]["group_id"],
                              verifier_version=VERIFIER_VERSION, labels=labels),
            )
            credits.append(group_credit(task, current, labels, arm=arm, block=block,
                                        step=update_step, group_index=assignment["group_index"]))
            records.extend(current)
        credit = dict(schema=CREDIT_SCHEMA, step=update_step, block=block, arm=arm,
                      policy=POLICY.name, beta=BETA, verifier_version=VERIFIER_VERSION,
                      groups=credits, counts=_counts(credits))
        _registered_write(directory / "credit" / f"{update_step:03d}.json", credit)
        final = [a for g in credits for a in g["final_advantages"]]
        receipt = update(runtime, optimizer, scheduler, records, [data.G] * 16, reference,
                         final_advantages=final)
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
