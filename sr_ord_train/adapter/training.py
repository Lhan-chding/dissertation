"""SR-ORD-TRAIN-1 J-only update under a new, independently registered identity.

This module neither samples nor publishes checkpoints. The caller holds its path
lock and supplies the frozen step plan. A return is UPDATED_UNCOMMITTED; after
any post-Adam exception the last committed checkpoint must be restored.
Historical access/phase/arm entrypoints are never invoked or patched.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from runtime_contract import PINS, check_numerics

EXPERIMENT = "SR-ORD-TRAIN-1"
TRAJECTORY_FIELDS = (
    "tokens", "sampler_logprobs", "prompt_tensor_hash", "native_prompt_inputs",
    "model_input", "protocol", "policy_hash", "reference_hash", "source_sha256",
    "request_id", "seed", "input_id", "assistant_prefill",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def trajectory_hash(record):
    return canonical_hash({key: record[key] for key in TRAJECTORY_FIELDS})


def native_j_advantages(values):
    """The exact scalar J-only definition from pinned sr_f6.credit, no epsilon.

    The historical function couples this expression to old arm identities;
    exhaustive binary G8 fixtures compare this projection to its native output.
    """
    if len(values) != 8 or any(type(v) not in (int, bool) or v not in (0, 1) for v in values):
        raise ValueError("A complete binary G8 is required")
    mean = sum(values) / 8
    deviation = math.sqrt(sum((value - mean) ** 2 for value in values) / 8)
    return [(value - mean) / deviation if deviation else 0.0 for value in values]


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _append(path, event):
    _require(not path.is_symlink(), "Evidence path cannot be a symlink")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_safe(event), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _paths(identity):
    root = Path(identity["run_root"])
    audit = Path(identity["audit_root"])
    for value in (root, audit):
        _require(value.is_absolute() and value.is_dir() and value.resolve(strict=True) == value,
                 "Run and audit roots must be canonical existing directories")
    _require(audit != root and audit.is_relative_to(root), "Audit must be a strict run descendant")
    return root, audit / "NUMERICAL_TOKEN_AUDIT.jsonl"


def validate_records(identity, records, plan_step, policy_hash, reference_hash):
    """Validate all identities and saved trajectories before any forward pass."""
    _require(identity.get("experiment_id") == EXPERIMENT and
             identity.get("execution_mode") == "TRAINING", "New training registration required")
    _require(identity.get("arm") in ("R", "G", "D") and
             identity.get("parent_id") in ("b0", "b1") and
             identity.get("train_seed") in (11031, 11032), "Unknown registered path")
    committed = identity.get("step")
    _require(type(committed) is int and 0 <= committed < 24 and
             type(plan_step.get("step")) is int and plan_step["step"] == committed + 1,
             "Plan step differs from committed cursor")
    _require(_sha(identity.get("manifest_sha256")) and
             identity.get("plan_step_sha256") == canonical_hash(plan_step), "Unbound frozen plan")
    _require(all(_sha(identity.get(k)) for k in ("source_sha256", "reference_hash")) and
             identity["reference_hash"] == reference_hash and _sha(policy_hash), "Unbound source/reference")
    _require(isinstance(identity.get("run_id"), str) and bool(identity["run_id"]), "Missing run identity")
    if committed == 0:
        _require(policy_hash == reference_hash, "Step zero is not the parent reference")
    root, _ = _paths(identity)
    groups = plan_step.get("groups", [])
    _require(len(groups) == 16 and len(records) == 128, "Exactly 16 G8 groups required")
    _require(len({g["root_id"] for g in groups}) == 16 and
             len({g["qid"] for g in groups}) == 16 and
             len({g["group_id"] for g in groups}) == 16, "Sixteen independent roots required")
    _require(len({r["row_id"] for r in records}) == 128, "Duplicate trajectory identity")
    coefficients, credit_groups, image_hashes = [], [], {}
    for offset, group in zip(range(0, 128, 8), groups, strict=True):
        rows = records[offset:offset + 8]
        _require(len(group["seeds"]) == len(group["request_ids"]) == 8 and
                 len(set(group["seeds"])) == len(set(group["request_ids"])) == 8, "Invalid frozen G8")
        common_input = canonical_hash(rows[0]["model_input"])
        common_tensor = rows[0]["prompt_tensor_hash"]
        js = []
        for draw, row in enumerate(rows):
            expected = dict(group_id=group["group_id"], qid=group["qid"], root_id=group["root_id"],
                            input_id=group["input_id"], case_id=group["case_id"],
                            request_id=group["request_ids"][draw], row_id=group["request_ids"][draw],
                            seed=group["seeds"][draw], draw=draw, step=committed + 1,
                            parent=identity["parent_id"], arm=identity["arm"], train_seed=identity["train_seed"],
                            run_id=identity["run_id"], policy_hash=policy_hash, reference_hash=reference_hash,
                            source_sha256=identity["source_sha256"], protocol="structured_v4", assistant_prefill="{")
            _require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in expected.items()),
                     "Trajectory differs from registered path/step/draw")
            _require(row.get("generation_status", "COMPLETE") == "COMPLETE" and
                     not row.get("technical_validation_errors"), "Unresolved generation error")
            model_input = row["model_input"]
            _require(isinstance(model_input, dict) and set(model_input) == {"prompt", "image_file", "image_sha256"},
                     "Private data or unknown fields at model input boundary")
            _require(isinstance(model_input["prompt"], str) and bool(model_input["prompt"]) and
                     row.get("prompt_sha256") == hashlib.sha256(model_input["prompt"].encode()).hexdigest(),
                     "Prompt text identity differs")
            _require(canonical_hash(model_input) == common_input and
                     row["prompt_tensor_hash"] == common_tensor and _sha(common_tensor), "Mixed G8 input")
            relative = Path(model_input["image_file"])
            _require(not relative.is_absolute() and ".." not in relative.parts, "Nonlocal image")
            image = root / relative
            _require(image.resolve(strict=True).is_relative_to(root) and image.is_file(), "Image escapes root")
            if str(image) not in image_hashes:
                image_hashes[str(image)] = hashlib.sha256(image.read_bytes()).hexdigest()
            _require(image_hashes[str(image)] == model_input["image_sha256"], "Image changed")
            tokens, sampler = row["tokens"], row["sampler_logprobs"]
            _require(isinstance(tokens, list) and 1 <= len(tokens) <= 1536 and
                     all(type(t) is int and t >= 0 for t in tokens), "Invalid original token path")
            _require(row.get("raw_tokens") == tokens and row.get("completion_token_count") == len(tokens),
                     "Original token aliases/count differ")
            _require(isinstance(sampler, list) and len(sampler) == len(tokens) and
                     all(type(p) in (int, float) and math.isfinite(p) for p in sampler), "Invalid sampler vector")
            _require(row.get("sampler_logprob_source") == "actual_generate_raw_logits_selected_token",
                     "Sampler probabilities are not original generation logits")
            _require(row.get("token_path_sha256") == trajectory_hash(row), "Trajectory digest differs")
            sampling = row.get("sampling_parameters", {})
            _require(all(sampling.get(k) == v for k, v in dict(max_new_tokens=1536, temperature=1.,
                     top_p=1., top_k=0, do_sample=True).items()), "Sampling recipe differs")
            score = row["score"]
            _require(all(type(score.get(k)) in (int, bool) and score[k] in (0, 1)
                         for k in ("J", "V", "finished")), "Invalid native scorer labels")
            _require(not score["J"] or (score["V"] and score["finished"]), "Invalid native joint reward")
            _require("J_M3" not in score or score["J_M3"] == score["J"], "Native J aliases differ")
            js.append(score["J"])
        advantage = native_j_advantages(js)
        coefficients.extend(advantage)
        credit_groups.append({**copy.deepcopy(group), "J": js, "native": advantage, "aux": [0.] * 8})
    return coefficients, credit_groups


def _verify_prepared(record, prepared, torch):
    _require(prepared.get("routing", {}).get("input_tensor_hash") == record["prompt_tensor_hash"],
             "Actual processor tensor hash differs")
    inputs = prepared["inputs"]
    native = {k: v.detach().cpu().tolist() for k, v in inputs.items()
              if k != "pixel_values" and isinstance(v, torch.Tensor)}
    _require(native == record["native_prompt_inputs"], "Prompt tokens/masks/grid differ")
    _require(inputs["input_ids"].ndim == 2 and inputs["input_ids"].shape[0] == 1 and
             inputs["input_ids"].shape[1] == record["prompt_token_count"], "Prompt boundary differs")
    _require(all(not v.is_floating_point() or v.dtype == torch.float32
                 for v in inputs.values() if isinstance(v, torch.Tensor)), "Non-FP32 processor input")


def update(runtime, optimizer, scheduler, records, reference, plan_step):
    """One guarded update, no checkpoint publication or runtime cursor mutation."""
    import torch
    import sr_f6.objective as objective
    from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
    from sr_f3_ce.training import finite_tree, validate_optimizer
    from sr_f5.training import numerical_audit

    _require(hashlib.sha256(Path(objective.__file__).read_bytes()).hexdigest() == PINS["sr_f6/objective.py"],
             "Native objective fingerprint differs")
    identity = copy.deepcopy(runtime.identity)
    _, audit_path = _paths(identity)
    before = trainable_state(runtime.model)
    before_hash, ref_hash = state_hash(before), state_hash(reference)
    coefficients, groups = validate_records(identity, records, plan_step, before_hash, ref_hash)
    _require(set(reference) == set(before) and all(isinstance(v, torch.Tensor) and not v.requires_grad and
             v.dtype == torch.float32 and v.shape == before[k].shape and bool(torch.isfinite(v).all())
             for k, v in reference.items()), "Frozen reference tensor contract differs")
    _require(all(not v.is_floating_point() or v.dtype == torch.float32
                 for v in (*runtime.model.parameters(), *runtime.model.buffers())), "Model must remain FP32")
    _require(not any(isinstance(m, torch.nn.Dropout) and m.p for m in runtime.model.modules()), "Dropout is forbidden")
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    _require(finite_tree(before) and finite_tree(optimizer.state_dict()) and finite_tree(scheduler.state_dict()),
             "Nonfinite committed state")
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    _require(bool(parameters) and scheduler.last_epoch == identity["step"], "Optimizer/scheduler cursor differs")
    if identity["step"] == 0:
        _require(not optimizer.state, "Inherited optimizer state is forbidden")
    else:
        _require(set(optimizer.state) == set(parameters) and all(float(optimizer.state[p].get("step", -1)) ==
                 identity["step"] for p in parameters), "Incomplete or stale Adam state")
    started, rng, old_training = time.monotonic(), capture_rng(), runtime.model.training
    traces, totals, numerics = [], {k: 0. for k in ("loss", "native", "aux", "kl", "kl_unweighted")}, {}
    optimizer_started = False
    event = dict(experiment_id=EXPERIMENT, run_id=identity["run_id"], step=plan_step["step"],
                 parent=identity["parent_id"], arm=identity["arm"], train_seed=identity["train_seed"],
                 policy_before_sha256=before_hash, reference_sha256=ref_hash,
                 manifest_sha256=identity["manifest_sha256"], plan_step_sha256=identity["plan_step_sha256"],
                 rng_before_sha256=state_hash(rng))
    try:
        optimizer.zero_grad(set_to_none=True)
        runtime.model.eval()
        prepared, previous = None, None
        for row, advantage in zip(records, coefficients, strict=True):
            key = (row["input_id"], canonical_hash(row["model_input"]))
            if key != previous:
                prepared = runtime.prepare_input(row["model_input"], Path(identity["run_root"]), protocol="structured_v4")
                previous = key
            _verify_prepared(row, prepared, torch)
            item = dict(row, prepared=prepared)
            with torch.no_grad():
                refs = runtime.batch_reference_forward([item], reference)
            currents = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)
            _require(len(refs) == len(currents) == 1, "Microbatch size must be one")
            current, ref = currents[0], refs[0]
            _require(current.ndim == 1 and len(current) == len(row["tokens"]) and ref.shape == current.shape,
                     "Teacher probabilities do not cover original token path")
            sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32, device=current.device)
            traces.append(dict(row_id=row["row_id"], qid=row["qid"], tokens=list(row["tokens"]),
                               sampler_logprobs=sampler.detach().cpu().tolist(),
                               train_logprobs=current.detach().cpu().tolist(), reference_logprobs=ref.detach().cpu().tolist(),
                               token_path_sha256=row["token_path_sha256"]))
            result = objective.sequence_loss(current, sampler, ref, advantage, 0.)
            (result["loss"] / 128).backward()
            for name in totals:
                totals[name] += float(result[name].detach()) / 128
        numerics = numerical_audit(traces)
        _require(numerics["status"] == "FINITE" and finite_tree(totals), "Nonfinite probability/objective")
        _require(state_hash(trainable_state(runtime.model)) == before_hash and state_hash(reference) == ref_hash,
                 "Forward mutated policy/reference")
        _require(runtime.identity == identity, "Runtime identity changed during update")
        _require(any(p.grad is not None for p in parameters), "No adapter gradient")
        norm_before = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        norm_after = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                   for p in parameters if p.grad is not None))
        numerics.update(kl_unweighted=totals["kl_unweighted"], gradient_norm_before_clip=norm_before,
                        gradient_norm_after_clip=norm_after)
        check_numerics(numerics)
        _append(audit_path, dict(event, status="PREUPDATE_PASS", precommit_gate="PASS_BEFORE_ADAM",
                                numerics=numerics, probability_vectors=traces, totals=totals))
        optimizer_started = True
        optimizer.step()
        scheduler.step()
        after = trainable_state(runtime.model)
        _require(finite_tree(after) and finite_tree(optimizer.state_dict()) and finite_tree(scheduler.state_dict()),
                 "Nonfinite state after Adam: restore committed checkpoint")
        _require(state_hash(reference) == ref_hash, "Reference changed after Adam")
        restore_rng(rng)
        _require(state_hash(capture_rng()) == event["rng_before_sha256"], "RNG restoration failed")
        return dict(event, status="UPDATED_UNCOMMITTED", policy_version_sampled=identity["step"],
                    policy_after_sha256=state_hash(after), optimizer_steps=1, optimizer_updates=1,
                    responses=128, sequence_count=128, aux_nonzero_rows=0, prefix_rows=0,
                    precommit_gate="PASS_BEFORE_ADAM", groups=groups, numerics=numerics,
                    probability_vectors=traces, **totals, group_sizes=[8] * 16, microbatch_size=1,
                    parameter_update_norm=math.sqrt(sum(float((after[k].double() - before[k].double()).square().sum())
                                                       for k in before)),
                    rng_after_sha256=state_hash(capture_rng()), training_seconds=time.monotonic() - started,
                    loss_reduction="mean of all 128 original completion token means",
                    forward_calls=256, backward_calls=128, diagnostic_gradient_calls=0)
    except Exception as exc:
        _append(audit_path, dict(event, status="FAILED_RESTORE_CHECKPOINT" if optimizer_started else "FAILED_BEFORE_ADAM",
                                optimizer_started=optimizer_started, error_type=type(exc).__name__, error=str(exc),
                                numerics=numerics, totals=totals, probability_vectors=traces,
                                original_trajectories=[{k: r.get(k) for k in TRAJECTORY_FIELDS} for r in records],
                                runtime_identity=identity, rng_at_failure_sha256=state_hash(capture_rng())))
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng)
        runtime.model.train(old_training)
