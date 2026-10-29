"""One bound SR-F6 update over exactly 16 G8 groups; no sampling or commits.

The caller owns the exclusive run lock and commits only after success. A failed
pre-update check leaves Adam, scheduler, and parameters untouched. An exception
after optimizer.step starts requires restoring the last committed checkpoint.
CPU fixtures exercise this adapter; they do not certify a real GPU trajectory.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import hash_json
from sr_f3_ce.training import finite_tree, validate_optimizer
from sr_f4.common import append
from sr_f5.training import numerical_audit
from sr_f12.training import ProbabilityMismatch, enforce_probability_gate

from .common import digest, file_hash, now, safe_evidence
from .contract import DIAGNOSTIC_STEPS
from .credit import build_credit_group as _build_credit_group
from .objective import sequence_loss

BATCH_SIZE = 128
TRAJECTORY_FIELDS = (
    "tokens",
    "sampler_logprobs",
    "prompt_tensor_hash",
    "native_prompt_inputs",
    "model_input",
    "protocol",
    "sampling_hash",
    "policy_hash",
    "source_sha256",
)


def trajectory_hash(record):
    """Hash the saved native trajectory, never a re-encoding of its text."""
    return digest({name: record[name] for name in TRAJECTORY_FIELDS})


def _score_row(record):
    score = record.get("score", record)
    result = dict(record)
    for name in ("J", "V", "finished", "S", "B"):
        if name not in score:
            raise ValueError(f"Missing scored field: {name}")
        if name in record and record[name] != score[name]:
            raise ValueError(f"Conflicting top-level and nested score: {name}")
        result[name] = score[name]
    return result


def build_credit_group(records, arm, seed):
    """Accept native records with either nested or flat immutable scores."""
    return _build_credit_group([_score_row(record) for record in records], arm, seed)


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"A content SHA256 is required for {name}")
    return value


def _audit_path(runtime):
    """Keep shared DATA resolution separate from a locked path's audit output."""
    root = Path(runtime.identity["run_root"])
    if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
        raise PermissionError("Runtime audit root must be an existing canonical directory")
    identity = runtime.identity
    if identity.get("execution_mode", "TRAINING") != "TRAINING":
        raise PermissionError("Only a training runtime can execute an optimizer update")
    registered_phase = identity.get("phase") in ("PILOT_V1", "PILOT_V3", "FORMAL_V1")
    if registered_phase and "audit_root" not in identity:
        raise PermissionError("Later training phases require an isolated audit directory")
    audit_root = Path(identity.get("audit_root", root))
    if (
        not audit_root.is_absolute()
        or audit_root.resolve(strict=True) != audit_root
        or not audit_root.is_dir()
        or not audit_root.is_relative_to(root)
        or ("audit_root" in identity and audit_root == root)
    ):
        raise PermissionError("Audit directory must be a canonical strict run descendant")
    path = audit_root / "NUMERICAL_TOKEN_AUDIT.jsonl"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PermissionError("Unsafe numerical evidence path")
    return root, path


def _validate_reference(runtime, reference, before):
    import torch

    identity = runtime.identity
    if identity.get("reference_parent") != identity.get("parent_id"):
        raise PermissionError("Reference must be the corresponding frozen parent")
    expected = _sha(identity.get("reference_hash"), "frozen parent reference")
    if not isinstance(reference, dict) or set(reference) != set(before):
        raise PermissionError("Frozen reference parameter names differ from the inherited adapter")
    for name, value in reference.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.requires_grad
            or value.dtype != torch.float32
            or value.shape != before[name].shape
            or not bool(torch.isfinite(value).all())
        ):
            raise PermissionError("Frozen parent reference tensor contract differs")
    if state_hash(reference) != expected:
        raise PermissionError("Frozen parent reference content hash differs")


def _validate_records(runtime, records, group_sizes, reference, credits, root, before):
    import torch

    if group_sizes != [8] * 16 or len(records) != BATCH_SIZE or len(credits) != 16:
        raise ValueError("Exactly sixteen G8 groups and all 128 sequences are required")
    identity = runtime.identity
    step = identity.get("step")
    if type(step) is not int or not 0 <= step < 72:
        raise ValueError("Invalid committed logical step")
    for name in ("run_id", "parent_id"):
        if not isinstance(identity.get(name), str) or not identity[name]:
            raise ValueError(f"Runtime lacks registered {name}")
    source = _sha(identity.get("source_sha256"), "frozen production source")
    policy = state_hash(before)
    _validate_reference(runtime, reference, before)
    if step == 0 and policy != identity["reference_hash"]:
        raise PermissionError("The first update must start from its corresponding frozen parent")
    if not finite_tree(before):
        raise FloatingPointError("Nonfinite committed adapter parameters")
    if len({record["row_id"] for record in records}) != 128:
        raise ValueError("Duplicate row across the update")
    if len({record["qid"] for record in records}) != 16:
        raise ValueError("Sixteen distinct question groups are required")
    if len({receipt["group_id"] for receipt in credits}) != 16:
        raise ValueError("Duplicate credit group")
    if len({receipt["arm"] for receipt in credits}) != 1:
        raise ValueError("One update cannot mix arms")
    root_views = defaultdict(set)
    for start in range(0, 128, 8):
        row = records[start]
        variant = row.get("variant")
        if type(variant) is not int or variant not in range(4):
            raise ValueError("Each question must name a registered variant 0..3")
        root_views[row["root_id"]].add(variant)
    if len(root_views) != 4 or any(views != set(range(4)) for views in root_views.values()):
        raise ValueError("Each update requires four roots with all four views")
    names = [name for name, value in runtime.model.named_parameters() if value.requires_grad]
    if not names or set(names) != set(before):
        raise PermissionError("Empty or changing trainable adapter")
    for value in (*runtime.model.parameters(), *runtime.model.buffers()):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise PermissionError("Every model parameter and floating buffer must remain FP32")
    if any(
        isinstance(module, torch.nn.Dropout) and module.p != 0 for module in runtime.model.modules()
    ):
        raise PermissionError("Registered adapter training has no dropout")
    coefficients, image_hashes = [], {}
    for offset, receipt in zip(range(0, 128, 8), credits, strict=True):
        group = records[offset : offset + 8]
        expected = build_credit_group(group, receipt["arm"], receipt["seed"])
        if receipt != expected:
            raise PermissionError("Credit receipt differs from the actual scored G8")
        common_input = digest(group[0]["model_input"])
        common_prompt = group[0]["prompt_tensor_hash"]
        for record in group:
            if record["run_root"] != str(root) or record.get("run_id") != identity["run_id"]:
                raise PermissionError("Rollout belongs to another run/root")
            if record.get("step") != step + 1 or type(record.get("step")) is not int:
                raise PermissionError("Rollout target step differs from the pending update")
            if record["policy_hash"] != policy or record.get("source_sha256") != source:
                raise PermissionError("Rollout policy/source differs from current content identity")
            adapter = record.get("adapter_identity", {})
            if any(
                adapter.get(a) != identity[b]
                for a, b in (("run_id", "run_id"), ("parent", "parent_id"), ("step", "step"))
            ):
                raise PermissionError("Rollout adapter identity is stale or from another parent")
            if record.get("parent") != identity["parent_id"]:
                raise PermissionError("Scored row parent differs from current runtime")
            if record.get("protocol") != "structured_v4":
                raise PermissionError("Unregistered SR-F6 training protocol")
            safe = record["model_input"]
            if not isinstance(safe, dict) or set(safe) != {"prompt", "image_file", "image_sha256"}:
                raise PermissionError("Model input must use the public whitelist")
            if not isinstance(safe["prompt"], str) or not safe["prompt"]:
                raise ValueError("A nonempty frozen prompt is required")
            if (
                record.get("prompt") != safe["prompt"]
                or record.get("prompt_sha256")
                != hashlib.sha256(safe["prompt"].encode()).hexdigest()
            ):
                raise PermissionError("Saved prompt text/hash changed")
            if digest(safe) != common_input or record["prompt_tensor_hash"] != common_prompt:
                raise PermissionError("A G8 cannot mix prompt or image identities")
            _sha(record["prompt_tensor_hash"], "native prompt tensor")
            image = safe_evidence(root, safe["image_file"])
            image_key = str(image)
            if image_key not in image_hashes:
                image_hashes[image_key] = file_hash(image)
            if _sha(safe["image_sha256"], "image") != image_hashes[image_key]:
                raise PermissionError("Image bytes differ from frozen input")
            tokens, sampler = record["tokens"], record["sampler_logprobs"]
            if not isinstance(tokens, list) or not 1 <= len(tokens) <= 1536:
                raise ValueError("Original completion must contain 1..1536 sampled tokens")
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("Invalid original completion token ID")
            if record.get("raw_tokens") != tokens or record.get("completion_token_count") != len(
                tokens
            ):
                raise PermissionError("Original raw token trajectory/count changed")
            if "raw_token_ids" in record and record["raw_token_ids"] != tokens:
                raise PermissionError("Raw token aliases disagree")
            if not isinstance(sampler, list) or len(sampler) != len(tokens):
                raise ValueError("Sampler probabilities must align with every original token")
            if any(
                type(value) not in (int, float) or not math.isfinite(value) for value in sampler
            ):
                raise FloatingPointError("Nonfinite or nonnumeric saved sampler probability")
            if record.get("sampler_logprob_source") != "actual_generate_raw_logits_selected_token":
                raise PermissionError("Sampler logprobs must be actual selected-token logits")
            sampling = record.get("sampling_parameters", {})
            for key, value in (
                ("max_new_tokens", 1536),
                ("temperature", 1.0),
                ("top_p", 1.0),
                ("top_k", 0),
                ("do_sample", True),
            ):
                if sampling.get(key) != value:
                    raise PermissionError("Saved sampler recipe differs")
            if record.get("max_new_tokens") != 1536 or record.get("sampling_hash") != hash_json(
                sampling
            ):
                raise PermissionError("Saved sampler configuration hash differs")
            if record.get("generation_status") != "COMPLETE" or record.get(
                "technical_validation_errors"
            ):
                raise PermissionError("Technically invalid original generation cannot be updated")
            if record.get("token_path_sha256") != trajectory_hash(record):
                raise PermissionError("Saved native token/probability trajectory hash differs")
            coefficients.append(receipt["coefficients_by_row_id"][record["row_id"]])
    return coefficients


def _diagnostic_indices(runtime, records, requested):
    step = runtime.identity["step"] + 1
    indices = tuple(requested)
    if step not in DIAGNOSTIC_STEPS:
        if indices:
            raise PermissionError("Gradient diagnostic outside registered steps")
        return indices
    if (
        len(indices) != 64
        or len(set(indices)) != 64
        or any(type(index) is not int or not 0 <= index < 128 for index in indices)
    ):
        raise PermissionError(
            "Registered diagnostic requires 64 unique rows from eight complete G8s"
        )
    expected = runtime.identity.get("diagnostic_row_indices")
    if not isinstance(expected, (list, tuple)) or list(indices) != list(expected):
        raise PermissionError("Diagnostic subset differs from its frozen registration")
    selected = set(indices)
    if any(0 < len(selected & set(range(start, start + 8))) < 8 for start in range(0, 128, 8)):
        raise PermissionError("Diagnostic subset must retain each full G8")
    if len({records[index]["qid"] for index in indices}) != 8:
        raise PermissionError("Diagnostic subset must contain eight distinct qids")
    return indices


def _validate_optimizer_cursor(optimizer, scheduler, parameters, committed_step):
    if scheduler.last_epoch != committed_step:
        raise PermissionError("Scheduler cursor differs from committed step")
    if committed_step == 0:
        if optimizer.state:
            raise PermissionError("First update must use a fresh optimizer without inherited state")
        return
    if set(optimizer.state) != set(parameters):
        raise PermissionError("Adam state is incomplete for the committed adapter")
    if any(float(optimizer.state[p].get("step", -1)) != committed_step for p in parameters):
        raise PermissionError("Adam logical step differs from committed checkpoint")


def _verify_prepared(record, prepared):
    import torch

    if prepared.get("routing", {}).get("input_tensor_hash") != record["prompt_tensor_hash"]:
        raise PermissionError("Actual native prepared prompt tensor hash differs")
    inputs = prepared["inputs"]
    native = {
        key: value.detach().cpu().tolist()
        for key, value in inputs.items()
        if key != "pixel_values" and isinstance(value, torch.Tensor)
    }
    if native != record.get("native_prompt_inputs"):
        raise PermissionError("Native prompt token/mask/grid trajectory changed")
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Expected exactly one original native prompt")
    if inputs["input_ids"].shape[1] != record.get("prompt_token_count"):
        raise PermissionError("Native prompt/completion boundary changed")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in inputs.values()
        if isinstance(value, torch.Tensor)
    ):
        raise PermissionError("Prepared floating inputs must be FP32")


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _numerical_audit(traces):
    audit = numerical_audit(traces)
    if traces:
        delta = [
            abs(a - b)
            for row in traces
            for a, b in zip(row["train_logprobs"], row["sampler_logprobs"], strict=True)
        ]
        audit["sampler_train_threshold_counts"] = {
            str(bound): sum(value > bound for value in delta) for bound in (0.01, 1.0, 5.0)
        }
    return audit


def _norm(parameters):
    return math.sqrt(
        sum(
            float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None
        )
    )


def _channel_summary(channels):
    vectors = dict(channels)
    vectors["total"] = [sum(items) for items in zip(*channels.values(), strict=True)]
    squared = {
        name: sum(float(t.double().square().sum()) for t in tensors)
        for name, tensors in vectors.items()
    }
    dots = {}
    cosines = {}
    for left, right in (("native", "aux"), ("native", "kl"), ("aux", "kl")):
        key = f"{left}_{right}"
        dot = sum(
            float((a.double() * b.double()).sum())
            for a, b in zip(vectors[left], vectors[right], strict=True)
        )
        dots[key] = dot
        denominator = math.sqrt(squared[left] * squared[right])
        cosines[key] = dot / denominator if denominator else None
    return {
        "norms": {key: math.sqrt(value) for key, value in squared.items()},
        "dot_products": dots,
        "cosines": cosines,
        "total_definition": "sum of native, aux and weighted KL vectors on the same subset",
        "normalization": "subset contributions to the full /128 update",
    }


def update(
    runtime,
    optimizer,
    scheduler,
    records,
    group_sizes,
    reference,
    credits,
    *,
    diagnostic_row_indices=(),
):
    """Require a live original-token access session for an actual new-phase model."""
    source = getattr(runtime, "_srf6_source_context", None)
    permit = nullcontext()
    from .runtime import _LOADED_RUNTIMES, Runtime, _check_loaded_source_identity

    if isinstance(runtime, Runtime) and runtime in _LOADED_RUNTIMES:
        _check_loaded_source_identity(runtime)
    if source is not None:
        if _check_loaded_source_identity(runtime) != "CENSUS_S0":
            from .access import _UPDATE_ACCESS, active_session

            permit = active_session(runtime).update(
                records, group_sizes, reference, credits, access=_UPDATE_ACCESS
            )
    elif hasattr(runtime, "_srf6_execution_identity"):
        raise PermissionError("Actual loaded source is missing; cannot downgrade to S0")
    with permit:
        return _update(
            runtime,
            optimizer,
            scheduler,
            records,
            group_sizes,
            reference,
            credits,
            diagnostic_row_indices=diagnostic_row_indices,
        )


def _update(
    runtime,
    optimizer,
    scheduler,
    records,
    group_sizes,
    reference,
    credits,
    *,
    diagnostic_row_indices=(),
):
    """Accumulate one /128 update, enforcing every probability gate before Adam.

    Required runtime identity includes run_root, source_sha256, reference_hash,
    reference_parent, run_id, parent_id, and committed step. At diagnostic steps
    it also includes the frozen 64 diagnostic_row_indices. Each row's step is
    the pending step, while adapter_identity.step is the committed source step.
    Later phases also require a separate audit_root below the shared run_root;
    public image paths continue to resolve relative to run_root.
    Successful return is UNCOMMITTED: this function does not advance identity.
    """
    import torch

    root, audit_path = _audit_path(runtime)
    started, rng = time.monotonic(), capture_rng()
    old_training = runtime.model.training
    before = trainable_state(runtime.model)
    parameters = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    traces, totals = [], {name: 0.0 for name in ("loss", "native", "aux", "kl")}
    audit_identity = {
        "at": now(),
        "task_id": runtime.identity.get("task_id"),
        "run_id": runtime.identity.get("run_id"),
        "committed_source_step": runtime.identity.get("step"),
        "arm": credits[0].get("arm") if credits else None,
        "policy_before_sha256": state_hash(before),
        "audit_root": str(audit_path.parent),
    }
    optimizer_started = False
    try:
        validate_optimizer(optimizer, scheduler, model=runtime.model)
        if not finite_tree(optimizer.state_dict()) or not finite_tree(scheduler.state_dict()):
            raise FloatingPointError("Nonfinite committed optimizer/scheduler state")
        coefficients = _validate_records(
            runtime, records, group_sizes, reference, credits, root, before
        )
        _validate_optimizer_cursor(optimizer, scheduler, parameters, runtime.identity["step"])
        indices = _diagnostic_indices(runtime, records, diagnostic_row_indices)
        channels = (
            {
                name: [torch.zeros_like(p, device="cpu") for p in parameters]
                for name in ("native", "aux", "kl")
            }
            if indices
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        runtime.model.eval()
        previous_key, prepared = None, None
        for index, (record, coefficient) in enumerate(zip(records, coefficients, strict=True)):
            key = (record["qid"], digest(record["model_input"]), record["protocol"])
            if key != previous_key:
                prepared = runtime.prepare_input(
                    record["model_input"], root, protocol=record["protocol"]
                )
                previous_key = key
            _verify_prepared(record, prepared)
            item = dict(record, prepared=prepared)
            with torch.no_grad():
                references = runtime.batch_reference_forward([item], reference)
            currents = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)
            if len(currents) != 1 or len(references) != 1:
                raise ValueError("Microbatch=1 returned an invalid row count")
            current, ref = currents[0], references[0]
            sampler = torch.tensor(
                record["sampler_logprobs"], device=current.device, dtype=torch.float32
            )
            if (
                current.ndim != 1
                or current.numel() != len(record["tokens"])
                or ref.shape != current.shape
            ):
                raise ValueError("Teacher forcing must cover every original completion token")
            traces.append(
                {
                    "row_id": record["row_id"],
                    "qid": record["qid"],
                    "tokens": list(record["tokens"]),
                    "sampler_logprobs": sampler.detach().cpu().tolist(),
                    "train_logprobs": current.detach().cpu().tolist(),
                    "reference_logprobs": ref.detach().cpu().tolist(),
                    "token_path_sha256": record["token_path_sha256"],
                }
            )
            result = sequence_loss(current, sampler, ref, coefficient["native"], coefficient["aux"])
            if index in indices:
                diagnostic_rng = capture_rng()
                try:
                    for name, target in channels.items():
                        gradients = torch.autograd.grad(
                            result[name] / 128, parameters, retain_graph=True, allow_unused=True
                        )
                        for buffer, gradient in zip(target, gradients, strict=True):
                            if gradient is not None:
                                if not bool(torch.isfinite(gradient).all()):
                                    raise FloatingPointError(
                                        "Nonfinite diagnostic channel gradient"
                                    )
                                buffer.add_(gradient.detach().cpu())
                finally:
                    restore_rng(diagnostic_rng)
                if hasattr(runtime, "reserve"):
                    runtime.reserve(
                        "diagnostic_gradient_calls",
                        3,
                        row_id=record["row_id"],
                        step=runtime.identity["step"] + 1,
                    )
            (result["loss"] / 128).backward()
            for name in totals:
                totals[name] += float(result[name].detach()) / 128
        audit = _numerical_audit(traces)
        if audit["status"] != "FINITE":
            raise ProbabilityMismatch("Nonfinite actual token probability/KL audit", audit)
        enforce_probability_gate(audit)
        if audit["sampler_train_max_absolute_difference"] > 5:
            raise ProbabilityMismatch("Token sampler/train discrepancy exceeds 5 nat", audit)
        if state_hash(trainable_state(runtime.model)) != audit_identity["policy_before_sha256"]:
            raise PermissionError("A forward pass changed the committed adapter before Adam")
        if state_hash(reference) != runtime.identity["reference_hash"]:
            raise PermissionError("A forward pass changed the frozen parent reference")
        if not any(parameter.grad is not None for parameter in parameters):
            raise RuntimeError("Actual forwards produced no trainable adapter gradient")
        norm_before = float(
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        )
        norm_after = _norm(parameters)
        channel_report = (
            {
                "status": "MEASURED",
                "row_indices": list(indices),
                "qid_count": 8,
                "parameter_names": list(before),
                **_channel_summary(channels),
            }
            if channels
            else {"status": "NOT_SCHEDULED"}
        )
        if not finite_tree(channel_report):
            raise FloatingPointError("Nonfinite accumulated diagnostic channel")
        append(
            audit_path,
            _safe(
                {
                    **audit_identity,
                    "status": "PREUPDATE_PASS",
                    "diagnostics": audit,
                    "channel_gradients": channel_report,
                }
            ),
        )
        optimizer_started = True
        optimizer.step()
        scheduler.step()
        if not finite_tree(optimizer.state_dict()) or not finite_tree(
            trainable_state(runtime.model)
        ):
            raise FloatingPointError(
                "Nonfinite adapter/Adam after update; restore committed checkpoint"
            )
        update_norm = math.sqrt(
            sum(
                float((p.detach().cpu().double() - before[name].double()).square().sum())
                for name, p in runtime.model.named_parameters()
                if p.requires_grad
            )
        )
        restore_rng(rng)
        restored_rng_hash = state_hash(capture_rng())
        if restored_rng_hash != state_hash(rng):
            raise RuntimeError("Update failed to restore the original RNG state")
        report = {
            **totals,
            **audit,
            "status": "UPDATED_UNCOMMITTED",
            "sequence_count": 128,
            "optimizer_updates": 1,
            "group_sizes": list(group_sizes),
            "microbatch_size": 1,
            "kl_coefficient": 0.02,
            "gradient_norm_before_clip": norm_before,
            "gradient_norm_after_clip": norm_after,
            "parameter_update_norm": update_norm,
            "channel_gradients": channel_report,
            "probability_vectors": traces,
            "actual_credit_groups": copy.deepcopy(credits),
            "policy_before_sha256": audit_identity["policy_before_sha256"],
            "policy_after_sha256": state_hash(trainable_state(runtime.model)),
            "reference_sha256": runtime.identity["reference_hash"],
            "rng_before_sha256": state_hash(rng),
            "rng_after_sha256": restored_rng_hash,
            "rng_restored": True,
            "training_seconds": time.monotonic() - started,
            "loss_reduction": "mean of all 128 original completion token means",
        }
        return report
    except Exception as exc:
        audit = _numerical_audit(traces)
        parameter_state_changed = (
            state_hash(trainable_state(runtime.model)) != audit_identity["policy_before_sha256"]
        )
        originals = [
            {
                key: record.get(key)
                for key in (
                    "row_id",
                    "qid",
                    "tokens",
                    "sampler_logprobs",
                    "token_path_sha256",
                    "policy_hash",
                    "prompt_tensor_hash",
                    "source_sha256",
                )
            }
            for record in records
            if isinstance(record, dict)
        ]
        append(
            audit_path,
            _safe(
                {
                    **audit_identity,
                    "status": "UPDATE_ATTEMPT_FAILED_RESTORE_CHECKPOINT"
                    if optimizer_started
                    else "UNCOMMITTED_FAILURE_RESTORE_CHECKPOINT"
                    if parameter_state_changed
                    else "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP",
                    "optimizer_started": optimizer_started,
                    "parameter_state_changed": parameter_state_changed,
                    "requires_checkpoint_restore": optimizer_started or parameter_state_changed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "diagnostics": audit,
                    "probability_vectors": traces,
                    "original_trajectories": originals,
                }
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        if isinstance(exc, ProbabilityMismatch):
            exc.diagnostics.update(probability_vectors=traces)
        raise
    finally:
        runtime.model.train(old_training)
        restore_rng(rng)
