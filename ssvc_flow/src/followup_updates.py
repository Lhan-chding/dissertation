"""Follow-up joint reward interventions and isolated complete-state Adam forks.

Historical modules remain byte-for-byte unchanged. The gradient loop delegates
its FP32 PPO objective and validation to their existing functions. Checkpoints
store trainable parameters plus the immutable frozen-weight fingerprint; they
must be restored against the same frozen base. Forward-only aliases deliberately
exclude Adam/RNG/sampler, and require a separate runtime input/processor binding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from . import fork_gradients as legacy
from . import optimizer_fork as old_state
from .grpo_update import grouped_advantages, reward_channels, torch_ppo_loss
from .optimizer_fork import parameter_hash, state_hash

SCHEMA = "ssvc-followup-complete-state-v1"
POSITION_ATTRIBUTES = ("rope_deltas", "cache_position", "position_ids", "past_key_values", "_cache")


def _unusable(adapter):
    return getattr(adapter, "_followup_state_unusable", False) or getattr(
        adapter, "_followup_unusable", False
    )


def _poison(adapter):
    if adapter is not None:
        adapter._followup_state_unusable = True
        adapter._followup_unusable = True


def _number(value, name, *, positive=False):
    try:
        return legacy._number(value, name, positive=positive)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error


def group_reward_statistics(categories, *, auxiliary_weight=0, policy="joint", epsilon=1e-4):
    """Apply the per-group gate before one population joint standardization."""
    if isinstance(categories, (str, bytes)):
        raise ValueError("categories must be a nonempty sequence of category names")
    categories = list(categories)
    if not categories or any(c not in legacy.CATEGORIES for c in categories):
        raise ValueError("Nonempty X/S/W/I categories required")
    if policy not in ("joint", "no_x_off"):
        raise ValueError("Unknown group reward policy")
    lam = _number(auxiliary_weight, "auxiliary_weight")
    epsilon = _number(epsilon, "epsilon", positive=True)
    counts = {c: categories.count(c) for c in legacy.CATEGORIES}
    effective = 0.0 if policy == "no_x_off" and not counts["X"] else lam
    values = [reward_channels(c, "X_BASE", effective)["sum"] for c in categories]
    try:
        normalized = grouped_advantages(values, epsilon)
    except OverflowError as error:
        raise ValueError("Reward variance overflow") from error
    if not all(
        math.isfinite(float(v))
        for v in [normalized["mean"], normalized["variance"], *normalized["advantages"]]
    ):
        raise ValueError("Reward statistics must remain finite")
    return {
        "policy": policy,
        "auxiliary_weight": lam,
        "effective_auxiliary_weight": effective,
        "category_counts": counts,
        "reward_vector": values,
        **normalized,
    }


def _safe(value, where="state"):
    """Accept only weights_only-compatible builtins/tensors; reject nonfinite data."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_sparse or value.is_complex() or not torch.isfinite(value).all():
            raise ValueError(f"Finite dense noncomplex tensor required in {where}")
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ValueError(f"Invalid dictionary key in {where}")
            _safe(item, where)
    elif type(value) in (list, tuple):
        for item in value:
            _safe(item, where)
    elif (
        value is None
        or type(value) in (str, bool, int)
        or (type(value) is float and math.isfinite(value))
    ):
        return
    else:
        raise ValueError(f"Unsupported type or nonfinite scalar in {where}: {type(value).__name__}")


def _forward_state(model, adapter=None):
    buffers, modes, positions = {}, {}, {}
    for name, module in model.named_modules():
        modes[name] = module.training
        for key, value in module._buffers.items():
            buffers[f"{name}.{key}" if name else key] = old_state._cpu_copy(value)
        for key in POSITION_ATTRIBUTES:
            if hasattr(module, key) and key not in module._buffers:
                positions[f"{name}:{key}"] = old_state._cpu_copy(getattr(module, key))
    adapter_positions = {
        key: old_state._cpu_copy(getattr(adapter, key))
        for key in POSITION_ATTRIBUTES
        if adapter is not None and hasattr(adapter, key)
    }
    result = {
        "buffers": buffers,
        "module_modes": modes,
        "position_state": positions,
        "adapter_position_state": adapter_positions,
    }
    _safe(result, "forward state")
    return result


def _restore_forward(model, state, adapter=None):
    modules = dict(model.named_modules())
    if set(modules) != set(state["module_modes"]):
        raise ValueError("Checkpoint module topology mismatch")
    if adapter is not None and hasattr(adapter, "_reset_positions"):
        adapter._reset_positions()
    # Known cache/position attributes may be lazily created by forward. Absence
    # at origin is part of the state, so remove attributes created by a candidate.
    for name, module in modules.items():
        for key in POSITION_ATTRIBUTES:
            if (
                key not in module._buffers
                and f"{name}:{key}" not in state["position_state"]
                and hasattr(module, key)
            ):
                delattr(module, key)
    if adapter is not None:
        for key in POSITION_ATTRIBUTES:
            if key not in state.get("adapter_position_state", {}) and hasattr(adapter, key):
                delattr(adapter, key)
    for full_name, value in state["buffers"].items():
        module_name, _, key = full_name.rpartition(".")
        module = modules[module_name]
        if key not in module._buffers:
            raise ValueError(f"Checkpoint buffer name mismatch: {full_name}")
        current = module._buffers[key]
        if value is None:
            module._buffers[key] = None
        else:
            device = current.device if current is not None else next(model.parameters()).device
            module._buffers[key] = value.detach().to(device=device).clone()
    for full_name, value in state["position_state"].items():
        module_name, key = full_name.rsplit(":", 1)
        module = modules[module_name]
        current = getattr(module, key, None)
        restored = old_state._cpu_copy(value)
        if hasattr(restored, "to"):
            device = (
                current.device if hasattr(current, "device") else next(model.parameters()).device
            )
            restored = restored.to(device)
        setattr(module, key, restored)
    for key, value in state.get("adapter_position_state", {}).items():
        if adapter is None:
            raise ValueError("Checkpoint requires an adapter position-state target")
        current = getattr(adapter, key, None)
        restored = old_state._cpu_copy(value)
        if hasattr(restored, "to") and hasattr(current, "device"):
            restored = restored.to(current.device)
        setattr(adapter, key, restored)
    for name, training in state["module_modes"].items():
        modules[name].training = training


def _finite_parameters(model):
    import torch

    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite model parameters")


def _ownership(model, optimizer):
    import torch

    parameters = legacy._parameters(model)
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise ValueError("Follow-up requires Adam/AdamW")
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if (
        len(owned) != len(parameters)
        or len({id(p) for p in owned}) != len(owned)
        or set(map(id, owned)) != set(map(id, parameters.values()))
    ):
        raise ValueError(
            "Optimizer must own every trainable parameter exactly once and no frozen parameters"
        )
    names = {id(p): name for name, p in parameters.items()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def direct_loss_gradients(
    adapter, groups, *, policy="joint", auxiliary_weight=0, lnorm=64, epsilon=1e-4, clip_epsilon=0.2
):
    """Real clipped PPO gradients with unchanged B*K*Lnorm denominator.

    Only exact sampled tokens in token_ids are scored: real EOS stays, padding
    must already be excluded by the validated raw-row adapter boundary. I rows
    and suppressed groups still contribute to the sequence denominator.
    """
    import torch

    if _unusable(adapter):
        raise RuntimeError("Adapter state is unusable after a failed restore")
    binding = legacy._bank_binding(groups)
    lnorm = _number(lnorm, "lnorm", positive=True)
    clip_epsilon = _number(clip_epsilon, "clip_epsilon", positive=True)
    if clip_epsilon >= 1:
        raise ValueError("clip_epsilon must be below one")
    statistics = [
        dict(
            prompt_id=g[0]["prompt_id"],
            **group_reward_statistics(
                [r["category"] for r in g],
                policy=policy,
                auxiliary_weight=auxiliary_weight,
                epsilon=epsilon,
            ),
        )
        for g in groups
    ]
    parameters = legacy._parameters(adapter.model)
    _finite_parameters(adapter.model)
    before = _forward_state(adapter.model, adapter)
    origin = parameter_hash(adapter.model, trainable=True)
    records, losses, advantages = [], [], []
    failure = None
    try:
        adapter.model.zero_grad(set_to_none=True)
        for group, stats in zip(groups, statistics, strict=True):
            for row, advantage in zip(group, stats["advantages"], strict=True):
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
                    raise FloatingPointError("Nonfinite follow-up direct loss")
                loss.backward()
                losses.append(float(loss.detach()))
                advantages.append(advantage)
        if any(p.grad is not None for p in adapter.model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameter received gradient")
        gradients = legacy._copy_gradients(parameters)
        if parameter_hash(adapter.model, trainable=True) != origin:
            raise RuntimeError("Parameters changed during gradient computation")
        return {
            "gradients": gradients,
            "audit": {
                **binding,
                **legacy._ratio_audit(records, clip_epsilon, advantages),
                "parameter_hash": origin,
                "policy": policy,
                "arm": "X_BASE",
                "auxiliary_weight": statistics[0]["auxiliary_weight"],
                "epsilon": statistics[0]["epsilon"],
                "Lnorm": lnorm,
                "gradient_stage": "PRE_CLIP",
                "loss": math.fsum(losses),
                "grad_norm": legacy._norm(gradients),
                "gradient_hash": state_hash(gradients),
                "backward_calls": len(losses),
                "optimizer_updates": 0,
                "microbatch": 1,
                "group_statistics": statistics,
                "ratio_records": records,
                "old_probabilities_rewritten": False,
            },
        }
    except BaseException as error:
        failure = error
        raise
    finally:
        adapter.model.zero_grad(set_to_none=True)
        try:
            _restore_forward(adapter.model, before, adapter)
        except BaseException as restore_error:
            _poison(adapter)
            if failure is None:
                raise
            failure.add_note(f"Forward state cleanup failed; adapter unusable: {restore_error}")


def apply_gradient_update(model, optimizer, gradients, grad_clip=1):
    """Prevalidate finite model/ownership, then use the unchanged historical Adam."""
    _ownership(model, optimizer)
    _finite_parameters(model)
    return legacy.apply_gradient_update(model, optimizer, gradients, grad_clip)


def _sampler_state(sampler):
    if sampler is None:
        return None
    if isinstance(sampler, Mapping):
        return {"kind": "mapping", "state": old_state._cpu_copy(dict(sampler))}
    if callable(getattr(sampler, "state_dict", None)) and callable(
        getattr(sampler, "load_state_dict", None)
    ):
        return {"kind": "state_dict", "state": old_state._cpu_copy(sampler.state_dict())}
    raise ValueError("Sampler must be a mutable mapping or expose state_dict/load_state_dict")


def capture_state(model, optimizer, metadata=None, *, adapter=None, sampler=None):
    """Capture parameters, Adam, all RNG, registered buffers, positions and modes."""
    names = _ownership(model, optimizer)
    _finite_parameters(model)
    legacy._finite_adam_state(optimizer)
    state = {
        **old_state.capture_state(model, optimizer, metadata),
        **_forward_state(model, adapter),
        "schema": SCHEMA,
        "frozen_parameter_hash": parameter_hash(model, trainable=False),
        "optimizer_parameter_names": names,
        "sampler": _sampler_state(sampler),
    }
    _validate_state(state, model, optimizer)
    return state


def _validate_state(state, model=None, optimizer=None):
    import numpy as np
    import torch

    if type(state) is not dict or state.get("schema") != SCHEMA:
        raise ValueError("Checkpoint complete-state schema mismatch")
    required = {
        "parameters",
        "optimizer",
        "scheduler",
        "rng",
        "metadata",
        "buffers",
        "module_modes",
        "position_state",
        "adapter_position_state",
        "frozen_parameter_hash",
        "sampler",
        "optimizer_parameter_names",
    }
    if not required <= set(state):
        raise ValueError("Checkpoint complete-state fields missing")
    _safe(state, "checkpoint")
    parameters = state["parameters"]
    if (
        not isinstance(parameters, dict)
        or not parameters
        or any(
            not isinstance(p, torch.Tensor) or not p.is_floating_point()
            for p in parameters.values()
        )
    ):
        raise ValueError("Checkpoint requires named floating trainable tensors")
    if any(type(v) is not bool for v in state["module_modes"].values()):
        raise ValueError("Checkpoint training modes must be boolean")
    if state["scheduler"] is not None:
        raise ValueError("Follow-up scheduler must remain unconfigured")
    rng = state["rng"]
    if type(rng) is not dict or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Checkpoint RNG fields mismatch")
    try:
        random.Random().setstate(rng["python"])
        numpy_rng = rng["numpy"]
        np.random.RandomState().set_state(
            (numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:])
        )
        if (
            not isinstance(rng["torch"], torch.Tensor)
            or rng["torch"].dtype != torch.uint8
            or rng["torch"].ndim != 1
        ):
            raise ValueError("CPU RNG must be a uint8 state vector")
        torch.Generator(device="cpu").set_state(rng["torch"])
        if type(rng["cuda"]) is not list or any(
            not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1
            for v in rng["cuda"]
        ):
            raise ValueError("CUDA RNG must contain uint8 state vectors")
    except (TypeError, ValueError, RuntimeError, IndexError) as error:
        raise ValueError(f"Checkpoint RNG type/state mismatch: {error}") from error
    groups = state["optimizer"]["param_groups"]
    group_names = state["optimizer_parameter_names"]
    if len(groups) != len(group_names) or any(
        len(g["params"]) != len(n) for g, n in zip(groups, group_names, strict=True)
    ):
        raise ValueError("Checkpoint optimizer ownership layout mismatch")
    flat_ids = [i for g in groups for i in g["params"]]
    flat_names = [n for g in group_names for n in g]
    if (
        len(set(flat_ids)) != len(flat_ids)
        or len(set(flat_names)) != len(flat_names)
        or set(flat_names) != set(parameters)
    ):
        raise ValueError("Checkpoint optimizer ownership is not complete/unique")
    if not set(state["optimizer"]["state"]) <= set(flat_ids):
        raise ValueError("Checkpoint optimizer contains unowned state")
    for index, name in zip(flat_ids, flat_names, strict=True):
        entry = state["optimizer"]["state"].get(index, {})
        if entry and not {"step", "exp_avg", "exp_avg_sq"} <= set(entry):
            raise ValueError("Checkpoint Adam state fields missing")
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in entry and (
                not isinstance(entry[key], torch.Tensor)
                or entry[key].shape != parameters[name].shape
                or entry[key].dtype != parameters[name].dtype
            ):
                raise ValueError(f"Checkpoint optimizer tensor shape/dtype mismatch: {name}/{key}")
        if entry:
            step = entry["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise ValueError("Checkpoint Adam step must be scalar")
                step = step.item()
            if (
                isinstance(step, bool)
                or not isinstance(step, (float, int))
                or step < 0
                or int(step) != step
            ):
                raise ValueError("Checkpoint Adam step must be a nonnegative integer")
    if model is not None:
        actual = legacy._parameters(model)
        if set(actual) != set(parameters):
            raise ValueError("Checkpoint parameter names mismatch")
        for name, parameter in actual.items():
            if (
                parameter.shape != parameters[name].shape
                or parameter.dtype != parameters[name].dtype
            ):
                raise ValueError(f"Checkpoint parameter shape/dtype mismatch: {name}")
        live_forward = _forward_state(model)
        if set(live_forward["buffers"]) != set(state["buffers"]) or set(
            live_forward["module_modes"]
        ) != set(state["module_modes"]):
            raise ValueError("Checkpoint buffer/module topology mismatch")
        for name, value in state["buffers"].items():
            live = live_forward["buffers"][name]
            if (
                value is not None
                and live is not None
                and (value.shape != live.shape or value.dtype != live.dtype)
            ):
                raise ValueError(f"Checkpoint buffer shape/dtype mismatch: {name}")
        if state["frozen_parameter_hash"] != parameter_hash(model, trainable=False):
            raise ValueError("Checkpoint frozen model fingerprint mismatch")
    if optimizer is not None:
        if model is None:
            raise ValueError("Optimizer validation also requires model")
        if _ownership(model, optimizer) != group_names:
            raise ValueError("Checkpoint optimizer named parameter ordering mismatch")
    return state


def restore_state(model, optimizer, state, *, adapter=None, sampler=None):
    """Validate before mutation, restore complete state, and verify exact equality.

    A failed mutation/verification poisons the adapter; subsequent candidates are
    refused. The caller must reload a trusted checkpoint into a fresh adapter.
    """
    _validate_state(state, model, optimizer)
    snapshot_sampler = state["sampler"]
    if (sampler is None) != (snapshot_sampler is None):
        raise ValueError("Checkpoint sampler target mismatch")
    if snapshot_sampler is not None:
        if snapshot_sampler["kind"] == "mapping" and not isinstance(sampler, MutableMapping):
            raise ValueError("Checkpoint sampler requires mutable mapping")
        if snapshot_sampler["kind"] == "state_dict" and not callable(
            getattr(sampler, "load_state_dict", None)
        ):
            raise ValueError("Checkpoint sampler requires load_state_dict")
    if state["adapter_position_state"] and adapter is None:
        raise ValueError("Checkpoint requires adapter position-state target")
    try:
        old_state.restore_state(model, optimizer, state)
        _restore_forward(model, state, adapter)
        if snapshot_sampler is not None:
            value = old_state._cpu_copy(snapshot_sampler["state"])
            if snapshot_sampler["kind"] == "mapping":
                sampler.clear()
                sampler.update(value)
            elif snapshot_sampler["kind"] == "state_dict":
                sampler.load_state_dict(value)
            else:
                raise ValueError("Unknown checkpoint sampler state kind")
        model.zero_grad(set_to_none=True)
        restored = capture_state(
            model, optimizer, state["metadata"], adapter=adapter, sampler=sampler
        )
        if state_hash(restored) != state_hash(state):
            raise RuntimeError("Full restored state checksum mismatch")
    except BaseException:
        _poison(adapter)
        raise
    return copy.deepcopy(state["metadata"])


def save_checkpoint(path, state, identity):
    """No-clobber atomic publication, returning file/content hashes."""
    import torch

    _validate_state(state)
    _safe(identity, "identity")
    if type(identity) is not dict or not identity:
        raise ValueError("Nonempty checkpoint identity mapping required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = {"identity": copy.deepcopy(identity), "state": state, "state_hash": state_hash(state)}
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        # Read the written format with the restricted loader before publication.
        verified = torch.load(temporary, map_location="cpu", weights_only=True)
        if (
            verified["identity"] != identity
            or state_hash(verified["state"]) != payload["state_hash"]
        ):
            raise ValueError("Checkpoint serialized identity/hash mismatch")
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(path), "sha256": _file_hash(path), "state_hash": payload["state_hash"]}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(
    path, expected_identity, *, model=None, optimizer=None, expected_file_sha256=None
):
    import torch

    if expected_file_sha256 is not None and _file_hash(path) != expected_file_sha256:
        raise ValueError("Checkpoint file SHA-256 mismatch before deserialization")
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if type(payload) is not dict or set(payload) != {"identity", "state", "state_hash"}:
        raise ValueError("Checkpoint payload type/fields mismatch")
    if payload["identity"] != expected_identity:
        raise ValueError("Checkpoint identity mismatch")
    if state_hash(payload["state"]) != payload["state_hash"]:
        raise ValueError("Checkpoint state checksum mismatch")
    return _validate_state(payload["state"], model, optimizer)


def forward_state_hash(state):
    """Exact inference-state fingerprint; training checkpoints remain independent."""
    required = ("parameters", "frozen_parameter_hash", "buffers", "module_modes", "position_state")
    if not all(k in state for k in required):
        raise ValueError("Complete forward state required for exact alias")
    return state_hash(
        {
            **{k: state[k] for k in required},
            "adapter_position_state": state.get("adapter_position_state", {}),
        }
    )


def compare_update_vectors(baseline_state, left_state, right_state):
    """Stream CPU FP64 tensor displacements relative to the joint0 baseline.

    relative_vector_difference uses the LEFT displacement norm. Zero vectors
    have undefined cosine; a zero left norm has undefined relative difference.
    These are measurements only and never authorize inference reuse.
    """
    import torch

    states = (baseline_state, left_state, right_state)
    params = [s["parameters"] for s in states]
    if not params[0] or any(set(p) != set(params[0]) for p in params[1:]):
        raise ValueError("Candidate parameter names mismatch")
    left_squares, right_squares, difference_squares, dots, maxima = [], [], [], [], []
    for name in sorted(params[0]):
        values = [p[name] for p in params]
        if any(
            not isinstance(v, torch.Tensor)
            or not v.is_floating_point()
            or v.shape != values[0].shape
            or not torch.isfinite(v).all()
            for v in values
        ):
            raise ValueError(f"Finite matching parameter tensors required: {name}")
        base, left, right = [v.detach().to(device="cpu", dtype=torch.float64) for v in values]
        dl, dr = left - base, right - base
        difference = dl - dr
        left_squares.append(float(dl.square().sum()))
        right_squares.append(float(dr.square().sum()))
        difference_squares.append(float(difference.square().sum()))
        dots.append(float((dl * dr).sum()))
        maxima.append(float(difference.abs().max()) if difference.numel() else 0.0)
    ln, rn = math.sqrt(math.fsum(left_squares)), math.sqrt(math.fsum(right_squares))
    difference = math.sqrt(math.fsum(difference_squares))
    return {
        "left_aux_l2": ln,
        "right_aux_l2": rn,
        "aux_l2": {"left": ln, "right": rn},
        "vector_difference_l2": difference,
        "max_abs_difference": max(maxima),
        "cosine": max(-1.0, min(1.0, math.fsum(dots) / (ln * rn))) if ln and rn else None,
        "relative_vector_difference": difference / ln if ln else None,
        "relative_denominator": "left_aux_l2",
        "accumulation_dtype": "CPU_FP64",
        "parameter_hashes": {
            n: state_hash(s["parameters"])
            for n, s in zip(("baseline", "left", "right"), states, strict=True)
        },
        "optimizer_hashes": {
            n: state_hash(s["optimizer"])
            for n, s in zip(("baseline", "left", "right"), states, strict=True)
        },
        "alias_authorized": False,
    }


def _optimizer_steps(state):
    return sorted({int(float(entry["step"])) for entry in state["optimizer"]["state"].values()})


def fork_one_candidate(
    adapter, optimizer, origin, groups, candidate_spec, *, sampler=None, failure_path=None
):
    """One scratch Adam update, complete candidate snapshot, unconditional restore."""
    if _unusable(adapter):
        raise RuntimeError("Adapter state is unusable after a failed restore")
    if not isinstance(candidate_spec, Mapping):
        raise ValueError("Candidate specification must be a mapping")
    policy = candidate_spec.get("policy", "joint")
    lam = candidate_spec.get("auxiliary_weight", candidate_spec.get("lambda", 0))
    failure = None
    result = None
    try:
        restore_state(adapter.model, optimizer, origin, adapter=adapter, sampler=sampler)
        gradients = direct_loss_gradients(adapter, groups, policy=policy, auxiliary_weight=lam)
        update = apply_gradient_update(adapter.model, optimizer, gradients["gradients"])
        metadata = {
            **origin["metadata"],
            "candidate_spec": dict(candidate_spec),
            "permanent_training_commit": False,
        }
        candidate = capture_state(
            adapter.model, optimizer, metadata, adapter=adapter, sampler=sampler
        )
        result = {
            "state": candidate,
            "audit": {
                **gradients["audit"],
                **update,
                "origin_checkpoint_step": origin["metadata"].get(
                    "checkpoint_step", origin["metadata"].get("step")
                ),
                "origin_optimizer_steps": _optimizer_steps(origin),
                "candidate_optimizer_steps": _optimizer_steps(candidate),
                "candidate_optimizer_step": _optimizer_steps(candidate)[0]
                if len(_optimizer_steps(candidate)) == 1
                else None,
                "permanent_training_commit": False,
                "origin_state_hash": state_hash(origin),
                "candidate_state_hash": state_hash(candidate),
                "forward_state_hash": forward_state_hash(candidate),
            },
        }
    except BaseException as error:
        failure = error
    finally:
        try:
            restore_state(adapter.model, optimizer, origin, adapter=adapter, sampler=sampler)
        except BaseException as restore_error:
            _poison(adapter)
            if failure is None:
                failure = restore_error
            else:
                failure.add_note(f"Origin restore failed; adapter unusable: {restore_error}")
        if failure is not None:
            record = {
                "status": "FAILED",
                "error_type": type(failure).__name__,
                "error": str(failure),
                "candidate_spec": dict(candidate_spec),
                "origin_restored": not _unusable(adapter),
                "permanent_training_commit": False,
            }
            adapter._followup_last_failure = record
            if failure_path is not None:
                try:
                    path = Path(failure_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("x") as stream:
                        json.dump(record, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                except Exception as record_error:
                    failure.add_note(f"Failure record could not be persisted: {record_error}")
    if failure is not None:
        raise failure
    return result
