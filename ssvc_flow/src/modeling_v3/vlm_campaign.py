"""V3 Qwen9B campaign: non-loading plans and explicitly gated real execution.

The production loader and Adam objective are inherited without editing their
historical certificates. V3 records its new initialization and schedule as new
contracts. All model imports occur behind the invocation authorization gate.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..core import PROJECT_ROOT, file_hash, frozen_writer, write_json
from .io import canonical_hash

NAMESPACE = "SSVC_MODELING_V3_20260915"
SOURCE_ARMS = {"X_BASE": ("joint", 0.0), "X_VALID": ("joint", 1.0)}
CANDIDATES = (
    {"id": "joint_0", "policy": "joint", "auxiliary_weight": 0.0},
    {"id": "joint_1", "policy": "joint", "auxiliary_weight": 1.0},
    {"id": "no_x_off_1", "policy": "no_x_off", "auxiliary_weight": 1.0},
)
CONTRASTS = (
    ("joint_1_minus_joint_0", "joint_1", "joint_0"),
    ("no_x_off_1_minus_joint_0", "no_x_off_1", "joint_0"),
    ("joint_1_minus_no_x_off_1", "joint_1", "no_x_off_1"),
)


def _read(path):
    from ..followup_protocol import strict_json

    return strict_json(path)


def _publish(path, value):
    path = Path(path)
    if path.exists():
        if canonical_hash(_read(path)) != canonical_hash(value):
            raise ValueError(f"Immutable V3 evidence differs: {path.name}")
    else:
        write_json(path, value)


def _bound_json(binding):
    if not isinstance(binding, dict) or not {"path", "sha256"} <= set(binding):
        raise ValueError("A path and SHA-256 binding are required")
    path = Path(binding["path"])
    if not path.is_file() or file_hash(path) != binding["sha256"]:
        raise ValueError(f"Bound input is missing or changed: {path.name}")
    return _read(path)


def source_hashes():
    return {
        str(path.relative_to(PROJECT_ROOT)): file_hash(path)
        for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
    }


def lora_initialization_seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Nonnegative integer source seed required")
    return int(canonical_hash([NAMESPACE, "lora_initialization", seed])[:16], 16) % (2**31)


def seed_role(config, seed):
    matches = [name for name, seeds in config["qwen"]["seed_roles"].items() if seed in seeds]
    if type(seed) is not int or len(matches) != 1:
        raise ValueError("Source seed must have exactly one frozen V3 role")
    return matches[0]


def build_training_schedule(train_scenes, *, seed, data_root=None):
    """512 unique positions, interleaved six strata, arm-independent ordering."""
    from ..r4_inputs import STRATA, _n_record, _validate_n

    lora_initialization_seed(seed)
    scenes = _validate_n(train_scenes, "train", 576)
    grouped = defaultdict(list)
    for scene in scenes:
        grouped[(scene["constraint_family"], scene["interface"])].append(scene)
    if {key: len(value) for key, value in grouped.items()} != {key: 96 for key in STRATA}:
        raise ValueError("V3 source requires 96 original train prompts in every stratum")
    for values in grouped.values():
        values.sort(
            key=lambda s: canonical_hash(
                [NAMESPACE, "source_order", seed, s["base_scene_id"], s["interface"]]
            )
        )
    prompts = [
        _n_record(grouped[group][i], group[1], data_root) for i in range(96) for group in STRATA
    ]
    consumed = [p["prompt_id"] for p in prompts[:512]]
    steps = [consumed[i : i + 4] for i in range(0, 512, 4)]
    return {
        "namespace": NAMESPACE,
        "seed": seed,
        "train_prompts": prompts,
        "train_steps": steps,
        "positions": 512,
        "available_prompts": 576,
        "schedule_hash": canonical_hash(steps),
        "lora_initialization_seed": lora_initialization_seed(seed),
        "generalization_unit": "training RNG plus independently seeded LoRA initialization",
    }


def build_bank_plan(train_prompts, *, origin_identity):
    """Partition 384/192, then sample 24/12 B4 banks without repeated prompts.

    Labels/rewards are neither arguments nor inspected. Bank allocation costs
    36 x B4 x K8 natural origin-policy rollouts and 108 real scratch Adam calls.
    """
    from ..r4_inputs import STRATA

    grouped = defaultdict(list)
    if len(train_prompts) != 576 or len({p["prompt_id"] for p in train_prompts}) != 576:
        raise ValueError("Bank plan requires all 576 distinct original prompts")
    for prompt in train_prompts:
        if prompt["split"] != "train":
            raise ValueError("Validation/confirm prompts cannot enter a training bank")
        grouped[(prompt["family"], prompt["interface"])].append(prompt)
    if {g: len(v) for g, v in grouped.items()} != {g: 96 for g in STRATA}:
        raise ValueError("Bank partition requires six balanced original strata")
    partition, selected, banks = {}, {}, []
    for role, left, right, per_group in (("calibration_pool", 0, 64, 16), ("heldout", 64, 96, 8)):
        role_groups = {}
        for group in STRATA:
            fixed = sorted(
                grouped[group],
                key=lambda p: canonical_hash([NAMESPACE, "train_partition", p["prompt_id"]]),
            )[left:right]
            role_groups[group] = sorted(
                fixed,
                key=lambda p: canonical_hash(
                    [NAMESPACE, "anchor_bank", origin_identity, role, p["prompt_id"]]
                ),
            )
        partition[role] = [p["prompt_id"] for g in STRATA for p in role_groups[g]]
        selected[role] = [role_groups[g][i]["prompt_id"] for i in range(per_group) for g in STRATA]
        for index in range(0, len(selected[role]), 4):
            bank_id = f"{role}_{index // 4:02d}"
            banks.append(
                {
                    "bank_id": bank_id,
                    "role": role,
                    "prompt_ids": selected[role][index : index + 4],
                    "candidates": copy.deepcopy(list(CANDIDATES)),
                }
            )
    result = {
        "schema": "modeling-v3-bank-plan-v1",
        "origin_identity": origin_identity,
        "partitions": partition,
        "banks": banks,
        "train_prompts": train_prompts,
        "generation_sequences": 36 * 4 * 8,
        "scratch_optimizer_updates": 108,
        "heldout_semantics_used_for_selection": False,
    }
    result["plan_hash"] = canonical_hash(result)
    return result


def _authorization(
    *, allow_gpu, acknowledge_new_experiment, allow_training=False, training=False, device="cuda:0"
):
    if allow_gpu is not True or acknowledge_new_experiment is not True:
        raise PermissionError(
            "V3 real execution requires --allow-gpu and "
            "--acknowledge-new-experiment after operator authorization"
        )
    if training and allow_training is not True:
        raise PermissionError("Source/scratch Adam execution requires --allow-training")
    # The inherited certified loader explicitly loads visible device zero. Two
    # workers receive independent Slurm one-GPU allocations, each called cuda:0.
    if device != "cuda:0":
        raise ValueError("One-GPU worker must use cuda:0 within Slurm CUDA_VISIBLE_DEVICES")


def _validate_qwen_contract(config):
    q = config["qwen"]
    expected = {
        "model_id": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "dtype": "bfloat16",
        "max_new_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "thinking": False,
        "steps": 128,
        "B": 4,
        "K": 8,
        "reward_epsilon": 1e-4,
        "Lnorm": 64,
        "ppo_clip": [0.8, 1.2],
        "KL_coef": 0.0,
        "source_arms": ["X_BASE", "X_VALID"],
    }
    for key, value in expected.items():
        if q.get(key) != value:
            raise ValueError(f"V3 production contract mismatch: qwen.{key}")
    lora = {
        "r": 8,
        "alpha": 16,
        "dropout": 0,
        "target_leaf_modules": ["gate_proj", "up_proj", "down_proj"],
        "expected_matrices": 96,
        "freeze_base": True,
        "freeze_vision": True,
    }
    optimizer = {
        "type": "AdamW",
        "lr": 1e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0,
        "grad_clip": 1.0,
        "step_explicit_zero_gradients": True,
    }
    if q.get("LoRA") != lora or q.get("optimizer") != optimizer:
        raise ValueError("V3 LoRA/Adam contract does not match the real inherited objective")


def audit_v3_compatibility(config, bindings):
    """CPU-only verification of original private plan and inherited contracts.

    New source/config hashes intentionally differ. The original certificate is
    validated against its own original config, and preserved byte-for-byte.
    A new V3 runtime smoke remains mandatory before source training.
    """
    _validate_qwen_contract(config)
    parent = _bound_json(bindings["parent_validated_plan"])
    inherited = parent["source_files"]
    current = source_hashes()
    changed = [name for name, digest in inherited.items() if current.get(name) != digest]
    if changed:
        raise ValueError("Inherited production files changed: " + ", ".join(changed))
    old = parent["config"]
    certificate = parent["gate"]["certificate"]
    from ..r1_reference_smoke import CERTIFICATE_CHECKS, PATH

    if (
        certificate.get("status") != "PASS"
        or certificate.get("selected_path") != PATH
        or any(
            certificate.get("checks", {}).get(k, {}).get("passed") is not True
            for k in CERTIFICATE_CHECKS
        )
    ):
        raise ValueError("Original real reference certificate is absent/incomplete")
    q = config["qwen"]
    comparisons = {
        "model": (old["model"]["id"], q["model_id"]),
        "revision": (old["model"]["revision"], q["revision"]),
        "epsilon": (old["reward"]["epsilon"], q["reward_epsilon"]),
        "Lnorm": (old["loss"]["Lnorm"], q["Lnorm"]),
        "KL": (old["loss"]["beta_KL"], q["KL_coef"]),
        "ppo_clip": ([old["loss"]["ppo_clip_lower"], old["loss"]["ppo_clip_upper"]], q["ppo_clip"]),
    }
    for key in ("temperature", "top_p", "top_k", "max_new_tokens"):
        comparisons[key] = (old["generation_proposed_N"][key], q[key])
    for new_key, old_key in (
        ("lr", "learning_rate"),
        ("betas", "betas"),
        ("eps", "eps"),
        ("weight_decay", "weight_decay"),
        ("grad_clip", "max_grad_norm"),
    ):
        comparisons["optimizer_" + new_key] = (old["optimizer"][old_key], q["optimizer"][new_key])
    if any(left != right for left, right in comparisons.values()):
        raise ValueError("V3 inherited mathematical/sampling contract is incompatible")
    data_root = Path(parent["paths"]["raw_dataset_root"])
    data_binding = parent["training_data_binding"]
    for relative, digest in data_binding["files"].items():
        candidate = (data_root / relative).resolve()
        if not candidate.is_relative_to(data_root.resolve()) or file_hash(candidate) != digest:
            raise ValueError("Inherited dataset file changed")
    if file_hash(data_root / "manifest.json") != data_binding["manifest_sha256"]:
        raise ValueError("Inherited dataset manifest changed")
    return {
        "status": "CPU_COMPATIBILITY_VERIFIED_GPU_SMOKE_REQUIRED",
        "parent_plan_sha256": bindings["parent_validated_plan"]["sha256"],
        "old_certificate_hash": canonical_hash(certificate),
        "inherited_source_files": inherited,
        "added_source_files": {k: v for k, v in current.items() if k not in inherited},
        "source_hash": canonical_hash(current),
        "config_hash": canonical_hash(config),
        "unchanged_contracts": {k: left for k, (left, _) in comparisons.items()},
        "new_contracts": {
            "steps": 128,
            "prompt_positions": 512,
            "lora_seed_namespace": NAMESPACE,
            "seed_roles": q["seed_roles"],
            "complete_state_schema": "ssvc-followup-complete-state-v1",
        },
        "historical_certificate_rewritten": False,
        "gpu_smoke_measured": False,
        "model_loaded": False,
    }


def _stage_gate(config, bindings, seed, *, operation):
    role = seed_role(config, seed)
    stage = _bound_json(bindings["v3_stage_lock"])
    if stage.get("config_hash") != canonical_hash(config) or stage.get(
        "source_hash"
    ) != canonical_hash(source_hashes()):
        raise ValueError("Frozen V3 stage configuration/source changed")
    _technical_gate(config, stage)
    if operation not in stage.get("operations", []) or seed not in stage.get("seeds", []):
        raise PermissionError("Seed/operation is outside the frozen authorized stage task list")
    if stage.get("phase") != "Q5" or stage.get("role") != role:
        raise PermissionError("Q5 execution stage must match the source seed role")
    current, seen = stage, set()
    while current.get("authorization_parent") is not None:
        binding = current["authorization_parent"]
        if canonical_hash(binding) in seen:
            raise ValueError("Cyclic execution authorization ancestry")
        seen.add(canonical_hash(binding))
        parent_stage = _bound_json(binding)
        if (
            parent_stage.get("role") != role
            or seed not in parent_stage.get("seeds", [])
            or parent_stage.get("source_hash") != stage["source_hash"]
            or parent_stage.get("config_hash") != stage["config_hash"]
        ):
            raise ValueError("Derived observation stage differs from its authorized parent")
        current = parent_stage
    smoke = _bound_json(stage["v3_gpu_smoke"])
    if (
        smoke.get("status") != "PASS"
        or smoke.get("config_hash") != canonical_hash(config)
        or smoke.get("source_hash") != canonical_hash(source_hashes())
        or smoke.get("execution_kind") != "REAL_CUDA_MODEL"
        or smoke.get("two_gpu_null_parity_passed") is not True
    ):
        raise ValueError("V3 real smoke and two-GPU numerical parity must pass first")
    if role != "development":
        from .schema import verify_selection_lock

        selection = _bound_json(stage["selection_lock"])
        verify_selection_lock(config, selection)
        if selection.get("locked_test_opened") is not False:
            raise ValueError("Methods/thresholds/at most two estimators and selectors must freeze")
        previous = "development" if role == "interval_calibration" else "interval_calibration"
        _verify_q5_completion(config, stage[previous + "_completion"], previous)
    return role


def _verify_runtime_stage(runtime):
    """Recheck authorization bytes at completion, independently of inference identity."""
    binding = runtime.get("execution_stage_binding")
    if binding is not None:
        stage = _bound_json(binding)
        if (
            stage.get("source_hash") != canonical_hash(source_hashes())
            or stage.get("config_hash") != runtime["identity"]["config_hash"]
        ):
            raise ValueError("Runtime source/config differs from its bound execution stage")
        while stage.get("authorization_parent") is not None:
            stage = _bound_json(stage["authorization_parent"])


def _technical_gate(config, stage):
    for phase in ("Q1", "Q2"):
        binding = stage.get("technical_receipts", {}).get(phase)
        if binding is None:
            raise ValueError("Q1/Q2 measured, hash-bound technical receipts are required")
        receipt = _bound_json(binding)
        if (
            receipt.get("status") != "PASS"
            or receipt.get("config_hash") != canonical_hash(config)
            or receipt.get("source_hash") != canonical_hash(source_hashes())
            or not receipt.get("checks")
            or not all(item.get("passed") is True for item in receipt["checks"])
        ):
            raise ValueError("Q1/Q2 technical receipt does not contain passing measured checks")


def initialize_lora(adapter, optimizer, seed):
    """New hash-derived seed; paired arms initialize identically, Adam stays empty."""
    import torch

    from ..optimizer_fork import parameter_hash
    from ..smoke_runtime import _seed_everything

    if optimizer.state:
        raise ValueError("LoRA initialization cannot reset a warm Adam trajectory")
    derived = lora_initialization_seed(seed)
    _seed_everything(derived)
    with torch.no_grad():
        for name, parameter in sorted(adapter.model.named_parameters()):
            if parameter.requires_grad:
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                elif "lora_B" in name:
                    parameter.zero_()
                else:
                    raise ValueError("Unexpected trainable parameter in V3 LoRA initialization")
    return {
        "seed": seed,
        "initialization_seed": derived,
        "namespace": NAMESPACE,
        "parameter_hash": parameter_hash(adapter.model, trainable=True),
    }


def planned_runtime_identity(config, bindings, *, parent=None):
    """Stable inference identity, independent of the authorization task whitelist.

    Binding a whitelist containing a manifest hash into that same manifest's
    runtime identity creates a cryptographic cycle. Stage authorization is
    checked independently and recorded in the execution receipt instead.
    """
    parent = parent or _bound_json(bindings["parent_validated_plan"])
    return {
        "protocol_version": config["protocol_version"],
        "execution_kind": "REAL_CUDA_MODEL",
        "model_hash": canonical_hash(parent["snapshot"]),
        "data_hash": canonical_hash(parent["training_data_binding"]),
        "config_hash": canonical_hash(config),
        "source_hash": canonical_hash(source_hashes()),
        "bindings_hash": canonical_hash(
            {
                "parent_plan_sha256": bindings["parent_validated_plan"]["sha256"],
                "probability_tolerances": bindings["v3_probability_tolerances"],
            }
        ),
        "environment": parent["r4_binding"]["environment"],
    }


def load_runtime(
    config,
    bindings,
    *,
    device="cuda:0",
    allow_gpu=False,
    acknowledge_new_experiment=False,
    seed=41001,
):
    _authorization(
        allow_gpu=allow_gpu, acknowledge_new_experiment=acknowledge_new_experiment, device=device
    )
    compatibility = audit_v3_compatibility(config, bindings)
    from .vlm_observation import _parity_values

    _parity_values([[0.0]], bindings["v3_probability_tolerances"], sequence_errors=[0.0])
    parent = _bound_json(bindings["parent_validated_plan"])
    import torch

    from ..followup_backend import _load_local_adapter
    from ..followup_updates import capture_state, forward_state_hash, load_checkpoint, restore_state
    from ..next_stage_runtime import validate_runtime_environment
    from ..r1_reference_smoke import _certificate_check
    from ..smoke_runtime import _seed_everything

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A Slurm-allocated one-visible-GPU worker is required")
    gpu = _allocated_gpu_info()
    if "PRO6000" not in gpu["name"].upper().replace(" ", ""):
        raise ValueError("The allocated GPU is not the requested PRO 6000")
    environment = validate_runtime_environment(parent["gate"])
    if environment != parent["r4_binding"]["environment"]:
        raise ValueError("Actual production dependencies changed since the inherited lock")
    _seed_everything(17)
    adapter = _load_local_adapter(parent["snapshot"], parent["config"]["model"])
    _certificate_check(parent["gate"]["certificate"], adapter, parent["config"])
    if len(adapter.audit["lora_modules"]) != 96:
        raise ValueError("V3 must have 96 language MLP LoRA target matrices")
    q = config["qwen"]["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=q["lr"],
        betas=tuple(q["betas"]),
        eps=q["eps"],
        weight_decay=q["weight_decay"],
    )
    initialization = initialize_lora(adapter, optimizer, seed)
    sampler = {"position": 0, "step": 0}
    identity = planned_runtime_identity(config, bindings, parent=parent)
    initial = capture_state(
        adapter.model,
        optimizer,
        {"checkpoint_step": 0, "initialization": initialization},
        adapter=adapter,
        sampler=sampler,
    )

    def policy_loader(spec):
        spec = spec.get("checkpoint", spec)
        state = load_checkpoint(
            spec["path"],
            spec["identity"],
            model=adapter.model,
            optimizer=optimizer,
            expected_file_sha256=spec["sha256"],
        )
        if (
            spec["identity"].get("source_hash") != identity["source_hash"]
            or spec["identity"].get("config_hash") != identity["config_hash"]
        ):
            raise ValueError("Observation checkpoint is outside the current V3 execution lock")
        restore_state(adapter.model, optimizer, state, adapter=adapter, sampler=sampler)
        adapter.model.eval()
        adapter._reset_positions()
        normalized = capture_state(
            adapter.model, optimizer, state["metadata"], adapter=adapter, sampler=sampler
        )
        return forward_state_hash(normalized)

    def state_guard():
        from ..optimizer_fork import parameter_hash, state_hash

        return state_hash(
            {
                "parameters": parameter_hash(adapter.model, trainable=True),
                "optimizer": optimizer.state_dict(),
                "buffers": {n: value for n, value in adapter.model.named_buffers()},
                "module_modes": {n: module.training for n, module in adapter.model.named_modules()},
                "frozen_versions": {
                    n: p._version
                    for n, p in adapter.model.named_parameters()
                    if not p.requires_grad
                },
            }
        )

    return {
        "adapter": adapter,
        "optimizer": optimizer,
        "sampler": sampler,
        "identity": identity,
        "initial_state": initial,
        "initialization": initialization,
        "compatibility": compatibility,
        "policy_loader": policy_loader,
        "data_root": parent["paths"]["raw_dataset_root"],
        "parent": parent,
        "allocated_gpu": gpu,
        "execution_stage_binding": copy.deepcopy(bindings.get("v3_stage_lock")),
        "parity_tolerances": bindings["v3_probability_tolerances"],
        "state_guard": state_guard,
        "certificate_thresholds": parent["gate"]["certificate"]["thresholds"],
    }


def _requests(prompts, identity, *, seed, arm, step, role, policy_hash, bank_id=None):
    requests = []
    for prompt in prompts:
        for index in range(8):
            rng = {
                "namespace": NAMESPACE,
                "seed": seed,
                "step": step,
                "role": role,
                "bank_id": bank_id,
                "policy_hash": policy_hash,
                "prompt_id": prompt["prompt_id"],
                "prompt_hash": prompt["prompt_hash"],
                "sample_index": index,
            }
            key = canonical_hash(rng)
            request = {
                **{
                    k: prompt[k]
                    for k in (
                        "prompt_id",
                        "prompt_hash",
                        "base_scene_id",
                        "family",
                        "interface",
                        "split",
                    )
                },
                "phase": "V3_SOURCE" if role == "source_train" else "V3_SCRATCH",
                "arm": arm,
                "train_seed": seed,
                "seed": seed,
                "checkpoint_step": step,
                "bank_role": role,
                "bank_id": bank_id,
                "group_id": prompt["prompt_id"],
                "sample_index": index,
                "rollout_index": index,
                "decode_mode": "sample",
                "do_sample": True,
                "enable_thinking": False,
                "max_new_tokens": 64,
                "policy_state_hash": policy_hash,
                "sample_rng_key": key,
                "sample_seed": int(key[:16], 16) % (2**31),
                "track": "N",
            }
            request["sample_key"] = canonical_hash({"identity": identity, "request": request})
            requests.append(request)
    return requests


def _groups(adapter, prompts, rows, data_root, *, thresholds):
    """Rebuild actual inputs and rescore same sampled tokens before any backward."""
    import numpy as np

    from ..followup_runtime import _prepared
    from ..optimizer_fork import state_hash

    if len(prompts) != 4 or len(rows) != 32:
        raise ValueError("Every V3 Adam update requires a natural B4 K8 bank")
    groups, errors = [], []
    for prompt in prompts:
        selected = [row for row in rows if row["prompt_id"] == prompt["prompt_id"]]
        if len(selected) != 8 or any(row["split"] != "train" for row in selected):
            raise ValueError("Training group is incomplete or contains nontraining prompts")
        prepared = _prepared(adapter, prompt, data_root)
        for row in selected:
            if state_hash(prepared) != row["prepared_hash"]:
                raise ValueError("Prepared input changed between sampling and training")
            scored = adapter.logprobs(prepared, row["token_ids"], require_grad=False)
            actual = scored.detach().cpu().double().numpy()
            old = np.asarray(row["old_logprobs"], dtype=float)
            if actual.shape != old.shape or not np.isfinite(actual).all():
                raise ValueError("Rescored action log probabilities have invalid shape/values")
            errors.extend(np.abs(actual - old).tolist())
        groups.append([{**row, "prepared": prepared} for row in selected])
    parity = {
        "mean_abs_token_logp": float(np.mean(errors)),
        "p99_abs_token_logp": float(np.quantile(errors, 0.99)),
        "tokens": len(errors),
    }
    if (
        parity["mean_abs_token_logp"] > thresholds["parity_alarm_mean_abs_token_logp"]
        or parity["p99_abs_token_logp"] > thresholds["parity_alarm_p99_abs_token_logp"]
    ):
        raise ValueError("Behavior/rescoring probability parity exceeds inherited thresholds")
    return groups, parity


def _restore_runtime(runtime, state):
    from ..followup_runtime import _restore

    _restore(runtime["adapter"], runtime["optimizer"], state, sampler=runtime["sampler"])


def _normalized_inference_fingerprint(runtime, state):
    """Record the actual eval/reset inference state separately from full recovery state."""
    from ..followup_updates import capture_state, forward_state_hash

    _restore_runtime(runtime, state)
    runtime["adapter"].model.eval()
    runtime["adapter"]._reset_positions()
    normalized = capture_state(
        runtime["adapter"].model,
        runtime["optimizer"],
        state["metadata"],
        adapter=runtime["adapter"],
        sampler=runtime["sampler"],
    )
    result = forward_state_hash(normalized)
    _restore_runtime(runtime, state)
    return result


def _complete(root, result):
    from ..r3_runtime import _manifest

    _publish(root / "result.json", result)
    manifest = _manifest(root, exclude=("manifest.json", "completed.json"))
    _publish(root / "manifest.json", manifest)
    _publish(
        root / "completed.json",
        {
            "status": "COMPLETED",
            "scientific_status": "NOT_CERTIFIED",
            "manifest_sha256": file_hash(root / "manifest.json"),
        },
    )


def _open_output(out, identity, resume):
    from ..followup_train import _safe_output
    from ..r3_runtime import _verify_manifest

    root = _safe_output(out)
    if root.exists() and any(p.name != ".writer.lock" for p in root.iterdir()) and not resume:
        raise FileExistsError("V3 output exists; explicit --resume required")
    root.mkdir(parents=True, exist_ok=True)
    _publish(root / "identity.json", identity)
    if (root / "completed.json").exists():
        marker = _read(root / "completed.json")
        if marker["manifest_sha256"] != file_hash(root / "manifest.json"):
            raise ValueError("V3 completion manifest changed")
        _verify_manifest(root, _read(root / "manifest.json"))
        return root, _read(root / "result.json")
    return root, None


def _meter(adapter):
    import torch

    return {
        "time": time.perf_counter(),
        "forward": adapter.forward_calls,
        "generation": adapter.generation_calls,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }


def _storage_preflight(runtime, root, checkpoint_count, *, extra_artifact_bytes=0):
    """Use actual trainable/buffer bytes to reject an impossible retained-state run."""
    model = runtime["adapter"].model
    parameters = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    buffers = sum(p.numel() * p.element_size() for p in model.buffers())
    # Adam first/second moments are charged even at the empty initial optimizer.
    per_checkpoint = 3 * parameters + buffers + 10 * 1024**2
    if type(extra_artifact_bytes) is not int or extra_artifact_bytes < 0:
        raise ValueError("Extra retained artifact bytes must be nonnegative integers")
    required = math.ceil(1.2 * (checkpoint_count * per_checkpoint + extra_artifact_bytes))
    free = shutil.disk_usage(root).free
    available_ram = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_ram = int(line.split()[1]) * 1024
    if free < required or (available_ram is not None and available_ram < 8 * parameters + buffers):
        raise RuntimeError("Actual available RAM/disk cannot hold the retained full checkpoints")
    receipt = {
        "actual_trainable_bytes": parameters,
        "actual_buffer_bytes": buffers,
        "checkpoint_count": checkpoint_count,
        "extra_retained_artifact_bytes": extra_artifact_bytes,
        "estimated_required_disk_bytes": required,
        "actual_free_disk_bytes": free,
        "actual_available_ram_bytes": available_ram,
        "estimate_includes_two_Adam_moment_tensors": True,
        "actual_checkpoint_bytes_measured_in_step_records": True,
    }
    if not (root / "storage_preflight.json").exists():
        _publish(root / "storage_preflight.json", receipt)
    return receipt


def run_source_trajectory(
    config, runtime, schedule, *, seed, arm, out, resume=False, fixture=False
):
    """Execute real objective and complete checkpoints; small CPU fixtures share this path."""
    from ..followup_runtime import collect_samples
    from ..followup_updates import (
        apply_gradient_update,
        capture_state,
        direct_loss_gradients,
        forward_state_hash,
        load_checkpoint,
        save_checkpoint,
    )
    from ..optimizer_fork import parameter_hash, state_hash
    from ..r3_runtime import _optimizer_step, run_atomic_unit

    adapter, optimizer, sampler = (runtime[k] for k in ("adapter", "optimizer", "sampler"))
    if fixture:
        if getattr(adapter, "execution_kind", None) != "CPU_FAKE_TORCH" or any(
            p.device.type != "cpu" for p in adapter.model.parameters()
        ):
            raise ValueError("CPU fixture requires an explicitly fake CPU adapter")
    elif runtime["identity"].get("execution_kind") != "REAL_CUDA_MODEL" or any(
        not p.is_cuda for p in adapter.model.parameters()
    ):
        raise ValueError("Production source requires actual CUDA tensors")
    if arm not in SOURCE_ARMS:
        raise ValueError("Unknown V3 source arm")
    if not fixture and (len(schedule["train_steps"]) != 128 or schedule["positions"] != 512):
        raise ValueError("V3 production trajectory must contain 128 B4 steps")
    identity = {
        **runtime["identity"],
        "unit": "V3_SOURCE",
        "seed": seed,
        "arm": arm,
        "schedule_hash": schedule["schedule_hash"],
        "fixture": fixture,
        "execution_stage_binding": runtime.get("execution_stage_binding"),
    }
    from ..followup_train import _safe_output

    root = _safe_output(out)
    with frozen_writer(root):
        root, completed = _open_output(out, identity, resume)
        if completed is not None:
            return completed
        _storage_preflight(runtime, root, 1 + len(schedule["train_steps"]))
        _publish(
            root / "schedule.json", {k: v for k, v in schedule.items() if k != "train_prompts"}
        )
        initial = runtime["initial_state"]
        initial_id = {**identity, "unit": "V3_SOURCE_CHECKPOINT", "step": 0}
        if (root / "initial.json").exists():
            binding = _read(root / "initial.json")
            state = load_checkpoint(
                root / "initial.pt", initial_id, expected_file_sha256=binding["sha256"]
            )
        else:
            state = copy.deepcopy(initial)
            state["metadata"].update(seed=seed, arm=arm, checkpoint_step=0)
            binding = save_checkpoint(root / "initial.pt", state, initial_id)
            _publish(root / "initial.json", {**binding, "identity": initial_id})
        _restore_runtime(runtime, state)
        prompts = {p["prompt_id"]: p for p in schedule["train_prompts"]}
        results = []
        for step, prompt_ids in enumerate(schedule["train_steps"], 1):
            bank = [prompts[p] for p in prompt_ids]
            step_root = root / "steps" / f"step_{step:03d}"
            requests = _requests(
                bank,
                identity,
                seed=seed,
                arm=arm,
                step=step - 1,
                role="source_train",
                policy_hash=forward_state_hash(state),
            )
            rows = collect_samples(
                adapter,
                optimizer,
                state,
                bank,
                requests,
                step_root / "rollouts",
                identity,
                runtime["data_root"],
                sampler=sampler,
            )
            unit = {
                **identity,
                "step": step,
                "origin_state_hash": state_hash(state),
                "requests_hash": canonical_hash(requests),
                "raw_hash": canonical_hash([r["record_hash"] for r in rows]),
            }
            prestate = state

            def operation(attempt, prestate=prestate, bank=bank, rows=rows, step=step, unit=unit):
                _restore_runtime(runtime, prestate)
                started = _meter(adapter)
                frozen = parameter_hash(adapter.model, trainable=False)
                frozen_versions = {
                    n: p._version
                    for n, p in adapter.model.named_parameters()
                    if not p.requires_grad
                }
                try:
                    groups, parity = _groups(
                        adapter,
                        bank,
                        rows,
                        runtime["data_root"],
                        thresholds=runtime["certificate_thresholds"],
                    )
                    policy, weight = SOURCE_ARMS[arm]
                    gradients = direct_loss_gradients(
                        adapter, groups, policy=policy, auxiliary_weight=weight
                    )
                    # Zero advantages produce explicit zero gradients. Adam still
                    # advances, including its existing momentum and step counter.
                    _publish(attempt / "optimizer_call_intent.json", {"possible_calls": 1})
                    update = apply_gradient_update(adapter.model, optimizer, gradients["gradients"])
                    _publish(attempt / "optimizer_call_completed.json", {"observed_calls": 1})
                    sampler.update(
                        position=step * 4, step=step, schedule_hash=schedule["schedule_hash"]
                    )
                    post = capture_state(
                        adapter.model,
                        optimizer,
                        {**prestate["metadata"], "checkpoint_step": step},
                        adapter=adapter,
                        sampler=sampler,
                    )
                    if _optimizer_step(post) != step or _optimizer_step(prestate) != step - 1:
                        raise ValueError(
                            "V3 source Adam counter does not match the actual source cursor"
                        )
                    if parameter_hash(
                        adapter.model, trainable=False
                    ) != frozen or frozen_versions != {
                        n: p._version
                        for n, p in adapter.model.named_parameters()
                        if not p.requires_grad
                    }:
                        raise RuntimeError("Frozen base/vision was modified by a source step")
                    checkpoint_id = {**unit, "unit": "V3_SOURCE_CHECKPOINT"}
                    saved = save_checkpoint(attempt / "checkpoint.pt", post, checkpoint_id)
                    # Retain every actual intermediate state. This is explicitly
                    # included in the storage estimate and supports dense Q6 replay.
                    norms = [
                        float(
                            (post["parameters"][n].double() - prestate["parameters"][n].double())
                            .square()
                            .sum()
                        )
                        for n in post["parameters"]
                    ]
                    result = {
                        "step": step,
                        "checkpoint_identity": checkpoint_id,
                        "checkpoint_sha256": saved["sha256"],
                        "state_hash": saved["state_hash"],
                        "parameter_hash": state_hash(post["parameters"]),
                        "inference_fingerprint": _normalized_inference_fingerprint(runtime, post),
                        "optimizer_hash": state_hash(post["optimizer"]),
                        "rng_hash": state_hash(post["rng"]),
                        "sampler_hash": state_hash(post["sampler"]),
                        "update_norm": math.sqrt(math.fsum(norms)),
                        "parity": parity,
                        "gradient_audit": gradients["audit"],
                        "update": update,
                        "forward_calls": adapter.forward_calls - started["forward"],
                        "backward_calls": len(rows),
                        "optimizer_calls": 1,
                        "generation_sequences": len(rows),
                        "generated_tokens": sum(len(r["token_ids"]) for r in rows),
                        "gradient_and_parity_scored_tokens": 2
                        * sum(len(r["token_ids"]) for r in rows),
                        "generation_forward_calls": sum(r["runtime_forward_calls"] for r in rows),
                        "generation_seconds": sum(r["elapsed_seconds"] for r in rows),
                        "update_seconds": time.perf_counter() - started["time"],
                        "checkpoint_bytes": (attempt / "checkpoint.pt").stat().st_size,
                        "peak_cuda_bytes": _meter(adapter)["peak_cuda_bytes"],
                    }
                    return result
                finally:
                    _restore_runtime(runtime, prestate)

            attempt, result, reused = run_atomic_unit(step_root / "update", unit, operation)
            state = load_checkpoint(
                attempt / "checkpoint.pt",
                result["checkpoint_identity"],
                model=adapter.model,
                optimizer=optimizer,
                expected_file_sha256=result["checkpoint_sha256"],
            )
            _restore_runtime(runtime, state)
            results.append(
                {**result, "path": str(attempt / "checkpoint.pt"), "reused_completed_step": reused}
            )
        anchors = (
            [32, 96]
            if arm == "X_BASE"
            else ([96] if seed in config["qwen"]["seed_roles"]["locked_test"] else [])
        )
        response_plans = []
        for entry in results:
            if entry["step"] not in anchors:
                continue
            bank_plan = build_bank_plan(
                schedule["train_prompts"],
                origin_identity={
                    "seed": seed,
                    "arm": arm,
                    "step": entry["step"],
                    "state_hash": entry["state_hash"],
                    "source_hash": identity["source_hash"],
                },
            )
            path = root / "response_plans" / f"step_{entry['step']:03d}.json"
            _publish(path, bank_plan)
            response_plans.append(
                {
                    "step": entry["step"],
                    "bank_plan": str(path),
                    "bank_plan_sha256": file_hash(path),
                    "checkpoint": entry["path"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                }
            )
        result = {
            "status": "ENGINEERING_EXECUTION_COMPLETED",
            "scientific_status": "NOT_CERTIFIED",
            "identity": identity,
            "steps": len(results),
            "source_optimizer_steps": len(results),
            "training_outputs": 32 * len(results),
            "checkpoints": results,
            "retained_all_intermediate_states": True,
            "online_feedback": False,
            "initial_parameter_hash": state_hash(initial["parameters"]),
            "response_plans": response_plans,
        }
        _verify_runtime_stage(runtime)
        _complete(root, result)
        return result


def train_source(
    config,
    bindings,
    *,
    seed,
    arm,
    device,
    out,
    allow_gpu=False,
    allow_training=False,
    acknowledge_new_experiment=False,
    resume=False,
):
    _authorization(
        allow_gpu=allow_gpu,
        allow_training=allow_training,
        training=True,
        acknowledge_new_experiment=acknowledge_new_experiment,
        device=device,
    )
    if arm not in SOURCE_ARMS:
        raise ValueError("Unknown V3 source arm")
    _stage_gate(config, bindings, seed, operation="train-source")
    runtime = load_runtime(
        config,
        bindings,
        seed=seed,
        device=device,
        allow_gpu=allow_gpu,
        acknowledge_new_experiment=acknowledge_new_experiment,
    )
    parent = runtime["parent"]
    train = parent["training_plan"]["train_prompts"]
    schedule = build_training_schedule(
        [p["scene"] for p in train], seed=seed, data_root=runtime["data_root"]
    )
    return run_source_trajectory(
        config, runtime, schedule, seed=seed, arm=arm, out=out, resume=resume
    )


def run_candidate_banks(
    runtime, origin, bank_plan, *, out, resume=False, fixture=False, bridge=False
):
    """Natural origin banks, same-full-origin Adam forks, exact realized e files."""
    from ..followup_runtime import collect_samples
    from ..followup_updates import (
        capture_state,
        fork_one_candidate,
        forward_state_hash,
        load_checkpoint,
        save_checkpoint,
    )
    from ..optimizer_fork import state_hash
    from ..r3_runtime import _save_tensor_payload, run_atomic_unit

    adapter, optimizer, sampler = (runtime[k] for k in ("adapter", "optimizer", "sampler"))
    if fixture:
        if getattr(adapter, "execution_kind", None) != "CPU_FAKE_TORCH":
            raise ValueError("Fixture execution requires a fake CPU adapter")
    elif runtime["identity"].get("execution_kind") != "REAL_CUDA_MODEL" or any(
        not p.is_cuda for p in adapter.model.parameters()
    ):
        raise ValueError("Production forks require actual CUDA model tensors")
    plan_without_hash = {k: v for k, v in bank_plan.items() if k != "plan_hash"}
    if canonical_hash(plan_without_hash) != bank_plan["plan_hash"]:
        raise ValueError("Frozen bank plan was changed")
    if bank_plan["origin_identity"].get("state_hash") != state_hash(origin):
        raise ValueError("Bank plan belongs to a different complete origin")
    expected_banks = {"calibration_pool": 6} if bridge else {"calibration_pool": 24, "heldout": 12}
    if not fixture and Counter(b["role"] for b in bank_plan["banks"]) != expected_banks:
        raise ValueError("Production fork plan differs from the frozen Q4/Q5 bank counts")
    identity = {
        **runtime["identity"],
        "unit": "V3_FORKS",
        "plan_hash": bank_plan["plan_hash"],
        "origin_state_hash": state_hash(origin),
        "fixture": fixture,
        "Q4_bridge": bridge,
        "execution_stage_binding": runtime.get("execution_stage_binding"),
    }
    from ..followup_train import _safe_output

    root = _safe_output(out)
    with frozen_writer(root):
        root, completed = _open_output(out, identity, resume)
        if completed is not None:
            return completed
        fp64_vectors = (
            4
            * len(bank_plan["banks"])
            * 8
            * sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
        )
        _storage_preflight(
            runtime, root, 3 * len(bank_plan["banks"]), extra_artifact_bytes=fp64_vectors
        )
        _publish(root / "bank_plan.json", bank_plan)
        prompts = {p["prompt_id"]: p for p in bank_plan["train_prompts"]}
        all_ids = [p for b in bank_plan["banks"] for p in b["prompt_ids"]]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("An origin bank pool must not repeat prompts")
        _restore_runtime(runtime, origin)
        results = []
        try:
            for bank in bank_plan["banks"]:
                if bank["candidates"] != list(CANDIDATES):
                    raise ValueError("All three predeclared candidate interventions are required")
                bank_root = root / "banks" / bank["bank_id"]
                panel = [prompts[p] for p in bank["prompt_ids"]]
                requests = _requests(
                    panel,
                    identity,
                    seed=origin["metadata"]["seed"],
                    arm=origin["metadata"]["arm"],
                    step=origin["metadata"]["checkpoint_step"],
                    role="scratch_bank",
                    policy_hash=forward_state_hash(origin),
                    bank_id=bank["bank_id"],
                )
                rows = collect_samples(
                    adapter,
                    optimizer,
                    origin,
                    panel,
                    requests,
                    bank_root / "rollouts",
                    identity,
                    runtime["data_root"],
                    sampler=sampler,
                )
                groups, parity = _groups(
                    adapter,
                    panel,
                    rows,
                    runtime["data_root"],
                    thresholds=runtime["certificate_thresholds"],
                )
                _publish(bank_root / "parity.json", parity)
                # Preparing/scoring can create position state. Restore before
                # every fork; groups retain their immutable prepared input bytes.
                _restore_runtime(runtime, origin)
                states, checkpoints = {}, {}
                for spec in CANDIDATES:
                    unit = {
                        **identity,
                        "bank_id": bank["bank_id"],
                        "candidate": spec,
                        "raw_hash": canonical_hash([r["record_hash"] for r in rows]),
                    }

                    def operation(attempt, groups=groups, spec=spec, unit=unit, rows=rows):
                        start = _meter(adapter)
                        _publish(attempt / "optimizer_call_intent.json", {"possible_calls": 1})
                        fork = fork_one_candidate(
                            adapter,
                            optimizer,
                            origin,
                            groups,
                            spec,
                            sampler=sampler,
                            failure_path=attempt / "failure_restore.json",
                        )
                        _publish(attempt / "optimizer_call_completed.json", {"observed_calls": 1})
                        checkpoint_id = {**unit, "unit": "V3_CANDIDATE_CHECKPOINT"}
                        saved = save_checkpoint(
                            attempt / "checkpoint.pt", fork["state"], checkpoint_id
                        )
                        try:
                            inference_fingerprint = _normalized_inference_fingerprint(
                                runtime, fork["state"]
                            )
                        finally:
                            _restore_runtime(runtime, origin)
                        return {
                            "checkpoint_identity": checkpoint_id,
                            "checkpoint_sha256": saved["sha256"],
                            "state_hash": saved["state_hash"],
                            "inference_fingerprint": inference_fingerprint,
                            "audit": fork["audit"],
                            "update_seconds": time.perf_counter() - start["time"],
                            "forward_calls": adapter.forward_calls - start["forward"],
                            "optimizer_calls": 1,
                            "backward_calls": 32,
                            "scored_tokens": sum(len(r["token_ids"]) for r in rows),
                            "checkpoint_bytes": (attempt / "checkpoint.pt").stat().st_size,
                            "peak_cuda_bytes": _meter(adapter)["peak_cuda_bytes"],
                        }

                    attempt, record, _ = run_atomic_unit(bank_root / spec["id"], unit, operation)
                    state = load_checkpoint(
                        attempt / "checkpoint.pt",
                        record["checkpoint_identity"],
                        expected_file_sha256=record["checkpoint_sha256"],
                    )
                    states[spec["id"]] = state
                    checkpoints[spec["id"]] = {
                        **record,
                        "path": str(attempt / "checkpoint.pt"),
                        "identity": record["checkpoint_identity"],
                        "sha256": record["checkpoint_sha256"],
                    }
                vectors = {}
                states["source_origin"] = origin
                for name, left, right in (
                    *CONTRASTS,
                    ("joint_0_minus_origin", "joint_0", "source_origin"),
                ):
                    # FP64 subtraction of the actual saved Adam parameters; no
                    # optimizer-gradient substitution or inferred direction.
                    e = {
                        n: states[left]["parameters"][n].double()
                        - states[right]["parameters"][n].double()
                        for n in sorted(states[left]["parameters"])
                    }
                    path = bank_root / (name + ".pt")
                    if path.exists():
                        from ..r3_runtime import _load_tensor_payload

                        binding = _read(bank_root / (name + ".json"))
                        actual = _load_tensor_payload(path, identity, binding)
                        if state_hash(actual) != state_hash(e):
                            raise ValueError("Retained actual candidate e changed")
                    else:
                        binding = _save_tensor_payload(path, e, identity)
                        _publish(bank_root / (name + ".json"), binding)
                    norm = math.sqrt(math.fsum(float(value.square().sum()) for value in e.values()))
                    vectors[name] = {
                        **binding,
                        "path": str(path),
                        "parameter_order": list(e),
                        "dimension": sum(value.numel() for value in e.values()),
                        "norm": norm,
                        "classification": "ZERO_VECTOR_UNKNOWN"
                        if norm == 0
                        else "NONZERO_MEASURED",
                        "left": left,
                        "right": right,
                    }
                fingerprints = {
                    key: checkpoints[key]["inference_fingerprint"] for key in checkpoints
                }
                aliases = {
                    key: next(
                        (
                            other
                            for other in fingerprints
                            if other != key and fingerprints[other] == fingerprints[key]
                        ),
                        None,
                    )
                    for key in fingerprints
                }
                results.append(
                    {
                        "bank_id": bank["bank_id"],
                        "role": bank["role"],
                        "checkpoints": checkpoints,
                        "contrasts": vectors,
                        "inference_aliases": aliases,
                        "training_resume_alias_allowed": False,
                        "generation_sequences": len(rows),
                        "generated_tokens": sum(len(r["token_ids"]) for r in rows),
                    }
                )
            restored = capture_state(
                adapter.model, optimizer, origin["metadata"], adapter=adapter, sampler=sampler
            )
            if state_hash(restored) != state_hash(origin):
                raise RuntimeError("Candidate banks altered the complete source origin")
        finally:
            _restore_runtime(runtime, origin)
        result = {
            "status": "ENGINEERING_EXECUTION_COMPLETED",
            "scientific_status": "NOT_CERTIFIED",
            "identity": identity,
            "banks": results,
            "scratch_optimizer_steps": 3 * len(results),
            "origin_restored": True,
            "source_training_committed": False,
        }
        _verify_runtime_stage(runtime)
        _complete(root, result)
        return result


def make_forks(
    config,
    bindings,
    *,
    checkpoint,
    bank_plan,
    device,
    out,
    allow_gpu=False,
    allow_training=False,
    acknowledge_new_experiment=False,
    resume=False,
):
    _authorization(
        allow_gpu=allow_gpu,
        allow_training=allow_training,
        training=True,
        acknowledge_new_experiment=acknowledge_new_experiment,
        device=device,
    )
    spec = resolve_checkpoint_binding(checkpoint)
    plan = _read(bank_plan) if isinstance(bank_plan, (str, Path)) else bank_plan
    seed = plan["origin_identity"]["seed"]
    arm, step = plan["origin_identity"]["arm"], plan["origin_identity"]["step"]
    allowed = {("X_BASE", 32), ("X_BASE", 96)}
    if seed_role(config, seed) == "locked_test":
        allowed.add(("X_VALID", 96))
    if (arm, step) not in allowed:
        raise ValueError("Candidate origin is outside the frozen Q5 response anchor matrix")
    if spec["identity"].get("source_hash") != canonical_hash(source_hashes()) or spec[
        "identity"
    ].get("config_hash") != canonical_hash(config):
        raise ValueError("Candidate source checkpoint is outside the current V3 execution lock")
    _stage_gate(config, bindings, seed, operation="make-forks")
    runtime = load_runtime(
        config,
        bindings,
        seed=seed,
        device=device,
        allow_gpu=allow_gpu,
        acknowledge_new_experiment=acknowledge_new_experiment,
    )
    from ..followup_updates import load_checkpoint

    origin = load_checkpoint(
        spec["path"],
        spec["identity"],
        expected_file_sha256=spec["sha256"],
        model=runtime["adapter"].model,
        optimizer=runtime["optimizer"],
    )
    if any(
        origin["metadata"].get(key) != value
        for key, value in (("seed", seed), ("arm", arm), ("checkpoint_step", step))
    ):
        raise ValueError("Candidate plan seed/arm/step differs from actual source metadata")
    return run_candidate_banks(runtime, origin, plan, out=out, resume=resume)


def resolve_checkpoint_binding(checkpoint):
    """Accept the documented .pt CLI argument via its immutable sibling receipt."""
    if isinstance(checkpoint, dict):
        spec = copy.deepcopy(checkpoint)
    else:
        path = Path(checkpoint)
        if path.suffix != ".pt":
            spec = _read(path)
        else:
            from ..r3_runtime import _verify_manifest

            receipt_path = path.parent / (
                "initial.json" if path.name == "initial.pt" else "result.json"
            )
            receipt = _read(receipt_path)
            if path.name == "initial.pt":
                spec = {
                    "path": str(path),
                    "identity": receipt["identity"],
                    "sha256": receipt["sha256"],
                }
            else:
                marker = _read(path.parent / "completed.json")
                if marker["manifest_sha256"] != file_hash(path.parent / "manifest.json"):
                    raise ValueError("Checkpoint owning attempt manifest changed")
                _verify_manifest(path.parent, _read(path.parent / "manifest.json"))
                spec = {
                    "path": str(path),
                    "identity": receipt["checkpoint_identity"],
                    "sha256": receipt["checkpoint_sha256"],
                }
    if not {"path", "identity", "sha256"} <= set(spec) or file_hash(spec["path"]) != spec["sha256"]:
        raise ValueError("Checkpoint file binding missing or changed")
    return spec


def estimate_workload(config, measurement=None):
    q = config["qwen"]
    seeds = sum(len(values) for values in q["seed_roles"].values())
    counts = {
        "source_trajectories": seeds * 2,
        "source_optimizer_steps": seeds * 2 * 128,
        "source_training_outputs": seeds * 2 * 128 * 32,
        "response_origins": seeds * 2 + len(q["seed_roles"]["locked_test"]),
        "full_checkpoints_per_source": 129,
        "full_checkpoints_per_response_origin": 108,
        "preserve_all_intermediate_source_states": True,
    }
    origins = counts["response_origins"]
    counts.update(
        scratch_optimizer_steps=origins * 108,
        train_bank_outputs=origins * 36 * 32,
        main_origin_draws=origins * 36 * 1024,
        main_candidate_score_sequences_before_alias=origins * 108 * 36 * 1024,
        reference_origin_draws_initial=origins * 36 * 4096,
        reference_candidate_score_sequences_initial=origins * 36 * 36 * 4096,
        reference_mix_spotcheck_draws_initial=origins * 2 * 36 * 4096,
    )
    result = {
        "counts": counts,
        "estimate_status": "UNMEASURED",
        "estimated_gpu_seconds": None,
        "two_worker_ideal_seconds": None,
        "calibrated_95_coverage_claim": False,
        "excludes": [
            "Q4",
            "pilot upgrades",
            "independent endpoint direct-count studies",
            "reference extensions",
            "extra mandatory MIX checks",
            "failed/interrupted attempts",
            "I/O and queue delays",
        ],
    }
    if measurement is not None:
        if (
            measurement.get("execution_kind") != "REAL_CUDA_MODEL"
            or measurement.get("measured") is not True
        ):
            raise ValueError("Runtime estimates require actual real-GPU measurements")
        generation = float(measurement["seconds_per_generated_sequence"])
        scoring = float(measurement["seconds_per_scored_sequence"])
        update = float(measurement["seconds_per_adam_update_excluding_generation"])
        if not all(math.isfinite(x) and x > 0 for x in (generation, scoring, update)):
            raise ValueError("Measured positive finite timings required")
        generations = (
            counts["source_training_outputs"]
            + counts["train_bank_outputs"]
            + counts["main_origin_draws"]
            + counts["reference_origin_draws_initial"]
            + counts["reference_mix_spotcheck_draws_initial"]
        )
        scores = (
            counts["main_candidate_score_sequences_before_alias"]
            + counts["reference_candidate_score_sequences_initial"]
            + 2 * counts["reference_mix_spotcheck_draws_initial"]
        )
        seconds = (
            generations * generation
            + scores * scoring
            + (counts["source_optimizer_steps"] + counts["scratch_optimizer_steps"]) * update
        )
        result.update(
            estimate_status="MEASURED_THROUGHPUT_EXTRAPOLATION",
            measurement=measurement,
            estimated_gpu_seconds=seconds,
            two_worker_ideal_seconds=seconds / 2,
            frozen_phase_dependencies_reduce_parallel_efficiency=True,
        )
        if "checkpoint_bytes" in measurement:
            result["checkpoint_storage_bytes"] = int(measurement["checkpoint_bytes"]) * (
                129 * counts["source_trajectories"] + 108 * counts["response_origins"]
            )
    return result


def plan_gpu(config, out=None, *, measurement=None):
    """Produce staged immutable task lists without importing Torch or a model."""
    _validate_qwen_contract(config)
    tasks = []
    for worker in range(2):
        tasks.append(
            {
                "id": f"Q4_bridge_worker_{worker}",
                "stage": "Q4",
                "operation": "vlm-smoke",
                "worker": worker,
                "depends_on": ["Q1_Q2_technical_acceptance"],
                "model_copies": 1,
                "status": "NOT_RUN",
            }
        )
    for role, seeds in config["qwen"]["seed_roles"].items():
        stage = "Q5_" + role
        dependency = {
            "development": "Q4_two_gpu_bridge_completion",
            "interval_calibration": "Q5_development_selection_freeze",
            "locked_test": "Q5_interval_calibration_completion",
        }[role]
        for seed in seeds:
            for arm_index, arm in enumerate(SOURCE_ARMS):
                tid = f"source_{seed}_{arm}"
                tasks.append(
                    {
                        "id": tid,
                        "stage": stage,
                        "seed_role": role,
                        "seed": seed,
                        "arm": arm,
                        "operation": "train-source",
                        "steps": 128,
                        "worker": (seeds.index(seed) * 2 + arm_index) % 2,
                        "depends_on": [dependency],
                        "status": "NOT_RUN",
                    }
                )
                anchors = [32, 96] if arm == "X_BASE" else ([96] if role == "locked_test" else [])
                for anchor in anchors:
                    origin = f"{seed}_{arm}_{anchor}"
                    tasks.append(
                        {
                            "id": "forks_" + origin,
                            "stage": stage,
                            "seed_role": role,
                            "seed": seed,
                            "arm": arm,
                            "anchor": anchor,
                            "operation": "make-forks",
                            "banks": 36,
                            "candidates": 108,
                            "worker": int(canonical_hash(origin)[:8], 16) % 2,
                            "depends_on": [tid],
                            "status": "NOT_RUN",
                        }
                    )
                    for worker in range(2):
                        tasks.append(
                            {
                                "id": f"observe_{origin}_worker{worker}",
                                "stage": stage,
                                "operation": "observe-vlm",
                                "origin": origin,
                                "worker": worker,
                                "depends_on": [
                                    "forks_" + origin,
                                    "frozen_observation_request_manifest",
                                ],
                                "status": "NOT_RUN",
                            }
                        )
    for row in tasks:
        row["task_hash"] = canonical_hash(row)
    result = {
        "schema": "modeling-v3-gpu-plan-v1",
        "config_hash": canonical_hash(config),
        "source_hash": canonical_hash(source_hashes()),
        "tasks": tasks,
        "tasks_hash": canonical_hash(tasks),
        "workers": 2,
        "max_concurrent_project_gpus": 2,
        "allocation": "one Slurm GPU per independent process",
        "workload": estimate_workload(config, measurement),
        "model_loaded": False,
        "submitted": False,
        "first_gpu_submission_requires_operator_authorization": True,
        "Q6_status": "BLOCKED_PENDING_POINTWISE_QUALIFICATION",
    }
    if out is not None:
        root = Path(out)
        with frozen_writer(root):
            _publish(root / "plan_gpu.json", result)
            _publish(
                root / "two_worker_manifest.json",
                {
                    "tasks_hash": result["tasks_hash"],
                    "workers": {
                        str(i): [t["id"] for t in tasks if t["worker"] == i] for i in range(2)
                    },
                },
            )
            lines = "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
            target = root / "tasks.jsonl"
            if target.exists() and target.read_text() != lines:
                raise ValueError("Frozen task list differs")
            if not target.exists():
                temporary = root / "tasks.jsonl.partial"
                temporary.write_text(lines)
                os.replace(temporary, target)
    return result


def build_probe_panel(control_scenes, *, excluded_base_scene_ids, data_root=None):
    """18 control base scenes x two interfaces, excluding all opened S1 panel bases."""
    from ..r4_inputs import DEV_CELLS, INTERFACES, _n_record, _validate_n

    scenes = _validate_n(control_scenes, "control")
    excluded = set(excluded_base_scene_ids)
    groups = defaultdict(list)
    for scene in scenes:
        if scene["base_scene_id"] not in excluded:
            groups[(scene["constraint_family"], scene["chart_type"], scene["operation"])].append(
                scene
            )
    missing = [cell for cell in DEV_CELLS if not groups[cell]]
    if missing:
        raise ValueError(
            "CONTROL_POOL_INSUFFICIENT: new namespace data required; sealed confirm "
            "remains closed; missing cells=" + repr(missing)
        )
    selected = [
        min(
            groups[cell],
            key=lambda s: canonical_hash([NAMESPACE, "probe_panel", s["base_scene_id"]]),
        )
        for cell in DEV_CELLS
    ]
    return [
        _n_record(scene, interface, data_root) for scene in selected for interface in INTERFACES
    ]


def _file_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def _verified_execution(root, config, *, unit):
    """Verify complete originals on CPU without deserializing model tensors."""
    from ..r3_runtime import _verify_manifest

    root = Path(root).resolve()
    marker = _read(root / "completed.json")
    if marker.get("status") != "COMPLETED" or marker.get("manifest_sha256") != file_hash(
        root / "manifest.json"
    ):
        raise ValueError("Execution is incomplete or its manifest changed")
    _verify_manifest(root, _read(root / "manifest.json"))
    result = _read(root / "result.json")
    identity = result["identity"]
    if (
        identity.get("unit") != unit
        or identity.get("fixture") is not False
        or identity.get("execution_kind") != "REAL_CUDA_MODEL"
        or identity.get("config_hash") != canonical_hash(config)
        or identity.get("source_hash") != canonical_hash(source_hashes())
    ):
        raise ValueError("Execution must be real, complete, and bound to this source/config")
    return result


def _q5_expected_tasks(config, role):
    tasks = {}
    for seed in config["qwen"]["seed_roles"][role]:
        for arm in SOURCE_ARMS:
            base = {"seed": seed, "arm": arm, "role": role}
            tasks[f"source_{seed}_{arm}"] = {**base, "kind": "source"}
            anchors = [32, 96] if arm == "X_BASE" else [96] if role == "locked_test" else []
            for step in anchors:
                origin_id = f"{seed}_{arm}_{step}"
                origin = {**base, "origin_id": origin_id, "step": step}
                for kind in ("forks", "prediction", "evaluation"):
                    tasks[f"{kind}_{origin_id}"] = {**origin, "kind": kind}
                for purpose in ("measurement", "reference"):
                    for operation in ("generate", "score"):
                        for worker in (0, 1):
                            tasks[f"{purpose}_{operation}_{origin_id}_worker{worker}"] = {
                                **origin,
                                "kind": "observation",
                                "purpose": purpose,
                                "operation": operation,
                                "worker_index": worker,
                            }
    return tasks


def _authorization_descends_from(binding, ancestor):
    seen = set()
    while binding is not None:
        key = canonical_hash(binding)
        if key in seen:
            raise ValueError("Cyclic execution authorization ancestry")
        seen.add(key)
        stage = _bound_json(binding)
        if binding == ancestor:
            return True
        binding = stage.get("authorization_parent")
    return False


def _verify_q5_task(config, stage_binding, expected, binding):
    """Verify task-owned original evidence, rather than accepting a caller's PASS."""
    from .io import verify_manifest

    receipt = _bound_json(binding)
    path = Path(binding["path"]).resolve()
    kind = expected["kind"]
    completion_markers = []
    common = {"config_hash": canonical_hash(config), "source_hash": canonical_hash(source_hashes())}
    if kind in {"source", "forks"}:
        receipt = _verified_execution(
            path.parent, config, unit="V3_SOURCE" if kind == "source" else "V3_FORKS"
        )
        identity = receipt["identity"]
        completion_markers = [
            _file_binding(path.parent / name) for name in ("manifest.json", "completed.json")
        ]
        if not _authorization_descends_from(identity.get("execution_stage_binding"), stage_binding):
            raise ValueError("Source/fork execution belongs to a different frozen role stage")
        origin = (
            identity
            if kind == "source"
            else _read(path.parent / "bank_plan.json")["origin_identity"]
        )
        keys = ("seed", "arm") if kind == "source" else ("seed", "arm", "step")
        if any(origin.get(key) != expected[key] for key in keys):
            raise ValueError("Completed source/fork identity differs from the frozen task")
        if kind == "source":
            if (
                receipt.get("steps") != 128
                or receipt.get("source_optimizer_steps") != 128
                or [item["step"] for item in receipt["checkpoints"]] != list(range(1, 129))
            ):
                raise ValueError("Frozen source task requires all 128 complete updates")
        elif (
            Counter(b["role"] for b in receipt["banks"]) != {"calibration_pool": 24, "heldout": 12}
            or receipt.get("scratch_optimizer_steps") != 108
            or receipt.get("origin_restored") is not True
        ):
            raise ValueError("Frozen fork task requires all 36 banks and full origin restoration")
    elif kind == "observation":
        from .vlm_observation import (
            _completed_task_rows,
            _validate_observation_stage,
            _worker_assignment,
        )

        authorization = receipt.get("execution_stage_binding")
        if not _authorization_descends_from(authorization, stage_binding):
            raise ValueError("Observation completion belongs to another role authorization")
        stage = _bound_json(authorization)
        manifest = _bound_json(stage["observation_manifest"])
        _validate_observation_stage(config, {"v3_stage_lock": authorization}, manifest)
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("runtime_identity") != manifest["runtime_identity"]
            or receipt.get("manifest_hash") != manifest["manifest_hash"]
            or receipt.get("worker_index") != expected["worker_index"]
            or receipt.get("authorization_reverified_after_work") is not True
            or stage.get("origin_id") != expected["origin_id"]
            or stage.get("observation_purpose") != expected["purpose"]
            or stage.get("observation_operation") != expected["operation"]
        ):
            raise ValueError("Observation worker identity differs from the frozen task")
        worker_root = path.parent
        if worker_root.name != f"worker_{expected['worker_index']}":
            raise ValueError("Observation receipt is not in its actual worker output")
        assigned = {
            t["task_id"]
            for t in manifest["tasks"]
            if _worker_assignment(t, manifest["policies"], manifest["workers"])
            == expected["worker_index"]
        }
        _completed_task_rows(worker_root.parent, manifest, allowed_task_ids=assigned)
        completion_markers = [
            _file_binding(worker_root / t["output_path"] / "COMPLETE.json")
            for t in manifest["tasks"]
            if t["task_id"] in assigned
        ]
        completion_markers.extend([stage["observation_manifest"], authorization])
    else:
        if not (path.parent / "COMPLETE.json").is_file():
            raise ValueError("Prediction/evaluation artifact has no completed manifest")
        manifest = verify_manifest(path.parent)
        completion_markers = [
            _file_binding(path.parent / name) for name in ("RUN_MANIFEST.json", "COMPLETE.json")
        ]
        if manifest.get("status") != "COMPLETE":
            raise ValueError("Prediction/evaluation artifact is partial")
        required_kind = (
            "V3_FROZEN_PREDICTIONS" if kind == "prediction" else "V3_RESPONSE_EVALUATION"
        )
        if (
            receipt.get("kind") != required_kind
            or any(receipt.get(k) != v for k, v in common.items())
            or receipt.get("origin_id") != expected["origin_id"]
            or receipt.get("role") != expected["role"]
        ):
            raise ValueError("Prediction/evaluation identity differs from the frozen task")
        if kind == "prediction":
            if file_hash(receipt["arrays"]["path"]) != receipt["arrays"]["sha256"]:
                raise ValueError("Frozen prediction bytes changed")
        else:
            prediction = _bound_json(receipt["prediction_binding"])
            if prediction.get("origin_id") != expected["origin_id"]:
                raise ValueError("Evaluation references predictions from another origin")
    return {
        "kind": kind,
        "binding": copy.deepcopy(binding),
        "completion_markers": completion_markers,
        "reference_status": receipt.get("reference_status"),
        "scientific_status": "NOT_CERTIFIED",
    }


def finalize_q5_stage(config, bindings, *, task_receipts, out):
    """Aggregate only measured, hash-verified task evidence; never submit work.

    task_receipts maps each frozen task id to its receipt {path,sha256}. Source
    and fork receipts are result.json; observation receipts are their worker's
    execution authorization receipt; model adapters publish PREDICTION_LOCK.json
    and RESPONSE_EVALUATION_RECEIPT.json with complete .io manifests.
    """
    from .io import finalize_run

    supplied = _read(task_receipts) if isinstance(task_receipts, (str, Path)) else task_receipts
    stage = _bound_json(bindings["v3_stage_lock"])
    role = stage["role"]
    expected = _q5_expected_tasks(config, role)
    if stage.get("required_tasks") != expected or stage.get("expected_task_ids") != sorted(
        expected
    ):
        raise ValueError("Role completion requires the entire task matrix frozen before execution")
    for seed in stage["seeds"]:
        _stage_gate(config, bindings, seed, operation="train-source")
    if not isinstance(supplied, dict) or set(supplied) - set(expected):
        raise ValueError("Task receipts must map only frozen role task ids to file bindings")
    verified, failures = {}, {}
    for task_id, binding in supplied.items():
        try:
            verified[task_id] = _verify_q5_task(
                config, bindings["v3_stage_lock"], expected[task_id], binding
            )
        except (ValueError, KeyError, OSError, PermissionError) as exc:
            failures[task_id] = {"error_type": type(exc).__name__, "reason": str(exc)}
    missing = sorted(set(expected) - set(supplied))
    status = "TECHNICAL_FAILURE" if failures else "INCOMPLETE" if missing else "COMPLETE"
    result = {
        "kind": "V3_Q5_STAGE_COMPLETION",
        "status": status,
        "role": role,
        "config_hash": canonical_hash(config),
        "source_hash": canonical_hash(source_hashes()),
        "stage_binding": copy.deepcopy(bindings["v3_stage_lock"]),
        "expected_task_ids": sorted(expected),
        "completed_task_ids": sorted(verified),
        "missing_task_ids": missing,
        "technical_failures": failures,
        "verified_tasks": verified,
        "scientific_status": "NOT_CERTIFIED",
        "positive_effect_is_completion_gate": False,
        "unresolved_reference_blocks_scientific_ranking": True,
        "reference_statuses": {
            k: v["reference_status"] for k, v in verified.items() if v["kind"] == "evaluation"
        },
        "online_ssvc": False,
    }
    root = Path(out).resolve()
    with frozen_writer(root):
        _publish(root / "TASK_RECEIPTS.json", supplied)
        _publish(root / "Q5_STAGE_COMPLETION.json", result)
        finalize_run(
            root,
            {
                "config": result["config_hash"],
                "source": result["source_hash"],
                "stage": bindings["v3_stage_lock"]["sha256"],
            },
            status=status,
        )
    return {**result, "receipt": _file_binding(root / "Q5_STAGE_COMPLETION.json")}


def _verify_q5_completion(config, binding, role):
    from .io import verify_manifest

    receipt = _bound_json(binding)
    root = Path(binding["path"]).parent
    if not (root / "COMPLETE.json").is_file():
        raise ValueError("Preceding frozen stage task list is incomplete")
    manifest = verify_manifest(root)
    expected = sorted(_q5_expected_tasks(config, role))
    if (
        manifest.get("status") != "COMPLETE"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("kind") != "V3_Q5_STAGE_COMPLETION"
        or receipt.get("role") != role
        or receipt.get("config_hash") != canonical_hash(config)
        or receipt.get("source_hash") != canonical_hash(source_hashes())
        or receipt.get("expected_task_ids") != expected
        or receipt.get("completed_task_ids") != expected
        or receipt.get("technical_failures")
    ):
        raise ValueError("Preceding frozen stage task list is incomplete or changed")
    # The aggregator audited original tensors/shards. Subsequent role gates bind
    # that immutable audit and every input completion marker without repeatedly
    # reading the full previous matrix before each GPU task. Consumers still
    # hash each raw file when it is used again.
    stage_binding = receipt["stage_binding"]
    stage = _bound_json(stage_binding)
    if stage.get("required_tasks") != _q5_expected_tasks(config, role):
        raise ValueError("Preceding role task matrix changed")
    for task_id in stage["required_tasks"]:
        verified = receipt["verified_tasks"][task_id]
        _bound_json(verified["binding"])
        if not verified.get("completion_markers"):
            raise ValueError("Completed task is missing its original completion markers")
        for marker in verified["completion_markers"]:
            if file_hash(marker["path"]) != marker["sha256"]:
                raise ValueError("Preceding task completion manifest changed")
    return receipt


def prepare_q5_stage(
    config, bindings, *, q4_receipt, role, out, selection_lock=None, previous_completion=None
):
    """Freeze fresh probes, role gates, and source tasks without loading a model.

    A prepared stage is an executable task list, never operator authorization.
    Observation whitelists are derived later from actual checkpoint identities.
    """
    if role not in config["qwen"]["seed_roles"]:
        raise ValueError("Unknown Q5 seed role")
    audit_v3_compatibility(config, bindings)
    parent = _bound_json(bindings["parent_validated_plan"])
    prior_stage = _bound_json(bindings["v3_stage_lock"])
    _technical_gate(config, prior_stage)
    stage = {
        "schema": "ssvc-v3-execution-stage-1",
        "phase": "Q5",
        "role": role,
        "config_hash": canonical_hash(config),
        "source_hash": canonical_hash(source_hashes()),
        "technical_receipts": prior_stage["technical_receipts"],
        "v3_gpu_smoke": copy.deepcopy(q4_receipt),
        "operations": ["train-source", "make-forks", "observe-vlm"],
        "seeds": list(config["qwen"]["seed_roles"][role]),
        "workers": [0, 1],
        "max_concurrent_project_gpus": 2,
        "observation_manifest_hashes": [],
        "operator_authorization": "REQUIRES_EXPLICIT_EXECUTION_FLAGS",
        "online_ssvc": False,
    }
    if role != "development":
        if selection_lock is None or previous_completion is None:
            raise ValueError("Later Q5 roles require frozen selection and preceding completion")
        stage["selection_lock"] = copy.deepcopy(selection_lock)
        previous = "development" if role == "interval_calibration" else "interval_calibration"
        stage[previous + "_completion"] = copy.deepcopy(previous_completion)

    # Read only audited control data and the historical, already opened S1 panel.
    warm = Path(parent["paths"]["parent_warm"]) / "bank_manifest.json"
    if file_hash(warm) != parent["warm_binding"]["files"]["bank_manifest.json"]:
        raise ValueError("Historical S1 control panel binding changed")
    old_panel = _read(warm)["plan"]["control_prompts"]
    excluded = sorted({p["scene"]["base_scene_id"] for p in old_panel})
    data_root = Path(parent["paths"]["raw_dataset_root"])
    matches = [
        p for p in parent["training_data_binding"]["files"] if Path(p).name == "control.jsonl"
    ]
    if len(matches) != 1:
        raise ValueError("One hash-audited original control.jsonl is required")
    control = data_root / matches[0]
    if file_hash(control) != parent["training_data_binding"]["files"][matches[0]]:
        raise ValueError("Original control pool changed")
    scenes = [json.loads(line) for line in control.read_text().splitlines() if line]
    probes = build_probe_panel(scenes, excluded_base_scene_ids=excluded, data_root=data_root)
    if len(probes) != 36 or len({p["scene"]["base_scene_id"] for p in probes}) != 18:
        raise ValueError("Q5 requires exactly 18 fresh control scenes and 36 paired probes")

    root = Path(out).resolve()
    with frozen_writer(root):
        _publish(root / "CONFIG.json", config)
        _publish(root / "PROBES.json", probes)
        stage["probe_panel"] = _file_binding(root / "PROBES.json")
        stage["probe_selection"] = {
            "rule": "hash of namespace and base scene within 18 fixed metadata cells",
            "excluded_historical_S1_base_scene_ids": excluded,
            "old_S1_bank_manifest": _file_binding(warm),
            "control_pool": _file_binding(control),
            "sealed_confirm_opened": False,
        }
        tasks = []
        for seed in stage["seeds"]:
            for arm in SOURCE_ARMS:
                tasks.append(
                    {
                        "task_id": f"source_{seed}_{arm}",
                        "seed": seed,
                        "arm": arm,
                        "worker": len(tasks) % 2,
                        "out": str(root / "sources" / f"{seed}_{arm}"),
                    }
                )
        stage["source_tasks"] = tasks
        stage["required_tasks"] = _q5_expected_tasks(config, role)
        stage["expected_task_ids"] = sorted(stage["required_tasks"])
        _publish(root / "Q5_STAGE_LOCK.json", stage)
        resolved = {**bindings, "v3_stage_lock": _file_binding(root / "Q5_STAGE_LOCK.json")}
        for seed in stage["seeds"]:
            _stage_gate(config, resolved, seed, operation="train-source")
        _publish(root / "SOURCE_BINDINGS.json", resolved)
        commands = [
            {
                **task,
                "argv": [
                    "python",
                    "-m",
                    "src.modeling_v3.cli",
                    "train-source",
                    "--config",
                    str(root / "CONFIG.json"),
                    "--bindings",
                    str(root / "SOURCE_BINDINGS.json"),
                    "--seed",
                    str(task["seed"]),
                    "--arm",
                    task["arm"],
                    "--device",
                    "cuda:0",
                    "--out",
                    task["out"],
                    "--allow-gpu",
                    "--allow-training",
                    "--acknowledge-new-experiment",
                ],
            }
            for task in tasks
        ]
        result = {
            "status": "PREPARED_GPU_NOT_SUBMITTED",
            "phase": "Q5",
            "role": role,
            "source_bindings": _file_binding(root / "SOURCE_BINDINGS.json"),
            "stage_lock": resolved["v3_stage_lock"],
            "probe_panel": stage["probe_panel"],
            "source_commands": commands,
            "model_loaded": False,
            "submitted": False,
        }
        _publish(root / "PREPARATION.json", result)
    return result


def prepare_q5_forks(config, bindings, *, source_root, out):
    """Resolve source response checkpoints into exact, reviewable fork commands."""
    source = _verified_execution(source_root, config, unit="V3_SOURCE")
    seed, arm = source["identity"]["seed"], source["identity"]["arm"]
    _stage_gate(config, bindings, seed, operation="make-forks")
    root = Path(out).resolve()
    with frozen_writer(root):
        _publish(root / "CONFIG.json", config)
        _publish(root / "SOURCE_BINDINGS.json", bindings)
        commands = []
        for origin in source["response_plans"]:
            if file_hash(origin["bank_plan"]) != origin["bank_plan_sha256"]:
                raise ValueError("Source-generated candidate bank plan changed")
            spec = resolve_checkpoint_binding(origin["checkpoint"])
            if spec["sha256"] != origin["checkpoint_sha256"]:
                raise ValueError("Source response checkpoint changed")
            origin_id = f"{seed}_{arm}_{origin['step']}"
            commands.append(
                {
                    "origin_id": origin_id,
                    "worker": int(canonical_hash(origin_id)[:8], 16) % 2,
                    "checkpoint": spec,
                    "bank_plan": _file_binding(origin["bank_plan"]),
                    "out": str(root / origin_id),
                    "argv": [
                        "python",
                        "-m",
                        "src.modeling_v3.cli",
                        "make-forks",
                        "--config",
                        str(root / "CONFIG.json"),
                        "--bindings",
                        str(root / "SOURCE_BINDINGS.json"),
                        "--checkpoint",
                        spec["path"],
                        "--bank-plan",
                        origin["bank_plan"],
                        "--device",
                        "cuda:0",
                        "--out",
                        str(root / origin_id),
                        "--allow-gpu",
                        "--allow-training",
                        "--acknowledge-new-experiment",
                    ],
                }
            )
        result = {
            "status": "PREPARED_GPU_NOT_SUBMITTED",
            "source": _file_binding(Path(source_root) / "result.json"),
            "fork_commands": commands,
            "model_loaded": False,
            "submitted": False,
        }
        _publish(root / "PREPARATION.json", result)
    return result


def _q5_generation_tasks(
    config,
    *,
    origin_id,
    policies,
    banks,
    probe_binding,
    prompt_ids,
    purpose,
    include_direct_count=True,
):
    from .vlm_observation import freeze_observation_task

    tasks = []
    common = {
        "operation": "generate",
        "origin_id": origin_id,
        "prompt_file": probe_binding,
        "prompt_ids": prompt_ids,
        "max_new_tokens": config["qwen"]["max_new_tokens"],
    }

    def add(candidate, proposal, pair, role, start, stop):
        tasks.append(
            freeze_observation_task(
                **common,
                candidate_id=candidate,
                proposal=proposal,
                proposal_candidates=pair,
                role=role,
                draw_start=start,
                draw_stop=stop,
                rng_namespace=canonical_hash([NAMESPACE, "Q5", origin_id, purpose]),
            )
        )

    if purpose == "measurement":
        n = config["qwen"]["observation_n_primary"]
        pilot = int(n * config["observation"]["pilot_fraction"])
        add("origin", "ORIGIN", ["origin"], "pilot", 0, pilot)
        add("origin", "ORIGIN", ["origin"], "main", pilot, n)
        if include_direct_count:
            # Each distinct heldout policy receives n independent direct draws;
            # the actual extra generation cost is retained in PREPARATION.json.
            for bank in banks:
                if bank["role"] == "heldout":
                    for spec in CANDIDATES:
                        candidate = bank["bank_id"] + "/" + spec["id"]
                        add(candidate, "DIRECT", [candidate], "main", 0, n)
    elif purpose == "reference":
        n = config["qwen"]["reference"]["initial_draws"]
        add("origin", "ORIGIN", ["origin"], "reference", 0, n)
        heldout = sorted(
            (b for b in banks if b["role"] == "heldout"),
            key=lambda b: canonical_hash([NAMESPACE, "Q5_MIX", origin_id, b["bank_id"]]),
        )
        count = config["qwen"]["reference"]["pair_mixture_spotcheck_banks_per_origin"]
        if len(heldout) < count:
            raise ValueError("Not enough heldout banks for the frozen MIX spot checks")
        for bank in heldout[:count]:
            pair = [bank["bank_id"] + "/joint_1", bank["bank_id"] + "/joint_0"]
            add(pair[0], "MIX", pair, "reference", 0, n)
    else:
        raise ValueError("Q5 observation purpose must be measurement or reference")
    if any(t["candidate_id"] not in policies for t in tasks):
        raise ValueError("Q5 generation requires actual bound candidate policies")
    return tasks


def prepare_q5_observation(
    config,
    bindings,
    *,
    forks_root,
    origin_checkpoint,
    out,
    purpose="measurement",
    operation="generate",
    generation_manifest=None,
    generation_root=None,
    prediction_binding=None,
    include_direct_count=True,
):
    """Freeze real Q5 tasks on CPU, then bind the stage to their exact manifest.

    Generation is frozen first. A score plan is created only after every source
    generation task completed with immutable shards. Reference planning requires
    a V3_FROZEN_PREDICTIONS receipt with config/source/origin_id and an arrays
    path+sha256 binding; the reference remains estimated and assessor-only.
    """
    from .vlm_observation import (
        _completed_task_rows,
        _worker_assignment,
        freeze_observation_task,
        freeze_task_manifest,
    )

    if operation not in {"generate", "score"}:
        raise ValueError("Q5 observation operation must be generate or score")
    forks = _verified_execution(forks_root, config, unit="V3_FORKS")
    if forks["identity"].get("Q4_bridge") is not False:
        raise ValueError("Q4 bridge banks cannot serve as new Q5 observations")
    bank_plan = _read(Path(forks_root) / "bank_plan.json")
    origin = bank_plan["origin_identity"]
    seed = origin["seed"]
    _stage_gate(config, bindings, seed, operation="observe-vlm")
    stage = _bound_json(bindings["v3_stage_lock"])
    probes = _bound_json(stage["probe_panel"])
    if len(probes) != 36 or len({p["prompt_id"] for p in probes}) != 36:
        raise ValueError("Q5 requires the frozen 36-probe panel")
    if Counter(b["role"] for b in forks["banks"]) != {"calibration_pool": 24, "heldout": 12}:
        raise ValueError("Q5 requires complete 24/12 real candidate banks")
    origin_id = f"{seed}_{origin['arm']}_{origin['step']}"
    spec = resolve_checkpoint_binding(origin_checkpoint)
    if (
        spec["identity"].get("source_hash") != canonical_hash(source_hashes())
        or spec["identity"].get("config_hash") != canonical_hash(config)
        or any(spec["identity"].get(key) != origin[key] for key in ("seed", "arm", "step"))
    ):
        raise ValueError("Observation source checkpoint differs from the frozen Q5 origin")
    receipt = _read(Path(spec["path"]).parent / "result.json")
    if receipt["state_hash"] != origin["state_hash"] or not receipt.get("inference_fingerprint"):
        raise ValueError(
            "Q5 origin needs its actual complete state and measured inference fingerprint"
        )
    policies = {
        "origin": {"checkpoint": spec, "inference_fingerprint": receipt["inference_fingerprint"]}
    }
    for bank in forks["banks"]:
        for candidate, record in bank["checkpoints"].items():
            policies[bank["bank_id"] + "/" + candidate] = {
                "checkpoint": {k: record[k] for k in ("path", "sha256", "identity")},
                "inference_fingerprint": record["inference_fingerprint"],
            }
    if purpose == "reference":
        from .io import verify_manifest

        if prediction_binding is None:
            raise PermissionError(
                "Freeze prediction bytes and source before opening heldout reference"
            )
        prediction = _bound_json(prediction_binding)
        prediction_root = Path(prediction_binding["path"]).parent
        if not (prediction_root / "COMPLETE.json").is_file():
            raise ValueError("Reference requires a completed prediction artifact")
        if verify_manifest(prediction_root).get("status") != "COMPLETE":
            raise ValueError("Reference cannot open on a partial prediction artifact")
        if (
            prediction.get("kind") != "V3_FROZEN_PREDICTIONS"
            or prediction.get("config_hash") != canonical_hash(config)
            or prediction.get("source_hash") != canonical_hash(source_hashes())
            or prediction.get("origin_id") != origin_id
        ):
            raise ValueError(
                "Reference requires the frozen predictions for this exact origin/config/source"
            )
        if file_hash(prediction["arrays"]["path"]) != prediction["arrays"]["sha256"]:
            raise ValueError("Frozen prediction arrays changed")

    runtime_identity = planned_runtime_identity(config, bindings)
    expected_generation = _q5_generation_tasks(
        config,
        origin_id=origin_id,
        policies=policies,
        banks=forks["banks"],
        probe_binding=stage["probe_panel"],
        prompt_ids=[p["prompt_id"] for p in probes],
        purpose=purpose,
        include_direct_count=include_direct_count,
    )
    generation = freeze_task_manifest(expected_generation, policies, runtime_identity, workers=2)
    if operation == "generate":
        manifest = generation
    else:
        if generation_manifest is None or generation_root is None:
            raise ValueError("Scoring must bind a completed generation manifest and shard root")
        actual = _bound_json(generation_manifest)
        if actual["manifest_hash"] != generation["manifest_hash"]:
            raise ValueError("Generation inputs differ from this frozen Q5 purpose/origin")
        _completed_task_rows(generation_root, actual)
        tasks = []
        heldout_policies = {
            b["bank_id"] + "/" + c["id"]
            for b in forks["banks"]
            if b["role"] == "heldout"
            for c in CANDIDATES
        }
        for task in actual["tasks"]:
            if task["proposal"] == "DIRECT":
                continue  # Generation already rescored its own policy for parity.
            worker = _worker_assignment(task, policies, 2)
            task_root = Path(generation_root) / f"worker_{worker}" / task["output_path"]
            marker = _read(task_root / "COMPLETE.json")
            samples = [_file_binding(task_root / row["path"]) for row in marker["shards"]]
            candidates = (
                task["proposal_candidates"]
                if task["proposal"] == "MIX"
                else sorted(heldout_policies if purpose == "reference" else policies)
            )
            for candidate in candidates:
                kwargs = {
                    key: task[key]
                    for key in (
                        "origin_id",
                        "prompt_file",
                        "prompt_ids",
                        "role",
                        "draw_start",
                        "draw_stop",
                        "rng_namespace",
                        "proposal",
                        "proposal_candidates",
                        "max_new_tokens",
                    )
                }
                tasks.append(
                    freeze_observation_task(
                        **kwargs, operation="score", candidate_id=candidate, sample_files=samples
                    )
                )
        manifest = freeze_task_manifest(tasks, policies, runtime_identity, workers=2)

    return _publish_q5_observation_plan(
        config,
        bindings,
        stage=stage,
        manifest=manifest,
        origin_id=origin_id,
        seed=seed,
        purpose=purpose,
        operation=operation,
        prediction_binding=prediction_binding,
        forks_binding=_file_binding(Path(forks_root) / "result.json"),
        out=out,
    )


def _publish_q5_observation_plan(
    config,
    bindings,
    *,
    stage,
    manifest,
    origin_id,
    seed,
    purpose,
    operation,
    prediction_binding,
    forks_binding,
    out,
    extra_stage=None,
):
    root = Path(out).resolve()
    with frozen_writer(root):
        _publish(root / "CONFIG.json", config)
        _publish(root / "OBSERVATION_TASKS.json", manifest)
        child = {
            **stage,
            "authorization_parent": copy.deepcopy(bindings["v3_stage_lock"]),
            "runtime_seed": seed,
            "seeds": [seed],
            "operations": ["observe-vlm"],
            "observation_manifest_hashes": [manifest["manifest_hash"]],
            "observation_manifest": _file_binding(root / "OBSERVATION_TASKS.json"),
            "observation_purpose": purpose,
            "observation_operation": operation,
            "origin_id": origin_id,
            "prediction_binding": prediction_binding,
            "forks_result": forks_binding,
            **(extra_stage or {}),
        }
        _publish(root / "Q5_OBSERVATION_STAGE_LOCK.json", child)
        resolved = {
            **bindings,
            "v3_stage_lock": _file_binding(root / "Q5_OBSERVATION_STAGE_LOCK.json"),
        }
        _publish(root / "SOURCE_BINDINGS.json", resolved)
        commands = [
            {
                "worker": worker,
                "argv": [
                    "python",
                    "-m",
                    "src.modeling_v3.cli",
                    "observe-vlm",
                    "--config",
                    str(root / "CONFIG.json"),
                    "--bindings",
                    str(root / "SOURCE_BINDINGS.json"),
                    "--task-manifest",
                    str(root / "OBSERVATION_TASKS.json"),
                    "--worker-index",
                    str(worker),
                    "--workers",
                    "2",
                    "--device",
                    "cuda:0",
                    "--out",
                    str(root / "observations"),
                    "--allow-gpu",
                    "--acknowledge-new-experiment",
                ],
            }
            for worker in range(2)
        ]
        result = {
            "status": "PREPARED_GPU_NOT_SUBMITTED",
            "origin_id": origin_id,
            "purpose": purpose,
            "operation": operation,
            "manifest": _file_binding(root / "OBSERVATION_TASKS.json"),
            "source_bindings": _file_binding(root / "SOURCE_BINDINGS.json"),
            "commands": commands,
            "counts": {
                name: sum(
                    len(t["request_keys"]) for t in manifest["tasks"] if t["operation"] == name
                )
                for name in ("generate", "score")
            },
            "labels": "AUDITOR_ONLY_UNTIL_SELECTED_BANK_QUERY"
            if purpose == "measurement"
            else "INDEPENDENT_ASSESSOR_ONLY",
            "reference_status": "NOT_RUN" if purpose == "reference" else "NOT_OPENED",
            "model_loaded": False,
            "submitted": False,
        }
        _publish(root / "PREPARATION.json", result)
    return result


def _reference_extension_specs(config, history_tasks, reports, *, origin_id):
    """Derive nonoverlapping next ranges using only reference precision reports."""
    looks = [
        config["qwen"]["reference"]["initial_draws"],
        *config["qwen"]["reference"]["next_draws"],
    ]
    if looks != [4096, 8192, 16384, 32768, 65536]:
        raise ValueError("Reference extensions require the frozen five cumulative looks")
    streams = defaultdict(list)
    prototype = None
    for task in history_tasks:
        if (
            task["operation"] != "generate"
            or task["role"] != "reference"
            or task["origin_id"] != origin_id
            or task["proposal"] not in {"ORIGIN", "MIX"}
        ):
            raise ValueError("Reference history contains a non-reference generation task")
        prototype = task if task["proposal"] == "ORIGIN" else prototype
        for prompt_id in task["prompt_ids"]:
            key = (task["proposal"], tuple(task["proposal_candidates"]), prompt_id)
            streams[key].append(task)
    if prototype is None:
        raise ValueError("Reference history needs its independent ORIGIN packet")
    measured = {}
    for key, tasks in streams.items():
        end, namespace = 0, tasks[0]["rng_namespace"]
        for task in sorted(tasks, key=lambda t: t["draw_start"]):
            next_look = (
                looks[0]
                if end == 0
                else (looks[looks.index(end) + 1] if end in looks[:-1] else None)
            )
            if (
                task["draw_start"] != end
                or task["draw_stop"] != next_look
                or task["rng_namespace"] != namespace
            ):
                raise ValueError(
                    "Reference history has repeated draws, gaps, or changed RNG stream"
                )
            end = task["draw_stop"]
        measured[key] = (end, namespace)
    origin_ns = {value for key, value in measured.items() if key[0] == "ORIGIN"}
    if len(origin_ns) != 1:
        raise ValueError("ORIGIN reference panel must share one cumulative count and namespace")
    current, namespace = next(iter(origin_ns))
    planned, seen_units = {}, set()
    pairs = {name: (left, right) for name, left, right in CONTRASTS}

    def extend(key, report, *, mandatory=False):
        n, rng = measured.get(key, (0, namespace))
        if report is not None and report.get("n") != n:
            raise ValueError("Precision report count differs from original reference draws")
        if n == 0:
            stop = looks[0] if mandatory else None
        elif n == looks[-1] or (report or {}).get("empirical_precision_met") is True:
            stop = None
        elif mandatory or (report or {}).get("status") in {
            "REFERENCE_EXTEND",
            "REFERENCE_UNRESOLVED",
            "REFERENCE_REQUIRES_INDEPENDENT_MIX",
        }:
            stop = looks[looks.index(n) + 1]
        else:
            stop = None
        if stop is not None:
            planned[key] = (n, stop, rng)

    for unit in reports:
        uid = (unit["bank_id"], unit["contrast_id"], unit["prompt_id"])
        if uid in seen_units or unit["contrast_id"] not in pairs:
            raise ValueError("Reference precision unit is duplicated or has an unknown contrast")
        seen_units.add(uid)
        prompt_id = unit["prompt_id"]
        origin_key = ("ORIGIN", ("origin",), prompt_id)
        if origin_key not in measured or unit["origin"].get("n") != current:
            raise ValueError("Reference precision report does not match its actual ORIGIN stream")
        report = unit["origin"]
        if report.get("status") == "REFERENCE_EXTEND":
            extend(origin_key, report)
        names = tuple(unit["bank_id"] + "/" + candidate for candidate in pairs[unit["contrast_id"]])
        mix_key = ("MIX", names, prompt_id)
        reversed_key = ("MIX", names[::-1], prompt_id)
        if reversed_key in measured:
            if mix_key in measured:
                raise ValueError("Duplicate equivalent MIX streams need explicit protocol handling")
            mix_key = reversed_key
        mix = unit.get("mix")
        if mix is not None and mix_key not in measured:
            raise ValueError("MIX precision report has no actual original packet")
        mandatory = (
            report.get("status") == "REFERENCE_REQUIRES_INDEPENDENT_MIX"
            or report.get("tail_diagnostics", {}).get("requires_independent_mix") is True
            or unit.get("crosscheck_consistent") is False
        )
        if mandatory or mix is not None:
            extend(mix_key, mix, mandatory=mandatory)
    # ORIGIN generation is shared by all candidates. An origin extension keeps
    # the complete fixed panel at the same look, even if one unit triggered it.
    if any(key[0] == "ORIGIN" for key in planned):
        stop = looks[looks.index(current) + 1]
        for key in measured:
            if key[0] == "ORIGIN":
                planned[key] = (current, stop, namespace)
    grouped = defaultdict(list)
    for (proposal, pair, prompt_id), (start, stop, rng) in planned.items():
        grouped[proposal, pair, start, stop, rng].append(prompt_id)
    return [
        {
            "proposal": proposal,
            "proposal_candidates": list(pair),
            "candidate_id": pair[0],
            "prompt_ids": sorted(prompts),
            "draw_start": start,
            "draw_stop": stop,
            "rng_namespace": rng,
        }
        for (proposal, pair, start, stop, rng), prompts in sorted(grouped.items())
    ]


def prepare_q5_reference_extension(
    config,
    bindings,
    *,
    precision_receipt,
    out,
    operation="generate",
    generation_manifest=None,
    generation_root=None,
):
    """Prepare registered reference increments or mandatory MIX, never submit.

    Only V3_REFERENCE_PRECISION receipts from the reference evaluator qualify.
    Each previous generation/scoring batch must already be complete. No model
    prediction value or method ranking enters the range-selection calculation.
    """
    from .io import verify_manifest
    from .vlm_observation import (
        _completed_task_rows,
        _worker_assignment,
        freeze_observation_task,
        freeze_task_manifest,
    )

    if operation not in {"generate", "score"}:
        raise ValueError("Reference continuation operation must be generate or score")
    receipt = _bound_json(precision_receipt)
    precision_root = Path(precision_receipt["path"]).parent
    if (
        not (precision_root / "COMPLETE.json").is_file()
        or verify_manifest(precision_root).get("status") != "COMPLETE"
    ):
        raise ValueError("Reference precision must come from a completed evaluator artifact")
    if (
        receipt.get("kind") != "V3_REFERENCE_PRECISION"
        or receipt.get("config_hash") != canonical_hash(config)
        or receipt.get("source_hash") != canonical_hash(source_hashes())
        or receipt.get("precision_only") is not True
        or receipt.get("predictor_rankings_used") is not False
    ):
        raise ValueError("Continuation requires a source-bound reference-only precision receipt")
    diagnostics = _bound_json(receipt["decision_evidence"])
    reports = diagnostics.get("unit_reports")
    if (
        not isinstance(reports, list)
        or not reports
        or diagnostics.get("precision_only") is not True
        or diagnostics.get("predictor_rankings_used") is not False
        or ("unit_reports" in receipt and receipt["unit_reports"] != reports)
    ):
        raise ValueError("Precision request differs from its original unit diagnostics")
    if receipt.get("current_draws") != reports[0]["origin"].get("n") or receipt.get(
        "next_draws"
    ) not in (None, 8192, 16384, 32768, 65536):
        raise ValueError("Reference continuation count is outside the registered precision looks")
    prediction_binding = receipt["prediction_binding"]
    prediction = _bound_json(prediction_binding)
    prediction_root = Path(prediction_binding["path"]).parent
    if (
        not (prediction_root / "COMPLETE.json").is_file()
        or verify_manifest(prediction_root).get("status") != "COMPLETE"
        or prediction.get("kind") != "V3_FROZEN_PREDICTIONS"
        or prediction.get("origin_id") != receipt["origin_id"]
        or prediction.get("config_hash") != canonical_hash(config)
        or prediction.get("source_hash") != canonical_hash(source_hashes())
        or file_hash(prediction["arrays"]["path"]) != prediction["arrays"]["sha256"]
    ):
        raise ValueError("Reference continuation prediction binding changed")
    expected_units = {
        (u["bank_id"], u["contrast_id"], p)
        for u in prediction["query_units"]
        for p in prediction["probe_ids"]
    }
    actual_units = {(u["bank_id"], u["contrast_id"], u["prompt_id"]) for u in reports}
    if actual_units != expected_units:
        raise ValueError("Reference precision needs every frozen heldout query unit")
    stage = _bound_json(bindings["v3_stage_lock"])
    seed = int(receipt["origin_id"].split("_", 1)[0])
    _stage_gate(config, bindings, seed, operation="observe-vlm")
    policies, runtime_identity, history, probe_binding = None, None, [], None
    if not receipt.get("reference_batches"):
        raise ValueError("Reference continuation needs completed original batches")
    for batch in receipt["reference_batches"]:
        gen, score = (
            _bound_json(batch["generation_manifest"]),
            _bound_json(batch["scoring_manifest"]),
        )
        for manifest, root in ((gen, batch["generation_root"]), (score, batch["scoring_root"])):
            if any(
                t["origin_id"] != receipt["origin_id"] or t["role"] != "reference"
                for t in manifest["tasks"]
            ):
                raise ValueError("Precision history contains another origin or predictor labels")
            if policies is None:
                policies, runtime_identity = manifest["policies"], manifest["runtime_identity"]
            if manifest["policies"] != policies or manifest["runtime_identity"] != runtime_identity:
                raise ValueError("Reference history changes actual policy/runtime identities")
            _completed_task_rows(root, manifest)
        history.extend(gen["tasks"])
        for task in gen["tasks"]:
            if probe_binding is None:
                probe_binding = task["prompt_file"]
            if task["prompt_file"] != probe_binding:
                raise ValueError("Reference continuation cannot change its fixed probe panel")
    if runtime_identity != planned_runtime_identity(config, bindings):
        raise ValueError("Reference history belongs to a different runtime lock")
    specs = _reference_extension_specs(config, history, reports, origin_id=receipt["origin_id"])
    if not specs:
        result = {
            "status": "NO_REGISTERED_REFERENCE_TASKS",
            "scientific_status": "NOT_CERTIFIED",
            "reference_status": "REFERENCE_UNRESOLVED_OR_PRECISION_MET_SEE_UNIT_REPORTS",
            "precision_receipt": precision_receipt,
            "model_loaded": False,
            "submitted": False,
            "protocol_cap": 65536,
            "no_extension_beyond_protocol": True,
        }
        root = Path(out).resolve()
        with frozen_writer(root):
            _publish(root / "PREPARATION.json", result)
        return result
    tasks = [
        freeze_observation_task(
            **spec,
            operation="generate",
            origin_id=receipt["origin_id"],
            prompt_file=probe_binding,
            role="reference",
            max_new_tokens=64,
        )
        for spec in specs
    ]
    generation = freeze_task_manifest(tasks, policies, runtime_identity, workers=2)
    if operation == "generate":
        manifest = generation
    else:
        if generation_manifest is None or generation_root is None:
            raise ValueError("Reference scoring must bind its completed incremental generation")
        actual = _bound_json(generation_manifest)
        if actual["manifest_hash"] != generation["manifest_hash"]:
            raise ValueError("Reference increment differs from the precision-only frozen request")
        _completed_task_rows(generation_root, actual)
        candidates = sorted(
            {u["bank_id"] + "/" + c["id"] for u in prediction["query_units"] for c in CANDIDATES}
        )
        tasks = []
        for task in actual["tasks"]:
            worker = _worker_assignment(task, policies, 2)
            task_root = Path(generation_root) / f"worker_{worker}" / task["output_path"]
            marker = _read(task_root / "COMPLETE.json")
            samples = [_file_binding(task_root / s["path"]) for s in marker["shards"]]
            endpoints = task["proposal_candidates"] if task["proposal"] == "MIX" else candidates
            for candidate in endpoints:
                kwargs = {
                    key: task[key]
                    for key in (
                        "origin_id",
                        "prompt_file",
                        "prompt_ids",
                        "role",
                        "draw_start",
                        "draw_stop",
                        "rng_namespace",
                        "proposal",
                        "proposal_candidates",
                        "max_new_tokens",
                    )
                }
                tasks.append(
                    freeze_observation_task(
                        **kwargs, operation="score", candidate_id=candidate, sample_files=samples
                    )
                )
        manifest = freeze_task_manifest(tasks, policies, runtime_identity, workers=2)
    forks_binding = prediction.get("forks_result") or stage.get("forks_result")
    if forks_binding is None:
        fit = _bound_json(prediction["fit_spec"])
        forks_binding = fit["forks"]
    return _publish_q5_observation_plan(
        config,
        bindings,
        stage=stage,
        manifest=manifest,
        origin_id=receipt["origin_id"],
        seed=seed,
        purpose="reference",
        operation=operation,
        prediction_binding=prediction_binding,
        forks_binding=forks_binding,
        out=out,
        extra_stage={
            "reference_precision_receipt": precision_receipt,
            "reference_batches": receipt["reference_batches"],
            "reference_extension_specs": specs,
            "no_method_rankings_used": True,
        },
    )


def build_q4_bridge_plan(config, bindings, out=None):
    """Freeze two historical development origins and actual old null action bytes.

    Only original train/control data are read. Q4 may reuse these opened control
    prompts for technical development; Q5's fresh fixed panel has its own selector.
    """
    audit = audit_v3_compatibility(config, bindings)
    parent = _bound_json(bindings["parent_validated_plan"])
    root = Path(parent["paths"]["parent_warm"])
    files = parent["warm_binding"]["files"]
    for name in ("bank_manifest.json", "samples.jsonl"):
        if file_hash(root / name) != files[name]:
            raise ValueError("Historical Q4 raw bank/sequence evidence changed")
    bank_manifest = _read(root / "bank_manifest.json")
    controls = bank_manifest["plan"]["control_prompts"]
    grouped = defaultdict(list)
    for prompt in controls:
        grouped[(prompt["family"], prompt["interface"])].append(prompt)
    if len(grouped) != 6 or any(len(v) < 2 for v in grouped.values()):
        raise ValueError("Q4 historical controls do not support 12 balanced probes")
    probes = [
        p
        for group in sorted(grouped)
        for p in sorted(
            grouped[group], key=lambda p: canonical_hash([NAMESPACE, "Q4_probe", p["prompt_id"]])
        )[:2]
    ]
    requested = {p["prompt_id"] for p in probes}
    raw = {}
    with (root / "samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("prompt_id") in requested and row.get("bank_role") == "control_proposal":
                key = row["prompt_id"]
                if key not in raw or row["sample_key"] < raw[key]["sample_key"]:
                    raw[key] = row
    if set(raw) != requested:
        raise ValueError("Q4 exact original null actions are incomplete")
    initial = parent["initial_checkpoint"]
    warm = parent["r4_binding"]["warm_checkpoint"]
    origins = [
        {
            "id": "historical_initial",
            "path": initial["path"],
            "sha256": initial["file_sha256"],
            "identity": initial["identity"],
            "state_hash": initial["state_hash"],
            "seed": 17,
            "arm": "X_BASE",
            "step": 0,
        },
        {
            "id": "historical_X_BASE_64",
            "path": warm["path"],
            "sha256": warm["file_sha256"],
            "identity": warm["identity"],
            "state_hash": warm["state_hash"],
            "seed": 17,
            "arm": "X_BASE",
            "step": 64,
        },
    ]
    for spec in origins:
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("Historical Q4 source checkpoint changed")
    null_actions = [
        {
            "prompt_id": p["prompt_id"],
            "token_ids": raw[p["prompt_id"]]["token_ids"],
            "sample_key": raw[p["prompt_id"]]["sample_key"],
            "source_record_hash": raw[p["prompt_id"]]["record_hash"],
        }
        for p in probes
    ]
    plan = {
        "schema": "modeling-v3-Q4-plan-v1",
        "config_hash": canonical_hash(config),
        "source_hash": audit["source_hash"],
        "origins": origins,
        "probes": probes,
        "null_actions": null_actions,
        "null_origin": origins[1],
        "workers": 2,
        "initial_draws": 64,
        "pilot_draws": 16,
        "main_draws": 48,
        "six_banks_per_origin": True,
        "historical_data_role": "development_only",
        "original_raw_binding": {
            "path": str(root / "samples.jsonl"),
            "sha256": files["samples.jsonl"],
        },
        "probability_tolerances": bindings["v3_probability_tolerances"],
        "tolerance_provenance": "predeclared V3 threshold; not yet measured on the new workers",
    }
    plan["plan_hash"] = canonical_hash(plan)
    if out is not None:
        _publish(Path(out) / "Q4_BRIDGE_PLAN.json", plan)
    return plan


def _bridge_legacy_checkpoint(runtime, spec):
    """Restore original bytes, then explicitly record newly captured forward state."""
    from ..followup_updates import capture_state
    from ..optimizer_fork import load_checkpoint as load_old
    from ..optimizer_fork import restore_state as restore_old
    from ..optimizer_fork import state_hash

    if file_hash(spec["path"]) != spec["sha256"]:
        raise ValueError("Historical checkpoint bytes changed before Q4 bridging")
    old = load_old(spec["path"], spec["identity"])
    if state_hash(old) != spec["state_hash"]:
        raise ValueError("Historical full recorded checkpoint hash changed")
    _restore_runtime(runtime, runtime["initial_state"])
    restore_old(runtime["adapter"].model, runtime["optimizer"], old)
    runtime["adapter"].model.eval()
    runtime["adapter"]._reset_positions()
    runtime["sampler"].update(
        step=spec["step"],
        position=4 * spec["step"],
        legacy_sampler_metadata=old["metadata"].get("sampler"),
    )
    state = capture_state(
        runtime["adapter"].model,
        runtime["optimizer"],
        {
            **old["metadata"],
            "seed": 17,
            "arm": spec["arm"],
            "checkpoint_step": spec["step"],
            "legacy_state_hash": spec["state_hash"],
            "V3_forward_bridge": (
                "certified eval/reset state captured now; not claimed present in old schema"
            ),
        },
        adapter=runtime["adapter"],
        sampler=runtime["sampler"],
    )
    return state


def vlm_smoke(
    config,
    bindings,
    *,
    device="cuda:0",
    out,
    allow_gpu=False,
    acknowledge_new_experiment=False,
    worker_index=0,
    resume=False,
):
    """Real Q4 worker: six natural banks, generated sequences, complete candidate scores.

    Worker 0 owns historical initial, worker 1 owns historical warm. Both score
    the exact same original warm null actions. A separate CPU merge verifies
    two-card null parity before publishing the gate consumed by Q5.
    """
    _authorization(
        allow_gpu=allow_gpu, acknowledge_new_experiment=acknowledge_new_experiment, device=device
    )
    if worker_index not in (0, 1):
        raise ValueError("Q4 worker_index must be zero or one")
    stage = _bound_json(bindings["v3_stage_lock"])
    confirmation = stage.get("first_gpu_confirmation", {})
    if (
        stage.get("operator_authorization") != "CONFIRMED_FIRST_GPU_SUBMISSION"
        or confirmation.get("scope") != "Q4"
        or not isinstance(confirmation.get("confirmation_text"), str)
        or not confirmation["confirmation_text"].strip()
    ):
        raise PermissionError("Q4 requires recorded first GPU authorization before runtime loading")
    handoff = confirmation["reviewed_handoff"]
    if file_hash(handoff["path"]) != handoff["sha256"]:
        raise ValueError("first GPU authorization command/timing handoff changed")
    _technical_gate(config, stage)
    if stage.get("phase") != "Q4" or "vlm-smoke" not in stage.get("operations", []):
        raise PermissionError("Q4 smoke must be in the frozen stage task list")
    plan = _bound_json(bindings["q4_bridge_plan"])
    if (
        plan["source_hash"] != canonical_hash(source_hashes())
        or plan["config_hash"] != canonical_hash(config)
        or plan["plan_hash"] != canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    ):
        raise ValueError("Q4 bridge plan changed after freeze")
    if stage.get("q4_plan_hash") != plan["plan_hash"]:
        raise PermissionError("Q4 bridge plan is outside the frozen stage task list")
    runtime = load_runtime(
        config,
        bindings,
        device=device,
        allow_gpu=allow_gpu,
        acknowledge_new_experiment=acknowledge_new_experiment,
    )
    from ..followup_runtime import _prepared
    from ..followup_updates import save_checkpoint
    from ..optimizer_fork import state_hash
    from .vlm_observation import (
        PATH,
        PrefixObservationBackend,
        execute_observation_tasks,
        freeze_observation_task,
        freeze_task_manifest,
    )

    root = Path(out)
    with frozen_writer(root):
        identity = {
            **runtime["identity"],
            "unit": "Q4_BRIDGE",
            "worker_index": worker_index,
            "Q4_plan_hash": plan["plan_hash"],
        }
        root, completed = _open_output(root, identity, resume)
        if completed is not None:
            return completed
        preflight = runtime_preflight(root)
        if not (root / "hardware.json").exists():
            _publish(root / "hardware.json", preflight)
        else:
            _publish(
                root / "hardware_invocations" / (canonical_hash(preflight) + ".json"), preflight
            )
        null_state = _bridge_legacy_checkpoint(runtime, plan["null_origin"])
        null_fingerprint = _normalized_inference_fingerprint(runtime, null_state)
        _restore_runtime(runtime, null_state)
        null_scores = []
        by_prompt = {p["prompt_id"]: p for p in plan["probes"]}
        for action in plan["null_actions"]:
            prompt = by_prompt[action["prompt_id"]]
            prepared = _prepared(runtime["adapter"], prompt, runtime["data_root"])
            scores = (
                runtime["adapter"]
                .logprobs(prepared, action["token_ids"], require_grad=False)
                .detach()
                .cpu()
                .double()
                .tolist()
            )
            null_scores.append(
                {
                    "proposal_sample_key": action["sample_key"],
                    "prompt_record_hash": canonical_hash(prompt),
                    "input_audit": prepared["audit"],
                    "token_ids": action["token_ids"],
                    "inference_fingerprint": null_fingerprint,
                    "probability_execution": PATH,
                    "max_new_tokens": 64,
                    "eos_token_ids": sorted(runtime["adapter"].eos_ids),
                    "token_logprobs": scores,
                    "sequence_logp": math.fsum(scores),
                }
            )
        gpu_uuid = runtime["allocated_gpu"]["uuid"]
        receipt = {
            "gpu_identity": gpu_uuid,
            "runtime_identity": runtime["identity"],
            "tolerances": runtime["parity_tolerances"],
            "scores": null_scores,
        }
        _publish(root / "null_receipt.json", receipt)
        spec = plan["origins"][worker_index]
        origin = _bridge_legacy_checkpoint(runtime, spec)
        checkpoint_identity = {
            **runtime["identity"],
            "unit": "V3_Q4_BRIDGED_ORIGIN",
            "historical_binding": spec,
        }
        if not (root / "origin.pt").exists():
            save_checkpoint(root / "origin.pt", origin, checkpoint_identity)
        else:
            from ..followup_updates import load_checkpoint

            saved = load_checkpoint(root / "origin.pt", checkpoint_identity)
            if state_hash(saved) != state_hash(origin):
                raise ValueError("Q4 bridged checkpoint differs on resume")
        bank_plan = build_bank_plan(
            runtime["parent"]["training_plan"]["train_prompts"],
            origin_identity={"state_hash": state_hash(origin), "legacy_id": spec["id"]},
        )
        bank_plan["banks"] = bank_plan["banks"][:6]
        bank_plan.update(generation_sequences=6 * 32, scratch_optimizer_updates=18)
        bank_plan["plan_hash"] = canonical_hash(
            {k: v for k, v in bank_plan.items() if k != "plan_hash"}
        )
        forks = run_candidate_banks(
            runtime, origin, bank_plan, out=root / "forks", resume=resume, bridge=True
        )
        policies = {
            "origin": {
                "checkpoint": {
                    "path": str(root / "origin.pt"),
                    "sha256": file_hash(root / "origin.pt"),
                    "identity": checkpoint_identity,
                },
                "inference_fingerprint": _normalized_inference_fingerprint(runtime, origin),
            }
        }
        for bank in forks["banks"]:
            for candidate, record in bank["checkpoints"].items():
                policies[bank["bank_id"] + "_" + candidate] = {
                    "checkpoint": {k: record[k] for k in ("path", "sha256", "identity")},
                    "inference_fingerprint": record["inference_fingerprint"],
                }
        _publish(root / "probes.json", plan["probes"])
        prompt_file = {"path": str(root / "probes.json"), "sha256": file_hash(root / "probes.json")}
        backend = PrefixObservationBackend(
            runtime["adapter"],
            runtime_identity=runtime["identity"],
            policy_loader=runtime["policy_loader"],
            data_root=runtime["data_root"],
            parity_tolerances=runtime["parity_tolerances"],
            state_guard=runtime["state_guard"],
        )
        generation_tasks = []
        first_bank = forks["banks"][0]["bank_id"]
        endpoints = [first_bank + "_joint_1", first_bank + "_joint_0"]
        requests = [
            ("pilot", "ORIGIN", ["origin"], 0, 16),
            ("main", "ORIGIN", ["origin"], 16, 64),
            ("reference", "ORIGIN", ["origin"], 0, 64),
            ("reference", "MIX", endpoints, 0, 64),
        ]
        requests += [("reference", "DIRECT", [candidate], 0, 64) for candidate in endpoints]
        for role, proposal, sources, start, stop in requests:
            generation_tasks.append(
                freeze_observation_task(
                    operation="generate",
                    origin_id=spec["id"],
                    candidate_id=sources[0],
                    prompt_file=prompt_file,
                    prompt_ids=list(by_prompt),
                    role=role,
                    draw_start=start,
                    draw_stop=stop,
                    rng_namespace=NAMESPACE + "_Q4",
                    proposal=proposal,
                    proposal_candidates=sources,
                    worker=0,
                )
            )
        generation_manifest = freeze_task_manifest(
            generation_tasks, policies, runtime["identity"], workers=1
        )
        _publish(root / "generation_tasks.json", generation_manifest)
        generated = execute_observation_tasks(
            generation_manifest,
            backend,
            out=root / "generation",
            worker_index=0,
            workers=1,
            resume=resume,
        )
        scoring_tasks = []
        for task in generation_tasks:
            if task["proposal"] == "DIRECT":
                # Direct event counts are a separate measurement baseline.
                continue
            task_root = root / "generation/worker_0" / task["output_path"]
            shards = [
                {"path": str(path), "sha256": file_hash(path)}
                for path in sorted(task_root.glob("attempt_*/samples.jsonl"))
            ]
            targets = list(policies) if task["proposal"] == "ORIGIN" else endpoints
            for target in targets:
                scoring_tasks.append(
                    freeze_observation_task(
                        operation="score",
                        origin_id=spec["id"],
                        candidate_id=target,
                        prompt_file=prompt_file,
                        prompt_ids=list(by_prompt),
                        role=task["role"],
                        draw_start=task["draw_start"],
                        draw_stop=task["draw_stop"],
                        rng_namespace=task["rng_namespace"],
                        proposal=task["proposal"],
                        proposal_candidates=task["proposal_candidates"],
                        sample_files=shards,
                        worker=0,
                    )
                )
        scoring_manifest = freeze_task_manifest(
            scoring_tasks, policies, runtime["identity"], workers=1
        )
        _publish(root / "scoring_tasks.json", scoring_manifest)
        scored = execute_observation_tasks(
            scoring_manifest,
            backend,
            out=root / "scoring",
            worker_index=0,
            workers=1,
            resume=resume,
        )
        from .vlm_observation import known_action_probability

        known_scores = []
        for candidate, policy in policies.items():
            backend.activate(policy)
            for prompt in plan["probes"]:
                text = json.dumps(prompt["scene"]["truth_world"], separators=(",", ":"))
                tokens = runtime["adapter"].processor.tokenizer.encode(
                    text, add_special_tokens=False
                )
                actions = [
                    [*tokens, eos]
                    for eos in sorted(runtime["adapter"].eos_ids)
                    if len(tokens) + 1 <= 64 and not set(tokens) & runtime["adapter"].eos_ids
                ]
                if not actions:
                    known_scores.append(
                        {
                            "candidate": candidate,
                            "prompt_id": prompt["prompt_id"],
                            "status": "NO_VALID_KNOWN_ACTION_WITHIN_HORIZON",
                        }
                    )
                    continue
                measured = backend.score_known_actions(prompt, actions, max_new_tokens=64)
                known_scores.append(
                    {
                        "candidate": candidate,
                        "prompt_id": prompt["prompt_id"],
                        "action_scores": measured,
                        "known_event_subset": known_action_probability(measured),
                        "exhaustive_event_enumeration_proved": False,
                    }
                )
        _publish(root / "known_event_scores.json", known_scores)
        from .vlm_observation import analyze_q4_worker

        observation = analyze_q4_worker(root, config)
        _restore_runtime(runtime, origin)
        result = {
            "status": "Q4_WORKER_COMPLETED_PENDING_TWO_GPU_NULL",
            "execution_kind": "REAL_CUDA_MODEL",
            "worker_index": worker_index,
            "config_hash": canonical_hash(config),
            "source_hash": canonical_hash(source_hashes()),
            "Q4_plan_hash": plan["plan_hash"],
            "origin": spec["id"],
            "banks": 6,
            "probes": 12,
            "initial_draws": 64,
            "generation": generated,
            "scoring": scored,
            "observations": observation,
            "null_receipt": {
                "path": str(root / "null_receipt.json"),
                "sha256": file_hash(root / "null_receipt.json"),
            },
            "independent_reference_draws_diagnostic_only": 64,
            "formal_reference_status": "REFERENCE_UNRESOLVED",
            "direct_count_baseline_endpoints": endpoints,
            "two_gpu_null_parity_passed": False,
            "scientific_status": "NOT_CERTIFIED",
        }
        _complete(root, result)
        return result


def finalize_q4_bridge(config, worker_bindings, *, out):
    """CPU merge verifies both immutable workers and paired null actions."""
    from ..r3_runtime import _verify_manifest
    from .vlm_observation import compare_cross_gpu_receipts

    workers = [_bound_json(binding) for binding in worker_bindings]
    if len(workers) != 2 or {r["worker_index"] for r in workers} != {0, 1}:
        raise ValueError("Q4 needs exactly the two distinct completed frozen workers")
    for worker, binding in zip(workers, worker_bindings, strict=True):
        if (
            worker["status"] != "Q4_WORKER_COMPLETED_PENDING_TWO_GPU_NULL"
            or worker["source_hash"] != canonical_hash(source_hashes())
            or worker["config_hash"] != canonical_hash(config)
        ):
            raise ValueError("Q4 worker source/config/completion mismatch")
        root = Path(binding["path"]).parent
        marker = _read(root / "completed.json")
        if marker["manifest_sha256"] != file_hash(root / "manifest.json"):
            raise ValueError("Q4 worker final manifest changed")
        _verify_manifest(root, _read(root / "manifest.json"))
        analysis = _bound_json(worker["observations"])
        if analysis.get("technical_chain_passed") is not True:
            raise ValueError(
                "Q4 finite observation/pilot/MIX/direct measurement chain remains unresolved"
            )
    if workers[0]["Q4_plan_hash"] != workers[1]["Q4_plan_hash"]:
        raise ValueError("Two Q4 workers used different frozen plans")
    receipts = [_bound_json(w["null_receipt"]) for w in workers]
    parity = compare_cross_gpu_receipts(*receipts, tolerances=receipts[0]["tolerances"])
    result = {
        "status": parity["status"],
        "execution_kind": "REAL_CUDA_MODEL",
        "source_hash": canonical_hash(source_hashes()),
        "config_hash": canonical_hash(config),
        "workers": worker_bindings,
        "two_gpu_null_parity_passed": parity["status"] == "PASS",
        "null_parity": parity,
        "observation_chain_scope": "two origins, six banks each, 12 probes, 64 initial draws",
        "observation_reference_precision": "REFERENCE_UNRESOLVED",
        "scientific_status": "NOT_CERTIFIED",
        "online_ssvc_authorized": False,
    }
    _publish(Path(out) / "Q4_BRIDGE_RECEIPT.json", result)
    return result


def _allocated_gpu_info():
    """Identify the actual process-visible GPU without guessing its physical index."""
    import torch

    properties = torch.cuda.get_device_properties(0)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        raise RuntimeError(
            "Runtime cannot expose the allocated GPU UUID; explicit binding required"
        )
    return {
        "uuid": str(uuid),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "visible_device": "cuda:0",
    }


def runtime_preflight(out):
    """Read allocated hardware/space only; no guessed PRO6000 capacity."""
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    usage = shutil.disk_usage(Path(out).resolve().parent)
    return {
        "nvidia_smi": gpu.stdout.strip(),
        "allocated_gpu": _allocated_gpu_info(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "disk_free_bytes": usage.free,
        "gpu_capacity_assumed": False,
    }
