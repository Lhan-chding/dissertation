"""Fail-closed real server backend for the frozen S1/S2 follow-up.

Preparation reads full parent evidence and local snapshot bytes without loading
model weights. Execution has an independent explicit CUDA boundary. Parent
production modules remain byte-for-byte unchanged; the new local-only loader is
qualified against their actual model certificate and a measured joint bridge.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
import time
from pathlib import Path, PurePosixPath

from .core import PROJECT_ROOT, canonical_hash, file_hash, write_json

DIRECTORY_KEYS = (
    "parent_warm",
    "parent_r4",
    "r0_dir",
    "r1_run",
    "supplement_dir",
    "r2_dir",
    "r3_cold_dir",
    "raw_dataset_root",
    "model_snapshot",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_files():
    return {
        str(p.relative_to(PROJECT_ROOT)): file_hash(p)
        for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))
    }


def _separate_output(output, parents):
    output = Path(output).resolve()
    for parent in parents:
        parent = Path(parent).resolve()
        if output.is_relative_to(parent) or parent.is_relative_to(output):
            raise ValueError("Follow-up output and immutable parent paths overlap")
    return output


def _resolved_paths(paths):
    paths = copy.deepcopy(paths)
    upstream = paths.pop("parent_r0_r1_r2", None)
    if isinstance(upstream, dict):
        paths = {**upstream, **paths}
    required = (*DIRECTORY_KEYS, "new_run_root")
    missing = [key for key in required if not isinstance(paths.get(key), str) or not paths[key]]
    if missing:
        raise ValueError("Unresolved server paths: " + ", ".join(missing))
    resolved = {key: str(Path(paths[key]).expanduser().resolve()) for key in required}
    absent = [key for key in DIRECTORY_KEYS if not Path(resolved[key]).is_dir()]
    if absent:
        raise ValueError(
            "BLOCKED_MISSING_PARENT_RAW: unavailable directories: " + ", ".join(absent)
        )
    _separate_output(resolved["new_run_root"], [resolved[k] for k in DIRECTORY_KEYS])
    return resolved


def _snapshot_binding(snapshot, revision):
    """Hash local-only snapshot, including every weight shard, before loading."""
    root = Path(snapshot).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision or "") or root.name != revision:
        raise ValueError("Local model snapshot must use the immutable revision directory name")
    required = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    )
    if any(not (root / name).is_file() for name in required):
        raise ValueError("Local model snapshot lacks required processor/tokenizer/config files")
    config = _read(root / "config.json")
    if config.get("_commit_hash", revision) != revision:
        raise ValueError("Snapshot config model revision differs")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = _read(index).get("weight_map", {})
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Local weight index is empty")
        shards = set(mapping.values())
        for name in shards:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
                raise ValueError("Local weight shard path escapes snapshot")
            if not (root / name).is_file() or (root / name).stat().st_size <= 0:
                raise ValueError("Local weight shard missing or empty")
    elif not (root / "model.safetensors").is_file():
        raise ValueError("Complete local safetensors weights are required; downloads forbidden")
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if any(name.endswith((".partial", ".incomplete")) for name in files):
        raise ValueError("Incomplete local model download cannot be loaded")
    return {
        "path": str(root),
        "revision": revision,
        "files": files,
        "model_weights_loaded": False,
        "network_download": False,
    }


def _source_bridge(parent_runtime):
    old = parent_runtime["source"]["source_files"]
    current = _source_files()
    changed = [name for name, digest in old.items() if current.get(name) != digest]
    if changed:
        raise ValueError(
            "Inherited production source changed; explicit new bridge review required: "
            + ", ".join(changed)
        )
    return {
        "status": "CPU_SOURCE_VERIFIED",
        "parent_source": parent_runtime["source"],
        "inherited_files_unchanged": len(old),
        "added_files": {name: digest for name, digest in current.items() if name not in old},
        "gpu_joint_bridge_required": True,
        "gpu_joint_bridge_measured": False,
    }


def _initial_evidence(parent_r4):
    from .optimizer_fork import load_checkpoint, state_hash
    from .r4_continuation import resolve_evidence

    evidence = resolve_evidence(parent_r4, "origin.pt")
    identity = {**evidence["runtime"]["identity"], "unit": "initial_origin"}
    state = load_checkpoint(evidence["path"], identity)
    if state["optimizer"]["state"] or state["metadata"].get("checkpoint_step") != 0:
        raise ValueError("Historical S2 origin must have step0 and empty Adam")
    return {
        "path": str(evidence["path"]),
        "identity": identity,
        "file_sha256": file_hash(evidence["path"]),
        "state_hash": state_hash(state),
    }


def _diagnostic_prompts(data_root):
    from .r2_inputs import variant_prompt
    from .r2_runtime import _load_panel

    panel, binding = _load_panel(data_root)
    prompts = []
    for scene in panel:
        for condition in ("SYM_ORIGINAL", "IMAGE_CUE", "IMAGE_ONLY"):
            prompt = variant_prompt(scene, condition)
            prompts.append(
                {
                    "prompt_id": canonical_hash(
                        {
                            "scene": scene["base_scene_id"],
                            "condition": condition,
                            "prompt_hash": prompt["prompt_hash"],
                        }
                    ),
                    "base_scene_id": scene["base_scene_id"],
                    "family": scene["constraint_family"],
                    "interface": "SYMBOLIC_FRESH"
                    if condition == "SYM_ORIGINAL"
                    else "IMAGE_CUE_FRESH",
                    "track": "R2",
                    "split": "calibration",
                    "condition": condition,
                    "diagnostic_condition": condition,
                    "max_new_tokens": 64,
                    "enable_thinking": False,
                    "prompt_hash": prompt["prompt_hash"],
                    "scene_hash": canonical_hash(scene),
                    "scene": scene,
                    "prompt": prompt,
                }
            )
    if len(prompts) != 216:
        raise ValueError("Follow-up interface panel requires 72 scenes times 3 conditions")
    return prompts, binding


def prepare_server_plan(design, paths, *, out=None):
    """Recursively verify actual parent raw/checkpoints, data and source on CPU.

    Existing audit helpers load only already-cached pinned CPU processors. They
    never download models. A compact metadata ZIP cannot satisfy this entry.
    """
    from .followup_protocol import validate_design

    design = copy.deepcopy(design)
    validate_design(design)
    paths = _resolved_paths(paths)
    from .followup_inputs import build_followup_bank
    from .followup_parent import validate_parent_metadata
    from .next_stage_runtime import (
        validate_prerequisites,
        validate_r2_gate,
        validate_r3_cold_gate,
        validate_r4_gate,
    )
    from .r3_gate import _ledger
    from .r3_warm_gate import validate_r3_warm_gate
    from .r4_continuation import policy_for_name
    from .r4_runtime import _load_plan

    snapshot = _snapshot_binding(paths["model_snapshot"], design["model"]["revision"])
    warm = Path(paths["parent_warm"])
    required = (
        "runtime_lock.json",
        "bank_manifest.json",
        "candidate_manifest.json",
        "samples.jsonl",
        "origin.pt",
    )
    if any(not (warm / name).is_file() for name in required):
        raise ValueError("BLOCKED_MISSING_PARENT_RAW: complete warm raw/checkpoints required")
    runtime, parent_bank, candidates = [_read(warm / name) for name in required[:3]]
    metadata = validate_parent_metadata(runtime, parent_bank, candidates)
    source_bridge = _source_bridge(runtime)
    gate = validate_prerequisites(paths["r0_dir"], paths["r1_run"], paths["supplement_dir"])
    if Path(gate["config"]["data_root"]).resolve() != Path(paths["raw_dataset_root"]):
        raise ValueError("Actual dataset path differs from the certified runtime data root")
    r2 = validate_r2_gate(paths["r2_dir"], gate)
    cold = validate_r3_cold_gate(paths["r3_cold_dir"], gate, r2)
    r4 = validate_r4_gate(paths["parent_r4"], gate, r2, cold)
    warm_gate = validate_r3_warm_gate(warm, gate, r2, cold, r4)
    rows = _ledger(warm / "samples.jsonl")
    units = [
        build_followup_bank(
            parent_bank, rows, spec, candidate_manifest=candidates, runtime_lock=runtime
        )
        for spec in design["S1"]["units"]
    ]
    training, training_data, legacy_lock = _load_plan(
        paths["raw_dataset_root"], gate, paths["r0_dir"]
    )
    initial = _initial_evidence(paths["parent_r4"])
    diagnostic, diagnostic_binding = _diagnostic_prompts(paths["raw_dataset_root"])
    decision = Path(paths["parent_r4"]) / "continuation_decision.json"
    if not decision.is_file():
        raise ValueError("Historical reviewed warning policy is required before preparing S2")
    reviewed = _read(decision)
    # Parent recursive audit already validates this decision and its hash.
    policy_name = reviewed.get("reviewed_warning_policy")
    if policy_name is None:
        policy_name = reviewed.get("policy", {}).get("reviewed_warning_policy")
    warning_policy = policy_for_name(policy_name)
    sources = _source_files()
    identity = {
        "model_hash": canonical_hash(gate["config"]["model"]),
        "data_hash": canonical_hash(
            {"warm": parent_bank["data_binding"], "training": training_data}
        ),
        "config_hash": canonical_hash(design),
        "protocol_version": design["protocol_version"],
        "parser_version_hash": canonical_hash(
            {
                name: sources[name]
                for name in ("src/verifiers.py", "src/legacy_frozen.py", "src/r2_runtime.py")
            }
        ),
    }
    result = {
        "schema_version": 1,
        "status": "CPU_PARENT_RAW_VERIFIED_REQUIRES_GPU_SMOKE",
        "design": design,
        "design_hash": canonical_hash(design),
        "paths": paths,
        "source_files": sources,
        "source_bridge": source_bridge,
        "snapshot": snapshot,
        "gate": gate,
        "r2_binding": r2,
        "r3_binding": cold,
        "r4_binding": r4,
        "warm_binding": warm_gate,
        "parent_metadata": metadata,
        "initial_checkpoint": initial,
        "parent_unit_bindings": [
            {k: v for k, v in unit.items() if k != "groups"} for unit in units
        ],
        "training_plan": training,
        "legacy_lock": legacy_lock,
        "training_data_binding": training_data,
        "diagnostic_prompts": diagnostic,
        "diagnostic_binding": diagnostic_binding,
        "warning_policy": warning_policy,
        "warning_decision_sha256": file_hash(decision),
        "identity": identity,
        "config": gate["config"],
        "gates": {
            "parent_raw_verified": True,
            "gpu_smoke_passed": False,
            "training_authorized": False,
            "s1_measurement_passed": False,
        },
        "model_weights_loaded": False,
        "gpu_started": False,
        "training_started": False,
    }
    if out is not None:
        path = Path(out)
        target = path / "validated_plan.json" if path.suffix != ".json" else path
        _separate_output(target.parent, [paths[k] for k in DIRECTORY_KEYS])
        write_json(target, result)
    return result


ROUNDOFF_OBSERVER_PATHS = (
    ("warm_binding", "response_roundoff"),
    ("warm_binding", "direct_validation", "roundoff"),
)


def _validate_roundoff_difference(value, *, direct=False):
    """Check the existing observer format/bound before removing it from identity."""
    import sys

    from .r3_report_gate import _derived_float

    fields = {"path", "stored", "recomputed", "absolute_difference", "allowed_bound"}
    if not isinstance(value, dict) or set(value) != fields or not isinstance(value["path"], list):
        raise ValueError("Malformed roundoff observer difference")
    path = tuple(value["path"])
    if direct:
        permitted = path[:1] == ("comparisons",) and (
            (len(path) == 6 and path[-1] in {"predicted_delta", "observed_delta", "residual"})
            or (
                len(path) == 8
                and path[5] == "ci"
                and path[-1]
                in {
                    "low",
                    "high",
                    "half_width",
                    "half_width_to_abs_point_estimate",
                    "half_width_to_abs_predicted_delta",
                }
            )
        )
    else:
        permitted = _derived_float(path)
    if not permitted or any(
        type(value[key]) is not float or not math.isfinite(value[key]) for key in fields - {"path"}
    ):
        raise ValueError("Nonfinite or unsupported roundoff observer")
    stored, recomputed = value["stored"], value["recomputed"]
    bound = 64 * sys.float_info.epsilon * max(1.0, abs(stored), abs(recomputed))
    difference = abs(stored - recomputed)
    if (
        value["allowed_bound"] != bound
        or value["absolute_difference"] != difference
        or difference > bound
    ):
        raise ValueError("Roundoff observer differs from the existing numerical bound")


def _validate_response_roundoff(summary, *, bank_index=False):
    fields = {"different_float_count", "maximum_absolute_difference", "differences"}
    if bank_index:
        fields.add("bank_index")
    if not isinstance(summary, dict) or set(summary) != fields:
        raise ValueError("Malformed response roundoff observer")
    differences = summary["differences"]
    if not isinstance(differences, list):
        raise ValueError("Roundoff observer differences must be a list")
    for value in differences:
        _validate_roundoff_difference(value)
    if (
        type(summary["different_float_count"]) is not int
        or summary["different_float_count"] != len(differences)
        or type(summary["maximum_absolute_difference"]) is not float
        or summary["maximum_absolute_difference"]
        != max((value["absolute_difference"] for value in differences), default=0.0)
    ):
        raise ValueError("Roundoff observer summary does not describe its differences")


def _revalidation_identity(plan):
    """Exclude two exact observation subtrees, retaining all actual evidence gates.

    r3_report_gate explicitly keeps accepted CPU-kernel roundoff observations
    outside its stable identity. The warm audit additionally exposes the same
    observations for direct validation. Neither records new scientific results.
    """
    projected = copy.deepcopy(plan)
    warm = projected.get("warm_binding", {})
    if "response_roundoff" in warm:
        responses = warm["response_roundoff"]
        if (
            not isinstance(responses, list)
            or len(responses) != 2
            or any(
                not isinstance(row, dict) or type(row.get("bank_index")) is not int
                for row in responses
            )
            or {row["bank_index"] for row in responses} != {0, 6}
        ):
            raise ValueError("Warm response roundoff requires banks 0 and 6")
        for row in responses:
            _validate_response_roundoff(row, bank_index=True)
        warm["response_roundoff"] = {"identity_scope": "CHECKED_CPU_ROUNDOFF_OBSERVER_ONLY"}
    direct = warm.get("direct_validation", {})
    if "roundoff" in direct:
        observations = direct["roundoff"]
        if not isinstance(observations, dict) or set(observations) != {"0", "6"}:
            raise ValueError("Direct roundoff observer requires banks 0 and 6")
        for row in observations.values():
            if not isinstance(row, dict) or set(row) != {"control_IS", "differences"}:
                raise ValueError("Malformed direct roundoff observer")
            _validate_response_roundoff(row["control_IS"])
            if not isinstance(row["differences"], list):
                raise ValueError("Direct roundoff differences must be a list")
            for value in row["differences"]:
                _validate_roundoff_difference(value, direct=True)
        direct["roundoff"] = {"identity_scope": "CHECKED_CPU_ROUNDOFF_OBSERVER_ONLY"}
    return projected


def _publish_revalidation_json(path, value):
    """Concurrent identical publication is idempotent; different bytes fail closed."""
    import hashlib
    import os
    import stat

    from .followup_protocol import write_new_json

    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    try:
        write_new_json(path, value)
    except FileExistsError:
        # The publisher checks every ancestor. O_NOFOLLOW also closes the final
        # component race when another process won the immutable publication.
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Revalidation audit may not traverse symbolic links") from None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or stream.read() != encoded:
                raise ValueError("Existing revalidation audit has different bytes") from None
    return hashlib.sha256(encoded).hexdigest()


def _record_revalidation(original, recomputed, original_identity, recomputed_identity):
    """Save both full plans and a non-identity, immutable comparison receipt."""
    import os
    import socket

    from .followup_train import _safe_output

    paths = original["paths"]
    root = _safe_output(Path(paths["new_run_root"]) / "revalidation_audits")
    _separate_output(root, [paths[name] for name in DIRECTORY_KEYS if name in paths])
    records = {}
    for name, plan in (("original", original), ("recomputed", recomputed)):
        digest = canonical_hash(plan)
        relative = f"plans/{digest}.json"
        file_digest = _publish_revalidation_json(root / relative, plan)
        records[name] = {"path": relative, "canonical_sha256": digest, "file_sha256": file_digest}
    same = original_identity == recomputed_identity
    receipt = {
        "schema_version": 1,
        "status": "PASS" if same else "FAIL_IDENTITY_CHANGED",
        "original": records["original"],
        "recomputed": records["recomputed"],
        "original_identity_hash": original_identity,
        "recomputed_identity_hash": recomputed_identity,
        "observer_only_paths": [list(path) for path in ROUNDOFF_OBSERVER_PATHS],
        "original_numerical_gates_reexecuted": True,
        "numerical_thresholds_relaxed": False,
        "runtime_observation": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _publish_revalidation_json(root / "comparisons" / f"{canonical_hash(receipt)}.json", receipt)


def _revalidate_plan(validated_plan):
    # This reruns every original numerical, source, data and checkpoint gate.
    # A tolerance failure propagates; it never becomes an ignored observation.
    rebuilt = prepare_server_plan(validated_plan["design"], validated_plan["paths"])
    original_identity = canonical_hash(_revalidation_identity(validated_plan))
    recomputed_identity = canonical_hash(_revalidation_identity(rebuilt))
    _record_revalidation(validated_plan, rebuilt, original_identity, recomputed_identity)
    if original_identity != recomputed_identity:
        raise ValueError("Validated plan/source/parent/data/snapshot changed; prepare again")
    # Downstream/CLI whole-plan hashes must continue to identify the exact frozen
    # original file. Current recomputation remains complete in the audit receipt.
    return copy.deepcopy(validated_plan)


def _load_local_adapter(snapshot, model_spec, *, image_token_limit=768):
    """Construct the original adapter using a local snapshot and no Hub resolver.

    This deliberately keeps the certified old loader unchanged. Each load is
    subsequently checked with its original actual model/processor certificate.
    """
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    from .model_adapters.base import _hash_json, select_language_mlp_modules
    from .model_adapters.qwen35 import Qwen35Adapter
    from .optimizer_fork import parameter_hash

    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA device is required before model loading")
    root = Path(snapshot["path"])
    if _snapshot_binding(root, model_spec["revision"]) != snapshot:
        raise ValueError("Local model snapshot changed before loading")
    model_cls = getattr(transformers, Qwen35Adapter.model_class, None)
    if model_cls is None:
        raise RuntimeError("Pinned Transformers lacks the original Qwen3.5 model class")
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    options = {"local_files_only": True, "trust_remote_code": False}
    model = model_cls.from_pretrained(
        str(root), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager", **options
    )
    processor = transformers.AutoProcessor.from_pretrained(str(root), **options)
    ip = processor.image_processor
    patch = int(getattr(ip, "patch_size", 16))
    merge = int(getattr(ip, "merge_size", getattr(ip, "spatial_merge_size", 2)))
    pixels = image_token_limit * (patch * merge) ** 2
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = pixels
    if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
        ip.size = {**ip.size, "longest_edge": pixels}
    elif dataclasses.is_dataclass(getattr(ip, "size", None)):
        ip.size = dataclasses.replace(ip.size, longest_edge=pixels)
    names = select_language_mlp_modules((name for name, _ in model.named_modules()), 32)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            target_modules=names,
            task_type="CAUSAL_LM",
        ),
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name or not any(target in name for target in names):
                raise RuntimeError("Unexpected trainable parameter in local adapter")
            parameter.data = parameter.data.float()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.eval()
    adapter = Qwen35Adapter(model, processor, model_spec["id"], model_spec["revision"])
    adapter._install_hooks()
    torch.cuda.synchronize()
    adapter.audit = {
        "execution_kind": "REAL_CUDA_INFERENCE",
        "model_id": model_spec["id"],
        "model_revision": model_spec["revision"],
        "model_class": Qwen35Adapter.model_class,
        "transformers_version": transformers.__version__,
        "base_dtype": "bfloat16",
        "probability_execution": "uncached_prefix_recompute",
        "eos_token_ids": sorted(adapter.eos_ids),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "lora_modules": names,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
        "model_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "load_seconds": time.perf_counter() - start,
        "load_peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "requested_image_token_limit": image_token_limit,
        "processor_max_pixels": pixels,
        "processor_patch_size": patch,
        "processor_merge_size": merge,
        "processor_hash": _hash_json(processor.to_dict()),
        "tokenizer_hash": _hash_json(processor.tokenizer.get_vocab()),
        "chat_template_hash": _hash_json(processor.chat_template),
        "frozen_parameter_hash": parameter_hash(model, trainable=False),
    }
    return adapter


def _seed_initial(adapter, optimizer, validated, seed):
    from .followup_updates import capture_state
    from .optimizer_fork import load_checkpoint, state_hash
    from .smoke_runtime import _seed_everything

    if seed == 17:
        binding = validated["initial_checkpoint"]
        if file_hash(binding["path"]) != binding["file_sha256"]:
            raise ValueError("Historical seed17 initial checkpoint bytes changed")
        state = load_checkpoint(binding["path"], binding["identity"])
        if state_hash(state) != binding["state_hash"]:
            raise ValueError("Historical seed17 initial state changed")
        from .r3_runtime import _restore

        _restore(adapter, optimizer, state)
        return capture_state(adapter.model, optimizer, state["metadata"], adapter=adapter)
    _seed_everything(seed)
    # Exact default linear-LoRA initialization, with original base weights fixed.
    import torch

    with torch.no_grad():
        for name, parameter in adapter.model.named_parameters():
            if parameter.requires_grad:
                if "lora_A" in name:
                    torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                elif "lora_B" in name:
                    parameter.zero_()
                else:
                    raise ValueError("Unsupported trainable module for prospective LoRA seed")
    return capture_state(
        adapter.model,
        optimizer,
        {
            "checkpoint_step": 0,
            "arm": "INITIAL",
            "seed": seed,
            "completed_sample_keys": [],
            "sampler": {"position": 0},
            "scheduler": None,
            "grad_scaler": None,
        },
        adapter=adapter,
    )


def _control_prompt_bindings(prompts, records):
    """Retain the complete original fixed-control processor input identities."""
    rows = list(records.values()) if isinstance(records, dict) else list(records)
    bound = []
    keys = (
        "final_prompt_hash",
        "tokenized_prompt_hash",
        "input_tensor_hash",
        "pixel_values_hash",
        "prepared_hash",
    )
    for prompt in prompts:
        selected = [
            r
            for r in rows
            if r.get("bank_role") == "control_proposal"
            and r.get("prompt_id") == prompt["prompt_id"]
        ]
        if len(selected) != 16 or {r.get("sample_index") for r in selected} != set(range(16)):
            raise ValueError("Original control prompt requires exactly sixteen raw input bindings")
        binding = {key: selected[0].get(key) for key in keys}
        if any(
            not isinstance(binding[key], str) or not re.fullmatch(r"[0-9a-f]{64}", binding[key])
            for key in keys
            if key != "pixel_values_hash"
        ):
            raise ValueError("Original control prepared input hashes are missing")
        if any({key: row.get(key) for key in keys} != binding for row in selected):
            raise ValueError("Original control rows disagree on prepared input hashes")
        bound.append({**copy.deepcopy(prompt), "prepared_binding": binding})
    return bound


def load_server_context(
    validated_plan, *, stage="S1", bank=None, seed=None, arm=None, allow_gpu_execution=False
):
    if allow_gpu_execution is not True:
        raise PermissionError("Real execution requires explicit GPU authorization")
    if stage not in ("S1", "S2"):
        raise ValueError("Unknown follow-up stage")
    if stage == "S2" and (seed, arm) not in {
        (r["seed"], r["arm"]) for r in validated_plan["design"]["S2"]["new_runs"]
    }:
        raise ValueError("S2 historical anchors or unplanned seed/arm cannot be run")
    validated = _revalidate_plan(validated_plan)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Follow-up requires an allocated CUDA device; no model requested")
    from .followup_updates import capture_state
    from .next_stage_runtime import validate_runtime_environment
    from .optimizer_fork import state_hash
    from .r1_reference_smoke import _certificate_check
    from .smoke_runtime import _seed_everything

    environment = validate_runtime_environment(validated["gate"])
    if environment != validated["r4_binding"]["environment"]:
        raise ValueError("Follow-up runtime differs from verified parent lock")
    _seed_everything(17)
    adapter = _load_local_adapter(validated["snapshot"], validated["config"]["model"])
    _certificate_check(validated["gate"]["certificate"], adapter, validated["config"])
    if validate_runtime_environment(validated["gate"], adapter.audit) != environment:
        raise ValueError("Loaded runtime differs from the verified environment")
    options = validated["config"]["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=options["learning_rate"],
        betas=tuple(options["betas"]),
        eps=options["eps"],
        weight_decay=options["weight_decay"],
    )
    paths = validated["paths"]
    base = {
        "design": validated["design"],
        "design_hash": validated["design_hash"],
        "identity": validated["identity"],
        "config": validated["config"],
        "execution_kind": "REAL_CUDA_FOLLOWUP",
        "source_files": validated["source_files"],
        "execution_identity": {
            "source_hash": canonical_hash(validated["source_files"]),
            "validated_plan_hash": canonical_hash(validated),
        },
        "gates": copy.deepcopy(validated["gates"]),
    }
    if stage == "S1":
        from .followup_inputs import build_followup_bank
        from .r3_gate import _ledger
        from .r3_runtime import _prepared
        from .r3_warm_runtime import load_warm_origin

        old_origin = load_warm_origin(
            adapter, optimizer, {"checkpoint": validated["r4_binding"]["warm_checkpoint"]}
        )
        origin = capture_state(adapter.model, optimizer, old_origin["metadata"], adapter=adapter)
        root = Path(paths["parent_warm"])
        parent_bank, candidates, runtime = [
            _read(root / name)
            for name in ("bank_manifest.json", "candidate_manifest.json", "runtime_lock.json")
        ]
        rows = _ledger(root / "samples.jsonl")
        specs = validated["design"]["S1"]["units"]
        if bank is not None:
            specs = [spec for spec in specs if spec["id"] == bank]
            if len(specs) != 1:
                raise ValueError("Unknown S1 bank")
        prepared = {
            p["prompt_id"]: _prepared(adapter, p, paths["raw_dataset_root"])
            for p in parent_bank["plan"]["train_prompts"]
        }
        units = [
            build_followup_bank(
                parent_bank,
                rows,
                spec,
                candidate_manifest=candidates,
                runtime_lock=runtime,
                prepared_inputs=prepared,
            )
            for spec in specs
        ]
        plan = {
            **base,
            "units": units,
            "control_prompts": _control_prompt_bindings(
                parent_bank["plan"]["control_prompts"], rows
            ),
            "warm_origin_state_hash": state_hash(old_origin),
        }
        return {
            "plan": plan,
            "adapter": adapter,
            "optimizer": optimizer,
            "origin": origin,
            "data_root": paths["raw_dataset_root"],
        }
    initial = _seed_initial(adapter, optimizer, validated, seed)
    from .followup_inputs import build_training_schedule

    train = validated["training_plan"]
    schedule = build_training_schedule(
        [p["scene"] for p in train["train_prompts"]], seed=seed, data_root=paths["raw_dataset_root"]
    )
    if seed == 17 and schedule["train_steps"] != train["train_steps"]:
        raise ValueError("Historical seed17 prompt sequence differs from original R4")
    cold_root = Path(paths["r3_cold_dir"])
    cold_rows = [
        _read_line
        for _read_line in (
            json.loads(line)
            for line in (cold_root / "samples.jsonl").read_text().splitlines()
            if line.strip()
        )
    ]
    probe = [
        r for r in cold_rows if r["bank_role"] == "control_proposal" and r["sample_index"] == 0
    ]
    control = _control_prompt_bindings(
        _read(cold_root / "bank_manifest.json")["plan"]["control_prompts"], cold_rows
    )
    if len(probe) != 48:
        raise ValueError("Original fixed control probe must contain 48 raw rows")
    plan = {
        **train,
        **base,
        **schedule,
        "legacy_lock": validated["legacy_lock"],
        "diagnostic_prompts": validated["diagnostic_prompts"],
        "control_probe_rows": probe,
        "control_prompts": control,
        "warning_policy": validated["warning_policy"],
    }
    return {
        "plan": plan,
        "adapter": adapter,
        "optimizer": optimizer,
        "initial_state": initial,
        "data_root": paths["raw_dataset_root"],
    }


def _measure_joint_bridge(adapter, optimizer, origin, groups):
    """Measured CPU/GPU algorithm bridge; this function grants no execution gate."""
    from .followup_updates import capture_state, fork_one_candidate, restore_state
    from .fork_gradients import apply_gradient_update, direct_loss_gradients
    from .optimizer_fork import parameter_hash, state_hash
    from .r3_updates import _compare_states

    frozen = parameter_hash(adapter.model, trainable=False)
    checks = {}
    try:
        for lam in (0.0, 1.0):
            restore_state(adapter.model, optimizer, origin, adapter=adapter)
            gradients = direct_loss_gradients(adapter, groups, arm="X_VALID", auxiliary_weight=lam)
            apply_gradient_update(adapter.model, optimizer, gradients["gradients"])
            specification = {"id": f"joint_{lam}", "policy": "joint", "auxiliary_weight": lam}
            candidate_metadata = {
                **origin["metadata"],
                "candidate_spec": specification,
                "permanent_training_commit": False,
            }
            legacy = capture_state(adapter.model, optimizer, candidate_metadata, adapter=adapter)
            restore_state(adapter.model, optimizer, origin, adapter=adapter)
            new = fork_one_candidate(adapter, optimizer, origin, groups, specification)
            comparison = _compare_states(legacy, new["state"])
            if comparison["passed"] is not True:
                raise ValueError("Measured GPU old/new joint Adam bridge failed")
            checks[str(lam)] = {
                "passed": True,
                "comparison": comparison,
                "legacy_parameter_hash": state_hash(legacy["parameters"]),
                "new_parameter_hash": state_hash(new["state"]["parameters"]),
                "legacy_optimizer_hash": state_hash(legacy["optimizer"]),
                "new_optimizer_hash": state_hash(new["state"]["optimizer"]),
            }
        if parameter_hash(adapter.model, trainable=False) != frozen:
            raise ValueError("GPU smoke changed frozen base model")
    finally:
        restore_state(adapter.model, optimizer, origin, adapter=adapter)
        if state_hash(
            capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
        ) != state_hash(origin):
            raise RuntimeError("GPU smoke failed exact full-state restoration")
    return {
        "checks": checks,
        "origin_hash": state_hash(origin),
        "full_origin_restored": True,
        "frozen_base_unchanged": True,
    }


def smoke_server(validated_plan, *, output_root, allow_gpu_execution=False):
    """Measure old/new joint Adam steps and exact restoration on actual bank03."""
    if allow_gpu_execution is not True:
        raise PermissionError("Server smoke requires explicit GPU authorization")
    output = _separate_output(output_root, [validated_plan["paths"][k] for k in DIRECTORY_KEYS])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke evidence requires a fresh output directory")
    context = load_server_context(
        validated_plan, stage="S1", bank="bank03", allow_gpu_execution=True
    )
    import torch

    adapter, optimizer, origin = context["adapter"], context["optimizer"], context["origin"]
    if not torch.cuda.is_available() or not all(p.is_cuda for p in adapter.model.parameters()):
        raise ValueError("Real smoke evidence requires actual CUDA model tensors")
    measured = _measure_joint_bridge(
        adapter, optimizer, origin, context["plan"]["units"][0]["groups"]
    )
    evidence = {
        "status": "PASS",
        "execution_kind": "REAL_CUDA_FOLLOWUP_SMOKE",
        "plan_hash": canonical_hash(validated_plan),
        "design_hash": validated_plan["design_hash"],
        "source_files": validated_plan["source_files"],
        "gpu_smoke_passed": True,
        "gpu_started": True,
        "training_started": False,
        "smoke_optimizer_updates": 4,
        "new_rollouts": 0,
        "bank_id": "bank03",
        "checks": measured["checks"],
        "full_origin_restored": True,
        "frozen_base_unchanged": True,
        "warm_origin_hash": measured["origin_hash"],
        "model_audit": adapter.audit,
        "safety_status": "NOT_CERTIFIED",
    }
    write_json(output / "smoke_evidence.json", evidence)
    return evidence


def _load_historical_context(validated_plan, arm):
    """Load an audited existing R4 endpoint; never construct a training update."""
    from .followup_updates import capture_state
    from .optimizer_fork import load_checkpoint, parameter_hash, state_hash
    from .r3_runtime import _restore
    from .r3_warm_gate import _check_mature
    from .r4_continuation import resolve_evidence
    from .r4_gate import _path

    context = load_server_context(
        validated_plan, stage="S1", bank="bank00", allow_gpu_execution=True
    )
    root = Path(validated_plan["paths"]["parent_r4"])
    files = validated_plan["r4_binding"]["files"]
    manifest_path = root / "checkpoint_manifest.json"
    if file_hash(manifest_path) != files["checkpoint_manifest.json"]:
        raise ValueError("Historical checkpoint manifest changed")
    candidates = [
        entry
        for entry in _read(manifest_path)["checkpoints"]
        if entry.get("arm") == arm and entry.get("step") == 64
    ]
    if len(candidates) != 1:
        raise ValueError("Historical endpoint requires one original arm step64 checkpoint")
    entry = candidates[0]
    path = _path(root, files, entry["checkpoint_path"])
    evidence = resolve_evidence(root, str(path.relative_to(root)))
    digest = file_hash(evidence["path"])
    if digest != files[str(path.relative_to(root))] or digest != entry["checkpoint_sha256"]:
        raise ValueError("Historical endpoint checkpoint bytes changed")
    state = load_checkpoint(evidence["path"], entry["checkpoint_identity"])
    _check_mature(state, step=64)
    if (
        state_hash(state) != entry["state_hash"]
        or state_hash(state["optimizer"]) != entry["optimizer_state_hash"]
        or state["metadata"].get("arm") != arm
        or state["metadata"].get("checkpoint_step") != 64
        or state["metadata"].get("sampler", {}).get("position") != 64
        or len(state["metadata"].get("completed_sample_keys", [])) != 2048
        or len(set(state["metadata"].get("completed_sample_keys", []))) != 2048
    ):
        raise ValueError("Historical endpoint state/Adam/sampler differs from verified R4")
    adapter, optimizer = context["adapter"], context["optimizer"]
    _restore(adapter, optimizer, state)
    if parameter_hash(adapter.model, trainable=True) != entry["parameter_hash"]:
        raise ValueError("Loaded historical endpoint parameters differ from R4")
    return {
        "adapter": adapter,
        "optimizer": optimizer,
        "origin": capture_state(adapter.model, optimizer, state["metadata"], adapter=adapter),
        "data_root": context["data_root"],
        "endpoint": {
            "arm": arm,
            "seed": 17,
            "step": 64,
            "path": str(path),
            "file_sha256": digest,
            "state_hash": entry["state_hash"],
            "checkpoint_identity": entry["checkpoint_identity"],
            "parameter_hash": entry["parameter_hash"],
            "optimizer_state_hash": entry["optimizer_state_hash"],
        },
    }


def _run_historical_r2(context, validated, *, arm, output_root, resume=False, fixture=False):
    """Read-only R2 endpoint measurement; fixture outputs cannot grant GPU gates."""
    from collections import Counter

    import torch

    from .core import frozen_writer
    from .followup_runtime import collect_samples
    from .followup_train import (
        _completion_manifest,
        _policy_hash,
        _publish,
        _safe_output,
        build_training_requests,
        isolated_evaluation,
    )
    from .followup_updates import capture_state
    from .optimizer_fork import state_hash
    from .r3_runtime import _verify_manifest

    adapter, optimizer, origin = context["adapter"], context["optimizer"], context["origin"]
    if arm not in ("X_BASE", "X_VALID") or context["endpoint"]["arm"] != arm:
        raise ValueError("Only original historical endpoint arms may be measured")
    if fixture:
        if getattr(adapter, "execution_kind", None) != "CPU_FAKE_TORCH" or any(
            parameter.device.type != "cpu" for parameter in adapter.model.parameters()
        ):
            raise ValueError("Historical fixture requires explicitly fake adapter and CPU tensors")
        kind = "CPU_FAKE_TORCH"
    else:
        if not torch.cuda.is_available() or not all(p.is_cuda for p in adapter.model.parameters()):
            raise ValueError("Historical endpoint inference requires actual CUDA model tensors")
        kind = "REAL_CUDA_HISTORICAL_R2"
    prompts = validated["diagnostic_prompts"]
    conditions = ("SYM_ORIGINAL", "IMAGE_CUE", "IMAGE_ONLY")
    if (
        len(prompts) != 216
        or len({p["prompt_id"] for p in prompts}) != 216
        or len({p["base_scene_id"] for p in prompts}) != 72
        or len({(p["base_scene_id"], p["diagnostic_condition"]) for p in prompts}) != 216
        or Counter(p["diagnostic_condition"] for p in prompts) != dict.fromkeys(conditions, 72)
        or any(p["max_new_tokens"] != 64 for p in prompts)
    ):
        raise ValueError("Historical R2 must contain exactly 216 fixed diagnostic prompts")
    semantic = {
        **validated["identity"],
        "phase": "S2_HISTORICAL_R2",
        "seed": 17,
        "endpoint_state_hash": context["endpoint"]["state_hash"],
    }
    identity = {
        **semantic,
        "arm": arm,
        "execution_kind": kind,
        "design_hash": validated["design_hash"],
        "validated_plan_hash": canonical_hash(validated),
        "source_hash": canonical_hash(validated["source_files"]),
        "endpoint_binding": context["endpoint"],
        "origin_state_hash": state_hash(origin),
    }
    requests = build_training_requests(
        prompts,
        semantic,
        17,
        arm,
        64,
        "historical_endpoint/R2",
        _policy_hash(origin),
        samples_per_prompt=4,
    )
    if len(requests) != 864:
        raise ValueError("Historical endpoint request budget must be 864")
    out = _safe_output(output_root)
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError("Historical endpoint output exists; explicit resume required")
    with frozen_writer(out):
        _publish(out / "identity.json", identity)
        completed = out / "completed.json"
        if completed.exists():
            manifest = _read(out / "completion_manifest.json")
            binding = _read(out / "completion_binding.json")
            _verify_manifest(out, manifest)
            if manifest != _completion_manifest(out) or binding != {
                "identity_hash": canonical_hash(identity),
                "manifest_sha256": file_hash(out / "completion_manifest.json"),
                "result_hash": canonical_hash(_read(completed)),
            }:
                raise ValueError("Completed historical endpoint evidence changed")
            return _read(completed)
        _publish(out / "request_manifest.json", {"identity": identity, "requests": requests})
        with isolated_evaluation(adapter, optimizer, metadata=origin["metadata"]):
            rows = collect_samples(
                adapter,
                optimizer,
                origin,
                prompts,
                requests,
                out / "samples",
                identity,
                context["data_root"],
            )
        if state_hash(
            capture_state(adapter.model, optimizer, origin["metadata"], adapter=adapter)
        ) != state_hash(origin):
            raise RuntimeError("Historical inference changed the complete endpoint state")
        if (
            not fixture
            and file_hash(context["endpoint"]["path"]) != context["endpoint"]["file_sha256"]
        ):
            raise ValueError("Historical source checkpoint changed during evaluation")
        if len(rows) != 864 or len({r["sample_key"] for r in rows}) != 864:
            raise ValueError("Historical endpoint has incomplete distinct evaluation outputs")
        groups = {}
        for row in rows:
            key = (row["family"], row["condition"])
            if (
                row["category"] not in ("X", "S", "W", "I")
                or row["execution_checks"]["passed"] is not True
            ):
                raise ValueError("Historical report requires valid measured rows")
            group = groups.setdefault(
                key,
                {"family": key[0], "condition": key[1], "counts": dict.fromkeys("XSWI", 0), "n": 0},
            )
            group["counts"][row["category"]] += 1
            group["n"] += 1
        _publish(out / "condition_counts.json", {"groups": [groups[k] for k in sorted(groups)]})
        result = {
            "status": "CPU_TESTED" if fixture else "PASS",
            "execution_kind": kind,
            "identity": identity,
            "arm": arm,
            "seed": 17,
            "checkpoint_step": 64,
            "evaluation_outputs": 864,
            "maximum_evaluation_outputs": 864,
            "prompt_count": 216,
            "samples_per_prompt": 4,
            "optimizer_updates": 0,
            "training_outputs": 0,
            "training_started": False,
            "full_endpoint_state_unchanged": True,
            "safety_status": "NOT_CERTIFIED",
        }
        manifest = _completion_manifest(out)
        _publish(out / "completion_manifest.json", manifest)
        _publish(
            out / "completion_binding.json",
            {
                "identity_hash": canonical_hash(identity),
                "manifest_sha256": file_hash(out / "completion_manifest.json"),
                "result_hash": canonical_hash(result),
            },
        )
        # Publish success last. Interrupted preparation resumes the existing
        # immutable ledger without drawing additional samples.
        _publish(completed, result)
        return result


def evaluate_historical_r2(
    validated_plan, *, arm, output_root, resume=False, allow_gpu_execution=False
):
    """Explicit optional 864-output R2 measurement of one existing seed17 endpoint."""
    if allow_gpu_execution is not True:
        raise PermissionError("Historical endpoint evaluation requires explicit GPU authorization")
    if arm not in ("X_BASE", "X_VALID"):
        raise ValueError("Only original historical X_BASE/X_VALID endpoints are allowed")
    from .followup_train import _safe_output

    output = _separate_output(
        _safe_output(output_root), [validated_plan["paths"][k] for k in DIRECTORY_KEYS]
    )
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Historical endpoint output exists; explicit resume required")
    context = _load_historical_context(validated_plan, arm)
    return _run_historical_r2(context, validated_plan, arm=arm, output_root=output, resume=resume)
