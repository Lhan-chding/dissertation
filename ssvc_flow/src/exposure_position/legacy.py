"""Read-only CPU verification of original SER-J2 evidence and checkpoint bytes.

No model is loaded, CUDA initialized, execution mode chosen, or legacy file
edited. REUSE_CANDIDATE still requires the separately budgeted numerical bridge.
Original identities remain unchanged; logical J23 aliases live beside them.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..exposure_substitution.schema import digest, file_digest, read_json, read_jsonl
from ..exposure_substitution.training import atomic_json

PHASE_ID = "SER_J23_20261008"
PARENTS = ("S96", "REP96")
STEPS = (0, 64, 128, 192, 256)
ARM_ALIASES = {"A_LOCAL_C1": "A2_LOCAL_C1_J2", "B_FORWARD_C4": "B2_FORWARD_C4_J2"}
LEGACY_ARMS = (*ARM_ALIASES, "C_FORWARD_C3")
REGISTERED_INPUT_SHA256 = {
    "run_v1/reports/METRICS_BY_CELL.csv": (
        "e3b9501eaf6ce84c937b6602a6f6953fbc33830a9d2b6043e95cf64dd6fd4222"
    ),
    "run_v1/manifests/new_root_registry_AUDIT_ONLY.jsonl": (
        "9ec91f5665370dce3ddabbce3233350b28044383ad327b993b50b00c553f11f6"
    ),
    "run_v1/protocol.json": "e2b73ae6480f3a6f34f0910e094d91ce19fb035f1026f5e7e39fece1d12fce53",
    "run_v1/machine.json": "23a9be4fb958651ed9b8f9d63323f69e53bc82174f697ec6440c58961a184173",
}
FULL_STATE_FIELDS = {
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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size}


def _tensor_inventory(state):
    """Check all 192 original FP32 rank-eight A/B tensors without a model load."""
    import torch

    from ..followup_updates import _safe

    _require(state.keys() >= FULL_STATE_FIELDS, "Incomplete trainable/Adam/RNG/forward state")
    _safe(state)
    parameters = state["parameters"]
    _require(
        isinstance(parameters, dict) and len(parameters) == 192,
        "Exactly 192 original LoRA A/B tensors required",
    )
    modules, inventory = {}, {}
    for name, value in parameters.items():
        match = re.search(
            r"(?:^|\.)model\.language_model\.layers\.(\d+)\.mlp\."
            r"(gate_proj|up_proj|down_proj)\.lora_([AB])\.default\.weight$",
            name,
        )
        _require(match is not None, "Unexpected original LoRA parameter: " + name)
        layer, leaf, side = match.groups()
        _require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.ndim == 2
            and value.shape[0 if side == "A" else 1] == 8,
            "LoRA tensor CPU/dtype/shape/rank mismatch: " + name,
        )
        key = (int(layer), leaf)
        _require(side not in modules.setdefault(key, set()), "Duplicate LoRA module side")
        modules[key].add(side)
        inventory[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    _require(
        set(modules)
        == {(i, leaf) for i in range(32) for leaf in ("gate_proj", "up_proj", "down_proj")}
        and all(sides == {"A", "B"} for sides in modules.values()),
        "Original 32-layer/96-module LoRA whitelist differs",
    )
    _require(
        isinstance(state["rng"], dict)
        and set(state["rng"]) == {"python", "numpy", "torch", "cuda"},
        "Complete Python/NumPy/Torch/CUDA RNG state required",
    )
    import numpy as np

    random.Random().setstate(state["rng"]["python"])
    numpy_state = state["rng"]["numpy"]
    np.random.RandomState().set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.Generator(device="cpu").set_state(state["rng"]["torch"])
    _require(
        isinstance(state["rng"]["cuda"], list)
        and len(state["rng"]["cuda"]) == 1
        and all(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.numel() > 0
            for value in state["rng"]["cuda"]
        ),
        "One complete original CUDA RNG vector required",
    )
    _require(
        isinstance(state["module_modes"], dict)
        and bool(state["module_modes"])
        and all(type(value) is bool for value in state["module_modes"].values()),
        "Saved module modes missing or invalid",
    )
    return inventory


def _checkpoint_payload(path, receipt):
    import torch

    from ..optimizer_fork import state_hash

    binding = _binding(path)
    _require(binding["sha256"] == receipt["sha256"], "Checkpoint file SHA256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(
        isinstance(payload, dict) and isinstance(payload.get("state"), dict),
        "Invalid checkpoint payload",
    )
    actual = state_hash(payload["state"])
    _require(
        actual == payload.get("state_hash") == receipt["state_hash"],
        "Full CPU checkpoint state hash differs from retained receipt",
    )
    return payload, {**binding, "state_hash": actual}


def _audit_parent(record, private, prompt_protocol, *, verify_weights):
    from ..protocol_state_probes.checkpoint_catalog import (
        map_path,
        read_bound_json,
        validate_manifest_identity,
    )
    from ..protocol_state_probes.inference import _validate_checkpoint_metadata, _validate_protocol

    name = record["checkpoint_id"]
    _require(
        name in PARENTS
        and record.get("status") == "AVAILABLE"
        and record.get("source", {}).get("id") == name,
        "Original parent source/availability identity differs",
    )
    source = record["source"]
    _require(
        source.get("kind") == "source"
        and source.get("source_step") == 96
        and source.get("lineage") == {"S96": 61001, "REP96": 61003}[name],
        "Original S96/REP96 lineage and step are required",
    )
    mappings = record.get("path_mappings", {})
    parent = read_bound_json(private["bindings"]["parent_validated_plan"], mappings)
    _validate_protocol(prompt_protocol, private["inherited_config"], parent)
    manifest = read_bound_json(record["manifest"])
    commit = read_bound_json(record["commit"])
    validate_manifest_identity(source, manifest, commit)
    original = record["checkpoint_original_binding"]
    _require(commit["checkpoint"] == original, "Authoritative parent COMMIT changed")
    expected = {**original, "path": str(map_path(original["path"], mappings))}
    _require(
        record["checkpoint"] == expected,
        "Mapped parent checkpoint differs from authoritative COMMIT",
    )
    history = manifest["runtime_identity"]
    _require(history == record["runtime_identity"], "Parent runtime identity changed")
    _require(
        history["model_hash"] == digest(parent["snapshot"]), "Parent base snapshot binding differs"
    )
    _require(
        history["config_hash"] == digest(private["inherited_config"])
        and manifest["config_hash"] == private["config_hash"],
        "Parent inherited/prospective configuration differs",
    )
    certificate = parent["gate"]["certificate"]
    _require(
        certificate.get("status") == "PASS"
        and certificate.get("selected_path") == "uncached_prefix_recompute",
        "Original uncached inference certificate missing",
    )
    audit = certificate["model_audit"]
    _require(
        len(audit.get("lora_modules", [])) == 96
        and audit.get("lora_rank") == 8
        and audit.get("lora_alpha") == 16
        and audit.get("lora_dropout") == 0,
        "Original LoRA certificate differs",
    )
    result = {
        "source_checkpoint_id": name,
        "source_experiment_id": "PROSPECTIVE_SELECTION_V2",
        "checkpoint": record["checkpoint"],
        "manifest": record["manifest"],
        "commit": record["commit"],
        "source": source,
        "historical_runtime_identity": history,
        "historical_model_audit": audit,
        "parent_plan": private["bindings"]["parent_validated_plan"],
        "checkpoint_full_cpu_state_verified": False,
        "model_loaded": False,
    }
    inventory = None
    if verify_weights:
        payload, checked = _checkpoint_payload(expected["path"], expected)
        _require(
            payload.get("identity") == expected["identity"],
            "Parent payload identity differs from COMMIT",
        )
        state = payload["state"]
        _require(
            state.get("schema") == "ssvc-v4-complete-trainable-state-1"
            and state.get("scheduler") is None,
            "Complete historical parent schema/scheduler required",
        )
        _require(
            isinstance(state.get("optimizer"), dict)
            and state.get("sampler", {}).get("kind") == "mapping",
            "Original complete parent optimizer/sampler required",
        )
        _validate_checkpoint_metadata(record, state["metadata"])
        inventory = _tensor_inventory(state)
        from ..optimizer_fork import state_hash

        forward_hash = state_hash(
            {
                key: state[key]
                for key in ("parameters", "buffers", "position_state", "adapter_position_state")
            }
        )
        result.update(
            # Preserve the authoritative COMMIT/backend checkpoint identity;
            # local CPU audit metadata is not part of that historical object.
            checkpoint=expected,
            checkpoint_verified_bytes=checked["bytes"],
            checkpoint_full_cpu_state_verified=True,
            parameter_inventory=inventory,
            parameter_inventory_hash=digest(inventory),
            inference_fingerprint=digest(
                {
                    "checkpoint_sha256": checked["sha256"],
                    "base_snapshot_hash": history["model_hash"],
                    "forward_state_hash": forward_hash,
                    "mode": "eval_positions_reset",
                    "generation": prompt_protocol["generation"],
                }
            ),
        )
    return result, inventory, parent


def _verify_student_checkpoint(path, receipt, identity, schedule, parent_inventory):
    """Validate immutable legacy identity and exact prefix exposure/optimizer state."""
    payload, binding = _checkpoint_payload(path, receipt)
    state, step = payload["state"], receipt["step"]
    _require(
        state.get("schema") == "ser-j2-student-state-v1"
        and state.get("identity") == identity
        and state.get("step") == step
        and state.get("seed") == identity["seed"],
        "Student schema/identity/step/seed differs",
    )
    inventory = _tensor_inventory(state)
    _require(
        inventory == parent_inventory, "Student parameter whitelist/dtype/shape differs from parent"
    )
    _require(state.get("parameter_names") == sorted(inventory), "Student parameter order differs")
    _require(
        state.get("scheduler") == {"kind": "linear8_constant", "completed_updates": step},
        "Student scheduler differs",
    )
    expected_sampler = {
        "schema": "SER-J2-explicit-cursor-v1",
        "schedule_id": digest(schedule),
        "arm": identity["arm"],
        "committed_step": step,
    }
    _require(state.get("sampler") == expected_sampler, "Student sampler identity/cursor differs")
    selected = [row for row in schedule if row["arm"] == identity["arm"] and row["update"] <= step]
    expected_exposures = dict(Counter(row["task_id"] for row in selected))
    _require(
        state.get("exposures") == expected_exposures,
        "Student per-task exposures differ from exact schedule prefix",
    )
    _require(
        state.get("role_exposures") == {"common": 11 * step, "donor": step, "replay": 4 * step},
        "Student role exposures differ",
    )
    optimizer = state["optimizer"]
    groups = optimizer.get("param_groups", [])
    _require(len(groups) == 1, "Exactly one original AdamW group required")
    group = groups[0]
    ids = group.get("params", [])
    _require(
        len(ids) == 192
        and len(set(ids)) == 192
        and tuple(group.get("betas", ())) == (0.9, 0.999)
        and group.get("eps") == 1e-8
        and group.get("weight_decay") == 0
        and group.get("lr") == 1e-5 * (min(step / 8, 1) if step else 1),
        "Student AdamW whitelist/hyperparameters differ",
    )
    states = optimizer.get("state", {})
    _require(
        set(states) == (set(ids) if step else set()), "Student AdamW state is missing or unexpected"
    )
    for param_id, name in zip(ids, sorted(inventory), strict=True):
        if not step:
            continue
        value = states[param_id]
        _require(float(value["step"]) == step, "Student Adam step differs")
        for field in ("exp_avg", "exp_avg_sq"):
            tensor = value[field]
            _require(
                list(tensor.shape) == inventory[name]["shape"]
                and str(tensor.dtype) == inventory[name]["dtype"],
                "Student Adam moment shape/dtype differs",
            )
    return {
        **binding,
        "step": step,
        "status": "VERIFIED_FULL_CPU_STATE",
        "parameter_inventory_hash": digest(inventory),
        "source_identity": identity,
    }


def _release_and_queue(run, frozen):
    """Read SQLite in mode=ro; do not call the old mutating Queue constructor."""
    from ..verified_discovery_transfer.queue import digest as queue_digest

    release = read_json(run / "FINAL_RELEASE.json")
    registry = read_json(run / "REGISTERED_MATRIX.json")
    _require(
        release.get("status") == "RELEASED"
        and release.get("scientific_completion") is True
        and release.get("technical_missing") == []
        and release.get("all_registered_models_terminal") is True
        and release.get("all_registered_terminal") is True
        and release.get("registered_training_jobs") == 18
        and release.get("registered_evaluation_jobs") == 95
        and release.get("plan_hash") == frozen["plan_hash"],
        "Legacy complete release contract differs",
    )
    _require(
        registry.get("plan_hash") == queue_digest(frozen)
        and registry.get("identity")
        == queue_digest({k: v for k, v in registry.items() if k != "identity"}),
        "Legacy matrix registration identity differs",
    )
    with sqlite3.connect((run / "queue.sqlite").as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            {
                **dict(row),
                "payload": json.loads(row["payload"]),
                "result": json.loads(row["result"]) if row["result"] else None,
            }
            for row in connection.execute("SELECT * FROM jobs ORDER BY rowid")
        ]
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    _require(
        len(rows) == 113
        and all(row["status"] == "COMPLETE" for row in rows)
        and [row["payload"] for row in rows] == registry["jobs"]
        and metadata.get("identity") == registry["identity"]
        and metadata.get("released") == release.get("matrix_digest") == queue_digest(rows),
        "Legacy read-only queue/release/matrix binding differs",
    )
    training = {row["id"]: row for row in rows if row["payload"]["kind"] == "train"}
    _require(
        set(training)
        == {f"train.{p}.{b}.{a}" for p in PARENTS for b in range(3) for a in LEGACY_ARMS},
        "Exact 18 legacy training paths required",
    )
    evaluations = [row for row in rows if row["payload"]["kind"] == "eval"]
    receipts = release.get("completed_evaluation_receipts", [])
    _require(
        len(receipts) == 95
        and len({row["job_id"] for row in receipts}) == 95
        and {row["job_id"] for row in receipts} == {row["id"] for row in evaluations}
        and all(row.get("status") == "COMPLETE" for row in receipts),
        "Released evaluation receipt coverage differs",
    )
    for row in evaluations:
        completed = read_json(run / "evaluations" / row["id"] / "completion.json")
        _require(completed in receipts, "Released evaluation completion receipt changed")
    return release, registry, training


def _audit_snapshot(parent, mappings):
    from ..followup_backend import _snapshot_binding
    from ..protocol_state_probes.checkpoint_catalog import map_path

    expected = parent["snapshot"]
    actual = _snapshot_binding(map_path(expected["path"], mappings), expected["revision"])
    _require(
        {k: v for k, v in actual.items() if k != "path"}
        == {k: v for k, v in expected.items() if k != "path"},
        "Base snapshot files/revision differ from original binding",
    )
    return {
        "status": "VERIFIED_ALL_SNAPSHOT_FILE_BYTES",
        "snapshot": actual,
        "original_snapshot_hash": digest(expected),
        "model_loaded": False,
    }


def _status(errors, verify_weights):
    if any(row["category"] == "scientific_contract" for row in errors):
        return "BLOCKED"
    if not verify_weights:
        return "WEIGHTS_NOT_VERIFIED"
    if any(row["category"] == "reuse_checkpoint" for row in errors):
        return "RERUN_CANDIDATE"
    return "REUSE_CANDIDATE"


def audit_legacy(design_path, legacy_run, output_dir, *, verify_weights=True):
    """Write audit and bindings, but never select or freeze an execution mode.

    Original parent/data/runtime failures block both modes. A2/B2 checkpoint
    failures permit only a later decision about the complete RERUN_24 fallback.
    Missing C diagnostics are reported independently. Disabling weight reads
    can never qualify checkpoint reuse.
    """
    _require(type(verify_weights) is bool, "verify_weights must be boolean")
    run, output = Path(legacy_run).resolve(), Path(output_dir).resolve()
    _require(
        output != run and not output.is_relative_to(run),
        "Audit output must be outside the immutable legacy run",
    )
    output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], {}
    bindings = {
        "schema": "ser-j23-parent-endpoint-bindings-v1",
        "phase_id": PHASE_ID,
        "legacy_run": str(run),
        "parents": {},
        "endpoints": [],
        "diagnostic_C": [],
        "runtime_environment": None,
        "model_loaded": False,
        "cuda_initialized_by_audit": False,
    }

    def attempt(scope, category, function):
        try:
            value = function()
            checks[scope] = {"status": "PASS"}
            return value
        except Exception as exc:
            error = {
                "scope": scope,
                "category": category,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            errors.append(error)
            checks[scope] = {"status": "FAILED", "error": error}
            return None

    design = attempt("design", "scientific_contract", lambda: read_json(design_path))
    if design is not None:
        attempt(
            "design_fixed_input_hashes",
            "scientific_contract",
            lambda: _require(
                design.get("legacy_input_sha256") == REGISTERED_INPUT_SHA256,
                "Design must bind the four supplied legacy SHA256 values exactly",
            ),
        )
    for relative, expected in REGISTERED_INPUT_SHA256.items():
        path = run / relative.removeprefix("run_v1/")

        def verify_file(path=path, expected=expected):
            record = _binding(path)
            _require(record["sha256"] == expected, "Registered legacy input SHA256 differs")
            return record

        value = attempt(relative, "scientific_contract", verify_file)
        if value is not None:
            checks[relative].update(value)
    from ..exposure_substitution.import_sources import cpu_check
    from ..exposure_substitution.schema import verify_frozen

    frozen = attempt("frozen_plan", "scientific_contract", lambda: verify_frozen(run))
    attempt("legacy_cpu_data_contract", "scientific_contract", lambda: cpu_check(run))
    if frozen is not None:
        bindings["frozen_plan"] = {
            **_binding(run / "FROZEN_PLAN.json"),
            "plan_hash": frozen["plan_hash"],
            "full_object_hash": digest(frozen),
        }
        attempt(
            "legacy_exclusion_status",
            "scientific_contract",
            lambda: _require(
                frozen.get("later_exclusion_status") == "VERIFIED_EXPLICIT_SCAN",
                "Legacy later-history exclusion audit is not verified",
            ),
        )
    source_context = (
        attempt(
            "legacy_release_and_queue",
            "scientific_contract",
            lambda: _release_and_queue(run, frozen),
        )
        if frozen
        else None
    )
    machine = attempt("machine", "scientific_contract", lambda: read_json(run / "machine.json"))
    private = prompt_protocol = None
    if frozen is not None and machine is not None:
        for key in ("runtime_path", "frozen_protocol_path", "checkpoint_catalog"):

            def read_external(key=key):
                record = frozen["external_inputs"][key]
                _require(
                    record.get("status") == "VERIFIED_BYTES"
                    and file_digest(machine[key]) == record["sha256"],
                    "Frozen external input bytes changed: " + key,
                )
                return read_json(machine[key])

            value = attempt("external_" + key, "scientific_contract", read_external)
            if key == "runtime_path":
                private = value
            if key == "frozen_protocol_path":
                prompt_protocol = value
        attempt(
            "python_executable",
            "scientific_contract",
            lambda: _require(
                Path(sys.executable).resolve() == Path(machine["python_executable"]).resolve(),
                "Current Python executable differs from original machine binding",
            ),
        )
    catalog = attempt(
        "parent_catalog", "scientific_contract", lambda: read_json(run / "parent_availability.json")
    )
    parents, inventories, parent_plans = {}, {}, {}
    if catalog is not None:
        records = catalog.get("checkpoints", [])
        attempt(
            "parent_catalog_coverage",
            "scientific_contract",
            lambda: _require(
                len(records) == 2 and {row["checkpoint_id"] for row in records} == set(PARENTS),
                "Exactly S96 and REP96 original parents required",
            ),
        )
        parents = {
            row["checkpoint_id"]: row for row in records if row.get("checkpoint_id") in PARENTS
        }
        for name in PARENTS:
            value = attempt(
                "parent_" + name,
                "scientific_contract",
                lambda name=name: _audit_parent(
                    parents[name], private, prompt_protocol, verify_weights=verify_weights
                ),
            )
            if value is not None:
                bindings["parents"][name], inventories[name], parent_plans[name] = value
    if parent_plans:
        from ..next_stage_runtime import validate_runtime_environment

        name = next(iter(parent_plans))

        def environment_check():
            observed = validate_runtime_environment(parent_plans[name]["gate"])
            for parent_name, parent_plan in parent_plans.items():
                _require(
                    observed
                    == parents[parent_name]["runtime_identity"]["environment"]
                    == parent_plan["r4_binding"]["environment"],
                    "Current package/Python fingerprint differs from historical parent",
                )
            return observed

        bindings["runtime_environment"] = attempt(
            "current_runtime_environment", "scientific_contract", environment_check
        )
        if verify_weights:
            bindings["base_snapshot"] = attempt(
                "base_snapshot_bytes",
                "scientific_contract",
                lambda: _audit_snapshot(parent_plans[name], parents[name].get("path_mappings", {})),
            )
    if source_context is not None:
        _release, registry, training = source_context
        bindings["release"] = _binding(run / "FINAL_RELEASE.json")
        bindings["registered_matrix"] = _binding(run / "REGISTERED_MATRIX.json")
        code_root = Path(__file__).resolve().parents[1]
        for relative, expected in registry.get("code_identity", {}).items():

            def code_check(relative=relative, expected=expected):
                path = Path(relative)
                _require(not path.is_absolute() and ".." not in path.parts, "Unsafe source binding")
                _require(
                    file_digest(code_root / path) == expected,
                    "Original scientific source changed: " + relative,
                )

            attempt("source:" + relative, "scientific_contract", code_check)
        schedules = {}
        for block in range(3):
            schedules[block] = attempt(
                "schedule:" + str(block),
                "scientific_contract",
                lambda block=block: read_jsonl(
                    run / "manifests" / f"schedule_{108701 + block}.jsonl"
                ),
            )
        encoded_hashes = {}
        for parent in PARENTS:
            for block in range(3):
                for arm in LEGACY_ARMS:
                    job_id = f"train.{parent}.{block}.{arm}"
                    directory = run / "training" / f"{parent}_block{block}_{arm}"
                    category = "reuse_checkpoint" if arm in ARM_ALIASES else "diagnostic_checkpoint"
                    row = {
                        "source_experiment_id": "SER_J2_20261007",
                        "source_checkpoint_id": job_id,
                        "logical_arm_id": ARM_ALIASES.get(arm),
                        "parent": parent,
                        "block": block,
                        "checkpoints": [],
                        "source_identity": None,
                    }
                    (
                        bindings["endpoints"] if arm in ARM_ALIASES else bindings["diagnostic_C"]
                    ).append(row)

                    def result_check(
                        directory=directory, job_id=job_id, parent=parent, block=block, arm=arm
                    ):
                        result = read_json(directory / "SFT_RESULT.json")
                        _require(
                            result == training[job_id]["result"],
                            "SFT result differs from released queue",
                        )
                        identity = result["identity"]
                        expected = {
                            "experiment_id": "SER_J2_20261007",
                            "parent": parent,
                            "block": block,
                            "arm": arm,
                            "frozen_plan_hash": digest(frozen),
                            "schedule_id": digest(schedules[block]),
                            "seed": 108701 + block,
                            "parent_checkpoint_sha256": parents[parent]["checkpoint"]["sha256"],
                            "loss": "completion_sequence_mean_slots16_donor_isolated_44314",
                            "steps": 256,
                            "technical_only": False,
                        }
                        _require(
                            set(identity) == set(expected) | {"encoded_training_hash"}
                            and all(identity.get(k) == v for k, v in expected.items())
                            and re.fullmatch(r"[0-9a-f]{64}", identity["encoded_training_hash"]),
                            "Original student identity differs from frozen contract",
                        )
                        prior = encoded_hashes.setdefault(arm, identity["encoded_training_hash"])
                        _require(
                            prior == identity["encoded_training_hash"],
                            "Encoded corpus identity differs across parent/block",
                        )
                        exposures = dict(
                            Counter(r["task_id"] for r in schedules[block] if r["arm"] == arm)
                        )
                        _require(
                            result.get("status") == "SFT_256_COMPLETE"
                            and result.get("execution_kind") == "REAL_CUDA_SFT"
                            and result.get("optimizer_updates") == 256
                            and result.get("exposures") == exposures
                            and result.get("role_exposures")
                            == {"common": 2816, "donor": 256, "replay": 1024},
                            "Legacy endpoint completion/exposure contract differs",
                        )
                        receipts = result["checkpoints"]
                        _require(
                            len(receipts) == 5 and {r["step"] for r in receipts} == set(STEPS),
                            "Legacy five-checkpoint receipt coverage differs",
                        )
                        return result

                    result_category = (
                        "scientific_contract"
                        if (directory / "SFT_RESULT.json").is_file()
                        else category
                    )
                    result = attempt(job_id + ":result", result_category, result_check)
                    if result is None:
                        continue
                    row["source_identity"] = result["identity"]
                    row["result_binding"] = _binding(directory / "SFT_RESULT.json")
                    for receipt in result["checkpoints"]:
                        step = receipt["step"]
                        if arm not in ARM_ALIASES and step != 256:
                            continue
                        path = directory / f"step{step:03d}.pt"

                        def checkpoint_check(
                            path=path, receipt=receipt, parent=parent, block=block, result=result
                        ):
                            _require(
                                Path(receipt["path"]).resolve() == path.resolve(),
                                "Checkpoint receipt path differs from original run",
                            )
                            if not verify_weights:
                                return {
                                    **receipt,
                                    "status": "NOT_VERIFIED",
                                    "path_exists": path.is_file(),
                                }
                            _require(
                                parent in inventories and inventories[parent] is not None,
                                "Parent tensor inventory unavailable; student cannot be certified",
                            )
                            return _verify_student_checkpoint(
                                path,
                                receipt,
                                result["identity"],
                                schedules[block],
                                inventories[parent],
                            )

                        checked = attempt(job_id + f":step{step}", category, checkpoint_check)
                        row["checkpoints"].append(
                            checked or {**receipt, "status": "UNAVAILABLE_OR_INVALID"}
                        )
    status = _status(errors, verify_weights)
    verified = sum(
        checkpoint.get("status") == "VERIFIED_FULL_CPU_STATE"
        for row in bindings["endpoints"]
        for checkpoint in row["checkpoints"]
    )
    lookup = {}
    for parent, record in bindings["parents"].items():
        checkpoint = record["checkpoint"]
        lookup["PARENT." + parent] = {
            "path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "source_experiment_id": record["source_experiment_id"],
            "source_checkpoint_id": parent,
            "expected_identity": checkpoint["identity"],
            "status": "VERIFIED_FULL_CPU_STATE"
            if record["checkpoint_full_cpu_state_verified"]
            else "NOT_VERIFIED",
        }
    for row in bindings["endpoints"] + bindings["diagnostic_C"]:
        arm = row["logical_arm_id"] or "C_FORWARD_C3"
        identity = row["source_identity"]
        for checkpoint in row["checkpoints"]:
            key = f"LEGACY.{row['parent']}.block{row['block']}.{arm}.step{checkpoint['step']}"
            lookup[key] = {
                "path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "source_experiment_id": "SER_J2_20261007",
                "source_checkpoint_id": (
                    f"{row['parent']}.block{row['block']}."
                    f"{identity['arm']}.step{checkpoint['step']}"
                ),
                "expected_identity": identity,
                "status": checkpoint["status"],
            }
    bindings.update(
        status=status,
        errors=errors,
        full_A2_B2_checkpoints_verified=verified,
        checkpoint_lookup=lookup,
    )
    audit = {
        "schema": "ser-j23-legacy-audit-v1",
        "phase_id": PHASE_ID,
        "status": status,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": _binding(design_path)
        if Path(design_path).is_file()
        else {"path": str(design_path)},
        "legacy_run": str(run),
        "verify_weights": verify_weights,
        "scientific_invariants_verified": not any(
            e["category"] == "scientific_contract" for e in errors
        ),
        "full_A2_B2_checkpoints_verified": verified,
        "expected_A2_B2_checkpoints": 60,
        "bridge_required": True,
        "execution_mode_frozen": False,
        "model_calls": 0,
        "optimizer_updates": 0,
        "checks": checks,
        "errors": errors,
        "remaining_gates": [
            "old/new numerical continuity bridge",
            "new-arm exact resume bridge",
            "new historical-exclusion audit",
            "current GPU/backend fingerprint",
            "actual prompt/target token and EOS comparison",
            "execution-mode freeze",
        ],
    }
    atomic_json(output / "PARENT_AND_ENDPOINT_BINDINGS.json", bindings)
    audit["bindings_file"] = _binding(output / "PARENT_AND_ENDPOINT_BINDINGS.json")
    atomic_json(output / "REUSE_AUDIT.json", audit)
    return audit
