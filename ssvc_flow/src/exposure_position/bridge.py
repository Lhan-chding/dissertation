"""A single preregistered 32-physical-update J23 continuity bridge.

Every attempt consumes its slot before execution. A failed/interrupted suite is
never retried automatically; its journal remains the authoritative budget record.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from ..optimizer_fork import state_hash
from .runtime import build_runtime, load_parent_backend, load_student_into_backend
from .training import atomic_json, measured_update

IDENTITY_METADATA_FIELDS = frozenset(
    {
        "experiment_id",
        "phase_id",
        "arm",
        "logical_arm_id",
        "frozen_plan_hash",
        "schedule_id",
        "encoded_training_hash",
        "source_bound_hash",
    }
)
SAMPLER_METADATA_FIELDS = frozenset({"schema", "schedule_id", "arm"})
MEASUREMENT_ONLY_FIELDS = frozenset(
    {
        "update_wall_seconds",
        "update_gpu_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    }
)
LEGACY_PAIRS = (("A_LOCAL_C1", "A2_LOCAL_C1_J2"), ("B_FORWARD_C4", "B2_FORWARD_C4_J2"))
RESUME_CASES = tuple(
    (p, a) for p in ("S96", "REP96") for a in ("A3_LOCAL_C1_J3", "B3_FORWARD_C4_J3")
)


def _metadata_changes(legacy, new, allowed):
    changes = {}
    for key in sorted(set(legacy) | set(new)):
        left, right = [key in legacy, legacy.get(key)], [key in new, new.get(key)]
        if left != right:
            if key not in allowed:
                raise ValueError("Unregistered numerical/identity difference: " + key)
            changes[key] = {"legacy": left, "new": right}
    return changes


def _apply_metadata(value, changes):
    result = dict(value)
    for key, difference in changes.items():
        if [key in result, result.get(key)] != difference["new"]:
            raise ValueError("Preregistered identity mapping no longer matches: " + key)
        present, target = difference["legacy"]
        if present:
            result[key] = copy.deepcopy(target)
        else:
            del result[key]
    return result


def _all_slots(runtime):
    cursor = copy.deepcopy(runtime.sampler)
    if cursor.step != 0:
        raise ValueError("Semantic bridge proof must precede all numerical updates")
    slots = []
    for step in range(1, 257):
        slots.extend(cursor.peek())
        cursor.commit(step)
    return slots


def _map_slot(slot, contract):
    mapped = dict(slot)
    if mapped["arm"] != contract["new_arm"]:
        raise ValueError("New slot arm differs from registered semantic mapping")
    mapped["arm"] = contract["legacy_arm"]
    mapped["task_id"] = contract["task_ids"][mapped["task_id"]]
    if "donor_root" in mapped:
        mapped["donor_root"] = contract["root_ids"][mapped["donor_root"]]
    return mapped


def preregister_semantics(legacy, new):
    """Prove all corpus IDs, prompts, token IDs and all 4,096 slots before updates."""
    if legacy.step or new.step:
        raise ValueError("Compatibility registration requires two fresh runtimes")
    if set(legacy.metadata) != set(legacy.encoded) or set(new.metadata) != set(new.encoded):
        raise ValueError("Every encoded task must have its source metadata")
    task_ids, root_ids, rows = {}, {}, []
    for new_id, row in sorted(new.metadata.items()):
        old_id = row.get("source_task_id")
        if old_id not in legacy.metadata:
            raise ValueError("Missing preregistered one-to-one source_task_id")
        old = legacy.metadata[old_id]
        if any(row[k] != old[k] for k in ("prompt", "target", "role", "weight")):
            raise ValueError("Inherited prompt/target/role/weight semantics changed")
        old_tokens, new_tokens = legacy.encoded[old_id], new.encoded[new_id]
        if (
            old_tokens.prompt_ids != new_tokens.prompt_ids
            or old_tokens.target_ids != new_tokens.target_ids
            or old_tokens.token_type_ids != new_tokens.token_type_ids
        ):
            raise ValueError("Tokenized prompt, completion/EOS or multimodal token types changed")
        if old_tokens.task_id != old_id or new_tokens.task_id != new_id:
            raise ValueError("Encoded task identity mismatch")
        if row.get("source_root_id") != old["root_id"]:
            raise ValueError("Explicit source_root_id mapping differs")
        task_ids[new_id] = old_id
        if row["root_id"] in root_ids and root_ids[row["root_id"]] != old["root_id"]:
            raise ValueError("Source root mapping is inconsistent")
        root_ids[row["root_id"]] = old["root_id"]
        rows.append(
            {
                "new_task_id": new_id,
                "source_task_id": old_id,
                "new_root_id": row["root_id"],
                "source_root_id": old["root_id"],
                "prompt_sha256": state_hash(row["prompt"]),
                "target_sha256": state_hash(row["target"]),
                "prompt_tokens_sha256": state_hash(new_tokens.prompt_ids),
                "target_eos_tokens_sha256": state_hash(new_tokens.target_ids),
                "token_type_ids_sha256": state_hash(new_tokens.token_type_ids),
                "role": row["role"],
            }
        )
    if len(set(task_ids.values())) != len(task_ids) or set(task_ids.values()) != set(
        legacy.encoded
    ):
        raise ValueError("Training task mapping must be complete and bijective")
    if len(set(root_ids.values())) != len(root_ids):
        raise ValueError("Root mapping must be bijective")
    contract = {
        "schema": "ser-j23-preupdate-semantic-mapping-v1",
        "legacy_arm": legacy.identity["arm"],
        "new_arm": new.identity["arm"],
        "task_ids": task_ids,
        "root_ids": root_ids,
        "task_proofs": rows,
        "identity_changes": _metadata_changes(
            legacy.identity, new.identity, IDENTITY_METADATA_FIELDS
        ),
        "sampler_changes": _metadata_changes(
            legacy.sampler.state_dict(), new.sampler.state_dict(), SAMPLER_METADATA_FIELDS
        ),
    }
    old_slots, new_slots = _all_slots(legacy), _all_slots(new)
    mapped = [_map_slot(s, contract) for s in new_slots]
    if len(old_slots) != 4096 or old_slots != mapped:
        raise ValueError("All 256x16 explicit slots must preserve exact inherited semantics")
    contract.update(
        all_4096_slots_equal=True,
        legacy_slots_sha256=state_hash(old_slots),
        new_slots_sha256=state_hash(new_slots),
        normalized_slots_sha256=state_hash(mapped),
    )
    return contract


def normalized_state(state, contract):
    """Transform only registered identity fields and IDs; retain every numerical field."""
    result = dict(state)
    result["identity"] = _apply_metadata(state["identity"], contract["identity_changes"])
    result["sampler"] = _apply_metadata(state["sampler"], contract["sampler_changes"])
    exposure = {contract["task_ids"][key]: value for key, value in state["exposures"].items()}
    if len(exposure) != len(state["exposures"]):
        raise ValueError("Exposure mapping is not bijective")
    result["exposures"] = exposure
    return result


def normalized_update(row, contract=None):
    result = {k: copy.deepcopy(v) for k, v in row.items() if k not in MEASUREMENT_ONLY_FIELDS}
    if contract is not None:
        result["task_ids"] = [contract["task_ids"][k] for k in result["task_ids"]]
        result["slots"] = [_map_slot(slot, contract) for slot in result["slots"]]
    return result


class PhysicalBudget:
    def __init__(self, output):
        self.output, self.started, self.completed, self.rows = Path(output), 0, 0, []

    def update(self, runtime, case, branch):
        if self.started >= 32:
            raise ValueError("J23 technical allowance exhausted: 32 physical attempts maximum")
        self.started += 1
        path = self.output / "attempts" / f"physical{self.started:02d}.json"
        started = {
            "status": "UPDATE_STARTED",
            "physical_update": self.started,
            "case": case,
            "branch": branch,
            "logical_update": runtime.step + 1,
            "physical_cost_complete": False,
        }
        atomic_json(path, started)
        stats = measured_update(runtime)
        self.completed += 1
        self.rows.append(stats)
        atomic_json(
            path, {**started, **stats, "status": "UPDATE_APPLIED", "physical_cost_complete": True}
        )
        return stats


def _save(runtime, output, name):
    return runtime.save(Path(output) / (name + ".pt"))


def _restore_case(runtime, directory, budget, case, inference_checker):
    directory.mkdir(parents=True, exist_ok=False)
    origin = runtime.capture()
    zero = _save(runtime, directory, "initial_step000")
    continuous_rows = [budget.update(runtime, case, "continuous") for _ in range(2)]
    continuous = runtime.capture()
    continuous_checkpoint = _save(runtime, directory, "continuous_step002")
    runtime.restore(origin)
    resumed_rows = [budget.update(runtime, case, "interrupted")]
    one = _save(runtime, directory, "resume_step001")
    runtime.restore(origin)
    # Different optimizer instance and disk deserialization exercise cold recovery.
    sampler = copy.deepcopy(runtime.sampler)
    resumed = type(runtime)(
        runtime.adapter,
        seed=runtime.seed,
        identity=runtime.identity,
        parameter_names=runtime.parameter_names,
        sampler=sampler,
        encoded=runtime.encoded,
        metadata=runtime.metadata,
    )
    resumed.resume(one["path"])
    resumed_rows.append(budget.update(resumed, case, "restored"))
    recovered = resumed.capture()
    recovered_checkpoint = _save(resumed, directory, "resumed_step002")
    checks = {
        "complete_state_bitwise_equal": state_hash(continuous) == state_hash(recovered),
        "all_update_fields_equal": all(
            normalized_update(a) == normalized_update(b)
            for a, b in zip(continuous_rows, resumed_rows, strict=True)
        ),
        "initial_adam_empty": not origin["optimizer"]["state"],
        "actual_parameter_updates_observed": any(
            r["parameter_delta_norm"] > 0 for r in continuous_rows
        )
        and any(r["parameter_delta_norm"] > 0 for r in resumed_rows),
        "optimizer_steps_correct": all(
            int(s["step"]) == 2 for s in recovered["optimizer"]["state"].values()
        ),
    }
    if inference_checker is not None:
        checks["inference_weight_restore_exact"] = inference_checker(
            runtime, zero, recovered_checkpoint
        )
    else:
        checks["inference_weight_restore_exact"] = False
    result = {
        "case": case,
        "checks": checks,
        "checkpoints": {
            "initial": zero,
            "continuous": continuous_checkpoint,
            "interrupted": one,
            "resumed": recovered_checkpoint,
        },
        "continuous_state_hash": state_hash(continuous),
        "recovered_state_hash": state_hash(recovered),
    }
    atomic_json(directory / "COMPARISON.json", result)
    return result


def run_bridge_suite(
    output,
    compatibility_factory,
    resume_factory,
    *,
    source_bound_hash,
    matched_targets_equal,
    inference_checker=None,
    code_sha256=None,
    preflight_receipts=(),
    g0_receipt=None,
    preflight=None,
):
    """Dependency-injected mechanics for real execution and explicitly labelled CPU tests."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = PhysicalBudget(output)
    receipt = {
        "status": "RUNNING",
        "code_sha256": copy.deepcopy(code_sha256 or {}),
        "runtime_receipts": copy.deepcopy(list(preflight_receipts)),
        "g0_receipt": copy.deepcopy(g0_receipt),
        "technical_only": True,
        "source_bound_hash": source_bound_hash,
        "maximum_physical_updates": 32,
        "generations": 0,
        "E_CONFIRM2_accessed": False,
        "execution_kind": "NOT_YET_DETERMINED",
        "physical_updates_started": 0,
        "physical_updates_completed": 0,
    }
    atomic_json(
        output / "PREREGISTRATION.json",
        {
            "compatibility": [
                {"parent": "S96", "block": 0, "old_arm": o, "new_arm": n, "updates_each": 4}
                for o, n in LEGACY_PAIRS
            ],
            "resume": [
                {
                    "parent": p,
                    "block": 0,
                    "arm": a,
                    "continuous_updates": 2,
                    "split_updates": [1, 1],
                }
                for p, a in RESUME_CASES
            ],
            "maximum_physical_updates": 32,
            "automatic_retries": False,
            "identity_metadata_allowlist": sorted(IDENTITY_METADATA_FIELDS),
            "sampler_metadata_allowlist": sorted(SAMPLER_METADATA_FIELDS),
            "update_measurement_exclusions": sorted(MEASUREMENT_ONLY_FIELDS),
        },
    )
    atomic_json(output / "BRIDGE_RESULT.json", receipt)
    compatibility, resumes, cuda_flags = [], [], []
    phase = "preflight"
    try:
        if preflight is not None:
            began = time.perf_counter()
            preflight_path = output / "preflight" / "PREFLIGHT_RESULT.json"
            atomic_json(
                preflight_path,
                {
                    "status": "PREFLIGHT_STARTED",
                    "physical_updates_started": 0,
                    "physical_updates_completed": 0,
                    "technical_only": True,
                },
            )
            verified = preflight(output)
            matched_targets_equal = verified.get("matched_targets_equal") is True
            if not matched_targets_equal:
                raise ValueError("Preflight did not prove matched donor targets/EOS")
            receipt["runtime_receipts"].extend(verified.get("runtime_receipts", []))
            atomic_json(
                preflight_path,
                {
                    **verified,
                    "status": "PREFLIGHT_COMPLETE",
                    "wall_seconds": time.perf_counter() - began,
                    "physical_updates_started": 0,
                    "physical_updates_completed": 0,
                    "technical_only": True,
                },
            )
        phase = "legacy_new_compatibility"
        for old_arm, new_arm in LEGACY_PAIRS:
            legacy, new = compatibility_factory(old_arm, new_arm)
            cuda_flags.extend(str(r.adapter.device).startswith("cuda") for r in (legacy, new))
            case = "S96_" + new_arm
            if hasattr(new, "parent_receipt"):
                receipt["runtime_receipts"].append(
                    {"case": case, "shared_legacy_new_backend": True, "receipt": new.parent_receipt}
                )
            directory = output / case
            directory.mkdir()
            contract = preregister_semantics(legacy, new)
            atomic_json(directory / "SEMANTIC_MAPPING_PREUPDATE.json", contract)
            old_origin, new_origin = legacy.capture(), new.capture()
            checkpoints = {"legacy_initial": _save(legacy, directory, "legacy_initial_step000")}
            cold_equal = state_hash(old_origin) == state_hash(
                normalized_state(new_origin, contract)
            )
            old_rows = [budget.update(legacy, case, "legacy") for _ in range(4)]
            old_end = legacy.capture()
            checkpoints["legacy_final"] = _save(legacy, directory, "legacy_step004")
            new.restore(new_origin)
            checkpoints["new_initial"] = _save(new, directory, "new_initial_step000")
            new_rows = [budget.update(new, case, "new") for _ in range(4)]
            new_end = new.capture()
            checkpoints["new_final"] = _save(new, directory, "new_step004")
            checks = {
                "initial_complete_state_equal": cold_equal,
                "final_complete_state_equal": state_hash(old_end)
                == state_hash(normalized_state(new_end, contract)),
                "all_update_fields_equal": all(
                    normalized_update(a) == normalized_update(b, contract)
                    for a, b in zip(old_rows, new_rows, strict=True)
                ),
                "initial_adam_empty": not old_origin["optimizer"]["state"]
                and not new_origin["optimizer"]["state"],
                "all_4096_slots_equal": contract["all_4096_slots_equal"],
                "actual_parameter_updates_observed": any(
                    r["parameter_delta_norm"] > 0 for r in old_rows
                )
                and any(r["parameter_delta_norm"] > 0 for r in new_rows),
            }
            result = {
                "case": case,
                "checks": checks,
                "checkpoints": checkpoints,
                "legacy_state_hash": state_hash(old_end),
                "new_state_hash": state_hash(new_end),
                "new_normalized_state_hash": state_hash(normalized_state(new_end, contract)),
            }
            atomic_json(directory / "COMPARISON.json", result)
            compatibility.append(result)
            del legacy, new, old_origin, new_origin, old_end, new_end
        phase = "new_arm_resume"
        for parent, arm in RESUME_CASES:
            runtime = resume_factory(parent, arm)
            cuda_flags.append(str(runtime.adapter.device).startswith("cuda"))
            case = parent + "_" + arm
            if hasattr(runtime, "parent_receipt"):
                receipt["runtime_receipts"].append(
                    {"case": case, "receipt": runtime.parent_receipt}
                )
            resumes.append(_restore_case(runtime, output / case, budget, case, inference_checker))
            del runtime
        rows = budget.rows
        checks = {
            "new_arm_resume_complete_state_equal": all(
                r["checks"]["complete_state_bitwise_equal"]
                and r["checks"]["optimizer_steps_correct"]
                for r in resumes
            ),
            "new_arm_resume_all_update_fields_equal": all(
                r["checks"]["all_update_fields_equal"] for r in resumes
            ),
            "all_initial_states_empty_adam": all(
                r["checks"]["initial_adam_empty"] for r in compatibility + resumes
            ),
            "all_sequence_weights_one_sixteenth": all(
                r["sequence_coefficient"] == 1 / 16 for r in rows
            ),
            "five_microbatches_observed": all(
                [s[0] for s in r["microbatch_shapes"]] == [4, 4, 3, 1, 4] for r in rows
            ),
            "actual_parameter_updates_observed": all(
                r["checks"]["actual_parameter_updates_observed"] for r in compatibility + resumes
            ),
            "matched_donor_target_eos_equal": matched_targets_equal is True,
            "inference_weight_restore_exact": all(
                r["checks"]["inference_weight_restore_exact"] for r in resumes
            ),
        }
        reuse_compatible = all(all(r["checks"].values()) for r in compatibility)
        status = "PASS" if reuse_compatible else "PASS_NEW_RUNTIME_REUSE_INCOMPATIBLE"
        status = status if all(checks.values()) else "BLOCKED_TECHNICAL"
        checks["reuse_compatible"] = reuse_compatible
        receipt.update(
            status=status,
            execution_kind="REAL_CUDA_BRIDGE" if all(cuda_flags) else "CPU_TEST_FIXTURE",
            checks=checks,
            legacy_compatibility_pass=all(all(r["checks"].values()) for r in compatibility),
            compatibility_cases=compatibility,
            resume_cases=resumes,
            physical_cost_complete=True,
            processed_target_sequences=sum(r["processed_target_sequences"] for r in rows),
            update_gpu_seconds=sum(r["update_gpu_seconds"] or 0 for r in rows),
            update_wall_seconds=sum(r["update_wall_seconds"] for r in rows),
        )
    except BaseException as exc:
        if phase == "preflight" and preflight is not None:
            atomic_json(
                output / "preflight" / "PREFLIGHT_RESULT.json",
                {
                    "status": "PREFLIGHT_FAILED",
                    "error": repr(exc),
                    "physical_updates_started": budget.started,
                    "physical_updates_completed": budget.completed,
                    "wall_seconds": time.perf_counter() - began,
                    "technical_only": True,
                },
            )
        receipt.update(
            status="BLOCKED_TECHNICAL",
            failed_phase=phase,
            error=repr(exc),
            physical_cost_complete=False,
            compatibility_cases=compatibility,
            resume_cases=resumes,
        )
        raise
    finally:
        receipt.update(
            physical_updates_started=budget.started, physical_updates_completed=budget.completed
        )
        atomic_json(output / "BRIDGE_RESULT.json", receipt)
    return receipt


def _inference_checker(runtime, origin, final):
    from .runtime import read_student_checkpoint

    expected = read_student_checkpoint(final["path"])["parameters"]
    runtime.resume(origin["path"])
    backend = runtime._bridge_backend
    load_student_into_backend(backend, final["path"], runtime.identity)
    live = {
        n: p.detach().cpu() for n, p in backend.adapter.model.named_parameters() if n in expected
    }
    return state_hash(expected) == state_hash(live) and all(
        not p.requires_grad for p in backend.adapter.model.parameters()
    )


def validate_g0(run, *, source=None):
    """Read-only cross-binding gate shared by bridge, mode selection and freeze.

    A source-bound training corpus alone does not authorize loading a model.
    Full CPU parent verification, scientific source identity, exact-code CPU
    reconstruction and integrated tests must all refer to this same run.
    """
    from .control import DESIGN_SHA256, PLAN_SHA256, code_bindings, validate_design
    from .schema import EXPERIMENT_ID, file_digest, read_json, verify_source_bound

    run = Path(run).resolve()
    verified_source = verify_source_bound(run)
    if source is not None and source != verified_source:
        raise ValueError("G0 source binding differs from verified current bytes")
    source = verified_source
    design = validate_design(run / "SER_J23_DESIGN.json")
    current_code = code_bindings()
    machine = read_json(run / "machine.json")
    audit = read_json(run / "REUSE_AUDIT.json")
    bindings = read_json(run / "PARENT_AND_ENDPOINT_BINDINGS.json")
    provenance = read_json(run / "SOURCE_PROVENANCE.json")
    cpu = read_json(run / "CPU_CHECK.json")
    validation = read_json(run / "VALIDATION_RECEIPT.json")
    legacy = Path(machine["legacy_run"]).resolve()

    if (
        audit.get("phase_id") != EXPERIMENT_ID
        or bindings.get("phase_id") != EXPERIMENT_ID
        or audit.get("status") not in ("REUSE_CANDIDATE", "RERUN_CANDIDATE")
        or bindings.get("status") != audit.get("status")
        or audit.get("scientific_invariants_verified") is not True
        or audit.get("verify_weights") is not True
        or audit.get("model_calls") != 0
        or audit.get("optimizer_updates") != 0
        or any(e.get("category") == "scientific_contract" for e in audit.get("errors", []))
    ):
        raise ValueError("G0 requires full CPU weights and scientific invariants verification")
    if (
        audit.get("bindings_file", {}).get("sha256")
        != file_digest(run / "PARENT_AND_ENDPOINT_BINDINGS.json")
        or Path(audit.get("legacy_run", "")).resolve() != legacy
        or Path(bindings.get("legacy_run", "")).resolve() != legacy
        or Path(provenance.get("legacy_run", "")).resolve() != legacy
        or bindings.get("frozen_plan", {}).get("sha256") != machine.get("legacy_frozen_plan_sha256")
        or file_digest(legacy / "FROZEN_PLAN.json") != machine.get("legacy_frozen_plan_sha256")
    ):
        raise ValueError("G0 audit, bindings and frozen historical source do not match this run")
    if (
        audit.get("design", {}).get("sha256") != DESIGN_SHA256
        or provenance.get("phase_id") != EXPERIMENT_ID
        or provenance.get("input_design_sha256") != DESIGN_SHA256
        or provenance.get("input_prose_sha256") != PLAN_SHA256
        or file_digest(run / "DESIGN_SOURCE.md") != PLAN_SHA256
        or provenance.get("source_bound_hash") != source["source_hash"]
    ):
        raise ValueError("G0 design or source provenance differs from the exact supplied package")
    records = provenance.get("source_files", [])
    source_files = {r["relative_path"]: r for r in records}
    if len(source_files) != len(records):
        raise ValueError("G0 source provenance contains duplicate paths")
    for registered_path, expected_hash in design["legacy_input_sha256"].items():
        relative = registered_path.removeprefix("run_v1/")
        record = source_files.get(relative, {})
        checked = audit.get("checks", {}).get(registered_path, {})
        path = legacy / relative
        if (
            record.get("sha256") != expected_hash
            or checked.get("status") != "PASS"
            or checked.get("sha256") != expected_hash
            or Path(record.get("path", "")).resolve() != path.resolve()
            or Path(checked.get("path", "")).resolve() != path.resolve()
            or file_digest(path) != expected_hash
        ):
            raise ValueError("G0 fixed historical input/provenance mismatch: " + registered_path)
    parents = bindings.get("parents", {})
    if set(parents) != {"S96", "REP96"}:
        raise ValueError("G0 requires both original parents")
    for name, parent in parents.items():
        checkpoint = parent.get("checkpoint", {})
        lookup = bindings.get("checkpoint_lookup", {}).get("PARENT." + name, {})
        if (
            parent.get("checkpoint_full_cpu_state_verified") is not True
            or not isinstance(checkpoint.get("state_hash"), str)
            or len(checkpoint["state_hash"]) != 64
            or not isinstance(checkpoint.get("sha256"), str)
            or len(checkpoint["sha256"]) != 64
            or lookup.get("status") != "VERIFIED_FULL_CPU_STATE"
            or lookup.get("checkpoint_sha256") != checkpoint["sha256"]
            or lookup.get("path") != checkpoint.get("path")
        ):
            raise ValueError("G0 original parent lacks a complete CPU state binding: " + name)
    if audit["status"] == "REUSE_CANDIDATE" and (
        audit.get("full_A2_B2_checkpoints_verified") != 60
        or audit.get("expected_A2_B2_checkpoints") != 60
        or bindings.get("full_A2_B2_checkpoints_verified") != 60
    ):
        raise ValueError("G0 reuse candidate requires all 60 inherited checkpoint states")
    if (
        cpu.get("phase_id") != EXPERIMENT_ID
        or cpu.get("status") != "PASS"
        or cpu.get("source_bound_hash") != source["source_hash"]
        or cpu.get("code_sha256") != current_code
        or cpu.get("model_calls") != 0
        or validation.get("schema") != "ser-j23-integrated-cpu-validation-v1"
        or validation.get("phase_id") != EXPERIMENT_ID
        or validation.get("status") != "PASS"
        or validation.get("source_bound_hash") != source["source_hash"]
        or type(validation.get("tests")) is not int
        or validation["tests"] <= 0
        or type(validation.get("passed")) is not int
        or not 0 < validation["passed"] <= validation["tests"]
        or validation.get("real_model_calls") != 0
        or validation.get("new_confirmation_generated") is not False
        or validation.get("code_sha256") != current_code
        or type(validation.get("failures")) is not int
        or validation["failures"] != 0
        or type(validation.get("errors")) is not int
        or validation["errors"] != 0
    ):
        raise ValueError(
            "G0 requires passing CPU reconstruction and integrated tests for exact code/source"
        )
    return {
        "status": "PASS",
        "phase_id": EXPERIMENT_ID,
        "source_bound_hash": source["source_hash"],
        "code_sha256": current_code,
        "candidate_status": audit["status"],
        "legacy_run": str(legacy),
        "legacy_frozen_plan_sha256": machine["legacy_frozen_plan_sha256"],
        "receipt_sha256": {
            name: file_digest(run / name)
            for name in (
                "REUSE_AUDIT.json",
                "PARENT_AND_ENDPOINT_BINDINGS.json",
                "SOURCE_PROVENANCE.json",
                "CPU_CHECK.json",
                "VALIDATION_RECEIPT.json",
                "SOURCE_BOUND.json",
            )
        },
    }


def run_bridge(run, machine=None, allow_gpu=False):
    """Real bridge entrypoint: source-bound only, before the formal plan freeze."""
    from ..exposure_substitution.runtime import build_runtime as build_legacy_runtime
    from ..modeling_v3.io import canonical_hash
    from .control import code_bindings
    from .runtime import legacy_run_directory
    from .schema import ARMS, training_rows, verify_source_bound
    from .training import encode_rows

    if allow_gpu is not True:
        raise PermissionError("J23 real GPU continuity bridge requires --allow-gpu")
    run = Path(run).resolve()
    if (run / "STOP").exists() or (run / "FROZEN_PLAN.json").exists():
        raise RuntimeError("Technical bridge must precede formal freeze and cannot run after STOP")
    source = verify_source_bound(run)
    g0 = validate_g0(run, source=source)
    output = run / "bridge"
    if output.exists() or (run / "CONTINUITY_BRIDGE.json").exists():
        raise ValueError(
            "Single technical bridge already started; preserve its physical budget journal"
        )
    legacy_run = legacy_run_directory(run, technical=True, machine=machine)

    def preflight(output):
        # The suite creates its immutable attempt directory before any model load.
        backend = load_parent_backend(run, "S96", machine=machine, allow_gpu=True, technical=True)
        parent_receipt = {
            "case": "target_token_preflight",
            "receipt": copy.deepcopy(backend.receipt),
        }
        atomic_json(output / "preflight" / "PARENT_LOAD.json", parent_receipt)
        tokens_by_root = {}
        for arm in ARMS:
            rows = {
                k: r
                for k, r in training_rows(run, arm, technical=True).items()
                if r["role"] == "donor"
            }
            encoded = encode_rows(backend, rows)
            for key, row in rows.items():
                root, target = row["source_root_id"], encoded[key].target_ids
                if root in tokens_by_root and tokens_by_root[root] != target:
                    raise ValueError("Four matched donor targets/EOS differ")
                tokens_by_root[root] = target
        if len(tokens_by_root) != 32:
            raise ValueError("Exactly 32 inherited donor roots required")
        return {
            "matched_targets_equal": True,
            "matched_donor_roots": 32,
            "runtime_receipts": [parent_receipt],
        }

    def compatibility_factory(old_arm, new_arm):
        backend = load_parent_backend(run, "S96", machine=machine, allow_gpu=True, technical=True)
        old = build_legacy_runtime(legacy_run, backend, "S96", 0, old_arm, technical=True)
        new = build_runtime(run, backend, "S96", 0, new_arm, technical=True)
        return old, new

    def resume_factory(parent, arm):
        backend = load_parent_backend(run, parent, machine=machine, allow_gpu=True, technical=True)
        runtime = build_runtime(run, backend, parent, 0, arm, technical=True)
        runtime._bridge_backend = backend
        return runtime

    result = run_bridge_suite(
        output,
        compatibility_factory,
        resume_factory,
        source_bound_hash=canonical_hash(source),
        matched_targets_equal=False,
        inference_checker=_inference_checker,
        code_sha256=code_bindings(),
        g0_receipt=g0,
        preflight=preflight,
    )
    # Preserve adverse numerical comparisons as immutable mode-selection input.
    path = run / "CONTINUITY_BRIDGE.json"
    with path.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result
