"""SER-J23 stage gates; design decisions remain independent of model outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from ..exposure_substitution.evaluate import _write_once
from .schema import ARMS, EXPERIMENT_ID, PARENTS, digest, file_digest, read_json

DESIGN_SHA256 = "1d784d72c08fd558eb0ef8a7266aebc8aae42f1215548f5edd235ee8ffec092e"
PLAN_SHA256 = "50e97675368c647d44db9954b31bf81a0f7158887665383570ecb6e5e423a38d"


def validate_design(path):
    path = Path(path)
    if file_digest(path) != DESIGN_SHA256:
        raise ValueError("This runner requires the exact user-supplied FINAL v1.0 design")
    design = read_json(path)
    if design["phase_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong phase")
    return design


def write_once_bytes(path, payload):
    """Publish private source material without replacing an earlier binding."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Immutable source changed: " + str(path))
        return
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def bind_design(design_path, run):
    design_path, run = Path(design_path), Path(run)
    design = validate_design(design_path)
    source = design_path.parent / "CODEX_NEXT_EXPERIMENT_PLAN_SER_J23_FINAL_zh.md"
    if file_digest(source) != PLAN_SHA256:
        raise ValueError("Design prose differs from the supplied final contract")
    write_once_bytes(run / "SER_J23_DESIGN.json", design_path.read_bytes())
    write_once_bytes(run / "DESIGN_SOURCE.md", source.read_bytes())
    _write_once(
        run / "DESIGN_SOURCE_SHA256.json",
        {
            "schema": "ser-j23-source-contract-v1",
            "phase_id": EXPERIMENT_ID,
            "design_sha256": DESIGN_SHA256,
            "prose_sha256": PLAN_SHA256,
        },
    )
    for name in ("GUARD_FEASIBILITY_ONLY.json", "MANUAL_FIXTURE_ORBIT_EXCLUSIONS.json"):
        write_once_bytes(run / name, (design_path.parent / "evidence" / name).read_bytes())
    return design


def write_manifests(run, files):
    for relative, value in sorted(files.items()):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe manifest path")
        path = Path(run) / "manifests" / rel
        payload = (
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                for row in value
            )
            if rel.suffix == ".jsonl"
            else json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        write_once_bytes(path, payload.encode())


def code_bindings():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.rglob("*.py"))
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(paths)}


def assert_code_bindings(frozen):
    if frozen.get("code_sha256") != code_bindings():
        raise ValueError(
            "Code changed after freeze; preserve the run and diagnose before any execution"
        )


def choose_mode(run):
    """Use verified availability and bridge only; never inspect scientific answers."""
    run = Path(run)
    audit = read_json(run / "REUSE_AUDIT.json")
    bridge = read_json(run / "CONTINUITY_BRIDGE.json")
    from ..modeling_v3.io import canonical_hash
    from .bridge import validate_g0
    from .schema import verify_source_bound
    from .training import REQUIRED_BRIDGE_CHECKS

    g0 = validate_g0(run)

    if (
        bridge.get("execution_kind") != "REAL_CUDA_BRIDGE"
        or bridge.get("g0_receipt") != g0
        or bridge.get("technical_only") is not True
        or bridge.get("code_sha256") != code_bindings()
        or bridge.get("source_bound_hash") != canonical_hash(verify_source_bound(run))
        or bridge.get("physical_updates_started") != 32
        or bridge.get("physical_updates_completed") != 32
        or bridge.get("generations") != 0
        or bridge.get("E_CONFIRM2_accessed") is not False
        or any(bridge.get("checks", {}).get(k) is not True for k in REQUIRED_BRIDGE_CHECKS)
    ):
        raise ValueError("Only the complete source-bound CUDA bridge can register a mode")
    if audit.get("status") not in ("REUSE_CANDIDATE", "RERUN_CANDIDATE"):
        raise ValueError("Verified parents and historical scientific contract required")
    if bridge.get("status") not in ("PASS", "PASS_NEW_RUNTIME_REUSE_INCOMPATIBLE"):
        raise ValueError("The complete numerical bridge must pass before choosing execution mode")
    if any((run / folder).exists() for folder in ("training", "evaluations")):
        raise ValueError("Execution mode cannot be chosen after formal execution starts")
    mode = (
        "REUSE_12"
        if audit["status"] == "REUSE_CANDIDATE" and bridge.get("legacy_compatibility_pass") is True
        else "RERUN_24"
    )
    result = {
        "phase_id": EXPERIMENT_ID,
        "mode": mode,
        "status": "REGISTERED_BEFORE_FORMAL_RESULTS",
        "reuse_audit_sha256": file_digest(run / "REUSE_AUDIT.json"),
        "bridge_sha256": file_digest(run / "CONTINUITY_BRIDGE.json"),
        "source_by_logical_arm": {
            arm: ("LEGACY" if mode == "REUSE_12" and value["legacy_arm"] else "NEW_TRAINING")
            for arm, value in ARMS.items()
        },
        "formal_updates": 3072 if mode == "REUSE_12" else 6144,
        "formal_target_exposures": 49152 if mode == "REUSE_12" else 98304,
        "confirmation_answers": 206336,
    }
    _write_once(run / "EXECUTION_MODE.json", result)
    return result


def directional_hypotheses():
    return {
        "phase_id": EXPERIMENT_ID,
        "status": "REGISTERED_FROM_HISTORICAL_RESULTS_BEFORE_NEW_MODEL_OUTPUTS",
        "hypotheses": ["H3 < 0", "Psi_E > 0", "G_3_2 > 0", "G_3_3 > 0", "G_3_2 < G_3_3"],
        "interpretation": (
            "Registered corruption-configuration interaction; delta and exposure are not isolated"
        ),
        "selection_from_confirmation_forbidden": True,
    }


def structure_reference():
    cells = []
    for arm, info in ARMS.items():
        center, donor = info["center_0based"], info["corrupted_index_0based"]
        edges = sorted(tuple(sorted((center, leaf))) for leaf in range(4) if leaf != center)
        support = {i for i, edge in enumerate(edges) if donor in edge}
        for receiver_center in range(4):
            for corruption in range(4):
                violated = {i for i, edge in enumerate(edges) if corruption in edge}
                overlap = center == receiver_center and support <= violated
                cells.append(
                    {
                        "logical_arm_id": arm,
                        "receiver_center_0based": receiver_center,
                        "receiver_corrupted_index_0based": corruption,
                        "donor_failed_edges": sorted(support),
                        "same_center_support_subset": overlap,
                        "forward_leaf": donor < center,
                        "risk_reference": overlap and donor < center and corruption != donor,
                        "repair_opportunity": overlap and corruption == donor,
                    }
                )
    return {
        "phase_id": EXPERIMENT_ID,
        "parameter_fitting": False,
        "cells": cells,
        "limitation": (
            "Trigger overlap is not an interference theorem "
            "and does not imply other-leaf positive transfer"
        ),
    }


def training_jobs(mode):
    if mode not in ("REUSE_12", "RERUN_24"):
        raise ValueError("Unregistered mode")
    return [
        {
            "id": f"train.{parent}.{block}.{arm}",
            "job_id": f"train.{parent}.{block}.{arm}",
            "kind": "train",
            "parent": parent,
            "block": block,
            "block_seed": 108701 + block,
            "logical_arm_id": arm,
            "arm": arm,
            "updates": 256,
            "priority": 0,
        }
        for parent in PARENTS
        for block in range(3)
        for arm, info in ARMS.items()
        if mode == "RERUN_24" or info["legacy_arm"] is None
    ]


def build_training(design_path, legacy_run, run):
    from .data import (
        build_development_manifests,
        build_training_manifests,
        write_training_manifests,
    )
    from .legacy import REGISTERED_INPUT_SHA256

    run, legacy_run = Path(run).resolve(), Path(legacy_run).resolve()
    bind_design(design_path, run)
    fixed_sources = {}
    for registered, expected in REGISTERED_INPUT_SHA256.items():
        relative = registered.removeprefix("run_v1/")
        path = legacy_run / relative
        if file_digest(path) != expected:
            raise ValueError("Historical source differs from the exact design: " + relative)
        fixed_sources[relative] = {
            "relative_path": relative,
            "path": str(path),
            "sha256": expected,
            "bytes": path.stat().st_size,
        }
    old_machine = read_json(legacy_run / "machine.json")
    machine = {
        **old_machine,
        "phase_id": EXPERIMENT_ID,
        "legacy_run": str(legacy_run),
        "legacy_frozen_plan_sha256": file_digest(legacy_run / "FROZEN_PLAN.json"),
        "run_root": str(run),
        "repository_worktree": str(Path(__file__).resolve().parents[2]),
        "test_results_sealed": True,
        "automatic_dependency_upgrade": False,
    }
    _write_once(run / "machine.json", machine)
    files, audit = build_training_manifests(legacy_run)
    source = write_training_manifests(run, files, audit)
    development, dev_audit = build_development_manifests(legacy_run, files)
    write_manifests(run, development)
    _write_once(run / "DEVELOPMENT_DATA_AUDIT.json", dev_audit)
    _write_once(run / "DIRECTIONAL_HYPOTHESES.json", directional_hypotheses())
    _write_once(run / "STRUCTURE_OVERLAP_REFERENCE.json", structure_reference())
    _write_once(
        run / "SOURCE_PROVENANCE.json",
        {
            "phase_id": EXPERIMENT_ID,
            "legacy_run": str(legacy_run),
            "source_files": list(
                {**{r["relative_path"]: r for r in audit["source_files"]}, **fixed_sources}.values()
            ),
            "source_bound_hash": source["source_hash"],
            "input_design_sha256": DESIGN_SHA256,
            "input_prose_sha256": PLAN_SHA256,
            "old_files_modified": False,
            "new_confirmation_generated": False,
        },
    )
    return {
        "status": "CPU_TRAINING_AND_DEVELOPMENT_BUILT_NOT_FROZEN",
        "source_bound_hash": source["source_hash"],
        "model_calls": 0,
        "confirmation_generated": False,
    }


def cpu_check(run):
    from .data import build_development_manifests, build_training_manifests
    from .schema import read_jsonl, training_rows, verify_source_bound

    run = Path(run)
    validate_design(run / "SER_J23_DESIGN.json")
    source = verify_source_bound(run)
    machine = read_json(run / "machine.json")
    files, audit = build_training_manifests(machine["legacy_run"])
    dev, dev_audit = build_development_manifests(machine["legacy_run"], files)
    for relative, expected in {**files, **dev}.items():
        path = run / "manifests" / relative
        actual = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        if actual != expected:
            raise ValueError("CPU reconstruction differs: " + relative)
    for arm in ARMS:
        if len(training_rows(run, arm, technical=True)) != 278:
            raise ValueError("Training corpus size changed")
    result = {
        "schema": "ser-j23-cpu-check-v1",
        "phase_id": EXPERIMENT_ID,
        "status": "PASS",
        "source_bound_hash": source["source_hash"],
        "training": audit,
        "development": dev_audit,
        "model_calls": 0,
        "new_confirmation_generated": False,
        "tokenizer_and_numerical_bridge_required": True,
        "code_sha256": code_bindings(),
    }
    # CPU checks may be repeated before any GPU bridge; retain each code-bound receipt.
    _write_once(run / "cpu_checks" / (digest(result) + ".json"), result)
    from .schema import write_json

    if (run / "FROZEN_PLAN.json").exists():
        if read_json(run / "CPU_CHECK.json") != result:
            raise ValueError("CPU receipt changed after freeze")
    else:
        write_json(run / "CPU_CHECK.json", result)
    return result


def freeze(run, exclusion_audit_path, validation_receipt_path):
    """Auditor finalizes the single immutable plan after all numerical gates pass."""
    from ..modeling_v3.io import canonical_hash
    from .bridge import validate_g0
    from .data import build_confirmation_from_audit
    from .evaluate import build_evaluation_jobs, checkpoint_key, evaluation_job_id
    from .exclusions import load_verified_exclusions
    from .legacy_trajectory import audit_historical_trajectory
    from .schema import verify_source_bound
    from .training import REQUIRED_BRIDGE_CHECKS

    run = Path(run).resolve()
    if (run / "FROZEN_PLAN.json").exists():
        raise FileExistsError("A frozen plan is immutable; do not refreeze")
    validate_design(run / "SER_J23_DESIGN.json")
    source = verify_source_bound(run)
    current_code = code_bindings()
    cpu = read_json(run / "CPU_CHECK.json")
    validation = read_json(validation_receipt_path)
    if validation != read_json(run / "VALIDATION_RECEIPT.json"):
        raise ValueError("Freeze must use the same exact-code G0 validation as the bridge")
    g0 = validate_g0(run, source=source)
    if (
        cpu.get("status") != "PASS"
        or cpu.get("code_sha256") != current_code
        or validation.get("status") != "PASS"
        or validation.get("code_sha256") != current_code
        or validation.get("failures") != 0
        or validation.get("errors") != 0
    ):
        raise ValueError(
            "Fresh CPU reconstruction and integrated tests for this exact code required"
        )
    bridge = read_json(run / "CONTINUITY_BRIDGE.json")
    if (
        bridge.get("execution_kind") != "REAL_CUDA_BRIDGE"
        or bridge.get("g0_receipt") != g0
        or bridge.get("code_sha256") != current_code
        or bridge.get("technical_only") is not True
        or bridge.get("source_bound_hash") != canonical_hash(source)
        or bridge.get("physical_updates_started") != 32
        or bridge.get("physical_updates_completed") != 32
        or bridge.get("generations") != 0
        or bridge.get("E_CONFIRM2_accessed") is not False
        or any(bridge.get("checks", {}).get(key) is not True for key in REQUIRED_BRIDGE_CHECKS)
    ):
        raise ValueError("Complete exact-source numerical bridge required")
    reuse = read_json(run / "REUSE_AUDIT.json")
    bindings_path = run / "PARENT_AND_ENDPOINT_BINDINGS.json"
    if reuse.get("bindings_file", {}).get("sha256") != file_digest(bindings_path):
        raise ValueError("CPU checkpoint audit binding changed")
    bindings = read_json(bindings_path)
    mode = choose_mode(run)["mode"]
    historical_trajectory, historical_rows = audit_historical_trajectory(run, bindings, mode=mode)
    _write_once(run / "HISTORICAL_TRAJECTORY_REUSE.json", historical_trajectory)
    write_manifests(run, {"historical_trajectory_reused_rows.jsonl": historical_rows})
    lookup = bindings["checkpoint_lookup"]
    for parent in PARENTS:
        lookup[f"PARENT.{parent}.baseline.PARENT.step0"] = lookup["PARENT." + parent]
    preliminary = build_evaluation_jobs(mode)
    unavailable, selected = [], {}
    for job in preliminary:
        if job["source"] == "NEW_TRAINING":
            continue
        key = checkpoint_key(job)
        bound = lookup.get(key, {})
        if bound.get("status") != "VERIFIED_FULL_CPU_STATE":
            if job["panel"] not in ("DEV_DELTA", "DEV_TRAJECTORY") or job["source"] != "LEGACY":
                raise ValueError("Required confirmation/full-fit model unavailable: " + key)
            unavailable.append(
                {
                    "job_id": evaluation_job_id(job),
                    "reason": "Legacy CPU state binding unavailable before freeze",
                }
            )
        else:
            selected[key] = bound
    jobs = build_evaluation_jobs(mode, selected, unavailable=unavailable)
    load_verified_exclusions(exclusion_audit_path, verify_sources=True)
    exclusion = read_json(exclusion_audit_path)
    _write_once(run / "EXCLUSION_AUDIT.json", exclusion)
    _write_once(run / "VALIDATION_RECEIPT.json", validation)
    # Execution mode is now fixed. Only this auditor constructs new confirmation;
    # trainer and model-generation capabilities never receive the truth registry.
    confirm_files, confirmation_audit = build_confirmation_from_audit(
        exclusion_audit_path, auditor_authorized=True
    )
    write_manifests(run, confirm_files)
    _write_once(run / "CONFIRMATION_GENERATION_AUDIT.json", confirmation_audit)
    write_manifests(
        run,
        {
            "training_jobs.jsonl": training_jobs(mode),
            "evaluation_jobs.jsonl": jobs,
            "diagnostic_availability.json": {
                "unavailable": unavailable,
                "historical_step0_256_answer_reuse": "HISTORICAL_TRAJECTORY_REUSE.json",
                "no_unbudgeted_generations": True,
            },
        },
    )
    _write_once(
        run / "MACHINE_BINDINGS.json",
        {
            "phase_id": EXPERIMENT_ID,
            "machine_sha256": file_digest(run / "machine.json"),
            "runtime_environment": bindings["runtime_environment"],
            "base_snapshot": bindings["base_snapshot"],
            "bridge_gpu_runtime": bridge.get("runtime_receipts", bridge.get("runtime", {})),
            "isolation": (
                "Separate worker capabilities and path allowlists under the same OS account; "
                "not cryptographic secrecy"
            ),
        },
    )
    filenames = [
        "SER_J23_DESIGN.json",
        "DESIGN_SOURCE.md",
        "DESIGN_SOURCE_SHA256.json",
        "machine.json",
        "SOURCE_BOUND.json",
        "SOURCE_PROVENANCE.json",
        "TRAINING_DATA_AUDIT.json",
        "DEVELOPMENT_DATA_AUDIT.json",
        "CPU_CHECK.json",
        "VALIDATION_RECEIPT.json",
        "REUSE_AUDIT.json",
        "PARENT_AND_ENDPOINT_BINDINGS.json",
        "CONTINUITY_BRIDGE.json",
        "EXECUTION_MODE.json",
        "EXCLUSION_AUDIT.json",
        "CONFIRMATION_GENERATION_AUDIT.json",
        "DIRECTIONAL_HYPOTHESES.json",
        "STRUCTURE_OVERLAP_REFERENCE.json",
        "HISTORICAL_TRAJECTORY_REUSE.json",
        "GUARD_FEASIBILITY_ONLY.json",
        "MANUAL_FIXTURE_ORBIT_EXCLUSIONS.json",
        "MACHINE_BINDINGS.json",
    ]
    filenames.extend(
        str(path.relative_to(run)) for path in (run / "manifests").rglob("*") if path.is_file()
    )
    frozen = {
        "schema": "ser-j23-frozen-plan-v1",
        "phase_id": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN",
        "execution_mode": mode,
        "code_sha256": current_code,
        "files": {
            name: {"sha256": file_digest(run / name), "bytes": (run / name).stat().st_size}
            for name in sorted(filenames)
        },
        "new_confirmation_answers": 206336,
        "formal_training_jobs": len(training_jobs(mode)),
        "registered_generation_budget": sum(j["tasks"] * j["draws"] for j in jobs),
        "model_outputs_before_freeze": "TECHNICAL_ONLY_NO_CONFIRMATION",
    }
    frozen["plan_hash"] = digest(frozen)
    _write_once(run / "FROZEN_PLAN.json", frozen)
    return {
        "status": "FROZEN",
        "plan_hash": frozen["plan_hash"],
        "mode": mode,
        "formal_training_jobs": frozen["formal_training_jobs"],
        "formal_evaluation_jobs": len(jobs),
        "registered_generation_budget": frozen["registered_generation_budget"],
    }
