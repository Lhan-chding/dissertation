"""Read-only CPU prerequisite/input checks before requesting a downstream GPU job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import canonical_hash, write_json
from .next_stage_common import load_yaml
from .next_stage_runtime import (
    validate_config_against_gate,
    validate_prerequisites,
    validate_r2_gate,
    validate_r3_cold_gate,
    validate_r4_gate,
    validate_runtime_environment,
)
from .r2_runtime import _source
from .r3_runtime import _load_plan as _cold_plan


def preflight(
    config,
    data_root,
    r0_dir,
    r1_run,
    supplement_dir,
    r2_dir,
    *,
    phase,
    r3_dir=None,
    r4_dir=None,
    continuation_decision=None,
):
    if phase not in ("R3-cold", "R4", "R3-warm", "R4-continuation"):
        raise ValueError(
            "Only R3-cold, R4, R3-warm or R4-continuation phase preflight is implemented"
        )
    if phase in ("R4", "R3-warm", "R4-continuation") and r3_dir is None:
        raise ValueError("Downstream preflight requires completed R3-cold evidence")
    if phase == "R3-warm" and r4_dir is None:
        raise ValueError("R3-warm preflight requires completed R4 evidence")
    if phase == "R4-continuation":
        if r4_dir is None:
            raise ValueError("R4-continuation preflight requires the stopped parent R4 stage")
        if continuation_decision is None:
            raise ValueError("R4-continuation preflight requires an explicit continuation decision")
    elif continuation_decision is not None:
        raise ValueError("A continuation decision is only valid for R4-continuation preflight")
    gate = validate_prerequisites(r0_dir, r1_run, supplement_dir)
    config = validate_config_against_gate(config, gate)
    if Path(data_root).resolve() != Path(config["data_root"]).resolve():
        raise ValueError("Preflight data root differs from the certified data")
    r2 = validate_r2_gate(r2_dir, gate)
    environment = validate_runtime_environment(gate)
    if environment != r2["environment"]:
        raise ValueError("Preflight environment differs from measured R2")
    binding = {"r0_r1": gate["binding"], "r2": r2}
    if phase in ("R4", "R3-warm", "R4-continuation"):
        binding["r3_cold"] = validate_r3_cold_gate(r3_dir, gate, r2)
        if binding["r3_cold"]["environment"] != environment:
            raise ValueError("Preflight environment differs from measured R3-cold")
    if phase in ("R3-cold", "R3-warm"):
        plan, data = _cold_plan(data_root, gate)
        budgets = {
            "new_outputs": 1152,
            "train": 384,
            "control_proposal": 768,
            "candidate_optimizer_updates": 60,
            "lambda_zero_replays": 12,
            "reuse_validation_optimizer_updates": 8,
            "control_sequence_scores": 7680,
            "direct_resample_outputs": 0,
        }
        if phase == "R3-warm":
            if plan["plan_hash"] != binding["r3_cold"]["plan_hash"]:
                raise ValueError("Warm train/control plan differs from the measured cold plan")
            binding["r4"] = validate_r4_gate(r4_dir, gate, r2, binding["r3_cold"])
            if binding["r4"]["environment"] != environment:
                raise ValueError("Preflight environment differs from measured R4")
            budgets.update({"new_outputs": 4224, "direct_resample_outputs": 3072})
    else:
        from .r4_runtime import _load_plan

        plan, data, _ = _load_plan(data_root, gate, r0_dir)
        budgets = {
            "new_outputs": 14400,
            "train": 4096,
            "shared_step0": 576,
            "step32": 1152,
            "step64": 8576,
            "optimizer_updates": 128,
            "control_proposal_new_outputs": 0,
            "fixed_control_sequence_scores": 6144,
            "preupdate_parity_sequence_scores": 4096,
            "postupdate_training_sequence_scores": 4096,
        }
    source = _source()
    continuation_budgets = {}
    if phase == "R4-continuation":
        from .r4_continuation import validate_continuation_decision
        from .r4_gate import audit_stopped_r4

        parent = audit_stopped_r4(r4_dir, gate, r2, binding["r3_cold"])
        if parent["environment"] != environment:
            raise ValueError("Preflight environment differs from measured R4 parent")
        if parent["plan_hash"] != plan["plan_hash"]:
            raise ValueError("Continuation plan differs from the measured R4 parent")
        binding["r4_parent"] = parent
        binding["continuation_decision"] = validate_continuation_decision(
            continuation_decision, parent, source
        )
        inherited = dict(parent["inherited_counts"])
        totals = {
            "new_outputs": budgets["new_outputs"],
            "optimizer_updates": budgets["optimizer_updates"],
            "training_rollouts": budgets["train"],
            "shared_step0_outputs": budgets["shared_step0"],
            "step32_outputs": budgets["step32"],
            "step64_outputs": budgets["step64"],
        }
        if set(inherited) != set(totals) or any(
            type(inherited[k]) is not int or not 0 <= inherited[k] <= total
            for k, total in totals.items()
        ):
            raise ValueError("R4 parent inherited counts differ from the fixed continuation budget")
        continuation_budgets = {
            "inherited_budgets": inherited,
            "remaining_budgets": {k: total - inherited[k] for k, total in totals.items()},
        }
    return {
        "status": "PASS",
        "phase": phase + "_PREFLIGHT",
        "execution_kind": "CPU_AUDIT",
        "model_loaded": False,
        "gpu_work_requested": False,
        "scope": (
            "Prerequisite files, CUDA evidence, fixed inputs and installed environment; "
            "actual loaded-model checks still run inside the GPU job"
        ),
        "source": source,
        "config_hash": canonical_hash(config),
        "gate_binding": binding,
        "data_binding": data,
        "environment": environment,
        "plan_hash": plan["plan_hash"],
        "plan_counts": plan["counts"],
        "budgets": budgets,
        **continuation_budgets,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("R3-cold", "R4", "R3-warm", "R4-continuation"), required=True
    )
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    for name in ("data-root", "r0-dir", "r1-run", "supplement-dir", "r2-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--r3-dir", type=Path)
    parser.add_argument("--r4-dir", type=Path)
    parser.add_argument("--continuation-decision", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            "Preflight requires a new output record; previous evidence is preserved"
        )
    result = preflight(
        load_yaml(args.config),
        args.data_root,
        args.r0_dir,
        args.r1_run,
        args.supplement_dir,
        args.r2_dir,
        phase=args.phase,
        r3_dir=args.r3_dir,
        r4_dir=args.r4_dir,
        continuation_decision=args.continuation_decision,
    )
    write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase": result["phase"],
                "execution_kind": result["execution_kind"],
                "out": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
