"""Frozen J23 training authorization with unchanged SER-J2 SFT numerics."""

from __future__ import annotations

from pathlib import Path

from ..exposure_substitution.training import (
    CHECKPOINT_STEPS as CHECKPOINT_STEPS,
)
from ..exposure_substitution.training import (
    MICROBATCH_SLOTS as MICROBATCH_SLOTS,
)
from ..exposure_substitution.training import (
    SCHEDULE_SEEDS as SCHEDULE_SEEDS,
)
from ..exposure_substitution.training import (
    _run_training_locked,
    latest_checkpoint,
    student_lock,
)
from ..exposure_substitution.training import (
    atomic_json as atomic_json,
)
from ..exposure_substitution.training import (
    encode_rows as encode_rows,
)
from ..exposure_substitution.training import (
    explicit_backward as explicit_backward,
)
from ..exposure_substitution.training import (
    measured_update as measured_update,
)

REQUIRED_BRIDGE_CHECKS = (
    "new_arm_resume_complete_state_equal",
    "new_arm_resume_all_update_fields_equal",
    "all_initial_states_empty_adam",
    "all_sequence_weights_one_sixteenth",
    "five_microbatches_observed",
    "actual_parameter_updates_observed",
    "matched_donor_target_eos_equal",
    "inference_weight_restore_exact",
)


def student_directory(run, parent, block, arm):
    from .schema import ARMS

    if (
        parent not in ("S96", "REP96")
        or type(block) is not int
        or block not in range(3)
        or arm not in ARMS
    ):
        raise ValueError("Unregistered J23 parent/block/logical arm")
    return Path(run).resolve() / "training" / f"{parent}_block{block}_{arm}"


def require_bridge(run, *, role="trainer"):
    """A CPU receipt or an unbound technical run cannot authorize formal work."""
    from ..modeling_v3.io import canonical_hash
    from .control import assert_code_bindings
    from .schema import read_bound, verify_frozen, verify_source_bound

    run = Path(run).resolve()
    if (run / "STOP").exists():
        raise RuntimeError("J23 STOP marker prohibits new formal execution")
    frozen = verify_frozen(run, role=role)
    assert_code_bindings(frozen)
    source = verify_source_bound(run, role="control")
    if read_bound(run, "SOURCE_BOUND.json") != source:
        raise ValueError("Source binding differs from the frozen J23 source receipt")
    bridge = read_bound(run, "CONTINUITY_BRIDGE.json")
    if (
        bridge.get("status") not in ("PASS", "PASS_NEW_RUNTIME_REUSE_INCOMPATIBLE")
        or bridge.get("execution_kind") != "REAL_CUDA_BRIDGE"
        or bridge.get("technical_only") is not True
        or not bridge.get("code_sha256")
        or bridge.get("code_sha256") != frozen.get("code_sha256")
        or bridge.get("source_bound_hash") != canonical_hash(source)
        or bridge.get("maximum_physical_updates") != 32
        or bridge.get("physical_updates_started") != 32
        or bridge.get("physical_updates_completed") != 32
        or bridge.get("generations") != 0
        or bridge.get("E_CONFIRM2_accessed") is not False
        or any(bridge.get("checks", {}).get(k) is not True for k in REQUIRED_BRIDGE_CHECKS)
    ):
        raise ValueError("Frozen completed 32-update real CUDA continuity bridge required")
    return bridge


def require_training_job(run, parent, block, arm):
    from .schema import ARMS, read_bound

    student_directory(run, parent, block, arm)
    bridge = require_bridge(run)
    mode_record = read_bound(run, "EXECUTION_MODE.json")
    mode = mode_record.get("mode")
    if mode not in ("REUSE_12", "RERUN_24"):
        raise ValueError("Formal execution mode is absent or BLOCKED")
    if mode == "REUSE_12" and bridge.get("legacy_compatibility_pass") is not True:
        raise ValueError("REUSE_12 requires complete old/new numerical compatibility")
    arms = tuple(ARMS) if mode == "RERUN_24" else ("A3_LOCAL_C1_J3", "B3_FORWARD_C4_J3")
    expected = {(p, b, a) for p in ("S96", "REP96") for b in range(3) for a in arms}
    rows = read_bound(run, "manifests/training_jobs.jsonl")
    if any(
        type(r.get("block")) is not int
        or r["block"] not in range(3)
        or r.get("updates") != 256
        or r.get("block_seed") != SCHEDULE_SEEDS[r["block"]]
        for r in rows
    ):
        raise ValueError("Registered training seed or update budget differs")
    found = [(r.get("parent"), r.get("block"), r.get("logical_arm_id", r.get("arm"))) for r in rows]
    if len(found) != len(expected) or set(found) != expected or len(set(found)) != len(found):
        raise ValueError(
            "Frozen training jobs must contain the complete selected mode exactly once"
        )
    key = parent, block, arm
    if key not in expected:
        raise PermissionError("This logical arm is not a new training job in the selected mode")
    return rows[found.index(key)]


def run_training(run, runtime, output, *, resume_path=None, checkpoint_callback=None):
    """Public runner always authorizes the frozen job before numerical execution."""
    from ..modeling_v3.io import canonical_hash
    from .schema import verify_frozen

    identity = runtime.identity
    require_training_job(run, identity["parent"], identity["block"], identity["logical_arm_id"])
    expected = student_directory(
        run, identity["parent"], identity["block"], identity["logical_arm_id"]
    )
    if (
        identity.get("technical_only") is not False
        or identity.get("frozen_plan_hash") != canonical_hash(verify_frozen(run, role="trainer"))
        or Path(output).resolve() != expected
    ):
        raise ValueError("Formal runtime/output must bind the current frozen J23 student")
    with student_lock(output):
        return _run_training_locked(
            runtime, output, resume_path=resume_path, checkpoint_callback=checkpoint_callback
        )


def train(
    run,
    parent,
    block,
    arm,
    machine=None,
    allow_gpu=False,
    *,
    resume_path=None,
    checkpoint_callback=None,
):
    from .runtime import build_runtime, load_parent_backend

    if allow_gpu is not True:
        raise PermissionError("J23 training requires --allow-gpu")
    run = Path(run).resolve()
    require_training_job(run, parent, block, arm)
    output = student_directory(run, parent, block, arm)
    with student_lock(output):
        backend = load_parent_backend(run, parent, machine=machine, allow_gpu=True)
        runtime = build_runtime(run, backend, parent, block, arm)
        return _run_training_locked(
            runtime,
            output,
            resume_path=resume_path or latest_checkpoint(output),
            checkpoint_callback=checkpoint_callback,
        )
