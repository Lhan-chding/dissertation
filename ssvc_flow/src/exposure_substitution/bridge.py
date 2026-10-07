"""One bounded real GPU bridge: eight uninterrupted versus four + restore + four."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .runtime import SERRuntime, build_runtime, load_parent_backend, load_student_into_backend
from .training import atomic_json, measured_update


def bridge_runtime(runtime, output):
    """Shared mechanics for CPU contract fixtures and the independently labelled GPU bridge."""
    from ..optimizer_fork import state_hash

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    is_cuda = str(runtime.adapter.device).startswith("cuda")
    receipt = {
        "status": "RUNNING",
        "execution_kind": "REAL_CUDA_BRIDGE" if is_cuda else "CPU_TEST_FIXTURE",
        "identity": runtime.identity,
        "maximum_physical_updates": 16,
        "generations": 0,
        "E_CONFIRM_accessed": False,
    }
    atomic_json(output / "BRIDGE_RESULT.json", receipt)
    origin = runtime.capture()
    zero = runtime.save(output / "step000.pt")
    physical = 0
    rows = []

    def update(value, branch):
        nonlocal physical
        if physical >= 16:
            raise ValueError("The single technical bridge exhausted its 16-update allowance")
        physical += 1
        path = output / "attempts" / f"physical{physical:02d}.json"
        atomic_json(
            path,
            {
                "status": "UPDATE_STARTED",
                "physical_update": physical,
                "branch": branch,
                "update": value.step + 1,
                "physical_cost_complete": False,
            },
        )
        result = measured_update(value)
        row = {
            **result,
            "status": "UPDATE_APPLIED",
            "physical_update": physical,
            "branch": branch,
            "physical_cost_complete": True,
        }
        atomic_json(path, row)
        rows.append(row)
        return result

    try:
        continuous_losses = [update(runtime, "continuous") for _ in range(8)]
        continuous = runtime.capture()
        continuous_checkpoint = runtime.save(output / "continuous_step008.pt")
        runtime.restore(origin)
        interrupted_losses = [update(runtime, "resumed") for _ in range(4)]
        four = runtime.save(output / "resume_step004.pt")
        # Change live weights back to origin so resume must actually overwrite them.
        runtime.restore(origin)
        # A distinct optimizer/runtime and checkpoint deserialization exercise the cold resume path.
        sampler = copy.deepcopy(runtime.sampler)
        sampler.load_state_dict(origin["sampler"])
        resumed = SERRuntime(
            runtime.adapter,
            seed=runtime.seed,
            identity=runtime.identity,
            parameter_names=runtime.parameter_names,
            sampler=sampler,
            encoded=runtime.encoded,
            metadata=runtime.metadata,
        )
        resumed.resume(four["path"])
        interrupted_losses.extend(update(resumed, "resumed") for _ in range(4))
        recovered = resumed.capture()
        recovered_checkpoint = resumed.save(output / "resumed_step008.pt")
        compared_fields = (
            "loss",
            "sequence_nll",
            "task_ids",
            "slots",
            "sequence_target_tokens",
            "microbatch_shapes",
            "gradient_norm",
            "parameter_delta_norm",
            "lr",
            "role_exposures",
        )
        details = {
            "parameters_bitwise_equal": state_hash(continuous["parameters"])
            == state_hash(recovered["parameters"]),
            "optimizer_bitwise_equal": state_hash(continuous["optimizer"])
            == state_hash(recovered["optimizer"]),
            "rng_bitwise_equal": state_hash(continuous["rng"]) == state_hash(recovered["rng"]),
            "complete_state_bitwise_equal": state_hash(continuous) == state_hash(recovered),
            "schedule_cursor_equal": continuous["sampler"] == recovered["sampler"],
            "loss_slots_counts_equal": all(
                all(left[k] == right[k] for k in compared_fields)
                for left, right in zip(continuous_losses, interrupted_losses, strict=True)
            ),
            "actual_parameter_updates_observed": any(r["parameter_delta_norm"] > 0 for r in rows),
            "five_microbatches_observed": all(
                [s[0] for s in r["microbatch_shapes"]] == [4, 4, 3, 1, 4] for r in rows
            ),
            "all_sequence_weights_one_sixteenth": all(
                r["sequence_coefficient"] == 1 / 16 for r in rows
            ),
            "optimizer_steps": sorted(
                {int(v["step"]) for v in recovered["optimizer"]["state"].values()}
            ),
            "max_parameter_abs_difference": max(
                float((continuous["parameters"][k] - recovered["parameters"][k]).abs().max())
                for k in continuous["parameters"]
            ),
        }
        checks = [v for v in details.values() if type(v) is bool]
        if details["optimizer_steps"] != [8]:
            checks.append(False)
        receipt.update(
            status=("RESTORE_COMPARISON_PASS" if is_cuda else "PASS")
            if all(checks)
            else "BLOCKED_TECHNICAL",
            checks=details,
            physical_updates_started=physical,
            physical_updates_completed=len(rows),
            processed_target_sequences=sum(r["processed_target_sequences"] for r in rows),
            update_gpu_seconds=sum(r["update_gpu_seconds"] or 0 for r in rows),
            update_wall_seconds=sum(r["update_wall_seconds"] for r in rows),
            checkpoints={
                "origin": zero,
                "continuous": continuous_checkpoint,
                "resume4": four,
                "resumed": recovered_checkpoint,
            },
            physical_cost_complete=True,
        )
        atomic_json(output / "BRIDGE_RESULT.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(
            status="BLOCKED_TECHNICAL",
            error=repr(exc),
            physical_updates_started=physical,
            physical_updates_completed=len(rows),
            physical_cost_complete=False,
        )
        atomic_json(output / "BRIDGE_RESULT.json", receipt)
        raise


def run_bridge(run, parent="S96", machine=None, allow_gpu=False):
    from ..optimizer_fork import state_hash
    from .schema import training_rows
    from .training import encode_rows

    if allow_gpu is not True:
        raise PermissionError("The SER-J2 real GPU restore bridge requires --allow-gpu")
    run = Path(run).resolve()
    output = run / "bridge"
    if output.exists():
        result_path = output / "BRIDGE_RESULT.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text())
            if (
                existing.get("status") == "PASS"
                and existing.get("execution_kind") == "REAL_CUDA_BRIDGE"
            ):
                return existing
        raise ValueError(
            "Single bridge already started; preserve its budget journal rather than rerun"
        )
    backend = load_parent_backend(run, parent, machine=machine, allow_gpu=True)
    # Encode all three matched donor variants before model updates; compare exact target+EOS.
    tokens_by_root = {}
    for arm in ("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3"):
        rows = {
            key: value for key, value in training_rows(run, arm).items() if value["role"] == "donor"
        }
        encoded = encode_rows(backend, rows)
        for key, row in rows.items():
            root = row["root_id"]
            token_ids = encoded[key].target_ids
            if root in tokens_by_root and tokens_by_root[root] != token_ids:
                raise ValueError("Matched donor target/EOS token bytes differ across arms")
            tokens_by_root[root] = token_ids
    if len(tokens_by_root) != 32:
        raise ValueError("Expected exactly 32 matched donor roots")
    runtime = build_runtime(run, backend, parent, 0, "A_LOCAL_C1", technical=True)
    result = bridge_runtime(runtime, output)
    if result["status"] == "RESTORE_COMPARISON_PASS":
        try:
            from .runtime import read_student_checkpoint

            checkpoint = result["checkpoints"]["resumed"]["path"]
            expected = read_student_checkpoint(checkpoint)["parameters"]
            # The inference loader must restore different live weights, not copy a no-op.
            runtime.resume(result["checkpoints"]["origin"]["path"])
            load_student_into_backend(backend, checkpoint, runtime.identity)
            live = {
                name: value.detach().cpu()
                for name, value in backend.adapter.model.named_parameters()
                if name in expected
            }
            inference_restored = state_hash(expected) == state_hash(live) and all(
                not p.requires_grad for p in backend.adapter.model.parameters()
            )
            result["checks"]["inference_weight_restore_exact"] = inference_restored
            result["checks"]["matched_donor_target_eos_equal"] = True
            result["parent_receipt"] = runtime.parent_receipt
            result["status"] = "PASS" if inference_restored else "BLOCKED_TECHNICAL"
            atomic_json(output / "BRIDGE_RESULT.json", result)
        except BaseException as exc:
            result.update(
                status="BLOCKED_TECHNICAL", error=repr(exc), failed_phase="inference_restore"
            )
            atomic_json(output / "BRIDGE_RESULT.json", result)
            raise
    return result
