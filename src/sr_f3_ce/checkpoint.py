"""Atomic, identity-bound recovery for SR-F3-CE's complete trainable state.

All trainable adapter tensors are included, together with Adam, the constant
scheduler, global and named RNG states, fixed reference tensors, and exact
input cursors. Immutable base-model files are bound by ``source_hashes``; the
checkpoint never silently substitutes another base or parent adapter.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

from mm_core.training import _cpu_tree, capture_rng, restore_rng, state_hash, trainable_state
from mm_dev.runtime import atomic_json
from sr_f1.runtime import copy_parameters, file_hash, read_json
from sr_f1.training import commit_checkpoint as _atomic_commit

from .training import (
    ARMS,
    PLAN_ID,
    RECIPE,
    finite_tree,
    validate_optimizer,
    validate_optimizer_state,
)

CHECKPOINT_FIELDS = frozenset(
    {
        "plan_id",
        "schema_version",
        "run_identity",
        "committed_logical_step",
        "parameters",
        "optimizer",
        "scheduler",
        "rng",
        "rng_stream_states",
        "reference",
        "reference_hash",
        "input_stream_hash",
        "sampling_hash",
        "source_hashes",
        "cursors",
        "used_task_qids",
        "used_support_qids",
        "committed_records_hash",
        "training_recipe",
        "microbatch_size",
        "step_statistics",
        "train_mode",
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value, label):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def capture_named_rng(streams=None):
    """Capture separately owned Random, NumPy Generator, or torch.Generator streams."""
    import random

    import numpy as np
    import torch

    states = {}
    for name, stream in (streams or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("RNG streams need nonempty names")
        if isinstance(stream, random.Random):
            states[name] = {"kind": "python.Random", "state": copy.deepcopy(stream.getstate())}
        elif isinstance(stream, np.random.Generator):
            states[name] = {
                "kind": "numpy.Generator",
                "bit_generator": type(stream.bit_generator).__name__,
                "state": copy.deepcopy(stream.bit_generator.state),
            }
        elif isinstance(stream, torch.Generator):
            states[name] = {
                "kind": "torch.Generator",
                "device": str(stream.device),
                "state": stream.get_state().cpu().clone(),
            }
        else:
            raise TypeError(f"Unsupported RNG stream {name}: {type(stream).__name__}")
    return states


def restore_named_rng(states, streams=None):
    actual = capture_named_rng(streams)
    if states.keys() != actual.keys():
        raise PermissionError("Named RNG stream inventory differs")
    for name, saved in states.items():
        if {key: value for key, value in saved.items() if key != "state"} != {
            key: value for key, value in actual[name].items() if key != "state"
        }:
            raise PermissionError(f"RNG stream {name} implementation differs")
    for name, saved in states.items():
        stream = streams[name]
        if saved["kind"] == "python.Random":
            stream.setstate(saved["state"])
        elif saved["kind"] == "numpy.Generator":
            stream.bit_generator.state = copy.deepcopy(saved["state"])
        else:
            stream.set_state(saved["state"])
    if state_hash(capture_named_rng(streams)) != state_hash(states):
        raise RuntimeError("Named RNG state restoration differs")


def checkpoint_cursors(step, arm):
    if arm not in ARMS or type(step) is not int or not 0 <= step <= 64:
        raise ValueError("Invalid step or arm")
    return {
        "next_logical_step": step + 1,
        "task": step * 16,
        "support": 0 if arm == "RL" else step * 4,
    }


def _validate_state(state):
    if set(state) != CHECKPOINT_FIELDS or state["plan_id"] != PLAN_ID:
        raise PermissionError("Checkpoint plan or complete-state schema differs")
    if state["schema_version"] != 1 or state["training_recipe"] != RECIPE:
        raise PermissionError("Checkpoint schema version or frozen recipe differs")
    identity = state["run_identity"]
    if not isinstance(identity, dict) or identity.get("arm") not in ARMS:
        raise ValueError("Run identity must include a registered arm")
    if not isinstance(identity.get("run_id"), str) or not identity["run_id"]:
        raise ValueError("Run identity must include run_id")
    if identity.get("stage") not in ("ENGINE", "SCIENCE"):
        raise ValueError("Run identity must explicitly distinguish ENGINE from SCIENCE")
    step, arm = state["committed_logical_step"], identity["arm"]
    if state["cursors"] != checkpoint_cursors(step, arm):
        raise PermissionError("Task/support/next-step cursors differ from the committed step")
    if identity["stage"] == "ENGINE" and step > 4:
        raise PermissionError("ENGINE is limited to four logical updates")
    if state["microbatch_size"] not in (4, 2, 1):
        raise PermissionError("Unregistered technical microbatch")
    for key, count in (
        ("used_task_qids", step * 16),
        ("used_support_qids", 0 if arm == "RL" else step * 4),
    ):
        values = state[key]
        if not isinstance(values, list) or len(values) != count:
            raise PermissionError(f"{key} does not match the consumed input stream")
        if any(not isinstance(value, str) or not value for value in values):
            raise PermissionError(f"{key} contains an invalid task identity")
        if len(set(values)) != len(values):
            raise PermissionError(f"{key} contains repeated task identities")
    if identity["stage"] == "SCIENCE" and set(state["used_task_qids"]) & set(
        state["used_support_qids"]
    ):
        raise PermissionError("Scientific RL and support pools overlap")
    for key in ("input_stream_hash", "sampling_hash", "reference_hash", "committed_records_hash"):
        _digest(state[key], key)
    if not isinstance(state["source_hashes"], dict) or not state["source_hashes"]:
        raise ValueError("Checkpoint must bind immutable model/data/code source hashes")
    for name, digest in state["source_hashes"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Source hash names must be nonempty")
        _digest(digest, name)
    if state_hash(state["reference"]) != state["reference_hash"]:
        raise PermissionError("Fixed parent reference hash differs")
    if not state["parameters"] or not state["reference"] or not finite_tree(state):
        raise FloatingPointError("Checkpoint trainable/reference state is empty or nonfinite")
    if type(state["train_mode"]) is not bool:
        raise ValueError("Model mode is not explicit")
    validate_optimizer_state(state["optimizer"], state["scheduler"])
    if state["scheduler"].get("last_epoch") != step:
        raise PermissionError("Constant scheduler does not match the committed update count")
    optimizer_steps = {int(value["step"]) for value in state["optimizer"]["state"].values()}
    if (step == 0 and optimizer_steps) or (step > 0 and optimizer_steps != {step}):
        raise PermissionError("Adam physical state does not match the logical checkpoint step")
    if not isinstance(state["step_statistics"], dict):
        raise ValueError("Current-step statistics must be retained")
    if identity["stage"] == "SCIENCE" and (
        state["step_statistics"].get("engineering_zero_aux", False)
        or state["step_statistics"].get("scientific_recipe_eligible") is False
    ):
        raise PermissionError("Engineering-only gradients/updates cannot be committed as science")


def checkpoint_state(
    runtime,
    optimizer,
    scheduler,
    *,
    run_identity,
    step,
    reference,
    stream_hash,
    sampling_hash,
    source_hashes,
    used_task_qids,
    used_support_qids,
    committed_records_hash,
    microbatch_size,
    step_statistics,
    cursors=None,
    rng_streams=None,
):
    """Snapshot without advancing a cursor or writing a completion ledger.

    ``committed_records_hash`` binds the ordered raw/update records through this
    checkpoint. Call ``commit_checkpoint`` before marking those updates durable.
    """
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    state = {
        "plan_id": PLAN_ID,
        "schema_version": 1,
        "run_identity": copy.deepcopy(run_identity),
        "committed_logical_step": step,
        "parameters": trainable_state(runtime.model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng(),
        "rng_stream_states": capture_named_rng(rng_streams),
        "reference": _cpu_tree(reference),
        "reference_hash": state_hash(reference),
        "input_stream_hash": stream_hash,
        "sampling_hash": sampling_hash,
        "source_hashes": copy.deepcopy(source_hashes),
        "cursors": copy.deepcopy(cursors)
        if cursors is not None
        else checkpoint_cursors(step, run_identity["arm"]),
        "used_task_qids": list(used_task_qids),
        "used_support_qids": list(used_support_qids),
        "committed_records_hash": committed_records_hash,
        "training_recipe": copy.deepcopy(RECIPE),
        "microbatch_size": microbatch_size,
        "step_statistics": copy.deepcopy(step_statistics),
        "train_mode": runtime.model.training,
    }
    _validate_state(state)
    return state


def commit_checkpoint(directory, state):
    """State bytes -> immutable commit marker -> atomic latest pointer.

    Existing checkpoints and orphaned bytes survive; an already committed step
    is idempotent only for identical content. The returned receipt is the only
    authority for advancing an external completion ledger.
    """
    _validate_state(state)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    markers = sorted(directory.glob("commit-[0-9][0-9].json"))
    if markers and int(markers[-1].stem.split("-")[-1]) > state["committed_logical_step"]:
        raise PermissionError("Cannot move the latest committed step backwards")
    return _atomic_commit(directory, state)


def read_checkpoint(directory, *, step=None):
    """Read only; authenticate bytes and all fields before any live state changes."""
    import torch

    directory = Path(directory).resolve()
    if step is None:
        markers = sorted(directory.glob("commit-[0-9][0-9].json"))
        if not markers:
            raise PermissionError("No immutable committed checkpoint exists")
        marker = markers[-1]
    else:
        if type(step) is not int or not 0 <= step <= 64:
            raise ValueError("Invalid requested checkpoint step")
        marker = directory / f"commit-{step:02d}.json"
    receipt = read_json(marker)
    if marker.name != f"commit-{receipt['step']:02d}.json":
        raise PermissionError("Checkpoint marker and step disagree")
    if set(receipt) != {"step", "path", "sha256", "state_hash", "field_hashes"}:
        raise PermissionError("Checkpoint commit receipt schema differs")
    relative = Path(receipt["path"])
    path = (directory / relative).resolve()
    if relative.name != str(relative) or path.parent != directory:
        raise PermissionError("Checkpoint receipt escapes its directory")
    if file_hash(path) != receipt["sha256"]:
        raise PermissionError("Checkpoint bytes differ from the committed hash")
    state = torch.load(path, map_location="cpu", weights_only=False)
    _validate_state(state)
    if state["committed_logical_step"] != receipt["step"]:
        raise PermissionError("Checkpoint logical step differs from receipt")
    if state_hash(state) != receipt["state_hash"] or receipt["field_hashes"] != {
        key: state_hash(value) for key, value in state.items()
    }:
        raise PermissionError("Checkpoint complete state/field hashes differ")
    latest = directory / "LATEST.json"
    if step is None and latest.exists():
        pointer = read_json(latest)
        if pointer["step"] > receipt["step"] or (
            pointer["step"] == receipt["step"] and pointer != receipt
        ):
            raise PermissionError("Latest pointer conflicts with immutable commit")
    return state, receipt


def load_checkpoint(
    directory,
    runtime,
    optimizer,
    scheduler,
    *,
    run_identity,
    reference,
    stream_hash,
    sampling_hash,
    source_hashes,
    committed_records_hash=None,
    rng_streams=None,
    step=None,
):
    """Verify identity before restoring; restore the prior live state on any error."""
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    state, receipt = read_checkpoint(directory, step=step)
    expected = {
        "run_identity": run_identity,
        "input_stream_hash": stream_hash,
        "sampling_hash": sampling_hash,
        "source_hashes": source_hashes,
        "reference_hash": state_hash(reference),
    }
    if committed_records_hash is not None:
        expected["committed_records_hash"] = committed_records_hash
    if any(state[key] != value for key, value in expected.items()):
        raise PermissionError("Checkpoint identity, source, input stream, or reference differs")
    live_parameters = trainable_state(runtime.model)
    if state["parameters"].keys() != live_parameters.keys() or any(
        state["parameters"][name].shape != tensor.shape
        or state["parameters"][name].dtype != tensor.dtype
        for name, tensor in live_parameters.items()
    ):
        raise PermissionError("Complete trainable parameter inventory/dtype/shape differs")
    previous = {
        "parameters": live_parameters,
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng(),
        "rng_stream_states": capture_named_rng(rng_streams),
        "train_mode": runtime.model.training,
    }

    def apply(value):
        copy_parameters(runtime.model, value["parameters"])
        optimizer.load_state_dict(copy.deepcopy(value["optimizer"]))
        scheduler.load_state_dict(copy.deepcopy(value["scheduler"]))
        optimizer.zero_grad(set_to_none=True)
        restore_rng(value["rng"])
        restore_named_rng(value["rng_stream_states"], rng_streams)
        runtime.model.train(value["train_mode"])

    try:
        apply(state)
        actual = {
            "parameters": trainable_state(runtime.model),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(),
            "rng_stream_states": capture_named_rng(rng_streams),
        }
        for key, value in actual.items():
            if state_hash(value) != state_hash(state[key]):
                raise RuntimeError(f"Restored {key} differs from committed state")
        validate_optimizer(optimizer, scheduler, model=runtime.model)
    except BaseException:
        apply(previous)
        raise
    if step is None:
        # A durable marker may survive a lease kill immediately before this pointer.
        atomic_json(Path(directory).resolve() / "LATEST.json", receipt)
    return state


def _logical_step_statistics(statistics):
    """Retain losses, diagnostics and cost counts; omit measured wall times only."""
    timing_names = {
        "training_seconds",
        "wall_seconds",
        "diagnostic_backward_host_wall_seconds",
        "update_backward_host_wall_seconds",
    }
    excluded = []

    def visit(value, path):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child = f"{path}.{key}"
                if key in timing_names:
                    excluded.append(child)
                else:
                    result[key] = visit(item, child)
            return result
        if isinstance(value, list):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    return visit(statistics, "step_statistics"), excluded


def compare_recovery_states(continuous, resumed):
    """Compare complete logical state; omit measured wall times from equality.

    This is evidence only, not a GPU gate receipt. The caller must separately
    prove the model/backend, new-process boundary, cached rollout identity, and
    at-most-six physical ENGINE updates. No tolerance is chosen after results.
    """
    _validate_state(continuous)
    _validate_state(resumed)
    logical_continuous, logical_resumed = dict(continuous), dict(resumed)
    logical_continuous["step_statistics"], left_excluded = _logical_step_statistics(
        continuous["step_statistics"]
    )
    logical_resumed["step_statistics"], right_excluded = _logical_step_statistics(
        resumed["step_statistics"]
    )
    comparisons = {
        key: state_hash(logical_continuous[key]) == state_hash(logical_resumed[key])
        for key in sorted(CHECKPOINT_FIELDS)
    }
    adapter_differences = {}
    if continuous["parameters"].keys() == resumed["parameters"].keys():
        for name, left in continuous["parameters"].items():
            right = resumed["parameters"][name]
            if left.shape != right.shape:
                continue
            delta = (left.double() - right.double()).abs()
            bound = 2e-6 + 2e-5 * left.double().abs()
            adapter_differences[name] = {
                "max_absolute_difference": float(delta.max()),
                "within_registered_sensitivity_bound": bool((delta <= bound).all()),
            }
    return {
        "all_logical_fields_exact": all(comparisons.values()),
        "comparisons": comparisons,
        "field_hashes": {
            key: {
                "continuous": state_hash(logical_continuous[key]),
                "resumed": state_hash(logical_resumed[key]),
            }
            for key in sorted(CHECKPOINT_FIELDS)
        },
        "adapter_sensitivity_measurements": adapter_differences,
        "sensitivity_bound": {"absolute": 2e-6, "relative": 2e-5},
        "sensitivity_is_automatic_pass": False,
        "excluded_cost_fields": sorted(set(left_excluded + right_excluded)),
        "step_statistics_comparison": "all_fields_except_measured_wall_times",
        "raw_step_statistics_exact": state_hash(continuous["step_statistics"])
        == state_hash(resumed["step_statistics"]),
        "actual_gpu_execution_confirmed": False,
    }
