"""SR-F2 uses the unchanged SR-F1.2 loss and a fixed, identity-bound recipe."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

from mm_core.training import _cpu_tree, capture_rng, restore_rng, state_hash, trainable_state
from mm_dev.runtime import atomic_json
from sr_f1.runtime import copy_parameters, file_hash, read_json
from sr_f1.training import commit_checkpoint as _commit_checkpoint
from sr_f12 import training as previous

PLAN_ID = "SR-F2"
FAMILIES = ("CROSS", "INTERVAL", "THRESHOLD", "TOPK")
LEARNING_RATE = 5e-5
RECIPE = dict(
    learning_rate=LEARNING_RATE,
    microbatch_size=1,
    tis_cap=2.0,
    kl_coefficient=0.02,
    epochs_per_rollout=1,
)
ProbabilityMismatch = previous.ProbabilityMismatch
sequence_objective = previous.sequence_objective
enforce_probability_gate = previous.enforce_probability_gate
CHECKPOINT_FIELDS = previous.CHECKPOINT_FIELDS | {
    "adaptive_state",
    "adaptive_state_hash",
    "monitor_due_step",
    "monitor_definition_hash",
}


def initial_adaptive_state():
    return dict(version=1, last_monitor_step=None, baseline=None, active_families=[], history=[])


def validate_adaptive_state(state):
    if set(state) != {"version", "last_monitor_step", "baseline", "active_families", "history"}:
        raise PermissionError("Adaptive state schema differs")
    if state["version"] != 1:
        raise PermissionError("Adaptive state version differs")
    active = state["active_families"]
    if active != sorted(set(active)) or not set(active) <= set(FAMILIES):
        raise PermissionError("Invalid adaptive family latch")
    history = state["history"]
    if not history:
        if state != initial_adaptive_state():
            raise PermissionError("Unmeasured adaptive state contains decisions")
        return
    if [r["step"] for r in history] != list(range(0, history[-1]["step"] + 1, 8)):
        raise PermissionError("Adaptive monitor sequence is incomplete")
    if state["last_monitor_step"] != history[-1]["step"]:
        raise PermissionError("Adaptive monitor cursor differs")
    expected = initial_adaptive_state()
    for row in history:
        expected = apply_monitor(expected, row["measurement"], validate_previous=False)
    if state != expected:
        raise PermissionError("Adaptive state does not replay from its measurements")


def apply_monitor(state, measurement, *, validate_previous=True):
    """R1 latches at >15pp; R2 is recorded only, using exact decimal thresholds."""
    if validate_previous:
        validate_adaptive_state(state)
    step = measurement.get("step")
    expected = 0 if state["last_monitor_step"] is None else state["last_monitor_step"] + 8
    if type(step) is not int or step != expected or step > 96:
        raise PermissionError("Monitor must complete in registered step order")
    families = measurement.get("families", {})
    if set(families) != set(FAMILIES):
        raise PermissionError("Monitor must measure all four registered families")
    values = {}
    for family, metrics in families.items():
        values[family] = {}
        for key in ("A", "E", "P"):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PermissionError("Monitor probabilities must be finite numbers")
            try:
                exact = Fraction(str(value))
            except (ValueError, ZeroDivisionError):
                raise PermissionError("Nonfinite monitor probability") from None
            if not 0 <= exact <= 1:
                raise PermissionError("Monitor probabilities outside [0,1]")
            values[family][key] = exact
    baseline = state["baseline"] or copy.deepcopy(families)
    active = set(state["active_families"])
    decisions = {}
    for family, metrics in values.items():
        r1 = metrics["A"] - metrics["P"] > Fraction(15, 100)
        r2 = metrics["E"] <= Fraction(str(baseline[family]["E"])) - Fraction(1, 10) and metrics[
            "A"
        ] >= Fraction(str(baseline[family]["A"])) + Fraction(1, 10)
        newly = r1 and family not in active
        if r1:
            active.add(family)
        decisions[family] = dict(R1=r1, R2=r2, newly_latched=newly, active=family in active)
    return dict(
        version=1,
        last_monitor_step=step,
        baseline=copy.deepcopy(baseline),
        active_families=sorted(active),
        history=[
            *copy.deepcopy(state["history"]),
            dict(step=step, measurement=copy.deepcopy(measurement), decisions=decisions),
        ],
    )


@contextmanager
def isolated_rng(runtime=None):
    """Monitoring may fail or reseed; neither changes the training RNG or mode."""
    saved = capture_rng()
    mode = runtime.model.training if runtime is not None else None
    try:
        yield
    finally:
        restore_rng(saved)
        if runtime is not None:
            runtime.model.train(mode)
        if state_hash(capture_rng()) != state_hash(saved):
            raise RuntimeError("Monitoring changed the training RNG")


def update(
    runtime,
    optimizer,
    scheduler,
    records,
    coefficients,
    *,
    reference,
    microbatch_size=1,
    save_train_logprobs=False,
):
    if microbatch_size != 1:
        raise PermissionError("SR-F2 freezes gradient microbatch at one sequence")
    previous._validate_lr(optimizer.state_dict(), scheduler.state_dict(), LEARNING_RATE)
    return previous.update(
        runtime,
        optimizer,
        scheduler,
        records,
        coefficients,
        reference=reference,
        microbatch_size=1,
        save_train_logprobs=save_train_logprobs,
    )


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
    adaptive_state,
    monitor_definition_hash,
    microbatch_size=1,
    diagnostics_hash=None,
    token_path_hash=None,
):
    if microbatch_size != 1:
        raise PermissionError("Frozen microbatch differs")
    previous._validate_lr(optimizer.state_dict(), scheduler.state_dict(), LEARNING_RATE)
    validate_adaptive_state(adaptive_state)
    due = step if step % 8 == 0 else None
    expected_last = None if step == 0 else ((step - 1) // 8) * 8
    if adaptive_state["last_monitor_step"] != expected_last:
        raise PermissionError("Prior boundary monitor must complete before another update")
    state = previous.checkpoint_state(
        runtime,
        optimizer,
        scheduler,
        run_identity=run_identity,
        step=step,
        reference=reference,
        stream_hash=stream_hash,
        sampling_hash=sampling_hash,
        microbatch_size=1,
        diagnostics_hash=diagnostics_hash,
        token_path_hash=token_path_hash,
    )
    state.update(
        plan_id=PLAN_ID,
        adaptive_state=copy.deepcopy(adaptive_state),
        adaptive_state_hash=state_hash(adaptive_state),
        monitor_due_step=due,
        monitor_definition_hash=monitor_definition_hash,
    )
    return state


def commit_checkpoint(directory, state):
    if set(state) != CHECKPOINT_FIELDS or state["plan_id"] != PLAN_ID:
        raise PermissionError("Not an SR-F2 checkpoint")
    validate_adaptive_state(state["adaptive_state"])
    if state_hash(state["adaptive_state"]) != state["adaptive_state_hash"]:
        raise PermissionError("Adaptive checkpoint hash differs")
    committed = state["committed_logical_step"]
    last = None if committed == 0 else ((committed - 1) // 8) * 8
    if state["training_recipe"] != RECIPE or state["adaptive_state"]["last_monitor_step"] != last:
        raise PermissionError("Checkpoint recipe or monitor cursor differs")
    return _commit_checkpoint(directory, state)


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
    monitor_definition_hash,
    microbatch_size=1,
    learning_rate=LEARNING_RATE,
    step=None,
):
    import torch

    if microbatch_size != 1 or learning_rate != LEARNING_RATE:
        raise PermissionError("Frozen SR-F2 training recipe differs")
    directory = Path(directory).resolve()
    latest_path = directory / "LATEST.json"
    if step is None:
        markers = sorted(directory.glob("commit-[0-9][0-9].json"))
        if not markers:
            raise PermissionError("No immutable committed checkpoint exists")
        marker = markers[-1]
        receipt = read_json(marker)
        if marker.name != f"commit-{receipt['step']:02d}.json":
            raise PermissionError("Checkpoint marker step differs")
        if latest_path.exists():
            latest = read_json(latest_path)
            if latest["step"] > receipt["step"] or (
                latest["step"] == receipt["step"] and latest != receipt
            ):
                raise PermissionError("Latest pointer conflicts with immutable checkpoint")
    else:
        receipt = read_json(directory / f"commit-{step:02d}.json")
    path = (directory / receipt["path"]).resolve()
    if path.parent != directory or file_hash(path) != receipt["sha256"]:
        raise PermissionError("Checkpoint path/bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if set(state) != CHECKPOINT_FIELDS or state["plan_id"] != PLAN_ID:
        raise PermissionError("Checkpoint plan/schema differs")
    if state_hash(state) != receipt["state_hash"] or receipt["field_hashes"] != {
        key: state_hash(value) for key, value in state.items()
    }:
        raise PermissionError("Checkpoint field hashes differ")
    if any(
        state[key] != value
        for key, value in dict(
            run_identity=run_identity,
            input_stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            monitor_definition_hash=monitor_definition_hash,
            training_recipe=RECIPE,
        ).items()
    ):
        raise PermissionError("Checkpoint identity, stream, monitoring, or recipe differs")
    if state["reference_hash"] != state_hash(reference) or (
        state_hash(state["reference"]) != state_hash(reference)
    ):
        raise PermissionError("Checkpoint reference differs")
    committed = state["committed_logical_step"]
    if type(committed) is not int or not 0 <= committed <= 96 or committed != receipt["step"]:
        raise PermissionError("Checkpoint committed step differs")
    if state["cursor"] != dict(next_logical_step=committed + 1, next_slot=0, next_sample_index=0):
        raise PermissionError("Checkpoint cursor differs")
    if state["monitor_due_step"] != (committed if committed % 8 == 0 else None):
        raise PermissionError("Checkpoint pending monitor differs")
    validate_adaptive_state(state["adaptive_state"])
    last = None if committed == 0 else ((committed - 1) // 8) * 8
    if state["adaptive_state"]["last_monitor_step"] != last:
        raise PermissionError("Checkpoint has an unfinished prior monitor")
    if state_hash(state["adaptive_state"]) != state["adaptive_state_hash"]:
        raise PermissionError("Checkpoint adaptive hash differs")
    previous._validate_lr(state["optimizer"], state["scheduler"], LEARNING_RATE)
    if not state["parameters"] or not previous._finite_tensor_tree(state):
        raise PermissionError("Checkpoint state is empty or nonfinite")
    copy_parameters(runtime.model, state["parameters"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    actual = dict(
        parameters=trainable_state(runtime.model),
        optimizer=_cpu_tree(optimizer.state_dict()),
        scheduler=_cpu_tree(scheduler.state_dict()),
        rng=capture_rng(),
    )
    for key, value in actual.items():
        if state_hash(value) != state_hash(state[key]):
            raise RuntimeError(f"Restored {key} differs from the committed checkpoint")
    if step is None:
        # An immutable commit may survive a kill immediately before the latest-pointer rename.
        atomic_json(latest_path, receipt)
    return state
