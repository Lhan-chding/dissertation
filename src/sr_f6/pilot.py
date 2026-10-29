"""Registered twelve-step pilot execution, retaining every physical attempt.

The production entry verifies actual source, admission and registration before
loading a model. A separate CPU-only fixture entry exercises the same transaction
loop but can only publish SYNTHETIC_ONLY. No caller-supplied step budget, reward,
seed, reference adapter, update callback or access override is a production API.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path

from mm_core.training import state_hash, trainable_state

from . import pilot_contract, rollouts, training
from .census_evidence import _strict_json
from .common import now
from .transactions import RunTransactions, _plain_path, _read_bytes, _require, _write_json

SCHEMA = "srf6-pilot-execution-v1"
PHASE = "PILOT_V1"


def _zero_sampler():
    return dict(
        next_step=1,
        committed_log_step=0,
        consumed_qids={},
        logical_calls=0,
        actual_calls=0,
        rollout_cursor=dict(step=0, groups=0, rows=0),
        rollout_rng_reference=None,
    )


def _reference(root, path):
    path = _plain_path(path)
    return dict(
        path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(_read_bytes(path)).hexdigest()
    )


def _write(path, value):
    path = _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)
    return path


def _task_id(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value),
        "A safe, unique physical execution task ID is required",
    )
    return value


def _audit_position(runtime):
    """Capture only metadata; do not rescan prior numerical events per step."""
    _, path = training._audit_path(runtime)
    if not path.exists():
        return dict(path=path, offset=0, inode=None)
    info = path.stat()
    return dict(path=path, offset=info.st_size, inode=info.st_ino)


def _freeze_preupdate(tx, runtime, position, update, step):
    """Freeze the actual event appended BEFORE Adam, never manufacture a PASS."""
    path = _plain_path(position["path"])
    _require(path == training._audit_path(runtime)[1], "Numerical audit destination changed")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(position["inode"] in (None, before.st_ino), "Numerical audit log was replaced")
        _require(
            before.st_size > position["offset"],
            "Update did not append its actual preupdate evidence",
        )
        stream.seek(position["offset"])
        payload = stream.read()
        after = os.fstat(stream.fileno())
        _require(
            (before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns),
            "Numerical audit changed while binding",
        )
    lines = payload.splitlines(keepends=True)
    _require(
        len(lines) == 1 and lines[0].endswith(b"\n"),
        "Exactly one complete event from this update required",
    )
    event = _strict_json(lines[0])
    _require(
        event.get("status") == "PREUPDATE_PASS"
        and event.get("task_id") == runtime.identity["task_id"]
        and event.get("run_id") == tx.identity["run_id"]
        and type(event.get("committed_source_step")) is int
        and event["committed_source_step"] == step - 1
        and event.get("policy_before_sha256") == update.get("policy_before_sha256")
        and event.get("arm") == tx.identity["arm"]
        and event.get("audit_root") == str(tx.directory),
        "Actual preupdate event identity differs",
    )
    _require(
        event.get("channel_gradients") == update.get("channel_gradients"),
        "Actual measured diagnostic channels differ",
    )
    _require(
        isinstance(event.get("diagnostics"), dict)
        and event["diagnostics"].get("status") == "FINITE"
        and update.get("status") == "UPDATED_UNCOMMITTED"
        and all(
            update.get(key) == value
            for key, value in event["diagnostics"].items()
            if key != "status"
        ),
        "Actual numerical event differs from returned update facts",
    )
    task = _task_id(runtime.identity["task_id"])
    destination = tx.directory / "PREUPDATE_EVIDENCE" / f"step-{step:03d}" / f"{task}.json"
    saved = {
        **event,
        "source_audit_event": dict(
            path=path.relative_to(tx.directory).as_posix(),
            byte_offset=position["offset"],
            byte_count=len(lines[0]),
            sha256=hashlib.sha256(lines[0]).hexdigest(),
            binding="Original append-only event bytes; later events do not change this slice",
        ),
    }
    _write(destination, saved)
    return _reference(tx.directory, destination)


def _drive(
    tx,
    runtime,
    optimizer,
    scheduler,
    *,
    experiment_root,
    registration_ref,
    access_factory,
    journal_factory,
    scope,
    storage_lane,
    recovery_binding_ref=None,
    failpoint=None,
):
    """Shared loop; callers establish production access or explicit CPU scope."""
    tx._locked()
    tx._runtime(runtime)
    _require(
        tx.identity["phase"] == pilot_contract.for_phase(runtime.identity["phase"]).phase
        and tx.identity["max_steps"] == 12
        and tx.identity["repeat"] == 0
        and tx.identity["arm"] in ("U10", "U11", "C_SHUF"),
        "Exactly the registered pilot matrix is executable",
    )
    # Freeze the original parent BEFORE loading any resumed policy/Adam state.
    reference = trainable_state(runtime.model)
    _require(
        state_hash(reference)
        == tx.identity["reference_sha256"]
        == runtime.identity["reference_hash"],
        "Freshly loaded original parent must define the frozen KL reference",
    )
    _require(
        runtime.identity["step"] == 0 and not optimizer.state,
        "A new process must load its original parent and fresh optimizer before restore",
    )
    runtime.identity["diagnostic_row_indices"] = list(tx.identity["diagnostic_row_indices"])
    restored = tx.restore_latest(runtime, optimizer, scheduler)
    if restored is None:
        _require(
            storage_lane == "PRIMARY", "Recovery cannot start without its verified step4 prefix"
        )
        tx.commit(runtime, optimizer, scheduler, step=0, sampler_state=_zero_sampler())
    starting_step = runtime.identity["step"]
    _require(storage_lane in ("PRIMARY", "RECOVERY"), "Unknown storage lane")
    if storage_lane == "RECOVERY":
        _require(
            4 <= starting_step <= 12
            and tx.identity["parent"] == "b0"
            and tx.identity["arm"] == "U11"
            and recovery_binding_ref is not None,
            "Only registered independent b0/U11 step4 recovery is allowed",
        )
    journal = journal_factory(
        tx,
        runtime,
        experiment_root=experiment_root,
        storage_lane=storage_lane,
        restored=restored is not None,
        resumed_after_data=starting_step > (4 if storage_lane == "RECOVERY" else 0),
        recovery_binding_ref=recovery_binding_ref,
    )
    commits, calls = [], 0
    with journal:
        for step in range(starting_step + 1, 13):
            with access_factory():
                ready = rollouts.collect_step(
                    tx, runtime, experiment_root=experiment_root, registration_ref=registration_ref
                )
                _require(ready["step"] == step, "Rollout must consume the next committed step")
                if failpoint:
                    failpoint("after_ready", step)
                position = _audit_position(runtime)
                indices = tx.identity["diagnostic_row_indices"] if step in (1, 4, 8, 12) else ()
                result = training.update(
                    runtime,
                    optimizer,
                    scheduler,
                    ready["records"],
                    [8] * 16,
                    reference,
                    ready["credits"],
                    diagnostic_row_indices=indices,
                )
                if failpoint:
                    failpoint("after_update", step)
                evidence_ref = _freeze_preupdate(tx, runtime, position, result, step)
                marker = tx.commit(
                    runtime,
                    optimizer,
                    scheduler,
                    step=step,
                    sampler_state=ready["sampler_state"],
                    update=result,
                    references={**ready["references"], "evidence": evidence_ref},
                    failpoint=(lambda name, current_step=step: failpoint(name, current_step))
                    if failpoint
                    else None,
                )
                journal.record_commit(marker, ready)
                commits.append(marker["commit_sha256"])
                calls += ready["new_model_calls"]
        execution_ref = journal.finish()
    return dict(
        schema=SCHEMA,
        status="COMPLETED_NOT_GATE_SIGNED" if scope == "SERVER_ACTUAL" else "SYNTHETIC_ONLY",
        execution_origin=scope,
        run_id=tx.identity["run_id"],
        storage_lane=storage_lane,
        starting_step=starting_step,
        committed_step=12,
        logical_updates=12,
        segment_updates=len(commits),
        segment_new_model_calls=calls,
        execution=execution_ref,
        final_commit=_reference(experiment_root, tx.checkpoints / "step-012/COMMIT.json"),
        final_policy_sha256=tx.published()[-1]["policy_sha256"],
        recovery_binding=recovery_binding_ref,
        gate_status="NOT_SIGNED",
        training_authorized=False,
        **({"phase": tx.identity["phase"]} if tx.identity["phase"] == "PILOT_V3" else {}),
    )


def _context(root, phase_identity_sha256, *, phase=PHASE):
    from .phases import verify_phase

    return verify_phase(
        root,
        pilot_contract.for_phase(phase).phase,
        deployment=Path(__file__).resolve(strict=True).parents[2],
        expected_identity_sha256=phase_identity_sha256,
    )


def _registered_path(root, phase_context, pilot_spec_sha256, parent, arm):
    checked = pilot_contract.verify_registration(
        root, phase_context=phase_context, pilot_spec_sha256=pilot_spec_sha256
    )
    matches = [p for p in checked["spec"]["paths"] if p["parent"] == parent and p["arm"] == arm]
    _require(len(matches) == 1, "Parent/arm is not one of the nine registered pilot paths")
    item = matches[0]
    base = _plain_path(checked["directory"])
    directory = _plain_path(base / item["run_directory"])
    identity = _strict_json(_read_bytes(directory / "RUN_IDENTITY.json"))
    _require(
        _reference(base, directory / "RUN_IDENTITY.json") == item["transaction_identity"],
        "Pilot run identity differs from fixed registration",
    )
    return checked, item, directory, identity


def run(
    root,
    parent,
    arm,
    task_id,
    *,
    phase_identity_sha256,
    pilot_spec_sha256,
    admission_ref,
    phase=PHASE,
    recovery_binding_ref=None,
):
    """Run exactly the remaining registered twelve-step primary or recovery path."""
    import torch

    from .access import open_training_access
    from .recovery import ExecutionJournal, verify_recovery
    from .runtime import Runtime, load

    root, task_id = _plain_path(root), _task_id(task_id)
    contract = pilot_contract.for_phase(phase)
    phase_context = _context(root, phase_identity_sha256, phase=contract.phase)
    pilot_contract.require_admission(
        root,
        phase=contract.phase,
        phase_identity_sha256=phase_identity_sha256,
        admission_ref=admission_ref,
        pilot_spec_sha256=pilot_spec_sha256,
    )
    _checked, _item, directory, identity = _registered_path(
        root, phase_context, pilot_spec_sha256, parent, arm
    )
    lane = "PRIMARY"
    if recovery_binding_ref is not None:
        _require(parent == "b0" and arm == "U11", "Only b0/U11 has a recovery lane")
        recovered = verify_recovery(
            root,
            phase_context=phase_context,
            pilot_spec_sha256=pilot_spec_sha256,
            binding_ref=recovery_binding_ref,
        )
        directory = _plain_path(recovered["directory"])
        _require(
            recovered["identity"] == identity, "Recovery logical identity differs from primary"
        )
        lane = "RECOVERY"
    with RunTransactions(directory, identity) as tx:
        attempt_directory = _plain_path(directory / "TRAINER_ATTEMPTS" / task_id)
        _require(
            not attempt_directory.exists(),
            "Previous physical attempt is immutable; use a new task ID",
        )
        attempt_directory.mkdir(parents=True)
        attempt = dict(
            schema=SCHEMA,
            status="STARTED",
            at=now(),
            task_id=task_id,
            execution_origin="SERVER_ACTUAL",
            storage_lane=lane,
            parent=parent,
            arm=arm,
            run_id=identity["run_id"],
            context=phase_context["identity"],
            phase_identity_sha256=phase_identity_sha256,
            pilot_spec_sha256=pilot_spec_sha256,
            admission_ref=copy.deepcopy(admission_ref),
            recovery_binding=copy.deepcopy(recovery_binding_ref),
            initial_published_step=tx.published()[-1]["step"] if tx.published() else None,
            **({"phase": contract.phase} if contract.phase == "PILOT_V3" else {}),
        )
        _write(attempt_directory / "STARTED.json", attempt)
        try:
            loaded, optimizer, scheduler = load(
                root,
                parent,
                task_id,
                mode="TRAINING",
                phase=contract.phase,
                phase_identity_sha256=phase_identity_sha256,
                audit_root=directory,
                run_id=identity["run_id"],
                **(
                    {"pilot_spec_sha256": pilot_spec_sha256} if contract.phase == "PILOT_V3" else {}
                ),
            )
            _require(
                type(loaded) is Runtime
                and torch.cuda.is_available()
                and torch.cuda.device_count() == 1
                and all(p.device.type == "cuda" for p in loaded.model.parameters()),
                "Production pilot requires its actually loaded single-GPU runtime",
            )
            result = _drive(
                tx,
                loaded,
                optimizer,
                scheduler,
                experiment_root=root,
                registration_ref=identity["rollout_registration"],
                access_factory=lambda: open_training_access(
                    loaded,
                    tx,
                    experiment_root=root,
                    pilot_spec_sha256=pilot_spec_sha256,
                    admission_ref=admission_ref,
                    recovery_binding_ref=recovery_binding_ref,
                ),
                journal_factory=ExecutionJournal,
                scope="SERVER_ACTUAL",
                storage_lane=lane,
                recovery_binding_ref=recovery_binding_ref,
            )
            result.update(
                context=phase_context["identity"],
                phase_identity_sha256=phase_identity_sha256,
                pilot_spec_sha256=pilot_spec_sha256,
                admission_ref=copy.deepcopy(admission_ref),
                task_id=task_id,
                inputs=[
                    _reference(root, attempt_directory / "STARTED.json"),
                    result["execution"],
                    result["final_commit"],
                ],
            )
            _write(attempt_directory / "COMPLETE.json", result)
            return result
        except BaseException as exc:
            _write(
                attempt_directory / "FAILURE.json",
                {
                    **attempt,
                    "status": "FAILED_PRESERVED",
                    "failed_at": now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "requires_fresh_process_restore": True,
                    "sampling_retry_authorized": False,
                },
            )
            raise


def run_synthetic(
    tx, runtime, optimizer, scheduler, *, experiment_root, registration_ref, failpoint=None
):
    """Explicit CPU-only transaction fixture; never a production admission path."""
    import torch

    from .recovery import ExecutionJournal

    _require(
        not hasattr(runtime, "_srf6_source_context")
        and all(p.device.type == "cpu" for p in runtime.model.parameters())
        and not torch.cuda.is_initialized(),
        "Only an unbound CPU fixture is supported",
    )
    return _drive(
        tx,
        runtime,
        optimizer,
        scheduler,
        experiment_root=_plain_path(experiment_root),
        registration_ref=registration_ref,
        access_factory=nullcontext,
        journal_factory=ExecutionJournal.synthetic,
        scope="SYNTHETIC_ONLY",
        storage_lane="PRIMARY",
        failpoint=failpoint,
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("PILOT_V1", "PILOT_V3"), default=PHASE)
    parser.add_argument("--parent", choices=("b0", "b1", "b2"), required=True)
    parser.add_argument("--arm", choices=("U10", "U11", "C_SHUF"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--phase-identity-sha256", required=True)
    parser.add_argument("--pilot-spec-sha256", required=True)
    parser.add_argument("--admission", nargs=2, metavar=("PATH", "SHA256"), required=True)
    parser.add_argument("--recovery-binding", nargs=2, metavar=("PATH", "SHA256"))
    args = parser.parse_args(argv)
    result = run(
        args.root,
        args.parent,
        args.arm,
        args.task_id,
        phase=args.phase,
        phase_identity_sha256=args.phase_identity_sha256,
        pilot_spec_sha256=args.pilot_spec_sha256,
        admission_ref=dict(zip(("path", "sha256"), args.admission, strict=True)),
        recovery_binding_ref=dict(zip(("path", "sha256"), args.recovery_binding, strict=True))
        if args.recovery_binding
        else None,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
