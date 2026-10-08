"""SER-J23 fixed evaluation matrix, public-only generation and complete release.

No scorer is imported by generation. New checkpoint hashes are bound only after
training commits its registered endpoint; historical files retain their identity.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections import Counter
from pathlib import Path

from ..exposure_substitution.evaluate import (
    _failure_payload,
    _file_hash,
    _job_lock,
    _raw_record,
    _safe_component,
)
from ..exposure_substitution.evaluate import (
    _write_once as write_once,
)
from ..verified_discovery_transfer.queue import digest, read_json
from .schema import ARMS, PARENTS, PHASE_ID

A2, B2, A3, B3 = tuple(ARMS)
LEGACY_C = "C_FORWARD_C3"
ROLES = {
    "E_CONFIRM2": "confirm_worker",
    "DEV_TRAJECTORY": "diagnostic",
    "TRAIN_FIT_FULL": "diagnostic",
    "DEV_DELTA": "diagnostic",
}
GENERATION = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 64,
    "enable_thinking": False,
    "grammar_constrained": False,
    "use_cache": False,
    "engine": "historical_uncached_prefix_recompute",
    "extra_likelihood_rescoring": False,
}
SEED_FIELDS = (
    "phase_id",
    "checkpoint_sha256",
    "logical_arm_id",
    "panel",
    "task_id",
    "draw_index",
    "salt",
)


def evaluation_job_id(job):
    block = job["block"] if job["block"] is not None else "baseline"
    cell = "." + job["cell"] if job["panel"] == "DEV_DELTA" else ""
    return (
        f"eval.{job['parent']}.{block}.{job['logical_arm_id']}"
        f".step{job['step']}.{job['panel']}{cell}"
    )


def checkpoint_key(job):
    block = "baseline" if job["block"] is None else f"block{job['block']}"
    return f"{job['source']}.{job['parent']}.{block}.{job['logical_arm_id']}.step{job['step']}"


def _full_jobs(mode):
    if mode not in {"REUSE_12", "RERUN_24"}:
        raise ValueError("Execution mode must be frozen before matrix construction")
    jobs = []

    def add(parent, block, arm, step, panel, tasks, draws, source, cell=None):
        row = {
            "phase_id": PHASE_ID,
            "parent": parent,
            "block": block,
            "logical_arm_id": arm,
            "arm": arm,
            "step": step,
            "panel": panel,
            "tasks": tasks,
            "draws": draws,
            "source": source,
            "kind": "confirm_eval_sealed" if panel == "E_CONFIRM2" else "diagnostic_eval",
        }
        if block is not None:
            row.update(block_seed=108701 + block, updates=256)
        if cell is not None:
            row["cell"] = cell
        row["job_id"] = evaluation_job_id(row)
        jobs.append(row)

    for parent in PARENTS:
        for block in range(3):
            for arm in ARMS:
                source = "LEGACY" if mode == "REUSE_12" and arm in (A2, B2) else "NEW_TRAINING"
                add(parent, block, arm, 256, "E_CONFIRM2", 992, 8, source)
                add(parent, block, arm, 256, "TRAIN_FIT_FULL", 46, 4, source)
            for arm in (A2, B2):
                for step in (64, 128, 192):
                    add(
                        parent,
                        block,
                        arm,
                        step,
                        "DEV_TRAJECTORY",
                        128,
                        8,
                        "LEGACY" if mode == "REUSE_12" else "NEW_TRAINING",
                    )
            for cell, arms, count in (("c4j4", (A2, B2), 106), ("c3j3", (A2, LEGACY_C), 109)):
                for arm in arms:
                    add(parent, block, arm, 256, "DEV_DELTA", count, 4, "LEGACY", cell)
        add(parent, None, "PARENT", 0, "E_CONFIRM2", 992, 8, "PARENT")
        for cell, count in (("c4j4", 106), ("c3j3", 109)):
            add(parent, None, "PARENT", 0, "DEV_DELTA", count, 4, "PARENT", cell)
    return jobs


def _unavailable_ids(full, unavailable):
    by_id = {evaluation_job_id(j): j for j in full}
    excluded = set()
    for row in unavailable:
        if not isinstance(row, dict) or not row.get("reason") or row.get("job_id") in excluded:
            raise ValueError(
                "Each unavailable diagnostic needs a unique job_id and preregistered reason"
            )
        job = by_id.get(row.get("job_id"))
        if (
            job is None
            or job["source"] != "LEGACY"
            or job["panel"] not in {"DEV_DELTA", "DEV_TRAJECTORY"}
        ):
            raise ValueError(
                "Only unavailable historical diagnostic checkpoints may reduce allocation"
            )
        excluded.add(row["job_id"])
    # A missing checkpoint cannot be declared absent for one cell but present for
    # another; this catches selective retention of a favorable diagnostic panel.
    missing_keys = {checkpoint_key(by_id[jid]) for jid in excluded}
    for job in full:
        if checkpoint_key(job) in missing_keys and job["panel"] not in {
            "DEV_DELTA",
            "DEV_TRAJECTORY",
        }:
            raise ValueError(
                "Missing historical checkpoint is required by the complete formal matrix"
            )
        if (
            checkpoint_key(job) in missing_keys
            and job["panel"] in {"DEV_DELTA", "DEV_TRAJECTORY"}
            and evaluation_job_id(job) not in excluded
        ):
            raise ValueError(
                "Missing historical checkpoint must omit all affected diagnostic cells"
            )
    return excluded


def build_evaluation_jobs(mode, checkpoint_bindings=None, *, unavailable=()):
    full = _full_jobs(mode)
    excluded = _unavailable_ids(full, unavailable)
    jobs = [j for j in full if evaluation_job_id(j) not in excluded]
    if checkpoint_bindings is not None:
        for job in jobs:
            if job["source"] != "NEW_TRAINING":
                key = checkpoint_key(job)
                if key not in checkpoint_bindings:
                    raise ValueError("Missing read-only checkpoint binding: " + key)
                job["checkpoint_binding"] = dict(checkpoint_bindings[key])
    return validate_evaluation_jobs(jobs, mode=mode, unavailable=unavailable)


def validate_evaluation_jobs(jobs, *, mode, unavailable=()):
    full = _full_jobs(mode)
    excluded = _unavailable_ids(full, unavailable)
    expected = {evaluation_job_id(j): j for j in full if evaluation_job_id(j) not in excluded}
    actual = {evaluation_job_id(j): j for j in jobs}
    if len(actual) != len(jobs) or set(actual) != set(expected):
        raise ValueError("Evaluation differs from the complete preregistered matrix")
    for jid, baseline in expected.items():
        if any(type(actual[jid].get(k)) is not int for k in ("tasks", "draws", "step")):
            raise ValueError("Evaluation counts and step require exact integers")
        if actual[jid]["block"] is not None and type(actual[jid]["block"]) is not int:
            raise ValueError("Training block requires an exact integer")
        if any(actual[jid].get(key) != value for key, value in baseline.items()):
            raise ValueError("Frozen evaluation identity, allocation or source changed: " + jid)
    counts = Counter()
    for j in jobs:
        counts[j["panel"]] += j["tasks"] * j["draws"]
    if counts["E_CONFIRM2"] != 206336 or counts["TRAIN_FIT_FULL"] != 4416:
        raise ValueError("Confirmation and full-fit allocations must never shrink")
    reduction = sum(j["tasks"] * j["draws"] for j in full if evaluation_job_id(j) in excluded)
    if sum(counts.values()) != 259656 - reduction:
        raise ValueError("Unused diagnostic allocation cannot transfer to another panel")
    return jobs


def sample_identity(job, task, draw, model_identity):
    if type(draw) is not int or not 0 <= draw < job["draws"]:
        raise ValueError("Draw index outside frozen allocation")
    sha = model_identity.get("checkpoint_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Requests require the actual checkpoint SHA256")
    return {
        "phase_id": PHASE_ID,
        "checkpoint_sha256": sha,
        "logical_arm_id": job["logical_arm_id"],
        "panel": job["panel"],
        "task_id": task["task_id"],
        "draw_index": draw,
        "salt": 2026100805,
        "source_experiment_id": model_identity["source_experiment_id"],
        "source_checkpoint_id": model_identity["source_checkpoint_id"],
        "checkpoint_id": model_identity["source_experiment_id"]
        + ":"
        + model_identity["source_checkpoint_id"],
        "parent": job["parent"],
        "block": job["block"],
        "step": job["step"],
        "root_id": task["root_id"],
        "job_id": evaluation_job_id(job),
    }


def sample_seed(identity):
    if identity.get("phase_id") != PHASE_ID or identity.get("salt") != 2026100805:
        raise ValueError("Sampling phase/salt differs from frozen design")
    return int(digest({key: identity[key] for key in SEED_FIELDS})[:16], 16) % (2**63)


def _directory(run, job):
    return Path(run) / "evaluations" / _safe_component(evaluation_job_id(job))


def _sample_path(directory, identity):
    return directory / "roots" / _safe_component(identity["root_id"]) / (digest(identity) + ".json")


def evaluation_tasks(run, job):
    from .schema import RoleDataset, read_bound

    tasks = RoleDataset(run, ROLES[job["panel"]]).public(job["panel"])
    if job["panel"] != "E_CONFIRM2":
        selections = read_bound(run, "manifests/panel_selection.json")[job["panel"]]
        if job["panel"] == "TRAIN_FIT_FULL":
            ids = selections["by_arm"][job["logical_arm_id"]]
        elif job["panel"] == "DEV_DELTA":
            ids = selections["by_cell"][job["cell"]]
        else:
            ids = selections["task_ids"]
        by_id = {t["task_id"]: t for t in tasks}
        if len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("Panel selection contains duplicate/unknown tasks")
        tasks = [by_id[tid] for tid in ids]
    if len(tasks) != job["tasks"] or len({t["task_id"] for t in tasks}) != job["tasks"]:
        raise ValueError("Frozen public panel coverage changed")
    return tasks


def _job_context(run, job_id, machine):
    from .schema import _frozen_header, read_bound

    run = Path(run).resolve()
    # Deliberately verify only named public/config files. verify_frozen() reads
    # every file's bytes, including truth; it must never run in a model worker.
    frozen = _frozen_header(run)
    mode = read_bound(run, "EXECUTION_MODE.json")["mode"]
    unavailable = read_bound(run, "manifests/diagnostic_availability.json").get("unavailable", [])
    jobs = validate_evaluation_jobs(
        read_bound(run, "manifests/evaluation_jobs.jsonl"), mode=mode, unavailable=unavailable
    )
    found = [j for j in jobs if evaluation_job_id(j) == job_id]
    if len(found) != 1:
        raise ValueError("Only frozen evaluation jobs may generate")
    bound_machine = read_bound(run, "machine.json")
    if isinstance(machine, (str, Path)):
        machine = read_json(machine)
    if machine is not None and machine != bound_machine:
        raise ValueError("Evaluation machine changed after freeze")
    return run, found[0], bound_machine, frozen


def resolve_checkpoint_binding(run, job):
    """Late-bind new states through their immutable result; retain old identities."""
    from ..modeling_v3.io import canonical_hash
    from .runtime import read_student_checkpoint
    from .schema import _frozen_header

    if job["source"] == "NEW_TRAINING":
        from .training import student_directory

        directory = student_directory(run, job["parent"], job["block"], job["logical_arm_id"])
        result = read_json(directory / "SFT_RESULT.json")
        identity = result.get("identity", {})
        expected = {
            "experiment_id": PHASE_ID,
            "parent": job["parent"],
            "block": job["block"],
            "arm": job["logical_arm_id"],
            "logical_arm_id": job["logical_arm_id"],
            "technical_only": False,
            "frozen_plan_hash": canonical_hash(_frozen_header(run)),
            "seed": 108701 + job["block"],
            "steps": 256,
        }
        if (
            result.get("status") != "SFT_256_COMPLETE"
            or result.get("optimizer_updates") != 256
            or result.get("execution_kind") != "REAL_CUDA_SFT"
            or any(identity.get(k) != v for k, v in expected.items())
        ):
            raise ValueError("New checkpoint lacks complete registered training identity")
        receipts = [r for r in result.get("checkpoints", []) if r["step"] == job["step"]]
        path = directory / f"step{job['step']:03d}.pt"
        if len(receipts) != 1 or Path(receipts[0]["path"]).resolve() != path.resolve():
            raise ValueError("Missing immutable new checkpoint receipt")
        binding = {
            "path": str(path),
            "checkpoint_sha256": receipts[0]["sha256"],
            "source_experiment_id": PHASE_ID,
            "source_checkpoint_id": (
                f"{job['parent']}.block{job['block']}.{job['logical_arm_id']}.step{job['step']}"
            ),
            "expected_identity": identity,
        }
    else:
        binding = dict(job.get("checkpoint_binding", {}))
    required = {"path", "checkpoint_sha256", "source_experiment_id", "source_checkpoint_id"}
    if not required <= binding.keys() or any(not binding[k] for k in required):
        raise ValueError("Missing read-only model source binding")
    if _file_hash(binding["path"]) != binding["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from registered source")
    if job["source"] != "PARENT":
        state = read_student_checkpoint(binding["path"])
        expected_identity = binding.get("expected_identity")
        if (
            not isinstance(expected_identity, dict)
            or state["identity"] != expected_identity
            or state["identity"].get("experiment_id") != binding["source_experiment_id"]
            or state["step"] != job["step"]
            or state["identity"].get("technical_only") is not False
        ):
            raise ValueError("Checkpoint source identity must remain unchanged")
        if (
            state["identity"].get("parent") != job["parent"]
            or state["identity"].get("block") != job["block"]
        ):
            raise ValueError("Checkpoint parent/block mismatch")
        source_arm = (
            (
                ARMS[job["logical_arm_id"]]["legacy_arm"]
                if job["logical_arm_id"] in ARMS
                else LEGACY_C
            )
            if job["source"] == "LEGACY"
            else job["logical_arm_id"]
        )
        if state["identity"].get("arm") != source_arm:
            raise ValueError("Read-only alias names a different source arm")
    return binding


@contextlib.contextmanager
def _backend_for_job(run, job, machine, allow_gpu):
    from .runtime import load_parent_backend, load_student_into_backend

    binding = resolve_checkpoint_binding(run, job)
    backend = load_parent_backend(run, job["parent"], machine=machine, allow_gpu=allow_gpu)
    if job["source"] != "PARENT":
        load_student_into_backend(
            backend, binding["path"], expected_identity=binding["expected_identity"]
        )
        if backend.receipt.get("student_checkpoint_sha256") != binding["checkpoint_sha256"]:
            raise ValueError("Loaded student differs from read-only binding")
    elif backend.receipt.get("checkpoint", {}).get("sha256") != binding["checkpoint_sha256"]:
        raise ValueError("Loaded parent differs from frozen source")
    backend.receipt = {
        **backend.receipt,
        **{
            k: binding[k]
            for k in ("checkpoint_sha256", "source_experiment_id", "source_checkpoint_id")
        },
    }
    try:
        yield backend
    finally:
        del backend


def _model_identity(job, backend, execution_kind):
    receipt = backend.receipt
    keys = (
        "checkpoint_sha256",
        "source_experiment_id",
        "source_checkpoint_id",
        "inference_fingerprint",
    )
    if any(not receipt.get(k) for k in keys):
        raise ValueError("Backend must expose actual source and inference identity")
    result = {k: receipt[k] for k in keys}
    result.update(execution_kind=execution_kind, logical_arm_id=job["logical_arm_id"])
    if execution_kind == "REAL_FROZEN_GPU" and not receipt.get(
        "checkpoint_full_cpu_state_verified"
    ):
        raise ValueError("Full checkpoint state verification is required")
    return result


def _manifest(job, tasks, frozen, machine):
    from .schema import public_prompt

    return {
        "schema": "ser-j23-evaluation-manifest-v1",
        "job": job,
        "plan_hash": frozen["plan_hash"],
        "machine_hash": digest(machine),
        "job_id": evaluation_job_id(job),
        "panel": job["panel"],
        "sealed": job["panel"] == "E_CONFIRM2",
        "expected_samples": job["tasks"] * job["draws"],
        "public_tasks_hash": digest(tasks),
        "task_prompt_hashes": {t["task_id"]: digest(public_prompt(t)) for t in tasks},
        "generation": GENERATION,
        "protocol": "O0",
    }


def _accepted_record(path, identity, manifest_hash, prompt_hash, model):
    envelope = read_json(path)
    row = envelope["record"]
    if (
        envelope.get("record_sha256") != digest(row)
        or envelope.get("manifest_hash") != manifest_hash
        or any(row.get(k) != v for k, v in identity.items())
        or row.get("sample_seed") != sample_seed(identity)
        or row.get("request_id") != digest(identity)
        or row.get("sample_id") != digest(identity)
        or row.get("public_prompt_hash") != prompt_hash
        or row.get("execution_kind") != model["execution_kind"]
        or row.get("inference_fingerprint") != model["inference_fingerprint"]
    ):
        raise ValueError("Accepted raw sample identity/hash mismatch: " + str(path))
    _raw_record(row)
    return row


def validate_completed_evaluation(run, job, machine=None, *, require_complete=True):
    from .schema import public_prompt

    run, bound, machine, frozen = _job_context(run, evaluation_job_id(job), machine)
    if bound != job:
        raise ValueError("Evaluation job differs from frozen matrix")
    tasks = evaluation_tasks(run, job)
    directory = _directory(run, job)
    manifest = _manifest(job, tasks, frozen, machine)
    if read_json(directory / "manifest.json") != manifest:
        raise ValueError("Evaluation manifest changed")
    model_path = directory / "MODEL_IDENTITY.json"
    model = read_json(model_path) if model_path.exists() else None
    if model is None:
        if require_complete or list((directory / "roots").glob("*/*.json")):
            raise ValueError("Accepted evaluation has no model identity")
        return [], None
    if model["execution_kind"] == "REAL_FROZEN_GPU":
        binding = resolve_checkpoint_binding(run, job)
        if any(
            model[k] != binding[k]
            for k in ("checkpoint_sha256", "source_experiment_id", "source_checkpoint_id")
        ):
            raise ValueError("Source model changed after generation")
    rows, expected_paths = [], set()
    for task in tasks:
        for draw in range(job["draws"]):
            identity = sample_identity(job, task, draw, model)
            path = _sample_path(directory, identity)
            expected_paths.add(path)
            if not path.exists():
                if require_complete:
                    raise ValueError("Missing registered sample: " + str(path))
                continue
            rows.append(
                _accepted_record(
                    path, identity, digest(manifest), digest(public_prompt(task)), model
                )
            )
    if set((directory / "roots").glob("*/*.json")) - expected_paths:
        raise ValueError("Unregistered accepted samples present")
    receipt = {
        "schema": "ser-j23-evaluation-completion-v1",
        "status": "COMPLETE",
        "job_id": evaluation_job_id(job),
        "manifest_hash": digest(manifest),
        "model_identity_hash": digest(model),
        "samples": len(rows),
        "records_hash": digest(rows),
        "sealed": manifest["sealed"],
        "generation_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
    }
    if require_complete and read_json(directory / "completion.json") != receipt:
        raise ValueError("Evaluation completion hash/coverage mismatch")
    return rows, receipt


def _check_unresolved_attempts(directory, accepted):
    for path in (directory / "attempts").glob("*/*/STARTED.json"):
        started = read_json(path)
        if (
            started["request_id"] not in accepted
            and not (path.parent / "FAILED.json").exists()
            and not (path.parent / "RECOVERY.json").exists()
        ):
            raise PermissionError(
                "UNKNOWN generation attempt requires explicit stopped-process recovery"
            )


def acknowledge_unknown_attempt(attempt_directory, *, reason, old_process_stopped, scheduler_state):
    """Operator evidence only; this never mutates a sample or changes its seed."""
    directory = Path(attempt_directory)
    if (
        not reason
        or old_process_stopped is not True
        or scheduler_state
        not in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "LOCAL_PROCESS_EXITED",
        }
    ):
        raise PermissionError("UNKNOWN recovery requires scheduler/process termination evidence")
    started = read_json(directory / "STARTED.json")
    write_once(
        directory / "RECOVERY.json",
        {
            "status": "UNKNOWN_ATTEMPT_RETAINED",
            "reason": reason,
            "old_process_stopped": True,
            "scheduler_state": scheduler_state,
            "started_sha256": digest(started),
        },
    )


def evaluate_job(run, job_id, machine=None, allow_gpu=False, *, backend_factory=None):
    from .schema import public_prompt

    run, job, machine, frozen = _job_context(run, job_id, machine)
    if (run / "RELEASE_RECEIPT.json").exists():
        raise PermissionError("Released matrix cannot generate/retry")
    if backend_factory is None and allow_gpu is not True:
        raise PermissionError("Real evaluation requires --allow-gpu")
    if backend_factory is None:
        from .training import require_bridge

        # The panel loader verifies its public bytes separately. Bridge control
        # must not open training targets merely because this is a diagnostic.
        require_bridge(run, role="control")
    tasks = evaluation_tasks(run, job)
    manifest = _manifest(job, tasks, frozen, machine)
    directory = _directory(run, job)
    with _job_lock(directory):
        write_once(directory / "manifest.json", manifest)
        if (directory / "completion.json").exists():
            return validate_completed_evaluation(run, job, machine)[1]
        prior, _ = validate_completed_evaluation(run, job, machine, require_complete=False)
        accepted = {r["request_id"] for r in prior}
        _check_unresolved_attempts(directory, accepted)
        if len(accepted) < manifest["expected_samples"]:
            with (backend_factory or _backend_for_job)(run, job, machine, allow_gpu) as backend:
                execution = "CPU_TEST_FIXTURE" if backend_factory else "REAL_FROZEN_GPU"
                model = _model_identity(job, backend, execution)
                write_once(directory / "MODEL_IDENTITY.json", model)
                invocation = directory / "attempts" / uuid.uuid4().hex
                write_once(
                    invocation / "BACKEND.json",
                    {"execution_kind": execution, "backend": backend.receipt},
                )
                for task in tasks:
                    prompt = public_prompt(task)
                    if set(prompt) != {"system", "user"}:
                        raise ValueError("Only public system/user may reach generation")
                    for draw in range(job["draws"]):
                        identity = sample_identity(job, task, draw, model)
                        request_id = digest(identity)
                        if request_id in accepted:
                            continue
                        seed = sample_seed(identity)
                        attempt = invocation / request_id
                        attempt_identity = {
                            **identity,
                            "request_id": request_id,
                            "sample_seed": seed,
                            "started_at": time.time(),
                        }
                        write_once(attempt / "STARTED.json", attempt_identity)
                        raw = None
                        try:
                            raw = backend.generate_public(prompt, seed, 64)
                            payload = _raw_record(raw)
                            if (
                                payload.get("inference_fingerprint")
                                != model["inference_fingerprint"]
                            ):
                                raise ValueError("Generated model fingerprint changed")
                            if "seed" in raw and raw["seed"] != seed:
                                raise ValueError("Backend used a different seed")
                            row = {
                                **identity,
                                **payload,
                                "request_id": request_id,
                                "sample_id": request_id,
                                "sample_seed": seed,
                                "protocol_id": "O0",
                                "public_prompt_version": task["template_version"],
                                "public_prompt_hash": digest(prompt),
                                "execution_kind": execution,
                            }
                            write_once(
                                _sample_path(directory, identity),
                                {
                                    "manifest_hash": digest(manifest),
                                    "record": row,
                                    "record_sha256": digest(row),
                                },
                            )
                        except Exception as exc:
                            failed_raw = (
                                raw if raw is not None else getattr(exc, "raw_result", None)
                            )
                            write_once(
                                attempt / "FAILED.json",
                                {
                                    **attempt_identity,
                                    "status": "FAILED_ATTEMPT",
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                    "raw_observation": _failure_payload(failed_raw),
                                    "finished_at": time.time(),
                                },
                            )
                            raise
                        write_once(
                            attempt / "ACCEPTED.json",
                            {
                                **attempt_identity,
                                "status": "ACCEPTED",
                                "record_sha256": digest(row),
                                "finished_at": time.time(),
                            },
                        )
                        accepted.add(request_id)
        rows, receipt = validate_completed_evaluation(run, job, machine, require_complete=False)
        if len(rows) != manifest["expected_samples"]:
            raise ValueError("Fixed evaluation coverage incomplete")
        write_once(directory / "completion.json", receipt)
        return receipt


def score_rows(run, job, rows):
    from .schema import RoleDataset
    from .semantics import score_output

    tasks = {t["task_id"]: t for t in evaluation_tasks(run, job)}
    audits = {a["task_id"]: a for a in RoleDataset(run, "analysis").audit(job["panel"])}
    return [
        {
            **row,
            **score_output(
                tasks[row["task_id"]], audits[row["task_id"]], row["raw_text"], row["stop_reason"]
            ),
        }
        for row in rows
    ]


def analyze_diagnostic_job(run, job_id, machine=None):
    run, job, machine, _ = _job_context(run, job_id, machine)
    if job["panel"] == "E_CONFIRM2":
        raise PermissionError("E_CONFIRM2 remains sealed until complete release")
    rows, _ = validate_completed_evaluation(run, job, machine)
    path = _directory(run, job) / "DIAGNOSTIC_SCORED.json"
    if path.exists():
        return read_json(path)
    result = {
        "job_id": job_id,
        "status": "DIAGNOSTIC_ONLY_NO_SELECTION",
        "records": score_rows(run, job, rows),
        "raw_records_hash": digest(rows),
    }
    write_once(path, result)
    return result


def generation_accounting(run):
    run = Path(run)
    starts = list((run / "evaluations").glob("*/attempts/*/*/STARTED.json"))
    failures = [p for p in starts if (p.parent / "FAILED.json").exists()]
    unknown = [
        p
        for p in starts
        if not (p.parent / "FAILED.json").exists() and not (p.parent / "ACCEPTED.json").exists()
    ]
    return {
        "physical_generation_attempts_started": len(starts),
        "failed_generation_attempts": len(failures),
        "generation_attempts_with_unknown_completion": len(unknown),
        "allocated_gpu_hours": None,
        "allocated_gpu_hours_status": "SCHEDULER_ACCOUNTING_NOT_IMPORTED",
    }


def validate_training_endpoint(run, job, frozen=None):
    """Require every retained new checkpoint and exact committed exposure dose."""
    from .runtime import read_student_checkpoint
    from .training import student_directory

    directory = student_directory(run, job["parent"], job["block"], job["logical_arm_id"])
    result = read_json(directory / "SFT_RESULT.json")
    if (
        result.get("role_exposures") != {"common": 2816, "donor": 256, "replay": 1024}
        or sum(result.get("exposures", {}).values()) != 4096
        or sorted(r["step"] for r in result.get("checkpoints", [])) != [0, 64, 128, 192, 256]
    ):
        raise ValueError("Training endpoint lacks five checkpoints and fixed exposure dose")
    for step in (0, 64, 128, 192, 256):
        binding = resolve_checkpoint_binding(run, {**job, "source": "NEW_TRAINING", "step": step})
        state = read_student_checkpoint(binding["path"])
        if (
            state["scheduler"]["completed_updates"] != step
            or state["role_exposures"] != {"common": 11 * step, "donor": step, "replay": 4 * step}
            or sum(state["exposures"].values()) != 16 * step
        ):
            raise ValueError("Retained checkpoint cursor or exposure mismatch")
    return result


def integrity(run):
    """Read all evidence without scoring and refuse any incomplete release matrix."""
    from .queue import registered_queue, validate_training_jobs
    from .schema import read_bound, verify_frozen

    run = Path(run).resolve()
    frozen = verify_frozen(run, role="auditor")
    mode = read_bound(run, "EXECUTION_MODE.json")["mode"]
    unavailable = read_bound(run, "manifests/diagnostic_availability.json").get("unavailable", [])
    jobs = validate_evaluation_jobs(
        read_bound(run, "manifests/evaluation_jobs.jsonl"), mode=mode, unavailable=unavailable
    )
    queue = registered_queue(run)
    try:
        rows = queue.rows()
        if not rows or any(row["status"] != "COMPLETE" for row in rows):
            raise PermissionError(
                "All registered training and evaluation jobs must COMPLETE before release"
            )
        training = [r["payload"] for r in rows if r["payload"]["kind"] == "train"]
        validate_training_jobs(training, mode)
        if {r["id"] for r in rows if r["payload"]["kind"] == "eval"} != {
            evaluation_job_id(j) for j in jobs
        }:
            raise ValueError("Registered evaluation matrix mismatch")
        for job in training:
            validate_training_endpoint(run, job, frozen)
        completed, all_raw = [], {}
        for job in jobs:
            raw, receipt = validate_completed_evaluation(run, job)
            if any(r["execution_kind"] != "REAL_FROZEN_GPU" for r in raw):
                raise ValueError("CPU fixture answers cannot enter formal release")
            all_raw[evaluation_job_id(job)] = raw
            completed.append(receipt)
        if (
            sum(len(all_raw[evaluation_job_id(j)]) for j in jobs if j["panel"] == "E_CONFIRM2")
            != 206336
        ):
            raise ValueError("Confirmation requests must form the entire 206336-sample matrix")
        receipt = {
            "schema": "ser-j23-integrity-v1",
            "status": "PASS",
            "plan_hash": frozen["plan_hash"],
            "all_registered_models_terminal": True,
            "all_confirmation_requests_complete": True,
            "all_registered_jobs_complete": True,
            "completed_evaluation_receipts": completed,
            "registered_training_jobs": len(training),
            "registered_evaluation_jobs": len(jobs),
            "unavailable_preregistered_diagnostics": unavailable,
            "generation_accounting": generation_accounting(run),
        }
        return receipt, all_raw, jobs
    finally:
        queue.db.close()


def release_and_analyze(run):
    from .queue import registered_queue
    from .reports import write_reports

    receipt, all_raw, jobs = integrity(run)
    queue = registered_queue(run)
    try:
        release = {
            **receipt,
            "schema": "ser-j23-release-v1",
            "status": "RELEASED",
            "matrix_digest": queue.seal_release(),
        }
        write_once(Path(run) / "RELEASE_RECEIPT.json", release)
    finally:
        queue.db.close()
    scored = [row for job in jobs for row in score_rows(run, job, all_raw[evaluation_job_id(job)])]
    return write_reports(Path(run) / "reports", scored, release_receipt=release)
