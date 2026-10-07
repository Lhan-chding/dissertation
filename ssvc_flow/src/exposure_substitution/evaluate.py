"""Frozen SER-J2 generation, durable sample resume, and final release gate.

Generation receives only the explicit SER public renderer's system/user strings.
E_CONFIRM raw answers are stored sealed and are never scored by evaluate_job.
Failed attempts are retained, and a resumed draw always uses its original seed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from ..verified_discovery_transfer.queue import digest, read_json

ARMS = ("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3")
PARENTS = ("S96", "REP96")
ROLES = {"E_DIAG": "diagnostic", "E_CONFIRM": "confirm_worker", "TRAIN_FIT": "trainer"}
EXPERIMENT = "SER_J2_20261007"


def evaluation_job_id(job):
    block = job["block"] if job["block"] is not None else "baseline"
    return f"eval.{job['parent']}.{block}.{job['arm']}.step{job['step']}.{job['panel']}"


def validate_evaluation_jobs(jobs):
    """Reject silent matrix growth, duplicate parent baselines, or missing draws."""
    expected = []
    for parent in PARENTS:
        for block in range(3):
            for arm in ARMS:
                for step, panel, tasks, draws, kind in (
                    (128, "E_DIAG", 48, 8, "diagnostic_eval"),
                    (128, "TRAIN_FIT", 16, 4, "fit_sentinel"),
                    (256, "E_DIAG", 48, 8, "diagnostic_eval"),
                    (256, "TRAIN_FIT", 16, 4, "fit_sentinel"),
                    (256, "E_CONFIRM", 800, 8, "confirm_eval_sealed"),
                ):
                    expected.append((parent, block, arm, step, panel, tasks, draws, kind))
        for panel, count in (("E_DIAG", 48), ("E_CONFIRM", 800)):
            expected.append((parent, None, "PARENT", 0, panel, count, 8, "frozen_parent_eval"))
    expected.append(("BASE_NO_ADAPTER", None, "BASE", 0, "E_DIAG", 48, 8, "frozen_base_diagnostic"))
    keys = ("parent", "block", "arm", "step", "panel", "tasks", "draws", "kind")
    actual = [tuple(row[k] for k in keys) for row in jobs]
    if len(jobs) != 95 or Counter(actual) != Counter(expected):
        raise ValueError("Evaluation must remain the exact registered 95-job matrix")
    if any(
        row.get("block_seed") != 108701 + row["block"] or row.get("updates") != 256
        for row in jobs
        if row["block"] is not None
    ):
        raise ValueError("Student evaluation schedule identity changed")
    if len({evaluation_job_id(row) for row in jobs}) != 95:
        raise ValueError("Duplicate evaluation job identity")
    if sum(row["tasks"] * row["draws"] for row in jobs) != 145280:
        raise ValueError("Formal generation budget changed")
    return jobs


def checkpoint_id(job):
    if job["arm"] in ("PARENT", "BASE"):
        return job["parent"]
    return f"{job['parent']}.block{job['block']}.{job['arm']}.step{job['step']}"


def sample_identity(job, task, draw):
    if type(draw) is not int or not 0 <= draw < job["draws"]:
        raise ValueError("Draw index is outside its frozen allocation")
    return {
        "experiment_id": EXPERIMENT,
        "checkpoint_id": checkpoint_id(job),
        "parent": job["parent"],
        "block": job["block"],
        "arm": job["arm"],
        "step": job["step"],
        "panel": job["panel"],
        "task_id": task["task_id"],
        "root_id": task["root_id"],
        "draw_index": draw,
        "role": ROLES[job["panel"]],
    }


def sample_seed(identity):
    # Include every registered identity component. Different arms have independent
    # streams; pairing is at the task/root level, never shared sampled trajectories.
    return int(digest(identity)[:16], 16) % (2**63)


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_once(path, value):
    """Durable atomic publish without replacing an already accepted answer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != value:
                raise ValueError("Immutable evaluation artifact changed: " + str(path)) from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _job_lock(directory):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An evaluation worker already owns this fixed job") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _safe_component(value):
    if (
        not isinstance(value, str)
        or not value
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in value
        )
    ):
        raise ValueError("Unsafe artifact identity")
    if value in (".", ".."):
        raise ValueError("Unsafe artifact identity")
    return value


def _evaluation_directory(run, job):
    return Path(run) / "evaluations" / _safe_component(evaluation_job_id(job))


def evaluation_tasks(run, job):
    """Return public records only, including arm-matched training sentinels."""
    from .schema import RoleDataset, read_bound
    from .semantics import public_center

    data = RoleDataset(run, ROLES[job["panel"]])
    if job["panel"] != "TRAIN_FIT":
        tasks = data.public(job["panel"])
    else:
        selected = read_bound(run, "manifests/fit_sentinels.json")
        tasks = []
        for label, split, count in (("common", "COMMON_TRAIN", 8), ("replay", "REPLAY", 4)):
            ids = selected[label]
            public = {task["task_id"]: task for task in data.public(split)}
            if len(ids) != count or len(set(ids)) != count or not set(ids) <= public.keys():
                raise ValueError("Training sentinel identity/count mismatch")
            tasks.extend(public[tid] for tid in ids)
        roots = selected["donor_roots"]
        if len(roots) != 4 or len(set(roots)) != 4:
            raise ValueError("Training donor sentinel roots must be four frozen roots")
        hub = dict(zip(ARMS, (0, 3, 2), strict=True))[job["arm"]]
        donors = [
            task
            for task in data.public("DONOR_TRAIN")
            if task["root_id"] in roots and public_center(task) == hub
        ]
        if len(donors) != 4 or {t["root_id"] for t in donors} != set(roots):
            raise ValueError("Matched donor training sentinels are incomplete")
        tasks.extend(sorted(donors, key=lambda task: roots.index(task["root_id"])))
    if len(tasks) != job["tasks"] or len({t["task_id"] for t in tasks}) != job["tasks"]:
        raise ValueError("Frozen evaluation panel task coverage changed")
    return tasks


def _job_context(run, job_id, machine):
    from .schema import read_bound, verify_frozen

    run = Path(run).resolve()
    frozen = verify_frozen(run)
    jobs = validate_evaluation_jobs(read_bound(run, "manifests/evaluation_jobs.jsonl"))
    found = [job for job in jobs if evaluation_job_id(job) == job_id]
    if len(found) != 1:
        raise ValueError("Unknown evaluation job; only frozen jobs may generate")
    bound_machine = read_bound(run, "machine.json")
    if machine is None:
        machine = bound_machine
    elif isinstance(machine, (str, Path)):
        machine = read_json(machine)
    if not isinstance(machine, dict):
        raise ValueError("Machine configuration must be an object or path")
    if machine != bound_machine:
        raise ValueError("Evaluation machine differs from frozen machine.json")
    return run, found[0], machine, frozen


@contextlib.contextmanager
def _backend_for_job(run, job, machine, allow_gpu):
    from .runtime import load_parent_backend, load_student_into_backend

    backend = load_parent_backend(
        run, "S96" if job["arm"] == "BASE" else job["parent"], machine=machine, allow_gpu=allow_gpu
    )
    if job["arm"] in ARMS:
        from ..modeling_v3.io import canonical_hash
        from .schema import verify_frozen
        from .training import student_directory

        path = student_directory(run, job["parent"], job["block"], job["arm"])
        load_student_into_backend(backend, path / f"step{job['step']:03d}.pt")
        identity = backend.receipt["student_identity"]
        expected = {
            "experiment_id": EXPERIMENT,
            "parent": job["parent"],
            "block": job["block"],
            "arm": job["arm"],
            "technical_only": False,
            "frozen_plan_hash": canonical_hash(verify_frozen(run)),
            "seed": job["block_seed"],
            "steps": 256,
        }
        if (
            any(identity.get(key) != value for key, value in expected.items())
            or backend.receipt["student_step"] != job["step"]
        ):
            raise ValueError("Loaded student is not the registered formal checkpoint")
    if job["arm"] != "BASE":
        yield backend
        return
    # PEFT's disable_adapter context is explicit and scoped to this newly loaded
    # diagnostic backend. It cannot write either parent or student checkpoint.
    from ..protocol_state_probes.inference import _model_guard

    model = backend.adapter.model
    if not callable(getattr(model, "disable_adapter", None)):
        raise ValueError("Base diagnostic requires an explicit adapter-disable capability")
    guard_before = _model_guard(model)
    with model.disable_adapter():
        adapters = [module for module in model.modules() if hasattr(module, "lora_A")]
        if not adapters or any(
            not getattr(module, "disable_adapters", False) for module in adapters
        ):
            raise ValueError("Base diagnostic adapter-disable state could not be verified")
        backend._guard_baseline = _model_guard(model)
        backend.receipt = {
            **backend.receipt,
            "evaluation_model_role": "BASE_NO_ADAPTER",
            "all_lora_adapters_disabled": True,
            "inference_fingerprint": digest(
                {
                    "parent_load_fingerprint": backend.receipt["inference_fingerprint"],
                    "all_lora_adapters_disabled": True,
                }
            ),
        }
        try:
            yield backend
        finally:
            if _model_guard(model) != backend._guard_baseline:
                raise RuntimeError("Base diagnostic mutated model tensors")
    if _model_guard(model) != guard_before:
        raise RuntimeError("Base diagnostic failed to restore adapter/model state")
    backend._guard_baseline = guard_before


def _raw_record(raw):
    """Validate transport, never reject a model's wrong or unparseable answer."""
    required = ("raw_text", "token_ids", "stop_reason", "generated_length", "elapsed_seconds")
    if not isinstance(raw, dict) or any(key not in raw for key in required):
        raise ValueError("Frozen backend omitted required raw observation fields")
    tokens = raw["token_ids"]
    if (
        not isinstance(raw["raw_text"], str)
        or not isinstance(tokens, list)
        or any(type(token) is not int or token < 0 for token in tokens)
        or type(raw["generated_length"]) is not int
        or raw["generated_length"] != len(tokens)
        or not 0 <= len(tokens) <= 64
        or not isinstance(raw["stop_reason"], str)
        or not isinstance(raw["elapsed_seconds"], (int, float))
        or not math.isfinite(raw["elapsed_seconds"])
        or raw["elapsed_seconds"] < 0
        or raw.get("extra_rescoring", False)
    ):
        raise ValueError("Invalid raw observation transport or forbidden extra rescoring")
    optional = (
        "chosen_token_logprobs",
        "input_audit",
        "inference_fingerprint",
        "eos_token_ids",
        "pad_token_id",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "truncated",
        "terminated_by_eos",
    )
    output = {key: raw[key] for key in required + optional if key in raw}
    if "chosen_token_logprobs" in output and output["chosen_token_logprobs"] is not None:
        values = output["chosen_token_logprobs"]
        if len(values) != len(tokens) or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
        ):
            raise ValueError("Backend token log probabilities are malformed")
    output["extra_rescoring"] = False
    return output


def _failure_payload(value):
    """Preserve malformed transports even when strict JSON cannot encode them."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"non_json_float": repr(value)}
    if isinstance(value, (list, tuple)):
        return [_failure_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _failure_payload(item) for key, item in value.items()}
    return {"non_json_type": type(value).__name__, "representation": repr(value)}


def _sample_path(directory, identity):
    root = _safe_component(identity["root_id"])
    return directory / "roots" / root / (digest(identity) + ".json")


def _model_identity(job, backend, execution_kind):
    receipt = backend.receipt
    result = {
        "checkpoint_id": checkpoint_id(job),
        "execution_kind": execution_kind,
        "inference_fingerprint": receipt["inference_fingerprint"],
        "parent_checkpoint": receipt.get("checkpoint"),
        "student_checkpoint_sha256": receipt.get("student_checkpoint_sha256"),
        "student_identity": receipt.get("student_identity"),
        "all_lora_adapters_disabled": receipt.get("all_lora_adapters_disabled", False),
    }
    if execution_kind == "REAL_FROZEN_GPU":
        if not receipt.get("checkpoint_full_cpu_state_verified"):
            raise ValueError("Frozen parent tensor identity was not verified")
        if job["arm"] == "BASE" and not result["all_lora_adapters_disabled"]:
            raise ValueError("Unverified base model adapter-disable state")
        if job["arm"] in ARMS and not result["student_checkpoint_sha256"]:
            raise ValueError("Student checkpoint hash must be recorded")
    return result


def _accepted_record(path, identity, manifest_hash, prompt_hash):
    envelope = read_json(path)
    row = envelope["record"]
    if (
        envelope.get("record_sha256") != digest(row)
        or envelope.get("manifest_hash") != manifest_hash
        or any(row.get(key) != value for key, value in identity.items())
        or row.get("sample_seed") != sample_seed(identity)
        or row.get("request_id") != digest(identity)
        or row.get("public_prompt_hash") != prompt_hash
    ):
        raise ValueError("Accepted raw sample identity/hash mismatch: " + str(path))
    _raw_record(row)
    return row


def _manifest(job, tasks, frozen, machine):
    from .schema import public_prompt

    return {
        "schema": "ser-j2-evaluation-manifest-v1",
        "job": job,
        "plan_hash": frozen["plan_hash"],
        "machine_hash": digest(machine),
        "job_id": evaluation_job_id(job),
        "checkpoint_id": checkpoint_id(job),
        "panel": job["panel"],
        "sealed": job["panel"] == "E_CONFIRM",
        "expected_samples": job["tasks"] * job["draws"],
        "task_prompt_hashes": {task["task_id"]: digest(public_prompt(task)) for task in tasks},
        "public_tasks_hash": digest(tasks),
        "protocol": "O0",
        "generation": {
            "temperature": 1,
            "top_p": 1,
            "top_k": 0,
            "max_new_tokens": 64,
            "enable_thinking": False,
            "use_cache": False,
            "extra_rescoring": False,
        },
    }


def validate_completed_evaluation(run, job, machine=None, *, require_complete=True):
    """Recompute unique coverage and every accepted sample's bound hash."""
    from .schema import public_prompt

    run, bound, machine, frozen = _job_context(run, evaluation_job_id(job), machine)
    if bound != job:
        raise ValueError("Evaluation job differs from frozen manifest")
    tasks = evaluation_tasks(run, job)
    manifest = _manifest(job, tasks, frozen, machine)
    directory = _evaluation_directory(run, job)
    if read_json(directory / "manifest.json") != manifest:
        raise ValueError("Evaluation manifest changed")
    model_path = directory / "MODEL_IDENTITY.json"
    model_identity = read_json(model_path) if model_path.exists() else None
    rows, expected_paths = [], set()
    for task in tasks:
        prompt_hash = digest(public_prompt(task))
        for draw in range(job["draws"]):
            identity = sample_identity(job, task, draw)
            path = _sample_path(directory, identity)
            expected_paths.add(path)
            if not path.exists():
                if require_complete:
                    raise ValueError("Missing registered evaluation sample: " + str(path))
                continue
            row = _accepted_record(path, identity, digest(manifest), prompt_hash)
            if (
                model_identity is None
                or row.get("inference_fingerprint") != model_identity["inference_fingerprint"]
                or row.get("execution_kind") != model_identity["execution_kind"]
            ):
                raise ValueError("Raw sample belongs to a different model identity")
            rows.append(row)
    if set((directory / "roots").glob("*/*.json")) - expected_paths:
        raise ValueError("Unregistered accepted samples present")
    if model_identity is not None and model_identity["execution_kind"] == "REAL_FROZEN_GPU":
        if job["arm"] in ARMS:
            from .training import student_directory

            checkpoint = student_directory(run, job["parent"], job["block"], job["arm"])
            if (
                _file_hash(checkpoint / f"step{job['step']:03d}.pt")
                != model_identity["student_checkpoint_sha256"]
            ):
                raise ValueError("Student checkpoint changed after sampling")
        else:
            from .schema import read_bound

            parent = "S96" if job["arm"] == "BASE" else job["parent"]
            bound = read_bound(run, "parent_availability.json")["checkpoints"]
            expected = next(row["checkpoint"] for row in bound if row["checkpoint_id"] == parent)
            if model_identity["parent_checkpoint"] != expected:
                raise ValueError("Parent checkpoint differs from its frozen binding")
    receipt = {
        "schema": "ser-j2-evaluation-completion-v1",
        "status": "COMPLETE",
        "job_id": evaluation_job_id(job),
        "manifest_hash": digest(manifest),
        "model_identity_hash": digest(model_identity),
        "samples": len(rows),
        "records_hash": digest(rows),
        "sealed": manifest["sealed"],
        "generation_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
    }
    if require_complete and read_json(directory / "completion.json") != receipt:
        raise ValueError("Evaluation completion receipt hash/coverage mismatch")
    return rows, receipt


def evaluate_job(run, job_id, machine=None, allow_gpu=False, *, backend_factory=None):
    """Execute exactly one fixed panel; backend injection is CPU-test-only.

    The injected context-manager factory is intentionally absent from the CLI.
    Its artifacts are marked CPU_TEST_FIXTURE and cannot pass formal release.
    """
    from .schema import public_prompt

    run, job, machine, frozen = _job_context(run, job_id, machine)
    if (run / "FINAL_RELEASE.json").exists():
        raise PermissionError("Released confirmation matrix cannot generate or retry")
    if backend_factory is None and allow_gpu is not True:
        raise PermissionError("Real frozen evaluation requires --allow-gpu")
    if backend_factory is None:
        from .training import require_bridge

        require_bridge(run)
    tasks = evaluation_tasks(run, job)
    manifest = _manifest(job, tasks, frozen, machine)
    directory = _evaluation_directory(run, job)
    with _job_lock(directory):
        _write_once(directory / "manifest.json", manifest)
        if (directory / "completion.json").exists():
            _, receipt = validate_completed_evaluation(run, job, machine)
            return receipt
        # Verify existing samples before touching a GPU. A corrupted accepted draw
        # stops execution; it is never deleted and resampled for a nicer result.
        prior, _ = validate_completed_evaluation(run, job, machine, require_complete=False)
        accepted = {row["request_id"] for row in prior}
        if len(accepted) < manifest["expected_samples"]:
            factory = backend_factory or _backend_for_job
            with factory(run, job, machine, allow_gpu) as backend:
                provenance = {
                    "execution_kind": "CPU_TEST_FIXTURE" if backend_factory else "REAL_FROZEN_GPU",
                    "backend": backend.receipt,
                }
                model_identity = _model_identity(job, backend, provenance["execution_kind"])
                _write_once(directory / "MODEL_IDENTITY.json", model_identity)
                # Loaded backend identities must remain unchanged across resumes.
                # Timing/counter fields in backend receipts are intentionally not
                # used as weight identities; the runtime enforces tensor bindings.
                invocation = directory / "attempts" / uuid.uuid4().hex
                _write_once(invocation / "BACKEND.json", provenance)
                for task in tasks:
                    prompt = public_prompt(task)
                    if set(prompt) != {"system", "user"}:
                        raise ValueError("Only public system/user strings may reach generation")
                    for draw in range(job["draws"]):
                        identity = sample_identity(job, task, draw)
                        request_id = digest(identity)
                        if request_id in accepted:
                            continue
                        seed = sample_seed(identity)
                        attempt = invocation / request_id
                        started = time.time()
                        attempt_identity = {
                            **identity,
                            "request_id": request_id,
                            "sample_seed": seed,
                            "started_at": started,
                        }
                        _write_once(attempt / "STARTED.json", attempt_identity)
                        raw = None
                        try:
                            raw = backend.generate_public(prompt, seed, 64)
                            payload = _raw_record(raw)
                            if (
                                payload.get("inference_fingerprint")
                                != model_identity["inference_fingerprint"]
                            ):
                                raise ValueError("Generated observation model fingerprint changed")
                            if "seed" in raw and raw["seed"] != seed:
                                raise ValueError("Backend sampled a different registered seed")
                            row = {
                                **identity,
                                **payload,
                                "request_id": request_id,
                                "sample_seed": seed,
                                "protocol_id": "O0",
                                "public_prompt_version": task["template_version"],
                                "public_prompt_hash": digest(prompt),
                                "execution_kind": provenance["execution_kind"],
                            }
                            _write_once(
                                _sample_path(directory, identity),
                                {
                                    "manifest_hash": digest(manifest),
                                    "record": row,
                                    "record_sha256": digest(row),
                                },
                            )
                        except Exception as exc:
                            # ObservationFault may carry a partial action; retain it
                            # even if transport validation failed before acceptance.
                            failed_raw = (
                                raw if raw is not None else getattr(exc, "raw_result", None)
                            )
                            _write_once(
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
                        _write_once(
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
            raise ValueError("Fixed evaluation draw coverage is incomplete")
        _write_once(directory / "completion.json", receipt)
        return receipt


def _score_rows(run, job, rows):
    from .schema import RoleDataset
    from .semantics import score_output

    tasks = {task["task_id"]: task for task in evaluation_tasks(run, job)}
    data = RoleDataset(run, "analysis")
    if job["panel"] != "TRAIN_FIT":
        audits = {a["task_id"]: a for a in data.audit(job["panel"])}
    else:
        audits = {}
        for split in ("COMMON_TRAIN", "DONOR_TRAIN", "REPLAY"):
            audits.update({a["task_id"]: a for a in data.audit(split)})
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
    """Optional technical/progress readout; confirmation has no early scorer."""
    run, job, machine, _ = _job_context(run, job_id, machine)
    if job["panel"] == "E_CONFIRM":
        raise PermissionError("E_CONFIRM remains sealed until final release")
    rows, _ = validate_completed_evaluation(run, job, machine)
    scored = _score_rows(run, job, rows)
    # CPU projection timings need not be bitwise repeatable; immutable raw is the
    # scientific input, while derived diagnostic rows are written once per job.
    path = _evaluation_directory(run, job) / "DIAGNOSTIC_SCORED.json"
    if path.exists():
        return read_json(path)
    result = {
        "job_id": job_id,
        "status": "DIAGNOSTIC_ONLY_NO_SELECTION",
        "records": scored,
        "raw_records_hash": digest(rows),
    }
    _write_once(path, result)
    return result


def validate_training_endpoint(run, job, frozen):
    """Validate the actual five retained snapshots, not a queue's status label."""
    from ..modeling_v3.io import canonical_hash
    from .runtime import read_student_checkpoint
    from .training import student_directory

    directory = student_directory(run, job["parent"], job["block"], job["arm"])
    result = read_json(directory / "SFT_RESULT.json")
    expected = {
        "experiment_id": EXPERIMENT,
        "parent": job["parent"],
        "block": job["block"],
        "arm": job["arm"],
        "technical_only": False,
        "frozen_plan_hash": canonical_hash(frozen),
        "seed": 108701 + job["block"],
        "steps": 256,
    }
    if (
        result.get("status") != "SFT_256_COMPLETE"
        or result.get("optimizer_updates") != 256
        or result.get("execution_kind") != "REAL_CUDA_SFT"
        or any(result.get("identity", {}).get(key) != value for key, value in expected.items())
        or result.get("role_exposures") != {"common": 2816, "donor": 256, "replay": 1024}
        or sum(result.get("exposures", {}).values()) != 4096
    ):
        raise ValueError("Completed training lacks bound fixed endpoint evidence")
    checkpoints = result.get("checkpoints", [])
    if len(checkpoints) != 5 or sorted(row["step"] for row in checkpoints) != [
        0,
        64,
        128,
        192,
        256,
    ]:
        raise ValueError("All five retained training checkpoints are required")
    for receipt in checkpoints:
        step = receipt["step"]
        path = directory / f"step{step:03d}.pt"
        if (
            Path(receipt["path"]).resolve() != path.resolve()
            or _file_hash(path) != receipt["sha256"]
        ):
            raise ValueError("Training checkpoint path/hash differs from its receipt")
        state = read_student_checkpoint(path)
        if (
            state["identity"] != result["identity"]
            or state["step"] != step
            or state["scheduler"]["completed_updates"] != step
            or state["role_exposures"] != {"common": 11 * step, "donor": step, "replay": 4 * step}
            or sum(state["exposures"].values()) != 16 * step
        ):
            raise ValueError("Retained checkpoint identity/step/exposure mismatch")
        del state
    return result


def _execution_accounting(run, queue_rows, jobs, all_raw):
    """Separate committed exposures, repeated physical work, and unknown costs."""
    from .schema import RoleDataset, read_bound
    from .training import student_directory

    exposure_rows, committed_updates = [], 0
    physical_updates, unknown_updates, update_gpu_seconds, update_wall_seconds = 0, 0, 0.0, 0.0
    schedule_cache, metadata = {}, None
    for qrow in queue_rows:
        if qrow["payload"]["kind"] != "train":
            continue
        job = qrow["payload"]
        directory = student_directory(run, job["parent"], job["block"], job["arm"])
        for path in sorted((directory / "attempts").glob("*/update*.json")):
            log = read_json(path)
            if log.get("status") != "UPDATE_APPLIED" or not log.get("physical_cost_complete"):
                unknown_updates += 1
                continue
            physical_updates += 1
            update_gpu_seconds += log["update_gpu_seconds"] or 0.0
            update_wall_seconds += log["update_wall_seconds"]
        if qrow["status"] != "COMPLETE":
            continue
        if metadata is None:
            data = RoleDataset(run, "analysis")
            metadata = {}
            for split in ("COMMON_TRAIN", "DONOR_TRAIN", "REPLAY"):
                audits = {row["task_id"]: row for row in data.audit(split)}
                for task in data.public(split):
                    audit = audits[task["task_id"]]
                    metadata[task["task_id"]] = {
                        "root_id": task["root_id"],
                        "family": task["family"],
                        "center": audit["center"],
                        "corrupted_index": audit["corrupted_index"],
                    }
        seed = 108701 + job["block"]
        if seed not in schedule_cache:
            schedule_cache[seed] = {
                (r["arm"], r["update"], r["slot"]): r
                for r in read_bound(run, f"manifests/schedule_{seed}.jsonl")
            }
        totals = {}
        for update in range(1, 257):
            log = read_json(directory / f"update{update:03d}.json")
            slots = log.get("slots", [])
            if log.get("status") != "UPDATE_APPLIED" or log["update"] != update or len(slots) != 16:
                raise ValueError("Completed SFT is missing a persisted 16-slot update log")
            committed_updates += 1
            for slot, row in enumerate(slots):
                expected = schedule_cache[seed][job["arm"], update, slot]
                if (
                    any(row.get(key) != value for key, value in expected.items())
                    or row.get("sequence_coefficient") != 1 / 16
                ):
                    raise ValueError("Persisted exposure differs from frozen step/slot")
                key = row["role"], row["task_id"]
                if key not in totals:
                    totals[key] = {
                        "parent": job["parent"],
                        "block": job["block"],
                        "arm": job["arm"],
                        "block_seed": seed,
                        "role": row["role"],
                        "task_id": row["task_id"],
                        **metadata[row["task_id"]],
                        "exposures": 0,
                        "first_update": update,
                        "last_update": update,
                        "sequence_nll_sum": 0.0,
                        "target_token_count_sum": 0,
                        "coefficient": 1 / 16,
                    }
                item = totals[key]
                item["exposures"] += 1
                item["last_update"] = update
                item["sequence_nll_sum"] += row["sequence_nll"]
                item["target_token_count_sum"] += row["target_token_count"]
            # These are observed training NLLs, not gradients or held-out scores.
        for item in totals.values():
            item["sequence_nll_mean"] = item["sequence_nll_sum"] / item["exposures"]
        exposure_rows.extend(totals.values())
    starts = list((run / "evaluations").glob("*/attempts/*/*/STARTED.json"))
    failures = list((run / "evaluations").glob("*/attempts/*/*/FAILED.json"))
    unknown_generations = sum(
        not (path.parent / "FAILED.json").exists() and not (path.parent / "ACCEPTED.json").exists()
        for path in starts
    )
    accepted_samples = sum(len(raw) for raw in all_raw.values())
    partial_raw = []
    for job in jobs:
        if (
            evaluation_job_id(job) in all_raw
            or not (_evaluation_directory(run, job) / "manifest.json").exists()
        ):
            continue
        retained, _ = validate_completed_evaluation(run, job, require_complete=False)
        if any(row.get("execution_kind") != "REAL_FROZEN_GPU" for row in retained):
            raise ValueError("Partial CPU fixture observations cannot enter formal accounting")
        partial_raw.extend(retained)
    bridge_path = run / "bridge" / "BRIDGE_RESULT.json"
    bridge = read_json(bridge_path) if bridge_path.exists() else None
    scope = {
        "planned_formal_generations": 145280,
        "completed_formal_generations": accepted_samples,
        "registered_sft_runs": 18,
        "completed_sft_runs": committed_updates // 256,
        "committed_formal_updates_in_complete_students": committed_updates,
        "committed_target_exposures_in_complete_students": 16 * committed_updates,
        "physical_training_updates_observed": physical_updates,
        "physical_training_updates_with_unknown_completion": unknown_updates,
        "physical_training_update_gpu_seconds_observed": update_gpu_seconds,
        "physical_training_update_wall_seconds_observed": update_wall_seconds,
        "physical_generation_attempts_started": len(starts),
        "failed_generation_attempts": len(failures),
        "accepted_samples_in_incomplete_jobs": len(partial_raw),
        "accepted_formal_samples_all_jobs": accepted_samples + len(partial_raw),
        "partial_job_generation_wall_seconds": sum(row["elapsed_seconds"] for row in partial_raw),
        "generation_attempts_with_unknown_completion": unknown_generations,
        "completed_generation_wall_seconds": sum(
            row["elapsed_seconds"] for raw in all_raw.values() for row in raw
        ),
        "allocated_gpu_hours": None,
        "allocated_gpu_hours_status": "SCHEDULER_ACCOUNTING_NOT_IMPORTED",
        "technical_bridge": bridge,
        "technical_generations": 0 if bridge else None,
        "technical_missing": [
            row["id"] for row in queue_rows if row["status"] == "BLOCKED_TECHNICAL"
        ],
        "registered_evaluation_job_ids": [evaluation_job_id(job) for job in jobs],
        "new_rl_updates": 0,
        "new_discovery_campaign": 0,
        "additional_model_scoring": 0,
        "cost_note": (
            "Unknown interrupted work remains unknown; "
            "repeated physical work is included separately."
        ),
        "exposure_scope": (
            "Actual persisted slots of complete students; "
            "incomplete students are listed as missing."
        ),
    }
    return scope, exposure_rows


def release_and_analyze(run):
    """Unseal only the exact terminal matrix and pre-frozen analysis source."""
    from .queue import registered_queue
    from .reports import write_reports
    from .schema import read_bound, verify_frozen

    run = Path(run).resolve()
    frozen = verify_frozen(run)
    registry = read_json(run / "REGISTERED_MATRIX.json")
    jobs = validate_evaluation_jobs(read_bound(run, "manifests/evaluation_jobs.jsonl"))
    machine = read_json(run / "machine.json")
    queue = registered_queue(run)
    try:
        rows = queue.rows()
        if registry["jobs"] != [row["payload"] for row in rows]:
            raise ValueError("Registered queue matrix changed")
        queue_identity = queue.db.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if not queue_identity or queue_identity[0] != registry["identity"]:
            raise ValueError("Queue registration identity changed")
        training = [row for row in rows if row["payload"]["kind"] == "train"]
        evaluations = [row for row in rows if row["payload"]["kind"] == "eval"]
        train_keys = {
            (row["payload"]["parent"], row["payload"]["block"], row["payload"]["arm"])
            for row in training
        }
        if (
            len(rows) != 113
            or len(training) != 18
            or len(evaluations) != 95
            or train_keys != {(p, b, a) for p in PARENTS for b in range(3) for a in ARMS}
            or {row["id"] for row in evaluations} != {evaluation_job_id(j) for j in jobs}
        ):
            raise ValueError("Release requires the exact two-parent/three-order/three-arm matrix")
        if not queue.all_terminal():
            raise PermissionError("All 113 registered jobs must be terminal before release")
        if any(row["status"] not in ("COMPLETE", "BLOCKED_TECHNICAL") for row in rows):
            raise ValueError("SER has no alias or return-parent terminal status")
        source = Path(__file__).parent
        actual_analysis = {
            name: _file_hash(source / name)
            for name in ("statistics.py", "semantics.py", "reports.py")
        }
        if registry.get("analysis_identity") != actual_analysis:
            raise ValueError("Analysis code must be frozen before confirmation release")
        all_raw, completed = {}, []
        for row in training:
            if row["status"] != "COMPLETE":
                continue
            validate_training_endpoint(run, row["payload"], frozen)
        queue_by_id = {row["id"]: row for row in evaluations}
        for job in jobs:
            qrow = queue_by_id[evaluation_job_id(job)]
            if qrow["status"] != "COMPLETE":
                continue
            raw, receipt = validate_completed_evaluation(run, job, machine)
            if any(row.get("execution_kind") != "REAL_FROZEN_GPU" for row in raw):
                raise ValueError("CPU fixtures cannot enter a formal model report")
            all_raw[evaluation_job_id(job)] = raw
            completed.append(receipt)
        technical_missing = [row["id"] for row in rows if row["status"] == "BLOCKED_TECHNICAL"]
        scope, exposure_rows = _execution_accounting(run, rows, jobs, all_raw)
        receipt = {
            "schema": "ser-j2-final-release-v1",
            "status": "RELEASED",
            "plan_hash": frozen["plan_hash"],
            "matrix_digest": queue.seal_release(),
            "all_registered_models_terminal": True,
            "all_registered_terminal": True,
            "registered_training_jobs": 18,
            "registered_evaluation_jobs": 95,
            "analysis_code_sha256": digest(actual_analysis),
            "analysis_code_hashes": actual_analysis,
            "completed_evaluation_receipts": completed,
            "technical_missing": technical_missing,
            "scientific_completion": not technical_missing,
        }
        _write_once(run / "FINAL_RELEASE.json", receipt)
    finally:
        queue.db.close()
    scored = []
    for job in jobs:
        raw = all_raw.get(evaluation_job_id(job))
        if raw is not None:
            scored.extend(_score_rows(run, job, raw))
    return write_reports(
        run / "reports",
        scored,
        release_receipt=receipt,
        execution_scope=scope,
        exposure_rows=exposure_rows,
    )
