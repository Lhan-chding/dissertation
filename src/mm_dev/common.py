"""Identity and accounting gates shared by the independently scoped F2 workers."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from mm_core.execution import (
    append_jsonl,
    checked_path,
    locked,
    read_json,
    sha256_file,
    utc_now,
)

from .contract import PLAN_ID

PLAN_SHA256 = "2e20d757d32493cde1f27184b40e15b7077df594a8c95ae0e4ad2705df5515bb"
BUNDLE_MANIFEST_SHA256 = "c45a10fd8745c93f9d732eec34e98e6d7c74692baf19b82e1f9a650e71e292a1"
ENGINE_PASSES = {"PASS_NATURAL_GRPO_RESUME", "PASS_NONZERO_SURROGATE_KERNEL_RESUME"}


def load_plan(plan_path):
    """Accept only the user-dispatched bytes, including its data and stream manifest."""
    path = Path(plan_path).resolve(strict=True)
    if sha256_file(path) != PLAN_SHA256:
        raise PermissionError("MM-DEV F2 plan differs from the user-dispatched identity")
    plan = read_json(path)
    if plan["plan_id"] != PLAN_ID:
        raise PermissionError("Unregistered plan ID")
    bundle = path.parent.parent
    if sha256_file(bundle / "BUNDLE_MANIFEST.json") != BUNDLE_MANIFEST_SHA256:
        raise PermissionError("Dispatched bundle manifest changed")
    manifest = read_json(bundle / "BUNDLE_MANIFEST.json")
    if manifest.get("plan_id") != PLAN_ID:
        raise PermissionError("Plan bundle identity differs")
    paths = set()
    for item in manifest["files"]:
        if item["path"] in paths:
            raise PermissionError("Duplicate plan manifest path")
        paths.add(item["path"])
        source = checked_path(bundle, item["path"])
        if source.is_symlink() or source.stat().st_size != item["bytes"]:
            raise PermissionError("Plan bundle file type/size changed: " + item["path"])
        if sha256_file(source) != item["sha256"]:
            raise PermissionError("Plan bundle file identity changed: " + item["path"])
    return plan


def verify_execution(plan_path, run_root, *, require_engine=False, full_hashes=False):
    """Recheck executable identities; individual data reads also check their own hashes.

    Full payload verification is available at freeze/release. The scheduler uses
    compact manifest verification between ticks, without rereading image bodies.
    """
    plan = load_plan(plan_path)
    root = Path(run_root).resolve(strict=True)
    freeze = read_json(root / "manifests/F2_FREEZE.json")
    if (
        freeze.get("plan_id") != PLAN_ID
        or freeze.get("status") != "FROZEN"
        or freeze.get("authorized") is not True
        or freeze.get("run_root") != str(root)
        or freeze.get("plan_sha256") != PLAN_SHA256
    ):
        raise PermissionError("Independent authorized F2 freeze is required")
    identity = freeze.get("model_identity", {})
    for field in (
        "model_id",
        "model_revision",
        "model_weights_hash",
        "processor_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
    ):
        if identity.get(field) != plan["model"][field]:
            raise PermissionError("Frozen model identity mismatch: " + field)
    if not identity.get("model_files"):
        raise PermissionError("Frozen per-file model identities missing")
    if freeze.get("resource_policy") != plan["resource"]:
        raise PermissionError("F2 resource contract changed")
    source_root = Path(__file__).resolve().parents[2]
    sources = freeze.get("source_hashes", {})
    if not sources or "src/mm_dev/common.py" not in sources:
        raise PermissionError("F2 executable source manifest missing")
    actual_sources = {
        str(path.relative_to(source_root))
        for directory in ("src/mm_core", "src/mm_dev", "scripts/mm_dev")
        for path in (source_root / directory).rglob("*.py")
    }
    if set(sources) != actual_sources:
        raise PermissionError("Frozen executable source set is incomplete or changed")
    for relative, expected in sources.items():
        if sha256_file(checked_path(source_root, relative)) != expected:
            raise PermissionError("Frozen executable source changed: " + relative)
    inputs = freeze.get("input_hashes", {})
    if not inputs or "data/questions.jsonl" not in inputs:
        raise PermissionError("F2 complete input manifest missing")
    for relative, expected in inputs.items():
        if (full_hashes or relative.endswith((".json", ".jsonl"))) and (
            sha256_file(checked_path(root, relative)) != expected
        ):
            raise PermissionError("Frozen input identity changed: " + relative)
    if require_engine:
        gate = read_json(root / "manifests/ENGINE_GATE.json")
        if gate.get("status") not in ENGINE_PASSES:
            raise PermissionError("New F2 engineering gate has not passed")
        if gate.get("freeze_sha256") != sha256_file(root / "manifests/F2_FREEZE.json"):
            raise PermissionError("Engineering gate is not bound to this exact freeze")
        evidence = gate.get("artifact_hashes", {})
        if not evidence:
            raise PermissionError("F2 engineering evidence identities are missing")
        for relative, expected in evidence.items():
            if sha256_file(checked_path(root, relative)) != expected:
                raise PermissionError("Verified F2 engineering evidence changed: " + relative)
    return freeze


def verify_analysis_readiness(plan_path, run_root):
    """Require all registered paths and immutable observations before any science release."""
    from mm_core.training import state_hash

    from .contract import digest, run_matrix, seed
    from .evaluation import load_panel
    from .runtime import get_state_identity

    root, plan_path = Path(run_root).resolve(strict=True), Path(plan_path).resolve(strict=True)
    plan = load_plan(plan_path)
    verify_execution(plan_path, root, require_engine=True)
    identities = {}

    def track(relative, expected=None):
        actual = sha256_file(checked_path(root, relative))
        if expected is not None and actual != expected:
            raise PermissionError("Release evidence identity changed: " + relative)
        identities[relative] = actual

    starts = ("S0", "S_A_0", "S_AP_0", "S_A_1", "S_AP_1")
    matrix = run_matrix()
    questions = {
        x["question_id"]: x
        for x in (
            json.loads(line) for line in (root / "data/questions.jsonl").read_text().splitlines()
        )
    }
    training_count = 0
    for run in matrix:
        base = "training/" + run["run_id"]
        complete = read_json(root / base / "COMPLETE.json")
        if complete.get("final_step") != 32 or complete["checkpoint"]["step"] != 32:
            raise PermissionError("Scientific path has not reached registered step32")
        state = get_state_identity(plan, root, run.get("output_state", run["run_id"]))
        if (
            state.get("producer_run_id") != run["run_id"]
            or state.get("logical_step") != 32
            or state.get("checkpoint_sha256") != complete["checkpoint"]["sha256"]
        ):
            raise PermissionError("Scientific endpoint is not bound to completed path")
        track(base + "/COMPLETE.json")
        latest = read_json(root / base / "checkpoints/LATEST.json")
        if latest != complete["checkpoint"]:
            raise PermissionError("Completed checkpoint is not current committed state")
        if state.get("trainable_state_hash") != latest["field_hashes"]["parameters"]:
            raise PermissionError("Scientific endpoint parameter identity differs")
        track(base + "/checkpoints/" + latest["path"], latest["sha256"])
        stream = [
            json.loads(line)
            for line in (plan_path.parent / "schedules" / (run["schedule_id"] + ".jsonl"))
            .read_text()
            .splitlines()
        ]
        for step in range(1, 33):
            relative = base + f"/steps/{step:02d}.json"
            metrics = read_json(root / relative)
            commit_rel = base + f"/checkpoints/commit-{step:02d}.json"
            commit = read_json(root / commit_rel)
            if (
                metrics["logical_step"] != step
                or len(metrics["groups"]) != 24
                or commit["step"] != step
                or commit["field_hashes"]["diagnostics_hash"] != state_hash(digest(metrics))
                or metrics.get("effective_sequences") != 192
                or metrics.get("stress_coefficient_override") is not False
                or commit["field_hashes"]["parameters"] != metrics["parameter_hash_after"]
                or commit["field_hashes"]["input_stream_hash"] != state_hash(digest(stream))
            ):
                raise PermissionError("Committed scientific step evidence is incomplete")
            track(relative)
            track(commit_rel)
            scheduled = {x["slot"]: x for x in stream if x["logical_step"] == step}
            groups = metrics["groups"]
            token_path = []
            if len({g["slot"] for g in groups}) != 24:
                raise PermissionError("Repeated training prompt slot")
            for group in groups:
                slot = group["slot"]
                if any(group.get(key) != value for key, value in scheduled[slot].items()):
                    raise PermissionError("Training group differs from registered input stream")
                if any(
                    len(group[key]) != 8 for key in ("completion_rewards", "rewards", "advantages")
                ):
                    raise PermissionError("Training group lacks all eight completions")
                row = questions[group["question_id"]]
                for index in range(8):
                    rel = base + f"/rollouts/{step:02d}-{slot:02d}-{index}.json"
                    raw = read_json(root / rel)
                    expected = dict(
                        run_id=run["run_id"],
                        logical_step=step,
                        slot=slot,
                        sample_index=index,
                        question_id=group["question_id"],
                        seed=seed(
                            "rollout", run["phase"], run["repeat"], group["question_id"], index
                        ),
                        policy_hash=metrics["parameter_hash_before"],
                        request_id=digest(
                            [
                                PLAN_ID,
                                run["phase"],
                                run["run_id"],
                                step,
                                slot,
                                index,
                                metrics["parameter_hash_before"],
                            ]
                        ),
                    )
                    if any(raw.get(key) != value for key, value in expected.items()):
                        raise PermissionError("Scientific rollout identity mismatch")
                    if raw["record_hash"] != digest(
                        {k: v for k, v in raw.items() if k != "record_hash"}
                    ):
                        raise PermissionError("Scientific rollout content hash mismatch")
                    if raw.get("input_hash") != digest(
                        {k: row[k] for k in ("question_id", "image_path", "image_sha256", "prompt")}
                    ):
                        raise PermissionError("Scientific rollout input binding mismatch")
                    if (
                        raw.get("generation_status") != "COMPLETE"
                        or raw.get("technical_validation_errors")
                        or state_hash(raw.get("sampling_hash"))
                        != commit["field_hashes"]["sampling_hash"]
                    ):
                        raise PermissionError("Scientific rollout technical identity mismatch")
                    token_path.append(
                        {
                            key: raw[key]
                            for key in (
                                "question_id",
                                "tokens",
                                "raw_text",
                                "image_routing",
                                "old_logprobs",
                                "seed",
                            )
                        }
                    )
                    track(rel)
                    training_count += 1
            token_hash = digest(token_path)
            if (
                token_hash != metrics["token_path_hash"]
                or state_hash(token_hash) != commit["field_hashes"]["token_path_hash"]
            ):
                raise PermissionError("Scientific token trajectory differs from committed update")
        for rel, sha in state["adapter_file_hashes"].items():
            track(state["adapter_path"] + "/" + rel, sha)
    models = (*starts, *(r["run_id"] for r in matrix if r["phase"] == "CONTINUE"))
    counts = {
        "training_rollouts": training_count,
        "probe_completions": 0,
        "evaluation_completions": 0,
        "probe_self_score_records": 0,
        "probe_gold_score_records": 0,
    }
    for panel, states in (("PROBE", starts), ("DEV_EVAL", models)):
        for state_id in states:
            observed = load_panel(plan_path, root, panel, state_id)
            expected_count = 6144 if panel == "PROBE" else 12288
            if not observed["complete"] or len(observed["outputs"]) != expected_count:
                raise PermissionError("Registered measurement panel is incomplete")
            key = "probe_completions" if panel == "PROBE" else "evaluation_completions"
            counts[key] += len(observed["outputs"])
            if panel == "PROBE":
                if len(observed["self_scores"]) != 6144 or len(observed["gold_scores"]) != 1536:
                    raise PermissionError("Registered PROBE field scoring is incomplete")
                counts["probe_self_score_records"] += 6144
                counts["probe_gold_score_records"] += 1536
            for receipt in observed["receipts"]:
                for relative, sha in receipt["artifacts"].items():
                    track(relative, sha)
                track(
                    f"raw/{panel}/{state_id}/shard_{receipt['shard']}of"
                    f"{receipt['shards']}/COMPLETE.json"
                )
    expected_counts = {
        **plan["expected_counts"],
        "probe_self_score_records": 30720,
        "probe_gold_score_records": 7680,
    }
    if any(value != expected_counts[key] for key, value in counts.items()):
        raise PermissionError("Full scientific count reconciliation failed")
    cost = read_json(root / "manifests/GPU_ACCOUNTING_FINAL.json")
    registration = read_json(root / "orchestration/REGISTRATION.json")
    from .orchestration import digest as registration_digest
    from .orchestration import summarize_accounting

    if (
        cost.get("status") != "COMPLETE"
        or cost.get("plan_id") != PLAN_ID
        or cost.get("registration_hash") != registration_digest(registration)
        or cost.get("policy") != "ACCOUNTING_ONLY"
    ):
        raise PermissionError("Complete scheduler cost reconciliation is required")
    scheduler_state = read_json(root / "orchestration/STATE.json")
    gpu_tasks = {k: v for k, v in registration["tasks"].items() if v["gpus"]}
    expected_attempts = {}
    for task_id in gpu_tasks:
        task = scheduler_state["tasks"][task_id]
        if task["status"] != "COMPLETE" or not task["attempts"]:
            raise PermissionError("Scientific GPU task accounting is incomplete")
        for attempt in task["attempts"]:
            key = (task_id, attempt["attempt_id"], attempt["job_id"])
            if key in expected_attempts:
                raise PermissionError("Duplicate scientific GPU attempt")
            expected_attempts[key] = attempt
    actual_keys = [(a["task_id"], a["attempt_id"], a["job_id"]) for a in cost["allocations"]]
    if (
        not gpu_tasks
        or cost.get("gpu_task_count") != len(gpu_tasks)
        or len(set(actual_keys)) != len(actual_keys)
        or set(actual_keys) != set(expected_attempts)
    ):
        raise PermissionError("Scientific GPU attempt coverage differs")
    total_seconds = 0
    for allocation in cost["allocations"]:
        record = allocation["scheduler_observation"]
        track(record["path"], record["sha256"])
        observation = read_json(root / record["path"])
        if observation["queue"]:
            raise PermissionError("Scientific GPU allocation is still queued or running")
        attempt = expected_attempts[
            (allocation["task_id"], allocation["attempt_id"], allocation["job_id"])
        ]
        recounted = summarize_accounting(
            observation["accounting"],
            attempt,
            registration["permission"],
            gpu_tasks[allocation["task_id"]]["gpus"],
        )
        if any(allocation.get(key) != value for key, value in recounted.items()):
            raise PermissionError("Scientific GPU accounting differs from scheduler evidence")
        total_seconds += recounted["gpu_seconds"]
    if cost["actual_gpu_seconds"] != total_seconds or not math.isclose(
        cost["actual_gpu_hours"], total_seconds / 3600
    ):
        raise PermissionError("Scientific GPU total differs from actual allocation intervals")
    for relative in (
        "manifests/GPU_ACCOUNTING_FINAL.json",
        "orchestration/REGISTRATION.json",
        "manifests/STATE_REGISTRY.json",
        "manifests/COMMON_LORA.json",
        "manifests/ENGINE_GATE.json",
    ):
        track(relative)
    for ledger in sorted((root / "accounting").glob("*.jsonl")):
        track(str(ledger.relative_to(root)))
    return dict(
        status="COMPLETE",
        plan_id=PLAN_ID,
        freeze_sha256=sha256_file(root / "manifests/F2_FREEZE.json"),
        scientific_training_paths=34,
        probe_models=5,
        evaluation_models=35,
        counts=counts,
        verified_evidence_files=len(identities),
        evidence_set_sha256=digest(identities),
        actual_gpu_seconds=cost["actual_gpu_seconds"],
    )


def _slurm_fields(text):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", text))


def verify_live_allocation(fields, allocation, permission, *, user, job_id):
    """Bind an unrequeued single-device grant to its immutable submission intent."""
    if fields.get("JobId") != str(job_id) or fields.get("JobState") != "RUNNING":
        raise PermissionError("Registered Slurm allocation is not running")
    owner = fields.get("UserId", "").split("(")[0]
    if owner != user or owner != permission["owner"] or owner != allocation["owner"]:
        raise PermissionError("Slurm allocation owner differs")
    for live, key in (("Account", "account"), ("QOS", "qos")):
        if fields.get(live) != permission[key] or fields.get(live) != allocation[key]:
            raise PermissionError("Allocation permission differs: " + key)
    if permission.get("partition") and (
        fields.get("Partition") not in permission["partition"].split(",")
    ):
        raise PermissionError("Allocation partition differs")
    for live, key in (("JobName", "job_name"), ("Comment", "comment")):
        if not allocation.get(key) or fields.get(live) != allocation[key]:
            raise PermissionError("Slurm submission intent differs: " + key)
    if fields.get("NumNodes") != "1":
        raise PermissionError("Exactly one allocated node is required")
    if fields.get("Requeue") != "0" or fields.get("Restarts") != "0":
        raise PermissionError("Implicit Slurm requeue/restart is forbidden; register a new attempt")
    allocated = dict(
        item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item
    )
    requested_gpu = [
        item
        for item in fields.get("TresPerNode", "").lower().split(",")
        if item.startswith("gres/gpu")
    ]
    actual_gpu = {key: value for key, value in allocated.items() if key.startswith("gres/gpu")}
    if actual_gpu != {"gres/gpu": "1", "gres/gpu:pro6000": "1"} or requested_gpu != [
        "gres/gpu:pro6000:1"
    ]:
        raise PermissionError("Exactly one actually allocated Pro6000 is required")
    return True


def require_allocation(run_root, task_id):
    # Registration and attempt manifests are immutable. ALLOCATIONS is an atomic
    # projection and may advance while other workers start; bind this attempt to
    # the registration itself rather than trusting its environment/projection alone.
    from .orchestration import digest

    root = Path(run_root).resolve(strict=True)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise PermissionError("GPU execution requires a registered Slurm allocation")
    permission_path = root / "manifests/ALLOCATION_PERMISSION.json"
    permission = read_json(permission_path)
    if (
        permission.get("plan_id") != PLAN_ID
        or permission.get("run_root") != str(root)
        or permission.get("authorized") is not True
        or permission.get("max_gpus_concurrent") != 5
        or permission.get("gpus_per_worker") != 1
        or permission.get("gres") != "gpu:pro6000:1"
    ):
        raise PermissionError("Current F2 GPU permission receipt is required")
    registration = read_json(root / "orchestration/REGISTRATION.json")
    registration_hash = digest(registration)
    if (
        registration.get("plan_id") != PLAN_ID
        or registration.get("permission") != permission
        or registration.get("permission_receipt_sha256") != sha256_file(permission_path)
        or registration.get("freeze_sha256") != sha256_file(root / "manifests/F2_FREEZE.json")
        or registration_hash != os.environ.get("MM_DEV_REGISTRATION_HASH")
        or registration.get("tasks", {}).get(task_id, {}).get("gpus") != 1
        or os.environ.get("MM_DEV_TASK_ID") != task_id
    ):
        raise PermissionError("Immutable allocation registration identity differs")
    registry = read_json(root / "manifests/ALLOCATIONS.json")
    if (
        registry.get("plan_id") != PLAN_ID
        or registry.get("permission_receipt_sha256") != sha256_file(permission_path)
        or registry.get("registration_hash") != registration_hash
    ):
        raise PermissionError("Allocation registration identity changed")
    allocations = registry["allocations"]
    matches = [a for a in allocations if str(a.get("job_id")) == job_id]
    if len(matches) != 1:
        raise PermissionError("This job has no unique allocation")
    allocation = matches[0]
    active = {"REGISTERED", "ACTIVE", "RUNNING"}
    if allocation.get("task_id") != task_id or allocation.get("status") not in active:
        raise PermissionError("This task/job has no unique active allocation")
    unresolved = active | {"SUBMITTING", "UNKNOWN"}
    if sum(a.get("task_id") == task_id and a.get("status") in unresolved for a in allocations) != 1:
        raise PermissionError("Multiple active attempts for one logical task")
    if (
        type(allocation.get("gpus")) is not int
        or allocation["gpus"] != 1
        or (allocation.get("gres") != permission["gres"])
    ):
        raise PermissionError("Registered GPU resources differ")
    attempt_id = allocation.get("attempt_id")
    if (
        not isinstance(attempt_id, str)
        or not re.fullmatch(re.escape(task_id) + r"_attempt[0-9]{4,}", attempt_id)
        or os.environ.get("MM_DEV_ATTEMPT_ID") != attempt_id
        or sum(a.get("attempt_id") == attempt_id for a in allocations) != 1
    ):
        raise PermissionError("Worker attempt identity differs")
    expected_path = f"orchestration/attempts/{attempt_id}.json"
    if allocation.get("attempt_manifest_path") != expected_path:
        raise PermissionError("Allocation submission-intent path differs")
    attempt_path = checked_path(root, expected_path)
    attempt = read_json(attempt_path)
    prefix = registration_hash[:12]
    expected_intent = {
        "plan_id": PLAN_ID,
        "registration_hash": registration_hash,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "manifest_path": expected_path,
        "job_name": f"mmdev-{prefix}-{attempt_id}",
        "comment": f"mmdev:{prefix}:{attempt_id}",
    }
    if any(attempt.get(key) != value for key, value in expected_intent.items()):
        raise PermissionError("Immutable submission intent identity differs")
    if any(allocation.get(key) != attempt[key] for key in ("job_name", "comment")):
        raise PermissionError("Allocation projection differs from submission intent")
    # The intent is persisted BEFORE sbatch, so its job_id is legitimately null.
    # The eventual job ID is bound by ALLOCATIONS plus exact live name/comment.
    if attempt.get("job_id") not in (None, job_id):
        raise PermissionError("Submission intent specifies a different job")
    allocation = {**allocation, "job_name": attempt["job_name"], "comment": attempt["comment"]}
    completed = subprocess.run(
        ["scontrol", "show", "job", job_id, "-o"],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    user = subprocess.run(
        ["id", "-un"], check=True, text=True, capture_output=True, timeout=10
    ).stdout.strip()
    fields = _slurm_fields(completed.stdout)
    verify_live_allocation(fields, allocation, permission, user=user, job_id=job_id)
    return {
        "allocation": allocation,
        "live": fields,
        "registration_sha256": sha256_file(root / "orchestration/REGISTRATION.json"),
        "attempt_manifest_sha256": sha256_file(attempt_path),
        "verified_at": utc_now(),
    }


class CostLedger:
    """Append physical attempts without imposing a GPU-hour or study walltime gate."""

    def __init__(self, run_root, task_id):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
            raise ValueError("Invalid task identity")
        self.root = Path(run_root).resolve()
        self.task_id = task_id
        self.path = self.root / "accounting" / (task_id + ".jsonl")

    def reserve(self, kind, count, metadata=None, **extra):
        if not re.fullmatch(r"[a-z][a-z0-9_]+", kind):
            raise ValueError("Invalid cost category")
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise ValueError("Cost count must be numeric")
        if not math.isfinite(count) or count < 0:
            raise ValueError("Cost count must be finite and nonnegative")
        record = dict(
            plan_id=PLAN_ID,
            task_id=self.task_id,
            kind=kind,
            count=count,
            metadata={**(metadata or {}), **extra},
            at=utc_now(),
            pid=os.getpid(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            measure="physical_attempt",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with locked(self.path.with_suffix(".lock")):
            append_jsonl(self.path, record)
        return record
