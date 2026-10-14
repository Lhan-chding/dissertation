"""One authorized full-K replay, original-row reuse, and immutable provenance.

Scientific source and old recovery plans stay unchanged. The physical replay is
charged to a separate ledger. Original endpoint ledgers receive only subsequent
ordinary generation events; no synthetic completion/token events are inserted.
"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from sr_f2.orchestration import task_id, verify_task
from sr_f2_recovery.orchestration import Recovery
from sr_f12.protocol import object_hash


def read_json(path):
    return _json(Path(path).read_text())


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key in replay evidence")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("Nonfinite replay JSON")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def within(root, value, *, exists=True):
    root, value = Path(root).resolve(strict=True), Path(value)
    require(not value.is_absolute() and ".." not in value.parts, "Unsafe replay relative path")
    target = root / value
    resolved = target.resolve(strict=exists)
    require(resolved.is_relative_to(root) and resolved == target, "Replay path escapes or aliases")
    return target


def durable(path, value):
    from sr_f12.evaluation import durable_json

    durable_json(path, value)
    return value


def exclusive_intent(path, value):
    """No second process may win a replay attempt, even before its first reservation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rows(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            require(bool(line.strip()), "Blank replay raw/accounting row")
            row = _json(line)
            require(isinstance(row, dict), "Malformed replay raw/accounting row")
            rows.append(row)
    return rows


def build_plan_bindings(parent_plan_path, *, parent_factory=Recovery):
    """Read-only plan scaffold using exact frozen slots and original audit integers."""
    from sr_f1.data import load_tasks
    from sr_f2.workers import endpoint_slots
    from sr_f12.evaluation import group_slots

    parent = parent_factory(parent_plan_path)
    parent.verify()
    questions = load_tasks(parent.root)
    targets = {}
    for key, old in parent.allowed.items():
        if old["status"] != "BLOCKED_INCOMPLETE_GENERATION":
            continue
        task = old["task"]
        require(task["kind"] == "endpoint", "Only interrupted endpoints are authorized")
        gaps = [
            gap
            for gap in old["generation_audit"]["gaps"]
            if gap["code"] == "ATTEMPT_WITHOUT_DURABLE_GROUP"
        ]
        require(
            len(gaps) == 1 and gaps[0]["count"] == 4 and gaps[0]["groups"] == 1,
            "Target does not have exactly one interrupted K4 group",
        )
        seed = gaps[0]["group_seed"]
        candidates = [
            group
            for group in group_slots(endpoint_slots(questions, task["model_id"]))
            if len(group["slots"]) == 4 and group["slots"][0]["group_seed"] == seed
        ]
        require(len(candidates) == 1, "Interrupted seed does not identify one frozen K4 group")
        group = candidates[0]
        pending = (
            parent.root
            / "sealed/evaluation"
            / task["model_id"]
            / "groups"
            / (group["group_id"] + ".pending.jsonl")
        )
        present = pending.exists()
        count = len(_rows(pending)) if present else 0
        require(
            count in (0, 3) and (not present or count == 3), "Unexpected original partial group"
        )
        ledger_relative = f"accounting/endpoint-{task['model_id']}.jsonl"
        ledger = within(parent.root, ledger_relative)
        baseline = old["mutable_prelaunch_files"]
        ledger_ref = next(ref for ref in baseline if ref["path"] == ledger_relative)
        require(file_hash(ledger) == ledger_ref["sha256"], "Original endpoint ledger changed")
        checkpoint_relative = f"runs/{task['model_id']}/checkpoints/commit-96.json"
        checkpoint = within(parent.root, checkpoint_relative)
        targets[key] = dict(
            task=task,
            group_id=group["group_id"],
            group_seed=seed,
            count=4,
            preserved_count=count,
            pending_path=str(pending.relative_to(parent.root)) if present else None,
            pending_sha256=file_hash(pending) if present else None,
            baseline_files=baseline,
            original_ledger_path=ledger_relative,
            original_ledger_sha256=ledger_ref["sha256"],
            original_ledger_bytes=ledger.stat().st_size,
            slots_sha256=object_hash(group["slots"]),
            checkpoint_commit_ref=dict(path=checkpoint_relative, sha256=file_hash(checkpoint)),
            checkpoint_payload_sha256=read_json(checkpoint)["sha256"],
            input_identity_sha256=file_hash(parent.root / "INPUT_IDENTITY.json"),
            model_environment_sha256=file_hash(parent.root / "MODEL_ENVIRONMENT_IDENTITY.json"),
        )
    require(
        len(targets) == 5 and sum(spec["preserved_count"] for spec in targets.values()) == 3,
        "Authorized five-group/three-preserved-row scope differs",
    )
    return dict(
        original_root=str(parent.root),
        parent_plan_path=str(parent.path),
        parent_plan_sha256=parent.sha256,
        parent_audit_sha256=object_hash(parent.audit),
        registration_sha256=parent.plan["registration_sha256"],
        task_matrix_sha256=parent.audit["task_matrix_sha256"],
        max_run_gpus=5,
        tasks=targets,
    )


class ReplayPlan:
    def __init__(self, path, *, parent_factory=Recovery):
        self.path = Path(path).resolve(strict=True)
        self.directory = self.path.parent
        self.plan = read_json(self.path)
        self.sha256 = object_hash(self.plan)
        self.root = Path(self.plan["original_root"]).resolve(strict=True)
        self.extension = Path(self.plan["extension_root"]).resolve(strict=True)
        self.parent = parent_factory(self.plan["parent_plan_path"])
        self.code = self.parent.code
        self.tasks = self.plan["tasks"]
        replay_id = self.plan["replay_id"]
        require(
            isinstance(replay_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", replay_id),
            "Unsafe replay identity",
        )
        require(
            self.path == self.root / "replay" / replay_id / "PLAN.json",
            "Replay plan is outside its dedicated directory",
        )
        require(
            self.root == self.parent.root and self.plan["max_run_gpus"] == 5,
            "Replay root or five-GPU cap differs",
        )
        require(
            self.extension != self.code and not self.extension.is_relative_to(self.code),
            "Replay extension must be separate from frozen source",
        )
        require(
            isinstance(self.tasks, dict) and len(self.tasks) == 5,
            "Exactly five interrupted endpoint groups are authorized",
        )
        blocked = {
            key
            for key, row in self.parent.allowed.items()
            if row["status"] == "BLOCKED_INCOMPLETE_GENERATION"
        }
        require(set(self.tasks) == blocked, "Replay changes the original blocked-task set")
        for key, spec in self.tasks.items():
            old = self.parent.allowed[key]
            task = spec["task"]
            require(
                task_id(task) == key and task == old["task"] and task["kind"] == "endpoint",
                "Replay task differs from original endpoint",
            )
            require(
                spec["count"] == 4
                and type(spec["group_seed"]) is int
                and spec["preserved_count"] in (0, 3)
                and re.fullmatch(r"[0-9a-f]{64}", spec["group_id"]),
                "Replay group/K/preserved-row identity differs",
            )
            gaps = [
                gap
                for gap in old["generation_audit"]["gaps"]
                if gap["code"] == "ATTEMPT_WITHOUT_DURABLE_GROUP"
            ]
            require(
                gaps
                == [
                    dict(
                        code="ATTEMPT_WITHOUT_DURABLE_GROUP",
                        count=4,
                        group_seed=spec["group_seed"],
                        groups=1,
                    )
                ],
                "Replay target is not the single original interrupted attempt",
            )
            require(
                spec["baseline_files"] == old["mutable_prelaunch_files"],
                "Replay baseline differs from immutable original audit",
            )
            ledger = f"accounting/endpoint-{task['model_id']}.jsonl"
            require(
                spec["original_ledger_path"] == ledger
                and dict(path=ledger, sha256=spec["original_ledger_sha256"])
                in spec["baseline_files"]
                and type(spec["original_ledger_bytes"]) is int
                and spec["original_ledger_bytes"] > 0,
                "Replay original ledger identity differs",
            )
            pending = (
                f"sealed/evaluation/{task['model_id']}/groups/{spec['group_id']}.pending.jsonl"
            )
            if spec["preserved_count"]:
                require(
                    spec["pending_path"] == pending
                    and dict(path=pending, sha256=spec["pending_sha256"]) in spec["baseline_files"],
                    "Original partial raw evidence differs",
                )
            else:
                require(
                    spec.get("pending_path") is None and spec.get("pending_sha256") is None,
                    "Unexpected partial evidence for zero-row interruption",
                )
        require(
            sum(spec["preserved_count"] for spec in self.tasks.values()) == 3,
            "Replay must preserve exactly three original rows and add seventeen",
        )

    def verify(self):
        """Check immutable global bindings only; running tasks may append normal outputs."""
        self.parent.verify()
        require(
            self.parent.sha256 == self.plan["parent_plan_sha256"]
            and object_hash(self.parent.audit) == self.plan["parent_audit_sha256"]
            and self.parent.plan["registration_sha256"] == self.plan["registration_sha256"]
            and self.parent.audit["task_matrix_sha256"] == self.plan["task_matrix_sha256"],
            "Original recovery plan, audit, or registration changed",
        )
        require(read_json(self.path) == self.plan, "Replay plan changed after loading")
        authorization_path = within(self.directory, self.plan["authorization_path"])
        require(
            file_hash(authorization_path) == self.plan["authorization_sha256"],
            "Replay authorization bytes changed",
        )
        authorization = read_json(authorization_path)
        require(
            all(
                authorization.get(key) == value
                for key, value in dict(
                    status="USER_AUTHORIZED",
                    purpose="one_replay_per_interrupted_endpoint_group",
                    max_groups=5,
                    max_group_size=4,
                    max_physical_replay_rows=20,
                    max_new_canonical_rows=17,
                    retain_original_rows=True,
                ).items()
            )
            and set(authorization.get("tasks", [])) == set(self.tasks)
            and len(authorization.get("tasks", [])) == 5,
            "Replay authorization scope differs",
        )
        manifest_path = within(self.directory, self.plan["extension_manifest"])
        require(
            file_hash(manifest_path) == self.plan["extension_manifest_sha256"],
            "Replay extension manifest changed",
        )
        manifest = read_json(manifest_path)
        require(
            manifest.get("commit") == self.plan["extension_commit"],
            "Replay extension revision differs",
        )
        required = {
            "src/sr_f2_replay/__init__.py",
            "src/sr_f2_replay/replay.py",
            "src/sr_f2_replay/orchestration.py",
            "scripts/sr_f2/replay_endpoint.py",
            "scripts/sr_f2/replay_recovery.py",
        }
        files = manifest.get("files", {})
        require(
            isinstance(files, dict) and required <= set(files),
            "Replay source manifest is incomplete",
        )
        for relative, expected in files.items():
            require(
                file_hash(within(self.extension, relative)) == expected,
                "Replay extension source bytes changed",
            )
        for spec in self.tasks.values():
            checkpoint = within(self.root, spec["checkpoint_commit_ref"]["path"])
            require(
                file_hash(checkpoint) == spec["checkpoint_commit_ref"]["sha256"]
                and read_json(checkpoint)["sha256"] == spec["checkpoint_payload_sha256"]
                and file_hash(self.root / "INPUT_IDENTITY.json") == spec["input_identity_sha256"]
                and file_hash(self.root / "MODEL_ENVIRONMENT_IDENTITY.json")
                == spec["model_environment_sha256"],
                "Replay checkpoint/model/input identity changed",
            )
        require(
            Path(__file__).resolve() == self.extension / "src/sr_f2_replay/replay.py",
            "Loaded replay code differs from bound extension",
        )
        return self.sha256

    def task_directory(self, task):
        key = task_id(task)
        require(key in self.tasks and self.tasks[key]["task"] == task, "Unknown replay task")
        return self.directory / "tasks" / key

    def baseline(self, task):
        spec = self.tasks[task_id(task)]
        for ref in spec["baseline_files"]:
            require(
                file_hash(within(self.root, ref["path"])) == ref["sha256"],
                "Original replay baseline changed before first attempt",
            )
        ledger = within(self.root, spec["original_ledger_path"])
        require(
            ledger.stat().st_size == spec["original_ledger_bytes"],
            "Original endpoint ledger length changed",
        )
        canonical = (
            self.root
            / "sealed/evaluation"
            / task["model_id"]
            / "groups"
            / (spec["group_id"] + ".json")
        )
        require(not canonical.exists(), "Interrupted group already has canonical output")
        if not spec["preserved_count"]:
            require(
                not canonical.with_suffix(".pending.jsonl").exists(),
                "Previously absent interrupted raw group appeared",
            )


def replay_account(plan, task):
    directory = plan.task_directory(task)
    path = directory / "accounting.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    spec = plan.tasks[task_id(task)]

    def reserve(kind, count, metadata):
        require(type(count) is int and count >= 0, "Invalid replay accounting count")
        if kind == "completion_attempts":
            require(
                count == 4 and metadata.get("group_seed") == spec["group_seed"],
                "Replay generation differs from authorized group",
            )
        elif kind == "generated_tokens":
            require(
                metadata.get("group_seed") == spec["group_seed"]
                and type(metadata.get("row_index")) is int
                and 0 <= metadata["row_index"] < 4
                and count > 0,
                "Replay generated-token accounting differs",
            )
        else:
            raise PermissionError("Unregistered compute operation in bounded replay")
        record = dict(
            experiment_id="SR-F2-20261013",
            task_id=task_id(task),
            replay_id=plan.plan["replay_id"],
            plan_sha256=plan.sha256,
            kind=kind,
            count=count,
            metadata=metadata,
            pid=os.getpid(),
            time_ns=time.time_ns(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        with path.open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            if kind == "completion_attempts":
                # The lock binds the one physical replay budget across processes.
                require(
                    not any(row["kind"] == kind for row in _rows(path)),
                    "Replay attempt has already been charged; no second generation",
                )
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return reserve


def verify_replay_account(plan, task, rows):
    directory, spec = plan.task_directory(task), plan.tasks[task_id(task)]
    path = directory / "accounting.jsonl"
    ledger = _rows(path)
    attempts, tokens = Counter(), Counter()
    for row in ledger:
        require(
            row.get("experiment_id") == "SR-F2-20261013"
            and row.get("task_id") == task_id(task)
            and row.get("replay_id") == plan.plan["replay_id"]
            and row.get("plan_sha256") == plan.sha256
            and type(row.get("count")) is int
            and row["count"] >= 0,
            "Replay ledger identity differs",
        )
        metadata = row.get("metadata", {})
        require(
            isinstance(metadata, dict) and type(metadata.get("group_seed")) is int,
            "Replay ledger seed metadata differs",
        )
        if row["kind"] == "completion_attempts":
            require(
                set(metadata) == {"group_seed", "qid"} and metadata["qid"] is None,
                "Replay attempt metadata differs",
            )
            attempts[(metadata.get("group_seed"), row["count"])] += 1
        elif row["kind"] == "generated_tokens":
            require(
                set(metadata) == {"group_seed", "row_index"}
                and type(metadata.get("row_index")) is int
                and 0 <= metadata["row_index"] < 4,
                "Replay token accounting metadata differs",
            )
            tokens[(metadata.get("group_seed"), metadata.get("row_index"), row["count"])] += 1
        else:
            raise PermissionError("Unexpected replay ledger compute kind")
    expected = Counter(
        (spec["group_seed"], index, len(row["tokens"])) for index, row in enumerate(rows)
    )
    require(
        attempts == Counter({(spec["group_seed"], 4): 1}) and tokens == expected,
        "Incomplete or repeated physical replay accounting",
    )
    return dict(
        path=str(path.relative_to(plan.directory)),
        sha256=file_hash(path),
        physical_attempt_groups=1,
        physical_attempt_rows=4,
        physical_generated_tokens=sum(len(row["tokens"]) for row in rows),
        canonical_new_rows=4 - spec["preserved_count"],
        duplicate_verification_rows=spec["preserved_count"],
        duplicate_verification_tokens=sum(
            len(row["tokens"]) for row in rows[: spec["preserved_count"]]
        ),
    )


def measured_generate(plan, task, generate, preparation_started):
    """Observe the unchanged frozen generation call; never infer missing end times."""
    directory = plan.task_directory(task)
    state = {}

    def measured(*args, **kwargs):
        started = time.perf_counter_ns()
        start = dict(
            status="GENERATION_STARTED",
            plan_sha256=plan.sha256,
            task=task,
            started_ns=time.time_ns(),
            preparation_wall_ns=started - preparation_started,
        )
        exclusive_intent(directory / "GENERATION_START.json", start)
        result = generate(*args, **kwargs)
        ended = time.perf_counter_ns()
        end = dict(
            status="GENERATION_RETURNED",
            plan_sha256=plan.sha256,
            task=task,
            start_sha256=object_hash(start),
            ended_ns=time.time_ns(),
            generation_call_wall_ns=ended - started,
        )
        durable(directory / "GENERATION_END.json", end)
        state.update(ended=ended, end=end)
        return result

    return measured, state


def verify_replay_timing(plan, task):
    directory = plan.task_directory(task)
    refs, values = [], {}
    for name in ("GENERATION_START", "GENERATION_END", "STAGING_TIMING"):
        path = directory / (name + ".json")
        if path.exists():
            value = read_json(path)
            require(
                value.get("plan_sha256") == plan.sha256 and value.get("task") == task,
                "Replay timing identity differs",
            )
            values[name] = value
            refs.append(dict(path=str(path.relative_to(plan.directory)), sha256=file_hash(path)))
    start, end, staging = (
        values.get(key) for key in ("GENERATION_START", "GENERATION_END", "STAGING_TIMING")
    )
    require(not end or start, "Generation end has no durable start")
    require(not staging or end, "Staging timing has no durable generation end")
    if start:
        require(
            start.get("status") == "GENERATION_STARTED"
            and type(start.get("preparation_wall_ns")) is int
            and start["preparation_wall_ns"] >= 0,
            "Replay preparation duration differs",
        )
    if end:
        require(
            end.get("status") == "GENERATION_RETURNED"
            and end.get("start_sha256") == object_hash(start)
            and type(end.get("generation_call_wall_ns")) is int
            and end["generation_call_wall_ns"] >= 0,
            "Replay generation duration differs",
        )
    if staging:
        require(
            staging.get("status") == "STAGED_VALIDATION_RETURNED"
            and staging.get("end_sha256") == object_hash(end)
            and type(staging.get("post_generation_validation_wall_ns")) is int
            and staging["post_generation_validation_wall_ns"] >= 0
            and staging.get("total_to_staged_validation_wall_ns")
            == start["preparation_wall_ns"]
            + end["generation_call_wall_ns"]
            + staging["post_generation_validation_wall_ns"],
            "Replay staging duration differs",
        )
    generation_seconds = end["generation_call_wall_ns"] / 1e9 if end else None
    return dict(
        status="COMPLETE" if staging else "PARTIAL" if values else "UNKNOWN",
        generation_call_wall_seconds=generation_seconds,
        single_gpu_allocated_generation_seconds=generation_seconds,
        preparation_wall_seconds=start["preparation_wall_ns"] / 1e9 if start else None,
        post_generation_staging_validation_wall_seconds=(
            staging["post_generation_validation_wall_ns"] / 1e9 if staging else None
        ),
        total_to_staged_validation_wall_seconds=(
            staging["total_to_staged_validation_wall_ns"] / 1e9 if staging else None
        ),
        measurement=(
            "Host elapsed wall time including input processing, persistence and timing I/O; "
            "not GPU kernel busy time"
        ),
        references=refs,
    )


def _prefix_rows(plan, task):
    spec, directory = plan.tasks[task_id(task)], plan.task_directory(task)
    if not spec["preserved_count"]:
        return [], None
    original = within(plan.root, spec["pending_path"], exists=False)
    archived = directory / "archive" / Path(spec["pending_path"]).name
    candidates = [path for path in (original, archived) if path.exists()]
    require(bool(candidates), "Original partial raw evidence is missing")
    for path in candidates:
        require(file_hash(path) == spec["pending_sha256"], "Original partial raw bytes changed")
    rows = _rows(candidates[0])
    require(len(rows) == spec["preserved_count"], "Original partial row count changed")
    return rows, archived


def compose_canonical(original_rows, replay_rows):
    require(len(replay_rows) == 4 and len(original_rows) in (0, 3), "Replay K changed")
    identity_fields = (
        "slot_id",
        "model_id",
        "step",
        "pool",
        "protocol",
        "view",
        "qid",
        "sampling_seed",
        "group_seed",
        "group_row_index",
        "group_size",
        "runtime_group_seed",
        "runtime_group_row_index",
        "runtime_group_size",
        "runtime_seed",
        "runtime_protocol",
        "source_sha256",
        "experiment_id",
        "input_hash",
        "image_hash",
        "prompt_tensor_hash",
        "model_identity",
        "adapter_identity",
        "sampling_parameters",
        "sampling_hash",
        "finish_reason",
        "truncated",
        "completion_token_count",
        "prompt_token_count",
    )
    for index, old in enumerate(original_rows):
        new = replay_rows[index]
        require(
            old.get("tokens") == new.get("tokens")
            and old.get("raw_tokens") == new.get("raw_tokens")
            and isinstance(old.get("tokens"), list)
            and bool(old["tokens"]),
            "Replay diverged from preserved original token sequence; no publication",
        )
        require(
            all(old.get(field) == new.get(field) for field in identity_fields),
            "Replay original-row model/input/sampling identity differs",
        )
    # Keep original dictionaries including original sampler probabilities/timings.
    return original_rows + replay_rows[len(original_rows) :]


def _verify_staged(plan, task, group, *, finalize_pending=False):
    from sr_f2.evaluation import validate_record

    directory, spec = plan.task_directory(task), plan.tasks[task_id(task)]
    intent = read_json(directory / "INTENT.json")
    require(
        intent.get("plan_sha256") == plan.sha256
        and intent.get("task") == task
        and intent.get("group_id") == spec["group_id"]
        and intent.get("physical_attempt_budget") == 1
        and intent.get("physical_rows") == 4
        and intent.get("slots_sha256") == object_hash(group["slots"]),
        "Replay attempt intent changed",
    )
    staged = directory / "staging/groups" / (spec["group_id"] + ".json")
    pending = staged.with_suffix(".pending.jsonl")
    recovering = not staged.exists()
    require(
        not recovering or (finalize_pending and pending.exists()),
        "Replay was interrupted before durable completion; no second attempt",
    )
    if recovering:
        retained = _rows(pending)
        require(len(retained) == 4, "Incomplete staged pending group; no second attempt")
        value = dict(
            group_id=spec["group_id"],
            group_index=group["group_index"],
            records=retained,
            records_sha256=object_hash(retained),
        )
    else:
        value = read_json(staged)
    rows = value["records"]
    require(
        value.get("group_id") == spec["group_id"]
        and len(rows) == 4
        and value.get("group_index") == group["group_index"]
        and value.get("records_sha256") == object_hash(rows),
        "Staged replay raw checksum differs",
    )
    for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
        validate_record(
            row,
            slot,
            input_hash=intent["input_hash"],
            model_identity=intent["model_identity"],
            adapter_identity=intent["adapter_identity"],
            source_sha256=read_json(plan.root / "REGISTRATION.json")["source_manifest_sha256"],
            index=index,
            count=4,
        )
    require(
        len({row["prompt_tensor_hash"] for row in rows}) == 1, "Replay group prompt tensors differ"
    )
    accounting = verify_replay_account(plan, task, rows)
    if pending.exists():
        require(_rows(pending) == rows, "Retained staging pending and committed rows differ")
    if recovering:
        # Only complete, validated raw rows AND token accounting can be published.
        # Preserve the pending bytes and add metadata; never invoke generation.
        durable(staged, value)
        durable(
            directory / "STAGING_RECEIPT_RECOVERY.json",
            dict(
                status="RECEIPT_ONLY",
                plan_sha256=plan.sha256,
                task=task,
                pending_path=str(pending.relative_to(plan.directory)),
                pending_sha256=file_hash(pending),
                staged_path=str(staged.relative_to(plan.directory)),
                staged_sha256=file_hash(staged),
            ),
        )
    return rows, accounting, staged, intent


def archive_partial(plan, task):
    spec, directory = plan.tasks[task_id(task)], plan.task_directory(task)
    rows, archived = _prefix_rows(plan, task)
    if not rows:
        return None
    original = within(plan.root, spec["pending_path"], exists=False)
    value = dict(
        status="ARCHIVED_ORIGINAL_RAW",
        plan_sha256=plan.sha256,
        task=task,
        original_path=spec["pending_path"],
        archived_path=str(archived.relative_to(plan.directory)),
        sha256=spec["pending_sha256"],
        rows=len(rows),
    )
    durable(directory / "ARCHIVE_INTENT.json", value)
    archived.parent.mkdir(parents=True, exist_ok=True)
    if not archived.exists():
        os.link(original, archived)  # Exclusive publication of the exact original bytes.
        _fsync_directory(archived.parent)
    require(file_hash(archived) == spec["pending_sha256"], "Archived original bytes differ")
    if original.exists():
        require(
            file_hash(original) == spec["pending_sha256"], "Original pending changed before archive"
        )
        original.unlink()  # Evidence retained at the verified immutable archive path.
        _fsync_directory(original.parent)
    return durable(directory / "ARCHIVE_COMPLETE.json", value)


def publish_replay(plan, task, group):
    """Receipt-only safe recovery: publish verified staging; never call generate here."""
    rows, accounting, staged, intent = _verify_staged(plan, task, group, finalize_pending=True)
    original, _ = _prefix_rows(plan, task)
    canonical_rows = compose_canonical(original, rows)
    value = dict(
        group_id=group["group_id"],
        group_index=group["group_index"],
        records=canonical_rows,
        records_sha256=object_hash(canonical_rows),
    )
    target = (
        plan.root
        / "sealed/evaluation"
        / task["model_id"]
        / "groups"
        / (group["group_id"] + ".json")
    )
    require(
        not target.exists() or read_json(target) == value,
        "Canonical target already exists with different bytes",
    )
    archived = archive_partial(plan, task)
    durable(target, value)
    directory, spec = plan.task_directory(task), plan.tasks[task_id(task)]
    provenance = dict(
        status="REPLAY_GROUP_PUBLISHED",
        plan_sha256=plan.sha256,
        task=task,
        group_id=group["group_id"],
        group_seed=spec["group_seed"],
        count=4,
        intent_sha256=object_hash(intent),
        staged_path=str(staged.relative_to(plan.directory)),
        staged_sha256=file_hash(staged),
        canonical_path=str(target.relative_to(plan.root)),
        canonical_sha256=file_hash(target),
        canonical_records_sha256=object_hash(canonical_rows),
        retained_original_row_hashes=[object_hash(row) for row in original],
        canonical_row_sources=["ORIGINAL"] * len(original) + ["REPLAY"] * (4 - len(original)),
        original_tokens_match=True,
        archive=archived,
        accounting=accounting,
        timing=verify_replay_timing(plan, task),
        original_ledger_prefix=dict(
            path=spec["original_ledger_path"],
            sha256=spec["original_ledger_sha256"],
            bytes=spec["original_ledger_bytes"],
        ),
    )
    return durable(directory / "REPLAY_COMPLETE.json", provenance)


def target_context(plan, task):
    from sr_f1.data import load_inputs, load_tasks
    from sr_f2.workers import endpoint_slots
    from sr_f12.evaluation import group_slots

    tasks, inputs = load_tasks(plan.root), load_inputs(plan.root)
    spec = plan.tasks[task_id(task)]
    groups = group_slots(endpoint_slots(tasks, task["model_id"]))
    found = [group for group in groups if group["group_id"] == spec["group_id"]]
    require(len(found) == 1, "Authorized interrupted group is not in frozen synthetic endpoint")
    group = found[0]
    require(
        len(group["slots"]) == 4
        and object_hash(group["slots"]) == spec["slots_sha256"]
        and group["slots"][0]["group_seed"] == spec["group_seed"],
        "Replay slots, seeds, or K differ from frozen endpoint",
    )
    return group, inputs, tasks


def replay_once(plan, task, group, inputs, tasks, *, runtime_factory=None):
    from sr_f2.diagnostics import prepare_input
    from sr_f2.evaluation import run_group
    from sr_f2.runner import load_real_runtime
    from sr_f2.workers import load_policy
    from sr_f12.evaluation import shard_lease

    directory = plan.task_directory(task)
    directory.mkdir(parents=True, exist_ok=True)
    with shard_lease(plan.root / "sealed/evaluation" / task["model_id"]), shard_lease(directory):
        if (directory / "INTENT.json").exists():
            return publish_replay(plan, task, group)  # Receipt-only; no second generation.
        require(
            not (directory / "accounting.jsonl").exists(), "Replay accounting exists without intent"
        )
        require(not (directory / "staging").exists(), "Replay staging exists without intent")
        preparation_started = time.perf_counter_ns()
        plan.baseline(task)
        runtime, *_ = (runtime_factory or load_real_runtime)(
            read_json(plan.root / "config/SR_F2.json"),
            plan.root,
            task_id="endpoint-" + task["model_id"],
        )
        runtime.account = replay_account(plan, task)
        load_policy(
            runtime, plan.root / "runs" / task["model_id"] / "checkpoints", 96, task["model_id"]
        )
        tokenizer = getattr(runtime, "tokenizer", None) or runtime.processor.tokenizer
        request = prepare_input(group["slots"][0], inputs, {}, tokenizer)
        image = within(plan.root, request["image_file"])
        input_hash = object_hash(dict(text=request["text"], image_sha256=file_hash(image)))
        exclusive_intent(
            directory / "INTENT.json",
            dict(
                task=task,
                group_id=group["group_id"],
                slots_sha256=object_hash(group["slots"]),
                plan_sha256=plan.sha256,
                model_identity=runtime.stable_model_identity(),
                adapter_identity=runtime.current_adapter_identity(),
                input_hash=input_hash,
                pid=os.getpid(),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                physical_attempt_budget=1,
                physical_rows=4,
                created_ns=time.time_ns(),
            ),
        )
        runtime.generate_group, timing_state = measured_generate(
            plan, task, runtime.generate_group, preparation_started
        )
        try:
            registration = read_json(plan.root / "REGISTRATION.json")
            with shard_lease(directory / "staging"):
                run_group(
                    runtime,
                    group,
                    inputs,
                    {},
                    tasks,
                    plan.root,
                    directory / "staging",
                    source_sha256=registration["source_manifest_sha256"],
                    score=False,
                )
            staged_at = time.perf_counter_ns()
            require(bool(timing_state), "Frozen replay did not perform its authorized generation")
            durable(
                directory / "STAGING_TIMING.json",
                dict(
                    status="STAGED_VALIDATION_RETURNED",
                    plan_sha256=plan.sha256,
                    task=task,
                    end_sha256=object_hash(timing_state["end"]),
                    post_generation_validation_wall_ns=staged_at - timing_state["ended"],
                    total_to_staged_validation_wall_ns=staged_at - preparation_started,
                ),
            )
            return publish_replay(plan, task, group)
        except BaseException as error:
            durable(
                directory / "FAILURE.json",
                dict(
                    status="FAILED_NO_RETRY",
                    plan_sha256=plan.sha256,
                    task=task,
                    error=repr(error),
                    at_ns=time.time_ns(),
                ),
            )
            raise
        finally:
            # The normal frozen endpoint opens a new runtime after this scope.
            # Clear callback/model cycles before it loads another model on the GPU.
            del runtime
            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


def verify_provenance(plan, task, group=None):
    group = group or target_context(plan, task)[0]
    rows, accounting, staged, intent = _verify_staged(plan, task, group)
    original, archived = _prefix_rows(plan, task)
    canonical = compose_canonical(original, rows)
    spec, directory = plan.tasks[task_id(task)], plan.task_directory(task)
    provenance = read_json(directory / "REPLAY_COMPLETE.json")
    require(
        provenance.get("status") == "REPLAY_GROUP_PUBLISHED"
        and provenance.get("plan_sha256") == plan.sha256
        and provenance.get("task") == task
        and provenance.get("group_id") == group["group_id"]
        and provenance.get("group_seed") == spec["group_seed"]
        and provenance.get("count") == 4
        and provenance.get("original_tokens_match") is True
        and provenance.get("intent_sha256") == object_hash(intent)
        and provenance.get("staged_sha256") == file_hash(staged)
        and provenance.get("retained_original_row_hashes") == [object_hash(row) for row in original]
        and provenance.get("canonical_records_sha256") == object_hash(canonical)
        and provenance.get("accounting") == accounting,
        "Replay provenance identity or accounting differs",
    )
    require(
        provenance.get("timing") == verify_replay_timing(plan, task),
        "Replay timing receipt changed",
    )
    target = within(plan.root, provenance["canonical_path"])
    expected_target = (
        plan.root / "sealed/evaluation" / task["model_id"] / "groups" / (spec["group_id"] + ".json")
    )
    require(
        target == expected_target
        and provenance.get("staged_path") == str(staged.relative_to(plan.directory))
        and provenance.get("canonical_row_sources")
        == ["ORIGINAL"] * len(original) + ["REPLAY"] * (4 - len(original))
        and file_hash(target) == provenance["canonical_sha256"]
        and read_json(target)
        == dict(
            group_id=group["group_id"],
            group_index=group["group_index"],
            records=canonical,
            records_sha256=object_hash(canonical),
        ),
        "Published replay group changed",
    )
    require(
        not target.with_suffix(".pending.jsonl").exists(), "Partial original group was not archived"
    )
    if archived is not None:
        archive = read_json(directory / "ARCHIVE_COMPLETE.json")
        require(
            archive == provenance["archive"]
            and archive["sha256"] == spec["pending_sha256"]
            and archived == within(plan.directory, archive["archived_path"]),
            "Original partial archive receipt differs",
        )
    ledger = within(plan.root, spec["original_ledger_path"])
    require(
        provenance.get("original_ledger_prefix")
        == dict(
            path=spec["original_ledger_path"],
            sha256=spec["original_ledger_sha256"],
            bytes=spec["original_ledger_bytes"],
        ),
        "Original ledger prefix receipt differs",
    )
    with ledger.open("rb") as stream:
        prefix = stream.read(spec["original_ledger_bytes"])
    require(
        hashlib.sha256(prefix).hexdigest() == spec["original_ledger_sha256"],
        "Original ledger prefix was modified",
    )
    return provenance


def verify_replay_receipt(plan, task):
    return verify_provenance(plan, task)


def verify_endpoint_task(plan, task):
    """Frozen complete-artifact checks plus explicit replay/archive provenance."""
    verify_provenance(plan, task)
    return verify_task(plan.root, task, plan.plan["registration_sha256"])


def run_endpoint(plan, task):
    from sr_f2.workers import endpoint

    plan.verify()
    group, inputs, tasks = target_context(plan, task)
    replay_once(plan, task, group, inputs, tasks)
    verify_provenance(plan, task, group)
    # All original complete groups and the repaired canonical group now use the
    # exact frozen endpoint path, while only previously unattempted groups sample.
    endpoint(plan.root, task["model_id"])
    return verify_endpoint_task(plan, task)
