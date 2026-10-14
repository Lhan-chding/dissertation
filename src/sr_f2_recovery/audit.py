"""Read-only recovery eligibility audit; never infer safety from missing pending files.

Generation reservations precede model.generate. Every reservation must therefore
have exactly one durable, complete raw group before an interrupted task can run
again. This module never scores records, repairs files, or imports a GPU runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            result.update(part)
    return result.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _json(text):
    def invalid_constant(_value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def read_json(path):
    return _json(Path(path).read_text())


def _relative(root, path):
    path = Path(path)
    target = path.resolve(strict=True)
    if not target.is_relative_to(root) or target != path.absolute() or not target.is_file():
        raise PermissionError("Audit input is not a regular, nonsymlink file under original root")
    return str(target.relative_to(root))


def _ref(root, path):
    return {"path": _relative(root, path), "sha256": file_hash(path)}


def _task_paths(root, task):
    kind, model_id = task["kind"], task["model_id"]
    prefixes = {
        "train": "runs",
        "pilot": "technical",
        "history": "history",
        "diagnostic": "diagnostics",
        "endpoint": "sealed/evaluation",
    }
    account_name = model_id if kind in {"train", "pilot"} else kind + "-" + model_id
    return root / prefixes[kind] / model_id, root / "accounting" / (account_name + ".jsonl")


def _lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank accounting or raw JSONL line")
            row = _json(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL row is not an object")
            yield row


def _raw_key(rows, *, expected_count=None):
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Empty or malformed raw group")
    count = rows[0].get("group_size")
    seed = rows[0].get("group_seed")
    if type(count) is not int or count not in {1, 2, 4, 8} or type(seed) is not int:
        raise ValueError("Malformed raw group seed or size")
    if len(rows) != count or (expected_count is not None and count != expected_count):
        raise ValueError("Incomplete durable group")
    for index, row in enumerate(rows):
        tokens, logps = row.get("tokens"), row.get("sampler_logprobs")
        if (
            row.get("group_seed") != seed
            or row.get("group_size") != count
            or row.get("group_row_index") != index
            or type(row.get("seed")) is not int
            or row.get("generation_status") not in (None, "COMPLETE")
            or row.get("technical_validation_errors")
            or not isinstance(tokens, list)
            or not tokens
            or not all(type(token) is int for token in tokens)
            or not isinstance(logps, list)
            or len(logps) != len(tokens)
            or not all(type(p) in (int, float) and math.isfinite(p) for p in logps)
        ):
            raise ValueError("Raw row identity, completeness, or sampler data differs")
    return seed, count


def _pending_matches(rows, pending_rows, training):
    if not training:
        return rows == pending_rows
    # Frozen training adds scorer/input fields before committing but retains raw pending rows.
    excluded = {"score", "row", "root", "qid"}
    return len(rows) == len(pending_rows) and all(
        {k: v for k, v in pending.items() if k not in excluded}
        == {k: row.get(k) for k in pending if k not in excluded}
        and row.get("qid") == pending.get("qid")
        for row, pending in zip(rows, pending_rows, strict=True)
    )


def generation_audit(root, task, *, source_sha256=None, manifest_sha256=None):
    """Return counts/gaps and file references, never raw answers or scientific scores.

    Counter multiplicity is required: many distinct greedy ChartQA groups reserve
    (group_seed=0, count=1). A set would silently hide lost generations.
    """
    root = Path(root).resolve(strict=True)
    directory, account = _task_paths(root, task)
    kind, model_id = task["kind"], task["model_id"]
    training = kind in {"train", "pilot"}
    locations = [(directory / "rollouts", True)] if training else []
    if training:
        locations.extend(
            (path / "groups", False) for path in sorted((directory / "monitor").glob("step-*"))
        )
    elif kind == "history":
        locations.extend((path / "groups", False) for path in sorted(directory.glob("step-*")))
    else:
        locations.append((directory / "groups", False))
    attempts, durable = Counter(), Counter()
    token_ledger, raw_tokens = Counter(), Counter()
    refs, mutable, absent, gaps, seen_slots = [], [], [], [], set()
    counts = dict(
        accounting_rows=0,
        attempt_groups=0,
        attempt_records=0,
        final_groups=0,
        full_pending_groups=0,
        retained_pending_groups=0,
        durable_records=0,
        token_ledger_records=0,
        token_ledger_tokens=0,
        durable_tokens=0,
    )

    def gap(code, path=None, **fields):
        item = dict(code=code, **fields)
        if path is not None:
            item["path"] = str(Path(path).relative_to(root))
        gaps.append(item)

    if account.exists():
        mutable.append(_ref(root, account))
        seen_ledger = set()
        try:
            for row in _lines(account):
                counts["accounting_rows"] += 1
                row_hash = object_hash(row)
                if row_hash in seen_ledger:
                    raise ValueError("Duplicate accounting row")
                seen_ledger.add(row_hash)
                if (
                    set(row)
                    != {
                        "experiment_id",
                        "task_id",
                        "kind",
                        "count",
                        "metadata",
                        "pid",
                        "time_ns",
                        "slurm_job_id",
                    }
                    or type(row.get("pid")) is not int
                    or row["pid"] <= 0
                    or type(row.get("time_ns")) is not int
                    or row["time_ns"] <= 0
                    or not isinstance(row.get("slurm_job_id"), str)
                    or not row["slurm_job_id"].isdigit()
                    or not isinstance(row.get("kind"), str)
                    or not row["kind"]
                    or row.get("experiment_id") != "SR-F2-20261013"
                    or row.get("task_id") != account.stem
                    or type(row.get("count")) is not int
                    or row["count"] < 0
                    or not isinstance(row.get("metadata"), dict)
                ):
                    raise ValueError("Accounting identity or count differs")
                if row.get("kind") == "completion_attempts":
                    seed, count = row["metadata"].get("group_seed"), row["count"]
                    if type(seed) is not int or count not in {1, 2, 4, 8}:
                        raise ValueError("Malformed completion reservation")
                    attempts[(seed, count)] += 1
                    counts["attempt_groups"] += 1
                    counts["attempt_records"] += count
                elif row["kind"] == "generated_tokens":
                    seed = row["metadata"].get("group_seed")
                    index = row["metadata"].get("row_index")
                    token_count = row["count"]
                    if (
                        type(seed) is not int
                        or type(index) is not int
                        or not 0 <= index < 8
                        or token_count <= 0
                        or set(row["metadata"]) != {"group_seed", "row_index"}
                    ):
                        raise ValueError("Malformed generated-token reservation")
                    token_ledger[(seed, index, token_count)] += 1
                    counts["token_ledger_records"] += 1
                    counts["token_ledger_tokens"] += token_count
        except (ValueError, KeyError, TypeError) as error:
            gap("MALFORMED_ACCOUNTING", account, detail=str(error))
    else:
        absent.append(str(account.relative_to(root)))

    for location, is_training in locations:
        if not location.exists():
            continue
        finals, pendings, requests = {}, {}, {}
        for path in sorted(location.iterdir()):
            if path.name.endswith(".failure.json"):
                refs.append(_ref(root, path))
                gap("PRESERVED_GENERATION_FAILURE", path)
            elif path.name.endswith(".request.json") and is_training:
                requests[path.name.removesuffix(".request.json")] = path
                refs.append(_ref(root, path))
            elif path.name.endswith(".pending.jsonl"):
                pendings[path.name.removesuffix(".pending.jsonl")] = path
            elif path.suffix == ".json":
                finals[path.stem] = path
            else:
                gap("UNEXPECTED_RAW_ARTIFACT", path)

        def validate_rows(rows, stem, identity=None, *, is_training=is_training):
            seed, count = _raw_key(rows, expected_count=8 if is_training else None)
            if is_training:
                if not re.fullmatch(r"[0-9]{2}-[0-9]{2}", stem) or not isinstance(identity, dict):
                    raise ValueError("Training request identity missing")
                if (
                    identity.get("run_id") != model_id
                    or identity.get("plan_id") != "SR-F2"
                    or identity.get("scoring") != "SR-F2-v2"
                    or identity.get("seeds") != [row["seed"] for row in rows]
                    or identity["seeds"][0] != seed
                    or stem != f"{identity['step']:02d}-{identity['slot']:02d}"
                    or any(row.get("qid") != identity.get("qid") for row in rows)
                    or (source_sha256 and identity.get("source_sha256") != source_sha256)
                    or (manifest_sha256 and identity.get("run_identity_sha256") != manifest_sha256)
                ):
                    raise ValueError("Training raw/request binding differs")
            else:
                if not re.fullmatch(r"[0-9a-f]{64}", stem):
                    raise ValueError("Malformed evaluation group identifier")
                fields = ("model_id", "step", "pool", "protocol", "view", "qid")
                group_key = [rows[0].get(field) for field in fields]
                if stem != object_hash(group_key):
                    raise ValueError("Evaluation group identity hash differs")
                for index, row in enumerate(rows):
                    slot = row.get("slot_id")
                    if (
                        not isinstance(slot, str)
                        or slot in seen_slots
                        or row.get("model_id") != model_id
                        or [row.get(field) for field in fields] != group_key
                        or row.get("draw") != index
                        or row.get("samples_per_prompt") != count
                        or row.get("sampling_seed") != row.get("seed")
                        or (source_sha256 and row.get("source_sha256") != source_sha256)
                    ):
                        raise ValueError(
                            "Evaluation raw slot/source identity differs or duplicates"
                        )
                    seen_slots.add(slot)
            return seed, count

        for stem, path in finals.items():
            try:
                value = read_json(path)
                rows = value["records"]
                expected_hash = value.get("sha256" if is_training else "records_sha256")
                if expected_hash != object_hash(rows):
                    raise ValueError("Committed raw group checksum differs")
                if not is_training and value.get("group_id") != stem:
                    raise ValueError("Committed raw group filename differs")
                identity = value.get("identity")
                if is_training and stem in requests and read_json(requests[stem]) != identity:
                    raise ValueError("Training request and committed identity differ")
                key = validate_rows(rows, stem, identity)
                refs.append(_ref(root, path))
                counts["final_groups"] += 1
                counts["durable_records"] += key[1]
                durable[key] += 1
                raw_tokens.update(
                    (key[0], index, len(row["tokens"])) for index, row in enumerate(rows)
                )
                counts["durable_tokens"] += sum(len(row["tokens"]) for row in rows)
                if stem in pendings:
                    pending = pendings.pop(stem)
                    pending_rows = list(_lines(pending))
                    if _raw_key(pending_rows) != key or not _pending_matches(
                        rows, pending_rows, is_training
                    ):
                        raise ValueError("Committed and retained pending rows differ")
                    # Evaluation removes this twin; training leaves it unchanged.
                    (refs if is_training else mutable).append(_ref(root, pending))
                    counts["retained_pending_groups"] += 1
            except (ValueError, KeyError, TypeError) as error:
                gap("INVALID_COMMITTED_GROUP", path, detail=str(error))

        for stem, path in pendings.items():
            mutable.append(_ref(root, path))
            absent.append(str((location / (stem + ".json")).relative_to(root)))
            try:
                rows = list(_lines(path))
                identity = read_json(requests[stem]) if is_training and stem in requests else None
                key = validate_rows(rows, stem, identity)
                counts["full_pending_groups"] += 1
                counts["durable_records"] += key[1]
                durable[key] += 1
                raw_tokens.update(
                    (key[0], index, len(row["tokens"])) for index, row in enumerate(rows)
                )
                counts["durable_tokens"] += sum(len(row["tokens"]) for row in rows)
            except (ValueError, KeyError, TypeError) as error:
                gap("INCOMPLETE_OR_INVALID_PENDING", path, detail=str(error))

        for stem, request in requests.items():
            if stem not in finals and stem not in pendings:
                # A request alone precedes generation; the ledger remains decisive.
                value = read_json(request)
                if (
                    value.get("run_id") != model_id
                    or value.get("plan_id") != "SR-F2"
                    or (source_sha256 and value.get("source_sha256") != source_sha256)
                    or (manifest_sha256 and value.get("run_identity_sha256") != manifest_sha256)
                ):
                    gap("REQUEST_IDENTITY_DIFFERS", request)
                absent.extend(
                    str((location / (stem + suffix)).relative_to(root))
                    for suffix in (".json", ".pending.jsonl", ".failure.json")
                )

    for (seed, count), multiplicity in sorted((attempts - durable).items()):
        gap("ATTEMPT_WITHOUT_DURABLE_GROUP", group_seed=seed, count=count, groups=multiplicity)
    for (seed, count), multiplicity in sorted((durable - attempts).items()):
        gap("RAW_GROUP_WITHOUT_ATTEMPT", group_seed=seed, count=count, groups=multiplicity)
    # Runtime fsyncs each raw row before reserving its generated_tokens. A kill
    # between those operations leaves reusable bytes but incomplete accounting;
    # the frozen worker cannot silently reconstruct that missing ledger entry.
    for (seed, index, count), multiplicity in sorted((raw_tokens - token_ledger).items()):
        gap(
            "RAW_ROW_WITHOUT_TOKEN_ACCOUNTING",
            group_seed=seed,
            row_index=index,
            token_count=count,
            records=multiplicity,
        )
    for (seed, index, count), multiplicity in sorted((token_ledger - raw_tokens).items()):
        gap(
            "TOKEN_ACCOUNTING_WITHOUT_RAW_ROW",
            group_seed=seed,
            row_index=index,
            token_count=count,
            records=multiplicity,
        )
    return dict(
        status="PASS" if not gaps else "BLOCKED_INCOMPLETE_GENERATION",
        counts=counts,
        gaps=gaps,
        reference_files=refs,
        mutable_prelaunch_files=mutable,
        absent_prelaunch_paths=sorted(set(absent)),
    )


def _claim_evidence(root, task, registration_sha256):
    from sr_f2.orchestration import task_id

    key = task_id(task)
    directory = root / "orchestration/queue/tasks" / key
    claim_path, state_path = directory / "CLAIM.json", directory / "STATE.json"
    claim, state = read_json(claim_path), read_json(state_path)
    fields = ("task", "pool_id", "device", "job_id", "registration_sha256", "claim_id")
    if (
        claim.get("task") != task
        or claim.get("registration_sha256") != registration_sha256
        or any(state.get(field) != claim.get(field) for field in fields)
        or state.get("claim_sha256") != object_hash(claim)
        or state.get("status") not in {"CLAIMED", "RUNNING", "FAILED", "COMPLETE"}
        or not re.fullmatch(r"GPU_POOL_[1-4]", str(claim.get("pool_id")))
        or not str(claim.get("job_id", "")).isdigit()
    ):
        raise PermissionError("Original claim/state binding differs")
    worker = root / "orchestration/pools" / claim["pool_id"] / "workers" / key
    matches = list((root / "orchestration/pools").glob("*/workers/" + key))
    if matches != [worker]:
        raise PermissionError("Original worker ownership missing or duplicated")
    launch_path = worker / "LAUNCH_INTENT.json"
    launch = read_json(launch_path)
    command = launch.get("command", [])
    if (
        launch.get("task") != task
        or launch.get("claim_id") != claim.get("claim_id")
        or launch.get("device") != claim.get("device")
        or not isinstance(command, list)
        or len(command) != 8
        or command[2:]
        != ["--root", str(root), "--kind", task["kind"], "--model-id", task["model_id"]]
    ):
        raise PermissionError("Original worker launch differs from claim")
    refs = [_ref(root, p) for p in (claim_path, state_path, launch_path)]
    exit_path = worker / "PROCESS_EXIT.json"
    exit_receipt = read_json(exit_path) if exit_path.exists() else None
    if exit_path.exists():
        refs.append(_ref(root, exit_path))
    return state, exit_receipt, refs, _ref(root, claim_path)


def _completed_evidence(root, task, state, exit_receipt, registration_hash):
    directory, _ = _task_paths(root, task)
    receipt_path = root / "tasks" / task["kind"] / task["model_id"] / "COMPLETE.json"
    receipt = read_json(receipt_path)
    expected_status = "PASS" if task["kind"] == "pilot" else "COMPLETE"
    if (
        receipt.get("status") != expected_status
        or any(receipt.get(k) != v for k, v in task.items())
        or receipt.get("registration_sha256") != registration_hash
        or exit_receipt is None
        or exit_receipt.get("returncode") != 0
        or exit_receipt.get("pool_aborted") is True
        or state.get("result") != dict(**task, receipt_sha256=object_hash(receipt))
    ):
        raise PermissionError("Completed task receipt/queue/process identity differs")
    expected = {str((directory / "COMPLETE.json").relative_to(root))}
    if task["kind"] in {"train", "pilot"}:
        step = 96 if task["kind"] == "train" else 8
        expected.add(str((directory / "checkpoints" / f"commit-{step:02d}.json").relative_to(root)))
        if task["kind"] == "train":
            expected.add(str((directory / "monitor_state/step-96.json").relative_to(root)))
    refs = [_ref(root, receipt_path)]
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(expected):
        raise PermissionError("Completed task artifact inventory differs")
    if {item.get("path") for item in artifacts} != expected:
        raise PermissionError("Completed task artifact names differ")
    for item in artifacts:
        actual = _ref(root, root / item["path"])
        if actual != item:
            raise PermissionError("Previously verified task metadata bytes changed")
        refs.append(actual)
    done = read_json(directory / "COMPLETE.json")
    if done.get("status") != "COMPLETE" or done.get("model_id") != task["model_id"]:
        raise PermissionError("Completed source receipt identity differs")
    if task["kind"] == "endpoint" and done.get("scores_sealed") is not True:
        raise PermissionError("Completed endpoint seal differs")
    return refs


def _checkpoint_evidence(root, task, registration):
    directory, _ = _task_paths(root, task)
    manifest_path = directory / "RUN_MANIFEST.json"
    manifest = read_json(manifest_path)
    registered = next(
        row for row in registration["main_matrix"] if row["model_id"] == task["model_id"]
    )
    if (
        any(manifest.get(k) != v for k, v in registered.items())
        or manifest.get("config_sha256") != registration["config_sha256"]
        or manifest.get("source_manifest_sha256") != registration["source_manifest_sha256"]
        or manifest.get("technical_only") is not False
    ):
        raise PermissionError("Original training manifest differs from registration")
    checkpoints = directory / "checkpoints"
    markers = sorted(checkpoints.glob("commit-[0-9][0-9].json"))
    if not markers:
        raise PermissionError("Interrupted training has no committed checkpoint")
    marker = markers[-1]
    commit = read_json(marker)
    step = commit.get("step")
    if (
        type(step) is not int
        or not 0 <= step <= 96
        or step % 8
        or marker.name != f"commit-{step:02d}.json"
        or not isinstance(commit.get("field_hashes"), dict)
        or not {
            "parameters",
            "optimizer",
            "scheduler",
            "rng",
            "run_identity",
            "cursor",
            "adaptive_state",
        }.issubset(commit.get("field_hashes", {}))
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in commit.get("field_hashes", {}).values()
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(commit.get("state_hash")))
    ):
        raise PermissionError("Highest checkpoint commit identity differs")
    relative = Path(commit["path"])
    if relative.is_absolute() or len(relative.parts) != 1:
        raise PermissionError("Checkpoint payload path escapes its directory")
    payload = checkpoints / relative
    payload_ref = _ref(root, payload)
    if payload_ref["sha256"] != commit.get("sha256"):
        raise PermissionError("Committed checkpoint payload checksum differs")
    refs = [_ref(root, manifest_path), _ref(root, marker)]
    latest_path = checkpoints / "LATEST.json"
    # Payload is immutable, but rehash it only for this task's own first launch.
    mutable, absent = [payload_ref], []
    if latest_path.exists():
        latest = read_json(latest_path)
        if (
            type(latest.get("step")) is not int
            or latest["step"] > step
            or (latest["step"] == step and latest != commit)
        ):
            raise PermissionError("LATEST conflicts with highest immutable commit")
        mutable.append(_ref(root, latest_path))
    else:
        absent.append(str(latest_path.relative_to(root)))
    return (
        dict(step=step, payload_sha256=payload_ref["sha256"], gpu_restore_validation_required=True),
        object_hash(manifest),
        refs,
        mutable,
        absent,
    )


def audit_recovery(root):
    """Audit all frozen tasks without mutating source, queue, registration, or outputs."""
    from sr_f2.orchestration import global_task_matrix, task_id
    from sr_f2.protocol import verify_registration

    root = Path(root).resolve(strict=True)
    registration = verify_registration(root, require_technical=True)
    registration_hash = object_hash(registration)
    matrix = global_task_matrix()
    known = {task_id(task) for task in matrix}
    queue_root = root / "orchestration/queue/tasks"
    if any(path.name not in known for path in queue_root.iterdir()):
        raise PermissionError("Unknown original queue task")
    queue_config = read_json(root / "orchestration/queue/CONFIG.json")
    if (
        queue_config.get("tasks") != matrix
        or queue_config.get("registration_sha256") != registration_hash
        or queue_config.get("task_matrix_sha256") != object_hash(matrix)
    ):
        raise PermissionError("Original shared queue registration differs")
    tasks = {}
    for task in matrix:
        key = task_id(task)
        entry = dict(
            task=task,
            status="BLOCKED_INCOMPLETE_GENERATION",
            reference_files=[],
            mutable_prelaunch_files=[],
            absent_prelaunch_paths=[],
        )
        tasks[key] = entry
        directory, accounting = _task_paths(root, task)
        queue_dir = queue_root / key
        receipt_dir = root / "tasks" / task["kind"] / task["model_id"]
        worker_dirs = list((root / "orchestration/pools").glob("*/workers/" + key))
        try:
            if not queue_dir.exists():
                if directory.exists() or accounting.exists() or receipt_dir.exists() or worker_dirs:
                    raise PermissionError(
                        "Unclaimed task already has source, worker, receipt, "
                        "or accounting artifacts"
                    )
                entry.update(
                    status="NEVER_ATTEMPTED",
                    reason="No original claim or task artifacts exist",
                    absent_prelaunch_paths=[
                        str(p.relative_to(root))
                        for p in (queue_dir, directory, accounting, receipt_dir)
                    ],
                )
                continue
            state, exit_receipt, refs, claim_ref = _claim_evidence(root, task, registration_hash)
            entry["reference_files"].extend(refs)
            entry["original_claim"] = claim_ref
            if state["status"] == "COMPLETE":
                entry["reference_files"].extend(
                    _completed_evidence(root, task, state, exit_receipt, registration_hash)
                )
                entry.update(
                    status="INHERITED_COMPLETE",
                    reason=(
                        "Original verified completion, claim and zero process exit agree; "
                        "metadata hashes rechecked"
                    ),
                )
                continue
            if receipt_dir.exists() and any(receipt_dir.iterdir()):
                raise PermissionError(
                    "Unharvested completion requires separate receipt-only recovery"
                )
            if task["kind"] not in {"train", "endpoint", "diagnostic"}:
                raise PermissionError("Interrupted historical/pilot task is outside this recovery")
            incidents = list(
                (root / "technical_incidents").glob(
                    task["kind"] + "-" + task["model_id"] + "-*.json"
                )
            )
            if incidents:
                entry["reference_files"].extend(_ref(root, path) for path in incidents)
                raise PermissionError(
                    "A preserved task technical incident requires explicit review"
                )
            manifest_hash = None
            if task["kind"] == "train":
                checkpoint, manifest_hash, refs, mutable, absent = _checkpoint_evidence(
                    root, task, registration
                )
                entry["checkpoint"] = checkpoint
                entry["reference_files"].extend(refs)
                entry["mutable_prelaunch_files"].extend(mutable)
                entry["absent_prelaunch_paths"].extend(absent)
            else:
                plan_path = directory / "PLAN.json"
                plan = read_json(plan_path)
                if plan.get("source_sha256") != registration["source_manifest_sha256"] or (
                    task["kind"] == "endpoint" and plan.get("scores_sealed") is not True
                ):
                    raise PermissionError("Evaluation plan source/seal differs")
                entry["reference_files"].append(_ref(root, plan_path))
            if task["kind"] == "endpoint" and any((directory / "scores").glob("*.json")):
                raise PermissionError("Scientific endpoint score files exist before release")
            generated = generation_audit(
                root,
                task,
                source_sha256=registration["source_manifest_sha256"],
                manifest_sha256=manifest_hash,
            )
            entry["generation_audit"] = {
                k: v for k, v in generated.items() if k in {"status", "counts", "gaps"}
            }
            # All raw hashes are retained, but checked only for this task's first
            # launch. Global checks retain small immutable lineage metadata only.
            entry["mutable_prelaunch_files"].extend(generated["reference_files"])
            entry["mutable_prelaunch_files"].extend(generated["mutable_prelaunch_files"])
            entry["absent_prelaunch_paths"].extend(generated["absent_prelaunch_paths"])
            if generated["status"] == "PASS":
                entry.update(
                    status="RESUME_ALLOWED",
                    reason=(
                        "All reserved generation groups have complete durable raw coverage; "
                        "frozen worker must validate restore before computing"
                    ),
                )
            else:
                entry["reason"] = (
                    "Generation coverage is incomplete or inconsistent; "
                    "no retry/resampling authorized"
                )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            entry["reason"] = f"Fail-closed audit: {type(error).__name__}: {error}"
    return dict(
        schema="sr-f2-recovery-audit-v1",
        original_root=str(root),
        created_at=datetime.now(timezone.utc).isoformat(),
        registration_sha256=registration_hash,
        task_matrix_sha256=object_hash(matrix),
        tasks=tasks,
        counts=dict(Counter(entry["status"] for entry in tasks.values())),
        scientific_scores_read=False,
        original_artifacts_modified=False,
        inherited_completions_scope=(
            "Previously fully verified receipts plus current "
            "metadata hashes; not a repeated full raw-data scan"
        ),
    )
