"""SR-F1.2 same-question evaluation with immutable, recoverable group ledgers."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from sr_f1.evaluation import prepare_input, read_jsonl, score_response, sha_file
from sr_f1.json_protocol import decoded_protocol

from .protocol import AMENDMENT_ID, TEST_POOLS, object_hash


def durable_json(path, value):
    """Publish exactly once; a completed durable group is never silently replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise PermissionError(f"Immutable evaluation artifact differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication cannot overwrite a concurrent publisher.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


@contextmanager
def shard_lease(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Duplicate worker for evaluation shard") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def group_slots(slots):
    """Require whole same-question groups, preserving the registered panel order."""
    groups, keys = [], set()
    for slot in slots:
        key = tuple(slot[k] for k in ("model_id", "step", "pool", "protocol", "view", "qid"))
        if not groups or key != groups[-1][0]:
            if key in keys:
                raise ValueError("Non-contiguous evaluation group")
            keys.add(key)
            groups.append((key, []))
        groups[-1][1].append(slot)
    output = []
    for index, (key, rows) in enumerate(groups):
        k = rows[0]["samples_per_prompt"]
        if k not in (1, 2, 4, 8) or [r["draw"] for r in rows] != list(range(k)):
            raise ValueError("Incomplete or reordered evaluation group")
        if any(
            r["samples_per_prompt"] != k or r["generation"] != rows[0]["generation"] for r in rows
        ):
            raise ValueError("Group recipe differs")
        output.append(dict(group_index=index, group_id=object_hash(key), slots=rows))
    if len({r["slot_id"] for g in output for r in g["slots"]}) != sum(
        len(g["slots"]) for g in output
    ):
        raise ValueError("Duplicate evaluation slot")
    return output


def _identity(runtime, name):
    item = getattr(runtime, name, None)
    return item() if callable(item) else item


def request_identity(slot, inputs, diagnostics, root, tokenizer):
    request = prepare_input(slot, inputs, diagnostics, tokenizer)
    image = (root / request["image_file"]).resolve(strict=True) if request["image_file"] else None
    if image is not None and not image.is_relative_to(root):
        raise PermissionError("Evaluation image escapes run root")
    image_hash = sha_file(image) if image is not None else None
    return (
        request,
        image,
        object_hash(dict(text=request["text"], image_sha256=image_hash)),
        image_hash,
    )


def validate_record(
    row, slot, *, input_hash, model_identity, adapter_identity, source_sha256, index, count
):
    if any(row.get(k) != v for k, v in slot.items()):
        raise PermissionError("Persisted evaluation slot identity differs")
    expected = dict(
        input_hash=input_hash,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        source_sha256=source_sha256,
        group_seed=slot["group_seed"],
        group_row_index=index,
        group_size=count,
        runtime_group_seed=slot["group_seed"],
        runtime_group_row_index=index,
        runtime_group_size=count,
        runtime_protocol=slot["protocol"],
        runtime_seed=slot["sampling_seed"],
        amendment_id=AMENDMENT_ID,
    )
    if any(row.get(k) != v for k, v in expected.items()):
        raise PermissionError("Persisted input/model/adapter/source/group identity differs")
    tokens, probabilities = row.get("tokens"), row.get("sampler_logprobs")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(t) is not int or t < 0 for t in tokens)
        or row.get("raw_tokens") != tokens
        or row.get("completion_token_count") != len(tokens)
        or not isinstance(probabilities, list)
        or len(probabilities) != len(tokens)
        or any(
            isinstance(p, bool)
            or not isinstance(p, (float, int))
            or not math.isfinite(p)
            or p > 1e-5
            for p in probabilities
        )
    ):
        raise ValueError("Incomplete generated token/probability trajectory")
    if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
        raise RuntimeError("Durable generation records a technical failure")
    if not row.get("prompt_tensor_hash"):
        raise ValueError("Missing actual prompt tensor identity")
    if slot["protocol"] in ("evidence_answer", "answer_only"):
        decoded = decoded_protocol(row["generated_text"])
        if any(row.get(k) != v for k, v in decoded.items()):
            raise ValueError("Prefill and decoded output identity differ")
        balanced = decoded["balanced_cut_char"] is not None
        if balanced:
            if (
                row.get("balanced_token_count") != len(tokens)
                or row.get("finish_reason") != "balanced"
                or row.get("truncated") is not False
            ):
                raise ValueError("Balanced stop trajectory differs")
        elif row.get("balanced_token_count") is not None or row.get("finish_reason") not in (
            "eos",
            "length",
        ):
            raise ValueError("Unbalanced stop trajectory differs")
    if len(tokens) > slot["generation"]["max_new_tokens"]:
        raise ValueError("Generated response exceeds registered cap")
    if row.get("truncated") is not (row.get("finish_reason") == "length"):
        raise ValueError("Truncation metadata differs")


def score_record(row, task):
    """Use the unchanged scientific scorer without legacy old_logprobs validation."""
    text = "" if row.get("format_protocol_error") else row["raw_text"]
    result = score_response(text, task, row["protocol"])
    if row.get("format_protocol_error"):
        result["reason"] = row["format_protocol_error"]
    return result


def evaluate_shard(
    runtime,
    slots,
    inputs,
    diagnostics,
    tasks,
    root,
    directory,
    *,
    shard_index,
    shard_count,
    source_sha256,
    score_non_test=True,
    boundary=None,
):
    """Each group is durably recorded before validation/scoring; resume never resamples it."""
    if (
        type(shard_count) is not int
        or not 1 <= shard_count <= 5
        or type(shard_index) is not int
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("Invalid registered shard allocation")
    root, directory = Path(root).resolve(strict=True), Path(directory).resolve()
    if not directory.is_relative_to(root):
        raise PermissionError("Evaluation output escapes run root")
    groups = group_slots(slots)
    assigned = [g for g in groups if g["group_index"] % shard_count == shard_index]
    model, adapter = (
        _identity(runtime, "stable_model_identity"),
        _identity(runtime, "current_adapter_identity"),
    )
    if not isinstance(model, dict) or not isinstance(adapter, dict):
        raise ValueError("Model and adapter identity required")
    tokenizer = getattr(runtime, "tokenizer", None) or runtime.processor.tokenizer
    plan = dict(
        amendment_id=AMENDMENT_ID,
        source_sha256=source_sha256,
        model_identity=model,
        adapter_identity=adapter,
        shard_index=shard_index,
        shard_count=shard_count,
        total_groups=len(groups),
        slots_sha256=object_hash([g["slots"] for g in groups]),
        assigned_groups=[g["group_id"] for g in assigned],
        expected_records=sum(len(g["slots"]) for g in assigned),
    )
    with shard_lease(directory):
        durable_json(directory / "PLAN.json", plan)
        allowed = {g["group_id"] for g in assigned}
        for path in (directory / "groups").glob("*") if (directory / "groups").exists() else []:
            if path.name.removesuffix(".pending.jsonl").removesuffix(".json") not in allowed:
                raise PermissionError("Unregistered raw evaluation group")
        completed, receipts = 0, []
        for group in assigned:
            if (root / "STOP").exists() or (boundary and boundary.get("requested")):
                return dict(
                    status="CHECKPOINTED",
                    generated_records=completed,
                    plan_sha256=object_hash(plan),
                )
            first, count = group["slots"][0], len(group["slots"])
            request, image, input_hash, image_hash = request_identity(
                first, inputs, diagnostics, root, tokenizer
            )
            path = directory / "groups" / (group["group_id"] + ".json")
            pending = path.with_suffix(".pending.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                value = json.loads(path.read_text())
                rows = value["records"]
                if (
                    value.get("records_sha256") != object_hash(rows)
                    or value.get("group_id") != group["group_id"]
                ):
                    raise PermissionError("Raw group checksum differs")
            elif pending.exists():
                rows = list(read_jsonl(pending))
                if len(rows) != count:
                    raise RuntimeError(
                        "Incomplete durable group; preserve it for technical recovery, "
                        "do not resample"
                    )
            else:
                rows = []

                def persist(
                    completion,
                    rows=rows,
                    count=count,
                    group=group,
                    request=request,
                    input_hash=input_hash,
                    image_hash=image_hash,
                    pending=pending,
                ):
                    index = len(rows)
                    if index >= count:
                        raise RuntimeError("Runtime emitted excess group records")
                    row = dict(completion)
                    for key in ("group_seed", "group_row_index", "group_size", "protocol", "seed"):
                        row["runtime_" + key] = completion.get(key)
                    row.update(group["slots"][index])
                    row.update(request)
                    row.update(
                        input_hash=input_hash,
                        image_hash=image_hash,
                        source_sha256=source_sha256,
                        amendment_id=AMENDMENT_ID,
                        group_row_index=index,
                        group_size=count,
                    )
                    # The actual runtime identities are retained, not overwritten by the plan.
                    with pending.open("a") as stream:
                        stream.write(
                            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    rows.append(row)

                generation = dict(first["generation"], num_return_sequences=count)
                runtime.generate_group(
                    None,
                    root,
                    [s["sampling_seed"] for s in group["slots"]],
                    text=request["text"],
                    image_path=image,
                    protocol=first["protocol"],
                    generation=generation,
                    on_completion=persist,
                )
                if len(rows) != count:
                    raise RuntimeError("Runtime did not durably deliver the complete group")
            # Raw group publication precedes any scientific scoring and survives validation failure.
            value = dict(
                group_id=group["group_id"],
                group_index=group["group_index"],
                records=rows,
                records_sha256=object_hash(rows),
            )
            durable_json(path, value)
            if len(rows) != count:
                raise ValueError("Persisted group coverage differs")
            for index, (row, slot) in enumerate(zip(rows, group["slots"], strict=True)):
                validate_record(
                    row,
                    slot,
                    input_hash=input_hash,
                    model_identity=model,
                    adapter_identity=adapter,
                    source_sha256=source_sha256,
                    index=index,
                    count=count,
                )
            if len({r["prompt_tensor_hash"] for r in rows}) != 1:
                raise ValueError("Same-question prompt tensors differ")
            if pending.exists():
                if list(read_jsonl(pending)) != rows:
                    raise PermissionError("Pending and committed raw group differ")
                pending.unlink()
            if score_non_test and first["pool"] not in TEST_POOLS:
                scores = [
                    dict(slot_id=r["slot_id"], score=score_record(r, tasks[r["qid"]])) for r in rows
                ]
                durable_json(
                    directory / "scores" / (group["group_id"] + ".json"),
                    dict(raw_sha256=sha_file(path), scores=scores),
                )
            completed += count
            receipts.append(
                dict(group_id=group["group_id"], raw_sha256=sha_file(path), records=count)
            )
        receipt = dict(
            status="COMPLETE",
            plan_sha256=object_hash(plan),
            generated_records=completed,
            groups=receipts,
            test_scores_released=False,
        )
        durable_json(directory / "COMPLETE.json", receipt)
        return receipt


def collect_scored_shards(directory, *, expected_shards, expected_records=6656):
    """Validate exact disjoint baseline coverage; join score sidecars without new inference."""
    directory = Path(directory)
    rows, seen, shared_plan = [], set(), None
    for rank in range(expected_shards):
        shard = directory / f"shard{rank:03d}"
        plan, done = (
            json.loads((shard / name).read_text()) for name in ("PLAN.json", "COMPLETE.json")
        )
        shared = (
            plan["slots_sha256"],
            plan["source_sha256"],
            object_hash(plan["model_identity"]),
            object_hash(plan["adapter_identity"]),
            plan["total_groups"],
        )
        if shared_plan is None:
            shared_plan = shared
        if (
            shared != shared_plan
            or plan["shard_index"] != rank
            or plan["shard_count"] != expected_shards
        ):
            raise PermissionError("Inconsistent evaluation shard plans")
        if done.get("status") != "COMPLETE" or done.get("plan_sha256") != object_hash(plan):
            raise PermissionError("Unverified shard completion")
        if {g["group_id"] for g in done["groups"]} != set(plan["assigned_groups"]):
            raise ValueError("Shard group coverage differs")
        count = 0
        for item in done["groups"]:
            path = shard / "groups" / (item["group_id"] + ".json")
            value = json.loads(path.read_text())
            if (
                sha_file(path) != item["raw_sha256"]
                or object_hash(value["records"]) != value["records_sha256"]
            ):
                raise PermissionError("Raw group checksum differs")
            scored = json.loads((shard / "scores" / path.name).read_text())
            if scored["raw_sha256"] != item["raw_sha256"]:
                raise PermissionError("Score sidecar raw binding differs")
            scores = {r["slot_id"]: r["score"] for r in scored["scores"]}
            if len(scores) != len(value["records"]):
                raise ValueError("Score coverage differs")
            for row in value["records"]:
                if row["slot_id"] in seen or row["pool"] in TEST_POOLS:
                    raise PermissionError("Duplicate slot or sealed TEST in baseline collection")
                seen.add(row["slot_id"])
                rows.append(dict(row, score=scores[row["slot_id"]]))
                count += 1
        if count != done["generated_records"] or count != plan["expected_records"]:
            raise ValueError("Shard record coverage differs")
    if len(rows) != expected_records:
        raise ValueError("Incomplete baseline slot coverage")
    if expected_records == 6656 and Counter(r["pool"] for r in rows) != {
        "MONITOR": 4096,
        "MONITOR_DIAGNOSTIC_ONLY": 2560,
    }:
        raise ValueError("Baseline panel coverage differs")
    return rows


def verify_source_manifest(path, code_root):
    manifest, code_root = json.loads(Path(path).read_text()), Path(code_root).resolve(strict=True)
    files = manifest.get("files")
    if isinstance(files, dict):
        files = [dict(path=path, sha256=checksum, bytes=None) for path, checksum in files.items()]
    if not isinstance(files, list) or not files:
        raise ValueError("Source manifest requires nonempty path/sha256/bytes entries")
    if len({entry["path"] for entry in files}) != len(files):
        raise ValueError("Duplicate source path")
    for entry in files:
        relative, expected = entry["path"], entry["sha256"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PermissionError("Source path must be relative and contained")
        item = (code_root / relative).resolve(strict=True)
        if (
            not item.is_relative_to(code_root)
            or not item.is_file()
            or sha_file(item) != expected
            or (entry.get("bytes") is not None and item.stat().st_size != entry["bytes"])
        ):
            raise PermissionError("Source file identity differs: " + relative)
    return object_hash(manifest)


def evaluation_account(root, task_id):
    """Append actual model work with the new experiment identity before execution."""
    import time

    path = Path(root) / "accounting" / (task_id + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative integer accounting count required")
        record = dict(
            amendment_id=AMENDMENT_ID,
            task_id=task_id,
            kind=kind,
            count=count,
            metadata=metadata,
            pid=os.getpid(),
            time_ns=time.time_ns(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        with path.open("a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    return reserve
