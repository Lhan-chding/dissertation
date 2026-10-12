"""Durable SR-F2 evaluation groups and RNG-isolated MONITOR-lite measurements."""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from sr_f1.data import load_inputs, load_tasks
from sr_f1.evaluation import TEST_POOLS, sha_file
from sr_f12.evaluation import durable_json, group_slots, shard_lease
from sr_f12.evaluation import validate_record as validate_historical_record
from sr_f12.protocol import AMENDMENT_ID as HISTORICAL_ID
from sr_f12.protocol import object_hash

from .diagnostics import EXPERIMENT_ID, monitor_slots, prepare_input


def _identity(runtime, name):
    value = getattr(runtime, name, None)
    return value() if callable(value) else value


def _read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def score_record(row, task):
    from .scoring import score_v2

    if row["protocol"] == "plain_answer":
        from sr_f1.chartqa import score_chartqa

        return score_chartqa(str(task["answer"]).strip(), row["raw_text"].strip())
    text = "" if row.get("format_protocol_error") else row["raw_text"]
    result = score_v2(text, task["world"], task["query"], protocol=row["protocol"])
    if row.get("format_protocol_error"):
        result["reason"] = row["format_protocol_error"]
    return result


def validate_record(
    row, slot, *, input_hash, model_identity, adapter_identity, source_sha256, index, count
):
    if row.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("SR-F2 experiment identity differs")
    # Reuse the immutable token/sampler/prefill validator, keeping its historical
    # format identity local to this validation view. No raw artifact is rewritten.
    validate_historical_record(
        dict(row, amendment_id=HISTORICAL_ID),
        slot,
        input_hash=input_hash,
        model_identity=model_identity,
        adapter_identity=adapter_identity,
        source_sha256=source_sha256,
        index=index,
        count=count,
    )
    if slot["protocol"] in ("free", "list"):
        if (
            row.get("assistant_prefill") != ""
            or row.get("generated_text") != row["raw_text"]
            or row.get("answer_extraction") != "last_balanced_json"
            or row.get("finish_reason") not in ("eos", "length")
            or row.get("balanced_token_count") is not None
        ):
            raise PermissionError("FREE/LIST prefill, extraction, or termination changed")
        if (
            row["finish_reason"] == "length"
            and len(row["tokens"]) != slot["generation"]["max_new_tokens"]
        ):
            raise ValueError("FREE/LIST length stop is not at registered budget")


def run_group(
    runtime,
    group,
    inputs,
    diagnostics,
    tasks,
    root,
    directory,
    *,
    source_sha256,
    score=True,
    test_released=False,
):
    """Persist all raw completions before validation/scoring; never resample partials.

    The caller holds one shard lease around a sequence of these operations. TEST
    raw generation is permitted, but scoring requires an explicit release flag.
    """
    root, directory = Path(root).resolve(strict=True), Path(directory).resolve()
    if not directory.is_relative_to(root):
        raise PermissionError("Evaluation directory escapes run root")
    slots = group["slots"]
    first, count = slots[0], len(slots)
    checked = group_slots(slots)
    if len(checked) != 1 or checked[0]["group_id"] != group["group_id"]:
        raise ValueError("Group identity differs from registered slots")
    if score and first["pool"] in (*TEST_POOLS, "CHARTQA") and not test_released:
        raise PermissionError("TEST scientific scores remain sealed")
    model = _identity(runtime, "stable_model_identity")
    adapter = _identity(runtime, "current_adapter_identity")
    if not isinstance(model, dict) or not isinstance(adapter, dict):
        raise ValueError("Runtime model and adapter identity required")
    tokenizer = getattr(runtime, "tokenizer", None) or runtime.processor.tokenizer
    request = prepare_input(first, inputs, diagnostics, tokenizer)
    image = (root / request["image_file"]).resolve(strict=True)
    if not image.is_relative_to(root):
        raise PermissionError("Evaluation image escapes run root")
    image_hash = sha_file(image)
    input_hash = object_hash(dict(text=request["text"], image_sha256=image_hash))
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
        rows = _read_lines(pending)
        if len(rows) != count:
            raise RuntimeError(
                "Incomplete durable group: preserve for technical recovery; no resampling"
            )
    else:
        rows = []

        def persist(completion):
            index = len(rows)
            if index >= count:
                raise RuntimeError("Runtime emitted excess group records")
            row = dict(completion)
            for key in ("group_seed", "group_row_index", "group_size", "protocol", "seed"):
                row["runtime_" + key] = completion.get(key)
            row.update(slots[index])
            row.update(request)
            row.update(
                input_hash=input_hash,
                image_hash=image_hash,
                source_sha256=source_sha256,
                experiment_id=EXPERIMENT_ID,
                group_row_index=index,
                group_size=count,
            )
            with pending.open("a") as stream:
                stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)

        runtime.generate_group(
            None,
            root,
            [s["sampling_seed"] for s in slots],
            text=request["text"],
            image_path=image,
            protocol=first["protocol"],
            generation=dict(first["generation"], num_return_sequences=count),
            on_completion=persist,
        )
        if len(rows) != count:
            raise RuntimeError("Runtime did not durably deliver complete group")
    value = dict(
        group_id=group["group_id"],
        group_index=group["group_index"],
        records=rows,
        records_sha256=object_hash(rows),
    )
    durable_json(path, value)
    if len(rows) != count:
        raise ValueError("Persisted group coverage differs")
    for index, (row, slot) in enumerate(zip(rows, slots, strict=True)):
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
        raise PermissionError("Same-question prompt tensors differ")
    if pending.exists():
        if _read_lines(pending) != rows:
            raise PermissionError("Pending and committed raw group differ")
        pending.unlink()
    scored = None
    if score:
        scored = [dict(slot_id=r["slot_id"], score=score_record(r, tasks[r["qid"]])) for r in rows]
        durable_json(
            directory / "scores" / path.name,
            dict(raw_sha256=sha_file(path), scorer_version="v2", scores=scored),
        )
    return dict(
        group_id=group["group_id"],
        raw_sha256=sha_file(path),
        records_sha256=object_hash(rows),
        records=rows,
        scores=scored,
    )


def summarize_scores(rows):
    """Equal prompt draws within family; report contrasts without outcome filtering."""
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    result = {}
    metrics = ("A", "E", "E6", "p_read", "P", "J", "P6", "J6")
    for family, items in sorted(families.items()):
        scores = [r["score"] for r in items]
        summary = dict(count=len(items), prompts=len({r["qid"] for r in items}))
        for key in metrics:
            values = [float(s[key]) for s in scores]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("Nonfinite/out-of-range semantic score")
            summary[key] = sum(values) / len(values)
        for field, alternatives in (
            ("missing_keys", ("missing_keys", "missing_count", "missing_items")),
            ("extra_keys", ("extra_keys", "extra_count", "extra_items")),
        ):
            values = []
            for s in scores:
                value = next((s[k] for k in alternatives if k in s), None)
                if value is None:
                    raise ValueError("Scorer omitted evidence item counts")
                values.append(len(value) if isinstance(value, (list, tuple, set)) else value)
            summary[field] = sum(values) / len(values)
        summary.update(
            A_minus_P=summary["A"] - summary["P"], contract_slack=6 if family == "TOPK" else 0
        )
        groups = defaultdict(list)
        for row in items:
            groups[row["qid"]].append(row["score"])
        contrast = {}
        for key in metrics:
            values = [[float(s[key]) for s in g] for g in groups.values()]
            variances = [sum((v - sum(g) / len(g)) ** 2 for v in g) / len(g) for g in values]
            contrast[key] = dict(
                nonconstant_group_fraction=sum(max(g) != min(g) for g in values) / len(values),
                mean_within_group_variance=sum(variances) / len(values),
            )
        summary["group_contrasts"] = contrast
        result[family] = summary
    return result


def evaluate_monitor(runtime, tasks, inputs, root, directory, *, model_id, step, source_sha256):
    """One fixed 512-response measurement, isolated from every training RNG stream."""
    from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state

    slots = monitor_slots(tasks, model_id, step)
    groups = group_slots(slots)
    policy_hash = state_hash(trainable_state(runtime.model))
    directory = Path(directory).resolve()
    if not directory.is_relative_to(Path(root).resolve(strict=True)):
        raise PermissionError("Monitor directory escapes run root")
    state, was_training = capture_rng(), runtime.model.training
    try:
        runtime.model.eval()
        plan = dict(
            experiment_id=EXPERIMENT_ID,
            model_id=model_id,
            step=step,
            source_sha256=source_sha256,
            policy_hash=policy_hash,
            model_identity=_identity(runtime, "stable_model_identity"),
            adapter_identity=_identity(runtime, "current_adapter_identity"),
            slots_sha256=object_hash(slots),
            expected_records=512,
        )
        with shard_lease(directory):
            durable_json(directory / "PLAN.json", plan)
            allowed = {g["group_id"] for g in groups}
            for path in (directory / "groups").glob("*"):
                if path.name.removesuffix(".pending.jsonl").removesuffix(".json") not in allowed:
                    raise PermissionError("Unregistered monitor raw group")
            rows, receipts = [], []
            for group in groups:
                receipt = run_group(
                    runtime,
                    group,
                    inputs,
                    {},
                    tasks,
                    root,
                    directory,
                    source_sha256=source_sha256,
                    score=True,
                )
                score_map = {s["slot_id"]: s["score"] for s in receipt["scores"]}
                rows.extend(dict(r, score=score_map[r["slot_id"]]) for r in receipt["records"])
                receipts.append(
                    {k: receipt[k] for k in ("group_id", "raw_sha256", "records_sha256")}
                )
            if len(rows) != 512 or state_hash(trainable_state(runtime.model)) != policy_hash:
                raise RuntimeError("Monitor coverage or policy changed")
            summary = dict(
                status="COMPLETE",
                experiment_id=EXPERIMENT_ID,
                model_id=model_id,
                step=step,
                policy_hash=policy_hash,
                count=len(rows),
                records=len(rows),
                records_sha256=object_hash(receipts),
                groups=receipts,
                plan_sha256=object_hash(plan),
                families=summarize_scores(rows),
                rng_isolated=True,
                test_access=False,
            )
            durable_json(directory / "COMPLETE.json", summary)
            return summary
    finally:
        runtime.model.train(was_training)
        restore_rng(state)


def run_monitor(runtime, root, directory, model_id, step, *, source_sha256=None):
    if source_sha256 is None:
        freeze = Path(root) / "REGISTRATION.json"
        source_sha256 = json.loads(freeze.read_text())["source_manifest_sha256"]
    return evaluate_monitor(
        runtime,
        load_tasks(root),
        load_inputs(root),
        root,
        Path(directory) / "monitor" / f"step-{step:02d}",
        model_id=model_id,
        step=step,
        source_sha256=source_sha256,
    )


def evaluation_account(root, task_id):
    if not task_id or Path(task_id).name != task_id:
        raise ValueError("Simple accounting task identifier required")
    path = Path(root) / "accounting" / (task_id + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative integer accounting count required")
        record = dict(
            experiment_id=EXPERIMENT_ID,
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
