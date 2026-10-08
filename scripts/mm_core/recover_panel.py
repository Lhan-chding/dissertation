#!/usr/bin/env python3
"""One registered, identity-preserving continuation using the frozen PYTHONPATH.

This helper deliberately does not modify sys.path or any frozen mm_core module.
The root operator registers EXPLICIT_TECHNICAL_RETRY.json after all original jobs
are terminal. Each affected shard may invoke this helper exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from mm_core import execution
from mm_core.allocations import require_allocation
from mm_core.execution import (
    BudgetLedger,
    append_jsonl,
    checked_path,
    exclusive_json,
    locked,
    object_hash,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_gate,
    verify_stage_plan,
)
from mm_core.scoring import join_field_scores
from mm_core.vl_runtime import QwenRuntime, gold_completion

PANELS = {
    "FORMAT_BASE_TEST": "FORMAT_TUNE",
    "FORMAT_TUNE_POST_BRIDGE": "FORMAT_TUNE",
    "FORMAT_CHECK": "FORMAT_CHECK",
    "MEASUREMENT_AUDIT": "AUDIT_MEASURE",
}
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
}


def prefix_rows(root, spec, *, exact=False):
    """Check byte identity and complete JSONL records; never repair/truncate tails."""
    import json

    path = checked_path(root, spec["path"])
    size = spec.get("bytes")
    if type(size) is not int or size < 0 or type(spec.get("rows")) is not int:
        raise ValueError("Prefix byte/row counts must be nonnegative integers")
    data = path.read_bytes() if path.exists() else b""
    if len(data) < size or (exact and len(data) != size):
        raise PermissionError("Registered evidence prefix length changed")
    prefix = data[:size]
    # Beyond the immutable registered prefix another worker may be mid-append.
    # Exact checks own the whole shard file; shared prefix checks ignore its live tail.
    if (prefix and not prefix.endswith(b"\n")) or (exact and data and not data.endswith(b"\n")):
        raise PermissionError("Partial JSONL tail requires a separate incident; never truncate")
    if hashlib.sha256(prefix).hexdigest() != spec["sha256"]:
        raise PermissionError("Registered evidence prefix bytes changed")
    rows = [json.loads(line) for line in prefix.decode("utf-8").splitlines()]
    if len(rows) != spec["rows"] or any(not isinstance(row, dict) for row in rows):
        raise PermissionError("Registered evidence prefix rows changed")
    return rows


def expected_request(stage, slot, question, freeze, plan):
    return dict(
        request_id=slot["request_id"],
        question_id=slot["question_id"],
        stage=stage,
        sample_index=slot["sample_index"],
        seed=slot["seed"],
        model_hash=plan["model_hash"],
        processor_hash=freeze["processor_hash"],
        image_sha256=question["image_sha256"],
        prompt_hash=question["prompt_sha256"],
    )


def validate_rows(raw_rows, score_rows, expected):
    by_id = {}
    for row in raw_rows:
        request_id = row.get("request_id")
        if request_id not in expected or request_id in by_id:
            raise PermissionError("Unexpected or duplicate original completion")
        if any(row.get(key) != value for key, value in expected[request_id].items()):
            raise PermissionError("Completion differs from frozen request identity")
        tokens = row.get("tokens")
        if (
            row.get("status") != "completed"
            or not isinstance(row.get("raw_text"), str)
            or not isinstance(tokens, list)
            or any(type(token) is not int or token < 0 for token in tokens)
            or ("raw_tokens" in row and row["raw_tokens"] != tokens)
        ):
            raise PermissionError("Recovery cannot replace failed or invalid original responses")
        by_id[request_id] = row
    join_field_scores(raw_rows, score_rows)
    return by_id, {row["request_id"] for row in score_rows}


def validate_registration(root, shard, shards, retry_manifest):
    root = Path(root).resolve()
    manifest_path = Path(retry_manifest).resolve()
    if manifest_path != (root / "manifests/EXPLICIT_TECHNICAL_RETRY.json").resolve():
        raise PermissionError("Only the unique run-level explicit retry registration is allowed")
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "REGISTERED"
        or manifest.get("retry_index") != 1
        or manifest.get("max_explicit_technical_retries") != 1
        or manifest.get("completed_answers_must_not_be_resampled") is not True
        or manifest.get("refund_unknown_attempts") is not False
        or manifest.get("reason") != "SLURM_PREEMPTION"
        or not manifest.get("retry_id")
    ):
        raise PermissionError(
            "One explicit identity-preserving preemption retry must be registered"
        )
    stage = manifest["stage"]
    if stage not in PANELS or manifest["shards"] != shards or not 0 <= shard < shards <= 5:
        raise PermissionError("Unregistered retry stage or shard topology")
    if manifest.get("helper_sha256") != sha256_file(__file__):
        raise PermissionError("Recovery helper differs from the registered code")
    if Path(manifest["frozen_source_root"]).resolve() != Path(execution.__file__).resolve().parent:
        raise PermissionError("PYTHONPATH must resolve the original frozen implementation")
    freeze = verify_gate(root, stage)
    plan = verify_stage_plan(root, stage, shard, shards)
    if (
        manifest["pre_freeze_hash"] != sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json")
        or manifest["stage_plan_hash"] != sha256_file(root / f"manifests/STAGE_PLAN_{stage}.json")
        or manifest["model_hash"] != plan["model_hash"]
        or manifest.get("common_start_hash") != plan["common_start_hash"]
    ):
        raise PermissionError("Retry identity does not match the immutable freeze and stage plan")
    affected = manifest.get("affected_shards", {})
    if str(shard) not in affected or any(
        not key.isdigit() or not 0 <= int(key) < shards for key in affected
    ):
        raise PermissionError("Shard is outside the one registered retry event")
    selected = affected[str(shard)]
    directory = root / f"raw/{stage}"
    claim_path = directory / f"SHARD_{shard}_STARTED.json"
    claim = read_json(claim_path)
    if (
        sha256_file(claim_path) != selected["original_claim_sha256"]
        or claim.get("stage") != stage
        or claim.get("shard") != shard
        or claim.get("shards") != shards
        or claim.get("slurm_job_id") != selected["original_job_id"]
        or claim.get("freeze_hash") != plan["pre_freeze_hash"]
        or claim.get("stage_plan_hash") != object_hash(plan)
    ):
        raise PermissionError("Original shard claim changed")
    if (directory / f"SHARD_{shard}_COMPLETE.json").exists():
        raise PermissionError("A completed shard cannot consume the technical retry")
    settled = {}
    for spec in manifest["settled_original_allocations"]:
        path = checked_path(root, spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise PermissionError("Original terminal allocation receipt changed")
        record = read_json(path)
        terminal = record.get("terminal", {})
        if (
            record.get("status") != "TERMINAL_VERIFIED"
            or record.get("stage") != stage
            or terminal.get("job_id") != record.get("job_id")
            or terminal.get("squeue_absent") is not True
            or terminal.get("state") not in TERMINAL
        ):
            raise PermissionError("Every original allocation must be verified terminal")
        if record["job_id"] in settled:
            raise PermissionError("Duplicate original terminal allocation identity")
        settled[record["job_id"]] = record
    original_claims = [
        read_json(directory / f"SHARD_{index}_STARTED.json") for index in range(shards)
    ]
    if set(settled) != {row["slurm_job_id"] for row in original_claims}:
        raise PermissionError("Terminal receipts must cover all original shard jobs")
    if (
        settled[selected["original_job_id"]]["allocation_key"]
        != selected["original_allocation_key"]
    ):
        raise PermissionError("Original allocation key mismatch")
    for name, path in (
        ("raw_prefix", f"raw/{stage}/outputs_{shard}.jsonl"),
        ("scores_prefix", f"raw/{stage}/scores_{shard}.jsonl"),
        ("requests_prefix", "raw/REQUESTS.jsonl"),
        ("ledger_prefix", "accounting/COST_LEDGER.jsonl"),
    ):
        spec = selected[name] if name in {"raw_prefix", "scores_prefix"} else manifest[name]
        if spec["path"] != path:
            raise PermissionError("Retry prefix path does not match its role")
    raw = prefix_rows(root, selected["raw_prefix"], exact=True)
    scores = prefix_rows(root, selected["scores_prefix"], exact=True)
    requests = prefix_rows(root, manifest["requests_prefix"])
    ledger_rows = prefix_rows(root, manifest["ledger_prefix"])
    questions = {row["question_id"]: row for row in read_jsonl(root / "data/questions.jsonl")}
    slots = [slot for slot in plan["expected_slots"] if slot["shard"] == shard]
    expected = {
        slot["request_id"]: expected_request(
            stage, slot, questions[slot["question_id"]], freeze, plan
        )
        for slot in slots
    }
    by_id, scored_ids = validate_rows(raw, scores, expected)
    original_requests = []
    for row in requests:
        if row.get("request_id") not in expected:
            continue
        if any(row.get(key) != value for key, value in expected[row["request_id"]].items()):
            raise PermissionError("Original REQUESTS identity differs from frozen request")
        original_requests.append(row)
    original_ids = [row["request_id"] for row in original_requests]
    if len(original_ids) != len(set(original_ids)) or not set(by_id) <= set(original_ids):
        raise PermissionError("Original attempts are duplicated or completions lack requests")
    unknown = set(original_ids) - set(by_id)
    if set(selected["unknown_request_ids"]) != unknown:
        raise PermissionError("Unknown original attempts must be explicitly registered")
    return dict(
        root=root,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_hash=sha256_file(manifest_path),
        freeze=freeze,
        plan=plan,
        selected=selected,
        questions=questions,
        slots=slots,
        expected=expected,
        raw=raw,
        scores=scores,
        by_id=by_id,
        scored_ids=scored_ids,
        unknown=unknown,
        ledger_rows=ledger_rows,
    )


def register_retry_start(context, shard, allocation):
    root, manifest = context["root"], context["manifest"]
    folder = root / "accounting/recovery"
    with locked(folder / "REGISTRATION.lock"):
        registration = dict(
            retry_id=manifest["retry_id"],
            retry_index=1,
            retry_manifest_hash=context["manifest_hash"],
            helper_sha256=manifest["helper_sha256"],
        )
        path = folder / "RETRY_REGISTRATION.json"
        if path.exists():
            if read_json(path) != registration:
                raise PermissionError("A different technical retry was already registered")
        else:
            exclusive_json(path, registration)
            append_jsonl(
                root / "accounting/events.jsonl",
                dict(time=utc_now(), event="EXPLICIT_TECHNICAL_RETRY_REGISTERED", **registration),
            )
        exclusive_json(
            folder / f"SHARD_{shard}_STARTED.json",
            dict(
                time=utc_now(),
                stage=manifest["stage"],
                shard=shard,
                allocation=allocation,
                **registration,
                original_claim_sha256=context["selected"]["original_claim_sha256"],
                raw_prefix=context["selected"]["raw_prefix"],
                scores_prefix=context["selected"]["scores_prefix"],
            ),
        )


def retain_unknown_costs(context, ledger):
    root, manifest = context["root"], context["manifest"]
    for request_id in sorted(context["unknown"]):
        request = context["expected"][request_id]
        consumed = sum(
            row["amount"]
            for row in context["ledger_rows"]
            if row.get("kind") == "completion_attempts"
            and row.get("identity", {}).get("question_id") == request["question_id"]
            and row.get("identity", {}).get("seed") == request["seed"]
        )
        if consumed < 1:
            ledger.reserve(
                "completion_attempts",
                1,
                dict(
                    request_id=request_id,
                    question_id=request["question_id"],
                    seed=request["seed"],
                    status="ORIGINAL_UNKNOWN_CONSERVATIVELY_CONSUMED",
                    retry_id=manifest["retry_id"],
                ),
            )
        append_jsonl(
            root / "accounting/events.jsonl",
            dict(
                time=utc_now(),
                event="ORIGINAL_UNKNOWN_ATTEMPT_RETAINED",
                request_id=request_id,
                retry_id=manifest["retry_id"],
                original_reserved_count=consumed,
                extra_conservative_reservation=0 if consumed >= 1 else 1,
                refunded=0,
            ),
        )


def score_saved_completion(runtime, question, root, raw):
    prepared = runtime.prepare(question, root)
    start = time.perf_counter()
    result = dict(raw)
    result["self_field_surprisal"] = (
        runtime.field_scores(
            prepared, raw["tokens"], provenance="generated_tokens_actual_self_prefix"
        )
        if raw["tokens"]
        else None
    )
    result["gold_teacher_forced_field_nll"] = runtime.field_scores(
        prepared,
        runtime.encode_completion(gold_completion(question)),
        provenance="gold_completion_answer_prefix_contains_gold_readings",
    )
    result["scoring_seconds"] = time.perf_counter() - start
    result["status"] = "scored"
    return result


def recover_panel(run_root, shard, shards, retry_manifest):
    root = Path(run_root).resolve()
    with locked(root / f"accounting/recovery/SHARD_{shard}.lock"):
        context = validate_registration(root, shard, shards, retry_manifest)
        manifest, freeze = context["manifest"], context["freeze"]
        stage = manifest["stage"]
        ledger = BudgetLedger(root)
        # The frozen allocation verifier also reads COST_LEDGER; serialize that
        # read with every reservation made by another retry worker.
        with locked(root / "accounting/BUDGET.lock"):
            allocation = require_allocation(root, stage)
            ledger.totals()
        if allocation["allocation_key"] != context["selected"]["retry_allocation_key"]:
            raise PermissionError("Live allocation is not the registered retry allocation")
        register_retry_start(context, shard, allocation)
        try:
            retain_unknown_costs(context, ledger)
            common = (
                None
                if stage == "FORMAT_BASE_TEST"
                else read_json(root / "manifests/COMMON_START.json")
            )
            runtime = QwenRuntime(
                freeze["model_path"],
                adapter_path=common.get("adapter_path") if common else None,
                dtype=freeze.get("dtype", "bfloat16"),
                attention_backend=freeze.get("attention_backend", "eager"),
                account=lambda kind, count, meta: ledger.reserve(kind, count, meta),
            )
            runtime.verify_identity(freeze, common)
            outputs = root / context["selected"]["raw_prefix"]["path"]
            score_path = root / context["selected"]["scores_prefix"]["path"]
            generated = repaired = 0
            for slot in context["slots"]:
                if sha256_file(context["manifest_path"]) != context["manifest_hash"]:
                    raise PermissionError("Retry registration changed during execution")
                request_id = slot["request_id"]
                if request_id in context["scored_ids"]:
                    continue
                question = context["questions"][slot["question_id"]]
                attempt_id = object_hash(
                    [
                        manifest["retry_id"],
                        shard,
                        request_id,
                        "score_saved" if request_id in context["by_id"] else "generate",
                    ]
                )
                append_jsonl(
                    root / "raw/REQUESTS.jsonl",
                    dict(
                        **context["expected"][request_id],
                        time=utc_now(),
                        status="REQUESTED",
                        attempt_id=attempt_id,
                        retry_id=manifest["retry_id"],
                        attempt_kind="SCORING_ONLY"
                        if request_id in context["by_id"]
                        else "GENERATION_RETRY",
                    ),
                )
                if request_id in context["by_id"]:
                    raw = context["by_id"][request_id]
                    result = score_saved_completion(runtime, question, root, raw)
                    append_jsonl(score_path, result)
                    repaired += 1
                else:
                    request = dict(
                        **context["expected"][request_id],
                        time=utc_now(),
                        attempt_id=attempt_id,
                        retry_id=manifest["retry_id"],
                    )
                    persisted = []

                    def persist(raw, request=request, persisted=persisted):
                        completed = {**request, **raw, "status": "completed"}
                        append_jsonl(outputs, completed)
                        persisted.append(completed)

                    result = runtime.generate(
                        question, root, seed=slot["seed"], score_fields=True, on_completion=persist
                    )
                    if len(persisted) != 1:
                        raise RuntimeError("Recovery runtime must persist exactly one completion")
                    scored = {**persisted[0], **result, "status": "scored"}
                    join_field_scores(persisted, [scored])
                    append_jsonl(score_path, scored)
                    generated += 1
            prefix_rows(root, context["selected"]["raw_prefix"])
            prefix_rows(root, context["selected"]["scores_prefix"])
            raw_rows, score_rows = read_jsonl(outputs), read_jsonl(score_path)
            by_id, scored_ids = validate_rows(raw_rows, score_rows, context["expected"])
            if set(by_id) != set(context["expected"]) or scored_ids != set(by_id):
                raise PermissionError("Recovery did not fill the exact registered raw/score slots")
            if sha256_file(context["manifest_path"]) != context["manifest_hash"]:
                raise PermissionError("Retry registration changed before completion receipt")
            if (
                sha256_file(root / f"raw/{stage}/SHARD_{shard}_STARTED.json")
                != context["selected"]["original_claim_sha256"]
            ):
                raise PermissionError("Original claim changed during recovery")
            receipt = dict(
                time=utc_now(),
                stage=stage,
                status="completed",
                shard=shard,
                shards=shards,
                completed=len(raw_rows),
                output_hash=sha256_file(outputs),
                scores_hash=sha256_file(score_path),
                retry_id=manifest["retry_id"],
                retry_manifest_hash=context["manifest_hash"],
                helper_sha256=manifest["helper_sha256"],
                preserved_completions=len(context["raw"]),
                new_completions=generated,
                repaired_score_sidecars=repaired,
                original_claim_sha256=context["selected"]["original_claim_sha256"],
            )
            exclusive_json(root / f"raw/{stage}/SHARD_{shard}_COMPLETE.json", receipt)
            append_jsonl(
                root / "accounting/events.jsonl",
                dict(event="EXPLICIT_RETRY_SHARD_COMPLETED", **receipt),
            )
            return receipt
        except BaseException as error:
            append_jsonl(
                root / "accounting/events.jsonl",
                dict(
                    time=utc_now(),
                    event="EXPLICIT_RETRY_SHARD_FAILED",
                    stage=stage,
                    shard=shard,
                    retry_id=manifest["retry_id"],
                    error_type=type(error).__name__,
                    no_automatic_retry=True,
                    prior_attempt_costs_refunded=0,
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--retry-manifest", type=Path, required=True)
    args = parser.parse_args()
    recover_panel(args.run_root, args.shard, args.shards, args.retry_manifest)


if __name__ == "__main__":
    main()
