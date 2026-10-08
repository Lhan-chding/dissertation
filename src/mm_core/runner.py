"""Registered native-backend audit sampling with durable request reservations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .execution import (
    BudgetLedger,
    append_jsonl,
    checked_path,
    derive_seed,
    exclusive_json,
    object_hash,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_gate,
    verify_stage_plan,
)

PANELS = {
    "FORMAT_BASE_TEST": "FORMAT_TUNE",
    "FORMAT_TUNE_POST_BRIDGE": "FORMAT_TUNE",
    "FORMAT_CHECK": "FORMAT_CHECK",
    "MEASUREMENT_AUDIT": "AUDIT_MEASURE",
}


def run_panel(run_root, stage, shard=0, shards=1):
    root = Path(run_root)
    freeze = verify_gate(root, stage)
    if stage not in PANELS or not 0 <= shard < shards <= 5:
        raise ValueError("Unregistered inference stage or shard")
    plan = verify_stage_plan(root, stage, shard, shards)
    from .allocations import require_allocation

    require_allocation(root, stage)
    # A restarted process must be a reviewed explicit technical retry, never an implicit resume.
    claim = root / f"raw/{stage}/SHARD_{shard}_STARTED.json"
    exclusive_json(
        claim,
        {
            "stage": stage,
            "shard": shard,
            "shards": shards,
            "time": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "stage_plan_hash": object_hash(plan),
            "freeze_hash": sha256_file(root / "manifests/PRE_INFERENCE_FREEZE.json"),
        },
    )
    ledger = BudgetLedger(root)
    from .vl_runtime import QwenRuntime

    common = (
        None if stage == "FORMAT_BASE_TEST" else read_json(root / "manifests/COMMON_START.json")
    )
    adapter = common.get("adapter_path") if common else None
    runtime = QwenRuntime(
        freeze["model_path"],
        adapter_path=adapter,
        dtype=freeze.get("dtype", "bfloat16"),
        attention_backend=freeze.get("attention_backend", "eager"),
        account=lambda kind, count, meta: ledger.reserve(kind, count, meta),
    )
    runtime.verify_identity(freeze, common)
    model_hash = freeze["model_weights_hash"] if common is None else common["model_hash"]
    rows = [
        row for row in read_jsonl(root / "data/questions.jsonl") if row["split"] == PANELS[stage]
    ]
    outputs = root / f"raw/{stage}/outputs_{shard}.jsonl"
    count = 0
    for index, row in enumerate(rows):
        if index % shards != shard:
            continue
        if sha256_file(checked_path(root, row["image_path"])) != row["image_sha256"]:
            raise ValueError("Image changed after freeze")
        for sample_index in range(4):
            seed = derive_seed(stage, model_hash, row, sample_index)
            request_id = object_hash([stage, model_hash, row["question_id"], sample_index])
            request = {
                "request_id": request_id,
                "question_id": row["question_id"],
                "stage": stage,
                "sample_index": sample_index,
                "seed": seed,
                "model_hash": model_hash,
                "processor_hash": freeze["processor_hash"],
                "image_sha256": row["image_sha256"],
                "prompt_hash": row["prompt_sha256"],
                "time": utc_now(),
                "status": "REQUESTED",
            }
            append_jsonl(root / "raw/REQUESTS.jsonl", request)
            persisted = []

            def persist_completion(
                raw, request=request, persisted=persisted, request_id=request_id
            ):
                append_jsonl(outputs, {**request, **raw, "status": "completed"})
                persisted.append(request_id)

            try:
                result = runtime.generate(
                    row, root, seed=seed, score_fields=True, on_completion=persist_completion
                )
                if not persisted:
                    raise RuntimeError("Runtime failed to persist completion before scoring")
                append_jsonl(
                    root / f"raw/{stage}/scores_{shard}.jsonl",
                    {**request, **result, "status": "scored"},
                )
                count += 1
            except BaseException as exc:
                if not persisted:
                    append_jsonl(
                        outputs,
                        {
                            **request,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "raw_text": None,
                            "tokens": [],
                            "truncated": None,
                        },
                    )
                append_jsonl(
                    root / "accounting/events.jsonl",
                    {
                        "time": utc_now(),
                        "stage": stage,
                        "status": "failed",
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                    },
                )
                raise
    receipt = {
        "time": utc_now(),
        "stage": stage,
        "status": "completed",
        "shard": shard,
        "shards": shards,
        "completed": count,
        "output_hash": sha256_file(outputs),
    }
    exclusive_json(root / f"raw/{stage}/SHARD_{shard}_COMPLETE.json", receipt)
    return receipt


def main(allowed_stages=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=sorted(allowed_stages or PANELS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run_panel(args.run_root, args.stage, args.shard, args.shards)
