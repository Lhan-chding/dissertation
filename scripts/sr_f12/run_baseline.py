#!/usr/bin/env python3
"""Run one independent one-GPU shard of the new SR-F1.2 baseline."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f1.evaluation import read_jsonl
from sr_f12.evaluation import evaluate_shard, evaluation_account, verify_source_manifest
from sr_f12.protocol import BASELINE, TEACHER_QOS, iter_model_slots, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--source-manifest", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if (root / "STOP").exists():
        raise PermissionError("STOP requested")
    config = validate_config(root / "config/SR_F1_2.json")
    source_hash = verify_source_manifest(args.source_manifest, Path(__file__).resolve().parents[2])
    identity = json.loads((root / "MODEL_ENVIRONMENT_IDENTITY.json").read_text())
    if (
        args.model_path.resolve(strict=True) != Path(identity["model_path"]).resolve(strict=True)
        or identity["model_revision"] != config["model"]["revision"]
        or identity["model_weights_hash"] != config["model"]["prior_composite_weight_hash"]
    ):
        raise PermissionError("Registered base model identity differs")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("A Slurm GPU allocation is required")
    result = subprocess.run(
        ["scontrol", "show", "job", job, "--oneliner"], capture_output=True, text=True, check=True
    )
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", result.stdout))
    if fields.get("QOS") != TEACHER_QOS or fields.get("JobState") != "RUNNING":
        raise PermissionError("Wrong GPU QoS or inactive allocation")
    import torch

    if torch.cuda.device_count() != 1:
        raise PermissionError("Each baseline worker must see exactly one GPU")
    from sr_f12.runtime import SRF12Runtime

    tasks = {
        r["qid"]: r
        for r in read_jsonl(root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")
        if r["pool"] == "MONITOR"
    }
    inputs = {
        r["qid"]: r for r in read_jsonl(root / "manifests/MODEL_INPUTS.jsonl") if r["qid"] in tasks
    }
    diagnostics = {
        (r["qid"], r["view"]): r
        for r in read_jsonl(root / "manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl")
        if r["qid"] in tasks
    }
    from mm_dev.runtime import configure_audited_backend

    determinism = configure_audited_backend()
    runtime = SRF12Runtime(
        str(args.model_path.resolve(strict=True)),
        device="cuda:0",
        output_root=root,
        account=evaluation_account(root, f"baseline-shard{args.shard_index:03d}"),
    )
    runtime.verify_identity(identity)
    runtime.identity.update(
        base_model_weights_hash=identity["model_weights_hash"], determinism=determinism
    )
    # No trained adapter is loaded. The fresh base is the B=0 common-zero policy.
    directory = (
        root / "raw/evaluation" / f"{BASELINE}_step0_baseline" / f"shard{args.shard_index:03d}"
    )
    result = evaluate_shard(
        runtime,
        iter_model_slots(tasks, [], BASELINE),
        inputs,
        diagnostics,
        tasks,
        root,
        directory,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        source_sha256=source_hash,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, sort_keys=True))


if __name__ == "__main__":
    main()
