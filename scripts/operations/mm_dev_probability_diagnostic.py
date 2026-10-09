#!/usr/bin/env python3
"""Replay the 192 saved ENGINE answers; no sampling or optimizer updates."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if not output.is_relative_to(root / "ops/numeric_gate_incident"):
        raise PermissionError("Diagnostic output must stay in the incident directory")
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(root / "code/src"))
    from mm_core.training import configure_training, trainable_state
    from mm_dev.common import verify_execution
    from mm_dev.contract import digest
    from mm_dev.runtime import atomic_json, file_hash, load_runtime, read_json

    plan_path = root / "code/docs/mm_dev_f2/design/config/MM_DEV_F2.json"
    freeze = verify_execution(plan_path, root)
    if file_hash(root / "manifests/F2_FREEZE.json") != (
        "b4279f3f1fcf46ef4e3e52326b15e74e89633dc2d8de4ec7f20d13d14435c3c9"
    ):
        raise PermissionError("Diagnostic belongs to the retained failed ENGINE freeze")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise PermissionError("A dedicated Slurm GPU allocation is required")
    allocation = subprocess.run(
        ["scontrol", "show", "job", job, "-o"],
        text=True,
        capture_output=True,
        timeout=45,
        check=True,
    ).stdout
    if not all(
        value in allocation
        for value in (
            "UserId=varun024(",
            "QOS=soujanya-poria-startfund-2026-03 ",
            "Account=rose ",
            "JobName=mmdev-f2-prob-diagnostic ",
            "gres/gpu:pro6000=1",
            "WorkDir=" + str(root) + " ",
        )
    ):
        raise PermissionError("Unexpected diagnostic allocation identity")
    paths = sorted((root / "engineering/natural/continuous/rollouts").glob("*.json"))
    if (
        len(paths) != 192
        or read_json(root / "engineering/natural/continuous/checkpoints/LATEST.json")["step"] != 0
    ):
        raise PermissionError("Diagnostic requires the exact failed first-step batch")
    original_hashes = {str(p.relative_to(root)): file_hash(p) for p in paths}
    atomic_json(
        output / "IDENTITY.json",
        {
            "job_id": job,
            "allocation": allocation,
            "script_sha256": file_hash(Path(__file__)),
            "freeze_sha256": file_hash(root / "manifests/F2_FREEZE.json"),
            "raw_hashes": original_hashes,
            "new_answers": 0,
            "optimizer_updates": 0,
        },
        exclusive=True,
    )

    def account(kind, count, metadata):
        with (output / "COST.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, "count": count, **metadata}) + "\n")

    runtime = load_runtime(
        read_json(plan_path), root, state_id="S0", freeze=freeze, account=account
    )
    configure_training(runtime)
    torch = runtime.torch
    before = trainable_state(runtime.model)
    questions = {
        row["question_id"]: row
        for row in map(json.loads, (root / "data/questions.jsonl").read_text().splitlines())
    }

    class ReplayTokens:
        def __init__(self, tokens, prompt_length):
            self.tokens, self.prompt_length = tokens, prompt_length

        def __call__(self, input_ids, scores):
            token = self.tokens[input_ids.shape[-1] - self.prompt_length]
            forced = torch.full_like(scores, -float("inf"))
            forced[:, token] = 0
            return forced

    rows = []
    for path in paths:
        record = read_json(path)
        if record["record_hash"] != digest({k: v for k, v in record.items() if k != "record_hash"}):
            raise PermissionError("Saved rollout identity changed")
        tokens = record["tokens"]
        prepared = runtime.prepare(questions[record["question_id"]], root)
        old = torch.tensor(record["old_logprobs"], dtype=torch.float32)
        full = runtime.sequence_forward(
            prepared, tokens, purpose="diagnostic_original_full_teacher_forcing"
        )["logprobs"].cpu()
        config = copy.deepcopy(runtime.training_generation_config())
        config.max_new_tokens = len(tokens)
        config.do_sample = False
        config.output_scores = False
        config.eos_token_id = None  # Replay exactly the saved length, including its EOS.
        config.min_length = 0
        config.min_new_tokens = None
        base = runtime.model.get_base_model()
        for module in (base, getattr(base, "model", None)):
            if module is not None and hasattr(module, "rope_deltas"):
                module.rope_deltas = None
        prompt_length = prepared["inputs"]["input_ids"].shape[-1]
        account("forced_cached_replay_sequences", 1, {"completion_tokens": len(tokens)})
        with torch.no_grad():
            replay = runtime.model.generate(
                **prepared["inputs"],
                generation_config=config,
                logits_processor=[ReplayTokens(tokens, prompt_length)],
            )
        if replay.sequences[0, prompt_length:].tolist() != tokens:
            raise RuntimeError("Forced replay did not retain every saved token")
        cached = torch.tensor(
            [
                float(raw[0].float().log_softmax(-1)[t])
                for raw, t in zip(replay.logits, tokens, strict=True)
            ]
        )
        row = {
            "source": str(path.relative_to(root)),
            "question_id": record["question_id"],
            "tokens": len(tokens),
            "old_logprobs": old.tolist(),
            "full_teacher_forcing_logprobs": full.tolist(),
            "cached_replay_logprobs": cached.tolist(),
            "full_absolute_differences": (full - old).abs().tolist(),
            "cached_absolute_differences": (cached - old).abs().tolist(),
        }
        atomic_json(output / "rows" / path.name, row, exclusive=True)
        rows.append(row)
        if len(rows) % 24 == 0:
            print(json.dumps({"replayed_sequences": len(rows), "target": 192}), flush=True)
    after = trainable_state(runtime.model)
    if before.keys() != after.keys() or any(not torch.equal(before[k], after[k]) for k in before):
        raise RuntimeError("Read-only diagnostic changed adapter parameters")
    if any(file_hash(root / name) != sha for name, sha in original_hashes.items()):
        raise RuntimeError("Read-only diagnostic changed original raw records")
    summary = {
        "status": "DIAGNOSTIC_COMPLETE",
        "sequences": len(rows),
        "new_answers": 0,
        "optimizer_updates": 0,
    }
    for key in ("full", "cached"):
        differences = [d for row in rows for d in row[key + "_absolute_differences"]]
        summary[key] = {
            "tokens": len(differences),
            "mean_absolute_difference": sum(differences) / len(differences),
            "maximum_absolute_difference": max(differences),
            "tokens_over_0_05": sum(d > 0.05 for d in differences),
        }
    summary["row_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {p.name: file_hash(p) for p in sorted((output / "rows").glob("*.json"))}, sort_keys=True
        ).encode()
    ).hexdigest()
    atomic_json(output / "SUMMARY.json", summary, exclusive=True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
