"""Bounded same-token diagnostic; never samples or updates scientific state.

Run with PYTHONPATH pointing at the immutable scientific deployment. Both
precisions use saved b1/SHUF step-7 group-6 tokens and checkpoint step 6.
FP32 is a diagnostic companion, not authorization to change training precision.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

OWNER = Path("/projects/_ssd/varunssd/louis-ssvc")
RUN_ID = "G3-b1-SHUF"
POSITIONS = (151, 162)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def select_records(saved):
    spec, records = saved["spec"], saved["records"]
    if (spec["identity"]["run_id"], spec["step"], spec["assignment"]["group_index"]) != (
        RUN_ID,
        7,
        6,
    ):
        raise PermissionError("Wrong frozen diagnostic group")
    if len(records) != 8:
        raise PermissionError("Diagnostic requires the complete original G8")
    for i, row in enumerate(records):
        if row["row_id"] != f"{RUN_ID}/007/06/{i}":
            raise PermissionError("Saved row order/identity changed")
        if row["adapter_identity"]["step"] != 6 or row["adapter_identity"]["run_id"] != RUN_ID:
            raise PermissionError("Saved policy identity differs from step six")
        if row["tokens"] != row["raw_tokens"] or len(row["tokens"]) != len(row["sampler_logprobs"]):
            raise PermissionError("Saved token identity/length mismatch")
        if row["generation_status"] != "COMPLETE" or row.get("technical_validation_errors"):
            raise PermissionError("Saved generation is not complete and valid")
    return records


def verify_prompt(prepared, row):
    actual = {
        key: value.detach().cpu().tolist()
        for key, value in prepared["inputs"].items()
        if key != "pixel_values" and hasattr(value, "detach")
    }
    if actual != row["native_prompt_inputs"]:
        raise PermissionError("Reconstructed prompt token/mask/grid inputs differ")
    if prepared["routing"]["input_tensor_hash"] != row["prompt_tensor_hash"]:
        raise PermissionError("Reconstructed processor/pixel identity differs")
    return {
        "native_prompt_inputs_equal": True,
        "processor_tensor_hash_equal": True,
        "prompt_tokens": len(actual["input_ids"][0]),
        "image_grid_thw": actual.get("image_grid_thw"),
        "routing": prepared["routing"],
    }


def differences(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Probability vectors must have the same nonempty token path")
    values = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {
        "mean_absolute_difference": sum(values) / len(values),
        "max_absolute_difference": max(values),
        "max_position": values.index(max(values)),
    }


def output_path(root, output):
    root = root.resolve(strict=True)
    if not root.is_relative_to(OWNER.resolve(strict=True)):
        raise PermissionError("Run root outside owned project")
    parent = output.parent.resolve(strict=True)
    if not parent.is_relative_to(root / "CONTINUATION") or output != parent / output.name:
        raise PermissionError("Output must be canonical in dedicated CONTINUATION directory")
    if output.exists() or output.with_suffix(".started.json").exists():
        raise FileExistsError("Diagnostic attempt already exists; inspect it before any retry")
    return root


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(args):
    root = output_path(args.root, args.output)
    with args.output.with_suffix(".started.json").open("x") as f:
        json.dump({"at": now(), "pid": os.getpid(), "precision": args.precision}, f)
    result = {
        "status": "RUNNING",
        "started_at": now(),
        "precision": args.precision,
        "run_id": RUN_ID,
        "checkpoint_step": 6,
        "attempted_step": 7,
        "group_index": 6,
        "new_samples": 0,
        "optimizer_updates": 0,
        "gate_decision": "NOT_AUTHORIZED_BY_DIAGNOSTIC",
        "rows": [],
        "limitations": [
            "No cached forced-token replay in this bounded diagnostic.",
            "FP32 changes only this diagnostic process; no production fix implied.",
        ],
    }
    save(args.output, result)
    try:
        import sr_f5.pilot as pilot
        from sr_f4.training import load_checkpoint
        from sr_f5.runtime import load

        frozen = read(root / "G3/FROZEN.json")
        code = Path(pilot.__file__).resolve().parents[2]
        if args.code is not None and code != args.code.resolve(strict=True):
            raise PermissionError("Imported scientific deployment differs from requested code")
        if args.revision is not None and frozen["code_revision"] != args.revision:
            raise PermissionError("Requested scientific revision differs from frozen G3")
        if pilot.source_identity() != frozen["source_identity"]:
            raise PermissionError("Imported scientific source differs from frozen G3")
        result.update(science_revision=frozen["code_revision"], science_code=str(code))
        lane = root / "G3/RUNS/b1/SHUF/primary"
        if (lane / "updates/007.json").exists() or (
            lane / "checkpoints/step-007/COMMITTED.json"
        ).exists():
            raise PermissionError("Step seven was already committed")
        saved = read(lane / "rollouts/007/06.json")
        records = select_records(saved)
        identity = read(lane / "IDENTITY.json")
        checkpoint = lane / "checkpoints/step-006"
        if read(checkpoint / "COMMITTED.json")["identity"] != identity:
            raise PermissionError("Checkpoint marker identity differs")
        pilot.checkpoint_header(checkpoint / "state.pt", identity, 6)
        task = f"LH-G3-DIAG-{args.precision}-{os.environ.get('SLURM_JOB_ID', 'local')}"
        runtime, optimizer, scheduler = load(root, "b1", task, training=True)
        step, cursor = load_checkpoint(
            checkpoint / "state.pt", runtime, optimizer, scheduler, identity=identity
        )
        if step != 6 or cursor != {"next_step": 7, "committed_log_step": 6}:
            raise PermissionError("Wrong restored checkpoint cursor")
        runtime.identity.update(run_id=RUN_ID, step=6, diagnostic_only=True)
        result["checkpoint_restored"] = True
        result["gradient_checkpointing"] = bool(getattr(runtime, "_sr_f12_checkpointing", False))
        result["original_parameter_dtypes"] = sorted(
            {str(p.dtype) for p in runtime.model.parameters()}
        )
        originals = {}
        for line in (root / "NUMERICAL_TOKEN_AUDIT.jsonl").open():
            audit = json.loads(line)
            if (
                audit.get("run_id") == RUN_ID
                and audit.get("step") == 7
                and audit.get("status") == "UNCOMMITTED_FAILURE"
            ):
                originals = {r["row_id"]: r for r in audit["probability_vectors"]}
                result["original_diagnostics"] = audit["diagnostics"]
        if not all(r["row_id"] in originals for r in records):
            raise PermissionError("Original failed probability vectors missing")
        prepared = runtime.prepare_input(
            records[0]["model_input"], root, protocol=records[0]["protocol"]
        )
        result["prompt_checks"] = [verify_prompt(prepared, row) for row in records]
        if args.precision == "fp32":
            runtime.model.float()
            prepared = dict(
                prepared,
                inputs={
                    k: v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v
                    for k, v in prepared["inputs"].items()
                },
            )
        result["diagnostic_parameter_dtypes"] = sorted(
            {str(p.dtype) for p in runtime.model.parameters()}
        )
        for row in records:
            original = originals[row["row_id"]]
            if (
                original["tokens"] != row["tokens"]
                or original["sampler_logprobs"] != row["sampler_logprobs"]
            ):
                raise PermissionError(
                    "Failed audit and original rollout token/probability vectors differ"
                )
            entry = {
                "row_id": row["row_id"],
                "tokens": row["tokens"],
                "sampler_logprobs": row["sampler_logprobs"],
                "original_train_logprobs": original["train_logprobs"],
                "forwards": [],
            }
            result["rows"].append(entry)
            lengths = [len(row["tokens"])]
            if row["row_id"].endswith(("/3", "/5")):
                lengths += sorted({p + 1 for p in POSITIONS if p + 1 < len(row["tokens"])})
            for length in lengths:
                for grad in (True, False):
                    item = dict(row, tokens=row["tokens"][:length], prepared=prepared)
                    values = runtime.batch_sequence_forward(
                        [item], purpose=f"diagnostic_{args.precision}_same_tokens", grad=grad
                    )[0]
                    probs = values.detach().cpu().tolist()
                    del values
                    record = {
                        "length": length,
                        "grad_enabled": grad,
                        "logprobs": probs,
                        "versus_sampler": differences(probs, row["sampler_logprobs"][:length]),
                        "versus_original_train": differences(
                            probs, original["train_logprobs"][:length]
                        ),
                        "selected_tokens": [
                            {"position": p, "token_id": row["tokens"][p], "logprob": probs[p]}
                            for p in POSITIONS
                            if p < length
                        ],
                    }
                    entry["forwards"].append(record)
                    save(args.output, result)
        result.update(
            status="DIAGNOSTIC_COMPLETE",
            finished_at=now(),
            model_calls=len([f for row in result["rows"] for f in row["forwards"]]),
        )
        save(args.output, result)
    except BaseException as exc:
        result.update(
            status="DIAGNOSTIC_FAILED",
            error=repr(exc),
            traceback=traceback.format_exc(),
            finished_at=now(),
        )
        save(args.output, result)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision", "--mode", dest="precision", choices=["bf16", "fp32"], required=True
    )
    parser.add_argument("--code", type=Path)
    parser.add_argument("--revision")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
