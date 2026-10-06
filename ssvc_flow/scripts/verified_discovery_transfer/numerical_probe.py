"""Disposable train-only precision/shape diagnosis; no sampling or optimizer step."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from src.verified_discovery_transfer.bridge import _gradient_comparison, _gradient_snapshot
from src.verified_discovery_transfer.canonical_targets import build_training_view, gold_targets
from src.verified_discovery_transfer.cli import _backend, _context
from src.verified_discovery_transfer.public_tasks import RoleDataset
from src.verified_discovery_transfer.queue import write_json
from src.verified_discovery_transfer.sft_loss import (
    accumulated_backward,
    collate_examples,
    encode_example,
    forward_batch,
    per_sequence_nll,
)
from src.verified_discovery_transfer.sft_runtime import SFTRuntime


@contextmanager
def preserve_matmul_settings(*, strict=False):
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        if strict:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@preserve_matmul_settings()
def diagnose(backend, rows, output):
    runtime = SFTRuntime.from_frozen_backend(
        backend, seed=73101, view_identity={"purpose": "TRAIN_ONLY_NUMERICAL_DIAGNOSIS"}
    )
    adapter = backend.adapter
    examples = [
        encode_example(adapter, row["prompt"], row["target"], task_id=row["task_id"])
        for row in rows
    ]
    result = {
        "parent_receipt": backend.receipt,
        "tasks": [row["task_id"] for row in rows],
        "generations": 0,
        "optimizer_updates": 0,
        "formal_sft_changed": False,
        "float32_scope": "Upcast the same BF16-rounded weights, not original FP32 weights",
        "phases": {},
    }
    for phase in ("inherited_bf16", "diagnostic_upcast_fp32"):
        if phase == "diagnostic_upcast_fp32":
            runtime.optimizer.zero_grad(set_to_none=True)
            adapter.model.float()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        adapter.model.train()
        phase_result = {
            "examples": [],
            "train": True,
            "use_cache": False,
            "forward_projection_probe_autograd": False,
            "gradient_probe_autograd": True,
            "same_hidden_fp32_projection_precision": "highest; TF32 disabled locally",
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "gradient_checkpointing": adapter.model.is_gradient_checkpointing,
        }
        with torch.no_grad():
            for index in (0, 4):
                example = examples[index]
                inputs, labels = collate_examples(
                    [example], pad_id=adapter.pad_id, device=adapter.device
                )
                adapter._reset_positions()
                out = adapter.model(**inputs, use_cache=False, output_hidden_states=True)
                _, nll, mask = per_sequence_nll(out.logits, labels)
                full_nll = nll[0][mask[0]].cpu()
                hidden = out.hidden_states[-1]
                head = adapter.model.get_output_embeddings()
                row_nll, fp32_row_nll = [], []
                for j, target in enumerate(example.target_ids):
                    position = len(example.prompt_ids) - 1 + j
                    same_hidden = hidden[:, position : position + 1]
                    single = head(same_hidden)[0, 0].float()
                    row_nll.append(float(-single.log_softmax(-1)[target]))
                    # Same BF16-rounded hidden/weights, FP32 projection only.
                    with preserve_matmul_settings(strict=True):
                        precise = torch.nn.functional.linear(
                            same_hidden.float(),
                            head.weight.float(),
                            head.bias.float() if head.bias is not None else None,
                        )[0, 0]
                    fp32_row_nll.append(float(-precise.log_softmax(-1)[target]))
                shapes = {
                    "input": list(inputs["input_ids"].shape),
                    "hidden_dtype": str(hidden.dtype),
                    "head_dtype": str(head.weight.dtype),
                    "logits_dtype": str(out.logits.dtype),
                }
                del out, hidden, nll, mask
                adapter._reset_positions()
                repeated, repeated_labels = forward_batch(adapter, [example])
                repeated_nll = per_sequence_nll(repeated, repeated_labels)[1][0]
                repeated_nll = repeated_nll[labels[0, 1:].ne(-100)].cpu().tolist()
                del repeated, repeated_labels
                prepared = adapter.prepare(rows[index]["prompt"], backend.data_root)
                prefix = -adapter.logprobs(
                    prepared, list(example.target_ids), require_grad=False
                )
                adapter.model.train()
                phase_result["examples"].append(
                    {
                        "task_id": example.task_id,
                        "target_ids": list(example.target_ids),
                        "prompt_length": len(example.prompt_ids),
                        "full_nll": full_nll.tolist(),
                        "repeat_full_nll": repeated_nll,
                        "same_hidden_single_row_nll": row_nll,
                        "same_hidden_fp32_single_row_nll": fp32_row_nll,
                        "prefix_nll": prefix.cpu().tolist(),
                        **shapes,
                    }
                )
        gradients = []
        for micro in (4, 4, 1):
            runtime.optimizer.zero_grad(set_to_none=True)
            accumulated_backward(adapter, examples[:4], microbatch_size=micro)
            gradients.append(_gradient_snapshot(runtime))
        phase_result["same_path_gradient"] = _gradient_comparison(gradients[0], gradients[1])
        phase_result["microbatch_gradient"] = _gradient_comparison(gradients[0], gradients[2])
        del gradients
        runtime.optimizer.zero_grad(set_to_none=True)
        logits, labels = forward_batch(adapter, examples[:1])
        full_loss = per_sequence_nll(logits, labels)[0].mean()
        phase_result["autograd_full_loss"] = float(full_loss.detach())
        full_loss.backward()
        full_gradient = _gradient_snapshot(runtime)
        del logits, labels, full_loss
        runtime.optimizer.zero_grad(set_to_none=True)
        prepared = adapter.prepare(rows[0]["prompt"], backend.data_root)
        loss = -adapter.logprobs(prepared, list(examples[0].target_ids), require_grad=True).mean()
        phase_result["autograd_prefix_loss"] = float(loss.detach())
        loss.backward()
        phase_result["full_prefix_gradient"] = _gradient_comparison(
            full_gradient, _gradient_snapshot(runtime)
        )
        del loss, full_gradient
        runtime.optimizer.zero_grad(set_to_none=True)
        result["phases"][phase] = phase_result
        write_json(output, result)
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, _, _, frozen = _context(args.run)
    data = RoleDataset(run / "cohort", "gold")
    tasks = data.public("T_train")
    tasks = [
        task
        for family in ("cross_series", "trend")
        for task in sorted(
            [t for t in tasks if t["family"] == family], key=lambda t: t["task_id"]
        )[:4]
    ]
    audit = {row["task_id"]: row for row in data.audit("T_train")}
    view = build_training_view(
        tasks, gold_targets(tasks, [audit[t["task_id"]] for t in tasks]), source="gold"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "identity.json",
        {
            "run_identity": frozen["run_identity"],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "train_only": True,
            "fixed_bridge_tasks": [t["task_id"] for t in tasks],
        },
    )
    for parent in ("S96", "REP96"):
        backend = _backend(run, parent)
        diagnose(backend, view, output / (parent + ".json"))
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"status": "DIAGNOSIS_COMPLETE", "output": str(output)}))


if __name__ == "__main__":
    main()
