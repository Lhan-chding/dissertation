#!/usr/bin/env python3
"""Synthetic native-Qwen CUDA compute parity; does not qualify the real 9B ENGINE."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
from validate_multigpu import bitwise_equal, check, tiny_runtime

from mm_core.training import capture_rng, state_hash, trainable_state
from mm_dev.runtime import F2Runtime, atomic_json, configure_audited_backend
from sr_f1.compute_workers import (
    ComputeParallelManager,
    ComputeReplica,
    decode_tensors,
    encode_tensors,
)
from sr_f1.recompute import gated_delta_recompute
from sr_f1.training import sequence_objective, update

NUMERICAL_FILES = (
    "src/sr_f1/recompute.py",
    "src/sr_f1/compute_parallel.py",
    "src/sr_f1/compute_workers.py",
    "src/sr_f1/runtime.py",
    "src/sr_f1/training.py",
)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in NUMERICAL_FILES
    }


def prepare_runtime(runtime, prepared, recompute):
    runtime.prepare = lambda _row, _root: prepared

    def sequence(prepared, tokens, *, purpose, grad=False):
        context = (
            gated_delta_recompute(runtime.model) if recompute and grad else contextlib.nullcontext()
        )
        with context:
            return F2Runtime.cached_training_forward(
                runtime, prepared, tokens, purpose=purpose, grad=grad
            )

    runtime.sequence_forward = sequence
    return runtime


def device_identity(device):
    if device == "cpu":
        return dict(device=device, uuid="CPU_TEST_ONLY", name="CPU_TEST_ONLY")
    properties = torch.cuda.get_device_properties(device)
    return dict(
        device=device,
        uuid=str(properties.uuid),
        name=properties.name,
        total_memory=properties.total_memory,
        capability=[properties.major, properties.minor],
    )


class ProbeReplica(ComputeReplica):
    def __init__(self, runtime, rank, context=None):
        super().__init__(runtime, rank=rank)
        self.context = context
        self.identity = dict(rank=rank, pid=os.getpid(), **device_identity(runtime.device))

    def handle(self, payload):
        gradient = payload["operation"] == "gradient"
        cuda = str(self.runtime.device).startswith("cuda")
        if gradient and cuda:
            torch.cuda.synchronize(self.runtime.device)
            torch.cuda.reset_peak_memory_stats(self.runtime.device)
            start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start_event.record()
        start = time.monotonic_ns()
        result = super().handle(payload)
        if gradient:
            result = decode_tensors(result)
            if cuda:
                end_event.record()
                torch.cuda.synchronize(self.runtime.device)
            end = time.monotonic_ns()
            result["probe_execution"] = dict(
                **self.identity,
                example_index=payload["example_index"],
                started_ns=start,
                ended_ns=end,
                wall_seconds=(end - start) / 1e9,
                cuda_ms=start_event.elapsed_time(end_event) if cuda else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.runtime.device)
                if cuda
                else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.runtime.device)
                if cuda
                else None,
            )
            result["content_hash"] = state_hash(
                {k: v for k, v in result.items() if k != "content_hash"}
            )
            return encode_tensors(result)
        return result

    def close(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)


def probe_factory(rank, config):
    torch.set_num_threads(1)
    configure_audited_backend()
    torch.cuda.set_device(rank)
    context = tiny_runtime(
        Path(config["directory"]), getattr(torch, config["dtype"]), f"cuda:{rank}"
    )
    runtime, prepared = context.__enter__()
    prepare_runtime(runtime, prepared, True)
    return ProbeReplica(runtime, rank, context)


class ProbeManager(ComputeParallelManager):
    def __init__(self, runtime, configs):
        super().__init__(
            runtime, configs, factory=probe_factory, ready_timeout=180, task_timeout=900
        )
        self.pool.local_handler = self.local = ProbeReplica(runtime, 0)
        self.pool.worker_identities[0] = self.local.identity
        self.records = []

    def iter_gradients(self, *args, **kwargs):
        for result in super().iter_gradients(*args, **kwargs):
            self.records.append(result["probe_execution"])
            yield result


def optimizer(runtime):
    opt = torch.optim.AdamW(
        [p for p in runtime.model.parameters() if p.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)


def examples(runtime, prepared, step):
    # Synthetic slots exercise zero/positive/negative policy terms and nonzero KL.
    result = []
    for index in range(128):
        tokens = [6 + index % 4, 9]
        old = runtime.sequence_forward(prepared, tokens, purpose="synthetic_saved_probability")[
            "logprobs"
        ]
        result.append(
            dict(
                record=dict(tokens=tokens, old_logprobs=old.tolist(), logical_step=step),
                row=dict(index=index),
                root=None,
                advantage=[0.0, 1.0, -1.0][index % 3],
            )
        )
    return result


def verify_worker_execution(records, gpu_count, *, require_cuda=True):
    check([r["example_index"] for r in records] == list(range(128)), "Result order differs")
    check(
        {r["rank"] for r in records} == set(range(gpu_count)),
        "Some allocated GPUs did no computation",
    )
    identities = {r["rank"]: r["uuid"] for r in records}
    check(all(r["uuid"] == identities[r["rank"]] for r in records), "Worker device changed")
    check(len(set(identities.values())) == gpu_count, "Compute workers share a physical device")
    if require_cuda:
        check(
            all(r["cuda_ms"] is not None and r["cuda_ms"] > 0 for r in records),
            "No actual CUDA elapsed time",
        )
        check(all(r["peak_allocated_bytes"] > 0 for r in records), "No actual CUDA allocation")
    overlaps = any(
        a["rank"] != b["rank"]
        and max(a["started_ns"], b["started_ns"]) < min(a["ended_ns"], b["ended_ns"])
        for a in records[:gpu_count]
        for b in records[:gpu_count]
    )
    if gpu_count > 1:
        check(overlaps, "GPU compute workers did not overlap")
    return dict(
        actual_compute_gpu_count=len(identities),
        concurrent_compute_observed=overlaps,
        device_uuids=identities,
        sequence_counts={str(r): sum(x["rank"] == r for x in records) for r in identities},
    )


def long_history(runtime, prepared, length):
    """Actual logits plus both backwards for a 768-token full-history trajectory."""
    tokens = [6 + i % 4 for i in range(length)]
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    outputs = []
    for recompute in (False, True):
        prepare_runtime(runtime, prepared, recompute)
        runtime.model.zero_grad(set_to_none=True)
        before = state_hash(capture_rng())
        values = runtime.sequence_forward(
            prepared, tokens, purpose="synthetic_long_history", grad=True
        )["logprobs"]
        objective = sequence_objective(values, values.detach(), values.detach() + 0.03, -1.0)
        pg = torch.autograd.grad(objective["policy"] / 128, parameters, retain_graph=True)
        (objective["loss"] / 128).backward()
        check(state_hash(capture_rng()) == before, "Long-history RNG changed")
        outputs.append(
            (
                values.detach().cpu(),
                [g.detach().cpu() for g in pg],
                [p.grad.detach().cpu().clone() for p in parameters],
            )
        )
        del objective, values, pg
    check(bitwise_equal(outputs[0][0], outputs[1][0]), "Long-history logits differ")
    for key in (1, 2):
        check(
            all(bitwise_equal(a, b) for a, b in zip(outputs[0][key], outputs[1][key], strict=True)),
            "Long-history double-backward differs",
        )
    # A last-token loss must reach prompt-only and earlier completion tokens.
    embedding = runtime.model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    gradients = []
    for recompute in (False, True):
        prepare_runtime(runtime, prepared, recompute)
        values = runtime.sequence_forward(
            prepared, [6, 7, 8, 9], purpose="synthetic_prompt_history", grad=True
        )["logprobs"]
        gradients.append(torch.autograd.grad(values[-1], embedding)[0])
    check(bitwise_equal(*gradients), "Prompt-history gradients differ")
    check(
        all(gradients[1][token].norm() > 0 for token in (3, 90, 6)),
        "Prompt/history gradient detached",
    )
    embedding.requires_grad_(False)
    return dict(
        tokens=length, logits_bitwise=True, both_gradients_bitwise=True, full_history_verified=True
    )


def run_case(directory, dtype, gpu_count, *, device="cuda:0", long_tokens=768):
    with (
        tiny_runtime(directory, dtype, device) as (serial, sp),
        tiny_runtime(directory, dtype, device) as (parallel, pp),
    ):
        prepare_runtime(serial, sp, False)
        prepare_runtime(parallel, pp, True)
        check(
            state_hash(serial.model.state_dict()) == state_hash(parallel.model.state_dict()),
            "Initial weights differ",
        )
        reference = {name: value + 0.001 for name, value in trainable_state(serial.model).items()}
        so, ss = optimizer(serial)
        po, ps = optimizer(parallel)
        configs = [
            dict(directory=str(directory), dtype=str(dtype).split(".")[-1])
            for _ in range(gpu_count - 1)
        ]
        manager = ProbeManager(parallel, configs)
        parallel.compute_parallel_manager = manager
        case = dict(dtype=str(dtype), updates=[], worker_identities=manager.worker_identities)
        try:
            for step in (1, 2):
                batch = examples(serial, sp, step)
                rng = state_hash(capture_rng())
                start = time.monotonic()
                expected = update(serial, so, ss, reference, batch, probability_gate=True)
                serial_seconds = time.monotonic() - start
                check(state_hash(capture_rng()) == rng, "Serial update RNG changed")
                offset = len(manager.records)
                start = time.monotonic()
                actual = update(parallel, po, ps, reference, batch, probability_gate=True)
                parallel_seconds = time.monotonic() - start
                check(state_hash(capture_rng()) == rng, "Parallel update RNG changed")
                check(actual == expected, "Update metrics or gradient norms differ")
                check(
                    state_hash(trainable_state(serial.model))
                    == state_hash(trainable_state(parallel.model)),
                    "Parameters differ",
                )
                check(state_hash(so.state_dict()) == state_hash(po.state_dict()), "Adam differs")
                check(
                    state_hash(ss.state_dict()) == state_hash(ps.state_dict()), "Scheduler differs"
                )
                for left, right in zip(
                    serial.model.parameters(), parallel.model.parameters(), strict=True
                ):
                    if left.requires_grad:
                        check(
                            bitwise_equal(left.grad, right.grad),
                            "Accumulated/clipped gradient differs",
                        )
                records = manager.records[offset:]
                execution = verify_worker_execution(
                    records, gpu_count, require_cuda=device != "cpu"
                )
                case["updates"].append(
                    dict(
                        step=step,
                        serial_seconds=serial_seconds,
                        parallel_seconds=parallel_seconds,
                        parameter_hash=state_hash(trainable_state(serial.model)),
                        adam_hash=state_hash(so.state_dict()),
                        exact_ordered_gradient_accumulation=True,
                        execution=execution,
                        sequence_execution=records,
                    )
                )
        finally:
            manager.close()
        case["long_history"] = long_history(serial, sp, long_tokens)
        return case


def run_validation(directory, gpu_count, report):
    check(torch.cuda.is_available(), "CUDA is unavailable")
    check(torch.cuda.device_count() == gpu_count, "Allocated GPU count differs")
    configure_audited_backend()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    report["cases"] = []
    for dtype in (torch.float32, torch.bfloat16):
        report["cases"].append(run_case(directory, dtype, gpu_count))
    report.update(
        status="PASS",
        numerical_equivalence=True,
        exact_ordered_gradient_accumulation=True,
        actual_compute_gpu_count=gpu_count,
        real_9b_engine_qualified=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-count", type=int, choices=range(2, 6), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Preserve existing compute probe receipt")
    report = dict(
        status="FAILED",
        started_utc=datetime.now(UTC).isoformat(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        scientific_data_read=False,
        real_9b_engine_qualified=False,
        numerical_equivalence=False,
        exact_ordered_gradient_accumulation=False,
        actual_compute_gpu_count=0,
        expected_gpus=args.gpu_count,
        source_file_hashes=source_hashes(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    directory = args.output.parent / (args.output.name + ".scratch")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        run_validation(directory, args.gpu_count, report)
        check(
            source_hashes() == report["source_file_hashes"], "Numerical source changed during probe"
        )
        code = 0
    except BaseException as error:
        report.update(
            status="FAILED",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            numerical_equivalence=False,
            exact_ordered_gradient_accumulation=False,
        )
        code = 1
    report["finished_utc"] = datetime.now(UTC).isoformat()
    atomic_json(args.output, report, exclusive=True)
    print(json.dumps(dict(status=report["status"], output=str(args.output))))
    return code


if __name__ == "__main__":
    sys.exit(main())
