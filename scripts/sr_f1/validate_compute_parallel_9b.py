#!/usr/bin/env python3
"""Bounded real-9B technical probe; never consumes TEST or qualifies full ENGINE.

Read four preserved ENGINE first-update answers, compare native serial auxiliary
storage against four recomputing replicas, and optionally teacher-force a labeled
synthetic 768-token extension. All receipts, gradients and accounting are PROBE
outputs; there is no production checkpoint, scheduler, or repair mutation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import stat
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_core.training import state_hash, trainable_state
from mm_dev.runtime import configure_audited_backend
from sr_f1.compute_workers import (
    ComputeParallelManager,
    ComputeReplica,
    decode_tensors,
    encode_tensors,
)
from sr_f1.contract import PACKAGE, digest
from sr_f1.recompute import POLICY
from sr_f1.runtime import (
    SRRuntime,
    atomic_json,
    bounded_path,
    configure_compute_execution,
    copy_parameters,
    file_hash,
    read_json,
    verified_adapter,
)
from sr_f1.training import configure_scientific_training

FREEZE_SHA256 = "c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8"
TEACHER_QOS = "soujanya-poria-startfund-2026-03"
NUMERICAL_FILES = (
    "src/sr_f1/recompute.py",
    "src/sr_f1/compute_parallel.py",
    "src/sr_f1/compute_workers.py",
    "src/sr_f1/runtime.py",
    "src/sr_f1/training.py",
)


def check(value, message):
    if not value:
        raise PermissionError(message)


def device_identity(index=0):
    import torch

    value = torch.cuda.get_device_properties(index)
    return dict(uuid=str(value.uuid), name=value.name, memory_bytes=value.total_memory)


def normalize_uuid(value):
    return value.lower().removeprefix("gpu-").replace("-", "")


def source_hashes():
    code = Path(__file__).resolve().parents[2]
    return {name: file_hash(code / name) for name in NUMERICAL_FILES}


def technical_policy():
    return dict(
        gpu_count=4,
        repair_sha256=digest(["PROBE_ONLY_NOT_A_PRODUCTION_REPAIR", source_hashes()]),
        recompute_policy=POLICY,
        activation_storage="LOCAL_GPU48G_CPU48G_EXACT_EXTERNAL_DISK",
        gpu_activation_budget_bytes=48 << 30,
        cpu_activation_budget_bytes=48 << 30,
    )


def private_spill_directory(quota, output, rank):
    """The parent creates one private scratch; children may only reuse that directory."""
    spill = quota / "louis-ssvc" / (output.name + "_activation_offload")
    check(spill.is_absolute() and spill.resolve() == spill, "Probe scratch must be canonical")
    check(not spill.is_symlink(), "Probe scratch must not be a symlink")
    if rank == 0:
        spill.mkdir(mode=0o700, parents=True, exist_ok=False)
    check(spill.is_dir(), "Parent has not created probe scratch")
    check(stat.S_IMODE(spill.stat().st_mode) == 0o700, "Probe scratch must be private mode0700")
    return spill


def verify_long_gradients(results):
    import torch

    check(len(results) == 4, "Long-token probe must return all four sequences")
    norms = []
    for result in results:
        check(result["token_count"] == 768, "Long-token probe was shortened")
        observed = {}
        for field in ("policy_gradients", "total_gradients"):
            values = result[field]
            check(
                values and all(bool(torch.isfinite(x).all()) for x in values),
                "Long-token probe has missing or nonfinite gradients",
            )
            norm2 = sum(float(x.double().square().sum()) for x in values)
            check(norm2 > 0, "Long-token probe has zero gradients")
            observed[field] = norm2**0.5
        norms.append(observed)
    return norms


def load_probe_runtime(config, rank, *, recompute):
    import torch
    from peft import PeftModel

    root, output = Path(config["root"]), Path(config["output"])
    if rank:
        check(not torch.cuda.is_initialized(), "Child initialized CUDA before isolation")
        os.environ["CUDA_VISIBLE_DEVICES"] = config["visibility"]
    torch.set_num_threads(4)
    configure_audited_backend()
    frozen = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    plan = config["plan"]
    check(frozen["model_revision"] == plan["model"]["revision"], "Model revision differs")
    check(
        frozen["model_weights_hash"] == plan["model"]["prior_composite_weight_hash"],
        "Model hash differs",
    )
    scratch = read_json(root / "ENGINE_IO_REPAIR.json")
    # Private derived scratch uses the same authenticated Ceph quota root and
    # production lossless byte format, but never the production scratch name.
    quota = Path(scratch["quota_root"])
    spill = private_spill_directory(quota, output, rank)
    ledger = output / f"PROBE_ACCOUNTING-rank{rank}.jsonl"

    def account(kind, count, metadata):
        with ledger.open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        kind=kind,
                        count=count,
                        metadata=metadata,
                        rank=rank,
                        pid=os.getpid(),
                        time_ns=time.time_ns(),
                    ),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    runtime = SRRuntime(
        frozen["model_path"],
        protocol_amendment=plan.get("protocol_amendment"),
        account=account,
        output_root=output,
        spill_directory=spill,
        quota_root=quota,
    )
    runtime.verify_identity(frozen)
    entry = verified_adapter(root, "SRF1_COMMON_START", step=0)
    runtime.model = PeftModel.from_pretrained(
        runtime.model, bounded_path(root, entry["adapter_path"]), is_trainable=False
    )
    runtime.identity.update(entry)
    runtime.training_learning_rate = plan["training"]["lr"]
    optimizer, scheduler, _ = configure_scientific_training(runtime)
    check(not optimizer.state, "Probe optimizer must start without moments")
    del optimizer, scheduler
    check(
        state_hash(trainable_state(runtime.model)) == entry["trainable_state_hash"],
        "Common LoRA differs",
    )
    if recompute:
        configure_compute_execution(runtime, technical_policy())
    else:
        runtime.activation_gpu_devices = (1, 2, 3)
        runtime.activation_gpu_budget_bytes = 80 << 30
        runtime.activation_cpu_budget_bytes = 48 << 30
    runtime._probe_logprobs = {}
    original_forward = runtime.sequence_forward

    def captured_forward(prepared, tokens, *, purpose, grad=False):
        result = original_forward(prepared, tokens, purpose=purpose, grad=grad)
        runtime._probe_logprobs[purpose] = result["logprobs"].detach().cpu().clone()
        return result

    runtime.sequence_forward = captured_forward
    actual = device_identity()
    if rank:
        check(
            normalize_uuid(actual["uuid"]) == normalize_uuid(config["visibility"]),
            "Child physical GPU differs",
        )
    runtime.identity["probe_hardware"] = actual
    runtime.identity.update(
        base_model_weights_hash=frozen["model_weights_hash"],
        compute_rank=rank,
        hardware=dict(
            cuda_uuid=actual["uuid"],
            cuda_visible_device_count=torch.cuda.device_count(),
            compute_devices=config["devices"],
        ),
    )
    return runtime


class RealProbeReplica(ComputeReplica):
    def __init__(self, runtime, rank):
        super().__init__(runtime, rank=rank)
        self.identity = dict(**self.identity, pid=os.getpid(), **device_identity())

    def handle(self, payload):
        import torch

        if payload["operation"] != "gradient":
            return super().handle(payload)
        self.runtime._probe_logprobs.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic_ns()
        result = decode_tensors(super().handle(payload))
        torch.cuda.synchronize()
        ended = time.monotonic_ns()
        result["probe_logprobs"] = dict(self.runtime._probe_logprobs)
        result["probe_execution"] = dict(
            **{key: value for key, value in self.identity.items() if key != "runtime"},
            started_ns=started,
            ended_ns=ended,
            wall_seconds=(ended - started) / 1e9,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            tokens=len(payload["example"]["record"]["tokens"]),
        )
        result["content_hash"] = state_hash(
            {k: v for k, v in result.items() if k != "content_hash"}
        )
        return encode_tensors(result)


def real_probe_factory(rank, config):
    return RealProbeReplica(load_probe_runtime(config, rank, recompute=True), rank)


def load_examples(root, raw_directory):
    check(raw_directory.resolve(strict=True) == raw_directory, "Raw directory must be canonical")
    check(raw_directory.is_relative_to(root), "Preserved raw directory must be inside run root")
    check(not raw_directory.is_symlink(), "Raw directory must not be a symlink")
    selected = sorted(raw_directory.glob("01-*.json"))[:4]
    check(len(selected) == 4, "Require four preserved first-update ENGINE responses")
    check(
        all(not path.is_symlink() and path.resolve().parent == raw_directory for path in selected),
        "Raw response leaves preserved directory",
    )
    records = [read_json(path) for path in selected]
    for record in records:
        check(
            record.get("logical_step") == 1 and record.get("qid", "").startswith("engine-"),
            "Only ENGINE step1 raw responses allowed",
        )
        check(
            record.get("generation_status") == "COMPLETE", "Raw response is technically incomplete"
        )
        check(
            record.get("record_hash")
            == digest({k: v for k, v in record.items() if k != "record_hash"}),
            "Raw response hash differs",
        )
        check(1 <= len(record["tokens"]) <= 768, "Saved token count outside frozen bounds")
        check(
            len(record["old_logprobs"]) == len(record["tokens"]), "Saved probability path shortened"
        )
    wanted = {record["qid"] for record in records}
    inputs = root / "manifests/MODEL_INPUTS.jsonl"
    check(
        file_hash(inputs) == file_hash(PACKAGE / "manifests/MODEL_INPUTS.jsonl"),
        "Frozen model input manifest differs",
    )
    rows = {}
    with inputs.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["qid"] in wanted:
                rows[row["qid"]] = row
                if set(rows) == wanted:
                    break
    check(set(rows) == wanted, "Missing ENGINE model inputs")
    examples = []
    for record in records:
        row = rows[record["qid"]]
        check(
            file_hash(bounded_path(root, row["image_file"])) == record["image_sha256"],
            "Saved raw image differs",
        )
        check(
            digest({k: row[k] for k in ("qid", "image_file", "text")}) == record["input_hash"],
            "Saved raw input differs",
        )
        examples.append(dict(record=record, row=row, root=str(root), advantage=1.0))
    return examples, [
        {"path": str(path.relative_to(root)), "sha256": file_hash(path)} for path in selected
    ]


def synchronize(replica, parameters, reference, step=1):
    binding = dict(
        step=step, policy_hash=state_hash(parameters), reference_hash=state_hash(reference)
    )
    replica.handle(
        dict(
            operation="synchronize",
            parameters=encode_tensors(parameters),
            reference=encode_tensors(reference),
            binding=binding,
        )
    )
    return binding


def numerical_result(result):
    keys = (
        "names",
        "policy_gradients",
        "total_gradients",
        "totals",
        "difference_sum",
        "difference_max",
        "differences",
        "token_count",
        "probability_summaries",
        "probe_logprobs",
    )
    return {key: result[key] for key in keys}


def probe_adam(runtime, results, parameters):
    import torch

    optimizer, scheduler, _ = configure_scientific_training(runtime)
    trainable = [p for p in runtime.model.parameters() if p.requires_grad]
    try:
        for result in results:
            for parameter, gradient in zip(trainable, result["total_gradients"], strict=True):
                gradient = gradient.to(parameter.device)
                if parameter.grad is None:
                    parameter.grad = gradient.clone()
                else:
                    parameter.grad.add_(gradient)
        before_clip = state_hash([p.grad for p in trainable])
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
        optimizer.step()
        scheduler.step()
        return dict(
            gradient_hash_before_clip=before_clip,
            norm_before_clip=norm,
            parameter_hash=state_hash(trainable_state(runtime.model)),
            optimizer_hash=state_hash(optimizer.state_dict()),
            scheduler_hash=state_hash(scheduler.state_dict()),
        )
    finally:
        copy_parameters(runtime.model, parameters)
        for parameter in trainable:
            parameter.grad = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-tokens", type=int, choices=[768])
    args = parser.parse_args()
    root, output = args.run_root.resolve(strict=True), args.output
    check(output.is_absolute() and output.resolve() == output, "Probe output must be canonical")
    check(
        output.is_relative_to(root / "technical_incidents") and output.name.startswith("PROBE"),
        "Probe output must be a new PROBE directory under technical_incidents",
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(
        status="RUNNING",
        scope="FOUR_SAVED_ENGINE_SEQUENCES_TECHNICAL_PROBE_ONLY",
        full_engine_accepted=False,
        scientific_training_started=False,
        started_at=datetime.now(UTC).isoformat(),
        source_hashes=source_hashes(),
    )
    manager = None
    try:
        import torch

        check(
            file_hash(root / "EXECUTION_FREEZE.json") == FREEZE_SHA256, "Execution freeze differs"
        )
        job = os.environ.get("SLURM_JOB_ID")
        check(bool(job), "Real 9B probe requires its own Slurm allocation")
        observed = subprocess.run(
            ["scontrol", "show", "job", job, "--oneliner"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", observed.stdout))
        check(
            fields.get("QOS") == TEACHER_QOS and fields.get("JobState") == "RUNNING",
            "Probe requires running teacher QoS job",
        )
        check(
            "gres/gpu=4" in fields.get("AllocTRES", "").split(","),
            "Probe requires exactly four allocated GPUs",
        )
        check(torch.cuda.device_count() == 4, "Probe must see exactly four GPUs")
        devices = [device_identity(index) for index in range(4)]
        check(
            len({normalize_uuid(device["uuid"]) for device in devices}) == 4,
            "Physical GPU UUIDs repeat",
        )
        check(
            all("PRO" in device["name"] and "6000" in device["name"] for device in devices),
            "Probe requires original PRO6000 GPU family",
        )
        report.update(slurm=fields, devices=devices)
        plan = read_json(args.plan)
        check(plan["training"]["lr"] == 1e-4, "Frozen learning rate differs")
        examples, raw_identity = load_examples(root, args.raw_directory)
        report["raw_identity"] = raw_identity
        config = dict(root=str(root), output=str(output), plan=plan, devices=devices)
        runtime = load_probe_runtime(config, 0, recompute=False)
        parameters = reference = trainable_state(runtime.model)
        check(
            all(example["record"]["policy_hash"] == state_hash(parameters) for example in examples),
            "Saved responses do not belong to common zero LoRA",
        )
        golden_replica = RealProbeReplica(runtime, 0)
        binding = synchronize(golden_replica, parameters, reference)
        golden = []
        for index, example in enumerate(examples):
            golden.append(
                decode_tensors(
                    golden_replica.handle(
                        dict(
                            operation="gradient",
                            binding=binding,
                            example=example,
                            sample_hash=digest(example),
                            example_index=index,
                            stress=False,
                        )
                    )
                )
            )
        with (output / "PROBE_NATIVE_SERIAL.pt").open("xb") as handle:
            handle.write(encode_tensors(golden))
        expected_adam = probe_adam(runtime, golden, parameters)
        del golden_replica
        gc.collect()
        for index in range(4):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        configure_compute_execution(runtime, technical_policy())
        configs = [{**config, "visibility": device["uuid"]} for device in devices[1:]]
        manager = ComputeParallelManager(
            runtime, configs, factory=real_probe_factory, ready_timeout=900, task_timeout=7200
        )
        manager.pool.local_handler = manager.local = RealProbeReplica(runtime, 0)
        manager.pool.worker_identities[0] = manager.local.identity
        actual = list(manager.iter_gradients(reference, examples))
        with (output / "PROBE_PARALLEL_RECOMPUTE.pt").open("xb") as handle:
            handle.write(encode_tensors(actual))
        matches = [
            state_hash(numerical_result(left)) == state_hash(numerical_result(right))
            for left, right in zip(golden, actual, strict=True)
        ]
        actual_adam = probe_adam(runtime, actual, parameters)
        executions = [value["probe_execution"] for value in actual]
        overlap = max(item["started_ns"] for item in executions) < min(
            item["ended_ns"] for item in executions
        )
        report.update(
            per_sequence_bitwise_equal=matches,
            numerical_equivalence=all(matches) and expected_adam == actual_adam,
            all_four_actual_compute=overlap and len({item["uuid"] for item in executions}) == 4,
            actual_compute_gpu_count=4,
            ordered_probe_adam_equal=expected_adam == actual_adam,
            golden_adam=expected_adam,
            parallel_adam=actual_adam,
            original_serial_seconds=sum(
                value["probe_execution"]["wall_seconds"] for value in golden
            ),
            parallel_execution=executions,
            four_gpu_execution_overlap=overlap,
            parallel_span_seconds=(
                max(item["ended_ns"] for item in executions)
                - min(item["started_ns"] for item in executions)
            )
            / 1e9,
        )
        check(all(matches), "Real9B native serial versus parallel numerical mismatch")
        check(expected_adam == actual_adam, "Ordered diagnostic Adam update differs")
        check(
            overlap and len({item["uuid"] for item in executions}) == 4,
            "Four GPUs did not execute gradients concurrently",
        )
        if args.long_tokens:
            import copy

            extended = copy.deepcopy(examples)
            for example in extended:
                record = example["record"]
                tokens = record["tokens"]
                record["tokens"] = (tokens * ((768 + len(tokens) - 1) // len(tokens)))[:768]
                # Synthetic old logp0 makes ratio<=1 for positive advantage;
                # the policy term cannot be flattened by the upper PPO clip.
                record["old_logprobs"] = [0.0] * 768
                record["probe_only_synthetic_extension"] = True
            long_results = list(manager.iter_gradients(reference, extended))
            gradient_norms = verify_long_gradients(long_results)
            report["synthetic_768_smoke"] = dict(
                status="PASS",
                scope="FIXED_SAVED_TOKEN_REPETITION_NOT_GENERATION_OR_SCIENTIFIC_DATA",
                sequences=4,
                gradient_norms=gradient_norms,
                executions=[item["probe_execution"] for item in long_results],
            )
            with (output / "PROBE_SYNTHETIC_768.pt").open("xb") as handle:
                handle.write(encode_tensors(long_results))
        manager.close()
        manager = None
        gc.collect()
        scratch_files = [
            dict(name=path.name, bytes=path.stat().st_size)
            for path in Path(runtime.activation_spill_directory).iterdir()
            if path.is_file()
        ]
        report["private_probe_scratch"] = dict(
            path=str(runtime.activation_spill_directory),
            remaining_files=scratch_files,
        )
        check(
            not scratch_files, "Probe activation scratch did not clean up or has failure receipts"
        )
        check(source_hashes() == report["source_hashes"], "Numerical source changed during probe")
        report.update(status="PASS", finished_at=datetime.now(UTC).isoformat())
    except BaseException as error:
        report.update(
            status="FAIL",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            finished_at=datetime.now(UTC).isoformat(),
        )
        raise
    finally:
        if manager is not None:
            manager.close(abort=True)
        atomic_json(output / "PROBE_9B.json", report, exclusive=True)
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
