#!/usr/bin/env python3
"""Synthetic CUDA qualification for auxiliary activation storage; not the 9B ENGINE."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, get_peft_model

from mm_dev.runtime import F2Runtime, atomic_json, configure_audited_backend
from sr_f1.runtime import SavedActivationOffload, SRRuntime
from sr_f1.training import sequence_objective

TOKENS = [6, 7, 8, 9, 2]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def bitwise_equal(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def rng_snapshot():
    return [torch.get_rng_state().clone(), *[x.clone() for x in torch.cuda.get_rng_state_all()]]


def check_rng(before):
    after = rng_snapshot()
    check(len(before) == len(after), "CUDA RNG device count changed")
    check(all(torch.equal(a, b) for a, b in zip(before, after, strict=True)), "RNG consumed")


def synchronize():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


@contextmanager
def tiny_runtime(directory, dtype, device="cuda:0"):
    """No pretrained checkpoint, dataset, or scientific output is opened."""
    torch.manual_seed(71010)
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            layer_types=[
                "full_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rope_parameters=dict(
                rope_type="default",
                mrope_section=[1, 1, 2],
                mrope_interleaved=True,
                partial_rotary_factor=1.0,
            ),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
    )
    config._attn_implementation = "eager"
    runtime = SRRuntime.__new__(SRRuntime)
    runtime.torch, runtime.device, runtime.image_calls = torch, device, 0
    runtime.output_root = directory
    runtime.model = get_peft_model(
        transformers.Qwen3_5ForConditionalGeneration(config),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"]),
    ).eval()
    with torch.no_grad():
        for name, parameter in runtime.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.03)
    runtime.model.to(device=device, dtype=dtype)
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    events = []
    runtime.events = events
    runtime.account = lambda *args: events.append(args)
    hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
    ids = torch.tensor([[3, 58, 60, 59, 4, 90]], device=device)
    prepared = dict(
        inputs=dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.full((4, 24), 0.5, device=device, dtype=dtype),
            image_grid_thw=torch.tensor([[1, 2, 2]], device=device),
            mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]], device=device),
        ),
        routing=dict(image_token_count=1),
    )
    try:
        yield runtime, prepared
    finally:
        hook.remove()
        runtime.model.zero_grad(set_to_none=True)
        del runtime, prepared
        gc.collect()


def measure(runtime, prepared, *, offloaded):
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    runtime.model.zero_grad(set_to_none=True)
    all_parameters = list(runtime.model.parameters())
    original = [p.detach().clone() for p in all_parameters]
    rng = rng_snapshot()
    synchronize()
    started = time.monotonic()
    forward = (
        runtime.cached_training_forward
        if offloaded
        else lambda *a, **k: F2Runtime.cached_training_forward(runtime, *a, **k)
    )
    result = forward(prepared, TOKENS, purpose="synthetic_multigpu_validation", grad=True)
    check(result["cached_model_forward_calls"] == len(TOKENS), "Cached prefix path was shortened")
    check(result["vision_forward_calls"] == 1, "Synthetic image did not get one visual forward")
    values = result["logprobs"]
    objective = sequence_objective(values, values.detach(), values.detach() + 0.03, -1.0)
    policy_grads = torch.autograd.grad(
        objective["policy"] / 128, parameters, retain_graph=True, allow_unused=False
    )
    statistics = result.get("activation_offload")
    after_policy = copy.deepcopy(statistics)
    if statistics and statistics["gpu_saved_tensors"]:
        check(
            any(d["live_bytes"] for d in statistics["gpu_devices"].values()),
            "First backward released the retained graph",
        )
    (objective["loss"] / 128).backward()
    synchronize()
    seconds = time.monotonic() - started
    check_rng(rng)
    gradients = [p.grad.detach().clone() for p in parameters]
    for parameter, previous in zip(all_parameters, original, strict=True):
        check(
            parameter.device == previous.device and bitwise_equal(parameter, previous),
            "Validation mutated or moved a model parameter",
        )
    for gradient in gradients:
        check(
            bool(torch.isfinite(gradient).all()) and gradient.norm().item() > 0,
            "Nonfinite or zero LoRA gradient",
        )
    values = values.detach().clone()
    # Keep only copied numerical results and the independent statistics dictionary.
    del objective, result
    gc.collect()
    if statistics:
        check(
            all(d["live_bytes"] == 0 for d in statistics["gpu_devices"].values()),
            "Auxiliary activations remain live after graph release",
        )
        check(
            statistics["gpu_released_bytes"] == statistics["gpu_storage_bytes"],
            "Auxiliary storage release accounting differs",
        )
        if statistics["spill_file_created"]:
            check(
                statistics["spill_file_cleaned"] and not statistics["spill_aborted"],
                "Derived disk activation cleanup failed",
            )
            check(not Path(statistics["spill_path"]).exists(), "Derived spill file remains")
    return dict(
        values=values,
        policy_grads=policy_grads,
        gradients=gradients,
        seconds=seconds,
        offload=copy.deepcopy(statistics),
        after_policy=after_policy,
    )


def compare_measurements(baseline, candidate):
    check(bitwise_equal(baseline["values"], candidate["values"]), "Token logprobs differ")
    for key in ("policy_grads", "gradients"):
        check(len(baseline[key]) == len(candidate[key]), "Parameter count differs")
        for index, (expected, actual) in enumerate(zip(baseline[key], candidate[key], strict=True)):
            check(bitwise_equal(expected, actual), f"{key}[{index}] differs")


def verify_history(runtime, prepared):
    embedding = runtime.model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    rng = rng_snapshot()
    baseline = F2Runtime.cached_training_forward(
        runtime, prepared, TOKENS, purpose="synthetic_history_baseline", grad=True
    )
    expected = torch.autograd.grad(baseline["logprobs"][-1], parameters)
    candidate = runtime.cached_training_forward(
        prepared, TOKENS, purpose="synthetic_history_auxgpu", grad=True
    )
    actual = torch.autograd.grad(candidate["logprobs"][-1], parameters)
    for a, b in zip(actual, expected, strict=True):
        check(bitwise_equal(a, b), "Prompt/recurrent history gradient differs")
    index = next(i for i, parameter in enumerate(parameters) if parameter is embedding)
    norms = {str(token): actual[index][token].norm().item() for token in (3, 90, 6)}
    check(all(value > 0 for value in norms.values()), "Earlier prompt/token gradient was detached")
    check_rng(rng)
    embedding.requires_grad_(False)
    del baseline, candidate, expected, actual
    gc.collect()
    return norms


def verify_layout_and_tiers(directory, auxiliary_devices):
    """A bounded 256-byte budget makes each route observable without large allocations."""
    model = torch.nn.Linear(4, 4, bias=False, device="cuda:0")
    rng = rng_snapshot()
    offload = SavedActivationOffload(
        model,
        output_root=directory,
        gpu_devices=auxiliary_devices,
        gpu_budget_bytes=256,
        cpu_budget_bytes=256,
    )
    source = torch.arange(64, dtype=torch.float32, device="cuda:0")
    with offload:
        packed = [offload.pack(source) for _ in range(len(auxiliary_devices) + 2)]
    kinds = [item[0] for item in packed]
    check(
        kinds == ["gpu_activation"] * len(auxiliary_devices) + ["activation", "disk_activation"],
        "Full GPU budgets did not fall back through CPU and disk",
    )
    for item in packed:
        for _ in range(2):
            check(bitwise_equal(source, offload.unpack(item)), "Tier roundtrip changes values")
    check(
        all(d["storage_bytes"] == 256 for d in offload.statistics["gpu_devices"].values()),
        "Not all allocated auxiliary GPUs were used",
    )
    check(
        all(d["live_bytes"] == 256 for d in offload.statistics["gpu_devices"].values()),
        "Unpack prematurely released a retained snapshot",
    )
    del item, packed
    gc.collect()
    check(offload.statistics["spill_file_cleaned"], "Tier fixture spill was not removed")
    check(
        all(d["live_bytes"] == 0 for d in offload.statistics["gpu_devices"].values()),
        "Tier fixture auxiliary bytes remain live",
    )
    tiers = copy.deepcopy(offload.statistics)
    layout_records = []
    for dtype in (torch.float32, torch.bfloat16):
        base = torch.arange(48, device="cuda:0", dtype=dtype).reshape(6, 8)
        views = {
            "transpose": base.T,
            "holes": base[::2, 1::2],
            "zero_stride": base[0, :1].expand(3, 4),
            "overlap": torch.as_strided(base, (3, 4), (1, 1), 2),
            "empty": base[:0],
        }
        for name, value in views.items():
            layout_offload = SavedActivationOffload(
                model, output_root=directory, gpu_devices=auxiliary_devices
            )
            item = layout_offload.pack(value)
            restored = layout_offload.unpack(item)
            check(
                restored.dtype == value.dtype and restored.shape == value.shape,
                "Layout dtype or shape changed",
            )
            check(
                restored.stride() == value.stride() and bitwise_equal(restored, value),
                "Layout stride or content changed",
            )
            elements = (
                0
                if not value.numel()
                else 1
                + sum(
                    (dimension - 1) * step
                    for dimension, step in zip(value.shape, value.stride(), strict=True)
                )
            )
            expected_storage = torch.as_strided(value, (elements,), (1,), value.storage_offset())
            restored_storage = torch.as_strided(
                restored, (elements,), (1,), restored.storage_offset()
            )
            check(
                bitwise_equal(expected_storage, restored_storage), "Physical storage span changed"
            )
            if value.numel():
                check(restored.data_ptr() != value.data_ptr(), "Activation snapshot aliases source")
            layout_records.append(
                dict(
                    dtype=str(dtype),
                    layout=name,
                    shape=list(value.shape),
                    stride=list(value.stride()),
                    route=item[0],
                )
            )
            del item, restored, layout_offload
    guard = SavedActivationOffload(model, gpu_devices=auxiliary_devices)
    packed_parameter = guard.pack(model.weight.T)
    check(packed_parameter[0] == "parameter", "Parameter view moved to storage GPU")
    with torch.no_grad():
        model.weight.add_(1)
    try:
        guard.unpack(packed_parameter)
    except RuntimeError as error:
        check("parameter changed" in str(error), "Unexpected parameter guard exception")
    else:
        raise AssertionError("Parameter mutation bypassed version guard")
    check_rng(rng)
    return dict(
        tier_statistics=tiers,
        layouts=layout_records,
        parameter_version_guard=True,
        rng_unchanged=True,
    )


def run_validation(spill_directory, expected_gpus, report=None):
    check(torch.cuda.is_available(), "CUDA is unavailable; CPU is not multi-GPU qualification")
    check(torch.cuda.device_count() == expected_gpus, "Allocated CUDA device count differs")
    check(bool(os.environ.get("SLURM_JOB_ID")), "Real CUDA validation requires a Slurm allocation")
    determinism = configure_audited_backend()
    identities = []
    for index in range(expected_gpus):
        properties = torch.cuda.get_device_properties(index)
        check("PRO 6000" in properties.name.upper(), "Validation GPU must be PRO 6000")
        identities.append(
            dict(
                index=index,
                name=properties.name,
                total_memory_bytes=properties.total_memory,
                compute_capability=list(torch.cuda.get_device_capability(index)),
                uuid=str(getattr(properties, "uuid", "UNAVAILABLE")),
            )
        )
    auxiliary_devices = tuple(range(1, expected_gpus))
    if report is None:
        report = {}
    report.update(
        status="RUNNING",
        qualification="SYNTHETIC_CUDA_ACTIVATION_STORAGE_ONLY",
        real_9b_engine_qualified=False,
        scientific_data_read=False,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        determinism=determinism,
        gpu_identities=identities,
        peer_access=[
            [
                bool(torch.cuda.can_device_access_peer(a, b)) if a != b else True
                for b in range(expected_gpus)
            ]
            for a in range(expected_gpus)
        ],
        cases=[],
    )
    report["active_case"] = "layout_and_tiers"
    report["layout_and_tiers"] = verify_layout_and_tiers(spill_directory, auxiliary_devices)
    for dtype in (torch.float32, torch.bfloat16):
        with tiny_runtime(spill_directory, dtype) as (runtime, prepared):
            runtime.activation_gpu_devices = auxiliary_devices
            baseline = measure(runtime, prepared, offloaded=False)
            for label, gpu_budget, cpu_budget in (
                ("auxiliary_storage", 80 << 30, 48 << 30),
                ("bounded_tier_fallback", 64 << 10, 32 << 10),
            ):
                report["active_case"] = dict(dtype=str(dtype), case=label)
                runtime.activation_gpu_budget_bytes = gpu_budget
                runtime.activation_cpu_budget_bytes = cpu_budget
                candidate = measure(runtime, prepared, offloaded=True)
                compare_measurements(baseline, candidate)
                stats = candidate["offload"]
                check(stats["gpu_saved_tensors"] > 0, "No synthetic activation used auxiliary GPU")
                if label == "bounded_tier_fallback":
                    check(
                        all(d["saved_tensors"] > 0 for d in stats["gpu_devices"].values()),
                        "Hybrid model did not exercise all auxiliary GPUs",
                    )
                    check(
                        stats["disk_saved_tensors"] > 0,
                        "Hybrid model did not exercise disk fallback",
                    )
                report["cases"].append(
                    dict(
                        dtype=str(dtype),
                        case=label,
                        logprobs_exact=True,
                        policy_gradients_exact=True,
                        total_gradients_exact=True,
                        parameter_values_unchanged=True,
                        rng_unchanged=True,
                        baseline_seconds=baseline["seconds"],
                        candidate_seconds=candidate["seconds"],
                        after_first_backward=candidate["after_policy"],
                        activation_offload=stats,
                    )
                )
                del candidate
            runtime.activation_gpu_budget_bytes = 80 << 30
            runtime.activation_cpu_budget_bytes = 48 << 30
            report["active_case"] = dict(dtype=str(dtype), case="full_prompt_and_recurrent_history")
            report["cases"].append(
                dict(
                    dtype=str(dtype),
                    case="full_prompt_and_recurrent_history",
                    gradient_norms=verify_history(runtime, prepared),
                )
            )
            del baseline
    gc.collect()
    synchronize()
    check(not list(spill_directory.rglob("*.bin")), "Synthetic derived spill files remain")
    report.update(status="PASS", cleanup_verified=True, active_case=None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spill-directory", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, choices=range(3, 6), default=3)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Validation receipt already exists; preserve the previous result")
    directory = args.spill_directory
    if not directory.is_absolute() or directory.resolve() != directory or ".." in directory.parts:
        raise PermissionError("Validation scratch must be an absolute canonical directory")
    # A distinct empty root prevents this check from touching live training scratch.
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(
        status="FAILED",
        started_utc=datetime.now(UTC).isoformat(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        expected_gpus=args.expected_gpus,
        spill_directory=str(directory),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification="SYNTHETIC_CUDA_ACTIVATION_STORAGE_ONLY",
        real_9b_engine_qualified=False,
        scientific_data_read=False,
    )
    source_root = Path(__file__).resolve().parents[2]
    report["source_hashes"] = {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in (
            "scripts/sr_f1/validate_multigpu.py",
            "src/sr_f1/runtime.py",
            "src/sr_f1/training.py",
            "src/mm_dev/runtime.py",
            "src/mm_core/training.py",
        )
    }
    try:
        report.update(run_validation(directory, args.expected_gpus, report))
        exit_code = 0
    except BaseException as error:
        report.update(
            status="FAILED",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        exit_code = 1
    report["finished_utc"] = datetime.now(UTC).isoformat()
    atomic_json(args.output, report, exclusive=True)
    print(json.dumps(dict(status=report["status"], output=str(args.output)), sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
