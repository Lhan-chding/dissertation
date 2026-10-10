"""Exact CPU/disk offload checks; actual 9B CUDA ENGINE qualification is separate."""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from peft import LoraConfig, get_peft_model

from mm_dev.runtime import F2Runtime
from sr_f1.runtime import SavedActivationOffload, SRRuntime
from sr_f1.training import sequence_objective


@pytest.fixture
def hybrid_runtime(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
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
    runtime.torch, runtime.device, runtime.image_calls = torch, "cpu", 0
    runtime.output_root = tmp_path
    runtime.model = get_peft_model(
        transformers.Qwen3_5ForConditionalGeneration(config),
        LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"]),
    ).eval()
    with torch.no_grad():
        for name, parameter in runtime.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.03)
    runtime.events = []
    runtime.account = lambda *args: runtime.events.append(args)
    hook = runtime._visual_module().register_forward_pre_hook(runtime._mark_vision)
    ids = torch.tensor([[3, 58, 60, 59, 4, 90]])
    prepared = dict(
        inputs=dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.full((4, 24), 0.5),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
        ),
        routing=dict(image_token_count=1),
    )
    yield runtime, prepared
    hook.remove()
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prompt_padding", [0, 64])
@pytest.mark.parametrize("cpu_budget", [48 << 30, 1024])
def test_offload_keeps_exact_logprobs_and_dual_backward_lora_gradients(
    hybrid_runtime, dtype, prompt_padding, cpu_budget
):
    runtime, prepared = hybrid_runtime
    runtime.activation_cpu_budget_bytes = cpu_budget
    runtime.model.to(dtype=dtype)
    if prompt_padding:
        inputs = prepared["inputs"]
        inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], torch.full((1, prompt_padding), 5)], dim=1
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros((1, prompt_padding), dtype=torch.long)],
            dim=1,
        )
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    original_values = [p.detach().clone() for p in parameters]
    tokens = [6, 7, 8, 9, 2]

    def measure(forward):
        runtime.model.zero_grad(set_to_none=True)
        result = forward(prepared, tokens, purpose="offload_cpu_fixture", grad=True)
        values = result["logprobs"]
        # Nonzero KL plus policy terms exercise the actual two-backward pattern.
        objective = sequence_objective(values, values.detach(), values.detach() + 0.03, -1.0)
        policy_grads = torch.autograd.grad(
            objective["policy"] / 128, parameters, retain_graph=True, allow_unused=False
        )
        if result.get("activation_offload", {}).get("spill_file_created"):
            assert Path(result["activation_offload"]["spill_path"]).is_file()
            assert not result["activation_offload"]["spill_file_cleaned"]
        (objective["loss"] / 128).backward()
        return result, policy_grads, [p.grad.detach().clone() for p in parameters]

    baseline, original_pg, original_grad = measure(
        lambda *args, **kwargs: F2Runtime.cached_training_forward(runtime, *args, **kwargs)
    )
    offloaded, copied_pg, copied_grad = measure(runtime.cached_training_forward)
    assert torch.equal(baseline["logprobs"], offloaded["logprobs"])
    assert offloaded["cached_model_forward_calls"] == len(tokens)
    assert offloaded["vision_forward_calls"] == 1
    for actual, expected, total, expected_total, parameter, original in zip(
        copied_pg, original_pg, copied_grad, original_grad, parameters, original_values, strict=True
    ):
        assert torch.equal(actual, expected)
        assert torch.equal(total, expected_total)
        assert torch.isfinite(total).all() and total.norm() > 0
        assert torch.equal(parameter, original)
    evidence = offloaded["activation_offload"]
    assert evidence["saved_activation_tensors"] > 0
    assert evidence["saved_activation_bytes"] > 0
    assert evidence["resident_parameter_views"] > 0
    assert evidence["unpacked_activation_tensors"] > evidence["saved_activation_tensors"]
    assert evidence["unpacked_activation_bytes"] > evidence["saved_activation_bytes"]
    assert evidence["cpu_storage_bytes"] <= cpu_budget
    if cpu_budget == 1024:
        assert evidence["disk_saved_tensors"] > 0 and evidence["disk_storage_bytes"] > 0
        assert evidence["disk_read_bytes"] > evidence["disk_storage_bytes"]
        assert evidence["spill_file_cleaned"] and not evidence["spill_aborted"]
        assert not Path(evidence["spill_path"]).exists()
        assert not list(runtime.output_root.rglob("*.bin"))
    rows = [args[2] for args in runtime.events if args[0] == "cached_training_model_forward_calls"]
    assert (
        rows[-1]["activation_offload"]["saved_activation_bytes"]
        == evidence["saved_activation_bytes"]
    )
    # Durable forward accounting is a snapshot, not mutated by later backward.
    assert rows[-1]["activation_offload"]["unpacked_activation_bytes"] == 0


def test_last_token_loss_retains_prompt_and_recurrent_history(hybrid_runtime):
    runtime, prepared = hybrid_runtime
    embedding = runtime.model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    tokens = [6, 7, 8, 9, 2]
    baseline = F2Runtime.cached_training_forward(
        runtime, prepared, tokens, purpose="history_baseline", grad=True
    )
    baseline_grad = torch.autograd.grad(baseline["logprobs"][-1], parameters)
    offloaded = runtime.cached_training_forward(
        prepared, tokens, purpose="history_offload", grad=True
    )
    actual_grad = torch.autograd.grad(offloaded["logprobs"][-1], parameters)
    for actual, expected in zip(actual_grad, baseline_grad, strict=True):
        assert torch.equal(actual, expected)
    embedding_index = next(i for i, parameter in enumerate(parameters) if parameter is embedding)
    # IDs 3 and 90 occur only in the prompt; 6 only in an earlier decoded step.
    for token in (3, 90, 6):
        assert actual_grad[embedding_index][token].norm() > 0


def test_parameter_views_stay_resident_and_mutation_is_rejected():
    model = torch.nn.Linear(3, 2, bias=False)
    offload = SavedActivationOffload(model)
    packed = offload.pack(model.weight.T)
    assert packed[0] == "parameter"
    assert packed[1].untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
    assert offload.statistics["saved_activation_bytes"] == 0
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(RuntimeError, match="parameter changed"):
        offload.unpack(packed)


def test_activation_snapshot_preserves_dtype_shape_stride_values_and_grad_edges():
    model = torch.nn.Linear(2, 2)
    source = torch.randn(3, 7, requires_grad=True)
    value = source.T
    offload = SavedActivationOffload(model)
    packed = offload.pack(value)
    restored = offload.unpack(packed)
    assert packed[0] == "activation"
    assert restored.data_ptr() != value.data_ptr()
    assert restored.dtype == value.dtype and restored.shape == value.shape
    assert restored.stride() == value.stride() and torch.equal(restored, value)
    with offload:
        recurrent = source.square()
        for _ in range(4):
            recurrent = (recurrent + source).tanh()
        loss = recurrent.sum()
    actual = torch.autograd.grad(loss, source)[0]
    expected_state = source.square()
    for _ in range(4):
        expected_state = (expected_state + source).tanh()
    expected = torch.autograd.grad(expected_state.sum(), source)[0]
    assert torch.equal(actual, expected)


def test_no_grad_reference_path_does_not_offload(hybrid_runtime):
    runtime, prepared = hybrid_runtime
    result = runtime.cached_training_forward(
        prepared, [6, 7, 2], purpose="reference_no_grad", grad=False
    )
    assert result["activation_offload"] is None
    assert not result["logprobs"].requires_grad
    assert result["entropy"] is not None


def test_partial_forward_failure_retains_offload_accounting(hybrid_runtime, monkeypatch):
    runtime, prepared = hybrid_runtime
    runtime.activation_cpu_budget_bytes = 0
    original = runtime.model.forward
    count = 0

    def failing_forward(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("simulated allocation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.model, "forward", failing_forward)
    with pytest.raises(RuntimeError, match="simulated allocation"):
        runtime.cached_training_forward(prepared, [6, 7, 2], purpose="failure_fixture", grad=True)
    final = runtime.events[-1][2]
    assert final["status"] == "TECHNICAL_FAILED"
    assert final["scored_completion_tokens"] == 1
    assert final["activation_offload"]["saved_activation_bytes"] > 0
    assert final["activation_offload"]["spill_file_cleaned"]
    assert final["activation_offload"]["spill_aborted"]
    assert not list(runtime.output_root.rglob("*.bin"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transpose", "sliced", "zero_stride", "overlap", "scalar"])
def test_disk_roundtrip_preserves_exact_layout_bits_and_last_reference_cleanup(
    tmp_path, monkeypatch, dtype, layout
):
    from sr_f1.runtime import _ActivationSpillFile

    monkeypatch.setattr(_ActivationSpillFile, "chunk_bytes", 7)
    monkeypatch.setattr(_ActivationSpillFile, "flush_bytes", 11)
    cache_advice = []
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: cache_advice.append(args), raising=False)
    source = torch.arange(35, dtype=dtype).reshape(5, 7)
    value = {
        "transpose": source.T,
        "sliced": source[::2, 1::3],
        "zero_stride": source[0].expand(9, 7),
        "overlap": source.flatten().unfold(0, 3, 1),
        "scalar": source[2, 3],
    }[layout]
    rng = torch.get_rng_state().clone()
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    with offload:
        packed = offload.pack(value)
    assert packed[0] == "disk_activation"
    path = Path(offload.statistics["spill_path"])
    assert path.is_file() and len(list(path.parent.glob("*.bin"))) == 1
    for _ in range(2):
        restored = offload.unpack(packed)
        assert torch.equal(restored, value)
        assert restored.shape == value.shape and restored.stride() == value.stride()
        assert restored.dtype == value.dtype and restored.device == value.device
    assert torch.equal(torch.get_rng_state(), rng)
    assert offload.statistics["cpu_storage_bytes"] == 0
    assert offload.statistics["disk_read_bytes"] == 2 * offload.statistics["disk_storage_bytes"]
    assert len(cache_advice) >= 3
    del packed
    gc.collect()
    assert not path.exists() and offload.statistics["spill_file_cleaned"]


def test_cpu_budget_counts_storage_span_including_strided_holes(tmp_path):
    source = torch.arange(32, dtype=torch.float32)
    value = source[::8]
    # Four logical floats reference a span of 25 floats, not 16 bytes.
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=32)
    with offload:
        packed = offload.pack(value)
    assert packed[0] == "disk_activation"
    assert offload.statistics["disk_storage_bytes"] == 100
    assert offload.statistics["cpu_storage_bytes"] == 0
    assert torch.equal(offload.unpack(packed), value)
    del packed
    assert offload.statistics["spill_file_cleaned"]


@pytest.fixture
def simulated_gpu_copies(monkeypatch):
    """CPU tests of tier accounting; real CUDA parity is checked separately."""
    original = torch.Tensor.to
    destinations = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    def copied(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if str(device).startswith("cuda:"):
            destinations.append(str(device))
            if args:
                args = ("cpu", *args[1:])
            else:
                kwargs["device"] = "cpu"
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copied)
    return destinations


def test_auxiliary_gpu_then_cpu_then_disk_budget_and_double_read(tmp_path, simulated_gpu_copies):
    offload = SavedActivationOffload(
        torch.nn.Identity(),
        output_root=tmp_path,
        gpu_devices=(1, 2),
        gpu_budget_bytes=16,
        cpu_budget_bytes=16,
    )
    value = torch.arange(4, dtype=torch.float32)
    with offload:
        records = [offload.pack(value) for _ in range(4)]
    assert [record[0] for record in records] == [
        "gpu_activation",
        "gpu_activation",
        "activation",
        "disk_activation",
    ]
    assert simulated_gpu_copies == ["cuda:1", "cuda:2"]
    for _ in range(2):
        for record in records:
            assert torch.equal(offload.unpack(record), value)
    del record
    stats = offload.statistics
    assert stats["gpu_storage_bytes"] == stats["gpu_read_bytes"] / 2 == 32
    assert stats["cpu_storage_bytes"] == stats["disk_storage_bytes"] == 16
    assert [s["live_bytes"] for s in stats["gpu_devices"].values()] == [16, 16]
    records.clear()
    gc.collect()
    assert stats["gpu_released_bytes"] == 32
    assert all(
        s["live_bytes"] == 0 and s["released_tensors"] == 1 for s in stats["gpu_devices"].values()
    )
    assert stats["spill_file_cleaned"]


def test_auxiliary_gpu_placement_uses_least_stored_bytes_deterministically(simulated_gpu_copies):
    offload = SavedActivationOffload(torch.nn.Identity(), gpu_devices=(1, 2))
    with offload:
        records = [offload.pack(torch.ones(size)) for size in (8, 4, 4, 8, 4)]
    # Ties preserve the authenticated device order; tensor count is not byte load.
    assert simulated_gpu_copies == ["cuda:1", "cuda:2", "cuda:2", "cuda:1", "cuda:2"]
    devices = offload.statistics["gpu_devices"]
    assert devices["1"]["storage_bytes"] == 64
    assert devices["2"]["storage_bytes"] == 48
    records.clear()
    assert all(device["live_bytes"] == 0 for device in devices.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transpose", "sliced", "zero_stride", "overlap", "scalar"])
def test_auxiliary_gpu_snapshot_preserves_exact_strided_bits(dtype, layout, simulated_gpu_copies):
    source = torch.arange(35, dtype=dtype).reshape(5, 7)
    value = {
        "transpose": source.T,
        "sliced": source[::2, 1::3],
        "zero_stride": source[0].expand(9, 7),
        "overlap": source.flatten().unfold(0, 3, 1),
        "scalar": source[2, 3],
    }[layout]
    rng = torch.get_rng_state().clone()
    offload = SavedActivationOffload(torch.nn.Identity(), gpu_devices=(1, 2))
    with offload:
        packed = offload.pack(value)
    assert packed[0] == "gpu_activation"
    for _ in range(2):
        result = offload.unpack(packed)
        assert result.dtype == value.dtype and result.device == value.device
        assert result.shape == value.shape and result.stride() == value.stride()
        assert torch.equal(result, value)
    assert torch.equal(torch.get_rng_state(), rng)
    del packed
    assert offload.statistics["gpu_devices"]["1"]["live_bytes"] == 0


def test_auxiliary_gpu_copy_failure_is_not_hidden(tmp_path, simulated_gpu_copies, monkeypatch):
    original = torch.Tensor.to
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, gpu_devices=(1, 2))

    def fail_gpu(tensor, *args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda:"):
            raise torch.cuda.OutOfMemoryError("simulated auxiliary GPU allocation failure")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_gpu)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="auxiliary GPU"), offload:
        offload.pack(torch.ones(8))
    assert offload.statistics["gpu_storage_bytes"] == 0
    assert offload.statistics["cpu_storage_bytes"] == 0
    assert not offload.statistics["spill_file_created"]


def test_auxiliary_gpu_backward_error_is_preserved_without_disk(
    tmp_path, simulated_gpu_copies, monkeypatch
):
    offload = SavedActivationOffload(
        torch.nn.Identity(), output_root=tmp_path, spill_directory=tmp_path, gpu_devices=(1, 2)
    )
    with offload:
        packed = offload.pack(torch.ones(8))

    def fail_copy(*args, **kwargs):
        raise RuntimeError("auxiliary return-copy failed")

    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    with pytest.raises(RuntimeError, match="return-copy failed"):
        offload.unpack(packed)
    receipts = list(tmp_path.glob("failure-*.json"))
    assert len(receipts) == 1
    failure = json.loads(receipts[0].read_text())
    assert failure["error"] == "auxiliary return-copy failed"
    assert failure["activation_offload"]["gpu_storage_bytes"] == 32
    assert failure["activation_offload"]["gpu_read_bytes"] == 0
    assert not failure["activation_offload"]["spill_file_created"]


def test_auxiliary_policy_retains_parameter_view_and_version_guard(simulated_gpu_copies):
    model = torch.nn.Linear(3, 2, bias=False)
    offload = SavedActivationOffload(model, gpu_devices=(1, 2))
    packed = offload.pack(model.weight.T)
    assert packed[0] == "parameter"
    assert packed[1].untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
    assert offload.statistics["gpu_storage_bytes"] == 0
    assert not simulated_gpu_copies
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(RuntimeError, match="parameter changed"):
        offload.unpack(packed)


@pytest.mark.parametrize("devices", [(0,), (2,), (1, 1), (1, 2, 3), (True, 2)])
def test_auxiliary_gpu_indices_fail_closed(devices, simulated_gpu_copies):
    with pytest.raises(PermissionError, match="allocated auxiliary"):
        SavedActivationOffload(torch.nn.Identity(), gpu_devices=devices)


def test_single_gpu_identity_keeps_legacy_guard(monkeypatch):
    from sr_f1 import runtime as module

    sentinel = object()
    monkeypatch.setattr(module, "actual_cuda_identity", lambda: sentinel)
    assert module.actual_srf1_cuda_identity() is sentinel


@pytest.fixture
def simulated_multigpu_identity(monkeypatch):
    from sr_f1 import runtime as module

    name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            name=name,
            uuid=f"GPU-{index + 1:032x}",
            total_memory=95 << 30,
            major=12,
            minor=0,
        ),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (94 << 30, 95 << 30))
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda source, target: True)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="\n".join(
                f"{name},GPU-{index + 1:032x},00000000:{index:02d}:00.0,590.00,97280"
                for index in range(4)
            )
        ),
    )
    return dict(
        gpu_count=4,
        compute_device_index=0,
        storage_device_indices=[1, 2, 3],
        gpu_activation_budget_bytes=80 << 30,
        cpu_activation_budget_bytes=48 << 30,
        activation_storage="AUXILIARY_GPU_THEN_BOUNDED_CPU_EXACT_EXTERNAL_DISK",
        repair_sha256="1" * 64,
    )


def test_authenticated_multigpu_identity_records_compute_storage_roles(simulated_multigpu_identity):
    from sr_f1.runtime import actual_srf1_cuda_identity

    result = actual_srf1_cuda_identity(simulated_multigpu_identity)
    assert result["cuda_visible_device_count"] == 4
    assert result["compute_device_index"] == 0
    assert result["cuda_uuid"] == "GPU-" + f"{1:032x}"
    assert [device["index"] for device in result["auxiliary_storage_devices"]] == [1, 2, 3]
    assert all(
        device["role"] == "saved_activation_storage"
        for device in result["auxiliary_storage_devices"]
    )
    assert result["peer_access_from_compute"] == [True] * 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("gpu_count", 3),
        ("gpu_count", True),
        ("gpu_count", 6),
        ("compute_device_index", 1),
        ("compute_device_index", False),
        ("storage_device_indices", [0, 1, 2]),
        ("storage_device_indices", [True, 2, 3]),
        ("gpu_activation_budget_bytes", 81 << 30),
        ("cpu_activation_budget_bytes", 64 << 30),
        ("activation_storage", "UNAUTHENTICATED"),
        ("repair_sha256", "bad"),
    ],
)
def test_multigpu_identity_rejects_changed_authorization(simulated_multigpu_identity, key, value):
    from sr_f1.runtime import actual_srf1_cuda_identity

    with pytest.raises(PermissionError, match="authenticated repair"):
        actual_srf1_cuda_identity({**simulated_multigpu_identity, key: value})


@pytest.mark.parametrize("problem", ["name", "uuid", "memory"])
def test_multigpu_identity_rejects_unavailable_or_wrong_hardware(
    simulated_multigpu_identity, monkeypatch, problem
):
    from sr_f1.runtime import actual_srf1_cuda_identity

    original = torch.cuda.get_device_properties
    if problem == "memory":
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (81 << 30, 95 << 30))
    else:

        def changed(index):
            properties = original(index)
            if index == 1:
                setattr(properties, problem, "WRONG")
            return properties

        monkeypatch.setattr(torch.cuda, "get_device_properties", changed)
    with pytest.raises(PermissionError):
        actual_srf1_cuda_identity(simulated_multigpu_identity)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires allocated real auxiliary CUDA GPU"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gpu_budget,cpu_budget", [(80 << 30, 48 << 30), (256, 512)])
def test_real_multigpu_exact_hybrid_double_backward_and_release(
    hybrid_runtime, dtype, gpu_budget, cpu_budget
):
    runtime, prepared = hybrid_runtime
    runtime.device = "cuda:0"
    runtime.model.to(device=runtime.device, dtype=dtype)
    prepared["inputs"] = {
        key: value.to(runtime.device) for key, value in prepared["inputs"].items()
    }
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    runtime.activation_gpu_devices = tuple(range(1, torch.cuda.device_count()))
    runtime.activation_gpu_budget_bytes = gpu_budget
    runtime.activation_cpu_budget_bytes = cpu_budget
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    def measure(forward):
        runtime.model.zero_grad(set_to_none=True)
        result = forward(prepared, [6, 7, 8, 9, 2], purpose="real_gpu_storage_fixture", grad=True)
        values = result["logprobs"]
        objective = sequence_objective(values, values.detach(), values.detach() + 0.03, -1.0)
        policy = torch.autograd.grad(objective["policy"] / 128, parameters, retain_graph=True)
        (objective["loss"] / 128).backward()
        return result, policy, [p.grad.clone() for p in parameters]

    baseline, baseline_policy, baseline_total = measure(
        lambda *args, **kwargs: F2Runtime.cached_training_forward(runtime, *args, **kwargs)
    )
    offloaded, policy, total = measure(runtime.cached_training_forward)
    assert torch.equal(baseline["logprobs"], offloaded["logprobs"])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(policy, baseline_policy, strict=True)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(total, baseline_total, strict=True)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), rng, strict=True)
    )
    stats = offloaded["activation_offload"]
    assert stats["gpu_storage_bytes"] > 0
    assert stats["gpu_read_bytes"] > stats["gpu_storage_bytes"]
    assert stats["gpu_released_bytes"] == stats["gpu_storage_bytes"]
    assert all(device["live_bytes"] == 0 for device in stats["gpu_devices"].values())
    if gpu_budget == 256:
        assert stats["cpu_storage_bytes"] > 0 and stats["disk_storage_bytes"] > 0
        assert stats["spill_file_cleaned"]


@pytest.mark.parametrize("failure", ["short_write", "disk_full", "short_read", "corrupt"])
def test_disk_errors_fail_closed_and_cleanup_only_private_spill(tmp_path, monkeypatch, failure):
    unrelated = tmp_path / "keep.bin"
    unrelated.write_bytes(b"keep")
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    value = torch.arange(16, dtype=torch.bfloat16)
    if failure in {"short_write", "disk_full"}:
        if failure == "short_write":
            monkeypatch.setattr(os, "write", lambda fd, value: len(value) - 1)
        else:

            def full(*args):
                raise OSError(28, "No space left on device")

            monkeypatch.setattr(os, "write", full)
        with pytest.raises(OSError), offload:
            offload.pack(value)
    else:
        with offload:
            packed = offload.pack(value)
        if failure == "short_read":
            os.ftruncate(offload.store.fd, 1)
        else:
            os.lseek(offload.store.fd, 0, os.SEEK_SET)
            os.write(offload.store.fd, b"\x01\x02")
        with pytest.raises(OSError, match=r"Short|checksum"):
            offload.unpack(packed)
    assert offload.statistics["spill_aborted"]
    assert offload.statistics["spill_file_cleaned"]
    assert not list((tmp_path / "technical_scratch/activation_offload").glob("*.bin"))
    assert unrelated.read_bytes() == b"keep"


def test_spill_requires_run_root_and_rejects_scratch_symlink(tmp_path):
    offload = SavedActivationOffload(torch.nn.Identity(), cpu_budget_bytes=0)
    with pytest.raises(PermissionError, match="output_root"), offload:
        offload.pack(torch.ones(2))
    (tmp_path / "outside").mkdir()
    (tmp_path / "technical_scratch").symlink_to(tmp_path / "outside", target_is_directory=True)
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    with pytest.raises(PermissionError, match="symlink"), offload:
        offload.pack(torch.ones(2))
    assert not list((tmp_path / "outside").iterdir())


def test_abandoned_graph_releases_its_only_spill_file(tmp_path):
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    value = torch.ones(8, requires_grad=True)
    with offload:
        output = value.square().tanh().sum()
    path = Path(offload.statistics["spill_path"])
    assert path.exists()
    del output
    gc.collect()
    assert not path.exists() and offload.statistics["spill_file_cleaned"]


def test_failed_restore_allocation_aborts_private_spill(tmp_path, monkeypatch):
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    with offload:
        packed = offload.pack(torch.ones(8))

    def exhausted(*args, **kwargs):
        raise RuntimeError("simulated restore allocation failure")

    monkeypatch.setattr(torch, "empty", exhausted)
    with pytest.raises(RuntimeError, match="restore allocation"):
        offload.unpack(packed)
    assert offload.statistics["spill_aborted"] and offload.statistics["spill_file_cleaned"]


def external_spill_fixture(tmp_path, monkeypatch, *, maximum=5 << 40, used=1 << 40):
    quota = tmp_path.resolve() / "quota"
    quota.mkdir()
    run = tmp_path.resolve() / "sr_f11_fixture"
    run.mkdir()
    directory = quota / "louis-ssvc" / (run.name + "_activation_offload")
    values = {"ceph.quota.max_bytes": str(maximum), "ceph.dir.rbytes": str(used)}
    monkeypatch.setattr(os, "getxattr", lambda path, key: values[key].encode(), raising=False)
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=195 << 40, f_frsize=1))
    return run, quota, directory, values


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_external_quota_spill_keeps_exact_dual_backward(tmp_path, monkeypatch, dtype):
    run, quota, directory, _ = external_spill_fixture(tmp_path, monkeypatch)
    value = torch.arange(16, dtype=dtype).div(20).requires_grad_()
    offload = SavedActivationOffload(
        torch.nn.Identity(),
        output_root=run,
        cpu_budget_bytes=0,
        spill_directory=directory,
        quota_root=quota,
    )
    with offload:
        actual = value.square().tanh().sum()
    expected = value.square().tanh().sum()
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual, value, retain_graph=True)[0],
        torch.autograd.grad(expected, value, retain_graph=True)[0],
    )
    assert list(directory.glob("*.bin"))
    assert torch.equal(
        torch.autograd.grad(actual, value)[0], torch.autograd.grad(expected, value)[0]
    )
    assert not list(directory.glob("*.bin"))
    assert offload.statistics["spill_file_cleaned"]
    observations = offload.statistics["capacity_observations"]
    assert observations and observations[0]["quota_max_bytes"] == 5 << 40
    assert observations[0]["safety_reserve_bytes"] == 10 << 30
    assert not (run / "technical_scratch").exists()


def test_ceph_quota_guard_rejects_before_file_creation_despite_global_free_space(
    tmp_path, monkeypatch
):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, _ = external_spill_fixture(
        tmp_path,
        monkeypatch,
        maximum=(10 << 30) + (16 << 20),
        used=0,
    )
    statistics = {}
    with pytest.raises(OSError, match="protected quota") as failure:
        _ActivationSpillFile(run, statistics, spill_directory=directory, quota_root=quota)
    assert failure.value.errno == 28
    assert not directory.exists()
    assert statistics["capacity_observations"][0]["statvfs_available_bytes"] == 195 << 40


def test_quota_windows_account_for_own_growth_when_ceph_usage_lags(tmp_path, monkeypatch):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, _ = external_spill_fixture(tmp_path, monkeypatch, maximum=128, used=80)
    monkeypatch.setattr(_ActivationSpillFile, "safety_reserve_bytes", 16)
    monkeypatch.setattr(_ActivationSpillFile, "flush_bytes", 16)
    monkeypatch.setattr(_ActivationSpillFile, "chunk_bytes", 8)
    offload = SavedActivationOffload(
        torch.nn.Identity(),
        output_root=run,
        cpu_budget_bytes=0,
        spill_directory=directory,
        quota_root=quota,
    )
    with pytest.raises(OSError, match="protected quota"), offload:
        offload.pack(torch.ones(16))
    assert offload.store.offset == 32
    assert offload.statistics["capacity_observations"][-1]["effective_quota_used_bytes"] == 112
    assert offload.statistics["spill_aborted"] and offload.statistics["spill_file_cleaned"]
    assert not list(directory.glob("*.bin"))


def test_statvfs_guard_also_applies_when_quota_has_space(tmp_path, monkeypatch):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, _ = external_spill_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(OSError, match="protected quota"):
        _ActivationSpillFile(run, {}, spill_directory=directory, quota_root=quota)
    assert not directory.exists()


def test_capacity_statistics_stay_bounded_across_one_thousand_checks(tmp_path, monkeypatch):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, values = external_spill_fixture(tmp_path, monkeypatch)
    statistics = {}
    store = _ActivationSpillFile(run, statistics, spill_directory=directory, quota_root=quota)
    try:
        for index in range(1000):
            values["ceph.dir.rbytes"] = str((3 if index == 499 else 2) << 40)
            store._check_capacity(store.flush_bytes)
            assert len(statistics["capacity_observations"]) <= 3
        assert statistics["capacity_check_count"] == 1001
        assert statistics["minimum_available_bytes"] == 2 << 40
        assert [row["check_index"] for row in statistics["capacity_observations"]] == [
            1,
            501,
            1001,
        ]
        assert statistics["capacity_observations"][-1]["effective_available_bytes"] == 3 << 40
    finally:
        store.finish_forward()


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_missing_or_invalid_ceph_quota_fails_closed(tmp_path, monkeypatch, value):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, values = external_spill_fixture(tmp_path, monkeypatch)
    values["ceph.quota.max_bytes"] = value
    with pytest.raises(PermissionError, match="quota"):
        _ActivationSpillFile(run, {}, spill_directory=directory, quota_root=quota)
    assert not directory.exists()


def test_unavailable_ceph_quota_fails_closed(tmp_path, monkeypatch):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, _ = external_spill_fixture(tmp_path, monkeypatch)

    def unavailable(*args):
        raise OSError(95, "xattrs unavailable")

    monkeypatch.setattr(os, "getxattr", unavailable)
    with pytest.raises(PermissionError, match="verify activation Ceph"):
        _ActivationSpillFile(run, {}, spill_directory=directory, quota_root=quota)
    assert not directory.exists()


@pytest.mark.parametrize("component", ["ancestor", "parent", "leaf"])
def test_external_spill_rejects_symlink_in_every_path_component(tmp_path, monkeypatch, component):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, _ = external_spill_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    if component == "ancestor":
        link = tmp_path / "linked-quota"
        link.symlink_to(quota, target_is_directory=True)
        quota, directory = link, link / "louis-ssvc" / directory.name
    elif component == "parent":
        directory.parent.symlink_to(outside, target_is_directory=True)
    else:
        directory.parent.mkdir()
        directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlink"):
        _ActivationSpillFile(run, {}, spill_directory=directory, quota_root=quota)
    assert not list(outside.iterdir())


def test_external_spill_requires_matching_run_leaf_and_private_permissions(tmp_path, monkeypatch):
    from sr_f1.runtime import _ActivationSpillFile

    run, quota, directory, _ = external_spill_fixture(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="current-run"):
        _ActivationSpillFile(run, {}, spill_directory=quota / "wrong", quota_root=quota)
    directory.mkdir(parents=True, mode=0o755)
    directory.chmod(0o755)
    with pytest.raises(PermissionError, match="private"):
        _ActivationSpillFile(run, {}, spill_directory=directory, quota_root=quota)
    assert not list(directory.iterdir())
    directory.chmod(0o700)
    monkeypatch.setattr(os, "geteuid", lambda: directory.stat().st_uid + 1)
    with pytest.raises(PermissionError, match="owned"):
        _ActivationSpillFile(run, {}, spill_directory=directory, quota_root=quota)
    assert not list(directory.iterdir())


def test_primary_disk_error_survives_cleanup_error(tmp_path, monkeypatch):
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    original_unlink = Path.unlink

    def write_failure(*args):
        raise OSError(28, "primary write failure")

    def unlink_failure(path, *args, **kwargs):
        if path.suffix == ".bin":
            raise OSError(5, "secondary unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "write", write_failure)
    monkeypatch.setattr(Path, "unlink", unlink_failure)
    with pytest.raises(OSError, match="primary write failure") as failure, offload:
        offload.pack(torch.ones(2))
    assert any("secondary unlink failure" in note for note in failure.value.__notes__)
    assert "secondary unlink failure" in offload.statistics["cleanup_error"]
    assert not offload.statistics["spill_file_cleaned"]


@pytest.mark.parametrize("unlink_fails", [False, True])
def test_delayed_close_error_still_attempts_private_spill_unlink(
    tmp_path, monkeypatch, unlink_fails
):
    offload = SavedActivationOffload(torch.nn.Identity(), output_root=tmp_path, cpu_budget_bytes=0)
    with offload:
        packed = offload.pack(torch.ones(2))
    path, descriptor = offload.store.path, offload.store.fd
    real_close, real_unlink = os.close, Path.unlink
    close_calls, unlink_calls = [], []

    def delayed_close(fd):
        close_calls.append(fd)
        real_close(fd)
        raise OSError(5, "delayed close writeback failure")

    def unlink(target, *args, **kwargs):
        unlink_calls.append(target)
        if unlink_fails:
            raise OSError(5, "independent unlink failure")
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "close", delayed_close)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="delayed close writeback failure") as failure:
        offload.store._close()
    assert close_calls == [descriptor] and unlink_calls == [path]
    assert offload.store.closed
    assert offload.statistics["spill_file_cleaned"] is not unlink_fails
    assert path.exists() is unlink_fails
    if unlink_fails:
        assert any("independent unlink failure" in note for note in failure.value.__notes__)
    offload.store._close()
    assert close_calls == [descriptor] and unlink_calls == [path]
    del packed


def test_primary_forward_error_survives_accounting_error(hybrid_runtime, monkeypatch):
    runtime, prepared = hybrid_runtime
    run, quota, directory, _ = external_spill_fixture(runtime.output_root, monkeypatch)
    directory.mkdir(parents=True, mode=0o700)
    runtime.output_root = run
    runtime.activation_spill_directory, runtime.activation_quota_root = directory, quota

    def forward_failure(*args, **kwargs):
        raise RuntimeError("primary native forward failure")

    def account(kind, count, metadata):
        if kind == "cached_training_model_forward_calls":
            raise OSError(28, "secondary accounting failure")

    runtime.account = account
    monkeypatch.setattr(runtime.model, "forward", forward_failure)
    with pytest.raises(RuntimeError, match="primary native forward failure") as failure:
        runtime.cached_training_forward(prepared, [6, 2], purpose="failure_fixture", grad=True)
    assert any("secondary accounting failure" in note for note in failure.value.__notes__)
    receipt = json.loads(next(directory.glob("failure-*.json")).read_text())
    assert receipt["error"] == "primary native forward failure"
    assert receipt["activation_offload"]["accounting_error"]["errno"] == 28
    assert set(receipt) == {
        "status",
        "pid",
        "attempt_id",
        "error_type",
        "error",
        "errno",
        "notes",
        "activation_offload",
    }
