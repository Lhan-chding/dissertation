"""Exact CPU/disk offload checks; actual 9B CUDA ENGINE qualification is separate."""

import gc
import os
from pathlib import Path

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
