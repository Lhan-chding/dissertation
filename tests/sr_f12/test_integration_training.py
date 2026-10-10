"""Native multimodal runtime plus SR-F1.2 update/checkpoint integration."""

import importlib.util
from pathlib import Path

import torch

from mm_core.training import state_hash, trainable_state
from sr_f12 import runner
from sr_f12.runtime import configure_training, enable_gradient_checkpointing
from sr_f12.training import checkpoint_state, commit_checkpoint, load_checkpoint, update

# Reuse the real tiny Qwen3.5 multimodal fixture, not a logits-only surrogate.
_spec = importlib.util.spec_from_file_location(
    "_sr_f12_native_test_fixture", Path(__file__).with_name("test_runtime.py")
)
_fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture_module)
native_runtime = _fixture_module.native_runtime


def test_native_runtime_complete_128_update_then_checkpoint_restore(native_runtime, tmp_path):
    runtime, prepared = native_runtime
    optimizer, scheduler, _ = configure_training(runtime)
    reference = trainable_state(runtime.model)
    token_rows = [[5, 6, 7], [9], [8, 12], [11, 15, 16, 17]]
    records = [
        dict(prepared=prepared, qid=f"q{i // 8}", tokens=token_rows[i % 4]) for i in range(128)
    ]
    sampler = runtime.batch_sequence_forward(records[:4], purpose="integration_sampler", grad=False)
    for i, record in enumerate(records):
        record["sampler_logprobs"] = sampler[i % 4].tolist()
    metrics = update(
        runtime,
        optimizer,
        scheduler,
        records,
        [(-1.0) ** i for i in range(128)],
        reference=reference,
        microbatch_size=4,
        save_train_logprobs=True,
    )
    assert metrics["parameters_changed"] and metrics["adam_nonzero_moments"]
    assert metrics["optimizer_updates"] == 1 and metrics["backward_calls"] == 32
    assert metrics["clip_fraction"] == 0
    assert metrics["sampler_train_mean_absolute_difference"] == 0
    assert len(metrics["train_logprobs"]) == 128
    kwargs = dict(
        run_identity={"model_id": "INTEGRATION_TECHNICAL_ONLY"},
        reference=reference,
        stream_hash="fixed128",
        sampling_hash="same-question",
        microbatch_size=4,
    )
    state = checkpoint_state(runtime, optimizer, scheduler, step=1, **kwargs)
    commit_checkpoint(tmp_path, state)
    for parameter in runtime.model.parameters():
        if parameter.requires_grad:
            parameter.data.add_(1)
    restored = load_checkpoint(
        tmp_path, runtime, optimizer, scheduler, learning_rate=5e-5, **kwargs
    )
    assert state_hash(trainable_state(runtime.model)) == state_hash(restored["parameters"])
    assert restored["cursor"]["next_logical_step"] == 2


def test_runner_checkpoint_mode_reaches_native_training_forward(native_runtime):
    runtime, prepared = native_runtime
    configure_training(runtime)
    case = dict(prepared=prepared, tokens=[5, 6, 7])
    # Actual helper certifies eval versus training mode before science is frozen.
    check = enable_gradient_checkpointing(runtime, [case])
    assert check["train_eval_bitwise_equal"]
    runner.enable_checkpointing(runtime, True)
    values = runtime.batch_sequence_forward([case], purpose="integration_checkpoint", grad=True)
    assert runtime.model.training
    assert getattr(runtime, "_sr_f12_checkpointing", False)
    assert runtime.model.is_gradient_checkpointing
    (-values[0].mean()).backward()
    assert any(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.any())
        for p in runtime.model.parameters()
        if p.requires_grad
    )
    runner.enable_checkpointing(runtime, False)
    runtime.batch_sequence_forward([case], purpose="integration_no_checkpoint", grad=True)
    assert not runtime.model.training
    assert not runtime.model.is_gradient_checkpointing


def test_technical_worker_is_present_at_derived_default_path():
    # Exercise the same source-layout resolution used by the process launcher.
    target = Path(runner.__file__).resolve().parents[2] / "scripts/sr_f12/run_worker.py"
    assert target.is_file()
