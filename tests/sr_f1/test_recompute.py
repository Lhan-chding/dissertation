"""CPU native hybrid Qwen checks; actual 9B/CUDA acceptance is separate."""

import contextlib
import gc
import importlib.util
import os
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as native

from sr_f1.recompute import KERNEL_NAMES, checkpoint_pure_delta_rule, gated_delta_recompute
from sr_f1.training import sequence_objective

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "srf1_recompute_activation_fixture", Path(__file__).with_name("test_activation_offload.py")
)
FIXTURE_MODULE = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURE_MODULE)


@pytest.fixture
def hybrid_runtime(tmp_path):
    yield from FIXTURE_MODULE.hybrid_runtime.__wrapped__(tmp_path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("coefficient", [0.0, -1.0, 1.0])
@pytest.mark.parametrize("length", [5, 33])
def test_native_hybrid_logits_both_gradients_and_rng_are_bitwise(
    hybrid_runtime, dtype, coefficient, length
):
    runtime, prepared = hybrid_runtime
    runtime.model.to(dtype=dtype)
    for p in runtime.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    tokens = [6 + i % 4 for i in range(length)]

    def measure(recompute):
        runtime.model.zero_grad(set_to_none=True)
        rng = torch.get_rng_state().clone()
        context = gated_delta_recompute(runtime.model) if recompute else contextlib.nullcontext()
        # Exit the wrapper context before both backwards: checkpoint closures
        # retain the verified native function independently of module attributes.
        with context as statistics:
            result = runtime.cached_training_forward(
                prepared, tokens, purpose="recompute_fixture", grad=True
            )
        objective = sequence_objective(
            result["logprobs"],
            result["logprobs"].detach(),
            result["logprobs"].detach() + 0.03,
            coefficient,
        )
        pg = torch.autograd.grad(objective["policy"] / 128, parameters, retain_graph=True)
        (objective["loss"] / 128).backward()
        gradients = [p.grad.clone() for p in parameters]
        assert torch.equal(torch.get_rng_state(), rng)
        values = result["logprobs"].detach().clone()
        saved_bytes = result["activation_offload"]["saved_activation_bytes"]
        del result, objective
        gc.collect()
        return values, pg, gradients, statistics, saved_bytes

    expected = measure(False)
    actual = measure(True)
    assert torch.equal(expected[0], actual[0])
    for left, right in zip(
        expected[1] + tuple(expected[2]), actual[1] + tuple(actual[2]), strict=True
    ):
        assert torch.equal(left, right)
        assert torch.isfinite(right).all()
    assert actual[3]["restored"]
    assert sum(actual[3]["recompute_calls"].values()) > 0
    assert actual[4] < expected[4]
    assert not list(runtime.output_root.rglob("*.bin"))


def test_configurable_long_native_history(hybrid_runtime):
    length = int(os.environ.get("SR_F1_RECOMPUTE_LONG_TOKENS", "65"))
    assert 33 <= length <= 768
    test_native_hybrid_logits_both_gradients_and_rng_are_bitwise(
        hybrid_runtime, torch.bfloat16, -1.0, length
    )


def test_last_token_preserves_prompt_and_full_recurrent_history(hybrid_runtime):
    runtime, prepared = hybrid_runtime
    embedding = runtime.model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]

    def measure(recompute):
        context = gated_delta_recompute(runtime.model) if recompute else contextlib.nullcontext()
        with context:
            result = runtime.cached_training_forward(
                prepared, [6, 7, 8, 9, 2], purpose="recompute_history", grad=True
            )
        return torch.autograd.grad(result["logprobs"][-1], parameters)

    expected, actual = measure(False), measure(True)
    for left, right in zip(expected, actual, strict=True):
        assert torch.equal(left, right)
    index = next(i for i, p in enumerate(parameters) if p is embedding)
    for token in (3, 90, 6):
        assert actual[index][token].norm() > 0


def test_restore_on_error_and_nested_failure_is_atomic(hybrid_runtime):
    runtime, _ = hybrid_runtime
    modules = [m for m in runtime.model.modules() if isinstance(m, native.Qwen3_5GatedDeltaNet)]
    original = [(m, n, getattr(m, n)) for m in modules for n in KERNEL_NAMES]
    with (
        pytest.raises(ValueError, match="intentional"),
        gated_delta_recompute(runtime.model) as statistics,
    ):
        wrapped = [(m, n, getattr(m, n)) for m, n, _ in original]
        with (
            pytest.raises(PermissionError, match="pure native"),
            gated_delta_recompute(runtime.model),
        ):
            pass
        assert all(getattr(m, n) is fn for m, n, fn in wrapped)
        raise ValueError("intentional")
    assert statistics["restored"]
    assert all(getattr(m, n) is fn for m, n, fn in original)


def test_unknown_kernel_and_training_mode_fail_closed(hybrid_runtime):
    runtime, _ = hybrid_runtime
    runtime.model.train()
    with pytest.raises(PermissionError, match="eval"), gated_delta_recompute(runtime.model):
        pass
    runtime.model.eval()
    with (
        pytest.raises(PermissionError, match="module count"),
        gated_delta_recompute(runtime.model, expected_linear_count=999),
    ):
        pass
    with pytest.raises(PermissionError, match="pure native"):
        checkpoint_pure_delta_rule(lambda *a: a, {}, "recurrent_gated_delta_rule")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pure_kernel_saved_bytes_reduce_and_no_grad_bypasses(dtype):
    statistics = {
        key: dict.fromkeys(KERNEL_NAMES, 0)
        for key in ("checkpoint_calls", "recompute_calls", "bypass_calls")
    }
    kernel = native.torch_recurrent_gated_delta_rule
    wrapped = checkpoint_pure_delta_rule(kernel, statistics, "recurrent_gated_delta_rule")
    torch.manual_seed(81)
    values = [torch.randn(1, 12, 2, 8, dtype=dtype, requires_grad=True) for _ in range(3)]
    g = (-torch.rand(1, 12, 2)).requires_grad_()
    beta = torch.rand(1, 12, 2, dtype=dtype, requires_grad=True)
    state = torch.randn(1, 2, 8, 8, requires_grad=True)
    kwargs = dict(initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)

    def measure(fn):
        sizes = []

        def pack(tensor):
            sizes.append(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            output, final = fn(*values, g, beta, **kwargs)
        grad = torch.autograd.grad(output.float().sum() + final.sum(), [*values, g, beta, state])
        return output, final, grad, sum(sizes)

    expected, actual = measure(kernel), measure(wrapped)
    assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
    assert all(torch.equal(a, b) for a, b in zip(expected[2], actual[2], strict=True))
    assert actual[3] < expected[3] / 2
    with torch.no_grad():
        left, right = kernel(*values, g, beta, **kwargs), wrapped(*values, g, beta, **kwargs)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    assert statistics["bypass_calls"]["recurrent_gated_delta_rule"] == 1
