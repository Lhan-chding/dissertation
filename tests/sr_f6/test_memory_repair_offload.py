"""Storage-only CPU regressions, never actual-model/GPU numerical admission."""

import copy
import gc
from contextlib import nullcontext

import pytest
import torch

from sr_f6.objective import sequence_loss
from sr_f6_memory_repair.offload import saved_activation_offload


def model():
    torch.manual_seed(181)
    return torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 19))


def run(original, length, offload):
    net = copy.deepcopy(original)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    inputs = torch.linspace(-1.3, 1.1, length * 7).reshape(length, 7)
    targets = torch.arange(length) % 19
    context = (
        saved_activation_offload(net, max_host_bytes=32 * 1024 * 1024, fixture=True)
        if offload
        else nullcontext()
    )
    rng_before = torch.get_rng_state().clone()
    with context as manager:
        values = net(inputs).log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        sampler = values.detach() + 0.03
        reference = values.detach() - 0.02
        loss = sequence_loss(values, sampler, reference, 0.4, -0.2)
        diagnostics = []
        for channel in ("native", "aux", "kl"):
            diagnostics.append(
                tuple(
                    g.clone()
                    for g in torch.autograd.grad(
                        loss[channel] / 128, tuple(net.parameters()), retain_graph=True
                    )
                )
            )
        (loss["loss"] / 128).backward()
        gradients = tuple(p.grad.clone() for p in net.parameters())
        optimizer.step()
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        summary = manager.summary() if manager else None
        return (
            values.detach(),
            gradients,
            diagnostics,
            net.state_dict(),
            optimizer.state_dict(),
            summary,
        )


@pytest.mark.parametrize("length", [4, 1536])
def test_exact_values_diagnostic_gradients_adam_and_rng(length):
    net = model()
    expected = run(net, length, False)
    actual = run(net, length, True)
    torch.testing.assert_close(actual[:5], expected[:5], rtol=0, atol=0)
    summary = actual[5]
    assert summary["scope"] == "SYNTHETIC_CPU_ONLY"
    assert summary["resident_count"] > 0
    assert summary["peak_host_bytes"] > 0
    assert summary["unpack_count"] > summary["pack_count"]  # retained diagnostic graph
    assert not summary["numerical_pass"]


def test_repeated_rows_release_saved_host_tensors():
    net = model()
    with saved_activation_offload(net, max_host_bytes=100000, fixture=True) as manager:
        for _ in range(128):
            (net(torch.ones(4, 7)).square().mean() / 128).backward()
            gc.collect()
            assert manager.summary()["live_host_bytes"] == 0
        assert (
            manager.summary()["total_offloaded_bytes"] > 10 * manager.summary()["peak_host_bytes"]
        )
    assert manager.summary()["closed"]


def test_complete_128_row_accumulation_clip_adam_and_scheduler_are_exact():
    original = model()

    def execute(offload):
        net = copy.deepcopy(original)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        context = (
            saved_activation_offload(net, max_host_bytes=100000, fixture=True)
            if offload
            else nullcontext()
        )
        with context:
            for row in range(128):
                logits = net(torch.full((3, 7), (row + 1) / 128))
                current = logits.log_softmax(-1)[:, row % 19]
                loss = sequence_loss(
                    current,
                    current.detach() + 0.01,
                    current.detach() - 0.02,
                    (row % 3 - 1) * 0.3,
                    0.2,
                )
                (loss["loss"] / 128).backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            gradients = tuple(p.grad.clone() for p in net.parameters())
            optimizer.step()
            scheduler.step()
        return net.state_dict(), optimizer.state_dict(), scheduler.state_dict(), gradients, norm

    torch.testing.assert_close(execute(True), execute(False), rtol=0, atol=0)


def test_host_bound_stops_before_next_copy_and_does_not_update_parameters():
    net = model()
    before = copy.deepcopy(net.state_dict())
    with (
        pytest.raises(MemoryError, match="host-memory bound"),
        saved_activation_offload(net, max_host_bytes=16, fixture=True) as manager,
    ):
        net(torch.ones(4, 7)).square().sum().backward()
    torch.testing.assert_close(net.state_dict(), before, rtol=0, atol=0)
    assert manager.summary()["peak_host_bytes"] <= 16
    assert manager.summary()["closed"]


def test_lazy_provider_resolves_only_once():
    net = model()
    calls = []

    def provider():
        calls.append("actual-bound-model")
        return net

    with saved_activation_offload(provider, max_host_bytes=100000, fixture=True) as manager:
        assert not calls
        net(torch.ones(4, 7)).square().sum().backward()
        assert calls == ["actual-bound-model"]
        assert manager.summary()["model_bound"]


def test_resident_views_are_not_copied_and_mutation_is_rejected():
    net = model()
    with saved_activation_offload(net, max_host_bytes=100000, fixture=True) as manager:
        view = net[0].weight.T
        packed = manager._pack(view)
        assert manager.summary()["resident_count"] == 1
        assert manager.summary()["peak_host_bytes"] == 0
        with torch.no_grad():
            net[0].weight.add_(1)
        with pytest.raises(RuntimeError, match="resident tensor was mutated"):
            manager._unpack(packed)


def test_live_activation_mutation_is_rejected():
    net = model()
    with saved_activation_offload(net, max_host_bytes=100000, fixture=True) as manager:
        activation = torch.ones(4, 7, requires_grad=True) * 2
        packed = manager._pack(activation)
        with torch.no_grad():
            activation.add_(1)
        with pytest.raises(RuntimeError, match="activation was mutated"):
            manager._unpack(packed)


def test_context_restores_hooks_after_failure():
    net = model()
    with (
        pytest.raises(RuntimeError, match="injected"),
        saved_activation_offload(net, max_host_bytes=100000, fixture=True) as manager,
    ):
        raise RuntimeError("injected")
    assert manager.summary()["closed"]
    net(torch.ones(4, 7)).square().sum().backward()
    assert manager.summary()["pack_count"] == 0
    with pytest.raises(RuntimeError, match="cannot be reused"):
        manager.__enter__()


@pytest.mark.parametrize("bound", [0, -1, True, 1.5, None])
def test_explicit_positive_host_bound_required(bound):
    with pytest.raises(ValueError):
        saved_activation_offload(model(), max_host_bytes=bound)


def test_production_context_rejects_cpu_model():
    net = model()
    with (
        pytest.raises(PermissionError, match="all-CUDA"),
        saved_activation_offload(net, max_host_bytes=100000),
    ):
        net(torch.ones(4, 7)).square().sum().backward()


def test_non_fp32_model_rejected():
    net = model().double()
    with (
        pytest.raises(PermissionError, match="FP32"),
        saved_activation_offload(net, max_host_bytes=100000, fixture=True),
    ):
        net(torch.ones(4, 7, dtype=torch.float64)).square().sum().backward()


@pytest.mark.parametrize("mutation", ["replace", "same_storage", "dtype", "add_buffer", "storage"])
def test_summary_rejects_live_model_inventory_replacement(mutation):
    net = model()
    with saved_activation_offload(net, max_host_bytes=100000, fixture=True) as manager:
        net(torch.ones(4, 7)).square().sum().backward()
        if mutation == "replace":
            net[0].weight = torch.nn.Parameter(net[0].weight.detach().clone())
        elif mutation == "same_storage":
            net[0].weight = torch.nn.Parameter(net[0].weight.detach())
        elif mutation == "dtype":
            net[0].weight = torch.nn.Parameter(net[0].weight.detach().double())
        elif mutation == "add_buffer":
            net.register_buffer("unregistered", torch.zeros(1))
        else:
            net[0].weight.data = net[0].weight.detach().clone()
        with pytest.raises(RuntimeError, match="model inventory changed"):
            manager.summary()
