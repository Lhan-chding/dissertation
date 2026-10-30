"""Candidate device residency: payload bounds, alias lifetime and exact compute."""

import copy
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier

import pytest
import torch

from sr_f6.objective import sequence_loss
from sr_f6_memory_repair.offload import saved_activation_offload
from sr_f6_memory_repair.residency import DeviceSaved


def model():
    return torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 19))


def manager(net=None, **kwargs):
    return saved_activation_offload(
        net if net is not None else model(),
        fixture=True,
        max_host_bytes=kwargs.pop("max_host_bytes", 2**20),
        **kwargs,
    )


@pytest.mark.parametrize("value", [-1, True, 1.5, None, "32GiB"])
def test_invalid_budget_rejected(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        manager(max_device_bytes=value)


def test_zero_default_retains_legacy_behavior_and_summary():
    source = torch.arange(8.0)
    for kwargs in ({}, {"max_device_bytes": 0}):
        with manager(**kwargs) as m:
            saved = m._pack(source)
            assert not isinstance(saved, DeviceSaved)
            assert m.live_host_bytes == 32
            assert m._unpack(saved).data_ptr() != source.data_ptr()
            assert not any(k.startswith("device_") or "device_bytes" in k for k in m.summary())


def test_full_backing_storage_not_tiny_logical_view_is_charged():
    base = torch.arange(1024.0)
    view = base[17:18]
    with manager(max_device_bytes=4095) as m:
        saved = m._pack(view)
        assert not isinstance(saved, DeviceSaved)
        assert m.summary()["live_device_bytes"] == 0
        assert m.summary()["device_spill_count"] == 1
        assert m.live_host_bytes == 4
    with manager(max_device_bytes=4096) as m:
        saved = m._pack(view)
        assert isinstance(saved, DeviceSaved)
        assert m.summary()["live_device_bytes"] == 4096
        assert m.live_host_bytes == 0
        out = m._unpack(saved)
        assert out.storage_offset() == 17
        assert out.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def test_aliases_keep_exact_views_and_charge_once_until_last_actual_view_dies():
    base = torch.arange(35.0).reshape(5, 7)
    with manager(max_device_bytes=140) as m:
        first = m._pack(base.t())
        second = m._pack(base[2:, ::2])
        third = m._pack(base[0:1, :].expand(5, 7))
        assert m.summary()["live_device_bytes"] == 140
        assert m.summary()["device_storage_count"] == 1
        assert m.summary()["device_alias_pack_count"] == 2
        for saved, expected in (
            (first, base.t()),
            (second, base[2:, ::2]),
            (third, base[0:1].expand(5, 7)),
        ):
            restored = m._unpack(saved)
            assert restored.dtype == expected.dtype
            assert restored.shape == expected.shape
            assert restored.stride() == expected.stride()
            assert restored.storage_offset() == expected.storage_offset()
            assert restored.data_ptr() == expected.data_ptr()
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        del saved, restored
        held = m._unpack(first)
        del first, second, third
        gc.collect()
        assert m.summary()["live_device_bytes"] == 140
        del held
        gc.collect()
        assert m.summary()["live_device_bytes"] == 140
        del base, expected
        gc.collect()
        assert m.summary()["live_device_bytes"] == 0
        assert not m.device_residency.storages


def test_model_parameters_and_buffers_excluded_from_activation_budget():
    net = model()
    net.register_buffer("native_buffer", torch.ones(8))
    with manager(net, max_device_bytes=1) as m:
        saved = [m._pack(value) for value in (*net.parameters(), net.native_buffer)]
        assert m.summary()["device_resident_pack_count"] == 0
        assert m.summary()["live_device_bytes"] == 0
        assert m.resident_count == len(saved)
        assert m.summary()["device_spill_count"] == 0


def test_budget_overflow_uses_unchanged_cpu_then_arena(tmp_path):
    with manager(
        max_device_bytes=32,
        max_host_bytes=32,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=1024,
        disk_backend="arena_v3",
    ) as m:
        source = torch.arange(8.0)
        device = m._pack(source)
        alias = m._pack(source[1:])
        host = m._pack(source.clone())
        disk = m._pack(source.clone())
        assert isinstance(device, DeviceSaved) and isinstance(alias, DeviceSaved)
        assert m.summary()["live_device_bytes"] == m.live_host_bytes == m.live_disk_bytes == 32
        assert m.summary()["device_spill_count"] == 2
        for saved in (device, host, disk):
            torch.testing.assert_close(m._unpack(saved), source, rtol=0, atol=0)
        del saved, device, alias, host, disk, source
        gc.collect()
        assert m.summary()["live_device_bytes"] == m.live_host_bytes == m.live_disk_bytes == 0
        assert m.summary()["arena_physical_bytes"] == 0


@pytest.mark.parametrize("mutation", ["source", "alias", "returned_metadata"])
def test_version_and_view_mutations_fail_closed(mutation):
    with manager(max_device_bytes=32) as m:
        source = torch.arange(8.0)
        saved = m._pack(source)
        if mutation == "source":
            source.add_(1)
        elif mutation == "alias":
            source[2:].mul_(2)
        else:
            m._unpack(saved).resize_(2, 4)
        with pytest.raises(RuntimeError, match="mutated"):
            m._unpack(saved)


def test_dead_source_keeps_version_guard_and_releases_without_cache_cycle():
    with manager(max_device_bytes=32) as m:
        source = torch.arange(8.0)
        reference = weakref.ref(source)
        saved = m._pack(source)
        del source
        assert reference() is None
        m._unpack(saved).add_(1)
        with pytest.raises(RuntimeError, match="mutated"):
            m._unpack(saved)
        del saved
        gc.collect()
        assert m.summary()["live_device_bytes"] == 0
        assert not m.device_residency.storages


@pytest.mark.parametrize("derive", [lambda x: x.detach(), lambda x: x[1:], lambda x: x.view(2, 4)])
def test_derived_alias_keeps_charge_after_original_python_tensor_dies(derive):
    with manager(max_device_bytes=32) as m:
        saved = m._pack(torch.arange(8.0))
        alias = derive(m._unpack(saved))
        reference = weakref.ref(saved.tensor)
        del saved
        gc.collect()
        assert m.summary()["live_device_bytes"] == 32
        assert not isinstance(m._pack(torch.zeros(8)), DeviceSaved)
        # A view may retain its Python base, whereas detach retains only storage.
        if alias._base is None:
            assert reference() is None
        del alias
        gc.collect()
        assert m.summary()["live_device_bytes"] == 0
        assert isinstance(m._pack(torch.zeros(8)), DeviceSaved)


def test_cpp_saved_alias_keeps_charge_without_original_python_wrapper():
    with manager(max_device_bytes=32) as m:
        saved = m._pack(torch.arange(8.0))
        value = m._unpack(saved)
        reference = weakref.ref(value)
        # Disable our hooks for this separate graph. Autograd owns saved native
        # tensors in C++; its lifetime must still count against residency.
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), lambda x: x):
            parameter = torch.ones(8, requires_grad=True)
            result = parameter * value
        del saved, value
        gc.collect()
        assert reference() is None
        assert m.summary()["live_device_bytes"] == 32
        assert not isinstance(m._pack(torch.zeros(8)), DeviceSaved)
        result.sum().backward()
        del result
        gc.collect()
        assert m.summary()["live_device_bytes"] == 0


def test_empty_storages_have_distinct_cpp_identity():
    with manager(max_device_bytes=1) as m:
        first = m._pack(torch.empty(0))
        second = m._pack(torch.empty(0))
        assert m.summary()["device_storage_count"] == 2
        assert m.summary()["device_alias_pack_count"] == 0
        del first, second
        assert m.summary()["device_storage_count"] == 0


def test_pending_storage_can_be_readmitted_without_double_charge():
    with manager(max_device_bytes=32) as m:
        source = torch.arange(8.0)
        first = m._pack(source)
        del first
        facts = m.summary()
        assert facts["device_active_view_count"] == 0
        assert facts["device_pending_storage_count"] == 1
        second = m._pack(source.view(torch.int32))
        facts = m.summary()
        assert facts["device_active_view_count"] == 1
        assert facts["live_device_bytes"] == 32
        assert facts["device_pending_storage_count"] == 0
        assert facts["device_alias_pack_count"] == 1
        assert m._unpack(second).dtype == torch.int32
        del second, source
        facts = m.summary()
        assert facts["device_active_view_count"] == facts["live_device_bytes"] == 0


def test_external_backing_resize_cannot_be_admitted_as_free_alias():
    with manager(max_device_bytes=32) as m:
        source = torch.arange(8.0)
        saved = m._pack(source)
        source.untyped_storage().resize_(64)
        with pytest.raises(RuntimeError, match="changed size"):
            m._pack(source)
        with pytest.raises(RuntimeError, match="mutated"):
            m._unpack(saved)


def test_repeated_rows_reuse_released_budget_even_after_context_exit():
    m = manager(max_device_bytes=32)
    with m:
        for _ in range(128):
            saved = m._pack(torch.arange(8.0))
            assert isinstance(saved, DeviceSaved)
            del saved
            assert m.summary()["live_device_bytes"] == 0
        remaining = m._pack(torch.ones(8))
    assert m.summary()["closed"]
    assert m.summary()["live_device_bytes"] == 32
    del remaining
    assert m.summary()["live_device_bytes"] == 0
    assert m.summary()["peak_device_bytes"] == 32
    assert m.summary()["device_resident_pack_count"] == 129


def test_capacity_reclaimed_without_calling_summary_or_retaining_payload():
    with manager(max_device_bytes=32) as m:
        for _ in range(128):
            saved = m._pack(torch.ones(8))
            assert isinstance(saved, DeviceSaved)
            del saved
        facts = m.summary()
        assert facts["live_device_bytes"] == facts["device_storage_count"] == 0
        assert facts["device_spill_count"] == 0


def test_context_exception_does_not_pin_storage_after_alias_release():
    m = manager(max_device_bytes=32)
    with pytest.raises(RuntimeError, match="intentional"), m:
        saved = m._pack(torch.ones(8))
        alias = m._unpack(saved).detach()
        del saved
        raise RuntimeError("intentional")
    assert m.summary()["closed"]
    assert m.summary()["live_device_bytes"] == 32
    del alias
    gc.collect()
    assert m.summary()["live_device_bytes"] == m.summary()["device_storage_count"] == 0


def test_concurrent_admission_cannot_exceed_unique_storage_budget():
    barrier = Barrier(8)
    with manager(max_device_bytes=32) as m:
        m._bind()

        def hold(_):
            saved = m._pack(torch.ones(8))
            barrier.wait(timeout=10)
            return saved

        with ThreadPoolExecutor(max_workers=8) as workers:
            saved = list(workers.map(hold, range(8)))
        assert sum(isinstance(value, DeviceSaved) for value in saved) == 1
        assert m.summary()["peak_device_bytes"] == 32
        assert m.summary()["device_spill_count"] == 7
        saved.clear()
        gc.collect()
        assert m.summary()["live_device_bytes"] == 0


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64, torch.bool]
)
def test_raw_values_dtype_and_zero_size_preserved(dtype):
    with manager(max_device_bytes=128) as m:
        source = torch.tensor([0, 1, 0, 1], dtype=dtype)
        saved = m._pack(source)
        actual = m._unpack(saved)
        assert actual.dtype == dtype and actual.data_ptr() == source.data_ptr()
        torch.testing.assert_close(actual, source, rtol=0, atol=0)
        empty = m._pack(torch.empty((0, 7), dtype=dtype))
        assert isinstance(empty, DeviceSaved)
        assert tuple(m._unpack(empty).shape) == (0, 7)


@pytest.mark.parametrize("budget", [256, 2**20])
def test_baseline_three_diagnostics_backward_adam_rng_exact(tmp_path, budget):
    torch.manual_seed(391)
    original = model()

    def run(enabled):
        net = copy.deepcopy(original)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
        before_rng = torch.get_rng_state().clone()
        context = (
            manager(
                net,
                max_device_bytes=budget,
                max_host_bytes=1,
                disk_backend="arena_v3",
                scratch_directory=tmp_path / "STORAGE",
                max_disk_bytes=2**24,
            )
            if enabled
            else nullcontext()
        )
        with context as storage:
            values = net(torch.linspace(-1, 1, 32 * 7).reshape(32, 7)).log_softmax(-1)[:, 3]
            result = sequence_loss(
                values, values.detach() + 0.03, values.detach() - 0.02, 0.4, -0.2
            )
            diagnostics = [
                torch.autograd.grad(result[k] / 128, tuple(net.parameters()), retain_graph=True)
                for k in ("native", "aux", "kl")
            ]
            (result["loss"] / 128).backward()
            gradients = [p.grad.clone() for p in net.parameters()]
            optimizer.step()
            torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)
            if enabled:
                del result, values
                gc.collect()
                facts = storage.summary()
                assert facts["device_resident_pack_count"] > 0
                assert facts["peak_device_bytes"] <= budget
                assert facts["live_device_bytes"] == facts["device_storage_count"] == 0
                assert facts["live_disk_bytes"] == facts["arena_physical_bytes"] == 0
                assert not facts["storage_failures"] and not facts["cleanup_errors"]
                if budget == 256:
                    assert facts["device_spill_count"] > 0
                else:
                    assert facts["device_spill_count"] == 0
        return diagnostics, gradients, net.state_dict(), optimizer.state_dict()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)
