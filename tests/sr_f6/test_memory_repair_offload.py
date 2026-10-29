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


def run(original, length, offload, scratch=None):
    net = copy.deepcopy(original)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    inputs = torch.linspace(-1.3, 1.1, length * 7).reshape(length, 7)
    targets = torch.arange(length) % 19
    context = (
        saved_activation_offload(
            net,
            max_host_bytes=1 if scratch else 32 * 1024 * 1024,
            fixture=True,
            scratch_directory=scratch,
            max_disk_bytes=32 * 1024 * 1024 if scratch else None,
        )
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


def test_hybrid_1536_values_gradients_diagnostics_and_adam_exact(tmp_path):
    original = model()
    expected = run(original, 1536, False)
    actual = run(original, 1536, True, tmp_path / "STORAGE")
    torch.testing.assert_close(actual[:5], expected[:5], rtol=0, atol=0)
    summary = actual[5]
    assert summary["disk_bytes_read"] > summary["disk_bytes_written"] > 0
    assert summary["peak_staging_host_bytes"] <= summary["staging_chunk_bytes"]
    assert summary["live_host_bytes"] == summary["live_disk_bytes"] == 0
    assert not summary["storage_failures"] and not summary["cleanup_errors"]
    assert not list((tmp_path / "STORAGE").iterdir())


@pytest.mark.parametrize("kind", ["transpose", "slice", "expand", "scalar", "integer", "bool"])
def test_hybrid_raw_values_exact_for_views_and_dtypes(tmp_path, monkeypatch, kind):
    from sr_f6_memory_repair import offload

    monkeypatch.setattr(offload, "STAGING_CHUNK_BYTES", 128)
    source = torch.arange(77, dtype=torch.float32).reshape(7, 11)
    values = {
        "transpose": source.T,
        "slice": source[:, ::2],
        "expand": source[:1].expand(9, 11),
        "scalar": torch.tensor(1.375),
        "integer": source.to(torch.int64).T,
        "bool": (source > 17).T,
    }
    value = values[kind]
    with saved_activation_offload(
        model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=10000,
    ) as m:
        # A bool scalar would fit in the host pool; these values all exceed it.
        packed = m._pack(value)
        torch.testing.assert_close(m._unpack(packed), value, rtol=0, atol=0)
        torch.testing.assert_close(m._unpack(packed), value, rtol=0, atol=0)
        assert m.summary()["peak_staging_host_bytes"] <= 128
        del packed
        gc.collect()
        assert m.summary()["live_disk_bytes"] == 0
        assert not list((tmp_path / "STORAGE").iterdir())


def test_hybrid_128_rows_release_all_owned_files(tmp_path, monkeypatch):
    from sr_f6_memory_repair import offload

    monkeypatch.setattr(
        offload.os, "fsync", lambda _: None
    )  # lifecycle test, not durability evidence
    net = model()
    with saved_activation_offload(
        net,
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=100000,
    ) as m:
        for _ in range(128):
            (net(torch.ones(4, 7)).square().mean() / 128).backward()
            assert m.summary()["live_disk_bytes"] == m.summary()["live_host_bytes"] == 0
            assert not list((tmp_path / "STORAGE").iterdir())
        assert m.summary()["disk_bytes_written"] > 100 * m.summary()["peak_disk_bytes"]


def test_corrupt_spill_preserved_and_rejected(tmp_path):
    with saved_activation_offload(
        model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=10000,
    ) as m:
        packed = m._pack(torch.ones(4, 7))
        with packed.path.open("r+b") as f:
            f.write(b"\xff")
        with pytest.raises(RuntimeError, match="digest differs"):
            m._unpack(packed)
        path = packed.path
        del packed
        gc.collect()
        assert path.exists()
        assert m.summary()["storage_failures"][0]["stage"] == "read"
        assert m.summary()["live_disk_bytes"] == 4 * 7 * 4


def test_foreign_file_and_replaced_spill_never_deleted(tmp_path):
    target = tmp_path / "protected"
    target.write_bytes(b"protected")
    with saved_activation_offload(
        model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=10000,
    ) as m:
        packed = m._pack(torch.ones(4, 7))
        foreign = tmp_path / "STORAGE/foreign"
        foreign.write_bytes(b"foreign")
        packed.path.unlink()
        packed.path.symlink_to(target)
        path = packed.path
        del packed
        gc.collect()
        assert path.is_symlink()
        assert target.read_bytes() == b"protected"
        assert foreign.read_bytes() == b"foreign"
        assert m.summary()["cleanup_errors"]


def test_replaced_scratch_directory_is_never_cleaned(tmp_path):
    scratch = tmp_path / "STORAGE"
    with saved_activation_offload(
        model(), max_host_bytes=1, fixture=True, scratch_directory=scratch, max_disk_bytes=10000
    ) as m:
        packed = m._pack(torch.ones(4, 7))
        filename = packed.path.name
        scratch.rename(tmp_path / "preserved")
        scratch.mkdir()
        replacement = scratch / filename
        replacement.write_bytes(b"foreign")
        del packed
        gc.collect()
        assert replacement.read_bytes() == b"foreign"
        assert (tmp_path / "preserved" / filename).exists()
        assert m.cleanup_errors
        with pytest.raises(PermissionError, match="directory identity changed"):
            m.summary()


def test_hybrid_disk_bound_and_free_space_fail_before_file_creation(tmp_path, monkeypatch):
    from sr_f6_memory_repair import offload

    with saved_activation_offload(
        model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "SMALL",
        max_disk_bytes=16,
    ) as m:
        with pytest.raises(MemoryError, match="disk bound"):
            m._pack(torch.ones(4, 7))
        assert not list((tmp_path / "SMALL").iterdir())
    with saved_activation_offload(
        model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "FULL",
        max_disk_bytes=10000,
    ) as m:
        original = offload.shutil.disk_usage(tmp_path)
        monkeypatch.setattr(offload.shutil, "disk_usage", lambda _: original._replace(free=0))
        with pytest.raises(OSError, match="free space"):
            m._pack(torch.ones(4, 7))
        assert not list((tmp_path / "FULL").iterdir())


def test_failed_spill_write_keeps_partial_file_and_metadata(tmp_path, monkeypatch):
    from sr_f6_memory_repair import offload

    def fail(_):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(offload.os, "fsync", fail)
    with saved_activation_offload(
        model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=10000,
    ) as m:
        with pytest.raises(OSError, match="injected fsync"):
            m._pack(torch.ones(4, 7))
        assert len(list((tmp_path / "STORAGE").iterdir())) == 1
        assert m.summary()["storage_failures"][0]["stage"] == "write"
        assert m.summary()["live_disk_bytes"] == 4 * 7 * 4


def test_scratch_must_be_new_canonical_and_project_scoped(tmp_path):
    existing = tmp_path / "EXISTING"
    existing.mkdir()
    with (
        pytest.raises(FileExistsError),
        saved_activation_offload(
            model(),
            max_host_bytes=1,
            fixture=True,
            scratch_directory=existing,
            max_disk_bytes=10000,
        ),
    ):
        pass
    linked = tmp_path / "linked"
    linked.symlink_to(existing, target_is_directory=True)
    with (
        pytest.raises(PermissionError, match="canonical"),
        saved_activation_offload(
            model(),
            max_host_bytes=1,
            fixture=True,
            scratch_directory=linked / "STORAGE",
            max_disk_bytes=10000,
        ),
    ):
        pass
    with (
        pytest.raises(PermissionError, match="canonical project"),
        saved_activation_offload(
            model(), max_host_bytes=1, scratch_directory=tmp_path / "OUTSIDE", max_disk_bytes=10000
        ),
    ):
        pass
    assert not (tmp_path / "OUTSIDE").exists()


def test_mixed_pool_and_spill_stay_bounded_and_preserve_raw_special_values(tmp_path):
    with saved_activation_offload(
        model(),
        max_host_bytes=16,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=10000,
    ) as m:
        host = m._pack(torch.ones(4))
        source = torch.tensor([float("nan"), -0.0, float("inf"), -float("inf"), 1.0])
        disk = m._pack(source)
        restored = m._unpack(disk)
        assert torch.equal(restored.view(torch.uint8), source.view(torch.uint8))
        assert m.summary()["live_host_bytes"] == 16
        assert m.summary()["live_disk_bytes"] == 20
        assert m.summary()["peak_host_bytes"] <= 16
        assert m.summary()["peak_disk_bytes"] <= 10000
        del host, disk
        gc.collect()
        assert m.summary()["live_host_bytes"] == m.summary()["live_disk_bytes"] == 0
        assert not list((tmp_path / "STORAGE").iterdir())
