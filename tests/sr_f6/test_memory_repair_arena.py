"""Exact-byte arena lifecycle and metadata amplification regressions."""

import copy
import gc
import os
from contextlib import nullcontext

import pytest
import torch

from sr_f6.objective import sequence_loss
from sr_f6_memory_repair.offload import saved_activation_offload


def model():
    torch.manual_seed(181)
    return torch.nn.Sequential(torch.nn.Linear(7, 11), torch.nn.Tanh(), torch.nn.Linear(11, 19))


def manager(tmp_path, net=None, **kwargs):
    return saved_activation_offload(
        net if net is not None else model(),
        max_host_bytes=1,
        fixture=True,
        scratch_directory=tmp_path / "STORAGE",
        max_disk_bytes=kwargs.pop("max_disk_bytes", 2**24),
        disk_backend="arena_v3",
        **kwargs,
    )


def test_baseline_gradients_three_diagnostics_adam_and_rng_exact(tmp_path):
    original = model()

    def run(use_arena):
        net = copy.deepcopy(original)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
        rng = torch.get_rng_state().clone()
        context = manager(tmp_path, net) if use_arena else nullcontext()
        with context as storage:
            values = net(torch.linspace(-1, 1, 1536 * 7).reshape(1536, 7)).log_softmax(-1)[:, 3]
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
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            if use_arena:
                facts = storage.summary()
                assert facts["arena_created_count"] == facts["arena_fsync_count"] == 1
                assert facts["arena_deleted_count"] == 1
                assert facts["arena_physical_bytes"] == facts["live_disk_bytes"] == 0
                assert facts["disk_bytes_read"] > facts["disk_bytes_written"]
                assert not facts["storage_failures"] and not facts["cleanup_errors"]
        return values.detach(), diagnostics, gradients, net.state_dict(), optimizer.state_dict()

    torch.testing.assert_close(run(True), run(False), rtol=0, atol=0)


def test_many_records_one_open_and_fsync_repeated_reads(tmp_path, monkeypatch):
    from sr_f6_memory_repair import arena

    calls = []
    opens = []
    original = arena.os.fsync
    open_file = arena.os.open

    def observed_open(path, *args, **kwargs):
        if str(path).endswith(".arena"):
            opens.append(str(path))
        return open_file(path, *args, **kwargs)

    def sync(fd):
        calls.append(fd)
        return original(fd)

    monkeypatch.setattr(arena.os, "fsync", sync)
    monkeypatch.setattr(arena.os, "open", observed_open)
    with manager(tmp_path) as m:
        records = [m._pack(torch.full((16,), float(i))) for i in range(100)]
        assert len(list((tmp_path / "STORAGE").iterdir())) == 1
        assert m.summary()["arena_open_count"] == 1
        assert not calls
        for _ in range(4):
            for i, record in enumerate(records):
                torch.testing.assert_close(m._unpack(record), torch.full((16,), float(i)))
        assert len(calls) == 1
        assert len(opens) == 1
        del record
        records.clear()
        gc.collect()
        assert m.summary()["arena_deleted_count"] == 1
        assert not list((tmp_path / "STORAGE").iterdir())


def test_new_pack_after_seal_opens_new_epoch_and_preserves_old_reads(tmp_path):
    with manager(tmp_path) as m:
        first = m._pack(torch.arange(16.0))
        m._unpack(first)
        second = m._pack(torch.arange(16.0) + 4)
        assert first.path != second.path
        torch.testing.assert_close(m._unpack(first), torch.arange(16.0))
        torch.testing.assert_close(m._unpack(second), torch.arange(16.0) + 4)
        del first, second
        assert m.summary()["arena_created_count"] == m.summary()["arena_deleted_count"] == 2


def test_128_rows_release_all_bytes_and_files(tmp_path):
    net = model()
    with manager(tmp_path, net) as m:
        for _ in range(128):
            net(torch.ones(4, 7)).square().mean().backward()
            assert m.arena_physical_bytes == m.live_disk_bytes == 0
            assert not list((tmp_path / "STORAGE").iterdir())
        assert m.summary()["arena_created_count"] == m.summary()["arena_fsync_count"] == 128


@pytest.mark.parametrize("kind", ["transpose", "slice", "expand", "scalar", "integer", "bool"])
def test_arena_noncontiguous_and_raw_values(tmp_path, monkeypatch, kind):
    from sr_f6_memory_repair import offload

    monkeypatch.setattr(offload, "STAGING_CHUNK_BYTES", 128)
    source = torch.arange(77.0).reshape(7, 11)
    value = {
        "transpose": source.T,
        "slice": source[:, ::2],
        "expand": source[:1].expand(9, 11),
        "scalar": torch.tensor(-0.0),
        "integer": source.to(torch.int64).T,
        "bool": (source > 9).T,
    }[kind]
    with manager(tmp_path) as m:
        record = m._pack(value)
        restored = m._unpack(record)
        assert torch.equal(
            restored.contiguous().reshape(-1).view(torch.uint8),
            value.contiguous().reshape(-1).view(torch.uint8),
        )
        assert m.peak_staging_host_bytes <= 128
        del record


@pytest.mark.parametrize("mutation", ["contents", "truncate", "extend"])
def test_corrupt_arena_preserved(tmp_path, mutation):
    with manager(tmp_path) as m:
        record = m._pack(torch.ones(16))
        m._unpack(record)
        path = record.path
        with path.open("r+b") as stream:
            if mutation == "contents":
                stream.write(b"\xff")
            elif mutation == "truncate":
                stream.truncate(1)
            else:
                stream.seek(0, os.SEEK_END)
                stream.write(b"extra")
        with pytest.raises(RuntimeError, match=r"digest|length"):
            m._unpack(record)
        del record
        assert path.exists()
        assert m.storage_failures and m.arena_physical_bytes == 64


def test_replacement_before_completion_is_rejected_and_not_deleted(tmp_path):
    with manager(tmp_path) as m:
        record = m._pack(torch.ones(16))
        m._unpack(record)
        path = record.path
        path.rename(path.with_suffix(".preserved"))
        path.write_bytes(b"foreign")
        with pytest.raises(PermissionError, match="identity or length"):
            m.summary()
        del record
        assert path.read_bytes() == b"foreign"
        assert m.cleanup_errors


def test_scratch_replacement_and_foreign_files_preserved(tmp_path):
    with manager(tmp_path) as m:
        record = m._pack(torch.ones(16))
        old_name = record.path.name
        scratch = tmp_path / "STORAGE"
        scratch.rename(tmp_path / "original")
        scratch.mkdir()
        foreign = scratch / old_name
        foreign.write_bytes(b"foreign")
        with pytest.raises(PermissionError, match="directory identity changed"):
            m._unpack(record)
        del record
        assert foreign.read_bytes() == b"foreign"
        assert (tmp_path / "original" / old_name).exists()


def test_physical_limit_counts_released_ranges_until_arena_deleted(tmp_path):
    with manager(tmp_path, max_disk_bytes=128) as m:
        first = m._pack(torch.ones(16))
        second = m._pack(torch.ones(16))
        del first
        assert m.live_disk_bytes == 64 and m.arena_physical_bytes == 128
        with pytest.raises(MemoryError, match="physical or live disk bound"):
            m._pack(torch.ones(16))
        del second
        assert m.arena_physical_bytes == 0
        third = m._pack(torch.ones(16))
        del third


def test_partial_write_failure_preserves_arena(tmp_path, monkeypatch):
    from sr_f6_memory_repair import arena

    write = arena.os.write
    calls = []

    def partial(fd, data):
        calls.append(None)
        if len(calls) == 1:
            return write(fd, data[:7])
        raise OSError("injected write failure")

    monkeypatch.setattr(arena.os, "write", partial)
    with manager(tmp_path) as m:
        with pytest.raises(OSError, match="injected write"):
            m._pack(torch.ones(16))
        assert m.arena_physical_bytes == m.live_disk_bytes == 7
        assert next((tmp_path / "STORAGE").iterdir()).stat().st_size == 7
        assert m.storage_failures[0]["stage"] == "write"


def test_seal_failure_preserves_all_records(tmp_path, monkeypatch):
    from sr_f6_memory_repair import arena

    def fail(_):
        raise OSError("injected seal failure")

    monkeypatch.setattr(arena.os, "fsync", fail)
    with manager(tmp_path) as m:
        first, second = m._pack(torch.ones(16)), m._pack(torch.ones(16))
        path = first.path
        with pytest.raises(OSError, match="injected seal"):
            m._unpack(first)
        del first, second
        assert path.exists() and m.live_disk_bytes == m.arena_physical_bytes == 128


def test_live_source_mutation_preserves_arena(tmp_path):
    with manager(tmp_path) as m:
        value = torch.ones(16)
        record = m._pack(value)
        value.add_(1)
        with pytest.raises(RuntimeError, match="mutated"):
            m._unpack(record)
        del record
        assert list((tmp_path / "STORAGE").iterdir())


def test_context_failure_preserves_active_arena(tmp_path):
    with pytest.raises(RuntimeError, match="injected model"), manager(tmp_path) as m:
        record = m._pack(torch.ones(16))
        path = record.path
        raise RuntimeError("injected model")
    del record
    assert path.exists() and m.arena_physical_bytes == 64


def test_short_io_and_special_float_bytes_are_exact(tmp_path, monkeypatch):
    from sr_f6_memory_repair import arena

    write, readv = arena.os.write, arena.os.readv
    monkeypatch.setattr(arena.os, "write", lambda fd, data: write(fd, data[:7]))
    monkeypatch.setattr(arena.os, "readv", lambda fd, views: readv(fd, [views[0][:5]]))
    value = torch.tensor([float("nan"), -0.0, float("inf"), -float("inf"), 1.0])
    with manager(tmp_path) as m:
        record = m._pack(value)
        restored = m._unpack(record)
        assert torch.equal(restored.view(torch.uint8), value.view(torch.uint8))
        assert m.arena_write_calls == 3 and m.arena_read_calls == 4
        del record


def test_unexpected_read_eof_preserves_arena(tmp_path, monkeypatch):
    from sr_f6_memory_repair import arena

    with manager(tmp_path) as m:
        record = m._pack(torch.ones(16))
        path = record.path
        monkeypatch.setattr(arena.os, "readv", lambda *_: 0)
        with pytest.raises(OSError, match="read ended early"):
            m._unpack(record)
        del record
        assert path.exists() and m.storage_failures[0]["stage"] == "read"


def test_symlink_replacement_is_not_followed_or_deleted(tmp_path):
    target = tmp_path / "PROTECTED"
    target.write_bytes(b"protected")
    with manager(tmp_path) as m:
        record = m._pack(torch.ones(16))
        path = record.path
        path.unlink()
        path.symlink_to(target)
        with pytest.raises(PermissionError, match="identity or length"):
            m._unpack(record)
        del record
        assert target.read_bytes() == b"protected" and path.is_symlink()


def test_free_space_checked_before_epoch_file_creation(tmp_path, monkeypatch):
    from sr_f6_memory_repair import arena

    free = arena.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(arena.shutil, "disk_usage", lambda _: free._replace(free=0))
    with manager(tmp_path) as m:
        with pytest.raises(OSError, match="free space"):
            m._pack(torch.ones(16))
        assert not list((tmp_path / "STORAGE").iterdir())
