"""Exact-byte activation arenas with one durable seal per packing epoch.

An epoch becomes immutable on its first read. Retained diagnostic graphs read
the same byte ranges repeatedly. A later pack opens another epoch; an arena is
removed only after every saved record is released and its path is still owned.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
import weakref


class _Arena:
    def __init__(self, owner, size):
        from .offload import STAGING_CHUNK_BYTES

        self.owner = owner
        self.path = owner.scratch_directory / (uuid.uuid4().hex + ".arena")
        self.length = self.records = 0
        self.sealed = self.preserve = self.deleted = False
        self.fd = None
        owner._check_scratch()
        if shutil.disk_usage(owner.scratch_directory).free < size + STAGING_CHUNK_BYTES:
            raise OSError("Insufficient free space for owned activation arena")
        self.fd = os.open(
            self.path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=owner.scratch_fd,
        )
        value = os.fstat(self.fd)
        self.identity = (value.st_dev, value.st_ino)
        owner.arena_open_count += 1
        owner.arena_created_count += 1
        owner._arenas.add(self)

    def check(self):
        self.owner._check_scratch()
        named = os.stat(self.path.name, dir_fd=self.owner.scratch_fd, follow_symlinks=False)
        opened = os.fstat(self.fd)
        if any(
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or (value.st_dev, value.st_ino) != self.identity
            or value.st_size != self.length
            for value in (named, opened)
        ):
            raise PermissionError("Owned activation arena file identity or length changed")

    def seal(self):
        if self.preserve:
            raise PermissionError("Failed activation arena is preserved")
        if not self.sealed:
            self.check()
            os.fsync(self.fd)
            self.owner.arena_fsync_count += 1
            self.sealed = True

    def release(self):
        self.records -= 1
        if self.records or self.preserve or self.deleted:
            return
        try:
            self.check()
            os.unlink(self.path.name, dir_fd=self.owner.scratch_fd)
            os.close(self.fd)
            self.fd = None
            self.deleted = True
            self.owner.arena_physical_bytes -= self.length
            self.owner.arena_deleted_count += 1
            self.owner._arenas.discard(self)
        except BaseException as exc:
            self.preserve = True
            self.owner.cleanup_errors.append(
                dict(file=self.path.name, error_type=type(exc).__name__, error=str(exc))
            )

    def __del__(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None


class ArenaSaved:
    def __init__(self, tensor, owner, size, arena):
        import torch

        from .offload import _blocks

        self.owner, self.arena, self.size = owner, arena, size
        self.source, self.version = weakref.ref(tensor), tensor._version
        self.device, self.dtype, self.shape = tensor.device, tensor.dtype, tuple(tensor.shape)
        self.offset, self.path = arena.length, arena.path
        self.finished = self.released = False
        owner.live_disk_bytes += size
        owner.peak_disk_bytes = max(owner.peak_disk_bytes, owner.live_disk_bytes)
        arena.records += 1
        written = 0
        digest = hashlib.sha256()
        try:
            os.lseek(arena.fd, self.offset, os.SEEK_SET)
            for block in _blocks(tensor):
                cpu = block.detach().to(
                    device="cpu",
                    copy=True,
                    non_blocking=False,
                    memory_format=torch.contiguous_format,
                )
                data = memoryview(cpu.reshape(-1).view(torch.uint8).numpy()).cast("B")
                owner.peak_staging_host_bytes = max(owner.peak_staging_host_bytes, len(data))
                digest.update(data)
                offset = 0
                while offset < len(data):
                    count = os.write(arena.fd, data[offset:])
                    owner.arena_write_calls += 1
                    if not count:
                        raise OSError("Activation arena write made no progress")
                    offset += count
                    written += count
                    arena.length += count
                    owner.disk_bytes_written += count
                    owner.arena_physical_bytes += count
                    owner.peak_arena_physical_bytes = max(
                        owner.peak_arena_physical_bytes, owner.arena_physical_bytes
                    )
                del data, cpu
            if written != size:
                raise RuntimeError("Activation arena logical byte count differs")
            self.sha256 = digest.hexdigest()
            self.finished = True
        except BaseException as exc:
            owner.live_disk_bytes -= size - written
            arena.preserve = True
            owner.storage_failures.append(
                dict(
                    file=arena.path.name,
                    stage="write",
                    bytes=written,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            raise

    def unpack(self):
        import torch

        from .offload import STAGING_CHUNK_BYTES

        with self.owner._arena_lock:
            try:
                original = self.source()
                if original is not None and original._version != self.version:
                    raise RuntimeError("Saved activation was mutated in-place")
                self.arena.seal()
                value = os.fstat(self.arena.fd)
                if value.st_size != self.arena.length:
                    raise RuntimeError("Activation arena length changed")
                if self.offset < 0 or self.size < 0 or self.offset + self.size > self.arena.length:
                    raise RuntimeError("Activation arena range differs")
                restored = torch.empty(self.shape, dtype=self.dtype, device=self.device)
                flat = restored.reshape(-1)
                buffer = bytearray(min(STAGING_CHUNK_BYTES, self.size))
                self.owner.peak_staging_host_bytes = max(
                    self.owner.peak_staging_host_bytes, len(buffer)
                )
                digest = hashlib.sha256()
                byte_offset = 0
                os.lseek(self.arena.fd, self.offset, os.SEEK_SET)
                while byte_offset < self.size:
                    needed = min(len(buffer), self.size - byte_offset)
                    view = memoryview(buffer)[:needed]
                    received = 0
                    while received < needed:
                        count = os.readv(self.arena.fd, [view[received:]])
                        self.owner.arena_read_calls += 1
                        if not count:
                            raise OSError("Activation arena read ended early")
                        received += count
                    digest.update(view)
                    self.owner.disk_bytes_read += received
                    staging = torch.frombuffer(view, dtype=self.dtype)
                    start = byte_offset // restored.element_size()
                    flat[start : start + staging.numel()].copy_(staging, non_blocking=False)
                    byte_offset += received
                    del staging, view
                if digest.hexdigest() != self.sha256:
                    raise RuntimeError("Activation arena content digest differs")
                if os.fstat(self.arena.fd).st_size != self.arena.length:
                    raise RuntimeError("Activation arena length changed during read")
                self.owner.unpack_count += 1
                return restored
            except BaseException as exc:
                self.arena.preserve = True
                self.owner.storage_failures.append(
                    dict(
                        file=self.path.name,
                        stage="read",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )
                raise

    def __del__(self):
        if not getattr(self, "finished", False) or self.released:
            return
        with self.owner._arena_lock:
            self.released = True
            if not self.arena.preserve:
                self.owner.live_disk_bytes -= self.size
            self.arena.release()


def pack(owner, tensor, size):
    with owner._arena_lock:
        if owner.storage_failures or owner.cleanup_errors:
            raise PermissionError("Failed activation arena context is preserved")
        if (
            owner.live_disk_bytes + size > owner.max_disk_bytes
            or owner.arena_physical_bytes + size > owner.max_disk_bytes
        ):
            raise MemoryError("Registered saved-activation physical or live disk bound exceeded")
        arena = owner._arena_current
        if arena is None or arena.sealed or arena.deleted:
            arena = _Arena(owner, size)
            owner._arena_current = arena
        return ArenaSaved(tensor, owner, size, arena)
