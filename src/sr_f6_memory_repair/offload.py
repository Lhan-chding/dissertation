"""Explicit saved-activation storage repair; never a model admission or retry.

The registered computation is unchanged: no casts, truncation, recomputation,
checkpoint mode or RNG operation. CUDA intermediates saved for backward use
bounded CPU storage and optionally a private raw-byte disk spill. Model
parameter/buffer storages stay resident instead of being copied once per use.
Disk transfers stage at most 64 MiB in addition to the retained CPU bound;
neither bound includes other process memory or the operating system page cache.

The frozen caller must not mutate saved activations in-place. Like PyTorch's
save_on_cpu, copied saved tensors do not retain a dead CUDA tensor's version
counter. Live-source and retained-model mutations are checked on unpack.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
import weakref
from contextlib import AbstractContextManager
from pathlib import Path

SCHEMA = "srf6-saved-activation-offload-v1"
HYBRID_SCHEMA = "srf6-saved-activation-hybrid-v2"
STAGING_CHUNK_BYTES = 64 * 1024 * 1024
PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")


def _storage(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def _model_inventory(model):
    # Keep aliases and names: replacement with an equal tensor or even another
    # Parameter sharing the old storage must not impersonate the bound model.
    return tuple(
        (
            kind,
            name,
            id(tensor),
            _storage(tensor),
            str(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.requires_grad,
        )
        for kind, values in (
            ("parameter", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        )
        for name, tensor in values
    )


class _Resident:
    def __init__(self, tensor):
        self.tensor = tensor.detach()
        self.version = tensor._version

    def unpack(self):
        if self.tensor._version != self.version:
            raise RuntimeError("Saved resident tensor was mutated in-place")
        return self.tensor


class _Offloaded:
    def __init__(self, tensor, owner, size):
        self.source = weakref.ref(tensor)
        self.version = tensor._version
        self.device = tensor.device
        self.owner = owner
        self.size = size
        self.cpu = tensor.detach().to(device="cpu", copy=True, non_blocking=False)
        self.released = False
        owner.live_host_bytes += size
        owner.peak_host_bytes = max(owner.peak_host_bytes, owner.live_host_bytes)

    def unpack(self):
        original = self.source()
        if original is not None and original._version != self.version:
            raise RuntimeError("Saved activation was mutated in-place")
        self.owner.unpack_count += 1
        # Do not cache CUDA copies: repeated diagnostic backward calls may unpack
        # the same retained graph, but each backward owns its transient copy.
        return self.cpu.to(device=self.device, non_blocking=False)

    def __del__(self):
        if hasattr(self, "released") and not self.released:
            self.owner.live_host_bytes -= self.size
            self.released = True


def _blocks(tensor):
    """Logical row-major views, never a full CUDA contiguous/reshape copy."""
    if tensor.numel() * tensor.element_size() <= STAGING_CHUNK_BYTES:
        yield tensor
        return
    row_bytes = tensor[0].numel() * tensor.element_size()
    rows = max(1, STAGING_CHUNK_BYTES // row_bytes)
    if row_bytes > STAGING_CHUNK_BYTES:
        for index in range(tensor.shape[0]):
            yield from _blocks(tensor[index])
    else:
        for start in range(0, tensor.shape[0], rows):
            yield tensor[start : start + rows]


class _DiskSaved:
    def __init__(self, tensor, owner, size):
        import torch

        self.owner, self.size = owner, size
        self.source, self.version = weakref.ref(tensor), tensor._version
        self.device, self.dtype, self.shape = tensor.device, tensor.dtype, tuple(tensor.shape)
        self.finished = self.released = self.preserve = False
        owner._check_scratch()
        if owner.live_disk_bytes + size > owner.max_disk_bytes:
            raise MemoryError("Registered saved-activation disk bound exceeded")
        if shutil.disk_usage(owner.scratch_directory).free < size + STAGING_CHUNK_BYTES:
            raise OSError("Insufficient free space for owned activation spill")
        self.path = owner.scratch_directory / (uuid.uuid4().hex + ".tensor")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(self.path.name, flags, 0o600, dir_fd=owner.scratch_fd)
        self.identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        owner.live_disk_bytes += size
        owner.peak_disk_bytes = max(owner.peak_disk_bytes, owner.live_disk_bytes)
        written = 0
        hasher = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb", buffering=0) as stream:
                for block in _blocks(tensor):
                    cpu = block.detach().to(
                        device="cpu",
                        copy=True,
                        non_blocking=False,
                        memory_format=torch.contiguous_format,
                    )
                    data = memoryview(cpu.reshape(-1).view(torch.uint8).numpy()).cast("B")
                    owner.peak_staging_host_bytes = max(owner.peak_staging_host_bytes, len(data))
                    hasher.update(data)
                    offset = 0
                    while offset < len(data):
                        count = stream.write(data[offset:])
                        if not count:
                            raise OSError("Activation spill write made no progress")
                        offset += count
                        written += count
                        owner.disk_bytes_written += count
                    del data, cpu
                if written != size:
                    raise RuntimeError("Activation spill logical byte count differs")
                os.fsync(stream.fileno())
            self.sha256 = hasher.hexdigest()
            self.finished = True
        except BaseException as exc:
            owner.live_disk_bytes -= size - written
            owner.storage_failures.append(
                dict(
                    file=self.path.name,
                    stage="write",
                    bytes=written,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            # Partial bytes remain as evidence; no retry or replacement file.
            raise

    def _check_file(self):
        self.owner._check_scratch()
        value = os.stat(self.path.name, dir_fd=self.owner.scratch_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or (value.st_dev, value.st_ino) != self.identity
            or value.st_size != self.size
        ):
            raise PermissionError("Owned activation spill file identity changed")

    def unpack(self):
        import torch

        original = self.source()
        if original is not None and original._version != self.version:
            raise RuntimeError("Saved activation was mutated in-place")
        try:
            self._check_file()
            restored = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            flat = restored.reshape(-1)
            buffer = bytearray(min(STAGING_CHUNK_BYTES, self.size))
            self.owner.peak_staging_host_bytes = max(
                self.owner.peak_staging_host_bytes, len(buffer)
            )
            digest = hashlib.sha256()
            byte_offset = 0
            fd = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.owner.scratch_fd)
            with os.fdopen(fd, "rb", buffering=0) as stream:
                current = os.fstat(stream.fileno())
                if (current.st_dev, current.st_ino) != self.identity:
                    raise PermissionError("Owned activation spill changed during open")
                while byte_offset < self.size:
                    needed = min(len(buffer), self.size - byte_offset)
                    view = memoryview(buffer)[:needed]
                    received = 0
                    while received < needed:
                        count = stream.readinto(view[received:])
                        if not count:
                            raise OSError("Activation spill read ended early")
                        received += count
                    digest.update(view)
                    self.owner.disk_bytes_read += received
                    staging = torch.frombuffer(view, dtype=self.dtype)
                    start = byte_offset // restored.element_size()
                    flat[start : start + staging.numel()].copy_(staging, non_blocking=False)
                    byte_offset += received
                    del staging, view
                if stream.read(1):
                    raise RuntimeError("Activation spill gained trailing bytes")
            if digest.hexdigest() != self.sha256:
                raise RuntimeError("Activation spill content digest differs")
            self.owner.unpack_count += 1
            return restored
        except BaseException as exc:
            self.preserve = True
            self.owner.storage_failures.append(
                dict(
                    file=self.path.name, stage="read", error_type=type(exc).__name__, error=str(exc)
                )
            )
            raise

    def __del__(self):
        if not getattr(self, "finished", False) or self.released or self.preserve:
            return
        try:
            self._check_file()
            os.unlink(self.path.name, dir_fd=self.owner.scratch_fd)
            self.owner.live_disk_bytes -= self.size
            self.released = True
        except BaseException as exc:
            self.owner.cleanup_errors.append(
                dict(file=self.path.name, error_type=type(exc).__name__, error=str(exc))
            )


class SavedActivationOffload(AbstractContextManager):
    """One scoped hook manager; each execution must supply its actual model.

    A provider is resolved lazily so an immutable producer may load its runtime
    inside this scope. It must return the actual, uniquely bound model object.
    `fixture=True` copies CPU tensors for regression tests and is labeled as
    synthetic evidence. Production mode never offloads CPU tensors.
    """

    def __init__(
        self,
        model_or_provider,
        *,
        max_host_bytes,
        fixture=False,
        scratch_directory=None,
        max_disk_bytes=None,
    ):
        if type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError("An explicit positive host-memory bound is required")
        if type(fixture) is not bool:
            raise TypeError("fixture must be an explicit boolean")
        if scratch_directory is None and max_disk_bytes is not None:
            raise ValueError("Disk bound requires a new private scratch directory")
        if scratch_directory is not None and (
            type(max_disk_bytes) is not int or max_disk_bytes <= 0
        ):
            raise ValueError("Scratch storage requires an explicit positive disk bound")
        self.provider = model_or_provider
        self.max_host_bytes = max_host_bytes
        self.scratch_directory = Path(scratch_directory) if scratch_directory is not None else None
        self.max_disk_bytes = max_disk_bytes
        self.scratch_identity = None
        self.scratch_fd = None
        self.live_disk_bytes = self.peak_disk_bytes = 0
        self.peak_staging_host_bytes = 0
        self.disk_bytes_written = self.disk_bytes_read = 0
        self.storage_failures, self.cleanup_errors = [], []
        self.fixture = fixture
        self.model = None
        self.resident = None
        self.model_tensors = None
        self.model_inventory = None
        self.pack_count = 0
        self.unpack_count = 0
        self.resident_count = 0
        self.cpu_passthrough_count = 0
        self.live_host_bytes = 0
        self.peak_host_bytes = 0
        self.total_offloaded_bytes = 0
        self.entered = False
        self.closed = False
        self.hooks = None

    def _bind(self):
        import torch

        if self.model is not None:
            return
        model = self.provider if isinstance(self.provider, torch.nn.Module) else self.provider()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("The provider must return the actual loaded torch model")
        tensors = tuple(model.parameters()) + tuple(model.buffers())
        if not tensors:
            raise ValueError("An empty model cannot establish resident storage identity")
        if not self.fixture and any(t.device.type != "cuda" for t in tensors):
            raise PermissionError("Production offload requires the actual all-CUDA model")
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
            raise PermissionError("Registered model floating tensors must remain FP32")
        self.model = model
        self.model_tensors = tuple((t, _storage(t)) for t in tensors)
        self.resident = frozenset(storage for _, storage in self.model_tensors)
        self.model_inventory = _model_inventory(model)

    def _pack(self, tensor):
        import torch

        self._bind()
        self.pack_count += 1
        if tensor.layout != torch.strided:
            raise TypeError("Only dense strided saved tensors are supported")
        key = _storage(tensor)
        if key in self.resident:
            self.resident_count += 1
            return _Resident(tensor)
        if tensor.device.type == "cpu" and not self.fixture:
            self.cpu_passthrough_count += 1
            return _Resident(tensor)
        if tensor.device.type != "cuda" and not self.fixture:
            raise PermissionError("Unexpected saved tensor device")
        size = tensor.numel() * tensor.element_size()
        if self.live_host_bytes + size > self.max_host_bytes:
            if self.scratch_directory is not None:
                packed = _DiskSaved(tensor, self, size)
                self.total_offloaded_bytes += size
                return packed
            raise MemoryError("Registered saved-activation host-memory bound exceeded")
        packed = _Offloaded(tensor, self, size)
        self.total_offloaded_bytes += size
        return packed

    @staticmethod
    def _unpack(packed):
        return packed.unpack()

    def __enter__(self):
        import torch

        if self.entered or self.closed:
            raise RuntimeError("An offload execution context cannot be reused")
        if self.scratch_directory is not None:
            path = self.scratch_directory
            if (
                not path.is_absolute()
                or path.resolve() != path
                or path.parent.resolve(strict=True) != path.parent
                or not path.parent.is_dir()
            ):
                raise PermissionError("Scratch must have an existing canonical parent")
            if not self.fixture and (path == PROJECT or not path.is_relative_to(PROJECT)):
                raise PermissionError("Production scratch must stay inside the canonical project")
            path.mkdir(mode=0o700, exist_ok=False)
            value = path.lstat()
            self.scratch_identity = (value.st_dev, value.st_ino)
            self.scratch_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._check_scratch()
        self.entered = True
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self.hooks.__enter__()
        return self

    def _check_scratch(self):
        path = self.scratch_directory
        value = path.lstat()
        opened = os.fstat(self.scratch_fd)
        if (
            not stat.S_ISDIR(value.st_mode)
            or path.resolve(strict=True) != path
            or (value.st_dev, value.st_ino) != self.scratch_identity
            or (opened.st_dev, opened.st_ino) != self.scratch_identity
        ):
            raise PermissionError("Private scratch directory identity changed")

    def __del__(self):
        if getattr(self, "scratch_fd", None) is not None:
            os.close(self.scratch_fd)
            self.scratch_fd = None

    def __exit__(self, *exception):
        try:
            return self.hooks.__exit__(*exception)
        finally:
            self.closed = True

    def summary(self):
        # Check the actual live model, not only the originally captured tensor
        # objects. This metadata-only traversal belongs at the receipt boundary,
        # not in every saved-tensor hook of the large model.
        if self.model is not None and _model_inventory(self.model) != self.model_inventory:
            raise RuntimeError("Bound model inventory changed during offload execution")
        if self.scratch_identity is not None:
            self._check_scratch()
        return dict(
            schema=HYBRID_SCHEMA if self.scratch_directory is not None else SCHEMA,
            scope="SYNTHETIC_CPU_ONLY"
            if self.fixture
            else (
                "CUDA_SAVED_ACTIVATION_HYBRID_STORAGE"
                if self.scratch_directory is not None
                else "CUDA_SAVED_ACTIVATION_STORAGE"
            ),
            model_bound=self.model is not None,
            max_host_bytes=self.max_host_bytes,
            pack_count=self.pack_count,
            unpack_count=self.unpack_count,
            resident_count=self.resident_count,
            cpu_passthrough_count=self.cpu_passthrough_count,
            live_host_bytes=self.live_host_bytes,
            peak_host_bytes=self.peak_host_bytes,
            total_offloaded_bytes=self.total_offloaded_bytes,
            closed=self.closed,
            numerical_pass=False,
            scratch_directory=str(self.scratch_directory) if self.scratch_directory else None,
            max_disk_bytes=self.max_disk_bytes,
            live_disk_bytes=self.live_disk_bytes,
            peak_disk_bytes=self.peak_disk_bytes,
            peak_staging_host_bytes=self.peak_staging_host_bytes,
            staging_chunk_bytes=STAGING_CHUNK_BYTES,
            disk_bytes_written=self.disk_bytes_written,
            disk_bytes_read=self.disk_bytes_read,
            storage_failures=list(self.storage_failures),
            cleanup_errors=list(self.cleanup_errors),
        )


def saved_activation_offload(
    model_or_provider, *, max_host_bytes, fixture=False, scratch_directory=None, max_disk_bytes=None
):
    return SavedActivationOffload(
        model_or_provider,
        max_host_bytes=max_host_bytes,
        fixture=fixture,
        scratch_directory=scratch_directory,
        max_disk_bytes=max_disk_bytes,
    )
