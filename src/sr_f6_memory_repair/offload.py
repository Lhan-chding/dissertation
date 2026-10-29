"""Explicit saved-activation storage repair; never a model admission or retry.

The registered computation is unchanged: no casts, truncation, recomputation,
checkpoint mode or RNG operation. Only CUDA intermediates saved for backward
are copied synchronously to CPU and restored to their original device. Model
parameter/buffer storages stay resident instead of being copied once per use.

The frozen caller must not mutate saved activations in-place. Like PyTorch's
save_on_cpu, copied saved tensors do not retain a dead CUDA tensor's version
counter. Live-source and retained-model mutations are checked on unpack.
"""

from __future__ import annotations

import weakref
from contextlib import AbstractContextManager

SCHEMA = "srf6-saved-activation-offload-v1"


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


class SavedActivationOffload(AbstractContextManager):
    """One scoped hook manager; each execution must supply its actual model.

    A provider is resolved lazily so an immutable producer may load its runtime
    inside this scope. It must return the actual, uniquely bound model object.
    `fixture=True` copies CPU tensors for regression tests and is labeled as
    synthetic evidence. Production mode never offloads CPU tensors.
    """

    def __init__(self, model_or_provider, *, max_host_bytes, fixture=False):
        if type(max_host_bytes) is not int or max_host_bytes <= 0:
            raise ValueError("An explicit positive host-memory bound is required")
        if type(fixture) is not bool:
            raise TypeError("fixture must be an explicit boolean")
        self.provider = model_or_provider
        self.max_host_bytes = max_host_bytes
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
        self.entered = True
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self.hooks.__enter__()
        return self

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
        return dict(
            schema=SCHEMA,
            scope="SYNTHETIC_CPU_ONLY" if self.fixture else "CUDA_SAVED_ACTIVATION_STORAGE",
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
        )


def saved_activation_offload(model_or_provider, *, max_host_bytes, fixture=False):
    return SavedActivationOffload(model_or_provider, max_host_bytes=max_host_bytes, fixture=fixture)
