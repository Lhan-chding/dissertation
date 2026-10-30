"""Optional bounded saved-activation residency; no copies or changed operators.

Charge each live backing storage once, including unused bytes behind views.
This bounds retained activation payload, not CUDA allocator reservation, model
weights, transient compute, restored spill tensors, or total device memory.
The budget owns metadata and C++ storage weak tokens only. A tensor finalizer
marks a charge reclaimable, but only expiration of the actual backing storage
returns the bytes: Python views and C++ autograd aliases can outlive that tensor.
"""

from __future__ import annotations

import threading
import weakref
from collections import deque
from dataclasses import dataclass


def _identity(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _view(tensor):
    return (
        _identity(tensor),
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
    )


class DeviceSaved:
    def __init__(self, tensor, budget, key):
        self.tensor = tensor.detach()
        self.version = tensor._version
        self.view = _view(self.tensor)
        self.budget = budget
        # This only drops a hook reference, not the storage charge. Neither a
        # Python wrapper's lifetime nor a Python Storage weakref proves that
        # C++ storage has expired. The callback must capture no tensor/storage.
        weakref.finalize(self.tensor, budget.release, key)

    def unpack(self):
        with self.budget.lock:
            if self.tensor._version != self.version or _view(self.tensor) != self.view:
                raise RuntimeError("Saved device activation was mutated")
            self.budget.unpack_count += 1
            return self.tensor


class DeviceResidency:
    def __init__(self, maximum):
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("Device residency requires an explicit positive byte bound")
        self.maximum = maximum
        self.lock = threading.RLock()
        self.storages = {}
        self.pending = deque()
        self.queued = set()
        self.live_bytes = self.peak_bytes = 0
        self.pack_count = self.alias_count = self.spill_count = self.unpack_count = 0
        self.peak_storages = 0

    def pack(self, tensor):
        from torch.multiprocessing.reductions import StorageWeakRef

        storage = tensor.untyped_storage()
        token = StorageWeakRef(storage)
        key, size = token.cdata, storage.nbytes()
        with self.lock:
            alias = key in self.storages
            if alias and self.storages[key].size != size:
                raise RuntimeError("Saved device backing storage changed size")
            if not alias and self.live_bytes + size > self.maximum:
                # Bound hook overhead even when many external aliases remain
                # alive. Conservative spilling is safe; exceeding the bound is
                # not. Summary performs a full sweep at the receipt boundary.
                self._reap(limit=256)
                if self.live_bytes + size > self.maximum:
                    self.spill_count += 1
                    return None
            # A weak token pins the C++ identity against pointer reuse without
            # keeping its payload alive, including distinct empty storages.
            if not alias:
                self.storages[key] = _StorageCharge(token, size)
                self.live_bytes += size
            self.storages[key].references += 1
            try:
                saved = DeviceSaved(tensor, self, key)
            except BaseException:
                self.release(key)
                raise
            self.pack_count += 1
            self.alias_count += int(alias)
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            self.peak_storages = max(self.peak_storages, len(self.storages))
            return saved

    def release(self, key):
        with self.lock:
            entry = self.storages[key]
            entry.references -= 1
            if not entry.references and key not in self.queued:
                self.pending.append(key)
                self.queued.add(key)

    def _reap(self, limit=None):
        count = len(self.pending) if limit is None else min(limit, len(self.pending))
        for _ in range(count):
            key = self.pending.popleft()
            self.queued.remove(key)
            entry = self.storages[key]
            if entry.references:
                continue
            if entry.token.expired():
                del self.storages[key]
                self.live_bytes -= entry.size
            else:
                self.pending.append(key)
                self.queued.add(key)

    def summary(self):
        with self.lock:
            self._reap()
            return dict(
                device_residency_schema="srf6-bounded-device-residency-candidate-v1",
                device_accounting_scope="UNIQUE_LIVE_SAVED_ACTIVATION_BACKING_STORAGE",
                device_accounting_release="CPP_STORAGE_EXPIRATION_CONSERVATIVE_REAP",
                device_peak_scope="CONSERVATIVE_CHARGED_PAYLOAD_HIGH_WATER_NOT_CUDA_PEAK",
                max_device_bytes=self.maximum,
                live_device_bytes=self.live_bytes,
                peak_device_bytes=self.peak_bytes,
                device_storage_count=len(self.storages),
                device_pending_storage_count=len(self.pending),
                device_active_view_count=sum(entry.references for entry in self.storages.values()),
                peak_device_storage_count=self.peak_storages,
                device_resident_pack_count=self.pack_count,
                device_alias_pack_count=self.alias_count,
                device_resident_unpack_count=self.unpack_count,
                device_spill_count=self.spill_count,
            )


@dataclass
class _StorageCharge:
    token: object
    size: int
    references: int = 0
