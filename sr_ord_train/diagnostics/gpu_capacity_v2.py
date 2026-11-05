"""Diagnostic-only, synchronous CUDA 0 -> CUDA 1 saved-activation snapshots.

One instance belongs to one original row and has a fixed cumulative 48 GiB
copy allowance. The model, optimizer, parameters, buffers and all arithmetic
stay on CUDA 0. This module does not move them or provide a CPU fallback.

Copied packets own only the CUDA 1 snapshot and a weak reference to the source.
Live source version changes and snapshot mutations are rejected on unpack.
PyTorch exposes no detached version-counter handle: after the original Python
tensor dies, mutations through a surviving storage alias cannot be diagnosed
by that weak reference. Such cases are counted, not asserted mutation-free.
This hook is therefore a diagnostic mechanism, not an equivalence certificate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
import weakref

CAP_BYTES = 48 * 1024 ** 3
GPU_OFFLOAD_CAP_BYTES = CAP_BYTES
OFFLOAD_MODE = 'bounded_gpu_saved_tensors_capacity48_v2'
SOURCE_DEVICE = 'cuda:0'
SNAPSHOT_DEVICE = 'cuda:1'


def _require(value, reason):
    if not value:
        raise RuntimeError(reason)


def _torch():
    import torch
    return torch


def _device_name(name):
    return ''.join(character for character in name.upper() if character.isalnum())


def validate_gpu_pair():
    """Inspect the two visible GPUs; never change visibility or execution device."""
    torch = _torch()
    cuda = torch.cuda
    _require(cuda.is_available() and cuda.device_count() == 2, 'Exactly two visible CUDA GPUs required')
    _require(cuda.current_device() == 0, 'Model/arithmetic device must remain cuda:0')
    properties = [cuda.get_device_properties(index) for index in (0, 1)]
    names = [cuda.get_device_name(index) for index in (0, 1)]
    _require(names[0] == names[1] and 'PRO6000' in _device_name(names[0]),
             'Both visible GPUs must be the same PRO6000 model')
    for field in ('major', 'minor', 'total_memory'):
        _require(getattr(properties[0], field) == getattr(properties[1], field),
                 'GPU capability/memory differs: ' + field)
    if all(hasattr(p, 'uuid') for p in properties):
        _require(str(properties[0].uuid) != str(properties[1].uuid), 'Visible GPUs must be distinct devices')
    peer_01 = bool(cuda.can_device_access_peer(0, 1))
    peer_10 = bool(cuda.can_device_access_peer(1, 0))
    _require(peer_01 and peer_10, 'Bidirectional CUDA peer access is required; no CPU fallback')
    return dict(status='GPU_PAIR_CAPABILITY_CHECKED', gpu_names=names,
                compute_capability=[properties[0].major, properties[0].minor],
                memory_bytes_each=properties[0].total_memory,
                peer_0_to_1=peer_01, peer_1_to_0=peer_10,
                source_device=SOURCE_DEVICE, snapshot_device=SNAPSHOT_DEVICE,
                cpu_offload=False, equivalence_verified=False)


def gpu_peak_stats(*, reset=False):
    """Return allocator counters on both GPUs; optional explicit peak reset."""
    torch = _torch()
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Two CUDA GPUs required')
    result = {}
    for index in (0, 1):
        torch.cuda.synchronize(index)
        if reset:
            torch.cuda.reset_peak_memory_stats(index)
        result[f'cuda:{index}'] = dict(
            allocated_bytes=torch.cuda.memory_allocated(index),
            reserved_bytes=torch.cuda.memory_reserved(index),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(index),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(index))
    return result


def _storage_identity(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


@dataclass(frozen=True)
class _Packet:
    tensor: object
    copied: bool
    source_ref: object
    source_version: int | None
    snapshot_version: int | None
    device: str
    dtype: object
    shape: tuple
    size_bytes: int


class GPUSavedTensors:
    """One-row hook. Copies are blocking and use no additional CUDA streams."""
    def __init__(self, persistent_tensors=(), *, row_id):
        _require(isinstance(row_id, str) and bool(row_id), 'Original row identity is required')
        self.gpu_pair = validate_gpu_pair()
        torch = _torch()
        self.persistent = set()
        for tensor in persistent_tensors:
            _require(torch.is_tensor(tensor) and str(tensor.device) == SOURCE_DEVICE,
                     'Every parameter/buffer must already be on cuda:0')
            if tensor.layout == torch.strided:
                self.persistent.add(_storage_identity(tensor))
        self._entered = False
        self._active = False
        self.stats = dict(row_id=row_id, mode=OFFLOAD_MODE, cap_bytes=CAP_BYTES,
                          copied_bytes=0, copied_tensors=0, unpacked_copies=0,
                          unpacked_bytes=0, kept_bytes=0, kept_tensors=0,
                          persistent_storage_skips=0, parameter_skips=0,
                          cap_skips=0, ineligible_skips=0, unversioned_skips=0,
                          source_version_checks=0, source_version_unobservable=0,
                          version_failures=0, copy_failures=0, unpack_failures=0,
                          pack_wall_seconds=0., unpack_wall_seconds=0.,
                          context_wall_seconds=0., transfer_wall_seconds=0.,
                          cpu_offload=False, async_streams=False,
                          scope='cumulative per original-row forward; never decremented on unpack',
                          source_version_check_scope='weakref live-source only; dead-source aliases unobservable',
                          equivalence_verified=False)

    def pack(self, tensor):
        torch = _torch()
        started = time.perf_counter()
        try:
            _require(torch.is_tensor(tensor), 'Saved value must be a tensor')
            _require(torch.cuda.current_device() == 0, 'Compute device changed from cuda:0')
            size = tensor.numel() * tensor.element_size()
            try:
                version = tensor._version
            except RuntimeError:
                version = None
            persistent = tensor.layout == torch.strided and _storage_identity(tensor) in self.persistent
            parameter = isinstance(tensor, torch.nn.Parameter)
            eligible = (str(tensor.device) == SOURCE_DEVICE and tensor.layout == torch.strided
                        and tensor.is_contiguous() and not tensor.is_conj() and not tensor.is_neg()
                        and not persistent and not parameter and version is not None and size > 0)
            under_cap = self.stats['copied_bytes'] + size <= CAP_BYTES
            if eligible and under_cap:
                transfer_started = time.perf_counter()
                try:
                    torch.cuda.synchronize(0)
                    snapshot = tensor.detach().to(device=SNAPSHOT_DEVICE, copy=True, non_blocking=False)
                    torch.cuda.synchronize(1)
                    _require(str(snapshot.device) == SNAPSHOT_DEVICE and snapshot.dtype == tensor.dtype
                             and tuple(snapshot.shape) == tuple(tensor.shape) and snapshot.is_contiguous()
                             and not snapshot.requires_grad, 'GPU snapshot changed device/dtype/shape/layout')
                    _require(tensor._version == version, 'Source mutated during synchronous snapshot')
                except BaseException:
                    self.stats['copy_failures'] += 1
                    raise
                finally:
                    self.stats['transfer_wall_seconds'] += time.perf_counter() - transfer_started
                self.stats['copied_bytes'] += size
                self.stats['copied_tensors'] += 1
                return _Packet(snapshot, True, weakref.ref(tensor), version, snapshot._version,
                               SOURCE_DEVICE, tensor.dtype, tuple(tensor.shape), size)
            self.stats['kept_bytes'] += size
            self.stats['kept_tensors'] += 1
            self.stats['persistent_storage_skips'] += int(persistent)
            self.stats['parameter_skips'] += int(parameter)
            self.stats['cap_skips'] += int(eligible and not under_cap)
            self.stats['ineligible_skips'] += int(not eligible)
            self.stats['unversioned_skips'] += int(version is None)
            # Kept tensors intentionally retain their storage/version counter.
            kept = tensor.detach()
            return _Packet(kept, False, None, version, version,
                           str(tensor.device), tensor.dtype, tuple(tensor.shape), size)
        finally:
            self.stats['pack_wall_seconds'] += time.perf_counter() - started

    def unpack(self, packed):
        torch = _torch()
        started = time.perf_counter()
        try:
            _require(isinstance(packed, _Packet), 'Unknown saved-tensor packet')
            _require(torch.cuda.current_device() == 0, 'Backward arithmetic device changed from cuda:0')
            snapshot = packed.tensor
            expected_device = SNAPSHOT_DEVICE if packed.copied else packed.device
            _require(str(snapshot.device) == expected_device and snapshot.dtype == packed.dtype
                     and tuple(snapshot.shape) == packed.shape, 'Saved snapshot metadata changed')
            if packed.snapshot_version is not None:
                if snapshot._version != packed.snapshot_version:
                    self.stats['version_failures'] += 1
                    raise RuntimeError('Saved snapshot/kept tensor was modified in place')
            if not packed.copied:
                return snapshot
            source = packed.source_ref()
            if source is None:
                self.stats['source_version_unobservable'] += 1
            else:
                self.stats['source_version_checks'] += 1
                if source._version != packed.source_version:
                    self.stats['version_failures'] += 1
                    raise RuntimeError('Original saved tensor was modified in place')
            del source
            transfer_started = time.perf_counter()
            try:
                torch.cuda.synchronize(1)
                restored = snapshot.to(device=packed.device, copy=True, non_blocking=False)
                torch.cuda.synchronize(0)
            finally:
                self.stats['transfer_wall_seconds'] += time.perf_counter() - transfer_started
            _require(str(restored.device) == SOURCE_DEVICE and restored.dtype == packed.dtype
                     and tuple(restored.shape) == packed.shape and restored.is_contiguous()
                     and not restored.requires_grad, 'Unpack must restore original CUDA dtype/shape without casts')
            self.stats['unpacked_copies'] += 1
            self.stats['unpacked_bytes'] += packed.size_bytes
            return restored
        except BaseException:
            self.stats['unpack_failures'] += 1
            raise
        finally:
            self.stats['unpack_wall_seconds'] += time.perf_counter() - started

    @contextmanager
    def context(self):
        _require(not self._entered and not self._active, 'Use a new hook for each original row')
        self._entered = self._active = True
        started = time.perf_counter()
        try:
            with _torch().autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
                yield self
        finally:
            self._active = False
            self.stats['context_wall_seconds'] += time.perf_counter() - started

    def memory_stats(self):
        return gpu_peak_stats()


BoundedGPUSavedTensors = GPUSavedTensors


def validate_snapshot_stats(stats, *, expected_cap=CAP_BYTES):
    """Fail closed when any original row lacks verified GPU snapshot telemetry."""
    _require(type(expected_cap) is int and expected_cap in (16*1024**3, CAP_BYTES),
             'Only original16 or capacity48 snapshot evidence is eligible')
    expected_mode = 'bounded_gpu_saved_tensors' if expected_cap == 16*1024**3 else OFFLOAD_MODE
    _require(stats.get('mode') == expected_mode and stats.get('cap_bytes') == expected_cap and
             type(stats.get('copied_bytes')) is int and 0 < stats['copied_bytes'] <= expected_cap and
             type(stats.get('unpacked_copies')) is int and stats['unpacked_copies'] > 0 and
             stats.get('cpu_offload') is False and
             all(type(stats.get(k)) is int and stats[k] == 0 for k in
                 ('copy_failures', 'version_failures', 'unpack_failures')),
             'Missing, failed, non-GPU, or out-of-cap snapshot evidence')
    return True
