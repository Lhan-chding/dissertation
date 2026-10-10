"""SR-F1 native Qwen runtime. Importing this module never loads model weights.

The old F2 modules are reused only for parameter ownership and native cache
mechanics; their prompts, samplers, gates, adapters and scientific state are not used.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import errno
import hashlib
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from mm_core.vl_runtime import (
    CHAT_TEMPLATE_KWARGS,
    GENERATION,
    QwenRuntime,
    hash_json,
    linear_kernel_identity,
    seed_all,
    validate_model_config,
)
from mm_dev.runtime import (
    actual_cuda_identity,
    atomic_json,
    bounded_path,
    cache_compatible_autograd,
    configure_audited_backend,
    file_hash,
    functional_training_cache,
    read_json,
    reference_parameters,
)
from mm_dev.runtime import (
    copy_parameters as copy_parameters,
)

from .json_protocol import (
    AMENDMENT_ID,
    PREFILL,
    PREFILL_TOKEN_ID,
    BalancedJSONStop,
    amendment_enabled,
    decode_generated,
    decoded_protocol,
    first_balanced_token,
    validate_record,
)

CANVAS = (1024, 768)
TARGET_PIXELS = 786432
PLAN_ID = "SR-F1-20261009"


def _activation_directory(path, *, create=False):
    """Check every component before touching the authenticated scratch location."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise PermissionError("Activation scratch must be an absolute canonical path")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise PermissionError("Activation scratch cannot follow a symlink")
        if create:
            current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise PermissionError("Activation scratch component is not a directory")
    return path


def _activation_failure_receipt(directory, error, statistics):
    """Best effort metadata only; a failed diagnostic must not hide its cause."""
    if directory is None:
        return
    try:
        directory = _activation_directory(directory)
        atomic_json(
            directory / f"failure-{os.getpid()}-{time.time_ns()}.json",
            dict(
                status="ACTIVATION_IO_TECHNICAL_FAILURE",
                pid=os.getpid(),
                attempt_id=os.environ.get("SR_F1_ATTEMPT_ID"),
                error_type=type(error).__name__,
                error=str(error),
                errno=getattr(error, "errno", None),
                notes=list(getattr(error, "__notes__", [])),
                activation_offload=copy.deepcopy(statistics),
            ),
            exclusive=True,
        )
    except BaseException as secondary:
        error.add_note(f"Activation failure receipt also failed: {secondary!r}")


class _DiskActivation:
    """One saved-tensor reference; retain_graph keeps this reference alive."""

    def __init__(self, store, offset, size, sha256):
        self.store, self.offset, self.size, self.sha256 = store, offset, size, sha256
        store.references += 1

    def __del__(self):
        store = getattr(self, "store", None)
        if store is not None:
            try:
                store.release()
            except BaseException as error:
                store.statistics["cleanup_error"] = repr(error)
                _activation_failure_receipt(store.path.parent, error, store.statistics)


class _ActivationSpillFile:
    """Private append-only scratch, shared by this forward's saved disk tensors."""

    chunk_bytes = 32 << 20
    flush_bytes = 64 << 20

    safety_reserve_bytes = 10 << 30

    def __init__(self, output_root, statistics, *, spill_directory=None, quota_root=None):
        if output_root is None:
            raise PermissionError("Activation spill requires an explicit runtime output_root")
        root = Path(output_root).resolve(strict=True)
        self.statistics, self.offset = statistics, 0
        self.quota_root = None
        self.quota_baseline_bytes = None
        if (spill_directory is None) != (quota_root is None):
            raise PermissionError("Activation scratch and quota root must be supplied together")
        if spill_directory is not None:
            self.quota_root = _activation_directory(quota_root)
            expected = self.quota_root / "louis-ssvc" / (root.name + "_activation_offload")
            if Path(spill_directory) != expected:
                raise PermissionError(
                    "Activation scratch is not the dedicated current-run directory"
                )
            # Capacity is checked before creating a directory or scratch file.
            self._check_capacity(self.flush_bytes)
            directory = _activation_directory(expected, create=True)
            if directory.stat().st_uid != os.geteuid() or directory.stat().st_mode & 0o077:
                raise PermissionError(
                    "Activation scratch leaf must be private and owned by this user"
                )
        else:
            directory = root
            for component in ("technical_scratch", "activation_offload"):
                directory = directory / component
                if directory.is_symlink():
                    raise PermissionError("Activation scratch cannot follow a symlink")
                directory.mkdir(mode=0o700, exist_ok=True)
        self.fd, filename = tempfile.mkstemp(prefix=f"{os.getpid()}-", suffix=".bin", dir=directory)
        self.path, self.statistics = Path(filename), statistics
        self.offset = self.unflushed_bytes = self.references = 0
        self.finished = self.closed = False
        self.lock = threading.RLock()
        statistics.update(spill_file_created=True, spill_path=str(self.path))

    def _check_capacity(self, required_bytes):
        if self.quota_root is None:
            return
        _activation_directory(self.quota_root)
        try:
            maximum = int(os.getxattr(self.quota_root, "ceph.quota.max_bytes"))
            used = int(os.getxattr(self.quota_root, "ceph.dir.rbytes"))
            filesystem = os.statvfs(self.quota_root)
        except (OSError, ValueError, AttributeError) as error:
            raise PermissionError("Cannot verify activation Ceph quota and free space") from error
        if maximum <= 0 or used < 0:
            raise PermissionError("Activation Ceph quota must have a positive authenticated limit")
        if self.quota_baseline_bytes is None:
            self.quota_baseline_bytes = used
        # Ceph recursive usage is asynchronous. Account for our own writes even
        # when a fresh xattr has not yet caught up with this file's growth.
        effective_used = max(used, self.quota_baseline_bytes + self.offset)
        available = filesystem.f_bavail * filesystem.f_frsize
        effective_available = min(maximum - effective_used, available)
        check_index = self.statistics.get("capacity_check_count", 0) + 1
        observation = dict(
            check_index=check_index,
            time_ns=time.time_ns(),
            quota_root=str(self.quota_root),
            quota_max_bytes=maximum,
            quota_used_bytes=used,
            effective_quota_used_bytes=effective_used,
            statvfs_available_bytes=available,
            effective_available_bytes=effective_available,
            spill_written_bytes=self.offset,
            requested_window_bytes=required_bytes,
            safety_reserve_bytes=self.safety_reserve_bytes,
        )
        retained = self.statistics.setdefault("capacity_observations", [])
        first = retained[0] if retained else observation
        minimum = min(
            [*retained, observation],
            key=lambda item: (item["effective_available_bytes"], item["check_index"]),
        )
        # Full quota checks still run for every window. The durable per-sequence
        # ledger needs only first/latest/worst, not thousands of duplicate rows.
        selected = {item["check_index"]: item for item in (first, minimum, observation)}
        retained[:] = [selected[index] for index in sorted(selected)]
        self.statistics["capacity_check_count"] = check_index
        self.statistics["minimum_available_bytes"] = minimum["effective_available_bytes"]
        if effective_available < self.safety_reserve_bytes + required_bytes:
            raise OSError(errno.ENOSPC, "Activation spill would consume protected quota headroom")

    def _drop_cache(self, offset, size):
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(self.fd, offset, size, os.POSIX_FADV_DONTNEED)

    def _flush(self):
        if self.unflushed_bytes:
            getattr(os, "fdatasync", os.fsync)(self.fd)
            self._drop_cache(0, 0)
            self.unflushed_bytes = 0

    def append(self, flat):
        """Write exact dtype bytes in bounded chunks; no pickle or numeric conversion."""
        import torch

        with self.lock:
            if self.closed or self.finished:
                raise RuntimeError("Activation spill append after forward completion")
            begin, size = self.offset, flat.numel() * flat.element_size()
            digest = hashlib.sha256()
            data = flat.view(torch.uint8)
            try:
                os.lseek(self.fd, self.offset, os.SEEK_SET)
                for start in range(0, size, self.chunk_bytes):
                    width = min(self.chunk_bytes, size - start)
                    if self.unflushed_bytes and self.unflushed_bytes + width > self.flush_bytes:
                        self._flush()
                    if not self.unflushed_bytes:
                        self._check_capacity(max(self.flush_bytes, width))
                    chunk = data[start : start + self.chunk_bytes].cpu()
                    view = memoryview(chunk.numpy())
                    if os.write(self.fd, view) != len(view):
                        raise OSError("Short activation spill write")
                    digest.update(view)
                    self.offset += len(view)
                    self.unflushed_bytes += len(view)
                    if self.unflushed_bytes >= self.flush_bytes:
                        self._flush()
                record = _DiskActivation(self, begin, size, digest.hexdigest())
                self.statistics["disk_storage_bytes"] += size
                self.statistics["disk_saved_tensors"] += 1
                return record
            except BaseException as error:
                self.abort(error)
                raise

    def restore(self, record, metadata):
        import torch

        with self.lock:
            if self.closed:
                raise RuntimeError("Activation spill was closed before its graph was released")
            try:
                storage = torch.empty(
                    metadata["elements"], dtype=metadata["dtype"], device=metadata["device"]
                )
                data, digest = storage.view(torch.uint8), hashlib.sha256()
                os.lseek(self.fd, record.offset, os.SEEK_SET)
                for start in range(0, record.size, self.chunk_bytes):
                    size = min(self.chunk_bytes, record.size - start)
                    buffer = bytearray(size)
                    if os.readv(self.fd, [buffer]) != size:
                        raise OSError("Short activation spill read")
                    digest.update(buffer)
                    data[start : start + size].copy_(torch.frombuffer(buffer, dtype=torch.uint8))
                if digest.hexdigest() != record.sha256:
                    raise OSError("Activation spill checksum mismatch")
                self._drop_cache(record.offset, record.size)
                self.statistics["disk_read_bytes"] += record.size
                return storage
            except BaseException as error:
                self.abort(error)
                raise

    def finish_forward(self):
        with self.lock:
            if self.closed:
                return
            try:
                self._flush()
                self.finished = True
                if not self.references:
                    self._close()
            except BaseException as error:
                self.abort(error)
                raise

    def release(self):
        with self.lock:
            self.references -= 1
            if self.finished and not self.references and not self.closed:
                self._close()

    def _close(self):
        if self.closed:
            return
        # Do not retry an indeterminate close and risk closing a reused fd.
        self.closed = True
        close_error = None
        try:
            os.close(self.fd)
        except BaseException as error:
            close_error = error
        # A delayed close error can arrive after the descriptor has closed.
        # Removing our private derived file is independent of that result.
        try:
            self.path.unlink()
        except BaseException as error:
            if close_error is None:
                raise
            close_error.add_note(f"Activation scratch unlink also failed: {error!r}")
        else:
            self.statistics["spill_file_cleaned"] = True
        if close_error is not None:
            raise close_error

    def abort(self, error=None):
        with self.lock:
            self.statistics["spill_aborted"] = True
            try:
                self._close()
            except BaseException as secondary:
                self.statistics["cleanup_error"] = repr(secondary)
                if error is None:
                    raise
                error.add_note(f"Activation scratch cleanup also failed: {secondary!r}")
            if error is not None:
                _activation_failure_receipt(self.path.parent, error, self.statistics)


class _GPUActivation:
    """A saved snapshot lives until the last retained autograd graph releases it."""

    def __init__(self, tensor, statistics, device_statistics):
        self.tensor = tensor
        self.statistics, self.device_statistics = statistics, device_statistics
        self.size = tensor.numel() * tensor.element_size()
        device_statistics["live_bytes"] += self.size
        device_statistics["peak_live_bytes"] = max(
            device_statistics["peak_live_bytes"], device_statistics["live_bytes"]
        )

    def __del__(self):
        # Dropping the tensor returns the allocation to PyTorch's reusable cache;
        # do not run empty_cache or synchronize once per tiny saved activation.
        self.tensor = None
        self.device_statistics["live_bytes"] -= self.size
        self.device_statistics["released_tensors"] += 1
        self.statistics["gpu_released_bytes"] += self.size


class SavedActivationOffload:
    """Bound CPU activation storage and spill exact bytes to authenticated scratch.

    Native CUDA forward/backward kernels and the differentiable recurrent cache
    are unchanged. Parameter-storage views stay resident: copying every saved
    base-weight view once per decoded token would exhaust host memory instead.
    The detach below is only the saved-tensor packing API's byte snapshot; it
    does not detach any forward value, cache state, or gradient edge.
    """

    policy = "saved_activation_cpu48g_lossless_disk_parameter_resident_v2"
    gpu_policy = "saved_activation_auxgpu80g_cpu48g_lossless_disk_parameter_resident_v3"
    default_cpu_budget_bytes = 48 << 30
    default_gpu_budget_bytes = 80 << 30

    def __init__(
        self,
        model,
        *,
        output_root=None,
        cpu_budget_bytes=default_cpu_budget_bytes,
        spill_directory=None,
        quota_root=None,
        gpu_devices=(),
        gpu_budget_bytes=default_gpu_budget_bytes,
    ):
        import torch

        if type(cpu_budget_bytes) is not int or cpu_budget_bytes < 0:
            raise ValueError("Activation CPU budget must be a nonnegative integer")
        if type(gpu_budget_bytes) is not int or gpu_budget_bytes <= 0:
            raise ValueError("Activation GPU budget must be a positive integer")
        gpu_devices = tuple(gpu_devices)
        if gpu_devices and (
            any(type(index) is not int for index in gpu_devices)
            or gpu_devices != tuple(range(1, len(gpu_devices) + 1))
            or len(gpu_devices) > 4
            or torch.cuda.device_count() != len(gpu_devices) + 1
        ):
            raise PermissionError("Activation GPUs must be the allocated auxiliary CUDA devices")
        self.gpu_devices, self.gpu_budget_bytes = gpu_devices, gpu_budget_bytes
        self.output_root, self.cpu_budget_bytes = output_root, cpu_budget_bytes
        self.spill_directory, self.quota_root = spill_directory, quota_root
        self.parameter_storages = {self._storage_key(parameter) for parameter in model.parameters()}
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
        self.store = None
        self.statistics = dict(
            policy=self.gpu_policy if gpu_devices else self.policy,
            pin_memory=False,
            saved_activation_tensors=0,
            saved_activation_bytes=0,
            largest_saved_activation_bytes=0,
            resident_parameter_views=0,
            resident_parameter_view_bytes=0,
            unpacked_activation_tensors=0,
            unpacked_activation_bytes=0,
            cpu_budget_bytes=cpu_budget_bytes,
            cpu_storage_bytes=0,
            gpu_storage_bytes=0,
            gpu_saved_tensors=0,
            gpu_read_bytes=0,
            gpu_released_bytes=0,
            gpu_devices={
                str(index): dict(
                    budget_bytes=gpu_budget_bytes,
                    storage_bytes=0,
                    saved_tensors=0,
                    read_bytes=0,
                    live_bytes=0,
                    peak_live_bytes=0,
                    released_tensors=0,
                )
                for index in gpu_devices
            },
            disk_storage_bytes=0,
            disk_saved_tensors=0,
            disk_read_bytes=0,
            spill_file_created=False,
            spill_file_cleaned=False,
            spill_aborted=False,
            io_chunk_bytes=_ActivationSpillFile.chunk_bytes,
            capacity_observations=[],
            capacity_check_count=0,
            minimum_available_bytes=None,
        )

    @staticmethod
    def _storage_key(tensor):
        storage = tensor.untyped_storage()
        return tensor.device, storage.data_ptr(), storage.nbytes()

    def pack(self, tensor):
        import torch

        size = tensor.numel() * tensor.element_size()
        if self._storage_key(tensor) in self.parameter_storages:
            self.statistics["resident_parameter_views"] += 1
            self.statistics["resident_parameter_view_bytes"] += size
            # Hooks bypass PyTorch's ordinary saved-tensor version check. Keep
            # an explicit check for the resident views, which are not snapshots.
            saved = tensor.detach()
            return "parameter", saved, saved._version
        if tensor.layout != torch.strided or tensor.is_conj() or tensor.is_neg():
            raise PermissionError("Activation offload requires ordinary strided tensor storage")
        shape, stride = tuple(tensor.shape), tuple(tensor.stride())
        if any(step < 0 for step in stride):
            raise PermissionError("Negative-stride activation storage is unsupported")
        elements = (
            0
            if not tensor.numel()
            else 1
            + sum((dimension - 1) * step for dimension, step in zip(shape, stride, strict=True))
        )
        metadata = dict(
            device=tensor.device, dtype=tensor.dtype, shape=shape, stride=stride, elements=elements
        )
        flat = torch.as_strided(tensor.detach(), (elements,), (1,), tensor.storage_offset())
        storage_bytes = elements * tensor.element_size()
        packed = None
        # Stable least-stored placement exercises the allocated auxiliary cards
        # without asynchronous streams or any change to compute/RNG ordering.
        for index in sorted(
            self.gpu_devices,
            key=lambda device: self.statistics["gpu_devices"][str(device)]["storage_bytes"],
        ):
            device_statistics = self.statistics["gpu_devices"][str(index)]
            if device_statistics["storage_bytes"] + storage_bytes > self.gpu_budget_bytes:
                continue
            # Synchronous copy preserves the exact storage span, including holes
            # in noncontiguous views. Only this hook snapshot moves; the model's
            # values and full recurrent graph remain on the original device.
            saved = flat.to(device=f"cuda:{index}", copy=True, non_blocking=False)
            packed = (
                "gpu_activation",
                metadata,
                _GPUActivation(saved, self.statistics, device_statistics),
            )
            device_statistics["storage_bytes"] += storage_bytes
            device_statistics["saved_tensors"] += 1
            self.statistics["gpu_storage_bytes"] += storage_bytes
            self.statistics["gpu_saved_tensors"] += 1
            break
        if packed is not None:
            pass
        elif self.statistics["cpu_storage_bytes"] + storage_bytes <= self.cpu_budget_bytes:
            saved = flat.to(device="cpu", copy=True)
            self.statistics["cpu_storage_bytes"] += storage_bytes
            packed = "activation", metadata, saved
        else:
            if self.store is None:
                self.store = _ActivationSpillFile(
                    self.output_root,
                    self.statistics,
                    spill_directory=self.spill_directory,
                    quota_root=self.quota_root,
                )
            packed = "disk_activation", metadata, self.store.append(flat)
        self.statistics["saved_activation_tensors"] += 1
        self.statistics["saved_activation_bytes"] += size
        self.statistics["largest_saved_activation_bytes"] = max(
            self.statistics["largest_saved_activation_bytes"], size
        )
        return packed

    def unpack(self, packed):
        kind, first, second = packed
        if kind == "parameter":
            if first._version != second:
                error = RuntimeError(
                    "Resident parameter changed before activation-offload backward"
                )
                self.abort(error)
                raise error
            return first
        try:
            if kind == "disk_activation":
                saved = second.store.restore(second, first)
            elif kind == "gpu_activation":
                saved = second.tensor.to(first["device"], copy=True, non_blocking=False)
                self.statistics["gpu_read_bytes"] += second.size
                second.device_statistics["read_bytes"] += second.size
            else:
                saved = second.to(first["device"])
            result = saved.as_strided(first["shape"], first["stride"])
            self.statistics["unpacked_activation_tensors"] += 1
            self.statistics["unpacked_activation_bytes"] += result.numel() * result.element_size()
            return result
        except BaseException as error:
            self.abort(error)
            raise

    def __enter__(self):
        self.hooks.__enter__()
        return self

    def __exit__(self, *args):
        self.hooks.__exit__(*args)
        if self.store is not None:
            if args[0] is not None:
                self.store.abort(args[1])
            else:
                self.store.finish_forward()

    def abort(self, error=None):
        if self.store is not None:
            self.store.abort(error)
        elif error is not None and self.gpu_devices:
            _activation_failure_receipt(self.spill_directory, error, self.statistics)


def generation_recipe(*, max_new_tokens=768, do_sample=True):
    if max_new_tokens not in (128, 768) or type(do_sample) is not bool:
        raise ValueError("Only registered 128/768 token output channels are permitted")
    return {
        **GENERATION,
        "do_sample": do_sample,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": max_new_tokens,
        "output_logits": True,
        "output_scores": True,
    }


class SRRuntime(QwenRuntime):
    def __init__(
        self,
        model_path,
        adapter_path=None,
        *,
        device="cuda:0",
        dtype="bfloat16",
        attention_backend="eager",
        account=None,
        protocol_amendment=None,
        output_root=None,
        spill_directory=None,
        quota_root=None,
    ):
        self.output_root = (
            Path(output_root).resolve(strict=True) if output_root is not None else None
        )
        self.activation_spill_directory, self.activation_quota_root = spill_directory, quota_root
        amended = amendment_enabled(protocol_amendment)
        self.protocol_amendment = copy.deepcopy(protocol_amendment)
        if dtype != "bfloat16" or attention_backend != "eager":
            raise PermissionError("SR-F1 requires BF16 base and native eager attention")
        super().__init__(
            model_path,
            adapter_path,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
            account=account,
        )
        from transformers import AutoProcessor, GenerationConfig

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=TARGET_PIXELS,
            max_pixels=TARGET_PIXELS,
        )
        self.generation_config = GenerationConfig(
            **generation_recipe(),
            eos_token_id=self.eos_ids,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
        )
        self.identity.update(
            processor_hash=hash_json(self.processor.to_dict()),
            tokenizer_hash=hash_json(self.processor.tokenizer.get_vocab()),
            chat_template_hash=hash_json(self.processor.chat_template),
            generation_config_expanded=self.generation_config.to_dict(),
            canvas_pixels=list(CANVAS),
            processor_target_pixels=TARGET_PIXELS,
        )
        if amended:
            self.identity["protocol_amendment"] = copy.deepcopy(self.protocol_amendment)

    @classmethod
    def processor_only(cls, model_path, *, protocol_amendment=None):
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor, GenerationConfig

        instance = cls.__new__(cls)
        amended = amendment_enabled(protocol_amendment)
        instance.protocol_amendment = copy.deepcopy(protocol_amendment)
        instance.device = "cpu"
        instance.model_path = str(Path(model_path).resolve())
        architecture = validate_model_config(read_json(Path(model_path) / "config.json"))
        instance.processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=TARGET_PIXELS,
            max_pixels=TARGET_PIXELS,
        )
        instance.model = SimpleNamespace(
            config=AutoConfig.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=False
            )
        )
        kernels = linear_kernel_identity()
        instance.identity = {
            **architecture,
            "linear_kernel_identity": kernels,
            "linear_kernel_identity_hash": hash_json(kernels),
            "processor_hash": hash_json(instance.processor.to_dict()),
            "tokenizer_hash": hash_json(instance.processor.tokenizer.get_vocab()),
            "chat_template_hash": hash_json(instance.processor.chat_template),
            "canvas_pixels": list(CANVAS),
            "processor_target_pixels": TARGET_PIXELS,
        }
        generation_file = Path(model_path) / "generation_config.json"
        stored_generation = (
            GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
            if generation_file.is_file()
            else GenerationConfig.from_model_config(instance.model.config)
        )
        eos = stored_generation.eos_token_id
        instance.eos_ids = eos if isinstance(eos, list) else [eos]
        instance.generation_config = GenerationConfig(
            **generation_recipe(),
            eos_token_id=instance.eos_ids,
            pad_token_id=instance.processor.tokenizer.pad_token_id,
            bos_token_id=instance.processor.tokenizer.bos_token_id,
        )
        instance.identity.update(
            generation_config_expanded=instance.generation_config.to_dict(),
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
            chat_template_kwargs_hash=hash_json(CHAT_TEMPLATE_KWARGS),
        )
        if amended:
            instance.identity["protocol_amendment"] = copy.deepcopy(instance.protocol_amendment)
        return instance

    def prepare(self, row, run_root, *, protocol="evidence_answer"):
        from .data import model_input

        safe = model_input(row, protocol=protocol)
        return self.prepare_text(
            safe["text"],
            safe["image_file"],
            run_root,
            expected_hash=row.get("image_sha256"),
            protocol=protocol,
        )

    def prepare_text(
        self, text, image_path, run_root, *, expected_hash=None, protocol="evidence_answer"
    ):
        """Only explicit text and pixels enter the processor; no gold sidecar is read."""
        from PIL import Image

        if not isinstance(text, str) or not text:
            raise ValueError("Nonempty registered prompt text required")
        if protocol not in ("evidence_answer", "answer_only", "plain_answer"):
            raise PermissionError("Unknown output protocol")
        amended = amendment_enabled(getattr(self, "protocol_amendment", None)) and (
            protocol == "evidence_answer"
        )
        root = Path(run_root).resolve()
        rgb, source_size, image_hash = None, None, None
        content = []
        if image_path is not None:
            path = Path(image_path)
            path = path.resolve() if path.is_absolute() else bounded_path(root, path)
            if not path.is_relative_to(root):
                raise PermissionError("Image path escapes run root")
            image_hash = file_hash(path)
            if expected_hash is not None and image_hash != expected_hash:
                raise PermissionError("Image bytes differ from registered identity")
            with Image.open(path) as original:
                source_size = original.size
                rgb = original.convert("RGB")
            # PIL object is represented by an image token by the native template.
            # A filesystem path or qid is never placed in the template's text.
            content.append({"type": "image", "image": rgb})
        content.append({"type": "text", "text": text})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        if not chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
            raise PermissionError("Native non-thinking assistant template differs")
        kwargs = dict(text=[chat], padding=False, return_tensors="pt")
        if rgb is not None:
            kwargs["images"] = [rgb]
        inputs = self.processor(**kwargs)
        base_prompt_tokens = inputs["input_ids"]
        if amended:
            tokenizer = self.processor.tokenizer
            if tokenizer.encode(PREFILL, add_special_tokens=False) != [PREFILL_TOKEN_ID]:
                raise PermissionError("Registered assistant prefill is not token 90")
            chat += PREFILL
            inputs = self.processor(**{**kwargs, "text": [chat]})
            amended_ids = inputs["input_ids"]
            if (
                amended_ids.shape != (1, base_prompt_tokens.shape[-1] + 1)
                or amended_ids[0, -1].item() != PREFILL_TOKEN_ID
                or not bool((amended_ids[:, :-1] == base_prompt_tokens).all())
            ):
                raise PermissionError("Assistant prefill changed the native prompt token boundary")
        ids = inputs["input_ids"]
        if ids.shape[0] != 1:
            raise PermissionError("One independent context per generation is required")
        image_id = getattr(self.model.config, "image_token_id", None)
        mm_types = inputs.get("mm_token_type_ids")
        if rgb is None and mm_types is None:
            # Native text-only processor may omit this field. The hybrid cached
            # path requires explicit zeros to preserve the same generation API.
            inputs["mm_token_type_ids"] = ids.new_zeros(ids.shape)
            mm_types = inputs["mm_token_type_ids"]
        if mm_types is None or mm_types.shape != ids.shape:
            raise RuntimeError("Native multimodal token types are missing")
        grid, processed_size, pixel_hash, visual_tokens = [], None, None, 0
        if rgb is not None:
            if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
                raise RuntimeError("Processor did not route real image pixels")
            grid = inputs["image_grid_thw"].tolist()
            ip = self.processor.image_processor
            patch, merge = int(ip.patch_size), int(ip.merge_size)
            if len(grid) != 1 or grid[0][0] != 1:
                raise PermissionError("Exactly one still image is required")
            processed_size = [grid[0][2] * patch, grid[0][1] * patch]
            if source_size == CANVAS and tuple(processed_size) != CANVAS:
                raise PermissionError("Native processor resized the fixed synthetic canvas")
            visual_tokens = int((ids == image_id).sum()) if image_id is not None else 0
            if visual_tokens <= 0 or visual_tokens != math.prod(grid[0]) // merge**2:
                raise RuntimeError("Image grid and visual token counts disagree")
            if not bool(((mm_types == 1) == (ids == image_id)).all()):
                raise RuntimeError("Native image token flags disagree")
            pixels = inputs["pixel_values"].detach().cpu().contiguous()
            pixel_hash = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        elif bool((mm_types != 0).any()) or "pixel_values" in inputs:
            raise RuntimeError("Text-only diagnostic unexpectedly contains visual inputs")
        tensor_hashes = {
            name: state_hash(value) for name, value in inputs.items() if hasattr(value, "detach")
        }
        routing = dict(
            source_image_sha256=image_hash,
            source_size=list(source_size) if source_size else None,
            processed_size=processed_size,
            image_grid_thw=grid,
            processed_pixel_sha256=pixel_hash,
            image_token_count=visual_tokens,
            image_token_id=image_id,
            processor_target_pixels=TARGET_PIXELS,
            native_input_tensor_hashes=tensor_hashes,
            input_tensor_hash=hash_json(tensor_hashes),
            chat_text_sha256=hashlib.sha256(chat.encode()).hexdigest(),
            input_ids_sha256=hash_json(ids.tolist()),
            processor_hash=self.identity["processor_hash"],
            chat_template_hash=self.identity["chat_template_hash"],
            chat_template_kwargs=dict(CHAT_TEMPLATE_KWARGS),
            pixels_per_count=(515 / 100 * processed_size[1] / 768)
            if source_size == CANVAS
            else None,
        )
        if amended:
            routing.update(
                protocol=protocol,
                protocol_amendment=copy.deepcopy(self.protocol_amendment),
                protocol_amendment_id=AMENDMENT_ID,
                assistant_prefill=PREFILL,
                assistant_prefill_token_ids=[PREFILL_TOKEN_ID],
                base_prompt_token_count=base_prompt_tokens.shape[-1],
            )
        return dict(inputs=inputs.to(self.device), routing=routing, chat_text=chat)

    def encode_completion(self, text, *, eos=True):
        if not amendment_enabled(getattr(self, "protocol_amendment", None)):
            return super().encode_completion(text, eos=eos)
        # Amended gold is the complete JSON object. The prefill is already in
        # the prompt; reject a tokenizer merge instead of teaching another `{`.
        if not isinstance(text, str) or not text.startswith(PREFILL):
            raise PermissionError("Amended gold must include its registered opening prefill")
        tokens = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if not tokens or tokens[0] != PREFILL_TOKEN_ID:
            raise PermissionError("Gold prefill must remain the independent token 90")
        if decoded_protocol(text[1:])["format_protocol_error"]:
            raise PermissionError("Amended gold must terminate at a balanced JSON boundary")
        return tokens[1:]

    def current_adapter_identity(self):
        parameters = {
            name: p.detach() for name, p in self.model.named_parameters() if "lora_" in name
        }
        return dict(
            model_id=self.identity.get("model_id"),
            step=self.identity.get("step"),
            adapter_path=self.adapter_path,
            base_model_weights_hash=self.identity.get("base_model_weights_hash"),
            adapter_parameter_hash=state_hash(parameters) if parameters else None,
            published_adapter_hash=self.identity.get("adapter_hash"),
        )

    def stable_model_identity(self):
        result = {
            key: value
            for key, value in self.identity.items()
            if key not in {"hardware", "active_adapter"}
        }
        hardware = self.identity.get("hardware", {})
        result["hardware"] = {
            key: hardware[key]
            for key in (
                "cuda_name",
                "cuda_total_memory_bytes",
                "compute_capability",
                "cuda_runtime",
                "driver_version",
            )
            if key in hardware
        }
        return result

    def training_generation_config(self):
        return copy.deepcopy(self.generation_config)

    def generate_training(self, row, run_root, seed, *, on_completion=None):
        return self.generate(row, run_root, seed, on_completion=on_completion)

    def generate(
        self,
        row=None,
        run_root=None,
        seed=None,
        *,
        text=None,
        image_path=None,
        generation=None,
        protocol="evidence_answer",
        on_completion=None,
        **unused,
    ):
        if unused:
            raise TypeError("Unknown generation options: " + ",".join(unused))
        if run_root is None or type(seed) is not int:
            raise ValueError("Run root and explicit integer sample seed are required")
        if row is not None and text is not None:
            raise ValueError("Use one prompt interface per generation")
        prepared = (
            self.prepare(row, run_root, protocol=protocol)
            if row is not None
            else (self.prepare_text(text, image_path, run_root, protocol=protocol))
        )
        amended = prepared["routing"].get("protocol_amendment_id") == AMENDMENT_ID
        config = self.training_generation_config()
        if generation:
            permitted = {
                "max_new_tokens",
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "num_return_sequences",
            }
            if set(generation) - permitted:
                raise PermissionError("Unregistered generation options")
            if generation.get("num_return_sequences", 1) != 1:
                raise PermissionError("Exactly one independent completion per call is required")
            for key in ("temperature", "top_p", "top_k"):
                if (
                    key in generation
                    and generation[key] != {"temperature": 1, "top_p": 1, "top_k": 0}[key]
                ):
                    raise PermissionError("Sampling channel differs from preregistration")
            recipe = generation_recipe(
                max_new_tokens=generation.get("max_new_tokens", 768),
                do_sample=generation.get("do_sample", True),
            )
            for key, value in recipe.items():
                setattr(config, key, value)
        self.reserve("completion_attempts", 1, seed=seed, qid=(row or {}).get("qid"))
        rng, before, started = capture_rng(), self.image_calls, time.perf_counter()
        try:
            seed_all(seed)
            self.model.eval()
            prompt_len = prepared["inputs"]["input_ids"].shape[-1]
            generate_kwargs = {}
            stop = None
            if amended:
                from transformers import StoppingCriteriaList

                stop = BalancedJSONStop(self.processor.tokenizer, prompt_len)
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList([stop])
            with self.torch.no_grad():
                result = self.model.generate(
                    **prepared["inputs"], generation_config=config, **generate_kwargs
                )
            tokens = result.sequences[0, prompt_len:].cpu().tolist()
            errors, logps, sampler = [], [], []
            generated_text = decode_generated(self.processor.tokenizer, tokens)
            amended_record = {}
            if amended:
                count, cut = first_balanced_token(self.processor.tokenizer, tokens)
                amended_record = {
                    **decoded_protocol(generated_text),
                    "balanced_token_count": count,
                }
                if (
                    (count, cut) != (stop.balanced_token_count, stop.balanced_cut_char)
                    or cut != amended_record["balanced_cut_char"]
                    or (count is not None and count != len(tokens))
                ):
                    errors.append("BALANCED_STOP_REPLAY_MISMATCH")
                if count is None and (
                    not tokens
                    or (tokens[-1] not in self.eos_ids and len(tokens) != config.max_new_tokens)
                ):
                    errors.append("UNREGISTERED_UNBALANCED_TERMINATION")
            if not tokens or len(result.logits) != len(tokens) or len(result.scores) != len(tokens):
                errors.append("INCOMPLETE_RAW_SAMPLER_LOGITS")
            for token, raw, transformed in zip(tokens, result.logits, result.scores, strict=False):
                if not self.torch.equal(raw.float(), transformed.float()):
                    errors.append("UNREGISTERED_LOGITS_TRANSFORM")
                logps.append(float(raw[0].float().log_softmax(-1)[token]))
                sampler.append(float(transformed[0].float().log_softmax(-1)[token]))
            if not all(math.isfinite(v) for v in logps + sampler):
                errors.append("NONFINITE_SAMPLER_LOGPROB")
            requires_image = prepared["routing"]["image_token_count"] > 0
            if requires_image and self.image_calls <= before:
                errors.append("NATIVE_VISUAL_ENCODER_NOT_CALLED")
            routing = {
                **prepared["routing"],
                "generation_vision_forward_calls": self.image_calls - before,
            }
            record = dict(
                raw_text=generated_text,
                tokens=tokens,
                raw_tokens=tokens,
                seed=seed,
                old_logprobs=[v if math.isfinite(v) else None for v in logps],
                sampler_logprobs=[v if math.isfinite(v) else None for v in sampler],
                old_logprob_source="actual_generate_raw_logits_selected_token",
                generation_status="TECHNICAL_INVALID" if errors else "COMPLETE",
                technical_validation_errors=errors,
                sampling_hash=hash_json(config.to_dict()),
                sampling_parameters=config.to_dict(),
                prompt_token_count=prompt_len,
                completion_token_count=len(tokens),
                generation_seconds=time.perf_counter() - started,
                truncated=bool(
                    tokens
                    and tokens[-1] not in self.eos_ids
                    and len(tokens) == config.max_new_tokens
                ),
                finish_reason="eos" if tokens and tokens[-1] in self.eos_ids else "length",
                image_routing=routing,
                input_routing=routing,
                adapter_identity=self.current_adapter_identity(),
                model_identity=self.stable_model_identity(),
                runtime_hardware=self.identity.get("hardware"),
            )
            if amended:
                record.update(amended_record)
                if amended_record["balanced_cut_char"] is not None:
                    record.update(finish_reason="balanced", truncated=False)
                if not errors:
                    try:
                        validate_record(record, expected_amendment_id=AMENDMENT_ID)
                    except (PermissionError, TypeError, ValueError):
                        errors.append("AMENDED_RECORD_METADATA_MISMATCH")
                record["generation_status"] = "TECHNICAL_INVALID" if errors else "COMPLETE"
            if on_completion:
                on_completion(record)
            self.reserve("generated_tokens", len(tokens), seed=seed)
            if errors:
                raise RuntimeError("Invalid generation preserved: " + ";".join(errors))
            return record
        finally:
            restore_rng(rng)

    def reference_forward(self, prepared, tokens, reference):
        with reference_parameters(self.model, reference):
            return self.sequence_forward(prepared, tokens, purpose="training_reference")

    def sequence_forward(self, prepared, tokens, *, purpose, grad=False):
        """The identical native cache path is used for old/current/reference probabilities."""
        return self.cached_training_forward(prepared, tokens, purpose=purpose, grad=grad)

    def cached_training_forward(self, prepared, tokens, *, purpose, grad=False):
        """Teacher-force saved tokens through the sampler's native cached path.

        All prefix states retain their computation graph. Only cache containers
        change ownership; no kernel, precision, token, or sampling setting changes.
        Measurement/PROBE teacher forcing continues to use the audited core path.
        """
        torch = self.torch
        if not tokens:
            raise ValueError("Cannot score empty completion")
        if self.model.training:
            raise PermissionError("Training policy forward must retain eval/dropout-zero mode")
        self.reserve("extra_forward_sequences", 1, purpose=purpose, completion_tokens=len(tokens))
        kwargs = dict(prepared["inputs"])
        input_ids = kwargs.pop("input_ids")
        if input_ids.shape[0] != 1:
            raise PermissionError("SR-F1 teacher forcing requires one complete sequence")
        if kwargs.get("mm_token_type_ids") is None:
            raise RuntimeError("Teacher forcing requires original native multimodal token types")
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        cache = functional_training_cache(base.config)
        kwargs.update(past_key_values=cache, use_cache=True, logits_to_keep=1)
        kwargs["position_ids"] = base._prepare_position_ids_for_generation(input_ids, kwargs)
        before = self.image_calls
        selected, entropies = [], []
        calls = dict(
            prefill_model_forward_calls=0,
            decode_model_forward_calls=0,
            prefill_input_tokens=0,
            decode_input_tokens=0,
            completed_model_forward_calls=0,
        )
        complete = False
        offload = (
            SavedActivationOffload(
                self.model,
                output_root=getattr(self, "output_root", None),
                spill_directory=getattr(self, "activation_spill_directory", None),
                quota_root=getattr(self, "activation_quota_root", None),
                cpu_budget_bytes=getattr(
                    self,
                    "activation_cpu_budget_bytes",
                    SavedActivationOffload.default_cpu_budget_bytes,
                ),
                gpu_devices=getattr(self, "activation_gpu_devices", ()),
                gpu_budget_bytes=getattr(
                    self,
                    "activation_gpu_budget_bytes",
                    SavedActivationOffload.default_gpu_budget_bytes,
                ),
            )
            if grad
            else None
        )
        try:
            with (
                cache_compatible_autograd(self.model),
                torch.enable_grad() if grad else torch.no_grad(),
                offload if offload is not None else contextlib.nullcontext(),
            ):
                for index, token in enumerate(tokens):
                    # Native one-token conv updates write their input cache in place.
                    # Clone without detach so older graph nodes own unchanged storage.
                    for layer in cache.layers:
                        for state_idx, state in getattr(layer, "conv_states", {}).items():
                            if state is not None:
                                layer.conv_states[state_idx] = state.clone()
                    model_inputs = base.prepare_inputs_for_generation(
                        input_ids,
                        next_sequence_length=None if index == 0 else 1,
                        is_first_iteration=index == 0,
                        **kwargs,
                    )
                    phase = "prefill" if index == 0 else "decode"
                    calls[phase + "_model_forward_calls"] += 1
                    calls[phase + "_input_tokens"] += model_inputs["input_ids"].numel()
                    output = self.model(**model_inputs, return_dict=True)
                    calls["completed_model_forward_calls"] += 1
                    if (
                        output.past_key_values is not cache
                        or cache.get_seq_length() != input_ids.shape[1]
                    ):
                        raise RuntimeError("Native teacher-forcing cache was dropped or misaligned")
                    logp = output.logits[0, -1].float().log_softmax(-1)
                    selected.append(logp[token])
                    if not grad:
                        entropies.append(-(logp.exp() * logp).sum())
                    kwargs = base._update_model_kwargs_for_generation(
                        output, kwargs, is_encoder_decoder=False
                    )
                    input_ids = torch.cat([input_ids, input_ids.new_tensor([[token]])], dim=1)
            if prepared["routing"]["image_token_count"] and self.image_calls <= before:
                raise RuntimeError("Forward did not call visual encoder")
            result = dict(
                logprobs=torch.stack(selected),
                entropy=None if grad else torch.stack(entropies),
                vision_forward_calls=self.image_calls - before,
                cached_model_forward_calls=len(tokens),
                activation_offload=offload.statistics if offload is not None else None,
            )
            complete = True
            return result
        finally:
            # One durable accounting row per sequence, including failed forwards.
            # The sequence was reserved before execution; these are actual calls,
            # with no GPU-hour or token-budget acceptance threshold.
            primary = sys.exc_info()[1]
            try:
                if not complete and offload is not None:
                    offload.abort(primary)
                self.reserve(
                    "cached_training_model_forward_calls",
                    calls["prefill_model_forward_calls"] + calls["decode_model_forward_calls"],
                    purpose=purpose,
                    **calls,
                    requested_completion_tokens=len(tokens),
                    scored_completion_tokens=len(selected),
                    vision_forward_calls=self.image_calls - before,
                    status="COMPLETE" if complete else "TECHNICAL_FAILED",
                    activation_offload=copy.deepcopy(offload.statistics)
                    if offload is not None
                    else None,
                )
            except BaseException as secondary:
                if offload is not None:
                    offload.statistics["accounting_error"] = dict(
                        error_type=type(secondary).__name__,
                        error=str(secondary),
                        errno=getattr(secondary, "errno", None),
                    )
                if primary is None:
                    _activation_failure_receipt(
                        getattr(self, "activation_spill_directory", None),
                        secondary,
                        offload.statistics if offload is not None else {},
                    )
                    raise
                primary.add_note(f"Forward failure accounting also failed: {secondary!r}")
            if primary is not None:
                _activation_failure_receipt(
                    getattr(self, "activation_spill_directory", None),
                    primary,
                    offload.statistics if offload is not None else {},
                )


def runtime_account(root, task_id):
    """Durably record real attempted work; there is no GPU-hour stop budget."""
    import fcntl
    import json

    path = Path(root) / "accounting" / (task_id + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(kind, count, metadata):
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative integer accounting count required")
        record = dict(
            plan_id=PLAN_ID,
            task_id=task_id,
            kind=kind,
            count=count,
            metadata=metadata,
            pid=os.getpid(),
            time_ns=time.time_ns(),
            attempt_id=os.environ.get("SR_F1_ATTEMPT_ID"),
        )
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_UN)

    return reserve


def verified_adapter(root, model_id, step=96):
    root = Path(root)
    path = root / (
        "COMMON_START.json"
        if model_id == "SRF1_COMMON_START"
        else str(Path("states") / model_id / f"step-{step:02d}" / "IDENTITY.json")
    )
    entry = read_json(path)
    if entry.get("plan_id") != PLAN_ID or entry.get("status") != "VERIFIED":
        raise PermissionError("No verified SR-F1 adapter identity")
    if entry.get("freeze_sha256") != file_hash(root / "EXECUTION_FREEZE.json"):
        raise PermissionError("Adapter belongs to another execution freeze")
    observed = adapter_identity(root, entry["adapter_path"])
    if any(entry.get(key) != value for key, value in observed.items()):
        raise PermissionError("Published adapter bytes changed")
    return entry


def actual_srf1_cuda_identity(repair=None):
    """Inspect an authenticated compute GPU plus exact auxiliary-storage GPUs.

    The shared F2 single-device contract is deliberately left unchanged. The
    caller obtains repair only from verify_engine_multigpu_repair, never a raw
    JSON read. Dynamic free-memory observations are checks, not stable identity.
    """
    if repair is None:
        return actual_cuda_identity()
    import torch

    count = repair.get("gpu_count")
    if (
        type(count) is not int
        or not 2 <= count <= 5
        or type(repair.get("compute_device_index")) is not int
        or repair.get("compute_device_index") != 0
        or repair.get("storage_device_indices") != list(range(1, count))
        or any(type(index) is not int for index in repair.get("storage_device_indices", []))
        or repair.get("gpu_activation_budget_bytes") != 80 << 30
        or repair.get("cpu_activation_budget_bytes") != 48 << 30
        or repair.get("activation_storage") != "AUXILIARY_GPU_THEN_BOUNDED_CPU_EXACT_EXTERNAL_DISK"
        or not re.fullmatch(r"[0-9a-f]{64}", repair.get("repair_sha256") or "")
        or torch.cuda.device_count() != count
    ):
        raise PermissionError("SR-F1 auxiliary GPU allocation differs from authenticated repair")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    devices = [
        dict(
            zip(
                ("name", "uuid", "pci_bus_id", "driver_version", "memory_mib"),
                (part.strip() for part in row),
                strict=True,
            )
        )
        for row in csv.reader(query.stdout.splitlines())
        if row
    ]

    def normalize(value):
        return value.lower().removeprefix("gpu-").replace("-", "")

    observed = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        uuid = str(getattr(properties, "uuid", ""))
        matches = [device for device in devices if normalize(device["uuid"]) == normalize(uuid)]
        if (
            "RTX PRO 6000" not in properties.name.upper()
            or not uuid
            or len(matches) != 1
            or matches[0]["name"] != properties.name
        ):
            raise PermissionError("Allocated SR-F1 CUDA/driver GPU identity differs")
        if index and (
            torch.cuda.mem_get_info(index)[0] < repair["gpu_activation_budget_bytes"] + (4 << 30)
        ):
            raise PermissionError("Auxiliary GPU has insufficient memory for authenticated budget")
        observed.append(
            dict(
                index=index,
                role="compute" if index == 0 else "saved_activation_storage",
                cuda_name=properties.name,
                cuda_total_memory_bytes=properties.total_memory,
                cuda_uuid=uuid,
                compute_capability=[properties.major, properties.minor],
                **matches[0],
            )
        )
    if len({normalize(device["cuda_uuid"]) for device in observed}) != count:
        raise PermissionError("Auxiliary GPU allocation contains duplicate devices")
    return {
        **{key: value for key, value in observed[0].items() if key not in {"index", "role"}},
        "cuda_visible_device_count": count,
        "cuda_runtime": torch.version.cuda,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compute_device_index": 0,
        "auxiliary_storage_devices": observed[1:],
        "peer_access_from_compute": [
            bool(torch.cuda.can_device_access_peer(0, index)) for index in range(1, count)
        ],
        "engine_multigpu_repair_sha256": repair["repair_sha256"],
    }


def load_runtime(plan, root, *, state_id=None, step=96, account=None):
    from peft import PeftModel

    root = Path(root)
    identity = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    if (
        identity.get("model_revision") != plan["model"]["revision"]
        or identity.get("model_weights_hash") != plan["model"]["prior_composite_weight_hash"]
    ):
        raise PermissionError("SR-F1 requires the exact original untrained 9B snapshot")
    scratch, multigpu = {}, None
    if (root / "ENGINE_IO_REPAIR.json").exists():
        from .freeze import verify_engine_io_repair

        repair = verify_engine_io_repair(root)
        scratch = dict(
            spill_directory=repair["activation_spill_directory"], quota_root=repair["quota_root"]
        )
    if (root / "ENGINE_MULTIGPU_REPAIR.json").exists():
        from .freeze import verify_engine_multigpu_repair

        repair = verify_engine_multigpu_repair(root)
        task_id = os.environ.get("SR_F1_TASK_ID")
        task = read_json(root / "orchestration/REGISTRATION.json").get("tasks", {}).get(task_id)
        if not task or task.get("operation") not in {"engine", "train", "common_start", "evaluate"}:
            raise PermissionError("Auxiliary GPU runtime requires its registered task identity")
        if task["operation"] in {"engine", "train"}:
            multigpu = repair
    determinism = configure_audited_backend()
    hardware = actual_srf1_cuda_identity(multigpu)
    runtime = SRRuntime(
        identity["model_path"],
        account=account,
        protocol_amendment=plan.get("protocol_amendment"),
        output_root=root,
        **scratch,
    )
    if multigpu is not None:
        runtime.activation_gpu_devices = tuple(multigpu["storage_device_indices"])
        runtime.activation_gpu_budget_bytes = multigpu["gpu_activation_budget_bytes"]
        runtime.activation_cpu_budget_bytes = multigpu["cpu_activation_budget_bytes"]
    runtime.training_learning_rate = plan["training"]["lr"]
    runtime.verify_identity(identity)
    runtime.identity.update(
        hardware=hardware,
        determinism=determinism,
        base_model_weights_hash=identity["model_weights_hash"],
    )
    if state_id is not None:
        entry = verified_adapter(root, state_id, step)
        runtime.model = PeftModel.from_pretrained(
            runtime.model, bounded_path(root, entry["adapter_path"]), is_trainable=False
        )
        runtime.adapter_path = str(bounded_path(root, entry["adapter_path"]))
        runtime.identity.update(entry)
    return runtime


def load_for_evaluation(plan, root, model_id, step=96):
    return load_runtime(
        plan,
        root,
        state_id=model_id,
        step=step,
        account=runtime_account(root, f"eval-{model_id}-{step}"),
    )


def publish_adapter(root, runtime, model_id, *, step=None, extra=None):
    import shutil
    import tempfile

    from .contract import digest

    root = Path(root)
    relative = Path("states") / model_id
    if step is not None:
        relative /= f"step-{step:02d}"
    destination = bounded_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-adapter-", dir=destination.parent))
    try:
        runtime.model.save_pretrained(staging, safe_serialization=True)
        if destination.exists():
            # IDENTITY.json is a receipt, never part of the adapter payload hash.
            existing = {
                p.name: file_hash(p)
                for p in destination.iterdir()
                if p.is_file() and p.name != "IDENTITY.json"
            }
            new = {p.name: file_hash(p) for p in staging.iterdir() if p.is_file()}
            if existing != new:
                raise PermissionError("Existing published adapter differs from committed state")
        else:
            os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    files = {
        p.name: file_hash(p)
        for p in destination.iterdir()
        if p.is_file() and p.name != "IDENTITY.json"
    }
    entry = dict(
        plan_id=PLAN_ID,
        status="VERIFIED",
        model_id=model_id,
        step=step,
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        base_model_weights_hash=read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")[
            "model_weights_hash"
        ],
        adapter_path=str(relative),
        adapter_file_hashes=files,
        adapter_hash=hash_json(files),
        trainable_state_hash=state_hash(trainable_state(runtime.model)),
        **(extra or {}),
    )
    # match the generic adapter_identity hash convention.
    from mm_dev.contract import digest as adapter_digest

    entry["adapter_hash"] = adapter_digest(files)
    entry["model_hash"] = digest({"base": entry["base_model_weights_hash"], "adapter": files})
    marker = destination / "IDENTITY.json"
    if marker.exists():
        if read_json(marker) != entry:
            raise PermissionError("Published adapter receipt differs")
    else:
        atomic_json(marker, entry, exclusive=True)
    runtime.identity.update(entry)
    runtime.adapter_path = str(destination)
    return entry


def adapter_identity(root, relative):
    from mm_dev.contract import digest as adapter_digest

    path = bounded_path(root, relative)
    files = {
        p.name: file_hash(p) for p in path.iterdir() if p.is_file() and p.name != "IDENTITY.json"
    }
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise PermissionError("Missing serialized adapter payload")
    return dict(
        adapter_path=str(relative), adapter_file_hashes=files, adapter_hash=adapter_digest(files)
    )


def _coverage(record, task):
    from .json_protocol import score_record

    result = score_record(record, task)
    # The reference scorer separates legal entities/values from semantic coverage.
    return bool(result["L_json"] and result["L_answer"] and result["L_evidence"])


def format_panel(runtime, root, pool, label, boundary=None, *, samples_per_prompt=8):
    from .contract import digest, stable_seed
    from .data import load_inputs, load_tasks
    from .json_protocol import score_record, validate_record

    root = Path(root)
    inputs, tasks = load_inputs(root), load_tasks(root)
    qids = sorted(q for q, t in tasks.items() if t["pool"] == pool)
    if len(qids) != 32:
        raise PermissionError("FORMAT panel must contain exactly 32 registered prompts")
    amended = bool(getattr(runtime, "protocol_amendment", None))
    expected_samples = 16 if amended and pool == "FORMAT_CONFIRM" else 8
    if samples_per_prompt != expected_samples:
        raise PermissionError("FORMAT sampling count differs from the frozen protocol")
    directory = root / "engineering" / "format" / label
    records = []
    policy = state_hash(trainable_state(runtime.model))
    for qid in qids:
        for index in range(samples_per_prompt):
            if boundary and boundary["requested"]:
                from .training import LeaseEnding

                raise LeaseEnding("PREEMPTION_DURING_FORMAT_PANEL")
            path = directory / f"{qid}-{index}.json"
            seed = stable_seed(PLAN_ID, "FORMAT", pool, qid, index)
            expected = dict(
                qid=qid,
                sample_index=index,
                seed=seed,
                policy_hash=policy,
                input_hash=digest(inputs[qid]),
            )
            if amended:
                expected["protocol_amendment_sha256"] = file_hash(root / "AMENDMENT.json")
            if path.exists():
                record = read_json(path)
                if any(record.get(k) != v for k, v in expected.items()):
                    raise PermissionError("FORMAT raw identity changed")
            else:

                def persist(raw, path=path, expected=expected):
                    value = {**expected, **raw}
                    value["record_hash"] = digest(value)
                    atomic_json(path, value, exclusive=True)

                runtime.generate_training(inputs[qid], root, seed, on_completion=persist)
                record = read_json(path)
            if record.get("record_hash") != digest(
                {k: v for k, v in record.items() if k != "record_hash"}
            ):
                raise PermissionError("FORMAT raw response bytes changed")
            if record["generation_status"] != "COMPLETE":
                raise RuntimeError("FORMAT contains technical generation failure")
            validate_record(record, runtime.protocol_amendment["id"] if amended else None)
            record["score"] = score_record(record, tasks[qid])
            records.append(record)
    covered = sum(_coverage(r, tasks[r["qid"]]) for r in records)
    receipt = dict(
        panel=pool,
        label=label,
        samples_per_prompt=samples_per_prompt,
        policy_hash=policy,
        responses=len(records),
        covered=covered,
        coverage=covered / len(records),
        truncated=sum(r["truncated"] for r in records),
        coverage_uses_gold_accuracy=False,
        component_means={
            k: sum(r["score"][k] for r in records) / len(records) for k in ("A", "E", "P", "J")
        },
    )
    atomic_json(directory / "COVERAGE.json", receipt)
    return receipt


def activate_reviewed_format_repair(root):
    """Retain the original failure and authorize only its audited guard repair."""
    from .format_review import verify_format_review
    from .freeze import verify_technical_repair

    root = Path(root)
    repair = verify_technical_repair(root)
    review = verify_format_review(root)
    blocked = root / "FORMAT_AND_BRIDGE_RECEIPT.json"
    archived_name = "technical_incidents/format_guard_20261009/FORMAT_AND_BRIDGE_RECEIPT.json"
    archived = bounded_path(root, archived_name)
    if (
        read_json(blocked).get("reason") != "FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW"
        or read_json(blocked).get("bridge_executed") is not False
        or file_hash(blocked) != review["artifact_hashes"].get(archived_name)
        or blocked.read_bytes() != archived.read_bytes()
    ):
        raise PermissionError("Only the preserved, reviewed pre-bridge block can be resumed")
    if (
        (root / "COMMON_START.json").exists()
        or any((root / "engineering/bridge").rglob("*.json"))
        or any((root / "training").rglob("*.json"))
    ):
        raise PermissionError("Format guard repair cannot restart existing training")
    receipt = dict(
        plan_id=PLAN_ID,
        status="ACTIVATED",
        repair_sha256=repair["repair_sha256"],
        review_sha256=repair["review_sha256"],
        retained_block_path=archived_name,
        retained_block_sha256=file_hash(archived),
        original_format_records_reused=True,
        new_format_before_generations=0,
    )
    marker = root / "FORMAT_REPAIR_ACTIVATION.json"
    if marker.exists():
        if read_json(marker) != receipt:
            raise PermissionError("Existing format repair activation differs")
    else:
        atomic_json(marker, receipt, exclusive=True)
    # The original bytes are already durably retained and authenticated above.
    # Removing only this duplicate permits the eventual new outcome receipt.
    blocked.unlink()


def publish_bridge_format_identity(root, runtime, zero, bridge):
    """Name the actual bridged weights before either post-bridge FORMAT panel."""
    runtime.identity.pop("zero_output_exact", None)
    runtime.identity.pop("base_logits_hash", None)
    return publish_adapter(
        root,
        runtime,
        "SRF1_FORMAT_BRIDGED",
        extra={
            "bridge_executed": True,
            "bridge_logical_updates": bridge["logical_updates"],
            "zero_output_initialization": zero["trainable_state_hash"],
        },
    )


def amended_format_decision(covered, responses):
    """The user-authorized SR-F1.1 gate is frozen before any new responses."""
    if type(covered) is not int or type(responses) is not int or responses != 512:
        raise ValueError("SR-F1.1 confirmation requires exactly 512 scored records")
    if not 0 <= covered <= responses:
        raise ValueError("Invalid confirmation numerator")
    if covered >= 487:
        return "PASS_95"
    if covered >= 461:
        return "PASS_90_WITH_FORMAT_REPORTING"
    return "F2_AMENDMENT_REQUIRED"


def _amended_common_start(runtime, plan, root, zero, boundary):
    """Fresh zero-LoRA FORMAT/CONFIRM; the original bridge is never loaded."""
    root = Path(root)
    if (root / "engineering/bridge").exists():
        raise PermissionError("SR-F1.1 common start cannot contain a bridge")
    amendment_hash = file_hash(root / "AMENDMENT.json")
    if zero.get("zero_output_exact") is not True:
        raise PermissionError("SR-F1.1 requires an independently verified zero-output adapter")
    if state_hash(trainable_state(runtime.model)) != zero["trainable_state_hash"]:
        raise PermissionError("SR-F1.1 common parameters differ from verified zero LoRA")
    initial = format_panel(runtime, root, "FORMAT", "before", boundary)
    confirmation = format_panel(
        runtime, root, "FORMAT_CONFIRM", "confirm", boundary, samples_per_prompt=16
    )
    decision = amended_format_decision(confirmation["covered"], confirmation["responses"])
    if state_hash(trainable_state(runtime.model)) != zero["trainable_state_hash"]:
        raise PermissionError("Parameters changed during SR-F1.1 format sampling")
    metadata = dict(
        common_start="SRF1_COMMON_START",
        bridge_executed=False,
        protocol_amendment=plan["protocol_amendment"],
        protocol_amendment_sha256=amendment_hash,
        format_gate_decision=decision,
        per_arm_per_update_format_reporting=True,
        old_bridge_used_for_scientific_start=False,
    )
    result = dict(
        status="PROTOCOL_BLOCKED" if decision == "F2_AMENDMENT_REQUIRED" else "COMPLETE",
        format_before=initial,
        format_confirmation=confirmation,
        format_after=None,
        bridge=None,
        bridge_executed=False,
        artifacts=[
            "COMMON_ZERO_LORA.json",
            "FORMAT_AND_BRIDGE_RECEIPT.json",
            "engineering/format/before/COVERAGE.json",
            "engineering/format/confirm/COVERAGE.json",
            "AMENDMENT.json",
        ],
        metadata=metadata,
    )
    if decision == "F2_AMENDMENT_REQUIRED":
        result["reason"] = "SR_F1_1_CONFIRMATION_BELOW_90_PERCENT"
    else:
        entry = publish_adapter(
            root,
            runtime,
            "SRF1_COMMON_START",
            extra={
                **metadata,
                "zero_output_initialization": zero["trainable_state_hash"],
                "zero_output_exact": True,
            },
        )
        common = root / "COMMON_START.json"
        if common.exists():
            if read_json(common) != entry:
                raise PermissionError("Partially published amended common start differs")
        else:
            atomic_json(common, entry, exclusive=True)
        result["artifacts"].append("COMMON_START.json")
    atomic_json(root / "FORMAT_AND_BRIDGE_RECEIPT.json", result, exclusive=True)
    return result


def _establish_common_start(plan, root, boundary):
    """Zero-output init, format-only trigger, at most one fixed shared bridge."""
    import gc

    from mm_core.training import configure_training, frozen_hash

    from .data import load_inputs, load_tasks
    from .training import run_common_bridge

    root = Path(root)
    final_receipt = root / "FORMAT_AND_BRIDGE_RECEIPT.json"
    if final_receipt.exists():
        result = read_json(final_receipt)
        if result["status"] == "COMPLETE":
            verified_adapter(root, "SRF1_COMMON_START")
            return result
        if (
            result.get("reason") == "FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW"
            and result.get("bridge_executed") is False
        ):
            activate_reviewed_format_repair(root)
        else:
            raise PermissionError("Protocol is blocked; repeated bridge is forbidden")
    runtime = load_runtime(plan, root, account=runtime_account(root, "COMMON_START"))
    inputs, tasks = load_inputs(root), load_tasks(root)
    zero_path = root / "COMMON_ZERO_LORA.json"
    if zero_path.exists():
        from peft import PeftModel

        zero = read_json(zero_path)
        if (
            zero.get("plan_id") != PLAN_ID
            or zero.get("freeze_sha256") != file_hash(root / "EXECUTION_FREEZE.json")
            or zero.get("zero_output_exact") is not True
        ):
            raise PermissionError("Common zero LoRA belongs to a different execution identity")
        actual = adapter_identity(root, zero["adapter_path"])
        if any(zero.get(k) != v for k, v in actual.items()):
            raise PermissionError("Common zero LoRA changed")
        runtime.model = PeftModel.from_pretrained(
            runtime.model, bounded_path(root, zero["adapter_path"]), is_trainable=False
        )
        configure_training(runtime)
        if state_hash(trainable_state(runtime.model)) != zero["trainable_state_hash"]:
            raise PermissionError("Restored zero LoRA parameter hash differs")
    else:
        qid = sorted(q for q, t in tasks.items() if t["pool"] == "ENGINE")[0]
        prepared = runtime.prepare(inputs[qid], root)
        kwargs = dict(prepared["inputs"], logits_to_keep=1, use_cache=False)
        runtime.reserve("extra_forward_sequences", 1, purpose="zero_lora_base")
        with runtime.torch.no_grad():
            before = runtime.model(**kwargs).logits.detach().cpu()
        seed_all(plan["model"]["lora"]["init_seed"])
        configure_training(runtime)
        for name, p in runtime.model.named_parameters():
            if "lora_B" in name and bool(p.detach().ne(0).any()):
                raise RuntimeError("Common initialization did not have exact zero B matrices")
        runtime.reserve("extra_forward_sequences", 1, purpose="zero_lora_invariance")
        with runtime.torch.no_grad():
            after = runtime.model(**kwargs).logits.detach().cpu()
        if not runtime.torch.equal(before, after):
            raise RuntimeError("Zero-output LoRA changed native logits")
        zero = publish_adapter(
            root,
            runtime,
            "SRF1_COMMON_ZERO",
            extra={
                "initialization_seed": plan["model"]["lora"]["init_seed"],
                "zero_output_exact": True,
                "base_logits_hash": state_hash(before),
                "frozen_base_hash": frozen_hash(runtime.model),
            },
        )
        atomic_json(zero_path, zero, exclusive=True)
    runtime.identity.update(zero)
    runtime.adapter_path = str(bounded_path(root, zero["adapter_path"]))
    if plan.get("protocol_amendment"):
        return _amended_common_start(runtime, plan, root, zero, boundary)
    initial = format_panel(runtime, root, "FORMAT", "before", boundary)
    bridge = None
    confirmation, repeated = None, None
    if initial["coverage"] < 0.95:
        if initial["truncated"]:
            # Natural length failures stay in the original denominator. Only an
            # authenticated audit excluding technical causes can qualify them
            # as part of genuine protocol non-adherence before the one bridge.
            from .format_review import verify_format_review

            try:
                review = verify_format_review(root)
                counts = review["counts"]
                if any(
                    counts[key] != initial[key] for key in ("responses", "covered", "truncated")
                ):
                    raise PermissionError("Reviewed FORMAT counts differ from actual panel")
            except (FileNotFoundError, PermissionError, ValueError, KeyError):
                blocked = dict(
                    status="PROTOCOL_BLOCKED",
                    reason="FORMAT_TRUNCATION_REQUIRES_TECHNICAL_REVIEW",
                    format_before=initial,
                    bridge_executed=False,
                    artifacts=[str(zero_path.relative_to(root))],
                )
                atomic_json(final_receipt, blocked, exclusive=True)
                return blocked
        # Native template, parser CPU contract and image routing are separately gated.
        preflight = read_json(root / "PROCESSOR_PREFLIGHT.json")
        if preflight.get("status") not in ("PASS", "VERIFIED", "COMPLETE"):
            raise PermissionError("Protocol coverage cannot trigger bridge before input preflight")
        bridge = run_common_bridge(runtime, root, plan, boundary)
        if bridge["status"] == "CHECKPOINTED":
            return bridge
        publish_bridge_format_identity(root, runtime, zero, bridge)
        confirmation = format_panel(runtime, root, "FORMAT_CONFIRM", "confirm", boundary)
        repeated = format_panel(runtime, root, "FORMAT", "after", boundary)
        if confirmation["coverage"] < 0.95:
            blocked = dict(
                status="PROTOCOL_BLOCKED",
                reason="ONE_BRIDGE_CONFIRMATION_BELOW_95_PERCENT",
                format_before=initial,
                bridge=bridge,
                format_confirmation=confirmation,
                format_after=repeated,
                artifacts=[str(zero_path.relative_to(root))],
            )
            atomic_json(final_receipt, blocked, exclusive=True)
            return blocked
    entry = publish_adapter(
        root,
        runtime,
        "SRF1_COMMON_START",
        extra={
            "bridge_executed": bridge is not None,
            "zero_output_initialization": zero["trainable_state_hash"],
        },
    )
    common_path = root / "COMMON_START.json"
    if common_path.exists():
        if read_json(common_path) != entry:
            raise PermissionError("Partially published common start identity differs")
    else:
        atomic_json(common_path, entry, exclusive=True)
    result = dict(
        status="COMPLETE",
        format_before=initial,
        bridge=bridge,
        format_confirmation=confirmation,
        format_after=repeated,
        artifacts=["COMMON_START.json", "COMMON_ZERO_LORA.json", "FORMAT_AND_BRIDGE_RECEIPT.json"],
        metadata={"common_start": "SRF1_COMMON_START", "bridge_executed": bridge is not None},
    )
    atomic_json(final_receipt, result, exclusive=True)
    del runtime
    gc.collect()
    return result


def establish_common_start(plan, root):
    from .training import LeaseEnding, stop_at_committed_boundary

    root = Path(root)
    with stop_at_committed_boundary() as boundary:
        try:
            return _establish_common_start(plan, root, boundary)
        except LeaseEnding:
            # FORMAT has independently seeded immutable raw slots and the already
            # persisted common zero adapter. Neither needs a gradient/Adam state.
            zero = read_json(root / "COMMON_ZERO_LORA.json")
            artifacts = ["COMMON_ZERO_LORA.json"]
            artifacts.extend(
                str(p.relative_to(root))
                for p in bounded_path(root, zero["adapter_path"]).iterdir()
                if p.is_file()
            )
            artifacts.extend(
                str(p.relative_to(root)) for p in (root / "engineering/format").rglob("*.json")
            )
            bridge_latest = root / "engineering/bridge/checkpoints/LATEST.json"
            if bridge_latest.exists():
                latest = read_json(bridge_latest)
                artifacts.extend(
                    [
                        str(bridge_latest.relative_to(root)),
                        str((bridge_latest.parent / latest["path"]).relative_to(root)),
                    ]
                )
            return dict(
                status="CHECKPOINTED",
                reason="PREEMPTION",
                artifacts=artifacts,
                metadata=dict(
                    full_state=True,
                    identity_verified=True,
                    next_update=None,
                    run_id="SRF1_COMMON_START",
                ),
            )
