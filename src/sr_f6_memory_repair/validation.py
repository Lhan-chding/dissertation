"""Fixed original-token storage validation; no generation or optimizer is exposed.

This is a separately identified technical consumer, not a legacy replay permit.
It compares row zero both ways, the original 27 forward vectors, and executes
all 128 retained rows with the original 64 diagnostic rows. It does not claim
paired baseline gradients for all 128 rows or scientific learning value.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import time
import uuid
from pathlib import Path

from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
from sr_f6.common import write_once
from sr_f6.numerical_replay import ATOL, RTOL, _metric, _Preserve
from sr_f6.objective import sequence_loss
from sr_f6.training import (
    _diagnostic_indices,
    _numerical_audit,
    _validate_records,
    _verify_prepared,
)
from sr_f12.training import enforce_probability_gate

from .offload import saved_activation_offload

SCHEMA = "srf6-memory-storage-validation-v3"
PLAN = dict(
    baseline_rows=[0],
    offload_rows=list(range(128)),
    archived_forward_rows=list(range(27)),
    failing_row=26,
    diagnostic_rows="original_registered_64",
    loss_denominator=128,
    atol=ATOL,
    rtol=RTOL,
    generated_responses=0,
    optimizer_updates=0,
    full_128_paired_gradient_equivalence=False,
)
RESTORE_KEYS = (
    "adapter_unchanged_during_replay",
    "reference_unchanged_during_replay",
    "adapter_restored",
    "rng_restored",
    "gradients_restored",
    "requires_grad_restored",
    "module_modes_restored",
    "frozen_base_buffer_versions_unchanged",
    "parameter_objects_unchanged",
)


def require(value, message):
    if not value:
        raise PermissionError(message)


V4_SCHEMA = "srf6-memory-storage-validation-v4"
MAX_DEVICE_BYTES = 24 * 1024**3
VALIDATION_WINDOW = dict(
    rows=[0, 1, 2, 3],
    diagnostic_required=True,
    max_total_row_seconds=902.8157889822033,
    continue_in_same_process=True,
)


class TechnicalWindowStop(RuntimeError):
    """A fixed technical throughput condition failed; not a scientific gate."""


def validation_schema(contract):
    maximum = contract.get("max_device_bytes", 0)
    require(type(maximum) is int and maximum >= 0, "Invalid device residency contract")
    if maximum:
        window = contract.get("validation_window")
        require(
            maximum == MAX_DEVICE_BYTES
            and type(window) is dict
            and window == VALIDATION_WINDOW
            and type(window["rows"]) is list
            and all(type(v) is int for v in window["rows"])
            and window["diagnostic_required"] is True
            and window["continue_in_same_process"] is True
            and type(window["max_total_row_seconds"]) is float,
            "Frozen v4 residency/window contract differs",
        )
        return V4_SCHEMA
    require("validation_window" not in contract, "Legacy validation cannot adopt a new window")
    return SCHEMA


def verify_device_storage(summary, expected_max_device_bytes):
    """Backing-storage charges may outlive released hooks; active hooks may not."""
    require(
        type(expected_max_device_bytes) is int and expected_max_device_bytes > 0,
        "Explicit device bound required",
    )
    expected = dict(
        device_residency_schema="srf6-bounded-device-residency-candidate-v1",
        device_accounting_scope="UNIQUE_LIVE_SAVED_ACTIVATION_BACKING_STORAGE",
        device_accounting_release="CPP_STORAGE_EXPIRATION_CONSERVATIVE_REAP",
        device_peak_scope="CONSERVATIVE_CHARGED_PAYLOAD_HIGH_WATER_NOT_CUDA_PEAK",
    )
    require(
        all(summary.get(k) == val for k, val in expected.items()),
        "Device accounting identity differs",
    )
    keys = (
        "max_device_bytes",
        "live_device_bytes",
        "peak_device_bytes",
        "device_storage_count",
        "device_pending_storage_count",
        "device_active_view_count",
        "peak_device_storage_count",
        "device_resident_pack_count",
        "device_alias_pack_count",
        "device_resident_unpack_count",
        "device_spill_count",
    )
    require(
        all(type(summary.get(k)) is int and summary[k] >= 0 for k in keys),
        "Incomplete device residency counters",
    )
    require(
        summary["max_device_bytes"] == expected_max_device_bytes
        and summary["live_device_bytes"]
        <= summary["peak_device_bytes"]
        <= expected_max_device_bytes
        and summary["device_active_view_count"] == 0
        and summary["device_pending_storage_count"]
        <= summary["device_storage_count"]
        <= summary["peak_device_storage_count"]
        and summary["device_alias_pack_count"] <= summary["device_resident_pack_count"]
        and summary["device_resident_pack_count"] > 0
        and summary["device_resident_unpack_count"] > 0,
        "Actual device residency bound/lifecycle differs",
    )
    return summary


class CudaObservations:
    """No CUDA peak resets: snapshots retain process-lifetime high water marks."""

    def __init__(self, *, fixture=False, clock=None):
        import socket

        import torch

        require(clock is None or fixture, "Custom timing is CPU fixture only")
        self.fixture = fixture
        self.clock = time.monotonic if clock is None else clock
        self.records = []
        self.identity = dict(
            scope="SYNTHETIC_CPU_OBSERVATIONS" if fixture else "ACTUAL_CUDA_PROCESS_OBSERVATIONS",
            process_id=os.getpid(),
            hostname=socket.gethostname(),
            peak_reset_count=0,
            peak_scope="PROCESS_LIFETIME_INCLUDES_LOAD_BASELINE_AND_ALL_ROWS_NO_RESET",
        )
        if not fixture:
            require(
                torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "Actual one-device CUDA observation required",
            )
            props = torch.cuda.get_device_properties(0)
            self.identity.update(
                cuda_uuid=str(props.uuid),
                cuda_name=props.name,
                cuda_total_memory_bytes=props.total_memory,
            )

    def observe(self, stage, row=None):
        import torch

        value = dict(
            sequence=len(self.records), stage=stage, row_index=row, monotonic_seconds=self.clock()
        )
        if not self.fixture:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            value.update(
                memory_allocated=torch.cuda.memory_allocated(),
                memory_reserved=torch.cuda.memory_reserved(),
                max_memory_allocated=torch.cuda.max_memory_allocated(),
                max_memory_reserved=torch.cuda.max_memory_reserved(),
                free_device_bytes=free,
                total_device_bytes=total,
            )
            value["monotonic_seconds"] = self.clock()
        self.records.append(value)
        return value

    def evidence(self):
        return dict(identity=dict(self.identity), observations=self.records)


def window_measurement(observations, indices):
    require(
        set(VALIDATION_WINDOW["rows"]) <= set(indices),
        "Original first four rows must be diagnostic",
    )
    spans = []
    for row in VALIDATION_WINDOW["rows"]:
        start = [r for r in observations if r["stage"] == "row_forward" and r["row_index"] == row]
        end = [r for r in observations if r["stage"] == "row_finished" and r["row_index"] == row]
        require(
            len(start) == len(end) == 1, "Technical window must use each original row exactly once"
        )
        duration = end[0]["monotonic_seconds"] - start[0]["monotonic_seconds"]
        require(
            type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
            "Invalid measured technical row duration",
        )
        spans.append(
            dict(
                row_index=row,
                start_sequence=start[0]["sequence"],
                end_sequence=end[0]["sequence"],
                seconds=duration,
            )
        )
    total = sum(s["seconds"] for s in spans)
    return dict(
        rule=copy.deepcopy(VALIDATION_WINDOW),
        rows=spans,
        total_row_seconds=total,
        scope="TECHNICAL_OBSERVATION_ONLY_NOT_TRAINING_FEASIBILITY_OR_SCIENTIFIC_GATE",
        status="CONTINUE_SAME_PROCESS"
        if total <= VALIDATION_WINDOW["max_total_row_seconds"]
        else "TECHNICAL_STOP_NO_SCIENTIFIC_CONCLUSION",
    )


def verify_cuda_observations(evidence, hardware, diagnostic_indices):
    observation = evidence["cuda_observations"]
    identity, records = observation["identity"], observation["observations"]
    require(
        identity["scope"] == "ACTUAL_CUDA_PROCESS_OBSERVATIONS"
        and identity["peak_reset_count"] == 0
        and identity["peak_scope"]
        == "PROCESS_LIFETIME_INCLUDES_LOAD_BASELINE_AND_ALL_ROWS_NO_RESET"
        and type(identity["process_id"]) is int
        and identity["process_id"] > 0
        and all(
            identity[k] == hardware[k]
            for k in ("cuda_uuid", "cuda_name", "cuda_total_memory_bytes", "hostname")
        ),
        "Actual CUDA observation identity differs",
    )
    expected = [
        ("before_load", None),
        ("after_load", None),
        ("baseline_forward", 0),
        ("baseline_backward", 0),
    ]
    for row in range(128):
        expected.append(("row_forward", row))
        if row in diagnostic_indices:
            expected.extend(("diagnostic_" + channel, row) for channel in ("native", "aux", "kl"))
        expected.extend((("row_backward", row), ("row_finished", row)))
    expected.extend((("restoration", None), ("recompute", None), ("measurement_finished", None)))
    require(
        [(r["stage"], r["row_index"]) for r in records] == expected,
        "Incomplete, reordered or repeated CUDA stage observations",
    )
    previous_time = previous_allocated_peak = previous_reserved_peak = 0
    for index, row in enumerate(records):
        require(
            row["sequence"] == index
            and type(row["monotonic_seconds"]) in (float, int)
            and math.isfinite(row["monotonic_seconds"])
            and row["monotonic_seconds"] >= previous_time,
            "CUDA observation sequence/timing differs",
        )
        keys = (
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
            "free_device_bytes",
            "total_device_bytes",
        )
        require(
            all(type(row.get(k)) is int and row[k] >= 0 for k in keys),
            "Actual CUDA byte counters absent",
        )
        require(
            row["memory_allocated"] <= row["memory_reserved"]
            and row["memory_allocated"] <= row["max_memory_allocated"] <= row["max_memory_reserved"]
            and row["memory_reserved"] <= row["max_memory_reserved"]
            and row["max_memory_allocated"] >= previous_allocated_peak
            and row["max_memory_reserved"] >= previous_reserved_peak
            and row["free_device_bytes"]
            <= row["total_device_bytes"]
            == identity["cuda_total_memory_bytes"],
            "CUDA high water or physical observation differs",
        )
        previous_time = row["monotonic_seconds"]
        previous_allocated_peak, previous_reserved_peak = (
            row["max_memory_allocated"],
            row["max_memory_reserved"],
        )
    window = window_measurement(records, diagnostic_indices)
    require(
        window == evidence["technical_window"] and window["status"] == "CONTINUE_SAME_PROCESS",
        "Fixed technical observation window did not pass",
    )
    return observation


class ProgressWriter:
    """Atomic latest observation owned by this process, never a restart checkpoint."""

    STAGES = frozenset(
        {
            "baseline_forward",
            "baseline_backward",
            "row_forward",
            "diagnostic_native",
            "diagnostic_aux",
            "diagnostic_kl",
            "row_backward",
            "row_finished",
            "restoration",
            "recompute",
            "measurement_finished",
        }
    )
    STORAGE_FIELDS = (
        "pack_count",
        "unpack_count",
        "live_host_bytes",
        "peak_host_bytes",
        "live_disk_bytes",
        "peak_disk_bytes",
        "disk_bytes_written",
        "disk_bytes_read",
        "arena_physical_bytes",
        "peak_arena_physical_bytes",
        "arena_created_count",
        "arena_deleted_count",
        "arena_fsync_count",
        "arena_open_count",
        "arena_read_calls",
        "arena_write_calls",
    )
    COUNT_FIELDS = frozenset(
        {
            "reference_forwards",
            "policy_forwards",
            "backward_calls",
            "diagnostic_gradient_calls",
            "generated_responses",
            "optimizer_updates",
        }
    )

    def __init__(self, directory, *, contract_sha256, repair_revision, cuda_observer=None):
        self.cuda_observer = cuda_observer
        self.directory = Path(directory)
        require(
            self.directory.is_absolute()
            and self.directory == self.directory.resolve(strict=True)
            and self.directory.is_dir(),
            "Canonical progress directory required",
        )
        require(re.fullmatch(r"[0-9a-f]{64}", contract_sha256), "Progress contract hash required")
        require(re.fullmatch(r"[0-9a-f]{40}", repair_revision), "Progress source required")
        self.binding = dict(contract_sha256=contract_sha256, repair_revision=repair_revision)
        st = self.directory.stat()
        self.directory_identity = (st.st_dev, st.st_ino)
        self.latest_identity = None
        self.sequence = 0
        self.started = self.last_at = time.monotonic()
        self.last_stage = None
        self.stage_seconds = {}

    def _check_directory(self, fd):
        current = self.directory.lstat()
        opened = os.fstat(fd)
        require(
            stat.S_ISDIR(current.st_mode)
            and (current.st_dev, current.st_ino) == self.directory_identity
            and (opened.st_dev, opened.st_ino) == self.directory_identity,
            "Owned progress directory changed",
        )

    def __call__(self, stage, row_index, completed_rows, counts, manager=None):
        require(stage in self.STAGES, "Unknown validation progress stage")
        require(
            row_index is None or (type(row_index) is int and 0 <= row_index < 128),
            "Invalid progress row",
        )
        require(
            type(completed_rows) is int and 0 <= completed_rows <= 128,
            "Invalid completed row count",
        )
        require(
            set(counts) == self.COUNT_FIELDS
            and all(type(v) is int and v >= 0 for v in counts.values()),
            "Invalid progress call counts",
        )
        require(
            counts["generated_responses"] == counts["optimizer_updates"] == 0,
            "Validation progress cannot include generation or updates",
        )
        now = time.monotonic()
        previous_seconds = now - self.last_at
        if self.last_stage is not None:
            self.stage_seconds[self.last_stage] = (
                self.stage_seconds.get(self.last_stage, 0.0) + previous_seconds
            )
        storage = {
            name: value
            for name in self.STORAGE_FIELDS
            if manager is not None and type(value := getattr(manager, name, None)) is int
        }
        value = dict(
            schema="srf6-validation-progress-v1",
            status="NOT_COMPLETE_NOT_RESUMABLE",
            **self.binding,
            sequence=self.sequence,
            stage=stage,
            row_index=row_index,
            completed_rows=completed_rows,
            counts=dict(counts),
            storage=storage,
            elapsed_seconds=now - self.started,
            previous_stage=self.last_stage,
            previous_stage_seconds=previous_seconds,
            stage_seconds=dict(self.stage_seconds),
        )
        if self.cuda_observer is not None:
            value["cuda_observation"] = self.cuda_observer.records[-1]
            value["cuda_observation_identity"] = self.cuda_observer.identity
        payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        require(len(payload) <= 32768, "Progress record exceeded fixed size bound")
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = ".PROGRESS." + uuid.uuid4().hex + ".tmp"
        try:
            self._check_directory(fd)
            if self.latest_identity is not None:
                old = os.stat("PROGRESS.json", dir_fd=fd, follow_symlinks=False)
                require(
                    stat.S_ISREG(old.st_mode)
                    and old.st_nlink == 1
                    and (old.st_dev, old.st_ino) == self.latest_identity,
                    "Owned progress record changed",
                )
            out = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
            )
            with os.fdopen(out, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                created = os.fstat(stream.fileno())
            self._check_directory(fd)
            if self.latest_identity is None:
                os.link(
                    temporary, "PROGRESS.json", src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False
                )
                os.unlink(temporary, dir_fd=fd)
            else:
                os.replace(temporary, "PROGRESS.json", src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
            self.latest_identity = (created.st_dev, created.st_ino)
        finally:
            # Never remove an unknown replacement or evidence after a failed publication.
            os.close(fd)
        self.sequence += 1
        self.last_at, self.last_stage = now, stage


def _trace(record, current, ref):
    return dict(
        row_id=record["row_id"],
        qid=record["qid"],
        tokens=record["tokens"],
        sampler_logprobs=record["sampler_logprobs"],
        train_logprobs=current.detach().cpu().tolist(),
        reference_logprobs=ref.detach().cpu().tolist(),
        token_path_sha256=record["token_path_sha256"],
    )


def _compare_trace(left, right):
    import torch

    for key in ("row_id", "qid", "tokens", "sampler_logprobs", "token_path_sha256"):
        require(left[key] == right[key], "Original trajectory differs: " + key)
    result = {}
    for key in ("train_logprobs", "reference_logprobs"):
        a, b = (torch.tensor(row[key], dtype=torch.float32) for row in (left, right))
        require(a.shape == b.shape == (len(left["tokens"]),), "Original token vector shape differs")
        result[key] = _metric(a, b, a.shape)
    return result


def recompute(evidence, tensors, originals):
    """Recompute comparisons from complete vectors and tensors, never status alone."""
    import torch

    require(evidence["plan"] == PLAN, "Validation plan changed")
    traces = evidence["traces"]
    require(len(traces) == 128 and len(originals) == 27, "Incomplete original coverage")
    require(len({row["row_id"] for row in traces}) == 128, "Duplicated original rows")
    comparisons = [_compare_trace(a, b) for a, b in zip(traces[:27], originals, strict=True)]
    comparisons.append(_compare_trace(evidence["baseline_trace"], traces[0]))
    baseline, offload = tensors["baseline"], tensors["offload_row0"]
    require(
        bool(baseline) and baseline.keys() == offload.keys() == tensors["accumulated"].keys(),
        "Adapter gradient inventory differs",
    )
    gradients = {
        key: _metric(baseline[key], offload[key], evidence["parameter_shapes"][key])
        for key in baseline
    }
    metrics = list(gradients.values()) + [v for row in comparisons for v in row.values()]
    require(
        all(v["finite"] and v["violations"] == 0 for v in metrics),
        "Fixed numerical tolerance failed",
    )
    for name in ("loss", "native", "aux", "kl"):
        a, b = evidence["baseline_loss"][name], evidence["offload_row0_loss"][name]
        require(abs(a - b) <= ATOL + RTOL * abs(b), "Fixed row-zero loss tolerance failed")
    require(any(v is not None for v in tensors["accumulated"].values()), "No accumulated gradient")
    for values in (tensors["accumulated"], *tensors["channels"].values()):
        require(values.keys() == baseline.keys(), "Diagnostic gradient inventory differs")
        require(
            all(
                v is None or (v.dtype == torch.float32 and bool(torch.isfinite(v).all()))
                for v in values.values()
            ),
            "Nonfinite or non-FP32 adapter gradient",
        )
        require(
            all(
                v is None or list(v.shape) == evidence["parameter_shapes"][name]
                for name, v in values.items()
            ),
            "Actual adapter gradient shape differs",
        )
    require(set(tensors["channels"]) == {"native", "aux", "kl"}, "Missing diagnostic channels")
    require(
        evidence["counts"]
        == dict(
            reference_forwards=129,
            policy_forwards=129,
            backward_calls=129,
            diagnostic_gradient_calls=192,
            generated_responses=0,
            optimizer_updates=0,
        ),
        "Actual call counts differ",
    )
    require(
        len(evidence["diagnostic_indices"]) == len(set(evidence["diagnostic_indices"])) == 64,
        "Incomplete diagnostic rows",
    )
    require(
        all(evidence["restoration"].get(k) is True for k in RESTORE_KEYS)
        and not evidence["restoration"]["restoration_errors"]
        and evidence["restoration"]["runtime_requires_reload"] is False,
        "Runtime restoration failed",
    )
    audit = _numerical_audit(traces)
    require(audit["status"] == "FINITE", "Nonfinite probability audit")
    enforce_probability_gate(audit)
    require(audit["sampler_train_max_absolute_difference"] <= 5, "Original probability gate failed")
    return dict(
        gradient_comparisons=gradients, forward_comparisons=comparisons, probability_audit=audit
    )


def verify_storage(
    summary,
    scratch_directory,
    max_host_bytes,
    max_disk_bytes,
    *,
    fixture=False,
    disk_backend="file_v2",
    max_device_bytes=0,
):
    """Check the complete owned hybrid storage lifecycle, including cold cleanup."""
    require(disk_backend in {"file_v2", "arena_v3"}, "Unknown validation storage backend")
    scratch = Path(scratch_directory)
    scope = "SYNTHETIC_CPU_ONLY" if fixture else "CUDA_SAVED_ACTIVATION_HYBRID_STORAGE"
    require(
        scratch.is_absolute() and scratch == scratch.resolve(strict=True) and scratch.is_dir(),
        "Canonical owned validation scratch required",
    )
    require(
        summary["scope"] == scope
        and summary["schema"]
        == (
            "srf6-saved-activation-arena-v3"
            if disk_backend == "arena_v3"
            else "srf6-saved-activation-hybrid-v2"
        )
        and summary["closed"] is True
        and summary["model_bound"] is True
        and summary["numerical_pass"] is False,
        "Actual hybrid storage context did not complete",
    )
    require(
        summary["max_host_bytes"] == max_host_bytes
        and summary["max_disk_bytes"] == max_disk_bytes
        and summary["scratch_directory"] == str(scratch),
        "Hybrid storage identity differs",
    )
    require(
        0 <= summary["peak_host_bytes"] <= max_host_bytes
        and 0 <= summary["peak_disk_bytes"] <= max_disk_bytes
        and summary["staging_chunk_bytes"] == 64 * 1024**2
        and 0 <= summary["peak_staging_host_bytes"] <= summary["staging_chunk_bytes"],
        "Hybrid storage bound exceeded",
    )
    require(
        summary["pack_count"] > 0
        and (
            summary["unpack_count"]
            + (summary.get("device_resident_unpack_count", 0) if max_device_bytes else 0)
        )
        > 0
        and summary["live_host_bytes"] == summary["live_disk_bytes"] == 0
        and not summary["storage_failures"]
        and not summary["cleanup_errors"]
        and not any(scratch.iterdir()),
        "Hybrid storage failed or payload remains",
    )
    if disk_backend == "arena_v3":
        require(summary.get("disk_backend") == "arena_v3", "Actual arena backend differs")
        require(
            summary.get("arena_physical_bytes") == 0
            and type(summary.get("peak_arena_physical_bytes")) is int
            and summary["peak_disk_bytes"]
            <= summary["peak_arena_physical_bytes"]
            <= max_disk_bytes,
            "Arena physical storage bound exceeded",
        )
        keys = (
            "arena_created_count",
            "arena_deleted_count",
            "arena_fsync_count",
            "arena_open_count",
            "arena_read_calls",
            "arena_write_calls",
        )
        require(
            all(type(summary.get(k)) is int and summary[k] >= 0 for k in keys),
            "Incomplete arena counters",
        )
        require(
            summary["arena_created_count"] == summary["arena_deleted_count"]
            and summary["arena_fsync_count"] <= summary["arena_created_count"]
            and summary["arena_open_count"] == summary["arena_created_count"],
            "Arena lifecycle counters differ",
        )
        require(
            (summary["disk_bytes_written"] == 0 and summary["arena_created_count"] == 0)
            or (
                summary["disk_bytes_written"] > 0
                and summary["arena_created_count"] > 0
                and summary["arena_read_calls"] > 0
                and summary["arena_write_calls"] > 0
            ),
            "Incomplete arena actual IO",
        )
    else:
        require(summary.get("disk_backend", "file_v2") == "file_v2", "Legacy backend differs")
    written, read = summary["disk_bytes_written"], summary["disk_bytes_read"]
    require(
        type(written) is int
        and type(read) is int
        and written >= summary["peak_disk_bytes"]
        and (summary["peak_disk_bytes"] == 0) == (written == 0)
        and read >= 0
        and ((written == read == 0) or (written > 0 and read > 0)),
        "Incomplete actual disk spill/read accounting",
    )
    if max_device_bytes:
        verify_device_storage(summary, max_device_bytes)
    return summary


def _measure(
    runtime,
    records,
    credits,
    reference,
    originals,
    *,
    max_host_bytes,
    scratch_directory=None,
    max_disk_bytes=None,
    fixture=False,
    disk_backend="file_v2",
    progress=None,
    max_device_bytes=0,
    cuda_observer=None,
):
    import torch

    if max_device_bytes:
        require(
            cuda_observer is not None and cuda_observer.fixture is fixture,
            "Explicit bound CUDA observer required for residency validation",
        )
        if not fixture:
            require(max_device_bytes == MAX_DEVICE_BYTES, "Fixed device candidate required")
    else:
        require(cuda_observer is None, "Legacy validation keeps its original observations")
    root = Path(runtime.identity["run_root"])
    coefficients = _validate_records(
        runtime, records, [8] * 16, reference, credits, root, trainable_state(runtime.model)
    )
    indices = _diagnostic_indices(runtime, records, runtime.identity["diagnostic_row_indices"])
    require(len(originals) == 27, "Exactly 27 preserved original forwards required")
    if max_device_bytes:
        require(
            set(VALIDATION_WINDOW["rows"]) <= set(indices),
            "Original first four rows must be diagnostic",
        )
    params = {
        name: value for name, value in runtime.model.named_parameters() if value.requires_grad
    }
    preserved = _Preserve(runtime, reference)
    counts = dict(
        reference_forwards=0,
        policy_forwards=0,
        backward_calls=0,
        diagnostic_gradient_calls=0,
        generated_responses=0,
        optimizer_updates=0,
    )
    evidence = dict(
        plan=copy.deepcopy(PLAN),
        traces=[],
        counts=counts,
        diagnostic_indices=list(indices),
        parameter_shapes={name: list(p.shape) for name, p in params.items()},
    )
    tensors = {
        "channels": {
            kind: {name: torch.zeros_like(p, device="cpu") for name, p in params.items()}
            for kind in ("native", "aux", "kl")
        }
    }

    if cuda_observer is not None:
        evidence["cuda_observations"] = cuda_observer.evidence()
        evidence["max_device_bytes"] = max_device_bytes
    completed_rows = 0
    offload = None

    def report(stage, row=None):
        try:
            if cuda_observer is not None:
                cuda_observer.observe(stage, row)
            if progress is not None:
                progress(stage, row, completed_rows, counts, offload)
        except BaseException as exc:
            exc.validation_partial = (evidence, tensors)
            raise

    def gradients():
        return {
            name: None if p.grad is None else p.grad.detach().cpu().clone()
            for name, p in params.items()
        }

    def forward(index):
        row = records[index]
        prepared = runtime.prepare_input(row["model_input"], root, protocol=row["protocol"])
        _verify_prepared(row, prepared)
        item = dict(row, prepared=prepared)
        with torch.no_grad():
            ref = runtime.batch_reference_forward([item], reference)[0]
        counts["reference_forwards"] += 1
        current = runtime.batch_sequence_forward([item], purpose="training_policy", grad=True)[0]
        counts["policy_forwards"] += 1
        sampler = torch.tensor(row["sampler_logprobs"], dtype=torch.float32, device=current.device)
        require(
            current.shape == ref.shape == sampler.shape == (len(row["tokens"]),),
            "Full token path required",
        )
        result = sequence_loss(
            current, sampler, ref, coefficients[index]["native"], coefficients[index]["aux"]
        )
        return result, _trace(row, current, ref)

    try:
        runtime.model.eval()
        if not fixture and not max_device_bytes:
            torch.cuda.reset_peak_memory_stats()
        runtime.model.zero_grad(set_to_none=True)
        baseline_rng = capture_rng()
        report("baseline_forward", 0)
        result, evidence["baseline_trace"] = forward(0)
        evidence["baseline_loss"] = {
            k: float(v.detach()) for k, v in result.items() if k in ("loss", "native", "aux", "kl")
        }
        report("baseline_backward", 0)
        (result["loss"] / 128).backward()
        counts["backward_calls"] += 1
        tensors["baseline"] = gradients()
        del result
        if not fixture:
            evidence["baseline_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
            if not max_device_bytes:
                torch.cuda.reset_peak_memory_stats()
        runtime.model.zero_grad(set_to_none=True)
        restore_rng(baseline_rng)
        with saved_activation_offload(
            runtime.model,
            max_host_bytes=max_host_bytes,
            scratch_directory=scratch_directory,
            max_disk_bytes=max_disk_bytes,
            fixture=fixture,
            disk_backend=disk_backend,
            max_device_bytes=max_device_bytes,
        ) as offload:
            for index in range(128):
                report("row_forward", index)
                result, trace = forward(index)
                evidence["traces"].append(trace)
                if index in indices:
                    rng = capture_rng()
                    try:
                        for name, buffers in tensors["channels"].items():
                            report("diagnostic_" + name, index)
                            values = torch.autograd.grad(
                                result[name] / 128,
                                tuple(params.values()),
                                retain_graph=True,
                                allow_unused=True,
                            )
                            counts["diagnostic_gradient_calls"] += 1
                            for key, gradient in zip(params, values, strict=True):
                                if gradient is not None:
                                    require(
                                        bool(torch.isfinite(gradient).all()),
                                        "Nonfinite diagnostic gradient",
                                    )
                                    buffers[key].add_(gradient.detach().cpu())
                    finally:
                        restore_rng(rng)
                report("row_backward", index)
                (result["loss"] / 128).backward()
                counts["backward_calls"] += 1
                if index == 0:
                    tensors["offload_row0"] = gradients()
                    evidence["offload_row0_loss"] = {
                        k: float(v.detach())
                        for k, v in result.items()
                        if k in ("loss", "native", "aux", "kl")
                    }
                del result
                completed_rows += 1
                report("row_finished", index)
                if max_device_bytes and index == 3:
                    window = window_measurement(cuda_observer.records, indices)
                    evidence["technical_window"] = window
                    if window["status"] != "CONTINUE_SAME_PROCESS":
                        raise TechnicalWindowStop(
                            "Fixed first-four-row technical throughput condition failed"
                        )
        evidence["offload"] = offload.summary()
        if not fixture:
            evidence["offload_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        tensors["accumulated"] = gradients()
        if scratch_directory is not None:
            verify_storage(
                evidence["offload"],
                scratch_directory,
                max_host_bytes,
                max_disk_bytes,
                fixture=fixture,
                disk_backend=disk_backend,
                max_device_bytes=max_device_bytes,
            )
    except BaseException as exc:
        exc.validation_partial = (evidence, tensors)
        raise
    finally:
        # Restoration is mandatory even when progress storage fails.
        try:
            report("restoration")
        finally:
            evidence["restoration"] = preserved.restore()
    try:
        report("recompute")
        evidence["comparisons"] = recompute(evidence, tensors, originals)
    except BaseException as exc:
        exc.validation_partial = (evidence, tensors)
        raise
    report("measurement_finished")
    evidence["scope"] = "SYNTHETIC_CPU_ONLY" if fixture else "ACTUAL_CUDA_STORAGE_VALIDATION"
    return evidence, tensors


def measure_fixture(
    runtime,
    records,
    credits,
    originals,
    *,
    max_host_bytes=10_000_000,
    scratch_directory=None,
    max_disk_bytes=None,
    disk_backend="file_v2",
    progress=None,
    max_device_bytes=0,
    fixture_clock=None,
):
    require(all(p.device.type == "cpu" for p in runtime.model.parameters()), "CPU fixture only")
    observer = CudaObservations(fixture=True, clock=fixture_clock) if max_device_bytes else None
    if observer is not None:
        observer.observe("before_load")
        observer.observe("after_load")
    return _measure(
        runtime,
        records,
        credits,
        runtime.reference,
        originals,
        max_host_bytes=max_host_bytes,
        scratch_directory=scratch_directory,
        max_disk_bytes=max_disk_bytes,
        fixture=True,
        disk_backend=disk_backend,
        progress=progress,
        max_device_bytes=max_device_bytes,
        cuda_observer=observer,
    )


def _originals(contract):
    from .execution import bound_ref, strict_ref

    root = Path(contract["experiment_root"])
    ready = bound_ref(root, contract["original_ready_ref"])
    run = (root / contract["original_ready_ref"]["path"]).parent.parent.parent
    rollout = bound_ref(run, ready["references"]["rollout"])
    credit = bound_ref(run, ready["references"]["credit"])
    audit_ref = contract["original_audit_ref"]
    path = Path(audit_ref["path"])
    if not path.is_absolute():
        path = root / path
    events = [
        json.loads(line)
        for line in strict_ref(path, audit_ref["sha256"]).splitlines()
        if line.strip()
    ]
    selected = [e for e in events if e.get("status") == "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"]
    require(len(selected) == 1, "Original failure event ambiguous")
    event = selected[0]
    require(
        event["optimizer_started"] is False and event["parameter_state_changed"] is False,
        "Original update may have started",
    )
    records = rollout["records"]
    require(
        len(records) == 128 and len(event["original_trajectories"]) == 128,
        "Original READY coverage incomplete",
    )
    for row, original in zip(records, event["original_trajectories"], strict=True):
        require(
            all(row.get(k) == v for k, v in original.items()), "Original failure trajectory differs"
        )
    return ready, records, credit["groups"], event["probability_vectors"]


def _technical_runtime(contract, ready, records, account, *, stage_callback=None):
    setup_started = time.monotonic()

    def measured(name, started):
        if stage_callback is not None:
            stage_callback(name, time.monotonic() - started)

    def starting(name):
        if stage_callback is not None:
            stage_callback(name + "_started", 0.0)
        return time.monotonic()

    import torch

    from mm_core.vl_runtime import seed_all
    from mm_dev.runtime import actual_cuda_identity, configure_audited_backend, reference_parameters
    from sr_f3_ce.runtime import configure_parent_training
    from sr_f5.runtime import NUMERICS_V2
    from sr_f5.runtime import Runtime as FP32Runtime
    from sr_f6.runtime import (
        PROTOCOL_VERSION,
        backend_protocol,
        validate_generation_config,
        verify_binding_inputs,
    )
    from srf6_ref.core import seed_for

    class TechnicalRuntime(FP32Runtime):
        def generate_group(self, *args, **kwargs):
            raise PermissionError("Technical validator cannot sample")

        def _generate_native_group(self, *args, **kwargs):
            raise PermissionError("Technical validator cannot sample")

        def prepare_input(self, model_input, root, *, protocol):
            prepared = super().prepare_input(model_input, root, protocol=backend_protocol(protocol))
            prepared["routing"]["sr_f6_protocol_version"] = PROTOCOL_VERSION
            return prepared

        def batch_reference_forward(self, rows, reference):
            with reference_parameters(self.model, reference):
                return self.batch_sequence_forward(rows, purpose="training_reference", grad=False)

        def batch_sequence_forward(self, rows, *, purpose, grad):
            require(
                len(rows) == 1 and purpose in ("training_reference", "training_policy"),
                "Fixed microbatch required",
            )
            require(grad is (purpose == "training_policy"), "Unexpected forward mode")
            row = rows[0]
            require(
                row["row_id"] in self.allowed
                and self.allowed[row["row_id"]]
                == {k: v for k, v in row.items() if k != "prepared"},
                "Unregistered retained trajectory",
            )
            prepared = row["prepared"]
            _verify_prepared(row, prepared)
            inputs = dict(prepared["inputs"])
            length = inputs["input_ids"].shape[-1]
            completion = inputs["input_ids"].new_tensor([row["tokens"]])
            require(1 <= completion.shape[1] <= 1536, "Original completion boundary differs")
            inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], -1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(completion)], -1
            )
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(completion)], -1
            )
            return self._native_forward(
                inputs,
                completion_start=length,
                lengths=[completion.shape[1]],
                purpose=purpose,
                grad=grad,
            )

    producer_src = Path(contract["producer"]["deployment"]) / "src"
    loaded_sources = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("sr_f", "srf6_ref", "mm_core", "mm_dev")) and not name.startswith(
            "sr_f6_memory_repair"
        ):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                require(
                    path.is_relative_to(producer_src),
                    "Technical computation loaded outside original producer: " + name,
                )
                loaded_sources[name] = str(path)
    root = Path(contract["experiment_root"])
    config = json.loads((root / "CONFIG.json").read_bytes())
    validate_generation_config(config)
    started = starting("verify_binding_before")
    bound, before = verify_binding_inputs(root, config, "b1")
    measured("verify_binding_before", started)
    backend_identity = configure_audited_backend()
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "One actual CUDA device required",
    )
    seed_all(seed_for(config["master_seed"], "load", "b1"))
    adapter = Path(config["model"]["parents"]["b1"])
    started = starting("model_load")
    runtime = TechnicalRuntime(
        config["model"]["model_path"],
        adapter_path=str(adapter),
        output_root=root,
        protocol_revision="v2",
        account=account,
    )
    if stage_callback is not None:
        torch.cuda.synchronize()
    measured("model_load", started)
    started = starting("configure_training_and_fp32")
    configure_parent_training(
        runtime, parent_adapter_config=json.loads((adapter / "adapter_config.json").read_bytes())
    )
    source_dtype = runtime.identity["dtype"]
    runtime.model.float()
    if stage_callback is not None:
        torch.cuda.synchronize()
    measured("configure_training_and_fp32", started)
    require(
        all(
            p.dtype == torch.float32 and p.device.type == "cuda" for p in runtime.model.parameters()
        ),
        "Actual model must be full CUDA FP32",
    )
    runtime._srf5_numerics = dict(NUMERICS_V2)
    runtime.allowed = {r["row_id"]: r for r in records}
    runtime.reference = trainable_state(runtime.model)
    require(
        state_hash(runtime.reference) == ready["spec"]["policy_before_sha256"],
        "Original step-zero parent tensors differ",
    )
    runtime.identity.update(
        run_root=str(root),
        source_sha256=records[0]["source_sha256"],
        reference_hash=state_hash(runtime.reference),
        reference_parent="b1",
        parent_id="b1",
        run_id=records[0]["run_id"],
        step=0,
        diagnostic_row_indices=ready["spec"]["identity"]["diagnostic_row_indices"],
        technical_consumer=validation_schema(contract),
        loaded_producer_modules=loaded_sources,
        numerics=dict(NUMERICS_V2),
        model_cast_from_dtype=source_dtype,
        forward_parameter_dtypes=sorted(
            {str(p.dtype).removeprefix("torch.") for p in runtime.model.parameters()}
        ),
        forward_float_buffer_dtypes=sorted(
            {
                str(p.dtype).removeprefix("torch.")
                for p in runtime.model.buffers()
                if p.is_floating_point()
            }
        ),
        input_float_dtype="float32",
        probability_dtype="float32",
        backend_identity=backend_identity,
        gradient_checkpointing=bool(getattr(runtime, "_sr_f12_checkpointing", False)),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        trainable_parameters={
            name: list(p.shape) for name, p in runtime.model.named_parameters() if p.requires_grad
        },
        dtype="float32",
        hardware=actual_cuda_identity(),
        parent_content_hash=bound["parents"]["b1"]["content_sha256"],
        bound_files_before=before,
    )
    started = starting("verify_binding_after")
    _, runtime.identity["bound_files_after"] = verify_binding_inputs(root, config, "b1")
    measured("verify_binding_after", started)
    measured("runtime_setup_total", setup_started)
    return runtime


def _ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run_validation(contract_path, expected_contract_sha256, output):
    import torch

    from .execution import verify_contract

    contract = verify_contract(contract_path, expected_contract_sha256)
    schema = validation_schema(contract)
    max_device_bytes = contract.get("max_device_bytes", 0)
    output = Path(output)
    canonical = Path("/projects/_ssd/varunssd/louis-ssvc")
    require(
        output.is_absolute() and output.parent.resolve().is_relative_to(canonical),
        "Technical output must be canonical",
    )
    require(
        output == Path(contract_path).parent / "VALIDATION",
        "Validation must use its registered new operations directory",
    )
    require(not output.exists(), "Preserve existing validation attempt")
    output.mkdir()
    write_once(
        output / "ATTEMPT.json",
        dict(schema=schema, contract_sha256=expected_contract_sha256, plan=PLAN),
    )
    ledger = []

    def account(kind, count, metadata):
        require(
            kind in ("extra_forward_sequences", "diagnostic_gradient_calls"),
            "Unregistered model call: " + kind,
        )
        ledger.append(dict(kind=kind, count=count, metadata=metadata))

    observer = None
    try:
        ready, records, credits, originals = _originals(contract)
        if max_device_bytes:
            observer = CudaObservations()
            observer.observe("before_load")
        runtime = _technical_runtime(contract, ready, records, account)
        if observer is not None:
            observer.observe("after_load")
        progress = ProgressWriter(
            output,
            contract_sha256=expected_contract_sha256,
            repair_revision=contract["repair_source"]["revision"],
            cuda_observer=observer,
        )
        evidence, tensors = _measure(
            runtime,
            records,
            credits,
            runtime.reference,
            originals,
            max_host_bytes=contract["max_host_bytes"],
            scratch_directory=output / "STORAGE",
            max_disk_bytes=contract["max_disk_bytes"],
            disk_backend=contract["storage_backend"],
            progress=progress,
            max_device_bytes=max_device_bytes,
            cuda_observer=observer,
        )
        evidence.update(
            runtime_identity=runtime.identity,
            ledger=ledger,
            contract_sha256=expected_contract_sha256,
            original_ready_ref=contract["original_ready_ref"],
            prior_allocation_failure_ref=contract["old_allocation_failure_ref"],
            producer_revision=contract["producer"]["revision"],
            repair_revision=contract["repair_source"]["revision"],
        )
        if max_device_bytes:
            try:
                verify_cuda_observations(
                    evidence, runtime.identity["hardware"], evidence["diagnostic_indices"]
                )
            except BaseException as exc:
                exc.validation_partial = (evidence, tensors)
                raise
        verify_contract(contract_path, expected_contract_sha256)
        torch.save(tensors, output / "TENSORS.pt")
        write_once(output / "EVIDENCE.json", evidence)
        result = dict(
            schema=schema,
            status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
            contract_sha256=expected_contract_sha256,
            evidence_ref=_ref(output / "EVIDENCE.json"),
            tensors_ref=_ref(output / "TENSORS.pt"),
            verifier_source_ref=_ref(Path(__file__).resolve()),
            plan=PLAN,
        )
        write_once(output / "RESULT.json", result)
        return verify_validation(
            contract_path, expected_contract_sha256, _ref(output / "RESULT.json")
        )
    except BaseException as exc:
        if observer is not None:
            write_once(output / "CUDA_OBSERVATIONS_PARTIAL.json", observer.evidence())
        if hasattr(exc, "validation_partial"):
            from sr_f6.training import _safe

            partial, partial_tensors = exc.validation_partial
            write_once(output / "PARTIAL_EVIDENCE.json", _safe(partial))
            torch.save(partial_tensors, output / "PARTIAL_TENSORS.pt")
        write_once(
            output / "FAILURE.json",
            dict(
                status="FAILED_PRESERVED",
                error_type=type(exc).__name__,
                error=str(exc),
                ledger=ledger,
            ),
        )
        raise


def verify_validation(contract_path, expected_contract_sha256, validation_ref):
    import torch

    from .execution import strict_ref, verify_contract

    contract = verify_contract(contract_path, expected_contract_sha256)
    schema = validation_schema(contract)
    max_device_bytes = contract.get("max_device_bytes", 0)
    result_path = Path(validation_ref["path"])
    require(
        result_path == Path(contract_path).parent / "VALIDATION/RESULT.json",
        "Validation result outside registered new operations directory",
    )
    result = json.loads(strict_ref(result_path, validation_ref["sha256"]))
    require(not (result_path.parent / "FAILURE.json").exists(), "Failed validation preserved")
    require(
        result["schema"] == schema
        and result["status"] == "VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH"
        and result["plan"] == PLAN
        and result["contract_sha256"] == expected_contract_sha256,
        "Validation identity differs",
    )
    require(
        result["verifier_source_ref"] == _ref(Path(__file__).resolve()),
        "Actual verifier source differs",
    )
    require(
        result_path.name == "RESULT.json"
        and Path(result["evidence_ref"]["path"]) == result_path.parent / "EVIDENCE.json"
        and Path(result["tensors_ref"]["path"]) == result_path.parent / "TENSORS.pt",
        "Validation evidence must be one complete attempt",
    )
    evidence = json.loads(
        strict_ref(Path(result["evidence_ref"]["path"]), result["evidence_ref"]["sha256"])
    )
    tensor_path = Path(result["tensors_ref"]["path"])
    tensor_bytes = strict_ref(tensor_path, result["tensors_ref"]["sha256"])
    tensors = torch.load(io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True)
    ready, records, _, originals = _originals(contract)
    require(
        evidence["scope"] == "ACTUAL_CUDA_STORAGE_VALIDATION",
        "Synthetic evidence cannot authorize recovery",
    )
    require(
        evidence["prior_allocation_failure_ref"] == contract["old_allocation_failure_ref"]
        and evidence["contract_sha256"] == expected_contract_sha256
        and evidence["producer_revision"] == contract["producer"]["revision"]
        and evidence["repair_revision"] == contract["repair_source"]["revision"],
        "Execution source binding differs",
    )
    require(
        evidence["diagnostic_indices"] == ready["spec"]["identity"]["diagnostic_row_indices"],
        "Diagnostic selection changed",
    )
    for trace, row in zip(evidence["traces"], records, strict=True):
        require(
            all(
                trace[key] == row[key]
                for key in ("row_id", "qid", "tokens", "sampler_logprobs", "token_path_sha256")
            ),
            "Actual original row binding differs",
        )
    from sr_f5.runtime import NUMERICS_V2

    identity = evidence["runtime_identity"]
    require(
        identity["technical_consumer"] == schema
        and identity["dtype"] == "float32"
        and identity["numerics"] == NUMERICS_V2
        and identity["parent_id"] == "b1"
        and identity["step"] == 0
        and identity["reference_hash"] == ready["spec"]["policy_before_sha256"]
        and identity["source_sha256"] == records[0]["source_sha256"],
        "Actual runtime identity differs",
    )
    require(
        identity["forward_parameter_dtypes"] == ["float32"]
        and all(v == "float32" for v in identity["forward_float_buffer_dtypes"])
        and identity["input_float_dtype"] == identity["probability_dtype"] == "float32"
        and identity["tf32_matmul"] is False
        and identity["tf32_cudnn"] is False
        and identity["gradient_checkpointing"] is False
        and identity["trainable_parameters"] == evidence["parameter_shapes"],
        "Actual numerical recipe changed",
    )
    backend = identity["backend_identity"]
    require(
        backend
        == dict(
            deterministic_algorithms=True,
            deterministic_warn_only=False,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            cudnn_benchmark=False,
            cublas_workspace_config=":4096:8",
        ),
        "Actual deterministic backend differs",
    )
    hardware = identity["hardware"]
    require(
        hardware["cuda_visible_device_count"] == 1
        and "PRO 6000" in hardware["cuda_name"]
        and bool(hardware["cuda_uuid"]),
        "Actual registered GPU identity missing",
    )
    loaded = identity["loaded_producer_modules"]
    require(
        bool(loaded)
        and all(
            Path(p).is_relative_to(Path(contract["producer"]["deployment"]) / "src")
            for p in loaded.values()
        ),
        "Loaded producer closure differs",
    )
    ledger = evidence["ledger"]
    require(
        len(ledger) == 258
        and all(e["kind"] == "extra_forward_sequences" and e["count"] == 1 for e in ledger),
        "Native call ledger incomplete",
    )
    require(
        contract["max_host_bytes"] == 8 * 1024**3 and contract["max_disk_bytes"] == 96 * 1024**3,
        "Frozen hybrid limits changed",
    )
    verify_storage(
        evidence["offload"],
        result_path.parent / "STORAGE",
        contract["max_host_bytes"],
        contract["max_disk_bytes"],
        disk_backend=contract["storage_backend"],
        max_device_bytes=max_device_bytes,
    )
    if max_device_bytes:
        require(evidence["max_device_bytes"] == max_device_bytes, "Actual device bound differs")
        verify_cuda_observations(evidence, hardware, evidence["diagnostic_indices"])
    computed = recompute(evidence, tensors, originals)
    require(
        computed == evidence["comparisons"], "Stored comparison differs from full raw recomputation"
    )
    return dict(
        status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
        contract_sha256=expected_contract_sha256,
        producer_revision=contract["producer"]["revision"],
        repair_revision=contract["repair_source"]["revision"],
        zero_new_samples=True,
        zero_optimizer_updates=True,
        generated_responses=0,
        optimizer_updates=0,
        compared_gradient_rows=[0],
        offload_rows=128,
        original_ready_ref=contract["original_ready_ref"],
        verifier_source_ref=result["verifier_source_ref"],
        validation_ref=validation_ref,
        scope=PLAN,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(run_validation(args.contract, args.contract_sha256, args.output), sort_keys=True)
    )


if __name__ == "__main__":
    main()
