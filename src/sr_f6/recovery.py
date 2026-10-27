"""Registered b0/U11 recovery import, execution journals and exact comparison.

This module never samples, updates parameters or signs a scientific gate. Imported
steps 0..4 retain the original logical identity; steps 5..12 must be generated in
the independent physical lane. CPU fixtures are permanently SYNTHETIC_ONLY.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import io
import json
import math
import os
import re
import socket
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath

from mm_core.training import state_hash
from sr_f3_ce.training import finite_tree

from . import registration
from .transactions import (
    RunTransactions,
    _canonical,
    _digest,
    _file_hash,
    _fsync_dir,
    _optimizer_cursor,
    _plain_path,
    _read_bytes,
    _require,
    _sha,
    _write_json,
)

DIRECTORY = "PHASE_RUNS/PILOT_V1/RECOVERY/b0/r0/U11"
BINDING = DIRECTORY + "/RECOVERY_BINDING.json"
FORMAT = "srf6-recovery-v1"
_CONTENT_CACHE = {}
_PREFIX_CACHE = {}
_PROCESS_CACHE = {}
# Exact structural locations, never recursive key deletion inside model output.
ROW_EXECUTION_KEYS = ("task_id", "generation_seconds")
MODEL_EXECUTION_KEYS = ("task_id", "audit_root")
UPDATE_EXECUTION_KEYS = ("training_seconds",)
EVIDENCE_EXECUTION_KEYS = ("at", "audit_root", "task_id", "source_audit_event")
ENVIRONMENT_KEYS = (
    "dtype",
    "numerics",
    "attention_backend",
    "torch_version",
    "transformers_version",
    "forward_parameter_dtypes",
    "probability_dtype",
)
HARDWARE_KEYS = (
    "cuda_name",
    "cuda_total_memory_bytes",
    "compute_capability",
    "cuda_runtime",
    "driver_version",
)


def _process_identity():
    pid = os.getpid()
    if pid not in _PROCESS_CACHE:
        if Path("/proc/self/stat").is_file():
            fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
            value = dict(
                kind="LINUX_PROC",
                hostname=socket.gethostname(),
                pid=pid,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                start_ticks=int(fields[19]),
            )
        else:
            value = dict(
                kind="SYNTHETIC_PROCESS",
                hostname=socket.gethostname(),
                pid=pid,
                process_instance=uuid.uuid4().hex,
            )
        _PROCESS_CACHE[pid] = value
    return copy.deepcopy(_PROCESS_CACHE[pid])


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        _require(math.isfinite(result), "Nonfinite JSON number")
        return result

    def bad(_):
        raise PermissionError("Nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=bad)


def _path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "Relative path required")
    value = PurePosixPath(relative)
    _require(
        not value.is_absolute()
        and ".." not in value.parts
        and value.as_posix() == relative
        and relative != ".",
        "Unsafe reference path",
    )
    return _plain_path(Path(root) / relative)


def _ref(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=_file_hash(path))


def _bound(root, ref, *, json_value=True):
    _require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256"} and _sha(ref["sha256"]),
        "Explicit content reference required",
    )
    path = _path(root, ref["path"])
    stamp = _stamp(path)
    key = (str(path), ref["sha256"], json_value)
    cached = _CONTENT_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return copy.deepcopy(cached[1]) if json_value else cached[1]
    data = _read_bytes(path)
    _require(hashlib.sha256(data).hexdigest() == ref["sha256"], "Referenced content differs")
    value = _json(data) if json_value else data
    _CONTENT_CACHE[key] = (stamp, value)
    return copy.deepcopy(value) if json_value else value


def _stamp(path):
    info = _plain_path(path).stat()
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino


def _prefix_stamps(primary, destination, inventory):
    return tuple(
        (str(path), _stamp(path))
        for item in inventory
        for path in (_path(primary, item["path"]), _path(destination, item["path"]))
    )


def _save(root, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)
    _fsync_dir(path.parent)
    return _ref(root, path)


def _events(path):
    if not path.exists():
        return []
    return [_json(line) for line in _read_bytes(path).splitlines() if line]


def _counts(events, task_id, parent):
    counts = {}
    for event in events:
        _require(
            event.get("task_id") == task_id and event.get("parent") == parent,
            "Accounting execution identity differs",
        )
        n = event.get("count")
        _require(
            type(n) is int and n >= 0 and isinstance(event.get("kind"), str),
            "Invalid actual accounting",
        )
        counts[event["kind"]] = counts.get(event["kind"], 0) + n
    return counts


def _registration(root, phase_context, pilot_spec_sha256):
    value = registration.verify_pilot_registration(
        root, phase_context=phase_context, expected_spec_sha256=pilot_spec_sha256
    )
    spec = value["spec"]
    item = next(p for p in spec["paths"] if (p["parent"], p["arm"]) == ("b0", "U11"))
    primary = _path(root, item["audit_root"])
    identity = _json(_read_bytes(primary / "RUN_IDENTITY.json"))
    _require(
        identity["lane"] == "PRIMARY"
        and identity["repeat"] == 0
        and identity["max_steps"] == 12
        and identity["phase"] == "PILOT_V1",
        "Only registered b0/U11 primary identity can be imported",
    )
    return spec, primary, identity


def _range(path, count, offset=0):
    _require(
        type(count) is int and count > 0 and type(offset) is int and offset >= 0,
        "Explicit positive original audit byte range required",
    )
    path = _plain_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        _require(
            os.fstat(stream.fileno()).st_size >= offset + count, "Truncated original audit bytes"
        )
        stream.seek(offset)
        data = stream.read(count)
    _require(len(data) == count, "Incomplete original audit slice")
    return data


def _audit_event(directory, evidence, *, required=False):
    reference = evidence.get("source_audit_event")
    if reference is None:
        _require(not required, "Actual original preupdate audit slice required")
        return None
    _require(
        reference.get("path") == "NUMERICAL_TOKEN_AUDIT.jsonl"
        and _sha(reference.get("sha256"))
        and reference.get("binding")
        == "Original append-only event bytes; later events do not change this slice",
        "Unregistered original preupdate slice",
    )
    data = _range(
        _path(directory, reference["path"]), reference["byte_count"], reference["byte_offset"]
    )
    _require(
        hashlib.sha256(data).hexdigest() == reference["sha256"]
        and _json(data) == _without(evidence, ("source_audit_event",)),
        "Original preupdate audit slice differs from saved event",
    )
    return reference


def _audit_prefixes(tx, markers):
    lengths = {}
    for marker in markers[1:5]:
        evidence = _bound(tx.directory, marker["references"]["evidence"])
        ref = _audit_event(tx.directory, evidence)
        if ref is not None:
            lengths[ref["path"]] = max(
                lengths.get(ref["path"], 0), ref["byte_offset"] + ref["byte_count"]
            )
    return lengths


def _inventory_bytes(directory, item):
    path = _path(directory, item["path"])
    if "source_byte_count" in item:
        return _range(path, item["source_byte_count"])
    return _read_bytes(path)


def _state(tx, marker, *, gpu=False):
    """Verify one delivery/import boundary without touching a live model or RNG."""
    import numpy as np
    import torch

    step = marker["step"]
    folder = tx.checkpoints / f"step-{step:03d}"
    _require(_file_hash(folder / "state.pt") == marker["state_sha256"], "State hash differs")
    value = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
    _require(
        value.get("identity") == tx.identity
        and value.get("step") == step
        and value.get("sampler_state") == marker["sampler_state"],
        "State metadata differs",
    )
    _require(not gpu or len(value["cuda_rng"]) == 1, "Actual single-GPU checkpoint RNG required")
    policy = value["policy"]
    _require(
        len(policy) == len(marker["parameter_names"])
        and set(policy) == set(marker["parameter_names"])
        and finite_tree(policy)
        and state_hash(policy) == marker["policy_sha256"],
        "Adapter content differs",
    )
    _optimizer_cursor(
        value["optimizer"],
        value["scheduler"],
        step,
        [policy[name] for name in marker["parameter_names"]],
    )
    nr = value["numpy_rng"]
    rng = dict(
        python=value["python_rng"],
        numpy=(nr[0], np.asarray(nr[1], dtype=np.uint32), *nr[2:]),
        cpu=value["torch_rng"],
        cuda=value["cuda_rng"],
    )
    _require(state_hash(rng) == marker["rng_sha256"], "RNG content differs")
    if step:
        update = _bound(
            tx.directory,
            dict(path=f"checkpoints/step-{step:03d}/UPDATE.json", sha256=marker["update_sha256"]),
        )
        for ref in marker["references"].values():
            _bound(tx.directory, ref)
        evidence = _bound(tx.directory, marker["references"]["evidence"])
        _audit_event(tx.directory, evidence, required=gpu)
        tx._evidence(
            marker["references"], update, step, marker["policy_before_sha256"], marker["rng_sha256"]
        )
        _require(update["policy_after_sha256"] == marker["policy_sha256"], "Update policy differs")
    return value


def _without(value, keys):
    return {key: copy.deepcopy(item) for key, item in value.items() if key not in keys}


def _row(row):
    result = _without(row, ROW_EXECUTION_KEYS)
    if "model_identity" in result:
        result["model_identity"] = _without(result["model_identity"], MODEL_EXECUTION_KEYS)
    return result


def _execution(root, ref, directory, identity, *, require_continuous=False):
    """Recompute journal facts from bound receipts, not its claimed status."""
    end = _bound(root, ref)
    task = end.get("task_id")
    _require(
        isinstance(task, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task),
        "Invalid execution identifier",
    )
    folder = directory / "EXECUTIONS" / task
    _require(_path(root, ref["path"]) == folder / "END.json", "Execution receipt path differs")
    begin = _bound(root, end["begin_reference"])
    rows = [
        _json(line)
        for line in _bound(root, end["commits_reference"], json_value=False).splitlines()
    ]
    start_marker = _json(
        _read_bytes(directory / "checkpoints" / f"step-{begin['start_step']:03d}" / "COMMIT.json")
    )
    _require(
        begin.get("format") == FORMAT
        and begin.get("status") == "BEGIN"
        and end.get("status") == "SEGMENT_COMPLETE"
        and end.get("scope") in ("SERVER_ACTUAL", "SYNTHETIC_ONLY")
        and type(begin.get("restored")) is bool
        and type(begin.get("resumed_after_data")) is bool
        and begin["start_commit_sha256"] == start_marker["commit_sha256"]
        and end.get("process_identity") == begin.get("process_identity")
        and begin.get("pid") == begin.get("process_identity", {}).get("pid"),
        "Invalid execution segment beginning/terminal",
    )
    _require(
        end["format"] == FORMAT
        and end["identity"] == identity
        and begin["identity"] == identity
        and begin["task_id"] == task
        and end["scope"] == begin["scope"]
        and end["storage_lane"] == begin["storage_lane"],
        "Execution identity differs",
    )
    _require(
        _path(root, end["begin_reference"]["path"]) == folder / "BEGIN.json"
        and _path(root, end["commits_reference"]["path"]) == folder / "COMMITS.jsonl"
        and not (folder / "FAILURE.json").exists(),
        "Unclosed/failed execution",
    )
    _require(
        [r["step"] for r in rows] == list(range(begin["start_step"] + 1, 13)),
        "Execution segment is incomplete",
    )
    new, new_tokens = 0, 0
    generation_events = []
    for row in rows:
        n, reused = row["new_model_calls"], row["reused_complete_groups"]
        _require(
            type(n) is int
            and 0 <= n <= 128
            and n % 8 == 0
            and type(reused) is int
            and reused == 16 - n // 8,
            "Invalid cache/call facts",
        )
        marker = _bound(root, row["commit_reference"])
        _require(
            marker["identity"] == identity
            and marker["step"] == row["step"]
            and marker["commit_sha256"] == row["commit_sha256"]
            and _path(root, row["commit_reference"]["path"])
            == directory / "checkpoints" / f"step-{row['step']:03d}" / "COMMIT.json",
            "Execution commit differs",
        )
        rollout = _bound(directory, marker["references"]["rollout"])
        _require(
            row["rollout_reference"] == marker["references"]["rollout"], "Rollout reference differs"
        )
        actual_new = [r for r in rollout["records"] if r.get("task_id") == task]
        _require(len(actual_new) == n, "New generation belongs to another execution")
        for offset in range(0, 128, 8):
            group = rollout["records"][offset : offset + 8]
            selected = [r for r in group if r.get("task_id") == task]
            _require(len(selected) in (0, 8), "Only entire completed G8 may be reused")
            if selected:
                seed = group[0].get("group_seed", group[0]["seed"])
                generation_events.append(("completion_attempts", 8, seed, "structured"))
                generation_events.extend(
                    ("generated_tokens", len(r["tokens"]), seed, i) for i, r in enumerate(group)
                )
        new_tokens += sum(len(r["tokens"]) for r in actual_new)
        new += n
    events = [
        _json(line)
        for line in _bound(root, end["accounting_reference"], json_value=False).splitlines()
    ]
    _require(
        _path(root, end["accounting_reference"]["path"])
        == directory / "accounting" / (task + ".jsonl"),
        "Accounting path differs",
    )
    counts = _counts(events, task, identity["parent"])
    observed_generation = [
        (
            e["kind"],
            e["count"],
            e.get("metadata", {}).get("group_seed"),
            e.get("metadata", {}).get("protocol")
            if e["kind"] == "completion_attempts"
            else e.get("metadata", {}).get("row_index"),
        )
        for e in events
        if e["kind"] in ("completion_attempts", "generated_tokens")
    ]
    _require(
        Counter(observed_generation) == Counter(generation_events),
        "Actual original G8/draw accounting differs",
    )
    _require(
        counts.get("completion_attempts", 0) == new
        and counts.get("generated_tokens", 0) == new_tokens,
        "Actual completion/token budget differs",
    )
    _require(
        end["new_model_calls"] == new
        and end["end_step"] == 12
        and end["commit_sha256"]
        == (rows[-1]["commit_sha256"] if rows else begin["start_commit_sha256"]),
        "Execution terminal facts differ",
    )
    runtime = _bound(root, begin["runtime_identity_reference"])
    _require(
        _path(root, begin["runtime_identity_reference"]["path"])
        == directory / "runtime_identity" / (task + ".json"),
        "Runtime receipt path differs",
    )
    for key, expected in (
        ("task_id", task),
        ("run_id", identity["run_id"]),
        ("source_sha256", identity["source_sha256"]),
        ("data_hash", identity["data_sha256"]),
        ("reference_hash", identity["reference_sha256"]),
        ("parent_id", identity["parent"]),
        ("phase", "PILOT_V1"),
        ("execution_mode", "TRAINING"),
    ):
        _require(runtime.get(key) == expected, "Runtime evidence identity differs: " + key)
    if end["scope"] == "SERVER_ACTUAL":
        hardware = runtime.get("hardware", {})
        process = begin.get("process_identity", {})
        _require(
            process.get("kind") == "LINUX_PROC"
            and bool(process.get("hostname"))
            and type(process.get("pid")) is int
            and process["pid"] > 0
            and bool(process.get("boot_id"))
            and type(process.get("start_ticks")) is int
            and process["start_ticks"] > 0,
            "Actual OS process identity required",
        )
        _require(
            hardware.get("cuda_visible_device_count") == 1
            and "RTX PRO 6000" in str(hardware.get("cuda_name", "")).upper()
            and bool(hardware.get("cuda_uuid"))
            and bool(hardware.get("cuda_runtime"))
            and bool(hardware.get("driver_version"))
            and runtime.get("step") == 0
            and runtime.get("policy_hash") == identity["reference_sha256"]
            and runtime.get("fresh_optimizer") is True
            and runtime.get("dtype") == "float32"
            and runtime.get("phase_identity_sha256") == identity.get("phase_identity_sha256")
            and _sha(identity.get("phase_identity_sha256"))
            and runtime.get("config_sha256") == identity["config_sha256"],
            "Actual GPU/runtime source evidence required; synthetic cannot be upgraded",
        )
    continuous = (
        begin["storage_lane"] == "PRIMARY"
        and begin["start_step"] == 0
        and not begin["resumed_after_data"]
        and len(rows) == 12
        and all(r["new_model_calls"] == 128 and r["reused_complete_groups"] == 0 for r in rows)
    )
    # Earlier pre-data failures remain visible, but an actual previous call is not continuous.
    if continuous:
        for other in (directory / "accounting").glob("*.jsonl"):
            if other.name != task + ".jsonl":
                old = _events(other)
                if any(e.get("kind") == "completion_attempts" and e.get("count", 0) for e in old):
                    continuous = False
    _require(
        end["continuity_verified"] == continuous, "Claimed continuity differs from actual facts"
    )
    if require_continuous:
        _require(continuous, "Registered recovery requires the actual continuous primary baseline")
    return dict(
        end=end,
        begin=begin,
        commits=rows,
        runtime=runtime,
        counts=counts,
        continuity_verified=continuous,
        new_model_calls=new,
    )


class ExecutionJournal:
    """One actual process segment while holding a RunTransactions lock.

    General resumes and complete-cache reuse are recorded honestly. Only a primary
    segment starting at zero, with all 1536 newly generated responses and no prior
    data attempt, can establish the unique continuous-baseline fact.
    """

    def __init__(
        self,
        tx,
        runtime,
        *,
        experiment_root,
        storage_lane="PRIMARY",
        restored=False,
        resumed_after_data=False,
        recovery_binding_ref=None,
    ):
        self.tx, self.runtime = tx, runtime
        self.root = _plain_path(experiment_root)
        self.lane, self.restored, self.resumed = storage_lane, restored, resumed_after_data
        self.binding_ref = copy.deepcopy(recovery_binding_ref)
        self.scope = "SERVER_ACTUAL"
        self.active = self.finished = False

    @classmethod
    def synthetic(cls, *args, **kwargs):
        value = cls(*args, **kwargs)
        value.scope = "SYNTHETIC_ONLY"
        return value

    def __enter__(self):
        import torch

        self.tx._locked()
        self.tx._runtime(self.runtime)
        process_identity = _process_identity()
        _require(not self.active and self.lane in ("PRIMARY", "RECOVERY"), "Invalid journal state")
        _require(
            type(self.restored) is bool and type(self.resumed) is bool,
            "Explicit resume flags required",
        )
        identity = self.runtime.identity
        task = identity.get("task_id")
        _require(
            isinstance(task, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task),
            "Unique safe task identifier required",
        )
        _require(
            _plain_path(identity["audit_root"]) == self.tx.directory
            and _plain_path(identity["run_root"]) == self.root,
            "Runtime physical storage differs",
        )
        markers = self.tx.published()
        _require(bool(markers), "Committed step zero/imported prefix required")
        self.start = markers[-1]
        from mm_core.training import trainable_state

        _require(
            identity.get("step") == self.start["step"]
            and state_hash(trainable_state(self.runtime.model)) == self.start["policy_sha256"],
            "Journal must begin at the actual restored committed policy",
        )
        _require(
            self.start["step"] in range(0 if self.lane == "PRIMARY" else 4, 13),
            "Invalid segment start",
        )
        if self.lane == "RECOVERY":
            _require(
                self.tx.directory == self.root / DIRECTORY and self.binding_ref is not None,
                "Only the fixed registered recovery lane is allowed",
            )
            binding = _bound(self.root, self.binding_ref)
            _require(
                binding["logical_identity"] == self.tx.identity, "Recovery logical identity differs"
            )
            _require(self.binding_ref["path"] == BINDING, "Fixed recovery binding required")
            if self.scope == "SERVER_ACTUAL":
                primary_end = _bound(self.root, binding["primary_execution_ref"])
                primary_begin = _bound(self.root, primary_end["begin_reference"])
                _require(
                    primary_end["scope"] == primary_begin["scope"] == "SERVER_ACTUAL"
                    and primary_begin["process_identity"] == primary_end["process_identity"],
                    "Actual source process proof required before recovery generation",
                )
                _require(
                    primary_begin["process_identity"] != process_identity,
                    "Recovery must start a new independent OS process before generation",
                )
        if self.scope == "SERVER_ACTUAL":
            from mm_dev.runtime import actual_cuda_identity

            from .runtime import Runtime

            _require(
                type(self.runtime) is Runtime
                and torch.cuda.is_available()
                and torch.cuda.device_count() == 1
                and all(p.device.type == "cuda" for p in self.runtime.model.parameters()),
                "Actual single-GPU runtime required; use synthetic for CPU fixtures",
            )
            hardware = actual_cuda_identity()
        else:
            _require(
                all(p.device.type == "cpu" for p in self.runtime.model.parameters()),
                "Synthetic journal is CPU only",
            )
            hardware = None
        runtime_path = self.tx.directory / "runtime_identity" / (task + ".json")
        receipt = _json(_read_bytes(runtime_path))
        if hardware is not None:
            _require(receipt.get("hardware") == hardware, "Actual hardware receipt differs")
        self.accounting = self.tx.directory / "accounting" / (task + ".jsonl")
        counts = _counts(_events(self.accounting), task, self.tx.identity["parent"])
        _require(
            counts.get("completion_attempts", 0) == 0,
            "Journal must begin before its first data call",
        )
        self.folder = self.tx.directory / "EXECUTIONS" / task
        self.folder.mkdir(parents=True, exist_ok=False)
        self.task, self.pid, self.rows = task, os.getpid(), []
        self.process_identity = process_identity
        _require(
            self.scope != "SERVER_ACTUAL" or self.process_identity["kind"] == "LINUX_PROC",
            "Actual recovery process proof requires the registered Linux execution host",
        )
        self.begin = dict(
            format=FORMAT,
            status="BEGIN",
            scope=self.scope,
            identity=copy.deepcopy(self.tx.identity),
            task_id=task,
            pid=self.pid,
            process_identity=self.process_identity,
            storage_lane=self.lane,
            start_step=self.start["step"],
            start_commit_sha256=self.start["commit_sha256"],
            restored=self.restored,
            resumed_after_data=self.resumed,
            recovery_binding_ref=self.binding_ref,
            runtime_identity_reference=_ref(self.root, runtime_path),
        )
        self.begin_ref = _save(self.root, self.folder / "BEGIN.json", self.begin)
        (self.folder / "COMMITS.jsonl").touch(exist_ok=False)
        self.active = True
        return self

    def _live(self):
        self.tx._locked()
        _require(
            self.active
            and not self.finished
            and self.pid == os.getpid()
            and self.process_identity == _process_identity(),
            "Live same-process journal required",
        )

    def record_commit(self, marker, rollout_result):
        self._live()
        expected = self.start["step"] + len(self.rows) + 1
        actual = self.tx.published()[-1]
        _require(
            marker == actual and marker["step"] == expected,
            "Journal requires next published commit exactly once",
        )
        n, reused = rollout_result["new_model_calls"], rollout_result["reused_complete_groups"]
        _require(
            type(n) is int
            and 0 <= n <= 128
            and n % 8 == 0
            and type(reused) is int
            and reused == 16 - n // 8,
            "Invalid actual cache/call count",
        )
        _require(
            rollout_result["references"]["rollout"] == marker["references"]["rollout"],
            "Journal rollout differs from committed original data",
        )
        row = dict(
            step=expected,
            commit_sha256=marker["commit_sha256"],
            commit_reference=_ref(
                self.root, self.tx.checkpoints / f"step-{expected:03d}" / "COMMIT.json"
            ),
            rollout_reference=marker["references"]["rollout"],
            new_model_calls=n,
            reused_complete_groups=reused,
        )
        with (self.folder / "COMMITS.jsonl").open("ab") as stream:
            stream.write(_canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)

    def finish(self):
        self._live()
        marker = self.tx.published()[-1]
        _require(
            marker["step"] == 12 and self.start["step"] + len(self.rows) == 12,
            "Only an actual complete segment may finish",
        )
        if not self.accounting.exists():
            self.accounting.parent.mkdir(parents=True, exist_ok=True)
            self.accounting.touch(exist_ok=False)
        continuous = (
            self.lane == "PRIMARY"
            and self.start["step"] == 0
            and not self.resumed
            and all(r["new_model_calls"] == 128 for r in self.rows)
        )
        if continuous:
            for other in self.accounting.parent.glob("*.jsonl"):
                if other != self.accounting and any(
                    e.get("kind") == "completion_attempts" and e.get("count", 0)
                    for e in _events(other)
                ):
                    continuous = False
        result = dict(
            format=FORMAT,
            status="SEGMENT_COMPLETE",
            scope=self.scope,
            identity=copy.deepcopy(self.tx.identity),
            task_id=self.task,
            process_identity=self.process_identity,
            storage_lane=self.lane,
            begin_reference=self.begin_ref,
            commits_reference=_ref(self.root, self.folder / "COMMITS.jsonl"),
            accounting_reference=_ref(self.root, self.accounting),
            end_step=12,
            commit_sha256=marker["commit_sha256"],
            new_model_calls=sum(r["new_model_calls"] for r in self.rows),
            continuity_verified=continuous,
            gate_status="NOT_SIGNED",
            training_authorized=False,
        )
        # Verify the prospective terminal before publication, retaining a diagnostic on failure.
        ref = _save(self.root, self.folder / "END.json", result)
        _execution(self.root, ref, self.tx.directory, self.tx.identity)
        self.finished = True
        return ref

    def __exit__(self, exc_type, exc, _tb):
        if self.active and not self.finished:
            path = self.folder / "FAILURE.json"
            if not path.exists():
                _save(
                    self.root,
                    path,
                    dict(
                        format=FORMAT,
                        scope=self.scope,
                        task_id=self.task,
                        status="INCOMPLETE_SEGMENT",
                        error_type=exc_type.__name__ if exc_type else "UnfinishedJournal",
                        error=str(exc) if exc else "No terminal finish",
                        begin_reference=self.begin_ref,
                    ),
                )
        self.active = False


def _markers(directory, identity):
    """Read-only small-chain verification; does not acquire or impersonate a run lock."""
    from .transactions import FORMAT as TX_FORMAT
    from .transactions import _cursor

    result = []
    for path in sorted((directory / "checkpoints").iterdir()):
        _plain_path(path)
        if path.name.startswith(".staging-"):
            continue
        marker = _json(_read_bytes(path / "COMMIT.json"))
        step = len(result)
        _require(
            path.name == f"step-{step:03d}"
            and marker.get("step") == step
            and step <= 12
            and marker.get("identity") == identity
            and marker.get("status") == "COMMITTED"
            and marker.get("format") == TX_FORMAT,
            "Published prefix identity/order differs",
        )
        _require(
            marker.get("commit_sha256")
            == _digest({k: v for k, v in marker.items() if k != "commit_sha256"}),
            "Commit checksum differs",
        )
        previous = result[-1] if result else None
        _require(
            marker["previous_commit_sha256"] == (previous["commit_sha256"] if previous else None)
            and marker["policy_before_sha256"]
            == (previous["policy_sha256"] if previous else identity["reference_sha256"]),
            "Commit chain differs",
        )
        _require(
            {p.name for p in path.iterdir()}
            == (
                {"COMMIT.json", "state.pt", "UPDATE.json"} if step else {"COMMIT.json", "state.pt"}
            ),
            "Partial checkpoint",
        )
        _cursor(
            marker["sampler_state"],
            step,
            previous["sampler_state"] if previous else None,
            rollout_reference=marker["references"].get("rollout"),
        )
        result.append(marker)
    return result


def _originals(tx, marker, *, synthetic=False):
    """Verify and expose original raw/RNG facts, including their annotated bindings."""
    import numpy as np
    import torch

    rollout = _bound(tx.directory, marker["references"]["rollout"])
    if synthetic and "complete_group_references" not in rollout:
        return {"fixture_original_rows": [_row(r) for r in rollout["records"]]}
    snapshots = []

    def rng(ref, expected=None):
        payload = torch.load(
            io.BytesIO(_bound(tx.directory, ref, json_value=False)),
            map_location="cpu",
            weights_only=True,
        )
        _require(
            set(payload) == {"python_rng", "numpy_rng", "torch_rng", "cuda_rng"},
            "Incomplete original rollout RNG",
        )
        nr = payload["numpy_rng"]
        actual = state_hash(
            dict(
                python=payload["python_rng"],
                numpy=(nr[0], np.asarray(nr[1], dtype=np.uint32), *nr[2:]),
                cpu=payload["torch_rng"],
                cuda=payload["cuda_rng"],
            )
        )
        if expected is not None:
            _require(actual == expected, "Original rollout RNG content differs")
        snapshots.append(payload)

    rng(rollout["rng_before_reference"], rollout["rng_before_sha256"])
    rng(rollout["rng_after_reference"], rollout["rng_after_sha256"])
    originals = []
    refs = rollout["complete_group_references"]
    _require(
        len(refs) == 16 and len(rollout["records"]) == 128, "Original G8/128 structure differs"
    )
    protected = (
        "tokens",
        "raw_tokens",
        "sampler_logprobs",
        "native_prompt_inputs",
        "prompt_tensor_hash",
        "full_text",
        "finish_reason",
        "model_input",
        "prompt",
        "seed",
        "sampling_hash",
        "policy_hash",
        "policy_tensor_hash",
        "task_id",
        "source_sha256",
        "data_hash",
        "backend_sequence_tokens",
        "backend_padding_count",
        "protocol",
        "group_seed",
    )
    for index, ref in enumerate(refs):
        group = _bound(tx.directory, ref)
        raw = [
            _json(line)
            for line in _bound(tx.directory, group["raw_reference"], json_value=False).splitlines()
        ]
        _require(
            group["status"] == "IMMUTABLE_COMPLETE_G8" and group["rows"] == len(raw) == 8,
            "Original group is incomplete",
        )
        for draw, row in enumerate(raw):
            annotated = rollout["records"][index * 8 + draw]
            _require(
                all(row.get(key) == annotated.get(key) for key in protected),
                "Original raw trajectory differs from scored/committed row",
            )
        rng(group["rng_before_reference"])
        rng(group["rng_after_reference"], group["rng_after_sha256"])
        originals.extend(_row(r) for r in raw)
    return {"original_raw": originals, "rollout_rng": snapshots}


def _prefix_files(tx, markers, *, synthetic=False):
    """Exact run-relative closure; model input image references are never followed."""
    files = {"RUN_IDENTITY.json", "ROLLOUT_REGISTRATION.json"}
    for marker in markers[:5]:
        step = marker["step"]
        _state(tx, marker, gpu=not synthetic)
        files.update(
            f"checkpoints/step-{step:03d}/{name}"
            for name in (
                ("COMMIT.json", "state.pt", "UPDATE.json") if step else ("COMMIT.json", "state.pt")
            )
        )
        if not step:
            continue
        files.update(ref["path"] for ref in marker["references"].values())
        _originals(tx, marker, synthetic=synthetic)
        rollout = _bound(tx.directory, marker["references"]["rollout"])
        if synthetic and "complete_group_references" not in rollout:
            continue
        stepdir = f"rollouts/step-{step:03d}"
        _require(
            marker["references"]["rollout"]["path"] == stepdir + "/rollout.json",
            "Production original rollout path differs",
        )
        files.update((stepdir + "/STARTED.json", stepdir + "/READY.json"))
        for key in ("rng_before_reference", "rng_after_reference"):
            _bound(tx.directory, rollout[key], json_value=False)
            files.add(rollout[key]["path"])
        refs = rollout["complete_group_references"]
        _require(len(refs) == 16, "Sixteen original complete G8 required")
        for index, ref in enumerate(refs):
            group = _bound(tx.directory, ref)
            groupdir = f"{stepdir}/groups/{index:02d}"
            _require(
                ref["path"] == groupdir + "/COMPLETE.json" and group["rows"] == 8,
                "Original complete group path/size differs",
            )
            files.update((ref["path"], groupdir + "/STARTED.json"))
            for key in ("raw_reference", "rng_before_reference", "rng_after_reference"):
                _bound(tx.directory, group[key], json_value=False)
                files.add(group[key]["path"])
    files.update(_audit_prefixes(tx, markers))
    return sorted(files)


def prepare_recovery(
    primary_tx,
    *,
    experiment_root,
    phase_context,
    pilot_spec_sha256,
    source_commit_sha256,
    primary_execution_ref,
):
    """Import verified 0..4 only, after the unique primary actually completed 0..12."""
    primary_tx._locked()
    root = _plain_path(experiment_root)
    _, primary, identity = _registration(root, phase_context, pilot_spec_sha256)
    _require(
        primary_tx.directory == primary and primary_tx.identity == identity, "Wrong primary run"
    )
    markers = primary_tx.published()
    _require(
        len(markers) == 13 and markers[4]["commit_sha256"] == source_commit_sha256,
        "Actual complete primary and pinned source step4 required",
    )
    execution = _execution(root, primary_execution_ref, primary, identity, require_continuous=True)
    scope = execution["end"]["scope"]
    files = _prefix_files(primary_tx, markers, synthetic=scope == "SYNTHETIC_ONLY")
    lengths = _audit_prefixes(primary_tx, markers)
    inventory = []
    for name in files:
        item = dict(path=name)
        if name in lengths:
            item["source_byte_count"] = lengths[name]
        item["sha256"] = hashlib.sha256(_inventory_bytes(primary, item)).hexdigest()
        inventory.append(item)
    binding = dict(
        format=FORMAT,
        status="IMPORTED_PREFIX_NOT_AUTHORIZED",
        scope=scope,
        storage_lane="RECOVERY",
        directory=DIRECTORY,
        logical_identity=identity,
        primary_directory=primary.relative_to(root).as_posix(),
        phase_identity_sha256=phase_context["identity_sha256"],
        pilot_spec_sha256=pilot_spec_sha256,
        primary_execution_ref=copy.deepcopy(primary_execution_ref),
        source_step=4,
        target_step=12,
        source_commit_sha256=source_commit_sha256,
        source_commit_reference=_ref(root, primary / "checkpoints/step-004/COMMIT.json"),
        prefix_commits=[m["commit_sha256"] for m in markers[:5]],
        prefix_files=inventory,
        source_sampler_state=markers[4]["sampler_state"],
        source_rng_sha256=markers[4]["rng_sha256"],
        source_policy_sha256=markers[4]["policy_sha256"],
        imported_actual_calls=512,
        additional_new_calls=1024,
        gate_status="NOT_SIGNED",
        training_authorized=False,
    )
    _require(
        markers[4]["sampler_state"]["actual_calls"] == 512, "Imported calls must be exactly512"
    )
    destination = root / DIRECTORY
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        _plain_path(destination.parent / ".recovery-import.lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            _require(
                _json(_read_bytes(destination / "RECOVERY_BINDING.json")) == binding,
                "Existing recovery binding is immutable",
            )
        else:
            staging = destination.with_name(".recovery-staging-" + uuid.uuid4().hex)
            staging.mkdir()
            for item in inventory:
                data = _inventory_bytes(primary, item)
                _require(
                    hashlib.sha256(data).hexdigest() == item["sha256"],
                    "Source changed during import",
                )
                path = _path(staging, item["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            _write_json(staging / "RECOVERY_BINDING.json", binding)
            for directory, _, _ in os.walk(staging, topdown=False):
                _fsync_dir(directory)
            _require(not destination.exists(), "Never replace recovery lane")
            os.rename(staging, destination)
            _fsync_dir(destination.parent)
    finally:
        os.close(fd)
    binding_ref = _ref(root, destination / "RECOVERY_BINDING.json")
    return verify_recovery(
        root,
        phase_context=phase_context,
        pilot_spec_sha256=pilot_spec_sha256,
        binding_ref=binding_ref,
    )


def verify_recovery(root, *, phase_context, pilot_spec_sha256, binding_ref, verify_payloads=True):
    """Validate fixed storage binding; payload verification is needed once before cached access."""
    root = _plain_path(root)
    _require(
        binding_ref.get("path") == BINDING, "Only the unique registered recovery binding is allowed"
    )
    binding = _bound(root, binding_ref)
    _, primary, identity = _registration(root, phase_context, pilot_spec_sha256)
    destination = _path(root, DIRECTORY)
    _require(
        binding.get("format") == FORMAT
        and binding.get("directory") == DIRECTORY
        and binding.get("storage_lane") == "RECOVERY"
        and binding.get("logical_identity") == identity
        and binding.get("primary_directory") == primary.relative_to(root).as_posix()
        and binding.get("phase_identity_sha256") == phase_context["identity_sha256"]
        and binding.get("pilot_spec_sha256") == pilot_spec_sha256
        and binding.get("source_step") == 4
        and binding.get("target_step") == 12
        and binding.get("imported_actual_calls") == 512
        and binding.get("additional_new_calls") == 1024,
        "Recovery binding is outside the frozen contract",
    )
    _require(
        _json(_read_bytes(destination / "RUN_IDENTITY.json")) == identity,
        "Imported identity changed",
    )
    source = _bound(root, binding["source_commit_reference"])
    _require(
        binding["source_commit_reference"]["path"]
        == (primary / "checkpoints/step-004/COMMIT.json").relative_to(root).as_posix()
        and source["commit_sha256"] == binding["source_commit_sha256"]
        and source["sampler_state"] == binding["source_sampler_state"]
        and source["rng_sha256"] == binding["source_rng_sha256"]
        and source["policy_sha256"] == binding["source_policy_sha256"],
        "Source checkpoint binding differs",
    )
    chain = _markers(destination, identity)
    _require(
        len(chain) >= 5 and [m["commit_sha256"] for m in chain[:5]] == binding["prefix_commits"],
        "Imported original commit prefix differs",
    )
    # The unchanged primary retains its independently bound terminal proof.
    execution = _execution(
        root, binding["primary_execution_ref"], primary, identity, require_continuous=True
    )
    _require(
        binding["scope"] == execution["end"]["scope"],
        "Recovery scope cannot upgrade synthetic evidence",
    )
    stamps = _prefix_stamps(primary, destination, binding["prefix_files"])
    cache_key = (str(root), binding_ref["sha256"])
    previous_stamps = _PREFIX_CACHE.get(cache_key)
    if verify_payloads and previous_stamps != stamps:
        if previous_stamps is None:
            original = RunTransactions(primary, identity)
            expected = _prefix_files(
                original,
                _markers(primary, identity),
                synthetic=binding["scope"] == "SYNTHETIC_ONLY",
            )
            lengths = _audit_prefixes(original, _markers(primary, identity))
            _require(
                [v["path"] for v in binding["prefix_files"]] == expected,
                "Incomplete/unexpected imported reference closure",
            )
            _require(
                {
                    v["path"]: v["source_byte_count"]
                    for v in binding["prefix_files"]
                    if "source_byte_count" in v
                }
                == lengths,
                "Original audit prefix boundary differs",
            )
        old = dict(previous_stamps or ())
        for item in binding["prefix_files"]:
            for directory in (primary, destination):
                path = _path(directory, item["path"])
                if old.get(str(path)) != _stamp(path):
                    actual = (
                        hashlib.sha256(_inventory_bytes(directory, item)).hexdigest()
                        if "source_byte_count" in item
                        else _file_hash(path)
                    )
                    _require(actual == item["sha256"], "Imported prefix content differs")
        _require(
            _prefix_stamps(primary, destination, binding["prefix_files"]) == stamps,
            "Prefix changed during verification",
        )
        _PREFIX_CACHE[cache_key] = stamps
    return dict(
        directory=str(destination),
        identity=identity,
        binding_ref=copy.deepcopy(binding_ref),
        binding=binding,
        source_step=4,
        target_step=12,
        imported_actual_calls=512,
        additional_new_calls=1024,
        scope=binding["scope"],
        status="VERIFIED_IMPORT_NOT_AUTHORIZED",
        gate_status="NOT_SIGNED",
        training_authorized=False,
    )


def _differences(a, b, path="", result=None):
    """Exact typed tree comparison, including actual tensor dtype and shape."""
    import torch

    result = [] if result is None else result
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        same = (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and bool(torch.equal(a, b))
        )
        if not same:
            result.append(path)
    elif type(a) is not type(b):
        result.append(path + " (type)")
    elif isinstance(a, dict):
        if set(a) != set(b):
            result.append(path + " (keys)")
        for key in sorted(a.keys() & b.keys(), key=lambda item: (type(item).__name__, str(item))):
            _differences(a[key], b[key], f"{path}/{key}", result)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            result.append(path + " (length)")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _differences(x, y, f"{path}/{i}", result)
    elif a != b:
        result.append(path)
    return result


def _comparison_facts(
    primary_tx,
    recovery_tx,
    *,
    root,
    phase_context,
    pilot_spec_sha256,
    binding_ref,
    primary_execution_ref,
    recovery_execution_ref,
):
    """Save exact delivery facts. Numeric mismatches are FAIL; invalid evidence raises."""
    primary_tx._locked()
    recovery_tx._locked()
    root = _plain_path(root)
    bound = verify_recovery(
        root,
        phase_context=phase_context,
        pilot_spec_sha256=pilot_spec_sha256,
        binding_ref=binding_ref,
    )
    _require(
        primary_tx.directory == root / bound["binding"]["primary_directory"]
        and recovery_tx.directory == Path(bound["directory"])
        and primary_tx.identity == recovery_tx.identity == bound["identity"],
        "Comparison runs differ",
    )
    _require(
        primary_execution_ref == bound["binding"]["primary_execution_ref"],
        "Primary execution differs",
    )
    primary = _execution(
        root,
        primary_execution_ref,
        primary_tx.directory,
        primary_tx.identity,
        require_continuous=True,
    )
    recovered = _execution(
        root, recovery_execution_ref, recovery_tx.directory, recovery_tx.identity
    )
    _require(
        recovered["begin"]["storage_lane"] == "RECOVERY"
        and recovered["begin"]["start_step"] == 4
        and recovered["begin"]["recovery_binding_ref"] == binding_ref
        and recovered["new_model_calls"] == 1024
        and all(
            r["new_model_calls"] == 128 and r["reused_complete_groups"] == 0
            for r in recovered["commits"]
        )
        and recovered["end"]["scope"] == primary["end"]["scope"] == bound["scope"],
        "Independent registered eight-step/1024-new-call execution required",
    )
    independent_process = (
        primary["begin"]["process_identity"] != recovered["begin"]["process_identity"]
    )
    _require(
        bound["scope"] != "SERVER_ACTUAL" or independent_process,
        "Registered recovery must execute in a new independent OS process",
    )
    # No unclosed or additional data attempt may disappear from recovery evidence.
    for folder in (recovery_tx.directory / "EXECUTIONS").iterdir():
        if folder.name != recovered["end"]["task_id"]:
            _require((folder / "END.json").is_file(), "Unclosed independent recovery attempt")
            _require(
                not any(
                    e.get("kind") == "completion_attempts" and e.get("count", 0)
                    for e in _events(
                        recovery_tx.directory / "accounting" / (folder.name + ".jsonl")
                    )
                ),
                "Extra recovery data calls exceed registered budget",
            )
    for ledger in (recovery_tx.directory / "accounting").glob("*.jsonl"):
        if ledger.name != recovered["end"]["task_id"] + ".jsonl":
            _require(
                not any(
                    e.get("kind") == "completion_attempts" and e.get("count", 0)
                    for e in _events(ledger)
                ),
                "Extra/orphan recovery data calls exceed budget",
            )
    left, right = primary_tx.published(), recovery_tx.published()
    _require(len(left) == len(right) == 13, "Both actual endpoints must be step12")
    differences = []
    for key in ENVIRONMENT_KEYS:
        _differences(
            primary["runtime"].get(key),
            recovered["runtime"].get(key),
            f"environment/{key}",
            differences,
        )
    for key in HARDWARE_KEYS:
        _differences(
            primary["runtime"].get("hardware", {}).get(key),
            recovered["runtime"].get("hardware", {}).get(key),
            f"hardware/{key}",
            differences,
        )
    step_facts = []
    for step in range(4, 13):
        a, b = (
            _state(primary_tx, left[step], gpu=bound["scope"] == "SERVER_ACTUAL"),
            _state(recovery_tx, right[step], gpu=bound["scope"] == "SERVER_ACTUAL"),
        )
        start = len(differences)
        for key in (
            "policy",
            "optimizer",
            "scheduler",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
        ):
            _differences(a[key], b[key], f"step{step}/state/{key}", differences)
        _differences(
            left[step]["parameter_names"],
            right[step]["parameter_names"],
            f"step{step}/parameter_order",
            differences,
        )
        _differences(
            _without(a["sampler_state"], ("rollout_rng_reference",)),
            _without(b["sampler_state"], ("rollout_rng_reference",)),
            f"step{step}/sampler",
            differences,
        )
        if step > 4:
            _differences(
                _originals(primary_tx, left[step], synthetic=bound["scope"] == "SYNTHETIC_ONLY"),
                _originals(recovery_tx, right[step], synthetic=bound["scope"] == "SYNTHETIC_ONLY"),
                f"step{step}/original_raw_rng",
                differences,
            )
            ar = _bound(primary_tx.directory, left[step]["references"]["rollout"])
            br = _bound(recovery_tx.directory, right[step]["references"]["rollout"])
            _differences(
                [_row(r) for r in ar["records"]],
                [_row(r) for r in br["records"]],
                f"step{step}/original_rows",
                differences,
            )
            for key in (
                "identity",
                "step",
                "policy_before_sha256",
                "rng_before_sha256",
                "rng_after_sha256",
            ):
                _differences(ar.get(key), br.get(key), f"step{step}/rollout/{key}", differences)
            for kind, exclusions in (("credit", ()), ("evidence", EVIDENCE_EXECUTION_KEYS)):
                av = _bound(primary_tx.directory, left[step]["references"][kind])
                bv = _bound(recovery_tx.directory, right[step]["references"][kind])
                if kind == "evidence":
                    _audit_event(
                        primary_tx.directory, av, required=bound["scope"] == "SERVER_ACTUAL"
                    )
                    _audit_event(
                        recovery_tx.directory, bv, required=bound["scope"] == "SERVER_ACTUAL"
                    )
                _differences(
                    _without(av, exclusions),
                    _without(bv, exclusions),
                    f"step{step}/{kind}",
                    differences,
                )
            au = _json(_read_bytes(primary_tx.checkpoints / f"step-{step:03d}/UPDATE.json"))
            bu = _json(_read_bytes(recovery_tx.checkpoints / f"step-{step:03d}/UPDATE.json"))
            _differences(
                _without(au, UPDATE_EXECUTION_KEYS),
                _without(bu, UPDATE_EXECUTION_KEYS),
                f"step{step}/numerical_gradient",
                differences,
            )
        step_facts.append(
            dict(
                step=step,
                exact=len(differences) == start,
                primary_commit=left[step]["commit_sha256"],
                recovery_commit=right[step]["commit_sha256"],
            )
        )
    result = dict(
        format=FORMAT,
        status="FAIL"
        if differences
        else ("PASS" if bound["scope"] == "SERVER_ACTUAL" else "SYNTHETIC_ONLY"),
        scope=bound["scope"],
        exact=not differences,
        binding_ref=binding_ref,
        primary_execution_ref=primary_execution_ref,
        recovery_execution_ref=recovery_execution_ref,
        new_generated_responses=1024,
        independent_process_verified=independent_process and bound["scope"] == "SERVER_ACTUAL",
        process_pair={
            "primary": primary["begin"]["process_identity"],
            "recovery": recovered["begin"]["process_identity"],
        },
        hardware_pair={
            "primary": primary["runtime"].get("hardware"),
            "recovery": recovered["runtime"].get("hardware"),
        },
        compared_hardware_fields=list(HARDWARE_KEYS),
        compared_environment_fields=list(ENVIRONMENT_KEYS),
        differences=differences,
        steps=step_facts,
        gate_status="NOT_SIGNED",
        training_authorized=False,
    )
    return result


def verify_comparison(primary_tx, recovery_tx, *, comparison_ref, **kwargs):
    """Recompute all saved comparison facts under both locks; never write a receipt."""
    primary_tx._locked()
    recovery_tx._locked()
    root = _plain_path(kwargs["root"])
    path = _path(root, comparison_ref["path"])
    _require(
        path.parent == recovery_tx.directory / "COMPARISONS"
        and re.fullmatch(r"[0-9a-f]{32}\.json", path.name),
        "Fixed comparison receipt path required",
    )
    saved = _bound(root, comparison_ref)
    actual = _comparison_facts(primary_tx, recovery_tx, **kwargs)
    _require(saved == actual, "Saved comparison differs from recomputed facts")
    return {
        **actual,
        "receipt_reference": copy.deepcopy(comparison_ref),
        "verified_from_saved_facts": True,
    }


def compare_recovery(primary_tx, recovery_tx, **kwargs):
    """Retain every comparison outcome, including malformed or incomplete evidence."""
    primary_tx._locked()
    recovery_tx._locked()
    try:
        result = _comparison_facts(primary_tx, recovery_tx, **kwargs)
        root = _plain_path(kwargs["root"])
        folder = recovery_tx.directory / "COMPARISONS"
        folder.mkdir(exist_ok=True)
        result["receipt_reference"] = _save(root, folder / (uuid.uuid4().hex + ".json"), result)
        return result
    except Exception as exc:
        root = _plain_path(kwargs["root"])
        _require(
            recovery_tx.directory == root / DIRECTORY, "Failure output must stay in recovery lane"
        )
        folder = recovery_tx.directory / "COMPARISONS"
        folder.mkdir(exist_ok=True)
        _save(
            root,
            folder / (uuid.uuid4().hex + "-FAILURE.json"),
            dict(
                format=FORMAT,
                status="INVALID_OR_INCOMPLETE_EVIDENCE",
                error_type=type(exc).__name__,
                error=str(exc),
                binding_ref=kwargs.get("binding_ref"),
                primary_execution_ref=kwargs.get("primary_execution_ref"),
                recovery_execution_ref=kwargs.get("recovery_execution_ref"),
                gate_status="NOT_SIGNED",
                training_authorized=False,
            ),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase-context", type=Path, required=True)
    parser.add_argument("--pilot-spec-sha256", required=True)
    parser.add_argument("--binding-sha256")
    parser.add_argument("--source-commit-sha256")
    parser.add_argument("--primary-execution", type=Path)
    parser.add_argument("--recovery-execution", type=Path)
    args = parser.parse_args(argv)
    root = _plain_path(args.root)
    context = _json(_read_bytes(args.phase_context))
    common = dict(phase_context=context, pilot_spec_sha256=args.pilot_spec_sha256)
    _, primary, identity = _registration(root, **common)
    binding_ref = dict(path=BINDING, sha256=args.binding_sha256)
    if args.command == "verify":
        result = verify_recovery(root, binding_ref=binding_ref, **common)
    else:
        _require(args.primary_execution is not None, "Primary execution receipt required")
        pref = _ref(root, _plain_path(args.primary_execution))
        with RunTransactions(primary, identity) as tx:
            if args.command == "prepare":
                result = prepare_recovery(
                    tx,
                    experiment_root=root,
                    source_commit_sha256=args.source_commit_sha256,
                    primary_execution_ref=pref,
                    **common,
                )
            else:
                _require(args.recovery_execution is not None, "Recovery execution receipt required")
                with RunTransactions(root / DIRECTORY, identity) as rtx:
                    result = compare_recovery(
                        tx,
                        rtx,
                        root=root,
                        binding_ref=binding_ref,
                        primary_execution_ref=pref,
                        recovery_execution_ref=_ref(root, _plain_path(args.recovery_execution)),
                        **common,
                    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 1 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
