"""Server identity checks and a process-owned resource watchdog.

Fresh identity verification is read-only and never constructs a model. Runtime
reconstruction is a separate, explicitly called preflight operation; the server
CLI owns its Slurm-only authorization. Neither operation reads oracle panels.
The watchdog writes only to a separate control directory, not a sealed stage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import resource
import signal
import stat as stat_module
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .protocol import ROOT

FLOAT_ATOL = 1e-14
FLOAT_RTOL = 1e-12
DATASET_SEED = 20260915
INITIALIZATION_SEED = 7001
ARRAY_FIELDS = tuple(
    f"{split}_{kind}" for split in ("train", "probe") for kind in ("features", "categories")
)


def _json(path):
    return json.loads(Path(path).read_text())


def _receipt(kind, parent_root):
    return {
        "schema": "server-semantic-identity-v1",
        "kind": kind,
        "parent_root": str(Path(parent_root).resolve()),
        "checks": [],
        "reasons": [],
        "float_atol": FLOAT_ATOL,
        "float_rtol": FLOAT_RTOL,
        "comparison": "semantic values; archive bytes are not an equality criterion",
        "new_training_steps": 0,
        "oracle_panels_read": False,
    }


def _finish(receipt):
    receipt["passed"] = not receipt["reasons"]
    receipt["status"] = "PASS" if receipt["passed"] else "IDENTITY_MISMATCH"
    return receipt


def _record(receipt, name, passed, **details):
    receipt["checks"].append({"name": name, "passed": bool(passed), **details})
    if not passed:
        receipt["reasons"].append(name)


def _compare_array(receipt, name, reference, observed, *, floating):
    left, right = np.asarray(reference), np.asarray(observed)
    details = {
        "reference_shape": list(left.shape),
        "observed_shape": list(right.shape),
        "reference_dtype": str(left.dtype),
        "observed_dtype": str(right.dtype),
    }
    if not floating:
        passed = (
            left.dtype.kind in "iu"
            and right.dtype.kind in "iu"
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
        _record(receipt, name, passed, comparison="exact integer values", **details)
        return
    valid = (
        left.dtype.kind == right.dtype.kind == "f"
        and left.dtype.itemsize == right.dtype.itemsize == 8
        and left.shape == right.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all()
    )
    error = float(np.max(np.abs(left - right), initial=0)) if valid else None
    passed = valid and np.allclose(left, right, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
    _record(
        receipt,
        name,
        passed,
        comparison="finite float64 allclose",
        max_abs_error=error,
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
        **details,
    )


def _compare_json(receipt, name, reference, observed):
    mismatches, errors = [], []

    def visit(left, right, path):
        if type(left) is not type(right):
            mismatches.append(path)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                mismatches.append(path + ".keys")
            for key in left.keys() & right.keys():
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                mismatches.append(path + ".length")
            for index, (a, b) in enumerate(zip(left, right, strict=False)):
                visit(a, b, f"{path}[{index}]")
        elif isinstance(left, float):
            if not (math.isfinite(left) and math.isfinite(right)):
                mismatches.append(path)
            else:
                errors.append(abs(left - right))
                if not np.isclose(left, right, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
                    mismatches.append(path)
        elif left != right:
            mismatches.append(path)

    visit(reference, observed, name)
    _record(
        receipt,
        name,
        not mismatches,
        comparison="ordered semantic JSON; exact strings/integers",
        mismatched_paths=sorted(mismatches)[:20],
        mismatch_count=len(mismatches),
        max_abs_float_error=max(errors, default=0.0),
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
    )


def _read_static(root):
    root = Path(root).resolve()
    with np.load(root / "dataset.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_FIELDS):
            raise ValueError("dataset array fields differ from the fixed schema")
        arrays = {key: archive[key] for key in ARRAY_FIELDS}
    metadata = {split: _json(root / f"{split}_metadata.json") for split in ("train", "probe")}
    scenes = {
        split: [
            json.loads(line)
            for line in (root / "generated_dataset" / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "control")
    }
    files = [
        "resolved_config.json",
        "manifest.json",
        "dataset.npz",
        "parameter_layout.json",
        "train_metadata.json",
        "probe_metadata.json",
        "generated_dataset/train.jsonl",
        "generated_dataset/control.jsonl",
    ]
    return {
        "arrays": arrays,
        "metadata": metadata,
        "scenes": scenes,
        "layout": _json(root / "parameter_layout.json"),
        "config": _json(root / "resolved_config.json"),
        "manifest": _json(root / "manifest.json"),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
    }


def _fixed_seeds(receipt, source, config):
    for section, key, expected in (
        ("dataset", "seed", DATASET_SEED),
        ("toy_model", "model_initialization_seed", INITIALIZATION_SEED),
    ):
        actual = config[section][key]
        _record(
            receipt,
            f"{source}.{section}.{key}",
            type(actual) is int and actual == expected,
            expected=expected,
            observed=actual,
        )


def _read_initialization(root, entry):
    root = Path(root).resolve()
    path = (root / entry["observations_file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("observation path escapes collection root")
    with np.load(path, allow_pickle=False) as archive:
        theta = archive["theta"]
    if theta.ndim != 2 or theta.shape[0] < 1 or theta.shape[1] != 737:
        raise ValueError("initialization must be theta[0] in a trajectory with 737 parameters")
    return theta[0].copy()


def _read_parent(receipt, root):
    parent = _read_static(root)
    receipt["parent_source_sha256"] = parent["source_sha256"]
    _fixed_seeds(receipt, "parent", parent["config"])
    for key, expected in (
        ("events", ["X", "S", "W", "I"]),
        ("parameter_count", 737),
        ("probe_count", 72),
    ):
        _compare_json(receipt, f"parent.manifest.{key}", expected, parent["manifest"][key])
    entries = parent["manifest"]["trajectories"]
    if not entries:
        raise ValueError("parent has no initialization reference")
    reference = _read_initialization(root, entries[0])
    receipt["parent_reference_trajectory"] = entries[0]["id"]
    receipt["reference_initialization_sha256"] = hashlib.sha256(reference.tobytes()).hexdigest()
    receipt["parent_initializations_checked"] = 0
    for entry in entries:
        _compare_array(
            receipt,
            f"parent.initialization.{entry['id']}",
            reference,
            _read_initialization(root, entry),
            floating=True,
        )
        receipt["parent_initializations_checked"] += 1
    parent["theta"] = reference
    return parent


def _compare_static(receipt, parent, observed):
    for key in ARRAY_FIELDS:
        _compare_array(
            receipt,
            f"dataset.{key}",
            parent["arrays"][key],
            observed["arrays"][key],
            floating=key.endswith("features"),
        )
    for split in ("train", "probe"):
        _compare_json(
            receipt, f"{split}_metadata", parent["metadata"][split], observed["metadata"][split]
        )
    _compare_json(receipt, "parameter_layout", parent["layout"], observed["layout"])
    for split in ("train", "control"):
        _compare_json(
            receipt,
            f"generated_dataset.{split}",
            parent["scenes"][split],
            observed["scenes"][split],
        )


def verify_fresh_identity(parent_root, fresh_raw_root):
    """Read semantic data and all 32 fresh theta[0] arrays, returning a gate receipt."""
    receipt = _receipt("fresh_collection_identity", parent_root)
    receipt["fresh_raw_root"] = str(Path(fresh_raw_root).resolve())
    receipt["fresh_initializations_checked"] = 0
    try:
        parent = _read_parent(receipt, parent_root)
        fresh = _read_static(fresh_raw_root)
        receipt["fresh_source_sha256"] = fresh["source_sha256"]
        _fixed_seeds(receipt, "fresh", fresh["config"])
        _compare_static(receipt, parent, fresh)
        for key in ("events", "parameter_count", "probe_count"):
            _compare_json(
                receipt, f"fresh.manifest.{key}", parent["manifest"][key], fresh["manifest"][key]
            )
        entries = fresh["manifest"]["trajectories"]
        expected = sorted(
            (
                f"seed{seed}_{arm}",
                seed,
                arm,
                "fresh_calibration" if seed < 600 else "fresh_locked_test",
            )
            for seed in [*range(501, 507), *range(601, 611)]
            for arm in ("X_BASE", "X_VALID")
        )
        actual = sorted((row["id"], row["seed"], row["arm"], row["split"]) for row in entries)
        _record(
            receipt,
            "fresh.trajectory_membership",
            actual == expected,
            expected_count=32,
            observed_count=len(actual),
        )
        for entry in entries:
            _compare_array(
                receipt,
                f"fresh.initialization.{entry['id']}",
                parent["theta"],
                _read_initialization(fresh_raw_root, entry),
                floating=True,
            )
            receipt["fresh_initializations_checked"] += 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt["reasons"].append(f"{type(error).__name__}: {error}")
    return _finish(receipt)


def _build_runtime_identity(out, dataset_seed, initialization_seed):
    # Deliberately lazy: merely importing this module cannot initialize a model.
    from src.modeling_qualification.toy import (
        build_toy_dataset,
        flatten_parameters,
        make_model,
        parameter_layout,
    )

    dataset = build_toy_dataset(out, dataset_seed)
    model = make_model(initialization_seed)
    return {
        "arrays": {key: getattr(dataset, key) for key in ARRAY_FIELDS},
        "metadata": {split: getattr(dataset, f"{split}_metadata") for split in ("train", "probe")},
        "scenes": dataset.scenes,
        "layout": parameter_layout(model),
        "theta": flatten_parameters(model),
    }


def verify_parent_runtime_identity(parent_root, *, temporary_root=None):
    """Explicit server preflight: regenerate fixed data and initialization, with no updates.

    Pass the job's monitored project scratch directory as ``temporary_root``.
    The fallback is beside, never inside, the sealed parent collection. All
    temporary generated data is removed before this function returns.
    """
    receipt = _receipt("server_runtime_identity", parent_root)
    try:
        parent = _read_parent(receipt, parent_root)
        if receipt["reasons"]:
            return _finish(receipt)
        scratch = (
            Path(temporary_root)
            if temporary_root is not None
            else Path(parent_root).resolve().parent / "contrast_identity_scratch"
        )
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="runtime_identity_", dir=scratch) as temp:
            runtime = _build_runtime_identity(
                Path(temp) / "generated_dataset", DATASET_SEED, INITIALIZATION_SEED
            )
            _compare_static(receipt, parent, runtime)
            _compare_array(
                receipt, "runtime.initialization", parent["theta"], runtime["theta"], floating=True
            )
        receipt["temporary_generated_data_removed"] = True
        receipt["runtime_dataset_seed"] = DATASET_SEED
        receipt["runtime_initialization_seed"] = INITIALIZATION_SEED
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt["reasons"].append(f"{type(error).__name__}: {error}")
    return _finish(receipt)


def peak_rss_gib(value=None, platform=None):
    """Convert SELF ru_maxrss: bytes on macOS, KiB on Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if value is None else value
    platform = sys.platform if platform is None else platform
    if platform != "darwin" and not platform.startswith("linux"):
        raise ValueError(f"unsupported ru_maxrss platform: {platform}")
    return float(value) / (2**30 if platform == "darwin" else 2**20)


def evaluate_resource_snapshot(snapshot, config):
    """Pure current-resource check; scientific config pinning is the CLI's responsibility."""
    limits = config["resources"]
    bounds = {
        "wall_seconds": float(limits["max_total_walltime_hours"]) * 3600,
        "peak_ram_gib": float(limits["max_ram_gib"]),
        "added_output_bytes": float(limits["max_added_output_gib"]) * 2**30,
        "temporary_bytes": float(limits["max_temporary_gib"]) * 2**30,
    }
    if not all(math.isfinite(value) and value > 0 for value in bounds.values()):
        raise ValueError("resource limits must be finite and positive")
    reasons = []
    for name, cap in bounds.items():
        value = snapshot.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            reasons.append(f"{name}: missing, negative, or nonfinite measurement")
        elif value > cap:
            reasons.append(f"{name}: {value} exceeds {cap}")
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "RESOURCE_REVIEW_REQUIRED",
        "snapshot": dict(snapshot),
        "limits": bounds,
        "reasons": reasons,
    }


def _disk_bytes(roots):
    """Logical file bytes, deduplicated by inode; do not follow nested symlinks."""
    seen, total = set(), 0

    def report_error(error):
        raise error

    for root in {Path(path).resolve() for path in roots}:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root, followlinks=False, onerror=report_error):
            subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
            for filename in files:
                path = Path(directory) / filename
                try:
                    stat = path.lstat()
                except FileNotFoundError:
                    continue  # atomic replacement or scratch removal during a scan
                if not stat_module.S_ISREG(stat.st_mode):
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    total += stat.st_size
    return total


class ResourceLimitExceeded(RuntimeError):
    """Raised if a testing termination seam returns after a fatal receipt."""


class RuntimeResourceWatchdog:
    """Check every ten seconds; flush evidence before terminating only this PID.

    ``terminate`` has the ``os.kill(pid, signal)`` signature. The production
    default is ``os.kill(os.getpid(), SIGTERM)``; it never targets a process
    group, scheduler job, or another process. A test seam may return, in which
    case entry/exit raises ResourceLimitExceeded and the receipt remains saved.
    """

    def __init__(
        self,
        config,
        run_root,
        control_out,
        temporary_root,
        prior_wall_seconds=0,
        *,
        interval_seconds=10,
        terminate=None,
    ):
        self.config = copy.deepcopy(config)
        evaluate_resource_snapshot({}, self.config)  # validate limits without I/O
        self.run_root = Path(run_root).resolve()
        self.control_out = Path(control_out).resolve()
        self.temporary_root = Path(temporary_root).resolve()
        if self.control_out.is_relative_to(self.run_root):
            relative = self.control_out.relative_to(self.run_root)
            if relative.parts and re.fullmatch(r"N\d+[A-Z]?(?:_.*)?", relative.parts[0]):
                raise ValueError("watchdog control output must be outside an experiment stage")
        for ancestor in (self.control_out, *self.control_out.parents):
            if ancestor == self.run_root.parent:
                break
            if any(
                (ancestor / name).is_file() for name in ("RUN_MANIFEST.json", "stage_result.json")
            ):
                raise ValueError("watchdog control output must be outside a frozen stage")
        self.prior_wall_seconds = float(prior_wall_seconds)
        self.interval_seconds = float(interval_seconds)
        if not math.isfinite(self.prior_wall_seconds) or self.prior_wall_seconds < 0:
            raise ValueError("prior wall seconds must be finite and nonnegative")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("watchdog interval must be finite and positive")
        self._terminate = os.kill if terminate is None else terminate
        self._pid = os.getpid()
        self._start = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self.last_receipt = None
        self.violation_receipt = None

    def _snapshot(self):
        return {
            "wall_seconds": self.prior_wall_seconds + time.monotonic() - self._start,
            "peak_ram_gib": peak_rss_gib(),
            "added_output_bytes": _disk_bytes(
                [self.run_root, ROOT / "docs/modeling_contrast/results"]
            ),
            "temporary_bytes": _disk_bytes([self.temporary_root]),
        }

    def _persist(self, receipt, *, fatal):
        # Invalid measured values are recorded as null, keeping strict JSON usable.
        def safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: safe(item) for key, item in value.items()}
            return value

        data = json.dumps(safe(receipt), sort_keys=True, allow_nan=False) + "\n"
        with (self.control_out / "resource_watchdog.jsonl").open("a") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if fatal:
            with (self.control_out / "resource_violation.json").open("w") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

    def check_now(self):
        with self._lock:
            if self.violation_receipt is not None:
                return self.violation_receipt
            if self._start is None:
                self._start = time.monotonic()
            self.control_out.mkdir(parents=True, exist_ok=True)
            try:
                receipt = evaluate_resource_snapshot(self._snapshot(), self.config)
            except Exception as error:
                receipt = {
                    "passed": False,
                    "status": "RESOURCE_REVIEW_REQUIRED",
                    "snapshot": {},
                    "reasons": [f"resource measurement failed: {type(error).__name__}: {error}"],
                }
            receipt.update(
                {
                    "schema": "runtime-resource-watchdog-v1",
                    "pid": self._pid,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "prior_wall_seconds": self.prior_wall_seconds,
                    "disk_basis": (
                        "regular file logical bytes; inode deduplication; no nested symlinks"
                    ),
                    "rss_basis": f"RUSAGE_SELF peak; platform={sys.platform}",
                    "sample_interval_seconds": self.interval_seconds,
                }
            )
            fatal = not receipt["passed"]
            try:
                self._persist(receipt, fatal=fatal)
            except (OSError, ValueError, TypeError) as error:
                # Disk exhaustion can prevent a control-file write. Preserve a
                # final receipt in the job's stderr stream and still stop work.
                fatal = True
                receipt["passed"] = False
                receipt["status"] = "RESOURCE_REVIEW_REQUIRED"
                receipt["reasons"].append(f"watchdog receipt persistence failed: {error}")
                print(json.dumps(receipt, sort_keys=True), file=sys.stderr, flush=True)
            self.last_receipt = receipt
            if fatal:
                self.violation_receipt = receipt
                self._stop.set()
                self._terminate(os.getpid(), signal.SIGTERM)
            return receipt

    def _monitor(self):
        while not self._stop.wait(self.interval_seconds):
            self.check_now()

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("watchdog contexts cannot be reused")
        if not self.check_now()["passed"]:
            raise ResourceLimitExceeded("resource limit exceeded; see resource_violation.json")
        self._thread = threading.Thread(target=self._monitor, name="resource-watchdog", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join(timeout=30)
        if self.violation_receipt is None:
            self.check_now()
        if self.violation_receipt is not None and exc_type is None:
            raise ResourceLimitExceeded("resource limit exceeded; see resource_violation.json")
        return False
