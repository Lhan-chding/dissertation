"""Read-only, bounded parent metadata audit; never loads model/checkpoint tensors.

The three pinned JSON files establish the historical metadata anchor. Their
presence cannot establish raw rollout, checkpoint, image, or GPU readiness.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .core import canonical_hash

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ORIGIN_HASH = "5f809aa3e46cf19feb0444b5ab932df765db9ef2bdb80fe8a4ced300174730e9"
PINNED_SHA256 = {
    "runtime_lock.json": "14ad5ec1bd7596b93720404e77f6ed220fa0123963c7a026438fcfdab81a545c",
    "bank_manifest.json": "dd4e543c6c8ae5076ce20ad46df7e51afd4cad7750f33b028e46fbf61536a416",
    "candidate_manifest.json": "01735c725d06f99a74c3c18d45f380153c73f9944a9658679a8828e2e47d0d12",
}
SELECTED_BANKS = (0, 3, 4, 5, 11)
CATEGORIES = ("X", "S", "W", "I")
EXPECTED_COUNTS = {
    0: ((8, 0, 0, 0), (8, 0, 0, 0), (4, 0, 4, 0), (7, 0, 1, 0)),
    3: ((8, 0, 0, 0), (8, 0, 0, 0), (0, 0, 4, 4), (8, 0, 0, 0)),
    4: ((0, 0, 7, 1), (8, 0, 0, 0), (8, 0, 0, 0), (8, 0, 0, 0)),
    5: ((0, 0, 8, 0), (8, 0, 0, 0), (0, 0, 7, 1), (8, 0, 0, 0)),
    11: ((0, 0, 8, 0), (8, 0, 0, 0), (4, 0, 1, 3), (8, 0, 0, 0)),
}


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a SHA-256 hash")
    return value


def _relative(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("Unsafe parent relative path")
    return name


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


class RestrictedParentReader:
    """JSON-only directory/ZIP reader. Archives are inspected, never extracted.

    Limits cover all archive entries before decompression; a selected JSON read
    is bounded again. Directory paths cannot follow symlinks, including ancestors.
    """

    def __init__(
        self,
        parent_path,
        *,
        max_file_bytes=64 * 1024**2,
        max_total_bytes=1024 * 1024**2,
        max_entries=20000,
        max_compression_ratio=1000,
    ):
        for value in (max_file_bytes, max_total_bytes, max_entries, max_compression_ratio):
            if type(value) is not int or value <= 0:
                raise ValueError("Parent reader limits must be positive integers")
        self.path = Path(parent_path).absolute()
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self.max_entries = max_entries
        self.archive = None
        self.entries = {}
        if self.path.is_symlink():
            raise ValueError("Parent root symlink is not allowed")
        if self.path.is_dir():
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"Parent metadata path does not exist: {self.path}")
        if self.path.suffix.lower() != ".zip":
            raise ValueError("Parent must be a directory or restricted ZIP")
        archive = zipfile.ZipFile(self.path, "r")
        try:
            entries = archive.infolist()
            if len(entries) > max_entries:
                raise ValueError("Archive entry limit exceeded")
            total = 0
            files, directories = set(), set()
            for entry in entries:
                name = _relative(entry.filename[:-1] if entry.is_dir() else entry.filename)
                if name in self.entries:
                    raise ValueError("Duplicate archive path")
                mode = entry.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Archive symlink or special file is forbidden")
                if (kind == stat.S_IFDIR) != entry.is_dir() and kind != 0:
                    raise ValueError("Archive entry type mismatch")
                if entry.flag_bits & 1:
                    raise ValueError("Encrypted archive entry is forbidden")
                if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("Unsupported archive compression")
                total += entry.file_size
                if total > max_total_bytes:
                    raise ValueError("Archive total size limit exceeded")
                if (
                    not entry.is_dir()
                    and entry.filename.endswith(".json")
                    and entry.file_size > max_file_bytes
                ):
                    raise ValueError("Archive JSON file size limit exceeded")
                if entry.file_size > max_compression_ratio * max(1, entry.compress_size):
                    raise ValueError("Archive compression ratio limit exceeded")
                self.entries[name] = entry
                (directories if entry.is_dir() else files).add(name)
            for name in self.entries:
                if any(
                    str(ancestor) in files
                    for ancestor in PurePosixPath(name).parents
                    if str(ancestor) != "."
                ):
                    raise ValueError("Archive path traverses a regular file")
            self.archive = archive
        except Exception:
            archive.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def _directory_file(self, relative):
        current = self.path
        # Refuse ancestor symlinks too; a root alias is not a second evidence copy.
        for ancestor in [self.path, *self.path.parents]:
            if ancestor.is_symlink():
                raise ValueError("Parent directory symlink is not allowed")
        for part in relative.split("/"):
            current = current / part
            if current.is_symlink():
                raise ValueError("Parent file symlink is not allowed")
        if not current.resolve().is_relative_to(self.path.resolve()):
            raise ValueError("Parent path escapes evidence root")
        return current

    def exists(self, relative):
        relative = _relative(relative)
        if self.archive is not None:
            return relative in self.entries and not self.entries[relative].is_dir()
        return self._directory_file(relative).is_file()

    def read_json(self, relative, *, expected_sha256=None):
        relative = _relative(relative)
        if not relative.endswith(".json"):
            raise ValueError("Restricted parent reader only accepts JSON files")
        if expected_sha256 is not None:
            _sha(expected_sha256)
        if self.archive is not None:
            entry = self.entries.get(relative)
            if entry is None or entry.is_dir():
                raise FileNotFoundError(f"Parent metadata file missing: {relative}")
            if entry.file_size > self.max_file_bytes:
                raise ValueError("JSON file limit exceeded")
            with self.archive.open(entry, "r") as source:
                raw = source.read(self.max_file_bytes + 1)
        else:
            path = self._directory_file(relative)
            # O_NOFOLLOW closes the final-component check/open race.
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as source:
                metadata = os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("Parent metadata must be a regular file")
                if metadata.st_size > self.max_file_bytes:
                    raise ValueError("JSON file limit exceeded")
                raw = source.read(self.max_file_bytes + 1)
        if len(raw) > self.max_file_bytes:
            raise ValueError("JSON file limit exceeded")
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"Parent metadata hash mismatch: {relative}")
        try:
            result = json.loads(raw, object_pairs_hook=_object_pairs, parse_constant=_nonfinite)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("Invalid parent JSON") from error
        if not isinstance(result, dict):
            raise ValueError("Parent JSON requires an object root")
        return result


def _warm_prefix(reader):
    choices = ("", "02_data/R3_warm/", "SSVC_GPT_PRO_DELIVERY_20260914/02_data/R3_warm/")
    found = [p for p in choices if reader.exists(p + "runtime_lock.json")]
    if len(found) != 1:
        raise ValueError("Exactly one warm metadata runtime_lock.json root is required")
    return found[0]


def load_parent_metadata(parent_path):
    """Read the original three metadata files and verify their pinned file hashes."""
    with RestrictedParentReader(parent_path) as reader:
        prefix = _warm_prefix(reader)
        return {
            name[:-5]: reader.read_json(prefix + name, expected_sha256=digest)
            for name, digest in PINNED_SHA256.items()
        }


def _counts(counts):
    if (
        not isinstance(counts, dict)
        or set(counts) != set(CATEGORIES)
        or any(type(v) is not int or v < 0 for v in counts.values())
        or sum(counts.values()) != 8
    ):
        raise ValueError("Parent group composition must contain exactly K8 events")


def validate_parent_metadata(
    runtime_lock, bank_manifest, candidate_manifest, *, required_origin=None
):
    """Cross-bind warm metadata. Accepts explicit CPU fixture identities for tests."""
    try:
        r, b, c = runtime_lock, bank_manifest, candidate_manifest
        identity, plan = r["identity"], b["plan"]
        origin = r["origin_hash"]
        if origin != ORIGIN_HASH or (required_origin is not None and required_origin != origin):
            raise ValueError("Parent origin hash mismatch")
        if (
            identity["phase"] != "R3-warm"
            or type(identity["checkpoint_step"]) is not int
            or identity["checkpoint_step"] != 64
        ):
            raise ValueError("Parent requires warm X_BASE seed17 step64")
        if identity["execution_kind"] not in ("REAL_CUDA_FORK", "CPU_FAKE_ADAPTER_FIXTURE"):
            raise ValueError("Unknown parent execution kind")
        if (
            r["config"]["model"]["id"] != MODEL_ID
            or r["config"]["model"]["revision"] != MODEL_REVISION
            or r["model_audit"]["model_id"] != MODEL_ID
            or r["model_audit"]["model_revision"] != MODEL_REVISION
        ):
            raise ValueError("Parent model revision binding mismatch")
        if (
            any(
                type(r["config"][section][key]) is not int
                for section, key in (("R3", "B"), ("R3", "K"), ("R4", "seed"))
            )
            or r["config"]["R3"]["B"] != 4
            or r["config"]["R3"]["K"] != 8
            or r["config"]["R4"]["seed"] != 17
            or r["config"]["R3"]["warm_checkpoint"] != "X_BASE/step64"
        ):
            raise ValueError("Parent B4/K8/origin arm/seed contract mismatch")
        if identity != b["identity"]:
            raise ValueError("Parent identity mismatch across metadata files")
        if b["request_identity"] != {**identity, "origin_hash": origin}:
            raise ValueError("Parent request identity mismatch")
        if (
            plan["plan_hash"] != canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
            or plan["plan_hash"] != identity["plan_hash"]
        ):
            raise ValueError("Parent plan content hash mismatch")
        if plan["banks_hash"] != canonical_hash(plan["banks"]):
            raise ValueError("Parent banks hash mismatch")
        if canonical_hash(b["data_binding"]) != identity["data_hash"]:
            raise ValueError("Parent data binding hash mismatch")
        train, control = plan["train_prompts"], plan["control_prompts"]
        train_ids, control_ids = [p["prompt_id"] for p in train], [p["prompt_id"] for p in control]
        if (
            len(train_ids) != 48
            or len(control_ids) != 48
            or len(set(train_ids + control_ids)) != 96
            or set(p["base_scene_id"] for p in train) & set(p["base_scene_id"] for p in control)
            or [pid for bank in plan["banks"] for pid in bank] != train_ids
            or len(plan["banks"]) != 12
            or any(len(bank) != 4 for bank in plan["banks"])
        ):
            raise ValueError("Parent duplicate prompt or train/control leakage")
        for rows, split in ((train, "train"), (control, "control")):
            for prompt in rows:
                if (
                    prompt["split"] != split
                    or prompt["scene"]["split"] != split
                    or prompt["scene_hash"] != canonical_hash(prompt["scene"])
                    or prompt["prompt_hash"] != prompt["prompt"]["prompt_hash"]
                ):
                    raise ValueError("Parent prompt/scene binding mismatch")
        for name, expected in (
            ("train_prompt_ids_hash", train_ids),
            ("control_prompt_ids_hash", control_ids),
        ):
            if plan[name] != canonical_hash(expected):
                raise ValueError("Parent prompt order hash mismatch")
        composition = []
        for bank_index in SELECTED_BANKS:
            summary = c["banks"][str(bank_index)]
            if summary["origin_state_hash"] != origin or any(
                summary["identity"].get(key) != value
                for key, value in b["request_identity"].items()
            ):
                raise ValueError("Candidate metadata origin/model/data/parser identity mismatch")
            groups = summary["group_composition"]["prompts"]
            if len(groups) != 4 or [g["prompt_id"] for g in groups] != plan["banks"][bank_index]:
                raise ValueError("Candidate and bank manifest prompt binding mismatch")
            for position, group in enumerate(groups):
                _counts(group["counts"])
                if (
                    tuple(group["counts"][key] for key in CATEGORIES)
                    != EXPECTED_COUNTS[bank_index][position]
                ):
                    raise ValueError("Parent fixed group composition mismatch")
            for candidate in summary["candidates"]:
                if any(
                    candidate["checkpoint_identity"].get(key) != value
                    for key, value in b["request_identity"].items()
                ):
                    raise ValueError("Candidate checkpoint identity mismatch")
                gradient = candidate["gradient"]
                if (
                    gradient["B"] != 4
                    or gradient["K"] != 8
                    or gradient["Lnorm"] != 64
                    or [g["prompt_id"] for g in gradient["group_statistics"]]
                    != plan["banks"][bank_index]
                    or [g["category_counts"] for g in gradient["group_statistics"]]
                    != [g["counts"] for g in groups]
                ):
                    raise ValueError("Candidate gradient groups differ from parent bank")
            composition.append(
                {
                    "bank_index": bank_index,
                    "status": "CHECKED",
                    "groups": [
                        {
                            "group_position_zero_based": i,
                            "prompt_id": g["prompt_id"],
                            "counts": dict(g["counts"]),
                        }
                        for i, g in enumerate(groups)
                    ],
                }
            )
        return composition
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Malformed parent metadata schema") from error


def audit_source_files(runtime_lock, project_root):
    """File SHA-256 comparisons, separate from commits and Git object hashes."""
    root = Path(project_root).resolve()
    result = []
    for name, expected in sorted(runtime_lock["source"]["source_files"].items()):
        _relative(name)
        _sha(expected)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Source comparison path escapes project")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result.append(
            {
                "path": name,
                "recorded_sha256": expected,
                "current_sha256": actual,
                "status": "NOT_AVAILABLE" if actual is None else "CHECKED",
                "comparison": "MISSING"
                if actual is None
                else "MATCH"
                if actual == expected
                else "DIFFERENT",
            }
        )
    return result


def audit_parent_evidence(parent_path, *, required_origin=None, readonly=True, project_root=None):
    """Metadata checked is distinct from verified raw tensors or GPU execution."""
    if readonly is not True:
        raise ValueError("Parent evidence audit is read-only")
    docs = load_parent_metadata(parent_path)
    composition = validate_parent_metadata(
        docs["runtime_lock"],
        docs["bank_manifest"],
        docs["candidate_manifest"],
        required_origin=required_origin,
    )
    with RestrictedParentReader(parent_path) as reader:
        prefix = _warm_prefix(reader)
        raw = {
            name: {
                "present": reader.exists(prefix + name),
                "status": "NOT_AVAILABLE",
                "verification_state": "PRESENT_NOT_VERIFIED"
                if reader.exists(prefix + name)
                else "MISSING",
                "verified": False,
            }
            for name in ("origin.pt", "samples.jsonl")
        }
    runtime = docs["runtime_lock"]
    missing = [name for name, item in raw.items() if not item["present"]]
    return {
        "status": "BLOCKED_MISSING_PARENT_RAW" if missing else "BLOCKED_PARENT_RAW_NOT_VERIFIED",
        "execution_kind": "CPU_READONLY_METADATA_AUDIT",
        "parent_path": str(Path(parent_path).resolve()),
        "parent_metadata_verified": True,
        "parent_raw_verified": False,
        "raw_tensors_verified": False,
        "gpu_smoke_passed": False,
        "gpu_started": False,
        "training_started": False,
        "metadata_files": [
            {"path": name, "sha256": digest, "status": "CHECKED"}
            for name, digest in PINNED_SHA256.items()
        ],
        "identity": dict(runtime["identity"]),
        "origin_hash": runtime["origin_hash"],
        "origin_file_sha256": runtime["origin_file_sha256"],
        "recorded_source_commit": runtime["source"]["source_commit"],
        "recorded_environment": runtime["environment"],
        "raw_artifacts": raw,
        "missing_parent_raw": missing,
        "additional_server_checks": [
            "candidate checkpoint tensors",
            "raw rollout content",
            "dataset split files",
            "original images",
            "prepared model input tensors",
        ],
        "bank_composition": composition,
        "source_files": audit_source_files(runtime, project_root)
        if project_root is not None
        else [],
        "source_comparison_status": "CHECKED" if project_root is not None else "NOT_AVAILABLE",
    }
