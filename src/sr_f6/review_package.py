"""Immutable review material from the complete verified S0 census.

The writer/consumer actually executes PILOT_V1 code. CENSUS_REVIEW_V1/86a760b
supplies only the frozen sampler recipe; equal algorithm bytes do not make a
new execution an old producer. No caller-supplied records, labels, summary,
seed, destination or support verdict are accepted by these production APIs.

write_package and verify_package each validate all 9,216 saved responses and
replay the fixed sampler once. Use verify_package at final evidence binding or
completed-review delivery, NOT once per review shard. Its verified artifacts
are returned for the audit aggregator to reuse in that same call.

No model is called. Support failures remain in the package and never modify
selection. A package is not an independent review, a scientific gate, or a
training authorization. Synthetic fixtures must carry their explicit origin.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

from . import census_evidence, census_support, phases, review
from .common import digest, file_hash, safe_evidence

SCHEMA = "srf6-review-package-binding-v2"
PHASE = "PILOT_V1"
RECIPE_PHASE = "CENSUS_REVIEW_V1"
RECIPE_REVISION = "86a760b7d2b0df93e4f137da13ad1aa222ee730e"
DESTINATION = "PHASES/PILOT_V1/REVIEW_PACKAGE"
SPEC_PATH = "PHASES/CENSUS_REVIEW_V1/REVIEW_SPEC.json"


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _sha(value):
    _require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "Full SHA256 pin required"
    )
    return value


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Nonfinite evidence")

    def finite_number(token):
        value = float(token)
        if not math.isfinite(value):
            nonfinite(token)
        return value

    return json.loads(
        raw, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_number
    )


def _payload(value, *, lines=False):
    if lines:
        return "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value
        ).encode()
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


class _Reader:
    def __init__(self, root):
        self.root, self.inputs = Path(root), {}

    def read(self, relative, *, expected=None, lines=False):
        path = safe_evidence(self.root, relative)
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        _require(
            (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Review input changed while reading",
        )
        ref = {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}
        if expected is not None:
            _require(ref["sha256"] == _sha(expected), "Review input differs from its frozen pin")
        _require(self.inputs.setdefault(relative, ref) == ref, "Conflicting review input identity")
        return [_json(line) for line in raw.splitlines() if line.strip()] if lines else _json(raw)

    def adopt(self, references):
        for ref in references:
            _require(
                isinstance(ref, dict) and set(ref) == {"path", "sha256"},
                "Explicit evidence reference required",
            )
            _sha(ref["sha256"])
            _require(
                self.inputs.setdefault(ref["path"], dict(ref)) == ref,
                "Conflicting verified evidence reference",
            )


def _deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _sources(root, reader, phase_pin, sampler_pin, spec_pin):
    from .runtime import _verify_module_origins

    for pin in (phase_pin, sampler_pin, spec_pin):
        _sha(pin)
    actual = _deployment()
    producer = phases.verify_phase(
        root, PHASE, deployment=actual, expected_identity_sha256=phase_pin
    )
    _require(
        producer.get("phase") == PHASE
        and producer.get("identity_sha256") == phase_pin
        and producer.get("status") == "BOUND_NOT_AUTHORIZED"
        and producer.get("training_allowed") is False
        and producer.get("deployment") == str(actual),
        "Actual writer source is not the pinned PILOT deployment",
    )
    identity = reader.read(f"PHASES/{PHASE}/IDENTITY.json", expected=phase_pin)
    _require(identity == producer["identity"], "PILOT identity differs from its file")
    current_manifest = reader.read(
        identity["source_manifest"]["path"], expected=identity["source_manifest"]["sha256"]
    )
    _verify_module_origins(actual, current_manifest, tuple(sys.modules.values()))
    recipe_identity = reader.read(f"PHASES/{RECIPE_PHASE}/IDENTITY.json", expected=sampler_pin)
    recipe = phases.verify_phase(
        root,
        RECIPE_PHASE,
        deployment=recipe_identity["deployment"],
        expected_identity_sha256=sampler_pin,
    )
    _require(
        recipe.get("identity") == recipe_identity
        and recipe.get("identity_sha256") == sampler_pin
        and recipe_identity.get("revision") == RECIPE_REVISION
        and recipe.get("phase") == RECIPE_PHASE,
        "Sampler must be the immutable 86a760b review recipe",
    )
    _require(
        identity["origin"] == recipe_identity["origin"]
        and identity["data_manifest"] == recipe_identity["data_manifest"]
        and identity["config_sha256"] == recipe_identity["config_sha256"],
        "PILOT and sampler recipe have different original inputs",
    )
    spec_ref = {"path": SPEC_PATH, "sha256": spec_pin}
    _require(
        any(
            entry.get("producer_phase") == RECIPE_PHASE
            and entry.get("identity_sha256") == sampler_pin
            and spec_ref in entry.get("evidence", [])
            for entry in identity["upstream"]
        ),
        "REVIEW_SPEC must be a frozen PILOT upstream dependency",
    )
    spec = reader.read(SPEC_PATH, expected=spec_pin)
    expected = dict(
        status="REGISTERED_REVIEW_SAMPLER_NOT_EXECUTED",
        context=recipe_identity,
        phase_identity_sha256=sampler_pin,
        version=review.VERSION,
        seed=2026102701,
        seed_domains=review.DOMAINS,
        required_count=360,
        per_parent_factor=60,
        per_family_within_cell=10,
        root_cap=4,
        root_cap_scope="parent_factor",
        unique_key=["parent", "factor", "qid", "exact_full_text"],
        raw_source=(
            "Original CENSUS/raw records, aligned by frozen qid and draw; "
            "never group/scorer-normalized finish_reason"
        ),
        labels_visible_to_blind_extractors=False,
        review_status="NOT_RUN",
        model_calls=0,
        optimizer_updates=0,
        training_allowed=False,
    )
    _require(
        all(spec.get(key) == value for key, value in expected.items()),
        "Frozen review recipe semantics changed",
    )
    manifest = reader.read(
        recipe_identity["source_manifest"]["path"],
        expected=recipe_identity["source_manifest"]["sha256"],
    )
    entries = [item for item in manifest["files"] if item["path"] == "src/sr_f6/review.py"]
    _require(len(entries) == 1, "Frozen sampler module is missing or duplicated")
    module = Path(review.__file__).resolve(strict=True)
    _require(
        module == actual / "src/sr_f6/review.py",
        "Loaded sampler module is outside actual PILOT deployment",
    )
    _require(
        file_hash(module) == entries[0]["sha256"],
        "Current sampler bytes differ from frozen 86a760b algorithm",
    )
    sampler = dict(
        phase=RECIPE_PHASE,
        identity=recipe_identity,
        identity_sha256=sampler_pin,
        deployment=recipe["deployment"],
        spec=spec_ref,
        source_manifest=recipe_identity["source_manifest"],
        module=dict(path=entries[0]["path"], sha256=entries[0]["sha256"]),
    )
    return producer, sampler


def _loaded_dependencies(reader, identity):
    """Also bind dependencies imported lazily during the complete CPU replay."""
    from .runtime import _verify_module_origins

    manifest = reader.read(
        identity["source_manifest"]["path"], expected=identity["source_manifest"]["sha256"]
    )
    _verify_module_origins(_deployment(), manifest, tuple(sys.modules.values()))


def _bound_data(reader, identity, relative, *, lines=False):
    manifest = reader.read(
        identity["data_manifest"]["path"], expected=identity["data_manifest"]["sha256"]
    )
    matches = [row for row in manifest["files"] if row["path"] == relative]
    _require(len(matches) == 1, "Required review data member missing or duplicated")
    return reader.read(relative, expected=matches[0]["sha256"], lines=lines)


def _derive(root, *, phase_identity_sha256, sampler_phase_identity_sha256, review_spec_sha256):
    reader = _Reader(root)
    producer, sampler = _sources(
        root, reader, phase_identity_sha256, sampler_phase_identity_sha256, review_spec_sha256
    )
    identity = producer["identity"]
    verified = census_evidence.validate_saved_census(
        root, origin_deployment=identity["origin"]["deployment"]
    )
    _require(
        verified.get("status") == "VERIFIED_STORED_CENSUS_NOT_GATE_SIGNED"
        and verified.get("training_authorized") is False
        and verified.get("actual_recorded_calls") == 9216
        and len(verified.get("records", [])) == len(verified.get("raw_by_row", {})) == 9216,
        "All 9216 original census responses must be validated",
    )
    _require(
        verified["input_producer"]["phase"] == "CENSUS_S0"
        and verified["input_producer"]["identity_sha256"]
        == identity["origin"]["identity"]["sha256"],
        "Verified census is not the frozen S0 producer",
    )
    reader.adopt(verified["inputs"])
    frozen_spec_ref = reader.inputs.get("CENSUS_SPEC_FROZEN.json")
    _require(frozen_spec_ref is not None, "Verified census specification reference missing")
    source_identity = dict(
        execution_phase_identity_sha256=phase_identity_sha256,
        execution_source_manifest_sha256=identity["source_manifest"]["sha256"],
        sampler_phase_identity_sha256=sampler_phase_identity_sha256,
        sampler_module_sha256=sampler["module"]["sha256"],
        review_spec_sha256=review_spec_sha256,
        census_origin_identity_sha256=verified["input_producer"]["identity_sha256"],
        census_spec_sha256=frozen_spec_ref["sha256"],
        census_inputs_sha256=digest(verified["inputs"]),
        data_manifest_sha256=identity["data_manifest"]["sha256"],
    )
    ledger, blind = review.build_review(
        verified["records"],
        raw_by_row=verified["raw_by_row"],
        source_identity=source_identity,
        seed=2026102701,
    )
    _require(
        ledger["sampling_status"] == "COMPLETE_360" and len(blind) == 360,
        "Insufficient unique original responses; never duplicate or change sampling",
    )
    task_source = "DATA/DEV_CAL/TASKS_PRIVATE.jsonl"
    tasks = _bound_data(reader, identity, task_source, lines=True)
    task_map = {row["qid"]: row for row in tasks}
    _require(len(task_map) == len(tasks) == 384, "Complete frozen DEV_CAL task matrix required")
    judgment = []
    for row in ledger["selected"]:
        task = task_map[row["qid"]]
        _require(
            task["split"] == "DEV_CAL"
            and all(task[k] == row[k] for k in ("root_id", "family", "factor")),
            "Selected answer lineage differs from frozen task",
        )
        judgment.append(
            dict(anonymous_id=row["anonymous_id"], world=task["world"], query=task["query"])
        )
    cpu_audit = _bound_data(reader, identity, "DATA/CPU_AUDIT.json")
    certificates = _bound_data(reader, identity, "DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", lines=True)
    train_roots = [row for row in certificates if str(row.get("root_id", "")).startswith("TRAIN_")]
    support = census_support.assess_support(verified["summary"], cpu_audit, train_roots=train_roots)
    _loaded_dependencies(reader, identity)
    support.update(
        context=identity,
        input_producer=verified["input_producer"],
        evidence_origin="VERIFIED_SAVED_CENSUS_AND_FROZEN_DATA",
        inputs=[reader.inputs[p] for p in sorted(reader.inputs)],
    )
    validation = {
        key: value for key, value in verified.items() if key not in ("records", "raw_by_row")
    }
    validation.update(
        context=identity,
        execution_producer=producer,
        selection_provenance="RECOMPUTED_FROM_VERIFIED_FULL_CENSUS",
        selection_inputs_sha256=ledger["selection_inputs_sha256"],
    )
    synthetic = any(
        row.get("execution_origin") == "SYNTHETIC_TEST" for row in verified["raw_by_row"].values()
    )
    if synthetic:
        support["execution_origin"] = validation["execution_origin"] = "SYNTHETIC_TEST"
    files = {
        "private/selection.json": _payload(ledger),
        "blind/EXTRACTION.jsonl": _payload(blind, lines=True),
        "judgment/TASKS.jsonl": _payload(judgment, lines=True),
        "private/SUPPORT.json": _payload(support),
        "private/CENSUS_VALIDATION.json": _payload(validation),
    }
    references = {
        name: {"path": f"{DESTINATION}/{name}", "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in files.items()
    }
    package = dict(
        schema=SCHEMA,
        context=identity,
        producer=producer,
        phase_identity_sha256=phase_identity_sha256,
        sampler_recipe=sampler,
        input_producer=verified["input_producer"],
        execution_origin="SYNTHETIC_TEST" if synthetic else "ACTUAL_CENSUS",
        status="BLIND_PACKAGE_ONLY",
        review_status="REVIEW_NOT_RUN",
        training_allowed=False,
        gate_status="NOT_SIGNED",
        model_calls=0,
        optimizer_updates=0,
        ledger=references["private/selection.json"],
        blind=references["blind/EXTRACTION.jsonl"],
        judgment_tasks=references["judgment/TASKS.jsonl"],
        support=references["private/SUPPORT.json"],
        census_validation=references["private/CENSUS_VALIDATION.json"],
        task_source=reader.inputs[task_source],
        ledger_source_identity=source_identity,
        selection_provenance="RECOMPUTED_FROM_VERIFIED_FULL_CENSUS",
        selection_inputs_sha256=ledger["selection_inputs_sha256"],
        blind_items_sha256=ledger["blind_items_sha256"],
        inputs=[reader.inputs[p] for p in sorted(reader.inputs)]
        + [references[name] for name in sorted(references)],
    )
    files["BINDING.json"] = _payload(package)
    return dict(
        producer=producer,
        package=package,
        files=files,
        ledger=ledger,
        blind=blind,
        tasks=task_map,
        support=support,
    )


def _directory(root, relative):
    root = Path(root)
    _require(
        root.is_absolute() and root.resolve(strict=True) == root, "Canonical run root required"
    )
    path = root / relative
    _require(
        path.is_relative_to(root) and ".." not in Path(relative).parts,
        "Review package path escaped run",
    )
    for item in (path, *path.parents):
        _require(not item.is_symlink(), "Symlinked review package directory forbidden")
        _require(not item.exists() or item.is_dir(), "Review package directory has wrong type")
        if item == root:
            break
    return path


def _require_commit(directory):
    _require(
        (directory / "BINDING.json").is_file() and not (directory / "BINDING.json").is_symlink(),
        "Partial package preserved; BINDING commit is absent",
    )


def _compare(directory, files):
    _require(
        directory.is_dir() and not directory.is_symlink(), "Missing complete immutable package"
    )
    _require_commit(directory)
    expected_directories = {str(parent) for name in files for parent in Path(name).parents}
    actual = set()
    for current, directories, names in os.walk(directory, followlinks=False):
        current = Path(current)
        _require(
            not current.is_symlink()
            and current.stat().st_mode & 0o777 == 0o700
            and str(current.relative_to(directory)) in expected_directories,
            "Review package directory permissions changed",
        )
        for name in directories:
            _require(not (current / name).is_symlink(), "Symlink in review package")
        for name in names:
            path = current / name
            relative = str(path.relative_to(directory))
            _require(
                not path.is_symlink() and path.is_file() and relative in files,
                "Unexpected package member",
            )
            _require(
                path.stat().st_mode & 0o777 == 0o600 and path.read_bytes() == files[relative],
                "Immutable package bytes or permissions differ from exact replay",
            )
            actual.add(relative)
    _require(actual == set(files), "Partial package preserved; missing members")


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename_exclusive(stage, target):
    """Atomic no-replace rename; even an uncooperative empty target is preserved."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        args = (os.fsencode(stage), os.fsencode(target), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        args = (-100, os.fsencode(stage), -100, os.fsencode(target), 1)  # AT_FDCWD, NOREPLACE
    else:
        raise OSError(errno.ENOSYS, "Atomic exclusive directory rename is unavailable")
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _linked_stage(stage, target, files):
    """Both byte-verified trees must still be the same original publication objects."""
    _compare(stage, files)
    _require(
        all(os.path.samestat((stage / name).stat(), (target / name).stat()) for name in files),
        "Foreign staging beside published package; preserve and diagnose",
    )


def _publish_links(stage, target):
    """Ceph fallback: reserve a directory, then commit BINDING with one exclusive link.

    Keep the complete stage. It is a small hard-link alias, not a second scientific
    identity. Interrupted prefixes are never repaired or mistaken for a package.
    The caller lock coordinates writers. Capture the reserved inode immediately;
    directory fds bind subsequent writes to it, rejecting later path replacement.
    """
    files = {}
    for path in stage.rglob("*"):
        _require(not path.is_symlink(), "Symlink in publication stage")
        if path.is_file():
            files[path.relative_to(stage).as_posix()] = path.read_bytes()
        else:
            _require(path.is_dir(), "Unexpected publication stage object")
    _compare(stage, files)
    target.mkdir(mode=0o700)  # Exclusive reservation; never adopt any existing directory.
    reserved = target.stat(follow_symlinks=False)
    root_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    directories = {".": root_fd}
    try:
        _require(
            os.path.samestat(reserved, os.fstat(root_fd)),
            "Reserved publication directory was replaced before opening",
        )
        _fsync_directory(target.parent)
        _require(
            os.path.samestat(reserved, target.stat(follow_symlinks=False)),
            "Reserved publication directory was replaced during reservation sync",
        )
        for relative in sorted(files, key=lambda name: (name == "BINDING.json", name)):
            parts = Path(relative).parts
            parent = "."
            for part in parts[:-1]:
                child = part if parent == "." else parent + "/" + part
                if child not in directories:
                    os.mkdir(part, mode=0o700, dir_fd=directories[parent])
                    directories[child] = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directories[parent],
                    )
                parent = child
            if relative == "BINDING.json":
                # All payload names must be durable before publishing the commit.
                for fd in directories.values():
                    os.fsync(fd)
                _require(
                    os.path.samestat(os.fstat(root_fd), target.stat(follow_symlinks=False)),
                    "Reserved publication directory was replaced",
                )
            os.link(
                stage / relative, parts[-1], dst_dir_fd=directories[parent], follow_symlinks=False
            )  # Link creation cannot replace an existing name.
        os.fsync(root_fd)
        _fsync_directory(target.parent)
        _compare(target, files)
        _linked_stage(stage, target, files)
    finally:
        for fd in directories.values():
            os.close(fd)


def _publish(stage, target):
    """Prefer atomic exclusive rename; use exclusive link publication only if unsupported."""
    try:
        _rename_exclusive(stage, target)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP):
            raise
        _publish_links(stage, target)


def write_package(
    root, *, phase_identity_sha256, sampler_phase_identity_sha256, review_spec_sha256
):
    """Validate full census once and atomically publish the one fixed package.

    Exact retries verify every byte without rewriting. Partial staging/targets
    are preserved and rejected; a complete matching stage can be safely resumed.
    Unsupported directory-rename filesystems retain the complete original stage;
    only its exact hard-link publication can coexist with a complete target.
    Support failures are retained and never change sampling or grant access.
    """
    target = _directory(root, DESTINATION)
    parent = _directory(root, "PHASES/PILOT_V1")
    _require(parent.is_dir(), "PILOT must be registered before package creation")
    lock_path = parent / ".review-package.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bundle = _derive(
            root,
            phase_identity_sha256=phase_identity_sha256,
            sampler_phase_identity_sha256=sampler_phase_identity_sha256,
            review_spec_sha256=review_spec_sha256,
        )
        files = bundle["files"]
        stage = _directory(root, "PHASES/PILOT_V1/.REVIEW_PACKAGE.staging")
        if target.exists():
            _compare(target, files)
            if stage.exists():
                _linked_stage(stage, target, files)
                # Complete link publication may have been interrupted at fsync.
                for directory in (p for p in target.rglob("*") if p.is_dir()):
                    _fsync_directory(directory)
                _fsync_directory(target)
                _fsync_directory(parent)
        else:
            if stage.exists():
                _compare(stage, files)
            else:
                stage.mkdir(mode=0o700)
                for relative, payload in files.items():
                    path = stage / relative
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    file_fd = os.open(
                        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
                    )
                    with os.fdopen(file_fd, "wb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
                    _fsync_directory(directory)
                _fsync_directory(stage)
                _compare(stage, files)
            _publish(stage, target)
            _fsync_directory(parent)
        ref = {
            "path": f"{DESTINATION}/BINDING.json",
            "sha256": hashlib.sha256(files["BINDING.json"]).hexdigest(),
        }
        return dict(
            status="BLIND_PACKAGE_ONLY",
            review_status="REVIEW_NOT_RUN",
            training_allowed=False,
            gate_status="NOT_SIGNED",
            package=ref,
            selection_provenance="RECOMPUTED_FROM_VERIFIED_FULL_CENSUS",
            support_status=bundle["support"]["status"],
            execution_origin=bundle["package"]["execution_origin"],
        )
    finally:
        os.close(fd)


def verify_package(
    root, package_ref, *, phase_identity_sha256, sampler_phase_identity_sha256, review_spec_sha256
):
    """Final-delivery check: replay all 9216 once, returning verified artifacts.

    Do not call per review shard. A single aggregate_review invocation reuses
    artifacts from this result and does not reread the full census a second time.
    """
    _require(
        isinstance(package_ref, dict)
        and set(package_ref) == {"path", "sha256"}
        and package_ref["path"] == f"{DESTINATION}/BINDING.json",
        "Only the fixed v2 package may be consumed",
    )
    _sha(package_ref["sha256"])
    target = _directory(root, DESTINATION)
    _require_commit(target)
    bundle = _derive(
        root,
        phase_identity_sha256=phase_identity_sha256,
        sampler_phase_identity_sha256=sampler_phase_identity_sha256,
        review_spec_sha256=review_spec_sha256,
    )
    _require(
        hashlib.sha256(bundle["files"]["BINDING.json"]).hexdigest() == package_ref["sha256"],
        "Package binding differs from exact full-census replay",
    )
    _compare(target, bundle["files"])
    return dict(
        status="VERIFIED_REVIEW_PACKAGE_NOT_GATE_SIGNED",
        training_allowed=False,
        gate_status="NOT_SIGNED",
        selection_provenance="RECOMPUTED_FROM_VERIFIED_FULL_CENSUS",
        reviewed_answers=0,
        producer=bundle["producer"],
        sampler_recipe=bundle["package"]["sampler_recipe"],
        input_producer=bundle["package"]["input_producer"],
        inputs=[*bundle["package"]["inputs"], dict(package_ref)],
        artifacts={key: bundle[key] for key in ("package", "ledger", "blind", "tasks", "support")},
    )
