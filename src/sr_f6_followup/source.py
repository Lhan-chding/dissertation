"""Bind a new consumer without relabeling or modifying the frozen producer.

Launch with the frozen producer's src before this deployment's src. Each package
is checked against its own inventory. A disk PASS alone grants no permission:
every cold process reconstructs the complete frozen confirmation and analysis.
Warm calls reuse that private result only while every bound observation and the
original evidence membership remain unchanged. Model payloads use the producer's
existing content/stat bindings, never a new weight inventory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")
ROOT = PROJECT / "sr_f6_confirm_followup_v2_20261029"
PRODUCER_ROOT = PROJECT / "sr_f6_confirm_v3_20261028_r1"
PRODUCER_DEPLOYMENT = PROJECT / "sr_f6_code_6e5d363"
PRODUCER_REVISION = "6e5d36351b8c1f6f18c61661246191772ba6a6dd"
PRODUCER_REGISTRATION_SHA256 = "483dfbb8efaf569c1a5041cb8f715baf23a46bd8e40e8e6f1f18d357523ccf79"
SCHEMA = "srf6-followup-registration-v2"
SOURCE_SCHEMA = "srf6-followup-source-v1"
SUPPORT_SCHEMA = "srf6-followup-support-binding-v1"
SUPPORT_STATUS = "CONFIRM_SUPPORT_PASS_REVIEW_PENDING"
ANALYSIS = "CONFIRM/ANALYSIS"
_SUPPORT_CACHE = {}
_SOURCE_VERIFIED = set()


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _payload(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path(path, *, exists=True):
    path = Path(path)
    _require(
        path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
        "Followup path is outside the authorized project",
    )
    _require(PROJECT.resolve(strict=True) == PROJECT, "Canonical project required")
    _require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "Symlinked write or evidence path forbidden",
    )
    _require(path.resolve(strict=exists) == path, "Canonical followup path required")
    return path


def _root(root, *, exists=True):
    root = _path(root, exists=exists)
    _require(root == ROOT, "Only the separately registered followup root is allowed")
    return root


def _write(path, value):
    path = _path(path, exists=False)
    _require(path.parent.is_dir(), "Create declared evidence directory before writing")
    with path.open("xb") as stream:
        stream.write(_payload(value))
        stream.flush()
        os.fsync(stream.fileno())


def _reference(path):
    path = _path(path)
    return dict(path=str(path), sha256=_hash(path))


def _bound(ref):
    _require(
        isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact content reference required"
    )
    path = _path(ref["path"])
    _require(_hash(path) == ref["sha256"], "Bound followup evidence changed")
    return _read(path)


def _actual_deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _deployment(revision):
    _require(
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "Full immutable consumer revision required",
    )
    deployment = _path(_actual_deployment())
    match = re.fullmatch(r"sr_f6_followup_code_([0-9a-f]{7,40})", deployment.name)
    _require(
        deployment.parent == PROJECT and match and revision.startswith(match.group(1)),
        "Actual consumer deployment differs from its revision",
    )
    _require(deployment != PRODUCER_DEPLOYMENT, "Consumer must have its own deployment")
    return deployment


def _source_paths(deployment):
    directory = _path(deployment / "src/sr_f6_followup")
    paths = sorted(directory.rglob("*.py"))
    _require(
        directory / "source.py" in paths and directory / "__init__.py" in paths,
        "Complete separate consumer package required",
    )
    # Only the new namespace can be deployed here; it cannot shadow 6e's sr_f6.
    _require(
        {p.name for p in (deployment / "src").iterdir()} == {"sr_f6_followup"},
        "Consumer deployment must not shadow frozen producer packages",
    )
    paths.extend(sorted((deployment / "scripts/sr_f6_followup").glob("*.sbatch")))
    return sorted({_path(path) for path in paths})


def _stamp(value):
    return {
        name: getattr(value, "st_" + name)
        for name in ("ino", "size", "mtime_ns", "ctime_ns", "mode")
    }


def _state(path):
    """Read-only aliases may target canonical project files, never external files."""
    path = Path(path)
    _require(
        path.is_absolute() and path.is_relative_to(PROJECT) and ".." not in path.parts,
        "Read dependency escapes the project",
    )
    resolved = _path(path.resolve(strict=True))
    _require(stat.S_ISREG(resolved.stat().st_mode), "Regular read dependency required")
    aliases = [
        dict(path=str(p), target=str(p.readlink()), lstat=_stamp(p.lstat()))
        for p in (path, *path.parents)
        if p.is_symlink()
    ]
    return dict(
        path=str(path),
        resolved_path=str(resolved),
        lstat=_stamp(path.lstat()),
        stat=_stamp(path.stat()),
        aliases=aliases,
    )


def _inventory(path):
    before = _state(path)
    pin = _hash(path)
    _require(_state(path) == before, "Consumer source changed while binding")
    return dict(**before, sha256=pin)


def _verify_module_origins(deployment, files, modules=None):
    """Unlike the producer helper, also verify -m entrypoints and package paths."""
    members = {row["path"] for row in files}
    for module in tuple(sys.modules.values()) if modules is None else modules:
        spec = getattr(module, "__spec__", None)
        names = (getattr(module, "__name__", ""), getattr(spec, "name", ""))
        path = getattr(module, "__file__", None)
        relevant = any(name and name.split(".")[0] == "sr_f6_followup" for name in names)
        relevant |= bool(
            path and Path(path).absolute().is_relative_to(deployment / "src/sr_f6_followup")
        )
        if not relevant:
            continue
        _require(path is not None, "Unbound consumer namespace module")
        actual = _path(Path(path).absolute())
        _require(
            actual.is_relative_to(deployment) and str(actual) in members,
            "Imported consumer source is outside its manifest",
        )
        for package_path in getattr(module, "__path__", ()):
            _require(
                _path(package_path).is_relative_to(deployment / "src/sr_f6_followup"),
                "Consumer package search path escaped its deployment",
            )


def _producer_context():
    from sr_f6 import confirm_registration

    _require(
        Path(confirm_registration.__file__).resolve(strict=True)
        == PRODUCER_DEPLOYMENT / "src/sr_f6/confirm_registration.py",
        "Import sr_f6 from the actual frozen 6e producer deployment",
    )
    root = _path(PRODUCER_ROOT)
    _require(
        _hash(root / "REGISTRATION.json") == PRODUCER_REGISTRATION_SHA256,
        "Frozen producer registration differs",
    )
    context = confirm_registration.verify_registered(root, require_preflight=True)
    _require(
        context["root"] == root
        and context["deployment"] == PRODUCER_DEPLOYMENT
        and context["registration_sha256"] == PRODUCER_REGISTRATION_SHA256
        and context["registration"]["revision"] == PRODUCER_REVISION,
        "Actual frozen producer identity differs",
    )
    return context


def _producer_identity(context):
    return dict(
        root=str(PRODUCER_ROOT),
        deployment=str(PRODUCER_DEPLOYMENT),
        revision=PRODUCER_REVISION,
        registration_sha256=PRODUCER_REGISTRATION_SHA256,
        source_manifest_sha256=context["registration"]["source_manifest_sha256"],
        data_sha256=context["registration"]["data_sha256"],
        spec_sha256=context["spec_sha256"],
        spec_object_sha256=context["spec_object_sha256"],
    )


def repair_history():
    """Read the immutable failed v1 closure; it conveys no authorization."""
    from .repair_history import verify

    return verify()


def _spec(context):
    from sr_f6.confirm_analysis import _followups

    from . import review_contract

    followups = _followups(context["spec"], list(context["tasks"].values()))
    history = repair_history()
    return dict(
        schema="srf6-followup-spec-v2",
        phase="CONFIRM_FOLLOWUP_V2",
        split="DEV_SCHEMA_CONFIRM",
        producer=_producer_identity(context),
        requirements=copy.deepcopy(context["spec"]["followup_requirements"]),
        original_requirements=followups,
        review_contract=review_contract.specification(),
        repair_history=dict(
            status="HISTORICAL_NEGATIVE_EVIDENCE_NOT_AUTHORITY",
            references=copy.deepcopy(history["references"]),
            consumer=copy.deepcopy(history["summary"]["consumer"]),
            review_status=history["summary"]["status"],
            completed=False,
        ),
        supported_purposes=["review", "t03", "admission"],
        review_requires="COMPLETE_FROZEN_CONFIRM_SUPPORT_PASS",
        t03_requires="COMPLETE_FROZEN_CONFIRM_SUPPORT_PASS_AND_NEW_360_REVIEW",
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        test_results_read=False,
    )


def register(root, revision):
    root = _root(root, exists=False)
    _require(not root.exists(), "Preserve existing or partial followup registration")
    deployment = _deployment(revision)
    producer = _producer_context()
    files = [_inventory(path) for path in _source_paths(deployment)]
    _verify_module_origins(deployment, files)
    manifest = dict(
        schema=SOURCE_SCHEMA, revision=revision, deployment=str(deployment), files=files
    )
    spec = _spec(producer)
    _verify_module_origins(deployment, files)
    root.mkdir(exist_ok=False)
    _write(root / "SOURCE_MANIFEST.json", manifest)
    _write(root / "SPEC.json", spec)
    _write(
        root / "REGISTRATION.json",
        dict(
            schema=SCHEMA,
            root=str(root),
            revision=revision,
            deployment=str(deployment),
            producer=_producer_identity(producer),
            source_manifest=_reference(root / "SOURCE_MANIFEST.json"),
            spec=_reference(root / "SPEC.json"),
            status="BOUND_NOT_AUTHORIZED",
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
            model_calls=0,
            optimizer_updates=0,
        ),
    )
    return verify_registered(root)


def _verify_base(root):
    root = _root(root)
    registration_path = _path(root / "REGISTRATION.json")
    record = _read(registration_path)
    _require(
        record.get("schema") == SCHEMA
        and record.get("root") == str(root)
        and record.get("status") == "BOUND_NOT_AUTHORIZED"
        and all(
            record.get(k) is False
            for k in ("training_allowed", "generation_allowed", "evaluation_allowed")
        )
        and type(record.get("model_calls")) is int
        and record["model_calls"] == 0
        and type(record.get("optimizer_updates")) is int
        and record["optimizer_updates"] == 0,
        "Followup registration grants no model or training permission",
    )
    deployment = _deployment(record["revision"])
    _require(record["deployment"] == str(deployment), "Consumer executing deployment differs")
    _require(
        record["source_manifest"]["path"] == str(root / "SOURCE_MANIFEST.json")
        and record["spec"]["path"] == str(root / "SPEC.json"),
        "Unexpected registration refs",
    )
    manifest = _bound(record["source_manifest"])
    files = manifest["files"]
    _require(
        manifest.get("schema") == SOURCE_SCHEMA
        and manifest.get("revision") == record["revision"]
        and manifest.get("deployment") == str(deployment)
        and len(files) == len({r["path"] for r in files})
        and {r["path"] for r in files} == {str(p) for p in _source_paths(deployment)},
        "Complete consumer source inventory differs",
    )
    for item in files:
        _require(
            _state(item["path"]) == {k: v for k, v in item.items() if k != "sha256"},
            "Consumer source changed after registration",
        )
        # Small new source files receive a cold process content check; never model weights.
        cache_key = (item["path"], item["sha256"], _digest(item))
        if cache_key not in _SOURCE_VERIFIED:
            _require(_hash(item["path"]) == item["sha256"], "Consumer source content differs")
            _SOURCE_VERIFIED.add(cache_key)
    _verify_module_origins(deployment, files)
    producer = _producer_context()
    _require(record["producer"] == _producer_identity(producer), "Registered producer changed")
    spec = _bound(record["spec"])
    _require(spec == _spec(producer), "Original 360/T03 selection or followup contract changed")
    _verify_module_origins(deployment, files)
    return dict(
        root=root,
        registration=record,
        spec=spec,
        consumer=dict(
            deployment=str(deployment),
            revision=record["revision"],
            source_manifest_sha256=record["source_manifest"]["sha256"],
            registration_sha256=_hash(_path(registration_path)),
        ),
        producer_context=producer,
        source_manifest=manifest,
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
    )


def _dependency_states(validation, context, extra_refs):
    refs = [
        dict(path=str(PRODUCER_ROOT / row["path"]), sha256=row["sha256"])
        for row in validation["inputs"]
    ]
    refs.extend(validation["external_inputs"])
    refs.extend(extra_refs)
    refs.extend(context["spec"]["repair_history"]["references"])
    refs.extend(
        _reference(context["root"] / name)
        for name in ("REGISTRATION.json", "SPEC.json", "SOURCE_MANIFEST.json")
    )
    refs.extend(
        dict(path=row["path"], sha256=row["sha256"]) for row in context["source_manifest"]["files"]
    )
    unique = {}
    for ref in refs:
        _require(
            isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Invalid dependency pin",
        )
        _require(unique.setdefault(ref["path"], ref) == ref, "Conflicting dependency pins")
    return [dict(**_state(path), sha256=unique[path]["sha256"]) for path in sorted(unique)]


def _unchanged(rows):
    for row in rows:
        _require(
            _state(row["path"]) == {k: v for k, v in row.items() if k != "sha256"},
            "Bound support dependency changed; do not reuse cached permission",
        )


def _analysis_ref(relative, pin):
    path = Path(relative)
    _require(
        not path.is_absolute() and ".." not in path.parts and path.parent == Path(ANALYSIS),
        "Unexpected frozen analysis reference",
    )
    return dict(path=str(PRODUCER_ROOT / path), sha256=pin)


def _reconstruct_support(context):
    from sr_f6 import confirm_analysis, confirm_evidence

    producer = context["producer_context"]
    assessment_ref = _reference(PRODUCER_ROOT / ANALYSIS / "ASSESSMENT.json")
    seal_ref = _reference(PRODUCER_ROOT / ANALYSIS / "SEAL.json")
    stored, seal = _bound(assessment_ref), _bound(seal_ref)
    _require(
        stored.get("status") == SUPPORT_STATUS and stored.get("support_status") == "PASS",
        "Complete frozen SUPPORT_PASS required; failure or UNKNOWN grants nothing",
    )
    refs = {
        name: _analysis_ref(ref["path"], ref["sha256"]) for name, ref in stored["reports"].items()
    }
    _require(
        all(ref["path"] == str(PRODUCER_ROOT / ANALYSIS / name) for name, ref in refs.items()),
        "Frozen report names must retain their exact saved paths",
    )
    _require("VALIDATION.json" in refs, "Complete frozen validation report required")
    validation = _bound(refs["VALIDATION.json"])
    states = _dependency_states(validation, context, [assessment_ref, seal_ref, *refs.values()])
    snapshot = confirm_evidence._snapshot(PRODUCER_ROOT)
    evidence = confirm_evidence.validate_saved_confirm(PRODUCER_ROOT)
    _require(
        evidence.get("status") == "VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED"
        and evidence.get("registration_sha256") == PRODUCER_REGISTRATION_SHA256
        and evidence.get("source_rows") == evidence.get("actual_recorded_calls") == 9216
        and len(evidence["records"]) == len(evidence["raw_by_row"]) == 9216
        and evidence["saved_token_validation"]["status"] == "PASS"
        and evidence["saved_token_validation"]["checked_responses"] == 9216,
        "All original confirmation records, tokens and accounting must verify",
    )
    report, files = confirm_analysis._analyze(PRODUCER_ROOT, evidence)
    _require(set(refs) == set(files), "Frozen analysis report membership differs")
    for name, value in files.items():
        _require(_bound(refs[name]) == value, "Reconstructed frozen analysis differs: " + name)
    report.update(
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
        gate_status="NOT_SIGNED",
        new_model_calls=0,
        optimizer_updates=0,
        reports=stored["reports"],
    )
    _require(
        report == stored
        and report["status"] == SUPPORT_STATUS
        and report["support_status"] == "PASS",
        "Saved support PASS is not reproducible",
    )
    _require(
        seal
        == dict(
            schema="srf6-confirm-analysis-seal-v1",
            status=SUPPORT_STATUS,
            assessment=dict(path=f"{ANALYSIS}/ASSESSMENT.json", sha256=assessment_ref["sha256"]),
            registration_sha256=PRODUCER_REGISTRATION_SHA256,
            gate_status="NOT_SIGNED",
            training_allowed=False,
            generation_allowed=False,
        ),
        "Frozen analysis seal differs",
    )
    _require(
        evidence["input_producer"]["revision"] == PRODUCER_REVISION
        and evidence["input_producer"]["deployment"] == str(PRODUCER_DEPLOYMENT),
        "Confirmation facts came from a different producer",
    )
    _unchanged(states)
    _require(
        confirm_evidence._snapshot(PRODUCER_ROOT) == snapshot,
        "Confirmation membership changed during support binding",
    )
    # Also recheck imported modules after analysis may have imported dependencies.
    checked = _verify_base(context["root"])
    _require(checked["consumer"] == context["consumer"], "Consumer changed during validation")
    binding = dict(
        schema=SUPPORT_SCHEMA,
        status="VERIFIED_SUPPORT_PASS_REVIEW_PENDING",
        consumer=context["consumer"],
        producer=_producer_identity(producer),
        assessment=assessment_ref,
        seal=seal_ref,
        reports=refs,
        records_sha256=_digest(evidence["records"]),
        raw_records_sha256=_digest(evidence["raw_by_row"]),
        dependencies=states,
        membership={k: sorted(v) for k, v in snapshot.items()},
        source_rows=9216,
        new_model_calls=0,
        optimizer_updates=0,
        training_allowed=False,
        generation_allowed=False,
        evaluation_allowed=False,
    )
    return binding, evidence


def bind_support(root):
    context = _verify_base(root)
    root = context["root"]
    _require(
        not any(
            (root / name).exists() or (root / name).is_symlink()
            for name in (
                "SUPPORT_BINDING.json",
                "SUPPORT_BINDING.attempt.json",
                "SUPPORT_BINDING.failure.json",
            )
        ),
        "Preserve prior or partial support binding",
    )
    _write(
        root / "SUPPORT_BINDING.attempt.json",
        dict(
            schema="srf6-followup-support-attempt-v1",
            consumer=context["consumer"],
            producer=_producer_identity(context["producer_context"]),
            status="STARTED",
            model_calls=0,
            optimizer_updates=0,
        ),
    )
    try:
        binding, evidence = _reconstruct_support(context)
        _write(root / "SUPPORT_BINDING.json", binding)
    except Exception as exc:
        _write(
            root / "SUPPORT_BINDING.failure.json",
            dict(
                status="STOP_PRESERVE",
                error=type(exc).__name__ + ": " + str(exc),
                training_allowed=False,
                generation_allowed=False,
                evaluation_allowed=False,
            ),
        )
        raise
    saved, reference, state = _read_support_file(root)
    _require(saved == binding, "Written support binding differs")
    _SUPPORT_CACHE[str(root)] = (copy.deepcopy(binding), evidence, reference, state)
    return verify_registered(root, require_support=True)


def _read_support_file(root):
    """Read and hash the binding once on a cold process, before any authority."""
    path = _path(root / "SUPPORT_BINDING.json")
    before = _state(path)
    payload = path.read_bytes()
    _require(_state(path) == before, "Support binding changed while reading")
    return (
        json.loads(payload),
        dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()),
        before,
    )


def _support_record(root):
    failure = root / "SUPPORT_BINDING.failure.json"
    _require(not (failure.exists() or failure.is_symlink()), "Failed support attempt preserved")
    cached = _SUPPORT_CACHE.get(str(root))
    if cached is None:
        return _read_support_file(root)
    path = _path(root / "SUPPORT_BINDING.json")
    _require(_state(path) == cached[3], "Cached support binding changed")
    # Only this module's completed reconstruction populates the private cache.
    return cached[0], cached[2], cached[3]


def _cached_or_rebuild(context, saved, reference, state):
    from sr_f6 import confirm_evidence

    root = context["root"]
    _require(saved.get("schema") == SUPPORT_SCHEMA, "Exact support binding required")
    attempt = _read(_path(root / "SUPPORT_BINDING.attempt.json"))
    _require(
        attempt
        == dict(
            schema="srf6-followup-support-attempt-v1",
            consumer=context["consumer"],
            producer=_producer_identity(context["producer_context"]),
            status="STARTED",
            model_calls=0,
            optimizer_updates=0,
        ),
        "Support attempt identity differs",
    )
    _unchanged(saved["dependencies"])
    cached = _SUPPORT_CACHE.get(str(root))
    if cached is None:
        actual, evidence = _reconstruct_support(context)
        _require(saved == actual, "Disk support binding differs from complete cold reconstruction")
        _require(
            _state(_path(root / "SUPPORT_BINDING.json")) == state,
            "Support binding changed during cold reconstruction",
        )
        cached = (copy.deepcopy(actual), evidence, reference, state)
        _SUPPORT_CACHE[str(root)] = cached
    _require(
        saved == cached[0]
        and reference == cached[2]
        and _state(_path(root / "SUPPORT_BINDING.json")) == cached[3],
        "Cached support binding changed",
    )
    _require(
        {k: sorted(v) for k, v in confirm_evidence._snapshot(PRODUCER_ROOT).items()}
        == saved["membership"],
        "Confirmation evidence membership changed",
    )
    evidence = copy.deepcopy(cached[1])
    return dict(
        context,
        support_binding=copy.deepcopy(saved),
        evidence=evidence,
        records=evidence["records"],
        raw_by_row=evidence["raw_by_row"],
        support_binding_ref=copy.deepcopy(cached[2]),
    )


def verify_registered(root, require_support=False):
    _require(type(require_support) is bool, "Explicit support requirement required")
    # Reject changes before producer validation could rehash a changed model.
    if require_support:
        root = _root(root)
        saved, reference, state = _support_record(root)
        _unchanged(saved["dependencies"])
    context = _verify_base(root)
    return _cached_or_rebuild(context, saved, reference, state) if require_support else context


def require_support(root, purpose):
    _require(purpose in ("review", "t03", "admission"), "Unknown followup permission purpose")
    context = verify_registered(root, require_support=True)
    if purpose in ("t03", "admission"):
        from .review_audit import verify_completed

        review = verify_completed(root)
        _require(
            review.get("status") == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
            and review.get("completed") is True
            and type(review.get("reviewed_original_answers")) is int
            and review["reviewed_original_answers"] == 360
            and type(review.get("independently_reviewed_original_answers")) is int
            and review["independently_reviewed_original_answers"] == 360
            and review.get("consumer") == context["consumer"]
            and review.get("input_producer") == context["evidence"]["input_producer"]
            and review.get("training_allowed") is False
            and review.get("gate_status") == "NOT_SIGNED"
            and review.get("synthetic") is False,
            "Actual complete new 360-answer independent review required",
        )
        _verify_module_origins(
            Path(context["consumer"]["deployment"]), context["source_manifest"]["files"]
        )
        context["review"] = review
    # A purpose-specific consumer must additionally enforce its own access/session.
    return dict(context, purpose=purpose, permission="BOUND_FOLLOWUP_PURPOSE_ONLY")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "bind-support", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.action == "register":
        result = register(args.root, args.revision)
    elif args.action == "bind-support":
        result = bind_support(args.root)
    else:
        result = verify_registered(args.root)
    print(
        json.dumps(
            dict(
                consumer=result["consumer"],
                training_allowed=False,
                generation_allowed=False,
                evaluation_allowed=False,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
