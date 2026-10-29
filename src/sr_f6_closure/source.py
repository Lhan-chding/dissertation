"""Register a separate verifier, preserving the original UNKNOWN and all inputs.

This namespace never replaces the frozen producer or execution consumer. Its
only scientific-reader change is the explicitly proven scored/raw row-id lookup.
No sampling, numerical replay, review or training entrypoint is reachable here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from sr_f6_followup import source as original

PROJECT = original.PROJECT
ROOT = PROJECT / "sr_f6_confirmation_closure_v1_20261029"
ORIGINAL_DEPLOYMENT = PROJECT / "sr_f6_followup_code_1e4355c"
ORIGINAL_REVISION = "1e4355cdf26eb2dfd70bb3e67520981b483f5350"
ORIGINAL_REGISTRATION = "ef7e3ba365d3675995c18f9314a5ea51c95cc4b57a60a2f7d2afe0d0d250b00a"
OPS = PROJECT / "sr_f6_operations_20261027/CONFIRM_FOLLOWUP_V2_EXECUTION_20261029"
SCHEMA = "srf6-technical-closure-source-v1"
PERMISSIONS = dict(training_allowed=False, generation_allowed=False, evaluation_allowed=False)
HISTORY = {
    original.ROOT / "ADMISSION" / name: pin
    for name, pin in (
        ("ATTEMPT.json", "e5f843867e61d41eae788b498123614a6bc4b81da7736a1d8d18c7e39716334f"),
        ("ASSESSMENT.json", "39146a0a887a3800477d758102846c1be28c6517ebf2ed47822abc312ef56c2d"),
        ("RECEIPT.json", "e36083bbfcb2d6c1a66f425b9188e7bcaf5b99289dea131088c569ee93b0753e"),
    )
}
HISTORY[OPS / "T03_VERIFY_DIAGNOSTIC.json"] = (
    "584943bc9ce5cb4b4c53ea195f5d35d0e0826ceeea9f4b4ad7f8f8fb09b89368"
)

_VERIFIED = set()


def require(condition, message):
    if not condition:
        raise PermissionError(message)


def deployment(revision):
    actual = original._path(Path(__file__).resolve().parents[2])
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "Full new verifier revision required")
    require(
        actual == PROJECT / ("sr_f6_closure_code_" + revision[:7]),
        "New verifier must execute from its separate immutable deployment",
    )
    return actual


def source_paths(code):
    require(
        {p.name for p in (code / "src").iterdir()} == {"sr_f6_closure"},
        "Repair deployment must not contain shadow producer/execution packages",
    )
    package = code / "src/sr_f6_closure"
    scripts = code / "scripts/sr_f6_closure"
    require(
        {p.name for p in package.iterdir()}
        == {"__init__.py", "source.py", "runner.py", "projection.py", "admission_projection.py"}
        and {p.name for p in scripts.iterdir()} == {"bootstrap.py", "verify.sbatch"},
        "Only the complete declared repair package and launchers may be deployed",
    )
    paths = sorted(package.glob("*.py"))
    paths += sorted(scripts.glob("*"))
    require(
        {p.name for p in paths} >= {"source.py", "runner.py", "__init__.py", "bootstrap.py"},
        "Complete actual verifier and isolated launcher required",
    )
    require(all(p.is_file() for p in paths), "Unexpected repair source member")
    return [original._path(p) for p in sorted(paths)]


def module_origins(code, files):
    members = {item["path"] for item in files}
    require(
        Path(sys.modules["__main__"].__file__).absolute()
        == code / "scripts/sr_f6_closure/bootstrap.py",
        "Only the inventoried isolated bootstrap may launch actual validation",
    )
    require(
        sys.path[:3]
        == [
            str(original.PRODUCER_DEPLOYMENT / "src"),
            str(ORIGINAL_DEPLOYMENT / "src"),
            str(code / "src"),
        ],
        "The three source origins or their order changed",
    )
    for module in tuple(sys.modules.values()):
        spec = getattr(module, "__spec__", None)
        names = (getattr(module, "__name__", ""), getattr(spec, "name", ""))
        if not any(n and n.split(".")[0] == "sr_f6_closure" for n in names):
            continue
        file = getattr(module, "__file__", None)
        require(file is not None, "Unbound repair namespace")
        actual = original._path(Path(file).absolute())
        require(str(actual) in members, "Repair module is not in the new inventory")
        origin = getattr(spec, "origin", None)
        require(origin == str(actual), "Repair module spec origin differs")
        for path in getattr(module, "__path__", ()):
            require(
                original._path(path) == code / "src/sr_f6_closure",
                "Repair package search path differs",
            )


def original_context():
    require(sys.flags.isolated and sys.dont_write_bytecode, "Use the isolated -I -B launcher")
    require(
        Path(original.__file__).resolve() == ORIGINAL_DEPLOYMENT / "src/sr_f6_followup/source.py",
        "Original execution consumer import differs",
    )
    context = original.verify_registered(original.ROOT)
    require(
        context["consumer"]["revision"] == ORIGINAL_REVISION
        and context["consumer"]["registration_sha256"] == ORIGINAL_REGISTRATION,
        "Original execution registration differs",
    )
    return context


def history():
    references = []
    for path, pin in HISTORY.items():
        reference = original._reference(path)
        require(reference["sha256"] == pin, "Original UNKNOWN/diagnostic changed")
        references.append(reference)
    assessment = original._read(original.ROOT / "ADMISSION/ASSESSMENT.json")
    diagnostic = original._read(OPS / "T03_VERIFY_DIAGNOSTIC.json")
    require(
        assessment["status"] == "UNKNOWN"
        and assessment["gate_status"] == "NOT_SIGNED"
        and assessment["checks"]["numerical"] == dict.fromkeys(("b0", "b1", "b2"), "INCOMPLETE")
        and all(assessment[k] is False for k in PERMISSIONS),
        "Retain actual unresolved original admission",
    )
    require(
        diagnostic["report"]["status"] == "TECHNICAL_INVALID"
        and diagnostic["report"]["errors"]
        == [
            dict(task_id=f"LH-SRF6-T03-233723-{p}", error="KeyError: 'row_id'")
            for p in ("b0", "b1", "b2")
        ],
        "Only the diagnosed original reader defect is in scope",
    )
    return references


def proof():
    from . import admission_projection, projection

    return dict(
        numerical=projection.verify_projection(),
        admission=admission_projection.verify_projection(),
    )


def contract():
    return dict(
        schema=SCHEMA,
        correction="TRACE_ROW_ID_FROM_SCORED_RECORD_RAW_TOKENS_UNCHANGED",
        forbidden=["resampling", "numerical_replay", "review_rerun", "old_receipt_reseal"],
        unchanged="All original source, evidence, selection, numerical and scientific predicates",
        history=history(),
        projection=proof(),
        **PERMISSIONS,
    )


def register(revision):
    original._path(ROOT, exists=False)
    require(not ROOT.exists(), "Preserve existing or partial repair registration")
    context = original_context()
    code = deployment(revision)
    files = [original._inventory(p) for p in source_paths(code)]
    frozen_contract = contract()
    module_origins(code, files)
    manifest = dict(schema=SCHEMA, revision=revision, deployment=str(code), files=files)
    ROOT.mkdir(mode=0o700)
    original._write(ROOT / "SOURCE_MANIFEST.json", manifest)
    original._write(ROOT / "CONTRACT.json", frozen_contract)
    original._write(
        ROOT / "REGISTRATION.json",
        dict(
            schema=SCHEMA,
            revision=revision,
            deployment=str(code),
            original_execution_consumer=context["consumer"],
            source_manifest=original._reference(ROOT / "SOURCE_MANIFEST.json"),
            contract=original._reference(ROOT / "CONTRACT.json"),
            status="BOUND_NOT_AUTHORIZED",
            **PERMISSIONS,
        ),
    )
    return verify_registered()


def verify_registered():
    context = original_context()
    registration = original._read(original._path(ROOT / "REGISTRATION.json"))
    code = deployment(registration["revision"])
    require(
        registration["schema"] == SCHEMA
        and registration["status"] == "BOUND_NOT_AUTHORIZED"
        and registration["deployment"] == str(code)
        and registration["original_execution_consumer"] == context["consumer"]
        and all(registration[k] is False for k in PERMISSIONS),
        "Technical registration changed or granted unauthorized execution",
    )
    require(
        registration["source_manifest"]["path"] == str(ROOT / "SOURCE_MANIFEST.json")
        and registration["contract"]["path"] == str(ROOT / "CONTRACT.json"),
        "Only this repair's actual bound records are allowed",
    )
    manifest = original._bound(registration["source_manifest"])
    files = manifest["files"]
    require(
        manifest["revision"] == registration["revision"]
        and manifest["deployment"] == str(code)
        and {p["path"] for p in files} == {str(p) for p in source_paths(code)}
        and len(files) == len({p["path"] for p in files}),
        "Repair source membership changed",
    )
    for item in files:
        require(
            original._state(item["path"]) == {k: v for k, v in item.items() if k != "sha256"},
            "Repair source state changed",
        )
        key = original._digest(item)
        if key not in _VERIFIED:
            require(original._hash(item["path"]) == item["sha256"], "Repair source bytes changed")
            _VERIFIED.add(key)
    module_origins(code, files)
    require(original._bound(registration["contract"]) == contract(), "Repair contract drift")
    module_origins(code, files)
    return dict(
        original_execution_consumer=context["consumer"],
        repair_verifier=dict(
            revision=registration["revision"],
            deployment=str(code),
            registration=original._reference(ROOT / "REGISTRATION.json"),
            source_manifest=registration["source_manifest"],
            contract=registration["contract"],
        ),
    )
