"""Independent closed V3 admission over actual confirmation, overlay and plan.

Registration/source/confirmation/kernel records individually grant no model
access. Only this complete assessment can seal the fixed V3 pilot scope. The old
V1 receipt and followup training=false semantics remain unchanged.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
from pathlib import Path

from . import historical_closure as historical
from . import phases, pilot_contract
from .common import digest, file_hash
from .historical_pilot import _json, _write

SCHEMA = "srf6-pilot-v3-admission-v1"
CONTRACT = pilot_contract.for_phase("PILOT_V3")
PATH = CONTRACT.admission_path
SPEC_PATH = CONTRACT.registration_directory + "/" + CONTRACT.spec_file
SCOPE = CONTRACT.admission_scope
_CACHE = {}


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _phase(root, pin):
    phases._sha(pin)
    context = phases.verify_phase(
        root, "PILOT_V3", deployment=historical._actual_deployment(), expected_identity_sha256=pin
    )
    return historical._phase(root, context)


def _model_context(root, registered):
    """Reuse the original content inventory and actual confirmation model pins."""
    from .source_binding import content_inventory, verify_bound_files

    spec = registered["spec"]
    path = phases._file(root, "SOURCE_BINDING.json")
    _require(file_hash(path) == spec["source_binding"]["sha256"], "Original model binding changed")
    binding = _json(path.read_bytes())
    config = _json(phases._file(root, "CONFIG.json").read_bytes())
    facts = registered["confirmation_binding"]["verification"]["facts"]
    verified = {ref["path"]: ref["sha256"] for ref in facts["inputs"]}
    rows = list(binding["model_files"])
    _require(
        digest(content_inventory(rows)) == binding["model_content_sha256"],
        "Original model inventory differs",
    )
    for parent in ("b0", "b1", "b2"):
        item = binding["parents"][parent]
        _require(
            item["path"] == config["model"]["parents"][parent]
            and digest(content_inventory(item["files"])) == item["content_sha256"],
            "Original parent inventory differs",
        )
        rows.extend(item["files"])
    refs = [dict(path=row["path"], sha256=row["sha256"]) for row in rows]
    _require(
        all(verified.get(ref["path"]) == ref["sha256"] for ref in refs),
        "Current model/parent files differ from actual original confirmation",
    )
    # This existing verifier reuses frozen content/stat bindings; it does not
    # rescan unchanged weight payloads. Historical proof is checked first.
    verify_bound_files(rows, boundary=phases.PROJECT)
    versions = binding["versions"]
    _require(
        all(
            value and importlib.metadata.version(name) == value for name, value in versions.items()
        ),
        "Current numerical library environment differs from original binding",
    )
    return dict(
        inputs=refs,
        versions=versions,
        model_path=config["model"]["model_path"],
        parent_tensors={p: spec["parent_tensors"][p]["tensor_sha256"] for p in ("b0", "b1", "b2")},
    )


def assess_pilot_v3_admission(root, *, phase_identity_sha256, pilot_spec_sha256):
    """Reconstruct all facts; this read-only result never authorizes training."""
    from . import pilot_v3_kernel_binding

    request = dict(phase_identity_sha256=phase_identity_sha256, pilot_spec_sha256=pilot_spec_sha256)
    context = registered = kernel = model = None
    checks = {}
    try:
        root = phases._root(root)
        context = _phase(root, phase_identity_sha256)
        checks["phase"] = "VERIFIED_CURRENT_V3_SOURCE"
        registered = pilot_contract.verify_registration(
            root, phase_context=context, pilot_spec_sha256=pilot_spec_sha256
        )
        _require(
            registered["status"] == "VERIFIED_REGISTRATION_NOT_AUTHORIZED"
            and registered["training_authorized"] is False
            and registered["generation_authorized"] is False,
            "Complete independently verified V3 registration required",
        )
        checks["registration"] = "VERIFIED_COMPLETE_FIXED_PLAN"
        spec = registered["spec"]
        _require(
            spec["phase"] == "PILOT_V3"
            and spec["input_overlay"] == registered["overlay"]["manifest_ref"]
            and spec["confirmation_binding_ref"]
            == registered["confirmation_binding"]["binding_ref"],
            "Registration overlay/confirmation references differ",
        )
        _require(
            spec["budgets"]
            == dict(
                primary_paths=9,
                updates_per_path=12,
                primary_logical_updates=108,
                responses_per_update=128,
                primary_responses=13824,
                recovery_new_responses=1024,
                dev_eval_responses=2304,
            ),
            "Fixed pilot budget changed",
        )
        _require(not historical._synthetic(registered), "Synthetic evidence cannot authorize V3")
        checks["overlay"] = "VERIFIED_384_DEV_PILOT_96_DEV_EVAL"
        checks["confirmation"] = "ACTUAL_FROZEN_CONFIRMATION_RECONSTRUCTED"
        kernel = pilot_v3_kernel_binding.verify_kernel_compatibility(
            root, phase_context=context, confirmation_binding=registered["confirmation_binding"]
        )
        _require(
            kernel["status"] == pilot_v3_kernel_binding.STATUS
            and kernel["new_trainer_numerical_pass"] is False
            and kernel["new_model_calls"] == kernel["optimizer_updates"] == 0,
            "Original T03 implementation correspondence is unproven",
        )
        model = _model_context(root, registered)
        _require(
            model["parent_tensors"] == kernel["original_parent_tensors"],
            "Current pilot parents differ from original T03 tensors",
        )
        checks["model_environment"] = "ORIGINAL_MODEL_PARENTS_AND_LIBRARY_VERSIONS"
        checks["numerical_correspondence"] = kernel["status"]
        _require(
            _phase(root, phase_identity_sha256) == context, "V3 source changed during assessment"
        )
        status, error = "PASS", None
    except (OSError, ValueError, KeyError, TypeError, ImportError, PermissionError) as exc:
        status = (
            "FAIL"
            if isinstance(exc, PermissionError)
            else "UNKNOWN"
            if isinstance(exc, (OSError, ImportError, KeyError))
            else "FAIL"
        )
        error = f"{type(exc).__name__}: {exc}"
    # Subprocess invocation UUID/timing are proof records, not deterministic gate
    # facts. Bind the original immutable report through confirmation_binding_ref.
    return dict(
        schema=SCHEMA,
        status=status,
        scope=SCOPE,
        execution_origin="ACTUAL_EVIDENCE_VALIDATION",
        request=request,
        checks=checks,
        error=error,
        phase_context=context,
        pilot_spec=None if registered is None else registered["spec"],
        overlay=None if registered is None else registered["overlay"],
        confirmation_binding=None
        if registered is None
        else registered["confirmation_binding"]["binding"],
        kernel_correspondence=kernel,
        model_context=model,
        training_authorized=False,
        generation_authorized=False,
        formal_authorized=False,
        test_authorized=False,
        gate_status="NOT_SIGNED",
    )


def _receipt(root, assessment):
    _require(
        assessment["status"] == "PASS" and not historical._synthetic(assessment),
        "Only complete actual V3 facts can be sealed",
    )
    return dict(
        schema=SCHEMA,
        status="PASS",
        scope=SCOPE,
        root=str(root),
        request=assessment["request"],
        phase_identity_sha256=assessment["phase_context"]["identity_sha256"],
        pilot_spec_sha256=assessment["request"]["pilot_spec_sha256"],
        input_overlay=assessment["overlay"]["manifest_ref"],
        confirmation_binding_ref=assessment["pilot_spec"]["confirmation_binding_ref"],
        kernel_correspondence_sha256=digest(assessment["kernel_correspondence"]),
        assessment_sha256=digest(assessment),
        training_authorized=True,
        generation_authorized=True,
        development_evaluation_authorized=True,
        formal_authorized=False,
        test_authorized=False,
        gate_status="SIGNED_FROM_RECOMPUTED_FACTS",
        execution_origin="ACTUAL_EVIDENCE_VALIDATION",
    )


def _authorized(receipt, reference, assessment):
    overlay = assessment["overlay"]
    # The overlay manifest owns the version. It is checked again against actual
    # runtime identity by access.py, never accepted from a caller identity dict.
    version = overlay["manifest"]["prompt_version"]
    return dict(
        status="VERIFIED_PILOT_ADMISSION",
        scope=SCOPE,
        training_authorized=True,
        generation_authorized=True,
        formal_authorized=False,
        test_authorized=False,
        receipt=receipt,
        admission_ref=reference,
        phase_context=assessment["phase_context"],
        pilot_spec=assessment["pilot_spec"],
        pilot_spec_ref=dict(path=SPEC_PATH, sha256=assessment["request"]["pilot_spec_sha256"]),
        overlay=overlay,
        input_overlay=overlay["manifest_ref"],
        input_overlay_sha256=overlay["manifest_ref"]["sha256"],
        prompt_version=version,
    )


def _references(root, assessment):
    """Collect actual verified source/data/plan leaves, never sweep model weights."""
    paths = {str(phases._file(root, SPEC_PATH))}
    directory = root / CONTRACT.registration_directory
    context = assessment["phase_context"]

    def walk(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                candidate = Path(value["path"])
                if candidate.is_absolute():
                    paths.add(str(candidate))
                else:
                    options = [p for p in (root / candidate, directory / candidate) if p.is_file()]
                    _require(
                        len(options) == 1, "Ambiguous or missing verified admission dependency"
                    )
                    paths.add(str(options[0]))
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(assessment)
    for relative in (
        "CONFIG.json",
        "IDENTITY.json",
        "SOURCE_BINDING.json",
        "IMPLEMENTATION_REGISTRATION.json",
    ):
        paths.add(str(phases._file(root, relative)))
    for reference, base in (
        (context["identity"]["source_manifest"], Path(context["deployment"])),
        (context["identity"]["data_manifest"], root),
        (
            context["identity"]["origin"]["source_manifest"],
            Path(context["identity"]["origin"]["deployment"]),
        ),
    ):
        manifest = _json(phases._file(root, reference["path"]).read_bytes())
        paths.add(str(root / reference["path"]))
        for row in manifest["files"]:
            path = Path(row["path"])
            paths.add(str(path if path.is_absolute() else base / path))
    for item in assessment["pilot_spec"]["paths"]:
        run = directory / item["run_directory"]
        for name in ("RUN_IDENTITY.json", "ROLLOUT_REGISTRATION.json"):
            paths.add(str(phases._path(run / name)))
    return paths


def _membership(root, assessment):
    """Immutable plan/source/input scopes only; training outputs may accumulate."""
    result = {}
    for name in ("DATA", "PHASES/PILOT_V3", "PHASE_RUNS/PILOT_V3/INPUTS_V3"):
        base = phases._path(root / name)
        result[name] = sorted(
            str(phases._path(p).relative_to(root))
            for p in base.rglob("*")
            if "ADMISSION_ASSESSMENTS" not in p.parts
        )
    model = assessment.get("model_context")
    if model is not None:
        base = Path(model["model_path"])
        result["model_membership"] = sorted(str(p) for p in base.rglob("*") if p.is_file())
    directory = root / CONTRACT.registration_directory
    result["registration_layout"] = sorted(
        str(phases._path(p).relative_to(directory))
        for pattern in ("*", "runs/*", "runs/*/*", "runs/*/*/*")
        for p in directory.glob(pattern)
    )
    deployment = Path(assessment["phase_context"]["deployment"])
    for name in ("src", "tests/sr_f6", "scripts/sr_f6"):
        base = phases._path(deployment / name, exists=False)
        result[str(base)] = sorted(
            str(phases._path(p)) for p in base.rglob("*") if p.suffix in (".py", ".sh", ".sbatch")
        )
    return result


def _remember(root, result, assessment):
    _require(
        result == _authorized(_receipt(root, assessment), result["admission_ref"], assessment),
        "Only recomputed actual V3 permission may populate process cache",
    )
    paths = _references(root, assessment) | {str(root / PATH)}
    key = (
        os.getpid(),
        str(root),
        assessment["request"]["phase_identity_sha256"],
        assessment["request"]["pilot_spec_sha256"],
        result["admission_ref"]["sha256"],
    )
    _CACHE[key] = dict(
        result=copy.deepcopy(result),
        assessment=copy.deepcopy(assessment),
        states={p: historical._state(p) for p in sorted(paths)},
        membership=_membership(root, assessment),
    )


def seal_pilot_v3_admission(root, *, phase_identity_sha256, pilot_spec_sha256):
    root = phases._root(root)
    target = phases._path(root / PATH, exists=False)
    _require(not target.exists(), "Preserve existing or partial V3 admission")
    assessment = assess_pilot_v3_admission(
        root, phase_identity_sha256=phase_identity_sha256, pilot_spec_sha256=pilot_spec_sha256
    )
    parent = phases._path(root / "PHASES/PILOT_V3/ADMISSION_ASSESSMENTS", exists=False)
    parent.mkdir(mode=0o700, exist_ok=True)
    output = parent / (digest(assessment) + ".json")
    if not output.exists():
        _write(output, assessment)
    if assessment["status"] != "PASS":
        return dict(
            status=assessment["status"],
            scope=SCOPE,
            training_authorized=False,
            generation_authorized=False,
            formal_authorized=False,
            test_authorized=False,
            assessment=assessment,
            assessment_ref=dict(path=str(output.relative_to(root)), sha256=file_hash(output)),
        )
    receipt = _receipt(root, assessment)
    # Check the complete cache closure before publishing an authorization.
    _references(root, assessment)
    _write(target, receipt)
    result = _authorized(receipt, dict(path=PATH, sha256=file_hash(target)), assessment)
    _remember(root, result, assessment)
    return result


def require_pilot_v3_admission(root, *, phase_identity_sha256, pilot_spec_sha256, admission_ref):
    root = phases._root(root)
    _require(
        set(admission_ref) == {"path", "sha256"} and admission_ref["path"] == PATH,
        "Only fixed PILOT_V3 admission can authorize this runtime",
    )
    phases._sha(admission_ref["sha256"])
    key = (
        os.getpid(),
        str(root),
        phase_identity_sha256,
        pilot_spec_sha256,
        admission_ref["sha256"],
    )
    cached = _CACHE.get(key)
    if cached is not None:
        assessment = cached["assessment"]
        _require(
            _phase(root, phase_identity_sha256) == assessment["phase_context"],
            "Actual V3 phase changed",
        )
        _require(
            _membership(root, assessment) == cached["membership"],
            "Admission evidence membership changed",
        )
        _require(
            all(historical._state(p) == state for p, state in cached["states"].items()),
            "Admission dependency changed; cached permission invalid",
        )
        historical.require_confirmation_binding(
            root,
            phase_context=assessment["phase_context"],
            confirmation_binding_ref=assessment["pilot_spec"]["confirmation_binding_ref"],
        )
        return copy.deepcopy(cached["result"])
    path = phases._file(root, PATH)
    _require(file_hash(path) == admission_ref["sha256"], "V3 admission content pin differs")
    receipt = _json(path.read_bytes())
    _require(
        receipt.get("schema") == SCHEMA
        and receipt.get("scope") == SCOPE
        and receipt.get("request")
        == dict(phase_identity_sha256=phase_identity_sha256, pilot_spec_sha256=pilot_spec_sha256),
        "V3 phase/spec/scope pins differ",
    )
    assessment = assess_pilot_v3_admission(
        root, phase_identity_sha256=phase_identity_sha256, pilot_spec_sha256=pilot_spec_sha256
    )
    _require(receipt == _receipt(root, assessment), "Stale or handwritten V3 admission")
    result = _authorized(receipt, admission_ref, assessment)
    _remember(root, result, assessment)
    return result


assess = assess_pilot_v3_admission
seal = seal_pilot_v3_admission
require = require_pilot_v3_admission
