"""Read-only S2 consumer of the historical PILOT verifier, never authorization.

Call only at a phase boundary, not per training step/path: every call starts one
cold isolated process and replays the original evidence. There is deliberately
no cross-call fact cache and no trusted persisted PASS. The historical verifier
keeps its own admission/source metadata caches within that process. This module
does not load a model, generate, update, retry, or publish historical FACTS.json.
Attempts and failures belong to the current FORMAL consumer, never to old S1.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import phases
from .common import digest, file_hash

FORMAT = "srf6-historical-pilot-verification-v1"
DIRECTORY = "PHASE_RUNS/FORMAL_V1/HISTORICAL_PILOT_AUDITS"
SPEC_PATH = "PHASE_RUNS/PILOT_V1/PILOT_FROZEN/PILOT_SPEC_FROZEN.json"
RECOVERY = "PHASE_RUNS/PILOT_V1/RECOVERY/b0/r0/U11"
ARGUMENTS = {
    "phase_identity_sha256",
    "pilot_spec_sha256",
    "admission_ref",
    "recovery_binding_ref",
    "recovery_comparison_ref",
    "evaluation_summary_ref",
}
STATUSES = {
    "COMPLETE_FACTS_NOT_GATE_SIGNED",
    "INSUFFICIENT_REGISTERED_EXPOSURE",
    "RECOVERY_MISMATCH",
    "RECOVERY_COMPARISON_HISTORY_UNKNOWN",
}

# A fixed transport shim, bound as part of this S2 file. It imports only the
# already-verified historical source; it never imports this consumer in S1.
BOOTSTRAP = r"""
import hashlib, json, pathlib, re, sys
def require(value, message):
    if not value:
        raise PermissionError(message)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
raw = sys.stdin.buffer.read()
request = json.loads(raw)
require(set(request) == {"root", "deployment", "nonce", "arguments"},
        "Exact transport request required")
args = request["arguments"]
require(set(args) == {"phase_identity_sha256", "pilot_spec_sha256", "admission_ref",
        "recovery_binding_ref", "recovery_comparison_ref", "evaluation_summary_ref"},
        "Exact historical arguments required")
for key in ("phase_identity_sha256", "pilot_spec_sha256"):
    require(isinstance(args[key], str) and re.fullmatch(r"[0-9a-f]{64}", args[key]),
            "Full historical pin required")
recovery = "PHASE_RUNS/PILOT_V1/RECOVERY/b0/r0/U11"
expected = {"admission_ref": "PHASES/PILOT_V1/ADMISSION.json",
            "recovery_binding_ref": recovery + "/RECOVERY_BINDING.json",
            "evaluation_summary_ref": "PHASE_RUNS/PILOT_V1/EVALUATION/SUMMARY.json"}
for key in (*expected, "recovery_comparison_ref"):
    ref = args[key]
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact ref required")
    require(isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Full reference pin required")
    require(ref["path"] == expected[key] if key in expected else
            re.fullmatch(re.escape(recovery) + r"/COMPARISONS/[0-9a-f]{32}\.json", ref["path"]),
            "Wrong historical reference path")
require(not any(n == "sr_f6" or n.startswith("sr_f6.") for n in sys.modules),
        "Historical package imported before isolation")
deployment = pathlib.Path(request["deployment"])
sys.path.insert(0, str(deployment / "src"))
from sr_f6 import phases, pilot_audit, admission
from sr_f6.runtime import _verify_module_origins
root = pathlib.Path(request["root"])
def source():
    context = phases.verify_phase(root, "PILOT_V1", deployment=deployment,
                expected_identity_sha256=args["phase_identity_sha256"])
    require(pathlib.Path(pilot_audit.__file__).resolve() ==
            deployment / "src/sr_f6/pilot_audit.py", "Wrong historical verifier")
    require(pathlib.Path(admission.__file__).resolve() ==
            deployment / "src/sr_f6/admission.py", "Wrong historical admission")
    require(pathlib.Path(pilot_audit.verify_runtime_pilot.__code__.co_filename).resolve()
            == pathlib.Path(pilot_audit.__file__).resolve(), "Historical callable replaced")
    manifest = json.loads((root / context["identity"]["source_manifest"]["path"]).read_bytes())
    _verify_module_origins(deployment, manifest, tuple(sys.modules.values()))
    return context
before = source()
result = pilot_audit.verify_runtime_pilot(root, **request["arguments"])
after = source()
require(after == before, "Historical source changed during verification")
response = dict(nonce=request["nonce"], request_sha256=hashlib.sha256(raw).hexdigest(),
                producer_context=before,
                verifier_file=str(pathlib.Path(pilot_audit.__file__).resolve()),
                admission_file=str(pathlib.Path(admission.__file__).resolve()),
                result=result, result_sha256=hashlib.sha256(canonical(result)).hexdigest())
sys.stdout.write(canonical(response).decode() + "\n")
"""


class HistoricalPilotVerificationError(PermissionError):
    """The preserved attempt did not yield verified historical facts."""


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise PermissionError("Nonfinite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _payload(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()


def _actual_deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _loaded(deployment, manifest):
    members = {item["path"] for item in manifest["files"]}
    packages = {Path(p).parts[1].removesuffix(".py") for p in members if p.startswith("src/")}
    for module in tuple(sys.modules.values()):
        if getattr(module, "__name__", "").split(".")[0] not in packages:
            continue
        name = getattr(module, "__file__", None)
        if name is None:
            continue
        path = Path(name).resolve(strict=True)
        _require(
            path.is_relative_to(deployment) and str(path.relative_to(deployment)) in members,
            "Current consumer imported source outside its FORMAL deployment",
        )


def _contexts(root, formal_pin, pilot_pin):
    phases._sha(formal_pin)
    phases._sha(pilot_pin)
    deployment = _actual_deployment()
    consumer = phases.verify_phase(
        root, "FORMAL_V1", deployment=deployment, expected_identity_sha256=formal_pin
    )
    _require(
        consumer["status"] == "BOUND_NOT_AUTHORIZED" and consumer["training_allowed"] is False,
        "FORMAL provenance is not scientific admission",
    )
    identity = consumer["identity"]
    upstream = [item for item in identity["upstream"] if item["producer_phase"] == "PILOT_V1"]
    _require(
        len(upstream) == 1 and upstream[0]["identity_sha256"] == pilot_pin,
        "Historical producer must be the explicitly bound FORMAL upstream",
    )
    historical = _json(phases._file(root, "PHASES/PILOT_V1/IDENTITY.json").read_bytes())
    producer = phases.verify_phase(
        root, "PILOT_V1", deployment=historical["deployment"], expected_identity_sha256=pilot_pin
    )
    _require(
        producer["deployment"] != consumer["deployment"]
        and producer["identity"]["source_manifest"]["sha256"]
        != identity["source_manifest"]["sha256"]
        and producer["identity"]["origin"] == identity["origin"]
        and producer["identity"]["data_manifest"] == identity["data_manifest"]
        and producer["identity"]["config_sha256"] == identity["config_sha256"],
        "S2 and historical PILOT must retain distinct sources and the same original data",
    )
    _loaded(deployment, _json(phases._file(root, identity["source_manifest"]["path"]).read_bytes()))
    return consumer, producer


def _reference(root, reference, expected):
    _require(
        isinstance(reference, dict) and set(reference) == {"path", "sha256"},
        "Exact evidence reference required",
    )
    phases._sha(reference["sha256"])
    _require(reference["path"] == expected, "Unregistered historical evidence path")
    path = phases._file(root, expected)
    _require(file_hash(path) == reference["sha256"], "Historical request content pin differs")
    return path


def _request(root, arguments):
    _require(set(arguments) == ARGUMENTS, "Exact historical verifier request required")
    phases._sha(arguments["phase_identity_sha256"])
    phases._sha(arguments["pilot_spec_sha256"])
    _reference(root, dict(path=SPEC_PATH, sha256=arguments["pilot_spec_sha256"]), SPEC_PATH)
    expected = {
        "admission_ref": "PHASES/PILOT_V1/ADMISSION.json",
        "recovery_binding_ref": RECOVERY + "/RECOVERY_BINDING.json",
        "evaluation_summary_ref": "PHASE_RUNS/PILOT_V1/EVALUATION/SUMMARY.json",
    }
    comparison = arguments["recovery_comparison_ref"]
    _require(
        isinstance(comparison, dict)
        and isinstance(comparison.get("path"), str)
        and re.fullmatch(
            re.escape(RECOVERY) + r"/COMPARISONS/[0-9a-f]{32}\.json", comparison["path"]
        ),
        "Only the registered recovery comparison UUID path is allowed",
    )
    expected["recovery_comparison_ref"] = comparison["path"]
    for key, path in expected.items():
        _reference(root, arguments[key], path)


def _stamp(path):
    value = path.stat()
    return (
        str(path.resolve(strict=True)),
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )


def _snapshot(root, arguments):
    """Membership and metadata only, including states; never hash model payloads."""
    relative = {
        "CONFIG.json",
        "IDENTITY.json",
        "SOURCE_BINDING.json",
        "SOURCE_MANIFEST.json",
        "IMPLEMENTATION_REGISTRATION.json",
    }
    for name in (
        "DATA",
        "CENSUS",
        "PREFLIGHT",
        "runtime_identity",
        "PHASES/PILOT_V1",
        "PHASES/CENSUS_REVIEW_V1",
        "PHASE_RUNS/PILOT_V1",
    ):
        folder = phases._path(root / name, exists=False)
        if folder.exists():
            relative.add(name)
            for path in folder.rglob("*"):
                phases._path(path)
                relative.add(path.relative_to(root).as_posix())
    # Admission's recorded dependency list broadens the snapshot; it is not
    # trusted as proof. The historical cold verifier independently rebuilds it.
    receipt = _json(phases._file(root, arguments["admission_ref"]["path"]).read_bytes())
    for reference in receipt.get("inputs", []):
        relative.add(phases._relative(reference["path"]).as_posix())
    result = {str(phases._path(root / name)): _stamp(root / name) for name in sorted(relative)}
    for item in receipt.get("external_inputs", []):
        path = Path(item["path"])
        _require(
            path.is_absolute() and path.resolve(strict=True).is_relative_to(phases.PROJECT),
            "External dependency outside canonical project",
        )
        result[str(path)] = _stamp(path)
    return result


def _write(path, value):
    path = phases._path(path, exists=False)
    with path.open("xb") as stream:
        stream.write(_payload(value))
        stream.flush()
        os.fsync(stream.fileno())


def _ref(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=file_hash(path))


def _references(root, items):
    """Expose original verified leaf references without rehashing their payloads."""
    _require(isinstance(items, list), "Explicit original historical input references required")
    unique = {}
    for item in items:
        _require(
            isinstance(item, dict) and set(item) == {"path", "sha256"},
            "Exact historical input reference required",
        )
        phases._sha(item["sha256"])
        phases._file(root, item["path"])
        _require(
            unique.setdefault(item["path"], item["sha256"]) == item["sha256"],
            "Conflicting historical input pins",
        )
    return [dict(path=path, sha256=sha) for path, sha in sorted(unique.items())]


def _result(envelope, transport, producer):
    _require(
        isinstance(envelope, dict)
        and set(envelope)
        == {
            "nonce",
            "request_sha256",
            "producer_context",
            "verifier_file",
            "admission_file",
            "result",
            "result_sha256",
        },
        "Exact child response required",
    )
    _require(
        envelope["nonce"] == transport["nonce"]
        and envelope["request_sha256"] == hashlib.sha256(_payload(transport)).hexdigest(),
        "Child response/request binding differs",
    )
    deployment = Path(producer["deployment"])
    _require(
        envelope["producer_context"] == producer
        and envelope["verifier_file"] == str(deployment / "src/sr_f6/pilot_audit.py")
        and envelope["admission_file"] == str(deployment / "src/sr_f6/admission.py"),
        "Child verifier producer differs",
    )
    facts = envelope["result"]
    _require(
        isinstance(facts, dict) and digest(facts) == envelope["result_sha256"],
        "Child facts digest differs",
    )
    _require(
        facts.get("format") == "srf6-runtime-pilot-audit-v1"
        and facts.get("scope") == "SERVER_ACTUAL"
        and facts.get("status") in STATUSES
        and facts.get("phase_identity_sha256") == transport["arguments"]["phase_identity_sha256"]
        and facts.get("pilot_spec_sha256") == transport["arguments"]["pilot_spec_sha256"]
        and facts.get("gate_status") == "NOT_SIGNED"
        and facts.get("training_authorized") is False
        and facts.get("formal_authorized") is False,
        "Historical facts must be an actual verifier result, never a PASS or authorization",
    )
    for key, expected in (
        ("paths", 9),
        ("logical_updates", 108),
        ("primary_responses", 13824),
        ("primary_actual_generated_responses", 13824),
        ("recovery_responses", 1024),
        ("evaluation_responses", 2304),
    ):
        _require(
            type(facts.get(key)) is int and facts[key] == expected,
            "Historical registered budget differs",
        )
    _require(
        isinstance(facts.get("runs"), list)
        and len(facts["runs"]) == 9
        and {(r.get("parent"), r.get("arm")) for r in facts["runs"]}
        == {(p, a) for p in ("b0", "b1", "b2") for a in ("U10", "U11", "C_SHUF")},
        "Complete original pilot run facts required",
    )

    def synthetic(value):
        if isinstance(value, dict):
            return any(
                (
                    k in ("scope", "execution_origin", "status")
                    and isinstance(v, str)
                    and v.startswith("SYNTHETIC")
                )
                or synthetic(v)
                for k, v in value.items()
            )
        return isinstance(value, list) and any(synthetic(v) for v in value)

    _require(not synthetic(facts), "Synthetic evidence cannot become historical scientific facts")
    return facts


def verify_historical_pilot(
    root,
    *,
    formal_phase_identity_sha256,
    pilot_phase_identity_sha256,
    pilot_spec_sha256,
    admission_ref,
    recovery_binding_ref,
    recovery_comparison_ref,
    evaluation_summary_ref,
):
    """Recompute historical facts once in an isolated process; always unauthorized.

    Successful transport preserves every original negative/UNKNOWN result.
    Invalid requests fail before launch. Every launched attempt retains request,
    command, stdout/stderr and RESULT or FAILURE under the S2 consumer directory.
    No saved FACTS/status is an input, no subprocess retry or cross-call cache.
    """
    root = phases._root(root)
    arguments = dict(
        phase_identity_sha256=pilot_phase_identity_sha256,
        pilot_spec_sha256=pilot_spec_sha256,
        admission_ref=admission_ref,
        recovery_binding_ref=recovery_binding_ref,
        recovery_comparison_ref=recovery_comparison_ref,
        evaluation_summary_ref=evaluation_summary_ref,
    )
    _request(root, arguments)
    consumer, producer = _contexts(root, formal_phase_identity_sha256, pilot_phase_identity_sha256)
    before = _snapshot(root, arguments)
    folder = phases._path(root / DIRECTORY, exists=False)
    folder.mkdir(parents=True, exist_ok=True)
    attempt = phases._path(folder / uuid.uuid4().hex, exists=False)
    attempt.mkdir()
    transport = dict(
        root=str(root), deployment=producer["deployment"], nonce=attempt.name, arguments=arguments
    )
    command = [sys.executable, "-I", "-B", "-c", BOOTSTRAP]
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    (attempt / "tmp").mkdir()
    (attempt / "cache").mkdir()
    environment.update(
        TMPDIR=str(attempt / "tmp"),
        TMP=str(attempt / "tmp"),
        TEMP=str(attempt / "tmp"),
        XDG_CACHE_HOME=str(attempt / "cache"),
        HF_HOME=str(attempt / "cache/huggingface"),
        TORCH_HOME=str(attempt / "cache/torch"),
    )
    for name, child in {
        "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "TRANSFORMERS_CACHE": "huggingface/hub",
        "HF_MODULES_CACHE": "huggingface/modules",
        "HF_DATASETS_CACHE": "huggingface/datasets",
        "HF_ASSETS_CACHE": "huggingface/assets",
        "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        environment[name] = str(attempt / "cache" / child)
    execution = dict(
        command=command,
        cwd=producer["deployment"],
        bootstrap_sha256=hashlib.sha256(BOOTSTRAP.encode()).hexdigest(),
        python=str(Path(sys.executable).resolve(strict=True)),
        cuda_visible_devices="",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    _write(
        attempt / "REQUEST.json",
        dict(
            format=FORMAT,
            context=consumer["identity"],
            producer_context=producer["identity"],
            transport=transport,
            execution=execution,
        ),
    )
    process = None
    try:
        with (
            (attempt / "stdout.json").open("xb") as stdout,
            (attempt / "stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=producer["deployment"],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
            )
            process.communicate(_payload(transport))
        _require(process.returncode == 0, f"Historical verifier exited {process.returncode}")
        envelope = _json((attempt / "stdout.json").read_bytes())
        facts = _result(envelope, transport, producer)
        _request(root, arguments)
        _require(
            _contexts(root, formal_phase_identity_sha256, pilot_phase_identity_sha256)
            == (consumer, producer),
            "Sources changed during historical verification",
        )
        _require(
            _snapshot(root, arguments) == before, "Historical evidence changed during verification"
        )
        historical_inputs = _references(root, facts.get("inputs"))
        _require(bool(historical_inputs), "Historical fact input closure is missing")
        report = dict(
            format=FORMAT,
            status="FACTS_RECOMPUTED_NOT_AUTHORIZED",
            context=consumer["identity"],
            producer_context=producer["identity"],
            phase_identity_sha256=formal_phase_identity_sha256,
            request=arguments,
            execution={
                **execution,
                "returncode": process.returncode,
                "child_pid": process.pid,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            historical_result=facts,
            historical_result_sha256=envelope["result_sha256"],
            inputs=_references(
                root,
                [
                    dict(path=SPEC_PATH, sha256=pilot_spec_sha256),
                    admission_ref,
                    recovery_binding_ref,
                    recovery_comparison_ref,
                    evaluation_summary_ref,
                    _ref(root, attempt / "REQUEST.json"),
                    _ref(root, attempt / "stdout.json"),
                    _ref(root, attempt / "stderr.log"),
                    *historical_inputs,
                ],
            ),
            gate_status="NOT_SIGNED",
            training_authorized=False,
            generation_authorized=False,
            evaluation_authorized=False,
            formal_authorized=False,
            limitation=(
                "Historical PILOT facts only; separate formal evaluation plans, access locks "
                "and admission remain required"
            ),
        )
        _write(attempt / "RESULT.json", report)
        return {**report, "report_ref": _ref(root, attempt / "RESULT.json")}
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        _write(
            attempt / "FAILURE.json",
            dict(
                format=FORMAT,
                status="HISTORICAL_VERIFICATION_FAILED_NOT_AUTHORIZED",
                context=consumer["identity"],
                producer_context=producer["identity"],
                error=type(exc).__name__ + ": " + str(exc),
                returncode=None if process is None else process.returncode,
                execution={
                    **execution,
                    "child_pid": None if process is None else process.pid,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                training_authorized=False,
                generation_authorized=False,
                evaluation_authorized=False,
                formal_authorized=False,
            ),
        )
        if not isinstance(exc, Exception):
            raise
        raise HistoricalPilotVerificationError(
            f"Historical verification failed; preserve {attempt}: {exc}"
        ) from exc
