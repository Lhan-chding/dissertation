"""Isolated, read-only consumption of the actual frozen 6e/followup-v2 closure.

A cold boundary reconstructs facts in the old packages' own interpreter. Warm
checks reuse only this process's successful proof and check every dependency's
state and evidence membership. No model payload is rehashed by this bridge.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path

from . import phases
from .common import digest, file_hash, now
from .historical_pilot import _json, _payload, _write

FOLLOWUP_ROOT = phases.PROJECT / "sr_f6_confirm_followup_v2_20261029"
FOLLOWUP_DEPLOYMENT = phases.PROJECT / "sr_f6_followup_code_1e4355c"
FOLLOWUP_REVISION = "1e4355cdf26eb2dfd70bb3e67520981b483f5350"
PRODUCER_ROOT = phases.PROJECT / "sr_f6_confirm_v3_20261028_r1"
PRODUCER_DEPLOYMENT = phases.PROJECT / "sr_f6_code_6e5d363"
HISTORY_ROOT = phases.PROJECT / "sr_f6_confirm_followup_v1_20261029"
HISTORY_DEPLOYMENT = phases.PROJECT / "sr_f6_followup_code_02fbf89"
PRODUCER_REVISION = "6e5d36351b8c1f6f18c61661246191772ba6a6dd"
PRODUCER_REGISTRATION_SHA256 = "483dfbb8efaf569c1a5041cb8f715baf23a46bd8e40e8e6f1f18d357523ccf79"
DIRECTORY = "PHASE_RUNS/PILOT_V3/HISTORICAL_CONFIRMATION_AUDITS"
BINDING_PATH = "PHASES/PILOT_V3/CONFIRMATION_BINDING.json"
SCHEMA = "srf6-historical-confirmation-v1"
BINDING_SCHEMA = "srf6-pilot-v3-confirmation-binding-v1"
_CACHE = {}

# The interpreter is isolated and imports the two immutable namespaces at their
# original paths. It never loads the new sr_f6 package or a model.
BOOTSTRAP = r"""
import hashlib, json, os, pathlib, sys
raw = sys.stdin.buffer.read()
r = json.loads(raw)
def require(ok, message):
    if not ok: raise PermissionError(message)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
require(set(r) == {"nonce", "followup_root", "followup_deployment", "producer_deployment",
                   "registration_ref", "receipt_ref"}, "Exact transport request required")
require(not any(n == "sr_f6" or n.startswith("sr_f6.") or
                n == "sr_f6_followup" or n.startswith("sr_f6_followup.")
                for n in sys.modules), "Packages loaded before isolation")
sys.path[:0] = [str(pathlib.Path(r["producer_deployment"]) / "src"),
                str(pathlib.Path(r["followup_deployment"]) / "src")]
from sr_f6 import confirm_registration
from sr_f6_followup import source, admission, t03
require(pathlib.Path(source.__file__).resolve() ==
        pathlib.Path(r["followup_deployment"]) / "src/sr_f6_followup/source.py", "Wrong consumer")
require(pathlib.Path(confirm_registration.__file__).resolve() ==
        pathlib.Path(r["producer_deployment"]) / "src/sr_f6/confirm_registration.py",
        "Wrong producer")
require(pathlib.Path(admission.require_confirmation.__code__.co_filename).resolve() ==
        pathlib.Path(r["followup_deployment"]) / "src/sr_f6_followup/admission.py",
        "Replaced verifier")
for ref in (r["registration_ref"], r["receipt_ref"]):
    require(source._reference(source._path(ref["path"])) == ref, "Transport pin differs")
receipt = admission.require_confirmation(r["followup_root"])
context = source.require_support(r["followup_root"], purpose="admission")
numerical = t03.verify_completed(r["followup_root"])
review = context["review"]
refs = [dict(path=row["path"], sha256=row["sha256"])
        for row in context["support_binding"]["dependencies"]]
refs += [r["registration_ref"], r["receipt_ref"], receipt["assessment"],
         context["support_binding_ref"]]
for item in [*review["inputs"], *review.get("external_inputs", []), *numerical["inputs"]]:
    p = pathlib.Path(item["path"])
    if not p.is_absolute(): p = pathlib.Path(r["followup_root"]) / p
    refs.append(dict(path=str(p), sha256=item["sha256"]))
unique = {}
for item in refs:
    require(unique.setdefault(item["path"], item) == item, "Conflicting input pins")
# Recheck both namespaces after all lazy evidence imports.
after = source.verify_registered(r["followup_root"], require_support=True)
require(after["consumer"] == context["consumer"], "Consumer changed during verification")
result = dict(receipt=receipt, consumer=context["consumer"],
              producer=context["support_binding"]["producer"],
              support_binding_ref=context["support_binding_ref"],
              support_rows=context["support_binding"]["source_rows"],
              review=review, numerical=numerical,
              repair_history=context["spec"]["repair_history"],
              inputs=[unique[p] for p in sorted(unique)],
              producer_file=str(pathlib.Path(confirm_registration.__file__).resolve()),
              consumer_file=str(pathlib.Path(admission.__file__).resolve()))
sys.stdout.write(canonical(dict(nonce=r["nonce"], request_sha256=hashlib.sha256(raw).hexdigest(),
                     result=result, result_sha256=hashlib.sha256(canonical(result)).hexdigest(),
                     isolated=sys.flags.isolated, child_pid=os.getpid())).decode() + "\n")
"""


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _actual_deployment():
    return Path(__file__).resolve(strict=True).parents[2]


def _phase(root, context):
    from .runtime import _verify_module_origins

    actual = phases.verify_phase(
        root,
        "PILOT_V3",
        deployment=_actual_deployment(),
        expected_identity_sha256=context["identity_sha256"],
    )
    _require(actual == context, "Actual PILOT_V3 source context differs")
    _require(actual["status"] == "BOUND_NOT_AUTHORIZED", "Source binding is not authority")
    manifest = _json(phases._file(root, actual["identity"]["source_manifest"]["path"]).read_bytes())
    _verify_module_origins(Path(actual["deployment"]), manifest, tuple(sys.modules.values()))
    return actual


def _absolute_ref(reference, expected=None):
    _require(
        isinstance(reference, dict) and set(reference) == {"path", "sha256"},
        "Exact absolute reference required",
    )
    phases._sha(reference["sha256"])
    path = phases._path(Path(reference["path"]))
    _require(path.is_file() and (expected is None or path == expected), "Wrong historical path")
    _require(file_hash(path) == reference["sha256"], "Historical content pin differs")
    return _json(path.read_bytes())


def _source_manifest(path, deployment):
    manifest = _json(phases._path(path).read_bytes())
    members = {}
    for row in manifest["files"]:
        name = Path(row["path"])
        name = name if name.is_absolute() else deployment / name
        path = phases._path(name)
        _require(
            path.is_relative_to(deployment) and path.is_file(), "Source member escapes deployment"
        )
        _require(str(path) not in members, "Duplicate source member")
        _require(file_hash(path) == row["sha256"], "Frozen source content differs")
        members[str(path)] = row["sha256"]
    actual = set()
    for directory in (deployment / "src", deployment / "scripts"):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.suffix in (".py", ".sh", ".sbatch"):
                    actual.add(str(phases._path(path)))
    _require(actual <= members.keys(), "Unregistered source member")
    return manifest


def _request(registration_ref, receipt_ref):
    registered = _absolute_ref(registration_ref, FOLLOWUP_ROOT / "REGISTRATION.json")
    _absolute_ref(receipt_ref, FOLLOWUP_ROOT / "ADMISSION/RECEIPT.json")
    _require(
        registered.get("schema") == "srf6-followup-registration-v2"
        and registered.get("revision") == FOLLOWUP_REVISION
        and registered.get("root") == str(FOLLOWUP_ROOT)
        and registered.get("deployment") == str(FOLLOWUP_DEPLOYMENT),
        "Only actual frozen followup-v2 registration is allowed",
    )
    _require(
        {p.name for p in (FOLLOWUP_DEPLOYMENT / "src").iterdir()} == {"sr_f6_followup"},
        "Historical consumer must not shadow sr_f6",
    )
    _absolute_ref(registered["source_manifest"], FOLLOWUP_ROOT / "SOURCE_MANIFEST.json")
    _source_manifest(FOLLOWUP_ROOT / "SOURCE_MANIFEST.json", FOLLOWUP_DEPLOYMENT)
    producer = registered["producer"]
    _require(
        producer.get("root") == str(PRODUCER_ROOT)
        and producer.get("deployment") == str(PRODUCER_DEPLOYMENT)
        and producer.get("revision") == PRODUCER_REVISION
        and producer.get("registration_sha256") == PRODUCER_REGISTRATION_SHA256,
        "Wrong original confirmation producer",
    )
    original = _absolute_ref(
        dict(path=str(PRODUCER_ROOT / "REGISTRATION.json"), sha256=PRODUCER_REGISTRATION_SHA256)
    )
    _require(
        original["source_manifest_sha256"] == producer["source_manifest_sha256"],
        "Original source identity differs",
    )
    _absolute_ref(
        dict(
            path=str(PRODUCER_ROOT / "SOURCE_MANIFEST.json"),
            sha256=producer["source_manifest_sha256"],
        )
    )
    _source_manifest(PRODUCER_ROOT / "SOURCE_MANIFEST.json", PRODUCER_DEPLOYMENT)
    return registered


def _state(path):
    """Track registered read aliases without following them outside the project."""
    path = Path(path)
    _require(
        path.is_absolute() and path.is_relative_to(phases.PROJECT) and ".." not in path.parts,
        "Dependency outside project",
    )
    target = phases._path(path.resolve(strict=True))
    value = target.stat()
    _require(stat.S_ISREG(value.st_mode), "Regular dependency required")

    def stamp(item):
        return [item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns, item.st_mode]

    aliases = [
        (str(p), str(p.readlink()), stamp(p.lstat()))
        for p in (path, *path.parents)
        if p.is_symlink()
    ]
    return dict(
        path=str(path),
        target=str(target),
        aliases=aliases,
        lstat=stamp(path.lstat()),
        stat=[value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode],
    )


def _trees():
    result = {}
    # Model snapshots are not walked. Every bound model leaf is checked by stat.
    for root in (
        FOLLOWUP_ROOT,
        PRODUCER_ROOT,
        FOLLOWUP_DEPLOYMENT,
        PRODUCER_DEPLOYMENT,
        HISTORY_ROOT,
        HISTORY_DEPLOYMENT,
    ):
        names = []
        for directory, children, files in os.walk(phases._path(root), followlinks=False):
            children[:] = sorted(
                n for n in children if n not in {"__pycache__", ".verification-cache", ".git"}
            )
            for name in (*children, *sorted(files)):
                path = phases._path(Path(directory) / name)
                names.append(str(path.relative_to(root)))
        result[str(root)] = sorted(names)
    return result


def _snapshot(inputs=()):
    support = _json(phases._path(FOLLOWUP_ROOT / "SUPPORT_BINDING.json").read_bytes())
    paths = {row["path"] for row in support["dependencies"]}
    paths.update(row["path"] for row in inputs)
    trees = _trees()
    for base in (FOLLOWUP_ROOT, PRODUCER_ROOT, HISTORY_ROOT):
        for name in trees[str(base)]:
            path = base / name
            if path.is_file():
                paths.add(str(path))
    for base in (FOLLOWUP_DEPLOYMENT, PRODUCER_DEPLOYMENT, HISTORY_DEPLOYMENT):
        for name in trees[str(base)]:
            path = base / name
            if path.is_file():
                paths.add(str(path))
    return dict(membership=trees, states={p: _state(p) for p in sorted(paths)})


def _synthetic(value):
    if isinstance(value, dict):
        return any(
            (
                k in ("scope", "execution_origin", "status")
                and isinstance(v, str)
                and v.startswith("SYNTHETIC")
            )
            or _synthetic(v)
            for k, v in value.items()
        )
    return isinstance(value, (list, tuple)) and any(_synthetic(v) for v in value)


def _facts(envelope, request, registered):
    _require(
        set(envelope)
        == {"nonce", "request_sha256", "result", "result_sha256", "isolated", "child_pid"},
        "Exact isolated result required",
    )
    _require(
        envelope["nonce"] == request["nonce"]
        and envelope["request_sha256"] == hashlib.sha256(_payload(request)).hexdigest()
        and envelope["isolated"] == 1
        and type(envelope["child_pid"]) is int,
        "Isolated nonce/request binding differs",
    )
    facts = envelope["result"]
    _require(
        digest(facts) == envelope["result_sha256"] and not _synthetic(facts),
        "Actual unchanged confirmation facts required",
    )
    receipt, review, numerical = facts["receipt"], facts["review"], facts["numerical"]
    _require(
        receipt["status"] == "CONFIRMATION_GATES_PASS"
        and receipt["gate_status"] == "CONFIRMATION_FACTS_SEALED"
        and all(
            receipt[k] is False
            for k in ("training_allowed", "generation_allowed", "evaluation_allowed")
        )
        and receipt["receipt_ref"] == request["receipt_ref"],
        "Sealed actual confirmation required",
    )
    _require(
        facts["producer"] == registered["producer"]
        and facts["consumer"]["revision"] == FOLLOWUP_REVISION
        and facts["consumer"]["deployment"] == str(FOLLOWUP_DEPLOYMENT)
        and facts["consumer"]["registration_sha256"] == request["registration_ref"]["sha256"]
        and facts["consumer"]["source_manifest_sha256"] == registered["source_manifest"]["sha256"],
        "Actual dual producer identities differ",
    )
    _require(
        facts["producer_file"] == str(PRODUCER_DEPLOYMENT / "src/sr_f6/confirm_registration.py")
        and facts["consumer_file"] == str(FOLLOWUP_DEPLOYMENT / "src/sr_f6_followup/admission.py"),
        "Actual verifier module origin differs",
    )
    _require(
        facts["support_rows"] == 9216
        and review["status"] == "REVIEW_COMPLETE_NOT_GATE_SIGNED"
        and review["completed"] is True
        and review["synthetic"] is False
        and type(review["reviewed_original_answers"]) is int
        and review["reviewed_original_answers"] == 360
        and type(review["independently_reviewed_original_answers"]) is int
        and review["independently_reviewed_original_answers"] == 360,
        "Complete support and sealed new review required",
    )
    _require(
        numerical["scope"] == "SERVER_ACTUAL"
        and numerical["status"] == "PASS"
        and set(numerical["parents"]) == {"b0", "b1", "b2"}
        and all(item["status"] == "PASS" for item in numerical["parents"].values())
        and numerical["generated_responses"] == numerical["optimizer_updates"] == 0,
        "All original T03 parents must pass without new responses or updates",
    )
    _require(
        facts["repair_history"]["status"] == "HISTORICAL_NEGATIVE_EVIDENCE_NOT_AUTHORITY"
        and facts["repair_history"]["completed"] is False,
        "Preserved failed repair history required",
    )
    _require(facts["inputs"], "Complete historical input closure required")
    return facts


def verify_followup_confirmation(root, *, followup_registration_ref, confirmation_receipt_ref):
    """One cold isolated attempt, permanently retaining errors and subprocess evidence."""
    root = phases._root(root)
    registered = _request(followup_registration_ref, confirmation_receipt_ref)
    before = _snapshot()
    directory = phases._path(root / DIRECTORY, exists=False)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / uuid.uuid4().hex
    attempt.mkdir()
    request = dict(
        nonce=attempt.name,
        followup_root=str(FOLLOWUP_ROOT),
        followup_deployment=str(FOLLOWUP_DEPLOYMENT),
        producer_deployment=str(PRODUCER_DEPLOYMENT),
        registration_ref=followup_registration_ref,
        receipt_ref=confirmation_receipt_ref,
    )
    command = [sys.executable, "-I", "-B", "-c", BOOTSTRAP]
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    }
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    for name in ("tmp", "cache"):
        (attempt / name).mkdir()
    environment.update(
        TMPDIR=str(attempt / "tmp"),
        XDG_CACHE_HOME=str(attempt / "cache"),
        HF_HOME=str(attempt / "cache/huggingface"),
        TORCH_HOME=str(attempt / "cache/torch"),
    )
    execution = dict(
        command=command,
        cwd=str(FOLLOWUP_DEPLOYMENT),
        started_at=now(),
        python=str(Path(sys.executable).resolve()),
        bootstrap_sha256=hashlib.sha256(BOOTSTRAP.encode()).hexdigest(),
    )
    _write(attempt / "REQUEST.json", dict(schema=SCHEMA, request=request, execution=execution))
    process = None
    try:
        with (
            (attempt / "stdout.json").open("xb") as out,
            (attempt / "stderr.log").open("xb") as err,
        ):
            process = subprocess.Popen(
                command,
                cwd=FOLLOWUP_DEPLOYMENT,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
            )
            process.communicate(_payload(request))
        _require(process.returncode == 0, f"Historical confirmation exited {process.returncode}")
        envelope = _json((attempt / "stdout.json").read_bytes())
        facts = _facts(envelope, request, registered)
        _require(envelope["child_pid"] == process.pid, "Subprocess PID differs")
        _require(
            _request(followup_registration_ref, confirmation_receipt_ref) == registered,
            "Historical registration changed during verification",
        )
        _require(
            _snapshot() == before, "Historical membership or dependency changed during verification"
        )
        report = dict(
            schema=SCHEMA,
            status="CONFIRMATION_RECOMPUTED_NOT_AUTHORIZED",
            facts=facts,
            facts_sha256=digest(facts),
            request=request,
            execution=dict(
                execution, completed_at=now(), child_pid=process.pid, returncode=process.returncode
            ),
            transcript=[
                dict(path=str(p.relative_to(root)), sha256=file_hash(p))
                for p in (attempt / "REQUEST.json", attempt / "stdout.json", attempt / "stderr.log")
            ],
            training_authorized=False,
            generation_authorized=False,
            formal_authorized=False,
        )
        _write(attempt / "RESULT.json", report)
        result = dict(
            report,
            report_ref=dict(
                path=str((attempt / "RESULT.json").relative_to(root)),
                sha256=file_hash(attempt / "RESULT.json"),
            ),
        )
        key = (
            os.getpid(),
            str(root),
            digest(followup_registration_ref),
            digest(confirmation_receipt_ref),
        )
        _CACHE[key] = (copy.deepcopy(result), _snapshot(facts["inputs"]))
        return result
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
                schema=SCHEMA,
                status="UNKNOWN",
                error=f"{type(exc).__name__}: {exc}",
                execution=dict(
                    execution,
                    completed_at=now(),
                    child_pid=None if process is None else process.pid,
                    returncode=None if process is None else process.returncode,
                ),
                training_authorized=False,
                generation_authorized=False,
            ),
        )
        raise


def _binding(context, registration_ref, receipt_ref, verification):
    return dict(
        schema=BINDING_SCHEMA,
        status="CONFIRMATION_BOUND_NOT_PILOT_AUTHORIZED",
        phase_identity_sha256=context["identity_sha256"],
        phase_context=context,
        followup_registration_ref=registration_ref,
        confirmation_receipt_ref=receipt_ref,
        facts_sha256=verification["facts_sha256"],
        original_verification_ref=verification["report_ref"],
        producer=verification["facts"]["producer"],
        consumer=verification["facts"]["consumer"],
        training_authorized=False,
        generation_authorized=False,
        formal_authorized=False,
    )


def freeze_confirmation_binding(
    root, *, phase_context, followup_registration_ref, confirmation_receipt_ref
):
    root = phases._root(root)
    phase_context = _phase(root, phase_context)
    target = phases._path(root / BINDING_PATH, exists=False)
    _require(not target.exists(), "Preserve existing confirmation binding")
    verification = verify_followup_confirmation(
        root,
        followup_registration_ref=followup_registration_ref,
        confirmation_receipt_ref=confirmation_receipt_ref,
    )
    _phase(root, phase_context)
    binding = _binding(
        phase_context, followup_registration_ref, confirmation_receipt_ref, verification
    )
    _write(target, binding)
    return dict(
        binding_ref=dict(path=BINDING_PATH, sha256=file_hash(target)),
        binding=binding,
        verification=verification,
    )


def require_confirmation_binding(root, *, phase_context, confirmation_binding_ref):
    root = phases._root(root)
    phase_context = _phase(root, phase_context)
    _require(
        set(confirmation_binding_ref) == {"path", "sha256"}
        and confirmation_binding_ref["path"] == BINDING_PATH,
        "Fixed V3 confirmation binding required",
    )
    path = phases._file(root, BINDING_PATH)
    _require(
        file_hash(path) == confirmation_binding_ref["sha256"], "V3 confirmation binding changed"
    )
    binding = _json(path.read_bytes())
    registration_ref, receipt_ref = (
        binding["followup_registration_ref"],
        binding["confirmation_receipt_ref"],
    )
    key = (os.getpid(), str(root), digest(registration_ref), digest(receipt_ref))
    cached = _CACHE.get(key)
    if cached is not None:
        verification = copy.deepcopy(cached[0])
        _require(
            _snapshot(verification["facts"]["inputs"]) == cached[1],
            "Historical dependency or complete membership changed; cached proof invalid",
        )
    else:
        verification = verify_followup_confirmation(
            root, followup_registration_ref=registration_ref, confirmation_receipt_ref=receipt_ref
        )
    original_ref = binding["original_verification_ref"]
    original_path = phases._file(root, original_ref["path"])
    _require(
        original_path.parent.parent == root / DIRECTORY
        and original_path.name == "RESULT.json"
        and file_hash(original_path) == original_ref["sha256"],
        "Original actual attempt changed",
    )
    original = _json(original_path.read_bytes())
    _require(
        original["facts_sha256"] == verification["facts_sha256"]
        and original["facts"] == verification["facts"],
        "Historical confirmation facts changed",
    )
    for ref in original["transcript"]:
        _require(
            file_hash(phases._file(root, ref["path"])) == ref["sha256"],
            "Actual transport evidence changed",
        )
    expected = _binding(
        phase_context, registration_ref, receipt_ref, dict(verification, report_ref=original_ref)
    )
    _require(binding == expected, "Handwritten or stale confirmation binding")
    return dict(
        binding_ref=copy.deepcopy(confirmation_binding_ref),
        binding=binding,
        verification=verification,
    )
