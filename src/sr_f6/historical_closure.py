"""Isolated consumption of the separately corrected confirmation evidence.

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
CLOSURE_ROOT = phases.PROJECT / "sr_f6_confirmation_closure_v1_20261029"
CLOSURE_DEPLOYMENT = phases.PROJECT / "sr_f6_closure_code_1879f8c"
CLOSURE_REVISION = "1879f8c8f0eb14a69aa65f4ee46557dd446957f5"
FOLLOWUP_REGISTRATION_SHA256 = "ef7e3ba365d3675995c18f9314a5ea51c95cc4b57a60a2f7d2afe0d0d250b00a"
CLOSURE_PINS = {
    "REGISTRATION.json": "aee0e4d7fc416d9ae8ab0fca4524af8c986187feec49bf87a5867411f21ab39b",
    "RECEIPT.json": "5da462e35256fbfd7b062da118c09099a2c59f06ca859b2ada074459b60fafa3",
    "ASSESSMENT.json": "07a8793e0e2d30eda417d090ddf13c52816ba6ebed20bdd9521fc6b8e1a639ac",
    "ATTEMPT.json": "9ef990b8d16fa409e940c3230710cefdb85c861e632db92961a0f30c6d1d8db8",
    "CONTRACT.json": "946caae220f577b80a6865c574cb0f480ac58a3da2f50382c255656531548033",
    "SOURCE_MANIFEST.json": "91337fab899af4fe718121f484d84751d16f4f7682ccf41aa46c7cda6b649924",
    "NUMERICAL.json": "1e0888f6ceb58c4a46a0e8ae7970ad6fe2a516d4036d54ab0d22df00aa22f77b",
    "ORIGINAL_RULE_ASSESSMENT.json": (
        "679d5ee3eb689e55e5a45700160d59272a00f1138184b7214109c51a29d94d6f"
    ),
}
DIRECTORY = "PHASE_RUNS/PILOT_V3/HISTORICAL_CORRECTED_CONFIRMATION_AUDITS"
BINDING_PATH = "PHASES/PILOT_V3/CONFIRMATION_BINDING.json"
SCHEMA = "srf6-historical-corrected-confirmation-v1"
BINDING_SCHEMA = "srf6-pilot-v3-corrected-confirmation-binding-v1"
_CACHE = {}
CACHE_LOCATIONS = {
    "XDG_CACHE_HOME": ".",
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
    "TRANSFORMERS_CACHE": "huggingface/hub",
    "PYTORCH_TRANSFORMERS_CACHE": "huggingface/hub",
    "PYTORCH_PRETRAINED_BERT_CACHE": "huggingface/hub",
    "HF_MODULES_CACHE": "huggingface/modules",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "HF_ASSETS_CACHE": "huggingface/assets",
    "HF_XET_CACHE": "huggingface/xet",
    "HF_METRICS_CACHE": "huggingface/metrics",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch_extensions",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba",
    "MPLCONFIGDIR": "matplotlib",
}

# This separately inventoried bridge reconstructs the old evidence through the
# three actual frozen namespaces. It never invokes either old sealing entrypoint.
BOOTSTRAP = r"""
import hashlib, json, os, pathlib, sys

raw = sys.stdin.buffer.read()
r = json.loads(raw)


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


require(
    set(r)
    == {
        "nonce",
        "followup_root",
        "followup_deployment",
        "producer_deployment",
        "closure_root",
        "closure_deployment",
        "closure_pins",
        "registration_ref",
        "receipt_ref",
    },
    "Exact corrected transport required",
)
require(sys.flags.isolated and sys.dont_write_bytecode, "Isolated -I -B child required")
require(
    not any(n.split(".")[0] in {"sr_f6", "sr_f6_followup", "sr_f6_closure"} for n in sys.modules),
    "Packages loaded before isolation",
)
sys.path[:0] = [
    str(pathlib.Path(r[k]) / "src")
    for k in ("producer_deployment", "followup_deployment", "closure_deployment")
]
from sr_f6 import confirm_registration
from sr_f6_followup import source
from sr_f6_closure import source as repair_source, admission_projection, runner

code = pathlib.Path(r["closure_deployment"])
root = pathlib.Path(r["closure_root"])
require(
    pathlib.Path(source.__file__).resolve()
    == pathlib.Path(r["followup_deployment"]) / "src/sr_f6_followup/source.py",
    "Wrong execution consumer",
)
require(
    pathlib.Path(confirm_registration.__file__).resolve()
    == source.PRODUCER_DEPLOYMENT / "src/sr_f6/confirm_registration.py"
    and source.PRODUCER_DEPLOYMENT == pathlib.Path(r["producer_deployment"]),
    "Wrong frozen producer",
)
require(
    repair_source.ROOT == root
    and repair_source.ORIGINAL_DEPLOYMENT == pathlib.Path(r["followup_deployment"])
    and repair_source.original is source,
    "Wrong independently registered repair namespace",
)
require(
    pathlib.Path(repair_source.__file__).resolve() == code / "src/sr_f6_closure/source.py"
    and pathlib.Path(admission_projection.__file__).resolve()
    == code / "src/sr_f6_closure/admission_projection.py",
    "Wrong repair modules",
)
# No writer or original launcher is invoked. The independently inventoried
# bridge is this -c program; no __main__ identity or filename is spoofed.
allowed = tuple(source._path(os.environ[k]) for k in ("TMPDIR", "XDG_CACHE_HOME"))
sys.addaudithook(lambda event, args: runner.guard_write(event, args, allowed))


def check_repair():
    require(
        {p.name for p in source._path(root).iterdir()} == set(r["closure_pins"]),
        "Complete immutable corrected closure membership required",
    )
    values = {}
    for name, pin in r["closure_pins"].items():
        require(pathlib.Path(name).name == name, "Fixed closure leaf required")
        values[name] = source._bound(dict(path=str(root / name), sha256=pin))
    reg = values["REGISTRATION.json"]
    manifest = values["SOURCE_MANIFEST.json"]
    require(
        reg["schema"] == "srf6-technical-closure-source-v1"
        and reg["status"] == "BOUND_NOT_AUTHORIZED"
        and reg["deployment"] == str(code)
        and repair_source.deployment(reg["revision"]) == code,
        "Wrong corrected registration",
    )
    permissions = ("training_allowed", "generation_allowed", "evaluation_allowed")
    require(all(reg[k] is False for k in permissions), "Registration is not permission")
    require(
        reg["source_manifest"]
        == dict(
            path=str(root / "SOURCE_MANIFEST.json"),
            sha256=r["closure_pins"]["SOURCE_MANIFEST.json"],
        )
        and reg["contract"]
        == dict(path=str(root / "CONTRACT.json"), sha256=r["closure_pins"]["CONTRACT.json"]),
        "Corrected registration refs changed",
    )
    require(
        manifest["schema"] == reg["schema"]
        and manifest["revision"] == reg["revision"]
        and manifest["deployment"] == str(code),
        "Corrected source identity differs",
    )
    files = manifest["files"]
    members = {row["path"] for row in files}
    require(
        len(files) == len(members)
        and members == {str(p) for p in repair_source.source_paths(code)},
        "Repair source membership changed",
    )
    for item in files:
        require(
            source._state(item["path"]) == {k: v for k, v in item.items() if k != "sha256"}
            and source._hash(item["path"]) == item["sha256"],
            "Repair source changed",
        )
    for module in tuple(sys.modules.values()):
        spec = getattr(module, "__spec__", None)
        names = (getattr(module, "__name__", ""), getattr(spec, "name", ""))
        if not any(n and n.split(".")[0] == "sr_f6_closure" for n in names):
            continue
        file = getattr(module, "__file__", None)
        require(file is not None, "Unbound repair namespace")
        actual = source._path(pathlib.Path(file).absolute())
        require(
            str(actual) in members and getattr(spec, "origin", None) == str(actual),
            "Loaded repair origin/spec differs",
        )
        for p in getattr(module, "__path__", ()):
            require(source._path(p) == code / "src/sr_f6_closure", "Repair search path differs")
    require(
        sys.path[:3]
        == [
            str(pathlib.Path(r[k]) / "src")
            for k in ("producer_deployment", "followup_deployment", "closure_deployment")
        ],
        "Source order changed",
    )
    old = repair_source.original_context()
    require(
        reg["original_execution_consumer"] == old["consumer"], "Original execution source changed"
    )
    require(
        values["CONTRACT.json"] == repair_source.contract(),
        "Exact reader projection or history changed",
    )
    identity = dict(
        original_execution_consumer=old["consumer"],
        repair_verifier=dict(
            revision=reg["revision"],
            deployment=str(code),
            registration=r["registration_ref"],
            source_manifest=reg["source_manifest"],
            contract=reg["contract"],
        ),
    )
    assessment, receipt = values["ASSESSMENT.json"], values["RECEIPT.json"]
    require(
        receipt
        == dict(
            schema="srf6-independent-technical-closure-v1",
            status="CONFIRMATION_GATES_PASS",
            assessment=dict(
                path=str(root / "ASSESSMENT.json"), sha256=r["closure_pins"]["ASSESSMENT.json"]
            ),
            identity=identity,
            gate_status="CORRECTED_CONFIRMATION_FACTS_SEALED",
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
        ),
        "Complete corrected receipt required",
    )
    require(
        assessment
        == dict(
            schema="srf6-independent-technical-closure-v1",
            status="CONFIRMATION_GATES_PASS",
            scope="EXISTING_CONFIRMATION_EVIDENCE_ONLY_SEPARATE_PILOT_REQUIRED",
            identity=identity,
            preserved_history=repair_source.history(),
            original_rule_assessment=dict(
                path=str(root / "ORIGINAL_RULE_ASSESSMENT.json"),
                sha256=r["closure_pins"]["ORIGINAL_RULE_ASSESSMENT.json"],
            ),
            numerical=dict(
                path=str(root / "NUMERICAL.json"), sha256=r["closure_pins"]["NUMERICAL.json"]
            ),
            old_admission_status="UNKNOWN_NOT_MODIFIED",
            new_model_calls=0,
            optimizer_updates=0,
            training_allowed=False,
            generation_allowed=False,
            evaluation_allowed=False,
        ),
        "Corrected assessment or immutable UNKNOWN changed",
    )
    require(
        values["ATTEMPT.json"]["identity"] == identity
        and values["ATTEMPT.json"]["new_model_calls"] == 0,
        "Actual corrected attempt identity differs",
    )
    return values, identity


for ref, name in ((r["registration_ref"], "REGISTRATION.json"), (r["receipt_ref"], "RECEIPT.json")):
    require(
        ref == dict(path=str(root / name), sha256=r["closure_pins"][name]), "Transport pin differs"
    )
values, identity = check_repair()
# Exactly one full numerical/confirmation reconstruction, no model or old seal.
bundle = admission_projection.evaluate(r["followup_root"])
require(
    bundle["assessment"] == values["ORIGINAL_RULE_ASSESSMENT.json"]
    and bundle["assessment"]["status"] == "CONFIRMATION_GATES_PASS"
    and bundle["numerical"] == values["NUMERICAL.json"],
    "Corrected actual facts changed",
)
context = source.require_support(r["followup_root"], purpose="admission")
review, numerical = context["review"], bundle["numerical"]
refs = [
    dict(path=row["path"], sha256=row["sha256"])
    for row in context["support_binding"]["dependencies"]
]
refs += [dict(path=str(root / name), sha256=pin) for name, pin in r["closure_pins"].items()]
refs += values["ASSESSMENT.json"]["preserved_history"]
refs += [context["support_binding_ref"]]
for item in [*review["inputs"], *review.get("external_inputs", []), *numerical["inputs"]]:
    p = pathlib.Path(item["path"])
    if not p.is_absolute():
        p = pathlib.Path(r["followup_root"]) / p
    refs.append(dict(path=str(p), sha256=item["sha256"]))
unique = {}
for item in refs:
    require(unique.setdefault(item["path"], item) == item, "Conflicting input pins")
after = source.verify_registered(r["followup_root"], require_support=True)
require(after["consumer"] == context["consumer"], "Execution consumer changed")
after_values, after_identity = check_repair()
require(
    after_values == values and after_identity == identity, "Corrected closure changed during read"
)
result = dict(
    receipt=dict(values["RECEIPT.json"], receipt_ref=r["receipt_ref"]),
    consumer=context["consumer"],
    repair_verifier=identity["repair_verifier"],
    producer=context["support_binding"]["producer"],
    support_binding_ref=context["support_binding_ref"],
    support_rows=context["support_binding"]["source_rows"],
    review=review,
    numerical=numerical,
    repair_history=context["spec"]["repair_history"],
    corrected_history=values["ASSESSMENT.json"]["preserved_history"],
    old_admission_status="UNKNOWN_NOT_MODIFIED",
    inputs=[unique[p] for p in sorted(unique)],
    producer_file=str(pathlib.Path(confirm_registration.__file__).resolve()),
    consumer_file=str(pathlib.Path(source.__file__).resolve()),
    repair_file=str(pathlib.Path(admission_projection.__file__).resolve()),
)
sys.stdout.write(
    canonical(
        dict(
            nonce=r["nonce"],
            request_sha256=hashlib.sha256(raw).hexdigest(),
            result=result,
            result_sha256=hashlib.sha256(canonical(result)).hexdigest(),
            isolated=sys.flags.isolated,
            child_pid=os.getpid(),
        )
    ).decode()
    + "\n"
)
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
    _require(
        registration_ref
        == dict(
            path=str(CLOSURE_ROOT / "REGISTRATION.json"), sha256=CLOSURE_PINS["REGISTRATION.json"]
        )
        and receipt_ref
        == dict(path=str(CLOSURE_ROOT / "RECEIPT.json"), sha256=CLOSURE_PINS["RECEIPT.json"]),
        "Only the actual separate corrected closure may be consumed",
    )
    _require(
        {p.name for p in phases._path(CLOSURE_ROOT).iterdir()} == set(CLOSURE_PINS),
        "Complete corrected closure membership required",
    )
    files = {
        name: _absolute_ref(dict(path=str(CLOSURE_ROOT / name), sha256=pin))
        for name, pin in CLOSURE_PINS.items()
    }
    registered = files["REGISTRATION.json"]
    _require(
        registered.get("schema") == "srf6-technical-closure-source-v1"
        and registered.get("revision") == CLOSURE_REVISION
        and registered.get("deployment") == str(CLOSURE_DEPLOYMENT)
        and registered.get("status") == "BOUND_NOT_AUTHORIZED"
        and all(
            registered.get(k) is False
            for k in ("training_allowed", "generation_allowed", "evaluation_allowed")
        ),
        "Only the actual independently registered repair verifier is allowed",
    )
    _require(
        {p.name for p in (CLOSURE_DEPLOYMENT / "src").iterdir()} == {"sr_f6_closure"}
        and {p.name for p in (FOLLOWUP_DEPLOYMENT / "src").iterdir()} == {"sr_f6_followup"},
        "Historical verifier deployments must not shadow the producer",
    )
    for field, name in (("source_manifest", "SOURCE_MANIFEST.json"), ("contract", "CONTRACT.json")):
        _require(
            registered[field] == dict(path=str(CLOSURE_ROOT / name), sha256=CLOSURE_PINS[name]),
            "Corrected registration source references differ",
        )
    _source_manifest(CLOSURE_ROOT / "SOURCE_MANIFEST.json", CLOSURE_DEPLOYMENT)
    followup = _absolute_ref(
        dict(path=str(FOLLOWUP_ROOT / "REGISTRATION.json"), sha256=FOLLOWUP_REGISTRATION_SHA256)
    )
    _require(
        followup.get("schema") == "srf6-followup-registration-v2"
        and followup.get("revision") == FOLLOWUP_REVISION
        and followup.get("root") == str(FOLLOWUP_ROOT)
        and followup.get("deployment") == str(FOLLOWUP_DEPLOYMENT),
        "Original execution consumer differs",
    )
    _absolute_ref(followup["source_manifest"], FOLLOWUP_ROOT / "SOURCE_MANIFEST.json")
    _source_manifest(FOLLOWUP_ROOT / "SOURCE_MANIFEST.json", FOLLOWUP_DEPLOYMENT)
    consumer = dict(
        deployment=str(FOLLOWUP_DEPLOYMENT),
        revision=FOLLOWUP_REVISION,
        registration_sha256=FOLLOWUP_REGISTRATION_SHA256,
        source_manifest_sha256=followup["source_manifest"]["sha256"],
    )
    _require(
        registered["original_execution_consumer"] == consumer,
        "Repair must bind the original T03 execution identity",
    )
    producer = followup["producer"]
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
    _require(
        files["ASSESSMENT.json"]["old_admission_status"] == "UNKNOWN_NOT_MODIFIED",
        "The original UNKNOWN must remain historical evidence",
    )
    return dict(
        closure=registered, followup=followup, consumer=consumer, producer=producer, files=files
    )


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
        CLOSURE_ROOT,
        CLOSURE_DEPLOYMENT,
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
    for base in (FOLLOWUP_ROOT, PRODUCER_ROOT, HISTORY_ROOT, CLOSURE_ROOT):
        for name in trees[str(base)]:
            path = base / name
            if path.is_file():
                paths.add(str(path))
    for base in (FOLLOWUP_DEPLOYMENT, PRODUCER_DEPLOYMENT, HISTORY_DEPLOYMENT, CLOSURE_DEPLOYMENT):
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
        "Actual unchanged corrected confirmation facts required",
    )
    receipt, review, numerical = facts["receipt"], facts["review"], facts["numerical"]
    _require(
        receipt == dict(registered["files"]["RECEIPT.json"], receipt_ref=request["receipt_ref"])
        and receipt["status"] == "CONFIRMATION_GATES_PASS"
        and receipt["gate_status"] == "CORRECTED_CONFIRMATION_FACTS_SEALED"
        and all(
            receipt[k] is False
            for k in ("training_allowed", "generation_allowed", "evaluation_allowed")
        ),
        "Sealed actual corrected confirmation required",
    )
    repair = dict(
        revision=CLOSURE_REVISION,
        deployment=str(CLOSURE_DEPLOYMENT),
        registration=request["registration_ref"],
        source_manifest=registered["closure"]["source_manifest"],
        contract=registered["closure"]["contract"],
    )
    _require(
        facts["producer"] == registered["producer"]
        and facts["consumer"] == registered["consumer"]
        and facts["repair_verifier"] == repair
        and receipt["identity"]
        == dict(original_execution_consumer=facts["consumer"], repair_verifier=repair),
        "Actual producer, execution consumer and repair verifier identities differ",
    )
    _require(
        facts["producer_file"] == str(PRODUCER_DEPLOYMENT / "src/sr_f6/confirm_registration.py")
        and facts["consumer_file"] == str(FOLLOWUP_DEPLOYMENT / "src/sr_f6_followup/source.py")
        and facts["repair_file"]
        == str(CLOSURE_DEPLOYMENT / "src/sr_f6_closure/admission_projection.py"),
        "Actual three verifier module origins differ",
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
        "Complete original support and sealed review required",
    )
    _require(
        numerical == registered["files"]["NUMERICAL.json"]
        and numerical["scope"] == "SERVER_ACTUAL"
        and numerical["status"] == "PASS"
        and set(numerical["parents"]) == {"b0", "b1", "b2"}
        and all(item["status"] == "PASS" for item in numerical["parents"].values())
        and numerical["generated_responses"] == numerical["optimizer_updates"] == 0,
        "All original T03 evidence must pass without new responses or updates",
    )
    _require(
        facts["repair_history"]["status"] == "HISTORICAL_NEGATIVE_EVIDENCE_NOT_AUTHORITY"
        and facts["repair_history"]["completed"] is False
        and facts["old_admission_status"] == "UNKNOWN_NOT_MODIFIED"
        and facts["corrected_history"]
        == registered["files"]["ASSESSMENT.json"]["preserved_history"],
        "Both earlier failed review and original UNKNOWN must remain preserved",
    )
    _require(facts["inputs"], "Complete historical input closure required")
    return facts


def verify_corrected_confirmation(
    root, *, closure_registration_ref, corrected_confirmation_receipt_ref
):
    """One cold isolated attempt, permanently retaining errors and subprocess evidence."""
    root = phases._root(root)
    registered = _request(closure_registration_ref, corrected_confirmation_receipt_ref)
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
        closure_root=str(CLOSURE_ROOT),
        closure_deployment=str(CLOSURE_DEPLOYMENT),
        closure_pins=copy.deepcopy(CLOSURE_PINS),
        registration_ref=closure_registration_ref,
        receipt_ref=corrected_confirmation_receipt_ref,
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
    # A scheduler wrapper may export its own cache directories. Every imported
    # library cache must stay within this child's permitted write boundary.
    environment.update({name: str(attempt / "tmp") for name in ("TMPDIR", "TMP", "TEMP")})
    environment.update(
        {name: str(attempt / "cache" / leaf) for name, leaf in CACHE_LOCATIONS.items()}
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
            _request(closure_registration_ref, corrected_confirmation_receipt_ref) == registered,
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
            digest(closure_registration_ref),
            digest(corrected_confirmation_receipt_ref),
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
        closure_registration_ref=registration_ref,
        corrected_confirmation_receipt_ref=receipt_ref,
        facts_sha256=verification["facts_sha256"],
        original_verification_ref=verification["report_ref"],
        producer=verification["facts"]["producer"],
        consumer=verification["facts"]["consumer"],
        repair_verifier=verification["facts"]["repair_verifier"],
        training_authorized=False,
        generation_authorized=False,
        formal_authorized=False,
    )


def freeze_confirmation_binding(
    root, *, phase_context, closure_registration_ref, corrected_confirmation_receipt_ref
):
    root = phases._root(root)
    phase_context = _phase(root, phase_context)
    target = phases._path(root / BINDING_PATH, exists=False)
    _require(not target.exists(), "Preserve existing confirmation binding")
    verification = verify_corrected_confirmation(
        root,
        closure_registration_ref=closure_registration_ref,
        corrected_confirmation_receipt_ref=corrected_confirmation_receipt_ref,
    )
    _phase(root, phase_context)
    binding = _binding(
        phase_context, closure_registration_ref, corrected_confirmation_receipt_ref, verification
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
        binding["closure_registration_ref"],
        binding["corrected_confirmation_receipt_ref"],
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
        verification = verify_corrected_confirmation(
            root,
            closure_registration_ref=registration_ref,
            corrected_confirmation_receipt_ref=receipt_ref,
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
