#!/usr/bin/env python3
"""Independent CPU dispatcher for the registered SR-F3-CE scientific deployment.

This operational script is installed under ROOT/operations/COMMIT. It imports
only ROOT/code/src, preserves the frozen CONFIG, and never evaluates a model score.
An explicitly bound operational adapter packs independent fixed tasks according
to the actual scheduler limits and eligible single-node capacity.
Lease continuation is an explicit new numbered CPU allocation, not a science
deadline or an automatic retry of an uncertain submission.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT = "SR-F3-CE-20261016"
FROZEN_COMMIT = "71483fbf31d57213fdc7e6f3c881848397b45329"
QOS = "soujanya-poria-startfund-2026-03"
ENGINES = {"p0", "engine--b0", "engine--b1", "engine--b2", "engine--resume"}
REVISION = "external/revisions/typed_v1"
HELPERS = (
    "dispatch_registered.py",
    "finalize_external_types.py",
    "build_delivery.py",
    "reconcile_submissions.py",
    "dispatch_efficiency.py",
    "pooled_execution.py",
    "scheduler_capacity.py",
)
TYPED_COMMIT = "4146e156ffa9a476b360daeb67503703faca138f"
TYPED_SCRIPT_SHA = "1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf"
_EXTENSIONS = {}
_POOL_SOURCES = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def bounded(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Unsafe operational artifact path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise PermissionError("Operational artifact escaped the run")
    return path


def save(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read(path) != value:
                    raise PermissionError("Immutable operational record differs") from None
        else:
            os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another dispatcher owns this run") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def source_revision(root):
    """Keep v1 pinned; authenticate the exact adopted revision before importing it."""
    root = Path(root).resolve(strict=True)
    registration = read(root / "RUN_IDENTITY.json")
    config_path = root / "plan/config/SR_F3_CE.json"
    config = read(config_path) if config_path.is_file() else {}
    revision = config.get("protocol_revision", "v1")
    if revision == "v1":
        if registration.get("code_commit") != FROZEN_COMMIT:
            raise PermissionError(
                "Original protocol identity changed: frozen scientific commit required"
            )
        return dict(code_commit=FROZEN_COMMIT, protocol_revision="v1")
    if revision not in {"v2", "c2"}:
        raise PermissionError("Unregistered protocol revision")
    code = root / "code"
    sys.dont_write_bytecode = True
    if any(p.is_file() and p.suffix in {".pyc", ".pyo"} for p in code.rglob("*")):
        raise PermissionError("Unbound scientific bytecode cache must not be imported")
    manifest = read(root / "deployment/SOURCE_MANIFEST.json")
    rows = manifest.get("files", [])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise PermissionError("Revised source inventory is empty or ambiguous")
    for row in rows:
        source = bounded(code, row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise PermissionError("Revised source changed before guard import")
    if not sys.path or sys.path[0] != str(code / "src"):
        sys.path.insert(0, str(code / "src"))
    module = importlib.import_module("sr_f3_ce.protocol")
    if Path(module.__file__).resolve() != code / "src/sr_f3_ce/protocol.py":
        raise PermissionError("Protocol source guard was imported from another deployment")
    checked = module.verify_run_source_revision(root, source_manifest=manifest)
    if checked.get("protocol_revision") != revision or (
        checked.get("code_commit") != registration.get("code_commit")
    ):
        raise PermissionError("Selected protocol/source guard differs from this run")
    return checked


def revision_binding(checked):
    return {} if checked["protocol_revision"] == "v1" else {"protocol_source_guard": checked}


def load_extension(root, manifest_path, *, expected_sha=None):
    """Verify the complete explicit source inventory before executing its launcher."""
    root = Path(root).resolve(strict=True)
    checked_source = source_revision(root)
    path = Path(manifest_path).resolve(strict=True)
    manifest_bytes = path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_sha is not None and manifest_sha != expected_sha:
        raise PermissionError("Operation-bound extension manifest changed before import")
    manifest = json.loads(manifest_bytes)
    commit = manifest.get("extension_commit", "")
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or (
        path != root / "native/observability" / commit / "SOURCE_MANIFEST.json"
        or manifest.get("status") != "FROZEN_OBSERVABILITY_EXTENSION"
        or manifest.get("base_code_commit") != checked_source["code_commit"]
        or manifest.get("run_identity_sha256") != sha(root / "RUN_IDENTITY.json")
        or manifest.get("base_source_manifest_sha256")
        != sha(root / "deployment/SOURCE_MANIFEST.json")
        or manifest.get("executed_sources") != ["instrumented_execution.py", "observe_runtime.py"]
    ):
        raise PermissionError("Explicit observability deployment identity differs")
    if any(manifest.get(k) != v for k, v in revision_binding(checked_source).items()):
        raise PermissionError("Observability manifest omits the selected protocol/source guard")
    records = manifest.get("sources", [])
    names = {row["path"] for row in records}
    if len(names) != len(records) or not set(manifest["executed_sources"]) <= names:
        raise PermissionError("Observation extension source inventory is incomplete")
    cpu = manifest.get("cpu_validation", {})
    if cpu.get("path") != "CPU_VALIDATION.json":
        raise PermissionError("Observation extension CPU validation is missing")
    for row in [*records, cpu]:
        source = bounded(path.parent, row["path"])
        if (
            not source.is_file()
            or source.stat().st_size != row["bytes"]
            or sha(source) != row["sha256"]
        ):
            raise PermissionError(
                "Observation extension source or CPU evidence changed before import"
            )
    key = (str(path), manifest_sha)
    if key not in _EXTENSIONS:
        entry = path.parent / "instrumented_execution.py"
        record = next(row for row in records if row["path"] == "instrumented_execution.py")
        body = entry.read_bytes()
        if len(body) != record["bytes"] or hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise PermissionError("Observation launcher changed before verified-byte import")
        spec = importlib.util.spec_from_file_location("srf3ce_observability_" + key[1], entry)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            exec(compile(body, str(entry), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _EXTENSIONS[key] = module
    module = _EXTENSIONS[key]
    if module.verify_extension(root, path)["manifest_sha256"] != manifest_sha:
        raise PermissionError("Extension manifest changed during verified import")
    return module


def load_repair_source(operation, *, expected_sha=None):
    path = Path(operation) / "reconcile_submissions.py"
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise PermissionError("Operational reconciliation source changed")
    name = "srf3ce_submission_repair_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


def load_pool_source(operation, *, expected_sha):
    path = Path(operation).resolve(strict=True) / "pooled_execution.py"
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha:
        raise PermissionError("Pooled operation source changed before import")
    key = (str(path), expected_sha)
    if key not in _POOL_SOURCES:
        spec = importlib.util.spec_from_file_location("srf3ce_pool_" + expected_sha, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            exec(compile(body, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        _POOL_SOURCES[key] = module
    return _POOL_SOURCES[key]


def load_capacity_source(operation, *, expected_sha):
    path = Path(operation).resolve(strict=True) / "scheduler_capacity.py"
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha:
        raise PermissionError("Scheduler capacity source changed before import")
    spec = importlib.util.spec_from_file_location("srf3ce_capacity_" + expected_sha, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


def capacity_limits(value, api, *, fresh=False):
    capacity = getattr(api, "capacity", None)
    if capacity is not None:
        return capacity.limits(fresh=fresh)
    if "task_pool" in value:
        raise PermissionError("Authorized pooled operation lacks live scheduler capacity")
    # Compatibility only for historical operational registrations without pooling.
    return dict(gpu_limit=value["gpu_limit"], submitted_limit=value["submitted_limit"])


def capacity_headroom(api, usage, limits):
    if getattr(api, "capacity", None) is not None:
        return api.capacity.headroom(usage, limits)
    return dict(
        gpus=limits["gpu_limit"] - usage["gpus"],
        submitted=limits["submitted_limit"] - usage["submitted"],
    )


def check_efficiency_authorization(root, path):
    """Bind one explicitly requested operational replacement, never a science retry."""
    root = Path(root).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root / "operations"):
        raise PermissionError("Efficiency authorization must belong to this run")
    authorization = read(path)
    if (
        authorization.get("schema_version") != 1
        or authorization.get("scope") != "SRF3CE_DISPATCH_EFFICIENCY"
        or authorization.get("scientific_settings_changed") is not False
        or authorization.get("new_scientific_tasks") != 0
        or not authorization.get("user_request")
        or authorization.get("run_identity_sha256") != sha(root / "RUN_IDENTITY.json")
        or authorization.get("execution_freeze_sha256") != sha(root / "EXECUTION_FREEZE.json")
    ):
        raise PermissionError("Efficiency authorization does not preserve this frozen run")
    prior = authorization["predecessor"]
    directory = root / "operations/dispatcher/attempts" / f"{prior['sequence']:04d}"
    old_identity = bounded(root, prior["identity_path"])
    if sha(old_identity) != prior["identity_sha256"] or read(old_identity).get("root") != str(root):
        raise PermissionError("Efficiency predecessor operation differs")
    intent, state = read(directory / "INTENT.json"), read(directory / "STATE.json")
    if (
        intent.get("operation_identity_sha256") != prior["identity_sha256"]
        or state.get("job_id") != prior["job_id"]
        or state.get("status") != "SUBMITTED"
        or state.get("intent_sha256") != sha(directory / "INTENT.json")
        or prior.get("intent_sha256") != sha(directory / "INTENT.json")
        or prior.get("state_sha256") != sha(directory / "STATE.json")
    ):
        raise PermissionError("Efficiency predecessor submission differs")
    end = directory / "END.json"
    if prior.get("end_sha256") != (sha(end) if end.exists() else None):
        raise PermissionError("Efficiency predecessor END history differs")
    transition = bounded(root, authorization["transition_path"])
    if sha(transition) != authorization["transition_sha256"]:
        raise PermissionError("Efficiency transition record differs")
    receipt = read(transition)
    cancelled = (
        receipt.get("status") == "OPERATOR_HANDOFF_CANCELLED"
        and receipt.get("cancel_returncode") == 0
        and authorization["terminal"].get("state") == "CANCELLED"
    )
    failed = (
        receipt.get("status") == "OPERATOR_HANDOFF_FAILED"
        and authorization["terminal"].get("state") == "FAILED"
        and authorization["terminal"].get("exit_code") == "1:0"
        and end.exists()
        and read(end).get("status") == "FAILED"
        and read(end).get("error")
        == "PermissionError('Actual teacher capacity exceeds the live scheduler limit')"
    )
    if failed:
        operation = Path(__file__).resolve().parent
        module = load_pool_source(
            operation,
            expected_sha=read(operation / "OPERATION_SOURCES.json")["files"]["pooled_execution.py"],
        )
        module.check_pending_split(root, authorization.get("pending_split"))
    if (
        not (cancelled or failed)
        or receipt.get("job_id") != prior["job_id"]
        or receipt.get("orchestration_lease_held") is not True
        or receipt.get("queue_absent") is not True
        or receipt.get("terminal") != authorization.get("terminal")
        or authorization["terminal"].get("job_id") != prior["job_id"]
    ):
        raise PermissionError("Efficiency transition is not a fenced terminal handoff")
    return dict(schema_version=1, authorization_path=str(path), authorization_sha256=sha(path))


def verify_efficiency_predecessor(value, api, prior, intent, state, queue):
    bound = value["dispatch_efficiency"]
    root = Path(value["root"])
    if check_efficiency_authorization(root, bound["authorization_path"]) != bound:
        raise PermissionError("Efficiency authorization changed")
    authorization = read(bound["authorization_path"])
    predecessor = authorization["predecessor"]
    if authorization["terminal"].get("state") == "FAILED" and (
        authorization.get("pending_split") != value.get("task_pool", {}).get("pending_split")
    ):
        raise PermissionError("Failed dispatcher handoff requires this bound pending-pool split")
    if (
        prior.name != f"{predecessor['sequence']:04d}"
        or state.get("job_id") != predecessor["job_id"]
        or state.get("job_id") in queue
        or intent.get("operation_identity_sha256") != predecessor["identity_sha256"]
    ):
        raise PermissionError("Only the explicitly retired dispatcher may be replaced")
    terminal = api.slurm.terminal(state["job_id"])
    if terminal != authorization["terminal"]:
        raise PermissionError("Retired dispatcher live accounting differs")
    return terminal


def register(
    root,
    python,
    user,
    operation_commit,
    extension_manifest,
    repair_authorization=None,
    efficiency_authorization=None,
    pool_authorization=None,
    dispatcher_qos=QOS,
):
    root, script = Path(root).resolve(strict=True), Path(__file__).resolve(strict=True)
    if not re.fullmatch(r"[a-f0-9]{40}", operation_commit) or (
        script.parent != root / "operations" / operation_commit
    ):
        raise PermissionError("Install this exact operation commit before registering")
    identity, config = read(root / "RUN_IDENTITY.json"), read(root / "orchestration/CONFIG.json")
    checked_source = source_revision(root)
    if identity.get("code_commit") != checked_source["code_commit"] or config.get("pool_size") != 1:
        raise PermissionError("Frozen scientific commit and single-GPU configuration required")
    if config.get("cpu_policy") != dict(partition="cluster02", qos=QOS, cpu_node="cpu-2"):
        raise PermissionError("Only the verified teacher CPU policy is authorized")
    extension_manifest = Path(extension_manifest).resolve(strict=True)
    extension = load_extension(root, extension_manifest)
    extension_identity = extension.verify_extension(root, extension_manifest)
    typed_script = root / "operations" / TYPED_COMMIT / "finalize_external_types.py"
    if sha(typed_script) != TYPED_SCRIPT_SHA:
        raise PermissionError("The already-published typed preparer source must be retained")
    sources_path = script.parent / "OPERATION_SOURCES.json"
    save(
        sources_path,
        dict(
            schema_version=1,
            experiment_id=EXPERIMENT,
            operation_commit=operation_commit,
            scientific_code_commit=checked_source["code_commit"],
            **revision_binding(checked_source),
            files={name: sha(script.parent / name) for name in HELPERS},
        ),
        immutable=True,
    )
    value = dict(
        schema_version=1,
        experiment_id=EXPERIMENT,
        operation_commit=operation_commit,
        root=str(root),
        code_root=str(root / "code"),
        frozen_code_commit=checked_source["code_commit"],
        **revision_binding(checked_source),
        run_identity_sha256=sha(root / "RUN_IDENTITY.json"),
        source_manifest_sha256=sha(root / "deployment/SOURCE_MANIFEST.json"),
        operation_sources_sha256=sha(sources_path),
        config_sha256=sha(root / "orchestration/CONFIG.json"),
        dispatcher_sha256=sha(script),
        extension_manifest_path=str(extension_manifest),
        extension_manifest_sha256=extension_identity["manifest_sha256"],
        typed_preparer_commit=TYPED_COMMIT,
        typed_preparer_sha256=TYPED_SCRIPT_SHA,
        python=str(Path(python).resolve(strict=True)),
        user=user,
        account="rose",
        qos=QOS,
        partition="cluster02",
        cpu_node="cpu-2",
        cpus=4,
        memory_gib=12,
        time_limit="3-00:00:00",
        pool_size=1,
        gpu_limit=5,
        submitted_limit=5,
        poll_seconds=60,
        handoff_margin_seconds=900,
    )
    if repair_authorization is not None:
        module = load_repair_source(
            script.parent,
            expected_sha=read(sources_path)["files"]["reconcile_submissions.py"],
        )
        value["submission_repair"] = module.check_authorization(root, repair_authorization)
    if efficiency_authorization is not None:
        value["dispatch_efficiency"] = check_efficiency_authorization(
            root, efficiency_authorization
        )
    if pool_authorization is not None:
        module = load_pool_source(
            script.parent, expected_sha=read(sources_path)["files"]["pooled_execution.py"]
        )
        value["task_pool"] = module.check_authorization(root, pool_authorization)
    if pool_authorization is not None:
        value.pop("gpu_limit")
        value.pop("submitted_limit")
        value["capacity_policy"] = "LIVE_SLURM_QOS_AND_ASSOCIATION"
    if dispatcher_qos != QOS:
        if (
            dispatcher_qos != "rose"
            or "task_pool" not in value
            or "dispatch_efficiency" not in value
        ):
            raise PermissionError("Ordinary CPU QoS requires the bound operational handoff")
        authorization = read(value["dispatch_efficiency"]["authorization_path"])
        if authorization.get("dispatcher_qos") != dispatcher_qos:
            raise PermissionError("CPU QoS change lacks explicit authorization")
        value["dispatcher_qos"] = dispatcher_qos
    path = script.parent / "OPERATION_IDENTITY.json"
    save(path, value, immutable=True)
    return path


def identity(path):
    path = Path(path).resolve(strict=True)
    value, script = read(path), Path(__file__).resolve(strict=True)
    root = Path(value["root"]).resolve(strict=True)
    checked_source = source_revision(root)
    expected = dict(
        schema_version=1,
        experiment_id=EXPERIMENT,
        frozen_code_commit=checked_source["code_commit"],
        **revision_binding(checked_source),
        code_root=str(root / "code"),
        dispatcher_sha256=sha(script),
        extension_manifest_sha256=sha(value["extension_manifest_path"]),
        typed_preparer_commit=TYPED_COMMIT,
        typed_preparer_sha256=TYPED_SCRIPT_SHA,
        run_identity_sha256=sha(root / "RUN_IDENTITY.json"),
        source_manifest_sha256=sha(root / "deployment/SOURCE_MANIFEST.json"),
        operation_sources_sha256=sha(script.parent / "OPERATION_SOURCES.json"),
        config_sha256=sha(root / "orchestration/CONFIG.json"),
        account="rose",
        qos=QOS,
        partition="cluster02",
        cpu_node="cpu-2",
        cpus=4,
        memory_gib=12,
        time_limit="3-00:00:00",
        pool_size=1,
        gpu_limit=5,
        submitted_limit=5,
        poll_seconds=60,
        handoff_margin_seconds=900,
    )
    if "task_pool" in value:
        expected.pop("gpu_limit")
        expected.pop("submitted_limit")
        expected["capacity_policy"] = "LIVE_SLURM_QOS_AND_ASSOCIATION"
        if "gpu_limit" in value or "submitted_limit" in value:
            raise PermissionError("Live scheduler policy cannot carry an artificial fixed cap")
    if any(value.get(k) != v for k, v in expected.items()) or (
        not re.fullmatch(r"[a-f0-9]{40}", str(value.get("operation_commit", "")))
        or script.parent != root / "operations" / value["operation_commit"]
        or path != script.parent / "OPERATION_IDENTITY.json"
        or Path(value["python"]).resolve(strict=True) != Path(value["python"])
    ):
        raise PermissionError("Operation/source/run identity changed")
    sources = read(script.parent / "OPERATION_SOURCES.json")
    if sources != dict(
        schema_version=1,
        experiment_id=EXPERIMENT,
        operation_commit=value["operation_commit"],
        scientific_code_commit=checked_source["code_commit"],
        **revision_binding(checked_source),
        files={name: sha(script.parent / name) for name in HELPERS},
    ):
        raise PermissionError("Auxiliary operation source changed")
    registration, config = (
        read(root / "RUN_IDENTITY.json"),
        read(root / "orchestration/CONFIG.json"),
    )
    if (
        registration.get("experiment_id") != EXPERIMENT
        or registration.get("code_commit") != checked_source["code_commit"]
    ):
        raise PermissionError("A different scientific experiment/source was supplied")
    if (
        config.get("pool_size") != 1
        or config.get("python") != value["python"]
        or (
            config.get("user") != value["user"]
            or config.get("cpu_policy") != dict(partition="cluster02", qos=QOS, cpu_node="cpu-2")
        )
    ):
        raise PermissionError("Dispatcher cannot alter registered scheduler configuration")
    load_extension(
        root, value["extension_manifest_path"], expected_sha=value["extension_manifest_sha256"]
    )
    if sha(root / "operations" / TYPED_COMMIT / "finalize_external_types.py") != TYPED_SCRIPT_SHA:
        raise PermissionError("Previously published typed preparer source changed")
    if "submission_repair" in value:
        module = load_repair_source(
            script.parent, expected_sha=sources["files"]["reconcile_submissions.py"]
        )
        if (
            module.check_authorization(root, value["submission_repair"]["authorization_path"])
            != value["submission_repair"]
        ):
            raise PermissionError("Operational repair authorization changed")
    if "dispatch_efficiency" in value and (
        check_efficiency_authorization(root, value["dispatch_efficiency"]["authorization_path"])
        != value["dispatch_efficiency"]
    ):
        raise PermissionError("Efficiency authorization changed")
    if "task_pool" in value:
        module = load_pool_source(
            script.parent, expected_sha=sources["files"]["pooled_execution.py"]
        )
        if (
            module.check_authorization(root, value["task_pool"]["authorization_path"])
            != value["task_pool"]
        ):
            raise PermissionError("Pooled operation authorization changed")
    if "dispatcher_qos" in value:
        if (
            value["dispatcher_qos"] != "rose"
            or "task_pool" not in value
            or "dispatch_efficiency" not in value
        ):
            raise PermissionError("Unsupported dispatcher-only QoS")
        if (
            read(value["dispatch_efficiency"]["authorization_path"]).get("dispatcher_qos")
            != value["dispatcher_qos"]
        ):
            raise PermissionError("Dispatcher QoS differs from its authorization")
    return value


def load_science(value):
    code = Path(value["code_root"]).resolve(strict=True)
    sys.dont_write_bytecode = True
    if any(path.is_file() and path.suffix in {".pyc", ".pyo"} for path in code.rglob("*")):
        raise PermissionError("Unbound scientific bytecode cache must not be imported")
    manifest = read(Path(value["root"]) / "deployment/SOURCE_MANIFEST.json")
    records = manifest["files"]
    if not records or len({row["path"] for row in records}) != len(records):
        raise PermissionError("Frozen source manifest is empty or ambiguous")
    for row in records:
        source = bounded(code, row["path"])
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise PermissionError("Frozen source changed before import")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code / "src"))
    modules = {}
    for name in ("orchestration", "protocol", "bootstrap", "external"):
        module = importlib.import_module("sr_f3_ce." + name)
        if Path(module.__file__).resolve() != code / "src/sr_f3_ce" / (name + ".py"):
            raise PermissionError("Dispatcher imported a different scientific source")
        modules[name] = module
    extension = load_extension(
        Path(value["root"]),
        value["extension_manifest_path"],
        expected_sha=value["extension_manifest_sha256"],
    )
    controller_class = extension.install(value["root"], value["extension_manifest_path"])
    if modules["orchestration"].Controller is not controller_class:
        raise PermissionError("The explicit future science Controller was not installed")
    c = controller_class(
        value["root"],
        value["code_root"],
        python=value["python"],
        user=value["user"],
        pool_size=1,
        cpu_partition="cluster02",
        cpu_qos=QOS,
        cpu_node="cpu-2",
    )
    api = SimpleNamespace(**modules, controller=c, slurm=c.slurm, extension=extension)
    if "task_pool" in value:
        operation = Path(value["identity_path"]).parent
        module = load_pool_source(
            operation,
            expected_sha=read(operation / "OPERATION_SOURCES.json")["files"]["pooled_execution.py"],
        )
        capacity_module = load_capacity_source(
            operation,
            expected_sha=read(operation / "OPERATION_SOURCES.json")["files"][
                "scheduler_capacity.py"
            ],
        )
        api.capacity = capacity_module.Capacity(value)
        api.task_pool = module.install(value, extension, capacity=api.capacity)
    if "submission_repair" in value:
        operation = Path(value["identity_path"]).parent
        source = read(operation / "OPERATION_SOURCES.json")["files"]["reconcile_submissions.py"]
        module = load_repair_source(operation, expected_sha=source)
        io = SimpleNamespace(read=read, save=save, sha=sha, state_rows=state_rows)
        api.repair = module.Repair(value, api, io)
    if "dispatch_efficiency" in value:
        operation = Path(value["identity_path"]).parent
        path = operation / "dispatch_efficiency.py"
        expected = read(operation / "OPERATION_SOURCES.json")["files"][path.name]
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected:
            raise PermissionError("Efficiency source changed before import")
        spec = importlib.util.spec_from_file_location("srf3ce_efficiency_" + expected, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(body, str(path), "exec"), module.__dict__)
        api.efficiency = module.Efficiency(value, api, SimpleNamespace(read=read, sha=sha))
    return api


def _lines(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise PermissionError("Empty external identity manifest")
    return rows


def check_blind_fields(value):
    if isinstance(value, dict):
        if set(value) & {
            "gold",
            "answer",
            "Answer",
            "Year",
            "prediction",
            "model_output",
            "raw_text",
        }:
            raise PermissionError("Reviewer evidence contains a forbidden answer/output field")
        for child in value.values():
            check_blind_fields(child)
    elif isinstance(value, list):
        for child in value:
            check_blind_fields(child)


def verify_typed(root, external):
    """Reconstruct the existing typed_v1 publication without gold or model scoring."""
    revision = root / REVISION
    if not (revision / "PUBLICATION.json").exists():
        return None
    publication = read(revision / "PUBLICATION.json")
    panels = read(root / "external/PANELS.json")
    audit_path = revision / "TYPE_AUDIT_RECEIPT.json"
    audit = read(audit_path)
    if (
        publication.get("status") != "PUBLISHED_TYPED_V1_BEFORE_SCIENCE"
        or publication.get("current_panels_sha256") != sha(root / "external/PANELS.json")
        or publication.get("previous_panels_sha256") != sha(revision / "PANELS.v0.json")
        or publication.get("typed_audit_receipt_sha256") != sha(audit_path)
        or sha(revision / "PANELS.v1.json") != sha(root / "external/PANELS.json")
        or panels.get("typed_audit_revision") != "typed_v1"
        or panels.get("typed_audit_receipt") != REVISION + "/TYPE_AUDIT_RECEIPT.json"
        or panels.get("typed_audit_receipt_sha256") != sha(audit_path)
    ):
        raise PermissionError("Typed publication/pointer identity differs")
    checked_source = source_revision(root)
    required = dict(
        status="TWO_INDEPENDENT_REVIEW_PASSES_COMPLETED",
        experiment_id=EXPERIMENT,
        run_identity_sha256=sha(root / "RUN_IDENTITY.json"),
        code_commit=checked_source["code_commit"],
        scientific_source_changed=False,
        model_calls=0,
        generation_budget_changed=False,
        before_execution_freeze=True,
        before_any_new_test=True,
        original_panels_sha256=sha(revision / "PANELS.v0.json"),
    )
    if any(type(audit.get(k)) is not type(v) or audit[k] != v for k, v in required.items()):
        raise PermissionError("Typed audit is not a completed prospective independent review")
    if not re.fullmatch(r"[a-f0-9]{40}", audit.get("auxiliary_code_commit", "")) or (
        audit.get("auxiliary_commit_identity_supplied_by_operator") is not True
        or not re.fullmatch(r"[a-f0-9]{64}", audit.get("preparation_script_sha256", ""))
    ):
        raise PermissionError("Typed preparation source identity is missing")
    original = read(revision / "PANELS.v0.json")
    if any(
        p.get("status") != "FROZEN" or p.get("experiment_id") != EXPERIMENT
        for p in (panels, original)
    ):
        raise PermissionError("Typed panel index is not the registered experiment")
    items, by_pool, registries = {}, {}, {}
    for pool in ("CHARTQA", "CHARTQAPRO"):
        if {k: v for k, v in panels[pool].items() if k not in {"root", "freeze_sha256"}} != {
            k: v for k, v in original[pool].items() if k not in {"root", "freeze_sha256"}
        }:
            raise PermissionError("A non-type external panel declaration changed")
        directory, old = bounded(root, panels[pool]["root"]), bounded(root, original[pool]["root"])
        freeze, old_freeze = read(directory / "FREEZE.json"), read(old / "FREEZE.json")
        if sha(directory / "FREEZE.json") != panels[pool]["freeze_sha256"] or (
            sha(old / "FREEZE.json") != original[pool]["freeze_sha256"]
        ):
            raise PermissionError("Typed external freeze bytes changed")
        for name in ("FREEZE.json", "TYPE_REGISTRY.json", "SELECTION.json"):
            if (old / name).exists() and sha(old / name) != sha(revision / "v0" / pool / name):
                raise PermissionError("Preserved original external identity changed")
        ignored = {"type_registry_sha256", "typed_unresolved_count"}
        if {k: v for k, v in freeze.items() if k not in ignored} != {
            k: v for k, v in old_freeze.items() if k not in ignored
        }:
            raise PermissionError("A non-type external selection/input setting changed")
        if pool == "CHARTQAPRO" and freeze.get("status") == "UNAVAILABLE":
            if not freeze.get("reason") or freeze.get("image_count") != 0:
                raise PermissionError("Missing explicit external-only unavailability")
            if audit["identities"].get(pool) != dict(status="UNAVAILABLE", reason=freeze["reason"]):
                raise PermissionError("Unavailable external identity changed")
            continue
        if freeze.get("status") != "FROZEN" or (
            pool == "CHARTQA" and freeze.get("image_count") != 2500
        ):
            raise PermissionError("External panel is incomplete")
        if pool == "CHARTQAPRO" and (
            type(freeze.get("image_count")) is not int or not 1 <= freeze["image_count"] <= 256
        ):
            raise PermissionError("ChartQAPro selected panel exceeds its fixed size")
        for name in ("MODEL_INPUTS.jsonl", "TASKS_AUDIT_ONLY.jsonl", "SELECTION.json"):
            if (old / name).exists() and sha(directory / name) != sha(old / name):
                raise PermissionError("Typed revision changed frozen original bytes")
        if sha(directory / "MODEL_INPUTS.jsonl") != freeze["inputs_sha256"] or (
            sha(directory / "TASKS_AUDIT_ONLY.jsonl") != freeze["audit_tasks_sha256"]
        ):
            raise PermissionError("External inputs/audit provenance changed")
        inputs = _lines(directory / "MODEL_INPUTS.jsonl")
        if pool == "CHARTQA":
            from sr_f1.chartqa import PROMPT

            prefix = PROMPT + "\n"
            if any(not r["prompt"].startswith(prefix) for r in inputs):
                raise PermissionError("ChartQA prompt interface changed")
            selected = [
                dict(
                    qid=r["qid"],
                    question=r["prompt"][len(prefix) :],
                    image_sha256=r["image_sha256"],
                )
                for r in inputs
            ]
        else:
            selection = read(directory / "SELECTION.json")
            if digest(selection) != freeze["selection_sha256"]:
                raise PermissionError("ChartQAPro selection identity changed")
            selected = selection["selected"]
        chosen = {r["qid"]: r for r in selected}
        if (
            len(chosen) != len(selected)
            or set(chosen) != {r["qid"] for r in inputs}
            or set(chosen) & set(items)
        ):
            raise PermissionError("External blind question coverage differs")
        store = external.ExternalInputStore(directory)
        for qid in chosen:
            request = store.model_input(qid)
            image = (store.image_root / request["image_file"]).resolve(strict=True)
            if not image.is_relative_to(store.image_root) or sha(image) != request["image_sha256"]:
                raise PermissionError("Reviewed visible image changed")
        registry = read(directory / "TYPE_REGISTRY.json")
        if digest(registry) != freeze["type_registry_sha256"]:
            raise PermissionError("Typed registry identity changed")
        expected_identity = dict(
            questions=len(chosen),
            inputs_sha256=freeze["inputs_sha256"],
            audit_tasks_sha256=freeze["audit_tasks_sha256"],
            type_registry_sha256=freeze["type_registry_sha256"],
            image_and_prompt_identity_unchanged=True,
            selection_unchanged=True,
        )
        if audit["identities"].get(pool) != expected_identity:
            raise PermissionError("Typed audit panel/input/selection identity differs")
        by_pool[pool], registries[pool] = chosen, registry
        items.update(chosen)
    reviews = []
    for label in ("A", "B"):
        name = "REVIEW_" + label + ".json"
        path = revision / name
        if sha(path) != audit["review_files"][name]:
            raise PermissionError("Original reviewer bytes changed")
        payload = read(path)
        reviewer = payload.get("reviewer_id")
        if (
            not isinstance(reviewer, str)
            or not reviewer.strip()
            or payload.get("experiment_id") != EXPERIMENT
            or payload.get("blind_inputs_only") is not True
            or any(
                payload.get(k) is not False
                for k in ("used_gold", "used_model_output", "saw_other_review")
            )
        ):
            raise PermissionError("Reviewer blindness or identity is missing")
        if (
            payload.get("annotation_scope_sha256", audit["annotation_scope_sha256"])
            != audit["annotation_scope_sha256"]
        ):
            raise PermissionError("Reviewer attested a different blind input scope")
        rows = payload.get("reviews", [])
        indexed = {row.get("qid"): row for row in rows}
        if len(rows) != len(indexed) or set(indexed) != set(items):
            raise PermissionError("Independent review does not cover all frozen questions")
        for qid, row in indexed.items():
            check_blind_fields(row)
            if (
                row.get("reviewer_id") != reviewer
                or row.get("input_identity") != external.blind_item_identity(items[qid])
                or row.get("method") not in external.REVIEW_METHODS
                or not row.get("evidence")
                or row.get("used_gold") is not False
                or row.get("used_model_output") is not False
                or row.get("kind") not in {*external.TYPE_KINDS, "UNRESOLVED"}
                or (isinstance(row["evidence"], str) and not row["evidence"].strip())
                or (row["kind"] == "UNRESOLVED" and row.get("comparison") is not None)
                or (row["kind"] in {"YEAR", "COUNT_INT"} and row.get("comparison") != "EXACT")
                or (
                    row["kind"] not in {"UNRESOLVED", "YEAR", "COUNT_INT"}
                    and row.get("comparison") not in {"EXACT", "RELATIVE_5PCT"}
                )
            ):
                raise PermissionError("Independent review input/evidence/rule differs")
        manifest_path = revision / "reviewer_evidence" / (label + "_MANIFEST.json")
        if sha(manifest_path) != audit["reviewer_evidence_manifests"][label]:
            raise PermissionError("Reviewer evidence manifest changed")
        manifest = read(manifest_path)
        records = manifest.get("files", [])
        if (
            manifest.get("reviewer_id") != reviewer
            or not records
            or len({r["path"] for r in records}) != len(records)
        ):
            raise PermissionError("Original reviewer evidence archive is incomplete")
        for row in records:
            evidence = bounded(revision / "reviewer_evidence" / label, row["path"])
            if sha(evidence) != row["sha256"] or evidence.stat().st_size != row["bytes"]:
                raise PermissionError("Original reviewer evidence bytes changed")
        if sum(row["bytes"] for row in records) != manifest.get("total_bytes"):
            raise PermissionError("Reviewer evidence archive byte total differs")
        reviews.append((reviewer, indexed))
    if reviews[0][0] == reviews[1][0] or audit.get("reviewer_ids") != [r[0] for r in reviews]:
        raise PermissionError("Two distinct independent reviewer identities are required")
    scope = {qid: external.blind_item_identity(item) for qid, item in sorted(items.items())}
    blind = read(revision / "BLIND_SCOPE.json")
    if (
        audit.get("annotation_scope_sha256") != digest(scope)
        or blind.get("inputs") != scope
        or (
            blind.get("experiment_id") != EXPERIMENT
            or blind.get("annotation_scope_sha256") != digest(scope)
            or blind.get("gold_accessed_for_type_assignment") is not False
            or blind.get("model_outputs_accessed") is not False
        )
    ):
        raise PermissionError("Blind annotation scope differs")
    for pool, chosen in by_pool.items():
        merged = [r[1][qid] for qid in sorted(chosen) for r in reviews]
        rebuilt = external.build_type_registry(list(chosen.values()), merged, outputs_unseen=True)
        if rebuilt != registries[pool]:
            raise PermissionError("Typed registry is not the exact independent-review merge")
        reconciled = []
        for row in rebuilt["rows"]:
            a, b = (review[1][row["qid"]] for review in reviews)
            reason = (
                "independent_agreement"
                if row["status"] == "RESOLVED"
                else (
                    "both_reviewers_unresolved"
                    if a["kind"] == b["kind"] == "UNRESOLVED"
                    else "independent_disagreement_or_insufficient_evidence"
                )
            )
            reconciled.append(
                dict(
                    {key: row[key] for key in ("qid", "status", "kind", "comparison")},
                    reason=reason,
                )
            )
        if read(revision / "RECONCILIATION.json").get(pool) != reconciled:
            raise PermissionError("Typed reconciliation differs from the independent merge")
    return dict(
        status="VERIFIED_TYPED_V1",
        receipt_sha256=sha(audit_path),
        publication_sha256=sha(revision / "PUBLICATION.json"),
        questions=len(items),
        reviewer_ids=[r[0] for r in reviews],
        auxiliary_code_commit=audit["auxiliary_code_commit"],
        preparation_script_sha256=audit["preparation_script_sha256"],
    )


def state_rows(root):
    rows = []
    for path in sorted((root / "orchestration/jobs").glob("*/INTENT.json")):
        intent = read(path)
        state_path = path.with_name("STATE.json")
        state = (
            read(state_path)
            if state_path.exists()
            else dict(
                key=path.parent.name,
                status="SUBMISSION_UNKNOWN",
                gpu_count=intent["gpu_count"],
                qos=intent["qos"],
            )
        )
        rows.append((intent, state))
    return rows


def active_state_rows(root, api):
    """Exclude only history authenticated by the bound pending-pool handoff."""
    rows = state_rows(root)
    pool = getattr(api, "task_pool", None)
    return pool.filter_rows(rows) if pool is not None and hasattr(pool, "filter_rows") else rows


def runtime_allocation(value, api, directory):
    intent, state = read(directory / "INTENT.json"), read(directory / "STATE.json")
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        intent.get("qos") != value.get("dispatcher_qos", QOS)
        or intent.get("gpu_count") != 0
        or not job.isdigit()
        or state.get("job_id") != job
        or state.get("status") != "SUBMITTED"
        or (
            state.get("intent_sha256") != sha(directory / "INTENT.json")
            or intent.get("operation_identity_sha256") != sha(Path(value["identity_path"]))
        )
    ):
        raise PermissionError("Dispatcher is not its verified Slurm allocation")
    fields = api.slurm.show(job)
    api.orchestration.verify_allocation(fields, dict(intent, job_id=job), value["user"])
    if (
        fields.get("JobState") != "RUNNING"
        or not fields.get("AllocTRES")
        or (
            api.orchestration.physical_gpu_count(fields["AllocTRES"])
            or fields.get("NodeList") != value["cpu_node"]
        )
    ):
        raise PermissionError("Dispatcher requires an actual running CPU-only allocation")
    if sha(intent["script"]) != intent["script_sha256"] or (
        hashlib.sha256(api.slurm.batch_script(job).encode()).hexdigest() != intent["script_sha256"]
        or read(directory / "RELEASE.json").get("returncode") != 0
    ):
        raise PermissionError("Dispatcher script/release verification differs")
    return fields


def duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    if not match:
        raise PermissionError("Actual scheduler lease time is unknown")
    days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


class Driver:
    def __init__(self, value, api, directory, *, typed=verify_typed, execute=subprocess.run):
        self.value, self.api, self.directory, self.typed = value, api, directory, typed
        self.execute = execute
        self.root = Path(value["root"])
        self.signature = None
        self.typed_receipt = None

    def snapshot(self):
        repair = getattr(self.api, "repair", None)
        if repair is not None:
            repair.reconcile_completed()
        rows = active_state_rows(self.root, self.api)
        states = repair.effective_states(rows) if repair is not None else [s for _, s in rows]
        limits = capacity_limits(self.value, self.api)
        queue = self.api.slurm.queue(self.value["user"])
        observed_at = time.time()
        usage = self.api.orchestration.teacher_usage(queue, states, self.value["user"])
        room = capacity_headroom(self.api, usage, limits)
        if room["gpus"] < 0 or room["submitted"] < 0:
            raise PermissionError("Actual teacher capacity exceeds the live scheduler limit")
        known = []
        for intent, state in rows:
            job = state.get("job_id")
            terminal = None
            if job and job not in queue:
                saved = self.root / "orchestration/jobs" / state["key"] / "TERMINAL.json"
                terminal = (
                    read(saved)
                    if state["status"] in {"COMPLETE", "FAILED"} and saved.exists()
                    else self.api.slurm.terminal(job)
                )
            known.append(
                dict(
                    key=state["key"],
                    job_id=job,
                    status=state["status"],
                    kind=intent["kind"],
                    task_ids=intent.get("task_ids", []),
                    terminal=terminal,
                    unknown=state["status"]
                    if state["status"] in self.api.orchestration.UNKNOWN
                    else None,
                )
            )
        typed_path = self.root / REVISION / "PUBLICATION.json"
        snapshot = dict(
            teacher_usage=usage,
            capacity_limits={
                key: limits[key]
                for key in (
                    "gpu_limit",
                    "submitted_limit",
                    "max_tasks_per_job",
                    "gpu_per_user_limit",
                )
                if key in limits
            },
            jobs=known,
            typed_publication_sha256=sha(typed_path) if typed_path.exists() else None,
        )
        save(
            self.directory / "QUEUE_SNAPSHOT.json",
            dict(
                snapshot,
                capacity_evidence=limits,
                queue=queue,
                at=time.time(),
                queue_observed_at=observed_at,
            ),
        )
        return snapshot

    def phase(self, name, **details):
        receipt = dict(phase=name, at=time.time(), **details)
        save(self.directory / "phases" / f"{time.time_ns()}.json", receipt, immutable=True)
        save(self.directory / "PHASE.json", receipt)

    def candidate_kind(self, snapshot):
        """Cheap scheduling hint only; the original submit() owns all task gates."""
        claimed = {task for row in snapshot["jobs"] for task in row["task_ids"]}
        done = {
            task
            for row in snapshot["jobs"]
            if row["status"] == "COMPLETE"
            for task in row["task_ids"]
        }
        tasks = self.api.orchestration.TASKS
        engines = {key for key, row in tasks.items() if row["kind"] == "engine"}
        trains = {key for key, row in tasks.items() if row["kind"] == "train"}
        evaluations = {key for key, row in tasks.items() if row["kind"] == "eval"}
        for key, task in tasks.items():
            if key in claimed:
                continue
            kind = task["kind"]
            if kind == "train" and engines <= done and "p0" in done:
                return kind
            if kind == "eval" and engines <= done and trains <= done:
                return kind
            if kind == "release" and engines <= done and trains <= done and evaluations <= done:
                return kind
        return None

    def has_capacity(self, snapshot, kind):
        limits = snapshot.get("capacity_limits") or capacity_limits(self.value, self.api)
        room = capacity_headroom(self.api, snapshot["teacher_usage"], limits)
        split = self.value.get("task_pool", {}).get("pending_split")
        if kind == "eval" and split is not None:
            claimed = {task for row in snapshot["jobs"] for task in row["task_ids"]}
            unknown_tasks = [
                set(row["task_ids"])
                for row in snapshot["jobs"]
                if row.get("unknown") and row.get("kind") == "eval"
            ]
            group_tasks = [set(group["task_ids"]) for group in split["groups"]]
            if any(tasks not in group_tasks for tasks in unknown_tasks):
                raise PermissionError("Unknown endpoint claim differs from pending split groups")
            remaining = []
            for group in split["groups"]:
                tasks = set(group["task_ids"])
                overlap = tasks.intersection(claimed)
                if overlap and overlap != tasks:
                    raise PermissionError("Pending split group is partially claimed")
                if not overlap or tasks in unknown_tasks:
                    remaining.append(group)
            if not remaining:
                return False
            # fast_science repairs blockers before claiming another group.
            selected = (
                next(
                    group for group in split["groups"] if set(group["task_ids"]) == unknown_tasks[0]
                )
                if unknown_tasks
                else remaining[0]
            )
            required = selected["gpu_count"]
            return (
                room["submitted"] > 0
                and room["gpus"] >= required
                and limits.get("max_tasks_per_job", 1) >= required
            )
        return room["submitted"] > 0 and (
            kind == "release" or (room["gpus"] > 0 and limits.get("max_tasks_per_job", 1) > 0)
        )

    def urgent_capacity(self, snapshot):
        kind = self.candidate_kind(snapshot)
        if kind is None:
            # A positively refused claim remains claimed until its registered repair succeeds.
            kind = next((row["kind"] for row in snapshot["jobs"] if row["unknown"]), None)
        return kind is not None and self.has_capacity(snapshot, kind)

    def fast_science(self, snapshot=None):
        """Observe capacity first; avoid status/tick's repeated full history scans."""
        self.api.efficiency.check()
        snapshot = self.snapshot() if snapshot is None else snapshot
        repair = getattr(self.api, "repair", None)
        submitted = False
        for _ in range(len(self.api.orchestration.TASKS) + 1):
            rows = active_state_rows(self.root, self.api)
            states = [state for _, state in rows]
            blockers = self.api.controller.blocking_attempts(states)
            if blockers:
                if repair is not None and repair.known_rejections_only(states):
                    by_key = {state["key"]: intent for intent, state in rows}
                    kind = by_key[blockers[0]["key"]]["kind"]
                    if not self.has_capacity(snapshot, kind):
                        return "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
                    self.phase("RETRY_EXPLICIT_REJECTION", kind=kind)
                    waiting = repair.retry_pending()
                    if waiting:
                        return waiting
                    snapshot = self.snapshot()
                    continue
                self.phase("VERIFY_RECOVERY")
                status = self.api.controller.status()
                if status["blocked"]:
                    waiting = self.recover(status)
                    if waiting:
                        return waiting
                snapshot = self.snapshot()
                continue
            # A newly terminal allocation must use the original verifier/recovery.
            if any(
                row["terminal"] and row["status"] not in {"COMPLETE", "FAILED"}
                for row in snapshot["jobs"]
            ):
                self.phase("VERIFY_TERMINAL")
                status = self.api.controller.status()
                if status["blocked"]:
                    waiting = self.recover(status)
                    if waiting:
                        return waiting
                snapshot = self.snapshot()
                continue
            kind = self.candidate_kind(snapshot)
            if kind is None:
                done = {
                    task
                    for row in snapshot["jobs"]
                    if row["status"] == "COMPLETE"
                    for task in row["task_ids"]
                }
                if set(self.api.orchestration.TASKS) <= done:
                    self.phase("VERIFY_FINAL_DELIVERY")
                    status = self.api.controller.status()
                    if not status["all_complete"] or status["blocked"]:
                        raise RuntimeError("Final scientific verification is incomplete")
                    return self.deliver()
                return "DISPATCHING_FIXED_TASKS" if submitted else "WAIT_FIXED_TASKS"
            if not self.has_capacity(snapshot, kind):
                return "DISPATCHING_FIXED_TASKS" if submitted else "WAIT_CAPACITY"
            self.require_typed()
            self.phase("SUBMIT_FIXED_TASK", kind=kind, teacher_usage=snapshot["teacher_usage"])
            began = time.monotonic()
            # Frozen ready/claim/complete gates remain in the bound operational
            # submit. GuardedSlurm reads quota immediately before sbatch/release.
            result = self.api.controller.submit(kind)
            self.phase(
                "SUBMIT_RETURNED", kind=kind, seconds=time.monotonic() - began, result=result
            )
            if result is None:
                # A slot can change during verification. Do not acknowledge a
                # headroom snapshot forever: poll() will retry even if unchanged.
                return "WAIT_CAPACITY_RECHECK"
            if result.get("status") != "SUBMITTED":
                if repair is not None and repair.known_rejections_only(
                    [state for _, state in active_state_rows(self.root, self.api)]
                ):
                    return "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
                raise RuntimeError("Submission outcome is unresolved; no duplicate submission")
            submitted = True
            snapshot = self.snapshot()
        raise RuntimeError("Dispatcher made no stable progress through the fixed task matrix")

    def recover(self, status):
        repair = getattr(self.api, "repair", None)
        if repair is not None and repair.known_rejections_only(status["attempts"]):
            return "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
        blockers = self.api.controller.blocking_attempts(status["attempts"])
        if not blockers:
            raise RuntimeError("Blocked status has no authenticated recovery candidates")
        for state in blockers:
            directory = self.root / "orchestration/jobs" / state["key"]
            intent = read(directory / "INTENT.json")
            terminal_path = directory / "TERMINAL.json"
            terminal = read(terminal_path) if terminal_path.exists() else {}
            if intent.get("execution_entry_schema") in {2, 3}:
                # A pool may already contain a successful sibling. Preserve its
                # independent receipt; replaying the allocation would duplicate it.
                evidence = dict(
                    status="HANDOFF_REQUIRED",
                    reason="POOLED_ALLOCATION_REQUIRES_MEMBER_RECONCILIATION",
                    key=state["key"],
                    state=state,
                    terminal=terminal,
                    intent_sha256=sha(directory / "INTENT.json"),
                    task_ids=intent["task_ids"],
                    completed_task_receipts={
                        task: sha(receipt)
                        for task in intent["task_ids"]
                        if (
                            receipt := self.api.orchestration.receipt_path(self.root, task)
                        ).is_file()
                    },
                    resubmitted=False,
                    at=time.time(),
                )
                save(self.directory / "POOL_HANDOFF_REQUIRED.json", evidence, immutable=True)
                return "HANDOFF_REQUIRED"
            if (
                state["status"] != "FAILED"
                or terminal.get("state") != "TIMEOUT"
                or intent["kind"] not in {"train", "eval"}
                or intent["gpu_count"] != 1
                or len(intent["task_ids"]) != 1
            ):
                raise RuntimeError("Unknown or non-recoverable failure: " + state["key"])
            if self.api.orchestration.receipt_path(self.root, intent["task_ids"][0]).exists():
                raise RuntimeError("Existing COMPLETE receipt requires receipt-only reconciliation")
        if (
            not set(status["completed"]) >= ENGINES
            or not (self.root / "EXECUTION_FREEZE.json").exists()
        ):
            raise PermissionError(
                "Science recovery requires completed ENGINE and its existing freeze"
            )
        self.require_typed()
        self.require_extension()
        # Validate all cheap blockers before starting any independent recovery.
        for state in blockers:
            result = self.api.controller.resume(state["key"])
            if result is None:
                return "WAIT_CAPACITY_FOR_VERIFIED_RESUME"
            save(
                self.directory / "recoveries" / (state["key"] + ".json"),
                dict(previous=state["key"], result=result),
                immutable=True,
            )
            if result.get("status") != "SUBMITTED":
                raise RuntimeError("Recovery submission outcome is unresolved")
        return None

    def require_slurm_p0(self, status):
        candidates = []
        for state in status["attempts"]:
            if state["status"] != "COMPLETE":
                continue
            directory = self.root / "orchestration/jobs" / state["key"]
            if read(directory / "INTENT.json").get("kind") == "p0":
                candidates.append(state)
        if len(candidates) != 1 or self.api.controller.verify_job(candidates[0]) != ["p0"]:
            raise PermissionError(
                "Automatic ENGINE dispatch requires a real completed P0 allocation"
            )
        state = candidates[0]
        # The frozen controller also accepts LOCAL_P0 and saved terminal evidence.
        # This operational entry requires the full allocation chain plus live accounting.
        terminal = self.api.slurm.terminal(state["job_id"])
        saved = self.root / "orchestration/jobs" / state["key"] / "TERMINAL.json"
        if terminal is None or terminal != read(saved):
            raise PermissionError("Completed P0 live accounting is missing or differs")

    def act(self):
        if (
            getattr(self.api, "efficiency", None) is not None
            and (self.root / "EXECUTION_FREEZE.json").exists()
        ):
            return self.fast_science()
        repair = getattr(self.api, "repair", None)
        if repair is not None:
            waiting = repair.retry_pending()
            if waiting:
                return waiting
        status = self.api.controller.status()
        if status["blocked"]:
            waiting = self.recover(status)
            if waiting:
                return waiting
            status = self.api.controller.status()
            if status["blocked"]:
                raise RuntimeError("Recovery left an unresolved scheduler state")
        freeze = self.root / "EXECUTION_FREEZE.json"
        if not freeze.exists():
            if "p0" in status["completed"]:
                self.require_slurm_p0(status)
            if not set(status["completed"]) >= ENGINES:
                submitted = False
                if "p0" in status["completed"]:
                    # The frozen controller owns task selection, fresh quota
                    # checks, and the held/verified/released submission chain.
                    for _ in range(len(ENGINES - {"p0"})):
                        if not (ENGINES - {"p0"}).intersection(status["ready"]):
                            break
                        result = self.api.controller.submit("engine")
                        if result is None:
                            break
                        if result.get("status") != "SUBMITTED":
                            raise RuntimeError("ENGINE submission outcome is unresolved")
                        submitted = True
                        status = self.api.controller.status()
                        if status["blocked"]:
                            raise RuntimeError(
                                "ENGINE submission left an unresolved scheduler state"
                            )
                return "DISPATCHING_ENGINE" if submitted else "WAIT_ENGINE"
            verified = self.typed(self.root, self.api.external)
            if verified is None:
                return "WAIT_TYPED_REVIEW"
            self.typed_receipt = verified
            self.check_typed_source(verified)
            save(self.directory / "TYPED_VERIFIED.json", verified, immutable=True)
            intent_path = self.directory.parent.parent / "FREEZE_INTENT.json"
            if intent_path.exists():
                raise RuntimeError("A previous scientific freeze attempt needs reconciliation")
            save(intent_path, dict(typed=verified), immutable=True)
            self.api.extension.seal_amendment(
                self.root, self.value["extension_manifest_path"], typed_verified=verified
            )
            self.api.bootstrap.freeze_science(self.root)
            save(
                self.directory.parent.parent / "FREEZE_RECEIPT.json",
                dict(sha256=sha(freeze)),
                immutable=True,
            )
        else:
            self.require_typed()
        self.require_extension()
        if status["all_complete"]:
            return self.deliver()
        status = self.api.controller.tick()
        if status["blocked"]:
            waiting = self.recover(status)
            if waiting:
                return waiting
            status = self.api.controller.status()
            if status["blocked"]:
                raise RuntimeError("Recovery left an unresolved scheduler state")
        return self.deliver() if status["all_complete"] else "DISPATCHING_FIXED_TASKS"

    def require_typed(self):
        if self.typed_receipt is None:
            verified = self.typed(self.root, self.api.external)
            if verified is None:
                raise PermissionError("Scientific freeze has no completed independent type review")
            self.check_typed_source(verified)
            save(self.directory / "TYPED_VERIFIED.json", verified, immutable=True)
            self.typed_receipt = verified

    def check_typed_source(self, verified):
        source = self.root / "operations" / TYPED_COMMIT / "finalize_external_types.py"
        if verified.get("auxiliary_code_commit") != TYPED_COMMIT or (
            verified.get("preparation_script_sha256") != TYPED_SCRIPT_SHA
            or sha(source) != TYPED_SCRIPT_SHA
        ):
            raise PermissionError(
                "Typed audit differs from the explicitly retained published preparer"
            )

    def require_extension(self):
        checked = self.api.extension.verify_extension(
            self.root, self.value["extension_manifest_path"], science=True
        )
        if checked["manifest_sha256"] != self.value["extension_manifest_sha256"]:
            raise PermissionError("Operation identity names another observability extension")
        return checked

    def deliver(self):
        extension = self.require_extension()
        operation = Path(self.value["identity_path"]).parent
        output = self.root / "delivery/SR_F3_CE_20261016.zip"
        manifest = operation / "OPERATION_SOURCES.json"
        command = [
            self.value["python"],
            str(operation / "build_delivery.py"),
            "--root",
            str(self.root),
            "--output",
            str(output),
            "--helper-manifest",
            str(manifest),
            "--extension-manifest",
            self.value["extension_manifest_path"],
        ]
        if "task_pool" in self.value:
            command += ["--operation-identity", self.value["identity_path"]]
        save(
            self.directory / "DELIVERY_INTENT.json",
            dict(status="NUMERICAL_RELEASE_COMPLETE", command=command),
            immutable=True,
        )
        result = self.execute(command, text=True, capture_output=True, check=False)
        save(
            self.directory / "DELIVERY_PROCESS.json",
            dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
            immutable=True,
        )
        if result.returncode:
            raise RuntimeError("NUMERICAL_RELEASE_COMPLETE; DELIVERY_REQUIRED: builder failed")
        receipt_path = Path(str(output) + ".verification.json")
        receipt = read(receipt_path)
        observed_identity = {
            key: extension[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "extension_commit",
                "base_code_commit",
                "observer_source_sha256",
                "amendment_sha256",
                "cpu_validation_sha256",
            )
        }
        if "protocol_source_guard" in extension:
            observed_identity["protocol_source_guard"] = extension["protocol_source_guard"]
        observed_identity["manifest_path"] = str(
            Path(extension["manifest_path"]).relative_to(self.root)
        )
        required = dict(
            schema_version=1,
            status="VERIFIED_DELIVERY",
            run_identity_sha256=self.value["run_identity_sha256"],
            scientific_source_manifest_sha256=self.value["source_manifest_sha256"],
            helper_manifest_sha256=sha(manifest),
            observability_extension=observed_identity,
        )
        if "task_pool" in self.value:
            required["task_pool_operation"] = dict(
                operation_identity_path=str(
                    Path(self.value["identity_path"]).relative_to(self.root)
                ),
                operation_identity_sha256=sha(self.value["identity_path"]),
                operation_commit=self.value["operation_commit"],
                task_pool=self.value["task_pool"],
            )
        if any(receipt.get(k) != v for k, v in required.items()):
            raise PermissionError("Delivery receipt identity differs")
        archive = receipt["archive"]
        if (
            archive.get("path") != str(output)
            or archive.get("sha256") != sha(output)
            or (
                archive.get("bytes") != output.stat().st_size
                or type(archive.get("entries")) is not int
                or archive["entries"] <= 0
            )
        ):
            raise PermissionError("Verified delivery archive identity differs")
        expected = {
            "crc",
            "safe_unique_names",
            "payload_hashes",
            "manifest_hashes",
            "entry_links",
            "gzip_read_to_eof",
            "raw_rescoring_chain",
        }
        if any(receipt.get("checks", {}).get(key) != "PASS" for key in expected):
            raise PermissionError("Delivery verification is incomplete")
        panels = read(self.root / "external/PANELS.json")
        pro = read(bounded(self.root, panels["CHARTQAPRO"]["root"]) / "FREEZE.json")
        count = pro["image_count"]
        if type(count) is not int or not 0 <= count <= 256:
            raise PermissionError("Delivery external panel count is unregistered")
        coverage = dict(
            verified_tasks=33,
            training_paths=12,
            evaluated_models=15,
            logical_updates=768,
            training_records=98304,
            support_examples=2304,
            monitor_records=26112,
            engine_diagnostic_records=2880,
            engine_new_rollouts=512,
            engine_replayed_rollouts=256,
            engine_physical_updates=6,
            records=229500 + 15 * count,
        )
        if any(
            type(receipt.get("coverage", {}).get(k)) is not int or receipt["coverage"][k] != v
            for k, v in coverage.items()
        ):
            raise PermissionError("Delivery coverage does not contain every registered task")
        save(
            self.directory / "DELIVERY_VERIFIED.json",
            dict(receipt=str(receipt_path), sha256=sha(receipt_path)),
            immutable=True,
        )
        return "DONE"

    def poll(self):
        snapshot = self.snapshot()
        signature = digest(snapshot)
        fast = (
            getattr(self.api, "efficiency", None) is not None
            and (self.root / "EXECUTION_FREEZE.json").exists()
        )
        if signature == self.signature and not (fast and self.urgent_capacity(snapshot)):
            return "UNCHANGED"
        result = self.fast_science(snapshot) if fast else self.act()
        # Keep the observed pre-action state: a terminal transition during a slow
        # verification must trigger the next pass, not be silently acknowledged.
        self.signature = signature
        save(
            self.directory / "PROGRESS.json",
            dict(status=result, at=time.time(), snapshot_sha256=self.signature),
        )
        return result


def attempts_root(value):
    return Path(value["root"]) / "operations/dispatcher/attempts"


def dispatcher_limits(value, api):
    if value.get("dispatcher_qos", QOS) == QOS:
        return capacity_limits(value, api, fresh=True)
    return api.capacity.dispatcher_limits(value["dispatcher_qos"])


def dispatcher_headroom(value, api, queue, states, limits):
    if value.get("dispatcher_qos", QOS) == QOS:
        usage = api.orchestration.teacher_usage(queue, states, value["user"])
        return usage, capacity_headroom(api, usage, limits)
    room = api.capacity.dispatcher_headroom(queue, limits)
    return room, dict(room, gpus=0)


def submit_driver(value, api, sequence, previous=None):
    if (
        type(sequence) is not int
        or sequence < 1
        or previous != (sequence - 1 if sequence > 1 else None)
    ):
        raise PermissionError("Driver sequence must identify its immediate predecessor")
    base = attempts_root(value)
    with (
        lock(base.parent / "SUBMIT.lock"),
        api.orchestration.lease(Path(value["root"]) / "orchestration"),
    ):
        directory = base / f"{sequence:04d}"
        if (directory / "INTENT.json").exists():
            raise RuntimeError(
                "Driver submission already attempted; observe/reconcile, never resubmit"
            )
        existing = sorted(int(p.name) for p in base.glob("[0-9][0-9][0-9][0-9]") if p.is_dir())
        if existing != list(range(1, sequence)):
            raise PermissionError("Driver predecessor history has a gap or a fork")
        queue = api.slurm.queue(value["user"])
        if previous is not None:
            prior = base / f"{previous:04d}"
            if (prior / "POOL_HANDOFF_REQUIRED.json").exists():
                raise PermissionError(
                    "Pooled member failure requires reconciliation before continuation"
                )
            old_intent, old_state = read(prior / "INTENT.json"), read(prior / "STATE.json")
            repair = getattr(api, "repair", None)
            repaired_terminal = None
            if (
                old_intent.get("operation_identity_sha256") != sha(value["identity_path"])
                and "dispatch_efficiency" in value
            ):
                repaired_terminal = verify_efficiency_predecessor(
                    value, api, prior, old_intent, old_state, queue
                )
            elif (
                old_intent.get("operation_identity_sha256") != sha(value["identity_path"])
                and repair is not None
            ):
                repaired_terminal = repair.verify_driver_predecessor(
                    prior, old_intent, old_state, queue
                )
            elif old_intent.get("operation_identity_sha256") != sha(value["identity_path"]) or (
                old_state.get("intent_sha256") != sha(prior / "INTENT.json")
            ):
                raise PermissionError("Predecessor operation identity changed")
            job = old_state.get("job_id")
            if not job or job in queue or old_state.get("status") != "SUBMITTED":
                raise RuntimeError("Predecessor is active or its submission is unknown")
            terminal = repaired_terminal or api.slurm.terminal(job)
            expected = dict(
                job_id=job,
                user=value["user"],
                account="rose",
                qos=old_intent["qos"],
                name=old_intent["job_name"],
            )
            if (
                not terminal
                or (
                    terminal.get("state") not in {"COMPLETED", "TIMEOUT"}
                    and repaired_terminal is None
                )
                or any(terminal.get(k) != v for k, v in expected.items())
            ):
                raise RuntimeError("Predecessor terminal identity requires reconciliation")
            if (
                terminal["state"] == "COMPLETED"
                and read(prior / "END.json").get("status") != "HANDOFF_REQUIRED"
            ):
                raise PermissionError(
                    "Only a declared lease handoff may continue a completed driver"
                )
            if (
                terminal["state"] == "TIMEOUT"
                and (prior / "END.json").exists()
                and (read(prior / "END.json").get("status") != "HANDOFF_REQUIRED")
            ):
                raise PermissionError("A recorded driver failure requires reconciliation")
            save(prior / "TERMINAL.json", terminal, immutable=True)
        repair = getattr(api, "repair", None)
        rows = active_state_rows(Path(value["root"]), api)
        science_states = (
            repair.effective_states(rows) if repair is not None else [s for _, s in rows]
        )
        limits = dispatcher_limits(value, api)
        queue = api.slurm.queue(value["user"])
        usage, room = dispatcher_headroom(value, api, queue, science_states, limits)
        if room["gpus"] < 0 or room["submitted"] <= 0:
            raise RuntimeError(
                "No submission slot in the dispatcher QoS under live scheduler limits"
            )
        save(
            directory / "PRE_SUBMIT_CAPACITY.json",
            dict(limits=limits, queue=queue, usage=usage, queue_observed_at=time.time()),
            immutable=True,
        )
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / "run.sbatch"
        command = [
            value["python"],
            str(Path(__file__).resolve()),
            "--mode",
            "run",
            "--identity",
            value["identity_path"],
            "--sequence",
            str(sequence),
        ]
        body = (
            "#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
        )
        body += "export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
        body += "OPENBLAS_NUM_THREADS=4\n"
        body += "exec " + shlex.join(command) + "\n"
        with script.open("x") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        intent = dict(
            sequence=sequence,
            previous=previous,
            gpu_count=0,
            cpus=4,
            memory_gib=12,
            account="rose",
            qos=value.get("dispatcher_qos", QOS),
            partition="cluster02",
            cpu_node="cpu-2",
            job_name=f"SRF3CE-DISPATCH-N{sequence:04d}-LH",
            script=str(script),
            script_sha256=sha(script),
            operation_identity_sha256=sha(value["identity_path"]),
            comment="srf3ce-dispatch-" + digest([sha(value["identity_path"]), sequence])[:20],
        )
        args = [
            "sbatch",
            "--hold",
            "--parsable",
            "--no-requeue",
            "--open-mode=append",
            "--account=rose",
            "--partition=cluster02",
            "--qos=" + intent["qos"],
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--mem=12G",
            "--time=4320",
            "--nodelist=cpu-2",
            "--job-name=" + intent["job_name"],
            "--comment=" + intent["comment"],
            "--output=" + str(directory / "slurm-%j.log"),
            str(script),
        ]
        intent["sbatch"] = args
        save(directory / "INTENT.json", intent, immutable=True)
        state = dict(status="SUBMISSION_UNKNOWN", intent_sha256=sha(directory / "INTENT.json"))
        save(directory / "STATE.json", state)
        result = api.slurm.run(args)
        save(
            directory / "SUBMISSION.json",
            dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr),
            immutable=True,
        )
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
        if result.returncode or not match:
            raise RuntimeError("CPU submission outcome unknown; no automatic resubmission")
        job = match[1]
        state.update(status="HELD", job_id=job)
        save(directory / "STATE.json", state)
        fields = api.slurm.show(job)
        api.orchestration.verify_allocation(
            fields, dict(intent, job_id=job), value["user"], held=True
        )
        if (
            hashlib.sha256(api.slurm.batch_script(job).encode()).hexdigest()
            != intent["script_sha256"]
        ):
            raise PermissionError("Slurm stored dispatcher script differs")
        limits = dispatcher_limits(value, api)
        queue = api.slurm.queue(value["user"])
        rows = active_state_rows(Path(value["root"]), api)
        states = repair.effective_states(rows) if repair is not None else [s for _, s in rows]
        usage, room = dispatcher_headroom(value, api, queue, states, limits)
        if room["gpus"] < 0 or room["submitted"] < 0:
            raise PermissionError("CPU driver remains held because live scheduler capacity changed")
        save(
            directory / "PRE_RELEASE_CAPACITY.json",
            dict(limits=limits, queue=queue, usage=usage, queue_observed_at=time.time()),
            immutable=True,
        )
        save(
            directory / "ALLOCATION_VERIFIED.json",
            dict(job_id=job, fields=fields, script_sha256=intent["script_sha256"]),
            immutable=True,
        )
        save(directory / "RELEASE_INTENT.json", dict(job_id=job), immutable=True)
        state["status"] = "RELEASE_UNKNOWN"
        save(directory / "STATE.json", state)
        result = api.slurm.run(["scontrol", "release", job])
        save(
            directory / "RELEASE.json",
            dict(
                job_id=job, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            ),
            immutable=True,
        )
        if result.returncode:
            raise RuntimeError("CPU release outcome unknown; no automatic release retry")
        state["status"] = "SUBMITTED"
        save(directory / "STATE.json", state)
        return state


def run(value, api, sequence):
    directory = attempts_root(value) / f"{sequence:04d}"
    with lock(directory.parent.parent / "DRIVER.lock"):
        if (directory / "STARTED.json").exists():
            raise PermissionError("A driver process cannot restart inside the same allocation")
        driver = Driver(value, api, directory)
        try:
            # scontrol may start a job before its successful release reply is persisted.
            deadline = time.monotonic() + 15
            while (
                not (directory / "RELEASE.json").exists()
                or read(directory / "STATE.json").get("status") != "SUBMITTED"
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError("CPU release acknowledgment is unresolved")
                time.sleep(0.1)
            runtime_allocation(value, api, directory)
            save(
                directory / "STARTED.json",
                dict(job_id=os.environ["SLURM_JOB_ID"], at=time.time()),
                immutable=True,
            )
            efficiency = getattr(api, "efficiency", None)
            if efficiency is not None:
                driver.phase("STARTUP_VALIDATION")
                save(directory / "EFFICIENCY_STARTED.json", efficiency.start(), immutable=True)
                driver.phase("STARTUP_READY")
            else:
                api.protocol.verify_run(Path(value["root"]))
            while True:
                if (Path(value["root"]) / "STOP").exists():
                    raise PermissionError("SR-F3-CE STOP exists")
                if efficiency is not None:
                    efficiency.check()
                else:
                    current = identity(value["identity_path"])
                    if any(current.get(k) != v for k, v in value.items() if k != "identity_path"):
                        raise PermissionError("Operation identity changed while running")
                fields = runtime_allocation(value, api, directory)
                remaining = duration(fields["TimeLimit"]) - duration(fields["RunTime"])
                if remaining <= value["handoff_margin_seconds"]:
                    result = "HANDOFF_REQUIRED"
                    break
                result = driver.poll()
                if result in {"DONE", "HANDOFF_REQUIRED"}:
                    break
                time.sleep(value["poll_seconds"])
            save(
                directory / "END.json",
                dict(status=result, at=time.time(), scientific_time_budget_changed=False),
                immutable=True,
            )
            return result
        except Exception as error:
            save(
                directory / "END.json",
                dict(status="FAILED", error=repr(error), at=time.time()),
                immutable=True,
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("register", "reconcile", "submit", "run"), required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--user")
    parser.add_argument("--operation-commit")
    parser.add_argument("--extension-manifest", type=Path)
    parser.add_argument("--repair-authorization", type=Path)
    parser.add_argument("--efficiency-authorization", type=Path)
    parser.add_argument("--pool-authorization", type=Path)
    parser.add_argument("--dispatcher-qos", choices=(QOS, "rose"), default=QOS)
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--previous", type=int)
    args = parser.parse_args()
    if args.mode == "register":
        if not all(
            (args.root, args.python, args.user, args.operation_commit, args.extension_manifest)
        ):
            parser.error(
                "register requires root, python, user, operation-commit and extension-manifest"
            )
        result = str(
            register(
                args.root,
                args.python,
                args.user,
                args.operation_commit,
                args.extension_manifest,
                args.repair_authorization,
                args.efficiency_authorization,
                args.pool_authorization,
                args.dispatcher_qos,
            )
        )
    else:
        if args.identity is None or (args.mode != "reconcile" and args.sequence is None):
            parser.error("identity and (for submit/run) sequence are required")
        value = identity(args.identity)
        value["identity_path"] = str(args.identity.resolve(strict=True))
        # Full data/checkpoint validation belongs inside the allocated CPU job.
        # Submission checks only the small identity/config/source inventory.
        api = load_science(value)
        if args.mode == "reconcile":
            if not hasattr(api, "repair"):
                raise PermissionError("Explicit operational repair registration is required")
            result = api.repair.prepare()
        else:
            result = (
                submit_driver(value, api, args.sequence, args.previous)
                if args.mode == "submit"
                else run(value, api, args.sequence)
            )
    print(json.dumps(dict(result=result), sort_keys=True))


if __name__ == "__main__":
    main()
