#!/usr/bin/env python3
"""Explicit, manifest-bound observation launcher for the run-bound SR-F3-CE core.

The run-bound core verifies the scientific implementation. This extension separately
verifies its own source, prospective amendment, real commands and observation
receipts. It never presents an instrumented command as the old run_worker path.
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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPERIMENT = "SR-F3-CE-20261016"
BASE_COMMIT = "71483fbf31d57213fdc7e6f3c881848397b45329"
TYPED_COMMIT = "4146e156ffa9a476b360daeb67503703faca138f"
TYPED_SCRIPT_SHA = "1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf"
ENGINE_IDS = {"p0", "engine--b0", "engine--b1", "engine--b2", "engine--resume"}
SCIENCE = {"train", "eval", "release"}
_CONTEXT = None
KINDS: Any = None
TASKS: Any = None
TEACHER_QOS: Any = None
GPU_LIMIT: Any = None
SUBMIT_LIMIT: Any = None
lease: Any = None
object_hash: Any = None
teacher_usage: Any = None
now: Any = None
verify_freeze: Any = None
read_json: Any = None
file_hash: Any = None
save_json: Any = None
verify_allocation: Any = None
visible_devices: Any = None
verify_task: Any = None
run_identity: Any = None
runtime_resume_receipt: Any = None


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _bounded(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Unsafe extension artifact path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise PermissionError("Extension artifact escapes its root")
    return path


def _check_file(root, record):
    path = _bounded(root, record["path"])
    if (
        type(record.get("bytes")) is not int
        or path.stat().st_size != record["bytes"]
        or (_sha(path) != record.get("sha256"))
    ):
        raise PermissionError("Extension artifact identity changed: " + record["path"])
    return path


def _reject_bytecode(code):
    sys.dont_write_bytecode = True
    if any(path.is_file() and path.suffix in {".pyc", ".pyo"} for path in Path(code).rglob("*")):
        raise PermissionError("Unbound scientific bytecode cache must not be imported")


def _source_revision(root):
    """Keep v1 pinned; authenticate the exact adopted revision before importing it."""
    root = Path(root).resolve(strict=True)
    registration = _read(root / "RUN_IDENTITY.json")
    config_path = root / "plan/config/SR_F3_CE.json"
    config = _read(config_path) if config_path.is_file() else {}
    revision = config.get("protocol_revision", "v1")
    if revision == "v1":
        if registration.get("code_commit") != BASE_COMMIT:
            raise PermissionError(
                "Original protocol identity changed: frozen scientific commit required"
            )
        return dict(code_commit=BASE_COMMIT, protocol_revision="v1")
    if revision not in {"v2", "c2"}:
        raise PermissionError("Unregistered protocol revision")
    code = root / "code"
    sys.dont_write_bytecode = True
    if any(p.is_file() and p.suffix in {".pyc", ".pyo"} for p in code.rglob("*")):
        raise PermissionError("Unbound scientific bytecode cache must not be imported")
    manifest = _read(root / "deployment/SOURCE_MANIFEST.json")
    rows = manifest.get("files", [])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise PermissionError("Revised source inventory is empty or ambiguous")
    for row in rows:
        source = _bounded(code, row["path"])
        if source.stat().st_size != row["bytes"] or _sha(source) != row["sha256"]:
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


def _revision_binding(checked):
    return {} if checked["protocol_revision"] == "v1" else {"protocol_source_guard": checked}


def verify_extension(root, manifest_path, *, science=False):
    """Validate static extension bytes; science additionally requires the full freeze chain."""
    root = Path(root).resolve(strict=True)
    _reject_bytecode(root / "code")
    checked_source = _source_revision(root)
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _read(manifest_path)
    commit = manifest.get("extension_commit", "")
    expected_directory = root / "native/observability" / commit
    if not re.fullmatch(r"[a-f0-9]{40}", commit) or (
        manifest_path != expected_directory / "SOURCE_MANIFEST.json"
        or Path(__file__).resolve() != expected_directory / "instrumented_execution.py"
    ):
        raise PermissionError("Extension must be loaded from its explicit immutable deployment")
    required = dict(
        schema_version=1,
        experiment_id=EXPERIMENT,
        status="FROZEN_OBSERVABILITY_EXTENSION",
        base_code_commit=checked_source["code_commit"],
        **_revision_binding(checked_source),
        run_identity_sha256=_sha(root / "RUN_IDENTITY.json"),
        base_source_manifest_sha256=_sha(root / "deployment/SOURCE_MANIFEST.json"),
    )
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in required.items()):
        raise PermissionError("Extension base/source identity differs")
    registration = _read(root / "RUN_IDENTITY.json")
    if (
        registration.get("experiment_id") != EXPERIMENT
        or registration.get("code_commit") != checked_source["code_commit"]
    ):
        raise PermissionError("Extension cannot replace the registered scientific source")
    sources = manifest.get("sources", [])
    if not isinstance(sources, list) or len({r.get("path") for r in sources}) != len(sources):
        raise PermissionError("Extension source inventory is ambiguous")
    names = {r["path"] for r in sources}
    if not {"instrumented_execution.py", "observe_runtime.py"} <= names or (
        {"SOURCE_MANIFEST.json", "CPU_VALIDATION.json", "AMENDMENT.json"} & names
    ):
        raise PermissionError("Extension source inventory is incomplete or self-referential")
    if manifest.get("executed_sources") != ["instrumented_execution.py", "observe_runtime.py"]:
        raise PermissionError("Only the two explicitly declared runtime sources may execute")
    for row in sources:
        _check_file(expected_directory, row)
    validation = manifest.get("cpu_validation", {})
    if validation.get("path") != "CPU_VALIDATION.json":
        raise PermissionError("Explicit CPU-only acceptance evidence required")
    cpu_path = _check_file(expected_directory, validation)
    cpu = _read(cpu_path)
    expected_cpu = dict(
        status="PASS",
        scope="CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION",
        gpu_validation_performed=False,
        new_model_generations=0,
        new_optimizer_steps=0,
        source_sha256={r["path"]: r["sha256"] for r in sources},
    )
    if any(type(cpu.get(k)) is not type(v) or cpu[k] != v for k, v in expected_cpu.items()) or (
        type(cpu.get("tests_passed")) is not int or cpu["tests_passed"] < 1
    ):
        raise PermissionError(
            "CPU transparency evidence is incomplete or overclaims GPU validation"
        )
    result = dict(
        root=str(root),
        manifest_path=str(manifest_path),
        manifest_sha256=_sha(manifest_path),
        extension_commit=commit,
        base_code_commit=checked_source["code_commit"],
        **_revision_binding(checked_source),
        observer_source_sha256=_sha(expected_directory / "observe_runtime.py"),
        run_identity_sha256=required["run_identity_sha256"],
        base_source_manifest_sha256=required["base_source_manifest_sha256"],
        cpu_validation_sha256=_sha(cpu_path),
    )
    if _CONTEXT is not None and (
        result["root"] != _CONTEXT.root
        or result["manifest_path"] != _CONTEXT.manifest
        or result["manifest_sha256"] != _CONTEXT.manifest_sha256
        or _CONTEXT.api.orchestration.Controller is not _CONTEXT.controller
    ):
        raise PermissionError("Installed extension identity changed")
    amendment_path = expected_directory / "AMENDMENT.json"
    if amendment_path.exists():
        amendment = _read(amendment_path)
        expected = dict(
            status="SEALED_BEFORE_SCIENCE",
            experiment_id=EXPERIMENT,
            extension_manifest_sha256=result["manifest_sha256"],
            run_identity_sha256=result["run_identity_sha256"],
            config_sha256=_sha(root / "orchestration/CONFIG.json"),
            scientific_code_commit=checked_source["code_commit"],
            **_revision_binding(checked_source),
            extension_commit=commit,
            scientific_matrix_changed=False,
            new_model_generations=0,
            new_optimizer_steps=0,
            repeat_engine=False,
        )
        if any(
            type(amendment.get(k)) is not type(v) or amendment[k] != v for k, v in expected.items()
        ):
            raise PermissionError("Prospective observation amendment identity differs")
        expected_engine = {
            task: _sha(root / "tasks" / task / "COMPLETE.json") for task in ENGINE_IDS
        }
        if amendment.get("engineering_receipts") != expected_engine:
            raise PermissionError("Observation amendment changed current run engineering evidence")
        if _CONTEXT is not None and amendment.get("scientific_matrix_sha256") != _digest(
            list(TASKS.values())
        ):
            raise PermissionError("Observation amendment changed the registered task matrix")
        typed = root / "external/revisions/typed_v1"
        if amendment.get("typed_publication_sha256") != _sha(typed / "PUBLICATION.json") or (
            amendment.get("typed_receipt_sha256") != _sha(typed / "TYPE_AUDIT_RECEIPT.json")
            or amendment.get("scientific_config_sha256") != _sha(root / "plan/config/SR_F3_CE.json")
        ):
            raise PermissionError("Amendment scientific settings or typed publication changed")
        result["amendment_sha256"] = _sha(amendment_path)
    if science:
        if not amendment_path.exists():
            raise PermissionError("Science requires the sealed observation amendment")
        freeze = _read(root / "EXECUTION_FREEZE.json")
        if freeze.get("status") != "FROZEN_BEFORE_SCIENCE" or (
            freeze.get("run_identity_sha256") != result["run_identity_sha256"]
        ):
            raise PermissionError("Observation extension has no prospective scientific freeze")
        for path in (manifest_path, amendment_path, cpu_path):
            if freeze.get("engineering_files", {}).get(str(path.relative_to(root))) != _sha(path):
                raise PermissionError("Scientific freeze omits the full extension static chain")
        manifests = [
            name
            for name in freeze.get("engineering_files", {})
            if re.fullmatch(r"native/observability/[a-f0-9]{40}/SOURCE_MANIFEST\.json", name)
        ]
        if manifests != [str(manifest_path.relative_to(root))]:
            raise PermissionError("Science requires exactly one explicit observability manifest")
        result["execution_freeze_sha256"] = _sha(root / "EXECUTION_FREEZE.json")
    return result


def _import_base(root):
    root = Path(root)
    code = (root / "code").resolve(strict=True)
    _reject_bytecode(code)
    manifest = _read(root / "deployment/SOURCE_MANIFEST.json")
    for row in manifest["files"]:
        _check_file(code, row)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code / "src"))
    modules = {}
    for name in ("protocol", "orchestration"):
        module = importlib.import_module("sr_f3_ce." + name)
        if Path(module.__file__).resolve() != code / "src/sr_f3_ce" / (name + ".py"):
            raise PermissionError("Extension imported another scientific implementation")
        modules[name] = module
    return SimpleNamespace(**modules)


def install(root, manifest_path):
    """Install one explicit Controller class; do not install model hooks on the CPU driver."""
    global _CONTEXT
    checked = verify_extension(root, manifest_path)
    if _CONTEXT is not None:
        if (
            checked["root"] != _CONTEXT.root
            or checked["manifest_path"] != _CONTEXT.manifest
            or (
                checked["manifest_sha256"] != _CONTEXT.manifest_sha256
                or _CONTEXT.api.orchestration.Controller is not _CONTEXT.controller
            )
        ):
            raise PermissionError("Only one explicit extension/root/source may be installed")
        return _CONTEXT.controller
    api = _import_base(checked["root"])
    original = api.orchestration.Controller
    if original.__module__ != "sr_f3_ce.orchestration":
        raise PermissionError("An unrelated Controller replacement is already installed")
    aliases = (
        "KINDS",
        "TASKS",
        "TEACHER_QOS",
        "GPU_LIMIT",
        "SUBMIT_LIMIT",
        "lease",
        "object_hash",
        "teacher_usage",
        "now",
        "verify_freeze",
        "read_json",
        "file_hash",
        "save_json",
        "verify_allocation",
        "visible_devices",
        "verify_task",
        "run_identity",
        "runtime_resume_receipt",
    )
    globals().update({name: getattr(api.orchestration, name) for name in aliases})

    class InstrumentedController(_ExecutionMethods, original):
        pass

    _CONTEXT = SimpleNamespace(
        root=checked["root"],
        manifest=checked["manifest_path"],
        manifest_sha256=checked["manifest_sha256"],
        api=api,
        original_controller=original,
        controller=InstrumentedController,
    )
    api.orchestration.Controller = InstrumentedController
    return InstrumentedController


def _context(root):
    if _CONTEXT is None or str(Path(root).resolve(strict=True)) != _CONTEXT.root:
        raise PermissionError("Explicit installation for this run is required")
    return _CONTEXT


def _job_command(root, key, python):
    context = _context(root)
    return [
        python,
        str(Path(__file__).resolve()),
        "--mode",
        "job",
        "--root",
        str(root),
        "--manifest",
        context.manifest,
        "--job-key",
        key,
    ]


def _worker_command(root, task_id, python):
    context = _context(root)
    return [
        python,
        str(Path(__file__).resolve()),
        "--mode",
        "worker",
        "--root",
        str(root),
        "--manifest",
        context.manifest,
        "--task-id",
        task_id,
    ]


def _script_body(root, code, command):
    body = "#!/bin/bash\nset -euo pipefail\n"
    body += "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
    body += "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
    body += "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
    body += "export PYTHONPATH=" + shlex.quote(str(code / "src")) + "\n"
    body += "cd " + shlex.quote(str(code)) + "\n"
    body += "test ! -e " + shlex.quote(str(root / "STOP")) + "\n"
    return body + "exec " + shlex.join(command) + "\n"


def _sbatch_args(intent):
    args = [
        "sbatch",
        "--hold",
        "--parsable",
        "--no-requeue",
        "--open-mode=append",
        "--account=rose",
        "--partition=" + intent["partition"],
        "--qos=" + intent["qos"],
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={intent['cpus']}",
        f"--mem={intent['memory_gib']}G",
        "--time=4320",
        "--job-name=" + intent["job_name"],
        "--comment=" + intent["comment"],
        "--output=" + str(Path(intent["script"]).parent / "slurm-%j.log"),
    ]
    args += (
        [f"--gres=gpu:pro6000:{intent['gpu_count']}", "--constraint=highmem"]
        if intent["gpu_count"]
        else (["--nodelist=" + intent["cpu_node"]] if intent["cpu_node"] else [])
    )
    return [*args, intent["script"]]


def _verify_intent(root, intent):
    root = Path(root).resolve(strict=True)
    context = _context(root)
    checked = verify_extension(root, context.manifest, science=True)
    config = _read(root / "orchestration/CONFIG.json")
    if not re.fullmatch(r"job-[a-f0-9]{16}", intent.get("key", "")) or (
        intent.get("kind") not in SCIENCE
        or not isinstance(intent.get("task_ids"), list)
        or len(intent["task_ids"]) != 1
        or intent["task_ids"][0] not in TASKS
        or config.get("pool_size") != 1
    ):
        raise PermissionError("Future science requires one registered task per allocation")
    gpu = int(intent["kind"] in {"train", "eval"})
    expected = dict(
        execution_entry_schema=1,
        extension_manifest_path=context.manifest,
        extension_manifest_sha256=checked["manifest_sha256"],
        amendment_sha256=checked["amendment_sha256"],
        execution_freeze_sha256=checked["execution_freeze_sha256"],
        job_entry_command=_job_command(root, intent["key"], intent["python"]),
        worker_entry_commands={
            task: _worker_command(root, task, intent["python"]) for task in intent["task_ids"]
        },
    )
    expected.update(
        root=str(root),
        code_root=str(root / "code"),
        python=config["python"],
        user=config["user"],
        run_identity_sha256=checked["run_identity_sha256"],
        gpu_count=gpu,
        cpus=4,
        memory_gib=90 if gpu else 12,
        partition="cluster02" if gpu else config["cpu_policy"]["partition"],
        qos=TEACHER_QOS if gpu else config["cpu_policy"]["qos"],
        cpu_node=None if gpu else config["cpu_policy"]["cpu_node"],
        script=str(root / "orchestration/jobs" / intent["key"] / "run.sbatch"),
        job_name=f"srf3ce-{intent['kind']}-{intent['key']}-LH",
        claim_id=object_hash(
            [
                checked["run_identity_sha256"],
                intent["key"],
                checked["manifest_sha256"],
                checked["amendment_sha256"],
            ]
        ),
        comment="srf3ce-"
        + object_hash([str(root), intent["key"], checked["run_identity_sha256"]])[:24],
    )
    if intent.get("kind") not in SCIENCE or any(intent.get(k) != v for k, v in expected.items()):
        raise PermissionError("Actual science entry/source/amendment identity differs")
    if any(TASKS[task]["kind"] != intent["kind"] for task in intent["task_ids"]):
        raise PermissionError("Instrumented claims include an unrelated registered stage")
    if Path(intent["script"]).read_text() != _script_body(
        root, root / "code", expected["job_entry_command"]
    ) or (
        intent.get("script_sha256") != _sha(intent["script"])
        or intent.get("sbatch") != _sbatch_args(intent)
    ):
        raise PermissionError("Actual instrumented batch script or submission fields changed")
    return checked


def seal_amendment(root, manifest_path, *, typed_verified):
    """Seal the amendment after this run's engineering allocations are terminal."""
    controller_class = install(root, manifest_path)
    root = Path(root).resolve(strict=True)
    checked = verify_extension(root, manifest_path)
    if any((root / name).exists() for name in ("EXECUTION_FREEZE.json", "runs", "sealed")):
        raise PermissionError("Observation amendment must precede all science")
    config = _read(root / "orchestration/CONFIG.json")
    controller = controller_class(
        root,
        config["code_root"],
        python=config["python"],
        user=config["user"],
        pool_size=config["pool_size"],
        cpu_partition=config["cpu_policy"]["partition"],
        cpu_qos=config["cpu_policy"]["qos"],
        cpu_node=config["cpu_policy"]["cpu_node"],
    )
    status = controller.status()
    if status["blocked"] or not set(status["completed"]) >= ENGINE_IDS:
        raise PermissionError(
            "All current run engineering allocations must have verified terminal success"
        )
    for state in controller.attempts():
        intent = _read(root / "orchestration/jobs" / state["key"] / "INTENT.json")
        if intent["kind"] in SCIENCE:
            raise PermissionError("A science submission already exists before the amendment")
    if any(
        _read(root / "tasks" / task / "COMPLETE.json").get("job_id") == "LOCAL_P0"
        for task in ENGINE_IDS
    ):
        raise PermissionError("Amendment requires actual server engineering completion")
    typed = root / "external/revisions/typed_v1"
    audit = _read(typed / "TYPE_AUDIT_RECEIPT.json")
    expected_typed = dict(
        status="VERIFIED_TYPED_V1",
        receipt_sha256=_sha(typed / "TYPE_AUDIT_RECEIPT.json"),
        publication_sha256=_sha(typed / "PUBLICATION.json"),
        auxiliary_code_commit=TYPED_COMMIT,
        preparation_script_sha256=TYPED_SCRIPT_SHA,
    )
    if any(typed_verified.get(k) != v for k, v in expected_typed.items()) or (
        audit.get("auxiliary_code_commit") != TYPED_COMMIT
        or audit.get("preparation_script_sha256") != TYPED_SCRIPT_SHA
        or audit.get("status") != "TWO_INDEPENDENT_REVIEW_PASSES_COMPLETED"
    ):
        raise PermissionError(
            "The existing independently verified typed publication must be retained"
        )
    value = dict(
        status="SEALED_BEFORE_SCIENCE",
        experiment_id=EXPERIMENT,
        extension_manifest_sha256=checked["manifest_sha256"],
        run_identity_sha256=checked["run_identity_sha256"],
        config_sha256=_sha(root / "orchestration/CONFIG.json"),
        scientific_code_commit=checked["base_code_commit"],
        **(
            {"protocol_source_guard": checked["protocol_source_guard"]}
            if "protocol_source_guard" in checked
            else {}
        ),
        extension_commit=checked["extension_commit"],
        scientific_matrix_changed=False,
        new_model_generations=0,
        new_optimizer_steps=0,
        repeat_engine=False,
        scientific_config_sha256=_sha(root / "plan/config/SR_F3_CE.json"),
        scientific_matrix_sha256=_digest(list(TASKS.values())),
        engineering_receipts={
            task: _sha(root / "tasks" / task / "COMPLETE.json") for task in ENGINE_IDS
        },
        typed_publication_sha256=expected_typed["publication_sha256"],
        typed_receipt_sha256=expected_typed["receipt_sha256"],
        pre_amendment_observation_coverage="NOT_COLLECTED_DO_NOT_IMPUTE_OR_REPEAT_ENGINE",
        scope="actual image-encoder and prefill/decode observations for future science only",
        cpu_validation_scope="CPU/toy and scheduler fixtures; no GPU equivalence claim",
    )
    save_json(Path(manifest_path).parent / "AMENDMENT.json", value, exclusive=True)
    return value


class _ExecutionMethods:
    """Three launch-sensitive methods retain the frozen scheduler's original checks."""

    def identity(self):
        original = super().identity()
        context = _context(self.root)
        verify_extension(self.root, context.manifest)
        return original

    def attempts(self):
        result = super().attempts()
        for state in result:
            intent = read_json(self.directory / "jobs" / state["key"] / "INTENT.json")
            if intent["kind"] in SCIENCE:
                _verify_intent(self.root, intent)
        return result

    def release(self, state, intent):
        if intent["kind"] in SCIENCE:
            _verify_intent(self.root, intent)
        return super().release(state, intent)

    def prepare_resume(self, previous_key, attempts, queue, verifier):
        intent = read_json(self.directory / "jobs" / previous_key / "INTENT.json")
        if intent["kind"] in SCIENCE:
            _verify_intent(self.root, intent)
        return super().prepare_resume(previous_key, attempts, queue, verifier)

    def submit(self, kind, *, resume_key=None, resume_verifier=None):
        if kind not in SCIENCE:
            if kind == "controller":
                raise PermissionError("Use the registered independent CPU dispatcher")
            return super().submit(kind, resume_key=resume_key, resume_verifier=resume_verifier)
        checked = verify_extension(self.root, _context(self.root).manifest, science=True)
        if kind not in (*KINDS, "controller"):
            raise ValueError("Unknown submission kind")
        with lease(self.directory):
            self.identity()
            required = [self.code / "scripts/sr_f3_ce/submit.py"]
            if kind != "controller":
                required.append(self.code / "scripts/sr_f3_ce/run_worker.py")
            if any(not path.is_file() for path in required):
                raise PermissionError("Deployed scheduler/worker entry point missing")
            if kind in {"p0", "release", "controller"} and not self.cpu_policy["qos"]:
                raise PermissionError("CPU submission requires an explicitly verified --cpu-qos")
            queue = self.slurm.queue(self.user)  # Fresh observation before every sbatch.
            self.refresh(queue)
            attempts = self.attempts()
            blockers = self.blocking_attempts(attempts)
            resume_envelope = None
            if resume_key is not None:
                key = "job-" + object_hash(["resume", resume_key, self.identity()])[:16]
                existing = next((row for row in attempts if row["key"] == key), None)
                if existing is not None:
                    return existing  # An explicit repeat observes; it never submits twice.
                self.recovery_blockers(attempts, resume_key)
                old_intent = read_json(self.directory / "jobs" / resume_key / "INTENT.json")
                selected = old_intent["task_ids"]
                if old_intent["kind"] != kind:
                    raise PermissionError("Recovery kind differs from its predecessor")
                usage = teacher_usage(queue, attempts, self.user)
                if usage["gpus"] >= GPU_LIMIT or usage["submitted"] >= SUBMIT_LIMIT:
                    return None
                resume_envelope = self.prepare_resume(resume_key, attempts, queue, resume_verifier)
                # CPU checkpoint verification can be slow. Re-observe before reserving anything.
                queue = self.slurm.queue(self.user)
                self.refresh(queue)
                attempts = self.attempts()
                self.recovery_blockers(attempts, resume_key)
                old_job = next(row["job_id"] for row in attempts if row["key"] == resume_key)
                if old_job in queue:
                    raise PermissionError("Recovery predecessor unexpectedly reappeared in queue")
            elif blockers:
                raise RuntimeError("Unresolved or failed submission; no automatic retry")
            elif kind == "controller":
                if any(row["key"] == "controller" for row in attempts):
                    return None
                selected, key = [], "controller"
            else:
                selected = [key for key in self.ready_tasks() if TASKS[key]["kind"] == kind]
                if not selected:
                    return None
                selected = selected[: self.pool_size if kind in {"engine", "train", "eval"} else 1]
                key = "job-" + object_hash(selected)[:16]
            count = len(selected) if kind in {"engine", "train", "eval"} else 0
            usage = teacher_usage(queue, attempts, self.user)
            if count:
                count = min(count, GPU_LIMIT - usage["gpus"])
                if count < 1 or usage["submitted"] >= SUBMIT_LIMIT:
                    return None
                selected = selected[:count]
                if resume_key is None:
                    key = "job-" + object_hash(selected)[:16]
            elif self.cpu_policy["qos"] == TEACHER_QOS and usage["submitted"] >= SUBMIT_LIMIT:
                return None
            directory = self.directory / "jobs" / key
            directory.mkdir(parents=True, exist_ok=True)
            if resume_envelope is not None:
                save_json(directory / "VERIFIED_RESUME.json", resume_envelope, exclusive=True)
            save_json(
                directory / "QUEUE_BEFORE_SUBMIT.json",
                dict(at=now(), jobs=queue, teacher_usage=usage),
                exclusive=True,
            )
            intent = dict(
                key=key,
                kind=kind,
                task_ids=selected,
                gpu_count=count,
                created=now(),
                run_identity_sha256=self.identity(),
                root=str(self.root),
                code_root=str(self.code),
                python=self.python,
                user=self.user,
                job_name=f"srf3ce-{kind}-{key}-LH",
                claim_id=object_hash([self.identity(), key]),
                cpus=4 * max(1, count),
                memory_gib=90 * count if count else 12,
                partition="cluster02" if count else self.cpu_policy["partition"],
                qos=TEACHER_QOS if count else self.cpu_policy["qos"],
                cpu_node=None if count else self.cpu_policy["cpu_node"],
            )
            intent.update(
                execution_entry_schema=1,
                extension_manifest_path=_context(self.root).manifest,
                extension_manifest_sha256=checked["manifest_sha256"],
                amendment_sha256=checked["amendment_sha256"],
                job_entry_command=_job_command(self.root, key, self.python),
                worker_entry_commands={
                    task: _worker_command(self.root, task, self.python) for task in selected
                },
            )
            intent["claim_id"] = object_hash(
                [self.identity(), key, checked["manifest_sha256"], checked["amendment_sha256"]]
            )
            if kind in {"train", "eval", "release"}:
                intent["execution_freeze_sha256"] = verify_freeze(self.root)
            if resume_envelope is not None:
                intent.update(
                    resume_of=resume_key,
                    resume_verification_sha256=file_hash(directory / "VERIFIED_RESUME.json"),
                )
            intent["comment"] = "srf3ce-" + object_hash([str(self.root), key, self.identity()])[:24]
            script = directory / "run.sbatch"
            intent["script"] = str(script)
            command = intent["job_entry_command"]
            body = _script_body(self.root, self.code, command)
            with script.open("x") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            intent["script_sha256"] = file_hash(script)
            args = _sbatch_args(intent)
            intent["sbatch"] = args
            _verify_intent(self.root, intent)
            save_json(directory / "INTENT.json", intent, exclusive=True)
            state = dict(
                key=key,
                status="SUBMISSION_UNKNOWN",
                gpu_count=count,
                run_identity_sha256=self.identity(),
                intent_sha256=file_hash(directory / "INTENT.json"),
                qos=intent["qos"],
            )
            save_json(directory / "STATE.json", state)
            try:
                result = self.slurm.run(args)
            except Exception as error:
                save_json(
                    directory / "SUBMISSION_EXCEPTION.json",
                    dict(error=repr(error), at=now()),
                    exclusive=True,
                )
                raise
            save_json(
                directory / "SUBMISSION.json",
                dict(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    at=now(),
                ),
                exclusive=True,
            )
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
            if result.returncode or not match:
                return state
            state.update(job_id=match[1], status="HELD")
            save_json(directory / "STATE.json", state)
            self.release(state, intent)
            return state

    def verify_job(self, state):
        """Authenticate accounting and process evidence before accepting any task."""
        directory = self.directory / "jobs" / state["key"]
        intent = read_json(directory / "INTENT.json")
        if intent["kind"] not in SCIENCE:
            return super().verify_job(state)
        _verify_intent(self.root, intent)
        job = state.get("job_id")
        terminal = read_json(directory / "TERMINAL.json")
        expected = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user=self.user,
            account="rose",
            qos=intent["qos"],
            name=intent["job_name"],
        )
        if any(terminal.get(key) != value for key, value in expected.items()):
            raise PermissionError("Completed accounting identity differs")
        allocation = read_json(directory / "ALLOCATION_VERIFIED.json")
        if (
            allocation.get("job_id") != job
            or allocation.get("script_sha256") != (intent["script_sha256"])
            or file_hash(intent["script"]) != intent["script_sha256"]
        ):
            raise PermissionError("Completed allocation/script identity differs")
        verify_allocation(allocation["fields"], dict(intent, job_id=job), self.user, held=True)
        release = read_json(directory / "RELEASE.json")
        if release.get("job_id") != job or release.get("returncode") != 0:
            raise PermissionError("Completed job lacks confirmed release")
        started = read_json(directory / "STARTED.json")
        complete = read_json(directory / "COMPLETE.json")
        if started.get("job_id") != job or complete != dict(
            status="COMPLETE", job_id=job, task_ids=intent["task_ids"], claim_id=intent["claim_id"]
        ):
            raise PermissionError("Job process completion identity differs")
        devices = started.get("devices")
        if not isinstance(devices, list):
            raise PermissionError("Physical device receipt missing")
        if intent["gpu_count"]:
            visible_devices(",".join(devices), intent["gpu_count"])
        elif devices:
            raise PermissionError("CPU job unexpectedly owns GPU devices")
        for index, task_id in enumerate(intent["task_ids"]):
            launch = read_json(directory / (task_id + ".LAUNCH.json"))
            process_exit = read_json(directory / (task_id + ".EXIT.json"))
            expected_command = _worker_command(self.root, task_id, intent["python"])
            if launch != dict(
                command=expected_command,
                claim_id=intent["claim_id"],
                device=devices[index] if devices else "",
            ) or (process_exit.get("returncode") != 0):
                raise PermissionError("Task process identity/exit differs")
            verify_task(self.root, task_id, binding=dict(job_id=job, claim_id=intent["claim_id"]))
            verify_observation_receipt(self.root, intent, task_id, job)
        return intent["task_ids"]


def run_job(root, job_key, *, popen=subprocess.Popen):
    """Run exactly the fixed claims; no dynamic scientific routing or idle GPUs."""
    verify_run = _context(root).api.protocol.verify_run

    root = Path(root).resolve(strict=True)
    verify_run(root)
    if not re.fullmatch(r"job-[a-f0-9]{16}|controller", job_key):
        raise PermissionError("Unsafe allocation key")
    directory = root / "orchestration/jobs" / job_key
    intent, state = read_json(directory / "INTENT.json"), read_json(directory / "STATE.json")
    _verify_intent(root, intent)
    job = os.environ.get("SLURM_JOB_ID", "")
    if state.get("job_id") != job or intent["run_identity_sha256"] != run_identity(root):
        raise PermissionError("Runner is not the bound Slurm job")
    if state.get("intent_sha256") != file_hash(directory / "INTENT.json"):
        raise PermissionError("Durable submission intent changed")
    release = read_json(directory / "RELEASE_INTENT.json")
    if release.get("job_id") != job or (directory / "STARTED.json").exists():
        raise PermissionError("Allocation already started or lacks verified release")
    count = intent["gpu_count"]
    devices = visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""), count) if count else []
    if count != len(intent["task_ids"]) and count:
        raise PermissionError("Each GPU must have exactly one ready task")
    allocation = read_json(directory / "ALLOCATION_VERIFIED.json")
    if allocation.get("job_id") != job or file_hash(intent["script"]) != intent["script_sha256"]:
        raise PermissionError("Runtime allocation/script identity differs")
    verify_allocation(allocation["fields"], dict(intent, job_id=job), intent["user"], held=True)
    resume_receipt = runtime_resume_receipt(root, intent)
    save_json(
        directory / "STARTED.json", dict(job_id=job, devices=devices, at=now()), exclusive=True
    )
    children = []
    for index, task_id in enumerate(intent["task_ids"]):
        env = dict(os.environ, SRF3CE_JOB_KEY=job_key, SRF3CE_CLAIM_ID=intent["claim_id"])
        env["CUDA_VISIBLE_DEVICES"] = devices[index] if count else ""
        if resume_receipt is not None:
            env["SRF3CE_RESUME_RECEIPT"] = str(resume_receipt)
            env["SRF3CE_PREVIOUS_JOB_KEY"] = intent["resume_of"]
        else:
            env.pop("SRF3CE_RESUME_RECEIPT", None)
            env.pop("SRF3CE_PREVIOUS_JOB_KEY", None)
        command = _worker_command(root, task_id, intent["python"])
        save_json(
            directory / (task_id + ".LAUNCH.json"),
            dict(
                command=command,
                device=env["CUDA_VISIBLE_DEVICES"],
                claim_id=intent["claim_id"],
            ),
            exclusive=True,
        )
        children.append((task_id, popen(command, env=env)))
    failed = []
    for task_id, child in children:
        code = child.wait()
        save_json(directory / (task_id + ".EXIT.json"), dict(returncode=code), exclusive=True)
        if code:
            failed.append(task_id)
        else:
            verify_task(root, task_id, binding=dict(job_id=job, claim_id=intent["claim_id"]))
            verify_observation_receipt(root, intent, task_id, job)
    if failed:
        raise RuntimeError("Workers failed; preserved without retry: " + ", ".join(failed))
    save_json(
        directory / "COMPLETE.json",
        dict(
            status="COMPLETE", job_id=job, task_ids=intent["task_ids"], claim_id=intent["claim_id"]
        ),
        exclusive=True,
    )
    return read_json(directory / "COMPLETE.json")


def _binding(root, intent, task_id, job):
    checked = _verify_intent(root, intent)
    return dict(
        experiment_id=EXPERIMENT,
        run_identity_sha256=checked["run_identity_sha256"],
        execution_freeze_sha256=checked["execution_freeze_sha256"],
        extension_manifest_sha256=checked["manifest_sha256"],
        amendment_sha256=checked["amendment_sha256"],
        job_key=intent["key"],
        job_id=job,
        task_id=task_id,
        claim_id=intent["claim_id"],
    )


def _events_summary(path, binding, installation_id):
    scopes, begins, ends, count = {}, 0, 0, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            if (
                row.get("execution_binding") != binding
                or row.get("installation_id") != installation_id
            ):
                raise PermissionError("Observation event belongs to another execution")
            key = (row.get("runtime_id"), row.get("scope_id"))
            if any(not isinstance(value, str) or not value for value in key):
                raise PermissionError("Observation scope identity missing")
            if row.get("event") == "SCOPE_BEGIN":
                if key in scopes:
                    raise PermissionError("Repeated observation BEGIN scope")
                scopes[key] = False
                begins += 1
            elif row.get("event") == "SCOPE_END":
                if (
                    key not in scopes
                    or scopes[key]
                    or row.get("coverage_status") != "COMPLETE"
                    or (
                        row.get("outcome")
                        not in {"RETURNED", "RAISED", "INTERRUPTED", "CHECKPOINT_EARLY_STOP"}
                    )
                ):
                    raise PermissionError(
                        "Observation END is missing, repeated or has UNKNOWN coverage"
                    )
                scopes[key] = True
                ends += 1
            else:
                raise PermissionError("Unregistered observation event")
    if any(not ended for ended in scopes.values()):
        raise PermissionError("Unfinished observation scope remains UNKNOWN")
    return dict(
        sha256=_sha(path), bytes=path.stat().st_size, rows=count, scope_begin=begins, scope_end=ends
    )


def verify_observation_receipt(root, intent, task_id, job):
    root = Path(root)
    binding = _binding(root, intent, task_id, job)
    directory = root / "accounting/observability" / intent["key"] / task_id
    installation_path, final_path = directory / "INSTALLATION.json", directory / "FINAL.json"
    installation, final = _read(installation_path), _read(final_path)
    if (
        installation.get("schema_version") != 1
        or installation.get("status") != "INSTALLED"
        or (
            installation.get("execution_binding") != binding
            or final.get("schema_version") != 1
            or final.get("status") != "COMPLETE"
            or final.get("execution_binding") != binding
            or final.get("installation_sha256") != _sha(installation_path)
            or final.get("task_receipt_sha256") != _sha(root / "tasks" / task_id / "COMPLETE.json")
        )
    ):
        raise PermissionError("Observation installation/final/task identity differs")
    observed = installation["observer"]
    observer_path = Path(_context(root).manifest).parent / "observe_runtime.py"
    if observed.get("observer_source_sha256") != _sha(observer_path) or (
        observed.get("runtime_class") != "sr_f3_ce.runtime.SRF3CERuntime"
        or observed.get("installation_status") != "INSTALLED"
    ):
        raise PermissionError("Observation receipt names another runtime/source")
    base_files = {
        row["path"]: row for row in _read(root / "deployment/SOURCE_MANIFEST.json")["files"]
    }
    if set(observed.get("original_methods", {})) != {
        "__init__",
        "generate_group",
        "_native_forward",
    }:
        raise PermissionError("Observation original method inventory is incomplete")
    for method in observed["original_methods"].values():
        path = Path(method["file"]).resolve(strict=True)
        if not path.is_relative_to(root / "code"):
            raise PermissionError("Observed method is outside the frozen scientific implementation")
        record = base_files.get(str(path.relative_to(root / "code")))
        if (
            not record
            or method.get("file_sha256") != record["sha256"]
            or _sha(path) != record["sha256"]
        ):
            raise PermissionError("Observed original method source changed")
    summary = _events_summary(directory / "events.jsonl", binding, observed["installation_id"])
    summary["path"] = str((directory / "events.jsonl").relative_to(root))
    snapshot = final.get("observer_snapshot", {})
    if (
        final.get("events") != summary
        or snapshot.get("installation_id") != observed["installation_id"]
        or (
            snapshot.get("sealed") is not True
            or snapshot.get("active_scopes") != []
            or snapshot.get("write_errors") != []
            or snapshot.get("events_written", {}).get("SCOPE_BEGIN", 0) != summary["scope_begin"]
            or snapshot.get("events_written", {}).get("SCOPE_END", 0) != summary["scope_end"]
        )
    ):
        raise PermissionError("Observation event totals or final snapshot differ")
    if intent["kind"] in {"train", "eval"} and summary["scope_begin"] < 1:
        raise PermissionError("A scientific model worker has no observation scopes")
    return dict(path=str(final_path.relative_to(root)), sha256=_sha(final_path))


def _load_observer(root):
    context = _context(root)
    checked = verify_extension(root, context.manifest, science=True)
    path = Path(context.manifest).parent / "observe_runtime.py"
    record = _source_record(
        context.manifest, checked["manifest_sha256"], "observe_runtime.py", "sources"
    )
    return _load_source(path, record, "srf3ce_explicit_observer")


def _source_record(manifest_path, expected_sha, relative, field):
    body = Path(manifest_path).read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha:
        raise PermissionError("Source manifest changed before selecting executable bytes")
    matches = [row for row in json.loads(body)[field] if row["path"] == relative]
    if len(matches) != 1:
        raise PermissionError("Executable source has no unique manifest identity")
    return matches[0]


def _load_source(path, record, name):
    """Compile exactly the bytes checked here; loaders must never choose a cached pyc."""
    body = Path(path).read_bytes()
    if len(body) != record["bytes"] or hashlib.sha256(body).hexdigest() != record["sha256"]:
        raise PermissionError("Executable source changed before verified-byte import")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(body, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run_worker(root, task_id):
    root = Path(root).resolve(strict=True)
    context = _context(root)
    context.api.protocol.verify_run(root, science=True)
    task = TASKS[task_id]
    if task["kind"] not in SCIENCE:
        raise PermissionError("Observation wrapper must not replace this run's ENGINE/P0 execution")
    bound = context.api.orchestration.worker_binding(root, task_id)
    key = os.environ["SRF3CE_JOB_KEY"]
    intent = _read(root / "orchestration/jobs" / key / "INTENT.json")
    binding = _binding(root, intent, task_id, bound["job_id"])
    directory = root / "accounting/observability" / key / task_id
    directory.mkdir(parents=True, exist_ok=True)
    events = directory / "events.jsonl"
    with events.open("x", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())

    def emit(event):
        if "execution_binding" in event:
            raise PermissionError("Observer cannot replace execution binding")
        body = (
            json.dumps(
                dict(event, execution_binding=binding),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        with events.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    observer, runtime_class = None, None
    try:
        observer = _load_observer(root)
        runtime_class = importlib.import_module("sr_f3_ce.runtime").SRF3CERuntime
        observation = observer.install(runtime_class, emit)
        installation = dict(
            schema_version=1, status="INSTALLED", execution_binding=binding, observer=observation
        )
        save_json(directory / "INSTALLATION.json", installation, exclusive=True)
        script = root / "code/scripts/sr_f3_ce/run_worker.py"
        checked = verify_extension(root, context.manifest, science=True)
        record = _source_record(
            root / "deployment/SOURCE_MANIFEST.json",
            checked["base_source_manifest_sha256"],
            "scripts/sr_f3_ce/run_worker.py",
            "files",
        )
        worker = _load_source(script, record, "srf3ce_original_registered_worker")
        worker.main(["--root", str(root), "--task-id", task_id])
        verify_task(root, task_id, binding=bound)
        summary = _events_summary(events, binding, observation["installation_id"])
        summary["path"] = str(events.relative_to(root))
        snapshot = observer.seal(runtime_class)
        if snapshot.get("active_scopes") != [] or snapshot.get("write_errors") != []:
            raise PermissionError("Observer cannot certify an incomplete or failed ledger")
        final = dict(
            schema_version=1,
            status="COMPLETE",
            execution_binding=binding,
            installation_sha256=_sha(directory / "INSTALLATION.json"),
            events=summary,
            observer_snapshot=snapshot,
            task_receipt_sha256=_sha(root / "tasks" / task_id / "COMPLETE.json"),
        )
        save_json(directory / "FINAL.json", final, exclusive=True)
        return verify_observation_receipt(root, intent, task_id, bound["job_id"])
    except BaseException as error:
        failure = dict(
            status="FAILED",
            execution_binding=binding,
            error=repr(error),
            at=time.time(),
        )
        if observer is not None and runtime_class is not None:
            try:
                failure["observer_snapshot"] = observer.seal(runtime_class)
            except BaseException as snapshot_error:
                failure["observer_snapshot_error"] = repr(snapshot_error)
        try:
            failure["events_sha256"] = _sha(events)
        except BaseException as hash_error:
            failure.update(events_sha256="UNKNOWN", events_hash_error=repr(hash_error))
        try:
            save_json(directory / "FAILURE.json", failure, exclusive=True)
        except BaseException as persistence_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "Observation failure record could not be persisted: " + repr(persistence_error)
                )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("job", "worker"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-key")
    parser.add_argument("--task-id")
    args = parser.parse_args()
    install(args.root, args.manifest)
    verify_extension(args.root, args.manifest, science=True)
    if args.mode == "job":
        if not args.job_key or args.task_id:
            parser.error("job requires only job-key")
        result = run_job(args.root, args.job_key)
    else:
        if not args.task_id or args.job_key:
            parser.error("worker requires only task-id")
        result = run_worker(args.root, args.task_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
