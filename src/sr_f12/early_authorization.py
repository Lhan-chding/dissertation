"""Separate authorization to generate sealed endpoints for terminal trained models.

This never authorizes scores. The original twelve-model acceptance remains the
only score release gate. Registration and per-model acceptances are immutable.
"""

from __future__ import annotations

import getpass
from pathlib import Path

from sr_f1.evaluation import sha_file

from .endpoint import read_json, verify_main_completion
from .evaluation import durable_json, verify_source_manifest
from .orchestration import Slurm
from .protocol import AMENDMENT_ID, TEACHER_QOS, object_hash, scientific_matrix
from .source_revision import resolve_source_revision

POLICY = "TRAINING_PRIORITY_SEALED_UNTIL_ALL_MAIN_COMPLETE"
FROZEN_FILES = (
    "SCIENCE_FREEZE.json",
    "config/SR_F1_2_FROZEN.json",
    "TECHNICAL_CHECK_SR_F1_2.json",
    "SEMANTIC_PREDICTIONS.json",
)


def verify_early_implementation(root, code_root, source_manifest, source_commit):
    root = Path(root).resolve(strict=True)
    if (root / "STOP").exists():
        raise PermissionError("Experiment STOP exists")
    registration = read_json(root / "EARLY_ENDPOINT_IMPLEMENTATION.json")
    primary = read_json(root / "orchestration/CONFIG.json")
    endpoint = read_json(root / "endpoint_orchestration/CONFIG.json")
    source = resolve_source_revision(root)
    code_root, source_manifest = (
        Path(code_root).resolve(strict=True),
        Path(source_manifest).resolve(strict=True),
    )
    expected = dict(
        status="AUTHORIZED_EARLY_SEALED_GENERATION",
        policy=POLICY,
        version=AMENDMENT_ID,
        user=getpass.getuser(),
        root=str(root),
        code_root=str(code_root),
        source_manifest=str(source_manifest),
        source_commit=source_commit,
    )
    if any(registration.get(key) != value for key, value in expected.items()):
        raise PermissionError("Early implementation registration identity differs")
    for config in (primary, endpoint):
        if any(config.get(key) != expected[key] for key in ("version", "user", "root")):
            raise PermissionError("Early endpoint owner or run differs")
    if (
        registration.get("endpoint_config_sha256")
        != sha_file(root / "endpoint_orchestration/CONFIG.json")
        or registration.get("frozen_files")
        != {name: sha_file(root / name) for name in FROZEN_FILES}
        or registration.get("training_source_commit") != source["source_commit"]
        or read_json(root / "SCIENCE_FREEZE.json").get("source_commit") != source["source_commit"]
    ):
        raise PermissionError("Early source or frozen registration differs")
    training_code = Path(registration["training_code_root"]).resolve(strict=True)
    training_manifest = Path(registration["training_source_manifest"]).resolve(strict=True)
    if training_code == code_root or training_manifest == source_manifest:
        raise PermissionError("Early evaluation requires independent implementation")
    if str(training_code) != (source["code_root"] or primary["code_root"]):
        raise PermissionError("Training code root differs from certified revision")
    training_hash = verify_source_manifest(training_manifest, training_code)
    endpoint_hash = verify_source_manifest(source_manifest, code_root)
    if (
        registration.get("training_source_manifest_sha256") != training_hash
        or (
            source["source_manifest_sha256"] is not None
            and source["source_manifest_sha256"] != training_hash
        )
        or read_json(training_manifest).get("git_commit") != source["source_commit"]
        or read_json(source_manifest).get("git_commit") != source_commit
        or registration.get("source_manifest_sha256") != endpoint_hash
    ):
        raise PermissionError("Early or training source manifest binding differs")
    return registration


def _model_acceptance(root, model_id, *, slurm=None, state_loader=None):
    root = Path(root).resolve(strict=True)
    models = [row["model_id"] for row in scientific_matrix()]
    if model_id not in models:
        raise PermissionError("Early generation requires a registered trained model")
    registered = read_json(root / "EARLY_ENDPOINT_IMPLEMENTATION.json")
    registered = verify_early_implementation(
        root, registered["code_root"], registered["source_manifest"], registered["source_commit"]
    )
    snapshot = read_json(root / "orchestration/STATE.json")
    directory = root / "orchestration/tasks" / model_id
    state = read_json(directory / "STATE.json")
    intent = read_json(directory / "INTENT.json")
    job = state.get("job_id")
    cached = snapshot.get("tasks", {}).get(model_id, {})
    if (
        state.get("task_id") != model_id
        or state.get("status") != "COMPLETE"
        or cached.get("status") != "COMPLETE"
        or cached.get("job_id") != job
        or not isinstance(job, str)
        or not job.isdigit()
        or intent.get("task_id") != model_id
        or intent.get("gpu_count") != 1
        or intent.get("job_name") != "srf12-" + model_id
        or intent.get("source_sha256") != registered["training_source_manifest_sha256"]
    ):
        raise PermissionError("Main scheduler model is not uniquely complete")
    for other in models:
        path = root / "orchestration/tasks" / other / "STATE.json"
        if other != model_id and path.exists() and read_json(path).get("job_id") == job:
            raise PermissionError("Duplicate scientific job identity")
    backend = slurm or Slurm()
    if job in backend.queue(registered["user"]):
        raise PermissionError("Main model remains in live queue")
    expected = dict(
        job_id=job,
        state="COMPLETED",
        exit_code="0:0",
        user=registered["user"],
        account="rose",
        qos=TEACHER_QOS,
        name=intent["job_name"],
    )
    if backend.terminal(job) != expected or read_json(directory / "TERMINAL.json") != expected:
        raise PermissionError("Main model live accounting terminal is not successful")
    script = Path(intent["script"]).resolve(strict=True)
    if (
        not script.is_relative_to(directory.resolve())
        or sha_file(script) != intent["script_sha256"]
    ):
        raise PermissionError("Main model script identity changed")
    verified_path = directory / "VERIFIED_COMPLETE.json"
    verified = read_json(verified_path)
    done = read_json(root / "runs" / model_id / "COMPLETE.json")
    if (
        verified.get("task_id") != model_id
        or verified.get("receipt_sha256") != object_hash(done)
        or verified.get("checkpoint_sha256") != done.get("checkpoint", {}).get("sha256")
    ):
        raise PermissionError("Scheduler completion no longer binds final state")
    completion = verify_main_completion(root, model_ids=[model_id], state_loader=state_loader)
    return dict(
        status="MODEL_TERMINAL_AND_STATE_VERIFIED_SEALED",
        model_id=model_id,
        implementation_sha256=object_hash(registered),
        training_source_commit=registered["training_source_commit"],
        terminal=expected,
        intent_sha256=object_hash(intent),
        completion_receipt_sha256=sha_file(verified_path),
        model_completion=completion,
        model_completion_sha256=object_hash(completion),
    )


def create_model_acceptance(root, model_id, *, slurm=None, state_loader=None):
    result = _model_acceptance(root, model_id, slurm=slurm, state_loader=state_loader)
    durable_json(
        Path(root) / "endpoint_orchestration/model_acceptance" / (model_id + ".json"), result
    )
    return result


def verify_generation_authorization(root, model_id, *, slurm=None, state_loader=None):
    result = _model_acceptance(root, model_id, slurm=slurm, state_loader=state_loader)
    accepted = read_json(
        Path(root) / "endpoint_orchestration/model_acceptance" / (model_id + ".json")
    )
    if accepted != result:
        raise PermissionError("Immutable model generation acceptance differs")
    return result
