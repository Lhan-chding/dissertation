"""Append-only, bounded technical-source repairs for an existing SR-F1.2 registration.

Baseline workers keep their original source directory and manifest. A repair gets
an independent code directory and may replace only a failed zero-update technical
attempt, retaining its evidence and reusing the exact first 32 generated answers.
"""

from __future__ import annotations

import ast
import getpass
import os
import re
import shutil
import tempfile
from pathlib import Path

from .orchestration import Slurm, controller_lease, file_hash, read_json, save_json
from .protocol import AMENDMENT_ID, TEACHER_QOS, object_hash

ALLOWED_IMPLEMENTATION_CHANGES = frozenset(
    {
        "src/sr_f12/training.py",
        "src/sr_f12/runner.py",
        "src/sr_f12/orchestration.py",
        "src/sr_f12/endpoint_orchestration.py",
        "src/sr_f12/source_revision.py",
        "src/sr_f12/endpoint.py",
        "scripts/sr_f12/controller.py",
        "scripts/sr_f12/endpoint_controller.py",
        "scripts/sr_f12/run_evaluation.py",
    }
)
GENERATION_NODES = frozenset(
    {
        "BatchedBalancedJSONStop",
        "language_linear_modules",
        "configure_training",
        "_repeat_prompt",
        "SRF12Runtime.__init__",
        "SRF12Runtime.processor_only",
        "SRF12Runtime.prepare_text",
        "SRF12Runtime.generate_training",
        "SRF12Runtime.generate",
        "SRF12Runtime.generate_group",
    }
)


def _manifest_files(manifest):
    files = manifest["files"]
    if isinstance(files, dict):
        return files
    result = {item["path"]: item["sha256"] for item in files}
    if len(result) != len(files):
        raise PermissionError("Duplicate source manifest entries")
    return result


def generation_identity(code_root):
    tree = ast.parse((Path(code_root) / "src/sr_f12/runtime.py").read_text())
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in GENERATION_NODES:
            result[node.name] = object_hash(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.ClassDef) and node.name == "SRF12Runtime":
            for member in node.body:
                name = "SRF12Runtime." + getattr(member, "name", "")
                if name in GENERATION_NODES:
                    result[name] = object_hash(ast.dump(member, include_attributes=False))
    if set(result) != GENERATION_NODES:
        raise PermissionError("Registered generation/zero-LoRA methods missing")
    return result


def training_identity(code_root):
    """Only the technical microbatch selector may change inside training.py."""
    tree = ast.parse((Path(code_root) / "src/sr_f12/training.py").read_text())
    kept = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "select_microbatch":
            continue
        if isinstance(node, ast.ClassDef) and node.name == "MicrobatchNumericalMismatch":
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            if (
                node.decorator_list
                or node.keywords
                or len(node.bases) != 1
                or not isinstance(node.bases[0], ast.Name)
                or node.bases[0].id != "RuntimeError"
                or any(not isinstance(body, ast.Pass) for body in node.body)
            ):
                raise PermissionError("Numerical mismatch exception contains unexpected behavior")
            continue
        kept.append(node)
    tree.body = kept
    return object_hash(ast.dump(tree, include_attributes=False))


def resolve_source_revision(root, source_commit=None):
    """Only a fully applied repair can advance the original registered source."""
    root = Path(root).resolve(strict=True)
    amendment = read_json(root / "AMENDMENT.json")
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise PermissionError("Not the SR-F1.2 registration")
    original = amendment["source_commit"]
    result = dict(
        source_commit=original,
        registration_source_commit=original,
        revision_id=None,
        revision_sha256=None,
        source_manifest_sha256=None,
        code_root=None,
    )
    config_hash = object_hash(read_json(root / "config/SR_F1_2.json"))
    if config_hash != amendment["config_sha256"]:
        raise PermissionError("Scientific candidate configuration was changed")
    previous = None
    for directory in sorted((root / "source_revisions").glob("r[0-9][0-9][0-9][0-9]")):
        if not (directory / "APPLIED.json").exists():
            # An authorized but interrupted maintenance cannot authorize execution.
            if (directory / "AUTHORIZATION.json").exists():
                raise PermissionError("Source repair maintenance is incomplete")
            continue
        revision = read_json(directory / "AUTHORIZATION.json")
        applied = read_json(directory / "APPLIED.json")
        if (
            revision.get("scope") != "failed_zero_update_technical_repair"
            or revision.get("revision_id") != directory.name
            or revision.get("amendment_sha256") != object_hash(amendment)
            or revision.get("config_sha256") != config_hash
            or revision.get("previous_source_commit") != result["source_commit"]
            or revision.get("previous_revision_sha256") != previous
            or revision.get("generation_before") != revision.get("generation_after")
            or revision.get("training_before") != revision.get("training_after")
            or applied.get("authorization_sha256") != object_hash(revision)
            or applied.get("status") != "APPLIED_ONCE_REUSE_32_NO_BASELINE_RESTART"
            or applied.get("reused_records") != 32
        ):
            raise PermissionError("Broken or out-of-scope technical source revision")
        if not re.fullmatch(r"[0-9a-f]{40}", revision.get("source_commit", "")):
            raise PermissionError("Revision source commit is invalid")
        previous = object_hash(revision)
        result.update(
            source_commit=revision["source_commit"],
            revision_id=directory.name,
            revision_sha256=previous,
            source_manifest_sha256=revision["source_manifest_sha256"],
            code_root=revision["code_root"],
        )
    if source_commit is not None and source_commit != result["source_commit"]:
        raise PermissionError("Requested source is not the active certified revision")
    return result


def _tree_manifest(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PermissionError("Technical preservation cannot follow symlinks")
        if path.is_file():
            result[str(path.relative_to(directory))] = file_hash(path)
    return result


def _verify_tree(directory, expected):
    if _tree_manifest(directory) != expected:
        raise PermissionError("Preserved technical evidence differs")


def _raw32(root):
    from sr_f1.data import load_inputs

    from .runner import technical_schedule

    root = Path(root)
    inputs = load_inputs(root)
    schedule = technical_schedule(root)[:4]
    paths = root / "technical/preflight/rollouts"
    expected = {f"{row['step']:02d}-{row['slot']:02d}.json" for row in schedule}
    if {p.name for p in paths.glob("*.json")} != expected:
        raise PermissionError("Require exactly the four preserved preflight groups")
    manifest, policies = {}, set()
    for slot in schedule:
        name = f"{slot['step']:02d}-{slot['slot']:02d}.json"
        path = paths / name
        saved = read_json(path)
        identity, records = saved["identity"], saved["records"]
        policies.add(identity["policy_hash"])
        target = dict(
            run_id="PREFLIGHT",
            step=slot["step"],
            slot=slot["slot"],
            qid=slot["qid"],
            policy_hash=identity["policy_hash"],
            seeds=slot["rollout_seeds"],
            input_hash=object_hash(inputs[slot["qid"]]),
        )
        if identity != target or len(records) != 8 or saved["sha256"] != object_hash(records):
            raise PermissionError("Preserved preflight input/policy/token identity differs")
        for i, row in enumerate(records):
            if (
                row["row"] != inputs[slot["qid"]]
                or row["qid"] != slot["qid"]
                or row["root"] != str(root.resolve())
                or row["group_row_index"] != i
                or row["group_seed"] != slot["rollout_seeds"][0]
                or not row["tokens"]
                or len(row["tokens"]) != len(row["sampler_logprobs"])
            ):
                raise PermissionError("Preserved preflight answer metadata differs")
        manifest[name] = file_hash(path)
    if len(policies) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(policies))):
        raise PermissionError("Preflight answers did not share one zero-LoRA policy")
    return dict(files=manifest, records=32, policy_hash=next(iter(policies)))


def prepare_technical_repair(
    root,
    *,
    revision_id,
    code_root,
    source_manifest,
    source_commit,
    controller_job_id,
    slurm=None,
    user=None,
):
    """Run once on CPU after a diagnosed technical failure; never cancels any job.

    All paths, source scopes, old controller/technical terminal states and preserved
    answers are checked before changes. Interrupted maintenance is resumed by the
    exact same arguments and immutable intent, without resubmission or resampling.
    """
    from .evaluation import verify_source_manifest

    root, code_root = Path(root).resolve(strict=True), Path(code_root).resolve(strict=True)
    source_manifest = Path(source_manifest).resolve(strict=True)
    if not re.fullmatch(r"r[0-9]{4}", revision_id):
        raise ValueError("Revision identifier must be rNNNN")
    slurm, user = slurm or Slurm(), user or getpass.getuser()
    directory = root / "source_revisions" / revision_id
    signature = dict(
        revision_id=revision_id,
        code_root=str(code_root),
        source_manifest=str(source_manifest),
        source_commit=source_commit,
        controller_job_id=str(controller_job_id),
        user=user,
    )
    with controller_lease(root / "orchestration"):
        if (directory / "APPLIED.json").exists():
            if read_json(directory / "REQUEST.json") != signature:
                raise PermissionError("Consumed repair cannot be used for another request")
            resolve_source_revision(root, source_commit)
            return read_json(directory / "APPLIED.json")
        if not (directory / "AUTHORIZATION.json").exists():
            previous = resolve_source_revision(root)
            if (
                any(
                    (root / name).exists() and any((root / name).rglob("*"))
                    for name in ("runs", "training")
                )
                or (root / "SCIENCE_FREEZE.json").exists()
            ):
                raise PermissionError("Zero-update repair must precede all scientific output")
            if any((root / "technical").glob("SRF1_2_TECHNICAL_*/checkpoints/*.pt")):
                raise PermissionError(
                    "This recovery authorizes zero-update preflight failures only"
                )
            config = (
                read_json(root / "orchestration/CONFIG.json")
                if previous["revision_id"] is None
                else read_json(
                    root / "orchestration/revisions" / previous["revision_id"] / "CONFIG.json"
                )
            )
            if code_root == Path(config["code_root"]).resolve():
                raise PermissionError("Repair source must be a separate deployment directory")
            old_manifest = read_json(config["source_manifest"])
            new_manifest = read_json(source_manifest)
            old_hash = verify_source_manifest(config["source_manifest"], config["code_root"])
            new_hash = verify_source_manifest(source_manifest, code_root)
            if (
                old_manifest.get("git_commit") != previous["source_commit"]
                or new_manifest.get("git_commit") != source_commit
            ):
                raise PermissionError("Repair source commit/manifest identity differs")
            old_files, new_files = _manifest_files(old_manifest), _manifest_files(new_manifest)
            changed = sorted(
                name
                for name in set(old_files) | set(new_files)
                if old_files.get(name) != new_files.get(name)
            )
            for name in changed:
                if (
                    name.startswith(("src/", "scripts/"))
                    and name not in ALLOWED_IMPLEMENTATION_CHANGES
                ):
                    raise PermissionError("Repair changes an unapproved executable: " + name)
            before, after = generation_identity(config["code_root"]), generation_identity(code_root)
            if before != after:
                raise PermissionError(
                    "Technical repair must retain generation and zero-LoRA initialization"
                )
            training_before = training_identity(config["code_root"])
            training_after = training_identity(code_root)
            if training_before != training_after:
                raise PermissionError(
                    "Technical repair must retain the numerical objective and LR rule"
                )
            task = root / "orchestration/tasks/TECHNICAL"
            state, intent = read_json(task / "STATE.json"), read_json(task / "INTENT.json")
            queue = slurm.queue(user)
            if (
                state.get("status") != "FAILED"
                or state.get("job_id") in queue
                or str(controller_job_id) in queue
            ):
                raise PermissionError("Old technical job/controller are not both terminal")
            terminal = slurm.terminal(state["job_id"])
            controller_terminal = slurm.terminal(str(controller_job_id))
            if terminal != read_json(task / "TERMINAL.json") or terminal.get("state") not in (
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
            ):
                raise PermissionError("Technical failure accounting differs")
            if any(
                terminal.get(k) != v
                for k, v in dict(
                    user=user, account="rose", qos=TEACHER_QOS, name=intent["job_name"]
                ).items()
            ):
                raise PermissionError("Failed technical job identity differs")
            controller_fields = slurm.show(str(controller_job_id))
            command = Path(controller_fields.get("Command", "")).resolve(strict=True)
            if (
                not controller_terminal
                or controller_terminal.get("user") != user
                or controller_terminal.get("account") != "rose"
                or controller_terminal.get("state")
                not in ("FAILED", "CANCELLED", "COMPLETED", "TIMEOUT")
                or controller_fields.get("UserId", "").split("(")[0] != user
                or not command.is_relative_to(root)
            ):
                raise PermissionError("Retired controller does not belong to this experiment")
            baseline = []
            original_config = read_json(root / "orchestration/CONFIG.json")
            baseline_source_hash = object_hash(read_json(original_config["source_manifest"]))
            for rank in range(3):
                path = root / "orchestration/tasks" / f"BASELINE_{rank}"
                bs, bi = read_json(path / "STATE.json"), read_json(path / "INTENT.json")
                if bs.get("status") in (
                    "FAILED",
                    "SUBMISSION_UNKNOWN",
                    "RELEASE_UNKNOWN",
                ) or not bs.get("job_id"):
                    raise PermissionError("Baseline is not a known retained attempt")
                if bi.get("source_sha256") != baseline_source_hash:
                    raise PermissionError(
                        "Baseline source identity differs from original registration"
                    )
                baseline.append(
                    dict(
                        task_id=f"BASELINE_{rank}",
                        job_id=bs["job_id"],
                        intent_sha256=object_hash(bi),
                    )
                )
            reuse = _raw32(root)
            amendment = read_json(root / "AMENDMENT.json")
            authorization = dict(
                scope="failed_zero_update_technical_repair",
                revision_id=revision_id,
                amendment_sha256=object_hash(amendment),
                config_sha256=amendment["config_sha256"],
                previous_source_commit=previous["source_commit"],
                previous_revision_sha256=previous["revision_sha256"],
                source_commit=source_commit,
                code_root=str(code_root),
                source_manifest=str(source_manifest),
                source_manifest_sha256=new_hash,
                previous_manifest_sha256=old_hash,
                changed_files=changed,
                generation_before=before,
                generation_after=after,
                training_before=training_before,
                training_after=training_after,
                terminal=terminal,
                controller_terminal=controller_terminal,
                baseline_attempts=baseline,
                reuse_preflight=reuse,
                task_files=_tree_manifest(task),
                technical_files=_tree_manifest(root / "technical"),
            )
            save_json(directory / "REQUEST.json", signature, exclusive=True)
            save_json(directory / "AUTHORIZATION.json", authorization, exclusive=True)
        else:
            if read_json(directory / "REQUEST.json") != signature:
                raise PermissionError("Interrupted maintenance arguments differ")
            authorization = read_json(directory / "AUTHORIZATION.json")
        # From this point the immutable authorization records every mutation target.
        target = root / "technical_incidents" / ("source_repair_" + revision_id)
        target.mkdir(parents=True, exist_ok=True)
        for source, destination, manifest in (
            (root / "orchestration/tasks/TECHNICAL", target / "task", authorization["task_files"]),
            (root / "technical", target / "technical", authorization["technical_files"]),
        ):
            if not destination.exists():
                _verify_tree(source, manifest)
                os.rename(source, destination)
                for parent in (source.parent, destination.parent):
                    handle = os.open(parent, os.O_RDONLY)
                    try:
                        os.fsync(handle)
                    finally:
                        os.close(handle)
            _verify_tree(destination, manifest)
        raw_source = target / "technical/preflight/rollouts"
        raw_target = root / "technical/preflight/rollouts"
        raw_target.mkdir(parents=True, exist_ok=True)
        for name, checksum in authorization["reuse_preflight"]["files"].items():
            old, new = raw_source / name, raw_target / name
            if file_hash(old) != checksum:
                raise PermissionError("Preserved raw answer changed")
            if not new.exists():
                handle, temporary = tempfile.mkstemp(prefix=".reuse-", dir=raw_target)
                try:
                    with old.open("rb") as src, os.fdopen(handle, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    os.link(temporary, new)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            if file_hash(new) != checksum:
                raise PermissionError("Reused raw answer changed")
        for retained in authorization["baseline_attempts"]:
            path = root / "orchestration/tasks" / retained["task_id"]
            if (
                object_hash(read_json(path / "INTENT.json")) != retained["intent_sha256"]
                or read_json(path / "STATE.json")["job_id"] != retained["job_id"]
            ):
                raise PermissionError("Baseline attempt was changed during maintenance")
        return save_json(
            directory / "APPLIED.json",
            dict(
                status="APPLIED_ONCE_REUSE_32_NO_BASELINE_RESTART",
                revision_id=revision_id,
                authorization_sha256=object_hash(authorization),
                reused_records=32,
                preserved_directory=str(target),
                preserved_task_files_sha256=object_hash(authorization["task_files"]),
                preserved_technical_files_sha256=object_hash(authorization["technical_files"]),
            ),
            exclusive=True,
        )
