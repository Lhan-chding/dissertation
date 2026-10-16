#!/usr/bin/env python3
"""Prepare selected data using immutable original images and frozen 652 APIs.

No model calls, rendering, parent audit, export, submission or overwrite occurs.
Run only inside an actual CPU Slurm allocation after complete protocol selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCIENTIFIC_COMMIT = "652e52d530520a276b6420ff0f4ee479dcde6db1"
SOURCE_MANIFEST_SHA256 = "dae3a506dda4fffee8da0e25a3e0881f4b45fa3259342c4160ecd0c09e03c9e0"
PREPARATION_MANIFEST_SHA256 = "db5a892d3c5afd175b4634ad75a1d87d3da0bdabe1d8e26f7b23bbfb5a0cf375"
ALLOWLIST_SHA256 = "761cc753e5c3173511c73a7c245af381cac8956f92f67d5f5e07c4bd3dad2ca1"
OLD_RUN_IDENTITY_SHA256 = "3667107b1bb90a90ffa1401067f3d222e513eb2de842648d5dd59d7c3ff8a489"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_code_tree(code, source):
    """Authenticate the exact regular-file set before importing any science code."""
    code = Path(code)
    if code.is_symlink() or not code.is_dir():
        raise PermissionError("Scientific code root must be a real directory, not a symlink")
    code = code.resolve(strict=True)
    rows = source.get("files")
    if not isinstance(rows, list) or not rows:
        raise PermissionError("Scientific source inventory must contain file rows")
    expected = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise PermissionError("Malformed scientific source inventory row")
        relative = Path(row["path"])
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.as_posix() != row["path"]
        ):
            raise PermissionError("Scientific inventory path is unsafe or noncanonical")
        if row["path"] in expected:
            raise PermissionError("Duplicate scientific inventory path: " + row["path"])
        if relative.suffix.lower() in (".pyc", ".pyo"):
            raise PermissionError("Bytecode is forbidden in scientific inventory")
        if (
            type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
        ):
            raise PermissionError("Malformed scientific source hash/size")
        expected[row["path"]] = row
    actual = set()
    for path in code.rglob("*"):
        mode = path.lstat().st_mode
        relative = path.relative_to(code).as_posix()
        if stat.S_ISLNK(mode):
            raise PermissionError("Symlink is forbidden in scientific code: " + relative)
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise PermissionError("Nonregular file is forbidden in scientific code: " + relative)
        if path.suffix.lower() in (".pyc", ".pyo"):
            raise PermissionError("Bytecode is forbidden in scientific code: " + relative)
        actual.add(relative)
    if actual != set(expected):
        raise PermissionError(
            "Scientific code file set differs; missing="
            + repr(sorted(set(expected) - actual))
            + "; extra="
            + repr(sorted(actual - set(expected)))
        )
    for relative in sorted(actual):
        row = expected[relative]
        path = code / relative
        if sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise PermissionError("Scientific source changed: " + relative)
    return {"regular_files_verified": len(actual), "exact_file_set": True}


def require_cpu_allocation():
    """Read live Slurm state; caller must first authenticate the imported 652 APIs.

    Shared with the static preparation helper: NEW need not exist. This function
    never submits or alters a job. It accepts only this process's real running
    CPU allocation with no requested/allocated GPU and an explicit empty CUDA view.
    """
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or int(job) < 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PermissionError("A real CPU Slurm allocation and empty CUDA view are required")
    from sr_f3_ce.orchestration import Slurm, physical_gpu_count

    fields = Slurm().show(job)
    uid = os.getuid()
    user = pwd.getpwuid(uid).pw_name
    if (
        fields.get("JobId") != job
        or fields.get("JobState") != "RUNNING"
        or fields.get("UserId") != f"{user}({uid})"
        or not fields.get("JobName", "").endswith("-LH")
        or fields.get("Account") != "rose"
    ):
        raise PermissionError("Actual Slurm job identity/state/owner/name/account differs")
    for key in ("ReqTRES", "AllocTRES"):
        tres = fields.get(key, "")
        cpus = re.findall(r"(?:^|,)cpu=([0-9]+)(?=,|$)", tres)
        if len(cpus) != 1 or int(cpus[0]) < 1 or physical_gpu_count(tres) != 0:
            raise PermissionError("Actual CPU/GPU allocation differs: " + key)
    node_list = fields.get("NodeList", "")
    if not node_list or node_list in ("(null)", "None"):
        raise PermissionError("Actual allocation nodes are unavailable")
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    hosts = result.stdout.split()
    hostname = socket.gethostname().split(".", 1)[0]
    if not hosts or hostname not in {host.split(".", 1)[0] for host in hosts}:
        raise PermissionError("This process is not on a node in the actual CPU allocation")
    return {
        "job_id": job,
        "fields": fields,
        "hostnames": hosts,
        "runtime_hostname": socket.gethostname(),
        "pid": os.getpid(),
    }


def parse_args(argv=None, *, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--prep-dir", type=Path, required=True)
    return parser.parse_args(argv)


def authenticated_context(args, *, fresh_data=False):
    if not __debug__:
        raise PermissionError("Python optimization is forbidden for these checks")
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdigit()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise PermissionError("A real CPU Slurm allocation and empty CUDA view are required")
    old, new, ablation, prep = (
        path.resolve(strict=True)
        for path in (args.old_root, args.new_root, args.ablation_root, args.prep_dir)
    )
    if (
        new in (old, ablation)
        or new.is_relative_to(old)
        or old.is_relative_to(new)
        or new.is_relative_to(ablation)
        or ablation.is_relative_to(new)
        or (new / "STOP").exists()
    ):
        raise PermissionError("Selected root must be independent and unstopped")
    manifest_path = prep / "PREPARATION_MANIFEST.json"
    if sha256(manifest_path) != PREPARATION_MANIFEST_SHA256:
        raise PermissionError("Original preparation manifest changed")
    preparation = json.loads(manifest_path.read_bytes())
    for row in preparation["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise PermissionError("Preparation inventory path is unsafe")
        path = prep / relative
        if (
            path.is_symlink()
            or sha256(path) != row["sha256"]
            or path.stat().st_size != row["bytes"]
        ):
            raise PermissionError("Original preparation artifact changed: " + row["path"])
    if sha256(prep / "COPY_ALLOWLIST.json") != ALLOWLIST_SHA256:
        raise PermissionError("Preparation allowlist changed")
    source_path = new / "deployment/SOURCE_MANIFEST.json"
    if sha256(source_path) != SOURCE_MANIFEST_SHA256:
        raise PermissionError("Selected scientific source inventory changed")
    source = json.loads(source_path.read_bytes())
    if source.get("code_commit") != SCIENTIFIC_COMMIT:
        raise PermissionError("Selected scientific commit differs from frozen 652")
    code = new / "code"
    if code.is_symlink() or not code.resolve(strict=True).is_relative_to(new):
        raise PermissionError("Selected code must remain inside the new root without a symlink")
    # Reject unregistered importable files/bytecode and all symlinks before imports.
    verify_code_tree(code, source)
    code = code.resolve(strict=True)
    sys.path.insert(0, str(code / "src"))
    from sr_f3_ce.ablation import verify_selection
    from sr_f3_ce.protocol import verify_source
    from sr_f3_ce.protocol_revision import revision_from_config

    for name, module in tuple(sys.modules.items()):
        if name.startswith(("sr_f3_ce", "sr_f12", "sr_f1", "sr_f2", "mm_core", "mm_dev")):
            module_file = getattr(module, "__file__", None)
            if module_file and not Path(module_file).resolve().is_relative_to(code):
                raise PermissionError("Imported scientific code is from another checkout: " + name)
    verify_source(source_path, code)
    cpu = require_cpu_allocation()
    selection = verify_selection(ablation)
    if selection.get("status") != "PROTOCOL_SELECTED":
        raise PermissionError("Complete nine-task protocol selection is required")
    revision = selection.get("selected_revision")
    if revision not in ("v2", "c2") or new != ablation.parent / ("selected_" + revision):
        raise PermissionError("Selected root/revision identity differs")
    if sha256(old / "RUN_IDENTITY.json") != OLD_RUN_IDENTITY_SHA256:
        raise PermissionError("Original input identity changed")
    expected = selection["candidate_identities"][{"v2": "P2", "c2": "P3"}[revision]]
    for path, key in (
        ("plan/PACKAGE_SHA256.json", "package_sha256"),
        ("plan/config/SR_F3_CE.json", "config_sha256"),
        ("plan/manifests/MODEL_INPUTS.jsonl", "model_inputs_sha256"),
    ):
        if sha256(new / path) != expected[key]:
            raise PermissionError("Selected plan differs from ablation: " + path)
    if (
        revision_from_config(json.loads((new / "plan/config/SR_F3_CE.json").read_bytes()))
        != revision
    ):
        raise PermissionError("Selected configuration revision differs")
    if fresh_data and (new / "data").exists():
        raise FileExistsError("Require a fresh data directory; preserve any earlier attempt")
    return {
        "old": old,
        "new": new,
        "ablation": ablation,
        "prep": prep,
        "selection": selection,
        "revision": revision,
        "cpu_allocation": cpu,
    }


def prepare_selected_data(args):
    context = authenticated_context(args, fresh_data=True)
    old, new, ablation, prep = (context[key] for key in ("old", "new", "ablation", "prep"))
    selection, revision = context["selection"], context["revision"]
    from sr_f3_ce.ablation import verify_selection
    from sr_f3_ce.data import ModelInputStore, load_frozen_dataset, object_sha256, read_jsonl
    from sr_f3_ce.preflight import bounded_file, copy_verified, file_hash, read_json
    from sr_f3_ce.rendering import fixed_spotchecks, write_json

    allow = read_json(prep / "COPY_ALLOWLIST.json")
    original = load_frozen_dataset(old / "plan")
    dataset = load_frozen_dataset(new / "plan")
    variant = {"v2": "P2", "c2": "P3"}[revision]
    expected = selection["candidate_identities"][variant]
    if (
        dataset.audit["protocol_revision"] != revision
        or file_hash(new / "plan/PACKAGE_SHA256.json") != expected["package_sha256"]
    ):
        raise PermissionError("Candidate package differs from selection")
    images = read_jsonl(old / "data/IMAGE_MANIFEST.jsonl")
    if list(original.tasks) != list(dataset.tasks) or list(dataset.tasks) != [
        r["qid"] for r in images
    ]:
        raise PermissionError("Task/image order changed")
    image_by_qid = {r["qid"]: r for r in images}
    for qid, task in dataset.tasks.items():
        before = original.tasks[qid]
        conserved = {k: v for k, v in before.items() if k not in ("prompt", "teacher")}
        current = {k: v for k, v in task.items() if k not in ("prompt", "teacher")}
        if current != conserved:
            raise PermissionError("Non-protocol task content changed: " + qid)
        render_key = object_sha256({k: task[k] for k in ("world", "chart", "render_style")})
        if render_key != image_by_qid[qid]["render_key"]:
            raise PermissionError("Image no longer renders the selected task: " + qid)
    spot = read_json(old / "data/SEMANTIC_RENDER_SPOTCHECK.json")
    visual = read_json(old / "data/VISUAL_SPOTCHECK_REVIEW.json")
    history = read_json(old / "data/HISTORICAL_WORLD_AUDIT.json")
    chain = read_json(old / "data/HISTORICAL_COVERAGE_CHAIN.json")
    if (
        fixed_spotchecks(dataset) != spot["items"]
        or visual["reviewed_count"] != len(spot["items"])
        or visual["issues_remaining"]
    ):
        raise PermissionError("Semantic/visual review is not equivalent")
    if visual["selection_sha256"] != file_hash(
        old / "data/SEMANTIC_RENDER_SPOTCHECK.json"
    ) or visual["image_manifest_sha256"] != file_hash(old / "data/IMAGE_MANIFEST.jsonl"):
        raise PermissionError("Visual review source hashes differ")
    if {r["qid"] for r in visual["items"]} != {r["qid"] for r in spot["items"]}:
        raise PermissionError("Visual review qid coverage differs")
    for row in visual["items"]:
        if (
            row["image_sha256"] != image_by_qid[row["qid"]]["image_sha256"]
            or row["semantic_review"] != "PASS"
            or row["source_image_review"] != "PASS"
        ):
            raise PermissionError("Visual review image/semantic result differs")
    if (
        history["status"] != "PASS"
        or history["overlapping_qids"]
        or chain["status"] != "REGISTERED_SR_F1_F12_F2_SYNTHETIC_WORLD_COVERAGE_VERIFIED"
    ):
        raise PermissionError("Historical world coverage failed")
    if chain["new_tasks_world_audit"]["sha256"] != file_hash(
        old / "data/HISTORICAL_WORLD_AUDIT.json"
    ):
        raise PermissionError("Historical world chain hash differs")
    for item in allow["copy_required_data"]:
        source = bounded_file(old, item["path"], allow_internal_symlink=False)
        if file_hash(source) != item["sha256"] or source.stat().st_size != item["bytes"]:
            raise PermissionError("Original data changed: " + item["path"])
    (new / "data").mkdir(exist_ok=False)
    for item in allow["copy_required_data"]:
        copy_verified(old / item["path"], new, item["path"], item["sha256"])
    write_json(new / "data/FROZEN_DATA_AUDIT.json", dataset.audit)
    expected_audit = {
        "v2": "9715e784490bf81d21674390c3d3cc65990648f0077053403a96d9bd958b6caa",
        "c2": "aeb37e548e551cab09a14419f08e8b38310faae6c790ee1e18414afcd9aa613c",
    }[revision]
    if file_hash(new / "data/FROZEN_DATA_AUDIT.json") != expected_audit:
        raise PermissionError("Candidate data audit differs from prechecked frozen source")
    store = ModelInputStore.from_files(
        new / "plan/manifests/MODEL_INPUTS.jsonl",
        new / "data/IMAGE_MANIFEST.jsonl",
        image_root=new / "data",
    )
    for qid in dataset.inputs:
        store.model_input(qid)
    receipt = {
        "status": "CPU_DATA_AND_SOURCE_IMAGES_VERIFIED",
        "protocol_revision": revision,
        "tasks": len(dataset.tasks),
        "historical_worlds": history["status"],
        "rendering": "SOURCE_IMAGES_VERIFIED",
        "model_input_routes_checked": len(dataset.inputs),
        "model_calls": 0,
        "native_processor": "NOT_RUN",
        "visual_spotchecks": "REUSED_BY_VERIFIED_IMAGE_AND_SEMANTIC_IDENTITY",
        "scientific_training_authorized_by_this_receipt": False,
    }
    write_json(new / "data/PREPARE_DATA_RECEIPT.json", receipt)
    final_allocation = require_cpu_allocation()
    if verify_selection(ablation) != selection:
        raise PermissionError("Selection changed during data preparation")
    write_json(
        new / "data/PROTOCOL_REVISION_REUSE.json",
        {
            "status": "BYTE_AND_SEMANTIC_EQUIVALENCE_VERIFIED",
            "protocol_revision": revision,
            "cpu_allocation_before": context["cpu_allocation"],
            "cpu_allocation_after": final_allocation,
            "helper_sha256": file_hash(Path(__file__).resolve()),
            "source_root": str(old),
            "source_run_identity_sha256": file_hash(old / "RUN_IDENTITY.json"),
            "protocol_selection_sha256": file_hash(ablation / "PROTOCOL_SELECTION.json"),
            "candidate_package_sha256": file_hash(new / "plan/PACKAGE_SHA256.json"),
            "model_inputs_sha256": file_hash(new / "plan/manifests/MODEL_INPUTS.jsonl"),
            "frozen_data_audit_sha256": file_hash(new / "data/FROZEN_DATA_AUDIT.json"),
            "prepare_data_receipt_sha256": file_hash(new / "data/PREPARE_DATA_RECEIPT.json"),
            "copy_allowlist_sha256": file_hash(prep / "COPY_ALLOWLIST.json"),
            "reused_receipts": {
                r["path"]: r["sha256"]
                for r in allow["copy_required_data"]
                if not r["path"].startswith("data/images/")
            },
            "task_order_and_nonprotocol_fields_equal": True,
            "render_keys_checked": 4096,
            "image_routes_hashed": 4096,
            "spotchecks_equal": len(spot["items"]),
            "all_task_worlds_equal_to_original": True,
            "historical_receipt_preserved": True,
            "historical_source_paths_are_original_provenance": True,
            "new_visual_review_claimed": False,
            "new_render_calls": 0,
            "model_calls": 0,
            "scientific_training_authorized_by_this_receipt": False,
        },
    )
    return {
        "status": "DATA_PREPARATION_PASS",
        "protocol_revision": revision,
        "root": str(new),
        "frozen_data_audit_sha256": expected_audit,
        "reuse_receipt_sha256": file_hash(new / "data/PROTOCOL_REVISION_REUSE.json"),
        "model_calls": 0,
        "gpu_calls": 0,
    }


def main(argv=None):
    result = prepare_selected_data(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
