#!/usr/bin/env python3
"""Bounded selected-root preparation; no sbatch, model call, ENGINE, or science launch.

Run only as a real CPU Slurm -LH job after complete PROTOCOL_SELECTION.
A failure preserves partial files. static refuses an existing selected root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

sys.dont_write_bytecode = True
COMMIT = "652e52d530520a276b6420ff0f4ee479dcde6db1"
TYPED = "4146e156ffa9a476b360daeb67503703faca138f"
SOURCE_SHA = "dae3a506dda4fffee8da0e25a3e0881f4b45fa3259342c4160ecd0c09e03c9e0"
REQUEST_SHA = "83f3688707b09d69d84fb4f7ce3a9a7f2706b26c978176d97a6fb9637189c220"
TYPE_SHA = "1a75df6bf5fa1c0ef22a23ee7fd766a8e22f5a1c6c1fe7d579c4ab6135106aaf"
HISTORY_SHA = "d67af2a3a00f44713c3bcb62d526f90f12e392f5fa0fe2d1872f54a3e1184d60"
HISTORY_NAME = "SR_F3_CE_PROTOCOL_STOP_EVIDENCE_FULL_UNDER400MB_20261016.tar.xz"
EXPERIMENT = "SR-F3-CE-20261016"
OLD_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016"
ABL_ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/ablation"
DATA_HELPER_SHA = "0d4bf63a9f5ca6c40e779ead0ff3973c2811839b01738ce43490db2c68938a16"
PREP_FILES = {
    "EXTERNAL_COPY_ALLOWLIST.json": (
        "04cd34a848e315f068be75d4351643d6f79bff357469f463c14d9726564c31ae"
    ),
    "OPERATIONS_SCHEMA.json": "87c3d351e041c9b7c6619042499e0419d6e2d9fd46e4c14a7deeb640594d6e11",
    "BOUND_FILES_POLICY.json": "3d1a12445e16118dba056e2172033db16adeabeb5a8d0bfa4f49682a48f16601",
    "RUN_IDENTITY.template.json": (
        "30158d7123c0a8835a4fac8a5279d4e24ed974c25c06556830fe14de493cdc16"
    ),
    "OBSERVABILITY_SOURCE_MANIFEST.template.json": (
        "be72b6626859a01e83f21cf4fd6ac5ca766638ebe28661a98dc9653a08e59319"
    ),
}


def fail(message):
    raise PermissionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded(root, relative, *, exists=True):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        fail("Unsafe relative path: " + str(relative))
    p = root / rel
    if p.is_symlink() or not p.resolve(strict=exists).is_relative_to(root):
        fail("Symlink or escaping path: " + str(p))
    return p


def check(path, record):
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["bytes"]
        or sha(path) != record["sha256"]
    ):
        fail("File identity differs: " + str(path))


def file_record(path):
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        fail("Already exists; retain and reconcile partial state: " + str(path))
    with path.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path, value):
    write_once(
        path,
        (
            json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode(),
    )


def copy_exact(source, target, expected, ledger):
    check(source, expected)
    if target.exists() or target.is_symlink():
        fail("Copy destination already exists: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    check(target, expected)
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    ledger.append(dict(source=str(source), target=str(target), **expected))


def verify_code(root):
    manifest_path = root / "deployment/SOURCE_MANIFEST.json"
    if sha(manifest_path) != SOURCE_SHA:
        fail("Frozen source manifest bytes differ")
    manifest = read(manifest_path)
    if manifest["code_commit"] != COMMIT:
        fail("Frozen scientific commit differs")
    if (root / "code").is_symlink():
        fail("Scientific code root cannot be a symlink")
    code = (root / "code").resolve(strict=True)
    files = manifest["files"]
    expected = {row["path"] for row in files}
    actual = set()
    for p in code.rglob("*"):
        if p.is_symlink():
            fail("Scientific source symlink")
        if p.is_file():
            actual.add(str(p.relative_to(code)))
        elif not p.is_dir():
            fail("Non-regular scientific code entry")
    if len(expected) != len(files) or actual != expected:
        fail("Scientific code complete regular-file inventory differs")
    for row in files:
        check(bounded(code, row["path"]), row)
    return manifest


def import_file(name, path, expected_sha256):
    """Compile the same authenticated bytes; never consult an auxiliary pyc."""
    if path.is_symlink() or not path.is_file():
        fail("Auxiliary source must be a regular file")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        fail("Auxiliary source changed before import: " + str(path))
    module = ModuleType(name)
    module.__file__ = str(path.resolve(strict=True))
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


def guard(args):
    abl = Path(args.ablation_root).resolve(strict=True)
    old = Path(args.old_root).resolve(strict=True)
    if str(abl) != ABL_ROOT or str(old) != OLD_ROOT:
        fail("This helper is scoped to the registered original and ablation roots")
    verify_code(abl)
    if args.mode == "static":
        source_root = abl
    else:
        # Reading the selection's declared revision only identifies the candidate
        # path. No mutation/import happens before complete validation below.
        revision = read(abl / "PROTOCOL_SELECTION.json").get("selected_revision")
        if revision not in {"v2", "c2"}:
            fail("No selected protocol")
        source_root = (abl.parent / ("selected_" + revision)).resolve(strict=True)
        verify_code(source_root)
    sys.path.insert(0, str(source_root / "code/src"))
    from sr_f3_ce import protocol
    from sr_f3_ce.ablation import verify_selection

    if Path(protocol.__file__).resolve() != source_root / "code/src/sr_f3_ce/protocol.py":
        fail("Wrong frozen scientific import root")
    selection = verify_selection(abl)
    if (
        selection.get("status") != "PROTOCOL_SELECTED"
        or selection.get("selected_revision") not in {"v2", "c2"}
        or selection.get("outputs") != 2304
        or len(selection.get("raw_evidence", [])) != 9
    ):
        fail("Complete successful selection required")
    new = abl.parent / ("selected_" + selection["selected_revision"])
    if new == old or new.is_relative_to(abl) or abl.is_relative_to(new):
        fail("Selected root overlaps its sources")
    if new.is_symlink():
        fail("Selected root cannot be a symlink")
    if args.mode != "static":
        if sha(new / "PROTOCOL_SELECTION.json") != sha(abl / "PROTOCOL_SELECTION.json"):
            fail("Copied selected decision differs")
        if any((new / p).exists() for p in ("STOP", "EXECUTION_FREEZE.json", "runs", "sealed")):
            fail("Preparation cannot mutate stopped or scientific state")
    prep = Path(args.prep_dir).resolve(strict=True)
    for relative, digest in PREP_FILES.items():
        if sha(bounded(prep, relative)) != digest:
            fail("Reviewed preparation artifact changed: " + relative)
    data_helper = Path(args.data_prep_dir).resolve(strict=True) / "prepare_selected_data.py"
    if sha(data_helper) != DATA_HELPER_SHA:
        fail("CPU allocation guard helper differs from its reviewed bytes")
    cpu = import_file(
        "selected_data_allocation_guard", data_helper, DATA_HELPER_SHA
    ).require_cpu_allocation()
    from sr_f3_ce.orchestration import TEACHER_QOS

    if (
        cpu["fields"].get("QOS") != TEACHER_QOS
        or cpu["fields"].get("Partition") != "cluster02"
        or cpu["runtime_hostname"].split(".", 1)[0] != "cpu-2"
    ):
        fail("Selected preparation requires the frozen TEACHER_QOS/cluster02/cpu-2 policy")
    return abl, old, new, selection, protocol, cpu


def candidate_plan(abl, selection):
    candidate = selection["candidate_identities"][
        {"v2": "P2", "c2": "P3"}[selection["selected_revision"]]
    ]
    plan = bounded(abl, candidate["plan"]).resolve(strict=True)
    for key, name in [
        ("package_sha256", "PACKAGE_SHA256.json"),
        ("config_sha256", "config/SR_F3_CE.json"),
        ("model_inputs_sha256", "manifests/MODEL_INPUTS.jsonl"),
    ]:
        if sha(plan / name) != candidate[key]:
            fail("Selected candidate differs: " + name)
    package = read(plan / "PACKAGE_SHA256.json")
    expected = {*package, "PACKAGE_SHA256.json"}
    actual = {str(p.relative_to(plan)) for p in plan.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in plan.rglob("*")):
        fail("Candidate plan contains unregistered members or symlinks")
    for name, digest in package.items():
        if sha(bounded(plan, name)) != digest:
            fail("Candidate package file changed: " + name)
    return plan, expected


def stage_static(args, abl, old, new, selection, cpu):
    if new.exists() or new.is_symlink():
        fail("static requires a never-created selected root")
    prep = Path(args.prep_dir).resolve(strict=True)
    plan, plan_names = candidate_plan(abl, selection)
    manifest = read(abl / "deployment/SOURCE_MANIFEST.json")
    allow = read(prep / "EXTERNAL_COPY_ALLOWLIST.json")
    if allow["old_server_root"] != str(old) or len(
        {r["destination"] for r in allow["records"]}
    ) != len(allow["records"]):
        fail("External copy allowlist root/targets differ")
    # All large source prerequisites are checked before creating the root.
    for row in allow["records"]:
        check(bounded(old, row["source_root_relative"]), row)
    history = Path(args.history_archive).resolve(strict=True)
    check(history, dict(bytes=219281700, sha256=HISTORY_SHA))
    history_receipt = history.with_name(history.name + ".verification.json")
    hr = read(history_receipt)
    if (
        history.name != HISTORY_NAME
        or hr.get("status") != "VERIFIED_FULL_LOSSLESS_REPACK_UNDER_400_DECIMAL_MB"
        or hr.get("actual_unique_new_output_rows") != 3904
        or hr.get("file_contents_changed") != 0
        or hr.get("files_removed") != 0
        or hr.get("scientific_experiment_complete") is not False
        or hr.get("archive", {}).get("bytes") != 219281700
        or hr.get("archive", {}).get("sha256") != HISTORY_SHA
    ):
        fail("Original complete history receipt differs")
    if sha(abl / "authorization/REQUEST.md") != REQUEST_SHA:
        fail("Adopted request differs")
    new.mkdir(exist_ok=False)
    ledger = []
    save(
        new / "deployment/STATIC_COPY_INTENT.json",
        dict(
            status="STARTED",
            root=str(new),
            ablation_root=str(abl),
            protocol_revision=selection["selected_revision"],
            selection_sha256=sha(abl / "PROTOCOL_SELECTION.json"),
            helper_sha256=sha(Path(__file__)),
            cpu_allocation=cpu,
        ),
    )
    for row in manifest["files"]:
        copy_exact(
            bounded(abl / "code", row["path"]),
            bounded(new, "code/" + row["path"], exists=False),
            row,
            ledger,
        )
    for relative in [
        "deployment/SOURCE_MANIFEST.json",
        "deployment/LOCAL_CHECKS.json",
        "deployment/LOCAL_VALIDATION.json",
        "deployment/LOCAL_TEST_OUTPUT.txt",
        "authorization/REQUEST.md",
        "PROTOCOL_SELECTION.json",
    ]:
        p = bounded(abl, relative)
        copy_exact(p, bounded(new, relative, exists=False), file_record(p), ledger)
    copy_exact(
        abl / "deployment/SOURCE_MANIFEST.json",
        new / "NATIVE_AUDIT_SOURCE.json",
        file_record(abl / "deployment/SOURCE_MANIFEST.json"),
        ledger,
    )
    for name in sorted(plan_names):
        copy_exact(
            bounded(plan, name),
            bounded(new, "plan/" + name, exists=False),
            file_record(plan / name),
            ledger,
        )
    for row in allow["records"]:
        copy_exact(
            bounded(old, row["source_root_relative"]),
            bounded(new, row["destination"], exists=False),
            {k: row[k] for k in ["sha256", "bytes"]},
            ledger,
        )
    for p in (history, history_receipt):
        copy_exact(p, new / "protocol_revision/history_v1" / p.name, file_record(p), ledger)
    raw_records = [selection["ablation_manifest"]]
    for cell in selection["raw_evidence"]:
        raw_records += [
            cell["complete"],
            cell["terminal"],
            cell["allocation"],
            *cell["raw_groups"],
            *cell["raw_artifacts"],
        ]
    if len({x["path"] for x in raw_records}) != len(raw_records):
        fail("Duplicate ablation retained path")
    for row in raw_records:
        copy_exact(
            bounded(abl, row["path"]),
            bounded(new, "protocol_revision/ablation/" + row["path"], exists=False),
            row,
            ledger,
        )
    copy_exact(
        abl / "PROTOCOL_SELECTION.json",
        new / "protocol_revision/ablation/PROTOCOL_SELECTION.json",
        file_record(abl / "PROTOCOL_SELECTION.json"),
        ledger,
    )
    ops = read(prep / "OPERATIONS_SCHEMA.json")
    for name, row in ops["helper_files"].items():
        copy_exact(new / "code" / row["path"], new / "operations" / COMMIT / name, row, ledger)
    finalizer = new / "code/scripts/sr_f3_ce/finalize_external_types.py"
    if sha(finalizer) != TYPE_SHA:
        fail("4146 finalizer source differs")
    copy_exact(
        finalizer,
        new / "operations" / TYPED / "finalize_external_types.py",
        file_record(finalizer),
        ledger,
    )
    for source, dest in ops["extension_copy_map"].items():
        p = new / "code" / source
        copy_exact(p, new / "native/observability" / COMMIT / dest, file_record(p), ledger)
    save(
        new / "deployment/INPUT_COPY_RECEIPT.json",
        dict(
            status="VERIFIED_SELECTED_STATIC_COPY",
            protocol_revision=selection["selected_revision"],
            source_manifest_sha256=SOURCE_SHA,
            selection_sha256=sha(new / "PROTOCOL_SELECTION.json"),
            external_allowlist_sha256=sha(prep / "EXTERNAL_COPY_ALLOWLIST.json"),
            records=ledger,
            model_calls=0,
            cpu_allocation=cpu,
        ),
    )
    verify_code(new)


def stage_external(new, selection, cpu):
    if (new / "RUN_IDENTITY.json").exists() or (new / "external/PANELS.json").exists():
        fail("external-v0 is a single pre-identity phase")
    from sr_f3_ce import external
    from sr_f3_ce.evaluation import object_hash
    from sr_f12.evaluation import durable_json

    history = read(new / "prior/HISTORICAL_IMAGES.json")
    s = new / "external/chartqapro_sources"
    qa = external.prepare_chartqa(
        new / "prior/raw/external/chartqa/TEST_MANIFEST.jsonl",
        new / "external/legacy_images",
        new / "external/chartqa",
        history_receipt=history,
    )
    pro = external.freeze_chartqapro(
        s / "test-00000-of-00001.parquet",
        new / "external/chartqapro",
        history_receipt=history,
        scorer_source=s / "evaluate_predictions.py",
        prompt_source=s / "vlmeval_chartqapro_utils.py",
        anls_wheel=s / "anls-0.0.2-py3-none-any.whl",
    )
    if (
        qa["status"] != "FROZEN"
        or pro["status"] != "FROZEN"
        or qa["image_count"] != 2500
        or pro["image_count"] != 256
    ):
        fail("Original external panel coverage changed")
    old = new / "protocol_revision/external_v1_original"
    equality = {}
    for pool in ("chartqa", "chartqapro"):
        names = ["MODEL_INPUTS.jsonl", "TASKS_AUDIT_ONLY.jsonl", "TYPE_REGISTRY.json"]
        if pool == "chartqapro":
            names += ["SELECTION.json", "FREEZE.json"]
        for name in names:
            if sha(old / pool / name) != sha(new / "external" / pool / name):
                fail("External non-path input changed")
            equality[pool + "/" + name] = sha(old / pool / name)
    oldqa = read(old / "chartqa/FREEZE.json")
    if {k: v for k, v in oldqa.items() if k != "image_root"} != {
        k: v for k, v in qa.items() if k != "image_root"
    } or qa["image_root"] != str(new / "external/legacy_images"):
        fail("More than the declared ChartQA absolute path changed")
    panels = dict(
        experiment_id=EXPERIMENT,
        status="FROZEN",
        CHARTQA=dict(
            root="external/chartqa",
            status="FROZEN",
            question_count=2500,
            freeze_sha256=sha(new / "external/chartqa/FREEZE.json"),
        ),
        CHARTQAPRO=dict(
            root="external/chartqapro",
            status="FROZEN",
            image_count=256,
            reason=None,
            freeze_sha256=sha(new / "external/chartqapro/FREEZE.json"),
        ),
    )
    finalizer = import_file(
        "selected_original_4146",
        new / "operations" / TYPED / "finalize_external_types.py",
        TYPE_SHA,
    )
    items = {}
    by_pool = {}
    images = {}
    for pool in ("CHARTQA", "CHARTQAPRO"):
        _, _, visible, store = finalizer.input_panel(
            new, panels[pool], pool, external, object_hash, sha
        )
        by_pool[pool] = visible
        items.update(visible)
        history_pixels = {row["image_sha256"]: row["pixel_sha256"] for row in history["rows"]}
        for qid, item in visible.items():
            request = store.model_input(qid)
            image = store.image_root / request["image_file"]
            pixel = external._pixel_hash(image.read_bytes())
            if pool == "CHARTQA":
                if history_pixels[request["image_sha256"]] != pixel:
                    fail("ChartQA history pixel mismatch")
            elif (
                pixel != item["image_pixel_sha256"]
                or pixel in history["pixel_sha256"]
                or request["image_sha256"] in history["image_sha256"]
            ):
                fail("ChartQAPro pixel or historical exclusion changed")
            images[str(image.relative_to(new))] = dict(
                sha256=request["image_sha256"], pixel_sha256=pixel
            )
    review_root = new / "protocol_revision/type_review_v1"
    a, by_a, scope, _ = finalizer.annotation_set(
        review_root / "REVIEW_A.json", items, EXPERIMENT, external, object_hash
    )
    b, by_b, other, _ = finalizer.annotation_set(
        review_root / "REVIEW_B.json", items, EXPERIMENT, external, object_hash
    )
    if (
        scope != other
        or scope != "692d36a065543555179e63d0454ec438115fe00227c75932bf5d152578c3fa48"
        or a["reviewer_id"] == b["reviewer_id"]
    ):
        fail("Original blind review scope changed")
    from collections import Counter

    summary = {}
    for pool, visible in by_pool.items():
        registry = external.build_type_registry(
            list(visible.values()),
            [r for q in sorted(visible) for r in (by_a[q], by_b[q])],
            outputs_unseen=True,
        )
        summary[pool] = dict(
            questions=len(registry["rows"]),
            status_counts=dict(Counter(r["status"] for r in registry["rows"])),
            type_counts=dict(Counter(r["kind"] for r in registry["rows"])),
        )
    if summary != read(review_root / "TYPE_AUDIT_RECEIPT.json")["summary"]:
        fail("Original type judgments changed")
    durable_json(new / "external/PANELS.json", panels)
    save(
        new / "protocol_revision/EXTERNAL_INPUT_REUSE.json",
        dict(
            status="VERIFIED_IDENTICAL_VISIBLE_INPUTS_WITH_DECLARED_ROOT_REBINDING",
            protocol_revision=selection["selected_revision"],
            protocol_selection_sha256=sha(new / "PROTOCOL_SELECTION.json"),
            same_file_hashes=equality,
            chartqa_image_root=dict(original=oldqa["image_root"], current=qa["image_root"]),
            history_receipt_sha256=sha(new / "prior/HISTORICAL_IMAGES.json"),
            new_v0_panels_sha256=sha(new / "external/PANELS.json"),
            visible_questions=len(items),
            images=images,
            annotation_scope_sha256=scope,
            original_review_files={
                n: sha(review_root / n) for n in ("REVIEW_A.json", "REVIEW_B.json")
            },
            summary=summary,
            new_human_review=False,
            model_calls=0,
            cpu_allocation=cpu,
        ),
    )


def stage_seal(args, new, selection, protocol, cpu):
    if (new / "RUN_IDENTITY.json").exists():
        fail("Identity is already sealed")
    prep = Path(args.prep_dir).resolve(strict=True)
    policy = read(prep / "BOUND_FILES_POLICY.json")
    names = set(policy["proposed_stable_paths"])
    names.update(str(p.relative_to(new)) for p in (new / "native/support_masks").glob("*.json"))
    names.update(
        "protocol_revision/type_review_v1/" + n
        for n in (
            "REVIEW_A.json",
            "REVIEW_B.json",
            "reviewer_evidence/A_MANIFEST.json",
            "reviewer_evidence/B_MANIFEST.json",
        )
    )
    for pool in ("chartqa", "chartqapro"):
        names.update(
            "external/" + pool + "/" + p.name
            for p in (new / "external" / pool).iterdir()
            if p.is_file()
        )
    if "external/PANELS.json" in names:
        fail("Live panel pointer cannot be identity-bound")
    from sr_f3_ce.bootstrap import verify_native_receipt_revision

    config = protocol.verify_package(new / "plan")["config"]
    processor = read(new / "native/PROCESSOR_RECEIPT.json")
    masks = read(new / "native/SUPPORT_MASK_RECEIPT.json")
    verify_native_receipt_revision(config, processor, masks)
    if (
        processor.get("status") != "PASS"
        or processor.get("model_calls") != 0
        or processor.get("weights_loaded") is not False
        or processor.get("cuda_initialized") is not False
        or processor.get("mask_receipt_sha256") != sha(new / "native/SUPPORT_MASK_RECEIPT.json")
    ):
        fail("Fresh native processor receipt failed")
    if (
        masks.get("status") != "PASS"
        or masks.get("target_cases") != 816
        or len(masks.get("files", [])) != 816
    ):
        fail("Fresh mask receipt coverage differs")
    for row in masks["files"]:
        if sha(bounded(new, row["path"])) != row["sha256"]:
            fail("Native mask bytes changed")
    if len(list((new / "native/support_masks").glob("*.json"))) != 816:
        fail("Fresh native mask inventory is incomplete")
    # Full native validation also runs in the mandatory future actual P0.
    value = read(prep / "RUN_IDENTITY.template.json")
    value.update(
        root=str(new),
        protocol_revision=selection["selected_revision"],
        package_manifest_sha256=sha(new / "plan/PACKAGE_SHA256.json"),
        bound_files={p: sha(bounded(new, p)) for p in sorted(names)},
    )
    save(new / "RUN_IDENTITY.json", value)
    protocol.verify_run(new)
    checked = protocol.verify_run_source_revision(new)
    ext = new / "native/observability" / COMMIT
    manifest = read(prep / "OBSERVABILITY_SOURCE_MANIFEST.template.json")
    for row in manifest["sources"]:
        check(bounded(ext, row["path"]), row)
    checks = read(new / "deployment/LOCAL_CHECKS.json")
    if (
        checks.get("status") != "PASS"
        or checks["code_commit"] != COMMIT
        or checks["source_manifest_sha256"] != SOURCE_SHA
        or checks["tests"]["passed"] != 687
        or checks["tests"]["output_sha256"] != sha(new / "deployment/LOCAL_TEST_OUTPUT.txt")
    ):
        fail("Actual 652 CPU validation provenance differs")
    if "687 passed" not in (new / "deployment/LOCAL_TEST_OUTPUT.txt").read_text():
        fail("Original CPU output lacks claimed count")
    cpu_validation = dict(
        status="PASS",
        scope="CPU_TOY_TRANSPARENCY_AND_ORCHESTRATION",
        gpu_validation_performed=False,
        new_model_generations=0,
        new_optimizer_steps=0,
        source_sha256={r["path"]: r["sha256"] for r in manifest["sources"]},
        tests_passed=687,
        evidence_origin=dict(
            original_local_checks_sha256=sha(new / "deployment/LOCAL_CHECKS.json"),
            original_validation_sha256=sha(new / "deployment/LOCAL_VALIDATION.json"),
            original_output_sha256=sha(new / "deployment/LOCAL_TEST_OUTPUT.txt"),
            source_archive_identity=SOURCE_SHA,
            original_checks_reused_for_byte_identical_sources=True,
            new_server_GPU_test=False,
        ),
    )
    save(ext / "CPU_VALIDATION.json", cpu_validation)
    manifest.update(
        protocol_source_guard=checked,
        run_identity_sha256=sha(new / "RUN_IDENTITY.json"),
        cpu_validation=dict(path="CPU_VALIDATION.json", **file_record(ext / "CPU_VALIDATION.json")),
    )
    save(ext / "SOURCE_MANIFEST.json", manifest)
    extension = import_file(
        "selected_static_extension",
        ext / "instrumented_execution.py",
        next(
            row["sha256"]
            for row in manifest["sources"]
            if row["path"] == "instrumented_execution.py"
        ),
    )
    extension.verify_extension(new, ext / "SOURCE_MANIFEST.json")
    save(
        new / "deployment/IDENTITY_SEAL_RECEIPT.json",
        dict(
            status="SELECTED_IDENTITY_SEALED_NO_SCIENCE",
            run_identity_sha256=sha(new / "RUN_IDENTITY.json"),
            protocol_source_guard=checked,
            cpu_allocation=cpu,
            initialization_job_terminal_not_attested_while_running=True,
            model_calls=0,
        ),
    )


def stage_typed(new, selection, protocol, cpu):
    protocol.verify_run(new)
    dispatcher = import_file(
        "selected_typed_verifier",
        new / "operations" / COMMIT / "dispatch_registered.py",
        next(
            row["sha256"]
            for row in read(new / "deployment/SOURCE_MANIFEST.json")["files"]
            if row["path"] == "scripts/sr_f3_ce/dispatch_registered.py"
        ),
    )
    from sr_f3_ce import external

    verified = dispatcher.verify_typed(new, external)
    if (
        not verified
        or verified["auxiliary_code_commit"] != TYPED
        or verified["preparation_script_sha256"] != TYPE_SHA
    ):
        fail("Actual new typed publication missing")
    original = new / "protocol_revision/type_review_v1"
    current = new / "external/revisions/typed_v1"
    for name in ("REVIEW_A.json", "REVIEW_B.json"):
        if sha(original / name) != sha(current / name):
            fail("Original review bytes changed")
    if (
        read(original / "TYPE_AUDIT_RECEIPT.json")["summary"]
        != read(current / "TYPE_AUDIT_RECEIPT.json")["summary"]
    ):
        fail("Original type judgments changed")
    save(
        new / "external/TYPE_REVIEW_REUSE_RECEIPT.json",
        dict(
            schema_version=1,
            status="ORIGINAL_BLIND_REVIEW_REPUBLISHED_FOR_IDENTICAL_VISIBLE_INPUTS",
            experiment_id=EXPERIMENT,
            new_root=str(new),
            run_identity_sha256=sha(new / "RUN_IDENTITY.json"),
            protocol_revision=selection["selected_revision"],
            protocol_selection_sha256=sha(new / "PROTOCOL_SELECTION.json"),
            original_type_audit_sha256=sha(original / "TYPE_AUDIT_RECEIPT.json"),
            original_review_files={
                n: sha(original / n) for n in ("REVIEW_A.json", "REVIEW_B.json")
            },
            input_equivalence_receipt_sha256=sha(
                new / "protocol_revision/EXTERNAL_INPUT_REUSE.json"
            ),
            typed_verification=verified,
            new_human_review=False,
            original_review_attestations_unchanged=True,
            original_review_limitations_preserved=True,
            model_calls=0,
            summary=read(current / "TYPE_AUDIT_RECEIPT.json")["summary"],
            cpu_allocation=cpu,
            initialization_job_terminal_not_attested_while_running=True,
            terminal_release_requirement=(
                "Main operator must verify this CPU job absent from squeue "
                "and sacct COMPLETED/0:0 before submitting P0; retain terminal "
                "in independent selected_init ledger."
            ),
            meaning=(
                "Original blind judgments republished after input equivalence checks. "
                "No new human audit is claimed."
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["static", "external-v0", "seal", "typed-receipt"], required=True
    )
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--data-prep-dir", required=True)
    parser.add_argument("--history-archive")
    args = parser.parse_args()
    if args.mode == "static" and not args.history_archive:
        parser.error("static requires --history-archive")
    abl, old, new, selection, protocol, cpu = guard(args)
    if args.mode == "static":
        stage_static(args, abl, old, new, selection, cpu)
    elif args.mode == "external-v0":
        stage_external(new, selection, cpu)
    elif args.mode == "seal":
        stage_seal(args, new, selection, protocol, cpu)
    else:
        stage_typed(new, selection, protocol, cpu)
    from sr_f3_ce.ablation import verify_selection

    if verify_selection(abl) != selection:
        fail("Selection changed during preparation")
    print(
        json.dumps(
            dict(
                status="COMPLETED_PREPARATION_PHASE",
                phase=args.mode,
                root=str(new),
                protocol_revision=selection["selected_revision"],
                job_id=cpu["job_id"],
                model_calls=0,
                science_submitted=False,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
