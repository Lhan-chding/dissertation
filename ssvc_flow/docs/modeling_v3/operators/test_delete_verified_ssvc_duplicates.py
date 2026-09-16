"""Cleanup safety fixtures: only tiny files under pytest's temporary root are removed.

Both subjects load beside this file, so these fixtures remain reproducible when
the three helpers and tests are copied together into the operator directory.
"""

import importlib.util
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_fixture_subject", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


audit = load("audit_ssvc_redundancy")
cleanup = load("delete_verified_ssvc_duplicates")


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "ssvc"
    next_root = root / "runs/NEXT_20260909"
    audit_root = root / "storage_audits/next_20260909_01"
    output = root / "storage_audits/duplicate_cleanup_20260916_01"
    next_root.mkdir(parents=True)
    audit_root.mkdir(parents=True)
    protected = next_root / "R4_server_147871_attempt_0/R4"
    copying = next_root / "R4_server_149841_attempt_0/R4/.inherited_parent.copying"
    for name in cleanup.NAMES:
        target = next_root / name
        target.mkdir()
        (target / "small.log").write_text("Preserve this tiny original log.\n")
        (target / "result.txt").write_text(name + "\n")
    protected.mkdir()
    (protected / "nested").mkdir()
    (protected / "state.pt").write_bytes(b"protected 35-step trajectory tiny fixture")
    (protected / "nested/adam.json").write_text('{"step":35}')
    copying.parent.mkdir()
    shutil.copytree(protected, copying)
    archives = []
    for name in cleanup.NAMES:
        path = next_root / (name + ".tar.gz")
        with tarfile.open(path, "w:gz") as stream:
            stream.add(next_root / name, arcname=name)
        archives.append(audit.audit_archive(path, next_root / name, audit_root))
    copied = audit.audit_copying(copying, protected, audit_root)
    assert all(row["status"] == "ALL_MEMBERS_PRESERVED" for row in archives)
    assert copied["status"] == "ALL_MEMBERS_PRESERVED"
    summary = {
        "schema": "ssvc-redundancy-audit-1",
        "deletion_performed": False,
        "archives": archives,
        "copying": copied,
    }
    summary_path = audit_root / "AUDIT_SUMMARY.json"
    audit.write_json(summary_path, summary)
    script_original = Path(cleanup.__file__).read_bytes()
    script = root / "delete_helper_fixture.py"
    script.write_bytes(script_original)
    for key, value in {
        "ROOT": root,
        "NEXT": next_root,
        "AUDIT": audit_root,
        "OUT": output,
        "COPYING": copying,
        "PROTECTED": protected,
        "__file__": str(script),
    }.items():
        monkeypatch.setattr(cleanup, key, value)
    monkeypatch.setattr(cleanup, "quota", lambda: {"fixture": "no external filesystem queried"})
    monkeypatch.setattr(sys, "platform", "linux")
    for key, value in {
        "SLURM_JOB_ID": "999001",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "SLURM_JOB_GPUS": "",
        "SLURM_STEP_GPUS": "",
    }.items():
        monkeypatch.setenv(key, value)
    original_files = {str(p): p.read_bytes() for p in next_root.rglob("*") if p.is_file()}
    return {
        "root": root,
        "next": next_root,
        "audit": audit_root,
        "out": output,
        "protected": protected,
        "copying": copying,
        "summary": summary_path,
        "summary_sha256": audit.hash_path(summary_path),
        "original_files": original_files,
        "script": script,
        "monkeypatch": monkeypatch,
    }


def run(case, names=None, copying=False, sha256=None):
    argv = [str(case["script"]), "--audit-summary-sha256", sha256 or case["summary_sha256"]]
    if names is not None:
        argv += ["--archives", *names]
    if copying:
        argv += ["--delete-copying"]
    case["monkeypatch"].setattr(sys, "argv", argv)
    cleanup.main()


def assert_archives_exist(case):
    assert all((case["next"] / (name + ".tar.gz")).is_file() for name in cleanup.NAMES)


def repin(case, value):
    case["summary"].write_text(json.dumps(value))
    case["summary_sha256"] = audit.hash_path(case["summary"])


def test_valid_archive_and_149841_chain_preserves_originals_and_small_logs(case):
    names = ["R4_server_147871_attempt_0", "R4_server_149841_attempt_0"]
    run(case, names, copying=True)
    assert not case["copying"].exists()
    for name in cleanup.NAMES:
        assert (case["next"] / name).is_dir()
        assert (
            case["next"] / name / "small.log"
        ).read_text() == "Preserve this tiny original log.\n"
        assert (case["next"] / (name + ".tar.gz")).exists() == (name not in names)
    for path, contents in case["original_files"].items():
        original = Path(path)
        if original.is_relative_to(case["copying"]) or original in [
            case["next"] / (n + ".tar.gz") for n in names
        ]:
            continue
        assert original.read_bytes() == contents
    receipt = json.loads((case["out"] / "DELETE_RECEIPT.json").read_text())
    assert receipt["status"] == "PASS" and receipt["all_targets_absent"]
    assert receipt["scientific_originals_deleted"] is False
    ledger = [
        json.loads(line) for line in (case["out"] / "DELETE_LEDGER.jsonl").read_text().splitlines()
    ]
    assert ledger[-1]["deleted"] == str(case["copying"])
    assert [row["deleted"] for row in ledger[:-1]] == [
        str(case["next"] / (n + ".tar.gz")) for n in names
    ]


def test_archive_only_keeps_copying(case):
    run(case, [cleanup.NAMES[0]])
    assert case["copying"].is_dir()
    assert (
        case["protected"] / "state.pt"
    ).read_bytes() == b"protected 35-step trajectory tiny fixture"


@pytest.mark.parametrize("target", ["protected", "archive", "copying", "retained_log"])
def test_prior_original_archive_or_copying_tamper_refused(case, target):
    paths = {
        "protected": case["protected"] / "state.pt",
        "archive": case["next"] / "R4_server_149841_attempt_0.tar.gz",
        "copying": case["copying"] / "state.pt",
        "retained_log": case["next"] / "R4_server_149841_attempt_0/small.log",
    }
    path = paths[target]
    previous = path.stat()
    old = path.read_bytes()
    path.write_bytes(bytes([old[0] ^ 1]) + old[1:])
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError):
        run(case, ["R4_server_149841_attempt_0"], copying=True)
    assert_archives_exist(case)
    assert case["copying"].is_dir()


def test_full_summary_pin_required(case):
    with pytest.raises(ValueError):
        run(case, [cleanup.NAMES[0]], sha256="0" * 64)
    assert_archives_exist(case)


def test_changed_ledger_refused(case):
    row = json.loads(case["summary"].read_text())["archives"][0]
    Path(row["member_ledger"]).write_text("{}\n")
    with pytest.raises(ValueError):
        run(case, [cleanup.NAMES[0]])
    assert_archives_exist(case)


def test_unverified_status_cannot_be_selected(case):
    summary = json.loads(case["summary"].read_text())
    summary["archives"][0]["status"] = "DIFFERENT"
    repin(case, summary)
    with pytest.raises(ValueError):
        run(case, [cleanup.NAMES[0]])
    assert_archives_exist(case)


def test_outside_archive_name_refused(case):
    outside = case["root"].parent / "outside.tar.gz"
    outside.write_bytes(b"do not remove")
    with pytest.raises(SystemExit):
        run(case, [str(outside)])
    assert outside.read_bytes() == b"do not remove"
    assert_archives_exist(case)


def test_pinned_archive_path_outside_fixed_scope_refused(case):
    summary = json.loads(case["summary"].read_text())
    outside = case["root"].parent / (cleanup.NAMES[0] + ".tar.gz")
    outside.write_bytes(b"outside original")
    summary["archives"][0]["archive"] = str(outside)
    repin(case, summary)
    with pytest.raises(ValueError):
        run(case, [cleanup.NAMES[0]])
    assert outside.read_bytes() == b"outside original"
    assert_archives_exist(case)


def test_copying_new_unverified_file_blocks_entire_selection(case):
    (case["copying"] / "unverified").write_bytes(b"new raw bytes")
    with pytest.raises(ValueError):
        run(case, ["R4_server_149841_attempt_0"], copying=True)
    assert_archives_exist(case)
    assert (case["copying"] / "unverified").read_bytes() == b"new raw bytes"


def test_symlink_in_protected_path_refused(case):
    path = case["protected"] / "state.pt"
    outside = case["root"].parent / "outside_original"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        run(case, ["R4_server_149841_attempt_0"], copying=True)
    assert_archives_exist(case)
    assert outside.is_file()


@pytest.mark.parametrize(
    "key,value", [("SLURM_JOB_ID", ""), ("SLURM_JOB_GPUS", "0"), ("CUDA_VISIBLE_DEVICES", "0")]
)
def test_cpu_job_gates(case, key, value):
    case["monkeypatch"].setenv(key, value)
    with pytest.raises(ValueError):
        run(case, [cleanup.NAMES[0]])
    assert_archives_exist(case)


def test_no_selection_refused(case):
    with pytest.raises(ValueError):
        run(case)
    assert_archives_exist(case)


def test_duplicate_selection_refused(case):
    with pytest.raises(ValueError):
        run(case, [cleanup.NAMES[0], cleanup.NAMES[0]])
    assert_archives_exist(case)


def test_existing_cleanup_receipt_directory_refused(case):
    case["out"].mkdir()
    (case["out"] / "original_receipt").write_text("keep")
    with pytest.raises(FileExistsError):
        run(case, [cleanup.NAMES[0]])
    assert_archives_exist(case)


def test_plan_write_failure_precedes_any_delete(case):
    original = cleanup.write_new

    def fail_plan(path, value):
        if path.name == "DELETE_PLAN.json":
            raise OSError("fixture quota failure")
        return original(path, value)

    case["monkeypatch"].setattr(cleanup, "write_new", fail_plan)
    with pytest.raises(OSError):
        run(case, [cleanup.NAMES[0]])
    assert_archives_exist(case)


def test_protected_change_after_plan_must_prevent_first_delete(case):
    original = cleanup.write_new

    def change_after_plan(path, value):
        original(path, value)
        if path.name == "DELETE_PLAN.json":
            (case["protected"] / "state.pt").write_bytes(b"concurrent protected-original change")

    case["monkeypatch"].setattr(cleanup, "write_new", change_after_plan)
    with pytest.raises(ValueError):
        run(case, ["R4_server_149841_attempt_0"], copying=True)
    assert_archives_exist(case)
    assert case["copying"].is_dir()
