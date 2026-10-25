"""Deletion boundary tests use temporary files, never the server project root."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/sr_f5/cleanup_project.py"
SPEC = importlib.util.spec_from_file_location("srf5_cleanup_project", PATH)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


@pytest.mark.parametrize("path", ["/tmp/x", "../x", "engine/../parents/x", "engine//x", "./x"])
def test_relative_paths_reject_escape(path):
    with pytest.raises(ValueError):
        cleanup.relative_parts(path)


@pytest.mark.parametrize(
    "path",
    [
        "parents/a",
        "runs/a/checkpoints/b",
        "code/x",
        "release/PRIMARY_ANALYSIS.json",
        "runs/a/rollouts/../x",
    ],
)
def test_protected_paths_cannot_enter_allowlist(path):
    with pytest.raises((ValueError, PermissionError)):
        cleanup.validate_member(dict(relative=path, bytes=65536, sha256="a" * 64))


def candidate(tmp_path, payload=b"x" * 65536):
    relative = "runs/example/rollouts/01.json"
    path = tmp_path / cleanup.SOURCE / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    member = dict(relative=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return path, member


def test_mismatch_preserves_file(tmp_path):
    path, member = candidate(tmp_path)
    member["sha256"] = "0" * 64
    fd = os.open(tmp_path, cleanup.DIR_FLAGS)
    try:
        result = cleanup.process_member(fd, member, apply=True)
    finally:
        os.close(fd)
    assert result["status"] == "PRESERVED_HASH_MISMATCH"
    assert path.exists()


def test_symlink_parent_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    source = tmp_path / cleanup.SOURCE
    source.mkdir(parents=True)
    (source / "runs").symlink_to(outside, target_is_directory=True)
    fd = os.open(tmp_path, cleanup.DIR_FLAGS)
    try:
        with pytest.raises(OSError), cleanup.parent_fd(fd, cleanup.SOURCE + "/runs/a/file"):
            pass
    finally:
        os.close(fd)


def test_regular_hardlink_removal_does_not_overclaim_space(tmp_path):
    path, member = candidate(tmp_path)
    retained = tmp_path / "retained"
    os.link(path, retained)
    fd = os.open(tmp_path, cleanup.DIR_FLAGS)
    try:
        result = cleanup.process_member(fd, member, apply=True)
    finally:
        os.close(fd)
    assert result["status"] == "DELETED"
    assert result["hash_verified"]
    assert result["allocated_bytes_released"] == 0
    assert retained.read_bytes() == b"x" * 65536
    assert not path.exists()


def test_dry_run_does_not_hash_or_delete(tmp_path, monkeypatch):
    path, member = candidate(tmp_path)
    monkeypatch.setattr(cleanup, "sha256_fd", lambda _: pytest.fail("dry run hashed content"))
    fd = os.open(tmp_path, cleanup.DIR_FLAGS)
    try:
        result = cleanup.process_member(fd, member, apply=False)
    finally:
        os.close(fd)
    assert result["status"] == "DRY_RUN_CANDIDATE"
    assert path.exists()


def test_change_during_hash_is_preserved(tmp_path, monkeypatch):
    path, member = candidate(tmp_path)
    original = cleanup.sha256_fd

    def alter(fd):
        digest = original(fd)
        path.write_bytes(b"y" * 65536)
        return digest

    monkeypatch.setattr(cleanup, "sha256_fd", alter)
    fd = os.open(tmp_path, cleanup.DIR_FLAGS)
    try:
        result = cleanup.process_member(fd, member, apply=True)
    finally:
        os.close(fd)
    assert result["status"] == "PRESERVED_CHANGED_DURING_HASH"
    assert path.exists()


def test_cache_symlink_unlinked_without_following(tmp_path):
    root = tmp_path / cleanup.OLD_CACHE
    root.mkdir(parents=True)
    target = tmp_path / "preserved"
    target.write_text("keep")
    link = root / "snapshot"
    link.symlink_to(target)
    fd = os.open(tmp_path, cleanup.DIR_FLAGS)
    try:
        entries = list(cleanup.cache_entries(fd))
        assert len(entries) == 1
        result = cleanup.process_cache_entry(fd, *entries[0], apply=True)
    finally:
        os.close(fd)
    assert result["status"] == "DELETED"
    assert target.read_text() == "keep"
    assert not link.is_symlink()


def test_preflight_must_cover_project_and_roots():
    base = dict(
        status="PASS",
        project_root=str(cleanup.ROOT),
        open_handles=[],
        checked_roots=[str(cleanup.ROOT / cleanup.SOURCE)],
        active_job_evidence=dict(
            status="PASS",
            project_root=str(cleanup.ROOT),
            conflicting_jobs=[],
            source="squeue/scontrol/open-handles",
        ),
    )
    cleanup.validate_preflight(base, [str(cleanup.ROOT / cleanup.SOURCE)])
    with pytest.raises(PermissionError):
        cleanup.validate_preflight(base, [str(cleanup.ROOT / cleanup.OLD_CACHE)])
    base["active_job_evidence"]["project_root"] = "/other"
    with pytest.raises(PermissionError):
        cleanup.validate_preflight(base, [])


def end_to_end_setup(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(cleanup, "ROOT", root)
    path, member = candidate(root)
    audit = root / "audit"
    audit.mkdir()
    archive = root / "retained.zip"
    archive.write_bytes(b"fixture archive identity")
    allowlist = root / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            {
                "schema": "SRF3_COMPACT_BACKED_CLEANUP_V1",
                "source_root": cleanup.SOURCE,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "existing_reconstruction_receipt": {
                    "status": "PASS_ALL_ORIGINAL_MEMBERS_BYTE_EXACT"
                },
                "members": [member],
            }
        )
    )
    (audit / "OPEN_HANDLES.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "project_root": str(root),
                "open_handles": [],
                "checked_roots": [str(root / cleanup.SOURCE)],
                "active_job_evidence": {
                    "status": "PASS",
                    "project_root": str(root),
                    "conflicting_jobs": [],
                    "source": "fixture preflight",
                },
            }
        )
    )
    args = SimpleNamespace(
        audit_dir=audit,
        archive=archive,
        allowlist=allowlist,
        apply=True,
        remove_old_model_cache=False,
    )
    return args, path


def test_apply_keeps_archive_and_unlisted_summary(tmp_path, monkeypatch):
    args, path = end_to_end_setup(tmp_path, monkeypatch)
    summary = path.parent / "summary.json"
    summary.write_text("retain this unlisted file")
    receipt = cleanup.run(args)
    assert receipt["status"] == "APPLY_COMPLETE"
    assert receipt["counts"] == {"DELETED": 1}
    assert receipt["logical_bytes_removed"] == 65536
    assert args.archive.exists() and summary.exists() and not path.exists()
    journal = [
        json.loads(line)
        for line in (args.audit_dir / "APPLY_JOURNAL.jsonl").read_text().splitlines()
    ]
    assert journal[0]["hash_verified"] is True
    assert receipt["directories_removed"] == 0


def test_archive_mismatch_blocks_all_deletion(tmp_path, monkeypatch):
    args, path = end_to_end_setup(tmp_path, monkeypatch)
    args.archive.write_bytes(b"changed archive identity")
    with pytest.raises(PermissionError, match="archive"):
        cleanup.run(args)
    assert path.exists()
