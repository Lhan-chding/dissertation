"""Focused regressions for the exact-project lossless cleanup boundary."""

import importlib.util
import json
import socket
import tarfile
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/sr_f6/storage_cleanup.py"
SPEC = importlib.util.spec_from_file_location("srf6_storage_cleanup", MODULE)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


@pytest.fixture
def source(tmp_path):
    root = tmp_path.resolve()
    directory = root / cleanup.SOURCE
    directory.mkdir(parents=True)
    (directory / "a.json").write_text(json.dumps({"answer": "1/3", "status": "FAIL"}))
    (directory / "nested").mkdir()
    (directory / "nested/b.jsonl").write_bytes(b'{"x":1}\n{"x":2}\n')
    return root


@pytest.mark.parametrize("path", ["../x", "/tmp/x", "a//b", "a/./b", "a/../b", "", "a\\b"])
def test_reject_unsafe_relative_paths(path):
    with pytest.raises(ValueError):
        cleanup.parts(path)


@pytest.mark.parametrize("path", ["cache/x", "sr_f3_ce_20261016/parents/a", cleanup.SOURCE])
def test_exact_source_allowlist_only(path):
    with pytest.raises(PermissionError):
        cleanup.source_member(path)


def test_audit_directory_creation_does_not_follow_symlinks(source):
    outside = source / "outside"
    outside.mkdir()
    (source / "storage_audits").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        cleanup.make_audit_directory(source)
    assert list(outside.iterdir()) == []


def test_archive_roundtrip_and_exact_unlink(source):
    entries = cleanup.inventory(source)
    archive = source / "kept.tar.gz"
    full = cleanup.archive_sources(source, archive, entries)
    check = cleanup.verify_archive(archive, full)
    assert check["members"] == 2
    assert check["gzip_crc_read_to_eof"] is True
    for row in full:
        result = cleanup.delete_member(source, row)
        assert result["sha256"] == row["sha256"]
        assert not (source / row["path"]).exists()
    assert (source / cleanup.SOURCE / "nested").is_dir()
    assert archive.is_file()


def test_symlink_file_and_directory_are_rejected(source):
    directory = source / cleanup.SOURCE
    (source / "outside").mkdir()
    (directory / "linked_dir").symlink_to(source / "outside", target_is_directory=True)
    with pytest.raises(PermissionError):
        cleanup.inventory(source)
    (directory / "linked_dir").unlink()
    (directory / "linked_file").symlink_to(directory / "a.json")
    with pytest.raises(PermissionError):
        cleanup.inventory(source)


def test_changed_source_is_not_deleted(source):
    full = cleanup.archive_sources(source, source / "kept.tar.gz", cleanup.inventory(source))
    path = source / full[0]["path"]
    path.write_text("changed content")
    with pytest.raises(PermissionError):
        cleanup.delete_member(source, full[0])
    assert path.read_text() == "changed content"


def test_only_mount_local_device_may_differ(source):
    current = cleanup.inventory(source)[0]["stat"]
    archived = {**current, "dev": current["dev"] + 99}
    assert cleanup.same_source_stat(source, archived, current)
    for key in ("ino", "size", "mtime_ns", "ctime_ns", "mode", "nlink", "blocks"):
        assert not cleanup.same_source_stat(source, {**archived, key: archived[key] + 1}, current)
    assert not cleanup.same_source_stat(source, archived, {**current, "dev": current["dev"] + 1})


def test_replaced_parent_symlink_cannot_escape(source):
    full = cleanup.archive_sources(source, source / "kept.tar.gz", cleanup.inventory(source))
    row = next(row for row in full if "/nested/" in row["path"])
    nested = source / cleanup.SOURCE / "nested"
    nested.rename(source / "outside")
    nested.symlink_to(source / "outside", target_is_directory=True)
    with pytest.raises(OSError):
        cleanup.delete_member(source, row)
    assert (source / "outside/b.jsonl").is_file()


def test_duplicate_archive_manifest_is_rejected(source):
    entries = cleanup.inventory(source)
    with pytest.raises(ValueError):
        cleanup.archive_sources(source, source / "bad.tar.gz", [entries[0], entries[0]])


def test_corrupt_archive_is_rejected(source):
    archive = source / "kept.tar.gz"
    full = cleanup.archive_sources(source, archive, cleanup.inventory(source))
    data = bytearray(archive.read_bytes())
    data[-8] ^= 1
    archive.write_bytes(data)
    with pytest.raises((OSError, EOFError, tarfile.ReadError)):
        cleanup.verify_archive(archive, full)


def test_archive_symlink_is_rejected(source):
    archive = source / "bad.tar.gz"
    entries = cleanup.inventory(source)
    row = entries[0]
    with tarfile.open(archive, "w:gz") as target:
        member = tarfile.TarInfo(row["path"])
        member.type = tarfile.SYMTYPE
        member.linkname = "/elsewhere"
        target.addfile(member)
    with pytest.raises(PermissionError):
        cleanup.verify_archive(archive, [{**row, "sha256": "0" * 64}])


def test_tampered_archive_content_is_rejected(source):
    archive = source / "kept.tar.gz"
    full = cleanup.archive_sources(source, archive, cleanup.inventory(source))
    full[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        cleanup.verify_archive(archive, full)


def test_cached_verification_reuses_proof_but_rejects_later_archive_change(source):
    directory = source / "audit"
    directory.mkdir()
    archive = directory / "RAW_COMPLETE.tar.gz"
    full = cleanup.archive_sources(source, archive, cleanup.inventory(source))
    check = cleanup.verify_archive(archive, full)
    (directory / "ARCHIVE_VERIFIED.json").write_text(json.dumps(check))
    (directory / "PREDELETE_PROOF.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "hostname": socket.gethostname(),
                "archive": check,
            }
        )
    )
    assert cleanup.cached_archive_verification(directory, full) == check
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(PermissionError):
        cleanup.cached_archive_verification(directory, full)
