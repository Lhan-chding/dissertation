"""Persistent replay of the 19 previously inline tiny archive-audit fixtures."""

import gzip
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("audit_ssvc_redundancy.py")
SPEC = importlib.util.spec_from_file_location("audit_ssvc_test_subject", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def files(tmp_path):
    source = tmp_path / "example"
    source.mkdir()
    out = tmp_path / "audit"
    out.mkdir()
    return {"root": tmp_path, "source": source, "out": out, "archive": tmp_path / "example.tar.gz"}


def make(files, entries, extra=b""):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as stream:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
            elif kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = data.decode()
                stream.addfile(member)
            else:
                stream.addfile(member)
    files["archive"].write_bytes(gzip.compress(raw.getvalue() + extra))


def check(files):
    return audit.audit_archive(files["archive"], files["source"], files["out"])


def test_contents_same_and_originals_unchanged(files):
    make(files, [("file", b"hello", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"hello")
    original = files["archive"].read_bytes()
    result = check(files)
    assert result["status"] == "ALL_MEMBERS_PRESERVED"
    assert result["archive_sha256"] == hashlib.sha256(original).hexdigest()
    assert files["archive"].read_bytes() == original
    assert (files["source"] / "file").read_bytes() == b"hello"


def test_named_directory_mapping(files):
    make(files, [("example", b"", tarfile.DIRTYPE), ("example/file", b"hello", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"hello")
    result = check(files)
    assert result["mapping"] == "NAMED_DIRECTORY"
    assert result["status"] == "ALL_MEMBERS_PRESERVED"
    assert result["member_count"] == 2


def test_byte_difference(files):
    make(files, [("file", b"hello", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"HELLO")
    result = check(files)
    assert result["status"] == "DIFFERENT"
    assert result["eligible_duplicate_transfer_archive"] is False


def test_size_difference(files):
    make(files, [("file", b"a", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"longer")
    assert check(files)["status"] == "DIFFERENT"


def test_missing_counterpart(files):
    make(files, [("missing", b"a", tarfile.REGTYPE)])
    assert check(files)["status"] == "DIFFERENT"


def test_traversal(files):
    make(files, [("../outside", b"a", tarfile.REGTYPE)])
    assert check(files)["status"] == "ERROR"
    assert not (files["root"] / "outside").exists()


def test_absolute_member(files):
    make(files, [("/etc/no-audit-write", b"a", tarfile.REGTYPE)])
    assert check(files)["status"] == "ERROR"


def test_tar_symlink(files):
    make(files, [("link", b"file", tarfile.SYMTYPE)])
    assert check(files)["status"] == "ERROR"


def test_tar_hardlink(files):
    make(files, [("link", b"file", tarfile.LNKTYPE)])
    assert check(files)["status"] == "ERROR"


def test_counterpart_symlink(files):
    make(files, [("file", b"a", tarfile.REGTYPE)])
    (files["root"] / "outside").write_bytes(b"a")
    (files["source"] / "file").symlink_to(files["root"] / "outside")
    assert check(files)["status"] == "ERROR"


def test_counterpart_parent_symlink(files):
    make(files, [("dir/file", b"a", tarfile.REGTYPE)])
    outside = files["root"] / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(b"a")
    (files["source"] / "dir").symlink_to(outside, target_is_directory=True)
    assert check(files)["status"] == "ERROR"


def test_extra_nonzero_after_tar_end(files):
    make(files, [("file", b"a", tarfile.REGTYPE)], b"UNPRESERVED TRAILING PAYLOAD")
    (files["source"] / "file").write_bytes(b"a")
    result = check(files)
    assert result["status"] == "ERROR"
    assert "TRAILING_NONZERO_DATA_AFTER_TAR_END_MARKER" in result["errors"]


def test_concatenated_gzip_tar(files):
    make(files, [("file", b"a", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"a")
    files["archive"].write_bytes(files["archive"].read_bytes() + gzip.compress(b"SECOND MEMBER"))
    assert check(files)["status"] == "ERROR"


def test_corrupt_gzip_crc(files):
    make(files, [("file", b"a", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"a")
    data = bytearray(files["archive"].read_bytes())
    data[-8] ^= 1
    files["archive"].write_bytes(data)
    result = check(files)
    assert result["status"] == "ERROR"
    assert result["archive_sha256"] == hashlib.sha256(data).hexdigest()


def test_empty_archive_not_auto_deletable(files):
    make(files, [])
    assert check(files)["status"] == "ERROR"


def test_no_output_overwrite(files):
    make(files, [("file", b"a", tarfile.REGTYPE)])
    (files["source"] / "file").write_bytes(b"a")
    check(files)
    with pytest.raises(FileExistsError):
        check(files)


def test_copying_same_protected_original(files):
    copying = files["root"] / "copying"
    copying.mkdir()
    (copying / "state").write_bytes(b"old35step")
    (files["source"] / "state").write_bytes(b"old35step")
    assert (
        audit.audit_copying(copying, files["source"], files["out"])["status"]
        == "ALL_MEMBERS_PRESERVED"
    )
    assert (files["source"] / "state").read_bytes() == b"old35step"
    assert (copying / "state").read_bytes() == b"old35step"


def test_copying_different(files):
    copying = files["root"] / "copying"
    copying.mkdir()
    (copying / "state").write_bytes(b"wrong")
    (files["source"] / "state").write_bytes(b"old35step")
    assert audit.audit_copying(copying, files["source"], files["out"])["status"] == "DIFFERENT"


def test_copying_link_refused(files):
    copying = files["root"] / "copying"
    copying.mkdir()
    (copying / "state").symlink_to(files["source"] / "state")
    (files["source"] / "state").write_bytes(b"old35step")
    assert audit.audit_copying(copying, files["source"], files["out"])["status"] == "ERROR"
