"""Byte-exact reconstruction checks for the compact analysis archive wrapper."""

import gzip
import importlib.util
import io
import json
import lzma
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "sr_f3_ce"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "compact_analysis_test_subject", SCRIPT_DIR / "compact_analysis.py"
)
compact_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact_analysis)
from compact_codec import Encoder  # noqa: E402


def gzip_bytes(raw):
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(raw)
    return out.getvalue()


def source_zip(tmp_path, members):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
    return source


def build_stream(path, records):
    with lzma.open(path, "wb") as stream:
        enc = Encoder(stream)
        for record in records:
            enc.write(record)


def intro():
    return {
        "format": "SR_F3_CE_REVERSIBLE_MEMBERS_V1",
        "source_zip": {"sha256": "0" * 64, "bytes": 0},
    }


def test_pack_verify_extract_every_original_member_exact(tmp_path):
    row = {"汉字": [1, 1.0, -0.0, 0.10000000000000002], "identity": {"x": "repeat"}}
    jsonl = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for _ in range(3))
    members = {
        "REPORT_ZH.md": "报告\n".encode(),
        "raw/blob.bin": b"\x00\xff\x00binary",
        "raw/empty.bin": b"",
        "records/answers.jsonl": jsonl,
        "release/ENDPOINT_RAW.jsonl.gz": gzip_bytes(jsonl),
        "records/empty.jsonl": b"",
        "settings/config.json": (json.dumps(row, indent=2) + "\n").encode(),
        "settings/compact.json": json.dumps(
            row, ensure_ascii=False, separators=(",", ":")
        ).encode(),
        "settings/unusual.json": b'{  "a":1,"a":2 }\n',
    }
    source = source_zip(tmp_path, members)
    data = tmp_path / "payload.xz"
    built = compact_analysis.pack_data(source, data, preset=0)
    verified = compact_analysis.verify_data(data)
    assert verified["status"] == "PASS_ALL_ORIGINAL_MEMBERS_BYTE_EXACT"
    assert verified["members"] == len(members)
    assert verified["endpoint_rows"] == 3
    assert verified["source_zip"] == built["source_zip"]
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    compact_analysis.verify_data(data, extracted)
    assert {
        path.relative_to(extracted).as_posix(): path.read_bytes()
        for path in extracted.rglob("*")
        if path.is_file()
    } == members


@pytest.mark.parametrize(
    "name", ["../escape", "/abs", "a/../b", "a\\b", "a//b", "a/./b", "a:", "", "a\x00b"]
)
def test_reject_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        compact_analysis.safe(name)


def test_pack_rejects_unsafe_member(tmp_path):
    source = source_zip(tmp_path, {"../escape.json": b"{}"})
    with pytest.raises(ValueError, match="Unsafe"):
        compact_analysis.pack_data(source, tmp_path / "payload.xz", preset=0)


def test_corrupted_lzma_rejected(tmp_path):
    source = source_zip(tmp_path, {"report.md": b"hello"})
    data = tmp_path / "payload.xz"
    compact_analysis.pack_data(source, data, preset=0)
    raw = bytearray(data.read_bytes())
    raw[len(raw) // 2] ^= 0x40
    data.write_bytes(raw)
    with pytest.raises((ValueError, EOFError, lzma.LZMAError)):
        compact_analysis.verify_data(data)


def test_reconstruction_hash_mismatch_rejected(tmp_path):
    data = tmp_path / "payload.xz"
    build_stream(
        data,
        [
            intro(),
            {"path": "a.txt", "kind": "bytes", "recipe": None},
            3,
            "aGVsbG8=",
            0,
            {
                "original": {"sha256": "0" * 64, "bytes": 5},
                "logical": {"sha256": "0" * 64, "bytes": 5},
                "records": 1,
            },
            {"end": True, "members": 1},
        ],
    )
    with pytest.raises(ValueError, match="Reconstruction mismatch"):
        compact_analysis.verify_data(data)


def test_overwrite_refused(tmp_path):
    source = source_zip(tmp_path, {"a.txt": b"hello"})
    data = tmp_path / "payload.xz"
    compact_analysis.pack_data(source, data, preset=0)
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_bytes(b"protected")
    with pytest.raises(ValueError, match="overwrite"):
        compact_analysis.verify_data(data, out)
    assert (out / "a.txt").read_bytes() == b"protected"


def test_duplicate_and_trailing_records_rejected(tmp_path):
    source = source_zip(tmp_path, {"a.txt": b"hello"})
    data = tmp_path / "payload.xz"
    built = compact_analysis.pack_data(source, data, preset=0)
    entry = built["members"][0]
    footer = {key: entry[key] for key in ("original", "logical", "records")}
    member = [{"path": "a.txt", "kind": "bytes", "recipe": None}, 3, "aGVsbG8=", 0, footer]
    duplicate = tmp_path / "duplicate.xz"
    build_stream(duplicate, [intro(), *member, *member, {"end": True, "members": 2}])
    with pytest.raises(ValueError, match="Duplicate"):
        compact_analysis.verify_data(duplicate)
    trailing = tmp_path / "trailing.xz"
    build_stream(trailing, [intro(), {"end": True, "members": 0}, None])
    with pytest.raises(ValueError, match="Trailing"):
        compact_analysis.verify_data(trailing)


def test_mixed_jsonl_bytes_preserved(tmp_path):
    raw = b'{"a":1}\n{ "a": 2 }\n{"a":3}'
    source = source_zip(tmp_path, {"answers.jsonl": raw})
    data = tmp_path / "payload.xz"
    compact_analysis.pack_data(source, data, preset=0)
    out = tmp_path / "out"
    out.mkdir()
    compact_analysis.verify_data(data, out)
    assert (out / "answers.jsonl").read_bytes() == raw


def test_extract_rejects_symlink_parent(tmp_path):
    source = source_zip(tmp_path, {"nested/escape.txt": b"hello"})
    data = tmp_path / "payload.xz"
    compact_analysis.pack_data(source, data, preset=0)
    out, external = tmp_path / "out", tmp_path / "external"
    out.mkdir()
    external.mkdir()
    (out / "nested").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        compact_analysis.verify_data(data, out)
    assert not (external / "escape.txt").exists()
