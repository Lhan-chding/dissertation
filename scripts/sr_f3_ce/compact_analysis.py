"""Lossless member-preserving repacking of an already verified analysis ZIP."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import lzma
import time
import zipfile
from pathlib import Path, PurePosixPath

from compact_codec import Decoder, Encoder

CHUNK = 8 * 1024 * 1024


def digest(f):
    h = hashlib.sha256()
    n = 0
    for b in iter(lambda: f.read(CHUNK), b""):
        h.update(b)
        n += len(b)
    return {"sha256": h.hexdigest(), "bytes": n}


def safe(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or any(x in ("", ".", "..") for x in name.split("/"))
        or PurePosixPath(name).is_absolute()
    ):
        raise ValueError("Unsafe member path")
    return name


def dumps(v, r):
    return json.dumps(
        v,
        ensure_ascii=r[0],
        indent=r[1],
        separators=None if r[2] == 0 else (",", ":"),
        allow_nan=False,
    ).encode() + (b"\n" if r[3] else b"")


def recipe(raw, obj):
    for indent in (None, 2, 4):
        for sep in (0, 1):
            for ascii in (False, True):
                for nl in (True, False):
                    r = [ascii, indent, sep, nl]
                    if dumps(obj, r) == raw:
                        return r
    return None


class HashSink:
    def __init__(self, output=None):
        self.h = hashlib.sha256()
        self.n = 0
        self.output = output

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        if self.output is not None:
            self.output.write(b)
        return len(b)

    def flush(self):
        if self.output is not None:
            self.output.flush()

    def pin(self):
        return {"sha256": self.h.hexdigest(), "bytes": self.n}


def pack_data(source, destination, preset=3, max_members=None):
    started = time.monotonic()
    entries = []
    with source.open("rb") as f:
        source_pin = digest(f)
    with (
        zipfile.ZipFile(source) as z,
        lzma.open(destination, "wb", preset=preset) as compressed,
        io.BufferedWriter(compressed, 1024 * 1024) as stream,
    ):
        enc = Encoder(stream)
        enc.write({"format": "SR_F3_CE_REVERSIBLE_MEMBERS_V1", "source_zip": source_pin})
        infos = z.infolist()
        if max_members:
            infos = infos[:max_members]
        # Large raw first, other members sorted so related groups remain nearby.
        infos.sort(key=lambda i: (i.filename != "release/ENDPOINT_RAW.jsonl.gz", i.filename))
        for index, info in enumerate(infos):
            name = safe(info.filename)
            rawhash = HashSink()
            count = 0
            with z.open(name) as inputfile:
                if name.endswith(".jsonl.gz") or name.endswith(".jsonl"):
                    isgz = name.endswith(".gz")
                    if isgz:
                        if inputfile.read(10) != bytes.fromhex("1f8b08000000000002ff"):
                            raise ValueError("Only deterministic level-9 gzip is supported")
                        inputfile.seek(0)
                    raw = gzip.GzipFile(fileobj=inputfile) if isgz else inputfile
                    first = raw.readline()
                    obj = json.loads(first) if first else None
                    r = recipe(first, obj) if first else None
                    enc.write(
                        {"path": name, "kind": "gzip_jsonl" if isgz else "jsonl", "recipe": r}
                    )
                    # Hash the original compressed bytes separately from the logical JSONL.
                    line = first
                    while line:
                        rawhash.write(line)
                        value = json.loads(line) if r is not None else None
                        canonical = r is not None and dumps(value, r) == line
                        enc.write(1 if canonical else 2)
                        enc.write(value if canonical else line.decode("utf-8"))
                        count += 1
                        if count % 10000 == 0:
                            print(
                                json.dumps(
                                    {
                                        "phase": "encode",
                                        "path": name,
                                        "rows": count,
                                        "compressed_bytes": destination.stat().st_size,
                                        "seconds": round(time.monotonic() - started, 1),
                                    }
                                ),
                                flush=True,
                            )
                        line = raw.readline()
                    enc.write(0)
                    if isgz:
                        raw.close()
                else:
                    raw = inputfile.read()
                    rawhash.write(raw)
                    try:
                        obj = json.loads(raw) if name.endswith(".json") else None
                        r = recipe(raw, obj) if name.endswith(".json") else None
                    except (UnicodeDecodeError, ValueError):
                        r = None
                    enc.write(
                        {"path": name, "kind": "json" if r is not None else "bytes", "recipe": r}
                    )
                    enc.write(1 if r is not None else 3)
                    enc.write(obj if r is not None else base64.b64encode(raw).decode("ascii"))
                    enc.write(0)
                    count = 1
            with z.open(name) as original:
                original_pin = digest(original)
            footer = {"original": original_pin, "logical": rawhash.pin(), "records": count}
            enc.write(footer)
            entries.append({"path": name, **footer})
            if index % 500 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "encode",
                            "files": index + 1,
                            "total": len(infos),
                            "compressed_bytes": destination.stat().st_size,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
        enc.write({"end": True, "members": len(entries)})
    return {
        "source_zip": source_pin,
        "members": entries,
        "encode_seconds": time.monotonic() - started,
        "compressed_bytes": destination.stat().st_size,
    }


def verify_data(data, output=None):
    started = time.monotonic()
    files = []
    seen = set()
    if output and (output.is_symlink() or not output.is_dir()):
        raise ValueError("Output must be a real directory")
    with lzma.open(data, "rb") as stream:
        dec = Decoder(stream)
        intro = dec.read()
        if intro.get("format") != "SR_F3_CE_REVERSIBLE_MEMBERS_V1":
            raise ValueError("Wrong format")
        while True:
            head = dec.read()
            if head.get("end") is True:
                if head["members"] != len(files):
                    raise ValueError("Member count mismatch")
                try:
                    dec.read()
                except EOFError:
                    break
                raise ValueError("Trailing data")
            name = safe(head["path"])
            if name in seen:
                raise ValueError("Duplicate path")
            seen.add(name)
            dest = None
            if output:
                target = output / name
                parent = output
                for part in PurePosixPath(name).parts[:-1]:
                    parent = parent / part
                    if parent.is_symlink():
                        raise ValueError("Refuse symlink parent")
                    parent.mkdir(exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise ValueError("Refuse overwrite")
                dest = target.open("xb")
            original = HashSink(dest)
            logical = HashSink()
            count = 0
            writer = (
                gzip.GzipFile(fileobj=original, mode="wb", filename="", mtime=0)
                if head["kind"] == "gzip_jsonl"
                else original
            )
            try:
                while True:
                    tag = dec.read()
                    if type(tag) is not int or tag not in (0, 1, 2, 3):
                        raise ValueError("Bad record tag")
                    if tag == 0:
                        break
                    value = dec.read()
                    raw = (
                        dumps(value, head["recipe"])
                        if tag == 1
                        else value.encode("utf-8")
                        if tag == 2
                        else base64.b64decode(value, validate=True)
                    )
                    logical.write(raw)
                    writer.write(raw)
                    count += 1
                if writer is not original:
                    writer.close()
            finally:
                if dest:
                    dest.close()
            footer = dec.read()
            if footer != {"original": original.pin(), "logical": logical.pin(), "records": count}:
                raise ValueError(f"Reconstruction mismatch: {name}")
            files.append({"path": name, **footer})
            if len(files) % 500 == 0 or head["kind"] == "gzip_jsonl":
                print(
                    json.dumps(
                        {
                            "phase": "verify",
                            "files": len(files),
                            "path": name,
                            "seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return {
        "status": "PASS_ALL_ORIGINAL_MEMBERS_BYTE_EXACT",
        "source_zip": intro["source_zip"],
        "members": len(files),
        "endpoint_rows": next(
            (x["records"] for x in files if x["path"] == "release/ENDPOINT_RAW.jsonl.gz"), None
        ),
        "member_manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "verify_seconds": time.monotonic() - started,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "verify", "extract"])
    p.add_argument("--source", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--preset", type=int, default=3)
    a = p.parse_args()
    if a.mode == "build":
        if not a.source or a.data.exists():
            raise ValueError("Missing source or output exists")
        result = pack_data(a.source, a.data, a.preset)
        a.data.with_suffix(".source_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    else:
        if a.mode == "extract":
            if not a.output:
                raise ValueError("Output required")
            a.output.mkdir(exist_ok=False)
        result = verify_data(a.data, a.output if a.mode == "extract" else None)
        a.data.with_suffix(".verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            result if a.mode != "build" else {k: v for k, v in result.items() if k != "members"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
