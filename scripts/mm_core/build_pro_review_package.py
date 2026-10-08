#!/usr/bin/env python3
"""Package a prepared local evidence tree without modifying its files.

The staging root must contain PACKAGE_EXCLUSIONS.json with an explicit
{"excluded": [...]} declaration. No entries are silently excluded by this tool.
The output ZIP and its adjacent .zip.json receipt must both be new paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path

# This exact source is included in the ZIP and also used for build verification.
# It has no repository imports or third-party dependencies.
VERIFIER_SOURCE = r'''#!/usr/bin/env python3
"""Verify an MM-CORE review ZIP with Python's standard library; never extract it."""
import argparse
import hashlib
import json
import re
import stat
import unicodedata
import zipfile

METADATA = ("MANIFEST.json", "SHA256SUMS.txt", "verify_package.py")
POLICY = "PACKAGE_EXCLUSIONS.json"
DENIED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".h5", ".hdf5",
    ".pkl", ".pickle", ".gguf", ".ggml", ".tflite", ".model", ".npz", ".npy",
    ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
}
DENIED_NAMES = {
    ".ssh", ".aws", ".git", ".codex", "credentials", "credentials.json",
    "secrets.json", "secrets.yaml", "secrets.yml", "id_rsa", "id_dsa",
    "id_ecdsa", "id_ed25519", "authorized_keys",
}
PRIVATE_KEY = re.compile(rb"(?m)^-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[ \t]*\r?$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(name):
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def safe_name(name):
    require(isinstance(name, str) and bool(name), "UNSAFE_PATH")
    require(not any(unicodedata.category(c).startswith("C") for c in name), "UNSAFE_PATH")
    require(not any(c in name for c in '\\:<>"|?*'), "UNSAFE_PATH")
    parts = name.split("/")
    for part in parts:
        require(part not in ("", ".", "..") and part == part.rstrip(" ."), "UNSAFE_PATH")
        lower = canonical(part)
        device = lower.split(".", 1)[0]
        require(device not in {"con", "prn", "aux", "nul"}
                and not re.fullmatch(r"(?:com|lpt)[0-9]+", device), "UNSAFE_PATH")
        require(lower not in DENIED_NAMES and not lower.startswith(".env"),
                "FORBIDDEN_CREDENTIAL_PATH:" + name)
        require(not any(lower.endswith(suffix) for suffix in DENIED_SUFFIXES),
                "FORBIDDEN_WEIGHT_KEY_OR_ARCHIVE:" + name)
    return parts


def register_name(name, seen, *, directory=False):
    parts = safe_name(name)
    for length in range(1, len(parts) + 1):
        spelling = "/".join(parts[:length])
        key = canonical(spelling)
        kind = "directory" if length < len(parts) or directory else "file"
        if key in seen:
            old_spelling, old_kind = seen[key]
            require(old_spelling == spelling, "CASE_OR_UNICODE_COLLISION:" + name)
            require(old_kind == kind, "FILE_DIRECTORY_COLLISION:" + name)
            require(kind == "directory", "DUPLICATE_FILE:" + name)
        else:
            seen[key] = (spelling, kind)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "DUPLICATE_JSON_KEY:" + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("NONSTANDARD_JSON_NUMBER:" + value)


def parse_json(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def check_policy(policy):
    require(isinstance(policy, dict) and isinstance(policy.get("excluded"), list),
            "EXPLICIT_EXCLUSIONS_REQUIRED")
    for item in policy["excluded"]:
        require((isinstance(item, str) and bool(item.strip()))
                or (isinstance(item, dict) and bool(item)), "INVALID_EXCLUSION_DECLARATION")
    return policy["excluded"]


def inspect_block(block, tail, name):
    data = tail + block
    require(PRIVATE_KEY.search(data) is None, "PRIVATE_KEY_CONTENT:" + name)
    return data[-512:]


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names, seen = [], {}
        for info in infos:
            require(info.orig_filename == info.filename, "UNSAFE_PATH")
            register_name(info.filename, seen)
            require(not info.is_dir(), "DIRECTORY_ZIP_ENTRY_NOT_ALLOWED")
            mode = (info.external_attr >> 16) & 0xffff
            require(stat.S_IFMT(mode) in (0, stat.S_IFREG), "NONREGULAR_ZIP_ENTRY")
            require(not info.flag_bits & 1, "ENCRYPTED_ENTRY_NOT_ALLOWED")
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "UNSUPPORTED_COMPRESSION")
            names.append(info.filename)
        require(set(METADATA) | {POLICY} <= set(names), "MISSING_REQUIRED_METADATA")
        manifest = parse_json(archive.read("MANIFEST.json"))
        require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
                "INVALID_MANIFEST_VERSION")
        files = manifest.get("files")
        require(isinstance(files, dict), "INVALID_MANIFEST_FILES")
        excluded_hashes = ["MANIFEST.json", "SHA256SUMS.txt"]
        require(manifest.get("hash_scope_excludes") == excluded_hashes, "INVALID_HASH_SCOPE")
        expected = set(files) | set(excluded_hashes)
        require(not set(files) & set(excluded_hashes), "CIRCULAR_MANIFEST_HASH")
        require(expected == set(names), "EXTRA_OR_MISSING_FILES")
        require(manifest.get("archive_members") == sorted(expected), "MANIFEST_MEMBER_MISMATCH")
        policy = parse_json(archive.read(POLICY))
        require(manifest.get("excluded") == check_policy(policy)
                and manifest.get("excluded_source") == POLICY, "EXCLUSION_DECLARATION_MISMATCH")
        sums = {}
        for line in archive.read("SHA256SUMS.txt").decode("utf-8").splitlines():
            require(len(line) > 66 and line[64:66] == "  "
                    and re.fullmatch(r"[0-9a-f]{64}", line[:64]), "INVALID_SHA256SUMS_LINE")
            name = line[66:]
            safe_name(name)
            require(name not in sums, "DUPLICATE_CHECKSUM_ENTRY")
            sums[name] = line[:64]
        require(set(sums) == set(names) - {"SHA256SUMS.txt"}, "CHECKSUM_FILE_SET_MISMATCH")
        total = 0
        for info in infos:
            digest, size, tail = hashlib.sha256(), 0, b""
            # Reading every entry through EOF also executes zipfile's CRC check.
            with archive.open(info) as stream:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
                    tail = inspect_block(block, tail, info.filename)
            require(size == info.file_size, "ZIP_SIZE_MISMATCH:" + info.filename)
            if info.filename in files:
                entry = files[info.filename]
                require(isinstance(entry, dict) and type(entry.get("bytes")) is int
                        and entry["bytes"] >= 0, "INVALID_MANIFEST_ENTRY")
                require(entry["bytes"] == size and entry.get("sha256") == digest.hexdigest(),
                        "MANIFEST_HASH_OR_SIZE_MISMATCH:" + info.filename)
            if info.filename in sums:
                require(sums[info.filename] == digest.hexdigest(),
                        "CHECKSUM_MISMATCH:" + info.filename)
            total += size
        return {"status": "VERIFIED", "file_count": len(infos), "uncompressed_bytes": total,
                "crc_read_to_eof": True, "exact_file_set": True,
                "all_declared_hashes_match": True, "excluded": manifest["excluded"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="ZIP path; verification does not extract files")
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
'''

_VERIFIER = {"__name__": "embedded_package_verifier"}
exec(compile(VERIFIER_SOURCE, "verify_package.py", "exec"), _VERIFIER)
verify_archive = _VERIFIER["verify_archive"]
require = _VERIFIER["require"]
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
METADATA = _VERIFIER["METADATA"]
POLICY = _VERIFIER["POLICY"]


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def inventory(staging):
    require(staging.is_dir() and not staging.is_symlink(), "STAGING_MUST_BE_REAL_DIRECTORY")
    files, seen = {}, {}

    def scan_failed(error):
        raise error

    for directory, dirs, names in os.walk(staging, followlinks=False, onerror=scan_failed):
        dirs.sort()
        for name in sorted([*dirs, *names]):
            path = Path(directory) / name
            info = path.lstat()
            relative = path.relative_to(staging).as_posix()
            require(not stat.S_ISLNK(info.st_mode), "SYMLINK_NOT_ALLOWED:" + relative)
            require(
                stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode),
                "NONREGULAR_STAGING_ENTRY:" + relative,
            )
            _VERIFIER["register_name"](relative, seen, directory=stat.S_ISDIR(info.st_mode))
            require(
                _VERIFIER["canonical"](relative)
                not in {_VERIFIER["canonical"](n) for n in METADATA},
                "RESERVED_METADATA_NAME:" + relative,
            )
            if stat.S_ISREG(info.st_mode):
                files[relative] = fingerprint(info)
    require(POLICY in files, "EXPLICIT_EXCLUSIONS_REQUIRED:" + POLICY)
    return files


def info_for(name):
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def descriptor(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def build_package(staging, output):
    staging, output = Path(staging).absolute(), Path(output).absolute()
    require(not staging.is_symlink(), "SYMLINK_STAGING_NOT_ALLOWED")
    staging = staging.resolve()
    require(not output.resolve().is_relative_to(staging), "OUTPUT_INSIDE_STAGING")
    receipt_path = Path(str(output) + ".json")
    require(not os.path.lexists(output) and not os.path.lexists(receipt_path), "OUTPUT_EXISTS")
    files = inventory(staging)
    policy_data = (staging / POLICY).read_bytes()
    policy = _VERIFIER["parse_json"](policy_data)
    exclusions = _VERIFIER["check_policy"](policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths, published = [], []
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".pro-package-", delete=False
        ) as f:
            temporary_zip = Path(f.name)
        temporary_paths.append(temporary_zip)
        entries = {}
        with zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(files):
                source = staging / name
                require(source.resolve().is_relative_to(staging), "SOURCE_ESCAPED_STAGING")
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(source, flags)
                with os.fdopen(fd, "rb") as stream:
                    require(
                        fingerprint(os.fstat(stream.fileno())) == files[name],
                        "STAGING_CHANGED:" + name,
                    )
                    digest, size, tail = hashlib.sha256(), 0, b""
                    with archive.open(info_for(name), "w", force_zip64=True) as target:
                        while True:
                            block = stream.read(1024 * 1024)
                            if not block:
                                break
                            tail = _VERIFIER["inspect_block"](block, tail, name)
                            digest.update(block)
                            size += len(block)
                            target.write(block)
                    require(
                        fingerprint(os.fstat(stream.fileno())) == files[name],
                        "STAGING_CHANGED:" + name,
                    )
                    entries[name] = {"sha256": digest.hexdigest(), "bytes": size}
            verifier = VERIFIER_SOURCE.encode("utf-8")
            archive.writestr(info_for("verify_package.py"), verifier)
            entries["verify_package.py"] = descriptor(verifier)
            manifest = {
                "schema_version": 1,
                "files": entries,
                "archive_members": sorted([*entries, "MANIFEST.json", "SHA256SUMS.txt"]),
                "excluded": exclusions,
                "excluded_source": POLICY,
                "hash_scope_excludes": ["MANIFEST.json", "SHA256SUMS.txt"],
                "hash_scope_explanation": (
                    "MANIFEST excludes itself and SHA256SUMS to avoid cycles; "
                    "SHA256SUMS covers MANIFEST and every other member except itself."
                ),
                "zip_timestamp": list(FIXED_TIMESTAMP),
            }
            manifest_bytes = encoded(manifest)
            archive.writestr(info_for("MANIFEST.json"), manifest_bytes)
            checksums = {**entries, "MANIFEST.json": descriptor(manifest_bytes)}
            sums = "".join(f"{checksums[n]['sha256']}  {n}\n" for n in sorted(checksums))
            archive.writestr(info_for("SHA256SUMS.txt"), sums.encode("utf-8"))
        require(inventory(staging) == files, "STAGING_CHANGED_DURING_BUILD")
        verification = verify_archive(temporary_zip)
        digest = hashlib.sha256()
        with temporary_zip.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        receipt = {
            **verification,
            "archive": output.name,
            "sha256": digest.hexdigest(),
            "bytes": temporary_zip.stat().st_size,
            "staging_file_count": len(files),
            "staging_modified": False,
            "new_model_calls": 0,
        }
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".pro-receipt-", delete=False
        ) as f:
            f.write(encoded(receipt))
            temporary_receipt = Path(f.name)
        temporary_paths.append(temporary_receipt)
        # Exclusive hard links publish only completed, verified files without overwrite races.
        for source, target in ((temporary_zip, output), (temporary_receipt, receipt_path)):
            os.link(source, target)
            published.append((source, target))
        return receipt
    except BaseException:
        for source, target in reversed(published):
            if target.exists() and os.path.samestat(source.stat(), target.stat()):
                target.unlink()
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_package(args.staging, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
