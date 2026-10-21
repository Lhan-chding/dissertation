#!/usr/bin/env python3
"""Portable streamed results-data delivery. Never loads model/checkpoint tensors."""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import resource
import stat
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

CHUNK = 1024 * 1024
FORBIDDEN = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".parquet",
    ".pyc",
    ".pyo",
    ".ttf",
    ".otf",
    ".ttc",
    ".woff",
    ".woff2",
}
SCHEMA = "SR_F3_CE_LOCAL_RESULTS_DATA_V1"
EXPECTED = dict(training_paths=12, evaluated_models=15, records=233340)
CHECKS = (
    "crc",
    "safe_unique_names",
    "payload_hashes",
    "manifest_hashes",
    "gzip_read_to_eof",
    "task_artifact_links",
    "independent_raw_rescoring",
)


def require(ok, message):
    if not ok:
        raise PermissionError(message)


def pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, "Duplicate JSON key: " + key)
        value[key] = item
    return value


def loads(value):
    return json.loads(
        value,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_stream(stream, destination=None):
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(CHUNK), b""):
        digest.update(chunk)
        size += len(chunk)
        if destination is not None:
            destination.write(chunk)
    return dict(sha256=digest.hexdigest(), bytes=size)


def digest_file(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)


def name(value):
    require(
        isinstance(value, str)
        and value
        and "\\" not in value
        and "\x00" not in value
        and not re.match(r"^[A-Za-z]:", value),
        "Unsafe relative path",
    )
    require(
        not PurePosixPath(value).is_absolute()
        and all(p not in ("", ".", "..") for p in value.split("/")),
        "Unsafe relative path: " + value,
    )
    return value


def allowed(value):
    name(value)
    require(
        PurePosixPath(value).suffix.lower() not in FORBIDDEN,
        "Excluded weight/binary/font artifact: " + value,
    )
    require("__pycache__" not in PurePosixPath(value).parts, "Python cache excluded")
    return value


def pin(value):
    require(
        isinstance(value, dict)
        and isinstance(value.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        and type(value.get("bytes")) is int
        and value["bytes"] >= 0,
        "Malformed file hash/size binding",
    )
    return dict(sha256=value["sha256"], bytes=value["bytes"])


def local_path(root, relative):
    relative = name(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        require(not path.is_symlink(), "Symlink source rejected: " + relative)
    require(
        path.is_file() and path.resolve(strict=True).is_relative_to(root),
        "Missing or escaped source: " + relative,
    )
    return path


def open_source(root, relative):
    path = local_path(root, relative)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    require(stat.S_ISREG(os.fstat(fd).st_mode), "Nonregular source file")
    return os.fdopen(fd, "rb")


def validate_manifest(manifest):
    require(
        manifest.get("schema_version") == 1
        and manifest.get("status") == "AUTHORIZED_RESULTS_DATA_TRANSFER",
        "Transport authorization schema/status differs",
    )
    require(manifest.get("expected_coverage") == EXPECTED, "Frozen result coverage differs")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Source inventory missing")
    for path, record in files.items():
        allowed(path)
        pin(record)
    excluded = manifest.get("server_only")
    require(isinstance(excluded, list), "Server-only index missing")
    server = {}
    for item in excluded:
        require(isinstance(item, dict), "Malformed server-only record")
        path = name(item.get("path"))
        require(
            path not in server
            and path not in files
            and isinstance(item.get("reason"), str)
            and bool(item["reason"].strip()),
            "Ambiguous server-only artifact",
        )
        require(
            type(item.get("bytes")) is int and item["bytes"] >= 0, "Malformed server-only byte size"
        )
        if item.get("sha256") is None:
            require(
                item.get("hash_status") == "size_only_not_reverified",
                "Missing server-only hash must be explicitly unverified",
            )
        else:
            pin(item)
        # Only the user's explicit binary exclusions may stay server-only.
        require(
            PurePosixPath(path).suffix.lower() in FORBIDDEN
            or "__pycache__" in PurePosixPath(path).parts
            or PurePosixPath(path).suffix == ".lock",
            "Raw/results evidence cannot be omitted: " + path,
        )
        server[path] = item
    receipts = manifest.get("task_receipts")
    require(isinstance(receipts, dict) and len(receipts) == 33, "Expected all 33 task receipts")
    counts = Counter()
    ids = set()
    for path, record in receipts.items():
        require(
            path in files and path.startswith("tasks/") and path.endswith("/COMPLETE.json"),
            "Task receipt missing from transport",
        )
        task_id = record.get("task_id")
        require(
            isinstance(task_id, str)
            and path == f"tasks/{task_id}/COMPLETE.json"
            and task_id not in ids,
            "Task identity differs",
        )
        ids.add(task_id)
        kind = record.get("kind")
        require(kind in ("train", "eval", "release", "other"), "Unknown task kind")
        require(
            (kind == "train") == task_id.startswith("train--")
            and (kind == "eval") == task_id.startswith("eval--")
            and (kind == "release") == (task_id == "release"),
            "Task kind/id differs",
        )
        counts[kind] += 1
        artifacts = record.get("artifact_paths")
        require(
            isinstance(artifacts, list) and artifacts and len(set(artifacts)) == len(artifacts),
            "Task artifact inventory missing/duplicated",
        )
        for artifact in artifacts:
            name(artifact)
            require(
                artifact in files or artifact in server, "Task artifact not classified: " + artifact
            )
            if artifact in server:
                require(
                    PurePosixPath(artifact).suffix != ".lock",
                    "Task artifact cannot be excluded as runtime lock",
                )
                pin(server[artifact])
    require(counts == dict(train=12, eval=15, release=1, other=5), "Frozen task matrix differs")
    return files, server, receipts


def verify_task_links(manifest, reader):
    files, server, receipts = validate_manifest(manifest)
    for path, item in receipts.items():
        raw = reader(path)
        require(
            dict(sha256=digest_bytes(raw), bytes=len(raw)) == pin(files[path]),
            "Task receipt hash changed",
        )
        receipt = loads(raw)
        require(
            receipt.get("status") == "COMPLETE"
            if item["kind"] != "other"
            else receipt.get("status") in ("PASS", "COMPLETE"),
            "Task not complete",
        )
        task = receipt.get("task", {})
        require(task.get("task_id") == item["task_id"], "Original task receipt identity differs")
        require(
            task.get("kind") == item["kind"]
            if item["kind"] != "other"
            else task.get("kind") in ("p0", "engine"),
            "Original task kind differs",
        )
        artifacts = receipt.get("artifacts")
        require(isinstance(artifacts, list) and artifacts, "Original artifacts missing")
        original = {}
        for artifact in artifacts:
            key = name(artifact.get("path"))
            require(key not in original, "Original artifact duplicate")
            original[key] = pin(artifact)
        require(
            set(original) == set(item["artifact_paths"]), "Original task artifact inventory omitted"
        )
        for key, binding in original.items():
            require(
                binding == pin(files.get(key, server.get(key))),
                "Original task artifact binding differs: " + key,
            )
    return dict(
        tasks=33,
        training_paths=12,
        evaluated_models=15,
        release=1,
        included_artifacts=sum(p in files for r in receipts.values() for p in r["artifact_paths"]),
        server_only_artifacts=sum(
            p in server for r in receipts.values() for p in r["artifact_paths"]
        ),
    )


def verify_rescore(result):
    require(
        isinstance(result, dict)
        and result.get("status") == "PASS"
        and result.get("records") == 233340
        and result.get("mismatches") == [],
        "Independent raw rescore incomplete or mismatched",
    )
    coverage = result.get("coverage", result)
    require(
        coverage.get("training_paths") == 12 and coverage.get("evaluated_models") == 15,
        "Independent rescore matrix differs",
    )
    return result


def run_rescore(root, manifest):
    path = Path(__file__).with_name("local_results_rescore.py")
    require(path.is_file() and not path.is_symlink(), "Independent rescorer unavailable")
    spec = importlib.util.spec_from_file_location("bound_local_results_rescore", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = verify_rescore(module.verify_results(root))
    return result, dict(path=path.name, **digest_file(path))


def verify_archive(archive):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        require(len(names) == len(set(names)), "Duplicate archive member")
        for info in package.infolist():
            allowed(info.filename)
            require(
                not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16),
                "Directory/symlink archive member",
            )
            if info.filename.startswith("evidence/"):
                allowed(info.filename[9:])
        require("PACKAGE_MANIFEST.json" in names, "Package manifest missing")
        package_manifest = loads(package.read("PACKAGE_MANIFEST.json"))
        require(package_manifest.get("schema") == SCHEMA, "Package schema differs")
        payloads = package_manifest.get("payloads", {})
        require(
            isinstance(payloads, dict) and set(names) == set(payloads) | {"PACKAGE_MANIFEST.json"},
            "Archive inventory differs",
        )
        for path, expected in payloads.items():
            with package.open(path) as stream:
                require(digest_stream(stream) == pin(expected), "Archive payload differs: " + path)
            if path.endswith(".gz"):
                with package.open(path) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
                    for _ in iter(lambda: stream.read(CHUNK), b""):
                        pass
        require(package.testzip() is None, "ZIP CRC failed")
        manifest = loads(package.read("TRANSPORT_MANIFEST.json"))
        files, _, _ = validate_manifest(manifest)
        require(
            {path[9:] for path in names if path.startswith("evidence/")} == set(files),
            "Source transport/archive inventory differs",
        )
        require(
            all(
                pin(payloads["evidence/" + path]) == pin(binding) for path, binding in files.items()
            ),
            "Source transport/archive hash differs",
        )
        links = verify_task_links(manifest, lambda path: package.read("evidence/" + path))
        rescore = verify_rescore(loads(package.read("RAW_RESCORE.json")))
        require(
            loads(package.read("SERVER_ONLY.json")) == manifest["server_only"],
            "Server-only index differs",
        )
        require(
            package_manifest.get("transport_manifest_sha256")
            == digest_bytes(package.read("TRANSPORT_MANIFEST.json")),
            "Transport manifest binding differs",
        )
        require(
            package_manifest.get("rescore_sha256")
            == digest_bytes(package.read("RAW_RESCORE.json")),
            "Raw rescore binding differs",
        )
    with archive.open("rb") as stream:
        archive_pin = digest_stream(stream)
    return dict(
        schema=SCHEMA,
        status="PASS",
        archive=archive_pin,
        checks=dict.fromkeys(CHECKS, "PASS"),
        task_links=links,
        records=rescore["records"],
        transport_manifest_sha256=package_manifest["transport_manifest_sha256"],
        rescore_sha256=package_manifest["rescore_sha256"],
    )


def peak_memory():
    require(
        sys.platform.startswith("linux") or sys.platform == "darwin", "Unknown RSS platform unit"
    )
    multiplier = 1 if sys.platform == "darwin" else 1024
    parent = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)
    children = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * multiplier)
    return dict(
        peak_rss_bytes=parent,
        scope="current packager process lifetime peak; CLI runs as an isolated child",
        completed_children_peak_rss_bytes=children,
        completed_children_scope=(
            "RUSAGE_CHILDREN lifetime maximum individual completed child RSS; "
            "includes waited-for rescoring child; not a sum"
        ),
        largest_observed_individual_process_peak_bytes=max(parent, children),
        combined_concurrent_peak_measured=False,
        minimum_resource_proven=False,
        system_or_experiment_peak=False,
    )


def progress(stage, **details):
    print(
        json.dumps(
            dict(stage=stage, at_unix=time.time(), **details), ensure_ascii=False, sort_keys=True
        ),
        file=sys.stderr,
        flush=True,
    )


def build(root, transport_manifest, output, *, rescorer=None):
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    manifest_path = Path(transport_manifest).resolve(strict=True)
    output = Path(output).absolute()
    require(
        output.parent.is_dir() and not output.exists() and not output.is_symlink(),
        "Output exists or parent unavailable",
    )
    require(not output.is_relative_to(root), "Output must be outside source inventory")
    receipt_path = output.with_name("FINAL_LOCAL_RESULTS_VERIFIED.json")
    require(
        not receipt_path.exists() and not receipt_path.is_symlink(), "Final receipt already exists"
    )
    raw_manifest = manifest_path.read_bytes()
    manifest = loads(raw_manifest)
    files, _, _ = validate_manifest(manifest)
    actual = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        for part in directories:
            require(not (Path(directory) / part).is_symlink(), "Symlink source directory")
        for filename in filenames:
            path = Path(directory) / filename
            require(not path.is_symlink(), "Symlink source file")
            if path.resolve() == manifest_path:
                continue
            actual.add(str(path.relative_to(root)))
    require(actual == set(files), "Local source inventory missing or contains unlisted files")
    for path in files:
        local_path(root, path)
    links = verify_task_links(manifest, lambda p: local_path(root, p).read_bytes())
    progress("RESCORE_BEGIN")
    if rescorer is None:
        rescore, rescore_source = run_rescore(root, manifest)
    else:
        rescore, rescore_source = verify_rescore(rescorer(root, manifest)), {"test_fixture": True}
    progress("RESCORE_END", records=rescore["records"])
    rescore_bytes = encoded(rescore)
    limits = [
        "结果数据包，不含模型、训练检查点、权重、Parquet、字体或Python缓存。",
        "保留原始输出、逐任务完成回执和技术失败历史；未重跑训练或评价采样。",
        "逐字节核验本次下载的数据，并对已有原文独立复算；不宣称重新核验服务器权重。",
        "SERVER_ONLY.json逐项列出未下载的原任务产物及原记录哈希。",
        "本地结果验收独立于未完成的服务器整包/迁移/恢复验收，不伪造旧PASS。",
    ]
    start_here = (
        "# SR-F3-CE 结果数据交付\n\n"
        + "\n".join("- " + v for v in limits)
        + "\n\n入口：RAW_RESCORE.json、TRANSPORT_MANIFEST.json、SERVER_ONLY.json。"
        "原始路径保留于evidence/。外部FINAL_LOCAL_RESULTS_VERIFIED.json绑定本ZIP哈希。\n"
    ).encode()
    generated = {
        "TRANSPORT_MANIFEST.json": raw_manifest,
        "RAW_RESCORE.json": rescore_bytes,
        "SERVER_ONLY.json": encoded(manifest["server_only"]),
        "START_HERE.md": start_here,
        "LOCAL_DELIVERY_PROVENANCE.json": encoded(
            dict(
                schema=SCHEMA,
                scope="RESULTS_DATA_ONLY_NO_WEIGHTS_NO_EXPERIMENT_RERUN",
                limits=limits,
                task_links=links,
                rescorer_source=rescore_source,
                packager_source=dict(path=Path(__file__).name, **digest_file(__file__)),
            )
        ),
    }
    generated["tools/local_results_delivery.py"] = Path(__file__).read_bytes()
    rescore_path = Path(__file__).with_name("local_results_rescore.py")
    if rescore_path.is_file():
        generated["tools/local_results_rescore.py"] = rescore_path.read_bytes()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".local_results_", suffix=".zip", delete=False
        ) as stream:
            temporary = Path(stream.name)
        payloads = {}
        progress("ARCHIVE_BEGIN", input_files=len(files))
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as package:
            for path, expected in sorted(files.items()):
                with (
                    open_source(root, path) as source,
                    package.open("evidence/" + path, "w", force_zip64=True) as destination,
                ):
                    actual = digest_stream(source, destination)
                require(actual == pin(expected), "Input bytes changed: " + path)
                payloads["evidence/" + path] = actual
            for path, body in generated.items():
                package.writestr(path, body)
                payloads[path] = dict(sha256=digest_bytes(body), bytes=len(body))
            package.writestr(
                "PACKAGE_MANIFEST.json",
                encoded(
                    dict(
                        schema=SCHEMA,
                        payloads=payloads,
                        transport_manifest_sha256=digest_bytes(raw_manifest),
                        rescore_sha256=digest_bytes(rescore_bytes),
                    )
                ),
            )
        progress("ARCHIVE_END", archive_bytes=temporary.stat().st_size)
        progress("VERIFY_BEGIN")
        verified = verify_archive(temporary)
        progress("VERIFY_END", archive_sha256=verified["archive"]["sha256"])
        os.link(temporary, output)
        receipt = dict(
            verified,
            status="FINAL_LOCAL_RESULTS_VERIFIED",
            archive={"path": str(output), **verified["archive"]},
            elapsed_seconds=time.monotonic() - started,
            memory_profile=peak_memory(),
            completed_at_unix=time.time(),
            scientific_reruns=0,
            weights_downloaded=False,
            verification_boundary=(
                "Local transferred result data and existing-raw rescore only; "
                "no fresh server checkpoint verification"
            ),
        )
        with receipt_path.open("xb") as stream:
            stream.write(encoded(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        return receipt
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    parser.add_argument("--transport-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_archive(args.output)
    else:
        require(
            args.root and args.transport_manifest, "Source root and transport manifest required"
        )
        result = build(args.root, args.transport_manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
