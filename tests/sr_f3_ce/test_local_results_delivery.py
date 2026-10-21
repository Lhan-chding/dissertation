"""Data-only delivery preserves raw evidence, rejects tensors and checks every archive byte."""

import gzip
import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/local_results_delivery.py"
spec = importlib.util.spec_from_file_location("local_results_delivery_test", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.encoded(value) if not isinstance(value, bytes) else value)


def record(path):
    with path.open("rb") as stream:
        return m.digest_stream(stream)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    raw_path = "release/ENDPOINT_RAW.jsonl.gz"
    write(root / raw_path, gzip.compress(b'{"raw_text":"original invalid answer retained"}\n'))
    receipts, server = {}, []
    task_ids = (
        [f"train--r{i}" for i in range(12)]
        + [f"eval--e{i}" for i in range(15)]
        + ["release", "p0", "engine--b0", "engine--b1", "engine--b2", "engine--resume"]
    )
    for task_id in task_ids:
        kind = (
            "train"
            if task_id.startswith("train--")
            else "eval"
            if task_id.startswith("eval--")
            else "release"
            if task_id == "release"
            else "other"
        )
        artifact_path = f"tasks/{task_id}/EVIDENCE.json"
        write(root / artifact_path, {"task_id": task_id, "unchanged": True})
        artifacts = [
            {"path": artifact_path, **record(root / artifact_path)},
            {"path": raw_path, **record(root / raw_path)},
        ]
        if kind == "train":
            weights = {
                "path": f"runs/{task_id}/checkpoint.pt",
                "sha256": "a" * 64,
                "bytes": 20000000000,
                "reason": "User excludes weights; retained server-only",
            }
            server.append(weights)
            artifacts.append({k: v for k, v in weights.items() if k != "reason"})
        receipt_path = f"tasks/{task_id}/COMPLETE.json"
        write(
            root / receipt_path,
            {
                "status": "PASS" if kind == "other" else "COMPLETE",
                "task": {
                    "task_id": task_id,
                    "kind": ("p0" if task_id == "p0" else "engine") if kind == "other" else kind,
                },
                "artifacts": artifacts,
            },
        )
        receipts[receipt_path] = {
            "kind": kind,
            "task_id": task_id,
            "artifact_paths": [item["path"] for item in artifacts],
        }
    manifest = dict(
        schema_version=1,
        status="AUTHORIZED_RESULTS_DATA_TRANSFER",
        expected_coverage=m.EXPECTED,
        files={str(p.relative_to(root)): record(p) for p in root.rglob("*") if p.is_file()},
        server_only=server,
        task_receipts=receipts,
    )
    manifest_path = tmp_path / "TRANSPORT.json"
    write(manifest_path, manifest)
    output = tmp_path / "results.zip"

    def rescorer(root, manifest):
        return dict(
            status="PASS",
            records=233340,
            mismatches=[],
            coverage={"training_paths": 12, "evaluated_models": 15},
            fixture_only=True,
        )

    return SimpleNamespace(
        root=root,
        manifest=manifest,
        manifest_path=manifest_path,
        output=output,
        rescorer=rescorer,
        raw_path=raw_path,
    )


def build(data):
    write(data.manifest_path, data.manifest)
    return m.build(data.root, data.manifest_path, data.output, rescorer=data.rescorer)


def test_streamed_results_delivery_and_external_receipt(data):
    receipt = build(data)
    assert receipt["status"] == "FINAL_LOCAL_RESULTS_VERIFIED"
    assert receipt["records"] == 233340
    assert receipt["weights_downloaded"] is False
    assert receipt["memory_profile"]["minimum_resource_proven"] is False
    assert set(receipt["checks"].values()) == {"PASS"}
    assert receipt["archive"]["sha256"] == record(data.output)["sha256"]
    assert m.verify_archive(data.output)["status"] == "PASS"
    with zipfile.ZipFile(data.output) as package:
        assert not any(Path(name).suffix in m.FORBIDDEN for name in package.namelist())
        assert package.read("evidence/" + data.raw_path) == (data.root / data.raw_path).read_bytes()
        assert m.loads(package.read("SERVER_ONLY.json")) == data.manifest["server_only"]
    assert not list(data.output.parent.glob(".local_results_*"))
    with pytest.raises(PermissionError, match="Output exists"):
        build(data)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        "/absolute.json",
        "a/../b.json",
        "a\\b.json",
        "a//b.json",
        "C:/x.json",
        "a/./b.json",
    ],
)
def test_unsafe_manifest_path_rejected(data, path):
    data.manifest["files"][path] = {"sha256": "0" * 64, "bytes": 1}
    with pytest.raises(PermissionError, match="Unsafe"):
        build(data)


@pytest.mark.parametrize("suffix", ["pt", "pth", "bin", "safetensors", "parquet", "ttf", "pyc"])
def test_forbidden_payload_never_read_or_archived(data, suffix):
    path = "forbidden." + suffix
    data.manifest["files"][path] = {"sha256": "0" * 64, "bytes": 1000}
    with pytest.raises(PermissionError, match="Excluded"):
        build(data)
    assert not data.output.exists()


def test_changed_raw_fails_hash_and_removes_partial_zip(data):
    (data.root / data.raw_path).write_bytes(gzip.compress(b"changed\n"))
    with pytest.raises(PermissionError, match="Input bytes changed"):
        build(data)
    assert not data.output.exists()
    assert not list(data.output.parent.glob(".local_results_*"))


def test_missing_raw_not_silently_dropped(data):
    (data.root / data.raw_path).unlink()
    with pytest.raises(PermissionError, match="inventory missing"):
        build(data)


def test_raw_cannot_be_reclassified_as_server_only(data):
    binding = data.manifest["files"].pop(data.raw_path)
    data.manifest["server_only"].append({"path": data.raw_path, **binding, "reason": "save space"})
    with pytest.raises(PermissionError, match="cannot be omitted"):
        build(data)


def test_artifact_list_cannot_omit_original_record(data):
    data.manifest["task_receipts"]["tasks/release/COMPLETE.json"]["artifact_paths"].pop()
    with pytest.raises(PermissionError, match="artifact inventory omitted"):
        build(data)


def test_original_artifact_hash_must_match_transport(data):
    data.manifest["files"][data.raw_path]["sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="binding differs"):
        build(data)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_symlink_sources_rejected(data, kind):
    if kind == "file":
        target = data.root / data.raw_path
        external = data.root.parent / "external.gz"
        target.rename(external)
        target.symlink_to(external)
    else:
        target = data.root / "release"
        external = data.root.parent / "outside"
        target.rename(external)
        target.symlink_to(external, target_is_directory=True)
    with pytest.raises(PermissionError, match="Symlink"):
        build(data)


def test_unknown_unlisted_input_rejected(data):
    write(data.root / "unlisted.json", {"unexpected": True})
    with pytest.raises(PermissionError, match="unlisted"):
        build(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAIL"),
        ("records", 233339),
        ("mismatches", ["wrong raw"]),
        ("coverage", {"training_paths": 11, "evaluated_models": 15}),
    ],
)
def test_rescore_must_cover_full_frozen_matrix(data, field, value):
    original = data.rescorer
    data.rescorer = lambda root, manifest: {**original(root, manifest), field: value}
    with pytest.raises(PermissionError, match="rescore"):
        build(data)
    assert not data.output.exists()


def test_invalid_gzip_even_with_transport_hash_rejected(data):
    path = data.root / data.raw_path
    original = path.read_bytes()
    path.write_bytes(original[:-5])
    binding = record(path)
    data.manifest["files"][data.raw_path] = binding
    for receipt_path in data.manifest["task_receipts"]:
        receipt = m.loads((data.root / receipt_path).read_bytes())
        for artifact in receipt["artifacts"]:
            if artifact["path"] == data.raw_path:
                artifact.update(binding)
        write(data.root / receipt_path, receipt)
        data.manifest["files"][receipt_path] = record(data.root / receipt_path)
    with pytest.raises((EOFError, OSError)):
        build(data)
    assert not data.output.exists()


def test_archive_content_tamper_rejected(data):
    build(data)
    with zipfile.ZipFile(data.output) as package:
        entries = {n: package.read(n) for n in package.namelist()}
    entries["START_HERE.md"] += b"tamper"
    data.output.unlink()
    with zipfile.ZipFile(data.output, "w") as package:
        for name, raw in entries.items():
            package.writestr(name, raw)
    with pytest.raises(PermissionError, match="payload differs"):
        m.verify_archive(data.output)


@pytest.mark.parametrize("platform,expected", [("darwin", 1234), ("linux", 1234 * 1024)])
def test_peak_rss_units_and_scope(monkeypatch, platform, expected):
    monkeypatch.setattr(m.sys, "platform", platform)
    monkeypatch.setattr(m.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1234))
    assert m.peak_memory()["peak_rss_bytes"] == expected
    assert m.peak_memory()["minimum_resource_proven"] is False


def test_unverified_unbound_server_only_weights_are_honestly_labeled(data):
    data.manifest["server_only"].append(
        {
            "path": "models/base.safetensors",
            "sha256": None,
            "bytes": 999,
            "hash_status": "size_only_not_reverified",
            "reason": "Not downloaded",
        }
    )
    data.manifest["server_only"].append(
        {
            "path": "operations/runtime.lock",
            "sha256": None,
            "bytes": 0,
            "hash_status": "size_only_not_reverified",
            "reason": "Runtime lock",
        }
    )
    assert build(data)["status"] == "FINAL_LOCAL_RESULTS_VERIFIED"


def test_task_bound_weight_cannot_lose_original_hash(data):
    data.manifest["server_only"][0]["sha256"] = None
    data.manifest["server_only"][0]["hash_status"] = "size_only_not_reverified"
    with pytest.raises(PermissionError, match="hash/size"):
        build(data)


def test_archive_rejects_tensor_outside_evidence_namespace(data):
    build(data)
    with zipfile.ZipFile(data.output, "a") as package:
        package.writestr("hidden/weights.pt", b"not allowed")
    with pytest.raises(PermissionError, match="Excluded"):
        m.verify_archive(data.output)


@pytest.mark.parametrize("platform,multiplier", [("darwin", 1), ("linux", 1024)])
def test_parent_and_completed_child_peaks_are_separate(monkeypatch, platform, multiplier):
    monkeypatch.setattr(m.sys, "platform", platform)
    monkeypatch.setattr(
        m.resource,
        "getrusage",
        lambda kind: SimpleNamespace(ru_maxrss=100 if kind == m.resource.RUSAGE_SELF else 250),
    )
    profile = m.peak_memory()
    assert profile["peak_rss_bytes"] == 100 * multiplier
    assert profile["completed_children_peak_rss_bytes"] == 250 * multiplier
    assert profile["largest_observed_individual_process_peak_bytes"] == 250 * multiplier
    assert profile["combined_concurrent_peak_measured"] is False
    assert profile["minimum_resource_proven"] is False


def test_progress_is_bounded_stages_on_stderr(data, capsys):
    build(data)
    captured = capsys.readouterr()
    assert captured.out == ""
    events = [m.loads(line) for line in captured.err.splitlines()]
    assert [row["stage"] for row in events] == [
        "RESCORE_BEGIN",
        "RESCORE_END",
        "ARCHIVE_BEGIN",
        "ARCHIVE_END",
        "VERIFY_BEGIN",
        "VERIFY_END",
    ]
    assert all(isinstance(row["at_unix"], float) for row in events)
