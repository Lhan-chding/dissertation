"""The storage benchmark must check bytes and exercise real storage, not models."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[2] / "scripts/sr_f6_memory_repair_ops/benchmark_storage.py"
    spec = importlib.util.spec_from_file_location("storage_benchmark", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_comparison_verifies_bytes_and_real_sync_counts(tmp_path, monkeypatch):
    benchmark = module()
    monkeypatch.setattr(benchmark, "PROJECT", tmp_path)
    monkeypatch.setattr(benchmark, "SIZES", (2004, 4096, 16384))
    benchmark.run(tmp_path / "comparison")
    report = json.loads((tmp_path / "comparison/RESULT.json").read_text())
    assert report["records"] == 3
    assert report["model_forwards"] == report["optimizer_updates"] == 0
    assert report["read_passes"] == 4
    old, new = report["results"]
    assert old["counters"]["fsync_calls"] == 3
    assert new["counters"]["fsync_calls"] == 1
    assert old["storage"]["disk_bytes_written"] == new["storage"]["disk_bytes_written"]
    assert old["storage"]["disk_bytes_read"] == new["storage"]["disk_bytes_read"]
    assert new["storage"]["disk_bytes_read"] == 4 * report["write_bytes"]
    assert not list((tmp_path / "comparison/arena_v3").iterdir())
    with pytest.raises(FileExistsError):
        benchmark.run(tmp_path / "comparison")


def test_rejects_symlink_or_outside_root(tmp_path, monkeypatch):
    benchmark = module()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(benchmark, "PROJECT", project)
    with pytest.raises(ValueError):
        benchmark.run(tmp_path / "outside")
    (project / "alias").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        benchmark.run(project / "alias/result")
