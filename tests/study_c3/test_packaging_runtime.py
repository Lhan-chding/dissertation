from __future__ import annotations

import tarfile
from pathlib import Path

from compensability.study_c3.io import sha256_file
from compensability.study_c3.packaging_runtime import (
    _evidence_sources,
    create_deterministic_evidence_archive,
)


def test_evidence_archive_is_sorted_deterministic_and_contains_raw_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "nested/raw_rows.jsonl").write_text('{"row": 1}\n', encoding="utf-8")
    (root / "summary.json").write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    sources = (Path("nested/raw_rows.jsonl"), Path("summary.json"))

    create_deterministic_evidence_archive(root=root, sources=sources, output=first)
    create_deterministic_evidence_archive(root=root, sources=reversed(sources), output=second)

    assert sha256_file(first) == sha256_file(second)
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == ["nested/raw_rows.jsonl", "summary.json"]
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_evidence_inventory_keeps_raw_rows_but_excludes_adapter_weights(tmp_path: Path) -> None:
    (tmp_path / "configs/v5").mkdir(parents=True)
    (tmp_path / "configs/v5/study_c3_resolution_validity.yaml").write_text("x: 1\n")
    (tmp_path / "configs/v5/server_package_lock.yaml").write_text("x: 1\n")
    training = tmp_path / "artifacts/v5/study_c3/training/A_BIN"
    (training / "final_adapter").mkdir(parents=True)
    (training / "final_adapter/adapter_model.safetensors").write_bytes(b"weights")
    (training / "checkpoint-48").mkdir()
    (training / "checkpoint-48/adapter_model.safetensors").write_bytes(b"weights")
    (training / "raw_reward_trace.jsonl").write_text('{"reward": 1}\n')
    report = tmp_path / "artifacts/v5/study_c3/report"
    report.mkdir(parents=True)
    (report / "old.tar.gz").write_bytes(b"archive")

    sources = _evidence_sources(tmp_path)

    assert Path("artifacts/v5/study_c3/training/A_BIN/raw_reward_trace.jsonl") in sources
    assert all("final_adapter" not in path.parts for path in sources)
    assert all(not any(part.startswith("checkpoint-") for part in path.parts) for path in sources)
    assert all(path.name != "old.tar.gz" for path in sources)
