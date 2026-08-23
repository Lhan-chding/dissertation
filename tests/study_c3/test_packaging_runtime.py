from __future__ import annotations

import tarfile
from pathlib import Path

from compensability.study_c3.io import sha256_file
from compensability.study_c3.packaging_runtime import create_deterministic_evidence_archive


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
