import json

import pytest

from sr_f3_ce.protocol import contained, stable_seed, verify_source


def test_registered_paths_reject_escape_and_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("unrelated")
    root = tmp_path / "run"
    root.mkdir()
    (root / "link").symlink_to(outside)
    for path in ("../outside", str(outside), "link"):
        with pytest.raises(PermissionError):
            contained(root, path)


def test_source_manifest_detects_changed_bytes(tmp_path):
    from sr_f3_ce.protocol import file_hash

    source = tmp_path / "trainer.py"
    source.write_text("fixed")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "trainer.py", "bytes": 5, "sha256": file_hash(source)}]})
    )
    assert verify_source(manifest, tmp_path)
    source.write_text("other")
    with pytest.raises(PermissionError):
        verify_source(manifest, tmp_path)


def test_stream_seeds_are_separate_and_reproducible():
    args = ("train", 73101, 1, "qid", 0)
    assert stable_seed(*args) == stable_seed(*args)
    assert stable_seed(*args) != stable_seed("monitor", *args[1:])
