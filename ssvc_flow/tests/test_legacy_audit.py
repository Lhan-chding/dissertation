"""Read-only P0 evidence and exact-reproduction boundaries."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from src.audit_legacy import audit_legacy


def _put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _archive(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for name, content in members:
            body = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(body)
            stream.addfile(member, io.BytesIO(body))
    return path


def test_empty_audit_is_explicitly_incomplete_and_writes_phase_files(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    out = tmp_path / "P0"
    result = audit_legacy(root=root, out=out)
    assert result["legacy_exact_reproduction"] is False
    assert "checkpoint_binaries" in result["missing"]
    assert "group_std_convention" in result["missing"]
    assert result["gpu_invoked"] is False
    assert result["inventory"] == []
    for filename in ("status.json", "report.md", "manifest.json", "legacy_vs_new_protocol.md", "audit.json"):
        assert (out / filename).is_file()
    assert "missing" in (out / "legacy_vs_new_protocol.md").read_text()


def test_yaml_values_have_hash_and_field_provenance_without_inferred_batches(tmp_path):
    root = tmp_path / "legacy"
    source = _put(root, "configs/v5/study_c3_resolution_validity.yaml", "model: Qwen2.5-VL-3B-Instruct\ntraining_seeds: [2026082501]\ntraining:\n  learning_rate: 0.000001\n  per_device_train_batch_size: 1\n  gradient_accumulation_steps: 8\n  group_size: 8\n")
    before = source.read_bytes()
    result = audit_legacy(root=root, out=tmp_path / "P0")
    lr = result["facts"]["learning_rate"]
    assert lr["value"] == 1e-6
    assert lr["evidence"][0]["field"] == "training.learning_rate"
    assert lr["evidence"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["facts"]["training_seeds"]["value"] == [2026082501]
    assert "prompts_per_optimizer_update" in result["missing"]
    assert source.read_bytes() == before


def test_archive_counts_actual_rows_and_pairs_and_keeps_archive_readonly(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    rows = [
        {"scene_id": "t", "pair_id": "t", "split": "train", "condition": "collision", "prompt": "train"},
        {"scene_id": "a", "pair_id": "p", "split": "test", "condition": "collision", "prompt": "A"},
        {"scene_id": "b", "pair_id": "p", "split": "test", "condition": "separating", "prompt": "B"},
        {"scene_id": "c", "pair_id": "q", "split": "dev", "condition": "collision", "prompt": "C"},
    ]
    archive = _archive(tmp_path / "evidence.tar.gz", [("artifacts/v5/study_c2/data/reward_fibers.jsonl", "\n".join(map(json.dumps, rows)))])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = audit_legacy(root=root, legacy_archive=archive, out=tmp_path / "P0")
    counts = result["data_inventory"][0]
    assert counts["row_count"] == 4
    assert counts["evaluation_scene_count"] == 3
    assert counts["evaluation_pair_count"] == 2
    assert counts["all_pairs_have_two_matched_scenes"] is False
    assert counts["split_counts"] == {"dev": 1, "test": 2, "train": 1}
    assert result["legacy_exact_reproduction"] is False
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    assert result["archives"][0]["sha256"] == digest


def test_static_source_is_not_imported_and_rewards_require_literal_evidence(tmp_path):
    root = tmp_path / "legacy"
    marker = tmp_path / "executed"
    _put(root, "src/compensability/study_c3/semantic_action_parser.py", f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef parse_p1(text):\n    return None\n")
    _put(root, "tests/study_c3/test_reward_argmax_preservation.py", 'EXPECTED = {"A_BIN": [1, 1, 0, 0], "X_BIN": [1, 0, 0, 0], "A_LEX": [3, 3, 1, 0], "X_LEX": [3, 1, 1, 0]}\n')
    result = audit_legacy(root=root, out=tmp_path / "P0")
    assert not marker.exists()
    assert result["facts"]["reward_vectors"]["value"]["X_LEX"] == [3, 1, 1, 0]
    assert result["facts"]["parser_source"]["value"] == ["parse_p1"]
    assert "parser_runtime_qualification" in result["missing"]


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json"])
def test_unsafe_archive_paths_rejected_without_extraction(tmp_path, name):
    archive = _archive(tmp_path / "bad.tar.gz", [(name, "{}")])
    with pytest.raises(ValueError, match="unsafe"):
        audit_legacy(root=tmp_path, legacy_archive=archive, out=tmp_path / "P0")
    assert not (tmp_path.parent / "escape.json").exists()


def test_symlinks_are_not_followed_and_protected_output_is_rejected(tmp_path):
    root = tmp_path / "legacy"
    outside = tmp_path / "outside"
    _put(outside, "secret.json", '{"secret": "do-not-read"}')
    (root / "src/compensability").mkdir(parents=True)
    (root / "src/compensability/study_c3").symlink_to(outside, target_is_directory=True)
    result = audit_legacy(root=root, out=tmp_path / "P0")
    assert result["inventory"] == []
    with pytest.raises(ValueError, match="read.only|protected"):
        audit_legacy(root=root, out=root / "src/compensability/study_c3/P0")


def test_cli_dry_run_reports_zero_gpu_budget_without_writing(tmp_path):
    out = tmp_path / "P0"
    result = subprocess.run([sys.executable, "-m", "src.audit_legacy", "--root", str(tmp_path), "--out", str(out), "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    budget = json.loads(result.stdout)
    assert budget["forward_calls"] == budget["backward_calls"] == budget["rollout_count"] == 0
    assert not out.exists()
