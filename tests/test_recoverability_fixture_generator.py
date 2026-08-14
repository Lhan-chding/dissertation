from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.fixture_generator import (
    audit_fixture,
    fixture_sha256,
    generate_fixture_50,
    serialize_fixture,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fifty_scene_fixture_passes_every_pre_model_design_audit() -> None:
    scenes = generate_fixture_50(seed=2026081606)
    report = audit_fixture(scenes)

    assert len(scenes) == 50
    assert report.records == 50
    assert dict(report.family_counts) == {
        "cross_series": 17,
        "duplicate_encoding": 17,
        "trend": 16,
    }
    assert min(dict(report.operation_counts).values()) >= 16
    assert report.exactly_recoverable_valid == 50
    assert report.nonrecoverable_ablated == 50
    assert report.legal_counterfactuals == 50
    assert report.valid_sham_surface_matched == 50
    assert report.gold_free_stage2_payloads == 50
    assert report.unique_numeric_tables == 50
    assert report.audit_passed is True


def test_fixture_is_order_independent_seeded_and_byte_canonical() -> None:
    first = generate_fixture_50(seed=2026081606)
    second = generate_fixture_50(seed=2026081606)

    assert first == second
    assert serialize_fixture(first) == serialize_fixture(second)
    assert len(fixture_sha256(first)) == 64
    assert fixture_sha256(first) == fixture_sha256(tuple(reversed(tuple(reversed(first)))))
    assert generate_fixture_50(seed=2026081607) != first


def test_fixture_canonical_records_separate_hidden_audit_from_stage2_payload() -> None:
    record = json.loads(serialize_fixture(generate_fixture_50(seed=2026081606))[0])

    assert set(record) == {
        "audit",
        "counterfactual",
        "family",
        "operation",
        "scene_id",
        "stage2_payload",
    }
    assert "gold_answer" in record["audit"]
    stage2 = record["stage2_payload"]
    assert "gold_answer" not in json.dumps(stage2, sort_keys=True)
    assert "gold_scene" not in json.dumps(stage2, sort_keys=True)
    assert stage2["image_available"] is False
    assert stage2["evidence"]["max_mismatches"] == 1


def test_local_fixture_cli_refuses_overwrite_and_never_imports_model(tmp_path: Path) -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "01_generate_local_fixture.py"
    output = tmp_path / "fixture"
    command = [sys.executable, str(script), "--output-directory", str(output)]

    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["audit_passed"] is True
    assert payload["records"] == 50
    assert payload["gpu_invoked"] is False
    assert payload["model_loaded"] is False
    assert payload["training_invoked"] is False
    assert (output / "fixture_50.jsonl").is_file()
    assert (output / "fixture_50.audit.json").is_file()
    refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert refused.returncode != 0
    source = script.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from_pretrained" not in source


def test_fixture_generator_rejects_unregistered_size_or_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        generate_fixture_50(seed=True)
