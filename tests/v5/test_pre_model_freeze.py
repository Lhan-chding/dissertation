"""Phase 2a must freeze model-independent inputs before any Qwen call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from compensability_v5.data.pre_model_freeze import freeze_pre_model_factorial


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase2a_freeze_is_deterministic_hash_bound_and_model_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = freeze_pre_model_factorial(first, seed=20260821, canonical_per_family=2)
    second_manifest = freeze_pre_model_factorial(second, seed=20260821, canonical_per_family=2)

    assert first_manifest["status"] == "PHASE_2A_PRE_MODEL_FROZEN"
    assert first_manifest["model_calls"] == 0
    assert first_manifest["observation_capture_required"] is True
    assert first_manifest["value_domain"] == [2, 18]
    assert first_manifest["image_size"] == [280, 280]
    assert first_manifest["rows_sha256"] == second_manifest["rows_sha256"]

    rows = _load_jsonl(first / "pre_model_rows.jsonl")
    assert len(rows) == 30  # 3 families * 2 parents * 5 graph axes
    assert {row["family"] for row in rows} == {"known_value", "pair_sum", "trend"}
    assert {row["graph_axis"] for row in rows} == {
        "familiar",
        "variable_permuted",
        "fact_order_permuted",
        "equivalent_basis",
        "sparse_mixed_ood",
    }
    assert all(row["observation_status"] == "pending_server_capture" for row in rows)
    assert all("natural_observation" not in row for row in rows)

    for row in rows:
        image = first / str(row["image_path"])
        prompt = first / str(row["prompt_path"])
        assert _sha256(image) == row["image_sha256"]
        assert _sha256(prompt) == row["prompt_sha256"]
        with Image.open(image) as opened:
            assert opened.size == (280, 280)


def test_phase2a_freeze_refuses_to_overwrite_immutable_parent(tmp_path: Path) -> None:
    output = tmp_path / "frozen"
    freeze_pre_model_factorial(output, seed=11, canonical_per_family=1)

    try:
        freeze_pre_model_factorial(output, seed=11, canonical_per_family=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Phase 2a freeze must not overwrite an existing parent")
