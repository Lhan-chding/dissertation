"""Receipt validation is metadata verification, never a claim of live model execution."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mm_core.inventory import BRANCHES, MODEL_FAMILY, inspect_model_receipt, inventory_assets


def receipt_fixture(*, count=4, prefix="model.safetensors", status="CACHED_AND_HASHED"):
    revision = "c" * 40
    names = [
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    ]
    shards = (
        ["model.safetensors"]
        if count == 1
        else [f"{prefix}-{index:05d}-of-{count:05d}.safetensors" for index in range(1, count + 1)]
    )
    if count != 1:
        names.append("model.safetensors.index.json")
    names.extend(shards)
    return {
        "status": status,
        "model_id": MODEL_FAMILY,
        "model_path": f"/registered/cache/models--Qwen--Qwen3.5-9B/snapshots/{revision}",
        "revision": revision,
        "files": [
            {"name": name, "bytes": len(name), "sha256": hashlib.sha256(name.encode()).hexdigest()}
            for name in names
        ],
        "weight_map": {f"tensor.{i}": name for i, name in enumerate(shards)},
    }


def inspect(tmp_path: Path, receipt):
    path = tmp_path / "MODEL_RECEIPT.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return inspect_model_receipt(path)


@pytest.mark.parametrize("count,prefix", [(1, "model"), (2, "model"), (4, "model.safetensors")])
def test_complete_native_and_conventional_weight_layouts(tmp_path, count, prefix):
    receipt = receipt_fixture(count=count, prefix=prefix)
    original = copy.deepcopy(receipt)
    result = inspect(tmp_path, receipt)
    assert result["model_id"] == "Qwen/Qwen3.5-9B"
    assert result["shard_count"] == count
    assert result["weight_files"] == sorted(set(receipt["weight_map"].values()))
    assert result["weight_index_mapping_checked"] is True
    assert result["status"] == "server_receipt_verified"
    assert receipt == original


@pytest.mark.parametrize(
    "status,downloaded", [("CACHED_AND_HASHED", False), ("DOWNLOADED_AND_HASHED", True)]
)
def test_cached_rehash_does_not_claim_download(tmp_path, status, downloaded):
    result = inspect(tmp_path, receipt_fixture(status=status))
    assert result["source_receipt_status"] == status
    assert result["new_base_asset"] is downloaded
    assert result["downloaded_in_this_receipt"] is downloaded
    assert result["verification_level"].startswith(
        "server_download_receipt" if downloaded else "server_cache_rehash_receipt"
    )
    assert "no local weight copy" in result["verification_level"]


@pytest.mark.parametrize("status", ["CACHED", "CACHE_PRESENT", "UNKNOWN", None, {}])
def test_presence_or_unknown_status_does_not_prove_rehash(tmp_path, status):
    with pytest.raises(ValueError, match="required verified model family"):
        inspect(tmp_path, receipt_fixture(status=status))


@pytest.mark.parametrize(
    "change", ["old_path", "old_id", "relative", "traversal", "fake_ancestor", "moving_revision"]
)
def test_other_model_or_unsafe_snapshot_identity_rejected(tmp_path, change):
    receipt = receipt_fixture()
    if change == "old_path":
        receipt["model_path"] = receipt["model_path"].replace(
            "Qwen3.5-9B", "Qwen2.5-VL-3B-Instruct"
        )
    elif change == "old_id":
        receipt["model_id"] = "Qwen/Qwen2.5-VL-3B-Instruct"
    elif change == "relative":
        receipt["model_path"] = receipt["model_path"].lstrip("/")
    elif change == "traversal":
        receipt["model_path"] = "/other/../" + receipt["model_path"].lstrip("/")
    elif change == "fake_ancestor":
        receipt["model_path"] = receipt["model_path"].replace("/snapshots/", "/other/snapshots/")
    else:
        receipt["model_path"] = receipt["model_path"].replace(receipt["revision"], "main")
        receipt["revision"] = "main"
    with pytest.raises(ValueError, match="required verified model family"):
        inspect(tmp_path, receipt)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_middle",
        "missing_index",
        "mixed_total",
        "duplicate_index",
        "mixed_prefix",
        "unexpected_weight",
    ],
)
def test_partial_or_inconsistent_shard_lists_are_rejected(tmp_path, failure):
    receipt = receipt_fixture()
    del receipt["weight_map"]
    weights = [item for item in receipt["files"] if item["name"].endswith(".safetensors")]
    if failure == "missing_middle":
        receipt["files"].remove(weights[1])
    elif failure == "missing_index":
        receipt["files"] = [
            row for row in receipt["files"] if row["name"] != "model.safetensors.index.json"
        ]
    elif failure == "mixed_total":
        weights[0]["name"] = weights[0]["name"].replace("of-00004", "of-00005")
    elif failure == "duplicate_index":
        weights[1]["name"] = "model.safetensors-1-of-4.safetensors"
    elif failure == "mixed_prefix":
        weights[0]["name"] = weights[0]["name"].replace("model.safetensors-", "model-")
    else:
        weights[0]["name"] = "adapter_model.safetensors"
    with pytest.raises(ValueError, match="safetensors"):
        inspect(tmp_path, receipt)


@pytest.mark.parametrize(
    "bad_map", [None, {}, [], {"tensor.0": "missing.safetensors"}, {"tensor.0": 5}]
)
def test_weight_index_must_match_complete_shard_enumeration(tmp_path, bad_map):
    receipt = receipt_fixture()
    receipt["weight_map"] = bad_map
    with pytest.raises(ValueError, match="weight index"):
        inspect(tmp_path, receipt)


def test_weight_index_missing_one_listed_shard_is_rejected(tmp_path):
    receipt = receipt_fixture()
    del receipt["weight_map"]["tensor.3"]
    with pytest.raises(ValueError, match="weight index"):
        inspect(tmp_path, receipt)


def test_weight_index_omission_remains_explicit(tmp_path):
    receipt = receipt_fixture()
    del receipt["weight_map"]
    assert inspect(tmp_path, receipt)["weight_index_mapping_checked"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "../config.json"),
        ("name", ".."),
        ("bytes", True),
        ("bytes", 0),
        ("sha256", "unknown"),
    ],
)
def test_invalid_file_records_are_rejected(tmp_path, field, value):
    receipt = receipt_fixture()
    receipt["files"][0][field] = value
    with pytest.raises(ValueError, match="malformed or duplicate"):
        inspect(tmp_path, receipt)


def test_duplicate_file_names_are_rejected(tmp_path):
    receipt = receipt_fixture()
    receipt["files"].append(dict(receipt["files"][0]))
    with pytest.raises(ValueError, match="malformed or duplicate"):
        inspect(tmp_path, receipt)


@pytest.mark.parametrize("receipt", [[], {"status": "CACHED_AND_HASHED"}])
def test_invalid_top_level_receipt_is_rejected(tmp_path, receipt):
    with pytest.raises(ValueError):
        inspect(tmp_path, receipt)


def test_processor_hash_covers_native_chat_and_video_configuration(tmp_path):
    receipt = receipt_fixture()
    original = inspect(tmp_path, receipt)
    for filename in ["chat_template.jinja", "video_preprocessor_config.json"]:
        changed = copy.deepcopy(receipt)
        next(row for row in changed["files"] if row["name"] == filename)["sha256"] = "f" * 64
        result = inspect(tmp_path, changed)
        assert result["processor_sha256"] != original["processor_sha256"]
        assert result["weights_sha256"] == original["weights_sha256"]


def test_receipt_order_and_extra_file_notes_do_not_change_identity_hashes(tmp_path):
    receipt = receipt_fixture()
    original = inspect(tmp_path, receipt)
    receipt["files"].reverse()
    for item in receipt["files"]:
        item["note"] = "metadata only"
    updated = inspect(tmp_path, receipt)
    assert updated["weights_sha256"] == original["weights_sha256"]
    assert updated["processor_sha256"] == original["processor_sha256"]


def test_missing_current_receipt_keeps_model_unknown_and_historical_branches(tmp_path):
    before = copy.deepcopy(BRANCHES)
    repo = tmp_path / "repo"
    repo.mkdir()
    result = inventory_assets(repo, tmp_path / "inventory")
    assert result["model_family"] == "Qwen/Qwen3.5-9B"
    assert result["model_snapshot"]["status"] == "not_verified"
    assert result["model_snapshot"]["path"] is None
    assert [row["branch"] for row in result["contracts"]] == [row["branch"] for row in before]
    assert result["model_calls_invoked"] is False
    assert before == BRANCHES
