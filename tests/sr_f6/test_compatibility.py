"""Interface-v2 adapters preserve originals and cannot sign scientific admission."""

from __future__ import annotations

import copy
import json

import pytest

from sr_f6 import compatibility
from sr_f6.common import digest, file_hash
from sr_f6.source_binding import verify_bound_files


def inventory(qid="q0", **changes):
    row = dict(
        qid=qid,
        path="/project/a.png",
        resolved_path="/project/a.png",
        sha256="a" * 64,
        bytes=12,
        stat_cache_key=dict(size=12),
        symlink=False,
    )
    row.update(changes)
    return row


def visual(**changes):
    row = dict(
        qid="q0",
        status="PASS",
        observation="SYNTHETIC recorded observation, not an actual visual review",
        source_sha256="a" * 64,
        processed_file_sha256="b" * 64,
        source_viewed=True,
        processed_viewed=True,
        defects=[],
    )
    row.update(changes)
    return row


def test_repeated_assets_preserve_each_question_and_original_order_without_mutation():
    rows = [inventory(), inventory("q1")]
    original = copy.deepcopy(rows)
    result = compatibility.coalesce_asset_inventory(rows)
    assert rows == original
    assert result["question_count"] == 2 and result["unique_file_count"] == 1
    assert result["input_object_sha256"] == digest(rows)
    assert result["question_references"] == [
        dict(qid=qid, file_index=0, path="/project/a.png") for qid in ("q0", "q1")
    ]
    assert result["verification_required"] and not result["gate_signed"]
    result["unique_files"][0]["stat_cache_key"]["size"] = 99
    assert rows == original


@pytest.mark.parametrize(
    "changes",
    [
        dict(sha256="b" * 64),
        dict(resolved_path="/project/b.png"),
        dict(stat_cache_key=dict(size=13)),
        dict(bytes=13),
        dict(symlink=True),
        dict(extra="not present on first reference"),
        dict(symlink=0),
    ],
)
def test_repeated_path_conflicts_are_not_masked_by_coalescing(changes):
    with pytest.raises(compatibility.CompatibilityError, match="Conflicting repeated"):
        compatibility.coalesce_asset_inventory([inventory(), inventory("q1", **changes)])


def test_distinct_logical_paths_equal_content_stay_distinct():
    result = compatibility.coalesce_asset_inventory(
        [inventory(), inventory("q1", path="/project/b.png", resolved_path="/project/b.png")]
    )
    assert result["unique_file_count"] == 2
    assert [item["file_index"] for item in result["question_references"]] == [0, 1]


@pytest.mark.parametrize(
    "changes",
    [
        dict(qid=""),
        dict(path="relative.png"),
        dict(path="/project/../outside.png"),
        dict(path="/project/./a.png"),
        dict(resolved_path="relative.png"),
        dict(path=None),
        dict(sha256="wrong"),
        dict(bytes=True),
        dict(bytes=-1),
    ],
)
def test_malformed_asset_inventory_is_rejected(changes):
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.coalesce_asset_inventory([inventory(**changes)])


def test_duplicate_qid_is_not_deduplicated():
    with pytest.raises(compatibility.CompatibilityError, match="duplicate image question"):
        compatibility.coalesce_asset_inventory([inventory(), inventory()])


def test_adapter_does_not_relax_original_content_or_path_boundary(tmp_path):
    boundary = tmp_path / "project"
    boundary.mkdir()
    path = boundary / "image.png"
    path.write_bytes(b"image-one")
    original = inventory(path=str(path), resolved_path=str(path), bytes=9, sha256=file_hash(path))
    original.pop("stat_cache_key")
    rows = [original, {**original, "qid": "q1"}]
    with pytest.raises(PermissionError, match="duplicated"):
        verify_bound_files(rows, boundary=boundary)
    adapted = compatibility.coalesce_asset_inventory(rows)
    assert verify_bound_files(adapted["unique_files"], boundary=boundary)["status"] == "PASS"
    path.write_bytes(b"image-two")
    with pytest.raises(PermissionError, match="content changed"):
        verify_bound_files(adapted["unique_files"], boundary=boundary)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image-one")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(PermissionError, match="changed, duplicated, or escaped"):
        verify_bound_files(adapted["unique_files"], boundary=boundary)


def test_visual_aliases_preserve_upstream_fields_and_exact_provenance():
    row = visual()
    original = copy.deepcopy(row)
    result = compatibility.normalize_visual_item(row)
    assert row == original
    assert all(result[name] == value for name, value in original.items())
    assert result["reviewed_source"] is result["reviewed_processed"] is True
    assert result["issues"] == []
    migration = result["_migration"]
    assert migration["source_object_sha256"] == digest(row)
    assert migration["field_mapping"] == dict(
        reviewed_source="source_viewed", reviewed_processed="processed_viewed", issues="defects"
    )
    assert not migration["new_visual_review_performed"] and not migration["gate_signed"]


def test_negative_visual_evidence_is_not_promoted():
    row = visual(status="FAIL", source_viewed=False, defects=["unreadable"])
    result = compatibility.normalize_visual_item(row)
    assert result["status"] == "FAIL" and result["reviewed_source"] is False
    assert result["issues"] == ["unreadable"]


@pytest.mark.parametrize("missing", ["source_viewed", "processed_viewed", "defects", "observation"])
def test_missing_visual_evidence_is_not_invented(missing):
    row = visual()
    row.pop(missing)
    with pytest.raises(compatibility.CompatibilityError, match="Missing visual"):
        compatibility.normalize_visual_item(row)


@pytest.mark.parametrize(
    "changes",
    [
        dict(reviewed_source=False),
        dict(reviewed_source=1),
        dict(reviewed_processed=False),
        dict(issues=["unreadable"]),
        dict(defects="none"),
        dict(source_viewed=1),
        dict(source_sha256="invalid"),
        dict(observation=" "),
    ],
)
def test_conflicting_or_malformed_visual_evidence_rejected(changes):
    with pytest.raises(compatibility.CompatibilityError):
        compatibility.normalize_visual_item(visual(**changes))


def test_matching_aliases_allowed_but_canonical_provenance_explicit():
    result = compatibility.normalize_visual_item(visual(reviewed_source=True))
    assert result["_migration"]["field_mapping"]["reviewed_source"] == "reviewed_source"


@pytest.fixture
def receipt_mirror(tmp_path):
    """Original negative visual fixture remains observable in derived report."""
    (tmp_path / "PREFLIGHT").mkdir()
    source = dict(
        retention_images={
            panel: [inventory(f"q{i}") for i in range(count)]
            for panel, count in (("LEGACY", 128), ("CHARTQA", 250))
        }
    )
    images = [
        dict(qid=f"q{i}", source_sha256="a" * 64, processed_file_sha256="b" * 64) for i in range(48)
    ]
    processor = tmp_path / "PREFLIGHT/PROCESSOR.json"
    processor.write_text(json.dumps(dict(images=images)))
    items = [visual(qid=f"q{i}") for i in range(48)]
    items[0]["status"] = "FAIL"
    (tmp_path / "PREFLIGHT/VISUAL_REVIEW.json").write_text(
        json.dumps(dict(status="FAIL", processor_receipt_sha256=file_hash(processor), items=items))
    )
    (tmp_path / "SOURCE_BINDING.json").write_text(json.dumps(source))
    return tmp_path


def test_readonly_assessment_reports_translations_not_production_pass(receipt_mirror):
    before = {path: path.read_bytes() for path in receipt_mirror.rglob("*.json")}
    result = compatibility.assess_receipt_compatibility(
        receipt_mirror, consumer_version=compatibility.CONSUMER_VERSION
    )
    assert result["status"] == "RECORDED_RECEIPTS_ADAPTED_NOT_ADMISSION"
    assert result["gate_status"] == "NOT_SIGNED"
    assert not result["training_authorized"] and not result["generation_authorized"]
    assert result["server_asset_validation"] == "NOT_RUN"
    assert result["processor_pixel_and_tokenizer_validation"] == "NOT_RUN"
    assert result["visual"]["item_count"] == 48 and result["visual"]["aliased_item_count"] == 48
    assert result["visual"]["unresolved_qids"] == ["q0"]
    assert {path: path.read_bytes() for path in receipt_mirror.rglob("*.json")} == before


@pytest.mark.parametrize("version", [None, True, "interface-v1", "interface-v2"])
def test_consumer_version_must_be_exact_opt_in(receipt_mirror, version):
    with pytest.raises(compatibility.CompatibilityError, match="Explicit interface-v2"):
        compatibility.assess_receipt_compatibility(receipt_mirror, consumer_version=version)


def test_visual_cannot_be_mapped_to_a_different_processor_receipt(receipt_mirror):
    path = receipt_mirror / "PREFLIGHT/VISUAL_REVIEW.json"
    value = json.loads(path.read_text())
    value["processor_receipt_sha256"] = "f" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(compatibility.CompatibilityError, match="different processor"):
        compatibility.assess_receipt_compatibility(
            receipt_mirror, consumer_version=compatibility.CONSUMER_VERSION
        )
