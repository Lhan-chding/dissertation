"""No-model exclusion audits fail closed on missing, malformed or drifting history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.exposure_position.exclusions import (
    REQUIRED_CATEGORIES,
    audit_exclusions,
    extract_truth_orbits,
    load_verified_exclusions,
    verify_exclusion_audit,
)
from src.exposure_position.schema import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


@pytest.fixture
def sources(tmp_path):
    historical = write(
        tmp_path / "historical_exclusion.json",
        {
            "schema": "vdt-historical-exclusion-v1",
            "complete": True,
            "errors": [],
            "sources": [
                {"path": "/historical/train.jsonl", "scope": "generated scenes", "sha256": "a" * 64}
            ],
            "exposed_scene_count": 1,
            "truth_orbits": [[1, 2, 3, 4]],
        },
    )
    legacy = write(
        tmp_path / "legacy" / "audit_only.jsonl",
        {
            "family": "cross_series",
            "true_world": [40, 30, 20, 10],
            "truth_orbit_key": [10, 20, 30, 40],
        },
    )
    later = write(
        tmp_path / "later" / "audit_only.jsonl",
        {
            "family": "trend",
            "true_world": [10, 20, 30, 40],
        },
    )
    manual = write(tmp_path / "manual.json", {"truth_orbit_keys": [[20, 40, 50, 60]]})
    return [
        {"path": str(historical), "category": "historical_index", "kind": "history_index"},
        {"path": str(legacy), "category": "legacy_j2", "kind": "audit"},
        {"path": str(later), "category": "later_local", "kind": "audit"},
        {"path": str(manual), "category": "manual_fixtures", "kind": "manual_fixture"},
    ]


def test_global_orbits_deduplicate_permutations_across_families_and_write_receipts(
    tmp_path, sources
):
    destination = tmp_path / "receipt"
    audit = audit_exclusions(exact_sources=sources, output_dir=destination)
    assert audit["status"] == "PASS"
    assert audit["truth_orbit_count"] == 3
    assert audit["model_calls"] == 0
    assert all(r["sha256"] and r["bytes"] and r["truth_orbits"] for r in audit["files"])
    assert load_verified_exclusions(destination / "EXCLUSION_AUDIT.json") == {
        (1, 2, 3, 4),
        (10, 20, 30, 40),
        (20, 40, 50, 60),
    }
    truth = json.loads((destination / "TRUTH_ORBITS.json").read_text())
    assert truth["audit_hash"] == audit["audit_hash"]
    assert truth["truth_orbit_keys"] == audit["truth_orbit_keys"]
    assert audit["coverage"]["not_claimed"] == ["all_private_files", "unlisted_hosts_or_roots"]


def test_missing_historical_index_blocks_instead_of_certifying_empty_history(sources):
    Path(sources[0]["path"]).unlink()
    audit = audit_exclusions(exact_sources=sources)
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"
    assert audit["truth_orbit_count"] == 2
    assert any(r["status"] == "UNREADABLE_OR_INVALID" for r in audit["files"])
    with pytest.raises(ValueError, match="PASS"):
        verify_exclusion_audit(audit)


@pytest.mark.parametrize(
    "mutation",
    [
        {"truth_orbits": []},
        {"complete": False},
        {"errors": ["unreadable source"]},
        {"sources": []},
        {"sources": [{"path": "/x", "scope": "history", "sha256": "bogus"}]},
        {"exposed_scene_count": 0},
    ],
)
def test_incomplete_or_empty_index_is_blocked(sources, mutation):
    path = Path(sources[0]["path"])
    write(path, {**json.loads(path.read_text()), **mutation})
    assert audit_exclusions(exact_sources=sources)["status"] == "BLOCKED_EXCLUSION_AUDIT"


@pytest.mark.parametrize(
    "vector", [[1, 2, 3], [1, 2, 3, 100], [1, 2, 3, True], [1, 2, 3, 4.0], "[1,2,3,4]", None]
)
def test_malformed_explicit_vector_fails_closed(vector):
    with pytest.raises(ValueError):
        extract_truth_orbits({"true_world": vector})


def test_never_reads_raw_answers_observed_or_arbitrary_target_strings_as_truth():
    value = {
        "true_world": [1, 2, 3, 4],
        "target": "[10,20,30,40]",
        "observed": [40, 30, 20, 10],
        "raw_text": '{"canonical_vector":[20,30,40,50]}',
        "responses": [{"canonical_vector": [20, 30, 40, 50]}],
        "answers": [{"true_world": [50, 60, 70, 80]}],
    }
    assert extract_truth_orbits(value) == {(1, 2, 3, 4)}


def test_declared_training_target_requires_source_and_consistent_explicit_truth():
    assert extract_truth_orbits(
        {"source": "frozen_label", "target": "[4,3,2,1]"}, kind="training_targets"
    ) == {(1, 2, 3, 4)}
    with pytest.raises(ValueError, match="source identity"):
        extract_truth_orbits({"target": "[1,2,3,4]"}, kind="training_targets")
    with pytest.raises(ValueError, match="disagree"):
        extract_truth_orbits(
            {"source": "frozen_label", "target": "[4,3,2,1]", "true_world": [10, 20, 30, 40]},
            kind="training_targets",
        )
    with pytest.raises(ValueError, match="already be sorted"):
        extract_truth_orbits({"truth_orbit_keys": [[4, 3, 2, 1]]})


def test_non_truth_exact_source_is_not_silently_accepted(tmp_path, sources):
    raw = write(tmp_path / "raw_answer.json", {"raw_text": "[1,2,3,4]"})
    audit = audit_exclusions(
        exact_sources=[
            *sources,
            {
                "path": str(raw),
                "kind": "audit",
                "category": "later_local",
            },
        ]
    )
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"
    assert any("no recognized truth" in error["error"] for error in audit["errors"])


def test_byte_and_directory_scans_are_reproducible_and_detect_new_files(tmp_path, sources):
    later = tmp_path / "later"
    write(later / "evaluations" / "audit_only.jsonl", {"true_world": [90, 91, 92, 93]})
    write(later / "not_selected.json", {"true_world": [80, 81, 82, 83]})
    audit = audit_exclusions(
        exact_sources=[s for s in sources if s["category"] != "later_local"],
        scan_roots=[{"path": str(later), "category": "later_local"}],
    )
    assert audit["status"] == "PASS"
    assert audit["coverage"]["scan_roots"][0]["matched_files"] == [str(later / "audit_only.jsonl")]
    assert len(verify_exclusion_audit(audit)) == 3
    write(later / "new_audit.jsonl", {"true_world": [70, 71, 72, 73]})
    with pytest.raises(ValueError, match="changed"):
        verify_exclusion_audit(audit)


def test_empty_scan_cannot_satisfy_later_history(tmp_path, sources):
    empty = tmp_path / "empty"
    empty.mkdir()
    audit = audit_exclusions(
        exact_sources=[s for s in sources if s["category"] != "later_local"],
        scan_roots=[{"path": str(empty), "category": "later_local"}],
    )
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"
    assert audit["coverage"]["scan_roots"][0]["matched_files"] == []


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_file_or_ancestor_is_rejected(tmp_path, sources, directory):
    target = Path(sources[2]["path"])
    link = tmp_path / "link"
    link.symlink_to(target.parent if directory else target, target_is_directory=directory)
    sources[2]["path"] = str(link / target.name if directory else link)
    audit = audit_exclusions(exact_sources=sources)
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"
    assert any("Symlink" in error["error"] for error in audit["errors"])


def test_symlink_encountered_in_scan_is_reported_and_blocks(tmp_path, sources):
    (tmp_path / "later" / "alias").symlink_to(tmp_path / "legacy", target_is_directory=True)
    audit = audit_exclusions(
        exact_sources=sources,
        scan_roots=[
            {
                "path": str(tmp_path / "later"),
                "category": "later_local",
            }
        ],
    )
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"
    assert any("symlink directory" in error["error"] for error in audit["errors"])


def test_path_traversal_size_bound_and_malformed_json_are_retained(tmp_path, sources):
    sources.append(
        {
            "path": str(tmp_path / "legacy" / ".." / "manual.json"),
            "kind": "audit",
            "category": "later_local",
        }
    )
    audit = audit_exclusions(exact_sources=sources, max_file_bytes=10)
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"
    assert any("traversal" in error["error"] for error in audit["errors"])
    assert any("byte bound" in error["error"] for error in audit["errors"])
    Path(sources[2]["path"]).write_text("{malformed}")
    assert audit_exclusions(exact_sources=sources[:4])["status"] == "BLOCKED_EXCLUSION_AUDIT"


def test_modified_sources_or_receipt_do_not_retain_pass(sources):
    audit = audit_exclusions(exact_sources=sources)
    original = json.loads(json.dumps(audit))
    audit["truth_orbit_keys"].append([90, 91, 92, 93])
    with pytest.raises(ValueError, match="PASS"):
        verify_exclusion_audit(audit)
    write(Path(sources[2]["path"]), {"true_world": [90, 91, 92, 93]})
    with pytest.raises(ValueError, match="changed"):
        verify_exclusion_audit(original)


def test_bad_truth_retains_file_hash_in_blocking_receipt(sources):
    write(Path(sources[2]["path"]), {"true_world": [10, 20, 30, 100]})
    audit = audit_exclusions(exact_sources=sources)
    bad = next(record for record in audit["files"] if record["category"] == "later_local")
    assert bad["status"] == "UNREADABLE_OR_INVALID"
    assert len(bad["sha256"]) == 64
    assert bad["bytes"] > 0
    assert bad["truth_orbits"] is None
    assert audit["status"] == "BLOCKED_EXCLUSION_AUDIT"


def test_receipt_cannot_overwrite_symlink_destination(tmp_path, sources):
    destination = tmp_path / "receipt"
    destination.mkdir()
    target = write(tmp_path / "protected.json", {"keep": True})
    (destination / "EXCLUSION_AUDIT.json").symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        audit_exclusions(exact_sources=sources, output_dir=destination)
    assert json.loads(target.read_text()) == {"keep": True}


def test_weak_required_category_contract_cannot_be_consumed(sources):
    audit = audit_exclusions(exact_sources=sources[:1], required_categories=["historical_index"])
    assert audit["status"] == "PASS"
    with pytest.raises(ValueError, match="coverage"):
        verify_exclusion_audit(audit)
    audit["coverage"]["required_categories"] = list(REQUIRED_CATEGORIES)
    audit["audit_hash"] = digest({k: v for k, v in audit.items() if k != "audit_hash"})
    with pytest.raises(ValueError, match="coverage"):
        verify_exclusion_audit(audit)
