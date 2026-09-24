import json

import pytest

from src.prospective_selection.history import HISTORICAL_ENDPOINTS, audit_history, audit_rows
from src.prospective_selection.semantics import semantic_features


def prompt():
    return {
        "prompt_id": "p",
        "base_scene_id": "scene",
        "family": "trend",
        "track": "N",
        "scene": {
            "truth_world": [10, 20, 30, 40],
            "observed_world": [10, 25, 30, 40],
            "cue": {"family": "trend"},
            "operation": "sum4",
        },
    }


def row(raw, sample_id):
    phi = semantic_features(raw, prompt())
    return {
        "sample_id": sample_id,
        "prompt_id": "p",
        "raw_completion": raw,
        "origin": "O1",
        "action": "R0",
        "panel": "P",
        **phi,
    }


def test_exact_identities_invalid_missing_and_same_reward_structure():
    # Both W,C=0; copy versus changing an originally correct coordinate.
    rows = [
        row("[10,25,30,40]", "a"),
        row("[11,25,30,40]", "b"),
        row("[10,20,30,40]", "c"),
        row("bad", "d"),
    ]
    result = audit_rows(rows, [prompt()])
    assert result["identities_pass"]
    assert result["counts"] == {"rows": 4, "W": 2, "trend_rows": 4, "X": 1, "I": 1}
    assert result["same_reward_different_repair_buckets"] == 1
    assert result["endpoint_metrics"][0]["coordinate_accuracy_invalid_zero"] == 9 / 16
    assert result["endpoint_metrics"][0]["damage_any_given_valid"] == 1 / 3


def test_reparse_rejects_corrupt_historical_labels_and_duplicate_samples():
    original = row("[10,20,30,40]", "a")
    with pytest.raises(ValueError, match="semantic mismatch"):
        audit_rows([{**original, "event": "W"}], [prompt()])
    with pytest.raises(ValueError, match="Duplicate"):
        audit_rows([original, original], [prompt()])


def test_p_results_never_satisfy_E_reuse_or_confirm_unseen(tmp_path):
    source = {
        "root": "fixture",
        "panels": {"panels": {"P": [prompt()], "E": [prompt()]}, "sealed_confirm_read": False},
        "rows": [
            dict(row("[10,20,30,40]", str(i)), origin=o, action=a)
            for o, a in HISTORICAL_ENDPOINTS
            for i in range(32)
        ],
        "endpoints": [
            {
                "origin": o,
                "action": a,
                "state_present": True,
                "observations": [{"panel": "P", "look": 32, "imported_rows": 32}],
            }
            for o, a in HISTORICAL_ENDPOINTS
        ],
        "source_receipts": [],
        "access_records": [],
        "confirm_payload_opened_by_audit": False,
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    result = audit_history({"history_import": str(path)}, tmp_path / "out")
    assert result["present_endpoints"] == 16
    assert len(result["E_missing_endpoints"]) == 16
    assert result["E_reusable_endpoints"] == []
    assert result["E_status"] == "E_MEASUREMENTS_REQUIRED"
    assert result["exposure"]["absence_of_access_record_proves_never_read"] is False

    source["rows"].pop()
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="exactly 32"):
        audit_history({"history_import": str(path)}, tmp_path / "out")
