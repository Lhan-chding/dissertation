"""Regression checks for read-only historical replay provenance boundaries."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "ssvc_flow/scripts/verified_discovery_transfer/prepare_historical_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("vdt_historical_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(root, *, split="train", raw="[1,2,3,4]", banned=False, prompt="SYS USER"):
    historical = root / "prospective_selection_v2_20260924"
    historical.mkdir()
    record = {
        "prompt_id": "p",
        "base_scene_id": "scene",
        "family": "duplicate_encoding",
        "split": "train",
        "interface": "SYMBOLIC_FRESH",
        "scene": {"base_scene_id": "scene", "truth_world": [1, 2, 3, 4]},
        "prompt": {"system": "SYS", "user": "USER"},
    }
    prepared = {"source_prompts": [record], "continuation_prompts": [], "panels": {"D": []}}
    (historical / "prepared_development.json").write_text(json.dumps(prepared))
    cases = (
        root
        / "protocol_state_probes_v1_20261005/code_ad09688"
        / "ssvc_flow/docs/protocol_state_probes/design/manifests/cases.jsonl"
    )
    cases.parent.mkdir(parents=True)
    cases.write_text(json.dumps({"base_scene_id": "scene" if banned else "other"}) + "\n")
    samples = historical / "campaign/sources/61001/segments/001_008/attempt_1/samples.jsonl"
    samples.parent.mkdir(parents=True)
    row = {
        "prompt_id": "p",
        "base_scene_id": "scene",
        "split": split,
        "role": split,
        "interface": "SYMBOLIC_FRESH",
        "event": "X",
        "category": "X",
        "raw_completion": raw,
        "input_audit": {"final_prompt": prompt},
        "sample_key": "real-request",
        "lineage_id": 61001,
        "step": 1,
    }
    samples.write_text(json.dumps(row) + "\n")
    commit = {
        "samples": {"path": str(samples), "sha256": MODULE.digest(samples.read_bytes())},
        "start": 1,
        "stop": 8,
    }
    (samples.parent.parent / "COMMIT.json").write_text(json.dumps(commit))


def test_actual_train_success_keeps_raw_provenance_and_exclusion(tmp_path):
    fixture(tmp_path)
    result = MODULE.collect(tmp_path)
    assert result["exclusion"]["truth_orbits"] == [[1, 2, 3, 4]]
    replay = result["replay_candidates"]
    assert len(replay) == 1
    assert replay[0]["source"]["request_id"] == "real-request"
    assert replay[0]["raw_completion"] == "[1,2,3,4]"
    assert not result["replay_receipt"]["complete"]


def test_evaluation_outputs_are_not_replay(tmp_path):
    fixture(tmp_path, split="dev")
    assert MODULE.collect(tmp_path)["replay_candidates"] == []


def test_panel_scene_is_banned_even_if_train_labeled(tmp_path):
    fixture(tmp_path, banned=True)
    assert MODULE.collect(tmp_path)["replay_candidates"] == []


def test_recorded_x_does_not_override_wrong_completion(tmp_path):
    fixture(tmp_path, raw="[1,2,3,9]")
    assert MODULE.collect(tmp_path)["replay_candidates"] == []


def test_prompt_identity_mismatch_is_rejected(tmp_path):
    fixture(tmp_path, prompt="OTHER PROTOCOL")
    assert MODULE.collect(tmp_path)["replay_candidates"] == []


def test_unparseable_metadata_makes_completeness_false(tmp_path):
    fixture(tmp_path)
    (tmp_path / "old_scenes.jsonl").write_text("broken\n")
    result = MODULE.collect(tmp_path)
    assert not result["exclusion"]["complete"]
    assert len(result["exclusion"]["errors"]) == 1


def test_uncommitted_attempt_never_enters_replay(tmp_path):
    fixture(tmp_path)
    for path in tmp_path.rglob("COMMIT.json"):
        path.unlink()
    assert MODULE.collect(tmp_path)["replay_candidates"] == []


def test_commit_hash_tampering_fails_closed(tmp_path):
    fixture(tmp_path)
    samples = next(tmp_path.rglob("samples.jsonl"))
    samples.write_text(samples.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.collect(tmp_path)


def test_scene_sort_happens_after_full_committed_scan(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(MODULE, "QUOTAS", {"duplicate_encoding": 1})
    prepared_path = tmp_path / "prospective_selection_v2_20260924/prepared_development.json"
    prepared = json.loads(prepared_path.read_text())
    later = dict(prepared["source_prompts"][0], prompt_id="a-prompt", base_scene_id="aaa")
    prepared["source_prompts"].append(later)
    prepared_path.write_text(json.dumps(prepared))
    samples = next(tmp_path.rglob("samples.jsonl"))
    row = json.loads(samples.read_text())
    row.update(prompt_id="a-prompt", base_scene_id="aaa", sample_key="later-but-lower-id")
    samples.write_text(samples.read_text() + json.dumps(row) + "\n")
    commit_path = next(tmp_path.rglob("COMMIT.json"))
    commit = json.loads(commit_path.read_text())
    commit["samples"]["sha256"] = MODULE.digest(samples.read_bytes())
    commit_path.write_text(json.dumps(commit))
    replay = MODULE.collect(tmp_path)["replay_candidates"]
    assert len(replay) == 1
    assert replay[0]["source"]["base_scene_id"] == "aaa"


def test_dataset_binding_audit_follows_explicit_root(tmp_path):
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "train.jsonl").write_text(json.dumps({"truth_world": [4, 3, 2, 1]}) + "\n")
    (tmp_path / "runtime.json").write_text(json.dumps({"data_root": str(data)}))
    audit = MODULE.audit_dataset_bindings(tmp_path)
    assert len(audit["bindings"]) == 1
    assert audit["datasets"][0]["truth_orbits"] == [[1, 2, 3, 4]]
    assert audit["errors"] == []
