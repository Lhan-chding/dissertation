"""M5 uses completed local evidence and never manufactures missing tensors."""

import hashlib
import json

import pytest

from src.core import canonical_hash
from src.modeling_qualification.real_audit import REQUIRED_FIELDS, _read_ledger, run_audit_real


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))


def fixture(root, *, fault=False, mismatched=False, raw=True):
    bank = root / "S1" / "bank00"
    identity = {
        "run_id": "run",
        "bank_id": "bank00",
        "origin_state_hash": "origin",
        "source_hash": "source",
        "execution_kind": "REAL_CUDA_FOLLOWUP",
    }
    write(bank / "identity.json", identity)
    write(bank / "status.json", {"status": "MEASURED", "execution_kind": "REAL_CUDA_FOLLOWUP"})
    write(bank / "origin_binding.json", {"origin_checkpoint_step": 64})
    write(
        bank / "policy_aliases.json",
        {
            "no_x_off_1": {
                "alias_of": "joint_1",
                "additional_samples": 0,
                "candidate_policy_fingerprint": "fingerprint1",
                "matching_rule": "EXACT_COMPLETE_FORWARD_FINGERPRINT",
            }
        },
    )
    for cid, index in (("joint_0", 0), ("joint_1", 1), ("no_x_off_1", 1)):
        write(
            bank / "candidates" / cid / "candidate.json",
            {
                "candidate_id": cid,
                "candidate_state_hash": cid,
                "candidate_optimizer_state_hash": "optim" + cid,
                "candidate_policy_fingerprint": "fingerprint" + str(index),
                "candidate_spec": {"id": cid, "policy": "joint", "auxiliary_weight": index},
                "audit": {"actual_step_norm": 0.1, "optimizer_hash_before": "optim_origin"},
            },
        )
        if cid == "no_x_off_1":
            write(
                bank / "candidates" / cid / "counts_by_prompt.json",
                {
                    "alias_of": "joint_1",
                    "additional_samples": 0,
                    "source_counts": "candidates/joint_1/counts_by_prompt.json",
                },
            )
            continue
        counts = {
            "prompt_id": "prompt",
            "base_scene_id": "scene",
            "family": "trend",
            "interface": "SYMBOLIC_FRESH",
            "n": 2,
            "counts": {"X": 1 + int(mismatched), "S": 0, "W": 0, "I": 1 - int(mismatched)},
        }
        write(bank / "candidates" / cid / "counts_by_prompt.json", [counts])
        if raw:
            records = []
            for i, category in enumerate(("X", "I")):
                row = {
                    **identity,
                    "prompt_id": "prompt",
                    "base_scene_id": "scene",
                    "family": "trend",
                    "interface": "SYMBOLIC_FRESH",
                    "operation": "sum4",
                    "category": category,
                    "candidate_id": cid,
                    "candidate_policy_fingerprint": "fingerprint" + str(index),
                    "train_seed": 17,
                    "sample_key": cid + str(i),
                    "sample_index": i,
                    "execution_checks": {"passed": not fault, "faults": ["bad"] if fault else []},
                }
                row["record_hash"] = canonical_hash(row)
                records.append(row)
            ledger = "\n".join(json.dumps(row) for row in records) + "\n"
            path = bank / "candidates" / cid / "direct" / "samples.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ledger)
            (path.parent.parent / "samples.jsonl").write_text(ledger)
    manifest = {
        "files": [
            {
                "path": str(p.relative_to(bank)),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(bank.rglob("*"))
            if p.is_file()
        ]
    }
    manifest["files"].append(
        {"path": "candidates/joint_0/checkpoint.pt", "bytes": 999, "sha256": "not-local"}
    )
    write(bank / "manifest.json", manifest)
    write(
        bank / "completed.json",
        {
            "status": "MEASURED",
            "identity_hash": canonical_hash(identity),
            "manifest_sha256": hashlib.sha256((bank / "manifest.json").read_bytes()).hexdigest(),
        },
    )
    return bank


def test_raw_counts_alias_missing_tensors_and_readonly(tmp_path):
    root = tmp_path / "real"
    fixture(root)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    out = tmp_path / "audit"
    result = run_audit_real(root, out)
    rows = [
        json.loads(line) for line in (out / "real_evidence_table.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert result["unique_sample_count"] == 4  # Aliases and flat copies are not new samples.
    assert result["real_response_dimension"] is None
    assert all(set(REQUIRED_FIELDS) <= set(row) for row in rows)
    assert all(
        row["update_vector_path"] is None and row["update_vector_hash"] is None for row in rows
    )
    assert all(row["null_reasons"]["update_vector_path"] for row in rows)
    assert rows[2]["alias_of"] == "joint_1"
    assert (
        result["bank_audits"][0]["candidate_counts"]["no_x_off_1"]
        == result["bank_audits"][0]["candidate_counts"]["joint_1"]
    )
    assert rows[0]["evidence_level"] == "LEDGER_COUNTS_RECOMPUTED"
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert str(root) not in (out / "audit_summary.json").read_text()
    with pytest.raises(FileExistsError):
        run_audit_real(root, out)


def test_summary_only_does_not_claim_raw_recomputation(tmp_path):
    fixture(tmp_path / "real", raw=False)
    result = run_audit_real(tmp_path / "real", tmp_path / "out")
    assert result["unique_sample_count"] == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "out/real_evidence_table.jsonl").read_text().splitlines()
    ]
    assert {row["evidence_level"] for row in rows} == {"METADATA_COUNTS"}


@pytest.mark.parametrize("kwargs", [{"fault": True}, {"mismatched": True}])
def test_execution_fault_or_mismatched_counts_is_not_usable_evidence(tmp_path, kwargs):
    fixture(tmp_path / "real", **kwargs)
    result = run_audit_real(tmp_path / "real", tmp_path / "out")
    assert result["audit_status"] == "INTEGRITY_FAILURE"
    assert result["usable_completed_banks"] == []
    assert result["unique_sample_count"] == 0


def test_manifest_corruption_and_unfinished_bank(tmp_path):
    bank = fixture(tmp_path / "real")
    (bank / "origin_binding.json").write_text("{}")
    result = run_audit_real(tmp_path / "real", tmp_path / "bad")
    assert result["audit_status"] == "INTEGRITY_FAILURE"
    (bank / "completed.json").unlink()
    result = run_audit_real(tmp_path / "real", tmp_path / "unfinished")
    assert result["usable_completed_banks"] == []
    assert result["unique_sample_count"] == 0


def test_symlink_source_and_sealed_root_rejected(tmp_path):
    fixture(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_audit_real(alias, tmp_path / "out")
    with pytest.raises(ValueError, match=r"sealed|confirm"):
        run_audit_real(tmp_path / "sealed_confirm", tmp_path / "out")


def test_independent_raw_semantic_reparse_and_scene_binding(tmp_path):
    from src.r2_runtime import annotate_diagnostic

    scene = {
        "base_scene_id": "scene",
        "truth_world": [1, 2, 3, 4],
        "observed_world": [1, 2, 3, 5],
        "changed_index": 3,
        "operation": "sum4",
        "cue": {"family": "duplicate_encoding", "known_index": 3, "known_value": 4},
    }
    candidate = {"candidate_id": "joint_0", "candidate_policy_fingerprint": "fp"}
    row = {
        **candidate,
        "sample_key": "one",
        "sample_index": 0,
        "train_seed": 17,
        "prompt_id": "prompt",
        "base_scene_id": "scene",
        "family": "duplicate_encoding",
        "interface": "SYMBOLIC_FRESH",
        "operation": "sum4",
        "raw_completion": "[1, 2, 3, 4]",
        "scene_hash": canonical_hash(scene),
        "execution_checks": {"passed": True, "faults": []},
        **annotate_diagnostic("[1, 2, 3, 4]", scene),
    }
    path = tmp_path / "samples.jsonl"

    def save():
        row["record_hash"] = canonical_hash({k: v for k, v in row.items() if k != "record_hash"})
        path.write_text(json.dumps(row) + "\n")

    save()
    grouped, keys, reparsed = _read_ledger(path, {}, candidate, {"scene": scene}, True)
    assert reparsed == 1 and grouped["prompt"]["counts"]["X"] == 1 and len(keys) == 1
    row["raw_completion"] = "[1, 2, 3, 5]"
    save()  # A self-consistent record hash cannot disguise a wrong recorded category.
    with pytest.raises(ValueError, match="reparse"):
        _read_ledger(path, {}, candidate, {"scene": scene}, True)
    row["scene_hash"] = "wrong"
    save()
    with pytest.raises(ValueError, match="scene hash"):
        _read_ledger(path, {}, candidate, {"scene": scene}, True)


def test_unsafe_manifest_path_rejected_before_read(tmp_path):
    bank = fixture(tmp_path / "real")
    manifest = json.loads((bank / "manifest.json").read_text())
    manifest["files"].append({"path": "../../outside", "bytes": 1, "sha256": "irrelevant"})
    write(bank / "manifest.json", manifest)
    completed = json.loads((bank / "completed.json").read_text())
    completed["manifest_sha256"] = hashlib.sha256((bank / "manifest.json").read_bytes()).hexdigest()
    write(bank / "completed.json", completed)
    result = run_audit_real(tmp_path / "real", tmp_path / "out")
    assert result["audit_status"] == "INTEGRITY_FAILURE"
    assert result["failures"][0]["reason"] == "unsafe or duplicate manifest entry"
