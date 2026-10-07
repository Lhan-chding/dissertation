"""Regression checks for fixed matrix scheduling and explicit CLI claims."""

import json
import shutil
from pathlib import Path

import pytest

from src.exposure_substitution import queue as module


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    plan = Path(__file__).resolve().parents[1] / "docs/exposure_substitution/design"
    (tmp_path / "manifests").mkdir()
    for name in ("training_jobs.jsonl", "evaluation_jobs.jsonl"):
        shutil.copyfile(plan / "manifests" / name, tmp_path / "manifests" / name)
    (tmp_path / "FROZEN_PLAN.json").write_text(json.dumps({"fixture": "queue-only"}))
    monkeypatch.setattr(module, "code_identity", lambda: {"fixture": "bound"})
    monkeypatch.setattr(module, "analysis_identity", lambda: {"fixture": "bound"})
    module.build_queue(tmp_path)
    return tmp_path


def test_exact_matrix_first_triad_and_named_claim(prepared):
    queue = module.registered_queue(prepared)
    assert len(queue.rows()) == 113
    initial = [queue.claim(f"lane-{i}") for i in range(3)]
    assert {j["arm"] for j in initial} == {"A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3"}
    assert all(j["parent"] == "S96" and j["block"] == 0 for j in initial)
    later = module.train_job_id("S96", 1, "A_LOCAL_C1")
    with pytest.raises(ValueError, match="Initial registered"):
        module.claim_registered(queue, later, "direct")
    with pytest.raises(ValueError, match="active"):
        module.claim_registered(queue, initial[0]["id"], "second-owner")


def test_named_eval_cannot_bypass_checkpoint_or_gpu_concurrency(prepared):
    queue = module.registered_queue(prepared)
    target = "eval.S96.0.A_LOCAL_C1.step128.E_DIAG"
    with pytest.raises(ValueError, match="checkpoints"):
        module.claim_registered(queue, target, "direct")
    for i in range(5):
        assert queue.claim(f"lane-{i}") is not None
    parent = "eval.REP96.baseline.PARENT.step0.E_DIAG"
    with pytest.raises(ValueError, match="concurrency"):
        module.claim_registered(queue, parent, "sixth")


def test_completed_named_job_is_not_repeated_and_code_drift_rejected(prepared, monkeypatch):
    queue = module.registered_queue(prepared)
    job = module.train_job_id("S96", 0, "A_LOCAL_C1")
    module.claim_registered(queue, job, "owner")
    queue.finish(job, "owner", "COMPLETE", {"fixture": True})
    assert module.claim_registered(queue, job, "again") is None
    monkeypatch.setattr(module, "code_identity", lambda: {"fixture": "changed"})
    with pytest.raises(ValueError, match="source changed"):
        module.registered_queue(prepared)


def test_scientific_arm_cannot_be_removed_from_registry(prepared):
    path = prepared / "manifests/training_jobs.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="exactly 18"):
        module.build_queue(prepared)
