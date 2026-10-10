"""The parallel parent must dispatch before any CUDA model is loaded."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest


def worker_module():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f1/run_worker.py"
    spec = importlib.util.spec_from_file_location("baseline_dispatch_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallel_dispatch_uses_fixed_registered_allocation_without_parent_model(
    tmp_path, monkeypatch
):
    import sr_f1.baseline_parallel as parallel
    import sr_f1.orchestration as orchestration
    import sr_f1.runtime as runtime

    worker = worker_module()
    (tmp_path / "BASELINE_PARALLEL_REPAIR.json").write_text("{}")
    registry = {"tasks": {"BASELINE": {"gpus": 1}}}
    monkeypatch.setattr(
        worker, "_worker_identity", lambda root, task: (registry, "BASELINE_attempt0000")
    )
    allocation = Mock(return_value={"gpus": 4})
    monkeypatch.setattr(orchestration, "baseline_parallel_allocation", allocation)
    run = Mock(return_value={"status": "COMPLETE", "artifacts": ["coverage.json"]})
    monkeypatch.setattr(parallel, "run_parallel_baseline", run)
    loader = Mock(side_effect=AssertionError("Parent must not load model"))
    monkeypatch.setattr(runtime, "load_for_evaluation", loader)
    plan = {"version": "test"}
    result = worker.dispatch(
        plan,
        tmp_path,
        {
            "operation": "evaluate",
            "stage": "baseline",
            "model_id": "SRF1_COMMON_START",
        },
    )
    assert result["status"] == "COMPLETE"
    allocation.assert_called_once_with(tmp_path, registry)
    run.assert_called_once_with(plan, tmp_path, 4)
    loader.assert_not_called()


def test_parallel_dispatch_rejects_unfrozen_model(tmp_path):
    worker = worker_module()
    (tmp_path / "BASELINE_PARALLEL_REPAIR.json").write_text("{}")
    with pytest.raises(ValueError, match="PARALLEL_BASELINE_REQUIRES_COMMON_START"):
        worker.dispatch(
            {},
            tmp_path,
            {
                "operation": "evaluate",
                "stage": "baseline",
                "model_id": "SRF1_A_s71001",
            },
        )
