"""Compute repair entrypoints preserve one-shot transitions and source precedence."""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / "sr_f1" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compute_cli_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def argv(tmp_path, *extra):
    return [
        "submit_matrix.py",
        "--plan",
        str(tmp_path / "plan.json"),
        "--run-root",
        str(tmp_path),
        "--code-root",
        str(tmp_path / "code"),
        "--python",
        sys.executable,
        "--resume-repaired-engine-compute-parallel",
        *extra,
    ]


def test_compute_resume_is_one_shot_and_does_not_tick_or_submit(tmp_path, monkeypatch, capsys):
    module = load_script("submit_matrix")
    scheduler = Mock()
    scheduler.resume_repaired_engine_compute_parallel.return_value = {"status": "AUTHORIZED"}
    monkeypatch.setattr(module, "Scheduler", Mock(return_value=scheduler))
    lease = Mock(return_value=contextlib.nullcontext())
    monkeypatch.setattr(module, "process_lease", lease)
    monkeypatch.setattr(sys, "argv", argv(tmp_path))
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "AUTHORIZED"}
    scheduler.resume_repaired_engine_compute_parallel.assert_called_once_with()
    scheduler.tick.assert_not_called()
    scheduler.resume_stopped.assert_not_called()
    lease.assert_called_once_with(tmp_path / "orchestration/controller_process.lock")


@pytest.mark.parametrize(
    "other",
    [
        "--watch",
        "--resume-stopped",
        "--controller-requeue-on-lease",
        "--resume-repaired-common-start",
        "--resume-repaired-engine",
        "--resume-repaired-engine-storage",
        "--resume-repaired-engine-io",
        "--resume-repaired-engine-multigpu",
    ],
)
def test_compute_resume_cannot_combine_with_other_modes(tmp_path, monkeypatch, other):
    module = load_script("submit_matrix")
    scheduler = Mock()
    monkeypatch.setattr(module, "Scheduler", scheduler)
    monkeypatch.setattr(sys, "argv", argv(tmp_path, other))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    scheduler.assert_not_called()


@pytest.mark.parametrize(
    "count,expected", [(0, None), (1, "multi"), (2, "baseline"), (3, "compute")]
)
def test_current_worker_source_repair_precedence(tmp_path, monkeypatch, count, expected):
    module = load_script("run_worker")
    names = [
        "ENGINE_MULTIGPU_REPAIR.json",
        "BASELINE_PARALLEL_REPAIR.json",
        "ENGINE_COMPUTE_PARALLEL_REPAIR.json",
    ]
    for name in names[:count]:
        (tmp_path / name).write_text("{}")
    checks = {}
    for name, result in (
        ("multigpu_repair", "multi"),
        ("baseline_parallel_repair", "baseline"),
        ("compute_parallel_repair", "compute"),
    ):
        checks[result] = Mock(return_value=result)
        monkeypatch.setattr(module, name, checks[result])
    assert module.current_worker_repair(tmp_path) == expected
    for result, check in checks.items():
        if result == expected:
            check.assert_called_once_with(tmp_path)
        else:
            check.assert_not_called()


def test_invalid_compute_repair_never_falls_back_to_old_worker_source(tmp_path, monkeypatch):
    module = load_script("run_worker")
    (tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").write_text("{}")
    (tmp_path / "ENGINE_MULTIGPU_REPAIR.json").write_text("{}")
    monkeypatch.setattr(
        module, "compute_parallel_repair", Mock(side_effect=PermissionError("changed"))
    )
    fallback = Mock()
    monkeypatch.setattr(module, "multigpu_repair", fallback)
    with pytest.raises(PermissionError, match="changed"):
        module.current_worker_repair(tmp_path)
    fallback.assert_not_called()


def test_compute_repair_dispatches_parallel_baseline_without_standalone_baseline_receipt(
    tmp_path, monkeypatch
):
    import sr_f1.baseline_parallel as parallel
    import sr_f1.orchestration as orchestration
    import sr_f1.runtime as runtime

    module = load_script("run_worker")
    (tmp_path / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").write_text("{}")
    assert not (tmp_path / "BASELINE_PARALLEL_REPAIR.json").exists()
    registration = {"tasks": {"BASELINE": {"gpus": 1}}}
    monkeypatch.setattr(
        module, "_worker_identity", lambda root, task: (registration, "BASELINE_attempt0000")
    )
    monkeypatch.setattr(
        orchestration, "baseline_parallel_allocation", lambda root, registry: {"gpus": 4}
    )
    run = Mock(return_value={"status": "COMPLETE"})
    monkeypatch.setattr(parallel, "run_parallel_baseline", run)
    loader = Mock(side_effect=AssertionError("Parent must not load evaluation model"))
    monkeypatch.setattr(runtime, "load_for_evaluation", loader)
    result = module.dispatch(
        {},
        tmp_path,
        {"operation": "evaluate", "stage": "baseline", "model_id": "SRF1_COMMON_START"},
    )
    assert result == {"status": "COMPLETE"}
    run.assert_called_once_with({}, tmp_path, 4)
    loader.assert_not_called()
