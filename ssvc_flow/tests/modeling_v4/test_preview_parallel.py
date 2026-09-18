import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import RunWriteConflict, frozen_writer
from src.modeling_v3.vlm_observation import _parity_values
from src.modeling_v4 import gpu_collect as gpu
from src.modeling_v4 import measurement_preview as preview

SCRIPT = Path(__file__).parents[2] / "scripts" / "modeling_v4_preview_parallel.py"
spec = importlib.util.spec_from_file_location("preview_parallel", SCRIPT)
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


def test_disjoint_assignment_and_shared_lock_excludes_serial(tmp_path):
    assert parallel.assigned_indices(0) == [0, 2, 4]
    assert parallel.assigned_indices(1) == [1, 3, 5]
    for invalid in (2, -1, True):
        with pytest.raises(ValueError):
            parallel.assigned_indices(invalid)
    with parallel.shared_measurement_lock(tmp_path), parallel.shared_measurement_lock(tmp_path):  # noqa: SIM117
        with pytest.raises(RunWriteConflict), frozen_writer(tmp_path):
            pass
    with (
        frozen_writer(tmp_path),
        pytest.raises(BlockingIOError),
        parallel.shared_measurement_lock(tmp_path),
    ):
        pass
    with pytest.raises(PermissionError):
        parallel.run("missing", worker_id=0)


def anchor():
    return {
        "runtime_identity": {"source_hash": "frozen"},
        "gpu_identity": "first-gpu",
        "scores": [
            {
                "token_ids": [1],
                "token_logprobs": [-1.0],
                "sequence_logp": -1.0,
                "prompt_id": "null",
                "inference_fingerprint": "null-policy",
            }
        ],
    }


def test_cross_gpu_parity_keeps_policy_actions_and_tolerances():
    old = anchor()
    current = anchor() | {"gpu_identity": "second-gpu"}
    tol = dict.fromkeys(
        ("mean_abs_token_logp", "max_abs_token_logp", "max_abs_sequence_logp"), 1e-6
    )
    assert parallel.check_anchor_parity(old, current, tol, _parity_values)["passed"]
    current["scores"][0]["token_ids"] = [2]
    with pytest.raises(ValueError, match="identity"):
        parallel.check_anchor_parity(old, current, tol, _parity_values)
    current = anchor()
    current["scores"][0]["sequence_logp"] = -1.1
    with pytest.raises(ValueError, match="tolerances"):
        parallel.check_anchor_parity(old, current, tol, _parity_values)


def test_two_workers_reuse_serial_output_and_publish_only_global_completion(tmp_path, monkeypatch):
    plan = {
        "root": str(tmp_path),
        "plan_hash": "unchanged",
        "measurement_source": {"sha256": "frozen"},
        "probes": [{"prompt_id": f"p{i}"} for i in range(6)],
        "draws": 64,
        "reference_draws": 64,
        "source_task_id": "old-origin",
        "scope": "EXPLORATORY",
        "observation_score_mode": "uncached_prefix_recompute",
    }
    tasks = {"config": {}, "bindings": {}, "analysis_rules": {}, "inputs": {}}
    monkeypatch.setattr(preview, "_load_plan", lambda p: (plan, tasks, [], {}))
    monkeypatch.setattr(preview, "_check_runtime", lambda *a: None)
    monkeypatch.setattr(gpu, "_null_receipt", lambda *a: (anchor(), {}))
    monkeypatch.setattr(gpu, "pair_observation_diagnostics", lambda r: {"units": []})
    gpu._publish(tmp_path / "null_receipt.json", anchor())
    sentinel = tmp_path / "prompts/00/response/work/serial-chunk-original"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"previously committed sample")
    root_runtime = tmp_path / "RUNTIME.json"
    root_runtime.write_text('{"frozen":true}')
    calls, model_loads = [], []

    def factory(*a, **kw):
        model_loads.append(True)
        return {
            "identity": anchor()["runtime_identity"],
            "parity_tolerances": dict.fromkeys(
                ("mean_abs_token_logp", "max_abs_token_logp", "max_abs_sequence_logp"), 1e-6
            ),
            "checkpoint_cache": SimpleNamespace(load=lambda p: {}),
            "state": SimpleNamespace(restore=lambda s: None),
        }

    def collect(config, runtime, forks, prompts, **kw):
        assert kw["resume"] is True
        assert forks["origin_id"] == "preview:unchanged:old-origin"
        assert kw["draws"] == kw["reference_draws"] == 64
        pid = prompts[0]["prompt_id"]
        calls.append(pid)
        result = {"prompt_id": pid}
        gpu._publish(Path(kw["out"]) / "COMPLETE.json", result)
        return result

    monkeypatch.setattr(gpu, "collect_response_map", collect)
    parallel.run(
        "plan", worker_id=1, execute_gpu=True, resume=True, preview=preview, runtime_factory=factory
    )
    assert calls == ["p1", "p3", "p5"]
    assert not (tmp_path / "COMPLETE.json").exists()
    assert gpu._read(tmp_path / "LATEST.json")["completed_prompts"] == 3
    parallel.run(
        "plan", worker_id=0, execute_gpu=True, resume=True, preview=preview, runtime_factory=factory
    )
    result = gpu._read(tmp_path / "COMPLETE.json")
    assert [r["prompt_id"] for r in result["prompts"]] == [f"p{i}" for i in range(6)]
    assert len(calls) == len(set(calls)) == 6
    assert sentinel.read_bytes() == b"previously committed sample"
    assert root_runtime.read_text() == '{"frozen":true}'
    assert len(model_loads) == 2
    for worker in (0, 1):
        parallel.run(
            "plan",
            worker_id=worker,
            execute_gpu=True,
            resume=True,
            preview=preview,
            runtime_factory=factory,
        )
    assert len(model_loads) == 2 and len(calls) == 6
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(lambda _: parallel.publish_summary(preview, plan), range(2)))
    assert gpu._read(tmp_path / "LATEST.json")["completed_prompts"] == 6
    # A malformed/wrong-origin completion cannot enter the global summary.
    receipt = tmp_path / "prompts/00/COMPLETE.json"
    d = gpu._read(receipt)
    d["plan_hash"] = "another-plan"
    preview.atomic_json(receipt, d)
    with pytest.raises(ValueError, match="another frozen"):
        parallel.publish_summary(preview, plan)


def test_resume_compares_actual_current_null_not_cached_worker_anchor(tmp_path, monkeypatch):
    previous = anchor()
    gpu._publish(tmp_path / "null_receipt.json", previous)
    actual = anchor()
    actual["scores"][0]["sequence_logp"] = -1.01

    def null(*args):
        gpu._publish(tmp_path / "null_repeat_123.json", {"receipt": actual})
        return previous, {}

    monkeypatch.setattr(gpu, "_null_receipt", null)
    current = parallel.current_null_receipt(gpu, {}, {}, tmp_path)
    assert current == actual
    tol = dict.fromkeys(
        ("mean_abs_token_logp", "max_abs_token_logp", "max_abs_sequence_logp"), 1e-6
    )
    with pytest.raises(ValueError, match="tolerances"):
        parallel.check_anchor_parity(anchor(), current, tol, _parity_values)
