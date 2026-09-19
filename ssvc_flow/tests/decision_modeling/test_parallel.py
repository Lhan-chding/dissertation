"""Parallel workers preserve serial draws and cannot merge partial evidence."""

import importlib.util
from pathlib import Path

import pytest

from src.decision_modeling import measurement
from src.decision_modeling.parallel import merge_workers, partition_prompts, run_worker
from src.decision_modeling.reporting import _load_stream
from src.modeling_v3.vlm_observation import digest


@pytest.fixture
def inputs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "decision_parallel_fixture",
        Path(__file__).with_name("test_runtime_measurement_integration.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime, original, origin, policies = module.inputs.__wrapped__(tmp_path)
    prompts = [
        {
            **original[0],
            "prompt_id": f"p{scene}_{interface}",
            "base_scene_id": f"scene{scene}",
            "interface": interface,
        }
        for scene in range(2)
        for interface in ("choice", "text")
    ]
    runtime["decision_private"] = {
        "config_hash": "fixture",
        "preview_policies": policies,
        "preview_origin": origin,
    }
    return runtime, prompts, origin, policies


def test_balanced_paired_panel_partitions():
    prompts = [
        {
            "prompt_id": f"{family}_{scene}_{interface}",
            "family": family,
            "base_scene_id": f"{family}_{scene}",
            "interface": interface,
        }
        for family in ("a", "b", "c")
        for scene in range(12)
        for interface in ("left", "right")
    ]
    for count in (2, 3):
        parts = partition_prompts(prompts, count)
        assert [len(p) for p in parts] == [72 // count] * count
        assert parts[0][0] == prompts[0]
        for part in parts:
            assert all(sum(p["family"] == f for p in part) == 24 // count for f in ("a", "b", "c"))
            assert all(
                sum(p["base_scene_id"] == row["base_scene_id"] for p in part) == 2 for row in part
            )
    with pytest.raises(ValueError, match="divide"):
        partition_prompts(prompts[:10], 2)


def test_parallel_matches_serial_and_merge_requires_complete_workers(inputs, tmp_path):
    runtime, prompts, origin, policies = inputs
    root, merged = tmp_path / "D1", tmp_path / "D1/merged"
    measurement.bridge(
        runtime, policies, prompts, out=tmp_path / "serial", stream_id="fixed", origin_policy=origin
    )
    kwargs = dict(root=root, out=merged, workers=2, stream_id="fixed")
    with pytest.raises(ValueError, match="incomplete"):
        merge_workers(runtime["decision_private"], prompts, **kwargs)
    assert not merged.exists()
    run_worker(
        runtime, prompts, out=root / "workers/0", worker_index=0, workers=2, stream_id="fixed"
    )
    with pytest.raises(ValueError, match="incomplete"):
        merge_workers(runtime["decision_private"], prompts, **kwargs)
    assert not merged.exists()
    worker1 = run_worker(
        runtime, prompts, out=root / "workers/1", worker_index=1, workers=2, stream_id="fixed"
    )
    assert worker1["scoring_paths"]["tested_sequences"] == 0
    result = merge_workers(runtime["decision_private"], prompts, **kwargs)
    assert result["total_rows"] == len(prompts) * 32 * 4
    assert result["scoring_paths"]["tested_sequences"] == 24
    assert all(len(x["units"]) == len(prompts) for x in result["proposal_comparisons"])
    for index in range(2):
        serial_identity, serial = _load_stream(tmp_path / f"serial/endpoint_{index}")
        parallel_identity, parallel = _load_stream(merged / f"endpoint_{index}")
        assert serial_identity == parallel_identity
        assert digest(serial) == digest(parallel)
        assert (
            merged / f"endpoint_{index}/samples/{prompts[0]['prompt_id']}/0000_0032.json"
        ).is_symlink()
    before = (merged / "COMPLETE.json").read_bytes()
    merge_workers(runtime["decision_private"], prompts, **kwargs)
    assert (merged / "COMPLETE.json").read_bytes() == before
    for proposal in ("origin", "mix"):
        for prompt in prompts:
            relative = Path(proposal) / prompt["prompt_id"] / "0000_0032.json"
            assert (tmp_path / "serial" / relative).read_bytes() == (merged / relative).read_bytes()


def test_merge_rejects_changed_worker_evidence(inputs, tmp_path):
    runtime, prompts, _, _ = inputs
    root = tmp_path / "D1"
    for index in range(2):
        run_worker(
            runtime,
            prompts,
            out=root / f"workers/{index}",
            worker_index=index,
            workers=2,
            stream_id="fixed",
        )
    path = root / "workers/1/endpoint_0/LOOK_32.json"
    path.write_text(path.read_text().replace('"total_rows": 64', '"total_rows": 63'))
    # Change even valid JSON bytes: physical receipt hashes are bound at sealing.
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        merge_workers(
            runtime["decision_private"],
            prompts,
            root=root,
            out=root / "merged",
            workers=2,
            stream_id="fixed",
        )
    assert not (root / "merged").exists()


def test_completed_72_prompt_global_budget(inputs, tmp_path):
    runtime, base, _, _ = inputs
    prompts = [
        {
            **base[0],
            "prompt_id": f"{family}_{scene}_{interface}",
            "family": family,
            "base_scene_id": f"{family}_{scene}",
            "interface": interface,
        }
        for family in ("a", "b", "c")
        for scene in range(12)
        for interface in ("choice", "text")
    ]
    root = tmp_path / "D1"
    for index in range(2):
        result = run_worker(
            runtime,
            prompts,
            out=root / f"workers/{index}",
            worker_index=index,
            workers=2,
            stream_id="full-panel-fixture",
        )
        assert sum(e["total_rows"] for e in result["endpoints"]) == 2304
    merged = merge_workers(
        runtime["decision_private"],
        prompts,
        root=root,
        out=root / "merged",
        workers=2,
        stream_id="full-panel-fixture",
    )
    assert merged["total_rows"] == merged["planned_total_rows"] == 9216
    assert runtime["adapter"].generation_calls == 9216
    assert all(len(c["units"]) == 72 for c in merged["proposal_comparisons"])
    assert all(e["total_rows"] == 2304 for e in merged["endpoints"])


def test_namespace_ignores_only_process_load_metrics(inputs):
    runtime, _, _, _ = inputs
    runtime["adapter"].audit.update(
        {
            "load_seconds": 1.5,
            "load_peak_cuda_bytes": 100,
            "processor_hash": "processor-a",
        }
    )
    first = measurement._namespace(runtime)
    runtime["adapter"].audit.update({"load_seconds": 99, "load_peak_cuda_bytes": 1000})
    assert measurement._namespace(runtime) == first
    assert "load_seconds" in runtime["adapter"].audit
    runtime["adapter"].audit["processor_hash"] = "processor-b"
    assert measurement._namespace(runtime) != first
