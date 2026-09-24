"""CPU fake-policy integration through the production task dispatcher and prestate path."""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from src.modeling_v3.io import canonical_hash
from src.prospective_selection import evaluation, origin_runtime
from src.prospective_selection import runtime as runtime_module
from src.prospective_selection.features import LEVELS, PreDecisionPacket
from src.prospective_selection.jobs import TaskRegistry
from src.prospective_selection.orchestration import _metadata, execute_task, observe_state
from src.prospective_selection.protocol import load_protocol


def tiny_runtime():
    path = Path(__file__).parents[1] / "modeling_v4/test_gpu_collect.py"
    spec = importlib.util.spec_from_file_location("orchestration_tiny_gpu", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runtime, module = helper.tiny_runtime()
    runtime["initial_state"] = runtime["state"].capture({"lora_seed": 17})
    runtime["prospective_config_hash"] = canonical_hash(load_protocol())
    return runtime, module


def schedule(prompts):
    value = {
        "train_steps": [
            [p["prompt_id"] for p in prompts[i : i + 4]] for i in range(0, len(prompts), 4)
        ]
    }
    value["schedule_hash"] = canonical_hash(value)
    value["schedule_id"] = "fixture-source"
    return value


def test_observe_state_reads_complete_adam_checkpoints_and_eight_nested_updates(
    tmp_path, monkeypatch
):
    run, module = tiny_runtime()
    train = module.fake_prompts(128, split="train")
    probe = module.fake_prompts(72, split="control")
    run["prepared_data"] = {"panels": {"P": probe}}
    source = tmp_path / "sources/61001"
    origin_runtime.run_source(
        run,
        prompts=train,
        schedule=schedule(train),
        lineage_id=61001,
        source_recipe="R0",
        steps=32,
        out=source,
        fixture=True,
    )
    audits = []
    for path in source.glob("segments/*/COMMIT.json"):
        commit = json.loads(path.read_text())
        for binding in commit["updates"]:
            audit = json.loads(Path(binding["path"]).read_text())
            if 24 < audit["step"] <= 32:
                audits.append(audit)
    assert len(audits) == 8
    metadata = _metadata(source, 32)
    assert metadata["gradient_norm_mean"] == statistics.mean(a["grad_norm_preclip"] for a in audits)
    groups = [g for a in audits for g in a["reward_statistics"]["advantages"]]
    assert metadata["effective_group_fraction"] == sum(any(g) for g in groups) / len(groups)
    # Incomplete attempt has no COMMIT: its extreme value must not enter metadata.
    pending = source / "segments/uncommitted/attempt_1"
    pending.mkdir(parents=True)
    (pending / "UPDATE_032.json").write_text(json.dumps({"step": 32, "grad_norm_preclip": 99999}))
    assert _metadata(source, 32) == metadata

    real_collect = evaluation.collect_evaluation

    def fixture_collect(*args, **kwargs):
        return real_collect(*args, **kwargs, fixture=True)

    monkeypatch.setattr(evaluation, "collect_evaluation", fixture_collect)
    result = observe_state(
        run,
        load_protocol(),
        tmp_path,
        {"lineage_id": 61001, "origin_step": 32, "origin_id": "61001_t32", "source_recipe": "R0"},
    )
    assert result["shared_generated_outputs"] == 4608
    packets = [
        PreDecisionPacket.from_dict(
            json.loads((tmp_path / "prestate/61001_t32" / f"{level}.json").read_text())
        )
        for level in LEVELS
    ]
    for packet in packets:
        obj = packet.to_dict()
        assert obj["current"]["n"] == obj["history"]["n"] == 2304
        assert obj["known_training_metadata"] == metadata
        assert obj["step"] == 32 and obj["history_step"] == 24
    assert "repair_histograms" not in packets[2].to_dict()["current"]
    assert "repair_histograms" in packets[3].to_dict()["current"]
    calls = run["adapter"].generation_calls
    observe_state(
        run,
        load_protocol(),
        tmp_path,
        {"lineage_id": 61001, "origin_step": 32, "origin_id": "61001_t32", "source_recipe": "R0"},
    )
    assert run["adapter"].generation_calls == calls


def test_smoke_dispatch_uses_bound_runtime_and_marks_one_completion(tmp_path, monkeypatch):
    run, module = tiny_runtime()
    prompts = module.fake_prompts(8, split="train")
    for prompt in prompts:
        if prompt["interface"] == "SYM_CUE":
            prompt["interface"] = "SYMBOLIC_FRESH"
    run["prepared_data"] = {"source_prompts": prompts, "source_schedule": schedule(prompts)}
    loads = []

    def fake_loader(config, path, *, allow_gpu):
        assert allow_gpu is True
        loads.append(str(path))
        return run

    monkeypatch.setattr(runtime_module, "load_runtime", fake_loader)
    real_smoke = origin_runtime.run_smoke

    def fixture_smoke(*args, **kwargs):
        return real_smoke(*args, **kwargs, fixture=True)

    monkeypatch.setattr(origin_runtime, "run_smoke", fixture_smoke)
    config = load_protocol()
    registry = TaskRegistry(tmp_path, config)
    task = registry.register_task("smoke", {"smoke_id": "two-recipes-two-updates"})
    result = execute_task(config, tmp_path, task["task_id"], "private-fixture.json", allow_gpu=True)
    assert result["status"] == "CPU_FIXTURE_COMPLETE"
    assert result["optimizer_updates"] == 4
    assert loads == ["private-fixture.json"]
    completion = json.loads((tmp_path / "completed" / f"{task['task_id']}.json").read_text())
    assert completion["receipt"]["result"] == result
    with pytest.raises(ValueError, match="already complete"):
        execute_task(config, tmp_path, task["task_id"], "private-fixture.json", allow_gpu=True)
    assert len(loads) == 1
