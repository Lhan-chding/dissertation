"""Real tiny Torch/Adam production wiring; these are explicitly CPU fixtures."""

import importlib.util
from pathlib import Path

import pytest
import torch

from src.decision_modeling.block_runner import update_batch
from src.decision_modeling.reward_recipes import compute_advantages
from src.modeling_v3.io import canonical_hash
from src.optimizer_fork import state_hash
from src.prospective_selection.branch_runtime import (
    CommonScheduleSampler,
    recipe_advantages,
    run_branch,
    run_training,
)
from src.prospective_selection.evaluation import collect_evaluation, isolated_evaluation
from src.prospective_selection.origin_runtime import run_smoke


@pytest.fixture
def tiny():
    path = Path(__file__).parents[1] / "modeling_v4/test_gpu_collect.py"
    spec = importlib.util.spec_from_file_location("prospective_tiny_helper", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runtime, module = helper.tiny_runtime()
    runtime["initial_state"] = runtime["state"].capture({"lora_initialization_seed": 17})
    runtime["prospective_config_hash"] = "fixture"
    return runtime, module


def schedule(prompts):
    result = {
        "train_steps": [
            [p["prompt_id"] for p in prompts[i : i + 4]] for i in range(0, len(prompts), 4)
        ],
        "role": "source",
    }
    result["schedule_hash"] = canonical_hash(result)
    result["schedule_id"] = result["schedule_hash"][:16]
    return result


def sampler_and_state(runtime, prompts):
    sched = schedule(prompts)
    runtime["sampler"].update(position=0, step=0, schedule_hash=sched["schedule_hash"])
    state = runtime["state"].capture({"checkpoint_step": 0})
    sampler = CommonScheduleSampler(
        runtime,
        prompts,
        sched,
        lineage_id=61001,
        repeat=1,
        experiment_seed=9,
        origin_id="O",
        recipe="R4",
    )
    sampler.state_identity = "initial"
    return sampler, state


def test_inherited_advantage_loss_and_adam_exact_parity(tiny):
    runtime, module = tiny
    sampler, state = sampler_and_state(runtime, module.fake_prompts(4, split="train"))
    groups = sampler(1, lambda _: None)
    for recipe in ("R0", "R4", "GDPO_R4", "SAW_R4"):
        runtime["state"].restore(state)
        old = update_batch(runtime, groups, recipe)
        old_state = runtime["state"].capture()
        runtime["state"].restore(state)
        new = update_batch(runtime, groups, recipe, advantage_callback=recipe_advantages)
        assert old["reward_statistics"] == new["reward_statistics"]
        assert old["loss"] == new["loss"]
        assert state_hash(old_state) == state_hash(runtime["state"].capture())


def test_direct_repair_five_channels_and_invalid_penalty(tiny):
    rewards = [[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]
    groups = [[{"semantic": {"Bdamage": b}} for b in (1, 0, 0, 1 / 3)]]
    result = recipe_advantages(rewards, groups, recipe="DIRECT_REPAIR_R4")
    assert result["total_rewards"][0] == [-0.5, 3.0, 1.0, 1 - 1 / 6]
    assert len(result["five_channel_rewards"][0][0]) == 5
    assert compute_advantages(rewards, "R4")["total_rewards"][0][0] == 0


def test_zero_advantage_still_advances_nonzero_adam_momentum(tiny):
    runtime, module = tiny
    sampler, _state = sampler_and_state(runtime, module.fake_prompts(4, split="train"))
    groups = sampler(1, lambda _: None)
    for p in runtime["adapter"].model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    runtime["optimizer"].step()
    runtime["optimizer"].zero_grad()
    before = {
        n: p.detach().clone()
        for n, p in runtime["adapter"].model.named_parameters()
        if p.requires_grad
    }

    def zeros(rewards, groups, **kw):
        return {"advantages": [[0] * len(g) for g in groups]}

    result = update_batch(runtime, groups, "R0", advantage_callback=zeros)
    assert result["optimizer_updates"] == 1
    assert all(int(s["step"]) == 2 for s in runtime["optimizer"].state_dict()["state"].values())
    assert any(
        not torch.equal(before[n], p)
        for n, p in runtime["adapter"].model.named_parameters()
        if p.requires_grad
    )


def test_evaluation_restores_full_state_even_when_failing(tiny):
    runtime, _ = tiny
    saved = runtime["state"].capture()

    def fail():
        runtime["sampler"]["changed"] = 1
        runtime["adapter"].model.eval()
        with torch.no_grad():
            runtime["adapter"].model.scale.add_(2)
        torch.rand(4)
        raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        isolated_evaluation(runtime, fail)
    assert state_hash(saved) == state_hash(runtime["state"].capture())


def test_smoke_and_independent_eval_idempotence(tiny, tmp_path):
    runtime, module = tiny
    prompts = module.fake_prompts(8, split="train")
    result = run_smoke(
        runtime, prompts=prompts, schedule=schedule(prompts), out=tmp_path / "smoke", fixture=True
    )
    assert result["optimizer_updates"] == 4
    assert result["status"] == "CPU_FIXTURE_COMPLETE"
    assert result["behavior_score_parity"]
    smoke_calls = runtime["adapter"].generation_calls
    repeated = run_smoke(
        runtime, prompts=prompts, schedule=schedule(prompts), out=tmp_path / "smoke", fixture=True
    )
    assert repeated == result
    assert runtime["adapter"].generation_calls == smoke_calls
    checkpoint = result["recipes"]["R0"]["checkpoint"]
    pre = state_hash(runtime["state"].capture())
    kwargs = dict(
        out=tmp_path / "eval",
        lineage_id=61001,
        origin_id="O",
        policy_id="R0",
        panel_id="D",
        horizon=2,
        draws=4,
        fixture=True,
    )
    summary, rows = collect_evaluation(runtime, checkpoint, module.fake_prompts(2), **kwargs)
    assert state_hash(runtime["state"].capture()) == pre
    calls = runtime["adapter"].generation_calls
    again, rows2 = collect_evaluation(
        runtime, checkpoint, module.fake_prompts(2), resume=True, **kwargs
    )
    assert rows2 == rows and again == summary
    assert calls == runtime["adapter"].generation_calls


def test_failed_segment_replays_only_uncommitted_updates(tiny, tmp_path):
    runtime, module = tiny
    prompts = module.fake_prompts(48, split="train")
    sampler, state = sampler_and_state(runtime, prompts)
    runtime["adapter"].fail_after = 9 * 32 + 3
    kwargs = dict(
        recipe="R4",
        steps=12,
        out=tmp_path / "run",
        identity={"fixture": True},
        checkpoints=(0, 8, 12),
        fixture=True,
    )
    with pytest.raises(RuntimeError, match="injected"):
        run_training(runtime, state, sampler, **kwargs)
    assert len(list((tmp_path / "run/segments").glob("*/COMMIT.json"))) == 1
    runtime["adapter"].fail_after = None
    result = run_training(runtime, state, sampler, resume=True, **kwargs)
    assert result["steps"] == 12 and result["training_outputs"] == 12 * 32
    actual = runtime["checkpoint_cache"].load(result["checkpoint"])
    runtime["state"].restore(state)
    sampler2, _ = sampler_and_state(runtime, prompts)
    clean = run_training(runtime, state, sampler2, **{**kwargs, "out": tmp_path / "clean"})
    expected = runtime["checkpoint_cache"].load(clean["checkpoint"])
    assert state_hash(actual) == state_hash(expected)
    calls = runtime["adapter"].generation_calls
    run_training(runtime, state, sampler, resume=True, **kwargs)
    assert runtime["adapter"].generation_calls == calls
    assert len(list((tmp_path / "run/segments").glob("*/attempt_*/FAILURE.json"))) == 1


def test_common_seed_and_schedule_ignore_recipe_and_origin(tiny):
    runtime, module = tiny
    prompts = module.fake_prompts(4, split="train")
    sampler, state = sampler_and_state(runtime, prompts)
    first = sampler(1, lambda _: None)
    runtime["state"].restore(state)
    other = CommonScheduleSampler(
        runtime,
        prompts,
        schedule(prompts),
        lineage_id=61001,
        repeat=1,
        experiment_seed=9,
        origin_id="OTHER",
        recipe="R7",
    )
    other.state_identity = "initial"
    second = other(1, lambda _: None)
    assert [[r["sample_seed"] for r in g] for g in first] == [
        [r["sample_seed"] for r in g] for g in second
    ]
    assert [[r["token_ids"] for r in g] for g in first] == [
        [r["token_ids"] for r in g] for g in second
    ]


def test_test_branch_requires_registry_receipt(tiny, tmp_path):
    runtime, _module = tiny
    with pytest.raises(PermissionError, match="registry"):
        run_branch(
            runtime,
            {},
            prompts=[],
            schedule={},
            lineage_id=63001,
            origin_id="O",
            recipe="R0",
            repeat=1,
            out=tmp_path,
            role="locked_test",
            fixture=True,
        )


def test_h8_diagnostic_cannot_change_later_training(tiny, tmp_path):
    runtime, module = tiny
    prompts = module.fake_prompts(48, split="train")
    sampler, state = sampler_and_state(runtime, prompts)
    kwargs = dict(
        recipe="R4", steps=12, identity={"fixture": True}, checkpoints=(0, 8, 12), fixture=True
    )
    clean = run_training(runtime, state, sampler, out=tmp_path / "clean", **kwargs)

    def diagnostic(rt, checkpoint, step):
        rt["adapter"].model.eval()
        rt["sampler"]["position"] = 999
        torch.rand(50)
        with torch.no_grad():
            rt["adapter"].model.scale.add_(4)
        return {"step": step}

    checked = run_training(
        runtime, state, sampler, out=tmp_path / "diagnostic", evaluate=diagnostic, **kwargs
    )
    assert state_hash(runtime["checkpoint_cache"].load(clean["checkpoint"])) == state_hash(
        runtime["checkpoint_cache"].load(checked["checkpoint"])
    )


def test_production_symbolic_interface_smoke(tiny, tmp_path):
    runtime, module = tiny
    prompts = module.fake_prompts(8, split="train")
    for prompt in prompts:
        if prompt["interface"] == "SYM_CUE":
            prompt["interface"] = "SYMBOLIC_FRESH"
    result = run_smoke(
        runtime, prompts=prompts, schedule=schedule(prompts), out=tmp_path / "smoke", fixture=True
    )
    assert result["optimizer_updates"] == 4
