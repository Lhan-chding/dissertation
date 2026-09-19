import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.decision_modeling.block_runner import _evaluate_isolated, run_block
from src.decision_modeling.reward_recipes import compute_advantages, reward_vector, saw_weights
from src.modeling_v4 import gpu_collect as gpu
from src.optimizer_fork import state_hash


def values():
    return np.array(
        [
            [[1, 1, 1, 1], [0, 1, 1, 0.5], [0, 0, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0.2], [0, 0, 1, 0.7], [1, 1, 1, 1], [0, 0, 0, 0.0]],
        ]
    )


def test_static_total_reward_normalized_after_sum_and_relation_changes_advantage():
    v = values()
    result = compute_advantages(v, "R4")
    total = 2 * v[:, :, 0] + 0.5 * v[:, :, 2] + 0.5 * v[:, :, 3]
    expected = (total - total.mean(axis=1, keepdims=True)) / np.sqrt(
        total.var(axis=1, keepdims=True) + 1e-8
    )
    np.testing.assert_allclose(result["advantages"], expected)
    assert result["advantages"][1][0] != result["advantages"][1][1]
    assert reward_vector({"event": "S", "relation_score": 0.25}) == [0, 1, 1, 0.25]
    with pytest.raises(ValueError):
        reward_vector({"event": "I", "relation_score": 0.1})


def test_gdpo_matches_official_torch_two_stage_example():
    v = torch.tensor(values(), dtype=torch.float64)
    by_reward = (v - v.mean(dim=1, keepdim=True)) / (v.std(dim=1, keepdim=True) + 1e-4)
    combined = by_reward @ torch.tensor([2, 0, 0.5, 0.5], dtype=torch.float64)
    expected = (combined - combined.mean()) / (combined.std() + 1e-4)
    result = compute_advantages(v.numpy(), "GDPO_R4")
    np.testing.assert_allclose(result["advantages"], expected.numpy(), atol=1e-12)
    assert result["batch_std"] > 0
    assert np.count_nonzero(compute_advantages(np.zeros((2, 8, 4)), "GDPO_R4")["advantages"]) == 0


def test_saw_uses_batch_cv_theoretical_offset_and_post_cv_priorities():
    v = values()
    flat = v.reshape(-1, 4)
    cv = flat.std(axis=0, ddof=1) / (flat.mean(axis=0) + 2e-4)
    expected = cv / cv[[0, 2, 3]].sum() * np.array([2, 0, 0.5, 0.5])
    out = saw_weights(v)
    np.testing.assert_allclose(out["weights"], expected)
    # A has no priority and cannot take weight budget from X,V,C.
    assert out["weights"][1] == 0
    zero = saw_weights(np.zeros((2, 8, 4)))
    assert zero["fallback_equal_active_weights"]
    assert zero["weights"] == [2, 0, 0.5, 0.5]


def tiny():
    path = Path(__file__).parent / "followup_test_adapter.py"
    spec = importlib.util.spec_from_file_location("dm_test_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.make_fake_adapter(17)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=0.001
    )
    runtime = gpu.attach_runtime(adapter, optimizer, identity={"fixture": "tiny"})
    runtime["sampler"].update(position=0)
    state = runtime["state"].capture({"seed": 41001, "arm": "X_BASE", "checkpoint_step": 32})
    return runtime, state


class Sample:
    def __init__(self, runtime, fail=None):
        self.runtime, self.calls, self.fail = runtime, [], fail

    def __call__(self, step, ledger):
        self.calls.append(step)
        if step == self.fail:
            raise RuntimeError("injected sample failure")
        adapter = self.runtime["adapter"]
        # Fresh probabilities from the live policy, never a shared saved bank.
        groups = []
        for b in range(2):
            group = []
            prepared = {"offset": b}
            for k in range(4):
                tokens = [k, 4]
                old = adapter.logprobs(prepared, tokens, require_grad=False).tolist()
                row = {
                    "prompt_id": str(b),
                    "sample_key": f"{step}:{b}:{k}",
                    "category": ("XSWI"[k] if b == 0 else "WWXI"[k]),
                    "token_ids": tokens,
                    "old_logprobs": old,
                    "reward_vector": values()[b, k].tolist(),
                }
                ledger(row)
                group.append({**row, "prepared": prepared})
            groups.append(group)
        self.runtime["sampler"]["position"] += 2
        random.random()
        torch.rand(2)
        return groups


def test_real_adam_cpu_branch_h8_resume_h32_preserves_all_and_no_base_hash(tmp_path, monkeypatch):
    runtime, origin = tiny()
    # Small CPU tensor operations dominate if torch creates huge thread pools.
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Whole model hashes forbidden within blocks")

    monkeypatch.setattr("src.followup_updates.parameter_hash", forbidden)
    monkeypatch.setattr("src.fork_gradients.parameter_hash", forbidden)
    sampler = Sample(runtime)
    out = run_block(
        runtime,
        origin,
        sampler,
        origin_id="O1",
        recipe="R4",
        steps=8,
        out=tmp_path / "a",
        fixture=True,
    )
    assert out["training_outputs"] == 64
    assert sampler.calls == list(range(1, 9))
    saved8 = runtime["checkpoint_cache"].load(out["checkpoint"])
    assert any(not torch.equal(p, saved8["parameters"][n]) for n, p in origin["parameters"].items())
    assert all(int(s["step"]) == 8 for s in saved8["optimizer"]["state"].values())
    continued = Sample(runtime)
    out32 = run_block(
        runtime,
        origin,
        continued,
        origin_id="O1",
        recipe="R4",
        steps=32,
        out=tmp_path / "a",
        fixture=True,
        resume=True,
    )
    assert continued.calls == list(range(9, 33))
    assert out32["completed_horizons"] == [0, 8, 32]
    state32 = runtime["checkpoint_cache"].load(out32["checkpoint"])
    # Independent fresh same-origin run must have exactly matching state.
    fresh = run_block(
        runtime,
        origin,
        Sample(runtime),
        origin_id="O1",
        recipe="R4",
        steps=32,
        out=tmp_path / "b",
        fixture=True,
    )
    assert state_hash(state32) == state_hash(runtime["checkpoint_cache"].load(fresh["checkpoint"]))
    prior = state_hash(runtime["state"].capture({}))

    def evaluate(rt, cp, step):
        random.random()
        np.random.rand()
        torch.rand(20)
        rt["sampler"]["position"] = -1
        return {"step": step}

    assert _evaluate_isolated(runtime, evaluate, None, 8) == {"step": 8}
    assert state_hash(runtime["state"].capture({})) == prior


def test_failure_preserves_rows_and_resumes_from_committed_state(tmp_path):
    runtime, origin = tiny()
    with pytest.raises(RuntimeError, match="injected"):
        run_block(
            runtime,
            origin,
            Sample(runtime, fail=3),
            origin_id="O1",
            recipe="R0",
            steps=8,
            out=tmp_path / "a",
            fixture=True,
        )
    failure = tmp_path / "a/steps/H03/attempt_0001/FAILURE.json"
    assert json.loads(failure.read_text())["last_committed_step"] == 2
    sample = Sample(runtime)
    result = run_block(
        runtime,
        origin,
        sample,
        origin_id="O1",
        recipe="R0",
        steps=8,
        out=tmp_path / "a",
        fixture=True,
        resume=True,
    )
    assert sample.calls == list(range(3, 9))
    assert failure.exists()
    assert result["steps"] == 8
    with pytest.raises(ValueError, match="Immutable"):
        run_block(
            runtime,
            origin,
            sample,
            origin_id="O1",
            recipe="R1",
            steps=8,
            out=tmp_path / "a",
            fixture=True,
            resume=True,
        )


def test_real_source_sampler_generates_live_policy_rewards_and_restores_cursor(
    tmp_path, monkeypatch
):
    from src.decision_modeling.block_runner import SourceGroupSampler

    path = Path(__file__).parent / "followup_test_adapter.py"
    spec = importlib.util.spec_from_file_location("dm_sampler_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime, _ = tiny()
    adapter = runtime["adapter"]
    adapter.pad_id = 4
    adapter.audit["probability_execution"] = "uncached_prefix_recompute"
    runtime["parity_tolerances"] = {
        "mean_abs_token_logp": 1e-5,
        "max_abs_token_logp": 1e-5,
        "max_abs_sequence_logp": 1e-4,
    }
    prompts = module.fake_prompts(4, split="train")
    schedule = {
        "train_steps": [[p["prompt_id"] for p in prompts]] * 128,
        "schedule_hash": "tiny-schedule",
    }
    monkeypatch.setattr(
        "src.modeling_v4.data_adapter.build_training_schedule", lambda prompts, seed: schedule
    )
    runtime["sampler"].update(position=128, step=32, schedule_hash="tiny-schedule")
    origin = runtime["state"].capture({"seed": 41001, "checkpoint_step": 32, "arm": "X_BASE"})
    sampler = SourceGroupSampler(runtime, prompts, origin, origin_id="O1", recipe="R4")
    out = run_block(
        runtime,
        origin,
        sampler,
        origin_id="O1",
        recipe="R4",
        steps=8,
        out=tmp_path / "live",
        fixture=True,
    )
    assert out["training_outputs"] == 256
    assert adapter.generation_calls == 256
    assert runtime["sampler"]["position"] == 160
    first = json.loads(
        (tmp_path / "live/steps/H01/attempt_0001/samples.jsonl").read_text().splitlines()[0]
    )
    assert first["reward_vector"] == reward_vector(first["semantic"])
    # Missing endpoint receipt after a successful step can be recovered with no samples.
    (tmp_path / "live/H08.json").unlink()
    run_block(
        runtime,
        origin,
        sampler,
        origin_id="O1",
        recipe="R4",
        steps=8,
        out=tmp_path / "live",
        fixture=True,
        resume=True,
    )
    assert adapter.generation_calls == 256
    assert (tmp_path / "live/H08.json").exists()


def test_evaluation_failure_after_h8_commit_retries_without_training(tmp_path):
    runtime, origin = tiny()

    def fail_eval(rt, checkpoint, step):
        torch.rand(10)
        raise RuntimeError("injected evaluation failure")

    with pytest.raises(RuntimeError, match="evaluation failure"):
        run_block(
            runtime,
            origin,
            Sample(runtime),
            origin_id="O1",
            recipe="R0",
            steps=8,
            out=tmp_path / "a",
            fixture=True,
            evaluate=fail_eval,
        )
    assert list((tmp_path / "a").glob("EVAL_H08_FAILURE_*.json"))
    sampler = Sample(runtime)

    def good_eval(rt, checkpoint, step):
        assert all(int(s["step"]) == 8 for s in rt["optimizer"].state.values())
        return {"status": "FIXTURE_MEASURED", "step": step}

    run_block(
        runtime,
        origin,
        sampler,
        origin_id="O1",
        recipe="R0",
        steps=8,
        out=tmp_path / "a",
        fixture=True,
        evaluate=good_eval,
        resume=True,
    )
    assert sampler.calls == []
    assert json.loads((tmp_path / "a/EVAL_H08.json").read_text())["step"] == 8


def test_cli_writer_and_dry_run_can_precede_first_block_and_config_is_bound(tmp_path):
    from src.core import frozen_writer

    runtime, origin = tiny()
    runtime["decision_config_hash"] = "frozen-protocol-v1"
    out = tmp_path / "block"
    out.mkdir()
    dry_run = out / "DRY_RUN_block.json"
    dry_run.write_text('{"status":"DRY_RUN_NO_MODEL_LOAD"}\n')
    with frozen_writer(out):
        result = run_block(
            runtime,
            origin,
            Sample(runtime),
            origin_id="O1",
            recipe="R0",
            steps=8,
            out=out,
            fixture=True,
        )
    assert result["steps"] == 8
    assert dry_run.read_text() == '{"status":"DRY_RUN_NO_MODEL_LOAD"}\n'
    assert (
        json.loads((out / "MANIFEST.json").read_text())["decision_config_hash"]
        == "frozen-protocol-v1"
    )
    with frozen_writer(out), pytest.raises(FileExistsError, match="nonempty"):
        run_block(
            runtime,
            origin,
            Sample(runtime),
            origin_id="O1",
            recipe="R0",
            steps=8,
            out=out,
            fixture=True,
        )
    runtime["decision_config_hash"] = "changed-protocol"
    with frozen_writer(out), pytest.raises(ValueError, match="Immutable"):
        run_block(
            runtime,
            origin,
            Sample(runtime),
            origin_id="O1",
            recipe="R0",
            steps=8,
            out=out,
            fixture=True,
            resume=True,
        )
