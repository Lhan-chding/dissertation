"""CPU mechanism contracts: real autograd/Adam, no downloaded model."""

import copy
import importlib
import math
import random

import numpy as np
import pytest
import torch

from src import fork_gradients as legacy
from src.grpo_update import grouped_advantages, torch_ppo_loss
from src.optimizer_fork import state_hash


@pytest.fixture
def api():
    return importlib.import_module("src.followup_updates")


class TinyAdapter:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.a = torch.nn.Linear(2, 4, bias=True)
        with torch.no_grad():
            self.model.a.weight.copy_(torch.arange(8).reshape(4, 2) / 20)
            self.model.a.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        self.model.register_parameter(
            "frozen", torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
        )
        self.model.register_buffer("counter", torch.tensor(0))
        self.model.register_buffer("optional", None, persistent=False)
        self.model.rope_deltas = torch.tensor([2])
        self.model.a.eval()
        self.reset_calls = 0

    def _reset_positions(self):
        self.model.rope_deltas = None
        self.reset_calls += 1

    def logprobs(self, prepared, completion, *, require_grad=False):
        self.model.train(require_grad)
        self.model.counter.add_(1)
        self.model.rope_deltas = torch.tensor([len(completion)])
        with torch.set_grad_enabled(require_grad):
            return torch.stack(
                [
                    (self.model.a(prepared["features"] + i * 0.1) + self.model.frozen).log_softmax(
                        -1
                    )[t]
                    for i, t in enumerate(completion)
                ]
            )


def groups(adapter, category_groups=(("X", "S", "W", "I"), ("W", "I", "W", "I"))):
    result = []
    for p, categories in enumerate(category_groups):
        prepared = {"features": torch.tensor([1.0 + p, -0.3 - p])}
        group = []
        for i, category in enumerate(categories):
            tokens = [[0, 3], [1], [2, 1, 3], [3, 0, 2, 1]][i % 4]
            group.append(
                {
                    "prompt_id": f"p{p}",
                    "sample_key": f"p{p}s{i}",
                    "category": category,
                    "token_ids": tokens,
                    "prepared": prepared,
                    "old_logprobs": adapter.logprobs(prepared, tokens).tolist(),
                }
            )
        result.append(group)
    adapter.model.train()
    adapter.model.a.eval()
    return result


def optimizer(adapter):
    return torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )


def stats(api, categories, lam=1, policy="joint"):
    return api.group_reward_statistics(
        categories, auxiliary_weight=lam, policy=policy, epsilon=1e-4
    )


def test_reward_channels_reference_and_constants(api):
    for lam, expected in [(0, [2, 0, 0, 0]), (1, [3, 1, 1, 0])]:
        result = stats(api, ["X", "S", "W", "I"], lam)
        assert result["reward_vector"] == expected
        for key, value in grouped_advantages(expected).items():
            assert result[key] == value
    for category in ["X", "W", "I"]:
        constant = stats(api, [category] * 8)
        assert (
            constant["advantages"] == [0.0] * 8
            and constant["variance"] == 0
            and math.isfinite(constant["mean"])
        )
    reference = stats(api, ["X", "S", "W", "X"], 0)["advantages"]
    for lam in [1e-6, 0.01, 1, 10]:
        assert stats(api, ["X", "S", "W", "X"], lam)["advantages"] == pytest.approx(
            reference, abs=1e-14
        )


@pytest.mark.parametrize("category", ["S", "W"])
def test_no_x_gate_small_lambda_and_present_x(api, category):
    categories = [category] * 4 + ["I"] * 4
    for lam in [0, 1e-6, 1e-5, 1e-4, 2e-4, 0.001, 0.01, 1]:
        expected = 0.5 * lam / math.sqrt(0.25 * lam * lam + 1e-8)
        assert stats(api, categories, lam)["advantages"] == pytest.approx(
            [expected] * 4 + [-expected] * 4
        )
        suppressed = stats(api, categories, lam, "no_x_off")
        assert (
            suppressed["effective_auxiliary_weight"] == 0 and suppressed["advantages"] == [0.0] * 8
        )
        assert len(suppressed["reward_vector"]) == 8
    categories[0] = "X"
    assert (
        stats(api, categories, 1, "no_x_off")["advantages"]
        == stats(api, categories, 1)["advantages"]
    )


def test_permutation_and_derivative(api):
    categories = ["X", "S", "W", "I"]
    lam, h = 0.25, 1e-6
    center = stats(api, categories, lam)
    assert stats(api, categories[::-1], lam)["advantages"] == center["advantages"][::-1]
    r = np.array(center["reward_vector"])
    v = np.array([1.0, 1.0, 1.0, 0.0])
    c, dv = r - r.mean(), v - v.mean()
    denom = math.sqrt(np.mean(c * c) + 1e-8)
    analytic = dv / denom - c * np.mean(c * dv) / denom**3
    finite = (
        np.array(stats(api, categories, lam + h)["advantages"])
        - np.array(stats(api, categories, lam - h)["advantages"])
    ) / (2 * h)
    np.testing.assert_allclose(finite, analytic, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize(
    "categories,lam,policy,epsilon",
    [
        ([], 1, "joint", 1e-4),
        (["Z"], 1, "joint", 1e-4),
        (["X"], True, "joint", 1e-4),
        (["X"], float("nan"), "joint", 1e-4),
        (["X"], float("inf"), "joint", 1e-4),
        (["X"], -1, "joint", 1e-4),
        (["X"], 1, "unknown", 1e-4),
        (["X"], 1, "joint", True),
        (["X"], 1, "joint", 0),
    ],
)
def test_bad_rewards_fail(api, categories, lam, policy, epsilon):
    with pytest.raises(ValueError):
        api.group_reward_statistics(
            categories, auxiliary_weight=lam, policy=policy, epsilon=epsilon
        )


def test_new_joint_matches_exact_legacy_fp32_and_restores_modes(api):
    adapter = TinyAdapter()
    bank = groups(adapter)
    opt = optimizer(adapter)
    before = api.capture_state(adapter.model, opt, adapter=adapter)
    new = api.direct_loss_gradients(adapter, bank, auxiliary_weight=0.25)
    assert state_hash(api.capture_state(adapter.model, opt, adapter=adapter)) == state_hash(before)
    old = legacy.direct_loss_gradients(adapter, bank, auxiliary_weight=0.25)
    for name, value in new["gradients"].items():
        assert torch.equal(value, old["gradients"][name])
    assert new["audit"]["total_tokens"] == sum(len(r["token_ids"]) for g in bank for r in g)
    assert (
        new["audit"]["backward_calls"] == 8 and new["audit"]["old_probabilities_rewritten"] is False
    )


def test_microbatch_full_batch_clipping_and_no_x_denominator(api):
    adapter = TinyAdapter()
    bank = groups(adapter)
    for g in bank:
        for i, row in enumerate(g):
            row["old_logprobs"] = [
                v - math.log([0.7, 0.8, 1.2, 1.3][i]) for v in row["old_logprobs"]
            ]
    digest = state_hash(bank)
    actual = api.direct_loss_gradients(adapter, bank, policy="no_x_off", auxiliary_weight=1)
    adapter.model.zero_grad(set_to_none=True)
    scores = []
    olds = []
    advantages = []
    for g in bank:
        av = stats(api, [r["category"] for r in g], 1, "no_x_off")["advantages"]
        for row, a in zip(g, av, strict=True):
            score = adapter.logprobs(row["prepared"], row["token_ids"], require_grad=True)
            scores.append(torch.nn.functional.pad(score, (0, 4 - len(score))))
            olds.append(row["old_logprobs"] + [0.0] * (4 - len(score)))
            advantages.append(a)
    mask = torch.tensor([[i < len(r["token_ids"]) for i in range(4)] for g in bank for r in g])
    loss, _ = torch_ppo_loss(
        torch.stack(scores), torch.tensor(olds), torch.tensor(advantages), mask, lnorm=64
    )
    loss.backward()
    for name, p in adapter.model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(actual["gradients"][name], p.grad, atol=1e-8, rtol=2e-6)
    assert (
        state_hash(bank) == digest
        and actual["audit"]["group_statistics"][1]["advantages"] == [0.0] * 4
    )
    assert actual["audit"]["sequences"] == 8


@pytest.mark.parametrize("ratio", [0.7, 0.8, 1.0, 1.2, 1.3])
@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_ppo_boundary_branches_fp32(api, ratio, advantage):
    new = torch.tensor([[math.log(ratio)]], requires_grad=True)
    loss, _ = torch_ppo_loss(
        new,
        torch.zeros_like(new),
        torch.tensor([advantage]),
        torch.ones_like(new, dtype=torch.bool),
    )
    loss.backward()
    actual_ratio = float(new.detach().exp())
    active = (advantage > 0 and actual_ratio > float(torch.tensor(1.2))) or (
        advantage < 0 and actual_ratio < float(torch.tensor(0.8))
    )
    assert float(new.grad) == pytest.approx(
        0 if active else -advantage * actual_ratio / 64, abs=1e-8
    )


def test_warm_explicit_zero_adam_and_global_clip(api):
    adapter = TinyAdapter()
    opt = optimizer(adapter)
    grads = {
        n: torch.full_like(p, 10.0) for n, p in adapter.model.named_parameters() if p.requires_grad
    }
    result = api.apply_gradient_update(adapter.model, opt, grads)
    assert result["grad_norm_preclip"] == pytest.approx(math.sqrt(12) * 10)
    assert result["grad_norm_postclip"] <= 1.000001
    origin = api.capture_state(adapter.model, opt, adapter=adapter)
    zeros = {n: torch.zeros_like(p) for n, p in adapter.model.named_parameters() if p.requires_grad}
    stepped = api.apply_gradient_update(adapter.model, opt, zeros)
    assert stepped["actual_step_norm"] > 0 and stepped["optimizer_updates"] == 1
    assert all(float(s["step"]) == 2 for s in opt.state.values())
    api.restore_state(adapter.model, opt, origin, adapter=adapter)
    opt.zero_grad(set_to_none=True)
    opt.step()
    assert state_hash(api.capture_state(adapter.model, opt, adapter=adapter)) == state_hash(origin)


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "frozen", "nan_gradient", "nan_parameter", "nan_state"]
)
def test_reject_invalid_update_before_mutation(api, case):
    adapter = TinyAdapter()
    opt = optimizer(adapter)
    grads = {n: torch.ones_like(p) for n, p in adapter.model.named_parameters() if p.requires_grad}
    api.apply_gradient_update(adapter.model, opt, grads)
    if case == "missing":
        opt.param_groups[0]["params"].pop()
    elif case == "duplicate":
        opt.param_groups[0]["params"].append(opt.param_groups[0]["params"][0])
    elif case == "frozen":
        opt.param_groups[0]["params"].append(adapter.model.frozen)
    elif case == "nan_gradient":
        next(iter(grads.values())).fill_(float("nan"))
    elif case == "nan_parameter":
        adapter.model.a.weight.data.fill_(float("nan"))
    elif case == "nan_state":
        next(iter(opt.state.values()))["exp_avg"].fill_(float("inf"))
    before = state_hash(adapter.model.state_dict())
    with pytest.raises((ValueError, FloatingPointError)):
        api.apply_gradient_update(adapter.model, opt, grads)
    assert state_hash(adapter.model.state_dict()) == before


def test_full_restore_checkpoint_and_candidate_order(api, tmp_path):
    adapter = TinyAdapter()
    bank = groups(adapter)
    opt = optimizer(adapter)
    sampler = {"index": 5, "seed": 17}
    # Real warm Adam history reaches the declared origin step, without a model download.
    warm_gradients = api.direct_loss_gradients(adapter, bank)["gradients"]
    for _ in range(64):
        api.apply_gradient_update(adapter.model, opt, warm_gradients)
    origin = api.capture_state(
        adapter.model, opt, {"checkpoint_step": 64}, adapter=adapter, sampler=sampler
    )
    expected = state_hash(origin)
    random_expected = (random.random(), np.random.random(), torch.rand(2))
    api.restore_state(adapter.model, opt, origin, adapter=adapter, sampler=sampler)
    assert (
        random.random() == random_expected[0]
        and np.random.random() == random_expected[1]
        and torch.equal(torch.rand(2), random_expected[2])
    )
    api.restore_state(adapter.model, opt, origin, adapter=adapter, sampler=sampler)
    outcomes = []
    for order in [(0.0, 1.0), (1.0, 0.0)]:
        results = {}
        for lam in order:
            result = api.fork_one_candidate(
                adapter,
                opt,
                origin,
                bank,
                {"policy": "joint", "auxiliary_weight": lam},
                sampler=sampler,
            )
            results[lam] = state_hash(result["state"])
            assert result["audit"]["permanent_training_commit"] is False
            assert result["audit"]["origin_checkpoint_step"] == 64
            assert result["audit"]["origin_optimizer_steps"] == [64]
            assert result["audit"]["candidate_optimizer_step"] == 65
            assert (
                state_hash(
                    api.capture_state(
                        adapter.model, opt, origin["metadata"], adapter=adapter, sampler=sampler
                    )
                )
                == expected
            )
        outcomes.append(results)
    assert outcomes[0] == outcomes[1]
    adapter.model.counter.add_(10)
    adapter.model.optional = torch.tensor([4])
    adapter.model.rope_deltas = None
    adapter.model.eval()
    sampler["index"] = 99
    api.restore_state(adapter.model, opt, origin, adapter=adapter, sampler=sampler)
    assert (
        adapter.model.training and not adapter.model.a.training and adapter.model.optional is None
    )
    assert sampler["index"] == 5 and adapter.reset_calls > 0
    path = tmp_path / "checkpoint.pt"
    identity = {"config": "test", "origin": "64"}
    api.save_checkpoint(path, origin, identity)
    assert (
        state_hash(api.load_checkpoint(path, identity, model=adapter.model, optimizer=opt))
        == expected
    )
    with pytest.raises(ValueError, match="identity"):
        api.load_checkpoint(path, {"different": True})
    with pytest.raises(FileExistsError):
        api.save_checkpoint(path, origin, identity)
    bad = copy.deepcopy(origin)
    bad["parameters"]["a.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="shape"):
        api.restore_state(adapter.model, opt, bad, adapter=adapter, sampler=sampler)


def test_exception_restore_and_poison_on_restore_failure(api, tmp_path, monkeypatch):
    adapter = TinyAdapter()
    bank = groups(adapter)
    opt = optimizer(adapter)
    origin = api.capture_state(adapter.model, opt, adapter=adapter)

    def broken(*a, **k):
        adapter.model.counter.add_(100)
        raise RuntimeError("original forward error")

    monkeypatch.setattr(adapter, "logprobs", broken)
    failure = tmp_path / "failure.json"
    with pytest.raises(RuntimeError, match="original forward error"):
        api.fork_one_candidate(
            adapter,
            opt,
            origin,
            bank,
            {"policy": "joint", "auxiliary_weight": 1},
            failure_path=failure,
        )
    assert failure.is_file() and state_hash(
        api.capture_state(adapter.model, opt, adapter=adapter)
    ) == state_hash(origin)
    original_restore = api.restore_state
    calls = []

    def fail_restore(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("cannot restore")
        return original_restore(*a, **k)

    monkeypatch.setattr(api, "restore_state", fail_restore)
    with pytest.raises(RuntimeError, match="original forward error") as caught:
        api.fork_one_candidate(
            adapter, opt, origin, bank, {"policy": "joint", "auxiliary_weight": 1}
        )
    assert any("cannot restore" in n for n in caught.value.__notes__)
    with pytest.raises(RuntimeError, match="unusable"):
        api.fork_one_candidate(
            adapter, opt, origin, bank, {"policy": "joint", "auxiliary_weight": 1}
        )


def test_vector_geometry_and_exact_forward_alias(api):
    def state(values, moment):
        return {
            "parameters": {"a": torch.tensor(values, dtype=torch.float64)},
            "optimizer": {"moment": moment},
            "buffers": {},
            "module_modes": {"": True},
            "position_state": {},
            "frozen_parameter_hash": "same",
        }

    base = state([0.0, 0.0], 0)
    left = state([1.0, 0.0], 1)
    right = state([0.0, 1.0], 2)
    result = api.compare_update_vectors(base, left, right)
    assert result["left_aux_l2"] == result["right_aux_l2"] == 1
    assert result["vector_difference_l2"] == pytest.approx(math.sqrt(2)) and result["cosine"] == 0
    assert result["max_abs_difference"] == 1 and result[
        "relative_vector_difference"
    ] == pytest.approx(math.sqrt(2))
    other = copy.deepcopy(left)
    other["optimizer"]["moment"] = 99
    assert api.forward_state_hash(left) == api.forward_state_hash(other) and state_hash(
        left
    ) != state_hash(other)
    other["parameters"]["a"][0] += 1e-12
    assert api.forward_state_hash(left) != api.forward_state_hash(other)


def test_fp64_streaming_vector_reference_and_group_reordering(api):
    generator = torch.Generator().manual_seed(8)
    baseline = {
        "parameters": {
            n: torch.randn(51, generator=generator, dtype=torch.float64) for n in ["a", "b", "c"]
        },
        "optimizer": {},
    }
    left = copy.deepcopy(baseline)
    right = copy.deepcopy(baseline)
    for n in left["parameters"]:
        left["parameters"][n] += torch.randn(51, generator=generator, dtype=torch.float64) * 0.001
        right["parameters"][n] += torch.randn(51, generator=generator, dtype=torch.float64) * 0.002
    dl = np.concatenate(
        [(left["parameters"][n] - baseline["parameters"][n]).numpy() for n in ["a", "b", "c"]]
    )
    dr = np.concatenate(
        [(right["parameters"][n] - baseline["parameters"][n]).numpy() for n in ["a", "b", "c"]]
    )
    result = api.compare_update_vectors(baseline, left, right)
    assert result["left_aux_l2"] == pytest.approx(np.linalg.norm(dl), rel=1e-14)
    assert result["right_aux_l2"] == pytest.approx(np.linalg.norm(dr), rel=1e-14)
    assert result["vector_difference_l2"] == pytest.approx(np.linalg.norm(dl - dr), rel=1e-14)
    assert result["cosine"] == pytest.approx(
        np.dot(dl, dr) / (np.linalg.norm(dl) * np.linalg.norm(dr)), abs=1e-14
    )
    adapter = TinyAdapter()
    bank = groups(adapter)
    one = api.direct_loss_gradients(adapter, bank, policy="no_x_off", auxiliary_weight=1)
    two = api.direct_loss_gradients(
        adapter, [g[::-1] for g in bank[::-1]], policy="no_x_off", auxiliary_weight=1
    )
    for name in one["gradients"]:
        torch.testing.assert_close(
            one["gradients"][name], two["gradients"][name], atol=1e-8, rtol=1e-6
        )


def test_all_recorded_buffers_positions_and_sampler_state_dict_restore(api):
    class Sampler:
        def __init__(self):
            self.cursor = 3

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    adapter = TinyAdapter()
    opt = optimizer(adapter)
    sampler = Sampler()
    origin = api.capture_state(adapter.model, opt, adapter=adapter, sampler=sampler)
    adapter.model.past_key_values = torch.ones(4)
    adapter._cache = torch.ones(2)
    sampler.cursor = 99
    api.restore_state(adapter.model, opt, origin, adapter=adapter, sampler=sampler)
    assert (
        not hasattr(adapter.model, "past_key_values")
        and not hasattr(adapter, "_cache")
        and sampler.cursor == 3
    )


def test_gradient_cleanup_does_not_mask_primary_failure(api, monkeypatch):
    adapter = TinyAdapter()
    bank = groups(adapter)

    def fail_forward(*a, **k):
        raise RuntimeError("primary forward failure")

    def fail_reset():
        raise RuntimeError("position cleanup failure")

    monkeypatch.setattr(adapter, "logprobs", fail_forward)
    monkeypatch.setattr(adapter, "_reset_positions", fail_reset)
    with pytest.raises(RuntimeError, match="primary forward failure") as caught:
        api.direct_loss_gradients(adapter, bank)
    assert any("position cleanup failure" in n for n in caught.value.__notes__)
    assert adapter._followup_state_unusable


@pytest.mark.parametrize(
    "mutation", ["hash", "tensor_type", "tensor_shape", "moment_dtype", "rng_type", "step_shape"]
)
def test_restricted_checkpoint_type_shape_hash_validation(api, tmp_path, mutation):
    adapter = TinyAdapter()
    opt = optimizer(adapter)
    api.apply_gradient_update(
        adapter.model,
        opt,
        {n: torch.ones_like(p) for n, p in adapter.model.named_parameters() if p.requires_grad},
    )
    state = api.capture_state(adapter.model, opt, adapter=adapter)
    identity = {"test": "validation"}
    payload = {"identity": identity, "state": copy.deepcopy(state), "state_hash": state_hash(state)}
    changed = payload["state"]
    if mutation == "hash":
        changed["parameters"]["a.weight"][0, 0] += 1
    elif mutation == "tensor_type":
        changed["parameters"]["a.weight"] = [[1.0]]
    elif mutation == "tensor_shape":
        changed["parameters"]["a.weight"] = torch.ones(1)
    elif mutation == "moment_dtype":
        next(iter(changed["optimizer"]["state"].values()))["exp_avg"] = torch.ones_like(
            adapter.model.a.weight, dtype=torch.int64
        )
    elif mutation == "rng_type":
        changed["rng"]["torch"] = torch.tensor([1.0])
    elif mutation == "step_shape":
        next(iter(changed["optimizer"]["state"].values()))["step"] = torch.ones(2)
    if mutation != "hash":
        payload["state_hash"] = state_hash(changed)
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    before = state_hash(adapter.model.state_dict())
    with pytest.raises(ValueError):
        api.load_checkpoint(path, identity, model=adapter.model, optimizer=opt)
    assert state_hash(adapter.model.state_dict()) == before


def test_checkpoint_file_hash_checked_before_deserialization(api, tmp_path, monkeypatch):
    adapter = TinyAdapter()
    opt = optimizer(adapter)
    state = api.capture_state(adapter.model, opt, adapter=adapter)
    path = tmp_path / "state.pt"
    binding = api.save_checkpoint(path, state, {"scope": "test"})
    assert state_hash(
        api.load_checkpoint(path, {"scope": "test"}, expected_file_sha256=binding["sha256"])
    ) == state_hash(state)

    def unsafe(*a, **k):
        raise AssertionError("deserialization must not be reached")

    monkeypatch.setattr(torch, "load", unsafe)
    with pytest.raises(ValueError, match="before deserialization"):
        api.load_checkpoint(path, {"scope": "test"}, expected_file_sha256="wrong")


def test_truncated_no_eos_tokens_reach_real_gradient_path_unchanged(api, monkeypatch):
    adapter = TinyAdapter()
    bank = groups(adapter, category_groups=(("X", "S", "W", "I"),))
    adapter.eos_ids = {3}
    truncated = bank[0][-1]
    truncated.update(
        {
            "token_ids": [0, 1, 2, 1],
            "max_new_tokens": 4,
            "stopped_on_eos": False,
            "truncated_at_max_new_tokens": True,
        }
    )
    truncated["old_logprobs"] = adapter.logprobs(
        truncated["prepared"], truncated["token_ids"]
    ).tolist()
    assert not (set(truncated["token_ids"]) & adapter.eos_ids)
    digest = state_hash(bank)
    observed = []
    original = adapter.logprobs

    def score_exact_tokens(prepared, completion, *, require_grad=False):
        if require_grad:
            observed.append(list(completion))
        return original(prepared, completion, require_grad=require_grad)

    monkeypatch.setattr(adapter, "logprobs", score_exact_tokens)
    result = api.direct_loss_gradients(adapter, bank, auxiliary_weight=1)
    assert observed == [row["token_ids"] for row in bank[0]]
    assert observed[-1] == [0, 1, 2, 1]
    assert result["audit"]["total_tokens"] == sum(map(len, observed))
    assert result["audit"]["backward_calls"] == 4
    assert result["audit"]["group_statistics"][0]["category_counts"]["I"] == 1
    assert state_hash(bank) == digest


def test_nonfinite_autograd_inside_fork_persists_failure_without_completion(api, tmp_path):
    import json

    adapter = TinyAdapter()
    bank = groups(adapter)
    opt = optimizer(adapter)
    api.apply_gradient_update(
        adapter.model, opt, api.direct_loss_gradients(adapter, bank)["gradients"]
    )
    origin = api.capture_state(adapter.model, opt, {"checkpoint_step": 1}, adapter=adapter)
    failure = tmp_path / "attempt" / "failure.json"
    hook = adapter.model.a.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        with pytest.raises(FloatingPointError, match="Nonfinite gradient"):
            api.fork_one_candidate(
                adapter,
                opt,
                origin,
                bank,
                {"policy": "joint", "auxiliary_weight": 1},
                failure_path=failure,
            )
    finally:
        hook.remove()
    record = json.loads(failure.read_text())
    assert record["status"] == "FAILED"
    assert record["error_type"] == "FloatingPointError"
    assert "Nonfinite gradient" in record["error"]
    assert record["origin_restored"] is True
    assert record["permanent_training_commit"] is False
    assert not list(tmp_path.rglob("completed.json"))
    assert not list(tmp_path.rglob("*.pt"))
    assert state_hash(
        api.capture_state(adapter.model, opt, origin["metadata"], adapter=adapter)
    ) == state_hash(origin)


def test_equal_parameters_distinct_real_adam_checkpoints_roundtrip_independently(api, tmp_path):
    adapter = TinyAdapter()
    opt = optimizer(adapter)
    bank = groups(adapter)
    api.apply_gradient_update(
        adapter.model, opt, api.direct_loss_gradients(adapter, bank)["gradients"]
    )
    left = api.capture_state(adapter.model, opt, {"candidate": "left"}, adapter=adapter)
    right = copy.deepcopy(left)
    right["metadata"]["candidate"] = "right"
    next(iter(right["optimizer"]["state"].values()))["exp_avg"].add_(0.125)
    assert state_hash(left["parameters"]) == state_hash(right["parameters"])
    assert state_hash(left["optimizer"]) != state_hash(right["optimizer"])
    assert api.forward_state_hash(left) == api.forward_state_hash(right)
    loaded = {}
    for name, state in (("left", left), ("right", right)):
        path = tmp_path / name / "checkpoint.pt"
        identity = {"origin": "same", "candidate": name}
        binding = api.save_checkpoint(path, state, identity)
        loaded[name] = api.load_checkpoint(
            path,
            identity,
            model=adapter.model,
            optimizer=opt,
            expected_file_sha256=binding["sha256"],
        )
        assert state_hash(loaded[name]) == state_hash(state)
    assert len(list(tmp_path.rglob("checkpoint.pt"))) == 2
    assert state_hash(loaded["left"]["parameters"]) == state_hash(loaded["right"]["parameters"])
    assert state_hash(loaded["left"]["optimizer"]) != state_hash(loaded["right"]["optimizer"])
    assert api.forward_state_hash(loaded["left"]) == api.forward_state_hash(loaded["right"])
    with pytest.raises(ValueError, match="identity"):
        api.load_checkpoint(
            tmp_path / "left" / "checkpoint.pt", {"origin": "same", "candidate": "right"}
        )
