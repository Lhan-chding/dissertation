"""CPU-only gradient/fork contracts; no model download or CUDA evidence."""

import copy
import importlib
import importlib.util

import pytest
import torch

from src.optimizer_fork import capture_state, restore_state, state_hash


def test_public_api_is_available():
    assert importlib.util.find_spec("src.fork_gradients") is not None


@pytest.fixture
def api():
    return importlib.import_module("src.fork_gradients")


class TinyAdapter:
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.register_parameter(
            "lora_A", torch.nn.Parameter(torch.tensor([[0.2, -0.3], [0.4, 0.1]]))
        )
        self.model.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(2, 4)))
        self.model.register_parameter(
            "base", torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3, -0.4]), requires_grad=False)
        )
        self.forward_calls = 0

    def logprobs(self, prepared, completion, *, require_grad=False):
        self.forward_calls += 1
        self.model.train(require_grad)
        with torch.set_grad_enabled(require_grad):
            scores = []
            for i, token in enumerate(completion):
                features = prepared["features"] + i * 0.15
                logits = features @ self.model.lora_A @ self.model.lora_B + self.model.base
                scores.append(logits.log_softmax(-1)[token])
            return torch.stack(scores)


def make_groups(adapter, categories=("X", "S", "W", "I"), prompts=2):
    sequences = ([0, 2], [1], [2, 3, 1], [3, 0, 2, 3])
    groups = []
    for p in range(prompts):
        prepared = {"features": torch.tensor([1.0 + p, -0.3 - p])}
        group = []
        for i, category in enumerate(categories):
            tokens = list(sequences[i % len(sequences)])
            group.append(
                {
                    "prompt_id": f"p{p}",
                    "sample_key": f"p{p}-s{i}",
                    "category": category,
                    "token_ids": tokens,
                    "old_logprobs": adapter.logprobs(prepared, tokens).tolist(),
                    "prepared": prepared,
                    "reward_sum": 123.0,
                }
            )
        groups.append(group)
    return groups


def manual_score_gradient(adapter, groups, *, arm="X_BASE", lam=0.0, lnorm=64, epsilon=1e-4):
    adapter.model.zero_grad(set_to_none=True)
    loss = 0
    for group in groups:
        rewards = torch.tensor(
            [
                2.0 * (r["category"] in (("X", "S") if arm.startswith("A") else ("X",)))
                + lam * (r["category"] != "I")
                for r in group
            ]
        )
        centered = rewards - rewards.mean()
        advantage = centered / (centered.square().mean() + epsilon**2).sqrt()
        for row, adv in zip(group, advantage, strict=True):
            loss = loss - adv * adapter.logprobs(
                row["prepared"], row["token_ids"], require_grad=True
            ).sum() / (sum(map(len, groups)) * lnorm)
    loss.backward()
    return {
        n: p.grad.detach().clone() for n, p in adapter.model.named_parameters() if p.requires_grad
    }


def test_direct_is_uncut_fixed_token_loss_without_update_or_input_mutation(api):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    original = state_hash(groups)
    parameters = state_hash(adapter.model.state_dict())
    expected = manual_score_gradient(adapter, groups)
    adapter.model.eval()
    result = api.direct_loss_gradients(adapter, groups)
    assert result["audit"]["backward_calls"] == 8
    assert result["audit"]["optimizer_updates"] == 0
    assert result["audit"]["ratio_one_exact"]
    assert result["audit"]["gradient_stage"] == "PRE_CLIP"
    assert result["audit"]["total_tokens"] == 20
    assert not adapter.model.training
    assert state_hash(groups) == original
    assert state_hash(adapter.model.state_dict()) == parameters
    assert all(p.grad is None for p in adapter.model.parameters())
    for name in expected:
        assert torch.allclose(result["gradients"][name], expected[name], atol=1e-7, rtol=1e-5)
        assert result["gradients"][name].device.type == "cpu"
        assert result["gradients"][name].dtype == torch.float32
    assert torch.count_nonzero(result["gradients"]["lora_A"]) == 0
    assert torch.count_nonzero(result["gradients"]["lora_B"]) > 0
    twice = api.direct_loss_gradients(adapter, groups, lnorm=32)
    assert torch.allclose(twice["gradients"]["lora_B"], result["gradients"]["lora_B"] * 2)


@pytest.mark.parametrize(
    "arm,lam", [("X_BASE", 0.0), ("X_VALID", 1.0), ("A_BASE", 0.0), ("A_VALID", 1.0)]
)
def test_four_category_scores_reproduce_independent_full_losses(api, arm, lam):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    scores = api.collect_category_scores(adapter, groups)
    assert set(scores["score_sums"]["p0"]) == {"X", "S", "W", "I"}
    assert scores["audit"]["backward_calls"] == 8
    assert scores["audit"]["requires_direct_gradient_validation"]
    assert not scores["audit"]["cannot_adopt_reuse"]
    direct = api.direct_loss_gradients(adapter, groups, arm=arm, auxiliary_weight=lam)
    reused = api.combine_category_gradients(scores, groups, arm=arm, auxiliary_weight=lam)
    comparison = api.compare_gradient_bundles(direct, reused)
    assert comparison["passed"], comparison
    assert comparison["informative"]
    assert not comparison["zero_consistent"]
    assert reused["audit"]["backward_calls"] == 0


def test_answer_reward_distinguishes_s_from_w_and_epsilon_is_used(api):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    scores = api.collect_category_scores(adapter, groups)
    x = api.combine_category_gradients(scores, groups, arm="X_BASE", auxiliary_weight=0)
    a = api.combine_category_gradients(scores, groups, arm="A_BASE", auxiliary_weight=0)
    assert not torch.allclose(x["gradients"]["lora_B"], a["gradients"]["lora_B"])
    direct = api.direct_loss_gradients(adapter, groups, epsilon=0.5)
    reused = api.combine_category_gradients(
        scores, groups, arm="X_BASE", auxiliary_weight=0, epsilon=0.5
    )
    assert api.compare_gradient_bundles(direct, reused)["passed"]
    expected = manual_score_gradient(adapter, groups, epsilon=0.5)
    assert torch.allclose(direct["gradients"]["lora_B"], expected["lora_B"], atol=1e-7)


@pytest.mark.parametrize("drift,active", [(0.08, False), (0.7, True)])
def test_original_old_probability_drift_blocks_score_adoption(api, drift, active):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    for group in groups:
        for row in group:
            row["old_logprobs"] = [value - drift for value in row["old_logprobs"]]
    original = state_hash(groups)
    direct = api.direct_loss_gradients(adapter, groups)
    scores = api.collect_category_scores(adapter, groups)
    reused = api.combine_category_gradients(scores, groups, arm="X_BASE", auxiliary_weight=0)
    assert direct["audit"]["max_abs_token_log_ratio"] == pytest.approx(drift, abs=1e-6)
    assert direct["audit"]["surrogate_clipping_active"] is active
    assert reused["audit"]["cannot_adopt_reuse"]
    assert not api.compare_gradient_bundles(direct, reused)["passed"]
    assert state_hash(groups) == original


def test_rewards_and_old_probabilities_are_detached(api):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    reward = torch.tensor(7.0, requires_grad=True)
    lam = torch.tensor(1.0, requires_grad=True)
    olds = []
    for group in groups:
        for row in group:
            row["reward_sum"] = reward
            row["old_logprobs"] = torch.tensor(row["old_logprobs"], requires_grad=True)
            olds.append(row["old_logprobs"])
    result = api.direct_loss_gradients(adapter, groups, auxiliary_weight=lam)
    assert result["audit"]["auxiliary_weight"] == 1.0
    assert reward.grad is None and lam.grad is None
    assert all(old.grad is None for old in olds)


def test_missing_categories_and_zero_bank_are_not_informative_pass(api):
    adapter = TinyAdapter()
    groups = make_groups(adapter, categories=("X", "X"))
    scores = api.collect_category_scores(adapter, groups)
    assert set(scores["score_sums"]["p0"]) == {"X"}
    assert scores["audit"]["missing_categories"]["p0"] == ["S", "W", "I"]
    assert (
        scores["audit"]["missing_category_interpretation"]
        == "EMPIRICAL_ZERO_CONDITIONAL_MEAN_UNESTIMABLE"
    )
    direct = api.direct_loss_gradients(adapter, groups)
    reused = api.combine_category_gradients(scores, groups, arm="X_BASE", auxiliary_weight=0)
    result = api.compare_gradient_bundles(direct, reused)
    assert result["zero_consistent"] and result["allclose"] and result["finite"]
    assert not result["informative"] and not result["passed"]
    assert all(torch.count_nonzero(g) == 0 for g in direct["gradients"].values())


@pytest.mark.parametrize("warm", [False, True])
def test_adam_candidates_restore_independently_in_both_orders(api, warm):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=0.01, weight_decay=0
    )
    if warm:
        api.apply_gradient_update(
            adapter.model, optimizer, api.direct_loss_gradients(adapter, groups)["gradients"]
        )
        groups = make_groups(adapter)
    origin = capture_state(adapter.model, optimizer)
    outcomes = []
    for order in ((0.0, 1.0), (1.0, 0.0)):
        candidates = {}
        for lam in order:
            restore_state(adapter.model, optimizer, origin)
            gradients = api.direct_loss_gradients(adapter, groups, auxiliary_weight=lam)
            report = api.apply_gradient_update(adapter.model, optimizer, gradients["gradients"])
            assert report["optimizer_updates"] == 1
            assert report["grad_norm_postclip"] <= 1.000001
            assert report["scheduler"] == report["grad_scaler"] == "NOT_CONFIGURED"
            candidates[lam] = state_hash(capture_state(adapter.model, optimizer))
        outcomes.append(candidates)
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0.0] != outcomes[0][1.0]


def test_mature_adam_zero_tensor_step_matches_manual_and_differs_from_none(api):
    adapter = TinyAdapter()
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=0.01, weight_decay=0
    )
    groups = make_groups(adapter)
    api.apply_gradient_update(
        adapter.model, optimizer, api.direct_loss_gradients(adapter, groups)["gradients"]
    )
    origin = capture_state(adapter.model, optimizer)
    zero_groups = make_groups(adapter, categories=("X", "X"))
    zeros = api.direct_loss_gradients(adapter, zero_groups)["gradients"]
    report = api.apply_gradient_update(adapter.model, optimizer, zeros)
    actual = capture_state(adapter.model, optimizer)
    assert report["grad_norm_preclip"] == report["grad_norm_postclip"] == 0
    assert report["actual_step_norm"] > 0
    assert report["optimizer_hash_before"] != report["optimizer_hash_after"]
    for key, before in origin["optimizer"]["state"].items():
        after = actual["optimizer"]["state"][key]
        assert float(after["step"]) == float(before["step"]) + 1
        assert torch.allclose(after["exp_avg"], before["exp_avg"] * 0.9)
        assert torch.allclose(after["exp_avg_sq"], before["exp_avg_sq"] * 0.999)
    restore_state(adapter.model, optimizer, origin)
    for parameter in adapter.model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert state_hash(capture_state(adapter.model, optimizer)) == state_hash(actual)
    restore_state(adapter.model, optimizer, origin)
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    assert state_hash(capture_state(adapter.model, optimizer)) == state_hash(origin)


@pytest.mark.parametrize("mutation", ["old", "category", "tokens", "scores"])
def test_score_bank_rejects_changed_source_or_cached_gradient(api, mutation):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    scores = api.collect_category_scores(adapter, groups)
    if mutation == "old":
        groups[0][0]["old_logprobs"][0] += 0.1
    elif mutation == "category":
        groups[0][0]["category"] = "W"
    elif mutation == "tokens":
        groups[0][0]["token_ids"][0] = 3
    else:
        scores["score_sums"]["p0"]["X"]["lora_B"][0, 0] += 1
    with pytest.raises(ValueError, match=r"binding|checksum"):
        api.combine_category_gradients(scores, groups, arm="X_BASE", auxiliary_weight=0)


@pytest.mark.parametrize("mutation", ["missing", "shape", "nan", "frozen_optimizer"])
def test_invalid_gradient_updates_are_rejected_before_mutation(api, mutation):
    adapter = TinyAdapter()
    groups = make_groups(adapter)
    gradients = api.direct_loss_gradients(adapter, groups)["gradients"]
    selected = (
        list(adapter.model.parameters())
        if mutation == "frozen_optimizer"
        else [p for p in adapter.model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(selected, lr=0.01)
    before = capture_state(adapter.model, optimizer)
    if mutation == "missing":
        gradients.pop("lora_A")
    elif mutation == "shape":
        gradients["lora_A"] = torch.zeros(1)
    elif mutation == "nan":
        gradients["lora_A"][0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        api.apply_gradient_update(adapter.model, optimizer, gradients)
    assert state_hash(capture_state(adapter.model, optimizer)) == state_hash(before)


def test_gradient_comparison_reports_mismatch_and_nonfinite_without_passing(api):
    adapter = TinyAdapter()
    gradients = api.direct_loss_gradients(adapter, make_groups(adapter))
    bad = copy.deepcopy(gradients)
    bad["gradients"]["lora_B"][0, 0] = float("nan")
    result = api.compare_gradient_bundles(gradients, bad)
    assert not result["finite"] and not result["passed"]
    bad["gradients"].pop("lora_A")
    result = api.compare_gradient_bundles(gradients, bad)
    assert not result["same_parameter_keys"] and not result["passed"]


@pytest.mark.parametrize("point", ["before", "after"])
def test_nonfinite_adam_moments_never_return_success(api, point):
    adapter = TinyAdapter()
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=0.01, weight_decay=0
    )
    gradients = api.direct_loss_gradients(adapter, make_groups(adapter))["gradients"]
    api.apply_gradient_update(adapter.model, optimizer, gradients)

    def corrupt():
        for value in optimizer.state.values():
            value["exp_avg_sq"].fill_(float("inf"))

    handle = None
    if point == "before":
        corrupt()
    else:
        handle = optimizer.register_step_post_hook(lambda *args: corrupt())
    before = state_hash(adapter.model.state_dict())
    try:
        with pytest.raises(FloatingPointError, match="Adam state"):
            api.apply_gradient_update(adapter.model, optimizer, gradients)
        if point == "before":
            assert state_hash(adapter.model.state_dict()) == before
    finally:
        if handle is not None:
            handle.remove()


def test_direct_gradient_and_step_match_frozen_update_helper(api, monkeypatch):
    from src.grpo_update import perform_update, reward_channels

    adapter = TinyAdapter()
    reference = copy.deepcopy(adapter)
    groups = make_groups(adapter)
    for group in groups:
        for row in group:
            row["reward_sum"] = reward_channels(row["category"], "A_VALID", 0.25)["sum"]
            row["old_logprobs"] = [v - 0.08 for v in row["old_logprobs"]]
    direct = api.direct_loss_gradients(adapter, groups, arm="A_VALID", auxiliary_weight=0.25)
    optimizer = torch.optim.AdamW(
        [p for p in adapter.model.parameters() if p.requires_grad], lr=0.01, weight_decay=0
    )
    reference_optimizer = torch.optim.AdamW(
        [p for p in reference.model.parameters() if p.requires_grad], lr=0.01, weight_decay=0
    )
    captured = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        captured.update(
            {
                n: p.grad.detach().cpu().clone()
                for n, p in reference.model.named_parameters()
                if p.requires_grad
            }
        )
        return original_clip(parameters, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
        perform_update(reference, reference_optimizer, groups)
    assert api.compare_gradient_bundles(direct, {"gradients": captured})["passed"]
    api.apply_gradient_update(adapter.model, optimizer, direct["gradients"])
    assert state_hash(adapter.model.state_dict()) == state_hash(reference.model.state_dict())
    assert state_hash(optimizer.state_dict()) == state_hash(reference_optimizer.state_dict())
