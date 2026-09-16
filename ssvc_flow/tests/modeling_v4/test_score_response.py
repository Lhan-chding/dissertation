"""Exact small mathematical and differentiable-adapter fixtures, never a VLM."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.modeling_v4.score_response import (
    certify_gradient_path,
    collect_score_response,
    differentiable_full_teacher_logprobs,
    directional_derivative,
    exact_score_response,
    numerical_directional_check,
    score_sample_gradient,
    sequence_log_probability,
)


def test_exact_origin_base_midpoint_use_full_direction_and_separate_finite_difference():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(11, 5))
    categories = np.arange(11) % 4
    origin, base, end = rng.normal(size=(3, 5)) * 0.1
    result = exact_score_response(
        origin, base, end, features, categories, projection_basis=np.eye(5)[:, :2]
    )
    direction = end - base
    for name, point in (("ORIGIN", origin), ("BASE", base), ("MIDPOINT", (base + end) / 2)):
        x = torch.tensor(point, requires_grad=True)
        f = torch.tensor(features)
        mask = torch.eye(4, dtype=torch.float64)[categories]
        jac = torch.autograd.functional.jacobian(
            lambda value, f=f, mask=mask: (f @ value).softmax(0) @ mask, x
        )
        np.testing.assert_allclose(result["derivatives"][name], jac.detach().numpy() @ direction)
    assert not np.allclose(result["derivatives"]["ORIGIN"], result["finite_response"])
    np.testing.assert_allclose(
        result["orthogonal_contribution"],
        result["jacobians"]["ORIGIN"] @ result["orthogonal_direction"],
    )


def test_jvp_reverse_fallback_and_finite_difference_remain_distinct():
    point, direction = (
        torch.tensor([0.2, -0.1], dtype=torch.float64),
        torch.tensor([0.4, 0.5], dtype=torch.float64),
    )

    def fn(value):
        return torch.stack((torch.sin(value[0]) + value[1] ** 2, torch.exp(value[1])))

    forward = directional_derivative(fn, point, direction)

    def unavailable(*args, **kwargs):
        raise NotImplementedError("forward AD not implemented for fixture operator")

    reverse = directional_derivative(fn, point, direction, jvp=unavailable)
    np.testing.assert_allclose(forward["derivative"], reverse["derivative"])
    assert reverse["method"] == "REVERSE_GRAD_DOT" and reverse["jvp_failure"]
    checked = numerical_directional_check(fn, point, direction, steps=[1e-2, 1e-4])
    assert checked["finite_difference_is_derivative"] is False
    assert checked["checks"][-1]["max_abs_error"] < 1e-8


def test_sequence_fp64_includes_eos_and_excludes_only_explicit_padding():
    logp = torch.tensor([-1e8, -0.1, -0.2, -123.0], dtype=torch.float32, requires_grad=True)
    score = sequence_log_probability(
        logp, [0, 1, 2, 0], eos_ids={2}, max_new_tokens=3, token_mask=[True, True, True, False]
    )
    assert score.dtype == torch.float64
    assert score.item() == pytest.approx(sum(logp.detach().double().numpy()[:3]), abs=2e-8)
    score.backward()
    np.testing.assert_array_equal(logp.grad, [1, 1, 1, 0])
    with pytest.raises(ValueError, match=r"EOS|mask"):
        sequence_log_probability(logp[:3], [0, 2, 1], eos_ids={2}, max_new_tokens=3)


class TinyAdapter:
    eos_ids, pad_id = {2}, 0

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.lora_A = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
        self.model.lora_B = torch.nn.Parameter(torch.tensor([0.1, 0.3], dtype=torch.float64))
        self.model.base = torch.nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64), requires_grad=False
        )
        self.model.eval()
        self.forward_calls = 0
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda tokens, **kw: " ".join(map(str, tokens)))
        )

    def prepare(self, prompt, data_root):
        return {"audit": {"input_tensor_hash": "fixture", "enable_thinking": False}}

    def logprobs(self, prepared, tokens, *, require_grad):
        self.model.train(require_grad)
        self.forward_calls += len(tokens)
        with torch.set_grad_enabled(require_grad):
            logits = torch.cat((self.model.lora_A + self.model.lora_B, self.model.base[None]))
            return logits.log_softmax(0)[tokens]


def runtime():
    adapter = TinyAdapter()
    return {
        "adapter": adapter,
        "identity": {"fixture": True},
        "parameter_order": ["lora_A", "lora_B"],
        "parity_tolerances": {
            "mean_abs_token_logp": 1e-12,
            "max_abs_token_logp": 1e-12,
            "max_abs_sequence_logp": 1e-12,
        },
    }


def prompt(key="p0", group="f0"):
    return {"prompt_id": key, "family": group, "interface": "SYMBOLIC_FRESH", "prompt": {}}


def sample(runtime, key="s0", prompt_id="p0", category="X"):
    adapter = runtime["adapter"]
    scores = adapter.logprobs({}, [0, 2], require_grad=False).tolist()
    return {
        "sample_key": key,
        "rng_namespace": "anchor",
        "role": "anchor",
        "prompt_id": prompt_id,
        "token_ids": [0, 2],
        "completion_length": 2,
        "stop_reason": "eos",
        "raw_completion": "0",
        "behavior_token_logprobs": scores,
        "generation_sequence_logp": sum(scores),
        "max_new_tokens": 4,
        "category": category,
        "proposal": "ORIGIN",
    }


def test_full_trainable_gradient_keeps_frozen_params_and_existing_grads_unchanged():
    rt = runtime()
    action = sample(rt)
    model = rt["adapter"].model
    model.lora_A.grad = torch.ones_like(model.lora_A)
    before = model.lora_A.grad.clone()
    result = score_sample_gradient(rt, prompt(), action)
    logits = torch.cat((model.lora_A + model.lora_B, model.base[None])).detach()
    expected = np.array([1, 0]) - 2 * logits.softmax(0).numpy()[:2]
    for gradient in result["gradients"].values():
        np.testing.assert_allclose(gradient, expected)
    assert result["parameter_order"] == ["lora_A", "lora_B"]
    assert result["cost"]["backward_calls"] == 1 and result["cost"]["forward_calls"] == 4
    assert result["parity"]["passed"] is True
    assert not model.training and model.base.grad is None
    torch.testing.assert_close(model.lora_A.grad, before)
    with pytest.raises(ValueError, match="trainable"):
        score_sample_gradient(rt, prompt(), action, parameter_order=["lora_A"])


def test_grouped_score_response_keeps_raw_mass_and_all_direction_coordinates():
    rt = runtime()
    prompts = [prompt(), prompt("p1", "f1")]
    samples = [
        sample(rt, f"s{i}", prompts[i % 2]["prompt_id"], "X" if i < 2 else "I") for i in range(4)
    ]
    directions = {"query": {"lora_A": np.array([0.4, 0.0]), "lora_B": np.array([0.0, 0.7])}}
    result = collect_score_response(
        rt,
        prompts,
        samples,
        expected_groups=[("f0", "SYMBOLIC_FRESH"), ("f1", "SYMBOLIC_FRESH")],
        prompt_subset=["p0"],
        directions=directions,
    )
    assert result["grouped_gradients"]["lora_A"].shape == (2, 4, 2)
    assert result["prompt_gradients"]["lora_A"].shape == (1, 4, 2)
    expected = sum(
        np.einsum("gcd,d->gc", result["grouped_gradients"][name], value)
        for name, value in directions["query"].items()
    )
    np.testing.assert_allclose(result["grouped_directional_response"][0], expected)
    assert np.any(result["raw_directional_mass"] != 0)
    assert result["cost"]["backward_calls"] == 4
    assert result["reference_labels_read"] is False
    raw = result["raw_directional_score_products"]
    assert raw.shape == (4, 1)
    gradient = score_sample_gradient(rt, prompts[0], samples[0])["gradients"]
    expected_dot = sum(np.sum(gradient[n] * v) for n, v in directions["query"].items())
    np.testing.assert_allclose(raw[:, 0], expected_dot)
    assert result["derivative_kind"] == "AUTOGRAD_FULL_PARAMETER_SCORE"
    assert result["sample_records"][2] == {
        "sample_id": "s2",
        "prompt_id": "p0",
        "group": ["f0", "SYMBOLIC_FRESH"],
        "category": "I",
        "rng_namespace": "anchor",
        "proposal": "ORIGIN",
        "proposal_policy_fingerprint": None,
    }
    assert len(result["parameter_layout_hash"]) == 64


def test_score_rejects_reference_draw_before_model_call():
    rt = runtime()
    action = sample(rt)
    action["role"] = "reference"
    calls = rt["adapter"].forward_calls
    with pytest.raises(ValueError, match="reference"):
        score_sample_gradient(rt, prompt(), action)
    assert rt["adapter"].forward_calls == calls


def test_bounded_direction_provider_contracts_actual_gradients_without_materializing_mapping():
    rt = runtime()
    draws = [sample(rt, "s0"), sample(rt, "s1")]

    class Provider:
        validated, calls = False, 0

        def __iter__(self):
            return iter(["cal", "query"])

        def __len__(self):
            return 2

        def validate(self, params):
            assert list(params) == ["lora_A", "lora_B"]
            self.validated = True

        def dot_all(self, gradients):
            assert self.validated
            self.calls += 1
            return np.array([gradients["lora_A"].sum(), gradients["lora_B"][1]])

    provider = Provider()
    result = collect_score_response(
        rt,
        [prompt()],
        draws,
        gradients=False,
        directions=provider,
        expected_groups=[("f0", "SYMBOLIC_FRESH")],
    )
    gradient = score_sample_gradient(rt, prompt(), draws[0])["gradients"]
    np.testing.assert_allclose(
        result["raw_directional_score_products"],
        [provider.dot_all(gradient), provider.dot_all(gradient)],
    )
    assert result["direction_ids"] == ["cal", "query"]
    assert result["grouped_gradients"] is None


def test_gradient_fastpath_requires_separate_measured_derivative_certificate():
    rt = runtime()
    action = sample(rt)

    def fast(adapter, prepared, tokens):
        return adapter.logprobs(prepared, tokens, require_grad=True)

    directions = {"full": {"lora_A": np.ones(2), "lora_B": np.ones(2)}}
    tolerances = {**rt["parity_tolerances"], "relative_l2": 1e-3, "absolute_directional_dot": 1e-5}
    with pytest.raises(ValueError, match="derivative certificate"):
        score_sample_gradient(
            rt,
            prompt(),
            action,
            scorer=fast,
            derivative_certificate={
                "passed": True,
                "scope": "evaluation_only_no_derivative_certification",
            },
        )
    receipt = certify_gradient_path(
        rt, prompt(), action, scorer=fast, directions=directions, tolerances=tolerances
    )
    assert receipt["passed"] is True and receipt["fixture"] is True
    assert receipt["cost"]["backward_calls"] == 2
    assert set(receipt["blocks"]) == {"lora_A", "lora_B"}
    result = score_sample_gradient(
        rt, prompt(), action, scorer=fast, derivative_certificate=receipt
    )
    assert result["numerical_path"] == "certified_differentiable_fastpath"
    rt["derivative_scorer"], rt["derivative_certificate"] = fast, receipt
    collected = collect_score_response(
        rt, [prompt()], [action], expected_groups=[("f0", "SYMBOLIC_FRESH")]
    )
    assert (
        collected["sample_score_records"][0]["numerical_path"]
        == "certified_differentiable_fastpath"
    )


def test_fault_costs_and_model_mode_survive_failed_gradient_parity():
    rt = runtime()
    action = sample(rt)
    original = rt["adapter"].logprobs

    def corrupt(*args, require_grad):
        values = original(*args, require_grad=require_grad)
        return values + 0.01 if require_grad else values

    rt["adapter"].logprobs = corrupt
    with pytest.raises(ValueError, match="parity"):
        score_sample_gradient(rt, prompt(), action)
    assert rt["score_cost_ledger"][-1]["cost"]["backward_calls"] == 1
    assert rt["score_cost_ledger"][-1]["cost"]["scored_sequences"] == 2
    assert rt["score_cost_ledger"][-1]["status"] == "FAILED"
    assert not rt["adapter"].model.training


def test_base_importance_weights_and_streaming_match_full_gradient_collection():
    rt = runtime()
    action = sample(rt)
    action["proposal_sequence_logp"] = action["generation_sequence_logp"] - np.log(2)
    directions = {"full": {"lora_A": np.ones(2), "lora_B": np.ones(2)}}
    args = dict(
        expansion_point="BASE", expected_groups=[("f0", "SYMBOLIC_FRESH")], directions=directions
    )
    full = collect_score_response(rt, [prompt()], [action], **args)
    stream = collect_score_response(rt, [prompt()], [action], gradients=False, **args)
    np.testing.assert_allclose(
        full["grouped_directional_response"], stream["grouped_directional_response"]
    )
    np.testing.assert_allclose(stream["importance_weights"], [2.0])
    assert stream["grouped_gradients"] is None
    with pytest.raises(ValueError, match="reference"):
        collect_score_response(
            rt, [prompt()], [action], reference_rng_stream_ids=["anchor"], **args
        )


def test_gradient_fastpath_mismatch_returns_not_certified_without_replacing_prefix():
    rt = runtime()
    action = sample(rt)

    def wrong_gradient(adapter, prepared, tokens):
        value = adapter.logprobs(prepared, tokens, require_grad=True)
        return value + 0.01 * (value - value.detach())

    certificate = certify_gradient_path(
        rt,
        prompt(),
        action,
        scorer=wrong_gradient,
        directions={"full": {"lora_A": np.ones(2), "lora_B": np.ones(2)}},
        tolerances={
            **rt["parity_tolerances"],
            "relative_l2": 1e-3,
            "absolute_directional_dot": 1e-5,
        },
    )
    assert certificate["passed"] is False
    assert certificate["status"] == "NOT_CERTIFIED"
    assert certificate["probability"]["passed"] is True
    assert (
        score_sample_gradient(rt, prompt(), action)["numerical_path"] == "uncached_prefix_recompute"
    )


def test_differentiable_teacher_selects_real_action_positions_and_retains_graph():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.Parameter(torch.arange(15).reshape(5, 3).float() / 10)

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.lora[None])

    model = Head()
    adapter = SimpleNamespace(
        model=model, _reset_positions=lambda: None, full_inputs=lambda p, t: {}
    )
    actual = differentiable_full_teacher_logprobs(
        adapter, {"audit": {"prompt_token_count": 3}}, [0, 2]
    )
    expected = model.lora[2:4].log_softmax(-1)[torch.arange(2), torch.tensor([0, 2])]
    torch.testing.assert_close(actual, expected)
    actual.double().sum().backward()
    assert model.lora.grad[:2].abs().sum() == 0 and model.lora.grad[4].abs().sum() == 0
    assert model.lora.grad[2:4].abs().sum() > 0
