import numpy as np
import pytest
import torch

from src.modeling_qualification.toy import (
    build_toy_dataset,
    event_jacobian,
    event_probabilities,
    event_probabilities_from_theta,
    flatten_parameters,
    make_model,
    perform_step,
    prompt_features,
    restore,
    same_answer_alternatives,
    sample_bank,
    snapshot,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return build_toy_dataset(tmp_path_factory.mktemp("toy") / "data", seed=20260915)


def test_dataset_identity_domain_and_structural_validation(dataset):
    assert dataset.train_features.shape == (72, 16, 44)
    assert dataset.probe_features.shape == (72, 16, 44)
    assert len(set(m["base_scene_id"] for m in dataset.probe_metadata)) == 36
    assert set(m["base_scene_id"] for m in dataset.train_metadata).isdisjoint(
        m["base_scene_id"] for m in dataset.probe_metadata
    )
    assert np.bincount([m["group"] for m in dataset.probe_metadata]).tolist() == [12] * 6
    assert np.all((dataset.probe_categories == 0).sum(1) == 1)
    assert np.all((dataset.probe_categories == 3).sum(1) == 2)
    assert all(m["solution_count"] == 1 for m in dataset.probe_metadata)


def test_symbolic_feature_does_not_accept_hidden_truth_copy(dataset):
    scene = dataset.scenes["train"][0]
    changed = dict(scene, truth_world=[99] * 4, changed_index=999)
    np.testing.assert_array_equal(
        prompt_features(scene, "SYMBOLIC_PROXY"), prompt_features(changed, "SYMBOLIC_PROXY")
    )
    assert len(prompt_features(scene, "SYMBOLIC_PROXY")) == 37
    assert prompt_features(scene, "SYMBOLIC_PROXY")[29:34] == [0.0] * 5


def test_model_layout_initialization_and_permutation_invariance(dataset):
    model = make_model()
    assert flatten_parameters(model).shape == (737,)
    np.testing.assert_array_equal(flatten_parameters(model), flatten_parameters(make_model()))
    assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in model.parameters())
    f, c = dataset.probe_features[:2], dataset.probe_categories[:2]
    perm = np.arange(15, -1, -1)
    p = event_probabilities(model, f, c)
    pp = event_probabilities(model, f[:, perm], c[:, perm])
    torch.testing.assert_close(p, pp, atol=1e-14, rtol=1e-14)
    p[:, 0].sum().backward()
    g = np.concatenate([p.grad.detach().numpy().ravel() for p in model.parameters()])
    model.zero_grad()
    event_probabilities(model, f[:, perm], c[:, perm])[:, 0].sum().backward()
    gp = np.concatenate([p.grad.detach().numpy().ravel() for p in model.parameters()])
    np.testing.assert_allclose(g, gp, atol=1e-14)


def test_probability_jacobian_matches_finite_difference(dataset):
    theta = flatten_parameters(make_model())
    f, c = dataset.probe_features[:2], dataset.probe_categories[:2]
    jac = event_jacobian(theta, f, c)
    direction = np.random.default_rng(12).normal(size=737)
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    fd = (
        event_probabilities_from_theta(theta + eps * direction, f, c)
        - event_probabilities_from_theta(theta - eps * direction, f, c)
    ) / (2 * eps)
    np.testing.assert_allclose(jac @ direction, fd, atol=1e-10)
    np.testing.assert_allclose(jac.sum(1), 0, atol=1e-14)


def test_branch_restoration_and_explicit_zero_grad_adam(dataset):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
    rng = np.random.default_rng(123)
    bank = sample_bank(model, dataset.train_features, dataset.train_categories, rng, 4, 8)
    before = snapshot(model, optimizer, rng)
    a = perform_step(model, optimizer, bank, lam=0)
    restore(model, optimizer, rng, before)
    b = perform_step(model, optimizer, bank, lam=0)
    np.testing.assert_array_equal(a["d"], b["d"])
    assert a["loss_denominator"] == 32
    zero_bank = dict(bank, categories=np.full((4, 8), 3))
    step = perform_step(model, optimizer, zero_bank, lam=0)
    assert np.linalg.norm(step["g"]) == 0
    assert step["zero_advantage_fraction"] == 1
    assert np.linalg.norm(step["d"]) > 0  # Adam momentum survives explicit zero gradient.
    assert all(p.grad is not None for p in model.parameters())


def test_same_answer_fiber_boundary():
    assert same_answer_alternatives([0, 0, 0, 0], "sum4", limit=2) == []
    assert len(same_answer_alternatives([99, 99, 99, 99], "range4", limit=2)) == 2


def test_snapshot_restores_none_and_tensor_gradients_after_exception():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    rng = np.random.default_rng(9)
    parameters = list(model.parameters())
    parameters[0].grad = torch.ones_like(parameters[0])
    before = snapshot(model, optimizer, rng)
    try:
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        raise RuntimeError("injected branch failure")
    except RuntimeError:
        restore(model, optimizer, rng, before)
    torch.testing.assert_close(parameters[0].grad, torch.ones_like(parameters[0]))
    assert all(p.grad is None for p in parameters[1:])
