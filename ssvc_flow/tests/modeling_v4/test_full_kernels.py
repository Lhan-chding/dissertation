import numpy as np
import pytest

from src.modeling_v4.full_kernels import (
    effective_delta_inner,
    effective_weight_gram,
    fit_effective_response,
    fit_full_response,
    fit_precomputed_response,
    load_model,
    product_inner,
    raw_gram,
    save_model,
)


@pytest.mark.parametrize(
    "method,rank",
    [
        ("FULL_DUAL_RIDGE", "FULL"),
        ("FULL_DUAL_RIDGE", 2),
        ("FULL_RBF_RAW", "FULL"),
        ("LEGACY_QPROJECTED_RBF", "FULL"),
    ],
)
def test_block_streamed_gram_preserves_full_input_predictions_and_saving(tmp_path, method, rank):
    rng = np.random.default_rng(713)
    e, query = rng.normal(size=(5, 13)), rng.normal(size=(3, 13))
    query[2] = 0
    y = rng.normal(size=(5, 2, 4))
    gram, cross, norms = e @ e.T, query @ e.T, (query**2).sum(1)
    direct = fit_full_response(e, y, method=method, rank_cap=rank, alpha=0.1)
    streamed = fit_precomputed_response(
        gram,
        y,
        query_cross_gram=cross,
        query_norms=norms,
        method=method,
        rank_cap=rank,
        alpha=0.1,
        input_metadata={"input_dimension": 13},
    )
    np.testing.assert_allclose(streamed["predict"](), direct["predict"](query), atol=1e-12)
    np.testing.assert_allclose(streamed["geometry"]()["rho"], direct["geometry"](query)["rho"])
    np.testing.assert_allclose(streamed["predict"]([1]), direct["predict"](query[1:2]))
    expected = streamed["predict"]().copy()
    gram[:] = cross[:] = norms[:] = 0
    np.testing.assert_array_equal(streamed["predict"](), expected)
    save_model(streamed, tmp_path / "model")
    np.testing.assert_allclose(load_model(tmp_path / "model")["predict"](), expected)
    with pytest.raises(ValueError):
        streamed["predict"]([-1])


def test_full_dual_equals_absolute_primal_and_preserves_raw_channels():
    rng = np.random.default_rng(12)
    e, y, query = rng.normal(size=(7, 13)), rng.normal(size=(7, 2, 4)), rng.normal(size=(3, 13))
    model = fit_full_response(e, y, alpha=0.13)
    expected = query @ np.linalg.solve(e.T @ e + 0.13 * np.eye(13), e.T @ y.reshape(7, -1))
    np.testing.assert_allclose(model["predict"](query), expected.reshape(3, 2, 4), atol=1e-12)
    assert model["metadata"]["ridge_lambda"] == 0.13
    assert not np.allclose(model["predict"](query).sum(-1), 0)


def test_full_keeps_weak_mode_even_below_diagnostic_rank_threshold():
    e = np.diag([1.0, 1e-6])
    y = np.asarray([[0.0], [1.0]])
    full = fit_full_response(e, y, alpha=1e-8)
    equivalent = fit_full_response(e, y, alpha=1e-8, rank_cap=full["k"])
    assert full["predict"](e)[1, 0] > 0
    np.testing.assert_array_equal(full["predict"](e), equivalent["predict"](e))
    assert equivalent["metadata"]["rank_label"] == "FULL_EQUIVALENT"


def test_true_compression_and_rank_equal_full_labels():
    e = np.diag([4.0, 3.0, 2.0, 1.0])
    y = np.eye(4)
    full = fit_full_response(e, y)
    small = fit_full_response(e, y, rank_cap=2)
    same = fit_full_response(e, y, rank_cap=8)
    assert small["metadata"]["rank_label"] == "TRUE_COMPRESSION"
    assert small["r"] == 2 < small["k"] == 4
    assert not np.allclose(small["predict"](e), full["predict"](e))
    np.testing.assert_array_equal(same["predict"](e), full["predict"](e))
    assert same["metadata"]["rank_label"] == "FULL_EQUIVALENT"
    with pytest.raises(ValueError):
        fit_full_response(e, y, rank_cap=0)


def test_block_standardization_uses_only_fit_data_and_complete_input():
    e = np.asarray([[1.0, 0, 100.0], [0, 2.0, -100.0]])
    blocks = {"a": [0, 2], "b": [2, 3]}
    model = fit_full_response(e, np.eye(2), blocks=blocks, standardize_blocks=True, alpha=0.2)
    scales = model["metadata"]["block_scales"]
    assert scales == {"a": np.sqrt(2.5), "b": 100.0}
    scaled = e / [scales["a"], scales["a"], scales["b"]]
    np.testing.assert_allclose(model["gram"], scaled @ scaled.T)
    query = np.asarray([[1, 3.0, 1e8]])
    before = dict(scales)
    model["predict"](query)
    assert model["metadata"]["block_scales"] == before
    np.testing.assert_allclose(raw_gram({"a": e[:, :2], "b": e[:, 2:]}), e @ e.T)
    with pytest.raises(ValueError, match="partition"):
        fit_full_response(e, np.eye(2), blocks={"missing": [0, 2]})


def test_full_rbf_sees_orthogonal_query_and_is_zero_anchored():
    e, query = np.asarray([[1.0, 0], [2.0, 0]]), np.asarray([[0.5, 0], [0.5, 2]])
    y = np.asarray([[1.0], [0.5]])
    full = fit_full_response(e, y, method="FULL_RBF_RAW", alpha=0.1)
    legacy = fit_full_response(e, y, method="LEGACY_QPROJECTED_RBF", alpha=0.1)
    assert not np.isclose(*full["predict"](query).ravel())
    np.testing.assert_array_equal(legacy["predict"](query)[0], legacy["predict"](query)[1])
    np.testing.assert_array_equal(full["predict"](np.zeros((1, 2))), [[0]])
    assert np.linalg.eigvalsh(full["gram"]).min() >= -1e-12
    assert full["metadata"]["length_scale"] == 1.0
    assert full["geometry"](query)["rho"][1] > 0


def test_unexcited_is_unknown_and_true_zero_remains_exact_zero():
    model = fit_full_response(np.zeros((2, 3)), np.zeros((2, 4)))
    assert np.isnan(model["predict"](np.ones((1, 3)))).all()
    np.testing.assert_array_equal(model["predict"](np.zeros((1, 3))), np.zeros((1, 4)))


def factors(rng, rank=2):
    return {"B": rng.normal(size=(7, rank)), "A": rng.normal(size=(rank, 5))}


def contrast(rng):
    return {
        "base_model_id": "fixed-base",
        "modules": {"layer": {"scaling": 2.0, "candidate": factors(rng), "baseline": factors(rng)}},
    }


def dense(value):
    layer = value["modules"]["layer"]
    c, b = layer["candidate"], layer["baseline"]
    return layer["scaling"] * (c["B"] @ c["A"] - b["B"] @ b["A"])


def test_low_rank_trace_and_effective_gram_equal_dense_without_gauge_dependence():
    rng = np.random.default_rng(25)
    values = [contrast(rng) for _ in range(3)]
    expected = np.asarray([[np.sum(dense(a) * dense(b)) for b in values] for a in values])
    np.testing.assert_allclose(effective_weight_gram(values), expected, atol=1e-11)
    a, b = factors(rng, 2), factors(rng, 3)
    np.testing.assert_allclose(
        product_inner(a["B"], a["A"], b["B"], b["A"]),
        np.sum((a["B"] @ a["A"]) * (b["B"] @ b["A"])),
        atol=1e-12,
    )
    transformed = []
    for value in values:
        layer = value["modules"]["layer"]
        changed = {"scaling": layer["scaling"]}
        for endpoint in ("candidate", "baseline"):
            matrix = np.diag([0.2, 4.0])
            factor = layer[endpoint]
            changed[endpoint] = {
                "B": factor["B"] @ matrix,
                "A": np.linalg.solve(matrix, factor["A"]),
            }
        transformed.append({"base_model_id": "fixed-base", "modules": {"layer": changed}})
    np.testing.assert_allclose(effective_weight_gram(transformed), expected, atol=1e-10)
    np.testing.assert_allclose(
        effective_delta_inner(values[0], values[1]), expected[0, 1], atol=1e-11
    )
    y = rng.normal(size=(3, 4))
    fit = fit_effective_response(values, y, alpha=0.3)
    np.testing.assert_allclose(
        fit["predict"](transformed),
        expected @ np.linalg.solve(expected + 0.3 * np.eye(3), y),
        atol=1e-12,
    )
    bad = contrast(rng)
    bad["base_model_id"] = "different-base"
    with pytest.raises(ValueError, match="identity"):
        effective_weight_gram(values, [bad])


def test_safe_save_load_rebuild_is_declared_and_query_shape_preserved(tmp_path):
    rng = np.random.default_rng(4)
    e, y, query = rng.normal(size=(5, 4)), rng.normal(size=(5, 2, 4)), rng.normal(size=(2, 4))
    fit = fit_full_response(e, y, method="FULL_RBF_RAW", alpha=0.01)
    save_model(fit, tmp_path / "model")
    restored = load_model(tmp_path / "model")
    np.testing.assert_array_equal(restored["predict"](query), fit["predict"](query))
    assert restored["metadata"]["load_mechanism"] == "REFIT_ON_LOAD"
    with pytest.raises(FileExistsError):
        save_model(fit, tmp_path / "model")


def test_raw_models_cannot_masquerade_as_effective_weights_and_partition_is_strict():
    with pytest.raises(ValueError, match="raw"):
        fit_full_response(np.eye(2), np.eye(2), method="FULL_EFFECTIVE_WEIGHT_RIDGE")
    with pytest.raises(ValueError, match="integer"):
        fit_full_response(np.eye(2), np.eye(2), blocks={"x": [0, 2.5]})


def test_legacy_rbf_projects_both_fit_and_query_at_the_same_rank():
    e = np.diag([1.0, 1e-6])
    y = np.asarray([[0.0], [1.0]])
    legacy = fit_full_response(e, y, method="LEGACY_QPROJECTED_RBF", alpha=1e-10)
    np.testing.assert_array_equal(legacy["gram"][1], np.zeros(2))
    np.testing.assert_array_equal(legacy["predict"](e), np.zeros((2, 1)))


def test_named_partition_and_effective_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(9)
    e, y = rng.normal(size=(4, 3)), rng.normal(size=(4, 2, 4))
    blocks = {"last": [2, 3], "first": [0, 2]}
    model = fit_full_response(e, y, blocks=blocks, standardize_blocks=True)
    expected = model["predict"](e)
    blocks["first"][1] = 1
    np.testing.assert_array_equal(model["predict"](e), expected)
    model["metadata"]["block_scales"]["last"] = 1e12
    np.testing.assert_array_equal(model["predict"](e), expected)
    save_model(model, tmp_path / "partition")
    np.testing.assert_array_equal(load_model(tmp_path / "partition")["predict"](e), expected)
    contrasts = [contrast(rng) for _ in range(4)]
    effective = fit_effective_response(contrasts, y, method="FULL_RBF_EFFECTIVE_WEIGHT")
    save_model(effective, tmp_path / "effective")
    restored = load_model(tmp_path / "effective")
    np.testing.assert_array_equal(effective["predict"](contrasts), restored["predict"](contrasts))
    with (tmp_path / "effective" / "FIT.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity"):
        load_model(tmp_path / "effective")
