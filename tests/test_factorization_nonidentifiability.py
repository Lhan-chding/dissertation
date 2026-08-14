from __future__ import annotations

import numpy as np

from compbias.identification.nonidentifiability import relabel_factorization


def test_latent_bijection_preserves_end_to_end_behavior() -> None:
    rng = np.random.default_rng(12)
    perception = rng.dirichlet(np.ones(4), size=9)
    reasoning = rng.dirichlet(np.ones(3), size=4)
    permutation = (2, 0, 3, 1)

    transformed = relabel_factorization(perception, reasoning, permutation)

    np.testing.assert_allclose(
        transformed.perception @ transformed.reasoning,
        perception @ reasoning,
        rtol=0.0,
        atol=1e-15,
    )
    assert not np.array_equal(transformed.perception, perception)


def test_non_bijection_is_rejected() -> None:
    perception = np.full((2, 3), 1.0 / 3.0)
    reasoning = np.full((3, 2), 0.5)
    try:
        relabel_factorization(perception, reasoning, (0, 0, 2))
    except ValueError as error:
        assert "bijection" in str(error)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("non-bijection should be rejected")
