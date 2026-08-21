"""Factorial manifest completeness and frozen-data isolation contracts."""

from __future__ import annotations

import copy

import pytest
from compensability_v5.data.correction_factorial import (
    FactorialManifestError,
    SplitIsolationError,
    validate_factorial_isolation,
    validate_factorial_manifest,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "scene_id": "v5-train-001",
        "split": "v5_support_train",
        "truth": [9, 4, 5, 6],
        "natural_observation": [8, 4, 5, 6],
        "error_count": 1,
        "error_magnitudes": [-1],
        "error_domain": "in_domain",
        "constraint_matrix": [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        "graph_signature": "pair-chain-v1",
        "answer_operation": {"operator": "sum", "indices": [0, 2]},
        "fiber_size": 3,
        "orbit_parent": "orbit-parent-001",
        "transformation": {"variable_permutation": [0, 1, 2, 3], "fact_permutation": [0, 1, 2]},
        "image_path": "artifacts/v5/data/images/v5-train-001.png",
        "prompt_path": "artifacts/v5/data/prompts/v5-train-001.txt",
        "image_hash": "a" * 64,
        "prompt_hash": "b" * 64,
    }
    row.update(updates)
    return row


def test_factorial_manifest_records_every_preregistered_field() -> None:
    row = _row()

    assert validate_factorial_manifest([row]) is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "truth",
        "natural_observation",
        "error_count",
        "error_magnitudes",
        "constraint_matrix",
        "graph_signature",
        "answer_operation",
        "fiber_size",
        "orbit_parent",
        "transformation",
        "image_hash",
        "prompt_hash",
    ],
)
def test_factorial_manifest_rejects_every_missing_preregistered_field(missing_field: str) -> None:
    row = _row()
    del row[missing_field]

    with pytest.raises(FactorialManifestError, match=missing_field):
        validate_factorial_manifest([row])


@pytest.mark.parametrize("hash_field", ["image_hash", "prompt_hash"])
def test_factorial_manifest_requires_lowercase_sha256(hash_field: str) -> None:
    row = _row(**{hash_field: "not-a-sha256"})

    with pytest.raises(FactorialManifestError, match=hash_field):
        validate_factorial_manifest([row])


def test_primary_rows_require_exactly_one_in_domain_natural_error() -> None:
    for updates in (
        {"error_count": 2, "error_magnitudes": [-1, 2]},
        {"error_count": 1, "error_magnitudes": []},
        {"error_domain": "out_of_domain"},
    ):
        with pytest.raises(FactorialManifestError, match=r"error|primary|domain"):
            validate_factorial_manifest([_row(**updates)])


def test_stress_rows_allow_multi_error_out_of_domain_observations() -> None:
    stress = _row(
        scene_id="v5-stress-001",
        split="v5_stress",
        natural_observation=[8, 6, 5, 6],
        error_count=2,
        error_magnitudes=[-1, 2],
        error_domain="out_of_domain",
    )

    assert validate_factorial_manifest([stress]) is None


@pytest.mark.parametrize(
    "updates",
    [
        {"scene_id": "phase8-confirm-current-001"},
        {"orbit_parent": "phase8-confirm-current-001"},
        {"image_path": "artifacts/compensability_v4/phase8/images/current.png"},
        {"prompt_path": "artifacts/compensability_v4/confirm/prompts/current.txt"},
    ],
)
def test_v5_train_and_dev_reject_current_phase8_or_confirm_sources(
    updates: dict[str, object],
) -> None:
    contaminated = _row(**updates)

    with pytest.raises(SplitIsolationError, match=r"phase8|confirm|reserved|isolation"):
        validate_factorial_isolation(
            [contaminated],
            reserved_scene_ids={"phase8-confirm-current-001"},
            reserved_path_fragments=(
                "artifacts/compensability_v4/phase8",
                "artifacts/compensability_v4/confirm",
            ),
        )


def test_disjoint_v5_train_dev_and_stress_rows_pass_isolation() -> None:
    train = _row()
    dev = _row(
        scene_id="v5-dev-001",
        split="v5_support_dev",
        orbit_parent="orbit-parent-dev",
        image_path="artifacts/v5/data/images/v5-dev-001.png",
        prompt_path="artifacts/v5/data/prompts/v5-dev-001.txt",
        image_hash="c" * 64,
        prompt_hash="d" * 64,
    )
    stress = copy.deepcopy(
        _row(
            scene_id="v5-stress-001",
            split="v5_stress",
            orbit_parent="orbit-parent-stress",
            natural_observation=[7, 6, 5, 6],
            error_count=2,
            error_magnitudes=[-2, 2],
            error_domain="out_of_domain",
            image_path="artifacts/v5/data/images/v5-stress-001.png",
            prompt_path="artifacts/v5/data/prompts/v5-stress-001.txt",
            image_hash="e" * 64,
            prompt_hash="f" * 64,
        )
    )

    assert validate_factorial_manifest([train, dev, stress]) is None
    assert (
        validate_factorial_isolation(
            [train, dev, stress],
            reserved_scene_ids={"phase8-confirm-current-001"},
            reserved_path_fragments=("artifacts/compensability_v4/phase8",),
        )
        is None
    )
