from __future__ import annotations

from compensability.study_c3.execution_contract import build_execution_contract
from compensability.study_c3.factorial_design import validate_study_c3_config

from .test_factorial_pairing import _config


def _training_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "scene_id": f"scene-{index:03d}",
            "prompt_sha256": f"{index:064x}",
            "split": "train",
        }
        for index in range(192)
    )


def test_execution_contract_binds_sources_and_common_random_numbers() -> None:
    contract = build_execution_contract(
        config=validate_study_c3_config(_config()),
        training_rows=_training_rows(),
        b3_adapter_sha256="a" * 64,
        model_snapshot_sha256="b" * 64,
        source_sha256s={
            "package_lock_sha256": "c" * 64,
            "fiber_rows_sha256": "d" * 64,
            "c2_training_pair_manifest_sha256": "e" * 64,
            "action_audit_manifest_sha256": "f" * 64,
            "decoder_intervention_manifest_sha256": "1" * 64,
        },
        git_commit_sha="2" * 40,
    )

    assert contract["status"] == "STUDY_C3_FACTORIAL_EXECUTION_CONTRACT_FROZEN"
    assert contract["training_prompt_count"] == 192
    assert contract["training_seed_list"] == [2026082501]
    assert contract["single_seed_mechanism_pilot"] is True
    assert contract["checkpoint_steps"] == [48, 96, 144, 192]
    assert tuple(contract["arms"]) == ("A_BIN", "X_BIN", "A_LEX", "X_LEX")
    assert all(
        details["rollout_seeds"] == contract["arms"]["A_BIN"]["rollout_seeds"]
        for details in contract["arms"].values()
    )
    assert contract["reward_only_factorial_verified"] is True
