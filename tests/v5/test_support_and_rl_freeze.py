"""Contracts for the CPU-only v5 support and common-space freezes."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from compensability_v5.audit.budget_audit import BudgetMismatchError
from compensability_v5.data.common_action_freeze import (
    ACTION_PARSER_ID,
    PILOT_SEED,
    CommonActionFreezeError,
    assert_common_action_preflight,
    freeze_common_action_space,
)
from compensability_v5.evaluation.build_v5_tables import build_advisor_packet
from compensability_v5.training.build_budget_matched_support import (
    SupportBuildError,
    build_budget_matched_support,
)

ROOT = Path(__file__).resolve().parents[2]


def _source(**updates: object) -> dict[str, object]:
    source: dict[str, object] = {
        "scene_id": "support-001",
        "semantic_scene_id": "semantic-001",
        "prompt": "Observed world: 8,2,3,4. Recover the true world.",
        "truth": [9, 2, 3, 4],
        "natural_observation": [8, 2, 3, 4],
        "constraint_matrix": [[1, 0, 0, 0], [0, 1, 0, 0]],
        "constraint_targets": [9, 2],
        "answer_operation": {"operator": "sum", "indices": [0, 1]},
        "transformation": {"kind": "identity"},
    }
    source.update(updates)
    return source


def _training_budget() -> dict[str, object]:
    return {
        "steps": 12,
        "optimizer": {"name": "adamw", "learning_rate": 2e-5, "weight_decay": 0.0},
        "lora_rank": 16,
        "lora_targets": ["q_proj", "v_proj"],
        "gradient_accumulation": 2,
        "approximate_flops": 1.0,
    }


def _provenance() -> dict[str, str]:
    return {
        "parent_manifest_sha256": "1" * 64,
        "child_manifest_sha256": "2" * 64,
        "frozen_scenes_sha256": "3" * 64,
    }


def test_support_builder_emits_exactly_six_rows_per_source_in_every_arm() -> None:
    calls: list[str] = []

    def token_counter(text: str) -> int:
        calls.append(text)
        return 8

    package = build_budget_matched_support(
        [_source(), _source(scene_id="support-002", semantic_scene_id="semantic-002")],
        token_counter=token_counter,
        training_budget=_training_budget(),
        source_provenance=_provenance(),
    )

    assert set(package["arms"]) == {"B0", "B1", "B2", "B3"}
    for arm, rows in package["arms"].items():
        assert len(rows) == 12
        assert {row["arm"] for row in rows} == {arm}
        assert all(row["target_tokens"] == 8 for row in rows)
        assert all(isinstance(row["prompt"], str) and row["prompt"] for row in rows)
        assert all(isinstance(row["completion"], str) and row["completion"] for row in rows)
    assert len(calls) == 48
    assert calls == [
        row["completion"] for arm in ("B0", "B1", "B2", "B3") for row in package["arms"][arm]
    ]
    assert {budget["rows"] for budget in package["budgets"].values()} == {12}
    assert {budget["unique_source_scenes"] for budget in package["budgets"].values()} == {2}


def test_support_builder_calls_budget_assertion_and_rejects_token_drift() -> None:
    calls = 0

    def drifting_counter(_text: str) -> int:
        nonlocal calls
        calls += 1
        return 100 if calls > 18 else 1

    with pytest.raises(BudgetMismatchError, match="target_tokens"):
        build_budget_matched_support(
            [_source()],
            token_counter=drifting_counter,
            training_budget=_training_budget(),
            source_provenance=_provenance(),
            target_token_relative_tolerance=0.01,
        )


def test_support_builder_uses_identical_completion_budget_across_all_arms() -> None:
    package = build_budget_matched_support(
        [_source()],
        token_counter=len,
        training_budget=_training_budget(),
        source_provenance=_provenance(),
    )

    completions = {row["completion"] for rows in package["arms"].values() for row in rows}
    assert completions == {"9,2,3,4"}
    assert {budget["target_tokens"] for budget in package["budgets"].values()} == {42}
    assert package["target_token_relative_tolerance"] == 0.01


@pytest.mark.parametrize(
    ("sources", "counter", "budget", "match"),
    [
        ([], lambda _text: 1, _training_budget(), "at least one"),
        ([{"scene_id": "bad"}], lambda _text: 1, _training_budget(), "truth"),
        ([_source(prompt="")], lambda _text: 1, _training_budget(), "prompt"),
        ([_source()], lambda _text: 0, _training_budget(), "positive integer"),
        (
            [_source(), _source()],
            lambda _text: 1,
            _training_budget(),
            "unique",
        ),
        (
            [_source()],
            lambda _text: 1,
            {**_training_budget(), "unregistered": True},
            "closed schema",
        ),
    ],
)
def test_support_builder_rejects_malformed_inputs(
    sources: list[dict[str, object]],
    counter: object,
    budget: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(SupportBuildError, match=match):
        build_budget_matched_support(
            sources,
            token_counter=counter,  # type: ignore[arg-type]
            training_budget=budget,
            source_provenance=_provenance(),
        )


def test_support_builder_requires_closed_source_provenance() -> None:
    with pytest.raises(SupportBuildError, match="source_provenance"):
        build_budget_matched_support(
            [_source()],
            token_counter=len,
            training_budget=_training_budget(),
            source_provenance={"parent_manifest_sha256": "1" * 64},
        )


def _rl_scene(**updates: object) -> dict[str, object]:
    scene: dict[str, object] = {
        "scene_id": "rl-001",
        "prompt": "Observed world: 8,2,3,4. Return four comma-separated integers only.",
        "truth": [9, 2, 3, 4],
        "answer_operation": {"operator": "sum", "indices": [0, 1]},
        "family": "pair_sum",
        "fiber_size": 3,
        "fiber_bin": "multi_state",
        "support_bin": "low",
        "candidate_worlds": [[9, 2, 3, 4], [8, 3, 3, 4]],
    }
    scene.update(updates)
    return scene


def _initializations() -> dict[str, str]:
    return {"B3": "a" * 64, "B2": "b" * 64, "Base": "c" * 64}


def test_common_action_freeze_stores_one_prompt_and_both_reward_labels() -> None:
    package = freeze_common_action_space(
        [_rl_scene()],
        initialization_hashes=_initializations(),
        action_parser_id=ACTION_PARSER_ID,
        rollout_seeds=[PILOT_SEED],
    )

    assert assert_common_action_preflight(package) is None
    assert len(package["scenes"]) == 1
    frozen = package["scenes"][0]
    assert frozen["prompt"] == _rl_scene()["prompt"]
    assert frozen["reward_labels"] == {"answer": 11, "exact_state": [9, 2, 3, 4]}
    assert frozen["family"] == "pair_sum"
    assert frozen["fiber_size"] == 3
    assert frozen["fiber_bin"] == "multi_state"
    assert frozen["support_bin"] == "low"
    assert frozen["candidate_worlds"] == [[9, 2, 3, 4], [8, 3, 3, 4]]
    assert "answer_prompt" not in frozen
    assert "exact_state_prompt" not in frozen
    assert package["preflight"]["only_reward_function_differs"] is True
    assert {arm["scene_metadata_hash"] for arm in package["arms"].values()} == {
        package["scene_metadata_hash"]
    }


@pytest.mark.parametrize(
    "forbidden",
    [
        {"prompt_path": "prompts/rl-001.txt"},
        {"answer_prompt": "answer-specific prompt"},
        {"exact_state_prompt": "state-specific prompt"},
        {"prompt_files": ["answer.txt", "state.txt"]},
    ],
)
def test_common_action_freeze_rejects_separate_prompt_surfaces(
    forbidden: dict[str, object],
) -> None:
    with pytest.raises(CommonActionFreezeError, match=r"single prompt|prompt file"):
        freeze_common_action_space(
            [_rl_scene(**forbidden)],
            initialization_hashes=_initializations(),
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )


@pytest.mark.parametrize(
    "field",
    ["prompt_hash", "initialization_hash", "action_parser_hash", "rollout_seed_hash"],
)
def test_common_action_preflight_rejects_hash_drift_between_reward_pairs(field: str) -> None:
    package = freeze_common_action_space(
        [_rl_scene()],
        initialization_hashes=_initializations(),
        action_parser_id=ACTION_PARSER_ID,
        rollout_seeds=[PILOT_SEED],
    )
    drifted = copy.deepcopy(package)
    drifted["arms"]["B3_exact_state"][field] = "f" * 64

    with pytest.raises(CommonActionFreezeError, match=field):
        assert_common_action_preflight(drifted)


def test_common_action_freeze_requires_exactly_one_fixed_pilot_seed() -> None:
    with pytest.raises(CommonActionFreezeError, match="exactly one"):
        freeze_common_action_space(
            [_rl_scene()],
            initialization_hashes=_initializations(),
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED, PILOT_SEED + 1],
        )


@pytest.mark.parametrize(
    ("parser_id", "seeds", "match"),
    [
        ("four-csv-int-parser-v1", [PILOT_SEED], "action_parser_id"),
        (ACTION_PARSER_ID, [11], "seed"),
    ],
)
def test_common_action_freeze_rejects_registered_runtime_drift(
    parser_id: str,
    seeds: list[int],
    match: str,
) -> None:
    with pytest.raises(CommonActionFreezeError, match=match):
        freeze_common_action_space(
            [_rl_scene()],
            initialization_hashes=_initializations(),
            action_parser_id=parser_id,
            rollout_seeds=seeds,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"family": ""},
        {"fiber_size": 0},
        {"candidate_worlds": []},
        {"candidate_worlds": [[1, 2, 3]]},
    ],
)
def test_common_action_freeze_validates_optional_registered_metadata(
    updates: dict[str, object],
) -> None:
    with pytest.raises(CommonActionFreezeError, match=r"family|fiber_size|candidate_worlds"):
        freeze_common_action_space(
            [_rl_scene(**updates)],
            initialization_hashes=_initializations(),
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("prompt_hash", "0" * 64, "prompt_hash"),
        ("action_parser_hash", "0" * 64, "action_parser_hash"),
        ("rollout_seed_hash", "0" * 64, "rollout_seed_hash"),
        ("scene_metadata_hash", "0" * 64, "scene_metadata_hash"),
    ],
)
def test_common_action_preflight_recomputes_package_hashes(
    field: str,
    replacement: str,
    match: str,
) -> None:
    package = freeze_common_action_space(
        [_rl_scene()],
        initialization_hashes=_initializations(),
        action_parser_id=ACTION_PARSER_ID,
        rollout_seeds=[PILOT_SEED],
    )
    package[field] = replacement

    with pytest.raises(CommonActionFreezeError, match=match):
        assert_common_action_preflight(package)


def test_advisor_packet_is_status_only_when_registered_results_are_missing() -> None:
    packet = build_advisor_packet({"support_results": {"complete": True}})

    assert packet == {
        "schema_version": 1,
        "status": "BLOCKED_MISSING_RESULTS",
        "missing_results": ["confirmation_results", "reward_results"],
    }
    assert "tables" not in packet
    assert "metrics" not in packet
    assert "advisor_brief" not in packet


def test_advisor_packet_copies_only_complete_registered_facts() -> None:
    results = {
        "support_results": {"complete": True, "exact_recovery": 0.25},
        "reward_results": {"complete": True, "exact_recovery": 0.5},
        "confirmation_results": {"complete": True, "exact_recovery": 0.375},
    }

    packet = build_advisor_packet(results)

    assert packet["status"] == "ADVISOR_PACKET_READY"
    assert packet["results"] == results
    assert packet["results"] is not results
    assert "advisor_brief" not in packet


def test_advisor_packet_allows_registered_study_b_stop_without_study_c_results() -> None:
    packet = build_advisor_packet(
        {
            "support_results": {
                "complete": True,
                "stop_signal": {"triggered": True, "rule": "registered_no_support"},
            },
            "confirmation_results": {"complete": True},
        }
    )

    assert packet == {
        "schema_version": 1,
        "status": "PARTIAL_DECISIVE_PILOT",
        "study_c_status": "NOT_RUN_DUE_TO_REGISTERED_STOP",
        "results": {
            "support_results": {
                "complete": True,
                "stop_signal": {"triggered": True, "rule": "registered_no_support"},
            },
            "confirmation_results": {"complete": True},
        },
    }


@pytest.mark.parametrize(
    "script",
    ["05_build_budget_matched_support.py", "08_freeze_common_space_rl.py"],
)
def test_freeze_scripts_have_cpu_only_fixture_preflights(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/v5" / script), "--fixture-dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "FIXTURE_DRY_RUN_OK"


def test_advisor_script_writes_only_status_and_exits_nonzero_without_results(
    tmp_path: Path,
) -> None:
    output = tmp_path / "advisor"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/v5/11_build_advisor_packet.py"),
            "--execute",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert sorted(path.name for path in output.iterdir()) == ["status.json"]
    assert json.loads((output / "status.json").read_text())["status"] == ("BLOCKED_MISSING_RESULTS")
