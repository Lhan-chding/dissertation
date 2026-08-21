"""Contracts for the CPU-only v5 support and common-space freezes."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tarfile
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
from compensability_v5.evaluation.build_v5_tables import (
    build_advisor_packet,
    write_advisor_artifacts,
)
from compensability_v5.qwen.study_b_inputs import validate_support_package
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
        "policy_support": 0.25,
        "candidate_worlds": [[9, 2, 3, 4], [8, 3, 3, 4]],
    }
    scene.update(updates)
    return scene


def _initializations() -> dict[str, str]:
    return {"B3": "a" * 64, "B2": "b" * 64, "Base": "c" * 64}


def _study_a_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "V5_STUDY_A_EXECUTED",
        "source_sha256": {"phase2a": "a" * 64, "Base": "b" * 64, "T": "c" * 64},
        "semantic_scene_count": 96,
        "scenario_count": 480,
        "scenario_checkpoint_count": 960,
        "by_checkpoint": {"Base": {}, "T": {}},
        "by_graph_axis": {"canonical": {}},
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
        "confirmatory_data_used": False,
    }


def _study_b_summary(*, triggered: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "STUDY_B_SINGLE_SEED_COMPLETE",
        "seed": 2026082201,
        "model_snapshot_sha256": "d" * 64,
        "arm_results": {arm: {} for arm in ("B0", "B1", "B2", "B3")},
        "primary_contrasts": {"paired_inference": {}},
        "stop_signal": {
            "triggered": triggered,
            "rule": "B3_minus_B2_paired_CI95_lower_gt_zero",
        },
    }


def _study_c_summary(*, triggered: bool = False) -> dict[str, object]:
    interaction = {
        "triggered": triggered,
        "rule": "B3 reward-by-fiber interaction > 0 and scene-bootstrap CI excludes 0",
    }
    trajectory = {
        "triggered": False,
        "rule": "large-fiber answer accuracy rises while world recovery falls vs B3 init",
    }
    return {
        "schema_version": 1,
        "status": "STUDY_C_DIAGNOSTICS_COMPLETE",
        "seed": 2026082301,
        "group_size": 8,
        "by_arm": {"B3_answer": {}},
        "per_scene": [],
        "registered_stop_signals": {
            "reward_by_fiber_interaction": interaction,
            "answer_up_world_down_large_fibers": trajectory,
            "any_registered_signal_triggered": triggered,
            "subjective_threshold_used": False,
        },
    }


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
    assert frozen["fiber_bin"] == "multi_2_4"
    assert frozen["support_bin"] == "medium"
    assert frozen["role"] == "rl_train"
    assert frozen["candidate_worlds"] == [[9, 2, 3, 4], [8, 3, 3, 4]]
    assert "answer_prompt" not in frozen
    assert "exact_state_prompt" not in frozen
    assert package["preflight"]["only_reward_function_differs"] is True
    assert {arm["scene_metadata_hash"] for arm in package["arms"].values()} == {
        package["scene_metadata_hash"]
    }


def test_common_action_freeze_derives_disjoint_registered_72_24_split() -> None:
    scenes = [
        _rl_scene(
            scene_id=f"rl-{index:03d}",
            family=("known_value", "pair_sum", "trend")[index % 3],
            fiber_size=(1, 3, 6)[index % 3],
            policy_support=index / 95,
        )
        for index in range(96)
    ]
    package = freeze_common_action_space(
        scenes,
        initialization_hashes=_initializations(),
        action_parser_id=ACTION_PARSER_ID,
        rollout_seeds=[PILOT_SEED],
    )

    assert package["role_counts"] == {"rl_train": 72, "rl_eval": 24}
    assert {scene["fiber_bin"] for scene in package["scenes"]} == {
        "singleton",
        "multi_2_4",
        "multi_5_plus",
    }
    assert {scene["support_bin"] for scene in package["scenes"]} == {
        "low",
        "medium",
        "high",
    }
    assert {scene["role"] for scene in package["scenes"]} == {"rl_train", "rl_eval"}


def test_common_action_freeze_rejects_user_supplied_bins_or_role() -> None:
    with pytest.raises(CommonActionFreezeError, match="derived"):
        freeze_common_action_space(
            [_rl_scene(support_bin="low")],
            initialization_hashes=_initializations(),
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )


def test_common_action_freeze_rejects_non_mapping_scene_cleanly() -> None:
    with pytest.raises(CommonActionFreezeError, match="must be a mapping"):
        freeze_common_action_space(
            [None],  # type: ignore[list-item]
            initialization_hashes=_initializations(),
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )


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
    packet = build_advisor_packet({"support_results": _study_b_summary()})

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
        "support_results": _study_b_summary(),
        "reward_results": _study_c_summary(triggered=True),
        "confirmation_results": _study_a_summary(),
    }

    packet = build_advisor_packet(results)

    assert packet["status"] == "ADVISOR_PACKET_READY"
    assert packet["results"] == results
    assert packet["results"] is not results
    assert packet["registered_stop_signals"]["study_c"]["any_registered_signal_triggered"] is True
    assert "advisor_brief" not in packet


def test_advisor_packet_allows_registered_study_b_stop_without_study_c_results() -> None:
    packet = build_advisor_packet(
        {
            "support_results": _study_b_summary(triggered=True),
            "confirmation_results": _study_a_summary(),
        }
    )

    assert packet == {
        "schema_version": 1,
        "status": "PARTIAL_DECISIVE_PILOT",
        "study_c_status": "NOT_RUN_DUE_TO_REGISTERED_STOP",
        "results": {
            "support_results": {
                **_study_b_summary(triggered=True),
            },
            "confirmation_results": _study_a_summary(),
        },
        "registered_stop_signals": {"study_b": _study_b_summary(triggered=True)["stop_signal"]},
    }


def test_advisor_packet_rejects_legacy_complete_wrappers() -> None:
    with pytest.raises(ValueError, match="native status"):
        build_advisor_packet({"support_results": {"complete": True}})


def test_advisor_artifacts_are_complete_and_byte_stable(tmp_path: Path) -> None:
    root = tmp_path / "study-a"
    root.mkdir()
    (root / "rows.jsonl").write_text('{"metric":1}\n')
    packet = build_advisor_packet(
        {
            "support_results": _study_b_summary(),
            "confirmation_results": _study_a_summary(),
            "reward_results": _study_c_summary(),
        }
    )
    first, second = tmp_path / "first", tmp_path / "second"
    write_advisor_artifacts(packet, artifact_roots={"study_a": root}, output_root=first)
    write_advisor_artifacts(packet, artifact_roots={"study_a": root}, output_root=second)

    expected = {
        "QWEN_V5_PILOT_RESULT_FACTS.md",
        "qwen_v5_pilot_raw_rows.tar.gz",
        "sha256_manifest.json",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert (first / "qwen_v5_pilot_raw_rows.tar.gz").read_bytes() == (
        second / "qwen_v5_pilot_raw_rows.tar.gz"
    ).read_bytes()


def test_advisor_artifacts_reject_symlinked_raw_inputs(tmp_path: Path) -> None:
    root = tmp_path / "study-a"
    root.mkdir()
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (root / "escape.json").symlink_to(target)
    packet = {"status": "PARTIAL_DECISIVE_PILOT"}

    with pytest.raises(ValueError, match="symlink"):
        write_advisor_artifacts(
            packet, artifact_roots={"study_a": root}, output_root=tmp_path / "output"
        )


def test_advisor_archive_allowlists_text_evidence_and_excludes_training_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study-b"
    (root / "final_adapter").mkdir(parents=True)
    (root / "checkpoint-72").mkdir()
    (root / "rows.jsonl").write_text('{"metric":1}\n')
    (root / "run_manifest.json").write_text("{}\n")
    (root / "config.yaml").write_text("schema_version: 1\n")
    (root / "training.log").write_text("complete\n")
    (root / "final_adapter/adapter_config.json").write_text("{}\n")
    (root / "final_adapter/adapter_model.safetensors").write_bytes(b"weights")
    (root / "checkpoint-72/trainer_state.json").write_text("{}\n")
    (root / "pytorch_model.bin").write_bytes(b"weights")
    output = tmp_path / "output"

    write_advisor_artifacts(
        {"status": "PARTIAL_DECISIVE_PILOT"},
        artifact_roots={"study_b": root},
        output_root=output,
    )

    with tarfile.open(output / "qwen_v5_pilot_raw_rows.tar.gz", "r:gz") as archive:
        assert archive.getnames() == [
            "study_b/config.yaml",
            "study_b/rows.jsonl",
            "study_b/run_manifest.json",
            "study_b/training.log",
        ]


def test_advisor_archive_rejects_binary_disguised_as_text_evidence(tmp_path: Path) -> None:
    root = tmp_path / "study-a"
    root.mkdir()
    (root / "rows.jsonl").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(ValueError, match="UTF-8 text"):
        write_advisor_artifacts(
            {"status": "PARTIAL_DECISIVE_PILOT"},
            artifact_roots={"study_a": root},
            output_root=tmp_path / "output",
        )


@pytest.mark.parametrize("unsafe_label", [".", "..", "a/b", r"a\b"])
def test_advisor_artifacts_reject_unsafe_archive_root_labels(
    tmp_path: Path, unsafe_label: str
) -> None:
    root = tmp_path / "study-a"
    root.mkdir()
    (root / "rows.jsonl").write_text("{}\n")

    with pytest.raises(ValueError, match="safe named directories"):
        write_advisor_artifacts(
            {"status": "PARTIAL_DECISIVE_PILOT"},
            artifact_roots={unsafe_label: root},
            output_root=tmp_path / "output",
        )


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


def test_support_script_hash_binds_phase2a_provenance(tmp_path: Path) -> None:
    scenes_path = tmp_path / "frozen-scenes.jsonl"
    scenes_path.write_text(
        "".join(
            json.dumps(
                _source(
                    scene_id=f"support-{index:03d}",
                    semantic_scene_id=f"semantic-{index:03d}",
                ),
                sort_keys=True,
            )
            + "\n"
            for index in range(96)
        )
    )
    parent_path = tmp_path / "parent.json"
    parent_path.write_text('{"status":"PARENT_FROZEN"}\n')
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    scenes_sha = hashlib.sha256(scenes_path.read_bytes()).hexdigest()
    child_path = tmp_path / "child.json"
    child_path.write_text(
        json.dumps(
            {
                "status": "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN",
                "parent_manifest_sha256": parent_sha,
                "frozen_scenes_sha256": scenes_sha,
                "parent_manifest_modified": False,
                "semantic_scene_count": 96,
            }
        )
        + "\n"
    )
    output = tmp_path / "support.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/v5/05_build_budget_matched_support.py"),
            "--execute",
            "--input-jsonl",
            str(scenes_path),
            "--parent-manifest",
            str(parent_path),
            "--child-manifest",
            str(child_path),
            "--token-counter",
            "builtins:len",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    package = json.loads(output.read_text())
    assert package["source_provenance"] == {
        "parent_manifest_sha256": parent_sha,
        "child_manifest_sha256": hashlib.sha256(child_path.read_bytes()).hexdigest(),
        "frozen_scenes_sha256": scenes_sha,
    }
    assert {len(rows) for rows in package["arms"].values()} == {576}
    assert package["pilot_schedule"]["optimizer_steps"] == 72
    assert validate_support_package(package)["pilot_schedule"] == package["pilot_schedule"]


def test_common_freeze_script_derives_study_a_support_bins_and_roles(tmp_path: Path) -> None:
    scenes_path = tmp_path / "scenes.jsonl"
    study_a_path = tmp_path / "study-a.jsonl"
    scene_rows: list[str] = []
    support_rows: list[str] = []
    for index in range(96):
        scene = _rl_scene(
            scene_id=f"rl-{index:03d}",
            family=("known_value", "pair_sum", "trend")[index % 3],
            fiber_size=(1, 3, 6)[index % 3],
        )
        scene.pop("policy_support")
        scene["fiber_bin"] = "phase2a_child_bin"
        scene_rows.append(json.dumps(scene, sort_keys=True) + "\n")
        support_rows.append(
            json.dumps(
                {
                    "checkpoint": "T",
                    "graph_axis": "canonical",
                    "source_scene_id": scene["scene_id"],
                    "exact_recovery_probability": index / 95,
                },
                sort_keys=True,
            )
            + "\n"
        )
    scenes_path.write_text("".join(scene_rows))
    study_a_path.write_text("".join(support_rows))
    output = tmp_path / "common-space.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/v5/08_freeze_common_space_rl.py"),
            "--execute",
            "--input-jsonl",
            str(scenes_path),
            "--study-a-rows",
            str(study_a_path),
            "--b3-initialization-sha256",
            "a" * 64,
            "--b2-initialization-sha256",
            "b" * 64,
            "--base-initialization-sha256",
            "c" * 64,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    package = json.loads(output.read_text())
    assert package["role_counts"] == {"rl_train": 72, "rl_eval": 24}
    assert package["scenes"][0]["policy_support"] == 0.0
    assert {scene["support_bin"] for scene in package["scenes"]} == {
        "low",
        "medium",
        "high",
    }
    assert {scene["fiber_bin"] for scene in package["scenes"]} == {
        "singleton",
        "multi_2_4",
        "multi_5_plus",
    }


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


def test_advisor_script_accepts_native_study_summaries_end_to_end(tmp_path: Path) -> None:
    inputs = {
        "b": ("support-results", _study_b_summary()),
        "a": ("confirmation-results", _study_a_summary()),
        "c": ("reward-results", _study_c_summary(triggered=True)),
    }
    arguments: list[str] = []
    for study, (option, payload) in inputs.items():
        root = tmp_path / f"study-{study}"
        root.mkdir()
        path = root / "summary.json"
        path.write_text(json.dumps(payload))
        arguments.extend([f"--{option}", str(path)])
        (root / "raw.jsonl").write_text(json.dumps({"study": study}) + "\n")
        arguments.extend([f"--study-{study}-root", str(root)])
    output = tmp_path / "advisor"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/v5/11_build_advisor_packet.py"),
            "--execute",
            "--output",
            str(output),
            *arguments,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert {path.name for path in output.iterdir()} == {
        "QWEN_V5_PILOT_RESULT_FACTS.md",
        "qwen_v5_pilot_raw_rows.tar.gz",
        "sha256_manifest.json",
    }
    facts = (output / "QWEN_V5_PILOT_RESULT_FACTS.md").read_text()
    assert "V5_STUDY_A_EXECUTED" in facts
    assert "STUDY_B_SINGLE_SEED_COMPLETE" in facts
    assert "STUDY_C_DIAGNOSTICS_COMPLETE" in facts
