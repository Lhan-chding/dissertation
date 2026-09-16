"""Pure metadata contracts; synthetic receipts never execute or certify a GPU run."""

import copy
import json
from pathlib import Path

import pytest

from src.modeling_v4.config import digest
from src.modeling_v4.phase_schedule import (
    append_phase_extension,
    build_confirmation_extension,
    build_development_extension,
    build_tracking_extension,
    freeze_calibration,
    freeze_selection,
    registered_matrix,
    validate_first_map_completion,
    validate_selection,
)


def bound(name):
    return {"path": "/fixture_only/" + name, "sha256": digest(name)}


@pytest.fixture
def config():
    path = Path(__file__).parents[2] / "configs/modeling_v4/protocol.json"
    return json.loads(path.read_text())


def evidence(config, phase, *, selection=None):
    roles = config["qwen"]["seed_roles"]
    role = {"C": "development", "D_DEVELOPMENT": "development", "D_CALIBRATION": "calibration"}[
        phase
    ]
    seeds = roles[role][:2] if phase == "C" else roles[role]
    arms = ["X_BASE"] if phase == "C" else ["X_BASE", "X_VALID"]
    result = {
        "schema": "ssvc-v4-phase-evidence-1",
        "phase": phase,
        "status": "COMPLETE",
        "config_hash": digest(config),
        "execution_kind": "REAL_CUDA_MODEL",
        "campaign_id": "fixture-campaign",
        "test_opened": False,
        "bridge_receipt": bound("bridge.json"),
        "sources": [],
        "origins": [],
    }
    for seed in seeds:
        for arm in arms:
            result["sources"].append(
                {
                    "seed": seed,
                    "arm": arm,
                    "steps": 128,
                    "status": "COMPLETED",
                    "receipt": bound(f"source{seed}{arm}"),
                }
            )
        for step in (32, 96):
            result["origins"].append(
                {
                    "seed": seed,
                    "arm": "X_BASE",
                    "step": step,
                    "calibration_banks": 32
                    if phase == "C"
                    else 96
                    if role == "development"
                    else selection["m"],
                    "query_banks": 16 if phase == "C" else 24,
                    "prompts": 24 if phase == "C" else 72,
                    "draws": 256 if phase == "C" else 1024,
                    "collection_status": "COMPLETED",
                    "evaluation_status": "COMPLETE",
                    "collection_receipt": bound(f"collection{seed}{step}"),
                    "fit_receipt": bound(f"fit{seed}{step}"),
                    "evaluation_receipt": bound(f"evaluation{seed}{step}"),
                    "fitted_model_ids": ["FULL_DUAL_RIDGE"],
                    "fit_scope": "calibration_banks",
                    "evaluation_scope": "heldout_query_banks",
                    "all_queries_retained": True,
                }
            )
    if phase == "D_DEVELOPMENT":
        result["whole_seed_cv"] = [
            {
                "train_seeds": [s for s in seeds if s != heldout],
                "validation_seeds": [heldout],
                "receipt": bound(f"cv{heldout}"),
            }
            for heldout in seeds
        ]
    if selection:
        result["selection_hash"] = selection["selection_hash"]
    return result


def selected(config):
    return freeze_selection(
        config,
        evidence(config, "C"),
        evidence(config, "D_DEVELOPMENT"),
        m=32,
        primary_models=[
            {
                "id": "raw",
                "model": "FULL_DUAL_RIDGE",
                "representation": "R2",
                "settings": {"alpha": 1e-5},
            },
            {
                "id": "effective",
                "model": "FULL_EFFECTIVE_WEIGHT_RIDGE",
                "representation": "R3",
                "settings": {"alpha": 1e-5},
            },
        ],
        gradient_reference="FULL_SCORE_JVP",
    )


def test_exact_registered_matrix(config):
    matrix = registered_matrix(config)
    assert len(matrix["sources"]) == 24
    assert len(matrix["origins"]) == 30
    assert matrix["source_optimizer_updates"] == 3072
    assert matrix["source_training_outputs"] == 98304
    shift = [row for row in matrix["origins"] if row["arm"] == "X_VALID"]
    assert len(shift) == 6
    assert {row["role"] for row in shift} == {"test"}
    assert {row["step"] for row in shift} == {96}


def test_first_map_needs_actual_fitted_evaluation_not_raw_collection(config):
    receipt = evidence(config, "C")
    validate_first_map_completion(config, receipt)
    for change in ("fit_receipt", "evaluation_receipt"):
        bad = copy.deepcopy(receipt)
        del bad["origins"][0][change]
        with pytest.raises(ValueError):
            validate_first_map_completion(config, bad)
    receipt["origins"] = receipt["origins"][:3]
    with pytest.raises(ValueError, match="matrix"):
        validate_first_map_completion(config, receipt)


def test_negative_scientific_result_does_not_block_D(config):
    receipt = evidence(config, "C")
    receipt["scientific_status"] = "METHODS_WORSE_THAN_DIRECT"
    result = build_development_extension(config, receipt, workers=3)
    maps = [row for row in result["tasks"] if row["kind"] == "map"]
    assert len(maps) == 6
    assert {(r["calibration_banks"], r["query_banks"]) for r in maps} == {(96, 24)}
    assert len([row for row in maps if row.get("reuse_prefix_task")]) == 4
    assert {row["worker"] for row in result["tasks"]} == {0, 1, 2}
    assert result["scientific_qualification"] == "NOT_INFERRED"


def test_exact_seed_roles_and_no_caller_mutation(config):
    original = copy.deepcopy(config)
    first = evidence(config, "C")
    previous = copy.deepcopy(first)
    build_development_extension(config, first)
    assert config == original and first == previous
    config["qwen"]["seed_roles"]["test"][0] = 45001
    with pytest.raises(ValueError, match=r"registered|seed"):
        registered_matrix(config)


def test_selection_freezes_one_m_and_preserves_baselines(config):
    selection = selected(config)
    validate_selection(config, selection)
    assert selection["m"] == 32
    assert selection["test_opened"] is False
    assert {"ZERO", "DIRECT_MEASURE", "FULL_DUAL_RIDGE", "LOW_RANK_COMPRESSION"} <= set(
        selection["mandatory_baselines"]
    )
    assert selection["compression_ranks"] == [2, 8, 32, "FULL"]
    altered = copy.deepcopy(selection)
    altered["m"] = 48
    with pytest.raises(ValueError, match="hash"):
        validate_selection(config, altered)


@pytest.mark.parametrize("m", [True, 0, 16, [24, 32]])
def test_selection_rejects_nonregistered_or_multiple_m(config, m):
    with pytest.raises(ValueError, match="m"):
        freeze_selection(
            config,
            evidence(config, "C"),
            evidence(config, "D_DEVELOPMENT"),
            m=m,
            primary_models=[],
            gradient_reference="FULL_SCORE_JVP",
        )


def test_selection_rejects_seed_leakage_and_test_use(config):
    dev = evidence(config, "D_DEVELOPMENT")
    dev["whole_seed_cv"][0]["train_seeds"].append(41001)
    with pytest.raises(ValueError, match=r"whole|seed"):
        freeze_selection(
            config,
            evidence(config, "C"),
            dev,
            m=32,
            primary_models=[],
            gradient_reference="FULL_SCORE_JVP",
        )
    dev = evidence(config, "D_DEVELOPMENT")
    dev["test_opened"] = True
    with pytest.raises(ValueError, match="test"):
        freeze_selection(
            config,
            evidence(config, "C"),
            dev,
            m=32,
            primary_models=[],
            gradient_reference="FULL_SCORE_JVP",
        )


def test_calibration_and_test_separate_empirical_gate(config):
    selection = selected(config)
    cal = build_confirmation_extension(config, evidence(config, "C"), selection, role="calibration")
    assert len([r for r in cal["tasks"] if r["kind"] == "source"]) == 6
    assert len([r for r in cal["tasks"] if r["kind"] == "map"]) == 6
    with pytest.raises(ValueError, match="calibration"):
        build_confirmation_extension(config, evidence(config, "C"), selection, role="test")
    frozen = freeze_calibration(
        config,
        selection,
        evidence(config, "D_CALIBRATION", selection=selection),
        thresholds={"empirical_error_q95": 0.001},
    )
    assert frozen["cross_seed_95"] == "NOT_CERTIFIED"
    test = build_confirmation_extension(
        config, evidence(config, "C"), selection, role="test", calibration=frozen
    )
    maps = [r for r in test["tasks"] if r["kind"] == "map"]
    assert len(maps) == 18
    assert {(r["calibration_banks"], r["query_banks"]) for r in maps} == {(32, 24)}
    assert all(r["evaluation_only"] and not r["selection_uses_query_labels"] for r in maps)
    assert test["calibration_used_as_model_training_data"] is False


def test_calibration_cannot_change_model_selection(config):
    selection = selected(config)
    cal = evidence(config, "D_CALIBRATION", selection=selection)
    cal["selection_hash"] = "different"
    with pytest.raises(ValueError, match="selection"):
        freeze_calibration(config, selection, cal, thresholds={"q95": 0.001})


def test_E_missing_evidence_waits_and_keeps_path(config):
    result = build_tracking_extension(config, selected(config), None)
    assert result["status"] == "WAITING_FOR_POINT_RESPONSE_EVIDENCE"
    assert result["tasks"] == []
    assert result["preserve_source_checkpoints"] == list(range(64, 81))
    assert result["online_control"] is False


def test_E_verified_unresolved_is_skipped(config, monkeypatch):
    selection = selected(config)
    receipt = {
        "schema": "ssvc-v4-point-response-evidence-1",
        "status": "POINT_RESPONSE_UNRESOLVED",
        "config_hash": digest(config),
        "selection_hash": selection["selection_hash"],
        "campaign_id": selection["campaign_id"],
        "execution_kind": "REAL_CUDA_MODEL",
        "assessments": [
            {
                "status": "UNRESOLVED",
                "evaluation_receipt": bound("eval"),
                "resolution_receipt": bound("resolution"),
            }
        ],
    }
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.modeling_v4.point_response_evidence.verify_point_response_evidence",
            lambda *args: receipt,
        )
        assert (
            build_tracking_extension(config, selection, receipt)["status"]
            == "SKIPPED_POINT_RESPONSE_UNRESOLVED"
        )
    del receipt["assessments"][0]["resolution_receipt"]
    with pytest.raises(ValueError, match="derived point-response"):
        build_tracking_extension(config, selection, receipt)


def test_E_requires_real_bound_resolution_and_is_dev_offline_only(config, monkeypatch):
    selection = selected(config)
    receipt = {
        "schema": "ssvc-v4-point-response-evidence-1",
        "status": "POINT_RESPONSE_RESOLVED",
        "config_hash": digest(config),
        "selection_hash": selection["selection_hash"],
        "campaign_id": selection["campaign_id"],
        "execution_kind": "REAL_CUDA_MODEL",
        "assessments": [
            {
                "model_id": "raw",
                "status": "RESOLVED",
                "nonalias_queries": 24,
                "reference_status": "RESOLVED",
                "working_resolution": 0.001,
                "criterion": "independent query assessment at declared resolution",
                "evaluation_receipt": bound("point_eval"),
                "resolution_receipt": bound("point_resolution"),
            }
        ],
    }
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.modeling_v4.point_response_evidence.verify_point_response_evidence",
            lambda *args: receipt,
        )
        result = build_tracking_extension(config, selection, receipt)
    assert len(result["tasks"]) == 3
    for row in result["tasks"]:
        assert row["arm"] == "X_BASE" and row["role"] == "development"
        assert row["anchor"] == 64 and row["horizons"] == [1, 2, 4, 8, 16]
        assert row["reference_rng_independent"] and not row["online_control"]
    del receipt["assessments"][0]["resolution_receipt"]
    with pytest.raises(ValueError, match="derived point-response"):
        build_tracking_extension(config, selection, receipt)


def test_phase_extension_preserves_original_campaign_and_C_tasks(config):
    from src.r4_inputs import STRATA

    first = evidence(config, "C")
    parent = {
        "schema": "ssvc-v4-task-list-1",
        "config": config,
        "workers": 2,
        "campaign_id": first["campaign_id"],
        "root": "/fixture_only/campaign",
        "stages": ["B", "C"],
        "inputs": {
            "panels": {
                "observation": [
                    {"family": family, "interface": interface, "prompt_id": f"p{i}"}
                    for i, (family, interface) in enumerate(STRATA)
                ]
            }
        },
        "tasks": [
            {
                "id": f"source_{seed}_X_BASE",
                "kind": "source",
                "stage": "C",
                "seed": seed,
                "arm": "X_BASE",
                "worker": i,
                "depends_on": ["B_MERGED"],
            }
            for i, seed in enumerate((41001, 41002))
        ],
    }
    parent["task_list_hash"] = digest(parent)
    before = copy.deepcopy(parent)
    child = append_phase_extension(parent, build_development_extension(config, first))
    assert parent == before
    assert child["campaign_id"] == parent["campaign_id"]
    assert child["parent_task_list_hash"] == parent["task_list_hash"]
    assert child["tasks"][:2] == parent["tasks"]
    assert len(child["tasks"]) == 12
    assert len({row["id"] for row in child["tasks"]}) == 12
    for task in child["tasks"]:
        if task["kind"] == "map":
            assert len(task["derivative_prompt_subset"]) == 6
            assert task["finite_difference_steps"] == [0.25, 0.5, 1.0]
            assert task["directional_bank_ids"] == [f"calibration_{i:03d}" for i in range(4)]


def test_rehashed_selection_cannot_replace_model_with_ZERO(config):
    selection = selected(config)
    selection["primary_models"][0]["model"] = "ZERO"
    selection["selection_hash"] = digest(
        {k: v for k, v in selection.items() if k != "selection_hash"}
    )
    with pytest.raises(ValueError, match="model"):
        validate_selection(config, selection)
