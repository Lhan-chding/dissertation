from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from compensability_v5.study_c2.shared_gradient_runtime import (
    SHARED_GRADIENT_ACK,
    group_support_rows,
    preflight_shared_gradient,
    run_shared_gradient_audit,
    summarize_shared_gradient_audit,
)

from compensability_v5.study_c2.gradient_audit import autograd_gradient_diagnostics


def _support_row(
    scene: str,
    rollout: int,
    kind: str,
    *,
    condition: str = "collision",
) -> dict[str, object]:
    return {
        "scene_id": scene,
        "pair_id": scene.rsplit("-", 1)[0],
        "condition": condition,
        "family": "cross_series",
        "rollout_index": rollout,
        "kind": kind,
        "state_reward": int(kind == "X"),
        "answer_reward": int(kind in {"X", "S"}),
        "prompt_sha256": "a" * 64,
        "token_ids": [rollout + 1],
    }


def test_group_support_rows_preserves_closed_scene_local_k8_groups() -> None:
    rows = [
        *[_support_row("pair-a-collision", index, "XSFU"[index % 4]) for index in range(16)],
        *[
            _support_row("pair-b-separating", index, "S", condition="separating")
            for index in range(8)
        ],
    ]

    groups = group_support_rows(rows, group_size=8)

    assert len(groups) == 3
    assert [row["rollout_index"] for row in groups[0]] == list(range(8))
    assert [row["rollout_index"] for row in groups[1]] == list(range(8, 16))
    assert {row["scene_id"] for row in groups[2]} == {"pair-b-separating"}

    drifted = [dict(row) for row in rows]
    drifted[7]["rollout_index"] = 99
    with pytest.raises(ValueError, match="rollout order"):
        group_support_rows(drifted, group_size=8)


def test_autograd_gradient_diagnostics_uses_the_same_shared_log_probabilities() -> None:
    first = torch.nn.Parameter(torch.tensor(0.5))
    second = torch.nn.Parameter(torch.tensor(-0.25))
    log_probabilities = torch.stack((first, second, first + second, first - second))

    result = autograd_gradient_diagnostics(
        log_probabilities=log_probabilities,
        trainable_parameters=(first, second),
        state_rewards=(1, 0, 0, 0),
        answer_rewards=(1, 1, 0, 0),
    )

    assert result["reward_hamming_distance"] == 1
    assert result["gradient_state_norm"] > 0
    assert result["gradient_answer_norm"] > 0
    assert result["gradient_difference_norm"] > 0
    assert -1.0 <= result["gradient_cosine"] <= 1.0
    assert result["finite"] is True


def test_shared_gradient_summary_reports_hard_gate_and_required_strata() -> None:
    rows = [
        {
            "condition": "collision",
            "family": "cross_series",
            "counts": {"X": 1, "S": 1, "F": 2, "U": 4},
            "reward_hamming_distance": 1,
            "RDGR": True,
            "ESGR": True,
            "gradient_state_norm": 2.0,
            "gradient_answer_norm": 3.0,
            "gradient_difference_norm": 1.5,
            "gradient_cosine": 0.5,
        },
        {
            "condition": "separating",
            "family": "cross_series",
            "counts": {"X": 0, "S": 0, "F": 4, "U": 4},
            "reward_hamming_distance": 0,
            "RDGR": False,
            "ESGR": False,
            "gradient_state_norm": 1.0,
            "gradient_answer_norm": 1.0,
            "gradient_difference_norm": 0.0,
            "gradient_cosine": 1.0,
        },
    ]

    summary = summarize_shared_gradient_audit(rows, group_size=8)

    assert summary["status"] == "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED"
    assert summary["continue_to_main_rl"] is True
    assert summary["reward_hamming_distance"] == 1
    assert summary["RDGR_group_count"] == 1
    assert summary["ESGR_group_count"] == 1
    assert set(summary["by_condition"]) == {"collision", "separating"}
    assert summary["counts"] == {"X": 1, "S": 1, "F": 6, "U": 8}

    null_rows = [dict(row, reward_hamming_distance=0, gradient_difference_norm=0.0) for row in rows]
    null = summarize_shared_gradient_audit(null_rows, group_size=8)
    assert null["status"] == "STUDY_C2_SHARED_GRADIENT_CONTRAST_NOT_ESTIMABLE"
    assert null["continue_to_main_rl"] is False


def test_preflight_binds_the_returned_stage23_hashes_and_selected_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from compensability_v5.study_c2 import shared_gradient_runtime as runtime

    raw = tmp_path / "raw_rows.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "REWARD_CONTRAST_IDENTIFIED",
                "counts": {"X": 143, "S": 635, "F": 655, "U": 4711},
                "rollout_count": 6144,
                "k_selection": {"selected_k": 8},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "STUDY_C2_FROZEN_SUPPORT_COMPLETE",
                "b3_adapter_sha256": "b" * 64,
                "raw_rows_sha256": "1" * 64,
                "summary_sha256": "2" * 64,
                "rollout_count": 6144,
            }
        ),
        encoding="utf-8",
    )
    fibers = tmp_path / "fibers.jsonl"
    fibers.write_text("{}\n", encoding="utf-8")
    package_lock = tmp_path / "lock.yaml"
    package_lock.write_text("schema_version: 1\n", encoding="utf-8")
    contract = tmp_path / "execution_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "STUDY_C2_STAGE24_EXECUTION_CONTRACT_FROZEN",
                "support_raw_rows_sha256": "1" * 64,
                "support_summary_sha256": "2" * 64,
                "support_manifest_sha256": "3" * 64,
                "fiber_rows_sha256": "4" * 64,
                "config_sha256": "5" * 64,
                "package_lock_sha256": "6" * 64,
                "b3_adapter_sha256": "b" * 64,
                "selected_k": 8,
                "rollout_count": 6144,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime, "SUPPORT_RAW_ROWS", raw)
    monkeypatch.setattr(runtime, "SUPPORT_SUMMARY", summary)
    monkeypatch.setattr(runtime, "SUPPORT_MANIFEST", manifest)
    monkeypatch.setattr(runtime, "FIBER_ROWS", fibers)
    monkeypatch.setattr(runtime, "PACKAGE_LOCK", package_lock)
    monkeypatch.setattr(
        runtime,
        "sha256_file",
        lambda path: {
            raw: "1" * 64,
            summary: "2" * 64,
            manifest: "3" * 64,
            fibers: "4" * 64,
            tmp_path / "config.yaml": "5" * 64,
            package_lock: "6" * 64,
        }[path],
    )
    monkeypatch.setattr(runtime, "load_contract", lambda path: {"group_candidates": (8, 16, 32)})
    monkeypatch.setattr(runtime, "verify_runtime_package_lock", lambda path: {"verified": True})
    monkeypatch.setattr(runtime, "_require_offline_cuda", lambda: None)
    monkeypatch.setattr(runtime, "require_server_model", lambda: None)
    monkeypatch.setattr(runtime, "tree_sha256", lambda path: "b" * 64)

    result = preflight_shared_gradient(
        config_path=tmp_path / "config.yaml",
        execution_contract_path=contract,
        b3_adapter=tmp_path / "B3",
        b3_sha256="b" * 64,
    )

    assert result["status"] == "STUDY_C2_SHARED_GRADIENT_PREFLIGHT_OK"
    assert result["group_size"] == 8
    assert result["group_count"] == 768
    assert result["gpu_invoked"] is False


def test_execute_requires_exact_acknowledgement_before_loading_the_model(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="acknowledgement"):
        run_shared_gradient_audit(
            config_path=tmp_path / "config.yaml",
            execution_contract_path=tmp_path / "contract.json",
            b3_adapter=tmp_path / "B3",
            b3_sha256="b" * 64,
            acknowledgement=SHARED_GRADIENT_ACK + "_WRONG",
        )
