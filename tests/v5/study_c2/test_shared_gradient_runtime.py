from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compensability_v5.study_c2.gradient_audit import (
    autograd_gradient_diagnostics,
    centered_advantages,
    shared_gradient_diagnostics,
)
from compensability_v5.study_c2.shared_gradient_runtime import (
    SHARED_GRADIENT_ACK,
    group_support_rows,
    preflight_shared_gradient,
    run_shared_gradient_audit,
    summarize_shared_gradient_audit,
)

MODEL_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"


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

    with pytest.raises(ValueError, match="at least two"):
        group_support_rows(rows, group_size=1)
    with pytest.raises(ValueError, match="cannot be empty"):
        group_support_rows([], group_size=8)
    with pytest.raises(ValueError, match="scene_id"):
        group_support_rows([{}], group_size=8)
    with pytest.raises(ValueError, match="do not divide"):
        group_support_rows(rows[:7], group_size=8)


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

    zero = autograd_gradient_diagnostics(
        log_probabilities=log_probabilities,
        trainable_parameters=(first, second),
        state_rewards=(0, 0, 0, 0),
        answer_rewards=(0, 0, 0, 0),
    )
    assert zero["gradient_state_norm"] == 0.0
    assert zero["gradient_answer_norm"] == 0.0
    assert zero["gradient_cosine"] == 1.0

    state_only = autograd_gradient_diagnostics(
        log_probabilities=log_probabilities,
        trainable_parameters=(first, second),
        state_rewards=(1, 0, 0, 0),
        answer_rewards=(0, 0, 0, 0),
    )
    assert state_only["gradient_state_norm"] > 0
    assert state_only["gradient_answer_norm"] == 0.0
    assert state_only["gradient_cosine"] == 0.0


def test_autograd_gradient_diagnostics_rejects_malformed_inputs() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        autograd_gradient_diagnostics(
            log_probabilities=torch.tensor([1.0]),
            trainable_parameters=(parameter,),
            state_rewards=(1,),
            answer_rewards=(1,),
        )
    with pytest.raises(ValueError, match="trainable parameters"):
        autograd_gradient_diagnostics(
            log_probabilities=torch.stack((parameter, parameter)),
            trainable_parameters=(),
            state_rewards=(1, 0),
            answer_rewards=(1, 0),
        )
    with pytest.raises(ValueError, match="finite"):
        autograd_gradient_diagnostics(
            log_probabilities=torch.stack((parameter, parameter * torch.tensor(float("nan")))),
            trainable_parameters=(parameter,),
            state_rewards=(1, 0),
            answer_rewards=(1, 0),
        )


def test_numpy_gradient_diagnostics_cover_centering_and_alignment() -> None:
    centered = centered_advantages((1, 3, 5))
    assert centered.tolist() == [-2.0, 0.0, 2.0]

    payload = shared_gradient_diagnostics(
        state_rewards=(1, 0, 0),
        answer_rewards=(1, 1, 0),
        score_vectors=((1, 0), (0, 1), (1, 1)),
    )
    assert payload["reward_hamming_distance"] == 1
    assert payload["gradient_state_norm"] > 0
    assert payload["gradient_answer_norm"] > 0
    assert payload["gradient_difference_norm"] > 0
    assert -1.0 <= float(payload["gradient_cosine"]) <= 1.0
    assert len(payload["state_gradient"]) == 2
    assert len(payload["answer_gradient"]) == 2

    zero = shared_gradient_diagnostics(
        state_rewards=(0, 0, 0),
        answer_rewards=(0, 0, 0),
        score_vectors=((1, 0), (0, 1), (1, 1)),
    )
    assert zero["gradient_state_norm"] == 0.0
    assert zero["gradient_answer_norm"] == 0.0
    assert zero["gradient_cosine"] == 1.0

    state_only = shared_gradient_diagnostics(
        state_rewards=(1, 0, 0),
        answer_rewards=(0, 0, 0),
        score_vectors=((1, 0), (0, 1), (1, 1)),
    )
    assert state_only["gradient_state_norm"] > 0
    assert state_only["gradient_answer_norm"] == 0.0
    assert state_only["gradient_cosine"] == 0.0

    with pytest.raises(ValueError, match="at least two finite values"):
        centered_advantages((1,))
    with pytest.raises(ValueError, match="at least two finite values"):
        centered_advantages((1, float("nan")))
    with pytest.raises(ValueError, match="score vectors and rewards must align"):
        shared_gradient_diagnostics(
            state_rewards=(1, 0, 0),
            answer_rewards=(1, 0, 0),
            score_vectors=((1, 0), (0, 1)),
        )


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

    from compensability_v5.study_c2 import shared_gradient_runtime as runtime

    with pytest.raises(ValueError, match="cannot be empty"):
        runtime._aggregate_gradient_rows([])
    with pytest.raises(ValueError, match="malformed"):
        runtime._aggregate_gradient_rows([dict(rows[0], counts={})])
    with pytest.raises(ValueError, match="non-negative"):
        runtime._aggregate_gradient_rows([dict(rows[0], counts={"X": -1, "S": 1, "F": 2, "U": 6})])
    with pytest.raises(ValueError, match="finite"):
        runtime._aggregate_gradient_rows([dict(rows[0], gradient_cosine=float("nan"))])
    with pytest.raises(ValueError, match="condition and family"):
        summarize_shared_gradient_audit([dict(rows[0], condition="unknown")], group_size=8)


def test_prompt_and_partial_helpers_reject_provenance_drift() -> None:
    from compensability_v5.study_c2 import shared_gradient_runtime as runtime

    assert runtime._prompt_map([{"scene_id": "scene-a", "prompt": "recover"}]) == {
        "scene-a": "recover"
    }
    with pytest.raises(ValueError, match="scene_id and prompt"):
        runtime._prompt_map([{}])
    with pytest.raises(ValueError, match="duplicate"):
        runtime._prompt_map(
            [
                {"scene_id": "scene-a", "prompt": "recover"},
                {"scene_id": "scene-a", "prompt": "recover again"},
            ]
        )

    group = tuple(_support_row("pair-a-collision", index, "X") for index in range(8))
    valid = {
        "group_index": 0,
        "scene_id": "pair-a-collision",
        "rollout_indices": list(range(8)),
        "finite": True,
    }
    runtime._validate_partial([valid], [group])
    with pytest.raises(ValueError, match="exceeds"):
        runtime._validate_partial([valid, valid], [group])
    with pytest.raises(ValueError, match="drifted"):
        runtime._validate_partial([dict(valid, finite=False)], [group])


def test_execution_contract_validation_is_fail_closed() -> None:
    from compensability_v5.study_c2 import shared_gradient_runtime as runtime

    base = {
        "schema_version": 2,
        "status": "STUDY_C2_STAGE24_EXECUTION_CONTRACT_FROZEN",
        "support_raw_rows_sha256": "1" * 64,
        "support_summary_sha256": "2" * 64,
        "support_manifest_sha256": "3" * 64,
        "fiber_rows_sha256": "4" * 64,
        "config_sha256": "5" * 64,
        "package_lock_sha256": "6" * 64,
        "b3_adapter_sha256": "b" * 64,
        "model_snapshot_sha256": MODEL_SHA256,
        "selected_k": 8,
        "rollout_count": 6144,
        "support_counts": {"X": 143, "S": 635, "F": 655, "U": 4711},
        "support_status": "REWARD_CONTRAST_IDENTIFIED",
    }
    runtime._validate_execution_contract(base)
    with pytest.raises(ValueError, match="contract drifted"):
        runtime._validate_execution_contract(dict(base, selected_k=16))
    with pytest.raises(ValueError, match="invalid config_sha256"):
        runtime._validate_execution_contract(dict(base, config_sha256="short"))
    with pytest.raises(ValueError, match="returned support facts drifted"):
        runtime._validate_execution_contract(
            dict(base, support_counts={"X": 1, "S": 2, "F": 3, "U": 4})
        )


def test_group_log_probabilities_masks_prefix_and_padding_with_live_gradients() -> None:
    from compensability_v5.study_c2 import shared_gradient_runtime as runtime

    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def apply_chat_template(
            self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
        ) -> str:
            assert messages == [{"role": "user", "content": "recover"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "prefix"

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "prefix"
            assert add_special_tokens is False
            return [1, 2]

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.1))
            self.device = torch.device("cpu")

        def forward(
            self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool
        ) -> SimpleNamespace:
            assert attention_mask.shape == input_ids.shape
            assert use_cache is False
            vocabulary = torch.arange(8, dtype=torch.float32)
            logits = self.scale * vocabulary.view(1, 1, -1)
            return SimpleNamespace(logits=logits.expand(*input_ids.shape, 8))

    model = Model()
    result = runtime._group_log_probabilities(
        model,
        Tokenizer(),
        prompt="recover",
        completion_token_ids=((3,), (4, 5)),
        max_prompt_length=8,
        max_completion_length=2,
    )
    assert result.shape == (2,)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_shared_gradient_execution_writes_hash_bound_outputs_without_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from compensability_v5.study_c2 import shared_gradient_runtime as runtime

    support_rows = tuple(
        _support_row("pair-a-collision", index, "XSFU"[index % 4]) for index in range(8)
    )
    support = tmp_path / "support.jsonl"
    support.write_text("".join(json.dumps(row) + "\n" for row in support_rows), encoding="utf-8")
    fibers = tmp_path / "fibers.jsonl"
    fibers.write_text(
        json.dumps({"scene_id": "pair-a-collision", "prompt": "recover"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "shared"
    rows_path = output / "per_group.jsonl"
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    monkeypatch.setattr(runtime, "SUPPORT_RAW_ROWS", support)
    monkeypatch.setattr(runtime, "FIBER_ROWS", fibers)
    monkeypatch.setattr(runtime, "SHARED_GRADIENT_ROOT", output)
    monkeypatch.setattr(runtime, "SHARED_GRADIENT_ROWS", rows_path)
    monkeypatch.setattr(runtime, "SHARED_GRADIENT_SUMMARY", summary_path)
    monkeypatch.setattr(runtime, "SHARED_GRADIENT_MANIFEST", manifest_path)
    monkeypatch.setattr(
        runtime,
        "preflight_shared_gradient",
        lambda **kwargs: {
            "status": "STUDY_C2_SHARED_GRADIENT_PREFLIGHT_OK",
            "group_size": 8,
            "group_count": 1,
            "rollout_count": 8,
            "b3_adapter_sha256": "b" * 64,
            "gpu_invoked": False,
        },
    )

    def evaluator(prompt: str, group: list[dict[str, object]]) -> dict[str, object]:
        assert prompt == "recover"
        assert len(group) == 8
        return {
            "finite": True,
            "counts": {"X": 2, "S": 2, "F": 2, "U": 2},
            "reward_hamming_distance": 2,
            "RDGR": True,
            "ESGR": True,
            "gradient_state_norm": 2.0,
            "gradient_answer_norm": 3.0,
            "gradient_difference_norm": 1.0,
            "gradient_cosine": 0.5,
        }

    result = run_shared_gradient_audit(
        config_path=tmp_path / "config.yaml",
        execution_contract_path=tmp_path / "contract.json",
        b3_adapter=tmp_path / "B3",
        b3_sha256="b" * 64,
        acknowledgement=SHARED_GRADIENT_ACK,
        group_evaluator=evaluator,
    )

    assert result["status"] == "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE"
    assert result["continue_to_main_rl"] is True
    assert result["optimizer_step_invoked"] is False
    assert rows_path.is_file() and summary_path.is_file() and manifest_path.is_file()
    with pytest.raises(RuntimeError, match="overwrite forbidden"):
        run_shared_gradient_audit(
            config_path=tmp_path / "config.yaml",
            execution_contract_path=tmp_path / "contract.json",
            b3_adapter=tmp_path / "B3",
            b3_sha256="b" * 64,
            acknowledgement=SHARED_GRADIENT_ACK,
            group_evaluator=evaluator,
        )


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
                "fiber_rows_sha256": "4" * 64,
                "config_sha256": "5" * 64,
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
                "model_snapshot_sha256": MODEL_SHA256,
                "selected_k": 8,
                "rollout_count": 6144,
                "support_counts": {"X": 143, "S": 635, "F": 655, "U": 4711},
                "support_status": "REWARD_CONTRAST_IDENTIFIED",
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
            contract: "7" * 64,
        }[path],
    )
    monkeypatch.setattr(
        runtime,
        "load_contract",
        lambda path: {
            "group_candidates": (8, 16, 32),
            "training": {"max_prompt_length": 512, "max_completion_length": 16},
        },
    )
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
