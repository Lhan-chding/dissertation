from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compensability_v5.study_c2 import training_runtime as runtime
from compensability_v5.study_c2.io import read_jsonl
from compensability_v5.study_c2.schemas import build_reward_arm_configs, validate_study_c2_config


def _contract() -> dict[str, object]:
    return validate_study_c2_config(
        {
            "schema_version": 2,
            "seed": 2026082401,
            "value_domain": [2, 18],
            "group_candidates": [8, 16, 32],
            "support_rollouts_per_prompt": 64,
            "training": {
                "precision": "bf16",
                "learning_rate": 1.0e-6,
                "kl_beta": 0.04,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "temperature": 0.7,
                "top_p": 1.0,
                "max_prompt_length": 512,
                "max_completion_length": 16,
                "optimizer": "adamw_torch",
                "epochs": 1,
            },
            "evaluation": {
                "sampled_rollouts": 16,
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 2026082403,
            },
        }
    )


def _row(scene_id: str = "c2-train-0000-collision") -> dict[str, object]:
    return {
        "schema_version": 2,
        "split": "train",
        "scene_id": scene_id,
        "pair_id": "c2-train-0000",
        "condition": "collision",
        "family": "cross_series",
        "prompt": "Recover the world.",
        "prompt_sha256": "a" * 64,
        "truth": [3, 8, 5, 18],
        "observation": [3, 8, 5, 17],
        "operation": {"operator": "sum", "indices": [0, 1]},
        "gold_answer": 11,
        "observed_answer": 11,
        "observed_is_answer_equivalent": True,
        "reward_identifiable": False,
    }


def test_training_rows_are_closed_paired_and_keep_frozen_order() -> None:
    rows = tuple(
        {
            **_row(f"c2-train-{index:04d}-{condition}"),
            "pair_id": f"c2-train-{index:04d}",
            "condition": condition,
        }
        for index in range(96)
        for condition in ("collision", "separating")
    )
    rows += ({**_row("support"), "split": "support_audit"},)

    selected = runtime.select_training_rows(rows)

    assert len(selected) == 192
    assert [row["scene_id"] for row in selected[:4]] == [
        "c2-train-0000-collision",
        "c2-train-0000-separating",
        "c2-train-0001-collision",
        "c2-train-0001-separating",
    ]
    assert runtime.expected_optimizer_steps(selected, group_size=8) == 192

    with pytest.raises(ValueError, match="192"):
        runtime.select_training_rows(rows[:-2])
    with pytest.raises(ValueError, match="paired"):
        runtime.select_training_rows(tuple(dict(row, condition="collision") for row in rows[:192]))


def test_traced_reward_uses_first_line_and_preserves_both_labels(tmp_path: Path) -> None:
    arm = build_reward_arm_configs(_contract(), initialization_hash="b" * 64)[1]
    trace = tmp_path / "raw_reward_trace.jsonl"
    reward = runtime.build_traced_reward(
        arm_config=arm,
        training_rows=(_row(),),
        trace_path=trace,
        group_size=8,
    )
    completions = [
        "3,8,5,18\nignored explanation",
        "3,8,5,17\nignored explanation",
        "2,2,2,2",
        "not an action",
    ] * 2

    selected = reward(
        completions,
        scene_id=["c2-train-0000-collision"],
        trainer_state=SimpleNamespace(global_step=7),
    )

    assert selected == [1.0, 0.0, 0.0, 0.0] * 2
    rows = read_jsonl(trace)
    assert [row["kind"] for row in rows[:4]] == ["X", "S", "F", "U"]
    assert rows[0]["answer_reward"] == rows[0]["state_reward"] == 1
    assert rows[1]["answer_reward"] == 1 and rows[1]["state_reward"] == 0
    assert all(row["group_index"] == 0 for row in rows)
    assert all(row["trainer_step"] == 7 for row in rows)


def test_grpo_kwargs_freeze_one_epoch_k8_checkpoints_and_newline_stop() -> None:
    arm = build_reward_arm_configs(_contract(), initialization_hash="b" * 64)[0]
    supported = {
        "output_dir",
        "learning_rate",
        "num_train_epochs",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_prompt_length",
        "max_completion_length",
        "bf16",
        "fp16",
        "temperature",
        "top_p",
        "top_k",
        "beta",
        "use_vllm",
        "gradient_checkpointing",
        "logging_steps",
        "logging_first_step",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "report_to",
        "remove_unused_columns",
        "seed",
        "data_seed",
        "optim",
        "shuffle_dataset",
        "generation_kwargs",
    }

    kwargs = runtime.build_grpo_config_kwargs(
        arm_config=arm,
        output_dir=Path("out"),
        group_size=8,
        eos_token_id=99,
        newline_token_id=13,
        supported_parameters=supported,
    )

    assert kwargs["num_train_epochs"] == 1
    assert "max_steps" not in kwargs
    assert kwargs["num_generations"] == 8
    assert kwargs["shuffle_dataset"] is False
    assert kwargs["save_steps"] == 48 and kwargs["save_total_limit"] == 4
    assert kwargs["generation_kwargs"] == {"eos_token_id": [99, 13]}
    assert kwargs["max_completion_length"] == 16

    with pytest.raises(RuntimeError, match="generation_kwargs"):
        runtime.build_grpo_config_kwargs(
            arm_config=arm,
            output_dir=Path("out"),
            group_size=8,
            eos_token_id=99,
            newline_token_id=13,
            supported_parameters=supported - {"generation_kwargs"},
        )


def test_completion_token_truncation_is_newline_or_eos_anchored() -> None:
    completions, logprobs = runtime.truncate_first_line_token_ids(
        completion_ids=[[1, 2, 13, 9, 9], [3, 4, 99, 9], [5, 6]],
        logprobs=[[0.1] * 5, [0.2] * 4, [0.3] * 2],
        newline_token_id=13,
        eos_token_id=99,
    )
    assert completions == [[1, 2, 13], [3, 4, 99], [5, 6]]
    assert logprobs == [[0.1] * 3, [0.2] * 3, [0.3] * 2]

    with pytest.raises(RuntimeError, match="align"):
        runtime.truncate_first_line_token_ids(
            completion_ids=[[1, 13]],
            logprobs=[[0.1]],
            newline_token_id=13,
            eos_token_id=99,
        )


def test_training_progress_callback_prints_every_optimizer_step(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = runtime.TrainingProgressCallback("C2_answer_reward", total_steps=192)
    control = object()
    assert callback.on_step_end(None, SimpleNamespace(global_step=48), control) is control
    assert "optimizer step 48/192" in capsys.readouterr().out


def test_stage25_execution_contract_is_fail_closed() -> None:
    payload = {
        "schema_version": 2,
        "status": "STUDY_C2_STAGE25_EXECUTION_CONTRACT_FROZEN",
        "stage24_per_group_sha256": "1" * 64,
        "stage24_summary_sha256": "2" * 64,
        "stage24_manifest_sha256": "3" * 64,
        "stage24_execution_contract_sha256": "4" * 64,
        "fiber_rows_sha256": "5" * 64,
        "config_sha256": "6" * 64,
        "package_lock_sha256": "7" * 64,
        "b3_adapter_sha256": "8" * 64,
        "model_snapshot_sha256": runtime.MODEL_SNAPSHOT_SHA256,
        "selected_k": 8,
        "training_prompt_count": 192,
        "shared_gradient_group_count": 768,
        "reward_hamming_distance": 635,
        "continue_to_main_rl": True,
    }
    runtime.validate_stage25_execution_contract(payload)
    with pytest.raises(ValueError, match="drifted"):
        runtime.validate_stage25_execution_contract(dict(payload, selected_k=16))
    with pytest.raises(ValueError, match="invalid config_sha256"):
        runtime.validate_stage25_execution_contract(dict(payload, config_sha256="short"))


def test_group_diagnostics_rejects_incomplete_or_nonfinite_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "group_index": 0,
                    "scene_id": "scene",
                    "kind": kind,
                    "answer_reward": int(kind in {"X", "S"}),
                    "state_reward": int(kind == "X"),
                    "reward": float(kind == "X"),
                }
            )
            + "\n"
            for kind in "XSFUUUUU"
        ),
        encoding="utf-8",
    )
    diagnostics = runtime.build_training_group_diagnostics(
        trace_path=trace, group_size=8, expected_group_count=1
    )
    assert diagnostics[0]["counts"] == {"X": 1, "S": 1, "F": 1, "U": 5}
    assert diagnostics[0]["reward_hamming_distance"] == 1

    with pytest.raises(ValueError, match="group size"):
        runtime.build_training_group_diagnostics(
            trace_path=trace, group_size=7, expected_group_count=1
        )
