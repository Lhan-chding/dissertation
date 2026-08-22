from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compensability_v5.study_c2 import training_backend as backend
from compensability_v5.study_c2 import training_execution as execution
from compensability_v5.study_c2 import training_runtime as runtime
from compensability_v5.study_c2.io import read_json, read_jsonl, write_json_new
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


def _training_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            **_row(f"c2-train-{index:04d}-{condition}"),
            "pair_id": f"c2-train-{index:04d}",
            "condition": condition,
        }
        for index in range(96)
        for condition in ("collision", "separating")
    )


def _execution_contract() -> dict[str, object]:
    return {
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


def test_training_rows_are_closed_paired_and_keep_frozen_order() -> None:
    rows = _training_rows()
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
    assert "top_k" not in kwargs

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
    payload = _execution_contract()
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


def test_stage25_preflight_binds_all_returned_stage24_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _execution_contract()
    paths = {
        runtime.STAGE25_EXECUTION_CONTRACT: frozen,
        runtime.SHARED_GRADIENT_SUMMARY: {
            "status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
            "continue_to_main_rl": True,
            "group_count": 768,
            "reward_hamming_distance": 635,
        },
        runtime.SHARED_GRADIENT_MANIFEST: {
            "status": "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE",
            "scientific_status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
            "continue_to_main_rl": True,
            "per_group_sha256": frozen["stage24_per_group_sha256"],
            "summary_sha256": frozen["stage24_summary_sha256"],
            "execution_contract_sha256": frozen["stage24_execution_contract_sha256"],
        },
    }
    hashes = {
        runtime.SHARED_GRADIENT_ROWS: frozen["stage24_per_group_sha256"],
        runtime.SHARED_GRADIENT_SUMMARY: frozen["stage24_summary_sha256"],
        runtime.SHARED_GRADIENT_MANIFEST: frozen["stage24_manifest_sha256"],
        runtime.FIBER_ROWS: frozen["fiber_rows_sha256"],
        Path("config.yaml"): frozen["config_sha256"],
        runtime.PACKAGE_LOCK: frozen["package_lock_sha256"],
        runtime.STAGE25_EXECUTION_CONTRACT: "9" * 64,
    }
    monkeypatch.setattr(runtime, "read_json", lambda path: dict(paths[path]))
    monkeypatch.setattr(runtime, "read_jsonl", lambda path: _training_rows())
    monkeypatch.setattr(runtime, "sha256_file", lambda path: hashes[path])
    monkeypatch.setattr(runtime, "load_contract", lambda path: _contract())
    monkeypatch.setattr(runtime, "_require_offline_cuda", lambda: None)
    monkeypatch.setattr(runtime, "require_server_model", lambda: None)
    monkeypatch.setattr(runtime, "tree_sha256", lambda path: "8" * 64)

    result = runtime.preflight_training_arm(
        arm="answer",
        config_path=Path("config.yaml"),
        b3_adapter=Path("adapter"),
        b3_sha256="8" * 64,
        backend_validator=lambda: {"reference_adapter_copy": True},
    )

    assert result["status"] == "STUDY_C2_TRAINING_PREFLIGHT_OK"
    assert result["training_prompt_count"] == 192
    assert result["expected_optimizer_steps"] == 192
    assert result["reward_only_pair_verified"] is True

    paths[runtime.SHARED_GRADIENT_MANIFEST]["execution_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not authorize"):
        runtime.preflight_training_arm(
            arm="state",
            config_path=Path("config.yaml"),
            b3_adapter=Path("adapter"),
            b3_sha256="8" * 64,
            backend_validator=lambda: {"reference_adapter_copy": True},
        )


def test_training_backend_tokenizer_and_prompt_boundaries() -> None:
    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def encode(text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "\n" and add_special_tokens is False
            return [13]

        @staticmethod
        def apply_chat_template(
            prompt: object, *, add_generation_prompt: bool, tokenize: bool
        ) -> list[int]:
            assert prompt and add_generation_prompt and tokenize
            return [1, 2, 3]

    processor = SimpleNamespace(tokenizer=Tokenizer())
    assert backend._newline_and_eos(processor) == (13, 99)
    assert backend._validate_prompt_lengths(
        processor,
        ({"prompt": [{"role": "user", "content": "x"}], "scene_id": "scene"},),
        maximum=3,
    ) == 3
    with pytest.raises(RuntimeError, match="exceeds 2"):
        backend._validate_prompt_lengths(
            processor,
            ({"prompt": [{"role": "user", "content": "x"}], "scene_id": "scene"},),
            maximum=2,
        )


def test_stage25_fake_trainer_writes_complete_recoverable_arm_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_root = tmp_path / "training"
    output_dir = training_root / "C2_answer_reward"
    fiber_rows = tmp_path / "fiber_rows.jsonl"
    fiber_rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _training_rows()),
        encoding="utf-8",
    )
    arm_config = dict(build_reward_arm_configs(_contract(), initialization_hash="8" * 64)[0])
    arm_config["output_directory"] = str(output_dir)
    preflight = {
        "schema_version": 2,
        "status": "STUDY_C2_TRAINING_PREFLIGHT_OK",
        "arm": "answer",
        "arm_config": arm_config,
        "group_size": 8,
        "expected_optimizer_steps": 192,
        "reward_only_pair_verified": True,
    }
    monkeypatch.setattr(execution, "FIBER_ROWS", fiber_rows)
    monkeypatch.setattr(execution, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(execution, "TRAINING_PAIR_MANIFEST", training_root / "manifest.json")
    monkeypatch.setattr(execution, "preflight_training_arm", lambda **kwargs: preflight)

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.dataset = kwargs["dataset"]
            self.reward = kwargs["reward_function"]
            self.callback = kwargs["callbacks"][0]  # type: ignore[index]
            self.state = SimpleNamespace(global_step=0, log_history=[])

        def train(self, *, resume_from_checkpoint: str | None = None) -> None:
            assert resume_from_checkpoint is None
            for index, row in enumerate(self.dataset, start=1):  # type: ignore[union-attr]
                self.state.global_step = index
                self.reward(  # type: ignore[operator]
                    ["3,8,5,18\n"] * 8,
                    scene_id=[row["scene_id"]],
                    trainer_state=self.state,
                )
                if index in {48, 96, 144, 192}:
                    self.callback.on_save(None, self.state, object())
            self.state.log_history = [{"loss": 0.25, "step": 192}]

        @staticmethod
        def save_model(destination: str) -> None:
            path = Path(destination)
            path.mkdir()
            (path / "adapter_model.safetensors").write_bytes(b"adapter")

    result = execution.run_training_arm(
        arm="answer",
        config_path=Path("config.yaml"),
        b3_adapter=Path("adapter"),
        b3_sha256="8" * 64,
        acknowledgement=runtime.TRAINING_ACK,
        trainer_factory=FakeTrainer,
    )

    assert result["status"] == "STUDY_C2_ARM_TRAINING_COMPLETE"
    assert result["optimizer_step_invoked"] is True
    assert result["pair_complete"] is False
    assert len(read_jsonl(output_dir / "raw_reward_trace.jsonl")) == 1536
    assert read_json(output_dir / "summary.json")["counts"] == {
        "F": 0,
        "S": 0,
        "U": 0,
        "X": 1536,
    }
    for step in (48, 96, 144, 192):
        assert (output_dir / f"checkpoint-{step}" / "raw_reward_trace.jsonl").is_file()

    with pytest.raises(PermissionError, match="acknowledgement"):
        execution.run_training_arm(
            arm="answer",
            config_path=Path("config.yaml"),
            b3_adapter=Path("adapter"),
            b3_sha256="8" * 64,
            acknowledgement="wrong",
            trainer_factory=FakeTrainer,
        )


def test_trace_restore_and_pair_manifest_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "training"
    output = root / "C2_answer_reward"
    checkpoint = output / "checkpoint-48"
    checkpoint.mkdir(parents=True)
    snapshot = checkpoint / "raw_reward_trace.jsonl"
    snapshot.write_text('{"group_index": 0}\n', encoding="utf-8")
    trace = output / "raw_reward_trace.jsonl"
    execution._restore_trace(trace, checkpoint, output)
    assert trace.read_text(encoding="utf-8") == snapshot.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        execution._restore_trace(trace, tmp_path, output)

    monkeypatch.setattr(execution, "TRAINING_ROOT", root)
    monkeypatch.setattr(execution, "TRAINING_PAIR_MANIFEST", root / "manifest.json")
    for name in ("C2_answer_reward", "C2_exact_state_reward"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        write_json_new(
            path / "manifest.json",
            {
                "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
                "final_adapter_sha256": name[3:].ljust(64, "0")[:64],
                "raw_reward_trace_sha256": name[3:].ljust(64, "1")[:64],
            },
        )
    pair = execution._pair_manifest_if_complete()
    assert pair is not None
    assert pair["status"] == "STUDY_C2_TWO_ARM_TRAINING_COMPLETE"


def test_runtime_helper_rejection_paths_are_explicit(tmp_path: Path) -> None:
    assert runtime._completion_text("answer") == "answer"
    assert runtime._completion_text([{"role": "assistant", "content": "answer"}]) == "answer"
    with pytest.raises(ValueError, match="unsupported"):
        runtime._completion_text(3)
    with pytest.raises(ValueError, match="malformed"):
        runtime._expand_metadata([], 8, "scene_id")
    with pytest.raises(ValueError, match="cannot align"):
        runtime._expand_metadata(["a", "b", "c"], 8, "scene_id")
    assert runtime._expand_metadata(["a", "b"], 4, "scene_id") == (
        "a",
        "a",
        "b",
        "b",
    )
    with pytest.raises(ValueError, match="K must be 8"):
        runtime.expected_optimizer_steps(_training_rows(), group_size=16)
    with pytest.raises(ValueError, match="192 prompts"):
        runtime.expected_optimizer_steps((), group_size=8)
    with pytest.raises(RuntimeError, match="align"):
        runtime.truncate_first_line_token_ids(
            completion_ids=[[1]],
            logprobs=[[0.1], [0.2]],
            newline_token_id=13,
            eos_token_id=99,
        )

    callback = runtime.TrainingProgressCallback("arm", total_steps=192)
    with pytest.raises(RuntimeError, match="incomplete"):
        callback.on_save(None, SimpleNamespace(global_step=1), object())

    runtime._validate_arm_pair(
        build_reward_arm_configs(_contract(), initialization_hash="8" * 64)
    )
    with pytest.raises(ValueError, match="exactly two"):
        runtime._validate_arm_pair(())
    with pytest.raises(ValueError, match="reward isolation"):
        left, right = build_reward_arm_configs(_contract(), initialization_hash="8" * 64)
        runtime._validate_arm_pair((left, dict(right, seed=0)))
    with pytest.raises(ValueError, match="--arm"):
        runtime._arm_config(
            build_reward_arm_configs(_contract(), initialization_hash="8" * 64), "bad"
        )

    arm = build_reward_arm_configs(_contract(), initialization_hash="8" * 64)[0]
    with pytest.raises(ValueError, match="stopping token IDs"):
        runtime.build_grpo_config_kwargs(
            arm_config=arm,
            output_dir=tmp_path,
            group_size=8,
            eos_token_id=True,
            newline_token_id=13,
            supported_parameters={
                "output_dir",
                "learning_rate",
                "num_train_epochs",
                "num_generations",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "max_completion_length",
                "bf16",
                "temperature",
                "top_p",
                "beta",
                "generation_kwargs",
                "shuffle_dataset",
            },
        )


def test_backend_rejects_malformed_tokenizer_and_reference_adapter() -> None:
    with pytest.raises(RuntimeError, match="encode/EOS"):
        backend._newline_and_eos(object())
    bad_newline = SimpleNamespace(
        tokenizer=SimpleNamespace(
            eos_token_id=99,
            encode=lambda text, add_special_tokens=False: [1, 2],
        )
    )
    with pytest.raises(RuntimeError, match="not one"):
        backend._newline_and_eos(bad_newline)
    with pytest.raises(RuntimeError, match="chat-template"):
        backend._validate_prompt_lengths(object(), ({"prompt": "x"},), maximum=3)

    import torch

    trainable = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    reference = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    class Model:
        def __init__(self) -> None:
            self.peft_config = {"default": object(), "ref": object()}

        @staticmethod
        def named_parameters():
            return iter(
                (
                    ("layer.default.weight", trainable),
                    ("layer.ref.weight", reference),
                )
            )

    backend._verify_reference_copy(Model())
    with pytest.raises(RuntimeError, match="reference adapter"):
        backend._verify_reference_copy(SimpleNamespace(peft_config={}))


def test_execution_helpers_reject_resume_and_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "arm_config.json"
    write_json_new(config_path, {"name": "answer"})
    execution._write_or_validate_arm_config(config_path, {"name": "answer"}, resume=True)
    with pytest.raises(ValueError, match="differs"):
        execution._write_or_validate_arm_config(config_path, {"name": "state"}, resume=True)

    root = tmp_path / "training"
    monkeypatch.setattr(execution, "TRAINING_ROOT", root)
    monkeypatch.setattr(execution, "TRAINING_PAIR_MANIFEST", root / "manifest.json")
    assert execution._pair_manifest_if_complete() is None
    for name in ("C2_answer_reward", "C2_exact_state_reward"):
        path = root / name
        path.mkdir(parents=True)
        write_json_new(
            path / "manifest.json",
            {
                "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
                "final_adapter_sha256": "a" * 64,
                "raw_reward_trace_sha256": "b" * 64,
            },
        )
    write_json_new(root / "manifest.json", {"status": "drifted"})
    with pytest.raises(ValueError, match="drifted"):
        execution._pair_manifest_if_complete()
