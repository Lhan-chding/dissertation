from __future__ import annotations

import json
from pathlib import Path

import pytest

from compensability_v5.study_c2 import evaluation_runtime as runtime
from compensability_v5.study_c2.io import read_json, read_jsonl, sha256_file, write_json_new
from compensability_v5.study_c2.schemas import validate_study_c2_config


def _config() -> dict[str, object]:
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


def _pair_row(pair_id: str, split: str, condition: str) -> dict[str, object]:
    operation = {"operator": "sum", "indices": [0, 1] if condition == "collision" else [3, 0]}
    return {
        "schema_version": 2,
        "split": split,
        "scene_id": f"{pair_id}-{condition}",
        "pair_id": pair_id,
        "condition": condition,
        "family": "cross_series",
        "prompt": f"Recover {pair_id} {condition}.",
        "prompt_sha256": ("a" if condition == "collision" else "b") * 64,
        "truth": [3, 8, 5, 18],
        "observation": [3, 8, 5, 17],
        "operation": operation,
        "gold_answer": 11 if condition == "collision" else 21,
        "observed_answer": 11 if condition == "collision" else 20,
        "observed_is_answer_equivalent": condition == "collision",
        "reward_identifiable": False,
    }


def _evaluation_rows() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for split, count in (("dev", 24), ("test", 48), ("positive_control", 16)):
        for index in range(count):
            pair_id = f"c2-{split}-{index:04d}"
            rows.append(_pair_row(pair_id, split, "collision"))
            rows.append(_pair_row(pair_id, split, "separating"))
    rows.append(_pair_row("c2-train-0000", "train", "collision"))
    rows.append(_pair_row("c2-support-0000", "support_audit", "collision"))
    return tuple(rows)


def _training_manifests(root: Path) -> dict[str, dict[str, object]]:
    manifests: dict[str, dict[str, object]] = {}
    for name in ("C2_answer_reward", "C2_exact_state_reward"):
        output = root / name
        output.mkdir(parents=True)
        final_adapter = output / "final_adapter"
        final_adapter.mkdir()
        (final_adapter / "adapter_model.safetensors").write_bytes(name.encode("utf-8"))
        payload = {
            "schema_version": 2,
            "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
            "arm": name,
            "final_adapter_sha256": ("a" if "answer" in name else "b") * 64,
            "raw_reward_trace_sha256": ("c" if "answer" in name else "d") * 64,
        }
        write_json_new(output / "manifest.json", payload)
        manifests[name] = payload
    pair = {
        "schema_version": 2,
        "status": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE",
        "arms": {
            name: {
                "manifest_sha256": sha256_file(root / name / "manifest.json"),
                "final_adapter_sha256": manifests[name]["final_adapter_sha256"],
                "raw_reward_trace_sha256": manifests[name]["raw_reward_trace_sha256"],
            }
            for name in sorted(manifests)
        },
        "reward_only_pair_verified": True,
        "training_prompt_count_per_arm": 192,
        "optimizer_steps_per_arm": 192,
        "training_invoked": True,
        "rl_invoked": True,
        "gpu_invoked": True,
    }
    write_json_new(root / "manifest.json", pair)
    return manifests


def test_evaluation_rows_are_fail_closed_and_exclude_stage23_25_inputs() -> None:
    selected = runtime.select_evaluation_rows(_evaluation_rows())

    assert len(selected) == 176
    assert [row["scene_id"] for row in selected[:4]] == [
        "c2-dev-0000-collision",
        "c2-dev-0000-separating",
        "c2-dev-0001-collision",
        "c2-dev-0001-separating",
    ]
    assert {row["split"] for row in selected} == {"dev", "test", "positive_control"}

    with pytest.raises(ValueError, match="176"):
        runtime.select_evaluation_rows(selected[:-2])
    with pytest.raises(ValueError, match="88 complete paired scenes"):
        runtime.select_evaluation_rows(tuple(dict(row, condition="collision") for row in selected))


def test_stage26_preflight_binds_stage25_pair_manifest_and_eval_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training_root = tmp_path / "training"
    _training_manifests(training_root)
    fiber_rows = tmp_path / "reward_fibers.jsonl"
    fiber_rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _evaluation_rows()),
        encoding="utf-8",
    )
    config = _config()
    pair_manifest = read_json(training_root / "manifest.json")
    arm_paths = {
        name: training_root / name / "manifest.json"
        for name in ("C2_answer_reward", "C2_exact_state_reward")
    }
    hashes = {
        fiber_rows: "1" * 64,
        Path("config.yaml"): "2" * 64,
        runtime.PACKAGE_LOCK: "3" * 64,
        training_root / "manifest.json": "4" * 64,
        arm_paths["C2_answer_reward"]: "5" * 64,
        arm_paths["C2_exact_state_reward"]: "6" * 64,
    }

    monkeypatch.setattr(runtime, "FIBER_ROWS", fiber_rows)
    monkeypatch.setattr(runtime, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(runtime, "TRAINING_PAIR_MANIFEST", training_root / "manifest.json")
    monkeypatch.setattr(runtime, "read_json", lambda path: read_json(path))
    monkeypatch.setattr(runtime, "read_jsonl", lambda path: read_jsonl(path))
    monkeypatch.setattr(runtime, "sha256_file", lambda path: hashes[path])
    monkeypatch.setattr(runtime, "load_contract", lambda path: config)
    monkeypatch.setattr(runtime, "_require_offline_cuda", lambda: None)
    monkeypatch.setattr(runtime, "require_server_model", lambda: None)
    monkeypatch.setattr(
        runtime, "tree_sha256", lambda path: ("a" if "answer" in str(path) else "b") * 64
    )

    result = runtime.preflight_post_training_evaluation(
        config_path=Path("config.yaml"),
        backend_validator=lambda: {"generation_available": True},
    )

    assert result["status"] == "STUDY_C2_EVALUATION_PREFLIGHT_OK"
    assert result["evaluation_scene_count"] == 176
    assert result["evaluation_pair_count"] == 88
    assert result["sampled_rollouts"] == 16
    assert result["training_pair_manifest_sha256"] == "4" * 64
    assert result["reward_only_pair_verified"] is True

    arm_paths["C2_answer_reward"].write_text(
        json.dumps(
            dict(read_json(arm_paths["C2_answer_reward"]), status="DRIFTED"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete arm manifest"):
        runtime.preflight_post_training_evaluation(
            config_path=Path("config.yaml"),
            backend_validator=lambda: {"generation_available": True},
        )


def test_stage26_run_writes_hash_bound_evaluation_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fiber_rows = tmp_path / "reward_fibers.jsonl"
    fiber_rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _evaluation_rows()),
        encoding="utf-8",
    )
    evaluation_root = tmp_path / "evaluation"
    training_root = tmp_path / "training"
    _training_manifests(training_root)
    monkeypatch.setattr(runtime, "FIBER_ROWS", fiber_rows)
    monkeypatch.setattr(runtime, "TRAINING_ROOT", training_root)
    monkeypatch.setattr(runtime, "TRAINING_PAIR_MANIFEST", training_root / "manifest.json")
    monkeypatch.setattr(runtime, "EVALUATION_ROOT", evaluation_root)
    monkeypatch.setattr(runtime, "EVALUATION_RAW_ROWS", evaluation_root / "raw_rows.jsonl")
    monkeypatch.setattr(runtime, "EVALUATION_SUMMARY", evaluation_root / "summary.json")
    monkeypatch.setattr(runtime, "EVALUATION_MANIFEST", evaluation_root / "manifest.json")
    monkeypatch.setattr(runtime, "tree_sha256", lambda path: ("e" if "answer" in str(path) else "f") * 64)

    preflight = {
        "schema_version": 2,
        "status": "STUDY_C2_EVALUATION_PREFLIGHT_OK",
        "config_sha256": "1" * 64,
        "fiber_rows_sha256": "2" * 64,
        "package_lock_sha256": "3" * 64,
        "training_pair_manifest_sha256": "4" * 64,
        "evaluation_scene_count": 176,
        "evaluation_pair_count": 88,
        "sampled_rollouts": 16,
        "group_size": 8,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 2026082403,
        "reward_only_pair_verified": True,
        "arm_manifests": {
            "C2_answer_reward": {
                "manifest_path": str(training_root / "C2_answer_reward" / "manifest.json"),
                "manifest_sha256": "5" * 64,
                "adapter_path": str(training_root / "C2_answer_reward" / "final_adapter"),
                "reward_function_id": "answer_reward_v1",
            },
            "C2_exact_state_reward": {
                "manifest_path": str(training_root / "C2_exact_state_reward" / "manifest.json"),
                "manifest_sha256": "6" * 64,
                "adapter_path": str(training_root / "C2_exact_state_reward" / "final_adapter"),
                "reward_function_id": "exact_state_reward_v1",
            },
        },
    }
    monkeypatch.setattr(runtime, "preflight_post_training_evaluation", lambda **kwargs: preflight)

    def sampler_factory(*, arm_config: dict[str, object], adapter_path: Path):
        del adapter_path

        def sample(row: dict[str, object], seeds: tuple[int, ...]) -> tuple[str, ...]:
            truth = ",".join(map(str, row["truth"])) + "\n"
            observed = ",".join(map(str, row["observation"])) + "\n"
            if arm_config["reward_function_id"] == "exact_state_reward_v1" and row["condition"] == "collision":
                return tuple(truth for _ in seeds)
            return tuple(observed for _ in seeds)

        return sample

    result = runtime.run_post_training_evaluation(
        config_path=Path("config.yaml"),
        acknowledgement=runtime.EVALUATION_ACK,
        sampler_factory=sampler_factory,
    )

    assert result["status"] == "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE"
    assert result["pair_bootstrap"]["pair_count"] == 88
    assert result["pair_bootstrap"]["estimate"] > 0.0
    assert result["training_invoked"] is False
    assert result["optimizer_step_invoked"] is False
    assert result["gpu_invoked"] is True
    assert read_json(runtime.EVALUATION_MANIFEST)["summary_sha256"] == sha256_file(
        runtime.EVALUATION_SUMMARY
    )
    assert len(read_jsonl(runtime.EVALUATION_RAW_ROWS)) == 176 * 16 * 2

    with pytest.raises(PermissionError, match="acknowledgement"):
        runtime.run_post_training_evaluation(
            config_path=Path("config.yaml"),
            acknowledgement="wrong",
            sampler_factory=sampler_factory,
        )
