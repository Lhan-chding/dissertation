from __future__ import annotations

import json
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.qwen.phase5_runtime import (
    Phase5MeasurementConfig,
    checkpoint_tree_hashes,
    load_phase5_config,
    measure_checkpoint,
    phase5_rollout_seed,
)
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, PolicyCheckpoint

SHA = "a" * 64


def _error(scene_id: str = "scene-a") -> HeldOutNaturalError:
    return HeldOutNaturalError(
        scene_id=scene_id,
        family="cross_series",
        split=DatasetSplit.SUPPORT_DEV,
        truth=(2, 3, 4, 5),
        observed=(9, 3, 4, 5),
        error_indices=(0,),
        facts=(
            {"type": "known_value", "index": 1, "value": 3},
            {"type": "known_value", "index": 2, "value": 4},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 5},
        ),
        image_path="images/example.png",
        stage1_model_sha256=SHA,
        stage1_raw_output="9,3,4,5",
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def phase5_generate(self, prompt: str, **options: object) -> tuple[str, tuple[int, ...]]:
        self.calls.append({"prompt": prompt, **options})
        if options["do_sample"] is False:
            return "2,3,4,5", (2, 3, 4, 5)
        seed = int(options["seed"])
        return ("2,3,4,5", (2, 3, 4, 5)) if seed % 2 else ("9,3,4,5", (9, 3, 4, 5))

    def phase5_score(self, prompt: str, completion: str) -> float:
        self.calls.append({"prompt": prompt, "completion": completion, "score": True})
        return -1.0 if completion == "2,3,4,5" else -2.0


def test_phase5_config_freezes_sampling_and_generation_contract() -> None:
    config = Phase5MeasurementConfig.from_mapping(
        {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 0,
            "rollout_count": 4,
            "pass_at_k": [1, 2, 4],
            "informative_group_size": 4,
            "max_new_tokens": 32,
            "sampling_seed": 2026082005,
        }
    )
    assert config.temperature == 0.7
    assert config.pass_at_k == (1, 2, 4)
    with pytest.raises(ValueError, match="temperature"):
        Phase5MeasurementConfig.from_mapping({**config.to_mapping(), "temperature": 1.0})


def test_rollout_seed_is_checkpoint_independent_and_scene_specific() -> None:
    seeds = [phase5_rollout_seed(7, "scene-a", index) for index in range(4)]
    assert len(set(seeds)) == 4
    assert seeds == [phase5_rollout_seed(7, "scene-a", index) for index in range(4)]
    assert seeds != [phase5_rollout_seed(7, "scene-b", index) for index in range(4)]


def test_checkpoint_measurement_runs_one_greedy_two_scores_and_all_fixed_samples() -> None:
    runtime = _FakeRuntime()
    config = Phase5MeasurementConfig.default_for_tests(rollout_count=4)

    rows = measure_checkpoint(
        model=runtime,
        processor=object(),
        checkpoint=PolicyCheckpoint.RECOVERY,
        checkpoint_sha256=SHA,
        errors=(_error(),),
        config=config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.greedy_success is True
    assert row.greedy_raw_output == "2,3,4,5"
    assert row.candidate_margin_true_observed == pytest.approx(1.0)
    assert len(row.sample_outputs) == 4
    assert row.sample_seeds == tuple(
        phase5_rollout_seed(config.sampling_seed, "scene-a", index) for index in range(4)
    )
    generation = [call for call in runtime.calls if "do_sample" in call]
    assert len(generation) == 5
    assert generation[0]["do_sample"] is False
    assert all(call["do_sample"] is True for call in generation[1:])
    assert all(call["temperature"] == 0.7 for call in generation[1:])
    assert len([call for call in runtime.calls if call.get("score")]) == 2


def test_checkpoint_tree_hashes_require_exact_three_non_symlink_adapters(tmp_path: Path) -> None:
    for directory in (
        "C0_format_only/final_adapter",
        "C1_forward_arithmetic/final_adapter",
        "T_constraint_recovery/final_adapter",
    ):
        target = tmp_path / directory
        target.mkdir(parents=True)
        (target / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(directory.encode())

    first = checkpoint_tree_hashes(tmp_path)
    second = checkpoint_tree_hashes(tmp_path)
    assert first == second
    assert set(first) == {"C0", "C1", "T"}
    assert all(len(value) == 64 for value in first.values())

    (tmp_path / "T_constraint_recovery/final_adapter/extra.txt").write_text("x")
    assert checkpoint_tree_hashes(tmp_path)["T"] != first["T"]


def test_checkpoint_tree_hashes_reject_missing_adapter(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="adapter"):
        checkpoint_tree_hashes(tmp_path)


def test_repository_phase5_config_is_closed_and_measurement_only() -> None:
    root = Path(__file__).resolve().parents[1]
    payload, config = load_phase5_config(root / "configs/recoverability/v4_phase_5.yaml")
    assert payload["authorization"] == {
        "measurement_authorized": True,
        "training_authorized": False,
        "rl_authorized": False,
        "downloads_authorized": False,
    }
    assert config.rollout_count == 16
    assert config.pass_at_k == (1, 2, 4, 8, 16)
