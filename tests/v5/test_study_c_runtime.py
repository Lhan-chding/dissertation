"""Executable Study-C contracts without importing a GPU stack."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from compensability_v5.data.common_action_freeze import (
    ACTION_PARSER_ID as FREEZE_PARSER_ID,
)
from compensability_v5.data.common_action_freeze import (
    freeze_common_action_space,
)
from compensability_v5.qwen.study_c_runtime import (
    STUDY_C_ACK,
    STUDY_C_SEED,
    StudyCArm,
    StudyCError,
    StudyCScene,
    build_grpo_config_kwargs,
    build_study_c_summary,
    load_study_c_scenes,
    make_reward_function,
    qwen_text_evaluation_sampler,
    registered_study_c_arms,
    run_pre_training_frozen_eval,
    run_study_c_arm,
    split_study_c_scenes,
    validate_reward_only_pair,
    validate_study_c_config_payload,
    validate_study_c_prompt_lengths,
)
from compensability_v5.server_runtime.study_c import (
    run_common_space_grpo,
    run_v5_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_study_c_cli_module():
    path = ROOT / "scripts/v5/14_run_study_c.py"
    spec = importlib.util.spec_from_file_location("test_run_study_c_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(
    scene_id: str = "scene-1",
    *,
    fiber_size: int = 4,
    fiber_bin: str = "multi_large",
    role: str = "rl_train",
) -> StudyCScene:
    return StudyCScene.from_mapping(
        {
            "scene_id": scene_id,
            "prompt": "Observed values: 8,2,3,4. Return four comma-separated integers only.",
            "truth": [9, 2, 3, 4],
            "answer_operation": {"operator": "difference", "indices": [0, 1]},
            "reward_labels": {"answer": 7, "exact_state": [9, 2, 3, 4]},
            "family": "cross_series",
            "fiber_size": fiber_size,
            "fiber_bin": fiber_bin,
            "support_bin": "medium",
            "role": role,
        }
    )


def _run_scenes() -> tuple[StudyCScene, StudyCScene]:
    return _scene(), _scene("scene-eval", role="rl_eval")


def test_primary_arms_are_one_seed_reward_only_pair() -> None:
    arms = registered_study_c_arms(
        initialization="B3",
        initialization_hash="a" * 64,
    )

    assert tuple(arm.name for arm in arms) == ("B3_answer", "B3_exact_state")
    assert {arm.seed for arm in arms} == {STUDY_C_SEED}
    assert validate_reward_only_pair(arms) is None
    left, right = (arm.to_mapping() for arm in arms)
    differing = {key for key in left if left[key] != right[key]}
    assert differing == {"name", "reward_function"}


def test_b2_is_secondary_and_requires_explicit_opt_in() -> None:
    default = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)
    secondary = registered_study_c_arms(
        initialization="B3",
        initialization_hash="a" * 64,
        include_b2=True,
        b2_initialization_hash="b" * 64,
    )

    assert len(default) == 2
    assert tuple(arm.name for arm in secondary) == (
        "B3_answer",
        "B3_exact_state",
        "B2_answer",
        "B2_exact_state",
    )

    with pytest.raises(StudyCError, match="differs outside reward"):
        validate_reward_only_pair((default[0], replace(default[1], temperature=0.8)))


def test_reward_trace_preserves_raw_completion_and_both_outcomes(tmp_path: Path) -> None:
    trace = tmp_path / "reward_trace.jsonl"
    reward = make_reward_function(
        scenes=_run_scenes(),
        arm=registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0],
        trace_path=trace,
    )

    completions = ["9,2,3,4", "8,1,3,4", "8,2,3,4", "not-a-world"] * 2
    scores = reward(
        completions,
        scene_id=["scene-1"],
        trainer_state=SimpleNamespace(global_step=3),
    )

    assert scores == [1.0, 1.0, 0.0, 0.0] * 2
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["completion"] for row in rows] == completions
    assert [row["exact_world_recovery"] for row in rows] == [
        True,
        False,
        False,
        False,
    ] * 2
    assert [row["answer_correct"] for row in rows] == [True, True, False, False] * 2
    assert all(row["fiber_size"] == 4 for row in rows)


def test_reward_function_rejects_metadata_or_group_drift(tmp_path: Path) -> None:
    reward = make_reward_function(
        scenes=(_scene(),),
        arm=registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0],
        trace_path=tmp_path / "trace.jsonl",
    )

    with pytest.raises(StudyCError, match="group size"):
        reward(["9,2,3,4"], scene_id=["scene-1"])
    with pytest.raises(StudyCError, match="unknown scene"):
        reward(
            ["9,2,3,4"] * 8,
            scene_id=["missing"],
        )


class _FakeTrainer:
    def __init__(self, reward, completions: list[str]) -> None:
        self._reward = reward
        self._completions = completions
        self.state = SimpleNamespace(log_history=[{"step": 1, "loss": 0.25}])
        self.resume_value: str | None = None

    def train(self, *, resume_from_checkpoint: str | None = None) -> None:
        self.resume_value = resume_from_checkpoint
        self._reward(
            self._completions,
            scene_id=["scene-1"],
            trainer_state=SimpleNamespace(global_step=1),
        )

    def save_model(self, output_dir: str) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True)
        (target / "adapter_model.safetensors").write_bytes(b"fake-adapter")


def test_fake_trainer_run_writes_hash_bound_outputs_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0]
    captured: list[_FakeTrainer] = []

    def factory(**kwargs: object) -> _FakeTrainer:
        trainer = _FakeTrainer(
            kwargs["reward_function"],  # type: ignore[arg-type]
            ["9,2,3,4", "8,1,3,4", "8,2,3,4", "oops"] * 2,
        )
        captured.append(trainer)
        return trainer

    output = tmp_path / arm.name
    evidence = run_study_c_arm(
        arm=arm,
        scenes=_run_scenes(),
        output_dir=output,
        trainer_factory=factory,
        provenance_sha256={"common_action_manifest": "c" * 64},
    )

    assert captured[0].resume_value is None
    assert evidence["status"] == "STUDY_C_ARM_COMPLETE"
    assert evidence["reward_trace_sha256"]
    assert (output / "raw_reward_trace.jsonl").is_file()
    assert (output / "group_diagnostics.json").is_file()
    assert (output / "execution_evidence.json").is_file()
    assert evidence["post_training_evaluation_invoked"] is False
    with pytest.raises(StudyCError, match=r"already complete|overwrite"):
        run_study_c_arm(
            arm=arm,
            scenes=_run_scenes(),
            output_dir=output,
            trainer_factory=factory,
            provenance_sha256={"common_action_manifest": "c" * 64},
        )


def test_verified_complete_rejects_tampered_final_adapter(tmp_path: Path) -> None:
    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0]

    def factory(**kwargs: object) -> _FakeTrainer:
        return _FakeTrainer(kwargs["reward_function"], ["9,2,3,4"] * 8)  # type: ignore[arg-type]

    output = tmp_path / arm.name
    provenance = {"common_action_manifest": "c" * 64}
    run_study_c_arm(
        arm=arm,
        scenes=_run_scenes(),
        output_dir=output,
        trainer_factory=factory,
        provenance_sha256=provenance,
        evaluation_sampler_factory=lambda trainer: lambda scene, seeds: ["9,2,3,4"] * len(seeds),
        pre_training_evaluation_sampler_factory=lambda trainer: (
            lambda scene, seeds: ["9,2,3,4"] * len(seeds)
        ),
    )
    cli = _load_study_c_cli_module()
    assert cli._verified_complete(arm, output, provenance) is True
    (output / "final_adapter/adapter_model.safetensors").write_bytes(b"tampered")

    with pytest.raises(StudyCError, match="evidence drifted"):
        cli._verified_complete(arm, output, provenance)


def test_resume_summary_verification_rejects_non_hash_field_tamper(tmp_path: Path) -> None:
    cli = _load_study_c_cli_module()
    expected = {
        "status": "STUDY_C_DIAGNOSTICS_COMPLETE",
        "source_trace_sha256": {"B3_answer": "a" * 64},
        "by_arm": {"B3_answer": {"answer_accuracy_from_world": 0.5}},
    }
    summary_path = tmp_path / "study_c_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **expected,
                "by_arm": {"B3_answer": {"answer_accuracy_from_world": 1.0}},
            }
        )
    )

    with pytest.raises(StudyCError, match="summary content drifted"):
        cli._verify_existing_summary(summary_path, expected)


def test_fake_trainer_runs_independent_frozen_16_rollout_evaluation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[1]

    training_scene_ids: list[str] = []

    def factory(**kwargs: object) -> _FakeTrainer:
        dataset = kwargs["dataset"]
        assert isinstance(dataset, tuple)
        training_scene_ids.extend(str(row["scene_id"]) for row in dataset)
        return _FakeTrainer(
            kwargs["reward_function"],  # type: ignore[arg-type]
            ["9,2,3,4", "8,1,3,4", "8,2,3,4", "oops"] * 2,
        )

    observed_seeds: list[tuple[int, ...]] = []

    def sampler_factory(trainer: object):
        assert isinstance(trainer, _FakeTrainer)

        def sample(scene: StudyCScene, seeds: tuple[int, ...]) -> list[str]:
            assert scene.scene_id == "scene-eval"
            observed_seeds.append(seeds)
            return ["9,2,3,4" if index % 2 == 0 else "8,1,3,4" for index in range(16)]

        return sample

    output = tmp_path / arm.name
    evidence = run_study_c_arm(
        arm=arm,
        scenes=_run_scenes(),
        output_dir=output,
        trainer_factory=factory,
        provenance_sha256={"common_action_manifest": "c" * 64},
        pre_training_evaluation_sampler_factory=sampler_factory,
        evaluation_sampler_factory=sampler_factory,
    )

    assert len(observed_seeds) == 2
    assert training_scene_ids == ["scene-1"]
    assert len(observed_seeds[0]) == 16
    assert evidence["post_training_evaluation_invoked"] is True
    assert evidence["pre_training_evaluation_invoked"] is True
    assert evidence["train_scene_count"] == 1
    assert evidence["eval_scene_count"] == 1
    assert evidence["train_scene_manifest_sha256"] != evidence["eval_scene_manifest_sha256"]
    rows = [json.loads(line) for line in (output / "eval_raw_rows.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    assert {row["trace_kind"] for row in rows} == {"post_training_frozen_eval"}
    summary = json.loads((output / "eval_summary.json").read_text())
    assert summary["measurement_scope"] == "post_training_frozen_eval"
    assert summary["rollouts_per_scene"] == 16
    baseline_rows = [
        json.loads(line)
        for line in (output / "pre_training_eval_raw_rows.jsonl").read_text().splitlines()
    ]
    assert len(baseline_rows) == 16
    assert {row["trace_kind"] for row in baseline_rows} == {"pre_training_frozen_eval"}
    progress = capsys.readouterr().out
    assert f"PROGRESS: Study C {arm.name} pre-training frozen evaluation" in progress
    assert f"PROGRESS: Study C {arm.name} training {arm.steps} optimizer steps" in progress
    assert f"PROGRESS: Study C {arm.name} post_training_frozen_eval scene 1/1 complete" in progress


def test_resume_reuses_hash_verified_pre_training_baseline(tmp_path: Path) -> None:
    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0]
    output = tmp_path / arm.name
    output.mkdir()
    run_pre_training_frozen_eval(
        arm=arm,
        scenes=(_scene("scene-eval", role="rl_eval"),),
        output_dir=output,
        sampler=lambda scene, seeds: ["9,2,3,4"] * len(seeds),
    )
    checkpoint = output / "checkpoint-16"
    checkpoint.mkdir()
    checkpoint_trace = checkpoint / "raw_reward_trace.jsonl"
    make_reward_function(scenes=(_scene(),), arm=arm, trace_path=checkpoint_trace)(
        ["9,2,3,4"] * 8, scene_id=["scene-1"]
    )

    def factory(**kwargs: object) -> _FakeTrainer:
        return _FakeTrainer(
            kwargs["reward_function"],  # type: ignore[arg-type]
            ["9,2,3,4"] * 8,
        )

    def forbidden_pre_factory(trainer: object):
        raise AssertionError("resume must reuse the immutable pre-training baseline")

    evidence = run_study_c_arm(
        arm=arm,
        scenes=_run_scenes(),
        output_dir=output,
        trainer_factory=factory,
        provenance_sha256={"common_action_manifest": "c" * 64},
        resume_from_checkpoint=checkpoint,
        pre_training_evaluation_sampler_factory=forbidden_pre_factory,
    )

    assert evidence["pre_training_evaluation_invoked"] is True
    assert evidence["resumed_from_checkpoint"] == str(checkpoint.resolve())


def test_qwen_eval_sampler_uses_each_fixed_seed_and_registered_decoding(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seeded: list[int] = []
    cuda_seeded: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = seeded.append  # type: ignore[attr-defined]
    fake_torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    fake_torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        manual_seed_all=cuda_seeded.append,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Batch(dict):
        def to(self, device: object) -> Batch:
            assert device == "cuda:0"
            return self

    class IDs:
        shape = (1, 3)

    class Generated:
        def __getitem__(self, key: object) -> list[list[int]]:
            assert isinstance(key, tuple)
            return [[4, 5]]

    class Processor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "templated"

        def __call__(self, **kwargs: object) -> Batch:
            assert kwargs["text"] == ["templated"]
            return Batch(input_ids=IDs())

        def batch_decode(self, ids: object, **kwargs: object) -> list[str]:
            assert ids == [[4, 5]]
            return ["9,2,3,4"]

    generation_kwargs: list[dict[str, object]] = []

    class Model:
        device = "cuda:0"

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: object) -> Generated:
            generation_kwargs.append(kwargs)
            return Generated()

    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0]
    sampler = qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=Processor())

    assert sampler(_scene(), (101, 102)) == ("9,2,3,4", "9,2,3,4")
    assert seeded == [101, 102]
    assert cuda_seeded == [101, 102]
    assert {item["temperature"] for item in generation_kwargs} == {arm.temperature}
    assert {item["top_p"] for item in generation_kwargs} == {arm.top_p}
    assert {item["top_k"] for item in generation_kwargs} == {arm.top_k}
    progress = capsys.readouterr().out
    assert f"PROGRESS: Study C {arm.name} scene scene-1 rollout 1/2" in progress
    assert f"PROGRESS: Study C {arm.name} scene scene-1 rollout 2/2" in progress


def test_qwen_eval_sampler_rejects_malformed_runtime_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.manual_seed = lambda _seed: None  # type: ignore[attr-defined]
    fake_torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class IDs:
        shape = (1, 3)

    class Generated:
        def __getitem__(self, key: object) -> list[list[int]]:
            return [[4, 5]]

    class Model:
        def generate(self, **kwargs: object) -> Generated:
            return Generated()

    class Processor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            return "templated"

        def __call__(self, **kwargs: object) -> dict[str, object]:
            return {"input_ids": IDs()}

        def batch_decode(self, ids: object, **kwargs: object) -> list[str]:
            return ["9,2,3,4"]

    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0]
    scene = _scene()

    with pytest.raises(StudyCError, match="chat template"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=object())(scene, (1,))

    class EmptyPromptProcessor(Processor):
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            return ""

    with pytest.raises(StudyCError, match="invalid chat prompt"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=EmptyPromptProcessor())(
            scene, (1,)
        )

    class NonCallableProcessor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            return "templated"

    with pytest.raises(StudyCError, match="processor is not callable"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=NonCallableProcessor())(
            scene, (1,)
        )

    with pytest.raises(StudyCError, match="generate method"):
        qwen_text_evaluation_sampler(arm=arm, model=object(), processor=Processor())(scene, (1,))

    class MalformedBatchProcessor(Processor):
        def __call__(self, **kwargs: object) -> object:
            return object()

    with pytest.raises(StudyCError, match="mapping-like"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=MalformedBatchProcessor())(
            scene, (1,)
        )

    class MissingInputProcessor(Processor):
        def __call__(self, **kwargs: object) -> dict[str, object]:
            return {}

    with pytest.raises(StudyCError, match="malformed input IDs"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=MissingInputProcessor())(
            scene, (1,)
        )

    class NoDecoderProcessor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            return "templated"

        def __call__(self, **kwargs: object) -> dict[str, object]:
            return {"input_ids": IDs()}

    with pytest.raises(StudyCError, match="batch decoder"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=NoDecoderProcessor())(
            scene, (1,)
        )

    class MalformedDecodeProcessor(Processor):
        def batch_decode(self, ids: object, **kwargs: object) -> list[str]:
            return []

    with pytest.raises(StudyCError, match="malformed completion"):
        qwen_text_evaluation_sampler(arm=arm, model=Model(), processor=MalformedDecodeProcessor())(
            scene, (1,)
        )


def test_summary_reports_group_signal_scene_fibers_and_interaction(tmp_path: Path) -> None:
    answer_path = tmp_path / "answer.jsonl"
    state_path = tmp_path / "state.jsonl"
    common = {
        "schema_version": 1,
        "trainer_step": 1,
        "reward_call_index": 0,
        "scene_id": "scene-1",
        "family": "cross_series",
        "fiber_size": 4,
        "fiber_bin": "multi_large",
        "support_bin": "medium",
        "parse_success": True,
    }
    answer_rows = []
    state_rows = []
    for position, (exact, answer) in enumerate(
        [(True, True), (False, True), (False, False), (False, False)] * 2
    ):
        answer_rows.append(
            {
                **common,
                "position": position,
                "arm": "B3_answer",
                "reward_function": "answer",
                "completion": "9,2,3,4",
                "reward": float(answer),
                "exact_world_recovery": exact,
                "answer_correct": answer,
            }
        )
        state_rows.append(
            {
                **common,
                "position": position,
                "arm": "B3_exact_state",
                "reward_function": "exact_state",
                "completion": "9,2,3,4",
                "reward": float(exact),
                "exact_world_recovery": exact,
                "answer_correct": answer,
            }
        )
    answer_path.write_text("".join(json.dumps(row) + "\n" for row in answer_rows))
    state_path.write_text("".join(json.dumps(row) + "\n" for row in state_rows))

    summary = build_study_c_summary(
        {"B3_answer": answer_path, "B3_exact_state": state_path},
        group_size=8,
    )

    assert summary["by_arm"]["B3_answer"]["mean_group_reward_variance"] == 0.25
    assert summary["by_arm"]["B3_answer"]["informative_group_rate"] == 1.0
    assert summary["by_arm"]["B3_answer"]["correction_bearing_group_rate"] == 1.0
    assert summary["by_arm"]["B3_answer"]["world_recovery_rate"] == 0.25
    assert summary["by_arm"]["B3_answer"]["answer_accuracy_from_world"] == 0.5
    assert summary["by_fiber_bin"]["multi_large"]["state_minus_answer_world_recovery"] == 0.0
    assert summary["per_scene"][0]["fiber_size"] == 4


def test_registered_reward_by_fiber_interaction_uses_scene_bootstrap(tmp_path: Path) -> None:
    traces: dict[str, Path] = {}
    for arm, exact_counts in (
        ("B3_answer", {"singleton": 4, "multi": 2}),
        ("B3_exact_state", {"singleton": 4, "multi": 6}),
    ):
        path = tmp_path / f"{arm}.jsonl"
        rows: list[dict[str, object]] = []
        for scene_id, exact_count in exact_counts.items():
            for position in range(8):
                exact = position < exact_count
                rows.append(
                    {
                        "schema_version": 1,
                        "trainer_step": -1,
                        "reward_call_index": 0 if scene_id == "singleton" else 1,
                        "position": position,
                        "arm": arm,
                        "reward_function": "answer" if arm.endswith("answer") else "exact_state",
                        "scene_id": scene_id,
                        "family": "cross_series",
                        "fiber_size": 1 if scene_id == "singleton" else 4,
                        "fiber_bin": scene_id,
                        "support_bin": "medium",
                        "completion": "9,2,3,4",
                        "parse_success": True,
                        "reward": float(exact),
                        "exact_world_recovery": exact,
                        "answer_correct": exact,
                        "shortcut_answer_success": False,
                    }
                )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        traces[arm] = path

    summary = build_study_c_summary(traces, group_size=8)
    interaction = summary["reward_by_fiber_interaction"]["B3"]

    assert interaction["status"] == "ESTIMATED"
    assert interaction["estimate"] == 0.5
    assert interaction["scene_bootstrap_95_ci"] == [0.5, 0.5]
    assert interaction["bootstrap_resamples"] == 10_000
    assert summary["registered_stop_signals"]["reward_by_fiber_interaction"]["triggered"] is True


def test_registered_answer_up_world_down_signal_requires_paired_baseline(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final.jsonl"
    baseline = tmp_path / "baseline.jsonl"

    def rows(answer_count: int, world_count: int, trace_kind: str) -> list[dict[str, object]]:
        return [
            {
                "schema_version": 1,
                "reward_call_index": index // 8,
                "scene_id": "large-scene",
                "arm": "B3_answer",
                "reward": float(index < answer_count),
                "answer_correct": index < answer_count,
                "exact_world_recovery": index < world_count,
                "parse_success": True,
                "shortcut_answer_success": world_count <= index < answer_count,
                "fiber_size": 5,
                "fiber_bin": "multi_large",
                "family": "cross_series",
                "support_bin": "medium",
                "rollout_seed": 2026082302 + index,
                "trace_kind": trace_kind,
            }
            for index in range(16)
        ]

    final.write_text(
        "".join(json.dumps(row) + "\n" for row in rows(12, 4, "post_training_frozen_eval"))
    )
    baseline.write_text(
        "".join(json.dumps(row) + "\n" for row in rows(8, 8, "pre_training_frozen_eval"))
    )

    signals = build_study_c_summary(
        {"B3_answer": final},
        group_size=8,
        baseline_trace_paths={"B3_answer": baseline},
    )["registered_stop_signals"]

    assert signals["answer_up_world_down_large_fibers"]["triggered"] is True
    assert signals["answer_up_world_down_large_fibers"]["evidence"] == {
        "paired_scene_count": 1,
        "rollouts_per_scene": 16,
        "answer_accuracy_delta": 0.25,
        "world_recovery_delta": -0.25,
    }
    assert signals["subjective_threshold_used"] is False


def test_paired_baseline_without_large_fibers_is_objectively_insufficient(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final-small.jsonl"
    baseline = tmp_path / "baseline-small.jsonl"
    rows = [
        {
            "schema_version": 1,
            "reward_call_index": 0,
            "scene_id": "singleton-scene",
            "arm": "B3_answer",
            "reward": 1.0,
            "answer_correct": True,
            "exact_world_recovery": True,
            "parse_success": True,
            "shortcut_answer_success": False,
            "fiber_size": 1,
            "fiber_bin": "singleton",
            "family": "cross_series",
            "support_bin": "medium",
            "rollout_seed": 2026082302 + index,
        }
        for index in range(8)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    final.write_text(payload)
    baseline.write_text(payload)

    signal = build_study_c_summary(
        {"B3_answer": final},
        group_size=8,
        baseline_trace_paths={"B3_answer": baseline},
    )["registered_stop_signals"]["answer_up_world_down_large_fibers"]

    assert signal["triggered"] is False
    assert signal["status"] == "INSUFFICIENT_LARGE_FIBER_ROWS"
    assert signal["evidence"] == {"final_row_count": 0, "baseline_row_count": 0}


def test_acknowledgement_literal_is_stable() -> None:
    assert STUDY_C_ACK == "I_UNDERSTAND_THIS_STARTS_V5_STUDY_C_GRPO"


def test_runtime_strictly_consumes_the_frozen_low_cost_config() -> None:
    payload = yaml.safe_load((ROOT / "configs/v5/common_space_grpo.yaml").read_text())

    contract = validate_study_c_config_payload(payload)

    assert contract["training"]["learning_rate"] == 1.0e-6
    assert contract["training"]["max_steps"] == 64
    assert contract["training"]["group_size"] == 8
    assert contract["evaluation"] == {
        "rollout_count": 16,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 2026082302,
    }
    assert contract["data_split"] == {
        "train_role": "rl_train",
        "eval_role": "rl_eval",
        "train_scene_count": 72,
        "eval_scene_count": 24,
        "require_scene_id_disjoint": True,
    }
    drifted = json.loads(json.dumps(payload))
    drifted["training"]["group_size"] = 7
    with pytest.raises(StudyCError, match="training contract drifted"):
        validate_study_c_config_payload(drifted)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("phase", "wrong", "common action/seed"),
        ("evaluation", {}, "evaluation contract"),
        ("data_split", {}, "split contract"),
        ("authorization", {}, "authorization contract"),
        ("offline", {}, "offline contract"),
    ],
)
def test_config_rejects_registered_contract_drift(
    field: str, replacement: object, message: str
) -> None:
    payload = yaml.safe_load((ROOT / "configs/v5/common_space_grpo.yaml").read_text())
    payload[field] = replacement

    with pytest.raises(StudyCError, match=message):
        validate_study_c_config_payload(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"initialization": "Base"},
        {"name": "wrong"},
        {"reward_function": "wrong", "name": "B3_wrong"},
        {"initialization_hash": "bad"},
        {"seed": 1},
        {"action_space": "text"},
        {"group_size": 1},
    ],
)
def test_arm_rejects_registered_contract_drift(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "name": "B3_answer",
        "initialization": "B3",
        "initialization_hash": "a" * 64,
        "reward_function": "answer",
    }
    values.update(changes)
    with pytest.raises(StudyCError):
        StudyCArm(**values)  # type: ignore[arg-type]


def test_scene_split_package_and_prompt_boundaries_reject_drift() -> None:
    with pytest.raises(StudyCError, match="non-empty"):
        split_study_c_scenes((_scene(),))
    with pytest.raises(StudyCError, match="rl_train count"):
        split_study_c_scenes(_run_scenes(), expected_train_count=72)
    with pytest.raises(StudyCError, match="rl_eval count"):
        split_study_c_scenes(_run_scenes(), expected_eval_count=24)
    with pytest.raises(StudyCError, match="no scenes"):
        load_study_c_scenes({})

    package = {
        "scenes": [_scene().to_dataset_row()],
        "rollout_seeds": [STUDY_C_SEED],
        "action_parser_id": "wrong",
    }
    with pytest.raises(StudyCError):
        load_study_c_scenes(package)

    with pytest.raises(StudyCError, match="callable tokenizer"):
        validate_study_c_prompt_lengths((_scene(),), object(), max_prompt_length=512)

    class Processor:
        @staticmethod
        def tokenizer(*args: object, **kwargs: object) -> dict[str, object]:
            return {"input_ids": [[0] * 513]}

    with pytest.raises(StudyCError, match="exceeds 512"):
        validate_study_c_prompt_lengths((_scene(),), Processor(), max_prompt_length=512)


def test_grpo_kwargs_execute_prompt_limit_with_locked_trl_compatibility(tmp_path: Path) -> None:
    arm = registered_study_c_arms(initialization="B3", initialization_hash="a" * 64)[0]
    legacy = build_grpo_config_kwargs(arm, tmp_path / "legacy", ("output_dir",))
    current = build_grpo_config_kwargs(
        arm, tmp_path / "current", ("output_dir", "max_prompt_length")
    )

    assert "max_prompt_length" not in legacy
    assert current["max_prompt_length"] == 512
    for field in ("learning_rate", "max_steps", "num_generations", "temperature", "beta"):
        assert legacy[field] == current[field]


def test_study_c_cli_has_cpu_only_fixture_preflight() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/v5/14_run_study_c.py"), "--fixture-dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "FIXTURE_DRY_RUN_OK"
    assert payload["seed"] == STUDY_C_SEED
    assert payload["arms"] == ["B3_answer", "B3_exact_state"]


def test_study_c_cli_is_explicitly_execute_gated() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/v5/14_run_study_c.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--execute" in completed.stdout


def test_generic_server_runtime_exports_callbacks_and_evaluation_is_immutable(
    tmp_path: Path,
) -> None:
    assert callable(run_common_space_grpo)
    trace_paths: list[Path] = []
    for arm, reward_kind in (
        ("B3_answer", "answer"),
        ("B3_exact_state", "exact_state"),
    ):
        trace = tmp_path / f"{arm}.jsonl"
        rows = []
        for position in range(16):
            within_group = position % 8
            exact = within_group == 0
            answer = within_group < 2
            rows.append(
                {
                    "schema_version": 1,
                    "trainer_step": 1,
                    "reward_call_index": position // 8,
                    "position": within_group,
                    "arm": arm,
                    "initialization": "B3",
                    "reward_function": reward_kind,
                    "scene_id": "scene-1",
                    "family": "cross_series",
                    "fiber_size": 4,
                    "fiber_bin": "multi_large",
                    "support_bin": "medium",
                    "completion": "9,2,3,4",
                    "parsed_world": [9, 2, 3, 4],
                    "parse_success": True,
                    "reward": float(answer if reward_kind == "answer" else exact),
                    "exact_world_recovery": exact,
                    "answer_correct": answer,
                    "shortcut_answer_success": answer and not exact,
                }
            )
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
        trace_paths.append(trace)
    output = tmp_path / "evaluation.json"
    validation = {
        "schema_version": 1,
        "phase": "phase8_v5_evaluation",
        "config_sha256": "a" * 64,
        "package_lock_sha256": "b" * 64,
        "input_sha256": {
            str(path): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
            for path in trace_paths
        },
        "output": str(output),
    }

    result = run_v5_evaluation(validation, {"task": "evaluate_v5", "k": 8})

    assert result["status"] == "STUDY_C_DIAGNOSTICS_COMPLETE"
    assert json.loads(output.read_text())["source_trace_sha256"] == validation["input_sha256"]
    with pytest.raises(StudyCError, match="overwrite"):
        run_v5_evaluation(validation, {"task": "evaluate_v5", "k": 8})


def test_phase08_freeze_is_accepted_by_phase09_study_c_loader(tmp_path: Path) -> None:
    from compensability_v5.server_runtime import study_c as server_study_c

    scenes = [
        {
            "scene_id": f"rl-{index}",
            "prompt": f"Observed world {index}. Return four integers.",
            "truth": [9, 2, 3, 4],
            "answer_operation": {"operator": "sum", "indices": [0, 1]},
            "family": "pair_sum",
            "fiber_size": 3,
            "policy_support": index / 4,
            "candidate_worlds": [[9, 2, 3, 4], [8, 3, 3, 4]],
        }
        for index in range(4)
    ]
    package = freeze_common_action_space(
        scenes,
        initialization_hashes={"B3": "a" * 64, "B2": "b" * 64, "Base": "c" * 64},
        action_parser_id=FREEZE_PARSER_ID,
        rollout_seeds=[STUDY_C_SEED],
    )
    manifest = tmp_path / "common_space_rl.json"
    manifest.write_text(json.dumps(package) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    selected_path, selected_package = server_study_c._common_action_package(
        {str(manifest.resolve()): digest}
    )

    assert package["status"] == "V5_COMMON_ACTION_SPACE_FROZEN"
    assert selected_path == manifest.resolve()
    assert selected_package == package


def test_study_c_loader_accepts_registered_max_minus_min_operation() -> None:
    payload = {
        "scene_id": "max-minus-min",
        "prompt": "Observed values: 8,2,3,4. Return four comma-separated integers only.",
        "truth": [9, 2, 3, 4],
        "answer_operation": {
            "operator": "max_minus_min",
            "indices": [0, 1, 2, 3],
        },
        "reward_labels": {"answer": 7, "exact_state": [9, 2, 3, 4]},
        "family": "trend",
        "fiber_size": 4,
        "fiber_bin": "multi_2_4",
        "support_bin": "medium",
        "role": "rl_train",
    }

    loaded = StudyCScene.from_mapping(payload)

    assert loaded.answer_operator == "max_minus_min"
    assert loaded.answer_indices == (0, 1, 2, 3)
    assert loaded.answer_label == 7
