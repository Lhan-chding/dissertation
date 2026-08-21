"""Executable Study-C contracts without importing a GPU stack."""

from __future__ import annotations

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

from compensability_v5.qwen.study_c_runtime import (
    STUDY_C_ACK,
    STUDY_C_SEED,
    StudyCError,
    StudyCScene,
    build_study_c_summary,
    make_reward_function,
    qwen_text_evaluation_sampler,
    registered_study_c_arms,
    run_study_c_arm,
    validate_reward_only_pair,
    validate_study_c_config_payload,
)
from compensability_v5.server_runtime.study_c import (
    run_common_space_grpo,
    run_v5_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]


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
    default = registered_study_c_arms(
        initialization="B3", initialization_hash="a" * 64
    )
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
        arm=registered_study_c_arms(
            initialization="B3", initialization_hash="a" * 64
        )[0],
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
        arm=registered_study_c_arms(
            initialization="B3", initialization_hash="a" * 64
        )[0],
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
    arm = registered_study_c_arms(
        initialization="B3", initialization_hash="a" * 64
    )[0]
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


def test_fake_trainer_runs_independent_frozen_16_rollout_evaluation(tmp_path: Path) -> None:
    arm = registered_study_c_arms(
        initialization="B3", initialization_hash="a" * 64
    )[1]

    def factory(**kwargs: object) -> _FakeTrainer:
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
        evaluation_sampler_factory=sampler_factory,
    )

    assert len(observed_seeds) == 1
    assert len(observed_seeds[0]) == 16
    assert evidence["post_training_evaluation_invoked"] is True
    rows = [json.loads(line) for line in (output / "eval_raw_rows.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    assert {row["trace_kind"] for row in rows} == {"post_training_frozen_eval"}
    summary = json.loads((output / "eval_summary.json").read_text())
    assert summary["measurement_scope"] == "post_training_frozen_eval"
    assert summary["rollouts_per_scene"] == 16


def test_qwen_eval_sampler_uses_each_fixed_seed_and_registered_decoding(
    monkeypatch: pytest.MonkeyPatch,
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

    arm = registered_study_c_arms(
        initialization="B3", initialization_hash="a" * 64
    )[0]
    sampler = qwen_text_evaluation_sampler(
        arm=arm, model=Model(), processor=Processor()
    )

    assert sampler(_scene(), (101, 102)) == ("9,2,3,4", "9,2,3,4")
    assert seeded == [101, 102]
    assert cuda_seeded == [101, 102]
    assert {item["temperature"] for item in generation_kwargs} == {arm.temperature}
    assert {item["top_p"] for item in generation_kwargs} == {arm.top_p}
    assert {item["top_k"] for item in generation_kwargs} == {arm.top_k}


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

    interaction = build_study_c_summary(traces, group_size=8)[
        "reward_by_fiber_interaction"
    ]["B3"]

    assert interaction["status"] == "ESTIMATED"
    assert interaction["estimate"] == 0.5
    assert interaction["scene_bootstrap_95_ci"] == [0.5, 0.5]
    assert interaction["bootstrap_resamples"] == 10_000


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
