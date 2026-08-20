"""Fail-closed Phase 6 GRPO data, reward, and diagnostic contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from compensability_v4.data.splits import DatasetSplit

_WORLD = re.compile(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\Z")
_INTEGER = re.compile(r"\s*([+-]?\d+)\s*\Z")


class RewardKind(str, Enum):
    RECOVERY_OUTCOME = "recovery_outcome"
    ANSWER_ONLY = "answer_only"


class Phase6Variant(str, Enum):
    BASE_ANSWER_ONLY = "Base_AnswerOnly_RL"
    RECOVERY_OUTCOME = "Recovery_LoRA_RecoveryOutcome_RL"
    RECOVERY_ANSWER_ONLY = "Recovery_LoRA_AnswerOnly_RL"

    @property
    def initial_checkpoint(self) -> str:
        return "Base" if self is Phase6Variant.BASE_ANSWER_ONLY else "T"

    @property
    def reward_kind(self) -> RewardKind:
        if self is Phase6Variant.RECOVERY_OUTCOME:
            return RewardKind.RECOVERY_OUTCOME
        return RewardKind.ANSWER_ONLY


@dataclass(frozen=True, slots=True)
class Phase6TrainingConfig:
    precision: str
    learning_rate: float
    max_steps: int
    group_size: int
    temperature: float
    top_p: float
    top_k: int
    kl_beta: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_prompt_length: int
    max_completion_length: int
    checkpoint_steps: int
    seed: int
    use_vllm: bool

    @classmethod
    def default(cls) -> Phase6TrainingConfig:
        return cls(
            precision="bf16",
            learning_rate=1.0e-6,
            max_steps=64,
            group_size=8,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            kl_beta=0.04,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_prompt_length=512,
            max_completion_length=32,
            checkpoint_steps=16,
            seed=2026082006,
            use_vllm=False,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Phase6TrainingConfig:
        if set(value) != set(asdict(cls.default())):
            raise ValueError("Phase 6 training config fields differ from the frozen contract")
        config = cls(**dict(value))  # type: ignore[arg-type]
        if (
            config.precision != "bf16"
            or config.learning_rate != 1.0e-6
            or config.max_steps != 64
            or config.group_size != 8
            or config.temperature != 0.7
            or config.top_p != 1.0
            or config.top_k != 0
            or config.kl_beta != 0.04
            or config.per_device_train_batch_size != 1
            or config.gradient_accumulation_steps != 8
            or config.max_prompt_length != 512
            or config.max_completion_length != 32
            or config.checkpoint_steps != 16
            or config.seed != 2026082006
            or config.use_vllm is not False
        ):
            raise ValueError("Phase 6 training parameters drifted from the frozen contract")
        return config


def load_phase6_config(path: Path) -> tuple[dict[str, object], Phase6TrainingConfig]:
    from compensability_v4.qwen.phase6_runtime import load_phase6_config as load_plan

    load_plan(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Phase 6 training config is malformed")
    config = Phase6TrainingConfig.from_mapping(training)
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation != {
        "checkpoints": [
            "Base",
            Phase6Variant.BASE_ANSWER_ONLY.value,
            "Recovery_LoRA",
            Phase6Variant.RECOVERY_OUTCOME.value,
            Phase6Variant.RECOVERY_ANSWER_ONLY.value,
        ],
        "support_dev_only": True,
        "rollout_count": 16,
        "group_size": 8,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 32,
        "seed": 2026082005,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 2026082007,
    }:
        raise ValueError("Phase 6 evaluation contract drifted")
    return payload, config


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"Phase 6 {label} must contain four integers")
    return tuple(value)  # type: ignore[return-value]


def _facts(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("Phase 6 facts must be a non-empty sequence")
    rows = tuple(dict(item) for item in value if isinstance(item, Mapping))
    if len(rows) != len(value):
        raise ValueError("Phase 6 facts must contain mappings")
    return rows


def _world_text(world: Sequence[int]) -> str:
    return ",".join(str(item) for item in world)


@dataclass(frozen=True, slots=True)
class Phase6Example:
    example_id: str
    scene_id: str
    family: str
    split: DatasetSplit
    reward_kind: RewardKind
    prompt: str
    expected_completion: str
    truth: tuple[int, int, int, int]
    observed: tuple[int, int, int, int]
    answer: int
    operation: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "example_id": self.example_id,
            "scene_id": self.scene_id,
            "family": self.family,
            "split": self.split.value,
            "reward_kind": self.reward_kind.value,
            "prompt": self.prompt,
            "expected_completion": self.expected_completion,
            "truth": list(self.truth),
            "observed": list(self.observed),
            "answer": self.answer,
            "operation": self.operation,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Phase6Example:
        expected = {
            "schema_version",
            "example_id",
            "scene_id",
            "family",
            "split",
            "reward_kind",
            "prompt",
            "expected_completion",
            "truth",
            "observed",
            "answer",
            "operation",
        }
        if set(value) != expected or value.get("schema_version") != 1:
            raise ValueError("Phase 6 example schema differs from the frozen contract")
        if value.get("split") != DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN.value:
            raise ValueError("Phase 6 examples must use natural_error_support_train")
        strings = ("example_id", "scene_id", "family", "prompt", "expected_completion", "operation")
        if any(not isinstance(value.get(key), str) or not value[key] for key in strings):
            raise ValueError("Phase 6 example string fields are malformed")
        if type(value.get("answer")) is not int:
            raise ValueError("Phase 6 answer must be an integer")
        example = cls(
            example_id=str(value["example_id"]),
            scene_id=str(value["scene_id"]),
            family=str(value["family"]),
            split=DatasetSplit(str(value["split"])),
            reward_kind=RewardKind(str(value["reward_kind"])),
            prompt=str(value["prompt"]),
            expected_completion=str(value["expected_completion"]),
            truth=_world(value["truth"], "truth"),
            observed=_world(value["observed"], "observation"),
            answer=int(value["answer"]),
            operation=str(value["operation"]),
        )
        if sum(a != b for a, b in zip(example.truth, example.observed, strict=True)) != 1:
            raise ValueError("Phase 6 training observations must contain exactly one error")
        expected_completion = (
            _world_text(example.truth)
            if example.reward_kind is RewardKind.RECOVERY_OUTCOME
            else str(example.answer)
        )
        if example.expected_completion != expected_completion:
            raise ValueError("Phase 6 expected completion does not match its reward contract")
        return example


def _index(rows: Iterable[Mapping[str, object]], label: str) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"Phase 6 {label} rows must be mappings")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or scene_id in output:
            raise ValueError(f"Phase 6 {label} scene identifiers are missing or duplicated")
        output[scene_id] = row
    if not output:
        raise ValueError(f"Phase 6 {label} rows are empty")
    return output


def build_phase6_examples(
    *,
    natural_scenes: Iterable[Mapping[str, object]],
    natural_observations: Iterable[Mapping[str, object]],
    dataset_records: Iterable[Mapping[str, object]],
) -> tuple[Phase6Example, ...]:
    """Build both reward views from Phase 4 natural-error training scenes only."""

    scenes = _index(natural_scenes, "natural scenes")
    observations = _index(natural_observations, "natural observations")
    records = _index(dataset_records, "dataset records")
    if set(scenes) != set(observations) or not set(scenes).issubset(records):
        raise ValueError("Phase 6 natural scene/observation/dataset closure drifted")
    output: list[Phase6Example] = []
    for scene_id in sorted(scenes):
        scene, observation, record = scenes[scene_id], observations[scene_id], records[scene_id]
        if (
            scene.get("split") != DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN.value
            or observation.get("split") != DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN.value
        ):
            raise ValueError("Phase 6 examples must use natural_error_support_train")
        truth = _world(scene.get("truth"), "truth")
        observed = _world(observation.get("observed_values"), "observation")
        if sum(a != b for a, b in zip(truth, observed, strict=True)) != 1:
            raise ValueError("Phase 6 requires exactly one natural observation error")
        facts = _facts(scene.get("facts"))
        family, question, operation, answer = (
            record.get("family"),
            record.get("question"),
            record.get("operation"),
            record.get("answer"),
        )
        if (
            not isinstance(family, str)
            or not isinstance(question, str)
            or not isinstance(operation, str)
            or type(answer) is not int
            or _world(record.get("values"), "dataset truth") != truth
        ):
            raise ValueError("Phase 6 dataset record metadata is malformed")
        common = (
            f"Observed values: {_world_text(observed)}\n"
            f"Facts: {json.dumps(facts, sort_keys=True, separators=(',', ':'))}\n"
            "Use only the observations and facts in this prompt. "
        )
        output.extend(
            (
                Phase6Example(
                    example_id=f"{scene_id}:recovery_outcome",
                    scene_id=scene_id,
                    family=family,
                    split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
                    reward_kind=RewardKind.RECOVERY_OUTCOME,
                    prompt=common
                    + "Recover the full world. Return exactly four comma-separated integers only.",
                    expected_completion=_world_text(truth),
                    truth=truth,
                    observed=observed,
                    answer=answer,
                    operation=operation,
                ),
                Phase6Example(
                    example_id=f"{scene_id}:answer_only",
                    scene_id=scene_id,
                    family=family,
                    split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
                    reward_kind=RewardKind.ANSWER_ONLY,
                    prompt=common + f"Question: {question}\nReturn the final integer answer only.",
                    expected_completion=str(answer),
                    truth=truth,
                    observed=observed,
                    answer=answer,
                    operation=operation,
                ),
            )
        )
    return tuple(output)


def _parse_world(text: str) -> tuple[int, int, int, int] | None:
    match = _WORLD.fullmatch(text)
    return None if match is None else tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _parse_answer(text: str) -> int | None:
    match = _INTEGER.fullmatch(text)
    return None if match is None else int(match.group(1))


@dataclass(frozen=True, slots=True)
class CompletionScore:
    reward: float
    exact_world_recovery: bool
    observation_copy: bool
    answer_exact: bool
    parse_success: bool


def score_phase6_completion(example: Phase6Example, completion: str) -> CompletionScore:
    if not isinstance(example, Phase6Example) or not isinstance(completion, str):
        raise TypeError("Phase 6 scoring requires an example and completion text")
    world = _parse_world(completion)
    answer = _parse_answer(completion)
    exact_world = world == example.truth
    observation_copy = world == example.observed
    answer_exact = answer == example.answer
    reward = exact_world if example.reward_kind is RewardKind.RECOVERY_OUTCOME else answer_exact
    return CompletionScore(
        reward=float(reward),
        exact_world_recovery=exact_world,
        observation_copy=observation_copy,
        answer_exact=answer_exact,
        parse_success=world is not None
        if example.reward_kind is RewardKind.RECOVERY_OUTCOME
        else answer is not None,
    )


@dataclass(frozen=True, slots=True)
class RewardGroupTrace:
    group_id: str
    scene_id: str
    variant: Phase6Variant
    rewards: tuple[float, ...]
    kl: float | None
    entropy: float | None
    exact_world_recovery_count: int = 0
    observation_copy_count: int = 0

    def __post_init__(self) -> None:
        if not self.group_id or not self.scene_id or not self.rewards:
            raise ValueError("Phase 6 reward group identifiers/rewards must be non-empty")
        if any(value not in (0.0, 1.0) for value in self.rewards):
            raise ValueError("Phase 6 outcome rewards must be binary")
        if any(
            value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value))
            for value in (self.kl, self.entropy)
        ):
            raise ValueError("Phase 6 KL/entropy values must be finite when present")
        if any(
            type(value) is not int or value < 0 or value > len(self.rewards)
            for value in (self.exact_world_recovery_count, self.observation_copy_count)
        ):
            raise ValueError("Phase 6 group diagnostic counts are invalid")


def summarize_reward_groups(
    groups: Sequence[RewardGroupTrace], *, group_size: int
) -> dict[str, object]:
    if type(group_size) is not int or group_size < 2 or not groups:
        raise ValueError("Phase 6 group summary requires groups and size >= 2")
    if any(len(group.rewards) != group_size for group in groups):
        raise ValueError("Phase 6 reward group size drifted")
    all_zero = sum(not any(group.rewards) for group in groups)
    all_one = sum(all(group.rewards) for group in groups)
    variances = tuple(
        (sum(group.rewards) / group_size) * (1.0 - sum(group.rewards) / group_size)
        for group in groups
    )
    total = len(groups) * group_size
    kl = tuple(float(group.kl) for group in groups if group.kl is not None)
    entropy = tuple(float(group.entropy) for group in groups if group.entropy is not None)
    return {
        "schema_version": 1,
        "group_count": len(groups),
        "group_size": group_size,
        "all_zero_group_rate": all_zero / len(groups),
        "all_one_group_rate": all_one / len(groups),
        "non_degenerate_group_rate": (len(groups) - all_zero - all_one) / len(groups),
        "mean_group_reward_variance": sum(variances) / len(variances),
        "mean_kl": None if not kl else sum(kl) / len(kl),
        "mean_entropy": None if not entropy else sum(entropy) / len(entropy),
        "exact_world_recovery_rate": sum(group.exact_world_recovery_count for group in groups)
        / total,
        "observation_copy_rate": sum(group.observation_copy_count for group in groups) / total,
        "subjective_success_threshold_applied": False,
    }


def validate_phase5_policy_support(payload: Mapping[str, object]) -> None:
    if payload.get("status") != "PHASE_5_POLICY_SUPPORT_EXECUTED":
        raise ValueError("Phase 6 requires completed Phase 5 policy-support evidence")
    if (
        type(payload.get("number_of_held_out_natural_errors")) is not int
        or int(payload["number_of_held_out_natural_errors"]) <= 0
        or payload.get("number_of_checkpoint_scene_rows")
        != 4 * int(payload["number_of_held_out_natural_errors"])
        or type(payload.get("informative_group_size")) is not int
        or int(payload["informative_group_size"]) < 2
        or type(payload.get("sampling_rollouts_per_scene")) is not int
        or int(payload["sampling_rollouts_per_scene"]) < int(payload["informative_group_size"])
    ):
        raise ValueError("Phase 6 Phase-5 measurement closure is malformed")
    if payload.get("subjective_success_threshold_applied") is not False:
        raise ValueError("Phase 6 must not use a subjective Phase-5 threshold")
    if any(
        payload.get(key) is not False
        for key in ("confirmatory_data_used", "training_invoked", "rl_invoked")
    ):
        raise ValueError("Phase 6 Phase-5 provenance is not measurement-only")


def sha256_path(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Phase 6 regular file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_phase6_package_lock(
    *, lock_path: Path, repository_root: Path, expected_paths: Sequence[str]
) -> str:
    from compensability_v4.qwen.phase6_runtime import verify_phase6_package_lock as verify

    return verify(
        lock_path=lock_path,
        repository_root=repository_root,
        expected_paths=tuple(expected_paths),
    )


__all__ = [
    "CompletionScore",
    "Phase6Example",
    "Phase6TrainingConfig",
    "Phase6Variant",
    "RewardGroupTrace",
    "RewardKind",
    "build_phase6_examples",
    "load_phase6_config",
    "score_phase6_completion",
    "sha256_path",
    "summarize_reward_groups",
    "validate_phase5_policy_support",
    "verify_phase6_package_lock",
]
