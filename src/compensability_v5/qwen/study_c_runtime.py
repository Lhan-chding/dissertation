"""Common-action-space, one-seed Study-C GRPO runtime.

The module keeps accelerator dependencies behind an injected trainer factory,
so reward isolation, tracing, diagnostics, and resume behavior can be tested on
CPU.  The server CLI is responsible for constructing the real TRL trainer only
after offline, acknowledgement, and hash gates have passed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation
from compensability_v5.qwen.study_c_evaluation import (
    STUDY_C_EVAL_ROLLOUTS,
    STUDY_C_EVAL_SEED,
    EvaluationSampler,
    make_reward_function,
    qwen_text_evaluation_sampler,
    run_post_training_frozen_eval,
    run_pre_training_frozen_eval,
)
from compensability_v5.qwen.study_c_metrics import (
    StudyCError,
    build_study_c_summary,
)
from compensability_v5.training.train_support_lora import canonical_json_sha256

STUDY_C_ACK = "I_UNDERSTAND_THIS_STARTS_V5_STUDY_C_GRPO"
STUDY_C_SEED = 2026082301
ACTION_PARSER_ID = "compensability_v4.qwen.phase5_support.parse_world:v1"
PRIMARY_INITIALIZATION = "B3"
SECONDARY_INITIALIZATION = "B2"
REWARD_FUNCTIONS = frozenset({"answer", "exact_state"})
_SHA256_LENGTH = 64
_TRAINING_CONTRACT: dict[str, object] = {
    "precision": "bf16",
    "learning_rate": 1.0e-6,
    "max_steps": 64,
    "group_size": 8,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "kl_beta": 0.04,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_prompt_length": 512,
    "max_completion_length": 32,
    "checkpoint_steps": 16,
    "use_vllm": False,
}
_EVALUATION_CONTRACT: dict[str, object] = {
    "rollout_count": 16,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 2026082302,
}


def validate_study_c_config_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and detach the exact registered one-seed low-cost contract."""

    expected_fields = {
        "schema_version",
        "phase",
        "action_schema",
        "prompt_files_per_scene",
        "reward_arms",
        "initializations",
        "shared_rollout_seeds",
        "seeds",
        "training",
        "evaluation",
        "data_split",
        "authorization",
        "offline",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise StudyCError("Study C config schema drifted")
    if (
        payload.get("schema_version") != 1
        or payload.get("phase") != "phase7_common_space_grpo"
        or payload.get("action_schema") != "four_comma_separated_integers"
        or payload.get("prompt_files_per_scene") != 1
        or payload.get("reward_arms") != ["answer", "exact_state"]
        or payload.get("initializations") != ["B3", "B2", "Base"]
        or payload.get("shared_rollout_seeds") is not True
        or payload.get("seeds") != [STUDY_C_SEED]
    ):
        raise StudyCError("Study C common action/seed contract drifted")
    training = payload.get("training")
    if not isinstance(training, Mapping) or dict(training) != _TRAINING_CONTRACT:
        raise StudyCError("Study C training contract drifted")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or dict(evaluation) != _EVALUATION_CONTRACT:
        raise StudyCError("Study C evaluation contract drifted")
    data_split = payload.get("data_split")
    if data_split != {
        "train_role": "rl_train",
        "eval_role": "rl_eval",
        "train_scene_count": 72,
        "eval_scene_count": 24,
        "require_scene_id_disjoint": True,
    }:
        raise StudyCError("Study C train/eval split contract drifted")
    authorization = payload.get("authorization")
    if authorization != {
        "inference_allowed": True,
        "training_allowed": True,
        "rl_allowed": True,
        "downloads_allowed": False,
    }:
        raise StudyCError("Study C authorization contract drifted")
    offline = payload.get("offline")
    if offline != {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }:
        raise StudyCError("Study C offline contract drifted")
    return {
        "training": dict(training),
        "evaluation": dict(evaluation),
        "data_split": dict(data_split),  # type: ignore[arg-type]
        "seed": STUDY_C_SEED,
    }


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StudyCError(f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class StudyCArm:
    """A closed GRPO arm; a reward pair may differ only in name and reward."""

    name: str
    initialization: str
    initialization_hash: str
    reward_function: str
    seed: int = STUDY_C_SEED
    action_space: str = "four_integer_world"
    action_parser_id: str = ACTION_PARSER_ID
    precision: str = "bf16"
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    max_completion_length: int = 32
    optimizer: str = "adamw_torch"
    learning_rate: float = 1.0e-6
    beta: float = 0.04
    steps: int = 64
    group_size: int = 8
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 512
    checkpoint_steps: int = 16
    use_vllm: bool = False

    def __post_init__(self) -> None:
        if self.initialization not in {PRIMARY_INITIALIZATION, SECONDARY_INITIALIZATION}:
            raise StudyCError("Study C initialization must be B3 or optional B2")
        if self.name != f"{self.initialization}_{self.reward_function}":
            raise StudyCError("Study C arm name does not encode initialization and reward")
        if self.reward_function not in REWARD_FUNCTIONS:
            raise StudyCError("Study C reward function is not registered")
        if not _valid_sha256(self.initialization_hash):
            raise StudyCError("Study C initialization hash must be lowercase SHA-256")
        if self.seed != STUDY_C_SEED:
            raise StudyCError(f"Study C pilot seed must be {STUDY_C_SEED}")
        if self.action_space != "four_integer_world" or self.action_parser_id != ACTION_PARSER_ID:
            raise StudyCError("Study C action space/parser drifted")
        if self.group_size < 2 or self.steps <= 0 or self.checkpoint_steps <= 0:
            raise StudyCError("Study C step/group/checkpoint values must be positive")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def build_grpo_config_kwargs(
    arm: StudyCArm, output_dir: Path, supported_parameters: Sequence[str]
) -> dict[str, object]:
    """Build the shared TRL args, passing max_prompt_length only when supported."""

    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "learning_rate": arm.learning_rate,
        "max_steps": arm.steps,
        "num_generations": arm.group_size,
        "per_device_train_batch_size": arm.per_device_train_batch_size,
        "gradient_accumulation_steps": arm.gradient_accumulation_steps,
        "max_completion_length": arm.max_completion_length,
        "bf16": True,
        "fp16": False,
        "temperature": arm.temperature,
        "top_p": arm.top_p,
        "top_k": arm.top_k,
        "beta": arm.beta,
        "use_vllm": arm.use_vllm,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": arm.checkpoint_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": arm.seed,
        "data_seed": arm.seed,
        "optim": arm.optimizer,
    }
    if "max_prompt_length" in supported_parameters:
        kwargs["max_prompt_length"] = arm.max_prompt_length
    return kwargs


def _reward_pair(initialization: str, initialization_hash: str) -> tuple[StudyCArm, ...]:
    return tuple(
        StudyCArm(
            name=f"{initialization}_{reward}",
            initialization=initialization,
            initialization_hash=initialization_hash,
            reward_function=reward,
        )
        for reward in ("answer", "exact_state")
    )


def registered_study_c_arms(
    *,
    initialization: str,
    initialization_hash: str,
    include_b2: bool = False,
    b2_initialization_hash: str | None = None,
) -> tuple[StudyCArm, ...]:
    """Return the B3 primary pair and, only on request, the B2 secondary pair."""

    if initialization != PRIMARY_INITIALIZATION:
        raise StudyCError("the primary Study C initialization must be B3")
    arms = _reward_pair(initialization, initialization_hash)
    if include_b2:
        if not _valid_sha256(b2_initialization_hash):
            raise StudyCError("--include-b2 requires a hash-bound B2 initialization")
        arms += _reward_pair(SECONDARY_INITIALIZATION, str(b2_initialization_hash))
    validate_reward_only_pair(arms)
    return arms


def validate_reward_only_pair(arms: Sequence[StudyCArm]) -> None:
    """Prove that each initialization pair differs only in reward and arm name."""

    if not arms:
        raise StudyCError("Study C arm collection is empty")
    by_initialization: dict[str, list[StudyCArm]] = defaultdict(list)
    for arm in arms:
        if not isinstance(arm, StudyCArm):
            raise StudyCError("Study C arms must be StudyCArm instances")
        by_initialization[arm.initialization].append(arm)
    expected = {PRIMARY_INITIALIZATION}
    if SECONDARY_INITIALIZATION in by_initialization:
        expected.add(SECONDARY_INITIALIZATION)
    if set(by_initialization) != expected:
        raise StudyCError("Study C initialization set drifted")
    for initialization, pair in by_initialization.items():
        if len(pair) != 2 or {arm.reward_function for arm in pair} != REWARD_FUNCTIONS:
            raise StudyCError(f"{initialization} must contain answer and exact-state arms")
        left, right = (arm.to_mapping() for arm in pair)
        differences = {key for key in left if left[key] != right[key]}
        if differences != {"name", "reward_function"}:
            raise StudyCError(
                f"{initialization} pair differs outside reward: {sorted(differences)}"
            )


@dataclass(frozen=True, slots=True)
class StudyCScene:
    """Immutable training metadata for one shared world-output prompt."""

    scene_id: str
    prompt: str
    truth: tuple[int, int, int, int]
    answer_operator: str
    answer_indices: tuple[int, ...]
    answer_label: int
    family: str
    fiber_size: int
    fiber_bin: str
    support_bin: str
    role: str

    @property
    def operation(self) -> dict[str, object]:
        return {"operator": self.answer_operator, "indices": list(self.answer_indices)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StudyCScene:
        if not isinstance(value, Mapping):
            raise StudyCError("Study C scene must be a mapping")
        scene_id, prompt = value.get("scene_id"), value.get("prompt")
        family, fiber_bin, support_bin = (
            value.get("family"),
            value.get("fiber_bin"),
            value.get("support_bin"),
        )
        role = value.get("role")
        if not isinstance(scene_id, str) or not scene_id:
            raise StudyCError("Study C scene_id must be non-empty")
        if not isinstance(prompt, str) or not prompt.strip():
            raise StudyCError(f"Study C scene {scene_id} has no shared prompt")
        for label, item in (
            ("family", family),
            ("fiber_bin", fiber_bin),
            ("support_bin", support_bin),
        ):
            if not isinstance(item, str) or not item:
                raise StudyCError(f"Study C scene {scene_id} requires frozen {label}")
        fiber_size = value.get("fiber_size")
        if type(fiber_size) is not int or int(fiber_size) <= 0:
            raise StudyCError(f"Study C scene {scene_id} requires positive fiber_size")
        if role not in {"rl_train", "rl_eval"}:
            raise StudyCError(f"Study C scene {scene_id} requires rl_train/rl_eval role")
        try:
            truth_action = WorldAction.from_mapping({"world": value.get("truth")})
            operation = value.get("answer_operation")
            if not isinstance(operation, Mapping):
                raise TypeError("answer_operation must be a mapping")
            labels = value.get("reward_labels")
            if not isinstance(labels, Mapping) or set(labels) != {"answer", "exact_state"}:
                raise TypeError("both frozen reward labels are required")
            if labels.get("exact_state") != list(truth_action.world):
                raise ValueError("exact-state reward label differs from truth")
            answer = apply_answer_operation(truth_action, operation)
            if labels.get("answer") != answer:
                raise ValueError("answer reward label differs from truth projection")
            operator = operation.get("operator")
            indices = operation.get("indices")
            if not isinstance(operator, str) or not isinstance(indices, list):
                raise TypeError("answer operation is malformed")
        except (TypeError, ValueError) as error:
            raise StudyCError(
                f"Study C scene {scene_id} reward metadata is invalid: {error}"
            ) from error
        return cls(
            scene_id=scene_id,
            prompt=prompt,
            truth=truth_action.world,
            answer_operator=operator,
            answer_indices=tuple(int(index) for index in indices),
            answer_label=answer,
            family=str(family),
            fiber_size=int(fiber_size),
            fiber_bin=str(fiber_bin),
            support_bin=str(support_bin),
            role=str(role),
        )

    def to_dataset_row(self) -> dict[str, object]:
        return {"prompt": self.prompt, "scene_id": self.scene_id, "role": self.role}


def load_study_c_scenes(package: Mapping[str, object]) -> tuple[StudyCScene, ...]:
    scenes = package.get("scenes") if isinstance(package, Mapping) else None
    if not isinstance(scenes, (list, tuple)) or not scenes:
        raise StudyCError("common-action package contains no scenes")
    result = tuple(
        StudyCScene.from_mapping(scene) for scene in scenes if isinstance(scene, Mapping)
    )
    if len(result) != len(scenes):
        raise StudyCError("common-action package contains a malformed scene")
    if len({scene.scene_id for scene in result}) != len(result):
        raise StudyCError("common-action package contains duplicate scene IDs")
    if package.get("rollout_seeds") != [STUDY_C_SEED]:
        raise StudyCError(f"common-action package must freeze seed {STUDY_C_SEED}")
    parser_id = package.get("action_parser_id")
    if parser_id != ACTION_PARSER_ID:
        raise StudyCError("common-action package action parser differs from Study C runtime")
    return result


def split_study_c_scenes(
    scenes: Sequence[StudyCScene],
    *,
    expected_train_count: int | None = None,
    expected_eval_count: int | None = None,
) -> tuple[tuple[StudyCScene, ...], tuple[StudyCScene, ...]]:
    """Partition frozen roles and reject missing, overlapping, or count-drifted splits."""

    train = tuple(scene for scene in scenes if scene.role == "rl_train")
    evaluation = tuple(scene for scene in scenes if scene.role == "rl_eval")
    if not train or not evaluation or len(train) + len(evaluation) != len(scenes):
        raise StudyCError("Study C requires non-empty closed rl_train and rl_eval roles")
    train_ids = {scene.scene_id for scene in train}
    evaluation_ids = {scene.scene_id for scene in evaluation}
    if train_ids & evaluation_ids:
        raise StudyCError("Study C train/eval scene IDs overlap")
    if expected_train_count is not None and len(train) != expected_train_count:
        raise StudyCError(
            f"Study C rl_train count must be {expected_train_count}, observed {len(train)}"
        )
    if expected_eval_count is not None and len(evaluation) != expected_eval_count:
        raise StudyCError(
            f"Study C rl_eval count must be {expected_eval_count}, observed {len(evaluation)}"
        )
    return train, evaluation


def validate_study_c_prompt_lengths(
    scenes: Sequence[StudyCScene], processor: object, *, max_prompt_length: int
) -> int:
    """Enforce the frozen prompt limit without asking TRL to truncate inputs."""

    tokenizer = getattr(processor, "tokenizer", None)
    if not callable(tokenizer):
        raise StudyCError("Study C processor exposes no callable tokenizer")
    encoded = tokenizer(
        [scene.prompt for scene in scenes],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(token_ids, Sequence) or len(token_ids) != len(scenes):
        raise StudyCError("Study C tokenizer returned malformed prompt token IDs")
    for scene, values in zip(scenes, token_ids, strict=True):
        if not isinstance(values, Sequence):
            raise StudyCError("Study C tokenizer returned malformed prompt token IDs")
        if len(values) > max_prompt_length:
            raise StudyCError(
                f"Study C prompt exceeds {max_prompt_length} tokens: {scene.scene_id}"
            )
    return max(len(values) for values in token_ids)


class StudyCRewardTraceCallback:
    """Snapshot the append-only reward trace into each trainer checkpoint."""

    def __init__(self, trace_path: Path, output_dir: Path) -> None:
        self.trace_path = trace_path
        self.output_dir = output_dir

    def on_save(self, args: object, state: object, control: object, **kwargs: object) -> object:
        step = getattr(state, "global_step", None)
        if type(step) is not int or step < 0 or not self.trace_path.is_file():
            raise StudyCError("cannot checkpoint an incomplete Study C reward trace")
        checkpoint = self.output_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.trace_path, checkpoint / "raw_reward_trace.jsonl")
        return control


def _restore_trace(trace_path: Path, checkpoint: Path | None, output_dir: Path) -> None:
    if checkpoint is None:
        if output_dir.exists() or output_dir.is_symlink():
            raise StudyCError(f"fresh Study C output exists; overwrite forbidden: {output_dir}")
        return
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise StudyCError("Study C resume checkpoint is missing or unsafe")
    try:
        checkpoint.resolve().relative_to(output_dir.resolve())
    except ValueError as error:
        raise StudyCError("Study C resume checkpoint must be inside its arm output") from error
    snapshot = checkpoint / "raw_reward_trace.jsonl"
    if snapshot.is_symlink() or not snapshot.is_file():
        raise StudyCError("Study C checkpoint is missing its reward-trace snapshot")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trace_path.with_suffix(".restore.tmp")
    shutil.copyfile(snapshot, temporary)
    temporary.replace(trace_path)


class _Trainer(Protocol):
    state: object

    def train(self, *, resume_from_checkpoint: str | None = None) -> object: ...

    def save_model(self, output_dir: str) -> object: ...


TrainerFactory = Callable[..., _Trainer]
EvaluationSamplerFactory = Callable[[_Trainer], EvaluationSampler]


def _resume_pre_training_summary(output_dir: Path) -> dict[str, object]:
    trace_path = output_dir / "pre_training_eval_raw_rows.jsonl"
    summary_path = output_dir / "pre_training_eval_summary.json"
    if not trace_path.is_file() or trace_path.is_symlink():
        raise StudyCError("resume checkpoint is missing the immutable pre-training trace")
    if not summary_path.is_file() or summary_path.is_symlink():
        raise StudyCError("resume checkpoint is missing the pre-training summary")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyCError("pre-training summary is malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("measurement_scope") != "pre_training_frozen_eval"
        or payload.get("rollouts_per_scene") != STUDY_C_EVAL_ROLLOUTS
        or payload.get("evaluation_seed") != STUDY_C_EVAL_SEED
        or payload.get("raw_rows_sha256") != _sha256_file(trace_path)
    ):
        raise StudyCError("pre-training summary does not match its frozen trace")
    return payload


def run_study_c_arm(
    *,
    arm: StudyCArm,
    scenes: Sequence[StudyCScene],
    output_dir: Path,
    trainer_factory: TrainerFactory,
    provenance_sha256: Mapping[str, str],
    resume_from_checkpoint: Path | None = None,
    pre_training_evaluation_sampler_factory: EvaluationSamplerFactory | None = None,
    evaluation_sampler_factory: EvaluationSamplerFactory | None = None,
) -> dict[str, object]:
    """Execute one arm through an injected TRL-compatible trainer factory."""

    if not scenes:
        raise StudyCError("Study C arm requires at least one scene")
    train_scenes, evaluation_scenes = split_study_c_scenes(scenes)
    if any(not _valid_sha256(digest) for digest in provenance_sha256.values()):
        raise StudyCError("Study C provenance values must be lowercase SHA-256")
    evidence_path = output_dir / "execution_evidence.json"
    if evidence_path.exists():
        raise StudyCError(f"Study C arm is already complete; overwrite forbidden: {arm.name}")
    trace_path = output_dir / "raw_reward_trace.jsonl"
    _restore_trace(trace_path, resume_from_checkpoint, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward = make_reward_function(scenes=train_scenes, arm=arm, trace_path=trace_path)
    callback = StudyCRewardTraceCallback(trace_path, output_dir)
    trainer = trainer_factory(
        arm=arm,
        dataset=tuple(scene.to_dataset_row() for scene in train_scenes),
        reward_function=reward,
        output_dir=output_dir,
        callbacks=(callback,),
    )
    pre_training_summary: dict[str, object] | None = None
    if pre_training_evaluation_sampler_factory is not None:
        if resume_from_checkpoint is None:
            print(
                f"PROGRESS: Study C {arm.name} pre-training frozen evaluation",
                flush=True,
            )
            pre_training_summary = run_pre_training_frozen_eval(
                arm=arm,
                scenes=evaluation_scenes,
                output_dir=output_dir,
                sampler=pre_training_evaluation_sampler_factory(trainer),
            )
        else:
            print(
                f"PROGRESS: Study C {arm.name} verifying resumed pre-training evaluation",
                flush=True,
            )
            pre_training_summary = _resume_pre_training_summary(output_dir)
    print(
        f"PROGRESS: Study C {arm.name} training {arm.steps} optimizer steps",
        flush=True,
    )
    trainer.train(
        resume_from_checkpoint=None
        if resume_from_checkpoint is None
        else str(resume_from_checkpoint)
    )
    final_adapter = output_dir / "final_adapter"
    if final_adapter.exists():
        raise StudyCError("Study C final adapter already exists; overwrite forbidden")
    trainer.save_model(str(final_adapter))
    if not final_adapter.is_dir() or not any(final_adapter.iterdir()):
        raise StudyCError("Study C trainer did not save a final adapter")
    evaluation_summary: dict[str, object] | None = None
    if evaluation_sampler_factory is not None:
        print(
            f"PROGRESS: Study C {arm.name} training complete; post-training frozen evaluation",
            flush=True,
        )
        evaluation_summary = run_post_training_frozen_eval(
            arm=arm,
            scenes=evaluation_scenes,
            output_dir=output_dir,
            sampler=evaluation_sampler_factory(trainer),
        )
    logs = getattr(getattr(trainer, "state", None), "log_history", None)
    if not isinstance(logs, list):
        raise StudyCError("Study C trainer exposes no log history")
    metrics_path = output_dir / "trainer_log_history.json"
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(logs, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    diagnostics = build_study_c_summary({arm.name: trace_path}, group_size=arm.group_size)
    diagnostics_path = output_dir / "group_diagnostics.json"
    with diagnostics_path.open("x", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    adapter_files = tuple(sorted(path for path in final_adapter.rglob("*") if path.is_file()))
    adapter_digest = hashlib.sha256()
    for path in adapter_files:
        adapter_digest.update(path.relative_to(final_adapter).as_posix().encode("utf-8"))
        adapter_digest.update(b"\0")
        adapter_digest.update(path.read_bytes())
        adapter_digest.update(b"\n")
    evidence: dict[str, object] = {
        "schema_version": 1,
        "status": "STUDY_C_ARM_COMPLETE",
        "arm": arm.to_mapping(),
        "scene_count": len(scenes),
        "train_scene_count": len(train_scenes),
        "eval_scene_count": len(evaluation_scenes),
        "train_scene_manifest_sha256": canonical_json_sha256(
            [scene.to_dataset_row() for scene in train_scenes]
        ),
        "eval_scene_manifest_sha256": canonical_json_sha256(
            [scene.to_dataset_row() for scene in evaluation_scenes]
        ),
        "provenance_sha256": dict(sorted(provenance_sha256.items())),
        "reward_trace_sha256": _sha256_file(trace_path),
        "trainer_log_sha256": _sha256_file(metrics_path),
        "diagnostics_sha256": _sha256_file(diagnostics_path),
        "post_training_evaluation_invoked": evaluation_summary is not None,
        "pre_training_evaluation_invoked": pre_training_summary is not None,
        "pre_training_eval_raw_rows_sha256": None
        if pre_training_summary is None
        else _sha256_file(output_dir / "pre_training_eval_raw_rows.jsonl"),
        "pre_training_eval_summary_sha256": None
        if pre_training_summary is None
        else _sha256_file(output_dir / "pre_training_eval_summary.json"),
        "eval_raw_rows_sha256": None
        if evaluation_summary is None
        else _sha256_file(output_dir / "eval_raw_rows.jsonl"),
        "eval_summary_sha256": None
        if evaluation_summary is None
        else _sha256_file(output_dir / "eval_summary.json"),
        "final_adapter_tree_sha256": adapter_digest.hexdigest(),
        "resumed_from_checkpoint": None
        if resume_from_checkpoint is None
        else str(resume_from_checkpoint.resolve()),
        "training_invoked": True,
        "rl_invoked": True,
        "raw_reward_trace_preserved": True,
        "output_overwrite_allowed": False,
        "prompt_length_enforcement": getattr(
            trainer,
            "_study_c_prompt_length_audit",
            {
                "mode": "external_preflight",
                "limit": arm.max_prompt_length,
                "max_observed": None,
                "passed_to_grpo_config": False,
            },
        ),
    }
    with evidence_path.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return evidence


__all__ = [
    "ACTION_PARSER_ID",
    "PRIMARY_INITIALIZATION",
    "SECONDARY_INITIALIZATION",
    "STUDY_C_ACK",
    "STUDY_C_EVAL_ROLLOUTS",
    "STUDY_C_EVAL_SEED",
    "STUDY_C_SEED",
    "StudyCArm",
    "StudyCError",
    "StudyCRewardTraceCallback",
    "StudyCScene",
    "build_grpo_config_kwargs",
    "build_study_c_summary",
    "load_study_c_scenes",
    "make_reward_function",
    "qwen_text_evaluation_sampler",
    "registered_study_c_arms",
    "run_post_training_frozen_eval",
    "run_pre_training_frozen_eval",
    "run_study_c_arm",
    "split_study_c_scenes",
    "validate_reward_only_pair",
    "validate_study_c_config_payload",
    "validate_study_c_prompt_lengths",
]
