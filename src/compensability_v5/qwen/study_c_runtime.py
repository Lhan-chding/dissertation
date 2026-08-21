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

from compensability_v4.qwen.phase5_support import parse_world
from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation
from compensability_v5.qwen.study_c_metrics import (
    StudyCError,
    build_study_c_summary,
    read_study_c_trace,
)
from compensability_v5.training.train_support_lora import canonical_json_sha256

STUDY_C_ACK = "I_UNDERSTAND_THIS_STARTS_V5_STUDY_C_GRPO"
STUDY_C_SEED = 2026082301
STUDY_C_EVAL_SEED = 2026082302
STUDY_C_EVAL_ROLLOUTS = 16
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


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = tuple(value)
        if parts and all(isinstance(part, Mapping) for part in parts):
            content = parts[-1].get("content")  # type: ignore[union-attr]
            if isinstance(content, str):
                return content
    raise StudyCError("TRL completion has an unsupported structure")


def _expanded(values: object, size: int, label: str) -> tuple[object, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise StudyCError(f"reward metadata {label} is malformed")
    items = tuple(values)
    if len(items) == size:
        return items
    if size % len(items):
        raise StudyCError(f"reward metadata {label} cannot align to completions")
    repetitions = size // len(items)
    return tuple(item for item in items for _ in range(repetitions))


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
    answer_indices: tuple[int, int]
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
            first, second = indices
        except (TypeError, ValueError) as error:
            raise StudyCError(
                f"Study C scene {scene_id} reward metadata is invalid: {error}"
            ) from error
        return cls(
            scene_id=scene_id,
            prompt=prompt,
            truth=truth_action.world,
            answer_operator=operator,
            answer_indices=(int(first), int(second)),
            answer_label=answer,
            family=str(family),
            fiber_size=int(fiber_size),
            fiber_bin=str(fiber_bin),
            support_bin=str(support_bin),
            role=str(role),
        )

    def to_dataset_row(self) -> dict[str, object]:
        return {"prompt": self.prompt, "scene_id": self.scene_id}


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
) -> None:
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


def make_reward_function(
    *, scenes: Sequence[StudyCScene], arm: StudyCArm, trace_path: Path
) -> Callable[..., list[float]]:
    """Build one audited reward callback while scoring both registered outcomes."""

    index = {scene.scene_id: scene for scene in scenes}
    if not index or len(index) != len(scenes):
        raise StudyCError("Study C reward scenes are empty or duplicated")
    call_index = 0
    if trace_path.exists():
        if trace_path.is_symlink() or not trace_path.is_file():
            raise StudyCError("existing Study C reward trace is unsafe")
        existing = tuple(
            json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line
        )
        if existing:
            indices = tuple(row.get("reward_call_index") for row in existing)
            if any(type(item) is not int or int(item) < 0 for item in indices):
                raise StudyCError("existing Study C reward trace has invalid call indices")
            call_index = max(int(item) for item in indices) + 1

    def reward(completions: Sequence[object], **kwargs: object) -> list[float]:
        nonlocal call_index
        if (
            not isinstance(completions, Sequence)
            or isinstance(completions, (str, bytes))
            or not completions
            or len(completions) % arm.group_size
        ):
            raise StudyCError(f"Study C reward group size must be a multiple of {arm.group_size}")
        scene_ids = _expanded(kwargs.get("scene_id"), len(completions), "scene_id")
        state = kwargs.get("trainer_state")
        step_value = getattr(state, "global_step", -1)
        step = step_value if type(step_value) is int else -1
        rows: list[dict[str, object]] = []
        rewards: list[float] = []
        for position, (completion, scene_id) in enumerate(
            zip(completions, scene_ids, strict=True)
        ):
            if not isinstance(scene_id, str) or scene_id not in index:
                raise StudyCError(f"Study C reward received unknown scene: {scene_id}")
            scene = index[scene_id]
            text = _completion_text(completion)
            parsed = parse_world(text)
            candidate = None if parsed is None else WorldAction(parsed)
            truth = WorldAction(scene.truth)
            exact = candidate == truth
            answer_correct = (
                candidate is not None
                and apply_answer_operation(candidate, scene.operation) == scene.answer_label
            )
            score = float(answer_correct if arm.reward_function == "answer" else exact)
            rewards.append(score)
            rows.append(
                {
                    "schema_version": 1,
                    "trainer_step": step,
                    "reward_call_index": call_index,
                    "position": position,
                    "arm": arm.name,
                    "initialization": arm.initialization,
                    "reward_function": arm.reward_function,
                    "scene_id": scene.scene_id,
                    "family": scene.family,
                    "fiber_size": scene.fiber_size,
                    "fiber_bin": scene.fiber_bin,
                    "support_bin": scene.support_bin,
                    "completion": text,
                    "parsed_world": None if parsed is None else list(parsed),
                    "parse_success": parsed is not None,
                    "reward": score,
                    "exact_world_recovery": exact,
                    "answer_correct": answer_correct,
                    "shortcut_answer_success": answer_correct and not exact,
                    **(
                        {"rollout_seed": kwargs["rollout_seed"][position]}
                        if isinstance(kwargs.get("rollout_seed"), Sequence)
                        and not isinstance(kwargs.get("rollout_seed"), (str, bytes))
                        and len(kwargs["rollout_seed"]) == len(completions)  # type: ignore[arg-type]
                        else {}
                    ),
                }
            )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        call_index += 1
        return rewards

    reward.__name__ = f"{arm.name}_reward"
    return reward


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
EvaluationSampler = Callable[[StudyCScene, tuple[int, ...]], Sequence[object]]
EvaluationSamplerFactory = Callable[[_Trainer], EvaluationSampler]


def qwen_text_evaluation_sampler(
    *, arm: StudyCArm, model: object, processor: object
) -> EvaluationSampler:
    """Create the real fixed-seed text-only Qwen sampler after training."""

    def sample(scene: StudyCScene, seeds: tuple[int, ...]) -> Sequence[object]:
        import torch

        set_eval = getattr(model, "eval", None)
        if callable(set_eval):
            set_eval()
        apply_template = getattr(processor, "apply_chat_template", None)
        if not callable(apply_template):
            raise StudyCError("Study C processor exposes no chat template")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": scene.prompt}]}
        ]
        prompt = apply_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(prompt, str) or not prompt:
            raise StudyCError("Study C processor returned an invalid chat prompt")
        prepare = processor if callable(processor) else None
        if prepare is None:
            raise StudyCError("Study C processor is not callable")
        outputs: list[str] = []
        model_device = getattr(model, "device", None)
        generate = getattr(model, "generate", None)
        if not callable(generate):
            raise StudyCError("Study C model exposes no generate method")
        for seed in seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            batch = prepare(text=[prompt], padding=True, return_tensors="pt")
            move = getattr(batch, "to", None)
            if model_device is not None and callable(move):
                batch = move(model_device)
            if not isinstance(batch, Mapping):
                keys = getattr(batch, "keys", None)
                if not callable(keys):
                    raise StudyCError("Study C processor batch is not mapping-like")
                batch = {key: batch[key] for key in keys()}
            input_ids = batch.get("input_ids")
            shape = getattr(input_ids, "shape", None)
            if shape is None or len(shape) != 2:
                raise StudyCError("Study C processor returned malformed input IDs")
            prompt_length = int(shape[1])
            with torch.inference_mode():
                generated = generate(
                    **dict(batch),
                    do_sample=True,
                    temperature=arm.temperature,
                    top_p=arm.top_p,
                    top_k=arm.top_k,
                    max_new_tokens=arm.max_completion_length,
                    use_cache=True,
                )
            completion_ids = generated[:, prompt_length:]
            decode = getattr(processor, "batch_decode", None)
            if not callable(decode):
                tokenizer = getattr(processor, "tokenizer", None)
                decode = getattr(tokenizer, "batch_decode", None)
            if not callable(decode):
                raise StudyCError("Study C processor exposes no batch decoder")
            decoded = decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, Sequence) or len(decoded) != 1:
                raise StudyCError("Study C decoder returned malformed completion text")
            outputs.append(str(decoded[0]))
        return tuple(outputs)

    return sample


def run_post_training_frozen_eval(
    *,
    arm: StudyCArm,
    scenes: Sequence[StudyCScene],
    output_dir: Path,
    sampler: EvaluationSampler,
) -> dict[str, object]:
    """Sample 16 fixed-seed completions per frozen scene into independent artifacts."""

    trace_path = output_dir / "eval_raw_rows.jsonl"
    summary_path = output_dir / "eval_summary.json"
    if trace_path.exists() or summary_path.exists():
        raise StudyCError("post-training Study C evaluation overwrite is forbidden")
    reward = make_reward_function(scenes=scenes, arm=arm, trace_path=trace_path)
    rollout_seeds = tuple(
        STUDY_C_EVAL_SEED + index for index in range(STUDY_C_EVAL_ROLLOUTS)
    )
    for scene in scenes:
        completions = tuple(sampler(scene, rollout_seeds))
        if len(completions) != STUDY_C_EVAL_ROLLOUTS:
            raise StudyCError(
                f"post-training sampler must return {STUDY_C_EVAL_ROLLOUTS} rollouts per scene"
            )
        for start in range(0, STUDY_C_EVAL_ROLLOUTS, arm.group_size):
            stop = start + arm.group_size
            reward(
                completions[start:stop],
                scene_id=[scene.scene_id],
                rollout_seed=rollout_seeds[start:stop],
            )
    rows = read_study_c_trace(trace_path)
    rewritten = tuple(
        {**row, "trace_kind": "post_training_frozen_eval"} for row in rows
    )
    temporary = trace_path.with_suffix(".rewrite.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rewritten:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(trace_path)
    summary = build_study_c_summary({arm.name: trace_path}, group_size=arm.group_size)
    summary.update(
        {
            "measurement_scope": "post_training_frozen_eval",
            "rollouts_per_scene": STUDY_C_EVAL_ROLLOUTS,
            "evaluation_seed": STUDY_C_EVAL_SEED,
            "shared_rollout_seeds": list(rollout_seeds),
            "raw_rows_sha256": _sha256_file(trace_path),
        }
    )
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def run_study_c_arm(
    *,
    arm: StudyCArm,
    scenes: Sequence[StudyCScene],
    output_dir: Path,
    trainer_factory: TrainerFactory,
    provenance_sha256: Mapping[str, str],
    resume_from_checkpoint: Path | None = None,
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
    "build_study_c_summary",
    "load_study_c_scenes",
    "make_reward_function",
    "qwen_text_evaluation_sampler",
    "registered_study_c_arms",
    "run_post_training_frozen_eval",
    "run_study_c_arm",
    "split_study_c_scenes",
    "validate_reward_only_pair",
    "validate_study_c_config_payload",
    "validate_study_c_prompt_lengths",
]
