"""Fail-closed, foreground Stage 25 Study C2 GRPO orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.model_loader import MODEL_SNAPSHOT_SHA256, require_server_model
from compensability_v5.qwen.study_b_backend import (
    require_offline_environment,
    verify_runtime_package_lock,
)
from compensability_v5.qwen.study_b_runtime import tree_sha256

from .action_protocol import parse_first_world_action
from .io import read_json, read_jsonl, sha256_file
from .paths import (
    FIBER_ROWS,
    SHARED_GRADIENT_MANIFEST,
    SHARED_GRADIENT_ROWS,
    SHARED_GRADIENT_SUMMARY,
    STAGE25_EXECUTION_CONTRACT,
)
from .rewards import classify_world
from .schemas import build_reward_arm_configs
from .stages import load_contract

PACKAGE_LOCK = Path("configs/v5/server_package_lock.yaml")
TRAINING_ACK = "I_UNDERSTAND_THIS_RUNS_STUDY_C2_IDENTIFIABLE_REWARD_GRPO"
_ARMS = ("answer", "state")
_KINDS = ("X", "S", "F", "U")
_EXPECTED_PROMPTS = 192
_EXPECTED_PAIRS = 96
_CHECKPOINT_INTERVAL = 48
_CHECKPOINT_STEPS = (48, 96, 144, 192)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_stage25_execution_contract(payload: Mapping[str, object]) -> None:
    """Validate the exact returned Stage 24 facts that authorize main RL."""

    expected = {
        "schema_version",
        "status",
        "stage24_per_group_sha256",
        "stage24_summary_sha256",
        "stage24_manifest_sha256",
        "stage24_execution_contract_sha256",
        "fiber_rows_sha256",
        "config_sha256",
        "package_lock_sha256",
        "b3_adapter_sha256",
        "model_snapshot_sha256",
        "selected_k",
        "training_prompt_count",
        "shared_gradient_group_count",
        "reward_hamming_distance",
        "continue_to_main_rl",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("schema_version") != 2
        or payload.get("status") != "STUDY_C2_STAGE25_EXECUTION_CONTRACT_FROZEN"
        or payload.get("selected_k") != 8
        or payload.get("training_prompt_count") != _EXPECTED_PROMPTS
        or payload.get("shared_gradient_group_count") != 768
        or payload.get("reward_hamming_distance") != 635
        or payload.get("continue_to_main_rl") is not True
        or payload.get("model_snapshot_sha256") != MODEL_SNAPSHOT_SHA256
    ):
        raise ValueError("Study C2 Stage 25 execution contract drifted")
    digest_keys = expected - {
        "schema_version",
        "status",
        "selected_k",
        "training_prompt_count",
        "shared_gradient_group_count",
        "reward_hamming_distance",
        "continue_to_main_rl",
    }
    for key in digest_keys:
        if not _valid_digest(payload.get(key)):
            raise ValueError(f"Study C2 Stage 25 contract has invalid {key}")


def select_training_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Select the frozen 96 matched train pairs without changing file order."""

    selected = tuple(dict(row) for row in rows if row.get("split") == "train")
    if len(selected) != _EXPECTED_PROMPTS:
        raise ValueError(
            f"Study C2 Stage 25 requires exactly 192 training prompts, observed {len(selected)}"
        )
    scene_ids = [row.get("scene_id") for row in selected]
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids):
        raise ValueError("Study C2 training rows require non-empty scene IDs")
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("Study C2 training rows contain duplicate scene IDs")
    by_pair: dict[str, list[Mapping[str, object]]] = {}
    for row in selected:
        pair_id = row.get("pair_id")
        condition = row.get("condition")
        if not isinstance(pair_id, str) or condition not in {"collision", "separating"}:
            raise ValueError("Study C2 training rows require paired condition metadata")
        by_pair.setdefault(pair_id, []).append(row)
        if (
            not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("prompt_sha256"), str)
            or not isinstance(row.get("truth"), list)
            or not isinstance(row.get("operation"), Mapping)
        ):
            raise ValueError("Study C2 training row reward metadata is malformed")
    if len(by_pair) != _EXPECTED_PAIRS or any(
        len(pair) != 2 or {row["condition"] for row in pair} != {"collision", "separating"}
        for pair in by_pair.values()
    ):
        raise ValueError("Study C2 training rows are not 96 complete paired scenes")
    return selected


def expected_optimizer_steps(
    rows: Sequence[Mapping[str, object]], *, group_size: int
) -> int:
    if group_size != 8:
        raise ValueError("Study C2 Stage 25 selected K must be 8")
    if len(rows) != _EXPECTED_PROMPTS:
        raise ValueError("Study C2 optimizer-step calculation requires 192 prompts")
    return len(rows)


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        messages = tuple(value)
        if messages and all(isinstance(message, Mapping) for message in messages):
            content = messages[-1].get("content")  # type: ignore[union-attr]
            if isinstance(content, str):
                return content
    raise ValueError("TRL completion has an unsupported structure")


def _expand_metadata(values: object, size: int, label: str) -> tuple[object, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"reward metadata {label} is malformed")
    items = tuple(values)
    if len(items) == size:
        return items
    if size % len(items):
        raise ValueError(f"reward metadata {label} cannot align to completions")
    repetitions = size // len(items)
    return tuple(item for item in items for _ in range(repetitions))


def build_traced_reward(
    *,
    arm_config: Mapping[str, object],
    training_rows: Sequence[Mapping[str, object]],
    trace_path: Path,
    group_size: int,
) -> Callable[..., list[float]]:
    """Score one verifier while preserving both labels for every sampled action."""

    if arm_config.get("reward_function_id") not in {
        "answer_reward_v1",
        "exact_state_reward_v1",
    }:
        raise ValueError("Study C2 reward function is unregistered")
    index = {str(row["scene_id"]): dict(row) for row in training_rows}
    if not index or len(index) != len(training_rows):
        raise ValueError("Study C2 reward rows are empty or duplicated")
    group_index = 0
    call_index = 0
    if trace_path.exists():
        existing = read_jsonl(trace_path)
        group_indices = [row.get("group_index") for row in existing]
        call_indices = [row.get("reward_call_index") for row in existing]
        if any(type(value) is not int or int(value) < 0 for value in group_indices + call_indices):
            raise ValueError("existing Study C2 reward trace indices are malformed")
        group_index = max(int(value) for value in group_indices) + 1
        call_index = max(int(value) for value in call_indices) + 1

    def reward(completions: Sequence[object], **kwargs: object) -> list[float]:
        nonlocal call_index, group_index
        if (
            not isinstance(completions, Sequence)
            or isinstance(completions, (str, bytes))
            or not completions
            or len(completions) % group_size
        ):
            raise ValueError(f"Study C2 reward batch must be a multiple of K={group_size}")
        scene_ids = _expand_metadata(kwargs.get("scene_id"), len(completions), "scene_id")
        state = kwargs.get("trainer_state")
        raw_step = getattr(state, "global_step", -1)
        trainer_step = raw_step if type(raw_step) is int else -1
        scored: list[float] = []
        trace_rows: list[dict[str, object]] = []
        for start in range(0, len(completions), group_size):
            group_completions = completions[start : start + group_size]
            group_scene_ids = scene_ids[start : start + group_size]
            if len(set(group_scene_ids)) != 1:
                raise ValueError("Study C2 reward group crosses prompt boundaries")
            scene_id = group_scene_ids[0]
            if not isinstance(scene_id, str) or scene_id not in index:
                raise ValueError(f"Study C2 reward received unknown scene: {scene_id}")
            row = index[scene_id]
            truth = tuple(row["truth"])
            operation = row["operation"]
            if len(truth) != 4 or not isinstance(operation, Mapping):
                raise ValueError("Study C2 reward row truth/operation is malformed")
            for position, completion in enumerate(group_completions):
                text = _completion_text(completion)
                parsed = parse_first_world_action(text)
                classified = classify_world(
                    parsed,
                    truth=truth,  # type: ignore[arg-type]
                    operation=operation,
                )
                selected_reward = (
                    classified.answer_reward
                    if arm_config["reward_function_id"] == "answer_reward_v1"
                    else classified.state_reward
                )
                scored.append(float(selected_reward))
                trace_rows.append(
                    {
                        "schema_version": 2,
                        "arm": arm_config["name"],
                        "reward_function_id": arm_config["reward_function_id"],
                        "trainer_step": trainer_step,
                        "reward_call_index": call_index,
                        "group_index": group_index,
                        "position": position,
                        "scene_id": scene_id,
                        "pair_id": row["pair_id"],
                        "condition": row["condition"],
                        "family": row["family"],
                        "prompt_sha256": row["prompt_sha256"],
                        "completion": text,
                        "parsed_world": None if parsed is None else list(parsed),
                        "parse_success": parsed is not None,
                        "kind": classified.kind.value,
                        "answer_reward": classified.answer_reward,
                        "state_reward": classified.state_reward,
                        "reward": float(selected_reward),
                    }
                )
            group_index += 1
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as stream:
            for row in trace_rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        call_index += 1
        return scored

    reward.__name__ = str(arm_config["reward_function_id"])
    return reward


def build_grpo_config_kwargs(
    *,
    arm_config: Mapping[str, object],
    output_dir: Path,
    group_size: int,
    eos_token_id: int,
    newline_token_id: int,
    supported_parameters: Sequence[str] | set[str],
) -> dict[str, object]:
    training = arm_config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Study C2 arm has no training contract")
    supported = set(supported_parameters)
    required = {
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
    }
    missing = sorted(required - supported)
    if missing:
        raise RuntimeError(f"installed TRL GRPOConfig lacks required fields: {missing}")
    if type(eos_token_id) is not int or type(newline_token_id) is not int:
        raise ValueError("Study C2 stopping token IDs must be integers")
    kwargs: dict[str, object] = {
        "output_dir": str(output_dir),
        "learning_rate": float(training["learning_rate"]),
        "num_train_epochs": int(training["epochs"]),
        "num_generations": group_size,
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "max_completion_length": int(training["max_completion_length"]),
        "bf16": True,
        "fp16": False,
        "temperature": float(training["temperature"]),
        "top_p": float(training["top_p"]),
        "beta": float(training["kl_beta"]),
        "use_vllm": False,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": _CHECKPOINT_INTERVAL,
        "save_total_limit": 4,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": int(arm_config["seed"]),
        "data_seed": int(arm_config["seed"]),
        "optim": str(training["optimizer"]),
        "shuffle_dataset": False,
        "generation_kwargs": {"eos_token_id": [eos_token_id, newline_token_id]},
    }
    if "max_prompt_length" in supported:
        kwargs["max_prompt_length"] = int(training["max_prompt_length"])
    return {key: value for key, value in kwargs.items() if key in supported}


def truncate_first_line_token_ids(
    *,
    completion_ids: Sequence[Sequence[int]],
    logprobs: Sequence[Sequence[float]] | None,
    newline_token_id: int,
    eos_token_id: int,
) -> tuple[list[list[int]], list[list[float]] | None]:
    """Remove batch padding after the newline/EOS stopping token."""

    if logprobs is not None and len(logprobs) != len(completion_ids):
        raise RuntimeError("Study C2 completion IDs and logprobs do not align")
    truncated_ids: list[list[int]] = []
    truncated_logprobs: list[list[float]] | None = [] if logprobs is not None else None
    for index, raw_ids in enumerate(completion_ids):
        ids = list(raw_ids)
        stop = len(ids)
        for position, token in enumerate(ids):
            if token in {newline_token_id, eos_token_id}:
                stop = position + 1
                break
        truncated_ids.append(ids[:stop])
        if logprobs is not None:
            values = list(logprobs[index])
            if len(values) != len(ids):
                raise RuntimeError("Study C2 completion IDs and logprobs do not align")
            assert truncated_logprobs is not None
            truncated_logprobs.append(values[:stop])
    return truncated_ids, truncated_logprobs


class TrainingProgressCallback:
    """Print every optimizer step and snapshot the append-only trace at checkpoints."""

    def __init__(
        self,
        arm_name: str,
        *,
        total_steps: int,
        trace_path: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.arm_name = arm_name
        self.total_steps = total_steps
        self.trace_path = trace_path
        self.output_dir = output_dir

    def _passthrough(
        self, args: object, state: object, control: object, **kwargs: object
    ) -> object:
        return control

    on_init_end = _passthrough
    on_train_begin = _passthrough
    on_train_end = _passthrough
    on_epoch_begin = _passthrough
    on_epoch_end = _passthrough
    on_step_begin = _passthrough
    on_pre_optimizer_step = _passthrough
    on_optimizer_step = _passthrough
    on_substep_end = _passthrough
    on_evaluate = _passthrough
    on_predict = _passthrough
    on_log = _passthrough
    on_prediction_step = _passthrough
    on_push_begin = _passthrough

    def on_step_end(
        self, args: object, state: object, control: object, **kwargs: object
    ) -> object:
        step = getattr(state, "global_step", None)
        if type(step) is not int or step < 0:
            raise RuntimeError("Study C2 trainer did not expose an integer optimizer step")
        print(
            f"PROGRESS: {self.arm_name} optimizer step {step}/{self.total_steps}",
            flush=True,
        )
        return control

    def on_save(
        self, args: object, state: object, control: object, **kwargs: object
    ) -> object:
        step = getattr(state, "global_step", None)
        if (
            type(step) is not int
            or self.trace_path is None
            or self.output_dir is None
            or self.trace_path.is_symlink()
            or not self.trace_path.is_file()
        ):
            raise RuntimeError("cannot checkpoint an incomplete Study C2 reward trace")
        checkpoint = self.output_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.trace_path, checkpoint / "raw_reward_trace.jsonl")
        return control


def build_training_group_diagnostics(
    *, trace_path: Path, group_size: int, expected_group_count: int
) -> tuple[dict[str, object], ...]:
    rows = read_jsonl(trace_path)
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        index = row.get("group_index")
        if type(index) is not int or index < 0:
            raise ValueError("Study C2 trace contains an invalid group index")
        grouped.setdefault(index, []).append(row)
    if set(grouped) != set(range(expected_group_count)):
        raise ValueError("Study C2 trace group count/order differs from the frozen epoch")
    diagnostics: list[dict[str, object]] = []
    for group_index in range(expected_group_count):
        group = grouped[group_index]
        if len(group) != group_size:
            raise ValueError("Study C2 reward trace group size differs from selected K")
        kinds = [row.get("kind") for row in group]
        if any(kind not in _KINDS for kind in kinds):
            raise ValueError("Study C2 reward trace contains an invalid X/S/F/U kind")
        numeric = [row.get("reward") for row in group]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("Study C2 reward trace contains a non-finite reward")
        scene_ids = {row.get("scene_id") for row in group}
        if len(scene_ids) != 1:
            raise ValueError("Study C2 reward trace group crosses scenes")
        counts = Counter(str(kind) for kind in kinds)
        diagnostics.append(
            {
                "schema_version": 2,
                "group_index": group_index,
                "scene_id": group[0]["scene_id"],
                "pair_id": group[0].get("pair_id"),
                "condition": group[0].get("condition"),
                "family": group[0].get("family"),
                "counts": {kind: counts.get(kind, 0) for kind in _KINDS},
                "reward_hamming_distance": sum(
                    row.get("answer_reward") != row.get("state_reward") for row in group
                ),
                "selected_reward_sum": sum(float(row["reward"]) for row in group),
            }
        )
    return tuple(diagnostics)


def _validate_arm_pair(arms: Sequence[Mapping[str, object]]) -> None:
    if len(arms) != 2:
        raise ValueError("Study C2 requires exactly two registered arms")
    left, right = (dict(arm) for arm in arms)
    differences = {key for key in left if left.get(key) != right.get(key)}
    if differences != {"name", "reward_function_id", "output_directory"}:
        raise ValueError(f"Study C2 arms drift outside reward isolation: {sorted(differences)}")


def _arm_config(arms: Sequence[Mapping[str, object]], arm: str) -> dict[str, object]:
    expected_name = "C2_answer_reward" if arm == "answer" else "C2_exact_state_reward"
    matches = [dict(value) for value in arms if value.get("name") == expected_name]
    if arm not in _ARMS or len(matches) != 1:
        raise ValueError("--arm must select one registered Study C2 reward arm")
    return matches[0]


def _require_offline_cuda() -> None:  # pragma: no cover - server-only dependency gate
    require_offline_environment()
    verify_runtime_package_lock(PACKAGE_LOCK)


def preflight_training_arm(
    *,
    arm: str,
    config_path: Path,
    execution_contract_path: Path = STAGE25_EXECUTION_CONTRACT,
    b3_adapter: Path,
    b3_sha256: str,
    backend_validator: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:  # pragma: no cover - server dependencies are mocked in tests
    print("PROGRESS: validating the frozen Stage 25 execution contract", flush=True)
    execution = read_json(execution_contract_path)
    validate_stage25_execution_contract(execution)
    contract = load_contract(config_path)
    training_rows = select_training_rows(read_jsonl(FIBER_ROWS))
    print("PROGRESS: verifying offline mode, package lock, and single bf16 4090", flush=True)
    _require_offline_cuda()
    print("PROGRESS: hashing the immutable Qwen snapshot and Study B B3 adapter", flush=True)
    require_server_model()
    if b3_sha256 != execution["b3_adapter_sha256"]:
        raise ValueError("operator B3 SHA-256 differs from the Stage 25 execution contract")
    observed_adapter = tree_sha256(b3_adapter)
    if observed_adapter != b3_sha256:
        raise ValueError(
            f"B3 adapter SHA-256 mismatch: expected {b3_sha256}, observed {observed_adapter}"
        )
    print("PROGRESS: binding returned Stage 24 gradient evidence", flush=True)
    sources = {
        "stage24_per_group_sha256": (SHARED_GRADIENT_ROWS, execution["stage24_per_group_sha256"]),
        "stage24_summary_sha256": (SHARED_GRADIENT_SUMMARY, execution["stage24_summary_sha256"]),
        "stage24_manifest_sha256": (SHARED_GRADIENT_MANIFEST, execution["stage24_manifest_sha256"]),
        "fiber_rows_sha256": (FIBER_ROWS, execution["fiber_rows_sha256"]),
        "config_sha256": (config_path, execution["config_sha256"]),
        "package_lock_sha256": (PACKAGE_LOCK, execution["package_lock_sha256"]),
    }
    for label, (path, expected) in sources.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} mismatch: expected {expected}, observed {observed}")
    stage24_summary = read_json(SHARED_GRADIENT_SUMMARY)
    stage24_manifest = read_json(SHARED_GRADIENT_MANIFEST)
    if (
        stage24_summary.get("status") != "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED"
        or stage24_summary.get("continue_to_main_rl") is not True
        or stage24_summary.get("group_count") != execution["shared_gradient_group_count"]
        or stage24_summary.get("reward_hamming_distance") != execution["reward_hamming_distance"]
        or stage24_manifest.get("status") != "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE"
        or stage24_manifest.get("scientific_status")
        != "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED"
        or stage24_manifest.get("continue_to_main_rl") is not True
        or stage24_manifest.get("per_group_sha256") != execution["stage24_per_group_sha256"]
        or stage24_manifest.get("summary_sha256") != execution["stage24_summary_sha256"]
        or stage24_manifest.get("execution_contract_sha256")
        != execution["stage24_execution_contract_sha256"]
    ):
        raise ValueError("returned Stage 24 evidence does not authorize main RL")
    arms = build_reward_arm_configs(contract, initialization_hash=observed_adapter)
    _validate_arm_pair(arms)
    selected = _arm_config(arms, arm)
    print("PROGRESS: verifying the pinned TRL first-line GRPO interface", flush=True)
    if backend_validator is None:
        from .training_backend import validate_training_backend_api

        backend_validator = validate_training_backend_api
    backend = dict(backend_validator())
    if backend.get("reference_adapter_copy") is not True:
        raise RuntimeError("TRL backend does not preserve B3 as the frozen KL reference")
    return {
        "schema_version": 2,
        "status": "STUDY_C2_TRAINING_PREFLIGHT_OK",
        "arm": arm,
        "arm_config": selected,
        "arm_config_sha256": canonical_sha256(selected),
        "reward_only_pair_verified": True,
        "training_prompt_count": len(training_rows),
        "matched_pair_count": len(training_rows) // 2,
        "group_size": int(execution["selected_k"]),
        "expected_optimizer_steps": expected_optimizer_steps(
            training_rows, group_size=int(execution["selected_k"])
        ),
        "checkpoint_steps": list(_CHECKPOINT_STEPS),
        "b3_adapter_sha256": observed_adapter,
        "model_snapshot_sha256": execution["model_snapshot_sha256"],
        "execution_contract_sha256": sha256_file(execution_contract_path),
        **{label: str(expected) for label, (_path, expected) in sources.items()},
        "backend": backend,
        "action_protocol": "anchored_first_line_world_v1",
        "stopping_rule": "newline_or_eos_with_max_16_tokens",
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def run_training_arm(**kwargs: object) -> dict[str, object]:
    """Load the artifact-producing runtime lazily to avoid a circular import."""

    from .training_execution import run_training_arm as execute

    return execute(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "MODEL_SNAPSHOT_SHA256",
    "TRAINING_ACK",
    "TrainingProgressCallback",
    "build_grpo_config_kwargs",
    "build_traced_reward",
    "build_training_group_diagnostics",
    "expected_optimizer_steps",
    "preflight_training_arm",
    "run_training_arm",
    "select_training_rows",
    "truncate_first_line_token_ids",
    "validate_stage25_execution_contract",
]
