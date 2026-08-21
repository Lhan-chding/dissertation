"""Reward tracing and frozen evaluation helpers for Study C."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from compensability_v4.qwen.phase5_support import parse_world
from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation
from compensability_v5.qwen.study_c_metrics import (
    StudyCError,
    build_study_c_summary,
    read_study_c_trace,
)

STUDY_C_EVAL_SEED = 2026082302
STUDY_C_EVAL_ROLLOUTS = 16


class ArmLike(Protocol):
    name: str
    initialization: str
    reward_function: str
    group_size: int
    temperature: float
    top_p: float
    top_k: int
    max_completion_length: int


class SceneLike(Protocol):
    scene_id: str
    prompt: str
    truth: tuple[int, int, int, int]
    answer_label: int
    family: str
    fiber_size: int
    fiber_bin: str
    support_bin: str

    @property
    def operation(self) -> dict[str, object]: ...


EvaluationSampler = Callable[[SceneLike, tuple[int, ...]], Sequence[object]]


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StudyCError(f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def make_reward_function(
    *, scenes: Sequence[SceneLike], arm: ArmLike, trace_path: Path
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
        rollout_seeds = kwargs.get("rollout_seed")
        has_rollout_seeds = (
            isinstance(rollout_seeds, Sequence)
            and not isinstance(rollout_seeds, (str, bytes))
            and len(rollout_seeds) == len(completions)
        )
        for position, (completion, scene_id) in enumerate(zip(completions, scene_ids, strict=True)):
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
                        {"rollout_seed": rollout_seeds[position]}  # type: ignore[index]
                        if has_rollout_seeds
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


def qwen_text_evaluation_sampler(
    *, arm: ArmLike, model: object, processor: object
) -> EvaluationSampler:
    """Create the real fixed-seed text-only Qwen sampler."""

    def sample(scene: SceneLike, seeds: tuple[int, ...]) -> Sequence[object]:
        import torch

        set_eval = getattr(model, "eval", None)
        if callable(set_eval):
            set_eval()
        apply_template = getattr(processor, "apply_chat_template", None)
        if not callable(apply_template):
            raise StudyCError("Study C processor exposes no chat template")
        messages = [{"role": "user", "content": [{"type": "text", "text": scene.prompt}]}]
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
        for rollout_index, seed in enumerate(seeds, start=1):
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
            print(
                f"PROGRESS: Study C {arm.name} scene {scene.scene_id} rollout "
                f"{rollout_index}/{len(seeds)}",
                flush=True,
            )
        return tuple(outputs)

    return sample


def _run_frozen_eval(
    *,
    arm: ArmLike,
    scenes: Sequence[SceneLike],
    output_dir: Path,
    sampler: EvaluationSampler,
    trace_name: str,
    summary_name: str,
    trace_kind: str,
    measurement_scope: str,
) -> dict[str, object]:
    trace_path = output_dir / trace_name
    summary_path = output_dir / summary_name
    if trace_path.exists() or summary_path.exists():
        raise StudyCError("Study C frozen evaluation overwrite is forbidden")
    reward = make_reward_function(scenes=scenes, arm=arm, trace_path=trace_path)
    rollout_seeds = tuple(STUDY_C_EVAL_SEED + index for index in range(STUDY_C_EVAL_ROLLOUTS))
    for scene_index, scene in enumerate(scenes, start=1):
        completions = tuple(sampler(scene, rollout_seeds))
        if len(completions) != STUDY_C_EVAL_ROLLOUTS:
            raise StudyCError(
                f"frozen sampler must return {STUDY_C_EVAL_ROLLOUTS} rollouts per scene"
            )
        for start in range(0, STUDY_C_EVAL_ROLLOUTS, arm.group_size):
            stop = start + arm.group_size
            reward(
                completions[start:stop],
                scene_id=[scene.scene_id],
                rollout_seed=rollout_seeds[start:stop],
            )
        print(
            f"PROGRESS: Study C {arm.name} {measurement_scope} scene "
            f"{scene_index}/{len(scenes)} complete",
            flush=True,
        )
    rows = read_study_c_trace(trace_path)
    rewritten = tuple({**row, "trace_kind": trace_kind} for row in rows)
    temporary = trace_path.with_suffix(".rewrite.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rewritten:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(trace_path)
    summary = build_study_c_summary({arm.name: trace_path}, group_size=arm.group_size)
    summary.update(
        {
            "measurement_scope": measurement_scope,
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


def run_post_training_frozen_eval(
    *,
    arm: ArmLike,
    scenes: Sequence[SceneLike],
    output_dir: Path,
    sampler: EvaluationSampler,
) -> dict[str, object]:
    """Sample the trained policy on the frozen rl_eval split."""

    return _run_frozen_eval(
        arm=arm,
        scenes=scenes,
        output_dir=output_dir,
        sampler=sampler,
        trace_name="eval_raw_rows.jsonl",
        summary_name="eval_summary.json",
        trace_kind="post_training_frozen_eval",
        measurement_scope="post_training_frozen_eval",
    )


def run_pre_training_frozen_eval(
    *,
    arm: ArmLike,
    scenes: Sequence[SceneLike],
    output_dir: Path,
    sampler: EvaluationSampler,
) -> dict[str, object]:
    """Sample the shared initialization before any optimizer step."""

    return _run_frozen_eval(
        arm=arm,
        scenes=scenes,
        output_dir=output_dir,
        sampler=sampler,
        trace_name="pre_training_eval_raw_rows.jsonl",
        summary_name="pre_training_eval_summary.json",
        trace_kind="pre_training_frozen_eval",
        measurement_scope="pre_training_frozen_eval",
    )


__all__ = [
    "STUDY_C_EVAL_ROLLOUTS",
    "STUDY_C_EVAL_SEED",
    "EvaluationSampler",
    "make_reward_function",
    "qwen_text_evaluation_sampler",
    "run_post_training_frozen_eval",
    "run_pre_training_frozen_eval",
]
