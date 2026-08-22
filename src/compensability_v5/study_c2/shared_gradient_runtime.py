"""Foreground, resumable shared-batch verifier-gradient audit for Study C2."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.model_loader import (
    MODEL_SNAPSHOT_SHA256,
    load_pinned_qwen,
    require_server_model,
)
from compensability_v5.qwen.study_b_backend import (
    require_offline_environment,
    verify_runtime_package_lock,
)
from compensability_v5.qwen.study_b_runtime import tree_sha256

from .contrast_rank import reward_contrast
from .gradient_audit import autograd_gradient_diagnostics
from .group_metrics import summarize_group
from .io import read_json, read_jsonl, sha256_file, write_json_new
from .paths import (
    FIBER_ROWS,
    SHARED_GRADIENT_MANIFEST,
    SHARED_GRADIENT_ROOT,
    SHARED_GRADIENT_ROWS,
    SHARED_GRADIENT_SUMMARY,
    STAGE24_EXECUTION_CONTRACT,
    SUPPORT_MANIFEST,
    SUPPORT_RAW_ROWS,
    SUPPORT_SUMMARY,
)
from .stages import load_contract

PACKAGE_LOCK = Path("configs/v5/server_package_lock.yaml")
SHARED_GRADIENT_ACK = "I_UNDERSTAND_THIS_RUNS_STUDY_C2_SHARED_BATCH_GRADIENT_AUDIT"
_KINDS = ("X", "S", "F", "U")
_ZERO_TOLERANCE = 1e-12


def _require_offline_cuda() -> None:  # pragma: no cover - server runtime
    require_offline_environment()


def group_support_rows(
    rows: Sequence[Mapping[str, object]], *, group_size: int
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    """Freeze scene-local consecutive groups without crossing prompt boundaries."""

    if group_size < 2:
        raise ValueError("shared-gradient group size must be at least two")
    by_scene: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        rollout_index = row.get("rollout_index")
        kind = row.get("kind")
        if not isinstance(scene_id, str) or type(rollout_index) is not int or kind not in _KINDS:
            raise ValueError("support rows require scene_id, rollout_index, and X/S/F/U kind")
        by_scene.setdefault(scene_id, []).append(row)
    if not by_scene:
        raise ValueError("shared-gradient support rows cannot be empty")

    groups: list[tuple[Mapping[str, object], ...]] = []
    for scene_rows in by_scene.values():
        indices = [int(row["rollout_index"]) for row in scene_rows]
        if indices != list(range(len(scene_rows))):
            raise ValueError("support rollout order drifted within a scene")
        if len(scene_rows) % group_size:
            raise ValueError("support scene rollouts do not divide into closed groups")
        for start in range(0, len(scene_rows), group_size):
            selected = tuple(scene_rows[start : start + group_size])
            if len({row.get("scene_id") for row in selected}) != 1:
                raise ValueError("shared-gradient group crosses scene boundaries")
            groups.append(selected)
    return tuple(groups)


def _aggregate_gradient_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("shared-gradient aggregate cannot be empty")
    counts = Counter({kind: 0 for kind in _KINDS})
    for row in rows:
        raw_counts = row.get("counts")
        if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(_KINDS):
            raise ValueError("shared-gradient row has malformed X/S/F/U counts")
        for kind in _KINDS:
            value = raw_counts[kind]
            if type(value) is not int or value < 0:
                raise ValueError("shared-gradient counts must be non-negative integers")
            counts[kind] += value
    numeric_fields = (
        "gradient_state_norm",
        "gradient_answer_norm",
        "gradient_difference_norm",
        "gradient_cosine",
    )
    for row in rows:
        if any(
            isinstance(row.get(field), bool)
            or not isinstance(row.get(field), (int, float))
            or not math.isfinite(float(row[field]))
            for field in numeric_fields
        ):
            raise ValueError("shared-gradient metrics must be finite")
    group_count = len(rows)
    hamming = sum(int(row["reward_hamming_distance"]) for row in rows)
    return {
        "group_count": group_count,
        "counts": {kind: counts[kind] for kind in _KINDS},
        "reward_hamming_distance": hamming,
        "RDGR_group_count": sum(row.get("RDGR") is True for row in rows),
        "ESGR_group_count": sum(row.get("ESGR") is True for row in rows),
        "gradient_state_norm_mean": sum(float(row["gradient_state_norm"]) for row in rows)
        / group_count,
        "gradient_answer_norm_mean": sum(float(row["gradient_answer_norm"]) for row in rows)
        / group_count,
        "gradient_difference_norm_mean": sum(float(row["gradient_difference_norm"]) for row in rows)
        / group_count,
        "gradient_difference_norm_max": max(float(row["gradient_difference_norm"]) for row in rows),
        "gradient_cosine_mean": sum(float(row["gradient_cosine"]) for row in rows) / group_count,
    }


def summarize_shared_gradient_audit(
    rows: Sequence[Mapping[str, object]], *, group_size: int
) -> dict[str, object]:
    """Summarize the logical RL gate and all registered Study C2 strata."""

    overall = _aggregate_gradient_rows(rows)
    by_condition_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_family_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        condition = row.get("condition")
        family = row.get("family")
        if condition not in {"collision", "separating"} or not isinstance(family, str):
            raise ValueError("shared-gradient rows require registered condition and family")
        by_condition_rows[str(condition)].append(row)
        by_family_rows[family].append(row)
    null_contrast = (
        overall["reward_hamming_distance"] == 0
        and float(overall["gradient_difference_norm_max"]) <= _ZERO_TOLERANCE
    )
    return {
        "schema_version": 2,
        "status": (
            "STUDY_C2_SHARED_GRADIENT_CONTRAST_NOT_ESTIMABLE"
            if null_contrast
            else "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED"
        ),
        "continue_to_main_rl": not null_contrast,
        "group_size": group_size,
        "rollout_count": len(rows) * group_size,
        "reward_hamming_rate": int(overall["reward_hamming_distance"]) / (len(rows) * group_size),
        **overall,
        "by_condition": {
            key: _aggregate_gradient_rows(selected)
            for key, selected in sorted(by_condition_rows.items())
        },
        "by_family": {
            key: _aggregate_gradient_rows(selected)
            for key, selected in sorted(by_family_rows.items())
        },
        "zero_tolerance": _ZERO_TOLERANCE,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": True,
    }


def _validate_execution_contract(payload: Mapping[str, object]) -> None:
    expected = {
        "schema_version",
        "status",
        "support_raw_rows_sha256",
        "support_summary_sha256",
        "support_manifest_sha256",
        "fiber_rows_sha256",
        "config_sha256",
        "package_lock_sha256",
        "b3_adapter_sha256",
        "model_snapshot_sha256",
        "selected_k",
        "rollout_count",
        "support_counts",
        "support_status",
    }
    if (
        set(payload) != expected
        or payload.get("schema_version") != 2
        or payload.get("status") != "STUDY_C2_STAGE24_EXECUTION_CONTRACT_FROZEN"
        or payload.get("selected_k") != 8
        or payload.get("rollout_count") != 6144
    ):
        raise ValueError("Study C2 Stage 24 execution contract drifted")
    digest_keys = expected - {
        "schema_version",
        "status",
        "selected_k",
        "rollout_count",
        "support_counts",
        "support_status",
    }
    for key in digest_keys:
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Study C2 Stage 24 contract has invalid {key}")
    if (
        payload.get("model_snapshot_sha256") != MODEL_SNAPSHOT_SHA256
        or payload.get("support_status") != "REWARD_CONTRAST_IDENTIFIED"
        or payload.get("support_counts") != {"X": 143, "S": 635, "F": 655, "U": 4711}
    ):
        raise ValueError("Study C2 Stage 24 returned support facts drifted")


def preflight_shared_gradient(
    *,
    config_path: Path,
    execution_contract_path: Path = STAGE24_EXECUTION_CONTRACT,
    b3_adapter: Path,
    b3_sha256: str,
) -> dict[str, object]:  # pragma: no cover - real server dependencies are injected in tests
    print("PROGRESS: validating the frozen Stage 24 execution contract", flush=True)
    execution = read_json(execution_contract_path)
    _validate_execution_contract(execution)
    contract = load_contract(config_path)
    print("PROGRESS: verifying offline mode, package lock, and single bf16 4090", flush=True)
    _require_offline_cuda()
    verify_runtime_package_lock(PACKAGE_LOCK)
    print("PROGRESS: hashing the immutable Qwen snapshot and Study B B3 adapter", flush=True)
    require_server_model()
    if b3_sha256 != execution["b3_adapter_sha256"]:
        raise ValueError("operator B3 SHA-256 differs from the Stage 24 execution contract")
    observed_adapter = tree_sha256(b3_adapter)
    if observed_adapter != b3_sha256:
        raise ValueError(
            f"B3 adapter SHA-256 mismatch: expected {b3_sha256}, observed {observed_adapter}"
        )
    print("PROGRESS: binding all returned Stage 23 evidence hashes", flush=True)
    sources = {
        "support_raw_rows_sha256": (SUPPORT_RAW_ROWS, execution["support_raw_rows_sha256"]),
        "support_summary_sha256": (SUPPORT_SUMMARY, execution["support_summary_sha256"]),
        "support_manifest_sha256": (SUPPORT_MANIFEST, execution["support_manifest_sha256"]),
        "fiber_rows_sha256": (FIBER_ROWS, execution["fiber_rows_sha256"]),
        "config_sha256": (config_path, execution["config_sha256"]),
        "package_lock_sha256": (PACKAGE_LOCK, execution["package_lock_sha256"]),
    }
    for label, (path, expected) in sources.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} mismatch: expected {expected}, observed {observed}")
    support_summary = read_json(SUPPORT_SUMMARY)
    support_manifest = read_json(SUPPORT_MANIFEST)
    if (
        support_summary.get("status") != "REWARD_CONTRAST_IDENTIFIED"
        or support_summary.get("counts") != execution["support_counts"]
        or support_summary.get("rollout_count") != execution["rollout_count"]
        or not isinstance(support_summary.get("k_selection"), Mapping)
        or support_summary["k_selection"].get("selected_k") != execution["selected_k"]  # type: ignore[union-attr]
        or support_manifest.get("status") != "STUDY_C2_FROZEN_SUPPORT_COMPLETE"
        or support_manifest.get("b3_adapter_sha256") != b3_sha256
        or support_manifest.get("raw_rows_sha256") != execution["support_raw_rows_sha256"]
        or support_manifest.get("summary_sha256") != execution["support_summary_sha256"]
        or support_manifest.get("rollout_count") != execution["rollout_count"]
        or support_manifest.get("fiber_rows_sha256") != execution["fiber_rows_sha256"]
        or support_manifest.get("config_sha256") != execution["config_sha256"]
        or execution["selected_k"] not in contract["group_candidates"]  # type: ignore[operator]
    ):
        raise ValueError("returned Stage 23 support evidence failed its logical gate")
    group_size = int(execution["selected_k"])
    rollout_count = int(execution["rollout_count"])
    training = contract.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Study C2 Stage 24 training token limits are unavailable")
    return {
        "schema_version": 2,
        "status": "STUDY_C2_SHARED_GRADIENT_PREFLIGHT_OK",
        "group_size": group_size,
        "group_count": rollout_count // group_size,
        "rollout_count": rollout_count,
        "max_prompt_length": int(training["max_prompt_length"]),
        "max_completion_length": int(training["max_completion_length"]),
        "b3_adapter_sha256": observed_adapter,
        "execution_contract_sha256": sha256_file(execution_contract_path),
        **{label: str(expected) for label, (_path, expected) in sources.items()},
        "gpu_invoked": False,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
    }


def _group_log_probabilities(
    model: object,
    processor: object,
    *,
    prompt: str,
    completion_token_ids: Sequence[Sequence[int]],
    max_prompt_length: int,
    max_completion_length: int,
) -> object:  # pragma: no cover - pinned Qwen server runtime
    import torch
    import torch.nn.functional as functional

    tokenizer = getattr(processor, "tokenizer", processor)
    render = getattr(tokenizer, "apply_chat_template", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(render) or not callable(encode):
        raise RuntimeError("Study C2 processor lacks chat-template tokenization")
    rendered = render(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    prefix = tuple(int(token) for token in encode(rendered, add_special_tokens=False))
    completions = tuple(tuple(int(token) for token in values) for values in completion_token_ids)
    if not prefix or not completions or any(not values for values in completions):
        raise RuntimeError("Study C2 shared-gradient tokenization is empty")
    if len(prefix) > max_prompt_length:
        raise RuntimeError("Study C2 shared-gradient prompt exceeds its frozen token limit")
    if any(len(completion) > max_completion_length for completion in completions):
        raise RuntimeError("Study C2 shared-gradient completion exceeds its frozen token limit")
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    if type(pad) is not int:
        pad = eos
    if type(pad) is not int:
        raise RuntimeError("Study C2 tokenizer lacks an integer padding token")
    sequences = tuple(prefix + completion for completion in completions)
    maximum = max(len(sequence) for sequence in sequences)
    device = getattr(model, "device", "cuda:0")
    input_ids = torch.full((len(sequences), maximum), int(pad), dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    completion_mask = torch.zeros((len(sequences), maximum - 1), dtype=torch.bool, device=device)
    for row_index, (sequence, completion) in enumerate(zip(sequences, completions, strict=True)):
        length = len(sequence)
        input_ids[row_index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
        start = len(prefix) - 1
        completion_mask[row_index, start : start + len(completion)] = True
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("Study C2 shared-gradient forward returned invalid logits")
    shifted = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    losses = functional.cross_entropy(shifted.transpose(1, 2), targets, reduction="none")
    return -(losses * completion_mask).sum(dim=1)


def _prompt_map(fiber_rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in fiber_rows:
        scene_id = row.get("scene_id")
        prompt = row.get("prompt")
        if not isinstance(scene_id, str) or not isinstance(prompt, str) or not prompt:
            raise ValueError("fiber rows require scene_id and prompt")
        if scene_id in result:
            raise ValueError("fiber rows contain duplicate scene IDs")
        result[scene_id] = prompt
    return result


def _evaluate_group(
    model: object,
    processor: object,
    trainable_parameters: Sequence[object],
    prompt: str,
    group: Sequence[Mapping[str, object]],
    max_prompt_length: int,
    max_completion_length: int,
) -> dict[str, object]:  # pragma: no cover - pinned Qwen server runtime
    token_ids = tuple(row.get("token_ids") for row in group)
    if any(
        not isinstance(values, list)
        or not values
        or any(type(token) is not int for token in values)
        for values in token_ids
    ):
        raise ValueError("support rows contain malformed completion token IDs")
    state_rewards = tuple(float(row["state_reward"]) for row in group)
    answer_rewards = tuple(float(row["answer_reward"]) for row in group)
    log_probabilities = _group_log_probabilities(
        model,
        processor,
        prompt=prompt,
        completion_token_ids=token_ids,  # type: ignore[arg-type]
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )
    gradient = autograd_gradient_diagnostics(
        log_probabilities=log_probabilities,
        trainable_parameters=trainable_parameters,
        state_rewards=state_rewards,
        answer_rewards=answer_rewards,
    )
    contrast = reward_contrast(state_rewards, answer_rewards)
    group_metrics = summarize_group(tuple(str(row["kind"]) for row in group))
    return {
        **gradient,
        **group_metrics,
        "state_rewards": list(state_rewards),
        "answer_rewards": list(answer_rewards),
        "contrast_rank": contrast["contrast_rank"],
        "second_singular_value": contrast["second_singular_value"],
        "normalized_contrast_strength": contrast["normalized_contrast_strength"],
        "log_probability_mean": float(log_probabilities.detach().mean().item()),
    }


def _validate_partial(
    rows: Sequence[Mapping[str, object]], groups: Sequence[Sequence[Mapping[str, object]]]
) -> None:
    if len(rows) > len(groups):
        raise ValueError("partial shared-gradient trace exceeds the frozen group count")
    for index, row in enumerate(rows):
        group = groups[index]
        if (
            row.get("group_index") != index
            or row.get("scene_id") != group[0].get("scene_id")
            or row.get("rollout_indices") != [item.get("rollout_index") for item in group]
            or row.get("finite") is not True
        ):
            raise ValueError(f"partial shared-gradient trace drifted at group {index}")


GroupEvaluator = Callable[[str, Sequence[Mapping[str, object]]], Mapping[str, object]]


def run_shared_gradient_audit(
    *,
    config_path: Path,
    execution_contract_path: Path = STAGE24_EXECUTION_CONTRACT,
    b3_adapter: Path,
    b3_sha256: str,
    acknowledgement: str,
    group_evaluator: GroupEvaluator | None = None,
) -> dict[str, object]:  # pragma: no cover - real server path uses injected-test seams
    if acknowledgement != SHARED_GRADIENT_ACK:
        raise PermissionError("exact Study C2 shared-gradient acknowledgement is required")
    preflight = preflight_shared_gradient(
        config_path=config_path,
        execution_contract_path=execution_contract_path,
        b3_adapter=b3_adapter,
        b3_sha256=b3_sha256,
    )
    if (
        SHARED_GRADIENT_ROWS.exists()
        or SHARED_GRADIENT_SUMMARY.exists()
        or SHARED_GRADIENT_MANIFEST.exists()
    ):
        raise RuntimeError("completed Study C2 shared-gradient output exists; overwrite forbidden")
    support_rows = read_jsonl(SUPPORT_RAW_ROWS)
    fibers = read_jsonl(FIBER_ROWS)
    prompts = _prompt_map(fibers)
    group_size = int(preflight["group_size"])
    groups = group_support_rows(support_rows, group_size=group_size)
    if len(groups) != preflight["group_count"]:
        raise ValueError("frozen Stage 24 group count differs from preflight")
    for row in support_rows:
        scene_id = str(row["scene_id"])
        if scene_id not in prompts:
            raise ValueError(f"support scene lacks its frozen prompt: {scene_id}")

    partial = SHARED_GRADIENT_ROOT / "per_group.partial.jsonl"
    SHARED_GRADIENT_ROOT.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    if partial.exists() and partial.stat().st_size:
        completed = list(read_jsonl(partial))
    _validate_partial(completed, groups)

    model = processor = None
    trainable_parameters: tuple[object, ...] = ()
    if group_evaluator is None:
        print("PROGRESS: loading immutable Qwen snapshot and trainable B3 LoRA", flush=True)
        base, processor = load_pinned_qwen()
        from compensability_v4.training.phase4 import freeze_base_parameters

        freeze_base_parameters(base)
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(b3_adapter), is_trainable=True)
        model.eval()
        trainable_parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not trainable_parameters:
            raise RuntimeError("Study C2 shared-gradient audit found no trainable LoRA parameters")

    with partial.open("a", encoding="utf-8") as stream:
        for group_index in range(len(completed), len(groups)):
            group = groups[group_index]
            scene_id = str(group[0]["scene_id"])
            prompt = prompts[scene_id]
            if group_evaluator is None:
                assert model is not None and processor is not None
                measured = _evaluate_group(
                    model,
                    processor,
                    trainable_parameters,
                    prompt,
                    group,
                    max_prompt_length=int(preflight["max_prompt_length"]),
                    max_completion_length=int(preflight["max_completion_length"]),
                )
            else:
                measured = dict(group_evaluator(prompt, group))
            if measured.get("finite") is not True:
                raise RuntimeError(f"non-finite shared gradient at group {group_index}")
            row = {
                "schema_version": 2,
                "group_index": group_index,
                "scene_id": scene_id,
                "pair_id": group[0]["pair_id"],
                "condition": group[0]["condition"],
                "family": group[0]["family"],
                "group_size": group_size,
                "rollout_indices": [item["rollout_index"] for item in group],
                "kinds": [item["kind"] for item in group],
                **measured,
            }
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed.append(row)
            print(
                f"PROGRESS: shared gradient group {group_index + 1}/{len(groups)} "
                f"scene={scene_id} hamming={row['reward_hamming_distance']} "
                f"difference_norm={float(row['gradient_difference_norm']):.8g}",
                flush=True,
            )
    partial.replace(SHARED_GRADIENT_ROWS)
    summary = summarize_shared_gradient_audit(completed, group_size=group_size)
    write_json_new(SHARED_GRADIENT_SUMMARY, summary)
    manifest: dict[str, object] = {
        **preflight,
        "status": "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE",
        "scientific_status": summary["status"],
        "continue_to_main_rl": summary["continue_to_main_rl"],
        "per_group_sha256": sha256_file(SHARED_GRADIENT_ROWS),
        "summary_sha256": sha256_file(SHARED_GRADIENT_SUMMARY),
        "gradient_definition": "sum_group_centered_advantage_times_sequence_log_probability",
        "same_rollouts_for_both_rewards": True,
        "optimizer_step_invoked": False,
        "training_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": True,
    }
    write_json_new(SHARED_GRADIENT_MANIFEST, manifest)
    return manifest


__all__ = [
    "SHARED_GRADIENT_ACK",
    "group_support_rows",
    "preflight_shared_gradient",
    "run_shared_gradient_audit",
    "summarize_shared_gradient_audit",
]
