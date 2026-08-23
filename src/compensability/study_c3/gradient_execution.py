"""Same-rollout-buffer gradient audit for semantic and validity rewards."""

from __future__ import annotations

import gc
import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.phase5_runtime import phase5_rollout_seed
from compensability_v5.qwen.study_b_runtime import tree_sha256
from compensability_v5.study_c2.shared_gradient_runtime import _group_log_probabilities

from .action_taxonomy import classify_action
from .config_runtime import b3_defaults, load_config, require_gpu_runtime, select_evaluation_rows
from .gradient_audit import (
    autograd_reward_gradient_diagnostics,
    validate_shared_action_batch,
)
from .io import read_json, read_jsonl, sha256_file, write_json_new, write_jsonl_new
from .paths import (
    C2_FIBER_ROWS,
    CONFIG,
    EXECUTION_CONTRACT,
    FACTORIAL_EVAL_MANIFEST,
    GRADIENT_BUFFER,
    GRADIENT_MANIFEST,
    GRADIENT_ROWS,
    GRADIENT_SUMMARY,
    TRAINING_MANIFEST,
    TRAINING_ROOT,
)
from .qwen_backend import (
    QwenCheckpointSampler,
    adapter_artifact_sha256,
    load_trainable_checkpoint,
)
from .reward_channels import rewards_for_class
from .semantic_action_parser import parse_p1


def _final_checkpoints(
    training: Mapping[str, object], contract: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    b3, expected_b3 = b3_defaults()
    observed_b3 = tree_sha256(b3)
    if observed_b3 != expected_b3 or observed_b3 != contract.get("b3_adapter_sha256"):
        raise ValueError("Study C3 gradient B3 initialization drifted")
    result: list[dict[str, object]] = [
        {
            "checkpoint_id": "B3-initialization",
            "adapter_path": str(b3),
            "adapter_sha256": observed_b3,
        }
    ]
    arms = training.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("Study C3 gradient audit lacks training arm provenance")
    for arm in ("A_BIN", "X_BIN", "A_LEX", "X_LEX"):
        details = arms.get(arm)
        if not isinstance(details, Mapping):
            raise ValueError(f"Study C3 gradient arm provenance is missing: {arm}")
        adapter = TRAINING_ROOT / arm / "final_adapter"
        observed = tree_sha256(adapter)
        observed_artifact = adapter_artifact_sha256(adapter)
        if observed != details.get("final_adapter_sha256") or observed_artifact != details.get(
            "final_adapter_artifact_sha256"
        ):
            raise ValueError(f"Study C3 final adapter drifted: {arm}")
        result.append(
            {
                "checkpoint_id": f"{arm}-final",
                "adapter_path": str(adapter),
                "adapter_sha256": observed_artifact,
                "adapter_tree_sha256": observed,
                "arm": arm,
            }
        )
    return tuple(result)


def preflight_shared_gradient_audit() -> dict[str, object]:
    runtime = require_gpu_runtime()
    config = load_config(CONFIG)
    contract = read_json(EXECUTION_CONTRACT)
    training = read_json(TRAINING_MANIFEST)
    evaluation = read_json(FACTORIAL_EVAL_MANIFEST)
    if (
        contract.get("status") != "STUDY_C3_FACTORIAL_EXECUTION_CONTRACT_FROZEN"
        or training.get("status") != "STUDY_C3_FACTORIAL_TRAINING_COMPLETE"
        or evaluation.get("status") != "STUDY_C3_FACTORIAL_CHECKPOINT_EVALUATION_COMPLETE"
        or training.get("execution_contract_sha256") != sha256_file(EXECUTION_CONTRACT)
        or evaluation.get("training_pair_manifest_sha256") != sha256_file(TRAINING_MANIFEST)
    ):
        raise ValueError("Study C3 shared-gradient prerequisites drifted")
    rows = select_evaluation_rows(read_jsonl(C2_FIBER_ROWS))
    gradient = config["shared_gradient"]
    assert isinstance(gradient, Mapping)
    checkpoints = _final_checkpoints(training, contract)
    return {
        "schema_version": 3,
        "status": "STUDY_C3_SHARED_GRADIENT_VALIDITY_PREFLIGHT_OK",
        "evaluation_scene_count": len(rows),
        "group_size": gradient["sampled_rollouts"],
        "decoder": gradient["decoder"],
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "rollout_seed_base": 2026082501,
        "rollout_seed_algorithm": "phase5_rollout_seed_v1",
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
        "training_pair_manifest_sha256": sha256_file(TRAINING_MANIFEST),
        "factorial_evaluation_manifest_sha256": sha256_file(FACTORIAL_EVAL_MANIFEST),
        "model_snapshot_sha256": runtime["model_snapshot_sha256"],
        "package_lock_sha256": runtime["package_lock_sha256"],
        "single_seed_mechanism_pilot": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def _buffer_group(
    *,
    checkpoint: Mapping[str, object],
    row: Mapping[str, object],
    sampler: QwenCheckpointSampler,
    group_size: int,
    seed_base: int,
) -> tuple[dict[str, object], ...]:
    scene_id = str(row["scene_id"])
    seeds = tuple(
        phase5_rollout_seed(seed_base, scene_id, rollout_index)
        for rollout_index in range(group_size)
    )
    generated = sampler(row, seeds, "free_48")
    truth = row.get("truth")
    operation = row.get("operation")
    if not isinstance(truth, list) or len(truth) != 4 or not isinstance(operation, Mapping):
        raise ValueError("Study C3 gradient scene truth/operation is malformed")
    result: list[dict[str, object]] = []
    for index, sample in enumerate(generated):
        completion = sample.get("completion")
        token_ids = sample.get("token_ids")
        if not isinstance(completion, str) or not isinstance(token_ids, list) or not token_ids:
            raise ValueError("Study C3 gradient sampler output is malformed")
        label = classify_action(
            parse_p1(completion),
            truth=tuple(int(value) for value in truth),  # type: ignore[arg-type]
            operation=operation,
        )
        a = rewards_for_class(label.action_class, arm="A_BIN")
        x = rewards_for_class(label.action_class, arm="X_BIN")
        alex = rewards_for_class(label.action_class, arm="A_LEX")
        xlex = rewards_for_class(label.action_class, arm="X_LEX")
        action_id = hashlib.sha256(
            (
                f"{checkpoint['checkpoint_id']}:{scene_id}:{index}:"
                f"{completion}:{','.join(str(token) for token in token_ids)}"
            ).encode()
        ).hexdigest()
        result.append(
            {
                "schema_version": 3,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "adapter_sha256": checkpoint["adapter_sha256"],
                "scene_id": scene_id,
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "family": row["family"],
                "rollout_index": index,
                "seed": seeds[index],
                "action_id": action_id,
                "completion": completion,
                "token_ids": token_ids,
                "action_class": label.action_class.value,
                "R_A": a["semantic_reward"],
                "R_X": x["semantic_reward"],
                "R_V": a["validity_reward"],
                "R_A_LEX": alex["combined_reward"],
                "R_X_LEX": xlex["combined_reward"],
            }
        )
    validate_shared_action_batch(result, expected_action_ids=[row["action_id"] for row in result])
    return tuple(result)


def _summarize_gradient_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Study C3 gradient summary cannot be empty")
    numeric = (
        "cosine_X_V",
        "cosine_A_V",
        "difference_X_A",
        "difference_X_LEX_X",
        "difference_A_LEX_A",
    )
    by_checkpoint: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_checkpoint[str(row["checkpoint_id"])].append(row)

    def aggregate(group: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "group_count": len(group),
            **{
                f"{field}_mean": sum(float(row[field]) for row in group) / len(group)
                for field in numeric
            },
        }

    return {
        "schema_version": 3,
        "status": "STUDY_C3_SHARED_GRADIENT_VALIDITY_AUDIT_COMPLETE",
        **aggregate(rows),
        "by_checkpoint": {
            checkpoint: aggregate(group) for checkpoint, group in sorted(by_checkpoint.items())
        },
        "same_action_batch_for_all_rewards": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": True,
    }


def run_shared_gradient_audit() -> dict[str, object]:
    preflight = preflight_shared_gradient_audit()
    rows = select_evaluation_rows(read_jsonl(C2_FIBER_ROWS))
    group_size = int(preflight["group_size"])
    buffer_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    for checkpoint in preflight["checkpoints"]:  # type: ignore[assignment]
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Study C3 gradient checkpoint descriptor is malformed")
        adapter_path = Path(str(checkpoint["adapter_path"]))
        sampler = QwenCheckpointSampler(adapter_path=adapter_path)
        groups: list[tuple[Mapping[str, object], tuple[dict[str, object], ...]]] = []
        try:
            for row in rows:
                group = _buffer_group(
                    checkpoint=checkpoint,
                    row=row,
                    sampler=sampler,
                    group_size=group_size,
                    seed_base=int(preflight["rollout_seed_base"]),
                )
                buffer_rows.extend(group)
                groups.append((row, group))
        finally:
            sampler.close()
        model, processor, parameters = load_trainable_checkpoint(adapter_path)
        try:
            for group_index, (source, group) in enumerate(groups):
                log_probabilities = _group_log_probabilities(
                    model,
                    processor,
                    prompt=str(source["prompt"]),
                    completion_token_ids=[row["token_ids"] for row in group],  # type: ignore[list-item]
                    max_prompt_length=512,
                    max_completion_length=48,
                )
                reward_vectors = {
                    "A": [float(row["R_A"]) for row in group],
                    "X": [float(row["R_X"]) for row in group],
                    "V": [float(row["R_V"]) for row in group],
                    "A_LEX": [float(row["R_A_LEX"]) for row in group],
                    "X_LEX": [float(row["R_X_LEX"]) for row in group],
                }
                measured = autograd_reward_gradient_diagnostics(
                    log_probabilities=log_probabilities,
                    trainable_parameters=parameters,
                    reward_vectors=reward_vectors,
                )
                gradient_rows.append(
                    {
                        "schema_version": 3,
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "adapter_sha256": checkpoint["adapter_sha256"],
                        "group_index": group_index,
                        "scene_id": source["scene_id"],
                        "pair_id": source["pair_id"],
                        "condition": source["condition"],
                        "family": source["family"],
                        "action_ids": [row["action_id"] for row in group],
                        "action_classes": [row["action_class"] for row in group],
                        **measured,
                    }
                )
        finally:
            model = None
            processor = None
            parameters = ()
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
    summary = _summarize_gradient_rows(gradient_rows)
    write_jsonl_new(GRADIENT_BUFFER, buffer_rows)
    write_jsonl_new(GRADIENT_ROWS, gradient_rows)
    write_json_new(GRADIENT_SUMMARY, summary)
    manifest = {
        **preflight,
        "status": "STUDY_C3_SHARED_GRADIENT_VALIDITY_AUDIT_COMPLETE",
        "shared_buffer_sha256": sha256_file(GRADIENT_BUFFER),
        "per_group_sha256": sha256_file(GRADIENT_ROWS),
        "summary_sha256": sha256_file(GRADIENT_SUMMARY),
        "shared_buffer_row_count": len(buffer_rows),
        "gradient_group_count": len(gradient_rows),
        "same_action_batch_for_all_rewards": True,
        "gpu_invoked": True,
    }
    write_json_new(GRADIENT_MANIFEST, manifest)
    return manifest


__all__ = [
    "preflight_shared_gradient_audit",
    "run_shared_gradient_audit",
]
