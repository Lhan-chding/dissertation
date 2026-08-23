"""Study C3-C checkpoint-grid evaluation with free and constrained decoding."""

from __future__ import annotations

from collections.abc import Mapping

from compensability_v5.qwen.study_b_runtime import tree_sha256

from .config_runtime import b3_defaults, load_config, require_gpu_runtime, select_evaluation_rows
from .evaluation_runtime import evaluate_grid, summarize_evaluation_rows
from .factorial_design import ARM_IDS, CHECKPOINT_STEPS
from .io import read_json, read_jsonl, sha256_file, write_json_new, write_jsonl_new
from .paths import (
    C2_FIBER_ROWS,
    CONFIG,
    EXECUTION_CONTRACT,
    FACTORIAL_EVAL_MANIFEST,
    FACTORIAL_EVAL_MASKS,
    FACTORIAL_EVAL_RAW,
    FACTORIAL_EVAL_SCENES,
    FACTORIAL_EVAL_SUMMARY,
    TRAINING_MANIFEST,
    TRAINING_ROOT,
)
from .qwen_backend import adapter_artifact_sha256, checkpoint_sampler_factory


def _checkpoint_descriptors(
    pair_manifest: Mapping[str, object], contract: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    arms = pair_manifest.get("arms")
    contract_arms = contract.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(contract_arms, Mapping):
        raise ValueError("Study C3 checkpoint provenance lacks arm records")
    b3_adapter, expected_b3 = b3_defaults()
    observed_b3 = tree_sha256(b3_adapter)
    if observed_b3 != expected_b3 or observed_b3 != contract.get("b3_adapter_sha256"):
        raise ValueError("Study C3 evaluation B3 initialization drifted")
    result: list[dict[str, object]] = [
        {
            "checkpoint_id": "B3-initialization",
            "checkpoint_step": 0,
            "verifier": None,
            "validity_channel": None,
            "adapter_path": str(b3_adapter),
            "adapter_sha256": observed_b3,
        }
    ]
    for arm in ARM_IDS:
        details = arms.get(arm)
        frozen = contract_arms.get(arm)
        if not isinstance(details, Mapping) or not isinstance(frozen, Mapping):
            raise ValueError(f"Study C3 checkpoint provenance is malformed: {arm}")
        config = frozen.get("arm_config")
        adapter_hashes = details.get("checkpoint_adapter_sha256s")
        tree_hashes = details.get("checkpoint_tree_sha256s")
        if (
            not isinstance(config, Mapping)
            or not isinstance(adapter_hashes, Mapping)
            or not isinstance(tree_hashes, Mapping)
        ):
            raise ValueError(f"Study C3 checkpoint hashes are unavailable: {arm}")
        for step in CHECKPOINT_STEPS:
            adapter = TRAINING_ROOT / arm / f"checkpoint-{step}"
            observed_tree = tree_sha256(adapter)
            observed_adapter = adapter_artifact_sha256(adapter)
            if observed_tree != tree_hashes.get(
                str(step)
            ) or observed_adapter != adapter_hashes.get(str(step)):
                raise ValueError(f"Study C3 checkpoint tree hash drifted: {arm} step {step}")
            result.append(
                {
                    "checkpoint_id": f"{arm}-step{step}",
                    "checkpoint_step": step,
                    "verifier": config["verifier"],
                    "validity_channel": config["validity_channel"],
                    "adapter_path": str(adapter),
                    "adapter_sha256": observed_adapter,
                    "checkpoint_tree_sha256": observed_tree,
                    "arm": arm,
                }
            )
    return tuple(result)


def preflight_factorial_evaluation() -> dict[str, object]:
    runtime = require_gpu_runtime()
    config = load_config(CONFIG)
    contract = read_json(EXECUTION_CONTRACT)
    pair = read_json(TRAINING_MANIFEST)
    if (
        contract.get("status") != "STUDY_C3_FACTORIAL_EXECUTION_CONTRACT_FROZEN"
        or pair.get("status") != "STUDY_C3_FACTORIAL_TRAINING_COMPLETE"
        or pair.get("execution_contract_sha256") != sha256_file(EXECUTION_CONTRACT)
        or pair.get("reward_only_factorial_verified") is not True
    ):
        raise ValueError("Study C3 factorial evaluation prerequisites drifted")
    rows = select_evaluation_rows(read_jsonl(C2_FIBER_ROWS))
    checkpoints = _checkpoint_descriptors(pair, contract)
    evaluation = config["evaluation"]
    assert isinstance(evaluation, Mapping)
    return {
        "schema_version": 3,
        "status": "STUDY_C3_FACTORIAL_EVALUATION_PREFLIGHT_OK",
        "evaluation_scene_count": len(rows),
        "evaluation_pair_count": len(rows) // 2,
        "sampled_rollouts": evaluation["sampled_rollouts"],
        "decoders": evaluation["decoders"],
        "rollout_seed_base": 2026082501,
        "rollout_seed_algorithm": "phase5_rollout_seed_v1",
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
        "training_pair_manifest_sha256": sha256_file(TRAINING_MANIFEST),
        "model_snapshot_sha256": runtime["model_snapshot_sha256"],
        "package_lock_sha256": runtime["package_lock_sha256"],
        "single_seed_mechanism_pilot": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def run_factorial_evaluation() -> dict[str, object]:
    preflight = preflight_factorial_evaluation()
    rows = select_evaluation_rows(read_jsonl(C2_FIBER_ROWS))
    raw, masks = evaluate_grid(
        rows=rows,
        checkpoints=preflight["checkpoints"],  # type: ignore[arg-type]
        decoders=preflight["decoders"],  # type: ignore[arg-type]
        rollout_count=int(preflight["sampled_rollouts"]),
        seed_base=int(preflight["rollout_seed_base"]),
        sampler_factory=checkpoint_sampler_factory,
    )
    per_scene, summary = summarize_evaluation_rows(
        raw, expected_rollouts=int(preflight["sampled_rollouts"])
    )
    summary = {
        **summary,
        "status": "STUDY_C3_FACTORIAL_CHECKPOINT_EVALUATION_COMPLETE",
        "checkpoint_count": preflight["checkpoint_count"],
        "evaluation_scene_count": preflight["evaluation_scene_count"],
        "sampled_rollouts": preflight["sampled_rollouts"],
        "single_seed_mechanism_pilot": True,
    }
    write_jsonl_new(FACTORIAL_EVAL_RAW, raw)
    write_jsonl_new(FACTORIAL_EVAL_MASKS, masks)
    write_jsonl_new(FACTORIAL_EVAL_SCENES, per_scene)
    write_json_new(FACTORIAL_EVAL_SUMMARY, summary)
    manifest = {
        **preflight,
        "status": "STUDY_C3_FACTORIAL_CHECKPOINT_EVALUATION_COMPLETE",
        "raw_rows_sha256": sha256_file(FACTORIAL_EVAL_RAW),
        "logit_masks_sha256": sha256_file(FACTORIAL_EVAL_MASKS),
        "per_scene_sha256": sha256_file(FACTORIAL_EVAL_SCENES),
        "summary_sha256": sha256_file(FACTORIAL_EVAL_SUMMARY),
        "raw_row_count": len(raw),
        "mask_row_count": len(masks),
        "scene_cell_count": len(per_scene),
        "gpu_invoked": True,
    }
    write_json_new(FACTORIAL_EVAL_MANIFEST, manifest)
    return manifest


__all__ = [
    "preflight_factorial_evaluation",
    "run_factorial_evaluation",
]
