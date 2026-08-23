"""Frozen-contract construction and four-arm Study C3-C GRPO execution."""

from __future__ import annotations

import gc
import math
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from compensability_v4.qwen.model_loader import MODEL_SNAPSHOT_SHA256
from compensability_v5.qwen.study_b_runtime import tree_sha256
from compensability_v5.study_c2.training_backend import (
    create_training_trainer,
    validate_training_backend_api,
)
from compensability_v5.study_c2.training_runtime import (
    TrainingProgressCallback,
    select_training_rows,
)

from .config_runtime import b3_defaults, load_config, require_gpu_runtime, require_offline_snapshot
from .execution_contract import build_execution_contract
from .factorial_design import ARM_IDS, CHECKPOINT_STEPS, build_factorial_arms
from .io import (
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    write_json_new,
    write_jsonl_new,
)
from .paths import (
    ACTION_AUDIT_MANIFEST,
    C2_FIBER_ROWS,
    C2_TRAINING_MANIFEST,
    CONFIG,
    EXECUTION_CONTRACT,
    EXISTING_EVAL_MANIFEST,
    PACKAGE_LOCK,
    TRAINING_MANIFEST,
    TRAINING_ROOT,
)
from .qwen_backend import adapter_artifact_sha256, load_token_counter
from .training_runtime import build_group_diagnostics, build_traced_reward

TRAINING_ACK = "I_UNDERSTAND_THIS_RUNS_A_SINGLE_SEED_MECHANISM_PILOT"


class TrainerLike(Protocol):
    state: object

    def train(self, *, resume_from_checkpoint: str | None = None) -> object: ...

    def save_model(self, output_dir: str) -> object: ...


TrainerFactory = Callable[..., TrainerLike]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Study C3 git commit SHA is malformed")
    return value


def _validate_complete_stage(path: Path, status: str) -> dict[str, object]:
    payload = read_json(path)
    if payload.get("status") != status:
        raise ValueError(f"Study C3 prerequisite is incomplete: {path}")
    return payload


def preflight_freeze_contract() -> dict[str, object]:
    snapshot = require_offline_snapshot()
    config = load_config(CONFIG)
    action = _validate_complete_stage(
        ACTION_AUDIT_MANIFEST, "STUDY_C3_ACTION_CHANNEL_AUDIT_COMPLETE"
    )
    decoder = _validate_complete_stage(
        EXISTING_EVAL_MANIFEST,
        "STUDY_C3_EXISTING_CHECKPOINT_DECODER_INTERVENTION_COMPLETE",
    )
    c2_pair = _validate_complete_stage(C2_TRAINING_MANIFEST, "STUDY_C2_TWO_ARM_TRAINING_COMPLETE")
    training_rows = select_training_rows(read_jsonl(C2_FIBER_ROWS))
    b3_adapter, expected_b3 = b3_defaults()
    observed_b3 = tree_sha256(b3_adapter)
    if observed_b3 != expected_b3:
        raise ValueError("Study C3 B3 initialization tree hash drifted")
    sources = {
        "package_lock_sha256": sha256_file(PACKAGE_LOCK),
        "fiber_rows_sha256": sha256_file(C2_FIBER_ROWS),
        "c2_training_pair_manifest_sha256": sha256_file(C2_TRAINING_MANIFEST),
        "action_audit_manifest_sha256": sha256_file(ACTION_AUDIT_MANIFEST),
        "decoder_intervention_manifest_sha256": sha256_file(EXISTING_EVAL_MANIFEST),
    }
    if action.get("fiber_rows_sha256") != sources["fiber_rows_sha256"]:
        raise ValueError("Study C3 action audit fiber provenance drifted")
    if decoder.get("package_lock_sha256") != sources["package_lock_sha256"]:
        raise ValueError("Study C3 decoder-intervention package lock drifted")
    if c2_pair.get("b3_adapter_sha256") not in {None, observed_b3}:
        raise ValueError("Study C2 pair manifest B3 provenance drifted")
    contract = build_execution_contract(
        config=config,
        training_rows=training_rows,
        b3_adapter_sha256=observed_b3,
        model_snapshot_sha256=str(snapshot["model_snapshot_sha256"]),
        source_sha256s=sources,
        git_commit_sha=_git_commit(),
    )
    return {
        **contract,
        "execution_contract_path": str(EXECUTION_CONTRACT),
        "execution_contract_exists": EXECUTION_CONTRACT.is_file(),
    }


def run_freeze_contract() -> dict[str, object]:
    preflight = preflight_freeze_contract()
    contract = {
        key: value
        for key, value in preflight.items()
        if key not in {"execution_contract_path", "execution_contract_exists"}
    }
    write_json_new(EXECUTION_CONTRACT, contract)
    return {
        **contract,
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
    }


def _validate_contract() -> tuple[dict[str, object], tuple[dict[str, object], ...], Path, str]:
    runtime = require_gpu_runtime()
    config = load_config(CONFIG)
    contract = read_json(EXECUTION_CONTRACT)
    if (
        contract.get("status") != "STUDY_C3_FACTORIAL_EXECUTION_CONTRACT_FROZEN"
        or contract.get("config_sha256") != canonical_sha256(config)
        or contract.get("model_snapshot_sha256") != MODEL_SNAPSHOT_SHA256
        or contract.get("package_lock_sha256") != sha256_file(PACKAGE_LOCK)
        or contract.get("fiber_rows_sha256") != sha256_file(C2_FIBER_ROWS)
        or contract.get("c2_training_pair_manifest_sha256") != sha256_file(C2_TRAINING_MANIFEST)
        or contract.get("action_audit_manifest_sha256") != sha256_file(ACTION_AUDIT_MANIFEST)
        or contract.get("decoder_intervention_manifest_sha256")
        != sha256_file(EXISTING_EVAL_MANIFEST)
        or contract.get("git_commit_sha") != _git_commit()
    ):
        raise ValueError("Study C3 factorial execution contract drifted")
    rows = select_training_rows(read_jsonl(C2_FIBER_ROWS))
    b3_adapter, expected_b3 = b3_defaults()
    observed_b3 = tree_sha256(b3_adapter)
    if observed_b3 != expected_b3 or observed_b3 != contract.get("b3_adapter_sha256"):
        raise ValueError("Study C3 B3 initialization drifted")
    if runtime["model_snapshot_sha256"] != contract["model_snapshot_sha256"]:
        raise ValueError("Study C3 model snapshot drifted")
    return contract, rows, b3_adapter, observed_b3


def preflight_training_arm(arm: str) -> dict[str, object]:
    if arm not in ARM_IDS:
        raise ValueError(f"unregistered Study C3 training arm: {arm}")
    contract, rows, b3_adapter, b3_sha = _validate_contract()
    backend = validate_training_backend_api()
    arms = contract.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise ValueError("Study C3 execution contract lacks the requested arm")
    arm_config = arms[arm]["arm_config"]  # type: ignore[index]
    expected = {
        value["arm"]: value
        for value in build_factorial_arms(load_config(CONFIG), initialization_hash=b3_sha)
    }[arm]
    if arm_config != expected:
        raise ValueError(f"Study C3 frozen arm config drifted: {arm}")
    return {
        "schema_version": 3,
        "status": "STUDY_C3_FACTORIAL_ARM_PREFLIGHT_OK",
        "arm": arm,
        "arm_config": arm_config,
        "training_prompt_count": len(rows),
        "expected_optimizer_steps": 192,
        "group_size": 8,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
        "b3_adapter_path": str(b3_adapter),
        "b3_adapter_sha256": b3_sha,
        "model_snapshot_sha256": contract["model_snapshot_sha256"],
        "package_lock_sha256": contract["package_lock_sha256"],
        "fiber_rows_sha256": contract["fiber_rows_sha256"],
        "git_commit_sha": contract["git_commit_sha"],
        "backend": backend,
        "single_seed_mechanism_pilot": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def _checkpoint_hashes(output_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    trees: dict[str, str] = {}
    adapters: dict[str, str] = {}
    for step in CHECKPOINT_STEPS:
        checkpoint = output_dir / f"checkpoint-{step}"
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise RuntimeError(f"Study C3 trainer lacks checkpoint-{step}")
        trees[str(step)] = tree_sha256(checkpoint)
        adapters[str(step)] = adapter_artifact_sha256(checkpoint)
    return trees, adapters


def _pair_manifest_if_complete(contract: Mapping[str, object]) -> dict[str, object] | None:
    paths = {arm: TRAINING_ROOT / arm / "manifest.json" for arm in ARM_IDS}
    if not all(path.is_file() and not path.is_symlink() for path in paths.values()):
        return None
    manifests = {arm: read_json(path) for arm, path in paths.items()}
    if any(
        payload.get("status") != "STUDY_C3_FACTORIAL_ARM_TRAINING_COMPLETE"
        or payload.get("arm") != arm
        or payload.get("single_seed_mechanism_pilot") is not True
        or payload.get("execution_contract_sha256") != sha256_file(EXECUTION_CONTRACT)
        for arm, payload in manifests.items()
    ):
        raise ValueError("Study C3 four-arm training manifests are incomplete or mismatched")
    result = {
        "schema_version": 3,
        "status": "STUDY_C3_FACTORIAL_TRAINING_COMPLETE",
        "single_seed_mechanism_pilot": True,
        "reward_only_factorial_verified": contract["reward_only_factorial_verified"],
        "training_prompt_count_per_arm": 192,
        "optimizer_steps_per_arm": 192,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "execution_contract_sha256": sha256_file(EXECUTION_CONTRACT),
        "arms": {
            arm: {
                "manifest_sha256": sha256_file(paths[arm]),
                "final_adapter_sha256": manifests[arm]["final_adapter_sha256"],
                "final_adapter_artifact_sha256": manifests[arm]["final_adapter_artifact_sha256"],
                "raw_reward_trace_sha256": manifests[arm]["raw_reward_trace_sha256"],
                "group_diagnostics_sha256": manifests[arm]["group_diagnostics_sha256"],
                "checkpoint_adapter_sha256s": manifests[arm]["checkpoint_adapter_sha256s"],
                "checkpoint_tree_sha256s": manifests[arm]["checkpoint_tree_sha256s"],
            }
            for arm in ARM_IDS
        },
        "training_invoked": True,
        "optimizer_step_invoked": True,
        "rl_invoked": True,
        "gpu_invoked": True,
    }
    if TRAINING_MANIFEST.exists():
        if read_json(TRAINING_MANIFEST) != result:
            raise ValueError("existing Study C3 training pair manifest drifted")
    else:
        write_json_new(TRAINING_MANIFEST, result)
    return result


def run_training_arm(
    *,
    arm: str,
    acknowledgement: str,
    trainer_factory: TrainerFactory = create_training_trainer,
) -> dict[str, object]:
    if acknowledgement != TRAINING_ACK:
        raise PermissionError("exact Study C3 single-seed training acknowledgement is required")
    preflight = preflight_training_arm(arm)
    arm_config = preflight["arm_config"]
    if not isinstance(arm_config, Mapping):
        raise RuntimeError("Study C3 training preflight lacks an arm config")
    output_dir = Path(str(arm_config["output_directory"]))
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Study C3 arm overwrite is forbidden: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json_new(output_dir / "arm_config.json", arm_config)
    rows = select_training_rows(read_jsonl(C2_FIBER_ROWS))
    trace_path = output_dir / "raw_reward_trace.jsonl"
    reward = build_traced_reward(
        arm_config=arm_config,
        training_rows=rows,
        trace_path=trace_path,
        group_size=8,
        token_counter=load_token_counter(),
    )
    callback = TrainingProgressCallback(
        arm, total_steps=192, trace_path=trace_path, output_dir=output_dir
    )
    dataset = tuple(
        {
            "prompt": [{"role": "user", "content": str(row["prompt"])}],
            "scene_id": row["scene_id"],
        }
        for row in rows
    )
    try:
        trainer = trainer_factory(
            arm_config=arm_config,
            b3_adapter=Path(str(preflight["b3_adapter_path"])),
            dataset=dataset,
            reward_function=reward,
            output_dir=output_dir,
            group_size=8,
            callbacks=(callback,),
        )
        trainer.train(resume_from_checkpoint=None)
        observed_steps = getattr(getattr(trainer, "state", None), "global_step", None)
        if observed_steps != 192:
            raise RuntimeError(
                f"Study C3 trainer executed {observed_steps} optimizer steps, expected 192"
            )
        logs = getattr(getattr(trainer, "state", None), "log_history", None)
        if not isinstance(logs, list) or any(
            any(isinstance(value, float) and not math.isfinite(value) for value in row.values())
            for row in logs
            if isinstance(row, Mapping)
        ):
            raise RuntimeError("Study C3 trainer log history is missing or non-finite")
        write_json_new(output_dir / "trainer_log_history.json", {"rows": logs})
        checkpoint_tree_hashes, checkpoint_adapter_hashes = _checkpoint_hashes(output_dir)
        final_adapter = output_dir / "final_adapter"
        if final_adapter.exists() or final_adapter.is_symlink():
            raise RuntimeError("Study C3 final adapter exists before final save")
        trainer.save_model(str(final_adapter))
        final_adapter_sha = tree_sha256(final_adapter)
        final_adapter_artifact_sha = adapter_artifact_sha256(final_adapter)
    finally:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    diagnostics = build_group_diagnostics(trace_path, expected_group_count=192, group_size=8)
    diagnostics_path = output_dir / "group_diagnostics.jsonl"
    write_jsonl_new(diagnostics_path, diagnostics)
    trace = read_jsonl(trace_path)
    counts = Counter(str(row["action_class"]) for row in trace)
    all_zero_count = sum(row["all_zero"] is True for row in diagnostics)
    all_one_count = sum(row["all_one"] is True for row in diagnostics)
    summary = {
        "schema_version": 3,
        "status": "STUDY_C3_FACTORIAL_ARM_TRAINING_SUMMARIZED",
        "arm": arm,
        "single_seed_mechanism_pilot": True,
        "training_seed": 2026082501,
        "training_prompt_count": 192,
        "group_size": 8,
        "rollout_count": len(trace),
        "optimizer_steps": 192,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "counts": {kind: counts[kind] for kind in ("X", "S", "W", "I")},
        "validity_bearing_group_count": sum(
            row["validity_bearing_group"] is True for row in diagnostics
        ),
        "truth_bearing_group_count": sum(row["truth_bearing_group"] is True for row in diagnostics),
        "answer_disagreement_group_count": sum(
            row["answer_disagreement_group"] is True for row in diagnostics
        ),
        "all_zero_group_count": all_zero_count,
        "all_zero_group_rate": all_zero_count / len(diagnostics),
        "all_one_group_count": all_one_count,
        "all_one_group_rate": all_one_count / len(diagnostics),
    }
    summary_path = output_dir / "summary.json"
    write_json_new(summary_path, summary)
    manifest = {
        **{key: value for key, value in preflight.items() if key != "arm_config"},
        "status": "STUDY_C3_FACTORIAL_ARM_TRAINING_COMPLETE",
        "arm_config_sha256": sha256_file(output_dir / "arm_config.json"),
        "raw_reward_trace_sha256": sha256_file(trace_path),
        "group_diagnostics_sha256": sha256_file(diagnostics_path),
        "trainer_log_history_sha256": sha256_file(output_dir / "trainer_log_history.json"),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_adapter_sha256s": checkpoint_adapter_hashes,
        "checkpoint_tree_sha256s": checkpoint_tree_hashes,
        "final_adapter_sha256": final_adapter_sha,
        "final_adapter_artifact_sha256": final_adapter_artifact_sha,
        "training_invoked": True,
        "optimizer_step_invoked": True,
        "rl_invoked": True,
        "gpu_invoked": True,
    }
    write_json_new(output_dir / "manifest.json", manifest)
    contract = read_json(EXECUTION_CONTRACT)
    pair = _pair_manifest_if_complete(contract)
    return {
        **manifest,
        "factorial_complete": pair is not None,
        "factorial_manifest_sha256": None if pair is None else sha256_file(TRAINING_MANIFEST),
    }


__all__ = [
    "TRAINING_ACK",
    "preflight_freeze_contract",
    "preflight_training_arm",
    "run_freeze_contract",
    "run_training_arm",
]
