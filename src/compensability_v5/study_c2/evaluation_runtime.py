"""Fail-closed, frozen post-training Study C2 evaluation runtime."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.model_loader import require_server_model
from compensability_v5.qwen.study_b_backend import (
    require_offline_environment,
    verify_runtime_package_lock,
)
from compensability_v5.qwen.study_b_runtime import tree_sha256

from .io import read_json, read_jsonl, sha256_file, write_json_new, write_jsonl_new
from .paths import (
    EVALUATION_MANIFEST,
    EVALUATION_RAW_ROWS,
    EVALUATION_ROOT,
    EVALUATION_SUMMARY,
    FIBER_ROWS,
    TRAINING_PAIR_MANIFEST,
    TRAINING_ROOT,
)
from .schemas import build_reward_arm_configs
from .stages import load_contract
from .statistics import paired_collision_difference_in_differences
from .training_runtime import build_traced_reward

PACKAGE_LOCK = Path("configs/v5/server_package_lock.yaml")
EVALUATION_ACK = "I_UNDERSTAND_THIS_RUNS_STUDY_C2_POST_TRAINING_EVALUATION"
_EVAL_SPLITS = ("dev", "test", "positive_control")
_EXPECTED_EVAL_PAIRS = 88
_EXPECTED_EVAL_PROMPTS = 176
_GROUP_SIZE = 8

EvaluationSampler = Callable[[Mapping[str, object], Sequence[int]], Sequence[str]]
SamplerFactory = Callable[[Path, Path], EvaluationSampler]


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_offline_cuda() -> None:  # pragma: no cover - server-only dependency gate
    require_offline_environment()
    verify_runtime_package_lock(PACKAGE_LOCK)


def select_evaluation_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    selected = tuple(dict(row) for row in rows if row.get("split") in _EVAL_SPLITS)
    if len(selected) != _EXPECTED_EVAL_PROMPTS:
        raise ValueError(
            "Study C2 Stage 26 requires exactly 176 held-out prompts, "
            f"observed {len(selected)}"
        )
    by_pair: dict[str, list[Mapping[str, object]]] = {}
    for row in selected:
        pair_id = row.get("pair_id")
        scene_id = row.get("scene_id")
        split = row.get("split")
        condition = row.get("condition")
        if (
            not isinstance(pair_id, str)
            or not isinstance(scene_id, str)
            or split not in _EVAL_SPLITS
            or condition not in {"collision", "separating"}
            or not isinstance(row.get("prompt"), str)
            or not isinstance(row.get("truth"), list)
            or not isinstance(row.get("observation"), list)
            or not isinstance(row.get("operation"), Mapping)
        ):
            raise ValueError("Study C2 evaluation row metadata is malformed")
        by_pair.setdefault(pair_id, []).append(row)
    if len(by_pair) != _EXPECTED_EVAL_PAIRS or any(
        len(pair) != 2 or {row["condition"] for row in pair} != {"collision", "separating"}
        for pair in by_pair.values()
    ):
        raise ValueError("Study C2 evaluation rows are not 88 complete paired scenes")
    return selected


def _validate_pair_manifest(payload: Mapping[str, object]) -> None:
    expected_arms = {"C2_answer_reward", "C2_exact_state_reward"}
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "STUDY_C2_TWO_ARM_TRAINING_COMPLETE"
        or payload.get("reward_only_pair_verified") is not True
        or payload.get("training_prompt_count_per_arm") != 192
        or payload.get("optimizer_steps_per_arm") != 192
    ):
        raise ValueError("returned Stage 25 pair manifest drifted")
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != expected_arms:
        raise ValueError("returned Stage 25 pair manifest lacks both registered arms")


def validate_evaluation_backend_api() -> dict[str, object]:  # pragma: no cover - server lock
    from .training_backend import create_evaluation_sampler

    if not callable(create_evaluation_sampler):
        raise RuntimeError("Stage 26 evaluation sampler backend is unavailable")
    return {"generation_available": True}


def preflight_post_training_evaluation(
    *,
    config_path: Path,
    backend_validator: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    print("PROGRESS: validating the frozen Study C2 evaluation contract", flush=True)
    contract = load_contract(config_path)
    evaluation = contract["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise ValueError("Study C2 evaluation contract is malformed")
    rows = select_evaluation_rows(read_jsonl(FIBER_ROWS))
    print("PROGRESS: verifying offline mode, package lock, and single bf16 4090", flush=True)
    _require_offline_cuda()
    print("PROGRESS: verifying the immutable Qwen snapshot", flush=True)
    require_server_model()
    pair_manifest = read_json(TRAINING_PAIR_MANIFEST)
    _validate_pair_manifest(pair_manifest)
    sources = {
        "fiber_rows_sha256": (FIBER_ROWS, sha256_file(FIBER_ROWS)),
        "config_sha256": (config_path, sha256_file(config_path)),
        "package_lock_sha256": (PACKAGE_LOCK, sha256_file(PACKAGE_LOCK)),
        "training_pair_manifest_sha256": (TRAINING_PAIR_MANIFEST, sha256_file(TRAINING_PAIR_MANIFEST)),
    }
    initialization_hashes = {
        name: pair_manifest["arms"][name]["final_adapter_sha256"] for name in pair_manifest["arms"]
    }
    arm_configs = {
        arm["name"]: arm
        for arm in build_reward_arm_configs(
            contract,
            initialization_hash=str(initialization_hashes["C2_answer_reward"]),
        )
    }
    arm_manifests: dict[str, dict[str, object]] = {}
    for name in ("C2_answer_reward", "C2_exact_state_reward"):
        manifest_path = TRAINING_ROOT / name / "manifest.json"
        adapter_path = TRAINING_ROOT / name / "final_adapter"
        manifest = read_json(manifest_path)
        if manifest.get("status") != "STUDY_C2_ARM_TRAINING_COMPLETE":
            raise ValueError(f"incomplete arm manifest: {name}")
        adapter_sha = tree_sha256(adapter_path)
        if adapter_sha != pair_manifest["arms"][name]["final_adapter_sha256"]:
            raise ValueError(f"Stage 25 final adapter SHA-256 drifted for {name}")
        arm_manifests[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "adapter_path": str(adapter_path),
            "reward_function_id": str(arm_configs[name]["reward_function_id"]),
        }
    if backend_validator is None:
        backend_validator = validate_evaluation_backend_api
    backend = dict(backend_validator())
    return {
        "schema_version": 2,
        "status": "STUDY_C2_EVALUATION_PREFLIGHT_OK",
        "sampled_rollouts": int(evaluation["sampled_rollouts"]),
        "bootstrap_resamples": int(evaluation["bootstrap_resamples"]),
        "bootstrap_seed": int(evaluation["bootstrap_seed"]),
        "group_size": _GROUP_SIZE,
        "evaluation_scene_count": len(rows),
        "evaluation_pair_count": len(rows) // 2,
        "reward_only_pair_verified": True,
        "arm_manifests": arm_manifests,
        "backend": backend,
        **{label: digest for label, (_path, digest) in sources.items()},
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def _aggregate_scene_rows(
    raw_rows: Sequence[Mapping[str, object]], index: Mapping[str, Mapping[str, object]]
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in raw_rows:
        scene_id = row.get("scene_id")
        arm = row.get("arm")
        if not isinstance(scene_id, str) or not isinstance(arm, str):
            raise ValueError("Stage 26 raw evaluation row is malformed")
        grouped.setdefault((scene_id, arm), []).append(row)
    results: list[dict[str, object]] = []
    for (scene_id, arm), group in sorted(grouped.items()):
        source = index[scene_id]
        exact = [float(row["state_reward"]) for row in group]
        answer = [float(row["answer_reward"]) for row in group]
        parse = [bool(row["parse_success"]) for row in group]
        results.append(
            {
                "schema_version": 2,
                "scene_id": scene_id,
                "pair_id": source["pair_id"],
                "split": source["split"],
                "condition": source["condition"],
                "family": source["family"],
                "arm": arm,
                "rollout_count": len(group),
                "exact": sum(exact) / len(exact),
                "answer": sum(answer) / len(answer),
                "parse_rate": sum(parse) / len(parse),
                "counts": dict(Counter(str(row["kind"]) for row in group)),
            }
        )
    return tuple(results)


def run_post_training_evaluation(
    *,
    config_path: Path,
    acknowledgement: str,
    sampler_factory: Callable[..., EvaluationSampler] | None = None,
    backend_validator: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if acknowledgement != EVALUATION_ACK:
        raise PermissionError("exact Stage 26 acknowledgement is required")
    preflight = preflight_post_training_evaluation(
        config_path=config_path,
        backend_validator=backend_validator,
    )
    if EVALUATION_RAW_ROWS.exists() or EVALUATION_SUMMARY.exists() or EVALUATION_MANIFEST.exists():
        raise RuntimeError("completed Stage 26 evaluation exists; overwrite forbidden")
    rows = select_evaluation_rows(read_jsonl(FIBER_ROWS))
    index = {str(row["scene_id"]): dict(row) for row in rows}
    if sampler_factory is None:
        from .training_backend import create_evaluation_sampler

        sampler_factory = create_evaluation_sampler
    seeds = tuple(
        int(preflight["bootstrap_seed"]) + index for index in range(int(preflight["sampled_rollouts"]))
    )
    for name, details in dict(preflight["arm_manifests"]).items():
        arm_config = {
            "name": name,
            "reward_function_id": details["reward_function_id"],
        }
        sampler = sampler_factory(
            arm_config=arm_config,
            adapter_path=Path(str(details["adapter_path"])),
        )
        reward = build_traced_reward(
            arm_config=arm_config,
            training_rows=rows,
            trace_path=EVALUATION_RAW_ROWS,
            group_size=int(preflight["group_size"]),
        )
        for scene_index, row in enumerate(rows, start=1):
            completions = tuple(sampler(dict(row), seeds))
            if len(completions) != len(seeds):
                raise ValueError("Stage 26 sampler returned the wrong rollout count")
            for start in range(0, len(completions), int(preflight["group_size"])):
                stop = start + int(preflight["group_size"])
                reward(completions[start:stop], scene_id=[row["scene_id"]])
            print(
                f"PROGRESS: {name} Stage 26 scene {scene_index}/{len(rows)} complete",
                flush=True,
            )
    raw_rows = read_jsonl(EVALUATION_RAW_ROWS)
    per_scene = _aggregate_scene_rows(raw_rows, index)
    contrast_rows = tuple(
        {
            **row,
            "arm": (
                "state"
                if str(row["arm"]) == "C2_exact_state_reward"
                else "answer"
            ),
        }
        for row in per_scene
    )
    pair_bootstrap = paired_collision_difference_in_differences(
        contrast_rows,
        resamples=int(preflight["bootstrap_resamples"]),
        seed=int(preflight["bootstrap_seed"]),
    )
    by_arm = {
        arm: {
            "scene_count": len(selected),
            "exact_mean": sum(float(row["exact"]) for row in selected) / len(selected),
            "answer_mean": sum(float(row["answer"]) for row in selected) / len(selected),
            "parse_rate_mean": sum(float(row["parse_rate"]) for row in selected) / len(selected),
        }
        for arm in sorted({str(row["arm"]) for row in per_scene})
        for selected in ([row for row in per_scene if row["arm"] == arm],)
    }
    summary = {
        "schema_version": 2,
        "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
        "sampled_rollouts": int(preflight["sampled_rollouts"]),
        "evaluation_scene_count": int(preflight["evaluation_scene_count"]),
        "evaluation_pair_count": int(preflight["evaluation_pair_count"]),
        "raw_row_count": len(raw_rows),
        "by_arm": by_arm,
        "pair_bootstrap": pair_bootstrap,
        "config_sha256": preflight["config_sha256"],
        "fiber_rows_sha256": preflight["fiber_rows_sha256"],
        "training_pair_manifest_sha256": preflight["training_pair_manifest_sha256"],
        "package_lock_sha256": preflight["package_lock_sha256"],
        "reward_only_pair_verified": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": True,
    }
    write_json_new(EVALUATION_SUMMARY, summary)
    manifest = {
        "schema_version": 2,
        "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
        "raw_rows_sha256": sha256_file(EVALUATION_RAW_ROWS),
        "summary_sha256": sha256_file(EVALUATION_SUMMARY),
        "config_sha256": preflight["config_sha256"],
        "training_pair_manifest_sha256": preflight["training_pair_manifest_sha256"],
        "arm_manifests": {
            name: {"manifest_sha256": details["manifest_sha256"]}
            for name, details in dict(preflight["arm_manifests"]).items()
        },
    }
    write_json_new(EVALUATION_MANIFEST, manifest)
    return summary


preflight_evaluation = preflight_post_training_evaluation
run_evaluation = run_post_training_evaluation

__all__ = [
    "EVALUATION_ACK",
    "preflight_evaluation",
    "preflight_post_training_evaluation",
    "run_evaluation",
    "run_post_training_evaluation",
    "select_evaluation_rows",
    "validate_evaluation_backend_api",
]
