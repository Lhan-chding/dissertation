"""Evaluate the five registered Phase 6 model groups on frozen support-dev."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _guards import PHASE_C_DATASET_RECORDS_SHA256, ROOT, sha256  # noqa: E402

from compensability_v4.eval.statistics import (  # noqa: E402
    holm_adjust,
    scene_clustered_bootstrap_ci,
)
from compensability_v4.qwen.model_loader import (  # noqa: E402
    MODEL_PATH,
    MODEL_SNAPSHOT_SHA256,
    load_pinned_qwen,
)
from compensability_v4.qwen.phase5_runtime import (  # noqa: E402
    freeze_inference_model,
    generate_completion,
    phase5_rollout_seed,
    recovery_prompt,
    tree_sha256,
)
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, parse_world  # noqa: E402
from compensability_v4.qwen.phase6_runtime import (  # noqa: E402
    PHASE6_LOCKED_PATHS,
    load_phase6_execution_manifest,
    verify_phase6_package_lock,
)
from compensability_v4.training.phase6 import (  # noqa: E402
    Phase6Variant,
    load_phase6_config,
)

CONFIG = ROOT / "configs/recoverability/v4_phase_6.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml"
SUPPORT_DEV_ROOT = ROOT / "artifacts/v4/support_dev"
DATASET_RECORDS = ROOT / "data/generated/cva_recoverability_causal_v2_screen/records.jsonl"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
PHASE6_RUN_ROOT = ROOT / "artifacts/v4/rl/runs/phase6-r1"
WORK_ROOT = ROOT / "artifacts/v4/rl/evaluation_work/phase6-r1"
OUTPUT_ROOT = ROOT / "artifacts/v4/rl/evaluation"
EXECUTION_MANIFEST = ROOT / "artifacts/v4/phase6/execution_manifest.json"
_ANSWER = re.compile(r"\s*([+-]?\d+)\s*\Z")
_CHECKPOINTS = (
    "Base",
    Phase6Variant.BASE_ANSWER_ONLY.value,
    "Recovery_LoRA",
    Phase6Variant.RECOVERY_OUTCOME.value,
    Phase6Variant.RECOVERY_ANSWER_ONLY.value,
)


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 6 {label} is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Phase 6 {label} must contain one object")
    return value


def _jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 6 {label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 6 {label} is empty or malformed")
    return rows  # type: ignore[return-value]


def _inputs(
    support_root: Path, dataset_records: Path
) -> tuple[tuple[HeldOutNaturalError, ...], dict[str, dict[str, object]], dict[str, str]]:
    errors_path, support_summary_path = (
        support_root / "held_out_natural_errors.jsonl",
        support_root / "summary.json",
    )
    support_summary = _json(support_summary_path, "support-dev summary")
    errors = tuple(
        HeldOutNaturalError.from_mapping(row)
        for row in _jsonl(errors_path, "held-out natural errors")
    )
    if (
        support_summary.get("status") != "PHASE_5_SUPPORT_DEV_FROZEN"
        or support_summary.get("held_out_natural_error_count") != len(errors)
        or support_summary.get("held_out_natural_errors_sha256") != sha256(errors_path)
        or support_summary.get("confirmatory_data_used") is not False
    ):
        raise RuntimeError("Phase 6 support-dev provenance drifted")
    if sha256(dataset_records) != PHASE_C_DATASET_RECORDS_SHA256:
        raise RuntimeError("Phase 6 dataset records hash drifted")
    records = {str(row["scene_id"]): row for row in _jsonl(dataset_records, "dataset records")}
    if len(records) != 8000 or any(error.scene_id not in records for error in errors):
        raise RuntimeError("Phase 6 evaluation dataset scene closure drifted")
    return (
        errors,
        records,
        {
            "held_out_natural_errors": sha256(errors_path),
            "support_dev_summary": sha256(support_summary_path),
            "dataset_records": PHASE_C_DATASET_RECORDS_SHA256,
        },
    )


def _adapter(checkpoint: str, phase4_root: Path, phase6_root: Path) -> Path | None:
    if checkpoint == "Base":
        return None
    if checkpoint == "Recovery_LoRA":
        return phase4_root / "T_constraint_recovery/final_adapter"
    return phase6_root / checkpoint / "final_adapter"


def _checkpoint_hashes(
    phase4_root: Path,
    phase6_root: Path,
    *,
    manifest: dict[str, object],
    execution_manifest_sha256: str,
) -> dict[str, str]:
    source_hashes = manifest.get("source_sha256")
    phase4_hashes = manifest.get("phase4_adapter_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(phase4_hashes, dict):
        raise RuntimeError("Phase 6 execution manifest is missing checkpoint hashes")
    if source_hashes.get("Base") != MODEL_SNAPSHOT_SHA256:
        raise RuntimeError("Phase 6 Base model hash drifted from the execution manifest")
    if manifest.get("model_snapshot_sha256") != MODEL_SNAPSHOT_SHA256:
        raise RuntimeError("Phase 6 model snapshot drifted from the execution manifest")
    hashes = {"Base": MODEL_SNAPSHOT_SHA256}
    for checkpoint in _CHECKPOINTS[1:]:
        adapter = _adapter(checkpoint, phase4_root, phase6_root)
        assert adapter is not None
        observed_hash = tree_sha256(adapter)
        if checkpoint == "Recovery_LoRA" and (
            observed_hash != phase4_hashes.get("T") or observed_hash != source_hashes.get("T")
        ):
            raise RuntimeError("Phase 6 Recovery_LoRA baseline drifted from the execution manifest")
        evidence_path = phase6_root / checkpoint / "execution_evidence.json"
        if checkpoint != "Recovery_LoRA":
            evidence = _json(evidence_path, f"{checkpoint} execution evidence")
            if (
                evidence.get("status") != "PHASE_6_VARIANT_TRAINED"
                or evidence.get("variant") != checkpoint
                or evidence.get("execution_manifest_sha256") != execution_manifest_sha256
                or evidence.get("final_adapter_tree_sha256") != observed_hash
            ):
                raise RuntimeError(f"Phase 6 {checkpoint} training evidence drifted")
        hashes[checkpoint] = observed_hash
    return hashes


def _load_model(checkpoint: str, phase4_root: Path, phase6_root: Path) -> tuple[object, object]:
    model, processor = load_pinned_qwen(model_path=Path(MODEL_PATH), device_map="cuda:0")
    adapter = _adapter(checkpoint, phase4_root, phase6_root)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    freeze_inference_model(model)
    return model, processor


def _release(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _answer_prompt(error: HeldOutNaturalError, record: dict[str, object]) -> str:
    facts = json.dumps([dict(item) for item in error.facts], sort_keys=True, separators=(",", ":"))
    return (
        f"Observed values: {','.join(map(str, error.observed))}\nFacts: {facts}\n"
        "Use only the observations and facts in this prompt. "
        f"Question: {record['question']}\nReturn the final integer answer only."
    )


def _evaluate_checkpoint(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    model: object,
    processor: object,
    errors: tuple[HeldOutNaturalError, ...],
    records: dict[str, dict[str, object]],
    rollout_count: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for completed, error in enumerate(sorted(errors, key=lambda item: item.scene_id), start=1):
        greedy_raw, greedy_ids = generate_completion(
            model,
            processor,
            recovery_prompt(error),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        greedy_world = parse_world(greedy_raw)
        sample_raw: list[str] = []
        sample_ids: list[list[int]] = []
        sample_success: list[bool] = []
        sample_copy: list[bool] = []
        sample_parse: list[bool] = []
        sample_seeds: list[int] = []
        for rollout_index in range(rollout_count):
            rollout_seed = phase5_rollout_seed(seed, error.scene_id, rollout_index)
            raw, ids = generate_completion(
                model,
                processor,
                recovery_prompt(error),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=rollout_seed,
            )
            world = parse_world(raw)
            sample_raw.append(raw)
            sample_ids.append(list(ids))
            sample_success.append(world == error.truth)
            sample_copy.append(world == error.observed)
            sample_parse.append(world is not None)
            sample_seeds.append(rollout_seed)
        record = records[error.scene_id]
        answer_raw, answer_ids = generate_completion(
            model,
            processor,
            _answer_prompt(error, record),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        match = _ANSWER.fullmatch(answer_raw)
        answer_value = None if match is None else int(match.group(1))
        rows.append(
            {
                "schema_version": 1,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "scene_id": error.scene_id,
                "family": error.family,
                "truth": list(error.truth),
                "observed": list(error.observed),
                "greedy_recovery_raw": greedy_raw,
                "greedy_recovery_token_ids": list(greedy_ids),
                "greedy_recovery_parse_success": greedy_world is not None,
                "greedy_exact_world_recovery": greedy_world == error.truth,
                "greedy_observation_copy": greedy_world == error.observed,
                "sample_raw_outputs": sample_raw,
                "sample_token_ids": sample_ids,
                "sample_seeds": sample_seeds,
                "sample_parse_success": sample_parse,
                "sample_exact_world_recovery": sample_success,
                "sample_observation_copy": sample_copy,
                "answer_raw": answer_raw,
                "answer_token_ids": list(answer_ids),
                "answer_parse_success": answer_value is not None,
                "answer_exact": answer_value == record["answer"],
                "operation": record["operation"],
            }
        )
        if completed % 10 == 0 or completed == len(errors):
            print(
                f"PROGRESS: Phase 6 {checkpoint} {completed}/{len(errors)} scenes complete",
                flush=True,
            )
    return tuple(rows)


def _cache(
    root: Path,
    checkpoint: str,
    *,
    rows: tuple[dict[str, object], ...] | None,
    checkpoint_sha256: str,
    input_sha256: dict[str, str],
    config_sha256: str,
    package_lock_sha256: str,
    execution_manifest_sha256: str,
) -> tuple[dict[str, object], ...] | None:
    rows_path, meta_path = root / f"{checkpoint}.jsonl", root / f"{checkpoint}.meta.json"
    if rows is None:
        if not rows_path.exists() and not meta_path.exists():
            return None
        meta = _json(meta_path, f"{checkpoint} evaluation cache")
        cached = _jsonl(rows_path, f"{checkpoint} evaluation rows")
        if meta != {
            "schema_version": 1,
            "status": "PHASE_6_CHECKPOINT_EVALUATION_COMPLETE",
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "input_sha256": input_sha256,
            "config_sha256": config_sha256,
            "package_lock_sha256": package_lock_sha256,
            "execution_manifest_sha256": execution_manifest_sha256,
            "row_count": len(cached),
            "rows_sha256": sha256(rows_path),
        }:
            raise RuntimeError(f"Phase 6 {checkpoint} evaluation cache drifted")
        return cached
    if rows_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6 {checkpoint} evaluation cache")
    root.mkdir(parents=True, exist_ok=True)
    with rows_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with meta_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "PHASE_6_CHECKPOINT_EVALUATION_COMPLETE",
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "input_sha256": input_sha256,
                "config_sha256": config_sha256,
                "package_lock_sha256": package_lock_sha256,
                "execution_manifest_sha256": execution_manifest_sha256,
                "row_count": len(rows),
                "rows_sha256": sha256(rows_path),
            },
            stream,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    return rows


def _mean(rows: tuple[dict[str, object], ...], key: str) -> float:
    return sum(row[key] is True for row in rows) / len(rows)


def _interval(
    values: dict[str, float], *, bootstrap_resamples: int, seed: int
) -> dict[str, object]:
    interval = scene_clustered_bootstrap_ci(
        ({"scene_id": scene_id, "value": value} for scene_id, value in sorted(values.items())),
        metric="value",
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "estimate": interval.estimate,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "confidence": interval.confidence,
        "number_of_scenes": interval.number_of_scenes,
    }


def _checkpoint_values(
    rows: tuple[dict[str, object], ...], checkpoint: str, metric: str
) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        if row["checkpoint"] != checkpoint:
            continue
        scene_id = str(row["scene_id"])
        if metric == "sample_exact_world_recovery":
            samples = row[metric]
            if not isinstance(samples, list) or not samples:
                raise RuntimeError("Phase 6 sampled recovery values are malformed")
            value = sum(item is True for item in samples) / len(samples)
        else:
            value = float(row[metric] is True)
        if scene_id in output:
            raise RuntimeError("Phase 6 evaluation contains duplicate checkpoint-scene rows")
        output[scene_id] = value
    if not output:
        raise RuntimeError(f"Phase 6 evaluation contains no {checkpoint} values")
    return output


def _effect(
    differences: dict[str, float], *, bootstrap_resamples: int, seed: int
) -> dict[str, object]:
    interval = _interval(differences, bootstrap_resamples=bootstrap_resamples, seed=seed)
    observed = abs(sum(differences.values()) / len(differences))
    values = tuple(differences[scene_id] for scene_id in sorted(differences))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(bootstrap_resamples):
        null_effect = abs(
            sum(value if rng.random() < 0.5 else -value for value in values) / len(values)
        )
        exceedances += null_effect >= observed
    return {
        **interval,
        "two_sided_sign_flip_p_value": (exceedances + 1) / (bootstrap_resamples + 1),
    }


def _summary(
    rows: tuple[dict[str, object], ...],
    rollout_count: int,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    by_checkpoint: dict[str, object] = {}
    interval_metrics = (
        "greedy_exact_world_recovery",
        "greedy_observation_copy",
        "sample_exact_world_recovery",
        "answer_exact",
    )
    for checkpoint_position, checkpoint in enumerate(_CHECKPOINTS):
        selected = tuple(row for row in rows if row["checkpoint"] == checkpoint)
        if not selected:
            raise RuntimeError(f"Phase 6 evaluation is missing {checkpoint}")
        samples = [
            value
            for row in selected
            for value in row["sample_exact_world_recovery"]  # type: ignore[union-attr]
        ]
        copies = [
            value
            for row in selected
            for value in row["sample_observation_copy"]  # type: ignore[union-attr]
        ]
        if len(samples) != len(selected) * rollout_count:
            raise RuntimeError("Phase 6 evaluation rollout closure drifted")
        by_checkpoint[checkpoint] = {
            "scene_count": len(selected),
            "greedy_exact_world_recovery_rate": _mean(selected, "greedy_exact_world_recovery"),
            "greedy_observation_copy_rate": _mean(selected, "greedy_observation_copy"),
            "sample_exact_world_recovery_rate": sum(samples) / len(samples),
            "sample_observation_copy_rate": sum(copies) / len(copies),
            "answer_exact_rate": _mean(selected, "answer_exact"),
            "confidence_intervals": {
                metric: _interval(
                    _checkpoint_values(rows, checkpoint, metric),
                    bootstrap_resamples=bootstrap_resamples,
                    seed=bootstrap_seed + checkpoint_position * 10 + metric_position,
                )
                for metric_position, metric in enumerate(interval_metrics)
            },
            "by_family": {
                family: {
                    "scene_count": len(family_rows),
                    "greedy_exact_world_recovery_rate": _mean(
                        family_rows, "greedy_exact_world_recovery"
                    ),
                    "answer_exact_rate": _mean(family_rows, "answer_exact"),
                }
                for family in sorted({str(row["family"]) for row in selected})
                if (family_rows := tuple(row for row in selected if row["family"] == family))
            },
        }

    answer = {
        checkpoint: _checkpoint_values(rows, checkpoint, "answer_exact")
        for checkpoint in _CHECKPOINTS
    }
    recovery = {
        checkpoint: _checkpoint_values(rows, checkpoint, "greedy_exact_world_recovery")
        for checkpoint in _CHECKPOINTS
    }
    scene_ids = tuple(sorted(answer["Base"]))
    if any(tuple(sorted(values)) != scene_ids for values in (*answer.values(), *recovery.values())):
        raise RuntimeError("Phase 6 registered effects lack paired scene closure")
    effect_values = {
        "answer_only_rl_effect_from_base_on_answer": {
            scene_id: answer[Phase6Variant.BASE_ANSWER_ONLY.value][scene_id]
            - answer["Base"][scene_id]
            for scene_id in scene_ids
        },
        "answer_only_rl_effect_from_recovery_on_answer": {
            scene_id: answer[Phase6Variant.RECOVERY_ANSWER_ONLY.value][scene_id]
            - answer["Recovery_LoRA"][scene_id]
            for scene_id in scene_ids
        },
        "recovery_outcome_rl_effect_from_recovery_on_world": {
            scene_id: recovery[Phase6Variant.RECOVERY_OUTCOME.value][scene_id]
            - recovery["Recovery_LoRA"][scene_id]
            for scene_id in scene_ids
        },
        "seeded_minus_base_answer_only_rl_effect": {
            scene_id: (
                answer[Phase6Variant.RECOVERY_ANSWER_ONLY.value][scene_id]
                - answer["Recovery_LoRA"][scene_id]
            )
            - (answer[Phase6Variant.BASE_ANSWER_ONLY.value][scene_id] - answer["Base"][scene_id])
            for scene_id in scene_ids
        },
    }
    effects = {
        name: _effect(
            values,
            bootstrap_resamples=bootstrap_resamples,
            seed=bootstrap_seed + 100 + position,
        )
        for position, (name, values) in enumerate(effect_values.items())
    }
    adjusted = holm_adjust(
        {name: float(effect["two_sided_sign_flip_p_value"]) for name, effect in effects.items()}
    )
    effects = {
        name: {**effect, "holm_adjusted_p_value": adjusted[name]}
        for name, effect in effects.items()
    }
    family_by_scene = {
        str(row["scene_id"]): str(row["family"]) for row in rows if row["checkpoint"] == "Base"
    }
    by_family_effect = {
        family: {
            name: _effect(
                {
                    scene_id: value
                    for scene_id, value in values.items()
                    if family_by_scene[scene_id] == family
                },
                bootstrap_resamples=bootstrap_resamples,
                seed=bootstrap_seed + 1_000 + family_position * 10 + effect_position,
            )
            for effect_position, (name, values) in enumerate(effect_values.items())
        }
        for family_position, family in enumerate(sorted(set(family_by_scene.values())))
    }

    return {
        "schema_version": 1,
        "status": "PHASE_6_RL_EVALUATED",
        "number_of_scenes": len(rows) // len(_CHECKPOINTS),
        "number_of_checkpoint_scene_rows": len(rows),
        "by_checkpoint": by_checkpoint,
        "registered_effects": effects,
        "registered_effects_by_family": by_family_effect,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "training_seeds": [2026082006],
        "number_of_training_seeds": 1,
        "scene_is_statistical_unit": True,
        "confirmatory_data_used": False,
        "subjective_success_threshold_applied": False,
        "training_invoked": False,
        "rl_invoked": False,
    }


def _publish(root: Path, rows: tuple[dict[str, object], ...], summary: dict[str, object]) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 6 evaluation")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase6-evaluation-", dir=str(root.parent)))
    try:
        rows_path, summary_path = temporary / "by_scene.jsonl", temporary / "summary.json"
        with rows_path.open("x", encoding="utf-8") as stream:
            for row in sorted(
                rows, key=lambda item: (str(item["scene_id"]), str(item["checkpoint"]))
            ):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        published_summary = {**summary, "by_scene_sha256": sha256(rows_path)}
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(published_summary, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.rename(root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--execution-manifest", type=Path, default=EXECUTION_MANIFEST)
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--support-dev-root", type=Path, default=SUPPORT_DEV_ROOT)
    parser.add_argument("--dataset-records", type=Path, default=DATASET_RECORDS)
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--phase6-run-root", type=Path, default=PHASE6_RUN_ROOT)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 6 evaluation requires explicit --execute.")
        return 2
    if not arguments.execution_manifest_sha256:
        print("BLOCKED: Phase 6 evaluation requires --execution-manifest-sha256.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        payload, _training = load_phase6_config(arguments.config)
        lock_hash = verify_phase6_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE6_LOCKED_PATHS,
        )
        config_hash = sha256(arguments.config)
        manifest = load_phase6_execution_manifest(
            arguments.execution_manifest,
            expected_sha256=arguments.execution_manifest_sha256,
            expected_config_sha256=config_hash,
            expected_package_lock_sha256=lock_hash,
        )
        evaluation = payload["evaluation"]
        assert isinstance(evaluation, dict)
        errors, records, input_hashes = _inputs(
            arguments.support_dev_root, arguments.dataset_records
        )
        if manifest["source_sha256"]["support_dev"] != input_hashes["held_out_natural_errors"]:
            raise RuntimeError("Phase 6 support-dev errors drifted from the execution manifest")
        checkpoint_hashes = _checkpoint_hashes(
            arguments.phase4_run_root,
            arguments.phase6_run_root,
            manifest=manifest,
            execution_manifest_sha256=arguments.execution_manifest_sha256,
        )
        all_rows: list[dict[str, object]] = []
        for checkpoint in _CHECKPOINTS:
            cached = _cache(
                arguments.work_root,
                checkpoint,
                rows=None,
                checkpoint_sha256=checkpoint_hashes[checkpoint],
                input_sha256=input_hashes,
                config_sha256=config_hash,
                package_lock_sha256=lock_hash,
                execution_manifest_sha256=arguments.execution_manifest_sha256,
            )
            if cached is not None:
                rows = cached
                print(f"RESUMED: Phase 6 {checkpoint} evaluation evidence", flush=True)
            else:
                model, processor = _load_model(
                    checkpoint, arguments.phase4_run_root, arguments.phase6_run_root
                )
                rows = _evaluate_checkpoint(
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_hashes[checkpoint],
                    model=model,
                    processor=processor,
                    errors=errors,
                    records=records,
                    rollout_count=int(evaluation["rollout_count"]),
                    seed=int(evaluation["seed"]),
                    temperature=float(evaluation["temperature"]),
                    top_p=float(evaluation["top_p"]),
                    top_k=int(evaluation["top_k"]),
                    max_new_tokens=int(evaluation["max_new_tokens"]),
                )
                _release(model)
                _cache(
                    arguments.work_root,
                    checkpoint,
                    rows=rows,
                    checkpoint_sha256=checkpoint_hashes[checkpoint],
                    input_sha256=input_hashes,
                    config_sha256=config_hash,
                    package_lock_sha256=lock_hash,
                    execution_manifest_sha256=arguments.execution_manifest_sha256,
                )
            all_rows.extend(rows)
        summary = {
            **_summary(
                tuple(all_rows),
                int(evaluation["rollout_count"]),
                bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
                bootstrap_seed=int(evaluation["bootstrap_seed"]),
            ),
            "config_sha256": config_hash,
            "package_lock_sha256": lock_hash,
            "execution_manifest_sha256": arguments.execution_manifest_sha256,
            "input_sha256": input_hashes,
            "checkpoint_sha256": checkpoint_hashes,
        }
        _publish(arguments.output_root, tuple(all_rows), summary)
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: Phase 6 evaluation written below {arguments.output_root}")
    for path in sorted(arguments.output_root.iterdir()):
        print(f"SHA256 {sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
