"""Evaluate the seven Phase 7 checkpoints through the real multimodal chain."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compensability_v4.data.splits import DatasetSplit  # noqa: E402
from compensability_v4.qwen.manual_generation import (  # noqa: E402
    generate_observation_with_cache,
)
from compensability_v4.qwen.model_loader import (  # noqa: E402
    MODEL_PATH,
    MODEL_SNAPSHOT_SHA256,
    load_pinned_qwen,
)
from compensability_v4.qwen.phase5_runtime import (  # noqa: E402
    freeze_inference_model,
    generate_completion,
    tree_sha256,
)
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, parse_world  # noqa: E402
from compensability_v4.qwen.phase7_runtime import (  # noqa: E402
    PHASE7_LOCKED_PATHS,
    Phase7ChainRow,
    load_phase7_config,
    summarize_phase7,
    validate_phase7_execution_manifest,
    validate_phase7_rows,
    verify_phase7_package_lock,
    write_phase7_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_7.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_7.yaml"
MANIFEST = ROOT / "artifacts/v4/phase7/execution_manifest.json"
SUPPORT_DEV = ROOT / "artifacts/v4/support_dev/held_out_natural_errors.jsonl"
DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"
DATASET_RECORDS = DATASET_ROOT / "records.jsonl"
PHASE4_SUMMARY = ROOT / "artifacts/v4/training/support_summary.json"
PHASE5_SUMMARY = ROOT / "artifacts/v4/support/informative_group_rate.json"
PHASE6_EVALUATION = ROOT / "artifacts/v4/rl/evaluation/summary.json"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
PHASE6_RUN_ROOT = ROOT / "artifacts/v4/rl/runs/phase6-r1"
PROMPTS = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
WORK_ROOT = ROOT / "artifacts/v4/phase7/work/phase7-r1"
OUTPUT_ROOT = ROOT / "artifacts/v4/phase7/evaluation"
_WORLD = re.compile(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\Z")
_ANSWER = re.compile(r"\s*([+-]?\d+)\s*\Z")
_OPERATIONS = frozenset({"sum", "difference", "max_minus_min"})
_CHECKPOINTS = (
    "Base",
    "C0",
    "C1",
    "T",
    "Base_AnswerOnly_RL",
    "Recovery_LoRA_RecoveryOutcome_RL",
    "Recovery_LoRA_AnswerOnly_RL",
)
_SOURCE_PATHS = {
    "dataset_records": DATASET_RECORDS,
    "support_dev": SUPPORT_DEV,
    "phase4_summary": PHASE4_SUMMARY,
    "phase5_summary": PHASE5_SUMMARY,
    "phase6_evaluation": PHASE6_EVALUATION,
}
_RESIZED_HEIGHT = 280
_RESIZED_WIDTH = 280
_STAGE1_MAX_NEW_TOKENS = 32
_RECOVERY_MAX_NEW_TOKENS = 32
_OPERATION_MAX_NEW_TOKENS = 8
_ANSWER_MAX_NEW_TOKENS = 8
_GENERATION_SEED = 2026082102
_EXECUTION_PARAMETERS = {
    "resized_height": _RESIZED_HEIGHT,
    "resized_width": _RESIZED_WIDTH,
    "stage1_max_new_tokens": _STAGE1_MAX_NEW_TOKENS,
    "recovery_max_new_tokens": _RECOVERY_MAX_NEW_TOKENS,
    "operation_max_new_tokens": _OPERATION_MAX_NEW_TOKENS,
    "answer_max_new_tokens": _ANSWER_MAX_NEW_TOKENS,
    "greedy_seed": _GENERATION_SEED,
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 7 {label} is missing or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Phase 7 {label} must contain one JSON object")
    return payload


def _jsonl(path: Path, label: str) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Phase 7 {label} is missing or unsafe")
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Phase 7 {label} is empty or malformed")
    return rows  # type: ignore[return-value]


def _support_dev_image_bundle_sha256(
    *, support_dev: Path, dataset_records: Path, dataset_root: Path
) -> str:
    import hashlib

    if not dataset_root.is_absolute() or dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RuntimeError("Phase 7 dataset root must be an absolute regular directory")
    root = dataset_root.resolve()
    records = {str(row.get("scene_id")): row for row in _jsonl(dataset_records, "dataset records")}
    errors = _jsonl(support_dev, "support-dev errors")
    if len(errors) != 32 or len({row.get("scene_id") for row in errors}) != 32:
        raise RuntimeError("Phase 7 support-dev image bundle requires exactly 32 unique scenes")
    bundle: list[tuple[str, str, Path]] = []
    for error in errors:
        scene_id, relative = error.get("scene_id"), error.get("image_path")
        record = records.get(str(scene_id))
        if (
            not isinstance(scene_id, str)
            or not isinstance(relative, str)
            or not isinstance(record, dict)
            or record.get("image") != relative
        ):
            raise RuntimeError("Phase 7 support-dev image mapping drifted")
        image = _image(root, relative)
        bundle.append((scene_id, relative, image))
    digest = hashlib.sha256()
    for scene_id, relative, image in sorted(bundle):
        digest.update(scene_id.encode())
        digest.update(b"\0")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(_sha256(image).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _stage1_prompt(path: Path) -> str:
    import yaml

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Phase 7 Stage-1 prompt config is missing or unsafe")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    prompt = prompts.get("stage_1_observation") if isinstance(prompts, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Phase 7 Stage-1 observation prompt is missing")
    return prompt


def _adapter(checkpoint: str, phase4_root: Path, phase6_root: Path) -> Path | None:
    phase4 = {
        "C0": "C0_format_only/final_adapter",
        "C1": "C1_forward_arithmetic/final_adapter",
        "T": "T_constraint_recovery/final_adapter",
    }
    if checkpoint == "Base":
        return None
    if checkpoint in phase4:
        return phase4_root / phase4[checkpoint]
    return phase6_root / checkpoint / "final_adapter"


def _checkpoint_hashes(
    phase4_root: Path,
    phase6_root: Path,
    *,
    phase5_summary: dict[str, object],
    phase6_evaluation: dict[str, object],
) -> dict[str, str]:
    hashes = {"Base": MODEL_SNAPSHOT_SHA256}
    for checkpoint in _CHECKPOINTS[1:]:
        adapter = _adapter(checkpoint, phase4_root, phase6_root)
        assert adapter is not None
        observed = tree_sha256(adapter)
        if checkpoint not in {"C0", "C1", "T"}:
            evidence = _json(phase6_root / checkpoint / "execution_evidence.json", checkpoint)
            if (
                evidence.get("status") != "PHASE_6_VARIANT_TRAINED"
                or evidence.get("variant") != checkpoint
                or evidence.get("final_adapter_tree_sha256") != observed
            ):
                raise RuntimeError(f"Phase 7 {checkpoint} training evidence drifted")
        hashes[checkpoint] = observed
    phase5_hashes = phase5_summary.get("source_sha256")
    phase6_hashes = phase6_evaluation.get("checkpoint_sha256")
    if not isinstance(phase5_hashes, dict) or not isinstance(phase6_hashes, dict):
        raise RuntimeError("Phase 7 upstream checkpoint provenance is missing")
    for checkpoint in ("Base", "C0", "C1", "T"):
        if hashes[checkpoint] != phase5_hashes.get(checkpoint):
            raise RuntimeError(f"Phase 7 {checkpoint} hash differs from Phase 5 evidence")
    for checkpoint in _CHECKPOINTS[4:]:
        if hashes[checkpoint] != phase6_hashes.get(checkpoint):
            raise RuntimeError(f"Phase 7 {checkpoint} hash differs from Phase 6 evidence")
    return hashes


def _load_model(checkpoint: str, phase4_root: Path, phase6_root: Path) -> tuple[object, object]:
    model, processor = load_pinned_qwen(model_path=Path(MODEL_PATH), device_map="cuda:0")
    adapter = _adapter(checkpoint, phase4_root, phase6_root)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    freeze_inference_model(model)
    return model, processor


def _release() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def revision_or_recovery(
    model: object,
    processor: object,
    *,
    observed_raw: str,
    facts: tuple[object, ...],
    seed: int,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...]]:
    """Generate a revised world from the checkpoint's own observation and frozen facts."""

    prompt = (
        f"Observed values: {observed_raw}\n"
        f"Facts: {json.dumps(list(facts), sort_keys=True, separators=(',', ':'))}\n"
        "Revise the observed values only when required by the facts. "
        "Return exactly four comma-separated integers and no other text."
    )
    raw, token_ids = generate_completion(
        model,
        processor,
        prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    return raw, tuple(token_ids)


def chart_operation(
    model: object,
    processor: object,
    *,
    question: str,
    seed: int,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], str | None]:
    """Generate the registered chart operator required by the question."""

    prompt = (
        f"Question: {question}\nChoose the required chart operation. "
        "Return exactly one of: sum, difference, max_minus_min."
    )
    raw, token_ids = generate_completion(
        model,
        processor,
        prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    parsed = raw.strip() if raw.strip() in _OPERATIONS else None
    return raw, tuple(token_ids), parsed


def final_answer(
    model: object,
    processor: object,
    *,
    recovered_raw: str,
    chosen_operation: str | None,
    seed: int,
    max_new_tokens: int,
) -> tuple[str, tuple[int, ...], int | None]:
    """Generate the final integer from the model-recovered world and model-chosen operator."""

    operation = chosen_operation if chosen_operation is not None else "INVALID_OPERATION"
    prompt = (
        f"Recovered values: {recovered_raw}\nChart operation: {operation}\n"
        "Apply the chart operation to the recovered values. Return the final integer only."
    )
    raw, token_ids = generate_completion(
        model,
        processor,
        prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    match = _ANSWER.fullmatch(raw)
    return raw, tuple(token_ids), None if match is None else int(match.group(1))


def _apply_operation(world: tuple[int, int, int, int], operation: str) -> int:
    if operation == "sum":
        return world[0] + world[1]
    if operation == "difference":
        return world[0] - world[1]
    if operation == "max_minus_min":
        return max(world) - min(world)
    raise ValueError("Phase 7 operation is outside the frozen set")


def _trace_mismatch(*, answer_value: int | None, chosen_execution: int | None) -> bool:
    return answer_value is None or chosen_execution is None or chosen_execution != answer_value


def _image(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    image = (root / Path(*posix.parts)).resolve()
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or posix.suffix.lower() != ".png"
        or root.resolve() not in image.parents
        or image.is_symlink()
        or not image.is_file()
    ):
        raise RuntimeError("Phase 7 support-dev image is missing or unsafe")
    return image


def _evaluate_checkpoint(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    model: object,
    processor: object,
    errors: tuple[HeldOutNaturalError, ...],
    records: dict[str, dict[str, object]],
    dataset_root: Path,
    stage1_prompt: str,
    resized_height: int,
    resized_width: int,
    stage1_max_new_tokens: int,
    recovery_max_new_tokens: int,
    operation_max_new_tokens: int,
    answer_max_new_tokens: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    evidence_rows: list[dict[str, object]] = []
    for completed, error in enumerate(errors, start=1):
        image = _image(dataset_root, error.image_path)
        stage1 = generate_observation_with_cache(
            model,
            processor,
            str(image),
            stage1_prompt,
            sample_id=f"phase7:{checkpoint}:{error.scene_id}:{seed}",
            resized_height=resized_height,
            resized_width=resized_width,
            max_new_tokens=stage1_max_new_tokens,
            rng_seed=seed,
        )
        stage1_raw = str(stage1["text"])
        stage1_world = parse_world(stage1_raw)
        recovery_raw, recovery_ids = revision_or_recovery(
            model,
            processor,
            observed_raw=stage1_raw,
            facts=tuple(dict(fact) for fact in error.facts),
            seed=seed,
            max_new_tokens=recovery_max_new_tokens,
        )
        recovered_world = parse_world(recovery_raw)
        record = records[error.scene_id]
        operation_raw, operation_ids, chosen_operation = chart_operation(
            model,
            processor,
            question=str(record["question"]),
            seed=seed,
            max_new_tokens=operation_max_new_tokens,
        )
        answer_raw, answer_ids, answer_value = final_answer(
            model,
            processor,
            recovered_raw=recovery_raw,
            chosen_operation=chosen_operation,
            seed=seed,
            max_new_tokens=answer_max_new_tokens,
        )
        true_operation, true_answer = str(record["operation"]), int(record["answer"])
        stage1_exact = stage1_world == error.truth
        world_exact = recovered_world == error.truth
        answer_exact = answer_value == true_answer
        perceived_answer = (
            None if recovered_world is None else _apply_operation(recovered_world, true_operation)
        )
        operator_invariant = bool(
            answer_exact
            and recovered_world is not None
            and not world_exact
            and perceived_answer == true_answer
        )
        error_cancellation = bool(answer_exact and not world_exact and not operator_invariant)
        chosen_execution = (
            None
            if recovered_world is None or chosen_operation is None
            else _apply_operation(recovered_world, chosen_operation)
        )
        mismatch_indices = (
            ()
            if stage1_world is None
            else tuple(
                index
                for index, pair in enumerate(zip(stage1_world, error.truth, strict=True))
                if pair[0] != pair[1]
            )
        )
        chain_row = Phase7ChainRow.from_mapping(
            {
                "scene_id": error.scene_id,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "family": error.family,
                "split": DatasetSplit.SUPPORT_DEV.value,
                "ood_axis": "iid",
                "seed": seed,
                "rollout_id": 0,
                "image_sha256": _sha256(image),
                "stage1_visual_exact": stage1_exact,
                "post_revision_world_exact": world_exact,
                "reasoning_operator_exact": chosen_operation == true_operation,
                "final_answer_exact": answer_exact,
                "operator_invariant_correct": operator_invariant,
                "genuine_recovery": bool(not stage1_exact and world_exact),
                "error_cancellation": error_cancellation,
                "trace_mismatch": _trace_mismatch(
                    answer_value=answer_value, chosen_execution=chosen_execution
                ),
                "error_mechanism_shift": bool(
                    stage1_world is None or mismatch_indices != error.error_indices
                ),
            }
        )
        evidence_rows.append(
            {
                "schema_version": 1,
                "chain_row": chain_row.to_mapping(),
                "truth": list(error.truth),
                "frozen_base_observed": list(error.observed),
                "frozen_base_error_indices": list(error.error_indices),
                "stage1_raw": stage1_raw,
                "stage1_token_ids": list(stage1["generated_token_ids"]),
                "stage1_parse_success": stage1_world is not None,
                "stage1_world": None if stage1_world is None else list(stage1_world),
                "stage1_error_indices": list(mismatch_indices),
                "revision_or_recovery_raw": recovery_raw,
                "revision_or_recovery_token_ids": list(recovery_ids),
                "revision_or_recovery_parse_success": recovered_world is not None,
                "recovered_world": None if recovered_world is None else list(recovered_world),
                "chart_operation_raw": operation_raw,
                "chart_operation_token_ids": list(operation_ids),
                "chart_operation_parse_success": chosen_operation is not None,
                "chosen_operation": chosen_operation,
                "ground_truth_operation": true_operation,
                "final_answer_operation_source": "model_chosen_operation",
                "final_answer_raw": answer_raw,
                "final_answer_token_ids": list(answer_ids),
                "final_answer_parse_success": answer_value is not None,
                "final_answer": answer_value,
                "ground_truth_answer": true_answer,
                "chosen_operation_execution": chosen_execution,
            }
        )
        if completed % 10 == 0 or completed == len(errors):
            print(
                f"PROGRESS: Phase 7 {checkpoint} {completed}/{len(errors)} scenes complete",
                flush=True,
            )
    return tuple(evidence_rows)


def _cache(
    root: Path,
    checkpoint: str,
    *,
    rows: tuple[dict[str, object], ...] | None,
    checkpoint_sha256: str,
    execution_manifest_sha256: str,
    config_sha256: str,
    package_lock_sha256: str,
) -> tuple[dict[str, object], ...] | None:
    destination = root / checkpoint
    if rows is None:
        if not destination.exists():
            return None
        evidence_path, meta_path = destination / "trace_evidence.jsonl", destination / "meta.json"
        cached, meta = (
            _jsonl(evidence_path, f"{checkpoint} trace cache"),
            _json(meta_path, f"{checkpoint} trace-cache metadata"),
        )
        expected = {
            "schema_version": 1,
            "status": "PHASE_7_CHECKPOINT_EVALUATION_COMPLETE",
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "execution_manifest_sha256": execution_manifest_sha256,
            "config_sha256": config_sha256,
            "package_lock_sha256": package_lock_sha256,
            "row_count": len(cached),
            "trace_evidence_sha256": _sha256(evidence_path),
        }
        if meta != expected:
            raise RuntimeError(f"Phase 7 {checkpoint} trace cache drifted")
        return cached
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase 7 {checkpoint} trace evidence")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint}-", dir=root))
    try:
        evidence_path = temporary / "trace_evidence.jsonl"
        with evidence_path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (temporary / "meta.json").open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "status": "PHASE_7_CHECKPOINT_EVALUATION_COMPLETE",
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": checkpoint_sha256,
                    "execution_manifest_sha256": execution_manifest_sha256,
                    "config_sha256": config_sha256,
                    "package_lock_sha256": package_lock_sha256,
                    "row_count": len(rows),
                    "trace_evidence_sha256": _sha256(evidence_path),
                },
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--execution-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--support-dev", type=Path, default=SUPPORT_DEV)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--dataset-records", type=Path, default=DATASET_RECORDS)
    parser.add_argument("--phase4-summary", type=Path, default=PHASE4_SUMMARY)
    parser.add_argument("--phase5-summary", type=Path, default=PHASE5_SUMMARY)
    parser.add_argument("--phase6-evaluation", type=Path, default=PHASE6_EVALUATION)
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--phase6-run-root", type=Path, default=PHASE6_RUN_ROOT)
    parser.add_argument("--prompt-config", type=Path, default=PROMPTS)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-split", default=DatasetSplit.SUPPORT_DEV.value)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 7 multimodal evaluation requires explicit --execute.")
        return 2
    if not arguments.execution_manifest_sha256:
        print("BLOCKED: Phase 7 requires --execution-manifest-sha256.")
        return 2
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("Phase 7 requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
        config = load_phase7_config(arguments.config)
        split = DatasetSplit(arguments.dataset_split)
        confirmatory = split in {
            DatasetSplit.CONFIRM_IID,
            DatasetSplit.CONFIRM_STYLE_OOD,
            DatasetSplit.CONFIRM_CONSTRAINT_OOD,
        }
        if confirmatory and not config.confirmatory_evaluation_authorized:
            raise RuntimeError(
                "Phase 7 confirmatory evaluation is not authorized; support_dev diagnostics only"
            )
        if split is not DatasetSplit.SUPPORT_DEV:
            raise RuntimeError("Phase 7 this entrypoint accepts support_dev diagnostics only")
        lock_hash = verify_phase7_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE7_LOCKED_PATHS,
        )
        manifest_hash = _sha256(arguments.execution_manifest)
        if manifest_hash != arguments.execution_manifest_sha256:
            raise RuntimeError("Phase 7 execution manifest SHA-256 mismatch")
        manifest = validate_phase7_execution_manifest(
            _json(arguments.execution_manifest, "execution manifest"),
            config=config,
            config_sha256=_sha256(arguments.config),
            package_lock_sha256=lock_hash,
        )
        if manifest.get("execution_parameters") != _EXECUTION_PARAMETERS:
            raise RuntimeError("Phase 7 execution parameters drifted")
        if manifest.get("stage1_prompt_config_sha256") != _sha256(arguments.prompt_config):
            raise RuntimeError("Phase 7 Stage-1 prompt config drifted")
        config_hash = _sha256(arguments.config)
        source_paths = {
            "dataset_records": arguments.dataset_records,
            "support_dev": arguments.support_dev,
            "phase4_summary": arguments.phase4_summary,
            "phase5_summary": arguments.phase5_summary,
            "phase6_evaluation": arguments.phase6_evaluation,
        }
        source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
        if manifest.get("source_sha256") != source_hashes:
            raise RuntimeError("Phase 7 source artifacts drifted from the execution manifest")
        image_bundle_hash = _support_dev_image_bundle_sha256(
            support_dev=arguments.support_dev,
            dataset_records=arguments.dataset_records,
            dataset_root=arguments.dataset_root,
        )
        if manifest.get("support_dev_image_bundle_sha256") != image_bundle_hash:
            raise RuntimeError("Phase 7 support-dev image bundle drifted")
        checkpoint_hashes = _checkpoint_hashes(
            arguments.phase4_run_root,
            arguments.phase6_run_root,
            phase5_summary=_json(arguments.phase5_summary, "Phase-5 summary"),
            phase6_evaluation=_json(arguments.phase6_evaluation, "Phase-6 evaluation"),
        )
        if manifest.get("checkpoint_sha256") != checkpoint_hashes:
            raise RuntimeError("Phase 7 checkpoint artifacts drifted from the execution manifest")
        errors = tuple(
            HeldOutNaturalError.from_mapping(row)
            for row in _jsonl(arguments.support_dev, "support-dev errors")
        )
        records = {
            str(row["scene_id"]): row
            for row in _jsonl(arguments.dataset_records, "dataset records")
        }
        if not errors or any(error.scene_id not in records for error in errors):
            raise RuntimeError("Phase 7 support-dev scenes do not close against dataset records")
        prompt = _stage1_prompt(arguments.prompt_config)
        for error in errors:
            record = records[error.scene_id]
            operation, answer, question = (
                record.get("operation"),
                record.get("answer"),
                record.get("question"),
            )
            if (
                operation not in _OPERATIONS
                or type(answer) is not int
                or not isinstance(question, str)
                or not question.strip()
                or _apply_operation(error.truth, str(operation)) != answer
            ):
                raise RuntimeError("Phase 7 support-dev operation/answer closure drifted")
            _image(arguments.dataset_root.resolve(), error.image_path)
        if arguments.output_root.exists() or arguments.output_root.is_symlink():
            raise FileExistsError("refusing to overwrite Phase 7 outputs")
        if arguments.preflight_only:
            for checkpoint in _CHECKPOINTS:
                model, processor = _load_model(
                    checkpoint, arguments.phase4_run_root, arguments.phase6_run_root
                )
                del model, processor
                _release()
                print(f"PREFLIGHT: Phase 7 {checkpoint} load passed", flush=True)
            print("READY: Phase 7 full multimodal preflight passed")
            return 0
        all_chain_rows: list[Phase7ChainRow] = []
        for checkpoint in _CHECKPOINTS:
            cached = _cache(
                arguments.work_root,
                checkpoint,
                rows=None,
                checkpoint_sha256=checkpoint_hashes[checkpoint],
                execution_manifest_sha256=manifest_hash,
                config_sha256=config_hash,
                package_lock_sha256=lock_hash,
            )
            if cached is None:
                model, processor = _load_model(
                    checkpoint, arguments.phase4_run_root, arguments.phase6_run_root
                )
                evidence = _evaluate_checkpoint(
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_hashes[checkpoint],
                    model=model,
                    processor=processor,
                    errors=errors,
                    records=records,
                    dataset_root=arguments.dataset_root,
                    stage1_prompt=prompt,
                    resized_height=_RESIZED_HEIGHT,
                    resized_width=_RESIZED_WIDTH,
                    stage1_max_new_tokens=_STAGE1_MAX_NEW_TOKENS,
                    recovery_max_new_tokens=_RECOVERY_MAX_NEW_TOKENS,
                    operation_max_new_tokens=_OPERATION_MAX_NEW_TOKENS,
                    answer_max_new_tokens=_ANSWER_MAX_NEW_TOKENS,
                    seed=_GENERATION_SEED,
                )
                del model, processor
                _release()
                cached = _cache(
                    arguments.work_root,
                    checkpoint,
                    rows=evidence,
                    checkpoint_sha256=checkpoint_hashes[checkpoint],
                    execution_manifest_sha256=manifest_hash,
                    config_sha256=config_hash,
                    package_lock_sha256=lock_hash,
                )
            else:
                print(f"RESUMED: Phase 7 {checkpoint} trace evidence", flush=True)
            assert cached is not None
            all_chain_rows.extend(
                Phase7ChainRow.from_mapping(row["chain_row"])  # type: ignore[arg-type]
                for row in cached
            )
        rows = validate_phase7_rows(
            tuple(all_chain_rows),
            confirmatory_evaluation_authorized=config.confirmatory_evaluation_authorized,
        )
        summary = {
            **summarize_phase7(
                rows,
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=config.bootstrap_seed,
                tost_margin=config.tost_margin,
            ),
            "config_sha256": config_hash,
            "package_lock_sha256": lock_hash,
            "execution_manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hashes,
            "support_dev_diagnostic": True,
            "confirmatory_evaluation_authorized": config.confirmatory_evaluation_authorized,
            "training_invoked": False,
            "rl_invoked": False,
        }
        write_phase7_outputs(
            output_root=arguments.output_root,
            rows=rows,
            summary=summary,
            source_sha256={"execution_manifest": manifest_hash, **source_hashes},
        )
    except Exception as error:
        print(f"BLOCKED: Phase 7 {error}")
        return 2
    print(f"READY: Phase 7 support_dev multimodal evaluation written below {arguments.output_root}")
    for path in sorted(arguments.output_root.iterdir()):
        print(f"SHA256 {_sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
