"""Evaluate the frozen seven-checkpoint Phase 8 confirmatory chain."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from compensability_v4.qwen.phase5_support import parse_world
from compensability_v4.qwen.phase8_confirm_runtime import (
    PHASE8_LOCKED_PATHS,
    Phase8ConfirmRow,
    build_phase8_execution_manifest,
    load_phase8_config,
    summarize_phase8,
    verify_phase8_package_lock,
    write_phase8_outputs,
)
from compensability_v4.qwen.phase8_execution import (
    PHASE8_CONFIRM_ACK,
    SEVEN_CHECKPOINTS,
    answer_source,
    apply_operation,
    cache_checkpoint_rows,
    chart_operation,
    checkpoint_hashes,
    deterministic_chain_answer_exact,
    final_answer,
    free_generation_answer_exact,
    generate_observation_with_cache,
    image_path,
    load_checkpoint_model,
    load_json,
    load_jsonl,
    load_stage1_prompt,
    observation_error_indices,
    release_model,
    require_ack,
    require_execute,
    require_offline_env,
    revision_or_recovery,
    sha256,
    trace_mismatch,
    validate_phase7_evaluation,
)
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_8.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_8.yaml"
PROMPTS = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
PHASE6_RUN_ROOT = ROOT / "artifacts/v4/rl/runs/phase6-r1"
PHASE7_EVALUATION = ROOT / "artifacts/v4/phase7/evaluation/summary.json"
CONFIRM_ROOT = ROOT / "artifacts/v4/phase8/confirm_data"
WORK_ROOT = ROOT / "artifacts/v4/phase8/work/phase8-r1"
OUTPUT_ROOT = ROOT / "artifacts/v4/phase8/evaluation"
_ACK_LITERAL = "I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET"
_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _image_bundle_sha256(root: Path, scenes: tuple[RecoveryScene, ...]) -> str:
    digest = hashlib.sha256()
    for scene in sorted(scenes, key=lambda item: item.scene_id):
        image = image_path(root, scene.image_path)
        digest.update(scene.scene_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(scene.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(image).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _evaluate_checkpoint(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    model: object,
    processor: object,
    scenes: tuple[RecoveryScene, ...],
    observations: dict[str, NaturalObservation],
    trace_metadata: dict[str, dict[str, object]],
    confirm_root: Path,
    stage1_prompt: str,
    seed: int,
) -> tuple[dict[str, object], ...]:
    evidence_rows: list[dict[str, object]] = []
    for completed, scene in enumerate(scenes, start=1):
        trace = trace_metadata[scene.scene_id]
        image = image_path(confirm_root, scene.image_path)
        stage1 = generate_observation_with_cache(
            model,
            processor,
            str(image),
            stage1_prompt,
            sample_id=f"phase8:{checkpoint}:{scene.scene_id}:{seed}",
            resized_height=scene.resized_height,
            resized_width=scene.resized_width,
            max_new_tokens=32,
            rng_seed=seed,
        )
        stage1_raw = str(stage1["text"])
        stage1_world = parse_world(stage1_raw)
        recovery_raw, recovery_ids, recovered_world = revision_or_recovery(
            model,
            processor,
            observed_raw=stage1_raw,
            facts=tuple(dict(fact) for fact in scene.facts),
            seed=seed,
            max_new_tokens=32,
        )
        operation_raw, operation_ids, chosen_operation = chart_operation(
            model,
            processor,
            question=str(trace["question"]),
            seed=seed,
            max_new_tokens=8,
        )
        answer_raw, answer_ids, answer_value = final_answer(
            model,
            processor,
            recovered_raw=recovery_raw,
            chosen_operation=chosen_operation,
            seed=seed,
            max_new_tokens=8,
        )
        ground_truth_answer = int(trace["ground_truth_answer"])
        ground_truth_operation = str(trace["operation"])
        stage1_exact = stage1_world == scene.truth
        world_exact = recovered_world == scene.truth
        operator_exact = chosen_operation == ground_truth_operation
        free_exact = free_generation_answer_exact(answer_value, ground_truth_answer)
        deterministic_exact, deterministic_value = deterministic_chain_answer_exact(
            recovered_world, chosen_operation, ground_truth_answer
        )
        operator_invariant = bool(
            free_exact
            and recovered_world is not None
            and not world_exact
            and apply_operation(recovered_world, ground_truth_operation) == ground_truth_answer
        )
        error_cancellation = bool(free_exact and not world_exact and not operator_invariant)
        frozen_indices = observation_error_indices(
            truth=scene.truth, observed=observations[scene.scene_id].observed_values
        )
        current_indices = observation_error_indices(truth=scene.truth, observed=stage1_world)
        source = answer_source(
            answer_correct=free_exact,
            world_recovered=bool(not stage1_exact and world_exact),
            operator_invariant=operator_invariant,
            error_cancelled=error_cancellation,
            visual_reread_evidence=stage1_exact,
        )
        row = Phase8ConfirmRow.from_mapping(
            {
                "scene_id": scene.scene_id,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "family": str(trace["family"]),
                "split": scene.split.value,
                "ood_axis": str(trace["ood_axis"]),
                "seed": seed,
                "rollout_id": 0,
                "image_sha256": sha256(image),
                "stage1_visual_exact": stage1_exact,
                "post_revision_world_exact": world_exact,
                "reasoning_operator_exact": operator_exact,
                "final_answer_exact": free_exact,
                "operator_invariant_correct": operator_invariant,
                "genuine_recovery": bool(not stage1_exact and world_exact),
                "error_cancellation": error_cancellation,
                "trace_mismatch": trace_mismatch(
                    free_answer_value=answer_value, deterministic_answer_value=deterministic_value
                ),
                "error_mechanism_shift": current_indices != frozen_indices,
                "free_generation_answer_exact": free_exact,
                "deterministic_chain_answer_exact": deterministic_exact,
                "answer_source": source,
            }
        )
        evidence_rows.append(
            {
                "schema_version": 1,
                "chain_row": row.to_mapping(),
                "family": trace["family"],
                "split": scene.split.value,
                "ood_axis": trace["ood_axis"],
                "question": trace["question"],
                "facts": [dict(fact) for fact in scene.facts],
                "image_sha256": sha256(image),
                "truth": list(scene.truth),
                "frozen_base_observed": list(observations[scene.scene_id].observed_values),
                "frozen_base_error_indices": list(frozen_indices),
                "frozen_base_stage1_raw": trace["stage1_raw_output"],
                "frozen_base_stage1_token_ids": trace["stage1_generated_token_ids"],
                "stage1_raw": stage1_raw,
                "stage1_token_ids": list(stage1["generated_token_ids"]),
                "stage1_parse_success": stage1_world is not None,
                "stage1_world": None if stage1_world is None else list(stage1_world),
                "stage1_error_indices": list(current_indices),
                "revision_or_recovery_raw": recovery_raw,
                "revision_or_recovery_token_ids": list(recovery_ids),
                "revision_or_recovery_parse_success": recovered_world is not None,
                "recovered_world": None if recovered_world is None else list(recovered_world),
                "chart_operation_raw": operation_raw,
                "chart_operation_token_ids": list(operation_ids),
                "chart_operation_parse_success": chosen_operation is not None,
                "chosen_operation": chosen_operation,
                "ground_truth_operation": ground_truth_operation,
                "final_answer_raw": answer_raw,
                "final_answer_token_ids": list(answer_ids),
                "final_answer_parse_success": answer_value is not None,
                "final_answer": answer_value,
                "ground_truth_answer": ground_truth_answer,
                "chosen_operation_execution": deterministic_value,
                "free_generation_answer_exact": free_exact,
                "deterministic_chain_answer_exact": deterministic_exact,
                "answer_source": source,
            }
        )
        if completed % 16 == 0 or completed == len(scenes):
            print(
                f"PROGRESS: Phase 8 {checkpoint} {completed}/{len(scenes)} scenes complete",
                flush=True,
            )
    return tuple(evidence_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument(
        "--execution-manifest", type=Path, default=CONFIRM_ROOT / "execution_manifest.json"
    )
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--confirm-root", type=Path, default=CONFIRM_ROOT)
    parser.add_argument("--prompt-config", type=Path, default=PROMPTS)
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--phase6-run-root", type=Path, default=PHASE6_RUN_ROOT)
    parser.add_argument("--phase7-evaluation", type=Path, default=PHASE7_EVALUATION)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 8 confirmatory evaluation requires explicit --execute.")
        return 2
    if not arguments.execution_manifest_sha256:
        print("BLOCKED: Phase 8 requires --execution-manifest-sha256.")
        return 2
    try:
        require_execute(True, phase="Phase 8", action="confirmatory evaluation")
        require_offline_env(phase="Phase 8")
        _ = _OFFLINE_ENV
        if _ACK_LITERAL != PHASE8_CONFIRM_ACK:
            raise RuntimeError("BLOCKED: Phase 8 ACK literal drifted")
        ack = require_ack(os.environ.get("COMPBIAS_V4_PHASE8_CONFIRM_ACK"), phase="Phase 8")
        config = load_phase8_config(arguments.config)
        lock_hash = verify_phase8_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE8_LOCKED_PATHS,
        )
        config_hash = sha256(arguments.config)
        manifest_hash = sha256(arguments.execution_manifest)
        if manifest_hash != arguments.execution_manifest_sha256:
            raise RuntimeError("Phase 8 execution manifest SHA-256 mismatch")
        manifest = load_json(arguments.execution_manifest, "Phase 8 execution manifest")
        confirm_scenes_path = arguments.confirm_root / "confirm_scenes.jsonl"
        confirm_observations_path = arguments.confirm_root / "confirm_observations.jsonl"
        confirm_summary_path = arguments.confirm_root / "summary.json"
        selection_trace_path = arguments.confirm_root / "selection_trace.jsonl"
        source_hashes = dict(manifest["source_sha256"])  # type: ignore[arg-type]
        phase7_evaluation = load_json(arguments.phase7_evaluation, "Phase 7 frozen evaluation")
        if source_hashes.get("phase7_evaluation") != sha256(arguments.phase7_evaluation):
            raise RuntimeError("Phase 8 Phase 7 evidence hash drifted")
        checkpoint_hash_map = checkpoint_hashes(
            arguments.phase4_run_root, arguments.phase6_run_root
        )
        frozen_phase7_checkpoint_hashes = validate_phase7_evaluation(phase7_evaluation)
        if frozen_phase7_checkpoint_hashes != checkpoint_hash_map:
            raise RuntimeError("Phase 8 checkpoint trees differ from frozen Phase 7 evidence")
        rebuilt_manifest = build_phase8_execution_manifest(
            config=config,
            source_sha256=source_hashes,
            checkpoint_sha256=checkpoint_hash_map,
            config_sha256=config_hash,
            package_lock_sha256=lock_hash,
            authorization_ack=ack,
        )
        if manifest != rebuilt_manifest:
            raise RuntimeError("Phase 8 execution manifest drifted from the frozen contract")
        if source_hashes["confirm_scenes"] != sha256(confirm_scenes_path):
            raise RuntimeError("Phase 8 confirm scenes hash drifted")
        if source_hashes["confirm_observations"] != sha256(confirm_observations_path):
            raise RuntimeError("Phase 8 confirm observations hash drifted")
        if source_hashes["confirm_summary"] != sha256(confirm_summary_path):
            raise RuntimeError("Phase 8 confirm summary hash drifted")
        if source_hashes["prompt_config"] != sha256(arguments.prompt_config):
            raise RuntimeError("Phase 8 prompt config hash drifted")
        summary = load_json(confirm_summary_path, "Phase 8 confirm summary")
        if summary.get("confirmatory_evaluation_authorized") is not True:
            raise RuntimeError("Phase 8 confirm summary authorization drifted")
        scenes = tuple(
            RecoveryScene.from_mapping(row)
            for row in load_jsonl(confirm_scenes_path, "Phase 8 confirm scenes")
        )
        observations = {
            item.scene_id: item
            for item in (
                NaturalObservation.from_mapping(row)
                for row in load_jsonl(confirm_observations_path, "Phase 8 confirm observations")
            )
        }
        trace_metadata = {
            str(row["scene_id"]): row
            for row in load_jsonl(selection_trace_path, "Phase 8 selection trace")
        }
        if set(observations) != {scene.scene_id for scene in scenes} or set(trace_metadata) != set(
            observations
        ):
            raise RuntimeError("Phase 8 confirm-data closure drifted")
        observed_outputs = summary.get("output_sha256")
        if not isinstance(observed_outputs, dict) or observed_outputs != {
            "confirm_scenes": sha256(confirm_scenes_path),
            "confirm_observations": sha256(confirm_observations_path),
            "selection_trace": sha256(selection_trace_path),
            "confirm_image_bundle": _image_bundle_sha256(arguments.confirm_root, scenes),
        }:
            raise RuntimeError("Phase 8 confirm-data output hashes drifted")
        if source_hashes["confirm_image_bundle"] != observed_outputs["confirm_image_bundle"]:
            raise RuntimeError("Phase 8 confirm image bundle hash drifted")
        selected_scene_ids = {
            scene_id
            for scene_id, trace in trace_metadata.items()
            if trace.get("selected_natural_error") is True
        }
        expected_scene_ids = frozenset(selected_scene_ids)
        scenes = tuple(scene for scene in scenes if scene.scene_id in selected_scene_ids)
        if (
            not scenes
            or len(scenes) != summary.get("natural_error_count")
            or {scene.scene_id for scene in scenes} != selected_scene_ids
        ):
            raise RuntimeError("Phase 8 natural-error selection closure drifted")
        prompt = load_stage1_prompt(arguments.prompt_config)
        if arguments.output_root.exists() or arguments.output_root.is_symlink():
            raise FileExistsError("refusing to overwrite Phase 8 outputs")
        if arguments.preflight_only:
            for checkpoint in SEVEN_CHECKPOINTS:
                model, processor = load_checkpoint_model(
                    checkpoint, arguments.phase4_run_root, arguments.phase6_run_root
                )
                del model, processor
                release_model()
                print(f"PREFLIGHT: Phase 8 {checkpoint} load passed", flush=True)
            print("READY: Phase 8 confirmatory preflight passed")
            return 0
        all_rows: list[Phase8ConfirmRow] = []
        for checkpoint in SEVEN_CHECKPOINTS:
            cached = cache_checkpoint_rows(
                root=arguments.work_root,
                checkpoint=checkpoint,
                rows=None,
                expected_scene_ids=expected_scene_ids,
                checkpoint_sha256=checkpoint_hash_map[checkpoint],
                execution_manifest_sha256=manifest_hash,
                config_sha256=config_hash,
                package_lock_sha256=lock_hash,
            )
            if cached is None:
                model, processor = load_checkpoint_model(
                    checkpoint, arguments.phase4_run_root, arguments.phase6_run_root
                )
                cached = cache_checkpoint_rows(
                    root=arguments.work_root,
                    checkpoint=checkpoint,
                    rows=_evaluate_checkpoint(
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_hash_map[checkpoint],
                        model=model,
                        processor=processor,
                        scenes=scenes,
                        observations=observations,
                        trace_metadata=trace_metadata,
                        confirm_root=arguments.confirm_root,
                        stage1_prompt=prompt,
                        seed=config.evaluation_seed,
                    ),
                    expected_scene_ids=expected_scene_ids,
                    checkpoint_sha256=checkpoint_hash_map[checkpoint],
                    execution_manifest_sha256=manifest_hash,
                    config_sha256=config_hash,
                    package_lock_sha256=lock_hash,
                )
                del model, processor
                release_model()
            else:
                print(f"RESUMED: Phase 8 {checkpoint} trace evidence", flush=True)
            assert cached is not None
            all_rows.extend(Phase8ConfirmRow.from_mapping(row["chain_row"]) for row in cached)  # type: ignore[arg-type]
        summary_payload = {
            **summarize_phase8(
                tuple(all_rows),
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=config.bootstrap_seed,
                tost_margin=config.tost_margin,
            ),
            "config_sha256": config_hash,
            "package_lock_sha256": lock_hash,
            "execution_manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash_map,
            "training_invoked": False,
            "rl_invoked": False,
        }
        write_phase8_outputs(
            output_root=arguments.output_root,
            rows=tuple(all_rows),
            summary=summary_payload,
            source_sha256={"execution_manifest": manifest_hash, **source_hashes},
        )
    except Exception as error:
        print(f"BLOCKED: Phase 8 {error}")
        return 2
    print(f"READY: Phase 8 confirmatory evaluation written below {arguments.output_root}")
    for relative in ("per_scene.jsonl", "summary.json"):
        path = arguments.output_root / relative
        print(f"SHA256 {sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
