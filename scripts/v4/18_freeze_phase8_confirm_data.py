"""Freeze the full Phase 8 confirmatory candidate set and execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from compbias.recoverability.phase_c_screen import build_family_constraints
from compensability_v4.qwen.model_loader import MODEL_SNAPSHOT_SHA256
from compensability_v4.qwen.phase5_support import parse_world
from compensability_v4.qwen.phase8_confirm_runtime import (
    PHASE8_LOCKED_PATHS,
    build_phase8_execution_manifest,
    freeze_phase8_natural_errors,
    load_phase8_config,
    validate_phase8_isolation,
    verify_phase8_package_lock,
)
from compensability_v4.qwen.phase8_execution import (
    PHASE8_CONFIRM_ACK,
    SOURCE_NAMES,
    build_scene,
    checkpoint_hashes,
    constraint_ood_facts,
    generate_observation_with_cache,
    load_checkpoint_model,
    load_json,
    load_jsonl,
    load_stage1_prompt,
    load_support_dev_scenes,
    load_symbolic_or_natural_scenes,
    observation_error_indices,
    parse_named_bindings,
    release_model,
    render_phase8_image,
    require_ack,
    require_execute,
    require_matching_hashes,
    require_offline_env,
    select_confirm_templates,
    sha256,
    validate_phase7_evaluation,
)
from compensability_v4.schemas.observation import NaturalObservation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/recoverability/v4_phase_8.yaml"
LOCK = ROOT / "configs/recoverability/v4/server_package_lock_phase_8.yaml"
PROMPTS = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
PHASE4_RUN_ROOT = ROOT / "artifacts/v4/training/runs/phase4-r1"
PHASE6_RUN_ROOT = ROOT / "artifacts/v4/rl/runs/phase6-r1"
OUTPUT_ROOT = ROOT / "artifacts/v4/phase8/confirm_data"
DEFAULT_INPUTS = {
    "legacy_diagnostic": ROOT / "data/generated/cva_recoverability_causal_v2_screen/records.jsonl",
    "symbolic_support_train": ROOT / "artifacts/v4/training/sources/symbolic_scenes.jsonl",
    "natural_error_support_train": ROOT / "artifacts/v4/training/sources/natural_scenes.jsonl",
    "support_dev": ROOT / "artifacts/v4/support_dev/held_out_natural_errors.jsonl",
    "phase7_evaluation": ROOT / "artifacts/v4/phase7/evaluation/summary.json",
}
_ACK_LITERAL = "I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET"
_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _legacy_truths(path: Path) -> set[tuple[int, int, int, int]]:
    truths: set[tuple[int, int, int, int]] = set()
    for row in load_jsonl(path, "Phase 8 legacy diagnostic records"):
        values = row.get("values", row.get("truth"))
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(type(value) is not int for value in values)
        ):
            raise RuntimeError("Phase 8 legacy numeric tables are malformed")
        truths.add(tuple(values))  # type: ignore[arg-type]
    return truths


def _image_bundle_sha256(root: Path, scenes: tuple[object, ...]) -> str:
    digest = hashlib.sha256()
    for scene in sorted(scenes, key=lambda item: item.scene_id):
        image = root / scene.image_path
        digest.update(scene.scene_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(scene.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(image).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument(
        "--input",
        action="append",
        help=(
            "hash-bound NAME=PATH for legacy_diagnostic, symbolic_support_train, "
            "natural_error_support_train, support_dev, and phase7_evaluation"
        ),
    )
    parser.add_argument("--input-sha256", action="append")
    parser.add_argument("--prompt-config", type=Path, default=PROMPTS)
    parser.add_argument("--phase4-run-root", type=Path, default=PHASE4_RUN_ROOT)
    parser.add_argument("--phase6-run-root", type=Path, default=PHASE6_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if not arguments.execute:
        print("BLOCKED: Phase 8 confirm freeze requires explicit --execute.")
        return 2
    try:
        require_execute(True, phase="Phase 8", action="confirm freeze")
        require_offline_env(phase="Phase 8")
        _ = _OFFLINE_ENV  # keep literal offline env names in source for fail-closed inspection
        if _ACK_LITERAL != PHASE8_CONFIRM_ACK:
            raise RuntimeError("BLOCKED: Phase 8 ACK literal drifted")
        ack = require_ack(os.environ.get("COMPBIAS_V4_PHASE8_CONFIRM_ACK"), phase="Phase 8")
        input_values = parse_named_bindings(
            arguments.input, option="--input", expected_names=SOURCE_NAMES
        )
        input_hashes = parse_named_bindings(
            arguments.input_sha256, option="--input-sha256", expected_names=SOURCE_NAMES
        )
        input_paths = {name: Path(value).resolve() for name, value in input_values.items()}
        observed_inputs = require_matching_hashes(
            input_paths, expected_sha256=input_hashes, phase="Phase 8"
        )
        config = load_phase8_config(arguments.config)
        lock_hash = verify_phase8_package_lock(
            lock_path=arguments.package_lock,
            repository_root=ROOT,
            expected_paths=PHASE8_LOCKED_PATHS,
        )
        prompt = load_stage1_prompt(arguments.prompt_config)
        symbolic_scenes = load_symbolic_or_natural_scenes(
            input_paths["symbolic_support_train"], "Phase 8 symbolic support scenes"
        )
        natural_scenes = load_symbolic_or_natural_scenes(
            input_paths["natural_error_support_train"], "Phase 8 natural support scenes"
        )
        support_dev_scenes = load_support_dev_scenes(input_paths["support_dev"])
        reserved_truths = _legacy_truths(input_paths["legacy_diagnostic"])
        reserved_truths.update(
            scene.truth for scene in (*symbolic_scenes, *natural_scenes, *support_dev_scenes)
        )
        scenes = []
        render_queue: list[tuple[object, object, object]] = []
        trace_rows: list[dict[str, object]] = []
        axis_offsets = {
            "iid": config.generation_seed,
            "style_ood": config.generation_seed + 101,
            "constraint_graph_ood": config.generation_seed + 202,
            "error_mechanism_ood": config.generation_seed + 303,
        }
        for axis in ("iid", "style_ood", "constraint_graph_ood", "error_mechanism_ood"):
            templates = select_confirm_templates(
                count=config.fixed_scene_counts[axis],
                seed=axis_offsets[axis],
                reserved_truths=reserved_truths,
            )
            reserved_truths.update(template.truth for template in templates)
            for index, template in enumerate(templates):
                base_facts = tuple(
                    dict(fact) for fact in build_family_constraints(template.family, template.truth)
                )
                facts = (
                    constraint_ood_facts(template) if axis == "constraint_graph_ood" else base_facts
                )
                scene = build_scene(template=template, axis=axis, index=index, facts=facts)
                scenes.append(scene)
                render_queue.append((scene, template, axis))
                trace_rows.append(
                    {
                        "scene_id": scene.scene_id,
                        "source_scene_id": template.source_scene_id,
                        "family": template.family,
                        "split": scene.split.value,
                        "ood_axis": axis,
                        "question": template.question,
                        "operation": template.operation,
                        "ground_truth_answer": template.answer,
                        "all_natural_stage1_errors_included": True,
                        "selection_uses_model_outcome_threshold": False,
                    }
                )
        confirm_scenes = tuple(scenes)
        prior_scenes = (*symbolic_scenes, *natural_scenes, *support_dev_scenes)
        validate_phase8_isolation(confirm_scenes, prior_scenes)
        if arguments.output_root.exists() or arguments.output_root.is_symlink():
            raise FileExistsError("refusing to overwrite Phase 8 confirm freeze")
        arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{arguments.output_root.name}.", dir=str(arguments.output_root.parent)
            )
        )
        try:
            for scene, template, axis in render_queue:
                render_phase8_image(scene=scene, template=template, axis=axis, output_root=staging)
            model, processor = load_checkpoint_model(
                "Base", arguments.phase4_run_root, arguments.phase6_run_root
            )
            observations: list[NaturalObservation] = []
            for completed, scene in enumerate(confirm_scenes, start=1):
                image = staging / scene.image_path
                evidence = generate_observation_with_cache(
                    model,
                    processor,
                    str(image),
                    prompt,
                    sample_id=f"phase8:freeze:{scene.scene_id}",
                    resized_height=scene.resized_height,
                    resized_width=scene.resized_width,
                    max_new_tokens=32,
                    rng_seed=config.generation_seed,
                )
                parsed_world = parse_world(str(evidence["text"]))
                if parsed_world is None:
                    raise RuntimeError(
                        f"Phase 8 Stage-1 output is unparseable for {scene.scene_id}"
                    )
                differences = observation_error_indices(truth=scene.truth, observed=parsed_world)
                error_index = differences[0] if differences else 0
                state = evidence["state"]
                observations.append(
                    NaturalObservation(
                        observation_id=f"phase8-observation-{scene.scene_id}",
                        scene_id=scene.scene_id,
                        observed_values=parsed_world,
                        error_index=error_index,
                        stage1_model_hash=MODEL_SNAPSHOT_SHA256,
                        image_grid_thw=state.image_grid_thw,
                        visual_token_count=state.visual_token_count,
                    )
                )
                trace_rows[completed - 1]["stage1_raw_output"] = str(evidence["text"])
                trace_rows[completed - 1]["stage1_generated_token_ids"] = list(
                    evidence["generated_token_ids"]
                )
                trace_rows[completed - 1]["stage1_parsed_world"] = list(parsed_world)
                trace_rows[completed - 1]["stage1_error_indices"] = list(differences)
                trace_rows[completed - 1]["selected_natural_error"] = bool(differences)
                trace_rows[completed - 1]["image_sha256"] = sha256(image)
                trace_rows[completed - 1]["image_grid_thw"] = list(state.image_grid_thw)
                trace_rows[completed - 1]["visual_token_count"] = state.visual_token_count
                if completed % 16 == 0 or completed == len(confirm_scenes):
                    print(
                        "PROGRESS: Phase 8 freeze Stage-1 "
                        f"{completed}/{len(confirm_scenes)} scenes complete",
                        flush=True,
                    )
            del model, processor
            release_model()
            frozen = freeze_phase8_natural_errors(
                confirm_scenes,
                tuple(observations),
                fixed_scene_counts=config.fixed_scene_counts,
            )
            selected_ids = {
                str(row["scene_id"]) for row in trace_rows if row["selected_natural_error"] is True
            }
            if selected_ids != {example.scene_id for example in frozen.examples}:
                raise RuntimeError("Phase 8 natural-error freeze lost a Stage-1 error")
            family_by_scene = {str(row["scene_id"]): str(row["family"]) for row in trace_rows}
            natural_by_axis = Counter(example.ood_axis for example in frozen.examples)
            natural_by_family = Counter(
                family_by_scene[example.scene_id] for example in frozen.examples
            )
            scenes_text = "".join(
                json.dumps(scene.to_mapping(), sort_keys=True, allow_nan=False) + "\n"
                for scene in confirm_scenes
            )
            observations_text = "".join(
                json.dumps(observation.to_mapping(), sort_keys=True, allow_nan=False) + "\n"
                for observation in observations
            )
            trace_text = "".join(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in trace_rows
            )
            (staging / "confirm_scenes.jsonl").write_text(scenes_text, encoding="utf-8")
            (staging / "confirm_observations.jsonl").write_text(observations_text, encoding="utf-8")
            (staging / "selection_trace.jsonl").write_text(trace_text, encoding="utf-8")
            output_hashes = {
                "confirm_scenes": sha256(staging / "confirm_scenes.jsonl"),
                "confirm_observations": sha256(staging / "confirm_observations.jsonl"),
                "selection_trace": sha256(staging / "selection_trace.jsonl"),
                "confirm_image_bundle": _image_bundle_sha256(staging, confirm_scenes),
            }
            phase7_evaluation = load_json(
                input_paths["phase7_evaluation"], "Phase 7 frozen evaluation"
            )
            frozen_phase7_checkpoint_hashes = validate_phase7_evaluation(phase7_evaluation)
            checkpoint_hash_map = checkpoint_hashes(
                arguments.phase4_run_root, arguments.phase6_run_root
            )
            if frozen_phase7_checkpoint_hashes != checkpoint_hash_map:
                raise RuntimeError("Phase 8 checkpoint hashes drift from frozen Phase 7 evidence")
            summary = {
                "schema_version": 1,
                "status": "PHASE_8_CONFIRM_DATA_FROZEN",
                "confirm_iid": config.fixed_scene_counts["iid"],
                "confirm_style_ood": config.fixed_scene_counts["style_ood"],
                "confirm_constraint_ood": config.fixed_scene_counts["constraint_graph_ood"],
                "confirm_error_mechanism_ood": config.fixed_scene_counts["error_mechanism_ood"],
                "candidate_scene_count": len(confirm_scenes),
                "natural_error_count": frozen.natural_error_count,
                "natural_error_count_by_ood_axis": dict(sorted(natural_by_axis.items())),
                "natural_error_count_by_family": dict(sorted(natural_by_family.items())),
                "all_natural_stage1_errors_included": frozen.all_natural_stage1_errors_included,
                "selection_uses_model_outcome_threshold": (
                    frozen.selection_uses_model_outcome_threshold
                ),
                "contains_confirmatory_data": True,
                "confirmatory_evaluation_authorized": config.confirmatory_evaluation_authorized,
                "subjective_success_threshold_applied": False,
                "source_sha256": dict(sorted(observed_inputs.items())),
                "source_paths": {name: str(path) for name, path in sorted(input_paths.items())},
                "output_sha256": output_hashes,
                "config_sha256": sha256(arguments.config),
                "package_lock_sha256": lock_hash,
                "training_invoked": False,
                "rl_invoked": False,
            }
            (staging / "summary.json").write_text(
                json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            summary_hash = sha256(staging / "summary.json")
            manifest = build_phase8_execution_manifest(
                config=config,
                source_sha256={
                    **dict(sorted(observed_inputs.items())),
                    "confirm_scenes": output_hashes["confirm_scenes"],
                    "confirm_observations": output_hashes["confirm_observations"],
                    "confirm_summary": summary_hash,
                    "confirm_image_bundle": output_hashes["confirm_image_bundle"],
                    "prompt_config": sha256(arguments.prompt_config),
                },
                checkpoint_sha256=checkpoint_hash_map,
                config_sha256=summary["config_sha256"],
                package_lock_sha256=lock_hash,
                authorization_ack=ack,
            )
            (staging / "execution_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if arguments.output_root.exists() or arguments.output_root.is_symlink():
                raise FileExistsError("refusing to overwrite Phase 8 confirm freeze")
            arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(arguments.output_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except Exception as error:
        print(f"BLOCKED: Phase 8 {error}")
        return 2
    print(f"READY: Phase 8 confirm data written below {arguments.output_root}")
    for relative in (
        "confirm_scenes.jsonl",
        "confirm_observations.jsonl",
        "selection_trace.jsonl",
        "summary.json",
        "execution_manifest.json",
    ):
        path = arguments.output_root / relative
        print(f"SHA256 {sha256(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
