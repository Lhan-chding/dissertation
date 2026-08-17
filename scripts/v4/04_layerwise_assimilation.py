"""Execute real Phase-2 layerwise constraint assimilation on frozen S3 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml
from _guards import (
    CANDIDATE_LABELS_SHA256,
    CANDIDATE_SCORES_SHA256,
    CANDIDATE_SUMMARY_SHA256,
    CAPABILITY_PAIRED_GAPS_SHA256,
    CAPABILITY_PER_SCENE_SHA256,
    CAPABILITY_SUMMARY_SHA256,
    CONFIG_PATH,
    LEGACY_SCREEN_RECORDS_SHA256,
    PACKAGE_LOCK_PATH,
    ROOT,
    _load_config,
    blocked_unless_execute,
    validate_runtime_evidence,
    validate_server_inputs,
)

from compensability_v4.diagnostics.capability_chain import select_legacy_capability_scenes
from compensability_v4.qwen.model_loader import load_pinned_qwen
from compensability_v4.qwen.phase2_candidate import (
    build_candidate_label_evidence,
    build_candidate_scoring_plan,
    validate_phase1_candidate_source,
)
from compensability_v4.qwen.phase2_layerwise import (
    build_layerwise_plan,
    execute_layerwise_plan,
    load_candidate_label_evidence,
    load_candidate_scoring_records,
    summarize_layerwise_records,
    validate_candidate_scoring_summary,
    write_layerwise_outputs,
)

PROMPT_CONFIG = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"


def _load_prompt_contract(path: Path) -> str:
    if path.is_symlink() or path.resolve() != PROMPT_CONFIG.resolve() or not path.is_file():
        raise RuntimeError(f"prompt config must be the canonical repository file: {PROMPT_CONFIG}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    prompt = prompts.get("candidate_scoring") if isinstance(prompts, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Phase 2 candidate-scoring prompt is missing")
    return prompt


def _output_paths(paths: dict[str, str]) -> dict[str, Path]:
    if set(paths) != {"per_scene", "summary"}:
        raise RuntimeError("S4 output paths differ from the frozen two-artifact contract")
    result = {name: Path(value) for name, value in paths.items()}
    if result["per_scene"].parent != result["summary"].parent:
        raise RuntimeError("S4 outputs must share the frozen artifact directory")
    if any(path.exists() or path.is_symlink() for path in result.values()):
        raise FileExistsError("refusing to overwrite an S4 layerwise artifact")
    return result


def run_layerwise_assimilation_cli(
    *,
    phase: str,
    expected_input_sha256: tuple[str, ...],
    expected_scenes: int,
    expected_conditions: int,
    expected_language_layers: int,
    output_paths: dict[str, str],
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--package-lock", type=Path, default=PACKAGE_LOCK_PATH)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--prompt-config", type=Path, default=PROMPT_CONFIG)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-sha256", action="append", default=[])
    arguments = parser.parse_args()
    if blocked_unless_execute(arguments.execute):
        return 2
    try:
        validation = validate_server_inputs(
            config=arguments.config,
            package_lock=arguments.package_lock,
            model_path=arguments.model_path,
            inputs=arguments.input,
            input_sha256=arguments.input_sha256,
            expected_input_sha256=expected_input_sha256,
            require_raw_evidence=True,
        )
        if len(arguments.input) != 7:
            raise RuntimeError("S4 requires the screen, all S2 artifacts, and all S3 artifacts")
        config = _load_config(arguments.config)
        validate_runtime_evidence(config["runtime_evidence"])
        contract = config["phase_2_layerwise_assimilation"]
        if (
            not isinstance(contract, dict)
            or contract.get("generation_allowed") is not False
            or contract.get("world_recoverable_scenes") != expected_scenes
            or len(contract.get("cue_conditions", ())) != expected_conditions
            or contract.get("language_layers") != expected_language_layers
        ):
            raise RuntimeError("S4 execution contract is malformed")
        paths = _output_paths(output_paths)
        selection = select_legacy_capability_scenes(arguments.input[0])
        scenes = selection.scenes
        if (
            selection.source_eligible_scenes != contract["source_scenes"]
            or len(scenes) != expected_scenes
            or Counter(scene.family for scene in scenes)
            != Counter(contract["included_family_counts"])
        ):
            raise RuntimeError("S4 scene selection drifted from the frozen S2/S3 set")
        validate_phase1_candidate_source(
            scenes,
            per_scene_path=arguments.input[1],
            summary_path=arguments.input[2],
            gaps_path=arguments.input[3],
            expected_source_scenes=int(contract["source_scenes"]),
            expected_family_counts=contract["included_family_counts"],
        )
        labels, label_token_ids = load_candidate_label_evidence(
            arguments.input[4],
            expected_model_snapshot_sha256=validation.model_snapshot_sha256,
            expected_config_sha256=str(contract["phase_2_config_sha256"]),
            expected_package_lock_sha256=str(contract["phase_2_package_lock_sha256"]),
        )
        scoring_records = load_candidate_scoring_records(arguments.input[5])
        validate_candidate_scoring_summary(
            arguments.input[6],
            expected_scenes=expected_scenes,
            expected_forward_calls=int(contract["model_forward_cap"]),
            expected_family_counts=contract["included_family_counts"],
            expected_model_snapshot_sha256=validation.model_snapshot_sha256,
            expected_config_sha256=str(contract["phase_2_config_sha256"]),
            expected_package_lock_sha256=str(contract["phase_2_package_lock_sha256"]),
        )
        prompt = _load_prompt_contract(arguments.prompt_config)
        scoring_calls = build_candidate_scoring_plan(
            scenes,
            prompt=prompt,
            candidate_labels=labels,
            seed=int(config["phase_2_candidate_scoring"]["seed"]),
        )
        calls = build_layerwise_plan(
            scoring_calls,
            scoring_records,
            expected_scenes=expected_scenes,
            expected_language_layers=expected_language_layers,
        )
        if len(calls) != contract["model_forward_cap"]:
            raise RuntimeError("S4 model-forward plan drifted from the frozen contract")
        model, processor = load_pinned_qwen(model_path=arguments.model_path)
        tokenizer = getattr(processor, "tokenizer", processor)
        observed_labels = build_candidate_label_evidence(
            tokenizer,
            labels,
            model_snapshot_sha256=validation.model_snapshot_sha256,
        )["labels"]
        expected_labels = [{"label": label, "token_id": label_token_ids[label]} for label in labels]
        if observed_labels != expected_labels:
            raise RuntimeError("runtime tokenizer differs from the frozen S3 label mapping")

        def report_progress(completed: int, total: int) -> None:
            if completed == total or completed % 50 == 0:
                print(f"PROGRESS: {completed}/{total} S4 forwards complete", flush=True)

        records = execute_layerwise_plan(
            model,
            processor,
            calls,
            label_token_ids=label_token_ids,
            final_logit_absolute_tolerance=float(contract["final_logit_absolute_tolerance"]),
            final_logit_relative_tolerance=float(contract["final_logit_relative_tolerance"]),
            numerical_equality_tolerance=float(contract["numerical_equality_tolerance"]),
            progress=report_progress,
        )
        if len(records) != contract["model_forward_cap"]:
            raise RuntimeError("S4 executed forward count drifted")
        summary = summarize_layerwise_records(
            records,
            bootstrap_resamples=int(contract["bootstrap_resamples"]),
        )
        provenance = {
            "config_sha256": validation.config_sha256,
            "package_lock_sha256": validation.package_lock_sha256,
            "model_snapshot_sha256": validation.model_snapshot_sha256,
            "hash_bound_inputs": list(validation.inputs),
            "phase_2_revision": contract["phase_2_revision"],
            "phase_2_config_sha256": contract["phase_2_config_sha256"],
            "phase_2_package_lock_sha256": contract["phase_2_package_lock_sha256"],
        }
        summary = {**summary, **provenance}
        write_layerwise_outputs(
            records_path=paths["per_scene"],
            summary_path=paths["summary"],
            records=records,
            summary=summary,
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: {phase} outputs written under {paths['per_scene'].parent}")
    return 0


def main() -> int:
    return run_layerwise_assimilation_cli(
        phase="phase_2_layerwise_assimilation",
        expected_input_sha256=(
            LEGACY_SCREEN_RECORDS_SHA256,
            CAPABILITY_PER_SCENE_SHA256,
            CAPABILITY_SUMMARY_SHA256,
            CAPABILITY_PAIRED_GAPS_SHA256,
            CANDIDATE_LABELS_SHA256,
            CANDIDATE_SCORES_SHA256,
            CANDIDATE_SUMMARY_SHA256,
        ),
        expected_scenes=579,
        expected_conditions=4,
        expected_language_layers=36,
        output_paths={
            "per_scene": str(ROOT / "artifacts/v4/layerwise_assimilation/per_scene.jsonl"),
            "summary": str(ROOT / "artifacts/v4/layerwise_assimilation/summary.json"),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
