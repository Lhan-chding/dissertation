"""Execute real teacher-forced Phase-2 candidate scoring on frozen Phase-1 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml
from _guards import (
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
from compensability_v4.qwen.candidate_scoring import find_single_token_labels
from compensability_v4.qwen.model_loader import load_pinned_qwen
from compensability_v4.qwen.phase2_candidate import (
    CueCondition,
    build_candidate_label_evidence,
    build_candidate_scoring_plan,
    execute_candidate_scoring_plan,
    summarize_candidate_scoring,
    validate_phase1_candidate_source,
    write_candidate_scoring_outputs,
)

PROMPT_CONFIG = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"


def _load_prompt_contract(path: Path) -> tuple[str, tuple[str, ...]]:
    if path.is_symlink() or path.resolve() != PROMPT_CONFIG.resolve() or not path.is_file():
        raise RuntimeError(f"prompt config must be the canonical repository file: {PROMPT_CONFIG}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    labels = payload.get("candidate_label_search_order") if isinstance(payload, dict) else None
    prompt = prompts.get("candidate_scoring") if isinstance(prompts, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Phase 2 candidate-scoring prompt is missing")
    if not isinstance(labels, list) or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise RuntimeError("candidate label search order is invalid")
    return prompt, tuple(labels)


def _validate_call_plan(calls: tuple[object, ...], phase_2: dict[str, object]) -> None:
    if len(calls) != phase_2["model_forward_cap"]:
        raise RuntimeError("Phase 2 model-forward plan drifted from the frozen contract")
    conditions = Counter(call.condition.value for call in calls)
    expected_conditions = Counter(
        {condition: phase_2["world_recoverable_scenes"] for condition in phase_2["cue_conditions"]}
    )
    if conditions != expected_conditions:
        raise RuntimeError("Phase 2 cue-condition allocation drifted")
    valid = tuple(call for call in calls if call.condition is CueCondition.VALID_CUE)
    labels = tuple(valid[0].candidate_labels)
    expected_slots = Counter(dict(zip(labels, phase_2["true_label_slot_counts"], strict=True)))
    if Counter(call.true_label for call in valid) != expected_slots:
        raise RuntimeError("Phase 2 true-label slot allocation drifted")


def run_candidate_scoring_cli(
    *,
    phase: str,
    expected_input_sha256: tuple[str, ...],
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
        if len(arguments.input) != 4:
            raise RuntimeError("Phase 2 requires the screen input and all three Phase 1 artifacts")
        config = _load_config(arguments.config)
        validate_runtime_evidence(config["runtime_evidence"])
        phase_2 = config["phase_2_candidate_scoring"]
        if not isinstance(phase_2, dict) or phase_2.get("generation_allowed") is not False:
            raise RuntimeError("Phase 2 candidate-scoring contract is malformed")
        prompt, search_order = _load_prompt_contract(arguments.prompt_config)
        selection = select_legacy_capability_scenes(arguments.input[0])
        scenes = selection.scenes
        if (
            selection.source_eligible_scenes != phase_2["source_scenes"]
            or len(scenes) != phase_2["world_recoverable_scenes"]
            or Counter(scene.family for scene in scenes)
            != Counter(phase_2["included_family_counts"])
        ):
            raise RuntimeError("Phase 2 scene selection drifted from the frozen Phase 1 set")
        validate_phase1_candidate_source(
            scenes,
            per_scene_path=arguments.input[1],
            summary_path=arguments.input[2],
            gaps_path=arguments.input[3],
            expected_source_scenes=int(phase_2["source_scenes"]),
            expected_family_counts=phase_2["included_family_counts"],
        )
        provisional_labels = search_order[:4]
        provisional_calls = build_candidate_scoring_plan(
            scenes,
            prompt=prompt,
            candidate_labels=provisional_labels,
            seed=int(phase_2["seed"]),
        )
        _validate_call_plan(provisional_calls, phase_2)
        model, processor = load_pinned_qwen(model_path=arguments.model_path)
        tokenizer = getattr(processor, "tokenizer", processor)
        labels = find_single_token_labels(tokenizer, search_order, minimum=4)[:4]
        label_evidence = build_candidate_label_evidence(
            tokenizer,
            labels,
            model_snapshot_sha256=validation.model_snapshot_sha256,
        )
        calls = build_candidate_scoring_plan(
            scenes,
            prompt=prompt,
            candidate_labels=labels,
            seed=int(phase_2["seed"]),
        )
        _validate_call_plan(calls, phase_2)

        def report_progress(completed: int, total: int) -> None:
            if completed == total or completed % 50 == 0:
                print(f"PROGRESS: {completed}/{total} Phase 2 forwards complete", flush=True)

        records = execute_candidate_scoring_plan(
            model,
            processor,
            calls,
            progress=report_progress,
        )
        if len(records) != phase_2["model_forward_cap"]:
            raise RuntimeError("Phase 2 executed forward count drifted")
        summary = summarize_candidate_scoring(
            records,
            bootstrap_resamples=int(phase_2["bootstrap_resamples"]),
        )
        provenance = {
            "config_sha256": validation.config_sha256,
            "package_lock_sha256": validation.package_lock_sha256,
            "model_snapshot_sha256": validation.model_snapshot_sha256,
            "hash_bound_inputs": list(validation.inputs),
            "phase_1_revision": phase_2["phase_1_revision"],
            "phase_1_config_sha256": phase_2["phase_1_config_sha256"],
            "phase_1_package_lock_sha256": phase_2["phase_1_package_lock_sha256"],
        }
        label_evidence = {**label_evidence, **provenance}
        summary = {**summary, **provenance}
        paths = {name: Path(path) for name, path in output_paths.items()}
        if set(paths) != {"labels", "per_scene", "summary"}:
            raise RuntimeError(
                "Phase 2 output paths differ from the frozen three-artifact contract"
            )
        write_candidate_scoring_outputs(
            labels_path=paths["labels"],
            records_path=paths["per_scene"],
            summary_path=paths["summary"],
            label_evidence=label_evidence,
            records=records,
            summary=summary,
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: {phase} outputs written under {Path(output_paths['per_scene']).parent}")
    return 0


def main() -> int:
    return run_candidate_scoring_cli(
        phase="phase_2_candidate_scoring",
        expected_input_sha256=(
            LEGACY_SCREEN_RECORDS_SHA256,
            CAPABILITY_PER_SCENE_SHA256,
            CAPABILITY_SUMMARY_SHA256,
            CAPABILITY_PAIRED_GAPS_SHA256,
        ),
        output_paths={
            "labels": str(ROOT / "artifacts/v4/tokenizer/candidate_labels.json"),
            "per_scene": str(ROOT / "artifacts/v4/candidate_scoring/per_scene.jsonl"),
            "summary": str(ROOT / "artifacts/v4/candidate_scoring/summary.json"),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
