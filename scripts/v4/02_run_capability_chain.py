"""Execute the real T1-T6 capability chain on frozen eligible screen records."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from _guards import (
    CONFIG_PATH,
    LEGACY_SCREEN_RECORDS_SHA256,
    PACKAGE_LOCK_PATH,
    ROOT,
    _load_config,
    blocked_unless_execute,
    validate_runtime_evidence,
    validate_server_inputs,
)

from compensability_v4.diagnostics.capability_chain import (
    build_capability_calls,
    load_legacy_capability_scenes,
    summarize_capability_run,
)
from compensability_v4.qwen.candidate_scoring import find_single_token_labels
from compensability_v4.qwen.capability_runner import (
    execute_capability_calls,
    write_capability_outputs,
)
from compensability_v4.qwen.model_loader import load_pinned_qwen

PROMPT_CONFIG = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"


def _load_prompt_contract(path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    if path.is_symlink() or path.resolve() != PROMPT_CONFIG.resolve() or not path.is_file():
        raise RuntimeError(f"prompt config must be the canonical repository file: {PROMPT_CONFIG}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    labels = payload.get("candidate_label_search_order") if isinstance(payload, dict) else None
    if not isinstance(prompts, dict) or not isinstance(labels, list):
        raise RuntimeError("phase prompt contract is malformed")
    required = {"T1", "T2", "T3", "T4", "T5", "T6"}
    if set(prompts) & required != required:
        raise RuntimeError("phase prompt contract is missing T1-T6 prompts")
    if any(not isinstance(value, str) or not value.strip() for value in prompts.values()):
        raise RuntimeError("phase prompt contract contains empty prompt text")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise RuntimeError("candidate label search order is invalid")
    return {name: prompts[name] for name in required}, tuple(labels)


def run_capability_chain_cli(
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
        config_payload = _load_config(arguments.config)
        if len(arguments.input) != 1:
            raise RuntimeError("Phase 1 requires exactly one screen_records.jsonl input")
        validate_runtime_evidence(config_payload["runtime_evidence"])
        phase_1 = config_payload["phase_1_capability_chain"]
        if not isinstance(phase_1, dict):
            raise RuntimeError("Phase 1 execution contract is malformed")
        prompts, search_order = _load_prompt_contract(arguments.prompt_config)
        scenes = load_legacy_capability_scenes(arguments.input[0])
        model, processor = load_pinned_qwen(model_path=arguments.model_path)
        tokenizer = getattr(processor, "tokenizer", processor)
        labels = find_single_token_labels(tokenizer, search_order, minimum=4)[:4]
        calls = build_capability_calls(
            scenes,
            prompts=prompts,
            candidate_labels=labels,
            seed=int(phase_1["seed"]),
        )
        if len(calls) != phase_1["model_call_cap"]:
            raise RuntimeError("Phase 1 model-call plan drifted from the frozen contract")

        def report_progress(completed: int, total: int) -> None:
            if completed == total or completed % 50 == 0:
                print(f"PROGRESS: {completed}/{total} Phase 1 calls complete", flush=True)

        records = execute_capability_calls(
            model,
            processor,
            calls,
            max_new_tokens=int(phase_1["max_new_tokens"]),
            progress=report_progress,
        )
        if len(records) != phase_1["model_call_cap"]:
            raise RuntimeError("Phase 1 model-call count drifted from the frozen contract")
        summaries, gaps = summarize_capability_run(
            records, bootstrap_resamples=int(phase_1["bootstrap_resamples"])
        )
        gaps.update(
            {
                "schema_version": 1,
                "status": "PHASE_1_EXECUTED",
                "source_scenes": len(scenes),
                "model_calls": len(records),
                "candidate_labels": list(labels),
                "do_sample": False,
                "max_new_tokens": phase_1["max_new_tokens"],
                "format_retries": 0,
                "training_invoked": False,
                "rl_invoked": False,
                "subjective_success_threshold_applied": False,
                "T5_establishes_full_recovery": False,
                "config_sha256": validation.config_sha256,
                "package_lock_sha256": validation.package_lock_sha256,
                "model_snapshot_sha256": validation.model_snapshot_sha256,
                "hash_bound_inputs": list(validation.inputs),
            }
        )
        paths = {name: Path(path) for name, path in output_paths.items()}
        parents = {path.parent for path in paths.values()}
        expected_names = {
            "per_scene": "per_scene.csv",
            "summary_by_family": "summary_by_family.csv",
            "paired_gaps": "paired_gaps.json",
        }
        names_differ = any(
            paths[name].name != filename for name, filename in expected_names.items()
        )
        if len(parents) != 1 or names_differ:
            raise RuntimeError(
                "Phase 1 output paths differ from the frozen three-artifact contract"
            )
        write_capability_outputs(parents.pop(), records=records, summaries=summaries, gaps=gaps)
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: {phase} outputs written under {Path(output_paths['per_scene']).parent}")
    return 0


def main() -> int:
    return run_capability_chain_cli(
        phase="phase_1_capability_chain",
        expected_input_sha256=(LEGACY_SCREEN_RECORDS_SHA256,),
        output_paths={
            "per_scene": str(ROOT / "artifacts/v4/capability_chain/per_scene.csv"),
            "summary_by_family": str(ROOT / "artifacts/v4/capability_chain/summary_by_family.csv"),
            "paired_gaps": str(ROOT / "artifacts/v4/capability_chain/paired_gaps.json"),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
