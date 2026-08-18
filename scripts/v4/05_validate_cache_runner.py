"""Execute S5 exact-cache/full-history parity on frozen S4 visual scenes."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _guards import (  # noqa: E402
    CONFIG_PATH,
    LAYERWISE_PROFILES_SHA256,
    LAYERWISE_SUMMARY_SHA256,
    PACKAGE_LOCK_PATH,
    PHASE_C_DATASET_MANIFEST_SHA256,
    PHASE_C_DATASET_RECORDS_SHA256,
    ROOT,
    _load_config,
    blocked_unless_execute,
    sha256,
    validate_runtime_evidence,
    validate_server_inputs,
)

from compensability_v4.qwen.manual_generation import generate_observation_with_cache  # noqa: E402
from compensability_v4.qwen.model_loader import load_pinned_qwen  # noqa: E402
from compensability_v4.qwen.phase2_candidate import CueCondition  # noqa: E402
from compensability_v4.qwen.phase3_cache import (  # noqa: E402
    build_cache_parity_plan,
    build_condition_turns,
    execute_cache_parity_plan,
    summarize_cache_parity,
    validate_cache_output_path,
    write_cache_parity_outputs,
)

PROMPT_CONFIG = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
DEFAULT_DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"


@dataclass(frozen=True, slots=True)
class _SceneContract:
    scene_id: str
    family: str
    truth: tuple[int, int, int, int]
    observed: tuple[int, int, int, int]
    counterfactual: tuple[int, int, int, int]
    image_path: Path


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"S5 {label} must contain four integers")
    return tuple(value)  # type: ignore[return-value]


def _load_prompts(path: Path) -> tuple[str, str]:
    if path.is_symlink() or path.resolve() != PROMPT_CONFIG.resolve() or not path.is_file():
        raise RuntimeError(f"prompt config must be the canonical repository file: {PROMPT_CONFIG}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    observation = prompts.get("stage_1_observation") if isinstance(prompts, dict) else None
    correction = prompts.get("correction_suffix") if isinstance(prompts, dict) else None
    if any(not isinstance(item, str) or not item.strip() for item in (observation, correction)):
        raise RuntimeError("S5 observation/correction prompts are missing")
    return observation, correction


def _safe_image(root: Path, relative: object) -> tuple[str, Path]:
    if not isinstance(relative, str):
        raise RuntimeError("Phase C image path must be a string")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or posix.suffix.lower() != ".png":
        raise RuntimeError("Phase C image path is not a safe relative PNG path")
    resolved = (root / Path(*posix.parts)).resolve()
    if root not in resolved.parents or resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("Phase C image path escaped or is missing")
    return relative, resolved


def _image_bundle_sha256(rows: tuple[dict[str, object], ...], root: Path) -> str:
    digest = hashlib.sha256()
    images = [_safe_image(root, row.get("image")) for row in rows]
    if len({relative for relative, _path in images}) != len(images):
        raise RuntimeError("Phase C image paths are not unique")
    for relative, path in sorted(images):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_dataset(dataset_root: Path) -> dict[str, tuple[str, tuple[int, int, int, int], Path]]:
    if not dataset_root.is_absolute() or dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RuntimeError("S5 Phase C dataset root must be an absolute regular directory")
    root = dataset_root.resolve()
    manifest_path, records_path = root / "manifest.json", root / "records.jsonl"
    if (
        manifest_path.is_symlink()
        or records_path.is_symlink()
        or sha256(manifest_path) != PHASE_C_DATASET_MANIFEST_SHA256
        or sha256(records_path) != PHASE_C_DATASET_RECORDS_SHA256
    ):
        raise RuntimeError("S5 Phase C dataset manifest/records SHA-256 drifted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = tuple(
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("record_count") != 8000
        or manifest.get("records_sha256") != PHASE_C_DATASET_RECORDS_SHA256
        or len(rows) != 8000
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise RuntimeError("S5 Phase C dataset structure drifted")
    if manifest.get("images_sha256") != _image_bundle_sha256(rows, root):
        raise RuntimeError("S5 Phase C image bundle SHA-256 drifted")
    result: dict[str, tuple[str, tuple[int, int, int, int], Path]] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        family = row.get("family")
        if not isinstance(scene_id, str) or family not in {
            "cross_series",
            "duplicate_encoding",
            "trend",
        }:
            raise RuntimeError("S5 Phase C dataset identifiers/families are invalid")
        _relative, image_path = _safe_image(root, row.get("image"))
        result[scene_id] = (str(family), _world(row.get("values"), "dataset truth"), image_path)
    if len(result) != 8000:
        raise RuntimeError("S5 Phase C dataset scene identifiers are not unique")
    return result


def _load_s4_contracts(
    records_path: Path,
    summary_path: Path,
    dataset: dict[str, tuple[str, tuple[int, int, int, int], Path]],
    *,
    expected_scenes: int,
    expected_conditions: int,
    expected_family_counts: dict[str, int],
) -> tuple[_SceneContract, ...]:
    rows = tuple(
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_calls = expected_scenes * expected_conditions
    if (
        not isinstance(summary, dict)
        or summary.get("status") != "PHASE_2_LAYERWISE_ASSIMILATION_EXECUTED"
        or summary.get("number_of_scenes") != expected_scenes
        or summary.get("number_of_forward_calls") != expected_calls
        or summary.get("final_forward_parity_verified") is not True
        or summary.get("subjective_success_threshold_applied") is not False
        or len(rows) != expected_calls
    ):
        raise RuntimeError("S5 frozen S4 structure/provenance drifted")
    grouped: dict[str, dict[CueCondition, dict[str, object]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("S5 S4 record is not an object")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str):
            raise RuntimeError("S5 S4 scene identifier is invalid")
        condition = CueCondition(row.get("cue_condition"))
        group = grouped.setdefault(scene_id, {})
        if condition in group:
            raise RuntimeError("S5 S4 scene condition is duplicated")
        group[condition] = row
    if len(grouped) != expected_scenes or any(
        set(group) != set(CueCondition) for group in grouped.values()
    ):
        raise RuntimeError("S5 S4 scene/condition closure drifted")
    contracts: list[_SceneContract] = []
    for scene_id, group in sorted(grouped.items()):
        exemplar = group[CueCondition.VALID_CUE]
        family = exemplar.get("family")
        labels = exemplar.get("candidate_labels")
        worlds = exemplar.get("candidate_worlds")
        if (
            family not in expected_family_counts
            or not isinstance(labels, list)
            or len(labels) != 4
            or not isinstance(worlds, list)
            or len(worlds) != 4
        ):
            raise RuntimeError("S5 S4 candidate semantics are malformed")
        by_label = {
            str(label): _world(world, "S4 candidate world")
            for label, world in zip(labels, worlds, strict=True)
        }
        try:
            truth = by_label[str(exemplar["true_label"])]
            observed = by_label[str(exemplar["observed_label"])]
            counterfactual = by_label[str(exemplar["counterfactual_label"])]
        except KeyError as error:
            raise RuntimeError("S5 S4 candidate labels do not resolve frozen worlds") from error
        if scene_id not in dataset:
            raise RuntimeError("S5 S4 scene is absent from the frozen visual dataset")
        dataset_family, dataset_truth, image_path = dataset[scene_id]
        if dataset_family != family or dataset_truth != truth:
            raise RuntimeError("S5 S4 scene differs from its frozen visual source")
        semantic_fields = (
            "family",
            "candidate_labels",
            "candidate_worlds",
            "true_label",
            "observed_label",
            "counterfactual_label",
        )
        if any(
            row.get(field) != exemplar.get(field)
            for row in group.values()
            for field in semantic_fields
        ):
            raise RuntimeError("S5 S4 cue conditions drifted across one scene")
        contracts.append(
            _SceneContract(
                scene_id=scene_id,
                family=str(family),
                truth=truth,
                observed=observed,
                counterfactual=counterfactual,
                image_path=image_path,
            )
        )
    if Counter(contract.family for contract in contracts) != Counter(expected_family_counts):
        raise RuntimeError("S5 S4 family counts drifted")
    return tuple(contracts)


def run_cache_parity_cli(
    *,
    phase: str,
    expected_input_sha256: tuple[str, ...],
    expected_scenes: int,
    expected_conditions: int,
    output_path: str,
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
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-sha256", action="append", default=[])
    arguments = parser.parse_args()
    if blocked_unless_execute(arguments.execute):
        return 2
    try:
        output = Path(output_path)
        validate_cache_output_path(output)
        validation = validate_server_inputs(
            config=arguments.config,
            package_lock=arguments.package_lock,
            model_path=arguments.model_path,
            inputs=arguments.input,
            input_sha256=arguments.input_sha256,
            expected_input_sha256=expected_input_sha256,
            require_raw_evidence=True,
        )
        if len(arguments.input) != 2:
            raise RuntimeError("S5 requires exactly the two frozen S4 artifacts")
        config = _load_config(arguments.config)
        validate_runtime_evidence(config["runtime_evidence"])
        contract = config["phase_3_cache_parity"]
        if (
            not isinstance(contract, dict)
            or contract.get("world_recoverable_scenes") != expected_scenes
            or len(contract.get("cue_conditions", ())) != expected_conditions
            or contract.get("parity_call_cap") != expected_scenes * expected_conditions
            or contract.get("do_sample") is not False
            or contract.get("temperature") != 0.0
            or contract.get("require_exact_generated_tokens") is not True
            or contract.get("require_exact_generated_logits") is not False
            or contract.get("require_stepwise_argmax_parity") is not True
            or contract.get("require_realized_token_top1") is not True
            or contract.get("report_stepwise_logit_drift") is not True
            or contract.get("continue_after_call_level_token_divergence") is not True
            or contract.get("report_token_divergence_evidence") is not True
            or contract.get("exclude_token_divergent_calls_from_i4_primary") is not True
            or contract.get("require_exact_suffix_positions") is not True
            or contract.get("require_exact_cache_positions") is not True
            or contract.get("logit_absolute_tolerance") != 0.0
            or contract.get("logit_relative_tolerance") != 0.0
            or "maximum_logit_drift" in contract
            or "allowed_token_divergence_count" in contract
            or "allowed_token_divergence_rate" in contract
        ):
            raise RuntimeError("S5 cache-parity execution contract is malformed")
        observation_prompt, correction_prompt = _load_prompts(arguments.prompt_config)
        dataset = _load_dataset(arguments.dataset_root)
        contracts = _load_s4_contracts(
            arguments.input[0],
            arguments.input[1],
            dataset,
            expected_scenes=expected_scenes,
            expected_conditions=expected_conditions,
            expected_family_counts=contract["included_family_counts"],
        )
        model, processor = load_pinned_qwen(model_path=arguments.model_path)
        records = []
        completed = 0
        for scene_index, scene in enumerate(contracts, start=1):
            observation = generate_observation_with_cache(
                model,
                processor,
                str(scene.image_path),
                observation_prompt,
                sample_id=scene.scene_id,
                resized_height=int(config["vision_input"]["resized_height"]),
                resized_width=int(config["vision_input"]["resized_width"]),
                max_new_tokens=int(contract["max_new_tokens"]),
                rng_seed=2026081701,
            )
            state = observation.get("state")
            if state is None:
                raise RuntimeError("S5 natural observation returned no cached state")
            turns = build_condition_turns(
                correction_prompt=correction_prompt,
                family=scene.family,
                truth=scene.truth,
                observed=scene.observed,
                counterfactual=scene.counterfactual,
            )
            calls = tuple(
                replace(call, family=scene.family)
                for call in build_cache_parity_plan(
                    (state,),
                    condition_turns={scene.scene_id: turns},
                    expected_scenes=1,
                )
            )

            def report_progress(
                local_completed: int,
                _local_total: int,
                base_completed: int = completed,
            ) -> None:
                global_completed = base_completed + local_completed
                total = int(contract["parity_call_cap"])
                if global_completed == total or global_completed % 25 == 0:
                    print(
                        f"PROGRESS: {global_completed}/{total} S5 parity calls complete",
                        flush=True,
                    )

            records.extend(
                execute_cache_parity_plan(
                    model,
                    processor,
                    calls,
                    max_new_tokens=int(contract["max_new_tokens"]),
                    logit_absolute_tolerance=float(contract["logit_absolute_tolerance"]),
                    logit_relative_tolerance=float(contract["logit_relative_tolerance"]),
                    progress=report_progress,
                )
            )
            completed += expected_conditions
            if scene_index % 25 == 0 or scene_index == expected_scenes:
                print(
                    f"PROGRESS: {scene_index}/{expected_scenes} S5 visual states complete",
                    flush=True,
                )
        if len(records) != contract["parity_call_cap"]:
            raise RuntimeError("S5 executed parity-call count drifted")
        summary = {
            **summarize_cache_parity(records),
            "config_sha256": validation.config_sha256,
            "package_lock_sha256": validation.package_lock_sha256,
            "model_snapshot_sha256": validation.model_snapshot_sha256,
            "hash_bound_inputs": list(validation.inputs),
            "phase_c_dataset_manifest_sha256": PHASE_C_DATASET_MANIFEST_SHA256,
            "phase_c_dataset_records_sha256": PHASE_C_DATASET_RECORDS_SHA256,
        }
        write_cache_parity_outputs(output, records=records, summary=summary)
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: {phase} output written to {output_path}")
    return 0


def main() -> int:
    return run_cache_parity_cli(
        phase="phase_3_cache_parity",
        expected_input_sha256=(
            LAYERWISE_PROFILES_SHA256,
            LAYERWISE_SUMMARY_SHA256,
        ),
        expected_scenes=579,
        expected_conditions=4,
        output_path=str(ROOT / "artifacts/v4/cache/cache_parity.json"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
