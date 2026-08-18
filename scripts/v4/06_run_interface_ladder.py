from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _guards import (  # noqa: E402
    CACHE_PARITY_SHA256,
    CANDIDATE_LABELS_SHA256,
    CANDIDATE_SCORES_SHA256,
    CANDIDATE_SUMMARY_SHA256,
    CAPABILITY_PAIRED_GAPS_SHA256,
    CAPABILITY_PER_SCENE_SHA256,
    CAPABILITY_SUMMARY_SHA256,
    CONFIG_PATH,
    LAYERWISE_PROFILES_SHA256,
    LAYERWISE_SUMMARY_SHA256,
    LEGACY_SCREEN_RECORDS_SHA256,
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

from compensability_v4.diagnostics.capability_chain import (  # noqa: E402
    CapabilityCall,
    CapabilityTaskType,
    parse_capability_output,
    select_legacy_capability_scenes,
)
from compensability_v4.diagnostics.interface_ladder import (  # noqa: E402
    CueCondition,
    Interface,
)
from compensability_v4.qwen.capability_runner import _decode_one  # noqa: E402
from compensability_v4.qwen.manual_generation import (  # noqa: E402
    generate_observation_with_cache,
)
from compensability_v4.qwen.model_loader import load_pinned_qwen  # noqa: E402
from compensability_v4.qwen.phase2_candidate import (  # noqa: E402
    CueCondition as CandidateCueCondition,
)
from compensability_v4.qwen.phase3_cache import facts_for_condition  # noqa: E402
from compensability_v4.qwen.phase3_interface import (  # noqa: E402
    InterfaceLadderRecord,
    summarize_interface_ladder,
    validate_interface_ladder_records,
    write_interface_ladder_outputs,
)

PROMPT_CONFIG = ROOT / "configs/recoverability/v4/phase_1_3_prompts.yaml"
DEFAULT_DATASET_ROOT = ROOT / "data/generated/cva_recoverability_causal_v2_screen"
_WORLD = re.compile(r"\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*")


class _Scene(NamedTuple):
    scene_id: str
    family: str
    truth: tuple[int, int, int, int]
    observed: tuple[int, int, int, int]
    counterfactual: tuple[int, int, int, int]
    value_domain: tuple[int, ...]
    image_path: Path


def _parse_world(text: object) -> tuple[int, int, int, int] | None:
    match = _WORLD.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _world(value: object, label: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise RuntimeError(f"S6 {label} must contain exactly four integers")
    return tuple(value)  # type: ignore[return-value]


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"S6 JSONL source is empty or malformed: {path}")
    return rows


def _numeric_token_sequences(
    tokenizer: object, value_domain: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    """Return lossless tokenizer renderings for the frozen numeric domain.

    Qwen's BPE represents some integers (for example, two-digit values) with
    more than one token.  The S6 diagnostic records the *first-token logit*
    for every complete numeric rendering, rather than silently selecting a
    digit token and calling it a score for the whole integer.
    """

    encode, decode = getattr(tokenizer, "encode", None), getattr(tokenizer, "decode", None)
    if not callable(encode) or not callable(decode):
        raise TypeError("S6 tokenizer must expose encode() and decode()")
    sequences: dict[int, tuple[int, ...]] = {}
    for value in value_domain:
        rendered = str(value)
        encoded = encode(rendered, add_special_tokens=False)
        if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence) or not encoded:
            raise RuntimeError("S6 numeric value has no stable tokenizer rendering")
        try:
            token_ids = tuple(int(token_id) for token_id in encoded)
            decoded = decode(token_ids, skip_special_tokens=True)
        except (TypeError, ValueError) as error:
            raise RuntimeError("S6 numeric value tokenizer rendering is malformed") from error
        if decoded != rendered:
            raise RuntimeError("S6 numeric value tokenizer rendering does not round-trip")
        sequences[value] = token_ids
    if len(set(sequences.values())) != len(sequences):
        raise RuntimeError("S6 numeric tokenizer renderings are not unique")
    return sequences


def _find_token_span(
    generated: Sequence[int], expected: Sequence[int], *, start: int
) -> tuple[int, int]:
    """Locate the next complete tokenizer rendering without truncating it."""

    if not expected:
        raise RuntimeError("S6 numeric tokenizer rendering is empty")
    width = len(expected)
    for index in range(start, len(generated) - width + 1):
        if tuple(generated[index : index + width]) == tuple(expected):
            return index, index + width
    raise RuntimeError("S6 could not align a complete Stage-1 numeric rendering to its logits")


def _preflight_soft_report_tokenization(
    tokenizer: object, value_domains: Sequence[Sequence[int]]
) -> None:
    """Fail before scene inference when a frozen numeric domain cannot be audited."""

    domains = {tuple(int(value) for value in domain) for domain in value_domains}
    if not domains:
        raise RuntimeError("S6 tokenizer preflight has no numeric domain")
    for domain in sorted(domains):
        _numeric_token_sequences(tokenizer, domain)


def _build_soft_report_payload(
    tokenizer: object,
    *,
    generated_token_ids: Sequence[int],
    generated_logits: Sequence[object],
    value_domain: Sequence[int],
    top_k: int,
) -> tuple[str, tuple[int, int, int, int], dict[str, object]]:
    encode, decode = getattr(tokenizer, "encode", None), getattr(tokenizer, "decode", None)
    if not callable(encode) or not callable(decode):
        raise TypeError("S6 tokenizer must expose encode() and decode()")
    domain = tuple(value_domain)
    if (
        not domain
        or any(type(value) is not int for value in domain)
        or len(set(domain)) != len(domain)
        or type(top_k) is not int
        or not 0 < top_k <= len(domain)
    ):
        raise ValueError("S6 numeric value domain/top-k contract is invalid")
    tokens_by_value = _numeric_token_sequences(tokenizer, domain)

    generated = tuple(int(token_id) for token_id in generated_token_ids)
    logits = tuple(generated_logits)
    if not generated or len(generated) != len(logits):
        raise RuntimeError("S6 Stage-1 token/logit trace is misaligned")
    raw = decode(generated, skip_special_tokens=True)
    parsed = _parse_world(raw)
    if parsed is None:
        raise RuntimeError("S6 Stage-1 output is not a four-integer CSV")
    emitted_tokens_by_value = _numeric_token_sequences(tokenizer, parsed)
    numeric_domain_valid = all(value in tokens_by_value for value in parsed)
    positions: list[dict[str, object]] = []
    cursor = 0
    for index, generated_value in enumerate(parsed):
        expected_tokens = emitted_tokens_by_value[generated_value]
        generated_step, next_cursor = _find_token_span(generated, expected_tokens, start=cursor)
        cursor = next_cursor
        step_logits = logits[generated_step]
        shape = getattr(step_logits, "shape", None)
        if shape is None or len(shape) not in (1, 2) or (len(shape) == 2 and shape[0] != 1):
            raise RuntimeError("S6 Stage-1 logits have an invalid shape")
        row = step_logits[0] if len(shape) == 2 else step_logits
        scored: list[tuple[int, float]] = []
        for value, token_ids in tokens_by_value.items():
            try:
                score = float(row[token_ids[0]].item())
            except (AttributeError, IndexError, TypeError) as error:
                raise RuntimeError("S6 numeric token is outside the Stage-1 vocabulary") from error
            if not math.isfinite(score):
                raise RuntimeError("S6 Stage-1 numeric logits must be finite")
            scored.append((value, score))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        best = ranked[0][1]
        positions.append(
            {
                "index": index,
                "generated_step": generated_step,
                "generated_token_ids": list(expected_tokens),
                "candidates": [
                    {
                        "value": value,
                        "relative_logit": score - best,
                        "token_ids": list(tokens_by_value[value]),
                    }
                    for value, score in ranked[:top_k]
                ],
            }
        )
    return (
        raw,
        parsed,
        {
            "top_k": top_k,
            "score_basis": "first_token_logit",
            "raw_output": raw,
            "output_format_valid": True,
            "numeric_domain_valid": numeric_domain_valid,
            "positions": positions,
        },
    )


def _load_prompts(path: Path) -> tuple[str, str]:
    if path.is_symlink() or path.resolve() != PROMPT_CONFIG.resolve() or not path.is_file():
        raise RuntimeError(f"prompt config must be canonical: {PROMPT_CONFIG}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts") if isinstance(payload, dict) else None
    observation = prompts.get("stage_1_observation") if isinstance(prompts, dict) else None
    recovery = prompts.get("T6") if isinstance(prompts, dict) else None
    if any(not isinstance(item, str) or not item.strip() for item in (observation, recovery)):
        raise RuntimeError("S6 observation/recovery prompts are missing")
    return str(observation), str(recovery)


def _load_dataset(root: Path) -> dict[str, tuple[str, tuple[int, int, int, int], Path]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("S6 dataset root must be an absolute regular directory")
    root = root.resolve()
    manifest, records = root / "manifest.json", root / "records.jsonl"
    if (
        manifest.is_symlink()
        or records.is_symlink()
        or sha256(manifest) != PHASE_C_DATASET_MANIFEST_SHA256
        or sha256(records) != PHASE_C_DATASET_RECORDS_SHA256
    ):
        raise RuntimeError("S6 dataset manifest/records SHA-256 drifted")
    metadata, rows = json.loads(manifest.read_text(encoding="utf-8")), _jsonl(records)
    if (
        not isinstance(metadata, dict)
        or metadata.get("record_count") != 8000
        or metadata.get("records_sha256") != PHASE_C_DATASET_RECORDS_SHA256
        or len(rows) != 8000
    ):
        raise RuntimeError("S6 dataset structure drifted")
    result: dict[str, tuple[str, tuple[int, int, int, int], Path]] = {}
    images: list[tuple[str, Path]] = []
    for row in rows:
        scene_id, family, relative = row.get("scene_id"), row.get("family"), row.get("image")
        if not all(isinstance(item, str) for item in (scene_id, family, relative)):
            raise RuntimeError("S6 dataset row identifiers are malformed")
        posix = PurePosixPath(str(relative))
        image = (root / Path(*posix.parts)).resolve()
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or posix.suffix.lower() != ".png"
            or root not in image.parents
            or image.is_symlink()
            or not image.is_file()
        ):
            raise RuntimeError("S6 dataset image path is unsafe or missing")
        if str(scene_id) in result:
            raise RuntimeError("S6 dataset scene identifier is duplicated")
        result[str(scene_id)] = (str(family), _world(row.get("values"), "dataset world"), image)
        images.append((str(relative), image))
    if len({relative for relative, _path in images}) != len(images):
        raise RuntimeError("S6 dataset image path is duplicated")
    image_digest = hashlib.sha256()
    for relative, image in sorted(images):
        image_digest.update(relative.encode())
        image_digest.update(b"\0")
        image_digest.update(sha256(image).encode())
        image_digest.update(b"\n")
    if metadata.get("images_sha256") != image_digest.hexdigest():
        raise RuntimeError("S6 dataset image bundle SHA-256 drifted")
    return result


def _indexed_rows(
    rows: Sequence[dict[str, object]],
    *,
    expected_scenes: int,
    expected_conditions: int,
    label: str,
) -> dict[tuple[str, CueCondition], dict[str, object]]:
    if len(rows) != expected_scenes * expected_conditions:
        raise RuntimeError(f"S6 {label} record count drifted")
    indexed: dict[tuple[str, CueCondition], dict[str, object]] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str):
            raise RuntimeError(f"S6 {label} scene identifier is malformed")
        key = (scene_id, CueCondition(row.get("cue_condition")))
        if key in indexed:
            raise RuntimeError(f"S6 {label} cell is duplicated")
        indexed[key] = row
    if len({key[0] for key in indexed}) != expected_scenes:
        raise RuntimeError(f"S6 {label} scene closure drifted")
    return indexed


def _candidate_rows(
    path: Path, scenes: int, conditions: int
) -> dict[tuple[str, CueCondition], dict[str, object]]:
    indexed = _indexed_rows(
        _jsonl(path), expected_scenes=scenes, expected_conditions=conditions, label="S3 candidate"
    )
    for row in indexed.values():
        labels, worlds, logits = (
            row.get("candidate_labels"),
            row.get("candidate_worlds"),
            row.get("candidate_logits"),
        )
        if (
            not isinstance(labels, list)
            or len(labels) != 4
            or len(set(labels)) != 4
            or not isinstance(worlds, list)
            or len(worlds) != 4
            or not isinstance(logits, dict)
            or set(logits) != set(labels)
            or any(not isinstance(logits[label], (int, float)) for label in labels)
        ):
            raise RuntimeError("S6 S3 candidate semantics are malformed")
    return indexed


def _cache_rows(
    path: Path, scenes: int, conditions: int
) -> dict[tuple[str, CueCondition], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise RuntimeError("S6 S5 records are malformed")
    if (
        not isinstance(summary, dict)
        or summary.get("number_of_scenes") != scenes
        or summary.get("number_of_parity_calls") != scenes * conditions
        or summary.get("subjective_success_threshold_applied") is not False
    ):
        raise RuntimeError("S6 S5 summary contract drifted")
    indexed = _indexed_rows(
        records, expected_scenes=scenes, expected_conditions=conditions, label="S5 cache"
    )
    for row in indexed.values():
        tokens_valid = all(
            isinstance(row.get(field), list) and all(type(item) is int for item in row[field])
            for field in ("cached_generated_token_ids", "full_generated_token_ids")
        )
        primary, diagnostic = row.get("primary_eligible"), row.get("diagnostic_only")
        if (
            not tokens_valid
            or row.get("suffix_parity_verified") is not True
            or row.get("mrope_parity_verified") is not True
            or row.get("cache_position_parity_verified") is not True
            or type(primary) is not bool
            or type(diagnostic) is not bool
            or primary == diagnostic
        ):
            raise RuntimeError("S6 S5 structural/eligibility evidence is invalid")
    return indexed


def _runtime_sha(model: str, prompt: str, inputs: Sequence[str]) -> str:
    material = json.dumps(
        {
            "stage": "S6_runtime",
            "model": model,
            "prompt": prompt,
            "inputs": list(inputs),
            "dataset_manifest": PHASE_C_DATASET_MANIFEST_SHA256,
            "dataset_records": PHASE_C_DATASET_RECORDS_SHA256,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _i0_messages(
    prompt: str, observed: tuple[int, ...], facts: Sequence[Mapping[str, object]]
) -> tuple[Mapping[str, str], ...]:
    return (
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"observed_values": list(observed), "facts": [dict(fact) for fact in facts]},
                separators=(",", ":"),
                sort_keys=True,
            ),
        },
    )


def _candidate_output(row: Mapping[str, object]) -> tuple[int, int, int, int]:
    labels, worlds, logits = (
        row["candidate_labels"],
        row["candidate_worlds"],
        row["candidate_logits"],
    )
    assert isinstance(labels, list) and isinstance(worlds, list) and isinstance(logits, dict)
    winner = min(labels, key=lambda label: (-float(logits[label]), labels.index(label)))
    return _world(worlds[labels.index(winner)], "selected candidate")


def _record(
    scene: _Scene,
    interface: Interface,
    condition: CueCondition,
    *,
    output: tuple[int, int, int, int] | None,
    parsed: bool | None,
    payload: Mapping[str, object] | None,
    stage: str,
    branch: str,
    source_call: str,
    source_hash: str,
    primary: bool,
    diagnostic: bool,
    reason: str | None,
) -> InterfaceLadderRecord:
    return InterfaceLadderRecord(
        call_id=f"S6:{scene.scene_id}:{interface.value}:{condition.value}",
        scene_id=scene.scene_id,
        family=scene.family,
        interface=interface,
        condition=condition,
        true_world=scene.truth,
        observed_world=scene.observed,
        counterfactual_world=scene.counterfactual,
        output_world=output,
        parse_success=parsed,
        diagnostic_payload=payload,
        source_stage=stage,
        source_branch=branch,
        source_call_id=source_call,
        source_artifact_sha256=source_hash,
        structural_validity_verified=True,
        primary_eligible=primary,
        diagnostic_only=diagnostic,
        diagnostic_reason=reason,
    )


def run_interface_ladder_cli(
    *,
    phase: str,
    expected_scenes: int,
    expected_conditions: int,
    expected_interfaces: int,
    expected_cells_per_scene: int,
    required_sources: tuple[str, ...],
    measurement_sources: Mapping[str, str],
    provenance_only_sources: tuple[str, ...],
    input_roles: Mapping[str, str],
    output_paths: Mapping[str, str],
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--package-lock", type=Path, default=PACKAGE_LOCK_PATH)
    parser.add_argument(
        "--model-path", type=Path, default=Path("/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct")
    )
    parser.add_argument("--prompt-config", type=Path, default=PROMPT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-sha256", action="append", default=[])
    args = parser.parse_args()
    if blocked_unless_execute(args.execute):
        return 2
    try:
        sources = (
            "screen",
            "capability_per_scene",
            "capability_summary",
            "capability_gaps",
            "candidate_labels",
            "candidate_scores",
            "candidate_summary",
            "layerwise_per_scene",
            "layerwise_summary",
            "cache_parity",
        )
        measurements = {
            "I0": "S6_runtime",
            "I1": "S6_runtime",
            "I2": "S3_candidate",
            "I3": "S5_cache.full_history",
            "I4": "S5_cache.cached_continuation",
        }
        provenance_sources = (
            "capability_per_scene",
            "capability_summary",
            "capability_gaps",
            "layerwise_per_scene",
            "layerwise_summary",
        )
        roles = {
            "screen": "runtime_scene_source",
            "capability_per_scene": "provenance_only",
            "capability_summary": "provenance_only",
            "capability_gaps": "provenance_only",
            "candidate_labels": "I2_provenance",
            "candidate_scores": "I2_measurement",
            "candidate_summary": "I2_provenance",
            "layerwise_per_scene": "provenance_only",
            "layerwise_summary": "provenance_only",
            "cache_parity": "I3_I4_measurement",
        }
        if (
            required_sources != sources
            or dict(measurement_sources) != measurements
            or provenance_only_sources != provenance_sources
            or dict(input_roles) != roles
            or expected_cells_per_scene != 17
        ):
            raise RuntimeError("S6 source/interface contract drifted")
        hashes = (
            LEGACY_SCREEN_RECORDS_SHA256,
            CAPABILITY_PER_SCENE_SHA256,
            CAPABILITY_SUMMARY_SHA256,
            CAPABILITY_PAIRED_GAPS_SHA256,
            CANDIDATE_LABELS_SHA256,
            CANDIDATE_SCORES_SHA256,
            CANDIDATE_SUMMARY_SHA256,
            LAYERWISE_PROFILES_SHA256,
            LAYERWISE_SUMMARY_SHA256,
            CACHE_PARITY_SHA256,
        )
        if len(args.input) != 10:
            raise RuntimeError("S6 requires exactly all ten frozen source artifacts")
        validation = validate_server_inputs(
            config=args.config,
            package_lock=args.package_lock,
            model_path=args.model_path,
            inputs=args.input,
            input_sha256=args.input_sha256,
            expected_input_sha256=hashes,
            require_raw_evidence=True,
        )
        config = _load_config(args.config)
        validate_runtime_evidence(config["runtime_evidence"])
        contract = config.get("phase_3_interface_ladder")
        if (
            not isinstance(contract, dict)
            or contract.get("world_recoverable_scenes") != expected_scenes
            or len(contract.get("cue_conditions", ())) != expected_conditions
            or len(contract.get("interfaces", ())) != expected_interfaces
            or contract.get("cells_per_scene") != 17
            or contract.get("interface_cell_count") != expected_scenes * 17
            or contract.get("i1_pre_cue_condition") != "no_cue"
            or contract.get("do_sample") is not False
            or contract.get("temperature") != 0.0
            or contract.get("subjective_success_thresholds_forbidden") is not True
            or contract.get("cache_parity_sha256") != CACHE_PARITY_SHA256
        ):
            raise RuntimeError("S6 config contract is malformed")
        paths = {name: ROOT / value for name, value in output_paths.items()}
        if set(paths) != {"per_scene", "summary"} or any(
            path.exists() or path.is_symlink() for path in paths.values()
        ):
            raise FileExistsError("refusing to overwrite an S6 artifact")

        selection = select_legacy_capability_scenes(args.input[0])
        if len(selection.scenes) != expected_scenes or Counter(
            scene.family for scene in selection.scenes
        ) != Counter(contract["included_family_counts"]):
            raise RuntimeError("S6 scene selection drifted")
        for path in (args.input[1], args.input[2]):
            if not path.read_text(encoding="utf-8").strip():
                raise RuntimeError("S6 capability provenance is empty")
        for path in (args.input[3], args.input[4], args.input[6], args.input[8]):
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                raise RuntimeError("S6 JSON provenance source is malformed")
        if len(_jsonl(args.input[7])) != expected_scenes * expected_conditions:
            raise RuntimeError("S6 S4 provenance record count drifted")
        candidates = _candidate_rows(args.input[5], expected_scenes, expected_conditions)
        cache = _cache_rows(args.input[9], expected_scenes, expected_conditions)
        dataset = _load_dataset(args.dataset_root)
        scenes: list[_Scene] = []
        for source in selection.scenes:
            candidate = candidates[(source.scene_id, CueCondition.VALID_CUE)]
            counterfactual = _world(candidate.get("counterfactual_world"), "counterfactual world")
            visual = dataset.get(source.scene_id)
            if visual is None or visual[:2] != (source.family, source.truth):
                raise RuntimeError("S6 visual dataset differs from the selected scene")
            scenes.append(
                _Scene(
                    source.scene_id,
                    source.family,
                    source.truth,
                    source.observed,
                    counterfactual,
                    source.value_domain,
                    visual[2],
                )
            )

        observation_prompt, recovery_prompt = _load_prompts(args.prompt_config)
        model, processor = load_pinned_qwen(model_path=args.model_path)
        tokenizer = getattr(processor, "tokenizer", processor)
        _preflight_soft_report_tokenization(tokenizer, [scene.value_domain for scene in scenes])
        runtime_hash = _runtime_sha(
            validation.model_snapshot_sha256, sha256(args.prompt_config), hashes
        )
        records: list[InterfaceLadderRecord] = []
        for scene_index, scene in enumerate(scenes, start=1):
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
            _raw, _parsed, soft = _build_soft_report_payload(
                tokenizer,
                generated_token_ids=observation["generated_token_ids"],
                generated_logits=observation["generated_logits"],
                value_domain=scene.value_domain,
                top_k=int(contract["i1_soft_report_top_k"]),
            )
            records.append(
                _record(
                    scene,
                    Interface.I1_SOFT_REPORT,
                    CueCondition.NO_CUE,
                    output=None,
                    parsed=None,
                    payload=soft,
                    stage="S6_runtime",
                    branch="stage1_soft_report_runtime",
                    source_call=f"S6-stage1:{scene.scene_id}",
                    source_hash=runtime_hash,
                    primary=False,
                    diagnostic=True,
                    reason="intervention_diagnostic",
                )
            )
            for condition in CueCondition:
                candidate, cached = (
                    candidates[(scene.scene_id, condition)],
                    cache[(scene.scene_id, condition)],
                )
                facts = facts_for_condition(
                    family=scene.family,
                    truth=scene.truth,
                    observed=scene.observed,
                    counterfactual=scene.counterfactual,
                    condition=CandidateCueCondition(condition.value),
                )
                call = CapabilityCall(
                    call_id=f"S6-I0:{scene.scene_id}:{condition.value}",
                    scene_id=scene.scene_id,
                    family=scene.family,
                    task_type=CapabilityTaskType.T6,
                    expected_output=",".join(str(value) for value in scene.truth),
                    messages=_i0_messages(recovery_prompt, scene.observed, facts),
                )
                i0_raw = _decode_one(
                    model, processor, call, max_new_tokens=int(contract["max_new_tokens"])
                )
                i0 = parse_capability_output(CapabilityTaskType.T6, i0_raw)
                records.append(
                    _record(
                        scene,
                        Interface.I0_HARD_TEXT,
                        condition,
                        output=i0 if isinstance(i0, tuple) else None,
                        parsed=isinstance(i0, tuple),
                        payload=None,
                        stage="S6_runtime",
                        branch="fresh_text_runtime",
                        source_call=call.call_id,
                        source_hash=runtime_hash,
                        primary=True,
                        diagnostic=False,
                        reason=None,
                    )
                )
                i2 = _candidate_output(candidate)
                records.append(
                    _record(
                        scene,
                        Interface.I2_CANDIDATE_WORLD,
                        condition,
                        output=i2,
                        parsed=True,
                        payload={"selected_world": list(i2)},
                        stage="S3_candidate",
                        branch="teacher_forced_candidate",
                        source_call=str(candidate.get("call_id")),
                        source_hash=CANDIDATE_SCORES_SHA256,
                        primary=False,
                        diagnostic=True,
                        reason="intervention_diagnostic",
                    )
                )
                for interface, token_field, branch in (
                    (Interface.I3_SAME_CONVERSATION, "full_generated_token_ids", "full_history"),
                    (Interface.I4_EXACT_CACHE, "cached_generated_token_ids", "cached_continuation"),
                ):
                    output = _parse_world(
                        tokenizer.decode(cached[token_field], skip_special_tokens=True)
                    )
                    diagnostic = (
                        bool(cached["diagnostic_only"])
                        if interface is Interface.I4_EXACT_CACHE
                        else False
                    )
                    records.append(
                        _record(
                            scene,
                            interface,
                            condition,
                            output=output,
                            parsed=output is not None,
                            payload=None,
                            stage="S5_cache",
                            branch=branch,
                            source_call=str(cached.get("call_id")),
                            source_hash=CACHE_PARITY_SHA256,
                            primary=not diagnostic,
                            diagnostic=diagnostic,
                            reason="token_divergence" if diagnostic else None,
                        )
                    )
            if scene_index % 25 == 0 or scene_index == expected_scenes:
                print(f"PROGRESS: {scene_index}/{expected_scenes} S6 scenes complete", flush=True)

        validated = validate_interface_ladder_records(
            records,
            expected_scenes=expected_scenes,
            expected_conditions=expected_conditions,
            expected_interfaces=expected_interfaces,
            expected_source_sha256={
                "S6_runtime": runtime_hash,
                "S3_candidate": CANDIDATE_SCORES_SHA256,
                "S5_cache": CACHE_PARITY_SHA256,
            },
        )
        if len(validated) != expected_scenes * expected_cells_per_scene:
            raise RuntimeError("S6 executed cell count drifted")
        summary = {
            **summarize_interface_ladder(
                validated,
                bootstrap_resamples=int(contract["bootstrap_resamples"]),
                seed=2026081701,
            ),
            "config_sha256": validation.config_sha256,
            "package_lock_sha256": validation.package_lock_sha256,
            "model_snapshot_sha256": validation.model_snapshot_sha256,
            "hash_bound_inputs": list(validation.inputs),
        }
        write_interface_ladder_outputs(
            paths["per_scene"], paths["summary"], records=validated, summary=summary
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"READY: {phase} outputs written under {paths['per_scene'].parent}")
    return 0


def main() -> int:
    return run_interface_ladder_cli(
        phase="phase_3_interface_ladder",
        expected_scenes=579,
        expected_conditions=4,
        expected_interfaces=5,
        expected_cells_per_scene=17,
        required_sources=(
            "screen",
            "capability_per_scene",
            "capability_summary",
            "capability_gaps",
            "candidate_labels",
            "candidate_scores",
            "candidate_summary",
            "layerwise_per_scene",
            "layerwise_summary",
            "cache_parity",
        ),
        measurement_sources={
            "I0": "S6_runtime",
            "I1": "S6_runtime",
            "I2": "S3_candidate",
            "I3": "S5_cache.full_history",
            "I4": "S5_cache.cached_continuation",
        },
        provenance_only_sources=(
            "capability_per_scene",
            "capability_summary",
            "capability_gaps",
            "layerwise_per_scene",
            "layerwise_summary",
        ),
        input_roles={
            "screen": "runtime_scene_source",
            "capability_per_scene": "provenance_only",
            "capability_summary": "provenance_only",
            "capability_gaps": "provenance_only",
            "candidate_labels": "I2_provenance",
            "candidate_scores": "I2_measurement",
            "candidate_summary": "I2_provenance",
            "layerwise_per_scene": "provenance_only",
            "layerwise_summary": "provenance_only",
            "cache_parity": "I3_I4_measurement",
        },
        output_paths={
            "per_scene": "artifacts/v4/interface_ladder/per_scene.jsonl",
            "summary": "artifacts/v4/interface_ladder/summary.json",
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
