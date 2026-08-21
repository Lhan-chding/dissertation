"""Inference execution, resume, metrics, and publication for v5 Study A."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from compensability_v4.qwen.phase5_runtime import (
    completion_log_probability,
    freeze_inference_model,
    generate_completion,
    phase5_rollout_seed,
    tree_sha256,
)
from compensability_v4.qwen.phase5_support import HeldOutNaturalError, parse_world
from compensability_v5.audit.audit_v4_raw import SafeTarArchive

from .study_a_scenarios import (
    BASE_SHA256,
    CHECKPOINTS,
    PROMPT_VERSION,
    RAW_ARCHIVE_MEMBER,
    RAW_ARCHIVE_SHA256,
    STUDY_A_ACK,
    T_ADAPTER_SHA256,
    StudyAScenario,
    World,
    _canonical_json,
    _fiber_bin,
    _permuted_world,
    _world,
    _world_text,
    build_study_a_scenarios,
    sha256_file,
)

CheckpointLoader = Callable[[str], tuple[object, object]]


def load_natural_errors(
    raw_archive: Path, *, expected_sha256: str = RAW_ARCHIVE_SHA256
) -> tuple[tuple[HeldOutNaturalError, ...], str]:
    """Read the one registered member without extracting the supplied archive."""

    with SafeTarArchive(raw_archive, expected_sha256=expected_sha256) as archive:
        rows = archive.read_jsonl(RAW_ARCHIVE_MEMBER)
        observed_sha256 = archive.archive_sha256
    errors = tuple(HeldOutNaturalError.from_mapping(row) for row in rows)
    if not errors or len({error.scene_id for error in errors}) != len(errors):
        raise RuntimeError("Study A natural-error source is empty or duplicated")
    if any(
        len(error.error_indices) != 1 or error.stage1_model_sha256 != BASE_SHA256
        for error in errors
    ):
        raise RuntimeError("Study A source is not the frozen single-error Base capture")
    return tuple(sorted(errors, key=lambda item: item.scene_id)), observed_sha256


def require_study_a_authorization(
    *, execute: bool, acknowledgement: str | None, environment: Mapping[str, str] | None = None
) -> None:
    if not execute:
        raise PermissionError("Study A requires explicit --execute")
    if acknowledgement != STUDY_A_ACK:
        raise PermissionError("Study A requires the exact execution acknowledgement")
    current = os.environ if environment is None else environment
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    if any(current.get(name) != "1" for name in required):
        raise RuntimeError("Study A requires a complete offline environment")


def require_t_adapter(path: Path) -> Path:
    if tree_sha256(path) != T_ADAPTER_SHA256:
        raise RuntimeError("Study A T adapter tree SHA-256 mismatch")
    return path


def load_gpu_checkpoint(  # pragma: no cover - real pinned CUDA/PEFT runtime
    checkpoint: str, *, t_adapter: Path
) -> tuple[object, object]:
    """Load one verified local checkpoint, with PEFT used only for T."""

    from compensability_v4.qwen.model_loader import load_pinned_qwen

    if checkpoint not in CHECKPOINTS:
        raise ValueError("Study A checkpoint is not registered")
    base, processor = load_pinned_qwen()
    model = base
    if checkpoint == "T":
        require_t_adapter(t_adapter)
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(t_adapter), is_trainable=False)
    freeze_inference_model(model)
    return model, processor


def _scenario_seed(base_seed: int, scenario: StudyAScenario, rollout_index: int) -> int:
    # Excludes checkpoint and transform identity so the full orbit receives
    # common random numbers for paired equivariance comparisons.
    return phase5_rollout_seed(base_seed, scenario.orbit_parent, rollout_index)


def _pushforward(output: World | None, scenario: StudyAScenario) -> World | None:
    return None if output is None else _permuted_world(output, scenario.pushforward_permutation)


def _measure_scenario(
    *,
    model: object,
    processor: object,
    checkpoint: str,
    scenario: StudyAScenario,
    canonical_row: Mapping[str, object] | None,
    k: int,
    sampling_seed: int,
) -> dict[str, object]:
    raw, greedy_ids = generate_completion(
        model,
        processor,
        scenario.prompt,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=32,
        seed=sampling_seed,
    )
    greedy = parse_world(raw)
    truth_text, observed_text = _world_text(scenario.truth), _world_text(scenario.observed)
    logp_truth = completion_log_probability(model, processor, scenario.prompt, truth_text)
    logp_observed = completion_log_probability(model, processor, scenario.prompt, observed_text)
    sample_raw: list[str] = []
    sample_ids: list[list[int]] = []
    sample_seeds: list[int] = []
    sample_outputs: list[World | None] = []
    for index in range(k):
        seed = _scenario_seed(sampling_seed, scenario, index)
        sampled_raw, sampled_ids = generate_completion(
            model,
            processor,
            scenario.prompt,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
            top_k=0,
            max_new_tokens=32,
            seed=seed,
        )
        sample_raw.append(sampled_raw)
        sample_ids.append(list(sampled_ids))
        sample_seeds.append(seed)
        sample_outputs.append(parse_world(sampled_raw))
    successes = [output == scenario.truth for output in sample_outputs]
    pushed_greedy: World | None = None
    paired_consistency: list[bool] = []
    if canonical_row is not None:
        canonical_greedy_raw = canonical_row.get("greedy_output")
        canonical_greedy = (
            _world(canonical_greedy_raw, "cached canonical greedy output")
            if canonical_greedy_raw is not None
            else None
        )
        pushed_greedy = _pushforward(canonical_greedy, scenario)
        paired_consistency.append(greedy is not None and greedy == pushed_greedy)
        canonical_samples = canonical_row.get("sample_outputs")
        if not isinstance(canonical_samples, list) or len(canonical_samples) != k:
            raise RuntimeError("Study A canonical sample trace is malformed")
        for canonical_output, transformed_output in zip(
            canonical_samples, sample_outputs, strict=True
        ):
            parsed_canonical = (
                _world(canonical_output, "cached canonical sampled output")
                if canonical_output is not None
                else None
            )
            expected = _pushforward(parsed_canonical, scenario)
            paired_consistency.append(
                transformed_output is not None and transformed_output == expected
            )
    defect = (
        0.0 if canonical_row is None else 1.0 - sum(paired_consistency) / len(paired_consistency)
    )
    return {
        "schema_version": 1,
        **scenario.to_mapping(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": BASE_SHA256 if checkpoint == "Base" else T_ADAPTER_SHA256,
        "greedy_raw_output": raw,
        "greedy_token_ids": list(greedy_ids),
        "greedy_output": list(greedy) if greedy is not None else None,
        "greedy_parse_success": greedy is not None,
        "greedy_exact_recovery": greedy == scenario.truth,
        "candidate_logp_truth": logp_truth,
        "candidate_logp_observed": logp_observed,
        "candidate_margin_true_observed": logp_truth - logp_observed,
        "sample_raw_outputs": sample_raw,
        "sample_token_ids": sample_ids,
        "sample_seeds": sample_seeds,
        "sample_outputs": [
            list(output) if output is not None else None for output in sample_outputs
        ],
        "sample_exact_recovery": successes,
        "exact_recovery_probability": sum(successes) / k,
        "pass_at_k": any(successes),
        "pushed_forward_canonical_greedy": (
            list(pushed_greedy) if pushed_greedy is not None else None
        ),
        "equivariance_consistent": defect == 0.0,
        "equivariance_defect": defect,
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
    }


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [row[field] for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise RuntimeError(f"Study A metric {field} is malformed")
    return sum(float(value) for value in values) / len(values)


def _group_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    noncanonical = [row for row in rows if row["graph_axis"] != "canonical"]
    return {
        "checkpoint_scene_count": len(rows),
        "semantic_scene_count": len({str(row["source_scene_id"]) for row in rows}),
        "mean_fiber_size": _mean(rows, "fiber_size"),
        "greedy_exact_recovery_rate": sum(bool(row["greedy_exact_recovery"]) for row in rows)
        / len(rows),
        "sample_exact_recovery_rate": _mean(rows, "exact_recovery_probability"),
        "pass_at_k_rate": sum(bool(row["pass_at_k"]) for row in rows) / len(rows),
        "candidate_margin_true_observed_mean": _mean(rows, "candidate_margin_true_observed"),
        "equivariance_defect_mean": (
            _mean(noncanonical, "equivariance_defect") if noncanonical else 0.0
        ),
    }


def summarize_study_a(
    rows: Sequence[Mapping[str, object]],
    *,
    source_name: str,
    source_sha256: str,
    additional_source_sha256: Mapping[str, str] | None = None,
    k: int,
    sampling_seed: int,
) -> dict[str, object]:
    expected_checkpoints = set(CHECKPOINTS)
    if not rows or {str(row.get("checkpoint")) for row in rows} != expected_checkpoints:
        raise RuntimeError("Study A summary requires complete Base/T evidence")
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = {
        "checkpoint": defaultdict(list),
        "family": defaultdict(list),
        "graph_axis": defaultdict(list),
        "capture_label": defaultdict(list),
        "split": defaultdict(list),
    }
    for row in rows:
        for dimension in grouped:
            grouped[dimension][str(row[dimension])].append(row)
    return {
        "schema_version": 1,
        "status": "V5_STUDY_A_EXECUTED",
        "source_sha256": {
            source_name: source_sha256,
            **dict(additional_source_sha256 or {}),
            "Base": BASE_SHA256,
            "T": T_ADAPTER_SHA256,
        },
        "prompt_template_version": PROMPT_VERSION,
        "semantic_scene_count": len({str(row["source_scene_id"]) for row in rows}),
        "scenario_count": len({str(row["scenario_id"]) for row in rows}),
        "scenario_checkpoint_count": len(rows),
        "k": k,
        "sampling_seed": sampling_seed,
        "by_checkpoint": {
            key: _group_summary(value) for key, value in sorted(grouped["checkpoint"].items())
        },
        "by_family": {
            key: _group_summary(value) for key, value in sorted(grouped["family"].items())
        },
        "by_graph_axis": {
            key: _group_summary(value) for key, value in sorted(grouped["graph_axis"].items())
        },
        "by_capture_label": {
            key: _group_summary(value) for key, value in sorted(grouped["capture_label"].items())
        },
        "by_split": {key: _group_summary(value) for key, value in sorted(grouped["split"].items())},
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
        "confirmatory_data_used": False,
    }


def _trace_metadata(
    *,
    scenarios: Sequence[StudyAScenario],
    source_name: str,
    source_sha256: str,
    additional_source_sha256: Mapping[str, str] | None,
    k: int,
    sampling_seed: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "V5_STUDY_A_RAW_TRACE_IN_PROGRESS",
        "source_sha256": {
            source_name: source_sha256,
            **dict(additional_source_sha256 or {}),
            "Base": BASE_SHA256,
            "T": T_ADAPTER_SHA256,
        },
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "k": k,
        "sampling_seed": sampling_seed,
        "prompt_template_version": PROMPT_VERSION,
        "training_invoked": False,
        "rl_invoked": False,
        "prompt_search_invoked": False,
    }


def _load_or_create_trace(
    work_root: Path, metadata: Mapping[str, object]
) -> tuple[Path, dict[tuple[str, str], dict[str, object]]]:
    if work_root.is_symlink():
        raise RuntimeError("Study A work root must not be a symlink")
    metadata_path, trace_path = work_root / "trace_meta.json", work_root / "raw_trace.jsonl"
    if work_root.exists():
        if not work_root.is_dir() or not metadata_path.is_file() or metadata_path.is_symlink():
            raise RuntimeError("Study A resume root is incomplete or unsafe")
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed != dict(metadata):
            raise RuntimeError("Study A resume metadata drifted")
    else:
        work_root.mkdir(parents=True)
        with metadata_path.open("x", encoding="utf-8") as stream:
            stream.write(_canonical_json(dict(metadata)) + "\n")
    rows: dict[tuple[str, str], dict[str, object]] = {}
    if trace_path.exists():
        if trace_path.is_symlink() or not trace_path.is_file():
            raise RuntimeError("Study A raw resume trace is unsafe")
        for line_number, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Study A raw trace is malformed at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError("Study A raw trace rows must be objects")
            key = (str(row.get("checkpoint")), str(row.get("scenario_id")))
            if key in rows:
                raise RuntimeError("Study A raw trace contains duplicate rows")
            rows[key] = row
    return trace_path, rows


def _row_sha256(row: Mapping[str, object]) -> str:
    payload = {key: value for key, value in row.items() if key != "row_sha256"}
    import hashlib

    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _append_trace(trace_path: Path, row: Mapping[str, object]) -> dict[str, object]:
    payload = {**dict(row), "row_sha256": _row_sha256(row)}
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _validate_resumed_rows(
    rows: Mapping[tuple[str, str], Mapping[str, object]],
    scenarios: Mapping[str, StudyAScenario],
    *,
    k: int,
    sampling_seed: int,
) -> None:
    for (checkpoint, scenario_id), row in rows.items():
        scenario = scenarios.get(scenario_id)
        expected_checkpoint_hash = BASE_SHA256 if checkpoint == "Base" else T_ADAPTER_SHA256
        if row.get("row_sha256") != _row_sha256(row):
            raise RuntimeError("Study A resumed raw row integrity drifted")
        if (
            checkpoint not in CHECKPOINTS
            or scenario is None
            or row.get("schema_version") != 1
            or row.get("checkpoint_sha256") != expected_checkpoint_hash
            or row.get("source_scene_id") != scenario.source_scene_id
            or row.get("family") != scenario.family
            or row.get("graph_axis") != scenario.graph_axis
            or row.get("split") != scenario.split
            or row.get("capture_label") != scenario.capture_label
            or row.get("error_count") != scenario.error_count
            or row.get("observation_in_domain") is not scenario.observation_in_domain
            or row.get("observation_strict_parse_success")
            is not scenario.observation_strict_parse_success
            or row.get("prompt_sha256") != scenario.prompt_sha256
            or row.get("truth") != list(scenario.truth)
            or row.get("observed") != list(scenario.observed)
            or row.get("training_invoked") is not False
            or row.get("rl_invoked") is not False
            or row.get("prompt_search_invoked") is not False
        ):
            raise RuntimeError("Study A resumed raw row provenance drifted")
        seeds = row.get("sample_seeds")
        expected_seeds = [_scenario_seed(sampling_seed, scenario, index) for index in range(k)]
        samples = row.get("sample_outputs")
        if seeds != expected_seeds or not isinstance(samples, list) or len(samples) != k:
            raise RuntimeError("Study A resumed raw row sampling contract drifted")
        raw_samples = row.get("sample_raw_outputs")
        token_samples = row.get("sample_token_ids")
        if (
            not isinstance(raw_samples, list)
            or len(raw_samples) != k
            or not all(isinstance(item, str) for item in raw_samples)
            or not isinstance(token_samples, list)
            or len(token_samples) != k
            or not all(
                isinstance(items, list) and all(type(token) is int for token in items)
                for items in token_samples
            )
        ):
            raise RuntimeError("Study A resumed raw row sampling evidence drifted")
        parsed_samples = [
            list(parsed) if (parsed := parse_world(raw)) is not None else None
            for raw in raw_samples
        ]
        successes = [parsed == list(scenario.truth) for parsed in parsed_samples]
        greedy_raw = row.get("greedy_raw_output")
        greedy = parse_world(greedy_raw) if isinstance(greedy_raw, str) else None
        truth_logp = row.get("candidate_logp_truth")
        observed_logp = row.get("candidate_logp_observed")
        if (
            samples != parsed_samples
            or row.get("sample_exact_recovery") != successes
            or row.get("exact_recovery_probability") != sum(successes) / k
            or row.get("pass_at_k") is not any(successes)
            or row.get("greedy_output") != (list(greedy) if greedy is not None else None)
            or row.get("greedy_parse_success") is not (greedy is not None)
            or row.get("greedy_exact_recovery") is not (greedy == scenario.truth)
            or isinstance(truth_logp, bool)
            or not isinstance(truth_logp, (int, float))
            or isinstance(observed_logp, bool)
            or not isinstance(observed_logp, (int, float))
            or row.get("candidate_margin_true_observed") != truth_logp - observed_logp
        ):
            raise RuntimeError("Study A resumed raw row scientific metric drifted")
        expected_pushed: World | None = None
        expected_defect = 0.0
        if scenario.graph_axis != "canonical":
            canonical = rows.get((checkpoint, f"{scenario.source_scene_id}::canonical"))
            if canonical is None:
                raise RuntimeError("Study A resumed orbit lacks its canonical row")
            canonical_greedy_raw = canonical.get("greedy_output")
            canonical_greedy = (
                _world(canonical_greedy_raw, "resumed canonical greedy")
                if canonical_greedy_raw is not None
                else None
            )
            expected_pushed = _pushforward(canonical_greedy, scenario)
            paired = [greedy is not None and greedy == expected_pushed]
            canonical_samples = canonical.get("sample_outputs")
            if not isinstance(canonical_samples, list) or len(canonical_samples) != k:
                raise RuntimeError("Study A resumed canonical samples drifted")
            for canonical_output, transformed_output in zip(
                canonical_samples, parsed_samples, strict=True
            ):
                parsed_canonical = (
                    _world(canonical_output, "resumed canonical sampled output")
                    if canonical_output is not None
                    else None
                )
                expected = _pushforward(parsed_canonical, scenario)
                paired.append(transformed_output is not None and transformed_output == expected)
            expected_defect = 1.0 - sum(paired) / len(paired)
        if (
            row.get("pushed_forward_canonical_greedy")
            != (list(expected_pushed) if expected_pushed is not None else None)
            or row.get("equivariance_defect") != expected_defect
            or row.get("equivariance_consistent") is not (expected_defect == 0.0)
        ):
            raise RuntimeError("Study A resumed raw row equivariance metric drifted")


def _atomic_publish(
    *,
    output_root: Path,
    trace_path: Path,
    ordered_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Study A output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        per_scenario = temporary / "per_scenario.jsonl"
        with per_scenario.open("x", encoding="utf-8") as stream:
            for row in ordered_rows:
                stream.write(_canonical_json(dict(row)) + "\n")
        raw_trace = temporary / "raw_trace.jsonl"
        raw_trace.write_bytes(trace_path.read_bytes())
        summary_path = temporary / "summary.json"
        summary_path.write_text(_canonical_json(dict(summary)) + "\n", encoding="utf-8")
        published_files = [per_scenario, raw_trace, summary_path]
        split_names = {
            "phase2a_training_source": "phase2a",
            "independent_v4_support_dev": "legacy_independent",
        }
        by_split = summary.get("by_split")
        for split, prefix in split_names.items():
            split_rows = [row for row in ordered_rows if row.get("split") == split]
            if not split_rows:
                continue
            rows_path = temporary / f"{prefix}_per_scenario.jsonl"
            with rows_path.open("x", encoding="utf-8") as stream:
                for row in split_rows:
                    stream.write(_canonical_json(dict(row)) + "\n")
            split_summary = {
                "schema_version": 1,
                "status": "V5_STUDY_A_SPLIT_EVALUATED",
                "split": split,
                "source_sha256": summary["source_sha256"],
                "metrics": by_split.get(split) if isinstance(by_split, Mapping) else None,
                "checkpoint_scenario_count": len(split_rows),
                "training_invoked": False,
                "rl_invoked": False,
                "prompt_search_invoked": False,
            }
            split_summary_path = temporary / f"{prefix}_summary.json"
            split_summary_path.write_text(_canonical_json(split_summary) + "\n", encoding="utf-8")
            published_files.extend((rows_path, split_summary_path))
        base_phase2 = sorted(
            (
                row
                for row in ordered_rows
                if row.get("split") == "phase2a_training_source"
                and row.get("checkpoint") == "Base"
                and row.get("graph_axis") == "canonical"
            ),
            key=lambda row: (
                float(row["exact_recovery_probability"]),
                str(row["source_scene_id"]),
            ),
        )
        if base_phase2:
            support_names = ("low", "medium", "high")
            enriched_path = temporary / "phase2a_enriched_frozen_scenes.jsonl"
            with enriched_path.open("x", encoding="utf-8") as stream:
                for index, row in enumerate(base_phase2):
                    support_bin = support_names[min(2, index * 3 // len(base_phase2))]
                    enriched = {
                        "schema_version": 1,
                        "scene_id": row["scenario_id"],
                        "semantic_scene_id": row["source_scene_id"],
                        "split": row["split"],
                        "family": row["family"],
                        "prompt": row["prompt"],
                        "truth": row["truth"],
                        "natural_observation": row["observed"],
                        "constraint_matrix": row["constraint_matrix"],
                        "constraint_targets": row["constraint_targets"],
                        "answer_operation": {
                            "operator": row["answer_operation"],
                            "indices": row["answer_indices"],
                        },
                        "transformation": row["transformation"],
                        "capture_label": row["capture_label"],
                        "error_count": row["error_count"],
                        "observation_in_domain": row["observation_in_domain"],
                        "observation_strict_parse_success": row["observation_strict_parse_success"],
                        "fiber_size": row["fiber_size"],
                        "fiber_bin": _fiber_bin(int(row["fiber_size"])),
                        "base_exact_recovery_probability": row["exact_recovery_probability"],
                        "support_bin": support_bin,
                    }
                    stream.write(_canonical_json(enriched) + "\n")
            published_files.append(enriched_path)
        manifest = {
            "schema_version": 1,
            "status": "V5_STUDY_A_ATOMICALLY_PUBLISHED",
            "source_sha256": dict(summary["source_sha256"]),  # type: ignore[arg-type]
            "files": {
                path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in published_files
            },
            "training_invoked": False,
            "rl_invoked": False,
            "prompt_search_invoked": False,
        }
        (temporary / "manifest.json").write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        temporary.rename(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_study_a(
    *,
    errors: Iterable[HeldOutNaturalError] | None = None,
    scenarios: Iterable[StudyAScenario] | None = None,
    raw_archive_sha256: str | None = None,
    source_name: str = "raw_archive",
    source_sha256: str | None = None,
    additional_source_sha256: Mapping[str, str] | None = None,
    output_root: Path,
    work_root: Path,
    checkpoint_loader: CheckpointLoader,
    k: int = 8,
    sampling_seed: int = 2026082101,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    """Run or resume every Base/T scenario, then publish one immutable directory."""

    if type(k) is not int or k <= 0 or type(sampling_seed) is not int or sampling_seed <= 0:
        raise ValueError("Study A K and sampling seed must be positive integers")
    effective_sha256 = source_sha256 or raw_archive_sha256
    if (
        not isinstance(effective_sha256, str)
        or len(effective_sha256) != 64
        or any(character not in "0123456789abcdef" for character in effective_sha256)
    ):
        raise ValueError("Study A source SHA-256 is malformed")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("Study A source name must be non-empty")
    if (errors is None) == (scenarios is None):
        raise ValueError("Study A requires exactly one of natural errors or frozen scenarios")
    if errors is not None:
        error_rows = tuple(sorted(errors, key=lambda item: item.scene_id))
        if not error_rows or len({error.scene_id for error in error_rows}) != len(error_rows):
            raise ValueError("Study A natural errors must be non-empty and unique")
        scenario_rows = tuple(
            scenario for error in error_rows for scenario in build_study_a_scenarios(error)
        )
    else:
        scenario_rows = tuple(scenarios or ())
        if not scenario_rows or len({row.scenario_id for row in scenario_rows}) != len(
            scenario_rows
        ):
            raise ValueError("Study A frozen scenarios must be non-empty and unique")
    metadata = _trace_metadata(
        scenarios=scenario_rows,
        source_name=source_name,
        source_sha256=effective_sha256,
        additional_source_sha256=additional_source_sha256,
        k=k,
        sampling_seed=sampling_seed,
    )
    trace_path, completed = _load_or_create_trace(work_root, metadata)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenario_rows}
    _validate_resumed_rows(
        completed,
        scenario_by_id,
        k=k,
        sampling_seed=sampling_seed,
    )
    expected_keys = {
        (checkpoint, scenario.scenario_id)
        for checkpoint in CHECKPOINTS
        for scenario in scenario_rows
    }
    if set(completed) - expected_keys:
        raise RuntimeError("Study A resume trace contains unregistered checkpoint/scenario rows")
    total = len(expected_keys)
    for checkpoint in CHECKPOINTS:
        missing = [
            scenario
            for scenario in scenario_rows
            if (checkpoint, scenario.scenario_id) not in completed
        ]
        if not missing:
            continue
        model, processor = checkpoint_loader(checkpoint)
        freeze_inference_model(model)
        try:
            for scenario in missing:
                canonical_key = (checkpoint, f"{scenario.source_scene_id}::canonical")
                canonical_row = (
                    None if scenario.graph_axis == "canonical" else completed.get(canonical_key)
                )
                if scenario.graph_axis != "canonical" and canonical_row is None:
                    raise RuntimeError("Study A canonical row must precede its orbit transforms")
                row = _measure_scenario(
                    model=model,
                    processor=processor,
                    checkpoint=checkpoint,
                    scenario=scenario,
                    canonical_row=canonical_row,
                    k=k,
                    sampling_seed=sampling_seed,
                )
                key = (checkpoint, scenario.scenario_id)
                completed[key] = _append_trace(trace_path, row)
                if progress is not None:
                    progress(checkpoint, len(completed), total)
        finally:
            del model
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
    if set(completed) != expected_keys:
        raise RuntimeError("Study A execution did not close all checkpoint/scenario rows")
    ordered = tuple(
        completed[(checkpoint, scenario.scenario_id)]
        for checkpoint in CHECKPOINTS
        for scenario in scenario_rows
    )
    summary = summarize_study_a(
        ordered,
        source_name=source_name,
        source_sha256=effective_sha256,
        additional_source_sha256=additional_source_sha256,
        k=k,
        sampling_seed=sampling_seed,
    )
    _atomic_publish(
        output_root=output_root,
        trace_path=trace_path,
        ordered_rows=ordered,
        summary=summary,
    )
    return summary
