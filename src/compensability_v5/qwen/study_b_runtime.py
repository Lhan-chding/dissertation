"""Single-GPU Study-B training and text-world evaluation runtime.

The orchestration layer is dependency-light and callback-testable.  Torch,
Transformers, PEFT, and Datasets are imported only by :class:`QwenStudyBBackend`
after the CLI has applied its explicit acknowledgement and offline gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from compensability_v5.qwen.study_b_backend import (
    QwenStudyBBackend,
    require_offline_environment,
    verify_runtime_package_lock,
)
from compensability_v5.qwen.study_b_inputs import (
    ARMS,
    CANONICAL_BATCH_SIZE,
    MODEL_SNAPSHOT_SHA256,
    PILOT_SEED,
    REGISTERED_AXES,
    RELATIONAL_FAMILIES,
    StudyBError,
    _text,
    evaluation_rows_from_study_a,
    parse_world,
    unified_world_prompt,
    validate_evaluation_rows,
    validate_support_package,
)
from compensability_v5.qwen.study_b_metrics import paired_bootstrap_contrasts


class StudyBBackend(Protocol):
    """Minimal backend boundary used by both the real Qwen runner and CPU tests."""

    def load_base(self, *, arm: str, expected_model_sha256: str) -> Mapping[str, object]: ...

    def train(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        budget: dict[str, object],
        seed: int,
        output: Path,
    ) -> Mapping[str, object]: ...

    def evaluate(
        self,
        *,
        session: Mapping[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        prompts: tuple[str, ...],
        seed: int,
        output: Path,
    ) -> Iterable[Mapping[str, object]]: ...

    def release(self, session: Mapping[str, object]) -> None: ...


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StudyBError(f"unsafe or missing regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    """Hash a complete adapter tree including paths, sizes, and file hashes."""

    if path.is_symlink() or not path.is_dir():
        raise StudyBError(f"adapter tree is missing or unsafe: {path}")
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise StudyBError("adapter tree must contain at least one regular file")
    digest = hashlib.sha256()
    for candidate in files:
        if candidate.is_symlink():
            raise StudyBError(f"adapter tree contains a symlink: {candidate}")
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(candidate).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metric_block(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(rows)
    relational = [row for row in rows if row["family"] in RELATIONAL_FAMILIES]
    return {
        "count": count,
        "parsed_rate": sum(row["parsed_world"] is not None for row in rows) / count,
        "exact_world_rate": sum(bool(row["exact_world"]) for row in rows) / count,
        "genuine_recovery_rate": sum(bool(row["genuine_recovery"]) for row in rows) / count,
        "observation_copy_rate": sum(bool(row["observation_copy"]) for row in rows) / count,
        "relational_count": len(relational),
        "relational_genuine_recovery_rate": (
            sum(bool(row["genuine_recovery"]) for row in relational) / len(relational)
            if relational
            else None
        ),
        "mean_candidate_margin": (
            sum(
                float(row["candidate_margin"])
                for row in rows
                if row.get("candidate_margin") is not None
            )
            / sum(row.get("candidate_margin") is not None for row in rows)
            if any(row.get("candidate_margin") is not None for row in rows)
            else None
        ),
    }


def summarize_evaluations(
    evaluation_rows: Iterable[Mapping[str, object]],
    outputs: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    rows = validate_evaluation_rows(evaluation_rows)
    output_rows = tuple(outputs)
    if len(output_rows) != len(rows):
        raise StudyBError("backend must return exactly one evaluation output per scene")
    by_scene: dict[str, Mapping[str, object]] = {}
    for output in output_rows:
        if not isinstance(output, Mapping):
            raise StudyBError("evaluation output must be a mapping")
        scene_id = _text(output.get("scene_id"), "evaluation output scene_id")
        if scene_id in by_scene:
            raise StudyBError("backend returned duplicate evaluation scene_id")
        if set(output) - {"scene_id", "completion", "candidate_margin"}:
            raise StudyBError("evaluation output contains unregistered fields")
        completion = _text(output.get("completion"), "evaluation completion")
        margin = output.get("candidate_margin")
        if margin is not None and (
            not isinstance(margin, (int, float))
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
        ):
            raise StudyBError("candidate_margin must be finite when supplied")
        by_scene[scene_id] = {"completion": completion, "candidate_margin": margin}
    if set(by_scene) != {str(row["scene_id"]) for row in rows}:
        raise StudyBError("backend evaluation scene set differs from the frozen evaluation set")

    enriched: list[dict[str, object]] = []
    by_axis: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        output = by_scene[str(row["scene_id"])]
        parsed = parse_world(output["completion"])
        truth = tuple(row["truth"])
        observed = tuple(row["natural_observation"])
        exact = parsed == truth
        enriched_row = {
            "scene_id": row["scene_id"],
            "semantic_scene_id": row["semantic_scene_id"],
            "family": row["family"],
            "graph_axis": row.get("graph_axis"),
            "evaluation_axes": row["evaluation_axes"],
            "truth": list(truth),
            "natural_observation": list(observed),
            "prompt_sha256": canonical_sha256(unified_world_prompt(row)),
            "completion": output["completion"],
            "parsed_world": list(parsed) if parsed is not None else None,
            "exact_world": exact,
            "genuine_recovery": exact and observed != truth,
            "observation_copy": parsed == observed,
            "candidate_margin": output.get("candidate_margin"),
        }
        enriched.append(enriched_row)
        by_family[str(row["family"])].append(enriched_row)
        for axis in row["evaluation_axes"]:
            by_axis[str(axis)].append(enriched_row)
            if axis != "iid":
                by_axis["structural_ood"].append(enriched_row)
    summary = {
        "schema_version": 1,
        "statistical_unit": "semantic_scene",
        "prompt_protocol": "unified_text_world_v1",
        "overall": _metric_block(enriched),
        "by_axis": {name: _metric_block(group) for name, group in sorted(by_axis.items())},
        "by_family": {name: _metric_block(group) for name, group in sorted(by_family.items())},
    }
    return summary, tuple(enriched)


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            )


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise StudyBError(f"missing immutable Study-B artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StudyBError(f"Study-B JSON artifact is not a mapping: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise StudyBError(f"missing immutable Study-B JSONL artifact: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise StudyBError(f"malformed Study-B JSONL line {line_number}: {path}") from error
        if not isinstance(value, dict):
            raise StudyBError(f"Study-B JSONL row is not a mapping: {path}")
        rows.append(value)
    if not rows:
        raise StudyBError(f"Study-B JSONL artifact is empty: {path}")
    return tuple(rows)


def _completed_arm_result(
    path: Path,
    *,
    arm: str,
    run_signature: str,
    seed: int,
    model_sha256: str,
    budget: Mapping[str, object],
    evaluation_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result = _read_json(path / "result.json")
    expected_fields = {
        "schema_version",
        "status",
        "arm",
        "seed",
        "run_signature",
        "base_model_sha256",
        "base_load_token",
        "budget",
        "adapter_tree_sha256",
        "training_metrics",
        "training_log_sha256",
        "observed_target_tokens",
        "trainable_manifest",
        "frozen_hashes",
        "evaluation_metrics",
        "evaluation_rows_sha256",
    }
    if set(result) != expected_fields:
        raise StudyBError(f"completed {arm} result has a malformed closed schema")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "STUDY_B_ARM_COMPLETE"
        or result.get("arm") != arm
        or result.get("seed") != seed
        or result.get("run_signature") != run_signature
        or result.get("base_model_sha256") != model_sha256
        or result.get("budget") != budget
        or result.get("observed_target_tokens") != budget["target_tokens"]
    ):
        raise StudyBError(f"completed {arm} artifact does not match this run")
    metrics = result.get("training_metrics")
    if not isinstance(metrics, Mapping) or metrics.get("train_steps") != budget["steps"]:
        raise StudyBError(f"completed {arm} training metrics differ from its budget")
    expected = result.get("adapter_tree_sha256")
    if expected != tree_sha256(path / "final_adapter"):
        raise StudyBError(f"completed {arm} adapter tree hash changed")
    evaluation_path = path / "evaluation_rows.jsonl"
    if result.get("evaluation_rows_sha256") != sha256_file(evaluation_path):
        raise StudyBError(f"completed {arm} evaluation row log changed")
    evidence = _read_jsonl(evaluation_path)
    outputs = tuple(
        {
            "scene_id": row.get("scene_id"),
            "completion": row.get("completion"),
            "candidate_margin": row.get("candidate_margin"),
        }
        for row in evidence
    )
    rebuilt_metrics, rebuilt_evidence = summarize_evaluations(evaluation_rows, outputs)
    if evidence != rebuilt_evidence:
        raise StudyBError(f"completed {arm} evaluation evidence drifted")
    if result.get("evaluation_metrics") != rebuilt_metrics:
        raise StudyBError(f"completed {arm} evaluation metrics drifted")
    training_log = path / "training_log.json"
    if training_log.is_symlink() or not training_log.is_file():
        raise StudyBError(f"completed {arm} training log is missing or unsafe")
    if result.get("training_log_sha256") != sha256_file(training_log):
        raise StudyBError(f"completed {arm} training log changed")
    _validate_freeze_evidence(result.get("trainable_manifest"), result.get("frozen_hashes"))
    return result


def _axis_rate(result: Mapping[str, object], axis: str) -> float:
    evaluation = result.get("evaluation_metrics")
    if not isinstance(evaluation, Mapping):
        raise StudyBError("arm result lacks evaluation_metrics")
    by_axis = evaluation.get("by_axis")
    if not isinstance(by_axis, Mapping) or not isinstance(by_axis.get(axis), Mapping):
        raise StudyBError(f"arm result lacks registered {axis} evaluation")
    rate = by_axis[axis].get("exact_world_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool):
        raise StudyBError(f"arm result has invalid {axis} exact-world rate")
    return float(rate)


def _primary_contrasts(
    results: Mapping[str, Mapping[str, object]],
    evaluation_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    b2, b3 = results["B2"], results["B3"]
    rates = {
        "B3_minus_B2": {
            "iid_exact_world_rate": _axis_rate(b3, "iid") - _axis_rate(b2, "iid"),
            "variable_permutation_exact_world_rate": _axis_rate(b3, "variable_permutation")
            - _axis_rate(b2, "variable_permutation"),
            "error_position_exact_world_rate": _axis_rate(b3, "error_position")
            - _axis_rate(b2, "error_position"),
            "fact_order_exact_world_rate": _axis_rate(b3, "fact_order")
            - _axis_rate(b2, "fact_order"),
            "constraint_graph_exact_world_rate": _axis_rate(b3, "constraint_graph")
            - _axis_rate(b2, "constraint_graph"),
            "structural_ood_exact_world_rate": _axis_rate(b3, "structural_ood")
            - _axis_rate(b2, "structural_ood"),
        }
    }
    rates["paired_inference"] = paired_bootstrap_contrasts(
        evaluation_rows["B2"], evaluation_rows["B3"]
    )
    return rates


def _validate_freeze_evidence(trainable: object, frozen: object) -> None:
    if not isinstance(trainable, Mapping) or any(
        trainable.get(field) is not True
        for field in ("vision_frozen", "merger_frozen", "base_language_frozen")
    ):
        raise StudyBError("Study B requires vision and merger frozen with language Base")
    target_modules = trainable.get("target_modules")
    parameter_names = trainable.get("trainable_parameter_names")
    if (
        not isinstance(target_modules, Sequence)
        or isinstance(target_modules, (str, bytes))
        or not target_modules
        or not isinstance(parameter_names, Sequence)
        or isinstance(parameter_names, (str, bytes))
        or not parameter_names
        or any(
            not isinstance(name, str)
            or "lora_" not in name
            or ".language_model." not in name
            or ".visual." in name
            or ".merger" in name
            for name in parameter_names
        )
    ):
        raise StudyBError("Study B trainable manifest is not language-LoRA-only")
    if not isinstance(frozen, Mapping):
        raise StudyBError("backend must return frozen component hashes")
    hashes = frozen.get("sha256_by_component")
    required = ("vision", "merger", "language_base")
    if not isinstance(hashes, Mapping) or not set(required).issubset(hashes):
        raise StudyBError("frozen evidence lacks vision, merger, or language Base hashes")
    if any(
        not isinstance(hashes[name], str) or re.fullmatch(r"[0-9a-f]{64}", hashes[name]) is None
        for name in required
    ):
        raise StudyBError("frozen component evidence contains an invalid SHA-256")


def run_study_b(
    *,
    support_package: Mapping[str, object],
    evaluation_rows: Iterable[Mapping[str, object]],
    output: Path,
    backend: StudyBBackend,
    expected_model_sha256: str,
    seed: int = PILOT_SEED,
    resume: bool = False,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Execute the immutable one-seed B0--B3 training/evaluation loop."""

    if seed != PILOT_SEED:
        raise StudyBError(f"Study B pilot seed must be exactly {PILOT_SEED}")
    if expected_model_sha256 != MODEL_SNAPSHOT_SHA256:
        raise StudyBError("Study B model SHA differs from the frozen Qwen v4 snapshot")
    validated_support = validate_support_package(support_package)
    eval_rows = validate_evaluation_rows(evaluation_rows)
    support_semantic_ids = {
        str(row["semantic_scene_id"]) for row in validated_support["arms"]["B0"]
    }
    evaluation_semantic_ids = {str(row["semantic_scene_id"]) for row in eval_rows}
    overlap = support_semantic_ids & evaluation_semantic_ids
    if overlap:
        raise StudyBError(
            f"Study B training/evaluation semantic scenes overlap: {sorted(overlap)[:3]}"
        )
    prompts = tuple(unified_world_prompt(row) for row in eval_rows)
    output = Path(output)
    if output.is_symlink():
        raise StudyBError("Study B output must not be a symlink")

    run_signature_payload = {
        "schema_version": 1,
        "study": "B_budget_matched_structural_support_lora",
        "seed": seed,
        "model_snapshot_sha256": expected_model_sha256,
        "support_package_sha256": canonical_sha256(support_package),
        "evaluation_rows_sha256": canonical_sha256(eval_rows),
        "arms": list(ARMS),
        "prompt_protocol": "unified_text_world_v1",
        "training_device": "single_4090",
        "per_device_train_batch_size": CANONICAL_BATCH_SIZE,
        "fixed_sequence_padding": True,
        "provenance": dict(provenance or {}),
    }
    run_signature = canonical_sha256(run_signature_payload)
    manifest = {**run_signature_payload, "run_signature": run_signature}
    manifest_path = output / "run_manifest.json"
    completed_path = output / "completed.json"
    if output.exists():
        if not resume:
            raise FileExistsError("Study B output already exists; use --resume only for this run")
        if not output.is_dir() or _read_json(manifest_path) != manifest:
            raise StudyBError("resume manifest differs from the requested Study B run")
        if completed_path.is_file():
            completed = _read_json(completed_path)
            completed_results: dict[str, dict[str, object]] = {}
            completed_evaluation_rows: dict[str, tuple[dict[str, object], ...]] = {}
            for arm in ARMS:
                completed_results[arm] = _completed_arm_result(
                    output / "arms" / arm,
                    arm=arm,
                    run_signature=run_signature,
                    seed=seed,
                    model_sha256=expected_model_sha256,
                    budget=validated_support["budgets"][arm],
                    evaluation_rows=eval_rows,
                )
                completed_evaluation_rows[arm] = _read_jsonl(
                    output / "arms" / arm / "evaluation_rows.jsonl"
                )
            expected_tokens = {
                arm: _text(result.get("base_load_token"), f"{arm}.base_load_token")
                for arm, result in completed_results.items()
            }
            if len(set(expected_tokens.values())) != len(ARMS):
                raise StudyBError("completed Study B arms did not use fresh Base sessions")
            expected_contrasts = _primary_contrasts(completed_results, completed_evaluation_rows)
            if (
                completed.get("status") != "STUDY_B_SINGLE_SEED_COMPLETE"
                or completed.get("seed") != seed
                or completed.get("model_snapshot_sha256") != expected_model_sha256
                or completed.get("run_signature") != run_signature
                or completed.get("run_manifest_sha256") != canonical_sha256(manifest)
                or completed.get("arm_results") != completed_results
                or completed.get("base_load_tokens") != expected_tokens
                or completed.get("primary_contrasts") != expected_contrasts
                or completed.get("stop_signal")
                != expected_contrasts["paired_inference"]["stop_signal"]
            ):
                raise StudyBError("completed Study B payload or referenced arm evidence drifted")
            return completed
    else:
        if resume:
            raise StudyBError("cannot resume a Study B output that does not exist")
        output.mkdir(parents=True)
        _write_json_new(manifest_path, manifest)

    arms_root = output / "arms"
    attempts_root = output / "attempts"
    arms_root.mkdir(exist_ok=True)
    attempts_root.mkdir(exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    load_tokens: dict[str, str] = {}
    observed_tokens: set[str] = set()
    observed_flops_reference: float | None = None
    for arm in ARMS:
        final_arm = arms_root / arm
        if final_arm.exists():
            result = _completed_arm_result(
                final_arm,
                arm=arm,
                run_signature=run_signature,
                seed=seed,
                model_sha256=expected_model_sha256,
                budget=validated_support["budgets"][arm],
                evaluation_rows=eval_rows,
            )
            results[arm] = result
            token = _text(result.get("base_load_token"), f"{arm}.base_load_token")
            if token in observed_tokens:
                raise StudyBError("each arm must reload a distinct fresh Base session")
            load_tokens[arm] = token
            observed_tokens.add(token)
            metrics = result.get("training_metrics")
            if not isinstance(metrics, Mapping):
                raise StudyBError(f"completed {arm} result lacks training metrics")
            observed_flops = metrics.get("observed_total_flos")
            if (
                not isinstance(observed_flops, (int, float))
                or isinstance(observed_flops, bool)
                or not math.isfinite(float(observed_flops))
                or float(observed_flops) <= 0
            ):
                raise StudyBError(f"completed {arm} lacks positive observed FLOPs")
            if observed_flops_reference is None:
                observed_flops_reference = float(observed_flops)
            elif float(observed_flops) != observed_flops_reference:
                raise StudyBError("observed training FLOPs differ across Study B arms")
            continue
        attempt = Path(tempfile.mkdtemp(prefix=f"{arm}-", dir=attempts_root))
        session: Mapping[str, object] | None = None
        try:
            session = backend.load_base(arm=arm, expected_model_sha256=expected_model_sha256)
            if not isinstance(session, Mapping):
                raise StudyBError("backend load_base must return a mapping")
            if session.get("model_sha256") != expected_model_sha256:
                raise StudyBError(f"{arm} backend loaded a different Base snapshot")
            load_token = _text(session.get("load_token"), f"{arm}.load_token")
            if load_token in observed_tokens:
                raise StudyBError("each arm must reload a distinct fresh Base session")
            observed_tokens.add(load_token)
            load_tokens[arm] = load_token
            rows = validated_support["arms"][arm]
            budget = validated_support["budgets"][arm]
            training = backend.train(
                session=session,
                arm=arm,
                rows=rows,
                budget=budget,
                seed=seed,
                output=attempt,
            )
            if not isinstance(training, Mapping):
                raise StudyBError("backend train must return a mapping")
            if training.get("observed_target_tokens") != budget["target_tokens"]:
                raise StudyBError(
                    f"{arm} actual tokenizer completion count differs from frozen budget"
                )
            training_metrics = training.get("training_metrics")
            if not isinstance(training_metrics, Mapping):
                raise StudyBError(f"{arm} backend must return training metrics")
            if training_metrics.get("train_steps") != budget["steps"]:
                raise StudyBError(f"{arm} did not execute the exact optimizer-step budget")
            observed_flops = training_metrics.get("observed_total_flos")
            if (
                not isinstance(observed_flops, (int, float))
                or isinstance(observed_flops, bool)
                or not math.isfinite(float(observed_flops))
                or float(observed_flops) <= 0
            ):
                raise StudyBError(f"{arm} backend must report positive observed FLOPs")
            if observed_flops_reference is None:
                observed_flops_reference = float(observed_flops)
            elif float(observed_flops) != observed_flops_reference:
                raise StudyBError("observed training FLOPs differ across Study B arms")
            manifest_value = training.get("trainable_manifest")
            frozen_hashes = training.get("frozen_hashes")
            _validate_freeze_evidence(manifest_value, frozen_hashes)
            assert isinstance(manifest_value, Mapping)
            assert isinstance(frozen_hashes, Mapping)
            adapter = Path(_text(training.get("adapter_path"), "adapter_path"))
            if adapter.resolve() != (attempt / "final_adapter").resolve():
                raise StudyBError(
                    "backend adapter must be written to the assigned attempt directory"
                )
            adapter_hash = tree_sha256(adapter)
            if not (attempt / "training_log.json").is_file():
                raise StudyBError(f"{arm} backend did not write the required training log")
            output_rows = tuple(
                backend.evaluate(
                    session=session,
                    arm=arm,
                    rows=eval_rows,
                    prompts=prompts,
                    seed=seed,
                    output=attempt,
                )
            )
            evaluation_metrics, enriched = summarize_evaluations(eval_rows, output_rows)
            evaluation_path = attempt / "evaluation_rows.jsonl"
            if evaluation_path.exists() or evaluation_path.is_symlink():
                raise StudyBError("backend must not pre-create the registered evaluation row log")
            _write_jsonl_new(evaluation_path, enriched)
            result = {
                "schema_version": 1,
                "status": "STUDY_B_ARM_COMPLETE",
                "arm": arm,
                "seed": seed,
                "run_signature": run_signature,
                "base_model_sha256": expected_model_sha256,
                "base_load_token": load_token,
                "budget": budget,
                "adapter_tree_sha256": adapter_hash,
                "training_metrics": dict(training_metrics),
                "training_log_sha256": sha256_file(attempt / "training_log.json"),
                "observed_target_tokens": training["observed_target_tokens"],
                "trainable_manifest": dict(manifest_value),
                "frozen_hashes": dict(frozen_hashes),
                "evaluation_metrics": evaluation_metrics,
                "evaluation_rows_sha256": sha256_file(evaluation_path),
            }
            _write_json_new(attempt / "result.json", result)
            attempt.rename(final_arm)
            results[arm] = result
        finally:
            if session is not None:
                backend.release(session)

    evidence_rows = {arm: _read_jsonl(arms_root / arm / "evaluation_rows.jsonl") for arm in ARMS}
    if len(set(load_tokens.values())) != len(ARMS):
        raise StudyBError("each arm must reload a distinct fresh Base session")
    contrasts = _primary_contrasts(results, evidence_rows)
    completed = {
        "schema_version": 1,
        "status": "STUDY_B_SINGLE_SEED_COMPLETE",
        "seed": seed,
        "model_snapshot_sha256": expected_model_sha256,
        "run_signature": run_signature,
        "run_manifest_sha256": canonical_sha256(manifest),
        "base_load_tokens": load_tokens,
        "arm_results": results,
        "primary_contrasts": contrasts,
        "stop_signal": contrasts["paired_inference"]["stop_signal"],
        "evidence_class": "single_seed_pilot",
    }
    _write_json_new(completed_path, completed)
    return completed


__all__ = [
    "ARMS",
    "MODEL_SNAPSHOT_SHA256",
    "PILOT_SEED",
    "REGISTERED_AXES",
    "QwenStudyBBackend",
    "StudyBBackend",
    "StudyBError",
    "canonical_sha256",
    "evaluation_rows_from_study_a",
    "parse_world",
    "require_offline_environment",
    "run_study_b",
    "sha256_file",
    "summarize_evaluations",
    "tree_sha256",
    "unified_world_prompt",
    "validate_evaluation_rows",
    "validate_support_package",
    "verify_runtime_package_lock",
]
