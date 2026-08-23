"""Paired, scene-clustered Study C3 factorial analysis without subjective gates."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from .config_runtime import load_config
from .io import read_json, read_jsonl, sha256_file, write_json_new
from .paths import (
    ACTION_AUDIT_MANIFEST,
    ACTION_AUDIT_SUMMARY,
    ANALYSIS_INTERVALS,
    ANALYSIS_MANIFEST,
    ANALYSIS_SUMMARY,
    ANALYSIS_TABLES,
    CONFIG,
    EXISTING_EVAL_MANIFEST,
    EXISTING_EVAL_SCENES,
    EXISTING_EVAL_SUMMARY,
    FACTORIAL_EVAL_MANIFEST,
    FACTORIAL_EVAL_SCENES,
    GRADIENT_MANIFEST,
    GRADIENT_SUMMARY,
    TRAINING_MANIFEST,
)
from .statistics import holm_adjust, paired_factorial_effects

PRIMARY_OUTCOMES = ("action_validity", "truth_purity_given_valid", "exact_recovery")


def _complete(path: object, expected: str) -> dict[str, object]:
    payload = read_json(path)  # type: ignore[arg-type]
    if payload.get("status") != expected:
        raise ValueError(f"Study C3 analysis prerequisite is incomplete: {path}")
    return payload


def preflight_analysis() -> dict[str, object]:
    load_config(CONFIG)
    action = _complete(ACTION_AUDIT_MANIFEST, "STUDY_C3_ACTION_CHANNEL_AUDIT_COMPLETE")
    existing = _complete(
        EXISTING_EVAL_MANIFEST,
        "STUDY_C3_EXISTING_CHECKPOINT_DECODER_INTERVENTION_COMPLETE",
    )
    _complete(TRAINING_MANIFEST, "STUDY_C3_FACTORIAL_TRAINING_COMPLETE")
    evaluation = _complete(
        FACTORIAL_EVAL_MANIFEST,
        "STUDY_C3_FACTORIAL_CHECKPOINT_EVALUATION_COMPLETE",
    )
    gradient = _complete(
        GRADIENT_MANIFEST,
        "STUDY_C3_SHARED_GRADIENT_VALIDITY_AUDIT_COMPLETE",
    )
    bindings = {
        "action_audit_manifest_sha256": sha256_file(ACTION_AUDIT_MANIFEST),
        "existing_decoder_manifest_sha256": sha256_file(EXISTING_EVAL_MANIFEST),
        "training_manifest_sha256": sha256_file(TRAINING_MANIFEST),
        "factorial_evaluation_manifest_sha256": sha256_file(FACTORIAL_EVAL_MANIFEST),
        "gradient_manifest_sha256": sha256_file(GRADIENT_MANIFEST),
    }
    if (
        action.get("summary_sha256") != sha256_file(ACTION_AUDIT_SUMMARY)
        or existing.get("summary_sha256") != sha256_file(EXISTING_EVAL_SUMMARY)
        or existing.get("per_scene_sha256") != sha256_file(EXISTING_EVAL_SCENES)
        or evaluation.get("per_scene_sha256") != sha256_file(FACTORIAL_EVAL_SCENES)
        or gradient.get("summary_sha256") != sha256_file(GRADIENT_SUMMARY)
    ):
        raise ValueError("Study C3 analysis input hashes drifted")
    return {
        "schema_version": 3,
        "status": "STUDY_C3_RESOLUTION_VALIDITY_ANALYSIS_PREFLIGHT_OK",
        **bindings,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 2026082503,
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "single_seed_mechanism_pilot": True,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


def _metric_counts(row: Mapping[str, object]) -> tuple[float, float, float]:
    rollouts = row.get("rollout_count")
    valid = row.get("action_validity")
    exact = row.get("exact_recovery")
    if (
        type(rollouts) is not int
        or rollouts <= 0
        or isinstance(valid, bool)
        or not isinstance(valid, (int, float))
        or isinstance(exact, bool)
        or not isinstance(exact, (int, float))
    ):
        raise ValueError("Study C3 scene metrics are malformed")
    return float(rollouts), float(valid) * rollouts, float(exact) * rollouts


def _pair_factorial_rows(
    rows: Sequence[Mapping[str, object]], *, outcome: str, checkpoint_step: int
) -> tuple[tuple[dict[str, object], ...], int]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    pair_ids: set[str] = set()
    for row in rows:
        if row.get("checkpoint_step") != checkpoint_step or row.get("verifier") not in {
            "answer",
            "state",
        }:
            continue
        decoder = row.get("eval_decoder")
        normalized = {"free_48": "free", "valid_world_fsa": "constrained"}.get(str(decoder))
        pair_id = row.get("pair_id")
        validity = row.get("validity_channel")
        verifier = row.get("verifier")
        if (
            not isinstance(pair_id, str)
            or validity not in {"binary", "lex"}
            or verifier not in {"answer", "state"}
            or normalized is None
        ):
            raise ValueError("Study C3 factorial evaluation metadata is malformed")
        pair_ids.add(pair_id)
        grouped[(pair_id, str(verifier), str(validity), normalized)].append(row)
    candidate: list[dict[str, object]] = []
    valid_pairs: set[str] = set()
    required_cells = {
        (verifier, validity, decoder)
        for verifier in ("answer", "state")
        for validity in ("binary", "lex")
        for decoder in ("free", "constrained")
    }
    for pair_id in sorted(pair_ids):
        cells = {
            (verifier, validity, decoder): grouped.get((pair_id, verifier, validity, decoder), [])
            for verifier, validity, decoder in required_cells
        }
        if any(len(cell) != 2 for cell in cells.values()):
            continue
        converted: list[dict[str, object]] = []
        complete = True
        for (verifier, validity, decoder), cell in cells.items():
            total = valid = exact = 0.0
            for scene in cell:
                scene_total, scene_valid, scene_exact = _metric_counts(scene)
                total += scene_total
                valid += scene_valid
                exact += scene_exact
            value: float | None
            if outcome == "action_validity":
                value = valid / total
            elif outcome == "exact_recovery":
                value = exact / total
            elif outcome == "truth_purity_given_valid":
                value = None if valid == 0.0 else exact / valid
            else:
                raise ValueError(f"unregistered Study C3 primary outcome: {outcome}")
            if value is None:
                complete = False
                break
            converted.append(
                {
                    "pair_id": pair_id,
                    "verifier": verifier,
                    "validity_channel": validity,
                    "eval_decoder": decoder,
                    outcome: value,
                }
            )
        if complete:
            valid_pairs.add(pair_id)
            candidate.extend(converted)
    return tuple(candidate), len(pair_ids - valid_pairs)


def _trajectory_tables(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def summarize_cell(cell: Sequence[Mapping[str, object]]) -> dict[str, object]:
        if not cell:
            raise ValueError("Study C3 trajectory lacks a registered evaluation cell")
        total = valid = exact = answer = shortcut = 0.0
        for row in cell:
            rollouts, row_valid, row_exact = _metric_counts(row)
            total += rollouts
            valid += row_valid
            exact += row_exact
            answer += float(row["answer_correctness"]) * rollouts
            shortcut += float(row["shortcut_rate"]) * rollouts
        return {
            "scene_count": len(cell),
            "rollout_count": int(total),
            "action_validity": valid / total,
            "truth_purity_given_valid": None if valid == 0 else exact / valid,
            "exact_recovery": exact / total,
            "answer_correctness": answer / total,
            "shortcut_rate": shortcut / total,
        }

    initialization = {
        f"B3-initialization:{decoder}": summarize_cell(
            [
                row
                for row in rows
                if row.get("checkpoint_step") == 0
                and row.get("checkpoint_id") == "B3-initialization"
                and row.get("eval_decoder") == decoder
            ]
        )
        for decoder in ("free_48", "valid_world_fsa")
    }
    tables: dict[str, object] = {"0": initialization}
    for step in (48, 96, 144, 192):
        cells: dict[str, object] = {}
        selected = [row for row in rows if row.get("checkpoint_step") == step]
        for verifier in ("answer", "state"):
            for validity in ("binary", "lex"):
                for decoder in ("free_48", "valid_world_fsa"):
                    cell = [
                        row
                        for row in selected
                        if row.get("verifier") == verifier
                        and row.get("validity_channel") == validity
                        and row.get("eval_decoder") == decoder
                    ]
                    if not cell:
                        raise ValueError("Study C3 trajectory lacks a registered evaluation cell")
                    cells[f"{verifier}:{validity}:{decoder}"] = summarize_cell(cell)
        tables[str(step)] = cells
    return tables


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def _bootstrap_named_contrasts(
    values: Mapping[str, Sequence[float]], *, resamples: int, seed: int
) -> dict[str, object]:
    rng = random.Random(seed)
    result: dict[str, object] = {}
    for name, pair_values in values.items():
        if not pair_values:
            continue
        estimate = sum(pair_values) / len(pair_values)
        draws = [
            sum(rng.choice(pair_values) for _ in pair_values) / len(pair_values)
            for _ in range(resamples)
        ]
        signs = [
            sum((1 if rng.random() < 0.5 else -1) * value for value in pair_values)
            / len(pair_values)
            for _ in range(resamples)
        ]
        result[name] = {
            "estimate": estimate,
            "bootstrap_95_ci": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
            "sign_flip_p_value": (1 + sum(abs(value) >= abs(estimate) for value in signs))
            / (resamples + 1),
        }
    return result


def _existing_decoder_effects(
    rows: Sequence[Mapping[str, object]], *, outcome: str, resamples: int, seed: int
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    pair_ids: set[str] = set()
    for row in rows:
        pair_id = row.get("pair_id")
        verifier = row.get("verifier")
        decoder = row.get("eval_decoder")
        if (
            not isinstance(pair_id, str)
            or verifier not in {"answer", "state"}
            or decoder not in {"free_16", "free_48", "valid_world_fsa"}
        ):
            raise ValueError("Study C3 existing-checkpoint scene metadata is malformed")
        pair_ids.add(pair_id)
        grouped[(pair_id, str(verifier), str(decoder))].append(row)
    contrasts: dict[str, list[float]] = defaultdict(list)
    excluded = 0
    for pair_id in sorted(pair_ids):
        cells: dict[tuple[str, str], float] = {}
        complete = True
        for verifier in ("answer", "state"):
            for decoder in ("free_16", "free_48", "valid_world_fsa"):
                cell = grouped.get((pair_id, verifier, decoder), [])
                if len(cell) != 2:
                    complete = False
                    break
                total = valid = exact = 0.0
                for scene in cell:
                    scene_total, scene_valid, scene_exact = _metric_counts(scene)
                    total += scene_total
                    valid += scene_valid
                    exact += scene_exact
                if outcome == "action_validity":
                    value = valid / total
                elif outcome == "exact_recovery":
                    value = exact / total
                elif outcome == "truth_purity_given_valid" and valid > 0:
                    value = exact / valid
                else:
                    complete = False
                    break
                cells[(verifier, decoder)] = value
            if not complete:
                break
        if not complete:
            excluded += 1
            continue
        for verifier in ("answer", "state"):
            contrasts[f"{verifier}:free48_minus_free16"].append(
                cells[(verifier, "free_48")] - cells[(verifier, "free_16")]
            )
            contrasts[f"{verifier}:constrained_minus_free48"].append(
                cells[(verifier, "valid_world_fsa")] - cells[(verifier, "free_48")]
            )
        for decoder in ("free_16", "free_48", "valid_world_fsa"):
            contrasts[f"state_minus_answer:{decoder}"].append(
                cells[("state", decoder)] - cells[("answer", decoder)]
            )
        contrasts["verifier_x_free48_vs_free16"].append(
            cells[("state", "free_48")]
            - cells[("state", "free_16")]
            - cells[("answer", "free_48")]
            + cells[("answer", "free_16")]
        )
        contrasts["verifier_x_constrained_vs_free48"].append(
            cells[("state", "valid_world_fsa")]
            - cells[("state", "free_48")]
            - cells[("answer", "valid_world_fsa")]
            + cells[("answer", "free_48")]
        )
    measured = _bootstrap_named_contrasts(contrasts, resamples=resamples, seed=seed)
    return {
        "outcome": outcome,
        "identifiable": bool(measured),
        "pair_count": len(pair_ids) - excluded,
        "excluded_pair_count": excluded,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "contrasts": measured,
    }


def run_analysis() -> dict[str, object]:
    preflight = preflight_analysis()
    per_scene = read_jsonl(FACTORIAL_EVAL_SCENES)
    existing_per_scene = read_jsonl(EXISTING_EVAL_SCENES)
    intervals: dict[str, object] = {}
    p_values: dict[str, float] = {}
    excluded: dict[str, int] = {}
    for index, outcome in enumerate(PRIMARY_OUTCOMES):
        paired, excluded_count = _pair_factorial_rows(
            per_scene, outcome=outcome, checkpoint_step=192
        )
        excluded[outcome] = excluded_count
        if not paired:
            intervals[outcome] = {
                "identifiable": False,
                "pair_count": 0,
                "excluded_pair_count": excluded_count,
            }
            continue
        measured = paired_factorial_effects(
            paired,
            outcome=outcome,
            resamples=int(preflight["bootstrap_resamples"]),
            seed=int(preflight["bootstrap_seed"]) + index,
        )
        measured["identifiable"] = True
        measured["excluded_pair_count"] = excluded_count
        intervals[outcome] = measured
        for effect, details in measured["effects"].items():  # type: ignore[union-attr]
            p_values[f"{outcome}:{effect}"] = float(details["sign_flip_p_value"])
    adjusted = holm_adjust(p_values) if p_values else {}
    for outcome, measured in intervals.items():
        if not isinstance(measured, Mapping) or measured.get("identifiable") is not True:
            continue
        effects = measured.get("effects")
        assert isinstance(effects, Mapping)
        for effect, details in effects.items():
            assert isinstance(details, dict)
            details["holm_adjusted_p_value"] = adjusted[f"{outcome}:{effect}"]
    existing_intervals = {
        outcome: _existing_decoder_effects(
            existing_per_scene,
            outcome=outcome,
            resamples=int(preflight["bootstrap_resamples"]),
            seed=int(preflight["bootstrap_seed"]) + 100 + index,
        )
        for index, outcome in enumerate(PRIMARY_OUTCOMES)
    }
    existing_p_values = {
        f"{outcome}:{contrast}": float(details["sign_flip_p_value"])
        for outcome, result in existing_intervals.items()
        for contrast, details in result["contrasts"].items()  # type: ignore[union-attr]
    }
    existing_adjusted = holm_adjust(existing_p_values) if existing_p_values else {}
    for outcome, result in existing_intervals.items():
        contrasts = result["contrasts"]
        assert isinstance(contrasts, Mapping)
        for contrast, details in contrasts.items():
            assert isinstance(details, dict)
            details["holm_adjusted_p_value"] = existing_adjusted[f"{outcome}:{contrast}"]
    tables = {
        "schema_version": 3,
        "status": "STUDY_C3_RESOLUTION_VALIDITY_TABLES_COMPLETE",
        "checkpoint_trajectories": _trajectory_tables(per_scene),
        "action_channel_audit": read_json(ACTION_AUDIT_SUMMARY),
        "existing_checkpoint_decoder_intervention": read_json(EXISTING_EVAL_SUMMARY),
        "shared_gradient_validity_audit": read_json(GRADIENT_SUMMARY),
    }
    interval_payload = {
        "schema_version": 3,
        "status": "STUDY_C3_RESOLUTION_VALIDITY_INTERVALS_COMPLETE",
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "registered_effects": [
            "verifier",
            "validity_channel",
            "verifier_x_validity",
            "decoder",
            "verifier_x_decoder",
        ],
        "holm_family_size": len(p_values),
        "outcomes": intervals,
        "existing_checkpoint_decoder_intervention": {
            "holm_family_size": len(existing_p_values),
            "outcomes": existing_intervals,
        },
    }
    summary = {
        "schema_version": 3,
        "status": "STUDY_C3_RESOLUTION_VALIDITY_ANALYSIS_COMPLETE",
        "primary_outcomes": list(PRIMARY_OUTCOMES),
        "single_seed_mechanism_pilot": True,
        "pair_count_registered": 88,
        "excluded_pair_count_by_outcome": excluded,
        "subjective_gate_reported": False,
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }
    write_json_new(ANALYSIS_TABLES, tables)
    write_json_new(ANALYSIS_INTERVALS, interval_payload)
    write_json_new(ANALYSIS_SUMMARY, summary)
    manifest = {
        **preflight,
        "status": "STUDY_C3_RESOLUTION_VALIDITY_ANALYSIS_COMPLETE",
        "summary_sha256": sha256_file(ANALYSIS_SUMMARY),
        "summary_tables_sha256": sha256_file(ANALYSIS_TABLES),
        "confidence_intervals_sha256": sha256_file(ANALYSIS_INTERVALS),
        "subjective_gate_reported": False,
    }
    write_json_new(ANALYSIS_MANIFEST, manifest)
    return manifest


__all__ = ["PRIMARY_OUTCOMES", "preflight_analysis", "run_analysis"]
