"""Paid, different-scene diagnostic for direct measurements and actual full AD.

Only development seed41001/X_BASE/step32 is registered. No parameter response
head trained on the fixed observation panel is applied to new prompts. Primary
comparisons use six semantic group means over all36 new prompts; raw prompt
actions, scores and reference statistics are retained separately.
"""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from ..modeling_v3.io import canonical_hash
from .data_adapter import CONTRASTS


def select_cross_prompt_inputs(panels, banks):
    probes = panels.get("cross_prompt", [])
    ids = [p["prompt_id"] for p in probes]
    scenes = {p["base_scene_id"] for p in probes}
    groups = {(p["family"], p["interface"]) for p in probes}
    if len(probes) != 36 or len(set(ids)) != 36 or len(scenes) != 18 or len(groups) != 6:
        raise ValueError("Frozen different-scene panel requires36 unique prompts and six groups")
    for name in ("observation", "signature"):
        other = panels.get(name, [])
        if scenes & {p["base_scene_id"] for p in other} or set(ids) & {
            p["prompt_id"] for p in other
        }:
            raise ValueError("Cross-prompt panel must be scene/prompt disjoint")
    if any(sum((p["family"], p["interface"]) == g for p in probes) != 6 for g in groups):
        raise ValueError("Each frozen semantic group requires six new prompts")
    eligible = []
    for bank in banks:
        if bank["role"] != "query":
            continue
        policies = bank["policies"]
        alias = (
            policies["joint_1"]["inference_fingerprint"]
            == policies["joint_0"]["inference_fingerprint"]
        )
        if bank["contrasts"][CONTRASTS[0][0]]["exact_inference_alias"] is not alias:
            raise ValueError("Nonalias selection must match exact full inference fingerprint")
        if not alias:
            eligible.append(bank)
    if len(eligible) < 2:
        raise ValueError("Two actual nonalias query banks required before cross-prompt acquisition")
    return list(probes), eligible[:2]


def contract_full_gradient(data, layout, gradient, candidate_index, baseline_index):
    if (
        gradient["parameter_order"] != list(layout)
        or gradient["reference_labels_read"] is not False
        or gradient["expansion_point"] != "ORIGIN"
        or set(gradient["grouped_gradients"]) != set(layout)
    ):
        raise ValueError("Actual full AD must cover the complete independent origin coordinates")
    result = np.zeros((len(data), len(gradient["groups"]), 4))
    offset = 0
    for name, bounds in layout.items():
        tensor = np.asarray(gradient["grouped_gradients"][name])
        size = bounds["stop"] - bounds["start"]
        if (
            bounds["start"] != offset
            or size != int(np.prod(bounds["shape"]))
            or tensor.shape != (len(gradient["groups"]), 4, *bounds["shape"])
        ):
            raise ValueError("Actual full AD complete coordinate layout differs")
        for start in range(offset, bounds["stop"], 65536):
            stop = min(bounds["stop"], start + 65536)
            delta = np.asarray(data[:, candidate_index, start:stop], np.float64) - np.asarray(
                data[:, baseline_index, start:stop], np.float64
            )
            values = tensor.reshape(-1, size)[:, start - offset : stop - offset]
            if not np.isfinite(values).all() or not np.isfinite(delta).all():
                raise ValueError("Nonfinite actual full AD or endpoint parameter")
            result += (delta @ values.T).reshape(result.shape)
        offset = bounds["stop"]
    if offset != data.shape[-1]:
        raise ValueError("Full AD cannot omit complete parameter coordinates")
    return result


def _group(values, probes, groups):
    return np.asarray(
        [
            np.mean(
                [
                    values[i]
                    for i, p in enumerate(probes)
                    if (p["family"], p["interface"]) == tuple(g)
                ],
                axis=0,
            )
            for g in groups
        ]
    )


def evaluation_rows(origin_id, bank, probes, groups, predictions, references, *, target_index):
    from .analysis_rules import reference_precision

    target = CONTRASTS[target_index][0]
    alias = bank["contrasts"][target]["exact_inference_alias"]
    truth = np.asarray([r["estimate"][target_index] for r in references])
    se = np.asarray([r["se"][target_index] for r in references])
    resolved = np.asarray([r["resolved"][target_index] for r in references])
    group_truth, group_se, group_resolved = [], [], []
    for g in groups:
        take = [i for i, p in enumerate(probes) if (p["family"], p["interface"]) == tuple(g)]
        group_truth.append(truth[take].mean(0))
        uncertainty = np.sqrt((se[take] ** 2).sum(0)) / len(take)
        group_se.append(uncertainty)
        # Retain unresolved prompt-level overlap/precision conservatively; no
        # aggregate mean is used to erase a failed endpoint reference check.
        precision = reference_precision(uncertainty, exact_alias=alias)
        group_resolved.append(precision["reference_resolved"] & resolved[take].all(0))
    group_ids = ["|".join(g) for g in groups]
    result = []
    for name, prediction in {**predictions, "FIXED_PANEL_OUTPUT_HEAD": None}.items():
        result.append(
            {
                "seed": 41001,
                "arm": "X_BASE",
                "step": 32,
                "role": "development",
                "origin_id": origin_id,
                "bank_id": bank["bank_id"],
                "target": target,
                "method": name,
                "design_id": "CROSS_PROMPT_36_GROUPS6",
                "n": 1024,
                "prompt_ids": ["group:" + g for g in group_ids],
                "groups": group_ids,
                "evaluation_unit": "semantic_group",
                "is_alias": bool(alias),
                "prediction_status": "UNKNOWN" if prediction is None else "PREDICTED",
                "prediction": None if prediction is None else np.asarray(prediction).tolist(),
                "reference": np.asarray(group_truth).tolist(),
                "reference_se": np.asarray(group_se).tolist(),
                "reference_resolved": np.asarray(group_resolved).tolist(),
                "reference_kind": "MONTE_CARLO",
                "reference_independent": True,
                "reference_prompt_independent": True,
                "reference_estimator": "PRESERVE_XI_WITH_REGISTERED_MIX_FALLBACK",
                "reference_group_rule": "GROUP_HALF_WIDTH_AND_ALL_MEMBER_PROMPTS_RESOLVED",
                "information_level": "CURRENT_QUERY_EVENT_SCORES"
                if name == "DIRECT_MEASURE"
                else "ACTUAL_FULL_ORIGIN_AD_ON_NEW_PROMPTS"
                if name == "FULL_SCORE_JVP"
                else "FIXED_PANEL_HEAD_CANNOT_ACCEPT_NEW_PROMPTS",
                "query_response_used_for_fit": False,
                "new_prompt_measurements_paid": True,
                "gradient_estimator": "FULL_AUTODIFF_SCORE_PREFIX32"
                if name == "FULL_SCORE_JVP"
                else None,
            }
        )
    return result


def _gate(runtime, task):
    if (
        task.get("stage") != "D"
        or task.get("role") != "development"
        or task.get("seed") != 41001
        or task.get("arm") != "X_BASE"
        or task.get("step") != 32
    ):
        raise ValueError("Cross-prompt acquisition is fixed to development41001/X_BASE/32")
    if runtime.get("identity", {}).get("execution_kind") != "REAL_CUDA_MODEL":
        raise ValueError("Actual model runtime required for cross-prompt acquisition")
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Actual cross-prompt acquisition requires server Linux/Slurm")
    from .analysis_rules import load_analysis_rules

    if runtime.get("analysis_rules") != load_analysis_rules():
        raise ValueError("Current frozen reference analysis rules required")


def run_cross_prompt(config, runtime, inputs, task, forks, *, out, resume=False):
    """Collect once per role, full AD once, then evaluate saved measurements.

    Called only from the already authorized D worker. No new candidate Adam or
    source updates occur. A completed raw receipt alone is not an evaluation.
    """
    from . import gpu_collect as collect
    from .analysis_rules import load_analysis_rules

    _gate(runtime, task)
    collect.verify_artifact_bindings(forks)
    probes, banks = select_cross_prompt_inputs(inputs["panels"], forks["banks"])
    root = Path(out)
    binding = {
        "schema": "ssvc-v4-cross-prompt-binding-1",
        "config_hash": canonical_hash(config),
        "task": task,
        "forks_hash": canonical_hash(forks),
        "probes": probes,
        "source_origin_id": forks["origin_id"],
        "runtime": runtime["identity"],
        "bank_ids": [b["bank_id"] for b in banks],
        "bank_selection": "FIRST_TWO_NONALIAS_FINGERPRINTS_ONLY",
        "work_draws": 1024,
        "direct_draws": 1024,
        "reference_draws": 4096,
        "gradient_draw_prefix": 32,
        "analysis_rules": load_analysis_rules(),
        "panel_splits": {
            k: {
                "prompt_ids": [p["prompt_id"] for p in inputs["panels"][k]],
                "base_scene_ids": sorted({p["base_scene_id"] for p in inputs["panels"][k]}),
            }
            for k in ("observation", "signature", "cross_prompt")
        },
    }
    if root.exists() and any(root.iterdir()) and not resume:
        raise ValueError("Existing cross-prompt output requires explicit resume")
    root.mkdir(parents=True, exist_ok=True)
    collect._publish(root / "BINDING.json", binding)
    if (root / "COMPLETE.json").exists():
        done = collect._read(root / "COMPLETE.json")
        if done.get("binding_hash") != canonical_hash(binding):
            raise ValueError("Cross-prompt completion binding changed")
        collect.verify_artifact_bindings(done)
        return done
    subset = {**forks, "origin_id": forks["origin_id"] + ":cross_prompt", "banks": banks}
    collect._publish(root / "SELECTED_FORKS.json", subset)
    entry = runtime["capture_complete"]()
    try:
        response = collect.collect_response_map(
            config,
            runtime,
            subset,
            probes,
            out=root / "response",
            draws=1024,
            reference_draws=4096,
            resume=resume,
        )
        derivative = collect._score_derivative(
            runtime,
            subset,
            probes,
            response,
            out=root / "derivative",
            prompt_subset=(),
        )
        raw = {
            "status": "DATA_COMPLETED",
            "is_evaluation_completion": False,
            "binding": collect._binding(root / "BINDING.json"),
            "selected_forks": collect._binding(root / "SELECTED_FORKS.json"),
            "response": collect._binding(root / "response" / "COMPLETE.json"),
            "derivative": derivative,
            "cost": {
                **measurement_counts(response),
                "registered_main_origin_actions": 36 * (1024 + 1024 + 4096),
                "full_ad_anchor_actions": derivative["anchor_samples"],
                "full_ad_reuses_work_prefix": True,
                "derivative_path_qualification": derivative.get("gradient_path"),
                "per_model_resampling": False,
                "actual_extra_counts_in_original_receipts": True,
            },
        }
        collect._publish(root / "DATA_COMPLETED.json", raw)
        result = _evaluate_completed(runtime, subset, response, derivative, probes, root)
        done = {
            "schema": "ssvc-v4-cross-prompt-completion-1",
            "status": result["status"],
            "task": task,
            "binding_hash": canonical_hash(binding),
            "raw": collect._binding(root / "DATA_COMPLETED.json"),
            "predictions": collect._binding(root / "PREDICTIONS_FROZEN.json"),
            "reference": collect._binding(root / "REFERENCE_STATISTICS.json"),
            "results": collect._binding(root / "RESULTS.json"),
            "table": collect._binding(root / "QUERY_RESULTS.parquet"),
            "source_updates": 0,
            "candidate_updates": 0,
            "fixed_panel_heads": "UNKNOWN_NEW_PROMPT_UNSUPPORTED",
            "scientific_status": "ONE_SOURCE_DEVELOPMENT_DIAGNOSTIC_NOT_CERTIFIED",
            "execution_kind": runtime["identity"]["execution_kind"],
        }
        collect._publish(root / "COMPLETE.json", done)
        return done
    finally:
        runtime["restore_complete"](entry)


def _evaluate_completed(runtime, forks, response, derivative, probes, root):
    from . import gpu_collect as collect
    from .evaluate import evaluate_records
    from .response_fit import (
        _action_rows,
        _apply_mixture_references,
        _endpoint_memmap,
        _jsonable,
        _observations,
        _read_gradient,
    )
    from .tracking import _npz

    banks, origin = forks["banks"], forks["origin_id"]
    raw = {}
    for role in ("work", "direct_work"):
        raw[role] = _action_rows(
            response[role],
            probes,
            role=role,
            origin_id=origin,
            policy_fingerprint=forks["origin_policy"]["inference_fingerprint"],
        )
    _independent_packets(raw, response)
    measured = {
        b["bank_id"]: _observations(
            b, response, response["direct_work"], raw["direct_work"], probes, ("PRESERVE_XI",)
        )["PRESERVE_XI"]
        for b in banks
    }
    gradient = _read_gradient(derivative, origin, raw["work"])
    groups = gradient["groups"]
    if {tuple(g) for g in groups} != {(p["family"], p["interface"]) for p in probes}:
        raise ValueError("Actual full AD groups differ from the complete new prompt panel")
    scratch = Path(
        os.environ.get("SSVC_V4_ANALYSIS_SCRATCH") or os.environ.get("SLURM_TMPDIR") or root
    )
    scratch.mkdir(parents=True, exist_ok=True)
    predictions = {}
    with tempfile.TemporaryDirectory(prefix=".cross-prompt-coordinates-", dir=scratch) as temp:
        data, layout, _ = _endpoint_memmap(
            forks,
            banks,
            Path(temp) / "endpoints.npy",
            runtime["checkpoint_cache"],
            runtime["identity"],
        )
        endpoint_names = ("joint_0", "joint_1", "no_x_off_1")
        for ti, (target, candidate, baseline) in enumerate(CONTRASTS):
            predicted = contract_full_gradient(
                data,
                layout,
                gradient,
                endpoint_names.index(candidate),
                endpoint_names.index(baseline),
            )
            for i, bank in enumerate(banks):
                predictions[bank["bank_id"] + "__" + target] = {
                    "FULL_SCORE_JVP": predicted[i],
                    "DIRECT_MEASURE": _group(measured[bank["bank_id"]][:, ti], probes, groups),
                }
        del data
    del gradient
    _npz(
        root / "PREDICTIONS_FROZEN.npz",
        {
            key + "__" + method: value
            for key, values in predictions.items()
            for method, value in values.items()
        },
    )
    collect._publish(
        root / "PREDICTIONS_FROZEN.json",
        {
            "predictions": collect._binding(root / "PREDICTIONS_FROZEN.npz"),
            "gradient": derivative,
            "groups": groups,
            "reference_labels_read": False,
            "fixed_panel_heads": "UNKNOWN_NEW_PROMPT_UNSUPPORTED",
            "derivative_kind": "FULL_ORIGIN_AUTODIFF_SCORE_RESPONSE",
            "finite_log_ratio_surrogate": False,
            "per_model_resampling": False,
            "new_prompt_measurements_paid": True,
            "group_prompts": [
                [p["prompt_id"] for p in probes if (p["family"], p["interface"]) == tuple(g)]
                for g in groups
            ],
        },
    )
    # Independent reference labels are first opened after this prediction seal.
    raw["reference"] = _action_rows(
        response["reference"],
        probes,
        role="reference",
        origin_id=origin,
        policy_fingerprint=forks["origin_policy"]["inference_fingerprint"],
    )
    _independent_packets(raw, response)
    ids = {r["sample_id"] for packet in raw.values() for rows in packet.values() for r in rows}
    seeds = {r["sample_seed"] for packet in raw.values() for rows in packet.values() for r in rows}
    streams = {response[k]["identity"]["rng_namespace"] for k in raw}
    references, diagnostics, rows = {}, {}, []
    for bank in banks:
        bid = bank["bank_id"]
        refs = _observations(bank, response, response["reference"], raw["reference"], probes, ())
        diagnostics[bid] = _apply_mixture_references(
            bank,
            response,
            probes,
            refs,
            origin_id=origin,
            forbidden_ids=ids,
            forbidden_streams=streams,
            forbidden_seeds=seeds,
        )
        references[bid] = refs
        for ti, (target, _, _) in enumerate(CONTRASTS):
            rows.extend(
                evaluation_rows(
                    origin,
                    bank,
                    probes,
                    groups,
                    predictions[bid + "__" + target],
                    refs,
                    target_index=ti,
                )
            )
    collect._publish(
        root / "REFERENCE_STATISTICS.json",
        _jsonable(
            {
                "prompt_statistics": references,
                "fallback": diagnostics,
                "group_resolution_requires_all_prompt_members_resolved": True,
            }
        ),
    )
    result = evaluate_records(rows)
    unresolved = any(not np.asarray(r["reference_resolved"]).all() for r in rows)
    result.update(
        status="EVALUATED_REFERENCE_UNRESOLVED" if unresolved else "EVALUATED_CROSS_PROMPT",
        fixed_panel_heads="UNKNOWN_NEW_PROMPT_UNSUPPORTED",
        bootstrap_inference="ONE_SOURCE_DESCRIPTIVE_ONLY",
    )
    if runtime["identity"]["execution_kind"] != "REAL_CUDA_MODEL":
        result["status"] = "FIXTURE_EVALUATED"
    _parquet(root / "QUERY_RESULTS.parquet", rows)
    collect._publish(root / "RESULTS.json", _jsonable(result))
    return result


def _independent_packets(raw, response):
    roles = list(raw)
    sets = {
        role: (
            {r["sample_id"] for rows in packet.values() for r in rows},
            {r["sample_seed"] for rows in packet.values() for r in rows},
            response[role]["identity"]["rng_namespace"],
        )
        for role, packet in raw.items()
    }
    for i, role in enumerate(roles):
        for other in roles[:i]:
            a, b = sets[role], sets[other]
            if a[0] & b[0] or a[1] & b[1] or a[2] == b[2]:
                raise ValueError("Independent cross-prompt work/direct/reference packets overlap")


def measurement_counts(response):
    """Count unique immutable row receipts, including pair MIX/count packets.

    Score rows are logical evaluations; actual forward reuse and qualification
    calls remain in the worker's measured attempt counters.
    """
    seen, actions, scores = set(), {}, 0

    def visit(value):
        nonlocal scores
        if isinstance(value, dict):
            if {"identity", "chunks", "count"} <= value.keys():
                key = canonical_hash({k: value[k] for k in ("identity", "chunks", "count")})
                if key in seen:
                    return
                seen.add(key)
                count = value["count"]
                if (
                    type(count) is not int
                    or count < 0
                    or sum(c["stop"] - c["start"] for c in value["chunks"]) != count
                ):
                    raise ValueError("Actual measurement row receipt count differs")
                identity = value["identity"]
                if "proposal" in identity:
                    label = identity["role"] + ":" + identity["proposal"]
                    actions[label] = actions.get(label, 0) + count
                elif "sample_identity" in identity:
                    scores += count
                else:
                    raise ValueError("Unrecognized measurement receipt identity")
                return
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(response)
    return {
        "actual_unique_action_rows_by_role_proposal": actions,
        "actual_unique_action_rows_total": sum(actions.values()),
        "unique_logical_score_rows": scores,
        "logical_score_rows_are_forward_calls": False,
        "physical_work_counters_location": "WORKER_MEASURED_ATTEMPT_RECORDS",
    }


def _parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ..modeling_v3.io import sha256_file

    table = pa.Table.from_pylist(rows)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        candidate = Path(tmp) / "table.parquet"
        pq.write_table(table, candidate, compression="zstd")
        if path.exists():
            if sha256_file(path) != sha256_file(candidate):
                raise ValueError("Immutable cross-prompt result table changed")
        else:
            os.link(candidate, path)
