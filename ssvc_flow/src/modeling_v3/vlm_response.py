"""CPU artifact bridge from real V3 observation shards to response fitting.

Only calibration-pool geometry is available to selection. The fit adapter opens
selected pilot/main candidate scores; heldout labels and reference stores remain
behind an immutable prediction lock. No model is constructed by this module.
"""

from __future__ import annotations

import json
import math
import os
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import (
    atomic_json,
    atomic_npz,
    canonical_hash,
    finalize_run,
    sha256_file,
    source_identity,
    verify_manifest,
)
from .vlm_observation import (
    Q4_CONTRASTS,
    _completed_task_rows,
    _q4_reference_diagnostic,
    _read_binding,
    packet_contributions,
    reference_packet_report,
    validate_role_independence,
    validate_task_manifest,
)


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _read(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text())
    if isinstance(value, dict) and {"path", "sha256"} <= set(value):
        return json.loads(_read_binding(value).read_text())
    if not isinstance(value, dict):
        raise ValueError("A JSON object, path, or hash-bound JSON reference is required")
    return value


def _bound_spec(value):
    if isinstance(value, dict) and {"path", "sha256"} <= set(value):
        _read_binding(value)
        return _read(value), dict(value)
    if isinstance(value, (str, Path)):
        return _read(value), _binding(value)
    raise ValueError("Artifact input must include its original file path and bytes")


def _cpu_scope(*, fixture):
    if not fixture and (platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("Real V3 vector/response processing must run on server CPU under Slurm")


def _identity(config):
    return {"config_hash": canonical_hash(config), "source_hash": source_identity()["sha256"]}


def _check_identity(value, config):
    for key, expected in _identity(config).items():
        if value.get(key) != expected:
            raise ValueError("V3 response source/config identity changed: " + key)


def _forks(root, config, *, fixture):
    _cpu_scope(fixture=fixture)
    root = Path(root).resolve()
    if fixture:
        verify_manifest(root)
        result = _read(root / "result.json")
        if (
            result["identity"].get("fixture") is not True
            or result["identity"].get("execution_kind") != "CPU_FIXTURE"
        ):
            raise ValueError("Fixture bypass cannot consume a real production experiment")
    else:
        from .vlm_campaign import _verified_execution

        result = _verified_execution(root, config, unit="V3_FORKS")
    _check_identity(result["identity"], config)
    plan = _read(root / "bank_plan.json")
    if result["identity"].get("Q4_bridge") is not False:
        raise ValueError("Q4 development origins are not Q5 response data")
    origin = plan["origin_identity"]
    origin_id = f"{origin['seed']}_{origin['arm']}_{origin['step']}"
    from .vlm_campaign import seed_role

    role = seed_role(config, origin["seed"])
    banks = {row["bank_id"]: row for row in result["banks"]}
    if len(banks) != len(result["banks"]) or set(banks) != {
        row["bank_id"] for row in plan["banks"]
    }:
        raise ValueError("Fork bank list differs from its frozen origin plan")
    if not fixture and (
        sum(b["role"] == "calibration_pool" for b in banks.values()),
        sum(b["role"] == "heldout" for b in banks.values()),
    ) != (24, 12):
        raise ValueError("Q5 requires complete 24 calibration and 12 heldout banks")
    return {
        "root": root,
        "result": result,
        "plan": plan,
        "banks": banks,
        "origin_id": origin_id,
        "role": role,
        "binding": _binding(root / "result.json"),
    }


def _vectors(forks, bank_ids):
    """Read actual hash-bound parameter differences, retaining ordered coordinates."""
    from ..r3_runtime import _load_tensor_payload

    rows, d0, order, dimension, bindings = [], [], None, None, []
    for bank_id in bank_ids:
        bank_rows = []
        for name in [row[0] for row in Q4_CONTRASTS] + ["joint_0_minus_origin"]:
            spec = forks["banks"][bank_id]["contrasts"][name]
            payload = _load_tensor_payload(Path(spec["path"]), forks["result"]["identity"], spec)
            if list(payload) != spec["parameter_order"] or not payload:
                raise ValueError("Actual e payload parameter order changed")
            if order is None:
                order, dimension = spec["parameter_order"], spec["dimension"]
            if order != spec["parameter_order"] or dimension != spec["dimension"]:
                raise ValueError(
                    "Candidate vectors do not use one common parameter coordinate space"
                )
            vector = np.concatenate(
                [payload[key].detach().cpu().double().numpy().reshape(-1) for key in order]
            )
            if vector.size != dimension or not np.isfinite(vector).all():
                raise ValueError("Actual e dimension or finite-value contract failed")
            if not math.isclose(
                float(np.linalg.norm(vector)), spec["norm"], rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError("Saved actual e norm does not match original vector bytes")
            bindings.append(
                {
                    "bank_id": bank_id,
                    "contrast_id": name,
                    "path": spec["path"],
                    "sha256": spec["sha256"],
                    "payload_hash": spec["payload_hash"],
                }
            )
            if name == "joint_0_minus_origin":
                d0.append(vector)
            else:
                bank_rows.append(vector)
        rows.append(bank_rows)
    if not rows:
        raise ValueError("At least one complete bank vector block is required")
    return (
        np.asarray(rows),
        np.asarray(d0),
        {"parameter_order": order, "dimension": dimension, "vector_bindings": bindings},
    )


def _method_lock(config, role, settings, selection_lock):
    if selection_lock is None:
        if role != "development":
            raise PermissionError(
                "Q5 calibration/test requires the frozen CPU method-selection lock"
            )
        return None
    from .schema import verify_selection_lock

    lock, binding = _bound_spec(selection_lock)
    verify_selection_lock(config, lock)
    chosen = lock["selected"]
    pairs = (
        ("selector", "selection_rules"),
        ("observation_method", "observation_methods"),
        ("method", "models"),
        ("rank_cap", "rank_caps"),
    )
    for key, selected in pairs:
        if key in settings and settings[key] not in chosen[selected]:
            raise PermissionError("Requested Q5 method was not frozen: " + key)
    if "alpha" in settings and settings["alpha"] != chosen["alpha"]:
        raise PermissionError("Q5 ridge parameter differs from frozen selection")
    return binding


def prepare_q5_geometry(
    config,
    *,
    forks_root,
    n_banks,
    selector,
    out,
    seed=2026091500,
    selection_lock=None,
    fixture=False,
):
    forks = _forks(forks_root, config, fixture=fixture)
    lock = _method_lock(config, forks["role"], {"selector": selector}, selection_lock)
    bank_ids = [
        row["bank_id"] for row in forks["plan"]["banks"] if row["role"] == "calibration_pool"
    ]
    updates, d0, vectors = _vectors(forks, bank_ids)
    if type(n_banks) is not int or not 1 <= n_banks <= len(bank_ids):
        raise ValueError("Select one or more complete calibration banks")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    arrays = root / "GEOMETRY_ARRAYS.npz"
    atomic_npz(arrays, {"updates": updates, "d0": d0})
    spec = {
        "kind": "V3_Q5_CALIBRATION_GEOMETRY",
        **_identity(config),
        "origin_id": forks["origin_id"],
        "role": forks["role"],
        "arrays": _binding(arrays),
        "bank_ids": bank_ids,
        "contrast_ids": [r[0] for r in Q4_CONTRASTS],
        "n_banks": n_banks,
        "method": selector,
        "seed": seed,
        "forks": forks["binding"],
        "selection_lock": lock,
        "vector_identity": vectors,
        "query_vectors_read": False,
        "semantic_labels_read": False,
        "requires_server_cpu": not fixture,
        "fixture": fixture,
        "d0_scope": "Retained context only; not substituted for candidate contrast e",
    }
    atomic_json(root / "GEOMETRY_SPEC.json", spec)
    finalize_run(
        root, _identity(config), metadata={"origin_id": forks["origin_id"], "role": forks["role"]}
    )
    return {
        "status": "CALIBRATION_GEOMETRY_PREPARED",
        "spec": _binding(root / "GEOMETRY_SPEC.json"),
        "next_command": "select",
        "heldout_labels_opened": False,
    }


def _bundle(value, config, *, origin_id):
    q = config["qwen"]
    if any(
        q[key] != value
        for key, value in (
            ("temperature", 1.0),
            ("top_p", 1.0),
            ("top_k", 0),
            ("max_new_tokens", 64),
        )
    ):
        raise ValueError("ORIGIN conditional support requires the frozen pure-softmax decoder")
    bundle = _read(value)
    required = {"generation_manifest", "generation_root", "scoring_manifest", "scoring_root"}
    if set(bundle) != required:
        raise ValueError(
            "Observation bundle needs exactly generation/scoring manifest bindings and roots"
        )
    gen, _ = _bound_spec(bundle["generation_manifest"])
    score, _ = _bound_spec(bundle["scoring_manifest"])
    for manifest in (gen, score):
        validate_task_manifest(manifest, workers=manifest["workers"])
        _check_identity(manifest["runtime_identity"], config)
        if {task["origin_id"] for task in manifest["tasks"]} != {origin_id}:
            raise ValueError("Observation bundle belongs to another response origin")
    if gen["runtime_identity"] != score["runtime_identity"] or gen["policies"] != score["policies"]:
        raise ValueError("Generation and scoring bundle runtime/policy identities differ")
    return bundle, gen, score


def _scoped_rows(bundle, gen, score, *, candidates, purpose, logs, include_direct=False):
    roles = {"pilot", "main"} if purpose == "measurement" else {"reference"}
    generation_tasks = [
        t
        for t in gen["tasks"]
        if t["role"] in roles
        and (
            t["proposal"] == "ORIGIN"
            or (purpose == "reference" and t["proposal"] == "MIX")
            or (include_direct and t["proposal"] == "DIRECT" and t["candidate_id"] in candidates)
        )
    ]
    score_tasks = [
        t
        for t in score["tasks"]
        if t["role"] in roles
        and t["candidate_id"] in candidates
        and (purpose == "reference" or t["proposal"] == "ORIGIN")
    ]
    logs.append(
        {
            "purpose": purpose,
            "candidate_ids": sorted(candidates),
            "generation_task_ids": [t["task_id"] for t in generation_tasks],
            "scoring_task_ids": [t["task_id"] for t in score_tasks],
        }
    )
    samples = _completed_task_rows(
        bundle["generation_root"], gen, allowed_task_ids={t["task_id"] for t in generation_tasks}
    )
    scores = _completed_task_rows(
        bundle["scoring_root"], score, allowed_task_ids={t["task_id"] for t in score_tasks}
    )
    grouped, scored = defaultdict(list), defaultdict(dict)
    for row in samples:
        expected = gen["policies"][row["proposal_source"]]["inference_fingerprint"]
        if (
            row["inference_fingerprint"] != expected
            or row["runtime_identity"] != gen["runtime_identity"]
            or row.get("generation_parity", {}).get("passed") is not True
            or row["probability_execution"] != "uncached_prefix_recompute"
            or row["max_new_tokens"] != 64
        ):
            raise ValueError("Sampled response packet policy/runtime/parity is not verified")
        grouped[row["prompt_id"], row["role"], row["proposal"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: (r["draw_index"], r["sample_key"]))
    for row in scores:
        if (
            row["inference_fingerprint"]
            != score["policies"][row["candidate_id"]]["inference_fingerprint"]
            or row["runtime_identity"] != score["runtime_identity"]
        ):
            raise ValueError("Selected candidate score has another inference/runtime identity")
        if row["proposal_sample_key"] in scored[row["candidate_id"]]:
            raise ValueError("Duplicate candidate/sample scoring alias row")
        scored[row["candidate_id"]][row["proposal_sample_key"]] = row
    return grouped, scored


def _verify_fork_policies(forks, policies):
    for bank_id, bank in forks["banks"].items():
        for candidate, record in bank["checkpoints"].items():
            policy = policies[bank_id + "/" + candidate]
            if (
                policy["inference_fingerprint"] != record["inference_fingerprint"]
                or policy["checkpoint"]["sha256"] != record["sha256"]
                or policy["checkpoint"]["identity"] != record["identity"]
            ):
                raise ValueError("Measured policy does not bind the actual fork e endpoint")


class UnresolvedObservation(ValueError):
    """A measured tail failure, retaining the raw response and its diagnostic."""

    def __init__(self, raw, diagnostic):
        super().__init__("ORIGIN observation moments unresolved; independent MIX required")
        self.raw, self.diagnostic = raw, diagnostic


def _observation_estimate(
    grouped, scored, policies, units, prompt_id, method, config, *, require_resolved=True
):
    from .covariance_pilot import cost_matched_estimates
    from .observation_geometry import ContributionBatch

    batches, tails = {}, {}
    pilot, main = (grouped[prompt_id, role, "ORIGIN"] for role in ("pilot", "main"))
    validate_role_independence(pilot=pilot, main=main, reference=[])
    total = config["qwen"]["observation_n_primary"]
    if (len(pilot), len(main)) != (
        int(total * config["observation"]["pilot_fraction"]),
        total - int(total * config["observation"]["pilot_fraction"]),
    ):
        raise ValueError("Measured pilot/main count differs from frozen observation budget")
    for role, packet in (("pilot", pilot), ("main", main)):
        raw, pairs = [], []
        for unit in units:
            name, suffix_a, suffix_b = next(r for r in Q4_CONTRASTS if r[0] == unit["contrast_id"])
            a, b = unit["bank_id"] + "/" + suffix_a, unit["bank_id"] + "/" + suffix_b
            z, diagnostic = packet_contributions(
                packet,
                [scored[a][r["sample_key"]] for r in packet],
                [scored[b][r["sample_key"]] for r in packet],
                proposal="ORIGIN",
                origin_support_certified=True,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                finite_moments = bool(np.isfinite(z).all() and np.isfinite(z * z).all())
            if (require_resolved and diagnostic["requires_independent_mix"]) or not finite_moments:
                raise UnresolvedObservation(
                    z, {**diagnostic, "finite_moments": finite_moments, "role": role, "unit": unit}
                )
            raw.append(z)
            pairs.append(
                (policies[a]["inference_fingerprint"], policies[b]["inference_fingerprint"])
            )
            tails[role + ":" + unit["bank_id"] + ":" + name] = diagnostic
        batches[role] = ContributionBatch(
            np.stack(raw, axis=1),
            tuple(r["sample_key"] for r in packet),
            canonical_hash([role, [r["sample_rng_key"] for r in packet]]),
            prompt_id,
            tuple(u["bank_id"] + "/" + u["contrast_id"] for u in units),
            tuple(pairs),
            tuple(tuple(r["token_ids"]) for r in packet),
            "origin",
        )
    estimates = cost_matched_estimates(
        batches["pilot"],
        batches["main"],
        shrink=0.1,
        l1_cap=config["observation"]["pilot_b_l1_cap"],
        include_crossfit=False,
    )
    if method not in estimates["cost_matched"]:
        raise ValueError(
            "Fit adapter requires simple or independent-pilot observations with joint covariance"
        )
    chosen = estimates["cost_matched"][method]
    return chosen, batches, tails


def prepare_q5_fit(
    config,
    *,
    forks_root,
    measurement_bundle,
    geometry_spec,
    selection_root,
    model_spec,
    out,
    selection_lock=None,
    fixture=False,
):
    forks = _forks(forks_root, config, fixture=fixture)
    geometry, geometry_binding = _bound_spec(geometry_spec)
    _check_identity(geometry, config)
    if geometry["origin_id"] != forks["origin_id"] or geometry["forks"] != forks["binding"]:
        raise ValueError("Selected geometry comes from another frozen fork origin")
    verify_manifest(selection_root)
    selection_receipt = _read(Path(selection_root) / "RECEIPT.json")
    if selection_receipt.get("input_binding", {}).get("spec_sha256") != geometry_binding["sha256"]:
        raise ValueError("Bank selection is not bound to this calibration-only geometry")
    selection = _read(Path(selection_root) / "SELECTION.json")
    selected = selection["selected_bank_ids"]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(forks["banks"][b]["role"] != "calibration_pool" for b in selected)
    ):
        raise PermissionError("Only selected calibration-pool bank labels may enter fit")
    supplied = _read(model_spec)
    if set(supplied) - {
        "method",
        "rank_cap",
        "alpha",
        "output_policy",
        "regression",
        "observation_method",
    }:
        raise ValueError("Unknown response model setting")
    settings = {
        "method": "FULL_RIDGE",
        "rank_cap": "FULL",
        "alpha": 1e-5,
        "output_policy": "RAW4",
        "regression": "RIDGE",
        "observation_method": "PILOT_SHRINK_ZERO_SUM",
        **supplied,
    }
    if settings["output_policy"] != "RAW4":
        raise ValueError("Real response adapter preserves raw four-event output coordinates")
    lock = _method_lock(
        config, forks["role"], {**settings, "selector": geometry["method"]}, selection_lock
    )
    query = [row["bank_id"] for row in forks["plan"]["banks"] if row["role"] == "heldout"]
    updates, d0, vectors = _vectors(forks, selected)
    query_updates, query_d0, query_vectors = _vectors(forks, query)
    units = [{"bank_id": b, "contrast_id": c[0]} for b in selected for c in Q4_CONTRASTS]
    query_units = [{"bank_id": b, "contrast_id": c[0]} for b in query for c in Q4_CONTRASTS]
    bundle, gen, score = _bundle(measurement_bundle, config, origin_id=forks["origin_id"])
    _verify_fork_policies(forks, gen["policies"])
    candidates = {
        b + "/" + suffix for b in selected for suffix in ("joint_0", "joint_1", "no_x_off_1")
    }
    logs = []
    grouped, scored = _scoped_rows(
        bundle, gen, score, candidates=candidates, purpose="measurement", logs=logs
    )
    prompts = gen["tasks"][0]["prompt_ids"]
    responses, covariances, pilot_raw, main_raw, diagnostics = [], [], [], [], []
    for prompt_id in prompts:
        estimate, batches, tails = _observation_estimate(
            grouped,
            scored,
            gen["policies"],
            units,
            prompt_id,
            settings["observation_method"],
            config,
        )
        responses.append(estimate.estimate)
        covariances.append(estimate.covariance_of_mean)
        pilot_raw.append(batches["pilot"].contributions)
        main_raw.append(batches["main"].contributions)
        diagnostics.append(
            {
                "prompt_id": prompt_id,
                "pilot": estimate.diagnostics,
                "tails": tails,
                "sample_ids": {r: list(batch.sample_ids) for r, batch in batches.items()},
            }
        )
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    arrays = root / "FIT_ARRAYS.npz"
    atomic_npz(
        arrays,
        {
            "updates": updates.reshape(-1, updates.shape[-1]),
            "responses": np.stack(responses, axis=1),
            "covariance": np.asarray(covariances),
            "query_updates": query_updates.reshape(-1, query_updates.shape[-1]),
            "d0": d0,
            "query_d0": query_d0,
            "raw_pilot_contributions": np.asarray(pilot_raw),
            "raw_main_contributions": np.asarray(main_raw),
        },
    )
    spec = {
        "kind": "V3_Q5_RESPONSE_FIT",
        **_identity(config),
        **settings,
        "origin_id": forks["origin_id"],
        "role": forks["role"],
        "arrays": _binding(arrays),
        "calibration_units": units,
        "query_units": query_units,
        "probe_ids": prompts,
        "geometry": geometry_binding,
        "bank_selection": _binding(Path(selection_root) / "SELECTION.json"),
        "selection_lock": lock,
        "measurement_bundle": bundle,
        "forks": forks["binding"],
        "vector_identity": vectors,
        "query_vector_identity": query_vectors,
        "covariance_scope": "JOINT_SHARED_MAIN_SAMPLES_CONDITIONAL_ON_FROZEN_INDEPENDENT_PILOT"
        if settings["observation_method"].startswith("PILOT_")
        else "JOINT_SHARED_ALL_PILOT_AND_MAIN_DRAWS",
        "covariance_axis_order": "prompt, bank/contrast/event, bank/contrast/event",
        "primary_delta_v": "-delta_pI",
        "heldout_labels_read": False,
        "reference_labels_read": False,
        "requires_server_cpu": not fixture,
        "fixture": fixture,
    }
    atomic_json(root / "FIT_SPEC.json", spec)
    atomic_json(
        root / "LABEL_ACCESS.json",
        {"entries": logs, "prediction_lock_present": False, "heldout_label_tasks_opened": False},
    )
    atomic_json(root / "CALIBRATION_DIAGNOSTICS.json", diagnostics)
    finalize_run(
        root, _identity(config), metadata={"origin_id": forks["origin_id"], "role": forks["role"]}
    )
    return {
        "status": "SELECTED_CALIBRATION_FIT_PREPARED",
        "spec": _binding(root / "FIT_SPEC.json"),
        "next_command": "fit",
        "heldout_labels_opened": False,
    }


def freeze_q5_predictions(config, *, fit_spec, fit_root, out, fixture=False):
    _cpu_scope(fixture=fixture)
    spec, spec_binding = _bound_spec(fit_spec)
    _check_identity(spec, config)
    if spec.get("kind") != "V3_Q5_RESPONSE_FIT" or spec.get("fixture") is not fixture:
        raise ValueError("Prediction fixture/production scope differs from fitted observations")
    verify_manifest(Path(spec_binding["path"]).parent)
    verify_manifest(fit_root)
    receipt = _read(Path(fit_root) / "RECEIPT.json")
    if (
        receipt.get("command") != "fit"
        or receipt.get("input_binding", {}).get("spec_sha256") != spec_binding["sha256"]
    ):
        raise ValueError("Model fit is not bound to this exact selected calibration artifact")
    model_arrays = Path(fit_root) / "MODEL_ARRAYS.npz"
    with np.load(model_arrays, allow_pickle=False) as data:
        predictions = data["predictions"].copy()
    expected_shape = (len(spec["query_units"]), len(spec["probe_ids"]), 4)
    if predictions.shape != expected_shape:
        raise ValueError("Frozen predictions do not align with query units and fixed probes")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    arrays = root / "PREDICTIONS.npz"
    atomic_npz(
        arrays, {"predictions": predictions, "accepted": np.zeros(expected_shape[:-1], dtype=bool)}
    )
    lock = {
        "kind": "V3_FROZEN_PREDICTIONS",
        **_identity(config),
        "origin_id": spec["origin_id"],
        "role": spec["role"],
        "arrays": _binding(arrays),
        "fit_spec": spec_binding,
        "model_arrays": _binding(model_arrays),
        "query_units": spec["query_units"],
        "probe_ids": spec["probe_ids"],
        "model_spec": {
            k: spec[k]
            for k in (
                "method",
                "rank_cap",
                "alpha",
                "output_policy",
                "regression",
                "observation_method",
            )
        },
        "reference_labels_read": False,
        "heldout_labels_read": False,
        "accepted_count": 0,
        "acceptance_status": "UNKNOWN_PENDING_REAL_CALIBRATION_CRITERIA",
        "fixture": fixture,
        "requires_server_cpu": not fixture,
    }
    atomic_json(root / "PREDICTION_LOCK.json", lock)
    finalize_run(
        root, _identity(config), metadata={"origin_id": spec["origin_id"], "role": spec["role"]}
    )
    return {
        "status": "PREDICTIONS_FROZEN_REFERENCE_NOT_OPENED",
        "prediction_lock": _binding(root / "PREDICTION_LOCK.json"),
    }


def _reference_protocol(config):
    """Freeze all origins and all query contrasts, allowing mandatory MIX followups."""
    q = config["qwen"]
    reference = q["reference"]
    return {
        "family_cells": config["derived_workload"]["qwen_total_response_origins"]
        * q["probe_panel"]["prompts"]
        * q["training_bank_partition"]["heldout_banks"]
        * len(Q4_CONTRASTS)
        * 2,
        "alpha": 0.05,
        "looks": [reference["initial_draws"], *reference["next_draws"]],
        "half_width_goal": reference["primary_half_width_goal"],
        "fallback_half_width": reference["half_width_fallback_reporting"],
    }


def _reference_report(packet, scored, a, b, *, predictors, protocol, fixture, proposal):
    validate_role_independence(
        pilot=[r for r in predictors if r["role"] == "pilot"],
        main=[r for r in predictors if r["role"] == "main"],
        reference=packet,
    )
    scores_a = [scored[a][r["sample_key"]] for r in packet]
    scores_b = [scored[b][r["sample_key"]] for r in packet]
    raw, tails = packet_contributions(
        packet, scores_a, scores_b, proposal=proposal, origin_support_certified=True
    )
    if fixture and len(packet) not in protocol["looks"]:
        report = _q4_reference_diagnostic(
            raw,
            tails,
            proposal=proposal,
            alpha=protocol["alpha"] / protocol["family_cells"] / 4 / 5,
        )
    else:
        report = reference_packet_report(
            packet,
            scores_a,
            scores_b,
            protocol=protocol,
            predictor_samples=predictors,
            proposal=proposal,
            origin_support_certified=True,
        )
    covariance = (
        np.cov(raw, rowvar=False, ddof=1) / len(raw)
        if np.isfinite(raw).all()
        else np.full((4, 4), np.nan)
    )
    return raw, covariance, report


def _direct_count(packet):
    if len(packet) < 2 or any(r["proposal"] != "DIRECT" for r in packet):
        raise ValueError("Independent direct counts require at least two actual DIRECT draws")
    if len({r["inference_fingerprint"] for r in packet}) != 1:
        raise ValueError("DIRECT count pools distinct policies")
    categories = ("X", "S", "W", "I")
    z = np.eye(4)[[categories.index(r["category"]) for r in packet]]
    return z.mean(axis=0), np.cov(z, rowvar=False, ddof=1) / len(z)


def _reference_batches(value, config, *, origin_id, policies, runtime_identity, candidates, logs):
    document = _read(value)
    batches = document["batches"] if set(document) == {"batches"} else [document]
    if not isinstance(batches, list) or not batches:
        raise ValueError("Reference bundle needs at least one complete independent batch")
    grouped, scored, bindings = defaultdict(list), defaultdict(dict), []
    for batch in batches:
        binding, gen, score = _bundle(batch, config, origin_id=origin_id)
        if gen["policies"] != policies or gen["runtime_identity"] != runtime_identity:
            raise ValueError("Reference and predictor measured different policy/runtime identities")
        if any(t["role"] != "reference" for t in gen["tasks"] + score["tasks"]):
            raise ValueError("Reference batches cannot include predictor-label roles")
        grows, srows = _scoped_rows(
            binding, gen, score, candidates=candidates, purpose="reference", logs=logs
        )
        for key, rows in grows.items():
            grouped[key].extend(rows)
        for candidate, rows in srows.items():
            if set(rows) & set(scored[candidate]):
                raise ValueError("Reference batches repeat an already measured candidate/sample")
            scored[candidate].update(rows)
        bindings.append(binding)
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["draw_index"], row["sample_key"]))
        if len({row["sample_key"] for row in rows}) != len(rows):
            raise ValueError("Reference extension repeats an original sampling draw")
        by_sources = defaultdict(list)
        for row in rows:
            by_sources[tuple(row["proposal_candidates"])].append(row)
        for values in by_sources.values():
            if len({row["rng_namespace"] for row in values}) != 1 or sorted(
                row["draw_index"] for row in values
            ) != list(range(len(values))):
                raise ValueError(
                    "Reference cumulative look requires one fixed contiguous RNG stream"
                )
    return bindings, grouped, scored


def prepare_q5_evaluation(
    config, *, prediction_lock, measurement_bundle, reference_bundle, out, fixture=False
):
    """Open heldout labels only after verifying the actual immutable predictions.

    Unresolved reference cells remain NaN in the evaluator input. Unmasked noisy
    estimates and every all-case direct baseline are retained for diagnosis.
    """
    _cpu_scope(fixture=fixture)
    lock, lock_binding = _bound_spec(prediction_lock)
    _check_identity(lock, config)
    if (
        lock.get("kind") != "V3_FROZEN_PREDICTIONS"
        or lock.get("fixture") is not fixture
        or lock.get("reference_labels_read") is not False
        or lock.get("heldout_labels_read") is not False
        or lock.get("accepted_count") != 0
    ):
        raise PermissionError("Valid pre-reference prediction lock required")
    verify_manifest(Path(lock_binding["path"]).parent)
    with np.load(_read_binding(lock["arrays"]), allow_pickle=False) as data:
        predictions, accepted = data["predictions"].copy(), data["accepted"].copy()
    units, prompts = lock["query_units"], lock["probe_ids"]
    shape = (len(units), len(prompts), 4)
    if predictions.shape != shape or accepted.shape != shape[:-1] or accepted.any():
        raise ValueError("Prediction axes/UNKNOWN acceptance mask changed")
    fit, fit_binding = _bound_spec(lock["fit_spec"])
    _check_identity(fit, config)
    if (
        fit["query_units"] != units
        or fit["probe_ids"] != prompts
        or fit["origin_id"] != lock["origin_id"]
    ):
        raise ValueError("Frozen query identity differs from selected calibration fit")
    # This is the first boundary that opens heldout/reference semantic rows.
    measured, gen, score = _bundle(measurement_bundle, config, origin_id=lock["origin_id"])
    if measured != fit["measurement_bundle"]:
        raise ValueError("Evaluator measurement packet differs from the frozen fit source")
    candidates = {
        u["bank_id"] + "/" + s for u in units for s in ("joint_0", "joint_1", "no_x_off_1")
    }
    logs = []
    grouped, scored = _scoped_rows(
        measured,
        gen,
        score,
        candidates=candidates,
        purpose="measurement",
        logs=logs,
        include_direct=True,
    )
    reference, refgrouped, refscored = _reference_batches(
        reference_bundle,
        config,
        origin_id=lock["origin_id"],
        policies=gen["policies"],
        runtime_identity=gen["runtime_identity"],
        candidates=candidates,
        logs=logs,
    )
    protocol = _reference_protocol(config)
    reference_mean, reference_var, direct, counts = (np.full(shape, np.nan) for _ in range(4))
    count_var = np.full(shape, np.nan)
    primary_reference = np.full(shape, np.nan)
    origin_mean, origin_variance, mix_mean, mix_variance = (
        np.full(shape, np.nan) for _ in range(4)
    )
    direct_baselines = {}
    arrays, reports, raw_index, direct_diagnostics = {}, [], [], []
    for p, prompt_id in enumerate(prompts):
        try:
            estimate, batches, tails = _observation_estimate(
                grouped,
                scored,
                gen["policies"],
                units,
                prompt_id,
                lock["model_spec"]["observation_method"],
                config,
                require_resolved=False,
            )
        except UnresolvedObservation as error:
            tails = {"status": "QUERY_OBSERVATION_UNRESOLVED", **error.diagnostic}
            arrays[f"query_unresolved_raw_{p}"] = error.raw
        else:
            direct[:, p] = estimate.estimate
            from .covariance_pilot import cost_matched_estimates

            for name, value in cost_matched_estimates(
                batches["pilot"],
                batches["main"],
                shrink=0.1,
                l1_cap=config["observation"]["pilot_b_l1_cap"],
                include_crossfit=False,
            )["cost_matched"].items():
                if name not in direct_baselines:
                    direct_baselines[name] = np.full(shape, np.nan)
                direct_baselines[name][:, p] = value.estimate
            arrays[f"query_measurement_covariance_{p}"] = estimate.covariance_of_mean
            for role, batch in batches.items():
                arrays[f"query_raw_{role}_{p}"] = batch.contributions
        predictors = grouped[prompt_id, "pilot", "ORIGIN"] + grouped[prompt_id, "main", "ORIGIN"]
        origin_reference = refgrouped[prompt_id, "reference", "ORIGIN"]
        mixture_reference = refgrouped[prompt_id, "reference", "MIX"]
        direct_packet = grouped[prompt_id, "main", "DIRECT"]
        # Check all actual RNG draws jointly; candidate scores reuse IDs by design.
        validate_role_independence(
            pilot=grouped[prompt_id, "pilot", "ORIGIN"],
            main=grouped[prompt_id, "main", "ORIGIN"] + direct_packet,
            reference=origin_reference + mixture_reference,
        )
        direct_diagnostics.append(
            {
                "prompt_id": prompt_id,
                "tails": tails,
                "covariance_scope": fit["covariance_scope"],
                "direct_count_draws": len(direct_packet),
            }
        )
        for i, unit in enumerate(units):
            _, sa, sb = next(c for c in Q4_CONTRASTS if c[0] == unit["contrast_id"])
            a, b = unit["bank_id"] + "/" + sa, unit["bank_id"] + "/" + sb
            raw, covariance, report = _reference_report(
                origin_reference,
                refscored,
                a,
                b,
                predictors=predictors,
                protocol=protocol,
                fixture=fixture,
                proposal="ORIGIN",
            )
            reference_mean[i, p], reference_var[i, p] = np.mean(raw, axis=0), np.diag(covariance)
            origin_mean[i, p], origin_variance[i, p] = reference_mean[i, p], reference_var[i, p]
            key = f"reference_origin_{i}_{p}"
            arrays[key] = raw
            arrays[key + "_covariance"] = covariance
            raw_index.append({**unit, "prompt_id": prompt_id, "key": key})
            mixture = [r for r in mixture_reference if set(r["proposal_candidates"]) == {a, b}]
            mix_report, consistent = None, True
            if mixture:
                mix_raw, mix_cov, mix_report = _reference_report(
                    mixture,
                    refscored,
                    a,
                    b,
                    predictors=predictors,
                    protocol=protocol,
                    fixture=fixture,
                    proposal="MIX",
                )
                arrays[f"reference_mix_{i}_{p}"] = mix_raw
                arrays[f"reference_mix_{i}_{p}_covariance"] = mix_cov
                mix_mean[i, p], mix_variance[i, p] = mix_raw.mean(0), np.diag(mix_cov)
                # Fixed family-wide diagnostic alarm; no method rankings enter it.
                from statistics import NormalDist

                z = NormalDist().inv_cdf(
                    1 - protocol["alpha"] / protocol["family_cells"] / 4 / 5 / 2
                )
                width = z * np.sqrt(np.maximum(0, np.diag(covariance + mix_cov)))
                consistent = bool(np.all(np.abs(raw.mean(0) - mix_raw.mean(0)) <= width))
                if not consistent:
                    report = {
                        **report,
                        "status": "REFERENCE_REQUIRES_INDEPENDENT_MIX",
                        "crosscheck_consistent": False,
                    }
            selected_proposal = "UNRESOLVED_ORIGIN_DIAGNOSTIC"
            if report.get("empirical_precision_met") and consistent:
                primary_reference[i, p] = reference_mean[i, p]
                selected_proposal = "ORIGIN"
            elif mix_report and mix_report.get("empirical_precision_met"):
                reference_mean[i, p], reference_var[i, p] = mix_mean[i, p], mix_variance[i, p]
                primary_reference[i, p] = reference_mean[i, p]
                selected_proposal = "MIX"
            reports.append(
                {
                    **unit,
                    "prompt_id": prompt_id,
                    "origin": report,
                    "mix": mix_report,
                    "crosscheck_consistent": consistent,
                    "selected_proposal": selected_proposal,
                    "reference_is_exact_truth": False,
                }
            )
            pa = [r for r in direct_packet if r["candidate_id"] == a]
            pb = [r for r in direct_packet if r["candidate_id"] == b]
            if pa and pb:
                if len(pa) != config["qwen"]["observation_n_primary"] or len(pb) != len(pa):
                    raise ValueError("DIRECT count budget differs from predeclared query baseline")
                ma, ca = _direct_count(pa)
                mb, cb = _direct_count(pb)
                counts[i, p], count_var[i, p] = mb - ma, np.diag(ca + cb)
    arrays.update(
        predictions=predictions,
        accepted=accepted,
        reference=primary_reference,
        reference_estimate_unmasked=reference_mean,
        reference_variance=reference_var,
        query_direct_measurement=direct,
        direct_count_measurement=counts,
        direct_count_variance=count_var,
        delta_v_predictions=-predictions[..., 3],
        delta_v_reference_unmasked=-reference_mean[..., 3],
        reference_origin_mean=origin_mean,
        reference_origin_variance=origin_variance,
        reference_mix_mean=mix_mean,
        reference_mix_variance=mix_variance,
    )
    arrays.update({"direct_measure_" + name: values for name, values in direct_baselines.items()})
    comparisons = {}
    for name, values in [
        ("MODEL", predictions),
        ("DIRECT_MEASURE", direct),
        ("DIRECT_COUNTS", counts),
        *(("DIRECT_MEASURE_" + name, values) for name, values in direct_baselines.items()),
    ]:
        residual = values - reference_mean
        comparisons[name] = {
            "all_case_count": len(units) * len(prompts),
            "measured_case_count": int(np.isfinite(values).all(-1).sum()),
            "raw_mse_against_noisy_reference": np.nanmean(residual**2, axis=(0, 1)).tolist()
            if np.isfinite(residual).any()
            else None,
            "aggregate_reference_noise_corrected_mse": np.nanmean(
                residual**2 - reference_var, axis=(0, 1)
            ).tolist()
            if np.isfinite(residual).any()
            else None,
            "reference_precision_qualified": False,
            "ranking_allowed": False,
        }
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    arrays_path = root / "EVALUATION_ARRAYS.npz"
    atomic_npz(arrays_path, arrays)
    spec = {
        "kind": "V3_Q5_REFERENCE_EVALUATION_INPUT",
        **_identity(config),
        "origin_id": lock["origin_id"],
        "role": lock["role"],
        "arrays": _binding(arrays_path),
        "prediction_lock": lock_binding,
        "query_units": units,
        "probe_ids": prompts,
        "reference_kind": "INDEPENDENT_NOISY_PRECISION_MASKED",
        "primary_delta_v": "-delta_pI",
        "fixture": fixture,
        "requires_server_cpu": not fixture,
    }
    atomic_json(root / "EVALUATION_INPUT.json", spec)
    from .workflow import _jsonable

    atomic_json(
        root / "REFERENCE_DIAGNOSTICS.json",
        _jsonable(
            {
                "protocol": protocol,
                "reports": reports,
                "raw_array_index": raw_index,
                "all_case_comparisons": comparisons,
                "direct_measurement_diagnostics": direct_diagnostics,
            }
        ),
    )
    atomic_json(
        root / "LABEL_ACCESS.json",
        {
            "prediction_lock": lock_binding,
            "entries": logs,
            "prediction_validated_before_any_heldout_reference_labels": True,
        },
    )
    statuses = defaultdict(int)
    for report in reports:
        statuses[report["origin"]["status"]] += 1
    current_counts = {len(refgrouped[p, "reference", "ORIGIN"]) for p in prompts}
    if len(current_counts) != 1:
        raise ValueError("Cumulative reference ORIGIN draws differ across fixed probes")
    current_draws = current_counts.pop()
    next_options, mandatory = [], []
    for report in reports:
        for proposal in ("origin", "mix"):
            evidence = report[proposal]
            if evidence and not evidence.get("empirical_precision_met"):
                later = [n for n in protocol["looks"] if n > evidence["n"]]
                if later:
                    next_options.append(later[0])
        if (
            report["origin"].get("tail_diagnostics", {}).get("requires_independent_mix")
            or not report["crosscheck_consistent"]
        ):
            mandatory.append({key: report[key] for key in ("bank_id", "contrast_id", "prompt_id")})
    precision_diagnostics = {
        "protocol": protocol,
        "unit_reports": reports,
        "predictor_rankings_used": False,
        "precision_only": True,
    }
    atomic_json(root / "REFERENCE_PRECISION_DIAGNOSTICS.json", precision_diagnostics)
    precision_receipt = {
        "kind": "V3_REFERENCE_PRECISION",
        **_identity(config),
        "origin_id": lock["origin_id"],
        "role": lock["role"],
        "prediction_binding": lock_binding,
        "reference_batches": reference,
        "current_draws": current_draws,
        "next_draws": max(next_options) if next_options else None,
        "precision_only": True,
        "predictor_rankings_used": False,
        "mandatory_mixture_units": mandatory,
        "decision_evidence": _binding(root / "REFERENCE_PRECISION_DIAGNOSTICS.json"),
        "fixture": fixture,
        "requires_server_cpu": not fixture,
    }
    atomic_json(root / "REFERENCE_PRECISION_RECEIPT.json", precision_receipt)
    receipt = {
        "kind": "V3_RESPONSE_EVALUATION",
        **_identity(config),
        "origin_id": lock["origin_id"],
        "role": lock["role"],
        "status": "REFERENCE_EVALUATION_PREPARED",
        "prediction_lock": lock_binding,
        "prediction_binding": lock_binding,
        "fit_spec": fit_binding,
        "evaluation_input": _binding(root / "EVALUATION_INPUT.json"),
        "measurement_bundle": measured,
        "reference_bundle": {"batches": reference},
        "reference_precision": _binding(root / "REFERENCE_PRECISION_RECEIPT.json"),
        "reference_status_counts": dict(statuses),
        "resolved_reference_cells": int(np.isfinite(primary_reference).all(-1).sum()),
        "all_case_count": len(units) * len(prompts),
        "accepted_count": 0,
        "ranking_allowed": False,
        "online_ssvc": "NOT_CERTIFIED",
        "fixture": fixture,
        "requires_server_cpu": not fixture,
    }
    # Recheck immutable prediction bytes after all reference accesses.
    _read_binding(lock_binding)
    _read_binding(lock["arrays"])
    atomic_json(root / "RESPONSE_EVALUATION_RECEIPT.json", receipt)
    finalize_run(
        root, _identity(config), metadata={"origin_id": lock["origin_id"], "role": lock["role"]}
    )
    return {
        "status": receipt["status"],
        "spec": _binding(root / "EVALUATION_INPUT.json"),
        "receipt": _binding(root / "RESPONSE_EVALUATION_RECEIPT.json"),
        "precision_receipt": _binding(root / "REFERENCE_PRECISION_RECEIPT.json"),
        "next_command": "evaluate",
    }
