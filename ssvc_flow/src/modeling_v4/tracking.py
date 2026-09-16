"""Conditional, offline evaluation of the saved actual step64--80 trajectory.

No source update is performed here. Calibration forks are genuine scratch Adam
updates from the complete step64 state. Intermediate derivatives are paid
measurements, never described as unobserved extrapolation. Collection alone
cannot produce COMPLETE_E.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..modeling_v3.io import atomic_npz, canonical_hash, sha256_file

STEPS = tuple(range(64, 81))
HORIZONS = (1, 2, 4, 8, 16)
REFRESH = (64, 68, 72, 76, 80)
REQUIRED_METHODS = (
    "FIXED_LOCAL_FULL",
    "ORIGIN_SCORE_CUMULATIVE",
    "REFRESHED_SCORE_EVERY4",
    "NONLINEAR",
)


def accumulate_refreshed_derivatives(products):
    """products[s][t] is J_s(theta_t-theta64); integrate actual one-step deltas."""
    if set(products) != set(REFRESH):
        raise ValueError("All registered refresh steps required")
    arrays = {s: np.asarray(v, dtype=float) for s, v in products.items()}
    shape = arrays[64].shape
    if (
        len(shape) != 3
        or shape[0] != 17
        or shape[-1] != 4
        or any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values())
    ):
        raise ValueError("Aligned complete 17-point gradient products required")
    result = np.zeros(shape)
    for index in range(1, 17):
        point = 64 + 4 * ((index - 1) // 4)
        result[index] = result[index - 1] + arrays[point][index] - arrays[point][index - 1]
    return result


def evaluate_tracking_paths(
    predictions,
    reference,
    reference_se,
    reference_resolved,
    *,
    groups,
    methods,
    unavailable=None,
    is_alias=None,
):
    """All errors concern differences from step64, with the baseline excluded.

    Window maxima of noisy references are descriptive statistics; no independent
    timepoint assumption or maximum-error confidence interval is manufactured.
    """
    truth, se = np.asarray(reference, float), np.asarray(reference_se, float)
    resolved = np.asarray(reference_resolved)
    if (
        truth.shape != (17, len(groups), 4)
        or se.shape != truth.shape
        or resolved.shape != truth.shape
        or resolved.dtype.kind != "b"
        or np.any(se < 0)
        or not np.isfinite(se).all()
        or not np.isfinite(truth[resolved]).all()
    ):
        raise ValueError("Exactly 17 aligned group/four-event reference points required")
    if len(set(groups)) != len(groups) or not predictions or set(predictions) != set(methods):
        raise ValueError("Unique groups and explicit method measurement metadata required")
    aliases = np.asarray(np.arange(17) == 0 if is_alias is None else is_alias)
    if aliases.shape != (17,) or aliases.dtype.kind != "b":
        raise ValueError("Explicit actual trajectory alias flags required")
    unavailable = dict(unavailable or {})
    for method in REQUIRED_METHODS:
        if method not in predictions:
            unavailable.setdefault(method, "NOT_AVAILABLE")
    rows, unknown = [], 0
    for method, values in predictions.items():
        prediction = np.asarray(values, float)
        if prediction.shape != truth.shape or np.isinf(prediction).any():
            raise ValueError("Each prediction needs 17 aligned points; unknown must be NaN")
        if type(methods[method].get("uses_intermediate_measurements")) is not bool:
            raise ValueError("Method must explicitly identify intermediate measurement use")
        error, known = prediction - truth, np.isfinite(prediction)
        unknown += int((~known[1:]).sum())
        for horizon in HORIZONS:
            valid = known[1 : horizon + 1] & resolved[1 : horizon + 1]
            window = np.abs(error[1 : horizon + 1])
            endpoint_valid = known[horizon] & resolved[horizon]
            observed = known[1 : horizon + 1] & np.isfinite(truth[1 : horizon + 1])
            finite_max = float(window[observed].max()) if observed.any() else None
            nonalias = np.broadcast_to(~aliases[1 : horizon + 1, None, None], valid.shape)
            nonalias_valid = valid & nonalias
            rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "anchor_step": 64,
                    "endpoint_step": 64 + horizon,
                    "endpoint_max_absolute_error": float(abs(error[horizon]).max())
                    if endpoint_valid.all()
                    else None,
                    "endpoint_four_event_mse": float(np.mean(error[horizon] ** 2))
                    if endpoint_valid.all()
                    else None,
                    "window_max_absolute_error": finite_max if valid.all() else None,
                    "finite_subset_window_max_absolute_error": finite_max,
                    "unknown_prediction_cells": int((~known[1 : horizon + 1]).sum()),
                    "reference_unresolved_cells": int((~resolved[1 : horizon + 1]).sum()),
                    "finite_resolved_cells": int(valid.sum()),
                    "all_cells": int(valid.size),
                    "all_finite_observed_cells": int(observed.sum()),
                    "finite_observed_endpoint_max_error": float(
                        abs(error[horizon])[known[horizon]].max()
                    )
                    if known[horizon].any()
                    else None,
                    "reference_se_max": float(se[1 : horizon + 1].max()),
                    "timepoints_share_reference_draws": True,
                    "maximum_is_a_confidence_bound": False,
                    "target": "actual_theta_t_minus_theta_64",
                    "endpoint_is_alias": bool(aliases[horizon]),
                    "window_alias_timepoints": int(aliases[1 : horizon + 1].sum()),
                    "window_nonalias_timepoints": int((~aliases[1 : horizon + 1]).sum()),
                    "nonalias_window_max_absolute_error": float(window[nonalias].max())
                    if nonalias.any() and valid[nonalias].all()
                    else None,
                    "nonalias_finite_observed_window_max_error": float(
                        window[nonalias & observed].max()
                    )
                    if (nonalias & observed).any()
                    else None,
                    "nonalias_finite_resolved_cells": int(nonalias_valid.sum()),
                    **methods[method],
                }
            )
    status = (
        "EVALUATED_UNKNOWN_PREDICTIONS"
        if unknown
        else "EVALUATED_REFERENCE_UNRESOLVED"
        if not resolved[1:].all()
        else "EVALUATED_INCOMPLETE_METHODS"
        if unavailable
        else "COMPLETE_E"
    )
    return {
        "schema": "ssvc-v4-offline-tracking-evaluation-1",
        "status": status,
        "windows": rows,
        "unknown_prediction_cells": unknown,
        "reference_unresolved_cells": int((~resolved[1:]).sum()),
        "unavailable_methods": unavailable,
        "source_updates_performed": 0,
        "reference_precision_is_certified": False,
        "online_control": False,
        "scientific_status": "OFFLINE_DEVELOPMENT_NOT_CERTIFIED",
    }


def _validate_gate(config, runtime, task):
    from .gpu_collect import verify_artifact_bindings
    from .phase_schedule import build_tracking_extension, validate_selection

    extension = runtime.get("tracking_extension")
    if not isinstance(extension, dict):
        raise ValueError("Verified tracking extension required before any E acquisition")
    if (
        runtime.get("identity", {}).get("execution_kind") != "REAL_CUDA_MODEL"
        or extension.get("status") != "PLANNED_NOT_EXECUTED"
        or extension.get("phase") != "E"
    ):
        raise ValueError("E needs actual model runtime and resolved conditional plan")
    from .analysis_rules import load_analysis_rules

    if runtime.get("analysis_rules") != load_analysis_rules():
        raise ValueError("Tracking requires the frozen current analysis rules")
    selection = validate_selection(config, extension.get("selection"))
    rebuilt = build_tracking_extension(
        config,
        selection,
        extension.get("point_response_evidence"),
        workers=extension.get("workers"),
    )
    if (
        rebuilt != extension
        or task not in rebuilt["tasks"]
        or task.get("kind") != "offline_track"
        or task.get("online_control") is not False
        or task.get("checkpoints") != list(STEPS)
        or task.get("horizons") != list(HORIZONS)
        or task.get("selection_hash") != selection["selection_hash"]
    ):
        raise ValueError("Tracking task differs from the actual resolved D evidence")
    verify_artifact_bindings(extension)
    return selection


def _read(path):
    return json.loads(Path(path).read_text())


def _npz(path, values):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(
                not np.array_equal(old[k], v, equal_nan=True) for k, v in values.items()
            ):
                raise ValueError("Immutable tracking arrays changed on resume")
    else:
        atomic_npz(path, values)


def _trajectory_batch(samples, score_rows, policies):
    """Join true timepoint identities with the shared strict observation reader."""
    from .response_fit import build_response_packet

    return build_response_packet(
        samples,
        score_rows,
        policies,
        contrasts=[(f"step_{step:03d}_minus_step_064", step, 64) for step in STEPS],
    )


def _path_observations(receipt, scores, policies, probes, *, origin_id, role, methods=()):
    from .analysis_rules import endpoint_overlap
    from .observations import observe_packet
    from .response_fit import _action_rows, _reference_statistics, _score_rows

    raw = _action_rows(
        receipt,
        probes,
        role=role,
        origin_id=origin_id,
        policy_fingerprint=policies[64]["inference_fingerprint"],
    )
    scored = {
        s: _score_rows(scores[str(s)], receipt, policies[s], receipt["identity"]["runtime"])
        for s in STEPS
    }
    estimates, statistics = {m: [] for m in methods}, []
    for prompt in probes:
        pid = prompt["prompt_id"]
        packet = _trajectory_batch(raw[pid], {s: scored[s][pid] for s in STEPS}, policies)
        if methods:
            result = observe_packet(packet, methods=methods, bootstrap_repetitions=None)
            for name in methods:
                estimates[name].append(result["methods"][name]["estimate"])
        else:
            origin_logs = np.asarray([r["generation_sequence_logp"] for r in raw[pid]])
            endpoints = {}
            for step in STEPS:
                lookup = {r["sample_id"]: r for r in scored[step][pid]}
                logs = np.array([lookup[s]["sequence_logp"] for s in packet.sample_ids])
                endpoints[step] = endpoint_overlap(logs - origin_logs)
            usable = np.array(
                [endpoints[64]["all_usable"] and endpoints[s]["all_usable"] for s in STEPS]
            )
            statistics.append(
                {
                    **_reference_statistics(packet, overlap_usable=usable),
                    "overlap_usable": usable,
                    "endpoint_ess_fraction": np.array(
                        [endpoints[s]["ess_fraction"] for s in STEPS]
                    ),
                    "endpoint_max_normalized_weight": np.array(
                        [endpoints[s]["max_normalized_weight"] for s in STEPS]
                    ),
                }
            )
    if methods:
        return {name: np.stack(v, axis=1) for name, v in estimates.items()}
    result = {
        name: np.stack([s[name] for s in statistics], axis=1)
        for name in (
            "estimate",
            "se",
            "resolved",
            "overlap_usable",
            "endpoint_ess_fraction",
            "endpoint_max_normalized_weight",
        )
    }
    result["joint_time_event_covariance_of_mean"] = np.stack(
        [v["covariance_of_mean"] for v in statistics]
    )
    result["is_alias"] = np.array(
        [
            policies[s]["inference_fingerprint"] == policies[64]["inference_fingerprint"]
            for s in STEPS
        ]
    )
    global_usable = result["overlap_usable"].all(1)
    result["all_prompt_overlap_usable"] = global_usable
    allowed = global_usable[:, None, None] | result["is_alias"][:, None, None]
    result["resolved"] &= allowed
    result["normal_approximation_half_width"] = np.stack(
        [v["precision"]["normal_approximation_half_width"] for v in statistics], axis=1
    )
    for scale in statistics[0]["precision"]["resolved_at_scale"]:
        result["resolved_at_scale_" + scale] = (
            np.stack([v["precision"]["resolved_at_scale"][scale] for v in statistics], axis=1)
            & allowed
        )
    return result


def _group(values, probes, groups):
    values = np.asarray(values)
    return np.stack(
        [
            values[
                :, [i for i, p in enumerate(probes) if (p["family"], p["interface"]) == tuple(g)]
            ].mean(1)
            for g in groups
        ],
        axis=1,
    )


def _group_reference(reference, probes, groups):
    output = {
        "estimate": _group(reference["estimate"], probes, groups),
        "is_alias": reference["is_alias"].copy(),
    }
    indices = [
        [i for i, p in enumerate(probes) if (p["family"], p["interface"]) == tuple(g)]
        for g in groups
    ]
    if any(not take for take in indices):
        raise ValueError("Every frozen semantic group must be present")
    # Each prompt has distinct iid draw seeds. Timepoints remain jointly sampled.
    output["se"] = np.stack(
        [np.sqrt((reference["se"][:, take] ** 2).sum(1)) / len(take) for take in indices], axis=1
    )
    from .analysis_rules import reference_precision

    precision = reference_precision(
        output["se"],
        exact_alias=reference["is_alias"][:, None, None],
        overlap_usable=reference["all_prompt_overlap_usable"][:, None, None],
    )
    output["resolved"] = precision["reference_resolved"]
    output["normal_approximation_half_width"] = precision["normal_approximation_half_width"]
    output["resolved_at_scale"] = precision["resolved_at_scale"]
    return output


def _endpoint_view(pairs, cache, runtime_identity, path):
    """Disposable disk view; every actual checkpoint remains in its source root."""
    from .gpu_collect import _fingerprint

    first = cache.load(pairs[0][0])["parameters"]
    layout, offset = {}, 0
    for name, tensor in sorted(first.items()):
        size = tensor.numel()
        layout[name] = {"start": offset, "stop": offset + size, "shape": tuple(tensor.shape)}
        offset += size
    if offset == 0:
        raise ValueError("Complete trainable parameter coordinates required")
    from .storage import require_space

    dtypes = {name: str(tensor.dtype) for name, tensor in first.items()}
    dtype = np.float64 if "torch.float64" in dtypes.values() else np.float32
    required_bytes = len(pairs) * 2 * offset * np.dtype(dtype).itemsize + 4096
    require_space(Path(path).parent, required_bytes)
    scalings = pairs[0][0]["module_scalings"]
    data = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(pairs), 2, offset))
    for i, pair in enumerate(pairs):
        for endpoint, policy in enumerate(pair):
            state = cache.load(policy)
            parameters = state["parameters"]
            if (
                list(sorted(parameters)) != list(layout)
                or policy["module_scalings"] != scalings
                or _fingerprint(state, runtime_identity) != policy["inference_fingerprint"]
                or [(r["name"], tuple(r["shape"])) for r in policy["parameter_layout"]]
                != [(n, item["shape"]) for n, item in layout.items()]
            ):
                raise ValueError("Saved tracking endpoint layout/scaling/fingerprint changed")
            for name, item in layout.items():
                if str(parameters[name].dtype) != dtypes[name]:
                    raise ValueError("Actual trajectory parameter dtype changed")
                tensor = parameters[name].cpu()
                value = (tensor.double() if dtype == np.float64 else tensor.float()).numpy()
                if value.shape != item["shape"] or not np.isfinite(value).all():
                    raise ValueError("Invalid actual tracking parameter tensor")
                data[i, endpoint, item["start"] : item["stop"]] = value.ravel()
        data.flush()
    return data, layout, scalings


def _gradient_products(data, layout, gradient, calibration_count):
    if (
        gradient["parameter_order"] != list(layout)
        or gradient["reference_labels_read"] is not False
        or gradient["expansion_point"] != "ORIGIN"
        or set(gradient["grouped_gradients"]) != set(layout)
    ):
        raise ValueError("Full actual independent derivative coordinates required")
    result = np.zeros((len(data) - calibration_count, len(gradient["groups"]), 4))
    for name, bounds in layout.items():
        tensor = np.asarray(gradient["grouped_gradients"][name])
        if tensor.shape != (len(gradient["groups"]), 4, *bounds["shape"]):
            raise ValueError("Gradient complete parameter shape differs")
        for start in range(bounds["start"], bounds["stop"], 65536):
            stop = min(bounds["stop"], start + 65536)
            local = start - bounds["start"]
            delta = np.asarray(
                data[calibration_count:, 0, start:stop], dtype=np.float64
            ) - np.asarray(data[calibration_count:, 1, start:stop], dtype=np.float64)
            values = tensor.reshape(-1, bounds["stop"] - bounds["start"])[
                :, local : local + stop - start
            ]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite measured derivative")
            result += (delta @ values.T).reshape(result.shape)
    return result


def _freeze_model(model, path):
    """Publish/recover an actual fit bundle before reading query reference labels."""
    from .full_kernels import load_model, save_model
    from .gpu_collect import _binding, _publish

    path = Path(path)
    if (path / "MODEL.json").exists():
        loaded = load_model(path)
        if (
            not np.array_equal(loaded["predict"](), model["predict"](), equal_nan=True)
            or loaded["_serialization"]["settings"] != model["_serialization"]["settings"]
        ):
            raise ValueError("Frozen tracking model differs on resume")
    else:
        path.mkdir(parents=True, exist_ok=True)
        if set(p.name for p in path.iterdir()) - {"FIT.npz"}:
            raise ValueError("Unexpected incomplete model files retained")
        with tempfile.TemporaryDirectory(prefix=".fit-publish-", dir=path.parent) as temp:
            staged = Path(temp) / "model"
            manifest = save_model(model, staged)
            fit = path / "FIT.npz"
            if fit.exists():
                with (
                    np.load(fit, allow_pickle=False) as old,
                    np.load(staged / "FIT.npz", allow_pickle=False) as new,
                ):
                    if set(old.files) != set(new.files) or any(
                        not np.array_equal(old[k], new[k], equal_nan=True) for k in old.files
                    ):
                        raise ValueError("Interrupted frozen fit differs from actual model")
            else:
                os.link(staged / "FIT.npz", fit)
            manifest["fit_sha256"] = sha256_file(fit)
            _publish(path / "MODEL.json", manifest)
    return {"model": _binding(path / "MODEL.json"), "fit": _binding(path / "FIT.npz")}


def run_tracking_task(config, runtime, inputs, task, root, resume=False):
    """Collect/evaluate only an actual, verified conditional E task; no online control."""
    from . import gpu_collect as collect
    from .data_adapter import NAMESPACE, build_bank_plan

    selection = _validate_gate(config, runtime, task)
    root = Path(root).resolve()
    target = root / "tasks" / task["id"]
    source_root = root / "tasks" / task["source_task"]
    source = _read(source_root / "COMPLETE.json")
    if (
        source.get("status") != "COMPLETED"
        or source.get("steps") != 128
        or source.get("seed") != task["seed"]
        or source.get("arm") != "X_BASE"
        or source.get("execution_kind") != "REAL_CUDA_MODEL"
        or "64" not in source["origins"]
    ):
        raise ValueError("Actual complete development source with full step64 required")
    policies = {step: _read(source_root / "tracking" / f"step_{step:03d}.json") for step in STEPS}
    collect.verify_artifact_bindings([source, *policies.values()])
    for step, policy in policies.items():
        identity = policy["checkpoint"]["identity"]
        if (
            identity.get("step") != step
            or identity.get("seed") != task["seed"]
            or identity.get("arm") != "X_BASE"
        ):
            raise ValueError("Actual saved trajectory step/seed/arm identity differs")
    origin = runtime["checkpoint_cache"].load(source["origins"]["64"])
    if (
        not {"optimizer", "rng", "parameters", "sampler"} <= origin.keys()
        or not {"python", "numpy", "torch", "cuda"} <= origin["rng"].keys()
        or collect._fingerprint(origin, runtime["identity"])
        != policies[64]["inference_fingerprint"]
    ):
        raise ValueError("Complete step64 parameters/Adam/RNG must match lightweight policy")
    probes = inputs["panels"]["observation"]
    if len(probes) != 72 or len({p["prompt_id"] for p in probes}) != 72:
        raise ValueError("E requires the complete registered 72-prompt panel")
    binding = {
        "task": task,
        "selection_hash": selection["selection_hash"],
        "runtime": runtime["identity"],
        "analysis_rules_hash": canonical_hash(runtime["analysis_rules"]),
        "extension_hash": runtime["tracking_extension"]["extension_hash"],
        "source": collect._binding(source_root / "COMPLETE.json"),
        "saved_policy_records": [
            collect._binding(source_root / "tracking" / f"step_{s:03d}.json") for s in STEPS
        ],
        "prompt_hash": canonical_hash(probes),
    }
    if (target / "COMPLETE.json").exists():
        if not resume:
            raise FileExistsError("Tracking output exists; explicit resume required")
        if _read(target / "BINDING.json") != binding:
            raise ValueError("Tracking resume binding changed")
        result = _read(target / "COMPLETE.json")
        collect.verify_artifact_bindings(result)
        return result
    if (target / "BINDING.json").exists() and not resume:
        raise FileExistsError("Tracking attempt exists; explicit resume required")
    collect._publish(target / "BINDING.json", binding)
    entry = runtime["capture_complete"]()
    try:
        banks = build_bank_plan(
            inputs["train_prompts"],
            origin_id=task["id"],
            calibration_banks=selection["m"],
            query_banks=0,
        )
        forks = collect.run_forks(
            runtime, origin, banks, inputs["train_prompts"], out=target / "forks", resume=resume
        )
        if forks["origin_policy"]["inference_fingerprint"] != policies[64]["inference_fingerprint"]:
            raise ValueError("Local calibration forks have a different step64 origin")
        response = collect.collect_response_map(
            config,
            runtime,
            forks,
            probes,
            out=target / "response",
            draws=config["observations"]["primary_n"],
            reference_draws=config["observations"]["reference_primary_n"],
            resume=resume,
        )
        signature_prediction = None
        if any(p["representation"] == "R4" for p in selection["primary_models"]):
            from .tracking_signature import collect_tracking_signatures

            def split(panel, namespaces):
                return {
                    "base_scene_ids": sorted({p["base_scene_id"] for p in panel}),
                    "prompt_ids": [p["prompt_id"] for p in panel],
                    "sample_ids": [],
                    "rng_namespaces": list(namespaces),
                }

            signature_prompts = inputs["panels"]["signature"]
            panels = {
                "train": split(
                    inputs["train_prompts"],
                    [b["train_samples"]["identity"]["rng_namespace"] for b in forks["banks"]],
                ),
                "observation": split(
                    probes,
                    [response[k]["identity"]["rng_namespace"] for k in ("work", "direct_work")],
                ),
                "reference": split(probes, [response["reference"]["identity"]["rng_namespace"]]),
                "signature": split(
                    signature_prompts, [f"{NAMESPACE}:{task['id']}:tracking_signature:THETA64"]
                ),
            }
            collect_tracking_signatures(
                config,
                runtime,
                task,
                policies,
                signature_prompts,
                panel_splits=panels,
                selection=selection,
                out=target / "tracking_signature",
                resume=resume,
            )
            signature_prediction = collect._binding(target / "tracking_signature" / "COMPLETE.json")
        score_tables = {}
        for role in ("work", "direct_work", "reference"):
            scores, cache = {}, {}
            for step in STEPS:
                policy = policies[step]
                fingerprint = policy["inference_fingerprint"]
                if fingerprint not in cache:
                    result = collect.collect_scores(
                        runtime,
                        policy,
                        probes,
                        response[role],
                        out=target / "trajectory_scores" / role / fingerprint,
                        resume=resume,
                    )
                    cache[fingerprint] = {
                        k: v for k, v in result.items() if k != "runtime_counters_this_invocation"
                    }
                scores[str(step)] = cache[fingerprint]
            score_tables[role] = scores
        collect._publish(
            target / "DATA_COMPLETED.json",
            {
                "status": "DATA_COMPLETED",
                "source_originals": binding["saved_policy_records"],
                "forks": collect._binding(target / "forks" / "COMPLETE.json"),
                "response": collect._binding(target / "response" / "COMPLETE.json"),
                "trajectory_scores": score_tables,
                "tracking_signature": signature_prediction,
                "is_E_analysis_completion": False,
            },
        )
        return _fit_and_evaluate(
            config,
            runtime,
            task,
            target,
            probes,
            policies,
            forks,
            response,
            score_tables,
            selection,
            resume,
            signature_prediction=signature_prediction,
        )
    finally:
        runtime["restore_complete"](entry)


def _fit_and_evaluate(
    config,
    runtime,
    task,
    target,
    probes,
    policies,
    forks,
    response,
    score_tables,
    selection,
    resume,
    *,
    signature_prediction=None,
):
    from . import gpu_collect as collect
    from .full_kernels import fit_precomputed_response
    from .response_fit import _action_rows, _observations, _read_gradient, compute_response_geometry

    work = _action_rows(
        response["work"],
        probes,
        role="work",
        origin_id=task["id"],
        policy_fingerprint=policies[64]["inference_fingerprint"],
    )
    fixed, rbf, unavailable = _tracking_models(selection)
    fixed_observation = _frozen_observation(fixed)
    nonlinear_observation = _frozen_observation(rbf)
    observation_methods = sorted(
        {fixed_observation["observation_method"], nonlinear_observation["observation_method"]}
    )
    measured = [
        _observations(
            bank, response, response["work"], work, probes, methods=tuple(observation_methods)
        )
        for bank in forks["banks"]
    ]
    labels_by_method = {
        name: np.asarray([row[name] for row in measured])[:, :, :2, :]
        .transpose(0, 2, 1, 3)
        .reshape(-1, 72, 4)
        for name in observation_methods
    }
    labels = labels_by_method[fixed_observation["observation_method"]]
    pairs = [
        (b["policies"][candidate], b["policies"]["joint_0"])
        for b in forks["banks"]
        for candidate in ("joint_1", "no_x_off_1")
    ]
    count = len(pairs)
    pairs.extend((policies[s], policies[64]) for s in STEPS)
    derivative_receipts, products = {}, {}
    subset = collect._diagnostic_prompt_ids(probes)
    gradient_receipt = collect._score_derivative(
        runtime,
        {"origin_id": task["id"], "origin_policy": policies[64]},
        probes,
        {"work": response["work"]},
        out=target / "derivatives" / "step_064",
        prompt_subset=subset,
    )
    derivative_receipts["64"] = gradient_receipt
    gradient = _read_gradient(gradient_receipt, task["id"], work)
    groups = gradient["groups"]
    group_names = ["|".join(g) for g in groups]
    predictions, methods, artifacts = {}, {}, {}
    scratch = Path(
        os.environ.get("SSVC_V4_ANALYSIS_SCRATCH") or os.environ.get("SLURM_TMPDIR") or target
    )
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".derived-endpoints-", dir=scratch) as temp:
        data, layout, scalings = _endpoint_view(
            pairs, runtime["checkpoint_cache"], runtime["identity"], Path(temp) / "endpoints.npy"
        )
        geometry = compute_response_geometry(
            data,
            layout,
            scalings,
            gradient,
            candidate="candidate",
            baseline="baseline",
            calibration_count=count,
            base_id=runtime["identity"]["model_hash"],
            endpoint_names=("candidate", "baseline"),
        )
        products[64] = geometry["derivative"]["group"]
        del gradient
        _npz(
            target / "CALIBRATION_AND_GEOMETRY.npz",
            {
                "calibration_responses": labels,
                **{"calibration_" + key: value for key, value in labels_by_method.items()},
                "raw_gram": geometry["raw"],
                "effective_gram": geometry["effective"],
            },
        )
        alpha = float(fixed["settings"].get("alpha", 1e-5)) if fixed else 1e-5
        gram = geometry["raw"]
        model = fit_precomputed_response(
            gram[:count, :count],
            labels,
            query_cross_gram=gram[count:, :count],
            query_norms=np.maximum(np.diag(gram)[count:], 0),
            alpha=alpha,
            method="FULL_DUAL_RIDGE",
            input_metadata={
                "origin_step": 64,
                "query_identity": "actual_theta_t_minus_theta_64",
                "query_labels_read": False,
            },
        )
        artifacts["FIXED_LOCAL_FULL"] = _freeze_model(model, target / "models" / "fixed_full")
        predictions["FIXED_LOCAL_FULL"] = _group(model["predict"](), probes, groups)
        methods["FIXED_LOCAL_FULL"] = {
            "uses_intermediate_measurements": False,
            "measurement_scope": "STEP64_LOCAL_CALIBRATION_ONLY",
            "alpha": alpha,
            **fixed_observation,
            "fixed_basis": True,
            "input_representation": "R2_RAW_COMPLETE_LORA_PARAMETERS",
            "selection_model_id": fixed["id"] if fixed else None,
            "settings_source": "SUPPORTED_FROZEN_SELECTION"
            if fixed
            else "EXPLICIT_FIXED_FULL_BASELINE",
        }
        if rbf is not None:
            settings = rbf["settings"]
            nonlinear = fit_precomputed_response(
                gram[:count, :count],
                labels_by_method[nonlinear_observation["observation_method"]],
                query_cross_gram=gram[count:, :count],
                query_norms=np.maximum(np.diag(gram)[count:], 0),
                method="FULL_RBF_RAW",
                alpha=float(settings.get("alpha", 1e-5)),
                bandwidth_multiplier=float(settings.get("bandwidth_multiplier", 1.0)),
                input_metadata={"origin_step": 64, "selection_model_id": rbf["id"]},
            )
            artifacts["NONLINEAR"] = _freeze_model(nonlinear, target / "models" / "nonlinear")
            predictions["NONLINEAR"] = _group(nonlinear["predict"](), probes, groups)
            methods["NONLINEAR"] = {
                "uses_intermediate_measurements": False,
                "measurement_scope": "STEP64_LOCAL_CALIBRATION_ONLY",
                "kind": "ACTUAL_FROZEN_RAW_RBF",
                **nonlinear_observation,
                "selection_model_id": rbf["id"],
                "input_representation": "R2_RAW_COMPLETE_LORA_PARAMETERS",
            }
        else:
            unavailable.setdefault(
                "NONLINEAR", "NO_SUPPORTED_FROZEN_RAW_RBF_OR_EXECUTABLE_MODEL_ARTIFACT"
            )
        for step in REFRESH[1:]:
            origin_id = task["id"] + f"_derivative_at_{step}"
            samples = collect.collect_actions(
                runtime,
                policies[step],
                probes,
                origin_id=origin_id,
                role="work",
                draws=32,
                out=target / "refresh_samples" / f"step_{step:03d}",
                resume=resume,
            )
            receipt = collect._score_derivative(
                runtime,
                {"origin_id": origin_id, "origin_policy": policies[step]},
                probes,
                {"work": samples},
                out=target / "derivatives" / f"step_{step:03d}",
                prompt_subset=subset,
            )
            actual = _action_rows(
                samples,
                probes,
                role="work",
                origin_id=origin_id,
                policy_fingerprint=policies[step]["inference_fingerprint"],
            )
            gradient = _read_gradient(receipt, origin_id, actual)
            if gradient["groups"] != groups:
                raise ValueError("Refresh gradient semantic groups changed")
            products[step] = _gradient_products(data, layout, gradient, count)
            derivative_receipts[str(step)] = receipt
            del gradient
        del data
    predictions["ORIGIN_SCORE_CUMULATIVE"] = products[64]
    predictions["REFRESHED_SCORE_EVERY4"] = accumulate_refreshed_derivatives(products)
    methods["ORIGIN_SCORE_CUMULATIVE"] = {
        "uses_intermediate_measurements": False,
        "measurement_scope": "STEP64_FULL_GROUP_SCORE_GRADIENT",
        "actual_increment_sum": True,
    }
    methods["REFRESHED_SCORE_EVERY4"] = {
        "uses_intermediate_measurements": True,
        "measurement_scope": "ACTUAL_GROUP_GRADIENTS_AT_64_68_72_76",
        "measured_derivative_steps": list(REFRESH),
        "step80_derivative_used_for_predictions": False,
        "unmeasured_extrapolation_claim": False,
        "intermediate_reference_truth_used": False,
    }
    direct = _path_observations(
        response["direct_work"],
        score_tables["direct_work"],
        policies,
        probes,
        origin_id=task["id"],
        role="direct_work",
        methods=("PRESERVE_XI",),
    )
    predictions["DIRECT_MEASURE"] = _group(direct["PRESERVE_XI"], probes, groups)
    predictions["ZERO"] = np.zeros_like(products[64])
    methods["DIRECT_MEASURE"] = {
        "uses_intermediate_measurements": True,
        "measurement_scope": "CURRENT_AND_INTERMEDIATE_POLICY_SCORES_ON_INDEPENDENT_WORK_DRAWS",
    }
    methods["ZERO"] = {
        "uses_intermediate_measurements": False,
        "measurement_scope": "STATISTICAL_BASELINE",
    }
    if signature_prediction is not None:
        from .tracking_signature import load_tracking_predictions

        r4_predictions, r4_methods, r4_artifacts = load_tracking_predictions(
            signature_prediction,
            selection=selection,
            origin_id=task["id"],
            policies=policies,
            probes=probes,
            groups=groups,
        )
        predictions.update(r4_predictions)
        methods.update(r4_methods)
        artifacts.update(r4_artifacts)
        for name in r4_predictions:
            unavailable.pop(name, None)
        nonlinear_r4 = next(
            (
                name
                for name, info in r4_methods.items()
                if info["kind"] in {"FULL_RBF_RAW", "NONLINEAR_SIGNATURE_RESIDUAL"}
            ),
            None,
        )
        if nonlinear_r4 is not None:
            if "NONLINEAR" not in predictions:
                predictions["NONLINEAR"] = r4_predictions[nonlinear_r4]
                methods["NONLINEAR"] = {
                    **r4_methods[nonlinear_r4],
                    "same_prediction_as": nonlinear_r4,
                }
                artifacts["NONLINEAR"] = r4_artifacts[nonlinear_r4]
            unavailable.pop("NONLINEAR", None)
    _npz(target / "PREDICTIONS_FROZEN.npz", predictions)
    collect._publish(
        target / "PREDICTIONS_FROZEN.json",
        {
            "methods": methods,
            "unavailable": unavailable,
            "model_artifacts": artifacts,
            "derivatives": derivative_receipts,
            "tracking_signature": signature_prediction,
            "predictions": collect._binding(target / "PREDICTIONS_FROZEN.npz"),
            "reference_labels_read": False,
            "target": "actual_theta_t_minus_theta_64",
        },
    )
    # This is the first read of independent reference labels by the evaluator.
    reference = _path_observations(
        response["reference"],
        score_tables["reference"],
        policies,
        probes,
        origin_id=task["id"],
        role="reference",
    )
    grouped = _group_reference(reference, probes, groups)
    _npz(target / "REFERENCE_STATISTICS.npz", reference)
    result = evaluate_tracking_paths(
        predictions,
        grouped["estimate"],
        grouped["se"],
        grouped["resolved"],
        groups=group_names,
        methods=methods,
        unavailable=unavailable,
        is_alias=reference["is_alias"],
    )
    if runtime["identity"]["execution_kind"] != "REAL_CUDA_MODEL":
        result = {
            **result,
            "status": "FIXTURE_EVALUATED",
            "fixture": True,
            "scientific_status": "FIXTURE_NOT_SCIENTIFIC_EVIDENCE",
        }
    _publish_evaluation_rows(target, task, group_names, predictions, grouped, methods)
    collect._publish(target / "EVALUATION.json", result)
    done = {
        "kind": "offline_track",
        "stage": "E",
        "status": result["status"],
        "task": task,
        "execution_kind": runtime["identity"]["execution_kind"],
        "data": collect._binding(target / "DATA_COMPLETED.json"),
        "predictions": collect._binding(target / "PREDICTIONS_FROZEN.json"),
        "reference": collect._binding(target / "REFERENCE_STATISTICS.npz"),
        "query_table": collect._binding(target / "TRACKING_RESULTS.parquet"),
        "evaluation": collect._binding(target / "EVALUATION.json"),
        "source_originals_preserved": True,
        "source_updates_performed": 0,
        "analysis_rules_hash": canonical_hash(runtime["analysis_rules"]),
        "origin_overlap_failure_action": "REFERENCE_UNRESOLVED_NO_UNREGISTERED_MIX_ACQUISITION",
        "online_control": False,
        "scientific_status": "OFFLINE_DEVELOPMENT_NOT_CERTIFIED",
    }
    collect._publish(target / "COMPLETE.json", done)
    return done


def _publish_evaluation_rows(target, task, groups, predictions, reference, methods):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = target / "TRACKING_RESULTS.parquet"
    rows = []
    for method, prediction in predictions.items():
        for index, step in enumerate(STEPS):
            rows.append(
                {
                    "method": method,
                    "seed": task["seed"],
                    "arm": task["arm"],
                    "anchor_step": 64,
                    "step": step,
                    "horizon": step - 64,
                    "target": "actual_theta_t_minus_theta_64",
                    "groups": groups,
                    "prediction": prediction[index].tolist(),
                    "reference": reference["estimate"][index].tolist(),
                    "reference_se": reference["se"][index].tolist(),
                    "reference_resolved": reference["resolved"][index].tolist(),
                    "is_alias": bool(reference["is_alias"][index]),
                    "normal_approximation_half_width": reference["normal_approximation_half_width"][
                        index
                    ].tolist(),
                    "resolved_at_scale": {
                        k: v[index].tolist() for k, v in reference["resolved_at_scale"].items()
                    },
                    "reference_group_se_assumption": "INDEPENDENT_PROMPT_DRAWS_SHARED_TIMEPOINTS",
                    "uses_intermediate_measurements": methods[method][
                        "uses_intermediate_measurements"
                    ],
                    "prediction_status": "PREDICTED"
                    if np.isfinite(prediction[index]).all()
                    else "UNKNOWN",
                }
            )
    table = pa.Table.from_pylist(rows)
    if path.exists():
        if not pq.read_table(path).equals(table):
            # Arrow NaN equality differs: compare the deterministic serialized file below.
            with tempfile.TemporaryDirectory(dir=target) as temp:
                check = Path(temp) / "check.parquet"
                pq.write_table(table, check, compression="zstd")
                if sha256_file(check) != sha256_file(path):
                    raise ValueError("Immutable tracking query table changed")
    else:
        with tempfile.TemporaryDirectory(dir=target) as temp:
            pending = Path(temp) / "query.parquet"
            pq.write_table(table, pending, compression="zstd")
            import os

            os.link(pending, path)


def _frozen_observation(model_spec):
    settings = model_spec.get("settings", {}) if model_spec else {}
    name = settings.get("observation_method", settings.get("observation", "PRESERVE_XI"))
    if name not in ("RAW4", "PRESERVE_XI", "CROSSFIT_COV_ZERO_SUM"):
        raise ValueError("Unsupported frozen tracking observation method")
    return {
        "observation_method": name,
        "observation_setting_source": "FROZEN_observation_method"
        if "observation_method" in settings
        else "FROZEN_observation"
        if "observation" in settings
        else "EXPLICIT_BASELINE_DEFAULT_PRESERVE_XI",
    }


def _tracking_models(selection):
    """Map complete frozen specifications, never model names alone, to executors.

    R4 weights cannot run without the 17 actual policies' paid signature scores.
    Unsupported selected models remain explicit incomplete results, even when
    the independent fixed raw FULL baseline can still be evaluated.
    """
    selected, unavailable = {}, {}
    for spec in selection["primary_models"]:
        settings, method = spec["settings"], spec["model"]
        reason = None
        if spec.get("representation") == "R4":
            reason = "MISSING_PAID_TRACK_SIGNATURES"
        elif spec.get("representation") != "R2":
            reason = "UNSUPPORTED_TRACK_REPRESENTATION"
        elif method not in {"FULL_DUAL_RIDGE", "FULL_RBF_RAW"}:
            reason = "UNSUPPORTED_TRACK_MODEL"
        elif set(settings) - {
            "alpha",
            "bandwidth_multiplier",
            "observation_method",
            "observation",
            "standardize_blocks",
            "rank_cap",
        }:
            reason = "UNSUPPORTED_TRACK_SETTINGS"
        elif settings.get("standardize_blocks", False) is not False:
            reason = "UNSUPPORTED_TRACK_INPUT_METRIC"
        elif settings.get("rank_cap", "FULL") != "FULL":
            reason = "UNSUPPORTED_TRACK_RANK"
        elif method in selected:
            reason = "ADDITIONAL_SELECTED_MODEL_REQUIRES_SEPARATE_TRACK_EXECUTOR"
        if reason:
            unavailable["SELECTED_MODEL:" + spec["id"]] = "NOT_AVAILABLE_" + reason
            if method == "FULL_RBF_RAW":
                unavailable.setdefault("NONLINEAR", "NOT_AVAILABLE_" + reason)
        else:
            _frozen_observation(spec)
            selected[method] = spec
    if "FULL_RBF_RAW" in selected:
        unavailable.pop("NONLINEAR", None)
    return selected.get("FULL_DUAL_RIDGE"), selected.get("FULL_RBF_RAW"), unavailable
