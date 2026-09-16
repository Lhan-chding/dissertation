"""Paid trajectory signatures on one independent theta64 anchor packet.

There are no bank abstractions, training labels, or fitting callbacks here.
Every policy is a bound actual source checkpoint; saved D models only predict.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..modeling_v3.io import canonical_hash
from .functional_features import (
    DIAGNOSTIC_NAMES,
    aggregate_diagnostics,
    build_signature,
    build_state_features,
    validate_panel_splits,
)
from .signature_collect import _completed, _finish, _typed_splits, _write_arrays
from .signature_fit import (
    _complete_document,
    _doc,
    load_signature_model,
    verify_selected_signature_model,
)

STEPS = tuple(range(64, 81))


def _selected_models(runtime, selection, prompts, fixture):
    """Validate frozen final-fit artifacts before any paid acquisition."""
    loaded, prompt_ids, groups = [], None, None
    for primary in selection["primary_models"]:
        if primary["representation"] != "R4":
            continue
        settings = primary["settings"]
        allowed = {
            "signature_model_selection",
            "signature_model_id",
            "observation_method",
            "observation",
            "alpha",
            "bandwidth_multiplier",
            "hidden_width",
            "state_mode",
        }
        if set(settings) - allowed:
            raise ValueError("Unsupported frozen signature settings")
        lock, binding = _complete_document(settings["signature_model_selection"])
        if (
            lock.get("kind") != "V4_SIGNATURE_SELECTION"
            or lock.get("status") != "FROZEN"
            or lock.get("fixture") is not fixture
            or lock.get("selection_hash")
            != canonical_hash({k: v for k, v in lock.items() if k != "selection_hash"})
            or any(
                lock.get(k) != runtime["identity"].get(k) for k in ("config_hash", "source_hash")
            )
            or lock.get("fit_scope") != "ALL_REGISTERED_DEVELOPMENT_ORIGINS"
            or lock.get("development_seeds") != [41001, 41002, 41003]
            or lock.get("query_labels_read") is not False
            or lock.get("reference_labels_read") is not False
        ):
            raise ValueError("Frozen signature selection provenance differs")
        models = [m for m in lock["models"] if m["id"] == settings["signature_model_id"]]
        if len(models) != 1:
            raise ValueError("Selected signature model ID must resolve exactly once")
        chosen = models[0]
        if (
            chosen["spec"]["model"] != primary["model"]
            or any(
                settings[k] != chosen["spec"].get(k)
                for k in ("alpha", "bandwidth_multiplier", "hidden_width", "state_mode")
                if k in settings
            )
            or any(
                settings[k] != chosen["observation_method"]
                for k in ("observation_method", "observation")
                if k in settings
            )
        ):
            raise ValueError("Selected method/settings differ from actual frozen model")
        if prompt_ids is not None and (
            prompt_ids != lock["prompt_ids"] or groups != lock["probe_groups"]
        ):
            raise ValueError("Frozen signature output panels differ")
        prompt_ids, groups = lock["prompt_ids"], lock["probe_groups"]
        if len(prompt_ids) != 72 or len(set(prompt_ids)) != 72 or len(groups) != 72:
            raise ValueError("E signature prediction requires registered 72-prompt output head")
        if not fixture:
            # Read only development metadata, never its response arrays/labels.
            data, _ = _doc(lock["dataset"])
            provenance = data.get("signature_provenance", [])
            if not provenance or any(
                p.get("prompt_ids") != [p["prompt_id"] for p in prompts] for p in provenance
            ):
                raise ValueError(
                    "Tracking signature coordinates differ from frozen development panel"
                )
        _doc(chosen["model"])
        model = load_signature_model(Path(chosen["model"]["path"]).parent)
        verify_selected_signature_model(chosen, model)
        if model["metadata"]["method"] != primary["model"] or model["metadata"]["output_shape"] != [
            72,
            4,
        ]:
            raise ValueError("Frozen model artifact method/output differs")
        loaded.append(
            {"primary_id": primary["id"], "selection": binding, "selected": chosen, "model": model}
        )
    if not loaded or len({m["primary_id"] for m in loaded}) != len(loaded):
        raise ValueError("Unique actual frozen R4 model selections required")
    return loaded, prompt_ids, groups


def collect_tracking_signatures(
    config,
    runtime,
    task,
    tracking_policies,
    signature_prompts,
    *,
    panel_splits,
    selection,
    out,
    resume=False,
    fixture=False,
):
    """Acquire native [17,36,16] features and predict [model,17,72,4].

    Production requires the actual resolved conditional E gate. The explicit
    fixture flag admits only an already marked tiny runtime, never a real model.
    The shared proposal stays theta64 for all seventeen actual source policies.
    """
    from . import gpu_collect as gpu

    if type(fixture) is not bool or bool(runtime["identity"].get("fixture", False)) != fixture:
        raise ValueError("Explicit fixture scope must match runtime")
    if not fixture:
        from .tracking import _validate_gate

        if _validate_gate(config, runtime, task) != selection:
            raise ValueError("Tracking signature selection differs from resolved E task")
    policies = dict(tracking_policies)
    if set(policies) != set(STEPS) or any(type(s) is not int for s in policies):
        raise ValueError("Actual complete theta64..80 trajectory required")
    for step in STEPS:
        identity = policies[step]["checkpoint"]["identity"]
        if (
            identity.get("step") != step
            or identity.get("seed") != task["seed"]
            or identity.get("arm") != "X_BASE"
        ):
            raise ValueError("Actual trajectory step/seed/arm differs")
    gpu.verify_artifact_bindings(list(policies.values()))
    prompts, splits = list(signature_prompts), _typed_splits(panel_splits)
    sig = splits["signature"]
    if (
        len(prompts) != 36
        or len({p["prompt_id"] for p in prompts}) != 36
        or [p["prompt_id"] for p in prompts] != sig["prompt_ids"]
        or {p["base_scene_id"] for p in prompts} != set(sig["base_scene_ids"])
        or len(sig["base_scene_ids"]) != 18
        or set(Counter(p["base_scene_id"] for p in prompts).values()) != {2}
        or any(
            len({p["interface"] for p in prompts if p["base_scene_id"] == s}) != 2
            for s in sig["base_scene_ids"]
        )
        or len(sig["rng_namespaces"]) != 1
        or sig["sample_ids"]
    ):
        raise ValueError("Freeze one independent ordered 18-scene/two-interface signature panel")
    loaded, output_prompts, groups = _selected_models(runtime, selection, prompts, fixture)
    identity = {
        "kind": "V4_TRACKING_SIGNATURE_ACQUISITION",
        "version": 1,
        "runtime": runtime["identity"],
        "task": task,
        "policies": policies,
        "selection": selection,
        "panel_splits": copy.deepcopy(splits),
        "signature_prompts": prompts,
        "draws_per_prompt": 16,
        "baseline_step": 64,
        "fixture": fixture,
    }
    root = Path(out).resolve()
    old = _completed(root, identity, resume)
    if old is not None:
        return old
    root.mkdir(parents=True, exist_ok=True)
    gpu._publish(root / "INPUT.json", identity)
    state, active = runtime["capture_complete"](), runtime.get("active_policy_identity")
    before = runtime["adapter"].forward_calls
    generations = runtime["adapter"].generation_calls
    started, returned, ledgers = time.perf_counter(), False, []
    try:
        raw = gpu.collect_actions(
            runtime,
            policies[64],
            prompts,
            origin_id=task["id"],
            role="signature",
            draws=16,
            out=root / "samples",
            resume=resume,
            proposal="ORIGIN",
            namespace=sig["rng_namespaces"][0],
        )
        ledgers.append(raw["runtime_counters_this_invocation"])
        rows = list(gpu._rows(raw))
        stable = canonical_hash(runtime["identity"])
        if any(
            r.get("proposal_fingerprint") != policies[64]["inference_fingerprint"]
            or r.get("runtime_identity") != stable
            for r in rows
        ):
            raise ValueError(
                "Trajectory signature samples must come from the actual theta64 policy"
            )
        splits["signature"]["sample_ids"] = [r["sample_id"] for r in rows]
        validate_panel_splits(splits)
        panel = {
            "panel_id": canonical_hash({"prompts": prompts, "namespace": sig["rng_namespaces"]}),
            "runtime_identity": stable,
            "origin_id": task["id"],
            "prompt_ids": [p["prompt_id"] for p in prompts],
            "eos_token_ids": sorted(runtime["adapter"].eos_ids),
            "splits": splits,
        }
        scores, score_receipts = {}, {}
        for step in STEPS:
            policy = policies[step]
            fp = policy["inference_fingerprint"]
            if fp in scores:
                continue
            receipt = gpu.collect_scores(
                runtime,
                policy,
                prompts,
                raw,
                out=root / "scores" / canonical_hash(fp),
                resume=resume,
            )
            ledgers.append(receipt["runtime_counters_this_invocation"])
            scores[fp] = list(gpu._rows(receipt))
            score_receipts[fp] = {
                k: v for k, v in receipt.items() if k != "runtime_counters_this_invocation"
            }
        features, provenance, units = [], [], []
        origin_fp = policies[64]["inference_fingerprint"]
        for step in STEPS:
            fp = policies[step]["inference_fingerprint"]
            result = build_signature(
                scores[origin_fp], scores[fp], sample_rows=rows, panel_manifest=panel
            )
            features.append(result["values"])
            provenance.append(result["provenance"])
            units.append(
                {
                    "step": step,
                    "baseline_step": 64,
                    "origin_id": task["id"],
                    "candidate_fingerprint": fp,
                    "baseline_fingerprint": origin_fp,
                    "is_alias": fp == origin_fp,
                    "target": "actual_theta_t_minus_theta_64",
                }
            )
        stats = [
            aggregate_diagnostics(
                [r["raw_completion"] for r in rows if r["prompt_id"] == p["prompt_id"]], p["scene"]
            )
            for p in prompts
        ]
        origin_state = {
            "stage": "ORIGIN",
            "origin_id": task["id"],
            "panel_role": "signature",
            "panel_id": panel["panel_id"],
            "event_probabilities": [s["event_probabilities"].tolist() for s in stats],
            "diagnostics": [s["diagnostics"].tolist() for s in stats],
            "diagnostic_names": list(DIAGNOSTIC_NAMES),
            "prompt_ids": panel["prompt_ids"],
            "sample_ids": panel["splits"]["signature"]["sample_ids"],
        }
        arrays = {
            "native_signatures": np.asarray(features),
            **{
                mode: build_state_features(origin_state, origin_id=task["id"], mode=mode)
                for mode in ("EVENT_ONLY", "EVENT_PLUS_DIAGNOSTICS")
            },
        }
        predictions, models = [], []
        for item in loaded:
            chosen = item["selected"]
            z = np.repeat(arrays[chosen["spec"].get("state_mode", "EVENT_ONLY")][None], 17, axis=0)
            predicted = item["model"]["predict"](arrays["native_signatures"], z)
            if predicted.shape != (17, 72, 4) or not np.isfinite(predicted).all():
                raise ValueError("Frozen trajectory prediction shape/values differ")
            predictions.append(predicted)
            models.append(
                {
                    "primary_id": item["primary_id"],
                    "selection": item["selection"],
                    "signature_model_id": chosen["id"],
                    "model": chosen["model"],
                    "method": chosen["spec"]["model"],
                    "observation_method": chosen["observation_method"],
                    "settings": chosen["spec"],
                }
            )
        arrays["predictions"] = np.asarray(predictions)
        bound = _write_arrays(root / "TRACKING_SIGNATURE.npz", arrays)
        gpu._publish(root / "ORIGIN_STATE.json", origin_state)
        gpu._publish(root / "PANEL.json", panel)
        gpu._publish(root / "SIGNATURE_PROVENANCE.json", provenance)
        result = {
            "kind": "V4_FROZEN_TRACKING_SIGNATURE_PREDICTIONS",
            "version": 1,
            "fixture": fixture,
            "runtime_identity": runtime["identity"],
            "origin_id": task["id"],
            "units": units,
            "steps": list(STEPS),
            "models": models,
            "model_ids": [m["primary_id"] for m in models],
            "prompt_ids": output_prompts,
            "probe_groups": groups,
            "signature_prompt_ids": panel["prompt_ids"],
            "feature_axes": ["trajectory_step", "signature_prompt", "draw"],
            "feature_shape": [17, 36, 16],
            "prediction_axes": ["model", "trajectory_step", "output_prompt", "event"],
            "event_order": list("XSWI"),
            "arrays": bound,
            "panel": gpu._binding(root / "PANEL.json"),
            "origin_state": gpu._binding(root / "ORIGIN_STATE.json"),
            "policies": {str(step): policy for step, policy in policies.items()},
            "samples": {k: v for k, v in raw.items() if k != "runtime_counters_this_invocation"},
            "score_receipts": score_receipts,
            "physical_scored_policies": len(scores),
            "completed_unique_generation_sequences": raw["count"],
            "completed_unique_scoring_sequences": sum(r["count"] for r in score_receipts.values()),
            "completed_counts_exclude_failed_attempts": True,
            "intermediate_paid_features": True,
            "shared_proposal_step": 64,
            "query_labels_read": False,
            "reference_labels_read": False,
            "fit_or_refit_performed": False,
            "online_control": False,
            "is_E_evaluation_completion": False,
        }
        gpu._publish(root / "PREDICTION_LOCK.json", result)
        returned = True
    finally:
        runtime["restore_complete"](state)
        runtime["active_policy_identity"] = active
        gpu._publish(
            root / f"COST_ATTEMPT_{len(list(root.glob('COST_ATTEMPT_*.json'))):04d}.json",
            {
                "status": "RETURNED" if returned else "FAILED",
                "operation_counters": ledgers,
                "forward_calls": runtime["adapter"].forward_calls - before,
                "generation_calls": runtime["adapter"].generation_calls - generations,
                "wall_seconds": time.perf_counter() - started,
                "scope": "entire_invocation_including_incomplete_chunks",
                "cost_recovered_after_process_kill": False,
            },
        )
    return _finish(root, identity, result)


def load_tracking_predictions(binding, *, selection, origin_id, policies, probes, groups):
    """Join completed real trajectory features to the evaluator before reference access."""
    from . import gpu_collect as gpu
    from .signature_fit import _arrays
    from .tracking import _group

    value, actual = _doc(binding)
    root = Path(actual["path"]).parent
    checked = _completed(root, gpu._read(root / "INPUT.json"), True)
    if checked != value or value.get("kind") != "V4_FROZEN_TRACKING_SIGNATURE_PREDICTIONS":
        raise ValueError("Completed bound tracking signature receipt required")
    if (
        value.get("origin_id") != origin_id
        or value.get("steps") != list(STEPS)
        or value.get("policies") != {str(k): v for k, v in policies.items()}
        or value.get("prompt_ids") != [p["prompt_id"] for p in probes]
        or value.get("probe_groups") != [p["family"] + "|" + p["interface"] for p in probes]
        or value.get("query_labels_read") is not False
        or value.get("reference_labels_read") is not False
        or value.get("fit_or_refit_performed") is not False
    ):
        raise ValueError("Actual trajectory prediction identity/panel differs")
    selected = {p["id"]: p for p in selection["primary_models"] if p["representation"] == "R4"}
    if set(value["model_ids"]) != set(selected) or len(value["models"]) != len(selected):
        raise ValueError("Complete selected R4 trajectory predictions required")
    arrays = _arrays(value["arrays"])
    values = arrays["predictions"]
    if values.shape != (len(selected), 17, 72, 4):
        raise ValueError("Actual trajectory prediction axes differ")
    predictions, methods, artifacts = {}, {}, {}
    for i, model in enumerate(value["models"]):
        primary = selected.get(model["primary_id"])
        if (
            primary is None
            or model["primary_id"] != value["model_ids"][i]
            or primary["model"] != model["method"]
        ):
            raise ValueError("Selected model label differs from actual frozen trajectory model")
        settings = primary["settings"]
        if (
            settings.get("signature_model_selection") != model["selection"]
            or settings.get("signature_model_id") != model["signature_model_id"]
        ):
            raise ValueError("Selected model binding differs from actual frozen trajectory model")
        for key in ("observation_method", "observation"):
            if key in settings and settings[key] != model["observation_method"]:
                raise ValueError("Frozen trajectory observation setting differs")
        name = "SELECTED_MODEL:" + primary["id"]
        predictions[name] = _group(values[i], probes, groups)
        methods[name] = {
            "uses_intermediate_measurements": True,
            "measurement_scope": "ACTUAL_64_TO_80_POLICY_SCORES_ON_SHARED_THETA64_SIGNATURE_DRAWS",
            "input_representation": "R4_NATIVE_SEQUENCE_SIGNATURE_576",
            "paid_features": True,
            "intermediate_signatures": True,
            "selection_model_id": primary["id"],
            "signature_model_id": model["signature_model_id"],
            "kind": model["method"],
            "observation_method": model["observation_method"],
            "fit_or_refit_performed": False,
            "independent_of_query_reference_labels": True,
        }
        artifacts[name] = {
            "tracking_signature": actual,
            "frozen_model": model["model"],
            "frozen_selection": model["selection"],
        }
    return predictions, methods, artifacts
