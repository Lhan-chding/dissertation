"""Paid native signatures and predeclared development directional diagnostics.

These helpers execute only on an already authorized, loaded runtime. They never
load query/reference labels or declare a mathematical midpoint Adam-reachable.
"""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..modeling_v3.io import atomic_npz, canonical_hash, sha256_file
from .data_adapter import CONTRASTS
from .functional_features import (
    DIAGNOSTIC_NAMES,
    aggregate_diagnostics,
    build_signature,
    build_state_features,
    validate_panel_splits,
)
from .score_response import (
    EVENTS,
    _layout,
    _proposal_logp,
    _sample_identity,
    collect_score_response,
    sequence_log_probability,
)


def _completed(root, identity, resume):
    from .gpu_collect import _read

    path = root / "COMPLETE.json"
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        result = _read(path)
        if result["input_hash"] != canonical_hash(identity):
            raise ValueError("Immutable helper input changed")
        for name, digest in result["artifact_hashes"].items():
            p = (root / name).resolve()
            if root.resolve() not in p.parents or not p.is_file() or sha256_file(p) != digest:
                raise ValueError("Completed helper artifact changed: " + name)
        actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != path}
        if actual != set(result["artifact_hashes"]):
            raise ValueError("Completed helper artifact inventory changed")
        return result
    return None


def _finish(root, identity, result):
    from .gpu_collect import _publish

    result = {
        **result,
        "input_hash": canonical_hash(identity),
        "artifact_hashes": {
            str(p.relative_to(root)): sha256_file(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p != root / "COMPLETE.json"
        },
    }
    _publish(root / "COMPLETE.json", result)
    return result


def _write_arrays(path, arrays):
    from .gpu_collect import _binding

    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[k], v) for k, v in arrays.items()
            ):
                raise ValueError("Immutable helper array content changed")
    else:
        atomic_npz(path, arrays)
    return _binding(path)


def _typed_splits(panel_splits):
    expected = {"base_scene_ids", "prompt_ids", "sample_ids", "rng_namespaces"}
    if not isinstance(panel_splits, dict) or set(panel_splits) != {
        "train",
        "observation",
        "reference",
        "signature",
    }:
        raise ValueError("Four explicit metadata-only panel splits are required")
    if any(not isinstance(v, dict) or set(v) != expected for v in panel_splits.values()):
        raise ValueError("Panel split input permits identity metadata only; no outcomes")
    validate_panel_splits(panel_splits)
    return copy.deepcopy(panel_splits)


def collect_signature_features(
    runtime, forks, signature_prompts, *, panel_splits, out, resume=False
):
    """Collect 36x16 origin actions once and score each unique endpoint once.

    Exact inference aliases reuse the same physical scoring artifact. Native
    signatures retain 576 coordinates, and origin features use only this
    independent signature panel's generated text and frozen scene metadata.
    """
    from . import gpu_collect as gpu

    splits = _typed_splits(panel_splits)
    prompts = list(signature_prompts)
    sig = splits["signature"]
    if (
        len(prompts) != 36
        or len({p["prompt_id"] for p in prompts}) != 36
        or [p["prompt_id"] for p in prompts] != sig["prompt_ids"]
        or {p["base_scene_id"] for p in prompts} != set(sig["base_scene_ids"])
        or len(sig["rng_namespaces"]) != 1
        or sig["sample_ids"]
    ):
        raise ValueError(
            "Freeze the ordered 36-prompt panel and one independent namespace before draws"
        )
    if (
        len(sig["base_scene_ids"]) != 18
        or set(Counter(p["base_scene_id"] for p in prompts).values()) != {2}
        or any(
            len({p["interface"] for p in prompts if p["base_scene_id"] == scene}) != 2
            for scene in sig["base_scene_ids"]
        )
    ):
        raise ValueError("Signature metadata requires eighteen scenes with two distinct interfaces")
    if not forks["banks"] or len({b["bank_id"] for b in forks["banks"]}) != len(forks["banks"]):
        raise ValueError("Unique actual fork banks are required")
    identity = {
        "kind": "V4_NATIVE_SIGNATURE_ACQUISITION",
        "runtime": runtime["identity"],
        "forks": forks,
        "signature_prompts": prompts,
        "panel_splits": copy.deepcopy(splits),
        "draws_per_prompt": 16,
        "native_width": 576,
    }
    root = Path(out)
    previous = _completed(root, identity, resume)
    if previous is not None:
        return previous
    root.mkdir(parents=True, exist_ok=True)
    gpu._publish(root / "INPUT.json", identity)
    complete_state = runtime["capture_complete"]()
    active = runtime.get("active_policy_identity")
    started = time.perf_counter()
    before_forward = runtime["adapter"].forward_calls
    before_generation = getattr(runtime["adapter"], "generation_calls", None)
    cost = []
    returned = False
    try:
        raw = gpu.collect_actions(
            runtime,
            forks["origin_policy"],
            prompts,
            origin_id=forks["origin_id"],
            role="signature",
            draws=16,
            out=root / "samples",
            resume=resume,
            proposal="ORIGIN",
            namespace=sig["rng_namespaces"][0],
        )
        samples = list(gpu._rows(raw))
        sample_by_id = {r["sample_id"]: r for r in samples}
        stable_runtime = canonical_hash(runtime["identity"])

        def normalized(rows):
            result = []
            for row in rows:
                if row.get("runtime_identity") not in (runtime["identity"], stable_runtime):
                    raise ValueError("Signature acquisition/scoring runtime identity changed")
                if "token_ids" not in row:
                    action = sample_by_id.get(row.get("sample_id"))
                    if action is None or row.get("shared_token_identity") != canonical_hash(
                        action["token_ids"]
                    ):
                        raise ValueError("Signature score shared action identity differs")
                    row = {**row, "token_ids": action["token_ids"]}
                result.append({**row, "runtime_identity": stable_runtime})
            return result

        samples_for_features = normalized(samples)
        splits["signature"]["sample_ids"] = [r["sample_id"] for r in samples]
        validate_panel_splits(splits)
        policies = {}
        for bank in forks["banks"]:
            for policy in bank["policies"].values():
                policies.setdefault(policy["inference_fingerprint"], policy)
        scored, score_receipts = {}, {}
        cost.append(raw.get("runtime_counters_this_invocation", {}))
        for fingerprint, policy in policies.items():
            receipt = gpu.collect_scores(
                runtime,
                policy,
                prompts,
                raw,
                out=root / "scores" / canonical_hash(fingerprint),
                resume=resume,
            )
            scored[fingerprint] = normalized(list(gpu._rows(receipt)))
            score_receipts[fingerprint] = {
                k: v for k, v in receipt.items() if k != "runtime_counters_this_invocation"
            }
            cost.append(receipt.get("runtime_counters_this_invocation", {}))
        panel = {
            "panel_id": canonical_hash({"prompts": prompts, "namespace": sig["rng_namespaces"]}),
            "runtime_identity": stable_runtime,
            "origin_id": forks["origin_id"],
            "prompt_ids": [p["prompt_id"] for p in prompts],
            "eos_token_ids": sorted(runtime["adapter"].eos_ids),
            "splits": splits,
        }
        features, units, provenance = [], [], []
        for bank in forks["banks"]:
            for name, candidate, baseline in CONTRASTS:
                fp_a = bank["policies"][baseline]["inference_fingerprint"]
                fp_b = bank["policies"][candidate]["inference_fingerprint"]
                built = build_signature(
                    scored[fp_a],
                    scored[fp_b],
                    sample_rows=samples_for_features,
                    panel_manifest=panel,
                )
                features.append(built["values"])
                units.append(
                    {
                        "bank_id": bank["bank_id"],
                        "contrast_id": name,
                        "baseline": baseline,
                        "candidate": candidate,
                    }
                )
                provenance.append(built["provenance"])
        stats = [
            aggregate_diagnostics(
                [r["raw_completion"] for r in samples if r["prompt_id"] == p["prompt_id"]],
                p["scene"],
            )
            for p in prompts
        ]
        origin_state = {
            "stage": "ORIGIN",
            "origin_id": forks["origin_id"],
            "panel_role": "signature",
            "panel_id": panel["panel_id"],
            "event_probabilities": [r["event_probabilities"].tolist() for r in stats],
            "diagnostics": [r["diagnostics"].tolist() for r in stats],
            "diagnostic_names": list(DIAGNOSTIC_NAMES),
            "prompt_ids": panel["prompt_ids"],
            "sample_ids": [r["sample_id"] for r in samples],
        }
        arrays = {
            "native_signatures": np.asarray(features),
            **{
                mode: build_state_features(origin_state, origin_id=forks["origin_id"], mode=mode)
                for mode in ("EVENT_ONLY", "EVENT_PLUS_DIAGNOSTICS")
            },
        }
        binding = _write_arrays(root / "SIGNATURE_FEATURES.npz", arrays)
        gpu._publish(root / "ORIGIN_STATE.json", origin_state)
        gpu._publish(root / "PANEL.json", panel)
        gpu._publish(root / "SIGNATURE_PROVENANCE.json", provenance)
        result = {
            "kind": "V4_NATIVE_SIGNATURE_FEATURES",
            "runtime_identity": runtime["identity"],
            "origin_id": forks["origin_id"],
            "arrays": binding,
            "units": units,
            "raw_draws": len(samples),
            "physical_scored_policies": len(policies),
            "score_receipts": score_receipts,
            "samples": {k: v for k, v in raw.items() if k != "runtime_counters_this_invocation"},
            "panel": gpu._binding(root / "PANEL.json"),
            "origin_state": gpu._binding(root / "ORIGIN_STATE.json"),
            "query_endpoint_target_labels_read": False,
            "native_width": 576,
            "paid_online_feature": True,
            "reference_labels_read": False,
            "cost_source": "COST_ATTEMPT files and raw chunk ledgers",
            "complete_failed_call_costs_available": False,
        }
        returned = True
    finally:
        runtime["restore_complete"](complete_state)
        runtime["active_policy_identity"] = active
        # This counter difference includes physical work in incomplete chunks.
        # Complete per-token accounting of a killed process remains unavailable.
        attempt = len(list(root.glob("COST_ATTEMPT_*.json")))
        gpu._publish(
            root / f"COST_ATTEMPT_{attempt:04d}.json",
            {
                "status": "RETURNED" if returned else "FAILED",
                "completed_operation_counters": cost,
                "forward_calls": runtime["adapter"].forward_calls - before_forward,
                "generation_calls": runtime["adapter"].generation_calls - before_generation
                if before_generation is not None
                else None,
                "wall_seconds": time.perf_counter() - started,
                "counter_scope": "entire_invocation_including_incomplete_chunks",
                "cost_recovered_after_process_kill": False,
            },
        )
    return _finish(root, identity, result)


def _set_point(runtime, baseline_policy, baseline, direction, fraction):
    """Materialize only trainable coordinates; report actual trained-dtype rounding."""
    import torch

    from ..optimizer_fork import state_hash

    runtime["policy_loader"](baseline_policy)
    params = _layout(runtime)
    error, realized, requested = 0.0, 0.0, 0.0
    with torch.no_grad():
        for name, param in params.items():
            base = baseline[name].double().cpu()
            delta = torch.as_tensor(direction[name], dtype=torch.float64)
            target = base + fraction * delta
            param.copy_(target.to(device=param.device, dtype=param.dtype))
            actual = param.detach().double().cpu()
            error += float((actual - target).square().sum())
            realized += float((actual - base).square().sum())
            requested += float((fraction * delta).square().sum())
    runtime["adapter"].model.eval()
    runtime["adapter"]._reset_positions()
    parameter_hash = state_hash({n: p.detach().cpu() for n, p in params.items()})
    runtime["active_policy_identity"] = {
        "kind": "MATHEMATICAL_PARAMETER_POINT",
        "baseline": baseline_policy,
        "fraction": fraction,
        "actual_trainable_hash": parameter_hash,
        "reachable_training_policy": False,
    }
    return {
        "dtype": {n: str(p.dtype) for n, p in params.items()},
        "requested_fraction": fraction,
        "rounding_l2": math.sqrt(error),
        "realized_delta_l2": math.sqrt(realized),
        "requested_delta_l2": math.sqrt(requested),
        "actual_trainable_hash": parameter_hash,
    }


def _anchor_event_means(runtime, prompts, samples):
    """A paid ordinary probability query on the same independent anchor packet."""
    from ..modeling_v3.vlm_observation import validate_action
    from ..r4_inputs import STRATA

    adapter = runtime["adapter"]
    by_id = {p["prompt_id"]: p for p in prompts}
    grouped = {key: [] for key in by_id}
    before, started = adapter.forward_calls, time.perf_counter()
    modes = [(m, m.training) for m in adapter.model.modules()]
    guard = runtime["state_guard"]()
    scored, tokens, raw_scores, raw_weights = 0, 0, [], []
    fault = None
    try:
        for sample in samples:
            p = by_id[sample["prompt_id"]]
            validate_action(
                sample, eos_ids=adapter.eos_ids, max_new_tokens=sample["max_new_tokens"]
            )
            prepared = adapter.prepare(p["prompt"], p.get("data_root") or runtime.get("data_root"))
            if prepared["audit"]["input_tensor_hash"] != sample["input_tensor_hash"]:
                raise ValueError("Numerical diagnostic input changed")
            scores = adapter.logprobs(prepared, sample["token_ids"], require_grad=False)
            scored += 1
            tokens += len(sample["token_ids"])
            value = sequence_log_probability(
                scores,
                sample["token_ids"],
                eos_ids=adapter.eos_ids,
                max_new_tokens=sample["max_new_tokens"],
            )
            weight = math.exp(value.item() - _proposal_logp(sample))
            if not math.isfinite(weight):
                raise ValueError("Nonfinite raw numerical-diagnostic importance ratio")
            row = np.zeros(4)
            row[EVENTS.index(sample["category"])] = weight
            grouped[p["prompt_id"]].append(row)
            raw_scores.append(scores.detach().double().cpu().numpy())
            raw_weights.append(weight)
        means = {p: np.mean(v, axis=0) for p, v in grouped.items()}
        values = np.array(
            [
                np.mean(
                    [means[p["prompt_id"]] for p in prompts if (p["family"], p["interface"]) == g],
                    axis=0,
                )
                for g in STRATA
            ]
        )
    except BaseException as exc:
        fault = exc
        raise
    finally:
        for module, training in modes:
            module.training = training
        cost = {
            "scored_sequences": scored,
            "scored_tokens": tokens,
            "forward_calls": adapter.forward_calls - before,
            "wall_seconds": time.perf_counter() - started,
        }
        runtime.setdefault("directional_numeric_cost_ledger", []).append(
            {"cost": cost, "status": "FAILED" if fault else "RETURNED"}
        )
        if guard != runtime["state_guard"]():
            raise RuntimeError("Numerical diagnostic changed model/optimizer state")
    width = max(map(len, raw_scores))
    token_logps = np.zeros((len(raw_scores), width))
    masks = np.zeros_like(token_logps, dtype=bool)
    for i, row in enumerate(raw_scores):
        token_logps[i, : len(row)] = row
        masks[i, : len(row)] = True
    return (
        values,
        cost,
        {
            "token_logprobs": token_logps,
            "token_mask": masks,
            "sequence_logprobs": token_logps.sum(1, dtype=np.float64),
            "importance_weights": np.asarray(raw_weights),
        },
    )


def collect_development_directional(
    runtime,
    forks,
    probes,
    anchor_samples,
    *,
    bank_ids,
    steps,
    out,
    prompt_subset=(),
    sample_limit=32,
    resume=False,
    stage_role="development",
):
    """Fixed four-bank BASE/MIDPOINT AD and independently computed FD diagnostics.

    Both use anchor actions independent of the reference. FD is a numerical
    check of the same Monte Carlo functional, not an event truth measurement.
    All AD outputs are persisted before the first finite-difference query.
    """
    from ..r4_inputs import STRATA
    from . import gpu_collect as gpu

    bank_ids, steps, probes = list(bank_ids), list(steps), list(probes)
    banks = {b["bank_id"]: b for b in forks["banks"]}
    if (
        stage_role != "development"
        or len(bank_ids) != 4
        or len(set(bank_ids)) != 4
        or set(bank_ids) - banks.keys()
    ):
        raise ValueError("Exactly four predeclared development banks are required")
    if (
        not steps
        or len(set(steps)) != len(steps)
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in steps)
    ):
        raise ValueError("Positive frozen finite-difference steps are required")
    if type(sample_limit) is not int or sample_limit < 1:
        raise ValueError("Positive frozen anchor sample limit required")
    if not runtime["identity"].get("fixture") and sample_limit not in {32, 128}:
        raise ValueError("Real development anchor budgets must be the frozen 32 or 128")
    samples = list(anchor_samples)
    for sample in samples:
        _sample_identity(sample)  # rejects reference before any policy/model call
        _proposal_logp(sample)
    samples = [s for s in samples if s["draw_index"] < sample_limit]
    if {s["prompt_id"] for s in samples} != {p["prompt_id"] for p in probes} or any(
        sum(s["prompt_id"] == p["prompt_id"] for s in samples) != sample_limit for p in probes
    ):
        raise ValueError("Frozen per-prompt anchor draw ranges must be complete")
    if {(p["family"], p["interface"]) for p in probes} != set(STRATA):
        raise ValueError("All six frozen probe strata are required")
    identity = {
        "kind": "V4_DEVELOPMENT_DIRECTIONAL",
        "runtime": runtime["identity"],
        "origin_id": forks["origin_id"],
        "forks": forks,
        "probes": probes,
        "anchor_samples": samples,
        "bank_ids": bank_ids,
        "steps": steps,
        "prompt_subset": list(prompt_subset),
        "sample_limit": sample_limit,
        "stage_role": stage_role,
    }
    root = Path(out)
    old = _completed(root, identity, resume)
    if old is not None:
        return old
    root.mkdir(parents=True, exist_ok=True)
    gpu._publish(root / "INPUT.json", identity)
    original = runtime["capture_complete"]()
    active = runtime.get("active_policy_identity")
    fast_scorer = runtime.pop("derivative_scorer", None)
    fast_certificate = runtime.pop("derivative_certificate", None)
    cost_offset = len(runtime.get("score_cost_ledger", []))
    numeric_cost_offset = len(runtime.get("directional_numeric_cost_ledger", []))
    records = []
    try:
        for bank_id in bank_ids:
            bank = banks[bank_id]
            base_policy, end_policy = bank["policies"]["joint_0"], bank["policies"]["joint_1"]
            baseline = runtime["checkpoint_cache"].load(base_policy)["parameters"]
            candidate = runtime["checkpoint_cache"].load(end_policy)["parameters"]
            params = _layout(runtime)
            if set(baseline) != set(params) or set(candidate) != set(params):
                raise ValueError("Fork checkpoints must contain every trainable coordinate")
            direction = {
                n: (candidate[n].double() - baseline[n].double()).cpu().numpy() for n in params
            }
            record = {
                "bank_id": bank_id,
                "baseline_policy": base_policy,
                "candidate_policy": end_policy,
                "direction_norm": math.sqrt(sum(float(np.sum(v * v)) for v in direction.values())),
            }
            # Freeze actual AD outputs for both expansion points before any FD.
            for name, fraction in (("BASE", 0.0), ("MIDPOINT", 0.5)):
                saved_summary = root / bank_id / f"{name}_AD.json"
                if saved_summary.exists():
                    if not resume:
                        raise FileExistsError(saved_summary)
                    summary = gpu._read(saved_summary)
                    if sha256_file(summary["arrays"]["path"]) != summary["arrays"]["sha256"]:
                        raise ValueError("Saved AD artifact changed")
                    record[name] = summary
                    continue
                rounding = _set_point(runtime, base_policy, baseline, direction, fraction)
                result = collect_score_response(
                    runtime,
                    probes,
                    samples,
                    expansion_point=name,
                    prompt_subset=prompt_subset,
                    gradients=False,
                    directions={"actual_e": direction},
                )
                arrays = {
                    "group_derivative": result["grouped_directional_response"][0],
                    "raw_mass": result["raw_directional_mass"][0],
                    "importance_weights": result["importance_weights"],
                }
                for i, probe in enumerate(probes):
                    arrays[f"anchor_{i}_directional_contributions"] = result[
                        "per_prompt_directional_contributions"
                    ][probe["prompt_id"]]
                for i, pid in enumerate(result["prompt_subset"]):
                    arrays[f"prompt_{i}_derivative"] = result[
                        "per_prompt_directional_contributions"
                    ][pid].mean(0)[0]
                bound = _write_arrays(root / bank_id / f"{name}_AD.npz", arrays)
                gpu._publish(
                    root / bank_id / f"{name}_SCORE_ROWS.json", result["sample_score_records"]
                )
                summary = {
                    "derivative_method": "FULL_REVERSE_SCORE_GRADIENT",
                    "expansion_point": name,
                    "arrays": bound,
                    "score_rows": gpu._binding(root / bank_id / f"{name}_SCORE_ROWS.json"),
                    "anchor_prompt_ids": [p["prompt_id"] for p in probes],
                    "parameter_order": result["parameter_order"],
                    "parameter_rounding": rounding,
                    "cost": result["cost"],
                    "prompt_subset": list(prompt_subset),
                    "groups": result["groups"],
                    "finite_difference_is_derivative": False,
                    "reachable_training_policy": name == "BASE",
                    "reference_labels_read": False,
                }
                gpu._publish(root / bank_id / f"{name}_AD.json", summary)
                record[name] = summary
            records.append(record)
        # All eight AD reports now exist before any numerical response query.
        for record in records:
            bank_id = record["bank_id"]
            base_policy, end_policy = record["baseline_policy"], record["candidate_policy"]
            baseline = runtime["checkpoint_cache"].load(base_policy)["parameters"]
            candidate = runtime["checkpoint_cache"].load(end_policy)["parameters"]
            direction = {
                n: (candidate[n].double() - baseline[n].double()).cpu().numpy()
                for n in _layout(runtime)
            }
            for name, fraction in (("BASE", 0.0), ("MIDPOINT", 0.5)):
                with np.load(record[name]["arrays"]["path"], allow_pickle=False) as payload:
                    derivative = payload["group_derivative"]
                checks = []
                for h in steps:
                    check_path = root / bank_id / f"{name}_FD_{canonical_hash(h)}.json"
                    if check_path.exists():
                        if not resume:
                            raise FileExistsError(check_path)
                        check = gpu._read(check_path)
                        if sha256_file(check["arrays"]["path"]) != check["arrays"]["sha256"]:
                            raise ValueError("Saved numerical-check artifact changed")
                        checks.append(check)
                        continue
                    perturbations = []
                    values = []
                    costs = []
                    raw_records = {}
                    positive_parameters = None
                    for sign in (1, -1):
                        perturbations.append(
                            _set_point(
                                runtime, base_policy, baseline, direction, fraction + sign * h
                            )
                        )
                        if sign == 1:
                            positive_parameters = {
                                n: p.detach().double().cpu().clone().numpy()
                                for n, p in _layout(runtime).items()
                            }
                        value, cost, raw = _anchor_event_means(runtime, probes, samples)
                        raw_records.update(
                            {("plus_" if sign == 1 else "minus_") + k: v for k, v in raw.items()}
                        )
                        values.append(value)
                        costs.append(cost)
                    effective_error, effective_norm, expected_norm = 0.0, 0.0, 0.0
                    for n, p in _layout(runtime).items():
                        realized = (positive_parameters[n] - p.detach().double().cpu().numpy()) / (
                            2 * h
                        )
                        effective_error += float(np.sum((realized - direction[n]) ** 2))
                        effective_norm += float(np.sum(realized**2))
                        expected_norm += float(np.sum(direction[n] ** 2))
                    del positive_parameters
                    finite = (values[0] - values[1]) / (2 * h)
                    binding = _write_arrays(
                        root / bank_id / f"{name}_FD_{canonical_hash(h)}.npz",
                        {
                            "central_difference": finite,
                            "ad_derivative": derivative,
                            "difference": finite - derivative,
                            **raw_records,
                        },
                    )
                    check = {
                        "step": h,
                        "arrays": binding,
                        "max_abs_error": float(np.max(np.abs(finite - derivative))),
                        "actual_perturbations": perturbations,
                        "effective_direction_norm": math.sqrt(effective_norm),
                        "effective_direction_error_l2": math.sqrt(effective_error),
                        "effective_direction_relative_error": math.sqrt(
                            effective_error / expected_norm
                        )
                        if expected_norm
                        else None,
                        "rounded_perturbation_vanished": bool(expected_norm and not effective_norm),
                        "costs": costs,
                        "same_independent_anchor_packet": True,
                        "derivative_replaced": False,
                        "reference_truth_used": False,
                    }
                    gpu._publish(check_path, check)
                    checks.append(check)
                record[name] = {**record[name], "numerical_checks": checks}
    finally:
        runtime["restore_complete"](original)
        runtime["active_policy_identity"] = active
        if fast_scorer is not None:
            runtime["derivative_scorer"] = fast_scorer
        if fast_certificate is not None:
            runtime["derivative_certificate"] = fast_certificate
        attempt = len(list(root.glob("COST_ATTEMPT_*.json")))
        gpu._publish(
            root / f"COST_ATTEMPT_{attempt:04d}.json",
            {
                "autograd_calls": runtime.get("score_cost_ledger", [])[cost_offset:],
                "numerical_calls": runtime.get("directional_numeric_cost_ledger", [])[
                    numeric_cost_offset:
                ],
            },
        )
    return _finish(
        root,
        identity,
        {
            "kind": "V4_DEVELOPMENT_DIRECTIONAL_DIAGNOSTIC",
            "origin_id": forks["origin_id"],
            "runtime_identity": runtime["identity"],
            "banks": records,
            "stage_role": stage_role,
            "reference_labels_read": False,
            "online_ssvc": False,
        },
    )
