"""Q6 absolute event observations from frozen source states, with no new training.

Plans contain DIRECT generation tasks. Every generated action is already scored
for parity by the production observer. The CPU reader rechecks original rows,
counts actual events, and retains finite-sample uncertainty even for zero counts.

Precision limitation: the normal empirical interval has zero width at zero/all
counts; its replacement is the separately reported [0,1] Hoeffding bound. With
the frozen family/looks and 65,536-draw ceiling this can remain around 0.01, above
the 0.00025 primary goal. Such cells stay UNRESOLVED. The implementation does not
change interval rules or exceed the registered looks to obtain qualification.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import statistics
from array import array
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
from .vlm_campaign import _bound_json
from .vlm_observation import (
    EVENTS,
    PATH,
    _load_task_inputs,
    _logps,
    _parity_values,
    _read_binding,
    _task_requests,
    freeze_observation_task,
    freeze_task_manifest,
    iter_completed_task_rows,
    validate_action,
    validate_task_manifest,
)


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _identity(config, fixture):
    return {
        "config_hash": canonical_hash(config),
        "source_hash": source_identity()["sha256"],
        "fixture": fixture,
    }


def _cpu(fixture):
    if not fixture and (platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID")):
        raise ValueError("Real Q6 artifact processing requires server CPU under Slurm")


def _complete(binding):
    value = _bound_json(binding)
    root = Path(binding["path"]).resolve().parent
    if not (root / "COMPLETE.json").is_file() or verify_manifest(root).get("status") != "COMPLETE":
        raise ValueError("Q6 needs complete immutable original artifacts")
    return value


def absolute_protocol(config, prompts):
    reference = config["qwen"]["reference"]
    looks = [reference["initial_draws"], *reference["next_draws"]]
    if looks != [4096, 8192, 16384, 32768, 65536]:
        raise ValueError("Q6 must use the original frozen reference looks")
    return {
        "looks": looks,
        "alpha": 1 - config["statistics"]["coverage_nominal"],
        "family_cells": len(config["qwen"]["seed_roles"]["locked_test"]) * 17 * prompts,
        "primary_half_width_goal": reference["primary_half_width_goal"],
        "fallback_reporting_scale": reference["half_width_fallback_reporting"],
        "event_range": [0, 1],
        "cross_seed_95": "NOT_CERTIFIED",
        "precision_only_extension": True,
    }


def absolute_statistics(counts, n, protocol):
    counts = np.asarray(counts)
    if (
        counts.ndim != 3
        or counts.shape[-1] != 4
        or counts.dtype.kind not in "iu"
        or (counts < 0).any()
        or not np.all(counts.sum(-1) == n)
        or n < 2
    ):
        raise ValueError(
            "Absolute event counts must be nonnegative integers summing to actual draws"
        )
    p = counts.astype(float) / n
    delta = protocol["alpha"] / (len(protocol["looks"]) * protocol["family_cells"] * 4)
    formal = math.sqrt(math.log(2 / delta) / (2 * n))
    se = np.sqrt(p * (1 - p) / (n - 1))
    empirical = statistics.NormalDist().inv_cdf(1 - delta / 2) * se
    width = np.where(se == 0, formal, empirical)
    met = bool(np.max(width) <= protocol["primary_half_width_goal"])
    next_looks = [look for look in protocol["looks"] if look > n]
    return {
        "probabilities": p.tolist(),
        "half_width": width.tolist(),
        "empirical_normal_half_width": empirical.tolist(),
        "formal_hoeffding_half_width": formal,
        "formal_precision_met": formal <= protocol["primary_half_width_goal"],
        "empirical_precision_met": met,
        "zero_count_is_not_zero_uncertainty": True,
        "status": "ABSOLUTE_EMPIRICAL_PRECISION_MET"
        if met
        else "REFERENCE_EXTEND"
        if next_looks
        else "REFERENCE_UNRESOLVED",
        "next_draws": next_looks[0] if next_looks and not met else None,
        "reference_is_exact": False,
        "uncertainty_scope": "EMPIRICAL_NORMAL_WITH_FORMAL_BOUND_FOR_ZERO_VARIANCE",
        "familywise_measurement_alpha": protocol["alpha"],
        "alpha_per_cell_event_look": delta,
    }


def _qualified_inputs(config, inputs, fixture):
    from .tracking_offline import _source_window, _tracking_model
    from .vlm_results import verify_vlm_qualification

    qualification = verify_vlm_qualification(
        config, inputs["qualification_binding"], fit_binding=inputs["fit_binding"], fixture=fixture
    )
    if (
        qualification.get("pointwise_qualified") is not True
        or qualification.get("tracking_qualified") is not True
        or qualification.get("design_id")
        not in qualification.get("tracking_eligible_design_ids", [])
    ):
        raise PermissionError("Q6 requires an identified qualified linear pointwise design")
    fit, basis, coefficients = _tracking_model(
        config, inputs["fit_binding"], inputs["model_binding"]
    )
    updates, checkpoints = _source_window(config, inputs["source_binding"], fit, [1, 2, 4, 8, 16])
    return qualification, fit, basis, coefficients, updates, checkpoints


def _prediction(config, binding, inputs, fixture):
    from .tracking_offline import verify_tracking_prediction_lock

    return verify_tracking_prediction_lock(config, binding, inputs, fixture=fixture)


def prepare_q6_observation(
    config,
    bindings,
    *,
    qualification_binding,
    fit_binding,
    model_binding,
    source_binding,
    out,
    phase="anchor",
    previous_measurement=None,
    prediction_lock=None,
    fixture=False,
    fixture_draws=None,
):
    """Prepare an exact two-worker job manifest; never load a model or submit a job."""
    _cpu(fixture)
    if phase not in {"anchor", "reference"}:
        raise ValueError("Q6 phase must be anchor or reference")
    inputs = {
        "qualification_binding": qualification_binding,
        "fit_binding": fit_binding,
        "model_binding": model_binding,
        "source_binding": source_binding,
    }
    qualified, fit, _basis, _coef, _updates, checkpoints = _qualified_inputs(
        config, inputs, fixture
    )
    from .vlm_campaign import _stage_gate, planned_runtime_identity

    stage = _bound_json(bindings["v3_stage_lock"])
    seed = int(fit["origin_id"].split("_", 1)[0])
    if not fixture:
        _stage_gate(config, bindings, seed, operation="observe-vlm")
    if (
        stage.get("phase") != "Q6"
        or stage.get("qualification_binding") != qualification_binding
        or stage.get("source_binding") != source_binding
        or stage.get("design_id") != qualified["design_id"]
    ):
        raise ValueError("Q6 observation differs from the frozen source/qualification/design stage")
    bundle = fit["measurement_bundle"]
    generation = _bound_json(bundle["generation_manifest"])
    validate_task_manifest(generation, workers=generation["workers"])
    prompt_binding = generation["tasks"][0]["prompt_file"]
    if any(task["prompt_file"] != prompt_binding for task in generation["tasks"]):
        raise ValueError("Q6 must retain the exact fitted probe input file")
    prompts, _ = _load_task_inputs(
        {"prompt_file": prompt_binding, "prompt_ids": fit["probe_ids"], "sample_files": []}
    )
    if not set(fit["probe_ids"]) <= prompts.keys():
        raise ValueError("Fitted Q6 probes are absent from the original input")
    protocol = absolute_protocol(config, len(fit["probe_ids"]))
    if phase == "reference":
        if prediction_lock is None:
            raise PermissionError(
                "Freeze Q6 predictions before preparing dense reference observations"
            )
        _prediction(config, prediction_lock, inputs, fixture)
    elif prediction_lock is not None:
        raise ValueError("Anchor acquisition must precede predictions")
    previous = None
    if previous_measurement is not None:
        previous = verify_q6_absolute(config, previous_measurement, fixture=fixture)
        previous_plan = _complete(previous["plan_binding"])
        if (
            previous_plan["inputs"] != inputs
            or previous_plan["phase"] != phase
            or previous_plan["prediction_binding"] != prediction_lock
            or previous["next_draws"] is None
        ):
            raise ValueError(
                "Q6 continuation must follow the same unresolved precision-only measurement"
            )
    start, stop = (
        (previous["draws"], previous["next_draws"]) if previous else (0, protocol["looks"][0])
    )
    if fixture_draws is not None:
        if (
            not fixture
            or previous
            or type(fixture_draws) is not int
            or not 2 <= fixture_draws <= 16
        ):
            raise ValueError("Tiny fixture draw override requires an explicit initial fixture")
        stop = fixture_draws
    states = [64] if phase == "anchor" else list(range(65, 81))
    policies = {}
    source = _bound_json(source_binding)
    originals = {record["step"]: record for record in source["checkpoints"]}
    for record in checkpoints:
        step = record["step"]
        if step in states:
            original = originals[step]
            policies[f"state_{step}"] = {
                "checkpoint": {
                    "path": original["path"],
                    "sha256": original["checkpoint_sha256"],
                    "identity": original["checkpoint_identity"],
                },
                "inference_fingerprint": original["inference_fingerprint"],
            }
    runtime = planned_runtime_identity(config, bindings)
    tasks = [
        freeze_observation_task(
            operation="generate",
            origin_id=fit["origin_id"],
            candidate_id=f"state_{step}",
            prompt_file=prompt_binding,
            prompt_ids=[prompt],
            role="main" if phase == "anchor" else "reference",
            draw_start=start,
            draw_stop=stop,
            rng_namespace=canonical_hash(["Q6_ABSOLUTE", source_binding, phase]),
            proposal="DIRECT",
            proposal_candidates=[f"state_{step}"],
            worker=index % 2,
        )
        for step in states
        for index, prompt in enumerate(fit["probe_ids"])
    ]
    manifest = freeze_task_manifest(tasks, policies, runtime, workers=2)
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "CONFIG.json", config)
    atomic_json(root / "OBSERVATION_TASKS.json", manifest)
    child = {
        **stage,
        "authorization_parent": copy.deepcopy(bindings["v3_stage_lock"]),
        "runtime_seed": seed,
        "seeds": [seed],
        "operations": ["observe-vlm"],
        "observation_manifest_hashes": [manifest["manifest_hash"]],
        "observation_manifest": _binding(root / "OBSERVATION_TASKS.json"),
        "q6_phase": phase,
        "observation_purpose": "q6_absolute_" + phase,
        "origin_id": fit["origin_id"],
        "fixture": fixture,
        "fit_binding": fit_binding,
        "model_binding": model_binding,
        "prediction_binding": prediction_lock,
        "previous_measurement": previous_measurement,
    }
    atomic_json(root / "Q6_OBSERVATION_STAGE_LOCK.json", child)
    resolved = {**bindings, "v3_stage_lock": _binding(root / "Q6_OBSERVATION_STAGE_LOCK.json")}
    atomic_json(root / "SOURCE_BINDINGS.json", resolved)
    plan = {
        "kind": "V3_Q6_ABSOLUTE_PLAN",
        **_identity(config, fixture),
        "phase": phase,
        "inputs": inputs,
        "origin_id": fit["origin_id"],
        "probe_ids": fit["probe_ids"],
        "states": states,
        "draw_start": start,
        "draw_stop": stop,
        "protocol": protocol,
        "manifest": _binding(root / "OBSERVATION_TASKS.json"),
        "bindings": _binding(root / "SOURCE_BINDINGS.json"),
        "fit_generation_manifest": bundle["generation_manifest"],
        "prompt_binding": prompt_binding,
        "prediction_binding": prediction_lock,
        "previous_measurement": previous_measurement,
        "checkpoint_ledger": [r for r in checkpoints if r["step"] in states],
        "qualification": qualified,
        "model_loaded": False,
        "submitted": False,
        "status": "PREPARED_GPU_NOT_SUBMITTED",
        "online_feedback": False,
        "counts": {
            "new_generation_sequences": len(states) * len(fit["probe_ids"]) * (stop - start),
            "generation_parity_scored_sequences": len(states)
            * len(fit["probe_ids"])
            * (stop - start),
            "maximum_sampled_tokens": 64 * len(states) * len(fit["probe_ids"]) * (stop - start),
            "new_source_optimizer_steps": 0,
            "new_candidate_updates": 0,
        },
        "commands": [
            [
                "python",
                "-m",
                "src.modeling_v3.cli",
                "observe-vlm",
                "--config",
                str(root / "CONFIG.json"),
                "--bindings",
                str(root / "SOURCE_BINDINGS.json"),
                "--task-manifest",
                str(root / "OBSERVATION_TASKS.json"),
                "--worker-index",
                str(worker),
                "--workers",
                "2",
                "--device",
                "cuda:0",
                "--out",
                str(root / "observations"),
                "--allow-gpu",
                "--acknowledge-new-experiment",
            ]
            for worker in range(2)
        ],
    }
    atomic_json(root / "Q6_ABSOLUTE_PLAN.json", plan)
    finalize_run(root, _identity(config, fixture), metadata={"stage": "Q6", "submitted": False})
    return {**plan, "plan_binding": _binding(root / "Q6_ABSOLUTE_PLAN.json")}


def _plan(config, binding, fixture):
    plan = _complete(binding)
    if (
        plan.get("phase") not in {"anchor", "reference"}
        or plan.get("kind") != "V3_Q6_ABSOLUTE_PLAN"
        or any(plan.get(k) != v for k, v in _identity(config, fixture).items())
    ):
        raise ValueError("Q6 plan source/config/fixture identity changed")
    if not fixture:
        qualified, fit, _q, _coef, _updates, states = _qualified_inputs(
            config, plan["inputs"], False
        )
        if (
            qualified != plan["qualification"]
            or fit["origin_id"] != plan["origin_id"]
            or fit["probe_ids"] != plan["probe_ids"]
            or [r for r in states if r["step"] in plan["states"]] != plan["checkpoint_ledger"]
        ):
            raise ValueError("Q6 plan differs from qualified fit/source window originals")
    expected_states = [64] if plan["phase"] == "anchor" else list(range(65, 81))
    if plan["states"] != expected_states or plan["protocol"] != absolute_protocol(
        config, len(plan["probe_ids"])
    ):
        raise ValueError("Q6 state window or frozen uncertainty protocol changed")
    if not fixture and plan["draw_stop"] not in plan["protocol"]["looks"]:
        raise ValueError("Q6 measurement count is outside the frozen reference looks")
    if plan["phase"] == "reference":
        _prediction(config, plan["prediction_binding"], plan["inputs"], fixture)
    return plan


def _manifest(plan):
    manifest = _bound_json(plan["manifest"])
    validate_task_manifest(manifest, workers=2)
    expected = {(step, prompt) for step in plan["states"] for prompt in plan["probe_ids"]}
    seen = set()
    for task in manifest["tasks"]:
        state = int(task["candidate_id"].removeprefix("state_"))
        if len(task["prompt_ids"]) != 1:
            raise ValueError("Q6 bounds original-reader memory with one probe per task")
        prompt = task["prompt_ids"][0]
        pair = (state, prompt)
        if (
            pair not in expected
            or pair in seen
            or task["origin_id"] != plan["origin_id"]
            or task["operation"] != "generate"
            or task["proposal"] != "DIRECT"
            or task["proposal_candidates"] != [task["candidate_id"]]
            or task["role"] != ("main" if plan["phase"] == "anchor" else "reference")
            or task["rng_namespace"]
            != canonical_hash(["Q6_ABSOLUTE", plan["inputs"]["source_binding"], plan["phase"]])
            or task["draw_start"] != plan["draw_start"]
            or task["draw_stop"] != plan["draw_stop"]
            or task["prompt_file"] != plan["prompt_binding"]
            or task["sample_files"]
            or task["worker"] != plan["probe_ids"].index(prompt) % 2
        ):
            raise ValueError("Q6 task changed its frozen state/probe/draw/RNG manifest")
        seen.add(pair)
    if seen != expected or set(manifest["policies"]) != {f"state_{s}" for s in plan["states"]}:
        raise ValueError("Q6 manifest omits or adds a source state/probe")
    for record in plan["checkpoint_ledger"]:
        policy = manifest["policies"][f"state_{record['step']}"]
        if (
            policy["inference_fingerprint"] != record["inference_fingerprint"]
            or policy["checkpoint"]["path"] != record["path"]
            or policy["checkpoint"]["sha256"] != record["checkpoint_sha256"]
        ):
            raise ValueError(
                "Q6 observer policy differs from the actual retained source checkpoint"
            )
    return manifest


def verify_q6_observation_stage(config, bindings, stage, manifest):
    """Verify the sibling completed plan without creating a stage/plan hash cycle."""
    _cpu(False)
    from .vlm_campaign import planned_runtime_identity

    if _bound_json(bindings["v3_stage_lock"]) != stage:
        raise ValueError("Q6 execution stage differs from its bound original bytes")
    plan_path = Path(bindings["v3_stage_lock"]["path"]).resolve().parent / "Q6_ABSOLUTE_PLAN.json"
    plan = _plan(config, _binding(plan_path), False)
    expected = _manifest(plan)
    if (
        _bound_json(plan["bindings"]) != bindings
        or stage.get("phase") != "Q6"
        or stage.get("q6_phase") != plan["phase"]
        or stage.get("origin_id") != plan["origin_id"]
        or stage.get("prediction_binding") != plan["prediction_binding"]
        or any(stage.get(key) != value for key, value in plan["inputs"].items())
        or expected != manifest
        or manifest["runtime_identity"] != planned_runtime_identity(config, bindings)
        or stage.get("observation_manifest") != plan["manifest"]
        or stage.get("observation_manifest_hashes") != [manifest["manifest_hash"]]
    ):
        raise ValueError(
            "Q6 execution is not the exact completed source/fit/DIRECT observation plan"
        )
    return plan


def _worker_receipts(config, plan, manifest, root, fixture):
    resolved = _bound_json(plan["bindings"])
    if not fixture:
        from .vlm_observation import _validate_observation_stage, _worker_assignment

        _validate_observation_stage(config, resolved, manifest)
        for worker in range(2):
            receipt = json.loads(
                (Path(root) / f"worker_{worker}/EXECUTION_RECEIPT.json").read_text()
            )
            expected = sorted(
                t["task_id"]
                for t in manifest["tasks"]
                if _worker_assignment(t, manifest["policies"], 2) == worker
            )
            if (
                receipt.get("status") != "COMPLETE"
                or receipt.get("execution_kind") != "REAL_CUDA_MODEL"
                or receipt.get("execution_stage_binding") != resolved["v3_stage_lock"]
                or receipt.get("manifest_hash") != manifest["manifest_hash"]
                or receipt.get("runtime_identity") != manifest["runtime_identity"]
                or sorted(receipt.get("completed_tasks", [])) != expected
                or receipt.get("worker_index") != worker
                or receipt.get("workers") != 2
            ):
                raise ValueError(
                    "Q6 absolute observations require both original real worker receipts"
                )
    return resolved


def _compute(config, plan, worker_root, fixture):
    manifest = _manifest(plan)
    resolved = _worker_receipts(config, plan, manifest, worker_root, fixture)
    role = "main" if plan["phase"] == "anchor" else "reference"
    prompts, _ = _load_task_inputs(
        {"prompt_file": plan["prompt_binding"], "prompt_ids": plan["probe_ids"], "sample_files": []}
    )
    counts = np.zeros((len(plan["states"]), len(plan["probe_ids"]), 4), dtype=np.int64)
    seeds = array("Q")
    input_audits = {}
    for row in iter_completed_task_rows(worker_root, manifest):
        state = int(row["candidate_id"].removeprefix("state_"))
        if (
            state not in plan["states"]
            or row["prompt_id"] not in plan["probe_ids"]
            or row["role"] != role
            or row["proposal"] != "DIRECT"
            or row["origin_id"] != plan["origin_id"]
            or row["runtime_identity"] != manifest["runtime_identity"]
            or row["probability_execution"] != PATH
            or row["max_new_tokens"] != 64
            or row["proposal_candidates"] != [row["candidate_id"]]
            or row["inference_fingerprint"]
            != manifest["policies"][row["candidate_id"]]["inference_fingerprint"]
            or row["proposal_policy_fingerprints"] != [row["inference_fingerprint"]]
            or row["prompt_record_hash"] != canonical_hash(prompts[row["prompt_id"]])
            or row["sample_seed"] < 2**31
        ):
            raise ValueError(
                "Absolute draw is not the frozen state/input/proposal "
                "or overlaps the training RNG range"
            )
        flags = validate_action(row, eos_ids=row["eos_token_ids"], max_new_tokens=64)
        if any(row.get(k) != v for k, v in flags.items()):
            raise ValueError("Original Q6 token/EOS/terminal ledger changed")
        behavior = np.asarray(_logps(row["behavior_token_logprobs"], len(row["token_ids"])))
        score = np.asarray(_logps(row["rescored_token_logprobs"], len(row["token_ids"])))
        parity = _parity_values(
            [abs(behavior - score).tolist()],
            resolved["v3_probability_tolerances"],
            sequence_errors=[float(abs(behavior.sum() - score.sum()))],
        )
        if not parity["passed"] or abs(behavior.sum() - row["generation_sequence_logp"]) > 1e-10:
            raise ValueError("Q6 action generation and normalized scoring paths disagree")
        if row["category"] not in EVENTS or row["event_onehot"] != [
            int(row["category"] == e) for e in EVENTS
        ]:
            raise ValueError("Q6 absolute event category/onehot ledger changed")
        if not fixture:
            from ..r2_runtime import annotate_diagnostic

            if (
                annotate_diagnostic(row["raw_completion"], prompts[row["prompt_id"]]["scene"])[
                    "category"
                ]
                != row["category"]
            ):
                raise ValueError("Q6 original semantic label differs from current bound verifier")
        audit = row["input_audit"]
        image = prompts[row["prompt_id"]].get("interface") == "IMAGE_CUE_FRESH"
        audit_record = {
            "input_audit": audit,
            "eos_token_ids": row["eos_token_ids"],
            "probability_execution": row["probability_execution"],
            "max_new_tokens": row["max_new_tokens"],
        }
        if (
            audit.get("enable_thinking") is not False
            or not audit.get("final_prompt_hash")
            or not audit.get("input_tensor_hash")
            or bool(audit.get("image_token_count")) != image
            or (image and not audit.get("pixel_values_hash"))
            or (row["prompt_id"] in input_audits and input_audits[row["prompt_id"]] != audit_record)
        ):
            raise ValueError("Q6 states do not share the same non-thinking prompt inputs")
        input_audits[row["prompt_id"]] = audit_record
        counts[
            plan["states"].index(state),
            plan["probe_ids"].index(row["prompt_id"]),
            EVENTS.index(row["category"]),
        ] += 1
        seeds.append(row["sample_seed"])
    n = plan["draw_stop"] - plan["draw_start"]
    if not np.all(counts.sum(-1) == n):
        raise ValueError("Q6 must observe every fixed state/probe exactly once per planned draw")
    seed_array = np.frombuffer(seeds, dtype=np.uint64).copy()
    if np.unique(seed_array).size != seed_array.size:
        raise ValueError("Q6 original draws reuse completion RNG seeds")
    fit_generation = _bound_json(plan["fit_generation_manifest"])
    validate_task_manifest(fit_generation, workers=fit_generation["workers"])
    fit_seeds = np.fromiter(
        (
            r["sample_seed"]
            for task in fit_generation["tasks"]
            if task["operation"] == "generate"
            for r in _task_requests(task)
        ),
        dtype=np.uint64,
    )
    if np.intersect1d(seed_array, fit_seeds).size:
        raise ValueError("Q6 observations overlap fitted predictor sampling RNG")
    if plan["previous_measurement"] is not None:
        previous = verify_q6_absolute(config, plan["previous_measurement"], fixture=fixture)
        with np.load(_read_binding(previous["arrays"]), allow_pickle=False) as saved:
            prior_counts, prior_seeds = saved["counts"], saved["sample_seeds"]
        prior_plan = _complete(previous["plan_binding"])
        if (
            prior_plan["inputs"] != plan["inputs"]
            or prior_plan["phase"] != plan["phase"]
            or previous["draws"] != plan["draw_start"]
            or previous["next_draws"] != plan["draw_stop"]
            or previous["input_audits"] != input_audits
            or np.intersect1d(seed_array, prior_seeds).size
        ):
            raise ValueError("Q6 cumulative precision look changed inputs, range, or reused RNG")
        counts += prior_counts
        seed_array = np.concatenate([prior_seeds, seed_array])
    elif plan["draw_start"] != 0:
        raise ValueError("Q6 continuation requires its prior actual measurement")
    if plan["phase"] == "reference":
        lock = _prediction(config, plan["prediction_binding"], plan["inputs"], fixture)
        anchor = verify_q6_absolute(config, lock["anchor_binding"], fixture=fixture)
        with np.load(_read_binding(anchor["arrays"]), allow_pickle=False) as saved:
            if np.intersect1d(seed_array, saved["sample_seeds"]).size:
                raise ValueError("Q6 dense references overlap anchor predictor draws")
        if anchor["input_audits"] != input_audits:
            raise ValueError("Q6 anchor and dense references use different prompt inputs")
    return counts, seed_array, input_audits


def measure_q6_absolute(config, *, plan_binding, worker_root, out, fixture=False):
    """Count complete original actions on CPU; a declared half-width is never an input."""
    _cpu(fixture)
    plan = _plan(config, plan_binding, fixture)
    counts, seeds, audits = _compute(config, plan, worker_root, fixture)
    stats = absolute_statistics(counts, plan["draw_stop"], plan["protocol"])
    root = Path(out).resolve()
    originals = [Path(plan_binding["path"]).resolve().parent, Path(worker_root).resolve()]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in originals):
        raise ValueError("Q6 measurement output must not overlap immutable observation originals")
    root.mkdir(parents=True, exist_ok=False)
    atomic_npz(root / "ABSOLUTE_ARRAYS.npz", {"counts": counts, "sample_seeds": seeds})
    result = {
        "kind": "V3_Q6_ABSOLUTE_MEASUREMENT",
        **_identity(config, fixture),
        **stats,
        "phase": plan["phase"],
        "origin_id": plan["origin_id"],
        "states": plan["states"],
        "probe_ids": plan["probe_ids"],
        "draws": plan["draw_stop"],
        "plan_binding": plan_binding,
        "worker_root": str(Path(worker_root).resolve()),
        "arrays": _binding(root / "ABSOLUTE_ARRAYS.npz"),
        "input_audits": audits,
        "precision_only": True,
        "predictor_rankings_used": False,
        "cross_seed_95": "NOT_CERTIFIED",
        "online_ssvc": "NOT_CERTIFIED",
    }
    atomic_json(root / "ABSOLUTE_MEASUREMENT.json", result)
    finalize_run(
        root,
        _identity(config, fixture),
        metadata={"phase": plan["phase"], "draws": result["draws"]},
    )
    return {**result, "receipt": _binding(root / "ABSOLUTE_MEASUREMENT.json")}


def verify_q6_absolute(config, binding, *, fixture=False):
    """Recompute measurement counts and uncertainty from original task shards."""
    _cpu(fixture)
    result = _complete(binding)
    if (
        result.get("kind") != "V3_Q6_ABSOLUTE_MEASUREMENT"
        or any(result.get(k) != v for k, v in _identity(config, fixture).items())
        or result.get("precision_only") is not True
        or result.get("predictor_rankings_used") is not False
    ):
        raise ValueError("Q6 absolute measurement source/config/precision scope changed")
    plan = _plan(config, result["plan_binding"], fixture)
    counts, seeds, audits = _compute(config, plan, result["worker_root"], fixture)
    with np.load(_read_binding(result["arrays"]), allow_pickle=False) as saved:
        if not np.array_equal(saved["counts"], counts) or not np.array_equal(
            saved["sample_seeds"], seeds
        ):
            raise ValueError("Q6 measured arrays differ from original independent action rows")
    expected = {
        **absolute_statistics(counts, plan["draw_stop"], plan["protocol"]),
        "phase": plan["phase"],
        "origin_id": plan["origin_id"],
        "states": plan["states"],
        "probe_ids": plan["probe_ids"],
        "draws": plan["draw_stop"],
        "input_audits": audits,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Q6 declared probabilities or uncertainty differ from actual raw counts")
    return result
