"""R3 warm-origin fork orchestration and predeclared direct validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import RunStore, canonical_hash, file_hash, write_json
from .optimizer_fork import load_checkpoint, parameter_hash, state_hash
from .r3_runtime import (
    _candidate_checkpoint,
    _collect_samples,
    _json,
    _optimizer_step,
    _Profile,
    _restore,
    _run_r3,
    _write_csv,
    validate_sample_ledger,
)

COUPLING = {
    "method": "PER_SEQUENCE_COMMON_RANDOM_SEED_WITHIN_BANK_LAMBDA0_LAMBDA1",
    "seed_root": 20260909,
    "shared_across": (
        "same warm origin, bank, prompt and rollout index; candidate lambda excluded from RNG key"
    ),
    "sample_identity": "candidate checkpoint and parameters remain part of distinct sample keys",
    "limits": (
        "Same seed is an attempted common-random-number coupling. Variable token paths/lengths "
        "can consume randomness differently; no natural output identity, token pairing, "
        "or category transport is claimed."
    ),
}


def build_direct_requests(plan, identity, candidate, bank_index, candidate_state_hash):
    if bank_index not in (0, 6) or float(candidate["lambda"]) not in (0.0, 1.0):
        raise ValueError("Direct validation is fixed to banks 0/6 and lambda 0/1")
    requests = []
    for prompt in plan["control_prompts"]:
        for index in range(16):
            rng = {
                "phase": "R3-warm/direct",
                "seed_root": 20260909,
                "warm_origin_hash": identity["origin_hash"],
                "model_hash": identity["model_hash"],
                "config_hash": identity["config_hash"],
                "data_hash": identity["data_hash"],
                "source_hash": identity["source_hash"],
                "bank_index": bank_index,
                "prompt_id": prompt["prompt_id"],
                "prompt_hash": prompt["prompt_hash"],
                "index": index,
            }
            rng_key = canonical_hash(rng)
            request = {
                "phase": "R3-warm",
                "bank_role": "direct_control",
                "bank_index": bank_index,
                "candidate_id": candidate["candidate_id"],
                "candidate_lambda": float(candidate["lambda"]),
                "candidate_parameter_hash": candidate["parameter_hash"],
                "candidate_optimizer_state_hash": candidate["optimizer_state_hash"],
                "candidate_origin_state_hash": candidate_state_hash,
                "candidate_updates_from_origin": 1,
                "candidate_optimizer_step": candidate["optimizer_step"],
                "origin_checkpoint_step": 64,
                "origin_state_hash": identity["origin_hash"],
                "checkpoint_step": 64,
                "base_scene_id": prompt["base_scene_id"],
                "prompt_id": prompt["prompt_id"],
                "group_id": prompt["prompt_id"],
                "family": prompt["family"],
                "interface": prompt["interface"],
                "split": "control",
                "prompt_hash": prompt["prompt_hash"],
                "scene_hash": prompt["scene_hash"],
                "sample_index": index,
                "rollout_index": index,
                "decode_mode": "sample",
                "max_new_tokens": 64,
                "enable_thinking": False,
                "sample_rng_key": rng_key,
                "sample_seed": int(rng_key[:8], 16) % (2**31),
            }
            requests.append(
                {
                    **request,
                    "sample_key": canonical_hash({"identity": identity, "request": request}),
                }
            )
    if len(requests) != 768 or len({r["sample_key"] for r in requests}) != 768:
        raise ValueError("Warm direct candidate requires 48 prompts x 16 distinct samples")
    return requests


def load_warm_origin(adapter, optimizer, context):
    checkpoint = context["checkpoint"]
    path = Path(checkpoint["path"])
    if file_hash(path) != checkpoint["file_sha256"]:
        raise ValueError("Verified R4 warm checkpoint bytes changed")
    state = load_checkpoint(path, checkpoint["identity"])
    if (
        state_hash(state) != checkpoint["state_hash"]
        or state_hash(state["optimizer"]) != checkpoint["optimizer_state_hash"]
    ):
        raise ValueError("Verified R4 warm optimizer/RNG/metadata hash changed")
    metadata = state["metadata"]
    keys = metadata.get("completed_sample_keys", [])
    adam = state["optimizer"]["state"]
    if (
        metadata.get("arm") != "X_BASE"
        or metadata.get("checkpoint_step") != 64
        or metadata.get("sampler", {}).get("position") != 64
        or len(keys) != len(set(keys))
        or len(keys) != 2048
        or not adam
        or any(float(value["step"]) != 64 for value in adam.values())
        or state["scheduler"] is not None
    ):
        raise ValueError("Warm origin must contain exact X_BASE step64 Adam/sampler/sample ledger")
    _restore(adapter, optimizer, state)
    if parameter_hash(adapter.model, trainable=True) != checkpoint["parameter_hash"]:
        raise ValueError("Warm checkpoint adapter parameters differ from R4 manifest")
    return state


def run_direct_validation(
    adapter,
    optimizer,
    origin,
    plan,
    proposal,
    summaries,
    attempts,
    scores_by_bank,
    out,
    identity,
    config,
    data_root,
    meter,
    profile,
    context,
):
    from .r3_warm_response import analyze_direct_validation

    request_identity = {**identity, "origin_hash": state_hash(origin)}
    coupling = {
        **COUPLING,
        "identity": request_identity,
        "bank_indices": [0, 6],
        "lambdas": [0.0, 1.0],
        "total_direct_outputs": 3072,
    }
    coupling_path = out / "direct_coupling.json"
    if coupling_path.exists() and _json(coupling_path) != coupling:
        raise ValueError("Warm direct coupling changed on resume")
    write_json(coupling_path, coupling)
    all_records, candidate_records, analyses = [], [], {}
    for index in plan["response_banks"]:
        bank_rows = []
        selected = [
            candidate
            for candidate in summaries[index]["candidates"]
            if float(candidate["lambda"]) in (0.0, 1.0)
        ]
        if len(selected) != 2 or {float(c["lambda"]) for c in selected} != {0.0, 1.0}:
            raise ValueError(
                "Warm direct candidates differ from predeclared baseline/validity pair"
            )
        for candidate in selected:
            state = _candidate_checkpoint(candidate, attempts[index])
            actual_step = _optimizer_step(state)
            if actual_step != _optimizer_step(origin) + 1:
                raise ValueError("Warm candidate Adam must be exactly one step after warm origin")
            candidate = {**candidate, "optimizer_step": actual_step}
            direct_identity = {
                **request_identity,
                "unit": "direct_control",
                "candidate_id": candidate["candidate_id"],
                "initial_adapter_hash": candidate["parameter_hash"],
                "candidate_state_hash": state_hash(state),
                "candidate_optimizer_state_hash": candidate["optimizer_state_hash"],
                "coupling_hash": canonical_hash(coupling),
            }
            requests = build_direct_requests(
                plan, request_identity, candidate, index, state_hash(state)
            )
            root = out / "direct_samples" / candidate["candidate_id"]
            store = RunStore(root, direct_identity, resume=(root / "identity.json").exists())
            validate_sample_ledger(store.records, requests)
            write_json(
                root / "request_manifest.json", {"identity": direct_identity, "requests": requests}
            )
            direct_profile = _Profile(
                out, profile.invocation, meter, profile.fixture, config, direct_identity
            )
            try:
                _restore(adapter, optimizer, state)
                _collect_samples(
                    adapter,
                    optimizer,
                    state,
                    plan,
                    requests,
                    store,
                    data_root,
                    meter,
                    direct_profile,
                )
                validate_sample_ledger(store.records, requests)
                if len(store.records) != 768:
                    raise ValueError("Warm direct sample coverage is incomplete")
                rows = [store.records[r["sample_key"]] for r in requests]
                all_records.extend(rows)
                bank_rows.extend(rows)
                candidate_records.append(
                    {
                        "bank_index": index,
                        "candidate_id": candidate["candidate_id"],
                        "lambda": candidate["lambda"],
                        "candidate_state_hash": state_hash(state),
                        "origin_checkpoint_step": 64,
                        "candidate_optimizer_step": actual_step,
                        "candidate_parameter_hash": candidate["parameter_hash"],
                        "candidate_optimizer_state_hash": candidate["optimizer_state_hash"],
                        "source_fork_attempt": str(attempts[index]),
                        "new_outputs": len(rows),
                        "sample_file": str(root / "samples.jsonl"),
                        "sample_file_sha256": file_hash(root / "samples.jsonl"),
                    }
                )
            finally:
                _restore(adapter, optimizer, origin)
                profile.write(
                    "DIRECT_VALIDATION",
                    completed=sum(r["new_outputs"] for r in candidate_records),
                    total=3072,
                )
        baseline = next(c["candidate_id"] for c in selected if c["lambda"] == 0.0)
        valid = next(c["candidate_id"] for c in selected if c["lambda"] == 1.0)
        analyses[index] = analyze_direct_validation(
            proposal,
            scores_by_bank[index],
            bank_rows,
            bank_index=index,
            baseline_key=baseline,
            candidate_key=valid,
            bootstrap_replicates=5000,
            seed=20260909,
        )
        if analyses[index].get("analysis_completed") is not True:
            raise ValueError("Warm direct response analysis did not complete")
        write_json(out / f"direct_validation_bank_{index:02d}.json", analyses[index])
    if len(all_records) != 3072 or len({row["sample_key"] for row in all_records}) != 3072:
        raise ValueError("Warm direct validation must contain 3072 distinct candidate outputs")
    effects = []
    for index, analysis in analyses.items():
        for comparison, value in analysis["comparisons"].items():
            for scope, metrics in value["responses"].items():
                for metric, measurement in metrics.items():
                    row = {
                        "bank_index": index,
                        "comparison": comparison,
                        "scope": scope,
                        "metric": metric,
                        **{k: v for k, v in measurement.items() if k != "ci"},
                    }
                    for kind, interval in measurement["ci"].items():
                        row.update({f"{kind}_CI_{k}": v for k, v in interval.items()})
                    effects.append(row)
    _write_csv(out / "direct_validation_effects.csv", effects)
    write_json(
        out / "direct_manifest.json",
        {
            "identity": request_identity,
            "coupling": coupling,
            "candidates": candidate_records,
            "direct_outputs": len(all_records),
            "statistical_status": {str(k): v["status"] for k, v in analyses.items()},
            "candidate_commit": False,
        },
    )
    if file_hash(context["checkpoint"]["path"]) != context["checkpoint"]["file_sha256"]:
        raise ValueError("R4 warm origin checkpoint changed during R3-warm")
    return {
        "direct_rollouts": 3072,
        "direct_candidates": 4,
        "direct_resample_warm": True,
        "warm_checkpoint": context["checkpoint"],
        "direct_validation_status": {str(k): v["status"] for k, v in analyses.items()},
        "coupling_contract": COUPLING,
    }


def run_r3_warm(
    config,
    data_root,
    out,
    r0_dir,
    r1_run,
    supplement_dir,
    r2_dir,
    r3_cold_dir,
    r4_dir,
    *,
    checkpoint=None,
    resume=False,
    _adapter_factory=None,
):
    from .next_stage_runtime import (
        validate_config_against_gate,
        validate_prerequisites,
        validate_r2_gate,
        validate_r3_cold_gate,
        validate_r4_gate,
    )

    gate = validate_prerequisites(r0_dir, r1_run, supplement_dir)
    config = validate_config_against_gate(config, gate)
    r2 = validate_r2_gate(r2_dir, gate)
    cold = validate_r3_cold_gate(r3_cold_dir, gate, r2)
    r4 = validate_r4_gate(r4_dir, gate, r2, cold)
    selected = dict(r4["warm_checkpoint"])
    path = Path(selected["path"]).resolve()
    if checkpoint is not None and Path(checkpoint).resolve() != path:
        raise ValueError("Explicit warm checkpoint must exactly equal verified R4 X_BASE step64")
    selected["path"] = str(path)
    if (
        config["R3"]["direct_resample_warm"] is not True
        or config["R3"]["direct_lambda_candidates"] != [0.0, 1.0]
        or config["R3"]["warm_checkpoint"] != "X_BASE/step64"
    ):
        raise ValueError("Warm direct protocol differs from locked configuration")
    if r4.get("r3_cold_plan_hash", cold["plan_hash"]) != cold["plan_hash"]:
        raise ValueError("R4 bound cold prompt plan differs from verified R3-cold")
    context = {
        "gate": gate,
        "r2_binding": r2,
        "r3_binding": cold,
        "r4_binding": r4,
        "checkpoint": selected,
    }
    return _run_r3(
        config,
        data_root,
        out,
        r0_dir,
        r1_run,
        supplement_dir,
        r2_dir,
        resume=resume,
        _adapter_factory=_adapter_factory,
        _warm_context=context,
    )


def main(argv=None):
    from .next_stage_common import load_yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    for name in (
        "data-root",
        "out",
        "r0-dir",
        "r1-run",
        "supplement-dir",
        "r2-dir",
        "r3-dir",
        "r4-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_r3_warm(
        load_yaml(args.config),
        args.data_root,
        args.out,
        args.r0_dir,
        args.r1_run,
        args.supplement_dir,
        args.r2_dir,
        args.r3_dir,
        args.r4_dir,
        checkpoint=args.checkpoint,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
