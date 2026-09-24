"""Independent-role sampling with complete training state isolation and raw output."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from ..modeling_v3.io import canonical_hash
from ..modeling_v4 import gpu_collect as gpu
from .branch_runtime import _restore_checked, make_backend


def isolated_evaluation(runtime, function):
    saved = runtime["state"].capture({"evaluation_isolation": True})
    try:
        return function()
    finally:
        _restore_checked(runtime, saved)


def summarize_rows(rows, *, require_six_strata=True):
    groups = defaultdict(list)
    by_prompt = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["interface"])].append(row)
        by_prompt[row["prompt_id"]].append(row)
    if not groups or (require_six_strata and len(groups) != 6):
        raise ValueError("Primary metric requires all six family/interface strata")

    def stats(items):
        n = len(items)
        semantics = [r["semantic"] for r in items]
        valid = [s for s in semantics if s["event"] != "I"]
        return {
            "n": n,
            "pX": sum(s["event"] == "X" for s in semantics) / n,
            "pI": sum(s["event"] == "I" for s in semantics) / n,
            "relation_full_wrong": sum(
                s["relation_score"] == 1 and s["event"] != "X" for s in semantics
            )
            / n,
            "coord_accuracy": sum(s["coord_accuracy"] for s in semantics) / n,
            "valid_n": len(valid),
            "valid_repair_mask": bool(valid),
            "damage_any_given_valid": (
                sum(s["damage_any"] for s in valid) / len(valid) if valid else None
            ),
            "F_given_valid": sum(s["F"] for s in valid) / len(valid) if valid else None,
            "B_given_valid": sum(s["B"] for s in valid) / len(valid) if valid else None,
            "M_given_valid": sum(s["M"] for s in valid) / len(valid) if valid else None,
            "single_edit_given_valid": (
                sum(s["single_edit"] for s in valid) / len(valid) if valid else None
            ),
            "copy_given_valid": sum(s["copy"] for s in valid) / len(valid) if valid else None,
            "Bdamage_all_outputs": sum(s["Bdamage"] for s in semantics) / n,
        }

    strata = {f"{f}/{i}": stats(r) for (f, i), r in sorted(groups.items())}
    # Equal prompt weights within equal strata; MC variance is conditional on this panel.
    mean = math.fsum(v["pX"] for v in strata.values()) / len(strata)
    variance = 0.0
    for items in groups.values():
        pids = {r["prompt_id"] for r in items}
        weight = 1 / len(groups) / len(pids)
        for pid in pids:
            values = [r["semantic"]["event"] == "X" for r in by_prompt[pid]]
            n = len(values)
            if n < 2:
                variance = float("nan")
                break
            p = sum(values) / n
            variance += weight**2 * p * (1 - p) / (n - 1)
    return {
        "J": mean,
        "six_stratum_macro_pX": mean,
        "strata": strata,
        "interfaces": {
            i: math.fsum(v["pX"] for (f, j), r in groups.items() if j == i for v in [stats(r)])
            / sum(j == i for f, j in groups)
            for i in sorted({i for f, i in groups})
        },
        "samples": len(rows),
        "prompts": len(by_prompt),
        "fixed_panel_mc_se": math.sqrt(variance) if math.isfinite(variance) else None,
        "observed_saturation_not_probability_one": any(v["pX"] == 1 for v in strata.values()),
    }


def collect_evaluation(
    runtime,
    checkpoint,
    prompts,
    *,
    out,
    lineage_id,
    origin_id,
    policy_id,
    panel_id,
    horizon,
    draws,
    repeat=0,
    experiment_seed=20260924,
    role="evaluation",
    resume=False,
    fixture=False,
):
    """Checkpoint policy uses an independent generation stream per policy and role."""
    from .sampler import evaluation_seed

    if draws < 1 or role not in ("evaluation", "predecision", "historical_E", "diagnostic"):
        raise ValueError("Positive draws and registered observation role required")
    if not prompts or len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("Unique nonempty evaluation prompts required")
    if any(p.get("split") == "train" for p in prompts):
        raise ValueError("Evaluation panels cannot contain training prompts")
    if not fixture:
        counts = {"P": 72, "D": 144, "T": 288, "T_H8": 24, "T8": 24, "D_H8": 24, "E_old": 72}
        budgets = {
            "P": (32,),
            "D": (16,),
            "T": (16, 32, 64),
            "T_H8": (8,),
            "T8": (8,),
            "D_H8": (8,),
            "E_old": (32,),
        }
        if (
            panel_id not in counts
            or len(prompts) != counts[panel_id]
            or draws not in budgets[panel_id]
        ):
            raise ValueError("Evaluation panel count/draw budget differs from registration")
    root = Path(out)
    identity = {
        "lineage_id": lineage_id,
        "origin_id": origin_id,
        "policy_id": policy_id,
        "panel_id": panel_id,
        "horizon": horizon,
        "draws": draws,
        "repeat": repeat,
        "experiment_seed": experiment_seed,
        "role": role,
        "checkpoint_state_hash": checkpoint["state_hash"],
        "prompts_hash": canonical_hash(prompts),
        "config_hash": runtime.get("prospective_config_hash"),
    }
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("Evaluation exists; explicit resume required")
    root.mkdir(parents=True, exist_ok=True)
    gpu._publish(root / "MANIFEST.json", identity)

    def execute():
        state = runtime["checkpoint_cache"].load(checkpoint)
        _restore_checked(runtime, state)
        runtime["adapter"].model.eval()
        backend = make_backend(runtime)
        backend.current_fingerprint = checkpoint["state_hash"]
        rows, chunks = [], []
        for prompt in prompts:
            chunk = root / "prompts" / canonical_hash(prompt["prompt_id"])
            receipt_path = chunk / "COMMIT.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["identity_hash"] != canonical_hash(identity):
                    raise ValueError("Evaluation chunk belongs to a different policy/role")
                gpu.verify_artifact_bindings(receipt)
                chunk_rows = [
                    json.loads(line)
                    for line in Path(receipt["samples"]["path"]).read_text().splitlines()
                ]
                if len(chunk_rows) != draws:
                    raise ValueError("Evaluation chunk count mismatch")
                rows.extend(chunk_rows)
                chunks.append(receipt)
                continue
            attempt = chunk / f"attempt_{time.time_ns()}"
            attempt.mkdir(parents=True)
            started = time.perf_counter()
            chunk_rows = []
            try:
                with (attempt / "samples.jsonl").open("x") as stream:
                    for draw in range(draws):
                        seed = evaluation_seed(
                            experiment_seed=experiment_seed,
                            lineage=lineage_id,
                            repeat=repeat,
                            policy_id=policy_id,
                            panel_id=panel_id,
                            horizon=horizon,
                            prompt_id=prompt["prompt_id"],
                            draw=draw,
                            role=role,
                        )
                        raw = backend.generate(prompt, seed=seed)
                        row = {
                            **raw,
                            "lineage_id": lineage_id,
                            "origin_id": origin_id,
                            "policy_id": policy_id,
                            "panel_id": panel_id,
                            "horizon": horizon,
                            "repeat": repeat,
                            "role": role,
                            "sample_seed": seed,
                            "draw_index": draw,
                            "sample_id": canonical_hash([identity, prompt["prompt_id"], draw]),
                        }
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        stream.flush()
                        chunk_rows.append(row)
                    os.fsync(stream.fileno())
                receipt = {
                    "identity_hash": canonical_hash(identity),
                    "samples": gpu._binding(attempt / "samples.jsonl"),
                    "count": len(chunk_rows),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                gpu._publish(receipt_path, receipt)
            except BaseException as exc:
                gpu._publish(
                    attempt / "FAILURE.json",
                    {"error": str(exc), "type": type(exc).__name__, "authoritative": False},
                )
                raise
            rows.extend(chunk_rows)
            chunks.append(receipt)
        summary = summarize_rows(rows, require_six_strata=not fixture)
        result = {
            "status": "CPU_FIXTURE_COMPLETE" if fixture else "EVALUATION_COMPLETE",
            "identity": identity,
            "summary": summary,
            "chunks": chunks,
            "elapsed_sampling_seconds": sum(c["elapsed_seconds"] for c in chunks),
            "generated_outputs": len(rows),
            "extra_scoring_calls": 0,
        }
        gpu._publish(root / "COMPLETE.json", result)
        return result, rows

    return isolated_evaluation(runtime, execute)
