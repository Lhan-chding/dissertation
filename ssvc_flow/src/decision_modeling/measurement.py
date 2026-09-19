"""Cached prefix observations with immutable, nested independent sample streams."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ..modeling_v3.vlm_observation import (
    ObservationFault,
    PrefixObservationBackend,
    _fault_json,
    _logps,
    _parity_values,
    digest,
    validate_action,
)
from ..modeling_v4 import gpu_collect as gpu
from .score_cache import ScoreCache, score_key
from .semantic_schema import semantic_features

LOOKS = (32, 128, 512)


class CachedObservationBackend(PrefixObservationBackend):
    """Cache evaluation scores only, keeping every draw and physical cost separate."""

    def __init__(self, *args, cache, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = cache

    def _score(self, prepared, tokens):
        key = score_key(
            self.current_fingerprint, prepared["audit"], tokens, eos_ids=self.adapter.eos_ids
        )

        def compute():
            with self._measurement():
                value = self.adapter.logprobs(prepared, tokens, require_grad=False)
                self.counters["scored_sequences"] += 1
                self.counters["scored_tokens"] += len(tokens)
                return _logps(value, len(tokens))

        return self.cache.score(key, length=len(tokens), compute=compute)

    def generate(self, prompt, *, seed, max_new_tokens=64):
        if self.current_fingerprint is None or max_new_tokens != 64:
            raise ValueError("Activate a policy with the frozen 64-token horizon")
        prepared = self._prepared(prompt)
        with self._measurement():
            raw = self.adapter.generate(prepared, seed=seed, max_new_tokens=64, do_sample=True)
            self.counters["generated_sequences"] += 1
            self.counters["generated_tokens"] += len(raw.get("token_ids", []))
        try:
            flags = validate_action(
                raw,
                eos_ids=self.adapter.eos_ids,
                max_new_tokens=64,
                tokenizer=self.adapter.processor.tokenizer,
            )
            behavior = _logps(raw["behavior_token_logprobs"], len(raw["token_ids"]))
            rescored = self._score(prepared, raw["token_ids"])
        except (ValueError, KeyError) as exc:
            raise ObservationFault(str(exc), raw) from exc
        parity = _parity_values(
            [[abs(a - b) for a, b in zip(behavior, rescored, strict=True)]],
            self.parity_tolerances,
            sequence_errors=[abs(math.fsum(behavior) - math.fsum(rescored))],
        )
        # A scoring mismatch disables LR/mass use but does not erase valid
        # endpoint samples. Retain the failed numeric evidence with the row.
        features = semantic_features(raw["raw_completion"], prompt)
        self.counters["accepted_samples"] += 1
        return {
            **raw,
            **flags,
            **features,
            "category": features["event"],
            "generation_sequence_logp": math.fsum(behavior),
            "rescored_token_logprobs": rescored,
            "generation_parity": parity,
            "prompt_id": prompt["prompt_id"],
            "base_scene_id": prompt["base_scene_id"],
            "family": prompt["family"],
            "interface": prompt["interface"],
            "prompt_record_hash": digest(prompt),
            "input_audit": prepared["audit"],
            "inference_fingerprint": self.current_fingerprint,
            "input_hash": prepared["audit"]["input_tensor_hash"],
            "scoring_usable": parity["passed"],
            "probability_execution": "uncached_prefix_recompute",
            "max_new_tokens": 64,
            "eos_token_ids": sorted(self.adapter.eos_ids),
        }

    def score(self, prompt, sample):
        prepared = self._prepared(prompt)
        if (
            sample["prompt_record_hash"] != digest(prompt)
            or sample["input_audit"] != prepared["audit"]
        ):
            raise ValueError("Scoring must use the exact sampled input")
        validate_action(
            sample,
            eos_ids=self.adapter.eos_ids,
            max_new_tokens=64,
            tokenizer=self.adapter.processor.tokenizer,
        )
        values = self._score(prepared, sample["token_ids"])
        return {
            "sample_id": sample["sample_id"],
            "token_logprobs": values,
            "sequence_logp": math.fsum(values),
            "inference_fingerprint": self.current_fingerprint,
        }


def _namespace(runtime):
    """Called once per observation process, not once per sample/token."""
    from ..modeling_v3.io import source_identity

    return {
        "runtime": runtime["identity"],
        # Operational load measurements vary across processes without changing
        # model probabilities; preserve every semantic adapter field.
        "adapter_audit": {
            key: value
            for key, value in runtime["adapter"].audit.items()
            if key not in ("load_seconds", "load_peak_cuda_bytes")
        },
        "implementation": source_identity()["sha256"],
        "path": "uncached_prefix_recompute",
        "generation": {"temperature": 1, "top_p": 1, "top_k": 0, "max_new_tokens": 64},
    }


def make_backend(runtime, cache):
    return CachedObservationBackend(
        runtime["adapter"],
        cache=cache,
        runtime_identity=runtime["identity"],
        policy_loader=runtime["policy_loader"],
        data_root=runtime["data_root"],
        parity_tolerances=runtime["parity_tolerances"],
        state_guard=runtime["state_guard"],
    )


def collect_stream(
    backend, policies, prompts, *, out, role, draws, stream_id, identity_prompts=None
):
    """One stream per role/policy; extending a look reuses whole 32-row chunks.

    MIX source coins are independent before grouping draws by source. Reordering
    physical generation therefore changes neither seeds nor mixture weights.
    """
    if draws not in (*LOOKS, 16, 256, 1024) or role not in (
        "endpoint",
        "origin",
        "mix",
        "discovery",
        "reference",
    ):
        raise ValueError("Unregistered role or fixed look")
    if len(policies) != (2 if role == "mix" else 1):
        raise ValueError("Only true MIX draws have two source policies")
    forward_ids = [p.get("inference_fingerprint", digest(p)) for p in policies]
    stream_id = f"{stream_id}:{role}:{digest(forward_ids)}"
    root = Path(out)
    identity = {
        "stream_id": stream_id,
        "role": role,
        "policies": policies,
        "prompts": digest(prompts if identity_prompts is None else identity_prompts),
        "cache_namespace": backend.cache.namespace,
    }
    gpu._publish(root / "IDENTITY.json", identity)
    all_rows = []
    for prompt in prompts:
        for start in range(0, draws, 32):
            stop = min(draws, start + 32)
            path = root / prompt["prompt_id"] / f"{start:04d}_{stop:04d}.json"
            if path.exists():
                chunk = gpu._read(path)
                if chunk["identity"] != digest(identity) or chunk["rows_hash"] != digest(
                    chunk["rows"]
                ):
                    raise ValueError("Frozen observation chunk changed")
                rows = chunk["rows"]
                if [r["draw_index"] for r in rows] != list(range(start, stop)):
                    raise ValueError("Incomplete observation prefix")
            else:
                requests = []
                for index in range(start, stop):
                    key = digest(
                        [
                            stream_id,
                            role,
                            [p.get("inference_fingerprint", digest(p)) for p in policies],
                            prompt["prompt_id"],
                            index,
                        ]
                    )
                    coin = int(digest([key, "source"]), 16) % len(policies)
                    requests.append((coin, index, key))
                rows = []
                try:
                    for coin, index, key in sorted(requests):
                        backend.activate(policies[coin])
                        row = backend.generate(
                            prompt, seed=int(digest([key, "completion"])[:16], 16) % (2**63)
                        )
                        rows.append(
                            {
                                **row,
                                "sample_id": key,
                                "sample_key": key,
                                "draw_index": index,
                                "role": role,
                                "rng_stream_id": stream_id,
                                "proposal_source_index": coin,
                            }
                        )
                except BaseException as exc:
                    gpu._publish(
                        path.with_name(f"{path.stem}_FAILURE_{time.time_ns()}.json"),
                        _fault_json(
                            {
                                "identity": digest(identity),
                                "completed_rows": rows,
                                "type": type(exc).__name__,
                                "error": str(exc),
                                "raw_result": getattr(exc, "raw_result", None),
                            }
                        ),
                    )
                    raise
                rows.sort(key=lambda row: row["draw_index"])
                gpu._publish(
                    path, {"identity": digest(identity), "rows": rows, "rows_hash": digest(rows)}
                )
            all_rows.extend(rows)
    return all_rows


def lr_contributions(samples, left_scores, right_scores, *, proposal):
    """Signed right-minus-left contribution; same-bank proposal noise cancels."""
    if proposal not in ("ORIGIN", "MIX") or not samples:
        raise ValueError("One nonempty ORIGIN or MIX packet required")
    if not len(samples) == len(left_scores) == len(right_scores):
        raise ValueError("Scores must align one-to-one with original sample identities")
    logs = np.array(
        [
            [left["sequence_logp"], right["sequence_logp"]]
            for left, right in zip(left_scores, right_scores, strict=True)
        ],
        dtype=float,
    )
    if not np.isfinite(logs).all() or (logs > 0).any():
        raise ValueError("Finite normalized sequence log probabilities required")
    result = []
    for i, (sample, left, right) in enumerate(zip(samples, left_scores, right_scores, strict=True)):
        if left["sample_id"] != sample["sample_id"] or right["sample_id"] != sample["sample_id"]:
            raise ValueError("Score/sample identity mismatch")
        if not sample["scoring_usable"]:
            raise ValueError("LR disabled: generation/scoring mismatch")
        proposal_log = (
            np.logaddexp(*logs[i]) - math.log(2)
            if proposal == "MIX"
            else sample["generation_sequence_logp"]
        )
        # Stable difference; expm1 keeps tiny same-policy effects from cancellation.
        a, b = logs[i]
        hi = max(a, b)
        scale = math.exp(hi - proposal_log)
        delta = scale * (-math.expm1(min(a, b) - hi)) * (1 if b >= a else -1)
        if not math.isfinite(delta):
            raise FloatingPointError("Unbounded ORIGIN importance ratio overflow")
        result.append([delta * int(sample["event"] == event) for event in "XSWI"])
    return np.asarray(result)


def observe(
    runtime,
    policies,
    prompts,
    *,
    out,
    look=32,
    role="endpoint",
    stream_id,
    cache_path=None,
    context=None,
    identity_prompts=None,
):
    """Actual model observation; caller supplies an already authorized runtime."""
    root = Path(out)
    started = time.perf_counter()
    with ScoreCache(cache_path or root / "scores.sqlite", namespace=_namespace(runtime)) as cache:
        backend = make_backend(runtime, cache)
        rows = collect_stream(
            backend,
            policies,
            prompts,
            out=root / "samples",
            role=role,
            draws=look,
            stream_id=stream_id,
            identity_prompts=identity_prompts,
        )
        result = {
            "status": "OBSERVED",
            "role": role,
            "look": look,
            **(context or {}),
            "panel_identity": digest(prompts if identity_prompts is None else identity_prompts),
            "total_rows": len(rows),
            "sample_ids_hash": digest([r["sample_id"] for r in rows]),
            "scoring_usable": all(r["scoring_usable"] for r in rows),
            "cost_this_invocation": {
                **backend.counters,
                **cache.counters,
                "wall_seconds": time.perf_counter() - started,
            },
            "execution_kind": runtime["identity"]["execution_kind"],
        }
        try:
            import torch

            result["cost_this_invocation"]["peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            )
        except ImportError:
            result["cost_this_invocation"]["peak_allocated_bytes"] = None
        gpu._publish(
            root / f"LOOK_{look}.json",
            {k: v for k, v in result.items() if k != "cost_this_invocation"},
        )
        gpu._publish(
            root / "costs" / f"invocation_{time.time_ns()}.json", result["cost_this_invocation"]
        )
    return result, rows


def bridge(
    runtime,
    policies,
    prompts,
    *,
    out,
    stream_id,
    origin_policy=None,
    identity_prompts=None,
    run_scoring_diagnostic=True,
):
    """D1: two endpoint streams plus fixed 24-action path comparison, no training.

    Prefix remains the scoring path after the diagnostic, including when an
    alternative happens to pass. Derivative parity is not inferred from it.
    """
    root = Path(out)
    results, samples = [], []
    for index, policy in enumerate(policies):
        result, rows = observe(
            runtime,
            [policy],
            prompts,
            out=root / f"endpoint_{index}",
            stream_id=f"{stream_id}:endpoint:{index}",
            cache_path=root / "scores.sqlite",
            identity_prompts=identity_prompts,
            context={
                "origin_id": "O1_PREVIEW",
                "horizon": 1,
                "candidate_id": policy["candidate_id"],
                "panel": "P",
            },
        )
        results.append(result)
        samples.append(rows)
    # Fixed first 24 actions of the first endpoint, chosen by index, not outcomes.
    diagnostic = (
        gpu._read(root / "SCORING_PATHS.json")
        if (root / "SCORING_PATHS.json").exists()
        else gpu.measure_scoring_paths(runtime, policies[0], prompts, samples[0][:24])
        if run_scoring_diagnostic
        else {"status": "DELEGATED_TO_WORKER_0", "tested_sequences": 0}
    )
    runtime["observation_score_mode"] = "uncached_prefix_recompute"
    certificate = {
        **diagnostic,
        "selected_for_production": "uncached_prefix_recompute",
        "derivative_path_verified": False,
    }
    gpu._publish(root / "SCORING_PATHS.json", certificate)
    usable = all(r["scoring_usable"] for r in results)
    comparisons = []
    if usable and origin_policy is not None:
        from .intervals import conditional_mass_bounds

        with ScoreCache(root / "scores.sqlite", namespace=_namespace(runtime)) as cache:
            backend = make_backend(runtime, cache)
            for proposal, source in (("ORIGIN", [origin_policy]), ("MIX", policies)):
                rows = collect_stream(
                    backend,
                    source,
                    prompts,
                    out=root / proposal.lower(),
                    role=proposal.lower(),
                    draws=32,
                    stream_id=f"{stream_id}:{proposal}",
                    identity_prompts=identity_prompts,
                )
                scores = []
                by_id = {p["prompt_id"]: p for p in prompts}
                for policy in policies:
                    backend.activate(policy)
                    scored = [backend.score(by_id[row["prompt_id"]], row) for row in rows]
                    scores.append(scored)
                    gpu._publish(
                        root / proposal.lower() / f"scores_{policy['candidate_id']}.json",
                        {"rows": scored, "rows_hash": digest(scored)},
                    )
                if not all(row["scoring_usable"] for row in rows):
                    comparisons.append({"proposal": proposal, "status": "DISABLED_SCORER_MISMATCH"})
                    continue
                contributions = lr_contributions(rows, scores[0], scores[1], proposal=proposal)
                units = []
                for prompt in prompts:
                    indices = [
                        i for i, row in enumerate(rows) if row["prompt_id"] == prompt["prompt_id"]
                    ]
                    raw = contributions[indices]
                    preserve = raw.copy()
                    preserve[:, 1:3] -= raw.sum(axis=1)[:, None] / 2
                    bounds = []
                    for scored in scores:
                        unique = {}
                        for i in indices:
                            key = tuple(rows[i]["token_ids"])
                            unique.setdefault(key, (rows[i], scored[i]["sequence_logp"]))
                        mass = {
                            event: math.fsum(
                                math.exp(lp) for row, lp in unique.values() if row["event"] == event
                            )
                            for event in "XSWI"
                        }
                        try:
                            bounds.append(
                                conditional_mass_bounds(
                                    mass,
                                    relation_mass=math.fsum(
                                        math.exp(lp) * row["relation_score"]
                                        for row, lp in unique.values()
                                    ),
                                )
                            )
                        except ValueError as exc:
                            bounds.append({"status": "INVALID_SCORER_MASS", "reason": str(exc)})
                    units.append(
                        {
                            "prompt_id": prompt["prompt_id"],
                            "n": len(indices),
                            "RAW4": raw.mean(0).tolist(),
                            "PRESERVE_XI": preserve.mean(0).tolist(),
                            "covariance_of_mean_RAW4": (
                                np.cov(raw, rowvar=False) / len(raw)
                            ).tolist(),
                            "covariance_of_mean_PRESERVE_XI": (
                                np.cov(preserve, rowvar=False) / len(raw)
                            ).tolist(),
                            "empirical_covariance_is_certificate": False,
                            "event_support_counts": {
                                e: sum(rows[i]["event"] == e for i in indices) for e in "XSWI"
                            },
                            "max_abs_contribution": np.abs(raw).max(0).tolist(),
                            "zero_empirical_variance_is_certainty": False,
                            "mass_bounds": bounds,
                        }
                    )
                comparison = {
                    "proposal": proposal,
                    "status": "MEASURED",
                    "units": units,
                    "ORIGIN_population_weight_bound": None,
                    "MIX_RAW_contribution_range": [-2, 2],
                    "same_policy_zero_effect": "EXACT_ALGEBRA_IF_IDENTICAL_FINGERPRINTS",
                }
                gpu._publish(root / proposal.lower() / "ANALYSIS.json", comparison)
                comparisons.append(comparison)
            gpu._publish(
                root / "costs" / f"lr_{time.time_ns()}.json", {**backend.counters, **cache.counters}
            )
    result = {
        "status": "D1_MEASURED",
        "endpoints": results,
        "scoring_paths": certificate,
        "lr_mass_enabled": usable
        and bool(comparisons)
        and all(item["status"] == "MEASURED" for item in comparisons),
        "proposal_comparisons": comparisons,
        "new_training_steps": 0,
        "automatic_successor": False,
    }
    gpu._publish(
        root / "COMPLETE.json",
        {
            **result,
            "endpoints": [
                {k: v for k, v in row.items() if k != "cost_this_invocation"} for row in results
            ],
        },
    )
    return result
