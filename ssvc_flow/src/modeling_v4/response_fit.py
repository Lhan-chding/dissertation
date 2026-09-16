"""CPU fitting of completed V4 maps; reference labels are opened after prediction.

The only production entrypoint requires Linux/Slurm CPU. It verifies the task,
checkpoint, action and score bytes, fits calibration banks, publishes predictions,
then evaluates the independent reference. Source seeds, exact aliases and missing
reference precision remain explicit. No new actions or model forwards occur.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from ..modeling_v3.io import atomic_json, canonical_hash, finalize_run, sha256_file, source_identity
from ..modeling_v3.observation_geometry import (
    ContributionBatch,
    stable_origin_weight,
    stable_pair_weight,
)
from .analysis_rules import endpoint_overlap, load_analysis_rules, reference_precision
from .data_adapter import CONTRASTS, bound_json
from .evaluate import write_evaluation
from .functional_features import output_diagnostics
from .observations import PRIMARY_METHODS, assert_predictor_independence, observe_packet
from .storage import require_space

EVENTS = ("X", "S", "W", "I")


def _require_server_cpu():
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Actual response fitting requires server Linux CPU under Slurm")
    if any(
        os.environ.get(k, "") not in ("", "-1", "NoDevFiles", "(null)")
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
        )
    ):
        raise RuntimeError("Response fitting must use a CPU-only allocation")


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _verified_rows(receipt):
    """Validate every immutable shard and exact complete contiguous row count."""
    import pyarrow.parquet as pq

    expected = 0
    for chunk in receipt["chunks"]:
        if chunk["start"] != expected or chunk["stop"] <= expected:
            raise ValueError("Chunk coverage is incomplete or duplicated")
        if sha256_file(chunk["path"]) != chunk["sha256"]:
            raise ValueError("Action/score shard bytes differ from completion binding")
        rows = pq.read_table(chunk["path"]).to_pylist()
        if len(rows) != chunk["stop"] - chunk["start"]:
            raise ValueError("Shard row count differs from completion")
        yield from rows
        expected = chunk["stop"]
    if expected != receipt["count"]:
        raise ValueError("Complete shard count differs")


def _action_rows(receipt, probes, *, role, origin_id, policy_fingerprint):
    from ..modeling_v3.vlm_observation import validate_action

    identity = receipt["identity"]
    if (
        identity["role"] != role
        or identity["origin_id"] != origin_id
        or identity["prompt_hash"] != canonical_hash(probes)
    ):
        raise ValueError("Action receipt role/origin/prompt identity differs")
    expected = (
        (policy_fingerprint,) if isinstance(policy_fingerprint, str) else tuple(policy_fingerprint)
    )
    if identity["proposal"] not in ("ORIGIN", "DIRECT", "MIX") or len(identity["policies"]) != (
        2 if identity["proposal"] == "MIX" else 1
    ):
        raise ValueError("Actual ORIGIN, DIRECT or ordered two-policy MIX required")
    if tuple(p["inference_fingerprint"] for p in identity["policies"]) != expected:
        raise ValueError("Actual proposal policy differs")
    by_id = {p["prompt_id"]: p for p in probes}
    output = {p: [] for p in by_id}
    seen, seen_seeds = set(), set()
    for row in _verified_rows(receipt):
        pid, draw = row["prompt_id"], row["draw_index"]
        key = canonical_hash([identity["rng_namespace"], pid, draw])
        source_index = (
            int(canonical_hash([key, "mixture_source"])[0], 16) % 2
            if identity["proposal"] == "MIX"
            else 0
        )
        chosen = identity["policies"][source_index]
        seed = int(key[:16], 16) % (2**63)
        if (
            pid not in by_id
            or type(draw) is not int
            or not 0 <= draw < identity["draws"]
            or (pid, draw) in seen
            or row["sample_id"] != key
            or row["sample_key"] != key
            or row["role"] != role
            or row["origin_id"] != origin_id
            or row["rng_namespace"] != identity["rng_namespace"]
            or row["proposal_fingerprint"] != chosen["inference_fingerprint"]
            or row.get("proposal_policy_id") != chosen["candidate_id"]
            or row.get("sample_seed") != seed
            or seed in seen_seeds
            or row["base_scene_id"] != by_id[pid]["base_scene_id"]
            or row["runtime_identity"] != canonical_hash(identity["runtime"])
        ):
            raise ValueError("Actual draw identity differs from registered RNG/prompt/policy")
        flags = validate_action(row, eos_ids=row["eos_token_ids"], max_new_tokens=64)
        if row.get("eos_seen") != flags["eos"] or row.get("truncated") != flags["truncated"]:
            raise ValueError("Recorded EOS/truncation flags differ")
        if row.get("action_mask") != [1] * len(row["token_ids"]):
            raise ValueError("All actual sequence tokens, including EOS, must remain active")
        if (
            not math.isclose(
                math.fsum(row["behavior_token_logprobs"]),
                row["generation_sequence_logp"],
                abs_tol=1e-10,
                rel_tol=1e-12,
            )
            or row["event_onehot"] != [int(row["category"] == e) for e in EVENTS]
            or output_diagnostics(row["raw_completion"], by_id[pid]["scene"])["category"]
            != row["category"]
        ):
            raise ValueError("Action probability sum or exhaustive event label differs")
        seen.add((pid, draw))
        seen_seeds.add(seed)
        output[pid].append(row)
    if len(seen) != len(probes) * identity["draws"]:
        raise ValueError("Every frozen prompt/draw must be present")
    for rows in output.values():
        rows.sort(key=lambda r: r["draw_index"])
        if len({r["input_hash"] for r in rows}) != 1:
            raise ValueError("Frozen prompt input changed across draws")
    return output


def _score_rows(receipt, samples, policy, runtime):
    if (
        receipt["identity"]["policy"]["inference_fingerprint"] != policy["inference_fingerprint"]
        or receipt["identity"]["runtime"] != runtime
    ):
        raise ValueError("Score receipt policy/runtime identity differs")
    if (
        receipt["identity"]["sample_identity"] != canonical_hash(samples["identity"])
        or receipt["identity"]["sample_chunks"] != samples["chunks"]
    ):
        raise ValueError("Score receipt references another draw packet")
    output = {}
    for row in _verified_rows(receipt):
        output.setdefault(row["prompt_id"], []).append(row)
    return output


def build_response_packet(samples, scores, policies, *, contrasts=CONTRASTS, proposal="origin"):
    """Join exact raw tokens/input/RNG/runtime; keep three joint contrast axes."""
    if (
        not samples
        or len({r["prompt_id"] for r in samples}) != 1
        or len({r["rng_namespace"] for r in samples}) != 1
    ):
        raise ValueError("One actual prompt and RNG packet required")
    sample_ids = [r["sample_id"] for r in samples]
    if len(set(sample_ids)) != len(samples):
        raise ValueError("Duplicate raw draw identity")
    logs = {}
    for name, policy in policies.items():
        lookup = {r["sample_id"]: r for r in scores[name]}
        if len(lookup) != len(scores[name]) or set(lookup) != set(sample_ids):
            raise ValueError("Scored sample identity coverage differs")
        values = []
        for raw in samples:
            row = lookup[raw["sample_id"]]
            if (
                any(
                    row.get(k) != raw[k]
                    for k in (
                        "prompt_id",
                        "input_hash",
                        "role",
                        "rng_namespace",
                        "runtime_identity",
                    )
                )
                or row["inference_fingerprint"] != policy["inference_fingerprint"]
            ):
                raise ValueError("Score/raw/policy identity mismatch")
            if (
                ("token_ids" in row and row["token_ids"] != raw["token_ids"])
                or (
                    "token_ids" not in row
                    and row.get("shared_token_identity") != canonical_hash(raw["token_ids"])
                )
                or (
                    "shared_token_identity" in row
                    and row["shared_token_identity"] != canonical_hash(raw["token_ids"])
                )
            ):
                raise ValueError("Scored shared token identity differs")
            logps = np.asarray(row["token_logprobs"], dtype=float)
            if (
                logps.shape != (len(raw["token_ids"]),)
                or not np.isfinite(logps).all()
                or np.any(logps > 0)
                or not math.isclose(
                    math.fsum(logps), row["sequence_logp"], rel_tol=1e-12, abs_tol=1e-10
                )
            ):
                raise ValueError("Normalized full-sequence score mismatch")
            values.append(row["sequence_logp"])
        logs[name] = np.asarray(values)
    origin = np.asarray([r["generation_sequence_logp"] for r in samples])
    labels = np.eye(4)[[EVENTS.index(r["category"]) for r in samples]]
    contributions, pairs = [], []
    for _, candidate, baseline in contrasts:
        pair = (
            policies[baseline]["inference_fingerprint"],
            policies[candidate]["inference_fingerprint"],
        )
        if pair[0] == pair[1] and not np.array_equal(logs[candidate], logs[baseline]):
            raise ValueError("Exact inference alias has inconsistent sequence scores")
        if proposal == "origin":
            weights = stable_origin_weight(logs[candidate], logs[baseline], origin)
        elif proposal == "equal_pair_mixture":
            weights = stable_pair_weight(logs[candidate], logs[baseline])
        else:
            raise ValueError("Unknown actual proposal")
        contributions.append(weights[:, None] * labels)
        pairs.append(pair)
    return ContributionBatch(
        np.stack(contributions, axis=1),
        tuple(sample_ids),
        samples[0]["rng_namespace"],
        samples[0]["prompt_id"],
        tuple(v[0] for v in contrasts),
        tuple(pairs),
        tuple(tuple(r["token_ids"]) for r in samples),
        proposal,
        "VERIFIED",
    )


_packet = build_response_packet


def _reference_statistics(batch, *, overlap_usable=True):
    if batch.n < 2:
        raise ValueError("Reference uncertainty needs independent repeated draws")
    values = batch.contributions
    se = np.std(values, axis=0, ddof=1) / np.sqrt(batch.n)
    aliases = np.array([a == b for a, b in batch.policy_pairs])
    # Remove roundoff of exactly constant arrays, not small genuine variation.
    se[np.ptp(values, axis=0) == 0] = 0
    precision = reference_precision(
        se, exact_alias=aliases[:, None], overlap_usable=np.asarray(overlap_usable)[..., None]
    )
    resolved = precision["reference_resolved"]
    return {
        "estimate": values.mean(0),
        "se": se,
        "resolved": resolved,
        "covariance_of_mean": np.cov(values.reshape(batch.n, -1), rowvar=False) / batch.n,
        "uncertainty": "IID_RAW_CONTRIBUTIONS_EMPIRICAL_SE",
        "coverage_certified": False,
        "precision": precision,
    }


def _effective_contrast(candidate, baseline, scalings, base_id):
    if set(candidate) != set(baseline):
        raise ValueError("Complete endpoint parameter layouts differ")
    modules, used = {}, set()
    for name, scale in scalings.items():
        a, b = f"{name}.lora_A.default.weight", f"{name}.lora_B.default.weight"
        if a not in candidate or b not in candidate:
            raise ValueError("Actual LoRA module scaling lacks both complete factors")
        used.update((a, b))
        modules[name] = {
            "scaling": scale,
            "candidate": {"A": np.asarray(candidate[a]), "B": np.asarray(candidate[b])},
            "baseline": {"A": np.asarray(baseline[a]), "B": np.asarray(baseline[b])},
        }
    if used != set(candidate):
        raise ValueError("Effective coordinates must cover all actual trainable parameters")
    return {"base_model_id": base_id, "modules": modules}


def _contract_derivative(gradient, delta):
    if (
        gradient["expansion_point"] != "ORIGIN"
        or gradient["reference_labels_read"] is not False
        or gradient["parameter_order"] != sorted(delta)
        or set(gradient["grouped_gradients"]) != set(delta)
        or set(gradient["prompt_gradients"]) != set(delta)
    ):
        raise ValueError(
            "Origin derivative must cover every trainable parameter without references"
        )
    output = {}
    for kind, count in [
        ("group", len(gradient["groups"])),
        ("prompt", len(gradient["prompt_subset"])),
    ]:
        arrays = gradient["grouped_gradients" if kind == "group" else "prompt_gradients"]
        value = np.zeros((count, 4))
        for name, direction in delta.items():
            array = np.asarray(arrays[name])
            if array.shape != (count, 4, *direction.shape) or not np.isfinite(array).all():
                raise ValueError("Full derivative coordinate shape/finiteness differs")
            value += np.sum(array * direction, axis=tuple(range(2, array.ndim)))
        output[kind] = value
    return output


def _observations(bank, response, samples, raw, probes, methods, *, draw_count=None):
    policies = bank["policies"]
    score_table = response[
        {"work": "work_scores", "reference": "reference_scores", "direct_work": "direct_scores"}[
            samples["identity"]["role"]
        ]
    ]
    scored = {
        name: _score_rows(
            score_table[p["candidate_id"]]["score_receipt"],
            samples,
            p,
            samples["identity"]["runtime"],
        )
        for name, p in policies.items()
    }
    prefix = response.get("_verified_prefix_sources", {}).get(samples["identity"]["role"])
    if prefix:
        old_samples, old_scores = prefix
        for name, policy in policies.items():
            prior = _score_rows(
                old_scores[policy["candidate_id"]]["score_receipt"],
                old_samples,
                policy,
                old_samples["identity"]["runtime"],
            )
            for pid, rows in prior.items():
                current = {r["sample_id"]: r for r in scored[name][pid]}
                if any(current.get(r["sample_id"]) != r for r in rows):
                    raise ValueError("Nested scores changed an already measured prefix")
    result = {method: [] for method in methods}
    references = []
    for p in probes:
        pid = p["prompt_id"]
        actual = raw[pid]
        selected_scores = {name: rows[pid] for name, rows in scored.items()}
        if draw_count is not None:
            if type(draw_count) is not int or not 2 <= draw_count <= len(actual):
                raise ValueError("Requested observation prefix is not actually available")
            ids = {r["sample_id"] for r in actual}
            for rows in selected_scores.values():
                if len(rows) != len(ids) or {r["sample_id"] for r in rows} != ids:
                    raise ValueError("Full scored packet coverage differs before prefix selection")
            actual = actual[:draw_count]
            ids = {r["sample_id"] for r in actual}
            selected_scores = {
                name: [r for r in rows if r["sample_id"] in ids]
                for name, rows in selected_scores.items()
            }
        if methods:
            packet = _packet(actual, selected_scores, policies)
            observation = observe_packet(packet, methods=methods, bootstrap_repetitions=None)
            for method in methods:
                result[method].append(observation["methods"][method]["estimate"])
        else:
            logs = {}
            ids = [r["sample_id"] for r in actual]
            for name in policies:
                lookup = {r["sample_id"]: r["sequence_logp"] for r in selected_scores[name]}
                logs[name] = np.array([lookup[k] for k in ids])
            logq = np.array([r["generation_sequence_logp"] for r in actual])
            overlap = {name: endpoint_overlap(value - logq) for name, value in logs.items()}
            usable = np.array(
                [overlap[a]["all_usable"] and overlap[b]["all_usable"] for _, a, b in CONTRASTS]
            )
            stats = _origin_reference_statistics(actual, selected_scores, policies, usable)
            references.append(
                {
                    **stats,
                    "overlap": overlap,
                    "overlap_usable": usable,
                    "selected_proposal": ["ORIGIN"] * 3,
                }
            )
    return {name: np.asarray(values) for name, values in result.items()} if methods else references


def _origin_reference_statistics(samples, scores, policies, usable):
    """Retain nonrepresentable ORIGIN estimates as missing until bounded MIX joins.

    Every contrast still passes the raw/score identity validation. Only arithmetic
    overflow is converted into an unresolved reference, never identity failures.
    """
    try:
        packet = build_response_packet(samples, scores, policies)
    except FloatingPointError as exc:
        if "NONFINITE_WEIGHT" not in str(exc):
            raise
    else:
        return {
            **_reference_statistics(packet, overlap_usable=usable),
            "origin_nonrepresentable_weight": [False] * len(CONTRASTS),
        }
    packets, overflow = [], []
    for contrast in CONTRASTS:
        try:
            packets.append(build_response_packet(samples, scores, policies, contrasts=[contrast]))
            overflow.append(False)
        except FloatingPointError as exc:
            if "NONFINITE_WEIGHT" not in str(exc):
                raise
            packets.append(None)
            overflow.append(True)
    if any(overflow):
        values = np.full((len(samples), 3, 4), np.nan)
        for i, packet in enumerate(packets):
            if packet is not None:
                values[:, i] = packet.contributions[:, 0]
        se = np.std(values, axis=0, ddof=1) / math.sqrt(len(samples))
        se[np.ptp(values, axis=0) == 0] = 0
        aliases = np.array(
            [
                policies[a]["inference_fingerprint"] == policies[b]["inference_fingerprint"]
                for _, a, b in CONTRASTS
            ]
        )
        precision = reference_precision(
            se, exact_alias=aliases[:, None], overlap_usable=np.asarray(usable)[:, None]
        )
        result = {
            "estimate": values.mean(0),
            "se": se,
            "resolved": precision["reference_resolved"],
            "precision": precision,
            "covariance_of_mean": np.cov(values.reshape(len(samples), -1), rowvar=False)
            / len(samples),
            "uncertainty": "IID_RAW_CONTRIBUTIONS_EMPIRICAL_SE",
            "coverage_certified": False,
        }
    else:
        raise RuntimeError("Nonfinite reference could not be traced to an actual contrast")
    result["origin_nonrepresentable_weight"] = overflow
    return result


def _apply_mixture_references(
    bank,
    response,
    probes,
    references,
    *,
    origin_id,
    forbidden_ids,
    forbidden_streams,
    forbidden_seeds,
):
    """Use only the frozen overlap rule to choose separately collected pair MIX.

    This runs after predictions. Only prompts failing frozen overlap switch;
    missing required fallback remains unresolved, never a poor-model verdict.
    """
    original = {
        k: np.array([r[k] for r in references])
        for k in ("estimate", "se", "resolved", "covariance_of_mean")
    }
    diagnostics = []
    for ti, contrast in enumerate(CONTRASTS):
        target, candidate, baseline = contrast
        needed = {
            p["prompt_id"]
            for p, r in zip(probes, references, strict=True)
            if not bool(r["overlap_usable"][ti])
            or r.get("origin_nonrepresentable_weight", [False] * 3)[ti]
        }
        usable = not needed
        diag = {
            "bank_id": bank["bank_id"],
            "contrast_id": target,
            "origin_overlap_usable_all_prompts": usable,
            "selected_proposal": "ORIGIN",
            "selection_uses_prediction_errors": False,
            "required_prompt_ids": sorted(needed),
        }
        if not usable:
            matches = [
                c
                for c in response.get("pair_checks", [])
                if c["bank_id"] == bank["bank_id"] and c.get("contrast_id") == target
            ]
            if len(matches) > 1:
                raise ValueError("Duplicated reference fallback for one contrast")
            if not matches:
                for p, r in zip(probes, references, strict=True):
                    if p["prompt_id"] not in needed:
                        continue
                    if bank["contrasts"][target]["exact_inference_alias"] is not True:
                        r["resolved"][ti] = False
                        for mask in r["precision"]["resolved_at_scale"].values():
                            mask[ti] = False
                diag.update(
                    selected_proposal="ORIGIN_REQUIRED_MIX_MISSING", reference_status="UNRESOLVED"
                )
            else:
                check = matches[0]
                policies = {
                    candidate: bank["policies"][candidate],
                    baseline: bank["policies"][baseline],
                }
                if (
                    check.get("left_candidate_id") != policies[candidate]["candidate_id"]
                    or check.get("right_candidate_id") != policies[baseline]["candidate_id"]
                ):
                    raise ValueError("Fallback mixture endpoints differ from actual contrast")
                samples = check["mixture"]
                mixture_ids = samples["identity"].get("prompt_ids")
                if mixture_ids is None:
                    # Compatibility for a full-panel packet explicitly hash-bound as such.
                    if samples["identity"]["prompt_hash"] != canonical_hash(probes):
                        raise ValueError("Fallback subset requires explicit ordered prompt IDs")
                    mixture_ids = [p["prompt_id"] for p in probes]
                mixture_probes = [p for p in probes if p["prompt_id"] in mixture_ids]
                if (
                    [p["prompt_id"] for p in mixture_probes] != mixture_ids
                    or not needed <= set(mixture_ids)
                    or samples["identity"]["draws"] != response["reference"]["identity"]["draws"]
                    or samples["identity"]["runtime"]
                    != response["reference"]["identity"]["runtime"]
                ):
                    raise ValueError("Fallback prompt coverage, draw budget or runtime differs")
                raw = _action_rows(
                    samples,
                    mixture_probes,
                    role="reference",
                    origin_id=origin_id,
                    policy_fingerprint=(
                        policies[candidate]["inference_fingerprint"],
                        policies[baseline]["inference_fingerprint"],
                    ),
                )
                ids = {r["sample_id"] for rows in raw.values() for r in rows}
                stream = samples["identity"]["rng_namespace"]
                seeds = {r["sample_seed"] for rows in raw.values() for r in rows}
                if ids & forbidden_ids or stream in forbidden_streams or seeds & forbidden_seeds:
                    raise ValueError(
                        "Fallback reference must retain actually independent draws and RNG"
                    )
                forbidden_ids.update(ids)
                forbidden_streams.add(stream)
                forbidden_seeds.update(seeds)
                scores = {
                    n: _score_rows(
                        check["mixture_scores"][p["candidate_id"]],
                        samples,
                        p,
                        samples["identity"]["runtime"],
                    )
                    for n, p in policies.items()
                }
                for j, prompt in enumerate(probes):
                    pid = prompt["prompt_id"]
                    if pid not in needed:
                        continue
                    packet = build_response_packet(
                        raw[pid],
                        {n: r[pid] for n, r in scores.items()},
                        policies,
                        contrasts=[contrast],
                        proposal="equal_pair_mixture",
                    )
                    lookups = {
                        n: {r["sample_id"]: r["sequence_logp"] for r in values[pid]}
                        for n, values in scores.items()
                    }
                    logs = {
                        n: np.array([lookup[k] for k in packet.sample_ids])
                        for n, lookup in lookups.items()
                    }
                    logq = np.logaddexp(logs[candidate], logs[baseline]) - math.log(2)
                    overlap = {n: endpoint_overlap(v - logq) for n, v in logs.items()}
                    usable_mix = all(o["all_usable"] for o in overlap.values())
                    stats = _reference_statistics(packet, overlap_usable=usable_mix)
                    old = references[j]
                    for key in ("estimate", "se", "resolved"):
                        old[key][ti] = stats[key][0]
                    old["precision"]["normal_approximation_half_width"][ti] = stats["precision"][
                        "normal_approximation_half_width"
                    ][0]
                    for scale, mask in old["precision"]["resolved_at_scale"].items():
                        mask[ti] = stats["precision"]["resolved_at_scale"][scale][0]
                    # This draw packet is independent of other selected reference packets.
                    part = slice(4 * ti, 4 * (ti + 1))
                    old["covariance_of_mean"][part, :] = 0
                    old["covariance_of_mean"][:, part] = 0
                    old["covariance_of_mean"][part, part] = stats["covariance_of_mean"]
                    old["selected_proposal"][ti] = "MIX"
                diag.update(
                    selected_proposal="MIX" if len(needed) == len(probes) else "ORIGIN_AND_MIX",
                    reference_status="EMPIRICAL_PRECISION_LABELS_ONLY",
                    mixture_sample_identity=canonical_hash(samples["identity"]),
                )
        diagnostics.append(diag)
    return {"original_origin_statistics": original, "fallback_diagnostics": diagnostics}


def _load_delta(bank, target, cache):
    from ..optimizer_fork import state_hash

    contrast = bank["contrasts"][target]
    left, right = [cache.load(contrast[k])["parameters"] for k in ("left", "right")]
    if sorted(left) != contrast["parameter_order"] or set(left) != set(right):
        raise ValueError("Complete raw parameter order differs")
    delta = {n: left[n].double() - right[n].double() for n in sorted(left)}
    if state_hash(delta) != contrast["delta_hash"]:
        raise ValueError("Actual endpoint delta differs from fork receipt")
    return {n: v.numpy() for n, v in delta.items()}


def _row(
    task,
    bank,
    target,
    method,
    prediction,
    probes,
    *,
    design,
    diagnostics=None,
    model_metadata=None,
    unit="prompt",
):
    alias = bank["contrasts"][target]["exact_inference_alias"]
    prediction = np.asarray(prediction)
    known = bool(np.isfinite(prediction).all())
    return {
        "seed": task["seed"],
        "arm": task["arm"],
        "step": task["step"],
        "origin_id": task["id"],
        "role": task.get("role", "development"),
        "canonical_policy_origin_id": task.get("canonical_origin_id", task["id"]),
        "bank_id": bank["bank_id"],
        "target": target,
        "method": method,
        "design_id": design,
        "n": task["draws"],
        "information_level": (
            "NO_QUERY_MEASUREMENT"
            if method == "ZERO"
            else "ANCHOR_SCORE_GRADIENT_AND_QUERY_PARAMETERS"
            if method == "FULL_SCORE_JVP"
            else "PARAMETER_AND_CALIBRATION_RESPONSE"
        ),
        "evaluation_unit": unit,
        "prompt_ids": [p["prompt_id"] for p in probes],
        "groups": [p["family"] + "|" + p["interface"] for p in probes],
        "is_alias": alias,
        "prediction_status": "PREDICTED" if known else "UNKNOWN",
        "prediction": prediction.tolist() if known else None,
        "diagnostics": diagnostics or {},
        "model_metadata": model_metadata or {},
        "query_response_used_for_fit": False,
    }


def _read_gradient(receipt, origin_id, work):
    import torch

    if (
        receipt["status"] != "MEASURED"
        or receipt["origin_id"] != origin_id
        or receipt["reference_used"] is not False
    ):
        raise ValueError("Missing actual independent origin derivative receipt")
    artifact = receipt["artifact"]
    if sha256_file(artifact["path"]) != artifact["sha256"]:
        raise ValueError("Derivative bytes differ")
    # The producer stores CPU tensors/builtin containers, permitting safe loading.
    value = torch.load(artifact["path"], map_location="cpu", weights_only=True)
    actual = {r["sample_id"] for rows in work.values() for r in rows}
    if not set(value["sample_ids"]) <= actual or len(set(value["sample_ids"])) != len(
        value["sample_ids"]
    ):
        raise ValueError("Derivative anchors are not the actual independent work draws")
    if set(value["rng_stream_ids"]) != {r["rng_namespace"] for rows in work.values() for r in rows}:
        raise ValueError("Derivative RNG origin differs")
    return value


def fit_response_map(
    tasks,
    *,
    task_id,
    out,
    alpha=1e-5,
    bandwidth_multiplier=1.0,
    observation_methods=None,
    calibration_banks=None,
    bank_selector=None,
    draw_count=None,
):
    """Fit one completed map on server CPU, then independent evaluation.

    Alpha/bandwidth are explicitly recorded grid members, never selected by query
    responses. Re-run other grid settings to separate immutable output directories.
    """
    _require_server_cpu()
    tasks_path = Path(tasks)
    plan = json.loads(tasks_path.read_text())
    if (
        canonical_hash({k: v for k, v in plan.items() if k != "task_list_hash"})
        != plan["task_list_hash"]
        or plan["source"] != source_identity()
    ):
        raise ValueError("Frozen task/source identity differs")
    selected = [t for t in plan["tasks"] if t["id"] == task_id]
    if len(selected) != 1 or selected[0]["kind"] != "map":
        raise ValueError("This entrypoint requires a registered actual response map")
    task = selected[0]
    config = plan["config"]
    role = task.get("role", "development")
    if (
        role not in config["qwen"]["seed_roles"]
        or task["seed"] not in config["qwen"]["seed_roles"][role]
    ):
        raise ValueError("Actual source seed role differs")
    frozen = None
    if task.get("stage") == "D" or task.get("phase") == "D" or role != "development":
        from .gpu_collect import _validate_phase_task

        extension = _validate_phase_task(plan, task, {})
        frozen = extension.get("selection")
        if role != "development" and (
            not frozen
            or alpha != 1e-5
            or bandwidth_multiplier != 1.0
            or observation_methods is not None
            or calibration_banks is not None
            or bank_selector is not None
            or draw_count is not None
        ):
            raise ValueError(
                "Confirmation uses only frozen model settings and mandatory fixed baselines"
            )
    if frozen:
        calibration_banks = frozen["m"]
        bank_selector = frozen["bank_selector"]
        draw_count = frozen["draw_count"]
    else:
        calibration_banks = (
            task["calibration_banks"] if calibration_banks is None else calibration_banks
        )
        draw_count = task["draws"] if draw_count is None else draw_count
        bank_selector = "STRATIFIED_RANDOM" if bank_selector is None else bank_selector
        if calibration_banks not in task.get(
            "nested_bank_prefixes", [task["calibration_banks"]]
        ) or draw_count not in task.get("nested_draw_counts", [task["draws"]]):
            raise ValueError("Development m/n must be registered nested curve points")
    if (
        type(calibration_banks) is not int
        or not 0 < calibration_banks <= task["calibration_banks"]
        or type(draw_count) is not int
        or draw_count < 2
        or bank_selector not in {"STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"}
    ):
        raise ValueError("Registered whole-bank selector, available m and positive n required")
    generation = config["qwen"]["generation"]
    if any(
        generation.get(k) != v
        for k, v in {
            "max_new_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "thinking": False,
        }.items()
    ):
        raise ValueError("Primary full-support normalized action distribution differs")
    if (
        alpha not in config["models"]["ridge_alpha"]
        or bandwidth_multiplier not in config["models"]["rbf_bandwidth_multipliers"]
    ):
        raise ValueError("Model penalty/bandwidth must be in the registered grid")
    methods = tuple(observation_methods or config["observations"]["methods"])
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(PRIMARY_METHODS):
        raise ValueError("Use registered primary observation methods")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    task_file = Path(plan["root"]) / "tasks" / task_id / "COMPLETE.json"
    completed = json.loads(task_file.read_text())
    if (
        completed.get("status") != "COMPLETED"
        or completed.get("task") != task
        or completed.get("execution_kind") != "REAL_CUDA_MODEL"
        or completed.get("config_hash") != canonical_hash(config)
        or completed.get("source_hash") != plan["source"]["sha256"]
    ):
        raise ValueError("Actual completed CUDA response-map receipt required")
    forks, response = bound_json(completed["forks"]), bound_json(completed["response"])
    origin = completed.get("origin_id", task_id)
    if origin != task_id and task.get("reuse_prefix_task") != origin:
        raise ValueError("Canonical policy origin differs from registered prefix reuse")
    if forks["origin_id"] != origin or response["origin_id"] != origin:
        raise ValueError("Fork/response origin differs")
    task = {**task, "canonical_origin_id": origin}
    probes = plan["inputs"]["panels"]["observation"][: task["prompts"]]
    config_binding = {
        "tasks": _binding(tasks_path),
        "task": _binding(task_file),
        "forks": completed["forks"],
        "response": completed["response"],
        "config_hash": canonical_hash(config),
        "source": plan["source"],
        "execution_kind": completed["execution_kind"],
        "task_id": task_id,
        "canonical_origin_id": origin,
        "selection_hash": frozen["selection_hash"] if frozen else None,
        "alpha": alpha,
        "bandwidth_multiplier": bandwidth_multiplier,
        "observation_methods": list(methods),
        "calibration_banks": calibration_banks,
        "bank_selector": bank_selector,
        "draw_count": draw_count,
        "analysis_rules": _binding(
            Path(__file__).parents[2] / "configs/modeling_v4/analysis_rules.json"
        ),
    }
    return _fit_completed_map(
        config,
        task,
        forks,
        response,
        completed["gradient"],
        probes,
        root,
        binding=config_binding,
        alpha=alpha,
        bandwidth_multiplier=bandwidth_multiplier,
        methods=methods,
        selection=frozen,
        completed=completed,
        calibration_banks=calibration_banks,
        bank_selector=bank_selector,
        draw_count=draw_count,
        training_prompts=plan["inputs"]["train_prompts"],
    )


def _endpoint_memmap(forks, banks, path, cache, runtime):
    """One checkpoint at a time into disposable disk-backed complete coordinates."""
    from ..optimizer_fork import state_hash
    from .gpu_collect import _fingerprint

    order = ("joint_0", "joint_1", "no_x_off_1")
    origin = cache.load(forks["origin_policy"])
    if _fingerprint(origin, runtime) != forks["origin_policy"]["inference_fingerprint"]:
        raise ValueError("Actual origin forward fingerprint differs")
    first = cache.load(banks[0]["policies"][order[0]])["parameters"]
    layout, offset = {}, 0
    for name, tensor in sorted(first.items()):
        size = tensor.numel()
        layout[name] = {"start": offset, "stop": offset + size, "shape": tuple(tensor.shape)}
        offset += size
    if not offset:
        raise ValueError("Complete nonempty trainable parameter layout required")
    if any(
        str(v.dtype) not in {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
        for v in first.values()
    ):
        raise ValueError("Only actual real floating point trainable parameters are supported")
    dtype = (
        np.float64 if any(str(v.dtype) == "torch.float64" for v in first.values()) else np.float32
    )
    required_bytes = len(banks) * 3 * offset * np.dtype(dtype).itemsize
    require_space(Path(path).parent, required_bytes + 64 * 1024**2)
    data = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(banks), 3, offset))
    scalings = forks["origin_policy"].get("module_scalings")
    if not scalings or any(not np.isfinite(v) or v <= 0 for v in scalings.values()):
        raise ValueError("Actual positive LoRA module scalings required")
    for i, bank in enumerate(banks):
        states = {}
        for j, name in enumerate(order):
            policy = bank["policies"][name]
            state = cache.load(policy)
            if (
                _fingerprint(state, runtime) != policy["inference_fingerprint"]
                or policy.get("module_scalings") != scalings
            ):
                raise ValueError("Actual candidate forward fingerprint or module scaling differs")
            parameters = state["parameters"]
            states[name] = parameters
            if sorted(parameters) != list(layout):
                raise ValueError("All candidate parameters must retain the complete origin layout")
            declared = policy.get("parameter_layout")
            if declared is None or [(r["name"], tuple(r["shape"])) for r in declared] != [
                (n, tuple(v.shape)) for n, v in sorted(parameters.items())
            ]:
                raise ValueError(
                    "Actual complete checkpoint parameter layout lacks matching receipt"
                )
            for key, bounds in layout.items():
                value = parameters[key].double().numpy()
                if data.dtype == np.float32 and str(parameters[key].dtype) == "torch.float64":
                    raise ValueError(
                        "Endpoint dtype differs; float64 input requires float64 scratch"
                    )
                if value.shape != bounds["shape"] or not np.isfinite(value).all():
                    raise ValueError("Candidate parameter shape or finiteness differs")
                data[i, j, bounds["start"] : bounds["stop"]] = value.ravel()
        for target, candidate, baseline in CONTRASTS:
            spec = bank["contrasts"][target]
            if (
                spec["left"] != bank["policies"][candidate]["checkpoint"]
                or spec["right"] != bank["policies"][baseline]["checkpoint"]
                or spec["parameter_order"] != list(layout)
            ):
                raise ValueError("Contrast endpoints/order differ from actual fork policies")
            difference = {
                n: states[candidate][n].double() - states[baseline][n].double() for n in layout
            }
            if state_hash(difference) != spec["delta_hash"]:
                raise ValueError("Actual complete endpoint delta differs")
            alias = (
                bank["policies"][candidate]["inference_fingerprint"]
                == bank["policies"][baseline]["inference_fingerprint"]
            )
            if (
                type(spec["exact_inference_alias"]) is not bool
                or alias != spec["exact_inference_alias"]
            ):
                raise ValueError("Alias must use exact actual full forward fingerprint")
        data.flush()
    return data, layout, scalings


def compute_response_geometry(
    data,
    layout,
    scalings,
    gradient,
    *,
    candidate,
    baseline,
    calibration_count,
    base_id,
    endpoint_names=("joint_0", "joint_1", "no_x_off_1"),
    calibration_indices=None,
):
    """Accumulate exact raw/effective Gram and J products by parameter/module.

    The largest raw temporary is banks x 65536 coordinates. Low-rank adapter
    products use factor identities; no dense effective weight matrix is formed.
    """
    from .full_kernels import effective_weight_gram

    names = tuple(endpoint_names)
    if (
        len(data.shape) != 3
        or data.shape[1] != len(names)
        or len(set(names)) != len(names)
        or candidate not in names
        or baseline not in names
        or candidate == baseline
        or type(calibration_count) is not int
        or not 0 < calibration_count < len(data)
    ):
        raise ValueError("Explicit aligned endpoint axes and nonempty fit/query rows required")
    offset = 0
    for _name, item in sorted(layout.items()):
        if (
            item["start"] != offset
            or item["stop"] <= offset
            or item["stop"] - offset != int(np.prod(item["shape"]))
        ):
            raise ValueError("Sorted parameter layout must partition every coordinate")
        offset = item["stop"]
    if offset != data.shape[-1] or list(layout) != sorted(layout):
        raise ValueError("Complete sorted parameter coordinate layout required")
    ci, bi = names.index(candidate), names.index(baseline)
    fit_indices = (
        np.arange(calibration_count)
        if calibration_indices is None
        else np.asarray(calibration_indices, dtype=int)
    )
    if (
        fit_indices.ndim != 1
        or not len(fit_indices)
        or len(set(fit_indices.tolist())) != len(fit_indices)
        or np.any(fit_indices < 0)
        or np.any(fit_indices >= calibration_count)
    ):
        raise ValueError("Fit-only block scales require actual selected calibration rows")
    if (
        gradient["parameter_order"] != list(layout)
        or gradient["expansion_point"] != "ORIGIN"
        or gradient["reference_labels_read"] is not False
        or set(gradient["grouped_gradients"]) != set(layout)
        or set(gradient["prompt_gradients"]) != set(layout)
    ):
        raise ValueError("Origin derivative layout must retain every trainable coordinate")
    gram = np.zeros((len(data), len(data)))
    standardized = np.zeros_like(gram)
    block_scales = {}
    derivative = {
        kind: np.zeros((len(data) - calibration_count, count, 4))
        for kind, count in [
            ("group", len(gradient["groups"])),
            ("prompt", len(gradient["prompt_subset"])),
        ]
    }
    for name, bounds in layout.items():
        block_gram = np.zeros_like(gram)
        for start in range(bounds["start"], bounds["stop"], 65536):
            stop = min(bounds["stop"], start + 65536)
            delta = np.asarray(data[:, ci, start:stop], dtype=np.float64) - np.asarray(
                data[:, bi, start:stop], dtype=np.float64
            )
            if not np.isfinite(delta).all():
                raise ValueError("All actual parameter differences must be finite")
            block_gram += delta @ delta.T
            for kind, field in [("group", "grouped_gradients"), ("prompt", "prompt_gradients")]:
                array = np.asarray(gradient[field][name])
                count = derivative[kind].shape[1]
                if array.shape != (count, 4, *bounds["shape"]) or not np.isfinite(array).all():
                    raise ValueError("Actual derivative shape/finiteness differs")
                if count:
                    local = start - bounds["start"]
                    product = (
                        delta[calibration_count:]
                        @ array.reshape(count * 4, -1)[:, local : local + stop - start].T
                    )
                    derivative[kind] += product.reshape(len(product), count, 4)
        gram += block_gram
        scale = float(np.sqrt(np.mean(np.maximum(np.diag(block_gram)[fit_indices], 0))))
        block_scales[name] = scale if scale > 0 else 1.0
        standardized += block_gram / block_scales[name] ** 2
    effective = np.zeros_like(gram)
    used = set()
    for module, scale in scalings.items():
        a, b = f"{module}.lora_A.default.weight", f"{module}.lora_B.default.weight"
        if a not in layout or b not in layout:
            raise ValueError("Effective factor coordinates do not match measured module scaling")
        used.update((a, b))

        def factor(i, endpoint, name):
            item = layout[name]
            return np.asarray(data[i, endpoint, item["start"] : item["stop"]]).reshape(
                item["shape"]
            )

        contrasts = [
            {
                "base_model_id": base_id,
                "modules": {
                    module: {
                        "scaling": scale,
                        "candidate": {"A": factor(i, ci, a), "B": factor(i, ci, b)},
                        "baseline": {"A": factor(i, bi, a), "B": factor(i, bi, b)},
                    }
                },
            }
            for i in range(len(data))
        ]
        effective += effective_weight_gram(contrasts)
    if used != set(layout):
        raise ValueError("Effective representation must cover all trainable coordinates")
    return {
        "raw": gram,
        "block_standardized": standardized,
        "block_scales": block_scales,
        "block_scale_definition": "fit-only RMS L2 parameter-block norm; zero scale becomes one",
        "effective": effective,
        "derivative": derivative,
        "input_dimension": data.shape[-1],
        "raw_parameter_dimension": data.shape[-1],
        "effective_weight_dimension": sum(
            layout[f"{m}.lora_B.default.weight"]["shape"][0]
            * layout[f"{m}.lora_A.default.weight"]["shape"][1]
            for m in scalings
        ),
        "dense_effective_weights_materialized": False,
        "raw_chunk_max_coordinates": 65536,
    }


# Private compatibility name for existing mathematical fixtures.
_precomputed_inputs = compute_response_geometry


def _model_specs(config, methods, alpha, bandwidth, selection=None):
    """All development diagnostics, or only frozen primaries and fixed baselines."""
    result = []

    def add(
        method,
        rank,
        representation,
        observation,
        *,
        model_id=None,
        penalty=alpha,
        bw=bandwidth,
        label=None,
    ):
        if (
            observation not in methods
            or penalty not in config["models"]["ridge_alpha"]
            or bw not in config["models"]["rbf_bandwidth_multipliers"]
        ):
            raise ValueError("Model settings are outside the frozen observation/penalty grid")
        if rank != "FULL" and rank not in config["models"]["compression_ranks"]:
            raise ValueError("Model rank is outside the frozen comparison list")
        result.append(
            {
                "method": method,
                "rank": rank,
                "representation": representation,
                "observation": observation,
                "alpha": penalty,
                "bandwidth_multiplier": bw,
                "frozen_model_id": model_id,
                "label": label or (method if rank == "FULL" else "LOW_RANK_COMPRESSION"),
            }
        )

    for observation in methods:
        add(
            "FULL_DUAL_RIDGE",
            "FULL",
            "raw",
            observation,
            penalty=1e-5 if selection else alpha,
            bw=1.0 if selection else bandwidth,
        )
        for rank in config["models"]["compression_ranks"]:
            if rank != "FULL":
                add(
                    "FULL_DUAL_RIDGE",
                    rank,
                    "raw",
                    observation,
                    penalty=1e-5 if selection else alpha,
                    bw=1.0 if selection else bandwidth,
                )
        if selection is None:
            for method, representation in [
                ("FULL_RBF_RAW", "raw"),
                ("FULL_DUAL_RIDGE", "block_standardized"),
                ("FULL_RBF_RAW", "block_standardized"),
                ("LEGACY_QPROJECTED_RBF", "raw"),
                ("FULL_EFFECTIVE_WEIGHT_RIDGE", "effective"),
                ("FULL_RBF_EFFECTIVE_WEIGHT", "effective"),
            ]:
                add(method, "FULL", representation, observation)
    if selection:
        for primary in selection["primary_models"]:
            if primary["representation"] == "R4" or primary["model"] in (
                "FULL_SCORE_JVP",
                "BASE_SCORE_JVP",
                "MIDPOINT_DIRECTIONAL",
            ):
                continue
            settings, method = primary["settings"], primary["model"]
            if method not in {
                "FULL_DUAL_RIDGE",
                "FULL_RBF_RAW",
                "LEGACY_QPROJECTED_RBF",
                "FULL_EFFECTIVE_WEIGHT_RIDGE",
                "FULL_RBF_EFFECTIVE_WEIGHT",
                "LOW_RANK_COMPRESSION",
            }:
                raise ValueError(
                    "Selected primary needs an actual registered predictor implementation"
                )
            if set(settings) - {
                "alpha",
                "bandwidth_multiplier",
                "observation_method",
                "standardize_blocks",
                "rank_cap",
            }:
                raise ValueError("Unknown frozen parameter-kernel setting")
            if type(settings.get("standardize_blocks", False)) is not bool:
                raise ValueError("Block standardization must be explicit boolean")
            representation = (
                "effective"
                if "EFFECTIVE" in method
                else "block_standardized"
                if settings.get("standardize_blocks")
                else "raw"
            )
            if primary["representation"] != ("R3" if representation == "effective" else "R2") or (
                representation == "effective" and settings.get("standardize_blocks", False)
            ):
                raise ValueError("Frozen representation or standardization has no matching kernel")
            add(
                "FULL_DUAL_RIDGE" if method == "LOW_RANK_COMPRESSION" else method,
                settings.get("rank_cap", "FULL"),
                representation,
                settings.get("observation_method", "RAW4"),
                model_id=primary["id"],
                penalty=settings.get("alpha", 1e-5),
                bw=settings.get("bandwidth_multiplier", 1.0),
                label=method,
            )
    return result


def _frozen_signature_rows(selection, completed, task, queries, probes, out):
    from .signature_fit import predict_signature_models

    rows, models, receipts = [], [], {}
    for primary in selection["primary_models"]:
        if primary["representation"] != "R4":
            continue
        settings = primary["settings"]
        if set(settings) != {"signature_model_selection", "signature_model_id"}:
            raise ValueError("Signature settings must point to one complete frozen model spec")
        lock = settings.get("signature_model_selection")
        wanted = settings.get("signature_model_id")
        if not lock or not wanted or "signature" not in completed:
            raise ValueError(
                "Selected signature predictor needs actual frozen model and signature bindings"
            )
        locked = bound_json(lock)
        specifications = [s for s in locked["models"] if s["id"] == wanted]
        if len(specifications) != 1 or specifications[0]["spec"]["model"] != primary["model"]:
            raise ValueError("Frozen primary model differs from actual signature model type")
        key = canonical_hash(lock)
        if key not in receipts:
            receipts[key] = predict_signature_models(
                lock, completed["signature"], out=Path(out) / key
            )
        prediction = receipts[key]
        if prediction["prompt_ids"] != [p["prompt_id"] for p in probes] or prediction[
            "origin_id"
        ] != task.get("canonical_origin_id", task["id"]):
            raise ValueError("Frozen signature prediction panel/origin differs")
        if wanted not in prediction["model_ids"]:
            raise ValueError("Selected signature model is absent from frozen prediction")
        artifact = prediction["arrays"]
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen signature prediction bytes changed")
        values = np.load(artifact["path"], allow_pickle=False)["predictions"][
            prediction["model_ids"].index(wanted)
        ]
        by_unit = {(u["bank_id"], u["contrast_id"]): i for i, u in enumerate(prediction["units"])}
        if len(by_unit) != len(prediction["units"]):
            raise ValueError("Duplicated native signature unit")
        for bank in queries:
            for target, _, _ in CONTRASTS:
                if (bank["bank_id"], target) not in by_unit:
                    raise ValueError("Selected signature predictor lacks heldout query")
                row = _row(
                    task,
                    bank,
                    target,
                    primary["model"],
                    values[by_unit[bank["bank_id"], target]],
                    probes,
                    design=primary["id"],
                    model_metadata={
                        "frozen_model_id": primary["id"],
                        "signature_model_id": wanted,
                        "prediction_artifact": artifact,
                        "input_representation": "NATIVE_SIGNATURE_576",
                        "model_refitted": False,
                        "actual_signature_spec": specifications[0],
                    },
                )
                row["information_level"] = "PAID_NATIVE_SEQUENCE_SIGNATURE_AND_ORIGIN_STATE"
                rows.append(row)
        models.append(
            {
                "method": primary["model"],
                "frozen_model_id": primary["id"],
                "model_id": wanted,
                "input_representation": "NATIVE_SIGNATURE_576",
                "predictions": artifact,
                "query_labels_read": False,
            }
        )
    return rows, models


def _directional_rows(task, queries, probes, groups, *, completed, requested, selection):
    """Actual BASE/MIDPOINT group diagnostics; absent query derivatives stay UNKNOWN."""
    rows, models = [], []
    receipt = None
    if completed and completed.get("development_directional"):
        receipt = bound_json(completed["development_directional"])
        if (
            receipt["origin_id"] != task.get("canonical_origin_id", task["id"])
            or receipt.get("stage_role") != "development"
        ):
            raise ValueError("Directional diagnostic origin/role differs")
    for method, expansion in [("BASE_SCORE_JVP", "BASE"), ("MIDPOINT_DIRECTIONAL", "MIDPOINT")]:
        if method not in requested:
            continue
        actual_count = 0
        ids = [
            p["id"] for p in (selection or {}).get("primary_models", []) if p["model"] == method
        ] or [method]
        for bank in queries:
            matches = [
                r for r in (receipt or {}).get("banks", []) if r["bank_id"] == bank["bank_id"]
            ]
            if len(matches) > 1:
                raise ValueError("Duplicated measured directional bank")
            for target, _, _ in CONTRASTS:
                value = np.full((len(groups), 4), np.nan)
                metadata = {
                    "status": "UNKNOWN_NO_ACTUAL_QUERY_DERIVATIVE",
                    "expansion_point": expansion,
                }
                if matches and target == "joint_1_minus_joint_0":
                    record = matches[0]
                    summary = record[expansion]
                    if (
                        record["candidate_policy"] != bank["policies"]["joint_1"]
                        or record["baseline_policy"] != bank["policies"]["joint_0"]
                    ):
                        raise ValueError("Directional endpoints differ from the actual query bank")
                    if (
                        summary["reference_labels_read"] is not False
                        or summary["finite_difference_is_derivative"] is not False
                        or summary["expansion_point"] != expansion
                        or summary["anchor_prompt_ids"] != [p["prompt_id"] for p in probes]
                        or summary["groups"] != [list(g) for g in groups]
                    ):
                        raise ValueError("Actual directional panel/anchor semantics differ")
                    artifact = summary["arrays"]
                    if (
                        sha256_file(artifact["path"]) != artifact["sha256"]
                        or sha256_file(summary["score_rows"]["path"])
                        != summary["score_rows"]["sha256"]
                    ):
                        raise ValueError("Actual AD numerical artifacts changed")
                    value = np.load(artifact["path"], allow_pickle=False)["group_derivative"]
                    if value.shape != (len(groups), 4) or not np.isfinite(value).all():
                        raise ValueError("Directional event response shape differs")
                    metadata.update(status="ACTUALLY_MEASURED", arrays=artifact)
                    actual_count += 1
                panel = [
                    {"prompt_id": "GROUP:" + "|".join(g), "family": g[0], "interface": g[1]}
                    for g in groups
                ]
                for model_id in ids:
                    rows.append(
                        _row(
                            task,
                            bank,
                            target,
                            method,
                            value,
                            panel,
                            design=model_id,
                            unit="semantic_group",
                            model_metadata={
                                **metadata,
                                "frozen_model_id": model_id if selection else None,
                            },
                        )
                    )
        if actual_count:
            models.append(
                {
                    "method": method,
                    "actual_query_contrasts": actual_count,
                    "frozen_model_ids": ids,
                    "query_labels_read": False,
                    "prediction_coverage": "ONLY_MEASURED_QUERY_DERIVATIVES",
                }
            )
    return rows, models


def _nested_work_packets(response, probes, origin_id, fingerprint, draw_count):
    """Validate physical packets, then expose one shared-prefix source per role."""
    updated, raw, provenance = dict(response), {}, {}
    updated["_verified_prefix_sources"] = {}
    for role, scores_key, extra, extra_scores in [
        ("work", "work_scores", "ncurve_work", "ncurve_scores"),
        ("direct_work", "direct_scores", "ncurve_direct_work", "ncurve_direct_scores"),
    ]:
        original = response[role]
        base = _action_rows(
            original, probes, role=role, origin_id=origin_id, policy_fingerprint=fingerprint
        )
        selected = response.get(extra, original)
        actual = (
            base
            if selected is original
            else _action_rows(
                selected, probes, role=role, origin_id=origin_id, policy_fingerprint=fingerprint
            )
        )
        if selected["identity"]["draws"] < draw_count:
            raise ValueError("Independent work/DIRECT packet lacks the requested matched budget")
        if selected is not original:
            if (
                selected["identity"]["rng_namespace"] != original["identity"]["rng_namespace"]
                or selected["identity"]["runtime"] != original["identity"]["runtime"]
                or any(actual[pid][: len(rows)] != rows for pid, rows in base.items())
            ):
                raise ValueError("Nested observations changed actual prefix draws or runtime")
            updated[role], updated[scores_key] = selected, response[extra_scores]
            updated["_verified_prefix_sources"][role] = (original, response[scores_key])
        raw[role] = actual
        provenance[role] = {
            "physical_draws": selected["identity"]["draws"],
            "used_prefix_draws": draw_count,
            "sample_identity": canonical_hash(selected["identity"]),
            "primary_sample_identity": canonical_hash(original["identity"]),
            "nested_prefix_not_independent_replicates": True,
        }
    if updated["work"]["identity"]["runtime"] != updated["direct_work"]["identity"]["runtime"]:
        raise ValueError("Independent DIRECT must retain the exact work policy runtime")
    return updated, raw, provenance


def _calibration_full_gram(data, count):
    """Whole-bank three-contrast Gram from every raw parameter, chunked on disk."""
    result = np.zeros((3 * count, 3 * count))
    for start in range(0, data.shape[-1], 65536):
        stop = min(start + 65536, data.shape[-1])
        base = np.asarray(data[:count, 0, start:stop], dtype=np.float64)
        joint = np.asarray(data[:count, 1, start:stop], dtype=np.float64) - base
        off = np.asarray(data[:count, 2, start:stop], dtype=np.float64) - base
        block = np.stack((joint, off, joint - off), axis=1).reshape(3 * count, -1)
        result += block @ block.T
    return result


def _bank_strata(banks, training_prompts):
    if training_prompts is None:
        return None
    registry = {p["prompt_id"]: (p["family"], p["interface"]) for p in training_prompts}
    if len(registry) != len(training_prompts):
        raise ValueError("Training metadata prompt identities are duplicated")
    return [sorted(registry[p] for p in b["prompt_ids"]) for b in banks]


def _reference_validation(
    response, task, queries, probes, reference_raw, primary, root, fingerprint
):
    scope = task.get("reference_validation")
    if not scope:
        return {"status": "NOT_REGISTERED_FOR_THIS_ORIGIN"}
    if response.get("reference_validation_scope") != scope:
        raise ValueError("Reference validation differs from the outcome-independent frozen subset")
    samples = response["reference_validation"]
    if (
        samples["identity"]["draws"] != scope["draws"]
        or samples["identity"]["rng_namespace"]
        != response["reference"]["identity"]["rng_namespace"]
        or samples["identity"]["runtime"] != response["reference"]["identity"]["runtime"]
    ):
        raise ValueError("Reference validation must extend the same actual independent stream")
    raw = _action_rows(
        samples,
        probes,
        role="reference",
        origin_id=task.get("canonical_origin_id", task["id"]),
        policy_fingerprint=fingerprint,
    )
    if any(raw[pid][: len(rows)] != rows for pid, rows in reference_raw.items()):
        raise ValueError("Reference validation changed the measured primary prefix")
    extended = {
        **response,
        "reference": samples,
        "reference_scores": response["reference_validation_scores"],
        "_verified_prefix_sources": {
            "reference": (response["reference"], response["reference_scores"])
        },
    }
    arrays = {}
    banks = [b for b in queries if b["bank_id"] in scope["bank_ids"]]
    if [b["bank_id"] for b in banks] != scope["bank_ids"]:
        raise ValueError("Actual reference validation bank identities differ")
    for bank in banks:
        values = _observations(bank, extended, samples, raw, probes, ())
        bid = bank["bank_id"]
        for key in ("estimate", "se", "resolved", "covariance_of_mean"):
            arrays[f"{bid}__primary_{key}"] = np.asarray([s[key] for s in primary[bid]])
            arrays[f"{bid}__validation_{key}"] = np.asarray([s[key] for s in values])
        arrays[f"{bid}__estimate_change"] = (
            arrays[f"{bid}__validation_estimate"] - arrays[f"{bid}__primary_estimate"]
        )
    path = root / "REFERENCE_VALIDATION.npz"
    np.savez_compressed(path, **arrays)
    result = {
        "status": "MEASURED_NESTED_REFERENCE_COMPARISON",
        "arrays": _binding(path),
        "scope": scope,
        "primary_draws": response["reference"]["identity"]["draws"],
        "validation_draws": samples["identity"]["draws"],
        "primary_sample_identity": canonical_hash(response["reference"]["identity"]),
        "validation_sample_identity": canonical_hash(samples["identity"]),
        "nested_prefix_not_independent_replicates": True,
        "proposal": "RAW_ORIGIN_BEFORE_MIX_FALLBACK",
        "difference_standard_error": None,
        "difference_uncertainty": "NOT_COMPUTED; SHARED_PREFIX_COVARIANCE_REQUIRED",
        "selection_uses_prediction_errors": False,
        "coverage_guarantee": False,
    }
    atomic_json(root / "REFERENCE_VALIDATION.json", result)
    return result


def _fit_completed_map(
    config,
    task,
    forks,
    response,
    gradient_receipt,
    probes,
    root,
    *,
    binding,
    alpha,
    bandwidth_multiplier,
    methods,
    selection=None,
    completed=None,
    calibration_banks=None,
    bank_selector=None,
    draw_count=None,
    training_prompts=None,
):
    from .gpu_collect import CheckpointCache, _write_chunk

    root = Path(root)
    banks = forks["banks"]
    calibration = [b for b in banks if b["role"] == "calibration"]
    queries = [b for b in banks if b["role"] == "query"]
    if (
        len(calibration) != task["calibration_banks"]
        or len(queries) != task["query_banks"]
        or len({b["bank_id"] for b in banks}) != len(banks)
    ):
        raise ValueError("Registered calibration/query bank coverage differs")
    if {p for b in calibration for p in b["prompt_ids"]} & {
        p for b in queries for p in b["prompt_ids"]
    }:
        raise ValueError("Training prompt calibration/query partition overlaps")
    if response["work"]["identity"]["draws"] != task["draws"]:
        raise ValueError("Work draw count differs from frozen task")
    if response["direct_work"]["identity"]["draws"] != task["draws"]:
        raise ValueError("Independent DIRECT packet budget differs")
    draw_count = task["draws"] if draw_count is None else draw_count
    calibration_banks = len(calibration) if calibration_banks is None else calibration_banks
    physical_calibration_count = len(calibration)
    response, packets, packet_provenance = _nested_work_packets(
        response,
        probes,
        task.get("canonical_origin_id", task["id"]),
        forks["origin_policy"]["inference_fingerprint"],
        draw_count,
    )
    work, direct_raw = packets["work"], packets["direct_work"]
    task = {**task, "draws": draw_count}
    # Only reference identities, never labels/scores, are inspected before fitting.
    assert_predictor_independence(
        [],
        [response["work"]["identity"]["rng_namespace"]],
        reference_rng_stream_ids=[response["reference"]["identity"]["rng_namespace"]],
    )
    measured = {
        b["bank_id"]: _observations(
            b, response, response["work"], work, probes, methods, draw_count=draw_count
        )
        for b in banks
    }
    np.savez_compressed(
        root / "CALIBRATION_OBSERVATIONS.npz",
        **{f"{b['bank_id']}__{m}": measured[b["bank_id"]][m] for b in calibration for m in methods},
    )
    gradient = _read_gradient(gradient_receipt, task.get("canonical_origin_id", task["id"]), work)
    cache = CheckpointCache(max_resident=2)
    assert_predictor_independence(
        [r["sample_id"] for rows in direct_raw.values() for r in rows],
        [response["direct_work"]["identity"]["rng_namespace"]],
        reference_sample_ids=[r["sample_id"] for rows in work.values() for r in rows],
        reference_rng_stream_ids=[
            response["work"]["identity"]["rng_namespace"],
            response["reference"]["identity"]["rng_namespace"],
        ],
    )
    direct_measured = {
        b["bank_id"]: _observations(
            b, response, response["direct_work"], direct_raw, probes, methods, draw_count=draw_count
        )
        for b in queries
    }
    from .full_kernels import fit_precomputed_response

    records, models = [], []
    requested_gradients = (
        {"FULL_SCORE_JVP"}
        if selection is None
        else (
            {selection["gradient_reference"]}
            | {
                p["model"]
                for p in selection["primary_models"]
                if p["model"] in {"FULL_SCORE_JVP", "BASE_SCORE_JVP", "MIDPOINT_DIRECTIONAL"}
            }
        )
    )
    specifications = _model_specs(config, methods, alpha, bandwidth_multiplier, selection)
    if selection:
        added_rows, added_models = _frozen_signature_rows(
            selection, completed, task, queries, probes, root / "signature_predictions"
        )
        records.extend(added_rows)
        models.extend(added_models)
    if banks != calibration + queries:
        raise ValueError("Frozen bank order must retain calibration then heldout queries")
    scratch_root = Path(
        os.environ.get("SSVC_V4_ANALYSIS_SCRATCH") or os.environ.get("SLURM_TMPDIR") or root.parent
    )
    if not scratch_root.is_dir():
        raise ValueError("Configured analysis scratch directory must exist")
    with tempfile.TemporaryDirectory(
        prefix="v4-response-coordinates-", dir=scratch_root
    ) as scratch:
        data, layout, scalings = _endpoint_memmap(
            forks,
            banks,
            Path(scratch) / "endpoints.npy",
            cache,
            response["work"]["identity"]["runtime"],
        )
        scratch_metadata = {
            "root": str(scratch_root),
            "temporary_bytes": data.nbytes,
            "dtype": str(data.dtype),
            "value_preservation": "EXACT_ORIGINAL_FP32_OR_LOWER; FP64_INPUT_RETAINS_FP64",
            "arithmetic_dtype": "float64",
            "capacity_check": "FILESYSTEM_AND_INHERITED_CEPH_QUOTAS",
        }
        from .calibration_selection import nested_bank_order

        selection_gram = _calibration_full_gram(data, physical_calibration_count)
        strata = _bank_strata(calibration, training_prompts)
        order = (
            np.arange(physical_calibration_count)
            if bank_selector is None
            else nested_bank_order(
                [b["bank_id"] for b in calibration],
                selection_gram,
                strata=strata,
                method=bank_selector,
                seed=task["seed"],
            )
        )
        fit_indices = order[:calibration_banks]
        calibration = [calibration[i] for i in fit_indices]
        selection_receipt = {
            "method": bank_selector or "ORIGINAL_REGISTERED_ORDER",
            "seed": task["seed"],
            "physical_calibration_banks": physical_calibration_count,
            "selected_bank_ids": [b["bank_id"] for b in calibration],
            "complete_nested_order": [banks[i]["bank_id"] for i in order],
            "training_metadata_strata": strata,
            "columns_per_bank": 3,
            "contrast_order": [c[0] for c in CONTRASTS],
            "parameter_coordinates": "ALL_RAW_PARAMETERS",
            "response_labels_used": False,
            "query_geometry_used": False,
            "query_reference_used": False,
            "selection_scope": "ALL_REGISTERED_CONFIRMATION_BANKS"
            if task.get("role", "development") != "development"
            else "NESTED_REGISTERED_DEVELOPMENT_BANK_POOL",
        }
        np.savez_compressed(root / "CALIBRATION_SELECTION_GRAM.npz", gram=selection_gram)
        atomic_json(root / "CALIBRATION_SELECTION.json", _jsonable(selection_receipt))
        for ti, (target, candidate, baseline) in enumerate(CONTRASTS):
            inputs = compute_response_geometry(
                data,
                layout,
                scalings,
                gradient,
                candidate=candidate,
                baseline=baseline,
                calibration_count=physical_calibration_count,
                calibration_indices=fit_indices,
                base_id=response["work"]["identity"]["runtime"]["model_hash"],
            )
            np.savez_compressed(
                root / f"GEOMETRY_{target}.npz",
                raw_gram=inputs["raw"],
                block_standardized_gram=inputs["block_standardized"],
                effective_gram=inputs["effective"],
                grouped_score_response=inputs["derivative"]["group"],
                prompt_score_response=inputs["derivative"]["prompt"],
            )
            from .empirical_diagnostics import (
                diagnostics_from_collection,
                unavailable_empirical_diagnostics,
            )

            query_direction_ids = [f"{b['bank_id']}/{target}" for b in queries]
            if gradient.get("direction_ids"):
                if (
                    gradient["expansion_policy_identity"]["inference_fingerprint"]
                    != (forks["origin_policy"]["inference_fingerprint"])
                ):
                    raise ValueError("Empirical score products use another expansion policy")
                empirical = diagnostics_from_collection(
                    gradient,
                    inputs["raw"][np.ix_(fit_indices, fit_indices)],
                    inputs["raw"][physical_calibration_count:, fit_indices],
                    np.diag(inputs["raw"])[physical_calibration_count:],
                    origin_id=task.get("canonical_origin_id", task["id"]),
                    calibration_direction_ids=[f"{b['bank_id']}/{target}" for b in calibration],
                    query_direction_ids=query_direction_ids,
                    reference_sample_ids=[],
                    reference_rng_stream_ids=[response["reference"]["identity"]["rng_namespace"]],
                )
            else:
                empirical = unavailable_empirical_diagnostics(
                    "Bound gradient artifact has no actual per-draw directional score products",
                    query_direction_ids,
                )
            atomic_json(root / f"EMPIRICAL_DIAGNOSTICS_{target}.json", _jsonable(empirical))
            for specification in specifications:
                observation = specification["observation"]
                method, rank, representation = (
                    specification[k] for k in ("method", "rank", "representation")
                )
                y = np.stack(
                    [measured[bank["bank_id"]][observation][:, ti] for bank in calibration]
                )
                gram = inputs[representation]
                model = fit_precomputed_response(
                    gram[np.ix_(fit_indices, fit_indices)],
                    y,
                    query_cross_gram=gram[physical_calibration_count:, fit_indices],
                    query_norms=np.diag(gram)[physical_calibration_count:],
                    method=method,
                    alpha=specification["alpha"],
                    rank_cap=rank,
                    bandwidth_multiplier=specification["bandwidth_multiplier"],
                    input_metadata={
                        "input_representation": representation,
                        "input_dimension": inputs["effective_weight_dimension"]
                        if representation == "effective"
                        else inputs["raw_parameter_dimension"],
                        "raw_parameter_dimension": inputs["raw_parameter_dimension"],
                        "dense_weight_materialized": False,
                        "disk_backed_parameter_blocks": True,
                        "block_scales": inputs["block_scales"]
                        if representation == "block_standardized"
                        else None,
                        "block_scale_definition": inputs["block_scale_definition"]
                        if representation == "block_standardized"
                        else None,
                    },
                )
                predictions, geometry = model["predict"](), model["geometry"]()
                label = specification["label"]
                design = specification["frozen_model_id"] or (
                    f"{observation}:input={representation}:alpha={specification['alpha']}:"
                    f"bw={specification['bandwidth_multiplier']}:rank={rank}"
                )
                model_id = f"MODEL_{len(models) + 1:04d}"
                models.append(
                    {
                        "model_id": model_id,
                        "frozen_model_id": specification["frozen_model_id"],
                        "target": target,
                        "design_id": design,
                        **model["metadata"],
                        "method": label,
                        "fit_method": method,
                    }
                )
                np.savez_compressed(
                    root / f"{model_id}.npz",
                    gram=model["gram"],
                    dual_coefficients=model["dual_coefficients"],
                )
                for i, bank in enumerate(queries):
                    diagnostic = (
                        "rho" if representation != "effective" else "effective_weight_residual"
                    )
                    records.append(
                        _row(
                            task,
                            bank,
                            target,
                            label,
                            predictions[i],
                            probes,
                            design=design,
                            diagnostics={
                                diagnostic: float(geometry["rho"][i]),
                                **{
                                    name: None
                                    if empirical[name] is None
                                    else float(empirical[name][i])
                                    for name in ("fisher_residual", "score_residual")
                                },
                            },
                            model_metadata={
                                **model["metadata"],
                                "model_id": model_id,
                                "frozen_model_id": specification["frozen_model_id"],
                            },
                        )
                    )
                del model
            if "FULL_SCORE_JVP" in requested_gradients:
                for i, bank in enumerate(queries):
                    groups = [
                        {"prompt_id": "GROUP:" + "|".join(g), "family": g[0], "interface": g[1]}
                        for g in gradient["groups"]
                    ]
                    records.append(
                        _row(
                            task,
                            bank,
                            target,
                            "FULL_SCORE_JVP",
                            inputs["derivative"]["group"][i],
                            groups,
                            design="ORIGIN_GROUP_GRADIENT",
                            unit="semantic_group",
                            model_metadata={
                                "expansion_point": "ORIGIN",
                                "sample_identity": canonical_hash(gradient["sample_ids"]),
                            },
                        )
                    )
                    if gradient["prompt_subset"]:
                        subset = [
                            next(p for p in probes if p["prompt_id"] == pid)
                            for pid in gradient["prompt_subset"]
                        ]
                        records.append(
                            _row(
                                task,
                                bank,
                                target,
                                "FULL_SCORE_JVP",
                                inputs["derivative"]["prompt"][i],
                                subset,
                                design="ORIGIN_FIXED_PROMPT_SUBSET",
                                model_metadata={
                                    "expansion_point": "ORIGIN",
                                    "diagnostic_subset_only": True,
                                },
                            )
                        )
            del inputs
        del data
    if selection is None and completed and completed.get("development_directional"):
        requested_gradients.update(("BASE_SCORE_JVP", "MIDPOINT_DIRECTIONAL"))
    extra_rows, extra_models = _directional_rows(
        task,
        queries,
        probes,
        gradient["groups"],
        completed=completed,
        requested=requested_gradients,
        selection=selection,
    )
    records.extend(extra_rows)
    models.extend(extra_models)
    if "FULL_SCORE_JVP" in requested_gradients:
        models.append(
            {
                "method": "FULL_SCORE_JVP",
                "artifact": gradient_receipt["artifact"],
                "query_labels_read": False,
                "prediction_coverage": "GROUPS_AND_FIXED_DIAGNOSTIC_PROMPTS",
            }
        )
    if selection:
        for primary in selection["primary_models"]:
            if primary["model"] not in {"FULL_SCORE_JVP", "BASE_SCORE_JVP", "MIDPOINT_DIRECTIONAL"}:
                continue
            if primary["representation"] != "R2" or primary["settings"]:
                raise ValueError(
                    "Actual gradient baseline requires R2 and its fixed derivative settings"
                )
            if primary["model"] != "FULL_SCORE_JVP":
                continue
            # Preserve the measured group / fixed-prompt scopes under the frozen
            # ID, and make the absent prompt derivatives explicitly UNKNOWN.
            selected_rows = []
            for row in records:
                if row["method"] == "FULL_SCORE_JVP" and row["design_id"] in {
                    "ORIGIN_GROUP_GRADIENT",
                    "ORIGIN_FIXED_PROMPT_SUBSET",
                }:
                    selected_rows.append(
                        {
                            **row,
                            "design_id": primary["id"],
                            "model_metadata": {
                                **row["model_metadata"],
                                "frozen_model_id": primary["id"],
                            },
                        }
                    )
            missing = [p for p in probes if p["prompt_id"] not in gradient["prompt_subset"]]
            if missing:
                for bank in queries:
                    for target, _, _ in CONTRASTS:
                        selected_rows.append(
                            _row(
                                task,
                                bank,
                                target,
                                "FULL_SCORE_JVP",
                                np.full((len(missing), 4), np.nan),
                                missing,
                                design=primary["id"],
                                model_metadata={
                                    "frozen_model_id": primary["id"],
                                    "status": "UNKNOWN_NO_ACTUAL_PROMPT_DERIVATIVE",
                                },
                            )
                        )
            records.extend(selected_rows)
            models.append(
                {
                    "method": "FULL_SCORE_JVP",
                    "frozen_model_id": primary["id"],
                    "artifact": gradient_receipt["artifact"],
                    "prediction_coverage": "MEASURED_SCOPES_PLUS_EXPLICIT_UNKNOWN_PROMPTS",
                }
            )
    del gradient
    gc.collect()
    for ti, (target, _, _) in enumerate(CONTRASTS):
        for bank in queries:
            records.append(
                _row(task, bank, target, "ZERO", np.zeros((len(probes), 4)), probes, design="ZERO")
            )
            for observation in methods:
                row = _row(
                    task,
                    bank,
                    target,
                    "DIRECT_SHARED_WORK_DIAGNOSTIC",
                    measured[bank["bank_id"]][observation][:, ti],
                    probes,
                    design=observation,
                )
                row["information_level"] = "CURRENT_QUERY_EVENT_SCORES"
                records.append(row)
                row = _row(
                    task,
                    bank,
                    target,
                    "DIRECT_MEASURE",
                    direct_measured[bank["bank_id"]][observation][:, ti],
                    probes,
                    design=observation,
                )
                row["information_level"] = "INDEPENDENT_CURRENT_QUERY_EVENT_SCORES"
                records.append(row)
    # Publish all predictions BEFORE opening any independent reference data.
    prediction_binding = _write_chunk(
        root / "PREDICTIONS.parquet",
        [
            {
                **{k: _jsonable(v) for k, v in r.items() if k != "model_metadata"},
                "model_metadata_json": json.dumps(
                    _jsonable(r["model_metadata"]), sort_keys=True, allow_nan=False
                ),
            }
            for r in records
        ],
    )
    atomic_json(
        root / "MODELS.json",
        {
            "models": models,
            "fitted_model_ids": sorted({m["method"] for m in models}),
            "binding": binding,
            "calibration_bank_ids": [b["bank_id"] for b in calibration],
            "query_labels_read_for_fit": False,
        },
    )
    ref = _action_rows(
        response["reference"],
        probes,
        role="reference",
        origin_id=task.get("canonical_origin_id", task["id"]),
        policy_fingerprint=forks["origin_policy"]["inference_fingerprint"],
    )
    if response["reference"]["identity"]["draws"] != config["observations"]["reference_primary_n"]:
        raise ValueError("Independent reference budget differs")
    assert_predictor_independence(
        [r["sample_id"] for rows in work.values() for r in rows],
        [response["work"]["identity"]["rng_namespace"]],
        reference_sample_ids=[r["sample_id"] for rows in ref.values() for r in rows],
        reference_rng_stream_ids=[response["reference"]["identity"]["rng_namespace"]],
    )
    references = {
        b["bank_id"]: _observations(b, response, response["reference"], ref, probes, ())
        for b in queries
    }
    validation_result = _reference_validation(
        response,
        task,
        queries,
        probes,
        ref,
        references,
        root,
        forks["origin_policy"]["inference_fingerprint"],
    )
    forbidden_ids = {
        r["sample_id"]
        for packet in (work, ref, direct_raw)
        for rows in packet.values()
        for r in rows
    }
    forbidden_streams = {
        response[k]["identity"]["rng_namespace"] for k in ("work", "reference", "direct_work")
    }
    seed_sets = [
        {r["sample_seed"] for rows in packet.values() for r in rows}
        for packet in (work, ref, direct_raw)
    ]
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Independent observation packets reused an actual generation seed")
    forbidden_seeds = set().union(*seed_sets)
    ref_diagnostics = {
        b["bank_id"]: _apply_mixture_references(
            b,
            response,
            probes,
            references[b["bank_id"]],
            origin_id=task.get("canonical_origin_id", task["id"]),
            forbidden_ids=forbidden_ids,
            forbidden_streams=forbidden_streams,
            forbidden_seeds=forbidden_seeds,
        )
        for b in queries
    }
    atomic_json(
        root / "REFERENCE_DIAGNOSTICS.json",
        _jsonable(
            {
                "rules": load_analysis_rules(),
                "banks": ref_diagnostics,
                "prompt_statistics": references,
                "coverage_guarantee": False,
                "zero_variance_nonalias": "UNRESOLVED",
            }
        ),
    )
    np.savez_compressed(
        root / "REFERENCE_STATISTICS.npz",
        **{
            f"{bid}__{key}": np.asarray([s[key] for s in values])
            for bid, values in references.items()
            for key in ("estimate", "se", "resolved", "covariance_of_mean")
        },
    )

    def evaluated():
        for row in records:
            ti = [c[0] for c in CONTRASTS].index(row["target"])
            stats = references[row["bank_id"]]
            truth = np.array([s["estimate"][ti] for s in stats])
            se = np.array([s["se"][ti] for s in stats])
            resolved = np.array([s["resolved"][ti] for s in stats])
            if row["evaluation_unit"] == "semantic_group":
                grouped = []
                for g in row["groups"]:
                    indices = [
                        i for i, p in enumerate(probes) if p["family"] + "|" + p["interface"] == g
                    ]
                    grouped.append(
                        (
                            truth[indices].mean(0),
                            np.sqrt(np.sum(se[indices] ** 2, axis=0)) / len(indices),
                            resolved[indices].all(0),
                        )
                    )
                truth, se, resolved = [np.array([v[i] for v in grouped]) for i in range(3)]
            else:
                indices = [[p["prompt_id"] for p in probes].index(pid) for pid in row["prompt_ids"]]
                truth, se, resolved = truth[indices], se[indices], resolved[indices]
            yield {
                **row,
                "reference": truth,
                "reference_se": se,
                "reference_resolved": resolved,
                "reference_kind": "EXACT" if row["is_alias"] else "MONTE_CARLO",
                "reference_estimator": "RAW4",
                "reference_independent": True,
                "reference_prompt_independent": True,
                "reference_record_id": f"{row['bank_id']}:{row['target']}",
                "prediction_record_id": prediction_binding["sha256"],
            }

    evaluation = write_evaluation(
        evaluated(), root / "evaluation", binding={**binding, "predictions": prediction_binding}
    )
    result = {
        "kind": "V4_RESPONSE_MAP_FIT",
        "status": "COMPLETED",
        "task_id": task["id"],
        "binding": binding,
        "predictions": prediction_binding,
        "evaluation": evaluation,
        "calibration_banks": len(calibration),
        "query_banks": len(queries),
        "prediction_records": len(records),
        "prompts": len(probes),
        "draws": task["draws"],
        "bank_selector": bank_selector or "ORIGINAL_REGISTERED_ORDER",
        "calibration_selection": _binding(root / "CALIBRATION_SELECTION.json"),
        "observation_packets": packet_provenance,
        "reference_validation": validation_result,
        "fitted_model_ids": sorted({m["method"] for m in models}),
        "reference_read_after_predictions": True,
        "query_response_used_for_fit": False,
        "reference_zero_empirical_variance_nonalias": "UNRESOLVED",
        "scientific_status": "NOT_CERTIFIED",
        "new_actions": 0,
        "new_model_forwards": 0,
        "scratch": scratch_metadata,
    }
    atomic_json(root / "RESULTS.json", result)
    finalize_run(root, binding={**binding, "kind": "V4_RESPONSE_MAP_FIT"}, metadata=result)
    return result


def export_calibration_labels(tasks, *, task_id, out, observation_methods=None):
    """Export only completed development calibration-bank work observations.

    This interface deliberately never opens query scores or reference shards.
    The labels are measured responses, not exact truths or validation outcomes.
    """
    _require_server_cpu()
    from .gpu_collect import CheckpointCache, _fingerprint

    path = Path(tasks)
    plan = json.loads(path.read_text())
    if (
        canonical_hash({k: v for k, v in plan.items() if k != "task_list_hash"})
        != plan["task_list_hash"]
        or plan["source"] != source_identity()
    ):
        raise ValueError("Frozen task/source identity differs")
    task = next(t for t in plan["tasks"] if t["id"] == task_id)
    config = plan["config"]
    if (
        task["kind"] != "map"
        or task["seed"] not in config["qwen"]["seed_roles"]["development"]
        or task.get("role", "development") != "development"
    ):
        raise ValueError("Signature learning labels require registered development origins")
    completed_path = Path(plan["root"]) / "tasks" / task_id / "COMPLETE.json"
    completed = json.loads(completed_path.read_text())
    if (
        completed.get("status") != "COMPLETED"
        or completed.get("task") != task
        or completed.get("execution_kind") != "REAL_CUDA_MODEL"
        or completed.get("config_hash") != canonical_hash(config)
        or completed.get("source_hash") != plan["source"]["sha256"]
    ):
        raise ValueError("Actual completed map/config/source identity required")
    origin = completed["origin_id"]
    if origin != task_id and task.get("reuse_prefix_task") != origin:
        raise ValueError("Canonical reused origin differs from registered prefix")
    forks, response = bound_json(completed["forks"]), bound_json(completed["response"])
    if forks["origin_id"] != origin or response["origin_id"] != origin:
        raise ValueError("Canonical fork/response origin mismatch")
    probes = plan["inputs"]["panels"]["observation"][: task["prompts"]]
    methods = tuple(observation_methods or config["observations"]["methods"])
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(PRIMARY_METHODS):
        raise ValueError("Only registered primary observations may be exported")
    banks = [b for b in forks["banks"] if b["role"] == "calibration"]
    if len(banks) != task["calibration_banks"] or len({b["bank_id"] for b in banks}) != len(banks):
        raise ValueError("Complete registered calibration bank set required")
    cache = CheckpointCache(max_resident=2)
    runtime = response["work"]["identity"]["runtime"]
    if (
        _fingerprint(cache.load(forks["origin_policy"]), runtime)
        != forks["origin_policy"]["inference_fingerprint"]
    ):
        raise ValueError("Actual origin policy fingerprint differs")
    raw = _action_rows(
        response["work"],
        probes,
        role="work",
        origin_id=origin,
        policy_fingerprint=forks["origin_policy"]["inference_fingerprint"],
    )
    if response["work"]["identity"]["draws"] != task["draws"]:
        raise ValueError("Measured calibration work budget differs")
    values = {m: [] for m in methods}
    units = []
    for bank in banks:
        for policy in bank["policies"].values():
            if _fingerprint(cache.load(policy), runtime) != policy["inference_fingerprint"]:
                raise ValueError("Actual calibration endpoint fingerprint differs")
        observation = _observations(bank, response, response["work"], raw, probes, methods)
        for i, (target, candidate, baseline) in enumerate(CONTRASTS):
            left, right = bank["policies"][candidate], bank["policies"][baseline]
            alias = left["inference_fingerprint"] == right["inference_fingerprint"]
            if bank["contrasts"][target]["exact_inference_alias"] is not alias:
                raise ValueError("Calibration alias identity mismatch")
            units.append(
                {
                    "origin_id": origin,
                    "task_id": task_id,
                    "bank_id": bank["bank_id"],
                    "contrast_id": target,
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_id": left["candidate_id"],
                    "baseline_id": right["candidate_id"],
                    "candidate_fingerprint": left["inference_fingerprint"],
                    "baseline_fingerprint": right["inference_fingerprint"],
                    "pair_id": canonical_hash(
                        [origin, right["inference_fingerprint"], left["inference_fingerprint"]]
                    ),
                    "seed": task["seed"],
                    "arm": task["arm"],
                    "step": task["step"],
                    "role": "development",
                    "bank_role": "calibration",
                    "is_alias": alias,
                }
            )
            for method in methods:
                values[method].append(observation[method][:, i])
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(root / "LABELS.npz", **{m: np.asarray(v) for m, v in values.items()})
    binding = {
        "tasks": _binding(path),
        "task": _binding(completed_path),
        "forks": completed["forks"],
        "response": completed["response"],
        "config_hash": canonical_hash(config),
        "source": plan["source"],
        "execution_kind": completed["execution_kind"],
    }
    receipt = {
        "kind": "V4_CALIBRATION_LABELS",
        "config_hash": canonical_hash(config),
        "source_hash": plan["source"]["sha256"],
        "origin_id": origin,
        "task_id": task_id,
        "seed": task["seed"],
        "arm": task["arm"],
        "step": task["step"],
        "role": "development",
        "bank_role_scope": "calibration",
        "calibration_bank_count": len(banks),
        "draws": task["draws"],
        "arrays": _binding(root / "LABELS.npz"),
        "units": units,
        "prompt_ids": [p["prompt_id"] for p in probes],
        "probe_groups": [p["family"] + "|" + p["interface"] for p in probes],
        "event_order": list(EVENTS),
        "observation_methods": list(methods),
        "array_axes": ["unit", "prompt", "event"],
        "work_sample_identity": canonical_hash(response["work"]["identity"]),
        "binding": binding,
        "query_labels_read": False,
        "reference_labels_read": False,
        "exact_event_mass_claim": False,
    }
    atomic_json(root / "LABELS.json", receipt)
    finalize_run(root, {**binding, "kind": "V4_CALIBRATION_LABELS"})
    return {
        "status": "COMPLETE",
        "out": str(root),
        "labels": _binding(root / "LABELS.json"),
        "arrays": receipt["arrays"],
    }
