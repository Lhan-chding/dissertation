"""Native paid sequence signatures and origin-only semantic state features.

Illegal outputs contribute zero to each of the seven unconditional diagnostics.
Constraint satisfaction conditional on parsing is a separate, nullable statistic.
Signature scores concern fixed, independently drawn sequences, not event masses.
"""

import hashlib
import json
from collections import Counter

import numpy as np

from ..constraint_solver import satisfies
from ..verifiers import classify, strict_parse

DIAGNOSTIC_NAMES = (
    "coordinate_correct_0",
    "coordinate_correct_1",
    "coordinate_correct_2",
    "coordinate_correct_3",
    "constraint_fraction",
    "edit_exactly_one",
    "copy_observation",
)
EVENTS = ("X", "S", "W", "I")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _world(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(v) is not int or not 0 <= v < 100 for v in value)
    ):
        raise ValueError("World must contain four integers in 0..99")
    return list(value)


def output_diagnostics(raw, scene):
    """Compute existing event and seven deterministic diagnostics without repair."""
    truth, observed = _world(scene["truth_world"]), _world(scene["observed_world"])
    cue = scene.get("cue")
    if not isinstance(cue, dict):
        raise ValueError("Explicit constraint cue required")
    satisfies(observed, cue)  # validate even for an illegal output
    parsed = strict_parse(raw)
    result = dict.fromkeys(DIAGNOSTIC_NAMES, 0.0)
    conditional = None
    if parsed is not None:
        if cue["family"] == "duplicate_encoding":
            checks = [parsed[cue["known_index"]] == cue["known_value"]]
        elif cue["family"] == "cross_series":
            checks = [parsed[i] + parsed[j] == total for i, j, total in cue["edges"]]
        else:
            a, b, c, d = parsed
            checks = [a - 2 * b + c == 0, b - 2 * c + d == 0]
        conditional = float(np.mean(checks))
        result.update(
            {
                f"coordinate_correct_{i}": float(x == y)
                for i, (x, y) in enumerate(zip(parsed, truth, strict=True))
            }
        )
        result.update(
            constraint_fraction=conditional,
            edit_exactly_one=float(sum(x != y for x, y in zip(parsed, observed, strict=True)) == 1),
            copy_observation=float(parsed == observed),
        )
    return {
        **result,
        "category": classify(raw, truth, scene["operation"]),
        "parsed": parsed is not None,
        "constraint_fraction_conditional": conditional,
    }


def aggregate_diagnostics(outputs, scene):
    rows = [output_diagnostics(raw, scene) for raw in outputs]
    if not rows:
        raise ValueError("At least one actual output required")
    valid = [r for r in rows if r["parsed"]]
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
        "diagnostic_names": list(DIAGNOSTIC_NAMES),
        "diagnostics": np.mean([[r[n] for n in DIAGNOSTIC_NAMES] for r in rows], axis=0),
        "event_probabilities": np.array(
            [sum(r["category"] == e for r in rows) / len(rows) for e in EVENTS]
        ),
        "constraint_fraction_conditional": float(
            np.mean([r["constraint_fraction_conditional"] for r in valid])
        )
        if valid
        else None,
    }


def build_state_features(origin_state, *, origin_id, mode="EVENT_ONLY"):
    """Only origin signature-panel state may enter a predictor; never endpoint labels."""
    allowed = {
        "stage",
        "origin_id",
        "panel_role",
        "event_probabilities",
        "diagnostics",
        "diagnostic_names",
        "prompt_ids",
        "panel_id",
        "sample_ids",
    }
    if (
        set(origin_state) - allowed
        or origin_state.get("stage") != "ORIGIN"
        or origin_state.get("origin_id") != origin_id
        or origin_state.get("panel_role") != "signature"
    ):
        raise ValueError("Predictor state must be bound origin-only signature-panel statistics")
    p = np.asarray(origin_state["event_probabilities"], dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 4
        or not len(p)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Four origin event probabilities must be normalized")
    if mode == "EVENT_ONLY":
        result = p.copy()
    elif mode == "EVENT_PLUS_DIAGNOSTICS":
        d = np.asarray(origin_state.get("diagnostics"), dtype=float)
        if (
            tuple(origin_state.get("diagnostic_names", ())) != DIAGNOSTIC_NAMES
            or d.shape != (len(p), 7)
            or not np.isfinite(d).all()
            or np.any(d < 0)
            or np.any(d > 1)
        ):
            raise ValueError("Seven named unconditional origin diagnostics required")
        if np.any(d > (1 - p[:, 3:4]) + 1e-12):
            raise ValueError("Unconditional diagnostics cannot exceed parse-valid probability")
        result = np.concatenate((p, d), axis=1)
    else:
        raise ValueError("Unknown state-feature mode")
    result.setflags(write=False)
    return result


def validate_panel_splits(splits):
    """Observation/reference may share scenes, but never samples or RNG streams."""
    roles = ("train", "observation", "reference", "signature")
    if set(splits) != set(roles):
        raise ValueError("Explicit train/observation/reference/signature split manifest required")
    fields = ("base_scene_ids", "prompt_ids", "sample_ids", "rng_namespaces")
    sets = {}
    for role in roles:
        for field in fields:
            values = splits[role].get(field)
            if (
                not isinstance(values, (list, tuple))
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Split identities must be unique explicit strings")
            if field != "sample_ids" and not values:
                raise ValueError("Scene, prompt and RNG split identities cannot be empty")
            sets[role, field] = set(values)
    for i, a in enumerate(roles):
        for b in roles[i + 1 :]:
            for field in fields:
                if {a, b} == {"observation", "reference"} and field in (
                    "base_scene_ids",
                    "prompt_ids",
                ):
                    continue
                if sets[a, field] & sets[b, field]:
                    raise ValueError(f"{a}/{b} {field} overlap")
    return {"status": "VERIFIED", "split_identity_sha256": _digest(splits)}


def build_signature(baseline_scores, candidate_scores, *, sample_rows, panel_manifest):
    """Align all 36x16 full sequence log-probability differences in native order.

    Raw samples are the independent fixed origin draws. Candidate labels are not
    read. Existing physical score-call accounting is passed through, never inferred
    from table size or alias status.
    """
    guard = validate_panel_splits(panel_manifest["splits"])
    runtime = panel_manifest.get("runtime_identity")
    if not isinstance(runtime, str) or not runtime:
        raise ValueError("Signature requires one normalized scoring/decoding runtime identity")
    prompts = panel_manifest["prompt_ids"]
    if (
        len(prompts) != 36
        or len(set(prompts)) != 36
        or set(prompts) != set(panel_manifest["splits"]["signature"]["prompt_ids"])
    ):
        raise ValueError("Native signature requires exactly 36 distinct fixed prompts")
    samples = list(sample_rows)
    if len(samples) != 576 or len({r["sample_id"] for r in samples}) != 576:
        raise ValueError("Native signature requires 576 unique draw identities")
    sig = panel_manifest["splits"]["signature"]
    if set(sig["sample_ids"]) != {r["sample_id"] for r in samples}:
        raise ValueError("Signature sample IDs differ from frozen panel")
    eos = panel_manifest.get("eos_token_ids")
    if not isinstance(eos, list) or not eos or any(type(v) is not int for v in eos):
        raise ValueError("Actual EOS token identities required")
    by_prompt = {}
    identities = []
    for r in samples:
        p, d, tokens = r["prompt_id"], r["draw_index"], r["token_ids"]
        if (
            p not in prompts
            or type(d) is not int
            or not 0 <= d < 16
            or (p, d) in by_prompt
            or r["origin_id"] != panel_manifest["origin_id"]
            or r.get("runtime_identity") != runtime
            or r["role"] != "signature"
            or r["rng_namespace"] not in sig["rng_namespaces"]
            or r["base_scene_id"] not in sig["base_scene_ids"]
            or not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or r.get("action_mask") != [1] * len(tokens)
            or not r.get("input_hash")
            or type(r.get("max_new_tokens")) is not int
            or not 0 < len(tokens) <= r["max_new_tokens"]
        ):
            raise ValueError("Signature draw/prompt/input/token identity differs")
        terminal = tokens[-1] in eos
        if (
            any(t in eos for t in tokens[:-1])
            or r.get("eos_seen") is not terminal
            or r.get("truncated") is not (not terminal)
        ):
            raise ValueError("EOS must be retained as the actual terminal action")
        if (terminal and r.get("stop_reason") not in ("eos", "EOS")) or (
            not terminal
            and (
                r.get("stop_reason") not in ("length", "max_new_tokens", "MAX_NEW_TOKENS")
                or len(tokens) != r["max_new_tokens"]
            )
        ):
            raise ValueError("Sequence terminal/truncation contract differs")
        by_prompt[p, d] = r
    ordered = [by_prompt[p, d] for p in prompts for d in range(16)]
    scenes = []
    for p in prompts:
        ids = {by_prompt[p, d]["base_scene_id"] for d in range(16)}
        hashes = {by_prompt[p, d]["input_hash"] for d in range(16)}
        if len(ids) != 1 or len(hashes) != 1:
            raise ValueError("Fixed signature prompt changed across draws")
        scenes.extend(ids)
    if len(set(scenes)) != 18 or set(Counter(scenes).values()) != {2}:
        raise ValueError("Signature panel must contain 18 scenes with two interfaces")
    scores, fingerprints = [], []
    for source in (baseline_scores, candidate_scores):
        source = list(source)
        scored = {r["sample_id"]: r for r in source}
        if (
            len(scored) != 576
            or len(source) != 576
            or set(scored) != {r["sample_id"] for r in ordered}
        ):
            raise ValueError("Each endpoint must score every exact signature draw once")
        fp = {r.get("inference_fingerprint") for r in source}
        if len(fp) != 1 or not next(iter(fp)):
            raise ValueError("Each score axis requires one actual policy fingerprint")
        values = []
        for r in ordered:
            s = scored[r["sample_id"]]
            if any(
                s.get(k) != r[k]
                for k in (
                    "prompt_id",
                    "input_hash",
                    "role",
                    "rng_namespace",
                    "runtime_identity",
                )
            ):
                raise ValueError("Scoring and generation sample identity differ")
            if (
                ("token_ids" in s and s["token_ids"] != r["token_ids"])
                or (
                    "token_ids" not in s
                    and s.get("shared_token_identity") != _digest(r["token_ids"])
                )
                or (
                    "shared_token_identity" in s
                    and s["shared_token_identity"] != _digest(r["token_ids"])
                )
            ):
                raise ValueError("Scored shared token identity differs")
            token_logp = np.asarray(s.get("token_logprobs"), dtype=float)
            value = s.get("sequence_logp")
            if (
                token_logp.shape != (len(r["token_ids"]),)
                or not np.isfinite(token_logp).all()
                or np.any(token_logp > 0)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or not np.isclose(value, token_logp.sum(dtype=np.float64), atol=1e-10, rtol=1e-12)
            ):
                raise ValueError(
                    "Scores must be full sequence log-probabilities including terminal"
                )
            values.append(value)
        fingerprints.append(next(iter(fp)))
        scores.append(np.array(values, dtype=float))
    for r in ordered:
        identities.append(
            {
                k: r[k]
                for k in (
                    "sample_id",
                    "prompt_id",
                    "base_scene_id",
                    "input_hash",
                    "token_ids",
                    "action_mask",
                    "rng_namespace",
                    "draw_index",
                    "eos_seen",
                    "truncated",
                )
            }
        )
    values = scores[1] - scores[0]
    if fingerprints[0] == fingerprints[1] and np.any(values != 0):
        raise ValueError("Same policy scores disagree; record numerical parity separately")
    values.setflags(write=False)
    paid = panel_manifest.get("paid_score_actions")
    if paid is not None and (type(paid) is not int or paid < 0):
        raise ValueError("Actual paid-score count must come from the acquisition ledger")
    return {
        "values": values,
        "provenance": {
            "native_width": 576,
            "prompts": 36,
            "draws_per_prompt": 16,
            "panel_id": panel_manifest["panel_id"],
            "panel_role": "signature",
            "origin_id": panel_manifest["origin_id"],
            "prompt_ids": list(prompts),
            "base_scene_ids": sorted(set(scenes)),
            "ordered_sample_ids": [r["sample_id"] for r in ordered],
            "baseline_fingerprint": fingerprints[0],
            "candidate_fingerprint": fingerprints[1],
            "sample_identity_sha256": _digest(identities),
            "split_guard": guard["status"],
            "split_identity_sha256": guard["split_identity_sha256"],
            "score_definition": "FULL_SEQUENCE_LOGPROB_DIFFERENCE",
            "runtime_identity": runtime,
            "target_outcomes_used": False,
            "paid_feature": True,
            "paid_score_actions": paid,
            "required_score_entries": 1152,
        },
    }
