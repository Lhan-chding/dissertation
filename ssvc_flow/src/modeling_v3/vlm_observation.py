"""V3 finite-horizon action observations and immutable two-worker execution.

Importing this module never imports a model framework. The production bridge
uses the inherited uncached-prefix adapter only after the caller's CUDA gate.
Reference observations remain estimates; their store is separate from predictors.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path, PurePosixPath

PATH = "uncached_prefix_recompute"
ROLES = {"pilot", "main", "reference", "direct_count", "null"}
EVENTS = "XSWI"
REFERENCE_LOOKS = (4096, 8192, 16384, 32768, 65536)


class ObservationFault(ValueError):
    """Retain a returned but invalid model result for the failed-attempt ledger."""

    def __init__(self, reason, raw_result):
        super().__init__(reason)
        self.raw_result = raw_result


def _fault_json(value):
    """Losslessly name invalid numeric values instead of dropping returned data."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: _fault_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fault_json(child) for child in value]
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_binding(binding):
    if not isinstance(binding, dict) or not {"path", "sha256"} <= set(binding):
        raise ValueError("Input needs an explicit path and sha256 binding")
    path = Path(binding["path"])
    if not path.is_file() or file_digest(path) != binding["sha256"]:
        raise ValueError("Input file missing or hash changed: " + str(path))
    return path


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _publish(path, value):
    """Publish complete bytes atomically without replacing any existing result."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    _write_new(temporary, value)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _logps(values, n):
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    values = list(values)
    if len(values) != n or any(
        type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in values
    ):
        raise ValueError("Every sampled token needs one finite nonpositive logp")
    return [float(v) for v in values]


def validate_action(generation, *, eos_ids, max_new_tokens=64, tokenizer=None):
    """EOS belongs to the action. A sampled PAD id is not batch padding."""
    tokens = generation.get("token_ids")
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(tokens) > max_new_tokens
        or any(type(t) is not int or t < 0 for t in tokens)
    ):
        raise ValueError("Invalid action tokens or generation horizon")
    terminal = tokens[-1] in eos_ids
    if any(t in eos_ids for t in tokens[:-1]):
        raise ValueError("Action continues after the first EOS")
    if not terminal and len(tokens) != max_new_tokens:
        raise ValueError("Non-EOS action must reach the full truncation horizon")
    if generation.get("completion_length") != len(tokens):
        raise ValueError("Action length differs from token ledger")
    if generation.get("stop_reason") != ("eos" if terminal else "length"):
        raise ValueError("EOS/truncation marker differs from sampled tokens")
    _logps(generation["behavior_token_logprobs"], len(tokens))
    if not isinstance(generation.get("raw_completion"), str):
        raise ValueError("Raw completion text is required, including invalid parses")
    if tokenizer is not None:
        raw = tokenizer.decode(
            tokens[:-1] if terminal else tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if generation["raw_completion"] != raw:
            raise ValueError("Raw completion differs from exact token decode")
    return {
        "eos": terminal,
        "truncated": not terminal,
        "terminal_token_id": tokens[-1] if terminal else None,
        "padding_positions": [],
        "sampled_special_tokens_retained": True,
    }


def request_identity(
    *, origin_id, prompt_id, role, draw_index, rng_namespace, proposal, proposal_candidates
):
    if role not in ROLES or proposal not in {"ORIGIN", "MIX", "DIRECT"}:
        raise ValueError("Unknown observation role or proposal")
    if type(draw_index) is not int or draw_index < 0 or not rng_namespace:
        raise ValueError("Each draw needs its nonnegative index and frozen RNG namespace")
    count = 2 if proposal == "MIX" else 1
    if len(proposal_candidates) != count or len(set(proposal_candidates)) != count:
        raise ValueError("MIX requires two distinct endpoints; ORIGIN/DIRECT one source")
    key = digest(
        {
            "origin_id": origin_id,
            "prompt_id": prompt_id,
            "role": role,
            "draw_index": draw_index,
            "rng_namespace": rng_namespace,
            "proposal": proposal,
            "proposal_candidates": proposal_candidates,
        }
    )
    coin_key = digest([key, "iid-source-coin"])
    # A separate per-request random stream; neither balanced halves nor alternation.
    source_index = int(random.Random(int(coin_key, 16)).random() < 0.5) if count == 2 else 0
    rng_key = digest([key, "completion"])
    return {
        "sample_key": key,
        "sample_rng_key": rng_key,
        "sample_seed": int(rng_key[:16], 16) % (2**63 - 1),
        "source_coin_key": coin_key if count == 2 else None,
        "proposal_source_index": source_index,
        "proposal_source": proposal_candidates[source_index],
    }


class PrefixObservationBackend:
    """Execution adapter. policy_loader restores a bound full checkpoint once per group."""

    def __init__(
        self,
        adapter,
        *,
        runtime_identity,
        policy_loader,
        data_root=None,
        annotator=None,
        parity_tolerances=None,
        state_guard=None,
    ):
        self.adapter, self.runtime_identity = adapter, dict(runtime_identity)
        self.policy_loader, self.data_root = policy_loader, data_root
        self.annotator, self.state_guard = annotator, state_guard
        self.parity_tolerances = parity_tolerances
        self.current_policy = None
        self.current_fingerprint = None
        self._prepared_cache = {}
        if adapter.audit.get("probability_execution") != PATH:
            raise ValueError("V3 observations require the certified prefix recompute path")
        if parity_tolerances is None:
            raise ValueError("Generation parity tolerances must be frozen before execution")
        _parity_values([[0.0]], parity_tolerances, sequence_errors=[0.0])
        self.counters = {
            "generated_sequences": 0,
            "generated_tokens": 0,
            "scored_sequences": 0,
            "scored_tokens": 0,
            "accepted_samples": 0,
            "score_alias_reuses": 0,
            "policy_loads": 0,
            "elapsed_seconds": 0.0,
            "forward_calls": 0,
            "backward_calls": 0,
            "adam_updates": 0,
        }

    def activate(self, policy):
        key = digest(policy)
        if self.current_policy != key:
            returned = self.policy_loader(policy)
            expected = policy["inference_fingerprint"]
            actual = (
                returned.get("inference_fingerprint") if isinstance(returned, dict) else returned
            )
            if actual != expected:
                raise ValueError(
                    "Restored policy inference fingerprint differs from frozen manifest"
                )
            self.current_policy, self.current_fingerprint = key, actual
            self.counters["policy_loads"] += 1
        return self.current_fingerprint

    def _prepared(self, prompt):
        key = digest(prompt)
        if key not in self._prepared_cache:
            prepared = self.adapter.prepare(
                prompt["prompt"], prompt.get("data_root") or self.data_root
            )
            audit = prepared["audit"]
            if audit.get("enable_thinking") is not False:
                raise ValueError("V3 requires the frozen non-thinking prompt")
            image = prompt.get("interface") == "IMAGE_CUE_FRESH"
            if bool(audit.get("image_token_count")) != image or (
                image and not audit.get("pixel_values_hash")
            ):
                raise ValueError("Visual input disappeared or changed interface")
            for field in ("final_prompt_hash", "input_tensor_hash"):
                if not audit.get(field):
                    raise ValueError("Prepared input lacks " + field)
            self._prepared_cache[key] = prepared
        return self._prepared_cache[key]

    @contextlib.contextmanager
    def _measurement(self):
        before = self.state_guard() if self.state_guard else None
        count = self.adapter.forward_calls
        started = time.perf_counter()
        try:
            yield
        finally:
            self.counters["elapsed_seconds"] += time.perf_counter() - started
            self.counters["forward_calls"] += self.adapter.forward_calls - count
            if self.state_guard and before != self.state_guard():
                raise ValueError("Observation mutated parameter/Adam/frozen state")

    def generate(self, prompt, *, seed, max_new_tokens=64):
        if self.current_fingerprint is None:
            raise ValueError("Activate a bound policy before model execution")
        prepared = self._prepared(prompt)
        with self._measurement():
            raw = self.adapter.generate(
                prepared, seed=seed, max_new_tokens=max_new_tokens, do_sample=True
            )
            self.counters["generated_sequences"] += 1
            self.counters["generated_tokens"] += len(raw.get("token_ids", []))
            try:
                flags = validate_action(
                    raw,
                    eos_ids=self.adapter.eos_ids,
                    max_new_tokens=max_new_tokens,
                    tokenizer=self.adapter.processor.tokenizer,
                )
            except (ValueError, KeyError) as exc:
                raise ObservationFault(str(exc), raw) from exc
            # This second probability path is measured, and explicitly charged.
            returned_score = self.adapter.logprobs(prepared, raw["token_ids"], require_grad=False)
            self.counters["scored_sequences"] += 1
            self.counters["scored_tokens"] += len(raw["token_ids"])
            rescored = _logps(returned_score, len(raw["token_ids"]))
        behavior = _logps(raw["behavior_token_logprobs"], len(raw["token_ids"]))
        delta = [abs(a - b) for a, b in zip(behavior, rescored, strict=True)]
        parity = _parity_values(
            [delta], self.parity_tolerances, sequence_errors=[abs(sum(behavior) - sum(rescored))]
        )
        if not parity["passed"]:
            raise ObservationFault(
                "Generation/rescore parity exceeds the frozen tolerance",
                {**raw, "rescored_token_logprobs": rescored, "parity": parity},
            )
        if self.annotator:
            annotation = self.annotator(raw["raw_completion"], prompt["scene"])
        else:
            from ..r2_runtime import _json_safe, annotate_diagnostic

            annotation = _json_safe(annotate_diagnostic(raw["raw_completion"], prompt["scene"]))
        if annotation.get("category") not in tuple(EVENTS):
            raise ValueError("Semantic parser must return exactly one X/S/W/I event")
        self.counters["accepted_samples"] += 1
        return {
            **raw,
            **flags,
            "annotation": annotation,
            "category": annotation["category"],
            "event_onehot": [int(annotation["category"] == event) for event in EVENTS],
            "generation_sequence_logp": sum(behavior),
            "rescored_token_logprobs": rescored,
            "generation_parity": parity,
            "prompt_id": prompt["prompt_id"],
            "prompt_record_hash": digest(prompt),
            "input_audit": prepared["audit"],
            "inference_fingerprint": self.current_fingerprint,
            "probability_execution": PATH,
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": sorted(self.adapter.eos_ids),
            "pad_token_id": self.adapter.pad_id,
            "runtime_identity": self.runtime_identity,
        }

    def score(self, prompt, sample):
        if self.current_fingerprint is None:
            raise ValueError("Activate a bound policy before scoring")
        prepared = self._prepared(prompt)
        if sample["prompt_record_hash"] != digest(prompt):
            raise ValueError("Scored prompt identity differs from sampled prompt")
        for field in ("final_prompt_hash", "input_tensor_hash", "pixel_values_hash"):
            if prepared["audit"].get(field) != sample["input_audit"].get(field):
                raise ValueError("Scored input identity differs from sampled input")
        validate_action(
            sample,
            eos_ids=self.adapter.eos_ids,
            max_new_tokens=sample["max_new_tokens"],
            tokenizer=self.adapter.processor.tokenizer,
        )
        with self._measurement():
            returned_score = self.adapter.logprobs(
                prepared, sample["token_ids"], require_grad=False
            )
            self.counters["scored_sequences"] += 1
            self.counters["scored_tokens"] += len(sample["token_ids"])
            scores = _logps(returned_score, len(sample["token_ids"]))
        return {
            "proposal_sample_key": sample["sample_key"],
            "sample_record_hash": digest(sample),
            "prompt_id": prompt["prompt_id"],
            "prompt_record_hash": digest(prompt),
            "input_audit": prepared["audit"],
            "token_ids": sample["token_ids"],
            "category": sample["category"],
            "token_logprobs": scores,
            "sequence_logp": sum(scores),
            "inference_fingerprint": self.current_fingerprint,
            "score_execution": "MEASURED",
            "score_is_independent_draw": False,
            "probability_execution": PATH,
            "max_new_tokens": sample["max_new_tokens"],
            "eos_token_ids": sorted(self.adapter.eos_ids),
            "runtime_identity": self.runtime_identity,
        }

    def score_known_actions(self, prompt, token_actions, *, max_new_tokens=64):
        """Score predeclared complete actions without inventing a sampled rollout."""
        if self.current_fingerprint is None:
            raise ValueError("Activate a bound policy before known-action scoring")
        prepared = self._prepared(prompt)
        if len({tuple(tokens) for tokens in token_actions}) != len(token_actions):
            raise ValueError("Known-action request has duplicated token paths")
        rows = []
        for tokens in token_actions:
            if (
                not isinstance(tokens, list)
                or not tokens
                or len(tokens) > max_new_tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or any(t in self.adapter.eos_ids for t in tokens[:-1])
                or (tokens[-1] not in self.adapter.eos_ids and len(tokens) != max_new_tokens)
            ):
                raise ValueError("Known-action scoring requires a complete EOS/truncation path")
            terminal = tokens[-1] in self.adapter.eos_ids
            raw = self.adapter.processor.tokenizer.decode(
                tokens[:-1] if terminal else tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if self.annotator:
                annotation = self.annotator(raw, prompt["scene"])
            else:
                from ..r2_runtime import _json_safe, annotate_diagnostic

                annotation = _json_safe(annotate_diagnostic(raw, prompt["scene"]))
            if annotation.get("category") not in tuple(EVENTS):
                raise ValueError("Known action lacks an exhaustive X/S/W/I semantic label")
            with self._measurement():
                returned = self.adapter.logprobs(prepared, tokens, require_grad=False)
                self.counters["scored_sequences"] += 1
                self.counters["scored_tokens"] += len(tokens)
                values = _logps(returned, len(tokens))
            rows.append(
                {
                    "token_ids": tokens,
                    "raw_completion": raw,
                    "annotation": annotation,
                    "category": annotation["category"],
                    "token_logprobs": values,
                    "sequence_logp": sum(values),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_record_hash": digest(prompt),
                    "input_audit": prepared["audit"],
                    "inference_fingerprint": self.current_fingerprint,
                    "probability_execution": PATH,
                    "eos_token_ids": sorted(self.adapter.eos_ids),
                    "max_new_tokens": max_new_tokens,
                    "runtime_identity": self.runtime_identity,
                    "action_source": "PREDECLARED_KNOWN_ACTION",
                    "is_sampling_draw": False,
                    "event_probability_exact": False,
                    "eos": terminal,
                    "truncated": not terminal,
                }
            )
        return rows


def _parity_values(differences, tolerances, *, sequence_errors):
    required = ("mean_abs_token_logp", "max_abs_token_logp", "max_abs_sequence_logp")
    if any(
        type(tolerances.get(k)) not in (float, int)
        or not math.isfinite(tolerances[k])
        or tolerances[k] < 0
        for k in required
    ):
        raise ValueError("Parity needs three finite predeclared tolerances")
    flat = [value for row in differences for value in row]
    if not flat or not sequence_errors:
        raise ValueError("Parity needs nonempty measured tokens and sequences")
    observed = {
        "mean_abs_token_logp": statistics.fmean(flat),
        "max_abs_token_logp": max(flat),
        "max_abs_sequence_logp": max(sequence_errors),
    }
    return {
        **observed,
        "tokens": len(flat),
        "sequences": len(differences),
        "tolerances": dict(tolerances),
        "passed": all(observed[k] <= tolerances[k] for k in required),
    }


def compare_probability_receipts(left, right, *, tolerances):
    """Check complete null identity and frozen tolerances across any two invocations."""
    if not left["gpu_identity"] or not right["gpu_identity"]:
        raise ValueError("Probability parity needs actual measured GPU identities")
    if left["runtime_identity"] != right["runtime_identity"]:
        raise ValueError("Cross-GPU source/model/runtime locks differ")
    if left["tolerances"] != tolerances or right["tolerances"] != tolerances:
        raise ValueError("Cross-GPU tolerance lock changed after collection")
    a = {r["proposal_sample_key"]: r for r in left["scores"]}
    b = {r["proposal_sample_key"]: r for r in right["scores"]}
    if not a or set(a) != set(b) or len(a) != len(left["scores"]) or len(b) != len(right["scores"]):
        raise ValueError("Cross-GPU request sets are missing or duplicated")
    differences, sequence = [], []
    for key in sorted(a):
        for field in (
            "prompt_record_hash",
            "input_audit",
            "token_ids",
            "inference_fingerprint",
            "probability_execution",
            "max_new_tokens",
            "eos_token_ids",
        ):
            if a[key][field] != b[key][field]:
                raise ValueError("Cross-GPU action/input/policy identity differs: " + field)
        x, y = (_logps(r["token_logprobs"], len(r["token_ids"])) for r in (a[key], b[key]))
        differences.append([abs(i - j) for i, j in zip(x, y, strict=True)])
        sequence.append(abs(sum(x) - sum(y)))
    result = _parity_values(differences, tolerances, sequence_errors=sequence)
    return {
        "status": "PASS" if result["passed"] else "FAIL_NULL_PARITY",
        **result,
        "left_receipt_hash": digest(left),
        "right_receipt_hash": digest(right),
        "zero_treatment": "IDENTICAL_POLICY",
        "candidate_effect_not_established": True,
    }


def compare_cross_gpu_receipts(left, right, *, tolerances):
    """Require the same actions, inputs and policy on two distinct actual GPUs."""
    if left["gpu_identity"] == right["gpu_identity"]:
        raise ValueError("Cross-GPU parity needs two distinct measured GPU identities")
    return compare_probability_receipts(left, right, tolerances=tolerances)


def audit_semantic_aliases(samples):
    x = [row for row in samples if row["category"] == "X"]
    strings = {row["raw_completion"] for row in x}
    actions = {tuple(row["token_ids"]) for row in x}
    return {
        "correct_draws": len(x),
        "distinct_correct_strings": len(strings),
        "distinct_correct_token_actions": len(actions),
        "observed_semantic_aliases": len(actions) > 1,
        "enumeration_complete": False,
        "known_string_is_whole_event": False,
        "scope": "Observed correct actions do not prove enumeration of the semantic event",
    }


def known_action_probability(scores):
    """A legal strong baseline for a known finite subset, never an event oracle."""
    keys = [tuple(r["token_ids"]) for r in scores]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate actions would double-count known-string probability")
    identities = {(r["prompt_record_hash"], r["inference_fingerprint"]) for r in scores}
    if len(identities) != 1 or not scores:
        raise ValueError("Known actions must share one prompt and policy")
    if any(row.get("category") != "X" for row in scores):
        raise ValueError("Known-X baseline includes a non-X action")
    for row in scores:
        tokens, eos = row["token_ids"], row["eos_token_ids"]
        if (
            not tokens
            or any(t in eos for t in tokens[:-1])
            or len(tokens) > row["max_new_tokens"]
            or (tokens[-1] not in eos and len(tokens) != row["max_new_tokens"])
        ):
            raise ValueError(
                "Known-action mass requires mutually exclusive complete finite-horizon actions"
            )
    value = sum(math.exp(sum(_logps(r["token_logprobs"], len(r["token_ids"])))) for r in scores)
    if value > 1 + 1e-12:
        raise ValueError("Known-action mass exceeds one")
    return {
        "known_X_action_mass": value,
        "enumerated_actions": len(scores),
        "event_probability_exact": False,
        "event_lower_bound": value,
    }


def signed_contributions(
    log_a,
    log_b,
    log_proposal,
    categories,
    *,
    proposal,
    origin_support_certified=False,
    origin_max_abs_weight=50.0,
):
    """Return unmodified raw four-event contributions and explicit tail alarms."""
    import numpy as np

    a, b, q = (np.asarray(v, dtype=float) for v in (log_a, log_b, log_proposal))
    if (
        a.ndim != 1
        or a.shape != b.shape
        or a.shape != q.shape
        or len(a) != len(categories)
        or not len(a)
    ):
        raise ValueError("Likelihood and category dimensions differ")
    if (
        any(c not in tuple(EVENTS) for c in categories)
        or np.any(a > 0)
        or np.any(b > 0)
        or np.any(q > 0)
    ):
        raise ValueError("Invalid category or action log probability")
    if proposal == "MIX":
        expected = np.logaddexp(a, b) - math.log(2)
        if not np.allclose(expected, q, rtol=0, atol=1e-10):
            raise ValueError("MIX denominator must be the equal endpoint mixture")
        if np.any(~np.isfinite(q)) or np.any(np.isnan(a)) or np.any(np.isnan(b)):
            raise ValueError("MIX sampled action has invalid endpoint support")
        w = 2 * np.tanh((b - a) / 2)
        bound, reasons = 2.0, []
    elif proposal == "ORIGIN":
        # Subtract in log space. Equal policies have exactly zero contrast,
        # even when both importance ratios individually overflow.
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            high, low = np.maximum(a, b), np.minimum(a, b)
            nonzero = a != b
            w = np.zeros_like(a)
            w[nonzero] = np.sign(b[nonzero] - a[nonzero]) * np.exp(
                high[nonzero] - q[nonzero] + np.log(-np.expm1(low[nonzero] - high[nonzero]))
            )
            w[~np.isfinite(q) | np.isnan(a) | np.isnan(b)] = np.nan
        bound, reasons = None, []
        if not origin_support_certified:
            reasons.append("ORIGIN_SUPPORT_UNCERTIFIED")
        if np.any(~np.isfinite(w)):
            reasons.append("NONFINITE_ORIGIN_WEIGHT")
        if not math.isfinite(origin_max_abs_weight) or origin_max_abs_weight <= 0:
            raise ValueError("Origin tail threshold must be finite, positive and frozen")
        if np.any(np.abs(w) > origin_max_abs_weight) or np.any(
            np.maximum(a, b) - q > math.log(origin_max_abs_weight)
        ):
            reasons.append("EXTREME_ORIGIN_WEIGHT")
    else:
        raise ValueError("Contribution proposal must be ORIGIN or MIX")
    raw = np.zeros((len(w), 4), dtype=float)
    raw[np.arange(len(w)), [EVENTS.index(c) for c in categories]] = w
    finite = w[np.isfinite(w)]
    diagnostics = {
        "proposal": proposal,
        "n": len(w),
        "absolute_weight_bound": bound,
        "max_abs_weight": float(np.max(np.abs(finite))) if len(finite) else None,
        "nonfinite_weights": int(np.sum(~np.isfinite(w))),
        "origin_max_abs_weight_threshold": origin_max_abs_weight,
        "requires_independent_mix": bool(reasons),
        "reasons": reasons,
        "weights_clipped": False,
    }
    if len(finite):
        diagnostics["absolute_weight_quantiles"] = {
            str(p): float(np.quantile(np.abs(finite), p)) for p in (0.5, 0.9, 0.99, 0.999)
        }
    if proposal == "ORIGIN":
        diagnostics["endpoint_effective_sample_sizes"] = []
        for endpoint in (a, b):
            ratios = endpoint - q
            if np.all(np.isfinite(ratios)):
                scaled = np.exp(ratios - np.max(ratios))
                diagnostics["endpoint_effective_sample_sizes"].append(
                    float(np.sum(scaled) ** 2 / np.sum(scaled**2))
                )
            else:
                diagnostics["endpoint_effective_sample_sizes"].append(None)
    return raw, diagnostics


def reference_precision(
    raw, *, cumulative_draws, protocol, proposal, diagnostics=None, crosscheck_consistent=True
):
    """Bonferroni across frozen looks and cells; normal SE remains empirical.

    The bounded MIX interval uses Hoeffding on [-2,2]. ORIGIN has no automatic
    finite range bound. Neither an empirical interval nor a noisy mean is truth.
    """
    import numpy as np

    looks = tuple(protocol.get("looks", REFERENCE_LOOKS))
    if looks != REFERENCE_LOOKS:
        raise ValueError("V3 reference cumulative looks must be frozen at 4096..65536")
    if cumulative_draws not in looks:
        raise ValueError("Reference precision checked outside a predeclared look")
    cells, alpha = protocol.get("family_cells"), protocol.get("alpha", 0.05)
    if type(cells) is not int or cells < 1 or not 0 < alpha < 1:
        raise ValueError("Predeclare the entire reference family cell count and alpha")
    x = np.asarray(raw, dtype=float)
    if x.shape != (cumulative_draws, 4):
        raise ValueError("Reference draw count differs from contribution ledger")
    target, fallback = (
        protocol.get("half_width_goal", 0.00025),
        protocol.get("fallback_half_width", 0.001),
    )
    if not 0 < target <= fallback:
        raise ValueError("Reference precision targets are invalid")
    reason = list((diagnostics or {}).get("reasons", []))
    if not crosscheck_consistent:
        reason.append("ORIGIN_MIX_CROSSCHECK_INCONSISTENT")
    finite_values = bool(np.all(np.isfinite(x)))
    with np.errstate(over="ignore", invalid="ignore"):
        finite = finite_values and bool(
            np.isfinite(x.mean(axis=0)).all() and np.isfinite(x.var(axis=0, ddof=1)).all()
        )
    if not finite:
        reason.append("NONFINITE_REFERENCE_MOMENTS" if finite_values else "NONFINITE_REFERENCE")
    delta = alpha / (len(looks) * cells * 4)
    mean = np.mean(x, axis=0) if finite else None
    se = np.std(x, axis=0, ddof=1) / math.sqrt(len(x)) if finite else None
    z = statistics.NormalDist().inv_cdf(1 - delta / 2)
    empirical = z * se if finite else None
    if proposal not in {"ORIGIN", "MIX"}:
        raise ValueError("Reference proposal must be ORIGIN or MIX")
    if proposal == "MIX" and finite and np.max(np.abs(x)) > 2 + 1e-12:
        raise ValueError("MIX contributions exceed their formal bound")
    formal = 4 * math.sqrt(math.log(2 / delta) / (2 * len(x))) if proposal == "MIX" else None
    degenerate = np.var(x, axis=0, ddof=1) == 0 if finite else np.ones(4, dtype=bool)
    # A normal interval of width zero after observing no variation is not an
    # uncertainty certificate. Require independent finite-range precision there.
    per_event_width = (
        np.where(degenerate, formal if formal is not None else float("inf"), empirical)
        if finite
        else None
    )
    good = finite and not reason and bool(np.max(per_event_width) <= target)
    coarse = finite and not reason and bool(np.max(per_event_width) <= fallback)
    index = looks.index(cumulative_draws)
    status = (
        "REFERENCE_EMPIRICAL_PRECISION_MET"
        if good
        else "REFERENCE_UNRESOLVED"
        if index == len(looks) - 1
        else "REFERENCE_EXTEND"
    )
    if reason:
        status = (
            "REFERENCE_REQUIRES_INDEPENDENT_MIX" if proposal == "ORIGIN" else "REFERENCE_UNRESOLVED"
        )
    return {
        "status": status,
        "n": len(x),
        "proposal": proposal,
        "mean": mean.tolist() if finite else None,
        "standard_error": se.tolist() if finite else None,
        "empirical_normal_half_width": empirical.tolist() if finite else None,
        "formal_hoeffding_half_width": formal,
        "formal_precision_met": formal is not None and formal <= target,
        "empirical_precision_met": good,
        "fallback_scale_empirically_resolved": coarse,
        "zero_variance_event_indices": np.flatnonzero(degenerate).tolist(),
        "zero_variance_is_not_zero_uncertainty": True,
        "next_cumulative_draws": looks[index + 1] if status == "REFERENCE_EXTEND" else None,
        "multi_look_control": "Bonferroni over prespecified looks, cells, and four events",
        "alpha_per_interval": delta,
        "protocol_hash": digest(protocol),
        "look_index": index + 1,
        "stopping_uses_predictor_rankings": False,
        "reference_is_exact_truth": False,
        "finite_reference_values": finite_values,
        "finite_reference_moments": finite,
        "reasons": reason,
        "uncertainty_scope": "Empirical normal intervals; bounded intervals reported separately",
    }


def freeze_observation_task(
    *,
    operation,
    origin_id,
    candidate_id,
    prompt_file,
    prompt_ids,
    role,
    draw_start,
    draw_stop,
    rng_namespace,
    proposal,
    proposal_candidates,
    sample_files=None,
    max_new_tokens=64,
    worker=None,
):
    """Build a complete immutable task; score tasks bind exact generated shards.

    Generation is shared by candidates through sample_files, not relabeled as
    independent samples. Every scoring task explicitly pays its request count.
    """
    task = {
        "operation": operation,
        "origin_id": origin_id,
        "candidate_id": candidate_id,
        "prompt_file": prompt_file,
        "prompt_ids": list(prompt_ids),
        "role": role,
        "draw_start": draw_start,
        "draw_stop": draw_stop,
        "rng_namespace": rng_namespace,
        "proposal": proposal,
        "proposal_candidates": list(proposal_candidates),
        "sample_files": list(sample_files or []),
        "max_new_tokens": max_new_tokens,
    }
    if worker is not None:
        task["worker"] = worker
    task["request_keys"] = _expected_request_keys(task)
    task["task_id"] = digest(task)
    task["output_path"] = "tasks/" + task["task_id"]
    return task


def _expected_request_keys(task):
    operation = task["operation"]
    if operation not in {"generate", "score"}:
        raise ValueError("Observation task operation must be generate or score")
    start, stop = task["draw_start"], task["draw_stop"]
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError("Observation draw interval must be nonempty, integer and nonnegative")
    ids = task["prompt_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Observation prompt ids must be nonempty and unique")
    keys = []
    for prompt_id in ids:
        for index in range(start, stop):
            request = request_identity(
                origin_id=task["origin_id"],
                prompt_id=prompt_id,
                role=task["role"],
                draw_index=index,
                rng_namespace=task["rng_namespace"],
                proposal=task["proposal"],
                proposal_candidates=task["proposal_candidates"],
            )
            keys.append(
                request["sample_key"]
                if operation == "generate"
                else digest(["score", task["candidate_id"], request["sample_key"]])
            )
    return keys


def validate_task_manifest(manifest, *, workers, verify_files=True):
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "V3_OBSERVATION_TASKS":
        raise ValueError("Unsupported observation task manifest")
    if type(workers) is not int or workers not in (1, 2) or manifest.get("workers") != workers:
        raise ValueError("Frozen worker count must be one or two")
    if not manifest.get("runtime_identity") or not manifest.get("policies"):
        raise ValueError("Manifest needs bound runtime and candidate policies")
    expected_hash = digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if manifest.get("manifest_hash") != expected_hash:
        raise ValueError("Task manifest hash differs from frozen contents")
    seen_tasks, generated_keys, scored_keys = set(), set(), set()
    policies = manifest["policies"]
    for key, policy in policies.items():
        if not policy.get("inference_fingerprint"):
            raise ValueError("Candidate requires an inference fingerprint: " + key)
        if verify_files:
            _read_binding(policy["checkpoint"])
    if not manifest.get("tasks"):
        raise ValueError("Task manifest is empty")
    fingerprint_workers = {}
    for task in manifest["tasks"]:
        body = {k: v for k, v in task.items() if k not in {"task_id", "output_path"}}
        if task.get("task_id") != digest(body) or task["task_id"] in seen_tasks:
            raise ValueError("Task identity changed or duplicate task")
        seen_tasks.add(task["task_id"])
        relative = PurePosixPath(task["output_path"])
        if relative.as_posix() != "tasks/" + task["task_id"]:
            raise ValueError("Worker output must be task-owned under its hash")
        if "worker" in task and (
            type(task["worker"]) is not int or not 0 <= task["worker"] < workers
        ):
            raise ValueError("Frozen task worker is out of range")
        if task["candidate_id"] not in policies or any(
            p not in policies for p in task["proposal_candidates"]
        ):
            raise ValueError("Unknown candidate or proposal source")
        if type(task["max_new_tokens"]) is not int or task["max_new_tokens"] != 64:
            raise ValueError("V3 production actions require the frozen 64-token horizon")
        expected = _expected_request_keys(task)
        if task.get("request_keys") != expected:
            raise ValueError("Task request keys differ from role/prompt/draw identity")
        seen = generated_keys if task["operation"] == "generate" else scored_keys
        if seen.intersection(expected):
            raise ValueError("Duplicate generation or score request; record an alias instead")
        seen.update(expected)
        if task["operation"] == "score" and not task["sample_files"]:
            raise ValueError("Score task requires hash-bound original sample shards")
        if task["operation"] == "generate" and task["sample_files"]:
            raise ValueError("Generation task may not substitute existing sample shards")
        if task["operation"] == "score" and task["role"] != "null":
            fp = policies[task["candidate_id"]]["inference_fingerprint"]
            owner = _worker_assignment(task, policies, workers)
            if fp in fingerprint_workers and fingerprint_workers[fp] != owner:
                raise ValueError("Inference aliases must share one score-cache worker")
            fingerprint_workers[fp] = owner
        if verify_files:
            _read_binding(task["prompt_file"])
            for binding in task["sample_files"]:
                _read_binding(binding)
    return {
        "manifest_hash": expected_hash,
        "tasks": len(seen_tasks),
        "generation_requests": len(generated_keys),
        "scoring_requests": len(scored_keys),
    }


def freeze_task_manifest(tasks, policies, runtime_identity, *, workers=2):
    result = {
        "schema_version": 1,
        "kind": "V3_OBSERVATION_TASKS",
        "workers": workers,
        "runtime_identity": runtime_identity,
        "policies": policies,
        "tasks": tasks,
    }
    result["manifest_hash"] = digest(result)
    validate_task_manifest(result, workers=workers)
    return result


def _task_requests(task):
    for prompt_id in task["prompt_ids"]:
        for index in range(task["draw_start"], task["draw_stop"]):
            sample = request_identity(
                origin_id=task["origin_id"],
                prompt_id=prompt_id,
                role=task["role"],
                draw_index=index,
                rng_namespace=task["rng_namespace"],
                proposal=task["proposal"],
                proposal_candidates=task["proposal_candidates"],
            )
            key = (
                sample["sample_key"]
                if task["operation"] == "generate"
                else digest(["score", task["candidate_id"], sample["sample_key"]])
            )
            yield {
                **sample,
                "request_key": key,
                "prompt_id": prompt_id,
                "draw_index": index,
                "origin_id": task["origin_id"],
                "role": task["role"],
                "proposal": task["proposal"],
                "rng_namespace": task["rng_namespace"],
                "proposal_candidates": task["proposal_candidates"],
            }


def _read_rows(path, *, allow_partial=False):
    rows = []
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not allow_partial:
            raise ValueError("Partial JSON line in original shard: " + str(path))
        # The interrupted bytes remain untouched. Resume regenerates only the
        # uncommitted request in a new attempt using its original per-draw seed.
        raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    for line in raw.splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("Sample shard rows must be objects")
        expected = value.pop("record_hash", None)
        if expected != digest(value):
            raise ValueError("Original sample row hash changed")
        rows.append(value)
    return rows


def _load_task_inputs(task):
    document = json.loads(_read_binding(task["prompt_file"]).read_text())
    prompt_rows = document.get("prompts") if isinstance(document, dict) else document
    prompts = {p["prompt_id"]: p for p in prompt_rows}
    if len(prompts) != len(prompt_rows) or not set(task["prompt_ids"]) <= set(prompts):
        raise ValueError("Missing or duplicated prompt identity in frozen input")
    samples = {}
    for binding in task["sample_files"]:
        for row in _read_rows(_read_binding(binding), allow_partial=True):
            key = row["sample_key"]
            if key in samples:
                raise ValueError("Original generation sample appears in multiple shards")
            samples[key] = row
    return prompts, samples


def _resume_task(task_root, task, run_identity, *, resume):
    if task_root.exists() and any(task_root.iterdir()) and not resume:
        raise ValueError("Task already has evidence; explicit resume required")
    task_root.mkdir(parents=True, exist_ok=True)
    identity_path = task_root / "identity.json"
    expected = {"task": task, "run_identity": run_identity}
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != expected:
            raise ValueError("Resume code/config/model/data/task identity drift")
    else:
        _write_new(identity_path, expected)
    rows, shards = {}, []
    for path in sorted(task_root.glob("attempt_*/samples.jsonl")):
        shard = _read_rows(path, allow_partial=True)
        for row in shard:
            key = row["request_key"]
            if key not in task["request_keys"] or key in rows:
                raise ValueError("Unexpected or duplicate request in resume shards")
            if row["task_id"] != task["task_id"] or row["run_identity_hash"] != digest(
                run_identity
            ):
                raise ValueError("Resume row task/source identity drift")
            rows[key] = row
        shards.append(
            {
                "path": str(path.relative_to(task_root)),
                "sha256": file_digest(path),
                "records": len(shard),
            }
        )
    completed = task_root / "COMPLETE.json"
    if completed.exists():
        receipt = json.loads(completed.read_text())
        if set(rows) != set(task["request_keys"]) or receipt.get("shards") != shards:
            raise ValueError("Completed shard bytes or request set changed")
        if receipt.get("run_identity_hash") != digest(run_identity):
            raise ValueError("Completed task runtime identity differs")
    return rows, shards, completed.exists()


def _worker_assignment(task, policies, workers):
    if "worker" in task:
        return task["worker"]
    if task["operation"] == "score" and task["role"] != "null":
        return (
            int(digest(policies[task["candidate_id"]]["inference_fingerprint"])[:16], 16) % workers
        )
    return int(task["task_id"][:16], 16) % workers


def _physical_score_key(sample, fingerprint):
    return digest(
        {
            "fingerprint": fingerprint,
            **{
                k: sample[k]
                for k in (
                    "runtime_identity",
                    "prompt_record_hash",
                    "input_audit",
                    "token_ids",
                    "probability_execution",
                    "max_new_tokens",
                    "eos_token_ids",
                )
            },
        }
    )


def _cache_score(cache, row, artifact):
    if row.get("score_execution") == "INFERENCE_ALIAS":
        return
    values = row.get("token_logprobs", row.get("rescored_token_logprobs"))
    if values is None:
        return
    values = _logps(values, len(row["token_ids"]))
    key = _physical_score_key(row, row["inference_fingerprint"])
    cache.setdefault(key, {"values": values, "artifact": artifact})


def _alias_score(sample, fingerprint, cached):
    return {
        "proposal_sample_key": sample["sample_key"],
        "sample_record_hash": digest(sample),
        **{
            k: sample[k]
            for k in (
                "prompt_id",
                "prompt_record_hash",
                "input_audit",
                "token_ids",
                "category",
                "probability_execution",
                "max_new_tokens",
                "eos_token_ids",
                "runtime_identity",
            )
        },
        "token_logprobs": cached["values"],
        "sequence_logp": sum(cached["values"]),
        "inference_fingerprint": fingerprint,
        "score_execution": "INFERENCE_ALIAS",
        "score_alias_of": cached["artifact"],
        "score_is_independent_draw": False,
        "training_resume_alias": False,
    }


def execute_observation_tasks(
    task_manifest, backend, *, out, worker_index, workers=2, resume=False
):
    """Execute only one frozen worker's tasks, grouping candidate weights.

    No sbatch or model loading happens here. The caller supplies the approved
    production backend. A failed attempt stays on disk; resume writes a new one.
    """
    import fcntl

    manifest = (
        json.loads(Path(task_manifest).read_text())
        if isinstance(task_manifest, (str, Path))
        else task_manifest
    )
    validate_task_manifest(manifest, workers=workers)
    if type(worker_index) is not int or not 0 <= worker_index < workers:
        raise ValueError("Worker index outside frozen assignment")
    if backend.runtime_identity != manifest["runtime_identity"]:
        raise ValueError("Loaded backend differs from frozen runtime identity")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    worker_root = out / f"worker_{worker_index}"
    worker_root.mkdir(exist_ok=True)
    lock_path = worker_root / "writer.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Worker already has an active writer") from exc
        identity = {
            "manifest_hash": manifest["manifest_hash"],
            "runtime_identity": backend.runtime_identity,
            "worker_index": worker_index,
            "workers": workers,
            "observer_source_sha256": file_digest(__file__),
        }
        if getattr(backend, "execution_stage_binding", None) is not None:
            _read_binding(backend.execution_stage_binding)
            identity["execution_stage_binding"] = backend.execution_stage_binding
        tasks = [
            t
            for t in manifest["tasks"]
            if _worker_assignment(t, manifest["policies"], workers) == worker_index
        ]
        tasks.sort(
            key=lambda t: (
                manifest["policies"][t["candidate_id"]]["inference_fingerprint"],
                t["candidate_id"],
                t["operation"],
                t["task_id"],
            )
        )
        score_cache = {}
        for prior in sorted(worker_root.glob("tasks/*/attempt_*/samples.jsonl")):
            for row in _read_rows(prior, allow_partial=True):
                if row.get("run_identity_hash") != digest(identity):
                    raise ValueError(
                        "Score-cache original belongs to another runtime/task manifest"
                    )
                _cache_score(
                    score_cache,
                    row,
                    {
                        "path": str(prior.relative_to(out)),
                        "request_key": row["request_key"],
                        "record_hash": digest(row),
                    },
                )
        completed_tasks = []
        for task in tasks:
            prompts, samples = _load_task_inputs(task)
            root = worker_root / task["output_path"]
            existing, shards, complete = _resume_task(root, task, identity, resume=resume)
            if complete:
                completed_tasks.append(task["task_id"])
                continue
            requests = [r for r in _task_requests(task) if r["request_key"] not in existing]
            if task["operation"] == "generate":
                requests.sort(key=lambda r: (r["proposal_source"], r["prompt_id"], r["draw_index"]))
            else:
                # All requests for a candidate stay together, including across tasks.
                requests.sort(key=lambda r: (r["prompt_id"], r["draw_index"]))
            attempt = root / f"attempt_{len(list(root.glob('attempt_*'))):05d}"
            attempt.mkdir()
            before = dict(backend.counters)
            _write_new(
                attempt / "REQUESTS.json",
                {
                    "task_id": task["task_id"],
                    "request_keys": [r["request_key"] for r in requests],
                    "run_identity": identity,
                    "counters_before": before,
                },
            )
            sample_path = attempt / "samples.jsonl"
            try:
                if task["operation"] == "score" and requests:
                    # Verify the bound checkpoint really restores the claimed
                    # inference fingerprint even if every score can be aliased.
                    backend.activate(manifest["policies"][task["candidate_id"]])
                with sample_path.open("x", encoding="utf-8") as stream:
                    for request in requests:
                        candidate = (
                            request["proposal_source"]
                            if task["operation"] == "generate"
                            else task["candidate_id"]
                        )
                        if task["operation"] == "generate":
                            backend.activate(manifest["policies"][candidate])
                            measured = backend.generate(
                                prompts[request["prompt_id"]],
                                seed=request["sample_seed"],
                                max_new_tokens=task["max_new_tokens"],
                            )
                        else:
                            source = samples.get(request["sample_key"])
                            if source is None:
                                raise ValueError("Bound original shards omit a requested sample")
                            for key in (
                                "role",
                                "proposal",
                                "prompt_id",
                                "draw_index",
                                "sample_rng_key",
                                "proposal_source",
                                "rng_namespace",
                            ):
                                if source.get(key) != request[key]:
                                    raise ValueError(
                                        "Score request differs from original sampling identity: "
                                        + key
                                    )
                            _cache_score(
                                score_cache,
                                source,
                                {
                                    "kind": "bound_original_generation",
                                    "files": task["sample_files"],
                                    "request_key": source["request_key"],
                                    "record_hash": digest(source),
                                },
                            )
                            fingerprint = manifest["policies"][candidate]["inference_fingerprint"]
                            cached = score_cache.get(_physical_score_key(source, fingerprint))
                            if cached is not None and task["role"] != "null":
                                measured = _alias_score(source, fingerprint, cached)
                                backend.counters["score_alias_reuses"] += 1
                            else:
                                backend.activate(manifest["policies"][candidate])
                                measured = backend.score(prompts[request["prompt_id"]], source)
                        row = {
                            **measured,
                            **request,
                            "candidate_id": candidate,
                            "task_id": task["task_id"],
                            "run_identity_hash": digest(identity),
                            "proposal_policy_fingerprints": [
                                manifest["policies"][name]["inference_fingerprint"]
                                for name in task["proposal_candidates"]
                            ],
                        }
                        row["record_hash"] = digest(row)
                        stream.write(
                            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                        _cache_score(
                            score_cache,
                            {k: val for k, val in row.items() if k != "record_hash"},
                            {
                                "path": str(sample_path.relative_to(out)),
                                "request_key": request["request_key"],
                                "record_hash": row["record_hash"],
                            },
                        )
                cost = {k: backend.counters[k] - before[k] for k in before}
                _write_new(
                    attempt / "COST.json", {"delta": cost, "counters_after": backend.counters}
                )
            except BaseException as exc:
                if isinstance(exc, ObservationFault):
                    _write_new(attempt / "RETURNED_FAULT.json", _fault_json(exc.raw_result))
                _write_new(
                    attempt / "FAILED.json",
                    {
                        "type": type(exc).__name__,
                        "reason": str(exc),
                        "counters_before": before,
                        "counters_after": backend.counters,
                    },
                )
                raise
            existing, shards, _ = _resume_task(root, task, identity, resume=True)
            if set(existing) != set(task["request_keys"]):
                raise ValueError("Cannot complete a task with missing request keys")
            _publish(
                root / "COMPLETE.json",
                {
                    "status": "COMPLETE",
                    "task_id": task["task_id"],
                    "run_identity_hash": digest(identity),
                    "expected_requests": len(task["request_keys"]),
                    "request_keys_hash": digest(sorted(existing)),
                    "shards": shards,
                },
            )
            completed_tasks.append(task["task_id"])
        return {
            "status": "COMPLETE",
            "worker_index": worker_index,
            "workers": workers,
            "manifest_hash": manifest["manifest_hash"],
            "completed_tasks": completed_tasks,
            "execution_counters_this_process": backend.counters,
            "scientific_measurement_qualified": False,
        }


def validate_role_independence(*, pilot, main, reference):
    """Shared candidate scoring is allowed; reuse across independent roles is not."""
    role_rows = {"pilot": pilot, "main": main, "reference": reference}
    sets = {}
    for role, rows in role_rows.items():
        for row in rows:
            if row.get("role") != role:
                raise ValueError("Observation role differs from its declared partition")
        for field in ("sample_key", "sample_rng_key", "sample_seed"):
            values = [r[field] for r in rows]
            if len(values) != len(set(values)):
                raise ValueError("Duplicate draws inside an independent observation role")
            sets[role, field] = set(values)
    for a, b in (("pilot", "main"), ("pilot", "reference"), ("main", "reference")):
        for field in ("sample_key", "sample_rng_key", "sample_seed"):
            if sets[a, field] & sets[b, field]:
                raise ValueError("Pilot/main/reference sample or RNG overlap")
    return {
        "status": "PASS",
        "counts": {role: len(rows) for role, rows in role_rows.items()},
        "same_actions_by_chance_are_not_aliases": True,
    }


def packet_contributions(
    samples,
    scores_a,
    scores_b,
    *,
    proposal,
    origin_support_certified=False,
    origin_max_abs_weight=50.0,
):
    """Join by original action identity, never by row order or decoded strings."""
    pairs = []
    for scores in (scores_a, scores_b):
        lookup = {r["proposal_sample_key"]: r for r in scores}
        if len(lookup) != len(scores):
            raise ValueError("A candidate contains duplicate score requests")
        pairs.append(lookup)
    sample_keys = [r["sample_key"] for r in samples]
    if len(set(sample_keys)) != len(samples) or any(set(sample_keys) != set(p) for p in pairs):
        raise ValueError("Candidate score requests do not equal the proposal packet")
    if (
        not samples
        or len({r["prompt_record_hash"] for r in samples}) != 1
        or len({r["origin_id"] for r in samples}) != 1
        or len({digest(r["input_audit"]) for r in samples}) != 1
    ):
        raise ValueError("Each response packet must contain one prompt/input and one origin")
    if proposal == "ORIGIN" and len({r["inference_fingerprint"] for r in samples}) != 1:
        raise ValueError("ORIGIN proposal policy changes within a response packet")
    a, b, q, categories = [], [], [], []
    fingerprints = [set(), set()]
    for sample in samples:
        behavior = _logps(sample["behavior_token_logprobs"], len(sample["token_ids"]))
        if abs(sum(behavior) - sample["generation_sequence_logp"]) > 1e-10:
            raise ValueError(
                "Proposal sequence probability differs from original token probabilities"
            )
        row_a, row_b = (p[sample["sample_key"]] for p in pairs)
        for i, scored in enumerate((row_a, row_b)):
            for field in (
                "token_ids",
                "prompt_record_hash",
                "input_audit",
                "category",
                "probability_execution",
                "max_new_tokens",
                "eos_token_ids",
                "runtime_identity",
            ):
                if scored[field] != sample[field]:
                    raise ValueError("Candidate scoring token/input/event identity differs")
            if scored["sample_record_hash"] != digest(sample):
                raise ValueError("Score does not bind the exact original completion record")
            values = _logps(scored["token_logprobs"], len(sample["token_ids"]))
            if abs(sum(values) - scored["sequence_logp"]) > 1e-10:
                raise ValueError("Sequence logp does not equal the token sum")
            fingerprints[i].add(scored["inference_fingerprint"])
        a.append(row_a["sequence_logp"])
        b.append(row_b["sequence_logp"])
        if proposal == "MIX":
            high, low = max(a[-1], b[-1]), min(a[-1], b[-1])
            q.append(high + math.log1p(math.exp(low - high)) - math.log(2))
            if sample["proposal"] != "MIX" or sample["source_coin_key"] is None:
                raise ValueError("MIX observations require saved iid source coins")
            source_fingerprints = sample.get("proposal_policy_fingerprints", [])
            if len(source_fingerprints) != 2 or sorted(source_fingerprints) != sorted(
                [row_a["inference_fingerprint"], row_b["inference_fingerprint"]]
            ):
                raise ValueError("Scored endpoints are not the actual MIX generation policies")
            coin = sample["proposal_source_index"]
            if (
                type(coin) is not int
                or coin not in (0, 1)
                or source_fingerprints[coin] != sample["inference_fingerprint"]
                or sample["proposal_candidates"][coin] != sample["proposal_source"]
            ):
                raise ValueError("MIX source coin does not identify its actual generating policy")
        else:
            q.append(sample["generation_sequence_logp"])
            if sample["proposal"] != "ORIGIN":
                raise ValueError("ORIGIN denominator must come from the sampled origin policy")
        categories.append(sample["category"])
    if any(len(group) != 1 for group in fingerprints):
        raise ValueError("A contrast endpoint changes policy inside one packet")
    return signed_contributions(
        a,
        b,
        q,
        categories,
        proposal=proposal,
        origin_support_certified=origin_support_certified,
        origin_max_abs_weight=origin_max_abs_weight,
    )


def reference_packet_report(
    samples,
    scores_a,
    scores_b,
    *,
    protocol,
    predictor_samples=(),
    proposal,
    origin_support_certified=False,
    origin_max_abs_weight=50.0,
    crosscheck_consistent=True,
):
    """Evaluator-only reference entrypoint with explicit predictor-data separation."""
    if not samples or any(r.get("role") != "reference" for r in samples):
        raise ValueError("Reference evaluator accepts only independent reference role samples")
    validate_role_independence(pilot=[], main=[], reference=samples)
    reference_ids = {r["sample_key"] for r in samples}
    reference_rng = {r["sample_rng_key"] for r in samples}
    reference_seeds = {r["sample_seed"] for r in samples}
    if any(
        r["sample_key"] in reference_ids
        or r["sample_rng_key"] in reference_rng
        or r["sample_seed"] in reference_seeds
        for r in predictor_samples
    ):
        raise ValueError("Reference data overlaps predictor/pilot/main observations")
    raw, diagnostics = packet_contributions(
        samples,
        scores_a,
        scores_b,
        proposal=proposal,
        origin_support_certified=origin_support_certified,
        origin_max_abs_weight=origin_max_abs_weight,
    )
    report = reference_precision(
        raw,
        cumulative_draws=len(samples),
        protocol=protocol,
        proposal=proposal,
        diagnostics=diagnostics,
        crosscheck_consistent=crosscheck_consistent,
    )
    return {
        **report,
        "tail_diagnostics": diagnostics,
        "reference_packet_hash": digest(samples),
        "endpoint_score_hashes": [digest(scores_a), digest(scores_b)],
        "independent_reference_used_for_all_compared_methods": True,
    }


def _validate_observation_stage(config, bindings, manifest):
    """Bind engineering evidence and this exact task set before touching weights."""
    from .io import canonical_hash, source_identity

    stage = json.loads(_read_binding(bindings["v3_stage_lock"]).read_text())
    code_hash = source_identity()["sha256"]
    if stage.get("config_hash") != canonical_hash(config) or stage.get("source_hash") != code_hash:
        raise ValueError("Frozen V3 observation stage config/source changed")
    if "observe-vlm" not in stage.get("operations", []):
        raise PermissionError("Observation operation is outside the frozen stage list")
    if manifest["manifest_hash"] not in stage.get("observation_manifest_hashes", []):
        raise PermissionError("Observation task manifest is outside the frozen stage list")
    receipts = stage.get("technical_receipts", {})
    for phase in ("Q1", "Q2"):
        if phase not in receipts:
            raise ValueError("Q1/Q2 technical receipts with hashes are required")
        receipt = json.loads(_read_binding(receipts[phase]).read_text())
        if (
            receipt.get("status") != "PASS"
            or receipt.get("config_hash") != canonical_hash(config)
            or receipt.get("source_hash") != code_hash
            or not receipt.get("checks")
            or not all(item.get("passed") is True for item in receipt["checks"])
        ):
            raise ValueError("Q1/Q2 engineering receipt missing measured passing checks")
    if stage.get("phase") not in {"Q4", "Q5", "Q6"}:
        raise ValueError("Observation stage must explicitly declare Q4, Q5 or Q6")
    if stage["phase"] in {"Q5", "Q6"}:
        from .vlm_campaign import _stage_gate

        _stage_gate(config, bindings, stage["runtime_seed"], operation="observe-vlm")
        smoke = json.loads(_read_binding(stage["v3_gpu_smoke"]).read_text())
        if (
            smoke.get("status") != "PASS"
            or smoke.get("execution_kind") != "REAL_CUDA_MODEL"
            or smoke.get("two_gpu_null_parity_passed") is not True
            or smoke.get("source_hash") != code_hash
            or smoke.get("config_hash") != canonical_hash(config)
        ):
            raise ValueError("Q5/Q6 requires the completed V3 real bridge and two-GPU parity")
        if stage["phase"] == "Q6":
            phase = stage.get("q6_phase")
            if phase is None:
                if stage.get("observation_purpose") != "measurement" or not stage.get(
                    "origin_id", ""
                ).endswith("_X_BASE_64"):
                    raise ValueError(
                        "Q6 calibration observation requires the explicit step-64 measurement stage"
                    )
                if any(task.get("role") == "reference" for task in manifest["tasks"]):
                    raise ValueError("Q6 calibration cannot open dense reference observations")
            elif phase not in {"anchor", "reference"}:
                raise ValueError("Q6 absolute observation phase must be anchor or reference")
            else:
                from .q6_observation import verify_q6_observation_stage

                verify_q6_observation_stage(config, bindings, stage, manifest)
    return stage


def observe_vlm(
    config,
    bindings,
    *,
    task_manifest,
    worker_index,
    workers=2,
    device="cuda:0",
    out,
    resume=False,
    allow_gpu=False,
    acknowledge_new_experiment=False,
):
    """CLI production entrypoint. Permission and frozen evidence precede model imports."""
    if not allow_gpu or not acknowledge_new_experiment:
        raise PermissionError("New V3 GPU observation requires explicit execution authorization")
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("Use the one Slurm-visible device for this independent worker")
    manifest = (
        json.loads(Path(task_manifest).read_text())
        if isinstance(task_manifest, (str, Path))
        else task_manifest
    )
    validate_task_manifest(manifest, workers=workers)
    stage = _validate_observation_stage(config, bindings, manifest)
    from .vlm_campaign import load_runtime

    runtime = load_runtime(
        config,
        bindings,
        device=device,
        allow_gpu=allow_gpu,
        acknowledge_new_experiment=acknowledge_new_experiment,
        seed=stage.get("runtime_seed", 41001),
    )
    if not runtime.get("state_guard"):
        raise ValueError("Production observations require a parameter/Adam mutation guard")
    backend = PrefixObservationBackend(
        runtime["adapter"],
        runtime_identity=runtime["identity"],
        policy_loader=runtime["policy_loader"],
        data_root=runtime["data_root"],
        parity_tolerances=runtime["parity_tolerances"],
        state_guard=runtime["state_guard"],
    )
    backend.execution_stage_binding = dict(bindings["v3_stage_lock"])
    if runtime.get("execution_stage_binding") != backend.execution_stage_binding:
        raise ValueError("Loaded execution authorization differs from the checked stage")
    result = execute_observation_tasks(
        manifest, backend, out=out, worker_index=worker_index, workers=workers, resume=resume
    )
    _validate_observation_stage(config, bindings, manifest)
    receipt = {
        "status": "COMPLETE",
        "execution_stage_binding": backend.execution_stage_binding,
        "manifest_hash": manifest["manifest_hash"],
        "worker_index": worker_index,
        "workers": workers,
        "runtime_identity": backend.runtime_identity,
        "authorization_reverified_after_work": True,
        "completed_tasks": result["completed_tasks"],
        "observation_root": str(Path(out).resolve()),
        "execution_kind": "REAL_CUDA_MODEL",
        "physical_cost_accounting": "Original per-attempt COST.json and FAILED.json ledgers",
    }
    receipt_path = Path(out) / f"worker_{worker_index}" / "EXECUTION_RECEIPT.json"
    _publish_analysis(receipt_path, receipt)
    result["execution_kind"] = "REAL_CUDA_MODEL"
    result["execution_stage_binding"] = backend.execution_stage_binding
    result["receipt"] = {"path": str(receipt_path.resolve()), "sha256": file_digest(receipt_path)}
    return result


Q4_CONTRASTS = (
    ("joint_1_minus_joint_0", "joint_0", "joint_1"),
    ("no_x_off_1_minus_joint_0", "joint_0", "no_x_off_1"),
    ("joint_1_minus_no_x_off_1", "no_x_off_1", "joint_1"),
)


def iter_completed_task_rows(root, manifest, *, allowed_task_ids=None):
    """Read only complete, immutable tasks; audit every original shard first."""
    validate_task_manifest(manifest, workers=manifest["workers"])
    if allowed_task_ids is not None and not set(allowed_task_ids) <= {
        t["task_id"] for t in manifest["tasks"]
    }:
        raise ValueError("Requested label task is absent from the frozen manifest")
    for task in manifest["tasks"]:
        if allowed_task_ids is not None and task["task_id"] not in allowed_task_ids:
            continue
        worker = _worker_assignment(task, manifest["policies"], manifest["workers"])
        task_root = Path(root) / f"worker_{worker}" / task["output_path"]
        identity = {
            "manifest_hash": manifest["manifest_hash"],
            "runtime_identity": manifest["runtime_identity"],
            "worker_index": worker,
            "workers": manifest["workers"],
            "observer_source_sha256": file_digest(__file__),
        }
        saved_identity = json.loads((task_root / "identity.json").read_text())
        authorization = saved_identity.get("run_identity", {}).get("execution_stage_binding")
        if authorization is not None:
            _read_binding(authorization)
            identity["execution_stage_binding"] = authorization
        if saved_identity != {"task": task, "run_identity": identity}:
            raise ValueError("Q4 analysis task/code/runtime identity changed")
        marker = json.loads((task_root / "COMPLETE.json").read_text())
        if marker.get("status") != "COMPLETE" or marker.get("run_identity_hash") != digest(
            identity
        ):
            raise ValueError("Q4 completion belongs to another task/runtime")
        saved_paths = [row["path"] for row in marker["shards"]]
        actual_paths = [
            str(path.relative_to(task_root))
            for path in sorted(task_root.glob("attempt_*/samples.jsonl"))
        ]
        if saved_paths != actual_paths:
            raise ValueError("Q4 complete task shard list changed")
        task_rows = []
        for binding in marker["shards"]:
            path = task_root / binding["path"]
            if file_digest(path) != binding["sha256"]:
                raise ValueError("Q4 complete task original bytes changed")
            shard_rows = _read_rows(path, allow_partial=True)
            if len(shard_rows) != binding["records"]:
                raise ValueError("Q4 complete shard record count changed")
            task_rows.extend(shard_rows)
        expected = {r["request_key"]: r for r in _task_requests(task)}
        keys = [r["request_key"] for r in task_rows]
        if set(keys) != set(expected) or len(keys) != len(expected):
            raise ValueError("Q4 analysis needs the full expected request set exactly once")
        if marker.get("expected_requests") != len(keys) or marker.get(
            "request_keys_hash"
        ) != digest(sorted(keys)):
            raise ValueError("Q4 completion request count/hash differs from original draws")
        for row in task_rows:
            if row["task_id"] != task["task_id"] or row["run_identity_hash"] != digest(identity):
                raise ValueError("Q4 original row task identity changed")
            for key, value in expected[row["request_key"]].items():
                if row.get(key) != value:
                    raise ValueError("Q4 original request identity differs: " + key)
        yield from task_rows


def _completed_task_rows(root, manifest, *, allowed_task_ids=None):
    """Compatibility list API over the once-validated streaming original reader."""
    return list(iter_completed_task_rows(root, manifest, allowed_task_ids=allowed_task_ids))


def _q4_reference_diagnostic(raw, diagnostics, *, proposal, alpha):
    """64-draw bridge reference is a diagnostic, outside the Q5 precision looks."""
    import numpy as np

    x = np.asarray(raw, dtype=float)
    finite = bool(np.isfinite(x).all())
    n = len(x)
    with np.errstate(over="ignore", invalid="ignore"):
        finite_moments = finite and bool(np.isfinite(x * x).all())
        mean = x.mean(axis=0) if finite_moments else None
        covariance = (
            np.atleast_2d(np.cov(x, rowvar=False, ddof=1)) / n if finite_moments and n > 1 else None
        )
    return {
        "status": "REFERENCE_UNRESOLVED",
        "proposal": proposal,
        "n": n,
        "mean": None if mean is None else mean.tolist(),
        "covariance_of_mean": None if covariance is None else covariance.tolist(),
        "standard_error": None if covariance is None else np.sqrt(np.diag(covariance)).tolist(),
        "formal_hoeffding_half_width": 4 * math.sqrt(math.log(2 / alpha) / (2 * n))
        if proposal == "MIX"
        else None,
        "tail_diagnostics": diagnostics,
        "raw_mass_mean": None if mean is None else float(mean.sum()),
        "raw_mass_standard_error": float(x.sum(axis=1).std(ddof=1) / math.sqrt(n))
        if finite_moments and n > 1
        else None,
        "zero_variance_events": np.flatnonzero(x.var(axis=0) == 0).tolist()
        if finite_moments
        else [],
        "finite_raw_second_moments": finite_moments,
        "reference_is_exact_truth": False,
        "enough_for_method_ranking": False,
        "reason": "Q4 diagnostic is below the preregistered Q5 reference look schedule",
    }


def _estimate_document(estimate):
    return {
        "estimate_raw4": estimate.estimate.tolist(),
        "raw_estimate": estimate.raw_estimate.tolist(),
        "delta_pX": estimate.delta_pX.tolist(),
        "delta_v": estimate.delta_v.tolist(),
        "raw_mass_residual": estimate.mass_residual.tolist(),
        "b": estimate.b.tolist(),
        "covariance_of_mean": None
        if estimate.covariance_of_mean is None
        else estimate.covariance_of_mean.tolist(),
        "diagnostics": estimate.diagnostics,
        "n": estimate.n,
    }


def _publish_analysis(path, value):
    path = Path(path)
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise ValueError("Existing Q4 analysis differs from original-data recomputation")
    else:
        _publish(path, value)


def analyze_q4_worker(root, config):
    """Recompute Q4 geometry, direct counts and independent reference diagnostics.

    Each unit is one origin/bank/prompt with its three shared-packet contrasts.
    Reference residuals are retained as noisy comparisons and cannot rank methods.
    This CPU-only evaluator loads no model and never opens any locked-test store.
    """
    from collections import defaultdict

    import numpy as np

    from .covariance_pilot import cost_matched_estimates
    from .io import atomic_npz, canonical_hash, source_identity
    from .observation_geometry import ContributionBatch

    root = Path(root)
    generation = json.loads((root / "generation_tasks.json").read_text())
    scoring = json.loads((root / "scoring_tasks.json").read_text())
    if (
        generation["runtime_identity"] != scoring["runtime_identity"]
        or generation["policies"] != scoring["policies"]
    ):
        raise ValueError("Q4 generation/scoring model, source, policy or configuration differs")
    runtime, policies = generation["runtime_identity"], generation["policies"]
    if runtime.get("config_hash") != canonical_hash(config):
        raise ValueError("Q4 analysis config does not match executed observations")
    if runtime.get("source_hash") != source_identity()["sha256"]:
        raise ValueError("Q4 analysis source does not match executed observations")
    samples = _completed_task_rows(root / "generation", generation)
    scores = _completed_task_rows(root / "scoring", scoring)
    sample_lookup = {row["sample_key"]: row for row in samples}
    if len(sample_lookup) != len(samples) or len({row["origin_id"] for row in samples}) != 1:
        raise ValueError("Q4 worker needs unique generated draws from one origin")
    if "origin" not in policies:
        raise ValueError("Q4 requires explicitly scored ORIGIN baseline policy")
    probe_ids = sorted({row["prompt_id"] for row in samples})
    candidate_suffixes = ("joint_0", "joint_1", "no_x_off_1")
    bank_ids = sorted(
        {name.removesuffix("_joint_0") for name in policies if name.endswith("_joint_0")}
    )
    expected_policies = {"origin"} | {
        bank + "_" + suffix for bank in bank_ids for suffix in candidate_suffixes
    }
    if set(policies) != expected_policies:
        raise ValueError("Q4 bank candidate set is incomplete or unexpected")
    bridge = config["qwen"]["two_gpu_bridge"]
    n = bridge["initial_draws"]
    npilot = int(n * config["observation"]["pilot_fraction"])
    if len(bank_ids) != bridge["dev_banks"] or len(probe_ids) != bridge["probes"] or npilot < 2:
        raise ValueError("Q4 executed bank/probe/pilot size differs from the frozen bridge")
    if any(not row.get("generation_parity", {}).get("passed") for row in samples):
        raise ValueError("Q4 original generation/rescore parity is not complete")
    scored = defaultdict(dict)
    for row in scores:
        key, candidate = row["proposal_sample_key"], row["candidate_id"]
        if key not in sample_lookup or key in scored[candidate]:
            raise ValueError("Q4 candidate scoring repeats or invents a proposal request")
        if row["inference_fingerprint"] != policies[candidate]["inference_fingerprint"]:
            raise ValueError("Q4 candidate inference fingerprint changed")
        scored[candidate][key] = row
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["prompt_id"], row["role"], row["proposal"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["draw_index"])
    for prompt_id in probe_ids:
        pilot, main, reference = (
            grouped[prompt_id, role, "ORIGIN"] for role in ("pilot", "main", "reference")
        )
        if (len(pilot), len(main), len(reference)) != (npilot, n - npilot, n):
            raise ValueError("Q4 pilot/main/reference draws differ from the frozen budget")
        all_reference = [
            row for row in samples if row["prompt_id"] == prompt_id and row["role"] == "reference"
        ]
        validate_role_independence(pilot=pilot, main=main, reference=all_reference)
        for row in (*pilot, *main, *reference):
            source_score = scored["origin"].get(row["sample_key"])
            if (
                source_score is None
                or source_score["inference_fingerprint"] != row["inference_fingerprint"]
            ):
                raise ValueError("Q4 ORIGIN generating policy lacks its matching baseline score")
            # Also apply the full token/input/decoder identity join to null policy scores.
            packet_contributions(
                [row],
                [source_score],
                [source_score],
                proposal="ORIGIN",
                origin_support_certified=True,
            )
            diffs = [
                abs(a - b)
                for a, b in zip(
                    row["behavior_token_logprobs"], source_score["token_logprobs"], strict=True
                )
            ]
            parity = _parity_values(
                [diffs],
                row["generation_parity"]["tolerances"],
                sequence_errors=[
                    abs(row["generation_sequence_logp"] - source_score["sequence_logp"])
                ],
            )
            if not parity["passed"]:
                raise ValueError(
                    "Q4 ORIGIN baseline score differs from actual sampling probability"
                )
    alpha = 0.05 / max(1, len(bank_ids) * len(probe_ids) * len(Q4_CONTRASTS) * 4)
    observations, references, mix_checks, arrays, array_index = [], [], [], {}, []
    failures, noisy_errors = [], defaultdict(list)
    support_assumptions = {
        "proposal": "Inherited untransformed pure softmax",
        "condition": "finite logits on every legal finite-horizon prefix",
        "finite_draws_alone_certify_global_overlap": False,
        "sampled_origin_probability_path_checked": True,
    }
    q = config["qwen"]
    if any(
        q[k] != val
        for k, val in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 0), ("max_new_tokens", 64))
    ):
        raise ValueError(
            "Q4 conditional support requires the frozen pure-softmax action distribution"
        )
    for bank in bank_ids:
        for prompt_id in probe_ids:
            unit = len(observations)
            batches, tail, reference_for_unit = {}, {}, []
            for role in ("pilot", "main", "reference"):
                packet = grouped[prompt_id, role, "ORIGIN"]
                raw, pairs = [], []
                for contrast, suffix_a, suffix_b in Q4_CONTRASTS:
                    a, b = bank + "_" + suffix_a, bank + "_" + suffix_b
                    za, diagnostics = packet_contributions(
                        packet,
                        [scored[a][r["sample_key"]] for r in packet],
                        [scored[b][r["sample_key"]] for r in packet],
                        proposal="ORIGIN",
                        origin_support_certified=True,
                    )
                    raw.append(za)
                    pairs.append(
                        (policies[a]["inference_fingerprint"], policies[b]["inference_fingerprint"])
                    )
                    tail[role + ":" + contrast] = diagnostics
                    if role == "reference":
                        diagnostic = _q4_reference_diagnostic(
                            za, diagnostics, proposal="ORIGIN", alpha=alpha
                        )
                        diagnostic.update(
                            bank_id=bank,
                            prompt_id=prompt_id,
                            contrast_id=contrast,
                            packet_hash=digest(packet),
                            independent_from_pilot_main=True,
                        )
                        references.append(diagnostic)
                        reference_for_unit.append(diagnostic)
                z = np.stack(raw, axis=1)
                array_key = f"unit_{unit:04d}_{role}_raw4"
                arrays[array_key] = z
                array_index.append(
                    {
                        "key": array_key,
                        "bank_id": bank,
                        "prompt_id": prompt_id,
                        "role": role,
                        "sample_keys": [r["sample_key"] for r in packet],
                        "contrast_ids": [r[0] for r in Q4_CONTRASTS],
                        "policy_pairs": pairs,
                    }
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    stable_moments = np.isfinite(z).all() and np.isfinite(z * z).all()
                if role != "reference" and stable_moments:
                    batches[role] = ContributionBatch(
                        z,
                        tuple(r["sample_key"] for r in packet),
                        digest([role, [r["sample_rng_key"] for r in packet]]),
                        prompt_id,
                        tuple(c[0] for c in Q4_CONTRASTS),
                        tuple(pairs),
                        tuple(tuple(r["token_ids"]) for r in packet),
                        "origin",
                    )
            row = {
                "bank_id": bank,
                "prompt_id": prompt_id,
                "contrasts": [r[0] for r in Q4_CONTRASTS],
                "cost_matched_total_draws": n,
                "pilot_draws": npilot,
                "main_draws": n - npilot,
                "tail_diagnostics": tail,
                "support_assumptions": support_assumptions,
                "reference_status": "REFERENCE_UNRESOLVED",
            }
            if set(batches) != {"pilot", "main"}:
                failures.append(
                    {
                        "bank_id": bank,
                        "prompt_id": prompt_id,
                        "reason": "UNSTABLE_ORIGIN_WEIGHT_MOMENTS",
                    }
                )
                row.update(status="MEASUREMENT_UNRESOLVED", estimators={})
            else:
                estimates = cost_matched_estimates(
                    batches["pilot"],
                    batches["main"],
                    shrink=0.1,
                    l1_cap=config["observation"]["pilot_b_l1_cap"],
                    include_crossfit=True,
                )
                row.update(
                    status="ENGINEERING_OBSERVATIONS_COMPUTED_REFERENCE_UNRESOLVED",
                    estimators={
                        name: _estimate_document(value)
                        for name, value in estimates["cost_matched"].items()
                    },
                    paired_main_only={
                        name: _estimate_document(value)
                        for name, value in estimates["paired_main_only"].items()
                    },
                )
                for name, estimate in estimates["cost_matched"].items():
                    comparisons = []
                    for ci, ref in enumerate(reference_for_unit):
                        if ref["mean"] is None or ref["standard_error"] is None:
                            comparisons.append({"status": "REFERENCE_UNRESOLVED"})
                            continue
                        residual = estimate.estimate[ci] - np.asarray(ref["mean"])
                        noise = np.asarray(ref["standard_error"]) ** 2
                        comparisons.append(
                            {
                                "residual_vs_noisy_reference": residual.tolist(),
                                "reference_standard_error": ref["standard_error"],
                                "status": "REFERENCE_UNRESOLVED_NOT_A_RANKING",
                            }
                        )
                        noisy_errors[name, Q4_CONTRASTS[ci][0]].append((residual**2, noise))
                    row["estimators"][name]["independent_reference_comparison"] = comparisons
            observations.append(row)
    # MIX is legal only for its actual sampled pair, and each prompt stays separate.
    for prompt_id in probe_ids:
        packet = grouped[prompt_id, "reference", "MIX"]
        if len(packet) != n:
            raise ValueError("Q4 needs the frozen independent MIX spotcheck per probe")
        endpoints = packet[0]["proposal_candidates"]
        matches = [
            (bank, name, bank + "_" + a, bank + "_" + b)
            for bank in bank_ids
            for name, a, b in Q4_CONTRASTS
            if set(endpoints) == {bank + "_" + a, bank + "_" + b}
        ]
        if len(matches) != 1:
            raise ValueError("Q4 MIX packet endpoints do not identify one registered bank contrast")
        bank, contrast, a, b = matches[0]
        zmix, diagnostic = packet_contributions(
            packet,
            [scored[a][r["sample_key"]] for r in packet],
            [scored[b][r["sample_key"]] for r in packet],
            proposal="MIX",
        )
        mixed = _q4_reference_diagnostic(zmix, diagnostic, proposal="MIX", alpha=alpha)
        origin = next(
            r
            for r in references
            if (r["bank_id"], r["prompt_id"], r["contrast_id"]) == (bank, prompt_id, contrast)
        )
        if origin["mean"] is not None and mixed["mean"] is not None:
            diff = np.asarray(origin["mean"]) - np.asarray(mixed["mean"])
            se = np.sqrt(
                np.asarray(origin["standard_error"]) ** 2 + np.asarray(mixed["standard_error"]) ** 2
            )
            z = statistics.NormalDist().inv_cdf(1 - alpha / 2)
            inconsistent = bool(np.any(np.abs(diff) > z * se))
        else:
            diff, se, inconsistent = None, None, True
        mixed.update(
            bank_id=bank,
            prompt_id=prompt_id,
            contrast_id=contrast,
            origin_minus_mix=None if diff is None else diff.tolist(),
            combined_empirical_standard_error=None if se is None else se.tolist(),
            empirical_crosscheck_alarm=inconsistent,
            empirical_alarm_certifies_agreement=False,
            independent_rng_checked=True,
            packet_hash=digest(packet),
        )
        if inconsistent:
            failures.append(
                {"bank_id": bank, "prompt_id": prompt_id, "reason": "ORIGIN_MIX_CROSSCHECK_ALARM"}
            )
        mix_checks.append(mixed)
        mix_key = f"mix_{len(mix_checks) - 1:04d}_raw4"
        arrays[mix_key] = zmix
        array_index.append(
            {
                "key": mix_key,
                "bank_id": bank,
                "prompt_id": prompt_id,
                "role": "reference",
                "proposal": "MIX",
                "sample_keys": [r["sample_key"] for r in packet],
                "contrast_ids": [contrast],
                "policy_pairs": [
                    [policies[a]["inference_fingerprint"], policies[b]["inference_fingerprint"]]
                ],
            }
        )
    direct, direct_contrasts = [], []
    for prompt_id in probe_ids:
        values = grouped[prompt_id, "reference", "DIRECT"]
        by_policy = defaultdict(list)
        for row in values:
            by_policy[row["candidate_id"]].append(row)
        if len(by_policy) != 2 or any(len(rows) != n for rows in by_policy.values()):
            raise ValueError("Q4 independent direct baseline needs two endpoints with frozen draws")
        counts_by_policy = {}
        for candidate, packet in sorted(by_policy.items()):
            onehot = np.asarray(
                [[int(r["category"] == e) for e in EVENTS] for r in packet], dtype=float
            )
            counts = onehot.sum(axis=0)
            covariance = np.cov(onehot, rowvar=False, ddof=1) / n
            direct.append(
                {
                    "candidate_id": candidate,
                    "prompt_id": prompt_id,
                    "n": n,
                    "event_counts": counts.astype(int).tolist(),
                    "event_probability": (counts / n).tolist(),
                    "covariance_of_mean": covariance.tolist(),
                    "independent_generation": True,
                    "is_exact_event_probability": False,
                }
            )
            counts_by_policy[candidate] = (counts / n, covariance)
        for bank, contrast, a, b in [
            (bank, name, bank + "_" + a, bank + "_" + b)
            for bank in bank_ids
            for name, a, b in Q4_CONTRASTS
        ]:
            if a not in counts_by_policy or b not in counts_by_policy:
                continue
            pa, ca = counts_by_policy[a]
            pb, cb = counts_by_policy[b]
            ref = next(
                r
                for r in references
                if (r["bank_id"], r["prompt_id"], r["contrast_id"]) == (bank, prompt_id, contrast)
            )
            comparison = {"status": "REFERENCE_UNRESOLVED_NOT_A_RANKING"}
            if ref["mean"] is not None:
                comparison.update(
                    residual_vs_noisy_reference=(pb - pa - np.asarray(ref["mean"])).tolist(),
                    reference_standard_error=ref["standard_error"],
                )
            direct_contrasts.append(
                {
                    "bank_id": bank,
                    "prompt_id": prompt_id,
                    "contrast_id": contrast,
                    "estimate_raw4": (pb - pa).tolist(),
                    "covariance_of_mean": (ca + cb).tolist(),
                    "delta_pX": float((pb - pa)[0]),
                    "delta_v": float(-(pb - pa)[3]),
                    "total_generated_sequences": 2 * n,
                    "origin_measurement_total_draws": n,
                    "same_cost_claim": False,
                    "independent_endpoint_counts": True,
                    "independent_reference_comparison": comparison,
                    "formal_hoeffding_half_width": 2 * math.sqrt(math.log(4 / alpha) / (2 * n)),
                    "reference_status": "REFERENCE_UNRESOLVED",
                }
            )
    aggregate = []
    for (method, contrast), rows in sorted(noisy_errors.items()):
        squared = np.asarray([r[0] for r in rows])
        noise = np.asarray([r[1] for r in rows])
        aggregate.append(
            {
                "method": method,
                "contrast_id": contrast,
                "units": len(rows),
                "raw_noisy_reference_mse": squared.mean(axis=0).tolist(),
                "aggregate_reference_noise_corrected_mse": (
                    squared.mean(axis=0) - noise.mean(axis=0)
                ).tolist(),
                "negative_cells_clipped": False,
                "ranking_allowed": False,
                "status": "REFERENCE_UNRESOLVED",
            }
        )
    raw_path = root / "q4_raw_contributions.npz"
    if raw_path.exists():
        with np.load(raw_path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[k], value, equal_nan=True) for k, value in arrays.items()
            ):
                raise ValueError("Original Q4 raw contribution archive changed")
    else:
        atomic_npz(raw_path, arrays)
    mandatory_mix = [
        {
            "bank_id": row["bank_id"],
            "prompt_id": row["prompt_id"],
            "role_contrast": name,
            "reasons": diag["reasons"],
        }
        for row in observations
        for name, diag in row["tail_diagnostics"].items()
        if diag["requires_independent_mix"]
    ]
    available_mix = {(r["bank_id"], r["prompt_id"], r["contrast_id"]) for r in mix_checks}
    for cell in mandatory_mix:
        if (
            cell["bank_id"],
            cell["prompt_id"],
            cell["role_contrast"].split(":", 1)[1],
        ) not in available_mix:
            failures.append({**cell, "reason": "REQUIRED_INDEPENDENT_MIX_NOT_COLLECTED"})
    report = {
        "schema": "modeling-v3-Q4-observation-analysis-v1",
        "config_hash": canonical_hash(config),
        "source_hash": source_identity()["sha256"],
        "runtime_identity": runtime,
        "input_manifest_hashes": [generation["manifest_hash"], scoring["manifest_hash"]],
        "technical_chain_passed": not failures,
        "technical_failures": failures,
        "observations": observations,
        "independent_origin_references": references,
        "independent_mix_crosschecks": mix_checks,
        "independent_direct_counts": direct,
        "independent_direct_contrasts": direct_contrasts,
        "noisy_reference_error_aggregates": aggregate,
        "required_independent_mix_cells": mandatory_mix,
        "all_originals_retained": True,
        "raw_contributions": {
            "path": str(raw_path),
            "sha256": file_digest(raw_path),
            "arrays": array_index,
        },
        "formal_reference_status": "REFERENCE_UNRESOLVED",
        "scientific_method_winner": None,
        "online_ssvc_validated": False,
        "bank_prompt_units": len(observations),
        "direct_count_units": len(direct),
        "reference_initial_draws": n,
        "cost_scope": "Physical calls and aliases are recorded in worker COST and sample ledgers",
    }
    path = root / "observation_geometry.json"
    _publish_analysis(path, report)
    return {
        "path": str(path),
        "sha256": file_digest(path),
        "bank_prompt_units": len(observations),
        "direct_count_units": len(direct),
        "technical_chain_passed": report["technical_chain_passed"],
        "reference_status": "REFERENCE_UNRESOLVED",
        "required_independent_mix_cells": len(mandatory_mix),
    }
