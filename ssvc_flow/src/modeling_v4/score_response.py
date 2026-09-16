"""Full-coordinate score derivatives, separate from finite response measurements.

Torch is imported only during execution. No checkpoint selection, generation,
reference-label access, optimizer steps, or implicit low-rank projection occurs
here. Callers load the declared expansion point before collecting its gradient.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .observations import assert_predictor_independence

EVENTS = "XSWI"


def exact_score_response(
    origin, baseline, candidate, features, categories, *, projection_basis=None
):
    """Exact finite-action softmax response and three distinct analytic Jacobians."""
    f = np.asarray(features, dtype=np.float64)
    points = [np.asarray(x, dtype=np.float64) for x in (origin, baseline, candidate)]
    cats = np.asarray(categories)
    if f.ndim != 2 or any(p.shape != (f.shape[1],) for p in points):
        raise ValueError("Full parameter vectors must match the feature dimension")
    if (
        cats.shape != (len(f),)
        or not np.issubdtype(cats.dtype, np.integer)
        or np.any((cats < 0) | (cats > 3))
    ):
        raise ValueError("Every finite action needs one of four event categories")
    if not all(np.isfinite(x).all() for x in [f, *points]):
        raise ValueError("Finite features and parameters required")
    onehot = np.eye(4)[cats]

    def evaluate(point):
        logits = f @ point
        p = np.exp(logits - logits.max())
        p /= p.sum()
        jac = onehot.T @ (p[:, None] * (f - p @ f))
        return p @ onehot, jac

    e = points[2] - points[1]
    expansion = dict(
        zip(
            ("ORIGIN", "BASE", "MIDPOINT"),
            (points[0], points[1], (points[1] + points[2]) / 2),
            strict=True,
        )
    )
    jacobians = {key: evaluate(value)[1] for key, value in expansion.items()}
    result = {
        "direction": e,
        "jacobians": jacobians,
        "derivatives": {key: jac @ e for key, jac in jacobians.items()},
        "finite_response": evaluate(points[2])[0] - evaluate(points[1])[0],
        "derivative_method": "ANALYTIC_FULL_SCORE",
        "finite_difference_is_derivative": False,
    }
    if projection_basis is not None:
        q = np.asarray(projection_basis, dtype=np.float64)
        if (
            q.ndim != 2
            or q.shape[0] != len(e)
            or not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)
        ):
            raise ValueError("Diagnostic projection basis must be orthonormal")
        residual = e - q @ (q.T @ e)
        result.update(
            orthogonal_direction=residual, orthogonal_contribution=jacobians["ORIGIN"] @ residual
        )
    return result


def directional_derivative(fn, point, direction, *, method="auto", jvp=None):
    """Use forward AD or full-coordinate reverse gradient dot, never a difference."""
    import torch

    if method not in {"auto", "jvp", "reverse"}:
        raise ValueError("Unknown differentiation method")
    point, direction = torch.as_tensor(point), torch.as_tensor(direction)
    if (
        point.shape != direction.shape
        or not torch.isfinite(point).all()
        or not torch.isfinite(direction).all()
    ):
        raise ValueError("Point and full direction must have matching finite coordinates")
    failure, calls, backwards = None, 0, 0
    if method != "reverse":
        try:
            calls += 1
            value, derivative = (jvp or torch.func.jvp)(fn, (point,), (direction,))
            used = "FORWARD_JVP"
        except (NotImplementedError, RuntimeError) as exc:
            unsupported = isinstance(exc, NotImplementedError) or any(
                text in str(exc).lower()
                for text in (
                    "forward ad",
                    "forward-mode ad",
                    "jvp not implemented",
                    "does not support forward",
                )
            )
            if method == "jvp" or not unsupported:
                raise
            failure = str(exc)
    if method == "reverse" or failure is not None:
        point = point.detach().requires_grad_(True)
        calls += 1
        value = fn(point)
        flat = value.reshape(-1)
        products = []
        for i, scalar in enumerate(flat):
            if scalar.requires_grad:
                gradient = torch.autograd.grad(
                    scalar, point, retain_graph=i + 1 < flat.numel(), allow_unused=True
                )[0]
                backwards += 1
                products.append(
                    (gradient * direction).sum() if gradient is not None else point.new_zeros(())
                )
            else:
                products.append(point.new_zeros(()))
        derivative, used = torch.stack(products).reshape(value.shape), "REVERSE_GRAD_DOT"
    return {
        "value": value.detach().cpu().numpy(),
        "derivative": derivative.detach().cpu().numpy(),
        "method": used,
        "jvp_failure": failure,
        "function_calls": calls,
        "backward_calls": backwards,
        "finite_difference_is_derivative": False,
    }


def numerical_directional_check(fn, point, direction, *, steps, method="auto"):
    """Independent central differences diagnose AD; they never replace its output."""
    import torch

    steps = list(steps)
    if not steps or any(not math.isfinite(h) or h <= 0 for h in steps):
        raise ValueError("Positive predeclared numerical-check steps required")
    point, direction = torch.as_tensor(point), torch.as_tensor(direction)
    result = directional_derivative(fn, point, direction, method=method)
    checks = []
    with torch.no_grad():
        for h in steps:
            difference = (
                ((fn(point + h * direction) - fn(point - h * direction)) / (2 * h))
                .detach()
                .cpu()
                .numpy()
            )
            checks.append(
                {
                    "step": h,
                    "central_difference": difference,
                    "max_abs_error": float(np.max(np.abs(difference - result["derivative"]))),
                }
            )
    return {**result, "checks": checks, "numerical_check_function_calls": 2 * len(steps)}


def _active_tokens(token_ids, eos_ids, max_new_tokens, token_mask=None):
    tokens = list(token_ids)
    mask = list(token_mask) if token_mask is not None else [True] * len(tokens)
    if (
        not tokens
        or len(mask) != len(tokens)
        or any(type(v) is not bool for v in mask)
        or any(type(t) is not int or t < 0 for t in tokens)
    ):
        raise ValueError("Explicit token IDs and boolean mask required")
    n = sum(mask)
    if n == 0 or mask != [True] * n + [False] * (len(mask) - n):
        raise ValueError("Action mask must be a nonempty contiguous prefix")
    if type(max_new_tokens) is not int or max_new_tokens < 1 or n > max_new_tokens:
        raise ValueError("Action exceeds its frozen horizon")
    action = tokens[:n]
    if any(t in eos_ids for t in action[:-1]):
        raise ValueError("An action cannot continue after EOS")
    if action[-1] not in eos_ids and n != max_new_tokens:
        raise ValueError("Non-EOS action must end at the frozen truncation horizon")
    return action


def sequence_log_probability(token_logps, token_ids, *, eos_ids, max_new_tokens, token_mask=None):
    """FP64 sum of real action tokens including EOS, without length normalization."""
    import torch

    action = _active_tokens(token_ids, eos_ids, max_new_tokens, token_mask)
    values = torch.as_tensor(token_logps)
    if values.ndim != 1 or len(values) != len(token_ids):
        raise ValueError("Token log probabilities must align with the entire explicit mask")
    active = values[: len(action)]
    if not torch.isfinite(active.detach()).all() or bool((active.detach() > 1e-6).any()):
        raise ValueError("Finite normalized token log probabilities required")
    return active.to(dtype=torch.float64).sum()


def differentiable_full_teacher_logprobs(adapter, prepared, tokens):
    """Candidate fast path; use requires separate measured derivative certification.

    This mirrors BaseAdapter's official continuation positions but retains the
    graph. An evaluation-only token probability certificate cannot enable it.
    """
    import torch

    adapter.model.eval()
    adapter._reset_positions()
    with torch.enable_grad():
        output = adapter.model(**adapter.full_inputs(prepared, tokens), use_cache=False)
        start = prepared["audit"]["prompt_token_count"] - 1
        logits = output.logits[0, start : start + len(tokens)]
        if len(logits) != len(tokens):
            raise ValueError("Teacher-forced action positions differ from the prefix action")
        ids = torch.tensor(tokens, dtype=torch.long, device=logits.device)
        return logits.float().log_softmax(-1).gather(1, ids[:, None])[:, 0]


def _layout(runtime, parameter_order=None):
    model = runtime["adapter"].model
    params = dict(
        sorted((name, value) for name, value in model.named_parameters() if value.requires_grad)
    )
    expected = list(
        parameter_order if parameter_order is not None else runtime.get("parameter_order", params)
    )
    if not params or expected != list(params):
        raise ValueError("Parameter order must contain every sorted trainable coordinate")
    return params


def _sample_identity(sample):
    role = sample.get("role")
    if role not in {"anchor", "main", "work", "pilot", "derivative", "score_anchor"}:
        raise ValueError("Only independent anchor draws are allowed; reference draws are forbidden")
    key, rng = sample.get("sample_key"), sample.get("rng_stream_id", sample.get("rng_namespace"))
    if not isinstance(key, str) or not key or not isinstance(rng, str) or not rng:
        raise ValueError("Anchor sample and RNG identities are required")
    return key, rng


def score_sample_gradient(
    runtime, prompt, sample, *, parameter_order=None, scorer=None, derivative_certificate=None
):
    """Execute the prefix derivative, or a separately certified same-policy fast path."""
    from ..modeling_v3.io import canonical_hash

    params = _layout(runtime, parameter_order)
    if scorer is not None:
        certificate = derivative_certificate or {}
        if (
            certificate.get("passed") is not True
            or certificate.get("scope") != "full_trainable_gradient_and_probability"
            or certificate.get("parameter_order") != list(params)
            or certificate.get("runtime_identity_hash") != canonical_hash(runtime["identity"])
            or certificate.get("expansion_policy_identity") != runtime.get("active_policy_identity")
            or certificate.get("candidate_path") != f"{scorer.__module__}.{scorer.__qualname__}"
        ):
            raise ValueError(
                "Fast scoring requires a separate full-trainable derivative certificate"
            )
    return _score_sample_gradient(
        runtime, prompt, sample, parameter_order=parameter_order, scorer=scorer
    )


def _score_sample_gradient(
    runtime, prompt, sample, *, parameter_order=None, scorer=None, enforce_probability_parity=True
):
    """Differentiate one exact terminal action through all trainable parameters.

    Gradients are CPU FP64 arrays. Existing parameter .grad buffers, optimizer
    state and frozen parameters remain untouched. Costs survive failed calls in
    runtime['score_cost_ledger']; exceptions also expose their current cost.
    """
    import torch

    from ..modeling_v3.vlm_observation import _parity_values, validate_action

    key, rng = _sample_identity(sample)
    params = _layout(runtime, parameter_order)
    adapter, model = runtime["adapter"], runtime["adapter"].model
    if prompt.get("prompt_id") != sample.get("prompt_id"):
        raise ValueError("Anchor and prepared prompt identities differ")
    action = _active_tokens(
        sample["token_ids"], adapter.eos_ids, sample["max_new_tokens"], sample.get("token_mask")
    )
    validation_sample = {
        **sample,
        "token_ids": action,
        "behavior_token_logprobs": sample["behavior_token_logprobs"][: len(action)],
    }
    tokenizer = getattr(getattr(adapter, "processor", None), "tokenizer", None)
    ledger = validate_action(
        validation_sample,
        eos_ids=adapter.eos_ids,
        max_new_tokens=sample["max_new_tokens"],
        tokenizer=tokenizer,
    )
    prepared = adapter.prepare(
        prompt["prompt"], prompt.get("data_root") or runtime.get("data_root")
    )
    if prepared.get("audit", {}).get("enable_thinking") is True:
        raise ValueError("Derivative scoring cannot alter the frozen thinking policy")
    expected_input = sample.get("input_tensor_hash")
    if expected_input is not None and expected_input != prepared.get("audit", {}).get(
        "input_tensor_hash"
    ):
        raise ValueError("Anchor input tensor identity changed")
    modes = [(module, module.training) for module in model.modules()]
    versions = {
        name: (id(p), p._version, id(p.grad), p.grad._version if p.grad is not None else None)
        for name, p in model.named_parameters()
    }
    guard = runtime.get("state_guard")
    before = guard() if guard else None
    forward_start = getattr(adapter, "forward_calls", None)
    device = next(iter(params.values())).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    cost = {
        "forward_calls": 0,
        "backward_calls": 0,
        "gradient_score_calls": 0,
        "parity_score_calls": 0,
        "scored_sequences": 0,
        "scored_tokens": 0,
    }
    fault = None
    try:
        cost["gradient_score_calls"] += 1
        values = (
            scorer(adapter, prepared, action)
            if scorer
            else adapter.logprobs(prepared, action, require_grad=True)
        )
        cost["scored_sequences"] += 1
        cost["scored_tokens"] += len(action)
        sequence = sequence_log_probability(
            values, action, eos_ids=adapter.eos_ids, max_new_tokens=sample["max_new_tokens"]
        )
        cost["backward_calls"] += 1
        gradients = torch.autograd.grad(sequence, tuple(params.values()), allow_unused=True)
        cpu = {
            name: (
                gradient.detach().double().cpu().numpy().copy()
                if gradient is not None
                else np.zeros(tuple(param.shape))
            )
            for (name, param), gradient in zip(params.items(), gradients, strict=True)
        }
        if not all(np.isfinite(v).all() for v in cpu.values()):
            raise ValueError("Nonfinite full score gradient")
        cost["parity_score_calls"] += 1
        ordinary = adapter.logprobs(prepared, action, require_grad=False)
        cost["scored_sequences"] += 1
        cost["scored_tokens"] += len(action)
        ordinary_sequence = sequence_log_probability(
            ordinary, action, eos_ids=adapter.eos_ids, max_new_tokens=sample["max_new_tokens"]
        )
        token_values = values.detach().double().cpu().numpy()
        differences = np.abs(token_values - ordinary.detach().double().cpu().numpy())
        parity = _parity_values(
            [differences.tolist()],
            runtime["parity_tolerances"],
            sequence_errors=[abs(sequence.item() - ordinary_sequence.item())],
        )
        if enforce_probability_parity and not parity["passed"]:
            raise ValueError(
                "Gradient execution token/sequence probabilities fail ordinary-path parity"
            )
        result = {
            "sample_key": key,
            "rng_stream_id": rng,
            "sequence_logp": sequence.item(),
            "token_logprobs": token_values,
            "gradients": cpu,
            "parameter_order": list(params),
            "parameter_shapes": {n: list(p.shape) for n, p in params.items()},
            "parameter_dtypes": {n: str(p.dtype) for n, p in params.items()},
            "parity": parity,
            "action": ledger,
            "cost": cost,
            "method": "FULL_REVERSE_SCORE_GRADIENT",
            "sequence_accumulation_dtype": "float64",
            "forward_precision_is_separate_from_accumulation": True,
            "numerical_path": "certified_differentiable_fastpath"
            if scorer
            else "uncached_prefix_recompute",
            "reference_labels_read": False,
        }
    except BaseException as exc:
        fault = exc
        exc.cost = cost
        raise
    finally:
        for module, training in modes:
            module.training = training
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            cost["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        cost["wall_seconds"] = time.perf_counter() - started
        cost["forward_calls"] = (
            adapter.forward_calls - forward_start if forward_start is not None else None
        )
        after = {
            name: (id(p), p._version, id(p.grad), p.grad._version if p.grad is not None else None)
            for name, p in model.named_parameters()
        }
        mutated = versions != after or (guard and before != guard())
        runtime.setdefault("score_cost_ledger", []).append(
            {
                "sample_key": key,
                "cost": dict(cost),
                "status": "FAILED" if fault or mutated else "RETURNED",
            }
        )
        if mutated:
            raise RuntimeError("Score derivative mutated model/optimizer state or gradient buffers")
    return result


def certify_gradient_path(runtime, prompt, sample, *, scorer, directions, tolerances):
    """Measure a candidate gradient path against prefix autograd before enabling it.

    Thresholds are explicit predeclared engineering tolerances, not scientific
    coverage guarantees. Both complete gradients and all requested full-vector
    dot products are compared. Failure leaves prefix autograd available.
    """
    from ..modeling_v3.io import canonical_hash
    from ..modeling_v3.vlm_observation import _parity_values

    params = _layout(runtime)
    required = (
        "mean_abs_token_logp",
        "max_abs_token_logp",
        "max_abs_sequence_logp",
        "relative_l2",
        "absolute_directional_dot",
    )
    if any(
        type(tolerances.get(k)) not in (int, float)
        or not math.isfinite(tolerances[k])
        or tolerances[k] < 0
        for k in required
    ):
        raise ValueError("Explicit finite predeclared probability and gradient tolerances required")
    if not directions or any(
        set(v) != set(params)
        or any(
            np.asarray(v[n]).shape != tuple(p.shape) or not np.isfinite(v[n]).all()
            for n, p in params.items()
        )
        for v in directions.values()
    ):
        raise ValueError("Certification directions must cover every trainable coordinate")
    if not runtime["identity"].get("fixture") and runtime.get("active_policy_identity") is None:
        raise ValueError("Real derivative certification must bind the active checkpoint policy")
    reference = _score_sample_gradient(runtime, prompt, sample)
    candidate = _score_sample_gradient(
        runtime, prompt, sample, scorer=scorer, enforce_probability_parity=False
    )
    probability = _parity_values(
        [np.abs(reference["token_logprobs"] - candidate["token_logprobs"]).tolist()],
        tolerances,
        sequence_errors=[abs(reference["sequence_logp"] - candidate["sequence_logp"])],
    )
    blocks, errors, norms, inner, candidate_norm = {}, 0.0, 0.0, 0.0, 0.0
    for name, a in reference["gradients"].items():
        b = candidate["gradients"][name]
        squared_error = float(np.sum((a - b) ** 2))
        squared_norm = float(np.sum(a * a))
        norm_b = float(np.sum(b * b))
        dot = float(np.sum(a * b))
        relative = (
            math.sqrt(squared_error / squared_norm)
            if squared_norm
            else (0.0 if squared_error == 0 else None)
        )
        cosine = dot / math.sqrt(squared_norm * norm_b) if squared_norm and norm_b else None
        blocks[name] = {
            "max_abs_error": float(np.max(np.abs(a - b))),
            "relative_l2": relative,
            "cosine": cosine,
            "coordinates": a.size,
            "finite": bool(np.isfinite(b).all()),
        }
        errors += squared_error
        norms += squared_norm
        inner += dot
        candidate_norm += norm_b
    dots = {}
    for key, direction in directions.items():
        a = sum(float(np.sum(reference["gradients"][n] * v)) for n, v in direction.items())
        b = sum(float(np.sum(candidate["gradients"][n] * v)) for n, v in direction.items())
        dots[key] = {"prefix_dot": a, "candidate_dot": b, "absolute_error": abs(a - b)}
    relative = math.sqrt(errors / norms) if norms else (0.0 if errors == 0 else None)
    passed = (
        probability["passed"]
        and all(
            v["finite"]
            and v["relative_l2"] is not None
            and v["relative_l2"] <= tolerances["relative_l2"]
            for v in blocks.values()
        )
        and all(
            v["absolute_error"] <= tolerances["absolute_directional_dot"] for v in dots.values()
        )
    )
    costs = {}
    for result in (reference, candidate):
        for name, value in result["cost"].items():
            if value is not None:
                costs[name] = (
                    max(costs.get(name, 0), value)
                    if name.startswith("peak_")
                    else costs.get(name, 0) + value
                )
    return {
        "scope": "full_trainable_gradient_and_probability",
        "status": "ENGINEERING_PARITY_PASSED" if passed else "NOT_CERTIFIED",
        "passed": bool(passed),
        "fixture": bool(runtime["identity"].get("fixture", False)),
        "runtime_identity_hash": canonical_hash(runtime["identity"]),
        "expansion_policy_identity": runtime.get("active_policy_identity"),
        "candidate_path": f"{scorer.__module__}.{scorer.__qualname__}",
        "parameter_order": list(params),
        "sample_key": sample["sample_key"],
        "prompt_id": prompt["prompt_id"],
        "probability": probability,
        "blocks": blocks,
        "directional_dots": dots,
        "relative_l2": relative,
        "cosine": inner / math.sqrt(norms * candidate_norm) if norms and candidate_norm else None,
        "tolerances": dict(tolerances),
        "cost": costs,
        "reference_path": "uncached_prefix_recompute",
        "fallback": "uncached_prefix_recompute",
        "scientific_coverage_certified": False,
    }


def _proposal_logp(sample):
    if "proposal_sequence_logp" in sample:
        value = sample["proposal_sequence_logp"]
    elif sample.get("proposal") in {"ORIGIN", "DIRECT"}:
        value = sample["generation_sequence_logp"]
    else:
        raise ValueError(
            "MIX anchor needs the actual mixture proposal probability, not its source component"
        )
    if not math.isfinite(value) or value > 1e-6:
        raise ValueError("Invalid actual anchor proposal log probability")
    return value


def collect_score_response(
    runtime,
    prompts,
    samples,
    *,
    expansion_point="ORIGIN",
    prompt_subset=(),
    expected_groups=None,
    reference_sample_ids=(),
    reference_rng_stream_ids=(),
    gradients=True,
    directions=None,
):
    """Stream full score gradients into equal-prompt group event responses.

    BASE and MIDPOINT use unnormalized importance ratios from the actual anchor
    proposal to the caller-loaded expansion policy. No raw mass is clipped or
    projected away. With gradients=False only directional responses are retained.
    Raw per-draw score products remain unweighted and retain input sample order,
    allowing empirical second moments without reconstructing scores from events.
    A bounded direction provider may expose ordered iteration, len, validate(params)
    and dot_all(gradients) to contract disk-backed full directions without loading
    the entire direction matrix into memory.
    """
    from ..modeling_v3.io import canonical_hash

    if expansion_point not in {"ORIGIN", "BASE", "MIDPOINT"}:
        raise ValueError("Unknown derivative expansion point")
    params = _layout(runtime)
    prompts, samples = list(prompts), list(samples)
    by_id = {p["prompt_id"]: p for p in prompts}
    if len(by_id) != len(prompts) or not prompts:
        raise ValueError("A fixed panel with unique prompt identities is required")
    if expected_groups is None:
        from ..r4_inputs import STRATA

        expected_groups = STRATA
    groups = [tuple(g) for g in expected_groups]
    prompt_groups = {key: (p.get("family"), p.get("interface")) for key, p in by_id.items()}
    if (
        len(set(groups)) != len(groups)
        or set(prompt_groups.values()) != set(groups)
        or any(None in g for g in groups)
    ):
        raise ValueError(
            "The frozen family/interface groups must all be present; no metadata inference"
        )
    subset = list(prompt_subset)
    if len(set(subset)) != len(subset) or set(subset) - by_id.keys():
        raise ValueError("Prompt diagnostic subset must be fixed within the panel")
    identities = [_sample_identity(s) for s in samples]
    if len(set(key for key, _ in identities)) != len(samples):
        raise ValueError("Anchor draws must have unique sample identities")
    assert_predictor_independence(
        [k for k, _ in identities],
        [r for _, r in identities],
        reference_sample_ids=reference_sample_ids,
        reference_rng_stream_ids=reference_rng_stream_ids,
    )
    counts = {p: 0 for p in by_id}
    for sample in samples:
        if sample.get("prompt_id") not in by_id or sample.get("category") not in tuple(EVENTS):
            raise ValueError("Anchor draw has an unknown prompt or event")
        counts[sample["prompt_id"]] += 1
        _proposal_logp(sample)
    if not all(counts.values()):
        raise ValueError("Every fixed prompt needs independent anchor observations")
    directions = directions or {}
    direction_ids = list(directions)
    if len(direction_ids) != len(set(direction_ids)) or len(direction_ids) != len(directions):
        raise ValueError("Every direction must have one unique stable identity")
    provider = callable(getattr(directions, "dot_all", None))
    if provider:
        if not callable(getattr(directions, "validate", None)):
            raise ValueError("Bounded directions require complete layout validation")
        directions.validate(params)
    else:
        for values in directions.values():
            if set(values) != set(params) or any(
                np.asarray(values[n]).shape != tuple(p.shape) or not np.isfinite(values[n]).all()
                for n, p in params.items()
            ):
                raise ValueError("Every direction must retain all trainable parameter coordinates")
    if not gradients and not directions:
        raise ValueError("Streaming collection needs directions when full gradients are omitted")
    grouped = (
        {n: np.zeros((len(groups), 4, *p.shape)) for n, p in params.items()} if gradients else None
    )
    prompt_grads = (
        {n: np.zeros((len(subset), 4, *p.shape)) for n, p in params.items()} if gradients else None
    )
    directional = np.zeros((len(directions), len(groups), 4))
    raw_products = np.zeros((len(samples), len(directions)), dtype=np.float64)
    per_prompt = {p: [] for p in by_id}
    weights, costs, score_records = [], {}, []
    group_sizes = {g: sum(value == g for value in prompt_groups.values()) for g in groups}
    for sample_index, sample in enumerate(samples):
        certificate = runtime.get("derivative_certificate")
        scorer = (
            runtime.get("derivative_scorer") if certificate and certificate.get("passed") else None
        )
        result = score_sample_gradient(
            runtime,
            by_id[sample["prompt_id"]],
            sample,
            scorer=scorer,
            derivative_certificate=certificate,
        )
        weight = math.exp(result["sequence_logp"] - _proposal_logp(sample))
        if not math.isfinite(weight):
            raise ValueError("Nonfinite anchor importance weight; retain raw fault, never clip")
        weights.append(weight)
        score_records.append(
            {
                "sample_key": sample["sample_key"],
                "prompt_id": sample["prompt_id"],
                "sequence_logp": result["sequence_logp"],
                "token_logprobs": result["token_logprobs"].tolist(),
                "proposal_sequence_logp": _proposal_logp(sample),
                "importance_weight": weight,
                "numerical_path": result["numerical_path"],
                "parity": result["parity"],
            }
        )
        pid = sample["prompt_id"]
        g, event = groups.index(prompt_groups[pid]), EVENTS.index(sample["category"])
        scale = weight / counts[pid]
        if gradients:
            for name, value in result["gradients"].items():
                grouped[name][g, event] += value * (scale / group_sizes[groups[g]])
                if pid in subset:
                    prompt_grads[name][subset.index(pid), event] += value * scale
        contributions = np.zeros((len(directions), 4))
        dots = (
            np.asarray(directions.dot_all(result["gradients"]), dtype=np.float64)
            if provider
            else np.asarray(
                [
                    sum(
                        float(np.sum(result["gradients"][n] * values[n], dtype=np.float64))
                        for n in params
                    )
                    for values in directions.values()
                ],
                dtype=np.float64,
            )
        )
        if dots.shape != (len(direction_ids),) or not np.isfinite(dots).all():
            raise ValueError("Full-direction score products must be finite and match frozen IDs")
        for i, dot in enumerate(dots):
            raw_products[sample_index, i] = dot
            contributions[i, event] = weight * dot
            directional[i, g, event] += weight * dot / counts[pid] / group_sizes[groups[g]]
        per_prompt[pid].append(contributions)
        for name, value in result["cost"].items():
            if value is not None:
                costs[name] = (
                    max(costs.get(name, 0), value)
                    if name.startswith("peak_")
                    else costs.get(name, 0) + value
                )
    return {
        "expansion_point": expansion_point,
        "groups": groups,
        "event_order": list(EVENTS),
        "prompt_subset": subset,
        "prompt_counts": counts,
        "parameter_order": list(params),
        "parameter_layout_hash": canonical_hash(
            [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in params.items()]
        ),
        "derivative_kind": "AUTOGRAD_FULL_PARAMETER_SCORE",
        "grouped_gradients": grouped,
        "prompt_gradients": prompt_grads,
        "direction_ids": direction_ids,
        "raw_directional_score_products": raw_products,
        "raw_score_products_importance_weighted": False,
        "sample_records": [
            {
                "sample_id": key,
                "prompt_id": s["prompt_id"],
                "group": list(prompt_groups[s["prompt_id"]]),
                "category": s["category"],
                "rng_namespace": rng,
                "proposal": s["proposal"],
                "proposal_policy_fingerprint": s.get("proposal_fingerprint"),
            }
            for s, (key, rng) in zip(samples, identities, strict=True)
        ],
        "grouped_directional_response": directional,
        "per_prompt_directional_contributions": {p: np.asarray(v) for p, v in per_prompt.items()},
        "raw_directional_mass": directional.sum(-1),
        "raw_gradient_mass": {n: v.sum(1) for n, v in grouped.items()} if gradients else None,
        "importance_weights": np.asarray(weights),
        "sample_score_records": score_records,
        "importance_weights_clipped": False,
        "sample_ids": [k for k, _ in identities],
        "rng_stream_ids": sorted({r for _, r in identities}),
        "cost": costs,
        "reference_labels_read": False,
        "finite_difference_is_derivative": False,
        "group_aggregation": "equal_weight_prompts_then_iid_draw_mean",
        "expansion_policy_identity": runtime.get("active_policy_identity"),
    }
