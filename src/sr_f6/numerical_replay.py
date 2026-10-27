"""Bound, read-only T03 loss/adapter-gradient comparison on original S0 tokens.

Production accepts only replay.open_replay's live, locked session. A separately
named CPU fixture API can only emit SYNTHETIC_ONLY. No optimizer, generation or
training update is invoked. Whole-G8 probability gates retain their registered
token denominator; loss/gradient tolerances apply per sequence and coefficient.

Adapter/reference snapshots, RNG, all grads/requires_grad and each module's mode
are restored even on numerical failure. Frozen 9B base/buffers use version
sentinels instead of a second 36GB copy: unexpected mutation fails and poisons
the runtime, requiring reload. That exceptional mutation is not called restored.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from mm_core.training import capture_rng, restore_rng, state_hash
from sr_f12.training import sequence_objective

from .common import digest, now, safe_evidence, write_once
from .objective import sequence_loss
from .training import _numerical_audit, _safe, _score_row, _verify_prepared
from .transactions import _plain_path, _require, _sha

SCHEMA = "srf6-t03-numerical-replay-v1"
ATOL, RTOL = 1e-6, 1e-5
CASES = ("actual_native_J_advantage", "fixed_positive_probe")
RESULT_FILE = "RESULT.json"


def _json(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key in replay evidence")
            result[key] = value
        return result

    def nonfinite(_):
        raise PermissionError("Nonfinite JSON literal in replay evidence")

    def finite_float(value):
        result = float(value)
        _require(math.isfinite(result), "Nonfinite JSON number in replay evidence")
        return result

    return json.loads(
        payload, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float
    )


def _attempt_reference(root, audit, *, parent, task_id, phase_pin, spec_ref):
    _require(
        audit == root / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS" / task_id,
        "Replay audit must be its exact registered task directory",
    )
    relative = (audit / "ATTEMPT.json").relative_to(root).as_posix()
    payload = safe_evidence(root, relative).read_bytes()
    attempt = _json(payload)
    for key, expected in (
        ("parent", parent),
        ("task_id", task_id),
        ("phase_identity_sha256", phase_pin),
        ("spec_ref", spec_ref),
        ("optimizer_updates", 0),
        ("generated_responses", 0),
    ):
        _require(
            type(attempt.get(key)) is type(expected) and attempt[key] == expected,
            "Replay ATTEMPT binding differs: " + key,
        )
    return dict(path=relative, sha256=hashlib.sha256(payload).hexdigest())


def _stamp(values):
    return {
        name: (id(value), value._version, tuple(value.shape), value.dtype, value.device)
        for name, value in values.items()
    }


def _adapter_state(model, names):
    params = dict(model.named_parameters())
    return {name: params[name].detach().cpu().clone() for name in names}


class _Preserve:
    def __init__(self, runtime, reference):
        import torch

        self.runtime, self.model = runtime, runtime.model
        self.params = dict(self.model.named_parameters())
        self.names = tuple(reference)
        _require(
            bool(self.names) and set(self.names) <= self.params.keys(), "Adapter inventory differs"
        )
        _require(
            set(self.names) == {name for name, param in self.params.items() if param.requires_grad},
            "Replay must start with exactly the inherited adapter trainable",
        )
        _require(
            all(p.dtype == torch.float32 for p in self.params.values()), "Actual model must be FP32"
        )
        self.adapter = _adapter_state(self.model, self.names)
        self.reference = {name: value.detach().cpu().clone() for name, value in reference.items()}
        self.reference_object = reference
        self.reference_tensors = dict(reference)
        _require(
            all(
                isinstance(v, torch.Tensor)
                and not v.requires_grad
                and v.dtype == torch.float32
                and v.shape == self.adapter[n].shape
                and bool(torch.isfinite(v).all())
                for n, v in reference.items()
            ),
            "Frozen reference tensor contract differs",
        )
        self.policy_hash = state_hash(self.adapter)
        self.reference_hash = state_hash(self.reference)
        _require(
            self.policy_hash == self.reference_hash == runtime.identity.get("reference_hash"),
            "Replay policy and reference must be the actual sampled parent",
        )
        self.base = _stamp({n: p for n, p in self.params.items() if n not in self.names})
        self.buffers = _stamp(dict(self.model.named_buffers()))
        self.flags = {n: p.requires_grad for n, p in self.params.items()}
        self.grads = {
            n: None if p.grad is None else p.grad.detach().cpu().clone()
            for n, p in self.params.items()
        }
        self.grad_hash = state_hash(self.grads)
        self.modes = [(module, module.training) for module in self.model.modules()]
        self.rng = capture_rng()
        self.rng_hash = state_hash(self.rng)

    def restore(self):
        import torch

        errors = []
        current = dict(self.model.named_parameters())
        objects_changed = current.keys() != self.params.keys() or any(
            current.get(name) is not param for name, param in self.params.items()
        )
        frozen_changed = (
            _stamp({n: p for n, p in current.items() if n not in self.names}) != self.base
            or _stamp(dict(self.model.named_buffers())) != self.buffers
            or current.keys() != self.params.keys()
        )
        adapter_changed, reference_changed = True, True
        try:
            adapter_changed = (
                objects_changed
                or state_hash(_adapter_state(self.model, self.names)) != self.policy_hash
            )
            reference_changed = state_hash(self.reference_object) != self.reference_hash
            with torch.no_grad():
                for name in self.names:
                    # Replacement is a poisoned runtime, not permission to overwrite
                    # an unknown parameter object. Restore original reachable tensors.
                    if current.get(name) is self.params[name] and adapter_changed:
                        current[name].copy_(self.adapter[name].to(current[name].device))
                if reference_changed:
                    self.reference_object.clear()
                    self.reference_object.update(self.reference_tensors)
                    for name, value in self.reference.items():
                        self.reference_object[name].copy_(
                            value.to(self.reference_object[name].device)
                        )
        except BaseException as exc:
            errors.append(dict(error_type=type(exc).__name__, error=str(exc)))
        finally:
            # One failed restoration must not skip the remaining independent state.
            for name, param in self.params.items():
                try:
                    param.requires_grad_(self.flags[name])
                    grad = self.grads[name]
                    param.grad = None if grad is None else grad.to(param.device).clone()
                except BaseException as exc:
                    errors.append(dict(error_type=type(exc).__name__, error=str(exc)))
            for module, training in self.modes:
                module.training = training
            restore_rng(self.rng)
        try:
            restored_policy = state_hash(_adapter_state(self.model, self.names))
        except (KeyError, RuntimeError):
            restored_policy = None
        requires_reload = frozen_changed or objects_changed or bool(errors)
        if requires_reload:
            self.runtime._srf6_replay_poisoned = True
        restored_rng = state_hash(capture_rng())
        restored_grads = state_hash(
            {n: None if p.grad is None else p.grad.detach().cpu() for n, p in self.params.items()}
        )
        return dict(
            policy_before_sha256=self.policy_hash,
            policy_after_sha256=restored_policy,
            reference_sha256=self.reference_hash,
            reference_after_sha256=state_hash(self.reference_object),
            rng_before_sha256=self.rng_hash,
            rng_after_sha256=restored_rng,
            gradients_before_sha256=self.grad_hash,
            gradients_after_sha256=restored_grads,
            adapter_unchanged_during_replay=not adapter_changed,
            reference_unchanged_during_replay=not reference_changed,
            adapter_restored=not objects_changed and restored_policy == self.policy_hash,
            rng_restored=restored_rng == self.rng_hash,
            gradients_restored=restored_grads == self.grad_hash,
            requires_grad_restored=all(
                p.requires_grad == self.flags[n] for n, p in self.params.items()
            ),
            module_modes_restored=all(m.training == mode for m, mode in self.modes),
            frozen_base_buffer_versions_unchanged=not frozen_changed,
            parameter_objects_unchanged=not objects_changed,
            restoration_errors=errors,
            runtime_requires_reload=requires_reload,
            scope="Adapter/reference restored; frozen base/buffers version-checked, not copied",
        )


def _metric(left, right, shape):
    """Compare every element; None is recorded as an unused zero gradient."""
    import torch

    count = math.prod(shape)
    empty = left is None and right is None
    exemplar = right if left is None else left
    if empty:
        a = b = torch.zeros((), dtype=torch.float32)
    else:
        a = torch.zeros_like(exemplar) if left is None else left.detach()
        b = torch.zeros_like(exemplar) if right is None else right.detach()
        _require(tuple(a.shape) == tuple(b.shape) == tuple(shape), "Gradient shape differs")
        _require(a.dtype == b.dtype == torch.float32, "Gradient must be FP32")
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a.double() - b.double()).abs()
    limit = ATOL + RTOL * b.double().abs()
    return dict(
        shape=list(shape),
        elements=count,
        f6_unused=left is None,
        native_unused=right is None,
        finite=finite,
        f6_nonzero_elements=int(torch.count_nonzero(a)),
        native_nonzero_elements=int(torch.count_nonzero(b)),
        violations=int(torch.count_nonzero(~(delta <= limit))) if finite else count,
        max_absolute_difference=float(delta.max()),
        max_tolerance_ratio=float((delta / limit).max()),
        f6_l2=float(torch.linalg.vector_norm(a.double())),
        native_l2=float(torch.linalg.vector_norm(b.double())),
        difference_l2=float(torch.linalg.vector_norm(delta)),
    )


def _advantages(records):
    scores = [_score_row(row)["J"] for row in records]
    _require(
        len(scores) == 8 and all(type(v) in (int, bool) and v in (0, 1) for v in scores),
        "Exactly eight actual binary J scores required",
    )
    mean = sum(scores) / 8
    sd = math.sqrt(sum((v - mean) ** 2 for v in scores) / 8)
    return scores, [(v - mean) / sd if sd else 0.0 for v in scores]


def _boundary(runtime, row, raw, prepared):
    _verify_prepared(raw, prepared)
    tokens, backend = raw["tokens"], raw["backend_sequence_tokens"]
    count = len(tokens)
    _require(
        1 <= count <= 1536 and all(type(v) is int and v >= 0 for v in tokens),
        "Original completion length/token IDs differ",
    )
    _require(
        row["tokens"] == tokens == raw["raw_tokens"] and raw["completion_token_count"] == count,
        "Annotated/raw original tokens differ",
    )
    _require(
        raw["sampler_logprobs"] == row["sampler_logprobs"]
        and len(raw["sampler_logprobs"]) == count,
        "Original sampler vector differs",
    )
    _require(
        backend[:count] == tokens
        and len(backend) >= count
        and raw["backend_padding_count"] == len(backend) - count,
        "Native backend sequence/padding extent differs",
    )
    pad = runtime.processor.tokenizer.pad_token_id
    _require(
        all(token == pad for token in backend[count:]), "Nonpadding tokens after original stop"
    )
    eos_positions = [index for index, token in enumerate(tokens) if token in runtime.eos_ids]
    _require(not eos_positions or eos_positions == [count - 1], "Tokens extend beyond original EOS")
    finish = raw["finish_reason"]
    _require(finish in ("balanced", "eos", "length"), "Unregistered original finish reason")
    _require(finish != "eos" or bool(eos_positions), "EOS finish lacks sampled EOS")
    _require(finish != "length" or count == 1536, "Length termination before registered cap")
    _require(
        raw.get("generation_status") == "COMPLETE" and not raw.get("technical_validation_errors"),
        "Technical-invalid original generation cannot certify replay",
    )
    native = raw["native_prompt_inputs"]
    mask = native.get("attention_mask")
    _require(
        isinstance(mask, list)
        and len(mask) == 1
        and len(mask[0]) == raw["prompt_token_count"]
        and all(v in (0, 1) for v in mask[0]),
        "Original prompt attention mask differs",
    )
    return dict(
        token_count=count,
        prompt_token_count=raw["prompt_token_count"],
        prompt_masked_positions=mask[0].count(0),
        backend_padding_count=len(backend) - count,
        finish_reason=finish,
        eos_positions=eos_positions,
        eos_included=bool(eos_positions),
        native_prompt_inputs_verified=True,
        original_completion_ids_verified=True,
        padding_excluded=True,
        prompt_excluded=True,
        boundary_scope="Exact native prompt and unpadded completion-vector length checked",
    )


def _metric_pass(metric, shape):
    try:
        numbers = (
            "max_absolute_difference",
            "max_tolerance_ratio",
            "f6_l2",
            "native_l2",
            "difference_l2",
        )
        count = math.prod(shape)
        return (
            metric["shape"] == list(shape)
            and type(metric["elements"]) is int
            and metric["elements"] == count
            and metric["finite"] is True
            and type(metric["violations"]) is int
            and metric["violations"] == 0
            and all(
                type(metric[k]) in (int, float) and math.isfinite(metric[k]) and metric[k] >= 0
                for k in numbers
            )
            and metric["max_tolerance_ratio"] <= 1
            and all(
                type(metric[k]) is int and 0 <= metric[k] <= count
                for k in ("f6_nonzero_elements", "native_nonzero_elements")
            )
            and type(metric["f6_unused"]) is bool
            and type(metric["native_unused"]) is bool
            and all(
                (metric[f"{prefix}_nonzero_elements"] == 0) == (metric[f"{prefix}_l2"] == 0)
                and (not metric[f"{prefix}_unused"] or metric[f"{prefix}_nonzero_elements"] == 0)
                for prefix in ("f6", "native")
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _accounting_facts(events, task_id, parent, lengths):
    expected = [
        (purpose, length) for length in lengths for purpose in ("t03_reference", "t03_native")
    ]
    actual, completions, generated, invalid = [], 0, 0, []
    for row in events:
        if (
            row.get("task_id") != task_id
            or row.get("parent") != parent
            or type(row.get("count")) is not int
            or row["count"] < 0
        ):
            invalid.append("Invalid event identity/count")
            continue
        kind, count = row.get("kind"), row["count"]
        if kind == "completion_attempts":
            completions += count
            invalid.append("Generation accounting is forbidden in REPLAY")
        elif kind == "generated_tokens":
            generated += count
            invalid.append("Generation accounting is forbidden in REPLAY")
        elif kind == "extra_forward_sequences":
            metadata = row.get("metadata", {})
            if count != 1 or type(metadata.get("completion_tokens")) is not int:
                invalid.append("Each registered native call must contain one original sequence")
            actual.append((metadata.get("purpose"), metadata.get("completion_tokens")))
        else:
            invalid.append("Unregistered replay accounting kind")
    return dict(
        status="PASS"
        if len(lengths) == 8
        and actual == expected
        and not invalid
        and completions == generated == 0
        else "FAIL",
        original_sequence_lengths=list(lengths),
        reference_calls=sum(p == "t03_reference" for p, _ in actual),
        current_calls=sum(p == "t03_native" for p, _ in actual),
        completion_attempts=completions,
        generated_tokens=generated,
        actual_forward_events=[dict(purpose=p, completion_tokens=n) for p, n in actual],
        invalid_events=invalid,
    )


def _terminal_accounting(session):
    if session.scope == "SYNTHETIC_ONLY":
        return dict(
            scope="SYNTHETIC_ONLY",
            events=copy.deepcopy(session.runtime.fixture_accounting),
            snapshot="AFTER_NUMERICAL_AND_RESTORATION",
            reference=None,
        )
    root = Path(session.runtime.identity["run_root"])
    relative = (
        (Path(session.runtime.identity["audit_root"]) / "accounting" / (session.task_id + ".jsonl"))
        .relative_to(root)
        .as_posix()
    )
    path = safe_evidence(root, relative)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    _require(
        (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "Accounting changed while binding terminal snapshot",
    )
    return dict(
        scope="SERVER_ACTUAL",
        events=[_json(line) for line in payload.splitlines()],
        snapshot="AFTER_NUMERICAL_AND_RESTORATION",
        reference=dict(path=relative, sha256=hashlib.sha256(payload).hexdigest()),
    )


def assess_facts(report):
    """Recompute gates from saved vectors/comparison facts, never trust status."""
    failures = list(report.get("errors", []))
    traces, comparisons = report.get("probability_vectors", []), report.get("comparisons", [])
    try:
        inventory = report["parameter_inventory"]
        names = {item["name"]: item for item in inventory}
        _require(
            bool(names) and len(names) == len(inventory),
            "Complete unique adapter inventory required",
        )
        _require(
            all(
                item.get("dtype") == "torch.float32"
                and isinstance(item.get("shape"), list)
                and all(type(n) is int and n > 0 for n in item["shape"])
                for item in inventory
            ),
            "Registered FP32 adapter shapes required",
        )
        _require(
            len(traces) == 8 and len({r["row_id"] for r in traces}) == 8,
            "All eight original probability vectors required",
        )
        _require(
            all(
                len(row["tokens"])
                == len(row["sampler_logprobs"])
                == len(row["train_logprobs"])
                == len(row["reference_logprobs"])
                and 1 <= len(row["tokens"]) <= 1536
                for row in traces
            ),
            "Per-row original probability vector lengths differ",
        )
        _require(len(comparisons) == 16, "Both registered coefficients required for every row")
        audit = _numerical_audit(traces)
        _require(audit["status"] == "FINITE", "Nonfinite actual token probability/KL")
        _require(
            audit["sampler_train_mean_absolute_difference"] <= 0.01
            and audit["sampler_train_max_absolute_difference"] <= 5
            and audit["tis_truncated_fraction"] <= 0.01,
            "Whole-G8 probability gate failed",
        )
        seen, probe_nonzero, native_nonzero = set(), 0, 0
        expected = dict(zip(report["row_ids"], report["native_advantages"], strict=True))
        _require(set(expected) == {r["row_id"] for r in traces}, "Comparison row identities differ")
        for comparison in comparisons:
            key = comparison["row_id"], comparison["case"]
            _require(
                key not in seen and key[0] in expected and key[1] in CASES,
                "Duplicate/unregistered row comparison",
            )
            seen.add(key)
            coefficient = expected[key[0]] if key[1] == CASES[0] else 1.0
            _require(
                comparison["coefficient"] == coefficient
                and comparison["auxiliary_coefficient"] == 0
                and comparison["kl_coefficient"] == 0.02,
                "Registered numerical coefficient differs",
            )
            loss = comparison["loss"]
            _require(
                all(
                    type(loss[n]) in (int, float) and math.isfinite(loss[n])
                    for n in ("f6", "native")
                ),
                "Nonfinite objective loss",
            )
            _require(
                abs(loss["f6"] - loss["native"]) <= ATOL + RTOL * abs(loss["native"]),
                "Per-sequence loss comparison failed",
            )
            gradients = comparison["parameters"]
            _require(set(gradients) == set(names), "A registered adapter parameter is missing")
            for name, metric in gradients.items():
                _require(
                    _metric_pass(metric, names[name]["shape"]), "Adapter gradient comparison failed"
                )
                nonzero = min(metric["f6_nonzero_elements"], metric["native_nonzero_elements"]) > 0
                if key[1] == CASES[1]:
                    probe_nonzero += int(nonzero)
                else:
                    native_nonzero += int(nonzero)
        restoration = report["restoration"]
        for key in (
            "adapter_restored",
            "rng_restored",
            "gradients_restored",
            "requires_grad_restored",
            "module_modes_restored",
            "frozen_base_buffer_versions_unchanged",
            "adapter_unchanged_during_replay",
            "reference_unchanged_during_replay",
        ):
            _require(restoration.get(key) is True, "Read-only state contract failed: " + key)
        for left, right in (
            ("policy_before_sha256", "policy_after_sha256"),
            ("reference_sha256", "reference_after_sha256"),
            ("rng_before_sha256", "rng_after_sha256"),
            ("gradients_before_sha256", "gradients_after_sha256"),
        ):
            _require(
                _sha(restoration.get(left)) and restoration[left] == restoration.get(right),
                "Restored state content differs",
            )
        _require(restoration.get("runtime_requires_reload") is False, "Poisoned runtime")
        _require(
            restoration.get("parameter_objects_unchanged") is True
            and restoration.get("restoration_errors") == [],
            "Restoration did not complete",
        )
        _require(report.get("records_unchanged") is True, "Original records changed")
        accounting = report["accounting"]
        _require(
            accounting["scope"] == report["scope"]
            and accounting["snapshot"] == "AFTER_NUMERICAL_AND_RESTORATION",
            "Terminal accounting required",
        )
        counted = _accounting_facts(
            accounting["events"],
            report["task_id"],
            report["parent"],
            [len(t["tokens"]) for t in traces],
        )
        _require(counted["status"] == "PASS", "Actual native-forward/generation accounting differs")
        boundaries = report["boundaries"]
        _require(
            len(boundaries) == 8
            and all(
                b["native_prompt_inputs_verified"] is True
                and b["original_completion_ids_verified"] is True
                and b["padding_excluded"] is True
                and b["prompt_excluded"] is True
                and b["token_count"] == len(t["tokens"])
                for b, t in zip(boundaries, traces, strict=True)
            ),
            "Original token boundaries failed",
        )
        status = "FAIL" if failures else "PASS" if probe_nonzero else "INCONCLUSIVE"
        return dict(
            status=status,
            failures=failures,
            probability_audit=audit,
            accounting=counted,
            probe_nonzero_parameter_rows=probe_nonzero,
            native_nonzero_parameter_rows=native_nonzero,
            native_advantage_nonzero_rows=sum(v != 0 for v in report["native_advantages"]),
        )
    except (KeyError, ValueError, TypeError, PermissionError, RuntimeError) as exc:
        failures.append(dict(error_type=type(exc).__name__, error=str(exc)))
        return dict(status="FAIL", failures=failures)


def _execute(session, *, hardware, attempt_ref=None):
    import torch

    output = _plain_path(session.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _require(
        not (output / "STARTED.json").exists() and not (output / RESULT_FILE).exists(),
        "Preserve prior replay attempt; use a new task_id",
    )
    report = dict(
        schema=SCHEMA,
        scope=session.scope,
        parent=session.parent,
        task_id=session.task_id,
        at=now(),
        spec_ref=copy.deepcopy(session.spec_ref),
        context=copy.deepcopy(session.context),
        inputs=copy.deepcopy(session.inputs),
        hardware=hardware,
        execution_attempt_ref=copy.deepcopy(attempt_ref),
        errors=[],
        optimizer_updates=0,
        generated_responses=0,
        training_authorized=False,
        gate_status="NOT_SIGNED",
        tolerances=dict(atol=ATOL, rtol=RTOL),
        probability_vectors=[],
        comparisons=[],
        boundaries=[],
        state_preservation_scope=(
            "Adapter/reference snapshots; frozen base/buffers version sentinels"
        ),
        native_special_tokens=dict(
            eos_ids=list(session.runtime.eos_ids),
            pad_token_id=session.runtime.processor.tokenizer.pad_token_id,
        ),
        accounting_expected_relative_path=(
            (
                Path(session.runtime.identity["audit_root"])
                / "accounting"
                / (session.task_id + ".jsonl")
            )
            .relative_to(Path(session.runtime.identity["run_root"]))
            .as_posix()
            if session.scope == "SERVER_ACTUAL"
            else None
        ),
    )
    write_once(
        output / "STARTED.json",
        {
            k: v
            for k, v in report.items()
            if k not in ("probability_vectors", "comparisons", "boundaries")
        },
    )
    saved, interrupted = None, None
    try:
        records, raw = session.records, session.raw_records
        _require(len(records) == len(raw) == 8, "Exactly one complete original G8 required")
        report["selected_records_sha256"], report["selected_raw_records_sha256"] = (
            digest(records),
            digest(raw),
        )
        report["row_ids"] = [row["row_id"] for row in records]
        report["qid"] = records[0]["qid"]
        _require(
            len(set(report["row_ids"])) == 8 and len({r["qid"] for r in records}) == 1,
            "Duplicate or mixed original G8",
        )
        report["J"], report["native_advantages"] = _advantages(records)
        saved = _Preserve(session.runtime, session.reference)
        params = [saved.params[name] for name in saved.names]
        report["parameter_inventory"] = [
            dict(name=n, shape=list(p.shape), dtype=str(p.dtype))
            for n, p in zip(saved.names, params, strict=True)
        ]
        session.runtime.model.eval()
        prepared = session.runtime.prepare_input(
            raw[0]["model_input"],
            Path(session.runtime.identity["run_root"]),
            protocol=raw[0]["protocol"],
        )
        for index, (record, original) in enumerate(zip(records, raw, strict=True)):
            report["boundaries"].append(_boundary(session.runtime, record, original, prepared))
            item = copy.deepcopy(record)
            item["prepared"] = prepared
            with torch.no_grad():
                refs = session.runtime.batch_reference_forward([item], session.reference)
            with torch.enable_grad():
                currents = session.runtime.batch_sequence_forward(
                    [item], purpose="t03_native", grad=True
                )
                _require(len(currents) == len(refs) == 1, "Microbatch=1 must return one sequence")
                current, reference = currents[0], refs[0]
                sampler = torch.tensor(
                    original["sampler_logprobs"], dtype=torch.float32, device=current.device
                )
                _require(
                    current.ndim == 1
                    and current.shape == reference.shape == sampler.shape
                    and current.numel() == len(original["tokens"]),
                    "Native completion vector extent differs",
                )
                report["probability_vectors"].append(
                    dict(
                        row_id=record["row_id"],
                        qid=record["qid"],
                        tokens=original["tokens"][:],
                        sampler_logprobs=sampler.detach().cpu().tolist(),
                        train_logprobs=current.detach().cpu().tolist(),
                        reference_logprobs=reference.detach().cpu().tolist(),
                    )
                )
                for case_index, case in enumerate(CASES):
                    coefficient = report["native_advantages"][index] if case_index == 0 else 1.0
                    f6 = sequence_loss(current, sampler, reference, coefficient, aux_adv=0.0)
                    native = sequence_objective(
                        current, sampler, reference, coefficient, kl_coefficient=0.02
                    )
                    grads_f6 = torch.autograd.grad(
                        f6["loss"], params, retain_graph=True, allow_unused=True
                    )
                    grads_native = torch.autograd.grad(
                        native["loss"], params, retain_graph=case_index == 0, allow_unused=True
                    )
                    comparison = dict(
                        row_id=record["row_id"],
                        case=case,
                        coefficient=coefficient,
                        auxiliary_coefficient=0,
                        kl_coefficient=0.02,
                        loss=dict(
                            f6=float(f6["loss"].detach()), native=float(native["loss"].detach())
                        ),
                        parameters={},
                    )
                    for name, param, left, right in zip(
                        saved.names, params, grads_f6, grads_native, strict=True
                    ):
                        comparison["parameters"][name] = _metric(left, right, param.shape)
                    report["comparisons"].append(comparison)
                    del grads_f6, grads_native, f6, native
                del current, reference, currents, refs
    except BaseException as exc:
        report["errors"].append(dict(error_type=type(exc).__name__, error=str(exc)))
        if not isinstance(exc, Exception):
            interrupted = exc
    finally:
        if saved is not None:
            try:
                report["restoration"] = saved.restore()
                if hasattr(session, "refresh_adapter_versions"):
                    session.refresh_adapter_versions()
            except BaseException as exc:
                session.runtime._srf6_replay_poisoned = True
                report["errors"].append(
                    dict(error_type=type(exc).__name__, error="State restoration: " + str(exc))
                )
        try:
            report["records_unchanged"] = digest(session.records) == report.get(
                "selected_records_sha256"
            ) and digest(session.raw_records) == report.get("selected_raw_records_sha256")
        except (TypeError, ValueError):
            report["records_unchanged"] = False
    try:
        report["accounting"] = _terminal_accounting(session)
    except (OSError, PermissionError, ValueError, AttributeError) as exc:
        report["errors"].append(
            dict(error_type=type(exc).__name__, error="Terminal accounting: " + str(exc))
        )
    report["assessment"] = assess_facts(report)
    report["numerical_result"] = report["assessment"]["status"]
    report["status"] = (
        report["numerical_result"] if session.scope == "SERVER_ACTUAL" else "SYNTHETIC_ONLY"
    )
    lengths = [len(row["tokens"]) for row in report["probability_vectors"]]
    report["observed_coverage"] = dict(
        sequence_lengths=lengths,
        min_tokens=min(lengths, default=None),
        max_tokens=max(lengths, default=None),
        tokens_1536="OBSERVED" if 1536 in lengths else "NOT_OBSERVED",
        eos="OBSERVED" if any(b["eos_included"] for b in report["boundaries"]) else "NOT_OBSERVED",
        backend_padding="OBSERVED"
        if any(b["backend_padding_count"] for b in report["boundaries"])
        else "NOT_OBSERVED",
        actual_native_advantage="NONZERO_OBSERVED"
        if any(report.get("native_advantages", []))
        else "ALL_ZERO_OR_NOT_AVAILABLE",
        positive_probe="Numerical coverage only; not reward, new data or an optimizer intervention",
    )
    report = _safe(report)
    write_once(output / RESULT_FILE, report)
    if interrupted is not None:
        raise interrupted
    return report


def run_synthetic(runtime, records, raw_records, reference, *, output_directory):
    """CPU fixture only; this entry can never emit a SERVER_ACTUAL receipt."""
    import torch

    _require(
        all(p.device.type == "cpu" for p in runtime.model.parameters()),
        "Synthetic entry is CPU-only",
    )
    output = _plain_path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    session = SimpleNamespace(
        runtime=runtime,
        records=records,
        raw_records=raw_records,
        reference=reference,
        parent=runtime.identity["parent_id"],
        task_id="SYNTHETIC_ONLY",
        scope="SYNTHETIC_ONLY",
        spec_ref=None,
        context={"fixture": "CPU_SYNTHETIC_ONLY"},
        inputs=[],
        output_directory=output,
    )
    _require(not torch.cuda.is_initialized(), "Synthetic fixture must not initialize CUDA")
    return _execute(session, hardware={"kind": "CPU_SYNTHETIC_ONLY"})


def run_session(session):
    """Execute only a live original-token session created by open_replay."""
    import torch

    from mm_dev.runtime import actual_cuda_identity

    from .replay import RULES, ReplaySession

    _require(
        type(session) is ReplaySession and session.scope == "SERVER_ACTUAL",
        "Authenticated actual replay session required",
    )
    session.assert_active()
    _require(
        session.spec["rules"] == RULES and RULES["atol"] == ATOL and RULES["rtol"] == RTOL,
        "Numerical thresholds differ from frozen replay specification",
    )
    for key, expected in (
        ("positive_probe", 1),
        ("microbatch_sequences", 1),
        ("auxiliary_coefficient", 0),
        ("kl_coefficient", 0.02),
        ("sampler_current_mean_maximum", 0.01),
        ("sampler_current_max_maximum", 5.0),
        ("tis_cap_fraction_maximum", 0.01),
        ("tis_cap", 2.0),
    ):
        _require(RULES[key] == expected, "Frozen T03 contract differs: " + key)
    attempt_ref = _attempt_reference(
        Path(session.runtime.identity["run_root"]),
        Path(session.runtime.identity["audit_root"]),
        parent=session.parent,
        task_id=session.task_id,
        phase_pin=session.runtime.identity["phase_identity_sha256"],
        spec_ref=session.spec_ref,
    )
    _require(
        torch.cuda.is_available()
        and torch.cuda.device_count() == 1
        and all(p.device.type == "cuda" for p in session.runtime.model.parameters()),
        "Actual replay requires one GPU and CUDA model parameters",
    )
    return _execute(session, hardware=actual_cuda_identity(), attempt_ref=attempt_ref)


def run(root, parent, task_id, *, phase_identity_sha256, replay_spec_sha256):
    """Production parent loader and exclusive retained output; no arbitrary rows."""
    from . import replay, runtime

    root = _plain_path(root)
    _require(
        parent in ("b0", "b1", "b2")
        and isinstance(task_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id),
        "Registered parent and safe task ID required",
    )
    spec_ref = dict(path=replay.SPEC_PATH, sha256=replay_spec_sha256)
    replay.verify_spec(root, spec_ref, phase_identity_sha256=phase_identity_sha256)
    audit = _plain_path(root / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS" / task_id)
    audit.mkdir(parents=True, exist_ok=False)
    with (audit / ".run.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = dict(
            parent=parent,
            task_id=task_id,
            phase_identity_sha256=phase_identity_sha256,
            spec_ref=spec_ref,
            at=now(),
            generated_responses=0,
            optimizer_updates=0,
        )
        write_once(audit / "ATTEMPT.json", start)
        try:
            loaded, optimizer, scheduler = runtime.load(
                root,
                parent,
                task_id,
                mode="REPLAY",
                phase="PILOT_V1",
                phase_identity_sha256=phase_identity_sha256,
                audit_root=audit,
            )
            _require(
                optimizer is None and scheduler is None, "REPLAY cannot create optimizer state"
            )
            with replay.open_replay(loaded, spec_ref) as session:
                return run_session(session)
        except BaseException as exc:
            write_once(
                audit / "FAILURE.json",
                {
                    **start,
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "training_authorized": False,
                },
            )
            raise


def summary(root, *, phase_identity_sha256, replay_spec_sha256, receipts):
    """Read bound real receipts and recompute numerical conditions for all parents."""
    from . import replay

    root = _plain_path(root)
    spec_ref = dict(path=replay.SPEC_PATH, sha256=replay_spec_sha256)
    spec = replay.verify_spec(root, spec_ref, phase_identity_sha256=phase_identity_sha256)
    results, seen, result_paths = [], set(), set()
    for ref in receipts:
        _require(
            isinstance(ref, dict) and set(ref) == {"path", "sha256"} and _sha(ref["sha256"]),
            "Explicit saved result reference required",
        )
        path = safe_evidence(root, ref["path"])
        _require(
            ref["path"] not in result_paths
            and path.name == RESULT_FILE
            and path.parent.name == "T03",
            "Duplicate or misplaced replay result",
        )
        result_paths.add(ref["path"])
        payload = path.read_bytes()
        _require(
            hashlib.sha256(payload).hexdigest() == ref["sha256"], "Numerical receipt bytes changed"
        )
        report = _json(payload)
        _require(
            report.get("schema") == SCHEMA and report.get("scope") == "SERVER_ACTUAL",
            "Synthetic/unknown receipt is not actual T03 evidence",
        )
        parent = report.get("parent")
        _require(
            parent in spec["groups"],
            "Result must name a registered parent",
        )
        seen.add(parent)
        expected_execution = dict(
            phase="PILOT_V1",
            status="BOUND_NOT_AUTHORIZED",
            training_allowed=False,
            deployment=spec["context"]["deployment"],
            identity=spec["context"],
            identity_sha256=phase_identity_sha256,
            origin_identity_sha256=spec["origin_identity_sha256"],
        )
        _require(
            report.get("spec_ref") == spec_ref
            and report.get("tolerances") == dict(atol=ATOL, rtol=RTOL)
            and report.get("context", {}).get("producer") == spec["producer_context"]
            and report.get("context", {}).get("execution") == expected_execution,
            "Saved replay provenance/rules differ",
        )
        raw, records, policy, inputs = replay._parent_evidence(root, spec, parent)
        _require(
            report.get("selected_records_sha256") == digest(records)
            and report.get("selected_raw_records_sha256") == digest(raw),
            "Saved replay original inputs differ",
        )
        task_id = report.get("task_id")
        _require(
            isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", task_id),
            "Actual replay task ID required",
        )
        audit = path.parent.parent
        attempt_ref = _attempt_reference(
            root,
            audit,
            parent=parent,
            task_id=task_id,
            phase_pin=phase_identity_sha256,
            spec_ref=spec_ref,
        )
        _require(
            report.get("execution_attempt_ref") == attempt_ref,
            "Result differs from its immutable pre-forward ATTEMPT",
        )
        execution_relative = (
            (audit / "runtime_identity" / (task_id + ".json")).relative_to(root).as_posix()
        )
        execution_bytes = safe_evidence(root, execution_relative).read_bytes()
        execution = _json(execution_bytes)
        execution_ref = dict(
            path=execution_relative, sha256=hashlib.sha256(execution_bytes).hexdigest()
        )
        for key, expected in (
            ("run_root", str(root)),
            ("audit_root", str(audit)),
            ("task_id", task_id),
            ("parent_id", parent),
            ("reference_parent", parent),
            ("step", 0),
            ("phase", "PILOT_V1"),
            ("execution_mode", "REPLAY"),
            ("phase_identity_sha256", phase_identity_sha256),
            ("source_sha256", spec["context"]["source_manifest"]["sha256"]),
            ("source_hash", spec["context"]["source_manifest"]["sha256"]),
            ("origin_identity_sha256", spec["origin_identity_sha256"]),
            ("data_hash", spec["producer_context"]["data_manifest"]["sha256"]),
            ("policy_hash", policy),
            ("reference_hash", policy),
            ("parent_content_hash", spec["groups"][parent]["policy_hash"]),
            ("fresh_optimizer", False),
            ("dtype", "float32"),
        ):
            _require(
                type(execution.get(key)) is type(expected) and execution[key] == expected,
                "Execution runtime identity differs: " + key,
            )
        expected_inputs = list(
            {r["path"]: r for r in [spec_ref, *spec["inputs"], *inputs, execution_ref]}.values()
        )
        _require(report.get("inputs") == expected_inputs, "Saved replay source references differ")
        if "parameter_inventory" in report:
            _require(
                report["parameter_inventory"] == execution.get("trainable_parameters"),
                "Complete actual loaded adapter parameter inventory differs",
            )
        native_config = execution.get("generation_config_expanded", {})
        eos = native_config.get("eos_token_id")
        eos = eos if isinstance(eos, list) else [eos]
        _require(
            report.get("native_special_tokens")
            == dict(eos_ids=eos, pad_token_id=native_config.get("pad_token_id")),
            "Native special token identities differ",
        )
        expected_ids = [r["row_id"] for r in records]
        scores, advantages = _advantages(records)
        _require(
            report.get("row_ids") == expected_ids
            and report.get("J") == scores
            and report.get("native_advantages") == advantages
            and report.get("qid") == spec["groups"][parent]["qid"],
            "Saved native advantages or row identities differ",
        )
        for trace, record, original in zip(
            report.get("probability_vectors", []), records, raw, strict=False
        ):
            _require(
                trace.get("row_id") == record["row_id"]
                and trace.get("qid") == record["qid"]
                and trace.get("tokens") == original["tokens"]
                and trace.get("sampler_logprobs") == original["sampler_logprobs"],
                "Saved original vectors differ",
            )
        if "restoration" in report:
            _require(
                report["restoration"].get("policy_before_sha256") == policy,
                "Replay actual parent tensor identity differs",
            )
        actual = report.get("hardware", {})
        _require(
            actual == execution.get("hardware")
            and actual.get("cuda_visible_device_count") == 1
            and bool(actual.get("cuda_uuid"))
            and "RTX PRO 6000" in actual.get("cuda_name", "").upper(),
            "Actual GPU receipt required",
        )
        if "accounting" in report:
            accounting = report["accounting"]
            relative = (audit / "accounting" / (task_id + ".jsonl")).relative_to(root).as_posix()
            terminal = safe_evidence(root, relative).read_bytes()
            _require(
                accounting.get("reference")
                == dict(path=relative, sha256=hashlib.sha256(terminal).hexdigest())
                and accounting.get("events") == [_json(line) for line in terminal.splitlines()],
                "Actual terminal accounting differs from saved replay",
            )
        start = _json(
            safe_evidence(root, (path.parent / "STARTED.json").relative_to(root)).read_bytes()
        )
        for key in (
            "scope",
            "parent",
            "task_id",
            "spec_ref",
            "context",
            "inputs",
            "hardware",
            "tolerances",
            "native_special_tokens",
            "optimizer_updates",
            "generated_responses",
            "accounting_expected_relative_path",
            "execution_attempt_ref",
        ):
            _require(
                start.get(key) == report.get(key), "Result differs from pre-forward input binding"
            )
        _require(
            type(report.get("optimizer_updates")) is int
            and report["optimizer_updates"] == 0
            and type(report.get("generated_responses")) is int
            and report["generated_responses"] == 0,
            "T03 cannot generate or update",
        )
        facts = assess_facts(report)
        observed = report.get("probability_vectors", [])
        lengths = [len(row["tokens"]) for row in observed]
        results.append(
            dict(
                parent=parent,
                receipt=ref,
                assessment=facts,
                recorded_status=report.get("status"),
                observed_coverage=dict(
                    sequence_lengths=lengths,
                    tokens_1536="OBSERVED" if 1536 in lengths else "NOT_OBSERVED",
                    eos="OBSERVED"
                    if any(token in eos for row in observed for token in row["tokens"])
                    else "NOT_OBSERVED",
                    actual_native_advantage_nonzero_rows=sum(v != 0 for v in advantages),
                ),
            )
        )
    attempts, unreported, unknown, failed_attempts = [], [], [], []
    directory = root / "PHASE_RUNS/PILOT_V1/REPLAY_RUNS"
    if directory.exists():
        for candidate in sorted(_plain_path(directory).iterdir()):
            candidate = _plain_path(candidate)
            _require(candidate.is_dir(), "Unexpected entry in replay execution directory")
            attempt_path = candidate / "ATTEMPT.json"
            if not attempt_path.exists():
                entries = sorted(p.name for p in candidate.iterdir())
                if entries:
                    unknown.append(candidate.name)
                    attempts.append(
                        dict(
                            task_id=candidate.name,
                            parent=None,
                            status="ORPHAN_WITHOUT_ATTEMPT",
                            entries=entries,
                        )
                    )
                continue
            attempt = _json(safe_evidence(root, attempt_path.relative_to(root)).read_bytes())
            if attempt.get("spec_ref") != spec_ref:
                continue
            _require(
                attempt.get("phase_identity_sha256") == phase_identity_sha256
                and attempt.get("task_id") == candidate.name
                and attempt.get("parent") in spec["groups"],
                "Replay attempt identity differs",
            )
            result_path = (candidate / "T03" / RESULT_FILE).relative_to(root).as_posix()
            failure_path = candidate / "FAILURE.json"
            item = dict(task_id=candidate.name, parent=attempt["parent"], result_path=result_path)
            if failure_path.exists():
                failure_bytes = safe_evidence(root, failure_path.relative_to(root)).read_bytes()
                item["failure"] = _json(failure_bytes)
                item["failure_ref"] = dict(
                    path=failure_path.relative_to(root).as_posix(),
                    sha256=hashlib.sha256(failure_bytes).hexdigest(),
                )
                failed_attempts.append(candidate.name)
            if (candidate / "T03" / RESULT_FILE).exists():
                if result_path not in result_paths:
                    unreported.append(result_path)
            elif not failure_path.exists():
                unknown.append(candidate.name)
            attempts.append(item)
    states = [r["assessment"]["status"] for r in results]
    status = (
        "FAIL"
        if "FAIL" in states or failed_attempts
        else "INCOMPLETE"
        if seen != set(spec["groups"]) or unknown or unreported
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in states
        else "PASS"
    )
    return dict(
        schema=SCHEMA,
        scope="SERVER_ACTUAL",
        status=status,
        spec_ref=spec_ref,
        parents=results,
        missing_parents=sorted(set(spec["groups"]) - seen),
        attempts=attempts,
        unknown_attempts=unknown,
        unreported_results=unreported,
        training_authorized=False,
        gate_status="NOT_SIGNED",
        generated_responses=0,
        optimizer_updates=0,
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "summary"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--root", type=Path, required=True)
        cmd.add_argument("--phase-identity-sha256", required=True)
        cmd.add_argument("--spec-sha256", required=True)
    commands.choices["run"].add_argument("--parent", choices=("b0", "b1", "b2"), required=True)
    commands.choices["run"].add_argument("--task-id", required=True)
    commands.choices["summary"].add_argument(
        "--receipt", nargs=2, action="append", default=[], metavar=("PATH", "SHA256")
    )
    args = parser.parse_args(argv)
    kwargs = dict(
        phase_identity_sha256=args.phase_identity_sha256, replay_spec_sha256=args.spec_sha256
    )
    result = (
        run(args.root, args.parent, args.task_id, **kwargs)
        if args.command == "run"
        else summary(
            args.root, receipts=[dict(path=p, sha256=h) for p, h in args.receipt], **kwargs
        )
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
