"""Bound T03 original-token access, without sampling or training authorization.

The immutable rule can be frozen before all parents have produced their first
G8. Opening a parent binds only that parent's original first G8; it never needs
the complete census. S0 records retain their original source identity. The
current PILOT source is recorded separately as the diagnostic executor.
"""

from __future__ import annotations

import copy
import fcntl
import re
from contextlib import contextmanager
from pathlib import Path

from mm_core.training import state_hash, trainable_state

from . import census, census_evidence
from .common import digest, file_hash, read, safe_evidence, write_once

SPEC_PATH = "PHASES/PILOT_V1/T03_SPEC.json"
RULES = {
    "selection": "first qid in sorted frozen DEV_CAL public; all original draws 0..7",
    "parents": ["b0", "b1", "b2"],
    "draws_per_parent": 8,
    "new_generated_responses": 0,
    "optimizer_updates": 0,
    "microbatch_sequences": 1,
    "coefficients": ["actual_native_J_advantage", "fixed_positive_probe"],
    "positive_probe": 1,
    "probe_interpretation": "numerical coverage only, not actual reward or intervention",
    "probe_requires_nonzero_finite_adapter_gradient_per_parent": True,
    "missing_probe_coverage_status": "INCONCLUSIVE",
    "atol": 1e-6,
    "rtol": 1e-5,
    "comparison": "loss and every adapter gradient element",
    "auxiliary_coefficient": 0,
    "kl_coefficient": 0.02,
    "sampler_current_mean_maximum": 0.01,
    "sampler_current_max_maximum": 5.0,
    "tis_cap": 2.0,
    "tis_cap_fraction_maximum": 0.01,
    "probability_mean_and_tis_denominator": "all completion tokens in the selected original G8",
    "nonfinite": "FAIL",
    "reference_current": "separate full vectors, tails and KL; no silent clipping",
    "trajectory": "original completion IDs including actual EOS; exclude prompt and padding",
    "state_preservation": (
        "adapter snapshot; frozen base/buffer version sentinels; RNG, modes, grads"
    ),
    "frozen_base_buffer_mutation": "FAIL; poison runtime; discard/reload; not claimed restored",
}
_ACTIVE = {}
_RUNTIME_ACCESS = object()


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _source(root, pin):
    from .runtime import verify_runtime_source

    return verify_runtime_source(root, phase="PILOT_V1", phase_identity_sha256=pin)


def _layout(root, source):
    reader = census_evidence._Reader(root)
    origin = reader.read("IDENTITY.json")
    _require(
        reader.inputs["IDENTITY.json"]["sha256"] == source["origin_identity_sha256"],
        "Replay origin identity changed",
    )
    config = reader.read("CONFIG.json")
    public, tasks = census.census_inputs(root)
    for relative in ("DATA/DEV_CAL/MODEL_INPUTS.jsonl", "DATA/DEV_CAL/TASKS_PRIVATE.jsonl"):
        reader.read(relative, lines=True)
    frozen = reader.read("CENSUS_SPEC_FROZEN.json")
    _require(
        census.validate_census_spec(root, config, public, tasks) == frozen, "Census spec changed"
    )
    ready = reader.read("PREFLIGHT/CENSUS_READY.json")
    _require(
        ready.get("context") == origin
        and ready.get("status") == "READY_FOR_CENSUS"
        and ready.get("execution_origin") == "SERVER_ACTUAL"
        and ready.get("phase") == "DEV_CAL"
        and ready.get("optimizer_updates") == 0,
        "Replay requires actual S0 census preflight",
    )
    # The phase verifier checks the S0 upstream receipt's transitive dependencies.
    upstream = source["identity"]["upstream"]
    _require(
        any(
            entry["producer_phase"] == "CENSUS_S0"
            and entry["identity_sha256"] == source["origin_identity_sha256"]
            and reader.inputs["PREFLIGHT/CENSUS_READY.json"] in entry["evidence"]
            for entry in upstream
        ),
        "Replay preflight is not in the registered PILOT upstream",
    )
    return reader, origin, config, public, tasks, frozen


def build_spec(root, *, phase_identity_sha256):
    """Freeze selection and criteria only; missing parent raw is not a substitute."""
    root = Path(root).resolve(strict=True)
    source = _source(root, phase_identity_sha256)
    reader, origin, config, public, tasks, frozen = _layout(root, source)
    first = public[0]
    return {
        "schema": "srf6-t03-original-token-spec-v1",
        "phase": "PILOT_V1",
        "status": "REGISTERED_REPLAY_ONLY_NOT_TRAINING_AUTHORIZED",
        "training_allowed": False,
        "context": source["identity"],
        "phase_identity_sha256": source["identity_sha256"],
        "producer_context": origin,
        "origin_identity_sha256": source["origin_identity_sha256"],
        "rules": copy.deepcopy(RULES),
        "groups": {
            p: census.group_spec(config, p, first, tasks[first["qid"]], frozen)
            for p in census.PARENTS
        },
        "inputs": [reader.inputs[p] for p in sorted(reader.inputs)],
    }


def freeze_spec(root, *, phase_identity_sha256):
    spec = build_spec(root, phase_identity_sha256=phase_identity_sha256)
    directory = Path(root) / "PHASES/PILOT_V1"
    _require(
        directory.resolve(strict=True) == directory and not directory.is_symlink(),
        "Unsafe phase directory",
    )
    destination = Path(root) / SPEC_PATH
    write_once(destination, spec)
    return {"path": SPEC_PATH, "sha256": file_hash(destination)}


def verify_spec(root, spec_ref, *, phase_identity_sha256):
    _require(
        isinstance(spec_ref, dict)
        and set(spec_ref) == {"path", "sha256"}
        and spec_ref["path"] == SPEC_PATH,
        "Exact registered T03 spec reference required",
    )
    path = safe_evidence(root, SPEC_PATH)
    _require(file_hash(path) == spec_ref["sha256"], "Replay spec bytes changed")
    spec = read(path)
    _require(
        spec == build_spec(root, phase_identity_sha256=phase_identity_sha256),
        "Replay rules or provenance changed",
    )
    return spec


def _parent_evidence(root, spec, parent):
    """Bind only the fixed first group, even while its worker is still running."""
    reader = census_evidence._Reader(root)
    public, tasks = census.census_inputs(root)
    group_spec = spec["groups"][parent]
    qid = public[0]["qid"]
    _require(group_spec["qid"] == qid, "Replay cannot select another qid")
    frozen = reader.read("CENSUS_SPEC_FROZEN.json")
    attempt = reader.read(f"CENSUS/attempts/{parent}/{qid}.json")
    task_id = attempt.get("task_id", "")
    match = re.fullmatch(rf"CENSUS-{parent}-0-of-([1-9][0-9]*)", task_id)
    _require(
        match is not None
        and int(match.group(1)) <= len(public)
        and attempt.get("status") == "STARTED"
        and attempt.get("spec") == group_spec,
        "Replay requires the original first-worker attempt",
    )
    original_runtime = reader.read(f"runtime_identity/{task_id}.json")
    tensor_hash = census_evidence._runtime_identity(
        original_runtime, task_id, parent, root, spec["producer_context"], frozen
    )
    raw = reader.read(f"CENSUS/raw/{parent}/{qid}.jsonl", lines=True)
    census_evidence._raw_identity(raw, original_runtime, task_id, parent, root, group_spec)
    records = census.annotate_group(raw, tasks[qid], group_spec)
    group = reader.read(f"CENSUS/groups/{parent}/{qid}.json")
    _require(
        group.get("status") == "COMPLETE"
        and group.get("spec") == group_spec
        and group.get("records") == records
        and group.get("credit") == census.counterfactual(records, group_spec),
        "Replay original raw and complete annotated group differ",
    )
    alignment = reader.read(f"CENSUS/alignment/{task_id}.json")
    census_evidence._alignment(alignment, group_spec, records, tensor_hash)
    return raw, records, tensor_hash, list(reader.inputs.values())


def _tensor_stamp(values):
    return {
        name: (
            id(value),
            value._version,
            tuple(value.shape),
            value.dtype,
            value.device,
            value.requires_grad,
        )
        for name, value in values.items()
    }


class ReplaySession:
    """Only open_replay creates an active session; constructing a lookalike grants nothing."""

    def assert_active(self):
        from .runtime import EXECUTION_IDENTITY_FIELDS, _check_loaded_source_identity

        r = self.runtime
        _require(_ACTIVE.get(id(r)) is self and not self._lock.closed, "Inactive replay session")
        _require(
            not getattr(r, "_srf6_replay_poisoned", False),
            "Poisoned replay runtime requires reload",
        )
        _require(
            digest(self.spec) == self._spec_digest and self.spec_ref == self._spec_ref,
            "Active replay specification changed",
        )
        _require(
            _check_loaded_source_identity(r) == "PILOT_V1" and r.execution_mode() == "REPLAY",
            "Replay phase/mode changed",
        )
        _require(
            all(
                r.identity.get(k) == r._srf6_execution_identity[k]
                for k in EXECUTION_IDENTITY_FIELDS
            ),
            "Replay execution identity changed",
        )
        _require(
            r.identity["step"] == 0
            and r.identity["parent_id"] == self.parent
            and r.identity["policy_hash"] == self._policy_hash
            and r.identity["reference_hash"] == self._policy_hash,
            "Replay parent policy identity changed",
        )
        params = dict(r.model.named_parameters())
        _require(
            _tensor_stamp({n: p for n, p in params.items() if n not in self._adapter_names})
            == self._base_stamp
            and _tensor_stamp(dict(r.model.named_buffers())) == self._buffer_stamp,
            "Frozen base/buffers changed; runtime must be discarded",
        )
        _require(
            _tensor_stamp({n: params[n] for n in self._adapter_names}) == self._adapter_stamp,
            "Replay adapter changed outside a verified restoration boundary",
        )

    def refresh_adapter_versions(self):
        _require(_ACTIVE.get(id(self.runtime)) is self, "Inactive replay session")
        _require(
            state_hash(trainable_state(self.runtime.model)) == self._policy_hash,
            "Replay must retain the original parent parameters",
        )
        params = dict(self.runtime.model.named_parameters())
        self._adapter_stamp = _tensor_stamp({n: params[n] for n in self._adapter_names})
        self.assert_active()

    def check_input(self, model_input, root, protocol):
        self.assert_active()
        _require(
            Path(root) == Path(self.runtime.identity["run_root"])
            and protocol in ("structured", "structured_v4"),
            "Replay input root/protocol changed",
        )
        _require(
            model_input == self._model_input, "Replay may only access its original model input"
        )

    def bind_prepared(self, prepared, *, access=None):
        self.assert_active()
        _require(access is _RUNTIME_ACCESS, "Only the verified runtime may bind prepared inputs")
        route, inputs = prepared["routing"], prepared["inputs"]
        _require(
            route.get("input_tensor_hash") == self._raw_binding[0]["prompt_tensor_hash"],
            "Native replay prompt tensor hash differs",
        )
        native = {
            k: v.detach().cpu().tolist()
            for k, v in inputs.items()
            if k != "pixel_values" and hasattr(v, "detach")
        }
        _require(
            all(row["native_prompt_inputs"] == native for row in self._raw_binding),
            "Native replay prompt tokens/masks/grid differ",
        )
        previous = self._prepared.get(id(prepared))
        if previous is not None:
            self.check_prepared(prepared)
        self._prepared[id(prepared)] = (prepared, digest(route), _tensor_stamp(inputs))

    def check_prepared(self, prepared):
        self.assert_active()
        entry = self._prepared.get(id(prepared))
        _require(entry is not None and entry[0] is prepared, "Unbound preconstructed replay input")
        _require(
            entry[1] == digest(prepared["routing"])
            and entry[2] == _tensor_stamp(prepared["inputs"]),
            "Prepared replay tensors or routing changed",
        )

    def check_records(self, records):
        self.assert_active()
        _require(len(records) == 1, "Replay uses one original sequence per forward")
        for row in records:
            item = {k: v for k, v in row.items() if k != "prepared"}
            _require(
                digest(item) in self._records,
                "Replay record differs from its original S0 trajectory",
            )
            if "prepared" in row:
                self.check_prepared(row["prepared"])

    def check_batch(self, prepared, token_rows):
        self.check_prepared(prepared)
        _require(
            len(token_rows) == 1 and list(token_rows[0]) in self._tokens,
            "Replay tokens are not the original completion",
        )

    @contextmanager
    def native_call(self, inputs, *, completion_start, lengths, purpose, grad, access=None):
        self.assert_active()
        _require(access is _RUNTIME_ACCESS, "Only the verified runtime may open a native call")
        _require(self._native is None, "Nested replay forward is forbidden")
        self._native = (inputs, completion_start, lengths, purpose, grad)
        try:
            yield
        finally:
            self._native = None

    def check_native(self, inputs, *, completion_start, lengths, purpose, grad):
        self.assert_active()
        allowed = self._native
        _require(
            allowed is not None
            and allowed[0] is inputs
            and allowed[1:] == (completion_start, lengths, purpose, grad),
            "Native forward requires an original-token replay call",
        )

    def check_reference(self, reference):
        self.assert_active()
        stamp = _tensor_stamp(reference)
        cached = getattr(self, "_reference_cache", None)
        if cached is not None and cached[0] is reference and cached[1] == stamp:
            return
        _require(
            state_hash(reference) == self._policy_hash, "Replay reference is not the frozen parent"
        )
        self._reference_cache = (reference, stamp)


def active_session(runtime):
    session = _ACTIVE.get(id(runtime))
    _require(type(session) is ReplaySession, "REPLAY requires open_replay and its live lock")
    session.assert_active()
    return session


@contextmanager
def open_replay(runtime, spec_ref):
    from .runtime import EXECUTION_IDENTITY_FIELDS, _audit_directory, _check_loaded_source_identity

    _require(
        runtime.execution_mode() == "REPLAY"
        and _check_loaded_source_identity(runtime) == "PILOT_V1",
        "Only PILOT REPLAY may open original tokens",
    )
    _require(
        hasattr(runtime, "_srf6_source_context") and hasattr(runtime, "_srf6_execution_identity"),
        "Actual loaded replay source is required",
    )
    _require(
        all(
            runtime.identity.get(k) == runtime._srf6_execution_identity[k]
            for k in EXECUTION_IDENTITY_FIELDS
        ),
        "Replay execution identity changed",
    )
    _require(
        not getattr(runtime, "_srf6_replay_poisoned", False),
        "Poisoned replay runtime requires reload",
    )
    root = Path(runtime.identity["run_root"])
    audit = _audit_directory(root, "PILOT_V1", runtime.identity["audit_root"])
    source = _source(root, runtime.identity["phase_identity_sha256"])
    _require(
        source == runtime._srf6_source_context,
        "Replay loaded source differs from actual deployment",
    )
    spec = verify_spec(root, spec_ref, phase_identity_sha256=source["identity_sha256"])
    parent = runtime.identity["parent_id"]
    _require(
        parent in census.PARENTS and runtime.identity["step"] == 0,
        "Replay requires a frozen step0 parent",
    )
    _require(id(runtime) not in _ACTIVE, "Runtime already has an active replay session")
    output = audit / "T03"
    _require(
        output.resolve(strict=False) == output and not output.is_symlink(),
        "Unsafe replay output directory",
    )
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".replay.lock"
    _require(not lock_path.is_symlink(), "Symlinked replay lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw, records, policy_hash, inputs = _parent_evidence(root, spec, parent)
        receipt_path = (
            (audit / "runtime_identity" / (runtime.identity["task_id"] + ".json"))
            .relative_to(root)
            .as_posix()
        )
        receipt_reader = census_evidence._Reader(root)
        receipt = receipt_reader.read(receipt_path)
        _require(
            all(receipt.get(k) == v for k, v in runtime.identity.items()),
            "Replay execution receipt differs from loaded identity",
        )
        _require(receipt.get("fresh_optimizer") is False, "Replay must not create an optimizer")
        reference = trainable_state(runtime.model)
        _require(
            state_hash(reference) == policy_hash == runtime.identity["reference_hash"],
            "Actual replay parameters differ from the sampled S0 parent",
        )
        _require(
            runtime.identity["parent_content_hash"] == spec["groups"][parent]["policy_hash"],
            "Replay parent content changed",
        )
        session = ReplaySession()
        session.runtime, session.parent, session.reference = runtime, parent, reference
        session._reference_cache = (reference, _tensor_stamp(reference))
        session.records, session.raw_records = copy.deepcopy(records), copy.deepcopy(raw)
        session.spec, session.spec_ref = copy.deepcopy(spec), dict(spec_ref)
        session._spec_digest, session._spec_ref = digest(spec), dict(spec_ref)
        session.execution_context, session.producer_context = source, spec["producer_context"]
        session.context = {"execution": source, "producer": spec["producer_context"]}
        references = [dict(spec_ref), *spec["inputs"], *inputs, *receipt_reader.inputs.values()]
        session.inputs = list({r["path"]: r for r in references}.values())
        session.output_directory, session.task_id = output, runtime.identity["task_id"]
        session.scope = "SERVER_ACTUAL"
        session._lock, session._policy_hash = lock, policy_hash
        session._adapter_names = tuple(runtime._srf6_adapter_parameter_names)
        params = dict(runtime.model.named_parameters())
        session._adapter_stamp = _tensor_stamp({n: params[n] for n in session._adapter_names})
        session._base_stamp = _tensor_stamp(
            {n: p for n, p in params.items() if n not in session._adapter_names}
        )
        session._buffer_stamp = _tensor_stamp(dict(runtime.model.named_buffers()))
        session._model_input = copy.deepcopy(raw[0]["model_input"])
        session._raw_binding = copy.deepcopy(raw)
        session._tokens = [row["tokens"][:] for row in raw]
        session._records = {digest(row) for row in (*raw, *records)}
        session._prepared, session._native = {}, None
        _ACTIVE[id(runtime)] = session
        try:
            session.assert_active()
            yield session
        finally:
            try:
                session.refresh_adapter_versions()
            except Exception:
                runtime._srf6_replay_poisoned = True
                raise
            finally:
                _ACTIVE.pop(id(runtime), None)
                fcntl.flock(lock, fcntl.LOCK_UN)
