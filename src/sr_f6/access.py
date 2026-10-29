"""Live PILOT access, separate from source binding and immutable registration.

Only an actually loaded runtime plus independently verified scientific admission
and a held transaction/evaluation ledger can enter. No serialized permit, bool,
or caller-provided panel grants access. S0 and T03 retain their own boundaries.
"""

from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from pathlib import Path

from mm_core.training import state_hash

from .common import digest
from .pilot_contract import for_phase
from .replay import _tensor_stamp
from .transactions import RunTransactions, _plain_path, _read_json

_ACTIVE = {}
_RUNTIME_ACCESS = object()
_UPDATE_ACCESS = object()


def _require(value, message):
    if not value:
        raise PermissionError(message)


def _loaded(runtime, root, mode):
    from .runtime import (
        _LOADED_RUNTIMES,
        Runtime,
        _check_loaded_source_identity,
    )

    _require(
        type(runtime) is Runtime, "Actual loaded Runtime required; synthetic is not authorized"
    )
    bound = _LOADED_RUNTIMES.get(runtime)
    _require(bound is not None and bound["model"] is runtime.model, "Actual runtime load required")
    _require(
        _check_loaded_source_identity(runtime) in ("PILOT_V1", "PILOT_V3")
        and runtime.execution_mode() == mode
        and runtime._srf6_source_context == bound["source"]
        and runtime.identity.get("run_root") == str(root),
        "Only the bound PILOT runtime and matching execution mode are authorized",
    )
    _require(
        all(runtime.identity.get(k) == v for k, v in bound["execution"].items()),
        "Runtime execution identity changed",
    )
    _require(
        all(runtime.identity.get(k) == v for k, v in bound["parent"].items()),
        "Actual loaded parent/reference identity changed",
    )
    _require(
        not getattr(runtime, "_srf6_access_poisoned", False),
        "Runtime requires reload after failed update",
    )
    return bound


def _admission(runtime, root, mode, pilot_spec_sha256, admission_ref):
    from .pilot_contract import require_admission

    bound = _loaded(runtime, root, mode)
    contract = for_phase(bound["source"]["phase"])
    result = require_admission(
        root,
        phase=contract.phase,
        phase_identity_sha256=runtime.identity["phase_identity_sha256"],
        admission_ref=admission_ref,
        pilot_spec_sha256=pilot_spec_sha256,
    )
    context = result["phase_context"]
    _require(
        result.get("status") == "VERIFIED_PILOT_ADMISSION"
        and result.get("scope") == contract.admission_scope
        and result.get("generation_authorized") is True
        and result.get("training_authorized") is True
        and result.get("admission_ref") == admission_ref
        and result["pilot_spec_ref"]["sha256"] == pilot_spec_sha256
        and context["identity"] == bound["source"]["identity"]
        and context["identity_sha256"] == bound["source"]["identity_sha256"]
        and context["deployment"] == bound["source"]["deployment"],
        "Admission does not bind the actual loaded PILOT source",
    )
    if contract.phase == "PILOT_V3":
        spec = result["pilot_spec"]
        _require(
            runtime.identity.get("pilot_spec_sha256") == pilot_spec_sha256
            and runtime.identity.get("input_overlay") == spec["input_overlay"]
            and runtime.identity.get("input_overlay_sha256") == spec["input_overlay"]["sha256"]
            and runtime.identity.get("prompt_version") == spec["prompt_version"],
            "Actual V3 runtime does not match the admitted input overlay",
        )
    return result


def _exclusive(path, value):
    """Reserve one physical generation call, even across fresh access sessions."""
    from .transactions import _canonical

    path = _plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError as exc:
        raise PermissionError(
            "Generation was already attempted; never resample unknown raw"
        ) from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(_canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _verified_recovery(root, admission, pilot_spec_sha256, binding_ref):
    """The recovery verifier owns the single content/stat cache for prefix payloads."""
    from .recovery import verify_recovery

    result = verify_recovery(
        root,
        phase_context=admission["phase_context"],
        pilot_spec_sha256=pilot_spec_sha256,
        binding_ref=binding_ref,
        verify_payloads=True,
    )
    _require(result["scope"] == "SERVER_ACTUAL", "Synthetic recovery cannot authorize model access")
    return result


class PilotSession:
    """Private live object; constructing the class does not enroll it in _ACTIVE."""

    @property
    def groups(self):
        """Inspection view only; caller edits cannot change the registered authority."""
        return copy.deepcopy(self._groups)

    @property
    def step_spec(self):
        return copy.deepcopy(self._step_spec)

    def assert_active(self):
        _require(
            _ACTIVE.get(id(self.runtime)) is self, "NOT_AUTHORIZED: no live PILOT access session"
        )
        _loaded(self.runtime, self.root, self.mode)
        self.owner._locked()
        _require(
            digest(self.owner.identity) == self._owner_identity, "Held ledger identity changed"
        )
        _require(
            self.runtime.identity["step"] == self.committed_step,
            "Access session committed step changed",
        )
        _require(self.runtime.identity["parent_id"] == self.parent, "Access session parent changed")
        if self.mode == "EVALUATION":
            endpoint = getattr(self, "_endpoint_binding", None)
            if endpoint is not None:
                endpoint["tx"]._locked()
                _require(
                    self.runtime.identity.get("checkpoint_commit_sha256")
                    == endpoint["commit_sha256"]
                    and self.runtime.identity.get("checkpoint_run_id") == endpoint["run_id"]
                    and self.runtime.identity.get("checkpoint_step") == 12,
                    "Active committed endpoint identity changed",
                )
            _require(
                self.runtime.policy_tensor_identity() == self.policy, "Evaluation endpoint changed"
            )
        elif not self._updating and not self._updated:
            _require(
                _tensor_stamp(dict(self.runtime.model.named_parameters())) == self._policy_stamp,
                "Policy changed outside the original-token update",
            )
        if self.mode == "TRAINING":
            _require(
                tuple(self.runtime.identity.get("diagnostic_row_indices", ()))
                == self._diagnostic_indices,
                "Frozen diagnostic subset changed",
            )

    def check_input(self, model_input, root, protocol):
        self.assert_active()
        _require(
            Path(root) == self.root and protocol in ("structured", "structured_v4"),
            "PILOT input root/protocol differs",
        )
        if self._generating is not None:
            allowed = [self._generating["model_input"]]
        else:
            _require(
                self._updating, "Input access requires a live generation or original-token update"
            )
            allowed = [row["model_input"] for row in self._records.values()]
        _require(model_input in allowed, "Unregistered PILOT model input")
        self._preparing = digest(model_input)

    def bind_prepared(self, prepared, *, access=None):
        self.assert_active()
        _require(
            access is _RUNTIME_ACCESS and self._preparing is not None,
            "Only runtime preparation can bind native inputs",
        )
        if self._updating:
            from .training import _verify_prepared

            matching = [
                r for r in self._records.values() if digest(r["model_input"]) == self._preparing
            ]
            _require(bool(matching), "Prepared prompt has no original records")
            for row in matching:
                _verify_prepared(row, prepared)
        self._prepared[id(prepared)] = (
            prepared,
            digest(prepared["routing"]),
            _tensor_stamp(prepared["inputs"]),
            self._preparing,
        )
        self._preparing = None

    def check_prepared(self, prepared):
        self.assert_active()
        item = self._prepared.get(id(prepared))
        _require(item is not None and item[0] is prepared, "Unbound preconstructed PILOT input")
        _require(
            item[1] == digest(prepared["routing"]) and item[2] == _tensor_stamp(prepared["inputs"]),
            "Prepared PILOT tensors/routing changed",
        )

    def check_records(self, records):
        self.assert_active()
        _require(
            self._updating and len(records) == 1,
            "Only original single-sequence update forwards allowed",
        )
        for row in records:
            original = {k: v for k, v in row.items() if k != "prepared"}
            _require(
                digest(original) in self._records,
                "Training record differs from READY original tokens",
            )
            if "prepared" in row:
                self.check_prepared(row["prepared"])

    def check_batch(self, prepared, token_rows):
        self.check_prepared(prepared)
        _require(self._updating and len(token_rows) == 1, "No unregistered teacher forcing")
        prompt = self._prepared[id(prepared)][3]
        _require(
            any(
                digest(row["model_input"]) == prompt and row["tokens"] == list(token_rows[0])
                for row in self._records.values()
            ),
            "Completion tokens do not belong to this original prompt",
        )

    def check_reference(self, reference):
        self.assert_active()
        _require(self._updating, "Reference forward requires an original-token update")
        stamp = _tensor_stamp(reference)
        previous = self._reference_cache
        if previous is not None and previous[0] is reference and previous[1] == stamp:
            return
        _require(
            state_hash(reference) == self.runtime.identity["reference_hash"],
            "Frozen parent reference changed",
        )
        self._reference_cache = (reference, stamp)

    @contextmanager
    def native_call(self, inputs, *, completion_start, lengths, purpose, grad, access=None):
        self.assert_active()
        _require(
            access is _RUNTIME_ACCESS and self._updating and self._native is None,
            "Native forward requires original-token update access",
        )
        _require(
            (purpose, grad) in (("training_policy", True), ("training_reference", False)),
            "Unregistered native forward purpose",
        )
        self._native = (inputs, completion_start, lengths, purpose, grad)
        try:
            yield
        finally:
            self._native = None

    def check_native(self, inputs, *, completion_start, lengths, purpose, grad):
        self.assert_active()
        item = self._native
        _require(
            item is not None
            and item[0] is inputs
            and item[1:] == (completion_start, lengths, purpose, grad),
            "Direct native forward bypass is forbidden",
        )

    def check_reservation(self, kind, count, metadata):
        self.assert_active()
        if kind == "diagnostic_gradient_calls":
            row_id = metadata.get("row_id")
            _require(
                self._updating
                and count == 3
                and metadata.get("step") == self.step
                and row_id in self._diagnostic_row_ids
                and row_id not in self._diagnostic_seen,
                "Unregistered or duplicate diagnostic gradient calls",
            )
            self._diagnostic_seen.add(row_id)
            return
        if kind == "extra_forward_sequences":
            _require(
                self._updating and self._native is not None and count == 1,
                "Extra forward requires original-token update",
            )
            return
        item = self._generating
        _require(
            item is not None and metadata.get("group_seed") == item["seeds"][0],
            "Direct native generation bypass is forbidden",
        )
        if kind == "completion_attempts":
            _require(
                not self._attempted and count == len(item["seeds"]),
                "Duplicate native completion call",
            )
            self._attempted = True
        elif kind == "generated_tokens":
            index = metadata.get("row_index")
            _require(
                self._attempted
                and type(index) is int
                and index not in self._emitted
                and 0 <= index < len(item["seeds"])
                and type(count) is int
                and 1 <= count <= 1536,
                "Unexpected completion accounting",
            )
            self._emitted.add(index)
        else:
            raise PermissionError("Unregistered PILOT accounting category")

    @contextmanager
    def generation(self, model_input, root, seeds, *, protocol, access=None):
        self.assert_active()
        _require(
            access is _RUNTIME_ACCESS
            and self._generating is None
            and not self._updating
            and not self._updated,
            "No live generation access",
        )
        _require(
            Path(root) == self.root and protocol == "structured_v4",
            "Only registered structured_v4 generation",
        )
        if self.mode == "TRAINING":
            candidates = [
                g
                for g in self.groups
                if g["model_input"] == model_input and g["seeds"] == list(seeds)
            ]
            _require(len(candidates) == 1, "Input/seeds are not the next registered 16 G8")
            item = candidates[0]
            folder = (
                self.owner.directory
                / "rollouts"
                / f"step-{self.step:03d}"
                / "groups"
                / f"{item['group_index']:02d}"
            )
            _require(
                not (folder / "COMPLETE.json").exists() and not (folder / "raw.jsonl").exists(),
                "Complete or partial original raw cannot be resampled",
            )
            started = _read_json(folder / "STARTED.json")
            _require(
                started.get("spec") == item["spec"]
                and started.get("requested_draws") == 8
                and started.get("status") == "STARTED",
                "Durable original rollout reservation required",
            )
            for previous in self.groups[: item["group_index"]]:
                prior = folder.parent / f"{previous['group_index']:02d}" / "COMPLETE.json"
                _require(
                    _read_json(prior).get("spec") == previous["spec"],
                    "Registered generation order differs",
                )
            _exclusive(
                self.owner.directory
                / "access"
                / f"step-{self.step:03d}"
                / f"group-{item['group_index']:02d}.json",
                dict(
                    status="STARTED_OUTCOME_UNKNOWN",
                    spec=item["spec"],
                    admission_ref=self.admission_ref,
                ),
            )
        else:
            pending = self.owner.pending
            _require(
                pending is not None
                and self._evaluation_schedule.get(pending.get("qid")) == pending.get("seeds")
                and self._evaluation_inputs.get(pending["qid"]) == model_input
                and list(seeds) == pending["seeds"],
                "Pending evaluation differs from frozen DEV_EVAL panel",
            )
            self.owner.authorize_generation(model_input, seeds)
            item = dict(model_input=copy.deepcopy(model_input), seeds=list(seeds))
        self._generating, self._attempted, self._emitted = item, False, set()
        try:
            yield
            _require(
                self._attempted and len(self._emitted) == len(seeds),
                "Incomplete native generation accounting",
            )
        finally:
            self._generating = None
            self._prepared.clear()

    def _ready_records(self):
        from . import rollouts
        from .training import build_credit_group

        tx = self.owner
        directory = tx.directory / "rollouts" / f"step-{self.step:03d}"
        ready = _read_json(directory / "READY.json")
        _require(
            ready.get("status") == "ROLLOUT_READY" and ready.get("spec") == self.step_spec,
            "Current original-token READY is required",
        )
        refs = ready["references"]
        for key in ("rollout", "credit"):
            _require(
                refs[key]["path"] == f"rollouts/step-{self.step:03d}/{key}.json",
                "READY reference path differs",
            )
        rollout = rollouts._bound(tx, tx.directory, refs["rollout"])
        credit = rollouts._bound(tx, tx.directory, refs["credit"])
        expected, credits, complete_refs = [], [], []
        for item in self.groups:
            folder = directory / "groups" / f"{item['group_index']:02d}"
            ref = rollouts._ref(tx.directory, folder / "COMPLETE.json")
            marker = rollouts._bound(tx, tx.directory, ref)
            _require(
                marker.get("status") == "IMMUTABLE_COMPLETE_G8"
                and marker.get("spec") == item["spec"]
                and marker.get("rows") == 8,
                "Original G8 completion differs",
            )
            raw_ref = marker["raw_reference"]
            _require(
                raw_ref["path"] == (folder / "raw.jsonl").relative_to(tx.directory).as_posix(),
                "Original raw path differs",
            )
            raw = rollouts._bound(tx, tx.directory, raw_ref, kind="jsonl")
            rollouts._validate_raw(
                raw, self.runtime, item["model_input"], item["seeds"], self.policy, self.root
            )
            rows = rollouts._annotate(
                raw, self._tasks[item["qid"]], tx, self.runtime, self.step, item["group_index"]
            )
            expected.extend(rows)
            credits.append(
                build_credit_group(
                    rows,
                    tx.identity["arm"],
                    self._registration["shuffle_seeds"][self.step - 1][item["group_index"]],
                )
            )
            complete_refs.append(ref)
        _require(
            rollout.get("identity") == tx.identity
            and rollout.get("step") == self.step
            and rollout.get("policy_before_sha256") == self.policy
            and rollout.get("registration_ref") == tx.identity["rollout_registration"]
            and rollout.get("complete_group_references") == complete_refs
            and rollout.get("records") == expected
            and len(expected) == 128,
            "READY records differ from complete original 16 G8",
        )
        _require(
            credit == dict(identity=tx.identity, step=self.step, groups=credits),
            "READY credit differs from original raw",
        )
        return expected, credits

    @contextmanager
    def update(self, records, group_sizes, reference, credits, *, access=None):
        self.assert_active()
        _require(
            access is _UPDATE_ACCESS
            and self.mode == "TRAINING"
            and not self._updated
            and not self._updating
            and self._generating is None,
            "No original-token update access",
        )
        expected, expected_credits = self._ready_records()
        _require(
            records == expected and list(group_sizes) == [8] * 16 and credits == expected_credits,
            "Update must consume exactly the persisted original READY batch",
        )
        self._records = {digest(row): row for row in expected}
        self._diagnostic_row_ids = (
            {expected[i]["row_id"] for i in self._diagnostic_indices}
            if self.step in (1, 4, 8, 12)
            else set()
        )
        self._diagnostic_seen = set()
        self._prepared.clear()
        self._updating = True
        try:
            self.check_reference(reference)
            yield
            _require(
                self._diagnostic_seen == self._diagnostic_row_ids,
                "Registered diagnostic gradient accounting incomplete",
            )
        except BaseException:
            self.runtime._srf6_access_poisoned = True
            raise
        finally:
            self._updating, self._updated = False, True
            self._records.clear()
            self._prepared.clear()


def active_session(runtime):
    session = _ACTIVE.get(id(runtime))
    _require(
        type(session) is PilotSession,
        "NOT_AUTHORIZED: no live PILOT scientific/panel access session",
    )
    session.assert_active()
    return session


def _new_session(runtime, root, owner, admission, mode):
    _require(id(runtime) not in _ACTIVE, "Runtime already has a live access session")
    session = PilotSession()
    session.runtime, session.root, session.owner, session.mode = runtime, root, owner, mode
    session.admission_ref = copy.deepcopy(admission["admission_ref"])
    session._owner_identity = digest(owner.identity)
    session.committed_step = runtime.identity["step"]
    session.parent, session.policy = runtime.identity["parent_id"], runtime.policy_tensor_identity()
    session._policy_stamp = _tensor_stamp(dict(runtime.model.named_parameters()))
    session._generating = session._preparing = session._native = session._reference_cache = None
    session._prepared, session._records = {}, {}
    session._updating = session._updated = False
    return session


@contextmanager
def open_training_access(
    runtime, tx, *, experiment_root, pilot_spec_sha256, admission_ref, recovery_binding_ref=None
):
    from . import rollouts

    root = _plain_path(experiment_root)
    admission = _admission(runtime, root, "TRAINING", pilot_spec_sha256, admission_ref)
    _require(type(tx) is RunTransactions, "Actual held RunTransactions required")
    tx._locked()
    paths = [
        p for p in admission["pilot_spec"]["paths"] if p["run_id"] == runtime.identity["run_id"]
    ]
    _require(len(paths) == 1, "Runtime is not one of the nine registered PILOT runs")
    path = paths[0]
    directory = for_phase(admission["phase_context"]["phase"]).registration_directory
    identity = rollouts._bound(tx, root / directory, path["transaction_identity"])
    _require(tx.identity == identity, "Transaction identity differs from registered primary")
    expected_directory = _plain_path(root / path["audit_root"])
    if tx.directory != expected_directory:
        _require(recovery_binding_ref is not None, "Unregistered physical recovery lane")
        recovered = _verified_recovery(root, admission, pilot_spec_sha256, recovery_binding_ref)
        _require(
            Path(recovered["directory"]) == tx.directory and recovered["identity"] == tx.identity,
            "Recovery physical/logical binding differs",
        )
    else:
        _require(recovery_binding_ref is None, "PRIMARY cannot masquerade as RECOVERY")
    _require(
        runtime.identity["audit_root"] == str(tx.directory),
        "Runtime audit path differs from held transaction",
    )
    _, registration, public, tasks, _committed, step, policy = rollouts._context(
        tx, runtime, root, tx.identity["rollout_registration"]
    )
    session = _new_session(runtime, root, tx, admission, "TRAINING")
    _require(policy == session.policy and 1 <= step <= 12, "PILOT policy/step differs")
    if recovery_binding_ref is not None:
        _require(5 <= step <= 12, "Recovery may only execute registered steps 5..12")
    session.step, session._registration, session._tasks = step, registration, tasks
    indices = admission["pilot_spec"]["diagnostics"]["row_indices"]
    _require(
        indices == tx.identity.get("diagnostic_row_indices") and len(indices) == 64,
        "Diagnostic subset differs from registration",
    )
    session._diagnostic_indices = tuple(indices)
    session._step_spec = dict(
        format=rollouts.FORMAT,
        identity=tx.identity,
        step=step,
        registration_ref=tx.identity["rollout_registration"],
        policy_before_sha256=policy,
        roots=registration["schedule"][step - 1],
    )
    session._groups = []
    for g in range(16):
        qid = f"{session.step_spec['roots'][g // 4]}_{registration['factor']}_v{g % 4}"
        safe, seeds = public[qid]["model_input"], registration["draw_seeds"][step - 1][g]
        spec = dict(
            step_spec_sha256=digest(session.step_spec),
            group_index=g,
            qid=qid,
            seeds=seeds,
            model_input_sha256=digest(safe),
            task_sha256=digest(tasks[qid]),
        )
        session._groups.append(
            dict(group_index=g, qid=qid, seeds=seeds, model_input=safe, spec=spec)
        )
    _ACTIVE[id(runtime)] = session
    try:
        session.assert_active()
        yield session
    finally:
        _ACTIVE.pop(id(runtime), None)


@contextmanager
def open_evaluation_access(runtime, *, experiment_root, pilot_spec_sha256, admission_ref, ledger):
    from .evaluation import EvaluationLedger, _inputs

    root = _plain_path(experiment_root)
    admission = _admission(runtime, root, "EVALUATION", pilot_spec_sha256, admission_ref)
    contract = for_phase(admission["phase_context"]["phase"])
    _require(type(ledger) is EvaluationLedger, "Actual held EvaluationLedger required")
    ledger._locked()
    plan = admission["pilot_spec"]["evaluation"]
    _require(
        ledger.plan == plan
        and ledger.identity.get("execution_origin") == "SERVER_ACTUAL"
        and ledger.root == root,
        "Synthetic or unregistered evaluation ledger",
    )
    endpoint = next(
        (e for e in plan["endpoints"] if e["endpoint_id"] == ledger.identity["endpoint_id"]), None
    )
    _require(endpoint is not None and endpoint == ledger.endpoint, "Unregistered endpoint")
    _require(
        ledger.directory
        == _plain_path(root / contract.evaluation_directory / endpoint["endpoint_id"])
        and runtime.identity["audit_root"] == str(ledger.directory),
        "Evaluation directory differs",
    )
    public, tasks = _inputs(
        root,
        plan,
        spec=admission["pilot_spec"],
        phase_context=admission["phase_context"],
    )
    _require(
        ledger.public == public
        and ledger.tasks == tasks
        and ledger.schedule == plan["seeds_by_parent"][endpoint["parent"]],
        "Evaluation input/seed panel changed",
    )
    mapping = {
        "evaluation_run_id": "run_id",
        "parent": "parent_id",
        "checkpoint_step": "step",
        "policy_tensor_sha256": "policy_hash",
        "phase_identity_sha256": "phase_identity_sha256",
        "source_sha256": "source_sha256",
        "data_sha256": "data_hash",
        "reference_sha256": "reference_hash",
    }
    _require(
        all(ledger.identity.get(k) == runtime.identity.get(v) for k, v in mapping.items())
        and ledger.identity.get("pilot_spec_sha256") == pilot_spec_sha256
        and ledger.identity.get("admission_ref") == admission_ref
        and ledger.identity.get("input_refs") == plan["inputs"],
        "Evaluation loaded identity differs",
    )
    if contract.phase == "PILOT_V3":
        _require(
            all(
                ledger.identity.get(key) == runtime.identity.get(key)
                for key in ("phase", "input_overlay", "input_overlay_sha256", "prompt_version")
            ),
            "Evaluation V3 source or input overlay changed",
        )
    _require(
        runtime.identity["step"] == endpoint["checkpoint_step"]
        and runtime.identity["parent_id"] == endpoint["parent"],
        "Evaluation actual endpoint differs",
    )
    if endpoint["kind"] == "FROZEN_PARENT":
        _require(
            runtime.policy_tensor_identity() == endpoint["tensor_sha256"]
            and runtime.identity.get("checkpoint_commit_sha256") is None,
            "Frozen parent differs",
        )
    else:
        from .runtime import _LOADED_ENDPOINTS

        bound = _LOADED_ENDPOINTS.get(runtime)
        _require(bound is not None, "Actually loaded committed endpoint required")
        bound["tx"]._locked()
        _require(
            bound["tx"].directory
            == root / Path(endpoint["expected_commit_path"]).parent.parent.parent
            and bound["run_id"] == endpoint["run_id"]
            and bound["step"] == 12
            and ledger.identity.get("checkpoint_commit_sha256") == bound["commit_sha256"]
            and runtime.identity.get("checkpoint_commit_sha256") == bound["commit_sha256"]
            and runtime.identity.get("checkpoint_run_id") == bound["run_id"],
            "Committed PRIMARY endpoint binding differs",
        )
    session = _new_session(runtime, root, ledger, admission, "EVALUATION")
    session._endpoint_binding = bound if endpoint["kind"] != "FROZEN_PARENT" else None
    session._evaluation_inputs = {
        qid: copy.deepcopy(row["model_input"]) for qid, row in public.items()
    }
    session._evaluation_schedule = {row["qid"]: list(row["seeds"]) for row in ledger.schedule}
    _ACTIVE[id(runtime)] = session
    try:
        session.assert_active()
        yield session
    finally:
        _ACTIVE.pop(id(runtime), None)
