"""Authenticated original-confirmation replay sessions; never generation access."""

from __future__ import annotations

import copy
import fcntl
from contextlib import contextmanager
from pathlib import Path

from mm_core.training import state_hash, trainable_state
from sr_f6.common import digest
from sr_f6.replay import _RUNTIME_ACCESS, _tensor_stamp
from sr_f6.replay import ReplaySession as ProducerReplaySession

_ACTIVE = {}


def require(value, message):
    if not value:
        raise PermissionError(message)


class ReplaySession(ProducerReplaySession):
    """Reuse input/token guards; execution authority belongs only to this registry."""

    def assert_active(self):
        from .t03_runtime import verify_loaded

        require(
            _ACTIVE.get(id(self.runtime)) is self and not self._lock.closed, "Inactive T03 session"
        )
        loaded = verify_loaded(self.runtime)
        require(
            digest(self.records) == loaded["records_sha256"]
            and digest(self.raw_records) == loaded["raw_sha256"]
            and self.context == self.spec["context"]
            and self.inputs == self.spec["inputs"]
            and self.parent == self.runtime.identity["parent_id"]
            and self.task_id == self.runtime.identity["task_id"]
            and self.scope == "SERVER_ACTUAL",
            "Live replay binding changed",
        )
        require(not getattr(self.runtime, "_srf6_replay_poisoned", False), "Poisoned T03 runtime")
        require(digest(self.spec) == self._spec_digest, "T03 specification changed")
        require(self.spec_ref == self._spec_ref, "T03 specification reference changed")
        params = dict(self.runtime.model.named_parameters())
        require(
            _tensor_stamp({n: p for n, p in params.items() if n not in self._adapter_names})
            == self._base_stamp
            and _tensor_stamp(dict(self.runtime.model.named_buffers())) == self._buffer_stamp
            and _tensor_stamp({n: params[n] for n in self._adapter_names}) == self._adapter_stamp,
            "Original parent parameters or buffers changed",
        )
        require(
            {n for n, p in params.items() if p.requires_grad} == set(self._adapter_names),
            "Only the inherited adapter may require gradients",
        )

    def refresh_adapter_versions(self):
        require(_ACTIVE.get(id(self.runtime)) is self, "Inactive T03 restoration")
        require(
            state_hash(trainable_state(self.runtime.model)) == self._policy_hash,
            "Parent restoration differs",
        )
        params = dict(self.runtime.model.named_parameters())
        self._adapter_stamp = _tensor_stamp({n: params[n] for n in self._adapter_names})
        self.assert_active()

    @contextmanager
    def forward(self, records, *, purpose, grad, access):
        self.assert_active()
        require(
            access is _RUNTIME_ACCESS and self._forward is None, "Unbound or nested replay forward"
        )
        self.check_records(records)
        require(self._calls < 16, "Original replay forward budget exhausted")
        index, reference = self._calls // 2, self._calls % 2 == 0
        expected_purpose = "t03_reference" if reference else "t03_native"
        require(
            records[0]["row_id"] == self.records[index]["row_id"]
            and purpose == expected_purpose
            and grad is (not reference),
            "Replay must follow original draw/reference/current order",
        )
        self._forward = dict(row=records[0], purpose=purpose, grad=grad, accounted=False)
        self._calls += 1  # Failed calls consume their attempt, never a retry.
        try:
            yield
            require(self._forward["accounted"], "Original replay forward lacks accounting")
        finally:
            self._forward = None

    def account(self, kind, count, metadata):
        self.assert_active()
        permit = self._forward
        require(
            permit is not None
            and self._native is not None
            and not permit["accounted"]
            and kind == "extra_forward_sequences"
            and type(count) is int
            and count == 1
            and metadata
            == dict(purpose=permit["purpose"], completion_tokens=len(permit["row"]["tokens"])),
            "Only one original sequence event is allowed; generation is forbidden",
        )
        permit["accounted"] = True
        return permit["row"]


def active_session(runtime):
    session = _ACTIVE.get(id(runtime))
    require(type(session) is ReplaySession, "Actual T03 requires a registered live session")
    session.assert_active()
    return session


@contextmanager
def open_replay(runtime, context, spec, spec_ref):
    from .t03 import selected_parent
    from .t03_runtime import verify_loaded

    loaded = verify_loaded(runtime)
    require(
        digest(spec) == loaded["spec_sha256"]
        and spec_ref == loaded["spec_ref"]
        and context["consumer"] == loaded["consumer"]
        and context["registration"]["producer"] == loaded["producer"],
        "Replay session differs from independently loaded frozen authority",
    )
    require(id(runtime) not in _ACTIVE, "Duplicate replay session")
    records, raw, policy = selected_parent(context, runtime.identity["parent_id"])
    require(
        digest(records) == loaded["records_sha256"] and digest(raw) == loaded["raw_sha256"],
        "Replay records differ from original loaded confirmation evidence",
    )
    require(
        policy == runtime.identity["policy_hash"] == runtime.identity["reference_hash"],
        "Loaded parent differs from original responses",
    )
    output = Path(runtime.identity["audit_root"]) / "T03"
    output.mkdir(parents=True, exist_ok=False)
    lock_path = output.parent / ".session.lock"
    with lock_path.open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = ReplaySession()
        session.runtime, session.parent = runtime, runtime.identity["parent_id"]
        session.reference = runtime._reference
        session._reference_cache = (session.reference, _tensor_stamp(session.reference))
        session.records, session.raw_records = copy.deepcopy(records), copy.deepcopy(raw)
        session.spec, session.spec_ref = copy.deepcopy(spec), dict(spec_ref)
        session._spec_digest, session._spec_ref = digest(spec), dict(spec_ref)
        session.context = copy.deepcopy(spec["context"])
        session.inputs = copy.deepcopy(spec["inputs"])
        session.output_directory, session.task_id = output, runtime.identity["task_id"]
        session.scope, session._lock = "SERVER_ACTUAL", lock
        session._policy_hash = policy
        session._adapter_names = tuple(runtime._adapter_names)
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
        session._prepared, session._native, session._forward, session._calls = {}, None, None, 0
        _ACTIVE[id(runtime)] = session
        try:
            session.assert_active()
            yield session
        finally:
            _ACTIVE.pop(id(runtime), None)
