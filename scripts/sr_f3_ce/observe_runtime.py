"""Explicit, host-only observation of the frozen SR-F3-CE runtime.

The external worker authenticates this file before importing it and owns durable
JSON event writes. Importing this module installs nothing. No model, tokenizer,
CUDA timer, RNG operation, autograd hook, or scientific retry is invoked here.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import hashlib
import inspect
import itertools
import json
import os
import sys
import threading
import time
import weakref
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = 1
EXPERIMENT_ID = "SR-F3-CE-20261016"
_ATTRIBUTE = "_sr_f3_ce_observation_installation"
_METHODS = ("__init__", "generate_group", "_native_forward")


class ObservationError(RuntimeError):
    """An observer failure; callers must not automatically repeat scientific work."""


class ObservationWriteError(ObservationError):
    def __init__(self, event, cause):
        self.event = event["event"]
        self.scope_id = event["scope_id"]
        self.original_call_started = event["event"] == "SCOPE_END"
        self.original_call_returned = event.get("original_call_returned", False)
        super().__init__(
            f"Observation {self.event} write failed for {self.scope_id}; "
            f"original_call_started={self.original_call_started}, "
            f"original_call_returned={self.original_call_returned}; "
            f"do not retry scientific work ({type(cause).__name__})"
        )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _method_identity(method):
    path = Path(inspect.getsourcefile(method)).resolve(strict=True)
    return dict(
        module=method.__module__,
        qualname=method.__qualname__,
        file=str(path),
        file_sha256=_sha(path),
        source_sha256=hashlib.sha256(inspect.getsource(method).encode()).hexdigest(),
        signature=str(inspect.signature(method)),
    )


def _text(value):
    return value if isinstance(value, str) and len(value) <= 256 else "UNKNOWN"


def _stats():
    return dict.fromkeys(
        (
            "entered",
            "returned",
            "raised",
            "checkpoint_early_stop",
            "unresolved",
            "input_batch_items",
            "input_sequence_slots",
            "shape_unknown_calls",
            "image_items",
            "image_shape_unknown_calls",
            "input_element_slots",
            "host_inclusive_ns",
            "host_exclusive_ns",
        ),
        0,
    )


def _shape(value, torch):
    # Only tensor metadata is read. No item(), cpu(), clone(), detach(), reduction,
    # cache inspection, or reference to an input/output tensor is retained.
    return tuple(int(n) for n in value.shape) if torch.is_tensor(value) else None


def _has_pixels(values, torch):
    if not isinstance(values, Mapping):
        return False
    for name in ("pixel_values", "pixel_values_videos"):
        shape = _shape(values.get(name), torch)
        if shape and all(n > 0 for n in shape):
            return True
    return False


def _checkpoint_stop(error):
    return (
        error is not None
        and type(error).__module__ == "torch.utils.checkpoint"
        and (type(error).__name__ == "_StopRecomputationError")
    )


def _input_counts(args, kwargs, kind, torch):
    if kind == "visual":
        value = kwargs.get("hidden_states", kwargs.get("pixel_values"))
        if value is None and args:
            value = args[0]
        shape = _shape(value, torch)
        grid = kwargs.get("grid_thw", kwargs.get("image_grid_thw"))
        if grid is None and len(args) > 1:
            grid = args[1]
        grid_shape = _shape(grid, torch)
        elements = 1
        for dim in shape or ():
            elements *= dim
        return dict(
            shape_unknown_calls=int(shape is None),
            input_element_slots=elements if shape is not None else 0,
            image_items=grid_shape[0]
            if grid_shape and len(grid_shape) == 2 and grid_shape[1] == 3
            else 0,
            image_shape_unknown_calls=int(
                not (grid_shape and len(grid_shape) == 2 and grid_shape[1] == 3)
            ),
        )
    value = kwargs.get("input_ids", kwargs.get("inputs_embeds"))
    if value is None and args:
        value = args[0]
    shape = _shape(value, torch)
    if shape is None or len(shape) not in (2, 3):
        return dict(shape_unknown_calls=1)
    return dict(input_batch_items=shape[0], input_sequence_slots=shape[0] * shape[1])


class _Scope:
    def __init__(self, owner, state, kind, purpose, protocol, parent):
        self.owner, self.state = owner, state
        self.scope_id = f"{state.runtime_id}/s{next(owner.scope_ids)}"
        self.parent_scope_id = parent.scope_id if parent else None
        self.kind, self.purpose, self.protocol = kind, purpose, protocol
        self.model_stages, self.visual_stages, self.model_hook_layers = {}, {}, {}
        self.overhead = Counter()
        self.generation_calls = 0
        self.started_ns = None
        self.token = None
        self.post_without_enter = 0
        self.visual_expected = False
        self.graph_task_id = None
        if kind.startswith("unscoped_"):
            query = getattr(owner.torch._C, "_current_graph_task_id", None)
            if callable(query):
                self.graph_task_id = int(query())

    def event(self, event):
        return dict(
            schema_version=SCHEMA_VERSION,
            experiment_id=EXPERIMENT_ID,
            installation_id=self.owner.identity["installation_id"],
            runtime_id=self.state.runtime_id,
            scope_id=self.scope_id,
            parent_scope_id=self.parent_scope_id,
            event=event,
            kind=self.kind,
            purpose=self.purpose,
            protocol=self.protocol,
            pid=os.getpid(),
            time_unix_ns=time.time_ns(),
            autograd_graph_task_id=self.graph_task_id,
            origin_scope_id="UNKNOWN" if self.kind.startswith("unscoped_") else self.scope_id,
        )

    def stage(self):
        if self.kind == "generation":
            stage = "prefill" if self.generation_calls == 0 else "incremental_decode"
            self.generation_calls += 1
            return stage
        return "teacher_forcing" if self.kind == "teacher_forcing" else "unscoped_forward"


class _Frame:
    def __init__(self, module_id, kind, scope, stats, primary, stage, parent, orphan):
        self.module_id, self.kind, self.scope, self.stats = module_id, kind, scope, stats
        self.primary, self.stage, self.parent, self.orphan = primary, stage, parent, orphan
        self.returned = False
        self.child_ns = 0
        self.started_ns = time.perf_counter_ns()


class _RuntimeState:
    def __init__(self, owner):
        self.owner = owner
        self.runtime_id = f"{owner.identity['installation_id']}/r{next(owner.runtime_ids)}"
        self.handles = []
        self.hook_targets = []
        self.frames = contextvars.ContextVar(self.runtime_id + "/frames", default=())

    def attach(self, runtime):
        model = runtime.model
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        visual = runtime._visual_module()
        if visual is model or visual is base:
            raise ObservationError("Visual module cannot also be the root/base model")
        targets = [(model, "model=base" if model is base else "model", "model")]
        if base is not model:
            targets.append((base, "base", "model"))
        targets.append((visual, "visual", "visual"))
        try:
            for module, alias, kind in targets:
                if not isinstance(module, self.owner.torch.nn.Module):
                    raise ObservationError("Observed targets must be actual torch Modules")
                self.hook_targets.append(
                    dict(alias=alias, kind=kind, module_class=type(module).__qualname__)
                )
                self.handles.append(
                    module.register_forward_pre_hook(
                        functools.partial(self.pre, alias=alias, kind=kind),
                        with_kwargs=True,
                    )
                )
                # A separate ordinary post hook distinguishes a legitimate None
                # return from always_call's exception path, including ambient except blocks.
                self.handles.append(module.register_forward_hook(self.returned, with_kwargs=True))
                self.handles.append(
                    module.register_forward_hook(
                        self.post,
                        with_kwargs=True,
                        always_call=True,
                    )
                )
        except BaseException:
            self.remove()
            raise

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def current(self):
        return next((s for s in reversed(self.owner.scopes.get()) if s.state is self), None)

    def pre(self, module, args, kwargs, *, alias, kind):
        if self.owner.sealed:
            raise ObservationError("Observer is sealed; no further model calls are authorized")
        scope = self.current()
        with self.owner.measure("hook_host_ns", scope):
            orphan = scope is None
            if orphan:
                scope = self.owner.begin(self, "unscoped_" + kind + "_forward", "UNKNOWN", None)
            stack = self.frames.get()
            primary = kind == "model" and not any(
                f.kind == "model" and f.scope is scope for f in stack
            )
            parent = next(
                (f for f in reversed(stack) if f.kind == kind and (kind == "visual" or f.primary)),
                None,
            )
            if kind == "model":
                scope.visual_expected = scope.visual_expected or _has_pixels(
                    kwargs, self.owner.torch
                )
                layers = scope.model_hook_layers.setdefault(alias, Counter())
                layers["entered"] += 1
                if primary:
                    stage = scope.stage()
                    stats = scope.model_stages.setdefault(stage, _stats())
                else:
                    stage, stats = parent.stage, None
            else:
                model_parent = next((f for f in reversed(stack) if f.kind == "model"), None)
                stage = model_parent.stage if model_parent else "unscoped_visual"
                stats = scope.visual_stages.setdefault(stage, _stats())
            if stats is not None:
                stats["entered"] += 1
                for key, count in _input_counts(args, kwargs, kind, self.owner.torch).items():
                    stats[key] += count
            frame = _Frame(id(module), kind, scope, stats, primary, stage, parent, orphan)
            frame.alias = alias
            self.frames.set((*stack, frame))
        return None

    def returned(self, module, _args, _kwargs, _output):
        if self.owner.sealed:
            return None
        frame = next((f for f in reversed(self.frames.get()) if f.module_id == id(module)), None)
        with self.owner.measure("hook_host_ns", frame.scope if frame else self.current()):
            if frame is not None:
                frame.returned = True
        return None

    def post(self, module, _args, _kwargs, _output):
        if self.owner.sealed:
            # A refused pre-hook must never close an older interrupted frame.
            return None
        ended = time.perf_counter_ns()
        stack = self.frames.get()
        frame = next((f for f in reversed(stack) if f.module_id == id(module)), None)
        with self.owner.measure("hook_host_ns", frame.scope if frame else self.current()):
            if frame is None:
                scope = self.current()
                if scope is not None:
                    scope.post_without_enter += 1
                return None
            self.frames.set(tuple(f for f in stack if f is not frame))
            elapsed = ended - frame.started_ns
            error = None if frame.returned else sys.exc_info()[1]
            result = (
                "returned"
                if frame.returned
                else ("checkpoint_early_stop" if _checkpoint_stop(error) else "raised")
            )
            if frame.stats is not None:
                frame.stats[result] += 1
                frame.stats["host_inclusive_ns"] += elapsed
                frame.stats["host_exclusive_ns"] += max(0, elapsed - frame.child_ns)
                if frame.parent is not None:
                    frame.parent.child_ns += elapsed
            if frame.kind == "model":
                frame.scope.model_hook_layers[frame.alias][result] += 1
            if frame.orphan:
                self.owner.finish(frame.scope, error, ended, returned=frame.returned)
        return None


class _Installation:
    def __init__(self, runtime_class, emit):
        import torch

        self.torch = torch
        self.originals = {name: getattr(runtime_class, name) for name in _METHODS}
        self.own_originals = {name: runtime_class.__dict__.get(name) for name in _METHODS}
        stamp = f"{os.getpid()}-{time.time_ns()}"
        self.identity = dict(
            schema_version=SCHEMA_VERSION,
            experiment_id=EXPERIMENT_ID,
            installation_id="observe-" + stamp,
            observer_source_sha256=_sha(__file__),
            runtime_class=f"{runtime_class.__module__}.{runtime_class.__qualname__}",
            original_methods={
                name: _method_identity(method) for name, method in self.originals.items()
            },
            torch_version=str(torch.__version__),
            timing="HOST_PERF_COUNTER_NS_NOT_CUDA_KERNEL_TIME",
            phase_rule="first root/base generation forward is prefill; subsequent are incremental",
            model_count_rule=(
                "outermost root/base invocation per runtime scope; nested base excluded"
            ),
            visual_count_rule="actual visual pre/post hooks; separate nested cost axis",
            old_engine_observed=False,
            installation_status="INSTALLED",
        )
        self.emit_callback = emit
        self.runtime_ids, self.scope_ids = itertools.count(1), itertools.count(1)
        self.states = weakref.WeakKeyDictionary()
        self.all_states = []  # Handles hold weak module references, never tensors.
        self.scopes = contextvars.ContextVar(stamp + "/scopes", default=())
        self.measures = contextvars.ContextVar(stamp + "/measures", default=())
        self.emitting = contextvars.ContextVar(stamp + "/emitting", default=False)
        self.lock = threading.RLock()
        self.overhead, self.events = Counter(), Counter()
        self.active, self.write_errors = {}, []
        self.sealed = False

    @contextlib.contextmanager
    def measure(self, key, scope=None):
        frame = [time.perf_counter_ns(), 0]
        parents = self.measures.get()
        token = self.measures.set((*parents, frame))
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - frame[0]
            exclusive = max(0, elapsed - frame[1])
            with self.lock:
                self.overhead[key] += exclusive
                if scope is not None:
                    scope.overhead[key] += exclusive
            self.measures.reset(token)
            if parents:
                parents[-1][1] += elapsed

    def write(self, event, scope):
        if self.emitting.get():
            raise ObservationError("Observer emitter must not reenter the runtime")
        with self.measure("event_build_host_ns", scope):
            detached = json.loads(json.dumps(event, allow_nan=False))
        token = self.emitting.set(True)
        try:
            with self.measure("emit_host_ns", scope), self.lock:
                try:
                    self.emit_callback(detached)
                except Exception as error:
                    self.write_errors.append(
                        dict(
                            event=event["event"],
                            scope_id=scope.scope_id,
                            exception_type=type(error).__name__,
                        )
                    )
                    raise ObservationWriteError(event, error) from error
                self.events[event["event"]] += 1
        finally:
            self.emitting.reset(token)

    def begin(self, state, kind, purpose, protocol):
        if self.emitting.get():
            raise ObservationError("Observer emitter must not reenter the runtime")
        parents = self.scopes.get()
        scope = _Scope(self, state, kind, purpose, protocol, parents[-1] if parents else None)
        with self.measure("scope_bookkeeping_host_ns", scope):
            self.write(scope.event("SCOPE_BEGIN"), scope)
            scope.token = self.scopes.set((*parents, scope))
            scope.started_ns = time.perf_counter_ns()
            with self.lock:
                self.active[scope.scope_id] = scope.event("SCOPE_BEGIN")
        return scope

    def finish(self, scope, error, ended_ns, *, returned):
        with self.measure("scope_bookkeeping_host_ns", scope):
            try:
                stack = scope.state.frames.get()
                unresolved = [f for f in stack if f.scope is scope]
                for frame in unresolved:
                    if frame.stats is not None:
                        frame.stats["unresolved"] += 1
                    if frame.kind == "model":
                        scope.model_hook_layers[frame.alias]["unresolved"] += 1
                scope.state.frames.set(tuple(f for f in stack if f.scope is not scope))
                outcome = (
                    "RETURNED"
                    if error is None and returned
                    else (
                        "CHECKPOINT_EARLY_STOP"
                        if _checkpoint_stop(error)
                        else (
                            "INTERRUPTED"
                            if isinstance(error, (KeyboardInterrupt, SystemExit))
                            else "RAISED"
                        )
                    )
                )
                issues = []
                if unresolved or scope.post_without_enter:
                    issues.append("UNRESOLVED_HOOK_CALLS")
                if error is None and returned and scope.kind in ("generation", "teacher_forcing"):
                    if not sum(s["entered"] for s in scope.model_stages.values()):
                        issues.append("NO_PRIMARY_MODEL_FORWARD_OBSERVED")
                    if scope.visual_expected and not sum(
                        s["entered"] for s in scope.visual_stages.values()
                    ):
                        issues.append("EXPECTED_VISUAL_FORWARD_NOT_OBSERVED")
                event = scope.event("SCOPE_END")
                event.update(
                    outcome=outcome,
                    original_call_returned=returned,
                    exception_type=type(error).__name__ if error is not None else None,
                    host_scope_elapsed_ns=max(0, ended_ns - scope.started_ns),
                    model_stages=copy.deepcopy(scope.model_stages),
                    visual_stages=copy.deepcopy(scope.visual_stages),
                    model_hook_layers=copy.deepcopy(scope.model_hook_layers),
                    hook_targets=copy.deepcopy(scope.state.hook_targets),
                    post_without_enter=scope.post_without_enter,
                    observer_overhead_ns=dict(scope.overhead),
                    overhead_scope=(
                        "snapshot_before_END_build_and_emit; final snapshot is authoritative"
                    ),
                    unresolved_calls_are_unknown=bool(unresolved or scope.post_without_enter),
                    coverage_status="UNKNOWN" if issues else "COMPLETE",
                    coverage_issues=issues,
                    visual_expected=scope.visual_expected,
                )
                try:
                    self.write(event, scope)
                except ObservationWriteError as write_error:
                    if error is None:
                        raise
                    if hasattr(error, "add_note"):
                        error.add_note(str(write_error))
                    # Preserve the scientific exception, including OOM identity.
                    # The BEGIN without a durable END remains UNKNOWN in the ledger.
                if issues and error is None:
                    raise ObservationError("Observation coverage is UNKNOWN: " + ",".join(issues))
            finally:
                self.scopes.reset(scope.token)
                with self.lock:
                    self.active.pop(scope.scope_id, None)

    def call(self, instance, name, args, kwargs):
        if self.sealed:
            raise ObservationError("Observer is sealed; no further runtime calls are authorized")
        if self.emitting.get():
            raise ObservationError("Observer emitter must not reenter the runtime")
        if name == "__init__":
            if instance in self.states:
                raise ObservationError("Runtime constructor cannot be observed twice")
            state = _RuntimeState(self)
            self.states[instance] = state
            self.all_states.append(state)
            kind, purpose, protocol = "initialization", "runtime_initialization", None
        else:
            state = self.states.get(instance)
            if state is None:
                raise ObservationError(
                    "Runtime predates observer installation; construct after install"
                )
            kind = "generation" if name == "generate_group" else "teacher_forcing"
            purpose = "generation" if kind == "generation" else _text(kwargs.get("purpose"))
            protocol = _text(kwargs.get("protocol", "structured")) if kind == "generation" else None
        scope = self.begin(state, kind, purpose, protocol)
        error, returned = None, False
        try:
            values = (
                args[0] if args else kwargs.get("model_input" if kind == "generation" else "inputs")
            )
            if kind == "generation" and isinstance(values, Mapping):
                scope.visual_expected = values.get("image_file") is not None
            elif kind == "teacher_forcing":
                scope.visual_expected = _has_pixels(values, self.torch)
            result = self.originals[name](instance, *args, **kwargs)
            returned = True
            if name == "__init__":
                state.attach(instance)
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            self.finish(scope, error, time.perf_counter_ns(), returned=returned)


def install(runtime_class, emit):
    """Wrap the explicit runtime class once; return JSON-safe installation identity.

    ``emit(event)`` must durably append the event before returning. A write error
    does not authorize replay. Existing instances are deliberately unsupported.
    The caller must authenticate this module and its external launch amendment.
    """
    if not isinstance(runtime_class, type) or not callable(emit):
        raise TypeError("install requires a runtime class and a durable emit callback")
    if getattr(runtime_class, _ATTRIBUTE, None) is not None:
        raise ObservationError("Observer already installed on this runtime class")
    if any(not callable(getattr(runtime_class, name, None)) for name in _METHODS):
        raise TypeError("Runtime must provide __init__, generate_group and _native_forward")
    owner = _Installation(runtime_class, emit)

    def wrapper(name, original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            return owner.call(self, name, args, kwargs)

        return wrapped

    for name, original in owner.originals.items():
        setattr(runtime_class, name, wrapper(name, original))
    setattr(runtime_class, _ATTRIBUTE, owner)
    return copy.deepcopy(owner.identity)


def snapshot(runtime_class):
    """Return observer-only final metadata, without emitting or touching the model.

    Includes the last END write's measured overhead. The external worker may
    persist this in its own FINAL receipt; that final receipt's write is outside
    this observer's measurement. Unmatched BEGIN events remain UNKNOWN.
    """
    owner = getattr(runtime_class, _ATTRIBUTE, None)
    if owner is None:
        raise ObservationError("No observer installation for this runtime class")
    with owner.lock:
        return dict(
            schema_version=SCHEMA_VERSION,
            installation_id=owner.identity["installation_id"],
            events_written=dict(owner.events),
            observer_overhead_ns=dict(owner.overhead),
            active_scopes=copy.deepcopy(list(owner.active.values())),
            write_errors=copy.deepcopy(owner.write_errors),
            sealed=owner.sealed,
            final_receipt_write_cost="UNKNOWN_NOT_MEASURED_BY_OBSERVER",
        )


def seal(runtime_class):
    """Close observation at the worker's terminal boundary, without fabricating ENDs.

    Idempotent. The worker must call this on success and on BaseException before
    considering any continuation. Unscoped hooks interrupted by BaseException
    cannot close themselves: their unmatched BEGIN/active scope remains UNKNOWN.
    Future observed runtime/module entry is refused, including after a caught
    interrupt. This is not a model-recovery or scientific-retry API.
    """
    owner = getattr(runtime_class, _ATTRIBUTE, None)
    if owner is None:
        raise ObservationError("No observer installation for this runtime class")
    with owner.lock:
        owner.sealed = True
        return snapshot(runtime_class)


def uninstall(runtime_class):
    """Test/process cleanup only; active scopes require an explicitly sealed worker."""
    owner = getattr(runtime_class, _ATTRIBUTE, None)
    if owner is None:
        raise ObservationError("No observer installation for this runtime class")
    if owner.active and not owner.sealed:
        raise ObservationError("Cannot remove observer with active scopes")
    for state in owner.all_states:
        state.remove()
        state.frames.set(())
    owner.scopes.set(())
    for name, original in owner.own_originals.items():
        if original is None:
            delattr(runtime_class, name)
        else:
            setattr(runtime_class, name, original)
    delattr(runtime_class, _ATTRIBUTE)
