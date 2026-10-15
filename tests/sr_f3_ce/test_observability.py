"""Actual CPU torch calls; no scientific models, samples, updates or GPU claims."""

import copy
import importlib.util
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

SOURCE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/observe_runtime.py"
SPEC = importlib.util.spec_from_file_location("srf3ce_observer_tests", SOURCE)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.failure = None
        self.none_return = False

    def forward(self, hidden_states, grid_thw=None):
        if self.failure is not None:
            raise self.failure
        if self.none_return:
            return None
        return self.linear(hidden_states).tanh()


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 3)
        self.linear = torch.nn.Linear(3, 20)
        self.visual = Visual()
        self.trace = []
        self.gradient_checkpointing = False
        self.use_cache = True
        self.nested_call = None

    def forward(self, input_ids, *, pixel_values=None, grid_thw=None):
        self.trace.append(tuple(input_ids.shape))
        callback, self.nested_call = self.nested_call, None
        if callback is not None:
            callback()
        hidden = self.embedding(input_ids)
        if pixel_values is not None:
            if self.gradient_checkpointing:
                visual = checkpoint(self.visual, pixel_values, grid_thw, use_reentrant=False)
            else:
                visual = self.visual(pixel_values, grid_thw=grid_thw)
            hidden = hidden + visual[:, None, :]
        return self.linear(hidden)

    def generate(self, input_ids, *, pixel_values=None, grid_thw=None, steps=3):
        tokens = []
        for step in range(steps):
            logits = self(
                input_ids,
                pixel_values=pixel_values if step == 0 else None,
                grid_thw=grid_thw if step == 0 else None,
            )
            token = torch.multinomial(logits[:, -1].softmax(-1), 1)
            tokens.append(token)
            input_ids = token
        return torch.cat(tokens, dim=-1)


class Adapter(torch.nn.Module):
    """PEFT-shaped delegation: generate never calls this wrapper's forward."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def get_base_model(self):
        return self.base

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)


def runtime_type():
    class ToyRuntime:
        def __init__(self, *, adapter=True, checkpointing=False):
            base = Core()
            base.gradient_checkpointing = checkpointing
            self.model = Adapter(base) if adapter else base

        def _visual_module(self):
            base = (
                self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            )
            return base.visual

        def generate_group(self, inputs, *, protocol="structured", steps=3, nested=False):
            if nested:
                self._native_forward(inputs, purpose="nested", grad=True)
            return self.model.generate(**inputs, steps=steps)

        def _native_forward(self, inputs, *, purpose="training_policy", grad=True):
            return self.model(**inputs)

    return ToyRuntime


def inputs():
    return dict(
        input_ids=torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        pixel_values=torch.tensor([[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]]),
        grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
    )


def base_model(runtime):
    return (
        runtime.model.get_base_model()
        if hasattr(runtime.model, "get_base_model")
        else runtime.model
    )


@pytest.fixture
def installation():
    owned = []

    def make(emit=None):
        events = []
        cls = runtime_type()
        identity = observer.install(cls, events.append if emit is None else emit)
        owned.append(cls)
        return cls, events, identity

    yield make
    for cls in owned:
        observer.uninstall(cls)


def ends(events, kind):
    return [e for e in events if e["event"] == "SCOPE_END" and e["kind"] == kind]


def rng():
    return torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state())


def assert_rng_equal(left, right):
    assert torch.equal(left[0], right[0])
    assert left[1] == right[1]
    assert left[2][0] == right[2][0]
    assert np.array_equal(left[2][1], right[2][1])
    assert left[2][2:] == right[2][2:]


def execution(cls, checkpointing=False):
    torch.manual_seed(781)
    random.seed(731)
    np.random.seed(881)
    value = inputs()
    runtime = cls(checkpointing=checkpointing)
    optimizer = torch.optim.AdamW(runtime.model.parameters(), lr=1e-5)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    parameter_before = {n: p.detach().clone() for n, p in runtime.model.named_parameters()}
    modes = [m.training for m in runtime.model.modules()]
    with torch.no_grad():
        generated = runtime.generate_group(value)
    forward = runtime._native_forward(value, purpose="training_support", grad=True)
    forward.square().mean().backward()
    assert optimizer.state_dict() == optimizer_before  # No Adam call is needed for validation.
    assert [m.training for m in runtime.model.modules()] == modes
    assert base_model(runtime).gradient_checkpointing is checkpointing
    assert base_model(runtime).use_cache is True
    assert all(torch.equal(parameter_before[n], p) for n, p in runtime.model.named_parameters())
    return dict(
        generated=generated,
        output=forward.detach(),
        gradients={n: p.grad.clone() for n, p in runtime.model.named_parameters()},
        parameters=parameter_before,
        rng=rng(),
        trace=base_model(runtime).trace,
        input_versions={k: v._version for k, v in value.items()},
    )


@pytest.mark.parametrize("checkpointing", [False, True])
def test_exact_outputs_gradients_parameters_rng_and_original_state(installation, checkpointing):
    baseline = execution(runtime_type(), checkpointing)
    cls, events, identity = installation()
    json.dumps(identity, allow_nan=False)
    observed = execution(cls, checkpointing)
    assert torch.equal(baseline["generated"], observed["generated"])
    assert torch.equal(baseline["output"], observed["output"])
    for name in baseline["gradients"]:
        assert torch.equal(baseline["gradients"][name], observed["gradients"][name])
        assert torch.equal(baseline["parameters"][name], observed["parameters"][name])
    assert_rng_equal(baseline["rng"], observed["rng"])
    assert baseline["trace"] == observed["trace"]
    assert baseline["input_versions"] == observed["input_versions"]
    assert all(e["coverage_status"] == "COMPLETE" for e in events if e["event"] == "SCOPE_END")
    json.dumps(observer.snapshot(cls), allow_nan=False)


def test_delegated_generation_actual_phases_and_nonadditive_layer_counts(installation):
    cls, events, _ = installation()
    runtime = cls()
    original_forward = base_model(runtime).forward.__func__
    runtime.generate_group(inputs())
    assert base_model(runtime).forward.__func__ is original_forward
    end = ends(events, "generation")[0]
    prefill, decode = end["model_stages"]["prefill"], end["model_stages"]["incremental_decode"]
    assert (prefill["entered"], prefill["returned"], prefill["input_sequence_slots"]) == (1, 1, 8)
    assert (decode["entered"], decode["returned"], decode["input_sequence_slots"]) == (2, 2, 4)
    assert (decode["input_batch_items"], end["visual_stages"]["prefill"]["image_items"]) == (4, 2)
    assert end["visual_stages"]["prefill"]["entered"] == 1
    assert set(end["model_hook_layers"]) == {"base"}
    assert len(events) == 4  # Init BEGIN/END + generation BEGIN/END; no per-token writes.
    assert end["visual_expected"] is True


@pytest.mark.parametrize("adapter", [False, True])
def test_tf_model_and_base_are_not_counted_twice(installation, adapter):
    cls, events, _ = installation()
    runtime = cls(adapter=adapter)
    runtime._native_forward(inputs(), purpose="training_reference", grad=False)
    end = ends(events, "teacher_forcing")[0]
    assert end["purpose"] == "training_reference"
    assert end["model_stages"]["teacher_forcing"]["entered"] == 1
    assert end["model_stages"]["teacher_forcing"]["input_sequence_slots"] == 8
    assert end["visual_stages"]["teacher_forcing"]["entered"] == 1
    assert set(end["model_hook_layers"]) == ({"base", "model"} if adapter else {"model=base"})


def test_nested_runtime_scopes_own_their_calls(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime.generate_group(inputs(), nested=True)
    parent = ends(events, "generation")[0]
    child = ends(events, "teacher_forcing")[0]
    assert child["parent_scope_id"] == parent["scope_id"]
    assert parent["model_stages"]["prefill"]["entered"] == 1
    assert "teacher_forcing" not in parent["model_stages"]
    assert child["model_stages"]["teacher_forcing"]["entered"] == 1
    assert parent["host_scope_elapsed_ns"] >= child["host_scope_elapsed_ns"]
    assert not observer.snapshot(cls)["active_scopes"]


def test_nested_model_reentry_exclusive_host_interval_is_not_double_counted(installation):
    cls, events, _ = installation()
    runtime = cls()
    base_model(runtime).nested_call = lambda: runtime._native_forward(inputs(), purpose="nested")
    runtime.generate_group(inputs())
    parent, child = ends(events, "generation")[0], ends(events, "teacher_forcing")[0]
    assert child["parent_scope_id"] == parent["scope_id"]
    prefill = parent["model_stages"]["prefill"]
    child_model = child["model_stages"]["teacher_forcing"]
    assert (
        prefill["host_inclusive_ns"] - prefill["host_exclusive_ns"]
        == child_model["host_inclusive_ns"]
    )
    assert prefill["entered"] == child_model["entered"] == 1


def test_original_exception_is_preserved_with_actual_failed_hooks(installation):
    cls, events, _ = installation()
    runtime = cls()
    failure = ValueError("original model failure")
    runtime._visual_module().failure = failure
    with pytest.raises(ValueError) as caught:
        runtime._native_forward(inputs())
    assert caught.value is failure
    end = ends(events, "teacher_forcing")[0]
    assert end["outcome"] == "RAISED"
    assert end["model_stages"]["teacher_forcing"]["raised"] == 1
    assert end["visual_stages"]["teacher_forcing"]["raised"] == 1
    assert end["coverage_status"] == "COMPLETE"
    assert not observer.snapshot(cls)["active_scopes"]


def test_normal_none_return_inside_ambient_exception_is_not_failure(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime._visual_module().none_return = True
    try:
        raise ValueError("ambient")
    except ValueError:
        assert runtime._visual_module()(inputs()["pixel_values"]) is None
    end = ends(events, "unscoped_visual_forward")[0]
    assert end["outcome"] == "RETURNED"
    assert end["visual_stages"]["unscoped_visual"]["returned"] == 1
    assert end["visual_stages"]["unscoped_visual"]["image_shape_unknown_calls"] == 1
    assert end["origin_scope_id"] == "UNKNOWN"


def test_checkpoint_recompute_has_independent_scope_and_is_not_scientific_failure(installation):
    cls, events, _ = installation()
    runtime = cls(checkpointing=True)
    forward = runtime._native_forward(inputs())
    forward.square().mean().backward()
    scope = ends(events, "unscoped_visual_forward")[0]
    assert scope["origin_scope_id"] == "UNKNOWN"
    assert scope["parent_scope_id"] is None
    assert scope["autograd_graph_task_id"] >= 0
    assert scope["outcome"] == "CHECKPOINT_EARLY_STOP"
    assert scope["visual_stages"]["unscoped_visual"]["checkpoint_early_stop"] == 1
    assert scope["visual_stages"]["unscoped_visual"]["raised"] == 0
    runtime._native_forward(inputs())
    last = ends(events, "teacher_forcing")[-1]
    assert last["visual_stages"]["teacher_forcing"]["entered"] == 1
    assert not observer.snapshot(cls)["active_scopes"]


@pytest.mark.parametrize("where", ["SCOPE_BEGIN", "SCOPE_END"])
def test_writer_error_never_retries_and_marks_whether_original_ran(installation, where):
    events = []

    def emit(event):
        if event["kind"] == "generation" and event["event"] == where:
            raise OSError("disk unavailable")
        events.append(event)

    cls, _, _ = installation(emit)
    runtime = cls()
    before = rng()
    with pytest.raises(observer.ObservationWriteError) as caught:
        runtime.generate_group(inputs())
    assert caught.value.original_call_started is (where == "SCOPE_END")
    assert caught.value.original_call_returned is (where == "SCOPE_END")
    assert len(base_model(runtime).trace) == (3 if where == "SCOPE_END" else 0)
    if where == "SCOPE_BEGIN":
        assert_rng_equal(before, rng())
    report = observer.snapshot(cls)
    assert len(report["write_errors"]) == 1
    assert not report["active_scopes"]
    assert not ends(events, "generation")


def test_model_error_and_writer_error_keep_original_error(installation):
    def emit(event):
        if event["kind"] == "teacher_forcing" and event["event"] == "SCOPE_END":
            raise OSError("disk unavailable")

    cls, _, _ = installation(emit)
    runtime = cls()
    failure = torch.cuda.OutOfMemoryError("toy failure; no CUDA initialization")
    runtime._visual_module().failure = failure
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        runtime._native_forward(inputs())
    assert caught.value is failure
    assert any("Observation SCOPE_END write failed" in note for note in caught.value.__notes__)
    assert observer.snapshot(cls)["write_errors"]


def test_emitter_reentry_is_rejected_before_additional_scientific_call(installation):
    current = []

    def emit(event):
        if current and event["kind"] == "generation":
            current[0].generate_group(inputs())

    cls, _, _ = installation(emit)
    current.append(cls())
    with pytest.raises(observer.ObservationWriteError):
        current[0].generate_group(inputs())
    assert not base_model(current[0]).trace


def test_interrupt_unreturned_hooks_unknown_and_next_scope_is_clean(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime._visual_module().failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        runtime.generate_group(inputs())
    end = ends(events, "generation")[0]
    assert end["outcome"] == "INTERRUPTED"
    assert end["coverage_status"] == "UNKNOWN"
    assert end["model_stages"]["prefill"]["unresolved"] == 1
    assert end["visual_stages"]["prefill"]["unresolved"] == 1
    runtime._visual_module().failure = None
    runtime.generate_group(inputs())
    assert ends(events, "generation")[-1]["coverage_status"] == "COMPLETE"


@pytest.mark.parametrize("failure", ["model", "visual"])
def test_missing_expected_hooks_cannot_be_reported_as_measured_zero(installation, failure):
    cls, events, _ = installation()
    runtime = cls()
    if failure == "model":
        base_model(runtime).generate = lambda *args, **kwargs: "bypassed"
    else:
        # Simulate a changed backend bypassing visual __call__, while root hooks remain.
        base_model(runtime).visual.forward = lambda *args, **kwargs: torch.zeros(2, 3)
        for handle in getattr(cls, observer._ATTRIBUTE).states[runtime].handles[-3:]:
            handle.remove()
    with pytest.raises(observer.ObservationError, match="coverage is UNKNOWN"):
        runtime.generate_group(inputs())
    end = ends(events, "generation")[0]
    assert end["coverage_status"] == "UNKNOWN"
    expected = (
        "NO_PRIMARY_MODEL_FORWARD_OBSERVED"
        if failure == "model"
        else "EXPECTED_VISUAL_FORWARD_NOT_OBSERVED"
    )
    assert expected in end["coverage_issues"]


def test_explicit_installation_json_overhead_duplicate_rejection_and_cleanup():
    cls, events = runtime_type(), []
    before = rng()
    originals = {name: getattr(cls, name) for name in observer._METHODS}
    identity = observer.install(cls, events.append)
    assert_rng_equal(before, rng())
    assert identity["old_engine_observed"] is False
    assert identity["observer_source_sha256"] == observer._sha(SOURCE)
    assert set(identity["original_methods"]) == set(observer._METHODS)
    with pytest.raises(observer.ObservationError, match="already installed"):
        observer.install(cls, events.append)
    runtime = cls()
    runtime.generate_group(inputs())
    state = observer.snapshot(cls)
    assert state["events_written"] == {"SCOPE_BEGIN": 2, "SCOPE_END": 2}
    assert all(value > 0 for value in state["observer_overhead_ns"].values())
    assert not state["write_errors"]
    json.dumps(events, allow_nan=False)
    observer.uninstall(cls)
    assert all(getattr(cls, name) is function for name, function in originals.items())
    assert not runtime._visual_module()._forward_hooks
    runtime.generate_group(inputs())
    assert len(events) == 4


def test_existing_runtime_refused_and_empty_installation_is_valid():
    cls, events = runtime_type(), []
    runtime = cls()
    observer.install(cls, events.append)
    try:
        assert observer.snapshot(cls)["events_written"] == {}
        with pytest.raises(observer.ObservationError, match="predates"):
            runtime.generate_group(inputs())
        assert events == []
    finally:
        observer.uninstall(cls)


def test_actual_frozen_class_can_install_without_constructing_model_or_changing_source():
    from sr_f3_ce.runtime import SRF3CERuntime

    inspect_path = Path(inspect.getsourcefile(SRF3CERuntime))
    before = observer._sha(inspect_path)
    identity = observer.install(SRF3CERuntime, lambda event: None)
    try:
        assert identity["runtime_class"] == "sr_f3_ce.runtime.SRF3CERuntime"
        assert not observer.snapshot(SRF3CERuntime)["events_written"]
        assert observer._sha(inspect_path) == before
    finally:
        observer.uninstall(SRF3CERuntime)


def test_constructor_return_then_hook_attachment_failure_retains_true_boundary(installation):
    cls, events, _ = installation()
    cls._visual_module = lambda self: None
    runtime = cls.__new__(cls)
    with pytest.raises(observer.ObservationError, match="actual torch Modules"):
        cls.__init__(runtime)
    end = ends(events, "initialization")[0]
    assert end["original_call_returned"] is True
    assert end["outcome"] == "RAISED"
    assert end["exception_type"] == "ObservationError"
    assert not runtime.model._forward_hooks
    assert not base_model(runtime)._forward_hooks
    assert not observer.snapshot(cls)["active_scopes"]


def test_unscoped_full_model_observed_once_and_following_scope_uncontaminated(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime.model(**inputs())
    unscoped = ends(events, "unscoped_model_forward")[0]
    assert unscoped["origin_scope_id"] == "UNKNOWN"
    assert unscoped["model_stages"]["unscoped_forward"]["entered"] == 1
    assert unscoped["visual_stages"]["unscoped_forward"]["entered"] == 1
    runtime.generate_group(inputs(), steps=17)
    end = ends(events, "generation")[0]
    assert end["model_stages"]["prefill"]["entered"] == 1
    assert end["model_stages"]["incremental_decode"]["entered"] == 16
    assert len(events) == 6  # No decode-length-dependent writes.


def test_text_only_forward_has_legitimate_zero_visual_calls(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime.generate_group(dict(input_ids=inputs()["input_ids"]))
    end = ends(events, "generation")[0]
    assert end["coverage_status"] == "COMPLETE"
    assert end["visual_expected"] is False
    assert end["visual_stages"] == {}


def test_hooks_removed_even_if_only_model_outlives_runtime():
    cls = runtime_type()
    observer.install(cls, lambda event: None)
    runtime = cls()
    model = runtime.model
    del runtime
    observer.uninstall(cls)
    assert not model._forward_hooks
    assert not model.get_base_model()._forward_hooks
    assert not model.get_base_model().visual._forward_hooks


def test_unscoped_interrupt_sealed_unknown_without_fabricating_end_or_continuation(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime._visual_module().failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        runtime._visual_module()(inputs()["pixel_values"])
    assert not ends(events, "unscoped_visual_forward")
    assert len(observer.snapshot(cls)["active_scopes"]) == 1
    terminal = observer.seal(cls)
    assert terminal["sealed"] is True
    assert terminal == observer.seal(cls)
    runtime._visual_module().failure = None
    with pytest.raises(observer.ObservationError, match="sealed"):
        runtime._native_forward(inputs())
    with pytest.raises(observer.ObservationError, match="sealed"):
        runtime._visual_module()(inputs()["pixel_values"])
    assert not base_model(runtime).trace
    assert observer.snapshot(cls)["active_scopes"] == terminal["active_scopes"]
    assert not ends(events, "unscoped_visual_forward")
    assert len(events) == 3  # Initializer pair and the unmatched real BEGIN only.


def test_normal_terminal_seal_is_idempotent_and_rejects_new_scopes(installation):
    cls, events, _ = installation()
    runtime = cls()
    runtime.generate_group(inputs())
    terminal = observer.seal(cls)
    assert terminal["sealed"] and not terminal["active_scopes"]
    assert terminal == observer.seal(cls)
    with pytest.raises(observer.ObservationError, match="sealed"):
        runtime.generate_group(inputs())
    assert len(events) == 4
