"""Explicit numerical recipe dispatch; CPU fixtures do not certify GPU parity."""

import copy
from types import SimpleNamespace

import pytest
import torch

from sr_f4.common import read, write
from sr_f5 import runtime as module


@pytest.mark.parametrize(
    "recipe",
    [
        None,
        {},
        {**module.NUMERICS_V2, "recipe_version": 1},
        {**module.NUMERICS_V2, "recipe_version": 2.0},
        {**module.NUMERICS_V2, "model_dtype": "bfloat16"},
        {**module.NUMERICS_V2, "input_float_dtype": "bfloat16"},
        {**module.NUMERICS_V2, "probability_dtype": "float64"},
        {**module.NUMERICS_V2, "extra": False},
        {k: v for k, v in module.NUMERICS_V2.items() if k != "probability_dtype"},
    ],
)
def test_unknown_or_incomplete_numerical_recipe_is_rejected(recipe):
    with pytest.raises(PermissionError, match="numerical recipe"):
        module.registered_numerics({"numerics": recipe})


def test_absent_recipe_is_v1_and_v2_is_copied():
    assert module.registered_numerics({}) is None
    config = {"numerics": dict(module.NUMERICS_V2)}
    registered = module.registered_numerics(config)
    assert registered == module.NUMERICS_V2
    registered["recipe_version"] = 99
    assert config["numerics"] == module.NUMERICS_V2


@pytest.fixture
def loading(tmp_path, monkeypatch):
    import mm_core.vl_runtime
    import mm_dev.runtime
    import sr_f3_ce.training

    events = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
            self.base.requires_grad_(False)
            self.adapter = torch.nn.Linear(2, 2, dtype=torch.float32)
            self.register_buffer("float_buffer", torch.ones(2, dtype=torch.bfloat16))
            self.register_buffer("int_buffer", torch.tensor([1, 2], dtype=torch.int64))

        def float(self):
            events.append("model.float")
            return super().float()

    def construct(self, *args, **kwargs):
        events.append("load")
        self.torch = torch
        self.model = Model()
        self.identity = dict(dtype="bfloat16", processor_hash="native-processor-hash")

    def configure(self, *, parent_adapter_config):
        events.append("configure_parent")
        assert self.model.base.weight.dtype == torch.bfloat16
        assert self.model.adapter.weight.dtype == torch.float32
        self.identity["parent_verified"] = True

    native_optimizer = sr_f3_ce.training.fresh_optimizer

    def fresh(model):
        events.append("fresh_optimizer")
        return native_optimizer(model)

    monkeypatch.setattr(module.NativeRuntime, "__init__", construct)
    monkeypatch.setattr(module, "configure_parent_training", configure)
    monkeypatch.setattr(mm_core.vl_runtime, "seed_all", lambda _: None)
    monkeypatch.setattr(mm_dev.runtime, "configure_audited_backend", lambda: None)
    monkeypatch.setattr(mm_dev.runtime, "actual_cuda_identity", lambda: {"fixture": "CPU_ONLY"})
    monkeypatch.setattr(sr_f3_ce.training, "fresh_optimizer", fresh)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    adapter = tmp_path / "parent"
    write(adapter / "adapter_config.json", {})
    config = dict(
        parents=["b0"],
        master_seed=1,
        model=dict(parents={"b0": str(adapter)}, model_path="unused_cpu_fixture", snapshot="test"),
    )
    return SimpleNamespace(root=tmp_path, config=config, events=events)


def test_v1_load_preserves_native_mixed_precision_and_identity(loading):
    write(loading.root / "CONFIG.json", loading.config)
    runtime, optimizer, scheduler = module.load(loading.root, "b0", "v1", training=True)
    assert loading.events == ["load", "configure_parent", "fresh_optimizer"]
    assert runtime.model.base.weight.dtype == torch.bfloat16
    assert runtime.model.adapter.weight.dtype == torch.float32
    assert runtime.model.float_buffer.dtype == torch.bfloat16
    assert runtime.identity["dtype"] == "bfloat16"
    assert "numerics" not in runtime.identity
    assert "numerics_recipe_version" not in runtime.stable_model_identity()
    assert not hasattr(runtime, "_srf5_numerics")
    assert optimizer.param_groups[0]["params"][0] is runtime.model.adapter.weight
    assert scheduler.optimizer is optimizer


def test_v2_casts_full_model_after_parent_verification_before_optimizer(loading):
    config = dict(loading.config, numerics=dict(module.NUMERICS_V2))
    write(loading.root / "CONFIG.json", config)
    runtime, optimizer, scheduler = module.load(loading.root, "b0", "v2", training=True)
    assert loading.events == ["load", "configure_parent", "model.float", "fresh_optimizer"]
    assert all(p.dtype == torch.float32 for p in runtime.model.parameters())
    assert runtime.model.float_buffer.dtype == torch.float32
    assert runtime.model.int_buffer.dtype == torch.int64
    assert runtime.model.int_buffer.tolist() == [1, 2]
    assert runtime.model.base.weight.requires_grad is False
    assert runtime.model.adapter.weight.requires_grad is True
    assert optimizer.param_groups[0]["params"][0] is runtime.model.adapter.weight
    assert scheduler.optimizer is optimizer
    assert runtime.identity["dtype"] == "float32"
    assert runtime.identity["model_cast_from_dtype"] == "bfloat16"
    assert runtime.identity["processor_hash"] == "native-processor-hash"
    stable = runtime.stable_model_identity()
    assert stable["numerics"] == module.NUMERICS_V2
    assert stable["numerics_recipe_version"] == 2
    assert stable["forward_parameter_dtypes"] == ["float32"]
    assert stable["dtype"] == "float32"
    saved = read(loading.root / "runtime_identity/v2.json")
    assert saved["dtype"] == "float32" and saved["numerics_recipe_version"] == 2
    stable["numerics"]["recipe_version"] = 99
    assert runtime._srf5_numerics == module.NUMERICS_V2


def test_invalid_recipe_stops_before_model_loading(loading):
    write(loading.root / "CONFIG.json", dict(loading.config, numerics=None))
    with pytest.raises(PermissionError, match="numerical recipe"):
        module.load(loading.root, "b0", "invalid", training=True)
    assert loading.events == []


@pytest.fixture
def prepared(monkeypatch):
    tensors = dict(
        input_ids=torch.tensor([[11, 12]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        boolean_mask=torch.tensor([[True, False]]),
        image_grid_thw=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        pixel_values=torch.tensor([[1.5, 2.5]], dtype=torch.bfloat16),
        another_float=torch.tensor([1.0], dtype=torch.float64),
    )
    route = dict(
        processor_hash="original-processor",
        processed_pixel_sha256="original-pixels",
        input_tensor_hash="original-inputs",
        native_input_tensor_hashes={"pixel_values": "native"},
    )
    native = dict(inputs=tensors, routing=copy.deepcopy(route))
    calls = []

    def native_prepare(self, model_input, run_root, *, protocol):
        calls.append(protocol)
        return native

    monkeypatch.setattr(module.NativeRuntime, "prepare_input", native_prepare)
    runtime = module.Runtime.__new__(module.Runtime)
    runtime.torch = torch
    return SimpleNamespace(runtime=runtime, native=native, route=route, calls=calls)


def test_v1_prepared_inputs_keep_original_objects_and_dtypes(prepared, tmp_path):
    model_input = dict(prompt="public", image_file=None, image_sha256=None)
    result = prepared.runtime.prepare_input(model_input, tmp_path)
    assert result is prepared.native
    assert result["inputs"]["pixel_values"].dtype == torch.bfloat16
    assert result["inputs"]["another_float"].dtype == torch.float64
    assert result["routing"] == dict(prepared.route, sr_f5_protocol_version=module.PROTOCOL_VERSION)
    assert prepared.runtime.prepare_input(model_input, tmp_path) is result
    assert prepared.calls == ["structured"]


def test_v2_inputs_are_cached_float32_without_changing_tokens_or_processor_hashes(
    prepared, tmp_path
):
    runtime = prepared.runtime
    runtime._srf5_numerics = dict(module.NUMERICS_V2)
    model_input = dict(prompt="public", image_file=None, image_sha256=None)
    result = runtime.prepare_input(model_input, tmp_path)
    assert result is not prepared.native
    for name in ("pixel_values", "another_float"):
        assert result["inputs"][name].dtype == torch.float32
        assert torch.equal(result["inputs"][name], prepared.native["inputs"][name].float())
    for name in ("input_ids", "attention_mask", "image_grid_thw", "boolean_mask"):
        assert result["inputs"][name] is prepared.native["inputs"][name]
    assert prepared.native["inputs"]["pixel_values"].dtype == torch.bfloat16
    assert prepared.native["inputs"]["another_float"].dtype == torch.float64
    assert prepared.native["routing"] == prepared.route
    for name, value in prepared.route.items():
        assert result["routing"][name] == value
    assert result["routing"]["forward_input_dtypes"]["pixel_values"] == "float32"
    assert result["routing"]["forward_input_dtypes"]["input_ids"] == "int64"
    assert result["routing"]["numerics_recipe_version"] == 2
    assert runtime.prepare_input(model_input, tmp_path) is result
    assert prepared.calls == ["structured"]
