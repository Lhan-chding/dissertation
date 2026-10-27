from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sr_f6 import runtime
from sr_f6.source_binding import inventory_file


def test_registered_cap_is_local_and_legacy_unchanged():
    from sr_f3_ce.runtime import generation_recipe as native

    assert native()["max_new_tokens"] == 1024
    assert runtime.generation_recipe()["max_new_tokens"] == 1536
    assert {k: v for k, v in runtime.generation_recipe().items() if k != "max_new_tokens"} == {
        k: v for k, v in native().items() if k != "max_new_tokens"
    }


def test_config_rejects_silent_numeric_or_sampling_change():
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    runtime.validate_generation_config(config)
    bad = copy.deepcopy(config)
    bad["training"]["max_new_tokens"] = 1024
    with pytest.raises(PermissionError):
        runtime.validate_generation_config(bad)
    bad = copy.deepcopy(config)
    bad["numerics"]["model_dtype"] = "bfloat16"
    with pytest.raises(PermissionError):
        runtime.validate_generation_config(bad)


def test_actual_native_body_uses_1536_and_saves_original_tokens(monkeypatch, tmp_path):
    class Config(SimpleNamespace):
        def to_dict(self):
            return vars(self)

    class Model:
        training = False

        def eval(self):
            pass

        def train(self, value):
            self.training = value

        def generate(self, input_ids, generation_config, **kwargs):
            assert generation_config.max_new_tokens == 1536
            assert generation_config.temperature == 1.0
            n = generation_config.max_new_tokens
            self.called = True
            logits = torch.tensor([[0.0, 1.0]])
            return SimpleNamespace(
                sequences=torch.cat((input_ids, torch.ones((1, n), dtype=torch.long)), 1),
                logits=[logits] * n,
                scores=[logits] * n,
            )

    r = runtime.Runtime.__new__(runtime.Runtime)
    r.torch = torch
    r.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda *a, **kw: "x")
    )
    r.model = Model()
    r.identity = dict(
        parent_content_hash="a" * 64,
        source_sha256="b" * 64,
        data_hash="c" * 64,
        run_id="fixture",
        step=0,
    )
    monkeypatch.setattr(runtime, "trainable_state", lambda model: {"p": torch.tensor([1.0])})
    r.protocol_revision = "v2"
    r.eos_ids = []
    r.image_calls = 0
    r.generation_config = Config()
    prepared = dict(inputs={"input_ids": torch.tensor([[0]])}, routing={"input_tensor_hash": "h"})
    r._srf5_prepared = prepared
    r.prepare_input = lambda *a, **kw: prepared
    r.reserve = lambda *a, **kw: None
    r.current_adapter_identity = lambda: {"parent": "b0", "step": 0}
    r.stable_model_identity = lambda: {"dtype": "float32"}
    monkeypatch.setattr(runtime, "capture_rng", lambda: None)
    monkeypatch.setattr(runtime, "restore_rng", lambda x: None)
    monkeypatch.setattr(runtime, "seed_all", lambda x: None)
    monkeypatch.setattr(runtime, "_repeat_prompt", lambda x, n: x)
    monkeypatch.setattr(
        runtime, "BatchedProtocolStop", lambda *a, **kw: SimpleNamespace(counts=[], cuts=[])
    )
    monkeypatch.setattr(runtime, "first_boundary_token", lambda *a, **kw: (None, None))
    monkeypatch.setattr(runtime, "decode_generated", lambda tok, ids: "x" * len(ids))
    monkeypatch.setattr(
        sys, "modules", dict(sys.modules, transformers=SimpleNamespace(StoppingCriteriaList=list))
    )
    rows = r.generate_group(
        {"prompt": "q", "image_file": None, "image_sha256": None}, tmp_path, [19]
    )
    assert r.model.called
    assert len(rows) == 1
    row = rows[0]
    assert len(row["tokens"]) == len(row["sampler_logprobs"]) == 1536
    assert row["max_new_tokens"] == 1536
    assert row["finish_reason"] == "length"
    assert row["protocol"] == "structured_v4"
    assert row["technical_validation_errors"] == []
    assert row["seed_semantics"] == "independent_per_draw_torch_generator"


def test_binding_refuses_external_symlink(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not ours")
    (owned / "escape").symlink_to(outside)
    with pytest.raises(PermissionError):
        inventory_file(owned / "escape", owned)
    (owned / "ok").write_text("ours")
    result = inventory_file(owned / "ok", owned)
    assert result["bytes"] == 4
    assert len(result["sha256"]) == 64


def test_teacher_forcing_accepts_1536_without_reencoding_and_rejects_1537():
    r = runtime.Runtime.__new__(runtime.Runtime)
    r.torch = torch
    r.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    r.eos_ids = []
    r._validate_prepared_revision = lambda _: None
    prepared = {
        "inputs": {
            "input_ids": torch.tensor([[6, 7]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0]]),
        }
    }
    original = torch.arange(1536).tolist()

    def forward(inputs, *, completion_start, lengths, purpose, grad):
        assert completion_start == 2
        assert lengths == [1536]
        assert inputs["input_ids"][0, 2:].tolist() == original
        assert inputs["attention_mask"].sum().item() == 1538
        return ["preserved"]

    r._native_forward = forward
    assert r._batch_forward(prepared, [original], purpose="training_policy", grad=True) == [
        "preserved"
    ]
    with pytest.raises(ValueError):
        r._batch_forward(prepared, [[*original, 0]], purpose="training_policy", grad=True)


def test_native_sampling_copy_preserves_all_algorithm_statements():
    import ast
    import inspect
    import textwrap

    from sr_f3_ce.runtime import SRF3CERuntime

    native = ast.parse(textwrap.dedent(inspect.getsource(SRF3CERuntime.generate_group)))
    copied = ast.parse(textwrap.dedent(inspect.getsource(runtime.Runtime._generate_native_group)))
    copied.body[0].name = native.body[0].name
    assert ast.dump(native, include_attributes=False) == ast.dump(copied, include_attributes=False)


def test_bound_content_cache_rehashes_only_metadata_changes(monkeypatch, tmp_path):
    import os

    from sr_f6 import source_binding as binding

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"original")
    bound = binding.inventory_file(path, tmp_path)
    original_hash = binding.file_sha256
    calls = []

    def counted(value):
        calls.append(str(value))
        return original_hash(value)

    monkeypatch.setattr(binding, "file_sha256", counted)
    result = binding.verify_bound_files([bound], tmp_path)
    assert result["content_rehashed"] == 0
    assert calls == []
    current = path.stat()
    os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
    result = binding.verify_bound_files([bound], tmp_path)
    assert result["content_rehashed"] == 1
    assert calls == [str(path)]
    result = binding.verify_bound_files([bound], tmp_path)
    assert result["content_rehashed"] == 0
    assert calls == [str(path)]
    path.write_bytes(b"modified")
    with pytest.raises(PermissionError, match="content changed"):
        binding.verify_bound_files([bound], tmp_path)
    assert calls == [str(path), str(path)]


def test_bound_cache_rejects_retargeted_symlink(tmp_path):
    from sr_f6 import source_binding as binding

    first, second, link = [tmp_path / name for name in ("first", "second", "link")]
    first.write_text("same")
    second.write_text("same")
    link.symlink_to(first)
    bound = binding.inventory_file(link, tmp_path)
    link.unlink()
    link.symlink_to(second)
    with pytest.raises(PermissionError, match="path changed"):
        binding.verify_bound_files([bound], tmp_path)


def test_stat_cache_metadata_is_not_scientific_content_identity(tmp_path):
    from sr_f6 import source_binding as binding

    path = tmp_path / "parent"
    path.write_text("parent weights")
    bound = binding.inventory_file(path, tmp_path)
    metadata_only = copy.deepcopy(bound)
    metadata_only["stat_cache_key"]["mtime_ns"] += 1
    assert binding.object_sha256(binding.content_inventory([bound])) == binding.object_sha256(
        binding.content_inventory([metadata_only])
    )


def test_policy_hash_cached_at_committed_step_boundaries(monkeypatch):
    r = runtime.Runtime.__new__(runtime.Runtime)
    r.model = torch.nn.Linear(1, 1)
    r.identity = {"step": 0}
    r._srf6_training = True
    calls = []

    def state(model):
        calls.append(r.identity["step"])
        return {"weight": torch.tensor([float(r.identity["step"])])}

    monkeypatch.setattr(runtime, "trainable_state", state)
    first = r.policy_tensor_identity()
    assert r.policy_tensor_identity() == first
    assert calls == [0]
    r.identity["step"] = 1
    second = r.policy_tensor_identity()
    assert second != first
    assert r.policy_tensor_identity() == second
    assert calls == [0, 1]
    r._srf6_training = False
    r.identity["step"] = 0
    r._srf6_policy_identity = (0, first)
    assert r.policy_tensor_identity() == first
    assert calls == [0, 1]
    r.identity["step"] = 1
    with pytest.raises(PermissionError):
        r.policy_tensor_identity()


def test_load_boundary_rejects_parent_drift_before_model_load(monkeypatch, tmp_path):
    from sr_f6 import source_binding as binding

    monkeypatch.setattr(binding, "PROJECT", tmp_path)
    model, parent = tmp_path / "model", tmp_path / "parent"
    model.mkdir()
    parent.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (model / name).write_text("model")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (parent / name).write_text("parent")
    config = {"model": {"model_path": str(model), "parents": {"b0": str(parent)}}}
    (tmp_path / "CONFIG.json").write_text(json.dumps(config))
    parent_files = [binding.inventory_file(p, tmp_path) for p in sorted(parent.iterdir())]
    bound = dict(
        config_file_sha256=binding.file_sha256(tmp_path / "CONFIG.json"),
        model_files=[binding.inventory_file(p, tmp_path) for p in sorted(model.iterdir())],
        parents={
            "b0": dict(
                path=str(parent),
                files=parent_files,
                content_sha256=binding.object_sha256(binding.content_inventory(parent_files)),
            )
        },
    )
    (tmp_path / "SOURCE_BINDING.json").write_text(json.dumps(bound))
    _, checked = runtime.verify_binding_inputs(tmp_path, config, "b0")
    assert checked["model"]["content_rehashed"] == checked["parent"]["content_rehashed"] == 0
    (parent / "adapter_model.safetensors").write_text("drifts")
    with pytest.raises(PermissionError, match="content changed"):
        runtime.verify_binding_inputs(tmp_path, config, "b0")


def test_shared_mount_device_number_is_not_a_content_cache_key():
    from sr_f6.source_binding import stat_cache_key

    one = SimpleNamespace(
        st_dev=101, st_ino=23, st_size=19, st_mtime_ns=31, st_ctime_ns=37, st_mode=0o100600
    )
    other = copy.copy(one)
    other.st_dev = 202
    assert stat_cache_key(one) == stat_cache_key(other)
    other.st_ino = 24
    assert stat_cache_key(one) != stat_cache_key(other)


def test_retention_route_uses_original_f4_data_root(tmp_path):
    from sr_f6.source_binding import retention_image_path

    source = tmp_path / "sr_f4"
    actual = source / "DATA/images/legacy.png"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"frozen image")
    wrong = source / "images/legacy.png"
    frozen = dict(source_root=str(source), legacy_source=str(source / "DATA/TASKS.jsonl"))
    route = dict(qid="legacy-id", image_file="images/legacy.png")
    assert retention_image_path(frozen, route) == actual
    assert not wrong.exists()
    chartqa = dict(qid="chartqa-id", image_file="external/chartqa/image.png")
    assert retention_image_path(frozen, chartqa) == source / "DATA/external/chartqa/image.png"
    with pytest.raises(PermissionError):
        retention_image_path(frozen, dict(route, image_file="../images/legacy.png"))
    with pytest.raises(PermissionError):
        retention_image_path(dict(frozen, legacy_source=str(source / "TASKS.jsonl")), route)


def test_binding_cache_survives_later_failure_without_rehash(monkeypatch, tmp_path):
    from sr_f6 import source_binding as binding

    model = tmp_path / "model.safetensors"
    model.write_bytes(b"original model")
    cache_path = tmp_path / "SOURCE_BINDING_CACHE.json"
    context = {"config_canonical_sha256": "a" * 64}
    first = binding.InventoryCache(cache_path, boundary=tmp_path, context=context)
    original = first.inventory(model)
    assert first.counts["new_content_hashes"] == 1
    with pytest.raises(FileNotFoundError):
        first.inventory(tmp_path / "missing-retention-image.png")
    persisted = json.loads(cache_path.read_text())
    assert persisted["objects"][str(model)] == original
    binding._VERIFIED_STATS.clear()
    actual_hash = binding.file_sha256
    called = []

    def counted(path):
        called.append(str(path))
        return actual_hash(path)

    monkeypatch.setattr(binding, "file_sha256", counted)
    restarted = binding.InventoryCache(cache_path, boundary=tmp_path, context=context)
    assert restarted.inventory(model) == original
    assert called == []
    assert restarted.counts["unchanged_cache_hits"] == 1
    next_file = tmp_path / "image.png"
    next_file.write_bytes(b"image")
    restarted.inventory(next_file)
    assert called == [str(next_file)]
    assert set(json.loads(cache_path.read_text())["objects"]) == {str(model), str(next_file)}


def test_binding_cache_rejects_changed_context_and_changed_content(tmp_path):
    from sr_f6 import source_binding as binding

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"original")
    cache_path = tmp_path / "SOURCE_BINDING_CACHE.json"
    cache = binding.InventoryCache(cache_path, boundary=tmp_path, context={"config": "a"})
    cache.inventory(path)
    incompatible = binding.InventoryCache(cache_path, boundary=tmp_path, context={"config": "b"})
    with pytest.raises(PermissionError, match="different binding"):
        incompatible.inventory(path)
    path.write_bytes(b"modified")
    restarted = binding.InventoryCache(cache_path, boundary=tmp_path, context={"config": "a"})
    with pytest.raises(PermissionError, match="content changed"):
        restarted.inventory(path)
    assert json.loads(cache_path.read_text())["objects"][str(path)][
        "sha256"
    ] != binding.file_sha256(path)
