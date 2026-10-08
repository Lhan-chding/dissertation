"""Verify inverse patch geometry with an independent forward layout fixture."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mm_core/processor_preflight.py"
SPEC = importlib.util.spec_from_file_location("processor_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def processor_fixture(patch_size=14):
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(patch_size * 2, patch_size * 4, 3), dtype=np.uint8)
    processor = SimpleNamespace(
        patch_size=patch_size,
        merge_size=2,
        temporal_patch_size=2,
        do_normalize=True,
        do_rescale=True,
        image_mean=[0.48, 0.45, 0.41],
        image_std=[0.26, 0.27, 0.28],
        rescale_factor=1 / 255,
    )
    normalized = (rgb.astype(np.float32) / 255 - processor.image_mean) / processor.image_std
    chw = normalized.transpose(2, 0, 1)
    patches = chw.reshape(3, 1, 2, patch_size, 2, 2, patch_size).transpose(1, 4, 2, 5, 0, 3, 6)
    temporal = np.repeat(patches[:, :, :, :, :, None], 2, axis=5)
    return rgb, temporal.reshape(8, 3 * 2 * patch_size**2), processor


@pytest.mark.parametrize("patch_size", [14, 16])
def test_inverse_qwen_layout_restores_every_rgb_pixel(patch_size):
    rgb, pixels, processor = processor_fixture(patch_size)
    restored = MODULE.reconstruct_rgb(pixels, [[1, 2, 4]], processor)
    np.testing.assert_array_equal(np.asarray(restored), rgb)


@pytest.mark.parametrize("patch_size", [14, 16])
def test_unequal_temporal_planes_fail(patch_size):
    _, pixels, processor = processor_fixture(patch_size)
    pixels[0, patch_size**2] += 0.1
    with pytest.raises(ValueError, match="temporal planes differ"):
        MODULE.reconstruct_rgb(pixels, [[1, 2, 4]], processor)


def test_nonstill_grid_fails():
    _, pixels, processor = processor_fixture()
    with pytest.raises(ValueError, match="one still-image"):
        MODULE.reconstruct_rgb(pixels, [[2, 2, 4]], processor)


@pytest.mark.parametrize("patch_size,expected_grid", [(14, [1, 48, 64]), (16, [1, 42, 56])])
def test_installed_qwen_processor_actual_cpu_roundtrip(patch_size, expected_grid):
    import torch
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    torch.set_num_threads(2)
    processor = Qwen2VLImageProcessor(
        patch_size=patch_size,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=896 * 672,
        max_pixels=896 * 672,
    )
    rgb = np.random.default_rng(123).integers(0, 256, size=(672, 896, 3), dtype=np.uint8)
    encoded = processor(images=[Image.fromarray(rgb)], return_tensors="pt")
    assert encoded["image_grid_thw"].tolist() == [expected_grid]
    reconstructed = MODULE.reconstruct_rgb(
        encoded["pixel_values"], encoded["image_grid_thw"].tolist(), processor
    )
    np.testing.assert_array_equal(np.asarray(reconstructed), rgb)
    assert encoded["pixel_values"].device.type == "cpu"


def test_wrong_model_rejected_before_processor_loading(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl", "hidden_size": 2048, "num_hidden_layers": 36})
    )
    with pytest.raises(ValueError, match=r"Qwen3\.5"):
        MODULE._verified_runtime(tmp_path)


@pytest.mark.parametrize("thinking", [True, None])
def test_unfrozen_thinking_template_is_rejected(tmp_path, monkeypatch, thinking):
    identity = {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_type": "qwen3_5",
        "architecture_verified": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "chat_template_kwargs_hash": MODULE.hash_json({"enable_thinking": thinking}),
    }
    monkeypatch.setattr(
        MODULE.QwenRuntime,
        "processor_only",
        lambda _: SimpleNamespace(identity=identity),
    )
    with pytest.raises(ValueError, match="enable_thinking=False"):
        MODULE._verified_runtime(tmp_path)


@pytest.mark.parametrize("patch_size", [14, 16])
def test_qwen35_preflight_requires_native_processor_geometry(tmp_path, monkeypatch, patch_size):
    identity = {
        "model_id": "Qwen/Qwen3.5-9B",
        "model_type": "qwen3_5",
        "architecture_verified": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "chat_template_kwargs_hash": MODULE.hash_json({"enable_thinking": False}),
    }
    runtime = SimpleNamespace(
        identity=identity,
        processor=SimpleNamespace(
            image_processor=SimpleNamespace(
                patch_size=patch_size, merge_size=2, temporal_patch_size=2
            )
        ),
    )
    monkeypatch.setattr(MODULE.QwenRuntime, "processor_only", lambda _: runtime)
    if patch_size == 14:
        with pytest.raises(ValueError, match="native vision architecture"):
            MODULE._verified_runtime(tmp_path)
    else:
        assert MODULE._verified_runtime(tmp_path) is runtime


def _write_generation_model_config(path, eos_ids):
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "eos_token_id": eos_ids,
                    "pad_token_id": 248040,
                    "bos_token_id": None,
                },
                "vision_config": {},
            }
        )
    )


@pytest.mark.parametrize("nested_eos", [248044, [248044, 248046]])
def test_missing_generation_json_uses_native_nested_text_config(tmp_path, nested_eos):
    _write_generation_model_config(tmp_path, nested_eos)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=248041, bos_token_id=None))
    result = MODULE._expanded_generation(tmp_path, processor)
    expected_eos = nested_eos if isinstance(nested_eos, list) else [nested_eos]
    assert result["generation_config_expanded"]["eos_token_id"] == expected_eos
    assert result["generation_config_expanded"]["pad_token_id"] == 248041
    assert result["generation_config_expanded"]["bos_token_id"] is None
    assert result["generation_config_expanded"]["max_new_tokens"] == 192
    assert result["original_generation_config_source"] == (
        "config.json:GenerationConfig.from_model_config"
    )
    assert result["original_generation_config_sha256"] is None
    assert result["model_config_sha256"] == MODULE.file_hash(tmp_path / "config.json")
    assert not (tmp_path / "generation_config.json").exists()


def test_explicit_generation_json_overrides_model_eos(tmp_path):
    _write_generation_model_config(tmp_path, 248044)
    generation_path = tmp_path / "generation_config.json"
    generation_path.write_text(json.dumps({"eos_token_id": [248044, 248046], "temperature": 0.2}))
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=248041, bos_token_id=248042))
    result = MODULE._expanded_generation(tmp_path, processor)
    assert result["generation_config_expanded"]["eos_token_id"] == [248044, 248046]
    assert result["generation_config_expanded"]["temperature"] == MODULE.GENERATION["temperature"]
    assert result["generation_config_expanded"]["bos_token_id"] == 248042
    assert result["original_generation_config_source"] == "generation_config.json"
    assert result["original_generation_config_sha256"] == MODULE.file_hash(generation_path)
