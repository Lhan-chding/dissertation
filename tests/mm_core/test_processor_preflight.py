"""Verify inverse patch geometry with an independent forward layout fixture."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mm_core/processor_preflight.py"
SPEC = importlib.util.spec_from_file_location("processor_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def processor_fixture():
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(28, 56, 3), dtype=np.uint8)
    processor = SimpleNamespace(
        patch_size=14,
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
    patches = chw.reshape(3, 1, 2, 14, 2, 2, 14).transpose(1, 4, 2, 5, 0, 3, 6)
    temporal = np.repeat(patches[:, :, :, :, :, None], 2, axis=5)
    return rgb, temporal.reshape(8, 1176), processor


def test_inverse_qwen_layout_restores_every_rgb_pixel():
    rgb, pixels, processor = processor_fixture()
    restored = MODULE.reconstruct_rgb(pixels, [[1, 2, 4]], processor)
    np.testing.assert_array_equal(np.asarray(restored), rgb)


def test_unequal_temporal_planes_fail():
    _, pixels, processor = processor_fixture()
    pixels[0, 196] += 0.1
    with pytest.raises(ValueError, match="temporal planes differ"):
        MODULE.reconstruct_rgb(pixels, [[1, 2, 4]], processor)


def test_nonstill_grid_fails():
    _, pixels, processor = processor_fixture()
    with pytest.raises(ValueError, match="one still-image"):
        MODULE.reconstruct_rgb(pixels, [[2, 2, 4]], processor)
