"""Manifest hashes and generation are canonical, deterministic, and seed-local."""

import random
import re

import numpy as np
import pytest

from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
from compbias.envs.cva_world.renderer import SUPPORTED_VISUAL_STYLES
from compbias.envs.cva_world.schema import SemanticSplit, TaskFamily
from compbias.io.manifests import build_dataset_manifest, manifest_sha256


def _payload(order: str = "normal") -> dict[str, object]:
    if order == "normal":
        return {"sample_id": "a", "scene": {"value": 7, "style": "font_a"}}
    return {"scene": {"style": "font_a", "value": 7}, "sample_id": "a"}


def _config(seed: int) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        samples_per_family_per_split=1,
        splits=tuple(SemanticSplit),
        task_families=tuple(TaskFamily),
        visual_styles=SUPPORTED_VISUAL_STYLES,
        train_error_mechanism="offset_plus_2",
        ood_error_mechanism="offset_minus_2",
        fully_cross_iid_visual_styles=True,
    )


def test_manifest_hash_uses_canonical_mapping_order_and_detects_content_change() -> None:
    first = manifest_sha256(_payload("normal"))
    reordered = manifest_sha256(_payload("reordered"))
    changed = manifest_sha256({"sample_id": "a", "scene": {"value": 8, "style": "font_a"}})

    assert first == reordered
    assert first != changed
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_dataset_manifest_and_samples_repeat_exactly_for_same_seed() -> None:
    config = _config(31)
    first_samples = generate_dataset(config)
    second_samples = generate_dataset(config)

    first = build_dataset_manifest(
        first_samples,
        config=config,
        dataset_name="cva_world_v1",
        schema_version="1",
    )
    second = build_dataset_manifest(
        second_samples,
        config=config,
        dataset_name="cva_world_v1",
        schema_version="1",
    )

    assert first_samples == second_samples
    assert first == second
    assert first.sample_count == len(first_samples)
    assert first.sample_ids == tuple(sorted(sample.sample_id for sample in first_samples))
    assert re.fullmatch(r"[0-9a-f]{64}", first.content_sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", first.config_sha256)
    assert first.to_mapping() == second.to_mapping()


def test_seed_change_updates_config_and_content_hashes() -> None:
    first_config = _config(31)
    second_config = _config(32)
    first = build_dataset_manifest(
        generate_dataset(first_config),
        config=first_config,
        dataset_name="cva_world_v1",
        schema_version="1",
    )
    second = build_dataset_manifest(
        generate_dataset(second_config),
        config=second_config,
        dataset_name="cva_world_v1",
        schema_version="1",
    )

    assert first.config_sha256 != second.config_sha256
    assert first.content_sha256 != second.content_sha256


@pytest.mark.parametrize(
    "sample_id",
    ("../probe", "nested/probe", "nested\\probe", ".", "bad\x00name", "white space"),
)
def test_dataset_manifest_rejects_sample_ids_that_are_not_safe_basenames(
    sample_id: str,
) -> None:
    with pytest.raises(ValueError, match="safe basename"):
        build_dataset_manifest(
            ({"sample_id": sample_id, "scene": {"value": 7}},),
            config={"seed": 1},
            dataset_name="fixture",
            schema_version="1",
        )


def test_generation_does_not_consume_process_global_rng_state() -> None:
    random.seed(123)
    np.random.seed(123)
    expected_python = random.random()
    expected_numpy = float(np.random.random())

    random.seed(123)
    np.random.seed(123)
    generate_dataset(_config(31))
    observed_python = random.random()
    observed_numpy = float(np.random.random())

    assert observed_python == expected_python
    assert observed_numpy == expected_numpy
