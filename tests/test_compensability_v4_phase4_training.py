"""Acceptance tests for the fail-closed Phase 4 LoRA support-injection surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compensability_v4.data.natural_error_pool import NaturalErrorExample
from compensability_v4.data.splits import DatasetSplit
from compensability_v4.schemas.scene import RecoveryScene
from compensability_v4.training.phase4 import (
    Phase4TrainingConfig,
    SupportVariant,
    _parameter_bytes,
    build_support_sets,
    discover_language_lora_targets,
    freeze_base_parameters,
    load_phase4_config,
    validate_phase4_preflight,
    verify_phase4_package_lock,
    write_support_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def _scene(*, scene_id: str, split: DatasetSplit) -> RecoveryScene:
    return RecoveryScene(
        scene_id=scene_id,
        split=split,
        semantic_scene_id=f"semantic-{scene_id}",
        numeric_table_id=f"numbers-{scene_id}",
        constraint_graph_id=f"graph-{scene_id}",
        truth=(9, 4, 5, 6),
        facts=(
            {"type": "known_value", "index": 1, "value": 4},
            {"type": "known_value", "index": 2, "value": 5},
            {"type": "known_value", "index": 3, "value": 6},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 13},
        ),
        resized_height=280,
        resized_width=280,
        image_path=f"images/{scene_id}.png",
    )


def _natural(scene: RecoveryScene) -> NaturalErrorExample:
    return NaturalErrorExample(
        scene_id=scene.scene_id,
        observation_id=f"obs-{scene.scene_id}",
        truth=scene.truth,
        observed_values=(8, 4, 5, 6),
        error_index=0,
        stage1_model_hash="a" * 64,
    )


def test_support_builder_separates_c0_c1_and_free_recovery_targets() -> None:
    symbolic = _scene(scene_id="symbolic", split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN)
    natural = _scene(scene_id="natural", split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN)

    support = build_support_sets(
        symbolic_scenes=(symbolic,),
        natural_scenes=(natural,),
        natural_errors=(_natural(natural),),
    )

    assert set(support) == set(SupportVariant)
    assert all(
        row.completion == "9,4,5,6"
        for row in support[SupportVariant.RECOVERY]
        if row.curriculum_stage == "final_free_recovery"
    )
    assert all("Observed values" not in row.prompt for row in support[SupportVariant.FORMAT_ONLY])
    assert all("Correct values" in row.prompt for row in support[SupportVariant.FORWARD_ARITHMETIC])
    assert all("Observed values" in row.prompt for row in support[SupportVariant.RECOVERY])
    recovery_rows = support[SupportVariant.RECOVERY]
    assert any(row.curriculum_stage == "final_free_recovery" for row in recovery_rows)


def test_support_builder_rejects_confirm_or_unpaired_natural_inputs() -> None:
    symbolic = _scene(scene_id="symbolic", split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN)
    confirm = _scene(scene_id="confirm", split=DatasetSplit.CONFIRM_IID)

    with pytest.raises(ValueError, match="natural_error_support_train"):
        build_support_sets(
            symbolic_scenes=(symbolic,),
            natural_scenes=(confirm,),
            natural_errors=(_natural(confirm),),
        )


def test_dedicated_phase_four_config_authorizes_only_language_lora_training() -> None:
    config = load_phase4_config(ROOT / "configs/recoverability/v4_phase_4.yaml")

    assert config.precision == "bf16"
    assert config.lora_rank == 16
    assert config.vision_frozen is True
    assert config.merger_frozen is True
    assert config.selection_split == "support_dev"


def test_phase_four_package_lock_covers_the_complete_executable_surface() -> None:
    expected = (
        "configs/recoverability/v4_phase_4.yaml",
        "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
        "docs/QWEN_V4_SERVER_HANDOFF.md",
        "pyproject.toml",
        "requirements-gpu.lock.txt",
        "scripts/v4/07_prepare_phase4_support_sources.py",
        "scripts/v4/07_build_support_data.py",
        "scripts/v4/08_train_phase4_lora.py",
        "src/compensability_v4/training/__init__.py",
        "src/compensability_v4/training/phase4.py",
        "src/compensability_v4/training/phase4_sources.py",
    )

    assert verify_phase4_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_4.yaml",
        repository_root=ROOT,
        expected_paths=expected,
    )


def test_phase_four_scripts_are_inert_without_execute() -> None:
    common = [
        sys.executable,
        "scripts/v4/07_build_support_data.py",
        "--symbolic-scenes",
        "missing.jsonl",
        "--symbolic-scenes-sha256",
        "a" * 64,
        "--natural-scenes",
        "missing.jsonl",
        "--natural-scenes-sha256",
        "b" * 64,
        "--natural-observations",
        "missing.jsonl",
        "--natural-observations-sha256",
        "c" * 64,
    ]
    build = subprocess.run(
        common,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    train = subprocess.run(
        [
            sys.executable,
            "scripts/v4/08_train_phase4_lora.py",
            "--support",
            "missing.jsonl",
            "--support-sha256",
            "d" * 64,
            "--support-summary",
            "missing-summary.json",
            "--output-root",
            "missing-output",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    prepare = subprocess.run(
        [sys.executable, "scripts/v4/07_prepare_phase4_support_sources.py"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    prepared_build = subprocess.run(
        [sys.executable, "scripts/v4/07_build_support_data.py", "--prepared-sources"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    prepared_train = subprocess.run(
        [sys.executable, "scripts/v4/08_train_phase4_lora.py", "--prepared-support"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert build.returncode == train.returncode == prepare.returncode == 2
    assert prepared_build.returncode == prepared_train.returncode == 2
    assert "BLOCKED" in build.stdout
    assert "BLOCKED" in train.stdout
    assert "BLOCKED" in prepare.stdout
    assert "BLOCKED" in prepared_build.stdout
    assert "BLOCKED" in prepared_train.stdout


class _Leaf:
    pass


class _FakeModel:
    def named_modules(self) -> list[tuple[str, object]]:
        return [
            ("", self),
            ("model.visual.blocks.0.attn.q_proj", _Leaf()),
            ("model.visual.merger", _Leaf()),
            ("model.language_model.layers.0.self_attn.q_proj", _Leaf()),
            ("model.language_model.layers.0.self_attn.k_proj", _Leaf()),
            ("model.language_model.layers.0.mlp.gate_proj", _Leaf()),
            ("lm_head", _Leaf()),
        ]


def test_lora_targets_are_discovered_from_actual_language_modules_only() -> None:
    targets = discover_language_lora_targets(_FakeModel())

    assert targets == (
        "model.language_model.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.q_proj",
    )


def test_frozen_parameter_hash_serializes_bfloat16_without_loss() -> None:
    torch = pytest.importorskip("torch")
    parameters = (
        torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16),
        torch.tensor(1.0, dtype=torch.bfloat16),
    )

    for parameter in parameters:
        expected = parameter.reshape(-1).view(torch.uint8).numpy().tobytes()
        assert _parameter_bytes(parameter) == expected


def test_freeze_base_parameters_separates_visual_merger_from_vision() -> None:
    torch = pytest.importorskip("torch")

    class _QwenNamedParameters:
        def named_parameters(self):
            return iter(
                (
                    (
                        "model.visual.blocks.0.attn.qkv.weight",
                        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)),
                    ),
                    (
                        "model.visual.merger.mlp.0.weight",
                        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)),
                    ),
                    (
                        "model.language_model.layers.0.self_attn.q_proj.weight",
                        torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)),
                    ),
                )
            )

    frozen = freeze_base_parameters(_QwenNamedParameters())

    assert frozen["parameter_counts"] == {
        "language_base": 1,
        "merger": 1,
        "vision": 1,
    }


def test_preflight_requires_cuda_bf16_and_all_gpu_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    config = Phase4TrainingConfig.default()
    monkeypatch.setattr("compensability_v4.training.phase4._import_torch", lambda: _Torch())
    monkeypatch.setattr(
        "compensability_v4.training.phase4._missing_gpu_dependencies", lambda: ("peft",)
    )

    with pytest.raises(RuntimeError, match="missing required GPU packages"):
        validate_phase4_preflight(config=config, output_root=Path("/tmp/phase4"))


def test_support_artifact_is_non_overwriting_and_carries_provenance(tmp_path: Path) -> None:
    symbolic = _scene(scene_id="symbolic", split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN)
    natural = _scene(scene_id="natural", split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN)
    output = tmp_path / "support.jsonl"
    summary = tmp_path / "summary.json"

    write_support_artifact(
        output_path=output,
        summary_path=summary,
        support_sets=build_support_sets(
            symbolic_scenes=(symbolic,),
            natural_scenes=(natural,),
            natural_errors=(_natural(natural),),
        ),
        source_hashes={
            "symbolic_scenes": "b" * 64,
            "natural_scenes": "d" * 64,
            "natural_observations": "c" * 64,
        },
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(rows) == payload["example_count"]
    assert payload["source_hashes"]["natural_observations"] == "c" * 64
    with pytest.raises(FileExistsError, match="overwrite"):
        write_support_artifact(
            output_path=output,
            summary_path=summary,
            support_sets={},
            source_hashes={},
        )
