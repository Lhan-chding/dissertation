from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v4"


def _subject():
    return importlib.import_module("compensability_v4.qwen.phase7_runtime")


def _load(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase7_server_entrypoints_are_execute_gated_hash_bound_and_evaluation_only() -> None:
    prepare = _load("test_phase7_prepare", "15_prepare_phase7_multimodal.py")
    evaluate = _load("test_phase7_evaluate", "16_evaluate_phase7_multimodal.py")
    assert callable(prepare.main)
    assert callable(evaluate.main)

    for filename in ("15_prepare_phase7_multimodal.py", "16_evaluate_phase7_multimodal.py"):
        source = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "--execute" in source
        assert "--input-sha256" in source or "--execution-manifest-sha256" in source
        assert "PHASE7_LOCKED_PATHS" in source
        assert any(value.startswith("BLOCKED: Phase 7") for value in strings)
        assert "Trainer(" not in source
        assert "optimizer.step" not in source
        assert "torch.optim" not in source


def test_phase7_evaluator_has_full_multimodal_chain_and_confirmatory_fail_closed_guard() -> None:
    source = (SCRIPT_DIR / "16_evaluate_phase7_multimodal.py").read_text(encoding="utf-8")

    for symbol in (
        "generate_observation_with_cache",
        "revision_or_recovery",
        "chart_operation",
        "final_answer",
        "summarize_phase7",
    ):
        assert symbol in source
    assert "confirmatory_evaluation_authorized" in source
    assert "support_dev" in source
    assert "confirmatory" in source.lower()
    assert "refusing to overwrite" in source


def test_phase7_atomic_publication_refuses_overwrite_and_leaves_no_partial_outputs(
    tmp_path: Path,
) -> None:
    subject = _subject()
    row = subject.Phase7ChainRow.from_mapping(
        {
            "scene_id": "scene-a",
            "checkpoint": "Base",
            "checkpoint_sha256": "a" * 64,
            "family": "trend",
            "split": "support_dev",
            "ood_axis": "iid",
            "seed": 11,
            "rollout_id": 0,
            "image_sha256": "b" * 64,
            "stage1_visual_exact": True,
            "post_revision_world_exact": True,
            "reasoning_operator_exact": True,
            "final_answer_exact": True,
            "operator_invariant_correct": False,
            "genuine_recovery": False,
            "error_cancellation": False,
            "trace_mismatch": False,
            "error_mechanism_shift": False,
        }
    )
    summary = {
        "schema_version": 1,
        "status": "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED",
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": False,
    }
    output = tmp_path / "phase7"

    with pytest.raises(ValueError, match="summary"):
        subject.write_phase7_outputs(
            output_root=output,
            rows=(row,),
            summary={**summary, "status": "NOT_EVALUATED"},
            source_sha256={"execution_manifest": "c" * 64},
        )
    assert not output.exists()

    paths = subject.write_phase7_outputs(
        output_root=output,
        rows=(row,),
        summary=summary,
        source_sha256={"execution_manifest": "c" * 64},
    )
    assert set(paths) == {"per_scene", "summary"}
    assert output.exists()
    assert json.loads((output / "summary.json").read_text())["confirmatory_data_used"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.write_phase7_outputs(
            output_root=output,
            rows=(row,),
            summary=summary,
            source_sha256={"execution_manifest": "c" * 64},
        )


def test_phase7_package_lock_closes_config_runtime_and_both_server_entrypoints() -> None:
    subject = _subject()
    digest = subject.verify_phase7_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_7.yaml",
        repository_root=ROOT,
        expected_paths=subject.PHASE7_LOCKED_PATHS,
    )

    assert len(digest) == 64
    assert {
        "configs/recoverability/v4_phase_7.yaml",
        "src/compensability_v4/qwen/phase7_runtime.py",
        "scripts/v4/15_prepare_phase7_multimodal.py",
        "scripts/v4/16_evaluate_phase7_multimodal.py",
    } <= set(subject.PHASE7_LOCKED_PATHS)
