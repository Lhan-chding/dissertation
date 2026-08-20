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
    assert "--preflight-only" in source
    assert "arguments.preflight_only" in source
    assert "READY: Phase 7" in source
    assert "support_dev_image_bundle_sha256" in source


def test_phase7_evaluator_fails_closed_on_unparseable_answer_and_upstream_checkpoint_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evaluate = _load("test_phase7_evaluate_semantics", "16_evaluate_phase7_multimodal.py")

    assert evaluate._trace_mismatch(answer_value=None, chosen_execution=3) is True
    assert evaluate._trace_mismatch(answer_value=3, chosen_execution=None) is True
    assert evaluate._trace_mismatch(answer_value=3, chosen_execution=3) is False

    hashes = {
        "C0": "1" * 64,
        "C1": "2" * 64,
        "T": "3" * 64,
        "Base_AnswerOnly_RL": "4" * 64,
        "Recovery_LoRA_RecoveryOutcome_RL": "5" * 64,
        "Recovery_LoRA_AnswerOnly_RL": "6" * 64,
    }
    monkeypatch.setattr(
        evaluate,
        "tree_sha256",
        lambda path: hashes[
            {
                "C0_format_only": "C0",
                "C1_forward_arithmetic": "C1",
                "T_constraint_recovery": "T",
            }.get(path.parts[-2], path.parts[-2])
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "_json",
        lambda path, _label: {
            "status": "PHASE_6_VARIANT_TRAINED",
            "variant": path.parts[-2],
            "final_adapter_tree_sha256": hashes[path.parts[-2]],
        },
    )
    phase5 = {
        "source_sha256": {
            "Base": evaluate.MODEL_SNAPSHOT_SHA256,
            "C0": hashes["C0"],
            "C1": hashes["C1"],
            "T": "f" * 64,
        }
    }
    phase6 = {
        "checkpoint_sha256": {
            name: hashes[name]
            for name in (
                "Base_AnswerOnly_RL",
                "Recovery_LoRA_RecoveryOutcome_RL",
                "Recovery_LoRA_AnswerOnly_RL",
            )
        }
    }
    with pytest.raises(RuntimeError, match="T hash differs"):
        evaluate._checkpoint_hashes(
            tmp_path / "phase4",
            tmp_path / "phase6",
            phase5_summary=phase5,
            phase6_evaluation=phase6,
        )


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
        "scripts/v4/17_audit_phase7_interface.py",
    } <= set(subject.PHASE7_LOCKED_PATHS)


def test_phase7_interface_audit_entrypoint_is_hash_bound_analysis_only() -> None:
    source = (SCRIPT_DIR / "17_audit_phase7_interface.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "--execute" in source
    assert "--phase7-summary-sha256" in source
    assert "summarize_phase7_interface_evidence" in source
    assert "generate_completion" not in source
    assert "load_pinned_qwen" not in source
    assert "Trainer(" not in source
    assert any(value.startswith("BLOCKED: Phase 7") for value in strings)
