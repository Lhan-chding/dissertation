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
    return importlib.import_module("compensability_v4.qwen.phase8_confirm_runtime")


def _execution_subject():
    return importlib.import_module("compensability_v4.qwen.phase8_execution")


def _load(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase8_data_freeze_and_evaluation_clis_are_execute_gated_hash_bound_and_offline() -> None:
    freeze = _load("test_phase8_freeze", "18_freeze_phase8_confirm_data.py")
    evaluate = _load("test_phase8_evaluate", "19_evaluate_phase8_confirmatory.py")
    assert callable(freeze.main)
    assert callable(evaluate.main)

    for filename in ("18_freeze_phase8_confirm_data.py", "19_evaluate_phase8_confirmatory.py"):
        source = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "--execute" in source
        assert "--input-sha256" in source or "--execution-manifest-sha256" in source
        assert "PHASE8_LOCKED_PATHS" in source
        assert "HF_HUB_OFFLINE" in source
        assert "TRANSFORMERS_OFFLINE" in source
        assert any(value.startswith("BLOCKED: Phase 8") for value in strings)
        assert "Trainer(" not in source
        assert "optimizer.step" not in source
        assert "torch.optim" not in source


def test_phase8_clis_require_confirm_authorization_and_explicit_ack_fail_closed() -> None:
    freeze_source = (SCRIPT_DIR / "18_freeze_phase8_confirm_data.py").read_text(encoding="utf-8")
    evaluate_source = (SCRIPT_DIR / "19_evaluate_phase8_confirmatory.py").read_text(
        encoding="utf-8"
    )
    for source in (freeze_source, evaluate_source):
        assert "confirmatory_evaluation_authorized" in source
        assert "COMPBIAS_V4_PHASE8_CONFIRM_ACK" in source
        assert "I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET" in source
        assert "refusing to overwrite" in source


def test_phase8_data_freeze_builds_all_four_axes_and_preserves_all_natural_errors() -> None:
    source = (SCRIPT_DIR / "18_freeze_phase8_confirm_data.py").read_text(encoding="utf-8")
    for symbol in (
        "confirm_iid",
        "confirm_style_ood",
        "confirm_constraint_ood",
        "confirm_error_mechanism_ood",
        "validate_phase8_isolation",
        "freeze_phase8_natural_errors",
        "all_natural_stage1_errors_included",
    ):
        assert symbol in source
    assert "selection_uses_model_outcome_threshold" in source


def test_phase8_evaluator_runs_full_chain_and_preserves_both_answer_endpoints() -> None:
    source = (SCRIPT_DIR / "19_evaluate_phase8_confirmatory.py").read_text(encoding="utf-8")
    for symbol in (
        "generate_observation_with_cache",
        "revision_or_recovery",
        "chart_operation",
        "final_answer",
        "free_generation_answer_exact",
        "deterministic_chain_answer_exact",
        "answer_source",
        "summarize_phase8",
    ):
        assert symbol in source


def test_phase8_confirm_template_selection_fails_closed_when_constraints_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = _execution_subject()

    def reject_all(_family: str, _truth: tuple[int, int, int, int]):
        raise ValueError("unavailable")

    monkeypatch.setattr(subject, "build_family_constraints", reject_all)
    with pytest.raises(RuntimeError, match="candidate search exhausted"):
        subject.select_confirm_templates(count=1, seed=2026082103, reserved_truths=set())


def test_phase8_resumed_cache_requires_exact_checkpoint_scene_closure(tmp_path: Path) -> None:
    subject = _execution_subject()
    bindings = {
        "checkpoint_sha256": "a" * 64,
        "execution_manifest_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "package_lock_sha256": "d" * 64,
    }
    rows = (
        {
            "chain_row": {
                "scene_id": "scene-a",
                "checkpoint": "T",
                "checkpoint_sha256": "a" * 64,
            }
        },
        {
            "chain_row": {
                "scene_id": "scene-b",
                "checkpoint": "T",
                "checkpoint_sha256": "a" * 64,
            }
        },
    )
    subject.cache_checkpoint_rows(
        root=tmp_path,
        checkpoint="T",
        rows=rows,
        expected_scene_ids=frozenset({"scene-a", "scene-b"}),
        **bindings,
    )
    assert (
        subject.cache_checkpoint_rows(
            root=tmp_path,
            checkpoint="T",
            rows=None,
            expected_scene_ids=frozenset({"scene-a", "scene-b"}),
            **bindings,
        )
        == rows
    )
    with pytest.raises(RuntimeError, match="checkpoint hash"):
        subject.cache_checkpoint_rows(
            root=tmp_path / "bad-hash",
            checkpoint="T",
            rows=({"chain_row": {**rows[0]["chain_row"], "checkpoint_sha256": "e" * 64}},),
            expected_scene_ids=frozenset({"scene-a"}),
            **bindings,
        )
    with pytest.raises(RuntimeError, match="scene closure"):
        subject.cache_checkpoint_rows(
            root=tmp_path,
            checkpoint="T",
            rows=None,
            expected_scene_ids=frozenset({"scene-a", "scene-b", "scene-c"}),
            **bindings,
        )


def test_phase8_rejects_nonformal_phase7_upstream_evidence() -> None:
    subject = _execution_subject()
    checkpoint_sha256 = {name: "a" * 64 for name in subject.SEVEN_CHECKPOINTS}
    valid = {
        "schema_version": 1,
        "status": "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED",
        "confirmatory_data_used": False,
        "support_dev_diagnostic": True,
        "training_invoked": False,
        "checkpoint_sha256": checkpoint_sha256,
    }
    assert subject.validate_phase7_evaluation(valid) == checkpoint_sha256
    with pytest.raises(RuntimeError, match="formal diagnostic artifact"):
        subject.validate_phase7_evaluation({**valid, "status": "NOT_PHASE_7"})


def test_phase8_atomic_publication_refuses_overwrite_and_no_partial_output(tmp_path: Path) -> None:
    subject = _subject()
    row = subject.Phase8ConfirmRow.from_mapping(
        {
            "scene_id": "scene-a",
            "checkpoint": "T",
            "checkpoint_sha256": "a" * 64,
            "family": "trend",
            "split": "confirm_iid",
            "ood_axis": "iid",
            "seed": 31,
            "rollout_id": 0,
            "image_sha256": "b" * 64,
            "stage1_visual_exact": False,
            "post_revision_world_exact": True,
            "reasoning_operator_exact": True,
            "final_answer_exact": True,
            "operator_invariant_correct": False,
            "genuine_recovery": True,
            "error_cancellation": False,
            "trace_mismatch": False,
            "error_mechanism_shift": False,
            "free_generation_answer_exact": True,
            "deterministic_chain_answer_exact": True,
            "answer_source": "genuine_recovery",
        }
    )
    summary = {
        "schema_version": 1,
        "status": "PHASE_8_CONFIRMATORY_EVALUATED",
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": True,
    }
    output = tmp_path / "phase8"

    with pytest.raises(ValueError, match="summary"):
        subject.write_phase8_outputs(
            output_root=output,
            rows=(row,),
            summary={**summary, "status": "NOT_EVALUATED"},
            source_sha256={"execution_manifest": "c" * 64},
        )
    assert not output.exists()

    paths = subject.write_phase8_outputs(
        output_root=output,
        rows=(row,),
        summary=summary,
        source_sha256={"execution_manifest": "c" * 64},
    )
    assert set(paths) == {"per_scene", "summary"}
    assert json.loads((output / "summary.json").read_text())["confirmatory_data_used"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.write_phase8_outputs(
            output_root=output,
            rows=(row,),
            summary=summary,
            source_sha256={"execution_manifest": "c" * 64},
        )


def test_phase8_package_lock_closes_config_runtime_and_both_server_entrypoints() -> None:
    subject = _subject()
    digest = subject.verify_phase8_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_8.yaml",
        repository_root=ROOT,
        expected_paths=subject.PHASE8_LOCKED_PATHS,
    )
    assert len(digest) == 64
    assert {
        "configs/recoverability/v4_phase_8.yaml",
        "configs/recoverability/v4/phase_1_3_prompts.yaml",
        "src/compensability_v4/qwen/phase8_confirm_runtime.py",
        "scripts/v4/18_freeze_phase8_confirm_data.py",
        "scripts/v4/19_evaluate_phase8_confirmatory.py",
    } <= set(subject.PHASE8_LOCKED_PATHS)
