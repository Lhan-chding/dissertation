"""Executable single-4090 Study-B runtime contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compensability_v5.qwen.study_b_runtime import (
    PILOT_SEED,
    StudyBError,
    canonical_sha256,
    evaluation_rows_from_study_a,
    parse_world,
    require_offline_environment,
    run_study_b,
    summarize_evaluations,
    tree_sha256,
    unified_world_prompt,
    validate_evaluation_rows,
    validate_support_package,
)

MODEL_SHA = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
RAW_SHA = "f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84"
ROOT = Path(__file__).resolve().parents[2]


def _budget() -> dict[str, object]:
    return {
        "unique_source_scenes": 96,
        "rows": 576,
        "target_tokens": 2304,
        "steps": 72,
        "optimizer": {"name": "adamw", "learning_rate": 2e-5, "weight_decay": 0.0},
        "lora_rank": 16,
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "gradient_accumulation": 8,
        "approximate_flops": 1000.0,
    }


def _support_package() -> dict[str, object]:
    arms: dict[str, object] = {}
    budgets: dict[str, object] = {}
    for arm in ("B0", "B1", "B2", "B3"):
        arms[arm] = [
            {
                "schema_version": 1,
                "arm": arm,
                "variant_index": variant_index,
                "scene_id": f"train-{scene_index}",
                "semantic_scene_id": f"train-parent-{scene_index}",
                "task_name": f"{arm}-task-{variant_index}",
                "prompt": f"{arm} semantic training prompt {variant_index}",
                "completion": "9,2,3,4",
                "target_tokens": 4,
            }
            for scene_index in range(96)
            for variant_index in range(1, 7)
        ]
        budgets[arm] = _budget()
    return {
        "schema_version": 1,
        "status": "V5_BUDGET_MATCHED_SUPPORT_FROZEN",
        "source_scene_count": 96,
        "arms": arms,
        "budgets": budgets,
        "target_token_relative_tolerance": 0.01,
        "pilot_schedule": {
            "hardware": "single_RTX_4090",
            "batch_size": 1,
            "gradient_accumulation": 8,
            "epochs": 1,
            "optimizer_steps": 72,
        },
        "source_provenance": {
            "parent_manifest_sha256": "1" * 64,
            "child_manifest_sha256": "2" * 64,
            "frozen_scenes_sha256": "3" * 64,
        },
    }


def _load_cli():
    path = ROOT / "scripts/v5/13_run_study_b.py"
    specification = importlib.util.spec_from_file_location("test_v5_study_b_cli", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _evaluation_rows() -> list[dict[str, object]]:
    axes = (
        ("iid", "familiar", 0),
        ("variable_permutation", "variable_permuted", 0),
        ("error_position", "familiar", 1),
        ("fact_order", "fact_order_permuted", 0),
        ("constraint_graph", "sparse_mixed_ood", 0),
    )
    rows = []
    for parent_index in range(32):
        for index, (axis, graph_axis, error_index) in enumerate(axes):
            truth = [9, 2, 3, 4]
            observed = list(truth)
            observed[error_index] -= 1
            rows.append(
                {
                    "scene_id": f"eval-{parent_index}-{index}",
                    "semantic_scene_id": f"eval-parent-{parent_index}",
                    "truth": truth,
                    "natural_observation": observed,
                    "family": "pair_sum",
                    "split": "independent_v4_support_dev",
                    "source_sha256": {"raw_archive": RAW_SHA},
                    "graph_axis": graph_axis,
                    "evaluation_axes": [axis],
                    "constraint_matrix": [
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 1],
                        [1, 0, 1, 0],
                    ],
                    "constraint_targets": [11, 5, 7, 12],
                }
            )
    return rows


class _FakeBackend:
    def __init__(self, *, fail_once_on: str | None = None) -> None:
        self.loaded: list[str] = []
        self.released: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self.fail_once_on = fail_once_on

    def load_base(self, *, arm: str, expected_model_sha256: str):
        assert expected_model_sha256 == MODEL_SHA
        self.loaded.append(arm)
        return {
            "model": {"arm": arm},
            "processor": object(),
            "model_sha256": MODEL_SHA,
            "load_token": f"fresh-base-{arm}-{len(self.loaded)}",
        }

    def train(
        self,
        *,
        session: dict[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        budget: dict[str, object],
        seed: int,
        output: Path,
    ) -> dict[str, object]:
        assert session["model"] == {"arm": arm}
        assert seed == PILOT_SEED
        assert len(rows) == budget["rows"] == 576
        if self.fail_once_on == arm:
            self.fail_once_on = None
            raise RuntimeError("simulated interruption")
        adapter = output / "final_adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(arm.encode("ascii"))
        (output / "training_log.json").write_text(
            json.dumps([{"step": 1, "loss": 1.0}]), encoding="utf-8"
        )
        return {
            "adapter_path": str(adapter),
            "training_metrics": {
                "train_steps": 72,
                "train_loss": 1.0,
                "observed_total_flos": 1000.0,
            },
            "observed_target_tokens": 2304,
            "trainable_manifest": {
                "vision_frozen": True,
                "merger_frozen": True,
                "base_language_frozen": True,
                "target_modules": ["model.language_model.layers.0.self_attn.q_proj"],
                "trainable_parameter_names": [
                    "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
                ],
            },
            "frozen_hashes": {
                "sha256_by_component": {
                    "vision": "a" * 64,
                    "merger": "b" * 64,
                    "language_base": "c" * 64,
                }
            },
        }

    def evaluate(
        self,
        *,
        session: dict[str, object],
        arm: str,
        rows: tuple[dict[str, object], ...],
        prompts: tuple[str, ...],
        seed: int,
        output: Path,
    ) -> tuple[dict[str, object], ...]:
        self.prompts[arm] = list(prompts)
        output_rows = []
        for row in rows:
            axis = row["evaluation_axes"][0]
            if arm == "B3" or (arm == "B2" and axis == "iid"):
                completion = "9,2,3,4"
            else:
                completion = ",".join(map(str, row["natural_observation"]))
            output_rows.append({"scene_id": row["scene_id"], "completion": completion})
        return tuple(output_rows)

    def release(self, session: dict[str, object]) -> None:
        self.released.append(session["model"]["arm"])


def test_support_validation_binds_actual_rows_sources_tokens_and_budget() -> None:
    package = _support_package()
    validate_support_package(package)

    package["arms"]["B3"][0]["scene_id"] = "different-source"
    with pytest.raises(StudyBError, match="same source scenes"):
        validate_support_package(package)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda package: package.update(status="not-frozen"),
        lambda package: package["arms"].pop("B3"),
        lambda package: package.update(source_scene_count=95),
        lambda package: package.update(target_token_relative_tolerance=0.02),
        lambda package: package["pilot_schedule"].update(optimizer_steps=71),
        lambda package: package["pilot_schedule"].update(batch_size=2),
        lambda package: package["arms"]["B0"][0].pop("completion"),
        lambda package: package["arms"]["B0"][0].update(variant_index=0),
        lambda package: package["arms"]["B0"].pop(),
        lambda package: [budget.update(steps=71) for budget in package["budgets"].values()],
        lambda package: [
            budget.update(gradient_accumulation=4) for budget in package["budgets"].values()
        ],
        lambda package: [budget.update(lora_rank=8) for budget in package["budgets"].values()],
        lambda package: [
            budget.update(lora_targets=["q_proj"]) for budget in package["budgets"].values()
        ],
        lambda package: [
            budget.update(optimizer={"name": "adamw", "learning_rate": 1e-5, "weight_decay": 0.0})
            for budget in package["budgets"].values()
        ],
    ],
)
def test_support_validation_fails_closed_on_schema_or_canonical_budget_drift(
    mutation: object,
) -> None:
    package = copy.deepcopy(_support_package())
    mutation(package)

    with pytest.raises(StudyBError):
        validate_support_package(package)


def test_unified_prompt_has_only_text_world_input_and_constraints() -> None:
    row = _evaluation_rows()[0]
    prompt = unified_world_prompt(row)

    assert "Observed values: 8,2,3,4" in prompt
    assert "Constraint rows (A | b):" in prompt
    assert "Return exactly four comma-separated integers only" in prompt
    assert "image" not in prompt.casefold()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.clear(),
        lambda rows: rows[0].update(scene_id=rows[1]["scene_id"]),
        lambda rows: rows[0].update(natural_observation=rows[0]["truth"]),
        lambda rows: rows[0].update(evaluation_axes=["unknown"]),
        lambda rows: rows[0].update(evaluation_axes=["iid", "fact_order"]),
        lambda rows: rows[0].update(constraint_targets=[1]),
        lambda rows: rows[0].pop("semantic_scene_id"),
    ],
)
def test_evaluation_validation_fails_closed_on_malformed_factorial_rows(
    mutation: object,
) -> None:
    rows = copy.deepcopy(_evaluation_rows())
    mutation(rows)

    with pytest.raises(StudyBError):
        validate_evaluation_rows(rows)


def test_study_a_base_rows_are_the_direct_five_axis_evaluation_input() -> None:
    graph_axes = (
        "canonical",
        "variable_permuted",
        "error_location_permuted",
        "fact_order_permuted",
        "equivalent_basis_graph_ood",
    )
    rows = []
    for checkpoint in ("Base", "T"):
        for parent_index in range(32):
            for index, graph_axis in enumerate(graph_axes):
                rows.append(
                    {
                        "scenario_id": f"study-a-{parent_index}::{graph_axis}",
                        "source_scene_id": f"study-a-source-{parent_index}",
                        "orbit_parent": f"study-a-parent-{parent_index}",
                        "checkpoint": checkpoint,
                        "checkpoint_sha256": MODEL_SHA if checkpoint == "Base" else "f" * 64,
                        "family": "cross_series",
                        "split": "independent_v4_support_dev",
                        "graph_axis": graph_axis,
                        "truth": [9, 2, 3, 4],
                        "observed": [8, 2, 3, 4],
                        "constraint_matrix": [[1, 1, 0, 0], [0, 1, 1, 0]],
                        "constraint_targets": [11, 5],
                        "prompt_sha256": f"{index}" * 64,
                        "source_sha256": {"raw_archive": RAW_SHA},
                    }
                )

    converted = evaluation_rows_from_study_a(rows)

    assert len(converted) == 160
    assert {row["evaluation_axes"][0] for row in converted} == {
        "iid",
        "variable_permutation",
        "error_position",
        "fact_order",
        "constraint_graph",
    }
    assert all(row["natural_observation"] == [8, 2, 3, 4] for row in converted)


def test_world_parser_and_offline_gate_are_strict() -> None:
    assert parse_world(" 9, 2,3,4 ") == (9, 2, 3, 4)
    assert parse_world("answer=9,2,3,4") is None
    assert parse_world(123) is None
    require_offline_environment(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    with pytest.raises(StudyBError, match="HF_DATASETS_OFFLINE"):
        require_offline_environment({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})


def test_summary_reports_registered_iid_and_structural_axes() -> None:
    rows = _evaluation_rows()
    outputs = [{"scene_id": row["scene_id"], "completion": "9,2,3,4"} for row in rows]
    summary, enriched = summarize_evaluations(rows, outputs)

    assert summary["overall"]["exact_world_rate"] == 1.0
    assert set(summary["by_axis"]) == {
        "iid",
        "variable_permutation",
        "error_position",
        "fact_order",
        "constraint_graph",
        "structural_ood",
    }
    assert summary["by_axis"]["constraint_graph"]["relational_genuine_recovery_rate"] == 1.0
    assert all(row["parsed_world"] == [9, 2, 3, 4] for row in enriched)


def test_summary_rejects_output_count_identity_schema_and_margin_drift() -> None:
    rows = _evaluation_rows()
    with pytest.raises(StudyBError, match="exactly one"):
        summarize_evaluations(rows, [])
    valid = [{"scene_id": row["scene_id"], "completion": "9,2,3,4"} for row in rows]
    duplicate = copy.deepcopy(valid)
    duplicate[1]["scene_id"] = duplicate[0]["scene_id"]
    with pytest.raises(StudyBError, match="duplicate"):
        summarize_evaluations(rows, duplicate)
    unknown = copy.deepcopy(valid)
    unknown[0]["extra"] = True
    with pytest.raises(StudyBError, match="unregistered"):
        summarize_evaluations(rows, unknown)
    invalid_margin = copy.deepcopy(valid)
    invalid_margin[0]["candidate_margin"] = float("nan")
    with pytest.raises(StudyBError, match="finite"):
        summarize_evaluations(rows, invalid_margin)


def test_full_fake_study_b_is_fresh_base_budget_matched_and_hashes_adapters(
    tmp_path: Path,
) -> None:
    backend = _FakeBackend()
    output = tmp_path / "study-b"

    result = run_study_b(
        support_package=_support_package(),
        evaluation_rows=_evaluation_rows(),
        output=output,
        backend=backend,
        expected_model_sha256=MODEL_SHA,
        seed=PILOT_SEED,
    )

    assert result["status"] == "STUDY_B_SINGLE_SEED_COMPLETE"
    assert backend.loaded == backend.released == ["B0", "B1", "B2", "B3"]
    assert len({token for token in result["base_load_tokens"].values()}) == 4
    assert backend.prompts["B0"] == backend.prompts["B3"]
    assert result["primary_contrasts"]["B3_minus_B2"]["iid_exact_world_rate"] == 0.0
    assert result["primary_contrasts"]["B3_minus_B2"]["constraint_graph_exact_world_rate"] == 1.0
    paired = result["primary_contrasts"]["paired_inference"]
    assert paired["bootstrap"] == {
        "method": "paired_scene_cluster_percentile",
        "seed": 2026082202,
        "resamples": 10_000,
        "confidence_level": 0.95,
    }
    assert paired["relational_constraint_graph"]["exact_world"]["ci95"] == [1.0, 1.0]
    assert result["stop_signal"]["triggered"] is True
    for arm in ("B0", "B1", "B2", "B3"):
        arm_result = json.loads((output / "arms" / arm / "result.json").read_text())
        assert arm_result["adapter_tree_sha256"] == tree_sha256(
            output / "arms" / arm / "final_adapter"
        )
        assert arm_result["trainable_manifest"]["vision_frozen"] is True
        assert (output / "arms" / arm / "training_log.json").is_file()
        assert (output / "arms" / arm / "evaluation_rows.jsonl").is_file()
    assert (
        canonical_sha256(json.loads((output / "run_manifest.json").read_text()))
        == result["run_manifest_sha256"]
    )


def test_resume_skips_atomically_completed_arms_and_completed_run_is_idempotent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "study-b"
    first = _FakeBackend(fail_once_on="B2")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=output,
            backend=first,
            expected_model_sha256=MODEL_SHA,
            seed=PILOT_SEED,
        )
    assert (output / "arms" / "B0" / "result.json").is_file()
    assert (output / "arms" / "B1" / "result.json").is_file()
    assert not (output / "arms" / "B2").exists()

    resumed = _FakeBackend()
    result = run_study_b(
        support_package=_support_package(),
        evaluation_rows=_evaluation_rows(),
        output=output,
        backend=resumed,
        expected_model_sha256=MODEL_SHA,
        seed=PILOT_SEED,
        resume=True,
    )
    assert resumed.loaded == ["B2", "B3"]

    no_calls = _FakeBackend()
    repeated = run_study_b(
        support_package=_support_package(),
        evaluation_rows=_evaluation_rows(),
        output=output,
        backend=no_calls,
        expected_model_sha256=MODEL_SHA,
        seed=PILOT_SEED,
        resume=True,
    )
    assert repeated == result
    assert no_calls.loaded == []


def test_resume_rejects_tampered_training_log(tmp_path: Path) -> None:
    output = tmp_path / "study-b"
    run_study_b(
        support_package=_support_package(),
        evaluation_rows=_evaluation_rows(),
        output=output,
        backend=_FakeBackend(),
        expected_model_sha256=MODEL_SHA,
    )
    (output / "arms" / "B0" / "training_log.json").write_text(
        '[{"step":1,"loss":0.0}]', encoding="utf-8"
    )

    with pytest.raises(StudyBError, match="training log changed"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=output,
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
            resume=True,
        )


def test_partial_resume_rebuilds_metrics_from_evaluation_evidence(tmp_path: Path) -> None:
    output = tmp_path / "study-b"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=output,
            backend=_FakeBackend(fail_once_on="B2"),
            expected_model_sha256=MODEL_SHA,
        )
    result_path = output / "arms" / "B0" / "result.json"
    result = json.loads(result_path.read_text())
    result["evaluation_metrics"]["overall"]["exact_world_rate"] = 1.0
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(StudyBError, match="evaluation metrics drifted"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=output,
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
            resume=True,
        )


def test_run_rejects_reused_base_load_token(tmp_path: Path) -> None:
    class _ReusedBaseBackend(_FakeBackend):
        def load_base(self, *, arm: str, expected_model_sha256: str):
            result = super().load_base(arm=arm, expected_model_sha256=expected_model_sha256)
            result["load_token"] = "same-base-session"
            return result

    with pytest.raises(StudyBError, match="fresh Base"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=tmp_path / "duplicate-base",
            backend=_ReusedBaseBackend(),
            expected_model_sha256=MODEL_SHA,
        )


def test_run_rejects_output_overwrite_missing_resume_and_manifest_drift(tmp_path: Path) -> None:
    output = tmp_path / "study-b"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=output,
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
        )
    with pytest.raises(StudyBError, match="does not exist"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=tmp_path / "missing",
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
            resume=True,
        )
    (output / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StudyBError, match="manifest differs"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=output,
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
            resume=True,
        )


def test_run_rejects_training_evaluation_semantic_overlap(tmp_path: Path) -> None:
    rows = _evaluation_rows()
    for row in rows[:5]:
        row["semantic_scene_id"] = "train-parent-0"

    with pytest.raises(StudyBError, match="overlap"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=rows,
            output=tmp_path / "overlap",
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
        )


def test_runtime_rejects_nonregistered_seed_and_false_freeze_claim(tmp_path: Path) -> None:
    with pytest.raises(StudyBError, match=str(PILOT_SEED)):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=tmp_path / "wrong-seed",
            backend=_FakeBackend(),
            expected_model_sha256=MODEL_SHA,
            seed=7,
        )

    class _UnsafeBackend(_FakeBackend):
        def train(self, **kwargs: object) -> dict[str, object]:
            outcome = super().train(**kwargs)
            outcome["trainable_manifest"]["merger_frozen"] = False
            return outcome

    with pytest.raises(StudyBError, match="vision and merger"):
        run_study_b(
            support_package=_support_package(),
            evaluation_rows=_evaluation_rows(),
            output=tmp_path / "unsafe",
            backend=_UnsafeBackend(),
            expected_model_sha256=MODEL_SHA,
            seed=PILOT_SEED,
        )


def test_study_b_cli_is_inert_without_execute_or_exact_ack() -> None:
    script = ROOT / "scripts/v5/13_run_study_b.py"
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    default = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    no_ack = subprocess.run(
        [sys.executable, str(script), "--execute"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert default.returncode == no_ack.returncode == 2
    assert "BLOCKED" in default.stdout
    assert "BLOCKED" in no_ack.stdout


def test_study_b_cli_loads_study_a_per_scenario_jsonl_as_evaluation_rows(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    path = tmp_path / "legacy_independent_per_scenario.jsonl"
    graph_axes = (
        "canonical",
        "variable_permuted",
        "error_location_permuted",
        "fact_order_permuted",
        "equivalent_basis_graph_ood",
    )
    path.write_text(
        "".join(
            json.dumps(
                {
                    "scenario_id": f"study-a-{parent_index}::{graph_axis}",
                    "source_scene_id": f"study-a-source-{parent_index}",
                    "orbit_parent": f"study-a-parent-{parent_index}",
                    "checkpoint": "Base",
                    "checkpoint_sha256": MODEL_SHA,
                    "family": "cross_series",
                    "split": "independent_v4_support_dev",
                    "graph_axis": graph_axis,
                    "truth": [9, 2, 3, 4],
                    "observed": [8, 2, 3, 4],
                    "constraint_matrix": [[1, 1, 0, 0], [0, 1, 1, 0]],
                    "constraint_targets": [11, 5],
                    "prompt_sha256": "c" * 64,
                },
                sort_keys=True,
            )
            + "\n"
            for parent_index in range(32)
            for graph_axis in graph_axes
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "V5_STUDY_A_ATOMICALLY_PUBLISHED",
        "source_sha256": {"Base": MODEL_SHA, "T": "f" * 64, "raw_archive": RAW_SHA},
        "files": {
            path.name: {
                "sha256": cli.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    rows = cli._load_evaluation(path)

    assert len(rows) == 160
    assert {row["evaluation_axes"][0] for row in rows} == {
        "iid",
        "variable_permutation",
        "error_position",
        "fact_order",
        "constraint_graph",
    }
    assert all(row["source_sha256"] == {"raw_archive": RAW_SHA} for row in rows)

    manifest["files"][path.name]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(StudyBError, match="manifest.*SHA-256 mismatch"):
        cli._load_evaluation(path)


def test_study_b_cli_rehashes_phase2a_source_chain(tmp_path: Path) -> None:
    cli = _load_cli()
    parent = tmp_path / "parent_manifest.json"
    child = tmp_path / "child_manifest.json"
    frozen = tmp_path / "frozen_scenes.jsonl"
    parent.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    frozen.write_text('{"semantic_scene_id":"train-parent-0"}\n', encoding="utf-8")
    parent_hash = cli.sha256_file(parent)
    frozen_hash = cli.sha256_file(frozen)
    child.write_text(
        json.dumps(
            {
                "status": "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN",
                "parent_manifest_sha256": parent_hash,
                "parent_manifest_modified": False,
                "frozen_scenes_sha256": frozen_hash,
                "semantic_scene_count": 96,
                "base_sha256": MODEL_SHA,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    support = _support_package()
    support["source_provenance"] = {
        "parent_manifest_sha256": parent_hash,
        "child_manifest_sha256": cli.sha256_file(child),
        "frozen_scenes_sha256": frozen_hash,
    }

    observed = cli._validate_support_source_provenance(
        support=support,
        parent_manifest=parent,
        child_manifest=child,
        frozen_scenes=frozen,
    )

    assert observed == support["source_provenance"]
    frozen.write_text('{"semantic_scene_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(StudyBError, match="frozen_scenes_sha256"):
        cli._validate_support_source_provenance(
            support=support,
            parent_manifest=parent,
            child_manifest=child,
            frozen_scenes=frozen,
        )
