"""Audited, non-executing plans for the large-GPU veRL boundary."""

from __future__ import annotations

import builtins
import hashlib
import json
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from compbias.models.qwen_vl import (
    DEFAULT_MODEL_NAME,
    PINNED_MODEL_REVISION,
    PINNED_TRANSFORMERS_REVISION,
    PINNED_VERL_REVISION,
    PINNED_VLLM_REVISION,
    VLMPreflightConfig,
)
from compbias.rl.verl_entrypoints import (
    AUDITED_GRPO_LEAF_KEYS,
    VerlExecutionPlan,
    VLMExecutionEvidence,
    _validate_phase_d_answer_balance,
    build_grpo_execution_plan,
    build_sft_execution_plan,
    load_execution_gate_evidence,
    require_machine_verified_cuda,
)


def _preflight(
    *,
    acknowledged: bool = True,
    audited: bool = True,
    model_revision: str = PINNED_MODEL_REVISION,
) -> VLMPreflightConfig:
    return VLMPreflightConfig(
        model_name=DEFAULT_MODEL_NAME,
        model_revision=model_revision,
        transformers_revision=PINNED_TRANSFORMERS_REVISION,
        verl_revision=PINNED_VERL_REVISION,
        vllm_revision=PINNED_VLLM_REVISION,
        acknowledge_large_gpu_run=acknowledged,
        verl_api_audited=audited,
    )


def _leaf_keys(value: Any, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {prefix}
    keys: set[str] = set()
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.update(_leaf_keys(child, path))
    return keys


def _snapshot_path() -> str:
    return f"/srv/hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/{PINNED_MODEL_REVISION}"


def test_sft_builder_returns_a_frozen_plan_and_never_claims_execution(
    sft_execution_evidence: VLMExecutionEvidence,
) -> None:
    plan = build_sft_execution_plan(
        _preflight(audited=False),
        evidence=sft_execution_evidence,
        dataset_manifest=sft_execution_evidence.dataset_manifest,
    )

    assert isinstance(plan, VerlExecutionPlan)
    assert plan.stage == "structured_sft"
    assert plan.execution_status == "not_started"
    assert plan.large_gpu_started is False
    assert plan.audited_verl_keys == ()
    assert plan.verl_config is None
    assert plan.requirements["minimum_parse_rate"] == pytest.approx(0.98)
    assert plan.requirements["verl_configuration_status"] == "deferred"
    assert plan.requirements["execution_permitted"] is False
    assert plan.preflight.verl_api_audited is False
    assert "verl" not in plan.to_mapping()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        plan.stage = "executed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.requirements["minimum_parse_rate"] = 0.0  # type: ignore[index]


def test_grpo_builder_emits_exactly_the_audited_verl_leaf_keys(
    execution_evidence: VLMExecutionEvidence,
) -> None:
    plan = build_grpo_execution_plan(
        _preflight(),
        evidence=execution_evidence,
        learning_rate=2.0e-6,
        mini_batch_size=8,
        rollout_samples=4,
        project_name="compbias",
        experiment_name="joint-outcome-fixture",
    )
    payload = plan.to_mapping()

    assert plan.stage == "joint_outcome_rl"
    assert plan.execution_status == "not_started"
    assert plan.large_gpu_started is False
    assert tuple(sorted(plan.audited_verl_keys)) == tuple(sorted(AUDITED_GRPO_LEAF_KEYS))
    assert _leaf_keys(payload["verl"]) == set(AUDITED_GRPO_LEAF_KEYS)
    assert "provenance" not in payload["verl"]
    assert (
        payload["verl"]["actor_rollout_ref"]["model"]["path"]
        == execution_evidence.model_snapshot.path
    )
    assert payload["requirements"]["model_snapshot_revalidation_required_at_execution"] is True
    assert payload["requirements"]["local_files_only"] is True
    assert payload["requirements"]["trust_remote_code"] is False
    assert payload["requirements"]["use_safetensors"] is True
    assert payload["requirements"]["network_access"] == "disabled"
    assert payload["requirements"]["execution_permitted"] is False
    assert payload["requirements"]["external_authorization_status"] == "not_granted"
    assert payload["requirements"]["snapshot_authenticity"] == "self_consistency_only"
    assert payload["requirements"]["trusted_upstream_snapshot_inventory_status"] == "missing"
    assert payload["requirements"]["hardened_container_evidence_status"] == "pending"
    assert payload["requirements"]["container_sbom_status"] == "pending"
    assert payload["artifact_visibility"] == "private_operator_only"
    assert payload["reward_contract"] == {
        "outcome_only": True,
        "perception_reward_weight": 0.0,
        "process_reward_weight": 0.0,
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_preflight(acknowledged=False), "acknowledg"),
        (_preflight(audited=False), r"veRL|audit"),
        (
            _preflight(model_revision="different-fixed-revision"),
            r"frozen|model.*revision",
        ),
    ],
)
def test_every_large_gpu_gate_is_mandatory(
    config: VLMPreflightConfig,
    message: str,
    execution_evidence: VLMExecutionEvidence,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        build_grpo_execution_plan(
            config,
            evidence=execution_evidence,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"learning_rate": True}, TypeError, "learning_rate"),
        ({"learning_rate": float("inf")}, ValueError, "learning_rate"),
        ({"mini_batch_size": 1.5}, TypeError, "mini_batch_size"),
        ({"rollout_samples": 0}, ValueError, "rollout_samples"),
        ({"project_name": "contains spaces"}, ValueError, "project_name"),
        ({"experiment_name": "../escape"}, ValueError, "experiment_name"),
    ],
)
def test_grpo_builder_rejects_ambiguous_or_unsafe_values(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
    execution_evidence: VLMExecutionEvidence,
) -> None:
    arguments: dict[str, object] = {
        "evidence": execution_evidence,
    }
    arguments.update(overrides)

    with pytest.raises(error, match=message):
        build_grpo_execution_plan(_preflight(), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("dataset_manifest", "", ValueError),
        ("dataset_manifest", "bad\x00path", ValueError),
        ("minimum_parse_rate", True, TypeError),
        ("minimum_parse_rate", 0.97, ValueError),
        ("minimum_parse_rate", 1.01, ValueError),
    ],
)
def test_sft_builder_validates_metadata_without_touching_the_filesystem(
    field: str,
    value: object,
    error: type[Exception],
    sft_execution_evidence: VLMExecutionEvidence,
) -> None:
    arguments: dict[str, object] = {
        "evidence": sft_execution_evidence,
        "dataset_manifest": sft_execution_evidence.dataset_manifest,
    }
    arguments[field] = value

    with pytest.raises(error):
        build_sft_execution_plan(_preflight(audited=False), **arguments)  # type: ignore[arg-type]


def test_plan_serialization_is_a_defensive_copy(
    execution_evidence: VLMExecutionEvidence,
) -> None:
    plan = build_grpo_execution_plan(
        _preflight(),
        evidence=execution_evidence,
    )

    first = plan.to_mapping()
    first["verl"]["trainer"]["total_epochs"] = 99
    second = plan.to_mapping()

    assert second["verl"]["trainer"]["total_epochs"] == 1
    assert second["artifact_type"] == "execution_plan"
    assert second["execution_status"] == "not_started"
    assert second["large_gpu_started"] is False


def test_builders_do_not_import_trainers_download_or_spawn_processes(
    monkeypatch,
    execution_evidence: VLMExecutionEvidence,
    sft_execution_evidence: VLMExecutionEvidence,
) -> None:
    forbidden_imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "verl", "vllm"}:
            forbidden_imports.append(name)
            raise AssertionError(f"heavy import attempted: {name}")
        return real_import(name, *args, **kwargs)

    def forbidden_process(*args, **kwargs):
        raise AssertionError("execution-plan builder attempted to start a process")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", forbidden_process)
    monkeypatch.setattr(subprocess, "Popen", forbidden_process)

    build_sft_execution_plan(
        _preflight(audited=False),
        evidence=sft_execution_evidence,
        dataset_manifest=sft_execution_evidence.dataset_manifest,
    )
    build_grpo_execution_plan(
        _preflight(),
        evidence=execution_evidence,
    )

    assert forbidden_imports == []


def test_machine_verified_cuda_requires_the_smoke_tested_device() -> None:
    with pytest.raises(RuntimeError, match=r"smoke|GPU"):
        require_machine_verified_cuda(
            detected_gpu_uuids=("GPU-real",),
            smoke_gpu_uuids=("GPU-other",),
        )


def test_builder_uses_machine_probe_and_rejects_no_local_cuda(
    execution_evidence: VLMExecutionEvidence,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: (),
    )

    with pytest.raises(RuntimeError, match=r"CUDA|GPU|detected"):
        build_grpo_execution_plan(_preflight(), evidence=execution_evidence)


def test_builder_revalidates_snapshot_files_after_evidence_load(
    execution_evidence: VLMExecutionEvidence,
) -> None:
    (Path(execution_evidence.model_snapshot.path) / "model.safetensors").write_bytes(
        b"changed-after-load"
    )

    with pytest.raises(RuntimeError, match=r"snapshot|SHA-256|size"):
        build_grpo_execution_plan(_preflight(), evidence=execution_evidence)


def test_builder_revalidates_dataset_tree_after_evidence_load(
    execution_evidence: VLMExecutionEvidence,
) -> None:
    artifact_root = Path(execution_evidence.dataset_manifest).parent
    dataset = artifact_root / "datasets/cva_v2/dataset.jsonl"
    dataset.write_text(dataset.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"dataset|JSONL|SHA-256"):
        build_grpo_execution_plan(_preflight(), evidence=execution_evidence)


@pytest.mark.parametrize("mutation", ("tamper", "delete", "symlink"))
def test_builder_revalidates_contact_sheets_after_evidence_load(
    execution_evidence: VLMExecutionEvidence,
    mutation: str,
) -> None:
    artifact_root = Path(execution_evidence.dataset_manifest).parent
    sheet = artifact_root / "figures/cva_v2/cva_contact_sheet_01.png"
    if mutation == "tamper":
        sheet.write_bytes(b"tampered")
    elif mutation == "delete":
        sheet.unlink()
    else:
        outside = artifact_root / "outside-contact-sheet.png"
        outside.write_bytes(sheet.read_bytes())
        sheet.unlink()
        sheet.symlink_to(outside)

    with pytest.raises(RuntimeError, match=r"contact-sheet|contact sheet|symlink|SHA-256"):
        build_grpo_execution_plan(_preflight(), evidence=execution_evidence)


def test_builder_binds_only_the_smoke_verified_cuda_intersection(
    execution_evidence: VLMExecutionEvidence,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-unverified", "GPU-fixture-a100"),
    )

    plan = build_grpo_execution_plan(_preflight(), evidence=execution_evidence)

    assert plan.preflight.gpu_devices == ("GPU-fixture-a100",)
    assert "GPU-unverified" not in plan.preflight.gpu_devices
    assert plan.requirements["executor_gpu_uuid_binding_status"] == "pending"


def test_execution_evidence_cannot_be_constructed_without_the_loader() -> None:
    with pytest.raises(TypeError):
        VLMExecutionEvidence(
            phase_d_reviewer="reviewer-fixture",
            phase_d_reviewed_images=200,
            target_container_gpu_uuids=("GPU-real",),
            parser_validity_rate=0.99,
            model_snapshot=None,
            dataset_manifest="/srv/dataset-manifest.json",
            dataset_manifest_sha256="c" * 64,
            dataset_content_sha256="d" * 64,
            phase_d_audit_sha256="e" * 64,
            execution_audit_sha256="f" * 64,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gate_artifacts(
    root: Path,
    snapshot: Path,
    *,
    stage: str = "joint_outcome_rl",
) -> tuple[Path, str, Path, str]:
    from collections import Counter

    from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
    from compbias.eval.dataset_contract import FROZEN_CVA_V2_GENERATOR_CONFIG
    from scripts.audit_dataset import (
        _answer_balance,
        _style_semantic_joint_independence,
        _visual_factor_realization_audit,
    )

    snapshot_files = {
        "config.json": b"{}",
        "preprocessor_config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": b"fixture-weights",
    }
    manifest_files = []
    for relative, content in snapshot_files.items():
        path = snapshot / relative
        path.write_bytes(content)
        manifest_files.append(
            {
                "path": relative,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    snapshot_manifest = root / "snapshot-manifest.json"
    snapshot_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete_snapshot": True,
                "model_name": DEFAULT_MODEL_NAME,
                "revision": PINNED_MODEL_REVISION,
                "files": manifest_files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    snapshot_manifest_sha256 = _sha256(snapshot_manifest)
    dataset_manifest = root / "dataset-manifest.json"
    generator_config = json.loads(json.dumps(FROZEN_CVA_V2_GENERATOR_CONFIG))
    canonical_samples = generate_dataset(GeneratorConfig(**generator_config))
    sample_ids = sorted(sample.sample_id for sample in canonical_samples)
    visual_styles = generator_config["visual_styles"]
    style_counts = Counter(sample.split_keys.visual_style for sample in canonical_samples)
    visual_factor_audit = _visual_factor_realization_audit(
        canonical_samples,
        configured_styles=tuple(visual_styles),
        expected_style_counts=style_counts,
        style_counterbalance_violations=(),
        fully_cross_iid_visual_styles=True,
    )
    joint_audit = _style_semantic_joint_independence(
        canonical_samples,
        fully_cross_iid_visual_styles=True,
    )
    answer_balance = _answer_balance(
        canonical_samples,
        expected_samples=canonical_samples,
        samples_per_family_per_split=generator_config["samples_per_family_per_split"],
    )
    canonical = lambda value: json.dumps(  # noqa: E731
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dataset_content_sha256 = hashlib.sha256(
        canonical([{"sample_id": sample_id} for sample_id in sample_ids])
    ).hexdigest()
    dataset_file = root / "datasets/cva_v2/dataset.jsonl"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}, sort_keys=True) + "\n" for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    images_dir = root / "datasets/cva_v2/images"
    images_dir.mkdir(parents=True)
    for sample_id in sample_ids:
        (images_dir / f"{sample_id}.png").write_bytes(sample_id.encode("utf-8"))
    image_sha256 = {
        sample_id: hashlib.sha256(sample_id.encode("utf-8")).hexdigest() for sample_id in sample_ids
    }
    image_set_sha256 = hashlib.sha256(canonical(image_sha256)).hexdigest()
    contact_sheets = [f"figures/cva_v2/cva_contact_sheet_{index:02d}.png" for index in range(1, 74)]
    contact_sheet_dir = root / "figures/cva_v2"
    contact_sheet_dir.mkdir(parents=True)
    for path in contact_sheets:
        (contact_sheet_dir / Path(path).name).write_bytes(path.encode("utf-8"))
    unsigned_dataset_manifest = {
        "dataset_name": "cva_v2",
        "schema_version": "2.0",
        "sample_count": 1820,
        "sample_ids": sample_ids,
        "content_sha256": dataset_content_sha256,
        "config_sha256": hashlib.sha256(canonical(generator_config)).hexdigest(),
        "generator_config": generator_config,
        "render_config": {"height": 256, "samples_per_contact_sheet": 25, "width": 256},
        "dataset_file_sha256": _sha256(dataset_file),
        "image_sha256": image_sha256,
        "jsonl_path": "datasets/cva_v2/dataset.jsonl",
        "images_dir": "datasets/cva_v2/images",
        "rendered_image_count": 1820,
        "solver_checks": 1820,
        "solver_pass_rate": 1.0,
        "roundtrip_checks": 4020,
        "roundtrip_pass_rate": 1.0,
        "contact_sheets": contact_sheets,
        "contact_sheet_sha256": {
            Path(path).name: hashlib.sha256(path.encode("utf-8")).hexdigest()
            for path in contact_sheets
        },
        "preregistered_ood_factors": ["visual_style", "error_mechanism"],
    }
    dataset_manifest_self_sha256 = hashlib.sha256(
        json.dumps(
            unsigned_dataset_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    dataset_manifest.write_text(
        json.dumps(
            {
                **unsigned_dataset_manifest,
                "manifest_sha256": dataset_manifest_self_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dataset_manifest_sha256 = _sha256(dataset_manifest)
    phase_d = root / "phase-d.json"
    phase_d_payload = {
        "audit_report_schema_version": 2,
        "sample_count": 1820,
        "split_audit": {
            "scene_template_leaks": [],
            "answer_leaks": [],
            "visual_style_leaks": [],
            "error_mechanism_leaks": [],
            "ood_pair_mismatches": [],
            "ood_pair_count": 100,
            "preregistered_ood_factors": ["visual_style", "error_mechanism"],
            "ood_changed_factors": ["visual_style", "error_mechanism"],
        },
        "split_audit_error": None,
        "split_clean": True,
        "solver_passes": 1820,
        "solver_pass_rate": 1.0,
        "roundtrip_passes": 4020,
        "roundtrip_total": 4020,
        "roundtrip_pass_rate": 1.0,
        "error_solver_passes": 4020,
        "error_solver_pass_rate": 1.0,
        "rendered_image_count": 1820,
        "missing_images": [],
        "extra_images": [],
        "image_set_matches": True,
        "rendered_image_count_matches": True,
        "contact_sheet_sha256_matches": True,
        "contact_sheet_hash_mismatches": [],
        "manifest_sample_count_matches": True,
        "manifest_sample_ids_match": True,
        "manifest_content_sha256_matches": True,
        "manifest_config_sha256_matches": True,
        "manifest_dataset_file_sha256_matches": True,
        "manifest_image_sha256_matches": True,
        "manifest_self_sha256_matches": True,
        "preregistered_ood_factors_match_config": True,
        "noncanonical_rows": [],
        "image_path_mismatches": [],
        "privacy_issues": [],
        "image_question_answer_collisions": [],
        "style_counterbalance_violations": [],
        "evidence_manifest_sha256": dataset_manifest_self_sha256,
        "evidence_image_set_sha256": image_set_sha256,
        "visual_review_present": True,
        "human_reviewer_signoff": True,
        "human_review_binding_matches": True,
        "human_review": {
            "signoff": True,
            "reviewer": "reviewer-fixture",
            "reviewer_type": "human",
            "review_date": "2026-08-14",
            "review_result": "pass",
            "reviewed_image_count": 200,
            "reviewed_sample_ids": sample_ids[:200],
            "contact_sheets_reviewed": 73,
            "binding_matches": True,
            "manifest_self_sha256": dataset_manifest_self_sha256,
            "integrity_scope": "self-reported review record; no external signature verified",
        },
        "visual_factor_realization_audit": visual_factor_audit,
        "ood_image_shift": {
            "complete": True,
            "checked_pair_count": 100,
            "violations": [],
        },
        "style_semantic_joint_independence": joint_audit,
        "deterministic_replay": {
            "complete": True,
            "generator_matches": True,
            "renderer_matches": True,
            "contact_sheets_match": True,
            "generator_mismatches": [],
            "renderer_mismatches": [],
            "contact_sheet_mismatches": [],
        },
        "answer_balance": answer_balance,
        "dataset": {
            "manifest_path": "dataset-manifest.json",
            "manifest_file_sha256": dataset_manifest_sha256,
            "manifest_self_sha256": dataset_manifest_self_sha256,
            "content_sha256": dataset_content_sha256,
            "image_set_sha256": image_set_sha256,
        },
        "automatic_audit_clean": True,
        "phase_d_ready": True,
    }
    phase_d.write_text(json.dumps(phase_d_payload), encoding="utf-8")
    execution_payload = {
        "schema_version": 2,
        "stage": stage,
        "training_invoked": False,
        "large_gpu_started": False,
        "pins": {
            "model_name": DEFAULT_MODEL_NAME,
            "model_revision": PINNED_MODEL_REVISION,
            "transformers_revision": PINNED_TRANSFORMERS_REVISION,
            "verl_revision": PINNED_VERL_REVISION,
            "vllm_revision": PINNED_VLLM_REVISION,
        },
        "target_container_smoke": {
            "passed": True,
            "dockerfile_sha256": (
                "be8bd117fc415690c2d433e2e3c8832e6a96dd6de4e799be6a4be05c9eb4f300"
            ),
            "container_image_digest": "sha256:" + "d" * 64,
            "gpu_uuids": ["GPU-fixture-a100"],
            "runtime_packages": {
                "torch": "2.11.0",
                "transformers": PINNED_TRANSFORMERS_REVISION,
                "vllm": PINNED_VLLM_REVISION,
            },
            "verl_revision": PINNED_VERL_REVISION,
            "verl_worktree_clean": True,
            "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
            "dataset_manifest_file_sha256": dataset_manifest_sha256,
        },
        "model_snapshot": {
            "path": str(snapshot),
            "revision": PINNED_MODEL_REVISION,
            "local_files_only": True,
            "manifest_path": str(snapshot_manifest),
            "manifest_sha256": snapshot_manifest_sha256,
        },
        "dataset": {
            "manifest_path": str(dataset_manifest),
            "manifest_file_sha256": dataset_manifest_sha256,
            "manifest_self_sha256": dataset_manifest_self_sha256,
            "content_sha256": dataset_content_sha256,
            "image_set_sha256": image_set_sha256,
        },
        "external_authorization": {
            "status": "not_granted",
        },
    }
    if stage == "joint_outcome_rl":
        checkpoint = root / "sft-checkpoint.safetensors"
        checkpoint.write_bytes(b"fixture-sft-checkpoint")
        checkpoint_sha256 = _sha256(checkpoint)
        adapter = root / "text-only-state-adapter.py"
        adapter.write_text("def adapt(state): return str(state)\n", encoding="utf-8")
        adapter_sha256 = _sha256(adapter)
        bindings = {
            "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
            "dataset_manifest_file_sha256": dataset_manifest_sha256,
            "sft_checkpoint_sha256": checkpoint_sha256,
        }
        execution_payload.update(
            {
                "sft_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha256,
                    "model_snapshot_manifest_sha256": snapshot_manifest_sha256,
                    "dataset_manifest_file_sha256": dataset_manifest_sha256,
                },
                "parser_audit": {
                    "measured_on_model": True,
                    "validity_rate": 0.99,
                    **bindings,
                },
                "state_injection_audit": {
                    "passed": True,
                    "image_hidden": True,
                    "isolation_mode": "separate_text_only_worker",
                    "adapter_path": str(adapter),
                    "adapter_sha256": adapter_sha256,
                    "reviewed_adapter_sha256": adapter_sha256,
                    **bindings,
                },
                "fixed_reasoner_h1_audit": {
                    "passed": True,
                    "sign_prediction_above_chance": True,
                    "measurable_coupling_task_count": 1,
                    **bindings,
                },
                "verl_api_audit": {
                    "passed": True,
                    "revision": PINNED_VERL_REVISION,
                    "audited_leaf_keys": list(AUDITED_GRPO_LEAF_KEYS),
                },
            }
        )
    execution = root / "execution.json"
    execution.write_text(json.dumps(execution_payload), encoding="utf-8")
    return phase_d, _sha256(phase_d), execution, _sha256(execution)


@pytest.fixture
def execution_evidence(tmp_path: Path, monkeypatch) -> VLMExecutionEvidence:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(tmp_path, snapshot)
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-fixture-a100",),
    )
    return load_execution_gate_evidence(
        phase_d,
        execution,
        stage="joint_outcome_rl",
        phase_d_sha256=phase_d_sha256,
        execution_audit_sha256=execution_sha256,
    )


@pytest.fixture
def sft_execution_evidence(tmp_path_factory, monkeypatch) -> VLMExecutionEvidence:
    root = tmp_path_factory.mktemp("sft-evidence")
    snapshot = root / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        root,
        snapshot,
        stage="structured_sft",
    )
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-fixture-a100",),
    )
    return load_execution_gate_evidence(
        phase_d,
        execution,
        stage="structured_sft",
        phase_d_sha256=phase_d_sha256,
        execution_audit_sha256=execution_sha256,
    )


def test_execution_gate_loader_requires_all_prior_artifacts(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(tmp_path, snapshot)

    evidence = load_execution_gate_evidence(
        phase_d,
        execution,
        stage="joint_outcome_rl",
        phase_d_sha256=phase_d_sha256,
        execution_audit_sha256=execution_sha256,
    )

    assert evidence.phase_d_reviewed_images == 200
    assert evidence.model_snapshot.path == str(snapshot.resolve())
    assert evidence.target_container_gpu_uuids == ("GPU-fixture-a100",)


@pytest.mark.parametrize(
    "mutation",
    (
        "dataset_tamper",
        "dataset_symlink",
        "missing_image",
        "extra_image",
        "tampered_image",
        "symlink_image",
        "missing_contact_sheet",
        "extra_contact_sheet",
        "tampered_contact_sheet",
        "symlink_contact_sheet",
    ),
)
def test_execution_gate_loader_revalidates_the_live_dataset_tree(
    tmp_path: Path,
    mutation: str,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    dataset = tmp_path / "datasets/cva_v2/dataset.jsonl"
    images = tmp_path / "datasets/cva_v2/images"
    sample_image = images / "bar_chart_aggregate_calibration_000000_r00.png"
    sheet_dir = tmp_path / "figures/cva_v2"
    sample_sheet = sheet_dir / "cva_contact_sheet_01.png"
    if mutation == "dataset_tamper":
        dataset.write_text(dataset.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    elif mutation == "dataset_symlink":
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes(dataset.read_bytes())
        dataset.unlink()
        dataset.symlink_to(outside)
    elif mutation == "missing_image":
        sample_image.unlink()
    elif mutation == "extra_image":
        (images / "extra.png").write_bytes(b"extra")
    elif mutation == "tampered_image":
        sample_image.write_bytes(b"tampered")
    elif mutation == "symlink_image":
        outside = tmp_path / "outside.png"
        outside.write_bytes(sample_image.read_bytes())
        sample_image.unlink()
        sample_image.symlink_to(outside)
    elif mutation == "missing_contact_sheet":
        sample_sheet.unlink()
    elif mutation == "extra_contact_sheet":
        (sheet_dir / "extra.png").write_bytes(b"extra")
    elif mutation == "tampered_contact_sheet":
        sample_sheet.write_bytes(b"tampered")
    else:
        outside = tmp_path / "outside-sheet.png"
        outside.write_bytes(sample_sheet.read_bytes())
        sample_sheet.unlink()
        sample_sheet.symlink_to(outside)

    with pytest.raises(
        RuntimeError,
        match=r"dataset|JSONL|image|PNG|symlink|SHA-256|contact-sheet",
    ):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=phase_d_sha256,
            execution_audit_sha256=execution_sha256,
        )


def test_execution_gate_loader_rejects_legacy_or_minimal_phase_d_contract(
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    payload["audit_report_schema_version"] = 1
    phase_d.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"schema|closed|cva_v2|Phase D"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=execution_sha256,
        )


def test_execution_gate_loader_rejects_self_consistent_non_v2_dataset_manifest(
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, _execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    dataset_payload["dataset_name"] = "fixture"
    unsigned = {key: value for key, value in dataset_payload.items() if key != "manifest_sha256"}
    dataset_self_hash = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    dataset_payload["manifest_sha256"] = dataset_self_hash
    dataset_manifest.write_text(json.dumps(dataset_payload, sort_keys=True), encoding="utf-8")
    dataset_file_hash = _sha256(dataset_manifest)

    phase_payload = json.loads(phase_d.read_text(encoding="utf-8"))
    phase_payload["evidence_manifest_sha256"] = dataset_self_hash
    phase_payload["dataset"]["manifest_self_sha256"] = dataset_self_hash
    phase_payload["dataset"]["manifest_file_sha256"] = dataset_file_hash
    phase_payload["human_review"]["manifest_self_sha256"] = dataset_self_hash
    phase_d.write_text(json.dumps(phase_payload), encoding="utf-8")

    execution_payload = json.loads(execution.read_text(encoding="utf-8"))
    execution_payload["dataset"]["manifest_self_sha256"] = dataset_self_hash
    execution_payload["dataset"]["manifest_file_sha256"] = dataset_file_hash
    execution_payload["target_container_smoke"]["dataset_manifest_file_sha256"] = dataset_file_hash
    for key in (
        "sft_checkpoint",
        "parser_audit",
        "state_injection_audit",
        "fixed_reasoner_h1_audit",
    ):
        execution_payload[key]["dataset_manifest_file_sha256"] = dataset_file_hash
    execution.write_text(json.dumps(execution_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"cva_v2|frozen"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=_sha256(execution),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("visual_factor_realization_audit", "complete"), False),
        (("visual_factor_realization_audit", "sample_counts", "baseline"), 49),
        (("answer_balance", "numeric_exact_balance"), False),
        (("ood_image_shift", "checked_pair_count"), 99),
        (("style_semantic_joint_independence", "complete"), False),
        (("deterministic_replay", "contact_sheets_match"), False),
        (("roundtrip_total",), 1099),
    ],
)
def test_execution_gate_loader_rejects_incomplete_frozen_phase_d_evidence(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    phase_d.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Phase D|frozen|visual|answer|OOD"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=execution_sha256,
        )


def test_execution_gate_loader_rejects_unsigned_phase_d_review(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path, snapshot
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    payload["human_review"]["signoff"] = False
    phase_d.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"human|sign-off|Phase D"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=execution_sha256,
        )


def test_execution_gate_loader_rejects_nonpublic_phase_d_reviewer(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path, snapshot
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    payload["human_review"]["reviewer"] = "reviewer@example.com"
    phase_d.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"human|reviewer|Phase D"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=execution_sha256,
        )


def test_phase_d_consumer_rejects_self_consistent_answer_substitution(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, _execution, _execution_sha256 = _write_gate_artifacts(
        tmp_path, snapshot
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    group = payload["answer_balance"]["groups"]["digit_offset/train"]
    group["support"][0] = 10_000
    group["frequencies"][0]["answer"] = 10_000

    with pytest.raises(RuntimeError, match=r"deterministic|frozen"):
        _validate_phase_d_answer_balance(payload)


def test_execution_gate_loader_rejects_integer_substitution_for_ratio_float(
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    group = payload["answer_balance"]["groups"]["bar_chart_aggregate/train"]
    float_answer = next(
        answer for answer in group["support"] if type(answer) is float and answer.is_integer()
    )
    group["support"] = [
        int(answer) if type(answer) is float and answer == float_answer else answer
        for answer in group["support"]
    ]
    for frequency in group["frequencies"]:
        if type(frequency["answer"]) is float and frequency["answer"] == float_answer:
            frequency["answer"] = int(float_answer)
    phase_d.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"deterministic|frozen"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=execution_sha256,
        )


@pytest.mark.parametrize("gate", ["automatic_audit_clean", "phase_d_ready"])
def test_execution_gate_loader_requires_phase_d_aggregate_gates(
    tmp_path: Path,
    gate: str,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, _phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    payload = json.loads(phase_d.read_text(encoding="utf-8"))
    payload[gate] = False
    phase_d.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=gate):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=_sha256(phase_d),
            execution_audit_sha256=execution_sha256,
        )


def test_execution_plan_cannot_be_constructed_outside_a_builder() -> None:
    with pytest.raises(TypeError):
        VerlExecutionPlan(
            stage="joint_outcome_rl",
            preflight=None,
            requirements={
                "execution_permitted": True,
                "external_authorization_status": "granted",
                "previous_phase_a_c_artifacts_verified": True,
            },
        )


def test_execution_gate_loader_detects_tampered_artifacts(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(tmp_path, snapshot)
    execution.write_text(execution.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"SHA-256|hash"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=phase_d_sha256,
            execution_audit_sha256=execution_sha256,
        )


def test_execution_gate_loader_detects_tampered_snapshot_file(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(tmp_path, snapshot)
    (snapshot / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match=r"snapshot|SHA-256|hash"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=phase_d_sha256,
            execution_audit_sha256=execution_sha256,
        )


def test_execution_gate_loader_rejects_snapshot_symlinks(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, execution_sha256 = _write_gate_artifacts(tmp_path, snapshot)
    weights = snapshot / "model.safetensors"
    outside = tmp_path / "outside-weights"
    outside.write_bytes(weights.read_bytes())
    weights.unlink()
    weights.symlink_to(outside)

    with pytest.raises(RuntimeError, match=r"symlink"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=phase_d_sha256,
            execution_audit_sha256=execution_sha256,
        )


@pytest.mark.parametrize(
    "invalid_fragment",
    [
        '"schema_version": 2, "schema_version": 2',
        '"schema_version": NaN',
    ],
)
def test_execution_gate_loader_rejects_ambiguous_json(
    tmp_path: Path,
    invalid_fragment: str,
) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, _execution_sha256 = _write_gate_artifacts(
        tmp_path, snapshot
    )
    execution.write_text("{" + invalid_fragment + "}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"JSON"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=phase_d_sha256,
            execution_audit_sha256=_sha256(execution),
        )


def test_execution_gate_loader_rejects_excessive_json_depth(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    phase_d, phase_d_sha256, execution, _execution_sha256 = _write_gate_artifacts(
        tmp_path,
        snapshot,
    )
    payload = json.loads(execution.read_text(encoding="utf-8"))
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(70):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    payload["irrelevant"] = nested
    execution.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"depth|complex"):
        load_execution_gate_evidence(
            phase_d,
            execution,
            stage="joint_outcome_rl",
            phase_d_sha256=phase_d_sha256,
            execution_audit_sha256=_sha256(execution),
        )
