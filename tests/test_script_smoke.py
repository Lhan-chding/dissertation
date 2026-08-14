"""Import and ``--help`` smoke tests for every promised command-line entry point."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from compbias.models.qwen_vl import (
    DEFAULT_MODEL_NAME,
    PINNED_MODEL_REVISION,
    PINNED_TRANSFORMERS_REVISION,
    PINNED_VERL_DOCKERFILE_SHA256,
    PINNED_VERL_REVISION,
    PINNED_VLLM_REVISION,
)
from compbias.rl.verl_entrypoints import AUDITED_GRPO_LEAF_KEYS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "verify_theory.py",
    "generate_cva.py",
    "audit_dataset.py",
    "train_tabular.py",
    "train_neural.py",
    "train_visual_neural.py",
    "estimate_compensability.py",
    "preflight_vlm.py",
    "train_vlm_sft.py",
    "train_vlm_rl.py",
    "evaluate_checkpoint.py",
    "build_paper_tables.py",
)


def _load_script(script_name: str):
    script = REPOSITORY_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"_execution_{script.stem}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leaf_keys(value, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {prefix}
    keys: set[str] = set()
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        keys.update(_leaf_keys(child, path))
    return keys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_vlm_gate_artifacts(
    tmp_path: Path,
    *,
    stage: str = "joint_outcome_rl",
) -> tuple[Path, str, Path, str, Path]:
    from collections import Counter

    from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
    from compbias.eval.dataset_contract import FROZEN_CVA_V2_GENERATOR_CONFIG
    from scripts.audit_dataset import (
        _answer_balance,
        _style_semantic_joint_independence,
        _visual_factor_realization_audit,
    )

    snapshot = (
        tmp_path / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / PINNED_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
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
    snapshot_manifest = tmp_path / "snapshot-manifest.json"
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
    dataset_manifest = tmp_path / "dataset-manifest.json"
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
    dataset_file = tmp_path / "datasets/cva_v2/dataset.jsonl"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}, sort_keys=True) + "\n" for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    images_dir = tmp_path / "datasets/cva_v2/images"
    images_dir.mkdir(parents=True)
    for sample_id in sample_ids:
        (images_dir / f"{sample_id}.png").write_bytes(sample_id.encode("utf-8"))
    image_sha256 = {
        sample_id: hashlib.sha256(sample_id.encode("utf-8")).hexdigest() for sample_id in sample_ids
    }
    image_set_sha256 = hashlib.sha256(canonical(image_sha256)).hexdigest()
    contact_sheets = [f"figures/cva_v2/cva_contact_sheet_{index:02d}.png" for index in range(1, 74)]
    contact_sheet_dir = tmp_path / "figures/cva_v2"
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
    phase_d = tmp_path / "phase-d.json"
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
            "dockerfile_sha256": PINNED_VERL_DOCKERFILE_SHA256,
            "container_image_digest": "sha256:" + "d" * 64,
            "gpu_uuids": ["GPU-real"],
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
        checkpoint = tmp_path / "sft-checkpoint.safetensors"
        checkpoint.write_bytes(b"fixture-sft-checkpoint")
        checkpoint_sha256 = _sha256(checkpoint)
        adapter = tmp_path / "text-only-state-adapter.py"
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
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps(execution_payload), encoding="utf-8")
    return phase_d, _sha256(phase_d), execution, _sha256(execution), snapshot


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_script_import_is_side_effect_free_and_exposes_main(script_name: str) -> None:
    script = REPOSITORY_ROOT / "scripts" / script_name
    assert script.is_file(), f"missing promised entry point: {script.relative_to(REPOSITORY_ROOT)}"
    spec = importlib.util.spec_from_file_location(f"_smoke_{script.stem}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module.main)
    parameters = tuple(inspect.signature(module.main).parameters.values())
    assert len(parameters) == 1
    assert parameters[0].name == "argv"
    assert parameters[0].default is None


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_every_script_supports_help_without_optional_training_dependencies(
    script_name: str,
) -> None:
    script = REPOSITORY_ROOT / "scripts" / script_name

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage" in completed.stdout.lower()


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_entrypoint_rejects_unacknowledged_run_before_training(
    script_name: str,
    tmp_path: Path,
) -> None:
    config = tmp_path / "vlm-smoke.yaml"
    config.write_text(
        """\
model:
  name: Qwen/Qwen2.5-VL-3B-Instruct
  revision: model-commit-fixture
training:
  framework: verl
  transformers_revision: transformers-commit-fixture
  verl_revision: verl-commit-fixture
  vllm_revision: vllm-commit-fixture
"""
    )
    script = REPOSITORY_ROOT / "scripts" / script_name

    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    assert completed.returncode != 0
    assert "acknowledg" in combined_output


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_entrypoint_requires_artifact_backed_prior_gates(
    script_name: str,
    tmp_path: Path,
) -> None:
    config_name = (
        "qwen25vl3b_sft.yaml" if script_name == "train_vlm_sft.py" else "qwen25vl3b_joint_grpo.yaml"
    )
    script = REPOSITORY_ROOT / "scripts" / script_name
    config = REPOSITORY_ROOT / "configs" / "vlm" / config_name
    phase_d = REPOSITORY_ROOT / "artifacts/metrics/cva_audit.json"
    execution = REPOSITORY_ROOT / "artifacts/manifests/vlm_preflight.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--acknowledge-large-gpu-run",
            "--phase-d-audit",
            str(phase_d),
            "--phase-d-audit-sha256",
            _sha256(phase_d),
            "--execution-audit",
            str(execution),
            "--execution-audit-sha256",
            _sha256(execution),
            "--output-config",
            str(tmp_path / f"{script_name}.yaml"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    assert completed.returncode != 0
    assert "phase d" in combined_output
    assert "schema" in combined_output


def test_vlm_rl_cli_reuses_the_strict_grpo_plan_without_provenance_leaves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    phase_d, phase_d_sha256, execution, execution_sha256, snapshot = _write_vlm_gate_artifacts(
        tmp_path
    )
    output = tmp_path / "rl-plan.yaml"
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-real",),
    )
    module = _load_script("train_vlm_rl.py")

    result = module.main(
        [
            "--config",
            str(REPOSITORY_ROOT / "configs/vlm/qwen25vl3b_joint_grpo.yaml"),
            "--acknowledge-large-gpu-run",
            "--phase-d-audit",
            str(phase_d),
            "--phase-d-audit-sha256",
            phase_d_sha256,
            "--execution-audit",
            str(execution),
            "--execution-audit-sha256",
            execution_sha256,
            "--output-config",
            str(output),
        ]
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert result == 0
    assert _leaf_keys(payload["verl"]) == set(AUDITED_GRPO_LEAF_KEYS)
    assert "provenance" not in payload["verl"]
    assert payload["verl"]["actor_rollout_ref"]["model"]["path"] == str(snapshot)


def test_vlm_sft_cli_emits_only_a_non_executable_deferred_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    phase_d, phase_d_sha256, execution, execution_sha256, _snapshot = _write_vlm_gate_artifacts(
        tmp_path,
        stage="structured_sft",
    )
    output = tmp_path / "sft-plan.yaml"
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-real",),
    )
    module = _load_script("train_vlm_sft.py")

    result = module.main(
        [
            "--config",
            str(REPOSITORY_ROOT / "configs/vlm/qwen25vl3b_sft.yaml"),
            "--acknowledge-large-gpu-run",
            "--phase-d-audit",
            str(phase_d),
            "--phase-d-audit-sha256",
            phase_d_sha256,
            "--execution-audit",
            str(execution),
            "--execution-audit-sha256",
            execution_sha256,
            "--output-config",
            str(output),
        ]
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["stage"] == "structured_sft"
    assert payload["execution_status"] == "not_started"
    assert payload["requirements"]["execution_permitted"] is False
    assert all(key not in payload for key in ("verl", "model", "data", "trainer"))


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_plan_output_is_scoped_and_never_overwrites(
    script_name: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script(script_name)
    existing = tmp_path / "existing-plan.yaml"
    existing.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(OSError, match=r"exist"):
        module._write_new_plan(existing, "replacement\n")
    with pytest.raises(ValueError, match=r"artifacts"):
        module._write_new_plan(REPOSITORY_ROOT / "README.yaml", "not-allowed\n")
    monkeypatch.setattr(
        module.os,
        "link",
        lambda *_args, **_kwargs: pytest.fail(
            "a non-private repository plan path reached the write syscall"
        ),
    )
    with pytest.raises(ValueError, match=r"private_vlm_plans"):
        module._write_new_plan(
            REPOSITORY_ROOT / "artifacts/reports/not-private.yaml",
            "not-allowed\n",
        )

    assert existing.read_text(encoding="utf-8") == "operator-owned\n"


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_cli_rejects_duplicate_yaml_keys(
    script_name: str,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_script(script_name)
    config = tmp_path / "ambiguous.yaml"
    config.write_text("training: {}\ntraining: {}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.main(
            [
                "--config",
                str(config),
                "--acknowledge-large-gpu-run",
                "--phase-d-audit",
                str(tmp_path / "phase-d.json"),
                "--phase-d-audit-sha256",
                "a" * 64,
                "--execution-audit",
                str(tmp_path / "execution.json"),
                "--execution-audit-sha256",
                "b" * 64,
                "--output-config",
                str(tmp_path / f"{script_name}.yaml"),
            ]
        )

    assert "duplicate yaml key" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_cli_help_names_the_exact_private_plan_boundary(
    script_name: str,
    capsys,
) -> None:
    module = _load_script(script_name)

    with pytest.raises(SystemExit) as error:
        module.main(["--help"])

    output = capsys.readouterr().out.lower()
    assert error.value.code == 0
    assert "artifacts/logs/private_vlm_plans" in output
    assert "system temp" in output


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
@pytest.mark.parametrize("location", ("top", "nested"))
def test_vlm_cli_rejects_unknown_yaml_fields_before_gate_evidence(
    script_name: str,
    location: str,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_script(script_name)
    source = (
        REPOSITORY_ROOT
        / "configs/vlm"
        / (
            "qwen25vl3b_sft.yaml"
            if script_name == "train_vlm_sft.py"
            else "qwen25vl3b_joint_grpo.yaml"
        )
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if location == "top":
        payload["misspelled_execution_status"] = "not_started"
    else:
        payload["model"]["revison"] = payload["model"]["revision"]
    config = tmp_path / f"{script_name}-{location}.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemExit):
        module.main(
            [
                "--config",
                str(config),
                "--acknowledge-large-gpu-run",
                "--phase-d-audit",
                str(tmp_path / "phase-d.json"),
                "--phase-d-audit-sha256",
                "a" * 64,
                "--execution-audit",
                str(tmp_path / "execution.json"),
                "--execution-audit-sha256",
                "b" * 64,
                "--output-config",
                str(tmp_path / "plan.yaml"),
            ]
        )

    assert "unknown fields" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_yaml_loader_enforces_size_depth_and_node_limits(
    script_name: str,
    tmp_path: Path,
) -> None:
    module = _load_script(script_name)
    oversized = tmp_path / "oversized.yaml"
    oversized.write_bytes(b"x" * (1_048_576 + 1))
    with pytest.raises(ValueError, match=r"1048576|1.?MiB|limit"):
        module._load_unique_yaml(oversized)

    deep = tmp_path / "deep.yaml"
    deep.write_text(
        "\n".join(f"{'  ' * depth}child:" for depth in range(66)) + f"\n{'  ' * 66}value: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"depth|complex"):
        module._load_unique_yaml(deep)

    nodes = tmp_path / "nodes.yaml"
    nodes.write_text("items: [" + ",".join("0" for _ in range(100_001)) + "]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"node|complex"):
        module._load_unique_yaml(nodes)


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_cli_requires_a_private_output_file(
    script_name: str,
    capsys,
) -> None:
    module = _load_script(script_name)

    with pytest.raises(SystemExit):
        module.main(
            [
                "--config",
                str(
                    REPOSITORY_ROOT
                    / "configs/vlm"
                    / (
                        "qwen25vl3b_sft.yaml"
                        if script_name == "train_vlm_sft.py"
                        else "qwen25vl3b_joint_grpo.yaml"
                    )
                ),
                "--acknowledge-large-gpu-run",
                "--phase-d-audit-sha256",
                "a" * 64,
                "--execution-audit-sha256",
                "b" * 64,
            ]
        )

    error = capsys.readouterr().err.lower()
    assert "--output-config" in error
    assert "required" in error


def test_vlm_rl_cli_binds_the_yaml_outcome_only_reward_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    phase_d, phase_d_sha256, execution, execution_sha256, _snapshot = _write_vlm_gate_artifacts(
        tmp_path
    )
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/vlm/qwen25vl3b_joint_grpo.yaml").read_text(encoding="utf-8")
    )
    config["reward"]["perception_reward_weight"] = 0.1
    config_path = tmp_path / "unsafe-reward.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-real",),
    )
    module = _load_script("train_vlm_rl.py")

    with pytest.raises(SystemExit):
        module.main(
            [
                "--config",
                str(config_path),
                "--acknowledge-large-gpu-run",
                "--phase-d-audit",
                str(phase_d),
                "--phase-d-audit-sha256",
                phase_d_sha256,
                "--execution-audit",
                str(execution),
                "--execution-audit-sha256",
                execution_sha256,
                "--output-config",
                str(tmp_path / "unsafe-reward-plan.yaml"),
            ]
        )

    assert "reward" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("script_name", ["train_vlm_sft.py", "train_vlm_rl.py"])
def test_vlm_cli_cannot_self_attest_a_fake_cuda_device(
    script_name: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    phase_d, phase_d_sha256, execution, execution_sha256, _snapshot = _write_vlm_gate_artifacts(
        tmp_path
    )
    monkeypatch.setattr(
        "compbias.rl.verl_entrypoints.probe_local_cuda_devices",
        lambda: ("GPU-real",),
    )
    module = _load_script(script_name)

    with pytest.raises(SystemExit):
        module.main(
            [
                "--config",
                str(
                    REPOSITORY_ROOT
                    / "configs/vlm"
                    / (
                        "qwen25vl3b_sft.yaml"
                        if script_name == "train_vlm_sft.py"
                        else "qwen25vl3b_joint_grpo.yaml"
                    )
                ),
                "--acknowledge-large-gpu-run",
                "--cuda-device",
                "GPU-fake",
                "--phase-d-audit",
                str(phase_d),
                "--phase-d-audit-sha256",
                phase_d_sha256,
                "--execution-audit",
                str(execution),
                "--execution-audit-sha256",
                execution_sha256,
                "--output-config",
                str(tmp_path / f"{script_name}.yaml"),
            ]
        )

    error = capsys.readouterr().err.lower()
    assert "unrecognized" in error
    assert "--cuda-device" in error


def test_evaluator_requires_checkpoint_for_the_preregistered_config() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate_checkpoint.py"),
            "--config",
            str(REPOSITORY_ROOT / "configs" / "eval" / "full.yaml"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "--checkpoint" in completed.stderr
    assert "required" in completed.stderr.lower()
