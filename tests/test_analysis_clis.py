"""End-to-end contracts for the configuration-driven analysis CLIs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from compbias.envs.cva_world.generator import GeneratorConfig, generate_dataset
from compbias.envs.cva_world.schema import SemanticSplit
from compbias.eval.dataset_contract import (
    FROZEN_CVA_V2_GENERATOR_CONFIG,
    validate_frozen_cva_v2_dataset,
)
from compbias.eval.post_gpu_evidence import (
    PostGPUAuthenticationPending,
    validate_post_gpu_execution_audit,
    validate_ready_phase_d_audit,
)
from compbias.io.logging import publishable_config_snapshot
from compbias.io.manifests import build_dataset_manifest, canonical_json, manifest_sha256
from scripts.estimate_compensability import _artifact_sha256 as estimate_checkpoint_sha256
from scripts.estimate_compensability import _load_config as load_compensability_config
from scripts.estimate_compensability import _read_hashed_json as read_estimate_hashed_json
from scripts.estimate_compensability import _read_strict_jsonl
from scripts.estimate_compensability import _require_disjoint_paths as require_estimate_disjoint
from scripts.evaluate_checkpoint import _checkpoint_sha256 as evaluate_checkpoint_sha256
from scripts.evaluate_checkpoint import _load_config as load_evaluation_config
from scripts.evaluate_checkpoint import _read_hashed_json as read_evaluate_hashed_json
from scripts.evaluate_checkpoint import _read_jsonl
from scripts.evaluate_checkpoint import _require_disjoint_paths as require_evaluate_disjoint
from scripts.evaluate_checkpoint import _validate_protocol as validate_evaluation_protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts"


@pytest.mark.parametrize(
    ("reader", "kwargs"),
    [
        (_read_strict_jsonl, {}),
        (_read_jsonl, {"label": "IID"}),
    ],
    ids=("compensability", "checkpoint-evaluation"),
)
def test_large_analysis_jsonl_readers_stream_instead_of_copying_entire_files(
    reader, kwargs: dict[str, object], tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text('{"sample_id":"one"}\n{"sample_id":"two"}\n', encoding="utf-8")

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError("whole-file read_bytes is forbidden for bounded JSONL")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    records = reader(
        source,
        max_bytes=1_000_000,
        max_line_bytes=1_000,
        **kwargs,
    )

    assert [record["sample_id"] for record in records] == ["one", "two"]


@pytest.mark.parametrize(
    "digest",
    [estimate_checkpoint_sha256, evaluate_checkpoint_sha256],
    ids=("compensability", "checkpoint-evaluation"),
)
def test_checkpoint_tree_digest_rejects_symlinked_directories(digest, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.bin").write_bytes(b"unbound")
    (checkpoint / "linked-directory").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        digest(checkpoint)


@pytest.mark.parametrize("validator", [require_estimate_disjoint, require_evaluate_disjoint])
def test_analysis_outputs_cannot_overlap_hashed_input_trees(tmp_path: Path, validator) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match=r"disjoint.*checkpoint"):
        validator(
            inputs={"checkpoint": checkpoint},
            outputs={"metrics": checkpoint / "results" / "metrics.json"},
        )
    with pytest.raises(ValueError, match=r"disjoint.*checkpoint"):
        validator(
            inputs={"checkpoint": checkpoint / "weights"},
            outputs={"log_root": checkpoint},
        )


def test_publishable_config_snapshot_redacts_only_registered_paths_and_is_immutable(
    tmp_path: Path,
) -> None:
    original = {
        "inputs": {"predictions": str(tmp_path / "private.jsonl")},
        "seed": 7,
    }

    snapshot = publishable_config_snapshot(
        original,
        path_fields=(("inputs", "predictions"),),
        worktree=REPOSITORY_ROOT,
    )

    assert snapshot["inputs"]["predictions"] == "<external>/private.jsonl"
    assert original["inputs"]["predictions"] == str(tmp_path / "private.jsonl")


@pytest.mark.parametrize(
    ("reader", "kwargs"),
    [
        (_read_strict_jsonl, {}),
        (_read_jsonl, {"label": "IID"}),
    ],
    ids=("compensability", "checkpoint-evaluation"),
)
def test_analysis_jsonl_readers_normalize_deep_json_recursion_errors(
    reader, kwargs: dict[str, object], tmp_path: Path
) -> None:
    source = tmp_path / "deep.jsonl"
    source.write_text(
        '{"value":' + "[" * 10_000 + "0" + "]" * 10_000 + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid"):
        reader(source, max_bytes=1_000_000, max_line_bytes=100_000, **kwargs)


@pytest.mark.parametrize(
    "reader",
    [read_estimate_hashed_json, read_evaluate_hashed_json],
    ids=("compensability", "checkpoint-evaluation"),
)
def test_analysis_hashed_json_readers_normalize_deep_recursion_errors(
    reader, tmp_path: Path
) -> None:
    source = tmp_path / "deep.json"
    source.write_text(
        '{"value":' + "[" * 10_000 + "0" + "]" * 10_000 + "}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot parse"):
        reader(source, _sha256(source), "fixture")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(script: str, *arguments: object) -> subprocess.CompletedProcess[str]:
    rendered_arguments = [str(value) for value in arguments]
    if script in {"estimate_compensability.py", "evaluate_checkpoint.py"}:
        config_index = rendered_arguments.index("--config") + 1
        rendered_arguments.extend(
            ["--artifact-root", str(Path(rendered_arguments[config_index]).parent)]
        )
    source_path = str(REPOSITORY_ROOT / "src")
    inherited_python_path = os.environ.get("PYTHONPATH", "")
    python_path_entries = [entry for entry in inherited_python_path.split(os.pathsep) if entry]
    if source_path not in python_path_entries:
        python_path_entries.append(source_path)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path_entries)}
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *rendered_arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@lru_cache(maxsize=1)
def _frozen_cva_records() -> tuple[dict[str, object], ...]:
    samples = generate_dataset(
        GeneratorConfig(
            seed=20260814,
            samples_per_family_per_split=10,
            realizations_per_semantic=2,
            fully_cross_iid_visual_styles=True,
        )
    )
    return tuple(sample.to_mapping() for sample in samples)


def _frozen_generator_config() -> dict[str, object]:
    return json.loads(json.dumps(FROZEN_CVA_V2_GENERATOR_CONFIG))


def _write_manifest(path: Path) -> str:
    from compbias.envs.cva_world.renderer import (
        RenderConfig,
        build_contact_sheet,
        render_sample,
        sample_render_coordinates,
    )

    records = _frozen_cva_records()
    config = _frozen_generator_config()
    dataset = path.parent / "dataset.jsonl"
    dataset.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )
    images = path.parent / "images"
    images.mkdir()
    image_hashes: dict[str, str] = {}
    rendered_batch = []
    figure_dir = path.parent / "figures/cva_v2"
    figure_dir.mkdir(parents=True)
    contact_sheet_hashes: dict[str, str] = {}
    contact_sheets: list[str] = []
    for index, sample in enumerate(generate_dataset(GeneratorConfig(**config)), start=1):
        render_seed, realization_index = sample_render_coordinates(
            sample.sample_id, base_seed=20260814
        )
        image = render_sample(
            sample,
            RenderConfig(
                width=256,
                height=256,
                seed=render_seed,
                realization_index=realization_index,
            ),
        )
        image_path = images / f"{sample.sample_id}.png"
        image.save(image_path, format="PNG", optimize=True)
        image_hashes[sample.sample_id] = _sha256(image_path)
        rendered_batch.append((sample.sample_id, image))
        if len(rendered_batch) == 25 or index == len(records):
            sheet_number = len(contact_sheets) + 1
            name = f"cva_contact_sheet_{sheet_number:02d}.png"
            sheet_path = figure_dir / name
            sheet = build_contact_sheet(rendered_batch)
            sheet.save(sheet_path, format="PNG", optimize=True)
            sheet.close()
            for _sample_id, rendered in rendered_batch:
                rendered.close()
            rendered_batch.clear()
            contact_sheets.append(f"figures/cva_v2/{name}")
            contact_sheet_hashes[name] = _sha256(sheet_path)
    manifest = build_dataset_manifest(
        records,
        config=config,
        dataset_name="cva_v2",
        schema_version="2.0",
    )
    unsigned = {
        **manifest.to_mapping(),
        "generator_config": config,
        "render_config": {
            "height": 256,
            "samples_per_contact_sheet": 25,
            "width": 256,
        },
        "dataset_file_sha256": _sha256(dataset),
        "image_sha256": image_hashes,
        "jsonl_path": "dataset.jsonl",
        "images_dir": "images",
        "rendered_image_count": len(records),
        "solver_checks": len(records),
        "solver_pass_rate": 1.0,
        "roundtrip_checks": 4020,
        "roundtrip_pass_rate": 1.0,
        "contact_sheets": contact_sheets,
        "contact_sheet_sha256": contact_sheet_hashes,
        "preregistered_ood_factors": ["visual_style", "error_mechanism"],
    }
    payload = {**unsigned, "manifest_sha256": manifest_sha256(unsigned)}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest.content_sha256


def _phase_d_payload(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_set_hash = manifest_sha256(manifest["image_sha256"])
    sample_ids = manifest["sample_ids"]
    from collections import Counter

    from scripts.audit_dataset import (
        _answer_balance,
        _style_semantic_joint_independence,
        _visual_factor_realization_audit,
    )

    samples = generate_dataset(GeneratorConfig(**FROZEN_CVA_V2_GENERATOR_CONFIG))
    style_counts = Counter(sample.split_keys.visual_style for sample in samples)
    visual_audit = _visual_factor_realization_audit(
        samples,
        configured_styles=tuple(FROZEN_CVA_V2_GENERATOR_CONFIG["visual_styles"]),
        expected_style_counts=style_counts,
        style_counterbalance_violations=(),
        fully_cross_iid_visual_styles=True,
    )
    answer_balance = _answer_balance(
        samples,
        expected_samples=samples,
        samples_per_family_per_split=10,
    )
    joint_audit = _style_semantic_joint_independence(
        samples,
        fully_cross_iid_visual_styles=True,
    )
    return {
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
        "evidence_manifest_sha256": manifest["manifest_sha256"],
        "evidence_image_set_sha256": image_set_hash,
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
            "manifest_self_sha256": manifest["manifest_sha256"],
            "integrity_scope": "self-reported review record; no external signature verified",
        },
        "visual_factor_realization_audit": visual_audit,
        "ood_image_shift": {"complete": True, "checked_pair_count": 100, "violations": []},
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
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": _sha256(manifest_path),
            "manifest_self_sha256": manifest["manifest_sha256"],
            "content_sha256": manifest["content_sha256"],
            "image_set_sha256": image_set_hash,
        },
        "automatic_audit_clean": True,
        "phase_d_ready": True,
    }


def test_phase_d_validator_rejects_contradictory_nested_audit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample_ids = manifest_payload["sample_ids"]
    phase_d = _phase_d_payload(manifest)
    phase_d["ood_image_shift"] = {
        "complete": False,
        "checked_pair_count": 100,
        "violations": ["fixture contradiction"],
    }

    with pytest.raises(ValueError, match="ood_image_shift audit is incomplete"):
        validate_ready_phase_d_audit(
            phase_d,
            dataset_manifest_sha256=_sha256(manifest),
            dataset_manifest_self_sha256=manifest_payload["manifest_sha256"],
            dataset_content_sha256=manifest_payload["content_sha256"],
            dataset_image_set_sha256=manifest_sha256(manifest_payload["image_sha256"]),
            sample_ids=sample_ids,
        )
    phase_d = _phase_d_payload(manifest)
    phase_d["split_audit"]["self_signed_extra"] = True
    with pytest.raises(ValueError, match="split audit must match the closed schema"):
        validate_ready_phase_d_audit(
            phase_d,
            dataset_manifest_sha256=_sha256(manifest),
            dataset_manifest_self_sha256=manifest_payload["manifest_sha256"],
            dataset_content_sha256=manifest_payload["content_sha256"],
            dataset_image_set_sha256=manifest_sha256(manifest_payload["image_sha256"]),
            sample_ids=sample_ids,
        )

    phase_d = _phase_d_payload(manifest)
    first_group = next(iter(phase_d["answer_balance"]["groups"].values()))
    first_group["support"][0] = "self-signed-answer-drift"
    first_group["frequencies"][0]["answer"] = "self-signed-answer-drift"
    with pytest.raises(ValueError, match="differs from deterministic CVA-v2"):
        validate_ready_phase_d_audit(
            phase_d,
            dataset_manifest_sha256=_sha256(manifest),
            dataset_manifest_self_sha256=manifest_payload["manifest_sha256"],
            dataset_content_sha256=manifest_payload["content_sha256"],
            dataset_image_set_sha256=manifest_sha256(manifest_payload["image_sha256"]),
            sample_ids=sample_ids,
        )


def test_phase_d_validator_rejects_integer_substitution_for_ratio_float(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    phase_d = _phase_d_payload(manifest)
    group = phase_d["answer_balance"]["groups"]["bar_chart_aggregate/train"]
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

    with pytest.raises(ValueError, match=r"answer-balance|deterministic"):
        validate_ready_phase_d_audit(
            phase_d,
            dataset_manifest_sha256=_sha256(manifest),
            dataset_manifest_self_sha256=manifest_payload["manifest_sha256"],
            dataset_content_sha256=manifest_payload["content_sha256"],
            dataset_image_set_sha256=manifest_sha256(manifest_payload["image_sha256"]),
            sample_ids=manifest_payload["sample_ids"],
        )


def test_post_gpu_execution_validator_rejects_self_attested_clearance(
    tmp_path: Path,
) -> None:
    producer_config = tmp_path / "producer.yaml"
    producer_config.write_text("stage: fixture\n", encoding="utf-8")
    producer_records = tmp_path / "records.jsonl"
    producer_records.write_text('{"fixture":true}\n', encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "artifact_type": "execution_plan",
                "execution_permitted": False,
                "large_gpu_started": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    expected = {
        "checkpoint": "1" * 64,
        "manifest_file": "2" * 64,
        "manifest_self": "3" * 64,
        "content": "4" * 64,
        "phase_d": "5" * 64,
        "artifact_manifest": "6" * 64,
        "records": _sha256(producer_records),
        "config": _sha256(producer_config),
    }
    audit = {
        "schema_version": 3,
        "artifact_type": "fixture_execution_audit",
        "stage": "fixture_stage",
        "status": "completed",
        "gpu_execution_completed": True,
        "started_at": "2026-08-14T01:00:00Z",
        "ended_at": "2026-08-14T02:00:00Z",
        "command": ["python", "producer.py"],
        "gpu_uuids": ["GPU-fixture-a100"],
        "seeds": [11, 17, 23],
        "model_revision": "model-revision",
        "verl_revision": "verl-revision",
        "checkpoint_sha256": expected["checkpoint"],
        "dataset": {
            "manifest_file_sha256": expected["manifest_file"],
            "manifest_self_sha256": expected["manifest_self"],
            "content_sha256": expected["content"],
        },
        "phase_d": {
            "audit_sha256": expected["phase_d"],
            "schema_version": 2,
            "phase_d_ready": True,
            "human_signoff": True,
        },
        "preflight_plan": {
            "path": str(preflight),
            "sha256": _sha256(preflight),
            "artifact_type": "execution_plan",
            "execution_permitted": False,
            "large_gpu_started": False,
        },
        "runtime_clearance": {
            "passed": True,
            "network_disabled": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "container_image_digest": "sha256:" + "7" * 64,
            "wheelhouse_manifest_sha256": "8" * 64,
            "sbom_sha256": "9" * 64,
            "vulnerability_audit_sha256": "a" * 64,
        },
        "producer": {
            "config_path": str(producer_config),
            "config_sha256": expected["config"],
            "records_path": str(producer_records),
            "records_sha256": expected["records"],
            "record_count": 1,
            "manifest_sha256": expected["artifact_manifest"],
        },
        "state_injection_audit": {
            "passed": True,
            "image_hidden": True,
            "isolation_mode": "separate_text_only_worker",
            "adapter_sha256": "b" * 64,
            "reviewed_adapter_sha256": "b" * 64,
        },
    }

    with pytest.raises(PostGPUAuthenticationPending, match="authenticated post-GPU"):
        validate_post_gpu_execution_audit(
            audit,
            artifact_type="fixture_execution_audit",
            stage="fixture_stage",
            checkpoint_sha256=expected["checkpoint"],
            dataset_manifest_sha256=expected["manifest_file"],
            dataset_manifest_self_sha256=expected["manifest_self"],
            dataset_content_sha256=expected["content"],
            phase_d_audit_sha256=expected["phase_d"],
            prediction_or_rollout_manifest_sha256=expected["artifact_manifest"],
            producer_config_sha256=expected["config"],
            producer_records_path=producer_records,
            producer_records_sha256=expected["records"],
            producer_record_count=1,
            seeds=[11, 17, 23],
            model_revision="model-revision",
            verl_revision="verl-revision",
            sha256_file=_sha256,
        )


def _rollouts(
    *,
    checkpoint_sha256: str = "a" * 64,
    dataset_manifest_sha256: str = "c" * 64,
    dataset_content_sha256: str = "b" * 64,
    model_revision: str = "model-revision",
    state_adapter_sha256: str = "d" * 64,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    calibration = tuple(
        record
        for record in _frozen_cva_records()
        if record["split_keys"]["semantic_split"] == SemanticSplit.CALIBRATION.value
    )
    for record in calibration:
        catalog = tuple(record["error_catalog"])
        for error in catalog:
            for seed in range(1000, 1032):
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "error_id": error["error_id"],
                        "severity": error["severity"],
                        "base_probability": 1.0 / len(catalog),
                        "checkpoint": "checkpoint-7",
                        "checkpoint_sha256": checkpoint_sha256,
                        "model_revision": model_revision,
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "dataset_content_sha256": dataset_content_sha256,
                        "state_adapter_sha256": state_adapter_sha256,
                        "rollout_seed": seed,
                        "reward": float(error["error_id"] == "truth"),
                        "view": "interventional",
                        "image": None,
                    }
                )
    return tuple(rows)


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _compensability_config(tmp_path: Path, rollouts: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    dataset_content_sha256 = _write_manifest(manifest)
    dataset_manifest_sha256 = _sha256(manifest)
    checkpoint = tmp_path / "checkpoint.bin"
    execution_audit = tmp_path / "execution-audit.json"
    phase_d_audit = tmp_path / "phase-d-audit.json"
    rollout_manifest = tmp_path / "rollout-manifest.json"
    checkpoint_sha256: str | None = None
    execution_audit_sha256: str | None = None
    phase_d_audit_sha256: str | None = None
    rollout_manifest_sha256: str | None = None
    status = "not_started"
    if rollouts.is_file():
        checkpoint.write_bytes(b"fixed reasoner checkpoint fixture")
        checkpoint_sha256 = _sha256(checkpoint)
        producer_config = tmp_path / "compensability-producer.yaml"
        producer_config.write_text(
            yaml.safe_dump(
                {
                    "stage": "fixed_reasoner_compensability",
                    "rollout_seeds": list(range(1000, 1032)),
                    "image_access": "forbidden",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        records = _rollouts(
            checkpoint_sha256=checkpoint_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_content_sha256=dataset_content_sha256,
        )
        _write_jsonl(rollouts, records)
        calibration_records = tuple(
            record
            for record in _frozen_cva_records()
            if record["split_keys"]["semantic_split"] == SemanticSplit.CALIBRATION.value
        )
        calibration_sample_ids = sorted(str(record["sample_id"]) for record in calibration_records)
        error_ids_by_sample = {
            str(record["sample_id"]): sorted(
                str(error["error_id"]) for error in record["error_catalog"]
            )
            for record in calibration_records
        }
        rollout_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "fixed_reasoner_compensability_rollouts",
                    "rollouts_sha256": _sha256(rollouts),
                    "record_count": len(records),
                    "dataset_partition": "calibration",
                    "prediction_scope": "exact_partition",
                    "sample_ids": calibration_sample_ids,
                    "error_ids_by_sample": error_ids_by_sample,
                    "checkpoint_label": "checkpoint-7",
                    "rollout_seeds": list(range(1000, 1032)),
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_revision": "model-revision",
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "dataset_content_sha256": dataset_content_sha256,
                    "state_adapter_sha256": "d" * 64,
                    "producer_config_sha256": _sha256(producer_config),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rollout_manifest_sha256 = _sha256(rollout_manifest)
        phase_d_audit.write_text(
            json.dumps(_phase_d_payload(manifest), sort_keys=True), encoding="utf-8"
        )
        phase_d_audit_sha256 = _sha256(phase_d_audit)
        preflight = tmp_path / "compensability-preflight-plan.json"
        preflight.write_text(
            json.dumps(
                {
                    "artifact_type": "execution_plan",
                    "execution_permitted": False,
                    "large_gpu_started": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        execution_audit.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "artifact_type": "fixed_reasoner_compensability_execution_audit",
                    "stage": "fixed_reasoner_compensability",
                    "status": "completed",
                    "gpu_execution_completed": True,
                    "started_at": "2026-08-14T01:00:00Z",
                    "ended_at": "2026-08-14T02:00:00Z",
                    "command": ["python", "producer.py", "--config", "compensability.yaml"],
                    "gpu_uuids": ["GPU-fixture-a100"],
                    "seeds": list(range(1000, 1032)),
                    "model_revision": "model-revision",
                    "verl_revision": "verl-revision",
                    "checkpoint_sha256": checkpoint_sha256,
                    "dataset": {
                        "manifest_file_sha256": dataset_manifest_sha256,
                        "manifest_self_sha256": json.loads(manifest.read_text())["manifest_sha256"],
                        "content_sha256": dataset_content_sha256,
                    },
                    "phase_d": {
                        "audit_sha256": phase_d_audit_sha256,
                        "schema_version": 2,
                        "phase_d_ready": True,
                        "human_signoff": True,
                    },
                    "preflight_plan": {
                        "path": str(preflight),
                        "sha256": _sha256(preflight),
                        "artifact_type": "execution_plan",
                        "execution_permitted": False,
                        "large_gpu_started": False,
                    },
                    "runtime_clearance": {
                        "passed": True,
                        "network_disabled": True,
                        "local_files_only": True,
                        "trust_remote_code": False,
                        "use_safetensors": True,
                        "container_image_digest": "sha256:" + "1" * 64,
                        "wheelhouse_manifest_sha256": "2" * 64,
                        "sbom_sha256": "3" * 64,
                        "vulnerability_audit_sha256": "4" * 64,
                    },
                    "producer": {
                        "config_path": str(producer_config),
                        "config_sha256": _sha256(producer_config),
                        "records_path": str(rollouts),
                        "records_sha256": _sha256(rollouts),
                        "record_count": len(records),
                        "manifest_sha256": rollout_manifest_sha256,
                    },
                    "state_injection_audit": {
                        "passed": True,
                        "image_hidden": True,
                        "isolation_mode": "separate_text_only_worker",
                        "adapter_sha256": "d" * 64,
                        "reviewed_adapter_sha256": "d" * 64,
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        execution_audit_sha256 = _sha256(execution_audit)
        status = "recorded_gpu_artifacts"
    config = {
        "schema_version": 1,
        "experiment": "compensability_fixture",
        "execution_status": status,
        "model": {"name": "fixture/model", "revision": "model-revision"},
        "checkpoint_role": "pre_rl_fixed_reasoner",
        "input": {
            "split": "calibration",
            "view": "interventional",
            "image_access": "forbidden",
            "require_image_is_none": True,
            "error_catalog": "exhaustive_per_sample",
            "dataset_manifest": str(manifest),
            "max_jsonl_bytes": 64_000_000,
            "max_jsonl_line_bytes": 100_000,
        },
        "sampling": {"rollout_seeds": list(range(1000, 1032))},
        "statistics": {
            "confidence": 0.95,
            "interval": "wilson",
            "covariance_scope": "per_prompt",
            "pooled_covariance_is_primary": False,
        },
        "provenance": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "execution_audit": str(execution_audit),
            "execution_audit_sha256": execution_audit_sha256,
            "phase_d_audit": str(phase_d_audit),
            "phase_d_audit_sha256": phase_d_audit_sha256,
            "rollout_manifest": str(rollout_manifest),
            "rollout_manifest_sha256": rollout_manifest_sha256,
            "verl_revision": "verl-revision",
        },
        "outputs": {
            "rollouts": str(rollouts),
            "long_table": str(tmp_path / "long.csv"),
            "prompt_covariances": str(tmp_path / "covariances.csv"),
            "log_root": str(tmp_path / "logs"),
        },
    }
    path = tmp_path / "compensability.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def _rebind_compensability_sources(config: Path, *, update_record_count: bool = True) -> None:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    rollouts = Path(payload["outputs"]["rollouts"])
    rollout_manifest = Path(payload["provenance"]["rollout_manifest"])
    manifest_payload = json.loads(rollout_manifest.read_text(encoding="utf-8"))
    manifest_payload["rollouts_sha256"] = _sha256(rollouts)
    if update_record_count:
        manifest_payload["record_count"] = len(rollouts.read_text().splitlines())
    rollout_manifest.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    payload["provenance"]["rollout_manifest_sha256"] = _sha256(rollout_manifest)
    execution_path = Path(payload["provenance"]["execution_audit"])
    execution_payload = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_payload["producer"]["records_sha256"] = _sha256(rollouts)
    execution_payload["producer"]["record_count"] = len(rollouts.read_text().splitlines())
    execution_payload["producer"]["manifest_sha256"] = _sha256(rollout_manifest)
    execution_path.write_text(json.dumps(execution_payload, sort_keys=True), encoding="utf-8")
    payload["provenance"]["execution_audit_sha256"] = _sha256(execution_path)
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_compensability_config_blocks_cleanly_when_gpu_rollouts_are_missing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-rollouts.jsonl"
    config = _compensability_config(tmp_path, missing)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "blocked" in completed.stderr.lower()
    assert "rollout" in completed.stderr.lower()
    assert not (tmp_path / "long.csv").exists()
    assert not (tmp_path / "covariances.csv").exists()
    assert not (tmp_path / "logs").exists()


def test_compensability_valid_recorded_inputs_remain_blocked_without_authenticated_gate(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "blocked" in completed.stderr.lower()
    assert "authenticated post-gpu gate extension" in completed.stderr.lower()
    assert not (tmp_path / "long.csv").exists()
    assert not (tmp_path / "covariances.csv").exists()
    assert not (tmp_path / "logs").exists()


def test_compensability_rejects_private_strings_in_logged_config(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["provenance"]["verl_revision"] = "/Users/alice/private-revision"
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert (
        "machine-specific absolute path" in completed.stderr.lower()
        or "verl_revision does not match" in completed.stderr.lower()
    )
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward", 0.5, "binary"),
        ("base_probability", None, "base_probability"),
        ("sample_id", '=HYPERLINK("https://invalid")', "spreadsheet formula"),
    ],
)
def test_compensability_rejects_invalid_long_table_inputs_without_writing(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    if value is None:
        records[0].pop(field)
    else:
        records[0][field] = value
    _write_jsonl(rollouts, tuple(records))
    _rebind_compensability_sources(config)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert message in completed.stderr.lower()
    assert not (tmp_path / "long.csv").exists()
    assert not (tmp_path / "covariances.csv").exists()


@pytest.mark.parametrize(
    "malformed",
    [
        '{"sample_id":"a","sample_id":"b"}\n',
        '{"sample_id":"a","reward":NaN}\n',
        "\n",
    ],
)
def test_compensability_rejects_noncanonical_jsonl(malformed: str, tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    rollouts.write_text(malformed, encoding="utf-8")
    _rebind_compensability_sources(config)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert any(word in completed.stderr.lower() for word in ("duplicate", "constant", "blank"))
    assert not (tmp_path / "long.csv").exists()


def test_compensability_binds_rollouts_to_actual_checkpoint_and_complete_catalog(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"tampered checkpoint")

    mismatch = _run("estimate_compensability.py", "--config", config)

    assert mismatch.returncode != 0
    assert "checkpoint" in mismatch.stderr.lower()
    assert "sha-256" in mismatch.stderr.lower()

    checkpoint.write_bytes(b"fixed reasoner checkpoint fixture")
    records = [
        json.loads(line)
        for line in rollouts.read_text(encoding="utf-8").splitlines()
        if '"error_id": "truth"' in line
    ]
    _write_jsonl(rollouts, tuple(records))
    _rebind_compensability_sources(config)

    incomplete = _run("estimate_compensability.py", "--config", config)

    assert incomplete.returncode != 0
    assert "error catalog" in incomplete.stderr.lower()
    assert not (tmp_path / "long.csv").exists()


def test_compensability_rejects_duplicate_sample_error_checkpoint_seed_rows(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    records.append(dict(records[0]))
    _write_jsonl(rollouts, tuple(records))
    _rebind_compensability_sources(config)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "duplicate" in completed.stderr.lower()
    assert "sample/error/checkpoint" in completed.stderr.lower()
    assert not (tmp_path / "long.csv").exists()


def test_compensability_binds_safe_unique_checkpoint_label(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    records[0]["checkpoint"] = "alice@example.com"
    _write_jsonl(rollouts, tuple(records))
    _rebind_compensability_sources(config)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "checkpoint label" in completed.stderr.lower()
    assert not (tmp_path / "long.csv").exists()


def test_compensability_rejects_open_record_schema_and_nonfinite_json_numbers(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    records[0]["api_key"] = "secret-that-must-not-be-logged"
    _write_jsonl(rollouts, tuple(records))
    _rebind_compensability_sources(config)

    extra_field = _run("estimate_compensability.py", "--config", config)

    assert extra_field.returncode != 0
    assert any(
        message in extra_field.stderr.lower() for message in ("unknown fields", "sensitive field")
    )
    assert not (tmp_path / "logs").exists()

    line = rollouts.read_text(encoding="utf-8").splitlines()[1]
    parsed = json.loads(line)
    parsed["severity"] = 1.0
    encoded = json.dumps(parsed).replace('"severity": 1.0', '"severity": 1e999')
    rollouts.write_text(encoded + "\n", encoding="utf-8")
    _rebind_compensability_sources(config)

    nonfinite = _run("estimate_compensability.py", "--config", config)

    assert nonfinite.returncode != 0
    assert "non-finite" in nonfinite.stderr.lower()
    assert not (tmp_path / "logs").exists()


def test_compensability_rejects_output_parent_symlink_even_inside_artifact_root(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["outputs"]["long_table"] = str(linked_output / "long.csv")
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "symbolic-link" in completed.stderr.lower()
    assert not (real_output / "full.json").exists()
    assert not (real_output / "long.csv").exists()


@pytest.mark.parametrize(
    ("script", "config_builder"),
    [
        ("estimate_compensability.py", "compensability"),
        ("evaluate_checkpoint.py", "evaluation"),
    ],
)
def test_analysis_configs_reject_duplicate_yaml_keys(
    tmp_path: Path,
    script: str,
    config_builder: str,
) -> None:
    if config_builder == "compensability":
        config = tmp_path / "duplicate.yaml"
        config.write_text(
            "schema_version: 1\nschema_version: 1\nexperiment: duplicate\n",
            encoding="utf-8",
        )
        arguments: tuple[object, ...] = ("--config", config)
    else:
        checkpoint = tmp_path / "checkpoint.bin"
        checkpoint.write_bytes(b"checkpoint fixture")
        config = tmp_path / "duplicate.yaml"
        config.write_text(
            "schema_version: 1\nschema_version: 1\nexperiment: duplicate\n",
            encoding="utf-8",
        )
        arguments = ("--config", config, "--checkpoint", checkpoint)

    completed = _run(script, *arguments)

    assert completed.returncode != 0
    assert "duplicate yaml" in completed.stderr.lower()


@pytest.mark.parametrize(
    "script",
    ["estimate_compensability.py", "evaluate_checkpoint.py"],
)
def test_analysis_configs_reject_oversized_yaml_before_parsing(script: str, tmp_path: Path) -> None:
    config = tmp_path / "oversized.yaml"
    config.write_bytes(b"#" * (1024 * 1024 + 1))
    arguments: tuple[object, ...] = ("--config", config)
    if script == "evaluate_checkpoint.py":
        arguments += ("--checkpoint", tmp_path / "missing-checkpoint")

    completed = _run(script, *arguments)

    assert completed.returncode != 0
    assert "configuration exceeds 1 mib" in completed.stderr.lower()


@pytest.mark.parametrize(
    "loader",
    [load_compensability_config, load_evaluation_config],
)
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("root: &shared []\ncopy: *shared\n", "aliases/cycles"),
        (
            "root:\n"
            + "".join(f"{'  ' * depth}child:\n" for depth in range(1, 70))
            + f"{'  ' * 70}value: 1\n",
            "depth/node",
        ),
    ],
)
def test_analysis_configs_reject_yaml_aliases_and_excessive_depth(
    loader,
    payload: str,
    message: str,
    tmp_path: Path,
) -> None:
    config = tmp_path / "unsafe.yaml"
    config.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        loader(config)


@pytest.mark.parametrize(
    ("module_name", "loader"),
    [
        ("scripts.estimate_compensability", load_compensability_config),
        ("scripts.evaluate_checkpoint", load_evaluation_config),
    ],
)
def test_analysis_configs_normalize_yaml_recursion_errors(
    module_name: str,
    loader,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")

    def _raise_recursion(*_args, **_kwargs):
        raise RecursionError("fixture recursion")

    module = sys.modules[module_name]
    monkeypatch.setattr(module.yaml, "load", _raise_recursion)

    with pytest.raises(ValueError, match=r"cannot read configuration.*fixture recursion"):
        loader(config)


def _iid_ood_records(
    *,
    checkpoint_sha256: str = "a" * 64,
    dataset_manifest_sha256: str = "c" * 64,
    dataset_content_sha256: str = "b" * 64,
    image_hashes: dict[str, str] | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    iid: list[dict[str, object]] = []
    ood: list[dict[str, object]] = []
    evaluation_records = tuple(
        record
        for record in _frozen_cva_records()
        if record["split_keys"]["semantic_split"] == SemanticSplit.IID_TEST.value
    )
    from compbias.envs.cva_world.schema import CVASample

    all_records = tuple(CVASample.from_mapping(record) for record in _frozen_cva_records())
    iid_samples_by_id = {
        sample.sample_id: sample
        for sample in all_records
        if sample.split_keys.semantic_split is SemanticSplit.IID_TEST
    }
    ood_samples = tuple(
        sample
        for sample in all_records
        if sample.split_keys.semantic_split is SemanticSplit.OOD_TEST
    )
    iid_samples = tuple(iid_samples_by_id[str(sample.source_id)] for sample in ood_samples)
    evaluation_records = tuple(sample.to_mapping() for sample in iid_samples)
    ood_by_id = {str(sample.source_id): sample.to_mapping() for sample in ood_samples}
    for sample_index, dataset_record in enumerate(evaluation_records):
        iid_error = next(
            error for error in dataset_record["error_catalog"] if error["error_id"] != "truth"
        )
        ood_record = ood_by_id[str(dataset_record["sample_id"])]
        ood_error = next(
            error for error in ood_record["error_catalog"] if error["error_id"] != "truth"
        )
        from compbias.envs.cva_world.corruptions import apply_error
        from compbias.envs.cva_world.schema import ErrorSpec

        canonical = dataset_record["canonical_answer"]
        numeric = isinstance(canonical, (int, float)) and not isinstance(canonical, bool)
        iid_profile = [
            {
                "error_id": error["error_id"],
                "severity": error["severity"],
                "base_probability": 1.0 / len(dataset_record["error_catalog"]),
                "rollout_rewards": [
                    1 if error["error_id"] == "truth" or offset % 5 else 0 for offset in range(32)
                ],
            }
            for error in dataset_record["error_catalog"]
        ]
        ood_profile = [
            {
                "error_id": error["error_id"],
                "severity": error["severity"],
                "base_probability": 1.0 / len(ood_record["error_catalog"]),
                "rollout_rewards": [
                    1 if error["error_id"] == "truth" or offset % 4 else 0 for offset in range(32)
                ],
            }
            for error in ood_record["error_catalog"]
        ]
        for seed in (11, 17, 23):
            iid_correct = (sample_index + seed) % 5 != 0
            ood_correct = (sample_index + seed) % 5 not in {0, 1}
            invariant = {
                "sample_id": dataset_record["sample_id"],
                "paired_sample_id": ood_record["sample_id"],
                "image_path": dataset_record["image_path"],
                "image_sha256": (image_hashes or {}).get(
                    str(dataset_record["sample_id"]), "f" * 64
                ),
                "scene": dataset_record["scene"],
                "checkpoint_sha256": checkpoint_sha256,
                "model_revision": "model-revision",
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "dataset_content_sha256": dataset_content_sha256,
                "decoder_revision": "decoder-v1",
                "seed": seed,
                "canonical_answer": canonical,
                "numeric_target": float(canonical) if numeric else None,
            }
            iid_prediction = (
                canonical if iid_correct else (float(canonical) + 1.0 if numeric else "incorrect")
            )
            ood_prediction = (
                canonical if ood_correct else (float(canonical) + 1.0 if numeric else "incorrect")
            )
            iid_spec = ErrorSpec.from_mapping(iid_error)
            ood_spec = ErrorSpec.from_mapping(ood_error)
            iid.append(
                {
                    **invariant,
                    "error_mechanism": iid_spec.error_id,
                    "error_family": iid_spec.family,
                    "severity": iid_spec.severity,
                    "perceived_scene": dict(apply_error(dataset_record["scene"], iid_spec)),
                    "predicted_answer": iid_prediction,
                    "numeric_prediction": float(iid_prediction) if numeric else None,
                    "answer_correct": iid_correct,
                    "counterfactual_consistent": iid_correct,
                    "prompt_error_profile": iid_profile,
                    "scaling_probe": {
                        "multiplier": 1.5,
                        "multiplier_derivative": 0.3,
                    },
                    "selection_probe": {
                        "reference_probabilities": [0.4, 0.6],
                        "selected_probabilities": [0.5, 0.5],
                        "rewards": [0.0, 1.0],
                        "beta": 1.0,
                    },
                }
            )
            ood.append(
                {
                    **invariant,
                    "sample_id": ood_record["sample_id"],
                    "paired_sample_id": dataset_record["sample_id"],
                    "image_path": ood_record["image_path"],
                    "image_sha256": (image_hashes or {}).get(
                        str(ood_record["sample_id"]), "e" * 64
                    ),
                    "error_mechanism": ood_spec.error_id,
                    "error_family": ood_spec.family,
                    "severity": ood_spec.severity,
                    "perceived_scene": dict(apply_error(dataset_record["scene"], ood_spec)),
                    "predicted_answer": ood_prediction,
                    "numeric_prediction": float(ood_prediction) if numeric else None,
                    "answer_correct": ood_correct,
                    "counterfactual_consistent": ood_correct,
                    "prompt_error_profile": ood_profile,
                    "scaling_probe": {
                        "multiplier": 1.5,
                        "multiplier_derivative": 0.3,
                    },
                    "selection_probe": {
                        "reference_probabilities": [0.4, 0.6],
                        "selected_probabilities": [0.5, 0.5],
                        "rewards": [0.0, 1.0],
                        "beta": 1.0,
                    },
                }
            )
    return tuple(iid), tuple(ood)


def _evaluation_config(
    tmp_path: Path, iid: Path, ood: Path, checkpoint: Path | None = None
) -> Path:
    manifest = tmp_path / "manifest.json"
    dataset_content_sha256 = _write_manifest(manifest)
    dataset_manifest_sha256 = _sha256(manifest)
    prediction_manifest = tmp_path / "prediction-manifest.json"
    execution_audit = tmp_path / "evaluation-execution-audit.json"
    phase_d_audit = tmp_path / "phase-d-audit.json"
    prediction_manifest_sha256: str | None = None
    execution_audit_sha256: str | None = None
    phase_d_audit_sha256: str | None = None
    status = "not_started"
    if checkpoint is not None and checkpoint.is_file() and iid.is_file() and ood.is_file():
        producer_config = tmp_path / "prediction-producer.yaml"
        producer_config.write_text(
            yaml.safe_dump(
                {
                    "stage": "full_checkpoint_evaluation",
                    "seeds": [11, 17, 23],
                    "network_disabled": True,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        checkpoint_sha256 = _sha256(checkpoint)
        iid_records, ood_records = _iid_ood_records(
            checkpoint_sha256=checkpoint_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_content_sha256=dataset_content_sha256,
            image_hashes=json.loads(manifest.read_text(encoding="utf-8"))["image_sha256"],
        )
        _write_jsonl(iid, iid_records)
        _write_jsonl(ood, ood_records)
        ood_sample_ids = sorted(
            str(record["sample_id"])
            for record in _frozen_cva_records()
            if record["split_keys"]["semantic_split"] == SemanticSplit.OOD_TEST.value
        )
        evaluation_sample_ids = sorted(
            str(record["source_id"])
            for record in _frozen_cva_records()
            if record["split_keys"]["semantic_split"] == SemanticSplit.OOD_TEST.value
        )
        prediction_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_type": "paired_checkpoint_predictions",
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_revision": "model-revision",
                    "dataset_manifest_sha256": dataset_manifest_sha256,
                    "dataset_content_sha256": dataset_content_sha256,
                    "decoder_revision": "decoder-v1",
                    "vlm_seeds": [11, 17, 23],
                    "dataset_partition": "paired_iid_ood_test",
                    "prediction_scope": "exact_100_source_pairs",
                    "iid_sample_ids": evaluation_sample_ids,
                    "ood_sample_ids": ood_sample_ids,
                    "primary_shift": "error_mechanism",
                    "producer_config_sha256": _sha256(producer_config),
                    "iid": {"sha256": _sha256(iid), "record_count": len(iid_records)},
                    "ood": {"sha256": _sha256(ood), "record_count": len(ood_records)},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        prediction_manifest_sha256 = _sha256(prediction_manifest)
        phase_d_audit.write_text(
            json.dumps(_phase_d_payload(manifest), sort_keys=True), encoding="utf-8"
        )
        phase_d_audit_sha256 = _sha256(phase_d_audit)
        status = "recorded_gpu_artifacts"
    config = {
        "schema_version": 1,
        "experiment": "full_checkpoint_evaluation_fixture",
        "execution_status": status,
        "selection": {
            "calibration_split_only": True,
            "test_set_checkpoint_selection": "forbidden",
        },
        "inputs": {
            "iid_predictions": str(iid),
            "ood_predictions": str(ood),
            "dataset_manifest": str(manifest),
            "prediction_manifest": str(prediction_manifest),
            "prediction_manifest_sha256": prediction_manifest_sha256,
            "execution_audit": str(execution_audit),
            "execution_audit_sha256": execution_audit_sha256,
            "phase_d_audit": str(phase_d_audit),
            "phase_d_audit_sha256": phase_d_audit_sha256,
            "max_jsonl_bytes": 8_000_000,
            "max_jsonl_line_bytes": 100_000,
        },
        "metrics": {
            "outcome": ["exact_answer_accuracy", "numeric_mae", "numeric_mse"],
            "perception": ["state_exact_match", "mean_severity", "error_family_frequency"],
            "reasoning": [
                "oracle_state_accuracy",
                "perceived_state_canonicality",
                "compensator_mode_frequency",
            ],
            "compensability": [
                "per_prompt_compensability",
                "severity_compensability_covariance",
                "relative_compensability_gain",
                "pairwise_odds_residual",
            ],
            "coupling": [
                "perception_loss",
                "reasoning_loss",
                "coupling",
                "outcome_loss",
                "normalized_cancellation",
            ],
            "ood": [
                "error_mechanism_generalization_gap",
                "compensation_generalization_gap",
            ],
        },
        "statistics": {
            "vlm_seeds": [11, 17, 23],
            "bootstrap_resamples": 10000,
            "confidence": 0.95,
            "paired_bootstrap": True,
            "multiple_comparisons": "holm",
        },
        "shifts": {
            "primary": "error_mechanism",
            "held_constant": ["task_rule", "semantic_state", "answer_distribution"],
        },
        "provenance": {
            "model_revision": "model-revision",
            "verl_revision": "verl-revision",
            "decoder_revision": "decoder-v1",
        },
        "outputs": {
            "metrics": str(tmp_path / "full.json"),
            "report": str(tmp_path / "full.md"),
            "log_root": str(tmp_path / "logs"),
        },
    }
    path = tmp_path / "full.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    if status == "recorded_gpu_artifacts":
        preflight = tmp_path / "preflight-plan.json"
        preflight.write_text(
            json.dumps(
                {
                    "artifact_type": "execution_plan",
                    "execution_permitted": False,
                    "large_gpu_started": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        execution_audit.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "artifact_type": "checkpoint_evaluation_execution_audit",
                    "stage": "full_checkpoint_evaluation",
                    "status": "completed",
                    "gpu_execution_completed": True,
                    "started_at": "2026-08-14T01:00:00Z",
                    "ended_at": "2026-08-14T02:00:00Z",
                    "command": ["python", "producer.py", "--config", "full.yaml"],
                    "gpu_uuids": ["GPU-fixture-a100"],
                    "seeds": [11, 17, 23],
                    "model_revision": "model-revision",
                    "verl_revision": "verl-revision",
                    "checkpoint_sha256": _sha256(checkpoint),
                    "dataset": {
                        "manifest_file_sha256": dataset_manifest_sha256,
                        "manifest_self_sha256": json.loads(manifest.read_text())["manifest_sha256"],
                        "content_sha256": dataset_content_sha256,
                    },
                    "phase_d": {
                        "audit_sha256": phase_d_audit_sha256,
                        "schema_version": 2,
                        "phase_d_ready": True,
                        "human_signoff": True,
                    },
                    "preflight_plan": {
                        "path": str(preflight),
                        "sha256": _sha256(preflight),
                        "artifact_type": "execution_plan",
                        "execution_permitted": False,
                        "large_gpu_started": False,
                    },
                    "runtime_clearance": {
                        "passed": True,
                        "network_disabled": True,
                        "local_files_only": True,
                        "trust_remote_code": False,
                        "use_safetensors": True,
                        "container_image_digest": "sha256:" + "1" * 64,
                        "wheelhouse_manifest_sha256": "2" * 64,
                        "sbom_sha256": "3" * 64,
                        "vulnerability_audit_sha256": "4" * 64,
                    },
                    "producer": {
                        "config_path": str(producer_config),
                        "config_sha256": _sha256(producer_config),
                        "records_path": str(iid),
                        "records_sha256": _sha256(iid),
                        "record_count": len(iid_records),
                        "manifest_sha256": prediction_manifest_sha256,
                    },
                    "state_injection_audit": {
                        "passed": True,
                        "image_hidden": True,
                        "isolation_mode": "separate_text_only_worker",
                        "adapter_sha256": "5" * 64,
                        "reviewed_adapter_sha256": "5" * 64,
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        config["inputs"]["execution_audit_sha256"] = _sha256(execution_audit)
        path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def _rebind_prediction_sources(config: Path) -> None:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    iid = Path(payload["inputs"]["iid_predictions"])
    ood = Path(payload["inputs"]["ood_predictions"])
    manifest = Path(payload["inputs"]["prediction_manifest"])
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["iid"] = {
        "sha256": _sha256(iid),
        "record_count": len(iid.read_text().splitlines()),
    }
    manifest_payload["ood"] = {
        "sha256": _sha256(ood),
        "record_count": len(ood.read_text().splitlines()),
    }
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")
    payload["inputs"]["prediction_manifest_sha256"] = _sha256(manifest)
    execution = Path(payload["inputs"]["execution_audit"])
    execution_payload = json.loads(execution.read_text(encoding="utf-8"))
    execution_payload["producer"]["records_sha256"] = _sha256(iid)
    execution_payload["producer"]["record_count"] = len(iid.read_text().splitlines())
    execution_payload["producer"]["manifest_sha256"] = _sha256(manifest)
    execution.write_text(json.dumps(execution_payload, sort_keys=True), encoding="utf-8")
    payload["inputs"]["execution_audit_sha256"] = _sha256(execution)
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_full_evaluation_blocks_before_writing_when_gpu_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    config = _evaluation_config(tmp_path, tmp_path / "missing-iid.jsonl", tmp_path / "ood")

    completed = _run(
        "evaluate_checkpoint.py",
        "--config",
        config,
        "--checkpoint",
        tmp_path / "missing-checkpoint",
    )

    assert completed.returncode != 0
    assert "blocked" in completed.stderr.lower()
    assert "checkpoint" in completed.stderr.lower()
    assert not (tmp_path / "full.json").exists()
    assert not (tmp_path / "full.md").exists()


def test_full_evaluation_valid_recorded_inputs_remain_blocked_without_authenticated_gate(
    tmp_path: Path,
) -> None:
    iid_path = tmp_path / "iid.jsonl"
    ood_path = tmp_path / "ood.jsonl"
    iid_path.touch()
    ood_path.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid_path, ood_path, checkpoint)

    completed = _run(
        "evaluate_checkpoint.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
    )

    assert completed.returncode != 0
    assert "blocked" in completed.stderr.lower()
    assert "authenticated post-gpu gate extension" in completed.stderr.lower()
    assert not (tmp_path / "full.json").exists()
    assert not (tmp_path / "full.md").exists()
    assert not (tmp_path / "logs").exists()


def test_full_evaluation_rejects_prediction_checkpoint_mismatch_and_open_schema(
    tmp_path: Path,
) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid, ood, checkpoint)
    checkpoint.write_bytes(b"different checkpoint")

    mismatch = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert mismatch.returncode != 0
    assert "checkpoint" in mismatch.stderr.lower()
    assert "prediction manifest" in mismatch.stderr.lower()
    assert not (tmp_path / "full.json").exists()

    checkpoint.write_bytes(b"checkpoint fixture")
    records = [json.loads(line) for line in iid.read_text().splitlines()]
    records[0]["password"] = "must-not-enter-run-logs"
    _write_jsonl(iid, tuple(records))
    _rebind_prediction_sources(config)

    extra_field = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert extra_field.returncode != 0
    assert any(
        message in extra_field.stderr.lower() for message in ("unknown fields", "sensitive field")
    )
    assert not (tmp_path / "logs").exists()


def test_full_evaluation_rejects_private_strings_in_logged_config(tmp_path: Path) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid, ood, checkpoint)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["provenance"]["verl_revision"] = "/Users/alice/private-revision"
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    completed = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert completed.returncode != 0
    assert (
        "machine-specific absolute path" in completed.stderr.lower()
        or "verl_revision does not match" in completed.stderr.lower()
    )
    assert not (tmp_path / "logs").exists()


def test_full_evaluation_requires_exact_sample_by_seed_cartesian_coverage(
    tmp_path: Path,
) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid, ood, checkpoint)
    iid_records = tuple(
        json.loads(line)
        for line in iid.read_text(encoding="utf-8").splitlines()
        if '"seed": 17' not in line
    )
    ood_records = tuple(
        json.loads(line)
        for line in ood.read_text(encoding="utf-8").splitlines()
        if '"seed": 17' not in line
    )
    _write_jsonl(iid, iid_records)
    _write_jsonl(ood, ood_records)
    _rebind_prediction_sources(config)

    completed = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert completed.returncode != 0
    assert "cartesian" in completed.stderr.lower()
    assert "seed" in completed.stderr.lower()
    assert not (tmp_path / "full.json").exists()


def test_full_evaluation_rejects_output_parent_symlink_inside_artifact_root(
    tmp_path: Path,
) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid, ood, checkpoint)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["outputs"]["metrics"] = str(linked_output / "full.json")
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    completed = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert completed.returncode != 0
    assert "symbolic-link" in completed.stderr.lower()


def test_compensability_rejects_minimal_manifest_and_calibration_subset(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    manifest_path = Path(payload["input"]["dataset_manifest"])
    rollout_manifest_path = Path(payload["provenance"]["rollout_manifest"])
    rollout_manifest = json.loads(rollout_manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "content_sha256": payload["provenance"]["checkpoint_sha256"],
                "sample_ids": rollout_manifest["sample_ids"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_hash = _sha256(manifest_path)
    rollout_manifest["dataset_manifest_sha256"] = manifest_hash
    rollout_manifest["dataset_content_sha256"] = payload["provenance"]["checkpoint_sha256"]
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["dataset_manifest_sha256"] = manifest_hash
        record["dataset_content_sha256"] = payload["provenance"]["checkpoint_sha256"]
    _write_jsonl(rollouts, tuple(records))
    rollout_manifest["rollouts_sha256"] = _sha256(rollouts)
    rollout_manifest_path.write_text(json.dumps(rollout_manifest, sort_keys=True), encoding="utf-8")
    payload["provenance"]["rollout_manifest_sha256"] = _sha256(rollout_manifest_path)
    execution_path = Path(payload["provenance"]["execution_audit"])
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["dataset_manifest_sha256"] = manifest_hash
    execution["dataset_content_sha256"] = payload["provenance"]["checkpoint_sha256"]
    execution["rollout_manifest_sha256"] = payload["provenance"]["rollout_manifest_sha256"]
    execution_path.write_text(json.dumps(execution, sort_keys=True), encoding="utf-8")
    payload["provenance"]["execution_audit_sha256"] = _sha256(execution_path)
    config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "frozen cva_v2" in completed.stderr.lower()
    assert not (tmp_path / "long.csv").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "dataset_name",
        "schema_version",
        "manifest_self_hash",
        "raw_dataset_hash",
        "canonical_content_hash",
        "image_inventory",
        "image_hash",
    ],
)
def test_frozen_cva_v2_validator_rejects_every_bound_entity_tamper(
    tmp_path: Path, tamper: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = tmp_path / "dataset.jsonl"
    images = tmp_path / "images"
    if tamper == "dataset_name":
        manifest["dataset_name"] = "lookalike"
    elif tamper == "schema_version":
        manifest["schema_version"] = "1.0"
    elif tamper == "manifest_self_hash":
        manifest["manifest_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="self SHA-256"):
            validate_frozen_cva_v2_dataset(manifest_path, artifact_root=tmp_path)
        return
    elif tamper == "raw_dataset_hash":
        dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif tamper == "canonical_content_hash":
        records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
        records[0]["scene"]["value"] += 1
        _write_jsonl(dataset, tuple(records))
        manifest["dataset_file_sha256"] = _sha256(dataset)
    elif tamper == "image_inventory":
        (images / f"{manifest['sample_ids'][0]}.png").unlink()
    elif tamper == "image_hash":
        image_path = images / f"{manifest['sample_ids'][0]}.png"
        image_path.write_bytes(image_path.read_bytes() + b"tamper")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = manifest_sha256(unsigned)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_frozen_cva_v2_dataset(manifest_path, artifact_root=tmp_path)


def test_compensability_requires_exact_calibration_partition_and_dataset_error_catalog(
    tmp_path: Path,
) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.touch()
    config = _compensability_config(tmp_path, rollouts)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    rollout_manifest_path = Path(payload["provenance"]["rollout_manifest"])
    rollout_manifest = json.loads(rollout_manifest_path.read_text(encoding="utf-8"))
    removed = rollout_manifest["sample_ids"].pop()
    rollout_manifest["error_ids_by_sample"].pop(removed)
    rollout_manifest_path.write_text(json.dumps(rollout_manifest, sort_keys=True), encoding="utf-8")
    records = tuple(
        json.loads(line)
        for line in rollouts.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["sample_id"] != removed
    )
    _write_jsonl(rollouts, records)
    _rebind_compensability_sources(config)

    completed = _run("estimate_compensability.py", "--config", config)

    assert completed.returncode != 0
    assert "exact calibration partition" in completed.stderr.lower()


def test_full_evaluation_rejects_unregistered_or_shrunken_evaluation_scope(
    tmp_path: Path,
) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid, ood, checkpoint)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    prediction_manifest_path = Path(payload["inputs"]["prediction_manifest"])
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    removed_iid = prediction_manifest["iid_sample_ids"].pop()
    removed_ood = prediction_manifest["ood_sample_ids"].pop()
    prediction_manifest_path.write_text(
        json.dumps(prediction_manifest, sort_keys=True), encoding="utf-8"
    )
    for path, removed in ((iid, removed_iid), (ood, removed_ood)):
        records = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["sample_id"] != removed
        )
        _write_jsonl(path, records)
    _rebind_prediction_sources(config)

    completed = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert completed.returncode != 0
    assert "exact" in completed.stderr.lower() and "source pairs" in completed.stderr.lower()
    assert not (tmp_path / "full.json").exists()


def test_full_evaluation_computes_seeded_paired_bootstrap_and_holm_statistics(
    tmp_path: Path,
) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    _evaluation_config(tmp_path, iid, ood, checkpoint)

    from compbias.eval.paired_inference import paired_ood_inference

    iid_records = tuple(json.loads(line) for line in iid.read_text().splitlines())
    ood_records = tuple(json.loads(line) for line in ood.read_text().splitlines())
    statistics = paired_ood_inference(
        iid_records,
        ood_records,
        seeds=(11, 17, 23),
        metric_names=(
            "error_mechanism_generalization_gap",
            "compensation_generalization_gap",
        ),
        confidence=0.95,
        n_resamples=10_000,
    )
    assert statistics["protocol"] == {
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260814,
        "confidence": 0.95,
        "multiple_comparisons": "holm",
        "paired_bootstrap": True,
        "resampling_unit": "sample_within_seed",
    }
    assert set(statistics["metrics"]) == {
        "error_mechanism_generalization_gap",
        "compensation_generalization_gap",
    }
    for metric in statistics["metrics"].values():
        assert set(metric["per_seed"]) == {"11", "17", "23"}
        assert metric["aggregate"]["n_pairs"] == 100 * 3
        assert 0.0 <= metric["aggregate"]["p_value"] <= 1.0
        assert 0.0 <= metric["aggregate"]["holm_adjusted_p_value"] <= 1.0
        assert metric["aggregate"]["ci_low"] <= metric["aggregate"]["estimate"]
        assert metric["aggregate"]["ci_high"] >= metric["aggregate"]["estimate"]
    assert not (tmp_path / "full.json").exists()


def test_full_evaluation_rejects_nonregistered_bootstrap_allocation_budget() -> None:
    config = yaml.safe_load((REPOSITORY_ROOT / "configs/eval/full.yaml").read_text())
    config["statistics"]["bootstrap_resamples"] = 100_001

    with pytest.raises(ValueError, match=r"bootstrap_resamples.*10000"):
        validate_evaluation_protocol(config)


def test_full_evaluation_does_not_publish_compensation_gap_without_authenticated_gate(
    tmp_path: Path,
) -> None:
    iid = tmp_path / "iid.jsonl"
    ood = tmp_path / "ood.jsonl"
    iid.touch()
    ood.touch()
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = _evaluation_config(tmp_path, iid, ood, checkpoint)
    for path in (iid, ood):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            for entry in record["prompt_error_profile"]:
                entry["rollout_rewards"] = [0] * 32
        _write_jsonl(path, tuple(records))
    _rebind_prediction_sources(config)

    completed = _run("evaluate_checkpoint.py", "--config", config, "--checkpoint", checkpoint)

    assert completed.returncode != 0
    assert "blocked" in completed.stderr.lower()
    assert not (tmp_path / "full.json").exists()


def test_paper_table_builder_parses_registry_and_nested_accepted_metrics(
    tmp_path: Path,
) -> None:
    accepted = tmp_path / "accepted.json"
    accepted.write_text(
        json.dumps(
            {
                "gates": {
                    "all_passed": True,
                    "thresholds": {"maximum_error": 1e-8},
                },
                "summary": {"seeds": 20, "endpoints": {"truthful": 10, "compensatory": 10}},
            }
        ),
        encoding="utf-8",
    )
    accepted_sha256 = _sha256(accepted)
    partial = tmp_path / "partial.json"
    partial.write_text('{"gates":{"human_signoff":false}}', encoding="utf-8")
    registry = tmp_path / "registry.md"
    verified_row = (
        "| A-TEST | A | Nested fixture? | CPU | VERIFIED_CPU | accepted "
        f"`accepted.json` SHA-256 `{accepted_sha256}` |"
    )
    registry.write_text(
        f"""\
# Registry

## Registered experiments

| ID | Phase | Question | Primary protocol | Current status | Evidence or blocker |
|---|---|---|---|---|---|
{verified_row}
| D-TEST | D | Partial fixture? | audit | PARTIAL_GATE | pending `partial.json` |

## Acceptance gates
""",
        encoding="utf-8",
    )
    output = tmp_path / "tables.md"

    completed = _run("build_paper_tables.py", "--registry", registry, "--output", output)

    assert completed.returncode == 0, completed.stderr
    report = output.read_text()
    assert "Experiment status" in report
    assert "A-TEST" in report and "VERIFIED_CPU" in report
    assert "D-TEST" in report and "PARTIAL_GATE" in report
    assert "gates.thresholds.maximum_error" in report
    assert "summary.endpoints.truthful" in report
    assert "gates.human_signoff" not in report
    assert _sha256(registry) in report
    assert _sha256(accepted) in report
    assert "Source hashes" in report

    repeated = _run("build_paper_tables.py", "--registry", registry, "--output", output)
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr.lower()


def test_paper_table_builder_rejects_malformed_registry_without_output(tmp_path: Path) -> None:
    registry = tmp_path / "registry.md"
    registry.write_text("# Registry\n\nNo registered table.\n", encoding="utf-8")
    output = tmp_path / "tables.md"

    completed = _run("build_paper_tables.py", "--registry", registry, "--output", output)

    assert completed.returncode != 0
    assert "registered experiments" in completed.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize("kind", ("fifo", "oversized"))
def test_paper_table_builder_rejects_nonregular_or_oversized_registry_without_output(
    tmp_path: Path, kind: str
) -> None:
    registry = tmp_path / "registry.md"
    if kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO test requires POSIX")
        os.mkfifo(registry)
    else:
        registry.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    output = tmp_path / "tables.md"

    completed = _run(
        "build_paper_tables.py",
        "--registry",
        registry,
        "--output",
        output,
    )

    assert completed.returncode != 0
    assert any(marker in completed.stderr.lower() for marker in ("regular file", "byte limit"))
    assert not output.exists()


def test_paper_table_builder_confines_output_and_rejects_symlink_parents(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.md"
    registry.write_text("# not parsed because output validation runs first\n", encoding="utf-8")
    repository_output = REPOSITORY_ROOT / "paper-tables-outside-artifacts.md"

    outside = _run(
        "build_paper_tables.py",
        "--registry",
        registry,
        "--output",
        repository_output,
    )

    assert outside.returncode != 0
    assert "repository artifacts directory" in outside.stderr.lower()
    assert not repository_output.exists()

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    symlinked = _run(
        "build_paper_tables.py",
        "--registry",
        registry,
        "--output",
        linked_parent / "tables.md",
    )

    assert symlinked.returncode != 0
    assert "symlink" in symlinked.stderr.lower()
    assert not (real_parent / "tables.md").exists()


@pytest.mark.parametrize("binding", (None, "0" * 64))
def test_paper_table_builder_requires_registered_metric_hash_binding(
    tmp_path: Path, binding: str | None
) -> None:
    accepted = tmp_path / "accepted.json"
    accepted.write_text('{"metric":1}\n', encoding="utf-8")
    evidence = "accepted `accepted.json`"
    if binding is not None:
        evidence += f" SHA-256 `{binding}`"
    registry = tmp_path / "registry.md"
    registry.write_text(
        f"""\
# Registry

## Registered experiments

| ID | Phase | Question | Primary protocol | Current status | Evidence or blocker |
|---|---|---|---|---|---|
| A-TEST | A | Bound fixture? | CPU | VERIFIED_CPU | {evidence} |

## Acceptance gates
""",
        encoding="utf-8",
    )
    output = tmp_path / "tables.md"

    completed = _run("build_paper_tables.py", "--registry", registry, "--output", output)

    assert completed.returncode != 0
    assert "sha-256" in completed.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize(
    "accepted_payload",
    (
        {"run": {"workspace": "/Users/alice/private-checkout"}},
        {"run": {"api_token": "not-a-real-token"}},
        {"operator": {"email": "alice@example.com"}},
    ),
)
def test_paper_table_builder_rejects_private_values_from_publishable_metrics(
    tmp_path: Path,
    accepted_payload: dict[str, object],
) -> None:
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps(accepted_payload), encoding="utf-8")
    registry = tmp_path / "registry.md"
    accepted_binding = f"accepted `accepted.json` SHA-256 `{_sha256(accepted)}`"
    registry.write_text(
        f"""\
# Registry

## Registered experiments

| ID | Phase | Question | Primary protocol | Current status | Evidence or blocker |
|---|---|---|---|---|---|
| A-TEST | A | Private fixture? | CPU | VERIFIED_CPU | {accepted_binding} |

## Acceptance gates
""",
        encoding="utf-8",
    )
    output = tmp_path / "tables.md"

    completed = _run("build_paper_tables.py", "--registry", registry, "--output", output)

    assert completed.returncode != 0
    assert "privacy" in completed.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize(
    ("column", "private_value"),
    (
        ("question", "Operator alice@example.com?"),
        ("protocol", "Run from /Users/alice/private-checkout"),
        ("protocol", "/Users/alice"),
        ("protocol", "/home/alice"),
        ("evidence", "token " + "sk" + "-fixture0123456789"),
        ("evidence", "Slack " + "xoxb" + "-123456789012-abcdefghijklmnop"),
        ("evidence", "Google " + "AIza" + "SyA1234567890abcdefghijklmnop"),
        ("evidence", "AWS " + "ASIA" + "1234567890ABCDEF"),
    ),
)
def test_paper_table_builder_rejects_private_registry_cells_before_output(
    tmp_path: Path, column: str, private_value: str
) -> None:
    cells = {
        "question": "Private fixture?",
        "protocol": "CPU",
        "evidence": "pending",
    }
    cells[column] = private_value
    registry = tmp_path / "registry.md"
    row = (
        f"| A-TEST | A | {cells['question']} | {cells['protocol']} | "
        f"IMPLEMENTED_NOT_RECORDED | {cells['evidence']} |"
    )
    registry.write_text(
        f"""\
# Registry

## Registered experiments

| ID | Phase | Question | Primary protocol | Current status | Evidence or blocker |
|---|---|---|---|---|---|
{row}

## Acceptance gates
""",
        encoding="utf-8",
    )
    output = tmp_path / "tables.md"

    completed = _run("build_paper_tables.py", "--registry", registry, "--output", output)

    assert completed.returncode != 0
    assert "privacy" in completed.stderr.lower()
    assert not output.exists()


def test_paper_table_builder_normalizes_deep_json_failure_without_traceback(
    tmp_path: Path,
) -> None:
    accepted = tmp_path / "accepted.json"
    accepted.write_text("[" * 2_000 + "0" + "]" * 2_000, encoding="utf-8")
    registry = tmp_path / "registry.md"
    accepted_binding = f"accepted `accepted.json` SHA-256 `{_sha256(accepted)}`"
    registry.write_text(
        f"""\
# Registry

## Registered experiments

| ID | Phase | Question | Primary protocol | Current status | Evidence or blocker |
|---|---|---|---|---|---|
| A-TEST | A | Deep fixture? | CPU | VERIFIED_CPU | {accepted_binding} |

## Acceptance gates
""",
        encoding="utf-8",
    )
    output = tmp_path / "tables.md"

    completed = _run("build_paper_tables.py", "--registry", registry, "--output", output)

    assert completed.returncode != 0
    assert any(marker in completed.stderr.lower() for marker in ("cannot parse", "nesting exceeds"))
    assert "traceback" not in completed.stderr.lower()
    assert not output.exists()
