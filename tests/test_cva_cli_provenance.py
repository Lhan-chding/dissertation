"""CLI provenance contracts for the generated CVA-World snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from compbias.envs.cva_world.renderer import (
    SUPPORTED_VISUAL_STYLES,
    VISUAL_STYLE_APPLICABILITY,
)
from compbias.io.manifests import manifest_sha256
from scripts import audit_dataset as audit_dataset_script
from scripts import generate_cva as generate_cva_script

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX")
def test_generator_rejects_fifo_config_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = tmp_path / "config.yaml"
    os.mkfifo(fifo)
    original = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path == fifo:
            pytest.fail("generator attempted to read a FIFO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(ValueError, match="regular file"):
        generate_cva_script._yaml_mapping(fifo)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX")
def test_audit_rejects_fifo_dataset_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = tmp_path / "dataset.jsonl"
    os.mkfifo(fifo)
    original = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path == fifo:
            pytest.fail("audit attempted to open a FIFO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="regular file"):
        audit_dataset_script._strict_jsonl(fifo)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX")
def test_audit_rejects_fifo_visual_review_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo = tmp_path / "review.json"
    os.mkfifo(fifo)
    original = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path == fifo:
            pytest.fail("audit attempted to read a FIFO review")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(ValueError, match="regular file"):
        audit_dataset_script._human_review_binding(
            fifo,
            manifest_sha256="a" * 64,
            image_set_sha256="b" * 64,
            sample_ids=("sample",),
            contact_sheets=("sheet.png",),
        )


def test_audit_rejects_oversized_prior_report_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "prior.json"
    report.write_bytes(b" " * (16 * 1024 * 1024 + 1))
    original = Path.read_text

    def guarded_read(path: Path, *args, **kwargs):
        if path == report:
            pytest.fail("audit read an oversized prior report without a byte cap")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(ValueError, match="byte limit"):
        audit_dataset_script._validate_prior_report(
            report,
            manifest_file_sha256="a" * 64,
        )


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha256_for_records(records: list[dict[str, object]]) -> str:
    return manifest_sha256(sorted(records, key=lambda item: str(item["sample_id"])))


def _small_config(tmp_path: Path) -> tuple[dict[str, object], Path]:
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/data/cva_v2.yaml").read_text(encoding="utf-8")
    )
    config["dataset"].update(
        {
            "name": "cva_test",
            "output": str(tmp_path / "output/dataset.jsonl"),
            "manifest": str(tmp_path / "output/manifest.json"),
            "samples_per_family_per_split": 2,
        }
    )
    config["rendering"].update(
        {
            "images_dir": str(tmp_path / "output/images"),
            "contact_sheet_dir": str(tmp_path / "figures"),
            "samples_per_contact_sheet": 25,
        }
    )
    config["logging"]["root"] = str(tmp_path / "logs")
    config["logging"]["experiment"] = "cva_test_generation"
    config_path = tmp_path / "cva.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, config_path


def _generate(tmp_path: Path, *, overwrite: bool = False) -> subprocess.CompletedProcess[str]:
    _config, config_path = _small_config(tmp_path)
    return _generate_from_config(tmp_path, config_path, overwrite=overwrite)


def _generate_from_config(
    tmp_path: Path, config_path: Path, *, overwrite: bool = False
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/generate_cva.py"),
        "--config",
        str(config_path),
        "--output-root",
        str(tmp_path / "output"),
        "--figure-root",
        str(tmp_path / "figures"),
        "--log-root",
        str(tmp_path / "logs"),
    ]
    if overwrite:
        command.append("--overwrite")
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _audit(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_dataset.py"),
            "--manifest",
            str(tmp_path / "output/manifest.json"),
            "--output",
            str(tmp_path / "audit.json"),
            "--report-root",
            str(tmp_path),
            "--log-root",
            str(tmp_path / "logs"),
            "--artifact-root",
            str(tmp_path),
            "--overwrite",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_generator_cli_requires_an_explicit_frozen_config(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/generate_cva.py"),
            "--output-root",
            str(tmp_path / "output"),
            "--figure-root",
            str(tmp_path / "figures"),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    assert "--config" in completed.stderr
    assert "required" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_generator_captures_provenance_before_creating_stage_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import compbias.io.logging as run_logging

    _config, config_path = _small_config(tmp_path)
    real_capture_environment = run_logging.capture_environment
    captures = 0

    def capture_before_staging(**kwargs):
        nonlocal captures
        assert not tuple(tmp_path.rglob(".cva-*-stage-*"))
        captures += 1
        environment = real_capture_environment(**kwargs)
        return {**environment, "git_dirty": False}

    monkeypatch.setattr(run_logging, "capture_environment", capture_before_staging)

    assert (
        generate_cva_script.main(
            [
                "--config",
                str(config_path),
                "--output-root",
                str(tmp_path / "output"),
                "--figure-root",
                str(tmp_path / "figures"),
                "--log-root",
                str(tmp_path / "logs"),
            ]
        )
        == 0
    )
    assert captures == 1


def test_audit_cli_requires_output_for_transactional_publication(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_dataset.py"),
            "--manifest",
            str(tmp_path / "missing-manifest.json"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    assert "--output" in completed.stderr
    assert "required" in completed.stderr


@pytest.mark.parametrize(
    ("section", "field", "invalid_value", "message"),
    [
        ("dataset", "seed", True, "seed must be an integer"),
        (
            "dataset",
            "samples_per_family_per_split",
            1.9,
            "samples_per_family_per_split must be an integer",
        ),
        (
            "dataset",
            "realizations_per_semantic",
            2.5,
            "realizations_per_semantic must be an integer",
        ),
        ("dataset", "visual_styles", ["baseline", 7], "visual_styles"),
        (
            "dataset",
            "preregistered_ood_factors",
            ["visual_style", 7],
            "preregistered_ood_factors",
        ),
        (
            "dataset",
            "train_error_mechanism",
            7,
            "train_error_mechanism must be a string",
        ),
        ("rendering", "width", True, "rendering.width must be an integer"),
        (
            "rendering",
            "samples_per_contact_sheet",
            25.5,
            "rendering.samples_per_contact_sheet must be an integer",
        ),
    ],
)
def test_generator_cli_rejects_values_that_would_require_type_coercion(
    tmp_path: Path,
    section: str,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    config, config_path = _small_config(tmp_path)
    config[section][field] = invalid_value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = _generate_from_config(tmp_path, config_path)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not (tmp_path / "output/dataset.jsonl").exists()


def test_generator_cli_rejects_combined_render_budget_before_writing(tmp_path: Path) -> None:
    config, config_path = _small_config(tmp_path)
    config["rendering"].update({"width": 4096, "height": 4096})
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = _generate_from_config(tmp_path, config_path)

    assert completed.returncode != 0
    assert "render pixel budget" in completed.stderr
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "figures").exists()


def _rewrite_manifest_with_valid_self_hash(path: Path, payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = manifest_sha256(unsigned)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_small_generation_and_audit_write_complete_publishable_run_bundles(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr

    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["generator_config"]
    assert set(manifest["image_sha256"]) == set(manifest["sample_ids"])
    assert manifest["contact_sheet_sha256"]
    assert manifest["manifest_sha256"] == manifest_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )

    audit_output = tmp_path / "audit.json"
    audited = _audit(tmp_path)
    assert audited.returncode == 0, audited.stderr
    report = json.loads(audit_output.read_text())
    assert report["audit_report_schema_version"] == 2
    assert report["manifest_content_sha256_matches"] is True
    assert report["manifest_config_sha256_matches"] is True
    assert report["manifest_sample_ids_match"] is True
    assert report["manifest_image_sha256_matches"] is True
    assert report["manifest_self_sha256_matches"] is True
    assert report["contact_sheet_sha256_matches"] is True
    assert report["style_counterbalance_violations"] == []
    assert report["automatic_audit_clean"] is True
    assert report["phase_d_ready"] is False
    assert report["deterministic_replay"] == {
        "complete": True,
        "generator_matches": True,
        "renderer_matches": True,
        "contact_sheets_match": True,
        "generator_mismatches": [],
        "renderer_mismatches": [],
        "contact_sheet_mismatches": [],
    }
    assert report["ood_image_shift"] == {
        "complete": True,
        "checked_pair_count": sum("_ood_test_" in value for value in manifest["sample_ids"]),
        "violations": [],
    }
    joint = report["style_semantic_joint_independence"]
    assert set(joint) == {"complete", "criterion", "groups", "violations"}
    assert joint["complete"] is True
    assert joint["criterion"] == "fully_crossed_style_by_semantic_state"
    assert len(joint["groups"]) == 20
    assert joint["violations"] == []
    for group in joint["groups"].values():
        assert set(group) == {
            "semantic_state_count",
            "expected_styles",
            "fully_crossed_state_count",
            "sample_count",
            "style_counts",
        }
        assert group["semantic_state_count"] == 2
        assert group["fully_crossed_state_count"] == 2
        assert group["sample_count"] == 2 * len(group["expected_styles"])
        assert set(group["style_counts"].values()) == {2}
    factor_audit = report["visual_factor_realization_audit"]
    assert factor_audit["complete"] is True
    assert factor_audit["catalog"] == list(SUPPORTED_VISUAL_STYLES)
    assert factor_audit["observed_styles"] == list(SUPPORTED_VISUAL_STYLES)
    assert factor_audit["applicability"] == {
        style: [family.value for family in VISUAL_STYLE_APPLICABILITY[style]]
        for style in SUPPORTED_VISUAL_STYLES
    }
    assert report["answer_balance"]["complete"] is True
    assert report["answer_balance"]["iid_ood_exact_match"] is True
    assert report["answer_balance"]["numeric_exact_balance"] is True
    assert report["answer_balance"]["relation_multiclass_coverage"] is True

    generation_runs = tuple((tmp_path / "logs/cva_test_generation").iterdir())
    assert len(generation_runs) == 1
    generation_environment = json.loads(
        (generation_runs[0] / "environment.json").read_text(encoding="utf-8")
    )
    generation_config = yaml.safe_load(
        (generation_runs[0] / "config.yaml").read_text(encoding="utf-8")
    )
    manifest_path = tmp_path / "output/manifest.json"
    assert generation_environment["dataset_manifest_hash"] == _sha256_path(manifest_path)
    assert generation_environment["checkpoint_hash"] is None
    assert generation_config["manifest_file_sha256"] == _sha256_path(manifest_path)
    assert generation_config["manifest_self_sha256"] == manifest["manifest_sha256"]
    assert generation_config["content_sha256"] == manifest["content_sha256"]
    assert generation_config["logging"] == {
        "experiment": "cva_test_generation",
        "root": "logs",
    }

    audit_runs = tuple((tmp_path / "logs/cva_test_audit").iterdir())
    assert len(audit_runs) == 1
    audit_environment = json.loads((audit_runs[0] / "environment.json").read_text(encoding="utf-8"))
    audit_config = yaml.safe_load((audit_runs[0] / "config.yaml").read_text(encoding="utf-8"))
    assert audit_environment["dataset_manifest_hash"] == _sha256_path(manifest_path)
    assert audit_environment["checkpoint_hash"] is None
    assert set(audit_config) == {
        "artifact_root",
        "dataset",
        "log_root",
        "manifest",
        "output",
        "overwrite",
        "report_root",
        "visual_audit",
    }


def test_audit_does_not_publish_ready_report_when_logger_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    output = tmp_path / "orphan-audit.json"

    def _fail_finalize(self, *, checkpoint_hash=None) -> None:
        raise RuntimeError("injected logger finalization failure")

    from compbias.io.logging import RunLogger

    monkeypatch.setattr(RunLogger, "finalize", _fail_finalize)
    with pytest.raises(RuntimeError, match="injected logger finalization failure"):
        audit_dataset_script.main(
            [
                "--manifest",
                str(tmp_path / "output/manifest.json"),
                "--output",
                str(output),
                "--report-root",
                str(tmp_path),
                "--log-root",
                str(tmp_path / "logs"),
                "--artifact-root",
                str(tmp_path),
            ]
        )

    assert not output.exists()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_generation_promotion_rolls_back_on_base_exception(
    interruption: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_staged = tmp_path / "stage-first"
    second_staged = tmp_path / "stage-second"
    first_destination = tmp_path / "first"
    second_destination = tmp_path / "second"
    first_staged.write_text("new-first", encoding="utf-8")
    second_staged.write_text("new-second", encoding="utf-8")
    first_destination.write_text("old-first", encoding="utf-8")
    second_destination.write_text("old-second", encoding="utf-8")
    original_replace = generate_cva_script.os.replace
    calls = 0

    def _interrupting_replace(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise interruption()
        original_replace(source, destination)

    monkeypatch.setattr(generate_cva_script.os, "replace", _interrupting_replace)

    with pytest.raises(interruption):
        generate_cva_script._promote_transaction(
            (
                (first_staged, first_destination),
                (second_staged, second_destination),
            )
        )

    assert first_destination.read_text(encoding="utf-8") == "old-first"
    assert second_destination.read_text(encoding="utf-8") == "old-second"
    assert not tuple(tmp_path.glob(".*.cva-backup-*"))


def test_audit_publication_rolls_back_log_when_report_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged_log = tmp_path / "staged-log"
    staged_log.mkdir()
    (staged_log / "environment.json").write_text("new-log", encoding="utf-8")
    staged_report = tmp_path / "staged-report.json"
    staged_report.write_text("new-report", encoding="utf-8")
    final_log = tmp_path / "logs/experiment/run"
    final_report = tmp_path / "audit.json"
    final_report.write_text("old-report", encoding="utf-8")
    original_replace = audit_dataset_script.os.replace
    calls = 0

    def _failing_replace(source, destination) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected report promotion failure")
        original_replace(source, destination)

    monkeypatch.setattr(audit_dataset_script.os, "replace", _failing_replace)

    with pytest.raises(OSError, match="injected report promotion failure"):
        audit_dataset_script._promote_transaction(
            (
                (staged_log, final_log),
                (staged_report, final_report),
            )
        )

    assert not final_log.exists()
    assert final_report.read_text(encoding="utf-8") == "old-report"
    assert not tuple(tmp_path.glob(".*.audit-backup-*"))


def test_audit_rejects_overlapping_report_and_log_outputs(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr

    audited = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_dataset.py"),
            "--manifest",
            str(tmp_path / "output/manifest.json"),
            "--output",
            str(tmp_path / "logs/audit.json"),
            "--report-root",
            str(tmp_path),
            "--log-root",
            str(tmp_path / "logs"),
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert audited.returncode == 2
    assert "disjoint" in audited.stderr


def test_audit_gates_style_shortcut_against_fully_crossed_semantics(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = tmp_path / "output/dataset.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    for record in records:
        if (
            record["task_family"] == "digit_offset"
            and record["split_keys"]["semantic_split"] == "train"
            and int(record["sample_id"].split("_r")[0].rsplit("_", 1)[1]) % 2 == 0
        ):
            record["split_keys"]["visual_style"] = "baseline"
    dataset.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest["dataset_file_sha256"] = _sha256_path(dataset)
    manifest["content_sha256"] = _content_sha256_for_records(records)
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["style_semantic_joint_independence"]["complete"] is False
    assert report["style_semantic_joint_independence"]["violations"]


def test_audit_rejects_solver_consistent_dataset_content_tampering(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    dataset = tmp_path / "output/dataset.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    records[0]["image_path"] = "images/solver-still-consistent-but-tampered.png"
    dataset.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["solver_pass_rate"] == 1.0
    assert report["manifest_content_sha256_matches"] is False


def test_audit_gates_iid_ood_answer_distribution_mismatch(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    dataset = tmp_path / "output/dataset.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    ood_records = [
        record
        for record in records
        if record["split_keys"]["semantic_split"] == "ood_test"
        and record["task_family"] == "digit_offset"
    ]
    target = ood_records[0]
    semantic_prefix = target["sample_id"].rsplit("_r", 1)[0]
    targets = [
        record
        for record in ood_records
        if record["sample_id"].rsplit("_r", 1)[0] == semantic_prefix
    ]
    donor = next(
        record
        for record in records
        if record["split_keys"]["semantic_split"] == "ood_test"
        and record["task_family"] != target["task_family"]
    )
    for target_record in targets:
        for field in (
            "task_family",
            "scene",
            "question",
            "canonical_answer",
            "canonical_reasoning",
            "error_catalog",
        ):
            target_record[field] = donor[field]
    dataset.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["answer_balance"]["complete"] is False
    assert report["answer_balance"]["iid_ood_exact_match"] is False
    assert report["answer_balance"]["violations"]


def test_audit_gates_noncanonical_or_incomplete_visual_factor_catalog(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    styles = manifest["generator_config"]["visual_styles"]
    manifest["generator_config"]["visual_styles"] = [styles[1], styles[0], *styles[2:]]
    manifest["config_sha256"] = manifest_sha256(manifest["generator_config"])
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["visual_factor_realization_audit"]["complete"] is False
    assert report["automatic_audit_clean"] is False


def test_audit_rejects_ood_images_replaced_by_their_iid_sources(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = [
        json.loads(line) for line in (tmp_path / "output/dataset.jsonl").read_text().splitlines()
    ]
    ood_records = [
        record for record in records if record["split_keys"]["semantic_split"] == "ood_test"
    ]
    images = tmp_path / "output/images"
    for record in ood_records:
        ood_id = record["sample_id"]
        source_id = record["source_id"]
        source_bytes = (images / f"{source_id}.png").read_bytes()
        (images / f"{ood_id}.png").write_bytes(source_bytes)
        manifest["image_sha256"][ood_id] = manifest["image_sha256"][source_id]
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["ood_image_shift"]["complete"] is False
    assert report["ood_image_shift"]["checked_pair_count"] == len(ood_records)
    assert len(report["ood_image_shift"]["violations"]) == len(ood_records)
    assert report["automatic_audit_clean"] is False


@pytest.mark.parametrize(
    ("tamper", "expected_field"),
    [
        ("config", "manifest_config_sha256_matches"),
        ("sample_ids", "manifest_sample_ids_match"),
        ("image_digest", "manifest_image_sha256_matches"),
    ],
)
def test_audit_independently_recomputes_manifest_contract_fields(
    tmp_path, tamper: str, expected_field: str
) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if tamper == "config":
        manifest["config_sha256"] = "0" * 64
    elif tamper == "sample_ids":
        manifest["sample_ids"] = list(reversed(manifest["sample_ids"]))
    else:
        first_id = manifest["sample_ids"][0]
        manifest["image_sha256"][first_id] = "0" * 64
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["manifest_self_sha256_matches"] is True
    assert report[expected_field] is False


def test_audit_rejects_image_byte_tampering(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    image = tmp_path / "output/images" / f"{manifest['sample_ids'][0]}.png"
    image.write_bytes(image.read_bytes() + b"tampered")

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["manifest_image_sha256_matches"] is False


def test_audit_replay_rejects_resigned_solver_valid_data_and_image_tampering(
    tmp_path: Path,
) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = tmp_path / "output/dataset.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    records[0]["question"].pop("text")
    dataset.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    image_id = records[1]["sample_id"]
    image_path = tmp_path / "output/images" / f"{image_id}.png"
    image_path.write_bytes(image_path.read_bytes() + b"resigned-tamper")
    manifest["dataset_file_sha256"] = _sha256_path(dataset)
    manifest["image_sha256"][image_id] = _sha256_path(image_path)
    manifest["content_sha256"] = _content_sha256_for_records(records)
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["deterministic_replay"]["complete"] is False
    assert report["deterministic_replay"]["generator_matches"] is False
    assert report["deterministic_replay"]["renderer_matches"] is False


def test_audit_replay_rejects_resigned_contact_sheet_tampering(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sheet_value = manifest["contact_sheets"][0]
    sheet_path = tmp_path / sheet_value
    sheet_path.write_bytes(sheet_path.read_bytes() + b"resigned-sheet-tamper")
    manifest["contact_sheet_sha256"][sheet_path.name] = _sha256_path(sheet_path)
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["contact_sheet_sha256_matches"] is True
    assert report["deterministic_replay"]["complete"] is False
    assert report["deterministic_replay"]["contact_sheets_match"] is False
    assert report["deterministic_replay"]["contact_sheet_mismatches"] == [sheet_path.name]


def test_audit_rejects_manifest_self_hash_mismatch(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rendered_image_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["manifest_self_sha256_matches"] is False


def test_phase_d_ready_requires_closed_bound_human_review_schema(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    audit_path = tmp_path / "visual-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "human_reviewer_signoff": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "image_set_sha256": manifest_sha256(manifest["image_sha256"]),
            }
        ),
        encoding="utf-8",
    )

    audited = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_dataset.py"),
            "--manifest",
            str(tmp_path / "output/manifest.json"),
            "--artifact-root",
            str(tmp_path),
            "--visual-audit",
            str(audit_path),
            "--output",
            str(tmp_path / "audit.json"),
            "--report-root",
            str(tmp_path),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert audited.returncode != 0
    assert "reviewer" in audited.stderr.lower() or "schema" in audited.stderr.lower()


def test_bound_agent_review_cannot_satisfy_human_phase_d_gate(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    audit_path = tmp_path / "visual-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "reviewer-codex",
                "reviewer_type": "codex_agent",
                "review_date": "2026-08-14",
                "review_result": "pass",
                "human_reviewer_signoff": False,
                "images_reviewed": len(manifest["sample_ids"]),
                "reviewed_sample_ids": manifest["sample_ids"],
                "contact_sheets_reviewed": len(manifest["contact_sheets"]),
                "reviewed_contact_sheets": manifest["contact_sheets"],
                "manifest_sha256": manifest["manifest_sha256"],
                "image_set_sha256": manifest_sha256(manifest["image_sha256"]),
            }
        ),
        encoding="utf-8",
    )

    audited = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_dataset.py"),
            "--manifest",
            str(tmp_path / "output/manifest.json"),
            "--artifact-root",
            str(tmp_path),
            "--visual-audit",
            str(audit_path),
            "--output",
            str(tmp_path / "audit.json"),
            "--report-root",
            str(tmp_path),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert audited.returncode == 0, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["human_review_binding_matches"] is True
    assert report["human_reviewer_signoff"] is False
    assert report["phase_d_ready"] is False


def test_audit_rejects_reviewer_pii_before_report_or_log_publication(tmp_path: Path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    audit_path = tmp_path / "visual-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "alice@example.com",
                "reviewer_type": "human",
                "review_date": "2026-08-14",
                "review_result": "pass",
                "human_reviewer_signoff": True,
                "images_reviewed": len(manifest["sample_ids"]),
                "reviewed_sample_ids": manifest["sample_ids"],
                "contact_sheets_reviewed": len(manifest["contact_sheets"]),
                "reviewed_contact_sheets": manifest["contact_sheets"],
                "manifest_sha256": manifest["manifest_sha256"],
                "image_set_sha256": manifest_sha256(manifest["image_sha256"]),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/audit_dataset.py"),
            "--manifest",
            str(tmp_path / "output/manifest.json"),
            "--artifact-root",
            str(tmp_path),
            "--visual-audit",
            str(audit_path),
            "--output",
            str(tmp_path / "audit.json"),
            "--report-root",
            str(tmp_path),
            "--log-root",
            str(tmp_path / "audit-logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "public pseudonym" in completed.stderr
    assert not (tmp_path / "audit.json").exists()
    assert not (tmp_path / "audit-logs").exists()


def test_audit_rejects_duplicate_json_keys_and_nonfinite_constants(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    dataset = tmp_path / "output/dataset.jsonl"
    lines = dataset.read_text().splitlines()
    first = lines[0]
    lines[0] = first[:-1] + ',"sample_id":"duplicate","probe":NaN}'
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")

    audited = _audit(tmp_path)

    assert audited.returncode != 0
    assert "duplicate" in audited.stderr.lower() or "non-finite" in audited.stderr.lower()


def test_audit_rejects_unknown_raw_fields_and_noncanonical_image_path(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    dataset = tmp_path / "output/dataset.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    records[0]["secret"] = "/Users/private/secret.txt"
    records[1]["image_path"] = "../images/probe.png"
    dataset.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    audited = _audit(tmp_path)

    assert audited.returncode != 0
    assert "unknown sample fields" in audited.stderr or "image_path" in audited.stderr


def test_audit_rejects_extra_image_and_symlink_entries(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    images = tmp_path / "output/images"
    extra = images / "extra.png"
    extra.write_bytes(b"extra")

    audited_extra = _audit(tmp_path)

    assert audited_extra.returncode == 1, audited_extra.stderr
    assert json.loads((tmp_path / "audit.json").read_text())["image_set_matches"] is False

    extra.unlink()
    link = images / "linked.png"
    link.symlink_to(next(images.glob("*.png")))
    audited_link = _audit(tmp_path)

    assert audited_link.returncode != 0
    assert "symlink" in audited_link.stderr.lower()


def test_audit_rejects_manifest_paths_outside_approved_artifact_root(tmp_path) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["jsonl_path"] = "../escaped.jsonl"
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode != 0
    assert "artifact root" in audited.stderr.lower()


def test_audit_rejects_image_question_collisions_with_multiple_answers(
    tmp_path: Path,
) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = tmp_path / "output/dataset.jsonl"
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    collision_pair = next(
        (first, second)
        for index, first in enumerate(records)
        for second in records[index + 1 :]
        if first["question"] == second["question"]
        and first["canonical_answer"] != second["canonical_answer"]
    )
    manifest["image_sha256"][collision_pair[1]["sample_id"]] = manifest["image_sha256"][
        collision_pair[0]["sample_id"]
    ]
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode == 1, audited.stderr
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["image_question_answer_collisions"]


def test_generation_checks_every_target_before_writing_any_output(tmp_path) -> None:
    blocked_image = tmp_path / "output/images/digit_offset_train_000000.png"
    blocked_image.parent.mkdir(parents=True)
    blocked_image.write_bytes(b"do-not-overwrite")

    generated = _generate(tmp_path)

    assert generated.returncode != 0
    assert "already exist" in generated.stderr
    assert blocked_image.read_bytes() == b"do-not-overwrite"
    assert not (tmp_path / "output/dataset.jsonl").exists()
    assert not (tmp_path / "output/manifest.json").exists()
    assert not tuple((tmp_path / "figures").glob("*.png"))
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("field", "section", "trusted_root"),
    [
        ("output", "dataset", "output"),
        ("manifest", "dataset", "output"),
        ("images_dir", "rendering", "output"),
        ("contact_sheet_dir", "rendering", "figures"),
        ("root", "logging", "logs"),
    ],
)
def test_generation_rejects_configured_paths_outside_explicit_roots(
    tmp_path: Path, field: str, section: str, trusted_root: str
) -> None:
    config, config_path = _small_config(tmp_path)
    escaped = tmp_path.parent / f"{tmp_path.name}-{section}-{field}"
    config[section][field] = str(escaped)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    generated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/generate_cva.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "output"),
            "--figure-root",
            str(tmp_path / "figures"),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert generated.returncode != 0
    assert trusted_root in generated.stderr
    assert not escaped.exists()


def test_generation_rejects_duplicate_yaml_keys_at_any_depth(tmp_path: Path) -> None:
    _config, config_path = _small_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("  seed: 20260814\n", "  seed: 1\n  seed: 20260814\n", 1),
        encoding="utf-8",
    )

    generated = _generate_from_config(tmp_path, config_path)

    assert generated.returncode != 0
    assert "duplicate YAML key: seed" in generated.stderr


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("dataset: &shared {}\nrendering: *shared\nlogging: {}\n", "aliases/cycles"),
        (
            "dataset:\n"
            + "".join(f"{'  ' * depth}child:\n" for depth in range(1, 70))
            + f"{'  ' * 70}value: 1\n",
            "depth/node",
        ),
    ],
)
def test_generation_rejects_yaml_aliases_and_excessive_depth(
    payload: str,
    message: str,
    tmp_path: Path,
) -> None:
    config = tmp_path / "unsafe.yaml"
    config.write_text(payload, encoding="utf-8")

    completed = _generate_from_config(tmp_path, config)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not (tmp_path / "output").exists()


def test_generation_rejects_unsafe_log_experiment_component(tmp_path: Path) -> None:
    config, config_path = _small_config(tmp_path)
    config["logging"]["experiment"] = "../escaped"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    generated = _generate_from_config(tmp_path, config_path)

    assert generated.returncode != 0
    assert "logging.experiment" in generated.stderr
    assert not (tmp_path / "escaped").exists()


def test_generation_rejects_symlink_target_even_when_it_resolves_inside_root(
    tmp_path: Path,
) -> None:
    config, config_path = _small_config(tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True)
    real_target = output_root / "real-dataset.jsonl"
    real_target.write_bytes(b"preserve")
    linked_target = output_root / "linked-dataset.jsonl"
    linked_target.symlink_to(real_target)
    config["dataset"]["output"] = str(linked_target)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    generated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/generate_cva.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--figure-root",
            str(tmp_path / "figures"),
            "--log-root",
            str(tmp_path / "logs"),
            "--overwrite",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert generated.returncode != 0
    assert "symlink" in generated.stderr.lower()
    assert real_target.read_bytes() == b"preserve"


def test_generation_rejects_overlapping_output_targets(tmp_path: Path) -> None:
    config, config_path = _small_config(tmp_path)
    config["dataset"]["manifest"] = config["dataset"]["output"]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    generated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/generate_cva.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "output"),
            "--figure-root",
            str(tmp_path / "figures"),
            "--log-root",
            str(tmp_path / "logs"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert generated.returncode != 0
    assert "distinct" in generated.stderr.lower() or "overlap" in generated.stderr.lower()


def test_overwrite_requires_valid_prior_generated_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "output/dataset.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"arbitrary-user-file")

    generated = _generate(tmp_path, overwrite=True)

    assert generated.returncode != 0
    assert "prior" in generated.stderr.lower()
    assert dataset.read_bytes() == b"arbitrary-user-file"


def test_overwrite_accepts_only_a_complete_self_bound_prior_generation(tmp_path: Path) -> None:
    first = _generate(tmp_path)
    assert first.returncode == 0, first.stderr

    second = _generate(tmp_path, overwrite=True)

    assert second.returncode == 0, second.stderr


def test_overwrite_rejects_ambiguous_duplicate_prior_manifest_keys(tmp_path: Path) -> None:
    first = _generate(tmp_path)
    assert first.returncode == 0, first.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace(
            '  "manifest_sha256":',
            '  "manifest_sha256": "duplicate",\n  "manifest_sha256":',
            1,
        ),
        encoding="utf-8",
    )

    second = _generate(tmp_path, overwrite=True)

    assert second.returncode != 0
    assert "duplicate JSON key" in second.stderr


def test_default_roots_reject_overwriting_repository_source_file(tmp_path: Path) -> None:
    config, config_path = _small_config(tmp_path)
    protected = REPOSITORY_ROOT / "src/compbias/__init__.py"
    before = protected.read_bytes()
    config["dataset"]["output"] = str(protected)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    generated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/generate_cva.py"),
            "--config",
            str(config_path),
            "--overwrite",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert generated.returncode != 0
    assert "approved root" in generated.stderr.lower()
    assert protected.read_bytes() == before


def test_audit_rejects_dataset_symlink_that_resolves_inside_artifact_root(
    tmp_path: Path,
) -> None:
    generated = _generate(tmp_path)
    assert generated.returncode == 0, generated.stderr
    manifest_path = tmp_path / "output/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset = tmp_path / "output/dataset.jsonl"
    linked = tmp_path / "output/linked.jsonl"
    linked.symlink_to(dataset)
    manifest["jsonl_path"] = "output/linked.jsonl"
    _rewrite_manifest_with_valid_self_hash(manifest_path, manifest)

    audited = _audit(tmp_path)

    assert audited.returncode != 0
    assert "symlink" in audited.stderr.lower()
