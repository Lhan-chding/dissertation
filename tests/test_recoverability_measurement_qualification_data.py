from __future__ import annotations

import json
from pathlib import Path

import pytest

from compbias.recoverability.measurement_qualification import (
    load_measurement_qualification_config,
)
from compbias.recoverability.measurement_qualification_data import (
    verify_measurement_qualification_dataset,
    write_measurement_qualification_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recoverability" / "measurement_qualification_v1.yaml"


def _reserved() -> frozenset[tuple[int, int, int, int]]:
    return frozenset((index, index + 1, index + 2, index + 3) for index in range(100))


def test_qualification_dataset_is_complete_balanced_and_replayable(tmp_path: Path) -> None:
    config = load_measurement_qualification_config(CONFIG)
    output = tmp_path / "measurement_qualification_v1"

    manifest = write_measurement_qualification_dataset(
        config,
        reserved_numeric_tables=_reserved(),
        output_dir=output,
    )
    verification = verify_measurement_qualification_dataset(
        output,
        config=config,
        reserved_numeric_tables=_reserved(),
    )

    assert manifest["artifact_type"] == "recoverability_measurement_qualification_dataset"
    assert manifest["status"] == "FROZEN_DATASET_NOT_EVALUATED"
    assert manifest["image_size"] == [512, 384]
    assert manifest["render_mode"] == "axis_scale_v0_3"
    assert manifest["source_stage2_v2_external_evidence_sha256"] == (
        "3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577"
    )
    assert manifest["record_count"] == 300
    assert manifest["split_counts"] == {"qualification": 300}
    assert manifest["strata_counts"] == {
        f"{chart}|{operation}": 50
        for chart in ("grouped_bar", "line")
        for operation in ("difference", "max_minus_min", "sum")
    }
    assert manifest["images_generated"] == 300
    assert manifest["numeric_table_overlap_with_source"] == 0
    assert manifest["model_calls"] == 0
    assert manifest["hypothesis_tested"] is False
    assert manifest["confirmatory_execution_authorized"] is False
    assert manifest["training_invoked"] is False
    assert verification.verified is True
    assert len(verification.scenes) == 300
    assert all(scene.image_path.is_file() for scene in verification.scenes)
    assert len(list((output / "images").glob("*.png"))) == 300


def test_qualification_dataset_refuses_overwrite_and_image_tampering(tmp_path: Path) -> None:
    config = load_measurement_qualification_config(CONFIG)
    output = tmp_path / "measurement_qualification_v1"
    write_measurement_qualification_dataset(
        config,
        reserved_numeric_tables=_reserved(),
        output_dir=output,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_measurement_qualification_dataset(
            config,
            reserved_numeric_tables=_reserved(),
            output_dir=output,
        )

    image = output / "images" / "qualification-000000.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="image bundle SHA-256"):
        verify_measurement_qualification_dataset(
            output,
            config=config,
            reserved_numeric_tables=_reserved(),
        )


def test_qualification_dataset_rejects_record_selection_or_schema_tampering(
    tmp_path: Path,
) -> None:
    config = load_measurement_qualification_config(CONFIG)
    output = tmp_path / "measurement_qualification_v1"
    write_measurement_qualification_dataset(
        config,
        reserved_numeric_tables=_reserved(),
        output_dir=output,
    )
    records = output / "records.jsonl"
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    rows[0]["split"] = "iid_test"
    records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="records SHA-256"):
        verify_measurement_qualification_dataset(
            output,
            config=config,
            reserved_numeric_tables=_reserved(),
        )


def test_qualification_dataset_rejects_provenance_tampering(tmp_path: Path) -> None:
    config = load_measurement_qualification_config(CONFIG)
    output = tmp_path / "measurement_qualification_v1"
    write_measurement_qualification_dataset(
        config,
        reserved_numeric_tables=_reserved(),
        output_dir=output,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_stage2_v2_external_evidence_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="registered contract"):
        verify_measurement_qualification_dataset(
            output,
            config=config,
            reserved_numeric_tables=_reserved(),
        )
