from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compbias.recoverability.measurement_qualification import (
    load_measurement_qualification_config,
)
from compbias.recoverability.measurement_qualification_anchor import (
    MeasurementQualificationDataAnchor,
    load_measurement_qualification_data_anchor,
    verify_measurement_qualification_data_evidence,
)
from compbias.recoverability.measurement_qualification_data import (
    write_measurement_qualification_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/recoverability/measurement_qualification_v1.yaml"
ANCHOR = ROOT / "configs/recoverability/measurement_qualification_data_anchor.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reserved() -> frozenset[tuple[int, int, int, int]]:
    return frozenset((index, index + 1, index + 2, index + 3) for index in range(100))


def _fixture(
    tmp_path: Path,
) -> tuple[MeasurementQualificationDataAnchor, Path, Path, Path]:
    config = load_measurement_qualification_config(CONFIG)
    dataset = tmp_path / "measurement_qualification_v1"
    manifest = write_measurement_qualification_dataset(
        config,
        reserved_numeric_tables=_reserved(),
        output_dir=dataset,
    )
    marker = tmp_path / "measurement_qualification_v1.attempted.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "MEASUREMENT_QUALIFICATION_DATA_GENERATION_STARTED",
                "dataset_id": config.dataset_id,
                "seed": config.seed,
                "server_package_lock_sha256": "1" * 64,
                "source_dataset_records_sha256": config.source_dataset_records_sha256,
                "source_stage2_v2_external_evidence_sha256": (
                    config.source_stage2_v2_external_evidence_sha256
                ),
                "model_calls": 0,
                "hypothesis_tested": False,
                "confirmatory_execution_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    console = tmp_path / "measurement-qualification-data.log"
    console.write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
        + "\nqualification_data_exit=0\n",
        encoding="utf-8",
    )
    frozen = load_measurement_qualification_data_anchor(ANCHOR)
    synthetic = replace(
        frozen,
        generation_server_package_lock_sha256="1" * 64,
        attempt_marker_sha256=_sha256(marker),
        manifest_sha256=_sha256(dataset / "manifest.json"),
        records_sha256=_sha256(dataset / "records.jsonl"),
        images_sha256=str(manifest["images_sha256"]),
        console_sha256=_sha256(console),
    )
    return synthetic, dataset, marker, console


def test_measurement_qualification_dataset_server_evidence_is_frozen() -> None:
    anchor = load_measurement_qualification_data_anchor(ANCHOR)

    assert anchor.status == "FINAL_GENERATED_DATASET_DO_NOT_RERUN"
    assert anchor.generation_code_commit == "93bd5f509728a0465e4891672b3dbff0cfd2f568"
    assert anchor.generation_server_package_lock_sha256 == (
        "25808fffdf62981163550084c36c4b37428fada7c85bc8b3a5e286b4bc75ec4c"
    )
    assert anchor.attempt_marker_sha256 == (
        "29667a9866e2f969aa05a10737abec1fe59f657076e603d8be6328802e5cfd97"
    )
    assert anchor.manifest_sha256 == (
        "6c85db5a4bb6dd11f798f7bb5ccce777954dd8e699463e45cb71503e1521d091"
    )
    assert anchor.records_sha256 == (
        "98c1ab1228480b58dc4309f7c64280c347e87ac44547d79e36ab6ceb52adff6d"
    )
    assert anchor.images_sha256 == (
        "e01ea67f4b5ace4cec3201018ceed9cb68a5699470711e4d233ce64b5263d760"
    )
    assert anchor.console_sha256 == (
        "1f9ef2a6382dccea5f9de78bdd3b2ed78cc30450d0be548bde505ee23546c7ee"
    )
    assert anchor.records == 300
    assert anchor.model_calls == 0
    assert anchor.hypothesis_tested is False
    assert anchor.confirmatory_execution_authorized is False
    assert anchor.training_invoked is False


def test_dataset_evidence_verifier_replays_all_bytes_and_semantics(tmp_path: Path) -> None:
    anchor, dataset, marker, console = _fixture(tmp_path)

    verification = verify_measurement_qualification_data_evidence(
        anchor,
        dataset_root=dataset,
        attempt_marker=marker,
        console_log=console,
        config=load_measurement_qualification_config(CONFIG),
        reserved_numeric_tables=_reserved(),
    )

    assert verification == anchor

    image = dataset / "images/qualification-000000.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="image bundle SHA-256"):
        verify_measurement_qualification_data_evidence(
            anchor,
            dataset_root=dataset,
            attempt_marker=marker,
            console_log=console,
            config=load_measurement_qualification_config(CONFIG),
            reserved_numeric_tables=_reserved(),
        )


@pytest.mark.parametrize("target", ["marker", "manifest", "records", "console"])
def test_dataset_evidence_verifier_rejects_each_tampered_artifact(
    tmp_path: Path,
    target: str,
) -> None:
    anchor, dataset, marker, console = _fixture(tmp_path)
    paths = {
        "marker": marker,
        "manifest": dataset / "manifest.json",
        "records": dataset / "records.jsonl",
        "console": console,
    }
    paths[target].write_bytes(paths[target].read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_measurement_qualification_data_evidence(
            anchor,
            dataset_root=dataset,
            attempt_marker=marker,
            console_log=console,
            config=load_measurement_qualification_config(CONFIG),
            reserved_numeric_tables=_reserved(),
        )


def test_dataset_evidence_verifier_rejects_semantic_marker_even_with_new_hash(
    tmp_path: Path,
) -> None:
    anchor, dataset, marker, console = _fixture(tmp_path)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["confirmatory_execution_authorized"] = True
    marker.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tampered = replace(anchor, attempt_marker_sha256=_sha256(marker))

    with pytest.raises(ValueError, match="attempt marker payload"):
        verify_measurement_qualification_data_evidence(
            tampered,
            dataset_root=dataset,
            attempt_marker=marker,
            console_log=console,
            config=load_measurement_qualification_config(CONFIG),
            reserved_numeric_tables=_reserved(),
        )
