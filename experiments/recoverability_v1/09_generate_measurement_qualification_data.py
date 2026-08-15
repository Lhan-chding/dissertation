#!/usr/bin/env python3
"""Generate the frozen measurement-qualification dataset without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_LOCK_RELATIVE = "configs/recoverability/server_package_lock_measurement_qualification_data.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_package_lock() -> None:
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return
    if "--execute" not in sys.argv:
        return
    try:
        raw = sys.argv[sys.argv.index("--server-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical qualification data lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    canonical = root / _LOCK_RELATIVE
    if lock_path != canonical or lock_path.is_symlink():
        raise SystemExit("BLOCKED: qualification data lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: qualification data lock is malformed")
    additions = {
        "configs/recoverability/measurement_qualification_v1.yaml",
        "configs/recoverability/server_package_lock_stage2_v2_evidence.yaml",
        "configs/recoverability/stage2_v2_external_evidence_anchor.yaml",
        "experiments/recoverability_v1/09_generate_measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification.py",
        "src/compbias/recoverability/measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification_server.py",
        "src/compbias/recoverability/stage2_v2_anchor.py",
    }
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: qualification data package mismatch for {relative}")
    inherited_lock = root / "configs/recoverability/server_package_lock_stage2_v2_evidence.yaml"
    inherited_lines = inherited_lock.read_text(encoding="utf-8").splitlines()
    inherited_paths = frozenset(
        line.split(":", 1)[1].strip() for line in inherited_lines if line.startswith("  - path:")
    )
    inherited_digests = [
        line.split(":", 1)[1].strip() for line in inherited_lines if line.startswith("    sha256:")
    ]
    if not inherited_paths or len(inherited_paths) != len(inherited_digests):
        raise SystemExit("BLOCKED: inherited Stage-2 v2 evidence lock is malformed")
    if frozenset(paths) != inherited_paths | additions:
        raise SystemExit("BLOCKED: qualification data lock closure is incomplete")
    current_registry = dict(zip(paths, digests, strict=True))
    for relative, expected in zip(sorted(inherited_paths), inherited_digests, strict=True):
        if current_registry.get(relative) != expected:
            raise SystemExit(f"BLOCKED: inherited package digest changed for {relative}")


_bootstrap_package_lock()

from compbias.gpu_pilot.config import load_pilot_paths  # noqa: E402
from compbias.recoverability.measurement_qualification import (  # noqa: E402
    load_measurement_qualification_config,
    load_reserved_numeric_tables,
)
from compbias.recoverability.measurement_qualification_data import (  # noqa: E402
    write_measurement_qualification_dataset,
)
from compbias.recoverability.measurement_qualification_server import (  # noqa: E402
    verify_measurement_qualification_data_package_lock,
)
from compbias.recoverability.stage1_v2 import (  # noqa: E402
    validate_stage1_v2_runtime_paths,
)
from compbias.recoverability.stage2_v2_anchor import (  # noqa: E402
    load_stage2_v2_external_evidence_anchor,
    verify_stage2_v2_external_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path, required=True)
    parser.add_argument("--stage2-v2-external-evidence", type=Path, required=True)
    return parser


def main() -> int:
    if "--execute" not in sys.argv and "--help" not in sys.argv and "-h" not in sys.argv:
        print("BLOCKED: pass --execute for the one registered model-free generation")
        return 2
    args = _parser().parse_args()
    if not args.execute:
        print("BLOCKED: pass --execute for the one registered model-free generation")
        return 2
    root = Path(__file__).resolve().parents[2]
    canonical = {
        "paths": root / "configs/paths.yaml",
        "config": root / "configs/recoverability/measurement_qualification_v1.yaml",
        "lock": root / _LOCK_RELATIVE,
        "external": Path(
            "/cloud/cloud-ssd1/recoverability-v1-evidence/stage2-v2-external-evidence.json"
        ),
    }
    supplied = {
        "paths": args.paths,
        "config": args.config,
        "lock": args.server_package_lock,
        "external": args.stage2_v2_external_evidence,
    }
    for label, expected in canonical.items():
        if supplied[label].resolve() != expected or supplied[label].is_symlink():
            raise ValueError(f"qualification {label} path is not canonical")
    if any(name.startswith("COMPBIAS_") for name in os.environ):
        raise RuntimeError("qualification generation forbids COMPBIAS path overrides")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("qualification generation requires the offline environment")
    verify_measurement_qualification_data_package_lock(canonical["lock"], repository_root=root)
    validate_stage1_v2_runtime_paths(
        canonical["paths"],
        registered_example=root / "configs/paths.example.yaml",
    )
    anchor = load_stage2_v2_external_evidence_anchor(
        root / "configs/recoverability/stage2_v2_external_evidence_anchor.yaml"
    )
    verify_stage2_v2_external_evidence(anchor, canonical["external"])
    config = load_measurement_qualification_config(canonical["config"])
    if config.source_stage2_v2_external_evidence_sha256 != anchor.external_evidence_sha256:
        raise ValueError("qualification config does not bind the external Stage-2 v2 evidence")
    paths = load_pilot_paths(canonical["paths"], environ={})
    if paths.project_root != root:
        raise ValueError("qualification project root differs from the active checkout")
    source_records = paths.data / "generated/cva_chart_pilot_v0_3/records.jsonl"
    reserved = load_reserved_numeric_tables(
        source_records,
        expected_sha256=config.source_dataset_records_sha256,
    )
    output_dir = paths.data / "generated" / config.output_subdirectory
    attempt = paths.data / "generated" / f"{config.output_subdirectory}.attempted.json"
    if output_dir.exists() or output_dir.is_symlink() or attempt.exists() or attempt.is_symlink():
        raise FileExistsError("qualification dataset or attempt marker already exists")
    if attempt.parent.is_symlink() or not attempt.parent.is_dir():
        raise ValueError("qualification data parent must be a regular directory")
    marker = {
        "schema_version": 1,
        "status": "MEASUREMENT_QUALIFICATION_DATA_GENERATION_STARTED",
        "dataset_id": config.dataset_id,
        "seed": config.seed,
        "model_calls": 0,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    with attempt.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    manifest = write_measurement_qualification_dataset(
        config,
        reserved_numeric_tables=reserved,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
