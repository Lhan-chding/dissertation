#!/usr/bin/env python3
"""Build a simultaneous multi-interface compensation certificate."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from compbias.identification.partial_identification import (
    InterfaceGammaEstimate,
    robust_compensation_interval,
)
from compbias.identification.validity_gates import evaluate_interface_validity
from compbias.io.artifact_paths import validated_artifact_path
from compbias.io.strict_json import load_strict_json_mapping, write_new_json

_TOP = frozenset({"schema_version", "bootstrap_draws", "confidence", "seed", "interfaces"})
_INTERFACE = frozenset({"interface_id", "cluster_gammas", "validity"})
_VALIDITY = frozenset(
    {
        "oracle_loss",
        "replay_js",
        "replay_accuracy_gap",
        "image_exclusion_gap",
        "parser_reliability",
        "natural_state_count_by_error",
        "natural_input_count_by_error",
    }
)


def _exact(mapping: Mapping[str, object], fields: frozenset[str], label: str) -> None:
    if set(mapping) != fields:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(fields))}")


def _output(path: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    target = validated_artifact_path(
        path,
        repository_root=repository_root,
        label="partial-identification certificate",
        suffix=".json",
    )
    if repository_root in target.parents:
        root = repository_root / "artifacts/partial_id"
        if root not in target.parents:
            raise ValueError("repository certificates must stay under artifacts/partial_id")
    return target


def _build(raw: Mapping[str, object]) -> dict[str, object]:
    _exact(raw, _TOP, "certificate input")
    if raw["schema_version"] != 2:
        raise ValueError("schema_version must equal 2")
    interfaces = raw["interfaces"]
    if not isinstance(interfaces, list) or not 1 <= len(interfaces) <= 32:
        raise ValueError("interfaces must contain 1-32 records")
    estimates: list[InterfaceGammaEstimate] = []
    reports = []
    seen: set[str] = set()
    interface_rows: list[dict[str, object]] = []
    for index, row in enumerate(interfaces):
        if not isinstance(row, Mapping) or any(not isinstance(key, str) for key in row):
            raise ValueError(f"interfaces[{index}] must be a string-keyed mapping")
        _exact(row, _INTERFACE, f"interfaces[{index}]")
        interface_id = row["interface_id"]
        if not isinstance(interface_id, str) or not interface_id or interface_id in seen:
            raise ValueError("interface IDs must be non-empty and unique")
        seen.add(interface_id)
        gammas = row["cluster_gammas"]
        if not isinstance(gammas, list) or not 2 <= len(gammas) <= 100_000:
            raise ValueError("cluster_gammas must contain 2-100000 sample-cluster values")
        estimate = InterfaceGammaEstimate(interface_id, tuple(gammas))
        validity = row["validity"]
        if not isinstance(validity, Mapping) or any(not isinstance(key, str) for key in validity):
            raise ValueError("validity must be a string-keyed mapping")
        _exact(validity, _VALIDITY, "validity")
        report = evaluate_interface_validity(interface_id=interface_id, **validity)
        estimates.append(estimate)
        reports.append(report)
        interface_rows.append(
            {
                "interface_id": interface_id,
                "gamma_mean": sum(estimate.cluster_gammas) / len(estimate.cluster_gammas),
                "valid": report.passed,
                "failed_gates": list(report.failed_gates),
                "cluster_count": len(estimate.cluster_gammas),
            }
        )
    result = robust_compensation_interval(
        tuple(estimates),
        tuple(reports),
        bootstrap_draws=raw["bootstrap_draws"],
        confidence=raw["confidence"],
        seed=raw["seed"],
    )
    return {
        "schema_version": 2,
        "artifact_type": "operational_multi_interface_compensation_certificate",
        "claim_scope": "operational interfaces only; no unique anatomical boundary",
        "valid_interfaces": list(result.valid_interfaces),
        "invalid_interfaces": list(result.invalid_interfaces),
        "point_interval": [result.point_lower, result.point_upper],
        "simultaneous_interval": [result.simultaneous_lower, result.simultaneous_upper],
        "critical_value": result.critical_value,
        "conclusion": result.conclusion,
        "interfaces": interface_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = load_strict_json_mapping(args.input, label="partial-identification input")
        output = _output(args.output)
        write_new_json(output, _build(raw))
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3
    print(f"COMPLETE: wrote operational partial-identification certificate to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
