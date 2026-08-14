#!/usr/bin/env python3
"""Audit frozen vision weights, fixed-image states, and representation probes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from compbias.audit.frozen_components import VLMRegimeSpec
from compbias.audit.representation_invariance import audit_representation_invariance
from compbias.io.artifact_paths import validated_artifact_path
from compbias.io.strict_json import load_strict_json_mapping, write_new_json

_FIELDS = frozenset({"vision_weight_sha256", "fixed_image_hidden", "frozen_representation_probe"})
_REGIMES = {
    "lm_only": ("frozen", "frozen", "lora"),
    "projector_lm": ("frozen", "full", "lora"),
}


def _output(path: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    target = validated_artifact_path(
        path,
        repository_root=repository_root,
        label="frozen-regime audit",
        suffix=".json",
    )
    if repository_root in target.parents:
        root = repository_root / "artifacts/metrics"
        if root not in target.parents:
            raise ValueError("repository frozen-regime audits must stay under artifacts/metrics")
    return target


def _validate_snapshot(raw: Mapping[str, object], label: str) -> None:
    if set(raw) != _FIELDS:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(_FIELDS))}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--regime", choices=sorted(_REGIMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-tolerance", type=float, default=1e-8)
    parser.add_argument("--probe-tolerance", type=float, default=1e-8)
    args = parser.parse_args(argv)
    try:
        before = load_strict_json_mapping(args.before, label="before snapshot")
        after = load_strict_json_mapping(args.after, label="after snapshot")
        _validate_snapshot(before, "before snapshot")
        _validate_snapshot(after, "after snapshot")
        updates = _REGIMES[args.regime]
        regime = VLMRegimeSpec(args.regime, *updates)
        report = audit_representation_invariance(
            before_weight_sha256=before["vision_weight_sha256"],
            after_weight_sha256=after["vision_weight_sha256"],
            before_hidden=before["fixed_image_hidden"],
            after_hidden=after["fixed_image_hidden"],
            probe_before=before["frozen_representation_probe"],
            probe_after=after["frozen_representation_probe"],
            hidden_tolerance=args.hidden_tolerance,
            probe_tolerance=args.probe_tolerance,
        )
        output = _output(args.output)
        payload = {
            "schema_version": 2,
            "artifact_type": "frozen_visual_acquisition_audit",
            "regime": args.regime,
            "acquisition_frozen": regime.acquisition_frozen,
            "allowed_claims": sorted(regime.allowed_claims),
            "forbidden_claims": ["acquisition_improvement"],
            **report.to_mapping(),
        }
        write_new_json(output, payload)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3
    print(f"{'PASS' if report.passed else 'FAIL'}: frozen-regime audit written to {output}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
