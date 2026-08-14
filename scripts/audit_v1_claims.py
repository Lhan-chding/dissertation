#!/usr/bin/env python3
"""Classify legacy compensability claims under the v2 evidence contract."""

from __future__ import annotations

import argparse
import csv
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from compbias.io.artifact_paths import validated_artifact_path


@dataclass(frozen=True, slots=True)
class _Rule:
    claim_id: str
    pattern: re.Pattern[str]
    legacy_claim: str
    decision: str
    replacement: str
    rationale: str


_RULES = (
    _Rule(
        "V1-PROPERTY-TESTS",
        re.compile(r"(?:7[,.]?000|7000).{0,80}(?:numerical|property|identity)", re.I | re.S),
        "Numerical identities validate the theory implementation.",
        "keep",
        "Property tests verify the implementation of the stated formulas.",
        "Software verification is retained but is not empirical VLM evidence.",
    ),
    _Rule(
        "V1-SELECTION-DO",
        re.compile(
            r"selection law.{0,100}do-compensability|do-compensability.{0,100}selection law",
            re.I | re.S,
        ),
        "The selection law must use interventional do-compensability.",
        "retract",
        "Natural conditional success c_sel is the exact trajectory-selection input.",
        "c_fork diagnoses whether natural selection is causally attributable to the mediator.",
    ),
    _Rule(
        "V1-SYNTHETIC-ORACLE",
        re.compile(r"exact KL|artificial c\[e\]|synthetic mechanism", re.I),
        "Synthetic c[e] is evidence for natural VLM error selection.",
        "demote",
        "Treat exact KL with synthetic c[e] as a synthetic mechanism oracle only.",
        "Artificial states require transport validation against natural mediator states.",
    ),
    _Rule(
        "V1-SINGLE-VISION",
        re.compile(
            r"(?:single|single-scene|one).{0,40}16\s*(?:x|\u00d7)\s*16|"
            r"16\s*(?:x|\u00d7)\s*16.{0,60}(?:PIL|CNN|scene)",
            re.I | re.S,
        ),
        "A single 16x16 PIL-CNN establishes the VLM mechanism.",
        "demote",
        "The single-scene result is a modular proof-of-mechanism only.",
        "It lacks broad semantics, natural mediator replay, and real VLM training.",
    ),
    _Rule(
        "V1-ADDITIVE-DECOMPOSITION",
        re.compile(r"e[_ ]?O\s*=\s*e[_ ]?P\s*\+\s*e[_ ]?R|additive decomposition", re.I),
        "Perception and reasoning errors add in a common Euclidean space.",
        "demote",
        "Use crossed risks D_P, D_R, and Gamma; retain additivity only as a special case.",
        "Real VLM states are high-dimensional and no unique Euclidean split is identified.",
    ),
    _Rule(
        "V1-UNIQUE-BOUNDARY",
        re.compile(
            r"unique.{0,30}(?:perception|reasoning).{0,30}(?:boundary|state)|"
            r"true internal perception",
            re.I | re.S,
        ),
        "Black-box behavior identifies a unique perception/reasoning boundary.",
        "retract",
        "Report an operational certificate over a pre-registered interface family.",
        "End-to-end factorization is non-identifiable from final behavior alone.",
    ),
    _Rule(
        "V1-REAL-VLM-RERUN",
        re.compile(r"real VLM rerun|real VLM|Qwen2\.5-VL", re.I),
        "The controlled mechanism has already been verified in a real VLM.",
        "rerun",
        "Run the registered natural-mediator VLM regimes after the authenticated GPU gate.",
        "Current CPU evidence stops before model download and large-GPU training.",
    ),
)


def _read_text(path: Path) -> str:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError("old plan must be a regular non-symlink file")
    if metadata.st_size > 16 * 1024 * 1024:
        raise ValueError("old plan exceeds the 16 MiB safety limit")
    return path.read_text(encoding="utf-8")


def _output_path(path: Path) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    return validated_artifact_path(
        path,
        repository_root=repository_root,
        label="claim audit output",
        suffix=".csv",
    )


def _write_new_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(name)
    fields = (
        "claim_id",
        "legacy_claim",
        "decision",
        "replacement_claim",
        "rationale",
        "source_found",
        "source_excerpt",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = _read_text(args.old_plan)
        output = _output_path(args.out)
        rows: list[dict[str, str]] = []
        for rule in _RULES:
            match = rule.pattern.search(source)
            excerpt = "" if match is None else " ".join(match.group(0).split())[:500]
            rows.append(
                {
                    "claim_id": rule.claim_id,
                    "legacy_claim": rule.legacy_claim,
                    "decision": rule.decision,
                    "replacement_claim": rule.replacement,
                    "rationale": rule.rationale,
                    "source_found": str(match is not None).lower(),
                    "source_excerpt": excerpt,
                }
            )
        _write_new_csv(output, rows)
    except (FileExistsError, OSError, TypeError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3
    print(f"COMPLETE: wrote {len(rows)} v1 claim decisions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
