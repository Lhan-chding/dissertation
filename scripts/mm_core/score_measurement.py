#!/usr/bin/env python3
"""Produce CPU-only MM-CORE measurements from retained raw completions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_core.scoring import discover_raw_files, join_field_scores, score_records


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, ArithmeticError) as exc:
                raise ValueError(f"INVALID_JSONL:{path.name}:{number}:{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL_ROW_NOT_OBJECT:{path.name}:{number}")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temp = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--outputs",
        nargs="*",
        type=Path,
        help="Explicit raw completion JSONL files; default raw/**/outputs_*.jsonl",
    )
    parser.add_argument(
        "--field-scores",
        nargs="*",
        type=Path,
        help="Optional identity-matched field metric sidecars; missing metrics stay missing",
    )
    parser.add_argument("--stage")
    parser.add_argument("--expected-k", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261008)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    question_path = root / "data" / "questions.jsonl"
    discovered_outputs, discovered_scores = discover_raw_files(root, args.stage)
    raw_files = discovered_outputs if args.outputs is None else args.outputs
    raw_files = [path.resolve() for path in raw_files]
    if any(path.name.lower().startswith("scores") for path in raw_files):
        raise ValueError("FIELD_SCORES_ARE_NOT_COMPLETIONS")
    score_files = args.field_scores
    if score_files is None:
        if args.outputs is None:
            score_files = discovered_scores
        else:
            score_files = []
            for path in raw_files:
                sibling = path.with_name("scores" + path.name[len("outputs") :])
                if path.name.lower().startswith("outputs") and sibling.exists():
                    score_files.append(sibling)
    score_files = [path.resolve() for path in score_files]
    questions = read_jsonl(question_path)
    outputs = [row for path in raw_files for row in read_jsonl(path)]
    field_scores = [row for path in score_files for row in read_jsonl(path)]
    outputs = join_field_scores(outputs, field_scores)
    result = score_records(
        questions,
        outputs,
        args.expected_k,
        args.stage,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    # A stage-specific rescore cannot overwrite the all-stage audit by accident.
    scoring_dir = root / "scoring"
    table_dir = root / "tables"
    if args.stage:
        scoring_dir = scoring_dir / args.stage
        table_dir = table_dir / args.stage
    scored_path = scoring_dir / "SCORED_OUTPUTS.jsonl"
    atomic_write(
        scored_path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in result["scored_outputs"]
        ),
    )
    receipt = result["tables"]["SCORING_RECEIPT"]
    receipt.update(
        {
            "question_file_sha256": sha256(question_path),
            "raw_file_sha256": [
                {
                    "name": str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                    "sha256": sha256(path),
                }
                for path in raw_files
            ],
            "field_score_file_sha256": [
                {
                    "name": str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
                    "sha256": sha256(path),
                }
                for path in score_files
            ],
            "field_score_records_joined": len(field_scores),
            "field_score_join": (
                "request identity and every raw completion field unchanged; "
                "status remains completed"
            ),
            "scored_outputs_sha256": sha256(scored_path),
            "implementation_sha256": sha256(
                Path(__file__).resolve().parents[2] / "src" / "mm_core" / "scoring.py"
            ),
            "contracts_sha256": sha256(
                Path(__file__).resolve().parents[2] / "src" / "mm_core" / "contracts.py"
            ),
        }
    )
    for name, report in result["tables"].items():
        atomic_write(
            table_dir / f"{name}.json",
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
        )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "scored_record_count": receipt["scored_record_count"],
                "format_gate_passed": result["tables"]["FORMAT_REPORT"]["gate_passed"],
                "tables_directory": str(table_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
