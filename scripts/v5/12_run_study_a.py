#!/usr/bin/env python3
"""Run the frozen, no-training v5 Study A Base/T audit on one offline GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.qwen.study_a_runtime import (  # noqa: E402
    RAW_ARCHIVE_SHA256,
    load_gpu_checkpoint,
    load_natural_errors,
    require_study_a_authorization,
    require_t_adapter,
    run_phase2a_study_a,
    sha256_file,
)

OUTPUT_ROOT = ROOT / "artifacts/v5/audits/study_a_4090_pilot"
WORK_ROOT = ROOT / "artifacts/v5/audits_work/study_a_4090_pilot"
CAPTURE_WORK_ROOT = ROOT / "artifacts/v5/audits_work/phase2a_observation_capture"
PHASE2A_ROOT = ROOT / "artifacts/v5/data/factorial_pre_model"
CHILD_ROOT = ROOT / "artifacts/v5/data/phase2a_natural_observations"
ALLOWED_OUTPUT_ROOT = ROOT / "artifacts/v5/audits"
ALLOWED_WORK_ROOT = ROOT / "artifacts/v5/audits_work"
ALLOWED_DATA_ROOT = ROOT / "artifacts/v5/data"
K = 8
SAMPLING_SEED = 2026082101


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--phase2a-root", type=Path, default=PHASE2A_ROOT)
    parser.add_argument("--child-root", type=Path, default=CHILD_ROOT)
    parser.add_argument("--raw-archive", type=Path)
    parser.add_argument("--t-adapter", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    return parser


def _within(path: Path, root: Path, label: str) -> Path:
    target = path.resolve(strict=False)
    try:
        target.relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"{label} must remain below {root}") from error
    cursor = path.parent
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError(f"{label} parent must not be a symlink: {cursor}")
        if cursor.resolve(strict=False) == root.resolve(strict=False):
            break
        cursor = cursor.parent
    return target


def _require_gpu_runtime() -> None:
    missing: list[str] = []
    for package in ("torch", "transformers", "peft"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError("Study A GPU runtime is missing: " + ", ".join(missing))
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Study A requires a CUDA GPU with bf16 support")


def main() -> int:
    arguments = _parser().parse_args()
    try:
        require_study_a_authorization(
            execute=arguments.execute,
            acknowledgement=arguments.ack,
        )
        output_root = _within(arguments.output_root, ALLOWED_OUTPUT_ROOT, "Study A output")
        work_root = _within(arguments.work_root, ALLOWED_WORK_ROOT, "Study A work trace")
        capture_work_root = _within(
            CAPTURE_WORK_ROOT,
            ALLOWED_WORK_ROOT,
            "Phase-2a capture trace",
        )
        child_root = _within(arguments.child_root, ALLOWED_DATA_ROOT, "Phase-2a child output")
        diagnostic_errors = None
        raw_digest = None
        if arguments.raw_archive is not None:
            diagnostic_errors, raw_digest = load_natural_errors(
                arguments.raw_archive,
                expected_sha256=RAW_ARCHIVE_SHA256,
            )
            print(
                "DIAGNOSTIC_SOURCE_VERIFIED: "
                f"legacy_v4_natural_errors={len(diagnostic_errors)} sha256={raw_digest}",
                flush=True,
            )
        require_t_adapter(arguments.t_adapter)
        _require_gpu_runtime()
        loader = lambda checkpoint: load_gpu_checkpoint(  # noqa: E731
            checkpoint,
            t_adapter=arguments.t_adapter,
        )
        summary = run_phase2a_study_a(
            phase2a_root=arguments.phase2a_root,
            child_root=child_root,
            capture_work_root=capture_work_root,
            output_root=output_root,
            audit_work_root=work_root,
            checkpoint_loader=loader,
            legacy_errors=diagnostic_errors,
            legacy_raw_archive_sha256=raw_digest,
            expected_parent_count=96,
            k=K,
            sampling_seed=SAMPLING_SEED,
            progress=lambda checkpoint, complete, total: print(
                f"PROGRESS: {checkpoint} {complete}/{total} checkpoint-scenarios complete",
                flush=True,
            ),
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print("READY: v5 Study A Base/T inference audit atomically published")
    print(f"FROZEN_SCENES {child_root / 'frozen_scenes.jsonl'}")
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    names = ["per_scenario.jsonl", "raw_trace.jsonl", "summary.json", "manifest.json"]
    for optional in (
        "phase2a_per_scenario.jsonl",
        "phase2a_summary.json",
        "phase2a_enriched_frozen_scenes.jsonl",
        "legacy_independent_per_scenario.jsonl",
        "legacy_independent_summary.json",
    ):
        if (output_root / optional).is_file():
            names.append(optional)
    for name in names:
        path = output_root / name
        print(f"SHA256 {sha256_file(path)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
