"""Aggregate next-stage status records without inventing missing measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import file_hash


def build_report(run_root):
    root = Path(run_root)
    statuses = []
    for path in sorted(root.rglob("status.json")):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            if status.get("phase", "").startswith("R"):
                statuses.append(status)
        except json.JSONDecodeError:
            statuses.append(
                {"phase": path.parent.name, "status": "FAIL", "reason": "malformed status"}
            )
    # Keep the newest record for each phase/track while preserving N and L as
    # separate audited endpoints.
    latest = {}
    for status in statuses:
        key = (
            status.get("phase"),
            status.get("details", {}).get("track", ""),
            status.get("details", {}).get("arm", ""),
        )
        if status.get("recorded_at", "") >= latest.get(key, {}).get("recorded_at", ""):
            latest[key] = status
    statuses = sorted(
        latest.values(),
        key=lambda s: (s.get("phase", ""), s.get("details", {}).get("track", "")),
    )
    tracked_phases = {s.get("phase") for s in statuses if s.get("details", {}).get("track")}
    statuses = [
        s
        for s in statuses
        if not (s.get("phase") in tracked_phases and not s.get("details", {}).get("track"))
    ]
    decisions = {
        "p3_audit": "INCONCLUSIVE_R0_TOKEN_IMAGE_AND_LEAKAGE_CHECKS_PENDING",
        "lambda_response": "NOT_MEASURED_R1_R3_BLOCKED",
        "x_valid_vs_x_base": "NOT_MEASURED_R4_BLOCKED",
        "next_bottleneck": "complete_R0_pending_checks_then_run_R1_real_CUDA_smoke",
    }
    return statuses, decisions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=Path("runs/NEXT_20260909"))
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    statuses, decisions = build_report(a.run_root)
    root = Path(a.run_root)
    out = a.out or a.run_root
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "actual_stage_statuses": statuses,
        "decisions": decisions,
        "execution_boundary": "No REAL_CUDA result was created locally",
    }
    (out / "NEXT_RESULTS_SUMMARY_zh.md").write_text(
        "# 下一阶段结果摘要\n\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "unresolved_items.json").write_text(
        json.dumps(
            {
                **decisions,
                "items": [
                    "R0 tokenizer/template token audit and 36 image spot checks",
                    "R1 real CUDA likelihood/cache/MLP-LoRA smoke",
                    "R2 real CUDA diagnostic rollouts",
                    "R3 cold and warm real optimizer forks",
                    "R4 explicit allow-training two-arm pilot",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {".writer.lock"}:
            hashes[str(path.relative_to(root))] = {
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
    (out / "all_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    (out / "run_costs.csv").write_text(
        "stage,execution_kind,new_generation_outputs,status\n"
        "R0,CPU_AUDIT,0,raw ledger audit only\n"
        "R1,CPU_AUDIT,0,BLOCKED_REAL_CUDA_REQUIRED\n"
        "R2,CPU_AUDIT,0,BLOCKED_REAL_CUDA_REQUIRED\n"
        "R3-cold,CPU_AUDIT,0,BLOCKED_REAL_CUDA_REQUIRED\n"
        "R4,CPU_AUDIT,0,BLOCKED_ALLOW_TRAINING_AND_REAL_CUDA_REQUIRED\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
