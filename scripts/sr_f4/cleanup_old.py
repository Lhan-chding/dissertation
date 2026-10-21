"""User-authorized removal of obsolete resume states and duplicate caches.

Only completed F1.2/F3 run intermediates and known CPU fixture directories are
eligible. Final checkpoints/adapters, models, raw committed groups and scores
are retained. No hashes or re-analysis are required.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/projects/_ssd/varunssd/louis-ssvc")


def candidates():
    values = []
    for relative, last in [
        ("sr_f12_20261010", 96),
        ("sr_f3_ce_protocol_fix_20261016/selected_v2", 64),
    ]:
        for run in (BASE / relative / "runs").glob("*"):
            if not (run / "COMPLETE.json").is_file():
                continue
            # Keep endpoint, all exported adapters, all completion/incident logs.
            for path in (run / "checkpoints").glob("step-*.pt"):
                match = re.match(r"step-(\d+)-", path.name)
                if match and int(match[1]) < last:
                    values.append(
                        (path, "obsolete completed-run intermediate optimizer/checkpoint")
                    )
            for path in (run / "rollouts").glob("*.pending.jsonl"):
                complete = path.with_name(path.name.removesuffix(".pending.jsonl") + ".json")
                if complete.is_file():
                    values.append(
                        (path, "duplicate raw pending cache; complete scored group retained")
                    )
    parent = BASE / "sr_f3_ce_protocol_fix_20261016"
    for folder in parent.glob("*_deployment_*"):
        path = folder / "cpu_validation_tmp"
        if path.is_dir():
            values.append((path, "obsolete synthetic CPU test workspace"))
    for name in ("conda-pkgs", "torchinductor"):
        path = BASE / "cache" / name
        if path.is_dir():
            values.append((path, "rebuildable download/compiler cache"))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    queue = subprocess.check_output(["squeue", "-h", "-u", "varun024", "-o", "%i|%j"], text=True)
    if queue.strip():
        raise RuntimeError(
            "Run this scoped cleanup before new GPU work; user still has active jobs"
        )
    plan = []
    for path, reason in candidates():
        resolved = path.resolve(strict=True)
        hdd_checkpoints = Path("/projects/_hdd/varunhdd/louis-ssvc/sr_f12_20261010_checkpoints")
        allowed = resolved.is_relative_to(BASE) or (
            path.suffix == ".pt" and resolved.is_relative_to(hdd_checkpoints)
        )
        if path.is_symlink() or not allowed or resolved in (BASE, hdd_checkpoints):
            raise ValueError("Unsafe cleanup path")
        size = (
            path.stat().st_size
            if path.is_file()
            else sum(
                p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()
            )
        )
        plan.append(dict(path=str(path), canonical_path=str(resolved), bytes=size, reason=reason))
    receipt = dict(
        at=datetime.now(timezone.utc).isoformat(),
        status="PLANNED",
        files=plan,
        bytes=sum(r["bytes"] for r in plan),
        hash_checks=0,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.apply:
        for row in plan:
            path = Path(row["path"])
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        receipt["status"] = "DELETED"
        receipt["remaining"] = sum(Path(r["path"]).exists() for r in plan)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
