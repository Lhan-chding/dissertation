"""Measured stage resources with bounded checks on the entire new result tree."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

from .protocol import ROOT, resource_gate


def disk_bytes(paths):
    return sum(p.stat().st_size for root in paths for p in Path(root).rglob("*") if p.is_file())


class StageResources:
    def __init__(self, config, run_root, out):
        self.config, self.run_root, self.out = config, Path(run_root), Path(out)
        self.start = time.perf_counter()
        self.prior_wall = 0.0
        for stage in self.run_root.glob("*/stage_result.json"):
            if stage.parent != self.out:
                self.prior_wall += json.loads(stage.read_text()).get("wall_seconds", 0.0)
        for stage in self.run_root.glob("smoke*/smoke_result.json"):
            result = json.loads(stage.read_text())
            self.prior_wall += result.get("observation_wall_seconds", 0.0) + result.get(
                "fit_wall_seconds", 0.0
            )
        self.peak_bytes = 0

    def check(self, label):
        measured = disk_bytes([self.run_root, ROOT / "docs/modeling_contrast/results"])
        self.peak_bytes = max(self.peak_bytes, measured)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        values = {
            "wall_seconds": self.prior_wall + time.perf_counter() - self.start,
            "peak_ram_gib": usage.ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20),
            "added_output_bytes": measured,
            "temporary_bytes": 0,
        }
        result = {
            "label": label,
            "measured": values,
            "gate": resource_gate(values, self.config),
            "process_user_cpu_seconds": usage.ru_utime,
            "process_system_cpu_seconds": usage.ru_stime,
            "disk_peak_bytes": self.peak_bytes,
            "wall_scope": (
                "completed formal stages and recorded smoke attempts plus current stage; "
                "excludes manual implementation idle time"
            ),
            "temporary_scope": (
                "no separate scratch arrays created by N2; "
                "compressed publication uses output directory"
            ),
            "ram_scope": "current process peak; sequential experiments; one numerical CPU thread",
        }
        with (self.out / "resource_checkpoints.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        if not result["gate"]["passed"]:
            raise ValueError("RESOURCE_REVIEW_REQUIRED: " + label)
        return result
