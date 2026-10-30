"""Bounded synthetic CPU storage comparison; never model or numerical admission.

The fixed workload exercises observed tiny and large tensor sizes, four retained
reads, and real fsync. Both backends receive identical bytes. No model forward,
generation, optimizer, or original experiment data is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch

from sr_f6_memory_repair.offload import saved_activation_offload

SIZES = (2004,) * 512 + (1024**2,) * 128 + (2 * 1024**2,) * 64 + (96 * 1024**2,)
PROJECT = Path("/projects/_ssd/varunssd/louis-ssvc")


def run(root: Path):
    if root != root.resolve() or not root.is_relative_to(PROJECT) or root == PROJECT:
        raise ValueError("New canonical project descendant required")
    root.mkdir(mode=0o700, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for backend in ("file_v2", "arena_v3"):
        model = torch.nn.Linear(1, 1)  # Storage inventory only: never called.
        counters = dict(open_calls=0, fsync_calls=0)
        original_open, original_fsync = os.open, os.fsync

        def counted_open(*args, counters=counters, original_open=original_open, **kwargs):
            counters["open_calls"] += 1
            return original_open(*args, **kwargs)

        def counted_fsync(*args, counters=counters, original_fsync=original_fsync, **kwargs):
            counters["fsync_calls"] += 1
            return original_fsync(*args, **kwargs)

        packed, expected = [], []
        with (
            patch("os.open", counted_open),
            patch("os.fsync", counted_fsync),
            saved_activation_offload(
                model,
                max_host_bytes=1,
                scratch_directory=root / backend,
                max_disk_bytes=1024**3,
                disk_backend=backend,
                fixture=True,
            ) as manager,
        ):
            start = time.monotonic()
            for index, size in enumerate(SIZES):
                value = torch.full((size // 4,), (index % 17) / 16, dtype=torch.float32)
                expected.append(hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest())
                packed.append(manager._pack(value))
                del value
            packing_seconds = time.monotonic() - start
            reads = []
            for _ in range(4):
                start = time.monotonic()
                for item, digest in zip(packed, expected, strict=True):
                    restored = manager._unpack(item)
                    actual = hashlib.sha256(memoryview(restored.numpy()).cast("B")).hexdigest()
                    if actual != digest:
                        raise RuntimeError("Synthetic exact-byte comparison failed")
                    del restored
                reads.append(time.monotonic() - start)
            del item
            packed.clear()
            gc.collect()
        summary = manager.summary()
        if (
            summary["live_disk_bytes"] != 0
            or summary["cleanup_errors"]
            or summary["storage_failures"]
            or any((root / backend).iterdir())
        ):
            raise RuntimeError("Synthetic storage lifecycle incomplete")
        rows.append(
            dict(
                backend=backend,
                packing_seconds=packing_seconds,
                read_seconds=reads,
                total_seconds=packing_seconds + sum(reads),
                counters=counters,
                storage=summary,
            )
        )
    report = dict(
        schema="srf6-synthetic-storage-benchmark-v1",
        status="SYNTHETIC_CPU_STORAGE_ONLY_NOT_MODEL_ADMISSION",
        records=len(SIZES),
        write_bytes=sum(SIZES),
        read_passes=4,
        data_sha256=hashlib.sha256(json.dumps(SIZES).encode()).hexdigest(),
        generated_responses=0,
        model_forwards=0,
        optimizer_updates=0,
        limitations="CPU copies only; excludes GPU transfers and model compute; one fixed order",
        results=rows,
        observed_speedup=rows[0]["total_seconds"] / rows[1]["total_seconds"],
    )
    with (root / "RESULT.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
