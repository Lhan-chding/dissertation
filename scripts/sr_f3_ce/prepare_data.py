#!/usr/bin/env python3
"""Validate frozen SR-F3-CE sources and materialize all registered chart images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f3_ce.data import (
    ModelInputStore,
    compare_historical_worlds,
    load_frozen_dataset,
)
from sr_f3_ce.rendering import render_dataset, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-worlds", type=Path, action="append", default=[])
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args(argv)
    dataset = load_frozen_dataset(args.plan_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "FROZEN_DATA_AUDIT.json", dataset.audit)
    historical = compare_historical_worlds(dataset.tasks, args.historical_worlds)
    write_json(output / "HISTORICAL_WORLD_AUDIT.json", historical)
    if historical["status"] == "FAIL":
        raise SystemExit(
            "Historical world collision: preserved HISTORICAL_WORLD_AUDIT.json"
        )
    rendering = render_dataset(dataset, output, font_path=args.font_path)
    store = ModelInputStore.from_files(
        args.plan_dir / "manifests/MODEL_INPUTS.jsonl",
        output / "IMAGE_MANIFEST.jsonl",
        image_root=output,
    )
    for qid in dataset.inputs:
        store.model_input(qid)
    receipt = {
        "status": "CPU_DATA_AND_SOURCE_IMAGES_VERIFIED",
        "tasks": len(dataset.tasks),
        "historical_worlds": historical["status"],
        "rendering": rendering["status"],
        "model_input_routes_checked": len(dataset.inputs),
        "model_calls": 0,
        "native_processor": "NOT_RUN",
        "visual_spotchecks": "PENDING",
        "scientific_training_authorized_by_this_receipt": False,
    }
    write_json(output / "PREPARE_DATA_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
