#!/usr/bin/env python3
"""Run the fixed 300-scene bridge on the reviewed offline GPU server."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from compbias.gpu_pilot.config import ACTIVE_PILOT_OUTPUT_SLUG, load_pilot_paths
from compbias.gpu_pilot.preflight import model_snapshot_sha256
from compbias.gpu_pilot.qwen_smoke import decode_qwen_once, load_local_qwen
from compbias.gpu_pilot.structured_generation import build_structured_messages
from compbias.recoverability.bridge import (
    BridgeScene,
    decode_text_qwen_once,
    run_bridge_protocol,
)
from compbias.recoverability.config import load_recoverability_protocol
from compbias.recoverability.evidence import verify_protocol_lock


def _read_scenes(dataset: Path) -> tuple[BridgeScene, ...]:
    records_path = dataset / "records.jsonl"
    scenes: list[BridgeScene] = []
    with records_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("split") != "iid_test":
                continue
            values = record.get("values")
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError("bridge values must contain four integers")
            scenes.append(
                BridgeScene(
                    scene_id=str(record["sample_id"]),
                    image_path=(dataset / str(record["image"])).resolve(),
                    question=str(record["question"]),
                    operation=str(record["operation"]),
                    values=tuple(values),  # type: ignore[arg-type]
                    answer=int(record["answer"]),
                )
            )
    if len(scenes) != 300:
        raise ValueError(f"bridge requires exactly 300 iid_test scenes, observed {len(scenes)}")
    if any(not scene.image_path.is_file() or scene.image_path.is_symlink() for scene in scenes):
        raise ValueError("bridge image is missing or is a symlink")
    return tuple(scenes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("BLOCKED: bridge inference requires explicit --execute on the reviewed GPU server")
        return 2
    if args.server_package_lock is None:
        print("BLOCKED: bridge inference requires the reviewed server package lock")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repository_root = Path(__file__).resolve().parents[2]
    verify_protocol_lock(args.server_package_lock, repository_root=repository_root)
    paths = load_pilot_paths(args.paths)
    protocol = load_recoverability_protocol(args.protocol)
    dataset = paths.data / "generated" / ACTIVE_PILOT_OUTPUT_SLUG
    output = paths.outputs / "recoverability_v1" / protocol.bridge.output_subdirectory
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite bridge output: {output}")
    output.mkdir(parents=True)
    scenes = _read_scenes(dataset)
    model_hash_before = model_snapshot_sha256(paths.model_path)
    model, processor = load_local_qwen(paths.model_path)

    def legacy(scene: BridgeScene) -> str:
        messages = build_structured_messages(
            question=scene.question,
            operation=scene.operation,
            retry_index=0,
            expected_value_count=4,
        )
        return decode_qwen_once(model, processor, scene.image_path, messages)

    report, records = run_bridge_protocol(
        scenes,
        legacy_generate=legacy,
        stage1_generate=lambda scene, messages: decode_qwen_once(
            model, processor, scene.image_path, messages
        ),
        stage2_generate=lambda _scene, messages: decode_text_qwen_once(model, processor, messages),
        equivalence_margin=0.03,
    )
    if model_snapshot_sha256(paths.model_path) != model_hash_before:
        raise RuntimeError("model snapshot changed during bridge inference")
    records_path = output / "bridge_records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True, allow_nan=False) + "\n")
    payload = {
        **asdict(report),
        "schema_version": 1,
        "artifact_type": "recoverability_v1_bridge_report",
        "dataset_id": protocol.bridge.dataset_id,
        "model_snapshot_sha256": model_hash_before,
        "format_retries": 0,
        "training_invoked": False,
    }
    (output / "bridge_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return (
        0 if report.stage1_parse_rate >= 0.98 and report.program_answer_consistency >= 0.95 else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
