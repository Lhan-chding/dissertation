"""CPU diagnostics for immutable collected maps with separate analysis provenance.

This reader does not change collection/fitting source guards or write campaign
receipts. It can report B or C maps collected by an older frozen source tree.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

from ..modeling_v3.io import atomic_json, canonical_hash, sha256_file, source_identity
from .analysis_rules import load_analysis_rules
from .data_adapter import bound_json
from .gpu_collect import _chunk_rows, _verify_bound_file, pair_observation_diagnostics
from .response_fit import _require_server_cpu


class VerifiedReceiptReader:
    """Bound row memory; verify file identity on every hit and bytes only once."""

    def __init__(self, campaign, *, max_cached_receipts=2):
        self.root = Path(campaign).resolve() / "tasks"
        self.cache = OrderedDict()
        self.max_cached_receipts = max_cached_receipts
        self.input_index = {}

    def rows(self, binding):
        expected = 0
        for chunk in binding["chunks"]:
            if (
                not Path(chunk["path"]).resolve().is_relative_to(self.root)
                or chunk["start"] != expected
                or chunk["stop"] <= expected
            ):
                raise ValueError("Input shard path or contiguous coverage differs")
            _verify_bound_file(chunk)
            expected = chunk["stop"]
        if expected != binding["count"]:
            raise ValueError("Complete shard count differs")
        key = canonical_hash(binding)
        if key not in self.cache:
            records = []
            for chunk in binding["chunks"]:
                part = _chunk_rows(chunk["path"])
                if len(part) != chunk["stop"] - chunk["start"]:
                    raise ValueError("Shard row count differs")
                records.extend(part)
                self.input_index[chunk["path"]] = {"sha256": chunk["sha256"], "rows": len(part)}
            self.cache[key] = records
            while len(self.cache) > self.max_cached_receipts:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return iter(self.cache[key])


def analyze_maps(tasks, *, task_ids, collection_code, out):
    _require_server_cpu()
    tasks, out = Path(tasks).resolve(), Path(out).resolve()
    plan = json.loads(tasks.read_text())
    if (
        canonical_hash({k: v for k, v in plan.items() if k != "task_list_hash"})
        != plan["task_list_hash"]
    ):
        raise ValueError("Frozen task list hash differs")
    rules = plan.get("analysis_rules")
    if (
        not isinstance(rules, dict)
        or canonical_hash(rules) != plan.get("analysis_rules_hash")
        or rules != load_analysis_rules()
    ):
        raise ValueError("Frozen analysis rules or registered scales differ")
    collection_source = source_identity(collection_code)
    if collection_source != plan["source"]:
        raise ValueError("Frozen collection source differs from registered source")
    campaign = Path(plan["root"]).resolve()
    if out == campaign or out.is_relative_to(campaign):
        raise ValueError("Analysis output must be outside the immutable campaign")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty analysis directory")
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Unique completed task IDs required")
    registered = {t["id"]: t for t in plan["tasks"]}
    completed = []
    for task_id in task_ids:
        task = registered[task_id]
        path = campaign / "tasks" / task_id / "COMPLETE.json"
        receipt = json.loads(path.read_text())
        if (
            receipt.get("status") != "COMPLETED"
            or receipt.get("task") != task
            or receipt.get("execution_kind") != "REAL_CUDA_MODEL"
            or receipt.get("source_hash") != collection_source["sha256"]
            or receipt.get("config_hash") != canonical_hash(plan["config"])
            or task["kind"] not in ("bridge", "map")
        ):
            raise ValueError("Completed map/collection identity differs")
        response = bound_json(receipt["response"])
        if response["origin_id"] != receipt["origin_id"]:
            raise ValueError("Response origin differs from completed task")
        completed.append((task_id, path, receipt, response))
    analysis_source = source_identity()
    out.mkdir(parents=True, exist_ok=True)
    results, input_index = {}, {}
    for task_id, path, receipt, response in completed:
        reader = VerifiedReceiptReader(campaign)
        diagnostics = pair_observation_diagnostics(response, row_reader=reader.rows)
        input_index.update(reader.input_index)
        result = {
            "task_id": task_id,
            "completed_task": {"path": str(path), "sha256": sha256_file(path)},
            "response": receipt["response"],
            "collection_source_hash": collection_source["sha256"],
            "analysis_source_hash": analysis_source["sha256"],
            "diagnostics": diagnostics,
            "model_calls": 0,
            "new_generated_samples": 0,
            "scientific_status": "NOT_CERTIFIED",
        }
        destination = out / f"{task_id}_PAIR_DIAGNOSTICS.json"
        atomic_json(destination, result)
        results[task_id] = {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "all_finite": diagnostics["all_finite"],
        }
    metadata = {
        "status": "CPU_MEASUREMENT_DIAGNOSTICS_COMPLETE_NOT_CERTIFIED",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "task_list": {"path": str(tasks), "sha256": sha256_file(tasks)},
        "collection_source": collection_source,
        "analysis_source": analysis_source,
        "source_guards_bypassed": False,
        "original_receipts_modified": False,
        "fitting_performed": False,
        "model_calls": 0,
        "results": results,
        "input_shards": input_index,
    }
    atomic_json(out / "COMPLETE.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--collection-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze_maps(
        args.tasks, task_ids=args.task_id, collection_code=args.collection_code, out=args.out
    )


if __name__ == "__main__":
    main()
