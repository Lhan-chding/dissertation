"""Deterministic paired-scene workers and a model-free immutable D1 merge."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

from ..core import frozen_writer
from ..modeling_v3.io import source_identity
from ..modeling_v3.vlm_observation import digest
from ..modeling_v4 import gpu_collect as gpu
from .measurement import _namespace, bridge
from .reporting import _load_stream


def partition_prompts(prompts, workers):
    """Keep interfaces paired and distribute each family by original scene order."""
    if workers not in (1, 2, 3):
        raise ValueError("D1 supports one, two or three workers")
    if len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("Duplicate panel prompt")
    families = defaultdict(dict)
    for prompt in prompts:
        families[prompt["family"]].setdefault(prompt["base_scene_id"], []).append(prompt)
    assigned = {}
    for scenes in families.values():
        if len(scenes) % workers:
            raise ValueError("Workers must divide each family's paired scenes evenly")
        for index, scene in enumerate(scenes.values()):
            if len({p["interface"] for p in scene}) != 2 or len(scene) != 2:
                raise ValueError("Expected exactly two paired interfaces per scene")
            for prompt in scene:
                assigned[prompt["prompt_id"]] = index % workers
    return [[p for p in prompts if assigned[p["prompt_id"]] == i] for i in range(workers)]


def _contract(private, prompts, workers, stream_id):
    return {
        "schema": "decision-d1-parallel-v1",
        "workers": workers,
        "runtime_binding_hash": digest(private),
        "config_hash": private["config_hash"],
        "policies": private["preview_policies"],
        "origin_policy": private["preview_origin"],
        "panel_hash": digest(prompts),
        "prompt_ids": [p["prompt_id"] for p in prompts],
        "assignments": [
            [p["prompt_id"] for p in part] for part in partition_prompts(prompts, workers)
        ],
        "stream_id": stream_id,
        "implementation": source_identity()["sha256"],
        "draws_per_prompt": 32,
        "planned_total_rows": len(prompts) * 32 * 4,
    }


def run_worker(runtime, prompts, *, out, worker_index, workers, stream_id):
    private = runtime["decision_private"]
    contract = _contract(private, prompts, workers, stream_id)
    if not 0 <= worker_index < workers:
        raise ValueError("Worker index outside registered worker count")
    subset = partition_prompts(prompts, workers)[worker_index]
    # Original first prompt stays on worker zero, so the diagnostic is the same
    # fixed 24 draws as serial execution, independent of worker count.
    if worker_index == 0 and subset[0] != prompts[0]:
        raise ValueError("Fixed diagnostic prompt must stay on worker zero")
    root = Path(out)
    gpu._publish(
        root / "WORKER.json",
        {
            "contract": contract,
            "worker_index": worker_index,
            "prompt_ids": [p["prompt_id"] for p in subset],
            "namespace": _namespace(runtime),
        },
    )
    gpu._publish(
        root / "costs" / f"startup_{time.time_ns()}.json",
        {
            "scope": "PROCESS_STARTUP_OPERATIONAL_ONLY",
            "allocated_gpu": runtime.get("allocated_gpu"),
            "adapter_load_seconds": runtime["adapter"].audit.get("load_seconds"),
            "adapter_load_peak_cuda_bytes": runtime["adapter"].audit.get("load_peak_cuda_bytes"),
        },
    )
    result = bridge(
        runtime,
        private["preview_policies"],
        subset,
        out=root,
        stream_id=stream_id,
        origin_policy=private["preview_origin"],
        identity_prompts=prompts,
        run_scoring_diagnostic=worker_index == 0,
    )
    files = {
        str(path.relative_to(root)): gpu._binding(path)["sha256"]
        for path in sorted(root.rglob("*.json"))
        if path.name != "WORKER_COMPLETE.json" and "costs" not in path.parts
    }
    gpu._publish(
        root / "WORKER_COMPLETE.json",
        {
            "status": "D1_WORKER_COMPLETE",
            "contract_hash": digest(contract),
            "worker_index": worker_index,
            "files": files,
        },
    )
    return result


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError("Merged evidence link changed")
    elif target.exists():
        raise ValueError("Merged evidence must reference original immutable file")
    else:
        target.symlink_to(os.path.relpath(source, target.parent))


def merge_workers(private, prompts, *, root, out, workers, stream_id):
    """Validate all workers before publishing any combined, reportable evidence."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or out.is_relative_to(root / "workers") or root.is_relative_to(out):
        raise ValueError("Merged output must be separate from worker sources")
    contract = _contract(private, prompts, workers, stream_id)
    completed, worker_roots, namespaces = [], [], []
    links, receipts, all_sample_ids = [], {}, set()
    stream_rows = {name: {} for name in ("endpoint_0", "endpoint_1", "origin", "mix")}
    stream_identities = {}
    for index in range(workers):
        folder = root / "workers" / str(index)
        seal_path = folder / "WORKER_COMPLETE.json"
        if not seal_path.is_file():
            raise ValueError(f"Worker {index} is incomplete")
        seal = gpu._read(seal_path)
        if (seal.get("status"), seal.get("contract_hash"), seal.get("worker_index")) != (
            "D1_WORKER_COMPLETE",
            digest(contract),
            index,
        ):
            raise ValueError("Worker completion provenance mismatch")
        actual_files = {
            str(path.relative_to(folder))
            for path in folder.rglob("*.json")
            if path.name != "WORKER_COMPLETE.json" and "costs" not in path.parts
        }
        if set(seal["files"]) != actual_files:
            raise ValueError("Worker sealed file inventory changed")
        for relative, sha in seal["files"].items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder):
                raise ValueError("Worker file escapes source directory")
            gpu._verify_bound_file({"path": str(path), "sha256": sha})
        manifest, done = gpu._read(folder / "WORKER.json"), gpu._read(folder / "COMPLETE.json")
        expected_ids = contract["assignments"][index]
        if manifest["contract"] != contract or manifest["worker_index"] != index:
            raise ValueError("Worker contract/checkpoint provenance mismatch")
        if manifest["prompt_ids"] != expected_ids or done["status"] != "D1_MEASURED":
            raise ValueError("Worker panel or completion mismatch")
        if not done["lr_mass_enabled"]:
            raise ValueError("Worker LR disabled; full planned D1 is not complete")
        namespace = manifest["namespace"]
        if namespace["implementation"] != contract["implementation"]:
            raise ValueError("Worker implementation changed")
        namespaces.append(namespace)
        for name in stream_rows:
            endpoint = name.startswith("endpoint")
            sample_root = folder / name / "samples" if endpoint else folder / name
            # The same validator can read proposal chunks through their parent.
            if endpoint:
                identity, by_prompt = _load_stream(folder / name)
            else:
                identity = gpu._read(sample_root / "IDENTITY.json")
                by_prompt = {}
                for pid in expected_ids:
                    chunk = gpu._read(sample_root / pid / "0000_0032.json")
                    if chunk["identity"] != digest(identity) or chunk["rows_hash"] != digest(
                        chunk["rows"]
                    ):
                        raise ValueError("Proposal chunk hash mismatch")
                    by_prompt[pid] = chunk["rows"]
            if identity["prompts"] != contract["panel_hash"] or identity[
                "cache_namespace"
            ] != digest(namespace):
                raise ValueError("Worker stream runtime/panel provenance mismatch")
            expected_role = "endpoint" if endpoint else name
            expected_policies = (
                [contract["policies"][int(name[-1])]]
                if endpoint
                else [contract["origin_policy"]]
                if name == "origin"
                else contract["policies"]
            )
            forward_ids = [p.get("inference_fingerprint", digest(p)) for p in expected_policies]
            base_stream = (
                f"{stream_id}:endpoint:{name[-1]}" if endpoint else f"{stream_id}:{name.upper()}"
            )
            expected_stream = f"{base_stream}:{expected_role}:{digest(forward_ids)}"
            if (
                identity["policies"] != expected_policies
                or identity["role"] != expected_role
                or identity["stream_id"] != expected_stream
            ):
                raise ValueError("Worker stream checkpoint/RNG contract mismatch")
            if name in stream_identities and stream_identities[name] != identity:
                raise ValueError("Worker stream identity mismatch")
            stream_identities[name] = identity
            if set(by_prompt) != set(expected_ids):
                raise ValueError("Worker incomplete or overlapping prompt coverage")
            for pid, rows in by_prompt.items():
                if [row["draw_index"] for row in rows] != list(range(32)):
                    raise ValueError("Worker missing fixed 32-draw prefix")
                for row in rows:
                    if row["prompt_id"] != pid or row["rng_stream_id"] != identity["stream_id"]:
                        raise ValueError("Worker sample prompt/RNG mismatch")
                    expected_sample = digest(
                        [expected_stream, expected_role, forward_ids, pid, row["draw_index"]]
                    )
                    if row["sample_id"] != expected_sample or row["role"] != expected_role:
                        raise ValueError(
                            "Worker sample key differs from global serial RNG identity"
                        )
                    if row["sample_id"] in all_sample_ids:
                        raise ValueError("Duplicated sample across workers or roles")
                    all_sample_ids.add(row["sample_id"])
                stream_rows[name][pid] = rows
                target = out / name / ("samples" if endpoint else "") / pid / "0000_0032.json"
                links.append((sample_root / pid / "0000_0032.json", target))
            if endpoint:
                meta = gpu._read(folder / name / "LOOK_32.json")
                if (
                    meta["total_rows"] != len(expected_ids) * 32
                    or meta["panel_identity"] != contract["panel_hash"]
                ):
                    raise ValueError("Worker endpoint count/panel mismatch")
                if meta["sample_ids_hash"] != digest(
                    [row["sample_id"] for pid in expected_ids for row in by_prompt[pid]]
                ):
                    raise ValueError("Worker sample receipt mismatch")
                if name in receipts and any(
                    meta[k] != receipts[name][k]
                    for k in (
                        "candidate_id",
                        "execution_kind",
                        "role",
                        "look",
                        "origin_id",
                        "horizon",
                        "panel",
                    )
                ):
                    raise ValueError("Worker endpoint metadata mismatch")
                receipts[name] = meta
        completed.append(done)
        worker_roots.append(folder)
    if any(value != namespaces[0] for value in namespaces[1:]):
        raise ValueError("Workers have different runtime provenance")
    if len(all_sample_ids) != contract["planned_total_rows"]:
        raise ValueError("Full planned D1 sample budget not acquired")
    comparisons = []
    for proposal in ("ORIGIN", "MIX"):
        values = [
            next(x for x in done["proposal_comparisons"] if x["proposal"] == proposal)
            for done in completed
        ]
        units = [unit for value in values for unit in value["units"]]
        if (
            len(units) != len(prompts)
            or {u["prompt_id"] for u in units} != set(contract["prompt_ids"])
            or any(u["n"] != 32 for u in units)
        ):
            raise ValueError("Incomplete LR summary coverage")
        comparisons.append(
            {
                **values[0],
                "units": sorted(units, key=lambda u: contract["prompt_ids"].index(u["prompt_id"])),
            }
        )
    diagnostic = completed[0]["scoring_paths"]
    if diagnostic["tested_sequences"] != 24 or any(
        d["scoring_paths"]["tested_sequences"] != 0 for d in completed[1:]
    ):
        raise ValueError("Expected exactly one global fixed 24-action diagnostic")
    result = {
        "status": "D1_MEASURED",
        "total_rows": len(all_sample_ids),
        "planned_total_rows": contract["planned_total_rows"],
        "contract": contract,
        "scoring_paths": diagnostic,
        "lr_mass_enabled": True,
        "proposal_comparisons": comparisons,
        "new_training_steps": 0,
        "automatic_successor": False,
        "worker_receipts": [gpu._binding(p / "WORKER_COMPLETE.json") for p in worker_roots],
    }
    with frozen_writer(out):
        for source, target in links:
            _link(source, target)
        for name, identity in stream_identities.items():
            gpu._publish(
                out / name / ("samples" if name.startswith("endpoint") else "") / "IDENTITY.json",
                identity,
            )
        endpoints = []
        for name, meta in receipts.items():
            rows = [r for pid in contract["prompt_ids"] for r in stream_rows[name][pid]]
            value = {
                **meta,
                "total_rows": len(rows),
                "sample_ids_hash": digest([r["sample_id"] for r in rows]),
                "scoring_usable": all(r["scoring_usable"] for r in rows),
            }
            gpu._publish(out / name / "LOOK_32.json", value)
            endpoints.append(value)
        result["endpoints"] = endpoints
        for comparison in comparisons:
            gpu._publish(out / comparison["proposal"].lower() / "ANALYSIS.json", comparison)
        for index, folder in enumerate(worker_roots):
            for path in folder.glob("**/costs/*.json"):
                _link(path, out / "worker_costs" / str(index) / path.relative_to(folder))
        gpu._publish(out / "SCORING_PATHS.json", diagnostic)
        gpu._publish(out / "COMPLETE.json", result)
    return result
