"""Explicit Slurm CPU preflight and execution of the frozen N4 panel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .protocol import ROOT, cpu_environment, load_config, resource_gate


def require_cpu_allocation():
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("server N4 requires an explicit Linux Slurm CPU allocation")
    if not platform.node().split(".")[0].startswith("cpu-"):
        raise ValueError("server N4 must execute on a CPU compute node")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("server N4 refuses GPU allocations")


def server_forecast(budget, old_smoke, server_smoke):
    """Use the slower observed CPU rate; never shrink old cost projections."""
    ratios = [1.0]
    for key in ("observation_wall_seconds", "fit_wall_seconds"):
        if old_smoke.get(key, 0) <= 0 or server_smoke.get(key, 0) <= 0:
            raise ValueError("missing positive measured smoke time")
        ratios.append(server_smoke[key] / old_smoke[key])
    scale = max(ratios)
    result = dict(budget["forecast"])
    result["wall_seconds"] *= scale
    result["peak_ram_gib"] = max(result["peak_ram_gib"], server_smoke["forecast"]["peak_ram_gib"])
    result["server_time_scale"] = scale
    result["basis"] = "full frozen N4 panel; slower of historical and server CPU smoke rates"
    return result


def _json(path):
    return json.loads(Path(path).read_text())


def _source_catalog(old_lock, old_root):
    from .io import sha256_file

    paths = {ROOT / Path(p).relative_to(old_root) for p in old_lock["source_hashes"]}
    paths.update((ROOT / "src/modeling_contrast").glob("*.py"))
    paths.update((ROOT / "src").glob("*.py"))
    paths.add(ROOT / "requirements-cpu.txt")
    paths.update((ROOT / "configs/modeling_qualification").glob("*.json"))
    paths.update((ROOT / "docs/modeling_qualification/design").glob("*.json"))
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


def _library_versions():
    return {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "pytest")
    }


def _preflight_completion(settings, run_root):
    from .io import sha256_file

    stage = run_root / "N3_server"
    preflight = _json(stage / "SERVER_RESOURCE_PREFLIGHT.json")
    path = Path(settings["control_root"]) / f"preflight_{preflight['job_id']}" / "COMPLETE.json"
    receipt = _json(path)
    elapsed = receipt.get("wall_seconds")
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("selection_manifest_sha256") != sha256_file(stage / "RUN_MANIFEST.json")
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("missing or changed complete server preflight timing")
    return receipt


def preflight(config, config_path, settings, run_root, control):
    from .cli import register_tree, run_smoke
    from .collect_fresh import scan_seed_collisions, validate_fresh_gate
    from .frozen_validation import project_frozen_validation_budget
    from .io import RunWriter, canonical_hash, sha256_file
    from .server_checks import verify_parent_runtime_identity
    from .server_transfer import build_server_transfer, write_server_transfer

    parent = Path(settings["parent_root"])
    original_path = run_root / "N3/MODEL_SELECTION_LOCK.json"
    original = _json(original_path)
    roots = settings["server_seed_inventory_roots"]
    seeds = set(original["fresh_calibration_seeds"] + original["fresh_locked_test_seeds"])
    collisions = scan_seed_collisions(roots, seeds)
    if collisions:
        raise ValueError("STOP_FOR_REVIEW seed collisions: " + json.dumps(collisions))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/modeling_contrast",
        "-q",
        "-ra",
        "--junitxml",
        str(control / "tests.xml"),
        "--basetemp",
        str(Path(settings["temporary_root"]) / "fixtures/preflight"),
    ]
    with (control / "tests.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    identity = verify_parent_runtime_identity(parent, temporary_root=settings["temporary_root"])
    if identity["status"] != "PASS":
        raise ValueError("server dataset or initialization identity mismatch")
    smoke_out = run_root / "smoke_server_preflight"
    smoke_out.mkdir(exist_ok=False)
    smoke = run_smoke(config, parent, smoke_out)
    budget = project_frozen_validation_budget(config, original, run_root, parent_root=parent)
    forecast = server_forecast(budget, _json(run_root / "smoke/smoke_result.json"), smoke)
    gate = resource_gate(forecast, config)
    receipt = {
        "status": "PASS" if gate["passed"] else "RESOURCE_REVIEW_REQUIRED",
        "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": platform.node(),
        "python": sys.version,
        "library_versions": _library_versions(),
        "test_command": command,
        "test_log_sha256": sha256_file(control / "tests.log"),
        "identity": identity,
        "budget": budget,
        "forecast": forecast,
        "resource_gate": gate,
        "source_commit": settings["source_commit"],
        "new_optimizer_updates": 0,
        "new_qwen_calls": 0,
        "new_gpu_calls": 0,
    }
    (control / "PREFLIGHT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not gate["passed"]:
        raise ValueError("RESOURCE_REVIEW_REQUIRED: server CPU preflight")
    sources = _source_catalog(original, Path(settings["original_project_root"]))
    payload = build_server_transfer(
        original_path,
        prefix_mappings=settings["prefix_mappings"],
        original_project_root=settings["original_project_root"],
        amended_config=config,
        amendment_path=config_path,
        resource_forecast=forecast,
        source_hashes=sources,
        server_seed_inventory_roots=roots,
        original_source_prefix_mappings=settings["original_source_prefix_mappings"],
    )
    out = run_root / "N3_server"
    with RunWriter(out, payload["lock"]["binding"]) as writer:
        write_server_transfer(writer, payload)
        writer.write_json("SERVER_RESOURCE_PREFLIGHT.json", receipt)
        writer.write_json("SERVER_EXECUTION_SETTINGS.json", settings)
        writer.write_json("SERVER_RUNTIME_IDENTITY.json", identity)
        writer.write_bytes("SERVER_TESTS.log", (control / "tests.log").read_bytes())
        writer.write_bytes("SERVER_TESTS.xml", (control / "tests.xml").read_bytes())
        register_tree(writer)
    validate_fresh_gate(
        config,
        out / "MODEL_SELECTION_LOCK.json",
        existing_run_roots=roots,
        resource_forecast=forecast,
    )
    print(
        json.dumps(
            {
                "status": "SERVER_PREFLIGHT_PASS",
                "selection": str(out),
                "selected_sha256": canonical_hash(payload["lock"]["selected"]),
                "forecast": forecast,
            }
        ),
        flush=True,
    )


def execute(config, settings, run_root, control):
    from .cli import register_tree
    from .collect_fresh import run_collect_fresh
    from .frozen_validation import project_frozen_validation_budget, run_frozen_validation
    from .io import RunWriter, sha256_file, source_snapshot, verify_run_manifest
    from .server_checks import verify_fresh_identity

    stage = run_root / "N3_server"
    verify_run_manifest(stage)
    if _json(stage / "SERVER_EXECUTION_SETTINGS.json") != settings:
        raise ValueError("server execution settings changed after preflight")
    selection = stage / "MODEL_SELECTION_LOCK.json"
    lock = _json(selection)
    preflight = _json(stage / "SERVER_RESOURCE_PREFLIGHT.json")
    if preflight["python"] != sys.version or preflight["library_versions"] != _library_versions():
        raise ValueError("Python or numerical library environment changed after preflight")
    completion = _preflight_completion(settings, run_root)
    prior = preflight["forecast"]
    live = project_frozen_validation_budget(
        config, lock, run_root, parent_root=Path(settings["parent_root"])
    )["forecast"]
    forecast = dict(prior)
    for key in ("wall_seconds", "peak_ram_gib", "added_output_bytes", "temporary_bytes"):
        forecast[key] = max(prior[key], live[key])
    forecast["wall_seconds"] += completion["wall_seconds"]
    roots = settings["server_seed_inventory_roots"]
    collection = run_collect_fresh(
        config,
        selection,
        run_root / "N4",
        binding=lock["binding"],
        existing_run_roots=roots,
        resource_forecast=forecast,
    )
    identity = verify_fresh_identity(Path(settings["parent_root"]), run_root / "N4/raw")
    (control / "FRESH_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
    if identity["status"] != "PASS":
        raise ValueError("fresh dataset or initialization identity mismatch")
    with RunWriter(run_root / "N4_validation", lock["binding"]) as writer:
        source_snapshot(ROOT / "src/modeling_contrast", writer.out / "source_snapshot")
        writer.write_json("FRESH_IDENTITY.json", identity)
        validation = run_frozen_validation(
            config,
            selection,
            run_root / "N4/raw",
            run_root,
            writer,
            existing_run_roots=roots,
            resource_forecast=forecast,
        )
        register_tree(writer)
    summary = {
        "status": "COMPLETE",
        "collection": collection,
        "validation": validation,
        "identity_status": identity["status"],
        "selection_sha256": sha256_file(selection),
    }
    (control / "EXECUTION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "execute"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args(argv)
    require_cpu_allocation()
    cpu_environment()
    config_path = ROOT / "configs/modeling_contrast/protocol_server.json"
    config = load_config(config_path)
    settings = _json(args.settings)
    control = Path(settings["control_root"]) / f"{args.mode}_{os.environ['SLURM_JOB_ID']}"
    control.mkdir(parents=True, exist_ok=False)
    run_root = ROOT / "runs/modeling_contrast_v2"
    from .frozen_validation import _current_resources
    from .server_checks import RuntimeResourceWatchdog

    prior_wall = _current_resources(run_root)["wall_seconds"]
    if args.mode == "execute":
        prior_wall += _preflight_completion(settings, run_root)["wall_seconds"]
    with RuntimeResourceWatchdog(
        config, run_root, control, settings["temporary_root"], prior_wall_seconds=prior_wall
    ):
        if args.mode == "preflight":
            preflight(config, config_path, settings, run_root, control)
        else:
            execute(config, settings, run_root, control)
    from .io import sha256_file

    complete = {
        "status": "COMPLETE",
        "mode": args.mode,
        "wall_seconds": time.perf_counter() - started,
        "wall_scope": "entire server driver including preflight tests, identity, smoke and freeze",
        "selection_manifest_sha256": sha256_file(run_root / "N3_server/RUN_MANIFEST.json"),
    }
    (control / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
