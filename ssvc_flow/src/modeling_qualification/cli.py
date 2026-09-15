"""Offline CPU qualification commands. No model hub or scheduler entry points."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

from .io import canonical_hash, new_output, sha256_file, source_hashes, write_json
from .protocol import ROOT, cpu_environment, load_config, validate_config


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _preflight(config: dict, out: Path) -> dict:
    path = out.parent / "preflight.json"
    if not path.is_file():
        raise ValueError("RESOURCE_REVIEW_REQUIRED: measured smoke preflight.json is required")
    value = json.loads(path.read_text())
    if value.get("status") != "PASS" or value.get("config_hash") != canonical_hash(config):
        raise ValueError("RESOURCE_REVIEW_REQUIRED: preflight/config mismatch")
    limits = config["resources"]
    if (
        value["estimated_total_wall_seconds"] > limits["estimated_core_walltime_limit_hours"] * 3600
        or value["estimated_peak_ram_gib"] > limits["peak_ram_limit_gib"]
        or value["estimated_total_output_bytes"] > limits["output_limit_gib"] * 1024**3
    ):
        raise ValueError("RESOURCE_REVIEW_REQUIRED: measured CPU projection exceeds budget")
    if not value.get("measurement_sources"):
        raise ValueError("RESOURCE_REVIEW_REQUIRED: missing smoke measurements")
    for relative, digest in value["measurement_sources"].items():
        source = (out.parent / relative).resolve()
        if not source.is_relative_to(out.parent.resolve()) or sha256_file(source) != digest:
            raise ValueError("RESOURCE_REVIEW_REQUIRED: changed smoke measurement")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "validate",
            "math",
            "witnesses",
            "collect",
            "fit-evaluate",
            "track",
            "audit-real",
            "report",
        ],
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/modeling_qualification/protocol.json"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--profile", choices=["smoke", "core"], default="smoke")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--readonly-root", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    overrides = cpu_environment()
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "config_hash": canonical_hash(config),
                    "budget_derived": validate_config(config),
                    "environment": overrides,
                }
            )
        )
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.out.exists():
        parser.error("output already exists; preserve it and use a new run path")
    if args.command in ("fit-evaluate", "track", "report") and args.input is None:
        parser.error("--input is required")
    if args.command == "track" and args.models is None:
        parser.error("--models is required")
    if args.command == "audit-real" and args.readonly_root is None:
        parser.error("--readonly-root is required")
    if args.command == "collect" and args.profile == "core":
        _preflight(config, args.out)
    receipt_dir = new_output(args.out.parent / ".receipts" / f"{args.out.name}_{time.time_ns()}")
    source_root = ROOT / "src/modeling_qualification"
    before = source_hashes(source_root)
    with tarfile.open(receipt_dir / "source_snapshot.tar.gz", "w:gz") as archive:
        for relative in before:
            archive.add(source_root / relative, arcname=relative, recursive=False)
    phase_modules = {
        "math": ["math_contracts.py"],
        "witnesses": ["math_contracts.py", "witnesses.py"],
        "collect": ["toy.py", "collection.py"],
        "fit-evaluate": [
            "toy.py",
            "collection.py",
            "math_contracts.py",
            "models.py",
            "evaluation.py",
        ],
        "track": [
            "toy.py",
            "collection.py",
            "math_contracts.py",
            "models.py",
            "evaluation.py",
            "tracking.py",
        ],
        "audit-real": ["real_audit.py"],
        "report": list(before),
    }
    execution_files = {
        "__init__.py",
        "cli.py",
        "protocol.py",
        "io.py",
        *phase_modules[args.command],
    }
    receipt = {
        "execution_source_files": sorted(execution_files),
        "command": [
            sys.executable,
            "-m",
            "src.modeling_qualification.cli",
            *(sys.argv[1:] if argv is None else argv),
        ],
        "started_unix": time.time(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "source_hashes_before": before,
        "source_snapshot_sha256": sha256_file(receipt_dir / "source_snapshot.tar.gz"),
        "config_hash": canonical_hash(config),
        "config_file_sha256": sha256_file(args.config),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": overrides,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "torch", "pytest", "matplotlib"]
        },
        "new_qwen_calls": False,
        "new_gpu_started": False,
        "online_ssvc_started": False,
    }
    write_json(receipt_dir / "invocation.json", receipt)
    write_json(receipt_dir / "resolved_config.json", config)
    start = time.perf_counter()
    try:
        if args.command == "math":
            from .math_contracts import run_math

            result = run_math(config, args.out)
        elif args.command == "witnesses":
            from .witnesses import run_witnesses

            result = run_witnesses(config, args.out)
        elif args.command == "collect":
            from .collection import run_collect

            result = run_collect(config, args.profile, args.out)
        elif args.command == "fit-evaluate":
            from .evaluation import run_fit_evaluate

            result = run_fit_evaluate(config, args.input, args.out)
        elif args.command == "track":
            from .tracking import run_track

            result = run_track(config, args.input, args.models, args.out)
        elif args.command == "audit-real":
            from .real_audit import run_audit_real

            result = run_audit_real(args.readonly_root, args.out)
        else:
            from .report import run_report

            result = run_report(config, args.input, args.out)
        receipt["exit_code"] = 0
        receipt["result"] = result
    except Exception as error:
        receipt["exit_code"] = 1
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
        (receipt_dir / "traceback.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        receipt["wall_seconds"] = time.perf_counter() - start
        # Darwin ru_maxrss is bytes; Linux reports KiB.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["peak_rss_gib"] = rss / 1024 ** (3 if sys.platform == "darwin" else 2)
        receipt["source_hashes_after"] = source_hashes(ROOT / "src/modeling_qualification")
        receipt["sources_unchanged_during_run"] = before == receipt["source_hashes_after"]
        receipt["execution_sources_unchanged_during_run"] = all(
            before.get(name) == receipt["source_hashes_after"].get(name) for name in execution_files
        )
        if args.out.is_dir():
            receipt["output_hashes"] = source_hashes(args.out)
            receipt["output_bytes"] = sum(
                p.stat().st_size for p in args.out.rglob("*") if p.is_file()
            )
        write_json(receipt_dir / "result.json", receipt)
    print(
        json.dumps(
            {
                "exit_code": receipt["exit_code"],
                "receipt": str(receipt_dir),
                "wall_seconds": receipt["wall_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
