"""CPU-only V2 experiment commands. New training exists only behind N3 locks."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from .protocol import ROOT, cpu_environment, load_config, resource_gate


def footprint(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def binding(config_path, parent_root, *, packets=None):
    from .io import canonical_hash, sha256_file, source_hashes

    return {
        "source": canonical_hash(source_hashes(ROOT / "src/modeling_contrast")),
        "config": sha256_file(config_path),
        "data": sha256_file(parent_root / "manifest.json"),
        "selector": canonical_hash({}),
        "packet": canonical_hash(packets or {}),
    }


def register_tree(writer):
    for path in sorted(writer.out.rglob("*")):
        if path.is_file() and path.name not in ("RUN_MANIFEST.json", ".writer.lock"):
            relative = path.relative_to(writer.out)
            if str(relative) not in writer.output_hashes:
                writer.register_existing(relative)


def run_smoke(config, parent_root, out):

    from .model_study import baseline_configs, fit_unit
    from .observation_study import compare_unit, measure_unit
    from .study import METHODS, experiment_grid, fwrite, jwrite

    entry = next(
        e
        for e in json.loads((parent_root / "manifest.json").read_text())["trajectories"]
        if e["seed"] == 101 and e["arm"] == "X_BASE"
    )
    t = time.perf_counter()
    packets = {}
    paths = {}
    for method in METHODS:
        for n in (64, 16, 256):
            path = out / "packets" / f"{method}_n{n}"
            packets[(method, n)] = measure_unit(parent_root, entry, 0, method, n, path)
            paths[(method, n)] = path
    sampling_wall = time.perf_counter() - t
    comparison = compare_unit(parent_root, entry, 0, paths, out / "oracle")
    tfit = time.perf_counter()
    fit_unit(
        parent_root,
        entry,
        0,
        experiment_grid("N2B")
        + experiment_grid("N2C", ["O_LR_ORIGIN", "O_LR_MIX"])
        + baseline_configs(),
        packets,
        out / "fit",
    )
    fitting_wall = time.perf_counter() - tfit
    packet_bytes = footprint(out / "packets")
    fit_bytes = footprint(out / "fit")
    # N1:36 anchor states * n16/64/256. N2:24 additional selection
    # anchor states, same instruments; development packets are reused.
    # Sample length scaling is conservative for compressed repeated logp.
    forecast = {
        "wall_seconds": sampling_wall * 60 + fitting_wall * 110 + 1200,
        "peak_ram_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**30 if sys.platform == "darwin" else 2**20)
        + 1,
        "added_output_bytes": int(packet_bytes * 60 + fit_bytes * 110 + 300 * 2**20),
        "temporary_bytes": 350 * 2**20,
        "basis": (
            "real seed101 X_BASE anchor8 all three n measured; 60 finite anchor states; "
            "110 equivalent model sweeps including oracle isolation and robustness; "
            "1200s overhead"
        ),
    }
    result = {
        "status": "PASS",
        "execution_kind": "LEGACY_FIXED_CPU_POLICY_SMOKE",
        "observation_wall_seconds": sampling_wall,
        "fit_wall_seconds": fitting_wall,
        "packet_bytes": packet_bytes,
        "fit_bytes": fit_bytes,
        "forecast": forecast,
        "resource_gate": resource_gate(forecast, config),
        "new_training_updates": 0,
        "new_qwen_calls": 0,
        "new_gpu_calls": 0,
        "online_ssvc": False,
    }
    fwrite(out / "OBSERVATION_COMPARISON.csv", comparison)
    jwrite(out / "smoke_result.json", result)
    return result


def run_observations(config, parent_root, out):
    from .observation_study import compare_unit, measure_unit
    from .study import METHODS, fwrite, jwrite

    entries = [
        e
        for e in json.loads((parent_root / "manifest.json").read_text())["trajectories"]
        if e["seed"] in config["data_roles"]["observation_development_seeds"]
    ]
    rows = []
    start = time.perf_counter()
    for entry in entries:
        for ai in range(3):
            paths = {}
            for method in METHODS:
                for n in (64, 16, 256):
                    path = out / "packets" / entry["id"] / f"a{ai}" / f"{method}_n{n}"
                    measure_unit(parent_root, entry, ai, method, n, path)
                    paths[(method, n)] = path
            rows.extend(
                compare_unit(parent_root, entry, ai, paths, out / "oracle" / entry["id"] / f"a{ai}")
            )
            print(
                json.dumps(
                    {
                        "phase": "N1",
                        "id": entry["id"],
                        "anchor": [8, 24, 40][ai],
                        "wall_seconds": round(time.perf_counter() - start, 2),
                    }
                ),
                flush=True,
            )
    fwrite(out / "OBSERVATION_COMPARISON.csv", rows)
    result = {
        "status": "COMPLETE",
        "trajectories": len(entries),
        "anchors": 3 * len(entries),
        "methods": list(METHODS),
        "n": [16, 64, 256],
        "noise_replicas": 8,
        "wall_seconds": time.perf_counter() - start,
        "new_optimizer_updates": 0,
    }
    jwrite(out / "stage_result.json", result)
    return result


def main(argv=None):
    cpu_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "audit-parent",
            "smoke",
            "observation-study",
            "fit-study",
            "freeze-selection",
            "verify-fresh",
            "report",
        ],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/modeling_contrast_v2")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    from .io import RunWriter, source_snapshot
    from .study import legacy_root

    parent_root = (args.parent or legacy_root()).resolve()
    if args.command == "verify-fresh":
        from .collect_fresh import run_collect_fresh
        from .frozen_validation import project_frozen_validation_budget, run_frozen_validation

        if args.selection is None:
            parser.error("verify-fresh requires --selection pointing to the frozen N3 lock")
        lock = json.loads(args.selection.read_text())
        if lock.get("allow_fresh_cpu") is not True:
            raise ValueError("N3 did not authorize fresh CPU validation")
        budget = project_frozen_validation_budget(config, lock, args.root)
        if not budget["resource_gate"]["passed"]:
            raise ValueError("RESOURCE_REVIEW_REQUIRED: full fresh collection and validation")
        collection = run_collect_fresh(
            config,
            args.selection,
            args.out,
            binding=lock["binding"],
            existing_run_roots=lock["data_gate"].get(
                "fresh_seed_inventory_roots", [parent_root.parent]
            ),
            resource_forecast=budget["forecast"],
        )
        validation_out = args.out.with_name("N4_validation")
        with RunWriter(validation_out, lock["binding"]) as writer:
            source_snapshot(ROOT / "src/modeling_contrast", writer.out / "source_snapshot")
            validation = run_frozen_validation(
                config,
                args.selection,
                args.out / "raw",
                args.root,
                writer,
                existing_run_roots=[parent_root.parent],
                resource_forecast=budget["forecast"],
            )
            register_tree(writer)
        print(json.dumps({"collection": collection, "validation": validation}, ensure_ascii=False))
        return
    if args.command == "report":
        from .report import run_report

        print(json.dumps(run_report(config, args.root, args.out, parent_root), ensure_ascii=False))
        return
    if args.command == "freeze-selection":
        from .selection_driver import build_selection, write_selection

        payload = build_selection(config, parent_root, args.root)
        with RunWriter(args.out, payload["lock"]["binding"], resume=args.resume) as writer:
            result = write_selection(writer, payload)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    bind = binding(args.config, parent_root)
    with RunWriter(args.out, bind, resume=args.resume) as writer:
        if writer.resumed:
            if writer.previous_status == "COMPLETE":
                print(json.dumps({"status": "VERIFIED_EXISTING_STAGE", "out": str(writer.out)}))
                return
            raise ValueError(
                "interrupted stage preserved; inspect registered receipts before retrying"
            )
        writer.write_json(
            "execution_identity.json",
            {
                "argv": sys.argv,
                "python": sys.executable,
                "environment": cpu_environment(),
                "binding": bind,
                "parent_root": str(parent_root),
            },
        )
        source_snapshot(ROOT / "src/modeling_contrast", writer.out / "source_snapshot")
        if args.command == "smoke":
            result = run_smoke(config, parent_root, writer.out)
        elif args.command == "audit-parent":
            from .parent import audit_parent_collection

            worktree = ROOT.parent
            result = audit_parent_collection(
                parent_root,
                worktree,
                ROOT / "docs/modeling_contrast/design/evidence/parent_module_inventory.json",
                [worktree.parent.parent / "reports/SSVC_MODELING_PRO_ANALYSIS_20260915.zip"],
                [parent_root.parent],
            )
            writer.write_json("parent_integrity_audit.json", result)
            writer.write_json(
                "MISSING_INPUTS.json",
                {"missing": result["missing_inputs"], "mismatches": result["mismatches"]},
            )
            writer.write_text(
                "N0_AUDIT_zh.md",
                (
                    f"# N0 原件审计\n\n状态：{result['status']}。"
                    f"完整旧轨迹 {result['complete_trajectories']}/60。\n"
                    "未新增训练。详见 parent_integrity_audit.json。\n"
                ),
            )
        elif args.command == "observation-study":
            smoke = json.loads((args.root / "smoke/smoke_result.json").read_text())
            if not smoke["resource_gate"]["passed"]:
                raise ValueError("RESOURCE_REVIEW_REQUIRED")
            result = run_observations(config, parent_root, writer.out)
        elif args.command == "fit-study":
            from .pipeline import run_fits

            result = run_fits(config, parent_root, args.root, writer.out)
        elif args.command == "freeze-selection":
            from .pipeline import run_selection

            result = run_selection(config, parent_root, args.root, writer)
        register_tree(writer)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
