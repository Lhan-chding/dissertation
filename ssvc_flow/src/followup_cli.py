"""Non-executing planning/auditing and separately gated server entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .followup_protocol import (
    ARMS,
    BANK_IDS,
    PROJECT_ROOT,
    build_plan,
    canonical_hash,
    load_design,
    render_slurm,
    source_files,
    strict_json,
    validate_design,
    write_new_json,
)


def _smoke_gate(plan, evidence_path):
    if not evidence_path:
        raise ValueError("A measured GPU smoke evidence file is required")
    evidence = strict_json(evidence_path)
    if (
        evidence.get("status") != "PASS"
        or evidence.get("execution_kind") != "REAL_CUDA_FOLLOWUP_SMOKE"
        or evidence.get("gpu_smoke_passed") is not True
        or evidence.get("plan_hash") != canonical_hash(plan)
        or evidence.get("source_files") != source_files()
        or evidence.get("full_origin_restored") is not True
        or evidence.get("frozen_base_unchanged") is not True
        or evidence.get("smoke_optimizer_updates") != 4
        or evidence.get("new_rollouts") != 0
        or evidence.get("gpu_started") is not True
        or set(evidence.get("checks", {})) != {"0.0", "1.0"}
        or any(check.get("passed") is not True for check in evidence.get("checks", {}).values())
    ):
        raise ValueError("GPU smoke has not passed for this exact plan and source")
    return evidence


def _s1_gate(plan, path):
    if not path:
        raise ValueError("S2 requires completed S1 measurement-chain evidence")
    report = strict_json(path)
    if (
        report.get("status") != "MEASURED"
        or report.get("execution_kind") != "REAL_CUDA_FOLLOWUP"
        or report.get("s1_measurement_chain_passed") is not True
        or report.get("design_hash") != canonical_hash(plan["design"])
        or set(report.get("bank_ids", [])) != set(BANK_IDS)
    ):
        raise ValueError("S1 measurement chain has not passed all six banks")
    # Recompute the evidence bundle so a hand-edited PASS cannot grant access.
    rebuilt = inspect_run(Path(path).parent)
    if canonical_hash(rebuilt) != canonical_hash(report):
        raise ValueError("S1 measurement evidence no longer matches its bank bundles")
    identity = report["common_identity"]
    expected = {
        **plan["identity"],
        "design_hash": canonical_hash(plan["design"]),
        "validated_plan_hash": canonical_hash(plan),
        "source_hash": canonical_hash(plan["source_files"]),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("S1 identity is not bound to this exact validated plan/source/origin")
    return report


def _verify_manifest(root, manifest):
    from .followup_protocol import file_hash

    files = manifest.get("files")
    if isinstance(files, dict):
        files = [
            {"path": name, **(value if isinstance(value, dict) else {"sha256": value})}
            for name, value in files.items()
        ]
    if not isinstance(files, list) or not files:
        raise ValueError("Empty or malformed evidence manifest")
    seen = set()
    for item in files:
        name = item["path"]
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name in seen
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("Evidence manifest escapes root or duplicates a file")
        seen.add(name)
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Evidence file changed: {name}")


def inspect_run(run_root):
    """Read six immutable bank bundles and require one complete common identity."""
    from .followup_protocol import file_hash

    root = Path(run_root).resolve()
    summaries, manifests, kinds, hashes = {}, {}, set(), set()
    common_identity = None
    identity_keys = (
        "execution_kind",
        "design_hash",
        "origin_state_hash",
        "model_hash",
        "data_hash",
        "config_hash",
        "parser_version_hash",
        "protocol_version",
        "source_hash",
        "validated_plan_hash",
        "control_panel_hash",
    )
    for bank in BANK_IDS:
        bank_root = root / bank
        manifest_path = bank_root / "manifest.json"
        status = strict_json(bank_root / "status.json")
        identity = strict_json(bank_root / "identity.json")
        if status.get("status") not in ("MEASURED", "CPU_TESTED"):
            raise ValueError(f"Bank {bank} is not completed")
        _verify_manifest(bank_root, strict_json(manifest_path))
        common = {key: identity.get(key) for key in identity_keys}
        if identity.get("bank_id") != bank:
            raise ValueError(f"Bank identity binding mismatch: {bank}")
        if common_identity is None:
            common_identity = common
        elif common != common_identity:
            raise ValueError("Banks disagree on model/data/parser/origin/source identity binding")
        kinds.add(identity.get("execution_kind", status.get("execution_kind")))
        hashes.add(identity.get("design_hash"))
        summaries[bank] = strict_json(bank_root / "paired_response.json")
        manifests[bank] = {
            "path": str(manifest_path.relative_to(root)),
            "sha256": file_hash(manifest_path),
        }
    if len(kinds) != 1 or len(hashes) != 1:
        raise ValueError("Banks disagree on design or execution kind")
    real = kinds == {"REAL_CUDA_FOLLOWUP"}
    if real and any(not value for value in common_identity.values()):
        raise ValueError("Real S1 bank identity binding is incomplete")
    report = {
        "status": "MEASURED" if real else "CPU_TESTED",
        "execution_kind": next(iter(kinds)),
        "design_hash": next(iter(hashes)),
        "bank_ids": list(BANK_IDS),
        "s1_measurement_chain_passed": real,
        "safety_status": "NOT_CERTIFIED",
        "responses": summaries,
        "bank_manifests": manifests,
        "common_identity": common_identity,
    }
    return report


def analyze_run(run_root, out):
    """Derive S1 readiness from six measured immutable bank bundles, never flags."""
    root = Path(run_root).resolve()
    destination = Path(out).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    report = inspect_run(root)
    # Keep the immutable authorization record beside the referenced bank roots.
    write_new_json(root / "s1_measurement_chain.json", report)
    destination.mkdir(parents=True, exist_ok=False)
    write_new_json(destination / "analysis.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Freeze design and logical budgets without loading weights")
    p.add_argument("--design", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("audit-parent", help="Read-only compact parent evidence audit")
    p.add_argument("--design", required=True)
    p.add_argument("--parent", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-server", help="Validate all original files without model execution")
    p.add_argument("--design", required=True)
    p.add_argument("--paths", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyze", help="Verify and summarize all six completed S1 banks")
    p.add_argument("--run-root", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("render-slurm", help="Generate a script; never submit it")
    p.add_argument("--stage", choices=("SMOKE", "S1", "S2"), required=True)
    p.add_argument("--out", required=True)
    for command in ("smoke", "run-s1", "run-s2", "eval-historical-r2"):
        p = sub.add_parser(command)
        p.add_argument("--validated-plan", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--allow-gpu-execution", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        if command != "smoke":
            p.add_argument("--smoke-evidence")
        if command == "run-s1":
            p.add_argument("--bank", choices=BANK_IDS, required=True)
        if command == "run-s2":
            p.add_argument("--seed", type=int, choices=(17, 29, 41), required=True)
            p.add_argument("--arm", choices=ARMS, required=True)
            p.add_argument("--allow-training", action="store_true")
            p.add_argument("--s1-evidence")
        if command == "eval-historical-r2":
            p.add_argument("--arm", choices=("X_BASE", "X_VALID"), required=True)
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    execution = args.command in ("smoke", "run-s1", "run-s2", "eval-historical-r2")
    if execution and not args.dry_run:
        if not args.allow_gpu_execution:
            p.error("Real model loading requires --allow-gpu-execution")
        if args.command == "run-s2" and not args.allow_training:
            p.error("S2 additionally requires --allow-training")
    try:
        if args.command == "plan":
            result = build_plan(load_design(args.design))
            write_new_json(args.out, result)
        elif args.command == "audit-parent":
            design = load_design(args.design)
            from .followup_parent import audit_parent_evidence

            parent = Path(args.parent).resolve()
            if parent.is_dir() and Path(args.out).resolve().is_relative_to(parent):
                raise ValueError("Audit output must not change the read-only parent directory")
            result = audit_parent_evidence(args.parent, project_root=PROJECT_ROOT)
            result = {**result, "design_hash": canonical_hash(design)}
            write_new_json(args.out, result)
        elif args.command == "prepare-server":
            from .followup_backend import DIRECTORY_KEYS, _separate_output, prepare_server_plan

            result = prepare_server_plan(load_design(args.design), strict_json(args.paths))
            _separate_output(args.out, [result["paths"][key] for key in DIRECTORY_KEYS])
            write_new_json(args.out, result)
        elif args.command == "analyze":
            result = analyze_run(args.run_root, args.out)
        elif args.command == "render-slurm":
            path = Path(args.out)
            if any(part.is_symlink() for part in (path, *path.parents)) or ".." in path.parts:
                raise ValueError("Unsafe script output path")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                stream.write(render_slurm(stage=args.stage))
            result = {"status": "SCRIPT_ONLY_NOT_SUBMITTED"}
        else:
            plan = strict_json(args.validated_plan)
            validate_design(plan["design"])
            if args.dry_run:
                result = {
                    "status": "DRY_RUN_NOT_EXECUTED",
                    "command": args.command,
                    "gpu_started": False,
                    "training_started": False,
                    "budgets": build_plan(plan["design"])["budgets"],
                }
            else:
                from .followup_backend import (
                    DIRECTORY_KEYS,
                    _separate_output,
                    load_server_context,
                    smoke_server,
                )

                output = _separate_output(args.out, [plan["paths"][key] for key in DIRECTORY_KEYS])
                if not output.is_relative_to(Path(plan["paths"]["new_run_root"]).resolve()):
                    raise ValueError("Execution output must lie within the frozen new_run_root")
                if args.command == "smoke":
                    if args.resume:
                        raise ValueError(
                            "Smoke evidence is immutable; use a fresh output directory"
                        )
                    result = smoke_server(plan, output_root=args.out, allow_gpu_execution=True)
                else:
                    _smoke_gate(plan, args.smoke_evidence)
                    if args.command == "run-s1":
                        context = load_server_context(
                            plan, stage="S1", bank=args.bank, allow_gpu_execution=True
                        )
                        context["plan"]["gates"].update(
                            gpu_smoke_passed=True, server_preflight_passed=True
                        )
                        from .followup_runtime import run_followup_bank

                        result = run_followup_bank(
                            output_root=args.out, resume=args.resume, **context
                        )
                    elif args.command == "eval-historical-r2":
                        from .followup_backend import evaluate_historical_r2

                        result = evaluate_historical_r2(
                            plan,
                            arm=args.arm,
                            output_root=args.out,
                            resume=args.resume,
                            allow_gpu_execution=True,
                        )
                    else:
                        _s1_gate(plan, args.s1_evidence)
                        context = load_server_context(
                            plan, stage="S2", seed=args.seed, arm=args.arm, allow_gpu_execution=True
                        )
                        context["plan"]["gates"].update(
                            gpu_smoke_passed=True,
                            training_authorized=True,
                            s1_measurement_passed=True,
                        )
                        from .followup_train import run_training_arm

                        result = run_training_arm(
                            seed=args.seed,
                            arm=args.arm,
                            output_root=args.out,
                            resume=args.resume,
                            **context,
                        )
        print(
            json.dumps(
                {"command": args.command, "status": result.get("status", "CHECKED")},
                ensure_ascii=False,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
