"""Run metadata/local analysis first; load Qwen only with --execute-gpu."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core import frozen_writer
from ..modeling_v3.io import canonical_hash
from ..modeling_v4 import gpu_collect as gpu
from .runtime import (
    load_runtime,
    panel_prompts,
    policy_from_checkpoint,
    prepare_runtime,
    read_config,
)

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs/decision_modeling/protocol.json"
DEFAULT_PANELS = DEFAULT_CONFIG.with_name("panels.json")


def plan(config, out):
    result = {
        "schema": "decision-modeling-execution-plan-v1",
        "config": config,
        "config_hash": canonical_hash(config),
        "execution_status": "LOCAL_PLAN_ONLY",
        "D0": "MODEL_FREE_REANALYSIS",
        "D1": "SERVER_REQUIRED",
        "D2": "SERVER_REQUIRED_AFTER_D1_REVIEW",
        "D3": "REQUIRES_MEASURED_ENDPOINTS",
        "D4": "REQUIRES_FROZEN_DEVELOPMENT_RESULTS",
        "D2_updates": 512,
        "D2_training_outputs": 16384,
        "maximum_D2_D4_updates": 768,
        "maximum_D2_D4_training_outputs": 24576,
        "first_server_action": "bridge on P, 32 draws per endpoint",
        "automatic_successor": False,
        "online_controller": False,
    }
    gpu._publish(Path(out), result)
    return result


def _freeze(path, config, *, candidate=None):
    if path is None:
        raise ValueError("E/D4 requires development freeze made after O1/O2 reports")
    lock = gpu._read(path)
    if lock["config_hash"] != canonical_hash(config) or lock["selected_exact_recipe"] not in (
        "R0",
        "R2",
        "R3",
        "R4",
    ):
        raise ValueError("Development freeze configuration/exact-family recipe mismatch")
    if not isinstance(lock.get("readout_rule"), str) or not lock["readout_rule"]:
        raise ValueError("Freeze the readout rule before E/D4")
    from .runtime import bound_json

    reports = [bound_json(b) for b in lock["development_reports"]]
    expected = {
        (origin, recipe)
        for origin in ("O1", "O2")
        for recipe in config["blocks"][f"{origin}_recipes"]
    }
    observed = [(r.get("origin_id"), r.get("recipe")) for r in reports]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Freeze needs every registered branch from both development origins")
    if any(r.get("status") != "BLOCK_COMPLETE" or r.get("steps") != 32 for r in reports):
        raise ValueError("Development origin blocks have not reached H32")
    if any(
        r["checkpoint"]["identity"].get("decision_config_hash") != canonical_hash(config)
        for r in reports
    ):
        raise ValueError("Development results use a different decision protocol")
    required = {
        "R0",
        "GDPO_R4",
        "SAW_R4",
        lock["selected_exact_recipe"],
        lock["simple_fixed_recipe"],
    }
    if not required <= set(lock["E_candidates"]):
        raise ValueError("E registry must include selection and all matched strong baselines")
    if candidate is not None and candidate not in lock["E_candidates"]:
        raise ValueError("Candidate outside frozen independent evaluation registry")
    return lock


def validate_endpoint(spec, *, origin, candidate, horizon, config):
    """User-facing labels must describe the actual saved training endpoint."""
    identity = spec["identity"]
    if identity.get("kind") == "DECISION_MODELING_BLOCK":
        if (identity["origin_id"], identity["recipe"], identity["step"]) != (
            origin,
            candidate,
            horizon,
        ):
            raise ValueError("Endpoint label differs from the committed branch")
    elif horizon == 0 and candidate == "ORIGIN" and origin in config["blocks"]["origins"]:
        if any(
            identity.get(key) != value for key, value in config["blocks"]["origins"][origin].items()
        ):
            raise ValueError("Starting checkpoint differs from the declared origin")
    else:
        raise ValueError("Expected a committed decision block or a registered full origin")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("reanalyze")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare-runtime", help="Server metadata only; writes private runtime.json")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--preview", type=Path, required=True)
    p.add_argument("--panels", type=Path, default=DEFAULT_PANELS)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    for name in ("bridge", "block", "observe"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--runtime", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--execute-gpu", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--resume", action="store_true")
        if name == "bridge":
            p.add_argument("--worker-index", type=int)
            p.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
        if name == "block":
            p.add_argument("--origin", choices=("O1", "O2", "O3", "O4"), required=True)
            p.add_argument("--recipe", required=True)
            p.add_argument("--steps", type=int, choices=(8, 32), required=True)
            p.add_argument("--development-freeze", type=Path)
        elif name == "observe":
            p.add_argument(
                "--checkpoint",
                type=Path,
                required=True,
                help="H00/H08/H32 receipt or full checkpoint binding JSON",
            )
            p.add_argument("--origin", required=True)
            p.add_argument("--candidate", required=True)
            p.add_argument("--horizon", type=int, choices=(0, 8, 32), required=True)
            p.add_argument("--panel", choices=("P", "E"), default="P")
            p.add_argument("--look", type=int, choices=(16, 32, 128, 256, 512, 1024), default=32)
            p.add_argument(
                "--role", choices=("endpoint", "discovery", "reference"), default="endpoint"
            )
            p.add_argument("--development-freeze", type=Path)
            p.add_argument(
                "--append-reason", help="Observed unresolved choice; never a p-value rule"
            )
    p = sub.add_parser("merge-bridge", help="Metadata-only validated merge of completed D1 workers")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 3), required=True)
    p = sub.add_parser("report")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "reanalyze":
        from .reanalysis import reanalyze

        result = reanalyze(args.source, args.out)
    else:
        config = read_config(args.config)
        if args.command == "plan":
            result = plan(config, args.out)
        elif args.command == "prepare-runtime":
            result = prepare_runtime(
                config,
                tasks_path=args.tasks,
                preview_path=args.preview,
                panels_path=args.panels,
                data_root=args.data_root,
                out=args.out,
            )
        elif args.command == "merge-bridge":
            from .parallel import merge_workers
            from .runtime import runtime_metadata

            private, _, panels = runtime_metadata(config, args.runtime, out=args.out)
            prompts = panel_prompts(
                {"decision_panels": panels, "data_root": private["data_root"]}, "P"
            )
            result = merge_workers(
                private,
                prompts,
                root=args.root,
                out=args.out,
                workers=args.workers,
                stream_id=f"{canonical_hash(config)}:D1",
            )
        elif args.command == "report":
            from .reporting import report

            result = report(args.root, args.out, config)
        elif args.dry_run or not args.execute_gpu:
            result = {
                "status": "DRY_RUN_NO_MODEL_LOAD",
                "command": args.command,
                "config_hash": canonical_hash(config),
                "server_execution_required": True,
                "execute_gpu": False,
                "new_training_steps": 0,
            }
            gpu._publish(args.out / f"DRY_RUN_{args.command}.json", result)
        else:
            from .measurement import bridge, observe

            if args.command == "block":
                from .block_runner import run_from_runtime

                allowed = config["blocks"].get(f"{args.origin}_recipes")
                if args.origin in ("O3", "O4"):
                    lock = _freeze(args.development_freeze, config)
                    allowed = {"R0", "GDPO_R4", "SAW_R4", lock["selected_exact_recipe"]}
                if args.recipe not in allowed:
                    raise ValueError("Recipe outside this origin's frozen branch registry")
            elif args.command == "observe":
                expected = {
                    "endpoint": (32, 128, 512),
                    "discovery": (16,),
                    "reference": (256, 1024),
                }
                if args.look not in expected[args.role]:
                    raise ValueError("Look/role combination not preregistered")
                previous = {128: 32, 512: 128, 1024: 256}.get(args.look)
                if previous is not None and (
                    not (args.out / f"LOOK_{previous}.json").is_file() or not args.append_reason
                ):
                    raise ValueError(
                        "Append requires the preceding look and its unresolved-choice reason"
                    )
                if args.panel == "E":
                    _freeze(args.development_freeze, config, candidate=args.candidate)
            if args.command == "bridge":
                if args.worker_index is None and args.workers != 1:
                    raise ValueError("Parallel bridge requires --worker-index")
                if args.worker_index is not None:
                    if not 0 <= args.worker_index < args.workers:
                        raise ValueError("Worker index outside registered worker count")
                    args.out = args.out / "workers" / str(args.worker_index)
            with frozen_writer(args.out):
                runtime = load_runtime(config, args.runtime, out=args.out, execute_gpu=True)
                prefix = canonical_hash(config)
                if args.command == "bridge":
                    private = runtime["decision_private"]
                    if args.worker_index is not None:
                        from .parallel import run_worker

                        result = run_worker(
                            runtime,
                            panel_prompts(runtime, "P"),
                            out=args.out,
                            worker_index=args.worker_index,
                            workers=args.workers,
                            stream_id=f"{prefix}:D1",
                        )
                    else:
                        result = bridge(
                            runtime,
                            private["preview_policies"],
                            panel_prompts(runtime, "P"),
                            out=args.out,
                            stream_id=f"{prefix}:D1",
                            origin_policy=private["preview_origin"],
                        )
                elif args.command == "block":
                    origin = runtime["decision_private"]["origins"][args.origin]
                    prompts = panel_prompts(runtime, "P")
                    # One shared starting observation per origin, reused by every arm.
                    starting = policy_from_checkpoint(runtime, origin, candidate_id="ORIGIN")
                    shared = args.out.parent / "H00_observations"
                    with frozen_writer(shared):
                        observe(
                            runtime,
                            [starting],
                            prompts,
                            out=shared,
                            stream_id=f"{prefix}:{args.origin}:ORIGIN:H0:P",
                            context={
                                "origin_id": args.origin,
                                "horizon": 0,
                                "candidate_id": "ORIGIN",
                                "panel": "P",
                            },
                        )

                    def evaluate(run, checkpoint, step):
                        policy = policy_from_checkpoint(run, checkpoint, candidate_id=args.recipe)
                        value, _ = observe(
                            run,
                            [policy],
                            prompts,
                            out=args.out / "observations" / f"H{step:02d}",
                            stream_id=f"{prefix}:{args.origin}:{args.recipe}:H{step}:P",
                            context={
                                "origin_id": args.origin,
                                "horizon": step,
                                "candidate_id": args.recipe,
                                "panel": "P",
                            },
                        )
                        return value

                    result = run_from_runtime(
                        runtime,
                        origin,
                        args.recipe,
                        args.steps,
                        args.out,
                        train_prompts=runtime["decision_train_prompts"],
                        origin_id=args.origin,
                        resume=args.resume,
                        evaluate=evaluate,
                    )
                else:
                    spec = gpu._read(args.checkpoint)
                    spec = spec.get("checkpoint", spec)
                    validate_endpoint(
                        spec,
                        origin=args.origin,
                        candidate=args.candidate,
                        horizon=args.horizon,
                        config=config,
                    )
                    if args.append_reason:
                        gpu._publish(
                            args.out / f"APPEND_{args.look}.json",
                            {
                                "reason": args.append_reason,
                                "look": args.look,
                                "reference_labels_used_for_selection": False,
                            },
                        )
                    policy = policy_from_checkpoint(runtime, spec, candidate_id=args.candidate)
                    result, _ = observe(
                        runtime,
                        [policy],
                        panel_prompts(runtime, args.panel),
                        out=args.out,
                        look=args.look,
                        role=args.role,
                        stream_id=f"{prefix}:{args.origin}:{args.candidate}:H{args.horizon}:{args.panel}",
                        context={
                            "origin_id": args.origin,
                            "horizon": args.horizon,
                            "candidate_id": args.candidate,
                            "panel": args.panel,
                        },
                    )
    # Summaries are compact; raw tables remain in the output directory.
    print(
        json.dumps(
            {
                "status": result.get("status", result.get("execution_status", "COMPLETE")),
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return result


if __name__ == "__main__":
    main()
