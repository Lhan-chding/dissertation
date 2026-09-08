"""Model-free budgets and the first NTU GPU compatibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (
    PROJECT_ROOT,
    canonical_hash,
    frozen_writer,
    load_config,
    load_split,
    phase_artifacts,
    validate_config,
    write_json,
)


def estimate_budget(config, phase="smoke", split="dev", *, track="N", scene_count=None):
    train, generation = config["training"], config["generation"]
    split_counts = {"dev": 144, "control": 144, "confirm": 288, "ood": 288}
    if phase == "frozen" and split not in split_counts:
        raise ValueError("unsupported frozen split budget")
    prompts = 36 if phase == "smoke" else 2 * split_counts[split]
    if phase == "frozen" and track == "L":
        if type(scene_count) is not int or scene_count <= 0:
            raise ValueError("L budget requires the verified legacy manifest scene count")
        prompts = scene_count
    k = train["K"] if phase == "smoke" else 16
    updates = train["smoke_updates"] if phase == "smoke" else 0
    # P1 second update is replayed for save/resume and same-state fork audits.
    audit_replays = 2 if updates else 0
    sampled = prompts * k + updates * train["B"] * k
    greedy = prompts if phase == "frozen" else 0
    generated = sampled + greedy
    # P3 disables the fresh adapter, so policy and reference are the same base.
    scoring = (2 if phase == "smoke" else 1) * generated
    update_forwards = (updates + audit_replays) * train["B"] * k
    postupdate_forwards = update_forwards
    scored = scoring + update_forwards + postupdate_forwards
    length = 48 if phase == "frozen" and track == "L" else generation["max_new_tokens"]
    budget = {
        "phase": phase,
        "track": track,
        "prompt_count": prompts,
        "rollout_count": generated,
        "sampled_rollout_count": sampled,
        "greedy_rollout_count": greedy,
        "max_completion_tokens": length,
        "completion_token_upper_bound": generated * length,
        "generation_forward_upper_bound": generated * length,
        "teacher_forcing_sequence_count": scored,
        "policy_reference_scoring_sequence_count": scoring,
        "update_sequence_count": update_forwards,
        "postupdate_likelihood_sequence_count": postupdate_forwards,
        "teacher_forcing_forward_count": None,
        "teacher_forcing_forward_upper_bound": scored * length,
        "probability_execution": "uncached_prefix_recompute",
        "backward_count": (updates + audit_replays) * train["B"] * k,
        "optimizer_updates": updates,
        "audit_replay_updates": audit_replays,
        "additional_cache_image_forward_count": "measured by P1; depends on sampled lengths"
        if phase == "smoke"
        else 0,
        "gpu_hours": None,
        "peak_gpu_bytes": None,
        "estimate_only": True,
        "budget_note": "Each scored token recomputes its uncached prefix, including for "
        "gradients. The forward bound excludes checkpoint backward recomputations and "
        "cache/image audits. Measured time and memory required before GPU-hour prediction.",
    }
    return budget


def calibration_panel(scenes):
    cells = {}
    for scene in sorted(scenes, key=lambda item: item["base_scene_id"]):
        key = (scene["constraint_family"], scene["chart_type"], scene["operation"])
        if key not in cells:
            cells = {**cells, key: scene}
    if len(cells) != 18:
        raise ValueError("P1 needs exactly 18 family/chart/operation cells")
    return [cells[key] for key in sorted(cells)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "frozen"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/locked.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--track", choices=["N", "L"], default="N")
    parser.add_argument("--p1-run-dir", type=Path)
    parser.add_argument("--human-review", type=Path)
    parser.add_argument(
        "--contact-manifest",
        type=Path,
        default=PROJECT_ROOT / "docs/local_evidence/P0/contact_sheets_manifest.json",
    )
    parser.add_argument("--legacy-data", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    args = parser.parse_args(argv)
    args.model = args.model or ("qwen25vl_3b" if args.phase == "frozen" else "qwen35_9b")
    config = load_config(args.config)
    phase = "P1" if args.phase == "smoke" else "P3"
    if args.out.exists() and any(args.out.iterdir()) and (args.dry_run or not args.resume):
        print("Output already contains evidence; choose a fresh --out or use --resume.")
        return 2
    legacy = None
    try:
        if args.phase == "frozen" and args.track == "L":
            if args.legacy_data is None or args.legacy_manifest is None:
                raise ValueError("L requires --legacy-data and --legacy-manifest")
            from .legacy_frozen import load_legacy_scenes

            legacy = load_legacy_scenes(args.legacy_data, args.legacy_manifest)
        if args.phase == "frozen" and args.split != "dev":
            raise ValueError("P3 runs the fixed dev bank; confirm/other split access is sealed")
        budget = estimate_budget(
            config,
            args.phase,
            args.split,
            track=args.track,
            scene_count=len(legacy) if legacy else None,
        )
    except ValueError as exc:
        if not args.resume:
            phase_artifacts(args.out, phase, "BLOCKED", {"reason": str(exc)})
        print(str(exc))
        return 2
    if args.dry_run:
        phase_artifacts(args.out, phase, "DRY_RUN", budget)
        print(json.dumps(budget, indent=2))
        return 0
    if args.phase == "frozen":
        try:
            if args.p1_run_dir is None:
                raise ValueError("P1 evidence directory is required for the selected model")
            from .frozen_runtime import load_frozen_dev, run_frozen, validate_p1_evidence

            validate_p1_evidence(args.p1_run_dir, config, args.model)
            validate_config(config, "frozen", args.model)
            root = args.data_root or PROJECT_ROOT / config["data_root"]
            scenes = legacy if legacy is not None else load_frozen_dev(root)
            run_frozen(
                {
                    **config,
                    "data_root": str(root.resolve()),
                    **(
                        {
                            "legacy_data": str(args.legacy_data.resolve()),
                            "legacy_manifest": str(args.legacy_manifest.resolve()),
                        }
                        if legacy is not None
                        else {}
                    ),
                },
                scenes,
                args.out,
                args.model,
                resume=args.resume,
                track=args.track,
                p1_run_dir=args.p1_run_dir,
                human_review=args.human_review,
                contact_manifest=args.contact_manifest,
            )
        except Exception as exc:
            failure = {
                "phase": "P3",
                "model_key": args.model,
                "track": args.track,
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
            if not args.resume and not isinstance(exc, FileExistsError):
                try:
                    with frozen_writer(args.out):
                        # Runtime owns failures after identity creation. A late preflight
                        # failure must not overwrite another writer's status or ledger.
                        if (
                            not (args.out / "identity.json").exists()
                            and not (args.out / "status.json").exists()
                        ):
                            phase_artifacts(args.out, "P3", "BLOCKED", failure)
                            with (args.out / "failures.jsonl").open(
                                "a", encoding="utf-8"
                            ) as stream:
                                stream.write(json.dumps(failure) + "\n")
                except FileExistsError:
                    pass
            print(json.dumps(failure, ensure_ascii=False))
            return 1 if (args.out / "identity.json").exists() else 2
        return 0
    validate_config(config, args.phase, args.model)
    root = args.data_root or PROJECT_ROOT / config["data_root"]
    scenes = calibration_panel(load_split(root, "calibration", purpose="P1 compatibility panel"))
    runtime_config = {
        **config,
        "data_root": str(root.resolve()),
        "locked_config_hash": canonical_hash(config),
    }
    try:
        from .smoke_runtime import run_smoke

        audit = run_smoke(runtime_config, scenes, args.out, args.model, resume=args.resume)
    except Exception as exc:
        if args.resume:
            # A rejected resume must never replace an existing manifest/status.
            print(
                json.dumps(
                    {"resume_rejected": True, "error_type": type(exc).__name__, "message": str(exc)}
                )
            )
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        failure = {"phase": phase, "error_type": type(exc).__name__, "message": str(exc)}
        with (args.out / "failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure) + "\n")
        phase_artifacts(args.out, phase, "FAILED", failure)
        print(json.dumps(failure))
        return 1
    write_json(args.out / "model_audit.json", audit)
    status = "PASS" if audit.get("passed") is True else "FAILED"
    phase_artifacts(args.out, phase, status, audit, [args.out / "model_audit.json"])
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
