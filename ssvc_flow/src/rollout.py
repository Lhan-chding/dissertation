"""Model-free budgets and the first NTU GPU compatibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import (
    PROJECT_ROOT,
    canonical_hash,
    load_config,
    load_split,
    phase_artifacts,
    validate_config,
    write_json,
)


def estimate_budget(config, phase="smoke", split="dev"):
    train, generation = config["training"], config["generation"]
    split_counts = {"dev": 144, "control": 144, "confirm": 288, "ood": 288}
    if phase == "frozen" and split not in split_counts:
        raise ValueError("unsupported frozen split budget")
    prompts = 36 if phase == "smoke" else 2 * split_counts[split]
    k = train["K"] if phase == "smoke" else 16
    updates = train["smoke_updates"] if phase == "smoke" else 0
    # P1 second update is replayed for save/resume and same-state fork audits.
    audit_replays = 2 if updates else 0
    sampled = prompts * k + updates * train["B"] * k
    greedy = prompts if phase == "frozen" else 0
    generated = sampled + greedy
    scoring = 2 * generated
    update_forwards = (updates + audit_replays) * train["B"] * k
    postupdate_forwards = update_forwards
    scored = scoring + update_forwards + postupdate_forwards
    budget = {
        "phase": phase,
        "prompt_count": prompts,
        "rollout_count": generated,
        "sampled_rollout_count": sampled,
        "greedy_rollout_count": greedy,
        "max_completion_tokens": generation["max_new_tokens"],
        "completion_token_upper_bound": generated * generation["max_new_tokens"],
        "generation_forward_upper_bound": generated * generation["max_new_tokens"],
        "teacher_forcing_forward_count": scored,
        "policy_reference_scoring_forward_count": scoring,
        "update_forward_count": update_forwards,
        "postupdate_likelihood_forward_count": postupdate_forwards,
        "backward_count": (updates + audit_replays) * train["B"] * k,
        "optimizer_updates": updates,
        "audit_replay_updates": audit_replays,
        "additional_cache_image_forward_count": "measured by P1; depends on sampled lengths",
        "gpu_hours": None,
        "peak_gpu_bytes": None,
        "estimate_only": True,
        "budget_note": "Generation counts exclude no hidden sampling; cache tests add forwards. "
        "Measured time and memory required before GPU-hour prediction.",
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
    parser.add_argument("--model", default="qwen35_9b")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/locked.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    budget = estimate_budget(config, args.phase, args.split)
    phase = "P1" if args.phase == "smoke" else "P3"
    if args.out.exists() and any(args.out.iterdir()) and (args.dry_run or not args.resume):
        print("Output already contains evidence; choose a fresh --out or use --resume.")
        return 2
    if args.dry_run:
        phase_artifacts(args.out, phase, "DRY_RUN", budget)
        print(json.dumps(budget, indent=2))
        return 0
    if args.phase != "smoke":
        phase_artifacts(
            args.out,
            phase,
            "NOT_RUN",
            {
                "reason": "P3 runtime requires completed P1 and a frozen dependency/model lock. "
                "This delivery ends at the P1 GPU boundary.",
            },
        )
        return 2
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
