"""Build the fixed R2 diagnostic panel and prompt variants without generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import canonical_hash, load_split
from .next_stage_common import dry_run_plan, load_yaml, stage_status
from .prompts import build_prompt

CONDITIONS = (
    "SYM_ORIGINAL",
    "SYM_CLEAR",
    "SYM_NO_OPERATION",
    "IMAGE_CUE",
    "IMAGE_ONLY",
    "ORACLE_INDEX",
)


def variant_prompt(scene, condition):
    base = build_prompt(scene, "SYMBOLIC_FRESH")
    if condition == "SYM_ORIGINAL":
        return base
    user = base["user"]
    if condition == "SYM_CLEAR":
        user = user.replace(
            "Exactly one value in this observed record is wrong.",
            "Exactly one and only one value in this observed record is wrong. "
            "Use every listed relationship.",
        )
    elif condition == "SYM_NO_OPERATION":
        user = (
            user.split("The downstream calculation:", 1)[0].rstrip()
            + "\nRecover the correct record. Return [a,b,c,d], not an answer."
        )
    elif condition == "IMAGE_CUE":
        return build_prompt(scene, "IMAGE_CUE_FRESH")
    elif condition == "IMAGE_ONLY":
        user = "The attached chart shows the true record. Return only [a,b,c,d] as four integers."
    elif condition == "ORACLE_INDEX":
        user += (
            f"\nThe wrong value is at index {scene['changed_index']}; "
            "do not infer its correct value from this hint."
        )
    content = {
        "system": base["system"],
        "user": user,
        "prompt_hash": canonical_hash(
            {"condition": condition, "base_prompt": base["prompt_hash"], "user": user}
        ),
    }
    if condition in {"IMAGE_CUE", "IMAGE_ONLY"}:
        content["image_path"] = scene["image_path"]
    return content


def build_panel(data_root, out, base_scenes=72):
    scenes = load_split(data_root, "calibration", purpose="R2 fixed diagnostic panel")[:base_scenes]
    rows = []
    for scene in scenes:
        for condition in CONDITIONS:
            prompt = variant_prompt(scene, condition)
            rows.append(
                {
                    "base_scene_id": scene["base_scene_id"],
                    "family": scene["constraint_family"],
                    "condition": condition,
                    "prompt_hash": prompt["prompt_hash"],
                    "image_path": prompt.get("image_path"),
                    "changed_index_in_prompt": condition == "ORACLE_INDEX",
                }
            )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel_manifest.json").write_text(
        json.dumps(
            {
                "status": "READY_FOR_SERVER_GENERATION",
                "base_scenes": len(scenes),
                "conditions": list(CONDITIONS),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (out / "prompt_diffs.md").write_text(
        "# R2 prompt variants\n\n"
        + "\n".join(f"- `{c}`: fixed intervention tag" for c in CONDITIONS)
        + "\n",
        encoding="utf-8",
    )
    (out / "paired_condition_effects.csv").write_text(
        "status,reason\nNOT_MEASURED,real model generation required\n", encoding="utf-8"
    )
    (out / "invalid_taxonomy.csv").write_text(
        "status,reason\nNOT_MEASURED,real model generation required\n", encoding="utf-8"
    )
    (out / "diagnosis_report.md").write_text(
        "# R2\n\nNo model generation was run locally; panel construction only.\n", encoding="utf-8"
    )
    stage_status(
        out,
        "R2",
        "BLOCKED",
        "CPU_AUDIT",
        {"panel_rows": len(rows), "reason": "REAL_CUDA_INFERENCE required for rollouts"},
    )
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--out", type=Path, default=Path("runs/NEXT_20260909/R2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "R2",
                    **dry_run_plan(config),
                    "conditions": list(CONDITIONS),
                    "execution_kind": "CPU_AUDIT",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    build_panel(args.data_root, args.out, config.get("R2", {}).get("base_scenes", 72))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
