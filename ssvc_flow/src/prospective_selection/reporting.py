"""Reports distinguish registered work, physical measurements and inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .runtime import read_json


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def development_report(config, root, out, *, first_four=False):
    from .jobs import TaskRegistry

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    registry = TaskRegistry(root, config)
    specs = (
        config["origins"]["development"][:4]
        if first_four
        else config["origins"]["development"] + config["origins"]["tuning"]
    )
    rows, missing, sources = [], [], []
    recipes = (*config["actions"]["selectable"], *config["actions"]["external_baselines"])
    for spec in specs:
        lid = spec["seed"]
        source = root / "sources" / str(lid) / "COMPLETE.json"
        if source.exists():
            sources.append(read_json(source))
        else:
            missing.append(f"source/{lid}")
        for step in spec["anchors"]:
            origin = f"{lid}_t{step}"
            if not (root / "prestate" / origin / "COMPLETE.json").exists():
                missing.append(f"prestate/{origin}")
            for recipe in recipes:
                branch = registry.branch_output(origin, recipe, 1)
                if (
                    not (branch / "COMPLETE.json").exists()
                    or not (branch / "EVAL_H32.json").exists()
                ):
                    missing.append(f"{origin}/{recipe}")
                    continue
                train, ev = read_json(branch / "COMPLETE.json"), read_json(branch / "EVAL_H32.json")
                if train["status"] != "TRAINING_COMPLETE" or ev["status"] != "EVALUATION_COMPLETE":
                    raise ValueError("Real complete measurements required")
                h8 = read_json(branch / "EVAL_H08.json")
                rows.append(
                    dict(
                        lineage_id=lid,
                        origin_step=step,
                        source_recipe=spec["source_recipe"],
                        recipe_id=recipe,
                        H32_J=ev["summary"]["J"],
                        H8_diagnostic_J=h8["summary"]["J"],
                        training_seconds=train["elapsed_training_seconds"],
                        H32_evaluation_seconds=ev["elapsed_sampling_seconds"],
                        training_updates=train["steps"],
                        training_outputs=train["training_outputs"],
                        H32_outputs=ev["generated_outputs"],
                    )
                )
    _csv(out / "development_responses.csv", rows)
    result = {
        "status": "FIRST_FOUR_COMPLETE"
        if first_four and not missing
        else "DEVELOPMENT_COMPLETE"
        if not missing
        else "INCOMPLETE",
        "branches_measured": len(rows),
        "expected_branches": len(specs) * 2 * 11,
        "independent_lineages_requested": len(specs),
        "missing": missing,
        "source_training_seconds": sum(s["elapsed_training_seconds"] for s in sources),
        "branch_training_seconds": sum(r["training_seconds"] for r in rows),
        "H32_evaluation_seconds": sum(r["H32_evaluation_seconds"] for r in rows),
        "rows": rows,
    }
    (out / "DEVELOPMENT_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 开发结果",
        "",
        f"状态：{result['status']}。真实完整分支 {len(rows)}/{result['expected_branches']}。",
        "同一 lineage 的两个原点不计作独立训练种子；未完成条目不填零。",
        "",
        "逐配方响应见 development_responses.csv；H8为次要诊断，不参与最终选择。",
    ]
    (out / "DEVELOPMENT_RESULTS_zh.md").write_text("\n".join(lines) + "\n")
    (out / "RUNTIME_AND_RESOURCE_zh.md").write_text(
        "# 实测工期\n\n"
        + "\n".join(
            f"- {k}: {result[k]:.3f} 秒"
            for k in (
                "source_training_seconds",
                "branch_training_seconds",
                "H32_evaluation_seconds",
            )
        )
        + "\n\n以上仅为已完成部分的阶段耗时。"
        "不含排队、加载、失败attempt或前置观测；不是总GPU占用小时。"
        "\n未完成首个source及两个branch前，不给出本轮全量实测工期推算。\n"
    )
    return result


def final_report(config, root, out):
    from .analysis import analyze_final
    from .jobs import TaskRegistry

    registry, out = TaskRegistry(root, config), Path(out)
    freeze = registry.load_freeze()
    out.mkdir(parents=True, exist_ok=True)
    specs = config["origins"]["test_pool"][: freeze["test_n"]]
    decisions, outcomes = [], []
    for spec in specs:
        origin = f"{spec['seed']}_t{spec['anchors'][0]}"
        d = registry.decision(origin)
        selections = {
            **d["choices"],
            "best_static": freeze["best_static"],
            "R0": "R0",
            **{r: r for r in config["actions"]["external_baselines"]},
        }
        decisions.append({**d, "selections": selections})
        for recipe in set(selections.values()):
            for repeat in (1, 2):
                branch = registry.branch_output(origin, recipe, repeat)
                train, ev = read_json(branch / "COMPLETE.json"), read_json(branch / "EVAL_H32.json")
                if train["status"] != "TRAINING_COMPLETE" or ev["status"] != "EVALUATION_COMPLETE":
                    raise ValueError("Final analysis requires complete real training/evaluation")
                if ev["identity"]["draws"] != freeze["test_draws"]:
                    raise ValueError("Frozen evaluation sample count changed")
                outcomes.append(
                    dict(
                        lineage_id=str(spec["seed"]),
                        origin_step=spec["anchors"][0],
                        source_recipe=spec["source_recipe"],
                        recipe_id=recipe,
                        continuation_repeat=repeat,
                        panel_id="T",
                        horizon=32,
                        n_draws=freeze["test_draws"],
                        execution_kind="REAL_CUDA_MODEL",
                        J=ev["summary"]["J"],
                    )
                )
    units = {
        str(s["seed"]): {"origin_step": s["anchors"][0], "source_recipe": s["source_recipe"]}
        for s in specs
    }
    result = analyze_final(decisions, outcomes, units)
    _csv(out / "test_lineage_outcomes.csv", outcomes)
    (out / "FINAL_ANALYSIS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions))
    # Detailed decision prose is generated from measured statistics, never from
    # a workflow or testing PASS. Keep the complete estimator record alongside it.
    (out / "FINAL_DECISION_zh.md").write_text(
        "# 最终前瞻决定\n\n"
        "以下结论条件于冻结T、单次H32选择与注册source总体；不代表连续在线控制。\n\n"
        "完整主比较、四个Holm校正对照及零方差状态见 FINAL_ANALYSIS.json。\n\n"
        "```json\n" + json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n"
    )
    return result
