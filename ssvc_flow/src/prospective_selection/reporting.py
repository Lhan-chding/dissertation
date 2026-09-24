"""Reports distinguish registered work, physical measurements and inference."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from ..modeling_v3.io import canonical_hash
from .runtime import read_json


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _match(actual, expected, label):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{label} mismatch: {key}")


def _nonnegative(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid measured {label}")
    return value


def validate_training(config, directory, expected_identity, recipe, steps):
    """Validate small provenance receipts; do not repeatedly hash model weights."""
    directory = Path(directory)
    manifest = read_json(directory / "MANIFEST.json")
    complete = read_json(directory / "COMPLETE.json")
    _match(
        manifest,
        {
            "kind": "PROSPECTIVE_TRAINING",
            "fixture": False,
            "config_hash": canonical_hash(config),
            "recipe": recipe,
            "steps": steps,
            "B": 4,
            "K": 8,
            "Lnorm": 64,
        },
        "Training manifest",
    )
    _match(
        manifest.get("runtime_identity", {}),
        {"execution_kind": "REAL_CUDA_MODEL"},
        "Training execution",
    )
    _match(manifest.get("identity", {}), expected_identity, "Training identity")
    _match(
        complete,
        {
            "status": "TRAINING_COMPLETE",
            "steps": steps,
            "identity": manifest["identity"],
            "training_outputs": steps * 32,
        },
        "Training completion",
    )
    endpoint = complete.get("checkpoints", {}).get(str(steps))
    if not endpoint or endpoint != complete.get("checkpoint") or not endpoint.get("state_hash"):
        raise ValueError("Training final checkpoint binding mismatch")
    _nonnegative(complete.get("elapsed_training_seconds"), "training seconds")
    if not manifest.get("initial_state_hash"):
        raise ValueError("Missing training initial state binding")
    return manifest, complete


def validate_evaluation(
    config, directory, expected_identity, *, prompts_hash=None, n_prompts, sidecar=None
):
    """Cross-check evaluation manifest, complete receipt and endpoint sidecar."""
    directory = Path(directory)
    manifest = read_json(directory / "MANIFEST.json")
    complete = read_json(directory / "COMPLETE.json")
    expected = {**expected_identity, "config_hash": canonical_hash(config)}
    if prompts_hash is not None:
        expected["prompts_hash"] = prompts_hash
    _match(manifest, expected, "Evaluation identity")
    if not manifest.get("prompts_hash") or not manifest.get("checkpoint_state_hash"):
        raise ValueError("Evaluation requires prompt and checkpoint bindings")
    _match(
        complete,
        {
            "status": "EVALUATION_COMPLETE",
            "identity": manifest,
            "generated_outputs": n_prompts * expected["draws"],
        },
        "Evaluation completion",
    )
    summary = complete.get("summary", {})
    _match(
        summary,
        {"samples": complete["generated_outputs"], "prompts": n_prompts},
        "Evaluation summary",
    )
    utility = summary.get("J")
    if not isinstance(utility, (int, float)) or not math.isfinite(utility) or not 0 <= utility <= 1:
        raise ValueError("Invalid measured utility")
    _nonnegative(complete.get("elapsed_sampling_seconds"), "evaluation seconds")
    if sidecar is not None and read_json(sidecar) != complete:
        raise ValueError("Evaluation endpoint sidecar differs from physical completion")
    return complete


def _branch_evidence(
    config,
    branch,
    spec,
    step,
    recipe,
    repeat,
    *,
    panel,
    draws,
    prompts_hash=None,
    decision=None,
    freeze=None,
    registry=None,
):
    origin = f"{spec['seed']}_t{step}"
    expected = {
        "lineage_id": spec["seed"],
        "origin_id": origin,
        "recipe_id": recipe,
        "repeat": repeat,
        "role": spec["role"],
    }
    manifest, complete = validate_training(config, branch, expected, recipe, 32)
    if decision is not None:
        from .jobs import content_id

        receipt = manifest["identity"].get("authorization_receipt")
        if not isinstance(receipt, dict) or registry.task(receipt["task_id"]) != receipt:
            raise ValueError("Test training authorization is not the registered task")
        _match(
            receipt,
            {
                "kind": "branch",
                "freeze_id": freeze["freeze_id"],
                "decision_id": content_id(decision),
            },
            "Prospective authorization",
        )
        _match(receipt.get("payload", {}), expected, "Authorized branch")
    ev = validate_evaluation(
        config,
        branch / "evaluations/H32",
        {
            "lineage_id": spec["seed"],
            "origin_id": origin,
            "policy_id": f"{origin}:{recipe}:{repeat}:H32",
            "panel_id": panel,
            "horizon": 32,
            "repeat": repeat,
            "role": "evaluation",
            "draws": draws,
            "checkpoint_state_hash": complete["checkpoint"]["state_hash"],
        },
        prompts_hash=prompts_hash,
        n_prompts=config["panels"][panel]["prompts"],
        sidecar=branch / "EVAL_H32.json",
    )
    return manifest, complete, ev


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
    panel_hashes = {}
    recipes = (*config["actions"]["selectable"], *config["actions"]["external_baselines"])
    for spec in specs:
        lid = spec["seed"]
        source = root / "sources" / str(lid) / "COMPLETE.json"
        if source.exists():
            _, source_complete = validate_training(
                config,
                source.parent,
                {"lineage_id": lid, "source_recipe": spec["source_recipe"], "role": "source"},
                spec["source_recipe"],
                spec["source_steps"],
            )
            sources.append(source_complete)
        else:
            missing.append(f"source/{lid}")
            source_complete = None
        for step in spec["anchors"]:
            origin = f"{lid}_t{step}"
            prestate = root / "prestate" / origin
            if not (prestate / "COMPLETE.json").exists():
                missing.append(f"prestate/{origin}")
            else:
                _match(
                    read_json(prestate / "COMPLETE.json"),
                    {
                        "status": "PRESTATE_COMPLETE",
                        "origin_id": origin,
                        "four_levels_same_samples": True,
                        "shared_generated_outputs": 4608,
                    },
                    "Prestate completion",
                )
                for snapshot in (step - 8, step):
                    source_endpoint = read_json(source.parent / f"H{snapshot:02d}.json")[
                        "checkpoint"
                    ]
                    pre = validate_evaluation(
                        config,
                        prestate / f"t{snapshot}",
                        {
                            "lineage_id": lid,
                            "origin_id": origin,
                            "policy_id": f"{lid}_t{snapshot}",
                            "panel_id": "P",
                            "horizon": snapshot,
                            "repeat": 0,
                            "role": "predecision",
                            "draws": 32,
                            "checkpoint_state_hash": source_endpoint["state_hash"],
                        },
                        n_prompts=config["panels"]["P"]["prompts"],
                        prompts_hash=panel_hashes.get("P"),
                    )
                    panel_hashes.setdefault("P", pre["identity"]["prompts_hash"])
            for recipe in recipes:
                branch = registry.branch_output(origin, recipe, 1)
                if (
                    not (branch / "COMPLETE.json").exists()
                    or not (branch / "EVAL_H32.json").exists()
                    or not (branch / "EVAL_H08.json").exists()
                ):
                    missing.append(f"{origin}/{recipe}")
                    continue
                _, train, ev = _branch_evidence(
                    config,
                    branch,
                    spec,
                    step,
                    recipe,
                    1,
                    panel="D",
                    draws=16,
                    prompts_hash=panel_hashes.get("D"),
                )
                panel_hashes.setdefault("D", ev["identity"]["prompts_hash"])
                h8 = validate_evaluation(
                    config,
                    branch / "evaluations/H08",
                    {
                        "lineage_id": lid,
                        "origin_id": origin,
                        "policy_id": f"{origin}:{recipe}:1:H8",
                        "panel_id": "D_H8",
                        "horizon": 8,
                        "repeat": 1,
                        "role": "diagnostic",
                        "draws": 8,
                        "checkpoint_state_hash": train["checkpoints"]["8"]["state_hash"],
                    },
                    n_prompts=24,
                    sidecar=branch / "EVAL_H08.json",
                    prompts_hash=panel_hashes.get("D_H8"),
                )
                panel_hashes.setdefault("D_H8", h8["identity"]["prompts_hash"])
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
        "expected_branches": sum(len(s["anchors"]) for s in specs) * len(recipes),
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
    if not freeze.get("T_panel", {}).get("prompts"):
        raise ValueError("Frozen T must contain the exact prompt panel")
    frozen_prompts_hash = canonical_hash(freeze["T_panel"]["prompts"])
    if len(freeze["T_panel"]["prompts"]) != config["panels"]["T"]["prompts"]:
        raise ValueError("Frozen T prompt count differs from protocol")
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
                manifest, _, ev = _branch_evidence(
                    config,
                    branch,
                    spec,
                    spec["anchors"][0],
                    recipe,
                    repeat,
                    panel="T",
                    draws=freeze["test_draws"],
                    prompts_hash=frozen_prompts_hash,
                    decision=d,
                    freeze=freeze,
                    registry=registry,
                )
                outcomes.append(
                    dict(
                        lineage_id=str(spec["seed"]),
                        origin_step=spec["anchors"][0],
                        source_recipe=spec["source_recipe"],
                        recipe_id=recipe,
                        continuation_repeat=repeat,
                        panel_id=ev["identity"]["panel_id"],
                        horizon=ev["identity"]["horizon"],
                        n_draws=ev["identity"]["draws"],
                        execution_kind=manifest["runtime_identity"]["execution_kind"],
                        J=ev["summary"]["J"],
                    )
                )
    units = {
        str(s["seed"]): {"origin_step": s["anchors"][0], "source_recipe": s["source_recipe"]}
        for s in specs
    }
    result = analyze_final(decisions, outcomes, units)
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "test_lineage_outcomes.csv", outcomes)
    _csv(out / "primary_comparison.csv", result["primary_lineage_rows"])
    secondary_rows = []
    for name, comparison in result["secondary"].items():
        secondary_rows.append(
            {
                "comparison": f"Z3 - {name}",
                "estimate": comparison["estimate"],
                "ci_lower": comparison["interval"][0] if comparison["interval"] else None,
                "ci_upper": comparison["interval"][1] if comparison["interval"] else None,
                "inference_status": comparison["status"],
                **result["secondary_holm"]["comparisons"][name],
            }
        )
    _csv(out / "secondary_holm.csv", secondary_rows)
    (out / "FINAL_ANALYSIS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions))
    (out / "FINAL_DECISION_zh.md").write_text(_final_chinese(result, freeze))
    return result


def _final_chinese(result, freeze):
    primary = result["primary"]

    def effect(value):
        return "未识别" if value is None else f"{100 * value:.4f} 个百分点"

    def interval(value):
        return (
            "未识别，不能声称等效"
            if value is None
            else f"[{100 * value[0]:.4f}, {100 * value[1]:.4f}] 个百分点"
        )

    def probability(value):
        return "未识别" if value is None else f"{value:.6g}"

    primary_answer = (
        "有正向证据" if result["extra_repair_beats_full_reward_information"] else "未证实改善"
    )
    all_answer = (
        "四项均有优越性证据"
        if result["extra_repair_beats_all_registered_controls"]
        else "未证实同时优于全部四项对照"
    )
    lines = [
        "# 最终前瞻决定",
        "",
        f"**额外修复结构是否改善未见训练原点的奖励选择：{primary_answer}。**",
        f"主比较 Z3−完整奖励统计 Z2：均值差 {effect(primary['estimate'])}，"
        f"95% Welch 区间 {interval(primary['interval'])}，状态 `{primary['status']}`。",
        f"主比较双侧 p={probability(primary['p_value'])}；四 cell 内按 lineage 重采样的"
        f"bootstrap 敏感性区间为 {interval(primary['bootstrap_interval'])}。",
        "",
        f"**是否优于固定配方、GDPO、SAW 和直接修复奖励：{all_answer}。**",
        "",
        "| 对照 | Z3−对照 | 95% Welch 区间 | Holm p | Holm 拒绝零差异 | 状态 |",
        "|---|---:|---|---:|---|---|",
    ]
    names = {
        "best_static": f"固定配方 {freeze['best_static']}",
        "GDPO_R4": "GDPO",
        "SAW_R4": "SAW",
        "DIRECT_REPAIR_R4": "直接修复奖励",
    }
    for name, comparison in result["secondary"].items():
        corrected = result["secondary_holm"]["comparisons"][name]
        lines.append(
            f"| {names[name]} | {effect(comparison['estimate'])} | "
            f"{interval(comparison['interval'])} | {probability(corrected['holm_p_value'])} | "
            f"{'是' if corrected['reject'] else '否'} | {comparison['status']} |"
        )
    lines += [
        "",
        f"有效独立单位为 {primary['n_independent_lineages']} 条 source lineage，"
        f"每条先平均两次未来训练重复；冻结每题样本量 m={freeze['test_draws']}。"
        "相同动作引用同一物理结果，不新增训练种子。",
        "CI 跨零本身不代表等效；零方差保留未识别状态。"
        + (
            "主比较区间完整位于预定 ±1pp 内，可在该固定面板对象上报告近似等效。"
            if primary["approximately_equivalent"]
            else "本报告未据主比较宣布近似等效。"
        ),
        "结论条件于冻结 T、冻结选择规则和单次 H32 操作；不涵盖未来场景总体或连续在线闭环安全。",
        "表示的代数非冗余本身不证明控制价值；信息审计与本次实际选择收益必须分开解释。",
        "逐 lineage 配对差见 primary_comparison.csv，四项多重比较见 secondary_holm.csv，"
        "全部估计参数与未识别状态见 FINAL_ANALYSIS.json。",
        "",
    ]
    return "\n".join(lines)
