"""Evidence-first reports separating discovery, fit, transfer and retention."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .evaluate import release_test, summarize_raw
from .queue import Queue, digest, read_json, write_json
from .statistics import finite_panel_mcse, paired_cluster_bootstrap


def _chunks(root):
    rows = []
    for path in sorted(Path(root).glob("chunks/*.json")):
        chunk = read_json(path)
        if digest(chunk["rows"]) != chunk["rows_hash"]:
            raise ValueError("Evidence chunk digest mismatch")
        rows.extend(chunk["rows"])
    return rows


def _eval_records(run, job, lookup):
    from .cli import _job_root, _records

    row = lookup[job["id"]]
    if row["status"] == "COMPLETE":
        return _chunks(_job_root(run, job))
    if row["status"] == "ALIAS" and row["result"].get("alias_of") in lookup:
        return _eval_records(run, lookup[row["result"]["alias_of"]]["payload"], lookup)
    source = lookup[job["source_job"]]
    while source["status"] == "ALIAS":
        source = lookup[source["result"]["alias_of"]]
    if source["status"] == "NO_VERIFIED_TARGETS_RETURN_PARENT":
        return [
            r
            for r in _records(run, job["parent"], job["split"], 0)
            if r["protocol_id"] == "O0" and r["draw_index"] < job["draws"]
        ]
    return []


def report(run, *, reveal=False):
    from .cli import _context, _job_root

    run, plan, machine, frozen = _context(run)
    queue = Queue(run / "queue.sqlite", maximum=machine.get("max_concurrent_gpus", 5))
    rows = queue.rows()
    lookup = {row["id"]: row for row in rows}
    release = release_test(run, queue) if reveal else None
    if reveal:
        cohort = read_json(run / "cohort/cohort_manifest.json")
        write_json(
            run / "cohort/FINAL_RELEASE.json",
            {
                "cohort_manifest_digest": cohort["manifest_digest"],
                "all_registered_models_terminal": True,
                "matrix_digest": release["matrix_digest"],
            },
        )
    counts = dict(Counter(row["status"] for row in rows))
    missing = [
        {"job": row["id"], "status": row["status"], "reason": (row["result"] or {}).get("reason")}
        for row in rows
        if row["status"] not in ("COMPLETE", "ALIAS", "NO_VERIFIED_TARGETS_RETURN_PARENT")
    ]
    registry = [
        {
            "job": row["id"],
            "parent": row["payload"].get("parent"),
            "arm": row["payload"].get("arm"),
            "repeat": row["payload"].get("repeat"),
            "status": row["status"],
            "result": row["result"],
        }
        for row in rows
        if row["payload"]["kind"] in ("sft", "r0")
    ]
    write_json(run / "TRAINING_REGISTRY.json", registry)
    discoveries = {}
    for row in rows:
        path = run / "evidence" / row["id"] / "discovery.json"
        if row["payload"]["kind"] == "discover" and path.exists():
            discoveries[row["id"]] = read_json(path)
    write_json(run / "discovery_summary.json", discoveries)
    training = {}
    for row in registry:
        root = run / "evidence" / row["job"]
        metrics = [read_json(path) for path in sorted(root.glob("update*.json"))]
        training[row["job"]] = {
            "status": row["status"],
            "updates": metrics,
            "exposures": (row["result"] or {}).get("exposures"),
            "alias_of": (row["result"] or {}).get("alias_of"),
        }
    write_json(run / "training_diagnostics.json", training)
    validation = {}
    for row in rows:
        job = row["payload"]
        if (
            job["kind"] == "evaluate"
            and job["split"] in ("V_selection", "T_train")
            and row["status"] == "COMPLETE"
        ):
            validation[job["id"]] = summarize_raw(
                _chunks(_job_root(run, job)), expected_draws=job["draws"]
            )
    write_json(run / "validation_and_fit.json", validation)
    scope = {
        "schema": "verified-discovery-execution-scope-v1",
        "status": "MATRIX_TERMINAL" if queue.all_terminal() else "IN_PROGRESS",
        "scientific_complete": queue.all_terminal() and not missing,
        "job_status_counts": counts,
        "missing_or_pending": missing,
        "test_results_revealed": bool(reveal),
        "fixture_results_are_not_model_evidence": True,
        "runtime_identity": frozen["run_identity"],
        "discovery": "public-verifier-only SELF; no solver fill",
        "fitting": "completion-only full sequence SFT; exact view aliases retained",
        "transfer": "E O0 fixed step256",
        "retention": "G separate from E; shared base scenes stay clustered",
        "no_automatic_expansion": True,
    }
    costs = execution_costs(run, queue, reveal=reveal)
    scope["execution_costs_path"] = "EXECUTION_COSTS.json"
    scope["physical_cost_complete"] = costs["physical_cost_complete"]
    write_json(run / "EXECUTION_COSTS.json", costs)
    write_json(run / "EXECUTION_SCOPE.json", scope)
    first = (
        "# 第一阶段事实记录\n\n"
        + f"当前任务状态：{counts}。E/G结果"
        + ("已按完整登记矩阵终态统一揭晓。" if reveal else "封存；本文件只展示发现、训练和V诊断。")
        + "\n\n"
    )
    first += (
        "发现集合及新增/丢失：`discovery_summary.json`。拟合/训"
        "练曲线：`training_diagnostics.json`、`validat"
        "ion_and_fit.json`。训练身份、别名及检查点：`TRAINING_"
        "REGISTRY.json`。\n\n"
    )
    first += (
        "技术缺失与未运行项目见 `EXECUTION_SCOPE.json`；负结果不从"
        "登记矩阵中删除。GPU吞吐只引用实际执行receipt中的token及秒数，尚未"
        "测量者不估计为完成。\n"
    )
    (run / "FIRST_STAGE_REPORT_zh.md").write_text(first)
    if not reveal:
        return scope
    endpoints = {}
    raw_endpoints = {}
    from .public_tasks import RoleDataset

    final_data = RoleDataset(run / "cohort", "final_eval")
    final_tasks = {split: final_data.public(split) for split in ("E_test", "G_guard")}
    final_audits = {split: final_data.audit(split) for split in ("E_test", "G_guard")}
    from .cli import _records

    parent_raw = {parent: _records(run, parent, "E_test", 0) for parent in ("S96", "REP96")}
    pre_zero = {}
    parent_endpoints = {}
    portfolios = {}
    for parent in ("S96", "REP96"):
        streams = {}
        for row in parent_raw[parent]:
            if row["protocol_id"] == "O0":
                streams.setdefault(row["task_id"], []).append(row)
        pre_zero[parent] = {
            tid
            for tid, stream in streams.items()
            if len(stream) == 16 and not any(r["public_verifier_pass"] for r in stream)
        }
        for split in ("E_test", "G_guard"):
            bank = [
                r
                for r in _records(run, parent, split, 0)
                if r["protocol_id"] == "O0" and r["draw_index"] < (8 if split == "E_test" else 4)
            ]
            if {r["task_id"] for r in bank} == {t["task_id"] for t in final_tasks[split]} and len(
                bank
            ) == len(final_tasks[split]) * (8 if split == "E_test" else 4):
                parent_endpoints[parent + ":" + split] = summarize_raw(
                    bank, expected_draws=8 if split == "E_test" else 4
                )
        selection_path = run / "evidence" / f"select.{parent}" / "selection.json"
        if selection_path.exists():
            try:
                portfolios[parent] = teacher_portfolios(
                    run,
                    parent,
                    final_tasks["E_test"],
                    parent_raw[parent],
                    read_json(selection_path),
                )
            except ValueError as exc:
                portfolios[parent] = {"status": "INCOMPLETE", "reason": str(exc)}
    write_json(run / "TEACHER_TWO_CALL_PORTFOLIOS.json", portfolios)
    write_json(run / "PARENT_ENDPOINTS.json", parent_endpoints)
    for row in rows:
        job = row["payload"]
        if (
            job["kind"] != "evaluate"
            or job["split"] not in ("E_test", "G_guard")
            or row["status"] not in ("COMPLETE", "ALIAS")
        ):
            continue
        records = _eval_records(run, job, lookup)
        if records:
            expected_ids = {task["task_id"] for task in final_tasks[job["split"]]}
            if {r["task_id"] for r in records} != expected_ids or any(
                r["protocol_id"] != "O0" for r in records
            ):
                raise ValueError(
                    "Registered final task panel incomplete or protocol changed: " + job["id"]
                )
            for tid in expected_ids:
                if {r["draw_index"] for r in records if r["task_id"] == tid} != set(
                    range(job["draws"])
                ):
                    raise ValueError("Registered final draw panel incomplete: " + job["id"])
            endpoints[job["id"]] = summarize_raw(records, expected_draws=job["draws"])
            raw_endpoints[job["id"]] = records
            from .evaluate import audit_endpoint

            audited = audit_endpoint(
                records,
                final_tasks[job["split"]],
                final_audits[job["split"]],
                pre_zero16_ids=pre_zero[job["parent"]],
            )
            write_json(run / "final_semantics" / (job["id"] + ".json"), audited)
            baseline = parent_endpoints.get(job["parent"] + ":" + job["split"])
            endpoints[job["id"]]["parent_delta"] = (
                endpoints[job["id"]]["pX_family_equal"] - baseline["pX_family_equal"]
                if baseline
                else None
            )
    write_json(run / "FINAL_ENDPOINTS.json", endpoints)
    comparisons = {}
    for comparator in ("SELF_O0", "SELF_SINGLE"):
        paired = []
        mc = []
        missing_cells = []
        cells = {}
        for parent in ("S96", "REP96"):
            for repeat in (0, 1):
                names = [
                    f"eval.{parent}.{repeat}.{arm}.E_test.256" for arm in ("SELF_MIX", comparator)
                ]
                if any(name not in endpoints for name in names):
                    missing_cells.append([parent, repeat])
                    continue
                left, right = [
                    {r["task_id"]: r for r in endpoints[name]["tasks"]} for name in names
                ]
                if set(left) != set(right):
                    raise ValueError("Mismatched endpoint tasks")
                unit = f"{parent}:{repeat}"
                cells[unit] = (
                    endpoints[names[0]]["pX_family_equal"] - endpoints[names[1]]["pX_family_equal"]
                )
                for tid, a in left.items():
                    paired.append(
                        {
                            "family": a["family"],
                            "base_instance_id": tid,
                            "unit": unit,
                            "left": a["pX"],
                            "right": right[tid]["pX"],
                        }
                    )
                for sign, name in ((1, names[0]), (-1, names[1])):
                    group = {}
                    for sample in raw_endpoints[name]:
                        group.setdefault(sample["task_id"], []).append(sample)
                    family_n = Counter(v[0]["family"] for v in group.values())
                    for stream in group.values():
                        mc.append(
                            {
                                "stream_id": digest(sorted(r["request_id"] for r in stream)),
                                "successes": sum(r["public_verifier_pass"] for r in stream),
                                "draws": len(stream),
                                "coefficient": sign / (4 * 2 * family_n[stream[0]["family"]]),
                            }
                        )
        key = "SELF_MIX - " + comparator
        comparisons[key] = {
            "cells": cells,
            "missing_cells": missing_cells,
            "classification": "primary" if comparator == "SELF_O0" else "key secondary",
        }
        if not missing_cells:
            comparisons[key].update(
                paired_cluster_bootstrap(
                    paired,
                    replicates=plan["statistics"]["bootstrap_replicates"],
                    seed=plan["statistics"]["seed"],
                )
            )
            comparisons[key]["fixed_panel_mc"] = finite_panel_mcse(mc)
        else:
            comparisons[key]["status"] = "INCOMPLETE_NO_AGGREGATE_CLAIM"
    for parent in ("S96", "REP96"):
        for repeat in (0, 1):
            for arm in ("GOLD_ALL", "GOLD_MATCH_MIX", "REPLAY_ONLY", "R0_RESET32"):
                step = 32 if arm == "R0_RESET32" else 256
                name = f"eval.{parent}.{repeat}.{arm}.E_test.{step}"
                if name in endpoints:
                    comparisons[name] = {
                        "classification": "exploratory context",
                        "pX": endpoints[name]["pX_family_equal"],
                        "parent_delta": endpoints[name]["parent_delta"],
                        "not_equal_compute_to_SFT": arm == "R0_RESET32",
                    }
    write_json(run / "FINAL_COMPARISONS.json", comparisons)
    final = "# 最终发现、拟合、迁移与保持性\n\n"
    final += f"登记矩阵终态：{counts}；科学数据完整：{scope['scientific_complete']}。技术缺失保留，不能视为零收益或成功实验。\n\n"  # noqa: E501
    final += (
        "1. 发现：见全部J、交集、新增和丢失的 `discovery_summary."
        "json`。16次发现和2次部署组合为不同问题。\n2. 拟合：见逐步loss、曝"
        "光及训练sentinel的 `training_diagnostics.json"
        "` 和 `validation_and_fit.json`。\n3. 迁移：固定s"
        "tep256的E O0；主比较及强基线配对区间在 `FINAL_COMPARIS"
        "ONS.json`。\n4. 保持性：G图像接口及duplicate单独在 `FI"
        "NAL_ENDPOINTS.json`，不与E混成总排行榜。\n\n"
    )
    for name, comparison in list(comparisons.items())[:2]:
        final += (
            f"- {name}: "
            + (
                f"{100 * comparison['difference']:.3f} pp，条件场景95%区间 {[100 * x for x in comparison['interval_95']]} pp。"  # noqa: E501
                if "difference" in comparison
                else "存在缺失单元，不计算完整主估计。"
            )
            + "\n"
        )
    final += "\n区间条件于这两条已有源轨迹和实际运行；两个repeat不是另外两条独立源轨迹。宽区间或跨零不表示等效；负结果不触发追加实验。\n"  # noqa: E501
    (run / "FINAL_FINDINGS_zh.md").write_text(final)
    (run / "DISCUSSION_FOR_CLAUDE_zh.md").write_text(
        "# 讨论边界\n\n事实见FINAL_FINDINGS_zh.md与JSON证据。发"
        "现覆盖、训练拟合、E迁移、G保持性分别报告；GOLD不是性能上界，GOLD_MA"
        "TCH只控制数量和登记结构。\n\n候选解释需结合逐题新增/丢失与训练诊断；单独一个"
        "最终分数不能识别内部机制。当前统计不估计训练种子总体区间；技术缺失和数据不足保持"
        "明确状态。后续路由、奖励、SFT后RL不在本轮自动执行范围。\n"
    )
    return scope


def teacher_portfolios(run, parent, tasks, records, selection):
    """Both unbiased estimates and registered 8-trial offline deployment replay."""
    from collections import defaultdict
    from statistics import mean

    from .config import COMPANION, PROTOCOLS
    from .portfolio import index_bank, pair_utility, replay_verify_first

    bank = index_bank(tasks, records, parent, 0, "E_test")
    task_results = []
    for task in tasks:
        family = task["family"]
        streams = bank[task["task_id"]]
        single = selection["best_single_B2"][family]
        policies = {
            "O0_REPEAT": ["O0", "O0"],
            "FIXED_MIX": ["O0", COMPANION[family]],
            "V_BEST_SINGLE": [single, single],
            "V_BEST_PAIR": selection["best_pair_B2"][family],
        }
        policies.update({p + "_REPEAT": [p, p] for p in PROTOCOLS[family]})
        for name, (first, second) in policies.items():
            left = streams[first]
            right = streams[second]
            same = first == second
            flags1 = [r["public_verifier_pass"] for r in left]
            flags2 = [r["public_verifier_pass"] for r in right]
            trials = [
                replay_verify_first(
                    task, left[2 * r] if same else left[r], right[2 * r + 1] if same else right[r]
                )
                for r in range(8)
            ]
            task_results.append(
                {
                    "parent": parent,
                    "task_id": task["task_id"],
                    "base_instance_id": task["base_instance_id"],
                    "family": family,
                    "policy": name,
                    "protocols": [first, second],
                    "unbiased_success": float(pair_utility(flags1, flags2, same)),
                    "early_stop_expected_calls": 2 - mean(flags1),
                    "offline_trial_success": mean(t["verified"] for t in trials),
                    "offline_trial_calls": mean(t["calls"] for t in trials),
                    "trials": trials,
                }
            )
    groups = defaultdict(lambda: defaultdict(list))
    for row in task_results:
        groups[row["policy"]][row["family"]].append(row)
    summary = {
        name: {
            metric: mean(mean(row[metric] for row in rows) for rows in families.values())
            for metric in (
                "unbiased_success",
                "early_stop_expected_calls",
                "offline_trial_success",
                "offline_trial_calls",
            )
        }
        for name, families in groups.items()
    }
    return {
        "parent": parent,
        "selection_digest": selection["selection_digest"],
        "task_results": task_results,
        "summary": summary,
        "physical_teacher_calls": len(records),
        "deployment_budget": 2,
        "offline_replay_only": True,
        "no_solver_fallback": True,
        "B1_REPEAT_scope": "cross only",
        "no_wall_time_speedup_claim": True,
    }


def validate_completed_artifacts(run, queue):
    """Validate full registered panels before release; never use outcome values."""
    from .cli import _job_root
    from .config import PROTOCOLS
    from .public_tasks import RoleDataset

    run = Path(run)
    lookup = {row["id"]: row for row in queue.rows()}
    for row in lookup.values():
        job = row["payload"]
        if row["status"] == "ALIAS":
            target = (row["result"] or {}).get("alias_of", "")
            if target in lookup and lookup[target]["status"] not in (
                "COMPLETE",
                "ALIAS",
                "NO_VERIFIED_TARGETS_RETURN_PARENT",
            ):
                raise ValueError("Alias target is not available: " + job["id"])
        if row["status"] != "COMPLETE" or job["kind"] not in ("teacher", "evaluate"):
            continue
        role = job.get(
            "role", "validation" if job["split"] in ("T_train", "V_selection") else "teacher_eval"
        )
        tasks = RoleDataset(run / "cohort", role).public(job["split"])
        if job.get("task_ids"):
            tasks = [t for t in tasks if t["task_id"] in job["task_ids"]]
        if job.get("sentinel"):
            tasks = [
                task
                for family in ("cross_series", "trend")
                for task in sorted(
                    [t for t in tasks if t["family"] == family], key=lambda t: t["task_id"]
                )[:8]
            ]
        expected = {
            (task["task_id"], protocol, draw)
            for task in tasks
            for protocol in (
                PROTOCOLS[task["family"]]
                if job["kind"] == "teacher" and job["split"] != "G_guard"
                else ("O0",)
            )
            for draw in range(job["draws"])
        }
        root = _job_root(run, job)
        records = _chunks(root)
        actual = {(r["task_id"], r["protocol_id"], r["draw_index"]) for r in records}
        receipt = read_json(root / "receipt.json")
        if (
            actual != expected
            or len(records) != len(expected)
            or receipt["rows"] != len(expected)
            or receipt["job_identity"] != digest(job)
        ):
            failure = {
                "status": "BLOCKED_ARTIFACT_INTEGRITY",
                "job": job["id"],
                "expected": len(expected),
                "actual": len(records),
                "missing_count": len(expected - actual),
                "extra_count": len(actual - expected),
            }
            write_json(run / "ARTIFACT_INTEGRITY_FAILURE.json", failure)
            raise ValueError("Registered complete sample panel is incomplete: " + job["id"])
    return True


def execution_costs(run, queue, *, reveal=False):
    """Separate retained physical work, logical updates and strict alias savings."""
    from .cli import _job_root
    from .public_tasks import RoleDataset

    run = Path(run)
    jobs = {}
    unknown = []
    aliases = {
        "sft_updates_avoided": 0,
        "sft_target_sequences_avoided": 0,
        "evaluation_calls_avoided": 0,
    }
    for row in queue.rows():
        job = row["payload"]
        root = _job_root(run, job)
        kind = job["kind"]
        if row["status"] in ("ALIAS", "NO_VERIFIED_TARGETS_RETURN_PARENT"):
            if kind == "sft":
                aliases["sft_updates_avoided"] += 256
                aliases["sft_target_sequences_avoided"] += 256 * (
                    4 if job["arm"] == "REPLAY_ONLY" else 16
                )
            if kind == "evaluate":
                role = (
                    "validation" if job["split"] in ("V_selection", "T_train") else "teacher_eval"
                )
                count = len(RoleDataset(run / "cohort", role).public(job["split"]))
                aliases["evaluation_calls_avoided"] += (16 if job.get("sentinel") else count) * job[
                    "draws"
                ]
            continue
        if kind in ("teacher", "evaluate"):
            atomic = list((root / "atomic").glob("*.json"))
            sealed = job["split"] in ("E_test", "G_guard")
            receipt = read_json(root / "receipt.json") if (root / "receipt.json").exists() else None
            costs = {
                "accepted_physical_calls": len(atomic),
                "generated_tokens": 0,
                "generation_seconds": 0.0,
            }
            if sealed and not reveal:
                costs.update(
                    generated_tokens=receipt["generated_tokens"] if receipt else None,
                    generation_seconds=receipt["generation_seconds"] if receipt else None,
                )
                if atomic and not receipt:
                    unknown.append(
                        {
                            "job": job["id"],
                            "reason": "partial sealed tokens/timing withheld until release",
                        }
                    )
            else:
                for path in atomic:
                    raw = read_json(path)["raw"]
                    costs["generated_tokens"] += len(raw.get("token_ids", []))
                    costs["generation_seconds"] += raw.get("elapsed_seconds", 0.0)
            failures = list((root / "attempts").glob("*.json"))
            costs["failed_attempt_records"] = len(failures)
            if failures:
                unknown.append(
                    {
                        "job": job["id"],
                        "reason": (
                            "failed generation GPU time/calls before durable receipt "
                            "may be unmeasured"
                        ),
                    }
                )
            jobs[job["id"]] = costs
        elif kind == "sft":
            entries = [read_json(p) for p in (root / "attempts").glob("sft_*/update*.json")]
            measured = [entry for entry in entries if entry.get("physical_cost_complete")]
            incomplete = [entry for entry in entries if not entry.get("physical_cost_complete")]
            committed = (row["result"] or {}).get("optimizer_updates", 0)
            jobs[job["id"]] = {
                "logical_optimizer_updates": committed,
                "measured_physical_optimizer_updates": len(measured),
                "recomputed_updates_lower_bound": max(0, len(measured) - committed),
                **{
                    key: sum(entry.get(key, 0) or 0 for entry in measured)
                    for key in (
                        "processed_target_sequences",
                        "processed_target_tokens",
                        "update_wall_seconds",
                        "update_gpu_seconds",
                    )
                },
                "incomplete_update_attempts": len(incomplete),
            }
            if incomplete:
                unknown.append(
                    {
                        "job": job["id"],
                        "reason": "failed/in-progress update work has only a lower-bound cost",
                    }
                )
        elif kind == "r0":
            raw = [read_json(path)["raw"] for path in (root / "rollouts").glob("update*/*.json")]
            updates = [read_json(path) for path in root.glob("update*.json")]
            failed = [read_json(path) for path in (root / "attempts").glob("*.json")]
            jobs[job["id"]] = {
                "accepted_on_policy_calls": len(raw),
                "generated_tokens": sum(len(r["token_ids"]) for r in raw),
                "generation_seconds": sum(r.get("elapsed_seconds", 0.0) for r in raw),
                "logical_optimizer_updates": len(updates),
                "gradient_processed_sequences": sum(
                    0 if r["zero_advantages"] else 32 for r in updates
                ),
                "zero_gradient_adam_updates": sum(r["zero_advantages"] for r in updates),
                "failed_attempts": sum(r.get("status") == "FAILED_ATTEMPT" for r in failed),
                "not_equal_compute_to_SFT": True,
            }
            if any(r.get("status") == "FAILED_ATTEMPT" for r in failed):
                unknown.append(
                    {
                        "job": job["id"],
                        "reason": (
                            "failed R0 optimization wall time not fully instrumented; "
                            "raw calls retained/reused"
                        ),
                    }
                )
    bridge = read_json(run / "bridge.json") if (run / "bridge.json").exists() else None
    return {
        "jobs": jobs,
        "strict_alias_and_parent_return_savings": aliases,
        "technical_bridge_draws": sum(
            r.get("generation_draws", 0) for r in bridge.get("parents", {}).values()
        )
        if bridge
        else 0,
        "extra_teacher_likelihood_scoring_calls": 0,
        "physical_cost_complete": not unknown,
        "unknown_or_lower_bound_costs": unknown,
        "retained_negative_outputs_included": True,
        "scope": (
            "Actual retained calls and all SFT attempt journals; "
            "no inference speedup inferred from offline early-stop counts"
        ),
    }
