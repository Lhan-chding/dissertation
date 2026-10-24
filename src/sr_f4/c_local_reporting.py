"""Fixed-denominator reporting for the rich-task, segment-credit C pilot.

Read each raw group once and keep compact statistics only. The new panel is
clustered by root lineage (not rendered world), with equal family weighting.
"""
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .c_pilot_reporting import CHART_METRICS, NATIVE_METRICS, _metric, _unique
from .common import read, rows, write

ARMS = ("PARENT", "J", "EVID", "SELECT", "SHUF")
BLOCKS = (0, 1, 2)
FAMILIES = ("BINDING", "IMPACT", "TIES", "SIGNED", "INTERVAL", "K_DIRECTION")
NEW_PANEL = "C_LOCAL_TEST_R1"
OLD_PANEL = "OLD_TEST_ID"
STEPS = 12
TRAIN_ANSWERS = 18432
EVAL_ANSWERS = 31970
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026102406
COMPLETE = "COMPLETE_EXPLORATORY_C_LOCAL_PILOT"
OBJECTIVE_METRICS = ("native_policy", "evidence_policy", "selection_policy", "kl", "loss")
LIMITATIONS = [
    "这是固定预算的探索性小规模验证，不以条件区间或点估计自动宣布科学通过或追加实验。",
    "每个父模型每个臂只有一条训练路径；三个父模型块不等于独立训练种子重复。",
    "条件95%区间固定已训练模型和生成记录；同一root的所有相关world和题目整体重采样，所有臂和父模型共享重采样。",
    "新题组先平均每题draw，再平均同root题目，再在家族内平均root，最后等权平均六家族及三个父模型。",
    "新题组使用共同的选择声明协议；旧TEST_ID和ChartQA保留原提示与评分，属于历史保持面板，不是同分布新题。",
    "ChartQA仅含b0，历史代码口径及v2宽松口径同时保留；不得推广至三个父模型。",
    "UNKNOWN不是FAIL。S全体PASS率、已知覆盖及已知内PASS率分别报告；不以AI标签回填UNKNOWN。",
    "AI开发审核不是人工盲核；既有C门控不改变。",
    "训练信号、信用L1及局部token系数质量是机制诊断，不是梯度占比，也不能替代跨臂端点效果。",
    "策略代理目标均值可正可负且可能相消，不是梯度范数；一次更新混合多个家族，因此更新级目标只按父模型/臂汇总，不伪造家族归因。",
    "局部辅助信用限制直接目标的token位置，不保证共享参数的变化或最终行为只发生在该段。",
    "无效模型回答保留在固定分母；缺失、额外或身份不符记录以及技术错误使报告不完整。",
]


def validate_plans(tasks, plans):
    """Reject a changed population before inspecting any model response."""
    if set(plans) != set(BLOCKS):
        raise ValueError("Exactly three evaluation block plans required")
    for block, assignments in plans.items():
        if [a["group_index"] for a in assignments] != list(range(len(assignments))):
            raise ValueError("Group indices must be consecutive and unique")
        counts = Counter(a["panel"] for a in assignments)
        expected = {NEW_PANEL: 384, OLD_PANEL: 128}
        if block == 0:
            expected["CHARTQA"] = 250
        if counts != expected or len({a["qid"] for a in assignments}) != len(assignments):
            raise ValueError("Wrong panel inventory or duplicate evaluation qid")
        for a in assignments:
            task = tasks[a["qid"]]
            chart = a["panel"] == "CHARTQA"
            if a["G"] != (1 if chart else 4) or a["protocol"] != ("chartqa_plain" if chart else "structured"):
                raise ValueError("Changed draw budget or generation protocol")
            if len(a["seeds"]) != a["G"] or any(type(s) is not int for s in a["seeds"]):
                raise ValueError("Incomplete seed identities")
            if a["panel"] == NEW_PANEL and (task["split"] != NEW_PANEL or task["cell"] not in FAMILIES):
                raise ValueError("Changed new family/split identity")
            if a["panel"] == OLD_PANEL and task["pool"] != "TEST_ID":
                raise ValueError("Retention must retain original TEST_ID pool")
            if chart and task["pool"] != "CHARTQA":
                raise ValueError("External panel task is not ChartQA")
        new = [tasks[a["qid"]] for a in assignments if a["panel"] == NEW_PANEL]
        if Counter(t["cell"] for t in new) != dict.fromkeys(FAMILIES, 64):
            raise ValueError("Each of six families requires 64 new test queries")
        root_families = defaultdict(set)
        for family in FAMILIES:
            selected = [t for t in new if t["cell"] == family]
            roots = Counter(t["root_id"] for t in selected)
            if len(roots) != 16 or set(roots.values()) != {4}:
                raise ValueError("Each family requires sixteen root lineages with four queries")
            for t in selected:
                root_families[t["root_id"]].add(family)
        if any(len(v) != 1 for v in root_families.values()):
            raise ValueError("Root lineage cannot cross independently resampled family strata")
    def paired(a):
        return a["qid"], a["panel"], a["G"], a["protocol"]
    base = [paired(a) for a in plans[0] if a["panel"] != "CHARTQA"]
    if any([paired(a) for a in plans[b]] != base for b in (1, 2)):
        raise ValueError("Native evaluation is not paired across parents")
    total = len(ARMS) * sum(a["G"] for plan in plans.values() for a in plan)
    if total != EVAL_ANSWERS:
        raise ValueError("Fixed evaluation answer budget changed")
    return total


def completion_inventory(root):
    root = Path(root)
    registry = read(root / "TASKS.json")
    registry = registry.get("tasks", []) if isinstance(registry, dict) else registry
    indexed = _unique(registry, "task_id")
    expected = {}
    for block in BLOCKS:
        for arm in ARMS:
            if arm != "PARENT":
                expected[f"CLC-T-b{block}-{arm}"] = dict(kind="c_local_train", block=block, arm=arm)
            for shard in range(4):
                expected[f"CLC-E-b{block}-{arm}-s{shard}"] = dict(kind="c_local_eval", block=block, arm=arm, shard=shard)
    issues = []
    if set(indexed) != {*expected, "CLC-REPORT"} or indexed.get("CLC-REPORT", {}).get("kind") != "c_local_report":
        issues.append(dict(reason="WRONG_REGISTERED_TASK_INVENTORY"))
    for task_id, identity in expected.items():
        if any(indexed.get(task_id, {}).get(k) != v for k, v in identity.items()):
            issues.append(dict(task_id=task_id, reason="REGISTERED_TASK_IDENTITY_MISMATCH"))
        try:
            receipt = read(root / "tasks" / task_id / "COMPLETE.json")
            if receipt.get("status") != "COMPLETE" or receipt.get("task_id") != task_id:
                raise ValueError("Completion receipt identity/status differs")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            issues.append(dict(task_id=task_id, reason="MISSING_OR_INVALID_COMPLETE_RECEIPT", detail=str(exc)))
    return dict(expected_training_tasks=12, expected_evaluation_tasks=60, issues=issues)


def _checked_records(saved, task, assignment, *, block, arm, step, group_id, panel):
    if saved["group_id"] != group_id or saved["block"] != block:
        raise ValueError("Atomic group identity differs")
    records = saved["rows"]
    if len(records) != assignment["G"]:
        raise ValueError("Missing or extra draw")
    for draw, record in enumerate(records):
        identity = dict(qid=task["qid"], block=block, arm=arm, panel=panel,
                        protocol=assignment["protocol"], draw=draw, seed=assignment["seeds"][draw],
                        group_id=group_id, row_id=f"{group_id}/d{draw:02d}", step=step, split=task["split"])
        if panel != "CHARTQA":
            identity["world_id"] = task["world_id"]
        if any(record.get(k) != v for k, v in identity.items()):
            raise ValueError(f"Response identity differs at draw {draw}")
        if record.get("generation_status") != "COMPLETE" or record.get("technical_validation_errors"):
            raise ValueError("Technical generation failure retained")
        if not isinstance(record.get("full_text"), str):
            raise ValueError("Missing raw response")
        for metric in CHART_METRICS if panel == "CHARTQA" else NATIVE_METRICS:
            _metric(record["score"], metric)
    return records


def _semantic(measured):
    result = {d: measured[d] for d in "PSOE"}
    if any(v not in {"PASS", "FAIL", "UNKNOWN"} for v in result.values()):
        raise ValueError("Invalid semantic label")
    return result


def scan_evaluations(root, tasks, plans, *, verifier=None):
    from .c_local_credit import VERIFIER_VERSION, measure

    verifier = verifier or measure
    root = Path(root)
    compact, issues, expected_paths = [], [], set()
    groups_read = 0
    for block, assignments in sorted(plans.items()):
        for arm in ARMS:
            for assignment in assignments:
                index, panel = assignment["group_index"], assignment["panel"]
                path = root / "EVAL" / f"b{block}" / arm / f"shard{index % 4}" / "groups" / f"{index:05d}.json"
                expected_paths.add(path)
                try:
                    task = tasks[assignment["qid"]]
                    step = 0 if arm == "PARENT" else STEPS
                    group_id = f"CLC-E-b{block}-{arm}/s{step:03d}/g{index:05d}"
                    records = _checked_records(read(path), task, assignment, block=block, arm=arm,
                                               step=step, group_id=group_id, panel=panel)
                    group = []
                    for record in records:
                        metrics = {m: _metric(record["score"], m)
                                   for m in (CHART_METRICS if panel == "CHARTQA" else NATIVE_METRICS)}
                        semantic = {}
                        if panel == NEW_PANEL:
                            measured = verifier(task, record)
                            if measured.get("verifier_version") != VERIFIER_VERSION:
                                raise ValueError("Frozen semantic measurement version differs")
                            semantic = _semantic(measured)
                        group.append(dict(block=block, arm=arm, panel=panel, qid=task["qid"], draw=record["draw"],
                                          root_id=task["root_id"] if panel == NEW_PANEL else task.get("world_id", task["qid"]),
                                          world_id=task.get("world_id", task["qid"]), family=task.get("family", "EXTERNAL"),
                                          cell=task.get("cell"), variant=task.get("variant"), score=metrics, semantic=semantic,
                                          truncated=bool(record.get("truncated", record.get("finish_reason") == "length")),
                                          completion_tokens=record.get("completion_token_count", len(record.get("tokens", [])))))
                    compact.extend(group)
                    groups_read += 1
                except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                    issues.append(dict(path=str(path.relative_to(root)), reason="INVALID_OR_MISSING_GROUP", detail=str(exc)))
    found = {p for p in (root / "EVAL").glob("b*/*/shard*/groups/*.json") if ".chunk" not in p.name}
    issues.extend(dict(path=str(p.relative_to(root)), reason="UNEXPECTED_GROUP_FILE") for p in sorted(found - expected_paths))
    return compact, dict(issues=issues, expected_groups=len(expected_paths), validated_group_files=groups_read,
                         extracted_answers=len(compact), expected_answers=sum(a["G"] for p in plans.values() for a in p) * len(ARMS))


def _root_values(items, metric):
    queries = defaultdict(list)
    for r in items:
        family = r["cell"] if r["panel"] == NEW_PANEL else "single_stratum"
        queries[r["block"], r["arm"], family, r["root_id"], r["qid"]].append(r["score"][metric])
    roots = defaultdict(list)
    for (*key, _qid), values in queries.items():
        roots[tuple(key)].append(sum(values) / len(values))
    return {key: sum(value) / len(value) for key, value in roots.items()}


def summarize_stratum(items, *, name, panel, bootstrap_repeats=BOOTSTRAPS):
    """Paired root bootstrap, stratified/equal weighted by new question family."""
    blocks = sorted({r["block"] for r in items})
    if not items or set(blocks) != ({0} if panel == "CHARTQA" else set(BLOCKS)):
        raise ValueError("Incomplete block coverage in analysis stratum")
    families = sorted({r["cell"] for r in items}) if panel == NEW_PANEL else ["single_stratum"]
    roots = {f: sorted({r["root_id"] for r in items if panel != NEW_PANEL or r["cell"] == f}) for f in families}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = {f: rng.integers(0, len(v), size=(bootstrap_repeats, len(v))) for f, v in roots.items()}
    endpoints, comparisons, measurement = [], [], []
    grouped = defaultdict(list)
    for row in items:
        grouped[row["block"], row["arm"]].append(row)
    for metric in CHART_METRICS if panel == "CHARTQA" else ("value_correct", "J"):
        values = _root_values(items, metric)
        expected = {(b, a, f, w) for b in blocks for a in ARMS for f, ws in roots.items() for w in ws}
        if set(values) != expected:
            raise ValueError("Unpaired root/arm coverage cannot produce scientific comparisons")
        cubes = {f: np.array([[[values[b, a, f, w] for w in ws] for a in ARMS] for b in blocks]) for f, ws in roots.items()}
        for bi, block in enumerate([*blocks, "equal_blocks"]):
            matrices = {f: cube[bi] if bi < len(blocks) else cube.mean(axis=0) for f, cube in cubes.items()}
            for ai, arm in enumerate(ARMS):
                endpoints.append(dict(stratum=name, panel=panel, block=block, arm=arm, metric=metric,
                                      mean=float(np.mean([m[ai].mean() for m in matrices.values()])),
                                      roots=sum(map(len, roots.values())), families=len(families),
                                      answers=sum(len(grouped[b, arm]) for b in blocks) if block == "equal_blocks" else len(grouped[block, arm])))
            for left, right in [("SELECT", a) for a in ("SHUF", "J", "EVID")] + [(a, "PARENT") for a in ARMS[1:]]:
                delta = {f: matrix[ARMS.index(left)] - matrix[ARMS.index(right)] for f, matrix in matrices.items()}
                samples = np.mean([d[resamples[f]].mean(axis=1) for f, d in delta.items()], axis=0)
                comparisons.append(dict(stratum=name, panel=panel, block=block, arm=left, reference=right, metric=metric,
                                        difference=float(np.mean([d.mean() for d in delta.values()])),
                                        ci95=[float(v) for v in np.quantile(samples, (0.025, 0.975))],
                                        roots=sum(map(len, roots.values())), families=len(families), blocks=blocks,
                                        bootstrap_repeats=bootstrap_repeats, bootstrap_seed=BOOTSTRAP_SEED,
                                        unit="PAIRED_ROOT_LINEAGE_SHARED_ACROSS_BLOCKS_AND_ARMS",
                                        conditional_on_fixed_trained_models=True,
                                        primary=name == "new/all" and metric == "value_correct" and left == "SELECT" and right == "SHUF" and block == "equal_blocks"))
    for (block, arm), values in sorted(grouped.items()):
        entry = dict(stratum=name, panel=panel, block=block, arm=arm, answers=len(values),
                     truncated=sum(r["truncated"] for r in values),
                     mean_completion_tokens=sum(r["completion_tokens"] for r in values) / len(values))
        for d in "PSOE":
            counts = Counter(r["semantic"].get(d) for r in values)
            if counts[None] == len(values):
                continue
            known = counts["PASS"] + counts["FAIL"]
            entry[d] = dict(counts={s: counts[s] for s in ("PASS", "FAIL", "UNKNOWN")},
                            all_denominator=len(values), known_denominator=known, known_coverage=known / len(values),
                            pass_all=counts["PASS"] / len(values), pass_given_known=counts["PASS"] / known if known else None)
        if panel != "CHARTQA":
            entry["native_metrics"] = {m: sum(r["score"][m] for r in values) / len(values) for m in NATIVE_METRICS}
        measurement.append(entry)
    return dict(endpoints=endpoints, comparisons=comparisons, measurement=measurement)


def analyze(items):
    strata = defaultdict(list)
    for row in items:
        if row["panel"] == NEW_PANEL:
            keys = ("new/all", "new/family/" + row["cell"])
        elif row["panel"] == OLD_PANEL:
            keys = ("old/all", "old/family/" + row["family"], f"old/family_variant/{row['family']}/{row['variant']}")
        else:
            keys = ("chartqa/b0",)
        for key in keys:
            strata[key].append(row)
    result = dict(endpoints=[], comparisons=[], measurement=[])
    for name, values in sorted(strata.items()):
        for key, entries in summarize_stratum(values, name=name, panel=values[0]["panel"]).items():
            result[key].extend(entries)
    return result


def _mask_diagnostics(records, saved):
    """Audit saved generated-token masks without reloading a tokenizer/model."""
    masks = saved["record_masks"]
    if len(masks) != len(records):
        raise ValueError("Missing token mask record")
    counts = Counter()
    for i, (record, mask) in enumerate(zip(records, masks, strict=True)):
        if mask.get("row_id") != record["row_id"]:
            raise ValueError("Token mask row identity differs")
        length = len(record["tokens"])
        for channel in ("evidence", "selection"):
            values = mask[channel + "_mask"]
            if len(values) != length or any(type(v) is not bool for v in values):
                raise ValueError("Malformed generated-token mask")
            token_count = sum(values)
            reported = mask["diagnostics"].get("span_token_counts", {}).get(channel, 0)
            if reported != token_count:
                raise ValueError("Mask token count differs from recorded diagnostic")
            coefficient = saved[channel + "_advantages"][i]
            counts[channel + "_mask_tokens"] += token_count
            counts[channel + "_mapped_rows"] += token_count > 0
            counts[channel + "_applied_rows"] += token_count > 0 and coefficient != 0
            counts[channel + "_unmapped_nonzero_rows"] += token_count == 0 and coefficient != 0
            # Span means divide each token's coefficient by its own span length.
            counts[channel + "_normalized_coefficient_l1_before_TIS"] += abs(coefficient) if token_count else 0
            counts["mask_reason/" + channel + "/" + mask["diagnostics"].get("channel_reasons", {}).get(channel, mask["diagnostics"].get("reason", "UNREPORTED"))] += 1
        if any(a and b for a, b in zip(mask["evidence_mask"], mask["selection_mask"], strict=True)):
            raise ValueError("Auxiliary token masks overlap")
        counts["generated_tokens"] += length
    return counts


def _conditional_swappability(saved):
    """Potential S exchanges inside P=PASS, same-J strata, before realized RNG."""
    counts = Counter(conditional_swappable_groups=0, conditional_swappable_strata=0,
                     conditional_swappable_rows=0, conditional_realized_changed_rows=0,
                     conditional_realized_changed_groups=0)
    labels = saved["labels"]
    for stratum in saved["shuffle"]["within_j_p_strata"]:
        if stratum["p"] != "PASS":
            continue
        destinations, sources = stratum["destination_indices"], stratum["source_indices"]
        states = {labels[i]["S"] for i in destinations}
        if saved["active"] and states == {"PASS", "FAIL"}:
            counts["conditional_swappable_strata"] += 1
            counts["conditional_swappable_rows"] += len(destinations)
        if saved["shuffle"]["applied"]:
            counts["conditional_realized_changed_rows"] += sum(
                labels[d]["S"] != labels[s]["S"] for d, s in zip(destinations, sources, strict=True))
    counts["conditional_swappable_groups"] = int(counts["conditional_swappable_strata"] > 0)
    counts["conditional_realized_changed_groups"] = int(counts["conditional_realized_changed_rows"] > 0)
    return counts


def _objective_diagnostics(update):
    """Retain finite actual surrogate values, not a proxy for gradient norms."""
    values = {metric: update[metric] for metric in OBJECTIVE_METRICS}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
        raise ValueError("Missing or nonfinite actual optimizer objective diagnostic")
    # Each stored update is already an equal mean over the same 128 responses.
    return Counter(objective_updates=1, **{"objective/" + k + "_sum": v for k, v in values.items()})


def training_mechanism(root, tasks):
    """Validate raw training identities, actual updates and recomputed credit once."""
    from srf4_ref.sampling import seed_for

    from . import c_local_data as data
    from .c_local_credit import group_credit, measure

    root = Path(root)
    issues, strata, signal_roots, signal_qids = [], defaultdict(Counter), defaultdict(set), defaultdict(set)
    swappable_roots, swappable_qids = defaultdict(set), defaultdict(set)
    answers = updates = 0
    expected_paths = set()
    for block in BLOCKS:
        plan = read(root / "DATA" / f"TRAIN_PLAN_b{block}.json")
        if set(plan) != {str(s) for s in range(1, STEPS + 1)}:
            raise ValueError("Missing or extra planned training update")
        for arm in ARMS[1:]:
            directory = root / "RUNS/C_LOCAL" / f"b{block}" / arm
            for step in range(1, STEPS + 1):
                path = directory / "credit" / f"{step:03d}.json"
                try:
                    assigned = plan[str(step)]
                    if assigned != data.training_assignments(list(tasks.values()), block, step):
                        raise ValueError("Registered training schedule differs")
                    if read(directory / "assignments" / path.name) != dict(step=step, groups=assigned):
                        raise ValueError("Consumed training schedule differs")
                    receipt = read(path)
                    if any(receipt.get(k) != v for k, v in dict(schema="srf4-c-local-credit-v1", step=step, block=block, arm=arm).items()):
                        raise ValueError("Credit receipt identity differs")
                    credit = _unique(receipt["groups"], "group_index")
                    if len(assigned) != 16 or set(credit) != set(range(16)):
                        raise ValueError("Expected exactly sixteen credit groups")
                    update = read(directory / "updates" / path.name)
                    if any(update.get(k) != v for k, v in dict(step=step, block=block, arm=arm, sequence_count=128, optimizer_updates=1).items()):
                        raise ValueError("Missing or invalid committed optimizer update")
                    step_mask_counts = Counter()
                    step_vectors = {k: [] for k in ("native_advantages", "evidence_advantages", "selection_advantages")}
                    for assignment in assigned:
                        index = assignment["group_index"]
                        raw = directory / "rollouts" / f"{step:03d}" / f"{index:05d}.json"
                        expected_paths.add(raw)
                        task = tasks[assignment["qid"]]
                        group_id = f"CLC-T-b{block}-{arm}/s{step:03d}/g{index:05d}"
                        records = _checked_records(read(raw), task, assignment, block=block, arm=arm,
                                                   step=step, group_id=group_id, panel=task["split"])
                        if len(records) != 8:
                            raise ValueError("Training must preserve G8")
                        if any(r.get("model_identity", {}).get("step") != step - 1 or r.get("model_identity", {}).get("run_id") != f"CLC-T-b{block}-{arm}" for r in records):
                            raise ValueError("Training responses did not use committed preceding policy")
                        labels = [measure(task, r) for r in records]
                        seed = seed_for(data.MASTER, "c-local-selection-shuffle", block, step, index, task["qid"])
                        recalculated = json.loads(json.dumps(group_credit(task, records, labels, arm=arm, seed=seed)))
                        saved = credit[index]
                        if any(saved.get(k) != v for k, v in recalculated.items()):
                            raise ValueError("Stored group credit differs from deterministic recomputation")
                        native = saved["native_advantages"]
                        evidence = saved["evidence_advantages"]
                        selection = saved["selection_advantages"]
                        if any(len(v) != 8 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in (native, evidence, selection)):
                            raise ValueError("Malformed credit vectors")
                        mask_counts = _mask_diagnostics(records, saved)
                        step_mask_counts.update(mask_counts)
                        for field in step_vectors:
                            step_vectors[field].extend(saved[field])
                        signal = bool(saved["selection_signal"])
                        counts = Counter(groups=1, answers=8, candidate_rows=len(saved["candidate_indices"]),
                                         known_rows=len(saved["known_indices"]), active_groups=bool(saved["active"]),
                                         selection_signal_groups=signal,
                                         zero_native_J_selection_signal_groups=bool(saved["zero_native_J_selection_signal"]),
                                         native_l1=sum(map(abs, native)), evidence_l1=sum(map(abs, evidence)),
                                         selection_l1=sum(map(abs, selection)),
                                         select_minus_shuffle_l1=saved["select_minus_shuffle_l1"],
                                         actual_shuffle_changed_rows=saved["shuffle"]["changed_count"] if arm == "SHUF" else 0,
                                         actual_shuffle_changed_groups=arm == "SHUF" and saved["shuffle"]["changed_count"] > 0,
                                         nonzero_evidence_groups=any(evidence), nonzero_selection_groups=any(selection))
                        counts.update({"token/" + k: v for k, v in mask_counts.items()})
                        conditional = _conditional_swappability(saved)
                        counts.update(conditional)
                        counts["reason/" + saved["eligibility_reason"]] += 1
                        for label in labels:
                            for d, value in _semantic(label).items():
                                counts[d + "/" + value] += 1
                        for key in (f"b{block}/{arm}/all", f"b{block}/{arm}/family/{task['cell']}"):
                            strata[key].update(counts)
                            if signal:
                                signal_roots[key].add(task["root_id"])
                                signal_qids[key].add(task["qid"])
                            if conditional["conditional_swappable_groups"]:
                                swappable_roots[key].add(task["root_id"])
                                swappable_qids[key].add(task["qid"])
                        answers += 8
                    if any(update.get(k) != v for k, v in step_vectors.items()):
                        raise ValueError("Actual optimizer coefficient vectors differ from credit receipt")
                    if any(update.get(k) != step_mask_counts[k] for k in ("evidence_mask_tokens", "selection_mask_tokens", "evidence_applied_rows", "selection_applied_rows")):
                        raise ValueError("Actual optimizer mask counts differ from credit receipt")
                    strata[f"b{block}/{arm}/all"].update(_objective_diagnostics(update))
                    updates += 1
                except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                    issues.append(dict(path=str(path.relative_to(root)), reason="INVALID_OR_MISSING_TRAINING_RECORD", detail=str(exc)))
    found = {p for p in (root / "RUNS/C_LOCAL").glob("b*/*/rollouts/*/*.json") if ".chunk" not in p.name}
    issues.extend(dict(path=str(p.relative_to(root)), reason="UNEXPECTED_TRAINING_GROUP") for p in sorted(found - expected_paths))
    if answers != TRAIN_ANSWERS or updates != 144:
        issues.append(dict(reason="TRAINING_BUDGET_DIFFERS", answers=answers, updates=updates, expected_answers=TRAIN_ANSWERS, expected_updates=144))
    output = {}
    for key, value in sorted(strata.items()):
        output[key] = dict(value, unique_signal_roots=len(signal_roots[key]), unique_signal_qids=len(signal_qids[key]),
                           unique_conditional_swappable_roots=len(swappable_roots[key]),
                           unique_conditional_swappable_qids=len(swappable_qids[key]))
        if value["objective_updates"]:
            output[key]["objective_mean"] = {metric: value["objective/" + metric + "_sum"] / value["objective_updates"]
                                             for metric in OBJECTIVE_METRICS}
    return dict(issues=issues, answers=answers, updates=updates, strata=output,
                signal_definition="Common reliable cohort has genuine selection variation; final applied auxiliary can differ by arm",
                conditional_swappable_definition="An active P=PASS, same-J shuffle stratum contains both S=PASS and S=FAIL, independently of the one realized permutation",
                objective_definition="Actual 128-response update surrogate means, averaged equally over committed updates; not gradient norms or family-level attribution",
                token_mass_is_not_gradient_fraction=True)


def _percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def write_report(root, result):
    lines = ["# C 新题组与局部信用小规模验证", "", f"状态：`{result['status']}`。", ""]
    if result["status"] != COMPLETE:
        lines += ["完整性检查未通过，不发布部分结果的科学结论。", f"发现 {len(result['issues'])} 项问题，详见 ANALYSIS/SUMMARY.json。"]
    else:
        primary = next(c for c in result["comparisons"] if c["primary"])
        lines += [f"主比较 SELECT−SHUF：新题组最终数值正确率差 {_percent(primary['difference'])} 个百分点，条件95%区间 [{_percent(primary['ci95'][0])}, {_percent(primary['ci95'][1])}]。",
                  "训练18,432条、评测31,970条，总计50,402条生成回答；144次优化器更新全部完成。",
                  "正值表示SELECT点估计更高，负值表示更低；区间跨零不表示等效。此结果检验题组和局部信用的组合，不能独立识别两个改动各自的因果贡献。", "",
                  "## 主要提升与退化", "", "| 面板 | 父模型块 | 对照 | 指标 | 差值(pp) | 条件95%区间 |", "|---|---|---|---|---:|---|"]
        for r in result["comparisons"]:
            if r["arm"] == "SELECT" and r["stratum"] in {"new/all", "old/all", "chartqa/b0"}:
                lines.append(f"| {r['stratum']} | {r['block']} | {r['reference']} | {r['metric']} | {_percent(r['difference'])} | [{_percent(r['ci95'][0])}, {_percent(r['ci95'][1])}] |")
        lines += ["", "## 各家族、父模型及端点均值", "", "| 面板/家族 | 块 | 臂 | 指标 | 均值(%) | 回答数 | root数 |", "|---|---|---|---|---:|---:|---:|"]
        for r in result["endpoints"]:
            lines.append(f"| {r['stratum']} | {r['block']} | {r['arm']} | {r['metric']} | {_percent(r['mean'])} | {r['answers']} | {r['roots']} |")
        lines += ["", "## 所有配对比较", "", "| 面板/家族 | 块 | 比较 | 指标 | 差值(pp) | 条件95%区间 |", "|---|---|---|---|---:|---|"]
        for r in result["comparisons"]:
            lines.append(f"| {r['stratum']} | {r['block']} | {r['arm']}−{r['reference']} | {r['metric']} | {_percent(r['difference'])} | [{_percent(r['ci95'][0])}, {_percent(r['ci95'][1])}] |")
        lines += ["", "## 新协议测量与原生证据评分", "", "S依据新协议判定；证据评分保留原生口径。P/S/O/E全部状态及原生其他分数见摘要。", "", "| 分层 | 块 | 臂 | 全体 | S已知 | S全体PASS(%) | S已知内PASS(%) | 证据有效(%) |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in result["measurement"]:
            if "S" in r:
                s = r["S"]
                lines.append(f"| {r['stratum']} | {r['block']} | {r['arm']} | {s['all_denominator']} | {s['known_denominator']} | {_percent(s['pass_all'])} | {_percent(s['pass_given_known'])} | {_percent(r['native_metrics']['certificate'])} |")
        lines += ["", "## 实际训练信号", "", "下表区分组数、独立root和题目数。信用L1为回答级系数，token级质量在摘要的token字段；均非真实参数梯度占比。", "", "| 分层 | 组 | 活跃 | 有选择分歧 | 独立signal root | 独立signal qid | Native L1 | Evidence L1 | Selection L1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, r in result["training_mechanism"]["strata"].items():
            fields = ("groups", "active_groups", "selection_signal_groups", "unique_signal_roots", "unique_signal_qids", "native_l1", "evidence_l1", "selection_l1")
            lines.append("| " + key + " | " + " | ".join(f"{r.get(k, 0):.5g}" for k in fields) + " |")
        lines += ["", "## 保持J/P条件后的可交换选择信号", "", "仅当同一J且P=PASS的可靠子层同时含S=PASS与S=FAIL，才记为可交换。可交换机会独立于单次SHUF抽签；实际置换允许不改变标签。", "", "| 分层 | 可交换组 | 独立root | 独立qid | 可交换子层 | 实际有效改变组 | 实际有效改变行 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, r in result["training_mechanism"]["strata"].items():
            fields = ("conditional_swappable_groups", "unique_conditional_swappable_roots", "unique_conditional_swappable_qids", "conditional_swappable_strata", "conditional_realized_changed_groups", "conditional_realized_changed_rows")
            lines.append("| " + key + " | " + " | ".join(str(r.get(k, 0)) for k in fields) + " |")
        lines += ["", "## 实际优化目标与局部覆盖", "", "以下为实际提交更新的代理目标均值，按每次128回答的更新等权平均；数值可相消，不能当作梯度范数或梯度贡献比例。更新混合多个家族，故只按父模型和臂报告。token和行数为全训练路径累计，已核对实际优化器回执。", "", "| 路径 | 更新数 | Native policy | Evidence policy | Selection policy | KL | 总loss | Evidence tokens/生效行 | Selection tokens/生效行 |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for key, r in result["training_mechanism"]["strata"].items():
            if "objective_mean" in r:
                fields = " | ".join(f"{r['objective_mean'][m]:.6g}" for m in OBJECTIVE_METRICS)
                lines.append(f"| {key} | {r['objective_updates']} | {fields} | {r.get('token/evidence_mask_tokens', 0)}/{r.get('token/evidence_applied_rows', 0)} | {r.get('token/selection_mask_tokens', 0)}/{r.get('token/selection_applied_rows', 0)} |")
    lines += ["", "## 解释边界", "", *["- " + s for s in LIMITATIONS], "",
              "交付为轻量报告、统计摘要、覆盖检查与引用清单。原始回答、逐token记录、模型及优化器保留在服务器；未创建或下载完整大归档。", ""]
    (Path(root) / "REPORT_ZH.md").write_text("\n".join(lines))


def delivery_manifest(root, status):
    root = Path(root)
    names = ["REPORT_ZH.md", "ANALYSIS/SUMMARY.json", "ANALYSIS/COVERAGE.json", "CONFIG.json", "FROZEN.json", "TASKS.json", "DATA/TASKS.jsonl"]
    names += [f"DATA/EVAL_GROUPS_b{b}.jsonl" for b in BLOCKS]
    return dict(status=status, root=str(root), files=[dict(path=n, bytes=(root / n).stat().st_size) for n in names if (root / n).is_file()],
                archive_created=False, server_only=["RUNS/C_LOCAL", "EVAL"],
                endpoint_references=[dict(block=b, arm=a, step=0 if a == "PARENT" else STEPS,
                                          reference="CONFIG.json#parents" if a == "PARENT" else f"RUNS/C_LOCAL/b{b}/{a}") for b in BLOCKS for a in ARMS])


def report(root):
    from .c_local import build_eval_groups, require_frozen

    root = Path(root).resolve()
    result = dict(status="INCOMPLETE_TECHNICAL", issues=[], expected_evaluation_answers=EVAL_ANSWERS,
                  expected_training_answers=TRAIN_ANSWERS, primary="new value_correct SELECT minus SHUF; equal root/family/parent weighting",
                  inference_limits=LIMITATIONS, old_c_gate_modified=False, human_audit_count=0)
    try:
        require_frozen(root)
        tasks = _unique(rows(root / "DATA/TASKS.jsonl"), "qid")
        plans = {b: list(rows(root / f"DATA/EVAL_GROUPS_b{b}.jsonl")) for b in BLOCKS}
        validate_plans(tasks, plans)
        if any(plans[b] != build_eval_groups(list(tasks.values()), b) for b in BLOCKS):
            raise ValueError("Evaluation plan differs from frozen deterministic schedule")
        completion = completion_inventory(root)
        result["completion_inventory"] = completion
        values, coverage = scan_evaluations(root, tasks, plans)
        mechanism = training_mechanism(root, tasks)
        result.update(coverage=coverage, training_mechanism=mechanism)
        for section in (completion, coverage, mechanism):
            result["issues"].extend(section["issues"])
        if coverage["extracted_answers"] != EVAL_ANSWERS:
            result["issues"].append(dict(reason="EVALUATION_ANSWER_COUNT_DIFFERS"))
        if not result["issues"]:
            result.update(analyze(values), status=COMPLETE)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        result["issues"].append(dict(reason="REPORT_INPUT_VALIDATION_FAILED", detail=str(exc)))
    write(root / "ANALYSIS/SUMMARY.json", result)
    write(root / "ANALYSIS/COVERAGE.json", dict(status=result["status"], **result.get("coverage", {}), all_issues=result["issues"]))
    write_report(root, result)
    write(root / "ANALYSIS/DELIVERY_MANIFEST.json", delivery_manifest(root, result["status"]))
    return dict(status=result["status"], issues=len(result["issues"]), expected_evaluation_answers=EVAL_ANSWERS,
                expected_training_answers=TRAIN_ANSWERS, summary=str(root / "ANALYSIS/SUMMARY.json"),
                report=str(root / "REPORT_ZH.md"), manifest=str(root / "ANALYSIS/DELIVERY_MANIFEST.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if report(args.root)["status"] != COMPLETE:
        raise SystemExit("INCOMPLETE_TECHNICAL: see ANALYSIS/SUMMARY.json")


if __name__ == "__main__":
    main()
