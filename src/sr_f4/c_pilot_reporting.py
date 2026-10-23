"""Fixed-inventory C F-prime pilot reporting, without loading model weights.

Raw groups are read once, reduced to small metrics, and released. Bootstrap
resamples paired worlds shared by all arms and blocks, never answer rows.
"""
# ruff: noqa: E501, RUF001

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import read, rows, write

ARMS = ("PARENT", "J", "EVID", "SELECT", "SELECT_SHUF")
BLOCKS = (0, 1, 2)
CELLS = ("T6-SEPARATED", "T6-BOUNDARY", "T8-SEPARATED", "T8-BOUNDARY")
NEW_PANEL = "C_FP_TEST_R1"
OLD_PANEL = "OLD_TEST_ID"
PANELS = (NEW_PANEL, OLD_PANEL, "CHARTQA")
VERSION = "srf4-observable-v3-programmatic-dev-r6"
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026102311
NATIVE_METRICS = (
    "value_correct", "J", "presentation", "finish", "certificate_truth",
    "certificate_sufficient", "certificate",
)
CHART_METRICS = ("historical_code.relaxed_accuracy", "v2.relaxed_accuracy")
COMPLETE = "COMPLETE_EXPLORATORY_C_FP_PILOT"
LIMITATIONS = [
    "探索性、有固定预算的小规模试验；没有正式功效证明或自动科学接受门槛。",
    "每个父模型、每个训练臂只有一条训练路径；三个父模型块不是新增独立训练种子。",
    "95%区间条件于已训练模型与固定生成记录，只对配对world重采样；不是训练种子置信区间。",
    "所有臂和块共享同一批重采样world；先按query平均draw、再按world平均query，最后等权平均块。",
    "旧TEST_ID及ChartQA是复用的保留性能面板；ChartQA只有b0，不能推广至三个父模型。",
    "AI开发审核不是人工真值或独立供应商认证；旧C门控保持原状态。",
    "S未知不等于S失败；S已知覆盖、已知内PASS率和全体PASS率分别报告。",
    "训练信用L1及置换计数是在各臂自身rollout上的机制描述，不能充当跨臂因果训练效果。",
    "所有无效模型回答保留在固定分母内，使用原生分数；结构缺失或技术异常使整份结果INCOMPLETE_TECHNICAL。",
]


def _metric(score, name):
    value = score
    for part in name.split("."):
        value = value[part]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid finite metric {name}")
    return float(value)


def _unique(items, key):
    result = {}
    for item in items:
        value = item[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = item
    return result


def validate_plans(tasks, plans):
    """Validate the fixed evaluation population before looking at responses."""
    if set(plans) != set(BLOCKS):
        raise ValueError("Exactly three evaluation block plans required")
    for block, assignments in plans.items():
        indices = [a["group_index"] for a in assignments]
        if indices != list(range(len(assignments))):
            raise ValueError("Group indices must be complete, unique, and consecutive")
        counts = Counter(a["panel"] for a in assignments)
        expected = {NEW_PANEL: 256, OLD_PANEL: 128}
        if block == 0:
            expected["CHARTQA"] = 250
        if counts != expected:
            raise ValueError(f"Wrong panel inventory for b{block}: {dict(counts)}")
        if len({a["qid"] for a in assignments}) != len(assignments):
            raise ValueError("Duplicate task in evaluation plan")
        for a in assignments:
            task = tasks[a["qid"]]
            chart = a["panel"] == "CHARTQA"
            if a["G"] != (1 if chart else 4) or a["protocol"] != (
                "chartqa_plain" if chart else "structured"
            ):
                raise ValueError("Evaluation protocol or draw budget changed")
            if len(a["seeds"]) != a["G"] or any(type(s) is not int for s in a["seeds"]):
                raise ValueError("Incomplete planned seed identities")
            if a["panel"] == NEW_PANEL and (
                task["split"] != NEW_PANEL or task["family"] != "TOPK"
                or task["cell"] not in CELLS
            ):
                raise ValueError("New TOPK evaluation identity changed")
            if a["panel"] == OLD_PANEL and task["pool"] != "TEST_ID":
                raise ValueError("Old retention tasks must retain original TEST_ID pool")
            if chart and task["pool"] != "CHARTQA":
                raise ValueError("Non-ChartQA task in external panel")
        new = [tasks[a["qid"]] for a in assignments if a["panel"] == NEW_PANEL]
        if Counter(t["cell"] for t in new) != dict.fromkeys(CELLS, 64):
            raise ValueError("New test must contain all four fixed cells")
        world_counts = Counter(t["world_id"] for t in new)
        if len(world_counts) != 128 or set(world_counts.values()) != {2}:
            raise ValueError("New test must contain two queries per each of 128 worlds")
    def common(a):
        return a["qid"], a["G"], a["protocol"], a["panel"]
    base = [common(a) for a in plans[0] if a["panel"] != "CHARTQA"]
    if any([common(a) for a in plans[b]] != base for b in (1, 2)):
        raise ValueError("Native evaluation tasks must be paired across all blocks")
    total = len(ARMS) * sum(a["G"] for plan in plans.values() for a in plan)
    if total != 24290:
        raise ValueError("Fixed total of 24,290 evaluation answers changed")
    return total


def completion_inventory(root):
    """File receipts alone are necessary, never sufficient for completion."""
    expected = {}
    for b in BLOCKS:
        for arm in ARMS:
            if arm != "PARENT":
                expected[f"CPF-T-b{b}-{arm}"] = ("c_pilot_train", b, arm, None)
            for shard in range(4):
                expected[f"CPF-E-b{b}-{arm}-s{shard}"] = ("c_pilot_eval", b, arm, shard)
    registry = read(Path(root) / "TASKS.json")
    registry = registry.get("tasks", []) if isinstance(registry, dict) else registry
    selected = [r for r in registry if r.get("kind") in {"c_pilot_train", "c_pilot_eval"}]
    indexed = _unique(selected, "task_id")
    issues = []
    if set(indexed) != set(expected):
        issues.append(dict(reason="WRONG_REGISTERED_TASK_INVENTORY",
                           missing=sorted(set(expected) - set(indexed)),
                           unexpected=sorted(set(indexed) - set(expected))))
    for task_id, (kind, block, arm, shard) in expected.items():
        spec = indexed.get(task_id, {})
        if any(spec.get(k) != v for k, v in dict(kind=kind, block=block, arm=arm).items()) or (
            shard is not None and spec.get("shard") != shard
        ):
            issues.append(dict(task_id=task_id, reason="REGISTERED_TASK_IDENTITY_MISMATCH"))
        path = Path(root) / "tasks" / task_id / "COMPLETE.json"
        try:
            receipt = read(path)
            if receipt.get("status") != "COMPLETE" or receipt.get("task_id") != task_id:
                raise ValueError("Completion receipt identity/status differs")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            issues.append(dict(task_id=task_id, reason="MISSING_OR_INVALID_COMPLETE_RECEIPT",
                               detail=str(exc)))
    return dict(expected_training_tasks=12, expected_evaluation_tasks=60, issues=issues)


def _read_group(path):
    try:
        return read(path), None
    except (OSError, ValueError) as exc:
        return None, exc


def _ordered_groups(paths):
    """At most sixteen raw groups resident; overlap reads with eight workers."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        for offset in range(0, len(paths), 16):
            yield from executor.map(_read_group, paths[offset:offset + 16])


def scan_evaluations(root, tasks, plans, *, verifier=None):
    """Read each expected atomic group exactly once; retain no token arrays."""
    from .verification_v3 import verify_semantics

    verifier = verifier or verify_semantics
    root = Path(root)
    compact, issues, seen_rows, expected_paths = [], [], set(), set()
    groups_read = 0
    paths = [root / "EVAL" / f"b{b}" / arm / f"shard{a['group_index'] % 4}" / "groups" / f"{a['group_index']:05d}.json"
             for b, assignments in sorted(plans.items()) for arm in ARMS for a in assignments]
    reader = _ordered_groups(paths)
    for block, assignments in sorted(plans.items()):
        for arm in ARMS:
            for assignment in assignments:
                index, panel = assignment["group_index"], assignment["panel"]
                path = root / "EVAL" / f"b{block}" / arm / f"shard{index % 4}" / "groups" / f"{index:05d}.json"
                expected_paths.add(path)
                try:
                    saved, error = next(reader)
                    if error is not None:
                        raise error
                    task = tasks[assignment["qid"]]
                    group_id = f"CPF-E-b{block}-{arm}/s{0 if arm == 'PARENT' else 16:03d}/g{index:05d}"
                    if saved["group_id"] != group_id or saved["block"] != block:
                        raise ValueError("Atomic group identity differs")
                    records = saved["rows"]
                    if len(records) != assignment["G"]:
                        raise ValueError("Missing or extra evaluation draw")
                    groups_read += 1
                    for draw, record in enumerate(records):
                        identity = dict(qid=task["qid"], block=block, arm=arm, panel=panel,
                                        protocol=assignment["protocol"], draw=draw,
                                        seed=assignment["seeds"][draw], group_id=group_id,
                                        row_id=f"{group_id}/d{draw:02d}",
                                        step=0 if arm == "PARENT" else 16,
                                        split=task["split"])
                        if any(record.get(k) != v for k, v in identity.items()):
                            raise ValueError(f"Response identity differs at draw {draw}")
                        if record["row_id"] in seen_rows:
                            raise ValueError("Duplicate response identity")
                        seen_rows.add(record["row_id"])
                        if record.get("generation_status") != "COMPLETE" or record.get("technical_validation_errors"):
                            raise ValueError("Technical generation failure retained")
                        if not isinstance(record.get("full_text"), str):
                            raise ValueError("Missing raw model response")
                        chart = panel == "CHARTQA"
                        metrics = {k: _metric(record["score"], k)
                                   for k in CHART_METRICS if chart}
                        semantics = {}
                        if not chart:
                            world = task.get("world_id", task.get("root_id"))
                            if not world or record.get("world_id") != world:
                                raise ValueError("Response world identity differs")
                            metrics = {k: _metric(record["score"], k) for k in NATIVE_METRICS}
                            label = verifier(task, record, record["score"])
                            if label["verifier_version"] != VERSION:
                                raise ValueError("Unexpected semantic measurement version")
                            semantics = {d: label["dimensions"][d]["status"] for d in "PS"}
                            if any(v not in {"PASS", "FAIL", "UNKNOWN"} for v in semantics.values()):
                                raise ValueError("Invalid semantic label")
                        compact.append(dict(
                            block=block, arm=arm, panel=panel, qid=task["qid"], draw=draw,
                            world_id=task.get("world_id", task.get("root_id", task["qid"])),
                            family=task.get("family", "EXTERNAL"), variant=task.get("variant"),
                            cell=task.get("cell"), score=metrics, semantic=semantics,
                            truncated=record.get("truncated", record.get("finish_reason") == "length"),
                            completion_tokens=record.get("completion_token_count", len(record.get("tokens", []))),
                        ))
                except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                    issues.append(dict(path=str(path.relative_to(root)), reason="INVALID_OR_MISSING_GROUP",
                                       detail=str(exc)))
    found = {p for p in (root / "EVAL").glob("b*/*/shard*/groups/*.json") if ".chunk" not in p.name}
    issues.extend(dict(path=str(p.relative_to(root)), reason="UNEXPECTED_GROUP_FILE")
                  for p in sorted(found - expected_paths))
    reader.close()
    return compact, dict(issues=issues, expected_groups=len(expected_paths),
                         validated_group_files=groups_read, extracted_answers=len(compact),
                         expected_answers=sum(a["G"] for p in plans.values() for a in p) * len(ARMS))


def _world_values(items, metric):
    """Equal draws within query, equal queries within world, then equal worlds."""
    queries = defaultdict(list)
    for r in items:
        queries[(r["block"], r["arm"], r["world_id"], r["qid"])].append(r["score"][metric])
    worlds = defaultdict(list)
    for (block, arm, world, _), values in queries.items():
        worlds[block, arm, world].append(sum(values) / len(values))
    return {key: sum(value) / len(value) for key, value in worlds.items()}


def summarize_stratum(items, *, name, panel, bootstrap_repeats=BOOTSTRAPS):
    """Build shared-world arrays once; bootstrap is bounded NumPy indexing."""
    blocks = sorted({r["block"] for r in items})
    worlds = sorted({r["world_id"] for r in items})
    metrics = CHART_METRICS if panel == "CHARTQA" else ("value_correct", "J")
    if not items or set(blocks) != ({0} if panel == "CHARTQA" else set(BLOCKS)):
        raise ValueError("Incomplete block coverage in analysis stratum")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(worlds), size=(bootstrap_repeats, len(worlds))
    )
    endpoints, comparisons = [], []
    grouped = defaultdict(list)
    for row in items:
        grouped[row["block"], row["arm"]].append(row)
    for metric in metrics:
        values = _world_values(items, metric)
        expected = {(b, a, w) for b in blocks for a in ARMS for w in worlds}
        if set(values) != expected:
            raise ValueError("Unpaired world/arm coverage cannot produce scientific comparisons")
        cube = np.array([[[values[b, a, w] for w in worlds] for a in ARMS] for b in blocks])
        for bi, block in enumerate([*blocks, "equal_blocks"]):
            matrix = cube[bi] if bi < len(blocks) else cube.mean(axis=0)
            for ai, arm in enumerate(ARMS):
                endpoints.append(dict(stratum=name, panel=panel, block=block, arm=arm,
                                      metric=metric, mean=float(matrix[ai].mean()), worlds=len(worlds),
                                      answers=sum(len(grouped[b, arm]) for b in blocks)
                                      if block == "equal_blocks" else len(grouped[block, arm])))
            pairs = [("SELECT", a) for a in ("SELECT_SHUF", "J", "EVID")]
            pairs += [(a, "PARENT") for a in ARMS if a != "PARENT"]
            for left, right in pairs:
                delta = matrix[ARMS.index(left)] - matrix[ARMS.index(right)]
                samples = delta[draws].mean(axis=1)
                lo, hi = np.quantile(samples, (0.025, 0.975))
                comparisons.append(dict(
                    stratum=name, panel=panel, block=block, arm=left, reference=right,
                    metric=metric, difference=float(delta.mean()), ci95=[float(lo), float(hi)],
                    worlds=len(worlds), blocks=blocks, bootstrap_repeats=bootstrap_repeats,
                    bootstrap_seed=BOOTSTRAP_SEED, unit="PAIRED_WORLD_SHARED_ACROSS_BLOCKS_AND_ARMS",
                    conditional_on_fixed_trained_models=True,
                    primary=(name == "new/all" and metric == "value_correct" and left == "SELECT"
                             and right == "SELECT_SHUF" and block == "equal_blocks"),
                ))
    measurement = []
    for (block, arm), values in sorted(grouped.items()):
        entry = dict(stratum=name, panel=panel, block=block, arm=arm, answers=len(values),
                     truncated=sum(bool(r["truncated"]) for r in values),
                     mean_completion_tokens=sum(r["completion_tokens"] for r in values) / len(values))
        for d in "PS":
            counts = Counter(r["semantic"].get(d) for r in values)
            if counts.get(None) == len(values):
                continue
            known = counts["PASS"] + counts["FAIL"]
            entry[d] = dict(counts={s: counts[s] for s in ("PASS", "FAIL", "UNKNOWN")},
                            all_denominator=len(values), known_denominator=known,
                            known_coverage=known / len(values), pass_all=counts["PASS"] / len(values),
                            pass_given_known=counts["PASS"] / known if known else None)
        if panel != "CHARTQA":
            entry["native_metrics"] = {k: sum(r["score"][k] for r in values) / len(values)
                                       for k in NATIVE_METRICS}
        measurement.append(entry)
    return dict(endpoints=endpoints, comparisons=comparisons, measurement=measurement)


def analyze(items):
    strata = defaultdict(list)
    for row in items:
        if row["panel"] == NEW_PANEL:
            keys = ("new/all", "new/cell/" + row["cell"])
        elif row["panel"] == OLD_PANEL:
            keys = ("old/all", "old/family/" + row["family"],
                    f"old/family_variant/{row['family']}/{row['variant']}")
        else:
            keys = ("chartqa/b0",)
        for key in keys:
            strata[key].append(row)
    result = dict(endpoints=[], comparisons=[], measurement=[])
    for name, values in sorted(strata.items()):
        summary = summarize_stratum(values, name=name, panel=values[0]["panel"])
        for key in result:
            result[key].extend(summary[key])
    return result


def training_mechanism(root, tasks):
    """Validate and aggregate compact credit receipts, not checkpoints/logprobs."""
    from srf4_ref.credit import Observation
    from srf4_ref.sampling import seed_for

    from . import c_pilot_data as data
    from .credit_design_replay import POLICIES, eligibility, proposed_residual
    from .training import native_advantages

    policy = next(p for p in POLICIES if p.name == "F_PRIME_KNOWN_75")
    issues, strata, seen = [], defaultdict(Counter), set()
    for block in BLOCKS:
        plan = read(Path(root) / "DATA" / f"TRAIN_PLAN_b{block}.json")
        if set(plan) != {str(s) for s in range(1, 17)}:
            raise ValueError("Missing or extra planned training update")
        for arm in ARMS[1:]:
            for step in range(1, 17):
                path = Path(root) / "RUNS/C" / f"b{block}" / arm / "credit" / f"{step:03d}.json"
                try:
                    assigned = plan[str(step)]
                    if assigned != data.training_assignments(tasks, block, step):
                        raise ValueError("Registered training schedule differs")
                    consumed = read(path.parent.parent / "assignments" / path.name)
                    if consumed != dict(step=step, groups=assigned):
                        raise ValueError("Consumed training schedule differs")
                    assignment_by_index = _unique(assigned, "group_index")
                    receipt = read(path)
                    expected = dict(schema="srf4-c-fprime-credit-v1", block=block, arm=arm, step=step,
                                    policy="F_PRIME_KNOWN_75", beta=0.25, verifier_version=VERSION)
                    if any(receipt.get(k) != v for k, v in expected.items()):
                        raise ValueError("Credit receipt identity/policy differs")
                    groups = receipt["groups"]
                    if len(groups) != 16 or {g["group_index"] for g in groups} != set(range(16)):
                        raise ValueError("Expected exactly 16 training groups per step")
                    for g in groups:
                        assignment = assignment_by_index[g["group_index"]]
                        task = tasks[assignment["qid"]]
                        group_id = f"CPF-T-b{block}-{arm}/s{step:03d}/g{g['group_index']:05d}"
                        if any(g.get(k) != v for k, v in dict(group_id=group_id, qid=task["qid"],
                                                            world_id=task["world_id"], family="TOPK",
                                                            cell=task["cell"]).items()):
                            raise ValueError("Training credit group/task identity differs")
                        observations = [Observation(**r) for r in g["observations"]]
                        if len(observations) != 8 or g["cell"] not in CELLS:
                            raise ValueError("Training G8/cell identity differs")
                        for draw, r in enumerate(observations):
                            if r.row_id != f"{group_id}/d{draw:02d}" or r.row_id in seen:
                                raise ValueError("Duplicate training response identity")
                            if any(type(getattr(r, k)) is not bool for k in ("protocol_valid", "finished", "defined")) or (r.sufficient is not None and type(r.sufficient) is not bool):
                                raise ValueError("Invalid eligibility boolean in credit observation")
                            seen.add(r.row_id)
                        candidates, known, reason = eligibility(observations, policy)
                        if g["candidate_indices"] != candidates or g["known_indices"] != known or g["eligibility_reason"] != reason or g["active"] != (reason == "ACTIVE"):
                            raise ValueError("Credit eligibility differs from shared F-prime policy")
                        for field in ("native_advantages", "residual", "final_advantages"):
                            if len(g[field]) != 8 or any(type(v) not in (int, float) or not math.isfinite(v) for v in g[field]):
                                raise ValueError("Malformed/nonfinite credit vector")
                        if any(abs(a + d - final) > 1e-10 for a, d, final in zip(g["native_advantages"], g["residual"], g["final_advantages"], strict=True)):
                            raise ValueError("Final advantages do not equal native plus residual")
                        seed = seed_for(data.MASTER, "c-pilot-selection-shuffle", block, step,
                                        g["group_index"], task["qid"])
                        expected_native = native_advantages([r.j for r in observations])
                        expected_residual = proposed_residual(observations, arm, policy,
                                                              shuffle_seed=seed)
                        if any(abs(a-b) > 1e-10 for field, expected in (("native_advantages", expected_native), ("residual", expected_residual)) for a, b in zip(g[field], expected, strict=True)):
                            raise ValueError("Stored credit does not match registered formula")
                        if g["shuffle"]["seed"] != seed or g["shuffle"]["applied"] != (reason == "ACTIVE" and arm == "SELECT_SHUF"):
                            raise ValueError("Shuffle provenance differs")
                        ppass = [i for i in known if observations[i].perception == "PASS"]
                        n, k = len(ppass), sum(observations[i].selection == "PASS" for i in ppass)
                        signal = reason == "ACTIVE" and 0 < k < n
                        positive = [i for i, value in enumerate(g["residual"]) if value > 1e-12]
                        counts = Counter(groups=1, answers=8, candidate_rows=len(candidates),
                                         known_candidate_rows=len(known), active_groups=reason == "ACTIVE",
                                         selection_content_signal_groups=signal,
                                         actual_nonzero_residual_groups=any(abs(v) > 1e-12 for v in g["residual"]),
                                         positive_failure_rows=sum(observations[i].j == 0 for i in positive),
                                         positive_final_failure_rows=sum(r.j == 0 and a > 1e-12 for r, a in zip(observations, g["final_advantages"], strict=True)),
                                         positive_selection_fail=sum(observations[i].selection == "FAIL" for i in positive),
                                         positive_evidence_incomplete=sum(observations[i].sufficient is False for i in positive),
                                         native_all_zero_groups=all(r.j == 0 for r in observations),
                                         zero_native_J_selection_signal_groups=all(r.j == 0 for r in observations) and signal,
                                         select_minus_evid_l1=g["select_minus_evid_l1"],
                                         select_minus_shuffle_l1=g["select_minus_shuffle_l1"],
                                         actual_shuffle_changed_rows=g["shuffle"]["changed_count"],
                                         actual_shuffle_changed_groups=g["shuffle"]["changed_count"] > 0,
                                         residual_l1=sum(abs(v) for v in g["residual"]),
                                         native_l1=sum(abs(v) for v in g["native_advantages"]),
                                         select_minus_shuffle_expected_l1=0.25 * k * (n-k) / n if signal else 0.0)
                        counts["reason/" + reason] += 1
                        for r in observations:
                            for d, value in (("P", r.perception), ("S", r.selection)):
                                counts[d + "/" + value] += 1
                        for key in (f"b{block}/{arm}/all", f"b{block}/{arm}/cell/{g['cell']}"):
                            strata[key].update(counts)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    issues.append(dict(path=str(path.relative_to(root)), reason="INVALID_OR_MISSING_CREDIT_RECEIPT", detail=str(exc)))
    if len(seen) != 24576:
        issues.append(dict(reason="TRAINING_ANSWER_COUNT_DIFFERS", observed=len(seen), expected=24576))
    return dict(issues=issues, answers=len(seen), strata={k: dict(v) for k, v in sorted(strata.items())},
                positive_descriptors_overlap=True,
                signal_definition="Active F-prime known subset has both S outcomes among P=PASS rows; distinct from total residual nonzero")


def _percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def write_report(root, result):
    lines = ["# C F′ 小规模训练试验", "", f"状态：`{result['status']}`。", ""]
    if result["status"] != COMPLETE:
        lines += ["结构、完成回执或技术完整性检查尚未通过；不发布部分结果的科学结论。",
                  f"发现 {len(result['issues'])} 项问题；完整明细见 `ANALYSIS/SUMMARY.json`。", ""]
    else:
        primary = next(c for c in result["comparisons"] if c["primary"])
        lo, hi = primary["ci95"]
        lines += [f"新TOPK主比较 SELECT−SELECT_SHUF 的 value_correct 为 {_percent(primary['difference'])} 个百分点，"
                  f"配对world条件95%区间 [{_percent(lo)}, {_percent(hi)}]。三个父模型等权；正负结果均保留。",
                  "训练24,576条、评估24,290条；12条训练路径、15个端点。没有以区间或点估计自动宣布科学通过。", "",
                  "## 主要发现", "", "下表先列新TOPK的主要比较。正值表示SELECT点估计更高，负值表示更低；区间跨零表示方向仍不确定，不表示两臂等效。", "",
                  "| 块 | 比较 | 指标 | 差值(百分点) | 条件95%区间 |",
                  "|---|---|---|---:|---|"]
        for r in result["comparisons"]:
            if r["stratum"] == "new/all" and r["arm"] == "SELECT" and r["reference"] != "PARENT":
                lines.append(f"| {r['block']} | SELECT−{r['reference']} | {r['metric']} | {_percent(r['difference'])} | [{_percent(r['ci95'][0])}, {_percent(r['ci95'][1])}] |")
        lines += ["", "旧保留性能和ChartQA的不利变化也保留；以下列出各块的总体比较，家族和变体细分见完整表。", "",
                  "| 面板 | 块 | 比较 | 指标 | 差值(百分点) | 条件95%区间 |",
                  "|---|---|---|---|---:|---|"]
        for r in result["comparisons"]:
            if r["stratum"] in {"old/all", "chartqa/b0"} and r["arm"] == "SELECT":
                lines.append(f"| {r['stratum']} | {r['block']} | SELECT−{r['reference']} | {r['metric']} | {_percent(r['difference'])} | [{_percent(r['ci95'][0])}, {_percent(r['ci95'][1])}] |")
        lines += ["",
                  "## 端点均值", "", "单位为百分比；equal_blocks是三个固定父模型的等权均值，ChartQA只含b0。", "",
                  "| 分层 | 块 | 臂 | 指标 | 均值(%) | 回答数 | world数 |",
                  "|---|---|---|---|---:|---:|---:|"]
        for r in result["endpoints"]:
            lines.append(f"| {r['stratum']} | {r['block']} | {r['arm']} | {r['metric']} | {_percent(r['mean'])} | {r['answers']} | {r['worlds']} |")
        lines += ["", "## 所有配对比较", "", "单位为百分点；区间共享world重采样，不按回答独立抽样。", "",
                  "| 分层 | 块 | 比较 | 指标 | 差值 | 条件95%区间 |",
                  "|---|---|---|---|---:|---|"]
        for r in result["comparisons"]:
            lines.append(f"| {r['stratum']} | {r['block']} | {r['arm']}−{r['reference']} | {r['metric']} | {_percent(r['difference'])} | [{_percent(r['ci95'][0])}, {_percent(r['ci95'][1])}] |")
        lines += ["", "## S测量覆盖", "", "全体分母包括UNKNOWN；已知内PASS率只以PASS+FAIL为分母。", "",
                  "| 分层 | 块 | 臂 | 全体 | 已知 | PASS | FAIL | UNKNOWN | 已知覆盖(%) | 已知内PASS(%) | 全体PASS(%) |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in result["measurement"]:
            if "S" in r:
                s = r["S"]
                lines.append(f"| {r['stratum']} | {r['block']} | {r['arm']} | {s['all_denominator']} | {s['known_denominator']} | {s['counts']['PASS']} | {s['counts']['FAIL']} | {s['counts']['UNKNOWN']} | {_percent(s['known_coverage'])} | {_percent(s['pass_given_known'])} | {_percent(s['pass_all'])} |")
        lines += ["", "## 训练信用机制", "", "选择内容信号与总残差非零分开；正信用成分可以重叠。", "",
                  "| 分层 | 组 | 候选失败 | 已知候选 | 活跃组 | 选择内容信号组 | 实际残差非零组 | 正信用失败 | 正信用S失败 | 正信用证据不完整 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, r in result["training_mechanism"]["strata"].items():
            fields = ("groups", "candidate_rows", "known_candidate_rows", "active_groups", "selection_content_signal_groups", "actual_nonzero_residual_groups", "positive_failure_rows", "positive_selection_fail", "positive_evidence_incomplete")
            lines.append("| " + key + " | " + " | ".join(str(r.get(k, 0)) for k in fields) + " |")
        lines += ["", "下表为各臂自身rollout上的信用向量差异和实际置换，不能解释为跨臂因果效果。", "",
                  "| 分层 | 全J=0且有选择信号组 | SELECT−EVID L1 | SELECT−SHUF L1 | 实际置换改变S行数 | 实际置换改变组数 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key, r in result["training_mechanism"]["strata"].items():
            fields = ("zero_native_J_selection_signal_groups", "select_minus_evid_l1", "select_minus_shuffle_l1", "actual_shuffle_changed_rows", "actual_shuffle_changed_groups")
            lines.append("| " + key + " | " + " | ".join(str(r.get(k, 0)) for k in fields) + " |")
    lines += ["", "## 解释边界", "", *["- " + s for s in LIMITATIONS], "",
              "模型权重、优化器、逐token记录及完整原始回答保留在服务器原目录。交付仅包含本报告、摘要和轻量引用清单；没有生成大归档。", ""]
    (Path(root) / "REPORT_ZH.md").write_text("\n".join(lines))


def delivery_manifest(root, status):
    root = Path(root)
    names = ["REPORT_ZH.md", "ANALYSIS/SUMMARY.json", "ANALYSIS/COVERAGE.json",
             "CONFIG.json", "FROZEN.json", "TASKS.json", "DATA/TASKS.jsonl"]
    names += [f"DATA/EVAL_GROUPS_b{b}.jsonl" for b in BLOCKS]
    files = []
    for name in names:
        path = root / name
        if path.is_file():
            files.append(dict(path=name, bytes=path.stat().st_size))
    return dict(status=status, root=str(root), files=files, archive_created=False,
                server_only=[dict(path="RUNS/C", contents="adapters/checkpoints/optimizer/training raw groups and credit receipts", weights_hashed=False),
                             dict(path="EVAL", contents="raw endpoint responses and generation metadata", full_raw_archive_created=False)],
                endpoint_references=[dict(block=b, arm=a, step=0 if a == "PARENT" else 16,
                                          reference="CONFIG.json#parents" if a == "PARENT" else f"RUNS/C/b{b}/{a}")
                                     for b in BLOCKS for a in ARMS])


def report(root):
    from .c_pilot import build_eval_groups, require_frozen

    root = Path(root).resolve()
    result = dict(status="INCOMPLETE_TECHNICAL", issues=[], verifier_version=VERSION,
                  primary="new TOPK value_correct SELECT minus SELECT_SHUF; equal parent blocks",
                  expected_evaluation_answers=24290, expected_training_answers=24576,
                  inference_limits=LIMITATIONS, old_c_gate_modified=False, human_audit_count=0)
    try:
        from .verification_v3 import VERIFIER_VERSION
        if VERIFIER_VERSION != VERSION:
            raise ValueError("Reporting must use frozen r6 measurement")
        require_frozen(root)
        tasks = _unique(rows(root / "DATA/TASKS.jsonl"), "qid")
        plans = {b: list(rows(root / f"DATA/EVAL_GROUPS_b{b}.jsonl")) for b in BLOCKS}
        validate_plans(tasks, plans)
        if any(plans[b] != build_eval_groups(list(tasks.values()), b) for b in BLOCKS):
            raise ValueError("Evaluation plan differs from registered deterministic schedule")
        completion = completion_inventory(root)
        result["completion_inventory"] = completion
        result["issues"].extend(completion["issues"])
        values, coverage = scan_evaluations(root, tasks, plans)
        result["coverage"] = coverage
        result["issues"].extend(coverage["issues"])
        mechanism = training_mechanism(root, tasks)
        result["training_mechanism"] = mechanism
        result["issues"].extend(mechanism["issues"])
        if coverage["extracted_answers"] != 24290:
            result["issues"].append(dict(reason="EVALUATION_ANSWER_COUNT_DIFFERS"))
        if not result["issues"]:
            result.update(analyze(values), status=COMPLETE)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        result["issues"].append(dict(reason="REPORT_INPUT_VALIDATION_FAILED", detail=str(exc)))
    write(root / "ANALYSIS/SUMMARY.json", result)
    write(root / "ANALYSIS/COVERAGE.json", dict(status=result["status"],
                                                **result.get("coverage", {}),
                                                all_issues=result["issues"]))
    write_report(root, result)
    write(root / "ANALYSIS/DELIVERY_MANIFEST.json", delivery_manifest(root, result["status"]))
    return dict(status=result["status"], issues=len(result["issues"]),
                expected_evaluation_answers=24290, expected_training_answers=24576,
                summary=str(root / "ANALYSIS/SUMMARY.json"),
                report=str(root / "REPORT_ZH.md"),
                manifest=str(root / "ANALYSIS/DELIVERY_MANIFEST.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    result = report(args.root)
    if result["status"] != COMPLETE:
        raise SystemExit("INCOMPLETE_TECHNICAL: see ANALYSIS/SUMMARY.json")


if __name__ == "__main__":
    main()
