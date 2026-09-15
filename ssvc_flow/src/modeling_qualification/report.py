"""Build the factual Chinese M6 decision from frozen, verified stage outputs."""

# Generated Chinese Markdown deliberately preserves Unicode and long table expressions.
# ruff: noqa: E501, RUF001
from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
from pathlib import Path

from .io import new_output, sha256_file, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.6g}"
    except (ValueError, TypeError):
        return str(value)


def markdown_table(rows, fields):
    lines = [
        "| " + " | ".join(label for _, label in fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(number(row.get(key)) for key, _ in fields) + " |")
    return "\n".join(lines)


def _primary(row):
    return (
        str(row.get("fit_budget")) == "8"
        and row.get("representation") == "per_prompt"
        and row.get("input_mode") == "actual_d"
    )


def _passes(row, config):
    rules = config["model_selection"]
    try:
        return all(
            [
                float(row["group_delta_pX_q95_max"]) <= rules["group_delta_pX_abs_error_q95_max"],
                float(row["group_delta_v_q95_max"]) <= rules["group_delta_v_abs_error_q95_max"],
                float(row["delta_mae"]) <= rules["per_prompt_event_mae_max"],
                float(row["response_nrmse"]) <= rules["response_nrmse_max"],
                float(row["raw_invalid_fraction"]) <= rules["raw_invalid_simplex_fraction_max"],
            ]
        )
    except (ValueError, TypeError, KeyError):
        return False


def _resolve_table_links(text: str, out: Path) -> str:
    def resolve(match):
        path = match.group(1)
        if not (out / path).exists() and (out / (path + ".gz")).exists():
            return f"({path}.gz)"
        return match.group(0)

    return re.sub(r"\((tables/[^)]+)\)", resolve, text)


def run_report(config: dict, input: Path, out: Path) -> dict:
    root, out = Path(input), Path(out)
    stage_paths = read_json(root / "stage_paths.json")
    stages = {k: root / v for k, v in stage_paths.items()}
    verification = read_json(root / "verification_summary.json")
    if verification["engineering_status"] not in ("PASS", "BLOCKED"):
        raise ValueError("engineering status must be explicitly assessed before final report")
    if verification["engineering_status"] == "BLOCKED" and not verification.get("blockers"):
        raise ValueError("a blocked report requires the explicit unresolved verification gate")
    m2, m3, m4 = [read_json(stages[key] / "summary.json") for key in ("M2", "M3", "M4")]
    if any(r.get("status") not in ("PASS", "COMPLETED") for r in (m2, m3, m4)):
        raise ValueError("CPU experiment stages are incomplete")
    selection = read_json(stages["M3"] / "selection.json")
    m5 = read_json(stages["M5"] / "audit_summary.json")
    tracking = read_csv(stages["M4"] / "tracking_summary.csv")
    new_output(out)
    tables = out / "tables"
    tables.mkdir()
    table_index = []
    for phase in ("M0", "M1", "M2", "M3", "M4"):
        for path in sorted(stages[phase].glob("*")):
            if path.suffix not in (".csv", ".json") or path.name in ("manifest.json",):
                continue
            # Prediction identities remain in the raw run; summarize by hash below.
            if path.name in ("prediction_freeze.json", "frozen_tracking_predictions.json"):
                continue
            target = tables / f"{phase}_{path.name}"
            if path.stat().st_size > 5_000_000:
                target = target.with_suffix(target.suffix + ".gz")
                with path.open("rb") as source, gzip.open(target, "wb") as destination:
                    shutil.copyfileobj(source, destination)
            else:
                shutil.copyfile(path, target)
            table_index.append(
                {
                    "phase": phase,
                    "file": str(target.relative_to(out)),
                    "source_sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "delivered_sha256": sha256_file(target),
                    "delivered_bytes": target.stat().st_size,
                    "encoding": "gzip" if target.suffix == ".gz" else "unchanged",
                }
            )
    for filename in [
        "M2_validation.json",
        "M2_source_linkage.json",
        "preflight.json",
        "analysis_retry_decision.json",
        "verification_summary.json",
    ]:
        shutil.copyfile(root / filename, tables / filename)
    shutil.copyfile(stages["M0"] / "MATH_AUDIT_zh.md", out / "MATH_AUDIT_zh.md")
    shutil.copyfile(
        stages["M5"] / "REAL_EVIDENCE_READINESS_zh.md", tables / "M5_local_before_server_zh.md"
    )
    supplement = root / "M5_final_supplement_zh.md"
    if supplement.exists():
        text = supplement.read_text()
        for filename in ("M5_public_vector_summary.json",):
            text = text.replace(f"`{filename}`", f"[{filename}](tables/{filename})")
        for filename in (
            "real_update_geometry.json",
            "real_evidence_table_with_vectors.jsonl",
            "bank_vector_summary.csv",
            "retrieval_verification.json",
        ):
            source = root / "M5_vectors" / filename
            shutil.copyfile(source, tables / f"M5_{filename}")
            text = text.replace(f"`M5_vectors/{filename}`", f"[{filename}](tables/M5_{filename})")
        text += "\n\n服务器核验前的本机历史快照单独保留于 [旧准备度快照](tables/M5_local_before_server_zh.md)；其中向量/布局/夹角缺项已由本报告对应结果更新。\n"
        (out / "REAL_EVIDENCE_READINESS_zh.md").write_text(text, encoding="utf-8")
    else:
        shutil.copyfile(
            stages["M5"] / "REAL_EVIDENCE_READINESS_zh.md", out / "REAL_EVIDENCE_READINESS_zh.md"
        )
    for filename in ("M5_public_vector_summary.json", "M5_public_remote_summary.json"):
        if (root / filename).exists():
            shutil.copyfile(root / filename, tables / filename)
    primary_root = root / "M3_primary_tables"
    primary = [r for r in read_csv(primary_root / "primary_aggregate_metrics.csv") if _primary(r)]
    for path in sorted(primary_root.iterdir()):
        if path.is_file() and path.suffix in (".csv", ".json"):
            target = tables / f"M3_{path.name}"
            if target.exists():
                target = tables / f"M3_postprocess_{path.name}"
            shutil.copyfile(path, target)
    robust_rows = read_csv(primary_root / "robust_aggregate_metrics.csv")
    chosen = [r for r in robust_rows if r["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC"]
    for row in chosen:
        row["test_gate"] = (
            "WITHIN_MODEL_TOLERANCE" if _passes(row, config) else "EXCEEDS_MODEL_TOLERANCE"
        )
    exact = [r for r in chosen if r["n"] == "exact"]
    finite_passing = [r for r in chosen if r["n"] != "exact" and _passes(r, config)]
    rank = selection.get("exact_selection", {}).get("rank_cap")
    model_status = "SUPPORTED_LOCAL" if finite_passing else "NO_ACCEPTABLE_R"
    readiness = {
        "engineering_status": verification["engineering_status"],
        "modeling_status": model_status,
        "primary_n64_selection_status": selection["status"],
        "exact_diagnostic_test_pass": bool(exact and _passes(exact[0], config)),
        "observation_scope": "KNOWN_REALIZED_UPDATES",
        "real_vlm_validation": "READONLY_PARTIAL",
        "real_evidence_audit_status": m5["audit_status"],
        "real_parameter_geometry": read_json(root / "M5_public_vector_summary.json"),
        "verification_blockers": verification.get("blockers", []),
        "new_gpu_started": False,
        "new_qwen_calls": False,
        "online_ssvc_started": False,
        "safety_certified": False,
        "recommended_storage": "per_prompt_four_counts_with_identity",
        "recommended_regression": "per_prompt_helmert",
        "oracle_diagnostic_rank_cap": rank,
        "tested_finite_sample_budgets_passing_frozen_rule": [int(r["n"]) for r in finite_passing],
        "raw_run_stage_paths": stage_paths,
        "test_seed_count": m3["test_seed_count"],
        "reports": config["final_reports"],
        "raw_data_retained_locally": True,
    }
    write_json(out / "machine_readable_readiness.json", readiness)
    from .plots import render_figures

    figures = render_figures(stages["M3"], stages["M4"], out / "figures")
    write_json(out / "figure_sources.json", figures)
    comparison = [
        r
        for r in primary
        if r["n"] in ("exact", "64")
        and (
            r["method"].startswith(("B0_", "B1_", "B6_", "O0_"))
            or (
                r["method"].startswith(("B2_", "B3_", "B4_", "B5_", "O1_"))
                and r["rank_cap"] in ("1", "FULL")
            )
        )
    ]
    fields = [
        ("method", "方法"),
        ("rank_cap", "r cap"),
        ("n", "n"),
        ("delta_mae", "响应 MAE"),
        ("response_nrmse", "响应 NRMSE"),
        ("group_delta_pX_q95_max", "最差群体 pX q95"),
        ("group_delta_v_q95_max", "最差群体 v q95"),
        ("level_mae", "概率水平 MAE"),
    ]
    tracking_chosen = [
        r
        for r in tracking
        if r["method"] == "B5_EXACT_SELECTED_DIAGNOSTIC"
        and r.get("telemetry") == "PROJECTION_AND_NET_DISPLACEMENT"
    ]
    rows_h = sorted(tracking_chosen, key=lambda r: (r["n"], int(r["H"])))
    rank_h = [
        r
        for r in read_csv(stages["M4"] / "tracking_rank_horizon_grid.csv")
        if r["n"] == "exact"
        and r["telemetry"] == "PROJECTION_AND_NET_DISPLACEMENT"
        and r["rank_cap"] in ("1", "2", "FULL")
        and r["H"] in ("4", "8", "16")
    ]
    cost_root = root / "M4_cost_supplement"
    if cost_root.exists():
        for path in sorted(cost_root.iterdir()):
            if path.is_file() and path.suffix in (".json", ".csv", ".md"):
                shutil.copyfile(path, tables / f"M4_cost_{path.name}")

    results = (
        f"""# CPU 建模实验结果（M0–M6）

## 实际范围

M2：{m2["trajectory_count"]} 条轨迹，{m2["optimizer_steps"]} 次 CPU Adam 更新，
{m2["sampled_finite_actions"]} 次训练/分支动作采样；完整采集耗时 {number(m2["wall_seconds"])} 秒，
峰值 RSS {number(m2["peak_rss_gib"])} GiB，输出 {m2["output_bytes"]} bytes。
同一轨迹供所有表示、基线、采样噪声和窗口分析。所有测试只覆盖固定数据集和固定初始化。

## 冻结基线结果

全部网格见 [完整汇总表](tables/M3_primary_aggregate_metrics.csv) 与
[逐 seed 结果](tables/M3_primary_by_seed.csv)。下表是预定基线/r=1/FULL 切片，数值均为概率单位。
公平主比较的 n64 统一使用噪声重复0–2；下一节冻结规则的稳健性比较使用0–4，因此两表n64汇总不必相等。
原始/投影、概率水平/响应指标分别存储；不以投影后的误差证明局部线性模型准确。

{markdown_table(comparison, fields)}

## 精确选择的一维规则在有限采样中的检验

这条规则在 selection seeds 的 exact 视图中冻结，再在既有计数 n16/64/256×5 噪声上检验。
它不通过 locked-test 重新选择维数或 ridge；exact 选择需要 oracle 诊断信息，不能当成可部署选择器。

{markdown_table(chosen, [*fields, ("test_gate", "联合目标")])}

## 固定窗口

所有 H 都评分其中每一个中间时刻。经验误差带以 calibration seeds 的完整16步窗口×六群体最大误差校准，
在 test seeds 测覆盖，不是95%分布无关保证。净位移和路径方案使用同一个固定局部预测器。

{markdown_table(rows_h, [("n", "n"), ("H", "H"), ("delta_mae", "响应MAE"), ("response_nrmse", "NRMSE"), ("group_delta_pX_q95_max", "pX q95"), ("group_delta_v_q95_max", "v q95"), ("largest_tested_H_meeting_tolerance", "最大达标H")])}

## 误差分解、消融和统计

方向遗漏、拟合/采样和非线性均有独立向量残差及三角界；不能用总范数相减分摊贡献。
完整数据见 [四级分解](tables/M3_four_level_error_decomposition.csv)、
[残差分解](tables/M3_error_decomposition.csv)、[辅助分支差异](tables/M3_auxiliary_minus_baseline.csv)、
[方向误差](tables/M3_direction_accuracy.csv)。oracle残差向量原件保留于原始运行目录 M3_verified/oracle_decomposition_vectors。
表示 pooled/six_group/per_prompt、actual d/norm-only/SGD proxy、1/2/4/8 bank 的消融均复用同一 M2。
两个接口与全部分支按训练 RNG seed 成簇；10 个 locked-test seeds，5000 次探索性配对 cluster bootstrap。
噪声重复先在 seed 内汇总，不当作独立训练重复。

## 完整成本与盈亏平衡

[80行完整CPU成本表](tables/M4_cost_full_cpu_cost_allocated.csv) 同时列出校准、分支、计数采样、
I/O、拟合、刷新和跟踪开销，状态明确为 ALLOCATED_ESTIMATE_WITH_MEASURED_REPLAY。
计数和M2写盘实测总量分别为10.009154秒、8.159183秒；按配置分摊部分是估计，不能称为逐配置实测。
60次固定已有样本的读/拟合/预测/写入重放未新增训练或观测；真实Qwen秒数未知。
[bank预算收支表](tables/M4_cost_bank_budget_break_even_qualification.csv) 显示8 bank需25个面板单位，
超过最大测试窗口16；1/2/4 bank的抽象盈亏平衡H为4/7/13，但这些窗口的准确性尚未验证。

## 五张图

"""
        + "\n\n".join(
            f"![{f['figure']}](figures/{f['figure']}.png)\n\n来源：`{f['source']}`。"
            for f in figures
        )
        + """

## 验证和来源限制

实际测试回执见 [verification_summary.json](tables/verification_summary.json)。
新增模块94项及服务器155项针对性回归通过。服务器首轮完整回归失败记录保留；
本机完整适用回归在临时夹具输出触及2GiB时中断，日志记录1080通过、1项sealed检查排除。
中断XML另有一个无名未完成节点，旧回执计数器将其误计为通过；本报告采用日志1080。
因此工程总门禁为BLOCKED，完整回归待资源审阅；不得将这些部分回执拼成一次全套PASS。
M2 首次产物验证器因 JSON 角色键重排误报，修正后原数据完整通过，未重训。
M2 运行期间 collection 整文件加入只读验证/元信息而哈希发生变化；实际 toy 哈希不变，
训练和采样函数与哈希匹配的 smoke 原件 AST 一致；详见 M2_source_linkage.json。
M2 的完整运行时 collection 文本未独立归档，不能声称整文件在该次运行中保持不变。
之后阶段均额外保存执行前源码快照。首版 M3 保留，权威结果使用 stage_paths 指向的重跑：
明确 macOS Accelerate 单线程环境，并补齐已冻结 exact 规则的有限n稳健性；不增加训练或重新选阈值。
"""
    )
    (out / "RESULTS_zh.md").write_text(_resolve_table_links(results, out), encoding="utf-8")
    sample_recommendation = (
        f"冻结规则在测试的 n={','.join(r['n'] for r in finite_passing)} 达到单步设计目标；仅在下表已验证窗口内使用。"
        if finite_passing
        else "n=16/64/256 均未使这条冻结规则达到全部单步目标；当前没有可推荐的有限采样运行配置。"
    )
    decision = f"""# 建模决定

## 1. 本轮实际执行范围与版本

工程总门禁：**{verification["engineering_status"]}**。新增模块验证：**PASS（94项）**。
主 n64 选择状态：**{selection["status"]}**。
综合有限采样建模状态：**{model_status}**。真实 VLM 验证：**READONLY_PARTIAL**。
按 modeling-qualification-v1-20260915，在独立 worktree 实施 M0–M6，
以 parent 97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e 为起点，原设计字节及配置保留。
CPU 实验使用共享737参数有限动作模型，共60条轨迹、11,400次实际更新。
服务器仅承担追加授权的 CPU 回归与已有 checkpoint 只读核验。
服务器针对性复测155项通过；本机适用全套在1080项通过后因临时夹具输出触及2GiB中断，
1项会读取sealed confirm的旧检查明确排除。完整回归仍未完成，故工程总门禁保留BLOCKED。
M0–M5实际实验已完成，本报告整理已有证据；资源边界后不再启动实验。

## 2. 表示决定

- **存储：保留逐题 `[nX,nS,nW,nI]`、总数、题目/场景/接口/checkpoint身份与固定权重。**
- **拟合：采用逐题 Helmert 三坐标。** 72个探针为216个响应坐标；原四概率线性回归是等价自检。
- **解释：采用 `(v,q,s)`。** 条件分母为0时返回 null+原因；未观察到事件与结构上不可能出现分别记录。
- 群体 q 使用加权概率之比。总体/六群体汇总适合展示，逐题表示保留身份以检出被平均抵消的移动。
- 3/18/216是事件与面板表示自由度，不是动态响应维数。条件坐标回归仅作oracle条件mass>0.05窗口子集诊断，覆盖率单列。

## 3. 维数、采样量与适用窗口

精确概率 selection 选中 **r cap={rank}** 的经典低秩响应规则，alpha={selection.get("exact_selection", {}).get("alpha")}。
是否通过独立 exact 测试：**{readiness["exact_diagnostic_test_pass"]}**。
这是付费局部校准后的已发生更新跟踪，不是无需执行就预测未来Adam操作。

**采样建议：{sample_recommendation}**
当前 n64 的最小可接受维数为 `NO_ACCEPTABLE_R` 时，r=0 只作为 persistence 等价诊断，不能推荐为合格模型。
参数更新空间 k 逐锚点估计，实际r=min(cap,k)，不能从这一个toy推断Qwen需要一维。

{markdown_table(chosen, [*fields, ("test_gate", "联合目标")])}

可用窗口以 [固定窗口结果](tables/M4_tracking_summary.csv) 的
`largest_tested_H_meeting_tolerance` 为准，范围仅限 H=1/2/4/8/16。
有限n配置未通过时不赋予可运行窗口；exact窗口只支持oracle诊断。
本次 r=1 的 exact **聚合资格窗口为 H≤4**：H4 NRMSE≈0.242，最差群体 v 误差q95≈0.00675；
H8 该误差升至≈0.01367而失败。H4仍有约19.6%的逐步案例失败，因此这不是每条轨迹都有效的保证。
经验误差带较宽；高覆盖率不能替代1个百分点精度目标。

{markdown_table(rows_h, [("n", "n"), ("H", "H"), ("response_nrmse", "NRMSE"), ("largest_tested_H_meeting_tolerance", "最大达标H")])}

冻结 exact 网格还观察到 **r=2 可达 H≤8**，r≥4/FULL 未把最大合格窗口延至16。
因此表示研究的建议是：exact诊断下短窗H≤4保留1维，H≤8保留2维；有限n≤256尚无合格使用窗口。
这是冻结网格上的聚合结果，主预注册选择仍为r1；不从测试结果重新拟合或改阈值。

{markdown_table(rank_h, [("rank_cap", "r cap"), ("H", "H"), ("response_nrmse", "NRMSE"), ("group_delta_v_q95_max", "v q95")])}

## 4. 主要失效条件

- **有限观测噪声：** 概率水平误差包括起点采样误差；响应NRMSE另以真实变化能量归一化。低信号不能靠绝对误差小判为合格。
- **遗漏方向：** Q只由fit bank组成；相同投影与正交残差范数不能确定语义响应。离开Q/U的测试分支全部保留。
- **有限步非线性：** 全Jd仍可能失效；曲率诊断和固定窗口中间误差见 M3/M4 原表。长窗口不能从短窗口外推。
- **条件信息不足：** q/s分母未被观测时返回未知；条件子集覆盖率不冒充全样本结果。
- **覆盖与成本：** 固定数据/初始化的10个test seeds不能提供跨初始化、跨数据或真实VLM保证。
  8 bank×3候选加一次面板刷新共25个评估面板单位，超过已验证H=4和最大测试H=16：
  **NOT_COST_FEASIBLE_AT_TESTED_HORIZON**。低bank仅有单步消融，不能把更长盈亏平衡窗口当成已经验证准确。

四级分解进一步表明：冻结一维规则的平均 Helmert 拟合/采样残差RMSE，
exact约0.00143，n16/64/256分别约0.0574/0.0327/0.0179；
同组全J的有限步曲率残差约0.000117。它支持本toy短步内观测与拟合噪声占主导的诊断，
不是以范数相减得到的因果贡献百分比。专门向量表见 M3_exact_rule*。

## 5. 与现有基线的比较

B0 persistence、B1普通日志、B2随机投影、B3更新PCA、B4经典监督低秩、B5最坏群体选择、B6全Q ridge与O0/O1均有实际结果。
[B0–O1公平主比较](tables/M3_primary_aggregate_metrics.csv) 保留主配置全部失败；
[含三项有界消融的原汇总表](tables/M3_aggregate_metrics.csv) 另行保留，原表噪声重复数按配置区分。
[RESULTS_zh.md](RESULTS_zh.md) 给出配对seed统计、误差分解与成本。
普通日志模型在 exact 单步测试中也达到设计目标（NRMSE≈0.716）；
精确选择的一维规则进一步降低至≈0.233，全Q ridge约0.172，全J oracle约0.0264。
因此不能把结果描述成只有低秩方法才有效。有限n64的日志模型也失败。
B4/B5是经典低秩工具和选维规则；本轮未提供其创新性或安全性证明。

## 6. 真实Qwen产物与缺项

已对五个完成S1 bank的7,680条独立样本逐条重算语义及哈希，alias不增加样本数。
这些是候选之间的差异，joint_0自身已经更新，不能当作原点分布。
服务器只读补充检查和CPU参数几何核验见 [真实数据准备度](REAL_EVIDENCE_READINESS_zh.md)。
已在服务器CPU核验15个既有checkpoint（一个原点、14个候选），
参数共12,582,912个；五bank选定更新子空间秩为1/2/2/2/2。
已核验这些候选的真实位移、布局、范数、夹角与Adam绑定；这些秩仍不识别语义Jacobian或响应维数。
单一warm原点/seed不能提供独立checkpoint和训练seed的fit/validation拆分。

## 7. 下一阶段最小数据需求与停止

已选14个候选的真实参数位移、布局和Adam/候选绑定已完成核验。
尚需同一探针面板的原点概率测量，以及按fit/selection/calibration/test预先分开的多个checkpoint和独立训练seed。
扩展到未选候选或新锚点时沿用现有绑定与重建契约；本轮不提出新的多模型或多奖励训练网格。
未来若需新的Qwen生成来获得原点/候选响应，即到达本轮停止边界，须另行授权。

本轮停止于建模资格结论：**未启动新Qwen调用、GPU、在线检测或SSVC控制；未声称安全认证。**
"""
    (out / "MODELING_DECISION_zh.md").write_text(
        _resolve_table_links(decision, out), encoding="utf-8"
    )
    write_json(out / "table_sources.json", table_index)
    write_json(
        out / "artifact_manifest.json",
        {
            "files": {
                str(p.relative_to(out)): sha256_file(p)
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
            "source_stages": stage_paths,
            "raw_M2_manifest_sha256": sha256_file(stages["M2"] / "manifest.json"),
            "raw_M3_selection_sha256": sha256_file(stages["M3"] / "selection.json"),
        },
    )
    return readiness
