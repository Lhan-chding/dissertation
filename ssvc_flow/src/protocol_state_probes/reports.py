"""Bounded evidence reports; engineering completion never supplies a scientific answer."""

from __future__ import annotations

from pathlib import Path

LIMITS = (
    "逐题区间为95%边际Clopper-Pearson区间，不是多题同时保证"
    "。固定面板区间以独立二项模型及Jeffreys先验为条件，另给均匀先验敏"
    "感性；同一别名共享随机变量。B1旧/新互斥预测偏好保留同次输出的协方差，"
    "使用三类Dirichlet抽样。C及损坏/修改计数不是二项概率，不报告伪"
    "造的Beta区间。场景bootstrap仅重抽题目，同一题的协议和检查点"
    "共同抽取，不再重抽回答。全零/全一时退化的plug-in SE和场景区间"
    "不表示精确已知。\n\nD48是已看过历史结果的开发诊断题；U22元数据已分"
    "析，是否仍未运行过历史结果须以单独服务器审计为准。主四状态来自同一610"
    "01源轨迹，不是四个独立训练种子；REP仅为61003一条预定复核轨迹。"
    "当前符号接口结果不支持一般视觉感知、唯一内部算法、潜在程序混合权重或训练"
    "控制收益的结论。"
)

TABLE_INDEX = (
    "- [逐题概率、分母及精确区间](tables/prompt_metri"
    "cs.csv)\n- [全部协议效应及逐题反例](tables/paire"
    "d_protocol_effects.csv)\n- [四格DPE结构转变"
    "](tables/DPE_transition_effects.csv)"
    "\n- [B1原新完整向量、越界预测及精确错误迁移](tables/B1_"
    "exact_error_shifts.csv)\n- [输入/输出2x2、"
    "四水平和命名格式](tables/input_output_factor"
    "ial.csv)\n- [T1及REP状态交互](tables/T1_in"
    "teractions.csv)、[T2状态交互](tables/T2_i"
    "nteractions.csv)\n- [独立执行对照](tables/e"
    "xecution_controls.csv)\n- [稀疏联合原子](ta"
    "bles/joint_behavior_atoms.parquet)、["
    "概率质量核对](tables/probability_mass_chec"
    "ks.csv)\n- [实际与剩余工作量](tables/workload"
    "_actual.csv)\n"
)


def _number(value, scale=1):
    return "N/A" if value is None else f"{value * scale:.3f}"


def _effect_table(rows, *, maximum=None):
    rows = list(rows)
    if not rows:
        return "无已登记适用单元。\n"
    text = (
        "| 面板 | 状态/交互 | 比较 | 分组 | 题数 | 点差pp |"
        " Jeffreys95% | Uniform95% | MC SE pp"
        " | 场景95% pp | 状态 |\n|---|---|---|---|"
        "---:|---:|---|---|---:|---|---|\n"
    )
    for row in rows[:maximum] if maximum else rows:
        group = row.get("DPE_transition", row.get("family", row.get("scope", "")))
        columns = [
            row.get("panel", ""),
            row.get("checkpoint", row.get("interaction", "")),
            row.get("comparison", ""),
            group,
            row.get("scene_count", ""),
            _number(row.get("estimate"), 100),
            f"[{_number(row.get('jeffreys_low'), 100)}, {_number(row.get('jeffreys_high'), 100)}]",
            f"[{_number(row.get('uniform_low'), 100)}, {_number(row.get('uniform_high'), 100)}]",
            _number(row.get("mc_se"), 100),
            (
                f"[{_number(row.get('scene_bootstrap_low'), 100)}, "
                f"{_number(row.get('scene_bootstrap_high'), 100)}]"
            ),
            row.get("status", ""),
        ]
        text += "| " + " | ".join(map(str, columns)) + " |\n"
    return text


def _workload_table(rows):
    text = (
        "| 检查点 | 阶段 | 面板 | 完成单元/计划 | 已提交回答/计划"
        " | 状态 |\n|---|---|---|---:|---:|---|\n"
    )
    for row in rows:
        columns = [
            row["checkpoint"],
            row["phase"],
            row["panel"],
            f"{row['complete_cells']}/{row['planned_cells']}",
            f"{row['committed_outputs']}/{row['planned_outputs']}",
            row["status"],
        ]
        text += "| " + " | ".join(map(str, columns)) + " |\n"
    return text


def _stage_status(rows):
    return "COMPLETE" if rows and all(row["status"] == "COMPLETE" for row in rows) else "INCOMPLETE"


def _rankings(rows):
    """Compare actions only on the SAME panel/family; never rank cross-only B1 vs all-family A1."""
    groups = {}
    for row in rows:
        if row["metric"] != "pX" or row["scope"] != "family":
            continue
        key = (row["panel"], row["family"], row["checkpoint"])
        group = groups.setdefault(key, {})
        for endpoint, value in row["endpoints"].items():
            protocol = endpoint.split(":", 1)[1]
            if protocol != "B0":
                group[protocol] = value
    text = "| 面板 | 家族 | 状态 | 各动作pX | 后见点值最优 |\n|---|---|---|---|---|\n"
    for (panel, family, checkpoint), actions in sorted(groups.items()):
        available = {key: value for key, value in actions.items() if value is not None}
        best = (
            "INCOMPLETE"
            if len(available) != len(actions) or not available
            else ",".join(
                sorted(key for key, value in available.items() if value == max(available.values()))
            )
        )
        text += (
            f"| {panel} | {family} | {checkpoint} | "
            + "; ".join(f"{key}={_number(value)}" for key, value in sorted(actions.items()))
            + f" | {best} |\n"
        )
    return (
        text + "\n这是后见样本点值排序，含选择乐观偏差；不构成路由收益估计。"
        "不同家族允许的动作集合不同。\n"
    )


def _control_summary(rows, checkpoint=None):
    selected = [
        row
        for row in rows
        if row["metric"] == "control_success"
        and (checkpoint is None or row["checkpoint"] == checkpoint)
    ]
    groups = {}
    for row in selected:
        key = (row["checkpoint"], row["panel"], row["protocol"])
        group = groups.setdefault(key, [0, 0, 0])
        group[0] += row["successes"] or 0
        group[1] += row["n"]
        group[2] += row["expected_outputs"]
    text = "| 状态 | 对照 | 协议 | 成功/已提交回答 | 计划回答 |\n|---|---|---|---:|---:|\n"
    for (cp, panel, protocol), (count, n, planned) in sorted(groups.items()):
        text += f"| {cp} | {panel} | {protocol} | {count}/{n} | {planned} |\n"
    return text + (
        "\n此处计数仅描述执行对照；逐题区间见独立表，不与D48/U22的修复pX"
        "合并，也不据控制成败删主样本。\n"
    )


def exposure_note(summary):
    audit = summary.get("U22_exposure_audit", {})
    status = audit.get("status", "U22_EXPOSURE_UNVERIFIED")
    if summary.get("U22_effective_split_role") == "development_diagnostic":
        return (
            "U22来源审计：CONTAMINATED_DOWNGRADED。实际历史"
            "执行记录发现题目暴露，整个U22按原清单保留，已降级为开发诊断资料（or"
            "iginal_split=train）；不能称为独立或未接触结果的复核面"
            "板。详细计数和来源见U22_EXPOSURE_AUDIT.json。"
        )
    role = summary.get("U22_effective_split_role", "exposure_unverified")
    return f"U22来源审计：{status}；有效角色为{role}。来源核实不能由本轮生成成功代替。"


def write_reports(run_path, summary, tables):
    root = Path(run_path)
    workloads = tables["workload_actual"]
    stages = {
        "INITIAL_S96_REPORT_zh.md": (
            "S96核心完整对照首批报告",
            lambda row: row["checkpoint"] == "S96" and row["phase"] == "primary_core_and_controls",
            lambda row: row.get("checkpoint") == "S96" and row["panel"] == "D48",
        ),
        "PRIMARY_REPORT_zh.md": (
            "主矩阵与输入输出因子报告",
            lambda row: row["phase"] in ("primary_core_and_controls", "input_output_factorial"),
            lambda row: (
                row.get("checkpoint") in ("S32", "S96", "R4_128", "DIRECT_128")
                and row["panel"] == "D48"
            ),
        ),
        "U22_REPLICATION_REPORT_zh.md": (
            "U22与REP复核报告",
            lambda row: row["phase"] in ("holdout_replication", "replication_existing_checkpoint"),
            lambda row: row["panel"] == "U22" or row.get("checkpoint") in ("REP32", "REP96"),
        ),
    }
    stage_statuses = {}
    for file_name, (title, select_workload, select_effect) in stages.items():
        stage_workload = [row for row in workloads if select_workload(row)]
        status = _stage_status(stage_workload)
        stage_statuses[file_name] = status
        effects = [
            row
            for row in tables["paired_protocol_effects"]
            if select_effect(row)
            and row["metric"] == "pX"
            and row["scope"] in ("family_balanced", "family")
        ]
        transitions = [
            row
            for row in tables["DPE_transition_effects"]
            if select_effect(row) and row["metric"] == "pX"
        ]
        text = (
            f"# {title}\n\n状态：**{status}**。"
            "本文件只读取本轮COMMIT确认的原始回答，smoke单列排除。\n\n## 做了什么\n\n"
            + _workload_table(stage_workload)
        )
        text += "\n\n" + exposure_note(summary) + "\n"
        text += (
            "\n## 哪些比较可解释\n\n只有全部组成端点有回答的固定面板对比才有估计；"
            "缺单元记MISSING_ENDPOINTS，不用0代替。PARTIAL_"
            "FIXED_N是固定预算尚未完成的进度，不用于提前停臂。O0/B0别名是"
            "一份观测，其差恒为0。执行对照单独分母。\n"
        )
        if not summary["statistics"]["registered_replication_counts"]:
            text += "\n**统计抽样次数为CPU测试设置，尚非登记的20000/5000次最终分析。**\n"
        text += (
            "\n## 结果\n\npX的完整汇总如下；负效应、零成功、非法和越界输出全部保"
            "留。全部组成端点与逐题表在CSV内。\n\n"
        ) + _effect_table(effects)
        text += "\nDPE四格：\n\n" + _effect_table(transitions)
        text += "\n执行对照：\n\n" + _control_summary(
            tables["execution_controls"], "S96" if file_name.startswith("INITIAL") else None
        )
        text += (
            "\n## 与假设的关系\n\nDPE方向、完整程序向量匹配、copy和anch"
            "or是不同观测，不能互相替代。B1的C_orig与C_display并列"
            "；其部分关系分数不是行变换不变量。程序预测与truth/copy重合的题"
            "不能单独区分程序机制。具体反例和未匹配质量已保留。\n"
        )
        text += (
            "\n## 仍未知与是否需要新训练\n\n"
            + LIMITS
            + (
                "\n\n本阶段不新增训练或状态路由。未完成的登记单元继续按既定矩阵执行；缺权"
                "重只局部记录，不能补训练。\n\n## 完整证据表\n\n"
            )
            + TABLE_INDEX
        )
        (root / file_name).write_text(text)
    summary["stage_statuses"] = stage_statuses
    write_final_decision(root, summary, tables)


def write_final_decision(root, summary, tables):
    complete = (
        summary["status"] == "COMPLETE" and summary["statistics"]["registered_replication_counts"]
    )
    status = (
        "FROZEN_MATRIX_COMPLETE_INTERPRET_WITH_LIMITS"
        if complete
        else "INCOMPLETE_NO_FINAL_SCIENTIFIC_DECISION"
    )
    text = (
        f"# 最终建模判断\n\n状态：**{status}**。"
        f"已提交{summary['committed_outputs']}/{summary['planned_outputs']}条主回答，"
        f"完成{summary['completed_cells']}/{summary['planned_cells']}个独立单元。"
        "所有结果以当前SUMMARY.json和完整CSV为准。\n\n"
    )
    text += exposure_note(summary) + "\n\n"
    if not complete:
        text += (
            "预定矩阵或登记统计设置尚未完成，以下是证据现状与判断边界，不能当作完成实"
            "验后的机制结论。缺失没有被解释成零效果、失败或无价值。\n\n"
        )
    text += "## 1. 反序是否符合DPE结构转变，能否区分纯位置解释？\n\n"
    text += _effect_table(
        row
        for row in tables["DPE_transition_effects"]
        if row["comparison"] == "A1-O0" and row["metric"] == "pX"
    )
    text += (
        "\n0→1预期方向为改善，1→0为损害；这些只是登记程序的预测。四格必须同"
        "时检查，反例保留。A1仅改变输出语义顺序，不能据此单独排除输入位置或命名"
        "格式；分离依据是第3项2x2。表中有值的差异是具体面板与冻结检查点上的条"
        "件响应，不能识别唯一内部程序。\n\n"
    )
    text += "## 2. B1是否改善两条关系消元子集，新增错误是否吻合字面预测？\n\n"
    text += _effect_table(
        row
        for row in tables["DPE_transition_effects"]
        if row["comparison"] == "B1-B0"
        and row["metric"] == "pX"
        and row["DPE_transition"] == "0->1"
    )
    text += (
        "\n具体旧/新向量、四个匹配概率及边际区间见[B1精确错误表](table"
        "s/B1_exact_error_shifts.csv)。仅预测改变的变"
        "体进入新增错误偏好检验；预测不变时偏好对比严格为0。新向量越界仍计入原始"
        "四整数匹配，其主事件为I。新增匹配也可能是真值改善，因此表中同时标记ne"
        "w_prediction_is_truth，不把所有变化都叫新错误。不能"
        "把历史8题统一设为同一方向。\n\n"
    )
    text += "## 3. 输入位置、输出位置与命名格式各有什么影响？\n\n"
    text += _effect_table(
        row
        for row in tables["input_output_factorial"]
        if row["metric"] == "pX" and row["scope"] == "family_balanced"
    )
    text += (
        "\n输入主效应平均L10−L00和L11−L01；输出主效应平均L01−L"
        "00和L11−L10。交互为L11−L10−L01+L00。命名格式单列"
        "L00−O0和L01−A1；不把L10−O0叫纯输入位置效应。缺任何端点"
        "时不形成完整因子判断。\n\n"
    )
    text += (
        "## 4. 锚定、完整程序匹配和其他输出是否需要分别保留？\n\n需要分别保"
        "留。它们是不同且重叠的可观察事件：anchor_joint以全体输出为分"
        "母，anchor_given_parse另列；程序匹配含域外四整数诊断并"
        "标记与copy/truth重合。other是可解析且不在copy/V1/"
        "V2集合中的输出，outside_reference_union还计入不"
        "可解析输出。联合原子保留这种重叠，不以边际均值或独立Beta样本伪装完整"
        "联合分布。实际概率和反例见[逐题表](tables/prompt_met"
        "rics.csv)及[联合原子](tables/joint_behavi"
        "or_atoms.parquet)。这些指标的保留不等于已证明其独立预测"
        "或路由价值。\n\n"
    )
    text += "## 5. T1/T2是否有条件效应差异，是否改变动作排序？\n\n"
    text += _effect_table(
        row
        for name in ("T1_interactions", "T2_interactions")
        for row in tables[name]
        if row["metric"] == "pX"
        and row["scope"] == "family_balanced"
        and row.get("interaction") != "REP_T1"
    )
    text += (
        "\n所有四端点均在表中，不合并检查点当作独立训练种子。接近0/1时改善空间"
        "不同，不能只比较差值幅度。相同面板/家族的点值排序如下：\n\n"
    ) + _rankings(tables["paired_protocol_effects"])
    text += (
        "\n条件效应幅度不同不足以证明状态路由必要；还要看到有价值的动作排序变化，"
        "并在U22/另一个既有源状态得到支持。这里没有前瞻路由评估，也没有无偏o"
        "racle收益估计，现阶段不能宣布路由优于最佳统一协议。\n\n"
    )
    text += "## 6. 哪些结果在U22与REP重现，哪些只在历史诊断题出现？\n\n"
    text += _effect_table(
        row
        for row in tables["paired_protocol_effects"]
        if row["metric"] == "pX"
        and row["scope"] == "family_balanced"
        and (row["panel"] == "U22" or row.get("checkpoint") in ("REP32", "REP96"))
    )
    audit_paths = sorted(path.name for path in root.glob("*U22*json"))
    text += (
        "\n复核必须逐一对照D48同一比较的方向、大小和区间；不能把未完成或宽区间"
        "叫作未重现。U22是否未被既往训练/评估使用，以服务器来源审计为条件。"
    )
    audit_names = ", ".join(audit_paths) if audit_paths else "尚无专门U22审计JSON文件"
    text += (
        f"本目录可见相关审计文件：{audit_names}。"
        "REP只增加一条既有源轨迹，不能估计一般训练种子方差。\n\n"
    )
    text += "## 7. 现阶段是否需要新训练或状态路由；若训练，要检验什么？\n\n"
    text += (
        (
            "本轮冻结矩阵已完成。冻结提示响应只支持相应条件下的行为描述，不能直接证明"
            "新训练必要或有效，也不能证明语义状态路由有增益。"
        )
        if complete
        else "本轮登记证据尚未完成，目前没有充分依据转入新的训练或状态路由。"
    )
    text += (
        "下一轮若要训练，必须预先选择可检验目标：协议是否促进学习、收益能否迁移到"
        "无辅助原协议、或同等信息权限下语义观测能否改善操作选择。三者需要各自的新"
        "设计与验证，本轮不替它们给出收益结论；不自动训练、不拟合路由、不删掉效果"
        "差的臂。\n\n"
    )
    text += "## 统计与解释边界\n\n" + LIMITS + "\n\n## 全部证据\n\n" + TABLE_INDEX
    (root / "FINAL_MODELING_DECISION_zh.md").write_text(text)
