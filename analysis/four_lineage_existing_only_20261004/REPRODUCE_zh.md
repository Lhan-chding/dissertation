# 复现入口与文件说明

先读[报告](ANALYSIS_REPORT_zh.md)，讨论用[单页摘要](FINDINGS_FOR_DISCUSSION_zh.md)。本目录只包含已有数据诊断；原始回答、权重和完整数据包不随此分析包重复分发。

依赖：Python 3，numpy（核谱用）；其余分析使用标准库。需要原实验仓库历史及同批完整数据包；保存选择器的四个导入模块会与实验commit `7af92dff42e9c0432716550f6611dc244af23ecb`做字节核对，遇差异停止，不用main替换解释。

1. 在仓库的本次分析分支工作。将完整包内`raw/FIRST_FOUR_RAW_ANALYSIS.tar.gz`解压到独立临时目录（已有安全路径/哈希回执可复用，勿再次执行全量审计器）。该导出明确不含最终T。
2. 选择已安装numpy的Python，执行：

```bash
python3 analysis/four_lineage_existing_only_20261004/run_analysis.py \
  --raw /path/to/extracted/FIRST_FOUR_RAW_ANALYSIS
```

脚本顺序为B（含H8/H32同题）、C/F（关联B表）、D、E、清单整合、验收。它只覆盖本新分析目录的派生表，不修改旧实验。不要运行旧`pilot_four`命令：它会重新拟合，超出本任务范围。

完整入口已实际执行并通过，退出码0，见[复现回执](REPRODUCTION_RUN.json)。

本次实际raw目录为`/private/tmp/ssvc-four-existing-20261004`。B、C/F、E已用系统Python执行；D采用已安装numpy 2.3.5的bundled Python，其完整命令见[inputs_D.json](inputs_D.json)。各组件实际文件列表分别在[inputs_B.json](inputs_B.json)、[inputs_CF.json](inputs_CF.json)、[inputs_D.json](inputs_D.json)、[inputs_E.json](inputs_E.json)，[input_scope.json](input_scope.json)引用清单并集中说明缺项、版本和脚本SHA。临时路径可替换；raw清单成员名映射保持一致。

最后执行[verify_analysis.py](verify_analysis.py)核对现有派生表的权重恒等式、保存预测、原折内BestStatic、分子分母、I缺失、H8交集、组内支持和交付链接。它使用实际已有证据，不生成toy数据。结果见[VALIDATION.json](VALIDATION.json)。这不是重新评分/重新验证所有原始实验；旧审计回执仍是原始权威记录来源。

## 索引与口径

- [统一观察索引](unified_observation_index.csv)：评估按原点/动作/题，训练按成功更新/组，P按快照/题；`draws`为该聚合记录内输出数。历史E的源步不适用，空值保留；source训练`source_step`为实际训练步、无分支H。
- [产物清单](artifact_inventory.csv)：H8/H32/E端点、保存模型/fold、已有表，以及未找到的独立分析输入。
- [机会与六群体](opportunity_by_stage_group.csv)：`aggregation=origin/stage`不可重复汇总；pX、C和坐标准确率用全部输出，F/B/M/copy用有效输出。C为原关系分数，copy为原语义字段。
- [逐题贡献](prompt_effect_contributions.csv)：H32分类仅事后使用，贡献权重固定1/144；`in_top_80pct_absolute_mass`包含跨过80%门槛的那一题。
- [训练支持](support_response_summary.csv)：不同aggregation_level不能混加；eval字段仅群体对应，绝非同prompt训练因果。
- [核几何](kernel_geometry.csv)：组件edf不相加；中心化值仅几何，X-X归因口径详见[D说明](D_FINDINGS_zh.md)。
- [前置条件结构](prestate_reward_conditional_repair.csv)：a为history、b为current，零支持UNKNOWN，I缺失；比较去噪须用相同支持域，详见[E说明](E_FINDINGS_zh.md)。
- [时间和成本](cost_existing_only.csv)：successful单GPU工作小时，不是并行墙钟；精确零优势成本不是已可节约时间，详见[CF说明](CF_FINDINGS_zh.md)。

所有概率与差值CSV默认0–1；报告乘100显示百分点。脚本只用保存语义字段/计数，不改变parser或reward，不导入GPU运行入口执行模型。固定8/8为一次确定性拆分，无随机bootstrap。详细零操作计数见[EXECUTION_SCOPE.json](EXECUTION_SCOPE.json)。
