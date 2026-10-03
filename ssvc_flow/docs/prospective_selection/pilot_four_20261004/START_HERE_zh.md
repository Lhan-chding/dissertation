# 供idea与算法设计分析的小实验数据包

## 先看什么

1. `PILOT_REPORT_zh.md`：首四条离线闭环结果与解释边界。
2. `PILOT_PLAN_zh.md`：结果计算前固定的嵌套整条留出规则。
3. `selected_outcomes.csv`、`all_candidate_predictions.csv`、`inner_tuning_288.csv`：选择、预测和调参依据。
4. `decision_diagnostics.csv`、`alpha_sensitivity_POSTHOC.csv`、`folds/`：逐原点边际、对alpha的敏感性及拟合核诊断。
5. `../first_four_20261004/`：88分支响应矩阵、六群体、信息冗余、旧E、吞吐。
6. 完整包`raw/FIRST_FOUR_RAW_ANALYSIS.tar.gz`：逐次生成文本、token/logprob、reward/semantic、训练更新、任务回执、输入题面/图像和原训练代码。

## 数据单位

四条独立训练lineage61001–61004；每条t32/t96两个原点；每原点R0–R7八候选及GDPO/SAW/DIRECT_REPAIR三个固定对照，共88分支，每分支32步、repeat1。新增修复结构能否帮助选择未来奖励配置，是要研究的核心问题。

小实验外层每次整条留出，余下三条内层按lineage调参。四层方法同候选/目标/网格。四层共16模型、32最终选择，均等于对应折内BestStatic；观察Z3−Z2为0。此结果可以用于重新研究表示与算法，但不能直接归纳为信息无价值，也不能通过事后更换规则改写本轮结果。

## 交给后续分析者的任务

请先检查数据权限隔离和统计单位，再分析：响应异质性是否足以让状态选择有收益；信息表示保留的差异是否对应响应；调参/正则/核权重是否导致静态选择；标签采样噪声与训练随机性哪些可以或不能从现有数据辨认。区分已证事实、分析推断、尚需新实验的问题；提出算法改进时注明使用了哪些开发结果、需要怎样的下一轮验证。

不要默认扩展到完整矩阵，不用外层结果重新选择本轮alpha，不把多个draw、动作或同源anchor当独立lineage。任何新设计都作为后续版本，不覆盖已有模型、选择与原始结果。

复算入口和原始数据覆盖见REPRODUCE_zh.md、DATA_COVERAGE_zh.md。轻量核心包足以复算选择器；完整包另含逐样本原始数据。两者均不含基座和Adam张量。
