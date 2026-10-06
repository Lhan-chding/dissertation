# Codex 从这里开始

## 任务

完成“可验证协议发现 → O0监督回迁”的代码和真实实验。不是只写计划；但也不启动未登记的新选择器、动态奖励或off-policy重要性修正。

读取顺序：`OVERVIEW_zh.md` → `CODEX_EXECUTION_PLAN_zh.md` → `protocol.json` → `THEORY_AND_STATISTICS_zh.md` → `GPU_RUNBOOK_zh.md` → `CPU_ACCEPTANCE_CHECKLIST_zh.md`。

### 重要状态

本包的参考脚本与测试已运行，但它们没有接入真实Qwen；新T/V/E/G尚未生成，新teacher推理、SFT与R0参照均未运行。历史checkpoint路径只是导出时记录，服务器可读性待确认。

仓库核读起点为`codex/protocol-state-probes-20261005`，`2f8c6b19970f49542b009f2816f1c1a91eb9974c`。建立独立worktree；不覆盖已有结果，也不强制回退用户未提交改动。

## 首先完成

1. 核实S96、REP96；准备新0–99真值数值域任务，并明确它不是旧ID复现。旧D/P/U22不作新确认。
2. 定义严格public-only生成、公开验证、规范目标和审计字段隔离。
3. 接入已有raw-only推理；新建response-only全序列SFT，恢复已有LoRA而非重置。
4. 完成针对性测试和小GPU桥接，按真实吞吐生成执行队列。

## 默认执行

最多5张PRO6000。teacher发现与GOLD参照可并行；同预算SELF_O0、SELF_MIX、SELF_SINGLE保持相同SFT目标。S96有匹配金标和REPLAY_ONLY诊断。空J返回父模型，不补金标；同训练view且同完整设置合并别名。

第一个阶段只展示发现、训练/V诊断和技术问题；主E结果封存。不能根据早期赢家删除预登记的其他臂。最终E/G完成后统一读出并分析。

## 交付

- 实际代码提交与改动列表、针对性测试记录；
- FIRST_STAGE_REPORT_zh.md、实际运行命令和吞吐；
- 数据/训练注册表、全部发现集合与原始输出、checkpoint路径；
- FINAL_FINDINGS_zh.md、EXECUTION_SCOPE.json；
- DISCUSSION_FOR_CLAUDE_zh.md：把发现、回迁、拟合和保持性分开解释。

## 不要做

不要新增源模型训练或88分支奖励扫描；不要把筛选SFT叫无偏GRPO；不要拿公开验证器失败后求解器补的标签当self；不要用真值修正输出排列；不要按测试结果新增采样、步数或超参数；不要反复全模型哈希和重复全仓测试。

如服务器运行授权未在Codex所在环境授予，完成代码、CPU验收和准确GPU启动清单后停在提交前；不能声称已经在服务器运行。缺少单个原件时继续其他不依赖项并记录具体缺项。
