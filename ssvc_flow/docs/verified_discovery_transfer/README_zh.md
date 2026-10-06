# 可验证发现与 O0 监督回迁

本实现对应用户提供的 2026-10-06 实验包。设计入口为
[START_HERE_FOR_CODEX.md](design/START_HERE_FOR_CODEX.md)。
原始 ZIP SHA-256：`51e558db4def4388b4a9bd4c8a96d5af7bef7ec148da73e5fee73aaa852095ff`。

新分支从 `2f8c6b19970f49542b009f2816f1c1a91eb9974c` 建立，旧工作区和旧流水线 STOP 保留。
模型调用、训练与科学结论各自记账；CPU 单测或队列注册不表示真实实验完成。

当前进度见 [第一阶段事实报告](FIRST_STAGE_REPORT_zh.md)：三小时监控发现五个worker于10月6日16:18停止；已完成6个SFT臂，保存数据及R0恢复点已核验，尚未自动重提，完整实验未结束。

## 输入及信息边界

- T384/V96/E384 使用新的合法 0–99 真值分布，不声称历史 ID 精确复现。
- G96 包含 64 个与 E 配对的图像提示及 32 个独立 duplicate；不计入 E 主分数。
- 历史排除采用完整真值排序多重集；已审计来源与范围限制见
  [历史输入说明](evidence/historical_inputs/README_zh.md)。
- R64 是既有训练 O0 成功输出，按固定场景 ID 选择；不是求解器补出的 SELF。
- RoleDataset 强制角色、固定 split、文件哈希及最终揭晓门槛。它防止程序误读，
  不是同一 Unix 用户下针对恶意 Python 的操作系统安全沙箱。
- SELF 仅使用 raw teacher 回答的公开验证结果；GOLD 的 T 真值读取权限单列。
- E/G 的模型回答在 `sealed/` 下；所有登记任务到达明确终态并通过完整性核查后统一揭晓。

## 代码与运行

新增代码在 `src/verified_discovery_transfer/`；复用已核验的 O0/L11/B1、
Qwen raw-only uncached 推理及 R0 on-policy 损失。SFT 是独立的全序列 causal 目标，
没有把筛选后的标签套入旧 PPO ratio。

CLI 提供 `preflight`、`prepare-data`、`bridge`、`build-queue`、`worker`、
`report` 及显式技术恢复 `retry`。入口命令可用 `--help` 检查。
机器路径必须来自实际核查，模板本身不可当作运行证据。

最多五个独立单 GPU worker；每个 Slurm 作业使用其实际分配的 `cuda:0`。
`scripts/verified_discovery_transfer/launch.sbatch` 固定原环境、缓存和任务临时目录，
不安装或升级依赖。父模型恢复检查、真实 GPU bridge 和训练队列分别留存回执。

严格相同训练 view、parent、repeat、tokenizer、初始化和完整设置才允许别名。
空发现集合返回父模型并保留端到端结果；不补金标，不删不利臂。
所有 SFT 固定 256 步、相同每题 completion 均值与 12 focus/4 replay 权重；
REPLAY_ONLY 的回放总权重仍为 0.25。R0_RESET32 是 fresh-Adam 的上下文参照，
不声称与 SFT 等计算预算。

## 可核验记录

- [父状态核验](evidence/PARENT_VERIFICATION.json)：两父文件及完整 CPU 状态哈希；GPU 恢复单独检查。
- [新数据清单](evidence/COHORT_MANIFEST.json)：配额、容量、文件哈希与历史排除范围。
- [技术预检](evidence/TECHNICAL_PREFLIGHT.json)：服务器 CPU 入口核查。
- 清理脚本只允许三个旧 pytest fixture 根，逐文件清单与删除日志保留在服务器独立 evidence 目录。
  真实研究数据、父检查点、失败日志和验收汇总不属于清理白名单。

实际作业与完成状态以 `EXECUTION_SCOPE.json`、Slurm 回执、bridge、训练 checkpoint
和完整响应数量为准。最终报告分别分析发现、拟合、迁移及保持性；缺失证据明确列出。
