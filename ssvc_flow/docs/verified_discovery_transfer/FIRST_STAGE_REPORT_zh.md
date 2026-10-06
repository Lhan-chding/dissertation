# 第一阶段事实记录：等待 GPU 调度

截至 2026-10-06 12:10（新加坡时间），本轮尚未产生真实 Qwen bridge、teacher、SFT、R0 或 E/G 结果。Slurm 作业 **187023** 已提交，状态 **PENDING / QOSMaxJobsPerUserLimit**。`rose` QOS 每用户最多同时运行 2 个作业、最多提交 5 个作业；当前两个运行名额被既有作业占用，未修改它们。排队时间不计作 GPU 执行或吞吐。

## 已完成且有证据的工作

- 从给定 `START_HERE_FOR_CODEX.md` 开始，原 ZIP 的 24 个文件按字节保留于 `design/`；隔离工作区、旧流水线 STOP 和用户原有改动保留。
- 清理三个旧 pytest fixture 根下可重新生成的白名单文件：103,723 个，逻辑大小 5,632,339,970 字节（5.63 GB）。删除前逐文件哈希登记，删除后剩余目标为 0。真实实验原始结果、唯一检查点、失败日志、验收汇总、源码和共享 inode 保留。完整删除清单在服务器项目的 `evidence/`，本地见 [清理回执](evidence/CLEANUP_RECEIPT.json)。
- S96、REP96 均核对 COMMIT/manifest、checkpoint 文件 SHA-256 与完整 CPU 状态哈希；每父 192 个有限、名称/形状/dtype 相符的 LoRA 张量。此项不替代 GPU 恢复及前向验证，见 [父状态核验](evidence/PARENT_VERIFICATION.json)。
- 新 T384/V96/E384 与 G96、R64 已生成并冻结。T/V/E 来自 0–99 新数值域；G 含 64 个 E 配对图像接口和 32 个独立 duplicate；R 为历史真实 O0 成功训练输出的 32/16/16 分层选择。
- 历史排除读取 84 份现存元数据，登记 3,323 个真值轨道、3,822 个 scene ID；29 个数据根绑定归并为 6 个现存根。已删除、外部或只在未审计归档内的来源不能被这一核验覆盖，范围见 [历史输入说明](evidence/historical_inputs/README_zh.md)。
- 48 项针对性 CPU 测试通过，附带参考脚本 53 项测试通过；新增/修改代码 Ruff 与 Slurm 脚本语法检查通过。mask、shift、梯度累积和崩溃恢复的 CPU 结果不称作真实 Qwen 验收。
- 冻结注册 294 个工作项，其中 SFT 20 个、R0_RESET32 4 个。当前全部 PENDING；矩阵登记不是 294 个 Slurm 提交。见 [注册矩阵](evidence/REGISTERED_MATRIX.json)、[执行范围](evidence/EXECUTION_SCOPE.json)。

## 已实现的实验边界

SELF_O0 / SELF_MIX / SELF_SINGLE 每题 16 次调用，同一 completion-only SFT 设置、256 次更新、12 focus + 4 replay。SELF 只接受原始回答经公开验证得到的目标，失败不补求解器标签；GOLD 权限单列。相同 parent/repeat/完整设置下严格相同训练 view 才别名复用；空发现集合返回父模型并保留结果。SFT 不使用旧 PPO ratio。

角色加载器约束 public、gold、trainer、selector、final_eval 的读取；E/G 在登记矩阵终态与完整性核验后才统一揭晓。该软件边界不是同一 Unix 用户下抵御恶意代码的操作系统沙箱。训练、发现、原始负结果、物理重算和技术失败均保留。最终分析分别报告发现、拟合、O0 迁移和 G 保持性。

## 真实 GPU 验收及未完成项

作业 187023 申请 1 张 PRO6000，调度器实际资源为 4 CPU / 33 GiB。它按顺序对 S96、REP96 做 8 个固定训练题的目标 mask、EOS、因果性、PAD、单例/批量 loss、microbatch 梯度、prefix/full 梯度和两步精确恢复验证。BF16 差异必须依据真实诊断审阅，不能用 CPU PASS 或任意放宽阈值替代。

目前 GPU 恢复、bridge、teacher 发现、GOLD/SELF SFT、R0、V 诊断及 E/G 评估均未完成，吞吐未知，无科学结论。正式 worker 尚未提交；bridge 完成也不自动放行数值审阅。最多 5 卡是上限，不是已获得资源。当前 QOS 限制下不能保证五个单 GPU 作业并行。

继续点：读取 187023 的 Slurm 状态、输出和 `run_v1/bridge.json`，先处理真实技术诊断，再启动已登记队列。不要重建新数据、修改冻结源代码或重复提交已有 bridge。若需技术修复，保留旧 run 与失败证据，明确新代码版本及身份绑定。未增加定时监控。

## 代码及路径

代码提交：`6902fe8fa7634a819ac9dd9a4cfdb537d27559b7`，远端分支 `codex/verified-discovery-transfer-20261006` 已核验。

服务器项目：`/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006`。
正式源码：`code_v1/ssvc_flow`；冻结运行目录：`run_v1`；配置：`machine.server.json`。
运行身份：`4de8c0c0358cf73617414b03da6e15304c6d03629057ff3a24893c43ef5e44fd`。
新 cohort digest：`ad1d1a7dc193ed0ae9e35d5a9c17e473c35f565d8e6e9a33e34eded3176e15af`。
矩阵身份：`717f161def098ac587140c4578bfc790544dccc76af8f7d3da2411170c599e5d`。

实际命令见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。未生成声称完整实验完成的 FINAL_FINDINGS。
