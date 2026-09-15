# Codex 执行入口：先完成建模资格，不启动新的大模型训练

## 任务

先阅读 `CODEX_MODELING_EXPERIMENT_PLAN_zh.md`，再对照 `protocol.json` 与 `CPU_ACCEPTANCE_CHECKLIST_zh.md` 实施 M0–M6。重点是：**先确定表示和局部响应维度是否够用，暂不实施在线检测/反馈控制。**

本包不是已经实现的仓库补丁。`reference/` 是独立数学参考，不是完整实验代码；本次已实际执行22项参考测试，全部通过，原始日志与环境信息在该目录。仓库级测试和完整 M0–M6 尚待 Codex 实施。本次没有修改仓库，也没有启动服务器工作。

## 当前代码起点

`Lhan-chding/dissertation` 的后续分支 `codex/ssvc-mechanism-followup-20260914` 在本次核读时已到：

```
97b7fd646046cfd58c45e0083f0e8fe6f37d6f1e
```

它已经包含 followup 实现和新的执行文档。不要回到旧的“分支仍为空”的判断。开始时重新读取当前实际 HEAD 和已有修改，记录来源。

在独立 worktree/新分支 `codex/ssvc-modeling-qualification-20260915` 实施。不要改写正在运行的旧实验源码，不操作现有队列，不自动进入旧 S2。这个文档不授权取消此前任务。

## 先做什么

1. 检查工作区与项目规范，建立代码/配置清单。
2. 将本设计原件放入新文档子目录，保持来源可追溯。
3. 实现 `src/modeling_qualification/` 与测试，完成 M0/M1。
4. 用微型 CPU 模型完成 smoke，测内存/时间，再决定是否在本地预算内运行 core。
5. 用同一份 CPU 轨迹完成 M3/M4，不能每个表示单独重训。
6. 对当前可访问的真实完成产物执行只读 M5。缺原件只阻塞相应项。
7. 输出最终建模选择、失败范围与下一步最小数据需求，停止。

## 拟实施 CLI 契约

以下是**需要实现并测试的新 CLI**，不是声称仓库当前已有这些命令。Codex 可以在确保同等契约的前提下调整参数名，但必须在报告中给出最终实际命令。

```bash
# 工作目录：新 worktree 的 ssvc_flow/
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
: "${PY:?请设置已经核验的 CPU Python 解释器}"

"$PY" -m src.modeling_qualification.cli validate --config configs/modeling_qualification/protocol.json
"$PY" -m src.modeling_qualification.cli math --config configs/modeling_qualification/protocol.json --out runs/modeling_qualification/<new_run>/M0
"$PY" -m src.modeling_qualification.cli witnesses --config configs/modeling_qualification/protocol.json --out runs/modeling_qualification/<new_run>/M1
"$PY" -m src.modeling_qualification.cli collect --profile smoke --config configs/modeling_qualification/protocol.json --out runs/modeling_qualification/<new_run>/smoke
# 实测预算合格之后才进入 core
"$PY" -m src.modeling_qualification.cli collect --profile core --config configs/modeling_qualification/protocol.json --out runs/modeling_qualification/<new_run>/M2
"$PY" -m src.modeling_qualification.cli fit-evaluate --config configs/modeling_qualification/protocol.json --input runs/modeling_qualification/<new_run>/M2 --out runs/modeling_qualification/<new_run>/M3
"$PY" -m src.modeling_qualification.cli track --config configs/modeling_qualification/protocol.json --input runs/modeling_qualification/<new_run>/M2 --models runs/modeling_qualification/<new_run>/M3 --out runs/modeling_qualification/<new_run>/M4
"$PY" -m src.modeling_qualification.cli audit-real --readonly-root <existing_completed_evidence> --out runs/modeling_qualification/<new_run>/M5
"$PY" -m src.modeling_qualification.cli report --input runs/modeling_qualification/<new_run> --out docs/modeling_qualification/<new_report>
```

尖括号必须替换为实际新路径。缺失真实证据时不要执行伪造的 `audit-real`；报告缺项并完成其它阶段。

## 不能走偏的地方

- 四事件只有三自由度，不等于动态维数是三。
- `(v,q,s)` 用于解释；原始四计数与逐题身份始终保留。
- 低维模型必须与随机投影、更新 PCA、经典监督低秩、全观测方向 ridge、普通日志和 persistence 比较。
- 已发生更新 d 的跟踪，不等于无需试算就预测未来 Adam 操作。
- 所有 oracle 信息与可用观测分开，测量开销计入成本。
- 更新范数相同不代表方向相同；投影加残差范数也不唯一决定语义后果。
- 在固定窗口内评估建模误差，不提前实现自适应 trigger 或 reward controller。
- 不隐藏失败，不以“必须达标”调整数据/阈值。`NO_ACCEPTABLE_R` 是允许的研究结论。

## 完成时交付

代码提交、实际测试日志、资源记录、`MATH_AUDIT_zh.md`、`RESULTS_zh.md`、`MODELING_DECISION_zh.md`、`REAL_EVIDENCE_READINESS_zh.md` 和 machine-readable 状态。

报告哪些命令实际执行、哪些被阻塞。首次需要新的真实 Qwen 调用或超出 CPU 预算时停止，给出精确原因，不自行启动 GPU。
