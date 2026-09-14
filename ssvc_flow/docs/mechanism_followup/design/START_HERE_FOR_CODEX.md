# Codex：从这里开始

本包用于接续 `Lhan-chding/dissertation` 的 SSVC 实验。请实现代码并实际运行 CPU 测试，完成后停在服务器启动之前。

## 阅读顺序

1. `CODEX_CONTINUE_EXPERIMENTS_zh.md`：实验、代码、统计和运行规范。
2. `CPU_ACCEPTANCE_CHECKLIST_zh.md`：逐项验收。
3. `configs/followup_design.json`：机器可读设计；不是旧 R3/R4 的运行配置。
4. `reference/REPOSITORY_AND_EVIDENCE_SNAPSHOT.json`：核实过的仓库与旧证据身份。
5. `reference/reference_math.py`、`reference/test_reference_math.py`：独立数学参考。

## 当前起点

- 工作分支：`codex/ssvc-mechanism-followup-20260914`。
- 2026-09-14 核实时 HEAD：`7e405faa241e0085f701cdc858ba423ea76ba5a8`。
- 该分支已创建，但尚没有已提交的 followup 功能；不要把之前聊天中的草稿当已落地代码。
- 实验实现位于 `ssvc_flow/`，不要从较早的 `main` 重新搭建项目。
- 接续前检查是否有新的提交、未提交修改和适用的 `AGENTS.md`，保留已有工作。

## 必须完成的工作

先核对来源，再实现 S0/S1/S2。S1 包含原 bank0/3/4/5/11 与新 composite；S2 包含 X_BASE、X_VALID、X_VALID_NO_X_OFF。默认算法路径沿用现有直接 PPO 梯度和真实 Adam，不启用未通过的梯度复用/cache 优化。

不要停留在计划、伪代码或空壳 CLI。资源允许时继续完成真实 PyTorch CPU 测试、fake-adapter 端到端、断点恢复与相关旧测试。原始 server 张量缺失时，记录真实阻塞，但继续完成不依赖这些文件的实现和测试。

代码提交到上述工作分支；不合并 main、不强制 push、不发布权重/原始输出/服务器私有资料。不要覆盖旧实验、重新生成旧结果或修改旧 PASS 标记。

## 停止条件

首次需要真实 9B 模型加载、GPU 推理、Slurm 提交或服务器训练时停止。输出：

```text
IMPLEMENTATION_REPORT_zh.md
SERVER_HANDOFF_zh.md
machine_readable_readiness.json
实际代码 diff / commit
CPU 测试命令、退出码、通过/失败/skip 数及日志
```

服务器交接要写可直接运行的最终命令、准确参数和需要的原件；不要只写“后续接入 GPU”。尚未实现或未验证的项明确标记。

## 当前包的验证范围

本包验证的是设计公式、预算计算及选定旧 JSON 元数据。随包参考测试的运行结果在 `reference/REFERENCE_TEST_RESULTS.txt`。它不代表仓库生产代码测试通过，也不代表新实验已经执行。
