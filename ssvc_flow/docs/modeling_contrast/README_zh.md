# 候选差异建模 V2：CPU 实现与复现

主协议原件在 `design/START_HERE_FOR_CODEX.md`；协议 JSON 不改写。
N0–N3 原始决定及验收位于 `results/MODELING_DECISION_V2_zh.md` 和
`results/CPU_ACCEPTANCE_RESULTS_zh.md`，保留原件及哈希。
服务器 N4 后续决定位于 `results/server_N4/MODELING_DECISION_V2_zh.md`。
旧模块 `src/modeling_qualification/` 保持不变。

## 执行位置

按用户要求，较大的 CPU 模型实验在已授权的服务器执行；本机用于代码修改、文件核验和轻量单元测试。
本轮服务器执行修订、原始锁保留方式和资源门禁见 `SERVER_N4_EXECUTION_zh.md`。
`scripts/modeling_contrast_server.sbatch` 分别执行预检及冻结收集/验证；
完成后由 `scripts/modeling_contrast_posthoc.sbatch` 在 CPU 节点独立复算并汇总。
后续实验仍需遵守科学选择、种子身份及资源门禁，不因本轮授权而自动启动新的实验。

## 环境和输入

在 `ssvc_flow/` 下使用含 NumPy、SciPy、PyTorch、pytest 的现有 Python 环境。
程序自动限制一个数值 CPU 线程、屏蔽 CUDA/HIP 并设置 Hugging Face 离线。
`--parent` 指向完整旧 M2 目录，须包含原 manifest、dataset、probe metadata 和逐轨迹原件。
未给 `--parent` 时使用相邻的 `ssvc-modeling-qualification-20260915` worktree。
旧原件不在 Git 代码交付中，准确路径和 SHA-256 见交付 coverage。

## 已实现命令

```bash
python -m pytest tests/modeling_contrast -q
python -m src.modeling_contrast.cli audit-parent --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N0 --parent /absolute/path/to/M2
python -m src.modeling_contrast.cli smoke --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/smoke --parent /absolute/path/to/M2
python -m src.modeling_contrast.cli observation-study --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N1 --parent /absolute/path/to/M2
python -m src.modeling_contrast.cli fit-study --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N2 --parent /absolute/path/to/M2
python -m src.modeling_contrast.recompute --run-root runs/modeling_contrast_v2 --parent-root /absolute/path/to/M2 --out runs/modeling_contrast_v2/recompute
python -m src.modeling_contrast.cli freeze-selection --config configs/modeling_contrast/protocol.json --out runs/modeling_contrast_v2/N3 --parent /absolute/path/to/M2
python -m src.modeling_contrast.cli report --config configs/modeling_contrast/protocol.json --out docs/modeling_contrast/results --parent /absolute/path/to/M2
```

输出不可覆盖。`--resume` 只验证已完成阶段的身份和产物；中断阶段保留原件并拒绝盲重跑。
源码改变后不能用旧身份继续。阶段源码快照与最终选择锁分别记录真实执行版本。
新 N4 必须由完整 N3 科学、成本、原件及包含新增测量的资源门禁授权；
本页不给绕过门禁的训练命令。原始新轨迹收集完成与冻结独立验证完成是不同状态。

## 原件和诊断边界

- 四事件固定为 X/S/W/I，216 维 Helmert 响应不等于动力学秩。
- 新观测保留付费样本、标签、标量 logp 和 proposal 身份；无损压缩后可逐位恢复。
- 原 O-IND 五份旧计数在 fit bank 的跨 bank alias 协方差保留；evaluation 使用独立新观测。
- C1 辅助计数的已知跨 bank 复用问题由独立、哈希绑定的辅助包修正，N1 原始包不覆盖。
- raw 预测先封存再评分。投影只作后验诊断；使用精确 baseline 的诊断不能进入有限观测排名。
- 已知 X/I 动作评分和完整枚举保留；操作数情景不能冒充真实 VLM 时间或成本。
- 全仓库旧测试缺失的历史 fixture 单独列明。新模块通过不意味着旧全套回归或科学门禁通过。
