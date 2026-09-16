# V4 完整信息概率响应建模

本轮使用独立分支 `codex/ssvc-modeling-v4-full-response`，保留 V3 工作区与原件。
设计输入为用户提供的 `SSVC_Modeling_V4_20260916.zip`，本地原文保存在
`docs/modeling_v4/design/`。原文、私有服务器绑定及实验原件不随代码提交。

## 科学范围

- A：服务器短 CPU 诊断，复用保存的原点与实际 Adam；补充缺失候选，不重训旧轨迹。
- B：两个历史原点、两张独立 PRO 6000 的真实 9B 技术桥接。
- C：development seed 41001/41002 的 step32/96，共四个真实原点。
- D：开发选择后扩展到登记的多种子矩阵，独立 calibration/test 分开记录。
- E：有可分辨点态响应时进行离线时间跟踪；在线 SSVC 不在本轮范围。

CPU 方法准确率不是 B 的前置条件。旧 Q3 百万拟合不参与本轮启动门禁。
最多三张 PRO 6000；默认两个独立单卡 worker，各自使用 Slurm 内的 `cuda:0`。
`--execute-gpu` 是本 campaign 的执行开关。

## 接口

在 `ssvc_flow/` 运行。输入绑定 JSON 包含父计划、历史 checkpoint 的原始身份和
校验值、冻结的评分容差。输入绑定只登记真实存在的服务器路径。

```bash
python -m src.modeling_v4.cli plan \
  --config configs/modeling_v4/protocol.json --out runs/modeling_v4/plan
python -m src.modeling_v4.cli cpu-diagnose \
  --config configs/modeling_v4/protocol.json \
  --reuse /absolute/path/cpu_reuse.json --out /absolute/path/campaign/cpu
python -m src.modeling_v4.cli build-task-list \
  --config configs/modeling_v4/protocol.json --bindings /absolute/path/server_bindings.json \
  --workers 2 --phase bridge-and-first-map --out /absolute/path/tasks.json
python scripts/submit_modeling_v4.py \
  --config configs/modeling_v4/protocol.json --campaign-root /absolute/path/campaign \
  --tasks /absolute/path/tasks.json --reuse /absolute/path/cpu_reuse.json \
  --workers 2 --stage both --print-only
```

实际提交使用 `--submit --execute-gpu`；恢复使用同一任务表并加 `--resume`。
CPU 真实诊断必须在服务器 CPU Slurm 作业中运行。完整的 sbatch 命令由提交器打印，
默认打印不提交。任务表中的阶段决定运行内容，提交器 `--phase` 只给日志命名。

每个完成的真实响应图可在服务器 CPU 上拟合与评价：

```bash
python -m src.modeling_v4.cli fit-response-map \
  --tasks /absolute/path/tasks.json --task-id map_41001_X_BASE_32 \
  --out /absolute/path/campaign/evaluations/map_41001_X_BASE_32
python -m src.modeling_v4.cli summarize \
  --root /absolute/path/campaign --out /absolute/path/campaign/report
```

大规模分析通过 `scripts/modeling_v4_analysis.sbatch` 的 CPU allocation 执行。
它接受 `PYTHON PROJECT_ROOT OUTPUT_ROOT ANALYSIS_COMMAND [ARGS]`，清空 CUDA
设备环境，并优先把临时参数矩阵放到节点的 `SLURM_TMPDIR`。以下接口也只允许
服务器 CPU 执行，测试夹具不通过生产 CLI 暴露：

- `export-calibration-labels`：只导出开发原点的 calibration bank 工作观测。
- `fit-signature-development`：六个开发原点，按完整 seed 留出；没有 query/reference 输入。
- `freeze-signature-models`：根据已完成 CV 选择，在六个开发原点上重拟合并保存权重。
- `predict-signature-models`：只接收冻结权重和目标原点的实际签名。

经验 Fisher 诊断复用真实逐样本 AD 梯度，计算它与完整更新方向的内积，不建立
参数维数平方的 Fisher 矩阵。方向临时矩阵需要 `8 × 参数数 × 方向数` 字节；
GPU worker 在有 `SLURM_TMPDIR` 时将其用于 `SSVC_V4_SCRATCH_DIR`，否则使用
实际 `TMPDIR`，并在写入前检查文件系统与祖先 Ceph quota。节点磁盘带宽与 CPU
内积耗时计入实测，不能只按桥接的小样本耗时推定完整 D 的速度。

后续阶段使用 `phase-evidence` 核验真实采集、拟合、评价原件，再通过
`extend-task-list` 追加任务。开发模型选择使用 `freeze-selection`，固定 m、
bank 选择器、两个主要模型和梯度参照；确认阶段使用协议的主工作样本数 1024。
跨 seed calibration 只用于误差规则，不能并入签名网络训练。

## 模型与数据语义

`fit` / `predict` / `compare-kernels` 提供数组级接口。完整参数核使用**绝对 ridge
惩罚 lambda**，不会隐式乘 Gram 最大特征值。FULL 保留完整输入；维数阈值用于诊断。
`r=k` 标记为 full-equivalent，只有 `0<r<k` 标记为压缩。

有效权重核使用 LoRA 低秩乘积计算内积，不构造完整权重矩阵。完整 RBF 与旧投影
RBF 分别命名。签名模型保留 576 个特征，网络的三个初始化不增加训练种子数。

开发曲线通过 `fit-response-map --calibration-banks 24|32|48|96
--bank-selector STRATIFIED_RANDOM|BLOCK_PIVOT_QR --draw-count 256|1024|4096`
指定。两个选择器只使用训练 metadata 和完整更新 Gram；同一 bank 的全部 contrast
一起进入或离开校准集。不同 m 共用嵌套顺序，不同 n 共用已登记大样本的前缀。
这些前缀不是独立实验重复。独立测试不能用命令行覆盖已冻结设置。
开发阶段从实际取得的 96 个 bank 中选择 m；确认阶段按方案只取得已冻结的 m 个
bank，因此全部进入拟合。确认结果不声称是在未生成的 96 个 bank 中再次做 QR 选择。

原始样本共享；工作样本和参考样本使用独立 RNG。候选只保存一次 trainable 状态，
完整 Adam/RNG 保存在原点、可恢复 latest 和选定桥接候选。相同前向策略可以共用评分，
不同 Adam 状态不能因此合并。

完整方向导数与有限差分分列。推理评分快路径的认证不能认证 autograd 路径。
BF16 前向数值误差与 FP64 累计分别记录。

`configs/modeling_v4/analysis_rules.json` 在新 V4 实测前登记参考精度与重叠诊断。
三个概率尺度均报告：参考的经验正态近似半宽 `1.96 × SE` 不超过尺度的一半时，
才在该尺度标记为可解析；主表的可解析子集采用 0.005 尺度。全部有限参考值的实测
残差另报，不能把不可解析样本从总分母删除。该标记不提供覆盖保证。
若任一端点的重要性权重 ESS/n 小于 0.1，或最大归一化权重大于 0.1，ORIGIN
参考触发预先登记的独立 MIX 测量；该诊断不修改原始权重，也不是模型精度认证。

`point-response-evidence` 从冻结模型的独立 calibration query 结果生成离线 E
执行依据，再核验实际文件内容。执行规则在 `analysis_rules.json` 中单独登记；
没有观测数据记为等待，已测但不满足规则记为未解析。该规则不是在线控制认证。

## 验证与报告

小型测试验证实现契约。`scripts/modeling_v4_validation.sbatch` 在最终源码版本上
执行一次服务器全仓回归，不能替代真实 9B 实测。

报告包括 `MODELING_DECISION_V4_zh.md`、`CORE_RESULTS.csv`、
`ERROR_DECOMPOSITION.parquet`、`GPU_BRIDGE_RESULT.md` 和 `RUN_SUMMARY.json`。
汇总器只读取已完成的派生评估，不重复扫描原始模型权重；没有真实结果时明确记录缺失。
技术桥接完成、预测文件完成、CPU 小测试通过都不等于模型可靠或在线 SSVC 有效。
