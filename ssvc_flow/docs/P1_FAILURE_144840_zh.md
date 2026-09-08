# P1 144840：概率一致性失败定位

## 原始证据

- 日期：2026-09-08；源码 `ba012a3f2c3037fda16e3df5d966db3a0df1e532`。
- 模型：`Qwen/Qwen3.5-9B`，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 环境：Python 3.12.14、torch 2.13.0+cu130、transformers 5.14.1、peft 0.19.1。
- Slurm `144840`，PRO6000，运行 6 分 44 秒，退出 `1:0`。
- 288 条初始样本中 17 条超过平均选中 token log-prob 差 0.02：
  SYMBOLIC_FRESH 11/144（最大 0.0355228346），IMAGE_CUE_FRESH 6/144（最大 0.0343717724）。
- 两接口抽样 cache audit 和图像影响审计通过；在 optimizer update 前停止。
- 原始证据保留于服务器
  `/projects/varunssd/louis-ssvc/runs/slurm-144840-attempt-0/`；
  同名 tar.gz SHA-256：`70dd18daf1738371569255ae76158a8b3aca339401c30aec44d1bee358dbdbe9`。

## 对照诊断

诊断作业 `144893` 与 `144900` 使用相同固定权重、BF16 基座、FP32 LoRA、
seed=17 和 4 条原始误差最大的样本（包含文本与图像），各运行 2 分 45 秒、退出 0。
脚本及日志保留在 `/projects/varunssd/louis-ssvc/` 和其中的 `logs/probe-<jobid>.out`。

`144893`：原路径重生成 token 完全一致，整段 likelihood 与原记录完全一致；
手工逐 token cache 重放与原 behavior 的最大差小于 2.4e-7。
因此差异可复现于 cache 生成与完整序列复算之间。关闭 BF16 reduced-precision
reduction、再单独提升 LM head 到 FP32 仍有超过 0.02 的样本，未采用这些精度改动。

`144900`：采样关闭 cache，打分对每个 token 的相同前缀执行无 cache 前向，
LM head 只计算最后一行；4 条样本的 generation/无梯度 likelihood/有梯度 likelihood
差异均为 0。全部样本有有限、非零的 LoRA 梯度；探针峰值 CUDA 分配 22,493,925,888 字节。
这只是定向诊断通过，不能替代完整 P1 的 352 条样本、更新和恢复验收。

另有独立 CPU 官方 tiny 模型诊断：直接对官方混合 cache 反传会因 recurrent state
被 `copy_` 原位更新触发 autograd 版本错误。因此训练仍不使用带梯度 cache。

## 两项修复

1. **一致的数值策略**：生成和 old/reference/current likelihood 都使用无 cache 前缀重算，
   保持 BF16/FP32 配置、纯 softmax 和原 0.02 报警值；不把 behavior 复制成 old likelihood。
   单独 cache 审计保留，预算增加逐 token 前向上界，运行锁记录执行模式。
2. **聊天结束符**：该 revision 的模型配置 EOS 为 248044（`<|endoftext|>`），
   官方 tokenizer EOS 为 248046（`<|im_end|>`）。原实现只选模型 EOS，
   导致输出数组后继续生成 `<|im_end|>\n`。修复为两套官方 EOS 的并集，
   EOS 仍参与概率计算，仅从最终展示文本去掉终止 EOS。内部特殊文本不删，parser 不放宽。

同时将 sbatch 默认 GPU 修正为此项目 QoS 允许的 PRO6000；新 P1 使用新作业目录，
不覆盖或恢复旧源码产生的失败样本。

## 回归与正式复验

- 修复提交：`73f744b3626f73800eba696ab53424d1e09819d5`，已推送至 `codex/ssvc-flow-ntu`。
- 本地全部测试：205 passed，29.50 秒；Ruff 检查、修改的 Python 文件格式检查、
  `bash -n scripts/ntu_p1.sbatch`、`git diff --check` 通过。
- 独立审查后补充了实际前向调用数与序列数的区分；新增 EOS、数值路径、BF16 文本/图像
  梯度和前向计数回归测试。未修改概率报警值或严格 parser。
- 正式复验作业：`144911`，相同模型 revision、1 张 PRO6000，源码为上述修复提交。
  服务器前置测试也为 205 passed（22.95 秒）。

新作业的 352 条样本均已生成，全部采样/likelihood 平均误差为 0。
其中与原作业相同的初始 288 条校准样本，分类由原来的 I=288 变为
X=164、S=7、W=96、I=21；原始展示文本含 `<|im_end|>` 的样本由 287 变为 0。
新样本是使用修复后数值路径重新生成的，不是对旧答案字符串进行清洗；严格 parser 保持不变。

### 最终 GPU 验收：PASS

- Slurm：`COMPLETED`、退出 `0:0`，2026-09-08 07:59:14–09:12:03 UTC，
  耗时 **1 小时 12 分 49 秒**，节点 `gpu-pro6000-3`。
- `result.txt` 为 `state=PASS`；`model_audit.json` 的七项检查全部为 true：
  采样概率、缓存、图像输入、基座冻结、非空梯度与实际步长、真实更新次数、恢复重放。
- 2 次真实更新耗时分别为 472.564 和 480.647 秒，实际参数步长分别为
  0.0270155 和 0.0263514；两次更新后的经验同批次 KL 为 0.00556049 和 0.00456537。
- 两次恢复重放分别耗时 480.663 和 480.157 秒；参数与连续运行**逐位一致**，
  最大绝对差为 0；优化器状态与随机状态也完全一致。
- 生成共 5,519 token，纯生成用时 735.246 秒（7.506 token/s）；
  此吞吐不包括重算 likelihood、更新或恢复重放。反向阶段峰值 CUDA 分配为
  23,150,378,496 字节（21.56 GiB），不等于显卡的全部保留显存。
- 验证锁标记 `P1_PASSED`、`REAL_CUDA_MODEL`；P3–P9 仍为 `NOT_RUN`。
- 完整证据包大小 316,382,801 字节，服务器重新计算 SHA-256 与归档校验文件一致：
  `f7eb87928ffb906d5d75b84dbf76daf651abb2dd1fb7016dc539dbaf3bd6f97e`。
  本地 `docs/ntu_evidence/P1_144911/` 保存了 20 个验收文件及复制校验记录；
  原始日志、配置和验收副本不随代码推送，Git 只保留此结果摘要。

### 验收后补正：前向调用计数

整理实测证据发现原计数 hook 位于 conditional-generation wrapper；PEFT 部分入口
直接调用 `.forward()`，绕过该 hook。因此本次原始报告中的
`forward_calls_this_invocation=5519` 只计入生成调用，
`post_update_likelihood_forwards=0` 不能代表实际打分次数，**不可用于总前向预算**。
原始文件保持不变；上述耗时、显存、概率、参数和恢复检查均为独立测量。

计数补正提交 `94f0be0` 仅将计数 hook 移到两条入口共同调用的 backbone，未改模型输入、输出、EOS、
概率或梯度逻辑。真实 tiny PEFT 文本/图像回归在旧代码上 2 项失败，在修复后通过；
同时验证 checkpoint 内部重算不重复计入外层前向。最终本地全套 **207 passed**
（29.74 秒），Ruff 和格式检查通过。该计数补正晚于 GPU 验收提交，不能把
`144911` 记作补正后代码的重新 GPU 运行。

下一阶段先审核此 P1 锁与实测预算、核对原有 P0/P2 遗留验收项，再实现 P3 冻结评估；
不将本次校准样本当作正式 dev/confirm 科学结果。

依据：[官方 Qwen3.5 说明](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、
[固定 revision 模型配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)、
[固定 revision tokenizer 配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer_config.json)。
