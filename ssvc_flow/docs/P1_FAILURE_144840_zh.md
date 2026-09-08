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

依据：[官方 Qwen3.5 说明](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、
[固定 revision 模型配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json)、
[固定 revision tokenizer 配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer_config.json)。
