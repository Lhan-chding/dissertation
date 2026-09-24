# 真实训练执行路径核对（2026-09-24）

新增执行器复用 `decision_modeling.block_runner.update_batch`，仅增加默认关闭的显式 `advantage_callback`。R0/R4/GDPO_R4/SAW_R4 的奖励、优势、PPO loss、参数和 Adam 状态在同一 CPU Torch bank 上逐值一致；DIRECT_REPAIR_R4 通过命名回调记录第五通道 Bdamage 和独立总奖励，不修改旧 R4。

本次在线重新核读 [NVlabs 官方 TRL GDPO 实现](https://github.com/NVlabs/GDPO#trl-gdpo-implementation)：各通道组内 `std + 1e-4`，乘静态优先权相加，再 completion batch 内 `std + 1e-4` 标准化；PyTorch 默认 correction=1。与继承实现一致。

重新核读 [SAW v1 Algorithm 1 与第4.4节](https://arxiv.org/html/2606.07705v1)：使用已知理论最小值构造偏移，在 batch 上计算 CV，按 CV 比例分配动态权重，最后乘静态优先权；GRPO reward mode 无额外通道数倍乘。本实验保留旧冻结参数 delta=1e-4、ddof=1 和 X/V/C 活跃通道；不等同于 ToolRL 专用代码。

每分支先恢复完整参数、Adam moments/counters、Python/NumPy/Torch/CUDA RNG、buffer、module mode、position state，再明确替换 continuation sampler，并保留 parent source sampler。共同题序与训练采样种子均不依赖 recipe；评估种子独立绑定 role/policy。

生成使用既有 uncached_prefix_recompute；不重复为每个回答做全序列重评分。GPU smoke 对前四个输入独立比较 behavior token log-probability 与原 scorer，正式 rollout 仍保留 behavior score，EOS 被继承校验器验证。CPU fixture 不能替代真实 GPU smoke。

训练每八步形成一个原子提交，source 另保存0/24/32/88/96实际经过节点，branch保存0/8/32。未提交 attempt 永久保留且不参与结果计数；恢复最多重放八步。评估每题独立提交，重启只重跑未提交题。所有原始文本、token IDs、语义标签、优势、更新审计和完整恢复状态保留。

开发 H8 诊断在首次实验前固定：从 D 按 family 取排序最前的4个 base_scene，每个保留两接口，共24 prompt，每题8次。三家族分别报告，此诊断不参与动作选择、超参数或主 H32 标签。最终 T_H8 使用数据准备阶段登记的24 prompt，每题8次。完整评估前后还原训练状态；CPU 回归验证启用/不启用 H8 诊断时后续训练终态完全一致。
