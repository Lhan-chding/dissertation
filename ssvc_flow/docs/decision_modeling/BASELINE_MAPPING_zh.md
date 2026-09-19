# 多奖励基线与代码映射（2026-09-19 冻结）

实现位于 `src/decision_modeling/reward_recipes.py`。这是独立实现，不复制或安装官方训练框架，不改变当前 Qwen revision、BF16、LoRA、Adam、prefix scoring 或 `B*K*Lnorm64` PPO 损失。

| 方法 | 原始定义与本地实现 | 冻结数值约定 | 状态 |
|---|---|---|---|
| R0–R7 | 文档 5.2 的四通道 `[X,A,V,C]`；先权重求总奖励，再组内归一化 | 继承原 `grpo_update.grouped_advantages`：population variance，分母 `sqrt(var+1e-8)`；零方差优势全 0 | CPU 测试通过；真实 9B 未运行 |
| GDPO_R4 | 各原始通道先组内标准化，乘 R4 优先权 `[2,0,.5,.5]`，求和后整个 B*K completion batch 再标准化 | 选 NVlabs 官方 TRL 实现而非 verl token-mask 加权版本：两阶段 `std(ddof=1)+1e-4`；零方差分量为 0；整批常量为 0 | CPU 与独立 torch 公式逐值比对通过 |
| SAW_R4 | 采用论文 Algorithm 1、§4.3 GRPO mode、§4.4 priority multiplication；B*K batch 算 CV 后乘 R4 优先权，再按源 GRPO 总奖励组内归一化 | 原始非负 `[0,1]` 通道；理论最小值 0，offset 为 `+delta`，CV 分母 `raw_mean+2delta`；`delta=1e-4`；batch std ddof=1；活跃通道 X/V/C 中 `sumCV<delta` 时动态因子回退为 1；A 优先权为 0，不占 CV 预算 | CPU 手算输入输出测试通过；真实 9B 未运行 |

GDPO 的依据为 [论文 v1 §3.1–3.2](https://arxiv.org/html/2601.05242v1) 与 [NVlabs 官方 README 的 TRL GDPO 片段](https://github.com/NVlabs/GDPO#trl-gdpo-implementation)。batch 归一化在全部 completion 上进行，不以 token 长度加权。静态 arm 与 GDPO 的标准差约定不同，属于所选基线定义，并在逐步日志明确记录。

SAW 的依据为 [论文 v1 Algorithm 1 与 §4](https://arxiv.org/html/2606.07705v1)。`delta=1e-4` 是本研究冻结选择；论文未要求本数据必须采用特定 delta。官方 [ray_trainer.py 的 grpo-h](https://github.com/Zhaolutuan/SAW/blob/main/verl/trainer/ppo/ray_trainer.py) 是两种 ToolRL 奖励专用实现，使用 correctness `+3`、epsilon `1e-8`、动态因子乘 2。本轮采用论文通用算法，**不声称与该专用代码逐位相同**。不使用 empirical batch minimum，不把任意方差权重称为 SAW。原通道保持不变；offset 仅用于 CV 统计。SAW 不累计跨步状态。

每一步 `UPDATE.json` 记录 `reward_vectors`、`total_rewards`、`advantages`、组统计、优先权与实际权重；SAW 另记每通道均值/std/CV、delta、fallback；GDPO 另记每通道优势及 batch 统计。四通道逐样本事实也写入 `samples.jsonl`，C 不由 event 常量替代。

## 后续方法 paper→code mapping（本轮不运行）

| 方法 | 论文对象 | 未来代码接点 | 必须新增的冻结/状态 |
|---|---|---|---|
| DVAO | [v1 §3.2 Eq.9](https://arxiv.org/html/2605.25604v1)：每个 group 内 `w_tilde[k]=w[k]*std[k]/sum_l(w[l]*std[l])`；对各通道标准化优势加权求和 | `compute_advantages` 已有 `[B,K,4]` 原始通道和 group statistics，可加独立分支；方差是 **group** 的，不是 SAW 的 batch CV；同一 PPO adapter 接收产生的优势 | 先冻结优先权、std correction、epsilon、全常量回退、是否包含论文外 batch norm；写零方差与相关奖励测试；目前无 DVAO 可执行 recipe |
| Constrained GRPO | [v1 §4](https://arxiv.org/html/2602.05863v1)：主 reward 与 indicator constraint cost 各自组内标准化，然后以可更新的非负 Lagrange multipliers 合成优势 | reward assembly 增加明确的 indicator cost；`block_runner` 的完整 state 扩展 multiplier/dual optimizer；日志保存 violation rate、阈值与 multiplier。候选评价群体非劣界不能直接当成训练 cost | 先冻结行为约束、允许违反率、dual 更新、初值和范围；连续关系比例 C 不能未经说明直接充当 indicator cost。后续需与目标/约束相同的静态最佳基线比较；目前无 CGRPO 可执行 recipe |

CGRPO 在这里专指 **Constrained Group Relative Policy Optimization (2602.05863)**，不与 Consensus C-GRPO 混用。上述映射不是已经实现的在线控制器或实验结果。

## 训练与恢复边界

`run_from_runtime(runtime, origin, recipe, steps, out, train_prompts=..., origin_id=...)` 是真实 adapter 路径。origin 必须是服务器原始完整 checkpoint 的绑定，不接受 CSV 重建。源 schedule 使用原 seed 和元数据确定，逐分支验证 position/hash。每个 arm 从同一完整 origin 开始，自己每一步从 live policy 采样 B4/K8。

`steps=8` 保存 H0/H8 后返回；同一路径 `resume=True, steps=32` 继续全部 arm。H8 不是新训练起点、不会重计 8 步。每一步均保存必要恢复 checkpoint，H0/H8/H32 收据引用它；失败 attempt 保留，不覆盖已提交步骤。分支开始比较恢复后的 LoRA、Adam、所有 RNG、sampler、buffers、modes 与位置状态；step 内只做 tensor-version 和有限性检查。没有逐样本/逐 step 对 BF16 全基座进行哈希。

默认评估通过独立 observe 命令；可选 evaluation callback 在完整状态保护内执行并还原全部训练 RNG。源 B4/K8、Lnorm64、epsilon1e-4、Adam 超参数由外层 runtime/config 校验；本模块不会升级框架或替换原模型。
