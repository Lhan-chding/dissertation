# Q4 工作量与计时依据

日期：2026-09-16。第1—6节按清单所绑定的源码快照与默认配置清点逻辑请求；不是新 V3 GPU 实测，也不是首次提交授权。准确启动绑定仍待服务器工程验收，不提供猜测的可执行路径。机器可读清单及所读文件 SHA-256 见 `Q4_WORKLOAD_ACCOUNTING.json`。

## 1. 两个 worker 的准确清点

worker 0 使用 historical_initial，worker 1 使用 historical_X_BASE_64；两者工作量相同，均从历史完整状态产生 scratch 分支，Q4 不提交源训练更新。下表按一次成功的新运行计数，失败、重试与恢复新增调用另计。

| 项目 | 单 worker | 两 worker 合计 | 推导 |
|---|---:|---:|---|
| 自然 bank | 6 | 12 | 固定取前六个 calibration bank |
| bank 生成序列 | 192 | 384 | 6×4题×8输出 |
| bank 梯度前 parity 重评分 | 192 | 384 | `_groups` 每条已生成动作重评分一次 |
| scratch Adam | 18 | 36 | 每 bank 三候选；显式零梯度仍更新 |
| 梯度评分及 sequence backward | 576 | 1,152 | 18×32；不是 token backward 次数 |
| ORIGIN pilot 生成 | 192 | 384 | 12题×16 |
| ORIGIN main 生成 | 576 | 1,152 | 12题×48 |
| ORIGIN reference 生成 | 768 | 1,536 | 12题×64 |
| MIX reference 生成 | 768 | 1,536 | 首 bank 两端点的一个 IID MIX stream；没有翻倍 |
| DIRECT reference 生成 | 1,536 | 3,072 | 首 bank 两端点，各12题×64 |
| 全部观测生成 | 3,840 | 7,680 | 上述五类相加 |
| 观测生成后 certified rescore | 3,840 | 7,680 | `PrefixObservationBackend.generate` 每次生成另调一次 `adapter.logprobs`，实际计费 |
| ORIGIN 评分任务，含 origin 自评分 | 29,184 | 58,368 | (16+48+64)×12题×19策略 |
| MIX 两端点评分任务 | 1,536 | 3,072 | 64×12×2 |
| 全部显式评分任务（alias 前） | 30,720 | 61,440 | 19策略=origin+6×3候选；DIRECT不再建 score task |
| 仅18候选端点评分任务（alias 前） | 29,184 | 58,368 | 1,536×18+768×2；与上一 ORIGIN 行数值巧合相同 |
| origin 自评分任务 | 1,536 | 3,072 | 已计入全部显式评分任务 |
| null 原始序列评分 | 12 | 24 | 每题一条实际旧动作；每次未完成运行的 invocation 都重新测量 |
| known-action 评分上界 | 228E | 456E | 19策略×12题×实际 EOS 集合大小 E |

known-action 实际数量取决于真实 tokenizer 输出及 EOS 集合：只有长度≤63且中间不含 EOS 的已知 JSON 路径才追加各 EOS。所读计时文件没有 E，不能擅自设为2。它只是一个已知正确字符串的有限动作子集，不是完整 X 事件。已有完整且通过 hash/request/policy 验证的 known-action 原件可在恢复时复用。

每 worker 共生成 4,032 条序列；评分请求含梯度评分共 35,340+K 次，其中 K≤228E。将一次生成和一次完整序列评分分别计作一个“序列遍历”，总计 39,372+K。这是便于核算的逻辑工作量，不能把它误作模型 forward 次数。

## 2. 复用与 token 级计算

`execute_observation_tasks` 会复用生成时已测的本策略 certified logp：每 worker 的 ORIGIN 自评分1,536条和 MIX 已选生成端点768条至少有2,304个可复用请求。在无其他策略 alias、无相同题目/token重复的条件下，30,720个显式评分请求对应28,416次新增序列评分；实际可进一步下降。所有请求和缓存命中仍须分别登记，生成抽样本身仍逐条执行。

`model_adapters/base.py::logprobs` 对每个有效 token 重新前向计算完整 prefix；生成配置也为 `use_cache=False`。因此“评分一条64-token输出”不是“一次 forward”。梯度评分另外触发 backward，梯度 checkpoint 重算、视觉输入、prefix长度、状态复制和I/O均会影响时间。bank 生成走 `collect_samples`，其生成后检查由 `_groups` 的192次重评分单列；不能把观测 wrapper 的3,840次 certified rescore再次套到 bank 生成上。

## 3. 历史计时实际记录

- 旧 PRO6000 smoke：4588.105149秒（1.2745小时），scope 是 warm bank03 old/new joint bridge。
- 旧 S1 bank03 completion invocation：14074.849609秒、40,124个记录 forward；失败尝试不在这个 completion elapsed 中。
- 两数相除得 0.350783810秒/记录forward。这是旧混合工作负载的比值，包含未分离的 backward、状态操作等；不是新 V3 生成或评分吞吐，也不是保守上界或乐观下界。
- 旧机器记录：NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition；实际总显存101,971,460,096 bytes，smoke peak allocated24,173,737,984 bytes。这些值不能代替新分配的硬件与显存实测。

## 4. Q4 量级情景：仅作敏感性展示

下面假设所有生成/评分/null序列共有平均有效长度 L，暂不抵扣 alias，不计 known-action；再**假设**沿用上述旧混合比值。计算为 `39,372 × L × 0.350783810 / 3600`，准确系数保存在 JSON。所有数值都是假设结果，不能引用为 V3 实测时长、置信区间或工期上下界。

| 假设 L | 单 worker 情景小时 | 两 worker GPU小时合计 | 两卡同时启动且等速的理想墙钟小时 |
|---:|---:|---:|---:|
| 8 | 30.69 | 61.38 | 30.69 |
| 16 | 61.38 | 122.76 | 61.38 |
| 32 | 122.76 | 245.53 | 122.76 |
| 64 | 245.53 | 491.06 | 245.53 |

两卡各负责一个不同原点，两个 worker 都完成后才做跨卡合并；理想墙钟不再除以2。模型加载、证书检查、实际别名、EOS长度、known-action、存储操作和失败重试均未被该混合比例独立标定。Q4 的候选实际策略可因相同参数而大量复用，所以这些情景既不保证快于真实，也不保证慢于真实。

## 5. Q5 默认规模的时间风险

Q5 默认24条源轨迹、3,072次源 Adam、30个响应原点、3,240次scratch Adam。当前 `prepare_q5_observation` 默认启用 DIRECT：每原点36个heldout候选策略各36题×1,024次独立输出，单 DIRECT 项就是1,327,104条生成。

| Q5 30原点初始方案项目 | 数量 |
|---|---:|
| 观测生成（主 ORIGIN、DIRECT、初始 reference ORIGIN 与两 bank MIX） | 54,190,080 |
| 加上源训练与 bank 的全部生成 | 54,322,944 |
| 观测生成自动 certified rescore | 54,190,080 |
| 显式评分请求（alias 前，含 origin 自评分） | 297,492,480 |
| 全部梯度前 parity 重评分 | 132,864 |
| 梯度评分/sequence backward | 201,984 |

上述仅到初始 reference n4,096，不含扩至65,536、额外强制MIX、Q4/Q6或失败重试。估计器的拒判也不能省略已冻结的独立参考任务。当前 `estimate_workload` 明确排除 DIRECT studies，且候选评分数只统计108候选；实际 measurement manifest还含origin自评分，不能直接用其数字作为默认完整CLI成本。

为展示54,190,080条观测生成本身的数量级，下表任意假设一次完整 `PrefixObservationBackend.generate`（含certified rescore）耗时τ；不使用旧 mixed ratio。这里只是这一组件的代数情景，不是完整工期，也不是新 V3 速度测量或上下界。

| 假设 τ（秒/观测生成） | 该组件 GPU天 | 两GPU完全均衡的组件墙钟天 |
|---:|---:|---:|
| 0.1 | 62.72 | 31.36 |
| 1 | 627.20 | 313.60 |
| 10 | 6272.00 | 3136.00 |

因此必须先拿到真实Q4生成长度、逐类生成/评分/Adam时间、alias比例与存储吞吐，再逐阶段报告Q5开销；不得将长期风险变成静默删减bank、seed、DIRECT或参考精度。阶段依赖还会减少两卡同时有可执行任务的时间。

## 6. 复核入口与交接状态

- `vlm_campaign.build_q4_bridge_plan`：12个固定probe、各一条历史null动作、两个历史原点。
- `vlm_campaign.vlm_smoke`：六bank、六类生成任务（DIRECT两个端点分别一类）、ORIGIN19策略与MIX两端点评分。
- `vlm_campaign.run_candidate_banks` / `_groups`、`followup_updates.direct_loss_gradients`：bank、parity、Adam与backward。
- `vlm_observation.PrefixObservationBackend.generate` / `execute_observation_tasks`：计费rescore与物理缓存。
- `vlm_campaign._q5_generation_tasks` / `prepare_q5_observation`：Q5默认DIRECT和初始独立参考。
- `model_adapters/base.py::generate/logprobs`：无cache生成、逐prefix评分。

最初工作量清点只读源码及本地留存计时。本次第7节额外通过SSH只读旧bank03有限原件；未调用GPU、未运行大型CPU实验、未提交任务。清单没有确认新的GPU执行授权；启动绑定与V3实测吞吐仍由主任务核验。


## 7. 2026-09-16 追加：旧 bank03 原始长度核验

本次通过 `eee-cluster` 只读 `/projects/varunssd/louis-ssvc/runs/MECHANISM_20260914_154599c/S1/bank03`。先检查字段，再读取三份 `candidates/{joint_0,joint_1,joint_1em2}/direct/samples.jsonl`。没有读取模型张量或 confirm 数据，也未提交作业。新证据文件 `HISTORICAL_Q4_TIMING_LENGTHS.json` 记录20个所用文件的路径、字节数和SHA-256：18个payload逐一匹配旧bank manifest，另核验manifest与completion的绑定。旧manifest SHA-256为 `3acaaa771cd7d4315c3e7cbbede135010abaf02ae1f164538203a13875cd79c6`。

### 7.1 可直接核实的历史生成记录

每个独立策略768条序列、48题；三策略共2,304个唯一sample key，与原汇总的unique rollout数一致。同内容的顶层账本副本只计一次；`no_x_off_1` 的精确策略别名增加0条独立生成。

| 历史范围 | 序列数 | 平均token长度 | 最小—最大 | 中位数 | P95 / P99 | 原逐序列elapsed均值（秒） |
|---|---:|---:|---:|---:|---:|---:|
| 全部三策略 | 2,304 | 15.309896 | 12—19 | 16 | 17 / 17 | 2.488219 |
| SYMBOLIC_FRESH | 1,152 | 16.060764 | 13—19 | 16 | 17 / 17 | 2.104794 |
| IMAGE_CUE_FRESH | 1,152 | 14.559028 | 12—17 | 14 | 17 / 17 | 2.871643 |

全部样本的 `max_new_tokens=64`、`stop_reason=eos`；逐条核验 `completion_length=len(token_ids)=runtime_forward_calls`，行为logprob长度也相等。实际生成token及生成forward合计35,274。原逐序列elapsed合计5,732.855892秒，单条范围1.672777—4.877717秒。图像题每条384个image token，prompt长度559—585；符号题prompt长度163—189。以上是这批旧生成的实测字段，**没有包含新V3观测wrapper额外certified rescore，也不是新V3吞吐**。

### 7.2 与旧混合计数的算术核对

旧bank包含十个候选更新原件（含一次baseline replay及未作direct生成的候选）。每个audit均记录B4×K8=32条梯度序列、485个token、32次sequence backward及1次Adam；其candidate elapsed为680.430050—691.731355秒，合计6,848.964202秒。这里没有把十次候选当作十个独立训练seed。

`35,274 + 10×485 = 40,124` 与旧completion profile forward计数数值一致。但候选audit的485是token计数，不是单独计时的forward计数；不能由该等式推出生成、评分、backward和状态操作具有相同单价。completion elapsed减原生成elapsed与候选elapsed合计，算术余项为1,493.029515秒；不同记录的计时边界未经独立核对，因此该余项不命名为纯I/O或模型加载时间。

### 7.3 用真实旧长度替换任意长度假设

仅在额外假设“新Q4所有生成、评分、梯度及null也具有同一个旧长度统计值，并继续采用旧混合0.350783810秒/记录forward比值”时，第4节公式得到：

| 作为假设代入的旧长度统计 | 单worker情景小时 | 两卡同起且等速的理想墙钟小时 |
|---|---:|---:|
| 旧均值15.309896 | 58.73 | 58.73 |
| 旧中位数16 | 61.38 | 61.38 |
| 旧P95 17 | 65.22 | 65.22 |
| 旧最大值19 | 72.89 | 72.89 |

这些是**历史长度支持的敏感性情景，全部仍非新V3实测或工期上下界**；P95那一行也不是95%工期保证。旧样本只覆盖step64之后三个候选策略的48题控制面板，不能证明新Q4初始策略、两个来源、bank题或known-action的长度分布相同。以上公式仍未扣除alias或重复token动作缓存、未计known-action、队列、加载与重试，也没有分别标定梯度和纯评分。两卡总GPU小时是表内墙钟的两倍。

因此可将“典型长度约15—17”作为有原件依据的历史条件，而不能据此报告新Q4准确吞吐。首次新GPU运行仍须记录每类真实生成长度、certified rescore、纯评分、Adam和状态/存储耗时，再更新后续阶段估计。
