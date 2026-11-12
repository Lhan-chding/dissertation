# 实施接口与测试矩阵

本文是 Codex 的工程任务书。下文“需实现”的模块不在当前包中，不能把其命令当成已经可运行的服务器程序。

## A. 工作目录与身份

历史服务器根目录（待在线核验）：
`/projects/_ssd/varunssd/louis-ssvc/sr_ord_train_20261103/package_v2`。

历史native源：`/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06`。
新建议目录：`/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112`。

新目录不存在时创建；若已存在，先读它的experiment_id、freeze和状态，不得覆盖。旧adapter、旧loss、旧权重、旧评估结果只读。新科学commit与新执行backend hash独立记录，不能冒用旧科学commit。

需要在新命名空间实现：

| 模块 | 输入 | 必须产出 | 不可做的事 |
|---|---|---|---|
| `registration` | 本配置、旧绑定、新代码 | 新身份、SHA与有限请求全集 | 不得扩预算 |
| `data_adapter` | frozen training slots、原8类生成器 | tasks、inputs、images、两个oracle报告 | 不得按模型结果选题 |
| `fork_loader` | 已验证R24 state、原parent | fresh step0 policy/optimizer/ref | 不得继承旧Adam或把R24作ref |
| `diagnostics` | 原生loss、DRY原tokens | PG/KL、split-half、影子更新证据 | 不得提交正式参数 |
| `training_driver` | 新登记、门禁回执 | 原子更新与checkpoint链 | 不得跳过门禁或多次step |
| `evaluation_adapter` | 锁定endpoint、冻结manifest | raw、trace、字段评分 | 不得加载latest代替指定step |
| `analysis_and_pack` | 完整raw/评分/manifest | root配对表、CI、报告、SHA | 不得挑成功子集 |

只修改调度、身份适配和观测层；原生科学计算函数不修改。旧 `runtime_contract.check_path` 规定父等于起点且只有R/G/D，**不能直接拿来验证本轮**。保留其数值检查，另写支持 `start_policy_sha != reference_sha`、T0/T1、0/2/8/16/24 的新wrapper。

## B. 必须先测试的夹具

### B1 数据与oracle（零模型调用）

- 每个结构、direction、aggregate的正常例；8类别训练、10类别prefix分别验证，不把10类送给只支持8类的J。
- 图和清单两者一致、仅顺序不同但选集仍一致、真正选集冲突三类。
- 图位置交换必须同时交换类别名及Alpha/Beta绑定；只换清单不能改变金答案。
- 所有合法边界候选集合的Beta聚合互异。
- 两臂只改唯一清单数组；T1一致槽input/image SHA与T0相同。
- 每步12冲突4一致，每细分cell18/6，总384/root/路径；另种子root不同。
- private gold不泄漏；去名字world-query hash跨split检查；请求ID不能重复。

### B2 评分（零模型调用）

复用pinned native scorer，正反例必须包含：

1. 正确选集+正确答案+完整真实证据：J=1。
2. 选集顺序反转但集合相同：S仍为1。
3. 列举顺序反转但事实相同：不能仅因顺序扣J。
4. 正确选集、证据不足：J=0，S可为1。
5. 错误选集/绑定、正确金答案且完整证据：**原生J可能为1**；必须保留其真实定义，不把测试“修”成J=0。
6. 重复JSON键、重复类别、不合法绑定、额外字段、NaN/Infinity、缺外括号、尾随文本、length截断。
7. 只有selected字段可读但其他字段协议不合法：字段诊断与完整J分开。
8. source四分类中“other_admissible”不等于chart；unknown不等于list；一致题记same。
9. authority/list与prefix/scale不调用不支持的native J，记NA。
10. prefix限定补括号只追加一个字符，且仅EOS；不能修正length或错误内容。

本地 `runtime_contract.native_fixtures()` 已跑过，包括第5项；未来服务器必须以同一SHA再跑。

### B3 数值与梯度（CPU能做代数测试，GPU另验）

- 穷举256个二值G8：优势均值0，非零方差时平方和8，|a|max≤sqrt(7)，全0/全1优势0。
- 同一奖励组中J方向与source反向的合成例，必须得到负d，不能因为平方或绝对值而误判正向。
- 0优势组仍保留batch分母和KL。
- J+lambda*S在全J0组经标准化后可能不随lambda缩小；本轮不训练此奖励，也不以“lambda小所以必然安全”作为论证。
- 原token->logprob对齐、prefill只作条件、不进入completion loss；EOS/停止token处理沿用原生产协议。

## C. 梯度探针的精确实现要求

### C1 复用什么

历史审计定位到的函数（必须在线查SHA与实际实现，不按文档猜行号）：

- `adapter/training.py::native_j_advantages`、`validate_records`、`update`。
- pinned `sr_f6.objective::sequence_loss`。
- `gpu_production_v2.backend.NativeBackend` 及无损saved-tensor执行层。
- `diagnostics/gpu_capacity_full_batch_v2.py::probe_update` 仅可作为参考；原hook抓的是裁剪后梯度，不能冒充裁剪前值。
- 原transaction恢复与state加载函数；新分叉不能调用“恢复旧路径训练”而无意继承Adam。

### C2 分量与方向

令g_J为J policy loss的梯度，g_K为未乘系数的KL梯度，则总loss梯度为g_J+0.02g_K。另计算source诊断loss梯度g_S。方向比较用g_J、g_S，不掺共同KL。

`old=current.detach()`导致ratio数值为1，但其导数不为0；不得将ratio替换成Python常数1，否则PG消失。也不能用sampler作为old而改变clipping语义。

TIS=`min(2,exp(current.detach()-sampler_logprob))`，保留detach与上截断，不能用重新计算的概率替代原sampler。每回答token均值，128回答等权。累计顺序固定。

对同一批次存储完整g_J、g_S、g_K、g_total的496个张量摘要/可复算文件；记录pre-clip和post-clip。用fresh AdamW、相同lr/betas/eps/裁剪，在隔离张量副本上得到真实影子参数差delta_theta，不用“负梯度就是更新”替代Adam。

数值验收要求：同backend重放生产J必须逐张量 `torch.equal`；分量重组因独立累加路径可能有FP32舍入，需另报告相对L2误差，默认上限1e-5，且不得改变生产训练的bit-exact主重放要求。若同backend重复都不稳定，先修determinism，不允许只放宽阈值。

### C3 held-out目标方向（不是已完成的连续终点）

在预定的16个独立witness root，每root8个来自当前R24策略的回答上，定义c=精确授权选集（unknown为0），组内leave-one-out基线b_-i。目标梯度估计：

`h = mean_i [(c_i - b_-i) * grad sum_t log pi(y_it | x, y_i,<t)]`。

这估计授权选择概率的局部梯度；其完成序列logprob用sum，不是生产目标的token mean。无需新生成或伪造未采样token概率，但需要新的teacher-forward/backward。

两个预先固定的witness半集各求h_half，报告 `h_half dot delta_theta`。该一阶估计有采样噪声，不能叫“真实概率提高了Xpp”。只把正方向且高于数值误差底线作为本轮工程准入。底线取同backend重复计算内积差绝对值最大值，不是用split-half的效应差来临时调门槛。

J/source各自split-half余弦只检查采样稳定性，>0是本轮固定准入门槛，不是通用理论值。若source的组内优势全0，方向不可定义，不能填1；HOLD并报告。

### C4 副作用与内存

不要直接调用未加安全截断的旧`update()`进行“只读探针”。它会写日志并调用optimizer。应在隔离run目录中，使用已经核验的hook/分离计算函数，明确捕获before-Adam；finally清空临时梯度并恢复RNG。

DRY记录optimizer_steps=0、parameter_updates=0；影子更新最多4次，单列，不能混写成“整个校准期间完全没有optimizer调用”。原policy/reference所有张量必须在结束后hash不变。

采用生产两卡无损保存tensor路径。用能在单卡完成的短序列做1卡/2卡比较；1536-token长行只在可运行的生产backend验完整batch，明确不声称单卡长行等价已经验证。不降dtype、不缩max_new_tokens、不换attention来绕过OOM。

### C5 原数值门槛

沿用old runtime contract：sampler/teacher mean abs diff≤0.01、max abs diff≤5、TIS截断比例≤0.01；unweighted KL≤1、preclip norm≤1000、postclip norm≤1.00001，全部有限且非负。这些是灾难性数值哨兵，不是“正常训练应接近这些上界”的建议。

新R24对旧parent的初始KL应实测，不能assert为0。若明显偏离旧R24同输入重放，则检查reference或policy加载。

## D. 正式事务与恢复

每步至少写：`INTENT`、请求ledger、raw与原tokens、native scores、precommit数值/身份检查、`UPDATE`、state、`COMMIT`。只有COMMIT存在且所有hash匹配才算正式完成。

恢复规则：

- 新分叉仅首次fresh optimizer；step1之后恢复必须带本路径Adam、scheduler、RNG与sampler state。
- 已保存的同policy原始rollout可精确重放；不能把其他臂rollout拿来训练。
- 已COMMIT步骤重复调用应幂等返回同COMMIT，不再次optimizer.step。
- 在Adam后但COMMIT前故障时，要么有验证过的原子恢复协议，要么HOLD；不能凭checkpoint修改时间猜是否更新。
- UNKNOWN调用必须人工/只读审计消歧；不得默默生成第二次并择优。
- 不同worker用独占锁，禁止同一路径并发写入。调度器重试不得改变科学request_id或seed。

step2重启测试需比对完整策略/Adam/scheduler/reference身份，并用已存tokens验证teacher-logprob。推理端只恢复policy且准确指定step；无需恢复训练optimizer，但不得加载错endpoint。

## E. 最小交付日志

每一步记录source/dtype/backend/图像/input/token/policy/reference/optimizer身份；完整G8的J、c、known、d、证书；累积真实生成数、forward/backward次数、影子更新与正式更新分别计数；所有technical hold和attempt保留。

任何结论必须对应原始请求、输出、评分器版本与统计脚本。只写“PASS”却无原始证据，不能放行后续阶段。


## 服务器必须补齐的历史完整测试

当前上传包的 `tests/test_score_eval.py` 在本地导入时缺少 `data_design.py`，完整日志保留在 `verification/HISTORICAL_SCORER_TESTS.log`。不得临时造一个同名生成器“修好”旧测试。请在服务器原完整 `package_v2` 中读取经SHA绑定的真实8类别生成模块，运行旧评分/生成器测试，再执行本轮独立oracle与黄金夹具。将完整日志绑定至 `native_scorer_golden_pass` 的证据。若旧完整项目仍缺失该模块，先报告缺口，禁止进入DRY。
