# SR-F4 原生生产契约

本记录核对继承的 SR-F3-CE 源码 `87b5d29e6790b0c1f8f43cbc74c704350693e3e5`，不是 GPU 训练结果，也不表示黄金样本回放已经通过。机器字段见 `configs/native_contract.resolved.json`。用户要求省去重复哈希和无关烟测，因此此次审查只读取实际调用链；环境版本、模型路径和父 adapter 的具体目标模块清单由主代理实时读取服务器后提供，已写入机器契约，不以旧笔记猜测。

## 原生 J 与字段

调用 `sr_f3_ce.contract.score(text, world, query, finish_reason=..., protocol=...)`。结构化奖励为：

`J = int(value_correct and certificate and presentation and finish)`。

J 是二元，源码未给连续奖励四舍五入。`value_correct` 使用 `Fraction` 精确答案；`null` 对应 oracle 的 `None`，例如空集合上的 mean/range/max/min 未定义，sum 和 count 仍有定义。`certificate_truth` 检查已报告事实是否与世界一致；`certificate_sufficient` 来自 `sufficient_template`，只使用已报告事实和公开取值范围，不偷看未报告值进行证明。`certificate = truth AND sufficient`，不能把 certificate 当作 truth。

`parse_facts` 限制列表长度、字段、实体、重复事实和整数域；无效证据使结构化 presentation 为 0。原生 `all(...)` 对空证据可能得到 truth=1；SR-F4 的 P 必须另行要求非空。充分性模板提供可保证成立的充分条件，但不保证识别所有合法证明。

`presentation` 要求合法初始 JSON 对象、合法 answer、允许的 evidence/work/answer 字段与类型、无非空后缀。重复 JSON 键拒绝；字段排列单独记诊断，未进入 J。`finish` 同时要求正确停止原因和已闭合输出：结构化 balanced/eos，自然 work 的 eos/final_tag。`reason='ok'` 仅在完整 J=1 时出现，不能用它作协议合法性的替代。技术生成错误由 `generation_status` 单独处理。

answer_only 和 natural_work 是独立数值测量协议；natural_work 分支返回 value/presentation 后不计算证据 J，不能把这些 J=0 当成结构化失败。

来源：`src/sr_f3_ce/contract.py` 的 `rational`、`aggregate`、`parse_facts`、`registered_completion`、`score`。

## 优势和目标函数

`sr_f3_ce.training.group_advantages` 使用 NumPy float64，每题组总体标准差（ddof=0）：

`A_i = (J_i - mean(J)) / (std(J) + 1e-8)`，常数组为 0。epsilon 位于标准差外部，不能换成方差内 epsilon。

原生 loss 接口是 `sr_f12.training.sequence_objective(current, sampler, reference, coefficient)`。三个 logp 为不含 padding 的 completion-token 向量，转 float32；coefficient 为单个有限、无梯度的最终 sequence advantage。prompt、图像和 assistant prefill 不作为 completion loss 目标。

令 `old = current.detach()`，`ratio = exp(current-old)`；单次 forward 数值上 ratio=1，但保留对 current 的导数。TIS 权重先截断 log 权重再指数化：`w = exp(clamp(old-sampler,max=log(2))).detach()`。policy 项为 `-w * min(A*ratio, A*clip(ratio,.8,1.2))`。KL 项为 `expm1(reference-current) - (reference-current)`，系数 .02，KL 不乘 TIS。

每条回答内先对 token 取均值，随后每条回答除以 128 再求和。不是全部 token 的统一均值；G32 组的 sequence 权重是 G8 的四倍。所有组使用同一次更新前的策略生成，完成 128 条后只更新一次，不重复优化同一 rollout。

原生训练记录 sampler/train 平均绝对 logp 差、TIS 截断率；原实现会在差值 >.01 nat 或截断率 >1% 时阻断更新。这是实际概率一致性约束，不是因验证准确率低而早停。SR-F4 如果保留此行为，应保留真实异常记录。

C 的额外 residual 应在调用 sequence_objective 前加到原生 A_J，之后不再 whitening。J 对照传原生 A_J；没有教师 CE。

来源：`src/sr_f3_ce/training.py:140`、`:380`；`src/sr_f12/training.py:56`、`:117`。

## 优化器、模型与生成

复用原父模型 LoRA 权重，r=8、alpha=16、dropout=0、bias=none；所有 language_model.layers 内的实际 Linear 都应在继承的 adapter_config 目标清单中。视觉模块、merger、projector、lm_head、embedding 不可作为可训练 adapter 目标。底模 BF16，LoRA FP32，eager attention；不初始化新 LoRA。

每条路径使用新的 AdamW 状态：lr=1e-5、betas=(.9,.999)、eps=1e-8、weight_decay=0、foreach=False；常数 LambdaLR，无 warmup。全部 128 sequence 的梯度累计后以 global norm=1.0 截断，再 Adam.step 和 scheduler.step。KL reference 是每块初始 J6 父 adapter，随训练不动。

训练和合成评测均为 temperature=1、top_p=1、top_k=0、max_new_tokens=1024、num_beams=1；v2 structured 使用注册 prefill 和第一完整初始 JSON 停止。ChartQA plain 是 greedy、上限256；natural_work 用闭合 final tag 停止。启用 thinking=False；原 native sampler 记录实际 raw-logit 选择 token 的 logp。

最终 SRRuntime processor 限定 min_pixels=max_pixels=786432，画布1024×768；不能误用底层 QwenRuntime 初始化中暂时出现的896×672。模型输入边界只有 prompt/image_file/image_sha256，oracle/world/gold 不进入推理 prompt。

来源：`src/sr_f3_ce/runtime.py:61`、`:193`、`:308`；`src/sr_f3_ce/training.py:64`、`:474`；`src/sr_f1/runtime.py:600`。

## 复用接口及必须改适配的部分

可复用 `SRF3CERuntime(...,protocol_revision='v2')`、`configure_parent_training`、`fresh_optimizer`、`batch_sequence_forward`、`batch_reference_forward` 和 `sequence_objective`。

不能直接调用旧整套 runner/trainer：

- `sr_f3_ce.training.group_advantages` 强制16×8；`_task_batches` 强制16个不同 qid。SR-F4 LZ 允许 G32 和重复 qid，须以独立 group_id 验证边界，并使用相同公式计算每组优势。
- `generate_group` 接受1至8个 slot seed。G32 可由四个原生 G8 块生成，再合为一个优势组；不能当四个独立 G8 优势组。
- 原生 generate_group 仅用 seeds[0] 初始化批量 RNG，其余 seed 是身份标签。冻结清单应如实记录批量 RNG 语义；如果要求每 draw 真正独立 RNG，就需不同实现，不可只写出每 draw seed 后冒称已使用。
- `sr_f3_ce.runner.load_runtime/load_parent_runtime` 绑定旧 experiment、task、父审核及旧根目录；新实验应自行实例化 runtime 并读取已存在父 adapter。
- 原 checkpoint 模块绑定旧 arm/cursor/recipe，且 `read_checkpoint` 使用 `torch.load(weights_only=False)`。不能把它直接作为新实验不可信输入的读取器。新恢复状态必须保存 adapter、optimizer、scheduler、成功更新游标、sampler/rollout/shuffle 等随机流；数据 JSON/JSONL 不用 pickle。

## 已知本地限制与运行状态

本代理未加载服务器模型、未运行 GPU 工程验收、未运行历史36组 golden test，也未修改原训练或评分源码。代码内语义与服务器身份已解析。服务器实测 Python3.12.14、torch2.13.0+cu130、transformers5.14.1、peft0.19.1、numpy2.5.2、Pillow12.3.0。底模 revision 为 c202236235762e1c871ad0ccb60c8ee5ba337b9a；三块父 adapter 均来自 selected_v2/parents/SRF2_J6_s71001、71002、71003/adapter。完整绝对路径及继承目标模块见机器契约。新 SR-F4 适配器的行为验证仍单独记录。`native_golden_tests_passed=false` 表示尚未执行，不能被本审查文字替代。

C 仍要求真实人工盲核；自动标签或此源码审查不提供 HUMAN 资格。A/B 可按执行包继续，不需要等待 C 人工核验。

SR-F4 已在 G8 批量生成内部为每个 draw 使用独立 torch.Generator，因此保留批量效率又落实逐 draw seed。CPU 测试确认逐 draw 结果与各自独立抽样流一致、拆分批次不改变同一 draw 的抽样、异常后恢复原生 multinomial。该结果不代替真实模型 GPU 执行。

## 本次新增适配验证

`tests/sr_f4/test_training.py` 的7项必要 CPU 测试通过（0.96秒，local torch2.13.0/numpy2.5.2）：枚举全部256种 G8 二元奖励与原生优势逐值一致；可变 G32 与重复 qid/独立 group_id；J 路径与原 trainer 的 loss、梯度和 Adam 单步一致；额外优势不再标准化；安全 checkpoint 的参数、优化器、游标和随机流恢复；逐 draw RNG 与批次拆分一致；异常后撤销临时采样函数。均为微型可微模型/CPU 检查，历史36组回放仍未执行，GPU 未执行。
