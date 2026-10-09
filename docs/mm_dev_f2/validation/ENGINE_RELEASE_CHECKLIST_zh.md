# MM-DEV F2 独立工程与科学释放验收清单

本清单是原合同的可执行验收解释，不修改 `docs/mm_dev_f2/design` 的任何字节，不增加实验分支。当前仅完成本地 CPU 合同复核与分配门槛回归；未新建 GPU 作业、运行模型、加载服务器 checkpoint，不能据此标记 F2 ENGINE 或科学结果 PASS。

## 1. 冻结前身份与数据

- 原计划 SHA256 为 `2e20d757d32493cde1f27184b40e15b7077df594a8c95ae0e4ad2705df5515bb`；完整 36 文件包的 manifest 和逐文件字节均核验。源码冻结须覆盖实际 `mm_dev`、被复用 `mm_core` 和执行脚本的完整规定集合，不能只接受任意非空子集。
- 必须使用独立 F2 run root，旧 MM-CORE、非零 SFT、SER-J23 的 artifact、源码和账本保持原样。模型为原 Qwen3.5-9B revision/weights，实际 tokenizer、processor、模板和 non-thinking kwargs、完整两通道 generation config、BF16 base/FP32 LoRA、eager/原 native unfused linear kernel、Python/torch/transformers 环境均绑定。
- 固定 584 数值源、292 roots、2336 图片、7008 问题；ENGINE/PREP/CONTINUE/PROBE/DEV_EVAL 分别为 4/32/64/64/128 roots。模型输入不得携带 truth、root、D、reward 或其他私有 sidecar。用改变这些私有字段而模型 tensor/prompt 不变的测试核验屏蔽。
- 每图经实际 processor，验证真实像素/tensor/hash、896×672 几何与可读性；保存 CPU processor 和人工可读性复核证据。单凭生成 PNG 的 hash 不能证明模型使用了这些像素。
- 九条题流与签收 JSONL 逐字节一致，使用 F2 的 `-dl/-dh/-vl/-vh` qid 和紧凑 canonical JSON seed 协议；不得误用旧 MM-CORE qid/seed 派生。
- 每步 24 个不同题，六 strata 各 4；每两步每 stratum 八个题的 V/D 配额为 a0=[2,2,2,2]、aP=[1,3,1,3]、aC=[1,1,3,3]。同一科学 path 768 个 qid 不重复。动作共用规定 root ranking 和符合协议的相同 qid/sample seed；请求身份还绑定当前 checkpoint，不能跨模型复用输出。

## 2. 完整训练内核

- 四条准备：同一零输出 common LoRA 起点 S0、每条新建 optimizer，A/AP 的固定成对流与 seed。三十条继续：5 starts × 2 future repeats × 3 actions；每条 reference 是该 path 的准确起点，不能以 `disable_adapter()` 的 base 替代准备后 reference。
- rAP 使用精确 Fraction：0.5 A_full + 0.5 必需 readings 正确比例；错 arity/missing readings 为 0，不添加 format、length、SFT 等奖励。所有九种可达常值（0,1/6,1/4,1/3,1/2,2/3,3/4,5/6,1）均先判定 exact constant，再返回严格 8 个零优势；保留常数组和原始记录，不重采样、不删组。
- 24×8=192 条序列先按各自 completion token 数求均值，再对全部 192 条等权平均。EOS 计入；prompt、图像和 pad 不计入。每次 microbatch 梯度按 1/192 缩放；不能按全部 token 总数平均，也不能仅对有对比组归一化。
- training sampler 固定 T=1、top_p=1、top_k=0，所有其他 logits filter 禁用，max_new_tokens=192。old selected-token raw logprob 在真实 sampling 当场保存并 detach；不得用事后 teacher forcing 冒充 old sampler。当前/reference raw logprob、ratio、KL 指标使用约定 FP32；不得隐藏额外 log-ratio clip。
- 收齐完整 batch 后恰好一个 optimizer update，无重复 epoch/旧 batch reuse。即使 policy 优势为零仍执行计划 logical step；分别报告 policy/KL 梯度，避免把 KL 或 Adam momentum 造成的参数变化称为 natural policy 学习信号。
- reference 临时切换须恢复 current 值且保持同一 optimizer Parameter 对象；每步记录 reference 身份、前后权重 hash、最大参数变化、grad、Adam/scheduler 步数和所有 batch 数量。

## 3. ENGINE_F2 仅工程验收

- 用真实图片和生产完整 192 序列 trainer，固定 rAP + KL=0.02；连续 4 步对独立进程 2+2，8 physical updates、1536 generations，3 个真正 fresh PID。结果引用同一冻结、题流、reference 和源文件身份。
- 按原固定容差核对 actual-sampler raw logprob vs 同前缀 teacher forcing：mean absolute ≤0.005 nat、max absolute ≤0.05 nat。不能因结果放宽；容差仅用于两种前向计算，不放宽 exact resume。
- 独立 CPU 读取本任务可信的真实 checkpoint，在 step2 和 step4 对参数、Adam 各 moments/steps、scheduler、reference、Python/NumPy/torch/CUDA RNG、stream position 与配置逐字段/逐元素比较；保存代码和实际 tensor/NumPy 比较统计，不能只信摘要 hash 或 COMPARE.status。
- 对全部 raw rollout 验证 token、seed、qid、sample、reward Fraction、advantage、old/ref/current logprob、loss 和更新顺序对应。保存请求/未知尝试/账本并核对全部失败成本，不将缺 raw 的已尝试调用扣除。
- 若至少一个真正 mixed-reward group，但实际 policy gradient 缺失，必须 `GRPO_GRADIENT_MISMATCH` 阻断；不得以 synthetic stress 绕过。
- 只有全部 natural groups 确实 exact constant、advantage/policy gradient 为零，才允许原合同登记的一次 fresh reset stress：+1 coefficient、仍是真实新生成、相同完整内核、独立 8 updates/1536 generations、同样 4 vs 2+2。压力测试结果不得称为自然奖励非零 GRPO，不得发布为科学起点 adapter。
- ENGINE_PASS 回执须绑定证据文件完整 hash、freeze、实际检查结果和 scheduler/accounting；禁止只写状态通过后推进。

## 4. 调度、恢复、成本

- 每 worker 一张实际 typed Pro6000、单节点，登记计划最多 5 GPU。已知和未知提交都保留并发占位；当前 owner/account/QoS/partition 权限核对实际 scontrol。job_name/comment/task/attempt/registration 和 immutable submission intent 必须一致。
- 禁止 Slurm 原 job 隐式 requeue/restart；正常租约/抢占按新 attempt 继续同一逻辑任务。提交前 durable intent，登记 job 后才 release；失联/未知提交不能盲目 sbatch 再试。
- 每 committed update 保存完整 state，两个 rolling 加不可变 0/8/16/24/32。部分 batch 只能在准确 policy/input/seed 身份和规范 slot 顺序下复用；未知 optimizer commit 回退到确证前一状态，physical 重做照计。
- 只对抢占/租约结束自动续跑。NaN、hash、技术异常停受影响任务并保留异常；行为改变需新代码版本和可比完整 block 处理。完成 marker 与 scheduler 成功、文件 hash 必须一致。
- GPU hours 仅记账，不设累计 GPU 或研究 wallclock gate；单 allocation walltime 是可续租约。旧 4096 generations/64 updates 等 caps 不能进入 F2。
- 科学固定总量：1088 logical updates，208896 training rollouts，30720 PROBE completions，430080 EVAL completions，总 669696。额外无生成前向 456192（reference208896、gradient208896、self30720、gold7680）。ENGINE、技术失败、未知尝试和重做分别额外记账。
- 实际 allocated GPU seconds 按 Slurm 所有 epoch 累加（含加载、空闲、失败）；保留 queue/wrapper 时间分别报告。释放前核验所有 attempts 的真实 terminal 与费用，不能以 summary COMPLETE 替代 Slurm COMPLETED 0:0。

## 5. 科学分析和释放

- 必须完成全部 34 科学 path 的固定 step32，以及 5 PROBE 和 35 模型身份完整 EVAL 后才能释放汇总；终点相同权重也不能去重模型身份。只评价 step32，不能按中途结果挑选。
- PROBE 为 5×1536×4：每 completion 自身字段评分；gold 每 qid/state 仅一次缓存，共 7680。DEV_EVAL 为 35×3072×4，无 endpoint field teacher forcing。
- 检验 raw/scoring 所有计划 slot 完整唯一，generation/模型/seed/像素/模板身份对应，invalid semantic 字段仍保留且对应 full 指标为 false；L 条件联合六格不能把缺字段塞进 000。
- 先 question 内 K 均值，再 root 内四 cell 均值，最后六 strata 等权；整 root 的 5000 次 paired bootstrap 同时绑定所有 endpoints/actions/futures/K。95% 和主四对比 98.75% 区间按冻结定义，不能称为独立训练重复置信区间。
- D=end−该准确起点；ΔD=action−a0；先平均 futures 再算受保护 strata 的负向部分；η=0，不扣 GPU cost。no-train 的 D、U 是0，而其相对 a0 的 ΔD 是 −D_a0，不能误填0。
- S 只在非零误差且满足支持条件时定义，root-equal 于有支持 roots；报告 undefined 与支持量。零误差/无支持不能补0，bootstrap 无支持保持 undefined。BNum 明确 error/arity/missing support，再加 S 才是 BGeo；context stability 独立描述，不偷偷进入主特征。
- crossfit 使用冻结 root halves 与 future 配对，不调 folds。oracle/feasibility 报告仅为本轮开发数据的探索性估计；不是已验证 selector、在线控制或安全证明。
- 最终 evidence SHA256 清单覆盖 manifests、raw、scoring、tables、accounting、ENGINE 和 reports，不自引用；巨型 checkpoint 若仅服务器保留，明确路径、大小/hash 和已执行独立 CPU 比较，不能声称 ZIP 可本地重算全部 checkpoint。
- 保留所有低分/失败/零梯度/同权重结果。完整 F2 后自然结束，不启动 MM-LOCK/CAL/ONLINE、新模型/seed/reward 或重启 SER-J23。
