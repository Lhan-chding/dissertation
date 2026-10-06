# 实验配置、矩阵和状态字段说明

## 数据来源和实验类型

模型为Qwen/Qwen3.5-9B，基座revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。冻结推理源提交 `ad09688d343c0387d5305077eda610a0248a482c`；运行identity `e8e99198b7b533d88a43936c5b0e4fba3d62cde04664c6bc720fd3c74f086943`。本轮使用既有检查点；训练步、优化器更新、backward、额外似然重评分与新选择器拟合均为0。实际加载证据在 `backend_receipts/`，检查点绑定在 `checkpoint_bindings/`。

本轮接口为SYMBOLIC_FRESH，公开输入为符号记录及可靠关系。真值、污染坐标和DPE用于离线审计，未作为主提示输入。公开提示、实际chat template和输入token都保存在原始记录。

## 六个检查点与正式回答数量

| ID | 历史来源 | D48核心 | 执行对照 | 输入输出因子 | U22 | 合计 |
|---|---|---:|---:|---:|---:|---:|
| S32 | 源61001，t32 | 4,608 | 2,304 | 6,144 | 4,224 | 17,280 |
| S96 | 源61001，t96 | 4,608 | 2,304 | 6,144 | 4,224 | 17,280 |
| R4_128 | 61001_t96，经R4继续32步的既有端点 | 4,608 | 2,304 | 未登记 | 4,224 | 11,136 |
| DIRECT_128 | 同一源t96，经DIRECT_REPAIR_R4继续32步的既有端点 | 4,608 | 2,304 | 未登记 | 4,224 | 11,136 |
| REP32 | 源61003，t32 | 4,608 | 2,304 | 未登记 | 未登记 | 6,912 |
| REP96 | 源61003，t96 | 4,608 | 2,304 | 未登记 | 未登记 | 6,912 |
| 总计 | | 27,648 | 13,824 | 12,288 | 16,896 | 70,656 |

smoke另计80：五lane各16，S32、S96、R4_128、DIRECT_128、REP96各16；REP32与REP96同lane，未增加另一组smoke。它们不是正式统计的额外draw。

## 面板、协议与单元

D48为24道cross_series和24道trend；每题每检查点每实际提示32次。cross实际提示为O0/A1/B1/BR，B0与O0完全同文共用；trend为O0/A1。因此每检查点(24×4+24×2)×32=4,608。

输入输出因子仅S32/S96：48题×4协议×32=6,144/状态。执行对照每状态为3类×12题×2协议×32=2,304。U22为11 cross+11 trend，每提示64次，(11×4+11×2)×64=4,224/状态。509个候选case对应474个唯一提示，35个提示别名；这些题目／提示数量不能直接与跨检查点登记单元数相加。

| 协议 | 输入或关系操作 | 输出语义坐标 |
|---|---|---|
| O0 | 原输入、原关系 | a,b,c,d |
| A1 | 与O0相同，仅输出契约改变 | d,c,b,a |
| B0 | B1的原关系匹配基线，本批与O0同文 | a,b,c,d |
| B1 | cross sum-star关系的可逆行变换，首叶按固定原坐标顺序确定 | a,b,c,d |
| BR | 原关系行的列举顺序反转，式子不变 | a,b,c,d |
| L00 | 命名输入a,b,c,d | a,b,c,d |
| L01 | 命名输入a,b,c,d | d,c,b,a |
| L10 | 命名输入d,c,b,a | a,b,c,d |
| L11 | 命名输入d,c,b,a | d,c,b,a |

L10−L00、L11−L01固定输出比较输入次序；L01−L00、L11−L10固定输入比较输出次序；L00−O0、L01−A1比较命名格式。各表实际contrast字段保留完整定义。

执行对照包括COPY（复制给定观察，O0/A1）、DUPLICATE（直接给出修复线索，O0/A1）和CLEAN_REL（给定正确观察，B0/B1）。控制用control_success字段；不把控制当D48修复题合并。raw中的event对执行对照可能为空。

## 固定解析和评分定义

整个输出字符串经严格JSON解析：恰好四项、每项为整数、拒绝bool/float及额外解释。按固定output_order还原 `canonical[output_order[k]]=emitted[k]`，反序不采用另一解释补救。数值域0..99。

X：合法且规范世界全对；S：合法、世界不全对但固定下游答案正确；W：合法且下游答案错；I：其余，包括解析失败和越域。四事件用于修复题且互斥完备。长度终止单独记录；若输出仍满足严格解析，按输出本身评分，不把全部length自动判I。

关系分数C_orig对原关系、C_display对显示关系；主分数I上为0。C_alg_orig/display只要求可解析四整数，允许越域。B1完整解集合不变，部分满足比例的两个口径分别保留。PTLC V1保留越域求值，V2越域时回退原观察，内部按有理数执行；预测是编译时给定的逐题数据，不是新增模型回答。

B1风险按每题每变体完整old/new规范向量是否不同登记。D48首叶风险中V1变更2题、V2变更3题；U22各2题，V1/V2在这些题上的预测相同。两个协议均记录匹配old和new的次数；分母按题数×每题draw单列，不把相同预测的两个变体视为独立样本。

## 采样与统计配置

历史uncached-prefix-recompute路径；BF16基座，adapter dtype沿既有加载路径记录。do_sample=true、temperature=1、top_p=1、top_k=0、max_new_tokens=64、enable_thinking=false、use_cache=false，无grammar约束，按原EOS或长度64停止。完整配置在 [protocol.json](protocol.json)；实际tokenizer/EOS、模型状态和环境以backend回执为准。chosen_token_logprobs来自生成过程，不是另做完整似然重评分；未保存全词表logits。

统计随机种子20261005。每提示概率Clopper–Pearson 95%边际区间；固定题二元效应使用每独立单元Beta(0.5,0.5)、20,000次，敏感性Beta(1,1)。B1互斥old/new/other使用Dirichlet对应先验；非二元分数使用经验均值、样本MCSE和场景bootstrap，不套Beta区间。场景bootstrap为5,000次，重抽题目、保持同题所有协议／状态配对，并按登记家族／结构分层。别名使用同一随机变量，不能当独立重复。具体字段、NA和区间单位见数据字典。

D48和U22分别报告；家族内场景等权、合并家族时cross与trend各1/2。T1为协议效应S96−S32，T2为DIRECT_128−R4_128，REP_T1为REP96−REP32。矩阵原登记统计与随后保存输出补表的来源区分见 [ANALYSIS_PROVENANCE.json](ANALYSIS_PROVENANCE.json)。

## 历史字段与实际终态

| 文件／字段 | 时间语义 | 当前对应证据 |
|---|---|---|
| protocol.json research_status=PROPOSED_NO_NEW_MODEL_RUNS | 提出时的登记原件，保留哈希 | SUMMARY.status=COMPLETE及实际raw |
| checkpoints.json status=SERVER_EXISTENCE_AND_COMMIT_POINTER_UNVERIFIED | 导出候选路径时的状态 | CHECKPOINT_AVAILABILITY、checkpoint_bindings、backend_receipts |
| protocol.json 的U22初始角色、holdout_replication阶段名 | 初始设计／稳定阶段键 | U22_EXPOSURE_AUDIT及每条effective_split_role |
| prior_audits与prior_export_manifests | 当时已交付阶段的范围 | 当前完整SUMMARY与最终审查 |
| FAILED作业状态 | 原失败运行的真实历史 | 恢复作业另有COMPLETED记录，原FAILED未覆盖 |

已确认11个登记场景有历史真实执行，U22全部22题保留original_split=train、effective_split_role=development_diagnostic，状态CONTAMINATED_DOWNGRADED。D48也是既有开发诊断面板。P24只有历史来源描述，本轮无新增回答。
