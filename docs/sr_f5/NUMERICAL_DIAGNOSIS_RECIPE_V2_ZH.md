# G3 v1 数值诊断与唯一开发配方修订

2026-10-26 03:18（新加坡）。v1 未通过 G3，不启动正式实验。保留所有 v1 原始回答、失败批、检查点和恢复证据，不把 FP32 诊断追认为 v1 成功。

## 真实证据

- 科学版本 `ea50c51f0f145a183df360ab82025216d6303aca`。226283 的 b1-SHUF step7 在更新前触发最大 sampler/train 差 5.12116194 nat，冻结阈值为 5；均值 0.00141256，不能用均值覆盖尾部。b1-SHUF 和联动终止的 b1-DENSE 均停在已提交 step006。
- 226846 COMPLETED/0:0；BF16/FP32 各 24 次同 token 前向。原 BF16 全序列八行全部逐值复现失败时训练概率。梯度开关不改变结果，两者实际均 eval。FP32 不能与旧 BF16 sampler 混作同精度概率。
- 226890 COMPLETED/0:0，19:16:20–19:17:06 UTC，gpu-pro6000-7。两种精度各做原 G8 及 row3 单条前 163 token 的强制缓存回放；共 652 次 native forward、2,934 extra forward sequences，无随机采样、无参数更新。每个 token 的 raw logits 与强制前 scores 精确相同。
- BF16 G8 全 1,304 个概率值与原 sampler 逐值相同。改变批次为单条 row3 时，与原 sampler 最大差 2.340447725728154 nat。
- 同精度 FP32 G8 cached 与 FP32 full：前 1,304 个值均值差 0.000001405155923 nat，最大差 0.000563859939575 nat；FP32 single row3 cached 与 full 最大差 0.001048922538757 nat。FP32 G8 与 single 最大差 0.000859379768372 nat。
- 两精度原始 token、prompt/mask/grid、processor 图像身份相符。缓存回放的实际语言模型位置与 cache/mask 元数据一致；G8 row3 与 single 的 163 次位置逐值相符，完整保存的 0/151/162 边界 mask 相符。prefill 1009 token；第162位置为 [1170,434,434,434]，cache 长度1170→1171，decode不再传入pixels。

这支持可重复、精度与计算形状敏感的数值差异。目前未观察到 token/图像/检查点/缓存偏移工程错配；尚不能归因到某一个具体内核。首次 full 诊断未记录实际语言模型位置张量，因此不声称所有 full/cached 内部位置均已逐值比对。没有必要继续重复大模型诊断。

## contract/05、07 决策

登记唯一 `recipe_v2`：全模型、所有浮点输入，以及 generation/current/reference 三条路径统一 FP32；适用于三父模型和全部五臂。log_softmax 原本已经 FP32，不把重复转换称为修复。改变模型计算精度会改变概率与训练轨迹，必须作为新开发版本，不能在旧 SHUF 路径上单独切换后继续。

新根固定为 `/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026`。保持原模型快照、父适配器、协议、scorer、mask、损失、优化器、阈值、步数、臂、批量、token预算和正式数据种子；仅增加明确数值配方，并把 dev/pilot 数据种子设为 `seed_for(master_seed, "recipe_v2", split)`。配置、来源与原因必须在任何新回答前登记。

重新完成 G2 全部4,608回答和 G3 全部五臂门；G0–G4真实通过并冻结后才进入正式实验。v2再失败则停止在可实施性诊断，不允许第三版、放宽阈值或筛选 token/seed。v1未完成路径保持未完成，不能合并到v2补齐分母。b1-DENSE部分采样组完整保留且不重采。现存v1 J/SHUF恢复继续完成原连续对照，交由已验证边界控制器在比较通过后退出，不再续跑v1剩余池。

## 资源与证据

226588 的 b2-DENSE/b2-LOCAL 原定两路径各12步、合计3,072回答均已 COMPLETE，作业 COMPLETED/0:0（19:16:20 UTC）。这不是整个 G3 PASS。等待 v1 恢复期间最多另开一个双卡 v2 G2 作业；第二个双卡作业仅在全用户/teacher实际与排队预约允许时提交，始终不超过5卡和5作业。

完整诊断在 `validation/G3_PROBABILITY_DIAG_226846/` 与 `validation/G3_CACHE_DIAG_226890/`；服务器原始失败向量、所有试验日志和旧版本保持不变。这里的v2为前瞻性决策，不代表其实现、CPU门或GPU科学门已经通过。
