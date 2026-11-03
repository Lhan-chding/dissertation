# 04 | 冻结统计规范与离线入口

本节针对实际分配的评测请求。程序只读取两份 JSONL，不加载模型。实验单位有两层：**评测 root**（同一原始世界及其变体）和**训练路径**（父模型、训练 seed、训练臂）。一个世界的多个视图、同题的两个 draw，以及同一训练路径的第 12/24 步，都不是独立训练重复。

## 1. 运行方式

在本包根目录运行，要求 Python ≥3.10 与 NumPy。将示例路径替换为实际导出路径：

```bash
python analyze.py --manifest run/eval_manifest.jsonl --scores run/eval_scores.jsonl --out run/analysis.json
python -m unittest discover -s tests -p test_analyze.py -v
```

正式运行应另外传入 `--manifest-sha256 <生成前归档的64位SHA256>`。程序也在输出写入两份文件的实际 SHA256。持有一个结果生成之后才计算的哈希并不能证明事前冻结。数据清单、父/检查点身份、提示及图片哈希、实际采样种子等由执行侧事前冻结并核查。下文的身份校验不能代替那些检查。`--retention-panel retention8` 可省略；不允许从看到的结果中另选保持集。

## 2. 输入契约：先登记，再评分，逐键匹配

manifest 每行是一次预分配的生成，至少有下列 9 个组成身份键的字段及 `U_applicable`：

| 字段 | 约束 |
|---|---|
| panel | 冻结名：transfer8、swap8、retention8、context8、scale10、authority8、prefix10 |
| root_id | 原始世界／谱系，非空字符串。同一面板内不得跨 structure |
| structure | 新题按实际平局结构；历史保持按原家族 |
| parent | b0 或 b1 |
| train_seed | 训练臂整数 11031 或 11032；父端点 JSON null |
| arm | R、G、D；或未更新端点 PARENT |
| step | 训练臂整数 12 或 24；PARENT 为 0 |
| view | 视图名。transfer8 为 aligned/conflict；swap8 为 p0l0/p0l1/p1l0/p1l1；retention8 为 v0/v1/v2/v3 |
| draw | 0 起算的整数；transfer8、swap8 为 0、1；其他面板仅 0 |
| U_applicable | 必须是 JSON 布尔值。transfer8、scale10 为 true；context8 仅 report_conflict 为 true；其余 false |

scores 每行重复上述键和 `U_applicable`，额外要求：

| 字段 | 定义 |
|---|---|
| A | 按上游冻结答案评分器记 0 或 1；全分母 |
| S_hit | 确认选对与否，全分母 0/1 |
| S_known | 选择声明可判为 1；不可判为 0。这时必须 S_hit=0，但不将潜在的选择真值标为错 |
| U | 有明确顺序命令且完整列出全体 Alpha 时，顺序与要求一致为 1，否则为 0。未完整列出、未知或不适用为 JSON null |
| J_native | 历史原生联合奖励的 0/1。retention8 必填；scale10、prefix10 必须缺省或 null。其他面板只在确实按原生接口评分时填写 |
| J_diag | 可选诊断指标，0/1 或 null，不得冒充 J_native；有几条适用就报几条 |

以下为格式示例（数字仅说明字段）：

```json
{"panel":"transfer8","root_id":"example-002","structure":"3_choose_2","parent":"b0","train_seed":11031,"arm":"D","step":24,"view":"conflict","draw":0,"U_applicable":true,"A":1,"S_hit":0,"S_known":0,"U":null}
```

允许额外字段。缺任意已登记行、存在多余行、重复身份键、双重 JSON 字段、非 0/1 数值、空训练 seed、未知 S 却记为命中、无指令却有 U、缺面板或路径不匹配，均拒绝计算。每个面板必须有完全相同题目/视图/draw 的 26 个端点：2 个各只评一次的 PARENT，加上 2 父 × 2 训练 seed × 3 臂 × 2 次检查点。不能把父模型记录复制给两份 seed 以制造重复。无效模型输出按原评分器保留在分母；技术上还未产生的请求应按原身份恢复，不能伪造为 0。

保持集固定为已经开放的 SR-F6 v3 DEV_EVAL/P0，沿用历史 root_id、问题、提示和图：6 个家族各 4 root × 4 原题变体 × 每题 1 draw，共 96 条/端点。这里的“8”指历史原生测量适用的类别数上界，保留历史 6/8 类别混合。分析器检查数目与键，不能独立确认实际图片和提示是否相同；生成前必须对历史清单与执行输入比对。

## 3. 唯一主要估计量：第 24 步 D − G

**transfer8、conflict 视图、数值答案 A**。该视图要求训练未见的逆字母报告顺序，且报告顺序与打印破平局顺序冲突。共有 96 个新 root，四种结构分别 24 个。每训练臂在本主比较中有 96 × 2 draws × 2 父 × 2 训练 seed = **768** 条回答；两个臂的差由配对决定。R、第12步、aligned、其他指标/面板、或一次效果最好的父模型都不能取代这个终点。

计算过程：

1. 对每个父模型、训练 seed、root，对齐两个臂的同一视图及两个 draw，各自求 A 均值并相减。没有完整公共支持即拒绝比较。
2. 在每个 root 内，将 b0/b1 × 11031/11032 四个训练块的差等权平均。
3. 在每个结构内对 root 等权平均，最后四个结构等权。输出四个训练块的点差、两个父的平均点差和两个 seed 的平均点差。相同 seed 标签在不同父模型下不等同于来自四个独立模型总体的抽样。
4. 条件 95% 区间：对已经配好的 root 差按结构有放回地抽原数量的 root；同一 root 的臂、父模型、seed、draw、视图一起移动。B=10,000，NumPy 的 default_rng 种子 2026110317，按结构和 root 名稳定排序，取 2.5%、97.5% 分位数。某结构不到两个 root 不生成有效区间。单位为比例，乘 100 得 pp。

这些区间只针对**固定的两个父模型、四个配对训练块和已生成记录**。不能用这 96 个 root 去论证跨随机训练种子、跨模型家族的显著性。分层 bootstrap 也没有对新模型解码随机性另外采样。只有两份配对训练 seed，不输出伪精确的训练 seed CI 或 p 值。应看四个块是否方向一致、是否由单父/单 seed 主导，并明确标作描述。本实验没有预登记 D≈G 的等效检验；D−G 区间跨零的结论是**未能分辨**，绝不写成两种方法等效。若在全新实验希望声明 ±2pp 等效，须生成前登记，且整段相应区间严格落在 [−2,+2]pp 内；不能事后启用本程序的数学判断辅助函数来为本轮增加结论。

## 4. 保持、服从和相同 draw 的四格

**保持**：第 24 步历史 retention8 上的 D−R 为首要保持描述，G−R 为背景；端点 A。按原家族分层，依次等权 view、root、家族、父与配对训练 seed，沿用上述配对重采样。使用同一双侧条件95%区间作保守的 −2pp 不劣判断：下界严格大于 −2pp 才支持条件不劣；上界严格低于 −2pp 表明观察到超界损害，其余为不确定。下界刚好等于线亦不宣称通过。分别报告六家族/各路径的异常时须标明探索性质；不能说“已有全面能力保持”或“跨种子稳定不劣”。

**服从**：有指令的面板/视图，分母是全部已分配回答。报告 A、S_hit、S_known、S在已判子集内的通过率、U 可判覆盖、确认 U=1 的全分母率，以及**同时 A=1 且确认 U=1**的全分母率。这最后一项衡量模型是否既答对又按要求报告；可以检验“看似进步只是忽略冲突指令”。同时报告所有 U 可判者中 A∩U 的比例作为选择性描述，不用其替代全分母。若 A=1 但 U 未知，真实联合成功的有限样本识别下界为确认联合成功率，上界再加这部分未判比例；它们不是CI。无报告顺序指令的单元，U与A∩U整体记为 NA，不把不可应用的结果计作0。

**swap8**：对同一 parent、arm、step、训练 seed、root、同一个 draw，取 p0l0/p0l1/p1l0/p1l1 的四个 A。输出四格全对；固定图 p0 或 p1、切换列表时两格均对；固定列表 l0 或 l1、切换图上顺序时两格均对。任何一个格缺失拒绝计算。输出原始配对分母和命中数，以及按 root/结构/训练块等权的比例。每个视图“在两个draw中任一个正确”与“同一draw四格全对”是完全不同的量。对照评分须由执行侧核实改变打印顺序后的 gold 已重算。

其余面板、所有臂、各 parent/seed、12/24步均输出逐 view 水平，用于解释方向和限定范围。主输出中的 `unweighted_counts` 和 `n_assigned` 是记录条数；`equal_weight_rates` 依次等权父、seed、结构、root、view、draw。非平衡面板两种口径可能不同。不把多个诊断面板的偶然正向值另列为“主发现”。历史 J 和诊断 J 分栏，各自报告适用数、命中数、覆盖率及适用者内通过率。可判数据稀少或训练块方向不稳时，明确报告不确定性，不按结果扩充预算或换终点。

## 5. 从原始回答得到 scores

正式评测和澄清面板都使用同一离线入口，模式由相应 manifest 确定：

```bash
python score_eval.py --manifest run/evaluation_manifest.jsonl --inputs run/inputs.jsonl --cases run/private_cases.jsonl --raw run/raw_eval.jsonl --out run/eval_scores.jsonl
python score_eval.py --manifest run/clarification_manifest.jsonl --inputs run/inputs.jsonl --cases run/private_cases.jsonl --raw run/raw_clarification.jsonl --out run/clarification_scores.jsonl
```

评分器按完整身份键匹配全部原始行；raw 至少包含相应清单的键、case_id、input_id、full_text 及 finish_reason（balanced/eos/length），建议同时保留实际 seed、assistant_prefill、suffix_text、运行回执。若这些附加字段已导出，程序核对实际 seed、已存 prefill 与完整文本重构。比较清单中的公开输入/私有题档内容摘要；生成文本必须逐字符以已冻结的 assistant_prefill 开头。不能将 gold、诱导集合、私有题档送入模型。不把技术缺失伪造为完成。全分母严格测量仅读顶层有效 JSON 的 answer、selected、bindings、evidence，不从 work 猜测；正确字段与协议是否完整分开计量。U 只关心有无完整且可读的 Alpha 事实及其列举顺序，数值是否读对是另一个问题。输出对可比任务调用原生 F6 score_response 取得 J_native，对 10 类别和名单作为合法规则的 authority8/list 不计算历史 J。

默认在输出旁保存 `<stem>_summary.json`。正式评测摘要另报 A 对但 S/B 明确错与未知、名单/前缀诱导集合的命中数及可判数；这些字段不当成奖励。prefix10 分别保留严格原分和预注册敏感性：仅在 EOS、原先不能严格解析、末尾追加恰好一个 `}` 即可严格解析时重新检查字段，仍不可判的 selected 保留未知；重复键、控制字符和其他需改写的文本不修。两种规则下的 S 全样本识别界均用本身分母，差值用一臂下界减另一臂上界及反向最有利组合。这不是 CI，且永远不以补括号分数覆盖冻结主数据。

澄清分析固定 24 新 root（每结构6个）× 4 视图 × 每视图2 draws × 2 未训练父模型 = 384。四格为 aligned_plain、aligned_clear、conflict_plain、conflict_clear。分别给出合并及每父的 A、S_hit 四格均值，冲突和一致条件下的澄清增益，以及两增益之差；使用同一 root 配对并在四结构内重采样的条件95%区间。这是训练前解释性分析，不能据其数值触发训练准入、选新提示或更换主要检验。
