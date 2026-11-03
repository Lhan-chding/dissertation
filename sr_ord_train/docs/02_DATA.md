# 02｜训练题组、干预与语义校验

`data_design.py` 是独立的 CPU 题组及提示构建模块。它不调用模型、不训练、不渲染图像，不根据任何已有回答筛题。背景任务、调度、图像渲染、生成和正式评分由包内相应模块执行。以下函数返回的是**私有运行对象**；模型实际收到的内容只有公开图、公开提示和 assistant prefill。

## 1. 一个数值世界就是一个 root

```python
from data_design import make_world, build_context, transfer_views, swap_four

task = make_world('TRAIN', 7, n=8, structure='3_choose_2')
ctx = build_context(task, channel='REPORT', policy='D', seed=19)
world_to_render = ctx['task']['world']      # 从这个视图绘图，不把数表直接发给模型
prompt = ctx['public']['prompt']
assistant_prefill = ctx['public']['assistant_prefill']  # 自然条件均为 '{'
```

`make_world(split,index,n=8,structure=None,*,master_seed=2026110301,direction=None,aggregate=None)` 的输出含 `root_id,qid,view,structure,world,query,gold,boundary_indices,boundary_categories,required_from_boundary,strictly_better_indices` 等。`gold.answer` 和 `gold.operands` 使用历史探查的精确数字字符串。`split` 是稳定标识，不是从文件名推测；同一 root 的所有视图留在同一 split。`root_id` 显式包含类别数、结构、序号、方向、聚合与主种子，不把父模型、训练臂、提示条件、采样 draw 当成新 root。

世界字段保持历史形式：`world.categories` 是图中从左到右的类别，`world.series.Alpha/Beta` 按同一索引与类别绑定，`world.domain=[10,90]`。查询为 `family=TOPK,selection_series=Alpha,value=Beta,k=2+r,direction=largest|smallest,aggregate=sum|mean`。选中集合在相等 Alpha 上按原图左到右取前 r 个，再在这些实名对象上取 Beta，不是按 Beta 排序。

- 恰好两个对象严格优于边界分值。边界同值块大小 m 在 2、3、4 中取，需从中选 r=1 或 2，且 r<m。默认平衡循环的四个结构为 `2_choose_1,3_choose_1,3_choose_2,4_choose_2`；每连续完整的 16 个默认 index 含四种结构 × 两个方向 × 两个聚合各一次。需要按设计指定训练支持范围时由调用方显式传 `structure,direction,aggregate`，并核查最终每格分母，不能默认所有子样本仍平衡。
- Alpha 与 Beta 取自 12,18,…,84；Beta 全类各异。只就结构检查所有合法边界 r 子集的 Beta 和：任意两个必须不同。固定了同样两个严格优对象与同一聚合后，不同合法选择便不可能得出相同 gold。这里没有排除模型给错绑定或算错后碰巧答对；A、S、B 仍须分开。对于 k=3 和 k=4，生成的正确均值都可用有限小数或整数表示。
- 两套独立 oracle：第一套排序后取前 k，第二套穷举 k 子集，要求每个入选项都按值与图序严格领先于每个未选项，另行用 `Fraction` 聚合。两套输出的索引、实名、操作数和精确答案必须全部相同；不一致即报错。数值防碰撞只对生成规律做筛查，不读模型输出；每个 root 记录结构性抽样次数。
- 10 类别的世界总可提供与主边界同规模、完全不入选的同值块供可选安慰剂。8 类别 `4_choose_2` 不可能同时容纳 2 个优对象、4 个边界项和另 4 个安慰剂项；`matched_placebo_available=False, placebo_indices=[]`。勿把更短的扰动宣称为等长 placebo。

`world_query_key` 保留实名；`unlabeled_world_query_key` 将名称替换为位置编号。两者沿用上轮探查的规范化散列方式，便于与历史清单核对。split 改名本身不是数值隔离证明：正式冻结时须对整个实际列表另做数值等价、跨分割和历史重复检查。训练数据出现结构性或身份重复应终止清单冻结并查原因，不得根据模型效果换题。任何多个配对视图共享同一 root；统计不能把图、清单和 draw 各自当作独立世界。

## 2. 公共提示只放一个类别清单

8 类采用 `vendor/native_src/sr_f6/interface_v3_protocol.py` 的实际 `public_prompt(categories,query)`，保留全部 F6 字段说明与用途：evidence 给每类原始 Alpha 和所选对象的原始 Beta；可选 work；selected 是最终类别字符串数组；bindings 给最终实名及其数值；aggregate 和精确 answer 在外层。10 类按冻结 ORDER-PROBE 的问题文字、同一个 `format_contract()` 和 TOPK evidence checklist 构造；J 不适用于 10 类，10 类只能采用已核准的独立 A/S 等口径。**不要用 V 代替 J，不要把 F6 简化成三字段旧 parser。**

`build_context(task,channel,policy,seed)` 返回 `task,prompt,assistant_prefill,public,audit`。`public` **仅有** `prompt,assistant_prefill`；`task` 是私有 oracle/渲染/评分所需的真实视图。`audit.list_order`,`audit.expected_report_order`,`audit.list_induced_selected`,`audit.list_conflict`,`audit.report_conflict` 等只写日志。图像须用 `ctx['task']['world']`，不要使用另一个视图缓存，也不要把 `ctx` 整体序列化进聊天模板。

| channel | policy | 实际变动 | 应用 |
|---|---|---|---|
| `BASE` | R/G/D 均不影响 | 历史 v3 标准提示。唯一清单按图中顺序 | 训练标准暴露、测试基准 |
| `LIST` | `G` 或 `D` | 只替换 checklist 中那一份清单，不再追加第二份；不给报告排序指令 | 清单对选择的干扰 |
| `REPORT` | `G` 或 `D` | 替换同一份清单，附加同一句英文指令：Alpha evidence 要按该名单依次报告，任务 tie rule 不变 | 报告顺序与任务参照不再相同 |
| `ENUM` | 忽略策略，固定定向 | 指定随机流内，条件化到清单诱导的边界集合与 gold 冲突 | 训练前因子探查或评测 |
| `CLARIFY` | 忽略 | BASE 后追加只澄清任务参照的语句，不改其余文字 | 与 `BASE` 配对 |
| `CLARIFY_CONFLICT` | 忽略 | 与同 seed 的 `ENUM` **完全相同名单**，只追加同一句澄清 | 与前三者构成 2×2 |
| `NO_LIST` | 忽略 | 删除具体名字清单，仍要求报告图上所有类别的 Alpha | 弱化外部清单的评估 |
| `ALT`、`PRINT` | 忽略 | 分别使用此前字母序、图序报告措辞；仍只出现一份公开清单 | 历史语境/描述性对照，不用于训练 |
| `TRANSFER` | `CONFLICT`/`ALIGNED` | 8 类主/对照视图及独立 10 类迁移视图；对 Alpha evidence 给**逆字母序**报告要求，仍按图序解题 | 未见报告规则；详见下节 |

三臂的训练格式契约、图、查询和 reward 均一致；训练时不重命名，不改变同一实名的 Alpha/Beta 值，图像或像素缓存键不随 G/D 清单改动。LIST 的 G/D 文字完全一致，REPORT 的 G/D 文字也完全一致；每个样本只有一处真的类别枚举，schema 里的虚拟类型示例与之无关。只允许名单顺序的采样法不同，不在可见提示里写“这是冲突题”“D 选择了错误项”之类的提示。`U`（按要求报告）的目标只在 REPORT、TRANSFER、ALT、PRINT 有定义；LIST 中模型不必按清单报告，不能因未服从未提出的要求扣分。

**G 与 D 的区别。** 给定相同 root、channel 与 seed，G 从所有类别名的均匀随机排列中抽一次；不读 boundary/gold，不去除巧合对齐。D 用同一随机流，保留首次使“按这份名单破同值”选出的实名集合与 gold 不同的排列。它只用已知题组结构，不读模型表现。两者若首个随机排列已经冲突，实际提示完全相同，这是设计应保留的结果。非边界其余次序不额外对抗。生成后双方才共同记录诱导集合。

随机 G 的理论自然冲突率为 `1−1/C(m,r)`，四种默认结构依次为 1/2、2/3、2/3、5/6；D 为 1。故“两臂同剂量”应指 LIST/REPORT 渠道、上下文数量与预算相同，而非边界冲突率相同。实际对比识别的是**基于语义结构定向增大冲突暴露**相对随机增强的效果；如未能区分 D/G，不可把随机 G 当作不带训练信号的空白组。

## 3. 两类互补的成对评估，不要混为一种

### 3.1 图序 × 名单序：实名与读数绑定固定

```python
four = swap_four(task, seed=41)
a = four['G0_L0']; b = four['G0_L1']; c = four['G1_L0']; e = four['G1_L1']
```

在边界块的图序前 r 名中选一个原本入选对象，从其余同值项中选一个未入选对象，将这两个实名连同各自的两列数值一起交换图上位置。`G0/G1` 是两张图；`L0/L1` 是固定按**实名**传递的两份清单。L0 是原图的类别序，L1 交换同样两名，均不随每格图像再次重建。每个返回值都含公开提示投影和实际私有 `task` 以及 `graph_swap,list_swap`。

| 对照 | 固定的东西 | 目标行为 |
|---|---|---|
| G0L0 ↔ G0L1，G1L0 ↔ G1L1 | 同一张图和同一 gold | 只换清单不应改变正确选择；要求两边都正确，不能以两边重复同一错误计成功。 |
| G0L0 ↔ G1L0，G0L1 ↔ G1L1 | 完全相同的文本清单、问题及指令；每个实名的两列数值不变 | 只换图位应正确换入另一名同值对象，并按其 Beta 数值改变答案。 |
| 对角 ↔ 交叉 | 同一 root | 对角清单与图诱导集合一致，非对角冲突；四格呈现，不能只选命中预期的一对。 |

同一图的两个 L 条件共享 qid 与像素；生成请求唯一性须在外层同时编码 `Gx_Ly`、父、臂和 draw。不同图位的子视图 qid 不同但 root 相同；变体不改变两列绑定，不对模型重绘或挑选成功样本。

### 3.2 逆字母序报告：未见语境与一致对照一起预制

```python
t10 = make_world('TRANSFER_TEST', 3, n=10)
views = transfer_views(t10)  # {'ALIGNED': task1, 'CONFLICT': task2}
c = build_context(t10, 'TRANSFER', 'CONFLICT', 0)
a = build_context(t10, 'TRANSFER', 'ALIGNED', 0)
```

不做模型筛选，也不随机抽到“恰好冲突”为止。在原边界槽位内，将**整行**按类别名逆字母顺序摆放生成 ALIGNED 视图，按正字母顺序摆放生成 CONFLICT 视图。名字集合、每个实名绑定的 Alpha/Beta、问题、边界结构及非边界槽位保持不变。两视图的图位和正确实名集合会改变，合法选择因此也有不同正确数值。两视图绝不是同一张图，必须用相应 `ctx.task.world` 渲染，且各训练臂看到完全相同的一对测试视图。不可把两视图的差仅解释为文字指令因果效应；要做纯文字对照，应在同一图再交叉报告条件。默认 TRANSFER 在没有指定关系也没有预制视图时返回 CONFLICT；**主清单必须显式生成并核验两个关系**，不能靠默认值遗漏一致控制。

逆字母序在查看数值之前就可执行，不增加读取排序另一列的负担；Beta 排序若另做必须单列压力测试。本模块只实现图序为真参照的 TOPK。题目若改为规定“按清单破平局”，正确 gold、提示及评分定义都必须另行注册；不能用本模块的图序 gold 推断模型也学会了任意合法参照。

## 4. 安全接线与可复现检查

1. 调用者按照预注册计划给同一训练题的 G/D 传入相同根、频道和 context seed；两臂的调度若自行另抽 seed，就失去共享首个候选排列的性质。生成模型的 rollout seed 与此处排列 seed 分开记录。
2. 对相同 root 的多视图、不同父和条件，元分析聚类按 root；不同 G8 同题采样不等于不同图，训练种子也不能从 draw 数目推出来。
3. 保存私有数据与公开请求到不同清单；所有任务生成、图像渲染、hash 和评分契约在正式调用模型前冻结。在正式 CPU 校验里分别确认 8 类正确 fixture 的历史原生 J=1、已知错答 J=0、10 类不会静默通过 J；不要为兼容而改写历史奖励。
4. 包内独立测试（从包根目录运行）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_data_design.py -v
```

测试覆盖合法子集答案无碰撞、双 oracle、原生 F6 契约、均匀 G/定向 D 的采样差、2×2 同名单澄清、四格图像/文本/绑定的独立性，以及跨逆字母语境的确定性构造。通过只代表实现满足这些生成与计分不变量，不预言训练增益。

**正式 TRANSFER 补充定义：** 逆字母序报告的两种视图都删掉枚举名称的清单，只要求报告图中每类 Alpha。两幅图使用逐字相同的提示，不从另一条有序公共清单泄露图上优先关系。8 类 conflict 为唯一主面板；10 类只作迁移。模板变化包含撤清单和新报告指令，因而 D−G 是对这个具体未见语境的比较，而非分解两种因素的单独因果效应。
