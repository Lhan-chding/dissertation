# 07 评测、统计与结果解释

## 1. 评测对象和次数
所有父模型零步结果去重共享；不把同一父模型复制成多个独立种子基线。
A主checkpoint64；C主checkpoint32。中间checkpoint只在DEV和训练日志看机制，不能按其表现选模型。所有模型在确认集上使用相同prompt、draw数量、解码参数、独立于训练的固定seed。
A：旧结构化TEST_ID（repair/legacy）、COMPOSITION、ARITY；B确认56prompt×4draw；自然work canonical256×2draw；answer_only64×4draw；ChartQA2500仅b0两臂，限定b0；语言/渲染各128prompt×2draw哨兵。
C：C_CONFIRM_ID 240prompt×4draw；C_CONFIRM_TRANSFER240prompt×4draw；B确认56prompt×4draw。旧保持面板64根（四题族各16）×8variant×一个预定chart×2draw=1,024回答/模型。chart在根之间平衡；它是旧TEST_ID回归，不能进保持训练。
C自然work检查：从新确认ID按cell均衡固定24world×2题×2draw=96回答/模型；它是协议交叉测量，单报，不与结构化主结果混合。外部真实图表不作为C主终点，本轮不声称已验证真实分布迁移。

## 2. C主要终点与主要对比
主要终点：TRANSFER中新世界的原native `value_correct` 正确率，按world内prompt/draw平均、cell等权、block和repeat等权；同时报告原生J、独立数值答案审查、协议合法率。禁止以新P/S验证器得分作为主要“成功”。
三个预登记对比：SELECT−J、SELECT−EVID、SELECT−SELECT_SHUF。三个都报告，不只挑正向对比。
重要次要：ID新世界；严格成对全对率（同draw序号两题各自独立seed，且两个都正确）；预定语义响应关系满足率（不能替代单题正确）；S的未知与失败率；旧保持宏平均、逐变体和最差差值；work/length/截断/单位状态。
TRANSFER的四子层分别报告；组合留出若不可实现，不声称该层结果。

## 3. 不确定性必须分层
固定已训练模型的题目不确定性：10,000次按world分层配对bootstrap，保留每world的全部问题、图像、draw及所有模型对应记录；在cell层内重采样。三个主要对比用98.333%区间（Bonferroni family-wise 95%）。统计单位不是单条回答。
训练重复：每个block有C两个seed，分别展示配对差，再对两seed取均值获得三个block差；三个block不是六个可随意互换的独立同分布模型。block均值的t区间（df2）只作描述性敏感性；不要用bootstrap题目数冒充训练重复数。
可额外给block内seed的nested bootstrap，但每块两条路径估计有限，不能宣称可靠模型群体泛化。父模型这三个固定状态下的结论与任意训练种子/任意模型分开。
A每block只有一个新LZ与一个新uniform；历史RL只作参考；不能把历史与当前不同采样日程差解释为纯训练随机性。

## 4. 保持不是点差阈值
逐block/seed/variant的差值必须展示。−2pp可作为预先定义的实用保持界；只有相应适用范围的单侧置信下界高于−2pp，才可作该范围的非劣描述。六/三条路径一致点差≥−2并不自动满足统计非劣。最差子组的选择也带不确定性，需说明多重查看。

## 5. 阶段判定（不触发追加训练）
“支持继续扩展”：SELECT三个主要对比点估计均>0，配对world调整区间均排除0；多数block/seed方向一致；保持没有明确损害。仍限定固定模型，训练随机性证据弱时直说。
“信息可能有效但未区分机制”：仅对J有收益，或与SHUF/EVID不分；不宣称新增选择语义已经得到识别。
“无清晰收益/损害”：按原结果交付。不得延长步数、换β或重抽种子追阳性。
若有技术阻断，成套结果标INCOMPLETE_TECHNICAL，不删除对应不利路径，不用替补seed掩盖。

## 6. 发布必要图表（由Codex生成）
总体与分关系正确率；各block×seed配对点图；新增信用激活率与UNKNOWN随步数；被赋予正/负信用的P/S状态分布；旧保持逐variant差；采样曝光与G分布；length/格式/单位副作用。
所有图标题必须含样本范围、checkpoint与是否开发/回归/确认，不将拟议期望曲线画成实验结果。
