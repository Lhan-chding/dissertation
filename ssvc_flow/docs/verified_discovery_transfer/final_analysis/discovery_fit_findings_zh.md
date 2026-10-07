# 发现与拟合：最终既有数据核读

本分析只读取已统一释放的现有数据。B16为预登记发现预算；B2/4/8是事后固定前缀描述，沿用B16在V冻结的single，未重新选协议。MIX各分一半调用。拟合与V成功率不能替代E迁移和G保持性。

## 核读解释

四组B16的MIX均比O0发现更多任务，同时仍丢失0–3题；集合不是嵌套关系。增益主要集中在DPE1_O0非正风险层：四组共347条新增membership中的344条来自该层（同一T任务在不同parent/repeat中重复计数，不能称为347个独立新题）。低数值域all_0_49只有16题，分层描述不宜过度外推。

MIX扩大训练集合，使固定3072次focus曝光分散到278–293题，每题约10.48–11.05次；O0为196–202题，每题约15.21–15.67次。曝光改变是注册流程的组成部分，不能把差异全部归因于协议本身。

低训练loss不等于广泛拟合：SELF_O0的末步loss约4.19e-6–1.89e-5，但四组固定16题sentinel终点均仅6/16；SELF_MIX末步loss更高，却达到10/16–13/16。该sentinel同时包含未进入某臂J的T题，所以不是仅对该臂已训练目标的再现率。

V128保留不利结果：S96两轮MIX均47.66%，低于O0的49.22%和SINGLE的51.30%；REP96第一轮MIX与O0同为48.70%，第二轮MIX49.22%对O0 47.40%。这些是固定步骤诊断，不据此更换256步终点，也不是E迁移结论。

GOLD_MATCH与MIX实际集合交集分别257/290、265/293；每侧独有33、28题，Jaccard约0.796、0.826。匹配成功不代表二者输入相同。

## 四组发现

| parent | repeat | J_O0 | J_MIX | J_SINGLE | mix_added | mix_lost | mix_o0_intersection | mix_single_intersection | o0_single_intersection | three_way_intersection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REP96 | 0 | 200 | 279 | 209 | 82 | 3 | 197 | 205 | 178 | 175 |
| REP96 | 1 | 196 | 278 | 204 | 82 | 0 | 196 | 203 | 174 | 174 |
| S96 | 0 | 200 | 290 | 209 | 91 | 1 | 199 | 207 | 180 | 179 |
| S96 | 1 | 202 | 293 | 211 | 92 | 1 | 201 | 208 | 180 | 180 |

## GOLD匹配实际重合

| parent | repeat | N_GOLD_MATCH | N_MIX | intersection | gold_only | mix_only | jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S96 | 0 | 290 | 290 | 257 | 33 | 33 | 0.7956656346749226 |
| S96 | 1 | 293 | 293 | 265 | 28 | 28 | 0.8255451713395638 |

匹配仅控制family×损坏位置及数量，不能视为完整难度控制。

## 训练拟合

| job | updates | focus_unique | focus_exposure | replay_exposure | last_loss | last_gradient_norm |
| --- | --- | --- | --- | --- | --- | --- |
| sft.REP96.0.GOLD_ALL | 256 | 384 | 3072 | 1024 | 0.01237167208455503 | 1.224047303199768 |
| sft.REP96.0.SELF_MIX | 256 | 279 | 3072 | 1024 | 0.010803206478158245 | 1.8961522579193115 |
| sft.REP96.0.SELF_O0 | 256 | 200 | 3072 | 1024 | 4.987423409374969e-06 | 0.0014318992616608739 |
| sft.REP96.0.SELF_SINGLE | 256 | 209 | 3072 | 1024 | 5.167750323220588e-06 | 0.002415332943201065 |
| sft.REP96.1.GOLD_ALL | 256 | 384 | 3072 | 1024 | 0.03765977616967575 | 1.5705671310424805 |
| sft.REP96.1.SELF_MIX | 256 | 278 | 3072 | 1024 | 0.023275773234260555 | 4.957441806793213 |
| sft.REP96.1.SELF_O0 | 256 | 196 | 3072 | 1024 | 4.188667308113736e-06 | 0.0006396144744940102 |
| sft.REP96.1.SELF_SINGLE | 256 | 204 | 3072 | 1024 | 4.140237746863562e-05 | 0.011453704908490181 |
| sft.S96.0.GOLD_ALL | 256 | 384 | 3072 | 1024 | 0.015282368985936046 | 1.9225698709487915 |
| sft.S96.0.GOLD_MATCH_MIX | 256 | 290 | 3072 | 1024 | 0.013987793958222028 | 2.1140201091766357 |
| sft.S96.0.REPLAY_ONLY | 256 | 0 | 0 | 1024 | 2.798507921397686e-05 | 0.0021627063397318125 |
| sft.S96.0.SELF_MIX | 256 | 290 | 3072 | 1024 | 0.0023273455244634533 | 1.6919480562210083 |
| sft.S96.0.SELF_O0 | 256 | 200 | 3072 | 1024 | 1.885222002329101e-05 | 0.0031632750760763884 |
| sft.S96.0.SELF_SINGLE | 256 | 209 | 3072 | 1024 | 2.493533298775219e-05 | 0.007195616140961647 |
| sft.S96.1.GOLD_ALL | 256 | 384 | 3072 | 1024 | 0.027696782855855417 | 2.798156261444092 |
| sft.S96.1.GOLD_MATCH_MIX | 256 | 293 | 3072 | 1024 | 0.002006622349668419 | 0.9283140897750854 |
| sft.S96.1.REPLAY_ONLY | 256 | 0 | 0 | 1024 | 2.199249684053939e-05 | 0.0019733791705220938 |
| sft.S96.1.SELF_MIX | 256 | 293 | 3072 | 1024 | 0.002343223460002264 | 0.3758239448070526 |
| sft.S96.1.SELF_O0 | 256 | 202 | 3072 | 1024 | 1.4871948451400385e-05 | 0.005741586443036795 |
| sft.S96.1.SELF_SINGLE | 256 | 211 | 3072 | 1024 | 8.718969874621507e-05 | 0.029263393953442574 |

两个repeat共享同一T任务集合，不能把四组当成1536个独立题目；预算前缀曲线也不是额外实验。每个非空SFT臂256步，focus应3072次、replay应1024次；REPLAY_ONLY只有1024次replay且损失仍使用统一1/16系数。曝光逐题分布与零曝光见CSV。

## Sentinel：训练集合内外的分解

末步loss是当步mini-batch损失，不是全T平均。将固定sentinel按该臂实际训练集合分开：四组SELF_O0均在J内6/6成功、J外0/10。SELF_MIX在J内依次为S96 r0 9/11、r1 12/12、REP96 r0 12/12、r1 10/11；仍有已训练题失败，应保留。该分解只含固定16题，不足以代表全训练集。

| parent | repeat | arm | group | N_tasks | successes | pX |
| --- | --- | --- | --- | --- | --- | --- |
| REP96 | 0 | GOLD_ALL | inside_training_J | 16 | 15 | 0.9375 |
| REP96 | 0 | GOLD_ALL | outside_training_J | 0 | 0 | None |
| REP96 | 0 | SELF_MIX | inside_training_J | 12 | 12 | 1.0 |
| REP96 | 0 | SELF_MIX | outside_training_J | 4 | 0 | 0.0 |
| REP96 | 0 | SELF_O0 | inside_training_J | 6 | 6 | 1.0 |
| REP96 | 0 | SELF_O0 | outside_training_J | 10 | 0 | 0.0 |
| REP96 | 0 | SELF_SINGLE | inside_training_J | 6 | 6 | 1.0 |
| REP96 | 0 | SELF_SINGLE | outside_training_J | 10 | 2 | 0.2 |
| REP96 | 1 | GOLD_ALL | inside_training_J | 16 | 14 | 0.875 |
| REP96 | 1 | GOLD_ALL | outside_training_J | 0 | 0 | None |
| REP96 | 1 | SELF_MIX | inside_training_J | 11 | 10 | 0.9090909090909091 |
| REP96 | 1 | SELF_MIX | outside_training_J | 5 | 1 | 0.2 |
| REP96 | 1 | SELF_O0 | inside_training_J | 6 | 6 | 1.0 |
| REP96 | 1 | SELF_O0 | outside_training_J | 10 | 0 | 0.0 |
| REP96 | 1 | SELF_SINGLE | inside_training_J | 6 | 6 | 1.0 |
| REP96 | 1 | SELF_SINGLE | outside_training_J | 10 | 1 | 0.1 |
| S96 | 0 | GOLD_ALL | inside_training_J | 16 | 14 | 0.875 |
| S96 | 0 | GOLD_ALL | outside_training_J | 0 | 0 | None |
| S96 | 0 | GOLD_MATCH_MIX | inside_training_J | 12 | 12 | 1.0 |
| S96 | 0 | GOLD_MATCH_MIX | outside_training_J | 4 | 1 | 0.25 |
| S96 | 0 | REPLAY_ONLY | inside_training_J | 0 | 0 | None |
| S96 | 0 | REPLAY_ONLY | outside_training_J | 16 | 6 | 0.375 |
| S96 | 0 | SELF_MIX | inside_training_J | 11 | 9 | 0.8181818181818182 |
| S96 | 0 | SELF_MIX | outside_training_J | 5 | 1 | 0.2 |
| S96 | 0 | SELF_O0 | inside_training_J | 6 | 6 | 1.0 |
| S96 | 0 | SELF_O0 | outside_training_J | 10 | 0 | 0.0 |
| S96 | 0 | SELF_SINGLE | inside_training_J | 7 | 7 | 1.0 |
| S96 | 0 | SELF_SINGLE | outside_training_J | 9 | 0 | 0.0 |
| S96 | 1 | GOLD_ALL | inside_training_J | 16 | 14 | 0.875 |
| S96 | 1 | GOLD_ALL | outside_training_J | 0 | 0 | None |
| S96 | 1 | GOLD_MATCH_MIX | inside_training_J | 10 | 10 | 1.0 |
| S96 | 1 | GOLD_MATCH_MIX | outside_training_J | 6 | 0 | 0.0 |
| S96 | 1 | REPLAY_ONLY | inside_training_J | 0 | 0 | None |
| S96 | 1 | REPLAY_ONLY | outside_training_J | 16 | 6 | 0.375 |
| S96 | 1 | SELF_MIX | inside_training_J | 12 | 12 | 1.0 |
| S96 | 1 | SELF_MIX | outside_training_J | 4 | 1 | 0.25 |
| S96 | 1 | SELF_O0 | inside_training_J | 6 | 6 | 1.0 |
| S96 | 1 | SELF_O0 | outside_training_J | 10 | 0 | 0.0 |
| S96 | 1 | SELF_SINGLE | inside_training_J | 8 | 8 | 1.0 |
| S96 | 1 | SELF_SINGLE | outside_training_J | 8 | 0 | 0.0 |

## V与训练sentinel

| parent | repeat | arm | split | step | N | draws_per_task | pX_family_equal | trend | cross_series |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REP96 | 0 | GOLD_ALL | V_selection | 128 | 96 | 4 | 0.515625 | 0.40625 | 0.625 |
| REP96 | 0 | GOLD_ALL | V_selection | 64 | 96 | 4 | 0.3723958333333333 | 0.21875 | 0.5260416666666666 |
| REP96 | 0 | R0_RESET32 | V_selection | 8 | 96 | 4 | 0.4401041666666667 | 0.359375 | 0.5208333333333334 |
| REP96 | 0 | SELF_MIX | V_selection | 128 | 96 | 4 | 0.4869791666666667 | 0.40625 | 0.5677083333333334 |
| REP96 | 0 | SELF_MIX | V_selection | 64 | 96 | 4 | 0.41145833333333337 | 0.2916666666666667 | 0.53125 |
| REP96 | 0 | SELF_O0 | V_selection | 128 | 96 | 4 | 0.48697916666666663 | 0.4114583333333333 | 0.5625 |
| REP96 | 0 | SELF_O0 | V_selection | 64 | 96 | 4 | 0.47395833333333337 | 0.4010416666666667 | 0.546875 |
| REP96 | 0 | SELF_SINGLE | V_selection | 128 | 96 | 4 | 0.4921875 | 0.4479166666666667 | 0.5364583333333334 |
| REP96 | 0 | SELF_SINGLE | V_selection | 64 | 96 | 4 | 0.44791666666666663 | 0.3854166666666667 | 0.5104166666666666 |
| REP96 | 1 | GOLD_ALL | V_selection | 128 | 96 | 4 | 0.5286458333333334 | 0.4322916666666667 | 0.625 |
| REP96 | 1 | GOLD_ALL | V_selection | 64 | 96 | 4 | 0.38020833333333337 | 0.2604166666666667 | 0.5 |
| REP96 | 1 | R0_RESET32 | V_selection | 8 | 96 | 4 | 0.44010416666666663 | 0.3541666666666667 | 0.5260416666666666 |
| REP96 | 1 | SELF_MIX | V_selection | 128 | 96 | 4 | 0.4921875 | 0.4114583333333333 | 0.5729166666666666 |
| REP96 | 1 | SELF_MIX | V_selection | 64 | 96 | 4 | 0.40364583333333337 | 0.2708333333333333 | 0.5364583333333334 |
| REP96 | 1 | SELF_O0 | V_selection | 128 | 96 | 4 | 0.4739583333333333 | 0.390625 | 0.5572916666666666 |
| REP96 | 1 | SELF_O0 | V_selection | 64 | 96 | 4 | 0.48177083333333337 | 0.4114583333333333 | 0.5520833333333334 |
| REP96 | 1 | SELF_SINGLE | V_selection | 128 | 96 | 4 | 0.48958333333333337 | 0.4270833333333333 | 0.5520833333333334 |
| REP96 | 1 | SELF_SINGLE | V_selection | 64 | 96 | 4 | 0.45052083333333337 | 0.3854166666666667 | 0.515625 |
| S96 | 0 | GOLD_ALL | V_selection | 128 | 96 | 4 | 0.5442708333333334 | 0.3958333333333333 | 0.6927083333333334 |
| S96 | 0 | GOLD_ALL | V_selection | 64 | 96 | 4 | 0.38802083333333337 | 0.28125 | 0.4947916666666667 |
| S96 | 0 | GOLD_MATCH_MIX | V_selection | 128 | 96 | 4 | 0.49479166666666663 | 0.4270833333333333 | 0.5625 |
| S96 | 0 | GOLD_MATCH_MIX | V_selection | 64 | 96 | 4 | 0.375 | 0.2604166666666667 | 0.4895833333333333 |
| S96 | 0 | R0_RESET32 | V_selection | 8 | 96 | 4 | 0.4739583333333333 | 0.40625 | 0.5416666666666666 |
| S96 | 0 | REPLAY_ONLY | V_selection | 128 | 96 | 4 | 0.42447916666666663 | 0.3333333333333333 | 0.515625 |
| S96 | 0 | REPLAY_ONLY | V_selection | 64 | 96 | 4 | 0.4192708333333333 | 0.3125 | 0.5260416666666666 |
| S96 | 0 | SELF_MIX | V_selection | 128 | 96 | 4 | 0.4765625 | 0.3802083333333333 | 0.5729166666666666 |
| S96 | 0 | SELF_MIX | V_selection | 64 | 96 | 4 | 0.3802083333333333 | 0.2864583333333333 | 0.4739583333333333 |
| S96 | 0 | SELF_O0 | V_selection | 128 | 96 | 4 | 0.4921875 | 0.421875 | 0.5625 |
| S96 | 0 | SELF_O0 | V_selection | 64 | 96 | 4 | 0.484375 | 0.4166666666666667 | 0.5520833333333334 |
| S96 | 0 | SELF_SINGLE | V_selection | 128 | 96 | 4 | 0.5130208333333334 | 0.4583333333333333 | 0.5677083333333334 |
| S96 | 0 | SELF_SINGLE | V_selection | 64 | 96 | 4 | 0.484375 | 0.4375 | 0.53125 |
| S96 | 1 | GOLD_ALL | V_selection | 128 | 96 | 4 | 0.5260416666666667 | 0.390625 | 0.6614583333333334 |
| S96 | 1 | GOLD_ALL | V_selection | 64 | 96 | 4 | 0.37760416666666663 | 0.2864583333333333 | 0.46875 |
| S96 | 1 | GOLD_MATCH_MIX | V_selection | 128 | 96 | 4 | 0.45572916666666663 | 0.3802083333333333 | 0.53125 |
| S96 | 1 | GOLD_MATCH_MIX | V_selection | 64 | 96 | 4 | 0.44010416666666663 | 0.3489583333333333 | 0.53125 |
| S96 | 1 | R0_RESET32 | V_selection | 8 | 96 | 4 | 0.45833333333333337 | 0.4010416666666667 | 0.515625 |
| S96 | 1 | REPLAY_ONLY | V_selection | 128 | 96 | 4 | 0.421875 | 0.3229166666666667 | 0.5208333333333334 |
| S96 | 1 | REPLAY_ONLY | V_selection | 64 | 96 | 4 | 0.40885416666666663 | 0.3177083333333333 | 0.5 |
| S96 | 1 | SELF_MIX | V_selection | 128 | 96 | 4 | 0.4765625 | 0.3854166666666667 | 0.5677083333333334 |
| S96 | 1 | SELF_MIX | V_selection | 64 | 96 | 4 | 0.421875 | 0.296875 | 0.546875 |
| S96 | 1 | SELF_O0 | V_selection | 128 | 96 | 4 | 0.4921875 | 0.4270833333333333 | 0.5572916666666666 |
| S96 | 1 | SELF_O0 | V_selection | 64 | 96 | 4 | 0.48177083333333337 | 0.4270833333333333 | 0.5364583333333334 |
| S96 | 1 | SELF_SINGLE | V_selection | 128 | 96 | 4 | 0.5130208333333333 | 0.453125 | 0.5729166666666666 |
| S96 | 1 | SELF_SINGLE | V_selection | 64 | 96 | 4 | 0.4765625 | 0.421875 | 0.53125 |
| REP96 | 0 | GOLD_ALL | T_train | 128 | 16 | 1 | 0.5 | 0.375 | 0.625 |
| REP96 | 0 | GOLD_ALL | T_train | 256 | 16 | 1 | 0.9375 | 0.875 | 1.0 |
| REP96 | 0 | GOLD_ALL | T_train | 64 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| REP96 | 0 | SELF_MIX | T_train | 128 | 16 | 1 | 0.5 | 0.375 | 0.625 |
| REP96 | 0 | SELF_MIX | T_train | 256 | 16 | 1 | 0.75 | 0.875 | 0.625 |
| REP96 | 0 | SELF_MIX | T_train | 64 | 16 | 1 | 0.3125 | 0.375 | 0.25 |
| REP96 | 0 | SELF_O0 | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 0 | SELF_O0 | T_train | 256 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 0 | SELF_O0 | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 0 | SELF_SINGLE | T_train | 128 | 16 | 1 | 0.5 | 0.375 | 0.625 |
| REP96 | 0 | SELF_SINGLE | T_train | 256 | 16 | 1 | 0.5 | 0.375 | 0.625 |
| REP96 | 0 | SELF_SINGLE | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 1 | GOLD_ALL | T_train | 128 | 16 | 1 | 0.375 | 0.25 | 0.5 |
| REP96 | 1 | GOLD_ALL | T_train | 256 | 16 | 1 | 0.875 | 0.875 | 0.875 |
| REP96 | 1 | GOLD_ALL | T_train | 64 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| REP96 | 1 | SELF_MIX | T_train | 128 | 16 | 1 | 0.5 | 0.5 | 0.5 |
| REP96 | 1 | SELF_MIX | T_train | 256 | 16 | 1 | 0.6875 | 0.875 | 0.5 |
| REP96 | 1 | SELF_MIX | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 1 | SELF_O0 | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 1 | SELF_O0 | T_train | 256 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 1 | SELF_O0 | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| REP96 | 1 | SELF_SINGLE | T_train | 128 | 16 | 1 | 0.3125 | 0.375 | 0.25 |
| REP96 | 1 | SELF_SINGLE | T_train | 256 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| REP96 | 1 | SELF_SINGLE | T_train | 64 | 16 | 1 | 0.3125 | 0.375 | 0.25 |
| S96 | 0 | GOLD_ALL | T_train | 128 | 16 | 1 | 0.5 | 0.375 | 0.625 |
| S96 | 0 | GOLD_ALL | T_train | 256 | 16 | 1 | 0.875 | 0.875 | 0.875 |
| S96 | 0 | GOLD_ALL | T_train | 64 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| S96 | 0 | GOLD_MATCH_MIX | T_train | 128 | 16 | 1 | 0.3125 | 0.375 | 0.25 |
| S96 | 0 | GOLD_MATCH_MIX | T_train | 256 | 16 | 1 | 0.8125 | 0.875 | 0.75 |
| S96 | 0 | GOLD_MATCH_MIX | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | REPLAY_ONLY | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | REPLAY_ONLY | T_train | 256 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | REPLAY_ONLY | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | SELF_MIX | T_train | 128 | 16 | 1 | 0.4375 | 0.5 | 0.375 |
| S96 | 0 | SELF_MIX | T_train | 256 | 16 | 1 | 0.625 | 0.75 | 0.5 |
| S96 | 0 | SELF_MIX | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | SELF_O0 | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | SELF_O0 | T_train | 256 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | SELF_O0 | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | SELF_SINGLE | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 0 | SELF_SINGLE | T_train | 256 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| S96 | 0 | SELF_SINGLE | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | GOLD_ALL | T_train | 128 | 16 | 1 | 0.375 | 0.25 | 0.5 |
| S96 | 1 | GOLD_ALL | T_train | 256 | 16 | 1 | 0.875 | 0.875 | 0.875 |
| S96 | 1 | GOLD_ALL | T_train | 64 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| S96 | 1 | GOLD_MATCH_MIX | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | GOLD_MATCH_MIX | T_train | 256 | 16 | 1 | 0.625 | 0.625 | 0.625 |
| S96 | 1 | GOLD_MATCH_MIX | T_train | 64 | 16 | 1 | 0.3125 | 0.375 | 0.25 |
| S96 | 1 | REPLAY_ONLY | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | REPLAY_ONLY | T_train | 256 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | REPLAY_ONLY | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | SELF_MIX | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | SELF_MIX | T_train | 256 | 16 | 1 | 0.8125 | 1.0 | 0.625 |
| S96 | 1 | SELF_MIX | T_train | 64 | 16 | 1 | 0.3125 | 0.375 | 0.25 |
| S96 | 1 | SELF_O0 | T_train | 128 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | SELF_O0 | T_train | 256 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | SELF_O0 | T_train | 64 | 16 | 1 | 0.375 | 0.375 | 0.375 |
| S96 | 1 | SELF_SINGLE | T_train | 128 | 16 | 1 | 0.4375 | 0.375 | 0.5 |
| S96 | 1 | SELF_SINGLE | T_train | 256 | 16 | 1 | 0.5 | 0.375 | 0.625 |
| S96 | 1 | SELF_SINGLE | T_train | 64 | 16 | 1 | 0.3125 | 0.375 | 0.25 |

T_train sentinel仅16题×1次，是固定训练拟合诊断；V为96题×4次，步骤冻结，不按曲线选终点。所有不利值保留。

文件索引：discovery_fit_summary.csv（集合）、budget_prefix.csv（预算）、strata.csv（分层）、tasks.csv（逐题集合标志）、gold_matching.csv（匹配）、training_curves.csv（逐步loss/grad）、training_summary.csv、exposures.csv（逐题曝光）、exposure_summary.csv、validation_sentinel.csv。
