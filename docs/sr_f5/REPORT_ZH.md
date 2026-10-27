# SR-F5 recipe_v2 最终可实施性诊断

核验日期：2026-10-27（新加坡）。结论：**G3 科学可实施性 FAIL，依 contract/05、07 停止，不进入正式 45 路矩阵。** 这不是“计算未跑完”，也不是效果显著性检验；全部登记先导已完成，但五臂对照的实际比较机会未满足预定门槛。

## 完成状态与身份

科学 revision `47d5e85fb0c9b39d8f996465045d21956825f59d`，服务器冻结部署 `sr_f5_recipe_v2_47d5e85`。运行根 `/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026`；冻结 digest `e3849e3a3250358e5dd55ba0bc59d95e29ded924d8f4e3c6bea78126b29dbca0`。

全模型、浮点输入及 generation/current/reference 统一 FP32。开发数据沿登记派生种子，臂、阈值、预算与原 POOL_PLAN 未变。b0/b1/b2 为三个登记父 adapter；本先导每父模型只含一次开发运行，不能冒充正式的三训练重复或九个配对效果。

- G2：4,608 回答真实 PASS，未重新汇总；既有 CPU 门 332 passed、12 skipped、9 subtests 复用，跳过项不计 PASS。
- G3：15 条 primary 均连续从第 0 步完成 12 步，各 1,536 回答，共 **23,040**；5 条 b0 fresh recovery 从第 4 步至 12 步，各 1,024 回答，共 **5,120**。总计 **28,160/28,160**，没有缺失路径。
- b0 五臂恢复各 17 项检查为 true，每臂 128 组精确匹配，包含模型、Adam、scheduler、RNG、原始 token、概率、信用和更新记录。b1/b2 不声称独立恢复重采样。
- GPU 228108 于 08:05:34、228121 于 08:18:52 新加坡时间结束，主作业、batch、extern 均 `COMPLETED/0:0`，已退出队列。两作业均在原时限内完成，未更改 TimeLimit，无额外 GPU 提交。
- 最终 CPU 汇总 229269 `COMPLETED/0:0`，76 秒；从原冻结源码读取原始回答、信用、更新与恢复回执。360 个预指定唯一末步回答的**程序审核**全部 PASS，不冒称人工逐条审核；证据中的继承充分性规则也不冒称独立证明。
- G0/G1/G4 的最终门回执仍为 NOT_RUN，占位未改成 PASS；正式训练、正式测试输出和正式效果统计均未启动。

## 按冻结门槛逐项判定

每个父模型的三个窗口依次为第 1–4、5–8、9–12 步。LOCAL 每窗口至少 4 个非零组，SHUF 每窗口至少 2 个实际改变组；整段支持至少 4 家族、12 roots。HARD 必须产生实际 J 与 J×1[S=PASS] 奖励差异。

|父模型|LOCAL 非零组（三窗口）|SHUF 实际改变组（三窗口）|信号家族 / roots|HARD 奖励改变行|最低单臂 S 可观测率|
|---|---|---|---|---|---|
|b0|14/13/11|5/3/3|6/22|0|99.09%|
|b1|20/7/9|10/1/1|5/18|1|97.20%|
|b2|24/14/9|11/6/4|6/25|3|98.63%|

**失败一：b0 HARD 没有实际奖励差异。** 其 1,536 行中，J 与 J×1[S=PASS] 改变行、改变组及 native advantage 改变组均为 0。该父模型全五臂汇总也未出现 S=FAIL/J=1；现有分布不能提供预登记 HARD 与 J 的实际奖励对照。b1、b2 HARD 分别仅 1、3 行发生改变；不能用它们替代 b0 门。

**失败二：b1 SHUF 持续对照不足。** 后两窗口各仅 1 个实际改变组，低于每窗口至少 2 个的门槛。全程总计 12 个改变组、存在非零梯度和后续干预，并不能补足两个指定窗口的失败。禁止合并窗口、改阈值或筛选 seed/token 追认为通过。

其他登记信号：LOCAL 三父模型的全部窗口满足最低数；信号家族/root 覆盖满足；三个父模型均出现干预→更新→各自后续生成→后续干预。全部单臂 S 可观测率高于 90%。DENSE/LOCAL/SHUF 在登记诊断步均具备实际非零辅助梯度证据；所有臂均有实际非零梯度及参数更新。这些有利事实仍不足以抵消上述失败。

原 `GATES/G3.json` 保留四条错误文本，其中 `numerical/runtime checks missing` 是汇总器把 b0 HARD 零奖励差异放入 errors 后生成的总括标记；**不能据此声称 FP32 出现 NaN、概率尾部报警或恢复失败。** METRICS 的细分 errors 仅列 b0 HARD 奖励差异；另外两条 SHUF 失败来自冻结窗口检查。未发现 G3/FAILURES.jsonl，最终判定来自完整先导的科学门。

独立读取原始回答及逐步 credit/updates 后，失败计数与冻结汇总一致。全部 180 次主路径更新均为 FINITE、每步一次 optimizer update，参数更新范数均大于 0；sampler/train 单步均值的最大值为 1.1705009228535346e-6 nat，全部 token 的最大差为 0.003695249557495117 nat，TIS 触顶比例全部为 0。分别低于冻结 0.01、5 nat、1% 门槛。这支持本次失败是比较机会不足，而非已观察到的概率数值失稳。诊断读取不生成新回答。

## 单独保留的技术事故

首次 CPU 汇总 229267 在 cpu-2 上 `FAILED/1:0`（7 秒），错误为 `Recovery evidence is stale or belongs to another path`。冻结代码把 `stat().st_dev` 纳入检查点文件元数据；同一共享文件在 GPU 节点 device=48，在 cpu-2 为 43（登录节点可为42）。十份检查点的 size、mtime_ns、inode 未变，十份 COMPLETE 的冻结规范化摘要均匹配。

修复采用同节点重新验证，而非改冻结科学源码或放宽相等标准：保留全部原 RECOVERY 字节、失败日志和提交回执，在一个有独占锁的 CPU 作业内用原 `compare_recovery` 对五臂重做完整精确比较，随后立即调用原 aggregate。复核结果除 checkpoint_metadata 中的设备编号外，与原回执完全一致；五臂 17 项检查仍全部 true。没有加载基模、重新采样、optimizer 更新或重复 G2 aggregate。

原回执与重验证回执位于 `validation/RECIPE_V2_G3_FINAL/G3_SAME_NODE_REVALIDATION/`；失败与成功作业日志均保留。第一次防重检查曾因 G3 的 NOT_RUN 占位而阻止提交；读取确认后才允许首次汇总，没有重复 GPU 池。

## 冻结决策与结论边界

recipe_v2 是 contract/05 允许的唯一开发修订；本次真实科学可实施性失败后停止扩展。自动跟进 `sr-f5` 已删除，不再周期检查。没有第三版、换 seed、筛 token、单臂例外、追加步数、重跑不利窗口或正式 45 路训练。v1 数值失败与所有原始/失败证据继续保留，不恢复 v1。

本结果支持的结论是：**当前冻结数据、父模型与配方下，五臂设计未获得足够且持续的预定对照机会。** 它不证明 LOCAL 无效，不证明语义方法普遍无效，也没有证明 LOCAL 优于 J/HARD/SHUF/DENSE。未进行正式组合泛化、保持评测或显著性检验，不报告先导排名为方法优胜证据。

## 证据索引

- `validation/RECIPE_V2_G3_FINAL/REENTRY_ALL_COMPLETE_20261027.json`：20 条完成路径、五臂恢复、worker 退出和两个 GPU 作业终态。
- `validation/RECIPE_V2_G3_FINAL/METRICS.json`、`G3.json`：原冻结汇总及科学 FAIL；服务器对应 `G3/METRICS.json`、`GATES/G3.json`。
- `validation/RECIPE_V2_G3_FINAL/PROGRAM_REVIEWS.jsonl`：360 条程序核对，含方法与独立性限制。
- `validation/RECIPE_V2_G3_FINAL/G3-AGG-229267.log`、`G3-AGG-CHECK-229269.log`：技术事故和最终汇总。
- `validation/RECIPE_V2_G3_FINAL/G3_SAME_NODE_REVALIDATION/`：原始/重新验证恢复回执和设备编号诊断。
- 服务器 `G3/RUNS/<parent>/<arm>/{primary,recovery}/`：完整原文、token、概率、逐步 credit/updates、检查点；未因下载摘要而删除服务器原件。
- `validation/RECIPE_V2_G3_FINAL/G3_FINAL_DIAGNOSTIC_COUNTS.json`：独立逐路径/逐窗口计数、180 步数值摘要、可定位的 HARD 改变行及最终队列/accounting。
- 第一版诊断见 `NUMERICAL_DIAGNOSIS_RECIPE_V2_ZH.md`；本报告不将两版合并分母或结果。
