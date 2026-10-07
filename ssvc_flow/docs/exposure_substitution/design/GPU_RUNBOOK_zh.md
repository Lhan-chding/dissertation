# Codex接入与4–5张PRO6000执行手册

## 1. 使用现存源码与现存父模型

结果归档记录的服务器根：

`/projects/varunssd/louis-ssvc/verified_discovery_transfer_20261006/run_v2`

这是历史记录，不代表本次已经登录服务器检查。用户可能迁移过目录；入口接受`--legacy-run-root`和显式路径映射，从原parent_catalog/COMMIT恢复S96、REP96，不猜一个同名学生权重。

GitHub本次读到的VDT分支HEAD：`e375caa9118645f5f93aa76705f634c971e1fb7d`。原实际运行源码冻结为`ed1d24839dc6bd374d17a92bd29678b7185c7605`。对训练/生成相关文件做一次差异核对，建立新worktree，不覆盖VDT产物。

父缺失时不训练替代父。先报告不可用范围；其他可用父上的完整A/B/C可以推进，但不能在最终平均中默默补值。

## 2. 必须新增的接口，不是现成可运行命令

以下是Codex需要实现并测试的CLI合同，当前仓库尚未由本次助手修改：

```
python -m ssvc_flow.src.exposure_substitution.cli prepare \
  --legacy-run-root "$LEGACY" --plan-dir "$PLAN" --out "$RUN"

python -m ssvc_flow.src.exposure_substitution.cli cpu-check --run "$RUN"

python -m ssvc_flow.src.exposure_substitution.cli bridge \
  --run "$RUN" --parent S96 --allow-gpu

python -m ssvc_flow.src.exposure_substitution.cli train \
  --run "$RUN" --parent S96 --block 0 --arm A_LOCAL_C1 --allow-gpu

python -m ssvc_flow.src.exposure_substitution.cli eval-diagnostic \
  --run "$RUN" --job <actual_job_id> --step 128 --allow-gpu

python -m ssvc_flow.src.exposure_substitution.cli eval-confirm-sealed \
  --run "$RUN" --job <actual_job_id> --step 256 --allow-gpu

python -m ssvc_flow.src.exposure_substitution.cli release-and-analyze --run "$RUN"
```

本包现成能跑的是`reference/build_manifests.py`和`reference/test_contracts.py`，不是上述真实GPU接口。不要把CPU单元测试日志称为实验已接入。

## 3. 顺序

1. 读取主文档，核对原T/replay/父来源以及新增历史排除。
2. 按reference重建manifest，比较数量、目标、共同slot与本包文件；先报告差异再统一冻结。
3. GPU单次bridge：8步连续/恢复、donor隔离microbatch、数据mask、更新与推理还原。
4. S96/block0三个臂先完成；检查吞吐、学习曲线、诊断面板。
5. 继续其余预定block，不按第一组赢家删除条件。
6. E_CONFIRM可以随checkpoint完成而密封生成，但不向训练进程返回聚合评分或用于选择。
7. 所有登记主终点终止后统一揭晓。技术缺项单列；不能因结果不利把作业改为异常移出。

## 4. 并行

一个GPU运行一个独立模型作业。五卡可以同时运行不同parent/block/arm；按固定随机任务队列轮转臂和物理卡，避免每个B都固定在同一异常卡。四卡时减少并发，不改变batch、步数或种子。

训练/评估启动前检查实际`nvidia-smi`、显存、文件可读与磁盘空间。具体RTX PRO6000型号、显存和可用QOS以当前分配为准，不在文档中虚构排队时间。

默认SFT五个microbatch：4 common、4 common、3 common、1 donor、4 replay。相比旧4个batch多一次forward/backward，但避免供体长度改变共同样本padding。不得为提速把它们重新按长度跨角色排序。

评估默认沿用已验证生成后端。同prompt/任务多次采样可缓存输入token；不得缓存一次回答来替代独立抽样。不切换推理引擎追求尚未验证的加速。

## 5. 工期与显存

首次完整三臂结束后，由实测：

- 每步训练GPU秒、每千token生成秒；
- 每个checkpoint载入时间、内存峰值；
- 平均prompt/target长度、实际生成终止长度；

估计剩余GPU小时和并发墙钟。不要用“5张卡足够”代替测量，也不预报未经测量的总天数。

固定工作量：4608正式SFT更新、73728监督曝光、145280正式生成回答。技术bridge上限16更新与256回答单列。没有新增发现matrix和RL rollout。

## 6. 检查点和恢复

- 主训练checkpoint：0/64/128/192/256。
- 每份包括原base绑定、LoRA、优化器、scheduler/step、RNG、forward状态、schedule ID和cursor。
- 原子写入临时文件成功后再提交；不要覆盖原父。
- 恢复读取最近完整提交；同数据同slot从该step继续，不重新shuffle。
- 中断后重算的真实工作记入attempt日志；正式统计每个逻辑update只取最终正确提交。
- `fresh AdamW`只能用于新分叉，不能用于resume。
- 若当前梯度确为零，保留显式零梯度并执行原optimizer逻辑；None与0不等价。

## 7. 日志与隐私/角色

训练只能访问COMMON_TRAIN/DONOR_TRAIN/REPLAY与自己的目标。评估worker只向模型传公开prompt；主审计标签交给隔离的评分流程。

模型输入中不能出现root_id、arm、center审计字段、corrupted_index、truth或E_confirm结果。center可由公开关系读出，但内部辅助字段仍不作为额外自然语言提示。

每step日志记录：slot、role、task、donor root、target token数、序列NLL、clip前范数、lr、实际步号、轻量更新摘要。不同臂的序列NLL不同是允许的。

逐样本不做完整模型哈希。开始一次身份绑定、阶段末一次完整性统计、故障时定点检查足够。源文件和模型权重不随最终小报告反复打包，给出明确保留位置。

## 8. 技术错误与科学结果

可停止对应作业：非有限数、源权重错载、数据泄漏、固定slot失配、目标mask错误、数值域或唯一解契约破坏、恢复状态不同。

不能停止/删除对应臂：准确率下降、供体未明显学会、投影无增益、无显著效应、符号与预期相反。它们正是实验结果。

## 9. 运行库参考（仅说明机制，不要求升级）

本次核读PyTorch官方reproducibility文档与AdamW文档：同seed不保证跨设备/版本完全一致；0梯度与None会导致优化器不同处理。实际运行保留现存环境，以一次短桥接确认，不为追求文档版本而自动升级。

- https://docs.pytorch.org/docs/2.13/notes/randomness.html
- https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW
