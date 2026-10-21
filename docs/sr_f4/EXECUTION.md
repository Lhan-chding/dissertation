# SR-F4 执行记录

依据用户提供的 `SRF4_CODEX_EXECUTION_PACKAGE_20261022.zip`；原始规范保存在 `spec/`，生产接线在 `src/sr_f4/`，包内参考算法在 `src/srf4_ref/`。用户本次明确省去重复哈希、额外烟测，要求老师 QoS、LH 前缀、可用额度内多卡并行、单卡完成后立即接续，以及清理不再使用的旧数据。执行时以这些直接要求优先。

## 冻结范围与边界

A 固定为三个父模型块、每块均匀采样与 LZ 两条路径，各64次更新，每次128条回答。B0、B2_DEV 与 C_TRAIN_PROBE 属于包内已登记测量。当前没有真实人工盲核，C 的24条训练路径登记为 `SKIPPED_BY_PREREG_GATE`，不启动、不以自动标签充当人工。B3 生成盲核材料和具体门控报告。A/B 可继续。CONFIRM 在全部 A 路径及 B3 决策结束前不调用模型。

复用原生二元 J、sequence loss、KL/TIS、优化器与父 LoRA。G32 合并成一个优势组；128条回答全部使用更新前策略，只更新一次。每个 draw 有独立随机流，G32 拆为四个高效 G8 推理批。原生实现和旧实验目录不修改。

B_CONFIRM 父模型按B测量要求每题8 draw，共1,344条；A两训练臂每题4 draw，A对父模型的配对统计只取共同前4 draw。ChartQA只在b0两训练臂各2,500题，不新增父模型ChartQA。全部登记终点评测共106,760条；加开发探针与A训练，本轮C被阻断时共165,704条计划回答，技术重放另记原因和次数。

## 服务器与调度

- 新运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_f4_consolidated_20261022`
- 旧原始数据及父模型：`/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2`
- Python：`/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12`
- QoS：`soujanya-poria-startfund-2026-03`；account=`rose`；partition=`cluster02`。
- 实时额度：共享 QoS 最多8张 PRO6000，本用户同时提交上限5个作业；计算其他用户占用、pending 和未知提交结果。
- GPU 作业名称为 `LH-SRF4-*`，通常四个双卡作业，各卡独立领取任务；每完成一个任务立即领取下一项就绪任务，没有人为批次屏障。A 路径只依赖本块 B0；确认评测的依赖属于科学冻结要求。
- 每卡申请96 GiB主机内存、8CPU；每个模型进程只看见一张卡。进程保留完整异常记录；不以退出码或作业结束代替样本完整性。

## 启动和状态

代码快照保存到运行根的 `code/`。首次准备在 CPU 作业执行 `python -m sr_f4.prepare --root RUN_ROOT --old-root OLD_ROOT --code CODE_ROOT`；随后 `python -m sr_f4.pool init --root RUN_ROOT`，再运行 `python -m sr_f4.pool submit --root RUN_ROOT --code CODE_ROOT --python PYTHON`。

状态为 `python -m sr_f4.pool status --root RUN_ROOT`，每项科学任务保存在 `tasks/<task_id>/COMPLETE.json`，训练进度在 `RUNS/A/b*/r1/*/updates/`。调度任务完成不等于所有科学条件通过，应检查嵌套 result.status、B3 技术缺口和最终 coverage。可恢复 TIMEOUT 只在旧租约结束、调度器终态确认后重新接续。非超时失败须读实际异常、修复后另行登记；不自动洗掉失败。

`REPORT` 在所有登记端点评测完成后生成 `REPORT_ZH.md` 和 `ANALYSIS/`，`PACKAGE` 随后一次写出归档与 `RESULT_MANIFEST.json`，不做额外哈希或重复读回。归档包含本轮原始回答、更新、检查点、分析与代码；继承的底模和父模型只引用服务器路径，不声称是自包含重训包。逐块、逐变体、负向保持差异及不确定性都保留。A 每块只有一个新训练重复，不能以大量回答替代训练重复数。C 未运行不能解释为语义信用无效。

## 存储清理

本轮清理限定为已完成 F1.2/F3 运行的旧中间优化器检查点、已有完整 scored group 的重复 pending 缓存、三处旧 CPU 测试临时目录以及可重建 conda/编译缓存。保留最终检查点和全部导出 adapter、父模型、原始已提交回答与更新指标、技术事件及发布结果。没有触碰其他用户或其他项目缓存。

操作清单和结果保存在服务器 `storage_audits/cleanup_20261022_srf4/receipt.json`。清理不执行哈希遍历。数据清单可含此前登记的身份字段；沿用现有身份不代表本轮重新计算全部文件哈希。

2026-10-22 已完成删除：89,308,871,077字节（83.18 GiB），其中SSD 57,374,523,733字节、HDD 31,934,347,344字节；12,533个清单条目，remaining=0。此处文件夹按一个条目计数，并非单个文件总数。

## 已完成的本地验证

针对本次实现的72项 CPU 测试通过（包含两项评测预算修正回归检查），执行包55项参考算法检查通过，Ruff 通过。详细覆盖与限制见 `VALIDATION.json`。这些结果不等于 GPU 模型运行成功、不等于历史 golden replay 已完成，也不代表科学结论成立。
