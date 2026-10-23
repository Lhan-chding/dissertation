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
- 2026-10-22按用户要求改为分阶段评测：旧回归面板的父模型立即可评，训练臂只等自身step64完成；仅B_CONFIRM保留全部A与B3结束的依赖。评测题目、种子、draw和检查点均保持不变。
- 每卡申请96 GiB主机内存、8CPU；集群提交插件实际将双卡作业设为8CPU、66GiB主机内存，保留其原始提交回执。每个模型进程只看见一张卡。进程保留完整异常记录；不以退出码或作业结束代替样本完整性。
- GPU租约缩短为6小时以改善backfill；CPU控制器 `LH-SRF4-CTRL` 每150秒核对终态，在已释放的TIMEOUT之后恢复并补足额度。控制器占第五个作业名额，没有GPU。三次启动失败会明确阻断，FAILED任务不会被自动洗成重试。
- 控制器的自身租约为3天；超过3天仍未完成时，需要重提同一 `controller.sbatch` 接续已有池。GPU租约不是科学停止预算。

## 启动和状态

代码快照保存到运行根的 `code/`。首次准备在 CPU 作业执行 `python -m sr_f4.prepare --root RUN_ROOT --old-root OLD_ROOT --code CODE_ROOT`；随后 `python -m sr_f4.pool init --root RUN_ROOT`，再运行 `python -m sr_f4.pool submit --root RUN_ROOT --code CODE_ROOT --python PYTHON`。

部署归档统一使用 `PYTHONPATH=src python -m sr_f4.deployment --output CODE_SNAPSHOT.tar`。除了src，还必须包括继承的 `docs/sr_f1/package` 参考代码、配置及manifest和amendments；只打包src与sr_f4文档不够。准备/控制器脚本在提交GPU前仅确认生产模块能导入，不加载模型或生成回答。

持续接续由 `sbatch CODE_ROOT/scripts/sr_f4/controller.sbatch` 启动。初次数据准备作业218288已经COMPLETED；5,408题图像路由准备完成，确认集模型调用数为0。最初四个3天GPU作业218289–218292因Priority排队，集群拒绝原地更改TimeLimit，因此在仍未开跑时撤回后改为6小时作业；替换终态和新作业编号以服务器 `RESOURCE_AMENDMENT_6H.json`、`pool/STATE.json` 为准。此前尚无新GPU回答，不能将已提交写成已训练完成。

状态为 `python -m sr_f4.pool status --root RUN_ROOT`，每项科学任务保存在 `tasks/<task_id>/COMPLETE.json`，训练进度在 `RUNS/A/b*/r1/*/updates/`。调度任务完成不等于所有科学条件通过，应检查嵌套 result.status、B3 技术缺口和最终 coverage。可恢复 TIMEOUT 只在旧租约结束、调度器终态确认后重新接续。非超时失败须读实际异常、修复后另行登记；不自动洗掉失败。

`REPORT` 在所有登记端点评测完成后生成 `REPORT_ZH.md` 和 `ANALYSIS/`，`PACKAGE` 随后一次写出归档与 `RESULT_MANIFEST.json`，不做额外哈希或重复读回。归档包含本轮原始回答、更新、检查点、分析与代码；继承的底模和父模型只引用服务器路径，不声称是自包含重训包。逐块、逐变体、负向保持差异及不确定性都保留。A 每块只有一个新训练重复，不能以大量回答替代训练重复数。C 未运行不能解释为语义信用无效。

## 存储清理

本轮清理限定为已完成 F1.2/F3 运行的旧中间优化器检查点、已有完整 scored group 的重复 pending 缓存、三处旧 CPU 测试临时目录以及可重建 conda/编译缓存。保留最终检查点和全部导出 adapter、父模型、原始已提交回答与更新指标、技术事件及发布结果。没有触碰其他用户或其他项目缓存。

操作清单和结果保存在服务器 `storage_audits/cleanup_20261022_srf4/receipt.json`。清理不执行哈希遍历。数据清单可含此前登记的身份字段；沿用现有身份不代表本轮重新计算全部文件哈希。

2026-10-22 已完成删除：89,308,871,077字节（83.18 GiB），其中SSD 57,374,523,733字节、HDD 31,934,347,344字节；12,533个清单条目，remaining=0。此处文件夹按一个条目计数，并非单个文件总数。

## 已完成的本地验证

针对本次实现的72项 CPU 测试通过（包含两项评测预算修正回归检查），执行包55项参考算法检查通过，Ruff 通过。详细覆盖与限制见 `VALIDATION.json`。这些结果不等于 GPU 模型运行成功、不等于历史 golden replay 已完成，也不代表科学结论成立。

后续控制器改动的23项 pool 专项测试通过，其中新增6项检查既有预约、CPU作业名额、超时续接、启动失败阻断和主动取消边界。

## 2026-10-22 首次启动故障与恢复

实查发现218310在新加坡时间02:50启动后，9个探针因部署缺少 `docs/sr_f1/package/PACKAGE_SHA256.json` 而在导入阶段失败，尚未加载模型或生成回答。其余双卡作业随后无就绪任务而退出；控制器218309于02:50停止。旧worker错误地返回0，导致Slurm显示COMPLETED而任务记录实际为FAILED。旧日志、9条失败attempt和控制器状态均保留，不能称作实验完成。

修复包括补齐继承资源、用独立部署目录复现并修复导入故障，以及传播真实子任务失败退出码。恢复沿用同一36项登记任务、数据、种子与回答/更新预算；仅将这9个尚未执行模型的FAILED任务恢复PENDING，保留其历史attempt。新的GPU作业仍由老师QoS下的CPU控制器补交，并按单卡独立接续。

## 2026-10-22 回归评测提前接续

用户要求空闲卡用于已有工作、每条训练完成即评测。规范03明确旧TEST_ID/COMPOSITION/ARITY/ChartQA属于已参与设计的回归面板；此前将全部评测与新B_CONFIRM放入同一全局依赖，限制过宽。现在把同一评测清单拆成回归与确认两个执行阶段：回归104,072条、确认2,688条，总数仍为106,760；训练仍为六路径各64步、每步128条。

原E任务负责非B_CONFIRM面板，PARENT可立即执行，其他臂只依赖自身A任务完成。新增EC任务只执行B_CONFIRM，仍依赖全部六条A和B3；直接调用确认入口也会检查这一条件。EC沿用原E的样本身份、全局group_index、分片、题目、seed、draw和输出组路径。两个阶段分别写PROGRESS_regression.json和PROGRESS_confirm.json；REPORT等待两阶段全部完成，不使用提前评测结果修改训练。调度记录由36项拆成54项，不增加实验回答。

运行中的六个单卡训练进程保持原样。直接改为多卡训练会涉及进程重启和梯度归约顺序，当前通过提前执行固定回归评测利用另外两卡。仅原子替换评测相关模块后，运行 `python -m sr_f4.scheduling --root RUN_ROOT` 迁移尚未启动的评测任务；保留运行状态、旧attempt及迁移前的TASKS/STATE，记录 `SCHEDULING_AMENDMENT_PIPELINED_EVAL.json`。迁移拒绝已有评测输出或已开放确认集的情况。

## 2026-10-23 报告读取提速

全部106,760条评测于新加坡17:28完成。报告输入是32,576个完整group文件，合计4,803,749,061字节，另有36个进度JSON；实查没有残留chunk文件或重复缓存需要清理。原报告串行读取小文件，12分钟累计读取约3.35GB、CPU仅约106秒，主要等待文件系统。

按用户要求，将报告读取改为8个并发reader、每批最多16个文件；每批仍按原排序返回，JSONL先于JSON，组内行顺序不变。只在主线程进行原有标准化及重复身份检查，不改任何指标、覆盖分母、配对world、bootstrap随机流或置信区间算法。保留workers=1用于等价检查，不新增模型调用或实验回答。测试比较串行/并行全部标准化字段与顺序，并检查重复身份仍报错、chunk仍排除和预取有界。若部署时原报告已经完成，则保留原结果，不为提速重跑；如需切换未完成报告，保留原attempt和技术切换记录后只恢复REPORT，PACKAGE继续依赖其完成。
