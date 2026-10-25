# SR-F5 执行状态（2026-10-25，新加坡）

**CPU 构建完成；G2 四卡普查已提交，等待队列优先级。尚无 GPU 测量结果，尚未开始正式训练。**

从 F4 提交 `750db3d` 建立独立分支 `sr-f5-semantic-j-v1`。旧模块、旧结果和原工作区修改保持原样。

本阶段接通原生评分、生产图像构建、新协议采样、父模型普查及单步训练数值门。正式 45 路训练仍受 G0–G4 约束；任何 NOT_RUN 都不会改写为 PASS。开发先导、恢复等价验证、正式保持面板和统计发布尚待实施及实测。

效率策略：不扫描基模权重；新图像只计算一次摘要；不重复生成独立烟测；同一模型加载中完成 tokenizer/processor 检查后直接执行登记 G2。四个工作进程现装入两个双卡作业，每张卡一个独立进程；新作业统一使用 `LH-` 前缀。账号全部 QoS 作业合计最多五个，并计入运行、排队及未知预约。其他作业不修改。

服务器 CPU 作业 `226030`（`SRF5-CPU-LH`）已 `COMPLETED / 0:0`。本地与服务器对应检查均为 168 passed、12 skipped、6 subtests passed；跳过项依赖未安装的旧私有包，不计作 PASS。另有自然接口别名回归的 9 项 measurement 测试通过。

已生成 780 roots、3,120 题、1,896 张独立图像；双 oracle、训练 holdout 扫描、跨分区隔离通过。与 705 条历史世界记录比较，碰撞为零。在任何新模型输出前冻结 128 道旧题、250 道 ChartQA 和 G2 普查规格。

|作业|名称|卡数|最近状态|
|---|---|---:|---|
|226056|LH-SRF5-G2-P0|2|PENDING / Resources|
|226057|LH-SRF5-G2-P1|2|PENDING / Priority|

四份等量分片各依次执行 b0/b1/b2，总计 4,608 条登记回答。均使用教师 QoS `soujanya-poria-startfund-2026-03`。旧单卡作业 `226033`–`226036` 均在排队状态取消，并核实终态后替换。计入其他作业 `225839` 后，GPU 作业共三个、五张运行或排队预约 GPU；其他作业未修改。清理 CPU 作业 `226060 / LH-SRF5-CLEANUP` 已正常完成，当前不再占作业名额。

G0/G1 的真实 GPU 身份、processor、token 与恢复证据尚未齐全；G2 尚未产生回答。G3 先导 runner 与恢复验证、G4 冻结、45 条正式训练路径、正式评测和统计报告尚未运行。CPU PASS 不是实验结果。

当前作业只运行 G2，不自动跳到 G3 或正式训练。全部分片完成后在部署目录运行 `python -m sr_f5.calibration --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025 --aggregate` 汇总，再按真实门结果继续。没有创建周期监控；排队预约不代表实际 GPU 利用率。

运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025`。G2 部署代码：`/projects/_ssd/varunssd/louis-ssvc/sr_f5_code_26e560b`，commit `26e560bac1a7c5b5563f52112bc1374685084b8d`。后续自然接口修复和调度保护位于 `78acca3`，未替换 G2 科学运行代码。双卡调度使用独立部署 `sr_f5_launcher_c3d37ac`（commit `c3d37ac`），四份分片及 4,608 条登记回答保持不变。

小型服务器回执已取回 `artifacts/sr_f5_initial_receipts/`，关键回执另存本目录 `validation/`。

本次存储清理已完成，限定在 `/projects/_ssd/varunssd/louis-ssvc`，没有操作其他项目或取消其他作业。清理作业 `226060` 为 `COMPLETED / 0:0`。删除 16,339 个可由已保留压缩包完整恢复的 F3 展开文件，以及指定旧 Qwen2.5-VL-3B 下载缓存中的 29 个文件或链接。逻辑数据合计 22,256,205,609 bytes（22.26 GB）；按最后一个硬链接的分配块统计为 22,260,400,640 bytes，不把共享文件系统的总空闲量变化归因于本次操作。保留归档另占约 298 MB，因此新增归档后净减占用约 21.96 GB（未扣少量审计记录）。

唯一末步模型、当前 Qwen3.5 模型、父适配器、环境、F4 数据和保持面板依赖均保留。13 个保护文件的大小、修改时间、inode 和设备号前后相同；16,368 个删除条目均已不存在，删除清单与回执计数一致。检查只读取一次候选 F3 内容、一次保留归档身份；不重复完整重建，不扫描当前基模权重，不对可重新下载缓存做全文件哈希。

清理证据：`validation/APPLY_RECEIPT.json`、`validation/POSTFLIGHT.json`、`validation/OPEN_HANDLES.json`。完整逐文件日志、允许清单及可恢复归档位于服务器 `storage_audits/cleanup_20261025_project_only/`。活动检查覆盖用户 Slurm 作业和登录节点可见进程；计算节点远程进程不可直接读取，此限制已明确保留，未宣称所有主机独占。

双卡调度的 13 项测试、清理保护的 19 项测试通过，相关 Ruff 和脚本语法检查通过。GPU 作业仍排队，尚无新的 GPU 科学结果。
