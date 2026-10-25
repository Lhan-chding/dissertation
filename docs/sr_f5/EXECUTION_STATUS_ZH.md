# SR-F5 执行状态（2026-10-25，新加坡）

**CPU 构建完成；G2 四卡普查已提交，等待队列优先级。尚无 GPU 测量结果，尚未开始正式训练。**

从 F4 提交 `750db3d` 建立独立分支 `sr-f5-semantic-j-v1`。旧模块、旧结果和原工作区修改保持原样。

本阶段接通原生评分、生产图像构建、新协议采样、父模型普查及单步训练数值门。正式 45 路训练仍受 G0–G4 约束；任何 NOT_RUN 都不会改写为 PASS。开发先导、恢复等价验证、正式保持面板和统计发布尚待实施及实测。

效率策略：不扫描基模权重；新图像只计算一次摘要；不重复生成独立烟测；同一模型加载中完成 tokenizer/processor 检查后直接执行登记 G2。默认最多四个单卡工作进程，账号全部 QoS 作业合计最多五个，并计入运行、排队及未知预约。其他作业不修改。

服务器 CPU 作业 `226030`（`SRF5-CPU-LH`）已 `COMPLETED / 0:0`。本地与服务器对应检查均为 168 passed、12 skipped、6 subtests passed；跳过项依赖未安装的旧私有包，不计作 PASS。另有自然接口别名回归的 9 项 measurement 测试通过。

已生成 780 roots、3,120 题、1,896 张独立图像；双 oracle、训练 holdout 扫描、跨分区隔离通过。与 705 条历史世界记录比较，碰撞为零。在任何新模型输出前冻结 128 道旧题、250 道 ChartQA 和 G2 普查规格。

|作业|名称|卡数|最近状态|
|---|---|---:|---|
|226033|SRF5-G2-W0-LH|1|PENDING / Priority|
|226034|SRF5-G2-W1-LH|1|PENDING / Priority|
|226035|SRF5-G2-W2-LH|1|PENDING / Priority|
|226036|SRF5-G2-W3-LH|1|PENDING / Priority|

四份等量分片各依次执行 b0/b1/b2，总计 4,608 条登记回答。均使用教师 QoS `soujanya-poria-startfund-2026-03`。计入其他作业 `225839` 后，共五个作业、五张运行或排队预约 GPU；其他作业未修改。

G0/G1 的真实 GPU 身份、processor、token 与恢复证据尚未齐全；G2 尚未产生回答。G3 先导 runner 与恢复验证、G4 冻结、45 条正式训练路径、正式评测和统计报告尚未运行。CPU PASS 不是实验结果。

当前作业只运行 G2，不自动跳到 G3 或正式训练。全部分片完成后在部署目录运行 `python -m sr_f5.calibration --root /projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025 --aggregate` 汇总，再按真实门结果继续。没有创建周期监控；排队预约不代表实际 GPU 利用率。

运行根：`/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025`。G2 部署代码：`/projects/_ssd/varunssd/louis-ssvc/sr_f5_code_26e560b`，commit `26e560bac1a7c5b5563f52112bc1374685084b8d`。后续自然接口修复和调度保护位于 `78acca3`，未替换 G2 运行代码。

小型服务器回执已取回 `artifacts/sr_f5_initial_receipts/`，关键回执另存本目录 `validation/`。
