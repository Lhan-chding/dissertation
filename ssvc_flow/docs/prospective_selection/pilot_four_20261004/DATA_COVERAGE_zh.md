# 首四条小实验数据覆盖

原始归档FIRST_FOUR_RAW_ANALYSIS.tar.gz：258,690,386字节（约258.69MB），内部原件42,605个，解压文件总计3,229,971,410字节（约3.23GB）。SHA256：`fc2e2f70310c7fabfb4fd1950dcf9351f125243fbdb8ec19a893f8bbdcf99e44`。

## 已核验的权威成功数据

|数据|实例数|原始生成记录|
|---|---:|---:|
|source训练|4|12,288|
|分支训练|88|90,112|
|前置P（8原点×当前/历史）|16|36,864|
|分支H8诊断|88|16,896|
|分支H32开发评估|88|202,752|
|历史E复核|16|36,864|
|合计||395,776|

另有3,200条有限训练更新（source384＋branch2816），已提取training_updates_3200.csv。原始记录包含生成文本、token ID/logprob、seed、prompt身份、reward/semantic等字段；未省略这些已存在字段。

原件还包含成功/失败/恢复痕迹、manifest/COMMIT/COMPLETE/任务与提交回执、32前置packet、训练/探针/开发/历史E题面及场景真值、720份图像（缺失0）、源/分支schedule、协议和原训练代码快照。非权威痕迹保留，不能将全部JSONL直接相加作为样本总数。每次样本读取以成功COMMIT的binding为准。

## 完整性证据

RAW_CONTENT_VERIFICATION.json确认：42,605文件SHA256全部一致；gzip读至EOF并校验CRC；路径安全唯一且均为常规文件；完整权威样本绑定与行数匹配；88分支H8/H32汇总和32packet与小实验输入逐项相同。下载归档SHA256与服务器回执一致。

REPLAY_VERIFICATION.json确认：在另一个新目录复算，summary、各CSV、16模型及16折诊断逐字节一致。27项针对性测试通过；包含外层标签拒绝、同源anchor整组留出、内层lineage重叠拒绝、partial结果防覆盖与归档路径安全。图已目视检查。

## 明确未包含

- 新训练树中590个张量/二进制/缓存文件，合计约74.07GB；逐项路径、大小和遗漏原因在raw/RAW_MANIFEST.json的omitted_files。检查点状态hash及绑定保留在小回执中，没有重复哈希这些大型张量。
- Qwen3.5-9B基座和历史D2完整Adam/adapter张量。历史16个端点的路径和state_hash在historical_e任务payload中；历史D2原件保持只读。
- 最终T/T_H8面板与测试结果、剩余开发/调参的新结果。本轮根本没有使用这些结果。
- 操作系统完整镜像及原GPU环境的所有wheel。原代码、运行参数与现有版本回执已保留。

因此“完整数据”指这个小实验的已记录输出、输入、训练过程元数据、模型决策及分析材料完整交付，**不是包含所有模型权重的独立训练重跑包**。轻量核心ZIP可复算选择器；完整ZIP增加raw/FIRST_FOUR_RAW_ANALYSIS.tar.gz供逐样本分析。
