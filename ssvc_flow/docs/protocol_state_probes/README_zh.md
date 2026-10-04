# 真实 Qwen3.5-9B 冻结协议与状态探针

入口为 [design/START_HERE_FOR_CODEX.md](design/START_HERE_FOR_CODEX.md)，唯一主规范为 [design/CODEX_EXECUTION_PLAN_zh.md](design/CODEX_EXECUTION_PLAN_zh.md)。本轮使用既存的 S32、S96、R4_128、DIRECT_128、REP32、REP96，不新增训练、似然重评分或选择器拟合。

新实现位于 `ssvc_flow/src/protocol_state_probes/`；旧训练、旧评价器和历史结果未修改。先以 S96 的完整 D48 核心及三个执行对照交付，再按冻结清单完成全部 70,656 条回答。五条固定单卡 PRO6000 队列分别处理 S96、S32、R4、DIRECT、REP96→REP32；共不超过五张卡。

## 已确认的证据边界

- 原包 ZIP SHA-256：`6b3a4378068e2c03f95e09490c584ef79d2043c7b1653d51e40ee05f04fff2d0`。
- 从 `eb58174c6992e8a82e04ca75cdf06a682f5e9a46` 建立独立工作树；相对规范运行基线，继承的生成、加载和旧评价源代码无改动。
- 服务器 CPU 核验六个登记 COMMIT/MANIFEST 与可读状态文件全部存在。权重内容及真实 forward 身份在每个检查点加载时单独验证，不能由此声称 GPU 已运行。
- 用服务器原件 `prepared_development.json` 逐字段重编译：509 cases、461 PTLC 预测、35 别名、1,944 单元、70,656 回答，与包中清单一致。
- U22 当前 prospective campaign 未命中，但旧 R3/R4/decision-modeling 真实执行记录包含其中 11 个场景。保留22题，整组降级为开发诊断复核；`original_split=train` 不变。详见 [U22_EXPOSURE_SUMMARY.json](U22_EXPOSURE_SUMMARY.json)。该面板不得称为独立未使用结果复核。

## 运行与结果

服务器独立目录为 `/projects/varunssd/louis-ssvc/protocol_state_probes_v1_20261005/`。运行和原始回答留在服务器与本地忽略目录，代码推送不自动发布新模型输出。大权重保持原路径。

CLI：`prepare`、`check-checkpoints`、`smoke`、`worker`、`summarize`。每8条原子提交，原始输出在评分前持久化，错误或非法答案不重试；技术错误最多重试两次。恢复绑定协议、代码、检查点、公开提示、角色、draw和seed，冲突留存并停止。

`worker --with-smoke --lane-id lane-N` 将16条技术对照与正式数据分开，复用一次模型加载。提交前先 dry-run `scripts/protocol_state_probes/submit_lanes.py`；它保留 sbatch 意图和回执，支持先提交 lane1 核验技术接入，再提交其余四条，禁止含糊提交后盲目重提。

CPU 测试和自动报表只证明各自的工程/统计口径。最终科学结论必须阅读完整效应、置信/可信区间、逐题反例、U22污染和REP结果后写出；不能以生成报告文件或程序 PASS 代替结论。
