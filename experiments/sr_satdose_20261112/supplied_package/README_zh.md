# SR-SATDOSE-1 / Codex 执行包

**交付日期：2026-11-12。状态：本地CPU验证完成；服务器GPU和真实模型准入尚未执行。**

## 从哪里开始

- **Codex入口：** `CODEX_START_HERE_zh.md`。把整个ZIP及已有原结果包一起交给Codex，从此文件开始。
- **完整实验计划：** `CODEX_EXPERIMENT_PLAN_zh.md`。
- **已经实际测试的项目与限制：** `verification/LOCAL_TEST_REPORT_zh.md`。
- **生产实施细节：** `docs/01_IMPLEMENTATION_AND_TESTS_zh.md`。
- **门禁schema、命令与统计接口：** `docs/02_GATES_AND_STATISTICS_zh.md`。

## 包内内容

`config/`：四条路径的冻结科学配置、checkpoint/source绑定、完整训练槽位。

`tools/`：历史只读审计、槽位生成与核验、真实干跑重评分和准入检查、root级主终点统计、共同底层函数。

`tests/`：70项本地已通过测试，涵盖异常阻断、预算、原生评分集成与统计。

`templates/`：默认所有项目未通过的GPU回执模板。**模板不是PASS，CPU测试也不是GPU准入。**

`evidence/`：Claude Stage0原始报告与定义/代码审计副本，保留来源SHA。

`verification/`：实际CPU日志、历史数据复算、缺失证据拒绝执行演示、未能启动的旧完整测试日志及明确限制。最终整套CPU重跑记录以`end_to_end/`为准，其他日志保留审计沿革。

## 科学和预算边界

正式部分：2个b0 R24起点 × T0/T1 ×24步；12,288条正式训练回答。连同干跑、DEV和MAIN的全部计划生成为28,936条；前向/反向与影子更新成本另记。

先经过静态检查和无更新干跑，再放行四路径各前两步；确认更新及恢复正确后才能跑满。没有新奖励训练、没有自动加步数/seed/臂、没有自动Stage2。真实模型门禁未通过时必须交付HOLD原因。

本包不包含服务器生产GPU训练驱动和模型权重；Codex需要按合同接入旧代码。这一点不能被改写为“所有实验测试已完成”。

## 完整性

`SHA256SUMS.json`列出包内文件的SHA与字节数（不含该清单自身）。解压后核对，任何科学配置/工具被修改均需新版本与重新测试，不能沿用旧PASS回执。
