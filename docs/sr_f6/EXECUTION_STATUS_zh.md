# SR-F6 执行状态与续接入口

2026-10-27（Asia/Singapore）。本页是运行事实入口，不替代原合同或阶段证据。

## 用户授权与当前重点

用户要求执行 `SR_F6_CODEX_实验实施包_20261027.zip`，并认真审查设计可能存在的错误。后续明确：题组是否真正有用是基础；只清理我们项目之前无用的缓存和结果，不碰其它项目；本项目 GPU 作业统一 `LH-` 前缀；避免冗余哈希；已授权自动跟进继续执行，仅关键变化通知。

原 ZIP 在 `/Users/louis/Downloads/SR_F6_CODEX_实验实施包_20261027.zip`。保留原包在本工作树 `docs/sr_f6/package`，独立分支 `sr-f6-semantic-factorial-v1`，父提交 `161e908`。工作树绝对路径 `/Users/louis/Documents/ChatGPT/dissertation for NTU/sr-f6`。旧 `sr-f5` 工作树及其未跟踪文件未改。

## 已完成的本地工作

- 原包完整性校验、67 项参考测试、数学检查、预算、全参考数据审核均通过。原包 NOT_RUN 启动门按预期 exit 2。
- 实施前设计复核和纠错见 `IMPLEMENTATION_DECISIONS_zh.md`；评分、信用、统计、数据、题组深诊断各有独立说明。所有更改均在首次 SR-F6 模型输出前。
- 独立 `sr_f6` 实现包括数据/渲染/调度、无损原生评分投影、完整 G8 信用、整序列 FP32 loss、单次 /128 更新、真实1536-token运行适配、来源绑定、processor预检、普查与严格门。旧 F5 源码保持不变。
- 本地数据 `.local/sr_f6_cpu_preparation/DATA`：4,704 主任务加128配对自然接口，3,887 张原图；状态 `CPU_PREPARED_NOT_RELEASED`。真实处理器图像不能用这些原图替代。
- 题组深诊断见 `GROUP_DESIGN_AUDIT_zh.md` 与 `validation/CPU_PRODUCTION/GROUP_DESIGN_MATH.json`。P1 active 覆盖平均增量 .131458，57.29% 根严格增加；覆盖净增280里，anchor同答案候选揭露净增43。保留3个未揭露强造碰撞、2个局部退步根、20个P0/P1相同根。尚无真实模型学习价值证据。
- 最近完整本地集成回归：278 passed、12 skipped、9 subtests passed，见 `validation/INTEGRATED_CPU`；12项skip是旧F3私有原包未安装。后续修正另有针对性回归，最终计数以实际日志为准。

## 服务器与调度

SSH `eee-cluster`。唯一写入/清理边界为 canonical `/projects/_ssd/varunssd/louis-ssvc`。正式run根为 `/projects/_ssd/varunssd/louis-ssvc/sr_f6_semantic_factorial_20261027`。环境 `/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12`，不升级依赖。

只读来源 `sr_f5_recipe_v2_47d5e85`、`sr_f5_semantic_j_recipe_v2_20261026`；模型snapshot和三个SRF2_J6父均按原配置，保持旧F5 FAIL。路径存在性已真实核实；完整服务器内容绑定及实际加载仍需各自运行回执。

实时权限查证：teacher QoS `soujanya-poria-startfund-2026-03` 的 GPU group cap 是8张pro6000，MaxSubmitPU=5；`rose` QoS另有MaxJobsPU=2、MaxSubmitPU=5，用户association未列全局提交上限。必须按当前QoS分别计数，不把rose作业计入teacher的5槽。当前其它项目在不断提交/结束，提交前重查即可，禁止取消或改名其它项目。此前共享账号一度teacher五槽全占；后来出现另QoS作业，不能沿用过时快照宣称永远无槽。

所有新作业名 `LH-SRF6-…`。优先两个独立单GPU worker/作业；CPU准备任务也使用LH前缀。不能为绕过上限把GPU工作放登录节点或换未授权QoS。

## 下一步（不能跳过）

1. 完成本地最终修正检查，commit/push，并用固定revision归档部署到本项目新代码目录；记录实际路径、revision、job ID。现有源码全部进入来源清单，回归测试代码也绑定。
2. `scripts/sr_f6/prepare.sbatch` 执行实际来源绑定、服务器数据构造、历史世界排除、真实processor 48图与tokenizer回归、native regression。原model内容只绑定一次；后续复用验证缓存，只核变动文件。
3. 把实际 `PREFLIGHT/processor` 图与原图交独立视觉审核，记录 MODEL_VISUAL/HUMAN，不能伪造人工审核。所有48图合格后，按 `sr_f6.release seal-census` 的schema写真实逐图 `PREFLIGHT/VISUAL_REVIEW.json` 并封存普查前置。
4. `scripts/sr_f6/census.sbatch` 每父两shard，3父最大9,216条DEV_CAL回答，0更新。首个登记G8必须用原tokens真实teacher-force概率回放，失败保留并停，不补采。完整普查及360条盲审核后才判 SOURCE_BINDING/DATA_AND_MEASUREMENT/PARENT_CENSUS 最终门。
5. 普查后专门分析题组有限库与真实错误、脆弱响应、可用信用的关系。只用登记数据，不依结果调题、seed、lambda或门。
6. 生产先导/事务恢复/正式训练/评测编排尚未完成；需在各阶段进入前实现并验证，禁止把单步update函数当完整trainer。新增编排若改变来源清单，保留旧证据并明确登记前后源码等价边界，不能重写普查原source身份或无因重采。原6臂/3父/3重复/72步仍不变。

## 自动跟进与通知

本聊天 heartbeat `sr-f6` 已创建为ACTIVE，每15分钟检查/推进。状态无变化时安静，只在关键进展、失败、完工或需要用户处理时通知。科学门失败按原合同停止扩展，不自动换配方。完整交付后关闭自动跟进。

## 清理

已授权唯一大体量归档清理目标：`sr_f3_ce_20261016/prior/raw`，其STOP为 `STOPPED_PROTOCOL_FAILURE`；当前F5/F6无依赖。先对45,454个普通文件（5,154,950,860B）做完整无损归档+成员校验，再按精确allowlist删原件。归档549,096,352B。实际删除是否完成以及净释放量必须以 `validation/STORAGE_20261027` 最终回执为准，不能仅凭归档完成断言释放空间。

其它项目 `/projects/_hdd/varunhdd/hgrel_hdd` 及其cafe作业均未修改。本项目模型、父adapter、环境、F4保持面板、F5 step004/step012、旧失败复现依赖保留。登录节点cache不等于无用实验文件，不清系统page cache。

## 状态边界

**当前没有 SR-F6 科学结果。** CPU通过、路径可见、归档成功、代码可运行分别不是实际模型加载、普查通过、先导通过、54路完成或泛化收益。后续实际job/来源绑定/processor/普查状态将追加在本页和带时间的验证回执中。
