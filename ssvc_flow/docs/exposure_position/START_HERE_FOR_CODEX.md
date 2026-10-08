# SER-J23 执行入口

本模块实现用户提供的 FINAL v1.0 四格受控 SFT 契约。唯一规范是私有输入包中的 `CODEX_NEXT_EXPERIMENT_PLAN_SER_J23_FINAL_zh.md` 和 `SER_J23_DESIGN.json`；代码校验两者原始 SHA256。输入包、任务、历史原始输出、权重、服务器路径和运行回执只放被 Git 忽略的私有 artifacts 目录。

实现使用独立 `src.exposure_position` 包，继承 `src.exposure_substitution` 的数值训练核心、完整状态格式与公开提示。旧包和旧运行均保持只读。学生序列化格式名称仍为 `ser-j2-student-state-v1`，每个新学生的实际 identity 是 `SER_J23_20261008`；旧学生通过只读映射进入新逻辑臂，不改写权重身份。

## 执行顺序

在 `ssvc_flow` 中使用已核验的原 Python 环境，不升级依赖：

```text
python -m src.exposure_position.cli audit-legacy --design DESIGN --legacy-run LEGACY --run RUN
python -m src.exposure_position.cli build-training --design DESIGN --legacy-run LEGACY --run RUN
python -m src.exposure_position.cli cpu-check --run RUN
python -m src.exposure_position.cli validate --run RUN --source-package PACKAGE
python -m src.exposure_position.cli bridge --run RUN --allow-gpu
python -m src.exposure_position.cli freeze --run RUN --exclusion-audit AUDIT --validation-receipt RUN/VALIDATION_RECEIPT.json
python -m src.exposure_position.cli register --run RUN
python -m src.exposure_position.cli worker --run RUN --worker NAME --allow-gpu
python -m src.exposure_position.cli integrity --run RUN
python -m src.exposure_position.cli release-and-analyze --run RUN
```

`audit-legacy --metadata-only` 永远不证明权重可复用。正常审计读取真实父状态、COMMIT/manifest、基座快照、旧 A2/B2 检查点及其完整 CPU state hash、Adam、RNG、逐题曝光和原身份。`REUSE_CANDIDATE` 仅表示进入桥接的候选；模式只能在数值桥接后登记。

CPU 审计应在分配的计算资源上执行。真实桥接必须使用单张 GPU，保留完整 32 次物理更新预算与失败尝试；不能重新调用来抹掉失败或扩充预算。所有代码完成验证后再桥接：桥接、冻结及注册均绑定源码哈希。

`validate` 实际运行两组 CPU 测试和 Ruff，把退出码、JUnit 计数、日志、源码和测试哈希写入运行目录。任何检查失败都不能进入桥接；旧审计、父权重、来源目录及冻结文件哈希也必须对应本次训练清单。复制另一个运行目录的合格回执不能通过门控。

排除审计通过 `exclusions.audit_exclusions` 接收明确的历史索引、SER-J2、后续本地数据和人工 fixtures。保存完整扫描范围与来源哈希；缺失索引、无法读取的真实来源或哈希变化均阻塞。补齐现场服务器清单后，才可冻结并由 auditor 生成 E_CONFIRM2。

## 权限、身份与恢复

训练 worker 只读取训练文件和控制绑定；确认生成只读取公共题面。各角色验证文件时也遵循白名单，不能因为“只计算哈希”而打开确认真值。当前能力与文件加载隔离共用 OS 账号，不宣称密码学保密。

一个物理 GPU、一个学生只能由一个任务持有。中断和网络不确定状态保留为 UNKNOWN，不能根据超时自动重新提交。重试需要核验原进程已停止及调度终态，沿用原 seed 和完整学生 Adam/RNG/游标。有效回答不可覆盖；正常模型错误、域外与截断保留在统计分母。

GPU 分配成本须从完整调度记录导入 `SCHEDULER_ACCOUNTING.json`，包括技术桥接、正式 worker、失败和取消记录；未核验成本不会写 STAGE_COMPLETE。所有科学任务完整后统一揭晓，报告全部主、次级、异质性、训练内与开发诊断，再结束本阶段。

## 验证

```text
python -m src.exposure_position.cli validate --run RUN --source-package PACKAGE
```

实际私有输入的 CPU 集成测试需要显式设置 `SER_J23_SOURCE_PACKAGE` 和 `SER_J23_LEGACY_RUN`。合成统计测试覆盖完整请求数和固定 10,000 次配对根 bootstrap；这些测试均不构成真实模型结果或数值 GPU 桥接。
