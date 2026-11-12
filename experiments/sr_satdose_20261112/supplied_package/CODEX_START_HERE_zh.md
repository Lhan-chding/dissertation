# Codex 执行入口：SR-SATDOSE-1

**任务：实现并执行一次经过分级验收的高冲突曝光实验。不是继续讨论，也不是直接提交全部长作业。**

先完整阅读 `CODEX_EXPERIMENT_PLAN_zh.md`、`docs/01_IMPLEMENTATION_AND_TESTS_zh.md`、`docs/02_GATES_AND_STATISTICS_zh.md`。数值配置以 `config/experiment.json` 为唯一机器来源；发现文档/配置/原生代码不一致时 HOLD，不自行猜测。

## 你被授权做什么

在**新的实验目录/分支**实现数据调度、R24 分叉加载、只读梯度探针、验收与执行 wrapper。旧项目和旧结果只读。复用经过 SHA 校验的原生评分器、损失、生成、优化器、图表生成与渲染；不得把当前项目改成另一套 RL 库。

按以下顺序推进：

1. 运行本包 CPU 测试和历史审计，检查已保存的本地测试报告。
2. 核验真实服务器代码、两个 R24 检查点、原 b0 reference，建立新科学身份。
3. 构造并冻结数据、图片、请求清单、DEV 与 MAIN 分离、统计规则；先做无需模型的结构与评分夹具。
4. 两起点各 64 个冲突 G8 + 16 个一致 G8，不更新参数；完成前缀小面板和 GPU 数值/方向/影子更新验收。
5. 调用 `tools/check_admission.py`。只有真实证据得到 `ADMITTED_FIRST_TWO_STEPS_ONLY` 才允许四路径各运行前两步。
6. 完成生产前两步的策略新鲜度、精确一次提交、恢复与端点加载验收，才放行到 8、16、24 步。**这两步计入24步，不另训、不回滚成更好的起点。**
7. 四路径都冻结/终止后，才解封 MAIN。执行评分、配对分析、机制报告和完整交付。

通过全部固定门槛即可继续，不需要逐项等待用户确认。出现 HOLD 必须停止相关后续模型调用，给出证据和失败原因；**不得降低门槛、挑题重抽、换起点、追加 seed/臂/步数或自动开始 Stage 2。**

## 第一批可直接运行的 CPU 命令

```bash
# SR_BUNDLE 是 SR_ORD_TRAIN_RESULTS_20261111(1).zip 的真实解压目录，不能猜路径。
export SR_BUNDLE=/absolute/path/to/verified/results_bundle
export SR_HISTORICAL_BUNDLE="$SR_BUNDLE"
python -m unittest discover -s tests -v
python tools/audit_historical.py --bundle "$SR_BUNDLE" --out /new/owned/path/HISTORICAL_AUDIT.json
python tools/build_schedule.py --check config/training_slots.jsonl
```

本包**没有服务器训练适配器、模型权重，也没有 GPU PASS 回执**。`templates/GPU_VERIFICATION.template.json` 全部为未通过，必须由真实执行生成证据，不能填字造 PASS。本包 CPU 工具不会提交训练任务。

## 最重要的禁止事项

- 正式训练只用原生 J；S、B、来源状态、证书分解只作观测，不能暗中乘入 J。
- `J=1` 并不保证 `selected` 正确。主行为指标是精确授权集合；答案 A、J、证书另报。
- policy 起点是 R24；reference 是**原 b0 parent**；新分叉 fresh AdamW，断点恢复则必须恢复已有 Adam 状态。
- 相同 seed/题目不等于共享 rollout。每臂每一步必须从自己的当前策略重新采样。
- 不得使用 MAIN 选配置、选检查点或触发提前停止。未知回答保留分母，技术缺失不得冒充错误回答。
- 不以几小时/几天作为科学停止条件；不因前2/8步没涨分就断言无效。

## 最终交付

`ADMISSION_REPORT_zh.md`、可复跑代码/命令、所有冻结清单与哈希、逐条原文/原 token/评分、更新与恢复回执、逐起点统计、最终 `FINAL_REPORT_zh.md`、可独立核验结果包。无论 PASS、HOLD 或阴性结果，都交付“确实做了什么、未做什么、下一步为何被允许/阻止”。
