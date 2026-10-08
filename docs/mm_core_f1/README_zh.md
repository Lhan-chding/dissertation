# MM-CORE F1 多模态审计

执行依据是 [用户提供的 F1 契约](design/CODEX_FINAL_PLAN_MM_AUDIT_ENGINE_zh.md)。
原包逐文件保留，其 `MANIFEST.json` 可复核输入。本轮用户覆盖条件见
[执行修订](USER_AMENDMENT_20261008_zh.md)：使用 Qwen/Qwen3.5-9B，最多五张 Pro 6000，
GPU 小时只记账，没有累计上限，也不作为执行或通过门槛；其余生成、更新、前向及阶段上限不变。
五卡是并发上限，不代表现场一定可同时分配五卡。所有账户和调度限制照常生效。
当前执行结果、验证与限制见[执行状态](EXECUTION_STATUS_zh.md)。

本轮只允许资产核验、CPU 契约/生成器、MM-AUDIT，以及登记触发的公共桥接与
真实多模态 ENGINE。完成报告后停止；`MM-DEV/MM-LOCK/MM-CAL/MM-ONLINE/SER-J23`
均不在本轮执行权限内。未来 1088 更新、五起点、30 响应只是未授权骨架。

## 代码和私有证据

- `src/mm_core/`：图表生成、精确 scorer、真实视觉输入、有限训练接口和阶段/预算门控。
- `scripts/mm_core/`：契约所要求的独立命令入口。
- `tests/mm_core/`：参考代数、生产解析/统计、根切分、预算与阶段回归测试。
- `artifacts/mm_core_f1_<run_id>/`：被 Git 忽略的原始图像、请求、token、模型身份、
  服务器路径、计费、失败和最终中文报告。模型权重和私有运行记录不提交到仓库。

使用已有且已核验的运行环境，设置 `PYTHONPATH=src`。不要为追求最新版本升级环境。
CPU 检查可运行 `python -m pytest tests/mm_core`；测试成功不能代替真实 GPU 回执。

## 执行顺序

1. `inventory_assets.py` 核验历史分支和真实模型。停止记录只匹配已登记旧作业，
   `UNKNOWN` 不能解释为已停止。
2. `validate_generator.py` 生成五个授权面板并验证每一题、原始图像和 split。
3. `processor_preflight.py` 在真实 processor 上执行 CPU 图像处理，保存实际 tensor、
   grid/token 数、重建图像与处理后去重证据；再完成处理后图像的视觉审查。
4. 环境、模型文件、数据和资源清单全部真实填妥后，`freeze_audit.py` 产生唯一冻结。
5. 先登记不可变阶段分片计划，再为精确 Slurm allocation 记录设备数×请求时限估计。
   加载、空闲和失败均计入实际分配时间；这些小时数不阻止执行。只有终态调度证据可以释放并发名额。
6. `run_format_check.py --stage FORMAT_BASE_TEST` 在 FORMAT_TUNE 执行 K=4。
   汇总器核验全部样本槽位、原始记录哈希和同一模型身份，不能只信报告中的 PASS。
7. 只有真实格式遵从不足才登记一次 `BRIDGE_TRIGGER` 并运行 16 步短 SFT。
   图像路由、模板或解码 bug 不能用 SFT 掩盖。桥接终点是新公共起点。
8. 同一公共起点只做一次独立 FORMAT_CHECK。失败时本轮停止；通过才可运行测量。
9. `run_measurement_audit.py` 执行 AUDIT_MEASURE，原始回答先落盘，附加字段评分随后
   独立保存。附加评分失败不能抹掉已经生成的回答。
10. 测量完成后以 `score_measurement.py --stage MEASUREMENT_AUDIT` 评分并核对全部1536个槽位。
    没有身份完全匹配的恢复回执时，独立登记 ENGINE，运行连续四步和重启恢复四步。
    精确比较参数、优化器、RNG、槽位和原始 token，不把“可加载”当作恢复一致。
    ENGINE配方和八题顺序在执行前冻结；提交前另绑定测量审阅回执与原始成本/请求账本前缀。
11. `release_audit_report.py` 输出原始状态、成本、未执行项和
    `authorized=false` 的 DEV 冻结补充表。
12. `build_scientific_report.py --run-root RUN` 将已有机器表展开成中文科学报告及六份附录，
    只读原始输入，不重新评分或调用模型。原始释放状态保留，DEV参数仍开放时明确解释为
    `READY_WITH_OPEN_DEV_FIELDS`，不增加执行权限。
13. `verify_final_evidence.py --run-root RUN --source-root FROZEN_SRC` 在服务器核验当前冻结、
    测量身份、三段ENGINE记录、真实CPU加载的自身检查点与所有物理成本，再生成最终文件哈希清单。
    只有先完成所有表、报告和调度回执，才能形成稳定快照；核验后不再修改已纳入清单的文件。
    核验工具的CPU测试不是实际ENGINE通过回执。交付后停止，不自动进入MM-DEV。

## 实现边界

路径 B 的两个字段均来自同一次真实图像自由生成。历史四整数/执行器答案以及分步
多次询问不合并成新协议的联合事件。模型输入和私有真值侧车分别保存，gold 字段 NLL
明确属于 teacher forcing，不能代替自由生成能力。

精确主表采用有理数；容差八格独立计数。缺失、技术失败、截断和超轴数值保留。
S 在正确读数上记 missing；跨提示量包含同提示噪声对照，允许负差异估计。
CPU 合成验证、实际模型测量、训练接口恢复与科学结论分别报告。

接口核对来源为 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、
[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
和 [TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)。这些资料只用于接口核对；
本次 native GRPO 的 reward、归一化、clipping、KL、优化器等由 ENGINE 冻结完整展开。
