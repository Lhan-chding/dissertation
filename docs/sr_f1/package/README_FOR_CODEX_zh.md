# 从这里开始：SR-F1

这是用户要求的下一步**真实科学实验执行包**。先读 `CODEX_NEXT_PLAN_SR_F1_zh.md`，再读 `config/SR_F1.json`。

核心：在新的可执行语言条件/视觉题组上，5种固定反馈、同数据同起点、3种子，每条96步。构建任务与技术核验通过后，应继续完整训练和评价；不能只交一份审计。

## 最重要的边界

- 模型仍为Qwen3.5-9B固定快照；不设GPU小时研究截止。
- 不是旧数据配方预测路线；F2不作为前置、不追溯改动、不自动取消。
- 不恢复SER-J23；不在本轮加动态选题、奖励路由或计算支持SFT。
- 96步、数据根、奖励和seed是科学配方，不能为了显著结果延长。
- 本包包含**CPU参考实现与清单**，不包含声称已运行的Qwen训练器。

## 内容

| 文件 | 用途 |
|---|---|
| `CODEX_NEXT_PLAN_SR_F1_zh.md` | 完整执行合同 |
| `config/SR_F1.json` | 机器参数 |
| `manifests/RUN_MATRIX.json` | 15条固定训练路径 |
| `manifests/schedules/*.jsonl` | 3个输入流，每个1536题一次遍历 |
| `manifests/*AUDIT_ONLY*` | 世界、真值和诊断权限；不能串入模型输入 |
| `manifests/MODEL_INPUTS.jsonl` | 根据qid取图和可见文本；qid/路径不进入prompt |
| `reference/semantic_contract.py` | AST、精确评分、五种最终优势接口 |
| `reference/build_manifest.py` | 确定性数据构造 |
| `reference/render_charts.py` | Pillow合成图参考实现 |
| `reference/build_diagnostic_inputs.py` | 五种只测不训的诊断视图 |
| `reference/recompute_scores.py` | 原文独立重评分 |
| `tests/` | 数学/解析/奖励测试 |
| `validation/` | 本地真实运行的测试与清单核查 |
| `examples/` | 36张不同例图、联系图、查询示例；无模型输出 |
| `DECISIONS_FROM_DISCUSSION_zh.md` | 本次补齐与继承的边界 |
| `SOURCES_AND_SCOPE_zh.md` | 文献和已完成范围 |
| `templates/` | 运行冻结与交付模板 |

## 可以立即运行的CPU命令

```bash
python -m pytest -q tests
python reference/validate_manifest.py
python reference/check_budget.py
```

重新生成清单应写入新目录后比对，不覆盖本包登记文件：

```bash
python reference/build_manifest.py --out /tmp/sr_f1_rebuild
```

全部正式图像应渲染到运行目录，而不是将示例目录当全部数据：

```bash
python reference/render_charts.py --tasks manifests/TASKS_GOLD_AUDIT_ONLY.jsonl --out <RUN_ROOT>
```

这一步仍不等于原生processor后的可读性核验。运行端要按主文件继续检查。

## GPU入口由Codex实现

不要把以下概念名称当成已存在的工具：prepare / sample / train / evaluate / release。它们是要求实现的阶段。可复用现有mm_core/mm_dev工程，但应建立独立sr_f1代码与运行根，并提供实际可运行入口、依赖锁和命令回执。

最终交付必须说明：实际训练了哪些路径、每条路径的收益与副作用、普通问答及迁移结果、哪些信用变化真的带来能力变化。没有模型结果时，不用CPU测试数或原型数据数代替科学结论。
