# 核验与复算

compact_input.json由export_development_progress.py --metadata-only --compact读取服务器固定流水配置、任务文件、完成回执、verified审计、H8/H32汇总及四层信息包导出。源文件哈希在source_files，输入SHA256在INPUT_SHA256.txt。沿用既有逐项原始输出核验，不重复哈希9B模型。CANONICAL_VALIDATION.json来自原训练快照reporting.development_report(first_four=True)，检查源轨迹/前置观测/分支评估小文件身份、端点sidecar、面板与检查点关联；88个分支J已与快照逐项比对。

不含原始生成文本或权重，不是完整重跑包。H8面板不同于H32，不能直接将两者相减解释为训练收益。baseline_timing.json为旧42项成功分支sacct，后46项时长来自快照审计。

在worktree根运行分析：

```bash
PYTHONPATH=ssvc_flow /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python ssvc_flow/scripts/prospective_selection/analyze_development_progress.py --input ssvc_flow/docs/prospective_selection/first_four_20261004/compact_input.json --baseline-timing ssvc_flow/docs/prospective_selection/first_four_20261004/baseline_timing.json --out /tmp/ssvc-first-four-reproduction
```

README为本次人工整理的正式交付入口，分析脚本生成统计表与图。复算不创建交付/冻结/测试决策文件。
