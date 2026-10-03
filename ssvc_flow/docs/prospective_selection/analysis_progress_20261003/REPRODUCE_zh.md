# 输入与复算

快照为2026-10-03T12:09:55Z的86/88分支、16/16历史E和8/8前置原点，不随服务器继续运行而改变。compact_input.json.gz和compact_input.json为同一输入的压缩与展开形式，INPUT_SHA256.txt绑定展开JSON和baseline_timing.json。

服务器完整元数据快照保存在fixed_pipelines_20260930/ANALYSIS_METADATA_20261003.json，其SHA256为681e2b4c6210d82ca4a9889aa5707cda08dd663fd75d4a7547a96c4cfde53a18；已下载压缩副本并解压校验该哈希。compact_input.json的parent_snapshot_sha256绑定它。精简只删除branch的training、manifest及评估的chunks列表，保留任务/完成审计、全部评估汇总、四层信息包及source_files小文件哈希索引。原始生成文本、权重及训练逐步元数据未包含。本目录不能称为完整重跑包。

export_metadata.py保留本轮服务器导出源码。后续快照可使用scripts/prospective_selection/export_development_progress.py --metadata-only --compact --out 新文件路径；不得覆盖本快照。baseline_timing.json为另外只读查询42个旧单分支作业的sacct记录，均COMPLETED/0:0；另外44分支时长来自当前快照审计。

从worktree根运行：

```bash
PYTHONPATH=ssvc_flow /Users/louis/Documents/ChatGPT/dissertation-ntu/.venv/bin/python \
  ssvc_flow/scripts/prospective_selection/analyze_development_progress.py \
  --input ssvc_flow/docs/prospective_selection/analysis_progress_20261003/compact_input.json \
  --baseline-timing ssvc_flow/docs/prospective_selection/analysis_progress_20261003/baseline_timing.json \
  --out ssvc_flow/docs/prospective_selection/analysis_progress_20261003
```

输出7份CSV、summary.json、4份分析/入口Markdown、历史E说明、响应矩阵PNG/PDF。候选选择空间只有R0–R7；均值表保留GDPO/SAW/DIRECT_REPAIR三个外部固定对照。只在配方齐全的7原点上比较，不填补两项运行中结果，不拟合选择器、不冻结、不创建首批交付回执或最终测试。

针对性测试69项通过，包含同奖励条件分桶、信息包投影破坏拒绝、外部对照不混入候选空间、重复提交竞态和原流水/任务门禁。响应图已目视检查。
