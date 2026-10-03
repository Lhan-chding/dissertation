# 小实验复算与数据口径

Python 3.12，numpy；绘图另需matplotlib。代码不调用GPU。数据来自首四条已完成开发端点，不使用最终测试T。

从数据包根目录进入ssvc_flow：

```bash
cd ssvc_flow
python -m scripts.prospective_selection.pilot_four \
  --input docs/prospective_selection/first_four_20261004/compact_input.json \
  --out ../reproduced_pilot
```

输出必须是新目录；脚本拒绝覆盖已有完整或部分决策。16个折模型和32选择由前置packet与仅训练部分标签生成，再关联留出端点评分。主实验冻结文件未写入。FrozenSelector对象内部FROZEN仅指该折已拟合对象。

原件归档可直接用tar工具展开；成员名均为安全相对路径。检查归档（可选择输出新的核验目录）：

```bash
python scripts/prospective_selection/verify_first_four_archive.py \
  ../raw/FIRST_FOUR_RAW_ANALYSIS.tar.gz ../raw_verification \
  --compact docs/prospective_selection/first_four_20261004/compact_input.json
```

核验器顺序读到gzip EOF，校验全部归档文件SHA256、唯一安全路径、完整成功训练与评估的COMMIT绑定、样本行数、3200条有限更新；将88分支H8/H32汇总和32前置packet与分析输入逐项比对。它不是重新运行模型或重新解析评分全部生成文本，之前的真实原件身份与semantic汇总核验回执另行保留。

完整权重、Adam张量和缓存留在服务器。RAW_MANIFEST.json逐项列出本次树内未下载的大文件；历史D2张量路径/状态身份保存在历史任务payload中。基座Qwen3.5-9B revision及训练配方/优化器参数在runtime和原代码版本中。该包支持完整结果分析、选择器复算和逐样本追查，不宣称包含可从零训练重跑的所有权重与依赖环境。

使用PNG图像时，准备记录中的data_root为原服务器位置；RAW_MANIFEST.json的image_path_map将原绝对路径映射到归档inputs/images/。历史E题面data_root=None沿用其historical_runtime.data_root，导出按这一原运行规则定位，没有改原题面。

旧失败attempt、cancelled/rejected记录与非权威样本可能同时在包内，必须按COMPLETE.authoritative_segments与评估COMPLETE.chunks选成功记录，不能重复统计目录内所有JSONL。

P为前置观测，D为开发H32评估，D_H8为D的既定诊断子集；H8/H32面板不同，不直接把两者分数相减解释训练提升。历史E是旧端点复核，不参与新选择器拟合。T/T_H8未导出。
