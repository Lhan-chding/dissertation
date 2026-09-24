# 4–5张PRO6000执行手册

## 1. 资源策略

- 默认最多5个项目GPU作业，每作业1张卡；资源只到4张时自动降为4个worker。
- 开发/测试分支互不依赖，可并行；source t32完成后，其分支可与同source继续训练到t96并行。
- 不默认DDP：它会改变当前小B/K训练的同步和数值条件，也不能替代独立种子。
- 不承诺未测量的小时数。先在首个source和2个branch上记录加载、32步训练、144prompt端点评估的实际时间，再更新全量预计工期。
- 充分使用资源不等于全部候选/全部面板/全部样本量组合穷举；最终只运行冻结选择和基线并集。

## 2. 一次性准备

1. 检查仓库分支和当前代码修复，不覆盖其它worktree。
2. 从旧D2运行配置继承模型缓存、LoRA、generation、optimizer、Python环境。
3. `nvidia-smi`记录GPU型号/显存/驱动；确认实际分配的设备，不凭PRO6000名称假设显存容量。
4. 读取train/control/dev/confirm元数据一次，生成面板和source/continuation池。
5. 第一次导入源checkpoint时检查可加载、必要状态字段和模型身份。之后用文件路径/尺寸/版本及完成标记，不每个step重新全量哈希。
6. 运行针对性CPU测试和一个真实2-step smoke；同环境与相同训练代码无需每个recipe重复smoke。

权限、网络或原件缺失时，继续不依赖它们的本地实现；不要手写PASS替代真实验证。

## 3. 调度DAG

### 开发

`prepare_data → source(seed) → checkpoint(t−8,t) → prestate(P) → branch(recipe) → evaluate(D) → development_table`。

source输入为公共基座，输出不可变LoRA+Adam checkpoint；不同worker读取时不共享可写对象。

每个source原点的11个recipe都进入任务表，采用round-robin完成小批完整原点，避免只跑R0/R4或只积累source checkpoint。

### 冻结及测试

`development+tuning → freeze → source(test_seed) → prestate → decision.json → unique_recipe_union → branch(repeat1/2) → evaluate(T)`。

测试source可在冻结后的空卡准备；任何测试候选训练必须有已存在且不可改的decision文件。所有测试选择器的参数在整个测试阶段固定。

### 每个作业的唯一键

- source：`protocol/source/lineage/source_recipe`；
- prestate：`lineage_id/origin_id/policy_id/P/snapshot_t/draw_role`；
- branch：`origin_id/recipe_id/repeat/schedule_id`；
- evaluation：`lineage_id/origin_id/repeat/policy_id/panel_id/horizon/draw_role`。

selector_id不进入branch键。同一recipe被三个selector选中，只运行一次。跨lineage或两个随机重复之间，不复用同一份生成计数来冒充独立观测；相同输出的确定性评分可缓存，但随机抽样流和统计权重仍分别记录。

## 4. 建议的worker分工

| 当前工作 | 5卡时 | 4卡时 |
|---|---|---|
| source/branch积压 | 4训练＋1评估 | 3训练＋1评估 |
| source尚未准备充分 | 5个独立source | 4个独立source |
| 端点积压 | 2训练＋3评估 | 2训练＋2评估 |
| 最终测试 | 根据去重task队列自动分配 | 同左 |

这是动态队列，不需要把某张物理卡永久绑到一个角色。每个作业只用CUDA_VISIBLE_DEVICES内设备；不自行占用未分配GPU。

## 5. Slurm与启动命令交付

优先继承已成功的NTU EEE wrapper和老师QOS；account/partition/内存/CPU请求以实际集群配置为准，不在本包猜测。

Codex最终生成：

- `scripts/prospective_selection/launch_worker.sbatch`；
- `scripts/prospective_selection/submit_ready.py`；
- `runtime_paths.example.yaml`（不含密钥）；
- `SERVER_HANDOFF_zh.md`（填好真实CLI参数的可执行命令）。

模板最小结构：

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${SSVC_ROOT:?set project root}"
: "${SSVC_PYTHON:?set validated Python executable}"
: "${SSVC_TASK_FILE:?set one registered task JSON}"
cd "$SSVC_ROOT/ssvc_flow"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$SSVC_PYTHON" -m src.prospective_selection.cli worker \
  --task "$SSVC_TASK_FILE" --runtime "$SSVC_ROOT/runtime_paths.yaml"
```

GPU申请参数由继承的站点wrapper添加。`submit_ready`必须查询本项目pending/running作业并限制在5以内；一次性写submission intent后单次提交，避免重启重复sbatch。此记录不需要全模型哈希。

## 6. 断点和存储

- source保存0、24、32、88、96中经过的节点；
- branch保存0、8、32；恢复用临时完整状态每8步保存一次；最终保留选定端点，其余中间状态可在结果验收后清理；
- 中断最多回放一个未提交8步块；旧attempt日志保留，只有从最近完整checkpoint延续的attempt成为权威数据；
- `.partial`写完后atomic rename；不能把半写的checkpoint当完成；
- 更新日志、原始文本、token IDs、标签共享存储，压缩为jsonl.zst/parquet可选；不能每个selector复制一份；
- 不把整个9B基座保存到每个训练分支；磁盘主要为LoRA/Adam和原始输出；
- 每阶段开始检查容量；低空间或写失败时再检查，不在每条样本前递归扫描目录。

改变resume微批只允许保持逻辑B4/K8和损失分母。改变模型、奖励、题序或随机重复不是resume，而是新run。

## 7. 性能优化的优先顺序

1. 同配方/原点/重复去重；
2. 相同不可变prepared prompt复用；
3. 生成分片按prompt批次，不反复加载权重；
4. 每样本不再额外运行全套似然差比较；
5. 只对涉及的新模块做测试，不重复整个历史资格矩阵；
6. 最后才优化teacher/cache路径。路径没有通过既有parity测试时，主实验继续用已验证prefix路径。

不要为吞吐静默改变EOS、sampling过滤、thinking或序列概率定义；不能为了batch size重用不同策略生成的旧rollout。

## 8. 必须记录的实际成本

分别记录source训练、未来分支训练、前置状态采样、最终评估、selector拟合和可选诊断。

- GPU小时是各GPU占用时间之和，不是并行墙钟；
- 新模型加载次数、真实更新数、生成输出/tokens、实际评分数、峰值显存；
- 研究性开发动作矩阵成本与部署时selector成本分开；
- 跨selector引用同一branch不增加物理工作量，但不能把开发校准全部当免费；
- 同时给出每个独立测试原点额外前置观测的开销。

## 9. 工作量不是一次性死队列

`protocol.json`给出保守上界。默认12个测试lineage时：

- source更新2,304；
- 开发/调参352个分支、11,264次更新；
- 测试最多216个实际分支、6,912次更新；
- 合计最多20,480次更新、655,360条训练输出。

测试上界按四selector＋BestStatic＋R0＋三外部基线全部选不同recipe计算，实际常常更低。最终N若经冻结前功效规划增加，脚本重新计算，不修改论文中的统计单位。

开发H32约811,008条输出；默认测试H32上界995,328条。它们是完整研究矩阵，不是未来每次控制都必须花费的测量量。先交付4个开发lineage结果和实测工期，不等待全部矩阵才能讨论研究方向。

如果原点间可利用的动作差异很小，先报告headroom和不确定性；不为了填满5卡而制造更多无信息的重复实验。
