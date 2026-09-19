# decision-modeling-v1 服务器交接

本地已实现 D0 及 D1–D4 所需代码。2026-09-19 用户明确同意开始服务器实验；按 D1、D2、开发冻结、D4 的顺序推进，每一后继阶段先检查前序实测结果。实际作业状态以运行目录的提交收据、Slurm 状态和原始结果为准。CPU 测试不是真实模型的概率一致性或研究成功证据。

分支：`codex/decision-modeling-v1-20260919`。从实际 V4 提交 `9f5d19c` 建立独立 worktree，历史模块和原始结果保持不变。原方案见 [CODEX_EXPERIMENT_PLAN_zh.md](design/CODEX_EXPERIMENT_PLAN_zh.md)；入口是本文件。附件引用的同包 `RESEARCH_REVIEW_zh.md`、`SOURCES.md` 及新的配套 protocol 原件未提供；本实现明确根据收到的方案冻结 `configs/decision_modeling/protocol.json`，基线另查原论文/官方实现，映射见 [BASELINE_MAPPING_zh.md](BASELINE_MAPPING_zh.md)。

实际本地结果见 [D0_REANALYSIS_zh.md](D0_REANALYSIS_zh.md)、[PANEL_AUDIT_zh.md](PANEL_AUDIT_zh.md)、[IMPLEMENTATION_REPORT_zh.md](IMPLEMENTATION_REPORT_zh.md)。D0 保留 2,304 个抽样观察，复核 3,072 条评分、446 个唯一评分键。P/E 各 72 prompts，互不重叠，旧 R3/debug 排除，无需生成补充场景。

## 服务器第一步：绑定现有原件

在服务器新 checkout 的 `ssvc_flow` 目录操作，使用原已验证环境，不更新 torch/transformers 等依赖。保留原 Qwen revision、BF16、LoRA、Adam、generation 和数据。`runtime.json` 含私有路径，放在 Git 忽略的 `runs/` 中。原始 `.pt` 不在本地包，必须从服务器读取完整 LoRA、Adam、RNG、sampler checkpoint；CSV 不能替代。

以下变量由当前服务器实际路径赋值；`DM_ROOT` 必须是新的目录，位于原 campaign 外。`DM_TASKS` 是原 `tasks_initial.json`，`DM_PREVIEW` 是原 preview 的 `PLAN.json`，二者须保持原 hash 对应。metadata 命令不加载模型。

```bash
cd /path/to/new-checkout/ssvc_flow
export DM_ROOT=/path/to/new-decision-modeling-run
export DM_TASKS=/path/to/original/tasks_initial.json
export DM_PREVIEW=/path/to/original/preview/PLAN.json
export DM_DATA=/path/to/original/generated

python -m src.decision_modeling.cli plan \
  --config configs/decision_modeling/protocol.json --out "$DM_ROOT/plan.json"
python -m src.decision_modeling.cli prepare-runtime \
  --tasks "$DM_TASKS" --preview "$DM_PREVIEW" --data-root "$DM_DATA" \
  --out "$DM_ROOT/runtime.json"
python -m src.decision_modeling.cli bridge \
  --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/D1" --dry-run
```

运行前一次核验 train/control 原件和 72 张面板图片 hash。推理 fingerprint 在载入 checkpoint 时计算并复用；Adam/RNG 不参与推理别名，但训练恢复必须保留它们。缓存 namespace 包括实现、运行配置及精度/模板信息；变化后写新 namespace，不清空旧缓存。

## 到这里需要服务器 GPU：D1

在原环境的一卡分配内运行，每个进程只可见分配给自己的 GPU。继承的生产 loader 检查单卡、既有环境与 PRO 6000；使用分配后的逻辑 `cuda:0`，没有写死物理 GPU 编号。默认两进程，最多三进程；本代码不申请、抢占或终止其他任务。

```bash
# 分别放在两个独立单卡分配内执行；不要在同一个 GPU 上同时运行。
python -m src.decision_modeling.cli bridge --workers 2 --worker-index 0 \
  --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/D1" --execute-gpu
python -m src.decision_modeling.cli bridge --workers 2 --worker-index 1 \
  --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/D1" --execute-gpu

# 两个 worker 完成后在 CPU 上合并并检查报告。
python -m src.decision_modeling.cli merge-bridge --workers 2 \
  --runtime "$DM_ROOT/runtime.json" --root "$DM_ROOT/D1" --out "$DM_ROOT/D1/merged"
python -m src.decision_modeling.cli report \
  --root "$DM_ROOT/D1/merged" --out "$DM_ROOT/reports_after_D1"
```

`scripts/decision_modeling_gpu.sbatch` 的参数为 `PYTHON PROJECT_ROOT RUN_ROOT CACHE_ROOT COMMAND [ARGS...]`；使用既有冻结环境和已有模型缓存，不下载或升级依赖。按完整 scene 对分片，每个 worker 为 36 prompts、4,608 条生成输出；全局 RNG 流身份保持不变。固定 24 条评分路径诊断仅由 worker 0 执行。合并会验证完整覆盖、checkpoint/config/runtime 身份和原始 chunk 内容，报告只读取 `D1/merged`，避免同时读取 worker 目录重复计数。串行命令仍可不传 worker 参数使用。

D1 在冻结 P 面板上对两个真实 preview 候选各采 32 条/题，另使用共同 ORIGIN 和真正随机 source 的 MIX 各 32 条/题。所有流明确分角色；同一流不因 RAW4/PRESERVE_XI 重复采样。固定前 24 条完整输出来比较 prefix/full/chunk/token。生产评分继续用 prefix；没有通过的加速或梯度路径不启用。有限但不一致的评分保留端点采样、标记禁用相关 LR/质量界；非有限值停止故障任务并保留失败记录。

输出含端点事实、ORIGIN/MIX RAW4/PRESERVE_XI、非对角协方差、各事件支持数和条件质量界；缓存物理评分键、逻辑评分数、生成 token、forward、秒数与峰值显存分列。命中率不是实测加速率。查看 `SCORING_PATHS.json`、`COMPLETE.json`、`ANALYSIS.json`、报告和失败记录后，才进入 D2；不要求全 Jacobian 或小数精度阈值。

## D2：先全部 H8，再继续全部 H32

每条 `block` 命令自己产生 on-policy B4/K8 样本，并自动做 H8/H32 的 P/32 观测；评估后完整恢复训练 RNG。各原点 H0 观测共享于其同级目录 `H00_observations`。先运行 R0 到 H8 创建共享 H0，再将其余不同 arm 分配到两个单卡进程，避免同一共享目录的首次并发写入。每个输出目录只容许一个 writer。

```bash
python -m src.decision_modeling.cli block --origin O1 --recipe R0 --steps 8 \
  --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/O1/R0" --execute-gpu

# O1 其余九臂均先运行到8；每个独立GPU进程执行其中一组，不启动多卡DDP。
for recipe in R1 R2 R3 R4 R5 R6 R7 GDPO_R4 SAW_R4; do
  python -m src.decision_modeling.cli block --origin O1 --recipe "$recipe" --steps 8 \
    --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/O1/$recipe" --execute-gpu
done
python -m src.decision_modeling.cli report \
  --root "$DM_ROOT" --out "$DM_ROOT/reports_O1_H8"

# 读取H8结果后，全部十臂继续；不按效果淘汰arm，不重计H8的更新。
for recipe in R0 R1 R2 R3 R4 R5 R6 R7 GDPO_R4 SAW_R4; do
  python -m src.decision_modeling.cli block --origin O1 --recipe "$recipe" --steps 32 \
    --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/O1/$recipe" --resume --execute-gpu
done

# O2仅以下六臂；先运行R0，之后可并行其他独立臂。
for recipe in R0 R2 R3 R4 GDPO_R4 SAW_R4; do
  python -m src.decision_modeling.cli block --origin O2 --recipe "$recipe" --steps 32 \
    --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/O2/$recipe" --execute-gpu
done
```

失败后使用同一命令加 `--resume`。不可变步提交、原始样本、失败 attempt、恢复状态及 H0/H8/H32 引用均保留。没有把另一臂样本用于训练，没有将 Adam 不同的状态当作同一起点。

## 只对仍可能改变选择的项目追加

H32 从每题32开始。追加必须保留同一目录/端点/面板，按32→128→512，并用 `--append-reason` 写明当前未判定的候选或群体。报告列出仍可能改变选择的对象；脚本不自动启动追加。以下仅展示格式，不代表应当追加 R4。

```bash
python -m src.decision_modeling.cli observe \
  --checkpoint "$DM_ROOT/O1/R4/H32.json" --origin O1 --candidate R4 --horizon 32 \
  --panel P --role endpoint --look 128 --append-reason '填入当前区间仍可能改变选择的具体原因' \
  --runtime "$DM_ROOT/runtime.json" --out "$DM_ROOT/O1/R4/observations/H32" --execute-gpu

# 独立参考使用另一个目录和reference角色；不要混入开发观测。
python -m src.decision_modeling.cli observe \
  --checkpoint "$DM_ROOT/O1/R4/H32.json" --origin O1 --candidate R4 --horizon 32 \
  --panel P --role reference --look 256 --runtime "$DM_ROOT/runtime.json" \
  --out "$DM_ROOT/reference/O1/R4/H32" --execute-gpu
```

reference 仅在重要选择仍未分辨时256→1024，仍带区间，不作为精确真值。独立 discovery 使用 `--role discovery --look 16` 和自己的目录。主报告不使用 discovery 来伪造独立 tail MC。CROSSFIT 的 full-refit 实现可复用旧模块，本轮正式命令默认 RAW4/PRESERVE_XI，没有把控制变量修正套入 RAW 的有界保证。

E 和 O3/O4 需要开发报告后生成的同一 `development_freeze.json`，包含 `config_hash`、`selected_exact_recipe`、`simple_fixed_recipe`、`readout_rule`、`E_candidates` 及两个原点全部 16 个分支 H32 结果的 `development_reports`（path/sha256）。E registry 须含选择项、R0、简单固定基线、GDPO_R4、SAW_R4。传入 `--development-freeze` 后才能执行。该文件应根据实际 O1/O2 结果制作；本地没有伪造选择。O3/O4 仅 R0、所选 exact-family R*、GDPO_R4、SAW_R4；R*=R0 不重复训练。它们是训练阶段迁移，不是新 seed。

## 本地复算与验证

```bash
python -m src.decision_modeling.cli reanalyze \
  --source /path/to/SSVC_V4_EXPERIMENT_DATA_COMPACT_20260919.zip --out runs/decision_modeling_ready/D0
python -m src.decision_modeling.cli report \
  --root runs/decision_modeling_ready --out runs/decision_modeling_ready/report
```

D0 产物包括真实 Parquet 原始事实、Z0–Z4、别名、评分键复用、质量界、LR 均值/协方差及输入/输出完整性收据。本地事实文件保留在忽略的 runs 目录，不把大型历史数据或服务器 checkpoint 提交到公开 Git。所有科学状态与服务器待验证事项见实施报告。
