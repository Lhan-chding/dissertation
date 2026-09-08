# P3 冻结评估执行交接（2026-09-08）

实现对应原计划 §8 P3；本文件是执行说明，**不是已完成 GPU 评估的结果报告**。
9B 已有真实 P1 证据（job 144911），3B、7B 必须分别通过自己的 P1。独立人工 36 张校准图检查尚待记录；P2 缺少 `THEORY_REVIEW` 原文，仍为 PARTIAL。不得把本地 fake/tiny 测试计入模型结果。

## 固定矩阵和门禁

按 **3B → 9B → 7B** 顺序，每次只运行一个模型的一条轨道。脚本只提交单个子任务，操作人负责顺序；六个子任务全部通过且同轨数据、解码协议一致，汇总才能给 P3 PASS。

| 轨道 | 固定评估集合 | 每 prompt 输出 | 每模型总输出 | token 上限 |
| --- | --- | --- | --- | --- |
| L | 原 manifest 全部 dev/test/positive_control，共 176 场景、88 配对单元 | 16 sampled + 1 greedy | 2,992 | 48 |
| N | dev144 场景 × 两接口，288 prompts | 16 sampled + 1 greedy | 4,896 | 64 |

完整三模型矩阵共 23,664 条输出；不使用 confirm，不因类别稀少筛选或自动扩样。冻结模型不执行 optimizer/backward，不载入 P1 更新后的 LoRA；使用新加载基座并禁用 adapter。L 保留原单 user prompt、parser 和 executor，但历史 checkpoint/实际模板等证据不全，始终 `legacy_exact_reproduction=false`。

| key | 官方模型 | 固定 revision |
| --- | --- | --- |
| qwen25vl_3b | Qwen/Qwen2.5-VL-3B-Instruct | 66285546d2b821cf421d4f5eb2576359d3770cd3 |
| qwen35_9b | Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| qwen25vl_7b | Qwen/Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 |

配置为 `configs/frozen.json`。revision 固定只证明选择不可变版本，不能替代每模型 GPU 资格测试。P3 检查 P1 的真实 CUDA 状态、manifest 哈希、模型版本、依赖、processor/tokenizer/template 和概率执行路径；身份漂移须先复验。续跑还锁定源码、完整依赖版本、数据、配置和每条记录哈希。

## 准备与分模型 P1

在登录节点更新已检查无本地改动的 checkout，使用现有个人环境；不要重新安装或覆盖已经通过 P1 的 PyTorch 栈。P3 新增 `pyarrow==25.0.0`（完整安装清单为 `requirements-frozen.txt`），可只补此包：

```bash
export SSVC_WORK=/projects/varunssd/louis-ssvc
export SSVC_PYTHON="$SSVC_WORK/envs/ssvc-py312/bin/python"
export PIP_CACHE_DIR="$SSVC_WORK/cache/pip" TMPDIR="$SSVC_WORK/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$SSVC_WORK/logs"
"$SSVC_PYTHON" -I -m pip --isolated install --only-binary=:all: \
  --prefix="$SSVC_WORK/envs/ssvc-py312" --cache-dir="$PIP_CACHE_DIR" \
  --index-url=https://pypi.org/simple pyarrow==25.0.0
"$SSVC_PYTHON" -I -m pip check
cd "$SSVC_WORK/dissertation-ssvc"
export SSVC_MODEL=qwen25vl_3b SSVC_CONFIG=configs/frozen.json
sbatch --test-only ssvc_flow/scripts/ntu_p1.sbatch
# test-only 成功后才提交一次：
sbatch ssvc_flow/scripts/ntu_p1.sbatch
```

P1 默认脚本使用允许 pro6000 的项目 QoS。不指定 CPU/内存，让集群按 GPU 自动分配。3B 通过后保存该 job 的 `runs/P1`；后续 7B 改模型 key 同样执行。9B 可引用保留的 `/projects/varunssd/louis-ssvc/runs/slurm-144911-attempt-0/runs/P1`，若实际依赖或模板不一致，门禁会要求复验。

## 人工校准记录

检查 `docs/local_evidence/P0/calibration_contact_sheet_01.png`、`calibration_contact_sheet_02.png` 的全部 36 场景，确认图像、真值、操作和注入错误与标注一致。原 manifest 是 `docs/local_evidence/P0/contact_sheets_manifest.json`。

由实际检查者另存 JSON：`status` 为 `PASS`，`reviewer`、`reviewed_at` 非空，`contact_manifest_sha256` 为该 manifest 文件哈希，`checked_scene_ids` 精确包含其中全部 36 个 `base_scene_id`。代码同时核对 calibration 数据和图片哈希。不得自动生成一份 PASS 代替人工检查；发现问题需保留检查结果并修复、重新核验数据。

## 单条 P3 任务

以下在 **3B P1 通过且人工检查完成后**使用。把变量中的实际 P1 路径和人工记录路径换成已验证文件，不能原样保留占位符：

```bash
export SSVC_CONFIG="$SSVC_WORK/dissertation-ssvc/ssvc_flow/configs/frozen.json"
export SSVC_MODEL=qwen25vl_3b SSVC_TRACK=N SSVC_RESUME=0
export SSVC_P1_RUN="$SSVC_WORK/runs/slurm-实际P1作业号-attempt-0/runs/P1"
export SSVC_DATA_ROOT="$SSVC_WORK/runs/slurm-144911-attempt-0/data/generated"
export SSVC_HUMAN_REVIEW="$SSVC_WORK/reviews/实际人工检查记录.json"
export SSVC_FROZEN_OUT="$SSVC_WORK/runs/P3/qwen25vl_3b/N"
sbatch --test-only ssvc_flow/scripts/ntu_p3.sbatch
sbatch ssvc_flow/scripts/ntu_p3.sbatch
```

L 子任务使用同模型 P1，另设 `SSVC_TRACK=L`、独立 `SSVC_FROZEN_OUT`，以及：

```bash
export SSVC_LEGACY_DATA="$SSVC_WORK/dissertation-ssvc/artifacts/v5/study_c2/data/reward_fibers.jsonl"
export SSVC_LEGACY_MANIFEST="$SSVC_WORK/dissertation-ssvc/artifacts/v5/study_c2/data/reward_fibers_manifest.json"
```

L 不需要 N 人工检查文件。原表共 464 行，程序验证全部原 prompt 哈希与配对结构后只读取上述 176 个评估场景；不从摘要硬编码数量。原 JSONL SHA256 为 `ebfe4dd7b2acc2afeaea7aa633dcf2f0f7af1eaaceda8af808720c087dd36719`。

脚本的 12 小时是调度上限，**不是预计耗时**；用实际 `progress.json`、生成秒数和已完成条数估算。断开 SSH 不会取消 sbatch。查队列用 `squeue -j 实际作业号`，查已结束状态用 `sacct -X -j 实际作业号 --format=JobID,State,ExitCode,Elapsed`。

中断后，确认原作业已结束，保持所有身份与目录不变，设置 `SSVC_RESUME=1` 再提交。每条 sample 独立确定 seed，已写入记录经哈希验证后跳过；同目录并发写入会被拒绝。源码/依赖改变时不能覆盖旧结果，应选择新目录重跑。原异常概率记录也不能靠 resume 绕过。

## 产物和验收

每子任务保存原始 ledger、模型/运行锁、数据与门禁记录，以及 `metrics.json`、真实 Parquet `metrics_by_scene.parquet`、`metrics_by_group.csv`、`results_report_zh.md`、`table_status.json`。逐 prompt 置信区间、K 组概率区间和实际不重叠组统计均保留；sampled 与 greedy 分开，零计数保留上界，零分母为缺失值。

```bash
cd "$SSVC_WORK/dissertation-ssvc/ssvc_flow"
"$SSVC_PYTHON" -m src.report --run-root "$SSVC_WORK/runs" \
  --out "$SSVC_WORK/reports/p3-review"
```

汇总另导出 `frozen_support_comparison.json`、CSV 和中文说明，按轨道/接口及预先固定的 pX、v 二维区间 `[0,.25), [.25,.5), [.5,.75), [.75,1]` 做辅助描述。各模型独立分箱，保留共同 prompt 数、缺失模型和空区间；这不是配对因果效应。全部 sampled/greedy overall 原样保留，greedy 不参与分箱。

汇总只导入 manifest 验证后的摘要；CPU 结果改标签、同一结果复制为多个模型、不同数据版本拼矩阵均不能通过验收。原始日志/配置/样本留本地和服务器，不随代码推送到公共 Git。

冻结输出只能用于初始支持和后续可观测性诊断，不能证明奖励训练安全或模型规模因果效应。P4–P9 仍未执行。

实现依据：[Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)、[Arrow Parquet API](https://arrow.apache.org/docs/python/parquet.html)；具体调用另有当前版本 tiny/fixture 回归测试。

## 本地交付验证

2026-09-08，Python 3.12 本地 CPU 环境，启用与 P1 相同版本的 Transformers/PEFT 及 pyarrow 后，`python -m pytest tests -q`：**324 passed（38.52 秒）**。包括真实 tiny Qwen3.5/Qwen2.5、冻结与恢复、PEFT 标志恢复、CLI 并发保护、旧 parser/manifest、矩阵身份和辅助比较测试。Ruff、格式检查、Bash 语法及 Git diff 检查通过；CLI dry-run 实测 N 4,896 / L 2,992 条。此证据只验证代码，不代表三模型 P3 GPU 结果。
