# SSVC mechanism-followup-v2 服务器交接

初始本地实现 commit 为 `d261d67daba5cac54a6736e366d951ede8d19932`，文档标记提交 `e1ce594591453f3970e96fa1dfcd6997fdc0ee05` 不改变该版源码。该版 CPU 完整套件 1228 项通过；交付前最终修复后全部新增 205 项和参考数学 15 项通过，均无失败或跳过。这些计数对应初始本地验收。

本交接对应分支 `codex/ssvc-mechanism-followup-20260914`，工作目录为该仓库的 `ssvc_flow/`。初始源码版本与 CPU 结果见 [IMPLEMENTATION_REPORT_zh.md](IMPLEMENTATION_REPORT_zh.md) 和 [machine_readable_readiness.json](machine_readable_readiness.json)。用户随后授权服务器验证；CPU 原件核验已通过，首次 5090 smoke 在模型分配前因计划指纹不一致退出。后续修复、验证事实及容量结论边界见 [SERVER_VALIDATION_20260914_zh.md](SERVER_VALIDATION_20260914_zh.md)，不能把初始 CPU 通过记录视为 GPU 通过证据。

## 当前停止点和原件

本地紧凑包只能完成元数据审计。开始服务器步骤前，必须定位完整 R3-warm、R4 及其递归上游原件。不要把紧凑包解压后当作完整 run。`prepare-server` 会实际复核文件、checkpoint、样本、数据、环境证书与源码；缺项或身份不一致时拒绝生成有效计划。

将 [server_paths.example.json](server_paths.example.json) 复制到新结果根目录下的私有 `server_paths.json`，填入服务器实际路径。所有字段必填，不能用 null 或当前目录代替。

| 字段 | 所需原件 |
|---|---|
| parent_warm | R3-warm 完整目录：runtime/bank/candidate manifest、`origin.pt`、`samples.jsonl`、所有原候选 checkpoint 与完整 manifest 链 |
| parent_r4 | 已完成 R4 的最终目录，以及 continuation 引用的所有父目录；原始初始 `origin.pt`、两 arm step64 checkpoint、checkpoint manifest、原始 decision 与所有评估/训练证据 |
| r0_dir | 通过核验的 R0 数据/协议/模型绑定与 L 协议原件 |
| r1_run | 通过核验的 R1 reference-path、模型、processor、tokenizer 证书 |
| supplement_dir | 原 R1 supplement 完整恢复、零梯度与候选顺序证据 |
| r2_dir | 原 R2 完整诊断结果及 panel/input 绑定 |
| r3_cold_dir | 原 R3-cold 原件与固定 control-proposal ledger；每题 sample_index=0 的48条用于 S2 统一诊断 |
| raw_dataset_root | 原认证数据根目录：split JSONL、manifest、图片及独立 L 数据；目录别名 resolve 后须匹配原认证路径 |
| model_snapshot | 完整本地 Hugging Face snapshot，目录名为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，含全部 safetensors 分片、index、config、processor、tokenizer |
| new_run_root | 本轮新的独立结果根目录；不得与任何 parent、数据或模型目录包含/重叠 |

保留原环境，不升级库。历史锁记录 Python 3.12.14、torch 2.13.0+cu130、transformers 5.14.1、peft 0.19.1。现有上游 CPU 验证器还要求固定 revision 的 processor 已在原 Hugging Face 缓存中。本实现本地加载器使用 `local_files_only=True`，不会下载模型。迁移到不同认证数据路径不能仅编辑 PASS 或 JSON 哈希解决，应另行进行路径/输入桥接；本实现会明确拒绝。

## 取得并核对代码

在服务器的既有 checkout 执行只读检查，再切换到本分支；存在未提交工作时保留它，不使用 reset/clean/force。

```bash
git status --short --branch
git fetch origin
git switch codex/ssvc-mechanism-followup-20260914
git pull --ff-only
git rev-parse HEAD
```

若服务器没有本地同名分支，使用 `git switch --track origin/codex/ssvc-mechanism-followup-20260914`。比较交付 commit 和源码清单后，在 `ssvc_flow/` 设置以下已核验变量。所有命令显式要求变量，不回落到当前目录。

```bash
: "${PY:?设置原已核验环境的 Python 绝对路径}"
: "${NEW_ROOT:?设置 server_paths.json 中的 new_run_root}"
: "${PARENT_WARM:?设置完整 R3-warm 原件目录}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
```

## 无模型执行的服务器准备

```bash
"$PY" -m src.followup_cli plan \
  --design configs/mechanism_followup.yaml --out "$NEW_ROOT/plan.json"

"$PY" -m src.followup_cli audit-parent \
  --design configs/mechanism_followup.yaml --parent "$PARENT_WARM" \
  --out "$NEW_ROOT/parent_audit.json"

"$PY" -m src.followup_cli prepare-server \
  --design configs/mechanism_followup.yaml --paths "$NEW_ROOT/server_paths.json" \
  --out "$NEW_ROOT/validated_plan.json"

"$PY" -m src.followup_cli run-s1 \
  --validated-plan "$NEW_ROOT/validated_plan.json" --bank bank03 \
  --out "$NEW_ROOT/S1/bank03" --dry-run
```

计划、审计和已完成证据均不覆盖。重新生成计划时使用新名字；源码或原件变化后必须重新准备并重新测 smoke。`audit-parent` 即使找到原始文件，也只报告元数据通过，不能替代 `prepare-server` 的完整核验。

## 首次需要 GPU 的边界：smoke

**以下命令本次未执行。** 获得实际授权和有效 Slurm GPU allocation 后，再运行：

```bash
"$PY" -m src.followup_cli smoke \
  --validated-plan "$NEW_ROOT/validated_plan.json" \
  --out "$NEW_ROOT/smoke" --allow-gpu-execution
```

smoke 加载固定9B，核验原模型/processor/LoRA/环境证书，在实际 warm bank03 对 joint0 和 joint1 分别执行旧/新直接 PPO 与 Adam：共4次 scratch Adam，无新增采样，不提交永久训练状态。参数、Adam、RNG、模式/buffer 恢复以及冻结参数必须通过。输出 `smoke/smoke_evidence.json`。失败保留证据，不能复制 CPU fixture 的 PASS 来绕过。

## S1：六个预设 bank

每个命令对应一个独立执行单元，按下表顺序运行并检查结果；不要并行保留多套候选 GPU 状态。共享参数如下：

```bash
run_s1() {
  "$PY" -m src.followup_cli run-s1 \
    --validated-plan "$NEW_ROOT/validated_plan.json" \
    --smoke-evidence "$NEW_ROOT/smoke/smoke_evidence.json" \
    --bank "$1" --out "$NEW_ROOT/S1/$1" --allow-gpu-execution
}
```

| 顺序 | 单次命令 | 候选 + joint0重放 | 直接逻辑候选 | 输出上限 |
|---|---|---:|---:|---:|
| 1 | `run_s1 bank00` | 2+1 | 2 | 1,536 |
| 2 | `run_s1 bank03` | 9+1 | 4 | 3,072 |
| 3 | `run_s1 bank04` | 9+1 | 3 | 2,304 |
| 4 | `run_s1 bank05` | 9+1 | 3 | 2,304 |
| 5 | `run_s1 bank11` | 9+1 | 3 | 2,304 |
| 6 | `run_s1 composite_no_x_plus_three_level` | 3+1 | 3 | 2,304 |

合计41个候选、6次重放、最多47次 Adam、1,504个反向序列、13,824条直接输出。实际推理完全相同的候选可按完整稳定指纹去重，但各自 Adam checkpoint 保留，alias 不增加样本量。失败的更新尝试会保守消耗一次可能的 Adam 配额；若无法在冻结上限内完成，停止并保留预算证据，不能通过删记录取得额外配额。生成阶段中断续跑不会重复更新或更换已完成样本的 seed。

每个 bank 检查 `status.json`、`baseline_replay.json`、`parameter_geometry.json`、`policy_aliases.json`、`all_candidate_response.json`、`paired_response.json`、`runtime_profile.json`、`manifest.json` 和 `faults/`。全候选统计采用共享场景 bootstrap 与完整候选/群体 union M；单独 pair 报告明确其较窄范围。结果允许零、不显著或与旧方向相反。

六组技术测量链完成后：

```bash
"$PY" -m src.followup_cli analyze \
  --run-root "$NEW_ROOT/S1" --out "$NEW_ROOT/S1_ANALYSIS"
```

它重新核验全部 bank manifest 和共同 model/data/parser/origin/source/validated-plan 绑定，输出 `S1/s1_measurement_chain.json`。CPU 结果只能产生 CPU_TESTED，不能授权 S2。

## S2：审核 S1 技术链及预算后逐次启动

```bash
run_s2() {
  "$PY" -m src.followup_cli run-s2 \
    --validated-plan "$NEW_ROOT/validated_plan.json" \
    --smoke-evidence "$NEW_ROOT/smoke/smoke_evidence.json" \
    --s1-evidence "$NEW_ROOT/S1/s1_measurement_chain.json" \
    --seed "$1" --arm "$2" --out "$NEW_ROOT/S2/seed$1/$2" \
    --allow-gpu-execution --allow-training
}
```

七个新增 run 的准确命令如下。逐项运行并读取完成结果和日志；不由程序按效果好坏决定补 seed。

| seed / 定位 | 命令 |
|---|---|
| 17 / 已观察探索锚点的新 arm | `run_s2 17 X_VALID_NO_X_OFF` |
| 29 / 前瞻 | `run_s2 29 X_BASE` |
| 29 / 前瞻 | `run_s2 29 X_VALID` |
| 29 / 前瞻 | `run_s2 29 X_VALID_NO_X_OFF` |
| 41 / 前瞻 | `run_s2 41 X_BASE` |
| 41 / 前瞻 | `run_s2 41 X_VALID` |
| 41 / 前瞻 | `run_s2 41 X_VALID_NO_X_OFF` |

旧 seed17 X_BASE/X_VALID 训练不可通过此入口重跑。每个新 run64步×B4×K8=2,048条训练输出；七个合计14,336条。milestone checkpoint0/16/32/48/64；为精确续跑还保存初始状态和每一步，共65份完整 checkpoint/run，服务器空间须按此计入。N-panel72×8、终点 N288×8、L176×8、OOD72×8在0/64、R2原72景×3条件×4在0/64。step64小面板从完整 N ledger 引用。每 run评估8,896条，七个上限62,272条，不能仅按训练样本估算总成本。N、L、OOD、R2分别输出。每个 arm 自己 on-policy 采样。

本实现保留同 seed 初始参数与顺序一致性；跨 arm step0评估目前各自执行，不节省其预算。同 seed17 初始 adapter 绑定历史原件；seed29/41采用确定性 LoRA 初始化。所有臂采用同一份历史已审核诊断警告政策；非有限数值、冻结参数变化、来源变化或恢复失败必须停止。

历史两臂终点可单独补 R2 诊断，不训练、不改写原结果，每臂864条，两臂上限1,728条：

```bash
"$PY" -m src.followup_cli eval-historical-r2 \
  --validated-plan "$NEW_ROOT/validated_plan.json" \
  --smoke-evidence "$NEW_ROOT/smoke/smoke_evidence.json" \
  --arm X_BASE --out "$NEW_ROOT/historical_R2/X_BASE" --allow-gpu-execution

"$PY" -m src.followup_cli eval-historical-r2 \
  --validated-plan "$NEW_ROOT/validated_plan.json" \
  --smoke-evidence "$NEW_ROOT/smoke/smoke_evidence.json" \
  --arm X_VALID --out "$NEW_ROOT/historical_R2/X_VALID" --allow-gpu-execution
```

## Slurm、续跑与停止

已生成 `scripts/followup_smoke.sbatch`、`followup_s1.sbatch`、`followup_s2.sbatch` 并通过 `bash -n`。它们沿用既有模板中的单任务、no-requeue、严格 shell 和环境设置，但没有复制旧 partition/account/QOS/GPU/墙钟值。使用服务器当前获批资源设置以下变量，再由用户提交；不能把本段当作已提交任务的记录。

```bash
export SSVC_PYTHON="$PY" SSVC_PROJECT="$PWD"
export SSVC_PLAN="$NEW_ROOT/validated_plan.json"
export SSVC_OUT="$NEW_ROOT/smoke"
# SSVC_PARTITION/ACCOUNT/QOS/GPU_MODEL/WALLTIME/LOG_DIR 均来自当前获批资源。
sbatch --partition="${SSVC_PARTITION:?}" --account="${SSVC_ACCOUNT:?}" \
  --qos="${SSVC_QOS:?}" --gres="gpu:${SSVC_GPU_MODEL:?}:1" --time="${SSVC_WALLTIME:?}" \
  --output="${SSVC_LOG_DIR:?}/followup-%j.out" --error="$SSVC_LOG_DIR/followup-%j.err" \
  scripts/followup_smoke.sbatch
```

S1模板额外需要 `SSVC_BANK`、`SSVC_SMOKE_EVIDENCE`；S2模板额外需要 `SSVC_SEED`、`SSVC_ARM`、`SSVC_SMOKE_EVIDENCE`、`SSVC_S1_EVIDENCE`。每次设置对应独立 `SSVC_OUT`。上表的普通命令须在获批 GPU allocation 内执行。

续跑用原完整命令追加 `--resume`，保持原 `--out`、来源、plan、arm、seed 和已冻结配置。Slurm模板用 `SSVC_RESUME=1`。smoke不续写已有证据，使用新的空输出目录。取消已有任务使用 `scancel "$JOB_ID"`；只取消明确记录的当前任务。保留所有失败、预算 reservation、ledger 和 checkpoint，不删除 hard_stop，不调整阈值，不覆盖原样本。若配额耗尽或恢复失败，先核对并记录新的处理决定，当前入口拒绝静默重试。

## 回传与解释边界

回传紧凑文件：新 plan/source/parent审计、smoke证据、S1上述JSON与分析、S2 `checkpoint_manifest.json`、`training_steps.json`、`evaluation_summary.json`、completion manifest/binding、每个step的更新/分组/控制诊断、各评估域的counts（包括R2三个condition分项）、S1 runtime profile、S2原始行的elapsed/runtime_forward_calls和step audit、fault/hard_stop与Slurm日志。原始输出、权重、Adam/RNG、图片和绝对服务器路径继续保存在私有结果目录，不上传公共Git。

先逐seed报告 `X_VALID-X_BASE` 和 `X_VALID_NO_X_OFF-X_VALID`，再用 `src.followup_statistics.summarize_seed_results` 汇总；seed17单列为探索，29/41列为前瞻。保留六个分组和重点 `trend/SYMBOLIC_FRESH`。固定面板误差与场景bootstrap分开，稀有事件无支持不等于安全，所有统计保留 `NOT_CERTIFIED`。S3在线筛选、seed53、追加64次/题采样、其他模型和训练优化均未启用。

## 已完成结果的多 seed 汇总

`evaluation_summary.json` 不能直接传给 `summarize_seed_results`。以下只读提取新 run 的 step64 N/L/OOD 结果，并从已核验的原 R4 `endpoint_metrics.json` 读取 seed17 两个历史锚点；输出采用新文件名，不覆盖旧分析。本段本次未对真实新增训练结果执行；源数据字段已按当前源码和已有 R4 元数据核对。执行前读取各 run 的完成记录和日志，并核验 completion manifest/binding。

```bash
"$PY" - <<'PY'
import json
import os
from pathlib import Path
from src.followup_statistics import summarize_seed_results

root = Path(os.environ['NEW_ROOT'])
plan = json.loads((root / 'validated_plan.json').read_text())
old = json.loads((Path(plan['paths']['parent_r4']) / 'endpoint_metrics.json').read_text())
metric_map = {'pX': 'pX', 'v': 'v', 'qX': 'qX_pool', 'qS': 'qS_pool'}
new = {}
for run in plan['design']['S2']['new_runs']:
    seed, arm = run['seed'], run['arm']
    path = root / 'S2' / f'seed{seed}' / arm / 'evaluation_summary.json'
    new[seed, arm] = json.loads(path.read_text())

result = {}
for track in ('N', 'L', 'OOD'):
    historic = old[track]['comparisons']['X_VALID_minus_X_BASE']['responses']
    for scope, values in historic.items():
        records = []
        for arm, field in [('X_BASE', 'reference_estimate'), ('X_VALID', 'target_estimate')]:
            records.append({'seed': 17, 'arm': arm,
                'metrics': {name: values[legacy][field] for name, legacy in metric_map.items()}})
        for (seed, arm), reports in new.items():
            selected = [r for r in reports if r['step'] == 64 and r['track'] == track]
            if len(selected) != 1:
                raise ValueError(f'Expected one {seed}/{arm}/{track}/step64 report')
            report = selected[0]
            values = report['overall'] if scope == 'overall' else report['groups'][scope]
            records.append({'seed': seed, 'arm': arm,
                            'metrics': {name: values[name] for name in metric_map}})
        result[f'{track}/{scope}'] = {
            baseline: summarize_seed_results(records, baseline_arm=baseline)
            for baseline in ('X_BASE', 'X_VALID')}
with (root / 'S2_seed_summary.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
PY
```

在 baseline=`X_BASE` 结果下取 `X_VALID`，以及 baseline=`X_VALID` 下取 `X_VALID_NO_X_OFF`。`per_seed` 保留全部三臂，`prospective_aggregate` 仅包括29/41。这里计算训练seed的描述性汇总，不产生固定面板或场景bootstrap区间；区间需用对应原始配对面板单独计算。

R2单列：新 S2 各 `evaluation/step_00/R2/counts.json`、`evaluation/step_64/R2/counts.json` 和 `evaluation_summary.json` 的 `conditions` / `condition_groups` 分别保留 SYM_ORIGINAL、IMAGE_CUE、IMAGE_ONLY。不要使用其旧兼容 `groups` 字段比较诊断条件，因为两个图像条件共享interface标签。历史补测使用 `historical_R2/<arm>/condition_counts.json` 及其 raw ledger；不与N/OOD合并。
