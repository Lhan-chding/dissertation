# Qwen v5 单卡 4090 决定性 pilot 运行手册

本手册只运行已经冻结的低成本 pilot，不运行论文最终规模实验。执行顺序是 Study A、Study B，随后仅在注册停止信号未触发时运行 Study C。所有命令都要求本地模型、adapter 和 Python wheel 已经在服务器上；运行期间禁止下载。

## 1. 固定代码与输入

服务器仓库位置固定为 `/cloud/cloud-ssd1/dissertation`。本次实验实现对应代码提交 `ad6d041`；含本手册的交付提交可以更新，但必须包含该提交。

```bash
export V5_REPO=/cloud/cloud-ssd1/dissertation
cd "$V5_REPO"
set -euo pipefail
git fetch origin codex/qwen-v5-structural-support
git checkout codex/qwen-v5-structural-support
git pull --ff-only origin codex/qwen-v5-structural-support
git merge-base --is-ancestor ad6d041 HEAD

export PYTHONPATH="$V5_REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export V5_MODEL=/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct
export V5_T_ADAPTER="$V5_REPO/artifacts/v4/training/runs/phase4-r1/T_constraint_recovery/final_adapter"
export V5_RAW_ARCHIVE=/absolute/path/to/qwen_v4_raw_rows_20260821.tar.gz
```

以下值已经冻结；任意不一致都应停止，不要修改配置绕过检查。

| 对象 | SHA-256 |
|---|---|
| Base 模型快照 | `e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87` |
| T adapter tree | `807a61c2e3f7b532b162554dee6e7df83d654fb1f10cc464e9dcb5f6f8efd5c7` |
| v4 raw archive | `f0ccb4d56415eecf90a2c456bfd7c92a33fc96a581f3603115edbcb253ba8c84` |
| Phase 2a parent manifest | `152d2f6ce7a473449578396c6471e876c682b3aded38c8b70e599de532ecb3a3` |
| Phase 2a rows | `6506330bed1d86cc040eb041a6d9b80697a261343c10325cf1e78811d44a9d9d` |
| Study B config | `3832a84e9bfefdf04bb4bdae65a780d975a738df04e56b811a7741884d3f9381` |
| Study C config | `f167c20904f9720f7c606609d89f58d64921b4678e2352eabf5e80fb2d921984` |
| Server package lock | `a8f351db7cadc904f6feecdd9cddb9e0d782c8356226cd866661710623b2e544` |
| GPU requirements lock | `d928379a590e5071d9b5042fe99d480f57ab187f0cb3a74e13af219a6048aeb3` |

先验证文件和固定目录 tree hash：

```bash
test -d "$V5_MODEL"
test -d "$V5_T_ADAPTER"
test -f "$V5_RAW_ARCHIVE"

sha256sum \
  "$V5_RAW_ARCHIVE" \
  artifacts/v5/data/factorial_pre_model/parent_manifest.json \
  artifacts/v5/data/factorial_pre_model/pre_model_rows.jsonl \
  configs/v5/budget_matched_lora.yaml \
  configs/v5/common_space_grpo.yaml \
  configs/v5/server_package_lock.yaml \
  requirements-gpu.lock.txt

.venv/bin/python -c 'from pathlib import Path; from compensability_v4.qwen.model_loader import require_server_model; from compensability_v4.qwen.phase5_runtime import tree_sha256; print(require_server_model()); print(tree_sha256(Path("'"$V5_T_ADAPTER"'")))'
```

最后一条命令应打印固定 Base 路径，并打印上表中的 T adapter tree SHA-256。

## 2. 环境预检

Python 固定为 3.12，CUDA 固定为 12.8。优先使用服务器已有的精确环境。若需要重建，只能从服务器已有的离线 wheelhouse 安装，例如：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links=/absolute/path/to/wheelhouse -r requirements-gpu.lock.txt
```

不得去掉 `--no-index` 临时联网补包。验证环境和 CPU 侧契约：

```bash
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch,transformers,peft,trl,accelerate,datasets,pyarrow; print(torch.__version__, torch.version.cuda); print(transformers.__version__, peft.__version__, trl.__version__, accelerate.__version__, datasets.__version__, pyarrow.__version__); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/v5

PYTHONPATH=src .venv/bin/python scripts/v5/05_build_budget_matched_support.py --fixture-dry-run
PYTHONPATH=src .venv/bin/python scripts/v5/08_freeze_common_space_rl.py --fixture-dry-run
PYTHONPATH=src .venv/bin/python scripts/v5/14_run_study_c.py --fixture-dry-run
PYTHONPATH=src .venv/bin/python scripts/v5/11_build_advisor_packet.py --fixture-dry-run
```

版本必须与 `configs/v5/server_package_lock.yaml` 完全一致。fixture 命令不会加载 GPU 模型、训练或写正式结果。

## 3. Study A：自然 observation、answer fiber 与五轴 orbit

Study A 只做推理。它先以 Base 对 96 个 familiar scene 捕获自然 observation，生成 append-only Phase 2a child manifest；随后对 Base/T 运行 canonical、variable permutation、error location、fact order、equivalent basis 五轴审计，并另行生成与 Study B 训练场景独立的 legacy 评估集。

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/12_run_study_a.py \
  --execute \
  --ack I_UNDERSTAND_THIS_RUNS_V5_STUDY_A_INFERENCE \
  --phase2a-root artifacts/v5/data/factorial_pre_model \
  --child-root artifacts/v5/data/phase2a_natural_observations \
  --raw-archive "$V5_RAW_ARCHIVE" \
  --t-adapter "$V5_T_ADAPTER" \
  --output-root artifacts/v5/audits/study_a_4090_pilot \
  --work-root artifacts/v5/audits_work/study_a_4090_pilot
```

成功后必须存在：

```bash
test -f artifacts/v5/data/phase2a_natural_observations/child_manifest.json
test -f artifacts/v5/data/phase2a_natural_observations/frozen_scenes.jsonl
test -f artifacts/v5/audits/study_a_4090_pilot/summary.json
test -f artifacts/v5/audits/study_a_4090_pilot/per_scenario.jsonl
test -f artifacts/v5/audits/study_a_4090_pilot/legacy_independent_per_scenario.jsonl
```

正式输出是不可覆盖的。中断后保留 `artifacts/v5/audits_work/` 的 trace，并以完全相同的命令恢复；不要删除 partial trace 后重新挑选 scene。

## 4. 冻结 Study B 的 B0–B3 训练支持集

该步骤使用真实 Qwen tokenizer 统计 completion token。每个 arm 固定为 96 个 source、每 source 6 行，共 576 行；单卡 4090、batch 1、gradient accumulation 8、1 epoch、72 optimizer steps、LoRA rank 16、七个 target module、seed `2026082201`。

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/05_build_budget_matched_support.py \
  --execute \
  --input-jsonl artifacts/v5/data/phase2a_natural_observations/frozen_scenes.jsonl \
  --parent-manifest artifacts/v5/data/factorial_pre_model/parent_manifest.json \
  --child-manifest artifacts/v5/data/phase2a_natural_observations/child_manifest.json \
  --token-counter compensability_v5.qwen.study_b_token_counter:count_completion_tokens \
  --output artifacts/v5/data/budget_matched_support.json
```

计算所有运行时绑定哈希：

```bash
export V5_SUPPORT=artifacts/v5/data/budget_matched_support.json
export V5_PARENT=artifacts/v5/data/factorial_pre_model/parent_manifest.json
export V5_CHILD=artifacts/v5/data/phase2a_natural_observations/child_manifest.json
export V5_SCENES=artifacts/v5/data/phase2a_natural_observations/frozen_scenes.jsonl
export V5_B_EVAL=artifacts/v5/audits/study_a_4090_pilot/legacy_independent_per_scenario.jsonl
export V5_B_CONFIG=configs/v5/budget_matched_lora.yaml
export V5_LOCK=configs/v5/server_package_lock.yaml
export V5_B_ROOT=artifacts/v5/study_b/pilot-2026082201

export V5_SUPPORT_SHA=$(sha256sum "$V5_SUPPORT" | awk '{print $1}')
export V5_PARENT_SHA=$(sha256sum "$V5_PARENT" | awk '{print $1}')
export V5_CHILD_SHA=$(sha256sum "$V5_CHILD" | awk '{print $1}')
export V5_SCENES_SHA=$(sha256sum "$V5_SCENES" | awk '{print $1}')
export V5_B_EVAL_SHA=$(sha256sum "$V5_B_EVAL" | awk '{print $1}')
export V5_B_CONFIG_SHA=$(sha256sum "$V5_B_CONFIG" | awk '{print $1}')
export V5_LOCK_SHA=$(sha256sum "$V5_LOCK" | awk '{print $1}')
```

## 5. Study B：单种子 budget-matched LoRA

首次运行：

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/13_run_study_b.py \
  --execute \
  --ack I_UNDERSTAND_THIS_STARTS_V5_STUDY_B_ON_ONE_4090 \
  --support-package "$V5_SUPPORT" \
  --support-sha256 "$V5_SUPPORT_SHA" \
  --parent-manifest "$V5_PARENT" \
  --parent-manifest-sha256 "$V5_PARENT_SHA" \
  --child-manifest "$V5_CHILD" \
  --child-manifest-sha256 "$V5_CHILD_SHA" \
  --frozen-scenes "$V5_SCENES" \
  --frozen-scenes-sha256 "$V5_SCENES_SHA" \
  --evaluation "$V5_B_EVAL" \
  --evaluation-sha256 "$V5_B_EVAL_SHA" \
  --config "$V5_B_CONFIG" \
  --config-sha256 "$V5_B_CONFIG_SHA" \
  --package-lock "$V5_LOCK" \
  --package-lock-sha256 "$V5_LOCK_SHA" \
  --model-path "$V5_MODEL" \
  --model-sha256 e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87 \
  --output-root artifacts/v5/study_b \
  --output "$V5_B_ROOT"
```

若中断且 `$V5_B_ROOT` 已存在，使用完全相同的命令并在末尾增加 `--resume`。成功标志是：

```bash
test -f "$V5_B_ROOT/completed.json"
.venv/bin/python -c 'import json,os; p=json.load(open(os.environ["V5_B_ROOT"]+"/completed.json")); print(json.dumps({"status":p["status"],"stop_signal":p["stop_signal"]},indent=2,sort_keys=True))'
```

如果 `stop_signal.triggered` 为 `true`，按照预注册规则停止，不运行 Study C，直接执行第 7 节的“注册早停导出”。如果为 `false`，继续 Study C。

## 6. Study C：共同动作空间的 reward-only GRPO 对照

先冻结共同动作数据。B3 是主分析初始化，B2 仅为可选次分析；Base hash 仍写入冻结 manifest 以闭合 provenance。

```bash
export V5_B3_ADAPTER="$V5_B_ROOT/arms/B3/final_adapter"
export V5_B2_ADAPTER="$V5_B_ROOT/arms/B2/final_adapter"
export V5_B3_SHA=$(PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from compensability_v5.qwen.study_b_runtime import tree_sha256; import os; print(tree_sha256(Path(os.environ["V5_B3_ADAPTER"])))')
export V5_B2_SHA=$(PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from compensability_v5.qwen.study_b_runtime import tree_sha256; import os; print(tree_sha256(Path(os.environ["V5_B2_ADAPTER"])))')

PYTHONPATH=src .venv/bin/python scripts/v5/08_freeze_common_space_rl.py \
  --execute \
  --input-jsonl "$V5_SCENES" \
  --study-a-rows artifacts/v5/audits/study_a_4090_pilot/per_scenario.jsonl \
  --b3-initialization-sha256 "$V5_B3_SHA" \
  --b2-initialization-sha256 "$V5_B2_SHA" \
  --base-initialization-sha256 e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87 \
  --output artifacts/v5/data/common_space_rl.json

export V5_C_MANIFEST=artifacts/v5/data/common_space_rl.json
export V5_C_CONFIG=configs/v5/common_space_grpo.yaml
export V5_C_MANIFEST_SHA=$(sha256sum "$V5_C_MANIFEST" | awk '{print $1}')
export V5_C_CONFIG_SHA=$(sha256sum "$V5_C_CONFIG" | awk '{print $1}')
export V5_C_ROOT=artifacts/v5/rl/study-c-pilot
```

先做服务器 preflight；该命令验证 CUDA/TRL 接口、输入、adapter 和所有哈希，但不开始训练：

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/14_run_study_c.py \
  --execute \
  --preflight-only \
  --ack I_UNDERSTAND_THIS_STARTS_V5_STUDY_C_GRPO \
  --config "$V5_C_CONFIG" \
  --config-sha256 "$V5_C_CONFIG_SHA" \
  --package-lock "$V5_LOCK" \
  --package-lock-sha256 "$V5_LOCK_SHA" \
  --common-action-manifest "$V5_C_MANIFEST" \
  --common-action-manifest-sha256 "$V5_C_MANIFEST_SHA" \
  --b3-adapter "$V5_B3_ADAPTER" \
  --b3-adapter-sha256 "$V5_B3_SHA" \
  --model-path "$V5_MODEL"
```

主分析只运行 B3 answer reward 与 B3 exact-state reward：

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/14_run_study_c.py \
  --execute \
  --ack I_UNDERSTAND_THIS_STARTS_V5_STUDY_C_GRPO \
  --config "$V5_C_CONFIG" \
  --config-sha256 "$V5_C_CONFIG_SHA" \
  --package-lock "$V5_LOCK" \
  --package-lock-sha256 "$V5_LOCK_SHA" \
  --common-action-manifest "$V5_C_MANIFEST" \
  --common-action-manifest-sha256 "$V5_C_MANIFEST_SHA" \
  --b3-adapter "$V5_B3_ADAPTER" \
  --b3-adapter-sha256 "$V5_B3_SHA" \
  --model-path "$V5_MODEL" \
  --output-root "$V5_C_ROOT"
```

中断后以完全相同的命令增加 `--resume`。仅当需要预注册的 B2 次分析时，preflight 和正式命令都增加：

```bash
--include-b2 --b2-adapter "$V5_B2_ADAPTER" --b2-adapter-sha256 "$V5_B2_SHA"
```

成功输出为 `$V5_C_ROOT/study_c_summary.json`。训练 trace 与固定的 16-rollout post-training evaluation 分开保存；最终汇总只使用冻结 evaluation rows，不把 on-policy training rollout 冒充最终评估。

## 7. 导出事实报告与原始文本证据

输出目录必须不存在，脚本拒绝覆盖。最终包只包含允许的 UTF-8 文本证据，不包含 adapter、checkpoint、模型权重或图片。

### Study B 注册早停导出

仅在 `stop_signal.triggered == true` 时使用：

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/11_build_advisor_packet.py \
  --execute \
  --confirmation-results artifacts/v5/audits/study_a_4090_pilot/summary.json \
  --support-results "$V5_B_ROOT/completed.json" \
  --study-a-root artifacts/v5/audits/study_a_4090_pilot \
  --study-b-root "$V5_B_ROOT" \
  --output artifacts/v5/evaluation/advisor_packet
```

状态应为 `PARTIAL_DECISIVE_PILOT`，并明确记录 `NOT_RUN_DUE_TO_REGISTERED_STOP`；这不是缺失实验伪装成完整结果。

### Study C 完成后的完整 pilot 导出

```bash
PYTHONPATH=src .venv/bin/python scripts/v5/11_build_advisor_packet.py \
  --execute \
  --confirmation-results artifacts/v5/audits/study_a_4090_pilot/summary.json \
  --support-results "$V5_B_ROOT/completed.json" \
  --reward-results "$V5_C_ROOT/study_c_summary.json" \
  --study-a-root artifacts/v5/audits/study_a_4090_pilot \
  --study-b-root "$V5_B_ROOT" \
  --study-c-root "$V5_C_ROOT" \
  --output artifacts/v5/evaluation/advisor_packet
```

验收：

```bash
test -f artifacts/v5/evaluation/advisor_packet/QWEN_V5_PILOT_RESULT_FACTS.md
test -f artifacts/v5/evaluation/advisor_packet/qwen_v5_pilot_raw_rows.tar.gz
test -f artifacts/v5/evaluation/advisor_packet/sha256_manifest.json
sha256sum artifacts/v5/evaluation/advisor_packet/*
```

将 `artifacts/v5/evaluation/advisor_packet/` 整个目录复制回本地即可。不要在服务器上补跑多种子、7B、A100、论文最终规模或完整 gradient alignment；这些不属于本次单卡决定性 pilot。
