# Study C2 first 4090 boundary

Study C2 stages 21 and 22 are deterministic CPU stages. Their frozen rows and manifests are
versioned under `artifacts/v5/study_c2/data/`. Stage 23 is the first command that loads Qwen and
the immutable Study B B3 adapter.

This handoff deliberately stops at that first model-dependent experiment. Stages 24--40 expose
fixture/preflight command surfaces but their real executions remain fail-closed until the frozen
Stage 23 support evidence exists. This pre-GPU delivery does not claim that the post-boundary
gradient, GRPO, evaluation, or derived audits have already been executed.

The sampler runs in the foreground with unbuffered output. It prints every rollout directly to
the attached terminal. It does not use `tail`, background execution, `nohup`, or a log-only
workflow. Use an attached `tmux` session so an SSH transport failure does not terminate it.

The server invocation must obtain the expected B3 hash from the completed Study B arm result,
then independently recompute the adapter tree hash during preflight. The stage refuses downloads,
multiple GPUs, a non-4090 GPU, a model snapshot mismatch, adapter drift, non-96 support input, or
an incomplete offline environment.

A successful preflight reports `STUDY_C2_FROZEN_SUPPORT_PREFLIGHT_OK`; the actual 6,144-rollout
measurement is a separate explicitly acknowledged invocation. Paste the following block inside the
existing `qwen-v5` tmux session:

```bash
run_study_c2_boundary() {
  set +e

  cd /cloud/cloud-ssd1/dissertation || {
    echo "BLOCKED: repository directory is unavailable"
    return 2
  }

  git fetch origin codex/qwen-v5-structural-support || return $?
  git checkout --detach origin/codex/qwen-v5-structural-support || return $?

  export PYTHONPATH="$PWD/src"
  export PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false

  export V5_B_ROOT=artifacts/v5/study_b/pilot-2026082201
  export V5_B3_ADAPTER="$V5_B_ROOT/arms/B3/final_adapter"
  export V5_B3_RESULT="$V5_B_ROOT/arms/B3/result.json"

  test -x .venv/bin/python || {
    echo "BLOCKED: .venv/bin/python is unavailable"
    return 2
  }
  test -f "$V5_B3_RESULT" || {
    echo "BLOCKED: Study B B3 result.json is unavailable"
    return 2
  }

  export V5_B3_SHA
  V5_B3_SHA=$(.venv/bin/python -c '
import json, os
from pathlib import Path
p = Path(os.environ["V5_B3_RESULT"])
v = json.loads(p.read_text(encoding="utf-8"))
assert v["status"] == "STUDY_B_ARM_COMPLETE"
assert v["arm"] == "B3"
print(v["adapter_tree_sha256"])
') || return $?

  echo "B3 registered SHA-256: $V5_B3_SHA"
  echo "[1/3] Read-only legacy Study C parser audit"
  .venv/bin/python -u scripts/v5/study_c2/20_audit_legacy_study_c_parser.py \
    --execute \
    --legacy-root artifacts/v5/rl/study-c-pilot
  C2_RC=$?
  echo "Stage 20 exit code: $C2_RC"
  if [ "$C2_RC" -ne 0 ]; then
    return "$C2_RC"
  fi

  echo "[2/3] Stage 23 preflight; no inference"
  .venv/bin/python -u scripts/v5/study_c2/23_measure_frozen_policy_support.py \
    --preflight-only \
    --b3-adapter "$V5_B3_ADAPTER" \
    --b3-sha256 "$V5_B3_SHA"
  C2_RC=$?
  echo "Stage 23 preflight exit code: $C2_RC"
  if [ "$C2_RC" -ne 0 ]; then
    return "$C2_RC"
  fi

  echo "[3/3] Stage 23 foreground support measurement: 96 prompts x 64 rollouts"
  .venv/bin/python -u scripts/v5/study_c2/23_measure_frozen_policy_support.py \
    --execute \
    --ack I_UNDERSTAND_THIS_RUNS_STUDY_C2_FROZEN_B3_SUPPORT \
    --b3-adapter "$V5_B3_ADAPTER" \
    --b3-sha256 "$V5_B3_SHA"
  C2_RC=$?
  echo "Study C2 boundary exit code: $C2_RC"
  echo "SSH and tmux remain open."
  return "$C2_RC"
}

run_study_c2_boundary
unset -f run_study_c2_boundary
```
