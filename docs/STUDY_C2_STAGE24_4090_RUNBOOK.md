# Study C2 Stage 24 shared-batch gradient boundary

Stage 23 returned 6,144 frozen B3 rollouts with `X=143`, `S=635`, `F=655`, and
`U=4711`. Its logical status is `REWARD_CONTRAST_IDENTIFIED`, and the registered group size is
`K=8`. Stage 24 therefore measures both verifier gradients on the same 768 frozen groups.

This stage performs no optimizer step, training, or RL. It loads the immutable B3 LoRA as
trainable only so autograd can measure
`sum(group-centered advantage * sequence log probability)` for both reward vectors. It writes one
hash-bound diagnostic row per group and reports collision/separating, family, X/S/F/U, RDGR,
ESGR, reward Hamming distance, gradient norms, difference norm, and cosine.

The command runs in the foreground with unbuffered progress. It does not use `tail`, `tee`,
`nohup`, or background execution. Run it inside the existing `qwen-v5` tmux session.

```bash
run_study_c2_stage24() {
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
  export V5_C2_STAGE24_CONTRACT=artifacts/v5/study_c2/stage24_execution_contract.json

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

  echo "[1/2] Stage 24 complete preflight; no gradient forward/backward"
  .venv/bin/python -u scripts/v5/study_c2/24_shared_batch_reward_gradient_audit.py \
    --preflight-only \
    --execution-contract "$V5_C2_STAGE24_CONTRACT" \
    --b3-adapter "$V5_B3_ADAPTER" \
    --b3-sha256 "$V5_B3_SHA"
  C2_RC=$?
  echo "Stage 24 preflight exit code: $C2_RC"
  if [ "$C2_RC" -ne 0 ]; then
    return "$C2_RC"
  fi

  echo "[2/2] Stage 24 foreground audit: 768 shared K=8 groups"
  .venv/bin/python -u scripts/v5/study_c2/24_shared_batch_reward_gradient_audit.py \
    --execute \
    --ack I_UNDERSTAND_THIS_RUNS_STUDY_C2_SHARED_BATCH_GRADIENT_AUDIT \
    --execution-contract "$V5_C2_STAGE24_CONTRACT" \
    --b3-adapter "$V5_B3_ADAPTER" \
    --b3-sha256 "$V5_B3_SHA"
  C2_RC=$?
  echo "Study C2 Stage 24 exit code: $C2_RC"
  if [ "$C2_RC" -eq 0 ]; then
    .venv/bin/python -m json.tool \
      artifacts/v5/study_c2/shared_gradient_audit/summary.json
    sha256sum \
      artifacts/v5/study_c2/shared_gradient_audit/per_group.jsonl \
      artifacts/v5/study_c2/shared_gradient_audit/summary.json \
      artifacts/v5/study_c2/shared_gradient_audit/manifest.json
  fi
  echo "SSH and tmux remain open."
  return "$C2_RC"
}

run_study_c2_stage24
unset -f run_study_c2_stage24
```

If transport disconnects, reconnect and run `tmux attach -t qwen-v5`. A partial append-only trace
is validated and resumed at the next unmeasured group. A completed output is never overwritten.
