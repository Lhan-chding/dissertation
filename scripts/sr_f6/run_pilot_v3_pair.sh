#!/usr/bin/env bash
#SBATCH --no-requeue
# Fixed V3 only, inside an already granted LH-SRF6 two-GPU allocation.
# Before submission, separately verify teacher QoS and actual/pending/unknown GPU use.
# This launcher never submits allocations or retries failed/unknown model work.
set -euo pipefail
: "${SLURM_JOB_ID:?existing two-GPU allocation required}"
: "${SLURM_JOB_NAME:?LH-SRF6 job name required}"
[[ "$SLURM_JOB_NAME" == LH-SRF6-* ]] || { echo 'Expected LH-SRF6- job prefix' >&2; exit 2; }
: "${SRF6_CODE:?immutable PILOT deployment required}"
: "${SRF6_ROOT:?registered experiment root required}"
: "${SRF6_PLAN_SHA256:?frozen V3 jobs plan hash required}"
: "${SRF6_PHASE_IDENTITY_SHA256:?frozen V3 phase hash required}"
: "${SRF6_PILOT_SPEC_SHA256:?frozen V3 complete pilot spec hash required}"
: "${SRF6_ADMISSION_SHA256:?independent V3 admission hash required}"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || { echo "Requeued execution is not authorized" >&2; exit 2; }
: "${SRF6_ROUND:?fixed A, B, or EVAL round required}"
: "${SRF6_ATTEMPT:?new unique worker attempt ID required}"
case "$SRF6_ROUND" in A|B|EVAL) ;; *) echo 'Unknown registered round' >&2; exit 2 ;; esac
cd "$SRF6_CODE"
export PYTHONPATH="$SRF6_CODE/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
# Additional arguments are only explicit --resume-task TASK_ID pairs, validated
# against each fixed worker queue by the Python entry. Use the per-worker arrays.
resume0=()
resume1=()
if [[ -n "${SRF6_RESUME_TASK_0:-}" ]]; then resume0=(--resume-task "$SRF6_RESUME_TASK_0"); fi
if [[ -n "${SRF6_RESUME_TASK_1:-}" ]]; then resume1=(--resume-task "$SRF6_RESUME_TASK_1"); fi
pid0=''
pid1=''
stop_owned_steps() {
  trap - TERM INT
  [[ -z "$pid0" ]] || kill -TERM "$pid0" 2>/dev/null || true
  [[ -z "$pid1" ]] || kill -TERM "$pid1" 2>/dev/null || true
  [[ -z "$pid0" ]] || wait "$pid0" || true
  [[ -z "$pid1" ]] || wait "$pid1" || true
  exit "$1"
}
trap 'stop_owned_steps 143' TERM
trap 'stop_owned_steps 130' INT
srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=33G \
  "$python" -u -m sr_f6.pilot_jobs worker --root "$SRF6_ROOT" \
  --phase PILOT_V3 --phase-identity-sha256 "$SRF6_PHASE_IDENTITY_SHA256" \
  --pilot-spec-sha256 "$SRF6_PILOT_SPEC_SHA256" --admission-sha256 "$SRF6_ADMISSION_SHA256" \
  --plan-sha256 "$SRF6_PLAN_SHA256" --round "$SRF6_ROUND" --worker 0 \
  --attempt-id "$SRF6_ATTEMPT" "${resume0[@]}" &
pid0=$!
srun --exclusive --ntasks=1 --gres=gpu:pro6000:1 --cpus-per-task=4 --mem=33G \
  "$python" -u -m sr_f6.pilot_jobs worker --root "$SRF6_ROOT" \
  --phase PILOT_V3 --phase-identity-sha256 "$SRF6_PHASE_IDENTITY_SHA256" \
  --pilot-spec-sha256 "$SRF6_PILOT_SPEC_SHA256" --admission-sha256 "$SRF6_ADMISSION_SHA256" \
  --plan-sha256 "$SRF6_PLAN_SHA256" --round "$SRF6_ROUND" --worker 1 \
  --attempt-id "$SRF6_ATTEMPT" "${resume1[@]}" &
pid1=$!
# Failure stops that lane. The independent peer is allowed to finish normally.
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if [[ "$status" == 0 && "$SRF6_ROUND" == EVAL ]]; then
  "$python" -u -m sr_f6.pilot_jobs aggregate --root "$SRF6_ROOT" \
    --phase PILOT_V3 --phase-identity-sha256 "$SRF6_PHASE_IDENTITY_SHA256" \
    --pilot-spec-sha256 "$SRF6_PILOT_SPEC_SHA256" --admission-sha256 "$SRF6_ADMISSION_SHA256" \
    --plan-sha256 "$SRF6_PLAN_SHA256" || status=1
fi
exit "$status"
