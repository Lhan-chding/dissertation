#!/usr/bin/env bash
# Frozen CPU initializer payload. No sbatch, ENGINE, training, or model generation.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
PYTHON=/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python3.12
NEWBASE=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016
ABL="$NEWBASE/ablation"
OLD=/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_20261016
PARENT=/projects/_hdd/varunhdd/louis-ssvc/sr_f2_20261013_r3
OPS_PREP="$NEWBASE/deployment/selected_preparation/selected_run_preparation"
DATA_PREP="$NEWBASE/deployment/selected_preparation/selected_data_preparation"
HISTORY="$NEWBASE/retained_history_v1/SR_F3_CE_PROTOCOL_STOP_EVIDENCE_FULL_UNDER400MB_20261016.tar.xz"
TYPED=4146e156ffa9a476b360daeb67503703faca138f
COMMON=(--ablation-root "$ABL" --old-root "$OLD" --prep-dir "$OPS_PREP" --data-prep-dir "$DATA_PREP")
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode static "${COMMON[@]}" --history-archive "$HISTORY"
# static authenticated the complete decision and created exactly its selected sibling.
NEW=$("$PYTHON" -B - "$ABL" <<'PY'
import json, sys
from pathlib import Path
ablation = Path(sys.argv[1]).resolve(strict=True)
revision = json.loads((ablation/'PROTOCOL_SELECTION.json').read_text())['selected_revision']
if revision not in {'v2', 'c2'}:
    raise PermissionError('No selected revision')
print(ablation.parent/('selected_'+revision))
PY
)
"$PYTHON" -B "$DATA_PREP/prepare_selected_data.py" --old-root "$OLD" --new-root "$NEW" --ablation-root "$ABL" --prep-dir "$DATA_PREP"
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/audit_parents.py" --parent-root "$PARENT" --plan-dir "$NEW/plan" --output-root "$NEW"
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/export_prior.py" --parent-root "$PARENT" --output-root "$NEW"
"$PYTHON" -B "$DATA_PREP/verify_parent_prior.py" --old-root "$OLD" --new-root "$NEW" --ablation-root "$ABL" --prep-dir "$DATA_PREP"
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode external-v0 "${COMMON[@]}"
"$PYTHON" -B "$NEW/code/scripts/sr_f3_ce/audit_native.py" --root "$NEW"
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode seal "${COMMON[@]}"
TYPE_ARGS=(--root "$NEW" --review-a "$NEW/protocol_revision/type_review_v1/REVIEW_A.json" --review-b "$NEW/protocol_revision/type_review_v1/REVIEW_B.json" --review-a-evidence-dir "$NEW/protocol_revision/type_review_v1/reviewer_evidence/A" --review-b-evidence-dir "$NEW/protocol_revision/type_review_v1/reviewer_evidence/B" --auxiliary-code-commit "$TYPED")
"$PYTHON" -B "$NEW/operations/$TYPED/finalize_external_types.py" "${TYPE_ARGS[@]}"
"$PYTHON" -B "$NEW/operations/$TYPED/finalize_external_types.py" "${TYPE_ARGS[@]}" --publish
"$PYTHON" -B "$OPS_PREP/prepare_selected_run.py" --mode typed-receipt "${COMMON[@]}"
# Main operator must now verify this CPU job's real COMPLETED/0:0 in the outside
# selected_init ledger before submitting the new P0. This payload stops here.
