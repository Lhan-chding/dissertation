#!/usr/bin/env python3
"""Minimal technical preflight. Does not inspect A/S/L effects or approve training."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
cfg=json.loads((R/'config/plan.json').read_text());r=json.loads(a.receipt.read_text())
required=['historical_identity_bound','historical_world_exclusion_pass','dual_oracle_pass',
          'prompt_diff_pass','processor_visual_inspected','prefill_boundary_verified',
          'prefill_stop_scanner_verified','scorer_fixtures_pass','no_optimizer_no_backprop',
          'smoke_complete_and_technical_pass','main_inputs_frozen']
failed=[k for k in required if r.get(k) is not True]
if r.get('experiment_id')!=cfg['experiment_id']:failed.append('experiment_id')
if r.get('model_updates')!=0:failed.append('model_updates')
if r.get('train_authorized') is not False:failed.append('train_authorized')
if r.get('main_requests')!=4320:failed.append('main_requests')
for name in ('source_commit','input_manifest_sha256','runtime_identity_sha256','smoke_receipt_sha256'):
    if not isinstance(r.get(name),str) or r[name].startswith(('NOT_','TO_')):failed.append(name)
if failed:
    print(json.dumps({'main_inference':'BLOCKED','training':'FORBIDDEN','missing':failed},ensure_ascii=False,indent=2));sys.exit(2)
print(json.dumps({'main_inference':'TECHNICAL_PREFLIGHT_PASS','training':'FORBIDDEN',
  'note':'Boolean receipts alone are not proof: inspect the actual linked runtime/input/smoke artifacts.'},ensure_ascii=False,indent=2))
