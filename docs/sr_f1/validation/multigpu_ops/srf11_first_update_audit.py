"""Read-only CPU audit of the first four-GPU ENGINE checkpoint; no model calls."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from mm_core.training import state_hash
from sr_f1.contract import digest
from sr_f1.engine import _read_state
from sr_f1.runtime import file_hash, read_json
from sr_f1.training import token_path_record

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_f11_20261010')
TRACK = ROOT / 'engineering/engine/natural/continuous'
assert os.environ.get('SLURM_JOB_ID'), 'Audit must run in Slurm'
assert not torch.cuda.is_initialized(), 'CPU audit must not initialize CUDA'
assert file_hash(ROOT / 'EXECUTION_FREEZE.json') == 'c7cf051d3aa6fb45889324e64b08d31a008137668aff2382283e4e19f99ff2b8'
assert file_hash(ROOT / 'ENGINE_MULTIGPU_REPAIR.json') == 'fdf036fc9836aa9de7b48d8978b03bb39926ac43cef113ed2b254329b61a21fd'
source = read_json(ROOT / 'code/SOURCE_DEPLOYMENT.json')
assert source['source_commit'] == '22735a2def1ee1ed79d69f603e580c67e301c5dd'
for name, expected in source['source_file_hashes'].items():
    assert file_hash(ROOT / 'code' / name) == expected, name
before, after = _read_state(TRACK, 0), _read_state(TRACK, 1)
metrics = read_json(TRACK / 'steps/01.json')
assert before['committed_logical_step'] == 0 and after['committed_logical_step'] == 1
assert after['cursor'] == dict(next_logical_step=2, next_slot=0, next_sample_index=0)
for key in ('plan_id', 'run_identity', 'reference', 'reference_hash', 'input_stream_hash', 'sampling_hash'):
    assert state_hash(before[key]) == state_hash(after[key]), key
assert len(after['rng']['cuda']) == 4
assert digest(metrics) == after['diagnostics_hash']
assert state_hash(before['parameters']) == metrics['parameter_hash_before']
assert state_hash(after['parameters']) == metrics['parameter_hash_after']
assert state_hash(after['optimizer']) == metrics['optimizer_state_hash']
assert metrics['parameter_changed'] and metrics['policy_gradient_norm'] > 0
assert metrics['effective_sequences'] == 128 and metrics['logical_step'] == 1
assert all(group['lr'] == 1e-4 for group in after['optimizer']['param_groups'])
assert after['scheduler']['_last_lr'] == [1e-4]
raw_files = sorted((TRACK / 'rollouts').glob('01-*.json'))
assert len(raw_files) == 128
records = [read_json(p) for p in raw_files]
assert digest([token_path_record(r) for r in records]) == metrics['token_path_hash'] == after['token_path_hash']
moments = after['optimizer']['state']
assert moments and all(float(v['step']) == 1 for v in moments.values())
for value in moments.values():
    assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
probability = metrics['sampler_teacher_forcing']
assert probability['token_count'] == metrics['completion_tokens']
assert not torch.cuda.is_initialized()
print(json.dumps(dict(
    status='PASS_FIRST_UPDATE_FULL_STATE', time=datetime.now(timezone.utc).isoformat(),
    job_id=os.environ['SLURM_JOB_ID'], source_commit=source['source_commit'],
    checkpoint=read_json(TRACK / 'checkpoints/commit-01.json'),
    state_fields_verified=len(after), cuda_rng_states=len(after['rng']['cuda']),
    raw_rollouts_verified=len(records), token_path_hash=after['token_path_hash'],
    probability_max_abs_difference=probability['maximum_absolute_logprob_difference_nats'],
    probability_mean_abs_difference=probability['mean_absolute_logprob_difference_nats'],
    probability_tokens=probability['token_count'], policy_gradient_norm=metrics['policy_gradient_norm'],
    natural_nonconstant_groups=metrics['natural_nonconstant_groups'],
    parameter_max_absolute_delta=metrics['parameter_max_absolute_delta'],
    adam_states=len(moments),
    adam_nonzero_moment_pairs=sum(bool(v['exp_avg'].count_nonzero()) and bool(v['exp_avg_sq'].count_nonzero()) for v in moments.values()),
    metrics_adam_nonzero_moments=metrics['adam_nonzero_moments'],
    format_failures=metrics['format_failures'], learning_rate=1e-4,
    engine_accepted=False, science_started=False, model_calls=0, cuda_initialized=False,
), sort_keys=True))
