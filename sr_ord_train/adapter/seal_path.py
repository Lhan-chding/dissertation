"""Seal one complete training path after independent raw/token/native-score audit.

No generation, forward, optimizer, historical CLI, or formal-evaluation-manifest
reads occur here. Exported evidence is subordinate to immutable original commits.
PATH_RECEIPT.json is published last, only after runtime_contract.check_path passes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from adapter.backend import file_sha, object_sha, text_sha
from adapter.registration import Registration
from adapter.seal_smoke import Evidence, independent_scores, unique
from adapter.training import canonical_hash, native_j_advantages, trajectory_hash, validate_records
from adapter.transactions import PathTransactions, read_json, write_once
from runtime_contract import PARENT_FILES, PINS, SNAPSHOT, check_numerics, check_path


def need(ok, message):
    if not ok:
        raise PermissionError(message)


def path_parts(name):
    match = re.fullmatch(r'(b[01])-(11031|11032)-([RGD])', name)
    need(match is not None, 'Unregistered training path')
    return match[1], int(match[2]), match[3]


def put_bytes(path, data):
    path = Path(path)
    need(not path.is_symlink() and path.parent.resolve() == path.parent, 'Unsafe receipt evidence path')
    if path.exists():
        need(path.is_file() and path.read_bytes() == data, 'Immutable receipt evidence differs: ' + str(path))
    else:
        import os
        with path.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())


def put_json(path, value):
    put_bytes(path, (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode())


def publish_receipt(path, value):
    """Atomic no-overwrite publication; an interrupted write is never a seal."""
    import os
    import uuid
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        need(not path.is_symlink() and path.read_bytes() == data, 'Existing path receipt differs')
        return
    staging = path.parent / ('.PATH_RECEIPT-' + uuid.uuid4().hex + '.tmp')
    put_bytes(staging, data)
    os.link(staging, path)
    staging.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reference(stage, path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.resolve() == path
         and path.is_relative_to(stage), 'Nonlocal receipt evidence')
    return {'path': str(path.relative_to(stage)), 'sha256': file_sha(path)}


def verify_token_vectors(records, update):
    """Recompute probability gates from every original token; never judge efficacy."""
    traces = update['probability_vectors']
    need(len(records) == len(traces) == 128, 'Incomplete raw/probability update')
    absolute, truncated, sequence_kl = [], 0, []
    for row, trace in zip(records, traces, strict=True):
        need(trace['row_id'] == row['row_id'] and trace['qid'] == row['qid']
             and trace['tokens'] == row['tokens'] and trace['token_path_sha256'] == trajectory_hash(row),
             'Teacher trace differs from original token identity')
        sampled = row['sampler_logprobs']
        need(trace['sampler_logprobs'] == sampled, 'Sampler trace changed')
        current, ref = trace['train_logprobs'], trace['reference_logprobs']
        need(len(sampled) == len(current) == len(ref) == len(row['tokens']) and len(current) > 0,
             'Original probability vectors do not align')
        need(all(type(v) in (float, int) and math.isfinite(v) for v in sampled + current + ref),
             'Nonfinite raw probability')
        delta = [c - s for c, s in zip(current, sampled)]
        absolute.extend(abs(v) for v in delta)
        truncated += sum(v > math.log(2) for v in delta)
        kl = [math.expm1(r - c) - (r - c) for r, c in zip(ref, current)]
        need(all(math.isfinite(v) and v >= -1e-14 for v in kl), 'Nonfinite or negative KL')
        sequence_kl.append(sum(kl) / len(kl))
    actual = dict(sampler_train_mean_absolute_difference=sum(absolute) / len(absolute),
                  sampler_train_max_absolute_difference=max(absolute),
                  tis_truncated_fraction=truncated / len(absolute),
                  kl_unweighted=sum(sequence_kl) / 128)
    for key, value in actual.items():
        need(math.isclose(update['numerics'][key], value, rel_tol=5e-5, abs_tol=3e-7),
             'Recorded numerical statistic differs from raw vectors: ' + key)
    check_numerics(update['numerics'])
    return dict(actual, tokens=len(absolute), responses=128)


def verify_groups(plan, update, scores):
    need(len(plan['groups']) == len(update['groups']) == 16, 'Missing planned G8')
    for expected, actual in zip(plan['groups'], update['groups'], strict=True):
        need(all(actual[k] == v for k, v in expected.items()), 'Update group differs from frozen plan')
        js = [scores[rid]['J'] for rid in expected['request_ids']]
        need(all(type(j) is int and j in (0, 1) for j in js), 'Native J must be binary')
        need(actual['J'] == js and actual['native'] == native_j_advantages(js)
             and actual['aux'] == [0.] * 8, 'Update credit differs from independent native J')


def verify_checkpoint_state(stage, marker):
    """Deserialize CPU checkpoint only; no model construction or forward."""
    import torch
    from mm_core.training import state_hash
    from sr_f3_ce.training import finite_tree, validate_optimizer_state
    from adapter.transactions import _rng_unpack
    path = stage / 'checkpoints' / f"step-{marker['step']:03d}" / 'state.pt'
    need(file_sha(path) == marker['state_sha256'], 'Checkpoint bytes changed')
    state = torch.load(path, map_location='cpu', weights_only=True)
    need(state['identity'] == marker['identity'] and state['step'] == marker['step']
         and state['sampler_state'] == marker['sampler_state'], 'Serialized checkpoint metadata differs')
    need(state_hash(state['policy']) == marker['policy_sha256']
         and state['reference_sha256'] == marker['reference_sha256']
         and state_hash(state['optimizer']) == marker['optimizer_sha256']
         and state_hash(state['scheduler']) == marker['scheduler_sha256']
         and state_hash(_rng_unpack(state['rng'])) == marker['rng_sha256'], 'Serialized checkpoint content differs')
    need(state['policy'] and all(v.dtype == torch.float32 for v in state['policy'].values())
         and finite_tree(state['policy']) and finite_tree(state['optimizer']), 'Nonfinite/non-FP32 checkpoint')
    validate_optimizer_state(state['optimizer'], state['scheduler'])
    need(state['scheduler']['last_epoch'] == marker['step'], 'Scheduler cursor differs')
    if marker['step']:
        need(state['optimizer']['state'] and
             all(float(v['step']) == marker['step'] for v in state['optimizer']['state'].values()), 'Adam cursor differs')
    else:
        need(not state['optimizer']['state'] and marker['policy_sha256'] == marker['reference_sha256'],
             'Step zero not a fresh corresponding parent')
    return dict(step=marker['step'], state_sha256=marker['state_sha256'],
                policy_sha256=marker['policy_sha256'], optimizer_sha256=marker['optimizer_sha256'],
                scheduler_sha256=marker['scheduler_sha256'], rng_sha256=marker['rng_sha256'])


def export_path(package, name):
    package = Path(package)
    need(package.is_absolute() and package.resolve(strict=True) == package, 'Canonical package root required')
    parent, seed, arm = path_parts(name)
    stage = package / 'run/paths' / name
    need(stage.is_dir() and stage.resolve() == stage, 'Training path absent/unsafe')
    evidence = Evidence()
    binding_path = package / 'execution/RUN_BINDING.json'
    binding = evidence.json(binding_path, package)
    registry = Registration(package)  # Does not open the formal evaluation manifest.
    need(binding['experiment_id'] == 'SR-ORD-TRAIN-1' and binding['package_root'] == str(package)
         and binding['manifest_sha256'] == registry.freeze_sha256, 'Run scientific binding differs')
    need(binding['source_sha256'] == object_sha(binding['source_files']), 'Native source inventory differs')
    for item in binding['source_files'] + binding['implementation_files']:
        need(file_sha(evidence.file(Path(item['path']))) == item['sha256'], 'Bound source bytes differ')
    native = Path(binding['native_code_root']) / 'src'
    need({str(p) for p in native.rglob('*.py') if p.is_file()} == {i['path'] for i in binding['source_files']},
         'Native dependency inventory incomplete')
    for relative, hx in PINS.items():
        need(file_sha(native / relative) == hx, 'Historical kernel fingerprint differs')
    need(binding['model']['snapshot'] == SNAPSHOT, 'Model snapshot differs')
    adapter = Path(binding['model']['parents'][parent])
    for filename, hx in zip(('adapter_config.json', 'adapter_model.safetensors'), PARENT_FILES[parent]):
        need(file_sha(adapter / filename) == hx, 'Actual parent artifact differs')
    need(file_sha(binding['historical_source_binding']) == binding['historical_source_binding_sha256'],
         'Historical source receipt changed')
    need(file_sha(binding['native_binding']['path']) == binding['native_binding']['sha256'], 'Native binding changed')
    plan_path = package / 'frozen/receipt_plans/paths' / (name + '.json')
    plan = evidence.json(plan_path, package)
    plan_sha = registry.plan_index['files']['paths/' + name + '.json']
    need(file_sha(plan_path) == plan_sha, 'Frozen path plan changed')
    expected = unique([r for r in registry.requests() if r.get('scope') == 'TRAIN'
                      and r['parent'] == parent and r['train_seed'] == seed and r['arm'] == arm])
    need(len(expected) == 3072, 'Registered path response count differs')
    manifest = binding['requests_manifest']
    need(file_sha(manifest['path']) == manifest['sha256'], 'Registered requests changed')
    deployed = unique([r for r in evidence.rows(Path(manifest['path']), package) if r['request_id'] in expected])
    need(deployed == expected, 'Deployed request registration differs from frozen plan')
    cases = unique(evidence.rows(package / 'frozen/private_cases.jsonl'), 'case_id')
    identity = dict(experiment_id='SR-ORD-TRAIN-1', parent=parent, arm=arm, train_seed=seed,
                    run_id='SR-ORD-TRAIN-1/' + name, manifest_sha256=registry.freeze_sha256,
                    plan_sha256=plan_sha, source_sha256=binding['source_sha256'],
                    implementation_sha256=object_sha(binding['implementation_files']), code_commit=binding['code_commit'])
    need(evidence.json(stage / 'IDENTITY.json', stage) == identity, 'Training path identity differs')
    with PathTransactions(stage, identity) as tx:
        evidence.file(stage / 'ATTEMPTS.jsonl', stage)
        markers = tx.published()
        need([m['step'] for m in markers] == list(range(25)), 'Path must publish every checkpoint 0..24')
        need(len(tx._attempts) == 3072 and all('complete' in a for a in tx._attempts.values()),
             'Incomplete/unknown/extra training attempts')
        parent_tensor = markers[0]['policy_sha256']
        need(parent_tensor == markers[0]['reference_sha256'], 'Step zero/reference differ')
        sys.path.insert(0, str(native))
        states = [verify_checkpoint_state(stage, m) for m in markers]
        loaded, preupdates, native_ids = {}, {}, []
        completions = []
        for execution in sorted((stage / 'executions').iterdir()):
            need(execution.is_dir() and execution.resolve() == execution, 'Unsafe execution directory')
            if not (execution / 'LOADED_IDENTITY.json').exists():
                need(not list(execution.glob('generation_attempts/*.BEGIN.json')), 'Unbound model attempt')
                continue
            actual = evidence.json(execution / 'LOADED_IDENTITY.json', stage)
            worker = evidence.json(execution / 'WORKER_BINDING.json', stage)
            wanted = dict(binding, path_identity=dict(arm=arm, train_seed=seed, run_id=identity['run_id']))
            need(worker == wanted and actual['binding_sha256'] == object_sha(worker), 'Worker binding changed')
            need(actual['experiment_id'] == 'SR-ORD-TRAIN-1' and actual['execution_mode'] == 'TRAINING'
                 and actual['run_id'] == identity['run_id'] and actual['parent_id'] == parent
                 and actual['arm'] == arm and actual['train_seed'] == seed and actual['step'] == 0
                 and actual['optimizer_created'] is True, 'Loaded path/optimizer identity differs')
            need(actual['source_sha256'] == identity['source_sha256']
                 and actual['manifest_sha256'] == registry.freeze_sha256
                 and actual['parent_tensor_sha256'] == actual['reference_hash'] == parent_tensor
                 and actual['native_runtime_sha256'] == PINS['sr_f6/runtime.py']
                 and actual['backend_sha256'] == file_sha(package / 'adapter/backend.py'), 'Loaded source/parent differs')
            need(actual['snapshot'] == SNAPSHOT and actual['dtype'] == actual['input_float_dtype'] == actual['probability_dtype'] == 'float32'
                 and actual['forward_parameter_dtypes'] == ['float32']
                 and set(actual['forward_float_buffer_dtypes']) <= {'float32'}
                 and actual['attention_backend'] == 'eager', 'Loaded runtime precision differs')
            loaded[execution] = dict(loaded=actual, worker_binding=worker)
            if (execution / 'PATH_COMPLETE.json').exists():
                completed = evidence.json(execution / 'PATH_COMPLETE.json', stage)
                need(completed['path'] == name and completed['optimizer_updates'] == 24
                     and completed['generated_responses'] == 3072
                     and completed['final_identity']['policy_sha256'] == markers[-1]['policy_sha256'], 'Path completion differs')
                completions.append(completed)
            audit = execution / 'NUMERICAL_TOKEN_AUDIT.jsonl'
            if audit.exists():
                for row in evidence.rows(audit, stage):
                    if row.get('status') == 'PREUPDATE_PASS':
                        preupdates.setdefault(row['step'], []).append(row)
            for begin_path in execution.glob('generation_attempts/*.BEGIN.json'):
                begin = evidence.json(begin_path, stage)
                rid = begin['request_id']
                need(rid in expected and begin['seed'] == expected[rid]['seed'], 'Native attempt seed/request differs')
                complete_path = begin_path.with_name(begin_path.name.replace('.BEGIN.json', '.COMPLETE.json'))
                need(complete_path.is_file(), 'UNKNOWN native attempt blocks seal')
                complete = evidence.json(complete_path, stage)
                need(begin['policy_sha256'] == complete['policy_hash']
                     and begin['reference_sha256'] == parent_tensor, 'Native attempt policy differs')
                native_ids.append(rid)
        need(completions and len(native_ids) == 3072 and set(native_ids) == set(expected),
             'Missing completion or duplicate/missing native generation attempts')
        destination = stage / 'receipt_evidence'
        need(not destination.is_symlink(), 'Unsafe receipt output')
        destination.mkdir(exist_ok=True)
        put_bytes(destination / 'RUN_BINDING.json', binding_path.read_bytes())
        put_bytes(destination / 'PLAN.json', plan_path.read_bytes())
        put_json(destination / 'TRAIN_RECIPE.json', binding['recipe'])
        put_json(destination / 'LOADED_BINDINGS.json', {str(k): v for k, v in loaded.items()})
        put_json(destination / 'CHECKPOINT_STATE_AUDIT.json', states)
        all_scores, normalized, audit_stats = [], [], []
        by_request = {a['begin']['request']['request_id']: a for a in tx._attempts.values()}
        need(set(by_request) == set(expected), 'Raw attempt request coverage differs')
        for step_plan, marker in zip(plan, markers[1:], strict=True):
            step = step_plan['step']
            update_path = stage / 'checkpoints' / f'step-{step:03d}' / 'UPDATE.json'
            update = evidence.json(update_path, stage)
            need(update['status'] == 'UPDATED_UNCOMMITTED' and update['step'] == step
                 and update['policy_before_sha256'] == markers[step-1]['policy_sha256']
                 and update['policy_after_sha256'] == marker['policy_sha256']
                 and update['reference_sha256'] == parent_tensor, 'Update policy chain differs')
            need(len(preupdates.get(step, [])) == 1, 'Missing or ambiguous pre-Adam pass evidence')
            before = preupdates[step][0]
            need(before['numerics'] == update['numerics'] and before['probability_vectors'] == update['probability_vectors']
                 and before['policy_before_sha256'] == update['policy_before_sha256']
                 and before['precommit_gate'] == 'PASS_BEFORE_ADAM', 'Pre-Adam evidence differs')
            raw = []
            for group in step_plan['groups']:
                for rid in group['request_ids']:
                    attempt = by_request[rid]
                    need(attempt['begin']['request'] == expected[rid] and attempt['begin']['step'] == step
                         and attempt['begin']['policy_sha256'] == update['policy_before_sha256'], 'Original attempt differs from plan')
                    row = tx._verify_raw(attempt)
                    need(all(row.get(k) == expected[rid][k] for k in expected[rid] if k not in ('prompt', 'image_file'))
                         and row['model_input'] == {k: expected[rid][k] for k in ('prompt', 'image_file', 'image_sha256')},
                         'Actual model input/control identity differs from frozen request')
                    need(row['generation_status'] == 'COMPLETE' and row['technical_validation_errors'] == []
                         and row.get('native_exception') is None, 'Technical-invalid rollout')
                    need(row['full_text'] == row['assistant_prefill'] + row['suffix_text']
                         and row['suffix_text'] == row['generated_text'] and row['assistant_prefill'] == '{'
                         and row['finished'] is (row['finish_reason'] == 'balanced'), 'Raw text/stop/prefill differs')
                    capture_path = evidence.file(Path(row['native_capture_path']), stage)
                    need(file_sha(capture_path) == row['native_capture_sha256'], 'Native capture bytes differ')
                    capture = evidence.json(capture_path, stage)
                    execution = capture_path.parent.parent
                    need(execution in loaded, 'Capture lacks loaded identity')
                    need(all(capture.get(k) == row.get(k) for k in ('tokens','raw_tokens','sampler_logprobs','prompt_tensor_hash',
                         'model_input','seed','full_text','assistant_prefill','generated_text','finish_reason','sampling_parameters')),
                         'Raw differs from native capture')
                    need(capture['adapter_identity']['run_id'] == identity['run_id']
                         and capture['adapter_identity']['step'] == step-1, 'Native capture policy version differs')
                    complete_path = capture_path.with_name(capture_path.name.replace('.CAPTURE.json', '.COMPLETE.json'))
                    need(evidence.json(complete_path, stage) == row, 'Native complete differs from transaction raw')
                    raw.append(row)
            scores_result = independent_scores(package, raw, cases)
            scores = {r['request_id']: r['score'] for r in scores_result['scores']}
            need(len(scores) == 128 and set(scores) == {r['request_id'] for r in raw}, 'Independent scorer coverage differs')
            verify_groups(step_plan, update, scores)
            actual_identity = dict(identity, parent_id=parent, execution_mode='TRAINING', step=step-1,
                                   reference_hash=parent_tensor, plan_step_sha256=canonical_hash(step_plan), run_root=str(package), audit_root=str(stage))
            _, groups = validate_records(actual_identity, [dict(r, score=scores[r['request_id']]) for r in raw],
                                          step_plan, update['policy_before_sha256'], parent_tensor)
            need(groups == update['groups'], 'Actual update credit/identity differs from independent reconstruction')
            audit_stats.append(dict(step=step, **verify_token_vectors(raw, update)))
            all_scores.extend(scores_result['scores'])
            raw_path, trace_path = destination / f'raw-step-{step:03d}.json', destination / f'token-trace-step-{step:03d}.json'
            put_json(raw_path, raw); put_json(trace_path, update['probability_vectors'])
            normalized.append({k: update[k] for k in ('step','policy_version_sampled','policy_before_sha256',
                'policy_after_sha256','reference_sha256','optimizer_steps','responses','aux_nonzero_rows','prefix_rows',
                'precommit_gate','groups','numerics')} | {'evidence': dict(raw=reference(stage, raw_path),
                token_trace=reference(stage, trace_path), native_update=reference(stage, update_path),
                commit=reference(stage, update_path.with_name('COMMIT.json')))})
        replay_path = destination / 'NATIVE_REPLAY.json'
        put_json(replay_path, dict(status='INDEPENDENT_NATIVE_REPLAY_PASS', responses=3072,
            scorer='srf6-native-measurement-v1', native_fixtures=scores_result['native_fixtures'], source_pins={k: v for k, v in PINS.items() if 'measurement' in k},
            rows=all_scores, numerical_recomputation=audit_stats))
        put_json(destination / 'ORIGINAL_EVIDENCE_INDEX.json', evidence.refs)
        result = dict(schema='order-runtime-v1', scope='TRAIN_PATH', manifest_sha256=registry.freeze_sha256,
            snapshot=SNAPSHOT, dtype='float32', reward='native_srf6_J_only', protocol='structured_v4',
            aux_enabled=False, prefix_training=False, has_gpu_hour_stopping=False,
            parent=parent, arm=arm, seed=seed, adapter_config_sha256=PARENT_FILES[parent][0],
            adapter_file_sha256=PARENT_FILES[parent][1], parent_tensor_sha256=parent_tensor,
            reference_tensor_sha256=parent_tensor, total_optimizer_steps=24, total_generated_responses=3072,
            evaluation_policy_sha256={str(s): markers[s]['policy_sha256'] for s in (12,24)}, updates=normalized,
            seal_status='PATH_NATIVE_REPLAY_AND_RECEIPT_VERIFIED', identity=identity,
            native_replay=reference(stage, replay_path),
            evidence=dict(native_binding=reference(stage, destination / 'RUN_BINDING.json'),
                          plan=reference(stage, destination / 'PLAN.json'), recipe=reference(stage, destination / 'TRAIN_RECIPE.json'),
                          zero_checkpoint=reference(stage, stage / 'checkpoints/step-000/state.pt'),
                          checkpoint_12=reference(stage, stage / 'checkpoints/step-012/state.pt'),
                          checkpoint_24=reference(stage, stage / 'checkpoints/step-024/state.pt'),
                          loaded_bindings=reference(stage, destination / 'LOADED_BINDINGS.json'),
                          checkpoint_state_audit=reference(stage, destination / 'CHECKPOINT_STATE_AUDIT.json'),
                          original_evidence=reference(stage, destination / 'ORIGINAL_EVIDENCE_INDEX.json')))
        check_path(result, registry.freeze_sha256, stage, plan_sha)
        publish_receipt(stage / 'PATH_RECEIPT.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    result = export_path(args.package, args.path)
    print(json.dumps(dict(status=result['seal_status'], path=args.path, responses=3072, optimizer_steps=24)))


if __name__ == '__main__':
    main()
