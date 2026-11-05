"""Seal an original or explicitly bridged GPU-only training path.

This copy retains the original raw/token/native-score and transaction audit.
Only explicit new execution bindings, GPU RNG lineage and committed provenance
are added. Historical files and original scientific identity remain unchanged.

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


def verify_execution_bridge(package, stage, execution, worker, actual, original, identity, evidence):
    """Return the actual backend path after verifying the explicit execution type."""
    wanted = dict(original, path_identity=dict(arm=identity['arm'], train_seed=identity['train_seed'],
                                               run_id=identity['run_id']))
    need(actual['binding_sha256'] == object_sha(worker), 'Worker binding changed')
    if 'gpu_production' not in worker:
        need(worker == wanted, 'Worker binding changed')
        need(not (execution / 'GPU_EXECUTION_BINDING.json').exists()
             and not (execution / 'GPU_RECOVERY_LINEAGE.json').exists(), 'Legacy worker has hidden GPU bridge')
        return package / 'adapter/backend.py'
    from gpu_production.binding import verify_production_binding
    source = verify_production_binding(worker, execution)
    need(source == original and worker['path_identity'] == wanted['path_identity'],
         'GPU worker changed original scientific binding')
    need(evidence.json(execution / 'GPU_EXECUTION_BINDING.json', stage) == worker,
         'GPU execution binding differs from loaded worker')
    hardware = actual.get('hardware', {})
    need(hardware.get('scope') == 'REGISTERED_GPU_PRODUCTION_EXECUTION'
         and hardware.get('cuda_visible_device_count') == 2 and len(hardware.get('devices', [])) == 2
         and hardware.get('peer_access') == {'0_to_1': True, '1_to_0': True},
         'GPU worker lacks actual two-GPU peer identity')
    devices = hardware['devices']
    need(devices[0]['cuda_uuid'] != devices[1]['cuda_uuid']
         and devices[0]['name'] == devices[1]['name']
         and all('RTX PRO 6000' in d['name'].upper() for d in devices), 'GPU worker hardware differs')
    # A backend may have loaded and then failed before restoration. Such an
    # execution is retained, but cannot supply any committed step provenance.
    lineage = execution / 'GPU_RECOVERY_LINEAGE.json'
    if lineage.exists():
        evidence.json(lineage, stage)
    return package / 'gpu_production/backend.py'


def verify_checkpoint_bridge(stage, marker, rng):
    """Require provenance for every two-GPU checkpoint, including step zero."""
    count = len(rng['cuda'])
    provenance = marker['sampler_state'].get('gpu_execution_provenance')
    need(count in (1, 2), 'Unexpected checkpoint CUDA RNG cardinality')
    if count == 1:
        need(provenance is None, 'Legacy one-GPU checkpoint claims GPU provenance')
        if marker['step']:
            update = read_json(stage / 'checkpoints' / f"step-{marker['step']:03d}" / 'UPDATE.json')
            need('gpu_execution_provenance' not in update, 'Legacy update claims GPU provenance')
        return None
    need(provenance is not None, 'Two-GPU checkpoint lacks explicit provenance')
    from gpu_production.state import verify_checkpoint_provenance
    verify_checkpoint_provenance(stage.parents[2], marker)
    if marker['step']:
        update = read_json(stage / 'checkpoints' / f"step-{marker['step']:03d}" / 'UPDATE.json')
        need(update.get('gpu_execution_provenance') == provenance, 'UPDATE and checkpoint GPU provenance differ')
    return copy.deepcopy(provenance)


def committed_preupdate(step, update, preupdates, loaded):
    """A failed prior pre-Adam attempt stays evidence, never a competing update."""
    candidates = preupdates.get(step, [])
    provenance = update.get('gpu_execution_provenance')
    if provenance is None:
        need(len(candidates) == 1, 'Missing or ambiguous pre-Adam pass evidence')
        before, execution = candidates[0]
        need('gpu_production' not in loaded[execution]['worker_binding'], 'GPU update missing version provenance')
        return before
    execution = Path(provenance['execution_binding']['path']).parent
    need(execution in loaded and 'gpu_production' in loaded[execution]['worker_binding'],
         'Committed GPU step has no loaded production worker')
    selected = [row for row, origin in candidates if origin == execution]
    need(len(selected) == 1, 'Missing or ambiguous committed GPU pre-Adam evidence')
    return selected[0]


def verify_gpu_update_evidence(stage, identity, step, update, records, evidence):
    """Recheck committed GPU storage/Adam accounting, not merely a PASS label."""
    provenance = update.get('gpu_execution_provenance')
    if provenance is None:
        need('gpu_pre_adam_check' not in update and 'gpu_update_accounting' not in update,
             'Legacy update contains unbound GPU evidence')
        return None
    execution = Path(provenance['execution_binding']['path']).parent
    need(execution.parent == stage / 'executions', 'GPU step execution outside path')
    documents = {}
    for key, filename in [('gpu_pre_adam_check', f'GPU_PRE_ADAM_CHECK_step-{step:03d}.json'),
                          ('gpu_update_accounting', f'GPU_UPDATE_ACCOUNTING_step-{step:03d}.json')]:
        ref = update.get(key, {})
        need(set(ref) == {'path', 'sha256'} and Path(ref['path']) == execution / filename,
             'Committed GPU update evidence path differs')
        path = evidence.file(Path(ref['path']), stage)
        need(file_sha(path) == ref['sha256'], 'Committed GPU update evidence bytes differ')
        documents[key] = evidence.json(path, stage)
    check, counts = documents['gpu_pre_adam_check'], documents['gpu_update_accounting']
    need(check['status'] == 'GPU_PRE_ADAM_PASS' and check['step'] == step
         and check['gradient_forwards'] == 128 and check['gpu1_rng_unchanged'] is True
         and check['cpu_activation_offload_bytes'] == 0
         and check['original_long_gradient_equivalence'] == 'UNVERIFIED_ORIGINAL_BASELINE_OOM',
         'GPU pre-Adam storage/RNG gate failed')
    wanted = dict(pre_adam_calls=1, adam_permitted=True, optimizer_returned_calls=1,
                  scheduler_entered_calls=1, scheduler_returned_calls=1)
    need(counts['status'] == 'UPDATED_UNCOMMITTED' and counts['step'] == step
         and counts['new_generations'] == 0 and counts['adam_permitted'] is True
         and all(counts[k] == v for k, v in wanted.items()),
         'GPU actual optimizer/scheduler accounting differs')
    forwards = counts['forwards']
    need(len(forwards) == len(records) == 128
         and [r['row_id'] for r in forwards] == [r['row_id'] for r in records],
         'GPU backward original response order differs')
    total = 0
    for row in forwards:
        stats = row['stats']
        need(row['forward_returned'] is True and 0 < stats['copied_bytes'] <= 16 * 2**30
             and stats['unpacked_copies'] > 0 and stats['cpu_offload'] is False
             and stats['version_failures'] == stats['copy_failures'] == stats['unpack_failures'] == 0,
             'GPU saved-tensor backward evidence invalid')
        total += stats['copied_bytes']
    need(total == check['copied_bytes'], 'GPU total transfer accounting differs')
    if identity['run_id'] == 'SR-ORD-TRAIN-1/b0-11031-R' and step == 5:
        worker = evidence.json(execution / 'GPU_EXECUTION_BINDING.json', stage)
        package = stage.parents[2]
        accepted_ref = worker['gpu_production']['acceptance']
        accepted_path = evidence.file(Path(accepted_ref['path']), package)
        need(file_sha(accepted_path) == accepted_ref['sha256'], 'GPU accepted recovery evidence changed')
        accepted = evidence.json(accepted_path, package)
        summary_ref = accepted['diagnostic_summary']
        summary_path = evidence.file(Path(summary_ref['path']), package)
        need(file_sha(summary_path) == summary_ref['sha256'], 'GPU diagnostic oracle changed')
        summary = evidence.json(summary_path, package)
        capture = summary['capture']
        need(check['first_recovery_comparison'] == 'EXACT_DIAGNOSTIC_128_AND_CLIPPED_GRADIENTS'
             and check['compared_gradients'] == len(capture['gradient_names'])
             and check['gradient_sha256'] == capture['gradient_file_sha256']
             and check['diagnostic_summary'] == summary_ref,
             'First production step lacks exact retained diagnostic comparison')
    else:
        need(check['first_recovery_comparison'] == 'NOT_THIS_REGISTERED_B0R_STEP5',
             'GPU step claims an inapplicable diagnostic comparison')
    return documents


def verify_checkpoint_state(stage, marker, verified_rng_counts=None):
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
    result = dict(step=marker['step'], state_sha256=marker['state_sha256'],
                  policy_sha256=marker['policy_sha256'], optimizer_sha256=marker['optimizer_sha256'],
                  scheduler_sha256=marker['scheduler_sha256'], rng_sha256=marker['rng_sha256'])
    rng = _rng_unpack(state['rng'])
    bridge = verify_checkpoint_bridge(stage, marker, rng)
    if verified_rng_counts is not None:
        need(marker['step'] == len(verified_rng_counts), 'Checkpoint RNG lineage order differs')
        if bridge is not None:
            lineage = read_json(bridge['recovery_lineage']['path'])
            source_step = bridge['source_step']
            expected_source_count = 0 if source_step is None else verified_rng_counts[source_step]
            need(lineage['source_cuda_count'] == expected_source_count,
                 'GPU lineage CUDA count differs from original source checkpoint')
        elif marker['step']:
            need(verified_rng_counts[marker['step'] - 1] == 1, 'Two-GPU chain silently reverted to legacy')
        verified_rng_counts[marker['step']] = len(rng['cuda'])
    if bridge is not None:
        result['gpu_execution_provenance'] = bridge
    return result


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
        verified_rng_counts = {}
        states = [verify_checkpoint_state(stage, m, verified_rng_counts) for m in markers]
        loaded, preupdates, native_ids = {}, {}, []
        completions = []
        for execution in sorted((stage / 'executions').iterdir()):
            need(execution.is_dir() and execution.resolve() == execution, 'Unsafe execution directory')
            if not (execution / 'LOADED_IDENTITY.json').exists():
                need(not list(execution.glob('generation_attempts/*.BEGIN.json')), 'Unbound model attempt')
                continue
            actual = evidence.json(execution / 'LOADED_IDENTITY.json', stage)
            worker = evidence.json(execution / 'WORKER_BINDING.json', stage)
            backend_path = verify_execution_bridge(package, stage, execution, worker, actual, binding, identity, evidence)
            need(actual['experiment_id'] == 'SR-ORD-TRAIN-1' and actual['execution_mode'] == 'TRAINING'
                 and actual['run_id'] == identity['run_id'] and actual['parent_id'] == parent
                 and actual['arm'] == arm and actual['train_seed'] == seed and actual['step'] == 0
                 and actual['optimizer_created'] is True, 'Loaded path/optimizer identity differs')
            need(actual['source_sha256'] == identity['source_sha256']
                 and actual['manifest_sha256'] == registry.freeze_sha256
                 and actual['parent_tensor_sha256'] == actual['reference_hash'] == parent_tensor
                 and actual['native_runtime_sha256'] == PINS['sr_f6/runtime.py']
                 and actual['backend_sha256'] == file_sha(backend_path), 'Loaded source/parent differs')
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
                        preupdates.setdefault(row['step'], []).append((row, execution))
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
            before = committed_preupdate(step, update, preupdates, loaded)
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
            verify_gpu_update_evidence(stage, identity, step, update, raw, evidence)
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
        gpu_states = [s for s in states if 'gpu_execution_provenance' in s]
        if gpu_states:
            bridge_audit = destination / 'GPU_BRIDGE_AUDIT.json'
            put_json(bridge_audit, dict(status='EXPLICIT_GPU_EXECUTION_CHAIN_VERIFIED',
                original_long_gradient_equivalence='UNVERIFIED_ORIGINAL_BASELINE_OOM',
                checkpoints=gpu_states,
                executions={str(e): v for e, v in loaded.items() if 'gpu_production' in v['worker_binding']},
                preupdate_passes={str(step): [dict(execution=str(e), audit=row) for row, e in entries]
                                  for step, entries in preupdates.items()}))
            result['evidence']['gpu_execution_bridge'] = reference(stage, bridge_audit)
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
