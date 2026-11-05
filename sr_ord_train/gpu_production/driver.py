"""Explicit GPU-only production execution; frozen driver qualification and science retained."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import uuid
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from gpu_production.backend import NativeBackend
from adapter.backend import file_sha, object_sha, read_json, write_once
from gpu_production.binding import make_production_binding, verify_production_binding
from gpu_production.state import restore_latest_gpu, execution_provenance
from gpu_production.update import production_update
_SCORER = None


def _native_score(payload):
    sys.path.insert(0, '/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06/src')
    from sr_f6.measurement import score_response
    text, world, query, finished, finish_reason = payload
    return score_response(text, world, query, finished=finished, finish_reason=finish_reason)


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def score(row, cases):
    global _SCORER
    task = cases[row['case_id']]['task']
    if row.get('technical_validation_errors') or row.get('generation_status') != 'COMPLETE':
        raise RuntimeError('Unresolved technical generation failure: ' + row['request_id'])
    if _SCORER is None:
        _SCORER = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        atexit.register(_SCORER.shutdown)
    return _SCORER.submit(_native_score, (row['full_text'], task['world'], task['query'],
                                        row['finished'], row['finish_reason'])).result()


def worker(binding_path, parent, mode, arm=None, train_seed=None, *, execution_code_commit, acceptance_path):
    if mode != "TRAINING":
        raise PermissionError("GPU production wrapper is training only")
    binding = read_json(binding_path)
    if binding['package_root'] != str(PACKAGE):
        raise PermissionError('Wrong deployed package')
    if mode != 'DEV_SMOKE':
        smoke = read_json(PACKAGE / 'run/SMOKE_TECHNICAL_REVIEW.json')
        if smoke['status'] != 'SMOKE_TECHNICAL_PASS' or smoke['manifest_sha256'] != binding['manifest_sha256']:
            raise PermissionError('Actual smoke review required')
    if mode == 'TRAINING':
        expected = {r['request_id']:r for r in jsonl(binding['requests_manifest']['path'])
                    if r['scope'] == 'CLARIFICATION'}
        actual = []
        for p in ('b0','b1'):
            stage = PACKAGE/'run/clarification'/p
            completed = list((stage/'executions').glob('*/WORKER_COMPLETE.json'))
            if not completed:
                raise PermissionError('Complete fixed clarification before training')
            raw_path = stage/'RAW.jsonl'
            if not any(read_json(f).get('raw_sha256') == file_sha(raw_path) for f in completed):
                raise PermissionError('Clarification completion bytes changed')
            actual.extend(jsonl(raw_path))
        if len(actual) != len(expected) or len({r['request_id'] for r in actual}) != 384:
            raise PermissionError('Clarification coverage differs')
        for row in actual:
            request = expected.get(row['request_id'])
            if request is None or row['seed'] != request['seed'] or row['input_id'] != request['input_id']:
                raise PermissionError('Clarification request identity differs')
            if row.get('generation_status') != 'COMPLETE' or row.get('technical_validation_errors'):
                raise PermissionError('Unresolved clarification technical failure')
    name = f'{parent}-{train_seed}-{arm}' if mode == 'TRAINING' else parent
    stage = PACKAGE / 'run' / ('paths' if mode == 'TRAINING' else mode.lower()) / name
    stage.mkdir(parents=True, exist_ok=True)
    execution = stage / 'executions' / (os.environ.get('SLURM_JOB_ID', 'manual') + '-' + uuid.uuid4().hex)
    execution.mkdir(parents=True)
    if mode == 'TRAINING':
        binding['path_identity'] = dict(arm=arm, train_seed=train_seed, run_id='SR-ORD-TRAIN-1/' + name)
    binding = make_production_binding(binding_path, execution, path_identity=binding['path_identity'],
        execution_code_commit=execution_code_commit, acceptance_path=acceptance_path)
    write_once(execution / 'GPU_EXECUTION_BINDING.json', binding)
    write_once(execution / 'WORKER_BINDING.json', binding)
    cases = {r['case_id']: r for r in jsonl(PACKAGE / 'frozen/private_cases.jsonl')}
    started = time.monotonic()
    backend = NativeBackend(binding, parent, execution, mode)
    runtime = backend.runtime
    runtime.identity['manifest_sha256'] = binding['manifest_sha256']
    if mode == 'TRAINING':
        train_path(backend, stage, name, cases)
    else:
        requests = [r for r in backend.requests.values() if r['parent'] == parent and r['scope'] == mode]
        expected = 96 if mode == 'DEV_SMOKE' else 192
        if len(requests) != expected:
            raise ValueError('Registered generation denominator differs')
        if mode == 'DEV_SMOKE':
            backend.processor_evidence(requests[0]['request_id'], execution / 'processor')
        # Every draw is persisted before the next call. Unknown BEGIN is never retried.
        existing = {r['request_id']: r for r in jsonl(stage / 'RAW.jsonl')} if (stage / 'RAW.jsonl').exists() else {}
        replayed = {r['request_id'] for r in jsonl(stage / 'REPLAY.jsonl')} if (stage / 'REPLAY.jsonl').exists() else set()
        for request in requests:
            rid = request['request_id']
            attempt = stage / (object_sha(rid) + '.BEGIN.json')
            if rid not in existing:
                write_once(attempt, dict(request=request, policy_sha256=backend.policy_hash(), status='ATTEMPTED_UNKNOWN'))
                row = backend.generate_one(request)
                append(stage / 'RAW.jsonl', row)
                existing[rid] = row
            row = existing[rid]
            if row['policy_hash'] != backend.policy_hash() or row['seed'] != request['seed']:
                raise PermissionError('Saved rollout does not match parent/request')
            native_score = score(row, cases)
            score_file = stage / 'scores' / (object_sha(rid) + '.json')
            score_file.parent.mkdir(exist_ok=True)
            if score_file.exists():
                if read_json(score_file) != native_score:
                    raise PermissionError('Scorer replay changed')
            else:
                write_once(score_file, native_score)
            if mode == 'DEV_SMOKE' and rid not in replayed:
                import torch
                prepared = runtime.prepare_input(row['model_input'], PACKAGE, protocol='structured_v4')
                if prepared['routing']['input_tensor_hash'] != row['prompt_tensor_hash']:
                    raise ValueError('Smoke original prompt tensor changed')
                with torch.no_grad():
                    current = runtime.batch_sequence_forward([dict(row, prepared=prepared)], purpose='smoke_replay', grad=False)[0]
                sampled = torch.tensor(row['sampler_logprobs'], dtype=torch.float32, device=current.device)
                if current.shape != sampled.shape or not bool(torch.isfinite(current).all()):
                    raise ValueError('Nonfinite/misaligned smoke replay')
                delta = (current - sampled).abs()
                append(stage / 'REPLAY.jsonl', dict(request_id=rid, tokens=row['tokens'],
                    sampler_logprobs=row['sampler_logprobs'], train_logprobs=current.cpu().tolist(),
                    mean_absolute_difference=float(delta.mean()), max_absolute_difference=float(delta.max()),
                    tis_truncated_fraction=float(((current-sampled) > __import__('math').log(2)).float().mean()),
                    prompt_tensor_hash=row['prompt_tensor_hash']))
        closed = backend.close()
        write_once(execution / 'WORKER_COMPLETE.json', dict(status='WORKER_COMPLETE_REVIEW_PENDING', scope=mode,
            parent=parent, generated_responses=len(existing), optimizer_steps=0, parameter_updates=0,
            policy_before_sha256=backend.loaded_identity['parent_tensor_sha256'],
            policy_after_sha256=closed['policy_sha256'], elapsed_seconds=time.monotonic()-started,
            raw_sha256=file_sha(stage/'RAW.jsonl'), execution=str(execution)))


def train_path(backend, stage, name, cases):
    from adapter.transactions import PathTransactions
    from adapter.training import update, canonical_hash
    plan_path = PACKAGE / 'frozen/receipt_plans/paths' / (name + '.json')
    index = read_json(PACKAGE / 'frozen/receipt_plans/INDEX.json')
    plan_sha = index['files']['paths/' + name + '.json']
    if file_sha(plan_path) != plan_sha:
        raise ValueError('Frozen path plan changed')
    plan = read_json(plan_path)
    r = backend.runtime
    original_binding = verify_production_binding(backend.binding, backend.run_dir)
    identity = dict(experiment_id='SR-ORD-TRAIN-1', parent=backend.parent,
        arm=r.identity['arm'], train_seed=r.identity['train_seed'], run_id=r.identity['run_id'],
        manifest_sha256=backend.binding['manifest_sha256'], plan_sha256=plan_sha,
        source_sha256=backend.binding['source_sha256'],
        implementation_sha256=object_sha(original_binding['implementation_files']),
        code_commit=original_binding['code_commit'])
    with PathTransactions(stage, identity) as tx:
        marker = restore_latest_gpu(tx, r, backend.optimizer, backend.scheduler, backend.reference,
                                    backend.binding, backend.run_dir)
        if marker is None:
            marker = tx.commit(r, backend.optimizer, backend.scheduler, backend.reference, step=0,
                               sampler_state=dict(committed_step=0, next_step=1, generated_responses=0,
                                   gpu_execution_provenance=execution_provenance(backend.binding, backend.run_dir, 0)))
        for step_plan in plan[marker['step']:]:
            step = step_plan['step']
            r.identity['plan_step_sha256'] = canonical_hash(step_plan)
            if name == 'b0-11031-R' and step == 5:
                verify_original_recovery_batch(tx, backend, marker, step_plan)
            records = []
            for group in step_plan['groups']:
                for rid in group['request_ids']:
                    request = backend.requests[rid]
                    attempt = tx.begin_attempt(request, backend.policy_hash(), step)
                    if attempt['status'] == 'COMPLETE':
                        row = attempt['raw']
                    else:
                        row = backend.generate_one(request)
                        tx.complete_attempt(attempt['attempt_id'], row)
                    records.append(dict(row, score=score(row, cases)))
            receipt = production_update(r, backend.optimizer, backend.scheduler, records, backend.reference,
                                        step_plan, backend.run_dir, backend.binding)
            provenance = execution_provenance(backend.binding, backend.run_dir, step)
            receipt["gpu_execution_provenance"] = provenance
            marker = tx.commit(r, backend.optimizer, backend.scheduler, backend.reference, step=step,
                sampler_state=dict(committed_step=step, next_step=step+1, generated_responses=step*128,
                                   gpu_execution_provenance=provenance), update=receipt)
            r.identity.update(step=step, policy_hash=marker['policy_sha256'])
            print(json.dumps({'path': name, 'committed_step': step, 'responses': step*128}), flush=True)
    closed = backend.close()
    write_once(backend.run_dir / 'PATH_COMPLETE.json', dict(status='PATH_COMPLETE_REVIEW_PENDING', path=name,
        optimizer_updates=24, generated_responses=3072, final_identity=closed))


def verify_original_recovery_batch(tx, backend, marker, step_plan):
    """Deny missing step5 originals before any new BEGIN or generation call."""
    from adapter.transactions import require
    acceptance_ref = backend.binding['gpu_production']['acceptance']
    require(file_sha(acceptance_ref['path']) == acceptance_ref['sha256'], 'Production acceptance changed')
    acceptance = read_json(acceptance_ref['path'])
    ref = acceptance['diagnostic_summary']
    require(file_sha(ref['path']) == ref['sha256'], 'Accepted diagnostic changed')
    summary = read_json(ref['path'])
    audit = summary['native_audit']
    require(marker['step'] == 4 and marker['policy_sha256'] == audit['policy_before_sha256']
            and marker['reference_sha256'] == audit['reference_sha256']
            and audit['step'] == 5 and audit['plan_step_sha256'] == backend.runtime.identity['plan_step_sha256'],
            'Recovery source checkpoint/plan differs from diagnostic')
    wanted = [rid for g in step_plan['groups'] for rid in g['request_ids']]
    prior = {a['begin']['request']['request_id']: a for a in tx._attempts.values()
             if a['begin']['step'] == 5}
    require(len(wanted) == 128 and set(prior) == set(wanted), 'All original128 required before recovery')
    vectors = {v['row_id']: v for v in audit['probability_vectors']}
    for rid in wanted:
        attempt = prior[rid]
        require('complete' in attempt and attempt['begin']['request'] == backend.requests[rid]
                and attempt['begin']['policy_sha256'] == marker['policy_sha256'], 'Incomplete/different recovery attempt')
        raw = tx._verify_raw(attempt)
        vector = vectors.get(raw['row_id'])
        require(vector is not None and raw['tokens'] == vector['tokens']
                and raw['sampler_logprobs'] == vector['sampler_logprobs']
                and raw['token_path_sha256'] == vector['token_path_sha256'], 'Original recovery token path differs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--parent', choices=['b0', 'b1'], required=True)
    parser.add_argument('--mode', choices=['DEV_SMOKE', 'CLARIFICATION', 'TRAINING'], required=True)
    parser.add_argument('--arm', choices=['R','G','D'])
    parser.add_argument('--train-seed', type=int, choices=[11031,11032])
    parser.add_argument('--execution-code-commit', required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    a = parser.parse_args()
    if a.mode == 'TRAINING' and (a.arm is None or a.train_seed is None):
        parser.error('Training path requires registered arm and seed')
    worker(a.binding, a.parent, a.mode, a.arm, a.train_seed,
           execution_code_commit=a.execution_code_commit, acceptance_path=a.acceptance)


if __name__ == '__main__':
    main()
