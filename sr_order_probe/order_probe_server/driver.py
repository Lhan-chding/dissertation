"""Small request journal; scientific content is never selected from outcomes."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from order_probe.common import canonical, digest, file_sha, read_jsonl, text_sha

SPLITS = {'SMOKE': 216, 'MAIN': 4320}
PRECHECK_NAMES = ('history', 'processor_visual', 'scorer_fixtures')
PRECHECK_STATUSES = {'history': {'PASS_RECORDED_SCOPE', 'PASS_WITH_RECORDED_SCOPE'},
                    'processor_visual': {'PASS_VISUAL_REVIEW'},
                    'scorer_fixtures': {'PASS_FIXED_FIXTURES'}}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    """Publish fully written bytes without replacing an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical(value) + '\n').encode()
    publish_bytes(path, data)


def publish_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Conflicting existing artifact: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another worker owns {path.name}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


def inventory(paths, base):
    return [{'path': str(p.relative_to(base)), 'sha256': file_sha(p), 'bytes': p.stat().st_size}
            for p in sorted(paths) if p.is_file()]


def source_inventory():
    paths = [*PACKAGE.glob('backend/*.py'), *PACKAGE.glob('order_probe_server/*.py')]
    return inventory(paths, PACKAGE)


def prepare(config_path, inputs, root):
    inputs, root = Path(inputs).resolve(strict=True), Path(root).resolve()
    config = read(config_path)
    if config != read(PACKAGE / 'config/plan.json'):
        raise ValueError('Use the package scientific config without substitution')
    if config['authorization']['train'] or config['authorization']['optimizer_updates'] != 0:
        raise PermissionError('Inference only')
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / '.prepare.lock'):
        receipt_path = root / 'INPUT_VALIDATION.json'
        command = [sys.executable, str(PACKAGE / 'scripts/validate_inputs.py'),
                   '--inputs', str(inputs), '--out', str(root / '.input_validation.pending.json')]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr + result.stdout)
        validation = read(root / '.input_validation.pending.json')
        immutable(receipt_path, validation)
        (root / '.input_validation.pending.json').unlink()
        immutable(root / 'CONFIG_FROZEN.json', config)
        immutable(root / 'INPUT_MANIFEST.json', {
            'inputs_dir': str(inputs), 'files': inventory(inputs.rglob('*'), inputs)})
        science_paths = [PACKAGE / 'config/plan.json', PACKAGE / 'CODEX_START_HERE.md',
                         *PACKAGE.glob('docs/*.md'), *PACKAGE.glob('src/order_probe/*.py'),
                         *PACKAGE.glob('assets/*'), PACKAGE / 'scripts/score_answers.py',
                         PACKAGE / 'scripts/analyze.py']
        immutable(root / 'SCIENCE_MANIFEST.json', {'files': inventory(science_paths, PACKAGE)})
        immutable(root / 'SOURCE_INITIAL.json', {'files': source_inventory()})
    return {'status': 'PREPARED_NOT_GPU_VERIFIED', 'run_root': str(root),
            'main_requests': 4320, 'smoke_requests': 216, 'optimizer_updates': 0}


def load_context(root):
    root = Path(root).resolve(strict=True)
    config = read(root / 'CONFIG_FROZEN.json')
    science = read(root / 'SCIENCE_MANIFEST.json')
    for item in science['files']:
        if file_sha(PACKAGE / item['path']) != item['sha256']:
            raise RuntimeError('Frozen scientific source changed: ' + item['path'])
    manifest = read(root / 'INPUT_MANIFEST.json')
    inputs = Path(manifest['inputs_dir']).resolve(strict=True)
    actual_names = {str(p.relative_to(inputs)) for p in inputs.rglob('*') if p.is_file()}
    if actual_names != {r['path'] for r in manifest['files']}:
        raise RuntimeError('Frozen input inventory changed')
    for item in manifest['files']:
        if file_sha(inputs / item['path']) != item['sha256']:
            raise RuntimeError('Frozen input changed: ' + item['path'])
    if config != read(PACKAGE / 'config/plan.json'):
        raise RuntimeError('Frozen config changed')
    return root, inputs, config


def requests_for(inputs, split, parent=None, shard_index=0, shard_count=1):
    if split not in SPLITS or not 0 <= shard_index < shard_count:
        raise ValueError('Invalid split/shard')
    rows = read_jsonl(Path(inputs) / f'{split}_REQUESTS_PUBLIC.jsonl')
    if len(rows) != SPLITS[split] or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected request denominator or duplicate IDs')
    if parent is not None:
        rows = [r for r in rows if r['parent'] == parent]
    return rows[shard_index::shard_count]


def result_path(root, request):
    return Path(root) / 'RECORDS' / request['split'] / request['parent'] / (text_sha(request['request_id']) + '.json')


def identity_errors(identity, config, parent):
    errors = []
    expected = {'parent': parent, 'snapshot': config['model']['snapshot'],
                'model_path': config['model']['model_path'],
                'adapter_path': config['model']['parents'][parent],
                'parameter_dtypes': ['float32'], 'trainable_parameters': [],
                'optimizer': None, 'optimizer_updates': 0}
    for key, value in expected.items():
        if key not in identity or identity[key] != value:
            errors.append('identity:' + key)
    if 'buffer_dtypes' not in identity or any(d != 'float32' for d in identity['buffer_dtypes']):
        errors.append('identity:buffer_dtypes')
    if not identity.get('parent_hashes_before') or identity.get('parent_hashes_before') != identity.get('parent_hashes_after'):
        errors.append('identity:parent_hashes')
    if not identity.get('policy_tensor_hash'):
        errors.append('identity:policy_tensor_hash')
    return errors


def native_object_end(text):
    """Mirror sr_f1.json_protocol.balanced_cut, including its brace-only semantics.

    This is a stopping check, not JSON parsing. A malformed array remains a
    model output for the frozen scorer, never a reason to resample or stop.
    """
    if not isinstance(text, str) or not text.startswith('{'):
        return None
    depth, quoted, escaped = 0, False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def closed_identity_errors(identity):
    return ([] if identity.get('end_model_unchanged') is True
            and identity.get('policy_tensor_hash_after') == identity.get('policy_tensor_hash')
            else ['end_model_unchanged'])


def reconcile_reused(root, reused, worker, identity):
    """Document a new same-parent verification, never invent the old terminal state."""
    if closed_identity_errors(identity):
        raise RuntimeError('Recovery worker did not verify final model identity')
    for old_worker in sorted({row['worker_id'] for row in reused}):
        directory = root / 'WORKERS' / old_worker
        if (directory / 'END.json').exists():
            continue
        if (directory / 'CLOSE_FAILED.json').exists():
            raise RuntimeError('A failed model closure needs explicit review: ' + old_worker)
        start = read(directory / 'START.json')
        if any(start['identity'].get(key) != identity.get(key)
               for key in ('parent', 'parent_hashes_before', 'policy_tensor_hash', 'snapshot')):
            raise RuntimeError('Interrupted worker and recovery parent identities differ')
        rows = [row for row in reused if row['worker_id'] == old_worker]
        proof = {'status': 'INTERRUPTED_RECORDS_REVERIFIED_WITH_SAME_PARENT',
                 'original_worker_id': old_worker, 'recovery_worker_id': worker,
                 'original_terminal_model_state': 'UNKNOWN_PROCESS_INTERRUPTED',
                 'original_start_sha256': file_sha(directory / 'START.json'),
                 'recovery_end_sha256': file_sha(root / 'WORKERS' / worker / 'END.json'),
                 'records': [{'request_id': row['request_id'],
                              'sha256': file_sha(result_path(root, row))} for row in rows]}
        immutable(root / 'RECOVERY' / (old_worker + '--' + worker + '.json'), proof)


def check_worker_closure(root, worker, rows, config):
    directory = root / 'WORKERS' / worker
    start = read(directory / 'START.json')
    errors = identity_errors(start['identity'], config, start['parent'])
    if errors:
        raise RuntimeError('Worker start identity invalid: ' + ','.join(errors))
    if (directory / 'END.json').exists():
        end = read(directory / 'END.json')
        errors = identity_errors(end['identity'], config, end['parent']) + closed_identity_errors(end['identity'])
        if end.get('status') != 'CLOSED_READONLY' or errors:
            raise RuntimeError('Worker closure invalid: ' + worker + ':' + ','.join(errors))
        return
    if (directory / 'CLOSE_FAILED.json').exists():
        raise RuntimeError('Failed closure cannot be reconciled automatically: ' + worker)
    covered = set()
    for path in (root / 'RECOVERY').glob(worker + '--*.json'):
        proof = read(path)
        end_path = root / 'WORKERS' / proof['recovery_worker_id'] / 'END.json'
        if proof.get('original_start_sha256') != file_sha(directory / 'START.json') or proof.get('recovery_end_sha256') != file_sha(end_path):
            raise RuntimeError('Recovery evidence changed')
        end = read(end_path)
        errors = identity_errors(end['identity'], config, start['parent']) + closed_identity_errors(end['identity'])
        if errors or any(start['identity'].get(key) != end['identity'].get(key)
                         for key in ('parent', 'parent_hashes_before', 'policy_tensor_hash', 'snapshot')):
            raise RuntimeError('Recovery parent identity differs')
        for item in proof['records']:
            row = next((r for r in rows if r['request_id'] == item['request_id']), None)
            if row is not None and file_sha(result_path(root, row)) == item['sha256']:
                covered.add(row['request_id'])
    if covered != {row['request_id'] for row in rows}:
        raise RuntimeError('Interrupted worker needs same-parent resume reconciliation: ' + worker)


def validate_record(row, request, config):
    errors = []
    for key in ('request_id', 'seed', 'parent', 'assistant_prefill', 'prompt_sha256', 'prefill_sha256', 'image_sha256'):
        if row.get(key) != request[key]:
            errors.append(key)
    if row.get('full_text') != request['assistant_prefill'] + row.get('suffix_text', ''):
        errors.append('literal_full_text')
    if row.get('sampling') != config['generation']:
        errors.append('sampling')
    prompt = row.get('prompt_input_ids')
    suffix = row.get('suffix_token_ids')
    for label, tokens in [('prompt_input_ids', prompt), ('suffix_token_ids', suffix)]:
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            errors.append(label)
    if isinstance(suffix, list) and (len(suffix) > 1536 or row.get('generated_tokens') != len(suffix)):
        errors.append('generated_tokens')
    boundary = row.get('prefill_boundary', {})
    if boundary.get('same_assistant_message') is not True:
        errors.append('same_assistant_message')
    prefix_ids = boundary.get('prefill_token_ids')
    if not isinstance(prefix_ids, list) or not prefix_ids:
        errors.append('prefill_token_ids')
    elif isinstance(prompt, list) and prompt[-len(prefix_ids):] != prefix_ids:
        errors.append('prefill_token_tail')
    if row.get('finish_reason') not in ('balanced', 'eos', 'length'):
        errors.append('finish_reason')
    if row.get('finish_reason') == 'balanced' and native_object_end(row.get('full_text', '')) is None:
        errors.append('premature_balanced_stop')
    if row.get('finish_reason') == 'length' and isinstance(suffix, list) and len(suffix) != 1536:
        errors.append('premature_length_stop')
    if row.get('technical_validation_errors') != []:
        errors.append('backend_technical_validation')
    if row.get('optimizer_updates') != 0:
        errors.append('optimizer_updates')
    errors += identity_errors(row.get('model_identity', {}), config, request['parent'])
    return errors


def execute_request(root, request, config, backend, worker):
    """Successful model calls are persisted before any technical quality checks."""
    path = result_path(root, request)
    journal = Path(root) / 'ATTEMPTS_BY_ID' / (text_sha(request['request_id']) + '.jsonl')
    with locked(path.with_suffix('.lock')):
        if path.exists():
            row = read(path)
            errors = validate_record(row, request, config)
            if errors:
                raise RuntimeError('Saved result requires technical review, never resampling: ' + ','.join(errors))
            return 'REUSED'
        attempts = read_jsonl(journal) if journal.exists() else []
        started = [a for a in attempts if a['event'] == 'START']
        if any(a['seed'] != request['seed'] for a in started):
            raise RuntimeError('Attempt seed changed')
        if len(started) >= 2:
            raise RuntimeError('Same-ID infrastructure retry exhausted')
        attempt = len(started) + 1
        common = {'request_id': request['request_id'], 'seed': request['seed'],
                  'parent': request['parent'], 'split': request['split'],
                  'attempt': attempt, 'worker_id': worker}
        event = dict(common, event='START', at=now())
        append(journal, event)
        append(Path(root) / 'ATTEMPTS.jsonl', event)
        begin = time.monotonic()
        try:
            row = backend.generate_one(copy.deepcopy(request))
        except BaseException as error:
            event = dict(common, event='INFRASTRUCTURE_FAILURE', at=now(),
                         error_type=type(error).__name__, error=str(error),
                         elapsed_seconds=time.monotonic() - begin,
                         generated_tokens_known=None)
            append(journal, event)
            append(Path(root) / 'ATTEMPTS.jsonl', event)
            raise
        row = dict(row, worker_id=worker, attempt=attempt,
                   request_sha256=digest(request), saved_at=now())
        immutable(path, row)
        errors = validate_record(row, request, config)
        event = dict(common, event='SAVED', at=now(), result_sha256=file_sha(path),
                     generated_tokens_known=row.get('generated_tokens'),
                     elapsed_seconds=time.monotonic() - begin, technical_errors=errors)
        append(journal, event)
        append(Path(root) / 'ATTEMPTS.jsonl', event)
        if errors:
            raise RuntimeError('Saved result has technical errors; preserve and inspect: ' + ','.join(errors))
        return 'GENERATED'


def check_prechecks(root):
    manifest = read(Path(root) / 'PRECHECKS.json')
    records = manifest.get('checks', {})
    for name in PRECHECK_NAMES:
        item = records.get(name, {})
        path = Path(item.get('path', ''))
        if not path.is_absolute():
            path = Path(root) / path
        if not path.is_file() or file_sha(path) != item.get('sha256'):
            raise RuntimeError('Missing or changed precheck evidence: ' + name)
        if read(path).get('status') not in PRECHECK_STATUSES[name]:
            raise RuntimeError('Precheck has not passed: ' + name)
    return {'manifest_sha256': file_sha(Path(root) / 'PRECHECKS.json'), 'checks': records}


def collect(root, split, *, require_workers=True):
    root, inputs, config = load_context(root)
    requests = requests_for(inputs, split)
    expected_paths = {result_path(root, r) for r in requests}
    found = set((root / 'RECORDS' / split).glob('*/*.json'))
    if found != expected_paths:
        raise RuntimeError(f'{split} incomplete/extra: expected {len(expected_paths)}, found {len(found)}, missing {len(expected_paths-found)}, extra {len(found-expected_paths)}')
    rows = []
    for request in requests:
        row = read(result_path(root, request))
        errors = validate_record(row, request, config)
        if row.get('request_sha256') != digest(request):
            errors.append('request_sha256')
        if errors:
            raise RuntimeError(request['request_id'] + ': ' + ','.join(errors))
        rows.append(row)
    workers = sorted({r['worker_id'] for r in rows})
    if require_workers:
        for worker in workers:
            check_worker_closure(root, worker, [r for r in rows if r['worker_id'] == worker], config)
    payload = ''.join(canonical(r) + '\n' for r in rows)
    target = root / 'RAW' / (split + '.jsonl')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text() != payload:
        raise FileExistsError('Preserve existing collection')
    publish_bytes(target, payload.encode())
    receipt = {'status': 'COMPLETE_TECHNICAL_RECORDS', 'split': split, 'requests': len(rows),
               'raw_sha256': file_sha(target), 'workers': workers, 'optimizer_updates': 0,
               'input_manifest_sha256': file_sha(root / 'INPUT_MANIFEST.json'),
               'science_manifest_sha256': file_sha(root / 'SCIENCE_MANIFEST.json'),
               'sampled_tokens': sum(r['generated_tokens'] for r in rows),
               'training_authorized': False}
    immutable(root / 'EXECUTION' / (split + '_COMPLETE.json'), receipt)
    return receipt


def seal_smoke(root):
    root = Path(root)
    evidence = check_prechecks(root)
    receipt = collect(root, 'SMOKE')
    seal = dict(receipt, status='SMOKE_TECHNICAL_PASS', evidence=evidence)
    immutable(root / 'EXECUTION' / 'SMOKE_RECEIPT.json', seal)
    return seal


def verify_smoke(root):
    saved = read(Path(root) / 'EXECUTION/SMOKE_RECEIPT.json')
    expected = dict(collect(root, 'SMOKE'), status='SMOKE_TECHNICAL_PASS', evidence=check_prechecks(root))
    if saved != expected:
        raise RuntimeError('Smoke evidence changed after sealing')


def run(root, split, parent, shard_index=0, shard_count=1):
    root, inputs, config = load_context(root)
    if parent not in config['model']['parents']:
        raise ValueError('Unknown frozen parent')
    if split == 'MAIN':
        verify_smoke(root)
    requests = requests_for(inputs, split, parent, shard_index, shard_count)
    config = copy.deepcopy(config)
    config['runtime_binding'] = read(root / 'RUNTIME_CONFIG.json')
    config['runtime_binding']['inputs_dir'] = str(inputs)
    from backend.native_adapter import make_backend
    worker = f'{split}-{parent}-{shard_index}of{shard_count}-{uuid.uuid4().hex[:12]}'
    worker_dir = root / 'WORKERS' / worker
    immutable(worker_dir / 'SOURCE.json', {'files': source_inventory()})
    backend = make_backend(config, parent, root)
    reused = []
    try:
        identity = backend.identity()
        errors = identity_errors(identity, config, parent)
        if errors:
            raise RuntimeError('Loaded identity failed: ' + ','.join(errors))
        immutable(worker_dir / 'START.json', {'at': now(), 'parent': parent,
                  'identity': identity, 'host': socket.gethostname(),
                  'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                  'shard_index': shard_index, 'shard_count': shard_count})
        counts = {'REUSED': 0, 'GENERATED': 0}
        for request in requests:
            status = execute_request(root, request, config, backend, worker)
            if status == 'REUSED':
                reused.append(read(result_path(root, request)))
            counts[status] += 1
            print(json.dumps({'worker': worker, 'request_id': request['request_id'],
                              'status': status, **counts}), flush=True)
    finally:
        try:
            identity = backend.close()
            if identity is None:
                identity = backend.identity()
            if closed_identity_errors(identity):
                raise RuntimeError('Backend did not verify final model identity')
            immutable(worker_dir / 'END.json', {'at': now(), 'parent': parent,
                      'identity': identity, 'status': 'CLOSED_READONLY'})
        except BaseException as error:
            immutable(worker_dir / 'CLOSE_FAILED.json', {'at': now(), 'parent': parent,
                      'error_type': type(error).__name__, 'error': str(error)})
            raise
        reconcile_reused(root, reused, worker, identity)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--config', type=Path, default=PACKAGE / 'config/plan.json')
    prep.add_argument('--inputs', type=Path, default=PACKAGE / 'data')
    prep.add_argument('--run-root', type=Path, required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--run-root', type=Path, required=True)
    runner.add_argument('--split', choices=SPLITS, required=True)
    runner.add_argument('--parent', choices=['b0', 'b1', 'b2'], required=True)
    runner.add_argument('--shard-index', type=int, default=0)
    runner.add_argument('--shard-count', type=int, default=1)
    seal = sub.add_parser('seal-smoke')
    seal.add_argument('--run-root', type=Path, required=True)
    gather = sub.add_parser('collect')
    gather.add_argument('--run-root', type=Path, required=True)
    gather.add_argument('--split', choices=SPLITS, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.config, args.inputs, args.run_root)
    elif args.command == 'run':
        result = run(args.run_root, args.split, args.parent, args.shard_index, args.shard_count)
    elif args.command == 'seal-smoke':
        result = seal_smoke(args.run_root)
    else:
        result = collect(args.run_root, args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))
