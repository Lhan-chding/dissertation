"""Read-only baseline/bounded saved-tensor comparison on original trajectories.

No sampling, optimizer step, scheduler step, production write, or recovery is
performed. Each probe/mode uses a fresh process and independently loaded model.
The output must be a NEW package/run/memory_diag_* directory. Original path
locks are opened read-only and shared; an active writer prevents diagnosis.
Baseline OOM is retained and yields EQUIVALENCE_UNVERIFIED, never a pass.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from adapter.backend import object_sha
from adapter.transactions import PathTransactions, file_hash, read_json, require, write_once

PROBES = {'b0-11031-R': 4, 'b1-11031-R': 2, 'b1-11031-G': 2}
OFFLOAD_CAP_BYTES = 4 * 1024 ** 3
MINIMUM_HOST_MIB = (24 + 4) * 1024
OFFLOAD_MODE = 'bounded_cpu_saved_tensors'


def storage_identity(tensor):
    return str(tensor.device), tensor.untyped_storage().data_ptr()


class BoundedSavedTensors:
    """Lossless pageable-CPU snapshots, capped cumulatively for one forward.

    Parameter/buffer storage aliases, CPU tensors, and noncontiguous tensors
    pass through detached. Nothing is compressed, cast, deduplicated, or retried.
    ``force_cpu_copy`` exists only for the independent small CPU fixture.
    """
    def __init__(self, persistent_tensors=(), cap_bytes=OFFLOAD_CAP_BYTES, force_cpu_copy=False):
        require(type(cap_bytes) is int and 0 <= cap_bytes <= OFFLOAD_CAP_BYTES, 'Invalid offload cap')
        self.persistent = {storage_identity(tensor) for tensor in persistent_tensors}
        self.cap_bytes = cap_bytes
        self.force_cpu_copy = force_cpu_copy
        self.stats = dict(cap_bytes=cap_bytes, copied_bytes=0, copied_tensors=0,
                          kept_bytes=0, kept_tensors=0, persistent_storage_skips=0,
                          unpacked_copies=0, scope='cumulative per isolated original-row forward')

    def pack(self, tensor):
        import torch
        size = tensor.numel() * tensor.element_size()
        persistent = storage_identity(tensor) in self.persistent
        eligible = (tensor.device.type == 'cuda' or self.force_cpu_copy) and \
            tensor.layout == torch.strided and tensor.is_contiguous() and not persistent
        if eligible and self.stats['copied_bytes'] + size <= self.cap_bytes:
            snapshot = tensor.detach().to(device='cpu', copy=True, non_blocking=False)
            require(snapshot.dtype == tensor.dtype and snapshot.shape == tensor.shape and
                    snapshot.is_contiguous() and not snapshot.is_pinned(), 'CPU snapshot layout/dtype differs')
            self.stats['copied_bytes'] += size
            self.stats['copied_tensors'] += 1
            return ('copied', snapshot, tensor.device, tensor.dtype)
        self.stats['kept_bytes'] += size
        self.stats['kept_tensors'] += 1
        self.stats['persistent_storage_skips'] += int(persistent)
        return ('kept', tensor.detach())

    def unpack(self, packed):
        if packed[0] == 'kept':
            return packed[1]
        _, snapshot, device, dtype = packed
        restored = snapshot.to(device=device, non_blocking=False)
        require(restored.dtype == dtype, 'Offload unpack cast is forbidden')
        self.stats['unpacked_copies'] += 1
        return restored

    def context(self):
        import torch
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


def select_probes(rows, expected_ids):
    require(len(rows) == len(expected_ids) == 128 and len(set(expected_ids)) == 128,
            'Exactly 128 distinct frozen next-step trajectories required')
    by_id = {row['request_id']: row for row in rows}
    require(len(by_id) == 128 and set(by_id) == set(expected_ids), 'Missing/duplicate original trajectory')
    ordered = [by_id[rid] for rid in expected_ids]
    require(all(isinstance(row.get('tokens'), list) and 1 <= len(row['tokens']) <= 1536
                and all(type(v) is int and v >= 0 for v in row['tokens']) for row in ordered),
            'Invalid original completion tokens')
    first = ordered[0]
    longest = max(ordered, key=lambda row: len(row['tokens']))  # First in frozen order wins ties.
    result = [{'request_id': first['request_id'], 'roles': ['first_original_next_step'],
               'completion_tokens': len(first['tokens'])}]
    if longest['request_id'] == first['request_id']:
        result[0]['roles'].append('longest_completion_next_step')
    else:
        result.append({'request_id': longest['request_id'], 'roles': ['longest_completion_next_step'],
                       'completion_tokens': len(longest['tokens'])})
    return result


def memory_budget(env, cgroup_limit_bytes=None, available_bytes=None):
    grant, tasks = env.get('SLURM_MEM_PER_NODE', ''), env.get('SLURM_NTASKS', '')
    require(grant.isdigit() and tasks.isdigit() and int(tasks) > 0, 'Explicit Slurm memory/task grant required')
    limits = {'slurm_per_worker_mib': int(grant) // int(tasks)}
    if cgroup_limit_bytes is not None:
        limits['cgroup_limit_mib'] = cgroup_limit_bytes // (1024 ** 2)
    if available_bytes is not None:
        limits['host_available_mib'] = available_bytes // (1024 ** 2)
    limits['required_minimum_mib'] = MINIMUM_HOST_MIB
    limits['offload_cap_mib'] = OFFLOAD_CAP_BYTES // (1024 ** 2)
    limits['effective_mib'] = min(v for k, v in limits.items() if k.endswith('_mib') and
                                k not in ('required_minimum_mib', 'offload_cap_mib'))
    limits['status'] = 'PASS' if limits['effective_mib'] >= MINIMUM_HOST_MIB else 'BLOCKED_HOST_MEMORY'
    return limits


def host_preflight():
    candidates = [Path('/sys/fs/cgroup/memory.max'),
                  Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')]
    membership = Path('/proc/self/cgroup')
    if membership.exists():
        for line in membership.read_text().splitlines():
            _, controllers, suffix = line.split(':', 2)
            if controllers == '' or 'memory' in controllers.split(','):
                base = Path('/sys/fs/cgroup') if controllers == '' else Path('/sys/fs/cgroup/memory')
                candidate = base / suffix.lstrip('/') / ('memory.max' if controllers == '' else 'memory.limit_in_bytes')
                if candidate.resolve().is_relative_to(base):
                    candidates.append(candidate)
    finite = []
    for path in candidates:
        if path.is_file():
            value = path.read_text().strip()
            if value.isdigit() and int(value) < 2 ** 60:
                finite.append(int(value))
    available = None
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
    return memory_budget(os.environ, min(finite) if finite else None, available)


@contextmanager
def readonly_path(directory):
    """Reuse exact transaction verification without its directory/LOCK creation."""
    directory = Path(directory)
    require(directory.is_absolute() and directory.resolve(strict=True) == directory,
            'Canonical original path required')
    identity = read_json(directory / 'IDENTITY.json')
    tx = PathTransactions(directory, identity)
    lock = directory / 'LOCK'
    require(lock.is_file() and not lock.is_symlink(), 'Existing original LOCK required')
    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        tx._fd = fd
        tx._load_attempts()
        require(not any('complete' not in a for a in tx._attempts.values()),
                'UNKNOWN original attempt forbids diagnostics')
        tx.published()  # Reject staging, unexpected checkpoints, broken hashes/chains.
        yield tx
    finally:
        tx._fd = None
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def production_snapshot(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        require(not path.is_symlink(), 'Symlink in original evidence')
        if path.is_file():
            result[str(path.relative_to(directory))] = file_hash(path)
    return result


def inspect_source(binding_path, path_name):
    require(path_name in PROBES, 'Unregistered diagnostic path')
    binding = read_json(binding_path)
    package = Path(binding['package_root'])
    require(package == PACKAGE and package.resolve(strict=True) == package, 'Wrong bound package')
    path = package / 'run/paths' / path_name
    expected_step = PROBES[path_name]
    with readonly_path(path) as tx:
        markers = tx.published()
        require(markers and markers[-1]['step'] == expected_step, 'Latest checkpoint differs from predeclared probe')
        marker = markers[-1]
        parent, seed, arm = path_name.split('-')
        require(all(tx.identity[k] == v for k, v in {'parent': parent, 'train_seed': int(seed),
                'arm': arm, 'run_id': 'SR-ORD-TRAIN-1/' + path_name,
                'manifest_sha256': binding['manifest_sha256'], 'source_sha256': binding['source_sha256'],
                'code_commit': binding['code_commit'],
                'implementation_sha256': object_sha(binding['implementation_files'])}.items()),
                'Original path/binding identity differs')
        plan_file = package / 'frozen/receipt_plans/paths' / (path_name + '.json')
        index = read_json(package / 'frozen/receipt_plans/INDEX.json')
        require(file_hash(plan_file) == tx.identity['plan_sha256'] == index['files']['paths/' + path_name + '.json'],
                'Frozen path plan differs')
        plan_step = read_json(plan_file)[expected_step]
        require(plan_step['step'] == expected_step + 1, 'Frozen next step differs')
        requests_file = Path(binding['requests_manifest']['path'])
        require(file_hash(requests_file) == binding['requests_manifest']['sha256'], 'Request manifest changed')
        requests = {r['request_id']: r for r in map(json.loads, requests_file.read_text().splitlines())}
        records = []
        for attempt in tx._attempts.values():
            begin = attempt['begin']
            if begin['step'] != expected_step + 1:
                continue
            request = begin['request']
            require(requests.get(request['request_id']) == request, 'Original attempt is not frozen request')
            require(begin['policy_sha256'] == marker['policy_sha256'], 'Original attempt policy differs')
            raw = tx._verify_raw(attempt)
            require(raw.get('policy_hash') == marker['policy_sha256'] and
                    raw.get('reference_hash') == marker['reference_sha256'] and
                    raw.get('generation_status') == 'COMPLETE' and
                    raw.get('native_exception') is None and raw.get('technical_validation_errors') == [],
                    'Unresolved or different original trajectory')
            capture = Path(raw['native_capture_path'])
            require(capture.resolve(strict=True).is_relative_to(path) and not capture.is_symlink() and
                    file_hash(capture) == raw['native_capture_sha256'], 'Original capture changed/escaped')
            records.append(raw)
        expected_ids = [rid for group in plan_step['groups'] for rid in group['request_ids']]
        probes = select_probes(records, expected_ids)
        records = sorted(records, key=lambda row: expected_ids.index(row['request_id']))
        manifest = {'path': path_name, 'checkpoint_step': expected_step, 'next_step': expected_step + 1,
                    'binding_sha256': file_hash(binding_path), 'identity': tx.identity,
                    'commit_sha256': marker['commit_sha256'], 'state_sha256': marker['state_sha256'],
                    'policy_sha256': marker['policy_sha256'], 'reference_sha256': marker['reference_sha256'],
                    'rng_sha256': marker['rng_sha256'], 'plan_step': plan_step, 'probes': probes,
                    'production_files': production_snapshot(path), 'source_selection':
                    'first and longest original next-step completion; ties use frozen request order; deduplicated',
                    'new_generations': 0, 'optimizer_steps': 0, 'production_recovery_authorized': False}
    return manifest, records


def require_output(output, create=False):
    output = Path(os.path.abspath(output))
    require(output.parent == PACKAGE / 'run' and output.name.startswith('memory_diag_'),
            'Output must be a separate package/run/memory_diag_* directory')
    require(output.parent.resolve(strict=True) == output.parent and not output.is_symlink(),
            'Noncanonical diagnostic output')
    if create:
        require(not output.exists(), 'Diagnostic output must be new; retain prior attempts')
        output.mkdir()
    else:
        require(output.is_dir() and output.resolve(strict=True) == output, 'Missing diagnostic output')
    return output


def record_final_audit(result, audit):
    """Keep the original OOM/failure even when a subsequent audit also fails."""
    try:
        audit()
    except Exception as error:
        result['final_audit_error'] = {'error_type': type(error).__name__, 'error': str(error),
                                       'traceback': traceback.format_exc()}
        if result['status'] == 'COMPLETE':
            result['status'] = 'FAILED_FINAL_AUDIT'


def run_probe(binding_path, output, index, mode):
    """One fresh model, one original row, one backward; never an update."""
    output = require_output(output)
    manifest = read_json(output / 'MANIFEST.json')
    execution = output / (f'probe-{index}-{mode}')
    execution.mkdir(exist_ok=False)
    result = {'status': 'UNFINISHED', 'mode': mode, 'probe': manifest['probes'][index],
              'new_generations': 0, 'optimizer_steps': 0, 'scheduler_steps': 0}
    backend = None
    hook = None
    before = None
    rng = None
    try:
        budget = host_preflight()
        result['host_memory_preflight'] = budget
        require(budget['status'] == 'PASS', 'BLOCKED_HOST_MEMORY: minimum 24 GiB base plus 4 GiB offload cap')
        current_manifest, records = inspect_source(binding_path, manifest['path'])
        require(current_manifest == manifest, 'Original evidence changed after probe registration')
        binding = read_json(binding_path)
        identity = manifest['identity']
        binding['path_identity'] = {k: identity[k] for k in ('arm', 'train_seed', 'run_id')}
        original_path = PACKAGE / 'run/paths' / manifest['path']
        with readonly_path(original_path) as tx:
            from adapter.backend import NativeBackend
            backend = NativeBackend(binding, identity['parent'], execution, 'TRAINING')
            runtime = backend.runtime
            import torch
            from mm_core.training import capture_rng, restore_rng, state_hash, trainable_state
            from adapter.training import validate_records, canonical_hash, _verify_prepared
            from adapter.seal_smoke import independent_scores
            from sr_f6.objective import sequence_loss
            marker = tx.restore_latest(runtime, backend.optimizer, backend.scheduler, backend.reference)
            require(marker['step'] == manifest['checkpoint_step'], 'Restored checkpoint cursor differs')
            runtime.identity['plan_step_sha256'] = canonical_hash(manifest['plan_step'])
            cases = {r['case_id']: r for r in map(json.loads,
                     (PACKAGE / 'frozen/private_cases.jsonl').read_text().splitlines())}
            scores = independent_scores(PACKAGE, records, cases)
            score_by_id = {s['request_id']: s['score'] for s in scores['scores']}
            records = [dict(row, score=score_by_id[row['request_id']]) for row in records]
            coefficients, _ = validate_records(runtime.identity, records, manifest['plan_step'],
                                                backend.policy_hash(), state_hash(backend.reference))
            probe_id = manifest['probes'][index]['request_id']
            position = next(i for i, row in enumerate(records) if row['request_id'] == probe_id)
            row, advantage = records[position], coefficients[position]
            rng = capture_rng()
            require(state_hash(rng) == manifest['rng_sha256'], 'Diagnostic starting RNG differs')
            before = {'policy': backend.policy_hash(), 'reference': state_hash(backend.reference),
                      'optimizer': state_hash(backend.optimizer.state_dict()),
                      'scheduler': state_hash(backend.scheduler.state_dict())}
            runtime.model.eval()
            backend.optimizer.zero_grad(set_to_none=True)
            prepared = runtime.prepare_input(row['model_input'], backend.inputs_dir, protocol='structured_v4')
            _verify_prepared(row, prepared, torch)
            item = dict(row, prepared=prepared)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                reference = runtime.batch_reference_forward([item], backend.reference)[0]
            hook = BoundedSavedTensors((*runtime.model.parameters(), *runtime.model.buffers()))
            context = hook.context() if mode == OFFLOAD_MODE else nullcontext()
            with context:
                current = runtime.batch_sequence_forward([item], purpose='training_policy', grad=True)[0]
                sampler = torch.tensor(row['sampler_logprobs'], device=current.device, dtype=torch.float32)
                loss = sequence_loss(current, sampler, reference, advantage, 0.)
                scaled = loss['loss'] / 128
            runtime.reserve('diagnostic_gradient_calls', 1, method='saved_tensor_memory_equivalence')
            scaled.backward()
            result['saved_tensor_offload'] = hook.stats
            gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                         for name, parameter in runtime.model.named_parameters() if parameter.requires_grad}
            require(gradients and any(value is not None for value in gradients.values()) and
                    all(value is None or bool(torch.isfinite(value).all()) for value in gradients.values()),
                    'Nonfinite or missing adapter gradient inventory')
            gradient_file = execution / 'adapter_gradients.pt'
            with gradient_file.open('xb') as stream:
                torch.save(gradients, stream)
            result.update(status='COMPLETE', request_id=probe_id, token_path_sha256=row['token_path_sha256'],
                          prompt_tensor_hash=row['prompt_tensor_hash'], advantage=advantage,
                          train_logprobs=current.detach().cpu().tolist(),
                          reference_logprobs=reference.detach().cpu().tolist(),
                          scalar_terms={k: float(v.detach()) for k, v in loss.items()},
                          scaled_loss=float(scaled.detach()), gradient_file=str(gradient_file),
                          gradient_file_sha256=file_hash(gradient_file),
                          gradient_names=list(gradients), gradient_none_names=[k for k, v in gradients.items() if v is None])
            require(all(__import__('math').isfinite(v) for v in result['train_logprobs']+
                        result['reference_logprobs']+list(result['scalar_terms'].values())), 'Nonfinite probability/loss')
            after = {'policy': backend.policy_hash(), 'reference': state_hash(backend.reference),
                     'optimizer': state_hash(backend.optimizer.state_dict()),
                     'scheduler': state_hash(backend.scheduler.state_dict())}
            require(after == before, 'Diagnostic changed policy/reference/optimizer/scheduler')
            restore_rng(rng)
            require(state_hash(capture_rng()) == manifest['rng_sha256'], 'Diagnostic RNG restore differs')
            result['immutable_state_hashes'] = after
            result['restored_rng_sha256'] = state_hash(capture_rng())
            result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            require(production_snapshot(original_path) == manifest['production_files'], 'Production evidence changed')
    except Exception as error:
        result.update(status='OOM' if error.__class__.__name__ == 'OutOfMemoryError' else 'FAILED',
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if backend is not None:
            torch = backend.torch
            result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    finally:
        def audit():
            if backend is not None:
                backend.optimizer.zero_grad(set_to_none=True)
                if before is not None:
                    after = {'policy': backend.policy_hash(), 'reference': state_hash(backend.reference),
                             'optimizer': state_hash(backend.optimizer.state_dict()),
                             'scheduler': state_hash(backend.scheduler.state_dict())}
                    result['state_unchanged_after_attempt'] = after == before
                    require(after == before, 'Diagnostic state changed during attempt')
                if rng is not None:
                    restore_rng(rng)
                    result['rng_restored_after_attempt'] = state_hash(capture_rng()) == manifest['rng_sha256']
                    require(result['rng_restored_after_attempt'], 'Diagnostic RNG failed final restoration')
        record_final_audit(result, audit)
        if hook is not None:
            result['saved_tensor_offload'] = hook.stats
        result['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)
        result['allocator_environment'] = {k: os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')}
        write_once(execution / 'RESULT.json', result)
    return result


def compare_results(baseline, offload):
    result = {'status': 'EQUIVALENCE_UNVERIFIED', 'baseline_status': baseline['status'],
              'offload_status': offload['status'], 'production_recovery_authorized': False}
    if baseline['status'] != 'COMPLETE' or offload['status'] != 'COMPLETE':
        return result
    require(baseline.get('mode') == 'baseline' and offload.get('mode') == OFFLOAD_MODE,
            'Comparison requires baseline and bounded saved-tensor modes')
    if not all(value.get('state_unchanged_after_attempt') is True and
               value.get('rng_restored_after_attempt') is True for value in (baseline, offload)):
        result['reason'] = 'State/RNG final audit not verified for both modes'
        return result
    if offload.get('saved_tensor_offload', {}).get('copied_bytes', 0) <= 0:
        result['reason'] = 'Bounded saved-tensor copy path was not exercised'
        return result
    import torch
    scalar_fields = ('request_id', 'token_path_sha256', 'prompt_tensor_hash', 'advantage',
                     'train_logprobs', 'reference_logprobs', 'scalar_terms', 'scaled_loss',
                     'immutable_state_hashes', 'restored_rng_sha256')
    result['exact_fields'] = {k: baseline[k] == offload[k] for k in scalar_fields}
    gradients = []
    for value in (baseline, offload):
        require(file_hash(value['gradient_file']) == value['gradient_file_sha256'], 'Gradient evidence changed')
        gradients.append(torch.load(value['gradient_file'], map_location='cpu', weights_only=True))
    left, right = gradients
    require(left.keys() == right.keys(), 'Adapter gradient inventory differs')
    result['gradient_comparison'] = {}
    for name in left:
        a, b = left[name], right[name]
        exact = a is None and b is None
        difference = None
        if a is not None and b is not None:
            require(a.shape == b.shape and a.dtype == b.dtype, 'Adapter gradient shape/dtype differs')
            exact = torch.equal(a, b)
            difference = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
        result['gradient_comparison'][name] = {'exact_equal': exact, 'max_absolute_difference': difference}
    equal = all(result['exact_fields'].values()) and all(v['exact_equal'] for v in result['gradient_comparison'].values())
    result['status'] = 'EXACT_PROBE_EQUIVALENCE' if equal else 'PROBE_DIFFERENCE'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--path', choices=tuple(PROBES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-mode', choices=('baseline', OFFLOAD_MODE))
    parser.add_argument('--probe-index', type=int)
    args = parser.parse_args()
    if args.worker_mode:
        require(args.probe_index is not None and args.probe_index in (0, 1), 'Declared probe index required')
        result = run_probe(args.binding.resolve(strict=True), args.output, args.probe_index, args.worker_mode)
        print(json.dumps({k: result[k] for k in ('status', 'mode', 'probe')}))
        return
    require(args.probe_index is None, 'Probe index only permitted in internal worker mode')
    output = require_output(args.output, create=True)
    budget = host_preflight()
    write_once(output / 'HOST_PREFLIGHT.json', budget)
    if budget['status'] != 'PASS':
        write_once(output / 'SUMMARY.json', {'status': 'BLOCKED_HOST_MEMORY', 'memory': budget,
                   'new_generations': 0, 'optimizer_steps': 0, 'production_recovery_authorized': False})
        print('BLOCKED_HOST_MEMORY: minimum 24 GiB base plus 4 GiB offload cap; no model loaded')
        return
    manifest, _ = inspect_source(args.binding.resolve(strict=True), args.path)
    write_once(output / 'MANIFEST.json', manifest)
    comparisons = []
    for index, probe in enumerate(manifest['probes']):
        values = []
        for mode in ('baseline', OFFLOAD_MODE):
            command = [sys.executable, '-B', '-m', 'diagnostics.memory_equivalence', '--binding',
                       str(args.binding.resolve(strict=True)), '--path', args.path, '--output', str(output),
                       '--worker-mode', mode, '--probe-index', str(index)]
            with (output / f'probe-{index}-{mode}.log').open('xb') as log:
                completed = subprocess.run(command, cwd=PACKAGE, stdout=log, stderr=subprocess.STDOUT)
            result_path = output / f'probe-{index}-{mode}/RESULT.json'
            value = read_json(result_path) if result_path.exists() else {
                'status': 'PROCESS_FAILED_WITHOUT_RESULT', 'returncode': completed.returncode}
            values.append(value)
        comparisons.append({'probe': probe, 'comparison': compare_results(*values)})
    original = PACKAGE / 'run/paths' / args.path
    with readonly_path(original):
        unchanged = production_snapshot(original) == manifest['production_files']
    summary = {'status': 'DIAGNOSTICS_COMPLETE_NOT_PRODUCTION_AUTHORIZATION', 'path': args.path,
               'production_files_unchanged': unchanged, 'comparisons': comparisons,
               'new_generations': 0, 'optimizer_steps': 0, 'scheduler_steps': 0,
               'diagnostic_source_sha256': file_hash(__file__), 'production_recovery_authorized': False}
    write_once(output / 'SUMMARY.json', summary)
    require(unchanged, 'Production files changed; do not accept comparison')
    print(json.dumps({'path': args.path, 'status': summary['status'],
                      'comparisons': [v['comparison']['status'] for v in comparisons]}))


if __name__ == '__main__':
    main()
