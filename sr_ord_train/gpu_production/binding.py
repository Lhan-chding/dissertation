"""Explicit execution-version bridge; never rewrite the frozen RUN_BINDING.

The original path identity remains its scientific lineage. Actual new code,
technical acceptance and execution directory are separately bound before load.
"""
from __future__ import annotations

import copy
from pathlib import Path
import re

from adapter.backend import object_sha, verify_file
from adapter.transactions import _plain, file_hash, read_json, require

SCHEMA = 'sr-ord-train-gpu-production-v1'
MODE = 'cuda0_compute_cuda1_saved_tensors_16GiB'
ACCEPTANCE_SCHEMA = 'sr-ord-train-gpu-acceptance-v1'
REQUIRED_FILES = ('gpu_production/binding.py', 'gpu_production/backend.py',
                  'gpu_production/state.py', 'gpu_production/driver.py', 'gpu_production/update.py',
                  'diagnostics/gpu_saved_tensors.py', 'diagnostics/gpu_recovery_state.py')


def path_name(identity):
    require(isinstance(identity, dict) and set(identity) == {'arm', 'train_seed', 'run_id'},
            'Exact registered path identity required')
    name = identity.get('run_id', '').removeprefix('SR-ORD-TRAIN-1/')
    match = re.fullmatch(r'(b[01])-(11031|11032)-([RGD])', name)
    require(match is not None and identity['run_id'] == 'SR-ORD-TRAIN-1/' + name
            and type(identity['train_seed']) is int and identity['train_seed'] == int(match[2])
            and identity['arm'] == match[3], 'Registered path identity fields disagree')
    return name


def execution_directory(package, path, identity):
    path = _plain(path)
    expected = package / 'run/paths' / path_name(identity) / 'executions'
    require(path.is_absolute() and path.is_dir() and path.resolve(strict=True) == path
            and path.parent == expected and bool(re.fullmatch(r'[A-Za-z0-9_-]+', path.name)),
            'Canonical exact registered production execution directory required')
    return path


def _reference(path):
    path = _plain(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            'Canonical immutable evidence file required')
    return dict(path=str(path), sha256=file_hash(path))


def _evidence(ref, package):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Exact evidence reference required')
    p = _plain(verify_file(ref))
    require(p.is_absolute() and p.resolve(strict=True) == p and p.is_relative_to(package),
            'Evidence outside canonical package')
    return p


def verify_acceptance(ref, original, added_rows, execution_code_commit, name):
    package = Path(original['package_root'])
    path = _evidence(ref, package)
    require(path.is_relative_to(package / 'run'), 'Production acceptance must be experiment run evidence')
    a = read_json(path)
    require(set(a) == {'schema', 'status', 'source_run_binding', 'diagnostic_summary', 'allowed_mode',
                       'original_long_gradient_equivalence', 'execution_code_commit',
                       'implementation_sha256', 'allowed_paths'}, 'Acceptance fields differ')
    require(a['schema'] == ACCEPTANCE_SCHEMA and a['status'] == 'GPU_ONLY_PRODUCTION_BRIDGE_ACCEPTED'
            and a['allowed_mode'] == MODE, 'GPU-only technical acceptance required')
    require(a['execution_code_commit'] == execution_code_commit
            and a['implementation_sha256'] == object_sha(added_rows), 'Accepted execution version differs')
    require(isinstance(a['allowed_paths'], list) and a['allowed_paths']
            and len(a['allowed_paths']) == len(set(a['allowed_paths']))
            and all(re.fullmatch(r'b[01]-(11031|11032)-[RGD]', n) for n in a['allowed_paths'])
            and name in a['allowed_paths'], 'Path not accepted for GPU production')
    source = _evidence(a['source_run_binding'], package)
    require(source == package / 'execution/RUN_BINDING.json' and read_json(source) == original,
            'Acceptance original binding differs')
    summary = read_json(_evidence(a['diagnostic_summary'], package))
    require(summary['status'] == 'FULL_BATCH_PRE_ADAM_PASS_NO_UPDATE'
            and summary['baseline_responses'] == summary['gradient_forwards'] == 128
            and summary['diagnostic_backward_accounting']['backward_calls_completed'] == 128
            and summary['probability_comparison'] == 'EXACT_ALL_128'
            and summary['historical_gradient_overlap']['status'] == 'EXACT_HISTORICAL_OVERLAP'
            and all(summary[k] is True for k in ('production_files_unchanged', 'rng_restored',
                                                'rng_restored_after_attempt', 'state_unchanged',
                                                'state_unchanged_after_attempt'))
            and all(summary[k] == 0 for k in ('new_generations', 'optimizer_body_calls',
                                             'optimizer_steps', 'scheduler_steps')),
            'Required original full-batch GPU diagnostic evidence absent')
    require(summary['activation_storage'] == 'cuda:1; arithmetic cuda:0; CPU fallback forbidden'
            and summary['saved_tensor_max_bytes_per_forward'] == 16 * 1024**3
            and summary['saved_tensor_copied_bytes_total'] > 0
            and summary['gpu_storage_source_sha256'] == file_hash(package / 'diagnostics/gpu_saved_tensors.py')
            and summary['diagnostic_source_sha256'] == file_hash(package / 'diagnostics/gpu_full_batch_probe.py')
            and summary['native_audit']['status'] == 'PREUPDATE_PASS'
            and summary['native_audit']['precommit_gate'] == 'PASS_BEFORE_ADAM'
            and summary['native_audit']['numerics']['status'] == 'FINITE',
            'Accepted diagnostic memory implementation or numerical gate differs')
    require(a['original_long_gradient_equivalence'] == 'UNVERIFIED_ORIGINAL_BASELINE_OOM'
            and summary['original_long_gradient_equivalence'] == 'UNVERIFIED',
            'Original long-baseline limitation must remain explicit')
    return a


def make_production_binding(source_binding_path, execution_root, additional_files=(), *,
                            path_identity, execution_code_commit, acceptance_path):
    source_path = _plain(source_binding_path)
    original = read_json(source_path)
    package = Path(original['package_root'])
    require(source_path == package / 'execution/RUN_BINDING.json', 'Original RUN_BINDING path required')
    root = execution_directory(package, execution_root, path_identity)
    rows = []
    seen = {r['path'] for r in original['implementation_files']}
    files = list(dict.fromkeys([*(package / p for p in REQUIRED_FILES), *(Path(p) for p in additional_files)]))
    for path in files:
        ref = _reference(path)
        require(Path(ref['path']).is_relative_to(package) and ref['path'] not in seen,
                'New implementation overlaps original or leaves package')
        rows.append(ref)
        seen.add(ref['path'])
    result = copy.deepcopy(original)
    result['implementation_files'] = copy.deepcopy(original['implementation_files']) + rows
    result['path_identity'] = copy.deepcopy(path_identity)
    result['gpu_production'] = dict(schema=SCHEMA, mode=MODE, visible_gpu_count=2,
        source_run_binding=_reference(source_path), execution_root=str(root),
        execution_code_commit=execution_code_commit, acceptance=_reference(acceptance_path),
        original_implementation_sha256=object_sha(original['implementation_files']),
        execution_implementation_sha256=object_sha(rows))
    verify_production_binding(result, root)
    return result


def verify_production_binding(binding, run_dir):
    d = binding.get('gpu_production', {})
    require(set(d) == {'schema', 'mode', 'visible_gpu_count', 'source_run_binding', 'execution_root',
                      'execution_code_commit', 'acceptance', 'original_implementation_sha256',
                      'execution_implementation_sha256'}, 'Exact production bridge fields required')
    require(d['schema'] == SCHEMA and d['mode'] == MODE and d['visible_gpu_count'] == 2
            and bool(re.fullmatch(r'[0-9a-f]{40}', d['execution_code_commit'])),
            'Explicit committed GPU-only production version required')
    source_path = _plain(verify_file(d['source_run_binding']))
    original = read_json(source_path)
    require('gpu_diagnostic' not in original and 'gpu_production' not in original,
            'Original non-derived RUN_BINDING required')
    package = _plain(original['package_root'])
    require(package.is_absolute() and package.resolve(strict=True) == package
            and source_path == package / 'execution/RUN_BINDING.json', 'Canonical source binding required')
    root = execution_directory(package, d['execution_root'], binding['path_identity'])
    require(execution_directory(package, run_dir, binding['path_identity']) == root,
            'Production execution directory differs')
    ignored = {'path_identity', 'implementation_files', 'gpu_production'}
    require({k: v for k, v in binding.items() if k not in ignored} ==
            {k: v for k, v in original.items() if k not in ignored}, 'Original RUN_BINDING fields changed')
    old_rows = original['implementation_files']; rows = binding['implementation_files']
    require(rows[:len(old_rows)] == old_rows and len(rows) > len(old_rows),
            'Original implementation inventory must remain an exact prefix')
    added = rows[len(old_rows):]
    require(d['original_implementation_sha256'] == object_sha(old_rows)
            and d['execution_implementation_sha256'] == object_sha(added), 'Implementation digest differs')
    paths = [_plain(verify_file(r)) for r in rows]
    require(len(paths) == len(set(paths)), 'Duplicate implementation file')
    new_paths = paths[len(old_rows):]
    require(all(p.resolve(strict=True) == p and p.is_relative_to(package) for p in new_paths)
            and {package / p for p in REQUIRED_FILES} <= set(new_paths)
            and Path(__file__).resolve() in new_paths, 'Required production dependency inventory absent')
    verify_acceptance(d['acceptance'], original, added, d['execution_code_commit'], path_name(binding['path_identity']))
    return original
