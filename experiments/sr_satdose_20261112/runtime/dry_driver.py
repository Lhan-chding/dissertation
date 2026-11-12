"""Single-start calibration CLI; never trains, retries an attempt, or unseals MAIN.

Requires the complete frozen request population and SHA-locked runtime sources.
CPU tests use isolated synthetic dependencies and do not generate GPU receipts.
Run as ``python -m <new_package>.runtime.dry_driver`` from its package parent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import sys

from .backend import (FrozenRegistry, GenerationBridge, NativeKernelRunner,
                      file_sha, read_json, require, write_once)
from .native_runtime import build_runtime, verify_source_lock

SCOPES = {'DRY': 640, 'PREFIX_CALIB': 48}


def verify_implementation(freeze_path):
    """Byte-verifiable deployment identity; remote .git is deliberately optional."""
    freeze = read_json(freeze_path)
    require(re.fullmatch(r'[0-9a-f]{40}', freeze.get('code_commit', '')) is not None,
            'Missing frozen scientific code commit')
    runtime = Path(__file__).resolve().parent
    actual = {p.name: file_sha(p) for p in runtime.glob('*.py') if p.is_file()}
    require(actual == freeze.get('runtime_sources'), 'Actual runtime source inventory/SHA differs')
    # Sources are also individually covered by DESIGN_FREEZE artifact references.
    source_refs = {ref['sha256'] for ref in freeze['artifacts'].values()}
    require(set(actual.values()) <= source_refs, 'Runtime sources absent from freeze artifacts')
    return dict(code_commit=freeze['code_commit'], runtime_sources=actual)


def append_raw(path, row):
    require(not path.is_symlink(), 'Raw output cannot be a symlink')
    with path.open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _runner(handle, lock):
    native = importlib.import_module('sr_f6.runtime')
    legacy = importlib.import_module('gpu_production_v2.backend')
    protocol = importlib.import_module('sr_f1.json_protocol')
    paths = {str((Path(lock['native_src']) / rel).resolve()): sha
             for rel, sha in lock['native_files'].items()}
    paths.update({str((Path(lock['legacy_root']) / rel).resolve()): sha
                  for rel, sha in lock['legacy_files'].items()})
    require(file_sha(protocol.__file__) == paths.get(str(Path(protocol.__file__).resolve())),
            'Unbound balanced JSON helper')
    return NativeKernelRunner(handle.runtime, native, legacy, protocol.balanced_cut, source_hashes=paths)


def _scorer(lock):
    mod = importlib.import_module('sr_f6.measurement')
    expected = Path(lock['native_src']).resolve() / 'sr_f6/measurement.py'
    require(Path(mod.__file__).resolve() == expected and
            file_sha(expected) == lock['native_files']['sr_f6/measurement.py'], 'Unbound native scorer')
    return mod.score_response


def _private_cases(registry):
    """Only scorer sees the private world/query; runner never receives it."""
    freeze = read_json(registry.freeze_path)
    ref = freeze['artifacts']['private_cases']
    path = registry.root / ref['path']
    require(path.resolve(strict=True).is_relative_to(registry.root) and file_sha(path) == ref['sha256'],
            'Private scoring cases changed')
    cases = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        require(row['case_id'] not in cases, 'Duplicate private case')
        cases[row['case_id']] = row
    return cases


def run_calibration(registry, source_lock, *, start_seed, scope, output,
                    runtime_builder=build_runtime, runner_factory=_runner, scorer_factory=_scorer):
    """One immutable output transaction. Dependency injection is for CPU tests only."""
    require(scope in SCOPES and start_seed in (11031, 11032), 'Unregistered calibration allocation')
    output = Path(output)
    require(output.is_absolute() and output.parent.resolve(strict=True).is_relative_to(registry.root),
            'Output must be beneath frozen experiment root')
    require(not output.exists() and not output.is_symlink(), 'Existing output requires audit; never auto-resume')
    selected = sorted((row for row in registry.requests.values()
                       if row['scope'] == scope and row.get('start_seed', row.get('train_seed')) == start_seed),
                      key=lambda row: row['request_id'])
    require(len(selected) == SCOPES[scope] and all(row['arm'] == 'START' and row['step'] == 0 for row in selected),
            'Calibration request population differs')
    # All input and source checks occur before loading a model.
    implementation = verify_implementation(registry.freeze_path)
    lock, *_ = verify_source_lock(registry, source_lock)
    cases = _private_cases(registry) if scope == 'DRY' else {}
    for row in selected:
        registry.request(row['request_id'])
        if scope == 'DRY':
            require(row['case_id'] in cases and 'group_id' in row, 'Missing DRY group/case')
    if scope == 'DRY':
        groups = Counter(r['group_id'] for r in selected)
        require(len(groups) == 80 and set(groups.values()) == {8}, 'DRY needs 80 complete G8')
        for row in selected:
            require(len(cases[row['case_id']]['task']['world']['categories']) == 8,
                    'Native training J only supports 8-category DRY')
    output.mkdir(exist_ok=False)
    handle = None
    raw_path = output / ('DRY_RAW.jsonl' if scope == 'DRY' else 'PREFIX_RAW.jsonl')
    lease = (output / 'WORKER.lock').open('x')
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    close_receipt = None
    completed = 0
    try:
        allocations = registry.root / 'calibration_allocations'
        require(not allocations.is_symlink(), 'Allocation directory cannot be symlink')
        allocations.mkdir(exist_ok=True)
        write_once(allocations / f'{scope}-{start_seed}.json', dict(
            output=str(output), design_freeze_sha256=registry.freeze_sha256,
            scope=scope, start_seed=start_seed, status='CLAIMED_NO_AUTOMATIC_RETRY'))
        write_once(output/'INTENT.json', dict(experiment_id=registry.config['experiment_id'],
            scope=scope, start_seed=start_seed, expected_responses=SCOPES[scope],
            design_freeze_sha256=registry.freeze_sha256, source_lock_sha256=file_sha(source_lock),
            implementation=implementation, request_ids=[r['request_id'] for r in selected],
            permission='ZERO_UPDATE_CALIBRATION_ONLY'))
        handle = runtime_builder(registry, source_lock, start_seed=start_seed, output_dir=output)
        before = handle.endpoint()
        expected = registry.config['start_bindings'][str(start_seed)]
        require(before['policy_sha256'] == expected['policy_sha256'] and
                before['reference_sha256'] == expected['reference_sha256'], 'Wrong loaded R24/reference')
        require(handle.fork.optimizer is None and handle.fork.scheduler is None,
                'Read-only calibration must not construct optimizer/scheduler')
        scorer = scorer_factory(lock) if scope == 'DRY' else None
        bridge = GenerationBridge(registry, output, runner_factory(handle, lock), handle.endpoint)
        for request in selected:
            result = bridge.generate_one(request['request_id'])
            require(result.get('seed') == request['seed'] and result.get('model_input') ==
                    {k: request[k] for k in ('prompt', 'image_file', 'image_sha256')},
                    'Native capture input/seed differs from frozen request')
            row = dict(result)
            for key in ('group_id', 'case_id', 'root_id', 'input_id', 'start_seed', 'arm', 'step', 'draw', 'seed', 'panel', 'view'):
                if key in request:
                    row[key] = request[key]
            row.update(scope=scope, suffix_text=result['generated_text'],
                       native_finished_semantics='balanced_only',
                       dry_optimizer_steps=0, dry_parameter_updates=0)
            if scope == 'DRY':
                task = cases[request['case_id']]['task']
                row['native_score'] = scorer(row['full_text'], task['world'], task['query'],
                    finished=row['finish_reason'] == 'balanced', finish_reason=row['finish_reason'])
            else:
                row['native_score'] = None
                row['native_J'] = None
                row['native_J_applicable'] = False
            append_raw(raw_path, row)
            completed += 1
        require(completed == SCOPES[scope], 'Incomplete calibration')
        require(handle.fork.optimizer is None and handle.fork.scheduler is None,
                'Calibration constructed optimizer/scheduler')
        close_receipt = handle.close()
        counts = Counter()
        for event in handle.runtime._satdose_accounting:
            counts[event['kind']] += event['count']
        receipt = dict(status='CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION', scope=scope,
            start_seed=start_seed, responses=completed, optimizer_steps=0, parameter_updates=0,
            source_lock_sha256=file_sha(source_lock), design_freeze_sha256=registry.freeze_sha256,
            policy_before=before, policy_after=close_receipt,
            optimizer_created=False, scheduler_created=False,
            native_accounting=dict(counts), raw=dict(path=raw_path.name, sha256=file_sha(raw_path)),
            complete_files=[dict(path=p.name, sha256=file_sha(p)) for p in sorted(output.glob('*.COMPLETE.json'))],
            no_training_authorized=True)
        write_once(output/'CALIBRATION_COMPLETE.json', receipt)
        return receipt
    except BaseException as exc:
        close_error = None
        if handle is not None and close_receipt is None:
            try:
                close_receipt = handle.close()
            except BaseException as close_exc:
                close_error = repr(close_exc)
        write_once(output/'HOLD.json', dict(status='HOLD_CALIBRATION_TECHNICAL', scope=scope,
            start_seed=start_seed, completed_rows=completed, error_type=type(exc).__name__,
            error=str(exc), close_receipt=close_receipt, close_error=close_error,
            recovery='AUDIT_EXISTING_ATTEMPTS_NO_AUTOMATIC_RETRY', no_training_authorized=True))
        raise
    finally:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'evidence-root', 'freeze', 'inputs', 'images', 'source-lock', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--start-seed', type=int, choices=(11031, 11032), required=True)
    parser.add_argument('--scope', choices=tuple(SCOPES), required=True)
    args = parser.parse_args(argv)
    try:
        registry = FrozenRegistry(args.config, args.evidence_root, args.freeze, args.manifests, args.inputs, args.images)
        result = run_calibration(registry, args.source_lock, start_seed=args.start_seed,
                                 scope=args.scope, output=args.output)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except BaseException as exc:
        # Persist pre-model failures as well, but never overwrite an existing run.
        root = args.evidence_root.resolve(strict=True)
        output = args.output
        if (output.is_absolute() and output.parent.resolve(strict=True).is_relative_to(root)
                and not output.exists() and not output.is_symlink()):
            output.mkdir(exist_ok=False)
            write_once(output/'HOLD.json', dict(status='HOLD_CALIBRATION_PREFLIGHT',
                scope=args.scope, start_seed=args.start_seed, error_type=type(exc).__name__,
                error=str(exc), model_calls=0, no_training_authorized=True))
        raise


if __name__ == '__main__':
    sys.exit(main())
