"""One-shot PREFIX_CALIB control after two fully verified passing DRY starts.

No model is created in this process. The only executable stage is the frozen
DRY driver's PREFIX_CALIB scope, 11031 then 11032, with 48 requests each. A
failed measurement stops the sequence. This never grants training permission.
The separately audited release receipt is supplied by path AND pinned SHA;
neither this module nor its launcher manufactures an audit receipt.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from analysis import calibration
sys.path.pop(0)
core = calibration.core

BASE = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
PYTHON = '/projects/_ssd/varunssd/louis-ssvc/envs/ssvc-py312/bin/python'
FREEZE_SHA = '6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
CONFIG_SHA = '58841639b0f36a69ac77824bc8c683ef7fbeb5c50a41e9975b89b54880f4ecfe'
CLARIFICATION_SHA = 'eac7a44cffd2ba159c5b209faad6bc52eda05c1a843306d659dbdc38dada0337'
PROOF_SHA = 'cd0ca568144bcd6f4058c9c09ec0dfc7c5345cfd6ee1916bfd6443b4e50100d8'
STARTS = (11031, 11032)
MANIFESTS = ('train', 'dry', 'dev', 'prefix_calib', 'main')


def source_inventory(root):
    """All release files, including scorer-pin JSON; no symlinks or pycache."""
    result = {}
    for path in root.rglob('*'):
        if '__pycache__' in path.relative_to(root).parts:
            continue
        core.require(not path.is_symlink(), 'Release symlink forbidden')
        if path.is_file():
            result[str(path.relative_to(root))] = core.file_sha(path)
    return result


def verify_release(receipt_path, expected_sha):
    core.require(re.fullmatch('[0-9a-f]{64}', expected_sha or '') is not None,
                 'A separately pinned audited release SHA is required')
    path = Path(receipt_path).resolve(strict=True)
    core.require(core.file_sha(path) == expected_sha, 'Technical release receipt SHA mismatch')
    receipt = core.read_json(path)
    core.require(receipt.get('schema') == 'sr-satdose-prefix-technical-release-v1' and
                 receipt.get('status') == 'AUDITED_FOR_PREFIX_CALIB_ONLY' and
                 receipt.get('permission') == 'NO_TRAINING', 'Unaudited PREFIX technical release')
    expected = dict(experiment_id='SR-SATDOSE-1-20261112', evidence_root=str(BASE),
                    design_freeze_sha256=FREEZE_SHA, config_sha256=CONFIG_SHA,
                    frozen_code_root=str(BASE/'code'))
    core.require(all(receipt.get(key) == value for key, value in expected.items()),
                 'Technical release frozen identity mismatch')
    root = Path(receipt['release_root']).resolve(strict=True)
    core.require(root == PACKAGE and root != BASE/'code' and not path.is_relative_to(root),
                 'Receipt must bind this separate release, outside its own inventory')
    required = {'execution/prefix_control.py', 'execution/diagnostic_worker.py',
                'scheduling/run_prefix.sh', 'analysis/calibration.py',
                'supplied_package/tools/satdose_core.py', 'supplied_package/config/experiment.json',
                'supplied_package/config/source_bindings.json'}
    actual = source_inventory(root)
    core.require(required <= set(actual) and actual == receipt.get('sources'),
                 'Deployed technical release full inventory/SHA differs')
    core.require(Path(calibration.__file__).resolve() == root/'analysis/calibration.py' and
                 Path(core.__file__).resolve() == root/'supplied_package/tools/satdose_core.py',
                 'Shadowed analysis/helper import')
    core.require(actual['supplied_package/config/experiment.json'] == CONFIG_SHA,
                 'Release scientific config differs')
    return dict(path=str(path), sha256=expected_sha, release_root=str(root), sources=actual)


def verify_deployment():
    """Read-only frozen artifact and deployed native/legacy SOURCE verification.

    The unchanged driver independently verifies model/checkpoint bytes before
    loading any model. This control layer verifies all frozen artifacts and
    source inventories without importing the GPU runtime or opening raw data.
    """
    root = BASE.resolve(strict=True)
    core.require(root == BASE and not BASE.is_symlink(), 'Frozen root changed')
    freeze = root/'DESIGN_FREEZE.json'
    config = root/'code/supplied_package/config/experiment.json'
    core.require(core.file_sha(freeze) == FREEZE_SHA and core.file_sha(config) == CONFIG_SHA,
                 'Immutable freeze/config changed')
    cfg, frozen, _ = calibration._freeze(config, freeze, root)
    for reference in frozen['artifacts'].values():
        core.check_artifact(root, reference)
    runtime = root/'code/runtime'
    actual = {p.name: core.file_sha(p) for p in runtime.glob('*.py') if p.is_file()}
    core.require(actual == frozen['runtime_sources'] and
                 set(actual.values()) <= {r['sha256'] for r in frozen['artifacts'].values()},
                 'Frozen runtime source inventory differs')
    # Match the subprocess's first import search path; no GPU modules imported.
    spec = importlib.machinery.PathFinder.find_spec('runtime.dry_driver', [str(runtime)])
    core.require(spec is not None and Path(spec.origin).resolve() == runtime/'dry_driver.py',
                 'Frozen driver import path differs')
    for rel in ('supplied_package/tools/satdose_core.py', 'supplied_package/config/source_bindings.json'):
        core.require(core.file_sha(PACKAGE/rel) == core.file_sha(root/'code'/rel),
                     'Release helper/scorer pins differ from frozen deployment')
    lock_path = core.check_artifact(root, frozen['artifacts']['source_lock'])
    lock = core.read_json(lock_path)
    for field, inventory in (('native_src', 'native_files'), ('legacy_root', 'legacy_files')):
        source = Path(lock[field]).resolve(strict=True)
        if field == 'native_src':
            actual_names = {str(p.relative_to(source)) for p in source.rglob('*.py') if p.is_file()}
            core.require(actual_names == set(lock[inventory]), 'Native source inventory changed')
        for rel, sha in lock[inventory].items():
            core.check_artifact(source, dict(path=rel, sha256=sha))
    return dict(config=config, freeze=freeze, native_src=Path(lock['native_src']),
                source_lock_sha256=core.file_sha(lock_path), runtime_sources=actual)


def prefix_attempts(root):
    """Inspect only names and intent/allocation metadata, never partial raw."""
    found = set()
    allocations = root/'calibration_allocations'
    runs = root/'runs'
    for directory in (allocations, runs):
        core.require(not directory.is_symlink(), 'Attempt directory cannot be a symlink')
    if allocations.exists():
        for path in allocations.iterdir():
            if path.name.startswith('PREFIX'):
                found.add(str(path.relative_to(root)))
            elif path.suffix == '.json' and core.read_json(path).get('scope') == 'PREFIX_CALIB':
                found.add(str(path.relative_to(root)))
    if runs.exists():
        for path in runs.iterdir():
            if path.name.startswith('PREFIX'):
                found.add(str(path.relative_to(root)))
            elif path.is_dir() and (path/'INTENT.json').exists():
                if core.read_json(path/'INTENT.json').get('scope') == 'PREFIX_CALIB':
                    found.add(str(path.relative_to(root)))
    return sorted(found)


def protocol_identity(config, freeze):
    path = BASE/'protocol_clarifications/PREFIX_PROTOCOL_CLARIFICATION_20261113.json'
    core.require(core.file_sha(path) == CLARIFICATION_SHA, 'Recorded clarification changed')
    identity = calibration.prefix_protocol_identity(config, freeze, BASE, path)
    core.require(identity is not None and identity['pre_prefix_confirmation']['sha256'] == PROOF_SHA,
                 'Recorded pre-PREFIX confirmation changed')
    return path, identity


def _worker_module():
    from . import diagnostic_worker as worker
    core.require(Path(worker.__file__).resolve() == PACKAGE/'execution/diagnostic_worker.py',
                 'Shadowed shared-lock/GPU slot validator import')
    return worker


def check_gpu_slots(root, *, current_prefix=False):
    """Permit absent or verified complete GPU slots; never bypass HOLD/UNKNOWN.

    A technical binding is not required just to run PREFIX. If one exists,
    validate its real identity and reuse the worker's complete-chain validator.
    This reads metadata/evidence hashes only and never constructs a model.
    """
    occupied = False
    for name in ('shadow_inputs', 'shadow_claims', 'backend_equivalence'):
        directory = root/name
        core.require(not directory.is_symlink() and (not directory.exists() or directory.is_dir()),
                     'Invalid GPU stage directory requires read-only audit')
        if directory.exists():
            for start in directory.iterdir():
                occupied = True
                core.require(start.name in ('start-11031', 'start-11032') and start.is_dir() and not start.is_symlink(),
                             'Unknown GPU start slot requires read-only audit')
                if name != 'backend_equivalence':
                    for trial in start.iterdir():
                        core.require(trial.name in ('trial-0', 'trial-1') and trial.is_dir() and not trial.is_symlink(),
                                     'Unknown GPU trial slot requires read-only audit')
    path = root/'TECHNICAL_EXECUTION_BINDING.json'
    core.require(not path.is_symlink(), 'Technical binding symlink forbidden')
    if not path.exists():
        core.require(not occupied, 'Occupied GPU slot without binding requires read-only audit')
        if current_prefix:
            _worker_module()._prefix_gate(root, dict(design_freeze=dict(sha256=FREEZE_SHA),
                config=dict(sha256=CONFIG_SHA)), current_prefix=True)
        return None
    binding_sha = core.file_sha(path)
    binding = core.read_json(path)
    core.require(binding.get('schema') == 'sr-satdose-technical-execution-v1' and
                 binding.get('mode') == 'production' and binding.get('experiment_id') == 'SR-SATDOSE-1-20261112' and
                 binding.get('experiment_root') == str(root), 'GPU technical binding identity mismatch')
    freeze = core.read_json(root/'DESIGN_FREEZE.json')
    core.require(binding.get('design_freeze') == dict(path='DESIGN_FREEZE.json', sha256=FREEZE_SHA),
                 'GPU technical binding freeze mismatch')
    for key in ('config', 'source_lock'):
        core.require(binding.get(key) == freeze['artifacts'][key], 'GPU technical binding substituted '+key)
        core.check_artifact(root, binding[key])
    expected_slots = {f'{start}/{trial}': {
        'input_receipt': f'shadow_inputs/start-{start}/trial-{trial}/INPUT_RECEIPT.json',
        'restore': f'shadow_inputs/start-{start}/trial-{trial}/RESTORE.json',
        'capture': f'shadow_inputs/start-{start}/trial-{trial}/PRODUCTION_CAPTURE.pt'}
        for start in STARTS for trial in (0, 1)}
    core.require(binding.get('shadow_claims_root') == 'shadow_claims' and
                 binding.get('shadow_slots') == expected_slots and binding.get('max_shadow_optimizer_steps') == 4 and
                 binding.get('steps_per_slot') == 1, 'GPU technical binding fixed slots/budget differ')
    # This code lives in a distinct reviewed release; verify the previously
    # bound source paths AND byte equality of the validator we actually call.
    for reference in binding['code'].values():
        source = Path(reference['path'])
        core.require(source.is_absolute() and source.resolve(strict=True) == source and not source.is_symlink() and
                     core.file_sha(source) == reference['sha256'], 'Bound GPU source changed')
    worker = _worker_module()
    core.require(core.file_sha(worker.__file__) == binding['code']['diagnostic_worker']['sha256'],
                 'GPU slot validator differs from bound worker')
    worker._no_holds(root, binding, binding_sha=binding_sha, current_prefix=current_prefix)
    core.require(core.file_sha(path) == binding_sha, 'GPU technical binding changed during validation')
    return dict(path=str(path.relative_to(root)), sha256=binding_sha)


def preflight(release_receipt, release_sha):
    """No files written and no model/subprocess launched by this function."""
    release = verify_release(release_receipt, release_sha)
    deployed = verify_deployment()
    config, freeze = deployed['config'], deployed['freeze']
    clarification, protocol = protocol_identity(config, freeze)
    gpu_binding = check_gpu_slots(BASE)
    core.require(not prefix_attempts(BASE) and not (BASE/'prefix_control').exists(),
                 'Existing PREFIX attempt/control: audit required; no automatic resume/retry')
    receipts = {start: BASE/f'runs/DRY_{start}/CALIBRATION_COMPLETE.json' for start in STARTS}
    # Check BOTH completion headers before either analyze_dry can open raw.
    for start, path in receipts.items():
        receipt = core.read_json(path)
        core.require(receipt.get('status') == 'CALIBRATION_CAPTURE_COMPLETE_NOT_GPU_ADMISSION' and
                     receipt.get('scope') == 'DRY' and receipt.get('start_seed') == start and
                     type(receipt.get('responses')) is int and receipt['responses'] == 640,
                     'Both DRY starts must be complete before PREFIX')
    reports = {str(start): calibration.analyze_dry(config, freeze, BASE, receipts[start], start,
                                                  deployed['native_src']) for start in STARTS}
    core.require(all(r['status'] == 'DRY_METRICS_PASS_GPU_CHECK_STILL_REQUIRED' and
                     not r['failures'] and all(c['passed'] for c in r['checks'].values())
                     for r in reports.values()), 'Both complete DRY gates must pass before PREFIX')
    return dict(schema='sr-satdose-prefix-preflight-v1', status='READY_FOR_PREFIX_CALIB_ONLY',
                permission='NO_TRAINING', model_calls=0, design_freeze_sha256=FREEZE_SHA,
                config_sha256=CONFIG_SHA, release=release, protocol_clarification=protocol,
                source_lock_sha256=deployed['source_lock_sha256'], runtime_sources=deployed['runtime_sources'],
                dry_analyses=reports, starts=list(STARTS), responses_per_start=48,
                clarification=str(clarification), existing_gpu_binding=gpu_binding)


def command(start):
    core.require(start in STARTS, 'Unregistered PREFIX start')
    return ['env', '-u', 'SLURM_TRES_PER_TASK', 'srun', '--exclusive', '--exact', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=8', '--gpus-per-task=pro6000:2', '--mem=66G', PYTHON, '-B', '-m', 'runtime.dry_driver',
            '--config', str(BASE/'code/supplied_package/config/experiment.json'),
            '--evidence-root', str(BASE), '--freeze', str(BASE/'DESIGN_FREEZE.json'),
            '--inputs', str(BASE/'inputs.jsonl'), '--images', str(BASE/'images.jsonl'),
            '--source-lock', str(BASE/'runtime_source_lock.json'), '--manifests',
            *(str(BASE/(name+'_manifest.jsonl')) for name in MANIFESTS),
            '--start-seed', str(start), '--scope', 'PREFIX_CALIB', '--output', str(BASE/f'runs/PREFIX_CALIB_{start}')]


def run_sequence(preflight_report, launch, analyze, write):
    """Sequencing seam for CPU tests; callbacks cannot themselves certify GPU PASS."""
    reports = {}
    for start in STARTS:
        code = launch(start)
        if code != 0:
            return dict(status='HOLD_PREFIX_CAPTURE', permission='NO_TRAINING', stopped_start=start,
                        subprocess_returncode=code, prefix_analyses=reports)
        report = analyze(start)
        core.require(report.get('permission') == 'NO_TRAINING', 'Unexpected training permission')
        write(start, report)
        reports[str(start)] = report
        if report.get('status') != 'PREFIX_METRICS_PASS_GPU_CHECK_STILL_REQUIRED':
            return dict(status='HOLD_PREFIX_MEASUREMENT', permission='NO_TRAINING', stopped_start=start,
                        prefix_analyses=reports)
    return dict(status='PREFIX_METRICS_COMPLETE_GPU_CHECK_STILL_REQUIRED', permission='NO_TRAINING',
                responses=96, prefix_analyses=reports,
                preflight_sha256=core.object_sha(preflight_report))


def execute(release_receipt, release_sha):
    # Import the shared lock only after verifying the complete audited release.
    # Take it before checking stage state: another worker may have finished or
    # failed since a previous read-only preflight. Hold through RESULT writing.
    verify_release(release_receipt, release_sha)
    with _worker_module()._worker_lock(BASE):
        return _execute_locked(release_receipt, release_sha)


def _execute_locked(release_receipt, release_sha):
    core.require(re.fullmatch('[0-9]+', os.environ.get('SLURM_JOB_ID', '')) is not None and
                 os.environ.get('SLURM_NTASKS') == '1' and os.environ.get('SLURM_NNODES') == '1',
                 'Expected one-task one-node Slurm allocation')
    report = preflight(release_receipt, release_sha)
    control = BASE/'prefix_control'
    # Exclusive directory is a durable single-use claim, even if interrupted.
    control.mkdir(exist_ok=False)
    calibration.write_once(control/'PREFLIGHT.json', report)
    env = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTORCH_CUDA_ALLOC_CONF'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    def launch(start):
        verify_release(release_receipt, release_sha)
        verify_deployment()
        protocol_identity(BASE/'code/supplied_package/config/experiment.json', BASE/'DESIGN_FREEZE.json')
        check_gpu_slots(BASE, current_prefix=True)
        for dry_report in report['dry_analyses'].values():
            for name in ('raw', 'completion_receipt'):
                core.check_artifact(BASE, dry_report[name])
        allowed = set() if start == STARTS[0] else {
            'calibration_allocations/PREFIX_CALIB-11031.json', 'runs/PREFIX_CALIB_11031'}
        core.require(set(prefix_attempts(BASE)) <= allowed, 'Unexpected concurrent PREFIX attempt')
        return subprocess.run(command(start), cwd=BASE/'code', env=env, check=False).returncode
    def analyze(start):
        return calibration.analyze_prefix(BASE/'code/supplied_package/config/experiment.json',
            BASE/'DESIGN_FREEZE.json', BASE, BASE/f'runs/PREFIX_CALIB_{start}/CALIBRATION_COMPLETE.json',
            start, clarification=Path(report['clarification']))
    try:
        result = run_sequence(report, launch, analyze,
                              lambda start, value: calibration.write_once(control/f'PREFIX_{start}_ANALYSIS.json', value))
    except Exception as exc:
        result = dict(status='HOLD_PREFIX_CONTROL', permission='NO_TRAINING', error=f'{type(exc).__name__}: {exc}')
    calibration.write_once(control/'RESULT.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-receipt', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--execute', action='store_true', help='Run exactly the two sequential PREFIX stages; default read-only preflight')
    args = parser.parse_args(argv)
    try:
        result = (execute if args.execute else preflight)(args.release_receipt, args.release_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 2 if result['status'].startswith('HOLD') else 0
    except Exception as exc:
        print(json.dumps(dict(status='HOLD_PREFIX_CONTROL', permission='NO_TRAINING', error=f'{type(exc).__name__}: {exc}')))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
