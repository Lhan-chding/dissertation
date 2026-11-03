"""Freeze the independent implementation and verified server inputs before model loading."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from adapter.backend import file_sha, object_sha, read_json, write_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    if len(args.code_commit) != 40 or any(c not in '0123456789abcdef' for c in args.code_commit):
        raise ValueError('Actual Git commit required')
    evidence = read_json(PACKAGE / 'execution/NATIVE_BINDING.json')
    registration = read_json(PACKAGE / 'execution/registration/INDEX.json')
    history = read_json(evidence['source_binding']['path'])
    config = read_json(evidence['config']['path'])
    if evidence['status'] != 'CPU_SERVER_BINDING_PASS_NOT_GPU':
        raise ValueError('Native binding incomplete')
    for name in ('source_binding', 'config', 'pilot_spec', 'overlay'):
        if file_sha(evidence[name]['path']) != evidence[name]['sha256']:
            raise ValueError('Historical evidence changed: ' + name)
    retention = read_json(PACKAGE / 'run/bound_assets/LEGACY_BINDING.json')
    if retention['status'] != 'ORIGINAL_DEV_EVAL_P0_V3_BOUND' or retention['rows'] != 96:
        raise ValueError('Original retention inputs not bound')
    implementation = [dict(path=str(p), sha256=file_sha(p)) for p in sorted((PACKAGE/'adapter').glob('*.py'))]
    manifest = PACKAGE / 'execution/registration/ALL_PRE_EVAL.jsonl'
    if file_sha(manifest) != registration['requests_sha256']:
        raise ValueError('Request registration changed')
    binding = dict(schema='sr-ord-train-binding-v1', experiment_id='SR-ORD-TRAIN-1',
        code_commit=args.code_commit, package_root=str(PACKAGE), inputs_root=str(PACKAGE),
        manifest_sha256=file_sha(PACKAGE/'frozen/FREEZE.json'),
        native_code_root=str(Path(evidence['source_files'][0]['path']).parents[2]),
        historical_source_binding=evidence['source_binding']['path'],
        historical_source_binding_sha256=evidence['source_binding']['sha256'],
        source_files=evidence['source_files'], source_sha256=object_sha(evidence['source_files']),
        implementation_files=implementation,
        model=dict(snapshot='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                   model_path=config['model']['model_path'],
                   parents={p:history['parents'][p]['path'] for p in ('b0','b1')}),
        requests_manifest=dict(path=str(manifest), sha256=file_sha(manifest)),
        recipe=evidence['recipe'],
        native_binding=dict(path=str(PACKAGE/'execution/NATIVE_BINDING.json'),sha256=file_sha(PACKAGE/'execution/NATIVE_BINDING.json')),
        legacy_binding=dict(path=str(PACKAGE/'run/bound_assets/LEGACY_BINDING.json'),sha256=file_sha(PACKAGE/'run/bound_assets/LEGACY_BINDING.json')),
        resource_policy=dict(account='rose', partition='cluster02', qos='soujanya-poria-startfund-2026-03',
            job_name_prefix='LH-', independent_workers_per_job=2, gpus_per_worker=1,
            gpu_hour_stopping=False, submitted_jobs_limit=5, physical_gpu_limit=8),
        recovery_policy='Same path and last committed checkpoint only; UNKNOWN calls block; no scientific resampling',
        status='PRE_MODEL_BINDING_NOT_GPU_VALIDATED', model_calls=0, optimizer_updates=0)
    # Every source is beneath this fixed documented deployment; do not infer from an arbitrary filename depth.
    binding['native_code_root'] = '/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_b7e7b06'
    write_once(PACKAGE/'execution/RUN_BINDING.json', binding)
    print(binding['status'])


if __name__ == '__main__':
    main()
