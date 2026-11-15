"""Read-only CPU evidence for the five non-gradient admission checks.

No model or optimizer is constructed. A separately supplied source SHA and the
original design freeze authenticate this verifier; output is not admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/projects/_ssd/varunssd/louis-ssvc/sr_satdose_20261112/prepared_data_v1')
FREEZE = '6c4b948c533fdd736e089bf3836378f4dbcb76ac66f1d480fa979eaca56c84e0'
BINDING = 'cf7da7f08e5cb11c5458cab1200fd5a2465a529a31d2258f6cc45cbe6efe73c5'
COUNTS = dict(TRAIN=12288, DRY=1280, DEV=1512, PREFIX_CALIB=96, MAIN=13760)


def require(ok, message):
    if not ok:
        raise ValueError('HOLD_SUPPLEMENTARY_EVIDENCE: ' + message)


class Files:
    def __init__(self, root):
        self.root, self.observed = Path(root), {}

    def sha(self, path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
                'Noncanonical file: ' + str(path))
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        value = h.hexdigest()
        require(expected is None or expected == value, 'File SHA differs: ' + str(path))
        key = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        require(key not in self.observed or self.observed[key] == value, 'Changed file: ' + key)
        self.observed[key] = value
        return value

    def read(self, path, expected=None):
        self.sha(path, expected)
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        return json.loads(path.read_text())

    def rows(self, path, expected=None):
        self.sha(path, expected)
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_disjoint(manifests, cases, images, inputs, historical_keys, historical_facts,
                   world_key, fact_key):
    """Recheck exact frozen cases, allowing only registered old DEV retention."""
    owners, fact_owners = {}, {}
    checked, retention = set(), set()
    for split, rows in manifests.items():
        for row in rows:
            require(row['split'] == split and row['case_id'] in cases, 'Case/split mismatch')
            case = cases[row['case_id']]
            task = case['task']
            require(case['input_id'] == row['input_id'] and task['root_id'] == row['root_id'],
                    'Case/request identity differs')
            key, fact = world_key(task), fact_key(task)
            if row.get('panel') == 'retention8':
                require(split in ('DEV', 'MAIN'), 'Retention outside evaluation')
                image = images[inputs[row['input_id']]['image_file']]
                require(key in historical_keys and fact in historical_facts and
                        image.get('original_sha256') == image['png_sha256'],
                        'Retention is not verified historical data')
                retention.add(row['case_id'])
                continue
            require(key not in historical_keys and fact not in historical_facts,
                    'Historical collision: ' + row['request_id'])
            owner = (split, row['root_id'])
            require(owners.setdefault(key, owner) == owner and fact_owners.setdefault(fact, owner) == owner,
                    'Cross-split/root collision: ' + row['request_id'])
            checked.add(row['case_id'])
    return dict(new_cases_checked=len(checked), retention_cases_checked=len(retention))


def audit(source_sha):
    files = Files(ROOT)
    files.sha(Path(__file__).resolve(), source_sha)
    freeze = files.read('DESIGN_FREEZE.json', FREEZE)
    binding = files.read('TECHNICAL_EXECUTION_BINDING.json', BINDING)
    for ref in freeze['artifacts'].values():
        files.sha(ref['path'], ref['sha256'])
    cfg = files.read(freeze['artifacts']['config']['path'], freeze['config_sha256'])
    lock = files.read(binding['source_lock']['path'], binding['source_lock']['sha256'])
    for rel, digest in lock['native_files'].items():
        files.sha(Path(lock['native_src']) / rel, digest)
    for rel, digest in lock['legacy_files'].items():
        files.sha(Path(lock['legacy_root']) / rel, digest)
    # Only authenticated original runtime code is imported, in a fresh process.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'code'))
    from runtime import backend, data_adapter
    require(Path(backend.__file__).resolve() == ROOT/'code/runtime/backend.py' and
            Path(data_adapter.__file__).resolve() == ROOT/'code/runtime/data_adapter.py',
            'Shadowed frozen runtime import')
    registry = backend.FrozenRegistry(ROOT/freeze['artifacts']['config']['path'], ROOT,
        ROOT/'DESIGN_FREEZE.json', [ROOT/(x.lower()+'_manifest.jsonl') for x in COUNTS],
        ROOT/'inputs.jsonl', ROOT/'images.jsonl')
    manifests = {scope: files.rows(scope.lower()+'_manifest.jsonl') for scope in COUNTS}
    require({k:len(v) for k,v in manifests.items()} == COUNTS, 'Frozen request count differs')
    inputs = {x['input_id']:x for x in files.rows('inputs.jsonl')}
    images = {x['image_file']:x for x in files.rows('images.jsonl')}
    for image in images.values():
        files.sha(image['image_file'], image['png_sha256'])
    private = files.rows('private_cases.jsonl')
    cases = {x['case_id']:x for x in private}
    require(len(cases) == len(private), 'Duplicate private case')
    for rid in registry.requests:
        request = registry.request(rid)
        require(set(request) <= backend.META | backend.PUBLIC | {'assistant_prefill'},
                'Unexpected private model request field')
        require(set({k:request[k] for k in backend.PUBLIC}) == {'prompt','image_file','image_sha256'},
                'Public projection differs')
    preparation = files.read('PREPARATION.json')
    history = ROOT.parent/'historical_results'
    history_manifest = files.read(history/'SHA256SUMS.json', preparation['historical_manifest_sha256'])
    old = files.rows(history/'frozen/private_cases.jsonl',
                     history_manifest['frozen/private_cases.jsonl']['sha256'])
    oldkeys = {data_adapter.world_key(x['task']) for x in old}
    oldfacts = {data_adapter.fact_key(x['task']) for x in old}
    historical_extra = Path(lock['legacy_root'])/'history/open_world_hashes.json'
    oldfacts.update(files.read(historical_extra, preparation['source_lock']['history/open_world_hashes.json']))
    require(len(oldkeys) == preparation['historical_ordered_keys'] and
            len(oldfacts) == preparation['historical_fact_keys'], 'Historical exclusion inventory differs')
    split_details = check_disjoint(manifests, cases, images, inputs, oldkeys, oldfacts,
                                   data_adapter.world_key, data_adapter.fact_key)
    # Execute the pre-existing nine golden cases against the actual locked scorer.
    fixture_path = Path(lock['legacy_root'])/'runtime_contract.py'
    files.sha(fixture_path, lock['legacy_files']['runtime_contract.py'])
    spec = importlib.util.spec_from_file_location('_satdose_golden_contract', fixture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = module.native_fixtures(Path(lock['native_src']))
    require(fixtures['status'] == 'HISTORICAL_NATIVE_CPU_FIXTURES_PASS', 'Native golden fixtures failed')
    starts, dry_starts = {}, {}
    for start in ('11031', '11032'):
        source = cfg['start_bindings'][start]
        files.sha(source['original_state_path'], source['state_sha256'])
        for trial in (0, 1):
            p = f'shadow_inputs/start-{start}/trial-{trial}'
            restore, close = files.read(p+'/RESTORE.json'), files.read(p+'/WORKER_COMPLETE.json')['close']
            require(restore['source_state_sha256'] == source['state_sha256'] and
                    restore['restored_from_original_state'] is True and close['closed'] is True,
                    'R24 restore/close differs')
            for obj in (restore['endpoint'], restore['actual_fork_receipt'], close):
                require(obj['policy_sha256'] == source['policy_sha256'] and
                        obj['reference_sha256'] == source['reference_sha256'], 'R24 policy/reference differs')
            require(close['optimizer_steps'] == close['parameter_updates'] == 0,
                    'Shadow producer mutated original policy')
        starts[start] = dict(policy_before_sha256=source['policy_sha256'],
            policy_after_sha256=source['policy_sha256'], reference_sha256=source['reference_sha256'],
            state_sha256=source['state_sha256'])
        dry = files.read(binding['dry_receipts'][start]['path'], binding['dry_receipts'][start]['sha256'])
        require(dry['scope'] == 'DRY' and dry['responses'] == 640 and
                dry['optimizer_created'] is False and dry['scheduler_created'] is False and
                dry['optimizer_steps'] == dry['parameter_updates'] == 0, 'DRY update/count differs')
        for obj in (dry['policy_before'], dry['policy_after']):
            require(obj['policy_sha256'] == source['policy_sha256'] and
                    obj['reference_sha256'] == source['reference_sha256'], 'DRY changed endpoint')
        dry_starts[start] = dict(responses=640, **{k:v for k,v in starts[start].items() if k!='state_sha256'})
    details = {
        'start_and_reference_hash_verified':dict(starts=starts),
        'native_scorer_golden_pass':dict(fixture_count=9, failures=0, fixtures=fixtures,
            native_source_lock_sha256=binding['source_lock']['sha256']),
        'disjoint_splits_checked':dict(request_counts=COUNTS, new_splits_disjoint=True,
            historical_exclusion_verified=True, retention_exception_only=True, **split_details),
        'gold_never_in_model_inputs':dict(total_requests=28936, model_input_fields=sorted(backend.PUBLIC),
            prefix_prefill_is_registered=True, private_metadata_absent=True),
        'dry_unchanged_policy':dict(total_responses=1280, optimizer_steps=0,
            parameter_updates=0, starts=dry_starts)}
    for path, digest in list(files.observed.items()):
        files.sha(path, digest)
    return {name:dict(schema='sr-satdose-supplementary-technical-proof-v1', check=name,
        passed=True, experiment_id=cfg['experiment_id'], config_sha256=freeze['config_sha256'],
        design_freeze_sha256=FREEZE, verifier=dict(path=str(Path(__file__).resolve()),sha256=source_sha),
        evidence=dict(files.observed), details=value) for name,value in details.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_sha256), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
