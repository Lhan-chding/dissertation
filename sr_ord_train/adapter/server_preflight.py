"""Read-only historical binding; writes only into the new experiment root."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from runtime_contract import native_fixtures, PARENT_FILES

PROJECT = Path('/projects/_ssd/varunssd/louis-ssvc')
OLD = PROJECT / 'sr_f6_semantic_factorial_20261027'
NATIVE = PROJECT / 'sr_f6_code_b7e7b06'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ref(path):
    p = Path(path)
    p.resolve(strict=True).relative_to(PROJECT.resolve(strict=True))
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def checked(base, entry):
    p = base / entry['path']
    actual = ref(p)
    if actual['sha256'] != entry['sha256']:
        raise ValueError('Historical evidence changed: ' + str(p))
    return actual


def key(world, query, unnamed=False):
    world = dict(world)
    if unnamed:
        world['categories'] = [f'position_{i}' for i in range(len(world['categories']))]
    return hashlib.sha256(json.dumps({'world': world, 'query': query}, sort_keys=True,
                                    separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    expected = PROJECT / 'sr_ord_train_20261103'
    if expected.resolve(strict=True) != expected or out.is_symlink():
        raise PermissionError('Noncanonical new root')
    out.resolve().relative_to(expected)
    if out.exists():
        raise FileExistsError('Preserve existing preflight receipt')
    native = native_fixtures(NATIVE / 'src', historical_kernel=True)
    source = read(OLD / 'SOURCE_BINDING.json')
    config = read(OLD / 'CONFIG.json')
    if sha(OLD / 'CONFIG.json') != source['config_file_sha256']:
        raise ValueError('Historical config fingerprint differs')
    expected_recipe = dict(lr=1e-5, betas=[.9, .999], eps=1e-8, weight_decay=0.,
        lr_schedule='constant', gradient_clip_norm=1., kl_coefficient=.02,
        tis_cap=2., clip_epsilon=.2, microbatch_sequences=1,
        optimizer_steps_per_rollout_batch=1, temperature=1., top_p=1., top_k=0,
        max_new_tokens=1536)
    for name, value in expected_recipe.items():
        if config['training'].get(name) != value:
            raise ValueError('Historical recipe conflict: ' + name)
    pilot = OLD / 'PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json'
    spec = read(pilot)
    overlay_ref = checked(OLD, spec['input_overlay'])
    overlay = read(overlay_ref['path'])
    retained = {name: checked(OLD, overlay['splits']['DEV_EVAL'][name])
                for name in ('private', 'public')}
    updates = []
    for parent in ('b0', 'b1'):
        root = pilot.parent / 'runs' / parent / 'r0/U10/checkpoints/step-001'
        commit, update = read(root / 'COMMIT.json'), read(root / 'UPDATE.json')
        if sha(root / 'UPDATE.json') != commit['update_sha256']:
            raise ValueError('Historical update/commit mismatch')
        if (commit['step'] != 1 or update['optimizer_updates'] != 1
                or update['sequence_count'] != 128 or update['microbatch_size'] != 1
                or update['kl_coefficient'] != .02 or update['aux'] != 0
                or update['policy_after_sha256'] != commit['policy_sha256']):
            raise ValueError('Actual historical J-only update differs')
        files = source['parents'][parent]['files']
        actual = {Path(f['path']).name: checked(Path('/'), f) for f in files}
        for name, digest in zip(('adapter_config.json', 'adapter_model.safetensors'), PARENT_FILES[parent]):
            if actual[name]['sha256'] != digest:
                raise ValueError('Parent differs from package')
        updates.append({'parent': parent, 'commit': ref(root / 'COMMIT.json'),
                        'update': ref(root / 'UPDATE.json'), 'parent_files': actual,
                        'historical_policy_after_sha256': commit['policy_sha256']})
    model_files = [checked(Path('/'), f) for f in source['model_files']]
    actual_paths = {str(p) for p in Path(config['model']['model_path']).rglob('*') if p.is_file()}
    if actual_paths != {f['path'] for f in model_files}:
        raise ValueError('Base model inventory changed')
    histories = [OLD / f'DATA/{split}/TASKS_PRIVATE.jsonl'
                 for split in ('TRAIN', 'DEV_CAL', 'DEV_PILOT', 'DEV_EVAL')]
    histories += [PROJECT / f'sr_f6_interface_v3_20261028/INPUT_PLAN/{split}/TASKS_PRIVATE.jsonl'
                  for split in ('STAGE1', 'STAGE2')]
    named, positional, historical_refs = set(), set(), []
    for p in histories:
        data = rows(p)
        for row in data:
            named.add(key(row['world'], row['query']))
            positional.add(key(row['world'], row['query'], True))
        historical_refs.append(dict(ref(p), rows=len(data)))
    matches, retained_matches = [], []
    cases = rows(PACKAGE / 'frozen/private_cases.jsonl')
    for case in cases:
        task = case['task']
        match = key(task['world'], task['query']) in named or key(task['world'], task['query'], True) in positional
        if match:
            target = retained_matches if task['qid'].startswith('DEV_EVAL_') else matches
            target.append({'case_id': case['case_id'], 'qid': task['qid']})
    if matches:
        raise ValueError('New inputs overlap historical open data: ' + json.dumps(matches))
    result = dict(status='CPU_SERVER_BINDING_PASS_NOT_GPU', experiment_id='SR-ORD-TRAIN-1',
        native=native, source_binding=ref(OLD / 'SOURCE_BINDING.json'), config=ref(OLD / 'CONFIG.json'),
        pilot_spec=ref(pilot), overlay=overlay_ref, retention=retained,
        historical_updates=updates, recipe=expected_recipe, model_files=model_files,
        source_files=[ref(p) for p in sorted((NATIVE / 'src').rglob('*.py'))],
        history_exclusion=dict(sources=historical_refs, new_matches=matches,
            intended_retention_reuse=len(retained_matches),
            unverified=['Additional old C diagnostics not available in this explicit source list'],
            sealed_test_read=False), model_calls=0, optimizer_updates=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'source_files': len(result['source_files']),
                      'model_files': len(model_files), 'retention_reuse': len(retained_matches)}))


if __name__ == '__main__':
    main()
