"""Register only pre-evaluation requests without loading a model or evaluation manifest.

The released bytes remain immutable. Re-running against an existing destination
checks exact file membership and SHA256 identity, never repairs or replaces it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

COUNTS = {'DEV_SMOKE': 192, 'CLARIFICATION': 384, 'TRAIN': 36864}
PUBLIC_FIELDS = ('prompt', 'assistant_prefill', 'image_file', 'image_sha256')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def strict_json(data):
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), 'duplicate JSON keys')
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def rows(path):
    with Path(path).open('rb') as handle:
        return [strict_json(line) for line in handle]


def indexed(records, key):
    result = {row[key]: row for row in records}
    require(len(result) == len(records), 'duplicate ' + key)
    return result


def model_payload(request):
    """Explicit allowlist: private/control-plane identity never enters the model."""
    return {key: request[key] for key in PUBLIC_FIELDS}


class Registration:
    def __init__(self, package):
        self.package = Path(package).resolve(strict=True)
        self.frozen = self.package / 'frozen'
        self.source_hashes = {}
        release = strict_json((self.package / 'RELEASE.json').read_bytes())['members']
        self.release = release
        self.verify('frozen/FREEZE.json', release['frozen/FREEZE.json'])
        self.freeze = strict_json((self.frozen / 'FREEZE.json').read_bytes())
        self.freeze_sha256 = digest(self.frozen / 'FREEZE.json')
        names = ('inputs', 'images', 'private_cases', 'smoke_groups',
                 'training_groups', 'clarification_manifest')
        for name in names:
            relative = f'frozen/{name}.jsonl'
            expected = self.freeze['files'][name + '.jsonl']
            require(release[relative] == expected, 'release/freeze disagreement: ' + name)
            self.verify(relative, expected)
        index_path = 'frozen/receipt_plans/INDEX.json'
        self.verify(index_path, release[index_path])
        self.plan_index = strict_json((self.package / index_path).read_bytes())
        require(self.plan_index['scientific_freeze_sha256'] == self.freeze_sha256,
                'plan index binds another freeze')
        for name, expected in self.plan_index['files'].items():
            relative = 'frozen/receipt_plans/' + name
            require(release.get(relative) == expected, 'unregistered plan ' + name)
            self.verify(relative, expected)
        self.inputs = indexed(rows(self.frozen / 'inputs.jsonl'), 'input_id')
        self.images = indexed(rows(self.frozen / 'images.jsonl'), 'image_file')
        # Retain identities only, never world/gold or diagnostic fields in requests.
        self.cases = {r['case_id']: {'input_id': r['input_id'],
                      'qid': r['task']['qid'], 'root_id': r['task']['root_id']}
                      for r in rows(self.frozen / 'private_cases.jsonl')}
        self.checked_images = set()

    def verify(self, relative, expected):
        rel = PurePosixPath(relative)
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe source path')
        path = self.package / relative
        require(path.is_file() and not path.is_symlink() and
                path.resolve().is_relative_to(self.package), 'unsafe source ' + relative)
        require(digest(path) == expected, 'source hash mismatch: ' + relative)
        self.source_hashes[relative] = expected

    def request(self, identity):
        case = self.cases[identity['case_id']]
        require(all(case[k] == identity[k] for k in ('input_id', 'qid', 'root_id')),
                'case identity mismatch')
        public = self.inputs[identity['input_id']]
        image_file = public['image_file']
        image = self.images[image_file]
        require(not image['legacy_original_required'] and image_file.startswith('images/'),
                'pre-evaluation request needs unbound legacy image')
        image_relative = 'frozen/' + image_file
        if image_file not in self.checked_images:
            self.verify(image_relative, image['png_sha256'])
            self.checked_images.add(image_file)
        prompt, prefill = public['prompt'], public['assistant_prefill']
        require(type(prompt) is str and type(prefill) is str, 'nontext model input')
        return {**identity, 'prompt': prompt, 'assistant_prefill': prefill,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'prefill_sha256': hashlib.sha256(prefill.encode()).hexdigest(),
                'image_file': image_relative, 'image_sha256': image['png_sha256']}

    def group_requests(self, group, scope, **extra):
        require(len(group['seeds']) == len(group['request_ids']) == 8 and
                len(set(group['seeds'])) == 8, 'invalid G8 seeds')
        for draw, (seed, request_id) in enumerate(zip(group['seeds'], group['request_ids'])):
            identity = {k: group[k] for k in
                        ('parent', 'group_id', 'qid', 'root_id', 'case_id', 'input_id')}
            yield self.request({**identity, **extra, 'scope': scope, 'phase': scope,
                                'draw': draw, 'seed': seed, 'request_id': request_id})

    def requests(self):
        smoke = strict_json((self.frozen / 'receipt_plans/smoke_slots.json').read_bytes())
        # Same smoke case is deliberately assigned to both parents.
        smoke_source = {(r['parent'], r['group_index']): r
                        for r in rows(self.frozen / 'smoke_groups.jsonl')}
        require(len(smoke) == len(smoke_source) == 24, 'wrong smoke group count')
        for group in smoke:
            group_index = int(group['group_id'].rsplit('/g', 1)[1])
            assigned = smoke_source[(group['parent'], group_index)]
            require(all(group[k] == assigned[k] for k in
                        ('case_id', 'input_id', 'root_id', 'group_seed')), 'smoke plan mismatch')
            yield from self.group_requests(group, 'DEV_SMOKE')
        clarification = rows(self.frozen / 'clarification_manifest.jsonl')
        require(len(clarification) == 384, 'wrong clarification count')
        for row in clarification:
            require(row['parent'] in ('b0', 'b1') and row['draw'] in (0, 1),
                    'invalid clarification identity')
            group_id = f"OT1/CLARIFICATION/{row['parent']}/{row['root_id']}/{row['view']}"
            yield self.request({**row, 'scope': 'CLARIFICATION', 'phase': 'CLARIFICATION',
                                'qid': self.cases[row['case_id']]['qid'], 'group_id': group_id,
                                'request_id': f"{group_id}/d{row['draw']:02d}"})
        train = rows(self.frozen / 'training_groups.jsonl')
        keys = ('parent', 'train_seed', 'arm', 'step', 'slot')
        train_index = {tuple(r[k] for k in keys): r for r in train}
        require(len(train_index) == len(train) == 4608, 'wrong/duplicate train groups')
        for parent in ('b0', 'b1'):
            for seed in (11031, 11032):
                for arm in ('R', 'G', 'D'):
                    name = f'paths/{parent}-{seed}-{arm}.json'
                    plan = strict_json((self.frozen / 'receipt_plans' / name).read_bytes())
                    require(len(plan) == 24, 'wrong path steps')
                    for step, item in enumerate(plan, 1):
                        require(item['step'] == step and len(item['groups']) == 16,
                                'wrong step/group count')
                        for slot, group in enumerate(item['groups']):
                            assigned = train_index[(parent, seed, arm, step, slot)]
                            require(group['parent'] == parent and all(group[k] == assigned[k]
                                    for k in ('case_id', 'input_id', 'root_id', 'group_seed')),
                                    'train plan mismatch')
                            yield from self.group_requests(group, 'TRAIN', arm=arm,
                                                           train_seed=seed, step=step, slot=slot)


def register(package, out):
    registration = Registration(package)
    out = Path(out).absolute()
    require(not out.is_symlink(), 'symlink destination')
    require(not out.resolve().is_relative_to(registration.frozen.resolve()),
            'destination cannot modify frozen inputs')
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.registration-', dir=out.parent))
    try:
        counts = Counter()
        seen = set()
        requests_path = staging / 'ALL_PRE_EVAL.jsonl'
        with requests_path.open('wb') as handle:
            for row in registration.requests():
                require(row['request_id'] not in seen, 'duplicate request ID')
                seen.add(row['request_id'])
                counts[row['scope']] += 1
                handle.write(canonical(row) + b'\n')
        require(dict(counts) == COUNTS, 'incomplete preregistered budget')
        summary = {'schema': 'sr-ord-train-registration-v1', 'status': 'PRE_EVAL_REGISTERED',
                   'freeze_sha256': registration.freeze_sha256,
                   'counts': dict(counts), 'total_requests': len(seen),
                   'sources': dict(sorted(registration.source_hashes.items())),
                   'requests_manifest': 'ALL_PRE_EVAL.jsonl',
                   'requests_sha256': digest(requests_path),
                   'requests_bytes': requests_path.stat().st_size,
                   'evaluation_manifest_opened': False, 'model_calls': 0,
                   'optimizer_updates': 0}
        (staging / 'INDEX.json').write_bytes(canonical(summary) + b'\n')
        if out.exists():
            require(out.is_dir() and {p.name for p in out.iterdir()} ==
                    {'ALL_PRE_EVAL.jsonl', 'INDEX.json'}, 'existing registration files differ')
            for name in ('ALL_PRE_EVAL.jsonl', 'INDEX.json'):
                require(not (out / name).is_symlink() and digest(out / name) == digest(staging / name),
                        'existing registration bytes differ: ' + name)
        else:
            os.rename(staging, out)
        return summary
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.package, args.out), indent=2))


if __name__ == '__main__':
    main()
