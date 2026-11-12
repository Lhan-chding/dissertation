"""Offline SATDOSE inventory. No model imports, inference, scoring or scheduler calls.

A preparation is NOT DESIGN_FREEZE or model admission. Final visual inspection,
source provenance and the independent verifier must be completed by the caller.
Legacy scientific constructors/rendering are imported read-only with SHA pins.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import itertools
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

MASTER = 2026111201
STRUCTURES = ('2_choose_1', '3_choose_1', '3_choose_2', '4_choose_2')
CELLS = tuple(itertools.product(STRUCTURES, ('largest', 'smallest'), ('sum', 'mean')))


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, values):
    Path(path).write_text(''.join(encode(v) + '\n' for v in values))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def world_key(task):
    world = copy.deepcopy(task['world'])
    world['categories'] = list(range(len(world['categories'])))
    return digest(dict(world=world, query=task['query']))


def fact_key(task):
    # Matches the historical exclusion inventory; stronger than the new ordered key.
    w = task['world']
    return hashlib.sha256(json.dumps(sorted(zip(w['series']['Alpha'], w['series']['Beta'])), separators=(',', ':')).encode()).hexdigest()


def source_inventory(root):
    root = Path(root)
    paths = [root / 'data_design.py', root / 'background.py', root / 'build_plan.py']
    paths += sorted((root / 'vendor').rglob('*.py'))
    paths += [root / 'history' / name for name in ('probe_config.json', 'historical_sr_f6.json', 'open_world_hashes.json')]
    return {str(p.relative_to(root)): sha(p) for p in paths}


def load_legacy(root, lock):
    root = Path(root).resolve()
    actual = source_inventory(root)
    require(actual == lock, 'legacy source inventory does not match the reviewed lock')
    for path in (root / 'vendor/native_src', root / 'vendor/probe_ref', root):
        sys.path.insert(0, str(path))
    modules = {}
    # Do not write bytecode into the historical checkout.
    sys.dont_write_bytecode = True
    for name in ('data_design', 'background', 'build_plan', 'order_probe.data', 'order_probe.inputs', 'order_probe.render', 'srf6_ref.core'):
        mod = importlib.import_module(name)
        require(Path(mod.__file__).resolve().is_relative_to(root), 'foreign module import: ' + name)
        modules[name] = mod
    return modules


def conflict_context(legacy, task, seed, rejection_log):
    """Only the public list changes; cap is 1000, not legacy's 4096."""
    d = legacy['data_design']
    rng = random.Random(seed)
    names = task['world']['categories']
    for attempt in range(1000):
        order = rng.sample(names, len(names))
        if set(d.induced_choice(task, order)) != set(task['gold']['selected']):
            return d._context(copy.deepcopy(task), 'LIST', 'SATDOSE', seed, order, tries=attempt + 1)
        rejection_log.append(dict(root_id=task['root_id'], kind='list_permutation', attempt=attempt, reason='same_authorised_set', candidate_order=order))
    raise ValueError('HOLD: 1000 permutations exhausted for ' + task['root_id'])


def strip_list(prompt, marker):
    require(prompt.count(marker) == 1, 'unique list marker required')
    start = prompt.index(marker) + len(marker)
    _, length = json.JSONDecoder().raw_decode(prompt[start:])
    return prompt[:start] + '<LIST>' + prompt[start + length:]


def verify_pair(legacy, aligned, other, conflict):
    d = legacy['data_design']
    for key in ('world', 'query', 'gold'):
        require(aligned['task'][key] == other['task'][key], 'pair changed ' + key)
    require(aligned['assistant_prefill'] == other['assistant_prefill'], 'pair changed prefill')
    require(strip_list(aligned['prompt'], d.RANK_PREFIX) == strip_list(other['prompt'], d.RANK_PREFIX), 'non-list prompt difference')
    require(bool(other['audit']['list_conflict']) == conflict, 'wrong selection conflict')
    if not conflict:
        require(aligned['prompt'] == other['prompt'], 'aligned public bytes differ')


class Bank:
    def __init__(self, legacy, historical, output, font, render_images, legacy_assets):
        self.legacy, self.output, self.font = legacy, Path(output), font
        self.render_images, self.legacy_assets = render_images, Path(legacy_assets) if legacy_assets else None
        self.inputs, self.cases, self.images = {}, {}, {}
        history_manifest = json.loads((Path(historical) / 'SHA256SUMS.json').read_text())
        for rel in ('frozen/private_cases.jsonl', 'provenance/LEGACY_BINDING.json'):
            require(sha(Path(historical) / rel) == history_manifest[rel]['sha256'], 'historical input checksum mismatch')
        self.history_manifest_sha256 = sha(Path(historical) / 'SHA256SUMS.json')
        receipt = json.loads((Path(historical) / 'provenance/LEGACY_BINDING.json').read_text())
        self.legacy_image_pins = {r['relative']: r['sha256'] for r in receipt['files']}
        old = rows(Path(historical) / 'frozen/private_cases.jsonl')
        self.historical_keys = {world_key(r['task']) for r in old}
        self.historical_facts = {fact_key(r['task']) for r in old}
        root = Path(legacy['data_design'].__file__).parent
        self.historical_facts.update(json.loads((root / 'history/open_world_hashes.json').read_text()))
        self.owners, self.fact_owners = {}, {}
        self.rejections, self.oracle_checks = [], 0

    def add(self, context, split, *, historical=False):
        task = context['task']; root = task['root_id']
        key, facts = world_key(task), fact_key(task)
        if not historical:
            require(key not in self.historical_keys and facts not in self.historical_facts, 'HOLD: historical world collision ' + root)
            require(self.owners.setdefault(key, (split, root)) == (split, root), 'HOLD: cross-split/root world collision')
            require(self.fact_owners.setdefault(facts, (split, root)) == (split, root), 'HOLD: cross-split/root fact collision')
        if 'boundary_indices' in task and task.get('authority') != 'list':
            d = self.legacy['data_design']
            a, b = d.oracle_sort(task['world'], task['query']), d.oracle_enumerate(task['world'], task['query'])
            require(a == b and set(a['selected']) == set(task['gold']['selected']) and a['answer'] == task['gold']['answer'], 'dual oracle mismatch')
            beta = task['world']['series']['Beta']; r = task.get('r', task.get('required_from_boundary'))
            sums = [sum(beta[i] for i in ids) for ids in itertools.combinations(task['boundary_indices'], r)]
            require(len(sums) == len(set(sums)), 'boundary answer collision')
            self.oracle_checks += 1
        world_sha = digest(task['world'])
        image_file = 'legacy/' + task['qid'] + '.png' if historical else 'images/' + world_sha + '.png'
        if image_file not in self.images:
            target = self.output / image_file
            record = dict(image_file=image_file, world_sha256=world_sha, root_id=root, split=split, structure=task.get('structure'), direction=task['query'].get('direction'), aggregate=task['query'].get('aggregate'), boundary_indices=task.get('boundary_indices', []), png_sha256=None)
            if self.render_images:
                if historical:
                    require(self.legacy_assets is not None, 'original DEV image assets required')
                    source = self.legacy_assets / image_file
                    require(source.is_file(), 'original DEV image missing: ' + str(source))
                    require(sha(source) == self.legacy_image_pins[image_file], 'historical DEV image checksum mismatch')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    record['original_sha256'] = sha(source)
                else:
                    self.legacy['order_probe.render'].render(task, target, self.font)
                record.update(inspect_image(target, task, check_bars=not historical))
                record['png_sha256'] = sha(target)
            self.images[image_file] = record
        public = dict(image_file=image_file, prompt=context['prompt'], assistant_prefill=context['assistant_prefill'])
        iid = digest(public); self.inputs[iid] = dict(input_id=iid, **public)
        private = dict(input_id=iid, task=task, audit=context['audit'])
        cid = digest(private); self.cases[cid] = dict(case_id=cid, **private)
        return dict(input_id=iid, case_id=cid)


def inspect_image(path, task, *, check_bars=True):
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        require(im.size == (1280, 896) and im.mode == 'RGB', 'unexpected image layout')
        require(not im.info, 'image metadata must be empty')
        if check_bars:
            w = task['world']; space = 1115 / len(w['categories']); bw = space * .28
            for i in range(len(w['categories'])):
                center = 110 + space * (i + .5)
                for j, (series, color) in enumerate((('Alpha', (63, 110, 156)), ('Beta', (184, 112, 48)))):
                    x = int(center + (j - .5) * bw * 1.12)
                    top = int(755 - 630 * w['series'][series][i] / 100)
                    require(im.getpixel((x, top + 2)) == color and im.getpixel((x, 752)) == color, 'bar geometry/value mismatch')
                    require(im.getpixel((x, top - 2)) != color, 'bar height mismatch')
        return dict(width=1280, height=896, image_structure_pass=True, bars_checked=check_bars, metadata_empty=True)


def construct(package, legacy_root, source_lock, historical, output, *, font=None, render_images=False, legacy_assets=None):
    package, output = Path(package), Path(output)
    require(not output.exists(), 'refuse to overwrite a preparation')
    lock = json.loads(Path(source_lock).read_text())
    legacy = load_legacy(legacy_root, lock)
    d = legacy['data_design']; core = legacy['srf6_ref.core']
    cfg = json.loads((package / 'config/experiment.json').read_text())
    slots = rows(package / 'config/training_slots.jsonl')
    output.mkdir(parents=True)
    bank = Bank(legacy, historical, output, font, render_images, legacy_assets)
    seed = lambda *parts: core.seed_for(MASTER, *parts)
    reqs = {phase: [] for phase in ('TRAIN', 'DRY', 'DEV', 'PREFIX_CALIB', 'MAIN')}
    train, dry = [], []
    def task8(split, index, structure, direction, aggregate, world_seed=None, root_id=None):
        task = d.make_world('SATDOSE_' + split, index, 8, structure, master_seed=world_seed if world_seed is not None else MASTER, direction=direction, aggregate=aggregate)
        if root_id:
            task['root_id'], task['qid'] = root_id, root_id + '_base'
        for attempt in range(task['structural_attempt']):
            bank.rejections.append(dict(root_id=task['root_id'], kind='native_beta_sampling', attempt=attempt, reason='boundary_aggregate_collision'))
        return task
    def context(task, conflict=False, perm_seed=0):
        return conflict_context(legacy, task, perm_seed, bank.rejections) if conflict else d.build_context(task, 'BASE', 'R', perm_seed)
    def request(phase, identity, ids, root, draw, draw_seed, **extra):
        reqs[phase].append(dict(request_id=identity, split=phase, root_id=root, draw=draw, seed=draw_seed, **ids, **extra))
    tasks = {}
    aligned = {}
    for slot in slots:
        rid = slot['root_id']
        if rid not in tasks:
            tasks[rid] = task8('TRAIN', len(tasks), slot['structure'], slot['direction'], slot['aggregate'], slot['world_seed'], rid)
            aligned[rid] = context(tasks[rid])
        ctx = context(tasks[rid], slot['planned_conflict'], slot['permutation_seed'])
        verify_pair(legacy, aligned[rid], ctx, slot['planned_conflict'])
        ids = bank.add(ctx, 'TRAIN'); train.append(dict(slot, **ids))
        for draw, (request_id, ds) in enumerate(zip(slot['request_ids'], slot['draw_seeds'])):
            request('TRAIN', request_id, ids, rid, draw, ds, group_id=slot['group_id'], start_seed=slot['start_seed'], arm=slot['arm'], step=slot['step'])
    dry_cases = []
    for kind, repetitions in (('CONFLICT', 4), ('ALIGNED', 1)):
        for repeat in range(repetitions):
            for cell, (st, direction, aggregate) in enumerate(CELLS):
                index = len(dry_cases); t = task8('DRY', index, st, direction, aggregate)
                ids = bank.add(context(t, kind == 'CONFLICT', seed('dry-list', index)), 'DRY')
                dry_cases.append(dict(kind=kind, repeat=repeat, cell=cell, task=t, ids=ids))
                for start in cfg['starts']:
                    gid = f'SAT1/DRY/{start}/g{index:03d}'
                    requests = [gid + f'/d{i:02d}' for i in range(8)]
                    seeds = [seed('dry-draw', index, i) for i in range(8)]
                    dry.append(dict(group_id=gid, start_seed=start, kind=kind, root_id=t['root_id'], structure=st, **ids, request_ids=requests, draw_seeds=seeds))
                    for draw, (ri, ds) in enumerate(zip(requests, seeds)):
                        request('DRY', ri, ids, t['root_id'], draw, ds, group_id=gid, start_seed=start, arm='START', step=0)
    # Freeze group halves, never select according to generated outcomes.
    probe_idx = [i for i in range(16) if i % 4 != i // 4] + [64 + 4 * i + i for i in range(4)]
    witness_idx = list(range(16, 32))
    probe = dict(probe_root_ids=[dry_cases[i]['task']['root_id'] for i in probe_idx], witness_root_ids=[dry_cases[i]['task']['root_id'] for i in witness_idx])
    for key in ('probe', 'witness'):
        probe[key + '_halves'] = [probe[key + '_root_ids'][parity::2] for parity in (0, 1)]
    plans = []; devplans = []; prefixplans = []
    def add_plan(target, phase, panel, view, ctx, draws=1, historical=False):
        ids = bank.add(ctx, phase, historical=historical); t = ctx['task']
        for draw in range(draws):
            target.append(dict(panel=panel, view=view, draw=draw, root_id=t['root_id'], structure=t['structure'], **ids, seed=core.seed_for(2026110301, 'eval', 'b0', panel, t['root_id'], draw)))
    for phase, count, target in (('DEV', 32, devplans), ('MAIN', 128, plans)):
        for i in range(count):
            st, dr, ag = CELLS[i % 16]; t = task8(phase + '_LIST', i, st, dr, ag)
            for view, ctx in [('aligned', context(t)), ('conflict', context(t, True, seed(phase, 'list', i))), ('no_list', d.build_context(t, 'NO_LIST', 'R', 0))]:
                add_plan(target, phase, 'list8', view, ctx, 1 if phase == 'DEV' else 2)
    for panel, count in (('swap8', 32), ('authority8', 96)):
        for i in range(count):
            st, dr, ag = CELLS[i % 16]; t = task8('MAIN_' + panel, i, st, dr, ag)
            contexts = d.swap_four(t, seed(panel, i)) if panel == 'swap8' else legacy['build_plan'].authority(t)
            for view, ctx in contexts.items():
                if panel == 'swap8':
                    view = 'p' + view[1] + 'l' + view[-1]
                add_plan(plans, 'MAIN', panel, view, ctx, 1 if panel == 'swap8' else 2)
    olddata, oldinput = legacy['order_probe.data'], legacy['order_probe.inputs']
    pcfg = json.loads((Path(legacy_root) / 'history/probe_config.json').read_text())
    pcfg['data']['master_seed'] = MASTER
    from sr_f6.interface_v3_protocol import format_contract
    for phase, count, target in (('PREFIX_CALIB', 16, prefixplans), ('MAIN', 96, plans)):
        for i in range(count):
            st, dr, ag = CELLS[i % 16]; m, r = map(int, st.split('_choose_'))
            rr = olddata.build_root(pcfg, 'SATDOSE_' + phase + '_PREFIX', i, m, r, dr, ag, 'unselected_matched_size')
            olddata.validate_root(rr)
            relation = ('ALIGNED', 'CONFLICT')[(i // 16 + (i // 4) + (i // 2) + i) % 2]
            t = next(v for v in rr['views'] if v['rename'] == relation); oldinput.validate_prefix_triplet(t)
            for attempt in range(rr['attempt']):
                bank.rejections.append(dict(root_id=t['root_id'], kind='native_prefix_sampling', attempt=attempt, reason='boundary_aggregate_collision'))
            for channel in ('P_STABLE', 'P_REVTIE', 'P_PLACEBO'):
                ctx = dict(task=t, prompt=oldinput.prompt(t, channel, format_contract()), assistant_prefill=oldinput.assistant_prefill(t, channel), audit=dict(channel=channel, native_J_applicable=False, prefix_induced_selected=oldinput.prefix_choice(t, channel)))
                add_plan(target, phase, 'prefix10', channel.lower(), ctx, 1 if phase == 'PREFIX_CALIB' else 2)
    retention = list(legacy['background'].retention())
    require(len(retention) == 96 and len({t['root_id'] for t in retention}) == 24, 'historical DEV retention size')
    for t in retention:
        add_plan(plans, 'MAIN', 'retention8', 'v' + str(t['variant']), legacy['build_plan'].native_ctx(t), historical=True)
    by_family = {f: sorted([t for t in retention if t['family'] == f and t['variant'] == 0], key=lambda t: t['root_id']) for f in legacy['background'].FAMILIES}
    selected_dev = [by_family[f][i] for i in range(4) for f in legacy['background'].FAMILIES][:12]
    require(len(selected_dev) == 12, 'DEV other family tasks')
    for t in selected_dev:
        add_plan(devplans, 'DEV', 'retention8', 'v0', legacy['build_plan'].native_ctx(t), historical=True)
    def expand(phase, plans0, endpoints):
        for start, arm, step in endpoints:
            for p in plans0:
                identity = f"SAT1/{phase}/{start}/{arm}/s{step:02d}/{p['panel']}/{p['root_id']}/{p['view']}/d{p['draw']:02d}"
                reqs[phase].append(dict(p, request_id=identity, split=phase, start_seed=start, arm=arm, step=step))
    starts = [(s, 'START', 0) for s in cfg['starts']]
    expand('PREFIX_CALIB', prefixplans, starts)
    expand('DEV', devplans, starts + list(itertools.product(cfg['starts'], cfg['arms'], (2, 8, 16))))
    expand('MAIN', plans, starts + list(itertools.product(cfg['starts'], cfg['arms'], (24,))))
    expand('MAIN', [p for p in plans if p['panel'] == 'list8' and p['view'] == 'conflict'], list(itertools.product(cfg['starts'], cfg['arms'], (8, 16))))
    expected = dict(TRAIN=12288, DRY=1280, DEV=1512, PREFIX_CALIB=96, MAIN=13760)
    require({k: len(v) for k, v in reqs.items()} == expected, 'generation budget mismatch')
    allids = [r['request_id'] for values in reqs.values() for r in values]
    require(len(allids) == len(set(allids)) == 28936, 'request ID coverage')
    for phase, values in reqs.items():
        write_rows(output / (phase.lower() + '_manifest.jsonl'), values)
    for name, values in [('training_groups', train), ('calibration_groups', dry), ('private_cases', bank.cases.values()), ('inputs', bank.inputs.values()), ('images', bank.images.values()), ('structural_rejections', bank.rejections)]:
        write_rows(output / (name + '.jsonl'), values)
    (output / 'PROBE_SPLITS.json').write_text(encode(probe) + '\n')
    review = []
    for st, dr, ag in CELLS:
        candidates = sorted((x for x in bank.images.values() if x['split'] == 'TRAIN' and (x['structure'], x['direction'], x['aggregate']) == (st, dr, ag)), key=lambda x: digest([MASTER, 'visual', x['image_file']]))
        review.extend(candidates[:2])
    require(len(review) == 32, 'visual panel incomplete')
    require(set().union(*(set(r['boundary_indices']) for r in review)) == set(range(8)), 'visual panel misses boundary positions')
    write_rows(output / 'visual_review_manifest.jsonl', review)
    report = dict(status='PREPARED_NOT_DESIGN_FROZEN', model_calls=0, images_rendered=render_images, manual_visual_review='PENDING', generation_counts=expected, oracle_checks=bank.oracle_checks, unique_images=len(bank.images), historical_manifest_sha256=bank.history_manifest_sha256, historical_ordered_keys=len(bank.historical_keys), historical_fact_keys=len(bank.historical_facts), source_lock=lock, source_lock_sha256=sha(source_lock), config_sha256=sha(package / 'config/experiment.json'), training_slots_sha256=sha(package / 'config/training_slots.jsonl'), font_sha256=sha(font) if render_images else None, files={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}, choices=dict(master_seed=MASTER, historical_DEV_monitor_selection='variant0; round-robin five non-TIES families, first 12 of 20 sorted roots', prefix_renaming='parity of replicate + structure + direction + aggregate cell index', collision_policy='HOLD without replacing root', exact_rejected_beta_values='not exposed by pinned native generator; rejection ordinal/reason preserved'))
    (output / 'PREPARATION.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('package', 'legacy-root', 'source-lock', 'historical', 'out'):
        ap.add_argument('--' + arg, type=Path, required=True)
    ap.add_argument('--font'); ap.add_argument('--render-images', action='store_true'); ap.add_argument('--legacy-assets', type=Path)
    a = ap.parse_args()
    result = construct(a.package, a.legacy_root, a.source_lock, a.historical, a.out, font=a.font, render_images=a.render_images, legacy_assets=a.legacy_assets)
    print(json.dumps({k: result[k] for k in ('status', 'model_calls', 'generation_counts', 'unique_images', 'oracle_checks')}, indent=2))


if __name__ == '__main__':
    main()
