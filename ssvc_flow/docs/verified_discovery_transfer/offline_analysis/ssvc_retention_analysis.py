#!/usr/bin/env python3
"""Existing-output G retention analysis. Never generates samples or changes run data.
Run only after the owner verifies the release gate and supplies --released.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

SEED = 106069
REPLICATES = 5000
PARENTS = ('S96', 'REP96')
ARMS = {
    'S96': ('SELF_O0', 'SELF_MIX', 'SELF_SINGLE', 'GOLD_ALL', 'GOLD_MATCH_MIX', 'REPLAY_ONLY', 'R0_RESET32'),
    'REP96': ('SELF_O0', 'SELF_MIX', 'SELF_SINGLE', 'GOLD_ALL', 'R0_RESET32'),
}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index(rows, key='task_id'):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + key)
    return result


def panel(endpoint, metadata, draws):
    rows = index(endpoint['tasks'])
    if set(rows) != set(metadata):
        raise ValueError('Endpoint task coverage mismatch')
    for tid, row in rows.items():
        if row['draws'] != draws or row['family'] != metadata[tid]['family']:
            raise ValueError('Endpoint draws/family mismatch')
        if not 0 <= row['successes'] <= draws or abs(row['pX'] - row['successes'] / draws) > 1e-12:
            raise ValueError('Invalid pX/count')
    return rows


def clustered(vectors, units, labels):
    """Family-stratified paired vector bootstrap; each scene carries all units/interfaces."""
    grouped = defaultdict(lambda: defaultdict(dict))
    for family, scene, unit, vector in vectors:
        if unit in grouped[family][scene]:
            raise ValueError('Duplicate scene/unit in bootstrap')
        grouped[family][scene][unit] = vector
    expected = set(units)
    if not grouped or any(set(got) != expected for scenes in grouped.values() for got in scenes.values()):
        raise ValueError('Incomplete paired scene/unit panel')
    strata = []
    for family in sorted(grouped):
        strata.append([
            [mean(got[u][k] for u in sorted(expected)) for k in range(len(labels))]
            for scene, got in sorted(grouped[family].items())
        ])
    estimate = [mean(mean(v[k] for v in stratum) for stratum in strata) for k in range(len(labels))]
    rng = random.Random(SEED)
    draws = [[] for _ in labels]
    for _ in range(REPLICATES):
        # The same index draws are used for every metric, E/G interface and unit.
        selected = [[stratum[rng.randrange(len(stratum))] for _ in stratum] for stratum in strata]
        for k in range(len(labels)):
            draws[k].append(mean(mean(v[k] for v in stratum) for stratum in selected))
    metrics = {}
    for k, label in enumerate(labels):
        ordered = sorted(draws[k])
        metrics[label] = {'estimate': estimate[k], 'interval_95': [ordered[int(.025*(REPLICATES-1))], ordered[int(.975*(REPLICATES-1))]]}
    return {
        'metrics': metrics,
        'scene_count': sum(len(s) for s in strata), 'unit_count': len(expected),
        'families': {f: len(s) for f, s in grouped.items()}, 'units': sorted(expected),
        'bootstrap_replicates': REPLICATES, 'bootstrap_seed': SEED,
        'interval_scope': 'conditional scene uncertainty given executed checkpoints; not seed-population CI or simultaneous confidence band',
        'joint_resampling': 'all metrics, parent/repeat units and E/G interfaces of a base scene move together',
    }



def detailed_summary(artifact):
    rows = artifact['results']
    text = ['', '## 图像与duplicate的实际变化', '']
    chosen = [r for r in rows if r['group'] == 'registered_parent_repeat_average' and r['status'] == 'COMPLETE']
    for arm in ('SELF_O0', 'SELF_MIX', 'SELF_SINGLE', 'GOLD_ALL', 'GOLD_MATCH_MIX', 'REPLAY_ONLY', 'R0_RESET32'):
        for stratum in ('image64_family_equal', 'duplicate32'):
            r = next((r for r in chosen if r['arm'] == arm and r['stratum'] == stratum), None)
            if not r:
                continue
            m = r['metrics']; d = m['student_minus_parent']; ci = d['interval_95']
            text.append(f"- {arm} / {stratum}：父模型{100*m['parent_pX']['estimate']:.3f}%，学生{100*m['student_pX']['estimate']:.3f}%，变化{100*d['estimate']:+.3f} pp，95%场景区间[{100*ci[0]:.3f}, {100*ci[1]:.3f}] pp。")
    text += ['', '## 共享64场景的E/G联合配对', '', '|训练臂|符号E相对父模型 pp|图像G相对父模型 pp|图像增益减符号增益 pp|后者95%场景区间 pp|', '|---|---:|---:|---:|---|']
    for r in chosen:
        if r['stratum'] != 'shared64_joint_E_G':
            continue
        m=r['metrics']; q=m['change_in_interface_gap']; ci=q['interval_95']
        text.append(f"|{r['arm']}|{100*m['E_student_minus_parent']['estimate']:+.3f}|{100*m['G_student_minus_parent']['estimate']:+.3f}|{100*q['estimate']:+.3f}|[{100*ci[0]:.3f}, {100*ci[1]:.3f}]|")
    text += ['', '变化差为（学生G−父G）−（学生E−父E）。负值表示该面板上符号端改善更多，不能单独解释为图像退化。图像端接近/达到天花板时，须结合G本身变化阅读。所有指标使用同一批场景bootstrap索引，各边际区间不等于同时覆盖。', '', '观测为100%时，场景bootstrap可能给出[0,0]变化区间；这不含未观测回答的固定面板采样不确定性，不能认定真实失败率为0。原FINAL_ENDPOINTS逐题Wilson区间必须一并保留。']
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--released', action='store_true')
    args = ap.parse_args()
    root, out = args.snapshot.resolve(), args.output.resolve()
    if not args.released:
        raise SystemExit('Refusing access without explicit --released confirmation')
    # Verify public release receipts before loading scientific endpoints or cohorts.
    release = read(root / 'TEST_RELEASE.json')
    final_release = read(root / 'cohort/FINAL_RELEASE.json')
    if release.get('status') != 'RELEASED' or not final_release.get('all_registered_models_terminal'):
        raise ValueError('Missing valid final release')
    if release.get('matrix_digest') != final_release.get('matrix_digest'):
        raise ValueError('Release matrix digest mismatch')
    source_files = [root / 'TEST_RELEASE.json', root / 'cohort/FINAL_RELEASE.json', root / 'FINAL_ENDPOINTS.json', root / 'PARENT_ENDPOINTS.json']
    endpoints, parents = read(source_files[-2]), read(source_files[-1])
    metadata = {}
    for split in ('E_test', 'G_guard'):
        path = root / 'cohort' / split / 'tasks_public.jsonl'
        source_files.append(path)
        metadata[split] = index(lines(path))
    e, g = metadata['E_test'], metadata['G_guard']
    ebase = index(list(e.values()), 'base_instance_id')
    images = {tid: row for tid, row in g.items() if row['interface'] == 'IMAGE_CUE_FRESH'}
    duplicate = {tid: row for tid, row in g.items() if row['family'] == 'duplicate_encoding'}
    if len(e) != 384 or len(images) != 64 or len(duplicate) != 32 or set(images) & set(duplicate) or set(images) | set(duplicate) != set(g):
        raise ValueError('Registered E384/G64+32 structure mismatch')
    if {f: sum(t['family'] == f for t in images.values()) for f in ('cross_series', 'trend')} != {'cross_series': 32, 'trend': 32}:
        raise ValueError('Registered image family counts mismatch')
    if len({r['base_instance_id'] for r in images.values()}) != 64 or any(r['base_instance_id'] not in ebase for r in images.values()):
        raise ValueError('Shared E/G base-scene pairing invalid')
    strata = {
        'image64_family_equal': images,
        'image_cross32': {k: v for k, v in images.items() if v['family'] == 'cross_series'},
        'image_trend32': {k: v for k, v in images.items() if v['family'] == 'trend'},
        'duplicate32': duplicate,
    }
    pp, missing, units_by_arm = {}, [], defaultdict(list)
    for parent in PARENTS:
        for split, n in (('E_test', 8), ('G_guard', 4)):
            name = parent + ':' + split
            if name not in parents:
                missing.append({'parent_endpoint': name})
            else:
                pp[parent, split] = panel(parents[name], metadata[split], n)
    sp = {}
    for parent in PARENTS:
        for repeat in (0, 1):
            for arm in ARMS[parent]:
                unit = parent + ':' + str(repeat)
                units_by_arm[arm].append(unit)
                for split, n in (('E_test', 8), ('G_guard', 4)):
                    step = 32 if arm == 'R0_RESET32' else 256
                    name = f'eval.{parent}.{repeat}.{arm}.{split}.{step}'
                    if name not in endpoints:
                        missing.append({'endpoint': name})
                        continue
                    sp[arm, unit, split] = panel(endpoints[name], metadata[split], n)
                    semantic_path = root / 'final_semantics' / (name + '.json')
                    if not semantic_path.exists():
                        missing.append({'semantic_file': str(semantic_path.relative_to(root))})
                    else:
                        sem = index(read(semantic_path)['tasks'])
                        source_files.append(semantic_path)
                        if set(sem) != set(metadata[split]) or any(abs(sem[t]['pX']-sp[arm,unit,split][t]['pX']) > 1e-12 or sem[t]['base_instance_id'] != metadata[split][t]['base_instance_id'] for t in sem):
                            raise ValueError('Semantic endpoint mismatch: ' + name)
    results = []
    def emit(arm, group, units, stratum, vectors, labels, absent):
        item = {'arm': arm, 'group': group, 'stratum': stratum, 'classification': 'exploratory retention diagnosis', 'status': 'INCOMPLETE' if absent else 'COMPLETE'}
        if absent:
            item['missing_units'] = sorted(set(absent))
        else:
            item.update(clustered(vectors, units, labels))
        results.append(item)
    for arm, allunits in sorted(units_by_arm.items()):
        groups = {u: [u] for u in allunits}
        groups.update({parent + ':repeat_average': [u for u in allunits if u.startswith(parent + ':')] for parent in PARENTS if any(u.startswith(parent + ':') for u in allunits)})
        groups['registered_parent_repeat_average'] = allunits
        for group, units in groups.items():
            for stratum, tasks in strata.items():
                vectors, absent = [], []
                for unit in units:
                    parent = unit.split(':')[0]
                    if (arm, unit, 'G_guard') not in sp or (parent, 'G_guard') not in pp:
                        absent.append(unit)
                        continue
                    for tid, task in tasks.items():
                        post, pre = sp[arm, unit, 'G_guard'][tid]['pX'], pp[parent, 'G_guard'][tid]['pX']
                        vectors.append((task['family'], task['base_instance_id'], unit, [pre, post, post-pre]))
                emit(arm, group, units, stratum, vectors, ['parent_pX', 'student_pX', 'student_minus_parent'], absent)
            vectors, absent = [], []
            for unit in units:
                parent = unit.split(':')[0]
                if any((arm, unit, split) not in sp or (parent, split) not in pp for split in ('E_test', 'G_guard')):
                    absent.append(unit)
                    continue
                for tid, task in images.items():
                    eid = ebase[task['base_instance_id']]['task_id']
                    gp, ep = pp[parent, 'G_guard'][tid]['pX'], pp[parent, 'E_test'][eid]['pX']
                    gs, es = sp[arm,unit,'G_guard'][tid]['pX'], sp[arm,unit,'E_test'][eid]['pX']
                    vectors.append((task['family'], task['base_instance_id'], unit, [gp, ep, gs, es, gs-gp, es-ep, gs-es, gp-ep, (gs-gp)-(es-ep)]))
            emit(arm, group, units, 'shared64_joint_E_G', vectors,
                 ['parent_G_pX', 'parent_E_pX', 'student_G_pX', 'student_E_pX', 'G_student_minus_parent', 'E_student_minus_parent', 'student_G_minus_E', 'parent_G_minus_E', 'change_in_interface_gap'], absent)
    artifact = {
        'schema': 'ssvc-retention-existing-output-analysis-v1',
        'matrix_digest': release['matrix_digest'], 'scientific_completion_from_release': release.get('scientific_completion'),
        'missing': missing, 'status': 'COMPLETE' if not missing else 'PARTIAL',
        'sample_counts': {'G_image': 64, 'G_duplicate': 32, 'shared_E_G_base_scenes': 64},
        'results': results,
        'limits': ['No overall E+G score is computed.', 'G image and duplicate strata stay separate.', '95% intervals are marginal intervals from common paired draws, not simultaneous coverage.', 'Intervals condition on these two existing source lineages and executed pipeline repeats.', 'Parent E uses registered first 8 O0 draws; parent G uses first 4; student E/G use 8/4.', 'Image scene contains additional image cues: paired interface gaps are descriptive, not isolated causal modality effects.', 'No noninferiority/equivalence threshold was registered; failure to reject a decrease is not retention certification.', 'R0 uses step32; SFT uses step256; neither equal-compute claims nor new samples are introduced.', 'No additional fixed-panel MCSE is inferred from aggregate pX; primary report supplies its separately computed MCSE.'],
        'sources': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_files))},
    }
    out.mkdir(parents=True, exist_ok=True)
    dests = [out / ('retention' + s) for s in ('.json', '.csv', '_zh.md')]
    if any(p.exists() for p in dests):
        raise FileExistsError('Refusing to overwrite existing retention artifacts')
    dests[0].write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    with dests[1].open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['arm', 'group', 'stratum', 'metric', 'estimate', 'ci95_low', 'ci95_high', 'scene_count', 'unit_count', 'status'])
        writer.writeheader()
        for row in results:
            for metric, value in row.get('metrics', {}).items():
                writer.writerow({k: row[k] for k in ('arm','group','stratum','scene_count','unit_count','status')} | {'metric': metric, 'estimate': value['estimate'], 'ci95_low': value['interval_95'][0], 'ci95_high': value['interval_95'][1]})
    text = ['# G保持性：已揭晓输出的配对分析', '', f"分析状态：{artifact['status']}；缺失项目：{len(missing)}。", '', '图像64题（cross/trend各32）与独立duplicate32题分开。图像和E共享的64个基础场景在bootstrap中联合移动。各区间为条件于已运行检查点的边际95%场景区间，不是种子总体区间，也不是同时置信带。', '', '|训练臂|分层|相对父模型变化 pp|条件场景95%区间 pp|', '|---|---|---:|---|']
    for row in results:
        if row['group'] == 'registered_parent_repeat_average' and row['stratum'] in strata and row['status'] == 'COMPLETE':
            m = row['metrics']['student_minus_parent']
            text.append(f"|{row['arm']}|{row['stratum']}|{100*m['estimate']:.3f}|[{100*m['interval_95'][0]:.3f}, {100*m['interval_95'][1]:.3f}]|")
    text += ['', '全部parent×repeat×arm单元、各parent重复均值及shared64的G/E联合变化见JSON/CSV。仅S96登记的臂，其汇总只覆盖S96。没有将G与E混合成排行榜；没有收益或负向变化均保留。区间包含0不等于无损或等效，未预登记非劣门槛，因此本表不输出“保持性认证”。', '', 'shared64的图像接口提供额外图像线索，接口差和变化差为描述性配对诊断，不能单独归因于模态。R0为32步，SFT为256步；计算预算不相等。']
    text += detailed_summary(artifact)
    dests[2].write_text('\n'.join(text) + '\n')
    print(json.dumps({'status': artifact['status'], 'results': len(results), 'missing': len(missing), 'outputs': [str(p) for p in dests]}))


if __name__ == '__main__':
    main()
