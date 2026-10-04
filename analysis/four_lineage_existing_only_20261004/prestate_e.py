#!/usr/bin/env python3
"""Existing saved P packet counts only; no model calls, refits, search or resampling."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def dump_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields,lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def compact(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def distances(a, b):
    n, m = sum(a.values()), sum(b.values())
    if not n or not m:
        return None, None
    keys = set(a) | set(b)
    raw = sum((a.get(k, 0) / n - b.get(k, 0) / m)**2 for k in keys)
    if min(n, m) < 2:
        return raw, None
    cross = sum(a.get(k, 0)*b.get(k, 0) for k in keys)/(n*m)
    deb = sum(v*(v-1) for v in a.values())/(n*(n-1))
    deb += sum(v*(v-1) for v in b.values())/(m*(m-1)) - 2*cross
    return raw, deb


def conditional(snapshot, pid):
    out = defaultdict(Counter)
    h = snapshot['repair_histograms'][pid]
    assert h['n'] == 32 and sum(h['counts'].values()) == 32
    for atom, count in h['counts'].items():
        reward, fbm = ('I:0/1', 'INVALID') if atom == 'INVALID' else atom.split('|', 1)
        out[reward][fbm] += count
    assert dict((k, sum(v.values())) for k, v in out.items()) == snapshot['reward_histograms'][pid]['counts']
    return out


def comparison(a, b, atom):
    na, nb = sum(a.values()), sum(b.values())
    valid = not atom.startswith('I:')
    raw, deb = distances(a, b) if valid else (None, None)
    status = ('INVALID_REPAIR_MISSING' if not valid else
              'UNKNOWN_ZERO_SUPPORT' if not na or not nb else
              'RAW_ONLY_N_LT_2' if min(na, nb) < 2 else 'ESTIMABLE_CONDITIONAL_IID')
    row = dict(a_n=na, b_n=nb, a_event_mass=na/32, b_event_mass=nb/32,
               a_valid_repair_n=na if valid else 0, b_valid_repair_n=nb if valid else 0,
               a_distinct_fbm=len(a) if valid else None, b_distinct_fbm=len(b) if valid else None,
               a_counts=compact(a), b_counts=compact(b),
               a_conditional_fbm=compact({k:v/na for k,v in a.items()}) if na and valid else '',
               b_conditional_fbm=compact({k:v/nb for k,v in b.items()}) if nb and valid else '',
               pooled_distinct_fbm=len(set(a)|set(b)) if valid else None,
               raw_conditional_l2_sq=raw, count_corrected_conditional_l2_sq=deb,
               overlapping_event_mass=min(na,nb)/32,
               overlap_mass_times_raw_l2_sq=min(na,nb)/32*raw if raw is not None else None,
               overlap_mass_times_corrected_l2_sq=min(na,nb)/32*deb if deb is not None else None,
               support_status=status)
    for label, counts, n in [('a', a, na), ('b', b, nb)]:
        for j, key in enumerate(('F','B','M')):
            row[f'{label}_{key}_distinct'] = len({k.split('|')[j] for k in counts}) if valid and n else None
            row[f'{label}_{key}_conditional_mean'] = sum(int(k.split('|')[j])*v for k,v in counts.items())/n if valid and n else None
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--raw-root', type=Path, default=Path('/private/tmp/ssvc-four-existing-20261004'))
    args = p.parse_args()
    root, out = args.root.resolve(), Path(__file__).resolve().parent
    inputs = []
    def read(rel, mode='json'):
        path = root / rel
        data = path.read_bytes()
        inputs.append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        return json.loads(data) if mode == 'json' else list(csv.DictReader(data.decode().splitlines())) if mode == 'csv' else data.decode()
    base = Path('ssvc_flow/docs/prospective_selection')
    data = read(base/'first_four_20261004/compact_input.json')
    selected = read(base/'pilot_four_20261004/selected_outcomes.csv', 'csv')
    prepared = read(args.raw_root/'inputs/prepared_first_four_ONLY.json')
    assert set(prepared['panels']) == {'P','D','E_old'}
    metadata = {p['prompt_id']:{k:p[k] for k in ('base_scene_id','operation')} for p in prepared['panels']['P']}
    # Saved experiment source is evidence for packet semantics, never imported/executed.
    src = args.raw_root/'code_afd3119/ssvc_flow/src/prospective_selection/features.py'
    source_text = read(src, 'text')
    assert '"INVALID" if invalid else f"{rbin}|{row[\'F\']}|{row[\'B\']}|{row[\'M\']}"' in source_text
    packets = {r['origin']:r['packets']['Z3_REPAIR_STRUCTURE'] for r in data['prestates']}
    assert len(packets) == 8
    conditional_counts, atoms = {}, defaultdict(set)
    prompts = sorted(next(iter(packets.values()))['current']['reward_histograms'])
    assert len(prompts) == 72
    for origin, packet in packets.items():
        assert packet['history_step'] == packet['step']-8
        for time in ('history','current'):
            snap = packet[time]
            assert sorted(snap['reward_histograms']) == prompts and snap['n'] == 2304
            for pid in prompts:
                c = conditional(snap, pid)
                conditional_counts[origin,time,pid] = c
                atoms[pid].update(c)
                atoms[pid].update(('I:0/1','X:1/1'))
    responses = defaultdict(dict)
    for branch in data['branches']:
        payload = branch['task']['payload']
        oid, recipe = payload['origin_id'], payload['recipe_id']
        assert recipe not in responses[oid]
        responses[oid][recipe] = branch['evaluations']['32']['summary']['J']
    temporal, prompt_rows, group_rows, points = [], [], [], []
    for oid, packet in sorted(packets.items()):
        prefix = dict(origin_id=oid, lineage_id=packet['lineage_id'], source_recipe=packet['source_recipe'], source_step=packet['step'], history_step=packet['history_step'], panel='P')
        local = []
        for pid in prompts:
            family, interface = packet['current']['prompt_strata'][pid].split('|')
            hp, cp = conditional_counts[oid,'history',pid], conditional_counts[oid,'current',pid]
            for atom in sorted(atoms[pid]):
                temporal.append({**prefix, 'prompt_id':pid, **metadata[pid], 'family':family, 'interface':interface, 'a_state':'P(t-8)', 'b_state':'P(t)', 'reward_atom':atom, **comparison(hp.get(atom,{}),cp.get(atom,{}),atom)})
            pr = {**prefix,'prompt_id':pid,**metadata[pid],'family':family,'interface':interface}
            for time, counts in [('history',hp),('current',cp)]:
                multi = [a for a,c in counts.items() if not a.startswith('I:') and len(c)>1]
                pr[f'{time}_multistructure_atoms'] = len(multi)
                pr[f'{time}_multistructure_mass'] = sum(sum(counts[a].values()) for a in multi)/32
                pr[f'{time}_valid_n'] = 32-sum(counts.get('I:0/1',{}).values())
                pr[f'{time}_pX'] = sum(counts.get('X:1/1',{}).values())/32
            subset = temporal[-len(atoms[pid]):]
            for key in ('overlap_mass_times_raw_l2_sq','overlap_mass_times_corrected_l2_sq'):
                pr[key] = sum(r[key] for r in subset if r[key] is not None)
            pr['overlap_mass_times_raw_l2_sq_on_corrected_support'] = sum(r['overlap_mass_times_raw_l2_sq'] for r in subset if r['count_corrected_conditional_l2_sq'] is not None)
            pr['raw_estimable_mass'] = sum(r['overlapping_event_mass'] for r in subset if r['raw_conditional_l2_sq'] is not None)
            pr['corrected_estimable_mass'] = sum(r['overlapping_event_mass'] for r in subset if r['count_corrected_conditional_l2_sq'] is not None)
            for name in ('reward','repair'):
                a = packet['history'][name+'_histograms'][pid]['counts']
                b = packet['current'][name+'_histograms'][pid]['counts']
                raw, deb = distances(a,b)
                pr[name+'_raw_l2_sq'], pr[name+'_count_corrected_l2_sq'] = raw, deb
            local.append(pr)
        prompt_rows.extend(local)
        groups = [('ALL', 'ALL',local)]
        groups.extend((fa, it,[r for r in local if r['family']==fa and r['interface']==it]) for fa,it in sorted(set((r['family'],r['interface']) for r in local)))
        means = ('overlap_mass_times_raw_l2_sq_on_corrected_support','history_multistructure_mass','current_multistructure_mass','history_pX','current_pX','overlap_mass_times_raw_l2_sq','overlap_mass_times_corrected_l2_sq','raw_estimable_mass','corrected_estimable_mass','reward_raw_l2_sq','reward_count_corrected_l2_sq','repair_raw_l2_sq','repair_count_corrected_l2_sq')
        for fa,it,rows in groups:
            g = {**prefix,'family':fa,'interface':it,'prompts_n':len(rows),'draws_per_snapshot':len(rows)*32}
            for time in ('history','current'):
                g[time+'_multistructure_prompt_n'] = sum(r[time+'_multistructure_atoms']>0 for r in rows)
                g[time+'_multistructure_atom_n'] = sum(r[time+'_multistructure_atoms'] for r in rows)
                g[time+'_valid_n'] = sum(r[time+'_valid_n'] for r in rows)
            for key in means:
                g[key] = sum(r[key] for r in rows)/len(rows)
            group_rows.append(g)
            if fa == 'ALL':
                point = dict(g)
        rr = responses[oid]
        assert len(rr)==11
        core = [rr[f'R{i}'] for i in range(8)]
        sel = [r for r in selected if r['origin']==oid and r['level']=='Z3_REPAIR_STRUCTURE']
        assert len(sel)==1
        point.update(r0_pX=rr['R0'],core_R0_R7_range=max(core)-min(core),core_noisy_max_minus_R0=max(core)-rr['R0'],all11_range=max(rr.values())-min(rr.values()),fold_best_static_recipe=sel[0]['best_static'],fold_best_static_pX=float(sel[0]['best_static_J']),z3_selected_recipe=sel[0]['recipe'])
        for recipe,value in sorted(rr.items()):
            point['delta_pX_'+recipe] = value-rr['R0']
        points.append(point)
    pair_rows, pair_summary = [], []
    for time in ('history','current'):
        for a,b in itertools.combinations(sorted(packets),2):
            pr = dict(a_origin=a,b_origin=b,time=time,a_step=packets[a]['step'],b_step=packets[b]['step'])
            start = len(pair_rows)
            for pid in prompts:
                family,interface=packets[a][time]['prompt_strata'][pid].split('|')
                ca,cb=conditional_counts[a,time,pid],conditional_counts[b,time,pid]
                for atom in sorted(atoms[pid]):
                    pair_rows.append({**pr,'prompt_id':pid,**metadata[pid],'family':family,'interface':interface,'reward_atom':atom,**comparison(ca.get(atom,{}),cb.get(atom,{}),atom)})
            rows=pair_rows[start:]
            for key in ('overlap_mass_times_raw_l2_sq','overlap_mass_times_corrected_l2_sq'):
                pr[key]=sum(r[key] for r in rows if r[key] is not None)/72
            pr['corrected_estimable_mass']=sum(r['overlapping_event_mass'] for r in rows if r['count_corrected_conditional_l2_sq'] is not None)/72
            pr['observed_different_conditionals_n']=sum(r['raw_conditional_l2_sq'] is not None and r['raw_conditional_l2_sq']>1e-12 for r in rows)
            pair_summary.append(pr)
    outputs = {'prestate_reward_conditional_repair.csv':temporal,'prestate_information_vs_response.csv':points,'prestate_E_prompt_summary.csv':prompt_rows,'prestate_E_group_summary.csv':group_rows,'prestate_E_between_origins.csv':pair_rows,'prestate_E_between_origins_summary.csv':pair_summary}
    for name,rows in outputs.items():
        dump_csv(out/name,rows)
    manifest=dict(inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),commands=['python3 analysis/four_lineage_existing_only_20261004/prestate_e.py'],scope='Only existing P(t-8), P(t) packets; existing H32 branch J and saved fold selections used solely for eight-point descriptive table.',rows={k:len(v) for k,v in outputs.items()},new_model_calls=0,new_samples=0,selector_refits=0,hyperparameter_searches=0,statistical_resamples=0,missing=[{'item':'Original independent-analysis count-noise L2 implementation','impact':'Not available in supplied checkout or located archive; standard multinomial collision U-statistic used as explicitly new algebraic analysis, not claimed original-code replay.'}],notes=['Atoms are union observed in the sixteen P snapshots for each question plus I and X; atoms never observed anywhere are not an identified full universe. Zero support is UNKNOWN, never a zero conditional distribution.','Count-corrected squared distance: sum c(c-1)/(n(n-1)) + sum d(d-1)/(m(m-1)) - 2 sum cd/(nm). Requires n,m>=2 and independent multinomial draws; nonpaired history/current samples, no paired-draw covariance estimated.','Overlap-mass weight min(n,m)/32 is empirical; its product with corrected conditional distance is descriptive and is NOT an unbiased unconditional estimator. Unsupported comparisons omitted from sums and coverage reported; never interpreted as zero.','Negative corrected estimates preserved as noise estimates, not negative true distances.','All prompt aggregation uses equal 1/72 weights, six fixed strata each 12 prompts. Eight origin points from four lineages are the state units; prompt rows are repeated measurements, not independent states.','I repair fields remain missing. No endpoint labels used to select features or atoms.'])
    (out/'inputs_E.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rows':manifest['rows'],'point_summary':[{k:r[k] for k in ['origin_id','current_multistructure_prompt_n','current_multistructure_atom_n','current_multistructure_mass','overlap_mass_times_raw_l2_sq','overlap_mass_times_corrected_l2_sq','core_R0_R7_range']} for r in points]},indent=2))

if __name__=='__main__':
    main()
