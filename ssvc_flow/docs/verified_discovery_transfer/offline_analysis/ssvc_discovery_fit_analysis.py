#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing-output, post-release discovery/fit summaries; no model or server access."""
import argparse,csv,hashlib,json,random,statistics
from pathlib import Path
from collections import defaultdict,Counter

def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def writecsv(p, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stats(x):
    x=sorted(x)
    return {'n':len(x),'sum':sum(x),'min':min(x) if x else None,'median':statistics.median(x) if x else None,'mean':statistics.mean(x) if x else None,'max':max(x) if x else None,'std_population':statistics.pstdev(x) if x else None}
def markdown(rows,keys):
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(str(r.get(k,'')) for k in keys)+' |' for r in rows])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot');ap.add_argument('--output');ap.add_argument('--overwrite',action='store_true');a=ap.parse_args()
    run=Path(a.snapshot);out=Path(a.output) if a.output else run/'final_analysis'
    release=read(run/'TEST_RELEASE.json')
    assert release['status']=='RELEASED' and release['scientific_completion'], 'Final release required'
    assert release['queue_identity']=='6e93b2305fdf089109b918945742f962d9f776ba2c432674504f14deb478ed27'
    out.mkdir(exist_ok=True)
    assert a.overwrite or not list(out.glob('discovery_fit*')), 'Refusing to overwrite existing analysis outputs'
    tasks={r['task_id']:r for r in lines(run/'cohort/T_train/tasks_public.jsonl')}
    audit={r['task_id']:r for r in lines(run/'cohort/T_train/audit_only.jsonl')}
    assert len(tasks)==384 and tasks.keys()==audit.keys()
    discoveries=read(run/'discovery_summary.json');train=read(run/'training_diagnostics.json');vf=read(run/'validation_and_fit.json')
    registry={r['job']:r for r in read(run/'TRAINING_REGISTRY.json')}
    summaries=[];prefix=[];strata=[];taskrows=[];matching=[];memberships={};checks=[]
    for key,d in sorted(discoveries.items()):
        parent=d['parent_id'];rep=d['pipeline_repeat'];sets={k:set(d[k]) for k in ('J_O0','J_MIX','J_SINGLE')}
        o,m,s=(sets[k] for k in ('J_O0','J_MIX','J_SINGLE'))
        summaries.append(dict(parent=parent,repeat=rep,N=384,J_O0=len(o),J_MIX=len(m),J_SINGLE=len(s),mix_added=len(m-o),mix_lost=len(o-m),mix_o0_intersection=len(m&o),mix_single_intersection=len(m&s),o0_single_intersection=len(o&s),three_way_intersection=len(o&m&s),union_all=len(o|m|s),calls_per_task_policy=16,logical_calls_per_policy=6144,physical_bank_calls=d['summary']['physical_atomic_calls']))
        for arm,j in [('SELF_O0',o),('SELF_MIX',m),('SELF_SINGLE',s),('GOLD_ALL',set(tasks)),('REPLAY_ONLY',set())]:memberships[f'sft.{parent}.{rep}.{arm}']=j
        selection=read(run/'evidence'/f'select.{parent}'/'selection.json')
        bank=defaultdict(lambda:defaultdict(dict)); ids=set();count=0
        for path in sorted((run/'evidence').glob(f'teacher.{parent}.T_train.{rep}.*/chunks/*.json')):
            ch=read(path);assert digest(ch['rows'])==ch['rows_hash']
            for r in ch['rows']:
                assert r['request_id'] not in ids;ids.add(r['request_id']);count+=1
                assert r['parent_id']==parent and r['pipeline_repeat']==rep and r['split']=='T_train'
                bank[r['task_id']][r['protocol_id']][r['draw_index']]=bool(r['public_verifier_pass'])
        assert count==15360
        for tid,t in tasks.items():
            family=t['family'];comp={'trend':'L11','cross_series':'B1'}[family];single=selection['best_single_B16'][family]
            for protocol in (('O0','L11') if family=='trend' else ('O0','L11','B1')):assert set(bank[tid][protocol])==set(range(16))
            aa=audit[tid]
            taskrows.append(dict(parent=parent,repeat=rep,task_id=tid,family=family,corrupted_index=aa['corrupted_index'],DPE1_O0=aa['DPE1']['O0'],DPE1_companion=aa['DPE1'][comp],truth_all_in_0_49=aa['truth_all_in_0_49'],J_O0=int(tid in o),J_MIX=int(tid in m),J_SINGLE=int(tid in s),mix_added=int(tid in m-o),mix_lost=int(tid in o-m)))
        for B in (2,4,8,16):
            selected={k:set() for k in sets}
            for tid,t in tasks.items():
                family=t['family'];comp={'trend':'L11','cross_series':'B1'}[family];single=selection['best_single_B16'][family]
                alloc={'J_O0':{'O0':B},'J_MIX':{'O0':B//2,comp:B//2},'J_SINGLE':{single:B}}
                for policy,protocols in alloc.items():
                    if any(bank[tid][p][i] for p,n in protocols.items() for i in range(n)):selected[policy].add(tid)
            if B==16:assert selected==sets
            for policy,j in selected.items():
                for family in ('ALL','trend','cross_series'):
                    pool={tid for tid,t in tasks.items() if family=='ALL' or t['family']==family}
                    prefix.append(dict(parent=parent,repeat=rep,budget=B,policy=policy,family=family,N=len(pool),covered=len(j&pool),coverage=len(j&pool)/len(pool),interpretation='registered_B16' if B==16 else 'posthoc_prefix_descriptive_fixed_B16_single',allocation='half_O0_half_companion' if policy=='J_MIX' else 'one_fixed_protocol'))
        for tid,t in tasks.items():
            aa=audit[tid];dim={'family':t['family'],'corrupted_index':str(aa['corrupted_index']),'family_x_position':f"{t['family']}:{aa['corrupted_index']}",'DPE1_O0_sign':'positive' if aa['DPE1']['O0']>0 else 'nonpositive','DPE1_O0_value':str(aa['DPE1']['O0']),'truth_range':'all_0_49' if aa['truth_all_in_0_49'] else 'has_50_99'}
            for dimension,value in dim.items():
                strata.append(dict(parent=parent,repeat=rep,dimension=dimension,value=value,N=1,J_O0=int(tid in o),J_MIX=int(tid in m),J_SINGLE=int(tid in s),mix_added=int(tid in m-o),mix_lost=int(tid in o-m)))
        matchjob=f'sft.{parent}.{rep}.GOLD_MATCH_MIX'
        if matchjob in registry:
            groups=defaultdict(list);quotas=Counter()
            for tid,t in tasks.items():groups[(t['family'],audit[tid]['corrupted_index'])].append(tid);quotas[(t['family'],audit[tid]['corrupted_index'])]+=tid in m
            rng=random.Random(106070+rep);chosen=set()
            for cell in sorted(groups):chosen.update(rng.sample(sorted(groups[cell]),quotas[cell]))
            actual={tid for tid in train[matchjob]['exposures'] if tid in tasks}
            assert actual==chosen;memberships[matchjob]=chosen
            receipt=read(run/'evidence'/matchjob/'gold_matching.json')
            assert abs(receipt['jaccard_with_mix']-len(chosen&m)/len(chosen|m))<1e-12
            matching.append(dict(parent=parent,repeat=rep,N_GOLD_MATCH=len(chosen),N_MIX=len(m),intersection=len(chosen&m),gold_only=len(chosen-m),mix_only=len(m-chosen),jaccard=len(chosen&m)/len(chosen|m),actual_exposure_ids_equal_frozen_matching=True,causal_scope='family_x_position_and_count_matched_not_difficulty_matched'))
        checks.append(dict(parent=parent,repeat=rep,chunks_hash_verified=True,physical_records=count,B16_reconstruction_matches=True,selection_digest=selection['selection_digest']))
    grouped={}
    for row in strata:
        k=tuple(row[x] for x in ('parent','repeat','dimension','value'))
        if k not in grouped:grouped[k]={x:row[x] for x in ('parent','repeat','dimension','value')}|{x:0 for x in ('N','J_O0','J_MIX','J_SINGLE','mix_added','mix_lost')}
        for x in ('N','J_O0','J_MIX','J_SINGLE','mix_added','mix_lost'):grouped[k][x]+=row[x]
    curves=[];exposure_rows=[];exposure_summary=[];training_summary=[]
    for job,tr in sorted(train.items()):
        if not job.startswith('sft.'):continue
        updates=tr['updates'];reg=registry[job];focus=memberships.get(job,set());exposures=tr['exposures'] or {}
        reconstructed=Counter()
        for u in updates:
            reconstructed.update(u['task_ids'])
            curves.append({'job':job,**{k:u.get(k) for k in ('update','loss','gradient_norm','parameter_delta_norm','lr','focus_mean_nll','replay_mean_nll','processed_target_sequences','processed_target_tokens','unique_exposed')}})
        if updates:assert dict(reconstructed)==exposures
        for tid,n in sorted(exposures.items()):
            exposure_rows.append(dict(job=job,task_id=tid,role='focus' if tid in tasks else 'replay',family=tasks[tid]['family'] if tid in tasks else 'historical_replay',exposures=n))
        for role in ('focus','replay'):
            vals=[exposures.get(tid,0) for tid in focus] if role=='focus' else [n for tid,n in exposures.items() if tid not in tasks]
            exposure_summary.append(dict(job=job,role=role,**stats(vals)))
        last=updates[-1] if updates else {}
        training_summary.append(dict(job=job,status=tr['status'],updates=len(updates),first_loss=updates[0].get('loss') if updates else None,last_loss=last.get('loss'),last_gradient_norm=last.get('gradient_norm'),last_focus_nll=last.get('focus_mean_nll'),last_replay_nll=last.get('replay_mean_nll'),focus_unique=len(focus),focus_exposure=sum(exposures.get(tid,0) for tid in focus),replay_exposure=sum(n for tid,n in exposures.items() if tid not in tasks),alias_of=tr.get('alias_of')))
    vrows=[]
    for job,v in sorted(vf.items()):
        parts=job.split('.');split='T_train' if parts[0]=='sentinel' else parts[-2];step=int(parts[-1])
        vrows.append(dict(job=job,parent=parts[1],repeat=int(parts[2]),arm=parts[3],split=split,step=step,pX_family_equal=v['pX_family_equal'],trend=v['families'].get('trend'),cross_series=v['families'].get('cross_series'),N=len(v['tasks']),draws_per_task=v['tasks'][0]['draws'],all_X=v['contrast_state_counts'].get('all-X',0),no_X=v['contrast_state_counts'].get('no-X',0),mixed_X=v['contrast_state_counts'].get('mixed-X',0)))
    sentinel_membership=[]
    for job,v in sorted(vf.items()):
        if not job.startswith('sentinel.'):continue
        _,parent,rep,arm,step=job.split('.')
        focus=memberships.get(f'sft.{parent}.{rep}.{arm}',set())
        for group in ('inside_training_J','outside_training_J'):
            rr=[r for r in v['tasks'] if (r['task_id'] in focus)==(group=='inside_training_J')]
            n=sum(r['draws'] for r in rr);correct=sum(r['successes'] for r in rr)
            sentinel_membership.append(dict(parent=parent,repeat=int(rep),arm=arm,step=int(step),group=group,N_tasks=len(rr),draws=n,successes=correct,pX=correct/n if n else None))
    outputs={'sentinel_membership':sentinel_membership,'summary':summaries,'budget_prefix':prefix,'strata':list(grouped.values()),'tasks':taskrows,'gold_matching':matching,'training_curves':curves,'training_summary':training_summary,'exposures':exposure_rows,'exposure_summary':exposure_summary,'validation_sentinel':vrows}
    for name,rows in outputs.items():writecsv(out/f'discovery_fit_{name}.csv',rows)
    (out/'discovery_fit_checks.json').write_text(json.dumps({'source_snapshot':str(run),'release':release,'checks':checks,'scope':'Existing data only; B<16 is descriptive prefix, no refit, no new model calls. Fit/V are not final transfer.'},ensure_ascii=False,indent=2)+'\n')
    report=['# 发现与拟合：最终既有数据核读','', '本分析只读取已统一释放的现有数据。B16为预登记发现预算；B2/4/8是事后固定前缀描述，沿用B16在V冻结的single，未重新选协议。MIX各分一半调用。拟合与V成功率不能替代E迁移和G保持性。','', '## 核读解释', '', '四组B16的MIX均比O0发现更多任务，同时仍丢失0–3题；集合不是嵌套关系。增益主要集中在DPE1_O0非正风险层：四组共347条新增membership中的344条来自该层（同一T任务在不同parent/repeat中重复计数，不能称为347个独立新题）。低数值域all_0_49只有16题，分层描述不宜过度外推。', '', 'MIX扩大训练集合，使固定3072次focus曝光分散到278–293题，每题约10.48–11.05次；O0为196–202题，每题约15.21–15.67次。曝光改变是注册流程的组成部分，不能把差异全部归因于协议本身。', '', '低训练loss不等于广泛拟合：SELF_O0的末步loss约4.19e-6–1.89e-5，但四组固定16题sentinel终点均仅6/16；SELF_MIX末步loss更高，却达到10/16–13/16。该sentinel同时包含未进入某臂J的T题，所以不是仅对该臂已训练目标的再现率。', '', 'V128保留不利结果：S96两轮MIX均47.66%，低于O0的49.22%和SINGLE的51.30%；REP96第一轮MIX与O0同为48.70%，第二轮MIX49.22%对O0 47.40%。这些是固定步骤诊断，不据此更换256步终点，也不是E迁移结论。', '', 'GOLD_MATCH与MIX实际集合交集分别257/290、265/293；每侧独有33、28题，Jaccard约0.796、0.826。匹配成功不代表二者输入相同。', '', '## 四组发现','',markdown(summaries,['parent','repeat','J_O0','J_MIX','J_SINGLE','mix_added','mix_lost','mix_o0_intersection','mix_single_intersection','o0_single_intersection','three_way_intersection']),'','## GOLD匹配实际重合','',markdown(matching,['parent','repeat','N_GOLD_MATCH','N_MIX','intersection','gold_only','mix_only','jaccard']), '','匹配仅控制family×损坏位置及数量，不能视为完整难度控制。','', '## 训练拟合','',markdown(training_summary,['job','updates','focus_unique','focus_exposure','replay_exposure','last_loss','last_gradient_norm']), '', '两个repeat共享同一T任务集合，不能把四组当成1536个独立题目；预算前缀曲线也不是额外实验。每个非空SFT臂256步，focus应3072次、replay应1024次；REPLAY_ONLY只有1024次replay且损失仍使用统一1/16系数。曝光逐题分布与零曝光见CSV。', '', '## Sentinel：训练集合内外的分解', '', '末步loss是当步mini-batch损失，不是全T平均。将固定sentinel按该臂实际训练集合分开：四组SELF_O0均在J内6/6成功、J外0/10。SELF_MIX在J内依次为S96 r0 9/11、r1 12/12、REP96 r0 12/12、r1 10/11；仍有已训练题失败，应保留。该分解只含固定16题，不足以代表全训练集。', '', markdown([r for r in sentinel_membership if r['step']==256],['parent','repeat','arm','group','N_tasks','successes','pX']), '', '## V与训练sentinel','',markdown(vrows,['parent','repeat','arm','split','step','N','draws_per_task','pX_family_equal','trend','cross_series']), '', 'T_train sentinel仅16题×1次，是固定训练拟合诊断；V为96题×4次，步骤冻结，不按曲线选终点。所有不利值保留。', '', '文件索引：discovery_fit_summary.csv（集合）、budget_prefix.csv（预算）、strata.csv（分层）、tasks.csv（逐题集合标志）、gold_matching.csv（匹配）、training_curves.csv（逐步loss/grad）、training_summary.csv、exposures.csv（逐题曝光）、exposure_summary.csv、validation_sentinel.csv。']
    (out/'discovery_fit_findings_zh.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'output':str(out),'groups':len(summaries),'training_arms':len(training_summary),'validation_points':len(vrows),'checks':'PASS','summary':summaries,'matching':matching},ensure_ascii=False))
if __name__=='__main__':main()
