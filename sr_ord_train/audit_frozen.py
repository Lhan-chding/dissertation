#!/usr/bin/env python3
"""CPU-only independent consistency audit of the scientific FREEZE and all its new PNGs.
Production requires rendered images. --allow-no-images is for explicitly labeled rehearsals only.
Legacy retention PNGs must be bound on the historical cluster, not rerendered/substituted here.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from export_runtime_plans import ARMS,PARENTS,SEEDS,ROOT,need,read_freeze,rows,indexed,file_hash,object_hash
from srf6_ref.core import seed_for
from data_design import CLEAR_SENTENCE

FAMILIES={'BINDING','IMPACT','TIES','SIGNED','INTERVAL','K_DIRECTION'}
PANELS={
    'transfer8':(96,('aligned','conflict'),2),'scale10':(24,('aligned','conflict'),1),
    'swap8':(32,('p0l0','p0l1','p1l0','p1l1'),2),'context8':(24,('base','nolist','list_conflict','report_conflict'),1),
    'authority8':(24,('chart','list'),1),'retention8':(24,('v0','v1','v2','v3'),1),
    'prefix10':(32,('p_stable','p_revtie','p_placebo'),1),
}

def audit(frozen,allow_no_images=False):
    frozen=Path(frozen).resolve(strict=True);m=read_freeze(frozen);rendered=m['images_rendered']
    need(type(rendered) is bool and (rendered or allow_no_images),'not a production image freeze; use --allow-no-images for rehearsal')
    cs=indexed(rows(frozen,'private_cases'),'case_id'); pub=indexed(rows(frozen,'inputs'),'input_id'); imgs=indexed(rows(frozen,'images'),'image_file')
    tr=rows(frozen,'training_groups');sm=rows(frozen,'smoke_groups');ev=rows(frozen,'evaluation_manifest');cl=rows(frozen,'clarification_manifest')
    for k,u in pub.items():
        need(set(u)=={'input_id','prompt','assistant_prefill','image_file'} and k==object_hash({n:v for n,v in u.items() if n!='input_id'}),'public input projection/hash')
        need(u['image_file'] in imgs,'unbound image')
    for k,c in cs.items():
        need(k==object_hash({n:v for n,v in c.items() if n!='case_id'}) and c['input_id'] in pub,'private case projection/hash')
        w=c['task']['world'];need(imgs[pub[c['input_id']]['image_file']]['world_sha256']==object_hash(w),'world/image mismatch')
    need({u['image_file'] for u in pub.values()}==set(imgs),'orphan image')
    new=0;legacy=[];pngs=set(); disk_pngs=set(str(p.relative_to(frozen)) for p in (frozen/'images').rglob('*.png'))
    for name,im in imgs.items():
        p=Path(name);need(not p.is_absolute() and '..' not in p.parts and p.suffix=='.png','illegal image path')
        if im['legacy_original_required']:
            need(p.parts[0]=='legacy','misrouted legacy image');legacy.append(name);continue
        need(name=='images/'+im['world_sha256']+'.png','new image identity');pngs.add(name)
        if rendered:
            need(file_hash(frozen/name)==im['png_sha256'],'new PNG content drift: '+name)
            with (frozen/name).open('rb') as f:need(f.read(8)==b'\x89PNG\r\n\x1a\n','image is not PNG '+name)
            new+=1
        else:need(im['png_sha256'] is None,'unrendered freeze has an unexpected PNG hash')
    need(disk_pngs==(pngs if rendered else set()),'extra/missing new PNG bytes')
    root_scope={};fact_owner={};used=set(); scopes=defaultdict(set); old=set(json.loads((ROOT/'history/open_world_hashes.json').read_text()))
    def get(row,scope):
        c=cs[row['case_id']];u=pub[row['input_id']];t=c['task'];rid=row['root_id'];im=imgs[u['image_file']]
        need(c['input_id']==row['input_id'] and t['root_id']==rid,'request-case-input/root identity')
        need(im['legacy_original_required']==(scope=='retention8'),'retention routing')
        if row['case_id'] not in used:
            w=t['world'];fingerprint=object_hash(sorted(zip(w['series']['Alpha'],w['series']['Beta'],strict=True)))
            need(fact_owner.setdefault(fingerprint,rid)==rid,'fact-table clone across distinct roots: '+rid)
            need(scope=='retention8' or fingerprint not in old,'collision with previously opened real worlds')
            used.add(row['case_id'])
        need(root_scope.setdefault(rid,scope)==scope,'root leakage across splits/panels: '+rid);scopes[scope].add(rid)
        return c,u,t
    groups=defaultdict(dict);path_counts=defaultdict(Counter);seedsets={};paths=defaultdict(set);pair_inputs={}
    for r in tr:
        c,u,t=get(r,'train');p,s,a,st,slot=(r[k] for k in ('parent','train_seed','arm','step','slot'))
        need(p in PARENTS and s in SEEDS and a in ARMS and st in range(1,25) and slot in range(16) and r['draws']==8,'invalid train slot')
        need(r['family'] in FAMILIES and r['planned_slot'] in ('BASE','LIST','REPORT') and r['channel']==('BASE' if a=='R' else r['planned_slot']),'registered arm/channel')
        need(r['group_seed']==seed_for(2026110301,'G8',p,s,st,slot),'train seed drift')
        need(a not in groups[p,s,st,slot],'duplicate train arm slot');groups[p,s,st,slot][a]=r
        path_counts[p,s,a][r['family']]+=1;paths[p,s,a].add(r['root_id'])
        need(u['assistant_prefill']=='{' and len(t['world']['categories'])==8 and c['audit']['native_J_applicable'] is True,'train prefill/size/scorer')
        key=(s,a,st,slot);v=(r['case_id'],r['input_id'],r['root_id'],r['family'],r['channel']);need(pair_inputs.setdefault(key,v)==v,'parent-specific prompts leaked into paired schedule')
    allpaths=set(product(PARENTS,SEEDS,ARMS));need(set(paths)==allpaths,'missing path')
    for path in allpaths:
        need(path_counts[path]==Counter({f:64 for f in FAMILIES}) and len(paths[path])==384,'unbalanced exposure/repeated world')
    need(all(x==next(iter(paths.values())) for x in paths.values()),'seeds/arms differ in world inventory')
    dose=defaultdict(lambda:defaultdict(lambda:{'slots':0,'G_conflicts':0,'D_conflicts':0,'different_inputs':0,'different_steps':set()}));sub=defaultdict(Counter)
    for (p,s,st,i),grp in groups.items():
        need(set(grp)==set(ARMS),'unmatched R/G/D slot');r,g,d=(grp[x] for x in ARMS)
        for field in ('root_id','family','planned_slot','structure','group_seed'):
            need(len({v[field] for v in grp.values()})==1,'unpaired '+field)
        sameworld=[object_hash(cs[v['case_id']]['task']['world']) for v in grp.values()];samequery=[object_hash(cs[v['case_id']]['task']['query']) for v in grp.values()]
        need(len(set(sameworld))==len(set(samequery))==1,'different chart or task across R/G/D')
        ch=g['channel'];need(ch==d['channel'],'G/D channel mismatch')
        if g['family']=='TIES':sub[p,s,(st-1)//3][ch]+=1
        if ch=='BASE':need(r['input_id']==g['input_id']==d['input_id'],'baseline public input differs');continue
        need(g['family']=='TIES' and ch in ('LIST','REPORT'),'non-TIES active intervention')
        data={}
        for a,v in [('G',g),('D',d)]:
            c=cs[v['case_id']];t=c['task'];x=c['audit'];names=x['list_order'];w=t['world'];cats=w['categories'];rank={name:j for j,name in enumerate(names)}
            need(len(names)==len(cats) and set(names)==set(cats),'listed categories differ')
            sign=-1 if t['query']['direction']=='largest' else 1
            choice={cats[j] for j in sorted(range(len(cats)),key=lambda j:(sign*w['series']['Alpha'][j],rank[cats[j]]))[:t['query']['k']]}
            conflict=choice!=set(t['gold']['selected']);need(x['list_conflict'] is conflict and set(x['list_induced_selected'])==choice,'private conflict label differs')
            need(x['expected_report_order']==(names if ch=='REPORT' else None),'unmatched report-order burden')
            need(x['order_attempts']>=1 and (a!='G' or x['order_attempts']==1),'G is not unconditional first candidate');data[a]=conflict
        need(data['D'],'D failed to create targeted conflict'); changed=g['input_id']!=d['input_id'];need(changed==(not data['G']),'identical first conflict must stay identical across G/D')
        need(cs[d['case_id']]['audit']['order_attempts']==1 if data['G'] else cs[d['case_id']]['audit']['order_attempts']>=2,'D candidate stream not matched')
        need(r['input_id']!=d['input_id'],'targeted perturbation missing')
        if p=='b0':
            z=dose[str(s)][ch];z['slots']+=1;z['G_conflicts']+=int(data['G']);z['D_conflicts']+=1;z['different_inputs']+=int(changed)
            if changed:z['different_steps'].add(st)
    need(len(sub)==32 and all(v==Counter(BASE=4,LIST=2,REPORT=2) for v in sub.values()),'three-update superblock dose drift')
    dose={s:{k:{**v,'different_steps':sorted(v['different_steps'])} for k,v in d.items()} for s,d in dose.items()};need(dose==m['doses'],'reconstructed dose differs')
    e_sets=defaultdict(dict);paired={}; eval_seed={};independent={}
    for r in ev:
        n=r['panel'];need(n in PANELS,'unknown evaluation panel');c,u,t=get(r,n);p,a,s,st=(r[k] for k in ('parent','arm','train_seed','step'))
        need(p in PARENTS and ((a=='PARENT' and s is None and st==0) or (a in ARMS and s in SEEDS and st in (12,24))),'unregistered evaluation endpoint')
        rid,v,dr=(r[k] for k in ('root_id','view','draw'));need(v in PANELS[n][1] and dr in range(PANELS[n][2]),'eval view/draw')
        need(r['seed']==seed_for(2026110301,'eval',p,n,rid,dr),'eval seed drift')
        key=(n,rid,v,dr);need(key not in e_sets[p,a,s,st],'duplicate endpoint row');ref=(r['case_id'],r['input_id'],r['U_applicable']);e_sets[p,a,s,st][key]=ref
        need(paired.setdefault(key,ref)==ref,'endpoints were shown different cases')
        need(r['U_applicable']==(c['audit'].get('expected_report_order') is not None),'U applicability')
    need(len(e_sets)==26 and all(len(o)==1024 and o==paired for o in e_sets.values()),'incomplete paired test inventories')
    panelcounts=Counter(k[0] for k in paired)
    for n,(nr,vs,nd) in PANELS.items():need(len(scopes[n])==nr and panelcounts[n]==nr*len(vs)*nd,'wrong panel count '+n)
    clar=defaultdict(dict)
    for r in cl:
        c,u,t=get(r,'clarification');rid,v,p,dr=(r[k] for k in ('root_id','view','parent','draw'))
        need(p in PARENTS and dr in (0,1) and v in ('aligned_plain','aligned_clear','conflict_plain','conflict_clear'),'clarification index')
        need(v not in clar[rid,p,dr],'duplicate clarification row');clar[rid,p,dr][v]=(r,c,u)
    for (rid,p,dr),g in clar.items():
        need(len(g)==4 and len({x[0]['seed'] for x in g.values()})==1,'unpaired 2x2 clarification')
        for phase in ['aligned','conflict']:
            r,c,u=g[phase+'_plain'];rr,cc,uu=g[phase+'_clear'];need(u['image_file']==uu['image_file'] and u['assistant_prefill']==uu['assistant_prefill'] and c['audit']['list_order']==cc['audit']['list_order'] and uu['prompt']==u['prompt']+'\n'+CLEAR_SENTENCE,'clarification changed more than clarification sentence: '+rid)
            need(c['audit']['list_conflict'] is (phase=='conflict') and cc['audit']['list_conflict'] is (phase=='conflict'),'clarification incorrectly marked conflict')
    need(len(clar)==96 and len(scopes['clarification'])==24,'clarification roots/replicates')
    sids=set()
    for r in sm:
        c,u,t=get(r,'smoke');k=(r['parent'],r['group_index']);need(r['draws']==8 and k[0] in PARENTS and k[1] in range(12) and k not in sids,'smoke id');sids.add(k)
        need(r['group_seed']==seed_for(2026110301,'smoke',k[1],k[0]),'smoke seed drift')
    need(len(sids)==24 and len(scopes['smoke'])==12,'smoke scope')
    need(used==set(cs) and {c['input_id'] for c in cs.values()}==set(pub),'orphan case/input')
    counts=dict(train_groups=len(tr),train_rollouts=len(tr)*8,optimizer_updates=len(paths)*24,parent_endpoints=2,trained_endpoints=24,endpoint_rows=len(paired),evaluation=len(ev),clarification=len(cl),smoke=len(sm)*8,total_generations=(len(tr)+len(sm))*8+len(ev)+len(cl),panel_per_endpoint=dict(panelcounts))
    need(counts==m['counts'] and [counts[k] for k in ('train_rollouts','optimizer_updates','evaluation','clarification','smoke','total_generations')]==[36864,288,26624,384,192,64064],'registered scientific budget changed')
    return dict(status='CPU_FROZEN_AUDIT_PASS' if rendered else 'CPU_REHEARSAL_ONLY_NO_IMAGE_GATE',freeze_sha256=file_hash(frozen/'FREEZE.json'),manifest_files_verified=len(m['files']),new_png_sha256_verified=new,legacy_original_pngs_to_bind_on_cluster=len(legacy),legacy_status='DEPLOYMENT_BIND_REQUIRED_NOT_VERIFIED_HERE',scope_unique_roots={k:len(v) for k,v in sorted(scopes.items())},distinct_fact_multisets=len(fact_owner),distinct_inputs=len(pub),distinct_cases=len(cs),counts=counts,doses=dose,clarification_pairs_have_identical_order=True,model_calls=0,optimizer_steps_executed=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen',type=Path,default=ROOT/'frozen');p.add_argument('--allow-no-images',action='store_true');a=p.parse_args()
    try:print(json.dumps(audit(a.frozen,a.allow_no_images),ensure_ascii=False,indent=2,sort_keys=True))
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:p.exit(2,'FAIL_CLOSED: '+str(e)+'\n')
if __name__=='__main__':main()
