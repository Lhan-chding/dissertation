#!/usr/bin/env python3
"""Build a deterministic, offline experiment inventory. No accelerator or model invocation."""
import argparse,hashlib,json,random,sys,copy
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product,combinations
from fractions import Fraction
from background import background,retention,FAMILIES,normalize
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/probe_ref'))
from order_probe import data as olddata, inputs as oldinput,render
from sr_f6.interface_v3_protocol import public_prompt,format_contract
from data_design import make_world,build_context,swap_four,induced_choice
from srf6_ref.core import seed_for,enc
MASTER=2026110301
ARMS=('R','G','D'); PARENTS=('b0','b1'); SEEDS=(11031,11032)
def sha(b):return hashlib.sha256(b).hexdigest()
def js(o):return json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def h(o):return sha(js(o).encode())
def seed(*x):return seed_for(MASTER,*x)
def native_ctx(t):
    return {'task':t,'prompt':public_prompt(t['world']['categories'],t['query']),'assistant_prefill':'{','audit':{'channel':'BASE','policy':'R','list_conflict':None,'expected_report_order':None,'native_J_applicable':True}}
def make(split,i,n=8):return make_world(split,i,n,master_seed=MASTER)
def authority(t):
    # Keep the chart, all facts and the only category list the same; replace exactly one tie rule sentence.
    c=build_context(t,'ENUM','D',seed('auth',t['root_id']));d=copy.deepcopy(c)
    phrase='Break ranking ties by printed left-to-right order.'
    if d['prompt'].count(phrase)!=1:raise ValueError('cannot bound authority replacement')
    d['prompt']=d['prompt'].replace(phrase,'Break ranking ties by the order of category names in the public ranking list below, from first to last.')
    names=c['audit']['list_order']; w=t['world']; q=t['query'];cs=w['categories']; a=w['series']['Alpha'];b=w['series']['Beta'];sign=-1 if q['direction']=='largest' else 1
    sel=induced_choice(t,names);ix=sorted(cs.index(n) for n in sel)
    candidates=[list(comb) for comb in combinations(range(8),q['k']) if all((sign*a[i],names.index(cs[i]))<(sign*a[j],names.index(cs[j])) for i in comb for j in range(8) if j not in comb)]
    if candidates!=[ix]:raise ValueError('list-authority independent selector mismatch')
    ans=sum(Fraction(b[i]) for i in ix)/(len(ix) if q['aggregate']=='mean' else 1)
    if ans==Fraction(t['gold']['answer']):raise ValueError('authority answer collision')
    d['task']['gold']={'selected':[cs[i] for i in ix],'indices':ix,'operands':[str(b[i]) for i in ix],'answer':enc(ans)}
    d['task']['authority']='list';d['audit']['native_J_applicable']=False
    d['public']={'prompt':d['prompt'],'assistant_prefill':d['assistant_prefill']}
    return {'chart':c,'list':d}
class Bank:
    def __init__(self,path,font,images):
        self.path=path;self.font=font;self.images=images;self.pub={};self.cases={};self.img={};self.owners={};self.blocklist=set(json.loads((ROOT/'history/open_world_hashes.json').read_text()))
    def add(self,c,panel):
        t=c['task'];w=t['world']; rid=t['root_id']; wk=h(w)
        # Fact-multiset key is invariant to row order and naming. Related variants in a root are allowed.
        fp=sha(json.dumps(sorted(zip(w['series']['Alpha'],w['series']['Beta'])),separators=(',',':')).encode())
        if fp in self.blocklist and panel!='retention8':raise ValueError('previously open world collision '+rid)
        if self.owners.setdefault(fp,rid)!=rid:raise ValueError('cross-root collision '+rid)
        legacy=panel=='retention8'
        image='legacy/'+t['qid']+'.png' if legacy else 'images/'+wk+'.png'
        if image not in self.img:
            self.img[image]={'image_file':image,'world_sha256':wk,'legacy_original_required':legacy,'png_sha256':None}
            if not legacy and self.images:
                render.render(t,self.path/image,self.font);self.img[image]['png_sha256']=sha((self.path/image).read_bytes())
        public={'image_file':image,'prompt':c['prompt'],'assistant_prefill':c['assistant_prefill']};k=h(public);self.pub[k]={'input_id':k,**public}
        private={'input_id':k,'task':t,'audit':c['audit']};cid=h(private);self.cases[cid]={'case_id':cid,**private}
        return {'input_id':k,'case_id':cid}
def put(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for row in rows:f.write(js(row)+'\n')
def construct(dest,font,images=True):
    if dest.exists():raise FileExistsError(f'use an empty new destination: {dest}')
    dest.mkdir(parents=True); bank=Bank(dest,font,images);train=[];dev=[];ev=[];smoke=[];plans=[]
    def base_row(t):return {'root_id':t['root_id'],'structure':t['structure']}
    # Exactly 8 superblocks of 3 updates. Never wait for model outcomes to select a prompt.
    ties=[make('TRAIN',i) for i in range(64)]
    bg={fam:[background(fam,i) for i in range(64)] for fam in FAMILIES}
    for s in SEEDS:
        rng=random.Random(seed('schedule',s)); bins=[ties[k::4].copy() for k in range(4)]
        for x in bins:rng.shuffle(x)
        bgp={f:rng.sample(bg[f],64) for f in FAMILIES}
        for blk in range(8):
            back=[t for f in FAMILIES for t in bgp[f][8*blk:8*blk+8]];rng.shuffle(back)
            m=['BASE','LIST','BASE','REPORT','BASE','LIST','REPORT','BASE']
            bsets=[[] for _ in range(3)]
            for j,(step,ch) in enumerate(zip((0,0,0,1,1,1,2,2),m)):
                t=bins[(j+blk)%4].pop();bsets[step].append((t,'TIES',ch))
            for kk,items in enumerate(bsets):
                while len(items)<16:
                    t=back.pop();items.append((t,t['family'],'BASE'))
                rng.shuffle(items)
                for slot,(t,f,ch) in enumerate(items):
                    step=blk*3+kk+1;perm=seed('list',s,t['root_id'],ch)
                    for arm in ARMS:
                        ach='BASE' if arm=='R' else ch
                        c=(build_context(t,ach,arm,perm) if f=='TIES' else native_ctx(t))
                        ids=bank.add(c,'train')
                        for p in PARENTS:
                            train.append(dict(arm=arm,parent=p,train_seed=s,step=step,slot=slot,**base_row(t),family=f,planned_slot=ch,channel=ach,**ids,
                                              group_seed=seed('G8',p,s,step,slot),draws=8))
    for i in range(24):
        t=make('CLARIFICATION',i); u=seed('clarify',i)
        for channel,view in [('BASE','aligned_plain'),('CLARIFY','aligned_clear'),('ENUM','conflict_plain'),('CLARIFY_CONFLICT','conflict_clear')]:
            c=build_context(t,channel,'R',u);ids=bank.add(c,'clarification')
            for p,dr in product(PARENTS,range(2)):
                dev.append(dict(panel='clarification',parent=p,view=view,draw=dr,**base_row(t),**ids,seed=seed('dev',i,p,dr)))
    for i in range(12):
        t=make('SMOKE',i);ch=('BASE','LIST','REPORT')[i%3];c=build_context(t,ch,'D',seed('smoke-order',i));ids=bank.add(c,'smoke')
        for p in PARENTS:smoke.append(dict(parent=p,group_index=i,**base_row(t),**ids,group_seed=seed('smoke',i,p),draws=8))
    def paneladd(p,v,c,n):
        t=c['task'];ids=bank.add(c,p)
        for dr in range(n):plans.append(dict(panel=p,view=v,draw=dr,**base_row(t),**ids,U_applicable=c['audit'].get('expected_report_order') is not None))
    for panel,count,n in [('transfer8',96,8),('scale10',24,10)]:
        for i in range(count):
            t=make(panel,i,n)
            for v in ('aligned','conflict'):paneladd(panel,v,build_context(t,'TRANSFER',v.upper(),seed(panel,i)),2 if n==8 else 1)
    for i in range(32):
        t=make('swap8',i)
        for k,c in swap_four(t,seed('swap',i)).items():paneladd('swap8','p'+k[1]+'l'+k[-1],c,2)
    for i in range(24):
        t=make('context8',i)
        for k,v in [('BASE','base'),('NO_LIST','nolist'),('LIST','list_conflict'),('REPORT','report_conflict')]:
            paneladd('context8',v,build_context(t,k,'D',seed('context',i,k)),1)
        for v,c in authority(make('authority8',i)).items():paneladd('authority8',v,c,1)
    for t in retention():paneladd('retention8','v'+str(t['variant']),native_ctx(t),1)
    # Prior prefix protocol; one naming view per fresh root, with both name alignments balanced across 32 roots.
    cfg=json.loads((ROOT/'history/probe_config.json').read_text());cfg['data']['master_seed']=2026110329
    for i in range(32):
        m,r=((2,1),(3,1),(3,2),(4,2))[i%4]
        rr=olddata.build_root(cfg,'OT1_PREFIX10',i,m,r,['largest','smallest'][(i//4)%2],['sum','mean'][(i//8)%2],'unselected_matched_size')
        olddata.validate_root(rr);t=next(v for v in rr['views'] if v['rename']==['ALIGNED','CONFLICT'][(i//16)%2]);oldinput.validate_prefix_triplet(t)
        for ch in ('P_STABLE','P_REVTIE','P_PLACEBO'):
            paneladd('prefix10',ch.lower(),{'task':t,'prompt':oldinput.prompt(t,ch,format_contract()),'assistant_prefill':oldinput.assistant_prefill(t,ch),'audit':{'channel':ch,'expected_report_order':None,'native_J_applicable':False,'prefix_induced_selected':oldinput.prefix_choice(t,ch)}},1)
    for p in PARENTS:
        for arm,s,step in [('PARENT',None,0)]+list(product(ARMS,SEEDS,(12,24))):
            for row in plans:
                ev.append({**row,'parent':p,'arm':arm,'step':step,'train_seed':s,'seed':seed('eval',p,row['panel'],row['root_id'],row['draw'])})
    counts={'train_groups':len(train),'train_rollouts':8*len(train),'optimizer_updates':12*24,'parent_endpoints':2,'trained_endpoints':24,'endpoint_rows':len(plans),'evaluation':len(ev),'clarification':len(dev),'smoke':len(smoke)*8,'total_generations':len(ev)+len(dev)+(len(train)+len(smoke))*8,'panel_per_endpoint':dict(sorted(Counter(r['panel'] for r in plans).items()))}
    if [counts[k] for k in ('train_rollouts','evaluation','clarification','smoke','total_generations')]!=[36864,26624,384,192,64064]:raise AssertionError(counts)
    for file,rows in [('inputs',bank.pub.values()),('private_cases',bank.cases.values()),('images',bank.img.values()),('training_groups',train),('evaluation_manifest',ev),('clarification_manifest',dev),('smoke_groups',smoke)]:put(dest/(file+'.jsonl'),rows)
    doses={}
    for s in SEEDS:
        g=[x for x in train if x['train_seed']==s and x['parent']=='b0' and x['arm']=='G'];d=[x for x in train if x['train_seed']==s and x['parent']=='b0' and x['arm']=='D']
        assert len(g)==len(d)==384 and all(x['root_id']==y['root_id'] for x,y in zip(g,d))
        counter={}
        for ch in ('LIST','REPORT'):
            ids=[(x,y) for x,y in zip(g,d) if x['channel']==ch]; diff=[x['step'] for x,y in ids if x['input_id']!=y['input_id']]
            counter[ch]=dict(slots=len(ids),G_conflicts=sum(bank.cases[x['case_id']]['audit']['list_conflict'] for x,y in ids),D_conflicts=sum(bank.cases[y['case_id']]['audit']['list_conflict'] for x,y in ids),different_inputs=len(diff),different_steps=sorted(set(diff)))
        doses[str(s)]=counter
    import PIL
    manifest=dict(version='SR-ORD-TRAIN-1.0',counts=counts,doses=doses,images_rendered=images,font_sha256=sha(Path(font).read_bytes()) if images else None,pillow=PIL.__version__,files={x.name:sha(x.read_bytes()) for x in sorted(dest.glob('*.jsonl'))})
    (dest/'FREEZE.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n');return manifest
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--font',default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');ap.add_argument('--no-images',action='store_true');a=ap.parse_args();print(json.dumps(construct(a.out,a.font,not a.no_images),ensure_ascii=False,indent=2))
