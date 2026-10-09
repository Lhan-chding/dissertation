"""Independent text/sidecar audit. No model weights or model calls are used."""
from __future__ import annotations
import argparse, collections, hashlib, json
from fractions import Fraction
from pathlib import Path


def q(x):
    if isinstance(x, bool): raise ValueError('bool is not a number')
    return Fraction(str(x))


def op(v, name):
    return sum(v) if name == 'sum' else v[0]-v[1] if name == 'difference' else max(v)-min(v)


def audit(root: Path):
    rows = {x['question_id']: x for x in map(json.loads, (root/'audit/data/questions.jsonl').read_text().splitlines())}
    scores = {x['request_id']: x for x in map(json.loads, (root/'audit/scoring/MEASUREMENT_AUDIT/SCORED_OUTPUTS.jsonl').read_text().splitlines())}
    counts=collections.Counter(); strata={}; err_support=set(); shape=collections.Counter(); bad=[]; unique=set(); mixed=collections.defaultdict(list); range_signature=collections.Counter()
    for path in sorted((root/'audit/raw/MEASUREMENT_AUDIT').glob('outputs_*.jsonl')):
        for line in path.read_text().splitlines():
            x=json.loads(line); rid=x['request_id']; row=rows[x['question_id']]; s=scores[rid]
            if rid in unique: raise ValueError('duplicate request')
            unique.add(rid)
            raw=json.loads(x['raw_text'], parse_float=str, parse_int=str)
            v=list(map(q,row['true_values'])); r=list(map(q,raw['readings'])); a=q(raw['answer'])
            P=r==v; C=a==op(r,row['operation']); A=a==op(v,row['operation']); cell=f'{int(P)}{int(C)}{int(A)}'
            counts.update({'N':1,'P':int(P),'A':int(A),'J':int(P and A),cell:1})
            for key, value in [('P',P),('C',C),('A',A),('cell_exact',cell)]:
                if s[key] != value: bad.append([rid,key])
            e=[(ri-vi)/q(row['delta']) for ri,vi in zip(r,v)]; eP=(op(r,row['operation'])-op(v,row['operation']))/q(row['delta']); eC=(a-op(r,row['operation']))/q(row['delta']); eO=(a-op(v,row['operation']))/q(row['delta'])
            if eP+eC!=eO: bad.append([rid,'residual'])
            if not P:
                nz=sum(x!=0 for x in e); err_support.add(row['root_family_id']); S=(sum(e)/len(e))**2/(sum(x*x for x in e)/len(e)); shape[(len(e),nz,str(max(map(abs,e))),str(S))]+=1
            if P and not A and row['operation']=='range':
                range_signature.update({'P_and_not_A_range':1,'equals_first_minus_second':int(a==r[0]-r[1]),'equals_first_minus_third':int(a==r[0]-r[2]),'equals_second_minus_third':int(a==r[1]-r[2])})
            key=f"{row['chart_type']}/{row['operation']}"; strata.setdefault(key,collections.Counter()).update({'N':1,'P':int(P),'A':int(A),'J':int(P and A)})
            mixed[x['question_id']].append(int(A))
    assert len(unique)==len(scores)==1536
    assert not bad, bad[:5]
    zero=[]
    for branch in ['continuous','resumed']:
        for line in (root/f'audit/engineering/engine/{branch}/STEPS.jsonl').read_text().splitlines():
            x=json.loads(line); zero.append(x['gradient_norm'])
    return {'scope':'raw model text + truth sidecars; all audit measurement responses; no checkpoint verification', 'measurement_counts':dict(counts),'strata':{k:dict(v) for k,v in strata.items()},'mismatches':bad,'reward_mixed_questions':sum(len(set(v))>1 for v in mixed.values()),'question_count':len(mixed),'geometry_support_roots':len(err_support),'geometry_patterns':[{'arity':k[0],'nonzero_coordinates':k[1],'max_abs_error':k[2],'S':k[3],'count':v} for k,v in sorted(shape.items())], 'range_output_signatures':dict(range_signature),'original_engine_physical_steps':len(zero),'original_engine_all_zero_gradient':all(v==0 for v in zero),'checkpoint_bodies_verified':False,'new_model_calls':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive_directory',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
    text=json.dumps(audit(args.archive_directory),ensure_ascii=False,indent=2)
    if args.out: args.out.write_text(text+'\n')
    print(text)
