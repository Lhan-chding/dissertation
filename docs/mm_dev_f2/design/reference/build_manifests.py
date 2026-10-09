"""Build CPU numeric-source and slot manifests from the supplied audited generator.
Does not render images, load a model, submit jobs or provide a GPU training loop.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json
from pathlib import Path
from plan_contract import PLAN_ID, canonical, digest, seed, run_matrix, slots, seed_usage, allocation_counts

POOLS={'ENGINE_F2':4,'PREP':32,'CONTINUE':64,'PROBE':64,'DEV_EVAL':128}
GENERATOR_SHA='fd188556668a165498032544fd90316c515e04b449fa44f80b509f8dc505984d'


def main(source:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    p=source/'src/mm_core/generator.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==GENERATOR_SHA, 'audited generator identity differs'
    spec=importlib.util.spec_from_file_location('audited_generator',p); gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
    banned=collections.defaultdict(set); triples=set(); graphs=set()
    for x in map(json.loads,(source/'audit/data/questions.jsonl').read_text().splitlines()):
        v=x['true_values']
        if len(v)>=2: banned[x['D']].add(tuple(v[:2]))
        if len(v)==3: triples.add(tuple(sorted(v)))
    prior_banned={k:len(v) for k,v in banned.items()}
    sources=[]; rejects=collections.Counter()
    for pool,n in POOLS.items():
        for root in range(n):
            family=f'mmdev-f2-{pool.lower()}-r{root:04d}'
            for D in ('low','high'):
                for attempt in range(4096):
                    rngseed=seed('numeric-source-candidate',pool,root,D,attempt)
                    graph,receipt=gen._numeric_source(rngseed,family,D,20000)
                    if graph is None: rejects['constructor_exhausted']+=1;continue
                    v=[graph['values'][0][c] for c in (0,2,4)]
                    p2=tuple(v[:2]);t3=tuple(sorted(v)); gh=digest(graph)
                    if p2 in banned[D]: rejects['ordered_pair_collision']+=1;continue
                    if t3 in triples: rejects['range_operand_set_collision']+=1;continue
                    if gh in graphs: rejects['full_source_collision']+=1;continue
                    banned[D].add(p2);triples.add(t3);graphs.add(gh)
                    sources.append({'plan_id':PLAN_ID,'pool':pool,'root_family_id':family,'root_index':root,'numeric_level':D,'source_seed':rngseed,'candidate_index':attempt,'original_constructor_receipt':receipt,'source_graph_hash_f2':gh,'truth_targets':v,'source':graph})
                    break
                else: raise RuntimeError(f'predeclared numeric construction failed: {family} {D}')
    (out/'numeric_sources.jsonl').write_bytes(b''.join(canonical(s)+b'\n' for s in sources))
    schedules={'ENGINE_e0':slots('ENGINE_F2','a0',0,4,steps=4)}
    for rep in (0,1):
        schedules[f'PREP_p{rep}']=slots('PREP','a0',rep,32)
        for action in ('a0','aP','aC'):
            schedules[f'CONT_f{rep}_{action}']=slots('CONTINUE',action,rep,64)
    sd=out/'schedules';sd.mkdir(exist_ok=True)
    for name,seq in schedules.items(): (sd/f'{name}.jsonl').write_bytes(b''.join(canonical(x)+b'\n' for x in seq))
    (out/'run_matrix.json').write_text(json.dumps(run_matrix(),ensure_ascii=False,indent=2)+'\n')
    (out/'seed_registry.json').write_text(json.dumps(seed_usage(),ensure_ascii=False,indent=2)+'\n')
    report={'status':'CPU_MANIFESTS_BUILT','model_calls':0,'images_rendered':0,'source_generator_sha256':GENERATOR_SHA,'root_counts':POOLS,'numeric_sources':len(sources),'root_families':sum(POOLS.values()),'questions_after_materialization':sum(POOLS.values())*24,'images_after_materialization':sum(POOLS.values())*8,'source_rejections_by_definition_only':dict(rejects),'historical_ordered_pair_exclusions':prior_banned,'source_policy':'no current or historical model outcomes used for construction; exact ordered pairs and sorted three-operand sets excluded globally','scientific_domain':'same audited renderer/domain/target query contract, new sources; additional deterministic numeric-content de-duplication disclosed','schedules':{k:{'slots':len(v),'sha256':hashlib.sha256((sd/f'{k}.jsonl').read_bytes()).hexdigest()} for k,v in schedules.items()},'counts':allocation_counts()}
    report['numeric_sources_sha256']=hashlib.sha256((out/'numeric_sources.jsonl').read_bytes()).hexdigest()
    (out/'MANIFEST_BUILD_RECEIPT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_root',type=Path);p.add_argument('output_config',type=Path);a=p.parse_args();main(a.source_root,a.output_config)
