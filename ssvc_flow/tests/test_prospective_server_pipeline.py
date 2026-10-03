"""No GPU: fixed ownership, immediate next task, and fail-closed handoff."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

FILE=Path(__file__).parents[1]/'scripts/prospective_selection/server_controller.py'
spec=importlib.util.spec_from_file_location('pipeline_control',FILE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_bound_rejects_corruption_and_external_path(tmp_path):
    p=tmp_path/'rows';p.write_text('ok')
    binding={'path':str(p),'sha256':c.digest(p)}
    assert c.bound(binding,tmp_path)==b'ok'
    p.write_text('bad')
    with pytest.raises(ValueError,match='hash'):c.bound(binding,tmp_path)
    with pytest.raises(ValueError,match='outside'):c.bound(binding,tmp_path/'narrow')


def test_summary_recompute_detects_metric_change():
    assert c.compare({'J':.5},{'J':.5+1e-15})<1e-12
    with pytest.raises(ValueError):c.compare({'J':.5},{'J':.51})
    with pytest.raises(ValueError):c.compare({'J':float('nan')},{'J':float('nan')})


def setup_claim(tmp_path,monkeypatch):
    for x in ('intents','submissions','completed'):(tmp_path/x).mkdir()
    tasks={'a':{'task_id':'a','kind':'branch','key':['a']},'b':{'task_id':'b','kind':'branch','key':['b']}}
    cfg={'root':str(tmp_path),'control':str(tmp_path),'lanes':{'1':['a'],'2':['b']},'targets':['a','b']}
    reg=SimpleNamespace(root=tmp_path,_lock=nullcontext,assert_can_execute=lambda tid:tasks[tid])
    monkeypatch.setattr(c,'inspect_scope',lambda *_:tasks)
    monkeypatch.setattr(c,'reviewed_records',lambda *_:{})
    monkeypatch.setattr(c,'lane_allocations',lambda *_:[{'job_id':'1','lane':1},{'job_id':'2','lane':2}])
    monkeypatch.setattr(c,'active_jobs',lambda:[{'job_id':str(i),'gpus':1} for i in (1,2)])
    return cfg,reg


def test_fixed_lane_cannot_claim_another_or_repeat(tmp_path,monkeypatch):
    cfg,reg=setup_claim(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='assigned'):c.claim_task(cfg,reg,'1',1,'b')
    assert c.claim_task(cfg,reg,'1',1,'a')['task_id']=='a'
    assert json.loads((tmp_path/'submissions/a.json').read_text())['job_id']=='1'
    with pytest.raises(ValueError,match='attempted'):c.claim_task(cfg,reg,'1',1,'a')


def test_stop_marker_prevents_new_execution(tmp_path,monkeypatch):
    cfg,reg=setup_claim(tmp_path,monkeypatch)
    (tmp_path/'STOP').touch()
    assert c.claim_task(cfg,reg,'1',1,'a') is None
    assert not (tmp_path/'intents/a.json').exists()


def test_pipeline_executes_fixed_order_without_poll_or_sbatch(tmp_path,monkeypatch):
    (tmp_path/'allocations').mkdir();c.atomic(tmp_path/'allocations/x.submitted.json',{'job_id':'1','lane':1})
    cfg={'control':str(tmp_path),'lanes':{'1':['a','b']},'python':'python','project_root':str(tmp_path),'runtime':'runtime'}
    reg=SimpleNamespace(root=tmp_path)
    events=[]
    monkeypatch.setenv('SLURM_JOB_ID','1')
    monkeypatch.setattr(c,'claim_task',lambda cfg,reg,job,lane,tid:{'task_id':tid,'key':[tid]})
    monkeypatch.setattr(c.subprocess,'run',lambda cmd,**kw:events.append(('execute',cmd[cmd.index('--task')+1])) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(c,'audit_completed',lambda cfg,t,e:events.append(('audit',t['task_id'])) or {'J':.5})
    monkeypatch.setattr(c.time,'sleep',lambda _:pytest.fail('must not wait between experiments'))
    c.pipeline(cfg,reg,1)
    assert [x[0] for x in events]==['execute','audit','execute','audit']
    assert events[1][1]=='a' and events[3][1]=='b'


def test_failed_worker_never_runs_next_task(tmp_path,monkeypatch):
    (tmp_path/'allocations').mkdir();c.atomic(tmp_path/'allocations/x.submitted.json',{'job_id':'1','lane':1})
    cfg={'control':str(tmp_path),'lanes':{'1':['a','b']},'python':'python','project_root':str(tmp_path),'runtime':'runtime'}
    monkeypatch.setenv('SLURM_JOB_ID','1')
    seen=[]
    monkeypatch.setattr(c,'claim_task',lambda cfg,reg,job,lane,tid:seen.append(tid) or {'task_id':tid,'key':[tid]})
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    with pytest.raises(ValueError,match='worker failed'):c.pipeline(cfg,SimpleNamespace(root=tmp_path),1)
    assert seen==['a']


def test_unknown_allocation_cannot_be_resubmitted(tmp_path):
    (tmp_path/'allocations').mkdir();c.atomic(tmp_path/'allocations/x.intent.json',{'lane':1})
    with pytest.raises(ValueError,match='unknown'):c.lane_allocations({'control':str(tmp_path)})


def test_handoff_audits_before_starting_lane_and_stops_on_failure(tmp_path,monkeypatch):
    (tmp_path/'intents').mkdir();(tmp_path/'intents/a.json').write_text('{}')
    task={'task_id':'a'};cfg={'root':str(tmp_path),'control':str(tmp_path),'targets':['a','b'],
                            'lanes':{'1':['b'],'2':[],'3':[],'4':[],'5':[]}}
    registry=SimpleNamespace(root=tmp_path,_lock=nullcontext)
    reviewed={};events=[]
    monkeypatch.setattr(c,'inspect_scope',lambda *_:{'a':task})
    monkeypatch.setattr(c,'reviewed_records',lambda *_:dict(reviewed))
    monkeypatch.setattr(c,'active_jobs',lambda:[])
    monkeypatch.setattr(c,'lane_allocations',lambda *_:[])
    monkeypatch.setattr(c,'latest_submission',lambda *_:'100')
    monkeypatch.setattr(c,'accounting',lambda *_:{'100':['100','FAILED','1:0']})
    monkeypatch.setattr(c,'launch_lane',lambda *a:events.append('launch') or {'job_id':'200'})
    with pytest.raises(ValueError,match='failed/unknown'):c.handoff_tick(cfg,registry,tmp_path/'config')
    assert events==[]
    monkeypatch.setattr(c,'accounting',lambda *_:{'100':['100','COMPLETED','0:0']})
    def audit(*a):
        events.append('audit');reviewed['a']={'kind':'branch'}
    monkeypatch.setattr(c,'audit_completed',audit)
    result=c.handoff_tick(cfg,registry,tmp_path/'config',submit=False)
    assert events==['audit'] and result['verified_branches']==1


def test_handoff_counts_pending_gpus_and_never_overfills(tmp_path,monkeypatch):
    (tmp_path/'intents').mkdir()
    for i in range(4):(tmp_path/'intents'/f't{i}.json').write_text('{}')
    cfg={'root':str(tmp_path),'control':str(tmp_path),'targets':['x'],
         'lanes':{str(i):[str(i)] for i in range(1,6)}}
    registry=SimpleNamespace(root=tmp_path,_lock=nullcontext)
    monkeypatch.setattr(c,'inspect_scope',lambda *_:{})
    monkeypatch.setattr(c,'reviewed_records',lambda *_:{})
    monkeypatch.setattr(c,'lane_allocations',lambda *_:[])
    monkeypatch.setattr(c,'latest_submission',lambda root,tid:tid[1:])
    monkeypatch.setattr(c,'active_jobs',lambda:[{'job_id':str(i),'gpus':1} for i in range(4)])
    monkeypatch.setattr(c,'accounting',lambda *_:{})
    monkeypatch.setattr(c,'teacher_qos_jobs',lambda:set())
    monkeypatch.setattr(c,'launch_lane',lambda *a:{'job_id':'10','lane':a[-1]})
    result=c.launch_only(cfg,registry,tmp_path/'config')
    assert len(result['launched_lanes'])==1


def expanded_fixture(tmp_path):
    from src.prospective_selection.protocol import load_protocol
    from src.prospective_selection.orchestration import register_development
    from src.prospective_selection.jobs import TaskRegistry
    protocol=load_protocol(Path(__file__).parents[1]/'docs/prospective_selection/design/protocol.json')
    c.atomic(tmp_path/'FIRST_FOUR_DELIVERED.json',{'status':'FIRST_FOUR_DELIVERED',
        'lineages':[61001,61002,61003,61004],'verified_branches':88,'verified_historical_E':16})
    prepared={'branch_schedules':{'1':{'schedule_id':'continuation-1-'+'a'*16,
        'sampler_seed':protocol['training']['branch_schedule_seeds'][0]}}}
    register_development(protocol,tmp_path,prepared,first_four=False)
    reg=TaskRegistry(tmp_path,protocol); tasks={t['task_id']:t for t in reg.tasks()}
    allowed=[s['seed'] for s in protocol['origins']['development'][4:]+protocol['origins']['tuning']]
    baseline={t:{} for t,v in tasks.items() if v['payload'].get('lineage_id') not in allowed}
    targets=set(tasks)-baseline.keys(); lanes={str(i):[] for i in range(1,6)}
    for i,lid in enumerate(allowed):
        ordered=sorted((t for t in tasks.values() if t['payload'].get('lineage_id')==lid),
            key=lambda t:({'source':0,'prestate':1,'branch':2}[t['kind']],
                          t['payload'].get('origin_step',0),t['payload'].get('recipe_id','')))
        lanes[str(i%5+1)].extend(t['task_id'] for t in ordered)
    cfg={'phase':'development-tuning','root':str(tmp_path),'control':str(tmp_path),
         'first_four_delivery':{'path':str(tmp_path/'FIRST_FOUR_DELIVERED.json'),
                                'sha256':c.digest(tmp_path/'FIRST_FOUR_DELIVERED.json')},
         'allowed_lineages':allowed,'targets':list(targets),'baseline':baseline,
         'inflight_at_setup':[],'lanes':lanes}
    return cfg,reg,tasks


def test_expanded_phase_requires_exact_matrix_delivery_and_local_dependencies(tmp_path):
    import copy
    cfg,reg,tasks=expanded_fixture(tmp_path)
    assert len(cfg['targets'])==300
    c.validate_phase(cfg,reg,tasks)
    bad=copy.deepcopy(cfg);bad['targets'].pop()
    with pytest.raises(ValueError,match='old tasks|matrix'):c.validate_phase(bad,reg,tasks)
    bad=copy.deepcopy(cfg);bad['lanes']['1'].reverse()
    with pytest.raises(ValueError,match='dependency'):c.validate_phase(bad,reg,tasks)
    bad=copy.deepcopy(cfg);bad['allowed_lineages'][0]=63001
    with pytest.raises(ValueError,match='whitelist'):c.validate_phase(bad,reg,tasks)
    (tmp_path/'FIRST_FOUR_DELIVERED.json').write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):c.validate_phase(cfg,reg,tasks)


def test_expanded_task_rejects_test_role_and_extra_repeat(tmp_path):
    import copy
    cfg,reg,tasks=expanded_fixture(tmp_path)
    task=next(t for t in tasks.values() if t['task_id'] in cfg['targets'] and t['kind']=='branch')
    bad=copy.deepcopy(task);bad['payload']['role']='locked_test'
    with pytest.raises(ValueError,match='role'):c.validate_expanded_task(cfg,reg.protocol,bad)
    bad=copy.deepcopy(task);bad['payload']['repeat']=2
    with pytest.raises(ValueError,match='repeat'):c.validate_expanded_task(cfg,reg.protocol,bad)


def source_fixture(root,task):
    from src.modeling_v3.io import canonical_hash
    lid=task['payload']['lineage_id'];path=root/'sources'/str(lid);path.mkdir(parents=True)
    m={'kind':'PROSPECTIVE_TRAINING','fixture':False,'runtime_identity':{'execution_kind':'REAL_CUDA_MODEL'},
       'steps':96,'initial_state_hash':'initial',
       'identity':{'lineage_id':lid,'source_recipe':task['payload']['source_recipe'],
                   'role':'source','schedule_id':'source-schedule'}}
    c.atomic(path/'MANIFEST.json',m);mh=canonical_hash(m)
    def checkpoint(step):
        f=path/f'H{step}.pt';f.write_bytes(b'fixture opaque checkpoint')
        return {'path':str(f),'state_hash':'initial' if step==0 else str(step),
                'identity':{'manifest_hash':mh,'step':step}}
    checkpoints={'0':checkpoint(0)};commits=[]
    for start in range(1,97,8):
        stop=start+7;segment=path/'segments'/str(start);segment.mkdir(parents=True)
        rows=[{'origin_id':f'source_{lid}','recipe':task['payload']['source_recipe'],'repeat':0,
               'role':'train','schedule_id':'source-schedule','sample_key':f'{start}:{i}'} for i in range(256)]
        f=segment/'samples.jsonl';f.write_text('\n'.join(json.dumps(r) for r in rows))
        updates=[]
        for step in range(start,stop+1):
            u=segment/f'update_{step}.json';c.atomic(u,{'step':step,'loss':.1,'grad_norm_preclip':.2})
            updates.append({'path':str(u),'sha256':c.digest(u)})
        cp=checkpoint(stop)
        if stop in (24,32,88,96):checkpoints[str(stop)]=cp
        commit=segment/'COMMIT.json';c.atomic(commit,{'manifest_hash':mh,'start':start,'stop':stop,
            'training_outputs':256,'samples':{'path':str(f),'sha256':c.digest(f)},'updates':updates,'checkpoint':cp})
        commits.append(str(commit))
    for step,cp in checkpoints.items():c.atomic(path/f'H{int(step):02d}.json',{'step':int(step),'checkpoint':cp})
    c.atomic(path/'COMPLETE.json',{'status':'TRAINING_COMPLETE','identity':m['identity'],
        'steps':96,'training_outputs':3072,'checkpoints':checkpoints,'checkpoint':checkpoints['96'],
        'authoritative_segments':commits})
    return path


def test_source_audit_checks_raw_updates_and_anchor_binding(tmp_path):
    cfg,reg,tasks=expanded_fixture(tmp_path)
    task=next(t for t in tasks.values() if t['task_id'] in cfg['targets'] and t['kind']=='source')
    path=source_fixture(tmp_path,task)
    assert c.verify_task(tmp_path,task,cfg)['finite_updates']==96
    checkpoint=c.read(path/'H32.json');checkpoint['checkpoint']['state_hash']='wrong'
    c.atomic(path/'H32.json',checkpoint)
    with pytest.raises(ValueError,match='anchor'):c.verify_task(tmp_path,task,cfg)
    u=next(path.glob('segments/*/update_1.json'));u.write_text('{}')
    with pytest.raises(ValueError,match='hash'):c.verify_task(tmp_path,task,cfg)


def test_source_and_prestate_audits_can_advance_without_J(tmp_path,monkeypatch):
    (tmp_path/'allocations').mkdir();c.atomic(tmp_path/'allocations/x.submitted.json',{'job_id':'1','lane':1})
    cfg={'control':str(tmp_path),'lanes':{'1':['source','prestate','branch']},
         'python':'python','project_root':str(tmp_path),'runtime':'runtime'}
    seen=[];monkeypatch.setenv('SLURM_JOB_ID','1')
    monkeypatch.setattr(c,'claim_task',lambda cfg,reg,job,lane,tid:{'task_id':tid,'key':[tid]})
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    monkeypatch.setattr(c,'audit_completed',lambda cfg,t,e:seen.append(t['task_id']) or {'kind':t['task_id']})
    c.pipeline(cfg,SimpleNamespace(root=tmp_path),1)
    assert seen==['source','prestate','branch']
    assert c.complete_status({'phase':'development-tuning'})=='DEVELOPMENT_TUNING_COLLECTED_AWAITING_ANALYSIS'
    assert c.complete_status({})=='FIRST_FOUR_COLLECTED_AWAITING_ANALYSIS'


def test_prestate_reconstructs_packets_from_both_verified_raw_snapshots(tmp_path,monkeypatch):
    from src.prospective_selection.features import LEVELS,build_predecision_packet
    from src.prospective_selection import orchestration
    cfg,reg,tasks=expanded_fixture(tmp_path)
    task=next(t for t in tasks.values() if t['task_id'] in cfg['targets'] and t['kind']=='prestate')
    p=task['payload'];step=p['origin_step'];origin=p['origin_id'];lid=p['lineage_id']
    path=tmp_path/'prestate'/origin;path.mkdir(parents=True)
    source=tmp_path/'sources'/str(lid);source.mkdir(parents=True)
    rows=[dict(prompt_id=str(i),family='trend',interface='SYMBOLIC_FRESH',event='S',
               relation_numerator=1,relation_denominator=2,F=0,B=j%2,M=j%2)
          for i in range(72) for j in range(32)]
    raw=[{**{k:r[k] for k in ('prompt_id','family','interface')},
          'semantic':{k:r[k] for k in ('event','relation_numerator','relation_denominator','F','B','M')}} for r in rows]
    calls=[]
    def verify(path,expected,prompts,draws,state_hash,**kwargs):
        calls.append((expected['horizon'],state_hash))
        assert prompts==72 and draws==32 and expected['repeat']==0 and kwargs['include_rows']
        return {'rows':raw}
    monkeypatch.setattr(c,'verify_evaluation',verify)
    monkeypatch.setattr(orchestration,'_metadata',lambda *_:{})
    for at in (step-8,step):c.atomic(source/f'H{at:02d}.json',{'checkpoint':{'state_hash':str(at)}})
    for level in LEVELS:
        packet=build_predecision_packet(origin_id=origin,lineage_id=str(lid),source_recipe=p['source_recipe'],
             step=step,current_rows=rows,history_rows=rows,feature_level=level).to_dict()
        c.atomic(path/f'{level}.json',packet)
    c.atomic(path/'COMPLETE.json',{'status':'PRESTATE_COMPLETE','origin_id':origin,
             'shared_generated_outputs':4608,'four_levels_same_samples':True})
    assert c.verify_task(tmp_path,task,cfg)['raw_packet_reconstruction']
    assert calls==[(step-8,str(step-8)),(step,str(step))]
    filename=path/f'{LEVELS[0]}.json';packet=c.read(filename);packet['known_training_metadata']={'loss_mean':.9}
    c.atomic(filename,packet)
    with pytest.raises(ValueError,match='fields differ'):c.verify_task(tmp_path,task,cfg)


def test_unrelated_teacher_qos_job_reserves_its_submit_slot(tmp_path,monkeypatch):
    (tmp_path/'intents').mkdir()
    cfg={'root':str(tmp_path),'control':str(tmp_path),'targets':['a','b','c','d','e'],
         'lanes':{str(i):[chr(96+i)] for i in range(1,6)}}
    reg=SimpleNamespace(root=tmp_path,_lock=nullcontext)
    monkeypatch.setattr(c,'inspect_scope',lambda *_:{})
    monkeypatch.setattr(c,'reviewed_records',lambda *_:{})
    monkeypatch.setattr(c,'lane_allocations',lambda *_:[])
    monkeypatch.setattr(c,'active_jobs',lambda:[])
    monkeypatch.setattr(c,'accounting',lambda *_:{})
    monkeypatch.setattr(c,'teacher_qos_jobs',lambda:{'unrelated'})
    monkeypatch.setattr(c,'launch_lane',lambda *a:{'job_id':str(a[-1]),'lane':a[-1]})
    result=c.launch_only(cfg,reg,tmp_path/'config')
    assert len(result['launched_lanes'])==4


def test_teacher_qos_query_includes_pending_and_unrelated_but_not_other_qos(monkeypatch):
    monkeypatch.setattr(c,'run',lambda cmd:'1|'+c.QOS+'\n2|other\n3|'+c.QOS)
    assert c.teacher_qos_jobs()=={'1','3'}
