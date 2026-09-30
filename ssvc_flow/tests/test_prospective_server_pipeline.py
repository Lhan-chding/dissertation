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
    monkeypatch.setattr(c,'launch_lane',lambda *a:{'job_id':'10','lane':a[-1]})
    result=c.launch_only(cfg,registry,tmp_path/'config')
    assert len(result['launched_lanes'])==1
