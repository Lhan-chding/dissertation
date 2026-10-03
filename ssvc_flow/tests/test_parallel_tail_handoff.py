import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE=Path(__file__).parents[1]/'scripts/prospective_selection'
def load(name):
    spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c=load('server_controller');h=load('parallel_tail_handoff')

def fixture(tmp_path,monkeypatch):
    tasks={'r6':{'key':['61004_t96','R6']},'gdpo':{'key':['61004_t96','GDPO_R4']}}
    cfg={'control':str(tmp_path),'targets':list(tasks),'lanes':{'2':['gdpo','r6'],'3':['done']}}
    reg=SimpleNamespace(root=tmp_path,assert_can_execute=lambda t:tasks[t])
    monkeypatch.setattr(c,'inspect_scope',lambda *_:tasks)
    monkeypatch.setattr(c,'reviewed_records',lambda *_:{'done':{}})
    monkeypatch.setattr(c,'active_jobs',lambda:[{'job_id':'1'}])
    monkeypatch.setattr(c,'lane_allocations',lambda _:[{'job_id':'1','lane':2},{'job_id':'2','lane':3}])
    monkeypatch.setattr(c,'accounting',lambda _: {'2':['2','COMPLETED','0:0']})
    return cfg,reg

def test_only_unattempted_tail_can_be_reserved(tmp_path,monkeypatch):
    cfg,reg=fixture(tmp_path,monkeypatch)
    assert h.validate_reservation(c,cfg,reg)=='r6'
    (tmp_path/'intents').mkdir();(tmp_path/'intents/r6.json').write_text('{}')
    with pytest.raises(ValueError,match='already attempted'):h.validate_reservation(c,cfg,reg)

@pytest.mark.parametrize('marker',['STOP','BLOCKED.json'])
def test_existing_stop_or_failure_cannot_be_overridden(tmp_path,monkeypatch,marker):
    cfg,reg=fixture(tmp_path,monkeypatch);(tmp_path/marker).touch()
    with pytest.raises(ValueError,match='existing'):h.validate_reservation(c,cfg,reg)

def test_failed_old_allocation_and_full_capacity_rejected(tmp_path,monkeypatch):
    cfg,reg=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(c,'accounting',lambda _: {'2':['2','FAILED','1:0']})
    with pytest.raises(ValueError,match='failed old'):h.validate_reservation(c,cfg,reg)
    monkeypatch.setattr(c,'active_jobs',lambda:[{'job_id':str(i)} for i in range(5)])
    with pytest.raises(ValueError,match='capacity'):h.validate_reservation(c,cfg,reg)

def test_race_refusal_does_not_block_healthy_original_lane(tmp_path,monkeypatch):
    blocks=[]
    monkeypatch.setattr(c,'publish_block',lambda *args:blocks.append(args))
    h.report_failure(c,tmp_path,ValueError('R6 already attempted'),True)
    assert blocks==[]
    (tmp_path/'TAIL_HANDOFF.json').write_text('{}')
    h.report_failure(c,tmp_path,ValueError('unknown submission'),True)
    assert len(blocks)==1
