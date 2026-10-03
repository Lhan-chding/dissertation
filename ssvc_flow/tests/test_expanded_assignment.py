import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('expanded_setup',Path(__file__).parents[1]/'scripts/prospective_selection/prepare_expanded_pipelines.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_fixed_assignment_has_no_cross_lane_dependency_or_duplicate():
    tasks={};specs=[{'seed':i} for i in range(12)]
    for s in specs:
        lid=s['seed']
        for n in range(25):
            tid=f'{lid}-{n}';kind='source' if n==0 else 'prestate' if n<3 else 'branch'
            tasks[tid]={'task_id':tid,'kind':kind,'payload':{'lineage_id':lid,'origin_step':32 if n%2 else 96,'recipe_id':str(n)}}
    lanes=m.assign(tasks,specs)
    assert list(map(len,lanes.values()))==[75,75,50,50,50]
    assert sum(map(len,lanes.values()))==len(tasks)
    for lid in range(12):
        lane=next(v for v in lanes.values() if f'{lid}-0' in v)
        subset=[t for t in lane if tasks[t]['payload']['lineage_id']==lid]
        assert [tasks[t]['kind'] for t in subset[:3]]==['source','prestate','prestate']
    tasks.pop('0-24')
    with pytest.raises(ValueError):m.assign(tasks,specs)
