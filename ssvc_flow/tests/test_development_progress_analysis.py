import importlib.util
from pathlib import Path

import pytest
from src.prospective_selection.features import LEVELS, SCHEMA, _snapshot

spec=importlib.util.spec_from_file_location('progress_analysis',Path(__file__).parents[1]/'scripts/prospective_selection/analyze_development_progress.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def packet_fixture():
    rows=[dict(prompt_id=str(i),family='trend',interface='SYMBOLIC_FRESH',event='S',
               relation_numerator=1,relation_denominator=2,F=0,B=j%2,M=j%2)
          for i in range(72) for j in range(32)]
    packets={level:dict(schema=SCHEMA,origin_id='61001_t32',lineage_id='61001',source_recipe='R0',
        step=32,history_step=24,feature_level=level,current=_snapshot(rows,level),
        history=_snapshot(rows,level),known_training_metadata={}) for level in LEVELS}
    return dict(origin='61001_t32',complete={'four_levels_same_samples':True},packets=packets)

def test_information_audit_keeps_prompt_and_reward_conditioning():
    rows=a.information_audit(packet_fixture())
    for r in rows:
        assert r['mixed_repair_buckets']==72
        assert r['same_reward_pairs']==72*(32*31//2)
        assert r['different_repair_pairs']==72*16*16
        assert r['reward_event_reconstruction_error']==0

def test_inconsistent_reward_projection_rejected():
    pre=packet_fixture()
    counts=pre['packets']['Z3_REPAIR_STRUCTURE']['current']['repair_histograms']['0']['counts']
    first=next(iter(counts));counts[first]+=1
    with pytest.raises(ValueError):a.information_audit(pre)


def test_external_baseline_never_enters_candidate_oracle():
    index={(o,r):0.5 for o in ('a','b') for r in a.RECIPES}
    index['a','R1']=0.7;index['b','R2']=0.7
    index['a','GDPO_R4']=0.99;index['b','GDPO_R4']=0.99
    result=a.candidate_comparison(index,['a','b'])
    assert result['descriptive_best_static_recipe'] in a.ACTIONS
    assert result['descriptive_best_static_mean']==0.6
    assert result['descriptive_hindsight_oracle_mean']==0.7
    assert result['descriptive_oracle_gap_pp']==pytest.approx(10)


def test_compact_export_handles_ssh_banner_and_preserves_snapshot(monkeypatch,tmp_path):
    import base64
    import gzip
    import json
    import sys
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('progress_export',Path(__file__).parents[1]/'scripts/prospective_selection/export_development_progress.py')
    exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    data=dict(branches=[],historical_E=[],prestates=[],missing=[])
    encoded=base64.b64encode(gzip.compress(json.dumps(data).encode()))
    def run(args,**kwargs):
        remote=args[-1].split("\n",1)[1].rsplit('\nSSVC_EXPORT_SCRIPT',1)[0]
        compile(remote,'remote_export','exec')
        assert 'for segment in item[' not in remote
        assert "b.pop('training')" in remote
        return SimpleNamespace(returncode=0,stdout=b'Cluster banner\nSSVC_GZIP_BASE64\n'+encoded+b'\n')
    monkeypatch.setattr(exporter.subprocess,'run',run)
    output=tmp_path/'snapshot.json'
    monkeypatch.setattr(sys,'argv',['export','--metadata-only','--compact','--out',str(output)])
    exporter.main()
    assert json.loads(output.read_text())==data
    with pytest.raises(ValueError,match='already exists'):exporter.main()
