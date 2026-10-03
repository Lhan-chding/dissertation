import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
s=importlib.util.spec_from_file_location('pilot',Path(__file__).parents[1]/'scripts/prospective_selection/pilot_four.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_alpha_practical_tie_chooses_stronger_regularization():
    assert m.alpha_choice({.0001:.8,.01:.7995,10:.798})==.01

def test_both_anchors_stay_in_same_outer_fold():
    p=[SimpleNamespace(lineage_id=str(l),origin_id=f'{l}_{t}') for l in range(4) for t in (32,96)]
    fit,held=m.split_lineages(p,'2')
    assert len(fit)==6 and {x.origin_id for x in held}=={'2_32','2_96'}
    assert not {x.lineage_id for x in fit}&{x.lineage_id for x in held}

def test_outer_labels_cannot_enter_fit_api():
    train=[SimpleNamespace(lineage_id=str(l),origin_id=f'{l}_{t}') for l in range(3) for t in (32,96)]
    held=[SimpleNamespace(lineage_id='3',origin_id='3_32')]
    labels={p.origin_id:{} for p in train+held}
    with pytest.raises(ValueError,match='Outer held labels'):m.fit_outer(train,labels,held)

def test_inner_lineage_overlap_is_rejected_before_fit():
    fit=[SimpleNamespace(lineage_id='1',origin_id='1_32')]
    held=[SimpleNamespace(lineage_id='1',origin_id='1_96')]
    with pytest.raises(ValueError,match='holdout overlap'):m.grid_predictions(fit,{'1_32':{}},held)

def test_partial_decisions_are_never_overwritten(tmp_path):
    (tmp_path/'decisions_before_scoring.json').write_text('prior evidence')
    with pytest.raises(ValueError,match='partial or complete'):m.run(tmp_path/'absent.json',tmp_path)
    assert (tmp_path/'decisions_before_scoring.json').read_text()=='prior evidence'
