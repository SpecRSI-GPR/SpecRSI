import importlib.util
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from specrsi.learning.heads import TinyRoleModel,TinyModelConfig
from specrsi.learning.records import DecisionRecord,SearchTrajectory,TransitionRecord


def test_additive_scores_replay_exact_saved_items():
    model=TinyRoleModel(TinyModelConfig())
    a=(1.,0.,0.,.1,1.,0.,0.,0.);b=(0.,1.,0.,.2,1.,0.,0.,0.)
    d=DecisionRecord('batch',('a','ab'),0,0.,state_features=(0.,)*8,option_features=(a,b),
                     option_item_features=((a,),(a,b)))
    out=model.decision_values(d)
    hidden=model.encode(d.state_features, "value")
    single=model.batch_head(hidden,torch.tensor([a,b]))
    assert torch.allclose(out,torch.stack([single[0],single.sum()]))
    # Union summary changes cannot create a hidden complementarity term.
    assert torch.equal(out,model.decision_values(replace(d,option_features=(b,a))))


def test_item_reward_needs_observed_records_and_keeps_complete_return():
    d=DecisionRecord('outer',('a',),0,0.,transition_index=0)
    t=SearchTrajectory('g','i','s',1,(TransitionRecord(0.,0.,2.,.7),TransitionRecord(0.,1.,3.,.8)),(d,),0.,1.)
    assert t.item_return(.1)==pytest.approx(1.)
    assert t.full_return(.1)==pytest.approx(.5)
    assert t.item_return(.1,d)==pytest.approx(1.)
    missing=replace(t,transitions=(TransitionRecord(0.,0.,2.),TransitionRecord(0.,1.,3.)))
    with pytest.raises(ValueError,match='recorded'):missing.item_return(.1)


def test_verified_edit_rejects_incomplete_child_and_changed_prefix():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_training.py'
    spec=importlib.util.spec_from_file_location('prepare_training',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    d=DecisionRecord('edit_token',('12',),0,0.,input_ids=(10,11))
    trajectory=SearchTrajectory('g','i','s',1,(TransitionRecord(0,0,1),),(d,),lineage_id='development')
    raw={'acceptance':{'complete_candidates':[]},'transitions':[{'index':0,'events':[{'event':'freeze','candidate_id':'child'}]}]}
    assert module.verified_edits(raw,trajectory)==[]
    raw['acceptance']['complete_candidates']=['child']
    edit=module.verified_edits(raw,trajectory)[0]
    assert edit.input_ids==(10,11,12) and edit.labels==(-100,-100,12)
    bad=replace(trajectory,decisions=(d,replace(d,input_ids=(10,99))))
    with pytest.raises(ValueError,match='discontinuous'):module.verified_edits(raw,bad)
