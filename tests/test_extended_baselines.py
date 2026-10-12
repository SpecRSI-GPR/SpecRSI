from dataclasses import dataclass
import random

import numpy as np
import pytest

from specrsi.baselines.extended_common import Design,Evaluation,ExperimentBudget,Generated,ItemResult,RunContext
from specrsi.baselines.hgm import CladeNode,HGMConfig,clade_observations,run_hgm,thompson_select
from specrsi.baselines.aflow import AFlowConfig,aflow_probabilities,run_aflow
from specrsi.baselines.janus import ArithmeticFeature,ArchiveEntry,JanusConfig,promotion_meta_score,retain_archive,run_janus
from specrsi.baselines.emas import EMASConfig,Hypothesis,paired_accept,run_emas,validate_revision


@dataclass(frozen=True)
class Item:
    id:str
    prompt:str="test"


class NumericBackend:
    def __init__(self,values=(.3,.7,.2,.9,.1,.8,.4,.95)):
        self.values=list(values)
        self.proposals=0
        self.proxy_calls=0

    def evaluate(self,design,items,seed,stage,token_cap):
        value=float(design.payload["value"])
        assert len(items)*2<=token_cap
        return Evaluation(design.uid,tuple(ItemResult(x.id,max(0.,min(1.,value)),2,valid=value>=0) for x in items),seed,stage)

    def propose(self,parent,feedback,seed,token_cap):
        value=self.values[self.proposals%len(self.values)]
        self.proposals+=1
        return Generated(Design({"value":value,"tag":str(self.proposals)},parent.uid),1)

    def propose_proxy(self,parent,archive,feature_names,seed,token_cap):
        self.proxy_calls+=1
        return Generated({"name":"evolved","expressions":["value","value ** 2"]},1)


def context(backend,items=2,real=100):
    return RunContext(backend,[Item(f"v{i}") for i in range(items)],seed=0,budget=ExperimentBudget(100000,real))


def test_clade_productivity_includes_descendants_and_caps_own_pseudodata():
    weak=CladeNode(Design({"value":.1}),observed={str(i):ItemResult(str(i),.1,0) for i in range(100)})
    child=CladeNode(Design({"value":1}),weak.design.uid,observed={str(i):ItemResult(str(i),1.,0) for i in range(200)})
    strong=CladeNode(Design({"value":.7}),observed={str(i):ItemResult(str(i),.7,0) for i in range(100)})
    weak.children=[child.design.uid]
    nodes={x.design.uid:x for x in (weak,child,strong)}
    observations=clade_observations(weak,nodes,10)
    assert len(observations)==210 and sum(observations)==pytest.approx(201)
    rng=random.Random(0)
    selected=[thompson_select([weak,strong],lambda n:clade_observations(n,nodes,10),rng).design.uid for _ in range(100)]
    assert selected.count(weak.design.uid)>85


def test_hgm_partial_mean_cannot_win_final_complete_comparison():
    ctx=context(NumericBackend([1.]),items=3,real=4)
    initial=Design({"value":.4})
    result=run_hgm(ctx,initial,HGMConfig(max_steps=3,expansion_power=.1))
    assert result["selected"]["id"]==initial.uid
    assert result["budget"]["real_items"]<=4
    assert any(e["stage"]=="hgm_item" for e in ctx.events if e["event"]=="real_evaluation")


def test_aflow_distribution_matches_upstream_mixture():
    p=aflow_probabilities([.8,.2],.3,.2)
    assert sum(p)==pytest.approx(1)
    assert p[1]>=.15 and p[0]>.849
    assert aflow_probabilities([1.,0.],1.)==[.5,.5]


def test_aflow_selects_mean_repeats_not_luckiest_score():
    class Noisy(NumericBackend):
        def evaluate(self,design,items,seed,stage,token_cap):
            score=.6 if "tag" not in design.payload else (.1 if seed%2==0 else 1.)
            return Evaluation(design.uid,tuple(ItemResult(x.id,score,1) for x in items),seed,stage)
    ctx=context(Noisy([.9]))
    initial=Design({"value":.6})
    result=run_aflow(ctx,initial,AFlowConfig(rounds=2,validation_repeats=2))
    assert result["selected"]["id"]==initial.uid
    assert len(result["metadata"]["versions"])==2


def test_janus_arithmetic_dsl_rejects_python_execution_and_is_bounded():
    for code in ("__import__('os').system('ls')","[x for x in values]","value.__class__","value ** value"):
        with pytest.raises((ValueError,SyntaxError)):ArithmeticFeature(code,["value"])
    assert ArithmeticFeature("sqrt(value ** 2) / (1 + abs(value))",["value"] )({"value":-2})==pytest.approx(2/3)
    assert ArithmeticFeature("value / 0",["value"])({"value":100})==1e6


def test_janus_meta_objective_rewards_top_promotion_quality():
    real=[0.,.2,.5,1.]
    assert promotion_meta_score(real,real)>promotion_meta_score(list(reversed(real)),real)


def test_janus_coevolves_and_only_real_valid_candidates_enter_population():
    backend=NumericBackend([.3,-1.,.8,.2,.9,.1,.6,.95])
    ctx=context(backend,items=1)
    result=run_janus(ctx,Design({"value":.2}),JanusConfig(real_evaluations=9,islands=2,minimum_archive=3,
        evaluator_evolution_period=2,refresh_batch=2,direct_exploration_period=3),features=lambda d:{"value":float(d.payload["value"])})
    assert backend.proxy_calls>0
    promotions=[e for e in result["events"] if e["event"]=="janus_real_promotion"]
    assert promotions and any(e["recommenders"] for e in (x["proxy"] for x in promotions))
    assert all(e["entered_population"]==e["real_valid"] for e in promotions)
    real_valid={e.design_id for e in ctx.evaluations if e.valid}
    assert result["selected"]["id"] in real_valid
    assert all(set(ids)<=real_valid for ids in result["metadata"]["population_ids"])


def test_janus_archive_retains_frontier_and_distinct_programs():
    entries=[]
    for i,value in enumerate([.95,.1,.2,.3,-1.,.4,.5,.6]):
        design=Design({"value":value})
        result=Evaluation(design.uid,(ItemResult("t",max(0,value),0,valid=value>=0),),0,"validation")
        entries.append(ArchiveEntry(design,result,i))
    retained=retain_archive(entries,4)
    assert len({x.design.uid for x in retained})==4
    assert entries[0].design.uid in {x.design.uid for x in retained}


def workflow(instruction,second=None):
    nodes=[{"id":"solve","instruction":instruction,"dependencies":[]}]
    if second:nodes.append({"id":"verify","instruction":second,"dependencies":["solve"]})
    return Design({"workflow":{"nodes":nodes,"output_node":"verify" if second else "solve"}})


def test_emas_primary_change_scope_is_enforced():
    parent=workflow("old","fixed")
    good=workflow("new","fixed")
    h=Hypothesis("accuracy","unclear-instruction","prompt-only","solve")
    validate_revision(parent,good,h)
    with pytest.raises(ValueError,match="scope"):
        validate_revision(parent,workflow("new","also changed"),h)
    validate_revision(workflow("old"),workflow("old","verify"),Hypothesis("accuracy","missing-operation","add-node","verify"))


def test_emas_cost_and_accuracy_rules_use_same_paired_registry():
    current=Evaluation("a",(ItemResult("x",1.,10),ItemResult("y",0.,10)),1,"paired")
    cheaper=Evaluation("b",(ItemResult("x",1.,5),ItemResult("y",0.,5)),1,"paired")
    assert paired_accept("cost",current,cheaper)
    assert not paired_accept("accuracy",current,cheaper)
    mismatch=Evaluation("b",cheaper.items,2,"paired")
    with pytest.raises(ValueError,match="same"):paired_accept("cost",current,mismatch)


class WorkflowTestBackend:
    def __init__(self):self.proposals=0
    def evaluate(self,design,items,seed,stage,token_cap):
        score=float(design.payload["workflow"]["nodes"][0]["instruction"]=="good")
        return Evaluation(design.uid,tuple(ItemResult(x.id,score,1) for x in items),seed,stage)
    def diagnose(self,design,item,result,seed,token_cap):
        return Generated([Hypothesis("accuracy" if result.score<1 else "cost","rare" if item.id=="e0" else "common","prompt-only","solve")],1)
    def propose(self,parent,feedback,seed,token_cap):
        self.proposals+=1
        good=workflow("good")
        return Generated(Design(good.payload,parent.uid),1)


def test_emas_distinct_samples_required_repeated_same_sample_does_not_trigger():
    backend=WorkflowTestBackend()
    ctx=context(backend)
    cfg=EMASConfig(epochs=1,thresholds={x:2 for x in ("add-node","split-node","add-edge","prompt-only","remove-edge","remove-node")})
    result=run_emas(ctx,workflow("bad"),[Item("e1"),Item("e1")],cfg)
    assert backend.proposals==0 and result["metadata"]["accepted_versions"]==0


def test_emas_acceptance_consumes_support_and_replays_remaining_evidence():
    backend=WorkflowTestBackend()
    ctx=context(backend)
    cfg=EMASConfig(epochs=1,thresholds={x:2 for x in ("add-node","split-node","add-edge","prompt-only","remove-edge","remove-node")})
    result=run_emas(ctx,workflow("bad"),[Item("e0"),Item("e1"),Item("e2")],cfg)
    assert result["metadata"]["accepted_versions"]==1
    assert result["metadata"]["consumed_sample_evidence"]==[(0,"e1"),(0,"e2")]
    replay=[d for d in result["metadata"]["diagnoses"] if d["replayed"]]
    assert replay and replay[0]["version"]==1 and replay[0]["hypothesis"]["objective"]=="cost"


def test_workflow_planbench_reaches_native_validator():
    """An actual native plan control through the baseline's task adapter."""
    from pathlib import Path
    import json
    from specrsi.baselines.workflow_backend import WorkflowBackend
    from specrsi.benchmarks.base import BenchmarkExample
    from specrsi.models.client import ModelReply
    source=Path('data/processed/planbench/selection.jsonl')
    if not source.exists():pytest.skip('Prepared native PlanBench data required')
    item=BenchmarkExample.from_dict(json.loads(source.read_text().splitlines()[0]))
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1]*5
    class FixedPlanControl:
        def generate(self,*args,**kwargs):
            return ModelReply(item.reference['ground_truth_plan'],5,1,0.,'stop','control','control-plan')
    backend=object.__new__(WorkflowBackend)
    backend.generation={'max_tokens':10,'temperature':0.,'max_seconds':30}
    backend.lock={'revision':'control'};backend.client=FixedPlanControl();backend.tokenizer=Tokenizer()
    backend.execution_cap=100;backend.calls=[]
    result=backend.evaluate(workflow('Return the plan'),[item],0,'validation',100)
    assert result.complete and result.score==1.0 and result.tokens==6
