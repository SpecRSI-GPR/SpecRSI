"""AFlow graph optimization with upstream mixed parent sampling and experience."""
from dataclasses import dataclass
import math

from .extended_common import Exhausted,RunContext,Design


@dataclass
class AFlowConfig:
    rounds:int=20
    top_k:int=4
    validation_repeats:int=3
    uniform_mixture:float=.3
    score_temperature_scale:float=.2  # Upstream alpha on score*100.
    duplicate_retries:int=3
    convergence_rounds:int=5
    convergence_top_k:int=3


def aflow_probabilities(scores,uniform_mixture=.3,alpha=.2):
    if not scores or not 0<=uniform_mixture<=1 or alpha<=0:
        raise ValueError("Invalid AFlow parent distribution")
    z=[100*alpha*x for x in scores]
    weights=[math.exp(x-max(z)) for x in z]
    total=sum(weights)
    return [uniform_mixture/len(z)+(1-uniform_mixture)*w/total for w in weights]


def run_aflow(context:RunContext,initial:Design,config:AFlowConfig|None=None):
    cfg=config or AFlowConfig()
    if min(cfg.rounds,cfg.top_k,cfg.validation_repeats,cfg.duplicate_retries)<1:
        raise ValueError("AFlow needs positive round/sample/repetition caps")
    versions=[]
    experience={}
    def assess(design):
        evaluations=[]
        for repeat in range(cfg.validation_repeats):
            evaluations.append(context.evaluate(design,seed=context.seed+repeat,stage="validation"))
        if not all(x.valid for x in evaluations):
            return None
        return {"design":design,"score":sum(x.score for x in evaluations)/len(evaluations),
                "tokens":sum(x.tokens for x in evaluations)/len(evaluations),"runs":evaluations}
    first=assess(initial)
    if first is None:
        raise ValueError("AFlow requires valid initial workflow")
    versions.append(first)
    previous_top=None
    unchanged=0
    for round_index in range(1,cfg.rounds):
        if context.budget.remaining_items<len(context.validation)*cfg.validation_repeats:
            break
        parents=sorted(versions,key=lambda x:(-x["score"],x["tokens"],x["design"].uid))[:cfg.top_k]
        probabilities=aflow_probabilities([x["score"] for x in parents],cfg.uniform_mixture,cfg.score_temperature_scale)
        parent=context.rng.choices(parents,weights=probabilities,k=1)[0]
        try:
            child=None
            for _ in range(cfg.duplicate_retries):
                proposed=context.propose(parent["design"],{"method":"aflow","parent_score":parent["score"],
                    "experience":experience.get(parent["design"].uid,[]),
                    "execution_examples":[vars(x) for x in parent["runs"][0].items[:3]],
                    "request":"Optimize workflow topology and node prompts; learn from previous successful and failed edits."})
                if proposed is not None and proposed.uid not in {x["design"].uid for x in versions}:
                    child=proposed
                    break
            if child is None:
                continue
            result=assess(child)
            experience.setdefault(parent["design"].uid,[]).append({"child":child.to_dict(),"score":result["score"] if result else None,
                "improved":bool(result and result["score"]>parent["score"])})
            context.events.append({"event":"aflow_expand_evaluate_update","round":round_index,"parent":parent["design"].uid,
                "support":[x["design"].uid for x in parents],"probabilities":probabilities,"child":child.uid,
                "validated_score":result["score"] if result else None})
            if result:
                versions.append(result)
            top=sorted((x["score"] for x in versions),reverse=True)[:cfg.convergence_top_k]
            mean=sum(top)/len(top)
            unchanged=unchanged+1 if previous_top is not None and mean==previous_top else 0
            previous_top=mean
            if unchanged>=cfg.convergence_rounds:
                break
        except Exhausted:
            break
    # Compare the declared mean across repeats, never choose the luckiest repeat.
    winner=min(versions,key=lambda x:(-x["score"],x["tokens"],x["design"].uid))
    return context.finish("aflow",winner["design"],{"configuration":vars(cfg),"validated_score":winner["score"],
        "versions":[{"id":x["design"].uid,"mean_score":x["score"],"mean_tokens":x["tokens"]} for x in versions],
        "experience":experience,"source":"third_party/aflow/scripts/optimizer.py and optimizer_utils/data_utils.py",
        "adaptation":"Registered immutable DAG operators replace upstream dynamically imported Python workflow files."})
