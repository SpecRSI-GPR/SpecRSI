"""HGM clade Thompson search, adapted to immutable workflow/program designs.

Selection equations follow pinned third_party/hgm hgm.py/tree.py. Unlike that
coding checkout, execution is supplied by the registered bounded backend.
"""
from dataclasses import dataclass,field
import math

from .extended_common import Design,Evaluation,Exhausted,RunContext


@dataclass
class HGMConfig:
    max_steps:int=100
    expansion_power:float=.5
    pseudo_descendant_evaluations:int=10
    cool_down:bool=True
    cooling_power:float=.5
    allow_zero_seed_expansion:bool=True  # Prospective fallback for zero-success task seeds.
    use_descendants:bool=True


@dataclass
class CladeNode:
    design:Design
    parent_id:str|None=None
    children:list[str]=field(default_factory=list)
    observed:dict=field(default_factory=dict)

    @property
    def utility(self):
        return [x.score for x in self.observed.values() if x.complete]

    @property
    def mean(self):
        return sum(self.utility)/len(self.utility) if self.utility else None


def clade_observations(node,nodes,pseudo_limit=10):
    own=node.utility
    result=list(own) if len(own)<pseudo_limit else [node.mean]*pseudo_limit
    pending=list(node.children)
    while pending:
        descendant=nodes[pending.pop()]
        result.extend(descendant.utility)
        pending.extend(descendant.children)
    return result


def thompson_select(nodes,observations,rng,*,temperature_scale=1.):
    draws=[]
    for node in nodes:
        values=observations(node)
        alpha=(1+sum(values))*temperature_scale
        beta=(1+len(values)-sum(values))*temperature_scale
        draws.append((rng.betavariate(alpha,beta),node.design.uid,node))
    return max(draws,key=lambda x:(x[0],x[1]))[2]


def run_hgm(context:RunContext,initial:Design,config:HGMConfig|None=None):
    cfg=config or HGMConfig()
    if not 0<cfg.expansion_power<=1 or cfg.pseudo_descendant_evaluations<1 or cfg.max_steps<1:
        raise ValueError("Invalid HGM expansion/proxy observation settings")
    first=context.evaluate(initial)
    if not first.valid:
        raise ValueError("HGM requires a complete valid initial design")
    root=CladeNode(initial,observed={x.item_id:x for x in first.items})
    nodes={initial.uid:root}
    expansion_failures=0
    for step in range(cfg.max_steps):
        if context.budget.remaining_items<1 or context.budget.remaining_tokens<1:
            break
        # Upstream's evolution counter excludes the initial agent evaluation;
        # physical initialization still remains charged in our experiment budget.
        used=context.budget.real_items-len(first.items)
        evolution_cap=max(1,context.budget.max_real_items-len(first.items))
        scale=(evolution_cap/max(1,context.budget.remaining_items))**cfg.cooling_power if cfg.cool_down else 1.
        pending=[n for n in nodes.values() if len(n.observed)<len(context.validation)]
        expand=(used**cfg.expansion_power>=len(nodes)-1 and expansion_failures<3) or not pending
        try:
            if expand:
                eligible=[n for n in nodes.values() if n.mean is not None and n.mean>0]
                if not eligible and cfg.allow_zero_seed_expansion:
                    eligible=[n for n in nodes.values() if n.mean is not None]
                if not eligible:
                    break
                parent=thompson_select(eligible,lambda n:clade_observations(n,nodes,cfg.pseudo_descendant_evaluations) if cfg.use_descendants else n.utility,
                    context.rng,temperature_scale=scale)
                child=context.propose(parent.design,{"method":"hgm","parent_observations":[vars(x) for x in parent.observed.values()],
                    "clade_observations":clade_observations(parent,nodes,cfg.pseudo_descendant_evaluations),
                    "request":"Revise the executable design using these measured failures; do not alter evaluator."})
                if child is None or child.uid in nodes:
                    expansion_failures+=1
                    continue
                expansion_failures=0
                nodes[child.uid]=CladeNode(child,parent.design.uid)
                parent.children.append(child.uid)
                context.events.append({"event":"clade_expansion","parent":parent.design.uid,"child":child.uid,
                    "parent_clade_mean":sum(clade_observations(parent,nodes,cfg.pseudo_descendant_evaluations))/max(1,len(clade_observations(parent,nodes,cfg.pseudo_descendant_evaluations)))})
            else:
                node=thompson_select(pending,lambda n:n.utility,context.rng,temperature_scale=scale)
                remaining=[x for x in context.validation if context.item_id(x) not in node.observed]
                item=context.rng.choice(remaining)
                evaluation=context.evaluate(node.design,[item],stage="hgm_item")
                node.observed.update({x.item_id:x for x in evaluation.items if x.complete})
                if len(node.observed)==len(context.validation):
                    full=Evaluation(node.design.uid,tuple(node.observed[context.item_id(x)] for x in context.validation),context.seed,"validation")
                    context.evaluations.append(full)
        except Exhausted:
            break
    # Use remaining budget to finish promising incomplete candidates. A partial
    # mean is an allocation statistic and never a final full-suite score.
    for node in sorted(nodes.values(),key=lambda n:-(n.mean if n.mean is not None else -1)):
        missing=[x for x in context.validation if context.item_id(x) not in node.observed]
        if missing and len(missing)<=context.budget.remaining_items:
            try:
                e=context.evaluate(node.design,missing,stage="hgm_completion")
                node.observed.update({x.item_id:x for x in e.items if x.complete})
                if len(node.observed)==len(context.validation):
                    context.evaluations.append(Evaluation(node.design.uid,tuple(node.observed[context.item_id(x)] for x in context.validation),context.seed,"validation"))
            except Exhausted:
                break
    selected,score=context.best_complete(initial)
    return context.finish("hgm",selected,{"configuration":vars(cfg),"complete_score":score.score,
        "tree":[{"id":n.design.uid,"parent":n.parent_id,"children":n.children,"observed_items":list(n.observed),
            "mean_observed":n.mean,"clade_samples":clade_observations(n,nodes,cfg.pseudo_descendant_evaluations)} for n in nodes.values()],
        "source":"third_party/hgm/hgm.py TS_sample/expand; tree.py descendant evaluation pooling"})
