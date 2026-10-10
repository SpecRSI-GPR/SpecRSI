"""Janus algorithm/evaluator co-evolution with real-validated promotion.

Proxy programs use a bounded arithmetic DSL over structural features. This is a
registered prospective specialization of evaluator-as-program, not author code.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass,field
import hashlib
import json
import math
import re

import numpy as np

from specrsi.core.models import thaw
from .extended_common import Design,Evaluation,Exhausted,RunContext


def structural_features(design:Design):
    raw=thaw(design.payload)
    text=json.dumps(raw,sort_keys=True)
    nodes=raw.get("workflow",{}).get("nodes",[])
    words=re.findall(r"\w+",text.lower())
    return {"nodes":float(len(nodes)),"edges":float(sum(len(n.get("dependencies",[])) for n in nodes)),
        "characters":len(text)/1000.,"words":len(words)/100.,"verify_terms":float(sum(w in {"verify","check","validate","test"} for w in words)),
        "reason_terms":float(sum(w in {"reason","solve","derive","plan"} for w in words)),
        "unique_ratio":len(set(words))/max(1,len(words)),"numeric_terms":float(sum(w.isdigit() for w in words))}


def code_embedding(design,dimension=64):
    vector=np.zeros(dimension)
    for token in re.findall(r"\w+|[^\w\s]",json.dumps(thaw(design.payload),sort_keys=True)):
        value=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],"big")
        vector[value%dimension]+=1 if value&(1<<63) else -1
    norm=np.linalg.norm(vector)
    return vector/max(norm,1e-12)


class ArithmeticFeature:
    FUNCTIONS={"abs":abs,"min":min,"max":max,"sqrt":lambda x:math.sqrt(max(0.,x)),"log1p":lambda x:math.log1p(max(0.,x))}
    def __init__(self,expression,feature_names):
        self.expression=expression
        self.tree=ast.parse(expression,mode="eval")
        if len(list(ast.walk(self.tree)))>64:
            raise ValueError("Proxy expression exceeds bounded DSL size")
        allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Load,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.USub,ast.UAdd,ast.Call)
        for node in ast.walk(self.tree):
            if not isinstance(node,allowed):
                raise ValueError("Unsupported proxy syntax")
            if isinstance(node,ast.Name) and node.id not in set(feature_names)|set(self.FUNCTIONS):
                raise ValueError("Unknown proxy feature")
            if isinstance(node,ast.Call) and (not isinstance(node.func,ast.Name) or node.func.id not in self.FUNCTIONS or node.keywords or not 1<=len(node.args)<=4):
                raise ValueError("Only bounded arithmetic proxy functions are allowed")
            if isinstance(node,ast.Constant) and (type(node.value) not in (int,float) or not math.isfinite(node.value) or abs(node.value)>1e6):
                raise ValueError("Invalid proxy constant")
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow) and (not isinstance(node.right,ast.Constant) or node.right.value not in (2,3)):
                raise ValueError("Proxy powers must be constant 2 or 3")

    def __call__(self,features):
        def visit(n):
            if isinstance(n,ast.Expression):return visit(n.body)
            if isinstance(n,ast.Constant):return float(n.value)
            if isinstance(n,ast.Name):return float(features[n.id])
            if isinstance(n,ast.UnaryOp):return -visit(n.operand) if isinstance(n.op,ast.USub) else visit(n.operand)
            if isinstance(n,ast.Call):return self.FUNCTIONS[n.func.id](*(visit(x) for x in n.args))
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):value=a+b
            elif isinstance(n.op,ast.Sub):value=a-b
            elif isinstance(n.op,ast.Mult):value=a*b
            elif isinstance(n.op,ast.Div):value=a/(b if abs(b)>1e-8 else 1e-8)
            else:value=a**b
            return max(-1e6,min(1e6,value))
        result=visit(self.tree)
        if not math.isfinite(result):raise ValueError("Nonfinite proxy feature")
        return result


def _ranks(values):
    ranks=np.zeros(len(values))
    for value in sorted(set(values)):
        indices=np.flatnonzero(np.asarray(values)==value)
        count=sum(x<value for x in values)
        ranks[indices]=count+(len(indices)-1)/2
    return ranks


def promotion_meta_score(predictions,actual,top_k=2,weights=(.45,.45,.1)):
    predictions=np.asarray(predictions,dtype=float)
    actual=np.asarray(actual,dtype=float)
    k=min(top_k,len(actual))
    if k<1 or not np.isfinite(predictions).all() or not np.isfinite(actual).all():
        raise ValueError("Finite real archive values required")
    order=np.argsort(-predictions,kind="stable")[:k]
    ideal=np.sort(actual)[::-1][:k]
    span=float(actual.max()-actual.min())
    regret=0. if span<1e-12 else float((actual.max()-actual[order].max())/span)
    discounts=np.log2(np.arange(k)+2)
    dcg=float(np.sum((2**actual[order]-1)/discounts))
    ideal_dcg=float(np.sum((2**ideal-1)/discounts))
    ndcg=dcg/ideal_dcg if ideal_dcg>1e-12 else 1.
    a,b=_ranks(predictions),_ranks(actual)
    correlation=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.
    return weights[0]*(1-regret)+weights[1]*ndcg+weights[2]*correlation


@dataclass
class ArchiveEntry:
    design:Design
    result:Evaluation
    iteration:int


@dataclass
class ProxyProgram:
    name:str
    expressions:tuple[str,...]
    feature_names:tuple[str,...]
    ridge:float=.01
    quality_weights:object=None
    validity_weights:object=None
    uncertainty:float=1.
    meta_score:float=0.

    def __post_init__(self):
        if not 1<=len(self.expressions)<=12 or self.ridge<=0:
            raise ValueError("Need 1..12 bounded proxy expressions and positive ridge")
        self.compiled=[ArithmeticFeature(x,self.feature_names) for x in self.expressions]

    @property
    def uid(self):
        return hashlib.sha256(json.dumps(self.expressions).encode()).hexdigest()

    def vector(self,design,features):
        values=features(design)
        return np.array([1.]+[f(values) for f in self.compiled])

    def _regress(self,x,y):
        if not len(y):return np.zeros(x.shape[1])
        penalty=np.eye(x.shape[1])*self.ridge
        penalty[0,0]=1e-8
        return np.linalg.solve(x.T@x+penalty,x.T@y)

    def fit(self,archive,features):
        x=np.stack([self.vector(e.design,features) for e in archive])
        valid=np.array([e.result.valid for e in archive],dtype=bool)
        y=np.array([e.result.score if e.result.complete and e.result.valid else 0. for e in archive])
        self.quality_weights=self._regress(x[valid],y[valid])
        self.validity_weights=self._regress(x,valid.astype(float))
        # Cross-fitted archive ranking prevents evaluating each flexible proxy on
        # the very labels that fitted its parameters. This split is prospective.
        predictions=[]
        for held in range(len(archive)):
            train=np.arange(len(archive))!=held
            w=self._regress(x[train&valid],y[train&valid])
            predictions.append(float(np.clip(x[held]@w,0,1)))
        self.meta_score=promotion_meta_score(predictions,y)
        errors=np.asarray(predictions)[valid]-y[valid]
        self.uncertainty=float(np.std(errors)) if len(errors)>1 else 1.

    def predict(self,design,features):
        if self.quality_weights is None:raise ValueError("Proxy must be fitted from actual observations")
        x=self.vector(design,features)
        return float(np.clip(x@self.quality_weights,0,1)),float(np.clip(x@self.validity_weights,0,1)),self.uncertainty

    def to_dict(self):
        return {"name":self.name,"expressions":self.expressions,"meta_score":self.meta_score,
            "uncertainty":self.uncertainty,"quality_weights":self.quality_weights.tolist() if self.quality_weights is not None else None,
            "validity_weights":self.validity_weights.tolist() if self.validity_weights is not None else None}


@dataclass
class JanusConfig:
    real_evaluations:int=20
    islands:int=2
    proposals_per_promotion:int=3
    minimum_archive:int=3
    archive_limit:int=64
    pool_limit:int=32
    evaluator_evolution_period:int=2
    refresh_batch:int=2
    evaluator_population:int=8
    portfolio_size:int=3
    regions:int=2
    direct_exploration_period:int=5
    online_credit_kappa:float=4.
    quality_weight:float=1.
    uncertainty_weight:float=.3
    novelty_weight:float=.2
    validity_gate:float=.2
    evolve_evaluators:bool=True
    proxy_guidance:bool=True


def retain_archive(entries,limit):
    unique={x.design.uid:x for x in entries}
    entries=list(unique.values())
    if len(entries)<=limit:return entries
    quarter=max(1,limit//4)
    selected={}
    for sequence in (sorted(entries,key=lambda e:-(e.result.score if e.result.valid else -1))[:quarter],
                     sorted(entries,key=lambda e:-e.iteration)[:quarter],
                     sorted(entries,key=lambda e:(e.result.valid,e.iteration))[:quarter]):
        for entry in sequence:selected[entry.design.uid]=entry
    while len(selected)<limit:
        candidates=[e for e in entries if e.design.uid not in selected]
        entry=max(candidates,key=lambda e:min(1-float(code_embedding(e.design)@code_embedding(s.design)) for s in selected.values()))
        selected[entry.design.uid]=entry
    return list(selected.values())[:limit]


def region_centroids(population,count):
    vectors=np.stack([code_embedding(e.design) for e in population])
    centroids=[vectors[0]]
    while len(centroids)<min(count,len(vectors)):
        index=max(range(len(vectors)),key=lambda i:min(1-float(vectors[i]@c) for c in centroids))
        centroids.append(vectors[index])
    centroids=np.stack(centroids)
    for _ in range(4):
        assignments=np.argmax(vectors@centroids.T,axis=1)
        for index in range(len(centroids)):
            if (assignments==index).any():
                mean=vectors[assignments==index].mean(axis=0)
                centroids[index]=mean/max(np.linalg.norm(mean),1e-12)
    return centroids


def run_janus(context:RunContext,initial:Design,config:JanusConfig|None=None,*,features=structural_features):
    cfg=config or JanusConfig()
    if min(cfg.real_evaluations,cfg.islands,cfg.proposals_per_promotion,cfg.minimum_archive,cfg.refresh_batch,cfg.direct_exploration_period)<1:
        raise ValueError("Janus periods and counts must be positive")
    first=context.evaluate(initial)
    if not first.valid:raise ValueError("Janus requires a real-valid initial population seed")
    archive=[ArchiveEntry(initial,first,0)]
    populations=[[archive[0]] for _ in range(cfg.islands)]
    pools=[{} for _ in populations]
    names=tuple(sorted(features(initial)))
    evaluators=[ProxyProgram("seed-structural",names[:min(8,len(names))],names)]
    credits={}
    revision=0
    real_count=1
    best=first.score
    centroids=[region_centroids(x,cfg.regions) for x in populations]

    def refresh():
        nonlocal evaluators,revision,centroids
        usable=[]
        for proxy in evaluators:
            try:proxy.fit(archive,features)
            except (ValueError,KeyError,OverflowError,np.linalg.LinAlgError):continue
            usable.append(proxy)
        evaluators=sorted(usable,key=lambda p:(-p.meta_score,p.uid))[:cfg.evaluator_population]
        revision+=1
        credits.clear()
        centroids=[region_centroids(x,cfg.regions) for x in populations]
        context.events.append({"event":"janus_archive_refresh","revision":revision,"archive_size":len(archive),
            "evaluators":[p.to_dict() for p in evaluators]})
        if evaluators:
            for island,pool in enumerate(pools):
                for entry in pool.values():entry.update(screen(entry["design"],island))

    def screen(design,island):
        vector=code_embedding(design)
        similarity=centroids[island]@vector
        region=int(np.argmax(similarity))
        novelty=float(np.clip((1-similarity[region])/2,0,1))
        def credit(proxy):
            n,hits,gain=credits.get((island,region,proxy.uid),(0,0,0.))
            alpha=n/(n+cfg.online_credit_kappa)
            online=.5*hits/max(1,n)+.5*gain/max(1,n)
            return (1-alpha)*proxy.meta_score+alpha*online
        active=sorted(evaluators,key=lambda p:-credit(p))[:cfg.portfolio_size]
        scores=[p.predict(design,features) for p in active]
        mass=np.array([math.exp(credit(p)) for p in active]);mass/=mass.sum()
        quality=float(sum(w*s[0] for w,s in zip(mass,scores)))
        validity=float(sum(w*s[1] for w,s in zip(mass,scores)))
        uncertainty=float(sum(w*s[2] for w,s in zip(mass,scores))+math.sqrt(sum(w*(s[0]-quality)**2 for w,s in zip(mass,scores))))
        acquisition=cfg.quality_weight*validity*quality+cfg.uncertainty_weight*validity*uncertainty+cfg.novelty_weight*novelty
        return {"region":region,"quality":quality,"validity":validity,"uncertainty":uncertainty,
                "novelty":novelty,"acquisition":acquisition,"recommenders":[p.uid for p in active],"revision":revision}

    try:
        while real_count<cfg.real_evaluations and context.budget.remaining_items>=len(context.validation):
            island=(real_count-1)%cfg.islands
            population=populations[island]
            can_proxy=cfg.proxy_guidance and len(archive)>=cfg.minimum_archive
            if can_proxy and (not any(p.quality_weights is not None for p in evaluators)):
                refresh()
            if can_proxy and cfg.evolve_evaluators and real_count%cfg.evaluator_evolution_period==0:
                parent=evaluators[0].to_dict() if evaluators else {"expressions":names[:3]}
                generated=context.propose_proxy(parent,[{"design":e.design.to_dict(),"score":e.result.score if e.result.complete else None,
                    "valid":e.result.valid,"features":features(e.design)} for e in archive],list(names))
                if generated:
                    try:
                        proxy=ProxyProgram(generated.get("name",f"evolved-{real_count}"),tuple(generated["expressions"]),names)
                        if proxy.uid not in {p.uid for p in evaluators}:evaluators.append(proxy)
                    except (KeyError,ValueError,SyntaxError):context.events.append({"event":"invalid_proxy_program"})
                refresh()
            direct=(not can_proxy) or not evaluators or real_count%cfg.direct_exploration_period==0
            attempts=1 if direct else cfg.proposals_per_promotion
            generated_candidates=[]
            for _ in range(attempts):
                parent=context.rng.choices(population,weights=[math.exp(e.result.score) for e in population],k=1)[0]
                child=context.propose(parent.design,{"method":"janus","real_parent_score":parent.result.score,
                    "recent_real_outcomes":[{"design":e.design.to_dict(),"score":e.result.score if e.result.complete else None,"valid":e.result.valid} for e in archive[-4:]],
                    "request":"Propose an executable target design; proxy scores only allocate real validation."})
                if child is None or child.uid in {e.design.uid for e in archive}:continue
                generated_candidates.append(child)
                if not direct:
                    pools[island][child.uid]={"design":child,**screen(child,island)}
            if direct:
                if not generated_candidates:
                    context.events.append({"event":"janus_no_novel_candidate"});break
                promoted={"design":generated_candidates[0],"recommenders":[],"region":-1,"revision":revision}
            else:
                pool=pools[island]
                if not pool:break
                legal=[e for e in pool.values() if e["validity"]>=cfg.validity_gate]
                promoted=max(legal or list(pool.values()),key=lambda e:(e["acquisition"],e["design"].uid))
                pool.pop(promoted["design"].uid)
                for key in sorted(pool,key=lambda key:pool[key]["acquisition"],reverse=True)[cfg.pool_limit:]:pool.pop(key)
            before=best
            result=context.evaluate(promoted["design"])
            real_count+=1
            entry=ArchiveEntry(promoted["design"],result,real_count)
            archive=retain_archive(archive+[entry],cfg.archive_limit)
            if result.valid:
                population.append(entry)
                best=max(best,result.score)
            for proxy_id in promoted["recommenders"]:
                key=(island,promoted["region"],proxy_id)
                n,hits,gain=credits.get(key,(0,0,0.))
                improvement=max(0.,result.score-before) if result.valid else 0.
                credits[key]=(n+1,hits+int(improvement>0),gain+improvement)
            context.events.append({"event":"janus_real_promotion","candidate":promoted["design"].uid,"island":island,
                "proxy":{k:v for k,v in promoted.items() if k!="design"},"real_valid":result.valid,
                "real_score":result.score if result.complete else None,"entered_population":result.valid,
                "incumbent_improved":best>before,"direct_exploration":direct})
            if len(archive)>=cfg.minimum_archive and real_count%cfg.refresh_batch==0:refresh()
    except Exhausted:context.events.append({"event":"budget_stop"})
    selected,outcome=context.best_complete(initial)
    return context.finish("janus",selected,{"configuration":vars(cfg),"real_evaluations":real_count,"real_best":outcome.score,
        "archive_revision":revision,"archive_designs":[e.design.uid for e in archive],
        "population_ids":[[e.design.uid for e in p] for p in populations],"evaluators":[p.to_dict() for p in evaluators],
        "source":"https://arxiv.org/html/2608.08189v1 Section III",
        "adaptation":"Arithmetic feature-program DSL, ridge calibration, leave-one-out meta scores, hashed structural embeddings and deterministic region clustering."})
