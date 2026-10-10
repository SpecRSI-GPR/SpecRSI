"""Shared auditable contracts for prospective baseline implementations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Protocol, Sequence

from specrsi.core.models import FrozenMap, digest, thaw


@dataclass(frozen=True)
class Design:
    payload: FrozenMap
    parent_id: str | None = None
    label: str = "candidate"

    def __post_init__(self):
        object.__setattr__(self,"payload",FrozenMap(self.payload))

    @property
    def uid(self):
        return digest(self.payload)

    def to_dict(self):
        return {"id":self.uid,"payload":thaw(self.payload),"parent_id":self.parent_id,"label":self.label}


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    score: float
    tokens: int
    complete: bool = True
    valid: bool = True
    trace: tuple[dict,...] = ()

    def __post_init__(self):
        if not math.isfinite(self.score) or not 0<=self.score<=1 or type(self.tokens) is not int or self.tokens<0:
            raise ValueError("Expected bounded native item score and actual nonnegative token cost")


@dataclass(frozen=True)
class Evaluation:
    design_id: str
    items: tuple[ItemResult,...]
    seed: int
    stage: str

    @property
    def complete(self):
        return bool(self.items) and all(x.complete for x in self.items)

    @property
    def valid(self):
        return self.complete and all(x.valid for x in self.items)

    @property
    def score(self):
        if not self.complete:
            raise ValueError("Incomplete evaluation has no complete aggregate score")
        return sum(x.score for x in self.items)/len(self.items)

    @property
    def tokens(self):
        return sum(x.tokens for x in self.items)

    @property
    def correct(self):
        if not self.complete:
            raise ValueError("Incomplete evaluation has no aggregate correct count")
        return sum(x.score==1 for x in self.items)


@dataclass(frozen=True)
class Generated:
    value: Any
    tokens: int
    note: str = ""


class Backend(Protocol):
    def evaluate(self,design:Design,items:Sequence[Any],seed:int,stage:str,token_cap:int)->Evaluation: ...
    def propose(self,parent:Design,feedback:dict,seed:int,token_cap:int)->Generated: ...
    def diagnose(self,design:Design,item:Any,result:ItemResult,seed:int,token_cap:int)->Generated: ...
    def propose_proxy(self,parent:dict,archive:list[dict],feature_names:list[str],seed:int,token_cap:int)->Generated: ...


class Exhausted(RuntimeError):
    pass


@dataclass
class ExperimentBudget:
    max_tokens: int = 1000000
    max_real_items: int = 1000
    tokens: int = 0
    real_items: int = 0
    charges: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.max_tokens,self.max_real_items)):
            raise ValueError("Positive explicit experiment budgets required")

    @property
    def remaining_tokens(self):
        return self.max_tokens-self.tokens

    @property
    def remaining_items(self):
        return self.max_real_items-self.real_items

    def charge(self,tokens:int,items:int,category:str):
        if type(tokens) is not int or tokens<0 or tokens>self.remaining_tokens or items>self.remaining_items:
            raise ValueError("Backend exceeded its enforced token/item budget")
        self.tokens+=tokens
        self.real_items+=items
        self.charges.append({"category":category,"tokens":tokens,"real_items":items})


class RunContext:
    def __init__(self,backend:Backend,validation:Sequence[Any],*,seed=0,budget=None):
        self.backend=backend
        self.validation=tuple(validation)
        if not self.validation or len({self.item_id(x) for x in self.validation})!=len(self.validation):
            raise ValueError("Need a nonempty fixed duplicate-free validation suite")
        self.seed=seed
        self.rng=random.Random(seed)
        self.budget=budget or ExperimentBudget()
        self.events=[]
        self.evaluations=[]
        self.designs={}

    @staticmethod
    def item_id(item):
        if isinstance(item,dict):
            return str(item["id"])
        return str(item.id)

    def evaluate(self,design,items=None,*,seed=None,stage="validation"):
        items=self.validation if items is None else tuple(items)
        if len(items)>self.budget.remaining_items or self.budget.remaining_tokens<1:
            raise Exhausted("Remaining budget cannot execute this required evaluation set")
        run_seed=self.seed if seed is None else seed
        result=self.backend.evaluate(design,items,run_seed,stage,self.budget.remaining_tokens)
        self.budget.charge(result.tokens,len(items),"evaluate:"+stage)
        if result.design_id!=design.uid or result.seed!=run_seed or result.stage!=stage:
            raise ValueError("Evaluator returned a result for the wrong frozen design/seed/stage")
        if tuple(x.item_id for x in result.items)!=tuple(self.item_id(x) for x in items):
            raise ValueError("Evaluator did not return exactly the fixed requested item registry")
        self.designs[design.uid]=design
        self.evaluations.append(result)
        self.events.append({"event":"real_evaluation","design_id":design.uid,"stage":stage,
            "seed":run_seed,"complete":result.complete,"valid":result.valid,
            "score":result.score if result.complete else None,"tokens":result.tokens,
            "items":[asdict(x) for x in result.items]})
        return result

    def _generate(self,method,*args):
        if self.budget.remaining_tokens<1:
            raise Exhausted("No remaining proposal/diagnosis token budget")
        result=getattr(self.backend,method)(*args,self.rng.randrange(2**32),self.budget.remaining_tokens)
        self.budget.charge(result.tokens,0,method)
        self.events.append({"event":method,"tokens":result.tokens,"note":result.note})
        return result.value

    def propose(self,parent,feedback):
        value=self._generate("propose",parent,feedback)
        if value is not None:
            if not isinstance(value,Design) or value.parent_id!=parent.uid:
                raise ValueError("Proposer must return an immutable child of the specified parent")
            self.designs[value.uid]=value
        return value

    def diagnose(self,design,item,result):
        return self._generate("diagnose",design,item,result)

    def propose_proxy(self,parent,archive,names):
        return self._generate("propose_proxy",parent,archive,names)

    def best_complete(self,initial,*,stage="validation"):
        required=tuple(self.item_id(x) for x in self.validation)
        options=[e for e in self.evaluations if e.stage==stage and e.valid and tuple(x.item_id for x in e.items)==required]
        if not options:
            raise ValueError("No valid complete initial/reference result")
        best=min(options,key=lambda e:(-e.score,e.tokens,e.design_id))
        return self.designs[best.design_id],best

    def finish(self,method,selected,metadata):
        return {"method":method,"implementation":"prospective_registered_adaptation","author_checkpoint_used":False,
            "selected":selected.to_dict(),"budget":asdict(self.budget),"metadata":metadata,
            "events":self.events,"designs":[x.to_dict() for x in self.designs.values()]}
