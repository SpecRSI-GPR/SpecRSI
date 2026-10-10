"""EMAS recurrence-gated workflow revision and paired validation."""
from __future__ import annotations
from dataclasses import dataclass,field

from specrsi.workflows import Workflow
from specrsi.core.models import thaw
from .extended_common import Design,Exhausted,RunContext


OPERATIONS=("add-node","split-node","add-edge","prompt-only","remove-edge","remove-node")


@dataclass(frozen=True,order=True)
class Hypothesis:
    objective:str
    defect:str
    operation:str
    location:str

    def __post_init__(self):
        if self.objective not in {"accuracy","cost"} or self.operation not in OPERATIONS or not self.defect or not self.location:
            raise ValueError("Diagnosis requires objective, canonical defect, permitted operation and location")


@dataclass
class EMASConfig:
    epochs:int=2
    max_revisions:int=20
    thresholds:dict=field(default_factory=lambda:{"add-node":4,"split-node":4,"add-edge":4,
        "prompt-only":6,"remove-edge":10,"remove-node":10})
    recurrence:bool=True
    paired_validation:bool=True


def _graph(design):
    payload=thaw(design.payload)
    # Fixed native coding actors expose exactly one editable strategy slot.
    # Project it to one node so recurrence and paired acceptance are unchanged.
    if set(payload)=={"system_prompt"}:
        payload={"workflow":{"nodes":[{"id":"strategy","instruction":payload["system_prompt"],"dependencies":[]}],"output_node":"strategy"}}
    workflow=Workflow.from_dict(payload["workflow"])
    nodes={x.id:x for x in workflow.nodes}
    edges={(source,node.id) for node in workflow.nodes for source in node.dependencies}
    return workflow,nodes,edges


def validate_revision(parent:Design,child:Design,h:Hypothesis):
    """Enforce one primary operation and its directly affected graph closure."""
    if "system_prompt" in parent.payload:
        if set(child.payload)!={"system_prompt"} or h.operation!="prompt-only" or h.location!="strategy":
            raise ValueError("Native instruction domain permits only prompt-only at strategy")
    p,pn,pe=_graph(parent)
    c,cn,ce=_graph(child)
    added=set(cn)-set(pn)
    removed=set(pn)-set(cn)
    added_edges,removed_edges=ce-pe,pe-ce
    changed_prompts={key for key in set(pn)&set(cn) if pn[key].instruction!=cn[key].instruction}
    if parent.uid==child.uid:
        raise ValueError("Unchanged revision")
    if h.operation=="prompt-only":
        legal=not added and not removed and pe==ce and p.output_node==c.output_node and changed_prompts=={h.location}
    elif h.operation in {"add-edge","remove-edge"}:
        try:
            source,target=h.location.split("->")
        except ValueError as exc:
            raise ValueError("Edge location must use canonical source->target") from exc
        legal=not added and not removed and p.output_node==c.output_node and changed_prompts<={target}
        legal &= (added_edges=={(source,target)} and not removed_edges) if h.operation=="add-edge" else (removed_edges=={(source,target)} and not added_edges)
    elif h.operation=="add-node":
        legal=len(added)==1 and not removed and not removed_edges
        new=next(iter(added),"")
        affected={target for source,target in added_edges}
        legal &= new==h.location and all(new in edge for edge in added_edges) and changed_prompts<=affected and c.output_node in {p.output_node,new}
    elif h.operation=="remove-node":
        removed_id=h.location
        parents={source for source,target in pe if target==removed_id}
        successors={target for source,target in pe if source==removed_id}
        bypass={(source,target) for source in parents for target in successors}
        legal=removed=={removed_id} and not added and all(removed_id in edge for edge in removed_edges)
        legal &= added_edges<=bypass and changed_prompts<=successors
        legal &= c.output_node==p.output_node if p.output_node!=removed_id else c.output_node in parents
    else:  # split-node retains the canonical target and expands its local group.
        target=h.location
        successors={b for a,b in pe if a==target}
        group=added|{target}
        legal=bool(added) and not removed and target in pn
        legal &= all(a in group or b in group for a,b in added_edges|removed_edges)
        legal &= changed_prompts<=(successors|{target})
        legal &= c.output_node==p.output_node if p.output_node!=target else c.output_node in group
    if not legal:
        raise ValueError(f"Revision violates {h.operation} scope at {h.location}")


def paired_accept(objective,current,candidate):
    if tuple(x.item_id for x in current.items)!=tuple(x.item_id for x in candidate.items) or current.seed!=candidate.seed:
        raise ValueError("Paired validation must use the same complete items and random schedule")
    if not current.valid or not candidate.valid:
        return False
    if objective=="accuracy":
        return candidate.correct>current.correct
    if objective=="cost":
        return candidate.correct>=current.correct and candidate.tokens<current.tokens
    raise ValueError("Unknown EMAS objective")


def run_emas(context:RunContext,initial:Design,evolve,config:EMASConfig|None=None):
    cfg=config or EMASConfig()
    if not evolve or cfg.epochs<1 or cfg.max_revisions<1 or set(cfg.thresholds)!=set(OPERATIONS) or min(cfg.thresholds.values())<1:
        raise ValueError("EMAS needs evolve items, positive limits and all operation thresholds")
    validation_ids={context.item_id(x) for x in context.validation}
    if validation_ids & {context.item_id(x) for x in evolve}:
        raise ValueError("EMAS evolve and validation contexts must be disjoint")
    _graph(initial)
    baseline=context.evaluate(initial)
    if not baseline.valid:
        raise ValueError("EMAS requires a valid initial system")
    current=initial
    version=0
    attempts=0
    buffers={}
    active={}
    diagnoses=[]
    consumed=[]

    def observe(item,replay=False):
        result=context.evaluate(current,[item],stage="evolve",seed=context.seed)
        if not result.complete:
            return
        records=context.diagnose(current,item,result.items[0]) or []
        sample_id=context.item_id(item)
        active[sample_id]=item
        for raw in records:
            try:
                h=raw if isinstance(raw,Hypothesis) else Hypothesis(**raw)
            except (TypeError,ValueError):
                context.events.append({"event":"invalid_diagnosis","sample":sample_id})
                continue
            if (h.objective=="accuracy")!=(result.items[0].score<1):
                continue
            buffers.setdefault(h,set()).add(sample_id)
            diagnoses.append({"version":version,"sample":sample_id,"hypothesis":vars(h),"replayed":replay})

    try:
        for epoch in range(cfg.epochs):
            for item in evolve:
                if attempts>=cfg.max_revisions:
                    break
                observe(item)
                eligible=[h for h,support in buffers.items() if len(support)>=(cfg.thresholds[h.operation] if cfg.recurrence else 1)]
                if not eligible:
                    continue
                h=min(eligible,key=lambda h:(OPERATIONS.index(h.operation),h))
                support=set(buffers[h])
                if cfg.paired_validation and context.budget.remaining_items<2*len(context.validation):
                    raise Exhausted("Paired validation must fit in full before proposing a revision")
                candidate=context.propose(current,{"method":"emas","hypothesis":vars(h),"support_samples":sorted(support),
                    "current_version":version,"request":"Make exactly this primary change plus only directly affected edges/prompts."})
                attempts+=1
                accepted=False
                rejection="no_candidate"
                if candidate is not None:
                    try:
                        validate_revision(current,candidate,h)
                    except ValueError as exc:
                        rejection=str(exc)
                    else:
                        if cfg.paired_validation:
                            pair_seed=context.seed+epoch*1000+attempts
                            reference=context.evaluate(current,seed=pair_seed,stage="paired_validation")
                            revised=context.evaluate(candidate,seed=pair_seed,stage="paired_validation")
                            accepted=paired_accept(h.objective,reference,revised)
                        else:
                            # Explicit no-validation ablation: structural checks
                            # remain, but no measured improvement is asserted.
                            accepted=True
                        rejection="paired_rule_rejected" if not accepted else ""
                context.events.append({"event":"emas_revision","version_before":version,"hypothesis":vars(h),
                    "support":sorted(support),"candidate":candidate.uid if candidate else None,"accepted":accepted,"rejection":rejection})
                # Consume ALL hypotheses arising from supporting samples, not only
                # the triggering key. A rejected proposal also consumes evidence.
                consumed.extend((version,s) for s in sorted(support))
                for key in tuple(buffers):
                    buffers[key]-=support
                    if not buffers[key]:
                        del buffers[key]
                for key in support:
                    active.pop(key,None)
                if accepted:
                    current=candidate
                    version+=1
                    buffers.clear()
                    for remaining in list(active.values()):
                        observe(remaining,replay=True)
            if attempts>=cfg.max_revisions:
                break
    except Exhausted:
        context.events.append({"event":"budget_stop"})
    return context.finish("emas",current,{"configuration":vars(cfg),"accepted_versions":version,
        "attempted_revisions":attempts,"diagnoses":diagnoses,"consumed_sample_evidence":consumed,
        "active_support":[{"hypothesis":vars(h),"samples":sorted(s)} for h,s in buffers.items()],
        "source":"https://arxiv.org/html/2608.07196v1 Sections 3.4-3.6; Appendix A.4",
        "adaptation":("Fixed native actor/pipeline: prompt-only strategy slot; recurrence and paired native validation retained."
            if "system_prompt" in initial.payload else "Flat node-prompt DAG slots replace task-specific phase/category prompt hierarchy.")})
