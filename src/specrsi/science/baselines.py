"""Real-simulator backend for the shared HGM/AFlow/Janus/EMAS runner.

The default proposer uses seeded parameter mutation. An explicitly supplied
local model enables JSON parameter proposals through the same metered interface.
Each evaluator call runs the registered native simulator for the requested cases.
"""
from __future__ import annotations
import random
import json
import re
from dataclasses import asdict
from urllib.parse import urlsplit

from specrsi.baselines.extended_common import Design,Evaluation,Generated,ItemResult
from specrsi.core.models import thaw
from .executor import ScienceExecutor,make_candidate,make_scope,mutate
from .registry import DOMAINS,load_config,registration


def numeric_features(design: Design,case_ids=()) -> dict:
    result={}
    payload=thaw(design.payload)
    blocks=[('',payload['parameters'])]
    for case_id in case_ids:
        blocks.append((f'case_{case_id}_',payload.get('scenario_overrides',{}).get('case:'+str(case_id),payload['parameters'])))
    for prefix,parameters in blocks:
        for key,value in parameters.items():
            if isinstance(value,list):
                result.update({f"{prefix}{key}_{i}":float(x) for i,x in enumerate(value)})
            else:
                result[prefix+key]=float(value)
    return result


class ScienceBackend:
    def __init__(self,domain: str,wall_cap_ms: int=180000,model: dict|None=None,*,case_ids=(0,),proposal_space='global_only'):
        self.domain=domain
        self.wall_cap_ms=wall_cap_ms
        self.executor=ScienceExecutor()
        self.calls=[]
        self.client=None
        if proposal_space not in {'global_only','global_and_case_routes'}:raise ValueError('Unknown science program space')
        self.case_ids=tuple(case_ids);self.proposal_space=proposal_space
        if model:
            from specrsi.models.client import LocalModelClient
            url=model.get("base_url","http://127.0.0.1:8000")
            if urlsplit(url).hostname not in {"127.0.0.1","localhost","::1"}:
                raise ValueError("Science generation requires an explicit loopback local model server")
            self.client=LocalModelClient(url,model.get("name","local-model"))
            self.health=self.client.health()

    def _generate_json(self,prompt,seed,token_cap):
        reply=self.client.generate([{"role":"system","content":"Return only the requested JSON object. Propose parameter data; never Python code."},
                                    {"role":"user","content":prompt}],max_tokens=min(768,max(1,token_cap-1)),
                                   total_token_cap=token_cap,seed=seed,max_seconds=90,enable_thinking=False)
        self.calls.append(asdict(reply))
        tokens=reply.prompt_tokens+reply.completion_tokens
        text=reply.text.strip()
        text=re.sub(r"^```(?:json)?\s*|\s*```$","",text)
        try:value=json.loads(text)
        except (ValueError,TypeError):value=None
        return value,tokens

    def evaluate(self,design,items,seed,stage,token_cap):
        payload=thaw(design.payload)
        allowed={'parameters','scenario_overrides'} if self.proposal_space=='global_and_case_routes' else {'parameters'}
        if 'parameters' not in payload or not set(payload)<=allowed:
            raise ValueError("Science baseline designs expose only registered parameter JSON")
        results=[]
        for item in items:
            if item["domain"]!=self.domain:
                raise ValueError("Cross-domain evaluator request")
            scope=make_scope(self.domain,item["scenario_set"],(item.get("case_id",0),))
            if item["evaluator_identity"]!=scope.tasks[0].metadata["evaluator_identity"]:
                raise ValueError("Scenario registry changed during experiment")
            candidate=make_candidate(self.domain,payload["parameters"],design.label,scenario_overrides=payload.get('scenario_overrides',{}))
            result=self.executor.run(candidate,scope.tasks[0],scope,self.wall_cap_ms)
            trace={"status":result.status,"wall_cost_ms":result.actual_cost,"receipt":thaw(result.receipt)}
            if result.outcome is None:
                trace["error"]=result.error
                results.append(ItemResult(item["id"],0.,0,complete=False,valid=False,trace=(trace,)))
            else:
                trace["native"]=thaw(result.receipt["native_result"])
                results.append(ItemResult(item["id"],result.outcome.score,0,valid=result.outcome.admissible,trace=(trace,)))
        return Evaluation(design.uid,tuple(results),seed,stage)

    def propose(self,parent,feedback,seed,token_cap):
        if self.proposal_space=='global_and_case_routes':
            from .proposals import parameter_catalog
            payload=thaw(parent.payload);parameters=payload['parameters'];overrides=payload.get('scenario_overrides',{})
            rng=random.Random(seed)
            catalog=parameter_catalog(self.domain,parameters,[overrides.get('case:'+str(i),parameters) for i in self.case_ids],rng)
            tokens=0
            if self.client:
                value,tokens=self._generate_json('Choose a promising feasible scientific edit. Return ONLY JSON {"edit": integer}. '
                    'task_index=-1 changes global parameters; other indices select registered stable case routes. '
                    'Every edit is evaluated by the unchanged complete native simulator.\n'+json.dumps({
                        'case_ids':self.case_ids,'parent':payload,'observed_feedback':str(feedback)[:2000],
                        'choices':[{'edit':i,'task_index':x['task_index'],'changes':x['changes']} for i,x in enumerate(catalog)]}),seed,token_cap)
                if not isinstance(value,dict) or set(value)!={'edit'} or type(value['edit']) is not int or not 0<=value['edit']<len(catalog):
                    return Generated(None,tokens,'Invalid catalog selection discarded after actual metering')
                chosen=catalog[value['edit']]
            else:chosen=rng.choice(catalog)
            if chosen['task_index']<0:parameters=chosen['parameters']
            else:overrides['case:'+str(self.case_ids[chosen['task_index']])]=chosen['parameters']
            child=Design({'parameters':parameters,'scenario_overrides':overrides},parent.uid,f'catalog-edit-{seed}')
            return Generated(child,tokens,'Shared SpecRSI/native-optimizer numeric catalog and stable case-route program space')
        if self.client:
            specification=load_config(self.domain,"development")
            value,tokens=self._generate_json("Improve this registered scientific design. Return a JSON object with exactly the same parameter keys. "
                "The independent simulator will enforce all bounds and score the complete native scenario.\n"
                +json.dumps({"specification":specification,"parent_parameters":thaw(parent.payload)["parameters"],
                             "observed_feedback":str(feedback)[:2000]}),seed,token_cap)
            child=Design({"parameters":value},parent.uid,f"local-model-edit-{seed}") if isinstance(value,dict) else None
            return Generated(child,tokens,"Actual local model parameter proposal; invalid JSON discarded after metering")
        parameters=mutate(self.domain,thaw(parent.payload)["parameters"],random.Random(seed))
        return Generated(Design({"parameters":parameters},parent.uid,f"parameter-edit-{seed}"),0,
                         "Registered seeded numerical proposal control; no LLM tokens")

    def diagnose(self,design,item,result,seed,token_cap):
        if self.client:
            value,tokens=self._generate_json("Diagnose the observed native simulation and return JSON with explanation and proposed_parameter_changes.\n"
                +json.dumps({"parameters":thaw(design.payload),"valid":result.valid,"native_trace":str(result.trace)[:5000]}),seed,token_cap)
            return Generated(value or {},tokens,"Actual local model diagnosis of observed native outputs")
        return Generated({"valid":result.valid,"complete":result.complete,"native_trace":list(result.trace)},0,
                         "Trusted observed simulator diagnostics")

    def propose_proxy(self,parent,archive,feature_names,seed,token_cap):
        if self.client:
            value,tokens=self._generate_json("Return JSON {\"name\":string,\"expressions\":[1 to 8 arithmetic expressions]}. "
                "Expressions may use only the listed variable names, numbers, + - * /, powers2or3, abs,min,max,sqrt,log1p. "
                "Propose physical features to predict quality and validity; coefficients are fitted later to real observations.\n"
                +json.dumps({"domain":self.domain,"feature_names":feature_names,"parent":parent,"observed_archive":str(archive)[-5000:]}),seed,token_cap)
            return Generated(value,tokens,"Actual local model arithmetic proxy proposal; downstream bounded DSL validation")
        rng=random.Random(seed)
        names=list(feature_names)
        rng.shuffle(names)
        expressions=names[:min(6,len(names))]
        if names:expressions.append(f"{names[0]}**2")
        if len(names)>1:expressions.append(f"{names[0]}*{names[1]}")
        return Generated({"name":f"numeric-proxy-{seed}","expressions":expressions},0,
                         "Seeded bounded arithmetic-feature mutation; fitted only on real simulator archive")


def make_backend(config: dict) -> dict:
    domain=config.get("science_domain",config.get("domain","reactor"))
    if domain not in DOMAINS:
        raise ValueError("Unknown science domain")
    bundle={}
    for scenario_set in ("development","selection","audit"):
        bundle[scenario_set]=[]
        for case_id in config.get("science_case_ids",[0]):
            identity=registration(domain,scenario_set,case_id)["evaluator_identity"]
            bundle[scenario_set].append({"id":f"{domain}:{scenario_set}:{case_id}:{identity}","domain":domain,"scenario_set":scenario_set,
                               "case_id":case_id,"evaluator_identity":identity})
    proposer=config.get("science_proposer","seeded_parameters")
    if proposer not in {"seeded_parameters","local_model"}:raise ValueError("Unknown science proposal mode")
    model=config.get("model",{}) if proposer=="local_model" else None
    if proposer=="local_model" and not model:raise ValueError("Local model proposal requires explicit model configuration")
    proposal_space=config.get('science_proposal_space','global_only');case_ids=config.get('science_case_ids',[0])
    initial={'parameters':load_config(domain)['baseline']}
    if proposal_space=='global_and_case_routes':initial['scenario_overrides']={}
    bundle.update(backend=ScienceBackend(domain,int(config.get("science_wall_cap_ms",180000)),model,case_ids=case_ids,proposal_space=proposal_space),
                  initial=Design(initial,label="registered-baseline"),
                  features=lambda design:numeric_features(design,case_ids if proposal_space=='global_and_case_routes' else ()),
                  provenance={"replication_scope":"prospective_new_scientific_scenarios","author_exact":False,
                              "proposal_mode":"actual_local_model" if model else "seeded_parameter_mutation_control_no_llm",
                              "proposal_space":proposal_space,
                              "native_scoring":"full native utility; fixed monotone atan map for bounded core score",
                              "budget_units":{"real_items":"complete native suite attempts","tokens":"actual LLM tokens (zero in parameter control)",
                                              "wall_milliseconds":"recorded separately on every simulator receipt"}})
    return bundle
