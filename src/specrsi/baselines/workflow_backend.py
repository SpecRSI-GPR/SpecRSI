"""Real local-model execution and revision tools for workflow baselines."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path

from specrsi.core.models import Candidate,EvaluationScope,Task,thaw
from specrsi.experiments import QwenExecutor
from specrsi.models import LocalModelClient
from specrsi.tokenization import chat_token_ids
from specrsi.workflows import Workflow
from .extended_common import Design,Evaluation,Generated,ItemResult


class WorkflowBackend:
    def __init__(self,config):
        from transformers import AutoTokenizer
        self.config=dict(config)
        self.lock=json.loads(Path(config.get("lock","configs/models/qwen35_35b_a3b_bf16.lock.json")).read_text())
        self.client=LocalModelClient(config.get("base_url","http://127.0.0.1:18085"),
            model=self.lock.get("served_model_name",self.lock.get("model_name","Qwen3.5-35B-A3B")))
        self.health=self.client.health()
        if self.health.get("protocol_version",0)<2 or not self.health.get("server_enforces_total_token_cap"):
            raise ValueError("Workflow baselines require protocol v2 enforcing total token caps")
        if self.health["revision"]!=self.lock["revision"]:
            raise ValueError("Deployed executor differs from locked model revision")
        self.tokenizer=AutoTokenizer.from_pretrained(self.lock["local_path"],local_files_only=True,
            trust_remote_code=self.lock.get("trust_remote_code",False))
        self.calls=[]
        self.execution_cap=int(config.get("execution_cap",4096))
        self.generation={"max_tokens":int(config.get("max_tokens",128)),"temperature":float(config.get("temperature",0.)),
            "top_p":float(config.get("top_p",1.)),"enable_thinking":False,"max_seconds":config.get("max_seconds",120)}

    def _json_call(self,system,payload,seed,token_cap,category):
        messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,ensure_ascii=False,default=lambda x:vars(x))}]
        count=len(chat_token_ids(self.tokenizer,messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,thinking=False))
        output=int(self.config.get("proposal_max_tokens",512))
        if count+output>min(token_cap,32768):
            return Generated(None,0,"Entire configured proposal input/output cap does not fit")
        reply=self.client.generate(messages,max_tokens=output,temperature=float(self.config.get("proposal_temperature",.7)),
            top_p=1.,seed=seed,enable_thinking=False,max_seconds=self.config.get("max_seconds",120),total_token_cap=token_cap)
        tokens=reply.prompt_tokens+reply.completion_tokens
        self.calls.append({"category":category,"seed":seed,**asdict(reply)})
        if reply.prompt_tokens!=count or tokens>token_cap or reply.model_revision!=self.lock["revision"]:
            raise ValueError("Role call token/revision contract mismatch")
        if reply.finish_reason=="timeout":
            return Generated(None,tokens,"Interrupted role call")
        text=reply.text.strip()
        if text.startswith("```"):
            text="\n".join(text.splitlines()[1:-1])
        try:
            value=json.loads(text)
        except json.JSONDecodeError:
            return Generated(None,tokens,"Invalid structured role output: "+text[:240])
        return Generated(value,tokens)

    def propose(self,parent,feedback,seed,token_cap):
        instruction=("You improve an executable language-agent workflow. Return only JSON: "
            '{"workflow":{"nodes":[{"id":"solve","instruction":"...","dependencies":[]}],"output_node":"solve"},"change":"..."}. '
            "Use 1 to 8 atomic LLM-call nodes. Dependencies must form a DAG; every node must contribute to output_node. "
            "A node receives original task input and only its direct predecessors' outputs. Preserve task answer format. "
            "Do not modify the evaluator, task inputs or reference answers. Follow any supplied EMAS hypothesis scope exactly.")
        generated=self._json_call(instruction,{"parent":parent.to_dict(),"feedback":feedback},seed,token_cap,"target_proposal")
        if generated.value is None:return generated
        try:
            workflow=generated.value["workflow"]
            Workflow.from_dict(workflow)
            child=Design({"workflow":workflow},parent.uid,generated.value.get("change","workflow-revision")[:200])
        except (KeyError,TypeError,ValueError) as exc:
            return Generated(None,generated.tokens,"Invalid workflow: "+str(exc))
        return Generated(child,generated.tokens,generated.value.get("change",""))

    def diagnose(self,design,item,result,seed,token_cap):
        instruction=("Diagnose the actual workflow execution. Return JSON {\"hypotheses\":[{\"objective\":\"accuracy\","
            "\"defect\":\"unclear-instruction\",\"operation\":\"prompt-only\",\"location\":\"solve\"}]}. "
            "For an incorrect scored answer objective=accuracy; for a correct answer objective=cost. Return [] if no supported diagnosis. "
            "Permitted operations: add-node, split-node, add-edge, prompt-only, remove-edge, remove-node. "
            "Use canonical node ID or source->target edge location. Reuse consistent defect labels: missing-operation, redundant-operation, "
            "non-atomic-operation, missing-information, harmful-information, unclear-instruction, answer-format, arithmetic-error. "
            "Base hypotheses on these actual traces, never invented outcomes.")
        value=self._json_call(instruction,{"workflow":design.to_dict(),"input":item.prompt,"observed":asdict(result)},seed,token_cap,"diagnosis")
        if value.value is None:return Generated([],value.tokens,value.note)
        return Generated(value.value.get("hypotheses",[]),value.tokens,value.note)

    def propose_proxy(self,parent,archive,feature_names,seed,token_cap):
        instruction=("Evolve a cheap executable evaluator for ranking workflow designs before real validation. "
            'Return JSON {"name":"proxy-name","expressions":["feature_name","feature_a / (1 + feature_b)"]}. '
            "Use only supplied feature names, numeric constants, + - * /, constant powers 2 or 3, abs/min/max/sqrt/log1p. "
            "Return 1 to 12 bounded expressions encoding useful interactions. Host code fits coefficients on actual labels, "
            "cross-validates top-ranking quality and never treats your prediction as a real outcome.")
        return self._json_call(instruction,{"parent":parent,"actual_archive":archive,"feature_names":feature_names},seed,token_cap,"proxy_evolution")

    def evaluate(self,design,items,seed,stage,token_cap):
        Workflow.from_dict(thaw(design.payload)["workflow"])
        generation=dict(self.generation)
        candidate=Candidate(design.label,thaw(design.payload),{"model_revision":self.lock["revision"],"seed":seed,"generation":generation},
            semantic_kind="opaque",deterministic=False)
        tasks=tuple(Task(e.id,{"prompt":e.prompt},{'_specrsi_example':e.to_dict()} if e.dataset=='planbench' else e.reference,
                         {"dataset":e.dataset,"metric_variant":e.metadata.get("metric_variant")}) for e in items)
        datasets={e.dataset for e in items}
        if len(datasets)!=1:raise ValueError("One registered native dataset per evaluation")
        scope=EvaluationScope(f"baseline-{stage}",tasks,"qwen-capped-generation-v2",next(iter(datasets))+"-registered-score-v1",
            scored_fields=("answer","metrics","finish_reason","node_outputs"),replay={"generation":generation,"model_revision":self.lock["revision"]},
            split="audit" if stage=="audit" else "dev")
        executor=QwenExecutor(self.client,self.tokenizer,generation)
        used=0
        results=[]
        for item,task in zip(items,tasks):
            cap=min(self.execution_cap,token_cap-used)
            before=len(executor.physical_records)
            output=executor.run(candidate,task,scope,max(0,cap))
            used+=output.actual_cost
            records=executor.physical_records[before:]
            self.calls.extend({"category":"task_execution","stage":stage,"design_id":design.uid,"seed":seed,**r} for r in records)
            traces=tuple({"node":r.get("node","solve"),"input":item.prompt,"output":r.get("text",""),
                "prompt_tokens":r.get("prompt_tokens",0),"completion_tokens":r.get("completion_tokens",0)} for r in records)
            complete=output.status=="complete"
            results.append(ItemResult(item.id,output.outcome.score if complete else 0.,output.actual_cost,complete,
                output.outcome.admissible if complete else False,traces))
        return Evaluation(design.uid,tuple(results),seed,stage)
