"""Native coding instruction domain for substantive adaptive baseline algorithms.

The editable artifact is the same system instruction as SpecRSI's opaque coding
adaptation. Native repository/pipeline actions and graders remain fixed.
"""
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

from specrsi.benchmarks import load_examples,validate_disjoint_splits
from specrsi.benchmarks.base import BenchmarkExample
from specrsi.core.models import Candidate,Task,EvaluationScope,thaw
from .extended_common import Design,Evaluation,Generated,ItemResult
from .workflow_backend import WorkflowBackend


def native_splits(config):
    """Freeze full disjoint registries before applying requested smoke limits."""
    dataset=config['dataset'];limits=config.get('data',{});root=Path(limits.get('root',config.get('data_root','data')))
    if dataset in {'swe_verified','swe_lite'}:
        import pyarrow.parquet as pq
        registry_path=Path(config.get('coding_registry','configs/coding/prospective_swe_tracks.json'))
        registry=json.loads(registry_path.read_text())
        # Official Lite development 23 is disjoint from all 707 test instances.
        raw={row['instance_id']:row for name in ('swe_lite','swe_verified')
             for file in sorted((root/'raw'/name).glob('*.parquet')) for row in pq.read_table(file).to_pylist()}
        ids={'development':config.get('development_ids',registry['development23']),
             'selection':config.get('selection_ids',registry[config.get('selection_track','selection60')]),
             'audit':config.get('audit_ids',registry[config.get('audit_track','lite_audit_filtered')])}
        parts={split:tuple(BenchmarkExample(key,dataset,split,
            raw[key]['problem_statement'],raw[key],{'source_dataset':'swe_lite' if split!='selection' else dataset,
                'registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest()})
            for key in keys) for split,keys in ids.items()}
    else:
        parts={split:tuple(load_examples(dataset,split,data_root=root,verify=index==0))
               for index,split in enumerate(('development','selection','audit'))}
    validate_disjoint_splits(parts)
    return {split:values[:limits.get(split,len(values))] for split,values in parts.items()}


class NativeInstructionBackend(WorkflowBackend):
    def __init__(self,config):
        super().__init__(config['model'])
        self.dataset=config['dataset']
        if self.dataset not in {'swe_verified','swe_lite','polyglot','hover','pupa'}:
            raise ValueError('Native instruction backend needs a registered native pipeline')
        # Preserve actor/test/retrieval configuration in the frozen runtime.
        self.generation={**self.config,**self.generation}
        self.generation.setdefault('served_model_name',self.lock.get('served_model_name','Qwen3.5-35B-A3B'))
        self.generation.setdefault('revision',self.lock['revision'])
        from specrsi.coding_agent import SWERepositoryExecutor
        from specrsi.task_pipelines import TaskPipelineExecutor
        executor=SWERepositoryExecutor if self.dataset.startswith('swe_') else TaskPipelineExecutor
        self.executor=executor(self.client,self.tokenizer,self.generation)
        self.native_receipts=[]

    def propose(self,parent,feedback,seed,token_cap):
        instruction=('Improve the system instruction of a fixed native coding/task agent. '
            'Return only JSON {"system_prompt":"complete revised strategy","change":"concise change"}. '
            'The tool interface, task inputs, code execution, and native grader are fixed. '
            'Optimize a general strategy, never hard-code an individual answer or modify evaluation tests. '
            'If an EMAS hypothesis is supplied, change only the strategy instruction to address that recurring defect.')
        result=self._json_call(instruction,{'domain':self.dataset,'parent':parent.to_dict(),'feedback':feedback},seed,token_cap,'target_proposal')
        if result.value is None:return result
        value=result.value
        if not isinstance(value,dict) or not isinstance(value.get('system_prompt'),str) or not 1<=len(value['system_prompt'].strip())<=12000:
            return Generated(None,result.tokens,'Invalid native instruction proposal')
        child=Design({'system_prompt':value['system_prompt']},parent.uid,str(value.get('change','strategy revision'))[:200])
        return Generated(child,result.tokens,child.label)

    def diagnose(self,design,item,result,seed,token_cap):
        instruction=('Diagnose this actual native agent execution. Return only JSON {"hypotheses":['
            '{"objective":"accuracy","defect":"canonical recurring defect","operation":"prompt-only","location":"strategy"}]}. '
            'Only the strategy instruction is editable. For score below 1 objective=accuracy; otherwise objective=cost. '
            'Use consistent specific defect names, supported by observed output. Return an empty list if unsupported. '
            'Never invent test outcomes or request changes to the grader.')
        generated=self._json_call(instruction,{'strategy':design.to_dict(),'public_task':item.prompt,'observed':asdict(result)},seed,token_cap,'diagnosis')
        records=generated.value.get('hypotheses',[]) if isinstance(generated.value,dict) else []
        records=[r for r in records if isinstance(r,dict) and r.get('operation')=='prompt-only' and r.get('location')=='strategy']
        return Generated(records,generated.tokens,generated.note)

    def evaluate(self,design,items,seed,stage,token_cap):
        payload=thaw(design.payload)
        if set(payload)!={'system_prompt'} or not isinstance(payload['system_prompt'],str) or not payload['system_prompt'].strip():
            raise ValueError('Native policy artifact must be one nonempty system_prompt')
        candidate=Candidate(design.label,payload,{'model_revision':self.lock['revision'],'seed':seed,
            'generation':self.generation},semantic_kind='opaque',deterministic=False)
        tasks=tuple(Task(e.id,{'prompt':e.prompt},e.reference,{'dataset':e.dataset,'split':e.split,
            'content_hash':e.content_hash}) for e in items)
        scope=EvaluationScope('native-baseline-'+stage,tasks,'fixed-native-instruction-pipeline-v1',
            self.dataset+'-native-grader-v1',split='audit' if stage=='audit' else 'dev',
            replay={'generation':self.generation,'model_revision':self.lock['revision']})
        used=0;results=[]
        self.executor.validate_prompt_candidate(candidate,tasks,self.execution_cap)
        for item,task in zip(items,tasks):
            cap=min(self.execution_cap,token_cap-used)
            if cap<1 or self.dataset.startswith('swe_') and cap<2048:
                results.append(ItemResult(item.id,0.,0,False,False,({'error':'Native item cannot fit remaining cap'},)))
                continue
            before=len(self.executor.physical_records)
            output=self.executor.run(candidate,task,scope,cap)
            records=self.executor.physical_records[before:]
            self.calls.extend({'category':'native_execution','stage':stage,'design_id':design.uid,'seed':seed,**r} for r in records)
            used+=output.actual_cost
            complete=output.status=='complete'
            receipt={'design_id':design.uid,'item_id':item.id,'stage':stage,'status':output.status,
                'actual_tokens':output.actual_cost,'receipt':thaw(output.receipt),'error':output.error}
            self.native_receipts.append(receipt)
            # Only observed output, score and native metrics enter optimization;
            # gold patches, test patches and sealed references stay in the grader.
            trace=({'input':item.prompt,'output':thaw(output.outcome.fields) if complete else None,
                'actual_tokens':output.actual_cost,'error':output.error},)
            results.append(ItemResult(item.id,output.outcome.score if complete else 0.,output.actual_cost,
                complete,output.outcome.admissible if complete else False,trace))
        return Evaluation(design.uid,tuple(results),seed,stage)


def make_backend(config):
    parts=native_splits(config)
    return {'backend':NativeInstructionBackend(config),**parts,
        'initial':Design({'system_prompt':config.get('initial_prompt','Inspect the task, implement the requested behavior, and validate it with native tools.')},label='initial-strategy'),
        'provenance':{'implementation':'prospective_native_instruction_search_not_author_historical_agent_code',
            'editable_artifact':'system_prompt','reuse':'all_U_opaque_native_execution',
            'selection_policy':'frozen_complete_native_scores','audit_policy':'detached_after_selected_artifact_frozen',
            'split_ids':{split:[e.id for e in items] for split,items in parts.items()},
            'emas_domain':'prompt-only at strategy; fixed executor topology'}}
