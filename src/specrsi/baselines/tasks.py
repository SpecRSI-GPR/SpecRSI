"""Attach the same native multistage task to official prompt optimizers."""
import json
from pathlib import Path
from specrsi.task_pipelines import TaskPipelineExecutor
from specrsi.core.models import Candidate,Task,EvaluationScope,thaw
from specrsi.benchmarks.base import ScoreResult

def attach_pipeline(model,lock:dict,*,dataset=None,native_config=None):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(lock['local_path'],local_files_only=True,trust_remote_code=lock.get('trust_remote_code',False))
    config={'max_tokens':model.max_tokens,'temperature':model.temperature,'enable_thinking':False,**(native_config or {})}
    model.task_protocol={'dataset':dataset,'metric_variant':
        'official_math_with_prospective_local_AMPS_fallback' if dataset=='livebench_math' else
        'GEPA_three_hop_all_supporting_documents' if dataset=='hover' else
        'prospective_local_PAPILLON_judge' if dataset=='pupa' else 'task_native',
        'judge_revision':lock['revision'] if dataset in {'livebench_math','pupa'} else None,
        'generation':config}
    if dataset in {'swe_verified','swe_lite'}:
        from specrsi.coding_agent import SWERepositoryExecutor
        executor=SWERepositoryExecutor(model.client,tokenizer,config)
    else:
        executor=TaskPipelineExecutor(model.client,tokenizer,config)
    model.model_revision=lock['revision']
    def evaluate(instruction,example):
        candidate=Candidate('baseline-prompt',{'system_prompt':instruction},
            {'model_revision':lock['revision'],'seed':model.seed,'generation':config},semantic_kind='opaque',deterministic=False)
        task=Task(example.id,{'prompt':example.prompt},example.reference,{'dataset':example.dataset})
        scope=EvaluationScope('baseline-'+example.dataset,(task,),'local-native-pipeline-v1',example.dataset+'-'+model.task_protocol['metric_variant'],split='dev' if example.split=='development' else example.split)
        before=len(executor.physical_records)
        result=executor.run(candidate,task,scope,int(config.get('execution_cap',131072)))
        for raw in executor.physical_records[before:]:
            record={**raw,'role':example.split,'example_id':example.id,'call_index':len(model.records),'status':result.status}
            model.records.append(record)
            if model.journal:
                model.journal.parent.mkdir(parents=True,exist_ok=True)
                with model.journal.open('a') as out: out.write(json.dumps(record,ensure_ascii=False)+'\n')
        receipt={'example_id':example.id,'split':example.split,'actual_tokens':result.actual_cost,
            'status':result.status,'receipt':thaw(result.receipt),'error':result.error}
        model.native_receipts.append(receipt)
        if model.journal:
            with model.journal.with_name('native_receipts.jsonl').open('a') as out:
                out.write(json.dumps(receipt,ensure_ascii=False)+'\n')
        if result.status!='complete': raise RuntimeError('Native baseline scenario incomplete: '+str(result.error))
        return result.outcome.fields['answer'],ScoreResult(result.outcome.score,thaw(result.outcome.fields['metrics']))
    model.task_evaluator=evaluate
    model.native_receipts=[]
