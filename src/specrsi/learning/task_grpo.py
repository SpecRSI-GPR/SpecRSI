"""Task-actor GRPO: train answer-generation LoRA, never SpecRSI control heads.

The sampling distribution has full vocabulary support. Frozen external judges
are separate from the trainable actor, including multistage native tasks.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json,math,time,uuid
from pathlib import Path
import torch
from torch.nn import functional as F
from specrsi.models.client import ModelReply,LocalModelClient
from specrsi.core.models import Candidate,Task,EvaluationScope,thaw
from specrsi.tokenization import chat_token_ids
from .objectives import group_advantages,clipped_surrogate


@dataclass(frozen=True)
class TaskSequence:
    prompt_ids:tuple[int,...]
    generated_ids:tuple[int,...]
    behavior_logprobs:tuple[float,...]
    temperature:float
    request_id:str

    def __post_init__(self):
        if not self.prompt_ids or not self.generated_ids or len(self.generated_ids)!=len(self.behavior_logprobs):
            raise ValueError('Task policy records need complete exact token continuations')
        if self.temperature<=0 or not all(math.isfinite(x) and x<=0 for x in self.behavior_logprobs):
            raise ValueError('Invalid task sampling probabilities')


def sequence_logprobs(model,sequence):
    """Only materialize vocabulary logits for generated positions, not prompts."""
    model._activate('draft')
    ids=sequence.prompt_ids+sequence.generated_ids
    tokens=torch.tensor([ids[:-1]],dtype=torch.long,device=model.device)
    base=model.backbone.get_base_model()
    hidden=base.base_model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True).last_hidden_state
    logits=base.lm_head(hidden[0,len(sequence.prompt_ids)-1:]).float()/sequence.temperature
    targets=torch.tensor(sequence.generated_ids,dtype=torch.long,device=model.device)
    return F.log_softmax(logits,-1).gather(1,targets[:,None]).squeeze(1)


class HFTaskClient:
    """In-process fixed-snapshot client; conventional request token accounting."""
    def __init__(self,model,revision):
        self.model=model;self.revision=revision;self.sequences=[];self.calls=[];self.episode_call_index=0

    @torch.no_grad()
    def generate(self,messages,*,max_tokens=512,temperature=1.,top_p=1.,seed=0,
                 enable_thinking=False,max_seconds=120,total_token_cap=None):
        # This protocol intentionally fixes nucleus filtering off. A request
        # from a generic task executor cannot silently change behavior support.
        if temperature<=0:raise ValueError('Task GRPO requires positive sampling temperature')
        model=self.model;model.eval();model._activate('draft')
        prompt=tuple(chat_token_ids(model.tokenizer,messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=enable_thinking,thinking=enable_thinking))
        if len(prompt)+max_tokens>model.config.max_length or (total_token_cap is not None and len(prompt)+max_tokens>total_token_cap):
            raise ValueError('Frozen task prompt and output contract exceed actor context/token cap')
        generator=torch.Generator(device=model.device);generator.manual_seed((seed+1000003*self.episode_call_index)%(2**63-1))
        self.episode_call_index+=1
        prefix=list(prompt);generated=[];logs=[];cache=None;start=time.monotonic();finish='length'
        eos=model.backbone.config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
        base=model.backbone.get_base_model()
        for step in range(max_tokens):
            if time.monotonic()-start>=max_seconds:finish='timeout';break
            tokens=torch.tensor([prefix if cache is None else [prefix[-1]]],device=model.device,dtype=torch.long)
            output=base.base_model(input_ids=tokens,attention_mask=torch.ones((1,len(prefix)),device=model.device,dtype=torch.long),
                                   past_key_values=cache,use_cache=True,return_dict=True)
            cache=output.past_key_values
            logprobs=F.log_softmax(base.lm_head(output.last_hidden_state[0,-1]).float()/temperature,-1)
            token=int(torch.multinomial(logprobs.exp(),1,generator=generator));logs.append(float(logprobs[token]))
            generated.append(token);prefix.append(token)
            if token in eos:finish='stop';break
        request_id='task-grpo-'+uuid.uuid4().hex
        if generated:self.sequences.append(TaskSequence(prompt,tuple(generated),tuple(logs),temperature,request_id))
        reply=ModelReply(model.tokenizer.decode(generated,skip_special_tokens=True),len(prompt),len(generated),
                         time.monotonic()-start,finish,self.revision,request_id)
        self.calls.append(asdict(reply));return reply


def update_group(model,optimizer,sequences,rewards,*,clip_epsilon=.2,kl_weight=.01,max_grad_norm=1.):
    if len(sequences)!=len(rewards) or any(not group for group in sequences):
        raise ValueError('Each complete task episode needs its actual actor token records')
    advantages=group_advantages(torch.tensor(rewards,dtype=torch.float32),'grpo')
    if not bool(advantages.abs().any()) and kl_weight==0:
        return {'updated':False,'reason':'equal_native_rewards','rewards':rewards,'loss':0.,'grad_norm':0.}
    model.eval();optimizer.zero_grad(set_to_none=True);loss_total=0.;kl_total=0.
    for records,advantage in zip(sequences,advantages):
        n=sum(len(record.generated_ids) for record in records)
        for record in records:
            current=sequence_logprobs(model,record)
            old=current.new_tensor(record.behavior_logprobs)
            policy=-clipped_surrogate(current,old,current.new_tensor(float(advantage)),clip_epsilon)
            if kl_weight:
                with torch.no_grad(),model.backbone.disable_adapter():
                    # Avoid reactivating an adapter inside the disabled context.
                    ids=record.prompt_ids+record.generated_ids;tokens=torch.tensor([ids[:-1]],device=model.device)
                    base=model.backbone.get_base_model()
                    hidden=base.base_model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False,return_dict=True).last_hidden_state
                    logits=base.lm_head(hidden[0,len(record.prompt_ids)-1:]).float()/record.temperature
                    reference=F.log_softmax(logits,-1).gather(1,current.new_tensor(record.generated_ids,dtype=torch.long)[:,None]).squeeze(1)
                difference=reference-current;kl=difference.exp()-difference-1
                loss=(policy+kl_weight*kl).sum()/n/len(rewards)
                kl_total+=float(kl.detach().sum())/n/len(rewards)
            else:loss=policy.sum()/n/len(rewards)
            loss_total+=float(loss.detach());loss.backward()
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
    updated=bool(norm>0)
    if updated:optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {'updated':updated,'rewards':rewards,'loss':loss_total,'kl':kl_total,'grad_norm':float(norm),
            'equal_native_rewards':not bool(advantages.abs().any())}


def make_task_executor(model,config):
    from specrsi.experiments import QwenExecutor
    from specrsi.task_pipelines import TaskPipelineExecutor
    revision=config['model']['revision']+':task-actor-initial'
    actor=HFTaskClient(model,revision)
    pipeline=config.get('task_pipeline')
    generation={**config.get('generation',{}),'temperature':config.get('temperature',1.),'top_p':1.,'enable_thinking':False}
    executor=(TaskPipelineExecutor if pipeline else QwenExecutor)(actor,model.tokenizer,generation)
    if config['dataset'] in {'pupa','livebench_math'}:
        judge_config=config['judge'];lock=json.loads(Path(judge_config['lock']).read_text())
        judge=LocalModelClient(judge_config['base_url'],lock.get('served_model_name','Qwen3.5-35B-A3B'))
        if judge.health()['revision']!=lock['revision']:raise ValueError('Frozen task reward judge revision mismatch')
        from transformers import AutoTokenizer
        judge_tokenizer=AutoTokenizer.from_pretrained(lock['local_path'],local_files_only=True,trust_remote_code=lock.get('trust_remote_code',False))
        for stage in ('quality_judge','leakage_judge','amps_equivalence_judge'):
            executor.stage_clients[stage]=judge;executor.stage_revisions[stage]=lock['revision']
            executor.stage_tokenizers[stage]=judge_tokenizer
        generation['fixed_reward_judge']=judge_config
    return actor,executor,generation


def evaluate_task(actor,executor,generation,example,instruction,seed,revision,cap):
    actor.revision=revision;actor.episode_call_index=0;before=len(actor.sequences)
    reference={'_specrsi_example':example.to_dict()} if example.dataset=='planbench' else example.reference
    task=Task(example.id,{'prompt':example.prompt},reference,{'dataset':example.dataset,'split':example.split})
    scope=EvaluationScope('task-grpo-'+example.split,(task,),'local-task-actor-grpo-v1',example.dataset+'-registered-native',
                          split='dev' if example.split=='development' else example.split)
    candidate=Candidate('task-policy',{'system_prompt':instruction},{'model_revision':revision,'seed':seed,'generation':generation},
                        semantic_kind='opaque',deterministic=False)
    result=executor.run(candidate,task,scope,cap)
    row={'id':example.id,'split':example.split,'score':result.outcome.score if result.outcome else None,
         'status':result.status,'cost':result.actual_cost,'receipt':thaw(result.receipt),'error':result.error}
    return row,actor.sequences[before:]


def run_task_grpo(config):
    from .hf import HFRoleModel,HFModelConfig
    from specrsi.benchmarks import load_examples
    output=Path(config['output'])
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh task-model output directory')
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config.get('seed',0));torch.set_num_threads(config.get('cpu_threads',4))
    paths=sorted(Path('src/specrsi').rglob('*.py'))
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    splits={role:load_examples(config['dataset'],role,verify=role=='development')[:config['sizes'][role]]
            for role in ('development','selection','audit')}
    if any(not values for values in splits.values()):raise ValueError('Task GRPO needs all three nonempty registered splits')
    if config.get('group_size',4)<2:raise ValueError('Task GRPO needs independent group members')
    from specrsi.benchmarks.data import validate_disjoint_splits
    validate_disjoint_splits(splits)
    model=HFRoleModel(HFModelConfig(**config['model']));model.eval()
    actor,executor,generation=make_task_executor(model,config)
    parameters=[p for name,p in model.backbone.named_parameters() if 'lora_' in name and '.draft.' in name]
    if not parameters:raise ValueError('No task actor LoRA parameters')
    optimizer=torch.optim.AdamW(parameters,lr=config.get('learning_rate',2e-5),weight_decay=config.get('weight_decay',1e-4))
    initial_heads={name:p.detach().cpu().clone() for name,p in model.named_parameters() if not name.startswith('backbone.')}
    verify_before={name:p.detach().cpu().clone() for name,p in model.backbone.named_parameters() if 'lora_' in name and '.verify.' in name}
    manifest={'kind':'task_model_GRPO','not_SpecRSI_improver_training':True,'sampling':'full_vocabulary_temperature_softmax_no_nucleus',
        'config':config,'sources':sources,'splits':{k:[e.id for e in v] for k,v in splits.items()},
        'trainable_parameters':[name for name,p in model.backbone.named_parameters() if any(p is q for q in parameters)],
        'reference_policy':'frozen_pretrained_base_with_all_adapters_disabled','author_exact':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    history=[];steps=0;best=-math.inf;selected_state=None;start=time.monotonic()
    instruction=config['instruction'];cap=config.get('execution_cap_tokens',131072)
    recorded_physical=0
    def run_one(example,seed,revision):
        nonlocal recorded_physical
        try:return evaluate_task(actor,executor,generation,example,instruction,seed,revision,cap)
        finally:
            with (output/'physical_calls.jsonl').open('a') as f:
                for row in executor.physical_records[recorded_physical:]:f.write(json.dumps(row)+'\n')
            recorded_physical=len(executor.physical_records)
    def evaluate_split(role,epoch):
        rows=[]
        for index,example in enumerate(splits[role]):
            row,_=run_one(example,config.get('seed',0)+index,config['model']['revision']+f':task-actor-step-{steps}')
            if row['status']!='complete':raise RuntimeError('Task model native evaluation incomplete: '+str(row))
            rows.append(row)
        return rows
    # Include the unmodified model as an eligible selected checkpoint.
    for epoch in range(config.get('epochs',3)+1):
        if epoch:
            for index,example in enumerate(splits['development']):
                records=[];rewards=[];rows=[]
                for member in range(config.get('group_size',4)):
                    seed=config.get('seed',0)+1000000*epoch+10000*index+member
                    row,sequence=run_one(example,seed,config['model']['revision']+f':task-actor-step-{steps}')
                    rows.append(row)
                    if row['status']!='complete':
                        (output/f'failed-epoch-{epoch}-item-{index}.json').write_text(json.dumps(rows,indent=2))
                        raise RuntimeError('Incomplete development group; no rewards fabricated')
                    records.append(sequence);rewards.append(row['score'])
                metrics=update_group(model,optimizer,records,rewards,clip_epsilon=config.get('clip_epsilon',.2),kl_weight=config.get('kl_weight',.01))
                steps+=int(metrics['updated']);history.append({'epoch':epoch,'item':example.id,'steps':steps,**metrics})
                with (output/'episodes.jsonl').open('a') as f:
                    f.write(json.dumps({'epoch':epoch,'item':example.id,'outcomes':rows,'sequences':[[asdict(s) for s in group] for group in records]})+'\n')
                print(json.dumps(history[-1]),flush=True)
        rows=evaluate_split('selection',epoch);quality=sum(row['score'] for row in rows)/len(rows)
        (output/f'selection-{epoch}.json').write_text(json.dumps(rows,indent=2))
        if quality>best:
            best=quality;selected_state={name:p.detach().cpu().clone() for name,p in model.backbone.named_parameters() if 'lora_' in name and '.draft.' in name}
            model.save(output/'selected_checkpoint')
            (output/'selected.json').write_text(json.dumps({'epoch':epoch,'training_steps':steps,'selection_score':quality,
                'checkpoint':'selected_checkpoint','kind':'task_actor_not_SpecRSI_roles',
                'checkpoint_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((output/'selected_checkpoint').rglob('*')) if p.is_file()}},indent=2)+'\n')
    assert selected_state is not None
    with torch.no_grad():
        for name,p in model.backbone.named_parameters():
            if name in selected_state:p.copy_(selected_state[name].to(p.device))
    selected_manifest=json.loads((output/'selected.json').read_text())
    steps=selected_manifest['training_steps']
    audit=evaluate_split('audit',selected_manifest['epoch'])
    if any(hashlib.sha256((output/p).read_bytes()).hexdigest()!=h for p,h in selected_manifest['checkpoint_sha256'].items()):
        raise RuntimeError('Frozen selected task checkpoint changed during audit')
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    unchanged=all(torch.equal(p.detach().cpu(),initial_heads[name]) for name,p in model.named_parameters() if name in initial_heads)
    unchanged=unchanged and all(torch.equal(p.detach().cpu(),verify_before[name]) for name,p in model.backbone.named_parameters() if name in verify_before)
    if not unchanged:raise RuntimeError('Task GRPO modified an SpecRSI head or Verify adapter')
    physical=executor.physical_records
    (output/'physical_calls.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in physical))
    summary={'status':'complete','training_steps':sum(int(x['updated']) for x in history),'history':history,
        'selected':selected_manifest,'audit_mean':sum(row['score'] for row in audit)/len(audit),
        'task_model_only':True,'SpecRSI_heads_and_verify_adapter_unchanged':unchanged,
        'actual_tokens':sum(row['prompt_tokens']+row['completion_tokens'] for row in physical),
        'inference_gpu_seconds':sum(row['elapsed_seconds']*row.get('gpu_count',1) for row in physical),
        'train_and_evaluation_wall_seconds':time.monotonic()-start,
        'source_changed_during_run':sources!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src/specrsi').rglob('*.py'))}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
