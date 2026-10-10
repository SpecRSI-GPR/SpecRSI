"""Actual HF role collection on complete native scientific development cases."""
from __future__ import annotations
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import time
import torch
from specrsi.core.budget import BudgetLedger
from specrsi.core.ledger import Ledger
from specrsi.core.models import thaw
from specrsi.core.search import DraftResult,SearchConfig,SearchEngine
from specrsi.planning.batches import MenuBuilder
from specrsi.science.executor import make_scope,make_candidate,ScienceExecutor,mutate,selected_parameters
from specrsi.science.checker import ScienceParameterChecker
from .bridge import NeuralSearchBridge
from .records import CheckerAudit,validate_group
from .sampler import sample_edit_tokens


def check_context(source,child,task,scope,cap):
    return json.dumps({'checker':ScienceParameterChecker.version,'source_policy':'paid-parent-first-one-source',
        'source':source.manifest(),'child':child.manifest(),'task_id':task.uid,'task_inputs':thaw(task.inputs),
        'scope_id':scope.uid,'cap_ms':cap},sort_keys=True,separators=(',',':'))


class HFParameterDrafter:
    def __init__(self,bridge,*,max_new_tokens=384,top_k=32,temperature=1.,exploration=.1,check_cap_ms=1000,snapshot='science-roles',proposal_mode='numeric_catalog',catalog_size=12):
        if bridge.cost_unit!='wall-ms' or bridge.model.backend!='hf':raise ValueError('Native HF science uses measured wall-ms')
        self.bridge=bridge;self.max_new_tokens=max_new_tokens;self.top_k=top_k;self.temperature=temperature
        self.exploration=exploration;self.check_cap_ms=check_cap_ms;self.snapshot=snapshot
        if proposal_mode not in {'numeric_catalog','freeform_json'}:raise ValueError('Unknown scientific proposal mode')
        self.proposal_mode=proposal_mode;self.catalog_size=catalog_size

    def catalog(self,parent,state,rng):
        from specrsi.science.proposals import parameter_catalog
        return parameter_catalog(parent.runtime['domain'],thaw(parent.artifacts['parameters']),
            [selected_parameters(parent,task,state.scope) for task in state.scope.tasks],rng,self.catalog_size)

    def _draft_input(self,parent,state,instruction,output_cap):
        """Keep task-index routing and actual parent feedback in the text input.

        V3 role context stores task/result text on graph nodes. The autoregressive
        drafter has no graph input, so it needs its own observed-only context.
        Never truncate a parameter schema, route map or finite edit catalog.
        """
        from .draft_context import parent_feedback_context
        model=self.bridge.model
        routes=[{'task_index':index,'task_id':task.uid[:12],'visible_input':thaw(task.inputs)}
                for index,task in enumerate(state.scope.tasks)]
        prefix=(instruction+'\nCurrent parent: '+json.dumps(thaw(parent.artifacts),separators=(',',':'))+
                '\nRegistered task routes: '+json.dumps(routes,separators=(',',':'))+
                '\nObserved parent feedback: ')

        def template(text):
            return model.tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False)

        input_cap=model.config.max_length-output_cap
        feedback_cap=input_cap-len(template(prefix))-4
        while feedback_cap>0:
            try:
                feedback=parent_feedback_context(self.bridge,state,parent,token_limit=feedback_cap)
            except ValueError as error:
                if 'Draft feedback token budget' not in str(error):raise
                return None
            ids=template(prefix+feedback)
            if len(ids)<=input_cap:return ids
            feedback_cap-=len(ids)-input_cap+1
        return None

    @torch.no_grad()
    def draft(self,parent,state,cap,rng):
        model=self.bridge.model;start=time.monotonic()
        if cap<=0:return DraftResult(None,0,status='interrupted',error='No native draft slice')
        instruction=('Improve one scientific task route. Return ONLY JSON {"task_index": integer, "parameters": object}. '
            'Use the registered zero-based task_index; -1 changes global parameters. '
            'Supply the complete parameter object with exactly the existing keys/types. '
            'Change one useful numeric parameter. Do not modify the simulator, constraints, task inputs or scoring. '
            'Use only the observed parent feedback; a missing result is not a failed check.')
        catalog=None;grammar=None;eos=None;output_cap=self.max_new_tokens
        snapshot=self.snapshot
        if self.proposal_mode=='numeric_catalog':
            import hashlib
            catalog=self.catalog(parent,state,rng)
            choices=[{'edit':i,'task_index':x['task_index'],'changes':x['changes']} for i,x in enumerate(catalog)]
            instruction=('Choose the most promising feasible scientific parameter edit. Return ONLY {"edit": integer}. '
                'Edits with task_index=-1 change the global parameters; other edits change one registered case route. '
                'Do not modify the simulator, constraints, task inputs or scoring. '
                'Every candidate is tested by the unchanged complete simulator. '
                'Use only the observed parent feedback; a missing result is not a failed check.'+
                '\nFeasible numeric edits: '+json.dumps(choices,separators=(',',':')))
            snapshot+=':catalog='+hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest()
            eos=getattr(model.tokenizer,'eos_token_id',None)
            if eos is None:raise ValueError('Constrained numeric draft requires tokenizer EOS')
            sequences=[tuple(model.tokenizer.encode(json.dumps({'edit':i},separators=(',',':')),add_special_tokens=False))+ (eos,) for i in range(len(catalog))]
            output_cap=max(map(len,sequences))
            grammar=lambda prefix: sorted({seq[len(prefix)] for seq in sequences if seq[:len(prefix)]==prefix and len(seq)>len(prefix)})
        ids=self._draft_input(parent,state,instruction,output_cap)
        if ids is None:
            return DraftResult(None,max(1,math.ceil((time.monotonic()-start)*1000)),status='interrupted',
                error='Scientific parameter schema, task routes and observed feedback do not fit draft context')
        room=min(output_cap,model.config.max_length-len(ids))
        if room<1:return DraftResult(None,math.ceil((time.monotonic()-start)*1000),status='interrupted',error='Draft context full')
        sample=sample_edit_tokens(model,ids,max_new_tokens=room,rng=rng,top_k=self.top_k,temperature=self.temperature,
            exploration=self.exploration,proposal_snapshot=snapshot,transition_index=getattr(state,'current_transition_index',state.version),deadline_monotonic=start+cap/1000,
            token_grammar=grammar,eos_token_ids=(eos,) if eos is not None else None)
        self.bridge.token_decisions.extend(sample.decisions)
        self.bridge.physical_inferences.append({'kind':'learned_control','role':'native_parameter_draft',
            'elapsed_seconds':time.monotonic()-start,'gpu_count':1 if str(model.device).startswith('cuda') else 0,
            'prompt_tokens':len(ids),'completion_tokens':len(sample.generated_ids),'proposal_mode':self.proposal_mode,'catalog':catalog})
        def cost():return max(1,math.ceil((time.monotonic()-start)*1000))
        if sample.stop_reason=='deadline' or cost()>cap:
            return DraftResult(None,cost(),status='interrupted',error='Cooperative generation deadline; actual cost retained')
        try:
            raw=sample.text.strip()
            if raw.startswith('```'):raw='\n'.join(raw.splitlines()[1:-1])
            parsed=json.loads(raw)
            if catalog is not None:
                if set(parsed)!={'edit'} or type(parsed['edit']) is not int or not 0<=parsed['edit']<len(catalog):raise ValueError('Invalid finite proposal selection')
                chosen=catalog[parsed['edit']];parsed={k:chosen[k] for k in ['task_index','parameters']}
            if set(parsed)!={'task_index','parameters'} or type(parsed['task_index']) is not int:raise ValueError('Expected task_index and parameters only')
            index=parsed['task_index']
            if not -1<=index<len(state.scope.tasks):raise ValueError('Invalid task route')
            params=parsed['parameters']
            if not isinstance(params,dict) or set(params)!=set(parent.artifacts['parameters']):raise ValueError('Changed parameter schema')
            overrides=thaw(parent.artifacts['scenario_overrides'])
            baseline=thaw(parent.artifacts['parameters'])
            if index<0:baseline=params
            else:
                task=state.scope.tasks[index];overrides.pop(task.uid,None)
                overrides['case:'+str(task.inputs['case_id'])]=params
            child=make_candidate(parent.runtime['domain'],baseline,f'hf-science-{state.version}',parent.uid,overrides)
        except (ValueError,KeyError,TypeError) as exc:
            return DraftResult(None,cost(),status='error',error='Invalid native parameter edit: '+str(exc))
        if hasattr(self.bridge, 'record_executed_edit'):
            self.bridge.record_executed_edit(child, parent, sample)
        forecasts={}
        if getattr(self.bridge, 'direct_only', False):
            return DraftResult(child,cost(),forecasts=forecasts)
        for task in state.scope.tasks:
            if cost()+1000>cap:break
            context=self.bridge.bounded_text(check_context(parent,child,task,state.scope,self.check_cap_ms))
            t=time.monotonic();forecasts[task.uid]=float(model.surface_logits(self.bridge.state_features(state),context).sigmoid())
            self.bridge.physical_inferences.append({'kind':'learned_control','role':'native_surface','elapsed_seconds':time.monotonic()-t,
                'gpu_count':1 if str(model.device).startswith('cuda') else 0,'prompt_tokens':len(model.tokenizer.encode(context)),'completion_tokens':0})
        if cost()>cap:return DraftResult(None,cost(),status='interrupted',error='Surface inference exceeded cooperative draft cap')
        return DraftResult(child,cost(),forecasts=forecasts)


class AuditedScienceChecker(ScienceParameterChecker):
    def __init__(self,bridge,lineage_id):
        super().__init__();self.bridge=bridge;self.lineage_id=lineage_id;self.state=None;self.audits=[]

    def check(self,source,child,task,scope,cap):
        context=self.bridge.bounded_text(check_context(source,child,task,scope,cap))
        features=self.bridge.state_features(self.state)
        result=super().check(source,child,task,scope,cap)
        self.audits.append(CheckerAudit(self.lineage_id,self.version+f':paid-parent-first-one-source:{cap}ms',
            result.status!='interrupted',int(result.status=='P') if result.status!='interrupted' else None,features,context))
        return result


def collect_science_specrsi(model,round_index,config):
    if model.backend!='hf':raise ValueError('Science role collection requires real HF backbone')
    c=config['collection'];domain=c['domain'];seed=config.get('training',{}).get('seed',0)
    from .continuations import ForcedCollectionConfig,FrozenRoleGuard,collect_paid_continuations,role_snapshot_id
    teacher=config.get('stage')=='warmup' and c.get('demonstration_policy') is not None
    from .demonstrations import demonstration_bridge_kwargs
    teacher_kwargs=demonstration_bridge_kwargs(c) if teacher else {}
    forced_config=ForcedCollectionConfig.from_settings(c,teacher=teacher)
    guard=FrozenRoleGuard(model)
    model.eval()
    from specrsi.paper_profile import algorithm_name, vats_config, vats_cost_accounting
    algorithm=algorithm_name(c)
    collection_snapshot=role_snapshot_id(model,f'science-frozen-round-{round_index}') if algorithm=='vats_pdf' else f'science-frozen-round-{round_index}'
    bridge_class,engine_class=NeuralSearchBridge,SearchEngine
    if algorithm=='vats_pdf':
        from specrsi.core.vats import VATSEngine
        from .paper_bridge import VATSSearchBridge
        bridge_class,engine_class=VATSSearchBridge,VATSEngine
        if teacher:
            from .demonstrations import DemonstrationQuality,DemonstrationVATSBridge
            DemonstrationQuality.from_settings(c.get('demonstration_quality'))
            bridge_class=DemonstrationVATSBridge
    elif teacher or forced_config.repetitions:
        raise ValueError('Demonstrations and paid forced continuations require the PDF VATS collector')
    group_size=c.get('group_size',2)
    if group_size<2:raise ValueError('Independent full-search group requires >=2 members')
    # Validate the registered work support, policy law and all resource caps
    # before acquiring even the initial root's native simulator outcomes.
    search_config=SearchConfig(execution_cap=c.get('execution_cap_ms',120000),draft_cap=c.get('draft_cap_ms',30000),
        check_cap=c.get('check_cap_ms',1000),max_sources=0 if c.get('all_u') else 1,all_u=c.get('all_u',False),
        planning_cost=0,outer_value_cap=c.get('head_cap_ms',5000),batch_value_cap=c.get('head_cap_ms',5000),
        max_candidates=c.get('max_candidates',3),max_children=c.get('max_children',2),max_actions=c.get('max_actions',6),
        reserve_completion=False,mode='deploy' if teacher else 'collect',seed=seed,cost_unit='measured-native-and-role-wall-ms',
        group_id=f'science-{domain}-round-{round_index}',exploration=c.get('exploration',.1),temperature=c.get('temperature',1.))
    if algorithm=='vats_pdf':
        search_config=vats_config(search_config,c,learned_query_cap=c.get('head_cap_ms',5000))
        search_config=replace(search_config,scorer_snapshot=collection_snapshot,policy_snapshot=collection_snapshot)
    search_budget=BudgetLedger(c.get('budget_ms',500000)).total
    scope=make_scope(domain,'development',tuple(c.get('case_ids',[0,1])),native_utility=algorithm=='vats_pdf');parent=make_candidate(domain)
    executor=ScienceExecutor();initial=[];initial_cost=0
    for task in scope.tasks:
        r=executor.run(parent,task,scope,c.get('execution_cap_ms',120000))
        if r.status!='complete' or not r.outcome.admissible:raise RuntimeError('Initial native science acquisition failed: '+str(r.error))
        initial.append((task,r));initial_cost+=r.actual_cost
    seeds=random.Random(seed+1000000*round_index).sample(range(2**63-1),group_size)
    group=[];raw=[];audits=[];physical=[];edits=[];forced_trajectories=[];pairs=[];forced_acquisition=[];edit_evidence=[];plan_evidence=[]
    teacher_choices=[]
    started=time.monotonic()
    for member,member_seed in enumerate(seeds):
        guard.check()
        executor=ScienceExecutor()
        bridge_options={'snapshot_id':collection_snapshot} if algorithm=='vats_pdf' else {}
        if teacher:
            bridge_options.update(teacher_kwargs)
        bridge=bridge_class(model,cost_unit='wall-ms',batch_mode=c.get('batch_mode','joint'),observation_mode=c.get('observation_mode','full'),observation_donors=c.get('observation_donors'),**bridge_options)
        checker=AuditedScienceChecker(bridge,f'development-{domain}-{scope.uid}')
        from specrsi.science.executor import make_science_ledger
        ledger=make_science_ledger(checker) if algorithm=='vats_pdf' else Ledger(checker)
        for task,r in initial:ledger.admit_direct(parent,task,scope,r.outcome,r.receipt)
        drafter=HFParameterDrafter(bridge,max_new_tokens=c.get('draft_max_tokens',384),top_k=c.get('draft_top_k',32),
            temperature=c.get('temperature',1.),exploration=c.get('exploration',.1),
            check_cap_ms=c.get('check_cap_ms',1000),snapshot=collection_snapshot,proposal_mode=c.get('proposal_mode','numeric_catalog'),catalog_size=c.get('catalog_size',12))
        cfg=replace(search_config,seed=member_seed)
        hooks={'outer_value':bridge.outer_value,'batch_value':bridge.batch_value,'menu_builder':MenuBuilder(max_options=4)}
        if algorithm=='vats_pdf':
            hooks={'scorer':bridge.scorer,'policy_menu':bridge.policy_menu,
                   'capture_paid_states':bool(forced_config.repetitions and member==0)}
        engine=engine_class(scope,parent,ledger,BudgetLedger(search_budget),executor,drafter,cfg,**hooks)
        if algorithm=='vats_pdf':
            bridge.initial_budget,bridge.workers,bridge.contract_id=engine.state.budget.total,engine.state.workers,scope.uid
        checker.state=engine.state
        result=engine.run();raw.append(result.to_dict());audits.extend(checker.audits)
        group.append(bridge.convert(result,lineage_id=checker.lineage_id));physical.extend(bridge.physical_inferences)
        edits.extend(getattr(bridge,'executed_edits',()))
        edit_evidence.extend(getattr(bridge,'edit_evidence',()))
        plan_evidence.extend(getattr(bridge,'plan_evidence',()))
        teacher_choices.extend(getattr(bridge,'teacher_choice_evidence',()))
        physical.extend({'kind':'native_check',**x} for x in checker.physical_checks)
        guard.check()
        if forced_config.repetitions and member==0:
            def suffix_drafter(suffix_bridge,suffix_executor):
                return HFParameterDrafter(suffix_bridge,max_new_tokens=c.get('draft_max_tokens',384),
                    top_k=c.get('draft_top_k',32),temperature=c.get('temperature',1.),exploration=c.get('exploration',.1),
                    check_cap_ms=c.get('check_cap_ms',1000),snapshot=bridge.snapshot_id,
                    proposal_mode=c.get('proposal_mode','numeric_catalog'),catalog_size=c.get('catalog_size',12))
            continuations=collect_paid_continuations(engine.paid_snapshots,model=model,snapshot_id=bridge.snapshot_id,
                bridge_factory=lambda:bridge_class(model,cost_unit='wall-ms',snapshot_id=bridge.snapshot_id),
                executor_factory=ScienceExecutor,drafter_factory=suffix_drafter,checker_factory=ScienceParameterChecker,
                environment_contract='native_science',config=forced_config,seed=seed+1000000*round_index ^ 0x534349,
                lineage_id=checker.lineage_id,training=config.get('training',{}),guard=guard,
                acquisition_path=Path(config.get('output','artifacts/training/science_gvo')) / f'forced_acquisition_round_{round_index}.jsonl')
            forced_trajectories.extend(continuations['forced_trajectories']);pairs.extend(continuations['pairs'])
            forced_acquisition.append(continuations['metadata']);physical.extend(continuations['physical'])
    if not teacher:
        validate_group(group)
    return {'groups':[group],'audits':audits,'edits':edits,'forced_trajectories':forced_trajectories,'pairs':pairs,
        'metadata':{'domain':domain,'split':'development','algorithm':algorithm,
        'author_checkpoint_used':False,'paper_budget_comparable':False,'cost_unit':'measured-wall-ms',
        'cost_accounting':vats_cost_accounting('wall-ms',cfg.policy_call_cost,
            architecture_version=getattr(model.config,'architecture_version',2)) if algorithm=='vats_pdf' else None,
        'initial_acquisition_ms':initial_cost,'initial_acquisition':[{'task':t.uid,'receipt':thaw(r.receipt)} for t,r in initial],
        'independent_initial_outcomes_shared_once':True,'raw_search_results':raw,'physical_inferences':physical,
        'search_cost':sum(t.total_cost for t in group),'forced_search_cost':sum(t.total_cost for t in forced_trajectories),
        'forced_acquisition':forced_acquisition,
        'edit_evidence':edit_evidence,
        **({'sft_plan_evidence':plan_evidence,'demonstration_quality':c.get('demonstration_quality',{}),
            'demonstration_policy':c['demonstration_policy'],'demonstration_stopping':c.get('demonstration_stopping'),
            'teacher_choice_evidence':teacher_choices} if teacher else {}),
        'total_acquisition_ms':initial_cost+sum(t.total_cost for t in group)+sum(x['total_acquisition_cost'] for x in forced_acquisition),
        'collection_wall_seconds':time.monotonic()-started,
        'wall_cap_contract':'cooperative HF/checker slices; measured overruns explicitly fail, never silently clamped'}}
