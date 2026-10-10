"""Connect neural roles to the real SpecRSI evidence/budget/search engine.

The bundled collector uses explicitly synthetic pure-integer contexts but runs
the actual full SpecRSI tree, checker, batch planner, reservations and acceptance.
External experiments reuse NeuralSearchBridge with their own trusted executors.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import random
import time

import torch

from specrsi.core.budget import BudgetLedger
from specrsi.core.checker import ConservativeChecker
from specrsi.core.execution import PureIntegerExecutor
from specrsi.core.ledger import Ledger
from specrsi.core.models import Candidate, EvaluationScope, Task, thaw, digest
from specrsi.core.search import DraftResult, SearchConfig, SearchEngine, ValueEstimate
from specrsi.planning.batches import MenuBuilder
from specrsi.planning.policy import TraceDecision

from .objectives import mixture_probabilities
from .records import CheckerAudit, DecisionRecord, SearchTrajectory, TransitionRecord, validate_group


def _pad(values, size):
    if len(values) > size:
        raise ValueError(f"Neural bridge needs dimension >= {len(values)} (configured {size})")
    return tuple(float(x) for x in values) + (0.,) * (size-len(values))


class NeuralSearchBridge:
    """Hooks cache exact collection inputs for fixed-support likelihood replay.

    Each instance belongs to ONE search. Inference cost is one declared logical
    work unit per scored option. This reference unit is not GPU seconds/tokens.
    Production time/token ledgers must supply their measured ValueEstimate costs.
    """
    def __init__(self, model, *, cost_unit="logical-work", batch_mode='joint', observation_mode='full', observation_donors=None):
        self.model = model
        self.captured = {}
        self.token_decisions = []
        self.physical_inferences = []
        self.score_costs = {}
        from .observations import MODES,FrozenDonor,views
        if observation_mode not in MODES:raise ValueError('Unknown observation intervention')
        self.observation_mode=observation_mode
        self.observation_records=[]
        self.observation_donors={}
        if observation_donors:
            from pathlib import Path
            donors=json.loads(Path(observation_donors).read_text()) if isinstance(observation_donors,(str,Path)) else observation_donors
            for key,value in donors.items():
                self.observation_donors[key]=value if isinstance(value,FrozenDonor) else FrozenDonor.from_view(
                    views(value['features'],value['context_text'],value['options'],value['kind']),donor_id=value['donor_id'])
        if cost_unit not in {"logical-work","tokens","wall-ms"}:
            raise ValueError("Bridge cost unit must be logical-work, tokens, or wall-ms")
        if cost_unit == "tokens" and model.backend != "hf":
            raise ValueError("Token cost metering requires an HF tokenizer")
        self.cost_unit = cost_unit
        if batch_mode not in {'joint','additive'}:
            raise ValueError('Batch value mode must be joint or additive')
        self.batch_mode = batch_mode
        if batch_mode=='additive' and observation_mode=='mismatched_state':
            raise ValueError('Additive mismatched-state control requires separately frozen per-item donors')
        self.feature_dim = getattr(model.config, "feature_dim", 8)
        self.option_dim = model.config.option_dim
        if self.feature_dim < 8 or self.option_dim < 8:
            raise ValueError("Core bridge needs feature_dim and option_dim >= 8")

    def state_features(self, state):
        n = max(1, len(state.scope.tasks))
        return _pad((*state.features(), state.version, len(state.ledger.records)/n, len(state.history)), self.feature_dim)

    def context_text(self, state, *, bounded=True):
        # This bounded-by-search-limits structural input contains observed scores
        # only. Expected task answers are never included in the role context.
        text = json.dumps({"scope": state.scope.name, "budget": {"available": state.budget.available,
            "pending": state.budget.pending, "total": state.budget.total},
            "tasks": [{"id": t.uid[:12], "inputs": thaw(t.inputs)} for t in state.scope.tasks],
            "candidates": [{"id": c.uid[:12], "artifacts": thaw(c.artifacts), "runtime": thaw(c.runtime),
                "missing": [t.uid[:12] for t in state.ledger.missing(c,state.scope)],
                "complete_score": state.ledger.score(c,state.scope) if state.ledger.complete(c,state.scope) else None}
                for c in state.candidates.values()],
            # An explicit observed-history summary, NOT a sufficient-statistic
            # assumption. Full history remains in the trusted state and raw trace.
            "recent_history": [{k:e[k] for k in ("event","candidate_id","task_id","status","route","cost") if k in e}
                for e in state.history[-8:]]}, separators=(",", ":"), sort_keys=True)
        return self.bounded_text(text, self.model.config.max_length // 3) if bounded and self.model.backend == "hf" else text

    def bounded_text(self, text, limit=None):
        """Choose a fixed bounded observation BEFORE collection, then save it.

        This explicit prefix summary is never regenerated during likelihood
        replay; full untruncated observations stay in the raw search trace.
        """
        if self.model.backend != "hf":
            return text
        limit = min(limit or self.model.config.max_length-16,self.model.config.max_length-16)
        ids = self.model.tokenizer.encode(text,add_special_tokens=False)
        return self.model.tokenizer.decode(ids[:limit],skip_special_tokens=False) if len(ids)>limit else text

    def inference_cost(self, text):
        # Discriminative heads process input tokens but generate no output token.
        return len(self.model.tokenizer.encode(text)) if self.cost_unit == "tokens" else 1

    @torch.no_grad()
    def _score(self, state, kind, option_id, option_features, item_features=None):
        from dataclasses import asdict
        from .observations import views
        features = self.state_features(state)
        raw_context=self.context_text(state,bounded=False)
        key=f'{state.version}:{kind}:{option_id}'
        mode=self.observation_mode if kind in {'outer','batch'} else 'full'
        donor=self.observation_donors.get(key,self.observation_donors.get(f'{state.version}:{kind}',self.observation_donors.get(kind)))
        observed=views(features,raw_context,(option_features,),kind,mode=mode,donor=donor)
        self.observation_records.append({'key':key,**asdict(observed)})
        features=observed.features;option_features=observed.options[0]
        text=self.bounded_text(observed.context_text,self.model.config.max_length//3) if self.model.backend=='hf' else observed.context_text
        if item_features is not None:
            item_features=tuple(views(self.state_features(state),raw_context,(f,),kind,mode=mode,donor=donor).options[0]
                                for f in item_features)
        d = DecisionRecord(kind, (option_id,), 0, 0., state_features=features,
            option_features=(option_features,), transition_index=state.version, context_text=text,
            option_item_features=(tuple(item_features),) if item_features is not None else ())
        started=time.monotonic()
        value = float(self.model.decision_logits(d)[0])
        elapsed=time.monotonic()-started
        self.score_costs[(state.version,kind,option_id)]=max(1,math.ceil(elapsed*1000)) if self.cost_unit=='wall-ms' else self.inference_cost(text)
        self.physical_inferences.append({'kind':'learned_control','role':kind,'elapsed_seconds':elapsed,
            'gpu_count':1 if str(getattr(self.model,'device','cpu')).startswith('cuda') else 0,
            'prompt_tokens':len(self.model.tokenizer.encode(text)) if self.model.backend=='hf' else 0,'completion_tokens':0})
        self.captured[(state.version,kind,option_id)] = d
        return value

    def outer_value(self, state, action):
        started=time.monotonic()
        candidate = state.candidates[action.candidate_id]
        missing = state.ledger.missing(candidate,state.scope)
        sources = sum(any(state.ledger.get(c,t,state.scope) for c in state.candidates.values() if c.uid != candidate.uid)
                      for t in missing)
        options = _pad((action.kind == "verify", action.work_cap / max(1,state.budget.total),
            len(missing)/len(state.scope.tasks), state.ledger.score(candidate,state.scope) if not missing else 0.,
            sources/len(state.scope.tasks), state.children_drafted.get(candidate.uid,0), candidate.operating_penalty,
            sum(state.forecasts.get(candidate.uid,{}).values())/len(state.scope.tasks)),self.option_dim)
        value = self._score(state,"outer",action.uid,options)
        if self.cost_unit=='wall-ms':
            elapsed=time.monotonic()-started
            self.score_costs[(state.version,'outer',action.uid)]=max(1,math.ceil(elapsed*1000))
            self.physical_inferences[-1]['controller_wall_seconds']=elapsed
        return ValueEstimate(value,self.score_costs[(state.version,"outer",action.uid)])

    def batch_value(self, state, candidate, batch, problem):
        started=time.monotonic()
        selected = [problem.by_id[x] for x in batch.option_ids]
        union = {t for option in selected for t in option.task_ids}
        missing = {t.uid for t in state.ledger.missing(candidate,state.scope)}
        options = _pad((len(union & missing)/len(state.scope.tasks),len(missing-union)/len(state.scope.tasks),
            missing.issubset(union),problem.cost(batch.option_ids)/max(1,state.budget.total),
            len({o.group for o in selected}),sum(len(o.source_ids) for o in selected)/len(state.scope.tasks),
            len(problem.dependencies),len(problem.conflicts)),self.option_dim)
        item_features=None
        if self.batch_mode=='additive':
            # Each item prediction sees only its own coverage/setup/cap facts;
            # the score of a batch is the sum, with no union/completion bonus.
            item_features=tuple(_pad((len(set(o.task_ids)&missing)/len(state.scope.tasks),
                0.,0.,o.cap/max(1,state.budget.total),1.,len(o.source_ids)/len(state.scope.tasks),0.,0.),self.option_dim)
                for o in selected)
        value = self._score(state,"batch",batch.uid,options,item_features)
        if self.cost_unit=='wall-ms':
            elapsed=time.monotonic()-started
            self.score_costs[(state.version,'batch',batch.uid)]=max(1,math.ceil(elapsed*1000))
            self.physical_inferences[-1]['controller_wall_seconds']=elapsed
        return ValueEstimate(value,self.score_costs[(state.version,"batch",batch.uid)])

    def convert(self, result, *, lineage_id):
        decisions = []
        for raw in result.decisions:
            snapshots = [self.captured.get((raw.state_version,raw.kind,option)) for option in raw.support]
            if any(x is None for x in snapshots):
                raise ValueError("Missing collection input snapshot; cannot replay a guessed policy context")
            if any((x.state_features,x.context_text) != (snapshots[0].state_features,snapshots[0].context_text) for x in snapshots):
                raise ValueError("Options were evaluated with inconsistent state representations")
            selected = raw.support.index(raw.selected_id)
            decisions.append(DecisionRecord(raw.kind,raw.support,selected,
                math.log(raw.probabilities[selected]) if raw.stochastic else None,
                state_features=snapshots[0].state_features,option_features=tuple(x.option_features[0] for x in snapshots),
                transition_index=raw.transition_index,temperature=raw.temperature,exploration=raw.exploration,
                proposal_snapshot=raw.proposal_snapshot,context_text=snapshots[0].context_text,
                behavior_probabilities=raw.probabilities,deterministic=not raw.stochastic,
                option_item_features=tuple(x.option_item_features[0] for x in snapshots)
                    if snapshots[0].option_item_features else (),
                # All roles here use the enclosing pre-selection/pre-planning
                # origin. Planning/ranking are already in transition.cost.
                prior_cost=0.,excluded_prefix_cost=0.))
        decisions.extend(self.token_decisions)
        decisions.sort(key=lambda d:d.transition_index)
        raw_ledger=json.loads(result.ledger_json)
        records={(r['candidate_id'],r['task_id']):r for r in raw_ledger['records']}
        raw_scope=raw_ledger['scopes'][0]
        weights={digest(task):weight for task,weight in zip(raw_scope['tasks'],raw_scope['weights'])}
        seen=set(); transitions=[]
        for transition in result.transitions:
            item_reward=0.
            for event in transition.events:
                if event.get('event')!='acquisition' or event.get('status')!='complete' or event.get('from_cache'):
                    continue
                key=(event['candidate_id'],event['task_id'])
                if key in seen:continue
                seen.add(key)
                outcome=records[key]['outcome']
                if outcome['admissible']:item_reward+=outcome['score']*weights[event['task_id']]
            transitions.append(TransitionRecord(transition.f_before,transition.f_after,transition.cost,item_reward))
        trajectory = SearchTrajectory(result.group_id,result.initial_context_id,result.search_id,result.seed,
            tuple(transitions),tuple(decisions),
            result.initial_f,result.final_f,lineage_id=lineage_id)
        if not math.isclose(trajectory.total_cost,result.total_cost):
            raise ValueError("Engine and learning trajectory cost ledgers disagree")
        return trajectory


class HFPromptDrafter:
    """Generate real language edits with draft LoRA and retain every token actor.

    Designed for opaque prompt agents whose executor reads ``system_prompt``.
    The conservative checker cannot authorize opaque model responses; all actual
    outcomes still require direct execution. Surface predictions stay advisory.
    """
    def __init__(self,bridge,*,max_new_tokens=128,top_k=32,temperature=1.,exploration=.1,
                 proposal_snapshot="hf-draft-initial",surface_reserve_per_item=128,candidate_validator=None):
        if bridge.model.backend != "hf" or bridge.cost_unit != "tokens":
            raise ValueError("HFPromptDrafter needs an HF NeuralSearchBridge with cost_unit='tokens'")
        self.bridge = bridge
        self.max_new_tokens,self.top_k = max_new_tokens,top_k
        self.temperature,self.exploration = temperature,exploration
        self.proposal_snapshot = proposal_snapshot
        self.surface_reserve_per_item = 0 if getattr(bridge, 'direct_only', False) else surface_reserve_per_item
        self.candidate_validator = candidate_validator

    @torch.no_grad()
    def draft(self,parent,state,cap,rng):
        from .sampler import sample_edit_tokens
        model = self.bridge.model
        workflow_mode = 'workflow' in parent.artifacts
        # Bound the *templated* prompt by both the model context and this paid
        # draft allowance. A large model context must not generate an over-cap
        # prompt when the actual work allowance is smaller.
        from .frozen_text import bounded_semantic_text
        parent_text = (json.dumps(thaw(parent.artifacts['workflow'])) if workflow_mode else
                       str(parent.artifacts.get("system_prompt", "")))
        from specrsi.output_contracts import draft_interface_guidance
        interface_guidance = draft_interface_guidance(parent.runtime.get("generation", {}))
        if interface_guidance and workflow_mode:
            raise ValueError("output_contract currently requires single-prompt execution")

        def template(text):
            return model.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)

        def prepare_input(reserved_output):
            input_cap = min(model.config.max_length, cap) - reserved_output
            if input_cap < 1:
                return None
            try:
                bounded_parent = bounded_semantic_text(model.tokenizer, parent_text, max(1, input_cap // 3))[0]
            except ValueError as error:
                if "Semantic token cap cannot fit" not in str(error):
                    raise
                return None
            prefix = ("Write an improved system instruction for the task agent. Return only the new instruction, "
                "without commentary. Make a concrete change using the observed feedback; do not copy the current instruction. "
                "Preserve the required output format and interface constraints in the current instruction. "
                "The checker and evaluation rules cannot be edited.\n" + interface_guidance + "Current instruction:\n"+
                bounded_parent+"\nObserved search context:\n")
            if workflow_mode:
                prefix = ('Improve the task-solving workflow. Return only JSON with a workflow object containing nodes and output_node. '
                          'Each node has id, instruction, dependencies. Use 1 to 8 acyclic connected nodes. '
                          'Keep the required final answer format. Current workflow: '+bounded_parent+
                          '\nObserved search context:\n')
            # A short parent releases its unused space to feedback. Check the
            # final chat template exactly; never cut a serialized feedback record.
            feedback_cap = input_cap - len(template(prefix)) - 4
            while feedback_cap > 0:
                try:
                    context = (self.bridge.drafter_context_text(state, parent, token_limit=feedback_cap)
                        if hasattr(self.bridge, "drafter_context_text") else
                        self.bridge.bounded_text(self.bridge.context_text(state), feedback_cap))
                except ValueError as error:
                    if "Draft feedback token budget" not in str(error):
                        raise
                    return None
                trial = template(prefix + context)
                if len(trial) <= input_cap:
                    return trial
                feedback_cap -= len(trial) - input_cap + 1
            return None

        # max_new_tokens is a ceiling. Keep its full allowance when affordable,
        # otherwise retry a smaller completion instead of skipping valid work.
        reserved_output = self.max_new_tokens
        input_ids = None
        while reserved_output >= 1:
            input_ids = prepare_input(reserved_output)
            if input_ids is not None or reserved_output == 1:
                break
            reserved_output = max(1, reserved_output // 2)
        if input_ids is None:
            return DraftResult(None, 0, status="interrupted", error="Draft input/output tokens do not fit enforced cap")
        output_room = min(self.max_new_tokens,model.config.max_length-len(input_ids),cap-len(input_ids))
        if output_room <= 0:
            return DraftResult(None,0,status="interrupted",error="Draft input/output tokens do not fit enforced cap")
        # Keep a declared portion available for Surface calls when affordable.
        reserved = min(self.surface_reserve_per_item*len(state.scope.tasks),max(0,cap-len(input_ids)-1))
        output_room = max(1,min(output_room,cap-len(input_ids)-reserved))
        started=time.monotonic()
        sample = sample_edit_tokens(model,input_ids,max_new_tokens=output_room,rng=rng,top_k=self.top_k,
            temperature=self.temperature,exploration=self.exploration,proposal_snapshot=self.proposal_snapshot,
            transition_index=getattr(state, "current_transition_index", state.version))
        self.bridge.physical_inferences.append({'kind':'learned_control','role':'draft','elapsed_seconds':time.monotonic()-started,
            'gpu_count':1 if str(model.device).startswith('cuda') else 0,'prompt_tokens':len(input_ids),'completion_tokens':sample.token_cost-len(input_ids)})
        self.bridge.token_decisions.extend(sample.decisions)
        cost = sample.token_cost
        instruction = sample.text.strip()
        if not instruction:
            return DraftResult(None,cost,status="error",error="Draft produced no executable instruction text")
        artifacts = thaw(parent.artifacts)
        if workflow_mode:
            from specrsi.workflows import Workflow
            try:
                raw=instruction
                if raw.startswith('```'): raw='\n'.join(raw.splitlines()[1:-1])
                workflow=json.loads(raw)['workflow']; Workflow.from_dict(workflow)
                artifacts['workflow']=workflow
            except (ValueError,KeyError,TypeError) as exc:
                return DraftResult(None,cost,status='error',error='Invalid workflow edit: '+str(exc))
        else:
            artifacts["system_prompt"] = instruction
        if artifacts == thaw(parent.artifacts):
            # Naming an identical artifact a child creates paid duplicate work.
            # Keep the sampled tokens and generation charge as a failed edit.
            return DraftResult(None, cost, status="error", error="Draft did not change the parent artifacts")
        child = Candidate(f"hf-draft-{state.version}",artifacts,parent.runtime,"opaque",False,
                          parent.operating_penalty,parent.uid)
        if self.candidate_validator is not None:
            try:
                self.candidate_validator(child)
            except ValueError as exc:
                return DraftResult(None,cost,status="error",error=f"Child does not fit complete execution contract: {exc}")
        if hasattr(self.bridge, "record_executed_edit"):
            self.bridge.record_executed_edit(child, parent, sample)
        forecasts = {}
        if getattr(self.bridge, 'direct_only', False):
            return DraftResult(child,cost,forecasts=forecasts)
        for task in state.scope.tasks:
            text = self.bridge.bounded_text("Forecast authorization under checker="+state.ledger.checker.version+
                "; source_policy=all-U; allowed_source_ids=[]; check_cap=0; semantic_kind=opaque. "
                "No checker attempt is performed in this context. New instruction: "+instruction+
                "\nTask inputs: "+json.dumps(thaw(task.inputs),separators=(",",":")))
            expense = self.bridge.inference_cost(text)
            if cost+expense>cap:
                continue
            started=time.monotonic()
            forecasts[task.uid] = float(model.surface_logits(self.bridge.state_features(state),text).sigmoid())
            self.bridge.physical_inferences.append({'kind':'learned_control','role':'surface','elapsed_seconds':time.monotonic()-started,
                'gpu_count':1 if str(model.device).startswith('cuda') else 0,'prompt_tokens':expense,'completion_tokens':0})
            cost += expense
        return DraftResult(child,cost,forecasts=forecasts)


class LearnedIntegerDrafter:
    """Finite executable edit policy for the supported integer reference domain."""
    def __init__(self, bridge):
        self.bridge = bridge

    @torch.no_grad()
    def draft(self, parent, state, cap, rng):
        cost = 4 + len(state.scope.tasks)  # Four option-head calls + one surface prediction per item.
        if cap < cost:
            return DraftResult(None,cap,status="interrupted",error="Insufficient cap for declared role work")
        choices = ("preserve", "repair-a", "repair-b", "repair-both")
        support = tuple(f"edit:{parent.uid}:{x}" for x in choices)
        options = [_pad((i == 0,i == 1,i == 2,i == 3,0,0,0,0),self.bridge.option_dim) for i in range(4)]
        values = [self.bridge._score(state,"edit",sid,features) for sid,features in zip(support,options)]
        p = mixture_probabilities(torch.tensor(values), exploration=.1,temperature=1.).tolist()
        index = rng.choices(range(4),weights=p,k=1)[0]
        routes = thaw(parent.artifacts["routes"])
        if index in (1,3):
            routes["a"] = ["add",["x"],["const",1]]
        if index in (2,3):
            routes["b"] = ["mul",["x"],["const",2]]
        child = Candidate(f"{parent.name}-{choices[index]}-{state.version}",{"routes":routes},
            parent.runtime,"pure_integer",True,parent_id=parent.uid)
        forecasts = {}
        features = self.bridge.state_features(state)
        for task in state.scope.tasks:
            text = (self.bridge.context_text(state) + "\nForecast completed checker authorization: " + json.dumps(
                {"child":child.manifest(),"task":thaw(task.inputs)},separators=(",",":"))) if self.bridge.model.backend == "hf" else ""
            forecasts[task.uid] = float(self.bridge.model.surface_logits(features,text).sigmoid())
        trace = TraceDecision("edit",support,support[index],tuple(p),tuple(values),state.version,state.version,1.,.1,
            "fixed-integer-edit-catalog-v1",state_features=features,option_features=tuple(options),value_source="learned-draft-head")
        return DraftResult(child,cost,forecasts=forecasts,edit_decision=trace)


class AuditedReferenceChecker(ConservativeChecker):
    """Record every actual bounded attempt prospectively; interruptions stay unlabeled."""
    def __init__(self, bridge, lineage_id):
        super().__init__()
        self.bridge,self.lineage_id = bridge,lineage_id
        self.state = None
        self.audits = []

    def check(self,source,child,task,scope,cap):
        features = self.bridge.state_features(self.state)
        text = (self.bridge.context_text(self.state)+"\nCheck "+json.dumps({"source":source.uid,
            "child":child.uid,"task":task.uid,"check_cap":cap},separators=(",",":"))) if self.bridge.model.backend == "hf" else ""
        result = super().check(source,child,task,scope,cap)
        protocol = f"{self.version}:allowed-paid-sources-parent-first:cap={cap}"
        self.audits.append(CheckerAudit(self.lineage_id,protocol,result.status != "interrupted",
            int(result.status == "P") if result.status != "interrupted" else None,features,text))
        return result


def collect_reference_specrsi(model, round_index, config):
    """CLI collector: actual isolated SpecRSI searches on disclosed reference tasks."""
    model.eval()
    settings = config.get("collection",{})
    group_size = settings.get("group_size",4)
    if group_size < 2:
        raise ValueError("GVO needs group_size >= 2")
    rng = random.Random(config.get("training",{}).get("seed",0)+round_index*1000000)
    groups,audits,raw_results = [],[],[]
    initialization_cost = 0
    for context_index in range(settings.get("contexts",2)):
        offset = context_index*5
        tasks = tuple(Task(f"{route}-{x}",{"route":route,"x":x},x+1 if route == "a" else x*2)
                      for route in ("a","b") for x in (offset+1,offset+2))
        scope = EvaluationScope(f"reference-train-{context_index}",tasks,"pure-int-v1","integer-equality-v1",split="train")
        incumbent = Candidate("initial",{"routes":{"a":["x"],"b":["x"]}},
            {"contract":"pure-int-v1"},"pure_integer",True)
        lineage_id = f"reference-train-lineage-{context_index}"
        group = []
        for member in range(group_size):
            # Fresh checker, ledger, budget, history, RNG and hook snapshots. Only
            # allowed initial immutable candidate/outcomes are shared in meaning.
            bridge = NeuralSearchBridge(model,batch_mode=settings.get("batch_mode","joint"),observation_mode=settings.get("observation_mode","full"),observation_donors=settings.get("observation_donors"))
            checker = AuditedReferenceChecker(bridge,lineage_id)
            ledger = Ledger(checker)
            executor = PureIntegerExecutor()
            for task in tasks:
                execution = executor.run(incumbent,task,scope,5)
                initialization_cost += execution.actual_cost
                ledger.admit_direct(incumbent,task,scope,execution.outcome,execution.receipt)
            search = SearchConfig(execution_cap=5,draft_cap=4+len(tasks),max_candidates=settings.get("max_candidates",6),
                max_actions=settings.get("max_actions",12),max_children=2,mode="collect",seed=rng.randrange(2**63-1),
                reserve_completion=settings.get("reserve_completion",False),group_id=f"round-{round_index}-context-{context_index}")
            engine = SearchEngine(scope,incumbent,ledger,BudgetLedger(settings.get("budget",150)),executor,
                LearnedIntegerDrafter(bridge),search,outer_value=bridge.outer_value,batch_value=bridge.batch_value,
                menu_builder=MenuBuilder(max_options=4))
            checker.state = engine.state
            result = engine.run()
            group.append(bridge.convert(result,lineage_id=lineage_id))
            audits.extend(checker.audits)
            raw_results.append(result.to_dict())
        validate_group(group)
        groups.append(group)
    return {"groups":groups,"audits":audits,"metadata":{"reference_tasks_only":True,"cost_unit":"logical-work",
        "initial_record_execution_cost":initialization_cost,"search_cost":sum(t.total_cost for g in groups for t in g),
        "raw_search_results":raw_results}}


def collect_prompt_specrsi(model, round_index, config):
    """Real local-Qwen prompt searches on the dataset's DEVELOPMENT split only.

    Draft/Surface/Q/V run through the local HF LoRA model. Task execution uses
    the separately deployed fixed local model service, with native task scoring.
    """
    from pathlib import Path
    from transformers import AutoTokenizer
    from specrsi.benchmarks import load_examples
    from specrsi.models import LocalModelClient
    from specrsi.experiments import QwenExecutor
    if model.backend != "hf":
        raise ValueError("Prompt search collector requires the actual HF role model")
    settings = config["collection"]
    from .continuations import (ForcedCollectionConfig, FrozenRoleGuard, collect_paid_continuations,
                                require_local_prompt_forks, role_snapshot_id)
    teacher = config.get("stage") == "warmup" and settings.get("demonstration_policy") is not None
    from .demonstrations import demonstration_bridge_kwargs
    teacher_kwargs = demonstration_bridge_kwargs(settings) if teacher else {}
    forced_config = ForcedCollectionConfig.from_settings(settings, teacher=teacher)
    guard = FrozenRoleGuard(model)
    model.eval()
    from specrsi.paper_profile import algorithm_name, vats_config, vats_cost_accounting
    algorithm = algorithm_name(settings)
    collection_snapshot = role_snapshot_id(model, f"hf-frozen-round-{round_index}") if algorithm == "vats_pdf" else f"hf-frozen-round-{round_index}"
    bridge_class, engine_class = NeuralSearchBridge, SearchEngine
    if algorithm == "vats_pdf":
        from specrsi.core.vats import VATSEngine
        from .paper_bridge import VATSSearchBridge
        bridge_class, engine_class = VATSSearchBridge, VATSEngine
        if teacher:
            from .demonstrations import DemonstrationQuality, DemonstrationVATSBridge
            DemonstrationQuality.from_settings(settings.get("demonstration_quality"))
            bridge_class = DemonstrationVATSBridge
    elif teacher or forced_config.repetitions:
        raise ValueError("Demonstrations and paid forced continuations require the PDF VATS collector")
    dataset = settings["dataset"]
    swe_mode=dataset in {'swe_verified','swe_lite'}
    if swe_mode:
        from specrsi.baselines.native_backend import native_splits
        examples=list(native_splits({'dataset':dataset,'data_root':settings.get('data_root','data'),
            'coding_registry':settings.get('coding_registry','configs/coding/swe_tracks.json'),
            **({'development_ids':settings['development_ids']} if 'development_ids' in settings else {}),
            'selection_track':'verified','audit_track':'audit_disjoint_verified'})['development'])
    else:
        examples = load_examples(dataset,"development",data_root=settings.get("data_root","data"))
    executor_config = settings["executor"]
    lock = json.loads(Path(executor_config.get("lock","configs/models/model.lock.json")).read_text())
    client = LocalModelClient(executor_config["base_url"],executor_config.get('served_model_name',lock.get('served_model_name','local-model')))
    health = client.health()
    if not health.get("server_enforces_total_token_cap") or health.get("protocol_version",0)<2:
        raise ValueError("Execution server must enforce total_token_cap with protocol v2")
    if health["revision"] != lock["revision"]:
        raise ValueError("Fixed execution model revision differs from locked checkpoint")
    execution_tokenizer = AutoTokenizer.from_pretrained(lock["local_path"],local_files_only=True,trust_remote_code=lock.get('trust_remote_code',False))
    executor_class=QwenExecutor
    if swe_mode:
        from specrsi.coding_agent import SWERepositoryExecutor
        executor_class=SWERepositoryExecutor
    elif settings.get('task_pipeline'):
        from specrsi.task_pipelines import TaskPipelineExecutor
        executor_class=TaskPipelineExecutor
    if forced_config.repetitions:
        require_local_prompt_forks(executor_class, settings, health)
    contexts,items = settings.get("contexts",1),settings.get("tasks_per_context",1)
    group_size = settings.get("group_size",2)
    if group_size<2 or contexts<1 or items<1 or len(examples)<contexts*items:
        raise ValueError("Need >=2 group members and enough disjoint development contexts")
    master_seed = config.get("training",{}).get("seed",0)
    sampling = settings.get("context_sampling", "fixed")
    if sampling not in {"fixed", "shuffled"}:
        raise ValueError("context_sampling must be fixed or shuffled")
    if sampling == "shuffled":
        examples = list(examples)
        random.Random(master_seed + round_index * 1000000).shuffle(examples)
    seeds = random.Random(master_seed+round_index*1000000).sample(range(2**63-1),contexts*group_size)
    groups,raw_results,physical,edits,forced_trajectories,pairs,forced_acquisition,edit_evidence,plan_evidence = [],[],[],[],[],[],[],[],[]
    teacher_choices = []
    actual_contexts = []
    initial_cost = 0
    def progress(status,extra=None):
        directory = Path(config.get("output","artifacts/training/roles"))
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f"collection_progress_round_{round_index}.json").write_text(json.dumps({
            "status":status,"valid_training_batch":status=="complete","initial_acquisition_tokens":initial_cost,
            "completed_searches":len(raw_results),"physical_executions":physical,
            "development_contexts":actual_contexts,**(extra or {})},indent=2))
    for context_index in range(contexts):
        batch = examples[context_index*items:(context_index+1)*items]
        actual_contexts.append({"context_index": context_index,
            "examples": [{"example_id": e.id, "content_hash": e.content_hash} for e in batch]})
        tasks = tuple(Task(e.id,{"prompt":e.prompt},{'_specrsi_example':e.to_dict()} if dataset=='planbench' else e.reference,
            {"dataset":dataset,"content_hash":e.content_hash,"split":"development","metric_variant":settings.get("metric_variant")}) for e in batch)
        scope = EvaluationScope(f"gvo-development-{dataset}-{context_index}",tasks,"swebench4.1.0-repository-actor-v1" if swe_mode else "qwen-capped-generation-v2",
            dataset+"-"+settings.get("metric_variant","registered-score")+"-v1",scored_fields=("answer","metrics","finish_reason"),
            replay={"generation":executor_config,"model_revision":lock["revision"]},split="train")
        artifacts={'workflow':settings['initial_workflow']} if 'initial_workflow' in settings else {'system_prompt':settings['initial_prompt']}
        incumbent = Candidate("initial",artifacts,
            {"model_revision":lock["revision"],"seed":master_seed+context_index,"generation":executor_config},
            semantic_kind="opaque",deterministic=False)
        # Acquire the allowed initial records once; independent member ledgers
        # start with precisely these same immutable outcomes/receipts.
        initial_executor = executor_class(client,execution_tokenizer,executor_config)
        initial_executor.validate_prompt_candidate(incumbent,tasks,settings.get("execution_cap",2048))
        initial = []
        for task in tasks:
            previous = len(initial_executor.physical_records)
            try:
                outcome = initial_executor.run(incumbent,task,scope,settings.get("execution_cap",2048))
            except Exception as exc:
                physical.extend(initial_executor.physical_records[previous:])
                progress("failed",{"stage":"initial_acquisition","error":str(exc)})
                raise
            physical.extend(initial_executor.physical_records[previous:])
            initial_cost += outcome.actual_cost
            progress("collecting")
            if outcome.status != "complete":
                raise RuntimeError("Initial context did not produce complete outcomes: "+str(outcome.error))
            initial.append((task,outcome))
        group = []
        for member in range(group_size):
            guard.check()
            ledger = Ledger()
            for task,outcome in initial:
                ledger.admit_direct(incumbent,task,scope,outcome.outcome,outcome.receipt)
            bridge_options = {"snapshot_id": collection_snapshot} if algorithm == "vats_pdf" else {}
            if teacher:
                bridge_options.update(teacher_kwargs)
            bridge = bridge_class(model,cost_unit="tokens",batch_mode=settings.get("batch_mode","joint"),observation_mode=settings.get("observation_mode","full"),observation_donors=settings.get("observation_donors"),**bridge_options)
            executor = executor_class(client,execution_tokenizer,executor_config)
            drafter = HFPromptDrafter(bridge,max_new_tokens=settings.get("draft_max_tokens",128),
                top_k=settings.get("draft_top_k",32),proposal_snapshot=collection_snapshot,
                temperature=settings.get("temperature",1.),exploration=settings.get("exploration",.1),
                candidate_validator=lambda c:executor.validate_prompt_candidate(c,tasks,settings.get("execution_cap",2048)))
            cfg = SearchConfig(execution_cap=settings.get("execution_cap",2048),
                draft_cap=settings.get("draft_cap",4096),max_candidates=settings.get("max_candidates",2),
                max_children=settings.get("max_children",1),max_actions=settings.get("max_actions",3),
                mode="deploy" if teacher else "collect",seed=seeds[context_index*group_size+member],all_u=True,max_sources=0,check_cap=0,
                planning_cost=0,outer_value_cap=model.config.max_length,batch_value_cap=model.config.max_length,
                reserve_completion=not ('workflow' in artifacts or settings.get('task_pipeline')),cost_unit="tokens",group_id=f"prompt-round-{round_index}-context-{context_index}",
                exploration=settings.get("exploration",.1),temperature=settings.get("temperature",1.))
            hooks = {'outer_value':bridge.outer_value,'batch_value':bridge.batch_value,'menu_builder':MenuBuilder(max_options=2)}
            if algorithm == "vats_pdf":
                from .frozen_text import semantic_input_cap
                cfg = vats_config(cfg, settings, learned_query_cap=semantic_input_cap(model.config))
                cfg = replace(cfg, scorer_snapshot=bridge.snapshot_id, policy_snapshot=bridge.snapshot_id)
                hooks = {'scorer':bridge.scorer,'policy_menu':bridge.policy_menu,
                         'capture_paid_states':bool(forced_config.repetitions and member == 0)}
            engine = engine_class(scope,incumbent,ledger,BudgetLedger(settings.get("budget_tokens",50000)),executor,
                drafter,cfg,**hooks)
            if algorithm == "vats_pdf":
                bridge.initial_budget, bridge.workers, bridge.contract_id = engine.state.budget.total, engine.state.workers, scope.uid
            try:
                result = engine.run()
            except Exception as exc:
                physical.extend(executor.physical_records)
                physical.extend(bridge.physical_inferences)
                progress("failed",{"stage":"search","error":str(exc),"failed_search_settled_tokens":engine.state.budget.spent,
                    "failed_search_pending_tokens":engine.state.budget.pending,
                    "failed_search_budget": {"total": engine.state.budget.total,
                        "charges": [asdict(charge) for charge in engine.state.budget.charges],
                        "pending_reservations": engine.state.budget.reserved,
                        "cost_valid": engine.state.budget.cost_valid,
                        "breached": engine.state.budget.breached},
                    "failed_search_physical_accounting": {
                        "executor_receipts": len(executor.physical_records),
                        "role_receipts": len(bridge.physical_inferences),
                        "complete": False,
                        "unmeasured_usage": "Unknown work remains represented by pending reservations or conservative-bound charges; known receipts are retained once, without fabricating actual usage."}})
                raise
            group.append(bridge.convert(result,lineage_id=f"development-{dataset}-{scope.uid}"))
            edits.extend(getattr(bridge, "executed_edits", ()))
            edit_evidence.extend(getattr(bridge, "edit_evidence", ()))
            plan_evidence.extend(getattr(bridge, "plan_evidence", ()))
            teacher_choices.extend(getattr(bridge, "teacher_choice_evidence", ()))
            raw_results.append(result.to_dict())
            physical.extend(executor.physical_records)
            physical.extend(bridge.physical_inferences)
            guard.check()
            if forced_config.repetitions and member == 0:
                def suffix_executor():
                    suffix_client = LocalModelClient(executor_config["base_url"],
                        executor_config.get('served_model_name',lock.get('served_model_name','local-model')))
                    return executor_class(suffix_client, execution_tokenizer, executor_config)

                def suffix_drafter(suffix_bridge, suffix_executor):
                    return HFPromptDrafter(suffix_bridge,max_new_tokens=settings.get("draft_max_tokens",128),
                        top_k=settings.get("draft_top_k",32),proposal_snapshot=bridge.snapshot_id,
                        temperature=settings.get("temperature",1.),exploration=settings.get("exploration",.1),
                        candidate_validator=lambda candidate:suffix_executor.validate_prompt_candidate(
                            candidate,tasks,settings.get("execution_cap",2048)))

                try:
                    continuations = collect_paid_continuations(engine.paid_snapshots, model=model,
                        snapshot_id=bridge.snapshot_id,
                        bridge_factory=lambda:bridge_class(model,cost_unit="tokens",snapshot_id=bridge.snapshot_id),
                        executor_factory=suffix_executor,drafter_factory=suffix_drafter,
                        checker_factory=ConservativeChecker,environment_contract="local_qwen_prompt",config=forced_config,
                        seed=(master_seed + round_index * 1000000 + context_index * 10007) ^ 0x47564F,
                        lineage_id=f"development-{dataset}-{scope.uid}",training=config.get("training",{}),guard=guard,
                        acquisition_path=Path(config.get("output","artifacts/training/roles")) / f"forced_acquisition_round_{round_index}.jsonl")
                except Exception as exc:
                    progress("failed", {"stage":"forced_continuations","error":str(exc),
                        "partial_acquisition_journal":f"forced_acquisition_round_{round_index}.jsonl"})
                    raise
                forced_trajectories.extend(continuations["forced_trajectories"])
                pairs.extend(continuations["pairs"])
                physical.extend(continuations["physical"])
                forced_acquisition.append(continuations["metadata"])
            progress("collecting")
        if not teacher:
            validate_group(group)
        groups.append(group)
    progress("complete")
    return {"groups":groups,"audits":[],"edits":edits,"forced_trajectories":forced_trajectories,"pairs":pairs,
        "metadata":{"reference_tasks_only":False,"dataset":dataset,"algorithm":algorithm,
        "development_contexts":actual_contexts,
        "split":"development","cost_unit":"tokens","author_checkpoint_used":False,"paper_budget_comparable":False,
        "cost_accounting": vats_cost_accounting("tokens", cfg.policy_call_cost,
            architecture_version=getattr(model.config, "architecture_version", 2)) if algorithm == "vats_pdf" else None,
        "initial_acquisition_tokens":initial_cost,"search_cost":sum(t.total_cost for g in groups for t in g),
        "forced_search_cost":sum(t.total_cost for t in forced_trajectories),"forced_acquisition":forced_acquisition,
        "total_acquisition_tokens":initial_cost+sum(t.total_cost for g in groups for t in g)+sum(t.total_cost for t in forced_trajectories),
        "execution_model":health,"physical_executions":physical,"raw_search_results":raw_results,
        "edit_evidence":edit_evidence,
        **({"sft_plan_evidence":plan_evidence,"demonstration_quality":settings.get("demonstration_quality", {}),
            "demonstration_policy": settings["demonstration_policy"],
            "demonstration_stopping": settings.get("demonstration_stopping"),
            "teacher_choice_evidence": teacher_choices} if teacher else {}),
        "surface_supervision":"none: opaque all-U direct execution is not a completed checker label"}}
