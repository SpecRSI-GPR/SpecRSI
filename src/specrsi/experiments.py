"""Prospective local-model experiments joined to the SpecRSI evidence/search kernel.

Tokens are an enforceable inference budget in this runner; measured GPU seconds
are recorded separately from the token budget.
"""
from __future__ import annotations

from dataclasses import asdict
from collections.abc import Mapping
import json
import hashlib
import os
from pathlib import Path
import random
import secrets
import stat
import time
import uuid

from .benchmarks import (BenchmarkExample, aime_score, game24_score, hotpotqa_score,
                         hover_score, ifbench_score, load_examples, validate_disjoint_splits)
from .models import LocalModelClient
from .models.client import sampling_options
from .output_contracts import generation_messages, output_contract_instruction
from .core.models import Candidate, EvaluationScope, Outcome, Task, thaw
from .core.execution import ExecutionResult


def receipt_secret(state_dir: str | Path = ".state") -> tuple[bytes, Path]:
    """Publish a complete local host key atomically, with exclusive 0600 files.

    Snapshot JSON records its path only. The key is never serialized into reports,
    prompts, model requests, or source-control artifacts.
    """
    directory = Path(state_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "ledger-receipt.key"
    if not path.exists():
        temporary = directory / (".ledger-receipt-" + uuid.uuid4().hex)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(secrets.token_bytes(32))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)  # Atomic publication without replacing an existing key.
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError(f"Ledger receipt key must be a regular 0600 file: {path}")
        key = handle.read()
    if len(key) != 32:
        raise ValueError(f"Ledger receipt key has invalid length: {path}")
    return key, path


def score_answer(dataset: str, prediction: str, reference, *, metric_variant: str | None = None):
    if dataset == "planbench":
        from .benchmarks.planning import planbench_example_score
        return planbench_example_score(BenchmarkExample(**reference['_specrsi_example']), prediction)
    if dataset == "math":
        from .benchmarks.math_native import math_score
        return math_score(prediction, reference)
    if dataset == "livebench_math":
        from .benchmarks.livebench import livebench_math_score
        return livebench_math_score(prediction, reference)
    if dataset == "mbpp":
        from .benchmarks.external import mbpp_bubblewrap_score
        text = prediction.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        return mbpp_bubblewrap_score(text, reference)
    if dataset == "hover":
        if metric_variant != "label_accuracy_not_GEPA_multihop_retrieval":
            raise ValueError("HoVer requires the registered retrieval harness, or an explicitly declared prospective label-accuracy variant")
        return hover_score(prediction, reference, metric="label_accuracy")
    functions = {"game24": game24_score, "aime25": aime_score, "aime_2025_2026": aime_score, "hotpotqa": hotpotqa_score,
                 "ifbench": ifbench_score}
    if dataset not in functions:
        raise ValueError(f"{dataset} requires its native harness; use specrsi.benchmarks.external rather than a substitute score")
    return functions[dataset](prediction, reference)


class QwenExecutor:
    def __init__(self, client: LocalModelClient, tokenizer, model_config: dict):
        self.client, self.tokenizer, self.config = client, tokenizer, json.loads(json.dumps(model_config))
        self.physical_records: list[dict] = []

    def prompt_tokens(self, messages, *, enable_thinking: bool | None = None):
        enabled = self.config.get("enable_thinking", False) if enable_thinking is None else enable_thinking
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                   enable_thinking=enabled, thinking=enabled))

    @staticmethod
    def _validate_output_contract(candidate):
        contract = output_contract_instruction(candidate.runtime["generation"])
        if contract and ("workflow" in candidate.artifacts or "answer_strategy" in candidate.artifacts):
            raise ValueError("output_contract currently requires single-prompt execution")

    @staticmethod
    def generation_messages(candidate, task):
        return generation_messages(candidate.artifacts["system_prompt"], task.inputs["prompt"],
            candidate.runtime["generation"], dataset=task.metadata.get("dataset"))

    def validate_prompt_candidate(self, candidate, tasks, execution_cap: int):
        """Require the same configured answer budget on every selection item."""
        self._validate_output_contract(candidate)
        options = sampling_options(candidate.runtime["generation"])
        if options:
            self.client.ensure_sampling_support(options)
        if "answer_strategy" in candidate.artifacts:
            from .answer_strategies import AnswerStrategy, evidence_messages, reasoning_enable_thinking
            if set(candidate.artifacts) != {"system_prompt", "answer_strategy"}:
                raise ValueError("Answer strategy requires exactly a system_prompt and answer_strategy artifact")
            strategy = AnswerStrategy.from_dict(thaw(candidate.artifacts["answer_strategy"]))
            enabled = reasoning_enable_thinking(strategy, candidate.runtime["generation"])
            for task in tasks:
                if strategy.version == "hotpot_evidence_v1" and task.metadata.get("dataset") != "hotpotqa":
                    raise ValueError("The answer strategy extension is registered for HotpotQA only")
                messages = evidence_messages(candidate.artifacts["system_prompt"], task.inputs["prompt"], strategy)
                required = self.prompt_tokens(messages, enable_thinking=enabled) + strategy.reasoning_tokens
                if required > min(execution_cap, 32768):
                    raise ValueError(f"Frozen evidence stage requires {required} tokens for {task.name}; execution cap is {execution_cap}")
            # Later prompts contain generated evidence, and are checked before
            # every request. This is not a proof of whole-recipe affordability.
            return
        if "workflow" in candidate.artifacts:
            from .workflows import Workflow
            Workflow.from_dict(thaw(candidate.artifacts["workflow"]))
            return
        config = thaw(candidate.runtime["generation"])
        for task in tasks:
            messages = self.generation_messages(candidate, task)
            required = self.prompt_tokens(messages, enable_thinking=config.get("enable_thinking", False)) + config.get("max_tokens", 512)
            if required > min(execution_cap, 32768):
                raise ValueError(f"Frozen prompt and configured output cap require {required} tokens for {task.name}; execution cap is {execution_cap}")

    def run(self, candidate, task, scope, cap, resume_token=None):
        if not scope.contains(task):
            raise ValueError("task is outside frozen evaluation scope")
        self._validate_output_contract(candidate)
        call_config = thaw(candidate.runtime["generation"])
        if "answer_strategy" in candidate.artifacts:
            from .answer_strategies import AnswerStrategy, execute_answer_strategy
            if set(candidate.artifacts) != {"system_prompt", "answer_strategy"}:
                raise ValueError("Answer strategy requires exactly a system_prompt and answer_strategy artifact")
            strategy = AnswerStrategy.from_dict(thaw(candidate.artifacts["answer_strategy"]))
            if strategy.version == "hotpot_evidence_v1" and task.metadata.get("dataset") != "hotpotqa":
                raise ValueError("The answer strategy extension is registered for HotpotQA only")
            completed = execute_answer_strategy(strategy, task.inputs["prompt"],
                candidate.artifacts["system_prompt"], self, generation=call_config,
                model_revision=candidate.runtime["model_revision"], cap=cap,
                seed=int(candidate.runtime["seed"]))
            for record in completed.records:
                self.physical_records.append({"candidate": candidate.uid, "task": task.uid,
                    "scope": scope.uid, "kind": "answer_strategy", **record})
            receipt = {"requests": [r["request_id"] for r in completed.records],
                "tokens": completed.tokens, "cost_unit": "tokens",
                "gpu_seconds": sum(r.get("elapsed_seconds", 0) * r.get("gpu_count", 1) for r in completed.records),
                "metering_complete": all(r["metering_status"] == "server_measured" for r in completed.records),
                "answer_strategy": strategy.to_dict(), "extension_beyond_pdf_reference": True}
            if completed.status != "complete":
                return ExecutionResult(completed.status, completed.tokens, receipt=receipt, error=completed.error)
            try:
                score = score_answer(task.metadata["dataset"], completed.answer, thaw(task.expected),
                                     metric_variant=task.metadata.get("metric_variant"))
            except Exception as error:
                return ExecutionResult("error", completed.tokens, receipt=receipt,
                                       error=f"Native scorer failed after metered strategy execution ({type(error).__name__})")
            return ExecutionResult("complete", completed.tokens, Outcome(score.score,
                {"answer": completed.answer, "metrics": dict(score.metrics),
                 "finish_reason": completed.records[-1]["finish_reason"],
                 "deployment_tokens": completed.tokens, "strategy_decision": completed.decision,
                 "node_outputs": {r["stage"]: r["text"] for r in completed.records}}), receipt=receipt)
        if "workflow" in candidate.artifacts:
            from .workflows import Workflow, execute_workflow
            workflow_executor = QwenExecutor(self.client, self.tokenizer, call_config)
            answer, actual, records = execute_workflow(Workflow.from_dict(thaw(candidate.artifacts["workflow"])),
                task.inputs["prompt"], workflow_executor, cap=cap, seed=int(candidate.runtime["seed"]))
            for node_id, reply in records:
                self.physical_records.append({"candidate": candidate.uid, "task": task.uid,
                    "scope": scope.uid, "node": node_id, **asdict(reply)})
            receipt = {"requests": [r.request_id for _, r in records], "tokens": actual,
                       "gpu_seconds": sum(r.elapsed_seconds * r.gpu_count for _, r in records), "cost_unit": "tokens"}
            if any(r.model_revision != candidate.runtime["model_revision"] for _, r in records):
                return ExecutionResult("error", actual, receipt=receipt, error="workflow model revision changed")
            if answer is None:
                return ExecutionResult("interrupted", actual, receipt=receipt, error="incomplete workflow prefix")
            score = score_answer(task.metadata["dataset"], answer, thaw(task.expected), metric_variant=task.metadata.get("metric_variant"))
            return ExecutionResult("complete", actual, Outcome(score.score, {"answer": answer,
                "metrics": dict(score.metrics), "finish_reason": records[-1][1].finish_reason, "deployment_tokens": actual,
                "node_outputs": {key: r.text for key, r in records}}), receipt=receipt)
        messages = self.generation_messages(candidate, task)
        prompt_cost = self.prompt_tokens(messages, enable_thinking=call_config.get("enable_thinking", False))
        max_new = call_config.get("max_tokens", 512)
        if max_new < 1 or prompt_cost + max_new > min(cap, 32768):
            return ExecutionResult("interrupted", 0, error="full frozen prompt/output contract does not fit token cap")
        reply = self.client.generate(messages, max_tokens=max_new,
            temperature=call_config.get("temperature", .7), top_p=call_config.get("top_p", .8), seed=int(candidate.runtime["seed"]),
            enable_thinking=call_config.get("enable_thinking", False),
            max_seconds=call_config.get("max_seconds", 120), total_token_cap=cap,
            **sampling_options(call_config))
        physical = {"candidate": candidate.uid, "task": task.uid, "scope": scope.uid, **asdict(reply)}
        self.physical_records.append(physical)
        actual = reply.prompt_tokens + reply.completion_tokens
        receipt = {"request_id": reply.request_id, "model_revision": reply.model_revision,
                   "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
                   "gpu_seconds": reply.elapsed_seconds * reply.gpu_count, "cost_unit": "tokens"}
        if "output_contract" in call_config:
            receipt.update(output_contract=call_config["output_contract"], extension_beyond_pdf_reference=True)
        if reply.model_revision != candidate.runtime["model_revision"]:
            return ExecutionResult("error", actual, receipt=receipt, error="model revision changed during frozen evaluation")
        if reply.prompt_tokens != prompt_cost:
            return ExecutionResult("error", actual, receipt=receipt, error="executor/server tokenizer counts differ from frozen request")
        if reply.finish_reason == "timeout":
            return ExecutionResult("interrupted", actual, receipt=receipt, error="generation timeout without complete scored outcome")
        score = score_answer(task.metadata["dataset"], reply.text, thaw(task.expected), metric_variant=task.metadata.get("metric_variant"))
        return ExecutionResult("complete", actual, Outcome(score.score,
            {"answer": reply.text, "metrics": dict(score.metrics), "finish_reason": reply.finish_reason}), receipt=receipt)


class QwenDrafter:
    def __init__(self, executor: QwenExecutor, execution_cap: int = 4096, *, task_executor=None, drafter_revision=None):
        self.executor = executor
        self.execution_cap = execution_cap
        self.task_executor = task_executor or executor
        self.drafter_revision = drafter_revision

    def draft(self, parent, state, cap, rng):
        from .core.search import DraftResult
        feedback = []
        for task in state.scope.tasks:
            record = state.ledger.get(parent, task, state.scope)
            if record:
                feedback.append({"input": task.inputs["prompt"], "score": record.outcome.score,
                                 "output": record.outcome.fields["answer"]})
        workflow_mode = "workflow" in parent.artifacts
        instruction = ("Improve the task-solving workflow. Return only JSON with field workflow containing nodes and output_node. Each node has id, instruction, dependencies (list of node IDs). Use 1 to 8 nodes in an acyclic graph; every node must contribute to output_node. Preserve the final answer format."
                       if workflow_mode else "Improve the system instruction for a task-solving language model. Return only JSON with one string field: system_prompt. Do not answer individual tasks. Preserve the required answer format.")
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps({"parent_artifacts": thaw(parent.artifacts),
                        "selection_feedback": feedback}, ensure_ascii=False)}]
        prompt_cost = self.executor.prompt_tokens(messages, enable_thinking=False)
        max_new = min(768, cap - prompt_cost)
        if max_new < 1:
            return DraftResult(None, 0, status="interrupted", error="draft prompt exceeds token cap")
        reply = self.executor.client.generate(messages, max_tokens=max_new, temperature=.7,
                                              seed=rng.randrange(2**32), enable_thinking=False,
                                              max_seconds=self.executor.config.get("max_seconds", 120), total_token_cap=cap,
                                              **sampling_options(self.executor.config))
        actual = reply.prompt_tokens + reply.completion_tokens
        self.executor.physical_records.append({"kind": "draft", "parent": parent.uid, **asdict(reply)})
        if reply.model_revision != (self.drafter_revision or parent.runtime["model_revision"]):
            return DraftResult(None, actual, status="error", error="drafter model revision changed")
        if reply.finish_reason == "timeout":
            return DraftResult(None, actual, status="interrupted", error="draft generation timed out")
        text = reply.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        try:
            proposed = json.loads(text)
            if workflow_mode:
                from .workflows import Workflow
                Workflow.from_dict(proposed["workflow"])
                artifacts = {"workflow": proposed["workflow"]}
            else:
                instruction = proposed["system_prompt"]
                if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 12000:
                    raise ValueError("empty or oversized instruction")
                artifacts = thaw(parent.artifacts)
                artifacts["system_prompt"] = instruction
        except (ValueError, KeyError, TypeError) as error:
            return DraftResult(None, actual, status="error", error=f"invalid draft JSON: {error}")
        child = Candidate(name=f"qwen-draft-{len(state.candidates)}", artifacts=artifacts,
            runtime=thaw(parent.runtime), semantic_kind="opaque", deterministic=False, parent_id=parent.uid)
        try:
            self.task_executor.validate_prompt_candidate(child, state.scope.tasks, self.execution_cap)
        except ValueError as error:
            return DraftResult(None, actual, status="error", error=f"unexecutable frozen draft: {error}")
        return DraftResult(child, actual, forecasts={task.uid: 0.0 for task in state.scope.tasks})


def carried_initial_artifacts(config, search):
    """Carry only a checked selected artifact; every target runtime is rebuilt."""
    path=config.get('initial_candidate_manifest')
    if path is None:
        artifacts = ({'workflow':search['initial_workflow']} if 'initial_workflow' in search
                     else {'system_prompt':search['initial_prompt']})
        if 'answer_strategy' in search:
            from .answer_strategies import AnswerStrategy
            if 'workflow' in artifacts:
                raise ValueError('Answer strategy cannot be combined with a workflow')
            artifacts['answer_strategy'] = AnswerStrategy.from_dict(search['answer_strategy']).to_dict()
        return artifacts, None
    path=Path(path)
    raw_bytes=path.read_bytes();actual_hash=hashlib.sha256(raw_bytes).hexdigest()
    expected=config.get('initial_candidate_sha256')
    if expected is not None and expected!=actual_hash:
        raise ValueError('Carried candidate manifest hash differs from the registered snapshot')
    manifest=json.loads(raw_bytes)
    source=Candidate(**manifest)
    if source.semantic_kind not in {'opaque', 'prompt_inference_extension'} or not isinstance(source.runtime.get('model_revision'),str):
        raise ValueError('Carry requires a recognized prompt-model candidate with explicit source runtime identity')
    if type(source.runtime.get('seed')) is not int or not isinstance(source.runtime.get('generation'),Mapping):
        raise ValueError('Carried source runtime is incomplete')
    artifacts=thaw(source.artifacts)
    if source.semantic_kind == 'prompt_inference_extension' and set(artifacts) not in ({'system_prompt'}, {'system_prompt', 'answer_strategy'}):
        raise ValueError('Prompt inference extensions can carry only checked prompt/strategy artifacts')
    if set(artifacts)=={'workflow'}:
        from .workflows import Workflow
        Workflow.from_dict(artifacts['workflow'])
    elif set(artifacts) in ({'system_prompt'}, {'system_prompt', 'answer_strategy'}):
        if not isinstance(artifacts['system_prompt'],str) or not artifacts['system_prompt'].strip() or len(artifacts['system_prompt'])>12000:
            raise ValueError('Invalid carried system instruction')
        if 'answer_strategy' in artifacts:
            from .answer_strategies import AnswerStrategy
            AnswerStrategy.from_dict(artifacts['answer_strategy'])
    else:raise ValueError('Only a single registered prompt or workflow artifact can be carried')
    return artifacts,{'manifest_path':str(path.resolve()),'manifest_sha256':actual_hash,
        'source_candidate_id':source.uid,'source_model_revision':source.runtime['model_revision'],
        'carried_fields':['artifacts'],'runtime_policy':'recomputed_from_current_model_config_and_seed',
        'ledger_policy':'fresh_full_initial_evaluation_no_previous_outcomes_imported'}


def run_local_experiment(config: dict):
    from transformers import AutoTokenizer
    from .core import BudgetLedger, Ledger, SearchConfig, SearchEngine
    from .paper_profile import (algorithm_name, algorithm_metadata, require_role_checkpoint, vats_config,
                               vats_cost_accounting, objective_coefficients, realized_search_return)
    algorithm = algorithm_name(config["search"])
    paper = algorithm_metadata(config) if algorithm == "vats_pdf" else None
    checkpoint_path = require_role_checkpoint(config["search"])
    search = config['search']
    objective_coefficients(search)
    execution_cap = search.get('execution_cap', search.get('execution_cap_tokens', 4096))
    options = SearchConfig(execution_cap=execution_cap,
        draft_cap=search.get('draft_cap', search.get('draft_cap_tokens', 4096)), lookup_cost=0, check_cap=0,
        max_sources=0, planning_cost=0, outer_value_cap=0, batch_value_cap=0,
        max_actions=search['max_actions'], max_candidates=search['max_candidates'],
        max_children=search.get('max_children', 2), temperature=search.get('temperature', 1.),
        exploration=search.get('exploration', .1), margin=search.get('eta', 0.),
        seed=config['seed'], all_u=True, cost_unit='tokens')
    if algorithm == 'vats_pdf':
        # Check the complete menu/resource contract before acquiring the root.
        role_manifest = json.loads((checkpoint_path/'model_config.json').read_text()) if checkpoint_path else {}
        from .learning.frozen_text import semantic_input_cap
        options = vats_config(options, search, learned_query_cap=semantic_input_cap(role_manifest) if checkpoint_path else 0)
    from .core.budget import cost_int
    cost_int(search['budget_tokens'])
    initial_cap = search.get('initial_cap_tokens', max(16384, execution_cap))
    cost_int(initial_cap)
    if algorithm == "vats_pdf":
        # Reserve a new result location before making any paid model request.
        Path(config["output"]).mkdir(parents=True, exist_ok=False)
    dataset = config["dataset"]
    if dataset not in {"game24", "aime25", "aime_2025_2026", "hotpotqa", "hover", "ifbench", "mbpp", "math", "livebench_math", "polyglot", "pupa", "planbench", "swe_verified", "swe_lite"}:
        raise ValueError(f"{dataset} needs its registered native harness; the prompt bootstrap runner does not substitute an answer-only metric")
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src/specrsi").rglob("*.py"))}
    swe_mode = dataset in {"swe_verified", "swe_lite"}
    structured_mode = swe_mode or config.get("task_pipeline") in {"hover_three_hop", "pupa_local_judge", "polyglot_native", "livebench_math_local_judge"}
    if dataset == "hover" and not structured_mode and config.get("metric_variant") != "label_accuracy_not_GEPA_multihop_retrieval":
        raise ValueError("Declare the prospective HoVer label-accuracy variant, or use the registered retrieval harness")
    if swe_mode:
        registry=json.loads(Path(config.get('coding_registry','configs/coding/swe_tracks.json')).read_text())
        def pool_of(name):
            return {e.id:e for i,split in enumerate(('selection','development','audit'))
                    for e in load_examples(name,split,data_root=config.get('data_root','data'),verify=i==0)}
        selection_pool=pool_of(dataset)
        audit_pool=pool_of(config.get('audit_dataset','swe_lite'))
        selection_ids=config.get('selection_ids',registry[config.get('selection_track','selection')])
        audit_ids=config.get('audit_ids',registry[config.get('audit_track','audit')])
        from dataclasses import replace
        selection=[replace(selection_pool[key],split='selection') for key in selection_ids]
        audit=[replace(audit_pool[key],split='audit') for key in audit_ids]
    else:
        selection = load_examples(dataset, "selection", data_root=config.get("data_root", "data"))
        # The selection load verified every file of this dataset's manifest,
        # including audit bytes; avoid hashing the same large source tree twice.
        audit = load_examples(dataset, "audit", data_root=config.get("data_root", "data"), verify=False)
    validate_disjoint_splits({"selection": selection, "audit": audit})
    selection = selection[:config.get("selection_limit", len(selection))]
    audit = audit[:config.get("audit_limit", len(audit))]
    if not selection:
        raise ValueError("No selection examples")
    model_config, search = config["model"], config["search"]
    if search["controller"] not in {"heuristic_bootstrap", "learned", "quality_rule", "completion_cost_rule"}:
        raise ValueError("Unknown controller; learned requires a frozen role_checkpoint")
    if search.get("reuse", "all_U") != "all_U":
        raise ValueError("Opaque LLM generation supports direct U execution only in this runner")
    lock = json.loads(Path(model_config["lock"]).read_text())
    client = LocalModelClient(model_config["base_url"], model_config.get("served_model_name", lock.get("served_model_name", "local-model")))
    health = client.health()
    if not health.get("server_enforces_total_token_cap") or health.get("protocol_version", 0) < 2:
        raise ValueError("Local model server needs protocol v2 with enforced total token caps; restart it from current source")
    if health["revision"] != lock["revision"]:
        raise ValueError("Running model revision differs from locked checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(lock["local_path"], local_files_only=True, trust_remote_code=lock.get("trust_remote_code", False))
    if swe_mode:
        from .coding_agent import SWERepositoryExecutor
        executor = SWERepositoryExecutor(client, tokenizer, model_config)
    elif structured_mode:
        from .task_pipelines import TaskPipelineExecutor
        executor = TaskPipelineExecutor(client, tokenizer, model_config)
    else:
        executor = QwenExecutor(client, tokenizer, model_config)
    artifacts,initial_candidate_provenance = carried_initial_artifacts(config,search)
    incumbent = Candidate("initial", artifacts,
        {"model_revision": lock["revision"], "seed": config["seed"], "generation": model_config},
        semantic_kind="opaque", deterministic=False)
    def task_of(example):
        reference = {'_specrsi_example': example.to_dict()} if dataset == 'planbench' else example.reference
        return Task(example.id, {"prompt": example.prompt}, reference,
                    {"dataset": example.dataset, "content_hash": example.content_hash, "split": example.split,
                     "metric_variant": config.get("metric_variant")})
    tasks = tuple(task_of(example) for example in selection)
    scorer_id = dataset + "-" + config.get("metric_variant", "registered-score") + "-v1"
    scope = EvaluationScope("local-" + dataset, tasks, "swebench4.1.0-repository-actor-v1" if swe_mode else "qwen-capped-generation-v2", scorer_id,
                            scored_fields=("answer", "metrics", "finish_reason"),
                            replay={"generation": model_config, "model_revision": lock["revision"]})
    secret, secret_path = receipt_secret(config.get("state_dir", ".state"))
    ledger = Ledger(receipt_secret=secret)
    executor.validate_prompt_candidate(incumbent, tasks, execution_cap)
    # Initial-incumbent acquisition is paid and reported separately, as the paper
    # starts from a fully evaluated S0. Audit never updates this selection ledger.
    initial_cost = 0
    for task in tasks:
        result = executor.run(incumbent, task, scope, initial_cap)
        if result.actual_cost > initial_cap:
            raise RuntimeError("Initial execution violated enforced token cap")
        initial_cost += result.actual_cost
        if result.status != "complete":
            raise RuntimeError(f"Initial incumbent incomplete: {result.error}")
        ledger.admit_direct(incumbent, task, scope, result.outcome, result.receipt)
    budget = BudgetLedger(search["budget_tokens"])
    workflow_mode = "workflow" in incumbent.artifacts
    answer_strategy_mode = "answer_strategy" in incumbent.artifacts
    if algorithm != 'vats_pdf':
        from dataclasses import replace
        options = replace(options, reserve_completion=not (workflow_mode or structured_mode or answer_strategy_mode))
    started = time.monotonic()
    draft_executor, draft_revision = executor, lock['revision']
    if config.get('drafter_model'):
        draft_config=config['drafter_model']; draft_lock=json.loads(Path(draft_config['lock']).read_text())
        draft_client=LocalModelClient(draft_config['base_url'],draft_config.get('served_model_name',draft_lock.get('served_model_name','local-model')))
        draft_health=draft_client.health()
        if draft_health['revision']!=draft_lock['revision']: raise ValueError('Drafter revision differs from its frozen lock')
        draft_tokenizer=AutoTokenizer.from_pretrained(draft_lock['local_path'],local_files_only=True,trust_remote_code=draft_lock.get('trust_remote_code',False))
        draft_executor=QwenExecutor(draft_client,draft_tokenizer,draft_config)
        draft_executor.physical_records=executor.physical_records
        draft_revision=draft_lock['revision']
    drafter=QwenDrafter(draft_executor,execution_cap,task_executor=executor,drafter_revision=draft_revision)
    bridge=None; rule_controller=None; hooks={}; role_checkpoint_hashes={}
    if search['controller']=='learned' or (search['controller'] in {'quality_rule','completion_cost_rule'} and search.get('role_checkpoint')):
        from dataclasses import replace
        from .learning.hf import HFRoleModel
        from .learning.bridge import NeuralSearchBridge,HFPromptDrafter
        checkpoint=Path(search['role_checkpoint'])
        if not (checkpoint/'model_config.json').is_file(): raise ValueError('Missing trained role checkpoint')
        role_checkpoint_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.rglob('*') if p.is_file()}
        role_model=HFRoleModel.load(checkpoint,device=search.get('role_device','cuda:1'))
        bridge_class = NeuralSearchBridge
        if algorithm == "vats_pdf":
            from .learning.paper_bridge import VATSSearchBridge
            bridge_class = VATSSearchBridge
        bridge_options = ({'snapshot_id': hashlib.sha256(json.dumps(role_checkpoint_hashes,sort_keys=True).encode()).hexdigest()}
                          if algorithm == 'vats_pdf' else {})
        role_model.eval(); bridge=bridge_class(role_model,cost_unit='tokens',batch_mode=search.get('batch_mode','joint'),observation_mode=search.get('observation_mode','full'),observation_donors=search.get('observation_donors'),**bridge_options)
        options=replace(options,outer_value_cap=role_model.config.max_length,batch_value_cap=role_model.config.max_length)
        drafter=HFPromptDrafter(bridge,max_new_tokens=search.get('draft_max_tokens',128),top_k=search.get('draft_top_k',32),
            temperature=search.get('draft_temperature',1.),exploration=search.get('draft_exploration',0.),
            proposal_snapshot=hashlib.sha256(json.dumps(role_checkpoint_hashes,sort_keys=True).encode()).hexdigest(),
            candidate_validator=lambda c:executor.validate_prompt_candidate(c,tasks,execution_cap))
        hooks=({'scorer':bridge.scorer,'policy_menu':bridge.policy_menu} if algorithm == "vats_pdf"
               else {'outer_value':bridge.outer_value,'batch_value':bridge.batch_value})
    if search['controller'] in {'quality_rule', 'completion_cost_rule'}:
        from .learning.observations import make_rule_controller
        rule_controller=make_rule_controller(options,
            rule='quality_only' if search['controller']=='quality_rule' else 'completion_cost',
            initial_fraction=search.get('rule_initial_fraction',.05),
            draft_fraction=search.get('rule_draft_fraction',.35))
        hooks={'outer_value':rule_controller.outer_value,'batch_value':rule_controller.batch_value}
    engine_class = SearchEngine
    if algorithm == "vats_pdf":
        from .core.vats import VATSEngine
        engine_class = VATSEngine
        from .learning.frozen_text import semantic_input_cap
        options = vats_config(options, search, learned_query_cap=semantic_input_cap(role_model.config) if bridge is not None else 0)
        if bridge is not None:
            options = replace(options, scorer_snapshot=bridge.snapshot_id, policy_snapshot=bridge.snapshot_id)
    engine = engine_class(scope, incumbent, ledger, budget, executor, drafter, options,**hooks)
    result = engine.run()
    if bridge is not None:
        executor.physical_records.extend(bridge.physical_inferences)
    if rule_controller is not None:
        executor.physical_records.extend(rule_controller.physical_inferences)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    data.update({"purpose": config["purpose"], "config": config, "model": health,
                 "algorithm": algorithm, "algorithm_profile": paper,
                 "cost_accounting": vats_cost_accounting("tokens", options.policy_call_cost,
                     role_model.config.architecture_version if bridge is not None else 2) if algorithm == "vats_pdf" else None,
                 "cost_unit": "tokens", "initial_acquisition_tokens": initial_cost,
                 "search_wall_seconds": time.monotonic() - started,
                 "author_checkpoint_used": False, "paper_budget_comparable": False,
                 "answer_strategy_extension": thaw(incumbent.artifacts.get("answer_strategy")),
                 "role_checkpoint_hashes":role_checkpoint_hashes,
                 "controller_observations":bridge.observation_records if bridge is not None else [],
                 "rule_completion_cost_variant":rule_controller.variant if rule_controller is not None else None,
                 "ledger_receipt_secret_path": str(secret_path),
                 "completion_reservation_enabled": options.reserve_completion,
                 "completion_reservation_reason": "PDF uses affordable completion seeds in the common plan menu" if algorithm == "vats_pdf" else ("single-prompt token cap validated" if not (workflow_mode or structured_mode or answer_strategy_mode) else "Generated intermediate context prevents a proven full-scenario token bound"),
                 "search_return": realized_search_return(result, search),
                 "objective_coefficients": dict(zip(('cost_lambda', 'time_mu'), objective_coefficients(search))),
                 "source_hashes": source_hashes,
                 "initial_candidate_provenance": initial_candidate_provenance,
                 "source_changed_during_run": source_hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(Path("src/specrsi").rglob("*.py"))}})
    # Save the selected immutable artifact before requesting independent audit work.
    (output / "search.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    selected = result.selected
    (output / "selected.json").write_text(json.dumps(selected.manifest(), indent=2, ensure_ascii=False) + "\n")
    audit_rows = []
    if audit:
        audit_tasks = tuple(task_of(example) for example in audit)
        audit_scope = EvaluationScope("audit-" + dataset, audit_tasks, scope.execution_id, scope.scorer_id,
                                      scored_fields=scope.scored_fields, replay=scope.replay, split="audit")
        for task in audit_tasks:
            acquired = executor.run(selected, task, audit_scope, initial_cap)
            if acquired.actual_cost > initial_cap:
                raise RuntimeError("Audit execution violated enforced token cap")
            audit_rows.append({"task": task.uid, "status": acquired.status, "tokens": acquired.actual_cost,
                               "score": acquired.outcome.score if acquired.outcome else None})
    (output / "audit.json").write_text(json.dumps(audit_rows, indent=2) + "\n")
    (output / "physical_calls.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in executor.physical_records))
    return {"output": str(output), "selected": selected.name, "search": data,
            "audit": audit_rows, "physical_gpu_seconds": sum(r.get("elapsed_seconds", 0) * r.get("gpu_count", 1) for r in executor.physical_records)}
