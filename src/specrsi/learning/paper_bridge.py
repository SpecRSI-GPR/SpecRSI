"""PDF VATS collection: independent policy, single scorer, paid-state replay."""
from __future__ import annotations

from dataclasses import replace
import math
import json
from pathlib import Path
import random
import time

import torch

from specrsi.core.budget import BudgetLedger
from specrsi.core.checker import ConservativeChecker
from specrsi.core.execution import PureIntegerExecutor
from specrsi.core.ledger import Ledger
from specrsi.core.models import Candidate, EvaluationScope, Task, thaw
from specrsi.core.search import DraftResult, SearchConfig, ValueEstimate
from specrsi.planning.policy import TraceDecision

from .bridge import NeuralSearchBridge, _pad
from .objectives import mixture_probabilities
from .records import DecisionRecord, PairwiseValueExample, SearchTrajectory, TransitionRecord, validate_group
from .work_graph import graph_schema, ACTUAL_GRAPH_SCHEMA, augment_actual_jobs, work_node_texts


class VATSSearchBridge(NeuralSearchBridge):
    """A menu gets one policy forward after its contracted expense is paid.

    Scorer calls used in refinement are separate paid calls. The actor sees
    frozen numeric features and text from the actual post-preparation state;
    no labels are assigned to unexecuted alternatives.
    """
    direct_only = True

    def __init__(self, model, *, cost_unit="logical-work", snapshot_id="", **kwargs):
        super().__init__(model, cost_unit=cost_unit, **kwargs)
        if model.config.option_dim < 12 or self.feature_dim < 8:
            raise ValueError("PDF VATS requires feature_dim >= 8 and option_dim >= 12")
        self.snapshot_id = snapshot_id
        self.initial_budget = None
        self.workers = None
        self.contract_id = ""
        graph_schema(model.config)
        from .frozen_text import role_observation_contract
        role_observation_contract(model.config)
        if getattr(model.config, "architecture_version", 2) == 3:
            from .frozen_text import node_text_cache
            node_text_cache(model).validate_identity()

    def state_features(self, state):
        features = list(state.features())
        candidates = max(1, len(state.candidates))
        features[1] = candidates / (1. + candidates)
        features[2] /= candidates
        features[3] /= candidates
        features[6] = state.workers / (1. + state.workers)
        return _pad(features, self.feature_dim)

    def context_text(self, state, *, bounded=True):
        """Prioritize paid resources and actual failure feedback in bounded text.

        A missing item has no guessed score. Only admitted outcomes and logged
        execution errors enter feedback; task expected/reference fields never do.
        Graph features separately retain the complete observed missing mask.
        """
        def preview(value, limit=240):
            text = value if isinstance(value, str) else json.dumps(thaw(value), ensure_ascii=False, separators=(",", ":"))
            return text if len(text) <= limit else text[:limit] + "…"

        observed_nodes = getattr(self.model.config, "role_observation_version", "legacy_v1") == "observed_results_v2"
        candidates, outcomes = [], []
        for candidate in state.candidates.values():
            missing = state.ledger.missing(candidate, state.scope)
            candidates.append({"id": candidate.uid[:12], "parent": (candidate.parent_id or "")[:12],
                "missing": [task.uid[:12] for task in missing],
                "complete_score": state.ledger.score(candidate, state.scope)
                    if state.ledger.admissible(candidate, state.scope) else None})
            for task in state.scope.tasks:
                record = state.ledger.get(candidate, task, state.scope)
                if record is None:
                    continue
                outcome = record.outcome
                outcomes.append({"candidate": candidate.uid[:12], "task": task.uid[:12],
                    "score": outcome.score, "admissible": outcome.admissible,
                    "observed": ({"text_location": "candidate_task_node"} if observed_nodes else
                        {key: preview(outcome.fields[key], 160) for key in
                         ("answer", "finish_reason", "native_score", "valid") if key in outcome.fields})})
        # Put unsuccessful/inadmissible observations first when the text budget
        # cannot fit every outcome. Numeric/graph inputs still hold every node.
        outcomes.sort(key=lambda item: (item["admissible"], item["score"], item["candidate"], item["task"]))
        errors = [{key: preview(event[key], 160) if isinstance(event[key], str) and not observed_nodes else event[key]
                   for key in ("event", "candidate_id", "task_id", "status", "error", "cost") if key in event}
                  for event in state.history if event.get("error") or event.get("status") in {"error", "interrupted"}][-4:]
        context = {"scope": state.scope.name,
            "resources": {"available_budget": state.budget.available, "pending": state.budget.pending,
                "initial_budget": state.budget.total, "remaining_attempts": state.remaining_attempts,
                "initial_attempts": state.initial_attempts, "workers": state.workers},
            "recent_execution_failures": errors, "candidates": candidates, "observed_outcomes": outcomes,
            "artifacts": [{"id": candidate.uid[:12], "frozen": preview(candidate.artifacts, 384)}
                          for candidate in state.candidates.values()],
            "task_inputs": [{"id": task.uid[:12], "visible_input": preview(task.inputs, 200)} for task in state.scope.tasks]}
        if observed_nodes:
            # Artifact/task/result text lives on its actual graph node and uses
            # the registered shared token budget. Do not first destroy it with
            # an unrelated character cap in a redundant global serialization.
            context.pop("artifacts")
            context.pop("task_inputs")
            context["contract"] = {key: getattr(state.scope, key) for key in
                ("execution_id", "scorer_id", "aggregation_id", "constraints_id", "scored_fields")}
        text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if bounded and self.model.backend == "hf" and getattr(self.model.config, "architecture_version", 2) == 3:
            from .frozen_text import bounded_semantic_text, semantic_budget
            budget = semantic_budget(self.model.config)
            if budget["mode"] == "redistribute":
                return bounded_semantic_text(self.model.tokenizer, text, budget["global"],
                                             strategy=budget["truncation"])[0]
        return self.bounded_text(text, self.model.config.max_length // 3) if bounded and self.model.backend == "hf" else text

    def plan_features(self, state, plan):
        """Recorded resource ratios keep token/millisecond caps well scaled.

        Raw costs remain in the engine's immutable plan manifests and ledger.
        Replay uses these exact saved ratios, never recomputed denominators.
        """
        features = list(plan.features(state))
        candidates = max(1, len(state.candidates))
        task_pairs = candidates * max(1, len(state.scope.tasks))
        attempts = max(1, state.initial_attempts)
        for index, scale in ((3, candidates), (4, attempts), (5, task_pairs),
                             (6, max(1, state.budget.total)), (7, attempts),
                             (8, candidates), (9, candidates), (10, task_pairs), (11, task_pairs)):
            features[index] /= scale
        return _pad(features, self.option_dim)

    def drafter_context_text(self, state, parent, *, token_limit=None):
        """Use the admitted text budget for the real parent's observed feedback."""
        from .draft_context import parent_feedback_context
        return parent_feedback_context(self, state, parent, token_limit=token_limit)

    def _record(self, state, plans):
        text = self.context_text(state)
        nodes, edges, masks = self._graph_state(state, plans)
        semantics = self._node_text_record(state, text) if getattr(self.model.config, "architecture_version", 2) == 3 else {}
        if self.model.backend == "hf" and getattr(self.model.config, "semantic_budget_mode", "legacy_equal") == "redistribute":
            semantics["input_ids"] = tuple(self.model.tokenizer.encode(text))
        return DecisionRecord("plan", tuple(p.uid for p in plans), 0, 0.,
            state_features=self.state_features(state),
            option_features=tuple(self.plan_features(state, p) for p in plans),
            context_text=text, proposal_snapshot=self.snapshot_id,
            graph_nodes=nodes, graph_edges=edges, option_node_masks=masks,
            transition_index=getattr(state, "current_transition_index", state.version),
            graph_schema=graph_schema(self.model.config), **semantics)

    def _graph_node_texts(self, state):
        candidates = sorted(state.candidates.values(), key=lambda candidate: candidate.uid)
        serialize = lambda value: json.dumps(thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if getattr(self.model.config, "role_observation_version", "legacy_v1") == "observed_results_v2":
            texts = ["candidate version: " + serialize({"artifacts": candidate.artifacts, "runtime": candidate.runtime,
                "semantic_kind": candidate.semantic_kind, "deterministic": candidate.deterministic}) for candidate in candidates]
            for candidate in candidates:
                for task in state.scope.tasks:
                    record = state.ledger.get(candidate, task, state.scope)
                    observed = None if record is None else {"score": record.outcome.score,
                        "admissible": record.outcome.admissible, "fields": {
                            key: record.outcome.fields[key] for key in ("answer", "finish_reason", "native_score", "valid")
                            if key in record.outcome.fields}}
                    # Only executor-observed, explicitly allowed output fields;
                    # neither task.expected, private metadata nor raw receipts.
                    texts.append("candidate task observation: " + serialize({"inputs": task.inputs, "observed": observed}))
        else:
            texts = ["candidate artifacts: " + serialize(candidate.artifacts) for candidate in candidates]
            texts.extend("task inputs: " + serialize(task.inputs) for _candidate in candidates for task in state.scope.tasks)
        if graph_schema(self.model.config) == ACTUAL_GRAPH_SCHEMA:
            texts.extend(work_node_texts(state))
        return texts

    def semantic_forward_cap(self, state):
        """Preflight the full observation before admitting any paid forward.

        New graph nodes share the fixed token quota. Failed minimum allocations
        cannot execute some encodings and only then discover an unfittable graph.
        Keep the historical physical admission bound for the legacy schema.
        """
        if (graph_schema(self.model.config) != ACTUAL_GRAPH_SCHEMA and
                getattr(self.model.config, "role_observation_version", "legacy_v1") == "legacy_v1"):
            return len(state.candidates) + len(state.scope.tasks) + 1
        texts, encoded = self._bounded_node_observations(state, self.context_text(state))
        return 1 + len(set(encoded.values()) if encoded else set(texts))

    def _bounded_node_observations(self, state, context_text):
        """Bind node semantics under the registered per-node and total caps.

        Legacy records retain the original equal share of max_length. Explicit
        redistribution recycles short observed nodes' unused allocation. Cache
        residency never changes the saved observation or search-budget quota.
        """
        from .frozen_text import allocate_node_tokens, bounded_semantic_text, semantic_budget
        texts = self._graph_node_texts(state)
        encoded = {}
        unique_texts = tuple(dict.fromkeys(texts))
        if self.model.backend == "hf":
            tokenizer = self.model.tokenizer
            budget = semantic_budget(self.model.config)
            global_tokens = len(tokenizer.encode(context_text))
            available = budget["total"] - global_tokens
            if budget["mode"] == "redistribute":
                if global_tokens > budget["global"]:
                    raise ValueError("Global semantic observation exceeds its registered token cap")
                allowances = allocate_node_tokens(tuple(len(tokenizer.encode(text)) for text in unique_texts),
                    available=available, minimum=budget["minimum"], maximum=budget["maximum"])
                bounded_texts = {text: bounded_semantic_text(tokenizer, text, allowance,
                    strategy=budget["truncation"])[0] for text, allowance in zip(unique_texts, allowances)}
            else:
                allowance = available // len(unique_texts)
                if allowance < 16:
                    raise ValueError("v3 node semantics cannot fit every node in the registered query input cap")

                def bounded(value):
                    if len(tokenizer.encode(value)) <= allowance:
                        return value
                    ids = tokenizer.encode(value, add_special_tokens=False)
                    for count in range(min(len(ids), allowance), 0, -1):
                        head = (count + 1) // 2
                        tail = count // 2
                        clipped = tokenizer.decode(ids[:head], skip_special_tokens=False)
                        clipped += " … " + (tokenizer.decode(ids[-tail:], skip_special_tokens=False) if tail else "")
                        if len(tokenizer.encode(clipped)) <= allowance:
                            return clipped
                    raise ValueError("v3 text observation cannot fit its explicit node token allowance")

                bounded_texts = {text: bounded(text) for text in unique_texts}
            texts = [bounded_texts[text] for text in texts]
            encoded = {text: tuple(tokenizer.encode(text)) for text in dict.fromkeys(texts)}
            if global_tokens + sum(len(ids) for ids in set(encoded.values())) > budget["total"]:
                raise ValueError("Semantic observations exceed the registered total input token cap")
        return texts, encoded

    def _node_text_record(self, state, context_text):
        from .frozen_text import node_text_cache, semantic_budget
        texts, encoded = self._bounded_node_observations(state, context_text)
        budget = semantic_budget(self.model.config)
        cache = node_text_cache(self.model)
        snapshot = cache.identity
        vectors = []
        for text in texts:
            try:
                vector, receipt = cache.encode(text, input_ids=encoded[text]) if (
                    self.model.backend == "hf" and budget["mode"] == "redistribute") else cache.encode(text)
            except Exception as error:
                if getattr(error, "node_text_receipt", None) is not None:
                    self.physical_inferences.append(error.node_text_receipt)
                raise
            vectors.append(vector)
            if receipt is not None:
                self.physical_inferences.append(receipt)
        if cache.identity != snapshot:
            raise RuntimeError("Frozen node encoder changed during preparation")
        return {"graph_node_texts": tuple(texts), "graph_node_text_features": tuple(vectors),
                "graph_text_snapshot": snapshot,
                "graph_node_input_ids": tuple(encoded[text] for text in texts) if self.model.backend == "hf" else ()}

    def _graph_state(self, state, plans):
        """Explicit observed candidate/job graph, frozen with each choice."""
        candidates = sorted(state.candidates.values(), key=lambda candidate: candidate.uid)
        candidate_nodes = {candidate.uid: i for i, candidate in enumerate(candidates)}
        nodes, edges, jobs = [], [], {}
        n = max(1, len(state.scope.tasks))
        for candidate in candidates:
            missing = state.ledger.missing(candidate, state.scope)
            complete = not missing
            nodes.append((1., 0., float(candidate.uid == state.incumbent.uid),
                len(missing) / n, float(complete),
                state.ledger.score(candidate, state.scope) if state.ledger.admissible(candidate, state.scope) else 0.,
                float(state.ledger.admissible(candidate, state.scope)) if complete else 0.,
                float(state.children_drafted.get(candidate.uid, 0)), float(candidate.operating_penalty), 0., 0., 0.))
            if candidate.parent_id in candidate_nodes:
                parent, child = candidate_nodes[candidate.parent_id], candidate_nodes[candidate.uid]
                edges.extend(((parent, child, 0), (child, parent, 1)))
        for candidate in candidates:
            for task, weight in zip(state.scope.tasks, state.scope.weights):
                record = state.ledger.get(candidate, task, state.scope)
                index = len(nodes)
                jobs[(candidate.uid, task.uid)] = index
                nodes.append((0., 1., float(record is None), float(weight),
                    float(record is not None), float(record.outcome.score) if record else 0.,
                    float(record.outcome.admissible) if record else 0., 0., 0., 0., 0., 0.))
                parent = candidate_nodes[candidate.uid]
                edges.extend(((parent, index, 2), (index, parent, 3)))
        masks = []
        for plan in plans:
            selected = {candidate_nodes[uid] for uid in plan.parent_ids if uid in candidate_nodes}
            for job in plan.jobs:
                for candidate_id, task_id in job.keys:
                    if (candidate_id, task_id) in jobs:
                        selected.add(jobs[(candidate_id, task_id)])
                        selected.add(candidate_nodes[candidate_id])
            masks.append(tuple(float(i in selected) for i in range(len(nodes))))
        if graph_schema(self.model.config) == ACTUAL_GRAPH_SCHEMA:
            return augment_actual_jobs(state, plans, nodes, edges, masks, jobs)
        return tuple(nodes), tuple(edges), tuple(masks)

    @torch.no_grad()
    def scorer(self, state, plan):
        """Eq. (6): predict the net verified return of a state--work continuation.

            Q^pi(S_t,omega) = E_pi[
                F(S_end)-F(S_t)-lambda*C_{t:end}-mu*L_{t:end}
                | S_t,omega].
        ``state`` supplies S and ``plan`` supplies omega; the single value head
        returns Q_hat_phi(S,omega), an approximation to this conditional mean.
        The continuation law pi includes future sampled edits, plan choices,
        real checks and branch switches under frozen improver parameters and
        fixed menu rules. F is the largest observed complete admissible gain,
        so a currently weak or incomplete parent can have positive continuation
        value when subsequent work yields a verified descendant improvement.

        The value origin S_t follows payment for the current preparation.
        C_{t:end} includes selected work, later preparation and failed work;
        it excludes the already paid current preparation o_t. L_{t:end} is
        the analogous suffix elapsed time on one host clock, excluding its
        current preparation interval. lambda converts the registered expense
        unit (tokens or milliseconds) to score units, while mu converts host
        seconds to score units. mu=0 gives cost-only return accounting.
        Eq. (11)'s G_t is one observed sample of this quantity, not another
        learned value. Q^pi(S,ACCEPT)=0 because acceptance adds no suffix work.

        During Eq. (7) bundle construction, the solver supplies the same
        conservative state S_tilde with preparation resources reserved to both
        compared bundles. The proposed plan's whole joint completion effect is
        scored, including shared jobs and complementary outcomes; individual
        item predictions are not added to reconstruct this value.

        The returned ValueEstimate separates ``value`` (utility units) from
        ``expense`` (the actual charge for this scorer query in the search's
        registered budget unit). The engine charges this query to preparation.
        No prediction inserts evaluator evidence or is adopted as F or J_Gamma.
        """
        preparation_started = time.monotonic()
        record = self._record(state, (plan,))
        started = time.monotonic()
        try:
            value = float(self.model.decision_values(record)[0])
        except Exception as error:
            self._inference("scorer", time.monotonic() - started, record.context_text, input_ids=record.input_ids,
                            status="error", error=str(error))
            raise
        elapsed = time.monotonic() - started
        if getattr(self.model.config, "architecture_version", 2) == 3:
            # Search returns must not depend on whether another group member or
            # forced arm happened to populate this immutable representation
            # cache first. Physical billing still records real misses only.
            node_tokens = (sum(len(ids) for ids in set(record.graph_node_input_ids or tuple(
                tuple(self.model.tokenizer.encode(text)) for text in record.graph_node_texts)))
                if self.cost_unit == "tokens" else 0)
            expense = (max(1, math.ceil((time.monotonic() - preparation_started) * 1000))
                       if self.cost_unit == "wall-ms" else self.inference_cost(record.context_text) + node_tokens)
        else:
            expense = max(1, math.ceil(elapsed * 1000)) if self.cost_unit == "wall-ms" else self.inference_cost(record.context_text)
        self._inference("scorer", elapsed, record.context_text, input_ids=record.input_ids)
        if getattr(self.model.config, "architecture_version", 2) == 3:
            self.physical_inferences[-1]["search_budget_charge"] = expense
            self.physical_inferences[-1]["search_budget_contract"] = (
                "fixed_record_context_plus_unique_node_input_quota" if self.cost_unit == "tokens" else
                "actual_elapsed_cache_dependent" if self.cost_unit == "wall-ms" else "registered_logical_query")
        return ValueEstimate(value, expense)

    def _inference(self, role, elapsed, text, *, input_ids=(), **status):
        row = {"kind": "learned_control", "role": role,
            "elapsed_seconds": elapsed,
            "gpu_count": int(str(self.model.device).startswith("cuda")),
            "prompt_tokens": len(input_ids or self.model.tokenizer.encode(text)) if self.model.backend == "hf" else 0,
            "completion_tokens": 0, **status,
            **({"input_token_ids": list(input_ids)} if input_ids else {})}
        if self.model.backend == "hf" and getattr(self.model.config, "frozen_global_cache", False):
            encoding = getattr(self.model, "_last_global_encoding", None)
            row.update(head_forward_count=int(getattr(self.model, "_last_role_head_started", False)),
                       encoder_forward_count=0, encoder_cache_hit=False,
                       context_input_token_ids=list(input_ids or self.model.tokenizer.encode(text)))
            if encoding is not None:
                if encoding["context_input_token_ids"] != row["context_input_token_ids"]:
                    raise ValueError("Global encoder receipt differs from the executed role observation")
                row.update(encoding)
            # A head invocation still exists on a cache hit; only its frozen
            # transformer token bill is zero. Exact conditioning IDs stay logged.
            if not row["encoder_forward_count"]:
                row["prompt_tokens"] = 0
                row.pop("input_token_ids", None)
            else:
                row["input_token_ids"] = row["context_input_token_ids"]
        self.physical_inferences.append(row)

    @torch.no_grad()
    def policy_menu(self, state, plans):
        self.initial_budget = state.budget.total
        self.workers = state.workers
        self.contract_id = state.scope.uid
        record = self._record(state, plans)
        started = time.monotonic()
        try:
            logits = self.model.decision_logits(record)
        except Exception as error:
            self._inference("policy", time.monotonic() - started, record.context_text, input_ids=record.input_ids,
                            status="error", error=str(error))
            raise
        self._inference("policy", time.monotonic() - started, record.context_text, input_ids=record.input_ids)
        for i, plan in enumerate(plans):
            self.captured[(state.version, "plan", plan.uid)] = replace(record,
                support=(plan.uid,), option_features=(record.option_features[i],),
                option_node_masks=(record.option_node_masks[i],))
        return logits.cpu().tolist()

    def policy(self, state, plan):
        """Compatibility hook; engines should use the single-call menu hook."""
        return self.policy_menu(state, (plan,))[0]

    def convert(self, result, *, lineage_id):
        decisions = []
        for raw in result.decisions:
            snapshots = [self.captured.get((raw.state_version, raw.kind, option)) for option in raw.support]
            if any(snapshot is None for snapshot in snapshots):
                raise ValueError("Missing paid-state policy snapshot; cannot reconstruct behavior inputs")
            first = snapshots[0]
            if any((snapshot.state_features, snapshot.context_text) != (first.state_features, first.context_text)
                   for snapshot in snapshots):
                raise ValueError("A menu's policy options must share exactly one paid state")
            selected = raw.support.index(raw.selected_id)
            decisions.append(DecisionRecord(raw.kind, raw.support, selected,
                math.log(raw.probabilities[selected]) if raw.stochastic else None,
                state_features=first.state_features,
                option_features=tuple(snapshot.option_features[0] for snapshot in snapshots),
                transition_index=raw.transition_index, temperature=raw.temperature, exploration=raw.exploration,
                proposal_snapshot=raw.proposal_snapshot, context_text=first.context_text,
                input_ids=first.input_ids,
                graph_nodes=first.graph_nodes, graph_edges=first.graph_edges,
                graph_node_texts=first.graph_node_texts, graph_node_text_features=first.graph_node_text_features,
                graph_text_snapshot=first.graph_text_snapshot,
                graph_node_input_ids=first.graph_node_input_ids,
                graph_schema=first.graph_schema,
                option_node_masks=tuple(snapshot.option_node_masks[0] for snapshot in snapshots)
                    if first.option_node_masks else (),
                behavior_probabilities=raw.probabilities, deterministic=not raw.stochastic))
        decisions.extend(self.token_decisions)
        decisions.sort(key=lambda decision: decision.transition_index)
        transitions = tuple(TransitionRecord(t.f_before, t.f_after, t.cost,
            elapsed_seconds=getattr(t, "elapsed_seconds", 0.)) for t in result.transitions)
        trajectory = SearchTrajectory(result.group_id, result.initial_context_id, result.search_id, result.seed,
            transitions, tuple(decisions), result.initial_f, result.final_f, lineage_id=lineage_id,
            collection_snapshot=self.snapshot_id, contract_id=self.contract_id,
            initial_budget=self.initial_budget, workers=self.workers)
        if not math.isclose(trajectory.total_cost, result.total_cost):
            raise ValueError("VATS and learning cost ledgers disagree")
        return trajectory


class ReferenceVATSIntegerDrafter:
    """Collect executed finite-action CPU searches and observed return targets."""
    def __init__(self, bridge):
        self.bridge = bridge

    @torch.no_grad()
    def draft(self, parent, state, cap, rng):
        if cap < 4:
            return DraftResult(None, cap, status="interrupted", error="Draft cap below the four-option reference call")
        choices = ("preserve", "repair-a", "repair-b", "repair-both")
        support = tuple(f"edit:{parent.uid}:{choice}" for choice in choices)
        options = tuple(_pad(tuple(float(i == j) for j in range(4)), self.bridge.option_dim) for i in range(4))
        logits = [self.bridge._score(state, "edit", uid, features) for uid, features in zip(support, options)]
        probabilities = mixture_probabilities(torch.tensor(logits), exploration=.1).tolist()
        selected = rng.choices(range(4), weights=probabilities)[0]
        routes = thaw(parent.artifacts["routes"])
        if selected in (1, 3):
            routes["a"] = ["add", ["x"], ["const", 1]]
        if selected in (2, 3):
            routes["b"] = ["mul", ["x"], ["const", 2]]
        child = Candidate(f"{parent.name}-{choices[selected]}-{state.version}", {"routes": routes},
            parent.runtime, "pure_integer", True, parent_id=parent.uid)
        transition_index = getattr(state, "current_transition_index", state.version)
        trace = TraceDecision("edit", support, support[selected], tuple(probabilities), tuple(logits),
            state.version, transition_index, 1., .1, self.bridge.snapshot_id,
            state_features=self.bridge.state_features(state), option_features=options, value_source="reference-draft-policy")
        return DraftResult(child, 4, edit_decision=trace)


def collect_reference_vats(model, round_index, config):
    """Run actual isolated PDF VATS searches for CPU integration and training."""
    from specrsi.core.vats import VATSConfig, VATSEngine

    if model.backend != "tiny":
        raise ValueError("Reference VATS is a finite-action tiny-model collector; use a dataset collector for HF")
    settings = config.get("collection", {})
    from specrsi.paper_profile import algorithm_name
    algorithm_name({**settings, "algorithm": "vats_pdf"})
    vats_options = dict(settings.get("vats", {}))
    allowed_options = set(VATSConfig.__dataclass_fields__) - set(SearchConfig.__dataclass_fields__)
    unknown = set(vats_options) - allowed_options
    if unknown:
        raise ValueError(f"collection.vats may only override PDF menu settings, not execution fields: {sorted(unknown)}")
    group_size = settings.get("group_size", 4)
    if group_size < 2:
        raise ValueError("GVO needs group_size >= 2")
    rng = random.Random(config.get("training", {}).get("seed", 0) + round_index * 1000000)
    model.eval()
    from .continuations import ForcedCollectionConfig, FrozenRoleGuard, collect_paid_continuations, role_snapshot_id
    teacher = config.get("stage") == "warmup" and settings.get("demonstration_policy") is not None
    from .demonstrations import demonstration_bridge_kwargs
    bridge_class = VATSSearchBridge
    if teacher:
        from .demonstrations import DemonstrationVATSBridge
        bridge_class = DemonstrationVATSBridge
    forced_config = ForcedCollectionConfig.from_settings(settings, teacher=teacher)
    guard = FrozenRoleGuard(model)
    collection_snapshot = role_snapshot_id(model, f"pdf-round-{round_index}")
    groups, raw_results, forced_trajectories, pairs, forced_raw, edits, forced_acquisition, edit_evidence = [], [], [], [], [], [], [], []
    plan_evidence, teacher_choices = [], []
    demonstration_kwargs = demonstration_bridge_kwargs(settings) if teacher else {}
    copy_elapsed = 0.
    repetitions = forced_config.repetitions
    initialization_cost = 0
    for context_index in range(settings.get("contexts", 2)):
        offset = context_index * 5
        tasks = tuple(Task(f"{route}-{x}", {"route": route, "x": x}, x + 1 if route == "a" else x * 2)
            for route in ("a", "b") for x in (offset + 1, offset + 2))
        scope = EvaluationScope(f"pdf-reference-train-{context_index}", tasks,
            "pure-int-v1", "integer-equality-v1", split="train")
        incumbent = Candidate("initial", {"routes": {"a": ["x"], "b": ["x"]}},
            {"contract": "pure-int-v1"}, "pure_integer", True)
        group = []
        for member in range(group_size):
            guard.check()
            bridge = bridge_class(model, snapshot_id=collection_snapshot, **demonstration_kwargs)
            ledger, executor = Ledger(ConservativeChecker()), PureIntegerExecutor()
            for task in tasks:
                execution = executor.run(incumbent, task, scope, 5)
                initialization_cost += execution.actual_cost
                ledger.admit_direct(incumbent, task, scope, execution.outcome, execution.receipt)
            search_kwargs = {key: value for key, value in settings.items() if key in VATSConfig.__dataclass_fields__}
            search_kwargs.update(vats_options)
            search_kwargs.update(execution_cap=5, draft_cap=4, mode="deploy" if teacher else "collect", seed=rng.randrange(2**63 - 1),
                all_u=True, max_sources=0, check_cap=0, lookup_cost=0,
                group_id=f"pdf-round-{round_index}-context-{context_index}")
            search_kwargs.setdefault("max_actions", 12)
            search_kwargs.setdefault("max_candidates", 6)
            search_kwargs.setdefault("max_queries", 12)
            search_kwargs.setdefault("scorer_snapshot", bridge.snapshot_id)
            search_kwargs.setdefault("policy_snapshot", bridge.snapshot_id)
            search = VATSConfig(**search_kwargs)
            engine = VATSEngine(scope, incumbent, ledger, BudgetLedger(settings.get("budget", 150)), executor,
                ReferenceVATSIntegerDrafter(bridge), search, scorer=bridge.scorer, policy_menu=bridge.policy_menu,
                capture_paid_states=bool(repetitions and member == 0))
            bridge.initial_budget, bridge.workers, bridge.contract_id = engine.state.budget.total, search.workers, scope.uid
            result = engine.run()
            group.append(bridge.convert(result, lineage_id=f"pdf-reference-lineage-{context_index}"))
            raw_results.append(result.to_dict())
            edits.extend(getattr(bridge, "executed_edits", ()))
            edit_evidence.extend(getattr(bridge, "edit_evidence", ()))
            plan_evidence.extend(getattr(bridge, "plan_evidence", ()))
            teacher_choices.extend(getattr(bridge, "teacher_choice_evidence", ()))
            guard.check()
            if repetitions and member == 0 and engine.paid_snapshots:
                continuation = collect_paid_continuations(engine.paid_snapshots, model=model,
                    snapshot_id=bridge.snapshot_id,
                    bridge_factory=lambda: bridge_class(model, snapshot_id=bridge.snapshot_id, **demonstration_kwargs),
                    executor_factory=PureIntegerExecutor, drafter_factory=lambda b, e: ReferenceVATSIntegerDrafter(b),
                    checker_factory=ConservativeChecker, environment_contract="pure_integer", config=forced_config,
                    seed=rng.randrange(2**63 - 1), lineage_id=f"pdf-reference-lineage-{context_index}",
                    training=config.get("training", {}), guard=guard,
                    acquisition_path=(Path(config["output"]) / f"forced_acquisition_round_{round_index}.jsonl") if config.get("output") else None)
                forced_trajectories.extend(continuation["forced_trajectories"])
                pairs.extend(continuation["pairs"])
                forced_raw.extend(continuation["metadata"]["raw_search_results"])
                copy_elapsed += continuation["metadata"]["forced_copy_elapsed_seconds"]
                forced_acquisition.append(continuation["metadata"])
        if not teacher:
            validate_group(group)
        groups.append(group)
    return {"groups": groups, "audits": [], "forced_trajectories": forced_trajectories, "pairs": pairs, "edits": edits,
        "metadata": {"reference_tasks_only": True,
        "algorithm": "vats_pdf", "cost_unit": "logical-work",
        "initial_record_execution_cost": initialization_cost,
        "search_cost": sum(t.total_cost for group in groups for t in group), "raw_search_results": raw_results,
        "forced_search_cost": sum(t.total_cost for t in forced_trajectories),
        "forced_copy_elapsed_seconds": copy_elapsed, "forced_repetitions": repetitions,
        "forced_raw_search_results": forced_raw, "forced_acquisition": forced_acquisition, "edit_evidence": edit_evidence,
        "sft_plan_evidence": plan_evidence, "demonstration_quality": settings.get("demonstration_quality"),
        **({"demonstration_policy": settings["demonstration_policy"],
            "demonstration_stopping": settings.get("demonstration_stopping"),
            "teacher_choice_evidence": teacher_choices} if teacher else {})}}
