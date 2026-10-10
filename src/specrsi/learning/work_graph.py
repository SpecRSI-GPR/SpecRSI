"""Explicit graph contracts; legacy candidate/item graphs keep their old layout.

actual_jobs_v2 adds physical jobs and shared setups, without new task outcomes.
Relations 0--3 are unchanged. Added directions are job->item (4), item->job
(5), prerequisite->dependent (6), dependent->prerequisite (7), conflict (8),
setup->job (9), and job->setup (10). New roles require newly trained heads.
"""
from __future__ import annotations

from dataclasses import asdict
import json


LEGACY_GRAPH_SCHEMA = "candidate_items_v1"
ACTUAL_GRAPH_SCHEMA = "actual_jobs_v2"


def graph_schema(config):
    read = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    value = read("graph_schema", LEGACY_GRAPH_SCHEMA)
    if value not in {LEGACY_GRAPH_SCHEMA, ACTUAL_GRAPH_SCHEMA}:
        raise ValueError("Unknown role graph_schema")
    if value == ACTUAL_GRAPH_SCHEMA and read("architecture_version", 2) != 3:
        raise ValueError("actual_jobs_v2 requires explicit architecture_version=3")
    return value


def relation_count(schema):
    if schema not in {LEGACY_GRAPH_SCHEMA, ACTUAL_GRAPH_SCHEMA}:
        raise ValueError("Unknown role graph_schema")
    return 4 if schema == LEGACY_GRAPH_SCHEMA else 11


def validate_role_graph(model, decision):
    schema = graph_schema(model.config)
    if decision.graph_schema != schema:
        raise ValueError("Recorded graph_schema differs from the role checkpoint")
    if any(len(getattr(model, name).messages) != relation_count(schema)
           for name in ("value_head", "policy_head")):
        raise ValueError("graph_schema requires separately trained matching role heads")


def catalog_parts(state):
    from specrsi.planning.completion import CompletionProblem
    problem = getattr(state, "job_catalog", None)
    if not isinstance(problem, CompletionProblem):
        raise ValueError("actual_jobs_v2 requires the observed immutable verification job catalog")
    return problem, tuple(sorted(problem.jobs, key=lambda job: job.uid)), tuple(sorted({job.group for job in problem.jobs}))


def work_node_texts(state):
    problem, jobs, groups = catalog_parts(state)
    return tuple("verification job: " + json.dumps(asdict(job), sort_keys=True, separators=(",", ":")) for job in jobs) + tuple(
        "shared verification setup: " + json.dumps({"group": group, "cap": problem.setup_costs[group]},
                                                   sort_keys=True, separators=(",", ":")) for group in groups)


def augment_actual_jobs(state, plans, nodes, edges, masks, item_nodes):
    problem, jobs, groups = catalog_parts(state)
    nodes, edges = list(nodes), list(edges)
    job_nodes = {job.uid: len(nodes) + i for i, job in enumerate(jobs)}
    setup_nodes = {group: len(nodes) + len(jobs) + i for i, group in enumerate(groups)}
    pairs = max(1, len(state.candidates) * len(state.scope.tasks))
    for job in jobs:
        dependencies = sum(a == job.uid for a, _ in problem.dependencies)
        conflicts = sum(job.uid in pair for pair in problem.conflicts)
        # Type at feature 2: result job=1, operational prerequisite=2, setup=3.
        nodes.append((0., 0., 1. if job.kind == "result" else 2.,
            job.cap / max(1, state.budget.total), job.attempts / max(1, state.initial_attempts),
            len(job.keys) / pairs, dependencies / max(1, len(jobs)), conflicts / max(1, len(jobs)),
            0., 0., 0., 0.))
        for key in job.keys:
            if key not in item_nodes:
                raise ValueError("Observed job catalog contains an unknown candidate/item")
            item = item_nodes[key]
            edges.extend(((job_nodes[job.uid], item, 4), (item, job_nodes[job.uid], 5)))
        setup = setup_nodes[job.group]
        edges.extend(((setup, job_nodes[job.uid], 9), (job_nodes[job.uid], setup, 10)))
    for group in groups:
        nodes.append((0., 0., 3., problem.setup_costs[group] / max(1, state.budget.total),
                      0., 0., 0., 0., 0., 0., 0., 0.))
    for dependent, prerequisite in problem.dependencies:
        edges.extend(((job_nodes[prerequisite], job_nodes[dependent], 6),
                      (job_nodes[dependent], job_nodes[prerequisite], 7)))
    for left, right in problem.conflicts:
        edges.extend(((job_nodes[left], job_nodes[right], 8), (job_nodes[right], job_nodes[left], 8)))
    expanded = []
    for plan, mask in zip(plans, masks):
        if any(job.uid not in job_nodes or job != problem.by_id[job.uid] for job in plan.jobs):
            raise ValueError("Proposed work differs from the observed immutable job catalog")
        selected = {job_nodes[job.uid] for job in plan.jobs} | {setup_nodes[job.group] for job in plan.jobs}
        expanded.append((*mask, *(float(i in selected) for i in range(len(mask), len(nodes)))))
    return tuple(nodes), tuple(edges), tuple(expanded)
