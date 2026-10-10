"""Frozen, bounded language-model DAGs for workflow search.

Nodes contain prompts and explicit dependencies. The evaluator executes the entire
registered graph; an interrupted prefix is never a complete task outcome.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

from .models.client import sampling_options


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    instruction: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Workflow:
    nodes: tuple[WorkflowNode, ...]
    output_node: str

    @classmethod
    def from_dict(cls, raw: Mapping):
        if not isinstance(raw, Mapping):
            raise ValueError("Workflow must be an object")
        if set(raw) != {"nodes", "output_node"}:
            raise ValueError("Workflow requires only nodes and output_node")
        nodes = []
        if not isinstance(raw["nodes"], (list, tuple)) or not isinstance(raw["output_node"], str):
            raise ValueError("Workflow nodes must be a list and output_node a string")
        for item in raw["nodes"]:
            if not isinstance(item, Mapping) or not {"id", "instruction"} <= set(item):
                raise ValueError("Every workflow node requires id and instruction")
            if set(item) - {"id", "instruction", "dependencies"}:
                raise ValueError("Unknown workflow node fields")
            if not isinstance(item["id"], str) or not isinstance(item["instruction"], str):
                raise ValueError("Node id and instruction must be strings")
            dependencies = item.get("dependencies", ())
            if not isinstance(dependencies, (list, tuple)) or any(not isinstance(k, str) for k in dependencies):
                raise ValueError("Workflow dependencies must be a list of node IDs")
            nodes.append(WorkflowNode(item["id"], item["instruction"], tuple(dependencies)))
        workflow = cls(tuple(nodes), raw["output_node"])
        workflow.ordered()
        return workflow

    def ordered(self):
        if not 1 <= len(self.nodes) <= 8:
            raise ValueError("A workflow must contain 1 to 8 nodes")
        nodes = {node.id: node for node in self.nodes}
        if len(nodes) != len(self.nodes) or self.output_node not in nodes:
            raise ValueError("Duplicate node or missing output node")
        for node in self.nodes:
            if not node.id or not node.instruction or len(node.instruction) > 12000:
                raise ValueError("Empty or oversized workflow node")
            if len(set(node.dependencies)) != len(node.dependencies) or not set(node.dependencies) <= set(nodes):
                raise ValueError("Unknown or duplicate dependency")
        done, ordered = set(), []
        while len(done) < len(nodes):
            ready = sorted(key for key, node in nodes.items() if key not in done and set(node.dependencies) <= done)
            if not ready:
                raise ValueError("Workflow contains a dependency cycle")
            for key in ready:
                done.add(key)
                ordered.append(nodes[key])
        # Every paid node must contribute to the declared output.
        ancestors = {self.output_node}
        pending = [self.output_node]
        while pending:
            for key in nodes[pending.pop()].dependencies:
                if key not in ancestors:
                    ancestors.add(key)
                    pending.append(key)
        if ancestors != set(nodes):
            raise ValueError("Workflow has nodes disconnected from its output")
        return tuple(ordered)


def execute_workflow(workflow: Workflow, task_prompt: str, executor, *, cap: int, seed: int):
    """Return answer, actual token cost and all physical call records, or interruption."""
    if type(cap) is not int or cap < 0:
        raise ValueError("Workflow token cap must be a nonnegative integer")
    outputs, records, used = {}, [], 0
    for node in workflow.ordered():
        context = task_prompt
        for dependency in node.dependencies:
            context += f"\n\nPrevious result from {dependency}:\n{outputs[dependency]}"
        messages = [{"role": "system", "content": node.instruction}, {"role": "user", "content": context}]
        prompt_tokens = executor.prompt_tokens(messages)
        max_tokens = executor.config.get("max_tokens", 512)
        # A budget interruption must not silently alter the fixed node-generation
        # contract. Only run nodes whose complete configured cap remains available.
        if max_tokens < 1 or prompt_tokens + max_tokens > min(cap - used, 32768):
            return None, used, records
        reply = executor.client.generate(messages, max_tokens=max_tokens,
            temperature=executor.config.get("temperature", .7), top_p=executor.config.get("top_p", .8), seed=(seed + len(records)) % 2**32,
            enable_thinking=executor.config.get("enable_thinking", False),
            max_seconds=executor.config.get("max_seconds", 120), total_token_cap=cap - used,
            **sampling_options(executor.config))
        used += reply.prompt_tokens + reply.completion_tokens
        records.append((node.id, reply))
        if used > cap or reply.prompt_tokens != prompt_tokens:
            # Preserve measured records for the enclosing trusted executor, which
            # will report the violated cap and halt its cost ledger.
            return None, used, records
        if reply.finish_reason == "timeout":
            return None, used, records
        outputs[node.id] = reply.text
    return outputs[workflow.output_node], used, records
