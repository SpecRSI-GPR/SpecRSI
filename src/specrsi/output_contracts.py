"""Opt-in, versioned task output interfaces for immutable candidate execution.

Contracts change the actual model input, never a returned answer or its native
score. The version belongs in candidate.runtime.generation, so it is included
in the immutable candidate and registered execution identities. Only public
task inputs are accepted here; references and observed scores are not needed.
"""
from __future__ import annotations

from collections.abc import Mapping


HOTPOT_SHORT_ANSWER_V1 = """Fixed final-answer interface:
Answer exactly what the question asks, using only the shortest complete answer.
Include qualifiers needed to identify the answer unambiguously. Return multiple
items only when the question asks for multiple items. Do not add intermediate
entities, supporting facts, an explanation, an answer label, quotation marks,
or alternative answers. For a yes/no question, return only yes or no.
These final-answer requirements apply to every question."""


def output_contract_instruction(generation: Mapping) -> str:
    """Resolve an explicitly registered version without changing legacy inputs."""
    if "output_contract" not in generation:
        return ""
    if generation["output_contract"] != "hotpot_short_answer_v1":
        raise ValueError("Unknown output_contract; expected hotpot_short_answer_v1")
    return HOTPOT_SHORT_ANSWER_V1


def draft_interface_guidance(generation: Mapping) -> str:
    """The same public task description is available to every prompt proposer."""
    contract = output_contract_instruction(generation)
    if not contract:
        return ""
    return ("The following task output interface is fixed and appended by the executor. "
        "Improve the general solving strategy within it; do not contradict or duplicate it. "
        "Observed failures are examples, not new requirements for every question. "
        "Do not hard-code their particular entities, answers, or question topics into the instruction.\n"
        + contract + "\n")


def generation_messages(instruction: str, task_prompt: str, generation: Mapping,
                        *, dataset: str | None = None) -> list[dict[str, str]]:
    contract = output_contract_instruction(generation)
    if contract:
        if dataset != "hotpotqa":
            raise ValueError("hotpot_short_answer_v1 requires the HotpotQA task interface")
        instruction += "\n\n" + contract
    return [{"role": "system", "content": instruction},
            {"role": "user", "content": task_prompt}]
