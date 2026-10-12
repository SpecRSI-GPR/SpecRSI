"""Finite, reproducible QA/math candidates with development-only demonstrations.

These are an optional expansion of the candidate space, not a change to the
paper's search algorithm or its scorer. Candidate construction makes no model
calls and never uses protected examples' references. Pass *all* tuning,
confirmation, and audit inputs as ``protected_examples`` before constructing a
pool. A token counter and every evaluation prompt are required when requesting
an executable token-cap guarantee.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import re
from typing import Any

from .benchmarks.base import BenchmarkExample
from .core.models import Candidate, FrozenMap, digest, thaw


_FORMAT = (
    "Return only the shortest complete answer to the question: an entity name, "
    "a number, a date, or yes/no as appropriate. Preserve necessary name qualifiers. "
    "Do not include reasoning, citations, an answer label, or a full sentence."
)


@dataclass(frozen=True)
class CandidateRecipe:
    id: str
    instruction: str
    demonstration_count: int = 0
    selection: str = "balanced"


RECIPES = (
    CandidateRecipe("direct", "Answer the question using the supplied context. " + _FORMAT),
    CandidateRecipe(
        "evidence_bridge",
        "Solve the question from the supplied passages. First identify exactly which "
        "attribute or relation is requested. For a multi-hop question, resolve the "
        "intermediate entity in one passage, then use that entity to locate the "
        "answer in another passage. Track aliases and dates carefully; distinguish "
        "the intermediate entity from the final answer. For comparisons, establish "
        "the requested attribute of both entities before comparing them. " + _FORMAT,
    ),
    CandidateRecipe(
        "comparison",
        "Read the question before considering the passages. If it asks which of two "
        "entities has a property, extract that same property for each entity and "
        "compare like with like: birth date with birth date, founding date with "
        "founding date, and occupation with occupation. Earlier dates mean older "
        "entities. If it asks whether both share a property, answer yes only when "
        "the passages establish the property for both. For other questions, follow "
        "the relationship between the named entity and the requested fact. " + _FORMAT,
    ),
    CandidateRecipe(
        "self_check",
        "Use the supplied context to find a supported answer. Before responding, "
        "check that the answer has the type requested by the question and belongs "
        "to the correct entity, time period, and relationship. Distractor passages "
        "may mention similar names or the same words; prefer the passage that "
        "satisfies all the question's constraints. Check each link of a multi-hop "
        "inference and both sides of a comparison. Resolve contradictions against "
        "the explicit evidence, then remove any unnecessary words. " + _FORMAT,
    ),
    CandidateRecipe(
        "evidence_2shot",
        "Answer using the supplied passages. Resolve intermediate entities for "
        "multi-hop questions and compare the same attribute for comparison questions. "
        "The demonstrations show relevant passages; the actual question may also "
        "include distracting passages. " + _FORMAT,
        demonstration_count=2,
    ),
    CandidateRecipe(
        "balanced_4shot",
        "Find the exact fact requested in the supplied context. Follow entity links "
        "across passages when needed. For comparisons, determine each entity's "
        "attribute independently before choosing an entity or answering yes/no. "
        "Use the examples as a guide to answer granularity. The examples include "
        "relevant passages; ignore unrelated passages in the actual question. " + _FORMAT,
        demonstration_count=4,
    ),
)

_MATH_DATASETS = frozenset({"aime", "aime24", "aime25", "aime_2025_2026", "math", "livebench_math"})


def _recipes_for(dataset: str) -> tuple[CandidateRecipe, ...]:
    if dataset == "hotpotqa":
        return RECIPES
    if dataset not in _MATH_DATASETS:
        raise ValueError(f"No registered performance candidate recipes for {dataset}")
    aime = dataset.startswith("aime")
    final_format = (
        r"End with exactly one final answer in \boxed{N}, where N is an integer from 0 to 999. "
        "Do not reduce modulo 1000 unless the problem explicitly requests it."
        if aime else r"End with exactly one final answer in \boxed{...}, using the exact "
        "expression, value, or set requested. Avoid decimal approximations unless requested."
    )
    return (
        CandidateRecipe("direct", "Solve the mathematical problem accurately. " +
                        ("Return only the final integer from 0 to 999." if aime else
                         r"Return only the final exact answer in \boxed{...}.")),
        CandidateRecipe(
            "careful_derivation",
            "Solve the mathematical problem step by step with a concise, rigorous derivation. "
            "Translate all constraints before calculating. Choose a representation that makes "
            "the structure explicit, such as equations, a counting argument, or a geometric "
            "invariant. Keep arithmetic exact, justify each case, and check for omitted or "
            "double-counted solutions. Allocate enough space to complete the computation. " + final_format,
        ),
        CandidateRecipe(
            "verify",
            "Derive a solution to the mathematical problem, then verify it before committing "
            "to the answer. Substitute proposed roots into the original conditions, check "
            "domains and extraneous roots, and confirm boundary cases and counting factors. "
            "Recompute the final arithmetic independently. Repair any failed check rather "
            "than reporting the first plausible result. Give a concise derivation and the "
            "verified result. " + final_format,
        ),
        CandidateRecipe(
            "independent_alternative",
            "Solve the mathematical problem using a systematic method. Once you have a "
            "candidate result, check it from a different perspective: an algebraic result "
            "with substitution or an invariant, a counting result with an independent "
            "enumeration or complement count, and a geometric result with coordinates or "
            "a second theorem when practical. If the checks disagree, identify and fix the "
            "specific assumption or arithmetic error. Keep the explanation focused so the "
            "final computation is completed. " + final_format,
        ),
    )


@dataclass(frozen=True)
class PerformanceCandidate:
    candidate: Candidate
    recipe_id: str
    provenance: FrozenMap

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", FrozenMap(self.provenance))

    def manifest(self) -> dict[str, Any]:
        return {"candidate": self.candidate.manifest(), "candidate_id": self.candidate.uid,
                "recipe_id": self.recipe_id, "provenance": thaw(self.provenance)}


def recipe_definitions(dataset: str = "hotpotqa") -> list[dict[str, Any]]:
    """Return detached JSON-ready definitions of the complete finite recipe set."""
    return [{"id": r.id, "instruction": r.instruction,
             "demonstration_count": r.demonstration_count, "selection": r.selection}
            for r in _recipes_for(dataset)]


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).casefold().encode()).hexdigest()


def _question(example: BenchmarkExample) -> str:
    value = example.metadata.get("question")
    if isinstance(value, str) and value.strip():
        return value.strip()
    match = re.search(r"(?:^|\n)Question:\s*(.*?)\s*\nAnswer:\s*$", example.prompt, re.S)
    return match.group(1).strip() if match else ""


def example_identity_keys(example: BenchmarkExample) -> frozenset[tuple[str, str]]:
    """Label-free input, question, explicit content, and root-lineage identities."""
    keys = {("id", f"{example.dataset}:{example.id}"), ("input", example.content_hash)}
    question = _question(example)
    if question:
        keys.add(("question", _text_hash(question)))
    for field in ("lineage_id", "root_lineage_id", "root_id", "split_group"):
        value = example.metadata.get(field)
        if value is not None and value != "":
            keys.add(("lineage", str(value)))
    for field in ("content_id", "source_example_id"):
        value = example.metadata.get(field)
        if value is not None and value != "":
            keys.add((field, str(value)))
    return frozenset(keys)


def _compact_demonstration(example: BenchmarkExample, max_chars: int) -> tuple[str, dict] | None:
    """Select complete supporting paragraphs using supervised development labels."""
    reference = example.reference
    if not isinstance(reference, Mapping) or not isinstance(reference.get("answer"), str):
        return None
    answer, question = reference["answer"].strip(), _question(example)
    if not answer or not question:
        return None
    facts = reference.get("supporting_facts", ())
    if not isinstance(facts, (list, tuple)):
        return None
    titles = {fact[0] for fact in facts if isinstance(fact, (list, tuple))
              and len(fact) >= 2 and isinstance(fact[0], str)}
    if not titles:
        return None
    context = example.prompt.rsplit("\nQuestion:", 1)[0].strip()
    paragraphs = []
    found_titles = set()
    for block in context.split("\n\n"):
        title, separator, body = block.partition(":\n")
        if separator and title in titles:
            paragraphs.append(title + ":\n" + body.strip())
            found_titles.add(title)
    if found_titles != titles:
        return None
    text = "\n\n".join(paragraphs) + f"\n\nQuestion: {question}\nAnswer: {answer}"
    if len(text) > max_chars:
        return None  # Never truncate away the evidence or answer.
    kind = ("yes_no" if answer.casefold() in {"yes", "no"} else
            "comparison" if re.search(r"\b(older|younger|earlier|later|first|more|less|longer|shorter|both)\b",
                                       question, re.I) else "bridge")
    return text, {
        "dataset": example.dataset, "example_id": example.id, "split": example.split,
        "content_hash": example.content_hash, "question_hash": _text_hash(question),
        "identity_keys": sorted(example_identity_keys(example)), "kind": kind,
        "source_mode": "supervised_development", "reference_sha256": digest(reference),
        "demonstration_sha256": digest(text), "supporting_titles": sorted(titles),
        "compaction": "complete_supporting_paragraphs_in_source_order",
    }


def _balanced_demos(demos: Sequence[tuple[str, dict]], count: int) -> tuple[tuple[str, dict], ...]:
    groups = {kind: [d for d in demos if d[1]["kind"] == kind]
              for kind in ("bridge", "comparison", "yes_no")}
    selected = []
    while len(selected) < count and any(groups.values()):
        for group in groups.values():
            if group and len(selected) < count:
                selected.append(group.pop(0))
    return tuple(selected)


def build_performance_candidates(
    development: Iterable[BenchmarkExample], *, runtime: Mapping[str, Any],
    protected_examples: Iterable[BenchmarkExample] = (), seed: int = 0, dataset: str | None = None,
    recipe_ids: Sequence[str] | None = None, max_system_chars: int = 12000,
    max_demo_chars: int = 2200, development_limit: int = 256,
    token_counter: Callable[..., int] | None = None, execution_cap: int | None = None,
    task_prompts: Iterable[str] = (),
) -> tuple[PerformanceCandidate, ...]:
    """Build frozen instructions and supervised few-shot candidates.

    Only ``development`` split labels may supply demonstrations (HotpotQA only;
    math recipes are instruction-only). Protected input
    overlaps are excluded, including reworded descendants with shared lineage.
    Demonstrations are selected by a fixed seeded ordering and question type,
    without inspecting tuning/audit outcomes. Few-shot recipes without enough
    eligible examples, or exceeding the fixed context cap, are omitted. Invalid
    zero-shot contracts raise rather than silently reducing the answer budget.

    ``token_counter(messages, enable_thinking=bool)`` is compatible with
    ``QwenExecutor.prompt_tokens``. Supply all evaluation inputs in
    ``task_prompts`` to guarantee that each emitted candidate fits every task.
    The caller must keep protected references out of model-facing prompts.
    """
    if type(seed) is not int or any(type(v) is not int or v < 1
                                  for v in (max_system_chars, max_demo_chars, development_limit)):
        raise ValueError("Seed must be an integer and candidate bounds must be positive integers")
    source = list(development)
    if dataset is None:
        source_datasets = {example.dataset for example in source}
        if len(source_datasets) > 1:
            raise ValueError("Performance candidate development pool must use one dataset")
        dataset = next(iter(source_datasets), "hotpotqa")
    definitions = {recipe.id: recipe for recipe in _recipes_for(dataset)}
    ids = tuple(definitions) if recipe_ids is None else tuple(recipe_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) - definitions.keys():
        raise ValueError("Recipe IDs must be nonempty, unique, and registered")
    generation = runtime.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("Candidate runtime requires a generation contract")
    output_cap = generation.get("max_tokens", 512)
    if type(output_cap) is not int or output_cap < 1:
        raise ValueError("generation.max_tokens must be a positive integer")
    prompts = tuple(task_prompts)
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("Task prompts must be nonempty strings")
    if execution_cap is not None:
        if type(execution_cap) is not int or execution_cap < 1 or token_counter is None or not prompts:
            raise ValueError("Token-cap validation requires a positive cap, token counter, and all task prompts")
    protected_keys = set()
    for example in protected_examples:
        protected_keys.update(example_identity_keys(example))
    if any(e.split != "development" or e.dataset != dataset for e in source):
        raise ValueError("Performance candidate sources must be development examples from the selected dataset only")
    source.sort(key=lambda e: (digest([seed, e.content_hash, e.id]), e.id))
    eligible, excluded, seen = [], [], set()
    for example in source:
        keys = example_identity_keys(example)
        reason = "protected_overlap" if keys & protected_keys else "duplicate" if keys & seen else None
        if reason:
            excluded.append({"id": example.id, "reason": reason})
            continue
        seen.update(keys)
        eligible.append(example)
    bounded = eligible[:development_limit]
    needs_demos = any(definitions[recipe_id].demonstration_count for recipe_id in ids)
    demos = ([demo for e in bounded if (demo := _compact_demonstration(e, max_demo_chars)) is not None]
             if needs_demos else [])
    built = []
    for recipe_id in ids:
        recipe = definitions[recipe_id]
        selected = _balanced_demos(demos, recipe.demonstration_count)
        if len(selected) != recipe.demonstration_count:
            continue
        instruction = recipe.instruction
        if selected:
            instruction += "\n\n" + "\n\n".join(
                f"Example {index}:\n{text}" for index, (text, _) in enumerate(selected, 1))
            instruction += "\n\nNow answer the user's question using the supplied context. " + _FORMAT
        required_tokens = []
        if execution_cap is not None:
            for prompt in prompts:
                tokens = token_counter(
                    [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}],
                    enable_thinking=generation.get("enable_thinking", False))
                if type(tokens) is not int or tokens < 0:
                    raise ValueError("Token counter must return a nonnegative integer")
                required_tokens.append(tokens + output_cap)
        too_large = (len(instruction) > max_system_chars or
                     bool(required_tokens) and max(required_tokens) > min(execution_cap, 32768))
        if too_large:
            if recipe.demonstration_count:
                continue
            raise ValueError(f"Frozen zero-shot recipe {recipe.id} exceeds the prompt/output cap")
        candidate = Candidate(name=f"development-{recipe.id}", artifacts={"system_prompt": instruction},
                              runtime=runtime, semantic_kind="opaque", deterministic=False)
        provenance = {
            "schema": "development_performance_candidates_v1", "recipe_id": recipe.id, "dataset": dataset,
            "recipe_sha256": digest({"instruction": recipe.instruction,
                                      "demonstration_count": recipe.demonstration_count}),
            "seed": seed, "selection_policy": "seeded_input_order_balanced_question_types",
            "supervision": "explicit_supervised_development" if selected else "none",
            "development_input_count": len(source), "eligible_input_count": len(eligible),
            "bounded_development_count": len(bounded), "development_limit": development_limit,
            "development_pool_sha256": digest([{"id": e.id, "content_hash": e.content_hash} for e in bounded]),
            "protected_identity_sha256": digest(sorted(protected_keys)),
            "excluded": excluded, "demonstrations": [record for _, record in selected],
            "system_prompt_sha256": digest(instruction), "max_system_chars": max_system_chars,
            "max_demo_chars": max_demo_chars,
            "cap_validation": {"validated": execution_cap is not None,
                               "execution_cap": execution_cap, "output_tokens": output_cap,
                               "task_count": len(prompts),
                               "task_input_sha256": digest(list(prompts)) if prompts else None,
                               "max_required_tokens": max(required_tokens, default=None)},
        }
        built.append(PerformanceCandidate(candidate, recipe.id, FrozenMap(provenance)))
    return tuple(built)
