"""Metered answer-generation strategies with versioned reasoning interfaces.

These recipes change the deployed QA solver, not the search algorithm or scorer.
The executor receives the public task prompt only. References and benchmark
feedback cannot be supplied through this API. Every attempted generation is
recorded; missing server usage fails closed and reserves its enforced upper bound.

reasoning_extract_v2 adds native thinking and extraction from separate backend
reasoning when final content is empty. The v1 interface uses nonthinking,
final-content-only responses. A timed-out response is never a successful
reasoning stage.

native_completion_v3 preserves a syntactically complete stopped native integer
answer. Otherwise it uses a separately budgeted completion of the actual prior
work; both calls retain their full physical costs. It does not test correctness
to decide whether to spend more computation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
import re
import time
import uuid

from .models.client import sampling_options


@dataclass(frozen=True)
class AnswerStrategy:
    version: str = "hotpot_evidence_v1"
    mode: str = "evidence_extract"
    samples: int = 1
    reasoning_tokens: int = 384
    answer_tokens: int = 48
    arbiter_tokens: int = 96
    temperature: float = .3
    agreement: str = "unanimous"
    answer_format: str = "short_answer"

    def __post_init__(self):
        if self.version not in {"hotpot_evidence_v1", "reasoning_extract_v1", "reasoning_extract_v2", "native_completion_v3"}:
            raise ValueError("Unknown answer strategy version")
        single_mode = "evidence_extract" if self.version == "hotpot_evidence_v1" else "reasoning_extract"
        if self.mode not in {single_mode, "independent_consensus"}:
            raise ValueError("Unknown answer strategy mode")
        if type(self.samples) is not int or not 1 <= self.samples <= 5:
            raise ValueError("Answer strategy requires 1 to 5 samples")
        if (self.mode == single_mode) != (self.samples == 1):
            raise ValueError("Single extraction requires one sample; consensus requires two or more")
        if self.version == "native_completion_v3" and (self.samples != 1 or self.answer_format != "boxed_integer"):
            raise ValueError("Native completion v3 requires one sample and boxed_integer answers")
        for name in ("reasoning_tokens", "answer_tokens", "arbiter_tokens"):
            value = getattr(self, name)
            limit = ({"reasoning_extract_v2": 8192, "native_completion_v3": 24576}.get(self.version, 4096)
                     if name == "reasoning_tokens" else 4096)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f"{name} must be an integer from 1 to {limit}")
        if (type(self.temperature) not in (int, float) or not math.isfinite(self.temperature)
                or not 0 <= self.temperature <= 2):
            raise ValueError("Answer strategy temperature must be finite and between 0 and 2")
        if self.agreement not in {"unanimous", "majority"}:
            raise ValueError("Answer strategy agreement must be unanimous or majority")
        if self.answer_format not in {"short_answer", "boxed_integer", "boxed_expression", "task_instruction"}:
            raise ValueError("Unknown answer strategy final answer format")
        if self.version == "hotpot_evidence_v1" and self.answer_format != "short_answer":
            raise ValueError("HotpotQA evidence extraction requires short_answer format")

    @classmethod
    def from_dict(cls, raw: Mapping):
        if not isinstance(raw, Mapping) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Answer strategy must contain only registered recipe fields")
        if "version" not in raw or "mode" not in raw:
            raise ValueError("Answer strategy requires an explicit version and mode")
        return cls(**dict(raw))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StrategyResult:
    status: str
    answer: str | None
    tokens: int
    records: tuple[dict, ...]
    error: str | None = None
    decision: str | None = None


_EVIDENCE_INSTRUCTION = """Solve the supplied multi-hop question using the supplied passages.
Read the question precisely. Identify the requested answer type and the entities
connecting the relevant passages. Check comparisons, dates, names, and yes/no
conditions explicitly. Distractor passages may mention similar entities.
Write a compact evidence note: the useful passage titles, the decisive facts, the
connection between them, and your proposed answer. Use only supported facts;
do not invent missing evidence. The note is intermediate work for a separate
answer extractor, so prioritize decisive evidence over restating the passages."""

_EXTRACTION_INSTRUCTION = """Answer the original question using the supplied passages.
An independent evidence note is included below as a fallible aid. Verify its
conclusion against the passages and correct mistakes. Return ONLY the shortest
complete answer: a name, place, date, number, or yes/no as the question requires.
Preserve necessary qualifiers in the entity name. Do not include an explanation,
citations, answer labels, quotation marks, or alternatives."""

_ARBITER_INSTRUCTION = """Resolve a disagreement between independent answers to the
original multi-hop question. Verify each proposed answer and its evidence against
the supplied passages. Check entity links, date comparisons, and the exact
relation asked. The proposals may all be wrong; the passages decide the answer.
Return ONLY the shortest complete answer, without explanation, citations,
answer labels, quotation marks, or alternatives."""


_REASONING_INSTRUCTION = """Solve the given problem independently. Identify every
constraint and the requested quantity. Derive the result step by step, keeping
exact arithmetic where possible. Check edge cases and verify the result against
the original constraints or with a second calculation. End your working notes
with the proposed final answer. These notes will be reviewed by a separate
answer extractor. Prioritize a complete derivation and check over verbosity."""

_REASONING_EXTRACTION = """Solve the original problem, using the proposed solution
notes below as a fallible aid. Check the derivation, arithmetic, assumptions, and
whether the proposed result answers the exact question. Correct any mistakes.
Return only the final answer, with no working or explanation."""

_REASONING_ARBITER = """Resolve the disagreement between independent solutions to
the original problem. Check derivations, arithmetic, and the original constraints.
The proposals may all be wrong. Determine the mathematically correct result.
Return only the final answer, with no working or explanation."""

_NATIVE_COMPLETION = """Finish solving the original mathematical problem. The
previous work below is incomplete and may contain errors. Preserve correct
progress, finish any unfinished calculation, and correct concrete mistakes.
Do not restart already completed work or repeat the same checks. Give only the
necessary remaining calculation, then end with exactly one final answer in
\\boxed{N}, where N is an integer from 0 to 999. Do not reduce modulo 1000 unless
the problem explicitly requests it."""

_FORMATS = {
    "short_answer": "Return only the shortest complete answer, without an answer label or explanation.",
    "boxed_integer": r"Return exactly one integer enclosed in \boxed{...}, with no other text.",
    "boxed_expression": r"Return exactly one final mathematical expression enclosed in \boxed{...}, with no other text.",
    "task_instruction": "Follow the final-answer format required by the task and the original system instruction exactly.",
}


def evidence_messages(instruction: str, task_prompt: str, strategy: AnswerStrategy | None = None):
    if strategy is not None and strategy.version == "native_completion_v3":
        # The first request is identical to the registered native control:
        # no extra instruction to create notes or reserve work for an extractor.
        return [{"role": "system", "content": instruction}, {"role": "user", "content": task_prompt}]
    reasoning = (_REASONING_INSTRUCTION if strategy is not None and strategy.version != "hotpot_evidence_v1"
                 else _EVIDENCE_INSTRUCTION)
    return [{"role": "system", "content": instruction + "\n\n" + reasoning},
            {"role": "user", "content": task_prompt}]


def reasoning_enable_thinking(strategy: AnswerStrategy, generation: Mapping) -> bool:
    """Versioned reasoning-stage contract shared by preflight and execution.

    The v1 interface uses nonthinking generation. Explicit v2/v3
    recipes opt into native thinking and separate reasoning notes; extraction,
    completion and arbitration remain nonthinking under every version.
    """
    if strategy.version not in {"reasoning_extract_v2", "native_completion_v3"}:
        return False
    enabled = generation.get("enable_thinking", False)
    if type(enabled) is not bool:
        raise ValueError("Native reasoning strategies require a boolean enable_thinking")
    if strategy.version == "native_completion_v3" and not enabled:
        raise ValueError("Native completion v3 requires enable_thinking=true")
    return enabled


def _complete_native_integer(record: Mapping) -> bool:
    """Output-format check only; never consult the reference or task score.

    Even a syntactically complete answer may be wrong. Preserve it only when
    the backend actually stopped and there is one unambiguous in-range box.
    """
    if record["finish_reason"] != "stop":
        return False
    boxes = re.findall(r"\\boxed\s*\{([^{}]*)\}", record["text"])
    return (len(re.findall(r"\\boxed\b", record["text"])) == len(boxes) == 1
            and re.fullmatch(r"\s*[0-9]{1,3}\s*", boxes[0]) is not None
            and 0 <= int(boxes[0]) <= 999)


def _agreement_key(answer: str):
    # Conservative agreement only; this is not the benchmark scorer.
    return re.sub(r"\s+", " ", answer.strip().casefold()).rstrip(".")


def execute_answer_strategy(strategy: AnswerStrategy, task_prompt: str, instruction: str,
                            executor, *, generation: Mapping, model_revision: str,
                            cap: int, seed: int) -> StrategyResult:
    """Execute a fixed recipe without ever receiving a Task or its reference.

    Stage output caps are never shortened to fit a remaining budget. A failed
    prefix is not an answer. A transport failure's token debit is conservatively
    its entire server-enforced reservation, explicitly marked as unmeasured.
    """
    if type(cap) is not int or cap < 0:
        raise ValueError("Answer strategy token cap must be a nonnegative integer")
    if not isinstance(task_prompt, str) or not isinstance(instruction, str):
        raise ValueError("Answer strategy requires a public prompt and instruction")
    records: list[dict] = []
    used = 0
    sampling = sampling_options(generation)

    def result(status, answer=None, error=None, decision=None):
        return StrategyResult(status, answer, used, tuple(records), error, decision)

    def call(stage, messages, output_cap, temperature, *, enable_thinking=False, allow_reasoning=False):
        nonlocal used
        prompt_cost = executor.prompt_tokens(messages, enable_thinking=enable_thinking)
        reservation = prompt_cost + output_cap
        if reservation > min(cap - used, 32768):
            return None, result("interrupted", error=f"Full frozen {stage} contract does not fit token cap")
        started = time.monotonic()
        try:
            reply = executor.client.generate(messages, max_tokens=output_cap,
                temperature=temperature, top_p=generation.get("top_p", .8),
                seed=(seed + len(records)) % 2**32, enable_thinking=enable_thinking,
                max_seconds=generation.get("max_seconds", 120), total_token_cap=reservation, **sampling)
        except Exception as error:
            # No free retries and no fabricated actual usage. The server enforced
            # this reservation, so it bounds any usage hidden by transport loss.
            used += reservation
            records.append({"stage": stage, "request_id": "unconfirmed-" + uuid.uuid4().hex,
                "enable_thinking": enable_thinking,
                "finish_reason": "transport_error", "error_type": type(error).__name__,
                "metering_status": "unavailable_reserved_upper_bound",
                "accounted_tokens": reservation, "reserved_tokens": reservation,
                "client_elapsed_seconds": time.monotonic() - started})
            return None, result("error", error=f"{stage} call failed without a usage receipt ({type(error).__name__}); full reservation charged")
        actual = reply.prompt_tokens + reply.completion_tokens
        used += actual
        records.append({"stage": stage, **asdict(reply), "accounted_tokens": actual,
                        "reserved_tokens": reservation, "metering_status": "server_measured",
                        "enable_thinking": enable_thinking})
        if actual > reservation or used > cap:
            return None, result("error", error=f"{stage} server violated enforced token cap")
        if reply.prompt_tokens != prompt_cost:
            return None, result("error", error=f"{stage} tokenizer counts differ from frozen request")
        if reply.model_revision != model_revision:
            return None, result("error", error=f"{stage} model revision changed")
        if reply.finish_reason == "timeout":
            return None, result("interrupted", error=f"{stage} generation timed out")
        if reply.text.strip():
            records[-1]["content_source"] = "final"
            return reply.text.strip(), None
        if allow_reasoning:
            if reply.reasoning_error is not None:
                return None, result("error", error=f"{stage} rejected reasoning notes ({reply.reasoning_error})")
            if reply.reasoning_content is not None and reply.reasoning_content.strip():
                records[-1]["content_source"] = "reasoning_content"
                return reply.reasoning_content.strip(), None
        return None, result("error", error=f"{stage} generated an empty result")

    proposals = []
    generic = strategy.version != "hotpot_evidence_v1"
    native_thinking = reasoning_enable_thinking(strategy, generation)
    use_reasoning_notes = strategy.version in {"reasoning_extract_v2", "native_completion_v3"}
    stage_name = "reasoning" if generic else "evidence"
    for index in range(strategy.samples):
        evidence, failure = call(f"{stage_name}_{index}", evidence_messages(instruction, task_prompt, strategy),
                                 strategy.reasoning_tokens, strategy.temperature,
                                 enable_thinking=native_thinking, allow_reasoning=use_reasoning_notes)
        if failure is not None:
            return failure
        if strategy.version == "native_completion_v3":
            if _complete_native_integer(records[-1]):
                return result("complete", records[-1]["text"], decision="native_final_preserved")
            if records[-1].get("reasoning_error") is not None:
                return result("error", error="Native completion rejected invalid reasoning notes")
            reasoning = records[-1].get("reasoning_content")
            partial = records[-1]["text"].strip()
            evidence = (reasoning.strip() if reasoning else "")
            if partial:
                evidence += "\n\nPartial final response:\n" + partial
            messages = [{"role": "system", "content": _NATIVE_COMPLETION},
                        {"role": "user", "content": task_prompt + "\n\nFallible prior work:\n" + evidence}]
            answer, failure = call("complete_0", messages, strategy.answer_tokens, 0.)
            if failure is not None:
                return failure
            return result("complete", answer, decision="native_budget_completion")
        extraction = _REASONING_EXTRACTION if generic else _EXTRACTION_INSTRUCTION
        messages = [{"role": "system", "content": instruction + "\n\n" + extraction + "\n" + _FORMATS[strategy.answer_format]},
                    {"role": "user", "content": task_prompt + "\n\nFallible solution notes:\n" + evidence}]
        answer, failure = call(f"extract_{index}", messages, strategy.answer_tokens, 0.)
        if failure is not None:
            return failure
        proposals.append((evidence, answer))
    if strategy.samples == 1:
        return result("complete", proposals[0][1], decision="single_" + stage_name + "_extraction")
    counts = Counter(_agreement_key(answer) for _, answer in proposals)
    winner, frequency = counts.most_common(1)[0]
    sufficient = frequency == strategy.samples if strategy.agreement == "unanimous" else frequency > strategy.samples / 2
    if winner and sufficient:
        answer = next(answer for _, answer in proposals if _agreement_key(answer) == winner)
        return result("complete", answer, decision=strategy.agreement + "_agreement")
    discussion = "\n\n".join(f"Proposal {index + 1}:\nEvidence: {evidence}\nAnswer: {answer}"
                              for index, (evidence, answer) in enumerate(proposals))
    arbitration = _REASONING_ARBITER if generic else _ARBITER_INSTRUCTION
    messages = [{"role": "system", "content": instruction + "\n\n" + arbitration + "\n" + _FORMATS[strategy.answer_format]},
                {"role": "user", "content": task_prompt + "\n\nIndependent proposals:\n" + discussion}]
    answer, failure = call("arbiter", messages, strategy.arbiter_tokens, 0.)
    if failure is not None:
        return failure
    return result("complete", answer, decision=stage_name + "_arbiter")
