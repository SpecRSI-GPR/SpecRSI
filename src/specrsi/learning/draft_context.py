"""Budgeted, replayable parent feedback made only from observed information."""
from __future__ import annotations

import json

from specrsi.core.models import thaw


def parent_feedback_context(bridge, state, parent, *, token_limit=None):
    """Share the actual text budget instead of imposing a 256-token ceiling.

    Select at most four parent examples in a deterministic failure-first order.
    Input and observed-output excerpts share unused space from short fields.
    Serialize before checking the final bound: JSON escaping and BPE boundaries
    are part of the input. References and task metadata are never inspected.
    """
    from .frozen_text import allocate_node_tokens, bounded_semantic_text

    tokenizer = bridge.model.tokenizer if bridge.model.backend == "hf" else None
    if token_limit is None and tokenizer is not None:
        token_limit = bridge.model.config.max_length // 3
    if token_limit is not None and (type(token_limit) is not int or token_limit < 1):
        raise ValueError("Draft feedback token budget must be a positive integer")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False) if tokenizer else list(text)

    def serialize(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    rows = [(task, state.ledger.get(parent, task, state.scope)) for task in state.scope.tasks]
    rows.sort(key=lambda item: (item[1] is None,
        bool(item[1].outcome.admissible) if item[1] else False,
        float(item[1].outcome.score) if item[1] else 0., item[0].uid))
    selected = rows[:4]

    while selected:
        examples, fields = [], []
        for task, record in selected:
            example = {"task_id": task.uid[:12], "visible_input": "", "parent_observed_outcome": None}
            text = serialize(thaw(task.inputs))
            fields.append((example, "visible_input", text, encode(text)))
            if record is not None:
                outcome = record.outcome
                observed = {"score": outcome.score, "admissible": outcome.admissible}
                for key in ("answer", "finish_reason", "native_score", "valid"):
                    if key in outcome.fields:
                        value = thaw(outcome.fields[key])
                        if isinstance(value, str):
                            observed[key] = ""
                            if value:
                                fields.append((observed, key, value, encode(value)))
                        else:
                            # Preserve scalar checker feedback rather than
                            # turning numbers/bools into guessed answer text.
                            observed[key] = value
                example["parent_observed_outcome"] = observed
            examples.append(example)
        context = {"feedback_schema": "parent_observed_v2", "parent_id": parent.uid[:12],
            "examples": examples, "examples_omitted": len(rows) - len(examples),
            "parent_missing_count": len(state.ledger.missing(parent, state.scope)),
            "feedback_rule": "null outcome means unobserved; ellipses mark omitted text"}

        for target, key, text, _ in fields:
            target[key] = text
        full = serialize(context)
        if token_limit is None or len(encode(full)) <= token_limit:
            return full

        for target, key, _, _ in fields:
            target[key] = ""
        available = token_limit - len(encode(serialize(context)))
        lengths = tuple(len(tokens) for _, _, _, tokens in fields)
        minimum = sum(min(length, 16) for length in lengths)
        while available >= minimum:
            allowances = allocate_node_tokens(lengths, available=available, minimum=16,
                                               maximum=max(16, max(lengths)))
            for (target, key, text, tokens), allowance in zip(fields, allowances):
                if len(tokens) <= allowance:
                    target[key] = text
                else:
                    if tokenizer:
                        target[key] = bounded_semantic_text(tokenizer, text, allowance)[0]
                    else:
                        head, tail = (allowance + 1) // 2, allowance // 2
                        target[key] = text[:head] + " … " + (text[-tail:] if tail else "")
            bounded = serialize(context)
            overflow = len(encode(bounded)) - token_limit
            if overflow <= 0:
                return bounded
            available -= max(1, overflow)
        # A complete feedback example needs its structure and minimal excerpts.
        # Omit the least-prioritized example explicitly, never cut its JSON.
        selected.pop()

    compact = serialize({"feedback_schema": "parent_observed_v2", "parent_id": parent.uid[:12],
        "examples": [], "examples_omitted": len(rows)})
    if token_limit is not None and len(encode(compact)) > token_limit:
        raise ValueError("Draft feedback token budget cannot fit its observation header")
    return compact
