"""Bounded autoregressive draft collection with replayable token supports."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
import time

import torch

from .objectives import mixture_probabilities
from .records import DecisionRecord


@dataclass(frozen=True)
class SampledEdit:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    decisions: tuple[DecisionRecord, ...]
    text: str
    token_cost: int
    stop_reason: str


def _eos_token_ids(backbone, explicit=None) -> frozenset[int]:
    """Resolve the actual generation contract before falling back to model config.

    Conditional Qwen3.5 models keep text EOS in ``text_config``; their runtime
    generation config can register multiple terminating tokens. An explicit
    empty sequence still disables EOS stopping for bounded replay fixtures.
    """
    def field(config, name):
        return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)

    value = explicit
    if value is None:
        config = getattr(backbone, "config", None)
        for source in (getattr(backbone, "generation_config", None), field(config, "text_config"), config):
            value = field(source, "eos_token_id")
            if value is not None:
                break
    if value is None:
        return frozenset()
    if type(value) is int:
        value = (value,)
    if not isinstance(value, (tuple, list, set, frozenset)) or any(type(token) is not int or token < 0 for token in value):
        raise ValueError("EOS token IDs must be nonnegative integers or a sequence of them")
    return frozenset(value)


@torch.no_grad()
def sample_edit_tokens(model, input_ids, *, max_new_tokens: int, rng: random.Random,
                       top_k: int = 32, temperature: float = 1., exploration: float = .1,
                       proposal_snapshot: str, transition_index: int = 0,
                       eos_token_ids=None, deadline_monotonic: float | None = None, token_grammar=None) -> SampledEdit:
    """Top-k is a frozen proposal support per token, never recomputed for PPO.

    Eq. (19)'s token case records
        p_vartheta(y_{g,j}|H_{g,j},Y_{g,j})
          = pi_theta(y_{g,j}|H_{g,j},Y_{g,j}),
    while its plan case is pi_psi(omega_t|S_t,R_t,Omega_t). Here H_{g,j}
    contains the actual parent/history prompt plus generated prefix, Y_{g,j}
    is the grammar/top-k allowed token set, and y_{g,j} is the sampled token.
    Temperature and exploration are applied after selecting that support, so
    the stored behavior probability is the actual sampling law. In Eq. (20),
        rho_{g,j} = p_current(y_{g,j}|H_{g,j},Y_{g,j})
                     / p_behavior(y_{g,j}|H_{g,j},Y_{g,j});
    only the numerator is recomputed, on the identical recorded H and Y.
    Recomputing top-k after an update would change Y and break this ratio.

    ``top_k=0`` records the entire vocabulary (large JSONL). All returned token
    decisions store their exact complete prefix and actual normalized behavior
    probabilities. KV caching accelerates collection but does not alter replay.
    The cost is the conventional input-token + generated-token ledger unit.
    A grammar-forced singleton token is deterministic and receives no sampled
    actor likelihood. A sampled EOS is retained before stopping; a failed edit
    still retains all tokens that were actually sampled and billed.
    """
    if model.backend != "hf":
        raise ValueError("Language token collection requires HFRoleModel")
    prompt = tuple(int(x) for x in input_ids)
    if not prompt or max_new_tokens < 1 or top_k < 0 or not proposal_snapshot:
        raise ValueError("Require nonempty bounded prompt, positive output limit, and frozen proposal snapshot")
    if len(prompt) + max_new_tokens > model.config.max_length:
        raise ValueError("Prompt + output cap exceeds configured context length")
    model.eval()
    model._activate("draft")
    eos_token_ids = _eos_token_ids(model.backbone, eos_token_ids)
    prefix = list(prompt)
    generated,decisions = [],[]
    cache = None
    stop = "output_cap"
    # Sampling consumes only the next-token row. On supported backbones avoid
    # projecting every prefill position into the (large) vocabulary; retain the
    # full causal context and KV cache. Other models keep their existing call.
    projection = {"logits_to_keep": 1} if getattr(model, "_supports_logit_positions", False) else {}
    for step in range(max_new_tokens):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            stop='deadline'
            break
        tokens = torch.tensor([prefix if cache is None else [prefix[-1]]],dtype=torch.long,device=model.device)
        attention = torch.ones((1,len(prefix)),dtype=torch.long,device=model.device)
        output = model.backbone(input_ids=tokens,attention_mask=attention,past_key_values=cache,
                                use_cache=True,return_dict=True,**projection)
        cache = output.past_key_values
        logits = output.logits[0,-1].float()
        if token_grammar is not None:
            allowed=tuple(dict.fromkeys(int(x) for x in token_grammar(tuple(generated))))
            if not allowed or min(allowed)<0 or max(allowed)>=logits.numel():raise ValueError('Invalid frozen token grammar support')
            support_ids=torch.tensor(allowed,dtype=torch.long,device=logits.device)
            if top_k and len(allowed)>top_k:support_ids=support_ids[torch.topk(logits[support_ids],top_k,sorted=True).indices]
        elif top_k:
            support_ids = torch.topk(logits,min(top_k,logits.numel()),sorted=True).indices
        else:
            support_ids = torch.arange(logits.numel(),device=logits.device)
        probabilities = mixture_probabilities(logits[support_ids],temperature,exploration)
        weights = probabilities.cpu().tolist()
        selected = rng.choices(range(len(weights)),weights=weights,k=1)[0]
        selected_token = int(support_ids[selected])
        forced=token_grammar is not None and len(support_ids)==1
        decisions.append(DecisionRecord("edit_token",tuple(str(int(x)) for x in support_ids.cpu().tolist()),
            selected,None if forced else float(probabilities[selected].log()),input_ids=tuple(prefix),deterministic=forced,
            behavior_probabilities=tuple(weights),temperature=temperature,exploration=exploration,
            proposal_snapshot=f"{proposal_snapshot}:top_k={top_k}:token={step}",transition_index=transition_index))
        generated.append(selected_token)
        prefix.append(selected_token)
        if selected_token in eos_token_ids:
            stop = "eos"
            break
    text = model.tokenizer.decode(generated,skip_special_tokens=True) if model.tokenizer is not None else ""
    return SampledEdit(prompt,tuple(generated),tuple(decisions),text,len(prompt)+len(generated),stop)
