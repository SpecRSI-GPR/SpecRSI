"""Plan selection by Eq. (8), with exact replay support.

pi_psi(omega|S_t,R_t,Omega_t) =
    (1-epsilon)*exp(z_psi(S_t,R_t,omega)/T_pol)
       / sum_{v in Omega_t} exp(z_psi(S_t,R_t,v)/T_pol)
    + epsilon/|Omega_t|.
Here ``values`` contains policy preference logits z_psi, independently of scorer
values Q_phi. VATS fixes the ACCEPT logit to zero and evaluates the policy on
the paid state S_t after preparation. Collection samples this distribution;
deployment maximizes logits. Replay retains the original canonical menu so
the probabilities and importance ratios in Eqs. (19)--(20) use the same support.

S_t is the paid state after menu preparation, R_t=Remaining(S_t) contains only
observed candidate/job relations, and Omega_t is the nonempty canonical menu.
T_pol>0 controls preference temperature; 0<=epsilon<1 is exploration mass in
the VATS configuration. epsilon is distinct from the actor clipping radius.
The first term allocates 1-epsilon mass according to policy preferences; the
second allocates epsilon uniformly across every legal plan, including ACCEPT.
With fixed logits, subtracting their maximum before exponentiation preserves
the distribution exactly because the common exponential factor cancels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


def menu_distribution(values: Sequence[float], exploration: float = .1, temperature: float = 1.) -> tuple[float, ...]:
    """Evaluate Eq. (8) on one fixed legal menu.

    For index i representing omega_i, z_i=values[i], T=temperature and
    epsilon=exploration, the returned component is
        p_i=(1-epsilon)*exp((z_i-z_max)/T)
             /sum_j exp((z_j-z_max)/T) + epsilon/len(values).
    The stabilizing shift z_max does not change p_i. These are normalized action
    probabilities, not probabilities that checks pass or candidates are valid.
    """
    if (not values or any(not math.isfinite(x) for x in values) or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(exploration) or not 0 <= exploration <= 1):
        raise ValueError("invalid action policy")
    shift = max(values)
    # Subtracting the same maximum from every logit leaves Eq. (8)'s softmax
    # unchanged, while avoiding exp(z/T) overflow. The uniform epsilon term is
    # mixed AFTER normalization, and applies to every legal plan including ACCEPT.
    masses = [math.exp((v - shift) / temperature) for v in values]
    total = math.fsum(masses)
    return tuple((1 - exploration) * x / total + exploration / len(values) for x in masses)


def menu_ratio(menu: Sequence[str], old_menu: Sequence[str], chosen: str,
               new_p: Sequence[float], old_p: Sequence[float]) -> float:
    if tuple(menu) != tuple(old_menu) or len(set(menu)) != len(menu) or len(menu) != len(new_p) or len(menu) != len(old_p):
        raise ValueError("menu/support changed during likelihood replay")
    if chosen not in menu:
        raise ValueError("chosen action absent from support")
    for vector in (new_p, old_p):
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in vector) or not math.isclose(math.fsum(vector), 1., abs_tol=1e-10):
            raise ValueError("invalid probability vector")
    k = list(menu).index(chosen)
    if old_p[k] <= 0:
        raise ValueError("behavior action had zero support")
    return new_p[k] / old_p[k]


@dataclass(frozen=True)
class TraceDecision:
    kind: str
    support: tuple[str, ...]
    selected_id: str
    probabilities: tuple[float, ...]
    values: tuple[float, ...]
    state_version: int
    transition_index: int
    temperature: float
    exploration: float
    proposal_snapshot: str = ""
    state_features: tuple[float, ...] = ()
    option_features: tuple[tuple[float, ...], ...] = ()
    planning_cost_paid: int = 0
    stochastic: bool = True
    value_source: str = "heuristic-fallback"
    cost_before_origin: int = 0
    context_text: str = ""


def choose(support: Sequence[str], values: Sequence[float], *, mode: str, rng: random.Random,
           exploration: float = .1, temperature: float = 1.) -> tuple[str, tuple[float, ...]]:
    """Sample Eq. (8) during collection; maximize preference during deployment.

    Collection records p_psi(omega|S_t,R_t,Omega_t) after temperature and the
    uniform exploration mixture. Deployment returns its actual deterministic
    one-hot choice law. Equal deployment logits use fixed lexical plan identity
    order, so the chosen descriptor does not depend on an accidental RNG draw.
    """
    if len(support) != len(values) or len(set(support)) != len(support) or not support:
        raise ValueError("invalid legal support")
    probabilities = menu_distribution(values, exploration, temperature)
    if mode == "collect":
        return rng.choices(list(support), weights=probabilities, k=1)[0], probabilities
    if mode == "deploy":
        selected = min(zip(values, support), key=lambda pair: (-pair[0], pair[1]))[1]
        return selected, tuple(float(x == selected) for x in support)
    raise ValueError("mode must be collect or deploy")
