"""Auditable training records. Predictions never become checker labels or outcomes."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class DecisionRecord:
    """The fixed conditioning data and support for one controlled choice.

    Eq. (19), with vartheta=(theta,psi), defines the typed likelihood
        p_vartheta(y_{g,j}|H_{g,j},Y_{g,j})
          = pi_psi(omega_t|S_t,R_t,Omega_t)       for a sampled plan,
          = pi_theta(y_{g,j}|H_{g,j},Y_{g,j})    for a sampled edit token.
    For a plan, ``support`` is Y=Omega_t and its selected element is omega_t.
    ``state_features``, ``context_text``, graph relations and option features
    jointly record H=(S_t,R_t,Omega_t). For a token, input_ids records the
    actual parent/history prompt followed by the sampled prefix; support is
    the allowed token set after the collection mask. ``selected_index`` names
    the sampled y, and behavior_logprob stores log p_vartheta_k(y|H,Y).
    Temperature, exploration mass and any nonuniform exploration probabilities
    are part of this replay distribution rather than recomputed defaults.

    Eq. (20) forms a separate ratio for each sampled token or plan. Invalid
    sampled edits retain these records and their paid work. A deterministic
    grammar/solver choice has no actor likelihood; its flag and None logprob
    prevent assigning a fictitious sampled probability. Forced trajectories
    may contain later sampled records, but their enclosing trajectory excludes
    every one of them from actor training. Only observed plan starts give Q
    labels; token likelihoods do not create independent value targets.
    """
    kind: str
    support: tuple[str, ...]
    selected_index: int
    behavior_logprob: float | None
    state_features: tuple[float, ...] = ()
    option_features: tuple[tuple[float, ...], ...] = ()
    transition_index: int = 0
    temperature: float = 1.0
    exploration: float = 0.1
    proposal_snapshot: str = ""
    context_text: str = ""
    input_ids: tuple[int, ...] = ()
    behavior_probabilities: tuple[float, ...] = ()
    exploration_probabilities: tuple[float, ...] = ()
    # If the decision was recorded after some planning, align its target with the
    # earlier value origin by subtracting this *already paid* cost once.
    prior_cost: float = 0.0
    excluded_prefix_cost: float = 0.0
    prior_elapsed: float = 0.0
    excluded_prefix_elapsed: float = 0.0
    deterministic: bool = False
    option_item_features: tuple[tuple[tuple[float, ...], ...], ...] = ()
    graph_nodes: tuple[tuple[float, ...], ...] = ()
    graph_edges: tuple[tuple[int, int, int], ...] = ()
    option_node_masks: tuple[tuple[float, ...], ...] = ()
    graph_node_texts: tuple[str, ...] = ()
    graph_node_text_features: tuple[tuple[float, ...], ...] = ()
    graph_text_snapshot: str = ""
    graph_node_input_ids: tuple[tuple[int, ...], ...] = ()
    graph_schema: str = "candidate_items_v1"

    def __post_init__(self):
        from .work_graph import relation_count
        relations = relation_count(self.graph_schema)
        if self.kind not in {"outer", "batch", "plan", "edit", "edit_token"}:
            raise ValueError(f"Unknown controlled decision kind: {self.kind}")
        if not self.support or len(set(self.support)) != len(self.support):
            raise ValueError("Stored support must be nonempty and canonicalized (unique IDs)")
        if not 0 <= self.selected_index < len(self.support):
            raise ValueError("Selected decision is outside stored support")
        if self.option_features and len(self.option_features) != len(self.support):
            raise ValueError("Each support option needs exactly one feature vector")
        if self.option_item_features and (self.kind != 'batch' or len(self.option_item_features) != len(self.support)):
            raise ValueError('Additive batch features require one frozen item collection per batch option')
        if self.graph_nodes and any(len(node) != 12 or any(not math.isfinite(x) for x in node) for node in self.graph_nodes):
            raise ValueError("Typed graph nodes require 12 finite observed features")
        if any(len(edge) != 3 or not 0 <= edge[0] < len(self.graph_nodes) or not 0 <= edge[1] < len(self.graph_nodes)
               or edge[2] not in range(relations) for edge in self.graph_edges):
            raise ValueError("Graph edges must name existing nodes and a registered directed relation type")
        if self.option_node_masks and (len(self.option_node_masks) != len(self.support) or
                any(len(mask) != len(self.graph_nodes) or any(value not in (0., 1.) for value in mask)
                    for mask in self.option_node_masks)):
            raise ValueError("Each plan needs its recorded graph membership mask")
        if self.graph_node_texts or self.graph_node_text_features or self.graph_text_snapshot:
            widths = {len(row) for row in self.graph_node_text_features}
            if (not self.graph_text_snapshot or not self.graph_nodes or
                    len(self.graph_node_texts) != len(self.graph_nodes) or
                    len(self.graph_node_text_features) != len(self.graph_nodes) or
                    any(not isinstance(text, str) or not text for text in self.graph_node_texts) or
                    len(widths) != 1 or not next(iter(widths), 0) or
                    any(not math.isfinite(value) for row in self.graph_node_text_features for value in row)):
                raise ValueError("Semantic graph records require aligned texts, finite vectors and encoder identity")
        if self.graph_node_input_ids and (not self.graph_text_snapshot
                or len(self.graph_node_input_ids) != len(self.graph_nodes)
                or any(not ids or any(type(token) is not int or token < 0 for token in ids)
                       for ids in self.graph_node_input_ids)):
            raise ValueError("Semantic node input token IDs must align with the recorded graph and encoder")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        if not 0 <= self.exploration <= 1 or any(not math.isfinite(x) or x < 0 for x in
                (self.prior_cost, self.excluded_prefix_cost, self.prior_elapsed, self.excluded_prefix_elapsed)):
            raise ValueError("Invalid exploration mixture or prior cost")
        if self.transition_index < 0:
            raise ValueError("Transition index must be nonnegative")
        if self.deterministic:
            if self.behavior_logprob is not None:
                raise ValueError("Forced/solver decisions must not receive an actor likelihood")
        elif self.behavior_logprob is None or not math.isfinite(self.behavior_logprob) or self.behavior_logprob > 1e-7:
            raise ValueError("Stochastic decisions require their actual behavior log probability")
        for name in ("behavior_probabilities", "exploration_probabilities"):
            values = getattr(self, name)
            if values and (len(values) != len(self.support) or any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1., abs_tol=1e-5)):
                raise ValueError(f"Invalid {name}")
        if self.exploration_probabilities and any(p <= 0 for p in self.exploration_probabilities):
            raise ValueError("Exploration distribution must give every legal option positive mass")
        if self.behavior_probabilities and not self.deterministic:
            p = self.behavior_probabilities[self.selected_index]
            if p <= 0 or not math.isclose(math.log(p), self.behavior_logprob, abs_tol=1e-5):
                raise ValueError("Behavior probability disagrees with logged selected likelihood")

    @property
    def support_digest(self) -> str:
        return hashlib.sha256(json.dumps([self.support, self.proposal_snapshot], separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TransitionRecord:
    """One disjoint paid interval on the Eq. (16) preparation/work clock.

    A decision t has two transitions: S_t^- --prepare--> S_t --work--> S_{t+1}^-.
    Eq. (16) updates the engine's resources and defines the interval rewards:
        B_t = B_t^- - o_t,
        B_{t+1}^- = B_t - kappa_t,
        N_{t+1} = N_t - n_t,
        r_t^m = -lambda*o_t - mu*ell_t^m,
        r_t^w = F(S_{t+1}^-)-F(S_t)-lambda*kappa_t-mu*ell_t^w.
    B is remaining expense allowance, N the primitive-attempt allowance, o_t
    paid preparation expense, kappa_t paid selected-work expense, and n_t the
    attempts actually charged by the engine. Superscripts m and w identify
    the menu/preparation and selected-work intervals; they are not powers.

    This record stores the observed F endpoints and the actual cost/time of
    one interval. Its implied reward is f_after-f_before-lambda*cost-mu*time.
    Preparation acquires no evaluator outcomes, so f_after=f_before there.
    ``cost`` uses the registered search-budget unit (for example tokens or
    milliseconds); ``elapsed_seconds`` always uses seconds on the host clock.
    lambda has score-units/cost-unit and mu has score-units/second. Concurrent
    worker runtimes must not be summed into this elapsed host interval.
    Attempt/resource updates occur in the execution engine and are not applied
    again while constructing these training records.

    For a sampled ACCEPT, S_{t+1}^-=S_t and kappa_t=ell_t^w=n_t=r_t^w=0;
    preparation paid before that choice is still represented separately.
    """
    f_before: float
    f_after: float
    cost: float
    completed_item_reward: float | None = None
    elapsed_seconds: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.f_before, self.f_after, self.cost, self.elapsed_seconds)) or min(self.cost, self.elapsed_seconds) < 0:
            raise ValueError("Transitions need finite scores and nonnegative actual metered cost")
        if self.f_after < self.f_before - 1e-9:
            raise ValueError("Best complete admissible improvement F cannot decrease")
        if self.completed_item_reward is not None and (not math.isfinite(self.completed_item_reward) or self.completed_item_reward < 0):
            raise ValueError('Completed-item reward must be a finite nonnegative observed score sum')


@dataclass(frozen=True)
class SearchTrajectory:
    group_id: str
    initial_context_id: str
    search_id: str
    seed: int
    transitions: tuple[TransitionRecord, ...]
    decisions: tuple[DecisionRecord, ...]
    initial_f: float = 0.
    final_f: float = 0.
    terminated: bool = True
    lineage_id: str = ""
    audit_only: bool = False
    forced: bool = False
    collection_snapshot: str = ""
    contract_id: str = ""
    initial_budget: float | None = None
    workers: int | None = None
    demonstration: bool = False

    def __post_init__(self):
        if not self.terminated:
            raise ValueError("GVO requires a terminated full search, including final completion work")
        if not all(math.isfinite(x) for x in (self.initial_f, self.final_f)):
            raise ValueError("Nonfinite terminal score")
        expected = self.initial_f
        for t in self.transitions:
            if not math.isclose(t.f_before, expected, abs_tol=1e-8):
                raise ValueError("Transition F values do not form one continuous full search")
            expected = t.f_after
        if not math.isclose(expected, self.final_f, abs_tol=1e-8):
            raise ValueError("Final F disagrees with final transition")
        if any(d.transition_index >= len(self.transitions) for d in self.decisions):
            raise ValueError("Decision must point to its actual transition origin")

    @property
    def total_cost(self) -> float:
        return math.fsum(t.cost for t in self.transitions)

    @property
    def total_elapsed(self) -> float:
        return math.fsum(t.elapsed_seconds for t in self.transitions)

    def full_return(self, cost_coefficient: float, time_coefficient: float = 0.) -> float:
        """Eq. (13): compute one realized full-search return before the group baseline.

            R_g = F(S_end^g)-F(S_0^-)-lambda*C_g-mu*L_g,
            A_g = R_g-(1/(G-1))*sum_{u=1,u!=g}^{G} R_u.
        This method computes R_g; objectives.group_advantages computes A_g
        only after G independent searches from matching starts have terminated.
        ``final_f`` is F(S_end^g), ``initial_f`` is F(S_0^-), and the two
        coefficients are lambda and mu. Both return and advantage have the
        same utility units as F; G counts searches, not tokens or decisions.

        Using Eq. (16)'s preparation/work intervals and T_g visited choices:
            C_g = sum_{t=0}^{T_g-1} (o_{g,t}+kappa_{g,t}),
            L_g = sum_{t=0}^{T_g-1} (ell_{g,t}^m+ell_{g,t}^w),
            R_g = sum_{t=0}^{T_g-1} (r_{g,t}^m+r_{g,t}^w).
        The stored transitions include every paid interval, including initial
        preparation, terminal preparation, failed attempts and unfinished work.
        The F differences telescope because preparation leaves F unchanged.
        For example, zero verified gain and total expense 7 at lambda=0.1,
        mu=0 gives R_g=-0.7. Costs cannot disappear when no child is adopted.
        The Eq. (21) KL penalty is an actor loss term and is excluded from R_g.
        """
        return self.final_f - self.initial_f - cost_coefficient * self.total_cost - time_coefficient * self.total_elapsed

    def target(self, decision: DecisionRecord, cost_coefficient: float, time_coefficient: float = 0.) -> float:
        """Eqs. (11) and (17): form the Monte Carlo label at a paid plan start.

        Eq. (11) defines both the observed label and its regression objective:
            G_t = F(S_end)-F(S_t)-lambda*C_{t:end}-mu*L_{t:end},
            L_Q(phi) = E_{(S_t,omega_t,G_t)~D_Q}
                         [(Q_hat_phi(S_t,omega_t)-G_t)^2].
        The trainer implements the empirical mean over all collected D_Q
        tuples; this method supplies only the selected plan's observed G_t.
        No label is invented for an unexecuted alternative. Ordinary plan
        starts and paid forced plan starts with the same continuation policy
        are both valid observations of Eq. (6)'s conditional expectation.

        For T paid choices, Eq. (17) expands the same suffix into intervals:
            G_t = r_t^w + sum_{u=t+1}^{T-1} (r_u^m+r_u^w),
            C_{t:end} = kappa_t + sum_{u=t+1}^{T-1} (o_u+kappa_u),
            L_{t:end} = ell_t^w + sum_{u=t+1}^{T-1} (ell_u^m+ell_u^w).
        The current o_t and ell_t^m are sunk at S_t and are excluded. Every
        later preparation, selected action, failed attempt and terminal menu
        fee stays in the suffix; the discount factor is exactly one.

        ``transition_index`` points at the chosen work interval in the two-step
        log. ``remaining[0].f_before`` therefore equals F(S_t). A combined
        interval instead removes its already paid ``excluded_prefix_*`` once;
        ``prior_*`` restores an earlier recorded value origin if needed.
        Costs use the registered budget unit, times use host seconds, and the
        coefficients convert both to utility units. For gain 2, work cost 3,
        later menu/work cost 4 and lambda=0.1, mu=0, G_t=2-0.1*(3+4)=1.3;
        a current menu cost 5 does not also enter this post-menu label.
        ACCEPT has G_t=0. No learned V baseline or bootstrapped Q enters G_t.
        """
        remaining = self.transitions[decision.transition_index:]
        if decision.excluded_prefix_cost > remaining[0].cost:
            raise ValueError("Excluded prefix cost exceeds the enclosing transition")
        if decision.excluded_prefix_elapsed > remaining[0].elapsed_seconds:
            raise ValueError("Excluded prefix elapsed time exceeds the enclosing transition")
        target = self.final_f - remaining[0].f_before - cost_coefficient * (math.fsum(t.cost for t in remaining)
            - decision.excluded_prefix_cost + decision.prior_cost) - time_coefficient * (
                math.fsum(t.elapsed_seconds for t in remaining) - decision.excluded_prefix_elapsed + decision.prior_elapsed)
        if decision.kind == "plan" and decision.support[decision.selected_index] == "ACCEPT" and not math.isclose(target, 0., abs_tol=1e-9):
            raise ValueError("ACCEPT has zero continuation return; check the recorded post-preparation origin")
        return target

    def item_return(self, cost_coefficient: float, decision: DecisionRecord | None = None) -> float:
        remaining = self.transitions[decision.transition_index:] if decision else self.transitions
        if any(t.completed_item_reward is None for t in remaining):
            raise ValueError('Item-reward ablation requires recorded completed-item rewards, never inferred labels')
        adjustment = 0.
        if decision:
            if decision.excluded_prefix_cost > remaining[0].cost:
                raise ValueError('Excluded prefix cost exceeds enclosing transition')
            adjustment = decision.prior_cost-decision.excluded_prefix_cost
        return sum(t.completed_item_reward for t in remaining)-cost_coefficient*(sum(t.cost for t in remaining)+adjustment)


@dataclass(frozen=True)
class CheckerAudit:
    lineage_id: str
    protocol_id: str
    completed: bool
    label: int | None
    features: tuple[float, ...]
    context_text: str = ""
    audit_only: bool = False

    def __post_init__(self):
        if not self.lineage_id or not self.protocol_id:
            raise ValueError("Checker audits require lineage and version-matched protocol IDs")
        if self.completed and self.label not in (0, 1):
            raise ValueError("Completed checks require binary P=1/U=0 authorization labels")
        if not self.completed and self.label is not None:
            raise ValueError("Interrupted checks have no binary label (they are not U)")


@dataclass(frozen=True)
class EditExample:
    lineage_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    verified: bool
    audit_only: bool = False
    token_decisions: tuple[DecisionRecord, ...] = ()

    def __post_init__(self):
        if not self.verified:
            raise ValueError("Edit demonstrations must have completed verification")
        if not self.lineage_id or len(self.input_ids) != len(self.labels) or len(self.input_ids) < 2:
            raise ValueError("Edit examples require aligned token IDs/labels and lineage")
        if all(x == -100 for x in self.labels[1:]):
            raise ValueError("Edit has no supervised continuation tokens")
        if self.token_decisions:
            positions = [len(d.input_ids) for d in self.token_decisions]
            supervised = [i for i, label in enumerate(self.labels) if label != -100]
            if positions != supervised:
                raise ValueError("Recorded edit decisions must cover every supervised token exactly once")
            for d in self.token_decisions:
                index = len(d.input_ids)
                if (d.kind != "edit_token" or d.input_ids != self.input_ids[:index] or
                        int(d.support[d.selected_index]) != self.labels[index] or self.input_ids[index] != self.labels[index]):
                    raise ValueError("Edit supervision must retain its actual prefix, token and frozen support")
                if d.deterministic and len(d.support) != 1:
                    raise ValueError("A forced grammar token must have singleton support")


@dataclass(frozen=True)
class PairwiseValueExample:
    """Paid repetitions from the same saved state and collection snapshot.

    Eq. (22), with prospectively fixed positive integers n_a and n_b:
        G_bar_a = (1/n_a) * sum_{s=1}^{n_a} G_{a,s},
        delta_hat_ab = G_bar_a-G_bar_b,
        Var(delta_hat_ab | S) = sigma_a^2/n_a + sigma_b^2/n_b,
        sigma_a^2 = Var(G_{a,s} | S,a,Theta_k).
    The variance equality assumes repetitions within each arm, and the two
    arms, are independent conditional on the same paid state S and frozen
    continuation parameters Theta_k. For coupled arms the general expression
    subtracts 2*Cov(G_bar_a,G_bar_b | S). Identical initial state alone does
    not establish independence or a variance reduction.

    ``left_returns`` and ``right_returns`` are observed G_{a,s} and G_{b,s};
    ``left_count`` and ``right_count`` are registered n_a,n_b, and the seed
    arrays record independent random streams. ``difference`` below evaluates
    the two empirical means and delta_hat_ab. The variance equations explain
    their statistical assumptions; this class does not estimate sigma^2 or
    insert an analytical variance into the training label.
    A missing repetition cannot be dropped or replaced with zero. A fully
    logged unsuccessful repetition, including its expense, stays in its mean.
    """
    left: DecisionRecord
    right: DecisionRecord
    left_returns: tuple[float, ...]
    right_returns: tuple[float, ...]
    paid_state_id: str
    snapshot_id: str
    left_seeds: tuple[int, ...]
    right_seeds: tuple[int, ...]
    left_count: int
    right_count: int
    audit_only: bool = False
    lineage_id: str = ""

    def __post_init__(self):
        if not self.paid_state_id or not self.snapshot_id:
            raise ValueError("Paired returns need a saved paid state and frozen collection snapshot")
        if (self.left.state_features, self.left.context_text, self.left.input_ids, self.left.graph_nodes, self.left.graph_edges,
                self.left.graph_node_texts, self.left.graph_node_text_features, self.left.graph_text_snapshot,
                self.left.graph_node_input_ids, self.left.graph_schema) != (
                self.right.state_features, self.right.context_text, self.right.input_ids, self.right.graph_nodes, self.right.graph_edges,
                self.right.graph_node_texts, self.right.graph_node_text_features, self.right.graph_text_snapshot,
                self.right.graph_node_input_ids, self.right.graph_schema):
            raise ValueError("Paired plans must start at exactly the same recorded paid state")
        if self.left.kind not in {"plan", "outer", "batch"} or self.right.kind not in {"plan", "outer", "batch"}:
            raise ValueError("Pair targets belong to executed work plans")
        # Appendix F.5 / Eq. (22) compares a,b drawn from ONE saved legal menu.
        # Equal state embeddings alone are insufficient: the same option ID can
        # encode different work/features after changing a menu or generator.
        # Keep the two different selected arms while requiring the replay menu
        # itself (including plan membership masks) to agree exactly.
        menu_fields = ("kind", "support", "option_features", "option_item_features",
                       "option_node_masks", "proposal_snapshot")
        if any(getattr(self.left, name) != getattr(self.right, name) for name in menu_fields):
            raise ValueError("Paired arms must come from the same recorded menu and proposal snapshot")
        if self.left.selected_index == self.right.selected_index:
            raise ValueError("Paired continuations must compare distinct legal arms")
        for returns, seeds, count in ((self.left_returns, self.left_seeds, self.left_count),
                                      (self.right_returns, self.right_seeds, self.right_count)):
            if count < 1 or len(returns) != count or len(seeds) != count:
                raise ValueError("Every prospectively registered repetition needs a recorded return and seed")
            if not all(math.isfinite(x) for x in returns):
                raise ValueError("Missing forced-return labels cannot be replaced with nonfinite values")
        if len(set(self.left_seeds + self.right_seeds)) != self.left_count + self.right_count:
            raise ValueError("Forced repetitions require independent random streams")

    @property
    def difference(self):
        # Eq. (22): delta_hat_ab = (1/n_a) sum_s G_{a,s}
        #                         - (1/n_b) sum_s G_{b,s}.
        # Counts are prospective, so unsuccessful observed repetitions remain
        # here; a missing log cannot be dropped or substituted with zero.
        return math.fsum(self.left_returns) / self.left_count - math.fsum(self.right_returns) / self.right_count


def validate_group(group: Sequence[SearchTrajectory]) -> None:
    if len(group) < 2:
        raise ValueError("A GVO group requires at least two isolated complete searches")
    if len({x.group_id for x in group}) != 1 or len({x.initial_context_id for x in group}) != 1:
        raise ValueError("Group searches must share exactly one group and fixed initial context")
    if len({x.search_id for x in group}) != len(group) or len({x.seed for x in group}) != len(group):
        raise ValueError("Group members need distinct search identities and random streams")
    if len({x.initial_f for x in group}) != 1 or any(x.audit_only for x in group):
        raise ValueError("Group initial F must agree and audit tasks cannot train roles")
    if any(x.forced for x in group):
        raise ValueError("Entire forced continuations train only the scorer, never the actor")
    if any(x.demonstration for x in group):
        raise ValueError("Demonstration-policy searches cannot enter a GVO actor group")
    for name in ("collection_snapshot", "contract_id", "initial_budget", "workers"):
        if len({getattr(x, name) for x in group}) != 1:
            raise ValueError(f"Group searches must share the same {name}")


def collect_independent_group(initial_context: Any, rollout: Callable, *, group_size: int, seed: int,
                              group_id: str, initial_context_id: str) -> list[SearchTrajectory]:
    """Rollout(state, rng, metadata) must run a whole search with frozen roles.

    Eq. (4) separates the inner searches from the later outer update:
        (c_{k+1}^{(g)}, D_k^{(g)}) = Search_{Theta_k}(c_k), g=1,...,G,
        D_k = union_{g=1}^{G} D_k^{(g)},
        Theta_{k+1} = GVO(Theta_k,D_k).
    Theta_k=(theta_k,psi_k,phi_k,nu) holds the Drafter, Policy, Scorer and fixed
    menu settings. Each rollout returns a complete trajectory D_k^{(g)} with
    its accepted output/costs; the resulting list is the collection passed to
    the trainer. No outer optimizer step occurs inside this collection loop.
    G=group_size counts independent searches, not descendants of a shared tree.
    All members retain the same initial root/evidence and registered resources;
    fresh seeds only change their independently sampled subsequent actions.
    State is deep-copied before EACH rollout. This avoids accidental sharing of
    paid discoveries through the passed state; rollout must not use a global ledger.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    seed_rng = random.Random(seed)
    seeds = seed_rng.sample(range(2**63 - 1), group_size)
    group = []
    for i, member_seed in enumerate(seeds):
        metadata = {"seed": member_seed, "group_id": group_id, "initial_context_id": initial_context_id,
                    "search_id": f"{group_id}/{i}"}
        group.append(rollout(copy.deepcopy(initial_context), random.Random(member_seed), metadata))
    validate_group(group)
    return group


def split_by_lineage(records: Sequence[Any], validation_fraction: float = .2, seed: int = 0):
    """All edits, descendant contexts, audits and searches of a lineage stay together."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    def get(x, key, default=None):
        return x.get(key, default) if isinstance(x, dict) else getattr(x, key, default)
    if any(not get(x, "lineage_id") for x in records):
        raise ValueError("Every record needs a stable root edit-lineage ID")
    protected = {get(x, "lineage_id") for x in records if get(x, "audit_only", False)}
    lineages = sorted({get(x, "lineage_id") for x in records} - protected)
    random.Random(seed).shuffle(lineages)
    n = min(len(lineages) - 1, max(1, round(len(lineages) * validation_fraction))) if len(lineages) > 1 and validation_fraction else 0
    validation_ids = set(lineages[:n])
    train, validation, audit = [], [], []
    for x in records:
        target = audit if get(x, "lineage_id") in protected else validation if get(x, "lineage_id") in validation_ids else train
        target.append(x)
    return train, validation, audit


def _decision(raw: dict) -> DecisionRecord:
    raw = dict(raw)
    for key in ("support", "state_features", "input_ids", "behavior_probabilities", "exploration_probabilities", "graph_node_texts"):
        if key in raw:
            raw[key] = tuple(raw[key])
    if "option_features" in raw:
        raw["option_features"] = tuple(tuple(v) for v in raw["option_features"])
    for key in ("graph_nodes", "graph_edges", "option_node_masks", "graph_node_text_features", "graph_node_input_ids"):
        if key in raw:
            raw[key] = tuple(tuple(v) for v in raw[key])
    if 'option_item_features' in raw:
        raw['option_item_features'] = tuple(tuple(tuple(v) for v in items) for items in raw['option_item_features'])
    return DecisionRecord(**raw)


def trajectory_from_dict(raw: dict) -> SearchTrajectory:
    raw = dict(raw)
    raw.pop("record_type", None)
    raw["transitions"] = tuple(TransitionRecord(**t) for t in raw["transitions"])
    raw["decisions"] = tuple(_decision(d) for d in raw["decisions"])
    return SearchTrajectory(**raw)


def load_jsonl(path: str | Path) -> list[dict]:
    result = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return result


def save_jsonl(path: str | Path, records: Sequence[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            raw = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            kind = {SearchTrajectory: "trajectory", CheckerAudit: "audit", EditExample: "edit"}.get(type(record))
            if kind:
                raw = {"record_type": kind, **raw}
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
