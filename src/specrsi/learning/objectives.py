"""GVO objectives from Sec. 3.4 and Appendix F.5.

The functions operate on *observed* returns and replayed action probabilities:
their outputs never insert predictions into the trusted outcome ledger.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def mixture_probabilities(values: Tensor, temperature: float = 1., exploration: float = .1,
                          exploration_probabilities: Tensor | None = None) -> Tensor:
    """Eq. (8): normalize policy preferences on one recorded feasible menu.

        pi_psi(omega | S_t, R_t, Omega_t)
          = (1-epsilon) * exp(z_psi(S_t,R_t,omega) / T_pol)
              / sum_{v in Omega_t} exp(z_psi(S_t,R_t,v) / T_pol)
            + epsilon / |Omega_t|.

    ``values`` is the vector z_psi, ``temperature`` is T_pol > 0, and
    ``exploration`` is epsilon. Every vector position corresponds to exactly
    one legal action in the caller's stored Omega_t; ACCEPT's logit is already
    fixed to zero by the policy head. R_t denotes observed remaining-work
    relations, not a return or a reward. Logits are preferences; the Scorer's
    Q_phi estimates are separate outputs and are not substituted here.

    Softmax implements the exponential ratio with its numerically stable
    normalization. The default u vector is 1/|Omega_t|, so every legal action
    receives epsilon/|Omega_t| exploration mass. A supplied nonuniform u is
    the explicit extension pi=(1-epsilon)*softmax(z/T)+epsilon*u; its exact
    recorded probabilities must also be reused in replay.

    For an edit token, Eq. (19) uses the corresponding actual distribution
    after the token mask and temperature have been applied. A new top-k set
    must not replace that recorded support during Eq. (20)'s numerator replay.
    """
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("Scores must be one finite vector on the recorded feasible support")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= exploration <= 1:
        raise ValueError("Invalid collection temperature or exploration probability")
    probs = torch.softmax(values / temperature, dim=-1)
    u = torch.full_like(probs, 1 / probs.numel()) if exploration_probabilities is None else exploration_probabilities.to(probs)
    if u.shape != probs.shape or (u <= 0).any() or not torch.isclose(u.sum(), u.new_tensor(1.), atol=1e-6):
        raise ValueError("Exploration needs positive normalized probability on every legal action")
    return (1 - exploration) * probs + exploration * u


def group_advantages(returns: Tensor, estimator: str = "gvo", epsilon: float = 1e-8) -> Tensor:
    """Eq. (13): compare full-search returns with a leave-one-out baseline.

        R_g = F(S_end^g) - F(S_0^-) - lambda*C_g - mu*L_g,
        A_g = R_g - (1/(G-1)) * sum_{u=1, u!=g}^{G} R_u.

    ``returns[g]`` already contains R_g from SearchTrajectory.full_return.
    G is the count of independent completed searches from the same root,
    contract, resource budgets, worker count and frozen improver. A shared
    tree's leaves are not independent group members. C_g includes preparation,
    failed attempts and unfinished work; L_g is elapsed time on one host clock.
    lambda and mu convert their respective units to the units of F.

    For R=[1,2,5], the excluded-member baselines are [3.5,3,1.5], giving
    A=[-2.5,-1,3.5]. Equal returns give exactly zero group advantages. There is
    no learned state-value baseline and no normalization by trajectory length.
    The GRPO branch explicitly uses a different, standardized estimator.
    Returns and A_g are constants in Eq. (23), hence the detach below.
    """
    if returns.ndim != 1 or returns.numel() < 2 or not torch.isfinite(returns).all():
        raise ValueError("Need at least two finite complete-search returns")
    returns = returns.detach()
    # Translation leaves the leave-one-out advantage unchanged. Subtract the
    # first return before summing, so a large shared acquisition/cost baseline
    # cannot erase a small observed improvement through cancellation. The
    # trainer supplies float64 labels *before* this operation; upcasting already
    # rounded float32 returns would not recover their lost improvement signal.
    shifted = returns - returns[0]
    centered = shifted - shifted.mean()
    if estimator in {"gvo", "rloo"}:
        # Algebraically G/(G-1) * (R_g - mean(R)), exactly Eq. (13).
        return centered * (returns.numel() / (returns.numel() - 1))
    if estimator == "grpo":
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("GRPO normalization epsilon must be positive and finite")
        return centered / (shifted.std(unbiased=False) + epsilon)
    raise ValueError("estimator must be gvo, rloo, or grpo")


def clipped_surrogate(new_logprob: Tensor, behavior_logprob: Tensor, advantage: Tensor, clip_epsilon: float = .2) -> Tensor:
    """Compute the unnegated per-choice contribution to the Eq. (20) surrogate.

        rho_{g,j}(vartheta)
          = p_vartheta(y_{g,j}|H_{g,j},Y_{g,j})
              / p_vartheta_k(y_{g,j}|H_{g,j},Y_{g,j}),
        L_GVO(vartheta) = -(1/G) * sum_{g=1}^{G} sum_{j=1}^{J_g}
          min(rho_{g,j}*A_g,
              clip(rho_{g,j},1-epsilon_c,1+epsilon_c)*A_g),
        clip(x,l,u) = min(max(x,l),u).

    ``new_logprob`` is log of the current numerator, ``behavior_logprob`` is
    the stored collection denominator's log, and ``advantage`` is Eq. (13)'s
    A_g. Therefore exp(new_logprob-behavior_logprob) computes rho without
    multiplying autoregressive token probabilities into a second edit ratio.
    ``clip_epsilon`` is epsilon_c in (0,1), distinct from epsilon in Eq. (8).
    The trainer supplies the leading minus sign, sums sampled choices j and
    divides by G. It never divides by J_g: each sampled token or plan receives
    its full search's advantage once; a search with J_g=0 contributes zero.

    H and Y must be identical for numerator and denominator, including parent,
    observed history, prefix, legal plan menu, mask, and sampling transforms.
    Only the current numerator differentiates. Old probabilities and A_g are
    detached even when this helper is called separately from the trainer.
    Failed sampled edits keep their likelihood; deterministic solver choices,
    evaluator results and every choice inside a forced suffix are excluded.
    """
    if not 0 < clip_epsilon < 1:
        raise ValueError("PPO clipping epsilon must lie in (0, 1)")
    log_ratio = new_logprob - behavior_logprob.detach()
    if not torch.isfinite(log_ratio).all():
        raise ValueError("Nonfinite policy log ratio")
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Policy ratio overflow; inspect stale behavior traces")
    adv = advantage.detach()
    return torch.minimum(ratio * adv, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * adv)


def current_reference_kl(probabilities: Tensor, reference: Tensor) -> Tensor:
    """Eq. (21): exact current-to-reference KL for one logged choice support.

        L_KL(vartheta) = (1/G) * sum_{g=1}^{G} sum_{j=1}^{J_g}
          D_KL(p_vartheta(.|H_{g,j},Y_{g,j}) || p_ref_{g,j}),
        D_KL(p||q) = sum_{y in Y_{g,j}} p(y)*log(p(y)/q(y)).

    This helper returns one D_KL summand; the trainer supplies the outer sums
    and 1/G. ``probabilities`` is the current distribution after its recorded
    sampling transforms. ``reference`` is the frozen q on exactly the same
    support, and must satisfy q(y)>0 wherever p(y)>0. By definition,
    0*log(0/q)=0, so the active mask removes p(y)=0 terms before taking logs;
    it does not invent positive mass or change either distribution.

    The optimization objective is L_GVO + beta_KL*L_KL. beta_KL=0 disables
    this penalty. It is applied once to actor parameters, with q detached,
    and never subtracted from the observed environment R_g or critic G_t.
    """
    if (probabilities.ndim != 1 or probabilities.numel() == 0
            or not torch.isfinite(probabilities).all() or (probabilities < 0).any()
            or not torch.isclose(probabilities.sum(), probabilities.new_tensor(1.), atol=1e-6)):
        raise ValueError("KL current probabilities must be a normalized finite distribution on the recorded support")
    reference = reference.detach().to(probabilities)
    if (reference.shape != probabilities.shape or not torch.isfinite(reference).all()
            or (reference < 0).any() or not torch.isclose(reference.sum(), reference.new_tensor(1.), atol=1e-6)):
        raise ValueError("KL reference must be a normalized finite distribution on the recorded support")
    active = probabilities > 0
    if (reference[active] <= 0).any():
        raise ValueError("Finite exact KL requires positive reference mass wherever current probability is positive")
    return (probabilities[active] * (probabilities[active].log() - reference[active].log())).sum()


def regression_loss(predicted: Tensor, target: Tensor, kind: str = "mse") -> Tensor:
    """Eqs. (11) and (12): squared errors against observed return labels.

    Eq. (11), for N_Q=|D_Q| actual plan-start records:
        G_t = F(S_end)-F(S_t)-lambda*C_{t:end}-mu*L_{t:end},
        L_Q(phi) = E_{(S_t,omega_t,G_t)~D_Q}
                     [(Q_hat_phi(S_t,omega_t)-G_t)^2]
                 = (1/N_Q) * sum_{(S,omega,G) in D_Q}
                     (Q_hat_phi(S,omega)-G)^2.
    Thus predicted=Q_hat_phi(S_t,omega_t) and target=G_t for the value call.

    Eq. (12), with observed arm means over paid repetitions:
        delta_hat_ab = G_bar_a-G_bar_b,
        L_pair(phi) = E_{D_pair}
          [(Q_hat_phi(S,a)-Q_hat_phi(S,b)-delta_hat_ab)^2]
          = (1/|D_pair|) * sum_{(S,a,b,G_bar_a,G_bar_b) in D_pair}
              (Q_hat_phi(S,a)-Q_hat_phi(S,b)-(G_bar_a-G_bar_b))^2.
    The pair call supplies predicted=Q_hat_phi(S,a)-Q_hat_phi(S,b) and
    target=delta_hat_ab. The trainer averages scalar calls over each dataset
    separately, so a paid forced tuple has the same L_Q weight as an ordinary
    tuple. Its return remains an observation, not a bootstrapped prediction.
    L_value=L_Q+beta_pair*L_pair uses one scorer; the pair term cancels a
    common Q offset, while L_Q anchors it. ``mse`` implements these equations;
    the explicitly selected Huber branch uses a different regression penalty.
    Observed labels are detached before either loss evaluates its gradient.
    """
    target = target.detach()
    if kind == "mse":
        return F.mse_loss(predicted, target)
    if kind == "huber":
        return F.smooth_l1_loss(predicted, target)
    raise ValueError("Regression loss must be explicitly declared as mse or huber")


def binary_calibration(logits: Tensor, labels: Tensor, bins: int = 10) -> dict[str, float]:
    if logits.numel() == 0:
        return {"count": 0}
    p = logits.detach().sigmoid().flatten()
    z = labels.detach().float().flatten()
    ece = p.new_zeros(())
    for i in range(bins):
        mask = (p >= i / bins) & ((p < (i + 1) / bins) if i + 1 < bins else (p <= 1))
        if mask.any():
            ece += mask.float().mean() * (p[mask].mean() - z[mask].mean()).abs()
    return {"count": int(p.numel()), "log_loss": float(F.binary_cross_entropy_with_logits(logits.flatten(), z)),
            "brier": float(((p - z) ** 2).mean()), "ece": float(ece),
            "accuracy_at_0.5": float(((p >= .5) == z.bool()).float().mean())}
