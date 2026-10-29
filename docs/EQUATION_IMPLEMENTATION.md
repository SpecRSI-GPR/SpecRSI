# Equations and executable operations

This equation-to-code index uses the numbering of its source implementation document. Current manuscript settings and protocols are on the [results page](results/README.md); the code locations below provide implementation navigation.

Reference identity for the implementation document: **Qwen3.5-35B-A3B**,
publisher BF16 revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
`learning/hf.py` implements Appendix F.5 with the native MoE text tower,
rank-16 LoRA on both full-attention and Gated DeltaNet projections, and
independent single-Q and policy relation heads (width 256, four heads).
Experts and routing weights remain frozen. `paper_profile.py` validates
this identity before a reference run. The exact executable configuration is
[`qwen35_paper33_science.yaml`](../configs/training/qwen35_paper33_science.yaml).
SFT projects only supervised hidden positions to the vocabulary; its edit
log-likelihood is still the **sum over all supervised tokens**, including EOS,
as required by Eq. (18). Native adapter reload checks every tensor name/shape.

The search state is `S = (T, E, H, B, N, w)`: an immutable version tree,
admitted observations, observed history, remaining expense, remaining primitive
attempts, and worker count. `theta` generates edits, `psi` selects work, and
`phi` estimates continuation returns. The three parameter roles are frozen
during collection and updated between rounds. `Gamma` fixes the evaluator.

The implementation comments contain all numbered main-text equations **(1)–(13)**
and the selected appendix equations **(14)–(23)**. They spell out each expression,
its variables, the corresponding operations, and the accounting or gradient
rules. The table is a navigation index; the full explanations are beside the
implementations. English technical comments also define the unnumbered state,
missing-item sets, relation reader, budget reservation, and bundle costs.

| Equation | Calculation | Executable location |
| --- | --- | --- |
| (1) | `c_tilde_t ~ Propose_Theta_t(. given V_t,H_t)`; `z_t=f_real(c_tilde_t)`; `(V_next,H_next)=Update(V_t,H_t,c_tilde_t,z_t)` | [core/search.py](../src/specrsi/core/search.py): module, `SearchState`, `_draft`; [core/vats.py](../src/specrsi/core/vats.py): `_execute_plan` |
| (2) | `f_real(c)=(J(c),v(c))`; `c_star_n in argmax_{(c,J,v) in H_n:v=1} J` | [core/acquisition.py](../src/specrsi/core/acquisition.py): module, `_acquire`; [core/ledger.py](../src/specrsi/core/ledger.py): `score`, `admissible`; [core/acceptance.py](../src/specrsi/core/acceptance.py): `accept` |
| (3) | `C_full = {c: m_ci=1 for every i, c admissible}`; `F(S)=max_c[J_Gamma(c)-J_Gamma(r)]` | `core/ledger.py`: `complete`, `admissible`, `score`; `core/acceptance.py`: `complete_improvement`; `core/vats.py`: `VATSState.features` |
| (4) | `(c_next_g,D_k_g)=Search_Theta_k(c_k), g=1,...,G`; `D_k=union_g D_k_g`; `Theta_next=GVO(Theta_k,D_k)` | [scripts/train.py](../scripts/train.py): module, collection/update loop; `learning/records.py`: `collect_independent_group`, `validate_group`; `learning/continuations.py`: `FrozenRoleGuard` |
| (5) | `Omega = drafts union verification_bundles union {ACCEPT}` | `core/vats.py`: `VATSEngine._prepare` |
| (6) | `Q^pi(S_t,omega)=E[F(S_end)-F(S_t)-lambda*C_t:end-mu*L_t:end given S_t,omega]`; `Q^pi(S_t,ACCEPT)=0` | `core/vats.py`: `_prepare`; `learning/records.py`: `SearchTrajectory.target` |
| (7) | `Delta(j given A,S_tilde)=Q(S_tilde,VERIFY(A union D_j))-Q(S_tilde,VERIFY(A))` | `core/vats.py`: `_prepare` whole-bundle refinement |
| (8) | `pi_psi(omega)=(1-epsilon)*softmax(z_psi/T)_omega+epsilon/len(Omega)` | `planning/policy.py`: `menu_distribution`, `choose`; `learning/objectives.py`: `mixture_probabilities` |
| (9) | `c_a,0=P_a`; sample `e_a,d~pi_theta(. given c_a,d-1,H_a,d)`; `c_a,d=Freeze(Apply(c_a,d-1,e_a,d))`, `d=1,...,h` | `core/vats.py`: `_execute_plan`; `learning/sampler.py`: `sample_edit_tokens`; `learning/science_bridge.py`: `HFParameterDrafter.draft` |
| (10) | `ACCEPT(S)=TieArgMax_{c in C_full} J_Gamma(c)` | `core/acceptance.py`: `accept` |
| (11) | `G_t=F(S_end)-F(S_t)-lambda*C_t:end-mu*L_t:end`; `L_Q=mean[(Q-G_t)^2]` | `learning/records.py`: `SearchTrajectory.target`; `learning/objectives.py`: `regression_loss`; `learning/trainer.py`: `_gvo_epoch_frozen` |
| (12) | `delta_ab=mean(G_a)-mean(G_b)`; `L_pair=mean[(Q(S,a)-Q(S,b)-delta_ab)^2]` | `learning/records.py`: `PairwiseValueExample`; `learning/trainer.py`: `_pair_loss` |
| (13) | `R_g=F(S_end^g)-F(S_0^-)-lambda*C_g-mu*L_g`; `A_g=R_g-sum_{u!=g}(R_u)/(G-1)` | `learning/records.py`: `SearchTrajectory.full_return`; `learning/objectives.py`: `group_advantages` |
| (14) | Minimize capped job plus shared setup expense, with binary coverage, dependency, conflict, setup, budget and attempt constraints | `planning/completion.py`: `solve_completion` |
| (15) | Independently validate `K subset K(A)`, `D(A)=A`, `C(A)<=B_plan`, `n(A)<=N` | `planning/completion.py`: `CompletionProblem.feasible` |
| (16) | `B_t=B_t^- -o_t`; `B_{t+1}^-=B_t-kappa_t`; `N_{t+1}=N_t-n_t`; `r_t^m=-lambda*o_t-mu*l_t^m`; `r_t^w=F(S_{t+1}^-)-F(S_t)-lambda*kappa_t-mu*l_t^w` | `core/vats.py`: `run`, `_execute_plan`; `learning/records.py`: `TransitionRecord` |
| (17) | `G_t=r_t^w+sum_{u>t}(r_u^m+r_u^w)` | `learning/records.py`: `SearchTrajectory.target` |
| (18) | `L_SFT=-mean_edit(log pi_theta(e given H))-mean_plan(log pi_psi(omega given S,R,Omega))` | `learning/hf.py`: `edit_nll`; `learning/trainer.py`: `warmup_epoch` |
| (19) | `p_vartheta(y given H,Y)=pi_psi(omega given S,R,Omega)` for plans, or `pi_theta(y given H,Y)` for edit tokens; `vartheta=(theta,psi)` | `learning/records.py`: `DecisionRecord`; `learning/trainer.py`: `_actor_terms`, `_probabilities` |
| (20) | `L_GVO=-(1/G)*sum_g sum_j min(rho_gj*A_g,clip(rho_gj,1-eps_c,1+eps_c)*A_g)` | `learning/objectives.py`: `clipped_surrogate`; `learning/trainer.py`: `_gvo_epoch_frozen` |
| (21) | `L_KL=(1/G)*sum_g sum_j sum_y p(y)*log[p(y)/p_ref(y)]` | `learning/objectives.py`: `current_reference_kl`; `learning/reference_policy.py`: `FrozenActorReference` |
| (22) | `G_bar_a=(1/n_a)*sum_s G_a,s`; `delta_hat_ab=G_bar_a-G_bar_b`; conditional independent-arm `Var(delta_hat_ab given S)=sigma_a^2/n_a+sigma_b^2/n_b` | `learning/continuations.py`: `collect_paid_continuations`; `learning/records.py`: `PairwiseValueExample`, `difference` |
| (23) | `phi_next=phi-eta_phi*grad_phi(L_Q+beta_pair*L_pair)`; `vartheta_next=vartheta-eta_pi*grad_vartheta(L_GVO+beta_KL*L_KL)` | `learning/trainer.py`: `_make_optimizer`, `_gvo_epoch_frozen` with `pdf_gradient_v1` |

Unlinked module paths in this table are relative to `src/specrsi`. Main-text
equations (1)–(2) define the general improvement/evaluation setting: VATS
implements its version and feedback updates through interleaved DRAFT/VERIFY
operations. A draft does not impose an immediate full-evaluation barrier. The
subscript `n` counts real-evaluation attempts, independently of expense, primitive
allowance `N`, and independent-search group size `G`. Equations (24)–(29) are
analytical cost comparisons, rather than extra losses or a replacement search
controller.

Detailed comments are concentrated in [search and execution](../src/specrsi/core/vats.py),
[completion constraints](../src/specrsi/planning/completion.py),
[policy probabilities](../src/specrsi/planning/policy.py),
[returns and pairs](../src/specrsi/learning/records.py),
[loss functions](../src/specrsi/learning/objectives.py), and
[training updates](../src/specrsi/learning/trainer.py).

## Complete evidence and acceptance

`m_ci=1` means that the required result exists with valid identity, scope, fields,
and execution evidence. It does not mean that the check passed. A complete failed
candidate can supply feedback to its descendant, while remaining ineligible for
adoption. The initial root is complete and admissible, so `F(S)>=0`.

`Ledger.score` computes `J_Gamma`, including a declared operating penalty once.
Callers use this result directly. Acceptance chooses the highest score over
complete admissible candidates. If the root is among the tied best, it wins;
otherwise the fixed candidate order resolves ties. There is no additional margin.

## Preparation and suffix returns

There are two transitions at each decision. Preparation consumes `o_t`, producing
the paid choice state `S_t`; work consumes `kappa_t`. The whole-search return
includes both charges, including the initial preparation. The suffix target at
`S_t` excludes only its already-paid preparation and includes every later charge.
An ACCEPT choice contributes zero work, after its preparation has been paid.

For example, let preparation/work expenses be `(2,5)` and `(3,0)`, with the second
choice ACCEPT. If the observed gain is `4`, `lambda=0.1`, and `mu=0`, the search
return is `4-0.1*(2+5+3)=3`; the first paid-state suffix is
`4-0.1*(5+3)=3.2`; the terminal ACCEPT suffix is `0`. Subtracting the first
preparation from both the episode and the suffix would count it twice.

Elapsed time follows a single search clock. Concurrent worker durations are
measured separately. Failed calls retain actual charges where measured, and
unresolved work retains explicit conservative accounting. Every launched
primitive consumes attempts, including failed drafts and checks.

## Solver and policy

The completion solver minimizes an enforceable cost cap. Shared setup expense
is counted once per block, even when several candidates use that setup.
Prerequisites enable result-producing jobs but do not themselves satisfy task
evidence requirements. Returned solutions undergo independent feasibility checks.

The menu retains its original affordable completion seeds before optional
positive-marginal extensions. Both terms of each marginal use the same
conservative state and scorer. The policy has independent preferences; its
ACCEPT logit is zero. Collection samples the recorded exploration mixture;
deployment selects the maximum preference. The solver has no actor likelihood.

## Learning and numerical precision

The leave-one-out advantage is computed as
`G/(G-1) * ((R_g-R_0)-mean_u(R_u-R_0))`. This equals Eq. (13), while avoiding
cancellation from a large common cost. The trainer constructs return labels as
float64 before subtracting them. Converting already-rounded float32 labels to
float64 would not recover a lost improvement.

The actor sum is divided by the number of independent searches, not by tokens or
choices. A search with no sampled choices still belongs to the group. Returns,
advantages, old probabilities and reference distributions are detached constants.
The scorer has its own parameters and receives value gradients only.

Each pair uses identical paid-state inputs and an identical recorded menu, with
distinct first plans. One scorer forward returns the whole menu vector; indexing
the two arm values from that vector gives the pair difference with one forward.
Forced rollouts train the scorer only, including their later sampled decisions.
Their acquisition expense remains charged separately from the restored decision
budget. A new outer round collects fresh searches under the updated frozen roles.
