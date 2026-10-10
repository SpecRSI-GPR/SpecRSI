"""Fixed complete-candidate acceptance, Eqs. (3) and (10).

Eq. (3):
    C_full(S)={c in C : m_ci=1 for every i in I,
                       c admissible under Gamma},
    F(S)=max_{c in C_full(S)} [J_Gamma(c)-J_Gamma(r)].
Eq. (10): ACCEPT(S)=TieArgMax_{c in C_full(S)} J_Gamma(c).
Here r is the complete admissible root, C is the actual immutable version set,
I is the required task-item registry, and J_Gamma is the fixed complete-outcome
aggregation. F is observed adoption-ready improvement in task-score units.
The root belongs to C_full, so F>=0, including when no revision is eligible.
The bit m_ci records whether a valid outcome exists, independently of passing.
A complete version that violates hard constraints remains ineligible; an
incomplete version can still parent a later DRAFT. Scorer/policy predictions
do not enter evidence admission or task-score aggregation. VATS fixes margin=0;
other callers may explicitly request the legacy nonzero-margin behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from .ledger import Ledger
from .models import Candidate, EvaluationScope


@dataclass(frozen=True)
class AcceptanceResult:
    selected: Candidate
    incumbent_score: float
    selected_score: float
    best_complete_score: float
    improved: bool
    margin: float
    complete_candidates: tuple[str, ...]


def accept(incumbent: Candidate, candidates: Iterable[Candidate], ledger: Ledger,
           scope: EvaluationScope, margin: float = 0.,
           hard_constraint: Callable[[Candidate], bool] | None = None) -> AcceptanceResult:
    """Apply Eq. (10): ACCEPT(S)=TieArgMax_{c in C_full(S)} J_Gamma(c).

    ``incumbent`` is r, ``scope`` defines Gamma, and the ledger supplies contracted
    scores. Evidence completeness is checked before aggregating. A hard_constraint
    is an additional fixed host predicate, independent of the learned policy.

    TieArgMax first maximizes the score. If r ties at that maximum, retain r;
    otherwise choose the lexicographically smallest immutable uid among maximizing
    versions. Thus root preference never overrides a strictly higher score. With
    margin=0, ``improved`` is exactly J_Gamma(selected)>J_Gamma(r), and every other
    visited version may remain incomplete at termination.
    """
    if scope.split == "audit":
        raise ValueError("independent audit tasks cannot select candidate versions")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("acceptance margin must be finite and nonnegative")
    predicate = hard_constraint or (lambda c: True)
    if not ledger.admissible(incumbent, scope) or not predicate(incumbent):
        raise ValueError("search requires a complete, admissible incumbent")
    options = {c.uid: c for c in candidates}
    options[incumbent.uid] = incumbent
    complete = [c for c in options.values() if ledger.admissible(c, scope) and predicate(c)]
    # Eq. (10): root wins only when tied for the BEST score. If several improved
    # candidates tie above the root, lexicographic immutable identity is the fixed
    # version order. The ``improved`` branch below implements this root preference
    # without adding a hidden adoption margin in the VATS path (margin=0).
    best = min(complete, key=lambda c: (-ledger.score(c, scope), c.uid))
    original, best_score = ledger.score(incumbent, scope), ledger.score(best, scope)
    improved = best_score > original + margin
    selected = best if improved else incumbent
    return AcceptanceResult(selected, original, ledger.score(selected, scope), best_score, improved, margin,
                            tuple(sorted(c.uid for c in complete)))


def complete_improvement(incumbent: Candidate, candidates: Iterable[Candidate], ledger: Ledger,
                         scope: EvaluationScope, hard_constraint: Callable[[Candidate], bool] | None = None) -> float:
    # Eq. (3): F(S)=max_{c in C_full(S)} [J_Gamma(c)-J_Gamma(r)] >= 0.
    # The root is eligible by construction; partial scores or forecasts are never
    # used as a substitute for this observed adoption-ready improvement.
    result = accept(incumbent, candidates, ledger, scope, 0., hard_constraint)
    return max(0., result.best_complete_score - result.incumbent_score)
