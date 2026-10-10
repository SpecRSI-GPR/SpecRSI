"""Public SpecRSI contracts. Search imports are lazy to avoid planning import cycles."""
from .models import Candidate, EvaluationScope, FrozenMap, Outcome, Task, digest, freeze, thaw
from .budget import BudgetLedger, BudgetExceeded, Charge, CostCapViolation
from .checker import CheckResult, ConservativeChecker, ReuseCertificate
from .execution import ExecutionResult, Executor, PureIntegerExecutor
from .ledger import EvidenceConflict, Ledger, Record
from .acquisition import AcquisitionCaps, AcquisitionResult, ResultAcquirer
from .acceptance import AcceptanceResult, accept, complete_improvement

_SEARCH_NAMES = {"Action", "BatchValue", "DraftResult", "Drafter", "OuterValue", "SearchConfig", "SearchEngine",
                 "SearchResult", "SearchState", "TraceTransition", "ValueEstimate", "heuristic_outer_value", "heuristic_batch_value"}
_VATS_NAMES = {"VATSConfig", "VATSEngine", "VATSState", "VATSPlan", "PaidStateSnapshot", "ScorerQueryFailure"}
_JOB_NAMES = {"JobContext", "ExecutedJobOutcome", "JobExecutionResult", "SetupExecutionResult", "VerificationBackend"}


def __getattr__(name):
    if name in _JOB_NAMES:
        from . import jobs
        return getattr(jobs, name)
    if name in _VATS_NAMES:
        from . import vats
        return getattr(vats, name)
    if name in _SEARCH_NAMES:
        from . import search
        return getattr(search, name)
    raise AttributeError(name)


__all__ = ["Candidate", "EvaluationScope", "FrozenMap", "Outcome", "Task", "digest", "freeze", "thaw",
           "BudgetLedger", "BudgetExceeded", "Charge", "CostCapViolation", "CheckResult", "ConservativeChecker",
           "ReuseCertificate", "ExecutionResult", "Executor", "PureIntegerExecutor", "EvidenceConflict", "Ledger", "Record",
           "AcquisitionCaps", "AcquisitionResult", "ResultAcquirer", "AcceptanceResult", "accept", "complete_improvement", *_SEARCH_NAMES, *_VATS_NAMES, *_JOB_NAMES]
