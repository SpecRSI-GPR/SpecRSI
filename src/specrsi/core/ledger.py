"""Evidence admission, scoped identity validation and complete task-score aggregation.

Eq. (3) separates evidence completeness from hard-constraint validity:
    m_ci=1 iff a valid required outcome record exists,
    C_full(S)={c in C : m_ci=1 for all i in I, c admissible under Gamma},
    F(S)=max_{c in C_full(S)} [J_Gamma(c)-J_Gamma(r)].
I is the fixed required-item registry, Gamma is the immutable evaluation scope,
and r is the complete admissible search root. Missing sets are
    U_c(S)={i in I : m_ci=0}, K_c(S)={(c,i):i in U_c(S)}.
Only admitted observations can change these masks. Predicted task quality,
continuation values, planned coverage and cached unvalidated scalars cannot.

An outcome can be complete and valid evidence even when it reports a failed
constraint. Such a record removes missingness but excludes the candidate from
C_full. ``score`` computes J_Gamma only from the complete ordered outcome vector;
``admissible`` separately enforces all outcome-level hard constraints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hmac
from hashlib import sha256
import json
import math
import secrets
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping

from .checker import ConservativeChecker, ReuseCertificate
from .models import Candidate, EvaluationScope, FrozenMap, Outcome, RecordKey, Task, digest, thaw


class EvidenceConflict(ValueError):
    pass


@dataclass(frozen=True)
class Record:
    candidate_id: str
    task_id: str
    scope_id: str
    outcome: Outcome
    route: str  # U = direct actual execution, P = checked inherited outcome
    source: RecordKey | None = None
    certificate: ReuseCertificate | None = None
    receipt: FrozenMap = FrozenMap()
    signature: str = ""

    @property
    def key(self) -> RecordKey:
        return self.candidate_id, self.task_id, self.scope_id

    def manifest(self) -> dict:
        return {"candidate_id": self.candidate_id, "task_id": self.task_id, "scope_id": self.scope_id,
                "outcome": self.outcome.manifest(), "route": self.route, "source": self.source,
                "certificate": asdict(self.certificate) if self.certificate else None,
                "receipt": thaw(self.receipt), "signature": self.signature}


Aggregator = Callable[[tuple[Outcome, ...], EvaluationScope], float]


class Ledger:
    """Host evidence store with keyed HMAC validation of admitted direct records.

    The signature binds the record key, complete outcome and execution receipt.
    Reading a direct record recomputes this signature before using its evidence;
    inherited records instead validate their scoped equality and source chain.

    Direct admission is a trusted evaluator API. Editable candidates must never
    receive access to the ledger, its secret, checker, or acceptance implementation.
    """
    def __init__(self, checker: ConservativeChecker | None = None, *, receipt_secret: bytes | None = None,
                 aggregators: Mapping[str, Aggregator] | None = None):
        self.checker = checker or ConservativeChecker()
        self._secret = secrets.token_bytes(32) if receipt_secret is None else receipt_secret
        if len(self._secret) < 16:
            raise ValueError("receipt secret must contain at least 16 bytes")
        self._candidates: dict[str, Candidate] = {}
        self._scopes: dict[str, EvaluationScope] = {}
        self._records: dict[RecordKey, Record] = {}
        self._quarantined: set[RecordKey] = set()
        self.conflicts: list[tuple[Record, Record]] = []
        self._aggregators = {"weighted_mean_v1": lambda ys, s: math.fsum(w * y.score for w, y in zip(s.weights, ys)),
                             "all_success_v1": lambda ys, s: float(all(y.score == 1 for y in ys))}
        for key, aggregator in (aggregators or {}).items():
            if key in self._aggregators:
                raise ValueError("built-in aggregator identities cannot be replaced")
            self._aggregators[key] = aggregator
        self._lock = RLock()

    @property
    def records(self) -> Mapping[RecordKey, Record]:
        return MappingProxyType(self._records)

    @property
    def candidates(self) -> Mapping[str, Candidate]:
        return MappingProxyType(self._candidates)

    @property
    def quarantined(self) -> frozenset[RecordKey]:
        return frozenset(self._quarantined)

    def register(self, candidate: Candidate, scope: EvaluationScope) -> None:
        with self._lock:
            self._candidates[candidate.uid] = candidate
            self._scopes[scope.uid] = scope

    def clone(self) -> "Ledger":
        """Independent host ledger with the same immutable authenticated evidence.

        This does not snapshot an external executor's filesystem or tool state.
        Continuation callers must also supply an isolated execution environment.
        """
        with self._lock:
            clone = Ledger(self.checker, receipt_secret=self._secret)
            clone._aggregators = dict(self._aggregators)
            clone._candidates = dict(self._candidates)
            clone._scopes = dict(self._scopes)
            clone._records = dict(self._records)
            clone._quarantined = set(self._quarantined)
            clone.conflicts = list(self.conflicts)
            return clone

    def _signature(self, key: RecordKey, outcome: Outcome, receipt: FrozenMap) -> str:
        message = digest({"key": key, "outcome": outcome.manifest(), "receipt": receipt})
        return hmac.new(self._secret, message.encode(), sha256).hexdigest()

    def _put(self, record: Record) -> Record:
        if record.key in self._quarantined:
            raise EvidenceConflict("record is quarantined")
        old = self._records.get(record.key)
        if old is not None:
            if old.outcome != record.outcome:
                self._quarantined.add(record.key)
                self.conflicts.append((old, record))
                raise EvidenceConflict("conflicting outcomes quarantined; never average or select the better score")
            return old
        self._records[record.key] = record
        return record

    def admit_direct(self, candidate: Candidate, task: Task, scope: EvaluationScope,
                     outcome: Outcome, receipt: Mapping) -> Record:
        with self._lock:
            if not scope.contains(task):
                raise ValueError("task outside scope")
            outcome.validate(scope)
            self.register(candidate, scope)
            receipt = FrozenMap(receipt)
            key = candidate.uid, task.uid, scope.uid
            record = Record(*key, outcome, "U", receipt=receipt,
                            signature=self._signature(key, outcome, receipt))
            return self._put(record)

    def inherit(self, source: Candidate, child: Candidate, task: Task, scope: EvaluationScope,
                certificate: ReuseCertificate) -> Record:
        with self._lock:
            source_record = self.get(source, task, scope)
            if source_record is None:
                raise ValueError("equality without valid source evidence supplies no outcome")
            if not self.checker.verify(certificate, source, child, task, scope):
                raise ValueError("certificate does not establish equality of complete scoped effects")
            self.register(child, scope)
            if source.uid == child.uid:
                return source_record
            return self._put(Record(child.uid, task.uid, scope.uid, source_record.outcome, "P", source_record.key, certificate))

    def validate_key(self, key: RecordKey, visiting: set[RecordKey] | None = None) -> bool:
        visiting = set() if visiting is None else visiting
        if key in visiting or key in self._quarantined or key not in self._records:
            return False
        visiting.add(key)
        try:
            record = self._records[key]
            if record.key != key:
                return False
            candidate, scope = self._candidates[record.candidate_id], self._scopes[record.scope_id]
            task = next(t for t in scope.tasks if t.uid == record.task_id)
            record.outcome.validate(scope)
            if record.route == "U":
                return (record.source is None and record.certificate is None
                        and hmac.compare_digest(record.signature, self._signature(key, record.outcome, record.receipt)))
            if record.route != "P" or record.source is None or record.certificate is None or record.signature or record.receipt:
                return False
            source = self._records.get(record.source)
            return (source is not None and source.task_id == task.uid and source.scope_id == scope.uid
                    and self.validate_key(source.key, visiting) and source.outcome == record.outcome
                    and self.checker.verify(record.certificate, self._candidates[source.candidate_id], candidate, task, scope))
        except (KeyError, StopIteration, ValueError, TypeError):
            return False
        finally:
            visiting.remove(key)

    def get(self, candidate: Candidate, task: Task, scope: EvaluationScope) -> Record | None:
        with self._lock:
            key = candidate.uid, task.uid, scope.uid
            return self._records[key] if self.validate_key(key) else None

    def missing(self, candidate: Candidate, scope: EvaluationScope) -> tuple[Task, ...]:
        # Eq. (3), completeness mask and required missing items:
        # m_ci=1 iff a Valid_Gamma record exists, and U_c={i:m_ci=0}.
        # A complete record reporting failure therefore removes missingness;
        # admissibility is a separate predicate checked below.
        return tuple(t for t in scope.tasks if self.get(candidate, t, scope) is None)

    def complete(self, candidate: Candidate, scope: EvaluationScope) -> bool:
        # Eq. (3), the evidence conjunct only:
        # complete(c) <=> U_c(S)=empty <=> m_ci=1 for every i in I.
        # A complete version can still violate a hard requirement.
        return not self.missing(candidate, scope)

    def outcomes(self, candidate: Candidate, scope: EvaluationScope) -> tuple[Outcome, ...]:
        records = [self.get(candidate, t, scope) for t in scope.tasks]
        if any(r is None for r in records):
            raise ValueError("partial candidates do not have complete aggregate scores")
        return tuple(r.outcome for r in records)

    def score(self, candidate: Candidate, scope: EvaluationScope) -> float:
        # J_Gamma(c) in Eqs. (3), (6), (10)--(13): invoke the fixed contracted
        # aggregator only after ALL required fields/items exist. Aggregation may
        # be nonlinear. The declared operating penalty is subtracted exactly once
        # here, so callers must not subtract it again when forming F(S) or targets.
        # For the weighted-mean aggregation specifically:
        # J_Gamma(c)=sum_{i in I} w_i*y_ci.score - operating_penalty(c),
        # with w_i>=0 and sum_i w_i=1. Other registered aggregators can be
        # nonlinear; completion and acceptance do not assume additivity.
        outcomes = self.outcomes(candidate, scope)
        if scope.aggregation_id not in self._aggregators:
            raise ValueError(f"unregistered native aggregation: {scope.aggregation_id}")
        value = float(self._aggregators[scope.aggregation_id](outcomes, scope)) - candidate.operating_penalty
        if not math.isfinite(value):
            raise ValueError("native aggregation returned a nonfinite score")
        return value

    def admissible(self, candidate: Candidate, scope: EvaluationScope) -> bool:
        # Eq. (3): evidence completeness AND contracted hard-constraint validity.
        # A score prediction cannot satisfy either conjunct.
        return self.complete(candidate, scope) and all(y.admissible for y in self.outcomes(candidate, scope))

    def audit_provenance(self) -> int:
        with self._lock:
            for key in self._records:
                if key not in self._quarantined and not self.validate_key(key):
                    raise ValueError(f"invalid provenance: {key}")
            return sum(self.validate_key(key) for key in self._records)

    def to_json(self) -> str:
        """Secret is intentionally omitted; restoration needs the host-held key."""
        data = {"schema": "specrsi-ledger-v1", "candidates": [c.manifest() for c in self._candidates.values()],
                "scopes": [s.manifest() for s in self._scopes.values()],
                "records": [r.manifest() for r in self._records.values()],
                "quarantined": sorted(self._quarantined),
                "conflicts": [[a.manifest(), b.manifest()] for a, b in self.conflicts]}
        return json.dumps(data, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str, *, receipt_secret: bytes,
                  checker: ConservativeChecker | None = None, aggregators: Mapping[str, Aggregator] | None = None) -> "Ledger":
        def parse_record(raw: dict) -> Record:
            raw = dict(raw)
            raw["outcome"] = Outcome(**raw["outcome"])
            raw["receipt"] = FrozenMap(raw["receipt"])
            raw["source"] = tuple(raw["source"]) if raw["source"] is not None else None
            raw["certificate"] = ReuseCertificate(**raw["certificate"]) if raw["certificate"] else None
            return Record(**raw)

        data = json.loads(text)
        if set(data) != {"schema", "candidates", "scopes", "records", "quarantined", "conflicts"} or data["schema"] != "specrsi-ledger-v1":
            raise ValueError("invalid ledger schema")
        ledger = cls(checker, receipt_secret=receipt_secret, aggregators=aggregators)
        for raw in data["candidates"]:
            c = Candidate(**raw)
            if c.uid in ledger._candidates:
                raise ValueError("duplicate candidate")
            ledger._candidates[c.uid] = c
        for raw in data["scopes"]:
            raw = dict(raw)
            raw["tasks"] = tuple(Task(**t) for t in raw["tasks"])
            scope = EvaluationScope(**raw)
            if scope.uid in ledger._scopes:
                raise ValueError("duplicate scope")
            ledger._scopes[scope.uid] = scope
        for raw in data["records"]:
            record = parse_record(raw)
            if record.key in ledger._records:
                raise ValueError("duplicate record")
            ledger._records[record.key] = record
        ledger._quarantined = {tuple(k) for k in data["quarantined"]}
        if not ledger._quarantined.issubset(ledger._records):
            raise ValueError("quarantined identity is unregistered")
        ledger.conflicts = [(parse_record(a), parse_record(b)) for a, b in data["conflicts"]]
        ledger.audit_provenance()
        return ledger
