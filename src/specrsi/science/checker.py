"""Pair-dependent equality for the registered science parameter kernel only."""
import math
import time
from specrsi.core.checker import ConservativeChecker,ReuseCertificate,CheckResult
from specrsi.core.budget import cost_int
from specrsi.core.models import digest
from .executor import selected_parameters


class ScienceParameterChecker(ConservativeChecker):
    version="specrsi-science-selected-parameters-checker-v2"

    def __init__(self):
        super().__init__()
        self.physical_checks=[]

    def check(self,source,child,task,scope,cap):
        cost_int(cap)
        if cap==0:
            return CheckResult("interrupted",0,reason="No checker wall-time slice")
        started=time.monotonic()
        certificate=self._proof(source,child,task,scope)
        elapsed=time.monotonic()-started
        cost=max(1,math.ceil(elapsed*1000))
        self.physical_checks.append({"operation":"check","wall_seconds":elapsed,"charged_ms":cost,"cap_ms":cap})
        if cost>cap:
            # Actual paid work is never clipped. Acquisition will explicitly
            # record a cap violation and reject the overrun certificate.
            return CheckResult("interrupted",cost,reason="Cooperative checker deadline exceeded; actual overrun retained")
        return CheckResult("P" if certificate else "U",cost,certificate,"registered selected-parameter equality" if certificate else "unsupported or unequal native effects")

    def verify(self,certificate,source,child,task,scope):
        started=time.monotonic()
        valid=certificate==self._proof(source,child,task,scope)
        self.physical_checks.append({"operation":"provenance_verify","wall_seconds":time.monotonic()-started,
                                     "charged_ms":0,"accounting":"host ledger overhead included in end-to-end wall time"})
        return valid

    def _proof(self,source,child,task,scope):
        if source.runtime!=child.runtime:
            return None
        try:
            a=selected_parameters(source,task,scope)
            b=selected_parameters(child,task,scope)
        except (KeyError,ValueError,TypeError,RecursionError):
            return None
        if a!=b:
            return None
        evidence={"selected_parameters":a,"runtime":source.runtime,"scope_id":scope.uid,"task_id":task.uid,
                  "native_identity":task.metadata["evaluator_identity"],"scored_fields":scope.scored_fields}
        return ReuseCertificate(self.version,"registered-native-selected-parameters-v2",source.uid,child.uid,task.uid,scope.uid,digest(evidence))
