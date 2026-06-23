from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from coscientist.schemas.v17 import CandidateSolution, ScientificProblem, VerifierResult, VerifierStage


class ScientificVerifier(ABC):
    verifier_id: str
    version = "v1"
    stage: VerifierStage = "cheap"
    capability = "generic"

    def verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]) -> VerifierResult:
        started = perf_counter()
        try:
            verdict, score, passed, failed, assumptions, counterexample = self._verify(candidate, problem, evidence_ids=evidence_ids)
        except Exception as exc:  # defensive boundary: verifier errors should not crash search
            verdict, score, passed, failed, assumptions, counterexample = "error", 0.0, [], [f"{type(exc).__name__}: {exc}"], [], False
        runtime_ms = (perf_counter() - started) * 1000
        return VerifierResult(
            verifier_id=self.verifier_id,
            verifier_version=self.version,
            verifier_result_id=f"vr-{self.verifier_id}-{candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            stage=self.stage,
            verdict=verdict,  # type: ignore[arg-type]
            score=score,
            checks_passed=passed,
            checks_failed=failed,
            assumptions=assumptions,
            counterexample_found=counterexample,
            uncertainty=round(1.0 - score, 3),
            runtime_ms=round(runtime_ms, 3),
            provenance=[f"verifier:{self.verifier_id}:{self.version}"],
        )

    @abstractmethod
    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        raise NotImplementedError
