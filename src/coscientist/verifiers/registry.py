from __future__ import annotations

from coscientist.verifiers.base import ScientificVerifier
from coscientist.verifiers.atomic import (
    AtomicSchemaVerifier,
    CounterexampleParameterSearchVerifier,
    LimitingCaseVerifier,
    NumericalDiagonalizationVerifier,
    ParameterFitVerifier,
    QuTiPDynamicsVerifier,
    QuTiPEigenVerifier,
    SelectionRuleVerifier,
    SpectrumConsistencyVerifier,
    SymbolicHamiltonianVerifier,
)
from coscientist.verifiers.generic import (
    CounterexampleHookVerifier,
    EvidenceConsistencyVerifier,
    ExperimentalConsistencyVerifier,
    LogicalConsistencyVerifier,
    SchemaConstraintVerifier,
)
from coscientist.verifiers.materials import MaterialsFormulaVerifier


class VerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, ScientificVerifier] = {}

    def register(self, verifier: ScientificVerifier) -> None:
        self._verifiers[verifier.verifier_id] = verifier

    def get(self, verifier_id: str) -> ScientificVerifier:
        return self._verifiers[verifier_id]

    def enabled(self, verifier_ids: list[str]) -> list[ScientificVerifier]:
        return [self._verifiers[item] for item in verifier_ids if item in self._verifiers]

    def ids(self) -> list[str]:
        return sorted(self._verifiers)


def default_verifier_registry() -> VerifierRegistry:
    registry = VerifierRegistry()
    for verifier in [
        SchemaConstraintVerifier(),
        LogicalConsistencyVerifier(),
        EvidenceConsistencyVerifier(),
        CounterexampleHookVerifier(),
        ExperimentalConsistencyVerifier(),
        MaterialsFormulaVerifier(),
        AtomicSchemaVerifier(),
        SelectionRuleVerifier(),
        SymbolicHamiltonianVerifier(),
        NumericalDiagonalizationVerifier(),
        SpectrumConsistencyVerifier(),
        LimitingCaseVerifier(),
        ParameterFitVerifier(),
        CounterexampleParameterSearchVerifier(),
        QuTiPEigenVerifier(),
        QuTiPDynamicsVerifier(),
    ]:
        registry.register(verifier)
    return registry
