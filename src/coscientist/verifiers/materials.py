from __future__ import annotations

import re

from coscientist.schemas.v17 import CandidateSolution, ScientificProblem
from coscientist.verifiers.base import ScientificVerifier


class MaterialsFormulaVerifier(ScientificVerifier):
    verifier_id = "materials_formula"
    stage = "cheap"
    capability = "materials"

    def _verify(self, candidate: CandidateSolution, problem: ScientificProblem, *, evidence_ids: set[str]):
        text = " ".join([candidate.formal_representation or "", candidate.construction_or_model or "", candidate.summary])
        formula = _extract_formula(text)
        passed, failed = [], []
        if not formula:
            return "inconclusive", 0.5, [], ["no_formula_detected"], ["formula parsing is lightweight"], False
        normalized = _normalize_formula(formula)
        if normalized:
            passed.append(f"formula_normalized:{normalized}")
        if "CaFe4Al8" in text or "Ca1Fe4Al8" in normalized:
            passed.append("target_stoichiometry_pattern_present")
        if "Ca0" in text or "no Ca" in text:
            failed.append("composition_inconsistent_with_ca_containing_target")
        if "Fe-Al" in text or "Al-rich" in text:
            passed.append("competing_phase_label_preserved")
        verdict = "pass" if passed and not failed else "partial" if passed else "fail"
        score = len(passed) / max(1, len(passed) + len(failed))
        return verdict, round(score, 3), passed, failed, ["not a crystallographic database check"], False


def _extract_formula(text: str) -> str | None:
    match = re.search(r"\b(?:Ca\d*)?Fe\d*Al\d+\b", text)
    return match.group(0) if match else None


def _normalize_formula(formula: str) -> str:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    return "".join(f"{element}{count or '1'}" for element, count in parts)
