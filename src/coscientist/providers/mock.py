from __future__ import annotations

import hashlib
from typing import Any, TypeVar

from pydantic import BaseModel

from coscientist.schemas.hypothesis import Hypothesis, HypothesisBatch
from coscientist.schemas.ranking import HypothesisRanking, PairwiseBatch, PairwiseComparison, RankingBatch
from coscientist.schemas.review import Review, ReviewBatch
from coscientist.providers.base import StructuredLLMProvider

T = TypeVar("T", bound=BaseModel)


class MockProvider(StructuredLLMProvider):
    name = "mock"

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
        context: dict[str, Any] | None = None,
    ) -> T:
        context = context or {}
        if output_schema is HypothesisBatch:
            return self._hypothesis_batch(context)  # type: ignore[return-value]
        if output_schema is ReviewBatch:
            return self._review_batch(context)  # type: ignore[return-value]
        if output_schema is RankingBatch:
            return self._ranking_batch(context)  # type: ignore[return-value]
        if output_schema is PairwiseBatch:
            return self._pairwise_batch(context)  # type: ignore[return-value]
        raise TypeError(f"MockProvider does not support schema {output_schema.__name__}")

    def _hypothesis_batch(self, context: dict[str, Any]) -> HypothesisBatch:
        mode = str(context.get("mode", "generate"))
        count = int(context.get("count", 3))
        strategy = str(context.get("strategy", "mechanistic"))
        round_number = int(context.get("round_number", 0))
        parents: list[Hypothesis] = list(context.get("parents", []))
        parent = context.get("parent")
        goal_id = str(context.get("goal_id", "goal"))
        hypotheses: list[Hypothesis] = []

        for index in range(count):
            parent_ids: list[str] = []
            version = 1
            base = f"{goal_id}-{strategy}-{round_number}-{index}"
            change_summary = None
            status = "active"
            if mode in {"repair", "branch"} and isinstance(parent, Hypothesis):
                parent_ids = [parent.id]
                version = parent.version + 1
                base = f"{parent.id}-{mode}-{round_number}-{index}"
                change_summary = f"{mode.title()} child adjusts assumptions and tests from parent {parent.id}."
                status = "repaired" if mode == "repair" else "branched"
            elif mode == "combine":
                parent_ids = [item.id for item in parents]
                version = max((item.version for item in parents), default=1) + 1
                base = f"combine-{round_number}-{index}-{'-'.join(parent_ids[:2])}"
                change_summary = f"Combines mechanisms from parents {', '.join(parent_ids)}."
                status = "combined"

            hid = self._id(base)
            noun = strategy.replace("-", " ")
            hypotheses.append(Hypothesis(
                id=hid,
                title=f"Synthetic {noun} hypothesis {index + 1}",
                core_claim=f"[MOCK] A {noun} mechanism may explain part of the goal through pathway {index + 1}.",
                mechanism=(
                    f"[MOCK] The proposal links a controllable variable to an observable response "
                    f"through a deterministic synthetic mechanism {hid[-4:]}."
                ),
                assumptions=[
                    "Mock assumption: the measured signal is not dominated by an uncontrolled confounder.",
                    "Mock assumption: the proposed variable can be independently perturbed.",
                ],
                supporting_evidence=["Synthetic support generated for workflow testing only."],
                contradicting_evidence=["Synthetic contradiction: an alternative pathway could produce the same signature."],
                novelty_statement="[MOCK] Novelty is illustrative and has not been checked against literature.",
                testable_predictions=[
                    f"Prediction {index + 1}: perturbation should shift the primary observable in direction {index % 2}.",
                    "A negative control should suppress the claimed mechanism.",
                ],
                falsification_criteria=[
                    "No measurable change under the proposed perturbation.",
                    "An alternative mechanism explains all discriminative observations with fewer assumptions.",
                ],
                proposed_experiments=[
                    "Run a blinded perturbation/control measurement series.",
                    "Compare against at least one mechanistically distinct control hypothesis.",
                ],
                uncertainty=round(0.35 + (index * 0.07) + (round_number * 0.03), 3),
                generation_strategy=strategy if mode == "generate" else mode,
                parent_ids=parent_ids,
                version=version,
                status=status,  # type: ignore[arg-type]
                change_summary=change_summary,
            ))
        return HypothesisBatch(hypotheses=hypotheses)

    def _review_batch(self, context: dict[str, Any]) -> ReviewBatch:
        hypotheses: list[Hypothesis] = list(context["hypotheses"])
        reviews = []
        for hypothesis in hypotheses:
            weak = self._score_seed(hypothesis.id) % 3
            fatal_flaws = [] if weak else ["Synthetic fatal-risk check: mechanism may be underdetermined by listed tests."]
            reviews.append(Review(
                hypothesis_id=hypothesis.id,
                fatal_flaws=fatal_flaws,
                nonfatal_weaknesses=["Needs sharper operational definitions for the main observable."],
                unsupported_claims=["Literature support is not available in this MVP mock mode."],
                conflicting_evidence=["Mock review notes a plausible competing explanation."],
                novelty_concerns=["Novelty cannot be verified until citation tools are added."],
                suggested_repairs=["Add a discriminative control experiment.", "State measurable thresholds for falsification."],
                recommendation="repair" if fatal_flaws else "keep",
                confidence=0.72,
            ))
        return ReviewBatch(reviews=reviews)

    def _ranking_batch(self, context: dict[str, Any]) -> RankingBatch:
        hypotheses: list[Hypothesis] = list(context["hypotheses"])
        weights: dict[str, float] = dict(context.get("weights", {}))
        rankings = []
        for hypothesis in hypotheses:
            seed = self._score_seed(hypothesis.id)
            values = {
                "correctness": 5.0 + (seed % 17) / 10,
                "novelty": 5.0 + (seed % 19) / 10,
                "testability": 6.0 + (seed % 13) / 10,
                "explanatory_power": 5.5 + (seed % 11) / 10,
                "feasibility": 6.0 + (seed % 7) / 10,
                "discriminative_power": 5.5 + (seed % 23) / 10,
                "evidence_quality": 4.0 + (seed % 9) / 10,
                "impact": 5.0 + (seed % 29) / 10,
                "parsimony": 6.0 + (seed % 5) / 10,
            }
            total_weight = sum(weights.get(key, 1.0) for key in values)
            weighted_total = sum(values[key] * weights.get(key, 1.0) for key in values) / total_weight
            rankings.append(HypothesisRanking(
                hypothesis_id=hypothesis.id,
                weighted_total=round(weighted_total, 3),
                pairwise_wins=0,
                pairwise_losses=0,
                judge_notes=["[MOCK] Deterministic synthetic rubric score; not scientific validation."],
                **{key: round(value, 3) for key, value in values.items()},
            ))
        return RankingBatch(rankings=rankings)

    def _pairwise_batch(self, context: dict[str, Any]) -> PairwiseBatch:
        pairs: list[tuple[Hypothesis, Hypothesis]] = list(context["pairs"])
        comparisons = []
        for a, b in pairs:
            score_a = self._score_seed(a.id)
            score_b = self._score_seed(b.id)
            winner = "tie"
            if score_a > score_b:
                winner = "a"
            elif score_b > score_a:
                winner = "b"
            comparisons.append(PairwiseComparison(
                hypothesis_a_id=a.id,
                hypothesis_b_id=b.id,
                winner=winner,  # type: ignore[arg-type]
                judge_notes="[MOCK] Anonymous pairwise comparison based on stable synthetic ordering.",
            ))
        return PairwiseBatch(comparisons=comparisons)

    @staticmethod
    def _id(raw: str) -> str:
        return "hyp-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _score_seed(raw: str) -> int:
        return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16)
