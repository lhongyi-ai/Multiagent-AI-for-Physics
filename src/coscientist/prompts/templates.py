from __future__ import annotations

from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.review import Review


def generator_prompt(goal: ResearchGoal, strategy: str, count: int) -> str:
    return (
        f"Generate {count} distinct hypotheses for the research goal '{goal.title}' "
        f"using the {strategy} strategy. Avoid superficial rewordings. Return only "
        "schema-valid JSON."
    )


def review_prompt(hypothesis: Hypothesis) -> str:
    return (
        "Adversarially review this hypothesis for scientific consistency, hidden "
        f"assumptions, alternatives, testability, falsifiability, novelty risk, and unsupported claims: {hypothesis.id}."
    )


def ranking_prompt(round_number: int) -> str:
    return f"Score and compare hypotheses for round {round_number} using the configured scientific rubric."


def evolution_prompt(hypothesis: Hypothesis, reviews: list[Review], mode: str, count: int) -> str:
    review_ids = ", ".join(review.hypothesis_id for review in reviews)
    return (
        f"Evolve hypothesis {hypothesis.id} with mode {mode}; create {count} immutable child records. "
        f"Use review feedback from {review_ids}. Explain what changed."
    )


def combination_prompt(parents: list[Hypothesis], count: int) -> str:
    parent_ids = ", ".join(parent.id for parent in parents)
    return f"Combine parent hypotheses {parent_ids} into {count} child hypotheses with preserved lineage."
