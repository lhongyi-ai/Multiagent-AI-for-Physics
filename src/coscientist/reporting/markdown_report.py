from __future__ import annotations

from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.ranking import HypothesisRanking
from coscientist.schemas.research_goal import ResearchGoal
from coscientist.schemas.run_state import RunState
from coscientist.literature.pipeline import LiteratureAcquisitionResult, no_literature_result


def build_markdown_report(
    goal: ResearchGoal,
    finalists: list[Hypothesis],
    rankings: list[HypothesisRanking],
    state: RunState,
    literature: LiteratureAcquisitionResult | None = None,
) -> str:
    literature = literature or no_literature_result()
    ranking_by_id = {ranking.hypothesis_id: ranking for ranking in rankings}
    lines = [
        f"# Co-Scientist MVP Report: {goal.title}",
        "",
        "> Mock-mode outputs are synthetic and are not literature-verified scientific claims.",
        "",
        f"Research question: {goal.question}",
        "",
        f"Run ID: `{state.run_id}`",
        f"LLM calls used: {state.llm_call_count}/{state.maximum_llm_calls}",
        "",
        "## Final Top Hypotheses",
        "",
    ]
    for index, hypothesis in enumerate(finalists, start=1):
        ranking = ranking_by_id[hypothesis.id]
        lines.extend([
            f"### {index}. {hypothesis.title}",
            "",
            f"- ID: `{hypothesis.id}`",
            f"- Weighted score: {ranking.weighted_total:.3f}",
            f"- Pairwise record: {ranking.pairwise_wins} wins, {ranking.pairwise_losses} losses",
            f"- Parent IDs: {', '.join(f'`{pid}`' for pid in hypothesis.parent_ids) if hypothesis.parent_ids else 'none'}",
            f"- Change summary: {hypothesis.change_summary or 'Initial generated hypothesis.'}",
            f"- Core claim: {hypothesis.core_claim}",
            f"- Mechanism: {hypothesis.mechanism}",
            f"- Falsification criteria: {'; '.join(hypothesis.falsification_criteria)}",
            f"- Proposed experiments: {'; '.join(hypothesis.proposed_experiments)}",
            "",
            "**Source-Backed Observations**",
            "",
            "- None verified in this run; citation verification requires retrieved passages.",
            "",
            "**Source-Author Interpretations**",
            "",
            "- None verified in this run.",
            "",
            "**Co-Scientist Inferences**",
            "",
            f"- {hypothesis.core_claim}",
            "",
            "**Contradicting Evidence**",
            "",
            f"- {'; '.join(hypothesis.contradicting_evidence) if hypothesis.contradicting_evidence else 'None verified.'}",
            "",
            "**Metadata Status**",
            "",
            f"- Metadata resolutions available this run: {len(literature.metadata_resolutions)}",
            f"- Metadata conflicts recorded: {len(literature.metadata_conflicts)}",
            "",
            "**Full-Text Status**",
            "",
            f"- Open/full-text locations found: {len(literature.full_text_locations)}",
            "- Search result rank, metadata, and PDF availability are not treated as claim support.",
            "",
            "**Remaining Unsupported Claims**",
            "",
            "- All hypothesis claims remain unsupported until exact passages pass citation verification.",
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "- External literature providers are optional and live network access is explicitly gated.",
        "- Scores are rubric judgments, not evidence of scientific correctness.",
        "- Mock mode is deterministic and synthetic for local workflow validation.",
        "",
    ])
    return "\n".join(lines)
