from __future__ import annotations

from coscientist.schemas.evaluation import RoundComparison, RoundEvaluation
from coscientist.schemas.evidence import EvidenceVerificationRecord
from coscientist.schemas.hypothesis import Hypothesis
from coscientist.schemas.literature import Paper
from coscientist.schemas.project import ResearchProjectSpec


def build_pilot_report(
    project: ResearchProjectSpec,
    final_hypotheses: list[Hypothesis],
    evaluations: list[RoundEvaluation],
    comparison: RoundComparison,
    verifications: list[EvidenceVerificationRecord],
) -> str:
    final_eval = next((item for item in evaluations if item.round_label == "final"), None)
    final_record_count = sum(len(hypothesis.evidence_links) for hypothesis in final_hypotheses)
    final_verifications = verifications[-final_record_count:] if final_record_count else []
    lines = [
        f"# Pilot Research Report: {project.title}",
        "",
        "> This deterministic pilot evaluates workflow behavior. It does not claim to solve the research question.",
        "",
        f"Research question: {project.research_question}",
        "",
        "## Evaluation Summary",
        "",
    ]
    if final_eval:
        for dimension, value in final_eval.mean_scores.items():
            delta = comparison.score_changes_by_dimension.get(dimension, 0.0)
            lines.append(f"- {dimension}: final mean {value:.3f}, change from initial {delta:+.3f}")
    lines.extend([
        "",
        "## Evidence Verification Summary",
        "",
        f"- All-round verified or partially verified records: {sum(1 for item in verifications if item.status in {'verified', 'partially_verified'})}",
        f"- All-round conflicting records: {sum(1 for item in verifications if item.status == 'conflicting')}",
        f"- All-round unsupported or invalid records: {sum(1 for item in verifications if item.status in {'unsupported', 'invalid_reference'})}",
        f"- Final-hypothesis conflicting records: {sum(1 for item in final_verifications if item.status == 'conflicting')}",
        f"- Final-hypothesis unsupported or invalid records: {sum(1 for item in final_verifications if item.status in {'unsupported', 'invalid_reference'})}",
        "",
        "## Final Top Hypotheses",
        "",
    ])
    for index, hypothesis in enumerate(final_hypotheses, start=1):
        lines.extend([
            f"### {index}. {hypothesis.title}",
            "",
            f"- ID: `{hypothesis.id}`",
            f"- Lineage: {', '.join(hypothesis.parent_ids) if hypothesis.parent_ids else 'initial'}",
            f"- Core claim: {hypothesis.core_claim}",
            f"- Mechanism: {hypothesis.mechanism}",
            f"- Testable predictions: {'; '.join(hypothesis.testable_predictions)}",
            f"- Falsification criteria: {'; '.join(hypothesis.falsification_criteria)}",
            f"- Evidence-linked claims: {len(hypothesis.evidence_links)}",
            "- Do not claim this hypothesis is established without independent validation.",
            "",
        ])
    lines.extend([
        "## Evaluator Self-Preference Note",
        "",
        comparison.evaluator_self_preference_note,
        "",
    ])
    return "\n".join(lines)


def build_human_review_package(
    project: ResearchProjectSpec,
    corpus: list[Paper],
    final_hypotheses: list[Hypothesis],
    comparison: RoundComparison,
    verifications: list[EvidenceVerificationRecord],
) -> str:
    verification_by_hypothesis: dict[str, list[EvidenceVerificationRecord]] = {}
    for record in verifications:
        verification_by_hypothesis.setdefault(record.hypothesis_id, []).append(record)
    lines = [
        f"# Human Review Package: {project.title}",
        "",
        f"Research question: {project.research_question}",
        "",
        "## Corpus Scope and Limitations",
        "",
        f"- Fixture corpus size: {len(corpus)} papers.",
        "- This corpus is a deterministic test corpus, not a complete literature review.",
        "- Search ranking, metadata, and fixture excerpts are not scientific proof.",
        "",
        "## Reviewer Decision Options",
        "",
        "- accept for further investigation",
        "- revise",
        "- reject",
        "- merge with another hypothesis",
        "- insufficient evidence",
        "",
    ]
    for index, hypothesis in enumerate(final_hypotheses, start=1):
        records = verification_by_hypothesis.get(hypothesis.id, [])
        lines.extend([
            f"## Hypothesis {index}: {hypothesis.title}",
            "",
            f"- ID: `{hypothesis.id}`",
            f"- Core argument: {hypothesis.core_claim}",
            f"- Mechanism: {hypothesis.mechanism}",
            f"- Lineage: {', '.join(hypothesis.parent_ids) if hypothesis.parent_ids else 'initial'}",
            f"- Uncertainty: {hypothesis.uncertainty:.3f}",
            "",
            "### Strongest Supporting Evidence",
            "",
        ])
        supports = [link for link in hypothesis.evidence_links if link.supporting_paper_ids]
        if supports:
            for link in supports[:2]:
                lines.append(f"- {link.claim_text} | papers: {', '.join(link.supporting_paper_ids)}")
        else:
            lines.append("- No structured supporting source link.")
        lines.extend([
            "",
            "### Strongest Counter-Evidence",
            "",
        ])
        counters = [link for link in hypothesis.evidence_links if link.contradicting_paper_ids]
        if counters:
            for link in counters[:2]:
                lines.append(f"- {link.claim_text} | papers: {', '.join(link.contradicting_paper_ids)}")
        else:
            lines.append("- No structured contradicting source link.")
        lines.extend([
            "",
            "### Unsupported Assumptions",
            "",
            *[f"- {item}" for item in hypothesis.assumptions],
            "",
            "### Testable Predictions",
            "",
            *[f"- {item}" for item in hypothesis.testable_predictions],
            "",
            "### Proposed Falsification Tests",
            "",
            *[f"- {item}" for item in hypothesis.falsification_criteria],
            "",
            "### Ranking Rationale",
            "",
            "- See `evaluation_by_round.json` and `round_comparison.json`; scores are decision aids, not truth.",
            "",
            "### Claims That Must Not Yet Be Made",
            "",
            "- Do not claim the mechanism is established.",
            "- Do not claim the fixture corpus is exhaustive.",
            "- Do not claim validation has occurred without independent tests.",
            "",
            "### Reviewer Decision",
            "",
            "- Decision:",
            "- Rationale:",
            "- Required revisions:",
            "- Merge target, if any:",
            "- Reviewer:",
            "- Date:",
            "",
            f"Verification statuses: {', '.join(sorted({record.status for record in records})) if records else 'none'}",
            "",
        ])
    lines.extend([
        "## Round Comparison Notes",
        "",
        f"- Citation coverage by round: {comparison.citation_coverage}",
        f"- Unsupported claim count by round: {comparison.unsupported_claim_count}",
        f"- Hypothesis diversity by round: {comparison.hypothesis_diversity}",
        "",
    ])
    return "\n".join(lines)
