from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime

from coscientist.agents.grounding import GroundingAgent
from coscientist.agents.meta_review import MetaReviewAgent
from coscientist.agents.proximity import ProximityAgent
from coscientist.pilot.artifacts import read_json, validate_v1_artifacts
from coscientist.pilot.evidence import attach_fixture_evidence, verify_hypothesis_evidence
from coscientist.pilot.project_io import load_fixture_corpus, load_project_spec
from coscientist.pilot.runner import run_pilot_project_sync
from coscientist.providers.mock import MockProvider
from coscientist.schemas.evaluation import RoundComparison
from coscientist.schemas.hypothesis import HypothesisBatch
from coscientist.schemas.v15b import GroundingConfig, MetaReviewConfig, ProximityConfig


PROJECT = "research-projects/code_assistant_fixture/project.yaml"


def _hypotheses(count: int = 3):
    provider = MockProvider()
    import asyncio

    return asyncio.run(provider.generate_structured(
        "prompt",
        HypothesisBatch,
        {"goal_id": "g", "strategy": "mechanistic", "count": count},
    )).hypotheses


def test_proximity_flags_duplicates_and_builds_valid_graph() -> None:
    hypotheses = _hypotheses(2)
    duplicate = hypotheses[0].model_copy(update={"id": "hyp-duplicate"})
    hypotheses = [hypotheses[0], duplicate, hypotheses[1]]
    proximity = ProximityAgent().analyze(
        project_id="p",
        run_id="r",
        round_label="final",
        round_number=0,
        hypotheses=hypotheses,
        rankings=[],
        verifications=[],
        config=ProximityConfig(similarity_threshold=0.7),
        model_mode="mock",
        literature_mode="fixture",
    )
    assert any(item.overall_similarity >= 0.9 for item in proximity.pairwise_similarities)
    assert all(edge.source_id in {h.id for h in hypotheses} for edge in proximity.graph_edges)
    assert proximity.search_space_coverage.unique_cluster_count >= 1
    assert proximity.validation_status == "validated"


def test_grounding_packet_and_diagnostics_are_deterministic() -> None:
    project = load_project_spec(PROJECT)
    corpus = load_fixture_corpus(Path(PROJECT).parent / "corpus.jsonl")
    hypotheses = attach_fixture_evidence(_hypotheses(2), corpus, "final")
    verifications = verify_hypothesis_evidence(hypotheses, corpus)
    agent = GroundingAgent()
    packet = agent.build_packet(project_id=project.project_id, run_id="r", round_label="final", corpus=corpus, verifications=verifications, config=GroundingConfig())
    diagnostics = agent.diagnostics(project_id=project.project_id, run_id="r", round_label="final", hypotheses=hypotheses, verifications=verifications, packet=packet, config=GroundingConfig())
    assert packet.evidence_items
    assert diagnostics.supported_claim_count > 0
    assert diagnostics.citation_hallucination_count == 0
    assert diagnostics.validation_status == "validated"


def test_meta_review_advisory_and_controlled_decisions() -> None:
    project = load_project_spec(PROJECT)
    corpus = load_fixture_corpus(Path(PROJECT).parent / "corpus.jsonl")
    hypotheses = attach_fixture_evidence(_hypotheses(2), corpus, "final")
    verifications = verify_hypothesis_evidence(hypotheses, corpus)
    proximity = ProximityAgent().analyze(
        project_id=project.project_id,
        run_id="r",
        round_label="final",
        round_number=0,
        hypotheses=hypotheses,
        rankings=[],
        verifications=verifications,
        config=ProximityConfig(),
        model_mode="mock",
        literature_mode="fixture",
    )
    grounding_agent = GroundingAgent()
    packet = grounding_agent.build_packet(project_id=project.project_id, run_id="r", round_label="final", corpus=corpus, verifications=verifications, config=GroundingConfig())
    diagnostics = grounding_agent.diagnostics(project_id=project.project_id, run_id="r", round_label="final", hypotheses=hypotheses, verifications=verifications, packet=packet, config=GroundingConfig())
    meta = MetaReviewAgent().review(
        project_id=project.project_id,
        run_id="r",
        round_label="final",
        round_number=0,
        hypotheses=hypotheses,
        rankings=[],
        verifications=verifications,
        evaluations=[],
        comparison=RoundComparison(project_id=project.project_id, evaluator_self_preference_note="test", generated_at=datetime.now(UTC)),
        proximity=proximity,
        grounding=diagnostics,
        config=MetaReviewConfig(),
        model_mode="mock",
        literature_mode="fixture",
    )
    advisory = MetaReviewAgent().decide(project_id=project.project_id, run_id="r", round_label="final", review=meta, config=MetaReviewConfig())
    assert advisory.feed_into_next_round is False
    controlled_config = MetaReviewConfig(feedback_mode="controlled_feedback", feed_into_next_round=True)
    controlled = MetaReviewAgent().decide(project_id=project.project_id, run_id="r", round_label="final", review=meta, config=controlled_config)
    assert controlled.feed_into_next_round is True
    assert controlled.accepted_generation_strategy_adjustments


def test_v15b_artifacts_are_persisted_and_validate(tmp_path: Path) -> None:
    run_dir = run_pilot_project_sync(PROJECT, runs_dir=tmp_path, run_id="v15b")
    assert validate_v1_artifacts(run_dir) == []
    assert (run_dir / "proximity_round_final.json").exists()
    assert (run_dir / "meta_review_round_final.json").exists()
    assert (run_dir / "grounding_diagnostics_round_final.json").exists()
    summary = read_json(run_dir / "v15b_summary.json")
    assert summary["schema_version"] == "v15b"
    assert "Hypothesis Landscape" in (run_dir / "report.md").read_text()
    assert "V1.5B Landscape Review Questions" in (run_dir / "human_review.md").read_text()
