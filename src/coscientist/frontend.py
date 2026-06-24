from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import yaml

from coscientist.discovery import (
    load_discovery_project,
    persist_expert_feedback,
    run_discovery_project,
    validate_discovery_artifacts,
)
from coscientist.pilot.artifacts import read_json, read_jsonl


@dataclass
class OfflineDiscoveryFrontend:
    """Small backend-backed facade for local smoke tests and future Gradio wiring."""

    runs_dir: str | Path = "runs"

    def load_project(self, project_path: str | Path) -> dict[str, object]:
        project = load_discovery_project(project_path)
        return {
            "project_id": project.project_id,
            "title": project.title,
            "problem_id": project.problem.problem_id,
            "candidate_count": len(project.initial_candidates),
            "model_mode": project.model_mode,
        }

    def run_fixture(self, project_path: str | Path, *, run_id: str = "discovery-frontend-smoke", force: bool = True) -> str:
        run_dir = run_discovery_project(project_path, runs_dir=self.runs_dir, run_id=run_id, force=force)
        return str(run_dir)

    def validate(self, run_dir: str | Path) -> list[str]:
        return validate_discovery_artifacts(run_dir)

    def persist_feedback(self, run_dir: str | Path, *, candidate_id: str, decision: str, rationale: str, reviewer: str = "local-human") -> str:
        path = Path(run_dir)
        if (path / "candidate_archive.jsonl").exists():
            return str(persist_expert_feedback(run_dir, candidate_id=candidate_id, decision=decision, rationale=rationale, reviewer=reviewer))
        feedback_path = path / "expert_feedback.jsonl"
        existing = read_jsonl(feedback_path) if feedback_path.exists() else []
        record = {"schema_version": "v19", "candidate_id": candidate_id, "decision": decision, "rationale": rationale, "reviewer": reviewer, "created_at": datetime.now(UTC).isoformat()}
        from coscientist.pilot.artifacts import write_jsonl

        write_jsonl(feedback_path, [*existing, record])
        return str(feedback_path)

    def dependency_status(self) -> dict[str, str]:
        return {name: _version(name) for name in ["sympy", "numpy", "scipy", "qutip", "gradio"]}

    def run_atomic_fixture(self, project_path: str | Path, *, run_id: str = "atomic-frontend-smoke", force: bool = True) -> str:
        from coscientist.atomic.discovery import run_atomic_discovery_project

        run_dir = run_atomic_discovery_project(project_path, runs_dir=self.runs_dir, run_id=run_id, force=force)
        return str(run_dir)

    def validate_atomic(self, run_dir: str | Path) -> list[str]:
        from coscientist.atomic.discovery import validate_atomic_discovery_artifacts

        return validate_atomic_discovery_artifacts(run_dir)

    def ask_research_question(
        self,
        question: str,
        *,
        context: str = "",
        domain: str = "general_science",
        run_id: str = "",
        ranking_mode: str = "elo",
        force: bool = True,
    ) -> str:
        question = question.strip()
        if not question:
            raise ValueError("research question is required")
        run_id = run_id.strip() or f"ask-{_slug(question)}"
        project_path = self._write_question_project(question, context=context, domain=domain, run_id=run_id, ranking_mode=ranking_mode)
        return self.run_fixture(project_path, run_id=run_id, force=force)

    def _write_question_project(self, question: str, *, context: str, domain: str, run_id: str, ranking_mode: str) -> Path:
        project_dir = Path(self.runs_dir) / "_frontend_projects"
        project_dir.mkdir(parents=True, exist_ok=True)
        project_path = project_dir / f"{run_id}.yaml"
        payload = _question_project_payload(question, context=context, domain=domain, run_id=run_id, ranking_mode=ranking_mode)
        project_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return project_path

    def candidate_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir)
        rows = []
        if not (path / "candidate_archive.jsonl").exists():
            return rows
        for item in read_jsonl(path / "candidate_archive.jsonl"):
            model = item.get("structured_model", {}).get("atomic_model", {})
            rows.append({
                "candidate_id": item.get("candidate_id"),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "status": item.get("scientific_status"),
                "type": item.get("candidate_type"),
                "model_family": model.get("model_family"),
                "lineage_depth": item.get("lineage_depth"),
                "aggregate_score": item.get("aggregate_search_score"),
            })
        return rows

    def verifier_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir)
        if not (path / "verifier_results.jsonl").exists():
            return []
        return [
            {
                "candidate_id": item.get("candidate_id"),
                "verifier_id": item.get("verifier_id"),
                "stage": item.get("stage"),
                "verdict": item.get("verdict"),
                "score": item.get("score"),
                "failed": ", ".join(item.get("checks_failed", [])),
            }
            for item in read_jsonl(path / "verifier_results.jsonl")
        ]

    def tournament_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir)
        if not (path / "tournament_comparisons.jsonl").exists():
            return []
        return read_jsonl(path / "tournament_comparisons.jsonl")

    def elo_rating_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "elo_tournament_state.json"
        if not path.exists():
            return []
        return read_json(path).get("ratings", [])

    def strategy_performance_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "search_strategy_metrics.json"
        return read_json(path) if path.exists() else []

    def adaptive_budget_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "adaptive_budget_allocation.json"
        return read_json(path).get("allocations", []) if path.exists() else []

    def task_queue_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "search_tasks.jsonl"
        return read_jsonl(path) if path.exists() else []

    def checkpoint_summary(self, run_dir: str | Path) -> dict[str, object]:
        path = Path(run_dir) / "search_checkpoint.json"
        if not path.exists():
            path = Path(run_dir) / "campaign_checkpoint.json"
        return read_json(path) if path.exists() else {}

    def claim_ledger_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "claim_ledger.jsonl"
        return read_jsonl(path) if path.exists() else []

    def prediction_ledger_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "prediction_ledger.jsonl"
        return read_jsonl(path) if path.exists() else []

    def reproduction_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "reproduction_results.jsonl"
        return read_jsonl(path) if path.exists() else []

    def reproduction_discrepancies(self, run_dir: str | Path) -> dict[str, object]:
        path = Path(run_dir) / "reproduction_discrepancies.json"
        return read_json(path) if path.exists() else {}

    def provider_routing_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "provider_routing_plan.json"
        return read_json(path).get("routes", []) if path.exists() else []

    def report_text(self, run_dir: str | Path) -> str:
        path = Path(run_dir)
        for name in ["superconductivity_report.md", "atomic_discovery_report.md", "discovery_report.md", "report.md"]:
            candidate = path / name
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        return ""

    def copyable_summary(self, run_dir: str | Path) -> str:
        rows = self.candidate_rows(run_dir)
        ratings = {item.get("candidate_id"): item for item in self.elo_rating_rows(run_dir)}
        claims = self.claim_ledger_rows(run_dir)
        predictions = self.prediction_ledger_rows(run_dir)
        lines = ["# Optimized Hypotheses", ""]
        for index, row in enumerate(rows, start=1):
            rating = ratings.get(row.get("candidate_id"), {})
            lines.extend([
                f"## {index}. {row.get('title') or row.get('candidate_id')}",
                "",
                f"- Candidate ID: `{row.get('candidate_id')}`",
                f"- Type: {row.get('type')}",
                f"- Status: {row.get('status')}",
                f"- Score: {row.get('aggregate_score')}",
                f"- Rating: {rating.get('rating', 'unrated')}",
                f"- Summary: {row.get('summary')}",
                "",
            ])
        if claims:
            lines.extend(["# Claim Ledger", ""])
            for claim in claims:
                lines.append(f"- `{claim.get('claim_id')}` [{claim.get('status')}]: {claim.get('claim_text')}")
            lines.append("")
        if predictions:
            lines.extend(["# Prediction Ledger", ""])
            for prediction in predictions:
                lines.append(f"- `{prediction.get('prediction_id')}` [{prediction.get('status')}]: {prediction.get('observable')}")
            lines.append("")
        report = self.report_text(run_dir)
        if report:
            lines.extend(["# Report", "", report])
        return "\n".join(str(item) for item in lines)

    def benchmark_metrics(self, run_dir: str | Path) -> dict[str, object]:
        path = Path(run_dir) / "atomic_benchmark_metrics.json"
        return read_json(path) if path.exists() else {}

    def run_campaign_fixture(self, project_path: str | Path, *, run_id: str = "campaign-frontend-smoke", force: bool = True) -> str:
        from coscientist.atomic.campaign import run_atomic_campaign_project

        run_dir = run_atomic_campaign_project(project_path, runs_dir=self.runs_dir, run_id=run_id, force=force)
        return str(run_dir)

    def validate_campaign(self, run_dir: str | Path) -> list[str]:
        from coscientist.atomic.campaign import validate_atomic_campaign_artifacts

        return validate_atomic_campaign_artifacts(run_dir)

    def source_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "source_manifest.json"
        return read_json(path).get("sources", []) if path.exists() else []

    def campaign_observation_rows(self, run_dir: str | Path, *, include_hidden: bool = False) -> list[dict[str, object]]:
        path = Path(run_dir) / ("atomic_transitions_normalized.jsonl" if include_hidden else "agent_visible_observations.jsonl")
        return read_jsonl(path) if path.exists() else []

    def campaign_comparison(self, run_dir: str | Path) -> dict[str, object]:
        path = Path(run_dir) / "model_comparison.json"
        return read_json(path) if path.exists() else {}

    def campaign_identifiability_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "identifiability_results.jsonl"
        return read_jsonl(path) if path.exists() else []

    def run_superconductivity_fixture(self, project_path: str | Path, *, run_id: str = "superconductivity-workbench", force: bool = True) -> str:
        from coscientist.superconductivity import run_superconductivity_campaign

        run_dir = run_superconductivity_campaign(project_path, runs_dir=self.runs_dir, run_id=run_id, force=force)
        return str(run_dir)

    def validate_superconductivity(self, run_dir: str | Path) -> list[str]:
        from coscientist.superconductivity import validate_superconductivity_campaign

        return validate_superconductivity_campaign(run_dir)

    def superconductivity_score_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "superconductivity_scores.jsonl"
        return read_jsonl(path) if path.exists() else []

    def superconductivity_energy_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "energy_decomposition.jsonl"
        return read_jsonl(path) if path.exists() else []

    def superconductivity_optical_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "optical_sum_results.jsonl"
        return read_jsonl(path) if path.exists() else []

    def superconductivity_material_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "material_mapping.jsonl"
        return read_jsonl(path) if path.exists() else []

    def run_v22_fixture(self, project_path: str | Path, *, run_id: str = "v22-superconductivity-workbench", force: bool = True) -> str:
        from coscientist.superconductivity import run_v22_campaign

        run_dir = run_v22_campaign(project_path, runs_dir=self.runs_dir, run_id=run_id, force=force)
        return str(run_dir)

    def validate_v22(self, run_dir: str | Path) -> list[str]:
        from coscientist.superconductivity import validate_v22_campaign

        return validate_v22_campaign(run_dir)

    def v22_live_agent_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "live_agent_dialogues.jsonl"
        return read_jsonl(path) if path.exists() else []

    def v22_database_status_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "provider_connection_results.json"
        return read_json(path).get("providers", []) if path.exists() else []

    def v22_material_family_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "material_family_candidates.jsonl"
        return read_jsonl(path) if path.exists() else []

    def v22_fingerprint_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "mechanism_fingerprints.jsonl"
        return read_jsonl(path) if path.exists() else []

    def v22_adversarial_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "adversarial_tests.jsonl"
        return read_jsonl(path) if path.exists() else []

    def v22_experiment_proposal_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir) / "experiment_proposals.jsonl"
        return read_jsonl(path) if path.exists() else []

    def build_claim_dag_database(self, run_dir: str | Path, *, candidate_id: str | None = None, force: bool = False) -> str:
        from coscientist.claim_dag import create_claim_dag_artifacts_from_run, rebuild_claim_dag_database

        create_claim_dag_artifacts_from_run(run_dir, candidate_id=candidate_id, force=force)
        return str(rebuild_claim_dag_database(run_dir))

    def validate_claim_dag_database(self, run_dir: str | Path) -> list[str]:
        from coscientist.claim_dag import validate_claim_dag_database

        return validate_claim_dag_database(run_dir)

    def claim_dag_node_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        from coscientist.claim_dag import query_claim_dag_database

        path = Path(run_dir) / "claim_dag.sqlite"
        return query_claim_dag_database(run_dir, "claim_nodes", limit=200) if path.exists() else []

    def claim_dag_edge_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        from coscientist.claim_dag import query_claim_dag_database

        path = Path(run_dir) / "claim_dag.sqlite"
        return query_claim_dag_database(run_dir, "claim_edges", limit=200) if path.exists() else []

    def claim_dag_gate_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        from coscientist.claim_dag import query_claim_dag_database

        path = Path(run_dir) / "claim_dag.sqlite"
        return query_claim_dag_database(run_dir, "total_gate_results", limit=20) if path.exists() else []

    def run_live_agent_meeting(self, run_dir: str | Path, question: str, *, live_model: bool = False, max_rounds: int = 2, force: bool = True) -> str:
        from coscientist.live_agents import run_live_agent_meeting

        return str(run_live_agent_meeting(run_dir, question, live_model=live_model, max_rounds=max_rounds, force=force))

    def stream_live_agent_meeting(self, run_dir: str | Path, question: str, *, live_model: bool = False, max_rounds: int = 2, force: bool = True):
        from coscientist.live_agents import stream_live_agent_meeting

        yield from stream_live_agent_meeting(run_dir, question, live_model=live_model, max_rounds=max_rounds, force=force)

    def validate_live_agent_meeting(self, run_dir: str | Path) -> list[str]:
        from coscientist.live_agents import validate_meeting_artifacts

        return validate_meeting_artifacts(run_dir)

    def live_agent_message_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        from coscientist.live_agents import meeting_message_rows

        return meeting_message_rows(run_dir)

    def live_agent_provider_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        from coscientist.live_agents import provider_call_rows

        return provider_call_rows(run_dir)

    def live_agent_transcript(self, run_dir: str | Path) -> str:
        from coscientist.live_agents import meeting_transcript

        return meeting_transcript(run_dir)

    def run_targeted_repair(self, run_dir: str | Path, *, claim_id: str = "", reason: str = "") -> dict[str, object]:
        from coscientist.live_agents import targeted_repair_from_claim_dag

        return targeted_repair_from_claim_dag(run_dir, claim_id=claim_id or None, reason=reason)

    def claim_dag_mermaid(self, run_dir: str | Path) -> str:
        from coscientist.live_agents import claim_dag_mermaid

        return claim_dag_mermaid(run_dir)


def create_app(*, runs_dir: str | Path = "runs") -> OfflineDiscoveryFrontend:
    return OfflineDiscoveryFrontend(runs_dir=runs_dir)


def _question_project_payload(question: str, *, context: str, domain: str, run_id: str, ranking_mode: str) -> dict[str, object]:
    problem_id = f"problem-{_slug(question)}"
    evidence_ids = _evidence_ids(context)
    candidate_specs = _candidate_specs(question)
    candidates = []
    for index, spec in enumerate(candidate_specs):
        candidate_id, candidate_type, strategy, title, summary, assumptions, predictions, falsifications = spec
        candidates.append({
            "candidate_id": candidate_id,
            "problem_id": problem_id,
            "candidate_type": candidate_type,
            "title": title,
            "summary": f"{summary} Question: {question}",
            "formal_representation": f"{title}: map assumptions to observable outcomes for `{question}`.",
            "assumptions": assumptions,
            "construction_or_model": f"Compare candidate explanation {index + 1} against observations and adversarial alternatives.",
            "predicted_observables": predictions,
            "falsification_conditions": falsifications,
            "parent_ids": [],
            "root_candidate_id": candidate_id,
            "lineage_depth": 0,
            "generation_strategy": strategy,
            "linked_evidence_ids": evidence_ids[:],
            "linked_cluster_ids": [],
            "verification_result_ids": [],
            "failure_reason_ids": [],
            "novelty_status": "unknown" if candidate_id != "cand-counterexample" else "possibly_novel",
            "scientific_status": "proposed",
            "component_scores": {},
            "aggregate_search_score": 0.2 + index * 0.03,
            "created_step": 0,
            "updated_step": 0,
            "provenance": ["frontend_question"],
        })
    return {
        "schema_version": "v17",
        "project_id": run_id,
        "title": f"Frontend question: {question[:80]}",
        "model_mode": "mock",
        "literature_mode": "fixture",
        "grounding_mode": "strict",
        "random_seed": 11,
        "problem": {
            "problem_id": problem_id,
            "title": question[:120],
            "precise_statement": question,
            "problem_type": "mechanism_discovery",
            "scientific_domain": domain or "general_science",
            "candidate_types": ["hypothesis", "counterexample", "mechanistic_model", "experiment_plan"],
            "known_constraints": [{"constraint_id": "offline-only", "description": "Use deterministic offline optimization; do not make live model or network calls.", "hard": True}],
            "success_criteria": [{"criterion_id": "s-falsifiable", "description": "A useful hypothesis is explicit, testable, and has a falsification route."}],
            "failure_criteria": [{"criterion_id": "f-unfalsifiable", "description": "Hypotheses without falsification criteria fail cheap filtering.", "severity": "high"}],
            "observable_targets": [{"observable_id": "discriminating-observable", "description": "A measurement or calculation that separates competing explanations."}],
            "accepted_evidence_types": ["frontend_context", "user_note"],
            "excluded_claims": ["scientific proof", "live literature validation", "autonomous laboratory action"],
            "known_baselines": ["insufficient evidence baseline"],
            "corpus_scope": _context_scope(context),
            "human_notes": ["Generated from the local frontend question form; expert review remains required."],
            "provenance": ["frontend_question"],
        },
        "evidence_ids": evidence_ids,
        "initial_candidates": candidates,
        "search": {
            "mode": "beam",
            "max_steps": 12,
            "beam_width": 4,
            "max_candidates_total": 20,
            "max_children_per_candidate": 2,
            "max_lineage_depth": 4,
            "preserve_diverse_clusters": True,
            "preserve_counterexample_branch": True,
            "tournament_max_candidates": 4,
            "tournament_max_comparisons": 6,
            "tournament_debate_turns": 1,
            "tournament_ranking_mode": ranking_mode if ranking_mode in {"bounded", "elo", "bradley_terry"} else "elo",
            "tournament_max_deep_comparisons": 2,
            "adaptive_compute_enabled": True,
            "preserve_minimum_contrarian_branches": 1,
            "role_model_routing": {},
            "plateau_window": 2,
            "plateau_minimum_improvement": 0.02,
            "token_budget": 12000,
            "model_call_budget": 0,
            "verifier_call_budget": 40,
        },
        "enabled_verifiers": ["schema_constraint", "logical_consistency", "evidence_consistency", "counterexample_hook", "experimental_consistency"],
        "evaluator_only_ground_truth": {},
        "created_at": datetime.now(UTC).isoformat(),
    }


def _evidence_ids(context: str) -> list[str]:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if not lines:
        return ["frontend-context-1"]
    return [f"frontend-context-{index + 1}" for index, _ in enumerate(lines[:6])]


def _candidate_specs(question: str) -> list[tuple[str, str, str, str, str, list[str], list[str], list[str]]]:
    lowered = question.lower()
    if "bcs" in lowered or "superconduct" in lowered or "pairing" in lowered:
        return [
            (
                "cand-mixed-pairing",
                "mechanistic_model",
                "mainstream_extension",
                "Mixed BCS pairing channel",
                "A generalized BCS variational state is generated by an effective pairing kernel with separable phonon attraction and correlated-hopping terms.",
                ["Mean-field reduction is valid for the proposed low-energy Hamiltonian.", "Phonon-mediated and correlated-hopping channels can be represented in a common pairing kernel."],
                ["Gap symmetry and doping dependence should change continuously with the relative channel weights.", "A fitted two-channel kernel should outperform either channel alone on held-out materials trends."],
                ["The mixed kernel fails if no stable superconducting solution appears within physical coupling bounds.", "The mixed kernel is disfavored if single-channel models explain the same observables with fewer parameters."],
            ),
            (
                "cand-phonon-dominant",
                "hypothesis",
                "mainstream_extension",
                "Interaction-energy-lowering dominated branch",
                "Conventional attraction supplies most condensation energy, while correlated hopping acts mainly as a perturbative asymmetry correction.",
                ["Condensation energy can be decomposed into interaction and kinetic expectation-value changes.", "Material-to-material variation is mostly captured by phonon coupling strength."],
                ["Isotope-sensitive observables should track the dominant pairing contribution.", "Interaction-energy lowering should remain positive and larger than kinetic-energy lowering across most fitted cases."],
                ["The branch is weakened if kinetic-energy lowering dominates in controlled optical sum-rule or model calculations.", "It is weakened if isotope trends are absent where phonon dominance is required."],
            ),
            (
                "cand-kinetic-dominant",
                "mechanistic_model",
                "assumption_relaxation",
                "Correlated-hopping kinetic-energy branch",
                "Electron-hole-asymmetric correlated hopping contributes directly to pairing and produces measurable kinetic-energy lowering.",
                ["The hopping term survives projection into the low-energy band basis.", "Electron-hole asymmetry is strong enough to affect the superconducting condensation energy."],
                ["Optical spectral-weight or band-kinetic proxies should show doping-dependent kinetic-energy lowering below Tc.", "The kinetic contribution should grow in regimes with stronger electron-hole asymmetry."],
                ["The branch fails if kinetic-energy expectation increases or remains negligible in the superconducting state.", "It is weakened if fitted asymmetry terms are not identifiable from available data."],
            ),
            (
                "cand-separation-protocol",
                "experiment_plan",
                "counterexample_search",
                "Energy-decomposition discrimination protocol",
                "A controlled calculation or experiment estimates interaction-energy and kinetic-energy contributions separately across materials and doping.",
                ["Comparable datasets or Hamiltonian calculations exist across at least two dopings or material families.", "The decomposition convention is preregistered before fitting."],
                ["A successful protocol reports separate kinetic and interaction contributions with uncertainty intervals.", "The protocol should identify where the two-channel model is underdetermined."],
                ["The protocol is inconclusive if decomposition is gauge/model dependent beyond stated uncertainty.", "It fails if fitted contributions are not stable under leave-one-material-out tests."],
            ),
        ]
    return [
        (
            "cand-mainstream",
            "hypothesis",
            "mainstream_extension",
            "Conservative mechanism",
            "A conservative explanation extends known mechanisms and checks the most direct observable first.",
            ["The question can be represented as competing falsifiable mechanisms.", "The supplied context is incomplete and must not be treated as proof."],
            ["A targeted measurement or calculation changes if this candidate is correct.", "A negative or control condition should separate this candidate from alternatives."],
            ["The candidate is weakened if the targeted observable does not differ from alternatives.", "A simpler competing mechanism explains the same observations with fewer assumptions."],
        ),
        (
            "cand-mechanism",
            "mechanistic_model",
            "assumption_relaxation",
            "Relaxed-assumption mechanism",
            "A second mechanism relaxes one background assumption and predicts a different discriminating observable.",
            ["At least one standard assumption may be relaxed without violating known constraints.", "The relaxed assumption has an observable consequence."],
            ["The relaxed-assumption branch predicts a different response under perturbation.", "A held-out observable should separate it from the conservative branch."],
            ["The branch fails if relaxing the assumption does not alter any observable.", "It is weakened if it only rephrases the conservative branch."],
        ),
        (
            "cand-counterexample",
            "counterexample",
            "counterexample_search",
            "Counterexample branch",
            "A counterexample branch searches for a case where the leading explanation fails.",
            ["A useful candidate should survive at least one adversarial or edge-case check.", "Counterexamples are preserved only if they pass deterministic validity checks."],
            ["A boundary case should violate the leading mechanism while preserving core constraints.", "The counterexample should suggest a decisive next test."],
            ["The branch fails if the proposed counterexample violates stated constraints.", "It is weakened if it cannot be made experimentally or computationally testable."],
        ),
        (
            "cand-experiment",
            "experiment_plan",
            "mainstream_extension",
            "Discriminating experiment",
            "A focused next experiment compares the two leading mechanisms under a controlled perturbation.",
            ["At least two mechanisms predict different outcomes under one feasible perturbation.", "The measurement can be interpreted without circularly assuming the answer."],
            ["The experiment should rank mechanisms by expected separation and feasibility.", "A null result should still update the hypothesis ranking."],
            ["The plan fails if all mechanisms predict the same outcome within uncertainty.", "It is weakened if the required precision is unavailable."],
        ),
    ]


def _context_scope(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if not lines:
        return "No external corpus supplied; only the user question is available."
    return f"{min(len(lines), 6)} local user-provided context notes; not independently verified."


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:36]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{slug or 'question'}-{digest}"


CANDIDATE_COLUMNS = ["candidate_id", "title", "summary", "status", "type", "aggregate_score", "lineage_depth", "model_family"]
RATING_COLUMNS = ["candidate_id", "rating", "uncertainty", "comparisons", "wins", "losses", "draws"]
STRATEGY_COLUMNS = ["strategy", "candidates_generated", "verification_pass_rate", "novelty_yield", "score_improvement", "surviving_lineages"]
ALLOCATION_COLUMNS = ["strategy", "historical_yield", "duplicate_penalty", "falsification_penalty", "verifier_call_budget", "preserve_branch", "rationale"]
REPRODUCTION_COLUMNS = ["reproduction_result_id", "candidate_id", "outcome", "comparison"]
CLAIM_COLUMNS = ["claim_id", "claim_type", "claim_text", "status", "novelty_status", "reproduction_status", "uncertainty"]
PREDICTION_COLUMNS = ["prediction_id", "candidate_id", "observable", "status", "evaluation_status"]
VERIFIER_COLUMNS = ["candidate_id", "verifier_id", "stage", "verdict", "score", "failed"]
TOURNAMENT_COLUMNS = ["comparison_id", "candidate_a_id", "candidate_b_id", "winner_id", "single_turn", "rationale"]
TASK_COLUMNS = ["task_id", "task_type", "status", "priority", "candidate_ids", "result_artifact_ids"]
ROUTING_COLUMNS = ["role", "provider", "model", "model_mode", "max_context_characters", "max_output_tokens"]
SOURCE_COLUMNS = ["source_id", "title", "source_type", "version", "content_hash"]
IDENTIFIABILITY_COLUMNS = ["group_id", "candidate_family_ids", "identifiability_status", "discriminating_observables"]
SC_SCORE_COLUMNS = ["model_id", "candidate_id", "aggregate_score", "hamiltonian_validity", "self_consistency", "free_energy_stability", "energy_closure_score", "optical_consistency", "identifiability", "counterexample_survival"]
SC_ENERGY_COLUMNS = ["model_id", "delta_kinetic_ev", "delta_interaction_ev", "delta_correlated_hopping_ev", "delta_pairing_mean_field_ev", "free_energy_change_ev", "condensation_energy_closure_error_ev"]
SC_OPTICAL_COLUMNS = ["model_id", "full_sum", "delta_sum", "partial_sum_by_cutoff", "interpretation_warnings"]
SC_MATERIAL_COLUMNS = ["material_id", "formula", "family", "tc_k", "doping_label", "mapping_uncertainty", "unsupported_fields"]
SC_IDENTIFIABILITY_COLUMNS = ["group_id", "model_ids", "status", "observables_compared", "required_precision", "discriminating_observable"]
CLAIM_DAG_NODE_COLUMNS = ["claim_id", "candidate_id", "parent_claim_id", "claim_type", "statement", "load_bearing", "uncertainty", "repairable"]
CLAIM_DAG_EDGE_COLUMNS = ["edge_id", "parent_claim_id", "child_claim_id", "dependency_type", "load_bearing_path"]
CLAIM_DAG_GATE_COLUMNS = ["candidate_id", "terminal_status", "selected_rule", "blocker_ids_json"]
MEETING_MESSAGE_COLUMNS = ["round_number", "agent_id", "role", "critic_influenced", "content"]
PROVIDER_CALL_COLUMNS = ["call_id", "agent_id", "provider", "model", "permission_mode", "parsing_result", "input_tokens", "output_tokens"]


def _table(records: list[dict[str, object]], columns: list[str]) -> list[list[object]]:
    rows: list[list[object]] = []
    for record in records:
        rows.append([_cell(record.get(column)) for column in columns])
    return rows


def _cell(value: object) -> object:
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, sort_keys=False, default_flow_style=True).strip()
    if value is None:
        return ""
    return value


def create_gradio_workbench(*, runs_dir: str | Path = "runs"):
    try:
        import gradio as gr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Gradio is not installed. Install the optional ui extra to launch the workbench.") from exc

    service = OfflineDiscoveryFrontend(runs_dir=runs_dir)

    def ask_question(question: str, context: str, domain: str, run_id: str, ranking_mode: str):
        run_dir = service.ask_research_question(question, context=context, domain=domain or "general_science", run_id=run_id, ranking_mode=ranking_mode or "elo")
        validation_errors = service.validate(run_dir)
        validation = "valid" if not validation_errors else "\n".join(validation_errors)
        return (
            run_dir,
            validation,
            _table(service.candidate_rows(run_dir), CANDIDATE_COLUMNS),
            _table(service.elo_rating_rows(run_dir), RATING_COLUMNS),
            _table(service.strategy_performance_rows(run_dir), STRATEGY_COLUMNS),
            _table(service.adaptive_budget_rows(run_dir), ALLOCATION_COLUMNS),
            _table(service.reproduction_rows(run_dir), REPRODUCTION_COLUMNS),
            _table(service.claim_ledger_rows(run_dir), CLAIM_COLUMNS),
            _table(service.prediction_ledger_rows(run_dir), PREDICTION_COLUMNS),
            service.report_text(run_dir),
            service.copyable_summary(run_dir),
        )

    def run_atomic(project_path: str, run_id: str) -> tuple[str, str, list[dict[str, object]], list[dict[str, object]], str, str]:
        run_dir = service.run_atomic_fixture(project_path, run_id=run_id or "atomic-workbench", force=True)
        return run_dir, str(service.benchmark_metrics(run_dir)), _table(service.candidate_rows(run_dir), CANDIDATE_COLUMNS), _table(service.verifier_rows(run_dir), VERIFIER_COLUMNS), service.report_text(run_dir), service.copyable_summary(run_dir)

    def run_campaign(project_path: str, run_id: str) -> tuple[str, str, list[dict[str, object]], list[dict[str, object]], str]:
        run_dir = service.run_campaign_fixture(project_path, run_id=run_id or "rb87-workbench", force=True)
        return run_dir, str(service.campaign_comparison(run_dir)), _table(service.source_rows(run_dir), SOURCE_COLUMNS), _table(service.campaign_identifiability_rows(run_dir), IDENTIFIABILITY_COLUMNS), service.report_text(run_dir)

    def run_superconductivity(project_path: str, run_id: str):
        run_dir = service.run_superconductivity_fixture(project_path, run_id=run_id or "superconductivity-workbench", force=True)
        validation_errors = service.validate_superconductivity(run_dir)
        validation = "valid" if not validation_errors else "\n".join(validation_errors)
        return (
            run_dir,
            validation,
            _table(service.superconductivity_score_rows(run_dir), SC_SCORE_COLUMNS),
            _table(service.superconductivity_energy_rows(run_dir), SC_ENERGY_COLUMNS),
            _table(service.superconductivity_optical_rows(run_dir), SC_OPTICAL_COLUMNS),
            _table(service.superconductivity_material_rows(run_dir), SC_MATERIAL_COLUMNS),
            _table(service.campaign_identifiability_rows(run_dir), SC_IDENTIFIABILITY_COLUMNS),
            service.report_text(run_dir),
        )

    def validate(run_dir: str) -> str:
        errors = service.validate_atomic(run_dir) if (Path(run_dir) / "atomic_benchmark_metrics.json").exists() else service.validate(run_dir)
        return "valid" if not errors else "\n".join(errors)

    def inspect_search_os(run_dir: str):
        return (
            _table(service.elo_rating_rows(run_dir), RATING_COLUMNS),
            _table(service.tournament_rows(run_dir), TOURNAMENT_COLUMNS),
            _table(service.strategy_performance_rows(run_dir), STRATEGY_COLUMNS),
            _table(service.adaptive_budget_rows(run_dir), ALLOCATION_COLUMNS),
            _table(service.verifier_rows(run_dir), VERIFIER_COLUMNS),
            _table(service.reproduction_rows(run_dir), REPRODUCTION_COLUMNS),
            _table(service.task_queue_rows(run_dir), TASK_COLUMNS),
            service.checkpoint_summary(run_dir),
        )

    def inspect_ledgers(run_dir: str):
        return (
            _table(service.claim_ledger_rows(run_dir), CLAIM_COLUMNS),
            _table(service.prediction_ledger_rows(run_dir), PREDICTION_COLUMNS),
            _table(service.provider_routing_rows(run_dir), ROUTING_COLUMNS),
            service.reproduction_discrepancies(run_dir),
        )

    def feedback(run_dir: str, candidate_id: str, decision: str, rationale: str) -> str:
        return service.persist_feedback(run_dir, candidate_id=candidate_id, decision=decision, rationale=rationale)

    def build_claim_dag(run_dir: str, candidate_id: str):
        db = service.build_claim_dag_database(run_dir, candidate_id=candidate_id or None, force=True)
        errors = service.validate_claim_dag_database(run_dir)
        return (
            db,
            "valid" if not errors else "\n".join(errors),
            _table(service.claim_dag_node_rows(run_dir), CLAIM_DAG_NODE_COLUMNS),
            _table(service.claim_dag_edge_rows(run_dir), CLAIM_DAG_EDGE_COLUMNS),
            _table(service.claim_dag_gate_rows(run_dir), CLAIM_DAG_GATE_COLUMNS),
            service.claim_dag_mermaid(run_dir),
        )

    def live_meeting(run_dir: str, question: str, rounds: int, live_model: bool):
        target = run_dir.strip() or str(Path(runs_dir) / f"meeting-{_slug(question or 'research-question')}")
        final_status: dict[str, object] = {}
        transcript = ""
        rows: list[dict[str, object]] = []
        calls: list[dict[str, object]] = []
        for transcript, rows, calls, final_status in service.stream_live_agent_meeting(target, question, live_model=live_model, max_rounds=int(rounds or 2), force=True):
            yield (
                target,
                transcript,
                _table(rows, MEETING_MESSAGE_COLUMNS),
                _table(calls, PROVIDER_CALL_COLUMNS),
                final_status,
                "validating...",
            )
        errors = service.validate_live_agent_meeting(target)
        yield (
            target,
            transcript,
            _table(service.live_agent_message_rows(target), MEETING_MESSAGE_COLUMNS),
            _table(service.live_agent_provider_rows(target), PROVIDER_CALL_COLUMNS),
            final_status,
            "valid" if not errors else "\n".join(errors),
        )

    def targeted_repair(run_dir: str, claim_id: str, reason: str):
        result = service.run_targeted_repair(run_dir, claim_id=claim_id, reason=reason)
        return result, service.claim_dag_mermaid(run_dir)

    with gr.Blocks(title="Coscientist Discovery Workbench") as app:
        gr.Markdown("# Coscientist Discovery Workbench\nOffline deterministic mode. Live model/network controls are disabled by default.")
        with gr.Tab("Ask & Optimize"):
            question = gr.Textbox(lines=3, label="Research question", placeholder="Example: What mechanism could explain near-absence of Ca in recovered Ca-Fe-Al crystals?")
            context = gr.Textbox(lines=6, label="Optional local context / observations", placeholder="Add one observation per line. These are treated as local notes, not verified literature.")
            domain = gr.Textbox(value="general_science", label="Domain")
            ask_run_id = gr.Textbox(value="", label="Run ID")
            ranking_mode = gr.Dropdown(["elo", "bounded", "bradley_terry"], value="elo", label="Ranking mode")
            ask_button = gr.Button("Optimize Hypotheses")
            ask_run_dir = gr.Textbox(label="Run directory")
            ask_validation = gr.Textbox(label="Validation")
            ask_candidates = gr.Dataframe(headers=CANDIDATE_COLUMNS, label="Optimized hypotheses")
            ask_elo = gr.Dataframe(headers=RATING_COLUMNS, label="Ratings")
            ask_strategy = gr.Dataframe(headers=STRATEGY_COLUMNS, label="Strategy performance")
            ask_allocation = gr.Dataframe(headers=ALLOCATION_COLUMNS, label="Budget allocation")
            ask_reproduction = gr.Dataframe(headers=REPRODUCTION_COLUMNS, label="Reproduction checks")
            ask_claims = gr.Dataframe(headers=CLAIM_COLUMNS, label="Claim ledger")
            ask_predictions = gr.Dataframe(headers=PREDICTION_COLUMNS, label="Prediction ledger")
            ask_report = gr.Textbox(lines=24, label="Report")
            ask_copy = gr.Textbox(lines=24, label="Copyable summary", show_copy_button=True)
        with gr.Tab("Project"):
            project = gr.Textbox(value="examples/atomic_spectroscopy_fixture/project.yaml", label="Atomic benchmark YAML")
            run_id = gr.Textbox(value="atomic-workbench", label="Run ID")
            run_button = gr.Button("Run Atomic Discovery")
            campaign_project = gr.Textbox(value="examples/rb87_real_spectroscopy/project.yaml", label="Rb87 campaign YAML")
            campaign_button = gr.Button("Run Rb87 Campaign")
            run_dir = gr.Textbox(label="Run directory")
            metrics = gr.Textbox(label="Benchmark metrics")
        with gr.Tab("Dashboard"):
            validate_button = gr.Button("Validate Current Run")
            validation = gr.Textbox(label="Validation")
            deps = gr.JSON(value=service.dependency_status(), label="Dependency status")
        with gr.Tab("Dataset Inspector"):
            sources = gr.Dataframe(headers=SOURCE_COLUMNS, label="Sources")
            identifiability = gr.Dataframe(headers=IDENTIFIABILITY_COLUMNS, label="Identifiability")
        with gr.Tab("Superconductivity"):
            sc_project = gr.Textbox(value="examples/superconductivity_bcs_campaign/project.yaml", label="Superconductivity campaign YAML")
            sc_run_id = gr.Textbox(value="superconductivity-workbench", label="Run ID")
            sc_button = gr.Button("Run Superconductivity Campaign")
            sc_run_dir = gr.Textbox(label="Run directory")
            sc_validation = gr.Textbox(label="Validation")
            sc_scores = gr.Dataframe(headers=SC_SCORE_COLUMNS, label="Scientific scores")
            sc_energy = gr.Dataframe(headers=SC_ENERGY_COLUMNS, label="Energy decomposition")
            sc_optical = gr.Dataframe(headers=SC_OPTICAL_COLUMNS, label="Optical sum-rule")
            sc_materials = gr.Dataframe(headers=SC_MATERIAL_COLUMNS, label="Material mappings")
            sc_identifiability = gr.Dataframe(headers=SC_IDENTIFIABILITY_COLUMNS, label="Identifiability")
            sc_report = gr.Textbox(lines=24, label="Superconductivity report")
        with gr.Tab("Candidates"):
            candidates = gr.Dataframe(headers=CANDIDATE_COLUMNS, label="Candidate explorer")
        with gr.Tab("Verifier Inspector"):
            verifiers = gr.Dataframe(headers=VERIFIER_COLUMNS, label="Verifier results")
        with gr.Tab("Search OS"):
            inspect_button = gr.Button("Inspect Current Run")
            elo = gr.Dataframe(headers=RATING_COLUMNS, label="Elo / tournament ratings")
            tournament = gr.Dataframe(headers=TOURNAMENT_COLUMNS, label="Tournament comparisons")
            strategy = gr.Dataframe(headers=STRATEGY_COLUMNS, label="Strategy performance")
            allocation = gr.Dataframe(headers=ALLOCATION_COLUMNS, label="Adaptive budget allocation")
            reproduction = gr.Dataframe(headers=REPRODUCTION_COLUMNS, label="Independent reproduction")
            tasks = gr.Dataframe(headers=TASK_COLUMNS, label="Task queue")
            checkpoint = gr.JSON(label="Checkpoint")
        with gr.Tab("Ledgers"):
            ledger_button = gr.Button("Load Ledgers")
            claims = gr.Dataframe(headers=CLAIM_COLUMNS, label="Claim ledger")
            predictions = gr.Dataframe(headers=PREDICTION_COLUMNS, label="Prediction ledger")
            routing = gr.Dataframe(headers=ROUTING_COLUMNS, label="Per-role provider routing")
            discrepancies = gr.JSON(label="Reproduction discrepancies")
        with gr.Tab("Claim DAG"):
            claim_dag_run_dir = gr.Textbox(label="Run directory")
            claim_dag_candidate = gr.Textbox(label="Candidate ID (optional)")
            claim_dag_button = gr.Button("Build / Refresh Claim DAG DB")
            claim_dag_db = gr.Textbox(label="Claim DAG database")
            claim_dag_validation = gr.Textbox(label="Validation")
            claim_dag_nodes = gr.Dataframe(headers=CLAIM_DAG_NODE_COLUMNS, label="Claims")
            claim_dag_edges = gr.Dataframe(headers=CLAIM_DAG_EDGE_COLUMNS, label="Dependencies")
            claim_dag_gate = gr.Dataframe(headers=CLAIM_DAG_GATE_COLUMNS, label="Deterministic total gate")
            claim_dag_graph = gr.Textbox(lines=18, label="Mermaid graph", show_copy_button=True)
        with gr.Tab("Live Agent Room"):
            meeting_run_dir = gr.Textbox(label="Run directory", placeholder="Leave blank to create a new meeting run directory")
            meeting_question = gr.Textbox(lines=4, label="Research question")
            meeting_rounds = gr.Slider(1, 8, value=2, step=1, label="Meeting rounds")
            meeting_live = gr.Checkbox(value=False, label="Use live model if credentials are configured")
            meeting_button = gr.Button("Start Agent Meeting")
            meeting_out_dir = gr.Textbox(label="Meeting run directory")
            meeting_transcript = gr.Textbox(lines=26, label="Streaming transcript", show_copy_button=True)
            meeting_messages = gr.Dataframe(headers=MEETING_MESSAGE_COLUMNS, label="Agent messages")
            meeting_calls = gr.Dataframe(headers=PROVIDER_CALL_COLUMNS, label="Provider calls")
            meeting_status = gr.JSON(label="Meeting status")
            meeting_validation = gr.Textbox(label="Validation")
        with gr.Tab("Targeted Repair"):
            repair_run_dir = gr.Textbox(label="Run directory")
            repair_claim_id = gr.Textbox(label="Claim ID (optional)")
            repair_reason = gr.Textbox(lines=3, label="Repair reason")
            repair_button = gr.Button("Run Verifier-Driven Repair")
            repair_result = gr.JSON(label="Repair result")
            repair_graph = gr.Textbox(lines=18, label="Updated Mermaid graph", show_copy_button=True)
        with gr.Tab("Reports"):
            report = gr.Textbox(lines=24, label="Report")
            copyable = gr.Textbox(lines=24, label="Copyable summary", show_copy_button=True)
        with gr.Tab("Expert Review"):
            candidate_id = gr.Textbox(label="Candidate ID")
            decision = gr.Dropdown(["accept_for_further_study", "reject", "request_repair", "request_simpler_model", "request_counterexample_search", "request_new_observable", "mark_verifier_limitation", "mark_evidence_gap", "mark_expert_validated"], value="accept_for_further_study", label="Decision")
            rationale = gr.Textbox(label="Rationale")
            feedback_button = gr.Button("Append Feedback")
            feedback_path = gr.Textbox(label="Feedback artifact")
        ask_button.click(
            ask_question,
            inputs=[question, context, domain, ask_run_id, ranking_mode],
            outputs=[ask_run_dir, ask_validation, ask_candidates, ask_elo, ask_strategy, ask_allocation, ask_reproduction, ask_claims, ask_predictions, ask_report, ask_copy],
        )
        run_button.click(run_atomic, inputs=[project, run_id], outputs=[run_dir, metrics, candidates, verifiers, report, copyable])
        campaign_button.click(run_campaign, inputs=[campaign_project, run_id], outputs=[run_dir, metrics, sources, identifiability, report])
        sc_button.click(run_superconductivity, inputs=[sc_project, sc_run_id], outputs=[sc_run_dir, sc_validation, sc_scores, sc_energy, sc_optical, sc_materials, sc_identifiability, sc_report])
        validate_button.click(validate, inputs=[run_dir], outputs=[validation])
        inspect_button.click(inspect_search_os, inputs=[run_dir], outputs=[elo, tournament, strategy, allocation, verifiers, reproduction, tasks, checkpoint])
        ledger_button.click(inspect_ledgers, inputs=[run_dir], outputs=[claims, predictions, routing, discrepancies])
        claim_dag_button.click(build_claim_dag, inputs=[claim_dag_run_dir, claim_dag_candidate], outputs=[claim_dag_db, claim_dag_validation, claim_dag_nodes, claim_dag_edges, claim_dag_gate, claim_dag_graph])
        meeting_button.click(live_meeting, inputs=[meeting_run_dir, meeting_question, meeting_rounds, meeting_live], outputs=[meeting_out_dir, meeting_transcript, meeting_messages, meeting_calls, meeting_status, meeting_validation])
        repair_button.click(targeted_repair, inputs=[repair_run_dir, repair_claim_id, repair_reason], outputs=[repair_result, repair_graph])
        feedback_button.click(feedback, inputs=[run_dir, candidate_id, decision, rationale], outputs=[feedback_path])
    return app


def _version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unavailable"


if __name__ == "__main__":
    facade = create_app()
    print("OfflineDiscoveryFrontend ready.")
    print(f"Dependency status: {facade.dependency_status()}")
    if facade.dependency_status()["gradio"] == "unavailable":
        print("Gradio is not installed. Run `python -m pip install -e '.[ui]'` and start this command again.")
    else:
        create_gradio_workbench().launch()
