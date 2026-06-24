from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from coscientist.discovery import run_discovery_project, validate_discovery_artifacts
from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_json, read_jsonl
from coscientist.schemas.v20 import AdaptiveBudgetAllocation, ProviderRoutingPlan, RoleModelRoute


FIXTURE = Path("examples/discovery_search_fixture/project.yaml")


def _elo_project(tmp_path: Path) -> Path:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["search"]["tournament_ranking_mode"] = "elo"
    data["search"]["tournament_max_deep_comparisons"] = 2
    data["search"]["role_model_routing"] = {"generation": "mock-generator-small", "comparison": "mock-comparison-small"}
    project = tmp_path / "elo_project.yaml"
    project.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return project


def test_elo_tournament_updates_ratings_and_avoids_repeated_pairings(tmp_path: Path) -> None:
    run_dir = run_discovery_project(_elo_project(tmp_path), runs_dir=tmp_path, run_id="elo")
    assert validate_discovery_artifacts(run_dir) == []
    state = read_json(run_dir / "elo_tournament_state.json")
    pairings = state["completed_pairings"]
    assert state["ranking_mode"] == "elo"
    assert pairings
    assert len(pairings) == len(set(pairings))
    assert len(state["deep_comparison_pairings"]) <= 2
    assert any(rating["comparisons"] > 0 and rating["uncertainty"] < 350 for rating in state["ratings"])
    comparisons = read_jsonl(run_dir / "tournament_comparisons.jsonl")
    assert any(not item["single_turn"] for item in comparisons)


def test_adaptive_allocation_routing_and_reproduction_are_bounded(tmp_path: Path) -> None:
    run_dir = run_discovery_project(_elo_project(tmp_path), runs_dir=tmp_path, run_id="bounded")
    allocation = AdaptiveBudgetAllocation.model_validate_json((run_dir / "adaptive_budget_allocation.json").read_text(encoding="utf-8"))
    assert sum(item.token_budget for item in allocation.allocations) <= allocation.total_token_budget
    assert sum(item.model_call_budget for item in allocation.allocations) == 0
    assert any(item.strategy == "counterexample_search" and item.preserve_branch for item in allocation.allocations)

    routing = ProviderRoutingPlan.model_validate_json((run_dir / "provider_routing_plan.json").read_text(encoding="utf-8"))
    assert {route.role for route in routing.routes} >= {"generation", "review", "comparison", "deep_reasoning", "novelty_audit", "meta_review"}
    assert all(route.provider == "mock" and route.model_mode == "mock" for route in routing.routes)
    assert read_json(run_dir / "model_usage.json")["call_count"] == 0

    reproduction = read_jsonl(run_dir / "reproduction_results.jsonl")
    assert reproduction
    assert all(len(item["runs"]) >= 2 for item in reproduction)
    assert all(item["outcome"] in {"reproduced", "partially_reproduced", "not_reproduced", "inconclusive"} for item in reproduction)


def test_provider_routing_rejects_live_routes_without_permission() -> None:
    with pytest.raises(ValidationError):
        ProviderRoutingPlan(
            project_id="p",
            run_id="r",
            live_model_enabled=False,
            routes=[RoleModelRoute(role="generation", provider="openai_compatible", model="gpt-test", model_mode="live", live_permission_required=True)],
        )


def test_frontend_exposes_v20_artifact_views(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_fixture(_elo_project(tmp_path), run_id="frontend-v20"))
    assert app.validate(run_dir) == []
    assert app.elo_rating_rows(run_dir)
    assert app.tournament_rows(run_dir)
    assert app.strategy_performance_rows(run_dir)
    assert app.adaptive_budget_rows(run_dir)
    assert app.task_queue_rows(run_dir)
    assert app.checkpoint_summary(run_dir)["run_id"] == "frontend-v20"
    assert app.claim_ledger_rows(run_dir)
    assert app.prediction_ledger_rows(run_dir)
    assert app.reproduction_rows(run_dir)
    assert app.provider_routing_rows(run_dir)


def test_frontend_ask_research_question_optimizes_hypotheses(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.ask_research_question(
        "What mechanism could explain a surprising missing element signal in a synthesized intermetallic?",
        context="EDX shows a low signal.\nBulk composition has not been measured.",
        domain="materials_synthesis",
        run_id="ask-ui",
    ))
    assert app.validate(run_dir) == []
    rows = app.candidate_rows(run_dir)
    assert rows
    assert any(row["status"] in {"promising", "partially_verified", "expert_review_required"} for row in rows)
    assert app.elo_rating_rows(run_dir)
    assert app.claim_ledger_rows(run_dir)
    assert "Discovery Search Report" in app.report_text(run_dir)
    summary = app.copyable_summary(run_dir)
    assert "Optimized Hypotheses" in summary
    assert "Claim Ledger" in summary
    assert "Prediction Ledger" in summary
