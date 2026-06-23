from __future__ import annotations

from pathlib import Path

from coscientist.frontend import create_app


PROJECT = "examples/rb87_real_spectroscopy/project.yaml"


def test_frontend_campaign_facade_hides_test_rows_and_shows_sources(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_campaign_fixture(PROJECT, run_id="rb87-ui"))
    assert app.validate_campaign(run_dir) == []
    assert app.source_rows(run_dir)
    visible = app.campaign_observation_rows(run_dir)
    assert visible
    assert all(item["split"] != "test" for item in visible)
    comparison = app.campaign_comparison(run_dir)
    assert comparison["selected_family"] == "hyperfine_linear_field"
    assert app.campaign_identifiability_rows(run_dir)


def test_frontend_campaign_feedback_persists(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_campaign_fixture(PROJECT, run_id="rb87-feedback"))
    path = app.persist_feedback(run_dir, candidate_id="hyperfine_linear_field", decision="accept_campaign_result", rationale="Ready for expert inspection.")
    assert Path(path).exists()
