from __future__ import annotations

from pathlib import Path

import pytest

from coscientist.frontend import create_app, create_gradio_workbench


FIXTURE = "examples/atomic_spectroscopy_fixture/project.yaml"


def test_frontend_facade_runs_atomic_fixture_and_exposes_tables(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_atomic_fixture(FIXTURE, run_id="atomic-ui"))
    assert app.validate_atomic(run_dir) == []
    assert app.benchmark_metrics(run_dir)["hidden_model_family_recovery"] == 1.0
    assert app.candidate_rows(run_dir)
    assert app.verifier_rows(run_dir)
    assert "Atomic Benchmark Summary" in app.report_text(run_dir)


def test_frontend_persists_structured_feedback_for_atomic_run(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    run_dir = Path(app.run_atomic_fixture(FIXTURE, run_id="atomic-feedback"))
    feedback = app.persist_feedback(run_dir, candidate_id="case-a-coupled-true", decision="accept_for_further_study", rationale="Fixture-correct model.")
    assert Path(feedback).exists()


def test_gradio_workbench_reports_missing_optional_dependency() -> None:
    if create_app().dependency_status()["gradio"] != "unavailable":
        workbench = create_gradio_workbench()
        assert workbench is not None
    else:
        with pytest.raises(RuntimeError):
            create_gradio_workbench()
