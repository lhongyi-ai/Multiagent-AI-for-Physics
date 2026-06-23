from __future__ import annotations

from pathlib import Path

from coscientist.frontend import create_app
from coscientist.pilot.artifacts import read_jsonl


FIXTURE = "examples/discovery_search_fixture/project.yaml"


def test_frontend_facade_uses_backend_runtime_and_feedback_store(tmp_path: Path) -> None:
    app = create_app(runs_dir=tmp_path)
    summary = app.load_project(FIXTURE)
    assert summary["project_id"] == "discovery-search-fixture"
    run_dir = Path(app.run_fixture(FIXTURE, run_id="frontend"))
    assert app.validate(run_dir) == []
    feedback_path = Path(app.persist_feedback(run_dir, candidate_id="cand-modest-pass", decision="hold", rationale="Human wants bulk check first."))
    records = read_jsonl(feedback_path)
    assert records[-1]["candidate_id"] == "cand-modest-pass"
    assert records[-1]["decision"] == "hold"
