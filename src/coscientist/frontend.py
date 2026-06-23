from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coscientist.discovery import (
    load_discovery_project,
    persist_expert_feedback,
    run_discovery_project,
    validate_discovery_artifacts,
)


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
        return str(persist_expert_feedback(run_dir, candidate_id=candidate_id, decision=decision, rationale=rationale, reviewer=reviewer))


def create_app(*, runs_dir: str | Path = "runs") -> OfflineDiscoveryFrontend:
    return OfflineDiscoveryFrontend(runs_dir=runs_dir)


if __name__ == "__main__":
    app = create_app()
    print("OfflineDiscoveryFrontend ready. Import create_app() to wire a UI without changing the backend.")
