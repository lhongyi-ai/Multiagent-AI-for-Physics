from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from coscientist.atomic.discovery import run_atomic_discovery_project, validate_atomic_discovery_artifacts
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
        return str(persist_expert_feedback(run_dir, candidate_id=candidate_id, decision=decision, rationale=rationale, reviewer=reviewer))

    def dependency_status(self) -> dict[str, str]:
        return {name: _version(name) for name in ["sympy", "numpy", "scipy", "qutip", "gradio"]}

    def run_atomic_fixture(self, project_path: str | Path, *, run_id: str = "atomic-frontend-smoke", force: bool = True) -> str:
        run_dir = run_atomic_discovery_project(project_path, runs_dir=self.runs_dir, run_id=run_id, force=force)
        return str(run_dir)

    def validate_atomic(self, run_dir: str | Path) -> list[str]:
        return validate_atomic_discovery_artifacts(run_dir)

    def candidate_rows(self, run_dir: str | Path) -> list[dict[str, object]]:
        path = Path(run_dir)
        rows = []
        if not (path / "candidate_archive.jsonl").exists():
            return rows
        for item in read_jsonl(path / "candidate_archive.jsonl"):
            model = item.get("structured_model", {}).get("atomic_model", {})
            rows.append({
                "candidate_id": item.get("candidate_id"),
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

    def report_text(self, run_dir: str | Path) -> str:
        path = Path(run_dir)
        for name in ["atomic_discovery_report.md", "discovery_report.md", "report.md"]:
            candidate = path / name
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        return ""

    def benchmark_metrics(self, run_dir: str | Path) -> dict[str, object]:
        path = Path(run_dir) / "atomic_benchmark_metrics.json"
        return read_json(path) if path.exists() else {}


def create_app(*, runs_dir: str | Path = "runs") -> OfflineDiscoveryFrontend:
    return OfflineDiscoveryFrontend(runs_dir=runs_dir)


def create_gradio_workbench(*, runs_dir: str | Path = "runs"):
    try:
        import gradio as gr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Gradio is not installed. Install the optional ui extra to launch the workbench.") from exc

    service = OfflineDiscoveryFrontend(runs_dir=runs_dir)

    def run_atomic(project_path: str, run_id: str) -> tuple[str, str, list[dict[str, object]], list[dict[str, object]], str]:
        run_dir = service.run_atomic_fixture(project_path, run_id=run_id or "atomic-workbench", force=True)
        return run_dir, str(service.benchmark_metrics(run_dir)), service.candidate_rows(run_dir), service.verifier_rows(run_dir), service.report_text(run_dir)

    def validate(run_dir: str) -> str:
        errors = service.validate_atomic(run_dir) if (Path(run_dir) / "atomic_benchmark_metrics.json").exists() else service.validate(run_dir)
        return "valid" if not errors else "\n".join(errors)

    def feedback(run_dir: str, candidate_id: str, decision: str, rationale: str) -> str:
        return service.persist_feedback(run_dir, candidate_id=candidate_id, decision=decision, rationale=rationale)

    with gr.Blocks(title="Coscientist Discovery Workbench") as app:
        gr.Markdown("# Coscientist Discovery Workbench\nOffline deterministic mode. Live model/network controls are disabled by default.")
        with gr.Tab("Project"):
            project = gr.Textbox(value="examples/atomic_spectroscopy_fixture/project.yaml", label="Project YAML")
            run_id = gr.Textbox(value="atomic-workbench", label="Run ID")
            run_button = gr.Button("Run Atomic Discovery")
            run_dir = gr.Textbox(label="Run directory")
            metrics = gr.Textbox(label="Benchmark metrics")
        with gr.Tab("Dashboard"):
            validate_button = gr.Button("Validate Current Run")
            validation = gr.Textbox(label="Validation")
            deps = gr.JSON(value=service.dependency_status(), label="Dependency status")
        with gr.Tab("Candidates"):
            candidates = gr.Dataframe(label="Candidate explorer")
        with gr.Tab("Verifier Inspector"):
            verifiers = gr.Dataframe(label="Verifier results")
        with gr.Tab("Reports"):
            report = gr.Textbox(lines=24, label="Report")
        with gr.Tab("Expert Review"):
            candidate_id = gr.Textbox(label="Candidate ID")
            decision = gr.Dropdown(["accept_for_further_study", "reject", "request_repair", "request_simpler_model", "request_counterexample_search", "request_new_observable", "mark_verifier_limitation", "mark_evidence_gap", "mark_expert_validated"], value="accept_for_further_study", label="Decision")
            rationale = gr.Textbox(label="Rationale")
            feedback_button = gr.Button("Append Feedback")
            feedback_path = gr.Textbox(label="Feedback artifact")
        run_button.click(run_atomic, inputs=[project, run_id], outputs=[run_dir, metrics, candidates, verifiers, report])
        validate_button.click(validate, inputs=[run_dir], outputs=[validation])
        feedback_button.click(feedback, inputs=[run_dir, candidate_id, decision, rationale], outputs=[feedback_path])
    return app


def _version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unavailable"


if __name__ == "__main__":
    app = create_app()
    print("OfflineDiscoveryFrontend ready.")
    print(f"Dependency status: {app.dependency_status()}")
    print("Use create_gradio_workbench().launch() when Gradio is installed.")
