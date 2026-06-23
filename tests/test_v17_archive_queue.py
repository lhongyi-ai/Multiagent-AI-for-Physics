from __future__ import annotations

from coscientist.discovery import CandidateArchive, SearchTaskQueue
from coscientist.schemas.v17 import CandidateSolution, SearchTask


def _candidate(candidate_id: str, *, formal: str = "same model") -> CandidateSolution:
    return CandidateSolution(
        candidate_id=candidate_id,
        problem_id="p",
        candidate_type="hypothesis",
        title=candidate_id,
        summary="Candidate",
        formal_representation=formal,
        assumptions=["a"],
        predicted_observables=["p"],
        falsification_conditions=["f"],
        generation_strategy="mainstream_extension",
        created_step=0,
        updated_step=0,
    )


def test_candidate_archive_tracks_duplicates_and_lineage() -> None:
    archive = CandidateArchive("p")
    archive.add(_candidate("cand-a"))
    archive.add(_candidate("cand-b"))
    archive.add(_candidate("cand-c", formal="different"))
    assert archive.duplicate_groups() == [["cand-a", "cand-b"]]
    assert archive.lineage_graph()["cand-a"] == []


def test_queue_respects_dependencies_and_completion() -> None:
    queue = SearchTaskQueue([
        SearchTask(task_id="a", problem_id="p", task_type="formalize_problem", created_step=0),
        SearchTask(task_id="b", problem_id="p", task_type="verify_candidate", dependencies=["a"], created_step=0),
    ])
    assert [task.task_id for task in queue.ready()] == ["a"]
    queue.mark("a", "completed", step=1)
    assert [task.task_id for task in queue.ready()] == ["b"]
