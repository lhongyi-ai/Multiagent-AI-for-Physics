# Expected V1 Run Artifacts

A completed deterministic pilot run should contain:

```text
run_manifest.json
project_snapshot.json
corpus.jsonl
normalized_papers.jsonl
hypotheses_initial.jsonl
reviews.jsonl
evidence_verification.jsonl
rankings.jsonl
evolution_round_1.jsonl
evolution_round_2.jsonl
hypotheses_final.json
evaluation_by_round.json
round_comparison.json
lineage.json
report.md
human_review.md
```

The repository may also contain legacy MVP artifacts in the same run directory, such as `hypotheses_initial.json`, `ranking_round_0.json`, and `final_report.md`.
