# LLNL Open AI Co-Scientist Design Comparison

Reference: `https://github.com/llnl/open-ai-co-scientist`

This note records design ideas inspected from LLNL's public implementation and how they map onto this repository's V1.5B architecture.

## Ideas Adopted

- Explicit `ProximityAgent` role after ranking/evolution to analyze hypothesis neighborhoods.
- Explicit `MetaReviewAgent` role to evaluate the overall research process, not only individual hypotheses.
- Hypothesis graph artifact that can later support an interactive graph UI.
- Cycle-end meta-review recommendations that can guide future rounds.
- OpenRouter-compatible model-selection concept for future UI work, while keeping CLI permission gates.

## Ideas Adapted

- LLNL uses proximity as a simpler whole-hypothesis similarity layer. This repository uses structured multi-dimensional similarity over claims, mechanisms, assumptions, predictions, experiments, evidence, and lineage.
- LLNL's UI-driven workflow inspired graph-readiness, but this repository persists machine-readable artifacts first and keeps UI deferred.
- LLNL's meta-review role inspired the agent boundary. This repository constrains meta-review with strict Pydantic schemas, artifact references, evidence-verification references, and deterministic offline output.
- LLNL's arXiv integration is useful as a product feature reference. This repository keeps its own arXiv Atom provider so requests stay behind the existing `--live-network`, cache, and artifact pipeline.

## Ideas Deferred

- Gradio UI and interactive hypothesis graph.
- Full Elo or Bradley-Terry replacement for the current rubric and pairwise ranking.
- Trend dashboards and broad arXiv exploration UI.
- Hugging Face or hosted deployment workflow.
- Multi-model tournament controls.

## Ideas Not Copied

- Unbounded UI-driven live model selection without explicit artifact-mode tracking.
- Proximity logic that collapses all structured hypothesis fields into one opaque text blob.
- Any behavior that bypasses this repository's deterministic tests, provenance, secret redaction, artifact validation, or separate live-model/live-network permission gates.

## Architectural Tradeoff

LLNL's repository is stronger as an interactive prototype. This repository is optimized for reproducible, auditable research runs with strict artifacts, deterministic offline tests, controlled live-provider gates, and schema-validated structured outputs.
