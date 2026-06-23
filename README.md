# Multi-Agent AI Co-Scientist MVP

This repository contains a local-first Python MVP for a multi-agent AI co-scientist workflow. It generates competing hypotheses, reviews and attempts to falsify them, ranks them with rubric and pairwise judgments, evolves the strongest candidates, and writes a Markdown report.

The default path is fully offline. Mock LLM outputs and mock literature records are deterministic synthetic data, not scientific claims.

## Architecture

```mermaid
flowchart TD
    A[Research goal YAML] --> B[Supervisor]
    B --> C[4 async generator strategies]
    C --> D[Structured hypothesis validation]
    D --> E[Adversarial reviewer]
    E --> F[Rubric ranker]
    F --> G[Anonymous pairwise comparisons]
    G --> H[Top hypothesis selection]
    H --> I[Evolution: repair, branch, combine]
    I --> E
    G --> J[Final top hypotheses]
    J --> K[Markdown report and JSON artifacts]
```

Literature acquisition is an optional, separately gated pipeline:

```mermaid
flowchart TD
    A[Research question] --> B[SearchQuery]
    B --> C[OpenAlex and arXiv]
    C --> D[Paper normalization]
    D --> E[Deduplication]
    E --> F[Crossref metadata resolution]
    F --> G[Unpaywall and arXiv full-text location]
    G --> H[Document retrieval boundary]
    H --> I[Evidence extraction]
    I --> J[Citation verification]
    J --> K[Review]
    K --> L[Ranking and evolution]
```

OpenAlex and arXiv discover papers. Crossref resolves publication metadata. Unpaywall locates legal open-access copies. Finding a source is not the same as verifying a claim; citation verification still requires an exact passage.

V1.5B adds proximity, grounding, and meta-review artifacts:

```mermaid
flowchart TD
    A[Corpus] --> B[Grounding Packet]
    B --> C[Generation]
    C --> D[Review]
    D --> E[Ranking]
    E --> F[Evolution]
    F --> G[Evaluation]
    G --> H[Proximity Analysis]
    H --> I[Meta-Review]
    I --> J[Report]
    J --> K[Human Review]
```

V1.5C adds deterministic controlled-feedback evaluation:

```mermaid
flowchart TD
    A[Round N artifacts] --> B[MetaReviewAgent]
    B --> C[Structured recommendations]
    C --> D[RecommendationValidator]
    D --> E[RecommendationDecisions]
    E --> F[NextRoundPlan]
    F --> G[RecommendationExecutor]
    G --> H[Round N+1]
    H --> I[Evaluation]
    I --> J[Advisory vs feedback comparison]
```

Controlled feedback is disabled by default. Advisory mode persists recommendations and validation diagnostics but does not mutate the next round. Controlled-feedback mode must be explicitly enabled by project configuration or by the deterministic A/B runner's treatment branch, and only validated bounded actions can affect generation, targeted search requests, repair, branch, combine, or hold decisions. Feedback cannot enable live network/model access, change API credentials, or raise project budget ceilings.

## Providers

- `mock`: deterministic offline search, metadata, and full-text-location fixtures.
- `openalex`: searches OpenAlex works and normalizes work metadata. API keys are optional.
- `crossref`: resolves DOI and bibliographic metadata, preserving field conflicts. A contact email is optional and polite.
- `unpaywall`: locates legal open-access copies for DOI-bearing papers. Unpaywall requires an email for live enrichment.
- `arxiv`: searches/parses arXiv Atom metadata and exposes canonical abstract/PDF locations.

Agents use provider-neutral abstractions: literature search, metadata resolution, full-text location, document retrieval, and citation verification remain separate concepts.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Offline Demo

No API key is required.

```bash
python -m coscientist.cli validate examples/demo_goal.yaml
python -m coscientist.cli run examples/demo_goal.yaml --provider mock
python -m coscientist.cli search-literature "linear magnetoresistance" --providers mock
```

## V1 Grounded Pilot Workflow

V1 adds a reproducible pilot-research and evaluation loop on top of the MVP. The MVP generates, reviews, ranks, evolves, and reports hypotheses. V1 adds a persistent project specification, fixture corpus, structured claim-level evidence links, deterministic citation/evidence verification, per-round evaluation, baseline-versus-evolved comparison, and a human-review package.

Run the deterministic offline pilot:

```bash
python -m coscientist.cli project-show research-projects/interdisciplinary_fixture/project.yaml
python -m coscientist.cli run-project research-projects/interdisciplinary_fixture/project.yaml --run-id urban-heat-pilot
python -m coscientist.cli validate-artifacts runs/urban-heat-pilot
python -m coscientist.cli verify-evidence runs/urban-heat-pilot
python -m coscientist.cli evaluate-run runs/urban-heat-pilot
python -m coscientist.cli compare-rounds runs/urban-heat-pilot
python -m coscientist.cli build-review-package runs/urban-heat-pilot
```

Project literature modes:

- `fixture`: default deterministic JSONL corpus mode. No network calls.
- `existing`: resume from a saved normalized corpus JSONL with `--corpus`.
- `live`: use configured public scholarly providers, only with explicit `--live-network`.

Plan live acquisition without network calls:

```bash
python -m coscientist.cli acquire-literature \
  research-projects/interdisciplinary_fixture/project.yaml \
  --literature-mode live \
  --search-providers openalex arxiv \
  --enrichment-providers crossref unpaywall \
  --dry-run
```

Resume from an existing corpus:

```bash
python -m coscientist.cli run-project \
  research-projects/interdisciplinary_fixture/project.yaml \
  --corpus runs/some-prior-run/corpus.jsonl \
  --run-id resumed-pilot
```

Important V1 artifacts:

- `run_manifest.json`: run mode, schema versions, and artifact list.
- `project_snapshot.json`: immutable project spec snapshot.
- `resolved_configuration.json`: final project literature configuration after CLI overrides.
- `corpus.jsonl` and `normalized_papers.jsonl`: normalized paper corpus from fixture, existing, or live acquisition.
- `literature_queries.jsonl`: planned and executed provider-specific queries.
- `literature_search_events.jsonl`: provider request/cache event logs with secret-like fields redacted.
- `provider_status.json` and `provider_usage.json`: provider enablement, skips, request counts, cache hits, and failures.
- `raw_openalex_records.jsonl` and `raw_arxiv_records.jsonl`: raw or provider-normalized search records retained for audit.
- `crossref_enrichment.jsonl` and `unpaywall_enrichment.jsonl`: metadata and open-access enrichment outputs.
- `metadata_conflicts.jsonl`: unresolved metadata conflicts from deduplication or enrichment.
- `deduplication_report.json`: merge counts and conservative merge rationale.
- `corpus_manifest.json`: corpus hash, provider list, and acquisition limitations.
- `hypotheses_initial.jsonl`, `evolution_round_1.jsonl`, `evolution_round_2.jsonl`, `hypotheses_final.json`: evidence-linked hypotheses.
- `evidence_verification.jsonl`: claim-level verification records.
- `evaluation_by_round.json`: per-round rubric records.
- `round_comparison.json`: initial-versus-final comparison.
- `report.md`: pilot report.
- `human_review.md`: researcher-facing review package.

Interpret V1 scores as workflow diagnostics, not scientific truth. The evaluator is deterministic and useful for regression testing, but it can prefer its own rubric. Evidence verification checks fixture references, duplicate IDs, excerpts, unsupported claims, conflicts, and overstatement; it does not prove semantic truth.

The included pilot corpus is intentionally small and incomplete. Add another pilot by creating a project directory with `project.yaml`, `corpus.jsonl`, `rubric.yaml`, and a README, then run `run-project` against that spec.

## V1.5B And V1.5C Grounded Feedback

V1.5B artifacts describe the hypothesis landscape:

- `proximity_round_final.json`: pairwise similarity, clusters, graph nodes/edges, and search-space coverage.
- `clusters_round_final.json` and `hypothesis_graph_round_final.json`: machine-readable hypothesis landscape.
- `grounding_packets_round_final.json` and `grounding_diagnostics_round_final.json`: strict evidence context and grounding diagnostics.
- `meta_review_round_final.json` and `meta_review_decisions_round_final.json`: advisory meta-review recommendations and default-off feedback decision.
- `v15b_summary.json`: compact landscape/grounding/meta-review summary.

V1.5C artifacts evaluate whether controlled feedback changes the next round:

- `meta_review_recommendations_round_0.json`: bounded structured actions.
- `recommendation_decisions_round_0.json`: validator decisions and rejection reasons.
- `next_round_plan_round_1.json`: accepted actions normalized into a plan.
- `feedback_execution_round_1.json`: actual executor actions.
- `feedback_ab_manifest.json`: shared control/treatment baseline and permission guarantees.
- `feedback_ab_comparison.json`: diversity, grounding, quality-proxy, process, and cost metrics.
- `feedback_ab_summary.md`, `report.md`, and `human_review.md`: researcher-facing summaries and review questions.

Run the offline materials feedback comparison:

```bash
python -m coscientist.cli compare-feedback \
  examples/materials_synthesis_grounded_pilot/project.yaml \
  --runs-dir runs \
  --experiment-id materials-feedback-ab

python -m coscientist.cli validate-feedback-ab runs/materials-feedback-ab
```

The default comparison uses the mock provider, fixture or existing corpus, strict grounding, deterministic seed, no live network, and no live model. Outcome labels are bounded: `improved`, `mixed`, `no_material_change`, `regressed`, or `insufficient_evidence`. A controlled-feedback branch is not considered scientifically successful based on one metric alone.

## Live Network And Model Opt-In

Live network access cannot happen accidentally. Literature APIs and live LLMs use separate permissions.

- Literature APIs require `--live-network`.
- OpenAI-compatible model calls require `--provider openai --live-model`.
- Selecting `--provider openai` alone is rejected.
- API key presence alone never triggers a live model call.

Environment variables:

- `OPENALEX_API_KEY`: optional OpenAlex key.
- `CROSSREF_MAILTO`: optional polite Crossref contact email.
- `UNPAYWALL_EMAIL`: required for live Unpaywall mode.
- `COSCIENTIST_USER_AGENT`: user agent for provider requests.
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`: required only when intentionally using the OpenAI-compatible LLM provider.
- `OPENROUTER_APP_NAME`, `OPENROUTER_SITE_URL`: optional descriptive OpenRouter headers.

Examples:

```bash
python -m coscientist.cli search-literature \
  "linear magnetoresistance in chromium selenide" \
  --providers openalex arxiv \
  --live-network

python -m coscientist.cli resolve-doi "10.xxxx/example" \
  --provider crossref \
  --live-network

python -m coscientist.cli locate-full-text "10.xxxx/example" \
  --providers unpaywall arxiv \
  --live-network

python -m coscientist.cli run examples/demo_goal.yaml \
  --provider mock \
  --literature-providers openalex arxiv \
  --metadata-resolver crossref \
  --full-text-locators unpaywall arxiv \
  --live-network
```

Provider-only commands do not require an LLM API key.

For V1 project runs, use `acquire-literature --dry-run` first to inspect the query plan and budgets without network calls. Live project acquisition is bounded by `max_queries`, `max_results_per_query`, `max_total_results`, `max_total_requests`, and `max_requests_per_provider` in the project spec.

## V1.5A Live Model Runs

OpenRouter works through the OpenAI-compatible provider:

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=
OPENROUTER_APP_NAME=Multiagent AI Co-Scientist
OPENROUTER_SITE_URL=
```

Dry-run without model or literature calls:

```bash
python -m coscientist.cli run-project \
  research-projects/code_assistant_fixture/project.yaml \
  --provider openai \
  --live-model \
  --literature-mode fixture \
  --dry-run \
  --run-id code-assistant-live-dry-run
```

Connectivity smoke test, after explicit human approval:

```bash
python -m coscientist.cli run-project \
  research-projects/code_assistant_fixture/project.yaml \
  --provider openai \
  --live-model \
  --literature-mode fixture \
  --smoke \
  --run-id code-assistant-live-smoke
```

Mini live reasoning pilot, after smoke succeeds:

```bash
python -m coscientist.cli run-project \
  research-projects/code_assistant_fixture/project.yaml \
  --provider openai \
  --live-model \
  --literature-mode fixture \
  --max-model-calls 12 \
  --max-evolution-rounds 1 \
  --run-id code-assistant-live-mini
```

Do not combine first-time live literature debugging with first-time live model debugging. Start with `fixture` or `existing` corpus mode.

Live model project artifacts include:

- `model_calls.jsonl`: sanitized per-call metadata, schema status, retry count, usage when reported, finish reason, latency, and agent stage.
- `model_usage.json`: aggregate call count, token usage when reported, structured-output failures, and repair attempts.
- `model_provider_status.json`: provider mode, sanitized base URL host, requested model, and whether authentication was configured.

Structured outputs are parsed as JSON only. Fenced JSON can be extracted, Pydantic validation is enforced, and bounded repair attempts count against the model-call budget. API keys and authorization headers are never written to artifacts.

Compare a deterministic mock run against a live candidate:

```bash
python -m coscientist.cli compare-model-runs \
  runs/code-assistant-mock \
  runs/code-assistant-live-smoke
```

## V1.5B Proximity And Meta-Review

V1.5B runs remain offline and deterministic by default. They add:

- `ProximityAgent`: structured claim/mechanism/assumption/prediction/experiment/evidence/lineage similarity.
- `MetaReviewAgent`: artifact-aware process review with stopping assessment and next-round recommendations.
- `GroundingAgent`: bounded grounding packet and deterministic grounding diagnostics.
- Advisory feedback mode by default: recommendations are persisted but do not alter the next round.
- Controlled feedback mode: supported by schema and decision artifacts, but disabled unless project configuration explicitly enables it.

New artifacts:

- `proximity_round_final.json`
- `hypothesis_graph_round_final.json`
- `clusters_round_final.json`
- `search_space_coverage_round_final.json`
- `meta_review_round_final.json`
- `meta_review_decisions_round_final.json`
- `grounding_packets_round_final.json`
- `grounding_diagnostics_round_final.json`
- `v15b_summary.json`

Deterministic offline pilot:

```bash
PYTHONPATH=src python -m coscientist.cli run-project \
  research-projects/code_assistant_fixture/project.yaml \
  --run-id code-assistant-v15b

PYTHONPATH=src python -m coscientist.cli validate-artifacts runs/code-assistant-v15b
```

Materials-synthesis preparation pilot:

```bash
PYTHONPATH=src python -m coscientist.cli run-project \
  examples/materials_synthesis_grounded_pilot/project.yaml \
  --run-id materials-grounded-v15b
```

Grounding modes:

- `strict`: supported scientific claims must cite supplied evidence; metadata-only records cannot support claims.
- `permissive`: supplied evidence remains primary; unverified background must be labeled and cannot raise evidence scores.
- `off`: legacy behavior, clearly marked in artifacts.

Limitations: V1.5B uses deterministic lexical structured similarity, not embeddings. Meta-review is an artifact-aware decision aid, not scientific proof. UI and interactive graph rendering are deferred.

## Cache

Provider responses use a local cache at `.coscientist_cache/provider_responses` by default. Cache keys include provider, operation, normalized request, and API version. API keys, authorization headers, and contact emails are redacted from cache keys and request logs. Cache writes are atomic and corrupted cache files are ignored safely.

Configure cache behavior in `config/default.yaml`:

- `cache_enabled`
- `cache_ttl_hours`
- `force_refresh`
- `cache_dir`

## Persistence

Normal runs write JSON/JSONL artifacts under `runs/<run_id>/`. Literature-enabled runs also write:

- `provider_requests_round_0.jsonl`
- `papers_raw_round_0.json`
- `papers_normalized_round_0.json`
- `metadata_resolutions_round_0.json`
- `metadata_conflicts_round_0.json`
- `full_text_locations_round_0.json`
- `document_retrieval_round_0.json`
- `citations_round_0.json`
- `citation_verifications_round_0.json`
- `evidence_claims_round_0.json`

V1 project runs write `resolved_configuration.json`, `literature_queries.jsonl`, provider status/usage files, raw provider records, enrichment outputs, `deduplication_report.json`, and `corpus_manifest.json`. They also write `model_calls.jsonl`, `model_usage.json`, and `model_provider_status.json`. These files are enough to resume later with `--corpus runs/<run_id>/corpus.jsonl` without repeating provider calls.

## Legal Retrieval Rules

Only retrieve documents from clearly authorized locations, such as arXiv public documents, Unpaywall-reported open-access locations, or explicitly configured local documents. The system does not bypass paywalls, authentication, robots restrictions, or access-control systems.

## Testing

Default tests are offline and deterministic:

```bash
pytest
```

Optional live smoke tests are excluded unless explicitly enabled:

```bash
RUN_LIVE_API_TESTS=1 pytest -m live
```

Live tests are intentionally small and should not assume stable result ordering.

## Limitations

- Provider search ranking is not scientific confidence.
- Metadata resolution is not full-text evidence.
- Open-access PDF availability is not proof that a claim is correct.
- Document retrieval and citation verification boundaries exist, but this MVP only persists unverified placeholders unless a retrieval/verifier implementation is added.
- No Semantic Scholar, PubMed, Europe PMC, graph database, vector database, frontend, OCR, browser automation, or paywall bypass.
- Resume support is minimal through saved phase artifacts and run state.

## Troubleshooting

- If a live command fails immediately, check `--live-network` and required environment variables.
- If OpenAlex returns throttling or policy errors, reduce request volume or set an optional `OPENALEX_API_KEY`.
- If Unpaywall fails before a request, set `UNPAYWALL_EMAIL`.
- If Crossref throttles or rejects usage, set `CROSSREF_MAILTO` and reduce request volume.
- Delete `.coscientist_cache/provider_responses` or use `--force-refresh` on `run` if cached responses are stale.

## Next Planned Modules

- Passage-level document retrieval
- Citation verification against exact passages
- Richer V1/V2 semantic evidence verification
- Live-provider V1 project execution
- Proximity clustering
- Meta-review
- Benchmark evaluation
- Code execution sandbox
- Domain scientific tools
