# Phase 2 Live Data Acquisition Plan

## Baseline

Baseline command run before implementation:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

Result: `155 passed, 4 skipped, 51 warnings`.

## Existing Architecture

- Agent and orchestration code lives under `src/coscientist/agents`, `src/coscientist/orchestration`, `src/coscientist/live_agents.py`, and domain workflows under `src/coscientist/superconductivity`.
- Literature connectors already exist under `src/coscientist/literature/providers` for arXiv, OpenAlex, Crossref, and Unpaywall.
- Live-network permission is explicit through CLI flags and guarded by provider names.
- Artifacts are written through `src/coscientist/pilot/artifacts.py`.
- The current LSCO Phase 2 coverage tool is `src/coscientist/superconductivity/phase2_data.py`.
- The canonical compatibility dataset is `data/phase2_lsco.csv`; source notes are in `data/phase2_lsco_sources.md`.
- Frontend adapters are concentrated in `src/coscientist/frontend.py`.
- Claim DAG persistence exists in `src/coscientist/claim_dag.py`.

## Current Blockers

- Phase 1 minimal mixed BCS/correlated-hopping computation is executable.
- Phase 2 material-level comparison remains blocked because no curated LSCO doping series has full overlap across `tc_k`, `gap_ev`, `isotope_alpha`, `penetration_depth_nm`, and optical spectral-weight quantities.
- Some needed optical data appear only in figures and must enter a digitization queue rather than be fabricated.

## Proposed Agent And Worker Architecture

`Phase2DataAcquisitionAgent` owns LSCO experimental data acquisition. It submits an `AcquisitionTask` to an `AcquisitionExecutor`.

Executors:

- `MockAcquisitionExecutor`: deterministic fixture mode for tests.
- `LocalAcquisitionExecutor`: in-process worker for existing and live modes.
- `RemoteAcquisitionExecutor`: documented extension point; not deployed in this phase.

The executor boundary keeps local Gradio/API orchestration separate from network-enabled acquisition work. It can later be replaced by a cloud worker while preserving task/result schemas.

## Data Flow

```text
Frontend / CLI
-> Phase2DataAcquisitionAgent
-> AcquisitionExecutor
-> query planner
-> arXiv/OpenAlex/Crossref/Unpaywall or deterministic fixtures
-> paper registry and deduplication
-> relevance/observable classification
-> text/table extraction or figure-digitization task creation
-> candidate rows staged
-> validation and conflict checks
-> optional promotion gate
-> canonical CSV update only for approved rows
-> coverage rerun
-> comparison trigger decision
```

## Schemas

Core schemas live in `coscientist.superconductivity.phase2_acquisition`:

- `AcquisitionTask`
- `AcquisitionTaskHandle`
- `AcquisitionTaskStatus`
- `AcquisitionResult`
- `PaperRecord`
- `PaperClassification`
- `ExtractionRecord`
- `CandidateDataRow`
- `DigitizationTask`
- `PromotionDecision`

The canonical CSV remains backward compatible. Staging records use normalized JSONL because a wide CSV cannot safely preserve paper, sample, observable, and provenance state for unreviewed candidates.

## Promotion Gates

Rows are promoted only when deterministic gates pass:

- LSCO identity and explicit doping.
- Recognized observable and unit.
- Deterministic unit normalization.
- Paper identity and source location.
- High confidence, unless explicit review is supplied.
- No unresolved conflict or exact duplicate.
- Optical quantities must retain their normalization definition.

Default configuration disables automatic promotion.

## Failure Handling

The worker records failed searches, network-disabled live runs, inaccessible full text, malformed responses, figure-only data, unit ambiguity, duplicate rows, and conflicts as artifacts. Live mode without permission returns `blocked_live_network_permission_required` and does not fall back to fixture data.

## Compatibility Risks

- Existing Phase 2 coverage accepts the current narrow CSV schema. The acquisition layer stages richer JSONL records and exports compatible rows only after validation.
- Live provider behavior may vary by API availability; deterministic tests use fixtures and no network.
- Figure digitization is queued and review-gated; automatic graph digitization is intentionally not claimed.

## Testing Strategy

- Unit tests for query generation, fixture acquisition, live permission rejection, deduplication, classification, extraction, staging, validation, promotion, digitization queue, coverage rerun, checksum-triggered comparison blocking, and frontend adapters.
- Existing V1-V2.4 tests must remain passing.
- No automated test requires live network or API keys.

## Acceptance Criteria

- A deterministic fixture acquisition run produces all required acquisition artifacts.
- A live-mode run without `live_network=True` returns an explicit blocked state.
- Candidate rows are staged first and canonical data are unchanged unless promotion gates pass.
- Approved fixture rows can be promoted atomically to a supplied canonical CSV copy.
- Coverage reruns after promotion.
- Frontend facade exposes acquisition status, sources, staged rows, data gaps, digitization queue, readiness, and comparison trigger information.
