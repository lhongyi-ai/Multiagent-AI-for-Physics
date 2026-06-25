# Platform Generalization and Optimizer V2 Plan

## Baseline

This branch starts from a clean deterministic baseline:

- Command: `PYTHONPATH=src python -m pytest`
- Result: `165 passed, 4 skipped`

No live model calls or live network calls are required for this baseline.

## Current Architecture

The repository already has a working staged research platform:

- V1-V1.6 project execution, evidence grounding, closed-question answering, and reports.
- V1.7-V1.9 discovery, verifier, campaign, archive, and frontend workbench flows.
- V2.1-V2.5 superconductivity workflows, including live-agent meeting artifacts, Claim DAG storage, a minimal mixed BCS solver, LSCO Phase 2 data acquisition, staging, review, promotion, digitization queue, readiness gates, and comparison readiness artifacts.
- Provider and permission gates for mock, live model, fixture literature, existing corpus, and live scholarly-data modes.
- A Gradio-facing backend facade in `coscientist.frontend`.

## Current LSCO Coupling

The Phase 2 acquisition path is scientifically useful but still centered on LSCO:

- LSCO aliases, observable names, readiness gates, query templates, and staging rules live in the superconductivity acquisition module.
- CLI commands such as `phase2-acquire` and frontend methods are LSCO-specific.
- The acquisition executor can search, resolve, locate, classify, parse deterministic fixture/table-like inputs, and queue digitization, but live PDF/XLSX/table extraction remains partial.

## Target Architecture

The immediate target is a compatibility layer, not a destructive directory migration:

- Add a domain-independent `core` package for scientific task types, DomainPack contracts, generic acquisition adapters, Hypothesis V2, and Optimizer V2.
- Add `domain_packs` modules for LSCO, CrSe magnetic transport, mathematical physics, and XRD phase identification.
- Keep existing LSCO commands and workflows working.
- Route new generic commands and frontend panels through the same backend services.

Domain-specific content belongs in Domain Packs:

- aliases
- observable ontology
- search query templates
- source classifiers
- readiness gates
- benchmark cases
- domain guardrails

Generic core code must not hard-code LSCO, CrSe, XRD, or superconductivity-specific readiness semantics.

## Migration Map

- `coscientist.superconductivity.phase2_acquisition` remains the implementation of the current LSCO vertical.
- `coscientist.domain_packs.superconductivity_lsco` wraps the LSCO vertical as a Domain Pack.
- `coscientist.core.acquisition.GenericDataAcquisitionAgent` dispatches by Domain Pack and delegates to the existing LSCO acquisition path where appropriate.
- Existing `phase2-*` CLI commands remain stable.
- New `domains-*`, `acquisition-run`, `hypotheses-*`, `failures-list`, and `benchmark-run` commands provide generic access.

## Backward Compatibility

- Existing artifacts, CLI commands, project specs, and tests remain valid.
- No existing schema is replaced in place.
- Hypothesis V2 is introduced as a migration target and artifact format, not a forced rewrite of V1 hypotheses.
- Live permissions remain explicit and disabled by default.

## Milestones

1. DomainPack protocol and registry.
2. ScientificTaskType policies.
3. Generic acquisition adapter with LSCO delegation and fixture support for non-LSCO packs.
4. Hypothesis V2 schema and migration helper.
5. Optimizer V2 checkpoint: hard gates, cheap kill tests, score provenance, Pareto frontier, portfolio roles, mutation operators, counterexample/EIG/failure-memory artifacts.
6. CLI and frontend exposure.
7. Cross-domain deterministic benchmark smoke tests.
8. Later milestone: deeper real PDF, TeX, XLSX, HTML extraction and robust comparison readiness transitions.

## Risks

- The complete requested platform is larger than one safe implementation pass.
- Adding UI without backend artifacts can create misleading scientific confidence.
- Live extraction quality depends on source availability, legal access, parser coverage, and human review.
- Domain Packs can look complete even when only fixture benchmarks exist, so status labels must be explicit.

## Acceptance Criteria For This Checkpoint

- The test baseline is recorded.
- Four Domain Packs are discoverable and inspectable.
- Generic acquisition works offline for LSCO and at least one non-LSCO domain.
- Hypothesis V2 migration works without breaking V1 schemas.
- Optimizer V2 produces deterministic artifacts for hard gates, cheap kills, score vectors, Pareto portfolio, mutations, counterexample tasks, EIG actions, and failure memory.
- CLI and frontend expose the new backend services.
- Focused tests pass.
- Full deterministic suite passes.

## Non-Claims

This checkpoint does not claim that the platform has solved LSCO, CrSe, XRD, or mathematical-physics research problems. A domain is considered operational only for the deterministic benchmark path that is explicitly tested.
