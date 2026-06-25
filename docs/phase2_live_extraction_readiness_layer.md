# Phase 2 Live Extraction, Curation, and Readiness Validation Layer

This layer extends `Phase2DataAcquisitionAgent` beyond source discovery into staged extraction, curation, review, readiness gating, and comparison integrity.

## Implemented Milestones

### Milestone 1: Extraction

Implemented deterministic parser entry points:

- TeX/text-style source parsing via explicit numeric patterns.
- PDF text fallback path through the same conservative text parser.
- Supplementary CSV parser.
- Supplementary ZIP parser for contained CSV files.
- Parser fallback reports.

The parser extracts only explicit values with units and doping. It does not infer graph coordinates or hallucinate values from prose.

### Milestone 2: Scientific Standardization

Implemented:

- Observable ontology.
- Sample identity schema.
- Nominal/measured/inferred/unknown doping definitions.
- Unit normalization for K, meV/eV, nm/um, isotope alpha, and dimensionless optical ratios.
- Measurement method and observable definition metadata.

### Milestone 3: Review And Conflict

Implemented:

- Candidate-row review decisions: approve, reject, edit.
- Reviewed promotion path.
- Conflict and duplicate detection before canonical promotion.
- Promotion audit artifacts.

Default behavior remains staged-only. Canonical promotion requires explicit review or explicit auto-promotion.

### Milestone 4: Optical Digitization

Implemented:

- Figure-only detection.
- Digitization queue records.
- Reviewed digitized CSV import with uncertainty fields.

Automatic graph coordinate extraction is not claimed. Figure-derived points remain review-gated.

### Milestone 5: Claim DAG Linkage

Implemented acquisition-level data claims:

```text
paper discovered
-> value extracted
-> candidate row staged
-> readiness gate
-> material-level comparison gate
```

These claims are written as `data_claims.jsonl` and can be consumed by the broader Claim DAG tooling.

### Milestone 6: Readiness Adversarial Tests

Implemented readiness gates that distinguish:

- bad provenance
- missing optical data
- missing same-doping overlap
- ambiguous gap or optical definitions
- held-out split unavailable
- exploratory-ready versus held-out-ready states

### Milestone 7: Comparison Robustness

Implemented comparison-readiness artifacts:

- paper/sample grouped holdout requirement
- per-observable metrics list
- model complexity penalty requirement
- bootstrap uncertainty requirement
- parameter identifiability requirement
- `comparison_inconclusive` as a valid bounded state

The runner still blocks material-level comparison unless readiness gates pass.

## Current Scientific Status

The canonical LSCO dataset still does not satisfy material-level held-out comparison gates. The correct status remains a blocked state until enough reviewed, overlapping, provenance-complete data are promoted.

## Known Limitations

- The current PDF support is text-parser compatible, not a full table-layout PDF engine.
- XLSX parsing is not implemented without adding a dependency; ZIP-contained CSV is supported.
- Automatic figure digitization is intentionally not implemented.
- Live extraction from arbitrary paper full text remains conservative and metadata-first.
