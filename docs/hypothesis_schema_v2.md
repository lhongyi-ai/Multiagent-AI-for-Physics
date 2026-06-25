# Hypothesis Schema V2

Hypothesis V2 is an additive migration layer. It does not replace existing V1 hypothesis schemas.

V2 adds:

- scoped claim
- task type and domain ID
- predictions and falsification criteria
- required tools
- hard-gate results
- score vector with provenance
- lineage and mutation metadata
- failure-memory references

The migration helper accepts existing hypothesis-like dictionaries and produces `HypothesisV2` records for Optimizer V2.
