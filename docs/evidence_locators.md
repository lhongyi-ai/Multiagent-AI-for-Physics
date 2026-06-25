# Evidence Locators

V3.0 requires citations to support the exact scientific claim, not merely point to an existing artifact.

## Implemented Locator Types

The deterministic validator currently supports:

- `JSON_POINTER`
- `JSONL_RECORD`
- `CSV_CELL`
- `CSV_ROW`
- `MATRIX_ELEMENT`

Other locator types are schema-visible but not yet executed by the deterministic validator.

## Validation Checks

`ArtifactClaimSupportValidator` checks:

- artifact path stays inside the run directory;
- artifact exists;
- optional SHA-256 checksum matches;
- internal location resolves;
- `required_condition` passes.

Supported conditions include bounded scalar comparisons such as:

```text
observed_value < 1e-12
observed_value > 0.001
observed_value == true
```

## Regression Guard

The V3 tests verify that `phase1_minimal_bcs_tool/minimal_bcs_verifier_results.jsonl` cannot support an unrelated continuity-equation claim unless the exact location and condition resolve.
