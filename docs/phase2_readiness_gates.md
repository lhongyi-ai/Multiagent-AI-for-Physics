# Phase 2 Readiness Gates

The default status remains `blocked_insufficient_existing_data`.

Readiness statuses:

- `blocked_insufficient_existing_data`: not enough usable rows.
- `blocked_missing_optical_data`: no usable optical spectral-weight quantity.
- `blocked_missing_overlap`: observables exist but do not overlap by doping/sample.
- `ready_for_exploratory_comparison`: enough same-doping data for an exploratory comparison.
- `ready_for_held_out_comparison`: enough data with train/held-out split and multi-observable overlap.
- `comparison_complete`: comparison artifact exists.
- `comparison_inconclusive`: comparison ran but cannot support a mechanism claim.

The pipeline must not claim a material-level mechanism winner when readiness gates fail.
