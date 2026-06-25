# LSCO Phase 2 Starter Dataset Sources

This file documents `data/phase2_lsco.csv`.

The CSV is a best-effort local starter dataset for Phase 2 coverage testing. It is not a publication-grade materials database. Rows marked `usable_for_fit=false` are placeholders, nearby-doping mappings, or proxy quantities and must not be used for final material-level claims.

## Included Observables

Required observables for a complete fitting point are:

- `tc_k`
- `gap_ev`
- `penetration_depth_nm`
- `isotope_alpha`
- `optical_spectral_weight_proxy`

The current file intentionally includes missing rows when a quantity was not found in the quick source pass. This lets the Phase 2 coverage tool report the exact blocker rather than silently pretending the point is complete.

## Sources Used

### Hofer et al. 1999

Source: `https://arxiv.org/pdf/cond-mat/9912493`

Used for:

- LSCO `x=0.080` and `x=0.086` oxygen isotope exponents.
- LSCO `x=0.080` and `x=0.086` Tc values from Table I.
- Optical-conductivity maximum-energy proxy `Em`, quoted in the discussion as `0.44 eV` for `x=0.06`, `0.24 eV` for `x=0.10`, and estimated around `0.34 eV` for their samples.

Caveat:

- `Em` is not an integrated optical spectral-weight change. It is included as `optical_spectral_weight_proxy` only as a clearly marked proxy and is `usable_for_fit=false`.
- The extracted text reports normalized `lambda_ab^-2` shifts, not clean absolute `penetration_depth_nm`.

### Naqib and Islam 2011

Source: `https://arxiv.org/pdf/1103.5200`

Used for:

- Zn-free underdoped La214 isotope exponent `alpha_p = 0.271` at `x=0.09`.
- Zn-free overdoped La214 isotope exponent `alpha_p = 0.0939` at `x=0.22`.

Caveat:

- The paper focuses on isotope exponent and disorder dependence. It does not by itself provide the full five-observable Phase 2 series.

### Lemberger et al. 2010

Source: `https://arxiv.org/pdf/1010.1243`

Used for:

- Absolute LSCO film penetration-depth values inferred from Table I `lambda^-2(0)` at nominal `x=0.06`, `0.09`, `0.12`, `0.15`, `0.18`, `0.21`, `0.24`, `0.27`, and `0.30`.
- Matching `Tc(lambda^-2)` values from the same table.
- A condensate spectral-weight proxy using the tabulated `lambda^-2(0)` values, because `lambda^-2` is proportional to superfluid density / delta-function condensate spectral weight.

Conversion:

- The CSV stores `penetration_depth_nm = 1000 / sqrt(lambda^-2_um^-2)`.
- Example: `lambda^-2(0)=17.4 um^-2` gives `lambda=239.7 nm` for nominal `x=0.15`.

Caveat:

- The source reports film values; Sr doping is nominal.
- The two `x=0.27` rows and the `x=0.30` row were grown later with a different protocol, as noted by the source.
- The `optical_spectral_weight_proxy` rows from this source are condensate spectral-weight proxies, not finite-frequency optical missing-area integrals.

### Mahmood et al. 2018

Source: `https://arxiv.org/abs/1802.02101`

Used for:

- Identifying the right true optical quantity for Phase 2: normalized superfluid spectral weight `S_delta/Sn` and uncondensed spectral weight `Su/Sn`, extracted from time-domain THz spectroscopy plus mutual inductance on overdoped LSCO films.
- Documenting that the finite-frequency spectral weight balance is the Ferrell-Glover-Tinkham-style relation `Sn = S_delta + Su` in the low-frequency window used by the paper.

Caveat:

- The exact numeric `S_delta/Sn` and `Su/Sn` values are plotted in the paper's Fig. 2e rather than present as a machine-readable table in the quick source pass.
- The CSV therefore includes non-fit placeholder rows for this true optical target. These should become `usable_for_fit=true` only after digitizing the figure or importing a verified supplementary data table.

### Ino et al. 1998

Source: `https://arxiv.org/abs/cond-mat/9809311`

Used for:

- LSCO superconducting gap `Delta = 10-15 meV` for `x=0.10` and `x=0.15` near `(pi,0)`.

Caveat:

- The CSV stores midpoint `0.0125 eV` and preserves the range in `uncertainty`.

### Yoshida et al. 2012

Source: `https://arxiv.org/abs/1208.2903`

Used for:

- LSCO d-wave order parameter `Delta0 ~ 12-14 meV` for `x=0.10` and `x=0.14`.

Caveat:

- The CSV stores midpoint `0.013 eV` and preserves the range in `uncertainty`.

### Local SuperCon Fixture

Source: `examples/superconductivity_bcs_campaign/sources/supercon_fixture.csv`

Used for:

- Local fixture row `La2-xSrxCuO4`, `Tc=38 K`, `x~0.15 ambiguous`.

Caveat:

- This is a local fixture, not a live SuperCon extraction.

## Current Blocker

The dataset is not yet sufficient for final Phase 2 model comparison because no doping point currently has all five required observables as direct, fit-usable records.

The highest-priority missing quantities are:

- same-sample/direct superconducting gap values aligned with the Lemberger penetration-depth series;
- direct isotope exponents aligned with the Lemberger penetration-depth series;
- digitized or tabulated true integrated optical spectral-weight redistribution values such as `S_delta/Sn` and `Su/Sn` from Mahmood et al., not only the condensate `lambda^-2(0)` proxy;
- Tc values extracted from the same sample/doping series as the gap, isotope, and optical rows.
