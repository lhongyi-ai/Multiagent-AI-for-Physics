# Condensation Energy Decomposition Theory Plan

## Research Objective

Determine whether the decomposition of superconducting condensation energy into bare kinetic-energy, phonon-interaction-energy, and correlated-hopping-energy contributions is uniquely defined, gauge invariant, and physically observable in a lattice superconducting model with ordinary hopping, onsite interaction, phonons, electron-phonon coupling, and density-assisted correlated hopping.

The system must not assume that a numerically closed energy ledger defines unique physical mechanism percentages.

## Main Line

The main line is theoretical and executable:

1. Gauge-couple every charge-transporting hopping term with Peierls substitution.
2. Include the density-assisted correlated-hopping term in electromagnetic response.
3. Derive paramagnetic current and diamagnetic kernel.
4. Check continuity equation and restricted lattice optical sum rule.
5. Test representation dependence of named energy components.
6. Use Hellmann-Feynman derivatives as operational diagnostics under fixed microscopic couplings.
7. Produce a Claim DAG seed and final outcome label.

## Parallel Validation Line

LSCO optical, isotope, penetration-depth, and gap data are external validation. They must not block the core theoretical result.

The LSCO line remains useful for:

- testing whether a proposed observable diagnostic has material support;
- checking finite-cutoff optical interpretations;
- validating qualitative trends after the theory audit is scoped.

It is not required for the core proof/counterexample.

## Claim DAG Tasks

- `CLM-1`: Gauge-coupled Hamiltonian is correctly defined.
- `CLM-2`: Correlated hopping contributes to the physical current operator.
- `CLM-3`: Continuity equation is satisfied.
- `CLM-4`: Lattice optical sum rule is gauge consistent.
- `CLM-5`: Total condensation energy is representation invariant under exact repartitioning.
- `CLM-6`: Individual named components are not representation invariant without fixed microscopic convention.
- `CLM-7`: Hellmann-Feynman coupling derivatives are operational model diagnostics.
- `CLM-8`: Finite ED checks support a counterexample but are not a general proof.
- `CLM-9`: Observable mapping avoids arbitrary Hamiltonian partitioning.

## Valid Outcomes

- proof of uniqueness under explicit microscopic assumptions;
- counterexample showing non-unique component partition;
- theorem identifying invariant combinations and operational decomposition conditions;
- inconclusive result with unresolved proof obligations.

## Current Checkpoint

Implemented deterministic audit artifacts:

- `research_objective.json`
- `gauge_coupled_hamiltonian.json`
- `electromagnetic_response_derivation.json`
- `representation_counterexample.json`
- `hellmann_feynman_diagnostics.json`
- `observable_classification.json`
- `energy_decomposition_verifier_results.jsonl`
- `claim_dag_seed_tasks.jsonl`
- `final_theory_outcome.json`

The current outcome is a scoped counterexample: total Hamiltonian/spectrum can remain invariant while named component expectations shift under exact repartitioning. This is evidence against treating ledger component percentages as unique physical observables. It is not a universal theorem for every allowed canonical transformation.

## Stopping Rule

If two consecutive Live Agent Meeting rounds create no new derivation artifact, counterexample artifact, executable test result, or Claim DAG status change after the round-zero tool audit, stop with `STOPPED_NO_PROGRESS`.
