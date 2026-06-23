# Atomic Adapter Development Guide

V1.8 scientific adapters must keep a hard boundary between structured model specifications and trusted package calls. Do not execute candidate-provided source code.

## Input Schema

Use `AtomicModelSpec` and related V1.8 schemas:

- `AtomicBasisState`
- `QuantumNumbers`
- `HamiltonianTerm`
- `ModelParameter`
- `ObservableRequest`
- `SpectrumObservation`
- `TransitionObservation`
- `DynamicsObservation`
- `ParameterBound`

Candidate payloads live under `CandidateSolution.structured_model.atomic_model`.

## Validation

Adapters must reject:

- unknown Hamiltonian term types
- unknown state or parameter references
- code-like strings, imports, lambdas, URLs, paths, or shell commands
- unbounded matrix dimensions
- unbounded time grids
- unbounded optimization iterations
- unsupported units

Validation failures should become `VerifierResult` records with `fail`, `inconclusive`, or `error`; they should not crash the whole search.

## Allowed Operations

Current allowlisted Hamiltonian terms:

- `diagonal_energy`
- `coherent_coupling`
- `zeeman_linear`
- `zeeman_quadratic`
- `hyperfine_scalar`
- `stark_linear`
- `stark_quadratic`
- `detuning`
- `rabi_drive`
- `custom_matrix_literal`

`custom_matrix_literal` accepts only bounded numeric matrices.

## Backend Invocation

Use `AtomicModelBuilder` to construct bounded matrices before calling packages. Internal Hamiltonian convention is ordinary frequency in Hz.

Backends currently used:

- SymPy for symbolic checks
- NumPy/SciPy for diagonalization, matching, fitting, and grid search
- QuTiP optionally for cross-check and dynamics

Do not add package calls that read files, open network connections, launch shells, or execute user callbacks.

## Result Normalization

Return package outputs through `VerifierResult`:

- `verifier_id`
- `verifier_version`
- `stage`
- `verdict`
- `score`
- `checks_passed`
- `checks_failed`
- `assumptions`
- `counterexample_found`
- `runtime_ms`
- `provenance`

Large arrays should be summarized or stored in bounded local artifacts with shape, dtype, hash, and package-version metadata.

## Artifact Persistence

Atomic runs add:

- `atomic_model_specs.jsonl`
- grouped verifier artifacts such as `symbolic_verification.jsonl`
- `atomic_candidate_equivalence.json`
- `atomic_benchmark_metrics.json`
- `atomic_benchmark_comparison.json`
- `atomic_discovery_report.md`

Evaluator-only hidden answers must stay in evaluator-only artifacts and must not appear in agent-visible project snapshots.

## Failure Handling

Adapter exceptions are captured by the `ScientificVerifier` boundary. Prefer specific failed checks when possible, such as:

- `numeric_non_hermitian`
- `spectrum_residual_exceeds_tolerance`
- `unmatched_observed_lines`
- `qutip_unavailable`
- `parameter_identifiability_warning`

## Tests

Every new adapter should include:

- schema rejection tests
- builder tests
- pass/fail/inconclusive verifier tests
- optional dependency skip tests
- artifact validation tests
- regression tests confirming offline defaults and permission gates remain intact

Package-backed verification improves encoded checking. It does not prove scientific correctness, uniqueness, or completeness; expert review remains mandatory.
