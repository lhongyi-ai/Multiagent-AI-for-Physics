# V3.0 Proof-Carrying Scientific Search Plan

## Baseline

- Branch: `feature/v3-proof-carrying-scientific-search`
- Baseline tests: `PYTHONPATH=src pytest -q` -> `183 passed, 4 skipped`
- Baseline demo: `PYTHONPATH=src python -m coscientist.cli run-closed-loop-demo --runs-dir /private/tmp/coscientist-v3-baseline --run-id v28-demo --force` -> valid

## Findings

The V2.8 closed-loop executor can run deterministic tools and attach action results to the Claim DAG, but the scientific state is still too permissive for V3.0:

- Claim DAG updates still rely heavily on artifact presence and broad verifier summaries.
- Agent meeting messages can contain completion language that is not tied to a completed action execution.
- Multiple agents can imply incompatible scientific states in prose without a deterministic state-dispute artifact.
- Existing artifacts are not required to cite exact internal locations plus deterministic acceptance conditions.
- Missing capabilities stop or defer work instead of always producing explicit tool-gap and build-action records.

## Implementation Scope For This Checkpoint

This checkpoint implements a focused, tested V3.0 foundation rather than pretending to complete the full long-term autonomous tool-evolution system.

1. Add strict V3 schemas for proof obligations, evidence locators, scientific certificates, state snapshots, proof-carrying claim updates, transition records, state disputes, tool capabilities, tool gaps, tool-build records, independent verification, and candidate archive entries.
2. Add deterministic scientific-state transition logic:
   - agents cannot directly transition claims into verified states;
   - accepted transitions require a completed execution ID, certificate, evidence locators, acceptance conditions, and independent verification where required;
   - stale snapshot versions reject updates.
3. Add `ArtifactClaimSupportValidator`:
   - validate artifact existence, checksum, exact JSON pointer / JSONL record / CSV cell / matrix element, and acceptance conditions;
   - reject generic artifact citations that do not support a requested claim.
4. Add immutable scientific-state snapshots and renderer output for agent-visible state.
5. Add deterministic dispute adjudication for incompatible agent claims.
6. Add capability registry and tool-gap generation:
   - missing scientific capability creates a `ToolGap`;
   - tool gaps create `BUILD_MISSING_TOOL` actions;
   - unvalidated tools cannot execute or register;
   - validated tools require a `ToolValidationCertificate`.
7. Add a bounded deterministic V3 demo:
   - compile the finite-lattice representation-dependence problem;
   - create proof obligations;
   - detect at least one initial missing capability;
   - register deterministic validated capabilities;
   - run a bounded counterexample fixture;
   - construct certificates;
   - independently verify them;
   - update scientific state through proof-carrying transitions only;
   - write V3 artifacts and a run summary.
8. Add CLI and frontend adapters for the focused V3 panels.
9. Add regression tests covering the observed V2.8 epistemic-state corruption cases.

## Non-Scope For This Checkpoint

- No live model calls.
- No live network calls.
- No arbitrary code-generation sandbox that writes executable tools into the repository.
- No claim that the bounded finite fixture proves a general theorem.
- No UI redesign.
- No replacement of V1-V2.8 workflows.

## Acceptance

The checkpoint is accepted only if:

- the baseline V2.8 demo remains valid;
- the new V3 deterministic demo validates;
- tests demonstrate unsupported completion claims are rejected;
- tests demonstrate exact artifact support is required;
- tests demonstrate tool gaps and validated-tool registration behavior;
- tests demonstrate stale snapshots and curator prose cannot override scientific state;
- all existing tests continue to pass offline.
