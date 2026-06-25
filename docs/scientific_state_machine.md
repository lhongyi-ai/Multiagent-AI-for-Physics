# Scientific State Machine

V3.0 introduces a deterministic scientific-state machine for verified Claim DAG changes.

## Rule

Agents can propose scientific claims and actions, but they cannot directly transition a claim to `SUPPORTED`, `CONTRADICTED`, or `INCONCLUSIVE`.

A verified transition requires:

- a known claim ID;
- the current snapshot version;
- a completed action execution ID;
- at least one verified certificate;
- exact evidence locators;
- passing deterministic acceptance conditions;
- independent verification for verified scientific-state transitions.

Rejected transitions are persisted as audit records rather than hidden.

## Implemented Statuses

Claim statuses:

`UNTESTED`, `PROPOSED`, `PLANNED`, `EXECUTING`, `CANDIDATE_RESULT`, `SUPPORTED`, `CONTRADICTED`, `INCONCLUSIVE`, `BLOCKED`, `RETRACTED`, `SUPERSEDED`

Proof-obligation statuses:

`UNSTARTED`, `PLANNED`, `EXECUTING`, `CANDIDATE_FOUND`, `VERIFIED`, `FAILED`, `BLOCKED`, `INCONCLUSIVE`, `SUPERSEDED`

## Snapshot Binding

Every proof-carrying update declares `based_on_snapshot_version`. If the update is stale, it is rejected.

The current deterministic snapshot records hashes for:

- claim state;
- obligation graph;
- artifact manifest;
- candidate archive version;
- failure memory version.

## Current Scope

The implementation is in `src/coscientist/core/proof_search.py`. The V3 demo is a bounded deterministic pilot, not a complete autonomous theorem prover.
