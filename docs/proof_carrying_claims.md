# Proof-Carrying Claims

V3.0 separates prose from verified scientific state.

## Agent Output

Agent output may include:

- proposed claims;
- claimed completed actions;
- artifact citations;
- suggested repairs or tools.

Unsupported completion language is labeled `UNSUPPORTED_COMPLETION_CLAIM` when no matching completed action execution exists.

## Proof-Carrying Transition

A `ProofCarryingClaimUpdate` is the only implemented path to verified claim-state changes. It carries:

- actor type;
- snapshot version;
- action execution ID;
- certificate IDs;
- evidence locators;
- independent verification IDs;
- requested status.

`ScientificStateMachine.apply_update()` creates a `ClaimTransitionRecord` with either `accepted` or `rejected`.

## V3 Demo Behavior

The deterministic demo intentionally writes both:

- a rejected agent attempt to mark a claim as `SUPPORTED`;
- an accepted executor transition marking the uniqueness claim as `CONTRADICTED`.

The accepted transition is backed by exact evidence locators into `v3_counterexample_tool/partition_comparison.json` and `v3_counterexample_tool/counterexample_certificate.json`.
