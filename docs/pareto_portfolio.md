# Pareto Portfolio

The Pareto portfolio keeps multiple useful hypothesis roles instead of selecting one scalar winner.

Current roles:

- `pareto_candidate`
- `reserve`
- `contrarian_guardrail`

The checkpoint implementation uses deterministic score dimensions and preserves at least one contrarian branch when possible.
