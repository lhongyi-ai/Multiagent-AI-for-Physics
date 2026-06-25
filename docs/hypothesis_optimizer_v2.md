# Hypothesis Optimizer V2

Optimizer V2 is a deterministic checkpoint optimizer.

Implemented now:

- hard gates
- cheap kill tests
- score vectors with provenance
- Pareto frontier
- portfolio roles
- mutation operator registry
- counterexample tasks
- expected-information-gain queue
- failure memory

It is not a full autonomous scientific optimizer yet. It is designed to give agents concrete next actions and block vague, unfalsifiable branches before expensive calls.
