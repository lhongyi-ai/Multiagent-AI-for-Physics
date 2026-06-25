# Closed-Loop Scientific Action Execution

V2.8 turns Live Agent Meeting Room recommendations into executable scientific actions.

The loop is:

```text
Optimizer queue
→ policy and permission checks
→ registered deterministic tool
→ action artifact bundle
→ verifier rows
→ Claim DAG update
→ optimizer queue refresh
→ next meeting round context
→ progress-aware stopping
```

## Action Lifecycle

Each action is normalized into `ScientificAction` with:

- `action_id`
- `hypothesis_id`
- `domain_id`
- `action_type`
- `tool_id`
- structured `arguments`
- expected information gain, success probability, feasibility, cost, and priority
- required permissions
- target claims and blockers
- execution status, retry count, provenance, timestamps, and idempotency key

Supported statuses are:

- `queued`
- `selected`
- `policy_blocked`
- `permission_blocked`
- `running`
- `succeeded`
- `failed`
- `inconclusive`
- `skipped_duplicate`
- `not_executable`

Only one eligible action is selected per productive meeting round. Duplicate deterministic tool actions are suppressed unless the action has materially different arguments or is a targeted repair.

## Tool Registry

The default registry reuses existing backend tools:

- `phase1_minimal_mixed_bcs_solver`
- `phase2_data_coverage_tool`
- `phase2_lsco_acquisition`
- `energy_decomposition_audit_tool`
- `representation_counterexample`
- `hellmann_feynman_diagnostic`
- `targeted_repair_task`

The registry is backend-only. The Gradio frontend displays action state but does not duplicate scientific logic.

## Permission Behavior

Default execution is offline and deterministic.

Live-network and live-model actions are blocked unless the corresponding explicit permission is enabled. The closed-loop executor does not silently fall back to mock output when a live action fails or lacks permission.

## Artifact Bundle

Each selected action writes:

- `action_request.json`
- `policy_decision.json`
- `tool_invocation.json`
- `execution_result.json`
- `generated_artifacts.json`
- `verifier_results.jsonl`
- `claim_dag_diff.json`
- `optimizer_diff.json`
- `execution_summary.json`

Bundles live under:

```text
<run_dir>/action_executions/<action_id>/
```

The meeting run also writes:

- `closed_loop_action_state.json`
- `closed_loop_action_executions.jsonl`

## Claim DAG Updates

After execution, the action result is attached to the Claim DAG. The current implementation can:

- create seed Claim DAG artifacts when absent;
- append action-backed claim checks;
- add representation or data-readiness blockers;
- add contradictions for invalid physical inferences;
- invalidate downstream claim labels;
- rebuild the local `claim_dag.sqlite` index.

For superconductivity, the executor preserves the distinction between:

- toy-model feasibility;
- numerical component ledgers under a fixed convention;
- observable electromagnetic response;
- LSCO material-level support;
- universal mechanism claims.

Lower-level evidence does not automatically validate higher-level claims.

## Scientific Guardrails

The condensation-energy audit currently returns:

```text
counterexample_found
counterexample_demonstrating_non_unique_component_partition
```

Therefore V2.8 keeps these hard gates:

- a closed numerical energy ledger does not prove unique physical mechanism percentages;
- component energy changes remain model and representation dependent unless an observable mapping is established;
- finite toy calculations are not general theorems;
- Tier A/B LSCO readiness is not Tier C quantitative separation;
- total condensation energy and full gauge-coupled response are preferred for physical conclusions.

## Frontend

The Live Agent Room now includes:

- closed-loop action state;
- closed-loop action execution table;
- generated artifact references;
- verifier counts;
- Claim DAG checks and blockers;
- optimizer queue updates.

The transcript includes action execution events before the agent round, so agents receive completed state rather than repeating future-work recommendations.

## Deterministic Demo

```bash
PYTHONPATH=src python -m coscientist.cli run-closed-loop-demo \
  --runs-dir /tmp/coscientist-v28 \
  --run-id closed-loop-demo \
  --force
```

Expected result:

- one optimizer action executes per productive round;
- the energy-decomposition audit generates a counterexample artifact bundle;
- Claim DAG blockers prevent unique mechanism-percentage claims;
- optimizer queue adds a targeted repair action;
- the meeting stops when no eligible action remains;
- no live model or live-network call occurs.

## Limitations

This layer executes registered local tools only. It does not perform unrestricted web search, autonomous lab control, or database mutation outside the existing reviewed data-promotion workflows.
