# Claim DAG Database

The Claim DAG database turns a run directory into a queryable SQLite view of load-bearing scientific claims, dependencies, checks, contradictions, independent-check records, blockers, and deterministic total-gate results.

Build from an existing run:

```bash
PYTHONPATH=src python -m coscientist.cli build-claim-dag-db runs/v22-fixture --force
```

Validate:

```bash
PYTHONPATH=src python -m coscientist.cli validate-claim-dag-db runs/v22-fixture
```

Query:

```bash
PYTHONPATH=src python -m coscientist.cli query-claim-dag-db runs/v22-fixture claim_nodes --limit 20
PYTHONPATH=src python -m coscientist.cli query-claim-dag-db runs/v22-fixture total_gate_results --limit 5
```

The database file is:

```text
claim_dag.sqlite
```

Main tables:

- `claim_nodes`
- `claim_edges`
- `claim_checks`
- `claim_contradictions`
- `independent_checks`
- `validation_blockers`
- `total_gate_results`
- `load_bearing_paths`

The total gate is deterministic. LLM messages, rankings, and high scores cannot directly mark a candidate as validated. `internally_validated` means only that encoded internal software checks passed; it is not external scientific establishment.
